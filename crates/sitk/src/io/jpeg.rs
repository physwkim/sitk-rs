//! JPEG (`.jpg`/`.jpeg`) reader and writer — `itk::JPEGImageIO`, on the
//! pure-Rust `jpeg-decoder` (read) and `jpeg-encoder` (write) crates (ledger
//! §5.8(a)/§5.26: `cargo tree -p sitk-io` shows no `*-sys` crate — both are
//! `#[deny(unsafe_code)]`/`#[forbid(unsafe_code)]` outside an unused `simd`
//! feature). See ledger §5.26 for the full crate-choice rationale.
//!
//! # Quality is `CompressionLevel`, and `UseCompression` is dead
//!
//! `SetQuality`/`GetQuality` are a thin alias over `SetCompressionLevel`/
//! `GetCompressionLevel` (itkJPEGImageIO.h:56-67) — there is no separate
//! `m_Quality` field. `itkSetClampMacro(CompressionLevel, int, 1,
//! GetMaximumCompressionLevel())` (itkImageIOBase.h:288) clamps it, and
//! unlike every other format ported so far, `JPEGImageIO`'s constructor never
//! calls `SetMaximumCompressionLevel` — so the ceiling stays at
//! `ImageIOBase`'s own default of **100** (itkImageIOBase.h:830), not the `9`
//! that bounds MetaImage/NRRD/PNG (see [`crate::io::compression`]). The
//! constructor sets the default quality to 95 (itkJPEGImageIO.cxx:300).
//! Ledger §3.49.
//!
//! Because that ITK alias is a *single* field, this port has nothing to
//! disambiguate: it exposes one knob, [`WriteOptions::compression_level`],
//! which `resolved_quality` interprets as the JPEG quality. There is no
//! separate `Quality` option, so the alias collapses to one field with one
//! meaning — verified 2026-07-11, no code change (ledger §3.49).
//!
//! `m_UseCompression = false` is set in the same constructor (`:299`) and
//! never referenced again anywhere in `itkJPEGImageIO.cxx` — dead code.
//! `Write` calls `jpeg_set_quality` unconditionally; there is no branch that
//! skips it. [`write()`] reproduces this: `resolved_quality` never consults
//! [`WriteOptions::use_compression`]. Ledger §2.138.
//!
//! # `Progressive` and `CMYKtoRGB` are always on upstream — configurable here
//!
//! Both are in-class initializers on `JPEGImageIO` itself — `m_Progressive{
//! true }`, `m_CMYKtoRGB{ true }` (itkJPEGImageIO.h:125,127) — and the
//! constructor never touches them, unlike `m_UseCompression` a few lines
//! above. SimpleITK's `ImageFileWriter`/`ImageFileReader` expose only the
//! generic `SetUseCompression`/`SetCompressionLevel`/`SetCompressor`; neither
//! `SetProgressive` nor `SetCMYKtoRGB` is wrapped anywhere in
//! `Code/IO/include/sitkImageFileWriter.h` or `sitkImageSeriesWriter.h`.
//! Through *SimpleITK's* public surface a JPEG is therefore always written
//! progressive and a CMYK source is always converted to RGB on read (ledger
//! §3.50; the chroma-subsampling factor is likewise fixed at 4:2:0, §5.27).
//!
//! This port keeps that as the default but exposes the capability upstream
//! hides:
//!
//! * **Write** — the [`ImageIo`] trait and [`write()`] use
//!   [`JpegWriteOptions::default`] (progressive on, 4:2:0), byte-identical to
//!   before; [`write_with_jpeg_options`] takes an explicit [`JpegWriteOptions`]
//!   to pick the progressive flag and one of 4:4:4 / 4:2:2 / 4:2:0 chroma
//!   subsampling.
//! * **Read** — [`read`] flattens CMYK to RGB, upstream's only path;
//!   [`read_preserving_cmyk`] keeps the raw four uninverted channels as a
//!   4-component vector image. `jpeg_decoder` hands back
//!   [`PixelFormat::CMYK32`] itself, so this is a genuine capability, not a
//!   decoder work-around.
//!
//! These live on the `jpeg` module's own surface because this crate's shared
//! [`WriteOptions`] carries no format-specific fields and `read` takes no
//! options at all; a future shared `ReadOptions`/`WriteOptions` unification
//! could lift progressive, chroma subsampling and CMYK-preservation into the
//! generic writer/reader surface.
//!
//! # CMYK → RGB: fixed in this port (upstream is right for plain CMYK, wrong for YCCK)
//!
//! Upstream's `Read` CMYK branch treats all four raw channels from libjpeg as
//! already *inverted* (`stored = 255 - true`) and recovers RGB as `C·K/255`
//! (and likewise for M, Y), discarding K — "the Gimp approach," per upstream's
//! own comment (itkJPEGImageIO.cxx:244-251). That formula is correct only
//! when the source really is plain CMYK (no Adobe colour transform):
//! libjpeg-turbo's `null_convert` leaves such data untouched, still inverted,
//! all four channels (jdcolor.c). A **YCCK**-encoded CMYK JPEG — the common
//! case for Photoshop exports — takes a different path: `ycck_cmyk_convert`
//! (jdcolor.c:544-583) recovers *un-inverted* C/M/Y from the YCbCr-style
//! transform while leaving K's raw byte untouched (`:574-578`; §1.65 works
//! the algebra through libjpeg-turbo's matching `cmyk_ycck_convert` encoder
//! to show the decode result is un-inverted C/M/Y against inverted K).
//! Upstream's read formula does not know which case it is in and applies the
//! same "everything is inverted" arithmetic regardless — silently wrong RGB
//! for a YCCK source. Filed as B77 of #6575.
//!
//! `jpeg_decoder::Decoder` does not have this problem: it reads the JPEG's
//! own Adobe APP14 colour-transform marker itself (`AdobeColorTransform`,
//! `decoder.rs`) and dispatches to a plain-CMYK or YCCK deconversion
//! *before* [`read`] ever sees a byte, so [`PixelFormat::CMYK32`] is always
//! the same *uninverted* (`true`) CMYK convention regardless of which of the
//! two encodings produced the file — verified by round-tripping both
//! `jpeg_encoder::ColorType::Cmyk` and `::CmykAsYcck` through
//! `jpeg_decoder::Decoder` and confirming they recover the same original
//! bytes (this crate's test module, `cmyk_and_ycck_sources_of_the_same_color_read_back_to_the_same_rgb`).
//! `cmyk_to_rgb` restores the inversion ITK's formula
//! expects — `invC = 255 - C`, etc. — before applying the same `invC·invK/255`
//! arithmetic, which is then correct uniformly for both source encodings.
//! Ledger §1.65, §4.93.
//!
//! # Spacing: JFIF density, hand-parsed
//!
//! `ReadImageInformation` derives spacing from libjpeg's `cinfo.density_unit`/
//! `X_density`/`Y_density` (itkJPEGImageIO.cxx:420-433), which libjpeg
//! populates from the JFIF `APP0` marker. `jpeg-decoder` parses that marker
//! only far enough to set a boolean (`is_jfif`) — the density fields
//! themselves are discarded (`crate::io::parser::AppData::Jfif` is a unit
//! variant). `scan_jfif_density` walks the marker stream by hand to recover
//! them, and `spacing_from_density` reproduces
//! `ReadImageInformation`'s exact unit-1-is-inches/unit-2-is-centimetres
//! arithmetic, `unit == 0` (or no marker at all) included default `[1.0,
//! 1.0]`.
//!
//! # The density *write* bug: the "prefer centimetres" branch tags `unit = 0`
//!
//! `WriteSlice` computes both an inches-based and a centimetres-based density
//! encoding, then picks whichever round-trips with less error
//! (itkJPEGImageIO.cxx:543-567). The inches branch is correct
//! (`cinfo.density_unit = 1`, `:558`); the centimetres branch sets
//! `cinfo.density_unit = 0` (`:564`) — JFIF's "no units, aspect ratio only"
//! tag — instead of `2` ("dots per cm"). `ReadImageInformation`'s own
//! `density_unit == 2` branch therefore never fires for a file this bug wrote
//! at whatever spacing favoured the cm encoding, and the read side falls
//! through to the default `[1.0, 1.0]`. Common spacings like `1.0` mask this
//! (the fallback default happens to be the right answer), but e.g. `2.0`
//! exposes it cleanly and `0.75`/`2.54` exercise the (unaffected) inches
//! branch. Ledger §1.64.
//!
//! `density_for_spacing` reproduces the *effect* — a cm-favoured spacing
//! round-trips to `[1.0, 1.0]`, not the value written — via
//! [`jpeg_encoder::Density::None`] rather than byte-exact `unit = 0` output:
//! `Density`'s own JFIF writer hardwires `X = Y = 1` under `Density::None`
//! (`jpeg_encoder::writer::JfifWriter::write_header`), where upstream's buggy
//! branch still writes its real (if unusable) computed `X_density`/
//! `Y_density`. `spacing_from_density` does not distinguish the two byte
//! patterns — both fail the `unit == 2` guard identically — so the observable
//! round-trip behaviour matches exactly. Ledger §4.95.
//!
//! # `Write` rejects non-2-D outright; PNG truncates silently
//!
//! `Write` throws immediately when `GetNumberOfDimensions() != 2`
//! (itkJPEGImageIO.cxx:459-463) — no slice of a 3-D image is ever written,
//! unlike `PNGImageIO::WriteSlice`, which silently takes only the first
//! Z-slice (ledger §2.125). [`write()`] reproduces the throw as
//! [`IoError::JpegWriteRejected`]. Ledger §2.135.
//!
//! # `CanReadFile` is stricter than PNG's
//!
//! PNG's `CanReadFile` checks only the 8-byte signature (ledger, see
//! [`crate::io::png`]). JPEG's checks three things in sequence
//! (itkJPEGImageIO.cxx:90-158): the extension (`HasSupportedReadExtension`,
//! case-**sensitive** — `:102`), the 2-byte `0xFFD8` magic (`:117-130`), and
//! then a full `jpeg_read_header` parse (`:141-155`) — a malformed JPEG with
//! a correct extension and magic number is still rejected.
//! [`JpegImageIo::can_read_file`] reproduces all three gates: extension, then
//! magic, then `jpeg_decoder::Decoder::read_info()`. Ledger §2.136.
//!
//! `CanWriteFile` is `HasSupportedWriteExtension(name, false)`
//! (`:439-450`) — case-sensitive like `CanReadFile`'s extension gate, and like
//! PNG's `CanWriteFile` (ledger §2.124), but with four registered spellings —
//! `.jpg`/`.JPG`/`.jpeg`/`.JPEG` (`:308-314`) — rather than PNG's two, so a
//! mixed-case spelling such as `.Jpg` still fails. Ledger §2.137.
//!
//! # Not implemented
//!
//! * **Byte-exact density-bug reproduction.** See the "density *write* bug"
//!   section above; ledger §4.95.
//! * **A mid-scanline decode error returns a partially filled buffer.**
//!   `Read`'s scanline loop catches a libjpeg longjmp with `itkWarningMacro`
//!   and a silent `return` (itkJPEGImageIO.cxx:226-235, 265-273), leaving
//!   whatever tail of the caller's buffer had not yet been decoded
//!   uninitialised while still reporting success — the same shape as the
//!   already-ledgered `MET_PerformUncompression` quirk (§4.75,
//!   [`crate::io::compression`]). Not expressible in safe Rust: [`read`] returns
//!   [`IoError::JpegDecode`] for a failure at any point, header or mid-scan.
//!   Ledger §4.92.
//! * **A 2- or 5-plus-component JPEG.** `ReadImageInformation`'s component
//!   switch has a `default:` arm that accepts *any* component count as an
//!   `IOPixelEnum::VECTOR` with a warning (itkJPEGImageIO.cxx:412-417).
//!   `jpeg_decoder::Decoder` refuses to parse such a frame at all —
//!   `Err(Unsupported(ComponentCount(n)))` before a `FrameInfo` is ever
//!   stored (jpeg-decoder's `decoder.rs:368-372`) — so this port cannot read
//!   a file upstream (oddly) can. Ledger §4.96.
//! * **A JPEG write with a component count other than 1 or 3.** `WriteSlice`'s
//!   `in_color_space` switch has arms only for 1 (`JCS_GRAYSCALE`) and 3
//!   (`JCS_RGB`); anything else falls to `default: JCS_UNKNOWN` plus a warning
//!   (itkJPEGImageIO.cxx:521-533) — an encoding with no defined colour
//!   transform that `jpeg_encoder::ColorType` has no counterpart for (its
//!   variants are all named, fixed-arity colour spaces). [`write()`] refuses
//!   any component count other than 1 or 3 with
//!   [`IoError::UnsupportedJpegFeature`]. Ledger §4.94.
//! * **A >8-bit-precision (lossless) JPEG.** `m_ComponentType` is hard-coded
//!   `UCHAR` at compile time — `#if BITS_IN_JSAMPLE == 8` / `#error` otherwise
//!   (itkJPEGImageIO.cxx:292-297) — so no mainline ITK build reads one either.
//!   `header_from_info` refuses `PixelFormat::L16` with
//!   [`IoError::UnsupportedJpegFeature`]. Ledger §4.97.

use std::io::Cursor;
use std::path::Path;

use crate::core::{Image, PixelBuffer, PixelId};
use jpeg_decoder::{Decoder as JpegDecoder, PixelFormat};
use jpeg_encoder::{ColorType, Density, Encoder as JpegEncoder, SamplingFactor};

use crate::io::compression::MIN_COMPRESSION_LEVEL;
use crate::io::error::{IoError, Result};
use crate::io::image_io::{ImageInformation, ImageIo, has_supported_extension};
use crate::io::writer::WriteOptions;

/// `ImageIOBase`'s own default `MaximumCompressionLevel`, never lowered by
/// `JPEGImageIO`'s constructor (itkImageIOBase.h:830) — see the module doc.
const JPEG_MAX_QUALITY: i32 = 100;

/// `this->Self::SetQuality(95)` (itkJPEGImageIO.cxx:300).
const JPEG_DEFAULT_QUALITY: i32 = 95;

/// `JPEG_MAX_DIMENSION` (jmorecfg.h:158) — "a tad under 64K to prevent
/// overflows." `WriteSlice` rejects any image wider or taller than this
/// (itkJPEGImageIO.cxx:488-491).
const JPEG_MAX_DIMENSION: usize = 65500;

/// `MAX_COMPONENTS` (jmorecfg.h:30). `WriteSlice` rejects any image with more
/// components than this (itkJPEGImageIO.cxx:492-496) — in practice this
/// crate's own {1, 3} restriction (see module doc, ledger §4.94) is always
/// hit first for a component count in `4..=10`, but a count above 10 hits
/// this check, and only this one, exactly as upstream does.
const MAX_COMPONENTS: usize = 10;

/// The chroma-subsampling scheme applied to a 3-component (YCbCr) JPEG write.
///
/// SimpleITK exposes no way to choose this (ledger §3.50); `JPEGImageIO`
/// always writes 4:2:0 via libjpeg's constant `jpeg_default_colorspace`
/// (§5.27). This port keeps 4:2:0 as the default but lets a caller pick
/// through [`JpegWriteOptions`] / [`write_with_jpeg_options`]. It has no
/// effect on a 1-component grayscale write, which carries no chroma.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JpegChromaSubsampling {
    /// 4:4:4 — no chroma subsampling (`jpeg_encoder::SamplingFactor::F_1_1`).
    None,
    /// 4:2:2 — chroma halved horizontally (`SamplingFactor::F_2_1`).
    Chroma422,
    /// 4:2:0 — chroma halved on both axes (`SamplingFactor::F_2_2`). The
    /// default, byte-identical to upstream's constant behaviour.
    #[default]
    Chroma420,
}

impl JpegChromaSubsampling {
    /// The `jpeg_encoder` factor this scheme maps to.
    fn sampling_factor(self) -> SamplingFactor {
        match self {
            JpegChromaSubsampling::None => SamplingFactor::F_1_1,
            JpegChromaSubsampling::Chroma422 => SamplingFactor::F_2_1,
            JpegChromaSubsampling::Chroma420 => SamplingFactor::F_2_2,
        }
    }
}

/// The two JPEG encode knobs `JPEGImageIO` hard-wires and SimpleITK cannot
/// reach — `m_Progressive` and the chroma-subsampling factor (ledger §3.50,
/// §5.27). [`write()`] and the [`ImageIo`] trait use [`JpegWriteOptions::default`],
/// whose values reproduce upstream exactly (progressive on, 4:2:0);
/// [`write_with_jpeg_options`] takes an explicit set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegWriteOptions {
    /// `m_Progressive` — upstream's in-class `true` (itkJPEGImageIO.h:125).
    pub progressive: bool,
    /// The chroma-subsampling scheme for a 3-component write.
    pub chroma_subsampling: JpegChromaSubsampling,
}

impl Default for JpegWriteOptions {
    /// Byte-identical to today's fixed behaviour: progressive, 4:2:0.
    fn default() -> Self {
        Self {
            progressive: true,
            chroma_subsampling: JpegChromaSubsampling::Chroma420,
        }
    }
}

/// `GetQuality`/`SetQuality`'s clamp, `1..=100` (see module doc, ledger
/// §3.49) — distinct from every other format's `1..=9`
/// ([`crate::io::compression::MAX_COMPRESSION_LEVEL`]), so this crate cannot
/// reuse [`WriteOptions::resolved_level`].
fn resolved_quality(options: &WriteOptions) -> u8 {
    let quality = if options.compression_level < 0 {
        JPEG_DEFAULT_QUALITY
    } else {
        options
            .compression_level
            .clamp(MIN_COMPRESSION_LEVEL, JPEG_MAX_QUALITY)
    };
    quality as u8
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// The JFIF `APP0` marker's density fields — `cinfo.density_unit`/
/// `X_density`/`Y_density` after `jpeg_read_header`, hand-recovered because
/// `jpeg-decoder` discards them (see module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct JfifDensity {
    unit: u8,
    x: u16,
    y: u16,
}

/// Walk the marker stream from `SOI` looking for a JFIF `APP0` segment,
/// stopping at the first marker that cannot carry one — `SOS` (entropy-coded
/// data follows, which is not marker-structured) or `EOI`. Bounds-checked and
/// best-effort: a truncated or malformed stream yields `None` rather than
/// panicking.
///
/// Unlike the JFIF spec's "must be the first marker after `SOI`" rule, this
/// does not require `APP0` to be literally first — matching libjpeg-turbo's
/// own marker reader, which recognises an `APP0` "JFIF\0" segment by content
/// wherever it appears before `SOS` (`jdmarker.c`'s `get_interesting_appn`).
fn scan_jfif_density(bytes: &[u8]) -> Option<JfifDensity> {
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut pos = 2;
    loop {
        // A marker is 0xFF, optional 0xFF fill bytes, then a non-0xFF code.
        while bytes.get(pos) == Some(&0xFF) {
            pos += 1;
        }
        let marker = *bytes.get(pos)?;
        pos += 1;

        // TEM (0x01) and the restart markers (0xD0-0xD7) carry no length
        // field. Neither should appear before SOS in a well-formed file, but
        // treating them as length-less here — rather than misreading the
        // next two bytes as a bogus length — keeps a malformed file from
        // desyncing the whole walk.
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        // EOI, or SOS: entropy-coded scan data follows, not more markers.
        if marker == 0xD9 || marker == 0xDA {
            return None;
        }

        let length = u16::from_be_bytes([*bytes.get(pos)?, *bytes.get(pos + 1)?]) as usize;
        if length < 2 {
            return None;
        }
        let payload = bytes.get(pos + 2..pos + length)?;

        if marker == 0xE0 && payload.len() >= 12 && payload[0..5] == *b"JFIF\0" {
            return Some(JfifDensity {
                unit: payload[7],
                x: u16::from_be_bytes([payload[8], payload[9]]),
                y: u16::from_be_bytes([payload[10], payload[11]]),
            });
        }

        pos += length;
    }
}

/// `ReadImageInformation`'s density-to-spacing arithmetic
/// (itkJPEGImageIO.cxx:421-433) exactly: `unit == 1` is inches, `unit == 2`
/// is centimetres, anything else — including no marker at all — is the
/// `[1.0, 1.0]` default set moments earlier in the same function (`:332-333`).
fn spacing_from_density(density: Option<JfifDensity>) -> [f64; 2] {
    match density {
        Some(JfifDensity { unit: 1, x, y }) if x > 0 && y > 0 => {
            [25.4 / f64::from(x), 25.4 / f64::from(y)]
        }
        Some(JfifDensity { unit: 2, x, y }) if x > 0 && y > 0 => {
            [10.0 / f64::from(x), 10.0 / f64::from(y)]
        }
        _ => [1.0, 1.0],
    }
}

/// What [`read_information`] and [`read`] keep from the decoded header.
struct Header {
    width: usize,
    height: usize,
    pixel_id: PixelId,
    number_of_components: usize,
    is_cmyk: bool,
}

/// `ReadImageInformation`'s `cinfo.output_components`/`out_color_space`
/// switch (itkJPEGImageIO.cxx:386-418), translated from `jpeg_decoder`'s
/// [`PixelFormat`]. `CMYKtoRGB` is always on (module doc, ledger §3.50), so a
/// 4-component source is always reported as 3-component RGB, matching
/// upstream's `case 4: ... if (m_CMYKtoRGB) { RGB, 3 }` branch
/// (`:400-404`) — the `VECTOR`/4-component `else` (`:406-409`) is unreachable
/// through this crate, exactly as it is unreachable through SimpleITK.
fn header_from_info(info: jpeg_decoder::ImageInfo) -> Result<Header> {
    let (pixel_id, number_of_components, is_cmyk) = match info.pixel_format {
        PixelFormat::L8 => (PixelId::UInt8, 1, false),
        PixelFormat::L16 => {
            return Err(IoError::UnsupportedJpegFeature(
                "a >8-bit-precision (lossless) JPEG: JPEGImageIO hard-codes UCHAR at \
                 compile time (itkJPEGImageIO.cxx:292-297) — doc/upstream-findings.md §4.97"
                    .to_string(),
            ));
        }
        PixelFormat::RGB24 => (PixelId::VectorUInt8, 3, false),
        PixelFormat::CMYK32 => (PixelId::VectorUInt8, 3, true),
    };
    Ok(Header {
        width: info.width as usize,
        height: info.height as usize,
        pixel_id,
        number_of_components,
        is_cmyk,
    })
}

/// Read the header only, reproducing `jpeg_read_header` +
/// `jpeg_calc_output_dimensions` (itkJPEGImageIO.cxx:361-370) without
/// decoding any scanlines.
fn open_and_probe(bytes: &[u8]) -> Result<(JpegDecoder<Cursor<&[u8]>>, Header)> {
    let mut decoder = JpegDecoder::new(Cursor::new(bytes));
    decoder.read_info()?;
    let info = decoder
        .info()
        .expect("read_info() returned Ok, so a frame is set");
    let header = header_from_info(info)?;
    Ok((decoder, header))
}

/// Read the header only, with no pixel data.
///
/// The meta-data dictionary is always empty — `JPEGImageIO` never writes to
/// `m_MetaDataDictionary`.
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    let bytes = std::fs::read(path)?;
    let (_, header) = open_and_probe(&bytes)?;
    let spacing = spacing_from_density(scan_jfif_density(&bytes));

    Ok(ImageInformation {
        pixel_id: header.pixel_id,
        dimension: 2,
        number_of_components: header.number_of_components,
        size: vec![header.width, header.height],
        spacing: spacing.to_vec(),
        origin: vec![0.0, 0.0],
        direction: identity(2),
        metadata: std::collections::BTreeMap::new(),
    })
}

/// `Read`'s CMYK branch (itkJPEGImageIO.cxx:220-262): recover R, G, B as
/// `invC · invK / 255` (and likewise for M, Y), discarding K — the "Gimp
/// approach," applied to `jpeg-decoder`'s already-normalized
/// [`PixelFormat::CMYK32`] output. See the module doc's "CMYK → RGB" section
/// (ledger §1.65) for why inverting first, rather than multiplying the raw
/// channels directly as upstream does, is what makes this formula correct
/// for both a plain-CMYK and a YCCK source.
fn cmyk_to_rgb(cmyk: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(cmyk.len() / 4 * 3);
    for px in cmyk.chunks_exact(4) {
        let inv_k = 255.0 - f32::from(px[3]);
        let inv_c = 255.0 - f32::from(px[0]);
        let inv_m = 255.0 - f32::from(px[1]);
        let inv_y = 255.0 - f32::from(px[2]);
        rgb.push((inv_c * inv_k / 255.0) as u8);
        rgb.push((inv_m * inv_k / 255.0) as u8);
        rgb.push((inv_y * inv_k / 255.0) as u8);
    }
    rgb
}

/// Read a `.jpg` file, converting a CMYK source to RGB — the only behaviour
/// SimpleITK can reach, since `m_CMYKtoRGB` is hard-wired on (ledger §3.50).
pub fn read(path: &Path) -> Result<Image> {
    read_impl(path, false)
}

/// Read a `.jpg` file, keeping a CMYK source as its raw four uninverted
/// channels instead of converting to RGB.
///
/// `JPEGImageIO::m_CMYKtoRGB` is an in-class `true` the constructor never
/// touches, and SimpleITK wraps no `SetCMYKtoRGB`, so a CMYK JPEG is always
/// flattened to RGB through the public surface (ledger §3.50). That is not a
/// decoder limitation: `jpeg_decoder` reads the Adobe APP14 marker and hands
/// back [`PixelFormat::CMYK32`] — four channels in the *uninverted* (`true`)
/// CMYK convention regardless of a plain-CMYK or YCCK encoding (module doc's
/// "CMYK → RGB" section). This crate-only reader exposes that: a CMYK source
/// reads back as a four-component [`PixelId::VectorUInt8`] carrying the raw
/// channels, and a non-CMYK source is identical to [`read`].
pub fn read_preserving_cmyk(path: &Path) -> Result<Image> {
    read_impl(path, true)
}

/// The body of [`read`] / [`read_preserving_cmyk`]: decode once, then either
/// flatten a CMYK source to RGB (`preserve_cmyk == false`, upstream's only
/// path) or keep its raw four channels (`preserve_cmyk == true`).
fn read_impl(path: &Path, preserve_cmyk: bool) -> Result<Image> {
    let bytes = std::fs::read(path)?;
    let (mut decoder, header) = open_and_probe(&bytes)?;
    let raw = decoder.decode()?;

    let size = vec![header.width, header.height];
    let spacing = spacing_from_density(scan_jfif_density(&bytes)).to_vec();
    let origin = vec![0.0, 0.0];
    let direction = identity(2);

    if header.is_cmyk && preserve_cmyk {
        // The raw four uninverted CMYK channels, kept as a 4-component vector
        // image rather than flattened to RGB.
        return Ok(Image::from_parts_vector(
            PixelBuffer::UInt8(raw),
            4,
            size,
            spacing,
            origin,
            direction,
        )?);
    }

    let pixels = if header.is_cmyk {
        cmyk_to_rgb(&raw)
    } else {
        raw
    };
    let buffer = PixelBuffer::UInt8(pixels);

    Ok(if header.pixel_id.is_vector() {
        Image::from_parts_vector(
            buffer,
            header.number_of_components,
            size,
            spacing,
            origin,
            direction,
        )?
    } else {
        Image::from_parts(buffer, size, spacing, origin, direction)?
    })
}

/// `WriteSlice`'s density-selection arithmetic (itkJPEGImageIO.cxx:543-567):
/// pick whichever of an inches or a centimetres encoding round-trips with
/// less error. See the module doc's density-write-bug section for why the
/// centimetres branch returns [`Density::None`] rather than a centimetre
/// value.
fn density_for_spacing(spacing_x: f64, spacing_y: f64) -> Density {
    let inch = [
        (25.4 / spacing_x + 0.5) as u16,
        (25.4 / spacing_y + 0.5) as u16,
    ];
    let cm = [
        (10.0 / spacing_x + 0.5) as u16,
        (10.0 / spacing_y + 0.5) as u16,
    ];
    let inch_error = (25.4 / spacing_x - f64::from(inch[0])).abs()
        + (25.4 / spacing_y - f64::from(inch[1])).abs();
    let cm_error =
        (10.0 / spacing_x - f64::from(cm[0])).abs() + (10.0 / spacing_y - f64::from(cm[1])).abs();

    if inch_error <= cm_error {
        Density::Inch {
            x: inch[0],
            y: inch[1],
        }
    } else {
        Density::None
    }
}

/// Write a `.jpg` file with the fixed encode settings SimpleITK reaches:
/// progressive, 4:2:0 chroma subsampling ([`JpegWriteOptions::default`]).
///
/// `WriteImageInformation` is a no-op upstream; the header and pixel data are
/// both emitted by `Write`/`WriteSlice` here in one pass, as they are
/// upstream.
pub fn write(image: &Image, path: &Path, options: &WriteOptions) -> Result<()> {
    write_with_jpeg_options(image, path, options, &JpegWriteOptions::default())
}

/// Write a `.jpg` file, choosing the progressive flag and chroma-subsampling
/// factor SimpleITK hard-wires (ledger §3.50, §5.27). [`write()`] is this with
/// [`JpegWriteOptions::default`], which is byte-identical to upstream.
pub fn write_with_jpeg_options(
    image: &Image,
    path: &Path,
    options: &WriteOptions,
    jpeg: &JpegWriteOptions,
) -> Result<()> {
    if image.dimension() != 2 {
        return Err(IoError::JpegWriteRejected(format!(
            "JPEG Writer can only write 2-dimensional images (itkJPEGImageIO.cxx:459-463), \
             not {}-dimensional",
            image.dimension()
        )));
    }
    let component = image.buffer().component_id();
    if component != PixelId::UInt8 {
        return Err(IoError::JpegWriteRejected(format!(
            "JPEG supports unsigned char only (itkJPEGImageIO.cxx:465-468), not {}",
            component.as_str()
        )));
    }

    let width = image.size()[0];
    let height = image.size()[1];
    if width > JPEG_MAX_DIMENSION || height > JPEG_MAX_DIMENSION {
        return Err(IoError::JpegWriteRejected(format!(
            "JPEG: image is too large ({width}x{height}, itkJPEGImageIO.cxx:488-491; \
             JPEG_MAX_DIMENSION = {JPEG_MAX_DIMENSION}, jmorecfg.h:158)"
        )));
    }

    let num_comp = image.buffer_stride();
    if num_comp > MAX_COMPONENTS {
        return Err(IoError::JpegWriteRejected(format!(
            "JPEG: too many components ({num_comp}, itkJPEGImageIO.cxx:492-496; \
             MAX_COMPONENTS = {MAX_COMPONENTS}, jmorecfg.h:30)"
        )));
    }
    let color_type = match num_comp {
        1 => ColorType::Luma,
        3 => ColorType::Rgb,
        _ => {
            return Err(IoError::UnsupportedJpegFeature(format!(
                "a {num_comp}-component JPEG write has no defined colour transform \
                 (itkJPEGImageIO.cxx:521-533 falls to JCS_UNKNOWN with only a warning; \
                 jpeg_encoder::ColorType has no raw-N-plane counterpart) — \
                 doc/upstream-findings.md §4.94"
            )));
        }
    };

    let data = match image.buffer() {
        PixelBuffer::UInt8(v) => v.as_slice(),
        other => unreachable!("component type was already checked to be UInt8, got {other:?}"),
    };

    let file = std::fs::File::create(path)?;
    let mut encoder = JpegEncoder::new(file, resolved_quality(options));
    encoder.set_progressive(jpeg.progressive);
    if num_comp == 3 {
        // `jpeg_default_colorspace` always applies 2x2/1x1/1x1 (4:2:0)
        // sampling for a JCS_YCbCr write, regardless of quality
        // (jcparam.c:374-382); `jpeg_encoder::Encoder::new` would otherwise
        // pick no subsampling at ITK's own default quality of 95 (>= 90).
        // The default `JpegWriteOptions` keeps upstream's constant 4:2:0
        // (ledger §5.27); a caller may pick another factor here.
        encoder.set_sampling_factor(jpeg.chroma_subsampling.sampling_factor());
    }
    let spacing = image.spacing();
    if spacing[0] > 0.0 && spacing[1] > 0.0 {
        encoder.set_density(density_for_spacing(spacing[0], spacing[1]));
    }

    encoder.encode(data, width as u16, height as u16, color_type)?;
    Ok(())
}

/// `itk::JPEGImageIO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct JpegImageIo;

impl ImageIo for JpegImageIo {
    fn name(&self) -> &'static str {
        "JPEGImageIO"
    }

    /// `.jpg`, `.JPG`, `.jpeg`, `.JPEG`, all registered for read and write
    /// (itkJPEGImageIO.cxx:308-314).
    fn supported_read_extensions(&self) -> &'static [&'static str] {
        &[".jpg", ".JPG", ".jpeg", ".JPEG"]
    }

    fn supported_write_extensions(&self) -> &'static [&'static str] {
        &[".jpg", ".JPG", ".jpeg", ".JPEG"]
    }

    /// `CanReadFile` (itkJPEGImageIO.cxx:90-158): extension (case-sensitive),
    /// then the 2-byte `0xFFD8` magic, then a full header parse. Ledger
    /// §2.136.
    fn can_read_file(&self, path: &Path) -> bool {
        if !has_supported_extension(path, self.supported_read_extensions(), false) {
            return false;
        }
        let Ok(bytes) = std::fs::read(path) else {
            return false;
        };
        if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
            return false;
        }
        JpegDecoder::new(Cursor::new(bytes.as_slice()))
            .read_info()
            .is_ok()
    }

    /// `CanWriteFile` is `HasSupportedWriteExtension(name, false)` —
    /// case-sensitive. Ledger §2.137.
    fn can_write_file(&self, path: &Path) -> bool {
        has_supported_extension(path, self.supported_write_extensions(), false)
    }

    fn read_information(&self, path: &Path) -> Result<ImageInformation> {
        read_information(path)
    }

    fn read(&self, path: &Path) -> Result<Image> {
        read(path)
    }

    fn write(&self, image: &Image, path: &Path, options: &WriteOptions) -> Result<()> {
        write(image, path, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn resolved_quality_defaults_to_95_when_unset() {
        assert_eq!(resolved_quality(&WriteOptions::default()), 95);
    }

    #[test]
    fn resolved_quality_clamps_to_one_through_one_hundred() {
        assert_eq!(
            resolved_quality(&WriteOptions {
                use_compression: false,
                compression_level: 0,
                compressor: None,
            }),
            1
        );
        assert_eq!(
            resolved_quality(&WriteOptions {
                use_compression: false,
                compression_level: 250,
                compressor: None,
            }),
            100
        );
        assert_eq!(
            resolved_quality(&WriteOptions {
                use_compression: false,
                compression_level: 42,
                compressor: None,
            }),
            42
        );
    }

    #[test]
    fn resolved_quality_ignores_use_compression() {
        // m_UseCompression is dead code (ledger §2.138): on or off, an
        // explicit level passes through identically.
        let on = WriteOptions {
            use_compression: true,
            compression_level: 80,
            compressor: None,
        };
        let off = WriteOptions {
            use_compression: false,
            compression_level: 80,
            compressor: None,
        };
        assert_eq!(resolved_quality(&on), resolved_quality(&off));
    }

    #[test]
    fn cmyk_to_rgb_inverts_before_applying_the_gimp_formula() {
        // K = 255 (true full black ink): invK = 0, so every channel is
        // forced to 0 regardless of C/M/Y — full black.
        let out = cmyk_to_rgb(&[10, 20, 30, 255]);
        assert_eq!(out, vec![0, 0, 0]);

        // K = 0 (no black ink): invK = 255, so invC·255/255 = invC exactly
        // — the inverted C/M/Y pass through unchanged.
        let out = cmyk_to_rgb(&[10, 20, 30, 0]);
        assert_eq!(out, vec![245, 235, 225]);

        // K = 128: invC=0, invM=155, invY=255, invK=127.
        // R = 0*127/255 = 0; G = 155*127/255 = 19685/255 = 77 (truncated);
        // B = 255*127/255 = 127.
        let out = cmyk_to_rgb(&[255, 100, 0, 128]);
        assert_eq!(out, vec![0, 77, 127]);
    }

    #[test]
    fn scan_jfif_density_finds_app0() {
        // SOI, APP0 "JFIF" with unit=1 (inches), X=300, Y=150, then EOI.
        let mut bytes = vec![0xFF, 0xD8];
        bytes.extend_from_slice(&[0xFF, 0xE0]);
        let mut app0 = b"JFIF\0".to_vec();
        app0.extend_from_slice(&[0x01, 0x02]); // version
        app0.push(0x01); // unit: inches
        app0.extend_from_slice(&300u16.to_be_bytes());
        app0.extend_from_slice(&150u16.to_be_bytes());
        app0.extend_from_slice(&[0, 0]); // no thumbnail
        bytes.extend_from_slice(&((app0.len() + 2) as u16).to_be_bytes());
        bytes.extend_from_slice(&app0);
        bytes.extend_from_slice(&[0xFF, 0xD9]);

        let density = scan_jfif_density(&bytes).expect("APP0 JFIF present");
        assert_eq!(
            density,
            JfifDensity {
                unit: 1,
                x: 300,
                y: 150
            }
        );
    }

    #[test]
    fn scan_jfif_density_skips_unrelated_markers_first() {
        // SOI, a COM marker, then APP0 JFIF with unit=2 (cm).
        let mut bytes = vec![0xFF, 0xD8];
        let comment = b"not density-bearing";
        bytes.extend_from_slice(&[0xFF, 0xFE]);
        bytes.extend_from_slice(&((comment.len() + 2) as u16).to_be_bytes());
        bytes.extend_from_slice(comment);

        bytes.extend_from_slice(&[0xFF, 0xE0]);
        let mut app0 = b"JFIF\0".to_vec();
        app0.extend_from_slice(&[0x01, 0x02]);
        app0.push(0x02); // unit: cm
        app0.extend_from_slice(&118u16.to_be_bytes());
        app0.extend_from_slice(&118u16.to_be_bytes());
        app0.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&((app0.len() + 2) as u16).to_be_bytes());
        bytes.extend_from_slice(&app0);
        bytes.extend_from_slice(&[0xFF, 0xD9]);

        let density = scan_jfif_density(&bytes).expect("APP0 JFIF present after COM");
        assert_eq!(density.unit, 2);
    }

    #[test]
    fn scan_jfif_density_stops_at_sos_and_finds_nothing_past_it() {
        let mut bytes = vec![0xFF, 0xD8];
        // SOS marker with a short (illegal-as-marker) payload — must never
        // be walked as if it were more marker structure.
        bytes.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 1, 2, 3, 4, 5, 6]);
        assert_eq!(scan_jfif_density(&bytes), None);
    }

    #[test]
    fn scan_jfif_density_rejects_a_non_jpeg_stream() {
        assert_eq!(scan_jfif_density(b"not a jpeg at all"), None);
        assert_eq!(scan_jfif_density(&[0xFF]), None);
        assert_eq!(scan_jfif_density(&[]), None);
    }

    #[test]
    fn spacing_from_density_reproduces_read_image_information() {
        assert_eq!(
            spacing_from_density(Some(JfifDensity {
                unit: 1,
                x: 96,
                y: 96
            })),
            [25.4 / 96.0, 25.4 / 96.0]
        );
        assert_eq!(
            spacing_from_density(Some(JfifDensity {
                unit: 2,
                x: 118,
                y: 118
            })),
            [10.0 / 118.0, 10.0 / 118.0]
        );
        // unit == 0 ("no units") never matches either branch — the density
        // write bug's effect (ledger §1.64, §4.95).
        assert_eq!(
            spacing_from_density(Some(JfifDensity {
                unit: 0,
                x: 200,
                y: 200
            })),
            [1.0, 1.0]
        );
        assert_eq!(spacing_from_density(None), [1.0, 1.0]);
    }

    #[test]
    fn density_for_spacing_prefers_the_lower_error_encoding() {
        // 25.4 / 0.75 = 33.86..., which rounds to inch density 34 with tiny
        // error; the cm equivalent (10 / 0.75 = 13.33) rounds with more
        // relative error, so the inches branch wins cleanly.
        assert_eq!(
            density_for_spacing(0.75, 0.75),
            Density::Inch { x: 34, y: 34 }
        );
    }

    #[test]
    fn density_for_spacing_returns_none_when_the_cm_branch_wins() {
        // Reproduces the density-write bug's effect (ledger §1.64, §4.95):
        // whichever spacing favours the centimetres encoding must not come
        // back as `Density::Centimeter`, since upstream tags that branch
        // `density_unit = 0`, not `2`.
        let d = density_for_spacing(2.0, 2.0);
        assert_eq!(d, Density::None);
        assert!(!matches!(d, Density::Centimeter { .. }));
    }

    #[test]
    fn can_write_file_is_case_sensitive_over_four_spellings() {
        let io = JpegImageIo;
        assert!(io.can_write_file(Path::new("a.jpg")));
        assert!(io.can_write_file(Path::new("a.JPG")));
        assert!(io.can_write_file(Path::new("a.jpeg")));
        assert!(io.can_write_file(Path::new("a.JPEG")));
        assert!(!io.can_write_file(Path::new("a.Jpg")));
        assert!(!io.can_write_file(Path::new("a.JpEg")));
    }

    #[test]
    fn can_read_file_requires_extension_magic_and_a_full_header_parse() {
        let io = JpegImageIo;

        // Right extension, but garbage content: extension and magic both
        // fail, or the header parse fails.
        let bad_ext = temp_path("sitk_io_jpeg_bad_ext.png");
        std::fs::write(&bad_ext, [0xFFu8, 0xD8, 0xFF, 0xE0]).unwrap();
        assert!(!io.can_read_file(&bad_ext));
        std::fs::remove_file(&bad_ext).ok();

        let bad_magic = temp_path("sitk_io_jpeg_bad_magic.jpg");
        std::fs::write(&bad_magic, b"not a jpeg").unwrap();
        assert!(!io.can_read_file(&bad_magic));
        std::fs::remove_file(&bad_magic).ok();

        let truncated = temp_path("sitk_io_jpeg_truncated.jpg");
        std::fs::write(&truncated, [0xFFu8, 0xD8, 0xFF, 0xE0, 0x00]).unwrap();
        assert!(!io.can_read_file(&truncated));
        std::fs::remove_file(&truncated).ok();
    }

    fn gray_image(width: usize, height: usize) -> Image {
        let data: Vec<u8> = (0..(width * height))
            .map(|i| ((i * 37 + i * i) % 256) as u8)
            .collect();
        Image::from_vec(&[width, height], data).unwrap()
    }

    #[test]
    fn grayscale_round_trip_preserves_dimensions_and_pixel_type() {
        let image = gray_image(16, 12);
        let path = temp_path("sitk_io_jpeg_gray_roundtrip.jpg");
        write(&image, &path, &WriteOptions::default()).unwrap();

        let read_back = read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(read_back.pixel_id(), PixelId::UInt8);
        assert_eq!(read_back.size(), &[16, 12]);
    }

    #[test]
    fn rgb_round_trip_preserves_dimensions_and_pixel_type() {
        let width = 20;
        let height = 10;
        let data: Vec<u8> = (0..(width * height * 3))
            .map(|i| ((i * 53) % 256) as u8)
            .collect();
        let image = Image::from_vec_vector(&[width, height], 3, data).unwrap();

        let path = temp_path("sitk_io_jpeg_rgb_roundtrip.jpg");
        write(&image, &path, &WriteOptions::default()).unwrap();

        let read_back = read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(read_back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(read_back.size(), &[width, height]);
        assert_eq!(read_back.number_of_components_per_pixel(), 3);
    }

    #[test]
    fn spacing_round_trips_on_the_inches_branch() {
        let image = Image::from_parts(
            PixelBuffer::UInt8(vec![0u8; 8 * 8]),
            vec![8, 8],
            vec![0.75, 0.75],
            vec![0.0, 0.0],
            identity(2),
        )
        .unwrap();

        let path = temp_path("sitk_io_jpeg_spacing_inches.jpg");
        write(&image, &path, &WriteOptions::default()).unwrap();
        let info = read_information(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let expected = density_for_spacing(0.75, 0.75);
        let Density::Inch { x, y } = expected else {
            panic!("0.75 spacing should favour the inches branch");
        };
        assert_eq!(info.spacing, vec![25.4 / f64::from(x), 25.4 / f64::from(y)]);
    }

    #[test]
    fn spacing_is_lost_on_the_cm_branch_reproducing_the_density_bug() {
        // Ledger §1.64/§4.95: a spacing that favours the (buggy) centimetres
        // branch round-trips to the default [1.0, 1.0], not the value
        // written.
        let image = Image::from_parts(
            PixelBuffer::UInt8(vec![0u8; 8 * 8]),
            vec![8, 8],
            vec![2.0, 2.0],
            vec![0.0, 0.0],
            identity(2),
        )
        .unwrap();

        let path = temp_path("sitk_io_jpeg_spacing_cm_bug.jpg");
        write(&image, &path, &WriteOptions::default()).unwrap();
        let info = read_information(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(info.spacing, vec![1.0, 1.0]);
    }

    #[test]
    fn write_rejects_a_three_dimensional_image() {
        let image = Image::from_vec(&[4, 4, 2], vec![0u8; 32]).unwrap();
        let path = temp_path("sitk_io_jpeg_reject_3d.jpg");
        let err = write(&image, &path, &WriteOptions::default()).unwrap_err();
        assert!(matches!(err, IoError::JpegWriteRejected(_)), "{err:?}");
        assert!(!path.exists());
    }

    #[test]
    fn write_rejects_a_non_uint8_component_type() {
        let image = Image::from_vec(&[4, 4], vec![0.0f32; 16]).unwrap();
        let path = temp_path("sitk_io_jpeg_reject_component_type.jpg");
        let err = write(&image, &path, &WriteOptions::default()).unwrap_err();
        assert!(matches!(err, IoError::JpegWriteRejected(_)), "{err:?}");
    }

    #[test]
    fn write_rejects_a_two_component_image() {
        let image = Image::from_vec_vector(&[4, 4], 2, vec![0u8; 32]).unwrap();
        let path = temp_path("sitk_io_jpeg_reject_2_component.jpg");
        let err = write(&image, &path, &WriteOptions::default()).unwrap_err();
        assert!(matches!(err, IoError::UnsupportedJpegFeature(_)), "{err:?}");
    }

    #[test]
    fn read_rejects_a_two_component_jpeg() {
        // A minimal SOI + SOF0 declaring 2 components. `parse_sof` itself
        // accepts any nonzero component count; jpeg_decoder's own
        // component-count gate (decoder.rs:368-372) is what refuses it,
        // right after the frame header parses and before any DQT/DHT/SOS is
        // needed — ledger §4.96, the divergence from upstream's `default:`
        // VECTOR-plus-warning branch (itkJPEGImageIO.cxx:412-417).
        #[rustfmt::skip]
        let bytes: [u8; 18] = [
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x0E, // length = 14 (12 payload bytes + the length field)
            0x08,       // precision
            0x00, 0x02, // height = 2
            0x00, 0x02, // width = 2
            0x02,       // component count = 2
            0x01, 0x11, 0x00, // component 1: id=1, sampling=1x1, qtable=0
            0x02, 0x11, 0x00, // component 2: id=2, sampling=1x1, qtable=0
        ];

        let path = temp_path("sitk_io_jpeg_two_component.jpg");
        std::fs::write(&path, bytes).unwrap();
        let err = read(&path).unwrap_err();
        std::fs::remove_file(&path).ok();

        assert!(matches!(err, IoError::JpegDecode(_)), "{err:?}");
    }

    /// `jpeg_encoder::ColorType::Cmyk` is used only as a **test fixture
    /// builder** — production `write` never emits CMYK (module doc, ledger
    /// §3.50 / §4.94 — a 4-component write is refused). A uniform-colour
    /// image round-trips losslessly through JPEG's DCT (no AC energy in any
    /// block), so the decoded RGB is checked for the exact hand-computed
    /// value, not just shape: C=10, M=200, Y=50, K=0 (true, un-inverted —
    /// `jpeg_encoder`'s `CmykImage` inverts internally before writing) gives
    /// invC=245, invM=55, invY=205, invK=255, so R=245·255/255=245,
    /// G=55·255/255=55, B=205·255/255=205 (ledger §1.65).
    #[test]
    fn cmyk_source_reads_back_as_the_inverted_gimp_formula_rgb() {
        let width = 16;
        let height = 16;
        let data: Vec<u8> = std::iter::repeat_n([10u8, 200, 50, 0], width * height)
            .flatten()
            .collect();

        let path = temp_path("sitk_io_jpeg_cmyk_fixture.jpg");
        let file = std::fs::File::create(&path).unwrap();
        let encoder = JpegEncoder::new(file, 100);
        encoder
            .encode(&data, width as u16, height as u16, ColorType::Cmyk)
            .unwrap();

        let read_back = read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(read_back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(read_back.size(), &[width, height]);
        assert_eq!(read_back.number_of_components_per_pixel(), 3);
        let pixels = match read_back.buffer() {
            PixelBuffer::UInt8(v) => v.as_slice(),
            other => panic!("expected UInt8, got {other:?}"),
        };
        for px in pixels.chunks_exact(3) {
            assert_eq!(px, [245, 55, 205]);
        }
    }

    /// The bug this row fixes: upstream applies the same "everything is
    /// inverted" formula to raw libjpeg bytes regardless of whether the
    /// source was plain CMYK or YCCK, which is only correct for the former
    /// (module doc's "CMYK → RGB" section). `jpeg_decoder::Decoder` resolves
    /// the ambiguity itself before this crate ever sees a byte, so encoding
    /// the *same* true-CMYK pixel via `ColorType::Cmyk` and
    /// `ColorType::CmykAsYcck` must read back to the *same* RGB — proving the
    /// YCCK path is no longer silently wrong.
    #[test]
    fn cmyk_and_ycck_sources_of_the_same_color_read_back_to_the_same_rgb() {
        let width = 16;
        let height = 16;
        let data: Vec<u8> = std::iter::repeat_n([10u8, 200, 50, 0], width * height)
            .flatten()
            .collect();

        let cmyk_path = temp_path("sitk_io_jpeg_cmyk_vs_ycck_cmyk.jpg");
        let file = std::fs::File::create(&cmyk_path).unwrap();
        JpegEncoder::new(file, 100)
            .encode(&data, width as u16, height as u16, ColorType::Cmyk)
            .unwrap();

        let ycck_path = temp_path("sitk_io_jpeg_cmyk_vs_ycck_ycck.jpg");
        let file = std::fs::File::create(&ycck_path).unwrap();
        JpegEncoder::new(file, 100)
            .encode(&data, width as u16, height as u16, ColorType::CmykAsYcck)
            .unwrap();

        let cmyk_read_back = read(&cmyk_path).unwrap();
        let ycck_read_back = read(&ycck_path).unwrap();
        std::fs::remove_file(&cmyk_path).ok();
        std::fs::remove_file(&ycck_path).ok();

        let cmyk_pixels = match cmyk_read_back.buffer() {
            PixelBuffer::UInt8(v) => v.as_slice(),
            other => panic!("expected UInt8, got {other:?}"),
        };
        let ycck_pixels = match ycck_read_back.buffer() {
            PixelBuffer::UInt8(v) => v.as_slice(),
            other => panic!("expected UInt8, got {other:?}"),
        };
        assert_eq!(cmyk_pixels, ycck_pixels);
        for px in cmyk_pixels.chunks_exact(3) {
            assert_eq!(px, [245, 55, 205]);
        }
    }

    // -- JPEG write options (§3.50 / §5.27) ---------------------------------

    /// Walk the marker stream to the first SOF (baseline `0xC0` or progressive
    /// `0xC2`) and return `(sof_marker_code, first_component_sampling_byte)`.
    /// The sampling byte packs H in the high nibble and V in the low nibble,
    /// so luma is `0x11` for 4:4:4, `0x21` for 4:2:2, `0x22` for 4:2:0.
    fn first_sof(bytes: &[u8]) -> (u8, u8) {
        assert_eq!(&bytes[0..2], &[0xFF, 0xD8], "not a JPEG stream");
        let mut pos = 2;
        loop {
            assert_eq!(bytes[pos], 0xFF, "expected a marker at offset {pos}");
            let marker = bytes[pos + 1];
            if marker == 0xC0 || marker == 0xC2 {
                // len(2), precision(1), height(2), width(2), ncomp(1), then
                // component 0: id(1), sampling(1) — the sampling byte is
                // `2 (0xFF+code) + 2 (len) + 1 + 2 + 2 + 1 + 1` past `pos`.
                return (marker, bytes[pos + 11]);
            }
            let len = u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
            pos += 2 + len;
        }
    }

    fn rgb_image() -> Image {
        let data: Vec<u8> = (0..(20 * 10 * 3)).map(|i| ((i * 53) % 256) as u8).collect();
        Image::from_vec_vector(&[20, 10], 3, data).unwrap()
    }

    /// `JpegWriteOptions::default` reproduces upstream's fixed behaviour —
    /// progressive, 4:2:0 — and `write` is byte-identical to
    /// `write_with_jpeg_options` with those defaults.
    #[test]
    fn jpeg_write_defaults_match_upstream_progressive_and_4_2_0() {
        assert_eq!(
            JpegWriteOptions::default(),
            JpegWriteOptions {
                progressive: true,
                chroma_subsampling: JpegChromaSubsampling::Chroma420,
            }
        );

        let image = rgb_image();
        let via_write = temp_path("sitk_io_jpeg_default_write.jpg");
        let via_options = temp_path("sitk_io_jpeg_default_options.jpg");
        write(&image, &via_write, &WriteOptions::default()).unwrap();
        write_with_jpeg_options(
            &image,
            &via_options,
            &WriteOptions::default(),
            &JpegWriteOptions::default(),
        )
        .unwrap();
        let write_bytes = std::fs::read(&via_write).unwrap();
        let options_bytes = std::fs::read(&via_options).unwrap();
        std::fs::remove_file(&via_write).ok();
        std::fs::remove_file(&via_options).ok();
        assert_eq!(write_bytes, options_bytes, "default path is byte-identical");

        let (sof, luma) = first_sof(&write_bytes);
        assert_eq!(sof, 0xC2, "default write is progressive (SOF2)");
        assert_eq!(luma, 0x22, "default write is 4:2:0");
    }

    /// `progressive = false` emits a baseline (SOF0) JPEG, and it still reads
    /// back as a 3-component image.
    #[test]
    fn jpeg_progressive_false_produces_a_baseline_jpeg() {
        let image = rgb_image();
        let path = temp_path("sitk_io_jpeg_baseline.jpg");
        write_with_jpeg_options(
            &image,
            &path,
            &WriteOptions::default(),
            &JpegWriteOptions {
                progressive: false,
                chroma_subsampling: JpegChromaSubsampling::Chroma420,
            },
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let read_back = read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let (sof, _) = first_sof(&bytes);
        assert_eq!(sof, 0xC0, "progressive=false is a baseline SOF0 JPEG");
        assert_eq!(read_back.number_of_components_per_pixel(), 3);
        assert_eq!(read_back.size(), &[20, 10]);
    }

    /// Each chroma-subsampling scheme sets the luma sampling factor it names,
    /// and every one still round-trips.
    #[test]
    fn jpeg_chroma_subsampling_selects_the_sampling_factor() {
        let image = rgb_image();
        for (scheme, luma) in [
            (JpegChromaSubsampling::None, 0x11u8),
            (JpegChromaSubsampling::Chroma422, 0x21),
            (JpegChromaSubsampling::Chroma420, 0x22),
        ] {
            let path = temp_path(&format!("sitk_io_jpeg_chroma_{luma:#x}.jpg"));
            write_with_jpeg_options(
                &image,
                &path,
                &WriteOptions::default(),
                &JpegWriteOptions {
                    progressive: true,
                    chroma_subsampling: scheme,
                },
            )
            .unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let read_back = read(&path).unwrap();
            std::fs::remove_file(&path).ok();

            let (_, got) = first_sof(&bytes);
            assert_eq!(got, luma, "{scheme:?} luma sampling factor");
            assert_eq!(
                read_back.number_of_components_per_pixel(),
                3,
                "{scheme:?} round-trips"
            );
            assert_eq!(read_back.size(), &[20, 10]);
        }
    }

    /// [`read_preserving_cmyk`] keeps the raw four uninverted CMYK channels as
    /// a 4-component vector image, where [`read`] flattens the same source to
    /// the Gimp-formula RGB (§3.50). Same CMYK fixture as
    /// `cmyk_source_reads_back_as_the_inverted_gimp_formula_rgb`.
    #[test]
    fn read_preserving_cmyk_keeps_the_raw_four_uninverted_channels() {
        let width = 16;
        let height = 16;
        let data: Vec<u8> = std::iter::repeat_n([10u8, 200, 50, 0], width * height)
            .flatten()
            .collect();

        let path = temp_path("sitk_io_jpeg_cmyk_preserve.jpg");
        let file = std::fs::File::create(&path).unwrap();
        JpegEncoder::new(file, 100)
            .encode(&data, width as u16, height as u16, ColorType::Cmyk)
            .unwrap();

        let preserved = read_preserving_cmyk(&path).unwrap();
        let converted = read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // Preserved: 4-component vector of the raw uninverted CMYK.
        assert_eq!(preserved.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(preserved.number_of_components_per_pixel(), 4);
        assert_eq!(preserved.size(), &[width, height]);
        let preserved_px = match preserved.buffer() {
            PixelBuffer::UInt8(v) => v.as_slice(),
            other => panic!("expected UInt8, got {other:?}"),
        };
        for chunk in preserved_px.chunks_exact(4) {
            assert_eq!(chunk, [10, 200, 50, 0]);
        }

        // `read` still flattens to RGB — upstream's only reachable path.
        assert_eq!(converted.number_of_components_per_pixel(), 3);
        let converted_px = match converted.buffer() {
            PixelBuffer::UInt8(v) => v.as_slice(),
            other => panic!("expected UInt8, got {other:?}"),
        };
        for chunk in converted_px.chunks_exact(3) {
            assert_eq!(chunk, [245, 55, 205]);
        }
    }
}
