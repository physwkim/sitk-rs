//! TIFF (`.tif`, `.tiff`) reader and writer — `itk::TIFFImageIO`, on the
//! pure-Rust `tiff` crate (ledger §5.8(a): `cargo tree -p sitk-io` shows no
//! `*-sys` crate — `tiff` rides on `weezl` for LZW, `flate2`/`miniz_oxide` for
//! Deflate, `zune-jpeg` for JPEG and `fax` for CCITT).
//!
//! Upstream is `itkTIFFImageIO.cxx` plus `itkTIFFReaderInternal.cxx`, both of
//! which delegate the hard parts to libtiff. Where libtiff offers a door the
//! `tiff` crate does not, this module refuses rather than substituting a
//! different behaviour; every such refusal is ledgered.
//!
//! # `TIFFReaderInternal::CanRead` splits the reader in two
//!
//! `ReadImageInformation` asks `m_InternalImage->CanRead()`
//! (itkTIFFReaderInternal.cxx:276-290), which is true only for an untiled,
//! contiguous (or single-sample) image whose photometric interpretation is
//! `RGB` / `MINISWHITE` / `MINISBLACK` / `PALETTE`(with `BitsPerSample != 32`),
//! whose orientation is `TOPLEFT` or `BOTLEFT`, and whose `BitsPerSample` is 8,
//! 16 or 32. That subset is read scanline by scanline, at the file's own
//! component type. **Everything else** — tiled images, CMYK, YCbCr, CIELab,
//! bilevel, planar multi-sample, exotic orientations — falls back to
//! `TIFFReadRGBAImageOriented`, which libtiff renders into 8-bit RGBA
//! regardless of the source depth, and `ReadImageInformation` rewrites the
//! pixel type to 4-component `UCHAR` `RGBA` (itkTIFFImageIO.cxx:494-533).
//!
//! The `tiff` crate has no equivalent of `TIFFReadRGBAImageOriented`: it
//! decodes to the file's own sample type and refuses the colour spaces libtiff
//! converts. Reproducing libtiff's RGBA renderer — its YCbCr and CMYK
//! conversions, its bit-depth scaling, its per-photometric alpha defaults —
//! would be writing a second libtiff, not porting `TIFFImageIO`. So this module
//! implements **only** the `CanRead()` branch, and reports
//! [`IoError::UnsupportedTiffFeature`] where upstream would fall back. Ledger
//! §4.102.
//!
//! # Palette images are not readable at all
//!
//! `PHOTOMETRIC_PALETTE` is inside `CanRead()`'s accepted set, and
//! `ReadImageInformation` classifies it through `GetFormat`
//! (itkTIFFImageIO.cxx:96-143): with `m_ExpandRGBPalette` on it scans the
//! colour map and picks `PALETTE_GRAYSCALE` when every entry has
//! `red == green == blue`, else `PALETTE_RGB` (expanded to 3 components); the
//! component type is promoted to `USHORT` when any colour-map entry exceeds 255
//! (`:466-491`). With `m_ExpandRGBPalette` off it keeps the raw indices and
//! sets `m_IsReadAsScalarPlusPalette`.
//!
//! As with PNG (§2.131), the off branch is unreachable through this crate:
//! `bool m_ExpandRGBPalette{}` in the header reads `false`, but
//! `ImageIOBase`'s constructor calls `Reset(false)`, which assigns
//! `m_ExpandRGBPalette = true` unconditionally (itkImageIOBase.cxx:28,45), and
//! `TIFFImageIO`'s own constructor never touches it (itkTIFFImageIO.cxx:206-232)
//! — exactly PNG's shape. SimpleITK calls neither `SetExpandRGBPalette` nor
//! `ExpandRGBPaletteOff` anywhere, so a palette TIFF is always expanded.
//!
//! The expansion cannot be performed here: `tiff`'s `Image::colortype` returns
//! `Err(TiffUnsupportedError::InterpretationWithBits)` for
//! `PhotometricInterpretation::RGBPalette` (decoder/image.rs:518-524), and
//! *every* decode entry point — `read_image`, `read_image_bytes`,
//! `read_chunk_bytes` — routes through `readout_for_size`, which calls
//! `colortype()?` (decoder/image.rs:698). The index samples are therefore
//! unreachable, and there is no lower-level door: the strip decompressors are
//! private. [`read`] and [`read_information`] both refuse a palette TIFF.
//! Ledger §4.100, §5.28.
//!
//! # `MINISWHITE` is read out un-inverted upstream — this port returns the true value
//!
//! `GetFormat` maps both `PHOTOMETRIC_MINISWHITE` and `PHOTOMETRIC_MINISBLACK`
//! to `GRAYSCALE` (itkTIFFImageIO.cxx:111-114), and `ReadGenericImage`'s
//! `GRAYSCALE` arm is `PutGrayscale`, a plain `std::copy_n` under a
//! `// check inverted` comment that never grew a body
//! (`:1399-1402`, `:1467-1482`). A white-is-zero TIFF therefore reads out with
//! its raw samples, tone-inverted relative to what a viewer shows — a silently
//! wrong value, not a format quirk to reproduce. Ledger §2.139.
//!
//! The `tiff` crate already inverts correctly, calling `invert_colors`
//! whenever `photometric_interpretation == WhiteIsZero` (decoder/image.rs:948)
//! for every arm it supports (unsigned 8/16/32-bit and `f32`,
//! decoder/mod.rs:718-748). **Fixed §2.139** — [`read`] no longer reverses
//! that correction back to upstream's raw, tone-inverted value; the crate's
//! own output is used as-is, which also means a 32-bit-float `MINISWHITE`
//! image is no longer refused (that refusal existed solely to avoid
//! round-tripping back through a non-invertible `1.0 - v`, a concern that
//! does not arise when the corrected value is kept rather than undone).
//!
//! # A multi-sample `MINISBLACK` image reads its first half-row, twice over — this port reads every sample
//!
//! `GetFormat` maps `MINISBLACK` to `GRAYSCALE` before ever looking at
//! `SamplesPerPixel`, so `ReadImageInformation` sets `NumberOfComponents` to 1
//! (itkTIFFImageIO.cxx:436-440) and `ReadGenericImage` sets `inc = 1`
//! (`:1357-1360`). `PutGrayscale` then copies `xsize == width` components off a
//! scanline that holds `width * SamplesPerPixel` of them (`:1474-1481`). A
//! two-sample grey+alpha TIFF thus yields, for each row, the interleaved pairs
//! of its first `width / 2` pixels. No error, no warning — `CanRead()` is
//! perfectly happy with it, and the declared-but-never-defined
//! `ReadTwoSamplesPerPixelImage` (itkTIFFImageIO.h:228-229) has no body in the
//! `.cxx` at all. Ledger §2.140.
//!
//! Dropping every sample past the first is silent data loss, not a shape to
//! reproduce: this crate's [`PixelId`] has no restriction against a
//! multi-component grayscale image, so **fixed §2.140** — [`layout_for`] now
//! sets `number_of_components` to the page's true `SamplesPerPixel` for
//! `GRAYSCALE` too, exactly as it already did for `RGB`, and a `MINISBLACK`
//! page reads back as a vector image with every sample intact.
//!
//! A `MINISWHITE` page is decodable only where the `tiff` crate can invert it:
//! its `invert_colors` handles single-sample `Gray` with an unsigned-integer
//! sample at a conformant bit depth (1/2/4/8/16/32/64) or a 32/64-bit float
//! sample and nothing else (decoder/mod.rs:713-769), so a multi-sample
//! `MINISWHITE` page (`ColorType::Multiband`), a signed / non-32-bit-float
//! single-sample one, or an odd-width `Gray` makes [`read`] error with
//! `UnknownInterpretation`. [`layout_for`] refuses those at
//! `read_information` time (`crate_can_invert_whiteiszero`) so the header does
//! not advertise components [`read`] cannot deliver — the same
//! `colortype()`-based decodability guard the `RGB` arm applies. This also makes
//! [`page_rows`]'s two stride parameters always equal — `layout.number_of_components`
//! and `page.samples_per_pixel` are the same count on every path once
//! `read_current_page`'s per-page geometry check (§1.67) has passed — so
//! [`page_rows`] takes a single `components` count and copies each row
//! verbatim rather than truncating it.
//!
//! # A 2-component image wrote as `PHOTOMETRIC_RGB` with 2 samples — this port writes `MINISBLACK` plus one extra sample
//!
//! `InternalWrite` picks its photometric with `if (m_NumberOfComponents == 1)
//! MINISBLACK else RGB` (itkTIFFImageIO.cxx:725-732), so a 2-component image
//! was written as `PHOTOMETRIC_RGB` with `SamplesPerPixel = 2` — a file no
//! TIFF reader can interpret as colour, and which this crate's own reader
//! refuses outright (§4.102). Ledger §2.141.
//!
//! **Fixed §2.141** — a 2-component image now writes `PHOTOMETRIC_MINISBLACK`
//! with one `ExtraSamples::Unspecified` sample past the declared grayscale
//! sample, a standard, unambiguous TIFF construct: `SamplesPerPixel = 2`
//! with an `ExtraSamples` tag describing the second. [`layout_for`]'s §2.140
//! fix reads that back as a 2-component vector image, every sample intact.
//!
//! # Multi-page TIFF → 3-D volume, and two ways upstream walks off the buffer
//!
//! `Initialize` counts directories, then — only when there is more than one —
//! counts pages whose `SUBFILETYPE` tag is exactly `0` into `m_SubFiles` and
//! pages flagged `FILETYPE_REDUCEDIMAGE` or `FILETYPE_MASK` into
//! `m_IgnoredSubFiles` (itkTIFFReaderInternal.cxx:231-256). A page carrying no
//! `SUBFILETYPE` tag lands in neither counter. `ReadImageInformation` then goes
//! 3-D when `m_NumberOfPages - m_IgnoredSubFiles > 1`, with
//! `m_Dimensions[2] = m_SubFiles > 0 ? m_SubFiles : m_NumberOfPages - m_IgnoredSubFiles`
//! (itkTIFFImageIO.cxx:352-365).
//!
//! `ReadVolume` then loops `page` over **all** `m_NumberOfPages` directories,
//! skipping the ignored ones, and offsets each page's pixels by
//! `width * height * components * page` — the *directory* index, not a running
//! count of pages actually written (`:153-175`). Two shapes overflow the
//! `m_Dimensions[2]`-slice buffer:
//!
//! 1. an ignored page **before** a kept one — `[REDUCEDIMAGE, 0, 0]` gives
//!    `m_SubFiles == 2` and slices `{1, 2}` are written into a two-slice
//!    buffer;
//! 2. a page with **no** `SUBFILETYPE` tag mixed with a `0`-tagged one —
//!    `[absent, 0]` gives `m_SubFiles == 1`, `m_IgnoredSubFiles == 0`, so the
//!    volume has one slice while both directories are read, and the second
//!    lands at slice 1.
//!
//! Both are heap buffer overflows in C++. They are not expressible in safe
//! Rust, so [`read`] keeps upstream's `page`-indexed offset and returns
//! [`IoError::UnsupportedTiffFeature`] when it would leave the buffer. Ledger
//! §1.66.
//!
//! # Mixed page geometry is an over-read upstream, an error here
//!
//! Neither `ReadVolume` nor `ReadCurrentPage` re-reads any page's tags:
//! `m_InternalImage`'s `m_Width`, `m_Height`, `m_SamplesPerPixel`,
//! `m_BitsPerSample`, `m_Photometrics`, `m_Orientation` and `m_PlanarConfig`
//! are all whatever `Initialize` read off directory 0
//! (itkTIFFReaderInternal.cxx:258-270). But `ReadGenericImage` sizes its
//! scanline buffer from `TIFFScanlineSize64` of the *current* directory
//! (itkTIFFImageIO.cxx:1332-1339), then copies page-0's `width * inc`
//! components out of it. A narrower second page reads past the buffer; a wider
//! one silently truncates; a shorter one throws `"Problem reading the row"`.
//!
//! This module closes the family with one uniform rule instead of a check per
//! symptom: **every directory must match directory 0** in width, height,
//! photometric interpretation, bits per sample, samples per pixel, sample
//! format, planar configuration, orientation, and tiling. A volume whose pages
//! agree reads exactly as upstream reads it; a volume whose pages disagree —
//! precisely the set upstream reads out of bounds — is refused. Ledger §1.67,
//! §4.99.
//!
//! # Spacing comes from the resolution tags, and defaults to `1.0` per axis
//!
//! `Clean()` seeds `m_XResolution = m_YResolution = 1` and
//! `m_ResolutionUnit = 1` ("none"), and `Initialize` overwrites them only when
//! the tags are present (itkTIFFReaderInternal.cxx:175-180, :202-204).
//! `ReadImageInformation` then converts: unit `2` (inch) gives
//! `25.4 / resolution`, unit `3` (centimetre) gives `10.0 / resolution`, and
//! every other unit — including the "none" default — leaves the spacing at
//! `1.0` (itkTIFFImageIO.cxx:375-387).
//!
//! A file carrying `RESOLUTIONUNIT = 2` but **no** `XRESOLUTION` therefore
//! reports a spacing of `25.4 / 1 == 25.4` mm rather than `1.0`: the guard
//! `m_XResolution > 0` passes on `Clean()`'s seed. Ledger §2.142.
//!
//! # 16-bit endianness needs no hand-swapping
//!
//! Unlike PNG, whose wire format is fixed big-endian and whose `PNGImageIO`
//! calls `png_set_swap` by hand, a TIFF declares its own byte order in the
//! header (`II` / `MM`) and libtiff's `TIFFReadScanline` normalises samples to
//! host order before `ReadGenericImage` ever sees them. The `tiff` crate does
//! the same, in `fix_endianness` (decoder/mod.rs:771-782). Both sides therefore
//! hand this module native-order `u16`/`i16`/`u32`/`i32`/`f32`, and [`read`]
//! swaps nothing.
//!
//! # Compression: `PackBits`, `LZW` and `Deflate` are selectable on write
//!
//! `TIFFImageIO`'s constructor calls `SetCompressor("")`, which
//! `InternalSetCompressor` resolves to `PackBits`
//! (itkTIFFImageIO.cxx:214, :260-290) — the empty string is *not* "no
//! compression". `InternalWrite` then picks `COMPRESSION_NONE` unless
//! `m_UseCompression` is on, in which case it maps `m_Compression` to
//! `LZW` / `PACKBITS` / `JPEG` / `DEFLATE` / `ADOBE_DEFLATE`
//! (`:692-722`). Selecting one of those needs `SetCompressor("LZW")` etc.
//! SimpleITK's own `ImageFileWriter` *does* expose that selector
//! (`sitkImageFileWriter.h:123`, forwarded at `sitkImageFileWriter.cxx:237-240`).
//! Ledger §3.51.
//!
//! **Fixed §3.51** — of that selector, this crate's encoder can write
//! `LZW`, `PackBits` and one `Deflate` (the `tiff` crate has no JPEG
//! encoder at all, and its `Deflate` writes only the modern tag `8`, never
//! upstream's legacy tag `32946`; see [`TiffCompressor`]). Those three are
//! now reachable through [`WriteOptions::compressor`] /
//! [`ImageFileWriter::set_compressor`](crate::writer::ImageFileWriter::set_compressor),
//! gated by [`WriteOptions::use_compression`] exactly as upstream gates
//! `m_Compression` on `m_UseCompression` (`:692-716`). `None` (the default)
//! keeps the prior behaviour: `COMPRESSION_NONE` ↔ `COMPRESSION_PACKBITS`.
//!
//! Upstream's `m_CompressionLevel` is TIFF's *JPEG quality*: `SetJPEGQuality`
//! is `SetCompressionLevel` (itkTIFFImageIO.h:171-180) and the constructor
//! seeds it with `75` (`:213`). It reaches libtiff only through
//! `TIFFSetField(tif, TIFFTAG_JPEGQUALITY, ...)` inside
//! `if (compression == COMPRESSION_JPEG)` (itkTIFFImageIO.cxx:747-751) — a
//! mapping this port cannot reproduce, since it cannot write JPEG at all
//! (§3.51). It is also the only `ImageIOBase` in this crate that leaves
//! `m_MaximumCompressionLevel` at its `100` default rather than lowering it to
//! `9` (itkImageIOBase.h:830). Ledger §3.52.
//!
//! **Fixed §3.52** — rather than leave [`WriteOptions::compression_level`]
//! permanently dead here (there being no pure-Rust JPEG encoder for it to
//! drive), it now controls the ratio of the one compressor this port can
//! actually vary: `deflate_level_for` maps the resolved `1..=9` level onto
//! [`TiffCompressor::Deflate`]'s three [`DeflateLevel`] tiers. This is a new
//! use of the knob, not a port of one upstream ever had — TIFF's
//! `CompressionLevel` never controlled a Deflate ratio in `itkTIFFImageIO`,
//! only JPEG quality.
//!
//! On read, every compression scheme libtiff has a codec for is accepted
//! (`TIFFIsCODECConfigured`). The `tiff` crate supports `None`, `LZW`,
//! `Deflate` (both `8` and the old `0x80B2`), `PackBits`, `ModernJPEG` and the
//! CCITT fax codecs; anything else surfaces as its own
//! `UnsupportedCompressionMethod`.
//!
//! # Not implemented
//!
//! * **The meta-data dictionary.** `ReadTIFFTags` walks libtiff's tag list and
//!   encapsulates every tag into `m_MetaDataDictionary` keyed by
//!   `TIFFFieldName(field)`, dispatching on `TIFFFieldDataType`,
//!   `TIFFFieldReadCount` and `TIFFFieldPassCount`
//!   (itkTIFFImageIO.cxx:1055-1256). That machinery is libtiff's field
//!   registry, which the `tiff` crate does not expose — its `Tag` enum carries
//!   no names, read counts or pass-count flags. [`read_information`] therefore
//!   returns an empty dictionary. Ledger §4.103.
//! * **`SetColorPalette` / `WritePalette`.** `InternalWrite` writes
//!   `PHOTOMETRIC_PALETTE` and a colour map when `GetWritePalette()` is on and
//!   the image is scalar (itkTIFFImageIO.cxx:724-738, :1000-1052). SimpleITK
//!   exposes no `WritePalette` setter, so — as for PNG (§4.87) — the branch is
//!   unreachable and every 1-component image writes `MINISBLACK`.
//! * **`SetCompressor`.** See above; ledger §6 covers the crate-wide decision.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::marker::PhantomData;
use std::path::Path;

use sitk_core::{Image, PixelBuffer, PixelId};
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::colortype::ColorType as EncoderColorType;
use tiff::encoder::{Compression, DeflateLevel, Rational, TiffEncoder, TiffKind, TiffValue};
use tiff::tags::{ExtraSamples, PhotometricInterpretation, ResolutionUnit, SampleFormat, Tag};

use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, ImageIo, has_supported_extension};
use crate::writer::WriteOptions;

/// `TIFFTAG_PAGENUMBER`, which `tiff::tags::Tag` has no variant for.
const TAG_PAGE_NUMBER: Tag = Tag::from_u16_exhaustive(297);

/// `ORIENTATION_TOPLEFT`.
const ORIENTATION_TOPLEFT: u16 = 1;
/// `ORIENTATION_BOTLEFT`.
const ORIENTATION_BOTLEFT: u16 = 4;

/// `PLANARCONFIG_CONTIG`.
const PLANARCONFIG_CONTIG: u16 = 1;

/// `FILETYPE_REDUCEDIMAGE`.
const FILETYPE_REDUCEDIMAGE: u32 = 1;
/// `FILETYPE_PAGE`, the `SUBFILETYPE` `InternalWrite` stamps on every page of a
/// 3-D image (itkTIFFImageIO.cxx:799-805).
const FILETYPE_PAGE: u16 = 2;
/// `FILETYPE_MASK`.
const FILETYPE_MASK: u32 = 4;

/// `1024 * 1024`, the strip size `InternalWrite` targets in place of libtiff's
/// own `STRIP_SIZE_DEFAULT` of 8 kiB (itkTIFFImageIO.cxx:759-790).
const TARGET_STRIP_BYTES: u64 = 1024 * 1024;

/// `2 * oneGibiByte`, above which `InternalWrite` opens the file in `"w8"`
/// (BigTIFF) mode (itkTIFFImageIO.cxx:619-633).
const BIG_TIFF_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;

fn unsupported<T>(message: impl Into<String>) -> Result<T> {
    Err(IoError::UnsupportedTiffFeature(message.into()))
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// The subset of one TIFF directory's tags that `TIFFReaderInternal::Initialize`
/// caches (itkTIFFReaderInternal.cxx:190-274), plus the tiling flag it derives
/// from `TIFFIsTiled`.
///
/// The defaults are `TIFFGetFieldDefaulted`'s: orientation `1`, samples per
/// pixel `1`, bits per sample `1`, planar configuration `1`, sample format `1`.
/// `photometric` is `None` when the tag is absent, which is exactly upstream's
/// `m_HasValidPhotometricInterpretation == false`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PageTags {
    width: u32,
    height: u32,
    photometric: Option<u16>,
    bits_per_sample: u16,
    samples_per_pixel: u16,
    sample_format: u16,
    planar_config: u16,
    orientation: u16,
    tiled: bool,
}

/// Everything `Initialize` learns about the file as a whole, read off directory
/// 0 except for the page counts.
struct FileTags {
    page0: PageTags,
    number_of_pages: usize,
    /// Directories whose `SUBFILETYPE` is exactly `0`.
    sub_files: usize,
    /// Directories flagged `FILETYPE_REDUCEDIMAGE` or `FILETYPE_MASK`.
    ignored_sub_files: usize,
    /// `SUBFILETYPE` per directory, `None` when the tag is absent.
    subfile_types: Vec<Option<u32>>,
    x_resolution: f64,
    y_resolution: f64,
    resolution_unit: u16,
}

type TiffDecoder = Decoder<BufReader<File>>;

fn open_decoder(path: &Path) -> Result<TiffDecoder> {
    Ok(Decoder::new(BufReader::new(File::open(path)?))?)
}

/// `TIFFFetchRational`: libtiff divides the two `uint32`s in `double` and
/// stores the result in the `float` field `td_xresolution`, which
/// `ReadImageInformation` then widens back to `double`
/// (itkTIFFImageIO.cxx:379).
fn rational_tag(decoder: &mut TiffDecoder, tag: Tag) -> Result<Option<f64>> {
    use tiff::decoder::ifd::Value;
    Ok(match decoder.find_tag(tag)? {
        Some(Value::Rational(n, d)) if d != 0 => Some(f64::from((n as f64 / d as f64) as f32)),
        _ => None,
    })
}

fn unsigned_tag(decoder: &mut TiffDecoder, tag: Tag, default: u16) -> Result<u16> {
    Ok(decoder
        .find_tag_unsigned_vec::<u16>(tag)?
        .and_then(|v| v.first().copied())
        .unwrap_or(default))
}

fn read_page_tags(decoder: &mut TiffDecoder) -> Result<PageTags> {
    let (width, height) = decoder.dimensions()?;
    Ok(PageTags {
        width,
        height,
        photometric: decoder.find_tag_unsigned::<u16>(Tag::PhotometricInterpretation)?,
        bits_per_sample: unsigned_tag(decoder, Tag::BitsPerSample, 1)?,
        samples_per_pixel: unsigned_tag(decoder, Tag::SamplesPerPixel, 1)?,
        sample_format: unsigned_tag(decoder, Tag::SampleFormat, 1)?,
        planar_config: unsigned_tag(decoder, Tag::PlanarConfiguration, PLANARCONFIG_CONTIG)?,
        orientation: unsigned_tag(decoder, Tag::Orientation, ORIENTATION_TOPLEFT)?,
        tiled: decoder.find_tag(Tag::TileWidth)?.is_some(),
    })
}

/// `TIFFReaderInternal::Initialize` (itkTIFFReaderInternal.cxx:190-274).
///
/// The `SUBFILETYPE` census runs only when there is more than one directory,
/// exactly as upstream's `if (this->m_NumberOfPages > 1)` gate does — so a
/// single-page file always reports `sub_files == ignored_sub_files == 0`.
fn read_file_tags(decoder: &mut TiffDecoder) -> Result<FileTags> {
    let page0 = read_page_tags(decoder)?;
    let x_resolution = rational_tag(decoder, Tag::XResolution)?.unwrap_or(1.0);
    let y_resolution = rational_tag(decoder, Tag::YResolution)?.unwrap_or(1.0);
    let resolution_unit = unsigned_tag(decoder, Tag::ResolutionUnit, 1)?;

    let mut subfile_types = vec![decoder.find_tag_unsigned::<u32>(Tag::NewSubfileType)?];
    while decoder.more_images() {
        decoder.next_image()?;
        subfile_types.push(decoder.find_tag_unsigned::<u32>(Tag::NewSubfileType)?);
    }
    let number_of_pages = subfile_types.len();

    let (mut sub_files, mut ignored_sub_files) = (0, 0);
    if number_of_pages > 1 {
        for subfile_type in subfile_types.iter().flatten() {
            if *subfile_type == 0 {
                sub_files += 1;
            } else if subfile_type & FILETYPE_REDUCEDIMAGE != 0 || subfile_type & FILETYPE_MASK != 0
            {
                ignored_sub_files += 1;
            }
        }
    }

    Ok(FileTags {
        page0,
        number_of_pages,
        sub_files,
        ignored_sub_files,
        subfile_types,
        x_resolution,
        y_resolution,
        resolution_unit,
    })
}

/// The two `GetFormat` results this module can reach: `PALETTE_RGB` and
/// `PALETTE_GRAYSCALE` need the colour map the `tiff` crate will not decode,
/// and `OTHER` is upstream's RGBA fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    /// `TIFFImageIO::GRAYSCALE` — `MINISWHITE` or `MINISBLACK`.
    Grayscale,
    /// `TIFFImageIO::RGB_` — `PHOTOMETRIC_RGB`. (`YCBCR` also maps here
    /// upstream, but `CanRead()` excludes it, so it can only be reached through
    /// the RGBA fallback.)
    Rgb,
}

/// What [`read_information`] and [`read`] agree on before a single pixel moves.
struct Layout {
    /// `ReadImageInformation`'s `SetNumberOfComponents`, which is also
    /// `ReadGenericImage`'s `inc` for both reachable formats.
    number_of_components: usize,
    /// `m_ComponentType`, as the `BitsPerSample`/`SampleFormat` ladder maps it
    /// (itkTIFFImageIO.cxx:395-431).
    component: PixelId,
    /// `m_PixelType`, once `GetPixelIDFromImageIO` has folded it in.
    pixel_id: PixelId,
    dimension: usize,
    size: Vec<usize>,
    spacing: Vec<f64>,
}

/// `ReadImageInformation`'s component-type ladder (itkTIFFImageIO.cxx:395-431).
///
/// The `BitsPerSample == 32` arm has no `default:` — a `SampleFormat` of `4`
/// (`SAMPLEFORMAT_VOID`) or an unrecognised one leaves `m_ComponentType` at
/// whatever the previous read, or the constructor's `UCHAR`, left there, and
/// `ReadGenericImage<unsigned char>` then copies one byte per four-byte sample.
/// That stale-state bug (ledger §1.68) has no analogue here — this function is
/// pure — so the case is refused instead.
fn component_type(bits_per_sample: u16, sample_format: u16) -> Result<PixelId> {
    Ok(match (bits_per_sample, sample_format) {
        (bits, 2) if bits <= 8 => PixelId::Int8,
        (bits, _) if bits <= 8 => PixelId::UInt8,
        (32, 1) => PixelId::UInt32,
        (32, 2) => PixelId::Int32,
        (32, 3) => PixelId::Float32,
        (32, other) => {
            return unsupported(format!(
                "TIFF SampleFormat {other} at 32 bits per sample leaves m_ComponentType \
                 unassigned upstream (itkTIFFImageIO.cxx:406-420) — doc/upstream-findings.md §1.68"
            ));
        }
        (_, 2) => PixelId::Int16,
        _ => PixelId::UInt16,
    })
}

/// Whether the `tiff` crate's `invert_colors` (decoder/mod.rs:713-769) can
/// invert a `WhiteIsZero` page that decodes as `color` with the raw
/// `SampleFormat` tag `sample_format` (1 = Uint, 2 = Int, 3 = IEEEFP). Its match
/// arms accept only `Gray(1|2|4|8|16|32|64)` with `Uint` and `Gray(32|64)` with
/// `IEEEFP`; everything else — `Multiband`, signed-integer `Gray`, non-32/64-bit
/// float `Gray` — falls to the `_` arm and returns `UnknownInterpretation`.
/// Mirrored exactly so `layout_for` refuses at `read_information` time whatever
/// `read` would fail to invert.
fn crate_can_invert_whiteiszero(color: &ColorType, sample_format: u16) -> bool {
    match *color {
        ColorType::Gray(bits) => match sample_format {
            2 => false,                   // SampleFormat::Int
            3 => matches!(bits, 32 | 64), // SampleFormat::IEEEFP
            // SampleFormat::Uint / default: invert_colors accepts only the
            // conformant bit depths; an odd width (e.g. Gray(24)) hits its `_`.
            _ => matches!(bits, 1 | 2 | 4 | 8 | 16 | 32 | 64),
        },
        _ => false, // Multiband and every non-gray interpretation
    }
}

/// `TIFFReaderInternal::CanRead` (itkTIFFReaderInternal.cxx:276-290) narrowed to
/// what the `tiff` crate can decode, plus `ReadImageInformation`'s pixel-type
/// assignment (itkTIFFImageIO.cxx:433-491).
fn layout_for(decoder: &mut TiffDecoder, tags: &FileTags) -> Result<Layout> {
    let page0 = &tags.page0;

    if page0.width == 0 || page0.height == 0 || page0.samples_per_pixel == 0 {
        return unsupported("TIFF image has a zero width, height or SamplesPerPixel");
    }
    if page0.tiled {
        return unsupported(
            "tiled TIFF images are read through TIFFReadRGBAImageOriented upstream \
             (itkTIFFReaderInternal.cxx:281) — doc/upstream-findings.md §4.102",
        );
    }
    if page0.planar_config != PLANARCONFIG_CONTIG && page0.samples_per_pixel != 1 {
        return unsupported(
            "PLANARCONFIG_SEPARATE with more than one sample per pixel is read through \
             TIFFReadRGBAImageOriented upstream (itkTIFFReaderInternal.cxx:287) — \
             doc/upstream-findings.md §4.102",
        );
    }
    if page0.orientation != ORIENTATION_TOPLEFT && page0.orientation != ORIENTATION_BOTLEFT {
        return unsupported(format!(
            "TIFF Orientation {} is read through TIFFReadRGBAImageOriented upstream \
             (itkTIFFReaderInternal.cxx:288) — doc/upstream-findings.md §4.102",
            page0.orientation
        ));
    }
    if !matches!(page0.bits_per_sample, 8 | 16 | 32) {
        return unsupported(format!(
            "TIFF BitsPerSample {} is read through TIFFReadRGBAImageOriented upstream \
             (itkTIFFReaderInternal.cxx:289) — doc/upstream-findings.md §4.102",
            page0.bits_per_sample
        ));
    }

    let photometric = match page0
        .photometric
        .and_then(PhotometricInterpretation::from_u16)
    {
        Some(p) => p,
        None => {
            return unsupported(
                "a TIFF with no or an unrecognised PhotometricInterpretation is read through \
                 TIFFReadRGBAImageOriented upstream (itkTIFFReaderInternal.cxx:283) — \
                 doc/upstream-findings.md §4.102",
            );
        }
    };

    let format = match photometric {
        PhotometricInterpretation::RGB => Format::Rgb,
        PhotometricInterpretation::BlackIsZero | PhotometricInterpretation::WhiteIsZero => {
            Format::Grayscale
        }
        PhotometricInterpretation::RGBPalette => {
            return unsupported(
                "the `tiff` crate cannot decode a palette TIFF: colortype() rejects \
                 PhotometricInterpretation::RGBPalette and every read path goes through it \
                 — doc/upstream-findings.md §4.100",
            );
        }
        other => {
            return unsupported(format!(
                "PhotometricInterpretation::{other:?} is read through TIFFReadRGBAImageOriented \
                 upstream (itkTIFFReaderInternal.cxx:284-286) — doc/upstream-findings.md §4.102"
            ));
        }
    };

    let component = component_type(page0.bits_per_sample, page0.sample_format)?;

    // `GetFormat() == RGB_` takes `SetNumberOfComponents(m_SamplesPerPixel)` and
    // copies `m_SamplesPerPixel * width` components per row, whatever that count
    // is, because the scanline path never interprets the channels. The `tiff`
    // crate names a `ColorType` first and decodes through it, so it can read
    // back only the counts its `ColorType` enumerates: an `RGB` page whose
    // `SamplesPerPixel` is 2 (which `InternalWrite` itself emits — see below) or
    // 5 has no name and cannot be decoded, and one whose count *is* named must
    // still agree, or the crate would silently drop the samples upstream keeps.
    let number_of_components = match format {
        // Fixed §2.140: upstream's `GetFormat` maps `MINISBLACK`/`MINISWHITE`
        // to `GRAYSCALE` before ever consulting `SamplesPerPixel`, so a
        // multi-sample grayscale page reads back as 1 component — dropping
        // every sample past the first, silently. The `tiff` crate's own
        // `colortype()` decodes a multi-sample `BlackIsZero`/`WhiteIsZero`
        // page as `Multiband { num_samples: SamplesPerPixel, .. }`
        // (decoder/image.rs:460-471) — every sample, verbatim — so the true
        // count is simply the page's own `SamplesPerPixel`.
        //
        // Decodability caveat, symmetric with the `Rgb` arm's `colortype()`
        // guard: `read` decodes through `decoder.read_image()`, which for a
        // `WhiteIsZero` page inverts every sample via the crate's
        // `invert_colors`. That table handles only single-sample `Gray` with an
        // unsigned-integer sample at a conformant bit depth (1/2/4/8/16/32/64)
        // or a 32/64-bit IEEE-float sample (decoder/mod.rs:713-769). A
        // multi-sample page (`ColorType::Multiband`), a signed-integer /
        // non-32-bit-float single-sample page, or an odd-width `Gray` is a shape
        // the table omits, so `invert_colors` returns `UnknownInterpretation`
        // and `read` errors. Refuse it here so `read_information` does not
        // advertise components `read` cannot deliver. `BlackIsZero` is never
        // inverted, so a multi-sample `MINISBLACK` page still reads as a
        // vector image.
        Format::Grayscale => {
            let stored = usize::from(page0.samples_per_pixel);
            if photometric == PhotometricInterpretation::WhiteIsZero {
                let color: ColorType = decoder.colortype().map_err(|source| {
                    IoError::UnsupportedTiffFeature(format!(
                        "the `tiff` crate cannot name a ColorType for this MINISWHITE TIFF \
                         ({source}) — doc/upstream-findings.md §2.140"
                    ))
                })?;
                if !crate_can_invert_whiteiszero(&color, page0.sample_format) {
                    return unsupported(format!(
                        "a MINISWHITE (WhiteIsZero) TIFF that decodes as {color:?} with \
                         SampleFormat {} cannot be inverted by the `tiff` crate's invert_colors, \
                         which handles only single-sample Gray with an unsigned-integer or \
                         32/64-bit float sample, so read() errors with UnknownInterpretation \
                         (SamplesPerPixel = {stored}, decoder/mod.rs:713-769) — \
                         doc/upstream-findings.md §2.140",
                        page0.sample_format
                    ));
                }
            }
            stored
        }
        Format::Rgb => {
            let color: ColorType = decoder.colortype().map_err(|source| {
                IoError::UnsupportedTiffFeature(format!(
                    "the `tiff` crate cannot decode this RGB TIFF ({source}); upstream's scanline \
                     path reads any SamplesPerPixel, including the SamplesPerPixel = 2 files \
                     `InternalWrite` itself emits (itkTIFFImageIO.cxx:722-745, :441-444) — \
                     doc/upstream-findings.md §4.102"
                ))
            })?;
            let named = usize::from(color.num_samples());
            let stored = usize::from(page0.samples_per_pixel);
            if named != stored {
                return unsupported(format!(
                    "an RGB TIFF with SamplesPerPixel {stored} decodes as {named} samples in the \
                     `tiff` crate, which drops the extra samples upstream keeps \
                     (itkTIFFImageIO.cxx:441-444) — doc/upstream-findings.md §4.102"
                ));
            }
            stored
        }
    };

    // `ReadImageInformation`'s dimension/spacing block
    // (itkTIFFImageIO.cxx:352-393).
    let volume_slices = if tags.sub_files > 0 {
        tags.sub_files
    } else {
        tags.number_of_pages - tags.ignored_sub_files
    };
    let is_volume = tags.number_of_pages - tags.ignored_sub_files > 1;

    let (sx, sy) = match tags.resolution_unit {
        2 if tags.x_resolution > 0.0 && tags.y_resolution > 0.0 => {
            (25.4 / tags.x_resolution, 25.4 / tags.y_resolution)
        }
        3 if tags.x_resolution > 0.0 && tags.y_resolution > 0.0 => {
            (10.0 / tags.x_resolution, 10.0 / tags.y_resolution)
        }
        _ => (1.0, 1.0),
    };

    let (dimension, size, spacing) = if is_volume {
        (
            3,
            vec![page0.width as usize, page0.height as usize, volume_slices],
            vec![sx, sy, 1.0],
        )
    } else {
        (
            2,
            vec![page0.width as usize, page0.height as usize],
            vec![sx, sy],
        )
    };

    let pixel_id = if number_of_components == 1 {
        component
    } else {
        component.vector_id()
    };

    Ok(Layout {
        number_of_components,
        component,
        pixel_id,
        dimension,
        size,
        spacing,
    })
}

/// Read the header only, with no pixel data.
///
/// The meta-data dictionary is always empty — `ReadTIFFTags` needs libtiff's
/// field registry, which the `tiff` crate does not expose (§4.98).
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    let mut decoder = open_decoder(path)?;
    let tags = read_file_tags(&mut decoder)?;
    decoder.seek_to_image(0)?;
    let layout = layout_for(&mut decoder, &tags)?;

    Ok(ImageInformation {
        pixel_id: layout.pixel_id,
        dimension: layout.dimension,
        number_of_components: layout.number_of_components,
        size: layout.size,
        spacing: layout.spacing,
        origin: vec![0.0; layout.dimension],
        direction: identity(layout.dimension),
        metadata: BTreeMap::new(),
    })
}

/// Pull `height` rows of `width * components` samples out of a decoded page,
/// placing row `r` at `r` for `ORIENTATION_TOPLEFT` and at `height - 1 - r`
/// for `ORIENTATION_BOTLEFT` (itkTIFFImageIO.cxx:1381-1395).
///
/// Fixed §2.140: this used to take the source and destination stride
/// separately, because upstream's `GRAYSCALE` arm keeps only the first
/// component of a multi-sample scanline. Now that [`layout_for`] reports the
/// page's true `SamplesPerPixel` for `GRAYSCALE` too, `read_current_page`'s
/// per-page geometry check (§1.67) guarantees the two strides are always
/// equal, so there is only one `components` count and every sample is kept.
fn page_rows<T: Copy>(
    decoded: &[T],
    out: &mut [T],
    width: usize,
    height: usize,
    components: usize,
    top_left: bool,
) {
    let stride = width * components;
    for row in 0..height {
        let src = &decoded[row * stride..(row + 1) * stride];
        let dst_row = if top_left { row } else { height - 1 - row };
        out[dst_row * stride..(dst_row + 1) * stride].copy_from_slice(src);
    }
}

macro_rules! place_page {
    ($decoded:expr, $out:expr, $variant:ident, $offset:expr, $len:expr, $ctx:expr) => {
        match (&$decoded, &mut *$out) {
            (PixelBuffer::$variant(src), PixelBuffer::$variant(dst)) => {
                page_rows(
                    src,
                    &mut dst[$offset..$offset + $len],
                    $ctx.0,
                    $ctx.1,
                    $ctx.2,
                    $ctx.3,
                );
                true
            }
            _ => false,
        }
    };
}

/// `DecodingResult` → [`PixelBuffer`], checking that the crate handed back the
/// sample type upstream's `BitsPerSample`/`SampleFormat` ladder predicted.
fn decoded_to_buffer(decoded: DecodingResult, expected: PixelId) -> Result<PixelBuffer> {
    let buffer = match decoded {
        DecodingResult::U8(v) => PixelBuffer::UInt8(v),
        DecodingResult::I8(v) => PixelBuffer::Int8(v),
        DecodingResult::U16(v) => PixelBuffer::UInt16(v),
        DecodingResult::I16(v) => PixelBuffer::Int16(v),
        DecodingResult::U32(v) => PixelBuffer::UInt32(v),
        DecodingResult::I32(v) => PixelBuffer::Int32(v),
        DecodingResult::F32(v) => PixelBuffer::Float32(v),
        other => {
            return unsupported(format!(
                "the `tiff` crate decoded this image to a sample type TIFFImageIO has no \
                 component type for: {other:?}"
            ));
        }
    };
    if buffer.component_id() != expected {
        return unsupported(format!(
            "TIFF BitsPerSample/SampleFormat predict component type {:?} but the `tiff` crate \
             decoded {:?}",
            expected,
            buffer.component_id()
        ));
    }
    Ok(buffer)
}

/// `ReadCurrentPage` (itkTIFFImageIO.cxx:1259-1324) for the `CanRead()` branch,
/// with the per-page geometry check that closes §1.67.
fn read_current_page(
    decoder: &mut TiffDecoder,
    tags: &FileTags,
    layout: &Layout,
    out: &mut PixelBuffer,
    pixel_offset: usize,
) -> Result<()> {
    let page = read_page_tags(decoder)?;
    if page != tags.page0 {
        return unsupported(
            "a multi-page TIFF whose directories disagree in geometry or sample layout is read \
             out of bounds upstream: ReadVolume reuses directory 0's width, height, \
             SamplesPerPixel and BitsPerSample for every page \
             (itkTIFFReaderInternal.cxx:258-270, itkTIFFImageIO.cxx:1332-1339) — \
             doc/upstream-findings.md §1.67",
        );
    }

    let width = page.width as usize;
    let height = page.height as usize;
    let components = layout.number_of_components;
    let top_left = page.orientation == ORIENTATION_TOPLEFT;
    let page_len = width * height * components;

    if pixel_offset
        .checked_add(page_len)
        .is_none_or(|end| end > out.len())
    {
        return unsupported(format!(
            "ReadVolume offsets page {} by its directory index rather than by a running count of \
             the pages it keeps, so this file overflows its {}-slice buffer upstream \
             (itkTIFFImageIO.cxx:170) — doc/upstream-findings.md §1.66",
            pixel_offset / page_len.max(1),
            layout.size.get(2).copied().unwrap_or(1)
        ));
    }

    let decoded = decoded_to_buffer(decoder.read_image()?, layout.component)?;

    let ctx = (width, height, components, top_left);
    let placed = place_page!(decoded, out, UInt8, pixel_offset, page_len, ctx)
        || place_page!(decoded, out, Int8, pixel_offset, page_len, ctx)
        || place_page!(decoded, out, UInt16, pixel_offset, page_len, ctx)
        || place_page!(decoded, out, Int16, pixel_offset, page_len, ctx)
        || place_page!(decoded, out, UInt32, pixel_offset, page_len, ctx)
        || place_page!(decoded, out, Int32, pixel_offset, page_len, ctx)
        || place_page!(decoded, out, Float32, pixel_offset, page_len, ctx);
    debug_assert!(
        placed,
        "decoded_to_buffer already matched the component type"
    );

    Ok(())
}

/// Read a `.tif` / `.tiff` file.
pub fn read(path: &Path) -> Result<Image> {
    let mut decoder = open_decoder(path)?;
    let tags = read_file_tags(&mut decoder)?;
    decoder.seek_to_image(0)?;
    let layout = layout_for(&mut decoder, &tags)?;

    let total = layout.size.iter().product::<usize>() * layout.number_of_components;
    let mut buffer = PixelBuffer::zeroed(layout.component, total);

    if layout.dimension > 2 {
        // `ReadVolume` (itkTIFFImageIO.cxx:147-176).
        let page_pixels = layout.size[0] * layout.size[1] * layout.number_of_components;
        for page in 0..tags.number_of_pages {
            if tags.ignored_sub_files > 0
                && let Some(subfile_type) = tags.subfile_types[page]
                && (subfile_type & FILETYPE_REDUCEDIMAGE != 0 || subfile_type & FILETYPE_MASK != 0)
            {
                continue;
            }
            decoder.seek_to_image(page)?;
            read_current_page(
                &mut decoder,
                &tags,
                &layout,
                &mut buffer,
                page_pixels * page,
            )?;
        }
    } else {
        read_current_page(&mut decoder, &tags, &layout, &mut buffer, 0)?;
    }

    let origin = vec![0.0; layout.dimension];
    let direction = identity(layout.dimension);
    Ok(if layout.number_of_components > 1 {
        Image::from_parts_vector(
            buffer,
            layout.number_of_components,
            layout.size,
            layout.spacing,
            origin,
            direction,
        )?
    } else {
        Image::from_parts(buffer, layout.size, layout.spacing, origin, direction)?
    })
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// `PHOTOMETRIC_MINISBLACK`, the photometric `InternalWrite` picks for a
/// 1-component image once `GetWritePalette()` is ruled out
/// (itkTIFFImageIO.cxx:725-738).
///
/// Fixed §2.141: also used for a 2-component image. Upstream writes
/// `PHOTOMETRIC_RGB` with `SAMPLESPERPIXEL = 2` for that case — a file no
/// TIFF reader can interpret as colour, and which this crate's own reader
/// refuses outright (§4.102). A grayscale-plus-one-extra-sample image is a
/// standard, unambiguous TIFF construct and is exactly what [`layout_for`]
/// now reads back correctly (§2.140), so `write_pages` reaches this type
/// through its generic `ImageEncoder::extra_samples` extension the same way
/// it already does for [`Rgb`] beyond 3 components.
struct MinIsBlack<T>(PhantomData<T>);

/// `PHOTOMETRIC_RGB` over `N` declared samples.
///
/// `InternalWrite` writes `PHOTOMETRIC_RGB` for every image with more than
/// two components, `SAMPLESPERPIXEL = scomponents`, and — for
/// `scomponents > 3` — an `EXTRASAMPLES` array (`:678-690`, `:739-746`).
/// `N == 3` covers 3 components and, with `ImageEncoder::extra_samples`,
/// everything above.
struct Rgb<T, const N: usize>(PhantomData<T>);

macro_rules! encoder_color_types {
    ($inner:ty, $bits:literal, $format:expr) => {
        impl EncoderColorType for MinIsBlack<$inner> {
            type Inner = $inner;
            const TIFF_VALUE: PhotometricInterpretation = PhotometricInterpretation::BlackIsZero;
            const BITS_PER_SAMPLE: &'static [u16] = &[$bits];
            const SAMPLE_FORMAT: &'static [SampleFormat] = &[$format];
            fn horizontal_predict(_row: &[$inner], _result: &mut Vec<$inner>) {
                unreachable!("InternalWrite only ever sets PREDICTOR_NONE")
            }
        }

        impl<const N: usize> EncoderColorType for Rgb<$inner, N> {
            type Inner = $inner;
            const TIFF_VALUE: PhotometricInterpretation = PhotometricInterpretation::RGB;
            const BITS_PER_SAMPLE: &'static [u16] = &[$bits; N];
            const SAMPLE_FORMAT: &'static [SampleFormat] = &[$format; N];
            fn horizontal_predict(_row: &[$inner], _result: &mut Vec<$inner>) {
                unreachable!("InternalWrite only ever sets PREDICTOR_NONE")
            }
        }
    };
}

// `InternalWrite`'s `bps` switch, which throws for anything else
// (itkTIFFImageIO.cxx:594-613), crossed with its `SAMPLEFORMAT` assignment
// (`:643-650`): `SHORT`/`SCHAR` are `SAMPLEFORMAT_INT`, `FLOAT` is
// `SAMPLEFORMAT_IEEEFP`, and `UCHAR`/`USHORT` get no tag, i.e. the default
// `SAMPLEFORMAT_UINT`.
encoder_color_types!(u8, 8, SampleFormat::Uint);
encoder_color_types!(i8, 8, SampleFormat::Int);
encoder_color_types!(u16, 16, SampleFormat::Uint);
encoder_color_types!(i16, 16, SampleFormat::Int);
encoder_color_types!(f32, 32, SampleFormat::IEEEFP);

/// `ToRationalEuclideanGCD` (libtiff `tif_dirwrite.c:2674-2795`), the continued-
/// fraction approximation `TIFFSetField(TIFFTAG_XRESOLUTION, double)` funnels
/// into. Only the unsigned, `blnUseSignedRange == FALSE` half is ported —
/// `TIFFTAG_XRESOLUTION` is a `TIFF_RATIONAL`, never an `SRATIONAL`.
fn to_rational_euclidean_gcd(mut value: f64, use_small_range: bool) -> (u64, u64) {
    let n_max: u64 = if use_small_range {
        (2147483647u64 - 1) / 2
    } else {
        (9223372036854775807u64 - 1) / 2
    };
    let f_max = n_max as f64;
    let max_denom: u64 = 0xFFFF_FFFF;
    let return_limit = max_denom;

    let mut num_sum = [0u64, 1, 0];
    let mut denom_sum = [1u64, 0, 0];

    let mut big_denom: u64 = 1;
    while value != value.floor() && value < f_max && big_denom < n_max {
        big_denom <<= 1;
        value *= 2.0;
    }
    let mut big_num = value as u64;

    let mut i = 0;
    while i < 64 {
        if big_denom == 0 {
            break;
        }
        let val = big_num / big_denom;

        let aux = big_num;
        big_num = big_denom;
        big_denom = aux % big_denom;

        let mut aux = val;
        if denom_sum[1] * val + denom_sum[0] >= max_denom {
            aux = (max_denom - denom_sum[0]) / denom_sum[1];
            if aux * 2 >= val || denom_sum[1] >= max_denom {
                // "exit but execute rest of for-loop"
                i = 64 + 1;
            } else {
                break;
            }
        }
        num_sum[2] = aux * num_sum[1] + num_sum[0];
        num_sum[0] = num_sum[1];
        num_sum[1] = num_sum[2];
        denom_sum[2] = aux * denom_sum[1] + denom_sum[0];
        denom_sum[0] = denom_sum[1];
        denom_sum[1] = denom_sum[2];

        i += 1;
    }

    while num_sum[1] > return_limit || denom_sum[1] > return_limit {
        num_sum[1] /= 2;
        denom_sum[1] /= 2;
    }
    (num_sum[1], denom_sum[1])
}

/// `DoubleToRational` (libtiff `tif_dirwrite.c:2802-2871`). `value` is what
/// libtiff stores in the `float` field `td_xresolution`, widened back to
/// `double`, so callers pass `f64::from(resolution as f32)`.
fn double_to_rational(value: f64) -> Rational {
    // libtiff writes this as `if (!(value >= 0.0))`, whose point is to catch a
    // NaN alongside the negatives; spelled out, because `!(a >= b)` on a
    // partially ordered type is exactly what `clippy::neg_cmp_op_on_partial_ord`
    // asks to see written explicitly.
    if value.is_nan() || value < 0.0 {
        return Rational { n: 0, d: 0 };
    }
    if value > f64::from(u32::MAX) {
        return Rational { n: u32::MAX, d: 0 };
    }
    if value == f64::from(value as u32) {
        return Rational {
            n: value as u32,
            d: 1,
        };
    }
    if value < 1.0 / f64::from(u32::MAX) {
        return Rational { n: 0, d: u32::MAX };
    }

    let (n1, d1) = to_rational_euclidean_gcd(value, false);
    let (n2, d2) = to_rational_euclidean_gcd(value, true);
    let diff1 = (value - (n1 as f64 / d1 as f64)).abs();
    let diff2 = (value - (n2 as f64 / d2 as f64)).abs();
    if diff1 < diff2 {
        Rational {
            n: n1 as u32,
            d: d1 as u32,
        }
    } else {
        Rational {
            n: n2 as u32,
            d: d2 as u32,
        }
    }
}

/// `rowsperstrip = 1 MiB / TIFFScanlineSize64(tif)`, floored at one
/// (itkTIFFImageIO.cxx:775-790). `TIFFDefaultStripSize` returns any `s >= 1`
/// unchanged (libtiff `tif_strip.c:222-245`), so the clamp is the whole of it.
fn rows_per_strip(width: usize, components: usize, bits_per_sample: u16) -> u32 {
    let scanline = (width as u64 * components as u64 * u64::from(bits_per_sample)).div_ceil(8);
    if scanline == 0 {
        return 1;
    }
    u32::try_from(TARGET_STRIP_BYTES / scanline)
        .unwrap_or(u32::MAX)
        .max(1)
}

/// The per-page constants `InternalWrite` hoists out of its page loop.
struct WriteContext {
    width: u32,
    height: u32,
    pages: u16,
    components: usize,
    bits_per_sample: u16,
    resolution: Option<(Rational, Rational)>,
    compression: Compression,
    is_volume: bool,
}

fn write_pages<W, K, C>(
    encoder: &mut TiffEncoder<W, K>,
    data: &[C::Inner],
    ctx: &WriteContext,
) -> Result<()>
where
    W: Write + Seek,
    K: TiffKind,
    C: EncoderColorType,
    [C::Inner]: TiffValue,
{
    let page_len = ctx.width as usize * ctx.height as usize * ctx.components;
    let strip_rows = rows_per_strip(ctx.width as usize, ctx.components, ctx.bits_per_sample);

    for page in 0..ctx.pages {
        let mut image = encoder.new_image::<C>(ctx.width, ctx.height)?;

        // `C::BITS_PER_SAMPLE.len()` is the sample count the photometric
        // interpretation declares on its own (1 for `MinIsBlack`, 3 for
        // `Rgb`); anything past that must be described via `ExtraSamples`
        // per the TIFF6 spec, which `extra_samples` folds into
        // `SamplesPerPixel` and `BitsPerSample`. Upstream's own extension
        // is RGB-only — `if (scomponents > 3)`, one associated-alpha sample
        // then unspecified (itkTIFFImageIO.cxx:678-690) — so a 2-component
        // `MinIsBlack` image's one extra sample (fixed §2.141) has no
        // alpha semantics to assume and is left `Unspecified`.
        let core_samples = C::BITS_PER_SAMPLE.len();
        if ctx.components > core_samples {
            let mut extras = vec![ExtraSamples::Unspecified; ctx.components - core_samples];
            if C::TIFF_VALUE == PhotometricInterpretation::RGB {
                extras[0] = ExtraSamples::AssociatedAlpha;
            }
            image.extra_samples(&extras)?;
        }

        image.rows_per_strip(strip_rows)?;

        if let Some((x, y)) = &ctx.resolution {
            image.resolution_unit(ResolutionUnit::Inch);
            image.x_resolution(x.clone());
            image.y_resolution(y.clone());
        }

        let directory = image.encoder();
        directory.write_tag(Tag::Orientation, ORIENTATION_TOPLEFT)?;
        directory.write_tag(Tag::PlanarConfiguration, PLANARCONFIG_CONTIG)?;
        directory.write_tag(Tag::Software, "InsightToolkit")?;
        if ctx.is_volume {
            directory.write_tag(Tag::NewSubfileType, u32::from(FILETYPE_PAGE))?;
            directory.write_tag(TAG_PAGE_NUMBER, &[page, ctx.pages][..])?;
        }

        let start = usize::from(page) * page_len;
        image.write_data(&data[start..start + page_len])?;
    }
    Ok(())
}

fn write_typed<W, K>(
    encoder: &mut TiffEncoder<W, K>,
    buffer: &PixelBuffer,
    ctx: &WriteContext,
) -> Result<()>
where
    W: Write + Seek,
    K: TiffKind,
{
    macro_rules! by_components {
        ($data:expr, $inner:ty) => {
            match ctx.components {
                1 | 2 => write_pages::<_, _, MinIsBlack<$inner>>(encoder, $data, ctx),
                _ => write_pages::<_, _, Rgb<$inner, 3>>(encoder, $data, ctx),
            }
        };
    }
    match buffer {
        PixelBuffer::UInt8(v) => by_components!(v, u8),
        PixelBuffer::Int8(v) => by_components!(v, i8),
        PixelBuffer::UInt16(v) => by_components!(v, u16),
        PixelBuffer::Int16(v) => by_components!(v, i16),
        PixelBuffer::Float32(v) => by_components!(v, f32),
        other => unsupported(format!(
            "TIFF supports unsigned/signed char, unsigned/signed short, and float, not {} \
             (itkTIFFImageIO.cxx:594-613)",
            other.component_id().as_str()
        )),
    }
}

/// The TIFF compressor to write with — the subset of
/// `TIFFImageIO::TIFFCompressionTypes` (itkTIFFImageIO.h:113-121) the `tiff`
/// crate's encoder can actually write. Selected through
/// [`WriteOptions::compressor`]; `None` keeps the previous behaviour, where
/// [`WriteOptions::use_compression`] alone toggles `PackBits` on or off.
///
/// **Fixed §3.51** — `JPEG` is not offered: the `tiff` crate's encoder has no
/// JPEG codec (`encoder::compression` lists only `Uncompressed`/`Lzw`/
/// `Deflate`/`Packbits`), and there is no pure-Rust path to it. Upstream's
/// legacy tag-`32946` `Deflate` is not offered either: the crate's own
/// `Compression::Deflate` always writes tag `8`
/// (`PhotometricInterpretation`-style — upstream's `AdobeDeflate`), the only
/// Deflate tag its encoder can produce; both upstream names decode to the
/// same zlib stream, so this port's [`TiffCompressor::Deflate`] covers what
/// upstream calls `DEFLATE` and `ADOBEDEFLATE` alike.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TiffCompressor {
    /// `COMPRESSION_LZW`.
    Lzw,
    /// `COMPRESSION_ADOBE_DEFLATE` (tag `8`). Ratio set by
    /// [`WriteOptions::compression_level`] through [`deflate_level_for`] —
    /// fixed §3.52.
    Deflate,
    /// `COMPRESSION_PACKBITS` — the same compressor a bare
    /// [`WriteOptions::use_compression`] already selects.
    PackBits,
}

/// Fixed §3.52: partitions [`WriteOptions::resolved_level`]'s clamped `1..=9`
/// (`itkImageIOBase.h:288`'s `1..=GetMaximumCompressionLevel()`) into the
/// three tiers [`DeflateLevel`] exposes, each discriminant sitting at the
/// middle of its own bucket (`Fast = 1`, `Balanced = 6`, `Best = 9`).
///
/// Upstream never mapped `CompressionLevel` to a Deflate ratio at all — on a
/// TIFF the field is JPEG quality, full stop (`itkTIFFImageIO.h:171-179`,
/// `:749`), and this port cannot write JPEG (§3.51). Rather than leave the
/// level permanently dead now that [`TiffCompressor::Deflate`] exists, it
/// drives the ratio of the one compressor this port can actually vary.
fn deflate_level_for(level: i32) -> DeflateLevel {
    match level {
        ..=3 => DeflateLevel::Fast,
        4..=6 => DeflateLevel::Balanced,
        _ => DeflateLevel::Best,
    }
}

/// Write a `.tif` / `.tiff` file.
///
/// `WriteImageInformation` is a no-op upstream (itkTIFFImageIO.cxx:555-557);
/// `Write` calls `InternalWrite`, which emits header and pixels in one pass.
pub fn write(image: &Image, path: &Path, options: &WriteOptions) -> Result<()> {
    if image.dimension() != 2 && image.dimension() != 3 {
        return unsupported(format!(
            "TIFF Writer can only write 2-d or 3-d images, not {}-d \
             (itkTIFFImageIO.cxx:559-570)",
            image.dimension()
        ));
    }

    let size = image.size();
    let width = u32::try_from(size[0]).map_err(|_| {
        IoError::UnsupportedTiffFeature("TIFF image width exceeds uint32".to_string())
    })?;
    let height = u32::try_from(size[1]).map_err(|_| {
        IoError::UnsupportedTiffFeature("TIFF image height exceeds uint32".to_string())
    })?;
    let pages = if image.dimension() == 3 {
        u16::try_from(size[2]).map_err(|_| {
            IoError::UnsupportedTiffFeature(
                "InternalWrite truncates the page count to uint16 (itkTIFFImageIO.cxx:583)"
                    .to_string(),
            )
        })?
    } else {
        1
    };

    let components = image.buffer_stride();
    let component = image.buffer().component_id();
    let bits_per_sample = match component {
        PixelId::UInt8 | PixelId::Int8 => 8,
        PixelId::UInt16 | PixelId::Int16 => 16,
        PixelId::Float32 => 32,
        other => {
            return unsupported(format!(
                "TIFF supports unsigned/signed char, unsigned/signed short, and float, not {} \
                 (itkTIFFImageIO.cxx:594-613)",
                other.as_str()
            ));
        }
    };

    // `resolution_x = m_Spacing[0] != 0.0 ? 25.4 / m_Spacing[0] : 0.0`, written
    // only when both are positive (itkTIFFImageIO.cxx:587-588, :792-797).
    let spacing = image.spacing();
    let resolution_of = |s: f64| if s != 0.0 { 25.4 / s } else { 0.0 };
    let (rx, ry) = (resolution_of(spacing[0]), resolution_of(spacing[1]));
    let resolution = (rx > 0.0 && ry > 0.0).then(|| {
        (
            double_to_rational(f64::from(rx as f32)),
            double_to_rational(f64::from(ry as f32)),
        )
    });

    // `if (m_UseCompression) { switch (m_Compression) {...} } else { compression
    // = COMPRESSION_NONE; }` (itkTIFFImageIO.cxx:692-716) — the toggle gates the
    // selector, not the other way around.
    let compression = if !options.use_compression {
        Compression::Uncompressed
    } else {
        match options.compressor {
            None | Some(TiffCompressor::PackBits) => Compression::Packbits,
            Some(TiffCompressor::Lzw) => Compression::Lzw,
            Some(TiffCompressor::Deflate) => {
                // `DeflateLevel::Balanced as u8`'s `6` is the crate's own
                // `Default`, used when `compression_level` is left at `-1`.
                Compression::Deflate(deflate_level_for(options.resolved_level(6)))
            }
        }
    };

    let ctx = WriteContext {
        width,
        height,
        pages,
        components,
        bits_per_sample,
        resolution,
        compression,
        is_volume: image.dimension() == 3,
    };

    let image_bytes =
        size.iter().product::<usize>() as u64 * components as u64 * u64::from(bits_per_sample / 8);
    let file = BufWriter::new(File::create(path)?);
    if image_bytes > BIG_TIFF_THRESHOLD {
        let mut encoder = TiffEncoder::new_big(file)?.with_compression(ctx.compression);
        write_typed(&mut encoder, image.buffer(), &ctx)
    } else {
        let mut encoder = TiffEncoder::new(file)?.with_compression(ctx.compression);
        write_typed(&mut encoder, image.buffer(), &ctx)
    }
}

/// `itk::TIFFImageIO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TiffImageIo;

impl ImageIo for TiffImageIo {
    fn name(&self) -> &'static str {
        "TIFFImageIO"
    }

    /// `.tif`, `.TIF`, `.tiff` and `.TIFF`, all four registered for both read
    /// and write (itkTIFFImageIO.cxx:225-231).
    fn supported_read_extensions(&self) -> &'static [&'static str] {
        &[".tif", ".TIF", ".tiff", ".TIFF"]
    }

    fn supported_write_extensions(&self) -> &'static [&'static str] {
        &[".tif", ".TIF", ".tiff", ".TIFF"]
    }

    /// `CanReadFile` is `m_InternalImage->Open(file, true)`
    /// (itkTIFFImageIO.cxx:30-50): the file opens as a TIFF and directory 0
    /// carries `IMAGEWIDTH` and `IMAGELENGTH`. No extension check, and — note —
    /// no `CanRead()` check, so a file this IO claims here may still fail in
    /// [`read`].
    fn can_read_file(&self, path: &Path) -> bool {
        open_decoder(path).is_ok()
    }

    /// `CanWriteFile` is `HasSupportedWriteExtension(name, false)`
    /// (itkTIFFImageIO.cxx:542-553) — case-**sensitive**, unlike the trait's
    /// case-insensitive default. `foo.TIFF` is claimed, `foo.Tiff` is not.
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

    // -- hand-built TIFF fixtures ------------------------------------------
    //
    // Every fixture below is assembled byte by byte — nothing comes from the
    // `tiff` crate's encoder — so the reader is tested against files of purely
    // upstream shape, including the tag combinations `TiffEncoder` will not
    // emit (a missing `XRESOLUTION`, `ORIENTATION_BOTLEFT`, a colour map).

    const TYPE_SHORT: u16 = 3;
    const TYPE_LONG: u16 = 4;
    const TYPE_RATIONAL: u16 = 5;

    const TAG_IMAGE_WIDTH: u16 = 256;
    const TAG_IMAGE_LENGTH: u16 = 257;
    const TAG_BITS_PER_SAMPLE: u16 = 258;
    const TAG_COMPRESSION: u16 = 259;
    const TAG_PHOTOMETRIC: u16 = 262;
    const TAG_STRIP_OFFSETS: u16 = 273;
    const TAG_ORIENTATION: u16 = 274;
    const TAG_SAMPLES_PER_PIXEL: u16 = 277;
    const TAG_ROWS_PER_STRIP: u16 = 278;
    const TAG_STRIP_BYTE_COUNTS: u16 = 279;
    const TAG_X_RESOLUTION: u16 = 282;
    const TAG_Y_RESOLUTION: u16 = 283;
    const TAG_RESOLUTION_UNIT: u16 = 296;
    const TAG_COLOR_MAP: u16 = 320;
    const TAG_SUBFILE_TYPE: u16 = 254;
    const TAG_SAMPLE_FORMAT: u16 = 339;

    fn short(v: u16) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn long(v: u32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    /// One IFD entry: `(tag, type, count, little-endian payload)`. A payload of
    /// four bytes or fewer is stored inline; anything longer is spilled after
    /// the directory, as the TIFF spec requires.
    type Entry = (u16, u16, u32, Vec<u8>);

    /// One directory's tags (`StripOffsets` / `StripByteCounts` excluded — the
    /// builder appends them) plus its uncompressed pixel bytes.
    struct Page {
        entries: Vec<Entry>,
        pixels: Vec<u8>,
    }

    /// A minimal, uncompressed, single-strip little-endian TIFF.
    fn build_tiff(pages: Vec<Page>) -> Vec<u8> {
        let mut out: Vec<u8> = b"II".to_vec();
        out.extend_from_slice(&42u16.to_le_bytes());
        let first_ifd_pointer = out.len();
        out.extend_from_slice(&0u32.to_le_bytes());

        // Pixel blobs first, so their offsets are known when the IFDs are laid
        // out.
        let mut strip_offsets = Vec::new();
        for page in &pages {
            strip_offsets.push(out.len() as u32);
            out.extend_from_slice(&page.pixels);
        }

        let mut previous_next_pointer = first_ifd_pointer;
        for (index, page) in pages.iter().enumerate() {
            let ifd_offset = out.len() as u32;
            out[previous_next_pointer..previous_next_pointer + 4]
                .copy_from_slice(&ifd_offset.to_le_bytes());

            let mut entries = page.entries.clone();
            entries.push((TAG_STRIP_OFFSETS, TYPE_LONG, 1, long(strip_offsets[index])));
            entries.push((
                TAG_STRIP_BYTE_COUNTS,
                TYPE_LONG,
                1,
                long(page.pixels.len() as u32),
            ));
            entries.sort_by_key(|e| e.0);

            let directory_bytes = 2 + 12 * entries.len() + 4;
            let mut external_offset = ifd_offset as usize + directory_bytes;
            let mut external = Vec::new();

            out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
            for (tag, ty, count, payload) in &entries {
                out.extend_from_slice(&tag.to_le_bytes());
                out.extend_from_slice(&ty.to_le_bytes());
                out.extend_from_slice(&count.to_le_bytes());
                if payload.len() <= 4 {
                    let mut inline = payload.clone();
                    inline.resize(4, 0);
                    out.extend_from_slice(&inline);
                } else {
                    out.extend_from_slice(&(external_offset as u32).to_le_bytes());
                    external_offset += payload.len();
                    external.extend_from_slice(payload);
                }
            }
            previous_next_pointer = out.len();
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&external);
        }
        out
    }

    /// The tags every fixture needs: an uncompressed, top-left, single-strip
    /// page of `width * height * samples` bytes.
    fn base_entries(
        width: u32,
        height: u32,
        bits: u16,
        samples: u16,
        photometric: u16,
    ) -> Vec<Entry> {
        let bits_payload: Vec<u8> = (0..samples).flat_map(|_| short(bits)).collect();
        vec![
            (TAG_IMAGE_WIDTH, TYPE_LONG, 1, long(width)),
            (TAG_IMAGE_LENGTH, TYPE_LONG, 1, long(height)),
            (
                TAG_BITS_PER_SAMPLE,
                TYPE_SHORT,
                u32::from(samples),
                bits_payload,
            ),
            (TAG_COMPRESSION, TYPE_SHORT, 1, short(1)),
            (TAG_PHOTOMETRIC, TYPE_SHORT, 1, short(photometric)),
            (TAG_SAMPLES_PER_PIXEL, TYPE_SHORT, 1, short(samples)),
            (TAG_ROWS_PER_STRIP, TYPE_LONG, 1, long(height)),
        ]
    }

    fn write_fixture(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }

    /// `PutGrayscale` is a plain `std::copy_n` — the `// check inverted`
    /// comment (itkTIFFImageIO.cxx:1400) never grew a body — so upstream reads
    /// a `PHOTOMETRIC_MINISWHITE` page out with its raw, tone-inverted samples.
    /// The `tiff` crate already inverts correctly; **fixed §2.139** — this
    /// port keeps that correction (`!x` for `u8`) instead of reversing it back
    /// to upstream's raw value.
    #[test]
    fn minis_white_reads_the_true_photometric_value_not_the_raw_sample() {
        let pixels = vec![0u8, 1, 254, 255];
        let bytes = build_tiff(vec![Page {
            entries: base_entries(2, 2, 8, 1, 0), // photometric 0 == MINISWHITE
            pixels,
        }]);
        let path = write_fixture("sitk_io_tiff_minis_white.tif", &bytes);
        let image = read(&path).expect("MINISWHITE reads");
        std::fs::remove_file(&path).ok();

        assert_eq!(image.pixel_id(), PixelId::UInt8);
        // `!raw`: 0 (white) inverts to 255 (max intensity), 255 (black) to 0.
        assert_eq!(image.buffer(), &PixelBuffer::UInt8(vec![255, 254, 1, 0]));
    }

    /// Fixed §2.139, 32-bit-float counterpart: keeping the crate's correction
    /// rather than undoing it removes the reason `layout_for` used to refuse a
    /// `WhiteIsZero` + `Float32` page (formerly ledger §4.101, whose rationale
    /// was that the crate's `1.0 - v` is not invertible under rounding) — that
    /// concern only applied to reversing the correction, not to keeping it.
    #[test]
    fn minis_white_float32_reads_the_crates_own_inverted_value() {
        let pixels: Vec<u8> = [0.0f32, 0.25, 0.75, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut entries = base_entries(2, 2, 32, 1, 0); // photometric 0 == MINISWHITE
        entries.push((TAG_SAMPLE_FORMAT, TYPE_SHORT, 1, short(3))); // IEEE float
        let bytes = build_tiff(vec![Page { entries, pixels }]);
        let path = write_fixture("sitk_io_tiff_minis_white_float32.tif", &bytes);
        let image = read(&path).expect("float32 MINISWHITE reads");
        std::fs::remove_file(&path).ok();

        assert_eq!(image.pixel_id(), PixelId::Float32);
        match image.buffer() {
            PixelBuffer::Float32(v) => assert_eq!(v, &[1.0, 0.75, 0.25, 0.0]),
            other => panic!("expected Float32, got {other:?}"),
        }
    }

    /// `GetFormat` calls a two-sample `MINISBLACK` page `GRAYSCALE` before ever
    /// consulting `SamplesPerPixel`, so upstream's `NumberOfComponents` is 1
    /// and `PutGrayscale` copies `width` components off a `2 * width`
    /// scanline — the interleaved samples of the row's first `width / 2`
    /// pixels, dropping the rest. Ledger §2.140. **Fixed** — every sample is
    /// silent data loss otherwise, not a shape to reproduce; this port reads
    /// the page as a 2-component vector image instead.
    #[test]
    fn two_sample_grayscale_reads_as_a_two_component_vector_image() {
        // Two rows of two pixels, two samples each: [g,a, g,a].
        let pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let bytes = build_tiff(vec![Page {
            entries: base_entries(2, 2, 8, 2, 1),
            pixels: pixels.clone(),
        }]);
        let path = write_fixture("sitk_io_tiff_gray_alpha.tif", &bytes);
        let image = read(&path).expect("two-sample grayscale reads");
        std::fs::remove_file(&path).ok();

        assert_eq!(image.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(image.number_of_components_per_pixel(), 2);
        assert_eq!(image.size(), &[2, 2]);
        assert_eq!(image.buffer(), &PixelBuffer::UInt8(pixels));
    }

    /// §2.140, `MINISWHITE` counterpart. Unlike `MINISBLACK`, a `WhiteIsZero`
    /// page is inverted sample-by-sample by the `tiff` crate's `invert_colors`,
    /// which handles only single-sample `Gray` with an unsigned-integer or
    /// 32/64-bit float sample (decoder/mod.rs:713-769). A multi-sample page
    /// (`ColorType::Multiband`) or a signed single-sample page
    /// (`Gray` + `SampleFormat::Int`) is a shape it omits, so `read` errors with
    /// `UnknownInterpretation`. `layout_for` now refuses both at
    /// `read_information` time too, so the header and the pixel read never
    /// disagree about whether the file is readable.
    #[test]
    fn minis_white_pages_the_crate_cannot_invert_are_refused_by_both_read_paths() {
        // Multi-sample MINISWHITE decodes as ColorType::Multiband.
        let multiband = build_tiff(vec![Page {
            entries: base_entries(2, 2, 8, 2, 0), // photometric 0 == MINISWHITE
            pixels: vec![1u8, 2, 3, 4, 5, 6, 7, 8],
        }]);
        let path = write_fixture("sitk_io_tiff_minis_white_multiband.tif", &multiband);
        let info = read_information(&path);
        let image = read(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&info, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("§2.140")),
            "read_information should refuse multiband MINISWHITE: {info:?}"
        );
        assert!(
            matches!(&image, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("§2.140")),
            "read should refuse multiband MINISWHITE: {image:?}"
        );

        // Signed single-sample MINISWHITE decodes as Gray + SampleFormat::Int.
        let mut entries = base_entries(2, 2, 8, 1, 0);
        entries.push((TAG_SAMPLE_FORMAT, TYPE_SHORT, 1, short(2))); // signed int
        let signed = build_tiff(vec![Page {
            entries,
            pixels: vec![1u8, 2, 3, 4],
        }]);
        let path = write_fixture("sitk_io_tiff_minis_white_signed.tif", &signed);
        let info = read_information(&path);
        let image = read(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&info, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("§2.140")),
            "read_information should refuse signed MINISWHITE: {info:?}"
        );
        assert!(
            matches!(&image, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("§2.140")),
            "read should refuse signed MINISWHITE: {image:?}"
        );
    }

    /// §2.140, odd-bit-depth `MINISWHITE` corner. `invert_colors` accepts a
    /// `Gray` + `Uint` sample only at the conformant depths 1/2/4/8/16/32/64
    /// (decoder/mod.rs:713-769); a `Gray(24)` page constructs fine in the crate
    /// (image.rs:252 rejects only zero/inconsistent bits) but errors at
    /// `invert_colors`' `_` arm. In this port such a page is refused earlier, by
    /// `layout_for`'s BitsPerSample guard (bits ∉ {8,16,32}, §4.102), for both
    /// `read_information` and `read` — so the two agree and never advertise a
    /// page the other rejects. `crate_can_invert_whiteiszero` is tightened to
    /// mirror `invert_colors` independently of that guard, verified directly.
    #[test]
    fn odd_bit_depth_minis_white_is_refused_by_both_read_paths() {
        // Gray(24) single-sample MINISWHITE, unsigned (no SampleFormat tag).
        let bytes = build_tiff(vec![Page {
            entries: base_entries(2, 2, 24, 1, 0), // photometric 0 == MINISWHITE
            pixels: vec![0u8; 2 * 2 * 3],          // 3 bytes per 24-bit sample
        }]);
        let path = write_fixture("sitk_io_tiff_minis_white_gray24.tif", &bytes);
        let info = read_information(&path);
        let image = read(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&info, Err(IoError::UnsupportedTiffFeature(_))),
            "read_information should refuse Gray(24) MINISWHITE: {info:?}"
        );
        assert!(
            matches!(&image, Err(IoError::UnsupportedTiffFeature(_))),
            "read should refuse Gray(24) MINISWHITE: {image:?}"
        );
        assert_eq!(
            info.is_err(),
            image.is_err(),
            "read_information and read must agree on refusal"
        );

        // Directly exercise the tightened Uint arm the BitsPerSample guard shadows.
        assert!(!crate_can_invert_whiteiszero(&ColorType::Gray(24), 1));
        assert!(crate_can_invert_whiteiszero(&ColorType::Gray(8), 1));
        assert!(crate_can_invert_whiteiszero(&ColorType::Gray(16), 1));
        assert!(!crate_can_invert_whiteiszero(&ColorType::Gray(16), 3)); // f16: unsupported IEEEFP width
        assert!(crate_can_invert_whiteiszero(&ColorType::Gray(32), 3)); // f32
        assert!(!crate_can_invert_whiteiszero(&ColorType::Gray(8), 2)); // signed
    }

    /// `ORIENTATION_BOTLEFT` sends scanline `row` to `height - 1 - row`
    /// (itkTIFFImageIO.cxx:1392-1395).
    #[test]
    fn bottom_left_orientation_flips_the_rows() {
        let mut entries = base_entries(3, 2, 8, 1, 1);
        entries.push((TAG_ORIENTATION, TYPE_SHORT, 1, short(ORIENTATION_BOTLEFT)));
        let bytes = build_tiff(vec![Page {
            entries,
            pixels: vec![1u8, 2, 3, 4, 5, 6],
        }]);
        let path = write_fixture("sitk_io_tiff_botleft.tif", &bytes);
        let image = read(&path).expect("BOTLEFT reads");
        std::fs::remove_file(&path).ok();

        assert_eq!(image.buffer(), &PixelBuffer::UInt8(vec![4, 5, 6, 1, 2, 3]));
    }

    /// `Clean()` seeds `m_XResolution = 1`, and `ReadImageInformation`'s guard
    /// is `m_XResolution > 0` — which that seed passes. A page with
    /// `RESOLUTIONUNIT = 2` and no `XRESOLUTION` therefore reports a spacing of
    /// `25.4 / 1`, not `1.0`. Ledger §2.142.
    #[test]
    fn resolution_unit_without_a_resolution_tag_yields_a_spacing_of_25_4() {
        let mut entries = base_entries(2, 1, 8, 1, 1);
        entries.push((TAG_RESOLUTION_UNIT, TYPE_SHORT, 1, short(2)));
        let bytes = build_tiff(vec![Page {
            entries,
            pixels: vec![0u8, 0],
        }]);
        let path = write_fixture("sitk_io_tiff_resunit_only.tif", &bytes);
        let information = read_information(&path).expect("header reads");
        std::fs::remove_file(&path).ok();

        assert_eq!(information.spacing, vec![25.4, 25.4]);
    }

    /// Unit `3` is centimetres: `10.0 / resolution` (itkTIFFImageIO.cxx:382-386).
    #[test]
    fn resolution_unit_centimetre_divides_ten() {
        let mut entries = base_entries(2, 1, 8, 1, 1);
        entries.push((TAG_RESOLUTION_UNIT, TYPE_SHORT, 1, short(3)));
        entries.push((TAG_X_RESOLUTION, TYPE_RATIONAL, 1, {
            let mut v = long(40);
            v.extend(long(1));
            v
        }));
        entries.push((TAG_Y_RESOLUTION, TYPE_RATIONAL, 1, {
            let mut v = long(20);
            v.extend(long(1));
            v
        }));
        let bytes = build_tiff(vec![Page {
            entries,
            pixels: vec![0u8, 0],
        }]);
        let path = write_fixture("sitk_io_tiff_resunit_cm.tif", &bytes);
        let information = read_information(&path).expect("header reads");
        std::fs::remove_file(&path).ok();

        assert_eq!(information.spacing, vec![0.25, 0.5]);
    }

    /// A `PHOTOMETRIC_PALETTE` page is inside `CanRead()`'s accepted set
    /// upstream, but the `tiff` crate refuses `RGBPalette` in `colortype()` and
    /// every decode path goes through it. Ledger §4.100.
    #[test]
    fn a_palette_tiff_is_refused_by_both_entry_points() {
        let mut entries = base_entries(2, 1, 8, 1, 3); // photometric 3 == PALETTE
        // A 256-entry colour map, three channels of `uint16`.
        let map: Vec<u8> = (0..3 * 256).flat_map(|_| short(0)).collect();
        entries.push((TAG_COLOR_MAP, TYPE_SHORT, 3 * 256, map));
        let bytes = build_tiff(vec![Page {
            entries,
            pixels: vec![0u8, 1],
        }]);
        let path = write_fixture("sitk_io_tiff_palette.tif", &bytes);
        let info = read_information(&path);
        let image = read(&path);
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&info, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("palette")),
            "{info:?}"
        );
        assert!(
            matches!(&image, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("palette")),
            "{image:?}"
        );
    }

    fn gray_page(width: u32, height: u32, subfile_type: Option<u32>, fill: u8) -> Page {
        let mut entries = base_entries(width, height, 8, 1, 1);
        if let Some(t) = subfile_type {
            entries.push((TAG_SUBFILE_TYPE, TYPE_LONG, 1, long(t)));
        }
        Page {
            entries,
            pixels: vec![fill; (width * height) as usize],
        }
    }

    /// Three directories in the order `[REDUCEDIMAGE, 0, 0]`: `m_SubFiles == 2`
    /// so the volume has two slices, but `ReadVolume` offsets the two kept pages
    /// by their *directory* indices, 1 and 2 — writing past the end of a
    /// two-slice buffer. Ledger §1.66, shape 1.
    #[test]
    fn an_ignored_page_before_a_kept_one_would_overflow_the_volume() {
        let bytes = build_tiff(vec![
            gray_page(2, 2, Some(FILETYPE_REDUCEDIMAGE), 9),
            gray_page(2, 2, Some(0), 1),
            gray_page(2, 2, Some(0), 2),
        ]);
        let path = write_fixture("sitk_io_tiff_overflow_leading_thumbnail.tif", &bytes);
        let information = read_information(&path).expect("header reads");
        let image = read(&path);
        std::fs::remove_file(&path).ok();

        assert_eq!(information.size, vec![2, 2, 2]);
        assert!(
            matches!(&image, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("§1.66")),
            "{image:?}"
        );
    }

    /// Two directories, the first with no `SUBFILETYPE` tag at all and the
    /// second tagged `0`: `m_SubFiles == 1` and `m_IgnoredSubFiles == 0`, so the
    /// volume gets one slice while `ReadVolume` reads both directories and
    /// places the second at slice 1. Ledger §1.66, shape 2.
    #[test]
    fn an_untagged_page_beside_a_tagged_one_would_overflow_the_volume() {
        let bytes = build_tiff(vec![gray_page(2, 2, None, 1), gray_page(2, 2, Some(0), 2)]);
        let path = write_fixture("sitk_io_tiff_overflow_untagged.tif", &bytes);
        let information = read_information(&path).expect("header reads");
        let image = read(&path);
        std::fs::remove_file(&path).ok();

        assert_eq!(information.size, vec![2, 2, 1]);
        assert!(
            matches!(&image, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("§1.66")),
            "{image:?}"
        );
    }

    /// The uniform "every directory must match directory 0" rule that closes the
    /// over-read family: `ReadGenericImage` sizes its scanline buffer from the
    /// *current* directory but copies directory 0's `width * inc` components out
    /// of it. Ledger §1.67, §4.99.
    #[test]
    fn a_volume_whose_pages_differ_in_geometry_is_refused() {
        let bytes = build_tiff(vec![gray_page(4, 2, None, 1), gray_page(2, 2, None, 2)]);
        let path = write_fixture("sitk_io_tiff_mixed_page_sizes.tif", &bytes);
        let image = read(&path);
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&image, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("§1.67")),
            "{image:?}"
        );
    }

    /// Three untagged directories: `m_SubFiles == m_IgnoredSubFiles == 0`, so
    /// `m_Dimensions[2] = m_NumberOfPages` and every directory index is its own
    /// slice — the common multi-page case, and the only one that assembles
    /// cleanly.
    #[test]
    fn untagged_pages_assemble_into_a_volume_in_directory_order() {
        let bytes = build_tiff(vec![
            gray_page(2, 1, None, 1),
            gray_page(2, 1, None, 2),
            gray_page(2, 1, None, 3),
        ]);
        let path = write_fixture("sitk_io_tiff_volume.tif", &bytes);
        let image = read(&path).expect("multi-page reads");
        std::fs::remove_file(&path).ok();

        assert_eq!(image.size(), &[2, 1, 3]);
        assert_eq!(image.buffer(), &PixelBuffer::UInt8(vec![1, 1, 2, 2, 3, 3]));
    }

    #[test]
    fn can_write_file_is_case_sensitive() {
        let io = TiffImageIo;
        assert!(io.can_write_file(Path::new("a.tif")));
        assert!(io.can_write_file(Path::new("a.TIF")));
        assert!(io.can_write_file(Path::new("a.tiff")));
        assert!(io.can_write_file(Path::new("a.TIFF")));
        assert!(!io.can_write_file(Path::new("a.Tiff")));
        assert!(!io.can_write_file(Path::new("a.TiF")));
    }

    /// `component_type`'s ladder, including the `BitsPerSample == 32` arm with
    /// no `default:` (§1.68) and the `bits > 32` fall-through to `USHORT`.
    #[test]
    fn component_type_follows_the_bits_and_sample_format_ladder() {
        assert_eq!(component_type(8, 1).unwrap(), PixelId::UInt8);
        assert_eq!(component_type(8, 2).unwrap(), PixelId::Int8);
        assert_eq!(component_type(8, 3).unwrap(), PixelId::UInt8);
        assert_eq!(component_type(1, 1).unwrap(), PixelId::UInt8);
        assert_eq!(component_type(16, 1).unwrap(), PixelId::UInt16);
        assert_eq!(component_type(16, 2).unwrap(), PixelId::Int16);
        // `SampleFormat 3` (half float) at 16 bits still reports USHORT.
        assert_eq!(component_type(16, 3).unwrap(), PixelId::UInt16);
        assert_eq!(component_type(32, 1).unwrap(), PixelId::UInt32);
        assert_eq!(component_type(32, 2).unwrap(), PixelId::Int32);
        assert_eq!(component_type(32, 3).unwrap(), PixelId::Float32);
        assert!(matches!(
            component_type(32, 4),
            Err(IoError::UnsupportedTiffFeature(_))
        ));
        // 64 bits lands in the trailing `else`, not in the 32-bit arm.
        assert_eq!(component_type(64, 1).unwrap(), PixelId::UInt16);
    }

    /// `rowsperstrip = 1 MiB / scanlinesize`, floored at 1.
    #[test]
    fn rows_per_strip_targets_one_mebibyte() {
        assert_eq!(rows_per_strip(1024, 1, 8), 1024);
        assert_eq!(rows_per_strip(1024, 3, 8), 341);
        assert_eq!(rows_per_strip(1024, 1, 16), 512);
        // A scanline larger than 1 MiB still gets one row.
        assert_eq!(rows_per_strip(2_000_000, 1, 8), 1);
    }

    /// Fixed §3.52: the clamped `1..=9` `compression_level` range partitions
    /// into exactly three `DeflateLevel` tiers, each discriminant at the
    /// middle of its own bucket.
    #[test]
    fn deflate_level_for_partitions_one_through_nine_into_three_tiers() {
        for level in 1..=3 {
            assert_eq!(
                deflate_level_for(level),
                DeflateLevel::Fast,
                "level {level}"
            );
        }
        for level in 4..=6 {
            assert_eq!(
                deflate_level_for(level),
                DeflateLevel::Balanced,
                "level {level}"
            );
        }
        for level in 7..=9 {
            assert_eq!(
                deflate_level_for(level),
                DeflateLevel::Best,
                "level {level}"
            );
        }
    }

    /// libtiff's `DoubleToRational` easy paths, and the continued-fraction one
    /// that `25.4 / spacing` almost always lands in.
    #[test]
    fn double_to_rational_matches_libtiff() {
        // `tiff::encoder::Rational` is neither `Debug` nor `PartialEq`.
        let parts = |v: f64| {
            let r = double_to_rational(v);
            (r.n, r.d)
        };
        assert_eq!(parts(0.0), (0, 1));
        assert_eq!(parts(25.0), (25, 1));
        assert_eq!(parts(-1.0), (0, 0));

        // The continued-fraction path must reproduce `25.4f32` exactly once the
        // stored rational is read back through libtiff's `float` field — so a
        // spacing of 1 mm survives as *the same* `f32` resolution, which is the
        // most libtiff's `TIFFTAG_XRESOLUTION` can promise. It is not a spacing
        // of exactly 1: `25.4 / static_cast<double>(25.4f)` is
        // 1.0000000150184933 (itkTIFFImageIO.cxx:378, §2.142).
        let r = double_to_rational(f64::from(25.4f32));
        let recovered = (f64::from(r.n) / f64::from(r.d)) as f32;
        assert_eq!(recovered, 25.4f32);
        assert_eq!(25.4 / f64::from(recovered), 1.000_000_015_018_493_3);
    }

    /// Fixed §2.140: `page_rows` no longer takes a separate declared
    /// component count to truncate against — every row is copied whole.
    #[test]
    fn page_rows_copies_every_sample_of_each_row() {
        let decoded: Vec<u8> = (0..8).collect();
        let mut out = vec![0u8; 8];
        page_rows(&decoded, &mut out, 2, 2, 2, true);
        assert_eq!(out, decoded);
    }

    /// `ORIENTATION_BOTLEFT` places scanline `row` at `height - 1 - row`
    /// (itkTIFFImageIO.cxx:1392-1395).
    #[test]
    fn page_rows_flips_a_bottom_left_image() {
        let decoded: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let mut out = vec![0u8; 6];
        page_rows(&decoded, &mut out, 3, 2, 1, false);
        assert_eq!(out, vec![4, 5, 6, 1, 2, 3]);
    }
}
