//! PNG (`.png`) reader and writer — `itk::PNGImageIO`, on the pure-Rust `png`
//! crate (ledger §5.8(a): `cargo tree -p sitk-io` shows no `*-sys` crate —
//! `png`'s own deflate/inflate ride on `fdeflate` and `miniz_oxide`).
//!
//! # Palette expansion always wins
//!
//! `ReadImageInformation` branches on `m_ExpandRGBPalette` for a
//! `PNG_COLOR_TYPE_PALETTE` source (itkPNGImageIO.cxx:381-414): `true` calls
//! `png_set_palette_to_rgb`, `false` calls `png_set_packing` instead and keeps
//! the raw indices plus `m_ColorPalette`
//! (`m_IsReadAsScalarPlusPalette = true`). The header's in-class initializer
//! reads `false` (`bool m_ExpandRGBPalette{};`, itkImageIOBase.h:853), but
//! `ImageIOBase`'s constructor calls `Reset(false)`, which sets
//! `m_ExpandRGBPalette = true` unconditionally (itkImageIOBase.cxx:28,45) —
//! `PNGImageIO`'s own constructor never touches it, so every instance starts
//! with expansion **on**. Ledger §2.127. SimpleITK never calls
//! `SetExpandRGBPalette`/`ExpandRGBPaletteOff` anywhere in its wrapping layer,
//! so through the surface this crate ports, a palette PNG is *always*
//! expanded to RGB (RGBA if a `tRNS` chunk is present). [`read`] and
//! [`read_information`] implement only that branch: the `png` crate's
//! [`Transformations::EXPAND`] does exactly `png_set_palette_to_rgb` +
//! `png_set_expand_gray_1_2_4_to_8` + `png_set_tRNS_to_alpha` in one flag. The
//! `m_IsReadAsScalarPlusPalette` / `m_ColorPalette` path is not implemented —
//! unreachable through this crate's [`ImageFileReader`](crate::ImageFileReader).
//! Ledger §4.84.
//!
//! # Pixel types: a 2-channel PNG is unrepresentable through SimpleITK
//!
//! `ReadImageInformation` sets `m_PixelType = SCALAR` unconditionally
//! (`:444`/`:449`), then `SetNumberOfComponents(png_get_channels(...))`
//! (`:452`), and *only afterward* upgrades to `RGB` for 3 components or `RGBA`
//! for 4 (`:454-461`) — 1 and 2 components are left as `SCALAR`.
//! `ImageReaderBase::GetPixelIDFromImageIO` accepts `SCALAR` only when
//! `numberOfComponents == 1` and otherwise requires `RGB`/`RGBA`/`VECTOR`/…
//! (sitkImageReaderBase.cxx:215-228); `SCALAR` with `numberOfComponents == 2`
//! matches neither arm and falls into `"Unknown PixelType: ..."`
//! (`:236-237`). A gray+alpha PNG is therefore readable and writable by raw
//! `PNGImageIO` but unrepresentable through this crate's public reader.
//! Ledger §3.44; see [`IoError::UnsupportedPngFeature`].
//!
//! | source channels | `m_PixelType` | reachable through this crate? |
//! |---|---|---|
//! | 1 (gray, no `tRNS`) | `SCALAR` | yes — scalar |
//! | 2 (gray + alpha) | `SCALAR` | **no** — `IoError::UnsupportedPngFeature` |
//! | 3 (RGB) | `RGB` | yes — `VectorUInt8`/`VectorUInt16` |
//! | 4 (RGBA) | `RGBA` | yes — `VectorUInt8`/`VectorUInt16` |
//!
//! # 16-bit endianness
//!
//! `Read` and `WriteSlice` both call `png_set_swap` when `bitDepth > 8` and
//! `ITK_WORDS_BIGENDIAN` is unset (`:214-219`, `:676-682`) — PNG's own wire
//! format is big-endian, and on a little-endian host libpng byte-swaps every
//! sample into the host's native order on the way in and back to big-endian
//! on the way out. The `png` crate has no such transform: [`Reader::next_frame`]
//! and [`Writer::write_image_data`] hand over raw, always-big-endian bytes.
//! [`pack_pixels`] and [`buffer_to_be_bytes`] do the swap by hand with
//! [`u16::from_be_bytes`]/[`u16::to_be_bytes`], which is a no-op on a
//! big-endian host and the swap upstream performs on a little-endian one.
//!
//! # Compression level is dead unless `UseCompression` is on
//!
//! Third instance of the `SetUseCompression`/`SetCompressionLevel` shape after
//! NIfTI (§3.40) and GIPL (§3.42), and distinct from both: `Write` only calls
//! `png_set_compression_level` **inside `if (m_UseCompression)`**
//! (itkPNGImageIO.cxx:661-665) — every PNG's pixel data is deflated either
//! way, so turning compression "off" does not skip deflating, it only skips
//! *choosing a level*, leaving libpng/zlib at their own built-in default
//! (`Z_DEFAULT_COMPRESSION`, 6) rather than at `PNGImageIO`'s own constructed
//! default of 4 (`:271-272`). See [`crate::compression`] and ledger §3.45.
//! [`write`] reproduces this: `options.use_compression` gates whether
//! [`Encoder::set_deflate_compression`] is called at all — left alone, the
//! `png` crate's own default is `DeflateCompression::Level(6)`
//! (`flate2::Compression::default()`), matching zlib's built-in default byte
//! for byte.
//!
//! # `CanWriteFile` is case-sensitive
//!
//! `HasSupportedWriteExtension(name, false)` (itkPNGImageIO.cxx:497) — unlike
//! `MetaImageIO`'s case-insensitive default (the trait's own default method).
//! `foo.PNG` is claimed (both `.png` and `.PNG` are registered extensions,
//! `:281-287`) but `foo.PnG` is not. Ledger §2.124.
//!
//! # Writing a 3-D (or higher) image writes only its first slice
//!
//! `WriteSlice` takes `height` from `GetDimensions(1)` alone —
//! `(m_NumberOfDimensions > 1) ? GetDimensions(1) : 1` (`:605`) — and never
//! consults any axis beyond that. `png_write_image` therefore reads exactly
//! `height` rows starting at the buffer's first byte; a 3-D image's second and
//! later slices are simply never addressed, with no error. Ledger §2.125;
//! [`write`] reproduces it structurally (see next section) rather than
//! special-casing the dimension.
//!
//! # A >4-component vector image writes deterministic garbage
//!
//! `WriteSlice`'s `colorType` switch on `numComp` has arms for 1–3 and a
//! `default:` that always picks `PNG_COLOR_TYPE_RGB_ALPHA` — 4 declared
//! channels — for anything else (`:579-600`). Row *pointers* are spaced by
//! the image's **actual** `numComp` (`rowInc = width * numComp * bitDepth /
//! 8`, `:686-693`), so each row starts at the right offset; but
//! `png_write_image` reads only `width * 4 * bitDepth / 8` bytes from each —
//! the declared channel count — leaving every row's trailing
//! `(numComp - 4)` components' worth of bytes untouched and unwritten. No
//! error. Ledger §2.126.
//!
//! [`write`] reproduces both quirks above with one routine,
//! [`rows_for_declared_channels`]: it is handed the whole flat buffer, the
//! *actual* per-row byte stride, and the *declared* one, and copies only the
//! declared-length prefix of each of `height` actual-length rows starting at
//! byte 0. A 3-D image's `height` is `size[1]` alone, so slices at `size[2..]`
//! are simply never reached by the row loop — the same "take the first
//! `height` actual-stride rows" rule that truncates a >4-component row also
//! truncates the image to its first Z-slice; neither needs its own branch.
//!
//! # Not implemented
//!
//! * **`sCAL`-chunk spacing.** `ReadImageInformation` reads `px_width`/
//!   `px_height` from `png_get_sCAL` when `PNG_sCAL_SUPPORTED` and
//!   `PNG_FLOATING_POINT_SUPPORTED` are both compiled in (`:467-478`), and
//!   `WriteSlice` writes them back with `png_set_sCAL` under the same guard
//!   (`:671-673`). The `png` crate parses no such chunk — there is no
//!   `sCAL` entry in [`png::chunk`] at all — so [`read_information`]/[`read`]
//!   always report unit spacing `[1.0, 1.0]` and [`write`] never emits the
//!   chunk. `gAMA` is not a substitute: `itkPNGImageIO.cxx` never references
//!   it. Ledger §4.85.
//! * **`sBIT`-driven shifting.** `png_get_sBIT`/`png_set_shift` renormalizes
//!   samples that use fewer than the full bit depth (`:227-232`); the `png`
//!   crate parses the chunk into `Info::sbit` but implements no shift
//!   transform, so a rare `sBIT`-bearing file reads at full range instead.
//!   Ledger §4.86.
//! * **`WritePalette` / indexed write.** `numComp == 1 && GetWritePalette()`
//!   selects `PNG_COLOR_TYPE_PALETTE` (`:582-585`) and a following block fills
//!   `png_set_PLTE` from `m_ColorPalette` (`:620-659`); SimpleITK exposes no
//!   `WritePalette` setter anywhere, so the branch is unreachable through this
//!   crate's public writer and every 1-component image writes `Grayscale`.
//!   Ledger §4.87.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

use png::{BitDepth, ColorType, Decoder, DeflateCompression, Encoder, Reader, Transformations};
use sitk_core::{Image, PixelBuffer, PixelId};

use crate::compression::PNG_DEFAULT_COMPRESSION_LEVEL;
use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, ImageIo, has_supported_extension};
use crate::writer::WriteOptions;

/// `png_sig_cmp`'s eight-byte signature (itkPNGImageIO.cxx:79-89).
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

fn malformed_too_short(path: &Path, len: usize) -> IoError {
    IoError::MalformedPngHeader(format!(
        "PNGImageIO failed to read header for file: {}\nReason: fread read only {len} instead of 8",
        path.display()
    ))
}

fn malformed_bad_signature(path: &Path) -> IoError {
    IoError::MalformedPngHeader(format!("File is not png type: {}", path.display()))
}

/// Check the 8-byte signature and open a decoder with
/// [`Transformations::EXPAND`], reproducing `Read`/`ReadImageInformation`'s
/// shared prologue up through `png_read_info` (itkPNGImageIO.cxx:130-173,
/// :325-368). `ReadImageInformation`'s own signature mismatch is a silent
/// `return` rather than a throw (`:334-338`); this port raises the same
/// [`IoError::MalformedPngHeader`] from both entry points instead — see the
/// module's ledger §1.60 note in `doc/upstream-findings.md`.
fn open_reader(path: &Path, bytes: Vec<u8>) -> Result<Reader<Cursor<Vec<u8>>>> {
    if bytes.len() < 8 {
        return Err(malformed_too_short(path, bytes.len()));
    }
    if bytes[..8] != SIGNATURE {
        return Err(malformed_bad_signature(path));
    }
    let mut decoder = Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(Transformations::EXPAND);
    Ok(decoder.read_info()?)
}

/// What [`read_information`] and [`read`] keep from the decoded header.
struct Header {
    width: u32,
    height: u32,
    pixel_id: PixelId,
    number_of_components: usize,
}

/// `png_get_channels` after every `png_set_*` transform, translated into this
/// crate's [`PixelId`] exactly as `ReadImageInformation`'s bit-depth/channel
/// switch does (itkPNGImageIO.cxx:442-461): `<= 8` bits is `UInt8`, `16` bits
/// is `UInt16`, 3 channels is the vector counterpart, 4 channels likewise, and
/// 2 channels — gray plus alpha — is unrepresentable (§3.44).
fn header_from_reader<R: std::io::BufRead + std::io::Seek>(reader: &Reader<R>) -> Result<Header> {
    let (color, depth) = reader.output_color_type();
    let channels = color.samples();
    let component = if depth == BitDepth::Sixteen {
        PixelId::UInt16
    } else {
        PixelId::UInt8
    };
    let pixel_id = match channels {
        1 => component,
        3 | 4 => component.vector_id(),
        _ => {
            return Err(IoError::UnsupportedPngFeature(format!(
                "Unknown PixelType: a {channels}-channel PNG (grayscale + alpha) \
                 keeps m_PixelType == SCALAR with NumberOfComponents == {channels} \
                 (itkPNGImageIO.cxx:444-461), which GetPixelIDFromImageIO has no \
                 arm for (sitkImageReaderBase.cxx:215-237) — doc/upstream-findings.md §3.44"
            )));
        }
    };
    let info = reader.info();
    Ok(Header {
        width: info.width,
        height: info.height,
        pixel_id,
        number_of_components: channels,
    })
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Read the header only, with no pixel data.
///
/// Spacing and origin are always `[1.0, 1.0]` / `[0.0, 0.0]`: the `png` crate
/// has no `sCAL` support (§4.85). The meta-data dictionary is empty —
/// `PNGImageIO` writes nothing into `m_MetaDataDictionary`.
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    let bytes = std::fs::read(path)?;
    let reader = open_reader(path, bytes)?;
    let header = header_from_reader(&reader)?;

    Ok(ImageInformation {
        pixel_id: header.pixel_id,
        dimension: 2,
        number_of_components: header.number_of_components,
        size: vec![header.width as usize, header.height as usize],
        spacing: vec![1.0, 1.0],
        origin: vec![0.0, 0.0],
        direction: identity(2),
        metadata: BTreeMap::new(),
    })
}

fn pack_pixels(component: PixelId, raw: Vec<u8>) -> PixelBuffer {
    match component {
        PixelId::UInt8 => PixelBuffer::UInt8(raw),
        PixelId::UInt16 => PixelBuffer::UInt16(
            raw.chunks_exact(2)
                .map(|c| u16::from_be_bytes([c[0], c[1]]))
                .collect(),
        ),
        other => unreachable!("{other:?} is not a PNG component type"),
    }
}

/// Read a `.png` file.
pub fn read(path: &Path) -> Result<Image> {
    let bytes = std::fs::read(path)?;
    let mut reader = open_reader(path, bytes)?;
    let header = header_from_reader(&reader)?;

    let len = reader
        .output_buffer_size()
        .ok_or_else(|| IoError::UnsupportedPngFeature("image too large to buffer".to_string()))?;
    let mut raw = vec![0u8; len];
    reader.next_frame(&mut raw)?;

    let buffer = pack_pixels(header.pixel_id.component_id(), raw);
    let size = vec![header.width as usize, header.height as usize];
    let spacing = vec![1.0, 1.0];
    let origin = vec![0.0, 0.0];
    let direction = identity(2);

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

/// `WriteSlice`'s component-type switch (itkPNGImageIO.cxx:531-553): only
/// `UCHAR` and `USHORT` have an arm.
fn component_bit_depth(component: PixelId) -> Option<BitDepth> {
    match component {
        PixelId::UInt8 => Some(BitDepth::Eight),
        PixelId::UInt16 => Some(BitDepth::Sixteen),
        _ => None,
    }
}

/// `WriteSlice`'s `colorType` switch on `numComp` (itkPNGImageIO.cxx:579-600).
/// `GetWritePalette()` is unreachable (§4.87), so `numComp == 1` always
/// selects `Grayscale`, never `Palette`.
fn color_type_for(number_of_components: usize) -> ColorType {
    match number_of_components {
        1 => ColorType::Grayscale,
        2 => ColorType::GrayscaleAlpha,
        3 => ColorType::Rgb,
        _ => ColorType::Rgba,
    }
}

fn buffer_to_be_bytes(buffer: &PixelBuffer) -> Vec<u8> {
    match buffer {
        PixelBuffer::UInt8(v) => v.clone(),
        PixelBuffer::UInt16(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
        other => unreachable!("{:?} is not a PNG component type", other.component_id()),
    }
}

/// Copy only the first `declared_channels` samples of each pixel, for the
/// first `height` rows of `all_bytes`. This is [`write`]'s single reproduction
/// of both the row-truncation quirk (§2.126) and the first-slice-only quirk
/// (§2.125) — see the module doc.
fn rows_for_declared_channels(
    all_bytes: &[u8],
    width: usize,
    height: usize,
    actual_components: usize,
    declared_channels: usize,
    bytes_per_sample: usize,
) -> Vec<u8> {
    let actual_row = width * actual_components * bytes_per_sample;
    let declared_row = width * declared_channels * bytes_per_sample;
    let mut out = Vec::with_capacity(declared_row * height);
    for row in 0..height {
        let start = row * actual_row;
        out.extend_from_slice(&all_bytes[start..start + declared_row]);
    }
    out
}

/// Write a `.png` file.
///
/// `WriteImageInformation` is a no-op upstream; the header and pixel data are
/// both emitted by `Write`/`WriteSlice` here in one pass, as they are
/// upstream.
pub fn write(image: &Image, path: &Path, options: &WriteOptions) -> Result<()> {
    let file = std::fs::File::create(path)?;

    let component = image.buffer().component_id();
    let Some(bit_depth) = component_bit_depth(component) else {
        return Err(IoError::UnsupportedPngFeature(format!(
            "PNG supports unsigned char and unsigned short, not {} \
             (itkPNGImageIO.cxx:550; the target file is left truncated to zero \
             bytes by fopen(\"wb\") — doc/upstream-findings.md §1.59)",
            component.as_str()
        )));
    };

    let width = image.size()[0];
    let height = if image.dimension() > 1 {
        image.size()[1]
    } else {
        1
    };
    let actual_components = image.buffer_stride();
    let color = color_type_for(actual_components);
    let declared_channels = color.samples();
    let bytes_per_sample = if bit_depth == BitDepth::Sixteen { 2 } else { 1 };

    let all_bytes = buffer_to_be_bytes(image.buffer());
    let data = rows_for_declared_channels(
        &all_bytes,
        width,
        height,
        actual_components,
        declared_channels,
        bytes_per_sample,
    );

    let mut encoder = Encoder::new(file, width as u32, height as u32);
    encoder.set_color(color);
    encoder.set_depth(bit_depth);
    if options.use_compression {
        let level = options.resolved_level(PNG_DEFAULT_COMPRESSION_LEVEL);
        encoder.set_deflate_compression(DeflateCompression::Level(level as u8));
    }

    let mut writer = encoder.write_header()?;
    writer.write_image_data(&data)?;
    writer.finish()?;
    Ok(())
}

/// `itk::PNGImageIO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PngImageIo;

impl ImageIo for PngImageIo {
    fn name(&self) -> &'static str {
        "PNGImageIO"
    }

    /// `.png` and `.PNG`, both registered for read and write
    /// (itkPNGImageIO.cxx:281-287).
    fn supported_read_extensions(&self) -> &'static [&'static str] {
        &[".png", ".PNG"]
    }

    fn supported_write_extensions(&self) -> &'static [&'static str] {
        &[".png", ".PNG"]
    }

    /// `CanReadFile` (itkPNGImageIO.cxx:61-112): the signature alone, no
    /// extension check.
    fn can_read_file(&self, path: &Path) -> bool {
        use std::io::Read;

        let Ok(mut file) = std::fs::File::open(path) else {
            return false;
        };
        let mut header = [0u8; 8];
        file.read_exact(&mut header).is_ok() && header == SIGNATURE
    }

    /// `CanWriteFile` is `HasSupportedWriteExtension(name, false)` —
    /// case-**sensitive**, unlike the trait's case-insensitive default.
    /// Ledger §2.124.
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

    /// IEEE 802.3 CRC-32 (reversed polynomial `0xEDB88320`), the same table-free
    /// bit-at-a-time form zlib's `crc32` computes, needed to hand-build a PNG
    /// chunk without going through the `png` crate's own encoder.
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 4 + data.len() + 4);
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let mut crc_input = Vec::with_capacity(4 + data.len());
        crc_input.extend_from_slice(kind);
        crc_input.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
        out
    }

    /// Hand-build a minimal 16-bit grayscale PNG: signature, `IHDR`, one
    /// zlib-deflated `IDAT` (built with `flate2`, which is infrastructure here,
    /// not the code under test), and `IEND` — to pin the big-endian sample
    /// order on the wire independently of the `png` crate's own encoder.
    fn hand_built_16bit_gray_png(width: u32, height: u32, samples: &[u16]) -> Vec<u8> {
        use std::io::Write as _;

        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.push(16); // bit depth
        ihdr.push(0); // color type: grayscale
        ihdr.push(0); // compression method
        ihdr.push(0); // filter method
        ihdr.push(0); // interlace method

        let mut raw = Vec::new();
        for row in 0..height as usize {
            raw.push(0u8); // filter type: none
            for col in 0..width as usize {
                raw.extend_from_slice(&samples[row * width as usize + col].to_be_bytes());
            }
        }
        let mut deflater =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        deflater
            .write_all(&raw)
            .expect("write to in-memory zlib encoder cannot fail");
        let idat = deflater.finish().expect("zlib finish cannot fail");

        let mut out = SIGNATURE.to_vec();
        out.extend_from_slice(&chunk(b"IHDR", &ihdr));
        out.extend_from_slice(&chunk(b"IDAT", &idat));
        out.extend_from_slice(&chunk(b"IEND", &[]));
        out
    }

    #[test]
    fn sixteen_bit_samples_are_read_back_in_native_endian_order() {
        let samples = [0x0102u16, 0xfffeu16, 0x0000u16, 0x8000u16];
        let bytes = hand_built_16bit_gray_png(2, 2, &samples);

        let dir = std::env::temp_dir().join("sitk_io_png_16bit_test.png");
        std::fs::write(&dir, &bytes).expect("write fixture");
        let image = read(&dir).expect("hand-built 16-bit PNG should read");
        std::fs::remove_file(&dir).ok();

        assert_eq!(image.pixel_id(), PixelId::UInt16);
        assert_eq!(image.size(), &[2, 2]);
        match image.buffer() {
            PixelBuffer::UInt16(v) => assert_eq!(v, &samples),
            other => panic!("expected UInt16, got {other:?}"),
        }
    }

    #[test]
    fn rows_for_declared_channels_truncates_each_row_to_the_declared_prefix() {
        // Two "rows" of 5 components each; declared channels is 4 (RGBA),
        // reproducing the >4-component write quirk (§2.126).
        let all: Vec<u8> = (0..10).collect();
        let out = rows_for_declared_channels(&all, 1, 2, 5, 4, 1);
        assert_eq!(out, vec![0, 1, 2, 3, 5, 6, 7, 8]);
    }

    #[test]
    fn rows_for_declared_channels_stops_after_the_requested_height() {
        // Three "slices" of one row each; only the first slice's row is kept,
        // reproducing the first-slice-only write quirk (§2.125).
        let all: Vec<u8> = (0..9).collect();
        let out = rows_for_declared_channels(&all, 1, 1, 3, 3, 1);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn color_type_for_never_selects_palette() {
        assert_eq!(color_type_for(1), ColorType::Grayscale);
        assert_eq!(color_type_for(2), ColorType::GrayscaleAlpha);
        assert_eq!(color_type_for(3), ColorType::Rgb);
        assert_eq!(color_type_for(4), ColorType::Rgba);
        assert_eq!(color_type_for(5), ColorType::Rgba);
    }

    #[test]
    fn can_write_file_is_case_sensitive() {
        let io = PngImageIo;
        assert!(io.can_write_file(Path::new("a.png")));
        assert!(io.can_write_file(Path::new("a.PNG")));
        assert!(!io.can_write_file(Path::new("a.PnG")));
        assert!(!io.can_write_file(Path::new("a.Png")));
    }

    #[test]
    fn can_read_file_is_signature_only_no_extension_required() {
        let io = PngImageIo;
        let dir = std::env::temp_dir().join("sitk_io_png_sig_test.dat");
        std::fs::write(&dir, SIGNATURE).expect("write fixture");
        assert!(io.can_read_file(&dir));
        std::fs::remove_file(&dir).ok();
    }

    /// `use_compression: false` (the default) never calls
    /// `set_deflate_compression` at all, so the `png` crate's own default —
    /// `DeflateCompression::Level(6)`, `flate2::Compression::default()` — is
    /// what ends up on disk, matching zlib's `Z_DEFAULT_COMPRESSION` byte for
    /// byte rather than `PNGImageIO`'s own constructed default of 4. Ledger
    /// §3.45.
    #[test]
    fn use_compression_off_matches_the_crates_own_default_level_not_pngs_own() {
        let data: Vec<u8> = (0..(32 * 32))
            .map(|i| ((i * 37 + i * i) % 256) as u8)
            .collect();
        let image = Image::from_vec(&[32, 32], data).unwrap();

        let off_path = std::env::temp_dir().join("sitk_io_png_compression_off.png");
        write(&image, &off_path, &WriteOptions::default()).unwrap();
        let off_bytes = std::fs::read(&off_path).unwrap();
        std::fs::remove_file(&off_path).ok();

        let level6_path = std::env::temp_dir().join("sitk_io_png_compression_level6.png");
        let file = std::fs::File::create(&level6_path).unwrap();
        let mut encoder = Encoder::new(file, 32, 32);
        encoder.set_color(ColorType::Grayscale);
        encoder.set_depth(BitDepth::Eight);
        encoder.set_deflate_compression(DeflateCompression::Level(6));
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(image.scalar_slice::<u8>().unwrap())
            .unwrap();
        writer.finish().unwrap();
        let level6_bytes = std::fs::read(&level6_path).unwrap();
        std::fs::remove_file(&level6_path).ok();

        assert_eq!(off_bytes, level6_bytes);

        // Turning compression on with no explicit level reaches
        // `PNG_DEFAULT_COMPRESSION_LEVEL` (4), `PNGImageIO`'s own constructed
        // default — not zlib's built-in one.
        let on_path = std::env::temp_dir().join("sitk_io_png_compression_on.png");
        write(
            &image,
            &on_path,
            &WriteOptions {
                use_compression: true,
                compression_level: -1,
            },
        )
        .unwrap();
        let on_bytes = std::fs::read(&on_path).unwrap();
        std::fs::remove_file(&on_path).ok();

        let level4_path = std::env::temp_dir().join("sitk_io_png_compression_level4.png");
        let file = std::fs::File::create(&level4_path).unwrap();
        let mut encoder = Encoder::new(file, 32, 32);
        encoder.set_color(ColorType::Grayscale);
        encoder.set_depth(BitDepth::Eight);
        encoder.set_deflate_compression(DeflateCompression::Level(4));
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(image.scalar_slice::<u8>().unwrap())
            .unwrap();
        writer.finish().unwrap();
        let level4_bytes = std::fs::read(&level4_path).unwrap();
        std::fs::remove_file(&level4_path).ok();

        assert_eq!(on_bytes, level4_bytes);
    }
}
