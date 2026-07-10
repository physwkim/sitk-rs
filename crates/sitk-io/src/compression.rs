//! The zlib and gzip streams the three image formats need, and nothing else.
//!
//! Upstream reaches zlib through three different doors, and they do not behave
//! alike. This module is the single owner of all three, so no format module
//! ever names `flate2` directly:
//!
//! | door | upstream | container | on garbage input |
//! |---|---|---|---|
//! | [`zlib_compress`] / [`inflate_auto`] | MetaIO `MET_PerformCompression` / `MET_PerformUncompression` (metaUtils.cxx:790-893) | writes **zlib**, reads zlib *or* gzip | error |
//! | [`gzip_compress`] / [`gunzip_transparent`] | NrrdIO `nrrd__GzOpen` (gzio.c:161-232) and nifti's `znzopen` → zlib's `gzopen` (znzlib.c:48-82) | **gzip** | *transparent copy* |
//!
//! The asymmetry is real and load-bearing:
//!
//! * MetaIO calls `deflateInit(&z, level)`, which emits a **zlib** wrapper
//!   (RFC 1950: two header bytes, an Adler-32 trailer), but reads with
//!   `inflateInit2(&d, 47)` — `windowBits = 15 + 32` — which auto-detects zlib
//!   *and* gzip. So a MetaImage this port writes is a zlib stream, and one it
//!   reads may be either (§2.93).
//! * NrrdIO's `gzio.c` and zlib's own `gzopen` both wrap a **raw** deflate
//!   stream (`deflateInit2(..., -MAX_WBITS, ...)`) in a hand-written gzip
//!   header and CRC-32/ISIZE trailer. Both, on reading, fall back to a
//!   byte-for-byte copy when the gzip magic is absent (`s->transparent = 1`,
//!   gzio.c:516; zlib's `gz_look` sets `state->how = COPY`). [`gunzip_transparent`]
//!   reproduces that (§2.94).
//!
//! # Compression level
//!
//! `itk::ImageIOBase` declares `itkSetClampMacro(CompressionLevel, int, 1,
//! GetMaximumCompressionLevel())` (itkImageIOBase.h:288). `MetaImageIO` and
//! `NrrdImageIO` both construct with `SetMaximumCompressionLevel(9)` then
//! `SetCompressionLevel(2)` (itkMetaImageIO.cxx:62-64, itkNrrdImageIO.cxx:353-355).
//! `itk::ImageFileWriter::GenerateData` forwards SimpleITK's level only when it
//! is non-negative (itkImageFileWriter.hxx:199-201), so SimpleITK's default of
//! `-1` (sitkImageFileWriter.h:191) leaves each IO on its own default of 2.
//!
//! NIfTI is the odd one out: `znzopen(fname, "wb", ...)` never names a level,
//! so nifti always compresses at zlib's `Z_DEFAULT_COMPRESSION` (6) and ignores
//! `SetCompressionLevel` entirely (§3.33).

use std::io::{Read, Write};

use flate2::Compression;
use flate2::GzBuilder;
use flate2::read::{GzDecoder, MultiGzDecoder, ZlibDecoder};
use flate2::write::ZlibEncoder;

use crate::error::{IoError, Result};

/// `itk::ImageIOBase`'s clamp floor for `CompressionLevel`
/// (itkImageIOBase.h:288).
pub const MIN_COMPRESSION_LEVEL: i32 = 1;

/// `SetMaximumCompressionLevel(9)`, as both `MetaImageIO` and `NrrdImageIO`
/// call it in their constructors.
pub const MAX_COMPRESSION_LEVEL: i32 = 9;

/// `SetCompressionLevel(2)`, likewise. The level an IO uses when SimpleITK
/// leaves `m_CompressionLevel` at its `-1` default.
pub const ITK_DEFAULT_COMPRESSION_LEVEL: i32 = 2;

/// `Z_DEFAULT_COMPRESSION` as zlib resolves it — the level `gzopen(path, "wb")`
/// uses, and therefore the only level nifti ever compresses at.
pub const ZLIB_DEFAULT_COMPRESSION_LEVEL: i32 = 6;

/// The two bytes `gz_look` / `check_header` test for.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// `OS_CODE` on Unix, which is what both zlib and NrrdIO's `gzio.c` stamp into
/// byte 9 of the gzip header (`N_OS_CODE 0x03`, gzio.c:72).
const GZIP_OS_UNIX: u8 = 3;

/// Turn an `int` compression level into the `Compression` the encoders take,
/// applying `itkSetClampMacro(CompressionLevel, int, 1, 9)`.
fn clamped(level: i32) -> Compression {
    Compression::new(level.clamp(MIN_COMPRESSION_LEVEL, MAX_COMPRESSION_LEVEL) as u32)
}

/// `MET_PerformCompression` (metaUtils.cxx:790-848): `deflateInit(&z, level)`
/// over the whole buffer, i.e. a **zlib**-wrapped deflate stream.
///
/// Upstream chunks the deflate at `MET_MaxChunkSize` (1 GiB, metaUtils.cxx:60)
/// only because `z_stream::avail_in` is a `uInt`; the emitted bytes do not
/// depend on the chunking, since every chunk but the last flushes with
/// `Z_NO_FLUSH`. This port hands the whole slice to the encoder at once.
pub(crate) fn zlib_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), clamped(level));
    encoder
        .write_all(data)
        .expect("writing to a Vec never fails");
    encoder.finish().expect("deflating into a Vec never fails")
}

/// `nrrd__GzOpen(file, "w<level>")` (gzio.c:161-232) and, at
/// `ZLIB_DEFAULT_COMPRESSION_LEVEL`, zlib's `gzopen(path, "wb")` as
/// `znzopen` calls it: a gzip stream with `MTIME = 0` and `OS = 0x03`.
///
/// The `XFL` byte flate2 derives from the level (`2` for 9, `4` for below 2,
/// `0` otherwise) is the rule zlib's `deflate.c` uses, so the ten header bytes
/// match upstream's byte for byte. The deflate payload does not: miniz_oxide
/// and zlib make different block and match choices at the same nominal level.
pub(crate) fn gzip_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut encoder = GzBuilder::new()
        .mtime(0)
        .operating_system(GZIP_OS_UNIX)
        .write(Vec::new(), clamped(level));
    encoder
        .write_all(data)
        .expect("writing to a Vec never fails");
    encoder.finish().expect("deflating into a Vec never fails")
}

/// `MET_PerformUncompression` (metaUtils.cxx:850-893): `inflateInit2(&d, 47)`,
/// which accepts a zlib *or* a gzip header, decoding at most `expected` bytes
/// and stopping at the first `Z_STREAM_END`.
///
/// **Divergence (§4.64).** Upstream returns `true` unconditionally. A stream
/// that fails to inflate, or that ends before `expected` bytes are out, leaves
/// the tail of a `new unsigned char[]` buffer uninitialised and the read
/// reports success. This port returns [`IoError::CorruptCompressedData`] or
/// [`IoError::TruncatedData`]; the upstream behaviour is not expressible in
/// safe Rust.
///
/// Bytes past `expected` are ignored, as they are upstream (`avail_out` hits
/// zero and the inner loop exits).
pub(crate) fn inflate_auto(source: &[u8], expected: usize) -> Result<Vec<u8>> {
    if source.starts_with(&GZIP_MAGIC) {
        // `inflateInit2(47)` decodes exactly one gzip member, so `GzDecoder`,
        // not `MultiGzDecoder`.
        read_exactly(GzDecoder::new(source), expected)
    } else {
        read_exactly(ZlibDecoder::new(source), expected)
    }
}

/// `gzread` on a `gzFile`, as both NrrdIO's `gzio.c` and zlib itself implement
/// it: inflate every gzip member in `source`, or — when the gzip magic is
/// absent — copy `source` through untouched.
///
/// The transparent fallback is upstream behaviour, not a convenience: NrrdIO
/// sets `s->transparent = 1` when `check_header` finds no magic (gzio.c:494-516)
/// and zlib's `gz_look` switches to `state->how = COPY`. So `encoding: gzip`
/// over a raw byte stream reads that stream verbatim, and a `.nii.gz` holding
/// an uncompressed header is read as if it were a `.nii` (§2.94).
pub(crate) fn gunzip_transparent(source: &[u8]) -> Result<Vec<u8>> {
    if !source.starts_with(&GZIP_MAGIC) {
        return Ok(source.to_vec());
    }
    let mut out = Vec::new();
    MultiGzDecoder::new(source)
        .read_to_end(&mut out)
        .map_err(|e| IoError::CorruptCompressedData(e.to_string()))?;
    Ok(out)
}

/// As [`gunzip_transparent`], but stopping once `limit` bytes are out — the
/// `znzread` a nifti header read performs, which never touches the pixel data
/// of a `.nii.gz`.
///
/// A stream shorter than `limit` yields what there was; the caller checks the
/// length, as `nifti_read_header` does.
pub(crate) fn gunzip_transparent_prefix(source: &[u8], limit: usize) -> Result<Vec<u8>> {
    if !source.starts_with(&GZIP_MAGIC) {
        return Ok(source[..limit.min(source.len())].to_vec());
    }
    let mut out = vec![0u8; limit];
    let read = read_up_to(MultiGzDecoder::new(source), &mut out)?;
    out.truncate(read);
    Ok(out)
}

/// Inflate exactly `expected` bytes, or fail.
fn read_exactly<R: Read>(reader: R, expected: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; expected];
    if read_up_to(reader, &mut out)? != expected {
        return Err(IoError::TruncatedData);
    }
    Ok(out)
}

/// Fill as much of `out` as the stream yields, mapping a decode failure to
/// [`IoError::CorruptCompressedData`].
fn read_up_to<R: Read>(mut reader: R, out: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < out.len() {
        match reader.read(&mut out[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(IoError::CorruptCompressedData(e.to_string())),
        }
    }
    Ok(filled)
}

/// A gzip stream built by hand out of a single *stored* (uncompressed) deflate
/// block: the ten header bytes, `BFINAL=1 BTYPE=00`, `LEN`/`NLEN`, the bytes
/// verbatim, then CRC-32 and ISIZE.
///
/// Nothing in this port produced it — not one byte comes from `flate2` — so
/// inflating it tests the readers against a stream of purely upstream shape.
/// It is what `nrrd__GzWrite` would emit at `zlibLevel = 0`.
#[cfg(test)]
pub(crate) fn stored_block_gzip(data: &[u8]) -> Vec<u8> {
    assert!(data.len() < 0xffff);
    let mut out = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, GZIP_OS_UNIX];
    out.push(0x01); // BFINAL = 1, BTYPE = 00 (stored)
    out.extend_from_slice(&(data.len() as u16).to_le_bytes());
    out.extend_from_slice(&(!(data.len() as u16)).to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// The CRC-32 of the gzip trailer, computed the long way so [`stored_block_gzip`]
/// owes nothing to the crate under test.
#[cfg(test)]
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zlib_round_trip_is_exact() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let z = zlib_compress(&data, 2);
        assert_eq!(&z[..1], &[0x78], "zlib CMF byte");
        assert_eq!(inflate_auto(&z, data.len()).unwrap(), data);
    }

    #[test]
    fn gzip_round_trip_is_exact() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let g = gzip_compress(&data, 2);
        assert_eq!(gunzip_transparent(&g).unwrap(), data);
    }

    #[test]
    fn gzip_header_matches_upstream_ten_bytes() {
        let g = gzip_compress(b"hello", 2);
        assert_eq!(&g[..10], &[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0x03]);
        // XFL follows zlib's rule: 2 at level 9, 4 below level 2, else 0.
        assert_eq!(gzip_compress(b"hello", 9)[8], 2);
        assert_eq!(gzip_compress(b"hello", 1)[8], 4);
        assert_eq!(gzip_compress(b"hello", 5)[8], 0);
    }

    #[test]
    fn compression_level_is_clamped_to_one_through_nine() {
        // `itkSetClampMacro(CompressionLevel, int, 1, 9)`: a level of 0 becomes
        // 1 (never a stored block) and 100 becomes 9.
        let data = vec![7u8; 4096];
        assert_eq!(gunzip_transparent(&gzip_compress(&data, 0)).unwrap(), data);
        assert_eq!(
            gunzip_transparent(&gzip_compress(&data, 100)).unwrap(),
            data
        );
        assert_ne!(gzip_compress(&data, 0), gzip_compress(&data, 9));
        assert_eq!(gzip_compress(&data, 0), gzip_compress(&data, 1));
        assert_eq!(gzip_compress(&data, 100), gzip_compress(&data, 9));
    }

    #[test]
    fn inflate_auto_accepts_a_gzip_stream_too() {
        // `inflateInit2(&d, 47)` auto-detects, so a MetaImage whose payload is
        // gzip rather than zlib reads fine.
        let data = b"MetaIO reads either wrapper".to_vec();
        let g = gzip_compress(&data, 2);
        assert_eq!(inflate_auto(&g, data.len()).unwrap(), data);
    }

    #[test]
    fn inflate_auto_accepts_a_hand_built_stored_block_stream() {
        let data = b"stored, not deflated".to_vec();
        let fixture = stored_block_gzip(&data);
        assert_eq!(inflate_auto(&fixture, data.len()).unwrap(), data);
        assert_eq!(gunzip_transparent(&fixture).unwrap(), data);
    }

    #[test]
    fn inflate_auto_rejects_garbage_where_upstream_returns_uninitialised_memory() {
        let err = inflate_auto(b"ObjectType = Image\n", 8).unwrap_err();
        assert!(matches!(err, IoError::CorruptCompressedData(_)), "{err:?}");
    }

    #[test]
    fn inflate_auto_rejects_a_short_stream() {
        let z = zlib_compress(b"four", 2);
        assert!(matches!(
            inflate_auto(&z, 8).unwrap_err(),
            IoError::TruncatedData
        ));
    }

    #[test]
    fn inflate_auto_ignores_bytes_past_the_expected_length() {
        let z = zlib_compress(b"abcdefgh", 2);
        assert_eq!(inflate_auto(&z, 3).unwrap(), b"abc");
    }

    #[test]
    fn gunzip_of_a_non_gzip_stream_is_a_transparent_copy() {
        // `s->transparent = 1` (gzio.c:516) / zlib's `state->how = COPY`.
        assert_eq!(gunzip_transparent(b"raw bytes").unwrap(), b"raw bytes");
        // Including a *zlib* stream, which has no gzip magic: the bytes come
        // back deflated.
        let z = zlib_compress(b"payload", 2);
        assert_eq!(gunzip_transparent(&z).unwrap(), z);
    }

    #[test]
    fn gunzip_concatenated_members_are_all_decoded() {
        // Both `gzio.c`'s reader and zlib's `gzread` restart on a new member.
        let mut both = gzip_compress(b"first", 2);
        both.extend_from_slice(&gzip_compress(b"second", 2));
        assert_eq!(gunzip_transparent(&both).unwrap(), b"firstsecond");
    }

    #[test]
    fn gunzip_prefix_stops_early() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 253) as u8).collect();
        let g = gzip_compress(&data, 2);
        assert_eq!(gunzip_transparent_prefix(&g, 348).unwrap(), data[..348]);
        assert_eq!(gunzip_transparent_prefix(b"short", 348).unwrap(), b"short");
    }
}
