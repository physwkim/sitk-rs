//! Compressed MetaImage, gzip NRRD, and `.gz` NIfTI-1.
//!
//! Byte-for-byte parity with what upstream's zlib emits is *not* achievable —
//! miniz_oxide and zlib make different block and match choices at the same
//! nominal level — so what is pinned here is:
//!
//! 1. this port's writer feeding this port's reader, exactly, for every scalar
//!    `PixelId` in all three formats;
//! 2. this port's reader accepting a stream of purely *upstream* shape, built
//!    by hand out of stored deflate blocks ([`stored_block_gzip`],
//!    [`stored_block_zlib`]) so that no byte of the fixture came from `flate2`;
//! 3. the header text a compressed write produces, which *is* deterministic and
//!    is upstream's byte for byte apart from the deflate payload's length.

use sitk_core::{Complex, Image, PixelId};
use sitk_io::{
    FileMode, ImageFileWriter, IoError, create_image_io, read_image, write_image, write_image_with,
};

fn tmp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sitk_io_comp_{}_{name}", std::process::id()));
    p
}

// ---------------------------------------------------------------------------
// Fixtures of upstream shape, owing nothing to the crate under test
// ---------------------------------------------------------------------------

/// CRC-32, the bit-at-a-time way.
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

/// Adler-32, the zlib trailer's checksum.
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// One BFINAL stored (`BTYPE = 00`) deflate block: `01`, `LEN`, `~LEN`, bytes.
fn stored_deflate_block(data: &[u8]) -> Vec<u8> {
    assert!(data.len() < 0xffff, "one stored block holds < 64 KiB");
    let mut out = vec![0x01];
    out.extend_from_slice(&(data.len() as u16).to_le_bytes());
    out.extend_from_slice(&(!(data.len() as u16)).to_le_bytes());
    out.extend_from_slice(data);
    out
}

/// A gzip stream `nrrd__GzWrite` at `zlibLevel = 0` would emit: the ten-byte
/// header with `MTIME = 0`, `XFL = 0`, `OS = 3`; a stored block; CRC-32 and
/// ISIZE.
fn stored_block_gzip(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0x03];
    out.extend_from_slice(&stored_deflate_block(data));
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// A zlib stream `MET_PerformCompression` at level 0 would emit: `CMF = 0x78`
/// (deflate, 32 KiB window), `FLG = 0x01` (so `CMF*256 + FLG` is a multiple of
/// 31); a stored block; Adler-32, big-endian.
fn stored_block_zlib(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    out.extend_from_slice(&stored_deflate_block(data));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[test]
fn the_hand_built_fixtures_have_the_headers_upstream_writes() {
    assert_eq!(
        &stored_block_gzip(b"x")[..10],
        &[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0x03]
    );
    let zlib = stored_block_zlib(b"x");
    assert_eq!(&zlib[..2], &[0x78, 0x01]);
    assert_eq!(
        (u16::from(zlib[0]) * 256 + u16::from(zlib[1])) % 31,
        0,
        "zlib header check bits"
    );
}

// ---------------------------------------------------------------------------
// MetaImage: CompressedData = True
// ---------------------------------------------------------------------------

#[test]
fn mha_compressed_round_trips_every_scalar_pixel_id() {
    macro_rules! case {
        ($ty:ty, $name:expr) => {{
            let data: Vec<$ty> = (0..24u32).map(|i| i as $ty).collect();
            let img = Image::from_vec(&[4, 3, 2], data.clone()).unwrap();
            let path = tmp_path($name);
            write_image_with(&img, &path, true, -1).unwrap();
            let back = read_image(&path).unwrap();
            std::fs::remove_file(&path).ok();
            assert_eq!(back.size(), &[4, 3, 2], $name);
            assert_eq!(back.scalar_slice::<$ty>().unwrap(), data.as_slice(), $name);
        }};
    }
    case!(u8, "u8.mha");
    case!(i8, "i8.mha");
    case!(u16, "u16.mha");
    case!(i16, "i16.mha");
    case!(u32, "u32.mha");
    case!(i32, "i32.mha");
    case!(u64, "u64.mha");
    case!(i64, "i64.mha");
    case!(f32, "f32.mha");
    case!(f64, "f64.mha");
}

#[test]
fn mha_compressed_round_trips_vector_and_complex_images() {
    let vector =
        Image::from_vec_vector(&[2, 2], 3, (0..12u32).map(|i| i as f32).collect()).unwrap();
    let path = tmp_path("vector.mha");
    write_image_with(&vector, &path, true, -1).unwrap();
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.pixel_id(), PixelId::VectorFloat32);
    assert_eq!(
        back.component_slice::<f32>().unwrap(),
        vector.component_slice::<f32>().unwrap()
    );

    // A complex image reads back as a two-channel vector, compressed or not —
    // MetaIO has no complex element type (module docs).
    let complex = Image::from_vec_complex(
        &[2, 2],
        (0..4)
            .map(|i| Complex::new(i as f64, -(i as f64)))
            .collect(),
    )
    .unwrap();
    let path = tmp_path("complex.mha");
    write_image_with(&complex, &path, true, -1).unwrap();
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.pixel_id(), PixelId::VectorFloat64);
    assert_eq!(
        back.component_slice::<f64>().unwrap(),
        complex.component_slice::<f64>().unwrap()
    );
}

/// The header a compressed `.mha` carries: `CompressedData = True` where the
/// uncompressed one says `False`, followed by `CompressedDataSize`
/// (metaObject.cxx:1421-1432). Every other line is unchanged, and the payload
/// is a *zlib* stream — `deflateInit`, not gzip (metaUtils.cxx:808).
#[test]
fn mha_compressed_header_is_upstreams_and_the_payload_is_zlib() {
    let img = Image::from_vec(&[2, 2], vec![9u8; 4]).unwrap();
    let path = tmp_path("header.mha");
    write_image_with(&img, &path, true, -1).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();

    let text = String::from_utf8_lossy(&bytes);
    let (header, _) = text.split_once("ElementDataFile = LOCAL\n").unwrap();
    let size_line = header
        .lines()
        .find(|l| l.starts_with("CompressedDataSize = "))
        .expect("CompressedDataSize is written");
    let declared: usize = size_line
        .trim_start_matches("CompressedDataSize = ")
        .parse()
        .unwrap();

    let expected_header = format!(
        "ObjectType = Image\n\
         NDims = 2\n\
         BinaryData = True\n\
         BinaryDataByteOrderMSB = False\n\
         CompressedData = True\n\
         {size_line}\n\
         TransformMatrix = 1 0 0 1\n\
         Offset = 0 0\n\
         ElementSpacing = 1 1\n\
         DimSize = 2 2\n\
         ElementNumberOfChannels = 1\n\
         ElementType = MET_UCHAR\n"
    );
    assert_eq!(header, expected_header);

    let prefix = expected_header.len() + "ElementDataFile = LOCAL\n".len();
    let payload = &bytes[prefix..];
    assert_eq!(
        payload.len(),
        declared,
        "CompressedDataSize is the payload length"
    );
    assert_eq!(payload[0], 0x78, "zlib CMF byte, not gzip's 0x1f");
}

/// `MET_SetFileSuffix(m_ElementDataFileName, "zraw")` (metaImage.cxx:1588).
#[test]
fn mhd_compressed_writes_a_zraw_sibling() {
    let img = Image::from_vec(&[3, 2], (0..6u16).collect()).unwrap();
    let mhd = tmp_path("detached.mhd");
    write_image_with(&img, &mhd, true, -1).unwrap();

    let header = std::fs::read_to_string(&mhd).unwrap();
    let zraw = mhd.with_file_name(format!("sitk_io_comp_{}_detached.zraw", std::process::id()));
    assert!(
        header.contains(&format!(
            "ElementDataFile = sitk_io_comp_{}_detached.zraw\n",
            std::process::id()
        )),
        "{header}"
    );
    assert!(zraw.exists(), "the sibling is .zraw, not .raw");

    let back = read_image(&mhd).unwrap();
    std::fs::remove_file(&mhd).ok();
    std::fs::remove_file(&zraw).ok();
    assert_eq!(back.scalar_slice::<u16>().unwrap(), &[0, 1, 2, 3, 4, 5]);
}

/// `inflateInit2(&d, 47)` accepts a zlib *or* a gzip header
/// (metaUtils.cxx:862), so both read. Neither fixture came from this crate's
/// encoder.
#[test]
fn mha_reads_a_hand_built_zlib_and_a_hand_built_gzip_payload() {
    for (name, wrap) in [
        (
            "zlib_fixture.mha",
            stored_block_zlib as fn(&[u8]) -> Vec<u8>,
        ),
        (
            "gzip_fixture.mha",
            stored_block_gzip as fn(&[u8]) -> Vec<u8>,
        ),
    ] {
        let pixels: [u8; 4] = [11, 22, 33, 44];
        let payload = wrap(&pixels);
        let header = format!(
            "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             BinaryDataByteOrderMSB = False\n\
             CompressedData = True\n\
             CompressedDataSize = {}\n\
             DimSize = 2 2\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LOCAL\n",
            payload.len()
        );
        let mut bytes = header.into_bytes();
        bytes.extend_from_slice(&payload);
        let path = tmp_path(name);
        std::fs::write(&path, &bytes).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &pixels, "{name}");
    }
}

/// Upstream, `CompressedDataSize` absent from a `LOCAL` header makes
/// `M_ReadElements` guess the whole file's size, seek to offset `0`, and hand
/// the *header text* to inflate; `MET_PerformUncompression` prints "Uncompress
/// failed" and returns `true` over an uninitialised buffer
/// (metaImage.cxx:2616-2632, metaUtils.cxx:879-892). This port refuses instead
/// (ledger §1.56, §4.75).
#[test]
fn mha_local_without_compressed_data_size_is_refused_not_read_as_garbage() {
    let path = tmp_path("nosize.mha");
    std::fs::write(
        &path,
        b"ObjectType = Image\nNDims = 2\nBinaryData = True\nCompressedData = True\n\
          DimSize = 2 2\nElementType = MET_UCHAR\nElementDataFile = LOCAL\n\x78\x01\x00",
    )
    .unwrap();
    let err = read_image(&path).unwrap_err();
    std::fs::remove_file(&path).ok();
    match err {
        IoError::Unsupported(message) => {
            assert!(message.contains("CompressedDataSize"), "{message}");
            assert!(message.contains("1.52"), "{message}");
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

/// A detached data file *is* the compressed data, so upstream's "guess the size
/// from the file length" arm is right there and this port takes it too.
#[test]
fn mhd_without_compressed_data_size_uses_the_zraw_file_length() {
    let payload = stored_block_zlib(&[7u8, 8, 9, 10]);
    let mhd = tmp_path("guess.mhd");
    let zraw = tmp_path("guess.zraw");
    std::fs::write(&zraw, &payload).unwrap();
    std::fs::write(
        &mhd,
        format!(
            "ObjectType = Image\nNDims = 2\nBinaryData = True\nCompressedData = True\n\
             DimSize = 2 2\nElementType = MET_UCHAR\nElementDataFile = sitk_io_comp_{}_guess.zraw\n",
            std::process::id()
        ),
    )
    .unwrap();
    let back = read_image(&mhd).unwrap();
    std::fs::remove_file(&mhd).ok();
    std::fs::remove_file(&zraw).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[7, 8, 9, 10]);
}

/// A `LIST` header inflates each slice on its own, because each goes through
/// `M_ReadElements` separately (metaImage.cxx:1375-1382).
#[test]
fn mha_list_of_compressed_slices_inflates_each_slice() {
    let slice0 = stored_block_zlib(&[1u8, 2]);
    let slice1 = stored_block_zlib(&[3u8, 4]);
    let s0 = tmp_path("list0.zraw");
    let s1 = tmp_path("list1.zraw");
    std::fs::write(&s0, &slice0).unwrap();
    std::fs::write(&s1, &slice1).unwrap();

    let pid = std::process::id();
    let mhd = tmp_path("list.mhd");
    std::fs::write(
        &mhd,
        format!(
            "ObjectType = Image\nNDims = 2\nBinaryData = True\nCompressedData = True\n\
             DimSize = 2 2\nElementType = MET_UCHAR\nElementDataFile = LIST\n\
             sitk_io_comp_{pid}_list0.zraw\nsitk_io_comp_{pid}_list1.zraw\n"
        ),
    )
    .unwrap();
    let back = read_image(&mhd).unwrap();
    for p in [&mhd, &s0, &s1] {
        std::fs::remove_file(p).ok();
    }
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
}

#[test]
fn mha_corrupt_compressed_payload_is_an_error() {
    let path = tmp_path("corrupt.mha");
    std::fs::write(
        &path,
        b"ObjectType = Image\nNDims = 2\nBinaryData = True\nCompressedData = True\n\
          CompressedDataSize = 4\nDimSize = 2 2\nElementType = MET_UCHAR\n\
          ElementDataFile = LOCAL\nJUNK",
    )
    .unwrap();
    let err = read_image(&path).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(matches!(err, IoError::CorruptCompressedData(_)), "{err:?}");
}

/// `CompressedDataSize` larger than the bytes on disk is `M_ReadElementData`'s
/// `gc != _dataQuantity` failure (metaImage.cxx:3583-3588).
#[test]
fn mha_compressed_data_size_beyond_the_file_is_truncation() {
    let path = tmp_path("shortsize.mha");
    let payload = stored_block_zlib(&[1u8, 2, 3, 4]);
    let mut bytes = format!(
        "ObjectType = Image\nNDims = 2\nBinaryData = True\nCompressedData = True\n\
         CompressedDataSize = {}\nDimSize = 2 2\nElementType = MET_UCHAR\n\
         ElementDataFile = LOCAL\n",
        payload.len() + 100
    )
    .into_bytes();
    bytes.extend_from_slice(&payload);
    std::fs::write(&path, &bytes).unwrap();
    let err = read_image(&path).unwrap_err();
    std::fs::remove_file(&path).ok();
    assert!(matches!(err, IoError::TruncatedData), "{err:?}");
}

// ---------------------------------------------------------------------------
// NRRD: encoding: gzip
// ---------------------------------------------------------------------------

#[test]
fn nrrd_gzip_round_trips_every_scalar_pixel_id() {
    macro_rules! case {
        ($ty:ty, $name:expr) => {{
            let data: Vec<$ty> = (0..24u32).map(|i| i as $ty).collect();
            let img = Image::from_vec(&[4, 3, 2], data.clone()).unwrap();
            let path = tmp_path($name);
            write_image_with(&img, &path, true, -1).unwrap();
            let back = read_image(&path).unwrap();
            std::fs::remove_file(&path).ok();
            assert_eq!(back.size(), &[4, 3, 2], $name);
            assert_eq!(back.scalar_slice::<$ty>().unwrap(), data.as_slice(), $name);
        }};
    }
    case!(u8, "u8.nrrd");
    case!(i8, "i8.nrrd");
    case!(u16, "u16.nrrd");
    case!(i16, "i16.nrrd");
    case!(u32, "u32.nrrd");
    case!(i32, "i32.nrrd");
    case!(u64, "u64.nrrd");
    case!(i64, "i64.nrrd");
    case!(f32, "f32.nrrd");
    case!(f64, "f64.nrrd");
}

/// NRRD's `kinds` field records "this was complex", so unlike MetaImage the
/// round trip is closed — compressed or not.
#[test]
fn nrrd_gzip_round_trips_complex_as_complex() {
    let img = Image::from_vec_complex(
        &[2, 2],
        (0..4)
            .map(|i| Complex::new(i as f32, -(i as f32)))
            .collect(),
    )
    .unwrap();
    let path = tmp_path("complex.nrrd");
    write_image_with(&img, &path, true, -1).unwrap();
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.pixel_id(), PixelId::ComplexFloat32);
    assert_eq!(
        back.component_slice::<f32>().unwrap(),
        img.component_slice::<f32>().unwrap()
    );
}

/// `nio->encoding = nrrdEncodingGzip` gives `encoding: gzip`, and the payload
/// is a gzip stream directly after the blank line.
#[test]
fn nrrd_gzip_header_names_the_encoding_and_the_payload_is_gzip() {
    let img = Image::from_vec(&[2, 2], vec![5u16; 4]).unwrap();
    let path = tmp_path("encoding.nrrd");
    write_image_with(&img, &path, true, -1).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();

    let text = String::from_utf8_lossy(&bytes);
    let header_end = text.find("\n\n").unwrap() + 2;
    let header = &text[..header_end];
    assert!(header.contains("\nencoding: gzip\n"), "{header}");
    // `endianMatters` is set for gzip, so the endian line survives.
    assert!(header.contains("\nendian: little\n"), "{header}");
    assert_eq!(&bytes[header_end..header_end + 2], b"\x1f\x8b");
    assert_eq!(
        &bytes[header_end + 8..header_end + 10],
        &[0x00, 0x03],
        "XFL, OS"
    );
}

/// `nrrd__FprintFieldInfo` names the data file `<base>.<encoding->suffix>`,
/// and `nrrdEncodingGzip->suffix` is `"raw.gz"` (encodingGzip.c:302).
#[test]
fn nhdr_gzip_writes_a_raw_gz_sibling() {
    let img = Image::from_vec(&[3, 2], (0..6i32).collect()).unwrap();
    let nhdr = tmp_path("detached.nhdr");
    write_image_with(&img, &nhdr, true, -1).unwrap();

    let header = std::fs::read_to_string(&nhdr).unwrap();
    let pid = std::process::id();
    assert!(
        header.contains(&format!("data file: sitk_io_comp_{pid}_detached.raw.gz\n")),
        "{header}"
    );
    let raw_gz = tmp_path("detached.raw.gz");
    assert!(raw_gz.exists());

    let back = read_image(&nhdr).unwrap();
    std::fs::remove_file(&nhdr).ok();
    std::fs::remove_file(&raw_gz).ok();
    assert_eq!(back.scalar_slice::<i32>().unwrap(), &[0, 1, 2, 3, 4, 5]);
}

fn write_nrrd_with_gzip_payload(
    name: &str,
    extra_fields: &str,
    payload: Vec<u8>,
) -> std::path::PathBuf {
    let path = tmp_path(name);
    let mut bytes = format!(
        "NRRD0004\ntype: unsigned char\ndimension: 1\nsizes: 4\nencoding: gzip\n{extra_fields}\n"
    )
    .into_bytes();
    bytes.extend_from_slice(&payload);
    std::fs::write(&path, &bytes).unwrap();
    path
}

#[test]
fn nrrd_reads_a_hand_built_stored_block_gzip_stream() {
    let path = write_nrrd_with_gzip_payload("fixture.nrrd", "", stored_block_gzip(&[1u8, 2, 3, 4]));
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
}

/// `encodingGzip_read` skips bytes inside the *decompressed* stream
/// (encodingGzip.c:146-155), not in the file — the file skip is suppressed for
/// compression encodings (formatNRRD.c:583-585).
#[test]
fn nrrd_gzip_byte_skip_applies_to_the_decompressed_stream() {
    let path = write_nrrd_with_gzip_payload(
        "byteskip.nrrd",
        "byte skip: 3\n",
        stored_block_gzip(&[99u8, 98, 97, 1, 2, 3, 4]),
    );
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
}

/// A negative `byte skip` is *legal* with gzip, where `nrrd__ByteSkipSkip`
/// would have rejected it for every encoding but raw: the gzip decoder handles
/// it itself, with `backwards = -byteSkip - 1` bytes ignored after the data
/// (encodingGzip.c:132-140).
#[test]
fn nrrd_gzip_negative_byte_skip_takes_the_tail_of_the_inflated_stream() {
    let path = write_nrrd_with_gzip_payload(
        "tail.nrrd",
        "byte skip: -1\n",
        stored_block_gzip(&[0xaa, 0xbb, 1, 2, 3, 4]),
    );
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);

    // `byte skip: -3` ignores 2 trailing bytes as well.
    let path = write_nrrd_with_gzip_payload(
        "tail3.nrrd",
        "byte skip: -3\n",
        stored_block_gzip(&[0xaa, 1, 2, 3, 4, 0xee, 0xff]),
    );
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
}

/// `nrrdLineSkip` runs on the *file* stream, before the gzip magic
/// (formatNRRD.c:579-582).
#[test]
fn nrrd_gzip_line_skip_applies_to_the_file_stream() {
    let mut payload = b"skip me\n".to_vec();
    payload.extend_from_slice(&stored_block_gzip(&[1u8, 2, 3, 4]));
    let path = write_nrrd_with_gzip_payload("lineskip.nrrd", "line skip: 1\n", payload);
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
}

/// `check_header` finds no gzip magic and sets `s->transparent = 1`
/// (gzio.c:516), so the raw bytes come straight through (ledger §2.113).
#[test]
fn nrrd_gzip_over_a_non_gzip_stream_reads_transparently() {
    let path = write_nrrd_with_gzip_payload("transparent.nrrd", "", vec![1u8, 2, 3, 4]);
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
}

/// Big-endian data under gzip is swapped after inflating, because
/// `nrrdEncodingGzip->endianMatters` is set (formatNRRD.c:642-651).
#[test]
fn nrrd_gzip_big_endian_data_is_swapped() {
    let path = tmp_path("bigendian.nrrd");
    let payload = stored_block_gzip(&[0x01, 0x02, 0x03, 0x04]);
    let mut bytes =
        b"NRRD0004\ntype: short\ndimension: 1\nsizes: 2\nencoding: gzip\nendian: big\n\n".to_vec();
    bytes.extend_from_slice(&payload);
    std::fs::write(&path, &bytes).unwrap();
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.scalar_slice::<i16>().unwrap(), &[0x0102, 0x0304]);
}

// ---------------------------------------------------------------------------
// NIfTI: .nii.gz, .hdr.gz, .img.gz
// ---------------------------------------------------------------------------

#[test]
fn nii_gz_round_trips_every_scalar_pixel_id() {
    macro_rules! case {
        ($ty:ty, $name:expr) => {{
            let data: Vec<$ty> = (0..24u32).map(|i| i as $ty).collect();
            let img = Image::from_vec(&[4, 3, 2], data.clone()).unwrap();
            let path = tmp_path($name);
            write_image(&img, &path).unwrap();
            let back = read_image(&path).unwrap();
            std::fs::remove_file(&path).ok();
            assert_eq!(back.size(), &[4, 3, 2], $name);
            assert_eq!(back.scalar_slice::<$ty>().unwrap(), data.as_slice(), $name);
        }};
    }
    case!(u8, "u8.nii.gz");
    case!(i8, "i8.nii.gz");
    case!(u16, "u16.nii.gz");
    case!(i16, "i16.nii.gz");
    case!(u32, "u32.nii.gz");
    case!(i32, "i32.nii.gz");
    case!(u64, "u64.nii.gz");
    case!(i64, "i64.nii.gz");
    case!(f32, "f32.nii.gz");
    case!(f64, "f64.nii.gz");
}

#[test]
fn nii_gz_round_trips_complex_and_preserves_geometry() {
    let mut img = Image::from_vec_complex(
        &[2, 2, 2],
        (0..8)
            .map(|i| Complex::new(i as f32, -(i as f32)))
            .collect(),
    )
    .unwrap();
    img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
    img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();

    let path = tmp_path("complex.nii.gz");
    write_image(&img, &path).unwrap();
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.pixel_id(), PixelId::ComplexFloat32);
    assert_eq!(back.spacing(), &[0.5, 1.25, 3.0]);
    assert_eq!(back.origin(), &[-2.0, 4.0, 7.5]);
    assert_eq!(
        back.component_slice::<f32>().unwrap(),
        img.component_slice::<f32>().unwrap()
    );
}

/// `nifti_makehdrname` / `nifti_makeimgname` put the `.gz` back on both names,
/// and `nifti_image_write_engine` closes the header's `znzFile` before opening
/// the image's — so these are two independent gzip streams.
#[test]
fn hdr_gz_and_img_gz_are_two_separate_gzip_streams() {
    let img = Image::from_vec(&[2, 2], vec![10u8, 20, 30, 40]).unwrap();
    let hdr = tmp_path("pair.hdr.gz");
    let dat = tmp_path("pair.img.gz");
    write_image(&img, &hdr).unwrap();

    assert!(hdr.exists() && dat.exists());
    assert_eq!(&std::fs::read(&hdr).unwrap()[..2], b"\x1f\x8b");
    assert_eq!(&std::fs::read(&dat).unwrap()[..2], b"\x1f\x8b");

    let back = read_image(&hdr).unwrap();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[10, 20, 30, 40]);

    // `nifti_findhdrname` resolves `.img.gz` to the `.hdr.gz` beside it.
    let back = read_image(&dat).unwrap();
    std::fs::remove_file(&hdr).ok();
    std::fs::remove_file(&dat).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[10, 20, 30, 40]);
}

/// Writing `.img.gz` names the header `.hdr.gz`, per `nifti_makehdrname`.
#[test]
fn writing_img_gz_produces_the_hdr_gz_beside_it() {
    let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
    let dat = tmp_path("viaimg.img.gz");
    let hdr = tmp_path("viaimg.hdr.gz");
    write_image(&img, &dat).unwrap();
    let exists = (hdr.exists(), dat.exists());
    let back = read_image(&hdr).unwrap();
    std::fs::remove_file(&hdr).ok();
    std::fs::remove_file(&dat).ok();
    assert_eq!(exists, (true, true));
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
}

/// NIfTI compresses on the extension alone. `use_compression = true` on a
/// `.nii` writes plain bytes, and `use_compression = false` on a `.nii.gz`
/// writes gzip anyway (itkNiftiImageIO.cxx:1173; ledger §3.40).
#[test]
fn nifti_ignores_use_compression_entirely() {
    let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();

    let plain = tmp_path("forced.nii");
    write_image_with(&img, &plain, true, 9).unwrap();
    let head = std::fs::read(&plain).unwrap();
    assert_ne!(&head[..2], b"\x1f\x8b", "a .nii is never gzipped");
    assert_eq!(
        read_image(&plain).unwrap().scalar_slice::<u8>().unwrap(),
        &[1, 2, 3, 4]
    );
    std::fs::remove_file(&plain).ok();

    let gz = tmp_path("forced.nii.gz");
    write_image_with(&img, &gz, false, -1).unwrap();
    let head = std::fs::read(&gz).unwrap();
    assert_eq!(&head[..2], b"\x1f\x8b", "a .nii.gz is always gzipped");
    // MTIME 0, XFL 0 (level 6), OS 3 — as zlib's `gzopen(path, "wb")` stamps.
    assert_eq!(&head[4..10], &[0, 0, 0, 0, 0x00, 0x03]);
    assert_eq!(
        read_image(&gz).unwrap().scalar_slice::<u8>().unwrap(),
        &[1, 2, 3, 4]
    );
    std::fs::remove_file(&gz).ok();
}

/// `zlib`'s `gz_look` copies through when the magic is absent, so a `.nii.gz`
/// holding an uncompressed NIfTI reads (ledger §2.113). The converse also holds:
/// `nifti_is_gzfile` is a name test, so a gzip stream named `.nii` is not
/// gunzipped and fails to parse.
#[test]
fn nifti_gz_reading_is_transparent_and_gz_detection_is_by_name() {
    let img = Image::from_vec(&[2, 2], vec![4u8, 3, 2, 1]).unwrap();

    let plain = tmp_path("plain.nii");
    write_image(&img, &plain).unwrap();
    let plain_bytes = std::fs::read(&plain).unwrap();
    std::fs::remove_file(&plain).ok();

    let misnamed_gz = tmp_path("misnamed.nii.gz");
    std::fs::write(&misnamed_gz, &plain_bytes).unwrap();
    assert!(create_image_io(&misnamed_gz, FileMode::Read).is_some());
    let back = read_image(&misnamed_gz).unwrap();
    std::fs::remove_file(&misnamed_gz).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[4, 3, 2, 1]);

    let gz_named_nii = tmp_path("gzipped.nii");
    std::fs::write(&gz_named_nii, stored_block_gzip(&plain_bytes)).unwrap();
    let claimed = create_image_io(&gz_named_nii, FileMode::Read).is_some();
    std::fs::remove_file(&gz_named_nii).ok();
    assert!(!claimed, "a gzip stream named .nii is not a NIfTI file");
}

/// A `.nii.gz` this port never wrote — one stored deflate block over the whole
/// 352-byte header block and the pixels — reads, because that is what
/// `znzread` would deliver.
#[test]
fn nii_reads_a_hand_built_stored_block_gzip_file() {
    let img = Image::from_vec(&[2, 2], vec![7u8, 6, 5, 4]).unwrap();
    let plain = tmp_path("source.nii");
    write_image(&img, &plain).unwrap();
    let plain_bytes = std::fs::read(&plain).unwrap();
    std::fs::remove_file(&plain).ok();

    let path = tmp_path("fixture.nii.gz");
    std::fs::write(&path, stored_block_gzip(&plain_bytes)).unwrap();
    let back = read_image(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.scalar_slice::<u8>().unwrap(), &[7, 6, 5, 4]);
}

// ---------------------------------------------------------------------------
// The write-side opt-in: ImageFileWriter and write_image
// ---------------------------------------------------------------------------

#[test]
fn image_file_writer_exposes_use_compression_and_compression_level() {
    let img = Image::from_vec(&[64, 64], vec![7u8; 4096]).unwrap();

    let mut writer = ImageFileWriter::new();
    assert!(!writer.use_compression(), "SimpleITK's default is false");
    assert_eq!(writer.compression_level(), -1, "SimpleITK's default is -1");

    let path = tmp_path("writer.mha");
    writer.set_file_name(&path).use_compression_on();
    assert!(writer.use_compression());
    writer.execute(&img).unwrap();
    let compressed = std::fs::read(&path).unwrap();
    assert!(String::from_utf8_lossy(&compressed).contains("CompressedData = True\n"));

    writer.use_compression_off();
    writer.execute(&img).unwrap();
    let plain = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert!(String::from_utf8_lossy(&plain).contains("CompressedData = False\n"));
    assert!(
        compressed.len() < plain.len(),
        "4096 constant bytes deflate well: {} vs {}",
        compressed.len(),
        plain.len()
    );
}

/// `Execute(image, fileName, useCompression, compressionLevel)` sets all three
/// properties on the writer before running it (sitkImageFileWriter.cxx:161-167),
/// so they persist into the next `Execute`.
#[test]
fn image_file_writer_execute_with_sets_all_three_and_they_persist() {
    let img = Image::from_vec(&[4, 4], (0..16u8).collect()).unwrap();
    let path = tmp_path("execwith.mha");
    let mut writer = ImageFileWriter::new();
    writer.execute_with(&img, &path, true, 9).unwrap();
    assert_eq!(writer.file_name(), path);
    assert!(writer.use_compression(), "SetUseCompression persisted");
    assert_eq!(
        writer.compression_level(),
        9,
        "SetCompressionLevel persisted"
    );

    let bytes = std::fs::read(&path).unwrap();
    let back = read_image(&path).unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("CompressedData = True\n"));
    assert_eq!(
        back.scalar_slice::<u8>().unwrap(),
        (0..16u8).collect::<Vec<_>>().as_slice()
    );

    // The bare `Execute` now reuses the persisted file name and both knobs.
    writer.execute(&img).unwrap();
    let again = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(again, bytes);
}

/// `itkSetClampMacro(CompressionLevel, int, 1, 9)`: a level below 1 or above 9
/// is clamped rather than rejected, so both extremes write a readable file, and
/// `0` produces the same bytes as `1`, `100` the same as `9`.
#[test]
fn compression_level_is_clamped_not_rejected() {
    let img = Image::from_vec(&[16, 16], (0..256u32).map(|i| (i % 17) as u8).collect()).unwrap();
    let mut bytes = Vec::new();
    for level in [-1, 0, 1, 2, 9, 100] {
        let path = tmp_path(&format!("level_{}.mha", level.max(0)));
        write_image_with(&img, &path, true, level).unwrap();
        let written = std::fs::read(&path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(
            back.scalar_slice::<u8>().unwrap().len(),
            256,
            "level {level} round-trips"
        );
        bytes.push((level, written));
    }
    let get = |lvl: i32| &bytes.iter().find(|(l, _)| *l == lvl).unwrap().1;
    assert_eq!(get(0), get(1), "level 0 clamps up to 1");
    assert_eq!(get(100), get(9), "level 100 clamps down to 9");
    assert_eq!(
        get(-1),
        get(2),
        "level -1 leaves MetaImageIO on its default of 2"
    );
    assert_ne!(get(1), get(9), "the levels are actually distinct");
}

/// A level of `-1` leaves `NrrdImageIO` on *its* default of 2 as well.
#[test]
fn nrrd_default_compression_level_is_two() {
    let img = Image::from_vec(&[16, 16], (0..256u32).map(|i| (i % 17) as u8).collect()).unwrap();
    let default = tmp_path("default.nrrd");
    let explicit = tmp_path("explicit.nrrd");
    write_image_with(&img, &default, true, -1).unwrap();
    write_image_with(&img, &explicit, true, 2).unwrap();
    let a = std::fs::read(&default).unwrap();
    let b = std::fs::read(&explicit).unwrap();
    std::fs::remove_file(&default).ok();
    std::fs::remove_file(&explicit).ok();
    assert_eq!(a, b);
}

/// `write_image` is `WriteImage(image, fileName)` — the two defaulted arguments
/// are `false` and `-1`, so it must produce exactly what `write_image_with`
/// does with those.
#[test]
fn write_image_is_write_image_with_the_defaults() {
    let img = Image::from_vec(&[4, 4], (0..16u8).collect()).unwrap();
    for name in ["defaults.mha", "defaults.nrrd", "defaults.nii"] {
        let a = tmp_path(&format!("a_{name}"));
        let b = tmp_path(&format!("b_{name}"));
        write_image(&img, &a).unwrap();
        write_image_with(&img, &b, false, -1).unwrap();
        let (ab, bb) = (std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
        assert_eq!(ab, bb, "{name}");
    }
}
