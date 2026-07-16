//! MetaImage (`.mha` / `.mhd` + `.raw`) reader and writer — `itk::MetaImageIO`.
//!
//! MetaImage is ITK's native uncompressed format: a plain-text `Key = Value`
//! header followed by (or referencing) a raw binary pixel dump. It round-trips
//! every scalar, vector, and complex pixel type, arbitrary dimension, and the
//! full spacing / origin / direction geometry, which makes it the right
//! Phase-0 format for exercising the whole core model without pulling in an
//! external image crate.
//!
//! [`MetaImageIo`] is this crate's first [`ImageIo`]
//! implementor; [`read`] and [`write()`] are its free-function core.
//!
//! # Channels: scalar, vector, and complex
//!
//! MetaIO's on-disk channel count, `ElementNumberOfChannels`, is
//! [`Image::buffer_stride`] — `1` for scalar, `2` for complex, the vector
//! length for a vector image — not SimpleITK's
//! `GetNumberOfComponentsPerPixel()` (which is `1` for a complex image). This
//! matches upstream exactly: `itk::Image<T>::GetNumberOfComponentsPerPixel()`
//! is `NumericTraits<T>::GetLength()`, `2` for `std::complex<T>`
//! (itkNumericTraits.h:1958-1967), and `MetaImageIO::Write` passes that count
//! straight through to `ElementNumberOfChannels` (itkMetaImageIO.cxx:551,665;
//! metaImage.cxx:2341-2346).
//!
//! MetaIO has no complex element type, though: on read,
//! `MetaImageIO::ReadImageInformation` forces `IOPixelEnum::VECTOR` whenever
//! `ElementNumberOfChannels() > 1`, regardless of what wrote the file
//! (itkMetaImageIO.cxx:241-244), and SimpleITK's reader only takes the
//! scalar/complex branch when `NumberOfComponents == 1`
//! (sitkImageReaderBase.cxx:215-233). So a complex image written by [`write()`]
//! reads back — in real ITK/SimpleITK as much as in [`read`] here — as a
//! same-sized **vector** image: `PixelId::ComplexFloat32` round-trips to
//! `PixelId::VectorFloat32`, not back to itself. The interleaved `re, im`
//! bytes are preserved; the "this was complex" bit is not, because MetaIO
//! never recorded it.
//!
//! # Header fields
//!
//! `MET_Read` compares a header line's key against the registered field names
//! with `strcmp` (metaUtils.cxx:1191), so key matching is **case-sensitive**,
//! and the registered set is exactly [`RECOGNIZED_FIELDS`]
//! (metaObject.cxx:1204-1306, metaImage.cxx:2144-2213). A key outside that set
//! is not an error: `MET_Read` stores it verbatim as a string in the object's
//! "additional read fields" (metaUtils.cxx:1398-1412), and
//! `MetaImageIO::ReadImageInformation` copies each one into the ITK meta-data
//! dictionary (itkMetaImageIO.cxx:280-287) — so custom tags survive a read.
//!
//! `ElementDataFile` is marked `terminateRead` (metaImage.cxx:2212), so parsing
//! stops there and anything after it is pixel data (`LOCAL`), a filename, or a
//! slice list — never another header field.
//!
//! Which value wins when a field has aliases is decided by MetaIO's *fixed
//! apply order* in `M_Read`, not by the order the lines appear in the file
//! (metaObject.cxx:1618-1707):
//!
//! * origin: `Offset`, then `Position`, then `Origin` — last write wins, so
//!   `Origin` beats `Position` beats `Offset`;
//! * direction: `Orientation`, then `Rotation`, then `TransformMatrix` — so
//!   `TransformMatrix` beats both;
//! * byte order: `ElementByteOrderMSB`, then `BinaryDataByteOrderMSB` — so
//!   `BinaryDataByteOrderMSB` beats `ElementByteOrderMSB`.
//!
//! `Position`, `Origin`, `Orientation` and `Rotation` are only consulted when
//! `FileFormatVersion` is `0`, its default (metaObject.cxx:1662).
//!
//! A boolean field is true iff its **first character** is `T`, `t`, or `1`
//! (metaObject.cxx:1586-1642) — not the string `"true"`. `BinaryData = 1` and
//! `CompressedData = TRUE` are both true; `CompressedData = yes` is false.
//!
//! # Byte order
//!
//! `MetaImageIO::Read` calls `ElementByteOrderFix` (itkMetaImageIO.cxx:348,359),
//! which swaps each component in place when the file's byte order differs from
//! the machine's (metaImage.cxx:845-852). Both `BinaryDataByteOrderMSB = True`
//! and `ElementByteOrderMSB = True` therefore select big-endian components on
//! read. [`read`] decodes with `from_be_bytes` instead of swapping, which is
//! the same result on every host.
//!
//! # `ElementDataFile` forms
//!
//! * `LOCAL` (case-insensitively `Local`/`local` only — metaImage.cxx:1311)
//!   puts the pixel data straight after the header line.
//! * `LIST` — see [`read`] — names one file per slice, on the header lines that
//!   follow.
//! * anything else is a single raw filename, resolved against the header's own
//!   directory.
//!
//! # Compressed data
//!
//! `CompressedData = True` marks the element data as a **zlib** stream —
//! `MET_PerformCompression` calls `deflateInit`, which writes an RFC-1950
//! wrapper (metaUtils.cxx:808). Reading goes through
//! `MET_PerformUncompression`'s `inflateInit2(&d, 47)`, which auto-detects zlib
//! *and* gzip, so a gzip payload also reads (metaUtils.cxx:862). See
//! [`crate::io::compression`].
//!
//! `CompressedDataSize` tells the reader how many bytes to hand to inflate. It
//! is written whenever it is non-zero (metaObject.cxx:1427-1432), so every
//! ITK-written compressed MetaImage carries it. When it is *absent*,
//! `M_ReadElements` guesses the compressed size as the whole file's size and
//! seeks back to offset `0` to read it (metaImage.cxx:2616-2622) — which for a
//! `LOCAL` `.mha` feeds the *header text* to inflate, gets `Z_DATA_ERROR`, and
//! then returns success over a `new`-ed buffer nobody wrote to. That is an
//! upstream bug (§1.56); this port refuses such a header (§4.75). A detached
//! `.zraw` has no header in the way, so there the guess is right and this port
//! guesses too.
//!
//! Not yet supported: MetaIO's `printf`-pattern slice form
//! (`ElementDataFile = slice%03d.raw 1 20 1`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::core::{Image, PixelBuffer, PixelId};

use crate::io::compression::{ITK_DEFAULT_COMPRESSION_LEVEL, inflate_auto, zlib_compress};
use crate::io::error::{IoError, Result};
use crate::io::image_io::{ImageInformation, ImageIo};
use crate::io::writer::WriteOptions;

/// Every header field name MetaIO registers for reading, in registration order:
/// `MetaObject::M_SetupReadFields` (metaObject.cxx:1204-1306) followed by
/// `MetaImage::M_SetupReadFields` (metaImage.cxx:2144-2213). Matched
/// case-sensitively, as `MET_Read`'s `strcmp` does. Anything else in the header
/// is an "additional read field" and lands in the meta-data dictionary.
pub const RECOGNIZED_FIELDS: &[&str] = &[
    // MetaObject
    "ObjectType",
    "ObjectSubType",
    "FileFormatVersion",
    "Comment",
    "AcquisitionDate",
    "NDims",
    "Name",
    "ID",
    "ParentID",
    "CompressedData",
    "CompressedDataSize",
    "BinaryData",
    "ElementByteOrderMSB",
    "BinaryDataByteOrderMSB",
    "Color",
    "Position",
    "Origin",
    "Offset",
    "TransformMatrix",
    "Rotation",
    "Orientation",
    "CenterOfRotation",
    "DistanceUnits",
    "AnatomicalOrientation",
    "ElementSpacing",
    // MetaImage
    "DimSize",
    "HeaderSize",
    "Modality",
    "ImagePosition",
    "ElementOrigin",
    "ElementDirection",
    "SequenceID",
    "ElementMin",
    "ElementMax",
    "ElementNumberOfChannels",
    "ElementSize",
    "ElementNBits",
    "ElementToIntensityFunctionSlope",
    "ElementToIntensityFunctionOffset",
    "ElementType",
    "ElementDataFile",
];

/// The six `MET_ImageModalityTypeName` strings (metaImageTypes.h:34-40).
const MODALITY_NAMES: &[&str] = &[
    "MET_MOD_CT",
    "MET_MOD_MR",
    "MET_MOD_NM",
    "MET_MOD_US",
    "MET_MOD_OTHER",
    "MET_MOD_UNKNOWN",
];

/// The `MET_DistanceUnitsTypeName` strings other than the unknown `"?"`
/// (metaTypes.h:168-171).
const DISTANCE_UNIT_NAMES: &[&str] = &["um", "mm", "cm"];

/// MetaIO's boolean rule: true iff the value's first character is `T`, `t`, or
/// `1` (metaObject.cxx:1586-1642). An empty value reads the field buffer's
/// terminating NUL, so it is false.
fn met_bool(value: &str) -> bool {
    matches!(value.as_bytes().first(), Some(b'T' | b't' | b'1'))
}

/// MetaIO `ElementType` string for a pixel id's *component* type — MetaIO names
/// the component type in `ElementType` and the count in
/// `ElementNumberOfChannels`. 64-bit integers use the width-explicit
/// `LONG_LONG` names so the file is unambiguous across platforms.
fn element_type(id: PixelId) -> &'static str {
    match id {
        PixelId::UInt8 | PixelId::VectorUInt8 => "MET_UCHAR",
        PixelId::Int8 | PixelId::VectorInt8 => "MET_CHAR",
        PixelId::UInt16 | PixelId::VectorUInt16 => "MET_USHORT",
        PixelId::Int16 | PixelId::VectorInt16 => "MET_SHORT",
        PixelId::UInt32 | PixelId::VectorUInt32 => "MET_UINT",
        PixelId::Int32 | PixelId::VectorInt32 => "MET_INT",
        PixelId::UInt64 | PixelId::VectorUInt64 => "MET_ULONG_LONG",
        PixelId::Int64 | PixelId::VectorInt64 => "MET_LONG_LONG",
        PixelId::Float32 | PixelId::ComplexFloat32 | PixelId::VectorFloat32 => "MET_FLOAT",
        PixelId::Float64 | PixelId::ComplexFloat64 | PixelId::VectorFloat64 => "MET_DOUBLE",
    }
}

/// Parse a MetaIO `ElementType`. `MET_LONG`/`MET_ULONG` are accepted as 64-bit
/// for interoperability with LP64 ITK writers.
fn parse_element_type(s: &str) -> Result<PixelId> {
    Ok(match s {
        "MET_UCHAR" => PixelId::UInt8,
        "MET_CHAR" => PixelId::Int8,
        "MET_USHORT" => PixelId::UInt16,
        "MET_SHORT" => PixelId::Int16,
        "MET_UINT" => PixelId::UInt32,
        "MET_INT" => PixelId::Int32,
        "MET_ULONG_LONG" | "MET_ULONG" => PixelId::UInt64,
        "MET_LONG_LONG" | "MET_LONG" => PixelId::Int64,
        "MET_FLOAT" => PixelId::Float32,
        "MET_DOUBLE" => PixelId::Float64,
        other => return Err(IoError::UnsupportedElementType(other.to_string())),
    })
}

fn buffer_to_le_bytes(buf: &PixelBuffer) -> Vec<u8> {
    macro_rules! pack {
        ($v:expr) => {{
            let mut out = Vec::with_capacity(std::mem::size_of_val(&$v[..]));
            for &x in $v.iter() {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }};
    }
    match buf {
        PixelBuffer::UInt8(v) => v.clone(),
        PixelBuffer::Int8(v) => v.iter().map(|&x| x as u8).collect(),
        PixelBuffer::UInt16(v) => pack!(v),
        PixelBuffer::Int16(v) => pack!(v),
        PixelBuffer::UInt32(v) => pack!(v),
        PixelBuffer::Int32(v) => pack!(v),
        PixelBuffer::UInt64(v) => pack!(v),
        PixelBuffer::Int64(v) => pack!(v),
        PixelBuffer::Float32(v) => pack!(v),
        PixelBuffer::Float64(v) => pack!(v),
    }
}

fn buffer_from_bytes(id: PixelId, bytes: &[u8], big_endian: bool) -> Result<PixelBuffer> {
    let expected = id.size_in_bytes();
    if !bytes.len().is_multiple_of(expected) {
        return Err(IoError::TruncatedData);
    }
    macro_rules! unpack {
        ($ty:ty, $variant:ident) => {{
            const S: usize = std::mem::size_of::<$ty>();
            let mut v = Vec::with_capacity(bytes.len() / S);
            for chunk in bytes.chunks_exact(S) {
                let arr: [u8; S] = chunk.try_into().expect("chunk size checked above");
                v.push(if big_endian {
                    <$ty>::from_be_bytes(arr)
                } else {
                    <$ty>::from_le_bytes(arr)
                });
            }
            PixelBuffer::$variant(v)
        }};
    }
    Ok(match id {
        PixelId::UInt8 | PixelId::VectorUInt8 => PixelBuffer::UInt8(bytes.to_vec()),
        PixelId::Int8 | PixelId::VectorInt8 => {
            PixelBuffer::Int8(bytes.iter().map(|&b| b as i8).collect())
        }
        PixelId::UInt16 | PixelId::VectorUInt16 => unpack!(u16, UInt16),
        PixelId::Int16 | PixelId::VectorInt16 => unpack!(i16, Int16),
        PixelId::UInt32 | PixelId::VectorUInt32 => unpack!(u32, UInt32),
        PixelId::Int32 | PixelId::VectorInt32 => unpack!(i32, Int32),
        PixelId::UInt64 | PixelId::VectorUInt64 => unpack!(u64, UInt64),
        PixelId::Int64 | PixelId::VectorInt64 => unpack!(i64, Int64),
        PixelId::Float32 | PixelId::ComplexFloat32 | PixelId::VectorFloat32 => {
            unpack!(f32, Float32)
        }
        PixelId::Float64 | PixelId::ComplexFloat64 | PixelId::VectorFloat64 => {
            unpack!(f64, Float64)
        }
    })
}

fn fmt_vec_f64(v: &[f64]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build the text header. `element_data_file` is `"LOCAL"` for `.mha` or the raw
/// filename for `.mhd`.
///
/// `compressed_size` is `None` for an uncompressed write and `Some(n)` for a
/// compressed one, where `n` is the *deflated* byte count. `MetaObject`'s
/// `M_SetupWriteFields` emits `CompressedData = True` followed by
/// `CompressedDataSize = n`, and skips the size line when `n` is zero
/// (metaObject.cxx:1421-1439) — a case this port cannot reach, since deflating
/// even an empty buffer yields a non-empty zlib stream.
///
/// `ElementNumberOfChannels` is [`Image::buffer_stride`], not
/// [`Image::number_of_components_per_pixel`] — see the module docs for why
/// those two disagree for a complex image.
///
/// The image's meta-data dictionary is **not** written.
/// `MetaImageIO::WriteImageInformation` emits every dictionary entry as a
/// header field (itkMetaImageIO.cxx:416-470); this port's writer is pinned to
/// its current bytes, so a read/write round trip drops the dictionary rather
/// than growing `ITK_InputFilterName` and `Modality` lines it never had.
fn build_header(img: &Image, element_data_file: &str, compressed_size: Option<u64>) -> String {
    let dim = img.dimension();
    let dim_size = img
        .size()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    // `MET_ULONG_LONG` is printed with `operator<<` on the field's `double`
    // store, so the size is a plain decimal integer (metaUtils.cxx:1502-1512).
    let compression_lines = match compressed_size {
        Some(n) if n > 0 => format!("CompressedData = True\nCompressedDataSize = {n}\n"),
        Some(_) => "CompressedData = True\n".to_string(),
        None => "CompressedData = False\n".to_string(),
    };
    format!(
        "ObjectType = Image\n\
         NDims = {dim}\n\
         BinaryData = True\n\
         BinaryDataByteOrderMSB = False\n\
         {compression_lines}\
         TransformMatrix = {matrix}\n\
         Offset = {offset}\n\
         ElementSpacing = {spacing}\n\
         DimSize = {dim_size}\n\
         ElementNumberOfChannels = {channels}\n\
         ElementType = {etype}\n\
         ElementDataFile = {element_data_file}\n",
        matrix = fmt_vec_f64(img.direction()),
        offset = fmt_vec_f64(img.origin()),
        spacing = fmt_vec_f64(img.spacing()),
        channels = img.buffer_stride(),
        etype = element_type(img.pixel_id()),
    )
}

/// Write an image as MetaImage. `.mha` embeds the data (`ElementDataFile =
/// LOCAL`); `.mhd` writes a sibling file, named `.raw` uncompressed and
/// `.zraw` compressed (`MET_SetFileSuffix`, metaImage.cxx:1586-1593).
///
/// With [`WriteOptions::use_compression`] set, `MetaImage::WriteStream` deflates
/// the whole element buffer *before* the header is laid out
/// (metaImage.cxx:1668-1690), which is why `CompressedDataSize` can appear in a
/// header written in one pass. The deflate is `MET_PerformCompression`'s
/// `deflateInit(&z, level)` — a zlib stream, not gzip — at
/// `MetaImageIO`'s level of 2 unless the caller names another
/// (itkMetaImageIO.cxx:62-64, 717-718).
///
/// Every pixel category is written: scalar, vector, and complex alike, since
/// `buffer_to_le_bytes` already serialises the image's full interleaved
/// buffer (`number_of_pixels * buffer_stride` components) regardless of
/// category, and `build_header` now writes that same `buffer_stride` as
/// `ElementNumberOfChannels`. See the module docs for the read-side
/// consequence: MetaIO cannot tell a complex image from a same-width vector
/// one, so a complex image does not read back as complex.
pub fn write(img: &Image, path: &Path, options: &WriteOptions) -> Result<()> {
    let data = buffer_to_le_bytes(img.buffer());
    let data = if options.use_compression {
        zlib_compress(&data, options.resolved_level(ITK_DEFAULT_COMPRESSION_LEVEL))
    } else {
        data
    };
    let compressed_size = options.use_compression.then_some(data.len() as u64);

    let is_mhd = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("mhd"))
        .unwrap_or(false);

    if is_mhd {
        let suffix = if options.use_compression {
            ".zraw"
        } else {
            ".raw"
        };
        let raw_name = path
            .file_stem()
            .map(|s| {
                let mut n = s.to_os_string();
                n.push(suffix);
                n
            })
            .ok_or_else(|| IoError::InvalidPath(path.to_path_buf()))?;
        let header = build_header(img, &raw_name.to_string_lossy(), compressed_size);
        std::fs::write(path, header)?;
        let raw_path = path.with_file_name(raw_name);
        std::fs::write(raw_path, data)?;
    } else {
        let header = build_header(img, "LOCAL", compressed_size);
        let mut bytes = header.into_bytes();
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

/// The resolved header of a MetaImage file.
struct Header {
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
    element_type: PixelId,
    channels: usize,
    big_endian: bool,
    compressed: bool,
    /// `CompressedDataSize`, `None` when the field is absent. `MET_ULONG_LONG`
    /// is parsed into a `double` upstream (metaUtils.cxx:1231-1240), so a size
    /// above 2^53 loses precision there; this port keeps the exact integer
    /// (§1.56).
    compressed_data_size: Option<u64>,
    element_data_file: String,
    /// Byte offset in the original buffer where the `ElementDataFile` line
    /// ends: pixel data for `LOCAL`, slice filenames for `LIST`.
    data_offset: usize,
    /// The ITK meta-data dictionary this header produces.
    metadata: BTreeMap<String, String>,
}

fn parse_f64_list(s: &str) -> Result<Vec<f64>> {
    s.split_whitespace()
        .map(|t| t.parse::<f64>().map_err(|_| IoError::MalformedHeader))
        .collect()
}

/// Split the header text into `(key, value)` pairs, stopping after
/// `ElementDataFile` — the field MetaIO marks `terminateRead`.
///
/// Returns the pairs in file order plus the byte offset just past the
/// `ElementDataFile` line.
fn scan_fields(bytes: &[u8]) -> Result<(Vec<(String, String)>, usize)> {
    let mut fields = Vec::new();
    let mut pos = 0usize;
    loop {
        let nl = bytes[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or(IoError::MalformedHeader)?;
        let line_end = pos + nl;
        let line = std::str::from_utf8(&bytes[pos..line_end])
            .map_err(|_| IoError::MalformedHeader)?
            .trim_end_matches('\r');
        let next = line_end + 1;

        let (key, value) = line.split_once('=').ok_or(IoError::MalformedHeader)?;
        let key = key.trim();
        let value = value.trim();
        fields.push((key.to_string(), value.to_string()));

        if key == "ElementDataFile" {
            return Ok((fields, next));
        }
        pos = next;
    }
}

/// The last occurrence of `name`, since `MET_Read` overwrites the one field
/// record each time the key reappears.
fn last<'a>(fields: &'a [(String, String)], name: &str) -> Option<&'a str> {
    fields
        .iter()
        .rev()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Build the ITK meta-data dictionary for a parsed header, in
/// `MetaImageIO::ReadImageInformation`'s insertion order
/// (itkMetaImageIO.cxx:270-304): class name, modality, then the additional
/// (unrecognized) fields — which therefore *overwrite* the first two if a file
/// carries a literal `ITK_InputFilterName` tag — then the two optional keys.
fn build_metadata(fields: &[(String, String)]) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    metadata.insert("ITK_InputFilterName".to_string(), "MetaImageIO".to_string());

    // `MET_StringToImageModality` falls back to MET_MOD_UNKNOWN on no match,
    // and MetaImage::Clear() initialises m_Modality to it (metaImageUtils.cxx:
    // 28-44, metaImage.cxx:438).
    let modality = last(fields, "Modality")
        .and_then(|v| MODALITY_NAMES.iter().find(|n| **n == v))
        .copied()
        .unwrap_or("MET_MOD_UNKNOWN");
    metadata.insert("Modality".to_string(), modality.to_string());

    for (key, value) in fields {
        if !RECOGNIZED_FIELDS.contains(&key.as_str()) {
            metadata.insert(key.clone(), value.clone());
        }
    }

    // ITK_VoxelUnits only when DistanceUnits parsed to something other than
    // the unknown "?" (itkMetaImageIO.cxx:294-298).
    let units = last(fields, "DistanceUnits")
        .and_then(|units| DISTANCE_UNIT_NAMES.iter().find(|n| **n == units));
    if let Some(name) = units {
        metadata.insert("ITK_VoxelUnits".to_string(), (*name).to_string());
    }
    if let Some(date) = last(fields, "AcquisitionDate").filter(|d| !d.is_empty()) {
        metadata.insert("ITK_ExperimentDate".to_string(), date.to_string());
    }
    metadata
}

fn parse_header(bytes: &[u8]) -> Result<Header> {
    let (fields, data_offset) = scan_fields(bytes)?;

    let dims: usize = last(&fields, "NDims")
        .ok_or(IoError::MalformedHeader)?
        .parse()
        .map_err(|_| IoError::MalformedHeader)?;
    let size: Vec<usize> = last(&fields, "DimSize")
        .ok_or(IoError::MalformedHeader)?
        .split_whitespace()
        .map(|t| t.parse::<usize>().map_err(|_| IoError::MalformedHeader))
        .collect::<Result<Vec<_>>>()?;
    let element_type =
        parse_element_type(last(&fields, "ElementType").ok_or(IoError::MalformedHeader)?)?;
    if size.len() != dims {
        return Err(IoError::MalformedHeader);
    }

    let spacing = match last(&fields, "ElementSpacing") {
        Some(v) => parse_f64_list(v)?,
        None => vec![1.0; dims],
    };

    // Alias precedence is MetaIO's fixed apply order, not the file's line
    // order: Offset, then Position, then Origin (metaObject.cxx:1653-1675).
    // Position/Origin/Orientation/Rotation only apply at FileFormatVersion 0.
    let legacy = last(&fields, "FileFormatVersion")
        .map(|v| v.trim() == "0")
        .unwrap_or(true);
    let mut origin = vec![0.0; dims];
    let mut direction = identity(dims);
    for key in ["Offset", "Position", "Origin"] {
        if (legacy || key == "Offset")
            && let Some(v) = last(&fields, key)
        {
            origin = parse_f64_list(v)?;
        }
    }
    for key in ["Orientation", "Rotation", "TransformMatrix"] {
        if (legacy || key == "TransformMatrix")
            && let Some(v) = last(&fields, key)
        {
            direction = parse_f64_list(v)?;
        }
    }
    if spacing.len() != dims || origin.len() != dims || direction.len() != dims * dims {
        return Err(IoError::MalformedHeader);
    }

    let channels = match last(&fields, "ElementNumberOfChannels") {
        Some(v) => v.parse::<usize>().map_err(|_| IoError::MalformedHeader)?,
        None => 1,
    };

    // ElementByteOrderMSB first, BinaryDataByteOrderMSB second, so the latter
    // wins when both are present (metaObject.cxx:1618-1642).
    let mut big_endian = false;
    for key in ["ElementByteOrderMSB", "BinaryDataByteOrderMSB"] {
        if let Some(v) = last(&fields, key) {
            big_endian = met_bool(v);
        }
    }
    let compressed = last(&fields, "CompressedData")
        .map(met_bool)
        .unwrap_or(false);
    // `MET_ULONG_LONG` reads with `fp >> value[0]` into a `double`, so a
    // malformed number leaves the field undefined and `m_CompressedDataSize`
    // keeps its `0` (metaObject.cxx:1599-1603) — here, `None`.
    let compressed_data_size =
        last(&fields, "CompressedDataSize").and_then(|v| v.trim().parse().ok());

    let element_data_file = last(&fields, "ElementDataFile")
        .ok_or(IoError::MalformedHeader)?
        .to_string();

    Ok(Header {
        size,
        spacing,
        origin,
        direction,
        element_type,
        channels,
        big_endian,
        compressed,
        compressed_data_size,
        element_data_file,
        data_offset,
        metadata: build_metadata(&fields),
    })
}

/// `MET_GetFilePath` + `FileIsFullPath`: resolve `name` against the header's
/// own directory unless it is already absolute (metaImage.cxx:1355-1363).
fn resolve_sibling(header_path: &Path, name: &str) -> PathBuf {
    let name = Path::new(name);
    if name.is_absolute() {
        name.to_path_buf()
    } else {
        header_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(name)
    }
}

/// Strip the trailing whitespace and non-printable characters MetaIO strips
/// from a `LIST` slice name (metaImage.cxx:1352-1356). The loop guard is
/// `j > 0`, so the first character is never stripped.
fn trim_slice_name(line: &str) -> &str {
    let mut end = line.len();
    while end > 1 {
        let b = line.as_bytes()[end - 1];
        if b.is_ascii_whitespace() || !(0x20..=0x7e).contains(&b) {
            end -= 1;
        } else {
            break;
        }
    }
    &line[..end]
}

/// `ElementDataFile = LIST [fileImageDim]`: how many axes live *inside* each
/// slice file, and therefore how many files there are.
///
/// `MetaImage::Read` reads the optional second word with `atof` and falls back
/// to `NDims - 1` when it is `0` or greater than `NDims`
/// (metaImage.cxx:1318-1333).
fn list_file_image_dim(element_data_file: &str, dims: usize) -> usize {
    let requested = element_data_file
        .split_whitespace()
        .nth(1)
        .and_then(|w| w.parse::<f64>().ok())
        .map(|f| f as i64);
    match requested {
        // Upstream's guard is `(fileImageDim == 0) || (fileImageDim > m_NDims)`.
        // A negative value slips through it and indexes `m_DimSize[-1]`; this
        // port folds negatives into the same fallback. `fileImageDim == NDims`
        // slips through too and reads the uninitialised `m_SubQuantity[NDims]`;
        // here it means "one file holding the whole image", which is what the
        // surrounding arithmetic implies.
        Some(d) if d > 0 && (d as usize) <= dims => d as usize,
        _ => dims.saturating_sub(1),
    }
}

/// Gather the pixel bytes named by an `ElementDataFile = LIST` header.
///
/// Each of the `totalFiles = prod(DimSize[fileImageDim..NDims])` lines after
/// the `ElementDataFile` line names one file holding
/// `prod(DimSize[..fileImageDim])` pixels, concatenated in line order
/// (metaImage.cxx:1318-1387).
///
/// Upstream's read loop is `for (i = 0; i < totalFiles && !_stream->eof(); ++i)`
/// and returns success even when the list ran out early, leaving the tail of
/// the freshly `new`-ed buffer uninitialised. That is unreproducible in safe
/// Rust and indistinguishable from a corrupt file, so a short list is
/// [`IoError::TruncatedData`] here.
/// Each slice goes through `M_ReadElements` in its own right
/// (metaImage.cxx:1375-1382), so a compressed `LIST` inflates each named file
/// separately. `compressedDataDeterminedFromFile` resets `m_CompressedDataSize`
/// to `0` after every slice (metaImage.cxx:2634-2637), so an absent
/// `CompressedDataSize` lets each slice use its own file length, while a
/// present one is applied to *every* slice alike.
fn read_list_data(path: &Path, header: &Header, tail: &[u8]) -> Result<Vec<u8>> {
    let dims = header.size.len();
    let file_image_dim = list_file_image_dim(&header.element_data_file, dims);
    let per_file_pixels: usize = header.size[..file_image_dim].iter().product();
    let per_file_bytes = per_file_pixels * header.channels * header.element_type.size_in_bytes();
    let total_files: usize = header.size[file_image_dim..].iter().product();

    let text = std::str::from_utf8(tail).map_err(|_| IoError::MalformedHeader)?;
    let mut lines = text.lines();
    let mut data = Vec::with_capacity(per_file_bytes * total_files);
    for _ in 0..total_files {
        let name = lines.next().ok_or(IoError::TruncatedData)?;
        let slice = std::fs::read(resolve_sibling(path, trim_slice_name(name)))?;
        if header.compressed {
            data.extend_from_slice(&uncompress_elements(
                &slice,
                header.compressed_data_size,
                per_file_bytes,
            )?);
            continue;
        }
        if slice.len() < per_file_bytes {
            return Err(IoError::TruncatedData);
        }
        data.extend_from_slice(&slice[..per_file_bytes]);
    }
    Ok(data)
}

/// `M_ReadElements`' compressed arm (metaImage.cxx:2610-2640): read
/// `CompressedDataSize` bytes — or, when the header did not say, the whole of
/// `source` — and inflate them into exactly `want` bytes.
///
/// `declared` larger than `source` is [`IoError::TruncatedData`], which is
/// where `M_ReadElementData`'s `gc != _dataQuantity` check lands upstream
/// (metaImage.cxx:3583-3588).
fn uncompress_elements(source: &[u8], declared: Option<u64>, want: usize) -> Result<Vec<u8>> {
    let size = match declared {
        Some(n) => usize::try_from(n).map_err(|_| IoError::TruncatedData)?,
        None => source.len(),
    };
    let compressed = source.get(..size).ok_or(IoError::TruncatedData)?;
    inflate_auto(compressed, want)
}

/// Read a MetaImage from `.mha` or `.mhd`.
///
/// `header.channels` (`ElementNumberOfChannels`, `1` when the key is absent)
/// maps back to a [`PixelId`] the same way `MetaImageIO::ReadImageInformation`
/// does: one channel stays the plain scalar `ElementType`; more than one
/// channel always becomes that type's **vector** variant, never complex
/// (itkMetaImageIO.cxx:241-244) — see the module docs for the round-trip
/// consequence this has for a complex image.
///
/// `0` channels is rejected: [`Image::from_parts_vector`] refuses zero
/// components per pixel ([`crate::core::Error::InvalidComponentCount`]), and a
/// channel count the file's actual data is too short for is rejected as
/// [`IoError::TruncatedData`].
///
/// The returned image carries the header's meta-data dictionary, as
/// `itk::ImageFileReader` copies the `ImageIO`'s onto its output
/// (itkImageFileReader.hxx:242).
pub fn read(path: &Path) -> Result<Image> {
    let bytes = std::fs::read(path)?;
    let header = parse_header(&bytes)?;

    let n: usize = header.size.iter().product();
    let byte_len = n * header.channels * header.element_type.size_in_bytes();

    let edf = &header.element_data_file;
    let data: Vec<u8> = if edf.eq_ignore_ascii_case("local") {
        let raw = &bytes[header.data_offset..];
        if header.compressed {
            // Upstream's "guess the compressed size from the file size" arm
            // seeks back to offset 0 and inflates the header text (§1.56); a
            // `LOCAL` header without `CompressedDataSize` is refused here.
            let size = header.compressed_data_size.ok_or_else(|| {
                IoError::Unsupported(
                    "CompressedData = True with no CompressedDataSize in a LOCAL header: \
                     upstream reads uninitialised memory here (ledger §1.56)"
                        .into(),
                )
            })?;
            uncompress_elements(raw, Some(size), byte_len)?
        } else {
            raw.to_vec()
        }
    } else if edf.len() >= 4 && &edf[..4] == "LIST" {
        read_list_data(path, &header, &bytes[header.data_offset..])?
    } else {
        let raw = std::fs::read(resolve_sibling(path, edf))?;
        if header.compressed {
            uncompress_elements(&raw, header.compressed_data_size, byte_len)?
        } else {
            raw
        }
    };

    if data.len() < byte_len {
        return Err(IoError::TruncatedData);
    }
    let buffer = buffer_from_bytes(header.element_type, &data[..byte_len], header.big_endian)?;

    let mut image = if header.channels == 1 {
        Image::from_parts(
            buffer,
            header.size,
            header.spacing,
            header.origin,
            header.direction,
        )
    } else {
        Image::from_parts_vector(
            buffer,
            header.channels,
            header.size,
            header.spacing,
            header.origin,
            header.direction,
        )
    }
    .map_err(IoError::Core)?;

    for (key, value) in &header.metadata {
        image.set_meta_data(key, value);
    }
    Ok(image)
}

/// Read just the header text: every line up to and including `ElementDataFile`.
///
/// `MetaImage::Read(name, /*_readElements=*/false)` stops at that line too —
/// its field record is `terminateRead` — so an `.mha`'s pixel tail is never
/// touched, however large it is.
fn read_header_bytes(path: &Path) -> Result<Vec<u8>> {
    use std::io::BufRead;

    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut bytes = Vec::new();
    loop {
        let start = bytes.len();
        if reader.read_until(b'\n', &mut bytes)? == 0 {
            return Ok(bytes);
        }
        let line = String::from_utf8_lossy(&bytes[start..]);
        if line.split_once('=').map(|(k, _)| k.trim()) == Some("ElementDataFile") {
            return Ok(bytes);
        }
    }
}

/// Read the header only — geometry, pixel type, and meta-data dictionary.
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    let bytes = read_header_bytes(path)?;
    let header = parse_header(&bytes)?;
    let dimension = header.size.len();
    let pixel_id = if header.channels == 1 {
        header.element_type
    } else {
        header.element_type.vector_id()
    };
    Ok(ImageInformation {
        pixel_id,
        dimension,
        number_of_components: header.channels,
        size: header.size,
        spacing: header.spacing,
        origin: header.origin,
        direction: header.direction,
        metadata: header.metadata,
    })
}

/// `MetaImage::CanRead`'s content probe: the first 8000 bytes, truncated at the
/// first NUL, must contain `NDims` (metaImage.cxx:1201-1228).
///
/// Upstream builds `std::string header(buf)` — which stops at the first NUL —
/// and then `resize`s it back up with NUL padding, so `NDims` occurring after
/// an embedded NUL is invisible to the probe. Reproduced by searching only the
/// prefix before the first NUL.
fn content_looks_like_meta_image(path: &Path) -> bool {
    use std::io::Read;

    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = Vec::new();
    if file.take(8000).read_to_end(&mut head).is_err() {
        return false;
    }
    let head = match head.iter().position(|&b| b == 0) {
        Some(nul) => &head[..nul],
        None => &head[..],
    };
    head.windows(5).any(|w| w == b"NDims")
}

/// `itk::MetaImageIO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetaImageIo;

impl ImageIo for MetaImageIo {
    fn name(&self) -> &'static str {
        "MetaImageIO"
    }

    fn supported_read_extensions(&self) -> &'static [&'static str] {
        &[".mha", ".mhd"]
    }

    fn supported_write_extensions(&self) -> &'static [&'static str] {
        &[".mha", ".mhd"]
    }

    /// `MetaImageIO::CanReadFile` delegates straight to `MetaImage::CanRead`
    /// (itkMetaImageIO.cxx:87-99), which re-checks the extension itself — with
    /// a **case-sensitive** `rfind(".mhd")` / `rfind(".mha")`
    /// (metaImage.cxx:1184-1199) — before looking at the content.
    ///
    /// Two consequences, both upstream's and both reproduced here. A file whose
    /// content is a MetaImage header but whose name is `data.foo` is *not*
    /// readable, even though the registry's phase 2 offers it a second chance:
    /// content never overrides a missing extension for this IO. And `IMG.MHA`
    /// is writable (`CanWriteFile` is case-insensitive) yet not readable,
    /// because `ImageIOFactory` matches the extension case-insensitively and
    /// then `MetaImage::CanRead` rejects the uppercase spelling.
    fn can_read_file(&self, path: &Path) -> bool {
        let name = path.as_os_str().to_string_lossy();
        if !(name.ends_with(".mha") || name.ends_with(".mhd")) {
            return false;
        }
        content_looks_like_meta_image(path)
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
