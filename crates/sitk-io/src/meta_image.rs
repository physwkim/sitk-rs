//! MetaImage (`.mha` / `.mhd` + `.raw`) reader and writer.
//!
//! MetaImage is ITK's native uncompressed format: a plain-text `Key = Value`
//! header followed by (or referencing) a raw binary pixel dump. It round-trips
//! every scalar pixel type, arbitrary dimension, and the full spacing / origin /
//! direction geometry, which makes it the right Phase-0 format for exercising the
//! whole core model without pulling in an external image crate.
//!
//! Not yet supported: compressed data, multi-channel (vector) pixels.

use std::path::{Path, PathBuf};

use sitk_core::{Image, PixelBuffer, PixelId};

use crate::error::{IoError, Result};

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
    if bytes.len() % expected != 0 {
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
fn build_header(img: &Image, element_data_file: &str) -> String {
    let dim = img.dimension();
    let dim_size = img
        .size()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "ObjectType = Image\n\
         NDims = {dim}\n\
         BinaryData = True\n\
         BinaryDataByteOrderMSB = False\n\
         CompressedData = False\n\
         TransformMatrix = {matrix}\n\
         Offset = {offset}\n\
         ElementSpacing = {spacing}\n\
         DimSize = {dim_size}\n\
         ElementNumberOfChannels = 1\n\
         ElementType = {etype}\n\
         ElementDataFile = {element_data_file}\n",
        matrix = fmt_vec_f64(img.direction()),
        offset = fmt_vec_f64(img.origin()),
        spacing = fmt_vec_f64(img.spacing()),
        etype = element_type(img.pixel_id()),
    )
}

/// Write an image as MetaImage. `.mha` embeds the data (`ElementDataFile =
/// LOCAL`); `.mhd` writes a sibling `.raw` file.
///
/// Only scalar images are written: [`build_header`] hard-codes
/// `ElementNumberOfChannels = 1`, so the file would claim one component per
/// pixel while carrying `buffer_stride()` of them — `n` for a vector image, `2`
/// for a complex one (MetaIO has no complex element type at all). This is the
/// write-side counterpart of [`read`]'s `header.channels != 1` check, and the
/// test is a whitelist on `PixelId::is_scalar` for the same reason
/// `Image::require_scalar` is.
pub fn write(img: &Image, path: &Path) -> Result<()> {
    if !img.pixel_id().is_scalar() {
        return Err(IoError::Unsupported(format!(
            "{:?}: MetaImage writes one component per pixel",
            img.pixel_id()
        )));
    }
    let data = buffer_to_le_bytes(img.buffer());
    let is_mhd = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("mhd"))
        .unwrap_or(false);

    if is_mhd {
        let raw_name = path
            .file_stem()
            .map(|s| {
                let mut n = s.to_os_string();
                n.push(".raw");
                n
            })
            .ok_or_else(|| IoError::InvalidPath(path.to_path_buf()))?;
        let header = build_header(img, &raw_name.to_string_lossy());
        std::fs::write(path, header)?;
        let raw_path = path.with_file_name(raw_name);
        std::fs::write(raw_path, data)?;
    } else {
        let header = build_header(img, "LOCAL");
        let mut bytes = header.into_bytes();
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

struct Header {
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
    element_type: PixelId,
    channels: usize,
    big_endian: bool,
    compressed: bool,
    element_data_file: String,
    /// Byte offset in the original buffer where pixel data begins (for LOCAL).
    data_offset: usize,
}

fn parse_f64_list(s: &str) -> Result<Vec<f64>> {
    s.split_whitespace()
        .map(|t| t.parse::<f64>().map_err(|_| IoError::MalformedHeader))
        .collect()
}

fn parse_header(bytes: &[u8]) -> Result<Header> {
    let mut dims = None;
    let mut size = None;
    let mut spacing = None;
    let mut origin = None;
    let mut direction = None;
    let mut element_type = None;
    let mut channels = 1usize;
    let mut big_endian = false;
    let mut compressed = false;

    // Scan line by line over the header text without decoding the binary tail.
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

        match key.to_ascii_lowercase().as_str() {
            "ndims" => {
                dims = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| IoError::MalformedHeader)?,
                )
            }
            "dimsize" => {
                size = Some(
                    value
                        .split_whitespace()
                        .map(|t| t.parse::<usize>().map_err(|_| IoError::MalformedHeader))
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            "elementspacing" => spacing = Some(parse_f64_list(value)?),
            "offset" | "position" | "origin" => origin = Some(parse_f64_list(value)?),
            "transformmatrix" | "orientation" | "rotation" => {
                direction = Some(parse_f64_list(value)?)
            }
            "elementtype" => element_type = Some(parse_element_type(value)?),
            "elementnumberofchannels" => {
                channels = value
                    .parse::<usize>()
                    .map_err(|_| IoError::MalformedHeader)?
            }
            "binarydatabyteordermsb" | "elementbyteordermsb" => {
                big_endian = value.eq_ignore_ascii_case("true")
            }
            "compresseddata" => compressed = value.eq_ignore_ascii_case("true"),
            "elementdatafile" => {
                let dims = dims.ok_or(IoError::MalformedHeader)?;
                let size = size.ok_or(IoError::MalformedHeader)?;
                let element_type = element_type.ok_or(IoError::MalformedHeader)?;
                if size.len() != dims {
                    return Err(IoError::MalformedHeader);
                }
                let spacing = spacing.unwrap_or_else(|| vec![1.0; dims]);
                let origin = origin.unwrap_or_else(|| vec![0.0; dims]);
                let direction = direction.unwrap_or_else(|| identity(dims));
                if spacing.len() != dims || origin.len() != dims || direction.len() != dims * dims {
                    return Err(IoError::MalformedHeader);
                }
                return Ok(Header {
                    size,
                    spacing,
                    origin,
                    direction,
                    element_type,
                    channels,
                    big_endian,
                    compressed,
                    element_data_file: value.to_string(),
                    data_offset: next,
                });
            }
            _ => {} // ignore unrecognised MetaIO tags
        }
        pos = next;
    }
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Read a MetaImage from `.mha` or `.mhd`.
pub fn read(path: &Path) -> Result<Image> {
    let bytes = std::fs::read(path)?;
    let header = parse_header(&bytes)?;

    if header.compressed {
        return Err(IoError::Unsupported("compressed MetaImage data".into()));
    }
    if header.channels != 1 {
        return Err(IoError::Unsupported("multi-channel (vector) pixels".into()));
    }

    let n: usize = header.size.iter().product();
    let byte_len = n * header.element_type.size_in_bytes();

    let data: Vec<u8> = if header.element_data_file.eq_ignore_ascii_case("local") {
        bytes[header.data_offset..].to_vec()
    } else {
        let raw_path: PathBuf = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&header.element_data_file);
        std::fs::read(raw_path)?
    };

    if data.len() < byte_len {
        return Err(IoError::TruncatedData);
    }
    let buffer = buffer_from_bytes(header.element_type, &data[..byte_len], header.big_endian)?;

    Image::from_parts(
        buffer,
        header.size,
        header.spacing,
        header.origin,
        header.direction,
    )
    .map_err(IoError::Core)
}
