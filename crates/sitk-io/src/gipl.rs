//! GIPL (`.gipl`) reader and writer — `itk::GiplImageIO`.
//!
//! GIPL is the Guy's Image Processing Lab volume format: a fixed **256-byte
//! big-endian header** followed by a raw, big-endian, scalar pixel dump. There
//! is no direction matrix, no component count, and no text: everything the
//! reader keeps lives in three of the header's fifteen fields.
//!
//! The upstream class is small enough to describe field by field, and almost
//! every field it reads it then throws away. What follows is what
//! `itkGiplImageIO.cxx` *does*, not what the format specifies.
//!
//! # The header
//!
//! | offset | size | field | what `ReadImageInformation` does with it |
//! |---|---|---|---|
//! | 0 | 4×`u16` | `dims` | axis lengths; drives the dimension count |
//! | 8 | `u16` | `image_type` | the pixel-type table below |
//! | 10 | 4×`f32` | `pixdim` | spacing, first `NDims` entries |
//! | 26 | 80×`char` | `line1` | read one byte at a time, **discarded** |
//! | 106 | 20×`f32` | `matrix` | byte-swapped, **discarded** |
//! | 186 | `char` | `flag1` | orientation flag, **discarded** |
//! | 187 | `char` | `flag2` | **discarded** |
//! | 188 | `f64` | `min` | **discarded**, and read *without* a byte swap |
//! | 196 | `f64` | `max` | **discarded**, and read *without* a byte swap |
//! | 204 | 4×`f64` | `origin` | origin, first `NDims` entries |
//! | 236 | `f32` | `pixval_offset` | byte-swapped, **discarded** |
//! | 240 | `f32` | `pixval_cal` | byte-swapped, **discarded** |
//! | 244 | `f32` | `user_def1` | inter-slice gap, byte-swapped, **discarded** |
//! | 248 | `f32` | `user_def2` | byte-swapped, **discarded** |
//! | 252 | `u32` | `magic_number` | byte-swapped and then **not compared** |
//!
//! The magic number is validated only by `CanReadFile`, which seeks straight to
//! offset 252 (itkGiplImageIO.cxx:122-139) and accepts either
//! [`GIPL_MAGIC_NUMBER`] (`0xefffe9b0`, 4026526128) or [`GIPL_MAGIC_NUMBER2`]
//! (`0x2ae389b8`, 719555000). `ReadImageInformation` re-reads the same four
//! bytes at the end of the header purely to leave the stream positioned at byte
//! 256, and discards them (`:566-584`). Ledger §2.93.
//!
//! `m_ByteOrder` is set to `BigEndian` in the constructor (`:78`) and nothing in
//! ITK or SimpleITK ever changes it, so every multi-byte field — header and
//! pixel alike — is big-endian. (`image_type` is byte-swapped only in the
//! `BigEndian` arm; the `LittleEndian` arm every other field carries is missing
//! at `:325-328`. Unobservable for that reason. Ledger §2.95.)
//!
//! # Pixel types
//!
//! `image_type` maps to a component type (`:331-360`), and `m_PixelType` is
//! hard-wired to `SCALAR`:
//!
//! | code | name | component |
//! |---|---|---|
//! | 1 | `GIPL_BINARY` | `UInt8` |
//! | 7 | `GIPL_CHAR` | `Int8` |
//! | 8 | `GIPL_U_CHAR` | `UInt8` |
//! | 15 | `GIPL_SHORT` | `Int16` |
//! | 16 | `GIPL_U_SHORT` | `UInt16` |
//! | 31 | `GIPL_U_INT` | `UInt32` |
//! | 32 | `GIPL_INT` | `Int32` |
//! | 64 | `GIPL_FLOAT` | `Float32` |
//! | 65 | `GIPL_DOUBLE` | `Float64` |
//!
//! **But `UInt32` and `Int32` cannot actually be read or written.**
//! `SwapBytesIfNecessary` (`:588-653`) has arms for `SCHAR`, `UCHAR`, `SHORT`,
//! `USHORT`, `FLOAT` and `DOUBLE`, and a `default:` that throws `"Pixel Type
//! Unknown"` — `INT` and `UINT` fall into it. `Read` calls it after the pixel
//! data has been read (`:243`) and `Write` calls it before the pixel data is
//! written (`:1010`/`:1024`), so a 32-bit-integer GIPL file throws on read and,
//! on write, throws *after* `Write` has already truncated the file and emitted
//! the full 256-byte header. Ledger §1.52; [`write`] reproduces the truncated
//! file exactly.
//!
//! 64-bit integers never reach that point: `Write`'s own `image_type` switch has
//! no `LONG`/`LONGLONG` arm and throws `"Invalid type"` (`:759-761`) — after the
//! four `dims` values are on disk, leaving an **8-byte file**. Reproduced.
//!
//! So the pixel types a SimpleITK caller can actually round-trip through GIPL
//! are `UInt8`, `Int8`, `UInt16`, `Int16`, `Float32` and `Float64`.
//!
//! # Dimensions
//!
//! `Write` pads the four `dims` slots with `1` above the image's own dimension
//! (`:709-729`), and `ReadImageInformation` counts a dimension for every
//! `dims[i] > 0` at `i < 3`, plus `dims[3] > 1` (`:294-304`). The two do not
//! compose: **a 2-D image written by ITK reads back as 3-D** with a unit third
//! axis (spacing `1`, origin `0`). Pixel values are unaffected. Ledger §2.94.
//!
//! The count is also not the length of the leading non-zero run — it is a
//! population count over the first three slots. A hand-written header with
//! `dims = [4, 0, 5, 1]` yields `NDims = 2` and `m_Dimensions = [4, 0]`, because
//! `m_Dimensions[i] = dims[i]` copies the *first* `NDims` slots. Reproduced.
//!
//! # Vector and complex images
//!
//! `Write` never consults `m_NumberOfComponents` when it fills the header, but
//! `GetImageSizeInBytes()` does — so a 3-component vector image writes
//! `image_type = GIPL_U_CHAR` and three times as many bytes as the header
//! describes. Reading it back yields a *scalar* image holding the first
//! `numPixels` components. Reproduced, pinned, ledger §2.96.
//!
//! # Extensions and compression
//!
//! `GiplImageIO` calls neither `AddSupportedReadExtension` nor
//! `AddSupportedWriteExtension`, so it advertises **no** extensions and
//! `ImageIOFactory` only ever finds it in its second, extension-blind probe
//! phase. [`GiplImageIo::supported_read_extensions`] is empty for that reason.
//! The real gate is `CheckExtension` (`:1065-1093`), a case-**sensitive**
//! suffix test for `.gipl` and `.gipl.gz` that also sets `m_IsCompressed`.
//! Ledger §2.97.
//!
//! `.gipl.gz` is handled upstream by a `gzFile`. This workspace has no zlib
//! (ledger §5.8), so [`read`], [`read_information`] and [`write`] refuse it with
//! [`IoError::UnsupportedGiplFeature`] at exactly the point upstream would have
//! called `gzopen`. [`GiplImageIo::can_read_file`] claims a `.gipl.gz` on its
//! name alone — upstream verifies the magic number *through* the gzip stream,
//! which is precisely what is unavailable — so that the failure is reported as
//! the missing zlib rather than as "no reader found". Ledger §4.68.
//!
//! # Truncated data
//!
//! `Read` tests `!m_Ifstream.bad()` for success (`:226`), but a short `read`
//! sets `failbit`/`eofbit`, never `badbit` — so upstream returns success with
//! the tail of ITK's freshly-allocated buffer left uninitialised. That is C++ UB
//! and unreachable in safe Rust: [`read`] returns [`IoError::TruncatedData`].
//! Ledger §1.53 / §4.69.

use std::collections::BTreeMap;
use std::path::Path;

use sitk_core::{Image, PixelBuffer, PixelId};

use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, ImageIo};

/// `GIPL_MAGIC_NUMBER` (itkGiplImageIO.cxx:72) — the value [`write`] emits.
pub const GIPL_MAGIC_NUMBER: u32 = 0xefff_e9b0;
/// `GIPL_MAGIC_NUMBER2` (itkGiplImageIO.cxx:73), accepted on read.
pub const GIPL_MAGIC_NUMBER2: u32 = 0x2ae3_89b8;

/// The fixed header length: `4·2 + 2 + 4·4 + 80 + 20·4 + 1 + 1 + 8 + 8 + 4·8 +
/// 4·4 + 4`.
pub const HEADER_SIZE: usize = 256;

const GIPL_BINARY: u16 = 1;
const GIPL_CHAR: u16 = 7;
const GIPL_U_CHAR: u16 = 8;
const GIPL_SHORT: u16 = 15;
const GIPL_U_SHORT: u16 = 16;
const GIPL_U_INT: u16 = 31;
const GIPL_INT: u16 = 32;
const GIPL_FLOAT: u16 = 64;
const GIPL_DOUBLE: u16 = 65;

/// The 80-byte `line1` field `Write` emits: `snprintf(line1, 80, "No Patient
/// Information")` over a zeroed buffer (itkGiplImageIO.cxx:827-845).
const PATIENT_TEXT: &[u8] = b"No Patient Information";

/// `CheckExtension` (itkGiplImageIO.cxx:1065-1093): a case-sensitive suffix
/// test. Returns `None` when neither suffix matches, `Some(is_compressed)`
/// otherwise.
///
/// `.gipl.gz` is tested after `.gipl` and overrides it, which is why a name can
/// never be both.
fn check_extension(path: &Path) -> Option<bool> {
    let name = path.as_os_str().to_string_lossy().into_owned();
    if name.ends_with(".gipl.gz") {
        Some(true)
    } else if name.ends_with(".gipl") {
        Some(false)
    } else {
        None
    }
}

/// `CheckExtension`'s side effect alone. `ReadImageInformation` and `Write` call
/// it for `m_IsCompressed` and ignore the return value (`:249`, `:665`), so a
/// file reached through `SetImageIO` under a foreign name is read as an
/// uncompressed GIPL.
fn is_compressed(path: &Path) -> bool {
    check_extension(path) == Some(true)
}

fn zlib_unavailable(what: &str) -> IoError {
    IoError::UnsupportedGiplFeature(format!(
        "{what}: gzip-compressed GIPL needs zlib, which this workspace does not \
         depend on (upstream calls gzopen here; see doc/upstream-findings.md §5.8)"
    ))
}

/// `image_type` → component type (itkGiplImageIO.cxx:331-360). An unrecognized
/// code leaves `m_ComponentType` at `UNKNOWNCOMPONENTTYPE`, which SimpleITK's
/// `ExecuteInternalReadScalar` turns into `"Logic error!"`
/// (sitkImageReaderBase.cxx:308-311).
fn component_type(image_type: u16) -> Option<PixelId> {
    Some(match image_type {
        GIPL_BINARY | GIPL_U_CHAR => PixelId::UInt8,
        GIPL_CHAR => PixelId::Int8,
        GIPL_SHORT => PixelId::Int16,
        GIPL_U_SHORT => PixelId::UInt16,
        GIPL_U_INT => PixelId::UInt32,
        GIPL_INT => PixelId::Int32,
        GIPL_FLOAT => PixelId::Float32,
        GIPL_DOUBLE => PixelId::Float64,
        _ => return None,
    })
}

/// The `image_type` `Write`'s switch emits (itkGiplImageIO.cxx:733-761). `None`
/// is upstream's `default:` arm, `itkExceptionMacro("Invalid type: ...")` — the
/// 64-bit integers, which have no GIPL code.
fn image_type_code(component: PixelId) -> Option<u16> {
    Some(match component {
        PixelId::Int8 => GIPL_CHAR,
        PixelId::UInt8 => GIPL_U_CHAR,
        PixelId::Int16 => GIPL_SHORT,
        PixelId::UInt16 => GIPL_U_SHORT,
        PixelId::UInt32 => GIPL_U_INT,
        PixelId::Int32 => GIPL_INT,
        PixelId::Float32 => GIPL_FLOAT,
        PixelId::Float64 => GIPL_DOUBLE,
        _ => return None,
    })
}

/// Which component types `SwapBytesIfNecessary` has an arm for
/// (itkGiplImageIO.cxx:590-652). `UInt32`/`Int32` are absent: they hit the
/// `default:` that throws `"Pixel Type Unknown"`, on both the read and the write
/// path. Ledger §1.52.
fn is_swappable(component: PixelId) -> bool {
    matches!(
        component,
        PixelId::Int8
            | PixelId::UInt8
            | PixelId::Int16
            | PixelId::UInt16
            | PixelId::Float32
            | PixelId::Float64
    )
}

fn pixel_type_unknown(component: PixelId) -> IoError {
    IoError::UnsupportedGiplFeature(format!(
        "Pixel Type Unknown: GiplImageIO::SwapBytesIfNecessary has no arm for \
         {} (doc/upstream-findings.md §1.52)",
        component.as_str()
    ))
}

fn be_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([bytes[at], bytes[at + 1]])
}

fn be_f32(bytes: &[u8], at: usize) -> f32 {
    f32::from_be_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
}

fn be_f64(bytes: &[u8], at: usize) -> f64 {
    f64::from_be_bytes(bytes[at..at + 8].try_into().expect("8 bytes"))
}

/// Everything `ReadImageInformation` keeps out of the 256 header bytes.
struct Header {
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    component: PixelId,
}

/// Parse the 256-byte header exactly as `ReadImageInformation` does.
///
/// A header shorter than 256 bytes leaves upstream's locals partly
/// indeterminate (`pixdim`, `origin` and the discarded fields are never
/// zero-initialised); this port refuses it with [`IoError::TruncatedData`].
/// Ledger §4.73.
fn parse_header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < HEADER_SIZE {
        return Err(IoError::TruncatedData);
    }

    // `numberofdimension` counts every non-zero slot below index 3, plus a
    // fourth slot greater than one — a population count, not the length of the
    // leading run (itkGiplImageIO.cxx:294-304).
    let mut dims = [0u16; 4];
    let mut number_of_dimensions = 0usize;
    for (i, dim) in dims.iter_mut().enumerate() {
        *dim = be_u16(bytes, i * 2);
        if *dim > 0 && (i < 3 || *dim > 1) {
            number_of_dimensions += 1;
        }
    }

    let image_type = be_u16(bytes, 8);
    let component = component_type(image_type).ok_or(IoError::UnsupportedGiplFeature(format!(
        "unknown GIPL image_type {image_type}"
    )))?;

    // `SetNumberOfDimensions(0)` leaves SimpleITK with a zero-dimensional image,
    // which `ImageFileReader::Execute` rejects (sitkImageFileReader.cxx:302-307).
    if number_of_dimensions == 0 {
        return Err(IoError::UnsupportedImageDimension(0));
    }

    // `m_Dimensions[i] = dims[i]`, `m_Spacing[i] = pixdim[i]`, `m_Origin[i] =
    // origin[i]` — always the *first* `NDims` slots.
    let size = dims[..number_of_dimensions]
        .iter()
        .map(|&d| d as usize)
        .collect();
    let spacing = (0..number_of_dimensions)
        .map(|i| f64::from(be_f32(bytes, 10 + i * 4)))
        .collect();
    let origin = (0..number_of_dimensions)
        .map(|i| be_f64(bytes, 204 + i * 8))
        .collect();

    Ok(Header {
        size,
        spacing,
        origin,
        component,
    })
}

fn buffer_from_be_bytes(component: PixelId, bytes: &[u8]) -> PixelBuffer {
    macro_rules! unpack {
        ($ty:ty, $variant:ident) => {{
            const S: usize = std::mem::size_of::<$ty>();
            PixelBuffer::$variant(
                bytes
                    .chunks_exact(S)
                    .map(|c| <$ty>::from_be_bytes(c.try_into().expect("chunk size")))
                    .collect(),
            )
        }};
    }
    match component {
        PixelId::UInt8 => PixelBuffer::UInt8(bytes.to_vec()),
        PixelId::Int8 => PixelBuffer::Int8(bytes.iter().map(|&b| b as i8).collect()),
        PixelId::UInt16 => unpack!(u16, UInt16),
        PixelId::Int16 => unpack!(i16, Int16),
        PixelId::Float32 => unpack!(f32, Float32),
        PixelId::Float64 => unpack!(f64, Float64),
        // Unreachable: `is_swappable` gates every caller.
        other => unreachable!("{other:?} is not a GIPL-swappable component"),
    }
}

/// `SwapBytesIfNecessary` on the write path: the whole interleaved buffer,
/// component by component, into big-endian bytes.
fn buffer_to_be_bytes(buffer: &PixelBuffer) -> Vec<u8> {
    macro_rules! pack {
        ($v:expr) => {{
            let mut out = Vec::with_capacity(std::mem::size_of_val(&$v[..]));
            for &x in $v.iter() {
                out.extend_from_slice(&x.to_be_bytes());
            }
            out
        }};
    }
    match buffer {
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

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Read a `.gipl` file.
///
/// `m_PixelType` is always `SCALAR` and `m_NumberOfComponents` always `1`, so
/// the image is always scalar however many components its writer meant to store.
/// The direction matrix is the identity: GIPL has no orientation field the
/// reader keeps (`flag1` is read and discarded).
pub fn read(path: &Path) -> Result<Image> {
    if is_compressed(path) {
        return Err(zlib_unavailable("read"));
    }
    let bytes = std::fs::read(path)?;
    let header = parse_header(&bytes)?;

    // `Read` reads `GetImageSizeInBytes()` bytes and *then* swaps, so the
    // "Pixel Type Unknown" throw happens either way; it is raised here first.
    if !is_swappable(header.component) {
        return Err(pixel_type_unknown(header.component));
    }

    let pixels: usize = header.size.iter().product();
    let byte_len = pixels * header.component.size_in_bytes();
    let data = &bytes[HEADER_SIZE..];
    if data.len() < byte_len {
        return Err(IoError::TruncatedData);
    }

    let buffer = buffer_from_be_bytes(header.component, &data[..byte_len]);
    let dim = header.size.len();
    Image::from_parts(
        buffer,
        header.size,
        header.spacing,
        header.origin,
        identity(dim),
    )
    .map_err(IoError::Core)
}

/// Read the header only. The meta-data dictionary is empty: `GiplImageIO` writes
/// nothing into `m_MetaDataDictionary` — not even the `ITK_InputFilterName`
/// `MetaImageIO` installs.
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    use std::io::Read;

    if is_compressed(path) {
        return Err(zlib_unavailable("read"));
    }
    let mut head = Vec::new();
    std::fs::File::open(path)?
        .take(HEADER_SIZE as u64)
        .read_to_end(&mut head)?;
    let header = parse_header(&head)?;
    let dimension = header.size.len();

    Ok(ImageInformation {
        pixel_id: header.component,
        dimension,
        number_of_components: 1,
        size: header.size,
        spacing: header.spacing,
        origin: header.origin,
        direction: identity(dimension),
        metadata: BTreeMap::new(),
    })
}

/// Write a `.gipl` file.
///
/// Upstream's `Write` truncates the file *before* it can discover that the pixel
/// type is unwritable, so both failure modes leave a partial file behind and
/// this port leaves the same bytes:
///
/// * a 64-bit integer image leaves the **8 bytes** of `dims`, then
///   `"Invalid type"` (itkGiplImageIO.cxx:759-761);
/// * a `UInt32`/`Int32` image leaves the full **256-byte header**, then
///   `"Pixel Type Unknown"` from `SwapBytesIfNecessary` (`:1010`/`:1024`).
///
/// `WriteImageInformation` is a no-op upstream ("not possible to write a Gipl
/// file", `:655-659`); the header is emitted by `Write` alone.
pub fn write(img: &Image, path: &Path) -> Result<()> {
    if is_compressed(path) {
        return Err(zlib_unavailable("write"));
    }
    let n = img.dimension();
    let size = img.size();
    let mut out = Vec::with_capacity(HEADER_SIZE);

    // dims: the image's first four axes, then `1`. `unsigned short value =
    // this->GetDimensions(i)` truncates an axis longer than 65535 (`:689`).
    for i in 0..4usize {
        let value: u16 = size.get(i).map_or(1, |&s| s as u16);
        out.extend_from_slice(&value.to_be_bytes());
    }

    let component = img.buffer().component_id();
    let Some(image_type) = image_type_code(component) else {
        std::fs::write(path, &out)?;
        return Err(IoError::UnsupportedGiplFeature(format!(
            "Invalid type: {} (GiplImageIO::Write has no image_type for it; \
             the file keeps the 8 dims bytes already written)",
            component.as_str()
        )));
    };
    out.extend_from_slice(&image_type.to_be_bytes());

    // pixdim: spacing, then 1.0.
    for i in 0..4 {
        let value: f32 = if i < n { img.spacing()[i] as f32 } else { 1.0 };
        out.extend_from_slice(&value.to_be_bytes());
    }

    // line1: 80 bytes, zeroed then `snprintf`ed.
    out.extend_from_slice(PATIENT_TEXT);
    out.resize(out.len() + (80 - PATIENT_TEXT.len()), 0);

    // matrix: 20 zeroed floats. flag1, flag2: zero. min, max: zeroed doubles,
    // written without a byte swap (a no-op on zero, as upstream's own read of
    // them is).
    out.resize(out.len() + 20 * 4 + 1 + 1 + 8 + 8, 0);

    // origin: the image's, then 0.0.
    for i in 0..4 {
        let value: f64 = if i < n { img.origin()[i] } else { 0.0 };
        out.extend_from_slice(&value.to_be_bytes());
    }

    // pixval_offset, pixval_cal, user_def1, user_def2: zeroed floats.
    out.resize(out.len() + 4 * 4, 0);
    out.extend_from_slice(&GIPL_MAGIC_NUMBER.to_be_bytes());
    debug_assert_eq!(out.len(), HEADER_SIZE);

    if !is_swappable(component) {
        std::fs::write(path, &out)?;
        return Err(pixel_type_unknown(component));
    }

    // `GetImageSizeInBytes()` counts every component, so a vector or complex
    // image writes more bytes than its scalar `image_type` describes (§2.96).
    out.extend_from_slice(&buffer_to_be_bytes(img.buffer()));
    std::fs::write(path, &out)?;
    Ok(())
}

/// `itk::GiplImageIO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GiplImageIo;

impl ImageIo for GiplImageIo {
    fn name(&self) -> &'static str {
        "GiplImageIO"
    }

    /// Empty, faithfully: `GiplImageIO`'s constructor calls no
    /// `AddSupportedReadExtension` (itkGiplImageIO.cxx:75-79), so upstream's
    /// factory can only reach it in the extension-blind second probe phase.
    fn supported_read_extensions(&self) -> &'static [&'static str] {
        &[]
    }

    /// Empty for the same reason; [`GiplImageIo::can_write_file`] is the gate.
    fn supported_write_extensions(&self) -> &'static [&'static str] {
        &[]
    }

    /// `CanReadFile` (itkGiplImageIO.cxx:97-175): the extension, then the magic
    /// number at offset 252.
    ///
    /// A `.gipl.gz` needs zlib to reach its magic number, so this port claims it
    /// on the name alone and lets [`read`] name the missing dependency. Ledger
    /// §4.68.
    fn can_read_file(&self, path: &Path) -> bool {
        use std::io::{Read, Seek, SeekFrom};

        match check_extension(path) {
            None => false,
            Some(true) => path.is_file(),
            Some(false) => {
                let Ok(mut file) = std::fs::File::open(path) else {
                    return false;
                };
                if file.seek(SeekFrom::Start(252)).is_err() {
                    return false;
                }
                let mut magic = [0u8; 4];
                if file.read_exact(&mut magic).is_err() {
                    return false;
                }
                let magic = u32::from_be_bytes(magic);
                magic == GIPL_MAGIC_NUMBER || magic == GIPL_MAGIC_NUMBER2
            }
        }
    }

    /// `CanWriteFile` is `CheckExtension` alone (itkGiplImageIO.cxx:177-196), so
    /// `.gipl.gz` is claimed for writing and then refused inside [`write`].
    fn can_write_file(&self, path: &Path) -> bool {
        check_extension(path).is_some()
    }

    fn read_information(&self, path: &Path) -> Result<ImageInformation> {
        read_information(path)
    }

    fn read(&self, path: &Path) -> Result<Image> {
        read(path)
    }

    fn write(&self, image: &Image, path: &Path) -> Result<()> {
        write(image, path)
    }
}
