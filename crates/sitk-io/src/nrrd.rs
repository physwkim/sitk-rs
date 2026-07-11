//! NRRD (`.nrrd` attached / `.nhdr` + data file) reader and writer —
//! `itk::NrrdImageIO` layered over the vendored NrrdIO C library.
//!
//! Two upstream layers are ported here, and they are kept distinct in the code
//! because they are distinct upstream:
//!
//! * **NrrdIO** (`Modules/ThirdParty/NrrdIO/src/NrrdIO/`) — the header grammar,
//!   the field enums, the encodings, the data-file forms. [`Nrrd`] and
//!   [`IoState`] below mirror teem's `Nrrd` and `NrrdIoState` structs, and
//!   [`read_header`] mirrors `formatNRRD_read` (formatNRRD.c:430-533) plus the
//!   per-field parsers in parseNrrd.c.
//! * **NrrdImageIO** (`Modules/IO/NRRD/src/itkNrrdImageIO.cxx`) — the mapping
//!   from a `Nrrd` to an ITK image: which axis becomes the pixel component
//!   axis, how `space directions` decomposes into spacing and direction, how
//!   `space` is converted to LPS, and what lands in the meta-data dictionary.
//!
//! What this port does *not* have is teem's `nrrdLoad`; the read path decodes
//! the pixel bytes directly. Everything else follows the upstream sources line
//! by line, upstream bugs included (see the ledger rows cited below).
//!
//! # Axes: domain, range, and the pixel component axis
//!
//! NRRD tags every axis with a `kind`. `nrrdDomainAxesGet` (axis.c:1077-1096)
//! calls an axis a *domain* axis when its kind is unknown, `domain`, `space`,
//! or `time`; every other kind is a *range* axis.
//! `GetAxisOrderForFileReading` (itkNrrdImageIO.cxx:41-126) then picks the
//! **first range axis whose kind is not `list`** as the pixel component axis;
//! failing that — an image whose only range axes are `list` — it takes
//! `rangeAxes[0]` anyway, because `AxesReorder::UseAnyRangeAxisAsPixel` is the
//! default (itkNrrdImageIO.h:228). The ITK image's axes are then the domain
//! axes followed by whatever range axes were left over, and the pixel axis is
//! permuted to be the fastest one if it was not already
//! (itkNrrdImageIO.cxx:1146-1170).
//!
//! So `kinds: vector domain domain` and `kinds: domain domain vector` name the
//! same ITK image; only the second needs a permutation. `kinds: list domain
//! domain` yields a *vector* image, not a 3-D scalar one.
//!
//! A `none` entry in `space directions` can therefore only ever sit on a range
//! axis: `nFieldCheckSpaceInfo` (simple.c) forbids a domain axis from carrying
//! both a space direction and a spacing/min/max, and ITK's spacing loop visits
//! domain axes only. The read path never looks at a range axis's space vector.
//!
//! # Spacing and direction
//!
//! `nrrdSpacingCalculate` (axis.c:921-958) is the whole rule, per domain axis:
//!
//! * `spacings` present and `spaceDim > 0` — `ScalarWithSpace`, which ITK
//!   rejects outright (itkNrrdImageIO.cxx:820-826);
//! * `spacings` present and no space — spacing is that number, direction stays
//!   identity;
//! * a space direction vector `V` — spacing is `|V|` and the direction column
//!   is `V / |V|`, sign-flipped per the LPS conversion below;
//! * neither — ITK leaves its defaults (spacing `1.0`, identity direction).
//!
//! `axis mins` / `axis maxs` never touch the spacing. They feed
//! `nrrdOriginCalculate` (simple.c:229-302), which is consulted only when
//! `spaceDim == 0`. Upstream bug [§1.47](../../../doc/upstream-findings.md) —
//! its `gotMin` loop tests `axis[0]->min` rather than `axis[ai]->min` — is
//! fixed in this port: each axis's own `min` is checked, so `axis mins: 0
//! nan` correctly reports the `NoMin` status and leaves the origin at zero.
//!
//! # `space` and the conversion to LPS
//!
//! `ReadImageInformation` (itkNrrdImageIO.cxx:764-786) flips axis signs to bring
//! the anatomical spaces to LPS: `right-anterior-superior` (flip axes 0 and 1),
//! `left-anterior-superior` (flip axis 1) and `left-posterior-superior` (no
//! flip). Upstream's `switch` then has a bare `default:` arm, so `scanner-xyz`,
//! `right-up`, `3D-left-handed` and every `*-time` space fall through and their
//! direction vectors are used **unconverted** — silently loaded as if already
//! LPS. This port **fixes** that (ledger §2.82): it additionally converts the
//! `-time` anatomical spaces (`right-anterior-superior-time` etc.), whose flip
//! is purely spatial and therefore well-defined, and it *rejects* a named
//! non-anatomical space — `scanner-xyz`, `right-up`, `3D-right-handed` and
//! friends — with an [`IoError::UnsupportedNrrdFeature`] rather than loading it
//! mis-oriented. The unknown space (no `space:` field) is used verbatim, as
//! before. The same sign flips are applied to `space origin`, and (only when
//! there are at most three domain axes) to the columns of `measurement frame`.
//!
//! # Pixel types
//!
//! The pixel axis's kind selects an `IOPixelEnum` (itkNrrdImageIO.cxx:701-759),
//! and `ImageReaderBase::GetPixelIDFromImageIO` (sitkImageReaderBase.cxx:
//! 215-240) turns that into a [`PixelId`]. Unlike MetaImage, NRRD **does**
//! round-trip a complex image: `kinds: complex` produces `IOPixelEnum::COMPLEX`
//! with two components, which SimpleITK maps to `ComplexFloat32`/`ComplexFloat64`.
//! `3D-symmetric-matrix` produces `SYMMETRICSECONDRANKTENSOR`, for which
//! SimpleITK's `GetPixelIDFromImageIO` has no pixel id and raises "Unknown
//! PixelType" — even though `itk::Image` reads the file fine. This port
//! **implements** the read (ledger §3.31): the tensor is loaded as a vector
//! image whose components are the unique matrix entries in the NRRD on-disk
//! order (`Dxx Dxy Dxz Dyy Dyz Dzz` for the 6-component symmetric matrix). A
//! `3D-masked-symmetric-matrix` carries a leading mask channel that upstream's
//! `Read` crops out; this port drops it too, so the vector image holds only the
//! six matrix entries.
//!
//! # Encodings
//!
//! `raw` and `gzip` are read and written; `gzip` is what
//! `NrrdImageIO::Write` emits under `SetUseCompression(true)`, because
//! `InternalSetCompressor("")` — the default compressor — resolves to
//! `nrrdEncodingGzip` (itkNrrdImageIO.cxx:380-392, 1404-1409). `ascii` (whose
//! canonical *name*, and therefore the string written into a header, is `ASCII`
//! — encodingAscii.c) is read only, which loses nothing: `GetFileType()` is
//! `Binary` for everything SimpleITK writes. `bzip2`, `hex` and `zrl` are
//! recognised and rejected: `bzip2` needs a dependency this workspace does not
//! take, and the other two have no ITK write path (ledger §5.8, §4.53).
//!
//! `line skip` and `byte skip` are *not* symmetric across encodings. For a
//! non-compression encoding both are applied to the file stream. For `gzip`
//! only `line skip` is (formatNRRD.c:579-585); the byte skip is performed by
//! `encodingGzip_read` on the **decompressed** buffer, which is why a negative
//! `byte skip` — rejected everywhere else but `raw` — is legal with `gzip` and
//! means "the data is the tail of the inflated stream"
//! (encodingGzip.c:81-140). A gzip stream without the two magic bytes is read
//! transparently, as a byte copy (ledger §2.113).
//!
//! # Header grammar
//!
//! `nrrd__ReadNrrdParseField` (parseNrrd.c) splits a header line on the first
//! `": "`; if the text before it is a known field name (matched
//! case-insensitively against `fieldStrEqv`, so `axismins` and `axis mins` are
//! the same field) the rest of the line is that field's value. Otherwise the
//! line must contain `":="`, and it is a key/value pair whose key and value are
//! `airUnescape`d (`\n` and `\\` only). A leading `#` is a comment. Every field
//! but comment, content, key/value and `data file` has its trailing spaces and
//! tabs stripped (formatNRRD.c:491-510). A field other than comment/key-value
//! appearing twice is an error.
//!
//! A blank line ends the header. For an attached `.nrrd` the pixel data starts
//! on the very next byte; for a detached `.nhdr` there is no blank line and the
//! `data file:` field names the data.
//!
//! `data file:` has three forms (parseNrrd.c:1231-1414). A bare filename;
//! `LIST [<dim>]` followed by one filename per line; and a `%d` printf template
//! with `<min> <max> <step> [<dim>]`. This port implements the first two and
//! rejects the template form and `SKIPLIST`. Filenames are header-relative
//! unless they are `-`, start with `/`, or have `:` at index 1.
//!
//! # Deliberate divergences
//!
//! * `sizes`, `spacings`, `thicknesses`, `axis mins` and `axis maxs` are parsed
//!   into a `dim`-long vector; upstream parses `dim + 1` values into a
//!   `NRRD_DIM_MAX`-long stack array to detect excess tokens, which overflows
//!   that array when `dim == 16` (ledger §1.46, §4.51).
//! * The meta-data dictionary of this port is `String -> String`, where ITK's
//!   is type-erased. Doubles (`NRRD_thicknesses[i]`, `NRRD_old min`) and the
//!   measurement frame are therefore stringified on read and parsed back on
//!   write. Ledger §4.52.
//! * `encoding: ascii` (and its `text`/`txt` spellings) is read but never
//!   written — ITK's writer reaches it only through `SetFileType(ASCII)`, which
//!   SimpleITK never calls. `bzip2` is recognised and rejected for want of a
//!   dependency (ledger §5.8); `hex` and `zrl` likewise, having no write path.
//!   `data file:` supports the bare-filename and `LIST` forms only.
//!   Ledger §4.53.
//! * The `kinds`/`sizes` consistency check (`nrrdKindSize`) runs unconditionally
//!   at the end of header parsing, where upstream reaches it later, by way of
//!   `nrrdSpacingCalculate`. Same predicate, earlier. Ledger §4.54.
//!
//! # Reproduced upstream quirks
//!
//! * A `data file: LIST` reads filenames until EOF, taking a blank line as an
//!   (empty) filename rather than a terminator (parseNrrd.c:1359-1398) —
//!   ledger §2.83.
//! * The writer emits comments and key/value pairs *after* the `data file:`
//!   field, which corrupts a `LIST` header it just wrote (write.c:726-728,
//!   formatNRRD.c) — ledger §2.84.
//! * `NRRD_`-prefixed meta-data keys are dispatched by leading-substring match,
//!   so `NRRD_space directions` is swallowed by the `space` handler and
//!   `NRRD_pixel_original_axis` — a key the reader itself writes — matches
//!   nothing and is silently dropped (itkNrrdImageIO.cxx:292-311) — ledger
//!   §2.85.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sitk_core::{Complex, Image, PixelBuffer, PixelId};

use crate::compression::{ITK_DEFAULT_COMPRESSION_LEVEL, gunzip_transparent, gzip_compress};
use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, ImageIo};
use crate::writer::WriteOptions;

/// `NRRD_DIM_MAX` (nrrdDefines.h).
const DIM_MAX: usize = 16;
/// `NRRD_SPACE_DIM_MAX` (nrrdDefines.h).
const SPACE_DIM_MAX: usize = 8;
/// `nrrd__FieldSep` (nrrdDefines.h): the only characters that separate values
/// inside a field. Note that `\r` is not one of them.
const FIELD_SEP: [char; 2] = [' ', '\t'];

fn bad(msg: impl Into<String>) -> IoError {
    IoError::MalformedNrrdHeader(msg.into())
}

fn unsupported(msg: impl Into<String>) -> IoError {
    IoError::UnsupportedNrrdFeature(msg.into())
}

// ---------------------------------------------------------------------------
// airEnum
// ---------------------------------------------------------------------------

/// `airEnumVal` for an enum whose `sense` is `AIR_FALSE`: case-insensitive
/// whole-string match against the `strEqv` table, `0` (unknown) on no match
/// (enum.c).
fn air_enum_val(eqv: &[(&str, u32)], s: &str) -> u32 {
    eqv.iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(s))
        .map(|(_, v)| *v)
        .unwrap_or(0)
}

/// `nrrdType` (enumsNrrd.c:109-194). The index *is* the teem enum value.
const TYPE_STR: [&str; 12] = [
    "(unknown_type)",
    "signed char",
    "unsigned char",
    "short",
    "unsigned short",
    "int",
    "unsigned int",
    "long long int",
    "unsigned long long int",
    "float",
    "double",
    "block",
];

const TYPE_CHAR: u32 = 1;
const TYPE_UCHAR: u32 = 2;
const TYPE_SHORT: u32 = 3;
const TYPE_USHORT: u32 = 4;
const TYPE_INT: u32 = 5;
const TYPE_UINT: u32 = 6;
const TYPE_LLONG: u32 = 7;
const TYPE_ULLONG: u32 = 8;
const TYPE_FLOAT: u32 = 9;
const TYPE_DOUBLE: u32 = 10;
const TYPE_BLOCK: u32 = 11;

/// `typeStrEqv` / `typeValEqv` (enumsNrrd.c:152-183). Note that bare `"char"`
/// is deliberately absent.
const TYPE_EQV: &[(&str, u32)] = &[
    ("signed char", TYPE_CHAR),
    ("int8", TYPE_CHAR),
    ("int8_t", TYPE_CHAR),
    ("uchar", TYPE_UCHAR),
    ("unsigned char", TYPE_UCHAR),
    ("uint8", TYPE_UCHAR),
    ("uint8_t", TYPE_UCHAR),
    ("short", TYPE_SHORT),
    ("short int", TYPE_SHORT),
    ("signed short", TYPE_SHORT),
    ("signed short int", TYPE_SHORT),
    ("int16", TYPE_SHORT),
    ("int16_t", TYPE_SHORT),
    ("ushort", TYPE_USHORT),
    ("unsigned short", TYPE_USHORT),
    ("unsigned short int", TYPE_USHORT),
    ("uint16", TYPE_USHORT),
    ("uint16_t", TYPE_USHORT),
    ("int", TYPE_INT),
    ("signed int", TYPE_INT),
    ("int32", TYPE_INT),
    ("int32_t", TYPE_INT),
    ("uint", TYPE_UINT),
    ("unsigned int", TYPE_UINT),
    ("uint32", TYPE_UINT),
    ("uint32_t", TYPE_UINT),
    ("longlong", TYPE_LLONG),
    ("long long", TYPE_LLONG),
    ("long long int", TYPE_LLONG),
    ("signed long long", TYPE_LLONG),
    ("signed long long int", TYPE_LLONG),
    ("int64", TYPE_LLONG),
    ("int64_t", TYPE_LLONG),
    ("ulonglong", TYPE_ULLONG),
    ("unsigned long long", TYPE_ULLONG),
    ("unsigned long long int", TYPE_ULLONG),
    ("uint64", TYPE_ULLONG),
    ("uint64_t", TYPE_ULLONG),
    ("float", TYPE_FLOAT),
    ("double", TYPE_DOUBLE),
    ("block", TYPE_BLOCK),
];

/// `nrrdElementSize` for a non-block type.
fn type_size(t: u32) -> usize {
    match t {
        TYPE_CHAR | TYPE_UCHAR => 1,
        TYPE_SHORT | TYPE_USHORT => 2,
        TYPE_INT | TYPE_UINT | TYPE_FLOAT => 4,
        TYPE_LLONG | TYPE_ULLONG | TYPE_DOUBLE => 8,
        _ => 0,
    }
}

/// `NrrdToITKComponentType` (itkNrrdImageIO.cxx) composed with SimpleITK's
/// component-type mapping: the scalar [`PixelId`] a NRRD type reads back as.
fn type_component_id(t: u32) -> Option<PixelId> {
    Some(match t {
        TYPE_CHAR => PixelId::Int8,
        TYPE_UCHAR => PixelId::UInt8,
        TYPE_SHORT => PixelId::Int16,
        TYPE_USHORT => PixelId::UInt16,
        TYPE_INT => PixelId::Int32,
        TYPE_UINT => PixelId::UInt32,
        TYPE_LLONG => PixelId::Int64,
        TYPE_ULLONG => PixelId::UInt64,
        TYPE_FLOAT => PixelId::Float32,
        TYPE_DOUBLE => PixelId::Float64,
        _ => return None,
    })
}

/// `ITKToNrrdComponentType`, the write-side inverse of [`type_component_id`].
fn component_id_type(id: PixelId) -> u32 {
    match id.component_id() {
        PixelId::Int8 => TYPE_CHAR,
        PixelId::UInt8 => TYPE_UCHAR,
        PixelId::Int16 => TYPE_SHORT,
        PixelId::UInt16 => TYPE_USHORT,
        PixelId::Int32 => TYPE_INT,
        PixelId::UInt32 => TYPE_UINT,
        PixelId::Int64 => TYPE_LLONG,
        PixelId::UInt64 => TYPE_ULLONG,
        PixelId::Float32 => TYPE_FLOAT,
        PixelId::Float64 => TYPE_DOUBLE,
        // `component_id()` only ever yields the ten scalar ids.
        other => unreachable!("component_id returned {other:?}"),
    }
}

const ENC_RAW: u32 = 1;
const ENC_ASCII: u32 = 2;
const ENC_HEX: u32 = 3;
const ENC_GZIP: u32 = 4;
const ENC_BZIP2: u32 = 5;
const ENC_ZRL: u32 = 6;

/// `encodingTypeStrEqv` / `encodingTypeValEqv` (enumsNrrd.c:220-238).
const ENC_EQV: &[(&str, u32)] = &[
    ("raw", ENC_RAW),
    ("txt", ENC_ASCII),
    ("text", ENC_ASCII),
    ("ascii", ENC_ASCII),
    ("hex", ENC_HEX),
    ("gz", ENC_GZIP),
    ("gzip", ENC_GZIP),
    ("bz2", ENC_BZIP2),
    ("bzip2", ENC_BZIP2),
    ("zrl", ENC_ZRL),
];

/// `NrrdEncoding::endianMatters` (encoding*.c). Only `ascii` and `hex` are
/// byte-order agnostic.
fn encoding_endian_matters(enc: u32) -> bool {
    !matches!(enc, ENC_ASCII | ENC_HEX)
}

/// `NrrdEncoding::isCompression` (encoding*.c). `formatNRRD_read` skips the
/// `line skip` / `byte skip` pair for these, because the byte skip happens
/// *inside* the decompressed stream (formatNRRD.c:583-585).
fn encoding_is_compression(enc: u32) -> bool {
    matches!(enc, ENC_GZIP | ENC_BZIP2)
}

/// `NrrdEncoding::name`, the spelling `nrrd__FprintFieldInfo` writes.
fn encoding_name(enc: u32) -> &'static str {
    match enc {
        ENC_RAW => "raw",
        ENC_ASCII => "ascii",
        ENC_HEX => "hex",
        ENC_GZIP => "gzip",
        ENC_BZIP2 => "bzip2",
        ENC_ZRL => "zrl",
        other => unreachable!("no encoding {other}"),
    }
}

/// `NrrdEncoding::suffix`, the extension a detached `.nhdr` gives its data file
/// (`"raw"`, encodingRaw.c:153; `"raw.gz"`, encodingGzip.c:302). It is glued on
/// with a `.` by `formatNRRD_write` (formatNRRD.c:720-728).
fn encoding_suffix(enc: u32) -> &'static str {
    match enc {
        ENC_GZIP => "raw.gz",
        _ => "raw",
    }
}

const ENDIAN_LITTLE: u32 = 1;
const ENDIAN_BIG: u32 = 2;

/// `airEndian` (endianAir.c).
const ENDIAN_EQV: &[(&str, u32)] = &[("little", ENDIAN_LITTLE), ("big", ENDIAN_BIG)];

const CENTER_NODE: u32 = 1;
const CENTER_CELL: u32 = 2;

/// `centerStr` (enumsNrrd.c:253-257). The centering enum has no `strEqv`.
const CENTER_EQV: &[(&str, u32)] = &[("node", CENTER_NODE), ("cell", CENTER_CELL)];
const CENTER_STR: [&str; 3] = ["(unknown_center)", "node", "cell"];

const KIND_DOMAIN: u32 = 1;
const KIND_SPACE: u32 = 2;
const KIND_TIME: u32 = 3;
const KIND_LIST: u32 = 4;
const KIND_POINT: u32 = 5;
const KIND_VECTOR: u32 = 6;
const KIND_COVARIANT_VECTOR: u32 = 7;
const KIND_NORMAL: u32 = 8;
const KIND_STUB: u32 = 9;
const KIND_SCALAR: u32 = 10;
const KIND_COMPLEX: u32 = 11;
const KIND_2VECTOR: u32 = 12;
const KIND_3COLOR: u32 = 13;
const KIND_RGB_COLOR: u32 = 14;
const KIND_HSV_COLOR: u32 = 15;
const KIND_XYZ_COLOR: u32 = 16;
const KIND_4COLOR: u32 = 17;
const KIND_RGBA_COLOR: u32 = 18;
const KIND_3VECTOR: u32 = 19;
const KIND_3GRADIENT: u32 = 20;
const KIND_3NORMAL: u32 = 21;
const KIND_4VECTOR: u32 = 22;
const KIND_QUATERNION: u32 = 23;
const KIND_2D_SYM_MATRIX: u32 = 24;
const KIND_2D_MASKED_SYM_MATRIX: u32 = 25;
const KIND_2D_MATRIX: u32 = 26;
const KIND_2D_MASKED_MATRIX: u32 = 27;
const KIND_3D_SYM_MATRIX: u32 = 28;
const KIND_3D_MASKED_SYM_MATRIX: u32 = 29;
const KIND_3D_MATRIX: u32 = 30;
const KIND_3D_MASKED_MATRIX: u32 = 31;

/// `kindStr` (enumsNrrd.c:321-354); the index is the teem enum value.
const KIND_STR: [&str; 32] = [
    "(unknown_kind)",
    "domain",
    "space",
    "time",
    "list",
    "point",
    "vector",
    "covariant-vector",
    "normal",
    "stub",
    "scalar",
    "complex",
    "2-vector",
    "3-color",
    "RGB-color",
    "HSV-color",
    "XYZ-color",
    "4-color",
    "RGBA-color",
    "3-vector",
    "3-gradient",
    "3-normal",
    "4-vector",
    "quaternion",
    "2D-symmetric-matrix",
    "2D-masked-symmetric-matrix",
    "2D-matrix",
    "2D-masked-matrix",
    "3D-symmetric-matrix",
    "3D-masked-symmetric-matrix",
    "3D-matrix",
    "3D-masked-matrix",
];

/// `kindStr_Eqv` / `kindVal_Eqv` (enumsNrrd.c:392-477).
const KIND_EQV: &[(&str, u32)] = &[
    ("domain", KIND_DOMAIN),
    ("space", KIND_SPACE),
    ("time", KIND_TIME),
    ("list", KIND_LIST),
    ("point", KIND_POINT),
    ("vector", KIND_VECTOR),
    ("contravariant-vector", KIND_VECTOR),
    ("covariant-vector", KIND_COVARIANT_VECTOR),
    ("normal", KIND_NORMAL),
    ("stub", KIND_STUB),
    ("scalar", KIND_SCALAR),
    ("complex", KIND_COMPLEX),
    ("2-vector", KIND_2VECTOR),
    ("3-color", KIND_3COLOR),
    ("RGB-color", KIND_RGB_COLOR),
    ("RGBcolor", KIND_RGB_COLOR),
    ("RGB", KIND_RGB_COLOR),
    ("HSV-color", KIND_HSV_COLOR),
    ("HSVcolor", KIND_HSV_COLOR),
    ("HSV", KIND_HSV_COLOR),
    ("XYZ-color", KIND_XYZ_COLOR),
    ("4-color", KIND_4COLOR),
    ("RGBA-color", KIND_RGBA_COLOR),
    ("RGBAcolor", KIND_RGBA_COLOR),
    ("RGBA", KIND_RGBA_COLOR),
    ("3-vector", KIND_3VECTOR),
    ("3-gradient", KIND_3GRADIENT),
    ("3-normal", KIND_3NORMAL),
    ("4-vector", KIND_4VECTOR),
    ("quaternion", KIND_QUATERNION),
    ("2D-symmetric-matrix", KIND_2D_SYM_MATRIX),
    ("2D-sym-matrix", KIND_2D_SYM_MATRIX),
    ("2D-symmetric-tensor", KIND_2D_SYM_MATRIX),
    ("2D-sym-tensor", KIND_2D_SYM_MATRIX),
    ("2D-masked-symmetric-matrix", KIND_2D_MASKED_SYM_MATRIX),
    ("2D-masked-sym-matrix", KIND_2D_MASKED_SYM_MATRIX),
    ("2D-masked-symmetric-tensor", KIND_2D_MASKED_SYM_MATRIX),
    ("2D-masked-sym-tensor", KIND_2D_MASKED_SYM_MATRIX),
    ("2D-matrix", KIND_2D_MATRIX),
    ("2D-tensor", KIND_2D_MATRIX),
    ("2D-masked-matrix", KIND_2D_MASKED_MATRIX),
    ("2D-masked-tensor", KIND_2D_MASKED_MATRIX),
    ("3D-symmetric-matrix", KIND_3D_SYM_MATRIX),
    ("3D-sym-matrix", KIND_3D_SYM_MATRIX),
    ("3D-symmetric-tensor", KIND_3D_SYM_MATRIX),
    ("3D-sym-tensor", KIND_3D_SYM_MATRIX),
    ("3D-masked-symmetric-matrix", KIND_3D_MASKED_SYM_MATRIX),
    ("3D-masked-sym-matrix", KIND_3D_MASKED_SYM_MATRIX),
    ("3D-masked-symmetric-tensor", KIND_3D_MASKED_SYM_MATRIX),
    ("3D-masked-sym-tensor", KIND_3D_MASKED_SYM_MATRIX),
    ("3D-matrix", KIND_3D_MATRIX),
    ("3D-tensor", KIND_3D_MATRIX),
    ("3D-masked-matrix", KIND_3D_MASKED_MATRIX),
    ("3D-masked-tensor", KIND_3D_MASKED_MATRIX),
];

/// `nrrdKindSize` (axis.c). `0` means "any size".
fn kind_size(kind: u32) -> usize {
    match kind {
        KIND_STUB | KIND_SCALAR => 1,
        KIND_COMPLEX | KIND_2VECTOR => 2,
        KIND_3COLOR | KIND_RGB_COLOR | KIND_HSV_COLOR | KIND_XYZ_COLOR => 3,
        KIND_4COLOR | KIND_RGBA_COLOR => 4,
        KIND_3VECTOR | KIND_3GRADIENT | KIND_3NORMAL => 3,
        KIND_4VECTOR | KIND_QUATERNION => 4,
        KIND_2D_SYM_MATRIX => 3,
        KIND_2D_MASKED_SYM_MATRIX => 4,
        KIND_2D_MATRIX => 4,
        KIND_2D_MASKED_MATRIX => 5,
        KIND_3D_SYM_MATRIX => 6,
        KIND_3D_MASKED_SYM_MATRIX => 7,
        KIND_3D_MATRIX => 9,
        KIND_3D_MASKED_MATRIX => 10,
        _ => 0,
    }
}

/// `nrrdKindIsDomain` (axis.c) widened by `nrrdDomainAxesGet`, which also
/// counts the unknown kind as a domain axis.
fn kind_is_domain(kind: u32) -> bool {
    matches!(kind, 0 | KIND_DOMAIN | KIND_SPACE | KIND_TIME)
}

const SPACE_RIGHT_UP: u32 = 1;
const SPACE_RIGHT_DOWN: u32 = 2;
const SPACE_RAS: u32 = 3;
const SPACE_LAS: u32 = 4;
const SPACE_LPS: u32 = 5;
const SPACE_RAST: u32 = 6;
const SPACE_LAST: u32 = 7;
const SPACE_LPST: u32 = 8;
const SPACE_SCANNER_XYZ: u32 = 9;
const SPACE_SCANNER_XYZ_TIME: u32 = 10;
const SPACE_3D_RIGHT: u32 = 11;
const SPACE_3D_LEFT: u32 = 12;
const SPACE_3D_RIGHT_TIME: u32 = 13;
const SPACE_3D_LEFT_TIME: u32 = 14;

/// `spaceStr` (enumsNrrd.c:678-694).
const SPACE_STR: [&str; 15] = [
    "(unknown_space)",
    "right-up",
    "right-down",
    "right-anterior-superior",
    "left-anterior-superior",
    "left-posterior-superior",
    "right-anterior-superior-time",
    "left-anterior-superior-time",
    "left-posterior-superior-time",
    "scanner-xyz",
    "scanner-xyz-time",
    "3D-right-handed",
    "3D-left-handed",
    "3D-right-handed-time",
    "3D-left-handed-time",
];

/// `spaceStrEqv` / `spaceValEqv` (enumsNrrd.c:715-739).
const SPACE_EQV: &[(&str, u32)] = &[
    ("right-up", SPACE_RIGHT_UP),
    ("right up", SPACE_RIGHT_UP),
    ("right-down", SPACE_RIGHT_DOWN),
    ("right down", SPACE_RIGHT_DOWN),
    ("right-anterior-superior", SPACE_RAS),
    ("right anterior superior", SPACE_RAS),
    ("rightanteriorsuperior", SPACE_RAS),
    ("RAS", SPACE_RAS),
    ("left-anterior-superior", SPACE_LAS),
    ("left anterior superior", SPACE_LAS),
    ("leftanteriorsuperior", SPACE_LAS),
    ("LAS", SPACE_LAS),
    ("left-posterior-superior", SPACE_LPS),
    ("left posterior superior", SPACE_LPS),
    ("leftposteriorsuperior", SPACE_LPS),
    ("LPS", SPACE_LPS),
    ("right-anterior-superior-time", SPACE_RAST),
    ("right anterior superior time", SPACE_RAST),
    ("rightanteriorsuperiortime", SPACE_RAST),
    ("RAST", SPACE_RAST),
    ("left-anterior-superior-time", SPACE_LAST),
    ("left anterior superior time", SPACE_LAST),
    ("leftanteriorsuperiortime", SPACE_LAST),
    ("LAST", SPACE_LAST),
    ("left-posterior-superior-time", SPACE_LPST),
    ("left posterior superior time", SPACE_LPST),
    ("leftposteriorsuperiortime", SPACE_LPST),
    ("LPST", SPACE_LPST),
    ("scanner-xyz", SPACE_SCANNER_XYZ),
    ("scanner-xyz-time", SPACE_SCANNER_XYZ_TIME),
    ("scanner-xyzt", SPACE_SCANNER_XYZ_TIME),
    ("3D-right-handed", SPACE_3D_RIGHT),
    ("3D right handed", SPACE_3D_RIGHT),
    ("3Drighthanded", SPACE_3D_RIGHT),
    ("3D-left-handed", SPACE_3D_LEFT),
    ("3D left handed", SPACE_3D_LEFT),
    ("3Dlefthanded", SPACE_3D_LEFT),
    ("3D-right-handed-time", SPACE_3D_RIGHT_TIME),
    ("3D right handed time", SPACE_3D_RIGHT_TIME),
    ("3Drighthandedtime", SPACE_3D_RIGHT_TIME),
    ("3D-left-handed-time", SPACE_3D_LEFT_TIME),
    ("3D left handed time", SPACE_3D_LEFT_TIME),
    ("3Dlefthandedtime", SPACE_3D_LEFT_TIME),
];

/// `nrrdSpaceDimension` (simple.c). `0` for the unknown space.
fn space_dimension(space: u32) -> usize {
    match space {
        SPACE_RIGHT_UP | SPACE_RIGHT_DOWN => 2,
        SPACE_RAS | SPACE_LAS | SPACE_LPS | SPACE_SCANNER_XYZ | SPACE_3D_RIGHT | SPACE_3D_LEFT => 3,
        SPACE_RAST
        | SPACE_LAST
        | SPACE_LPST
        | SPACE_SCANNER_XYZ_TIME
        | SPACE_3D_RIGHT_TIME
        | SPACE_3D_LEFT_TIME => 4,
        _ => 0,
    }
}

// Field enum values; the index into `fieldStr` (enumsNrrd.c:493-530), which is
// also the order `formatNRRD_write` emits them in.
const F_COMMENT: usize = 1;
const F_CONTENT: usize = 2;
const F_NUMBER: usize = 3;
const F_TYPE: usize = 4;
const F_BLOCK_SIZE: usize = 5;
const F_DIMENSION: usize = 6;
const F_SPACE: usize = 7;
const F_SPACE_DIMENSION: usize = 8;
const F_SIZES: usize = 9;
const F_SPACINGS: usize = 10;
const F_THICKNESSES: usize = 11;
const F_AXIS_MINS: usize = 12;
const F_AXIS_MAXS: usize = 13;
const F_SPACE_DIRECTIONS: usize = 14;
const F_CENTERS: usize = 15;
const F_KINDS: usize = 16;
const F_LABELS: usize = 17;
const F_UNITS: usize = 18;
const F_MIN: usize = 19;
const F_MAX: usize = 20;
const F_OLD_MIN: usize = 21;
const F_OLD_MAX: usize = 22;
const F_ENDIAN: usize = 23;
const F_ENCODING: usize = 24;
const F_LINE_SKIP: usize = 25;
const F_BYTE_SKIP: usize = 26;
const F_KEYVALUE: usize = 27;
const F_SAMPLE_UNITS: usize = 28;
const F_SPACE_UNITS: usize = 29;
const F_SPACE_ORIGIN: usize = 30;
const F_MEASUREMENT_FRAME: usize = 31;
const F_DATA_FILE: usize = 32;
const F_MAX_FIELD: usize = 32;

/// `fieldStr` (enumsNrrd.c:493-530): the canonical spelling the *writer* uses.
/// Note `centerings`, where the reader also accepts `centers`.
const FIELD_STR: [&str; 33] = [
    "(unknown_field)",
    "#",
    "content",
    "number",
    "type",
    "block size",
    "dimension",
    "space",
    "space dimension",
    "sizes",
    "spacings",
    "thicknesses",
    "axis mins",
    "axis maxs",
    "space directions",
    "centerings",
    "kinds",
    "labels",
    "units",
    "min",
    "max",
    "old min",
    "old max",
    "endian",
    "encoding",
    "line skip",
    "byte skip",
    "key/value",
    "sample units",
    "space units",
    "space origin",
    "measurement frame",
    "data file",
];

/// `fieldStrEqv` / `fieldValEqv` (enumsNrrd.c:571-641).
const FIELD_EQV: &[(&str, u32)] = &[
    ("#", F_COMMENT as u32),
    ("content", F_CONTENT as u32),
    ("number", F_NUMBER as u32),
    ("type", F_TYPE as u32),
    ("block size", F_BLOCK_SIZE as u32),
    ("blocksize", F_BLOCK_SIZE as u32),
    ("dimension", F_DIMENSION as u32),
    ("space", F_SPACE as u32),
    ("space dimension", F_SPACE_DIMENSION as u32),
    ("spacedimension", F_SPACE_DIMENSION as u32),
    ("sizes", F_SIZES as u32),
    ("spacings", F_SPACINGS as u32),
    ("thicknesses", F_THICKNESSES as u32),
    ("axis mins", F_AXIS_MINS as u32),
    ("axismins", F_AXIS_MINS as u32),
    ("axis maxs", F_AXIS_MAXS as u32),
    ("axismaxs", F_AXIS_MAXS as u32),
    ("space directions", F_SPACE_DIRECTIONS as u32),
    ("spacedirections", F_SPACE_DIRECTIONS as u32),
    ("centers", F_CENTERS as u32),
    ("centerings", F_CENTERS as u32),
    ("kinds", F_KINDS as u32),
    ("labels", F_LABELS as u32),
    ("units", F_UNITS as u32),
    ("min", F_MIN as u32),
    ("max", F_MAX as u32),
    ("old min", F_OLD_MIN as u32),
    ("oldmin", F_OLD_MIN as u32),
    ("old max", F_OLD_MAX as u32),
    ("oldmax", F_OLD_MAX as u32),
    ("endian", F_ENDIAN as u32),
    ("encoding", F_ENCODING as u32),
    ("line skip", F_LINE_SKIP as u32),
    ("lineskip", F_LINE_SKIP as u32),
    ("byte skip", F_BYTE_SKIP as u32),
    ("byteskip", F_BYTE_SKIP as u32),
    ("key/value", F_KEYVALUE as u32),
    ("sample units", F_SAMPLE_UNITS as u32),
    ("sampleunits", F_SAMPLE_UNITS as u32),
    ("space units", F_SPACE_UNITS as u32),
    ("spaceunits", F_SPACE_UNITS as u32),
    ("space origin", F_SPACE_ORIGIN as u32),
    ("spaceorigin", F_SPACE_ORIGIN as u32),
    ("measurement frame", F_MEASUREMENT_FRAME as u32),
    ("measurementframe", F_MEASUREMENT_FRAME as u32),
    ("data file", F_DATA_FILE as u32),
    ("datafile", F_DATA_FILE as u32),
];

/// The magic strings `formatNRRD_contentStartsLike` accepts (formatNRRD.c:
/// 140-147), compared with `strcmp` against the whole first line.
const MAGICS: [&str; 7] = [
    "NRRD00.01",
    "NRRD0001",
    "NRRD0002",
    "NRRD0003",
    "NRRD0004",
    "NRRD0005",
    "NRRD0006",
];

// ---------------------------------------------------------------------------
// Number formatting and parsing
// ---------------------------------------------------------------------------

/// C's `%.<prec>g` for a finite `f64`, exponent padded to at least two digits.
pub(crate) fn c_format_g(v: f64, prec: usize) -> String {
    let prec = prec.max(1);
    // The decimal exponent C would use for `%e` at precision `prec - 1`, i.e.
    // after rounding. Rust's `{:e}` prints `m.mmme<exp>` with no `+` and no
    // zero padding, which is exactly the pre-rounded pair we need.
    let sci = format!("{:.*e}", prec - 1, v);
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` always emits `e`");
    let exp: i32 = exp.parse().expect("`{:e}` exponent is an integer");

    if exp < -4 || exp >= prec as i32 {
        let mantissa = strip_trailing_zeros(mantissa);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        let places = (prec as i32 - 1 - exp) as usize;
        strip_trailing_zeros(&format!("{v:.places$}")).to_string()
    }
}

/// Drop the fractional trailing zeros (and a bare trailing `.`) that C's `%g`
/// suppresses in the absence of the `#` flag.
fn strip_trailing_zeros(s: &str) -> &str {
    if !s.contains('.') {
        return s;
    }
    let s = s.trim_end_matches('0');
    s.strip_suffix('.').unwrap_or(s)
}

/// `airSinglePrintf(..., "%lg", v)` (miscAir.c:210-350): print with `%g`, read
/// it back, and reprint with `%.17g` if that lost the value. The IEEE specials
/// are spelled `NaN`, `inf` and `-inf`.
fn format_g(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let short = c_format_g(v, 6);
    if short.parse::<f64>() == Ok(v) {
        short
    } else {
        c_format_g(v, 17)
    }
}

/// `airSingleSscanf(str, "%lg", &v)` (parseAir.c): a lowercased copy is
/// searched for `nan`, `-inf` and `inf` as *substrings* before any numeric
/// parse is attempted, so `"banana"` parses as NaN. Otherwise C's `sscanf`
/// consumes the longest numeric prefix and ignores the rest.
fn air_single_sscanf_double(s: &str) -> Option<f64> {
    let low = s.to_ascii_lowercase();
    if low.contains("nan") {
        return Some(f64::NAN);
    }
    if low.contains("-inf") {
        return Some(f64::NEG_INFINITY);
    }
    if low.contains("inf") {
        return Some(f64::INFINITY);
    }
    parse_leading_f64(&low)
}

/// The longest prefix of `s` matching C's `strtod` decimal grammar.
fn parse_leading_f64(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let mut digits = i > int_start;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        digits |= i > frac_start;
    }
    if !digits {
        return None;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let exp_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_start {
            i = j;
        }
    }
    s[..i].parse().ok()
}

/// `airSingleSscanf(str, "%zu"|"%u", ...)`: a leading-digit scan.
fn parse_leading_usize(s: &str) -> Option<usize> {
    let s = s.trim_start_matches([' ', '\t']);
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 {
        None
    } else {
        s[..end].parse().ok()
    }
}

/// `airSingleSscanf(str, "%ld", ...)`.
fn parse_leading_i64(s: &str) -> Option<i64> {
    let s = s.trim_start_matches([' ', '\t']);
    let (sign, rest) = match s.strip_prefix('-') {
        Some(r) => (-1i64, r),
        None => (1i64, s.strip_prefix('+').unwrap_or(s)),
    };
    parse_leading_usize(rest).map(|v| sign * v as i64)
}

/// `AIR_EXISTS`: finite, so neither NaN nor an infinity.
fn air_exists(v: f64) -> bool {
    v.is_finite()
}

// ---------------------------------------------------------------------------
// The Nrrd struct
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Axis {
    size: usize,
    spacing: f64,
    thickness: f64,
    min: f64,
    max: f64,
    space_direction: [f64; SPACE_DIM_MAX],
    center: u32,
    kind: u32,
    label: String,
    units: String,
}

impl Default for Axis {
    fn default() -> Self {
        Axis {
            size: 0,
            spacing: f64::NAN,
            thickness: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
            space_direction: [f64::NAN; SPACE_DIM_MAX],
            center: 0,
            kind: 0,
            label: String::new(),
            units: String::new(),
        }
    }
}

/// teem's `Nrrd`, restricted to the fields NrrdImageIO reads or writes.
#[derive(Clone, Debug)]
struct Nrrd {
    dim: usize,
    ntype: u32,
    block_size: usize,
    axis: Vec<Axis>,
    space: u32,
    space_dim: usize,
    space_origin: [f64; SPACE_DIM_MAX],
    /// `measurementFrame[column][coefficient]`, matching teem's own indexing:
    /// the writer prints `measurementFrame[dd]` as one parenthesised vector.
    measurement_frame: [[f64; SPACE_DIM_MAX]; SPACE_DIM_MAX],
    content: String,
    old_min: f64,
    old_max: f64,
    sample_units: String,
    space_units: Vec<String>,
    kvp: Vec<(String, String)>,
    comments: Vec<String>,
}

impl Default for Nrrd {
    fn default() -> Self {
        Nrrd {
            dim: 0,
            ntype: 0,
            block_size: 0,
            axis: Vec::new(),
            space: 0,
            space_dim: 0,
            space_origin: [f64::NAN; SPACE_DIM_MAX],
            measurement_frame: [[f64::NAN; SPACE_DIM_MAX]; SPACE_DIM_MAX],
            content: String::new(),
            old_min: f64::NAN,
            old_max: f64::NAN,
            sample_units: String::new(),
            space_units: Vec::new(),
            kvp: Vec::new(),
            comments: Vec::new(),
        }
    }
}

impl Nrrd {
    fn element_number(&self) -> usize {
        self.axis.iter().map(|a| a.size).product()
    }

    /// `nrrdDomainAxesGet` (axis.c:1077-1096).
    fn domain_axes(&self) -> Vec<usize> {
        (0..self.dim)
            .filter(|&a| kind_is_domain(self.axis[a].kind))
            .collect()
    }

    /// `nrrdRangeAxesGet` (axis.c:1102-1121).
    fn range_axes(&self) -> Vec<usize> {
        (0..self.dim)
            .filter(|&a| !kind_is_domain(self.axis[a].kind))
            .collect()
    }
}

/// teem's `NrrdIoState`, restricted to what this port uses.
#[derive(Clone, Debug, Default)]
struct IoState {
    encoding: u32,
    endian: u32,
    line_skip: usize,
    byte_skip: i64,
    data_fn: Vec<String>,
    data_file_dim: usize,
    /// `true` once a `data file:` field has been seen — teem's
    /// `nio->dataFNArr->len > 0` test, but robust to an empty filename list.
    saw_data_file: bool,
}

// ---------------------------------------------------------------------------
// Header reading
// ---------------------------------------------------------------------------

/// `airOneLine` (string.c): `\n`, `\r\n` and `\r` all terminate. A final line
/// with no terminator returns length `0` — i.e. it is indistinguishable from
/// EOF and its content is silently discarded.
struct Lines<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Lines<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Lines { buf, pos: 0 }
    }

    fn next_line(&mut self) -> Option<&'a [u8]> {
        if self.pos >= self.buf.len() {
            return None;
        }
        let rest = &self.buf[self.pos..];
        let idx = rest.iter().position(|&b| b == b'\n' || b == b'\r')?;
        let line = &rest[..idx];
        let mut next = self.pos + idx + 1;
        if rest[idx] == b'\r' && next < self.buf.len() && self.buf[next] == b'\n' {
            next += 1;
        }
        self.pos = next;
        Some(line)
    }
}

fn line_str(line: &[u8]) -> Result<&str> {
    std::str::from_utf8(line).map_err(|_| bad("header line is not valid UTF-8"))
}

/// `airUnescape` (string.c:216-241): `\n` and `\\` only.
fn air_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `getQuotedString` (parseNrrd.c:655-710): skip separators, require `"`, read
/// to the next unescaped `"`, and unescape `\"` alone.
fn get_quoted_string(h: &mut &str) -> Result<String> {
    let rest = h.trim_start_matches(FIELD_SEP);
    let rest = rest
        .strip_prefix('"')
        .ok_or_else(|| bad("quoted string didn't start with `\"`"))?;
    let mut out = String::new();
    let mut it = rest.char_indices();
    while let Some((i, c)) = it.next() {
        if c == '"' {
            *h = &rest[i + 1..];
            return Ok(out);
        }
        if c == '\\' && rest[i + 1..].starts_with('"') {
            out.push('"');
            it.next();
            continue;
        }
        out.push(c);
    }
    Err(bad("didn't see ending `\"` soon enough"))
}

/// `nrrd__WriteEscaped(.., "\"", whitespace)` (keyvalue.c:228-274) for a label
/// or unit: escape `"`, and fold every other whitespace character to a space.
fn write_escaped(s: &str, escape: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if escape.contains(c) {
            match c {
                '\n' => out.push_str("\\n"),
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                _ => out.push(c),
            }
        } else if c != '\t' && c.is_whitespace() {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// `airOneLinify` (string.c): collapse each whitespace run to one space and
/// trim the ends. Applied to `content` and `sample units` on write.
fn one_linify(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `spaceVectorParse` (parseNrrd.c:398-500). Consumes one `(a,b,c)` or the
/// literal `none` from the front of `h`.
fn space_vector_parse(h: &mut &str, space_dim: usize) -> Result<[f64; SPACE_DIM_MAX]> {
    let mut val = [f64::NAN; SPACE_DIM_MAX];
    let rest = h.trim_start_matches(FIELD_SEP);
    if rest.is_empty() {
        return Err(bad("hit end of string before seeing `(`"));
    }
    if let Some(after) = rest.strip_prefix("none") {
        if after.is_empty() || after.starts_with(FIELD_SEP) {
            *h = after;
            return Ok(val);
        }
        return Err(bad(format!("couldn't parse non-vector \"{rest}\"")));
    }
    let inner = rest
        .strip_prefix('(')
        .ok_or_else(|| bad(format!("vector in \"{rest}\" didn't start with `(`")))?;
    let close = inner
        .find(')')
        .ok_or_else(|| bad("didn't see `)` at end of vector"))?;
    let coefficients: Vec<&str> = inner[..close].split(',').collect();
    if coefficients.len() > space_dim {
        return Err(bad(format!(
            "space dimension is {space_dim}, but seem to have {} coefficients",
            coefficients.len()
        )));
    }
    if coefficients.len() != space_dim {
        return Err(bad(format!(
            "parsed {} values, but space dimension is {space_dim}",
            coefficients.len()
        )));
    }
    for (i, token) in coefficients.iter().enumerate() {
        val[i] = air_single_sscanf_double(token)
            .ok_or_else(|| bad(format!("couldn't parse \"{token}\" as a double")))?;
    }
    // "make sure all coefficients exist or not together", and reject an
    // infinity (parseNrrd.c:491-508).
    for i in 0..space_dim {
        if val[0].is_nan() != val[i].is_nan() {
            return Err(bad("space vector coefficients don't all exist"));
        }
        if val[i].is_infinite() {
            return Err(bad("space vector coefficient is infinite"));
        }
    }
    *h = &inner[close + 1..];
    Ok(val)
}

/// Split a header line the way `nrrd__ReadNrrdParseField` does, returning
/// either a recognised field with its (already right-trimmed, where upstream
/// trims) value, or a key/value pair.
enum Field {
    Known(usize, String),
    KeyValue(String, String),
}

fn parse_field_line(line: &str) -> Result<Field> {
    if let Some(comment) = line.strip_prefix('#') {
        return Ok(Field::Known(F_COMMENT, comment.to_string()));
    }
    let known = line.find(": ").and_then(|colon| {
        let id = air_enum_val(FIELD_EQV, &line[..colon]);
        (id != 0).then_some((id as usize, colon + 2))
    });
    if let Some((id, start)) = known {
        let mut value = line[start..].trim_start_matches(FIELD_SEP);
        // formatNRRD.c:491-510 right-trims every field but these four. The
        // upstream loop guard is `last > info`, so an all-whitespace value
        // keeps exactly one character.
        if !matches!(id, F_COMMENT | F_CONTENT | F_KEYVALUE | F_DATA_FILE) {
            let trimmed = value.trim_end_matches(FIELD_SEP);
            value = if trimmed.is_empty() && !value.is_empty() {
                &value[..1]
            } else {
                trimmed
            };
        }
        return Ok(Field::Known(id, value.to_string()));
    }
    let assign = line.find(":=").ok_or_else(|| {
        bad(format!(
            "trouble parsing NRRD field identifier from \"{line}\""
        ))
    })?;
    Ok(Field::KeyValue(
        air_unescape(&line[..assign]),
        air_unescape(&line[assign + 2..]),
    ))
}

/// Parse `count` doubles from a field value, as `airParseStrD` does.
fn parse_doubles(info: &str, count: usize, what: &str) -> Result<Vec<f64>> {
    let tokens: Vec<&str> = info.split(FIELD_SEP).filter(|t| !t.is_empty()).collect();
    if tokens.len() != count {
        return Err(bad(format!(
            "expected {count} {what}, got {}",
            tokens.len()
        )));
    }
    tokens
        .iter()
        .map(|t| {
            air_single_sscanf_double(t)
                .ok_or_else(|| bad(format!("couldn't parse \"{t}\" as a double")))
        })
        .collect()
}

/// `nFieldCheckSpaceInfo` (simple.c), for the subset of state this port keeps:
/// a domain axis may not carry both a space direction and a min, max, spacing,
/// or units.
fn check_space_info(nrrd: &Nrrd) -> Result<()> {
    if nrrd.space != 0 && space_dimension(nrrd.space) != nrrd.space_dim {
        return Err(bad("space and space dimension disagree"));
    }
    for axis in &nrrd.axis {
        if !air_exists(axis.space_direction[0]) {
            continue;
        }
        if air_exists(axis.min)
            || air_exists(axis.max)
            || air_exists(axis.spacing)
            || !axis.units.is_empty()
        {
            return Err(bad(
                "axis with a space direction may not have min, max, spacing, or units",
            ));
        }
    }
    if nrrd.space_dim == 0 {
        if nrrd.space != 0 {
            return Err(bad("space set but space dimension is zero"));
        }
        if !nrrd.space_units.is_empty() || air_exists(nrrd.space_origin[0]) {
            return Err(bad("space units or origin without a space dimension"));
        }
    }
    Ok(())
}

fn need_dim(nrrd: &Nrrd) -> Result<usize> {
    if nrrd.dim == 0 {
        return Err(bad("need `dimension` to have been set first"));
    }
    Ok(nrrd.dim)
}

fn need_space_dim(nrrd: &Nrrd) -> Result<usize> {
    if nrrd.space_dim == 0 {
        return Err(bad(
            "need `space` or `space dimension` to have been set first",
        ));
    }
    Ok(nrrd.space_dim)
}

/// `nrrd__HeaderCheck` (simple.c): the four required fields plus the endian
/// rule.
fn header_check(nrrd: &Nrrd, io: &IoState, seen: &[bool]) -> Result<()> {
    for field in [F_TYPE, F_DIMENSION, F_SIZES, F_ENCODING] {
        if !seen[field] {
            return Err(bad(format!(
                "didn't see required field `{}`",
                FIELD_STR[field]
            )));
        }
    }
    if nrrd.ntype == TYPE_BLOCK && nrrd.block_size == 0 {
        return Err(bad("type is `block` but block size is unset"));
    }
    let element_size = if nrrd.ntype == TYPE_BLOCK {
        nrrd.block_size
    } else {
        type_size(nrrd.ntype)
    };
    if element_size == 0 {
        return Err(bad("element size is zero"));
    }
    if io.endian == 0 && encoding_endian_matters(io.encoding) && element_size > 1 {
        return Err(bad(format!(
            "type `{}` with encoding requiring endianness, but no `endian` field",
            TYPE_STR[nrrd.ntype as usize]
        )));
    }
    Ok(())
}

/// `nrrd__FieldCheck_kinds` (simple.c), reached from `nrrdCheckMore` — which
/// `nrrdSpacingCalculate` calls before it will report anything, so every image
/// NrrdImageIO reads has passed it.
fn check_kinds(nrrd: &Nrrd) -> Result<()> {
    for axis in &nrrd.axis {
        let want = kind_size(axis.kind);
        if want != 0 && want != axis.size {
            return Err(bad(format!(
                "axis of kind `{}` has size {}, not {want}",
                KIND_STR[axis.kind as usize], axis.size
            )));
        }
    }
    Ok(())
}

/// The parsed header plus the offset of the attached data, if any.
struct ParsedHeader {
    nrrd: Nrrd,
    io: IoState,
    /// Byte offset just past the header's blank-line terminator. Meaningful
    /// only when `io.data_fn` is empty.
    data_offset: usize,
}

/// `formatNRRD_read` (formatNRRD.c:430-545) plus the per-field parsers in
/// parseNrrd.c.
fn read_header(bytes: &[u8]) -> Result<ParsedHeader> {
    let mut lines = Lines::new(bytes);
    let magic = lines
        .next_line()
        .ok_or_else(|| bad("file is empty"))
        .and_then(line_str)?;
    if !MAGICS.contains(&magic) {
        return Err(bad("this doesn't look like a NRRD file"));
    }

    let mut nrrd = Nrrd::default();
    let mut io = IoState::default();
    let mut seen = [false; F_MAX_FIELD + 1];

    loop {
        let Some(line) = lines.next_line() else {
            // llen == 0: EOF. Legal only when a data file was named.
            if !io.saw_data_file {
                return Err(bad("hit end of header, but no `data file` given"));
            }
            break;
        };
        if line.is_empty() {
            // llen == 1: the blank line separating header from attached data.
            break;
        }
        let field = parse_field_line(line_str(line)?)?;
        let Field::Known(id, info) = field else {
            let Field::KeyValue(key, value) = field else {
                unreachable!("Field has two variants");
            };
            nrrd.kvp.push((key, value));
            seen[F_KEYVALUE] = true;
            continue;
        };
        if seen[id] && id != F_COMMENT {
            return Err(bad(format!("already set field `{}`", FIELD_STR[id])));
        }
        parse_known_field(id, &info, &mut nrrd, &mut io, &mut lines, &seen)?;
        seen[id] = true;
    }

    header_check(&nrrd, &io, &seen)?;
    check_kinds(&nrrd)?;
    Ok(ParsedHeader {
        nrrd,
        io,
        data_offset: lines.pos,
    })
}

fn parse_known_field(
    id: usize,
    info: &str,
    nrrd: &mut Nrrd,
    io: &mut IoState,
    lines: &mut Lines<'_>,
    seen: &[bool],
) -> Result<()> {
    match id {
        F_COMMENT => nrrd.comments.push(one_linify(info)),
        // "number" is entirely ignored (parseNrrd.c:152-181), and "min"/"max"
        // no longer mean anything (parseNrrd.c:784-810).
        F_NUMBER | F_MIN | F_MAX => {}
        F_CONTENT => nrrd.content = info.to_string(),
        F_TYPE => {
            nrrd.ntype = air_enum_val(TYPE_EQV, info);
            if nrrd.ntype == 0 {
                return Err(bad(format!("couldn't parse type \"{info}\"")));
            }
        }
        F_BLOCK_SIZE => {
            nrrd.block_size =
                parse_leading_usize(info).ok_or_else(|| bad("couldn't parse block size"))?;
        }
        F_DIMENSION => {
            let dim = parse_leading_usize(info).ok_or_else(|| bad("couldn't parse dimension"))?;
            if !(1..=DIM_MAX).contains(&dim) {
                return Err(bad(format!(
                    "dimension {dim} outside valid range [1,{DIM_MAX}]"
                )));
            }
            nrrd.dim = dim;
            nrrd.axis = vec![Axis::default(); dim];
        }
        F_SPACE => {
            if nrrd.space_dim != 0 {
                return Err(bad("can't set space after space dimension"));
            }
            let space = air_enum_val(SPACE_EQV, info);
            if space == 0 {
                return Err(bad(format!("couldn't parse space \"{info}\"")));
            }
            nrrd.space = space;
            nrrd.space_dim = space_dimension(space);
            check_space_info(nrrd)?;
        }
        F_SPACE_DIMENSION => {
            if nrrd.space != 0 {
                return Err(bad("can't set space dimension after space"));
            }
            let sd =
                parse_leading_usize(info).ok_or_else(|| bad("couldn't parse space dimension"))?;
            if !(1..=SPACE_DIM_MAX).contains(&sd) {
                return Err(bad(format!(
                    "space dimension {sd} outside valid range [1,{SPACE_DIM_MAX}]"
                )));
            }
            nrrd.space_dim = sd;
            check_space_info(nrrd)?;
        }
        F_SIZES => {
            let dim = need_dim(nrrd)?;
            let tokens: Vec<&str> = info.split(FIELD_SEP).filter(|t| !t.is_empty()).collect();
            if tokens.len() != dim {
                return Err(bad(format!("expected {dim} sizes, got {}", tokens.len())));
            }
            for (i, token) in tokens.iter().enumerate() {
                let size = parse_leading_usize(token)
                    .ok_or_else(|| bad(format!("couldn't parse size \"{token}\"")))?;
                if size == 0 {
                    return Err(bad("axis size must be positive"));
                }
                nrrd.axis[i].size = size;
            }
        }
        F_SPACINGS => {
            let dim = need_dim(nrrd)?;
            for (i, v) in parse_doubles(info, dim, "spacings")?
                .into_iter()
                .enumerate()
            {
                nrrd.axis[i].spacing = v;
            }
            check_space_info(nrrd)?;
        }
        F_THICKNESSES => {
            let dim = need_dim(nrrd)?;
            for (i, v) in parse_doubles(info, dim, "thicknesses")?
                .into_iter()
                .enumerate()
            {
                nrrd.axis[i].thickness = v;
            }
        }
        F_AXIS_MINS => {
            let dim = need_dim(nrrd)?;
            for (i, v) in parse_doubles(info, dim, "axis mins")?
                .into_iter()
                .enumerate()
            {
                nrrd.axis[i].min = v;
            }
            check_space_info(nrrd)?;
        }
        F_AXIS_MAXS => {
            let dim = need_dim(nrrd)?;
            for (i, v) in parse_doubles(info, dim, "axis maxs")?
                .into_iter()
                .enumerate()
            {
                nrrd.axis[i].max = v;
            }
            check_space_info(nrrd)?;
        }
        F_SPACE_DIRECTIONS => {
            let dim = need_dim(nrrd)?;
            let space_dim = need_space_dim(nrrd)?;
            let mut h = info;
            for i in 0..dim {
                nrrd.axis[i].space_direction = space_vector_parse(&mut h, space_dim)?;
            }
            if !h.trim_matches(FIELD_SEP).is_empty() {
                return Err(bad("more than the expected space directions"));
            }
            check_space_info(nrrd)?;
        }
        F_CENTERS => {
            let dim = need_dim(nrrd)?;
            let tokens: Vec<&str> = info.split(FIELD_SEP).filter(|t| !t.is_empty()).collect();
            if tokens.len() != dim {
                return Err(bad(format!("expected {dim} centers, got {}", tokens.len())));
            }
            for (i, token) in tokens.iter().enumerate() {
                if *token == "???" || *token == "none" {
                    nrrd.axis[i].center = 0;
                    continue;
                }
                nrrd.axis[i].center = air_enum_val(CENTER_EQV, token);
                if nrrd.axis[i].center == 0 {
                    return Err(bad(format!("couldn't parse center \"{token}\"")));
                }
            }
        }
        F_KINDS => {
            let dim = need_dim(nrrd)?;
            let tokens: Vec<&str> = info.split(FIELD_SEP).filter(|t| !t.is_empty()).collect();
            if tokens.len() != dim {
                return Err(bad(format!("expected {dim} kinds, got {}", tokens.len())));
            }
            for (i, token) in tokens.iter().enumerate() {
                if *token == "???" {
                    nrrd.axis[i].kind = 0;
                    continue;
                }
                if *token == "none" {
                    // Fixed (ledger §1.48): upstream's `none` arm assigns to
                    // `.center` instead of `.kind` (parseNrrd.c:621-623); this
                    // port clears `.kind`, matching the `???` arm right above
                    // it and leaving any `centers:` value untouched.
                    nrrd.axis[i].kind = 0;
                    continue;
                }
                nrrd.axis[i].kind = air_enum_val(KIND_EQV, token);
                if nrrd.axis[i].kind == 0 {
                    return Err(bad(format!("couldn't parse kind \"{token}\"")));
                }
            }
        }
        F_LABELS | F_UNITS => {
            let dim = need_dim(nrrd)?;
            let mut h = info;
            for i in 0..dim {
                let s = get_quoted_string(&mut h)?;
                if id == F_LABELS {
                    nrrd.axis[i].label = s;
                } else {
                    nrrd.axis[i].units = s;
                }
            }
            if !h.trim_matches(FIELD_SEP).is_empty() {
                return Err(bad("more than the expected labels or units"));
            }
            if id == F_UNITS {
                check_space_info(nrrd)?;
            }
        }
        F_OLD_MIN => {
            nrrd.old_min =
                air_single_sscanf_double(info).ok_or_else(|| bad("couldn't parse old min"))?;
        }
        F_OLD_MAX => {
            nrrd.old_max =
                air_single_sscanf_double(info).ok_or_else(|| bad("couldn't parse old max"))?;
        }
        F_ENDIAN => {
            io.endian = air_enum_val(ENDIAN_EQV, info);
            if io.endian == 0 {
                return Err(bad(format!("couldn't parse endian \"{info}\"")));
            }
        }
        F_ENCODING => {
            io.encoding = air_enum_val(ENC_EQV, info);
            if io.encoding == 0 {
                return Err(bad(format!("couldn't parse encoding \"{info}\"")));
            }
        }
        F_LINE_SKIP => {
            io.line_skip =
                parse_leading_usize(info).ok_or_else(|| bad("couldn't parse line skip"))?;
        }
        F_BYTE_SKIP => {
            io.byte_skip =
                parse_leading_i64(info).ok_or_else(|| bad("couldn't parse byte skip"))?;
        }
        F_SAMPLE_UNITS => nrrd.sample_units = info.to_string(),
        F_SPACE_UNITS => {
            let space_dim = need_space_dim(nrrd)?;
            let mut h = info;
            nrrd.space_units = (0..space_dim)
                .map(|_| get_quoted_string(&mut h))
                .collect::<Result<_>>()?;
            check_space_info(nrrd)?;
        }
        F_SPACE_ORIGIN => {
            let space_dim = need_space_dim(nrrd)?;
            let mut h = info;
            nrrd.space_origin = space_vector_parse(&mut h, space_dim)?;
            check_space_info(nrrd)?;
        }
        F_MEASUREMENT_FRAME => {
            let space_dim = need_space_dim(nrrd)?;
            let mut h = info;
            for column in 0..space_dim {
                nrrd.measurement_frame[column] = space_vector_parse(&mut h, space_dim)?;
            }
            check_space_info(nrrd)?;
        }
        F_DATA_FILE => parse_data_file(info, nrrd, io, lines, seen)?,
        other => return Err(bad(format!("unhandled field {other}"))),
    }
    Ok(())
}

/// `rnParse_data_file` (parseNrrd.c:1231-1414).
///
/// The `%d` printf-template form and `SKIPLIST` are rejected: neither is ever
/// written by `NrrdImageIO::Write`, and both need multi-file bookkeeping this
/// port has no reader for.
fn parse_data_file(
    info: &str,
    nrrd: &Nrrd,
    io: &mut IoState,
    lines: &mut Lines<'_>,
    seen: &[bool],
) -> Result<()> {
    io.saw_data_file = true;
    if contains_percent_d(info) {
        return Err(unsupported(
            "`data file:` printf-format form (`<format>.%d <min> <max> <step>`)",
        ));
    }
    if info.starts_with("SKIPLIST") {
        return Err(unsupported("`data file: SKIPLIST`"));
    }
    if let Some(after) = info.strip_prefix("LIST") {
        let dim = need_dim(nrrd)?;
        // `nrrd__HeaderCheck(nrrd, nio, AIR_TRUE)`: the LIST form demands a
        // complete header up to this point.
        header_check(nrrd, io, seen)?;
        io.data_file_dim = if after.is_empty() {
            dim - 1
        } else {
            let d = parse_leading_usize(after.trim_start_matches(FIELD_SEP))
                .ok_or_else(|| bad("couldn't parse info after `LIST` as an int"))?;
            if !(1..=dim).contains(&d) {
                return Err(bad(format!(
                    "datafile dimension {d} outside valid range [1,{dim}]"
                )));
            }
            d
        };
        // Upstream reads until EOF, taking a blank line as an (empty)
        // filename rather than a terminator (ledger §2.83).
        while let Some(line) = lines.next_line() {
            io.data_fn.push(line_str(line)?.to_string());
        }
        data_fn_check(nrrd, io)?;
    } else {
        io.data_fn.push(info.to_string());
        io.data_file_dim = 0;
    }
    Ok(())
}

/// `nrrdContainsPercentThisAndMore(info, 'd')` (parseNrrd.c): a `%` followed,
/// after optional flags/width, by `d`, with at least one more character in the
/// string.
fn contains_percent_d(info: &str) -> bool {
    let Some(at) = info.find('%') else {
        return false;
    };
    let rest = &info[at + 1..];
    let digits = rest.trim_start_matches(|c: char| c == '0' || c.is_ascii_digit());
    digits.starts_with('d') && digits.len() > 1
}

/// `nrrdIoDataFNCheck` (simple.c) for the `LIST` form.
fn data_fn_check(nrrd: &Nrrd, io: &IoState) -> Result<()> {
    let files = io.data_fn.len();
    if files == 0 {
        return Err(bad("`data file: LIST` named no files"));
    }
    if io.data_file_dim < nrrd.dim {
        let pieces: usize = nrrd.axis[io.data_file_dim..nrrd.dim]
            .iter()
            .map(|a| a.size)
            .product();
        if pieces != files {
            return Err(bad(format!(
                "expected {pieces} data files for data file dimension {}, got {files}",
                io.data_file_dim
            )));
        }
    } else {
        let last = nrrd.axis[nrrd.dim - 1].size;
        if files > last || !last.is_multiple_of(files) {
            return Err(bad(format!(
                "{files} data files don't evenly divide the slowest axis size {last}"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Data reading
// ---------------------------------------------------------------------------

/// `NEED_PATH` (formatNRRD.c:166) plus `nrrd__DataFNCheck`'s path prefixing:
/// header-relative unless the name is `-`, starts with `/`, or has `:` at
/// index 1.
fn resolve_data_file(header_path: &Path, name: &str) -> PathBuf {
    let bytes = name.as_bytes();
    let absolute = name == "-" || bytes.first() == Some(&b'/') || bytes.get(1) == Some(&b':');
    if absolute {
        PathBuf::from(name)
    } else {
        header_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(name)
    }
}

/// `nrrdLineSkip` (read.c:280-301): drop `line skip` lines from the *file*
/// stream. It runs for every encoding, compressed ones included
/// (formatNRRD.c:579-582).
fn skip_lines(data: &[u8], line_skip: usize) -> Result<&[u8]> {
    let mut pos = 0usize;
    for _ in 0..line_skip {
        let Some(nl) = data[pos..].iter().position(|&b| b == b'\n' || b == b'\r') else {
            return Err(IoError::TruncatedData);
        };
        pos += nl + 1;
        if data.get(pos - 1) == Some(&b'\r') && data.get(pos) == Some(&b'\n') {
            pos += 1;
        }
    }
    Ok(&data[pos..])
}

/// `nrrd__ByteSkipSkip` (read.c:303-361): the `byte skip` that `formatNRRD_read`
/// applies to the *file* stream, and only for a non-compression encoding — for
/// a compressed one the skip belongs to the decompressed stream instead
/// (formatNRRD.c:583-585), so [`byte_skip_decompressed`] owns it there.
///
/// `want` is the byte count the decode will consume; it is used only by the
/// negative `byte skip`, which seeks from the end of the file and which
/// upstream rejects for any encoding but raw (read.c:320-327).
fn byte_skip_file<'a>(data: &'a [u8], io: &IoState, want: usize) -> Result<&'a [u8]> {
    debug_assert!(!encoding_is_compression(io.encoding));
    let start = if io.byte_skip >= 0 {
        io.byte_skip as usize
    } else {
        if io.encoding != ENC_RAW {
            return Err(bad(
                "backwards byte skip is only possible in raw encoding, not the encoding named here",
            ));
        }
        let back = want + (-io.byte_skip - 1) as usize;
        data.len().checked_sub(back).ok_or(IoError::TruncatedData)?
    };
    data.get(start..).ok_or(IoError::TruncatedData)
}

/// `nrrdLineSkip` then `nrrd__ByteSkipSkip`, for a non-compression encoding.
fn skip_to_data<'a>(data: &'a [u8], io: &IoState, want: usize) -> Result<&'a [u8]> {
    byte_skip_file(skip_lines(data, io.line_skip)?, io, want)
}

/// [`skip_to_data`] followed by the raw decode of exactly `want` bytes.
fn raw_slice<'a>(data: &'a [u8], io: &IoState, want: usize) -> Result<&'a [u8]> {
    let rest = skip_to_data(data, io, want)?;
    if rest.len() < want {
        // `encodingRaw_read` errors on a short read; extra bytes are only a
        // warning.
        return Err(IoError::TruncatedData);
    }
    Ok(&rest[..want])
}

/// The `byte skip` `encodingGzip_read` performs on its *decompressed* buffer
/// (encodingGzip.c:81-181).
///
/// A non-negative skip drops that many bytes off the front and then requires
/// `want` more; a short stream is the `sizeRed != sizeData` error (`:174-180`).
///
/// A negative skip is legal here — unlike for raw — because the gzip decoder
/// implements it itself: `backwards = -byteSkip - 1` bytes are ignored *after*
/// the data, and the data is the `want` bytes ending `backwards` from the end
/// (`:132-140`). So `byte skip: -1` means "the data is the tail of the stream".
fn byte_skip_decompressed(data: &[u8], byte_skip: i64, want: usize) -> Result<&[u8]> {
    let start = if byte_skip >= 0 {
        byte_skip as usize
    } else {
        let backwards = (-byte_skip - 1) as usize;
        data.len()
            .checked_sub(want + backwards)
            .ok_or(IoError::TruncatedData)?
    };
    let end = start.checked_add(want).ok_or(IoError::TruncatedData)?;
    data.get(start..end).ok_or(IoError::TruncatedData)
}

/// `nrrdIStore`-style parse of `values` whitespace-separated ASCII numbers
/// (encodingAscii.c). Integer types narrower than `int` are read as `int` and
/// C-cast down.
fn ascii_decode(text: &str, ntype: u32, values: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(values * type_size(ntype));
    let mut tokens = text.split_whitespace();
    for _ in 0..values {
        let token = tokens.next().ok_or(IoError::TruncatedData)?;
        macro_rules! narrow {
            ($ty:ty) => {{
                let v: i64 = token
                    .parse::<i32>()
                    .map_err(|_| bad(format!("couldn't parse ascii value \"{token}\"")))?
                    .into();
                out.extend_from_slice(&(v as $ty).to_le_bytes());
            }};
        }
        macro_rules! direct {
            ($ty:ty) => {{
                let v: $ty = token
                    .parse()
                    .map_err(|_| bad(format!("couldn't parse ascii value \"{token}\"")))?;
                out.extend_from_slice(&v.to_le_bytes());
            }};
        }
        match ntype {
            TYPE_CHAR => narrow!(i8),
            TYPE_UCHAR => narrow!(u8),
            TYPE_SHORT => narrow!(i16),
            TYPE_USHORT => narrow!(u16),
            TYPE_INT => direct!(i32),
            TYPE_UINT => direct!(u32),
            TYPE_LLONG => direct!(i64),
            TYPE_ULLONG => direct!(u64),
            TYPE_FLOAT => direct!(f32),
            TYPE_DOUBLE => direct!(f64),
            _ => return Err(unsupported("ascii encoding of this type")),
        }
    }
    Ok(out)
}

/// Gather the element bytes for `nrrd`, in NRRD axis order, little-endian.
fn read_data(path: &Path, header_bytes: &[u8], parsed: &ParsedHeader) -> Result<Vec<u8>> {
    let ParsedHeader { nrrd, io, .. } = parsed;
    match io.encoding {
        ENC_RAW | ENC_ASCII | ENC_GZIP => {}
        ENC_BZIP2 => {
            return Err(unsupported(
                "`encoding: bzip2` — this build has no bzip2 dependency (ledger §5.8)",
            ));
        }
        ENC_HEX => return Err(unsupported("`encoding: hex`")),
        ENC_ZRL => return Err(unsupported("`encoding: zrl`")),
        _ => return Err(bad("unknown encoding")),
    }

    let element_size = type_size(nrrd.ntype);
    let elements = nrrd.element_number();
    let files = io.data_fn.len().max(1);
    if elements % files != 0 {
        return Err(bad("data files don't evenly divide the element count"));
    }
    let per_file_values = elements / files;
    let per_file_bytes = per_file_values * element_size;

    let mut data = Vec::with_capacity(elements * element_size);
    for index in 0..files {
        let owned;
        let bytes: &[u8] = if io.data_fn.is_empty() {
            &header_bytes[parsed.data_offset..]
        } else {
            let name = &io.data_fn[index];
            if name.is_empty() {
                return Err(bad("`data file: LIST` contained an empty filename"));
            }
            owned = std::fs::read(resolve_data_file(path, name))?;
            &owned
        };
        match io.encoding {
            ENC_RAW => data.extend_from_slice(raw_slice(bytes, io, per_file_bytes)?),
            ENC_GZIP => {
                // `nrrdLineSkip` runs on the file stream; the gzip decoder then
                // owns the byte skip inside what it inflates.
                let stream = skip_lines(bytes, io.line_skip)?;
                let inflated = gunzip_transparent(stream)?;
                data.extend_from_slice(byte_skip_decompressed(
                    &inflated,
                    io.byte_skip,
                    per_file_bytes,
                )?);
            }
            _ => {
                let rest = skip_to_data(bytes, io, per_file_bytes)?;
                let text =
                    std::str::from_utf8(rest).map_err(|_| bad("ascii data is not valid UTF-8"))?;
                data.extend_from_slice(&ascii_decode(text, nrrd.ntype, per_file_values)?);
            }
        }
    }

    // `nrrdSwapEndian` after all data files (read.c). Only encodings whose
    // `endianMatters` is set care, and only for multi-byte types.
    if encoding_endian_matters(io.encoding) && io.endian == ENDIAN_BIG && element_size > 1 {
        for chunk in data.chunks_exact_mut(element_size) {
            chunk.reverse();
        }
    }
    Ok(data)
}

fn buffer_from_le_bytes(id: PixelId, bytes: &[u8]) -> PixelBuffer {
    macro_rules! unpack {
        ($ty:ty, $variant:ident) => {{
            const S: usize = std::mem::size_of::<$ty>();
            PixelBuffer::$variant(
                bytes
                    .chunks_exact(S)
                    .map(|c| <$ty>::from_le_bytes(c.try_into().expect("chunks_exact")))
                    .collect(),
            )
        }};
    }
    match id.component_id() {
        PixelId::UInt8 => PixelBuffer::UInt8(bytes.to_vec()),
        PixelId::Int8 => PixelBuffer::Int8(bytes.iter().map(|&b| b as i8).collect()),
        PixelId::UInt16 => unpack!(u16, UInt16),
        PixelId::Int16 => unpack!(i16, Int16),
        PixelId::UInt32 => unpack!(u32, UInt32),
        PixelId::Int32 => unpack!(i32, Int32),
        PixelId::UInt64 => unpack!(u64, UInt64),
        PixelId::Int64 => unpack!(i64, Int64),
        PixelId::Float32 => unpack!(f32, Float32),
        PixelId::Float64 => unpack!(f64, Float64),
        other => unreachable!("component_id returned {other:?}"),
    }
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

/// `nrrdAxesPermute` (axis.c): output axis `i` takes input axis `axmap[i]`,
/// axis `0` fastest.
fn permute(data: &[u8], sizes: &[usize], axmap: &[usize], element_size: usize) -> Vec<u8> {
    let dim = sizes.len();
    let mut in_stride = vec![1usize; dim];
    for a in 1..dim {
        in_stride[a] = in_stride[a - 1] * sizes[a - 1];
    }
    let out_sizes: Vec<usize> = axmap.iter().map(|&a| sizes[a]).collect();
    let total: usize = sizes.iter().product();

    let mut out = Vec::with_capacity(total * element_size);
    let mut coordinate = vec![0usize; dim];
    for _ in 0..total {
        let mut offset = 0usize;
        for (k, &c) in coordinate.iter().enumerate() {
            offset += c * in_stride[axmap[k]];
        }
        let start = offset * element_size;
        out.extend_from_slice(&data[start..start + element_size]);
        for k in 0..dim {
            coordinate[k] += 1;
            if coordinate[k] < out_sizes[k] {
                break;
            }
            coordinate[k] = 0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The ITK layer
// ---------------------------------------------------------------------------

/// `IOPixelEnum`, restricted to the values NrrdImageIO produces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PixelKind {
    Scalar,
    Vector,
    Complex,
    SymmetricSecondRankTensor,
}

/// `GetAxisOrderForFileReading` (itkNrrdImageIO.cxx:41-126) with the default
/// `AxesReorder::UseAnyRangeAxisAsPixel`.
struct AxisOrder {
    pixel_axis: Option<usize>,
    image_axes: Vec<usize>,
    domain_axes: usize,
    permute: bool,
}

fn axis_order(nrrd: &Nrrd) -> AxisOrder {
    let domain = nrrd.domain_axes();
    let range = nrrd.range_axes();

    let mut pixel_axis = range
        .iter()
        .find(|&&a| nrrd.axis[a].kind != KIND_LIST)
        .copied();
    if pixel_axis.is_none() {
        pixel_axis = range.first().copied();
    }

    let mut image_axes = domain.clone();
    image_axes.extend(range.iter().copied().filter(|a| Some(*a) != pixel_axis));

    let permute = match pixel_axis {
        Some(p) if p > 0 => true,
        pixel => {
            let offset = usize::from(pixel == Some(0));
            image_axes.iter().enumerate().any(|(i, &a)| a != i + offset)
        }
    };
    AxisOrder {
        pixel_axis,
        image_axes,
        domain_axes: domain.len(),
        permute,
    }
}

/// The pixel type and component count `ReadImageInformation` derives from the
/// pixel axis's kind (itkNrrdImageIO.cxx:701-759).
fn pixel_kind_of(kind: u32, size: usize) -> Result<(PixelKind, usize)> {
    Ok(match kind {
        KIND_DOMAIN | KIND_SPACE | KIND_TIME => {
            return Err(bad(format!(
                "range axis kind ({}) seems more like a domain axis than a range axis",
                KIND_STR[kind as usize]
            )));
        }
        KIND_STUB | KIND_SCALAR => (PixelKind::Scalar, size),
        KIND_3COLOR | KIND_RGB_COLOR | KIND_4COLOR | KIND_RGBA_COLOR => (PixelKind::Vector, size),
        KIND_VECTOR | KIND_2VECTOR | KIND_3VECTOR | KIND_4VECTOR | KIND_LIST => {
            (PixelKind::Vector, size)
        }
        KIND_POINT => (PixelKind::Vector, size),
        KIND_COVARIANT_VECTOR | KIND_3GRADIENT | KIND_NORMAL | KIND_3NORMAL => {
            (PixelKind::Vector, size)
        }
        KIND_3D_SYM_MATRIX => (PixelKind::SymmetricSecondRankTensor, size),
        KIND_3D_MASKED_SYM_MATRIX => (PixelKind::SymmetricSecondRankTensor, size - 1),
        KIND_COMPLEX => (PixelKind::Complex, size),
        KIND_HSV_COLOR
        | KIND_XYZ_COLOR
        | KIND_QUATERNION
        | KIND_2D_SYM_MATRIX
        | KIND_2D_MASKED_SYM_MATRIX
        | KIND_2D_MATRIX
        | KIND_2D_MASKED_MATRIX
        | KIND_3D_MATRIX => (PixelKind::Vector, size),
        other => return Err(bad(format!("nrrdKind {other} not known!"))),
    })
}

/// `ImageReaderBase::GetPixelIDFromImageIO` (sitkImageReaderBase.cxx:215-240).
///
/// `RGB`, `RGBA`, `POINT` and `COVARIANTVECTOR` all collapse onto the vector
/// pixel id here, exactly as they do upstream: this port has no distinct RGB
/// pixel type. `SYMMETRICSECONDRANKTENSOR` falls off the end of upstream's
/// SimpleITK if-ladder and raises "Unknown PixelType" even though `itk::Image`
/// reads it fine; this port instead loads the tensor as a **vector image** of
/// its unique matrix entries, which its [`Image`] can hold (ledger §3.31).
fn sitk_pixel_id(kind: PixelKind, components: usize, component: PixelId) -> Result<PixelId> {
    match kind {
        PixelKind::Scalar if components == 1 => Ok(component),
        PixelKind::Scalar | PixelKind::Vector | PixelKind::SymmetricSecondRankTensor => {
            Ok(component.vector_id())
        }
        PixelKind::Complex => match component {
            PixelId::Float32 => Ok(PixelId::ComplexFloat32),
            PixelId::Float64 => Ok(PixelId::ComplexFloat64),
            other => Err(unsupported(format!(
                "complex NRRD with component type {}",
                other.as_str()
            ))),
        },
    }
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Everything `ReadImageInformation` computes.
struct Information {
    pixel_id: PixelId,
    components: usize,
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
    metadata: BTreeMap<String, String>,
    order: AxisOrder,
}

/// `NrrdImageIO::ReadImageInformation` (itkNrrdImageIO.cxx:560-1065).
fn image_information(nrrd: &Nrrd) -> Result<Information> {
    if nrrd.ntype == TYPE_BLOCK {
        return Err(unsupported("Cannot currently handle nrrdTypeBlock"));
    }
    let component = type_component_id(nrrd.ntype)
        .ok_or_else(|| unsupported("Nrrd type could not be mapped to an ITK component type"))?;

    let order = axis_order(nrrd);
    if nrrd.space_dim > 0 && nrrd.space_dim != order.domain_axes {
        return Err(bad(format!(
            "number of domain axes in the NRRD file ({}) doesn't match dimension of \
             space in which orientation is defined ({}). This is not supported.",
            order.domain_axes, nrrd.space_dim
        )));
    }

    let (kind, components, dimension) = match order.pixel_axis {
        None => (PixelKind::Scalar, 1, nrrd.dim),
        Some(p) => {
            let (kind, components) = pixel_kind_of(nrrd.axis[p].kind, nrrd.axis[p].size)?;
            (kind, components, order.image_axes.len())
        }
    };
    let pixel_id = sitk_pixel_id(kind, components, component)?;

    let mut size = Vec::with_capacity(dimension);
    let mut spacing = vec![1.0; dimension];
    let mut origin = vec![0.0; dimension];
    let mut direction = identity(dimension);

    // itkNrrdImageIO.cxx:764-786 converts the anatomical spaces to LPS and, in a
    // bare `default:` arm, silently leaves every other named space unconverted —
    // so `scanner-xyz`, `3D-right-handed`, `right-up` and the like load
    // mis-oriented, their direction cosines used verbatim as if already LPS
    // (ledger §2.82). This port instead converts every space with a well-defined
    // LPS mapping — the anatomical spaces and their `-time` variants, whose flip
    // is purely spatial — and rejects a named non-anatomical space with a typed
    // error rather than loading it silently mis-oriented. The unknown space (no
    // `space:` field) carries no anatomical claim, so its directions are used
    // as-is, exactly as before.
    let mut factors = [1.0f64; SPACE_DIM_MAX];
    let normalize_to_lps = match nrrd.space {
        SPACE_RAS | SPACE_RAST => {
            factors[0] = -1.0; // R -> L
            factors[1] = -1.0; // A -> P
            true
        }
        SPACE_LAS | SPACE_LAST => {
            factors[1] = -1.0; // A -> P
            true
        }
        SPACE_LPS | SPACE_LPST => true,
        SPACE_RIGHT_UP
        | SPACE_RIGHT_DOWN
        | SPACE_SCANNER_XYZ
        | SPACE_SCANNER_XYZ_TIME
        | SPACE_3D_RIGHT
        | SPACE_3D_LEFT
        | SPACE_3D_RIGHT_TIME
        | SPACE_3D_LEFT_TIME => {
            return Err(unsupported(format!(
                "NRRD space \"{}\" has no well-defined conversion to LPS; refusing to \
                 load it silently mis-oriented (ledger §2.82)",
                SPACE_STR[nrrd.space as usize]
            )));
        }
        _ => false,
    };

    for itk_axis in 0..order.domain_axes {
        let nrrd_axis = order.image_axes[itk_axis];
        let axis = &nrrd.axis[nrrd_axis];
        size.push(axis.size);
        if air_exists(axis.spacing) {
            if nrrd.space_dim > 0 {
                return Err(bad(
                    "Error interpreting nrrd spacing (nrrdSpacingStatusScalarWithSpace)",
                ));
            }
            spacing[itk_axis] = axis.spacing;
        } else if nrrd.space_dim > 0 && air_exists(axis.space_direction[0]) {
            let norm = axis.space_direction[..nrrd.space_dim]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            if !air_exists(norm) {
                continue;
            }
            spacing[itk_axis] = norm;
            // `spaceDirStd` is `imageDimensions` long and zero-filled, so the
            // rows past the domain axes are zero, not identity.
            for row in 0..dimension {
                direction[row * dimension + itk_axis] = if row < order.domain_axes {
                    factors[row] * axis.space_direction[row] / norm
                } else {
                    0.0
                };
            }
        }
    }
    // Extra range axes become ITK domain axes with unit spacing and a unit
    // direction (itkNrrdImageIO.cxx:830-849).
    for itk_axis in order.domain_axes..order.image_axes.len() {
        size.push(nrrd.axis[order.image_axes[itk_axis]].size);
    }

    if nrrd.space_dim > 0 {
        if air_exists(nrrd.space_origin[0]) {
            for axis in 0..nrrd.space_dim {
                origin[axis] = factors[axis] * nrrd.space_origin[axis];
            }
        }
    } else {
        let domain: Vec<usize> = nrrd.domain_axes();
        match origin_calculate(nrrd, &domain) {
            OriginStatus::Okay(values) => origin[..domain.len()].copy_from_slice(&values),
            OriginStatus::NoMin | OriginStatus::NoMaxOrSpacing => {}
            OriginStatus::Bad => return Err(bad("Error interpreting nrrd origin status")),
        }
    }

    Ok(Information {
        pixel_id,
        components,
        size,
        spacing,
        origin,
        direction,
        metadata: build_metadata(nrrd, &order, normalize_to_lps, &factors),
        order,
    })
}

enum OriginStatus {
    Okay(Vec<f64>),
    NoMin,
    NoMaxOrSpacing,
    Bad,
}

/// `nrrdOriginCalculate` (simple.c:229-302) with `defaultCenter = cell`.
///
/// Upstream's `gotMin` loop reads `axis[0]->min` on every iteration instead of
/// `axis[ai]->min` (simple.c:274) — bug §1.47, fixed in this port: each axis's
/// own `min` is tested, matching the `gotMaxOrSpacing` loop directly below it
/// in upstream, which correctly indexes by `ai`.
fn origin_calculate(nrrd: &Nrrd, domain: &[usize]) -> OriginStatus {
    let axes: Vec<&Axis> = domain.iter().map(|&a| &nrrd.axis[a]).collect();
    if axes.iter().any(|a| air_exists(a.space_direction[0])) && nrrd.space_dim > 0 {
        return OriginStatus::Bad;
    }
    if !axes.iter().all(|a| air_exists(a.min)) {
        return OriginStatus::NoMin;
    }
    if !axes
        .iter()
        .all(|a| air_exists(a.max) || air_exists(a.spacing))
    {
        return OriginStatus::NoMaxOrSpacing;
    }
    let origin = axes
        .iter()
        .map(|a| {
            let center = if a.center != 0 { a.center } else { CENTER_CELL };
            let denom = if center == CENTER_CELL {
                a.size as f64
            } else {
                (a.size - 1) as f64
            };
            let spacing = if air_exists(a.spacing) {
                a.spacing
            } else {
                (a.max - a.min) / denom
            };
            a.min
                + if center == CENTER_CELL {
                    spacing / 2.0
                } else {
                    0.0
                }
        })
        .collect();
    OriginStatus::Okay(origin)
}

/// The dictionary `ReadImageInformation` fills (itkNrrdImageIO.cxx:924-1057).
///
/// Upstream stores `NRRD_thicknesses[i]`, `NRRD_old min`, `NRRD_old max` as
/// `double` and `NRRD_measurement frame` as `vector<vector<double>>`. This
/// port's dictionary is `String -> String`, so each is stringified — with
/// Rust's shortest round-trip form for the doubles, and with the header's own
/// parenthesised column spelling for the frame. Ledger §4.52.
fn build_metadata(
    nrrd: &Nrrd,
    order: &AxisOrder,
    normalize_to_lps: bool,
    factors: &[f64; SPACE_DIM_MAX],
) -> BTreeMap<String, String> {
    let mut dict = BTreeMap::new();
    dict.insert("ITK_InputFilterName".to_string(), "NrrdImageIO".to_string());
    for (key, value) in &nrrd.kvp {
        dict.insert(key.clone(), value.clone());
    }

    for (itk_axis, &nrrd_axis) in order.image_axes.iter().enumerate() {
        let axis = &nrrd.axis[nrrd_axis];
        if air_exists(axis.thickness) {
            dict.insert(
                format!("NRRD_{}[{itk_axis}]", FIELD_STR[F_THICKNESSES]),
                axis.thickness.to_string(),
            );
        }
        if axis.center != 0 {
            dict.insert(
                format!("NRRD_{}[{itk_axis}]", FIELD_STR[F_CENTERS]),
                CENTER_STR[axis.center as usize].to_string(),
            );
        }
        if axis.kind != 0 {
            dict.insert(
                format!("NRRD_{}[{itk_axis}]", FIELD_STR[F_KINDS]),
                KIND_STR[axis.kind as usize].to_string(),
            );
        }
        if !axis.label.is_empty() {
            dict.insert(
                format!("NRRD_{}[{itk_axis}]", FIELD_STR[F_LABELS]),
                axis.label.clone(),
            );
        }
        if !axis.units.is_empty() {
            dict.insert(
                format!("NRRD_{}[{itk_axis}]", FIELD_STR[F_UNITS]),
                axis.units.clone(),
            );
        }
    }

    if let Some(pixel) = order.pixel_axis {
        let axis = &nrrd.axis[pixel];
        if axis.kind != 0 {
            dict.insert(
                format!("NRRD_{}[pixel]", FIELD_STR[F_KINDS]),
                KIND_STR[axis.kind as usize].to_string(),
            );
        }
        if !axis.label.is_empty() {
            dict.insert(
                format!("NRRD_{}[pixel]", FIELD_STR[F_LABELS]),
                axis.label.clone(),
            );
        }
        if !axis.units.is_empty() {
            dict.insert(
                format!("NRRD_{}[pixel]", FIELD_STR[F_UNITS]),
                axis.units.clone(),
            );
        }
        dict.insert("NRRD_pixel_original_axis".to_string(), pixel.to_string());
    }

    if !nrrd.content.is_empty() {
        dict.insert(
            format!("NRRD_{}", FIELD_STR[F_CONTENT]),
            nrrd.content.clone(),
        );
    }
    if air_exists(nrrd.old_min) {
        dict.insert(
            format!("NRRD_{}", FIELD_STR[F_OLD_MIN]),
            nrrd.old_min.to_string(),
        );
    }
    if air_exists(nrrd.old_max) {
        dict.insert(
            format!("NRRD_{}", FIELD_STR[F_OLD_MAX]),
            nrrd.old_max.to_string(),
        );
    }
    if nrrd.space != 0 {
        let space = if normalize_to_lps {
            SPACE_LPS
        } else {
            nrrd.space
        };
        dict.insert(
            format!("NRRD_{}", FIELD_STR[F_SPACE]),
            SPACE_STR[space as usize].to_string(),
        );
    }
    if air_exists(nrrd.measurement_frame[0][0]) {
        let n = order.domain_axes;
        let columns: Vec<String> = (0..n)
            .map(|column| {
                let coefficients: Vec<String> = (0..n)
                    .map(|i| {
                        let scale = if n <= 3 { factors[i] } else { 1.0 };
                        format_g(scale * nrrd.measurement_frame[column][i])
                    })
                    .collect();
                format!("({})", coefficients.join(","))
            })
            .collect();
        dict.insert(
            format!("NRRD_{}", FIELD_STR[F_MEASUREMENT_FRAME]),
            columns.join(" "),
        );
    }
    dict
}

/// `NrrdImageIO::ReadImageInformation`, exposed the way this crate's
/// [`ImageIo`] wants it.
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    let bytes = read_header_bytes(path)?;
    let parsed = read_header(&bytes)?;
    let info = image_information(&parsed.nrrd)?;
    Ok(ImageInformation {
        pixel_id: info.pixel_id,
        dimension: info.size.len(),
        number_of_components: info.components,
        size: info.size,
        spacing: info.spacing,
        origin: info.origin,
        direction: info.direction,
        metadata: info.metadata,
    })
}

/// Read the header text only: up to and including the blank line that
/// terminates it, or the whole file for a `.nhdr` with no blank line.
fn read_header_bytes(path: &Path) -> Result<Vec<u8>> {
    use std::io::BufRead;

    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut bytes = Vec::new();
    loop {
        let start = bytes.len();
        if reader.read_until(b'\n', &mut bytes)? == 0 {
            return Ok(bytes);
        }
        if bytes[start..].iter().all(|&b| b == b'\n' || b == b'\r') {
            return Ok(bytes);
        }
    }
}

/// Read a NRRD image — `NrrdImageIO::ReadImageInformation` followed by
/// `NrrdImageIO::Read` (itkNrrdImageIO.cxx:1069-1213).
///
/// The pixel axis is permuted to be the fastest one when the file did not
/// already put it there, so `kinds: domain domain vector` and `kinds: vector
/// domain domain` yield the same image.
pub fn read(path: &Path) -> Result<Image> {
    let bytes = std::fs::read(path)?;
    let parsed = read_header(&bytes)?;
    let info = image_information(&parsed.nrrd)?;
    let nrrd = &parsed.nrrd;

    let mut data = read_data(path, &bytes, &parsed)?;
    if info.order.permute {
        let mut axmap = Vec::with_capacity(nrrd.dim);
        axmap.extend(info.order.pixel_axis);
        axmap.extend(info.order.image_axes.iter().copied());
        let sizes: Vec<usize> = nrrd.axis.iter().map(|a| a.size).collect();
        data = permute(&data, &sizes, &axmap, type_size(nrrd.ntype));
    }

    // A `3D-masked-symmetric-matrix` pixel axis carries a leading mask channel
    // that `ReadImageInformation` excludes from the component count and `Read`
    // then crops out of the data (itkNrrdImageIO.cxx:1177-1204). After the
    // permute above the pixel axis is the fastest one, so its on-disk size can
    // exceed the reported component count only for this masked case; drop the
    // leading mask component(s) of every pixel (ledger §3.31).
    if let Some(pixel_axis) = info.order.pixel_axis {
        let on_disk = nrrd.axis[pixel_axis].size;
        if on_disk > info.components {
            let esz = type_size(nrrd.ntype);
            let stride = on_disk * esz;
            let drop = (on_disk - info.components) * esz;
            let mut cropped = Vec::with_capacity(data.len() / stride * info.components * esz);
            for chunk in data.chunks_exact(stride) {
                cropped.extend_from_slice(&chunk[drop..]);
            }
            data = cropped;
        }
    }

    let buffer = buffer_from_le_bytes(info.pixel_id, &data);
    let mut image = build_image(buffer, &info)?;
    for (key, value) in &info.metadata {
        image.set_meta_data(key, value);
    }
    Ok(image)
}

fn build_image(buffer: PixelBuffer, info: &Information) -> Result<Image> {
    let Information {
        pixel_id,
        components,
        size,
        spacing,
        origin,
        direction,
        ..
    } = info;

    if !pixel_id.is_complex() {
        return if *components == 1 {
            Image::from_parts(
                buffer,
                size.clone(),
                spacing.clone(),
                origin.clone(),
                direction.clone(),
            )
        } else {
            Image::from_parts_vector(
                buffer,
                *components,
                size.clone(),
                spacing.clone(),
                origin.clone(),
                direction.clone(),
            )
        }
        .map_err(IoError::Core);
    }

    // `Image::assemble` is private, so a complex image is built through
    // `from_vec_complex` and then given its geometry.
    let mut image = match &buffer {
        PixelBuffer::Float32(v) => Image::from_vec_complex(
            size,
            v.chunks_exact(2)
                .map(|c| Complex::new(c[0], c[1]))
                .collect(),
        ),
        PixelBuffer::Float64(v) => Image::from_vec_complex(
            size,
            v.chunks_exact(2)
                .map(|c| Complex::new(c[0], c[1]))
                .collect(),
        ),
        _ => {
            return Err(unsupported(
                "complex NRRD with a non-floating component type",
            ));
        }
    }
    .map_err(IoError::Core)?;
    image.set_spacing(spacing).map_err(IoError::Core)?;
    image.set_origin(origin).map_err(IoError::Core)?;
    image.set_direction(direction).map_err(IoError::Core)?;
    Ok(image)
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// `nrrd__FormatNRRD_whichVersion` (formatNRRD.c).
fn which_version(nrrd: &Nrrd, io: &IoState) -> u32 {
    if io.encoding == ENC_ZRL || nrrd.space == SPACE_RIGHT_UP || nrrd.space == SPACE_RIGHT_DOWN {
        6
    } else if air_exists(nrrd.measurement_frame[0][0]) {
        5
    } else if nrrd.axis.iter().any(|a| air_exists(a.thickness))
        || nrrd.space != 0
        || nrrd.space_dim > 0
        || !nrrd.sample_units.is_empty()
        || io.data_fn.len() > 1
    {
        4
    } else if nrrd.axis.iter().any(|a| a.kind != 0) {
        3
    } else if !nrrd.kvp.is_empty() {
        2
    } else {
        1
    }
}

/// `strcatSpaceVector` (write.c:216-236).
fn space_vector_str(v: &[f64]) -> String {
    if !air_exists(v[0]) {
        return "none".to_string();
    }
    let coefficients: Vec<String> = v.iter().map(|x| format_g(*x)).collect();
    format!("({})", coefficients.join(","))
}

/// `nrrd__FieldInteresting` + `nrrd__SprintFieldInfo` (write.c:239-780), for
/// the fields `NrrdImageIO::Write` can produce. Emitted in field-enum order,
/// as `formatNRRD_write` does.
fn build_header(nrrd: &Nrrd, io: &IoState) -> String {
    let mut out = format!("NRRD{:04}\n", which_version(nrrd, io));
    out.push_str("# Complete NRRD file format specification at:\n");
    out.push_str("# http://teem.sourceforge.net/nrrd/format.html\n");

    let dim = nrrd.dim;
    let field = |id: usize| FIELD_STR[id];
    let join = |values: Vec<String>| values.join(" ");

    if !nrrd.content.is_empty() {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_CONTENT),
            one_linify(&nrrd.content)
        ));
    }
    out.push_str(&format!(
        "{}: {}\n",
        field(F_TYPE),
        TYPE_STR[nrrd.ntype as usize]
    ));
    out.push_str(&format!("{}: {dim}\n", field(F_DIMENSION)));
    if nrrd.space != 0 {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_SPACE),
            SPACE_STR[nrrd.space as usize]
        ));
    } else if nrrd.space_dim > 0 {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_SPACE_DIMENSION),
            nrrd.space_dim
        ));
    }
    out.push_str(&format!(
        "{}: {}\n",
        field(F_SIZES),
        join(nrrd.axis.iter().map(|a| a.size.to_string()).collect())
    ));
    if nrrd.axis.iter().any(|a| air_exists(a.thickness)) {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_THICKNESSES),
            join(nrrd.axis.iter().map(|a| format_g(a.thickness)).collect())
        ));
    }
    if nrrd.space_dim > 0 {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_SPACE_DIRECTIONS),
            join(
                nrrd.axis
                    .iter()
                    .map(|a| space_vector_str(&a.space_direction[..nrrd.space_dim]))
                    .collect()
            )
        ));
    }
    if nrrd.axis.iter().any(|a| a.center != 0) {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_CENTERS),
            join(
                nrrd.axis
                    .iter()
                    .map(|a| if a.center != 0 {
                        CENTER_STR[a.center as usize].to_string()
                    } else {
                        "???".to_string()
                    })
                    .collect()
            )
        ));
    }
    if nrrd.axis.iter().any(|a| a.kind != 0) {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_KINDS),
            join(
                nrrd.axis
                    .iter()
                    .map(|a| if a.kind != 0 {
                        KIND_STR[a.kind as usize].to_string()
                    } else {
                        "???".to_string()
                    })
                    .collect()
            )
        ));
    }
    for (id, get) in [(F_LABELS, 0usize), (F_UNITS, 1usize)] {
        let pick = |a: &Axis| {
            if get == 0 {
                a.label.clone()
            } else {
                a.units.clone()
            }
        };
        if nrrd.axis.iter().any(|a| !pick(a).is_empty()) {
            let quoted: Vec<String> = nrrd
                .axis
                .iter()
                .map(|a| format!("\"{}\"", write_escaped(&pick(a), "\"")))
                .collect();
            out.push_str(&format!("{}: {}\n", field(id), quoted.join(" ")));
        }
    }
    if air_exists(nrrd.old_min) {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_OLD_MIN),
            format_g(nrrd.old_min)
        ));
    }
    if air_exists(nrrd.old_max) {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_OLD_MAX),
            format_g(nrrd.old_max)
        ));
    }
    // `nio->endian` is `airEndianUnknown` for everything ITK writes, so the
    // header records `airMyEndian()` and no swapping happens (write.c:637-651).
    if encoding_endian_matters(io.encoding) && type_size(nrrd.ntype) > 1 {
        out.push_str(&format!("{}: little\n", field(F_ENDIAN)));
    }
    out.push_str(&format!(
        "{}: {}\n",
        field(F_ENCODING),
        encoding_name(io.encoding)
    ));
    if nrrd.space_dim > 0 && air_exists(nrrd.space_origin[0]) {
        out.push_str(&format!(
            "{}: {}\n",
            field(F_SPACE_ORIGIN),
            space_vector_str(&nrrd.space_origin[..nrrd.space_dim])
        ));
    }
    if nrrd.space_dim > 0 && air_exists(nrrd.measurement_frame[0][0]) {
        let columns: Vec<String> = (0..nrrd.space_dim)
            .map(|c| space_vector_str(&nrrd.measurement_frame[c][..nrrd.space_dim]))
            .collect();
        out.push_str(&format!(
            "{}: {}\n",
            field(F_MEASUREMENT_FRAME),
            columns.join(" ")
        ));
    }
    // `data file` is the last field for a reason: the `LIST` form needs the
    // filenames on the lines that follow. Upstream then emits comments and
    // key/value pairs anyway, which corrupts such a header (ledger §2.84);
    // this port only ever writes the single-filename form, where the bug is
    // unreachable.
    if !io.data_fn.is_empty() {
        out.push_str(&format!("{}: {}\n", field(F_DATA_FILE), io.data_fn[0]));
    }
    for comment in &nrrd.comments {
        out.push_str(&format!("#{comment}\n"));
    }
    for (key, value) in &nrrd.kvp {
        out.push_str(&format!(
            "{}:={}\n",
            write_escaped(key, "\n\\"),
            write_escaped(value, "\n\\")
        ));
    }
    if io.data_fn.is_empty() {
        out.push('\n');
    }
    out
}

/// `TryDispatchNrrdReservedField` (itkNrrdImageIO.cxx:292-311): a leading
/// **substring** match against the reserved field names, in table order.
/// Returns `true` when the key was consumed, whether or not it did anything.
fn dispatch_reserved(
    nrrd: &mut Nrrd,
    key_field: &str,
    value: &str,
    pixel_axes: usize,
) -> Result<bool> {
    const PER_AXIS: [usize; 5] = [F_THICKNESSES, F_CENTERS, F_KINDS, F_LABELS, F_UNITS];
    const SCALAR: [usize; 5] = [
        F_OLD_MIN,
        F_OLD_MAX,
        F_SPACE,
        F_CONTENT,
        F_MEASUREMENT_FRAME,
    ];

    for id in PER_AXIS {
        let name = FIELD_STR[id];
        let Some(rest) = key_field.strip_prefix(name) else {
            continue;
        };
        // `sscanf(keyField + nameLen, "[%u]", &axi) == 1`: the trailing `]` is
        // not required by sscanf, and `[pixel]` simply fails to match.
        let index = rest
            .strip_prefix('[')
            .and_then(parse_leading_usize)
            .filter(|i| i + pixel_axes < nrrd.dim);
        if let Some(index) = index {
            let axis = &mut nrrd.axis[index + pixel_axes];
            match id {
                // `double thickness = 0.0; ExposeMetaData(...)` — an
                // unparseable value leaves the zero in place, which teem then
                // writes as a real thickness.
                F_THICKNESSES => axis.thickness = value.parse().unwrap_or(0.0),
                F_CENTERS => axis.center = air_enum_val(CENTER_EQV, value),
                F_KINDS => axis.kind = air_enum_val(KIND_EQV, value),
                F_LABELS => axis.label = value.to_string(),
                F_UNITS => axis.units = value.to_string(),
                _ => unreachable!("PER_AXIS is exhaustive"),
            }
        }
        return Ok(true);
    }

    for id in SCALAR {
        let name = FIELD_STR[id];
        if !key_field.starts_with(name) {
            continue;
        }
        match id {
            // `ExposeMetaData<double>` leaves the target untouched on failure.
            F_OLD_MIN => {
                if let Ok(v) = value.parse() {
                    nrrd.old_min = v;
                }
            }
            F_OLD_MAX => {
                if let Ok(v) = value.parse() {
                    nrrd.old_max = v;
                }
            }
            F_SPACE => {
                let space = air_enum_val(SPACE_EQV, value);
                if space_dimension(space) == nrrd.space_dim {
                    nrrd.space = space;
                }
            }
            F_CONTENT => nrrd.content = value.to_string(),
            F_MEASUREMENT_FRAME => write_measurement_frame(nrrd, key_field, value)?,
            _ => unreachable!("SCALAR is exhaustive"),
        }
        return Ok(true);
    }
    Ok(false)
}

/// `WriteMeasurementFrame` (itkNrrdImageIO.cxx:233-269). Upstream reads a
/// `vector<vector<double>>` from the dictionary and raises if it is smaller
/// than `spaceDim` on either side; here the value is the header's own
/// `(a,b,c) (d,e,f) ...` column spelling (ledger §4.52).
fn write_measurement_frame(nrrd: &mut Nrrd, key_field: &str, value: &str) -> Result<()> {
    let mut h = value;
    let mut columns = Vec::new();
    while !h.trim_matches(FIELD_SEP).is_empty() {
        columns.push(space_vector_parse(&mut h, nrrd.space_dim)?);
    }
    if columns.len() < nrrd.space_dim {
        return Err(bad(format!(
            "NRRD '{key_field}': supplied measurement frame ({} columns) does not match \
             image space dimension {}. The measurement frame must be fully specified \
             for every space axis.",
            columns.len(),
            nrrd.space_dim
        )));
    }
    for (column, coefficients) in columns.into_iter().take(nrrd.space_dim).enumerate() {
        nrrd.measurement_frame[column] = coefficients;
    }
    Ok(())
}

/// `NrrdImageIO::Write` (itkNrrdImageIO.cxx:1259-1420).
///
/// `.nhdr` selects the detached header `nrrdSave` produces: the data goes into
/// a sibling file named `<stem>.` plus the encoding's `suffix` — `raw`, or
/// `raw.gz` when compressing (formatNRRD.c:720-728). `.nrrd` embeds the data
/// after the blank line.
///
/// With [`WriteOptions::use_compression`] set, the encoding becomes `gzip`:
/// `NrrdImageIO::Write` picks `m_NrrdCompressionEncoding`, which
/// `InternalSetCompressor("")` resolved to `nrrdEncodingGzip` at construction
/// (itkNrrdImageIO.cxx:380-392, 1404-1409), and passes the compression level
/// down as `nio->zlibLevel`. Otherwise the encoding is `raw` — `GetFileType()`
/// is `Binary` for everything SimpleITK writes, so the `ascii` arm is dead.
///
/// The image's meta-data dictionary is written, as upstream's does: keys
/// beginning with `NRRD_` are routed to the reserved-field handlers, and every
/// other key becomes a `key:=value` pair. `NRRD_pixel_original_axis` — a key
/// the reader itself sets — matches no reserved field and is silently dropped
/// (ledger §2.85).
pub fn write(image: &Image, path: &Path, options: &WriteOptions) -> Result<()> {
    let dimension = image.dimension();
    let components = image.buffer_stride();
    let pixel_axes = usize::from(components > 1);

    let mut nrrd = Nrrd {
        dim: dimension + pixel_axes,
        ntype: component_id_type(image.pixel_id()),
        axis: vec![Axis::default(); dimension + pixel_axes],
        ..Nrrd::default()
    };
    if pixel_axes == 1 {
        nrrd.axis[0].size = components;
        nrrd.axis[0].kind = if image.pixel_id().is_complex() {
            KIND_COMPLEX
        } else {
            KIND_VECTOR
        };
    }

    // A `NRRD_kinds[i]` of `list` marks an ITK axis as a NRRD list axis, which
    // leaves it out of the space and gives it a `none` space direction.
    let mut list_axes = 0usize;
    for itk_axis in 0..dimension {
        let key = format!("NRRD_{}[{itk_axis}]", FIELD_STR[F_KINDS]);
        if image.meta_data(&key) == Some("list") {
            nrrd.axis[itk_axis + pixel_axes].kind = KIND_LIST;
            list_axes += 1;
        }
    }
    let space_dim = dimension - list_axes;
    if space_dim == 0 || space_dim > SPACE_DIM_MAX {
        return Err(unsupported(format!(
            "NRRD space dimension {space_dim} outside [1,{SPACE_DIM_MAX}]"
        )));
    }
    nrrd.space_dim = space_dim;
    // "special case: ITK is LPS in 3-D" (itkNrrdImageIO.cxx:1362-1365).
    if space_dim == 3 {
        nrrd.space = SPACE_LPS;
    }

    let mut origin = Vec::with_capacity(space_dim);
    for itk_axis in 0..dimension {
        let nrrd_axis = itk_axis + pixel_axes;
        nrrd.axis[nrrd_axis].size = image.size()[itk_axis];
        if nrrd.axis[nrrd_axis].kind == KIND_LIST {
            continue;
        }
        nrrd.axis[nrrd_axis].kind = KIND_DOMAIN;
        origin.push(image.origin()[itk_axis]);
        let spacing = image.spacing()[itk_axis];
        for row in 0..space_dim {
            nrrd.axis[nrrd_axis].space_direction[row] =
                spacing * image.direction()[row * dimension + itk_axis];
        }
    }
    nrrd.space_origin[..space_dim].copy_from_slice(&origin);

    for key in image.meta_data_keys() {
        let value = image.meta_data(key).unwrap_or_default();
        if let Some(key_field) = key.strip_prefix("NRRD_") {
            dispatch_reserved(&mut nrrd, key_field, value, pixel_axes)?;
        } else {
            nrrd.kvp.push((key.to_string(), value.to_string()));
        }
    }

    let mut io = IoState {
        encoding: if options.use_compression {
            ENC_GZIP
        } else {
            ENC_RAW
        },
        ..IoState::default()
    };
    let data = buffer_to_le_bytes(image.buffer());
    let data = if io.encoding == ENC_GZIP {
        gzip_compress(&data, options.resolved_level(ITK_DEFAULT_COMPRESSION_LEVEL))
    } else {
        data
    };

    let detached = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("nhdr"));
    if detached {
        let stem = path
            .file_stem()
            .ok_or_else(|| IoError::InvalidPath(path.to_path_buf()))?;
        let mut raw_name = stem.to_os_string();
        raw_name.push(format!(".{}", encoding_suffix(io.encoding)));
        io.data_fn.push(raw_name.to_string_lossy().into_owned());
        std::fs::write(path, build_header(&nrrd, &io))?;
        std::fs::write(path.with_file_name(raw_name), data)?;
    } else {
        let mut bytes = build_header(&nrrd, &io).into_bytes();
        bytes.extend_from_slice(&data);
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ImageIo
// ---------------------------------------------------------------------------

/// `itk::NrrdImageIO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NrrdImageIo;

impl ImageIo for NrrdImageIo {
    fn name(&self) -> &'static str {
        "NrrdImageIO"
    }

    fn supported_read_extensions(&self) -> &'static [&'static str] {
        &[".nrrd", ".nhdr"]
    }

    fn supported_write_extensions(&self) -> &'static [&'static str] {
        &[".nrrd", ".nhdr"]
    }

    /// `NrrdImageIO::CanReadFile` (itkNrrdImageIO.cxx): the extension must be
    /// supported *and* the first four bytes must be `NRRD`. A file shorter
    /// than four bytes fails the `inputStream.eof()` check.
    fn can_read_file(&self, path: &Path) -> bool {
        use std::io::Read;

        if !crate::image_io::has_supported_extension(path, self.supported_read_extensions(), true) {
            return false;
        }
        let Ok(mut file) = std::fs::File::open(path) else {
            return false;
        };
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).is_ok() && &magic == b"NRRD"
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

    #[test]
    fn format_g_matches_c_printf() {
        assert_eq!(format_g(0.0), "0");
        assert_eq!(format_g(-0.0), "-0");
        assert_eq!(format_g(1.0), "1");
        assert_eq!(format_g(0.5), "0.5");
        assert_eq!(format_g(1.25), "1.25");
        assert_eq!(format_g(3.0), "3");
        assert_eq!(format_g(0.0001), "0.0001");
        assert_eq!(format_g(0.00001), "1e-05");
        assert_eq!(format_g(1e20), "1e+20");
        assert_eq!(format_g(1.0 / 3.0), "0.33333333333333331");
        assert_eq!(format_g(f64::NAN), "NaN");
        assert_eq!(format_g(f64::INFINITY), "inf");
        assert_eq!(format_g(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn air_single_sscanf_matches_substring_rules() {
        assert!(air_single_sscanf_double("banana").unwrap().is_nan());
        assert_eq!(
            air_single_sscanf_double("-infinity"),
            Some(f64::NEG_INFINITY)
        );
        assert_eq!(air_single_sscanf_double("infty"), Some(f64::INFINITY));
        assert_eq!(air_single_sscanf_double("1.5junk"), Some(1.5));
        assert_eq!(air_single_sscanf_double("junk"), None);
    }

    #[test]
    fn space_vector_parse_reads_none_and_tuples() {
        let mut h = "none (1,0,0)";
        let v = space_vector_parse(&mut h, 3).unwrap();
        assert!(v[0].is_nan());
        let v = space_vector_parse(&mut h, 3).unwrap();
        assert_eq!(&v[..3], &[1.0, 0.0, 0.0]);
        assert!(h.is_empty());
    }

    #[test]
    fn kinds_none_clears_kind_and_leaves_centering_alone() {
        // Fixed §1.48: upstream's `kinds: ... none` arm writes `.center`
        // instead of `.kind` (parseNrrd.c:621-623). This port clears
        // `.kind`, matching the `???` arm, and leaves `centerings:` intact.
        let header = b"NRRD0004\n\
                       type: float\n\
                       dimension: 2\n\
                       sizes: 2 2\n\
                       centerings: cell cell\n\
                       kinds: domain none\n\
                       endian: little\n\
                       encoding: raw\n\
                       data file: x.raw\n";
        let parsed = read_header(header).unwrap();
        assert_eq!(parsed.nrrd.axis[1].kind, 0, "none must clear the kind");
        assert_eq!(
            parsed.nrrd.axis[1].center, CENTER_CELL,
            "none must not touch the centering"
        );
        assert_eq!(parsed.nrrd.axis[0].kind, KIND_DOMAIN);
        assert_eq!(parsed.nrrd.axis[0].center, CENTER_CELL);
    }
}
