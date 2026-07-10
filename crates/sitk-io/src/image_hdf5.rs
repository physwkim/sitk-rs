//! HDF5 image files (`.h5`, `.hdf5`, and six more extensions) — a port of
//! `itk::HDF5ImageIO` (`Modules/IO/HDF5/src/itkHDF5ImageIO.cxx`), reached
//! through [`crate::read_image`] / [`crate::write_image`] once
//! [`crate::create_image_io`] picks the IO.
//!
//! The on-disk layout, from `WriteImageInformation` (`:1114-1204`) and the
//! class comment (`itkHDF5ImageIO.h:56-81`), with `N` the image dimension:
//!
//! ```text
//! /ITKVersion                 -- 1 variable-length string
//! /HDFVersion                 -- 1 variable-length string
//! /ITKImage                   -- group
//! /ITKImage/0                 -- group ("0" is the only index ITK ever writes)
//! /ITKImage/0/Origin          -- N doubles
//! /ITKImage/0/Directions      -- N x N doubles
//! /ITKImage/0/Spacing         -- N doubles
//! /ITKImage/0/Dimension       -- N uint64
//! /ITKImage/0/VoxelType       -- 1 variable-length string
//! /ITKImage/0/VoxelData       -- chunked, deflated voxel array
//! /ITKImage/0/MetaData        -- group
//! /ITKImage/0/MetaData/<key>  -- one dataset per dictionary entry
//! ```
//!
//! # Axis order is not uniform across the datasets
//!
//! `Dimension`, `Origin` and `Spacing` are written straight out of ITK's
//! `m_Dimensions` / `m_Origin` / `m_Spacing`, so they are in ITK order —
//! fastest-moving axis first. Only `VoxelData`'s *dataspace* is reversed:
//! "HDF5 dimensions listed slowest moving first, ITK are fastest moving first"
//! (`:1170-1176`). A 4x3 2-D image therefore stores `Dimension = [4, 3]` and a
//! `VoxelData` of shape `[3, 4]`.
//!
//! `Directions` holds `m_Direction[i]` as its row `i`, and `m_Direction[i]` is
//! *column* `i` of the direction matrix — "direction cosines are stored as
//! columns of the direction matrix" (`itkImageFileReader.hxx:180-183`,
//! `itkImageFileWriter.hxx:188-194`). The dataset is the transpose of the
//! row-major matrix [`sitk_core::Image::direction`] returns.
//!
//! # `VoxelType` is written and never read
//!
//! `ReadImageInformation` derives the component type from `VoxelData`'s own
//! HDF5 datatype and never opens `VoxelType`, which exists only to "make the
//! file more user friendly with respect to HDF5 viewers"
//! (`itkHDF5ImageIO.h:68-71`). A file whose `VoxelType` says `FLOAT` over an
//! `int16` `VoxelData` reads back as `Int16` (ledger §2.130), and a file with
//! no `VoxelType` dataset at all reads fine.
//!
//! # SimpleITK cannot read a non-scalar HDF5 image — fixed in this port
//!
//! `ReadImageInformation` never calls `SetPixelType`, so `m_PixelType` keeps
//! its `IOPixelEnum::SCALAR` default (`itkImageIOBase.h:801`) even when it has
//! just set `m_NumberOfComponents` to 3. `ImageReaderBase::
//! GetPixelIDFromImageIO` takes the scalar branch only for
//! `numberOfComponents == 1`, finds `SCALAR` in none of the vector pixel types,
//! and throws `"Unknown PixelType: ..."` (`sitkImageReaderBase.cxx:215-238`).
//! So a vector or complex image *writes* through SimpleITK and can never be
//! read back by it — the one missing `SetPixelType(VECTOR)` call (ledger §3.47,
//! §5.25).
//!
//! This port closes that at source: [`write`] still takes any pixel type, and
//! [`read`] reconstructs the vector image the trailing component axis records.
//! A `VoxelData` of rank `dimension + 1` reads back as the vector pixel type
//! whose components are `VoxelData`'s own datatype ([`PixelId::vector_id`]),
//! exactly what `SetPixelType(VECTOR)` would have produced. The interleaved
//! on-disk buffer is already in SimpleITK's own vector-buffer order, so a
//! vector image round-trips byte-for-byte. A **complex** image reaches the
//! `ImageIO` as two `float` components per voxel and carries no complex marker
//! in the file, so it reads back as a two-component `VectorFloat32` /
//! `VectorFloat64`: the samples survive exactly, only the pixel-type label
//! widens from complex to vector.
//!
//! # Compression is unconditional upstream — gated by `use_compression` here
//!
//! `WriteImageInformation` calls `plist.setDeflate(this->GetCompressionLevel())`
//! with no `GetUseCompression()` guard — its own comment reads "we have implicit
//! compression enabled here?" (`:1186-1195`). `SetCompressionLevel` clamps to
//! `[1, GetMaximumCompressionLevel()]` and the constructor sets the maximum to
//! `9` and the level to `5`, so every HDF5 image ITK writes is deflated at
//! level 1 or above; `SetUseCompression(false)` cannot turn it off
//! (ledger §3.48).
//!
//! This port makes the flag live: [`write`] applies the deflate filter only
//! when [`WriteOptions::use_compression`] is set, and
//! [`WriteOptions::compression_level`] is meaningful only then. The chunk — the
//! `N-1` dimensional slab `[1, ...]` — is written unconditionally either way,
//! matching ITK's own unconditional `plist.setChunk`; an uncompressed write is
//! a chunked dataset with no filter, not a contiguous one.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::Path;

use rust_hdf5::{ByteOrder, DatatypeMessage, H5Dataset, H5File, H5Group};
use sitk_core::{Image, PixelBuffer, PixelId};

use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, ImageIo};
use crate::nrrd::c_format_g;
use crate::transform_hdf5::{HDF_VERSION, ITK_VERSION};
use crate::writer::WriteOptions;

/// `ImageGroup` (`itkHDF5ImageIO.cxx:64`), without the leading `/` that
/// `rust-hdf5` strips from every stored link path.
const IMAGE_GROUP: &str = "ITKImage";
/// The `<name>` subgroup. `WriteImageInformation` hardcodes `"/0"` and
/// `ReadImageInformation` hardcodes it back (`:706`, `:1147`); the class
/// comment's "name is arbitrary" is aspirational.
const INSTANCE: &str = "0";
/// `Origin` (`:65`).
const ORIGIN: &str = "Origin";
/// `Directions` (`:66`).
const DIRECTIONS: &str = "Directions";
/// `Spacing` (`:67`).
const SPACING: &str = "Spacing";
/// `Dimensions` (`:68`) — whose *value* is the singular `"/Dimension"`.
const DIMENSION: &str = "Dimension";
/// `VoxelType` (`:69`).
const VOXEL_TYPE: &str = "VoxelType";
/// `VoxelData` (`:70`).
const VOXEL_DATA: &str = "VoxelData";
/// `MetaDataName` (`:71`).
const META_DATA: &str = "MetaData";

/// The extensions `HDF5ImageIO`'s constructor registers for *both* reading and
/// writing (`itkHDF5ImageIO.cxx:36-43`) — the HDF4 spellings included, though
/// this IO can neither read nor write HDF4.
const EXTENSIONS: &[&str] = &[
    ".hdf", ".h4", ".hdf4", ".h5", ".hdf5", ".he4", ".he5", ".hd5",
];

/// `SetCompressionLevel(5)` in the constructor (`:45`), under the
/// `SetMaximumCompressionLevel(9)` set one line above.
const DEFAULT_COMPRESSION_LEVEL: i32 = 5;

/// `/ITKImage/0`.
fn instance_path() -> String {
    format!("{IMAGE_GROUP}/{INSTANCE}")
}

/// `/ITKImage/0/<name>`.
fn instance_dataset(name: &str) -> String {
    format!("{IMAGE_GROUP}/{INSTANCE}/{name}")
}

// ---------------------------------------------------------------------------
// Component types
// ---------------------------------------------------------------------------

/// `ComponentToString` (`itkHDF5ImageIO.cxx:214-260`) for the component types
/// SimpleITK can hand it. `SCHAR` spells itself `CHAR`, and a 64-bit
/// `int64_t` / `uint64_t` reaches `ComponentToString` as `LONG` / `ULONG` on
/// every LP64 platform, never as `LONGLONG` / `ULONGLONG`.
fn component_to_string(component: PixelId) -> &'static str {
    match component {
        PixelId::UInt8 => "UCHAR",
        PixelId::Int8 => "CHAR",
        PixelId::UInt16 => "USHORT",
        PixelId::Int16 => "SHORT",
        PixelId::UInt32 => "UINT",
        PixelId::Int32 => "INT",
        PixelId::UInt64 => "ULONG",
        PixelId::Int64 => "LONG",
        PixelId::Float32 => "FLOAT",
        PixelId::Float64 => "DOUBLE",
        other => unreachable!("{other:?} is not a component type"),
    }
}

/// `PredTypeToComponentType` (`itkHDF5ImageIO.cxx:119-175`), which compares the
/// stored datatype against each `H5::PredType::NATIVE_*` in turn and raises
/// `"unsupported HDF5 data type with id ..."` when none matches.
///
/// Two narrowings, both because [`rust_hdf5`] hands back the stored bytes where
/// `H5Dread` would convert them (ledger §4.82 sets the precedent):
///
/// * a big-endian dataset is rejected; libhdf5 would byte-swap it into the
///   `NATIVE_*` memory type;
/// * an integer whose bit precision or offset does not fill its bytes is
///   rejected; libhdf5 would unpack it.
fn datatype_to_component(datatype: &DatatypeMessage, what: &str) -> Result<PixelId> {
    let unsupported = |detail: String| IoError::UnsupportedHdf5Image(format!("{what}: {detail}"));
    match *datatype {
        DatatypeMessage::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            if byte_order != ByteOrder::LittleEndian {
                return Err(unsupported("big-endian integer dataset".to_string()));
            }
            if bit_offset != 0 || u32::from(bit_precision) != size * 8 {
                return Err(unsupported(format!(
                    "{bit_precision}-bit integer at bit offset {bit_offset} in {size} bytes"
                )));
            }
            match (size, signed) {
                (1, false) => Ok(PixelId::UInt8),
                (1, true) => Ok(PixelId::Int8),
                (2, false) => Ok(PixelId::UInt16),
                (2, true) => Ok(PixelId::Int16),
                (4, false) => Ok(PixelId::UInt32),
                (4, true) => Ok(PixelId::Int32),
                (8, false) => Ok(PixelId::UInt64),
                (8, true) => Ok(PixelId::Int64),
                _ => Err(IoError::MalformedHdf5Image(format!(
                    "unsupported HDF5 data type with {size}-byte integer elements in {what}"
                ))),
            }
        }
        DatatypeMessage::FloatingPoint {
            size, byte_order, ..
        } => {
            if byte_order != ByteOrder::LittleEndian {
                return Err(unsupported("big-endian floating-point dataset".to_string()));
            }
            match size {
                4 => Ok(PixelId::Float32),
                8 => Ok(PixelId::Float64),
                _ => Err(IoError::MalformedHdf5Image(format!(
                    "unsupported HDF5 data type with {size}-byte floating-point elements in {what}"
                ))),
            }
        }
        _ => Err(IoError::MalformedHdf5Image(format!(
            "unsupported HDF5 data type in {what}"
        ))),
    }
}

/// The pixel type of an HDF5 image, from the component type and count that
/// `HDF5ImageIO::ReadImageInformation` leaves behind.
///
/// Upstream never calls `SetPixelType`, so `m_PixelType` keeps its `SCALAR`
/// default even for a multi-component `VoxelData`, and
/// `ImageReaderBase::GetPixelIDFromImageIO` then throws `"Unknown PixelType"`
/// for any count above 1 (`sitkImageReaderBase.cxx:200-241`, ledger §3.47).
/// This port closes that at source: a trailing component axis reads back as
/// the corresponding *vector* pixel type — the one `SetPixelType(VECTOR)`
/// would have selected. A single component stays scalar.
///
/// The HDF5 file records no complex marker, so a complex image (two `float`
/// components per voxel to the `ImageIO`) reads back as a two-component
/// `VectorFloat32`/`VectorFloat64`: the samples are preserved exactly, only
/// the pixel-type label widens from complex to vector.
fn pixel_id_from_image_io(component: PixelId, number_of_components: usize) -> PixelId {
    if number_of_components == 1 {
        component
    } else {
        component.vector_id()
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Everything `ReadImageInformation` deposits in the `ImageIOBase`.
struct Header {
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
    component: PixelId,
    number_of_components: usize,
    metadata: BTreeMap<String, String>,
}

impl Header {
    fn dimension(&self) -> usize {
        self.size.len()
    }
}

/// `HDF5ImageIO::CanReadFile` (`itkHDF5ImageIO.cxx:619-661`): the file must
/// exist, be an HDF5 file (`H5Fis_hdf5`), and hold a `/ITKImage` *link*. The
/// extension is never consulted, so an HDF5 image named `.mha` is still
/// claimed. Every failure — an absent file, a truncated superblock, an HDF5
/// file with no image group — is swallowed by the `catch (...)`.
///
/// `h5file.exists(ImageGroup)` asks only whether the link exists, not whether
/// it names a group, so a *dataset* called `/ITKImage` makes `CanReadFile`
/// answer yes and the following `ReadImageInformation` throw. That is
/// reproduced here (ledger §2.131), and it is where this IO parts company with
/// [`crate::transform_hdf5::can_read_file`], which consults `openGroup` and so
/// declines (ledger §2.123).
pub fn can_read_file(path: &Path) -> bool {
    let Ok(file) = H5File::open(path) else {
        return false;
    };
    let root = file.root_group();
    let has_link = |names: Result<Vec<String>>| {
        names
            .map(|names| names.iter().any(|name| name == IMAGE_GROUP))
            .unwrap_or(false)
    };
    has_link(root.group_names().map_err(IoError::from))
        || has_link(root.dataset_names().map_err(IoError::from))
}

/// `ReadDirections` (`itkHDF5ImageIO.cxx:512-560`): a rank-2 float dataset,
/// read as `double` when its elements are `sizeof(double)` wide and as `float`
/// otherwise. Row `i` of the dataset is `m_Direction[i]`, which is column `i`
/// of the direction matrix, so the return value is the row-major matrix.
///
/// Upstream sizes `rval` from `dim[1]` and each `rval[i]` from `dim[0]`, then
/// hands `rval[i]` — `dim[0]` long — to `SetDirection(i, ...)` for `i` in
/// `0..dim[1]`. A non-square dataset therefore walks off the end of ITK's
/// `m_Direction`; this port rejects it (ledger §4.89).
///
/// Returns the image dimension `SetNumberOfDimensions` is given alongside the
/// matrix, since `directions.size()` is where upstream learns it.
fn read_directions(file: &H5File, path: &str) -> Result<(usize, Vec<f64>)> {
    let dataset = file.dataset(path)?;
    let shape = dataset.shape();
    if shape.len() != 2 {
        // Upstream's leading space is upstream's.
        return Err(IoError::MalformedHdf5Image(
            " Wrong # of dims for Image Directions in HDF5 File".to_string(),
        ));
    }
    if shape[0] != shape[1] {
        return Err(IoError::UnsupportedHdf5Image(format!(
            "non-square {}x{} Directions matrix",
            shape[0], shape[1]
        )));
    }
    if shape[0] == 0 {
        // `SetNumberOfDimensions(0)` leaves an ITK image no axes at all; here
        // `Image::assemble` would assert. Upstream fails later, in `Read`.
        return Err(IoError::MalformedHdf5Image(
            "an empty Directions matrix gives the image no dimension".to_string(),
        ));
    }
    let transposed = read_doubles(&dataset, DIRECTIONS)?;
    let n = shape[0];
    if transposed.len() != n * n {
        return Err(IoError::TruncatedData);
    }
    // `dataset[i][j]` is direction cosine `j` of axis `i`, i.e. `D[j][i]`.
    let mut direction = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            direction[j * n + i] = transposed[i * n + j];
        }
    }
    Ok((n, direction))
}

/// `dirSet.getFloatType()` then a `NATIVE_DOUBLE` or `NATIVE_FLOAT` read: a
/// non-float dataset makes `getFloatType()` throw.
fn read_doubles(dataset: &H5Dataset, what: &str) -> Result<Vec<f64>> {
    match datatype_to_component(&dataset.datatype()?, what)? {
        PixelId::Float64 => Ok(dataset.read_raw::<f64>()?),
        PixelId::Float32 => Ok(dataset
            .read_raw::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect()),
        _ => Err(IoError::MalformedHdf5Image(format!(
            "{what} is not a floating-point dataset"
        ))),
    }
}

/// `ReadVector<double>` (`:468-487`) for `Origin` and `Spacing`.
///
/// `ReadImageInformation` assigns the whole vector to `m_Origin` and indexes
/// `spacing[i]` for `i < numDims`, so a dataset shorter than the dimension
/// reads out of bounds upstream; this port rejects it. A *longer* one is
/// truncated, as upstream's `SetSpacing(i, ...)` loop truncates it.
fn read_double_vector(file: &H5File, path: &str, what: &str, dimension: usize) -> Result<Vec<f64>> {
    let dataset = file.dataset(path)?;
    if dataset.shape().len() != 1 {
        // The copy-paste of `ReadVector`'s message from the transform IO says
        // `TransformType` in an image file too (ledger §2.129).
        return Err(IoError::MalformedHdf5Image(
            "Wrong # of dims for TransformType in HDF5 File".to_string(),
        ));
    }
    let mut values = read_doubles(&dataset, what)?;
    if values.len() < dimension {
        return Err(IoError::MalformedHdf5Image(format!(
            "{what} has {} entries for a {dimension}-dimensional image",
            values.len()
        )));
    }
    values.truncate(dimension);
    Ok(values)
}

/// `ReadVector<ImageIOBase::SizeValueType>` for `Dimension`. `SizeValueType` is
/// `unsigned long` (`itkIntTypes.h:86`), so `GetType` selects `NATIVE_ULONG`:
/// eight unsigned bytes on every platform SimpleITK ships.
fn read_dimensions(file: &H5File, path: &str, dimension: usize) -> Result<Vec<usize>> {
    let dataset = file.dataset(path)?;
    if dataset.shape().len() != 1 {
        return Err(IoError::MalformedHdf5Image(
            "Wrong # of dims for TransformType in HDF5 File".to_string(),
        ));
    }
    if datatype_to_component(&dataset.datatype()?, DIMENSION)? != PixelId::UInt64 {
        return Err(IoError::UnsupportedHdf5Image(format!(
            "{DIMENSION} is not a 64-bit unsigned dataset"
        )));
    }
    let values = dataset.read_raw::<u64>()?;
    if values.len() < dimension {
        return Err(IoError::MalformedHdf5Image(format!(
            "{DIMENSION} has {} entries for a {dimension}-dimensional image",
            values.len()
        )));
    }
    Ok(values[..dimension].iter().map(|&v| v as usize).collect())
}

/// `ReadImageInformation` (`itkHDF5ImageIO.cxx:690-955`), in its order:
/// `Directions` fixes the dimension, then `Origin`, `Spacing`, `Dimension`,
/// `VoxelData`'s datatype and rank, then the metadata group. `VoxelType` is
/// never opened.
fn read_header(file: &H5File) -> Result<Header> {
    let (dimension, direction) = read_directions(file, &instance_dataset(DIRECTIONS))?;

    let origin = read_double_vector(file, &instance_dataset(ORIGIN), ORIGIN, dimension)?;
    let spacing = read_double_vector(file, &instance_dataset(SPACING), SPACING, dimension)?;
    let size = read_dimensions(file, &instance_dataset(DIMENSION), dimension)?;

    let voxels = file.dataset(&instance_dataset(VOXEL_DATA))?;
    let component = datatype_to_component(&voxels.datatype()?, VOXEL_DATA)?;

    // "if this isn't a scalar image, deduce the # of components by comparing
    // the size of the Directions matrix with the reported # of dimensions in
    // the voxel dataset" (`:754-766`). Upstream reads a hyperslab shaped from
    // `m_Dimensions` whatever `VoxelData`'s own shape is; a shape that cannot
    // hold the image is rejected here instead.
    let voxel_shape = voxels.shape();
    let number_of_components = match voxel_shape.len().checked_sub(dimension) {
        Some(0) => 1,
        Some(1) => voxel_shape[dimension],
        _ => {
            return Err(IoError::MalformedHdf5Image(format!(
                "{VOXEL_DATA} has rank {} for a {dimension}-dimensional image",
                voxel_shape.len()
            )));
        }
    };
    // ITK dimensions are fastest-moving first, the HDF5 dataspace slowest first.
    if !voxel_shape[..dimension].iter().eq(size.iter().rev()) {
        return Err(IoError::MalformedHdf5Image(format!(
            "{VOXEL_DATA} has shape {voxel_shape:?} for an image of size {size:?}"
        )));
    }

    let metadata = read_metadata(file)?;
    Ok(Header {
        size,
        spacing,
        origin,
        direction,
        component,
        number_of_components,
        metadata,
    })
}

/// `ReadImageInformation`'s metadata loop (`:768-923`).
///
/// The group must exist — upstream's `openGroup` throws otherwise — and every
/// link under it is opened as a dataset, so a *subgroup* there makes
/// `openDataSet` throw. Rank-2-and-up datasets are skipped ("ignore > 1D
/// metadata"), and a datatype that is neither one of the fourteen `NATIVE_*`
/// types nor a variable-length string is dropped without a word: the final
/// `else` tests only `strType` and has no `else` of its own.
fn read_metadata(file: &H5File) -> Result<BTreeMap<String, String>> {
    let group = file
        .root_group()
        .group(IMAGE_GROUP)?
        .group(INSTANCE)?
        .group(META_DATA)?;
    if let Some(subgroup) = group.group_names()?.first() {
        return Err(IoError::MalformedHdf5Image(format!(
            "{META_DATA}/{subgroup} is a group, not a dataset"
        )));
    }

    let mut metadata = BTreeMap::new();
    for name in group.dataset_names()? {
        let dataset = file.dataset(&format!("{}/{META_DATA}/{name}", instance_path()))?;
        if dataset.shape().len() != 1 {
            continue;
        }
        if let Some(value) = metadata_value(&dataset, &name)? {
            metadata.insert(name, value);
        }
    }
    Ok(metadata)
}

/// One `MetaData` dataset, stringified the way SimpleITK stringifies the
/// `itk::MetaDataObject` that `EncapsulateMetaData` built from it.
///
/// `GetMetaDataDictionaryCustomCast::CustomCast`
/// (`sitkMetaDataDictionaryCustomCast.hxx:59-73`) returns an
/// `std::string` entry verbatim and otherwise streams the value through
/// `MetaDataObject<T>::Print`, which is `os << value`
/// (`itkMetaDataObject.hxx:65-82`). So an `int` comes back as `"42"`, a
/// `double` through `std::ostream`'s default `%g` precision of 6, an
/// `itk::Array<T>` as its elements joined by a single space
/// (`vnl_vector.hxx:829-836`), and a `signed char` / `unsigned char` as the
/// **character** that byte spells, not as a number.
///
/// The `is*` attributes exist because "HDF5 can't distinguish between long and
/// int datasets in a disk file" (`:307-310`): `WriteScalar(long)` narrows to
/// `int` and tags the dataset `isLong`, `WriteScalar(unsigned long)` narrows to
/// `unsigned int` and tags it `isUnsignedLong`, and so on. Widening back to the
/// tagged type changes no decimal digit, so only `isBool` — which turns a
/// `uint8` into `"1"` / `"0"` — and `isUnsignedLong` on a *signed* 32-bit
/// dataset change the string this returns.
fn metadata_value(dataset: &H5Dataset, name: &str) -> Result<Option<String>> {
    let datatype = dataset.datatype()?;
    if let DatatypeMessage::VarLenString { .. } = datatype {
        return Ok(dataset.read_vlen_strings()?.into_iter().next());
    }
    // Anything that is not one of the `NATIVE_*` types falls off the end of the
    // `if`/`else if` chain and never reaches the dictionary.
    let Ok(component) = datatype_to_component(&datatype, name) else {
        return Ok(None);
    };
    let attrs = dataset.attr_names()?;
    let has = |attr: &str| attrs.iter().any(|a| a == attr);
    let elements = dataset.shape()[0];

    // `ReadScalar` throws "Elements > 1 for scalar type in HDF5 File" for a
    // tagged dataset that holds more than one element; every `is*` branch calls
    // it. `EncapsulateMetaData<bool>` is likewise only reachable through it.
    let scalar_only = |attr: &str| -> Result<()> {
        if elements != 1 {
            return Err(IoError::MalformedHdf5Image(format!(
                "Elements > 1 for scalar type in HDF5 File ({name} is tagged {attr})"
            )));
        }
        Ok(())
    };

    let value = match component {
        PixelId::UInt8 if has("isBool") => {
            scalar_only("isBool")?;
            // `bool val = tmpVal != 0` then `os << val`.
            u8::from(dataset.read_raw::<u8>()?[0] != 0).to_string()
        }
        PixelId::UInt8 => join_chars(dataset.read_raw::<u8>()?.into_iter()),
        PixelId::Int8 => join_chars(dataset.read_raw::<i8>()?.into_iter().map(|v| v as u8)),
        PixelId::UInt16 => join_decimal(&dataset.read_raw::<u16>()?),
        PixelId::Int16 => join_decimal(&dataset.read_raw::<i16>()?),
        // `WriteScalar(unsigned long)` writes `NATIVE_UINT`, so ITK never tags a
        // *signed* dataset `isUnsignedLong`; the branch that reads one back
        // (`:801-805`) is dead for ITK-written files (ledger §2.132). libhdf5's
        // int32 -> uint64 conversion sends a negative source to 0.
        PixelId::Int32 if has("isUnsignedLong") => {
            scalar_only("isUnsignedLong")?;
            u64::try_from(dataset.read_raw::<i32>()?[0])
                .unwrap_or(0)
                .to_string()
        }
        PixelId::UInt32 => join_decimal(&dataset.read_raw::<u32>()?),
        PixelId::Int32 => join_decimal(&dataset.read_raw::<i32>()?),
        PixelId::UInt64 => join_decimal(&dataset.read_raw::<u64>()?),
        PixelId::Int64 => join_decimal(&dataset.read_raw::<i64>()?),
        PixelId::Float32 => join_g(dataset.read_raw::<f32>()?.into_iter().map(f64::from)),
        PixelId::Float64 => join_g(dataset.read_raw::<f64>()?.into_iter()),
        other => unreachable!("{other:?} is not a component type"),
    };
    Ok(Some(value))
}

/// `os << v` for an integer, or `os << itk::Array<T>` — a single space between
/// elements and none at the end (`vnl_vector.hxx:829-836`).
fn join_decimal<T: Display>(values: &[T]) -> String {
    values
        .iter()
        .map(T::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

/// `os << (signed char)65` writes `A`, not `65`: `std::ostream`'s `char`
/// overloads take precedence for both `signed char` and `unsigned char`.
///
/// A byte above 127 becomes the Latin-1 code point of that value here, because
/// this crate's dictionary holds `String`; SimpleITK's `std::string` holds the
/// raw byte (ledger §4.90).
fn join_chars(bytes: impl Iterator<Item = u8>) -> String {
    let mut out = String::new();
    for (i, byte) in bytes.enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push(char::from(byte));
    }
    out
}

/// `os << v` for a `float` or a `double`: `std::ostream`'s default floating
/// format is `%g` at precision 6, applied to the `double`-promoted value.
fn join_g(values: impl Iterator<Item = f64>) -> String {
    values
        .map(|v| {
            if v.is_nan() {
                "nan".to_string()
            } else if v.is_infinite() {
                if v < 0.0 { "-inf" } else { "inf" }.to_string()
            } else {
                c_format_g(v, 6)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `ImageFileReader::ReadImageInformation` over `HDF5ImageIO`.
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    let file = H5File::open(path)?;
    let header = read_header(&file)?;
    let pixel_id = pixel_id_from_image_io(header.component, header.number_of_components);
    Ok(ImageInformation {
        pixel_id,
        dimension: header.dimension(),
        number_of_components: header.number_of_components,
        size: header.size,
        spacing: header.spacing,
        origin: header.origin,
        direction: header.direction,
        metadata: header.metadata,
    })
}

/// `HDF5ImageIO::Read` (`:999-1012`) over a hyperslab that, for a
/// non-streaming reader, is the whole `VoxelData` dataset. The memory type is
/// the dataset's own, so no conversion happens on the way out.
pub fn read(path: &Path) -> Result<Image> {
    let file = H5File::open(path)?;
    let header = read_header(&file)?;
    let pixel_id = pixel_id_from_image_io(header.component, header.number_of_components);

    let dataset = file.dataset(&instance_dataset(VOXEL_DATA))?;
    let buffer = read_buffer(&dataset, header.component)?;
    let mut image = if pixel_id.is_vector() {
        // The interleaved `VoxelData` — `[..reverse(size), components]`
        // row-major, component fastest — is exactly SimpleITK's own vector
        // buffer order, so the raw bytes go straight in.
        Image::from_parts_vector(
            buffer,
            header.number_of_components,
            header.size,
            header.spacing,
            header.origin,
            header.direction,
        )?
    } else {
        Image::from_parts(
            buffer,
            header.size,
            header.spacing,
            header.origin,
            header.direction,
        )?
    };
    for (key, value) in &header.metadata {
        image.set_meta_data(key, value);
    }
    Ok(image)
}

/// The `VoxelData` bytes, reinterpreted as the component type
/// [`datatype_to_component`] already validated.
fn read_buffer(dataset: &H5Dataset, component: PixelId) -> Result<PixelBuffer> {
    Ok(match component {
        PixelId::UInt8 => PixelBuffer::UInt8(dataset.read_raw()?),
        PixelId::Int8 => PixelBuffer::Int8(dataset.read_raw()?),
        PixelId::UInt16 => PixelBuffer::UInt16(dataset.read_raw()?),
        PixelId::Int16 => PixelBuffer::Int16(dataset.read_raw()?),
        PixelId::UInt32 => PixelBuffer::UInt32(dataset.read_raw()?),
        PixelId::Int32 => PixelBuffer::Int32(dataset.read_raw()?),
        PixelId::UInt64 => PixelBuffer::UInt64(dataset.read_raw()?),
        PixelId::Int64 => PixelBuffer::Int64(dataset.read_raw()?),
        PixelId::Float32 => PixelBuffer::Float32(dataset.read_raw()?),
        PixelId::Float64 => PixelBuffer::Float64(dataset.read_raw()?),
        other => unreachable!("{other:?} is not a component type"),
    })
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// `HDF5ImageIO::WriteImageInformation` (`:1114-1382`) followed by `Write`
/// (`:1387-1442`), which together create every dataset in the order listed in
/// this module's docs and then fill `VoxelData`.
pub fn write(image: &Image, path: &Path, options: &WriteOptions) -> Result<()> {
    let dimension = image.dimension();
    let components = image.buffer_stride();
    // §3.48: `use_compression` gates the deflate filter. Upstream deflates
    // unconditionally (`setDeflate` with no `GetUseCompression()` guard); this
    // port honours the flag. Chunking stays unconditional, matching ITK's own
    // `plist.setChunk` — an uncompressed write is still a chunked dataset, just
    // without the filter. The level is meaningful only when compression is on.
    let deflate = options
        .use_compression
        .then(|| options.resolved_level(DEFAULT_COMPRESSION_LEVEL) as u32);

    let file = H5File::create(path)?;
    file.write_vlen_strings("ITKVersion", &[ITK_VERSION])?;
    file.write_vlen_strings("HDFVersion", &[HDF_VERSION])?;
    let instance = file.create_group(IMAGE_GROUP)?.create_group(INSTANCE)?;

    write_doubles(&instance, ORIGIN, image.origin())?;
    write_directions(&instance, image.direction(), dimension)?;
    write_doubles(&instance, SPACING, image.spacing())?;
    instance
        .new_dataset::<u64>()
        .shape([dimension])
        .create(DIMENSION)?
        .write_raw(&image.size().iter().map(|&s| s as u64).collect::<Vec<_>>())?;
    let component = image.buffer().component_id();
    instance.write_vlen_strings(VOXEL_TYPE, &[component_to_string(component)])?;

    // "HDF5 dimensions listed slowest moving first, ITK are fastest moving
    // first", then the intra-voxel index as the fastest axis of all.
    let mut dims: Vec<usize> = image.size().iter().rev().copied().collect();
    if components > 1 {
        dims.push(components);
    }
    // "set the chunk size to be the N-1 dimension region" — `dims[0] = 1`.
    let mut chunk = dims.clone();
    chunk[0] = 1;
    write_voxels(&instance, image.buffer(), &dims, &chunk, deflate)?;

    let metadata = instance.create_group(META_DATA)?;
    for key in image.meta_data_keys() {
        let value = image.meta_data(key).expect("key came from meta_data_keys");
        metadata.write_vlen_strings(key, &[value])?;
    }
    file.close()?;
    Ok(())
}

/// `WriteVector<double>` (`:456-466`): a contiguous rank-1 `NATIVE_DOUBLE`
/// dataset.
fn write_doubles(group: &H5Group, name: &str, values: &[f64]) -> Result<()> {
    group
        .new_dataset::<f64>()
        .shape([values.len()])
        .create(name)?
        .write_raw(values)?;
    Ok(())
}

/// `WriteDirections` (`:489-510`): row `i` is `m_Direction[i]`, which is column
/// `i` of the row-major `direction`.
fn write_directions(group: &H5Group, direction: &[f64], dimension: usize) -> Result<()> {
    let mut transposed = vec![0.0; dimension * dimension];
    for i in 0..dimension {
        for j in 0..dimension {
            transposed[i * dimension + j] = direction[j * dimension + i];
        }
    }
    group
        .new_dataset::<f64>()
        .shape([dimension, dimension])
        .create(DIRECTIONS)?
        .write_raw(&transposed)?;
    Ok(())
}

/// The `VoxelData` dataset: chunked at `[1, ...]` always, and deflated at
/// `deflate` when `Some` — i.e. only when `use_compression` asked for it
/// (§3.48).
///
/// The buffer is written verbatim. ITK's dataspace is the reverse of the image
/// size with the component axis appended, which is exactly the order of
/// SimpleITK's interleaved buffer.
fn write_voxels(
    group: &H5Group,
    buffer: &PixelBuffer,
    dims: &[usize],
    chunk: &[usize],
    deflate: Option<u32>,
) -> Result<()> {
    macro_rules! write {
        ($ty:ty, $values:expr) => {{
            let mut builder = group.new_dataset::<$ty>().shape(dims).chunk(chunk);
            if let Some(level) = deflate {
                builder = builder.deflate(level);
            }
            builder.create(VOXEL_DATA)?.write_raw($values)?;
        }};
    }
    match buffer {
        PixelBuffer::UInt8(v) => write!(u8, v),
        PixelBuffer::Int8(v) => write!(i8, v),
        PixelBuffer::UInt16(v) => write!(u16, v),
        PixelBuffer::Int16(v) => write!(i16, v),
        PixelBuffer::UInt32(v) => write!(u32, v),
        PixelBuffer::Int32(v) => write!(i32, v),
        PixelBuffer::UInt64(v) => write!(u64, v),
        PixelBuffer::Int64(v) => write!(i64, v),
        PixelBuffer::Float32(v) => write!(f32, v),
        PixelBuffer::Float64(v) => write!(f64, v),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Registry entry
// ---------------------------------------------------------------------------

/// `itk::HDF5ImageIO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hdf5ImageIo;

impl ImageIo for Hdf5ImageIo {
    fn name(&self) -> &'static str {
        "HDF5ImageIO"
    }

    fn supported_read_extensions(&self) -> &'static [&'static str] {
        EXTENSIONS
    }

    fn supported_write_extensions(&self) -> &'static [&'static str] {
        EXTENSIONS
    }

    fn can_read_file(&self, path: &Path) -> bool {
        can_read_file(path)
    }

    fn read_information(&self, path: &Path) -> Result<ImageInformation> {
        read_information(path)
    }

    fn read(&self, path: &Path) -> Result<Image> {
        read(path)
    }

    /// `options.use_compression` gates the deflate filter (§3.48 — upstream
    /// deflates unconditionally); when on, `options.compression_level` passes
    /// through `itkSetClampMacro(CompressionLevel, int, 1, 9)`.
    fn write(&self, image: &Image, path: &Path, options: &WriteOptions) -> Result<()> {
        write(image, path, options)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::image_io::{FileMode, create_image_io};
    use crate::{read_image, write_image};

    /// A path no other test in this crate shares. `H5File::create` truncates,
    /// so no test needs to remove its file first.
    fn tmp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("sitk_rs_image_hdf5_{name}"));
        path
    }

    /// The ten component types `ComponentToString` accepts, each with a value
    /// that only that width can hold.
    fn every_scalar_buffer(pixels: usize) -> Vec<PixelBuffer> {
        let n = pixels as i64;
        vec![
            PixelBuffer::UInt8((0..pixels).map(|i| (i as u8).wrapping_add(200)).collect()),
            PixelBuffer::Int8((0..pixels).map(|i| (i as i8).wrapping_sub(100)).collect()),
            PixelBuffer::UInt16((0..pixels).map(|i| i as u16 + 60_000).collect()),
            PixelBuffer::Int16((0..pixels).map(|i| i as i16 - 30_000).collect()),
            PixelBuffer::UInt32((0..pixels).map(|i| i as u32 + 4_000_000_000).collect()),
            PixelBuffer::Int32((0..pixels).map(|i| i as i32 - 2_000_000_000).collect()),
            PixelBuffer::UInt64((0..pixels).map(|i| i as u64 + (1 << 63)).collect()),
            PixelBuffer::Int64((0..n).map(|i| i - (1 << 62)).collect()),
            PixelBuffer::Float32((0..pixels).map(|i| i as f32 * 0.5 - 1.0).collect()),
            PixelBuffer::Float64((0..pixels).map(|i| i as f64 * 0.25 - 1.0).collect()),
        ]
    }

    fn image_2d(buffer: PixelBuffer) -> Image {
        Image::from_parts(
            buffer,
            vec![4, 3],
            vec![0.5, 1.5],
            vec![-1.0, 2.0],
            vec![0.0, -1.0, 1.0, 0.0],
        )
        .unwrap()
    }

    fn image_3d(buffer: PixelBuffer) -> Image {
        Image::from_parts(
            buffer,
            vec![4, 3, 2],
            vec![0.5, 1.5, 2.5],
            vec![-1.0, 2.0, 3.0],
            vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap()
    }

    /// Lay down `/ITKImage/0` for a 2x1 `uint8` image whose voxels are `[1, 2]`,
    /// omitting each dataset or group named in `skip`. The caller writes its own
    /// version of whatever it skipped, then closes the file.
    ///
    /// The fixtures are built from scratch rather than by editing a file [`write`]
    /// produced, because `rust-hdf5`'s append mode rebuilds its group table from
    /// the file's *datasets* and so loses the empty `MetaData` group.
    fn fixture(path: &Path, skip: &[&str]) -> (H5File, H5Group) {
        let file = H5File::create(path).unwrap();
        let instance = file
            .create_group(IMAGE_GROUP)
            .unwrap()
            .create_group(INSTANCE)
            .unwrap();
        let wanted = |name: &str| !skip.contains(&name);
        if wanted(ORIGIN) {
            write_doubles(&instance, ORIGIN, &[0.0, 0.0]).unwrap();
        }
        if wanted(DIRECTIONS) {
            write_directions(&instance, &[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        }
        if wanted(SPACING) {
            write_doubles(&instance, SPACING, &[1.0, 1.0]).unwrap();
        }
        if wanted(DIMENSION) {
            instance
                .new_dataset::<u64>()
                .shape([2usize])
                .create(DIMENSION)
                .unwrap()
                .write_raw(&[2u64, 1])
                .unwrap();
        }
        if wanted(VOXEL_TYPE) {
            instance.write_vlen_strings(VOXEL_TYPE, &["UCHAR"]).unwrap();
        }
        if wanted(VOXEL_DATA) {
            instance
                .new_dataset::<u8>()
                .shape([1usize, 2])
                .create(VOXEL_DATA)
                .unwrap()
                .write_raw(&[1u8, 2])
                .unwrap();
        }
        if wanted(META_DATA) {
            instance.create_group(META_DATA).unwrap();
        }
        (file, instance)
    }

    /// The image [`fixture`] describes.
    fn fixture_image() -> Image {
        Image::from_parts(
            PixelBuffer::UInt8(vec![1, 2]),
            vec![2, 1],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap()
    }

    // -- layout -------------------------------------------------------------

    /// Every group, dataset, shape and datatype `WriteImageInformation`
    /// creates, in the order it creates them.
    #[test]
    fn the_on_disk_layout_matches_itk_hdf5_image_io() {
        let path = tmp_path("layout.h5");
        let mut image = image_2d(PixelBuffer::Int16((0..12).map(|i| i as i16).collect()));
        image.set_meta_data("ITK_InputFilterName", "HDF5ImageIO");
        write(&image, &path, &WriteOptions::default()).unwrap();

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset_names(),
            vec![
                "ITKVersion".to_string(),
                "HDFVersion".to_string(),
                "ITKImage/0/Origin".to_string(),
                "ITKImage/0/Directions".to_string(),
                "ITKImage/0/Spacing".to_string(),
                "ITKImage/0/Dimension".to_string(),
                "ITKImage/0/VoxelType".to_string(),
                "ITKImage/0/VoxelData".to_string(),
                "ITKImage/0/MetaData/ITK_InputFilterName".to_string(),
            ]
        );
        assert_eq!(file.root_group().group_names().unwrap(), ["ITKImage"]);
        let instance = file.root_group().group("ITKImage").unwrap();
        assert_eq!(instance.group_names().unwrap(), ["0"]);
        assert_eq!(
            instance.group("0").unwrap().group_names().unwrap(),
            ["MetaData"]
        );

        assert_eq!(
            file.dataset("ITKVersion")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["6.0.0"]
        );
        assert_eq!(
            file.dataset("ITKImage/0/VoxelType")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["SHORT"]
        );
        assert_eq!(
            file.dataset("ITKImage/0/MetaData/ITK_InputFilterName")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["HDF5ImageIO"]
        );

        // Origin, Spacing: rank-1 NATIVE_DOUBLE, ITK axis order.
        for (name, expected) in [
            ("ITKImage/0/Origin", [-1.0, 2.0]),
            ("ITKImage/0/Spacing", [0.5, 1.5]),
        ] {
            let dataset = file.dataset(name).unwrap();
            assert_eq!(dataset.shape(), [2], "{name} shape");
            assert!(
                matches!(
                    dataset.datatype().unwrap(),
                    DatatypeMessage::FloatingPoint {
                        size: 8,
                        byte_order: ByteOrder::LittleEndian,
                        ..
                    }
                ),
                "{name} datatype"
            );
            assert_eq!(dataset.read_raw::<f64>().unwrap(), expected, "{name}");
        }

        // Dimension: rank-1 NATIVE_ULONG, ITK axis order (fastest first).
        let dimension = file.dataset("ITKImage/0/Dimension").unwrap();
        assert_eq!(dimension.shape(), [2]);
        assert!(matches!(
            dimension.datatype().unwrap(),
            DatatypeMessage::FixedPoint {
                size: 8,
                signed: false,
                byte_order: ByteOrder::LittleEndian,
                ..
            }
        ));
        assert_eq!(dimension.read_raw::<u64>().unwrap(), [4, 3]);

        // VoxelData: the reverse of the image size, chunked at [1, ...]. The
        // default `WriteOptions::use_compression` is false, so no deflate
        // filter — chunking is unconditional either way (§3.48).
        let voxels = file.dataset("ITKImage/0/VoxelData").unwrap();
        assert_eq!(voxels.shape(), [3, 4]);
        assert_eq!(voxels.chunk_dims(), Some(vec![1, 4]));
        assert!(voxels.is_chunked());
        assert!(matches!(
            voxels.datatype().unwrap(),
            DatatypeMessage::FixedPoint {
                size: 2,
                signed: true,
                byte_order: ByteOrder::LittleEndian,
                ..
            }
        ));
        assert_eq!(
            voxels.read_raw::<i16>().unwrap(),
            (0..12).collect::<Vec<_>>()
        );
    }

    /// `Directions` row `i` is direction cosine vector `i`, i.e. **column** `i`
    /// of the row-major matrix `Image::direction` returns.
    #[test]
    fn the_directions_dataset_is_the_transpose_of_the_direction_matrix() {
        let path = tmp_path("directions.h5");
        // A matrix whose transpose differs from itself in every off-diagonal.
        let direction = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut image = image_3d(PixelBuffer::UInt8(vec![0; 24]));
        image.set_direction(&direction).unwrap();
        write(&image, &path, &WriteOptions::default()).unwrap();

        let file = H5File::open(&path).unwrap();
        let dataset = file.dataset("ITKImage/0/Directions").unwrap();
        assert_eq!(dataset.shape(), [3, 3]);
        assert_eq!(
            dataset.read_raw::<f64>().unwrap(),
            // transpose(direction)
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        );
        drop(file);

        assert_eq!(read(&path).unwrap().direction(), direction);
    }

    /// A 3-D image's `Dimension` is `[x, y, z]` and its `VoxelData` is
    /// `[z, y, x]`; the buffer itself is written verbatim.
    #[test]
    fn dimension_is_itk_order_while_voxel_data_is_reversed() {
        let path = tmp_path("axis_order.h5");
        let buffer: Vec<u8> = (0..24).collect();
        write(
            &image_3d(PixelBuffer::UInt8(buffer.clone())),
            &path,
            &WriteOptions::default(),
        )
        .unwrap();

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("ITKImage/0/Dimension")
                .unwrap()
                .read_raw::<u64>()
                .unwrap(),
            [4, 3, 2]
        );
        let voxels = file.dataset("ITKImage/0/VoxelData").unwrap();
        assert_eq!(voxels.shape(), [2, 3, 4]);
        assert_eq!(voxels.chunk_dims(), Some(vec![1, 3, 4]));
        assert_eq!(voxels.read_raw::<u8>().unwrap(), buffer);
    }

    // -- round trips --------------------------------------------------------

    #[test]
    fn every_scalar_pixel_type_round_trips_in_2d() {
        for buffer in every_scalar_buffer(12) {
            let image = image_2d(buffer);
            let name = format!("roundtrip_2d_{:?}.h5", image.pixel_id());
            let path = tmp_path(&name);
            write(&image, &path, &WriteOptions::default()).unwrap();
            assert_eq!(read(&path).unwrap(), image, "{name}");
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn every_scalar_pixel_type_round_trips_in_3d() {
        for buffer in every_scalar_buffer(24) {
            let image = image_3d(buffer);
            let name = format!("roundtrip_3d_{:?}.h5", image.pixel_id());
            let path = tmp_path(&name);
            write(&image, &path, &WriteOptions::default()).unwrap();
            assert_eq!(read(&path).unwrap(), image, "{name}");
            let _ = std::fs::remove_file(&path);
        }
    }

    /// Through the registry, over every extension the constructor registers.
    #[test]
    fn every_registered_extension_round_trips_through_read_image_and_write_image() {
        for extension in EXTENSIONS {
            let path = tmp_path(&format!("dispatch{extension}"));
            let image = image_2d(PixelBuffer::Float32((0..12).map(|i| i as f32).collect()));
            write_image(&image, &path).unwrap();
            // `read_image` runs the reader's geometry normalization, which
            // records the raw spacing/direction under `ITK_original_*`; this
            // dispatch test only cares that the image itself round-trips.
            let mut back = read_image(&path).unwrap();
            back.erase_meta_data("ITK_original_spacing");
            back.erase_meta_data("ITK_original_direction");
            assert_eq!(back, image, "{extension}");
            let _ = std::fs::remove_file(&path);
        }
    }

    /// With compression on, `SetCompressionLevel` clamps to `[1, 9]` and every
    /// level reads back the same bytes.
    #[test]
    fn every_compression_level_round_trips() {
        let image = image_2d(PixelBuffer::Int32((0..12).collect()));
        for level in [-1, 0, 1, 5, 9, 100] {
            let path = tmp_path(&format!("level_{level}.h5"));
            let options = WriteOptions {
                use_compression: true,
                compression_level: level,
            };
            write(&image, &path, &options).unwrap();
            assert_eq!(read(&path).unwrap(), image, "level {level}");
            let _ = std::fs::remove_file(&path);
        }
    }

    /// The dictionary SimpleITK can put back is string-only, and it survives.
    #[test]
    fn string_meta_data_round_trips() {
        let path = tmp_path("metadata_roundtrip.h5");
        let mut image = image_2d(PixelBuffer::UInt8(vec![7; 12]));
        image.set_meta_data("ITK_InputFilterName", "HDF5ImageIO");
        image.set_meta_data("anEmptyValue", "");
        image.set_meta_data("aUnicodeValue", "µm");
        write(&image, &path, &WriteOptions::default()).unwrap();

        let read_back = read(&path).unwrap();
        assert_eq!(
            read_back.meta_data_keys(),
            ["ITK_InputFilterName", "aUnicodeValue", "anEmptyValue"]
        );
        assert_eq!(read_back.meta_data("aUnicodeValue"), Some("µm"));
        assert_eq!(read_back.meta_data("anEmptyValue"), Some(""));

        // `read_information` reports the same dictionary.
        let information = read_information(&path).unwrap();
        assert_eq!(
            information.meta_data_keys(),
            read_back.meta_data_keys().as_slice()
        );
        assert_eq!(information.meta_data("aUnicodeValue"), Some("µm"));
        let _ = std::fs::remove_file(&path);
    }

    /// `read_information` reports the same geometry `read` does, without the
    /// pixels.
    #[test]
    fn read_information_matches_read() {
        let path = tmp_path("information.h5");
        let image = image_3d(PixelBuffer::Float64((0..24).map(f64::from).collect()));
        write(&image, &path, &WriteOptions::default()).unwrap();

        let information = read_information(&path).unwrap();
        assert_eq!(information.pixel_id, PixelId::Float64);
        assert_eq!(information.dimension, 3);
        assert_eq!(information.number_of_components, 1);
        assert_eq!(information.size, [4, 3, 2]);
        assert_eq!(information.spacing, [0.5, 1.5, 2.5]);
        assert_eq!(information.origin, [-1.0, 2.0, 3.0]);
        assert_eq!(information.direction, image.direction());
        let _ = std::fs::remove_file(&path);
    }

    // -- upstream quirks ----------------------------------------------------

    /// `ReadImageInformation` derives the component type from `VoxelData`'s own
    /// datatype and never opens `VoxelType` (ledger §2.130): a file that says
    /// `FLOAT` over `int16` voxels reads as `Int16`, and a file with no
    /// `VoxelType` dataset reads at all.
    #[test]
    fn the_voxel_type_string_is_written_but_never_read() {
        // A `VoxelType` of `FLOAT` over `uint8` voxels.
        let lying = tmp_path("lying_voxel_type.h5");
        let (file, instance) = fixture(&lying, &[VOXEL_TYPE]);
        instance.write_vlen_strings(VOXEL_TYPE, &["FLOAT"]).unwrap();
        file.close().unwrap();
        assert_eq!(read(&lying).unwrap(), fixture_image());
        let _ = std::fs::remove_file(&lying);

        // And with the dataset absent entirely.
        let absent = tmp_path("absent_voxel_type.h5");
        let (file, _) = fixture(&absent, &[VOXEL_TYPE]);
        file.close().unwrap();
        assert_eq!(read(&absent).unwrap(), fixture_image());
        let _ = std::fs::remove_file(&absent);
    }

    /// §3.48 fixed: `use_compression` gates the deflate filter. The two
    /// boundary cases of that gate — off and on — are checked side by side on
    /// the same image. Chunking is unconditional in both; only the filter (and
    /// so the file size) changes.
    ///
    /// The image is two rows of 64 KiB so that the chunk — the `N-1`
    /// dimensional slab, here one row — is large enough that deflating it
    /// dominates the per-chunk B-tree and heap overhead: with the filter off,
    /// 128 KiB of zeros must stay in the file; with it on, they must not.
    #[test]
    fn use_compression_gates_the_deflate_filter() {
        const RAW_BYTES: usize = 2 * 65536;
        let image = Image::from_parts(
            PixelBuffer::UInt8(vec![0; RAW_BYTES]),
            vec![65536, 2],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        // Off: chunked, no deflate — the zeros are stored uncompressed.
        let off = tmp_path("deflate_off.h5");
        write(
            &image,
            &off,
            &WriteOptions {
                use_compression: false,
                compression_level: -1,
            },
        )
        .unwrap();
        let file = H5File::open(&off).unwrap();
        let voxels = file.dataset("ITKImage/0/VoxelData").unwrap();
        assert!(voxels.is_chunked());
        assert_eq!(voxels.chunk_dims(), Some(vec![1, 65536]));
        drop(file);
        let off_bytes = std::fs::metadata(&off).unwrap().len() as usize;
        assert!(
            off_bytes >= RAW_BYTES,
            "uncompressed {RAW_BYTES}-byte image landed in a {off_bytes}-byte file"
        );
        assert_eq!(read(&off).unwrap(), image);
        let _ = std::fs::remove_file(&off);

        // On: chunked and deflated — the zeros compress away.
        let on = tmp_path("deflate_on.h5");
        write(
            &image,
            &on,
            &WriteOptions {
                use_compression: true,
                compression_level: -1,
            },
        )
        .unwrap();
        let file = H5File::open(&on).unwrap();
        let voxels = file.dataset("ITKImage/0/VoxelData").unwrap();
        assert!(voxels.is_chunked());
        assert_eq!(voxels.chunk_dims(), Some(vec![1, 65536]));
        drop(file);
        let on_bytes = std::fs::metadata(&on).unwrap().len() as usize;
        assert!(
            on_bytes < RAW_BYTES / 4,
            "{RAW_BYTES} bytes of zeros landed in a {on_bytes}-byte deflated file"
        );
        assert_eq!(read(&on).unwrap(), image);
        let _ = std::fs::remove_file(&on);
    }

    /// `ReadImageInformation` sets `m_NumberOfComponents` but never
    /// `m_PixelType` (ledger §3.47/§5.25): SimpleITK can never read the file it
    /// writes. This port infers the vector pixel type from the trailing
    /// component axis, so a vector image round-trips byte-for-byte.
    #[test]
    fn a_vector_image_round_trips_through_the_component_axis() {
        let path = tmp_path("vector.h5");
        let image = Image::from_parts_vector(
            PixelBuffer::UInt8((0..36).map(|i| i as u8).collect()),
            3,
            vec![4, 3],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        write(&image, &path, &WriteOptions::default()).unwrap();

        let file = H5File::open(&path).unwrap();
        let voxels = file.dataset("ITKImage/0/VoxelData").unwrap();
        assert_eq!(voxels.shape(), [3, 4, 3]);
        assert_eq!(voxels.chunk_dims(), Some(vec![1, 4, 3]));
        assert_eq!(
            file.dataset("ITKImage/0/VoxelType")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["UCHAR"]
        );
        drop(file);

        // The whole image comes back — pixel type, component count, buffer.
        let read_back = read(&path).unwrap();
        assert_eq!(read_back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(read_back.number_of_components_per_pixel(), 3);
        assert_eq!(read_back, image);

        let information = read_information(&path).unwrap();
        assert_eq!(information.pixel_id, PixelId::VectorUInt8);
        assert_eq!(information.number_of_components, 3);
        let _ = std::fs::remove_file(&path);
    }

    /// A complex image reaches the `ImageIO` as two `float` components per
    /// voxel and the file records no complex marker, so it reads back as a
    /// two-component `VectorFloat32`: the samples are preserved exactly, only
    /// the pixel-type label widens from complex to vector (ledger §3.47/§5.25).
    #[test]
    fn a_complex_image_reads_back_as_a_two_component_float_vector() {
        let path = tmp_path("complex.h5");
        let mut image = Image::new(&[4, 3], PixelId::ComplexFloat32);
        image.set_spacing(&[1.0, 1.0]).unwrap();
        // Distinct samples so a dropped or reordered component cannot pass.
        let real: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let imag: Vec<f32> = (0..12).map(|i| -(i as f32) - 0.5).collect();
        let interleaved: Vec<f32> = real.iter().zip(&imag).flat_map(|(&r, &i)| [r, i]).collect();
        *image.buffer_mut() = PixelBuffer::Float32(interleaved.clone());
        write(&image, &path, &WriteOptions::default()).unwrap();

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("ITKImage/0/VoxelData").unwrap().shape(),
            [3, 4, 2]
        );
        assert_eq!(
            file.dataset("ITKImage/0/VoxelType")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["FLOAT"]
        );
        drop(file);

        let read_back = read(&path).unwrap();
        assert_eq!(read_back.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(read_back.number_of_components_per_pixel(), 2);
        assert_eq!(read_back.size(), &[4, 3]);
        assert_eq!(read_back.buffer(), &PixelBuffer::Float32(interleaved));

        let information = read_information(&path).unwrap();
        assert_eq!(information.pixel_id, PixelId::VectorFloat32);
        assert_eq!(information.number_of_components, 2);
        let _ = std::fs::remove_file(&path);
    }

    /// `CanReadFile` never looks at the extension: an HDF5 image named `.mha`
    /// is claimed, and a `.h5` holding anything else is not.
    #[test]
    fn can_read_file_is_content_based_not_extension_based() {
        let claimed = tmp_path("content_probe.mha");
        write(
            &image_2d(PixelBuffer::UInt8(vec![0; 12])),
            &claimed,
            &WriteOptions::default(),
        )
        .unwrap();
        assert!(can_read_file(&claimed));
        assert_eq!(
            create_image_io(&claimed, FileMode::Read).map(ImageIo::name),
            Some("HDF5ImageIO")
        );
        let _ = std::fs::remove_file(&claimed);

        // A transform `.h5` has a `/TransformGroup`, not an `/ITKImage`.
        let transform = tmp_path("a_transform.h5");
        {
            let file = H5File::create(&transform).unwrap();
            file.create_group("TransformGroup").unwrap();
            file.close().unwrap();
        }
        assert!(!can_read_file(&transform));
        let _ = std::fs::remove_file(&transform);

        // Plain bytes, a truncated HDF5 superblock, and an absent file.
        let text = tmp_path("plain.h5");
        std::fs::write(&text, b"ObjectType = Image\nNDims = 2\n").unwrap();
        assert!(!can_read_file(&text));
        let _ = std::fs::remove_file(&text);

        let truncated = tmp_path("truncated.h5");
        write(
            &image_2d(PixelBuffer::UInt8(vec![0; 12])),
            &truncated,
            &WriteOptions::default(),
        )
        .unwrap();
        let bytes = std::fs::read(&truncated).unwrap();
        std::fs::write(&truncated, &bytes[..16]).unwrap();
        assert!(!can_read_file(&truncated));
        assert!(read(&truncated).is_err());
        let _ = std::fs::remove_file(&truncated);

        assert!(!can_read_file(&tmp_path("absent.h5")));
    }

    /// `h5file.exists(ImageGroup)` tests for a *link*, so a **dataset** named
    /// `/ITKImage` makes `CanReadFile` say yes and the following
    /// `ReadImageInformation` throw (ledger §2.131). The transform IO's
    /// `CanReadFile` calls `openGroup` and so declines (ledger §2.123) — the
    /// two HDF5 IOs disagree.
    #[test]
    fn an_itkimage_dataset_is_claimed_and_then_fails_to_read() {
        let path = tmp_path("group_is_dataset.h5");
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<f64>()
                .shape([1usize])
                .create("ITKImage")
                .unwrap()
                .write_raw(&[1.0f64])
                .unwrap();
            file.close().unwrap();
        }
        assert!(can_read_file(&path));
        assert!(matches!(read(&path), Err(IoError::Hdf5(_))));
        let _ = std::fs::remove_file(&path);
    }

    // -- meta-data value types ----------------------------------------------

    /// [`fixture`] with a `MetaData` group `fill` populates.
    fn write_metadata_file(path: &Path, fill: impl FnOnce(&H5Group)) {
        let (file, instance) = fixture(path, &[META_DATA]);
        fill(&instance.create_group(META_DATA).unwrap());
        file.close().unwrap();
    }

    /// Every `NATIVE_*` type `ReadImageInformation`'s chain recognises, as
    /// SimpleITK's `GetMetaDataDictionaryCustomCast` stringifies it: `os <<
    /// value` for a scalar, elements joined by one space for an `itk::Array`,
    /// and the *character* a `char`-width byte spells.
    #[test]
    fn each_meta_data_value_type_stringifies_the_way_simpleitk_prints_it() {
        let path = tmp_path("metadata_types.h5");
        write_metadata_file(&path, |md| {
            // `WriteScalar(bool)`: STD_U8LE tagged `isBool`.
            let flag = md
                .new_dataset::<u8>()
                .shape([1usize])
                .create("aBool")
                .unwrap();
            flag.write_raw(&[1u8]).unwrap();
            flag.new_attr::<u8>()
                .shape([1usize])
                .create("isBool")
                .unwrap()
                .write_numeric(&1u8)
                .unwrap();
            // `WriteScalar(long)`: NATIVE_INT tagged `isLong`.
            let long = md
                .new_dataset::<i32>()
                .shape([1usize])
                .create("aLong")
                .unwrap();
            long.write_raw(&[-7i32]).unwrap();
            long.new_attr::<u8>()
                .shape([1usize])
                .create("isLong")
                .unwrap()
                .write_numeric(&1u8)
                .unwrap();
            // `WriteScalar(unsigned long)`: NATIVE_UINT tagged `isUnsignedLong`.
            let ulong = md
                .new_dataset::<u32>()
                .shape([1usize])
                .create("aULong")
                .unwrap();
            ulong.write_raw(&[4_000_000_000u32]).unwrap();
            ulong
                .new_attr::<u8>()
                .shape([1usize])
                .create("isUnsignedLong")
                .unwrap()
                .write_numeric(&1u8)
                .unwrap();
            // `WriteScalar(long long)`: STD_I64LE tagged `isLLong`.
            let llong = md
                .new_dataset::<i64>()
                .shape([1usize])
                .create("aLLong")
                .unwrap();
            llong.write_raw(&[-(1i64 << 40)]).unwrap();
            llong
                .new_attr::<u8>()
                .shape([1usize])
                .create("isLLong")
                .unwrap()
                .write_numeric(&1u8)
                .unwrap();
            // Untagged scalars.
            md.new_dataset::<i8>()
                .shape([1usize])
                .create("aSChar")
                .unwrap()
                .write_raw(&[65i8])
                .unwrap();
            md.new_dataset::<u8>()
                .shape([1usize])
                .create("aUChar")
                .unwrap()
                .write_raw(&[66u8])
                .unwrap();
            md.new_dataset::<i16>()
                .shape([1usize])
                .create("aShort")
                .unwrap()
                .write_raw(&[-9i16])
                .unwrap();
            md.new_dataset::<u16>()
                .shape([1usize])
                .create("aUShort")
                .unwrap()
                .write_raw(&[9u16])
                .unwrap();
            md.new_dataset::<i32>()
                .shape([1usize])
                .create("anInt")
                .unwrap()
                .write_raw(&[42i32])
                .unwrap();
            md.new_dataset::<u32>()
                .shape([1usize])
                .create("aUInt")
                .unwrap()
                .write_raw(&[43u32])
                .unwrap();
            md.new_dataset::<u64>()
                .shape([1usize])
                .create("aULLong")
                .unwrap()
                .write_raw(&[u64::MAX])
                .unwrap();
            md.new_dataset::<f32>()
                .shape([1usize])
                .create("aFloat")
                .unwrap()
                .write_raw(&[0.5f32])
                .unwrap();
            // `os << double` is `%g` at precision 6: 1/3 loses its tail.
            md.new_dataset::<f64>()
                .shape([1usize])
                .create("aDouble")
                .unwrap()
                .write_raw(&[1.0f64 / 3.0])
                .unwrap();
            md.new_dataset::<f64>()
                .shape([1usize])
                .create("aBigDouble")
                .unwrap()
                .write_raw(&[1.5e21f64])
                .unwrap();
            // Arrays: `itk::Array<T>`, printed space-separated.
            md.new_dataset::<i32>()
                .shape([3usize])
                .create("anIntArray")
                .unwrap()
                .write_raw(&[1i32, -2, 3])
                .unwrap();
            md.new_dataset::<f64>()
                .shape([2usize])
                .create("aDoubleArray")
                .unwrap()
                .write_raw(&[0.25f64, -0.5])
                .unwrap();
            // `Array<signed char>` is a run of characters.
            md.new_dataset::<i8>()
                .shape([3usize])
                .create("aSCharArray")
                .unwrap()
                .write_raw(&[97i8, 98, 99])
                .unwrap();
            // A string.
            md.write_vlen_strings("aString", &["hello"]).unwrap();
            // Skipped: rank 2 ("ignore > 1D metadata").
            md.new_dataset::<i32>()
                .shape([2usize, 2])
                .create("aMatrix")
                .unwrap()
                .write_raw(&[0i32; 4])
                .unwrap();
        });

        let metadata = read_information(&path).unwrap().metadata;
        let expected: BTreeMap<String, String> = [
            ("aBool", "1"),
            ("aLong", "-7"),
            ("aULong", "4000000000"),
            ("aLLong", "-1099511627776"),
            ("aSChar", "A"),
            ("aUChar", "B"),
            ("aShort", "-9"),
            ("aUShort", "9"),
            ("anInt", "42"),
            ("aUInt", "43"),
            ("aULLong", "18446744073709551615"),
            ("aFloat", "0.5"),
            ("aDouble", "0.333333"),
            ("aBigDouble", "1.5e+21"),
            ("anIntArray", "1 -2 3"),
            ("aDoubleArray", "0.25 -0.5"),
            ("aSCharArray", "a b c"),
            ("aString", "hello"),
        ]
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
        assert_eq!(metadata, expected);
        let _ = std::fs::remove_file(&path);
    }

    /// `WriteScalar(unsigned long)` writes `NATIVE_UINT`, so ITK never tags a
    /// *signed* dataset `isUnsignedLong`; the branch reading one back
    /// (`:801-805`) is dead for ITK-written files (ledger §2.132). A
    /// hand-written file still takes it.
    #[test]
    fn the_dead_is_unsigned_long_on_int32_branch_reads_as_unsigned() {
        let path = tmp_path("dead_branch.h5");
        write_metadata_file(&path, |md| {
            let ds = md.new_dataset::<i32>().shape([1usize]).create("v").unwrap();
            ds.write_raw(&[7i32]).unwrap();
            ds.new_attr::<u8>()
                .shape([1usize])
                .create("isUnsignedLong")
                .unwrap()
                .write_numeric(&1u8)
                .unwrap();
        });
        assert_eq!(
            read_information(&path)
                .unwrap()
                .metadata
                .get("v")
                .map(String::as_str),
            Some("7")
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Every `is*` branch reads through `ReadScalar`, which throws
    /// `"Elements > 1 for scalar type in HDF5 File"` on a multi-element dataset.
    #[test]
    fn a_tagged_multi_element_meta_data_dataset_is_rejected() {
        let path = tmp_path("tagged_array.h5");
        write_metadata_file(&path, |md| {
            let ds = md.new_dataset::<u8>().shape([2usize]).create("v").unwrap();
            ds.write_raw(&[1u8, 0]).unwrap();
            ds.new_attr::<u8>()
                .shape([1usize])
                .create("isBool")
                .unwrap()
                .write_numeric(&1u8)
                .unwrap();
        });
        let Err(IoError::MalformedHdf5Image(message)) = read_information(&path) else {
            panic!("expected a malformed-image error");
        };
        assert!(
            message.starts_with("Elements > 1 for scalar type in HDF5 File"),
            "{message}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The final `else` tests only `strType` and has no `else` of its own, so a
    /// datatype in neither list — here a variable-length byte sequence — is
    /// dropped without an error.
    #[test]
    fn an_unrecognised_meta_data_datatype_is_dropped_silently() {
        let path = tmp_path("unknown_metadata_type.h5");
        write_metadata_file(&path, |md| {
            md.write_vlen_bytes("bytes", &[&[1u8, 2, 3]]).unwrap();
            md.write_vlen_strings("kept", &["yes"]).unwrap();
        });
        let metadata = read_information(&path).unwrap().metadata;
        assert_eq!(metadata.keys().collect::<Vec<_>>(), ["kept"]);
        let _ = std::fs::remove_file(&path);
    }

    /// Upstream opens every link under `MetaData` with `openDataSet`, which
    /// throws on a subgroup.
    #[test]
    fn a_subgroup_under_the_meta_data_group_is_an_error() {
        let path = tmp_path("metadata_subgroup.h5");
        write_metadata_file(&path, |md| {
            md.create_group("nested").unwrap();
        });
        assert!(matches!(
            read_information(&path),
            Err(IoError::MalformedHdf5Image(_))
        ));
        let _ = std::fs::remove_file(&path);
    }

    // -- error paths --------------------------------------------------------

    /// Each of the five datasets and the one group `ReadImageInformation`
    /// opens, omitted in turn.
    #[test]
    fn a_missing_dataset_or_group_is_an_error() {
        for absent in [
            DIRECTIONS, ORIGIN, SPACING, DIMENSION, VOXEL_DATA, META_DATA,
        ] {
            let path = tmp_path(&format!("absent_{absent}.h5"));
            let (file, _) = fixture(&path, &[absent]);
            file.close().unwrap();
            assert!(matches!(read(&path), Err(IoError::Hdf5(_))), "{absent}");
            let _ = std::fs::remove_file(&path);
        }
    }

    /// `ReadDirections` needs rank 2, and this port needs the matrix square —
    /// upstream indexes `SetDirection(i, rval[i])` for `i < dim[1]` with each
    /// `rval[i]` only `dim[0]` long (ledger §4.89).
    #[test]
    fn a_malformed_directions_dataset_is_rejected() {
        let rank_one = tmp_path("directions_rank1.h5");
        let (file, instance) = fixture(&rank_one, &[DIRECTIONS]);
        write_doubles(&instance, DIRECTIONS, &[1.0, 0.0, 0.0, 1.0]).unwrap();
        file.close().unwrap();
        let Err(IoError::MalformedHdf5Image(message)) = read_information(&rank_one) else {
            panic!("expected a malformed-image error");
        };
        assert_eq!(
            message,
            " Wrong # of dims for Image Directions in HDF5 File"
        );
        let _ = std::fs::remove_file(&rank_one);

        let oblong = tmp_path("directions_oblong.h5");
        let (file, instance) = fixture(&oblong, &[DIRECTIONS]);
        instance
            .new_dataset::<f64>()
            .shape([2usize, 3])
            .create(DIRECTIONS)
            .unwrap()
            .write_raw(&[1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0])
            .unwrap();
        file.close().unwrap();
        assert!(matches!(
            read_information(&oblong),
            Err(IoError::UnsupportedHdf5Image(_))
        ));
        let _ = std::fs::remove_file(&oblong);
    }

    /// `ReadVector`'s rank check carries the transform IO's message verbatim,
    /// naming `TransformType` in an image file (ledger §2.129).
    #[test]
    fn a_rank_two_origin_is_rejected_with_the_transform_ios_message() {
        let path = tmp_path("origin_rank2.h5");
        let (file, instance) = fixture(&path, &[ORIGIN]);
        instance
            .new_dataset::<f64>()
            .shape([1usize, 2])
            .create(ORIGIN)
            .unwrap()
            .write_raw(&[0.0f64, 0.0])
            .unwrap();
        file.close().unwrap();
        let Err(IoError::MalformedHdf5Image(message)) = read_information(&path) else {
            panic!("expected a malformed-image error");
        };
        assert_eq!(message, "Wrong # of dims for TransformType in HDF5 File");
        let _ = std::fs::remove_file(&path);
    }

    /// `Origin` shorter than the dimension reads past ITK's `m_Origin`; longer
    /// is simply truncated by the `SetSpacing(i, ...)` loop.
    #[test]
    fn a_short_origin_is_rejected_and_a_long_one_is_truncated() {
        for (name, origin, ok) in [
            ("short", vec![1.0], false),
            ("long", vec![1.0, 2.0, 3.0], true),
        ] {
            let path = tmp_path(&format!("origin_{name}.h5"));
            let (file, instance) = fixture(&path, &[ORIGIN]);
            write_doubles(&instance, ORIGIN, &origin).unwrap();
            file.close().unwrap();

            let result = read_information(&path);
            if ok {
                assert_eq!(result.unwrap().origin, [1.0, 2.0]);
            } else {
                assert!(matches!(result, Err(IoError::MalformedHdf5Image(_))));
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    /// A `VoxelData` whose dataspace does not match `Dimension` is rejected;
    /// upstream reads a hyperslab shaped from `Dimension` regardless.
    #[test]
    fn a_voxel_data_shape_that_contradicts_dimension_is_rejected() {
        let path = tmp_path("voxel_shape_mismatch.h5");
        let (file, instance) = fixture(&path, &[VOXEL_DATA]);
        instance
            .new_dataset::<u8>()
            .shape([2usize, 1])
            .create(VOXEL_DATA)
            .unwrap()
            .write_raw(&[1u8, 2])
            .unwrap();
        file.close().unwrap();
        let Err(IoError::MalformedHdf5Image(message)) = read_information(&path) else {
            panic!("expected a malformed-image error");
        };
        assert!(message.contains("VoxelData has shape [2, 1]"), "{message}");
        let _ = std::fs::remove_file(&path);
    }

    /// The `Dimension` dataset is the one place ITK stores the axis order
    /// unreversed, so a `VoxelData` of rank `N + 2` is rejected outright.
    #[test]
    fn a_voxel_data_of_too_high_a_rank_is_rejected() {
        let path = tmp_path("voxel_rank4.h5");
        let (file, instance) = fixture(&path, &[VOXEL_DATA]);
        instance
            .new_dataset::<u8>()
            .shape([1usize, 2, 1, 1])
            .create(VOXEL_DATA)
            .unwrap()
            .write_raw(&[1u8, 2])
            .unwrap();
        file.close().unwrap();
        let Err(IoError::MalformedHdf5Image(message)) = read_information(&path) else {
            panic!("expected a malformed-image error");
        };
        assert_eq!(message, "VoxelData has rank 4 for a 2-dimensional image");
        let _ = std::fs::remove_file(&path);
    }

    /// [`fixture`] itself: a 2x1 `uint8` image with an empty dictionary. Pins
    /// the helper the error-path tests mutate.
    #[test]
    fn the_hand_written_fixture_reads_as_a_two_by_one_uint8_image() {
        let path = tmp_path("fixture.h5");
        let (file, _) = fixture(&path, &[]);
        file.close().unwrap();
        let information = read_information(&path).unwrap();
        assert_eq!(information.pixel_id, PixelId::UInt8);
        assert_eq!(information.size, [2, 1]);
        assert!(information.metadata.is_empty());
        assert_eq!(read(&path).unwrap(), fixture_image());
        let _ = std::fs::remove_file(&path);
    }

    /// `datatype_to_component` rejects what `rust-hdf5` will not convert.
    #[test]
    fn a_big_endian_voxel_dataset_is_rejected() {
        let big_endian = DatatypeMessage::FixedPoint {
            size: 2,
            byte_order: ByteOrder::BigEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 16,
        };
        assert!(matches!(
            datatype_to_component(&big_endian, VOXEL_DATA),
            Err(IoError::UnsupportedHdf5Image(_))
        ));
    }

    /// `ComponentToString` for the ten types SimpleITK reaches. An 8-byte
    /// integer is `LONG` / `ULONG`, never `LONGLONG` / `ULONGLONG`.
    #[test]
    fn the_component_type_names_match_itk() {
        for (id, voxel_type) in [
            (PixelId::UInt8, "UCHAR"),
            (PixelId::Int8, "CHAR"),
            (PixelId::UInt16, "USHORT"),
            (PixelId::Int16, "SHORT"),
            (PixelId::UInt32, "UINT"),
            (PixelId::Int32, "INT"),
            (PixelId::UInt64, "ULONG"),
            (PixelId::Int64, "LONG"),
            (PixelId::Float32, "FLOAT"),
            (PixelId::Float64, "DOUBLE"),
        ] {
            assert_eq!(component_to_string(id), voxel_type);
        }
    }

    /// `os << double` is `%g` at precision 6, and the IEEE specials keep their
    /// C++ spellings.
    #[test]
    fn join_g_matches_the_default_ostream_double_format() {
        assert_eq!(join_g([0.0].into_iter()), "0");
        assert_eq!(join_g([1.0 / 3.0].into_iter()), "0.333333");
        assert_eq!(join_g([1.5e21].into_iter()), "1.5e+21");
        assert_eq!(join_g([1e-5].into_iter()), "1e-05");
        assert_eq!(join_g([-2.5].into_iter()), "-2.5");
        assert_eq!(join_g([f64::INFINITY].into_iter()), "inf");
        assert_eq!(join_g([f64::NEG_INFINITY].into_iter()), "-inf");
        assert_eq!(join_g([f64::NAN].into_iter()), "nan");
        assert_eq!(join_g([1.0, 2.0].into_iter()), "1 2");
    }
}
