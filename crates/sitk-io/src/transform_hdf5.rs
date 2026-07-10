//! HDF5 transform files (`.h5` / `.hdf5`) — a port of
//! `itk::HDF5TransformIOTemplate<double>`
//! (`Modules/IO/TransformHDF5/src/itkHDF5TransformIO.cxx`), reached through
//! [`crate::read_transform`] / [`crate::write_transform`], which is where
//! SimpleITK's `ReadTransform` / `WriteTransform` (`sitkTransform.cxx:668-737`)
//! end up once `itk::TransformFileReader` / `Writer` pick the IO by extension.
//!
//! The on-disk layout, verbatim from the comment above `Read()`
//! (`itkHDF5TransformIO.cxx:266-271`) and from `WriteOneTransform`:
//!
//! ```text
//! /ITKVersion                                  -- string
//! /HDFVersion                                  -- string
//! /OSName                                      -- string
//! /OSVersion                                   -- string
//! /TransformGroup                              -- group
//! /TransformGroup/N                            -- group, N = 0, 1, 2, ...
//! /TransformGroup/N/TransformType              -- 1 variable-length string
//! /TransformGroup/N/TransformFixedParameters   -- 1-D list of double
//! /TransformGroup/N/TransformParameters        -- 1-D list of double
//! ```
//!
//! `N` is the transform's index in the file, decimal, unpadded
//! (`GetTransformName`, `:436-442`). A `CompositeTransform` is written as index
//! `0` with a `TransformType` dataset and *no* parameter datasets, its queue
//! following at `1..n` — the flattening `CompositeTransformIOHelper::
//! BuildTransformList` performs (`itkCompositeTransformIOHelper.cxx:151-171`).
//! Upstream refuses a composite anywhere but index `0`.
//!
//! The reader walks `i in 0..transformGroup.getNumObjs()`, so the transform
//! count is the number of *links* directly under `/TransformGroup`, not the
//! number of subgroups (ledger §2.117).
//!
//! # Fidelity notes
//!
//! Every datatype here is `NATIVE_DOUBLE`: `WriteFixedParameters` hardcodes it,
//! and `WriteParameters` asks `GetH5TypeFromString()`, which for the `double`
//! instantiation SimpleITK uses returns the same. The reader accepts either
//! precision on either dataset — a file written by ITK's `float` IO reads back
//! here, widened (`ReadParameters`, `:159-198`).
//!
//! `GetUseCompression()` would make `WriteParameters` deflate-5 its dataset in
//! 1 MiB chunks; SimpleITK exposes no way to set it for a transform, so this
//! port always writes contiguous (ledger §3.41, §5.20).
//!
//! Upstream's own misspellings `TranformFixedParameters` / `TranformParameters`
//! are accepted on read, as `Read()` does (`:299-323`), and never written.

use std::path::Path;

use rust_hdf5::{ByteOrder, DatatypeMessage, H5File, H5Group};
use sitk_transform::{ParametricTransform, Transform};

use crate::error::{IoError, Result};
use crate::transform_io::{correct_transform_precision_type, create_transform};

/// `HDF5CommonPathNames::transformGroupName` (`itkHDF5TransformIO.cxx:415`),
/// without the leading `/` that `rust-hdf5` strips from every stored link path.
const TRANSFORM_GROUP: &str = "TransformGroup";
/// `HDF5CommonPathNames::transformTypeName` (`:416`).
const TRANSFORM_TYPE: &str = "TransformType";
/// `HDF5CommonPathNames::transformFixedName` (`:420`).
const TRANSFORM_FIXED: &str = "TransformFixedParameters";
/// `HDF5CommonPathNames::transformParamsName` (`:421`).
const TRANSFORM_PARAMS: &str = "TransformParameters";
/// `HDF5CommonPathNames::transformFixedNameMisspelled` (`:418`).
const TRANSFORM_FIXED_MISSPELLED: &str = "TranformFixedParameters";
/// `HDF5CommonPathNames::transformParamsNameMisspelled` (`:419`).
const TRANSFORM_PARAMS_MISSPELLED: &str = "TranformParameters";

/// `itk::Version::GetITKVersion()` — the ITK release this port mirrors
/// (`ITK/CMake/itkVersion.cmake:2-4`).
const ITK_VERSION: &str = "6.0.0";

/// What this port writes to `/HDFVersion` in place of libhdf5's `H5_VERS_INFO`,
/// there being no libhdf5 here (ledger §4.79). Nothing reads it.
const HDF_VERSION: &str = "rust-hdf5 library version: 0.3.2";

/// The extensions `HDF5TransformIOTemplate::CanWriteFile` claims
/// (`itkHDF5TransformIO.cxx:76-96`) — every extension Wikipedia listed for HDF,
/// plus `hd5`, including the HDF*4* ones this IO cannot actually produce
/// (ledger §2.114).
const WRITE_EXTENSIONS: [&str; 8] = ["hdf", "h4", "hdf4", "h5", "hdf5", "he4", "he5", "hd5"];

/// `HDF5TransformIOTemplate::CanWriteFile` (`itkHDF5TransformIO.cxx:76-96`):
/// extension only. HDF5 "doesn't care about extensions at all and this is just
/// by convention", says the comment above it.
pub fn can_write_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| WRITE_EXTENSIONS.contains(&ext))
}

/// `HDF5TransformIOTemplate::CanReadFile` (`itkHDF5TransformIO.cxx:37-71`):
/// the extension is never consulted. The file must be an HDF5 file
/// (`H5Fis_hdf5`) that holds a `/TransformGroup` link; every failure — a
/// missing file, a truncated superblock, a group-less HDF5 file — is swallowed
/// by the `catch (...)` and reported as "cannot read".
pub fn can_read_file(path: &Path) -> bool {
    let Ok(file) = H5File::open(path) else {
        return false;
    };
    file.root_group().group(TRANSFORM_GROUP).is_ok()
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// `HDF5TransformIOTemplate::Read` (`itkHDF5TransformIO.cxx:272-345`) — the flat
/// `ReadTransformList`, before `TransformFileReader::Update`'s composite fold.
pub(crate) fn read_transform_list(path: &Path) -> Result<Vec<Transform>> {
    let file = H5File::open(path)?;
    let group = file.root_group().group(TRANSFORM_GROUP)?;

    // `transformGroup.getNumObjs()` counts every link under the group, not just
    // the subgroups the writer put there (ledger §2.117).
    let count = group.group_names()?.len() + group.dataset_names()?.len();
    let dataset_names = file.dataset_names();
    let exists = |name: &str| dataset_names.iter().any(|n| n == name);

    let mut list = Vec::with_capacity(count);
    for index in 0..count {
        let transform_name = transform_name(index);
        // `openGroup(transformName)` — the group must be there.
        group.group(&index.to_string())?;

        let type_name = read_transform_type(&file, &format!("{transform_name}/{TRANSFORM_TYPE}"))?;
        let type_name = correct_transform_precision_type(&type_name)?;
        let mut transform = create_transform(&type_name)?;

        // "Composite transform doesn't store its own parameters".
        if !type_name.contains("CompositeTransform") {
            let fixed_name = format!("{transform_name}/{TRANSFORM_FIXED}");
            let fixed_name = if exists(&fixed_name) {
                fixed_name
            } else {
                format!("{transform_name}/{TRANSFORM_FIXED_MISSPELLED}")
            };
            let fixed = read_double_array(&file, &fixed_name)?;
            transform.set_fixed_parameters(&fixed)?;

            let params_name = format!("{transform_name}/{TRANSFORM_PARAMS}");
            let params_name = if exists(&params_name) {
                params_name
            } else {
                format!("{transform_name}/{TRANSFORM_PARAMS_MISSPELLED}")
            };
            let params = read_double_array(&file, &params_name)?;
            transform.set_parameters(&params)?;
        }
        list.push(transform);
    }
    Ok(list)
}

/// `itk::GetTransformName(int)` (`itkHDF5TransformIO.cxx:436-442`), minus the
/// leading `/`.
fn transform_name(index: usize) -> String {
    format!("{TRANSFORM_GROUP}/{index}")
}

/// The `TransformType` string dataset, read as ITK's `H5T_VARIABLE` `StrType`
/// over a one-element dataspace (`itkHDF5TransformIO.cxx:284-294`).
fn read_transform_type(file: &H5File, name: &str) -> Result<String> {
    let dataset = file.dataset(name)?;
    let strings = dataset.read_vlen_strings()?;
    strings
        .into_iter()
        .next()
        .ok_or_else(|| IoError::MalformedTransformFile(format!("Wrong # of dims for {name}")))
}

/// `ReadParameters` / `ReadFixedParameters` (`itkHDF5TransformIO.cxx:159-244`),
/// which differ only in the array they fill: reject a non-float datatype, reject
/// a rank other than 1, then read as `double` when the element is
/// `sizeof(double)` wide and as `float` otherwise.
///
/// Two deviations, both because `rust-hdf5` hands back the stored bytes rather
/// than converting them the way `H5Dread` does:
///
/// * a float element that is neither 4 nor 8 bytes wide is rejected here;
///   upstream reads it through `NATIVE_FLOAT` and lets libhdf5 convert
///   (ledger §4.77).
/// * a big-endian dataset is rejected here; upstream byte-swaps it
///   (ledger §4.78).
fn read_double_array(file: &H5File, name: &str) -> Result<Vec<f64>> {
    let dataset = file.dataset(name)?;

    // `if (Type != H5T_FLOAT) itkExceptionMacro("Wrong data type for " <<
    // DataSetName << "in HDF5 File")` — the missing space is upstream's
    // (ledger §2.115).
    let wrong_type =
        || IoError::MalformedTransformFile(format!("Wrong data type for {name}in HDF5 File"));
    let DatatypeMessage::FloatingPoint {
        size, byte_order, ..
    } = dataset.datatype()?
    else {
        return Err(wrong_type());
    };
    if byte_order != ByteOrder::LittleEndian {
        return Err(IoError::UnsupportedHdf5Transform(format!(
            "big-endian floating-point dataset {name}"
        )));
    }

    // `if (Space.getSimpleExtentNdims() != 1)` — and the message names
    // `TransformType` even when it is a parameter array that is malformed
    // (ledger §2.116).
    if dataset.shape().len() != 1 {
        return Err(IoError::MalformedTransformFile(
            "Wrong # of dims for TransformType in HDF5 File".to_string(),
        ));
    }

    match size {
        8 => Ok(dataset.read_raw::<f64>()?),
        4 => Ok(dataset
            .read_raw::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect()),
        other => Err(IoError::UnsupportedHdf5Transform(format!(
            "{other}-byte floating-point elements in dataset {name}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// `HDF5TransformIOTemplate::Write` (`itkHDF5TransformIO.cxx:373-424`) over
/// `WriteOneTransform` (`:349-370`).
///
/// The four root strings are written before `/TransformGroup`, in ITK's order.
/// `/OSName` and `/OSVersion` come from `itksys::SystemInformation`'s `uname`;
/// this port has no libc, so it writes what `std::env::consts::OS` knows and an
/// empty release (ledger §4.79). No reader consumes either.
pub(crate) fn write_transform(transform: &Transform, path: &Path) -> Result<()> {
    // `CompositeTransformIOHelper::GetTransformList`: the composite first, then
    // its queue front-to-back.
    let list: Vec<&Transform> = match transform {
        Transform::Composite(composite) => std::iter::once(transform)
            .chain(composite.transforms())
            .collect(),
        _ => vec![transform],
    };

    let file = H5File::create(path)?;
    file.write_vlen_strings("ITKVersion", &[ITK_VERSION])?;
    file.write_vlen_strings("HDFVersion", &[HDF_VERSION])?;
    file.write_vlen_strings("OSName", &[os_name()])?;
    file.write_vlen_strings("OSVersion", &[""])?;

    let group = file.create_group(TRANSFORM_GROUP)?;
    for (index, transform) in list.iter().enumerate() {
        write_one_transform(&group, index, transform)?;
    }
    file.close()?;
    Ok(())
}

/// `itksys::SystemInformation::GetOSName()` — `uname`'s `sysname`.
fn os_name() -> &'static str {
    match std::env::consts::OS {
        "linux" => "Linux",
        "macos" => "Darwin",
        "windows" => "Windows",
        other => other,
    }
}

/// `HDF5TransformIOTemplate::WriteOneTransform` (`itkHDF5TransformIO.cxx:349-370`).
fn write_one_transform(group: &H5Group, index: usize, transform: &Transform) -> Result<()> {
    let type_name = transform.itk_transform_type_name();
    let current = group.create_group(&index.to_string())?;
    current.write_vlen_strings(TRANSFORM_TYPE, &[type_name.as_str()])?;

    if type_name.contains("CompositeTransform") {
        if index != 0 {
            return Err(IoError::MalformedTransformFile(
                "Composite Transform can only be 1st transform in a file".to_string(),
            ));
        }
        return Ok(());
    }
    write_double_array(&current, TRANSFORM_FIXED, &transform.fixed_parameters())?;
    write_double_array(&current, TRANSFORM_PARAMS, &transform.parameters())?;
    Ok(())
}

/// `WriteFixedParameters` / `WriteParameters` with `GetUseCompression()` false:
/// a contiguous rank-1 `NATIVE_DOUBLE` dataset of `values.len()` elements.
///
/// A zero-length dataset — every `TranslationTransform`'s fixed parameters — is
/// created but never written to: HDF5 leaves a contiguous dataset of no elements
/// unallocated, and `rust-hdf5` refuses a write to that undefined address.
fn write_double_array(group: &H5Group, name: &str, values: &[f64]) -> Result<()> {
    let dataset = group
        .new_dataset::<f64>()
        .shape([values.len()])
        .create(name)?;
    if !values.is_empty() {
        dataset.write_raw(values)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use sitk_transform::{
        AffineTransform, BSplineTransform, ComposeScaleSkewVersor3DTransform, CompositeTransform,
        DisplacementFieldTransform, Euler2DTransform, Euler3DTransform, ScaleLogarithmicTransform,
        ScaleSkewVersor3DTransform, ScaleTransform, ScaleVersor3DTransform, Similarity2DTransform,
        Similarity3DTransform, TranslationTransform, VersorRigid3DTransform, VersorTransform,
    };

    use super::*;
    use crate::{read_transform, write_transform as dispatch_write};

    /// A path that no other test in this crate shares. Each test removes its own
    /// file; `H5File::create` truncates anyway.
    fn tmp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("sitk_rs_transform_hdf5_{name}"));
        path
    }

    /// Every concrete `Transform` variant except `Composite`, with parameters
    /// that are not the identity's, so a dropped field cannot pass.
    fn every_transform() -> Vec<Transform> {
        vec![
            TranslationTransform::new(vec![1.0, -2.5]).into(),
            ScaleTransform::new(vec![2.0, 3.0], vec![1.0, 1.0]).into(),
            ScaleLogarithmicTransform::new(vec![2.0, 3.0, 4.0], vec![1.0, 0.0, -1.0]).into(),
            Euler2DTransform::new(0.3, [1.0, 2.0], [0.5, -0.5]).into(),
            Euler3DTransform::new(0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [0.5, 0.0, -0.5]).into(),
            Similarity2DTransform::new(1.5, 0.3, [1.0, 2.0], [0.5, -0.5]).into(),
            Similarity3DTransform::new(1.5, 0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [0.5, 0.0, -0.5])
                .into(),
            VersorTransform::new(0.1, 0.2, 0.3, [0.5, 0.0, -0.5]).into(),
            VersorRigid3DTransform::new(0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [0.5, 0.0, -0.5]).into(),
            ScaleVersor3DTransform::new(
                [1.1, 1.2, 1.3],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [0.5, 0.0, -0.5],
            )
            .into(),
            ScaleSkewVersor3DTransform::new(
                [1.1, 1.2, 1.3],
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [0.5, 0.0, -0.5],
            )
            .into(),
            ComposeScaleSkewVersor3DTransform::new(
                [1.1, 1.2, 1.3],
                [0.01, 0.02, 0.03],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [0.5, 0.0, -0.5],
            )
            .into(),
            AffineTransform::new(
                2,
                vec![1.0, 0.2, -0.3, 1.1],
                vec![4.0, 5.0],
                vec![0.5, -0.5],
            )
            .into(),
            displacement_field().into(),
            bspline().into(),
        ]
    }

    fn displacement_field() -> DisplacementFieldTransform {
        let mut field = DisplacementFieldTransform::new(
            2,
            &[3, 2],
            &[1.0, 2.0],
            &[0.5, 0.25],
            &[0.0, -1.0, 1.0, 0.0],
        )
        .unwrap();
        let n = field.number_of_parameters();
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.125).collect();
        field.set_parameters(&values).unwrap();
        field
    }

    fn bspline() -> BSplineTransform {
        let mut transform =
            BSplineTransform::new(2, &[0.0, 0.0], &[4.0, 4.0], &[1.0, 0.0, 0.0, 1.0], &[2, 2])
                .unwrap();
        let n = transform.number_of_parameters();
        let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        transform.set_parameters(&values).unwrap();
        transform
    }

    fn composite() -> Transform {
        let mut composite = CompositeTransform::new(2);
        composite
            .add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        composite
            .add_transform(Euler2DTransform::new(0.25, [3.0, 4.0], [0.5, 0.5]).into())
            .unwrap();
        composite.into()
    }

    // -- CanReadFile / CanWriteFile ----------------------------------------

    #[test]
    fn can_write_file_claims_all_eight_upstream_extensions() {
        for ext in WRITE_EXTENSIONS {
            assert!(
                can_write_file(Path::new(&format!("t.{ext}"))),
                "extension {ext}"
            );
        }
        assert!(!can_write_file(Path::new("t.tfm")));
        assert!(!can_write_file(Path::new("t.txt")));
        assert!(!can_write_file(Path::new("t")));
        // Upstream compares the last extension against a lowercase table, so an
        // uppercase spelling is not claimed.
        assert!(!can_write_file(Path::new("t.H5")));
    }

    #[test]
    fn can_read_file_is_content_based_not_extension_based() {
        let path = tmp_path("content_probe.tfm");
        let transform: Transform = TranslationTransform::new(vec![1.0, 2.0]).into();
        // A `.tfm` name over HDF5 bytes: `CanReadFile` says yes.
        write_transform(&transform, &path).unwrap();
        assert!(can_read_file(&path));
        let _ = std::fs::remove_file(&path);

        // A plain text file, an absent file, and an HDF5 file with no
        // `/TransformGroup` all say no.
        let text = tmp_path("plain.h5");
        std::fs::write(&text, b"#Insight Transform File V1.0\n").unwrap();
        assert!(!can_read_file(&text));
        let _ = std::fs::remove_file(&text);

        assert!(!can_read_file(&tmp_path("absent.h5")));

        let groupless = tmp_path("groupless.h5");
        H5File::create(&groupless).unwrap().close().unwrap();
        assert!(!can_read_file(&groupless));
        let _ = std::fs::remove_file(&groupless);
    }

    // -- layout -------------------------------------------------------------

    /// The exact group / dataset / datatype layout `itkHDF5TransformIO.cxx`
    /// writes for one non-composite transform.
    #[test]
    fn the_on_disk_layout_matches_itk_hdf5_transform_io() {
        let path = tmp_path("layout.h5");
        let transform: Transform = TranslationTransform::new(vec![1.0, -2.5]).into();
        write_transform(&transform, &path).unwrap();

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset_names(),
            vec![
                "ITKVersion".to_string(),
                "HDFVersion".to_string(),
                "OSName".to_string(),
                "OSVersion".to_string(),
                "TransformGroup/0/TransformType".to_string(),
                "TransformGroup/0/TransformFixedParameters".to_string(),
                "TransformGroup/0/TransformParameters".to_string(),
            ]
        );
        assert_eq!(file.root_group().group_names().unwrap(), ["TransformGroup"]);
        assert_eq!(
            file.root_group()
                .group("TransformGroup")
                .unwrap()
                .group_names()
                .unwrap(),
            ["0"]
        );

        assert_eq!(
            file.dataset("ITKVersion")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["6.0.0"]
        );
        assert_eq!(
            file.dataset("TransformGroup/0/TransformType")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["TranslationTransform_double_2_2"]
        );

        // Both parameter datasets are rank-1 NATIVE_DOUBLE.
        for name in [
            "TransformGroup/0/TransformParameters",
            "TransformGroup/0/TransformFixedParameters",
        ] {
            let dataset = file.dataset(name).unwrap();
            assert_eq!(dataset.shape().len(), 1, "{name} rank");
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
        }
        assert_eq!(
            file.dataset("TransformGroup/0/TransformParameters")
                .unwrap()
                .read_raw::<f64>()
                .unwrap(),
            [1.0, -2.5]
        );
        // A TranslationTransform has no fixed parameters: an empty dataset, not
        // an absent one.
        assert_eq!(
            file.dataset("TransformGroup/0/TransformFixedParameters")
                .unwrap()
                .shape(),
            [0]
        );
        let _ = std::fs::remove_file(&path);
    }

    /// `WriteOneTransform` gives the composite index 0 and a `TransformType`
    /// dataset only; its queue follows at 1, 2, ....
    #[test]
    fn a_composite_is_flattened_with_no_parameter_datasets_of_its_own() {
        let path = tmp_path("composite_layout.h5");
        write_transform(&composite(), &path).unwrap();

        let file = H5File::open(&path).unwrap();
        let group = file.root_group().group("TransformGroup").unwrap();
        assert_eq!(group.group_names().unwrap(), ["0", "1", "2"]);
        assert_eq!(
            file.dataset("TransformGroup/0/TransformType")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["CompositeTransform_double_2_2"]
        );
        for absent in [
            "TransformGroup/0/TransformParameters",
            "TransformGroup/0/TransformFixedParameters",
        ] {
            assert!(
                !file.dataset_names().iter().any(|n| n == absent),
                "{absent}"
            );
        }
        assert_eq!(
            file.dataset("TransformGroup/1/TransformType")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["TranslationTransform_double_2_2"]
        );
        assert_eq!(
            file.dataset("TransformGroup/2/TransformType")
                .unwrap()
                .read_vlen_strings()
                .unwrap(),
            ["Euler2DTransform_double_2_2"]
        );
        let _ = std::fs::remove_file(&path);
    }

    // -- round trips --------------------------------------------------------

    #[test]
    fn every_transform_round_trips_through_an_h5_file() {
        for (i, transform) in every_transform().into_iter().enumerate() {
            let path = tmp_path(&format!("roundtrip_{i}.h5"));
            dispatch_write(&transform, &path).unwrap();
            let read = read_transform(&path).unwrap();
            let _ = std::fs::remove_file(&path);

            assert_eq!(
                read.itk_transform_type_name(),
                transform.itk_transform_type_name()
            );
            assert_eq!(
                read.parameters(),
                transform.parameters(),
                "parameters of {}",
                transform.itk_transform_type_name()
            );
            assert_eq!(
                read.fixed_parameters(),
                transform.fixed_parameters(),
                "fixed parameters of {}",
                transform.itk_transform_type_name()
            );
        }
    }

    #[test]
    fn a_composite_round_trips_through_an_h5_file() {
        let path = tmp_path("composite_roundtrip.hdf5");
        let transform = composite();
        dispatch_write(&transform, &path).unwrap();
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read, transform);
    }

    /// A transform written as `.tfm` and re-written as `.h5` reads back with the
    /// same type name and the same parameters, and the other way round.
    ///
    /// Parameters, not the transform itself: `ScaleLogarithmicTransform`'s
    /// parameters are `ln(scale)` and `SetParameters` takes `std::exp` of them,
    /// so `exp(ln(3.0))` lands one ulp off `3.0` and the *stored* scale differs
    /// after any parameter round trip, in ITK exactly as here.
    #[test]
    fn a_transform_crosses_between_the_two_formats() {
        for (i, transform) in every_transform().into_iter().enumerate() {
            let tfm = tmp_path(&format!("cross_{i}.tfm"));
            let h5 = tmp_path(&format!("cross_{i}.h5"));
            let name = transform.itk_transform_type_name();

            dispatch_write(&transform, &tfm).unwrap();
            let from_tfm = read_transform(&tfm).unwrap();
            dispatch_write(&from_tfm, &h5).unwrap();
            let from_h5 = read_transform(&h5).unwrap();
            assert_eq!(from_h5.itk_transform_type_name(), name);
            assert_eq!(from_h5.parameters(), transform.parameters(), "{name}");
            assert_eq!(
                from_h5.fixed_parameters(),
                transform.fixed_parameters(),
                "{name}"
            );

            // .h5 -> .tfm -> compare
            dispatch_write(&from_h5, &tfm).unwrap();
            let round = read_transform(&tfm).unwrap();
            assert_eq!(round.itk_transform_type_name(), name);
            assert_eq!(round.parameters(), transform.parameters(), "{name}");
            assert_eq!(
                round.fixed_parameters(),
                transform.fixed_parameters(),
                "{name}"
            );

            let _ = std::fs::remove_file(&tfm);
            let _ = std::fs::remove_file(&h5);
        }
    }

    #[test]
    fn a_composite_crosses_between_the_two_formats() {
        let tfm = tmp_path("cross_composite.tfm");
        let h5 = tmp_path("cross_composite.h5");
        let transform = composite();
        dispatch_write(&transform, &tfm).unwrap();
        dispatch_write(&read_transform(&tfm).unwrap(), &h5).unwrap();
        assert_eq!(read_transform(&h5).unwrap(), transform);
        let _ = std::fs::remove_file(&tfm);
        let _ = std::fs::remove_file(&h5);
    }

    // -- upstream quirks ----------------------------------------------------

    /// `Read()` falls back to `TranformFixedParameters` / `TranformParameters`
    /// — upstream's own typos, kept for files written before they were noticed.
    #[test]
    fn the_misspelled_parameter_names_are_read() {
        let path = tmp_path("misspelled.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            zero.write_vlen_strings(TRANSFORM_TYPE, &["TranslationTransform_double_2_2"])
                .unwrap();
            write_double_array(&zero, TRANSFORM_FIXED_MISSPELLED, &[]).unwrap();
            write_double_array(&zero, TRANSFORM_PARAMS_MISSPELLED, &[7.0, 8.0]).unwrap();
            file.close().unwrap();
        }
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read.parameters(), [7.0, 8.0]);
    }

    /// `ReadParameters` reads a `float` dataset through `NATIVE_FLOAT` and
    /// widens each element — how the `double` IO reads a file the `float` IO
    /// wrote. `CorrectTransformPrecisionType` rewrites the type name to match.
    #[test]
    fn a_float_precision_file_reads_as_double() {
        let path = tmp_path("float_precision.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            zero.write_vlen_strings(TRANSFORM_TYPE, &["TranslationTransform_float_2_2"])
                .unwrap();
            zero.new_dataset::<f32>()
                .shape([0usize])
                .create(TRANSFORM_FIXED)
                .unwrap();
            zero.new_dataset::<f32>()
                .shape([2usize])
                .create(TRANSFORM_PARAMS)
                .unwrap()
                .write_raw(&[0.5f32, -0.25])
                .unwrap();
            file.close().unwrap();
        }
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            read.itk_transform_type_name(),
            "TranslationTransform_double_2_2"
        );
        assert_eq!(read.parameters(), [0.5, -0.25]);
    }

    /// `WriteParameters` under `GetUseCompression()` writes `TransformParameters`
    /// as a deflate-5 chunked dataset and leaves `TransformFixedParameters`
    /// contiguous (ledger §3.41, §5.20). Nothing in SimpleITK sets the flag, but
    /// a file that carries it must still read.
    #[test]
    fn a_deflate_compressed_parameter_dataset_reads() {
        let path = tmp_path("deflated.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            zero.write_vlen_strings(TRANSFORM_TYPE, &["AffineTransform_double_2_2"])
                .unwrap();
            write_double_array(&zero, TRANSFORM_FIXED, &[0.5, -0.25]).unwrap();
            zero.new_dataset::<f64>()
                .shape([6usize])
                .chunk(&[6])
                .deflate(5)
                .create(TRANSFORM_PARAMS)
                .unwrap()
                .write_raw(&[1.0f64, 0.2, -0.3, 1.1, 4.0, 5.0])
                .unwrap();
            file.close().unwrap();
        }
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read.parameters(), [1.0, 0.2, -0.3, 1.1, 4.0, 5.0]);
        assert_eq!(read.fixed_parameters(), [0.5, -0.25]);
    }

    /// `getNumObjs()` counts every link under `/TransformGroup`, so a stray
    /// dataset there makes the reader look for one transform group too many
    /// (ledger §2.117). ITK dies inside `openGroup`; here it is a `NotFound`.
    #[test]
    fn a_stray_dataset_under_the_transform_group_inflates_the_count() {
        let path = tmp_path("stray.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            zero.write_vlen_strings(TRANSFORM_TYPE, &["TranslationTransform_double_2_2"])
                .unwrap();
            write_double_array(&zero, TRANSFORM_FIXED, &[]).unwrap();
            write_double_array(&zero, TRANSFORM_PARAMS, &[1.0, 2.0]).unwrap();
            write_double_array(&group, "stray", &[0.0]).unwrap();
            file.close().unwrap();
        }
        // Two objects counted, only group "0" exists: "1" is missing.
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::Hdf5(_))), "{result:?}");
    }

    /// Upstream refuses a composite anywhere but index 0, so a composite nested
    /// inside a composite cannot be written.
    #[test]
    fn a_composite_may_not_be_nested() {
        let mut inner = CompositeTransform::new(2);
        inner
            .add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        let mut outer = CompositeTransform::new(2);
        outer.add_transform(inner.into()).unwrap();
        let transform: Transform = outer.into();

        let path = tmp_path("nested.h5");
        let result = write_transform(&transform, &path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::MalformedTransformFile(_))));
    }

    // -- error paths --------------------------------------------------------

    #[test]
    fn a_file_without_a_transform_group_is_not_a_transform_file() {
        let path = tmp_path("no_group.h5");
        H5File::create(&path).unwrap().close().unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::NoTransformReaderFound(_))));
    }

    /// `CanReadFile` asks only whether the *link* `/TransformGroup` exists, so a
    /// **dataset** of that name makes it claim a file whose `openGroup` then
    /// throws (ledger §2.118). Here `has_group` is consulted, so the file is
    /// simply not claimed.
    #[test]
    fn a_transform_group_that_is_a_dataset_is_not_claimed() {
        let path = tmp_path("group_is_dataset.h5");
        {
            let file = H5File::create(&path).unwrap();
            write_double_array(&file.root_group(), TRANSFORM_GROUP, &[1.0]).unwrap();
            file.close().unwrap();
        }
        assert!(!can_read_file(&path));
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::NoTransformReaderFound(_))));
    }

    #[test]
    fn an_empty_transform_group_has_no_transform() {
        let path = tmp_path("empty_group.h5");
        {
            let file = H5File::create(&path).unwrap();
            file.create_group(TRANSFORM_GROUP).unwrap();
            file.close().unwrap();
        }
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::NoTransformInFile(_))));
    }

    #[test]
    fn an_unknown_transform_type_string_is_rejected() {
        let path = tmp_path("unknown_type.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            zero.write_vlen_strings(TRANSFORM_TYPE, &["NoSuchTransform_double_2_2"])
                .unwrap();
            write_double_array(&zero, TRANSFORM_FIXED, &[]).unwrap();
            write_double_array(&zero, TRANSFORM_PARAMS, &[1.0, 2.0]).unwrap();
            file.close().unwrap();
        }
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(result, Err(IoError::UnknownTransformType(name)) if name == "NoSuchTransform_double_2_2")
        );
    }

    #[test]
    fn a_parameter_count_that_does_not_match_the_transform_is_an_error() {
        let path = tmp_path("short_parameters.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            zero.write_vlen_strings(TRANSFORM_TYPE, &["TranslationTransform_double_2_2"])
                .unwrap();
            write_double_array(&zero, TRANSFORM_FIXED, &[]).unwrap();
            write_double_array(&zero, TRANSFORM_PARAMS, &[1.0]).unwrap();
            file.close().unwrap();
        }
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            result,
            Err(IoError::Transform(
                sitk_transform::TransformError::InvalidParameters {
                    got: 1,
                    expected: 2
                }
            ))
        ));
    }

    #[test]
    fn a_missing_transform_type_dataset_is_an_error() {
        let path = tmp_path("no_type.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            write_double_array(&zero, TRANSFORM_PARAMS, &[1.0, 2.0]).unwrap();
            file.close().unwrap();
        }
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::Hdf5(_))), "{result:?}");
    }

    /// `ReadParameters` rejects a non-float dataset:
    /// `"Wrong data type for <name>in HDF5 File"`, missing space and all.
    #[test]
    fn an_integer_parameter_dataset_is_rejected() {
        let path = tmp_path("int_parameters.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            zero.write_vlen_strings(TRANSFORM_TYPE, &["TranslationTransform_double_2_2"])
                .unwrap();
            zero.new_dataset::<i32>()
                .shape([2usize])
                .create(TRANSFORM_FIXED)
                .unwrap()
                .write_raw(&[0i32, 0])
                .unwrap();
            write_double_array(&zero, TRANSFORM_PARAMS, &[1.0, 2.0]).unwrap();
            file.close().unwrap();
        }
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        let Err(IoError::MalformedTransformFile(message)) = result else {
            panic!("expected a malformed-transform-file error, got {result:?}");
        };
        assert_eq!(
            message,
            "Wrong data type for TransformGroup/0/TransformFixedParametersin HDF5 File"
        );
    }

    /// `if (Space.getSimpleExtentNdims() != 1)` — and the message upstream
    /// raises names `TransformType`, whatever dataset was actually rank-2.
    #[test]
    fn a_rank_two_parameter_dataset_is_rejected() {
        let path = tmp_path("rank2.h5");
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group(TRANSFORM_GROUP).unwrap();
            let zero = group.create_group("0").unwrap();
            zero.write_vlen_strings(TRANSFORM_TYPE, &["TranslationTransform_double_2_2"])
                .unwrap();
            write_double_array(&zero, TRANSFORM_FIXED, &[]).unwrap();
            zero.new_dataset::<f64>()
                .shape([1usize, 2])
                .create(TRANSFORM_PARAMS)
                .unwrap()
                .write_raw(&[1.0f64, 2.0])
                .unwrap();
            file.close().unwrap();
        }
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        let Err(IoError::MalformedTransformFile(message)) = result else {
            panic!("expected a malformed-transform-file error, got {result:?}");
        };
        assert_eq!(message, "Wrong # of dims for TransformType in HDF5 File");
    }
}
