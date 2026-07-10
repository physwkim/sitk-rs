//! Image and transform file IO for sitk-rs.
//!
//! Every format is an [`ImageIo`] implementor sitting in one [`registry`];
//! [`ImageFileReader`] and [`ImageFileWriter`] ask the registry which IO
//! handles a path, exactly as SimpleITK's readers and writers ask
//! `itk::ImageIOFactory`. Adding PNG or DICOM later is a new module plus
//! one registry entry — no dispatch to extend. See [`image_io`] for the probe
//! order.
//!
//! [`read_image`] and [`write_image`] are the procedural shorthand SimpleITK
//! also provides (`itk::simple::ReadImage` / `WriteImage`).
//!
//! Three formats are supported, each in an uncompressed and a compressed form:
//!
//! * [`meta_image`] — MetaImage (`.mha`, `.mhd` + `.raw` / `.zraw`), ITK's
//!   native format. Round-trips every scalar and vector pixel type and the full
//!   geometry; a complex image survives as a two-channel vector image (see that
//!   module for the upstream quirk). `CompressedData = True` is read and
//!   written.
//! * [`nrrd`] — NRRD (`.nrrd` / `.nhdr`), `raw` and `gzip` encodings, both of
//!   which round-trip a complex image because the `kinds` field records the
//!   distinction. `bzip2`, `hex` and `zrl` remain unimplemented.
//! * [`nifti`] — NIfTI-1 (`.nii`, `.hdr` + `.img`, and the `.gz` spelling of
//!   each). Round-trips every scalar pixel type, vector images, and complex
//!   images as complex.
//!
//! Compression on write is opt-in through [`ImageFileWriter::set_use_compression`]
//! or [`write_image_with`] — except for NIfTI, where the `.gz` extension alone
//! decides, exactly as upstream's `nifti_is_gzfile` does. [`compression`] owns
//! every zlib and gzip stream the three formats produce or consume.
//!
//! Transforms have their own reader and writer, [`read_transform`] and
//! [`write_transform`], over three formats: the Insight legacy text format
//! (`.tfm` / `.txt`, see [`transform_io`]), HDF5 (`.h5` / `.hdf5`, see
//! [`transform_hdf5`]), and MATLAB Level-4 (`.mat`, see [`transform_matlab`]).

pub mod compression;
pub mod error;
pub mod gipl;
pub mod image_hdf5;
pub mod image_io;
pub mod jpeg;
pub mod meta_image;
pub mod nifti;
pub mod nrrd;
pub mod png;
pub mod reader;
pub mod tiff;
pub mod transform_hdf5;
pub mod transform_io;
pub mod transform_matlab;
pub mod vtk;
pub mod writer;

use std::path::Path;

pub use error::{IoError, Result};
pub use image_io::{
    FileMode, ImageInformation, ImageIo, create_image_io, image_io_by_name, registered_image_ios,
    registry,
};
pub use reader::ImageFileReader;
use sitk_core::Image;
pub use transform_io::{read_transform, write_transform};
pub use writer::{ImageFileWriter, WriteOptions};

/// Read an image, letting the [`registry`] pick the format —
/// `itk::simple::ReadImage` (sitkImageFileReader.cxx:70-78).
///
/// The returned image carries the file's meta-data dictionary, and its geometry
/// is normalized exactly as [`ImageFileReader::execute`] does — the negative
/// spacing sign-flip plus the `ITK_original_*` records
/// ([`reader::normalize_reader_geometry`], `itkImageFileReader.hxx:216-239`).
pub fn read_image<P: AsRef<Path>>(path: P) -> Result<Image> {
    let path = path.as_ref();
    let mut image = image_io::reader_for(path)?.read(path)?;
    reader::normalize_reader_geometry(&mut image)?;
    Ok(image)
}

/// Write an image, letting the [`registry`] pick the format —
/// `itk::simple::WriteImage(image, fileName)`.
///
/// Upstream's remaining two parameters default to `useCompression = false` and
/// `compressionLevel = -1` (sitkImageFileWriter.h:221); Rust has no default
/// arguments, so [`write_image_with`] takes them.
pub fn write_image<P: AsRef<Path>>(image: &Image, path: P) -> Result<()> {
    write_image_with(image, path, false, -1)
}

/// `itk::simple::WriteImage(image, fileName, useCompression, compressionLevel)`.
///
/// `use_compression` is a request: a format that cannot compress ignores it,
/// and NIfTI — which compresses on the `.gz` extension alone — ignores both.
/// `compression_level` of `-1` leaves each format on its own default (`2` for
/// MetaImage and NRRD); any other value is clamped to
/// [`compression::MIN_COMPRESSION_LEVEL`]`..=`[`compression::MAX_COMPRESSION_LEVEL`].
pub fn write_image_with<P: AsRef<Path>>(
    image: &Image,
    path: P,
    use_compression: bool,
    compression_level: i32,
) -> Result<()> {
    let path = path.as_ref();
    let options = WriteOptions {
        use_compression,
        compression_level,
    };
    image_io::writer_for(path)?.write(image, path, &options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::{Complex, Image, PixelId};

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("sitk_io_test_{}_{name}", std::process::id()));
        p
    }

    /// The dictionary `MetaImageIO::ReadImageInformation` always installs
    /// (itkMetaImageIO.cxx:270-278), which a written-then-read image therefore
    /// carries and its in-memory original does not. Strip it so the two can be
    /// compared with `assert_eq!`.
    fn without_metadata(mut img: Image) -> Image {
        for key in img
            .meta_data_keys()
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
        {
            img.erase_meta_data(&key);
        }
        img
    }

    #[test]
    fn mha_roundtrip_preserves_buffer_and_geometry() {
        let data: Vec<i16> = (0..24).map(|i| i as i16 - 5).collect();
        let mut img = Image::from_vec(&[4, 3, 2], data.clone()).unwrap();
        img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
        img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();
        img.set_direction(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();

        let path = tmp_path("roundtrip.mha");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.size(), img.size());
        assert_eq!(back.pixel_id(), PixelId::Int16);
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(back.scalar_slice::<i16>().unwrap(), data.as_slice());
        assert_eq!(without_metadata(back), img);
    }

    #[test]
    fn mha_roundtrip_all_scalar_types() {
        macro_rules! case {
            ($ty:ty, $name:expr) => {{
                let data: Vec<$ty> = (0..8u32).map(|i| i as $ty).collect();
                let img = Image::from_vec(&[4, 2], data.clone()).unwrap();
                let path = tmp_path($name);
                write_image(&img, &path).unwrap();
                let back = read_image(&path).unwrap();
                std::fs::remove_file(&path).ok();
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
    fn mhd_writes_separate_raw_and_reads_back() {
        let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let img = Image::from_vec(&[3, 2], data.clone()).unwrap();
        let path = tmp_path("pair.mhd");
        write_image(&img, &path).unwrap();
        assert!(
            path.with_file_name(format!("sitk_io_test_{}_pair.raw", std::process::id()))
                .exists()
        );
        let back = read_image(&path).unwrap();
        assert_eq!(back.scalar_slice::<f32>().unwrap(), data.as_slice());
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(
            path.with_file_name(format!("sitk_io_test_{}_pair.raw", std::process::id())),
        )
        .ok();
    }

    #[test]
    fn mha_roundtrip_vector_float32_three_components() {
        let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25 - 4.0).collect();
        let mut img = Image::from_vec_vector::<f32>(&[4, 3], 3, data.clone()).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        img.set_direction(&[0.0, 1.0, -1.0, 0.0]).unwrap();

        let path = tmp_path("vector_f32.mha");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.size(), img.size());
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(back.component_slice::<f32>().unwrap(), data.as_slice());
        assert_eq!(without_metadata(back), img);
    }

    #[test]
    fn mha_roundtrip_vector_uint8() {
        let data: Vec<u8> = (0..48u32).map(|i| (i % 256) as u8).collect();
        let mut img = Image::from_vec_vector::<u8>(&[4, 3], 4, data.clone()).unwrap();
        img.set_spacing(&[2.0, 0.25]).unwrap();
        img.set_origin(&[10.0, -5.0]).unwrap();
        img.set_direction(&[1.0, 0.0, 0.0, -1.0]).unwrap();

        let path = tmp_path("vector_u8.mha");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(back.number_of_components_per_pixel(), 4);
        assert_eq!(back.size(), img.size());
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(back.component_slice::<u8>().unwrap(), data.as_slice());
        assert_eq!(without_metadata(back), img);
    }

    /// MetaIO has no complex element type, so a complex image's
    /// `ElementNumberOfChannels = 2` is indistinguishable on read from a
    /// same-width vector image — real ITK/SimpleITK reconstruct it as
    /// `VectorFloat32`, not `ComplexFloat32` (see the `meta_image` module
    /// docs), and this pins that upstream quirk rather than treating it as a
    /// bug to paper over.
    #[test]
    fn mha_roundtrip_complex_float32_reads_back_as_vector() {
        let data: Vec<Complex<f32>> = (0..6)
            .map(|i| Complex::new(i as f32 * 1.5, -(i as f32) - 0.5))
            .collect();
        let mut img = Image::from_vec_complex::<f32>(&[3, 2], data).unwrap();
        img.set_spacing(&[1.5, 0.5]).unwrap();
        img.set_origin(&[2.0, -3.0]).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let path = tmp_path("complex_f32.mha");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(back.number_of_components_per_pixel(), 2);
        assert_eq!(back.size(), img.size());
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(
            back.component_slice::<f32>().unwrap(),
            img.component_slice::<f32>().unwrap(),
        );
    }

    /// The `f64` counterpart of
    /// [`mha_roundtrip_complex_float32_reads_back_as_vector`].
    #[test]
    fn mha_roundtrip_complex_float64_reads_back_as_vector() {
        let data: Vec<Complex<f64>> = (0..6)
            .map(|i| Complex::new(i as f64 * 1.5, -(i as f64) - 0.5))
            .collect();
        let mut img = Image::from_vec_complex::<f64>(&[3, 2], data).unwrap();
        img.set_spacing(&[0.75, 3.0]).unwrap();
        img.set_origin(&[-4.0, 6.0]).unwrap();
        img.set_direction(&[1.0, 0.0, 0.0, 1.0]).unwrap();

        let path = tmp_path("complex_f64.mha");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorFloat64);
        assert_eq!(back.number_of_components_per_pixel(), 2);
        assert_eq!(back.size(), img.size());
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(
            back.component_slice::<f64>().unwrap(),
            img.component_slice::<f64>().unwrap(),
        );
    }

    fn raw_extra_path(path: &std::path::Path, stem_suffix: &str) -> std::path::PathBuf {
        path.with_file_name(format!(
            "sitk_io_test_{}_{stem_suffix}.raw",
            std::process::id()
        ))
    }

    #[test]
    fn mhd_header_pins_element_number_of_channels_scalar() {
        let data: Vec<f32> = vec![0.0; 4];
        let img = Image::from_vec(&[2, 2], data).unwrap();
        let path = tmp_path("scalar_header.mhd");
        write_image(&img, &path).unwrap();
        let header = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(raw_extra_path(&path, "scalar_header")).ok();

        assert!(header.contains("ElementNumberOfChannels = 1\n"), "{header}");
        assert!(header.contains("ElementType = MET_FLOAT\n"), "{header}");
    }

    #[test]
    fn mhd_header_pins_element_number_of_channels_vector() {
        let img = Image::from_vec_vector::<f32>(&[2, 2], 3, vec![0.0; 12]).unwrap();
        let path = tmp_path("vector_header.mhd");
        write_image(&img, &path).unwrap();
        let header = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(raw_extra_path(&path, "vector_header")).ok();

        assert!(header.contains("ElementNumberOfChannels = 3\n"), "{header}");
        assert!(header.contains("ElementType = MET_FLOAT\n"), "{header}");
    }

    #[test]
    fn mhd_header_pins_element_number_of_channels_complex() {
        let data: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 4];
        let img = Image::from_vec_complex::<f64>(&[2, 2], data).unwrap();
        let path = tmp_path("complex_header.mhd");
        write_image(&img, &path).unwrap();
        let header = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(raw_extra_path(&path, "complex_header")).ok();

        assert!(header.contains("ElementNumberOfChannels = 2\n"), "{header}");
        assert!(header.contains("ElementType = MET_DOUBLE\n"), "{header}");
    }

    /// `ElementNumberOfChannels = 0` is meaningless for every pixel category
    /// and is rejected via [`sitk_core::Error::InvalidComponentCount`]
    /// ([`Image::from_parts_vector`]'s zero-component guard), not silently
    /// coerced to `1`.
    #[test]
    fn read_rejects_zero_channels() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             BinaryDataByteOrderMSB = False\n\
             CompressedData = False\n\
             TransformMatrix = 1 0 0 1\n\
             Offset = 0 0\n\
             ElementSpacing = 1 1\n\
             DimSize = 2 2\n\
             ElementNumberOfChannels = 0\n\
             ElementType = MET_FLOAT\n\
             ElementDataFile = LOCAL\n";
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 16]);
        let path = tmp_path("zero_channels.mha");
        std::fs::write(&path, bytes).unwrap();

        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(matches!(result, Err(IoError::Core(_))), "{result:?}");
    }

    /// A declared channel count the raw data is too short for is truncated
    /// data, not a channel-count problem: 4 pixels * 3 channels * 4 bytes = 48
    /// bytes are declared, but only 12 are present.
    #[test]
    fn read_rejects_channel_count_data_length_mismatch() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             BinaryDataByteOrderMSB = False\n\
             CompressedData = False\n\
             TransformMatrix = 1 0 0 1\n\
             Offset = 0 0\n\
             ElementSpacing = 1 1\n\
             DimSize = 2 2\n\
             ElementNumberOfChannels = 3\n\
             ElementType = MET_FLOAT\n\
             ElementDataFile = LOCAL\n";
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 12]);
        let path = tmp_path("channel_mismatch.mha");
        std::fs::write(&path, bytes).unwrap();

        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(matches!(result, Err(IoError::TruncatedData)), "{result:?}");
    }

    /// No registered `ImageIo` advertises `.bmp`, so `CreateImageIO` returns
    /// null and `ImageFileWriter::GetImageIOBase` throws "Unable to determine
    /// ImageIO writer" (sitkImageFileWriter.cxx:207-210).
    #[test]
    fn unknown_extension_errors() {
        let img = Image::new(&[2, 2], PixelId::UInt8);
        assert!(matches!(
            write_image(&img, tmp_path("x.bmp")),
            Err(IoError::NoWriterFound(_))
        ));
    }

    // ---- registry --------------------------------------------------------

    /// `ImageFileWriter::GetRegisteredImageIOs` lists `GetNameOfClass`, not
    /// extensions (sitkImageIOUtilities.cxx:59-77).
    #[test]
    fn registry_lists_each_image_io_by_class_name() {
        assert_eq!(
            registered_image_ios(),
            vec![
                "MetaImageIO",
                "NrrdImageIO",
                "NiftiImageIO",
                "GiplImageIO",
                "VTKImageIO",
                "PNGImageIO",
                "HDF5ImageIO",
                "JPEGImageIO",
                "TIFFImageIO"
            ]
        );
        assert_eq!(
            image_io_by_name("MetaImageIO").unwrap().name(),
            "MetaImageIO"
        );
        assert_eq!(
            image_io_by_name("NrrdImageIO").unwrap().name(),
            "NrrdImageIO"
        );
        assert_eq!(
            image_io_by_name("NiftiImageIO").unwrap().name(),
            "NiftiImageIO"
        );
        assert_eq!(
            image_io_by_name("GiplImageIO").unwrap().name(),
            "GiplImageIO"
        );
        assert_eq!(image_io_by_name("VTKImageIO").unwrap().name(), "VTKImageIO");
        assert_eq!(image_io_by_name("PNGImageIO").unwrap().name(), "PNGImageIO");
        assert_eq!(
            image_io_by_name("HDF5ImageIO").unwrap().name(),
            "HDF5ImageIO"
        );
        assert_eq!(
            image_io_by_name("JPEGImageIO").unwrap().name(),
            "JPEGImageIO"
        );
        assert_eq!(
            image_io_by_name("TIFFImageIO").unwrap().name(),
            "TIFFImageIO"
        );
        assert!(matches!(
            image_io_by_name("BMPImageIO"),
            Err(IoError::UnknownImageIo(name)) if name == "BMPImageIO"
        ));
    }

    /// `MetaImageIO::CanReadFile` opens the file and looks for `NDims` in the
    /// first 8000 bytes (metaImage.cxx:1201-1228). A `.mhd` extension is not
    /// enough: `CreateImageIO`'s phase 1 strikes the IO off, phase 2 finds
    /// nobody, and `GetImageIOBase` reports it cannot determine a reader.
    #[test]
    fn extension_alone_does_not_claim_a_file_for_reading() {
        let path = tmp_path("not_really.mhd");
        std::fs::write(&path, b"this is a text file, not a MetaImage\n").unwrap();

        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();

        assert!(!claimed);
        assert!(
            matches!(result, Err(IoError::NoReaderFound(_))),
            "{result:?}"
        );
    }

    /// The mirror image: `MetaImage::CanRead` rejects a name that does not end
    /// in `.mhd`/`.mha` *before* it looks at the content (metaImage.cxx:
    /// 1182-1199), so a genuine MetaImage header under a foreign name is not
    /// rescued by `CreateImageIO`'s phase 2 either. Content beats extension in
    /// the factory; it does not beat `MetaImageIO`'s own extension check.
    #[test]
    fn meta_image_content_under_a_foreign_name_is_still_not_read() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let mha = tmp_path("content_probe.mha");
        write_image(&img, &mha).unwrap();
        let foreign = tmp_path("content_probe.foo");
        std::fs::rename(&mha, &foreign).unwrap();

        let claimed = create_image_io(&foreign, FileMode::Read).is_some();
        let result = read_image(&foreign);
        std::fs::remove_file(&foreign).ok();

        assert!(!claimed);
        assert!(
            matches!(result, Err(IoError::NoReaderFound(_))),
            "{result:?}"
        );
    }

    /// `MetaImageIO::CanWriteFile` is `HasSupportedWriteExtension(name, true)` —
    /// case-**insensitive** (itkMetaImageIO.cxx:370-380) — while
    /// `MetaImage::CanRead` compares `.mha` case-**sensitively**
    /// (metaImage.cxx:1190-1194). So upstream writes `IMG.MHA` happily and then
    /// cannot read it back. Pinned, not fixed.
    #[test]
    fn uppercase_extension_is_writable_but_not_readable() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("shouty.MHA");
        write_image(&img, &path).unwrap();
        assert!(path.exists());

        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(result, Err(IoError::NoReaderFound(_))),
            "{result:?}"
        );
    }

    /// A read of a path that does not exist is reported as such before "unable
    /// to determine ImageIO reader" (sitkImageReaderBase.cxx:87-100).
    #[test]
    fn reading_a_missing_file_reports_file_not_found() {
        let result = read_image(tmp_path("does_not_exist.mha"));
        assert!(
            matches!(result, Err(IoError::FileNotFound(_))),
            "{result:?}"
        );
    }

    /// `SetImageIO` bypasses `CreateImageIO` entirely
    /// (sitkImageFileWriter.cxx:198-205), so a named IO writes any path.
    #[test]
    fn writer_set_image_io_overrides_extension_detection() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("named_io.foo");

        let mut writer = ImageFileWriter::new();
        writer.set_file_name(&path);
        assert!(matches!(
            writer.execute(&img),
            Err(IoError::NoWriterFound(_))
        ));

        writer.set_image_io(Some("MetaImageIO"));
        writer.execute(&img).unwrap();
        let written = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(written.starts_with(b"ObjectType = Image\n"));

        // A registered IO that cannot make sense of the name still runs, and
        // fails on its own terms: `nifti_find_file_extension` finds no NIfTI
        // extension in `named_io.foo`.
        writer.set_image_io(Some("NiftiImageIO"));
        assert!(matches!(
            writer.execute(&img),
            Err(IoError::NiftiWriteRejected(_))
        ));

        // NRRD, like MetaImage, writes under whatever name it is handed: the
        // attached-header form has no extension requirement of its own.
        writer.set_image_io(Some("NrrdImageIO"));
        writer.execute(&img).unwrap();
        let written = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(written.starts_with(b"NRRD"));

        // PNG, HDF5 and JPEG are now registered (unlike when this test was
        // written), so an unregistered name is needed for the negative case.
        writer.set_image_io(Some("BMPImageIO"));
        assert!(matches!(
            writer.execute(&img),
            Err(IoError::UnknownImageIo(_))
        ));
        assert_eq!(
            writer.registered_image_ios(),
            vec![
                "MetaImageIO",
                "NrrdImageIO",
                "NiftiImageIO",
                "GiplImageIO",
                "VTKImageIO",
                "PNGImageIO",
                "HDF5ImageIO",
                "JPEGImageIO",
                "TIFFImageIO"
            ]
        );
    }

    // ---- ReadImageInformation --------------------------------------------

    /// `ReadImageInformation` parses the header and stops: `ElementDataFile` is
    /// MetaIO's `terminateRead` field (metaImage.cxx:2209-2212). This header
    /// declares 10^10 doubles and carries not one byte of them, so only a
    /// reader that never touches the pixel tail can answer.
    #[test]
    fn read_image_information_does_not_load_pixels() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             BinaryDataByteOrderMSB = False\n\
             CompressedData = False\n\
             TransformMatrix = 1 0 0 1\n\
             Offset = 3 4\n\
             ElementSpacing = 0.5 2\n\
             DimSize = 100000 100000\n\
             ElementType = MET_DOUBLE\n\
             ElementDataFile = LOCAL\n";
        let path = tmp_path("huge_header.mha");
        std::fs::write(&path, header).unwrap();

        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        let info = reader.read_image_information().unwrap().clone();
        let loaded = reader.execute();
        std::fs::remove_file(&path).ok();

        assert_eq!(info.pixel_id, PixelId::Float64);
        assert_eq!(info.dimension, 2);
        assert_eq!(info.number_of_components, 1);
        assert_eq!(info.size, vec![100000, 100000]);
        assert_eq!(info.spacing, vec![0.5, 2.0]);
        assert_eq!(info.origin, vec![3.0, 4.0]);
        assert_eq!(info.direction, vec![1.0, 0.0, 0.0, 1.0]);
        assert!(matches!(loaded, Err(IoError::TruncatedData)), "{loaded:?}");
    }

    /// A `.mhd`'s `ReadImageInformation` never opens the `.raw` either.
    #[test]
    fn read_image_information_of_an_mhd_does_not_need_the_raw_file() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             CompressedData = False\n\
             DimSize = 2 2\n\
             ElementNumberOfChannels = 3\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = nowhere.raw\n";
        let path = tmp_path("no_raw.mhd");
        std::fs::write(&path, header).unwrap();

        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        let info = reader.read_image_information().unwrap().clone();
        std::fs::remove_file(&path).ok();

        assert_eq!(info.pixel_id, PixelId::VectorUInt8);
        assert_eq!(info.number_of_components, 3);
        // Absent ElementSpacing/Offset/TransformMatrix default to unit geometry.
        assert_eq!(info.spacing, vec![1.0, 1.0]);
        assert_eq!(info.origin, vec![0.0, 0.0]);
    }

    // ---- meta-data dictionary --------------------------------------------

    /// `MetaImageIO::ReadImageInformation` always installs `ITK_InputFilterName`
    /// and `Modality`, adds every unrecognized header field verbatim, and adds
    /// `ITK_VoxelUnits` / `ITK_ExperimentDate` when `DistanceUnits` /
    /// `AcquisitionDate` are present (itkMetaImageIO.cxx:270-304). Field-name
    /// matching is `strcmp`, so `elementspacing` is *not* `ElementSpacing`: it
    /// is a custom tag, and the real spacing falls back to its default.
    #[test]
    fn read_populates_the_itk_meta_data_dictionary() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             Modality = MET_MOD_CT\n\
             DistanceUnits = mm\n\
             AcquisitionDate = 2026.07.10\n\
             MyTag = some value\n\
             elementspacing = 9 9\n\
             DimSize = 2 2\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LOCAL\n";
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(&[7u8; 4]);
        let path = tmp_path("dictionary.mha");
        std::fs::write(&path, bytes).unwrap();

        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // The reader's geometry normalization adds `ITK_original_spacing` and
        // `ITK_original_direction` on top of the IO's own dictionary keys — the
        // header's spacing is the default `1 1` (positive, so no flip) and the
        // direction is the identity.
        assert_eq!(
            img.meta_data_keys(),
            vec![
                "ITK_ExperimentDate",
                "ITK_InputFilterName",
                "ITK_VoxelUnits",
                "ITK_original_direction",
                "ITK_original_spacing",
                "Modality",
                "MyTag",
                "elementspacing",
            ]
        );
        assert_eq!(img.meta_data("ITK_InputFilterName"), Some("MetaImageIO"));
        assert_eq!(img.meta_data("Modality"), Some("MET_MOD_CT"));
        assert_eq!(img.meta_data("ITK_VoxelUnits"), Some("mm"));
        assert_eq!(img.meta_data("ITK_ExperimentDate"), Some("2026.07.10"));
        assert_eq!(img.meta_data("MyTag"), Some("some value"));
        assert_eq!(img.meta_data("elementspacing"), Some("9 9"));
        assert_eq!(img.meta_data("ITK_original_spacing"), Some("1 1"));
        assert_eq!(img.meta_data("ITK_original_direction"), Some("1 0 0 1"));
        assert_eq!(img.spacing(), &[1.0, 1.0]);
    }

    /// A header with none of the optional keys still gets the two mandatory
    /// ones, and an unparsable `Modality` falls back to `MET_MOD_UNKNOWN`
    /// (metaImageUtils.cxx:28-44).
    #[test]
    fn default_dictionary_is_the_filter_name_and_unknown_modality() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("default_dict.mha");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // `ITK_original_*` ride along from the reader's geometry normalization
        // (the written image is default 2-D geometry: unit spacing, identity
        // direction).
        assert_eq!(
            back.meta_data_keys(),
            vec![
                "ITK_InputFilterName",
                "ITK_original_direction",
                "ITK_original_spacing",
                "Modality",
            ]
        );
        assert_eq!(back.meta_data("Modality"), Some("MET_MOD_UNKNOWN"));
        assert_eq!(back.meta_data("ITK_original_spacing"), Some("1 1"));
        assert_eq!(back.meta_data("ITK_original_direction"), Some("1 0 0 1"));
    }

    /// End-to-end: a MetaImage on disk carries a negative `ElementSpacing`
    /// component (MetaIO preserves the sign; upstream's own producer of this is
    /// GDCM's negative Z-spacing, `itkGDCMImageIO.cxx:703`). The reader's
    /// geometry normalization flips it positive, negates the matching direction
    /// column, and records both raw values under `ITK_original_*`
    /// (`itkImageFileReader.hxx:216-239`).
    #[test]
    fn read_image_flips_a_negative_metaimage_spacing() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             ElementSpacing = 2 -3\n\
             DimSize = 2 2\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LOCAL\n";
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(&[10u8, 20, 30, 40]);
        let path = tmp_path("negative_spacing.mha");
        std::fs::write(&path, bytes).unwrap();

        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // Spacing is positive on both axes; axis 1's direction column flipped.
        assert_eq!(img.spacing(), &[2.0, 3.0]);
        assert_eq!(img.direction(), &[1.0, 0.0, 0.0, -1.0]);
        // The buffer is untouched by the geometry flip.
        assert_eq!(img.scalar_slice::<u8>().unwrap(), &[10, 20, 30, 40]);
        // The raw on-disk geometry stays recoverable, negative sign and all.
        assert_eq!(img.meta_data("ITK_original_spacing"), Some("2 -3"));
        assert_eq!(img.meta_data("ITK_original_direction"), Some("1 0 0 1"));
    }

    // ---- header field precedence and boolean parsing ----------------------

    fn write_mha(name: &str, header: &str, data: &[u8]) -> std::path::PathBuf {
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(data);
        let path = tmp_path(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// `MetaObject::M_Read` applies `Offset`, then `Position`, then `Origin`,
    /// and `Orientation`, then `Rotation`, then `TransformMatrix`
    /// (metaObject.cxx:1653-1707) — a fixed order that ignores where the lines
    /// sit in the file. Here `Origin` and `TransformMatrix` come *first* and
    /// still win.
    #[test]
    fn alias_precedence_is_metaios_apply_order_not_file_order() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             Origin = 7 7\n\
             TransformMatrix = 0 -1 1 0\n\
             Position = 5 5\n\
             Rotation = 1 0 0 1\n\
             Offset = 1 1\n\
             DimSize = 2 2\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LOCAL\n";
        let path = write_mha("precedence.mha", header, &[0u8; 4]);
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.origin(), &[7.0, 7.0]);
        assert_eq!(img.direction(), &[0.0, -1.0, 1.0, 0.0]);
    }

    /// `BinaryDataByteOrderMSB` is applied after `ElementByteOrderMSB`
    /// (metaObject.cxx:1618-1642), so it wins regardless of file order — even
    /// when it turns big-endian *off*.
    #[test]
    fn binary_data_byte_order_msb_overrides_element_byte_order_msb() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryDataByteOrderMSB = False\n\
             ElementByteOrderMSB = True\n\
             DimSize = 2 1\n\
             ElementType = MET_SHORT\n\
             ElementDataFile = LOCAL\n";
        let path = write_mha("byte_order_precedence.mha", header, &[1, 0, 2, 0]);
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.scalar_slice::<i16>().unwrap(), &[1, 2]);
    }

    /// A MetaIO boolean is true iff its first character is `T`, `t` or `1`
    /// (metaObject.cxx:1586-1642) — `1` is not the string `"true"`, and `yes`
    /// is false.
    #[test]
    fn meta_io_booleans_read_only_the_first_character() {
        let msb = |value: &str, name: &str| {
            let header = format!(
                "ObjectType = Image\n\
                 NDims = 2\n\
                 BinaryDataByteOrderMSB = {value}\n\
                 DimSize = 2 1\n\
                 ElementType = MET_SHORT\n\
                 ElementDataFile = LOCAL\n"
            );
            let path = write_mha(name, &header, &[0x01, 0x02, 0x03, 0x04]);
            let img = read_image(&path).unwrap();
            std::fs::remove_file(&path).ok();
            img.scalar_slice::<i16>().unwrap().to_vec()
        };
        // Big-endian: 0x0102 = 258, 0x0304 = 772.
        assert_eq!(msb("True", "bool_true.mha"), vec![258, 772]);
        assert_eq!(msb("true", "bool_lower.mha"), vec![258, 772]);
        assert_eq!(msb("1", "bool_one.mha"), vec![258, 772]);
        assert_eq!(msb("TRUE", "bool_shout.mha"), vec![258, 772]);
        // Little-endian: 0x0201 = 513, 0x0403 = 1027.
        assert_eq!(msb("False", "bool_false.mha"), vec![513, 1027]);
        assert_eq!(msb("yes", "bool_yes.mha"), vec![513, 1027]);
        assert_eq!(msb("0", "bool_zero.mha"), vec![513, 1027]);
    }

    /// `MetaImageIO::Read` calls `ElementByteOrderFix`
    /// (itkMetaImageIO.cxx:348,359), so a big-endian file round-trips through
    /// `write` — which always emits little-endian — to the same values.
    #[test]
    fn msb_round_trip_recovers_every_component() {
        let values: Vec<i32> = vec![i32::MIN, -1, 0, 1, 0x0102_0304, i32::MAX];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             ElementByteOrderMSB = True\n\
             DimSize = 3 2\n\
             ElementType = MET_INT\n\
             ElementDataFile = LOCAL\n";
        let big = write_mha("msb.mha", header, &data);
        let from_big = read_image(&big).unwrap();
        assert_eq!(from_big.scalar_slice::<i32>().unwrap(), values.as_slice());

        let little = tmp_path("msb_out.mha");
        write_image(&from_big, &little).unwrap();
        let round = read_image(&little).unwrap();
        std::fs::remove_file(&big).ok();
        std::fs::remove_file(&little).ok();
        assert_eq!(round.scalar_slice::<i32>().unwrap(), values.as_slice());
    }

    /// Multi-channel data is swapped per component, not per pixel
    /// (metaImage.cxx:806-838 iterates `quantity * m_ElementNumberOfChannels`).
    #[test]
    fn msb_swaps_each_channel_of_a_vector_pixel() {
        let header = "ObjectType = Image\n\
             NDims = 2\n\
             BinaryDataByteOrderMSB = True\n\
             DimSize = 2 1\n\
             ElementNumberOfChannels = 2\n\
             ElementType = MET_USHORT\n\
             ElementDataFile = LOCAL\n";
        let path = write_mha(
            "msb_vector.mha",
            header,
            &[0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04],
        );
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.pixel_id(), PixelId::VectorUInt16);
        assert_eq!(img.component_slice::<u16>().unwrap(), &[1, 2, 3, 4]);
    }

    // ---- ElementDataFile = LIST ------------------------------------------

    /// `ElementDataFile = LIST` names one file per slice on the header lines
    /// that follow; each holds `prod(DimSize[..NDims-1])` pixels
    /// (metaImage.cxx:1318-1387).
    #[test]
    fn list_reads_one_file_per_slice() {
        let s0 = tmp_path("list_s0.raw");
        let s1 = tmp_path("list_s1.raw");
        std::fs::write(&s0, [1u8, 2, 3, 4]).unwrap();
        std::fs::write(&s1, [5u8, 6, 7, 8]).unwrap();

        let header = format!(
            "ObjectType = Image\n\
             NDims = 3\n\
             BinaryData = True\n\
             DimSize = 2 2 2\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LIST\n\
             {}\n{}\n",
            s0.file_name().unwrap().to_string_lossy(),
            s1.file_name().unwrap().to_string_lossy(),
        );
        let path = tmp_path("list.mhd");
        std::fs::write(&path, header).unwrap();

        let img = read_image(&path).unwrap();
        for p in [&path, &s0, &s1] {
            std::fs::remove_file(p).ok();
        }
        assert_eq!(img.size(), &[2, 2, 2]);
        assert_eq!(img.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    /// `LIST <n>` overrides how many axes live inside each file: `LIST 1` on a
    /// 2-D image means one file per *row*. The word is read with `atof` and
    /// falls back to `NDims - 1` when it is `0` or exceeds `NDims`
    /// (metaImage.cxx:1319-1333). Trailing whitespace and carriage returns are
    /// stripped from each name (metaImage.cxx:1352-1356).
    #[test]
    fn list_honours_an_explicit_file_image_dimension() {
        let r0 = tmp_path("list_r0.raw");
        let r1 = tmp_path("list_r1.raw");
        std::fs::write(&r0, [10u8, 20]).unwrap();
        std::fs::write(&r1, [30u8, 40]).unwrap();

        let header = format!(
            "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             DimSize = 2 2\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LIST 1\n\
             {}  \r\n{}\n",
            r0.file_name().unwrap().to_string_lossy(),
            r1.file_name().unwrap().to_string_lossy(),
        );
        let path = tmp_path("list_dim.mhd");
        std::fs::write(&path, header).unwrap();

        let img = read_image(&path).unwrap();
        for p in [&path, &r0, &r1] {
            std::fs::remove_file(p).ok();
        }
        assert_eq!(img.scalar_slice::<u8>().unwrap(), &[10, 20, 30, 40]);
    }

    /// Upstream's `for (i = 0; i < totalFiles && !_stream->eof(); ++i)` returns
    /// success on a short list, leaving the tail of the pixel buffer
    /// uninitialised. That is unreproducible in safe Rust; a short list is
    /// truncated data here.
    #[test]
    fn list_with_too_few_slices_is_truncated_data() {
        let s0 = tmp_path("short_list_s0.raw");
        std::fs::write(&s0, [1u8, 2, 3, 4]).unwrap();
        let header = format!(
            "ObjectType = Image\n\
             NDims = 3\n\
             DimSize = 2 2 2\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LIST\n\
             {}\n",
            s0.file_name().unwrap().to_string_lossy(),
        );
        let path = tmp_path("short_list.mhd");
        std::fs::write(&path, header).unwrap();

        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&s0).ok();
        assert!(matches!(result, Err(IoError::TruncatedData)), "{result:?}");
    }

    // ---- ImageFileReader extraction ---------------------------------------

    /// A 3x3x2 `Int16` volume with an oblique direction, used by the extraction
    /// tests. Pixel `(x, y, z)` holds `x + 3y + 9z`.
    fn write_volume(name: &str) -> std::path::PathBuf {
        let data: Vec<i16> = (0..18).collect();
        let mut img = Image::from_vec(&[3, 3, 2], data).unwrap();
        img.set_spacing(&[1.0, 2.0, 4.0]).unwrap();
        img.set_origin(&[10.0, 20.0, 30.0]).unwrap();
        img.set_direction(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let path = tmp_path(name);
        write_image(&img, &path).unwrap();
        path
    }

    /// An extraction region equal to the whole file, at index zero, is the full
    /// read: same buffer, same geometry, same dictionary.
    #[test]
    fn extract_of_the_whole_region_equals_a_full_read() {
        let path = write_volume("extract_full.mha");
        let full = read_image(&path).unwrap();

        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        reader.set_extract_size(&[3, 3, 2]);
        reader.set_extract_index(&[0, 0, 0]);
        let extracted = reader.execute().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(extracted, full);
    }

    /// A zero-size axis collapses to the **identity** direction, per the
    /// documented contract (§3.27): reduction sets the direction to the
    /// identity (`SetDirectionCollapseToIdentity`), not the file direction's
    /// submatrix. The origin is shifted by the retained axes' index through
    /// that identity direction (`FixNonZeroIndex`, sitkImageFileReader.cxx:
    /// 39-67); the collapsed axis's own index selects the slice (`z = 1`) but
    /// never shifts the origin (itkExtractImageFilter.hxx:162-179).
    #[test]
    fn extract_with_a_zero_size_axis_gets_the_identity_direction_and_honours_every_index() {
        let path = write_volume("extract_slice.mha");
        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        reader.set_extract_size(&[2, 2, 0]);
        reader.set_extract_index(&[1, 1, 1]);
        let img = reader.execute().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.dimension(), 2);
        assert_eq!(img.size(), &[2, 2]);
        assert_eq!(img.spacing(), &[1.0, 2.0]);
        assert_eq!(img.direction(), &[1.0, 0.0, 0.0, 1.0]);
        // origin + I * (spacing .* index) = [10, 20] + [1*1, 2*1] = [11, 22];
        // the collapsed z index (1) selects the slice but does not shift it.
        assert_eq!(img.origin(), &[11.0, 22.0]);
        // z = 1 slice at x,y in {1,2}: value x + 3y + 9z.
        assert_eq!(img.scalar_slice::<i16>().unwrap(), &[13, 14, 16, 17]);
        // The dictionary rides along (sitkImageFileReader.cxx:453).
        assert_eq!(img.meta_data("ITK_InputFilterName"), Some("MetaImageIO"));
    }

    /// The one-fewer-`0` spelling of the same request. Upstream sent this
    /// through a *second* pipeline that gave the identity direction but dropped
    /// `extract_index[2]` (reading `z = 0`), so `[2, 2]` and `[2, 2, 0]`
    /// returned different pixels (§3.27). This port runs a single pipeline, so
    /// the two spellings agree byte for byte.
    #[test]
    fn extract_without_a_zero_axis_agrees_with_the_zero_axis_spelling() {
        let path = write_volume("extract_direct.mha");
        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        reader.set_extract_size(&[2, 2]);
        reader.set_extract_index(&[1, 1, 1]);
        let img = reader.execute().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.size(), &[2, 2]);
        assert_eq!(img.direction(), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(img.origin(), &[11.0, 22.0]);
        // The trailing index is now honoured — `z = 1`, as for `[2, 2, 0]`.
        assert_eq!(img.scalar_slice::<i16>().unwrap(), &[13, 14, 16, 17]);
    }

    /// Fewer than two non-zero axes is rejected before any pixel is read
    /// (sitkImageFileReader.cxx:319-324), and a region reaching past the file
    /// is rejected against the file's largest possible region (:440-444).
    #[test]
    fn extract_rejects_a_degenerate_or_out_of_bounds_region() {
        let path = write_volume("extract_bad.mha");
        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);

        reader.set_extract_size(&[3, 0, 0]);
        assert!(matches!(
            reader.execute(),
            Err(IoError::ExtractOutputDimension(1))
        ));

        reader
            .set_extract_size(&[3, 3, 0])
            .set_extract_index(&[0, 0, 2]);
        let out_of_range = reader.execute();
        assert!(
            matches!(out_of_range, Err(IoError::ExtractRegionOutOfBounds { .. })),
            "{out_of_range:?}"
        );

        reader
            .set_extract_size(&[4, 3, 1])
            .set_extract_index(&[0, 0, 0]);
        let too_wide = reader.execute();
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(too_wide, Err(IoError::ExtractRegionOutOfBounds { .. })),
            "{too_wide:?}"
        );
    }

    /// Collapsing to the identity never inverts a submatrix, so a file
    /// direction whose retained axes map onto the same physical axis — one that
    /// `DIRECTIONCOLLAPSETOSUBMATRIX` would reject as singular
    /// (itkExtractImageFilter.hxx:194-200) — is no longer an error under the
    /// documented `…ToIdentity` contract (§3.27): the output direction is
    /// simply the identity.
    #[test]
    fn extract_of_a_singular_file_direction_gives_the_identity_not_an_error() {
        let header = "ObjectType = Image\n\
             NDims = 3\n\
             TransformMatrix = 0 0 1 0 0 1 1 0 0\n\
             DimSize = 2 2 2\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LOCAL\n";
        let path = write_mha("singular.mha", header, &[0u8; 8]);
        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        reader.set_extract_size(&[2, 2, 0]);
        let img = reader.execute().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.size(), &[2, 2]);
        assert_eq!(img.direction(), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(img.scalar_slice::<u8>().unwrap(), &[0, 0, 0, 0]);
    }

    // ---- NIfTI-1 ----------------------------------------------------------

    fn patch_i16(bytes: &mut [u8], off: usize, v: i16) {
        bytes[off..off + 2].copy_from_slice(&v.to_le_bytes());
    }

    fn patch_f32(bytes: &mut [u8], off: usize, v: f32) {
        bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// Write `img` as `name`, read the file back as bytes, apply `patch`, and
    /// write it out again — the way to build a NIfTI fixture whose header this
    /// crate's writer would never emit.
    fn patched_nii(
        name: &str,
        img: &Image,
        patch: impl FnOnce(&mut Vec<u8>),
    ) -> std::path::PathBuf {
        let path = tmp_path(name);
        write_image(img, &path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        patch(&mut bytes);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn nii_roundtrip_all_scalar_types() {
        macro_rules! case {
            ($ty:ty, $name:expr) => {{
                let data: Vec<$ty> = (0..8u32).map(|i| i as $ty).collect();
                let img = Image::from_vec(&[4, 2], data.clone()).unwrap();
                let path = tmp_path($name);
                write_image(&img, &path).unwrap();
                let back = read_image(&path).unwrap();
                std::fs::remove_file(&path).ok();
                assert_eq!(back.scalar_slice::<$ty>().unwrap(), data.as_slice(), $name);
            }};
        }
        case!(u8, "u8.nii");
        case!(i8, "i8.nii");
        case!(u16, "u16.nii");
        case!(i16, "i16.nii");
        case!(u32, "u32.nii");
        case!(i32, "i32.nii");
        case!(u64, "u64.nii");
        case!(i64, "i64.nii");
        case!(f32, "f32.nii");
        case!(f64, "f64.nii");
    }

    #[test]
    fn nii_roundtrip_all_scalar_types_3d() {
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
        case!(u8, "u8_3d.nii");
        case!(i8, "i8_3d.nii");
        case!(u16, "u16_3d.nii");
        case!(i16, "i16_3d.nii");
        case!(u32, "u32_3d.nii");
        case!(i32, "i32_3d.nii");
        case!(u64, "u64_3d.nii");
        case!(i64, "i64_3d.nii");
        case!(f32, "f32_3d.nii");
        case!(f64, "f64_3d.nii");
    }

    /// The LPS↔RAS involution: the writer negates rows 0 and 1 of the direction
    /// and the origin, the reader negates them back. `origin[2]` is *not*
    /// negated on either side (itkNiftiImageIO.cxx:1858, :2044).
    #[test]
    fn nii_roundtrip_preserves_direction_and_origin_through_the_lps_ras_flip() {
        let data: Vec<i16> = (0..24).map(|i| i as i16 - 5).collect();
        let mut img = Image::from_vec(&[4, 3, 2], data.clone()).unwrap();
        img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
        img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();
        img.set_direction(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();

        let path = tmp_path("geometry.nii");
        write_image(&img, &path).unwrap();

        // The RAS+ srow the file actually carries: rows 0 and 1 negated, row 2
        // as-is, each column scaled by its spacing.
        let bytes = std::fs::read(&path).unwrap();
        let srow = |row: usize, col: usize| {
            f32::from_le_bytes(
                bytes[280 + 16 * row + 4 * col..284 + 16 * row + 4 * col]
                    .try_into()
                    .unwrap(),
            )
        };
        assert_eq!([srow(0, 0), srow(0, 1), srow(0, 2)], [0.0, 1.25, 0.0]);
        assert_eq!([srow(1, 0), srow(1, 1), srow(1, 2)], [-0.5, 0.0, 0.0]);
        assert_eq!([srow(2, 0), srow(2, 1), srow(2, 2)], [0.0, 0.0, 3.0]);
        assert_eq!([srow(0, 3), srow(1, 3), srow(2, 3)], [2.0, -4.0, 7.5]);

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.size(), img.size());
        assert_eq!(back.pixel_id(), PixelId::Int16);
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(back.scalar_slice::<i16>().unwrap(), data.as_slice());
    }

    /// The whole 348-byte header of a 3x2 `Float32` image, spacing `(0.5, 2)`,
    /// origin `(-2, 4)`, identity direction. `dim[0] = 2` and every unused
    /// `dim[i]` is `1`; `pixdim[0]` is `qfac`; `vox_offset` is `348 + 4`;
    /// `xyzt_units` is `NIFTI_UNITS_MM | NIFTI_UNITS_SEC`; both xform codes are
    /// `NIFTI_XFORM_SCANNER_ANAT`, which
    /// `SetNIfTIOrientationFromImageIO` writes unconditionally (:2077-2078).
    #[test]
    fn nii_header_is_byte_pinned() {
        let mut img = Image::from_vec(&[3, 2], vec![0.0f32; 6]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-2.0, 4.0]).unwrap();

        let path = tmp_path("pinned.nii");
        write_image(&img, &path).unwrap();
        let file = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let mut want = [0u8; 348];
        want[0..4].copy_from_slice(&348i32.to_le_bytes());
        want[38] = b'r';
        for (i, d) in [2i16, 3, 2, 1, 1, 1, 1, 1].iter().enumerate() {
            want[40 + 2 * i..42 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }
        want[70..72].copy_from_slice(&16i16.to_le_bytes()); // datatype = NIFTI_TYPE_FLOAT32
        want[72..74].copy_from_slice(&32i16.to_le_bytes()); // bitpix
        for (i, p) in [1.0f32, 0.5, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            .iter()
            .enumerate()
        {
            want[76 + 4 * i..80 + 4 * i].copy_from_slice(&p.to_le_bytes());
        }
        want[108..112].copy_from_slice(&352.0f32.to_le_bytes()); // vox_offset
        want[112..116].copy_from_slice(&1.0f32.to_le_bytes()); // scl_slope
        want[116..120].copy_from_slice(&0.0f32.to_le_bytes()); // scl_inter
        want[123] = 10; // xyzt_units = MM | SEC
        want[252..254].copy_from_slice(&1i16.to_le_bytes()); // qform_code
        want[254..256].copy_from_slice(&1i16.to_le_bytes()); // sform_code
        want[256..260].copy_from_slice(&0.0f32.to_le_bytes()); // quatern_b
        want[260..264].copy_from_slice(&0.0f32.to_le_bytes()); // quatern_c
        want[264..268].copy_from_slice(&1.0f32.to_le_bytes()); // quatern_d
        want[268..272].copy_from_slice(&2.0f32.to_le_bytes()); // qoffset_x
        want[272..276].copy_from_slice(&(-4.0f32).to_le_bytes()); // qoffset_y
        want[276..280].copy_from_slice(&0.0f32.to_le_bytes()); // qoffset_z
        for (i, v) in [-0.5f32, 0.0, 0.0, 2.0].iter().enumerate() {
            want[280 + 4 * i..284 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [0.0f32, -2.0, 0.0, -4.0].iter().enumerate() {
            want[296 + 4 * i..300 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [0.0f32, 0.0, 1.0, 0.0].iter().enumerate() {
            want[312 + 4 * i..316 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        want[344..348].copy_from_slice(b"n+1\0");

        assert_eq!(&file[..348], &want[..]);
        // The four-byte zero extender `nifti_write_extensions` emits when
        // `num_ext == 0` and `skip_blank_ext == 0` (nifti1_io.c:6062-6072).
        assert_eq!(&file[348..352], &[0, 0, 0, 0]);
        assert_eq!(file.len(), 352 + 6 * 4);
    }

    /// A `.hdr` writes a 352-byte header file (header plus extender) and a
    /// sibling `.img` whose pixels start at offset zero; the magic is `ni1`.
    /// Either name reads the pair back — `nifti_findhdrname` walks from an
    /// `.img` to its `.hdr` (nifti1_io.c:2779-2801).
    #[test]
    fn hdr_img_pair_roundtrips_from_either_name() {
        let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let img = Image::from_vec(&[3, 2], data.clone()).unwrap();
        let hdr = tmp_path("pair.hdr");
        write_image(&img, &hdr).unwrap();

        let raw = std::fs::read(&hdr).unwrap();
        assert_eq!(raw.len(), 352);
        assert_eq!(&raw[344..348], b"ni1\0");
        assert_eq!(f32::from_le_bytes(raw[108..112].try_into().unwrap()), 0.0);

        let img_file = tmp_path("pair.img");
        assert_eq!(std::fs::read(&img_file).unwrap().len(), 24);

        for name in [&hdr, &img_file] {
            let back = read_image(name).unwrap();
            assert_eq!(back.size(), &[3, 2]);
            assert_eq!(back.scalar_slice::<f32>().unwrap(), data.as_slice());
        }
        std::fs::remove_file(&hdr).ok();
        std::fs::remove_file(&img_file).ok();
    }

    /// Unlike MetaImage, NIfTI has complex datatypes, and `NiftiImageIO` reports
    /// `numberOfComponents == 2` alongside `IOPixelEnum::COMPLEX` — the exact
    /// shape `GetPixelIDFromImageIO` needs to hand back a complex pixel ID
    /// (sitkImageReaderBase.cxx:216-236). So a complex image survives the round
    /// trip as complex.
    #[test]
    fn nii_roundtrip_complex_stays_complex() {
        let data32: Vec<Complex<f32>> = (0..6)
            .map(|i| Complex::new(i as f32 * 1.5, -(i as f32) - 0.5))
            .collect();
        let mut img = Image::from_vec_complex::<f32>(&[3, 2], data32).unwrap();
        img.set_spacing(&[1.5, 0.5]).unwrap();
        img.set_origin(&[2.0, -3.0]).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();

        let path = tmp_path("complex_f32.nii");
        write_image(&img, &path).unwrap();
        assert_eq!(
            i16::from_le_bytes(std::fs::read(&path).unwrap()[70..72].try_into().unwrap()),
            32, // NIFTI_TYPE_COMPLEX64
        );
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::ComplexFloat32);
        assert_eq!(back.number_of_components_per_pixel(), 1);
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(
            back.component_slice::<f32>().unwrap(),
            img.component_slice::<f32>().unwrap()
        );

        let data64: Vec<Complex<f64>> = (0..6)
            .map(|i| Complex::new(i as f64 * 1.5, -(i as f64) - 0.5))
            .collect();
        let img = Image::from_vec_complex::<f64>(&[3, 2], data64).unwrap();
        let path = tmp_path("complex_f64.nii");
        write_image(&img, &path).unwrap();
        assert_eq!(
            i16::from_le_bytes(std::fs::read(&path).unwrap()[70..72].try_into().unwrap()),
            1792, // NIFTI_TYPE_COMPLEX128
        );
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::ComplexFloat64);
        assert_eq!(
            back.component_slice::<f64>().unwrap(),
            img.component_slice::<f64>().unwrap()
        );
    }

    /// A vector image is `dim[0] = 5`, `dim[5] = numComponents`,
    /// `intent_code = NIFTI_INTENT_VECTOR`, and the components are the *slowest*
    /// axis on disk — the transpose of ITK's interleaved buffer
    /// (itkNiftiImageIO.cxx:2151-2175).
    #[test]
    fn nii_roundtrip_vector_float32_stores_components_slowest() {
        let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25 - 4.0).collect();
        let mut img = Image::from_vec_vector::<f32>(&[4, 3], 3, data.clone()).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        img.set_direction(&[0.0, 1.0, -1.0, 0.0]).unwrap();

        let path = tmp_path("vector_f32.nii");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(i16::from_le_bytes(bytes[40..42].try_into().unwrap()), 5);
        assert_eq!(i16::from_le_bytes(bytes[50..52].try_into().unwrap()), 3);
        assert_eq!(i16::from_le_bytes(bytes[68..70].try_into().unwrap()), 1007);
        // On disk, component 0 of every voxel comes first.
        let on_disk: Vec<f32> = bytes[352..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let d = &data;
        let expect: Vec<f32> = (0..3)
            .flat_map(|c| (0..12).map(move |v| d[v * 3 + c]))
            .collect();
        assert_eq!(on_disk, expect);

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.size(), img.size());
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(back.component_slice::<f32>().unwrap(), data.as_slice());
    }

    /// `m_ConvertRASDisplacementVectors` defaults to `true`
    /// (itkNiftiImageIO.h:289), so a `NIFTI_INTENT_DISPVECT` image is stored in
    /// RAS+: the `x` and `y` component planes are negated on write
    /// (`ConvertRASToFromLPS_XYZTC`, :279-289) and again on read
    /// (`ConvertRASToFromLPS_CXYZT`, :263-274). The pixels survive; the bytes
    /// on disk do not match a plain `NIFTI_INTENT_VECTOR` file.
    #[test]
    fn nii_dispvect_negates_the_x_and_y_component_planes_on_disk() {
        let data: Vec<f32> = (1..=36).map(|i| i as f32).collect();
        let mut img = Image::from_vec_vector::<f32>(&[4, 3], 3, data.clone()).unwrap();
        img.set_meta_data("intent_code", "1006");

        let path = tmp_path("dispvect.nii");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(i16::from_le_bytes(bytes[68..70].try_into().unwrap()), 1006);

        let on_disk: Vec<f32> = bytes[352..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // Planes 0 and 1 (24 of the 36 components) are negated; plane 2 is not.
        let d = &data;
        let expect: Vec<f32> = (0..3)
            .flat_map(|c| {
                (0..12).map(move |v| {
                    let x = d[v * 3 + c];
                    if c < 2 { -x } else { x }
                })
            })
            .collect();
        assert_eq!(on_disk, expect);

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.component_slice::<f32>().unwrap(), data.as_slice());
    }

    /// Fixed §1.51 (write side): upstream's RAS-conversion guard
    /// (itkNiftiImageIO.cxx:2177-2183) checks only the pixel type, never that
    /// `numComponents == 3`, so a 2- or 4-component `NIFTI_INTENT_DISPVECT`
    /// image would get a stride-3 walk over a differently-strided buffer.
    /// `write` now rejects it instead.
    #[test]
    fn nii_write_of_a_non_three_component_dispvect_image_is_rejected() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut img = Image::from_vec_vector::<f32>(&[4, 1], 2, data).unwrap();
        img.set_meta_data("intent_code", "1006"); // NIFTI_INTENT_DISPVECT

        let path = tmp_path("dispvect_2component_write.nii");
        let result = write_image(&img, &path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::NiftiWriteRejected(m)) if m.contains("3-component")),
            "{result:?}"
        );
    }

    /// Fixed §1.51 (read side): the mirror-image case of the write-side test
    /// above, for a file whose header was hand-patched to `intent_code =
    /// NIFTI_INTENT_DISPVECT` over a 2-component vector body — the shape no
    /// legitimate writer (including this port's own) produces, since it now
    /// refuses to write one. `read` rejects it rather than applying the
    /// stride-3 sign flip across voxel boundaries.
    #[test]
    fn nii_read_of_a_non_three_component_dispvect_image_is_rejected() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let img = Image::from_vec_vector::<f32>(&[4, 1], 2, data).unwrap();
        let path = patched_nii("dispvect_2component_read.nii", &img, |b| {
            patch_i16(b, 68, 1006); // intent_code = NIFTI_INTENT_DISPVECT
        });

        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::UnsupportedNiftiFeature(m)) if m.contains("3-component")),
            "{result:?}"
        );
    }

    /// `ReadImageInformation` derives a vector image's dimension from `dim[4]`,
    /// `dim[3]` and `dim[2]` only (itkNiftiImageIO.cxx:788-805) — `dim[1]` is
    /// never consulted. A 2-D vector image one row tall therefore reads back
    /// 1-D. Upstream quirk, pinned (ledger §2.89).
    #[test]
    fn nii_vector_image_one_row_tall_reads_back_as_one_dimensional() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let img = Image::from_vec_vector::<f32>(&[4, 1], 3, data.clone()).unwrap();
        let path = tmp_path("flat_vector.nii");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.dimension(), 2);
        assert_eq!(back.dimension(), 1);
        assert_eq!(back.size(), &[4]);
        assert_eq!(back.component_slice::<f32>().unwrap(), data.as_slice());
    }

    /// The "HACK ALERT KW" loop (itkNiftiImageIO.cxx:820-824) trims trailing
    /// unit dimensions of a *scalar* image while the index stays above three, so
    /// a 4-D volume with one time point reads back 3-D.
    #[test]
    fn nii_scalar_4d_with_one_time_point_reads_back_as_three_dimensional() {
        let data: Vec<u8> = (0..8).collect();
        let img = Image::from_vec(&[2, 2, 2, 1], data.clone()).unwrap();
        let path = tmp_path("collapse_t.nii");
        write_image(&img, &path).unwrap();
        assert_eq!(
            i16::from_le_bytes(std::fs::read(&path).unwrap()[40..42].try_into().unwrap()),
            4,
        );
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.dimension(), 3);
        assert_eq!(back.size(), &[2, 2, 2]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), data.as_slice());
    }

    /// `scl_slope`/`scl_inter` rescaling promotes an integer on-disk type to
    /// `float` (itkNiftiImageIO.cxx:1005-1016) and applies `v * slope + inter`.
    #[test]
    fn nii_rescale_promotes_an_integer_datatype_to_float32() {
        let data: Vec<i16> = (0..6).map(|i| i as i16).collect();
        let img = Image::from_vec(&[3, 2], data.clone()).unwrap();
        let path = patched_nii("rescale_i16.nii", &img, |b| {
            patch_f32(b, 112, 2.0); // scl_slope
            patch_f32(b, 116, 1.0); // scl_inter
        });

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::Float32);
        assert_eq!(
            back.scalar_slice::<f32>().unwrap(),
            &[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        );
    }

    /// Fixed §1.50: `RescaleFunction(buffer, slope, inter, count)` now passes
    /// `numElts * GetNumberOfComponents()`, not just `numElts` — the *voxel*
    /// count — so a complex image's full `2 * numElts` interleaved buffer is
    /// rescaled, both real and imaginary parts of every voxel alike.
    #[test]
    fn nii_rescale_of_a_complex_image_covers_every_component() {
        let data: Vec<Complex<f32>> = (1..=6)
            .map(|i| Complex::new(i as f32, -(i as f32)))
            .collect();
        let img = Image::from_vec_complex::<f32>(&[3, 2], data).unwrap();
        let path = patched_nii("rescale_complex.nii", &img, |b| {
            patch_f32(b, 112, 10.0); // scl_slope
        });

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::ComplexFloat32);
        assert_eq!(
            back.component_slice::<f32>().unwrap(),
            // every component of every voxel scaled by 10
            &[
                10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 40.0, -40.0, 50.0, -50.0, 60.0, -60.0
            ]
        );
    }

    /// `DT_RGB24` loads as `IOPixelEnum::RGB` with three `unsigned char`
    /// components (itkNiftiImageIO.cxx:906-910), which SimpleITK maps onto a
    /// vector pixel ID (sitkImageReaderBase.cxx:220-231). RGB is interleaved on
    /// disk, so it takes the memcpy branch, not the de-interleave.
    #[test]
    fn nii_rgb24_reads_as_a_three_component_vector_image() {
        let img = Image::from_vec(&[3, 2], vec![0u8; 6]).unwrap();
        let path = patched_nii("rgb24.nii", &img, |b| {
            patch_i16(b, 70, 128); // datatype = NIFTI_TYPE_RGB24
            patch_i16(b, 72, 24); // bitpix
            b.truncate(352);
            b.extend((0..18u8).map(|i| i + 1));
        });

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(
            back.component_slice::<u8>().unwrap(),
            &(1..=18u8).collect::<Vec<_>>()[..]
        );
    }

    /// `NIFTI_TYPE_FLOAT128` and `NIFTI_TYPE_COMPLEX256` fall through
    /// `ReadImageInformation`'s datatype switch to `default: break`
    /// (itkNiftiImageIO.cxx:924), leaving `UNKNOWNCOMPONENTTYPE`.
    #[test]
    fn nii_rejects_datatypes_itk_leaves_unknown() {
        let img = Image::from_vec(&[3, 2], vec![0.0f32; 6]).unwrap();
        for (datatype, name) in [(1536i16, "f128.nii"), (2048, "c256.nii")] {
            let path = patched_nii(name, &img, |b| patch_i16(b, 70, datatype));
            let result = read_image(&path);
            std::fs::remove_file(&path).ok();
            assert!(
                matches!(result, Err(IoError::UnsupportedNiftiDatatype(d)) if d == datatype),
                "{result:?}"
            );
        }
    }

    /// `nifti_convert_nhdr2nim` rejects `DT_UNKNOWN` and `DT_BINARY` outright
    /// (nifti1_io.c:3653-3658), and a non-positive `dim[1]` (:3691-3695).
    #[test]
    fn nii_rejects_a_malformed_header() {
        let img = Image::from_vec(&[3, 2], vec![0.0f32; 6]).unwrap();

        let path = patched_nii("bad_datatype.nii", &img, |b| patch_i16(b, 70, 1));
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::MalformedNiftiHeader(m)) if m == "bad datatype"),
            "{result:?}"
        );

        let path = patched_nii("bad_dim1.nii", &img, |b| patch_i16(b, 42, 0));
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::MalformedNiftiHeader(m)) if m == "bad dim[1]"),
            "{result:?}"
        );
    }

    /// `need_nhdr_swap` reads `dim[0]`: out of `1..=7` in the host's order but
    /// in range byte-swapped means the whole file is foreign-endian
    /// (nifti1_io.c:4143-4176), and `nifti_read_buffer` then swaps the pixels by
    /// the datatype's `swapsize` (:5030-5034).
    #[test]
    fn nii_big_endian_file_reads_on_a_little_endian_host() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        let mut img = Image::from_vec(&[4, 3, 2], data.clone()).unwrap();
        img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
        img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();

        // Every numeric field of `nifti_1_header`, as (offset, width).
        let mut fields: Vec<(usize, usize)> = vec![(0, 4), (32, 4), (36, 2)];
        fields.extend((0..8).map(|i| (40 + 2 * i, 2)));
        fields.extend([
            (56, 4),
            (60, 4),
            (64, 4),
            (68, 2),
            (70, 2),
            (72, 2),
            (74, 2),
        ]);
        fields.extend((0..8).map(|i| (76 + 4 * i, 4)));
        fields.extend([
            (108, 4),
            (112, 4),
            (116, 4),
            (120, 2),
            (124, 4),
            (128, 4),
            (132, 4),
            (136, 4),
            (140, 4),
            (144, 4),
            (252, 2),
            (254, 2),
        ]);
        fields.extend((0..6).map(|i| (256 + 4 * i, 4)));
        fields.extend((0..12).map(|i| (280 + 4 * i, 4)));

        let path = patched_nii("big_endian.nii", &img, |b| {
            for (off, width) in fields {
                b[off..off + width].reverse();
            }
            for chunk in b[352..].chunks_exact_mut(4) {
                chunk.reverse();
            }
        });

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.size(), &[4, 3, 2]);
        assert_eq!(back.spacing(), &[0.5, 1.25, 3.0]);
        assert_eq!(back.origin(), &[-2.0, 4.0, 7.5]);
        assert_eq!(back.scalar_slice::<f32>().unwrap(), data.as_slice());
    }

    /// `nifti_read_buffer`'s `#ifdef isfinite` block (nifti1_io.c:5036-5070) is
    /// live on glibc, which defines `isfinite` as a macro: every non-finite
    /// float read from disk becomes zero. Platform-dependent upstream behaviour,
    /// pinned for Linux (ledger §2.90).
    #[test]
    fn nii_non_finite_pixels_are_zeroed_on_read() {
        let img = Image::from_vec(&[3, 1], vec![f32::NAN, f32::INFINITY, 1.5]).unwrap();
        let path = tmp_path("nonfinite.nii");
        write_image(&img, &path).unwrap();
        // The writer stores them verbatim; only the reader sanitises.
        let bytes = std::fs::read(&path).unwrap();
        assert!(f32::from_le_bytes(bytes[352..356].try_into().unwrap()).is_nan());

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<f32>().unwrap(), &[0.0, 0.0, 1.5]);
    }

    /// A gzipped NIfTI is claimed by the registry and round-trips. Compression
    /// follows the `.gz` name, never `WriteOptions` — see
    /// [`crate::nifti::write`] and `tests/compression.rs` for the rest.
    #[test]
    fn nii_gz_is_recognised_and_round_trips() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("compressed.nii.gz");

        assert!(create_image_io(&path, FileMode::Write).is_some());
        write_image(&img, &path).unwrap();
        assert_eq!(&std::fs::read(&path).unwrap()[..2], b"\x1f\x8b");

        assert!(create_image_io(&path, FileMode::Read).is_some());
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
    }

    /// `.nia`, the NIfTI ASCII variant, is a valid write target for
    /// `nifti_is_complete_filename` — and then `WriteImageInformation` picks
    /// `NIFTI_FTYPE_ASCII`, which this port does not implement.
    #[test]
    fn nia_is_claimed_for_writing_and_then_refused() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("ascii.nia");
        assert!(create_image_io(&path, FileMode::Write).is_some());
        let result = write_image(&img, &path);
        assert!(
            matches!(&result, Err(IoError::UnsupportedNiftiFeature(m)) if m.contains(".nia")),
            "{result:?}"
        );
    }

    /// `NiftiImageIO::CanReadFile` resolves the *header* file through
    /// `nifti_findhdrname` before it looks at any content
    /// (itkNiftiImageIO.cxx:604), so a path with no extension at all is claimed
    /// when the matching `.nii` exists. That is `CreateImageIO`'s phase 2 doing
    /// real work — the opposite of `MetaImageIo`, which re-checks the extension
    /// itself and declines.
    #[test]
    fn nifti_claims_an_extensionless_path_in_phase_two() {
        let stem = tmp_path("stemonly");
        let nii = tmp_path("stemonly.nii");
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        write_image(&img, &nii).unwrap();

        assert!(create_image_io(&stem, FileMode::Read).is_some());
        let back = read_image(&stem).unwrap();
        std::fs::remove_file(&nii).ok();
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
    }

    /// `ReadImageInformation` never touches the pixel tail.
    #[test]
    fn nii_read_image_information_does_not_load_pixels() {
        let img = Image::from_vec(&[3, 2], vec![0.0f32; 6]).unwrap();
        let path = patched_nii("info_only.nii", &img, |b| {
            patch_i16(b, 42, 20000); // dim[1]
            patch_i16(b, 44, 20000); // dim[2]
            b.truncate(352);
        });

        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        let info = reader.read_image_information().unwrap().clone();
        let loaded = reader.execute();
        std::fs::remove_file(&path).ok();

        assert_eq!(info.pixel_id, PixelId::Float32);
        assert_eq!(info.dimension, 2);
        assert_eq!(info.number_of_components, 1);
        assert_eq!(info.size, vec![20000, 20000]);
        assert!(matches!(loaded, Err(IoError::TruncatedData)), "{loaded:?}");
    }

    /// `ReadImageInformation` encapsulates `ITK_InputFilterName` at
    /// itkNiftiImageIO.cxx:1102 and `SetImageIOMetadataFromNIfTI` then calls
    /// `thisDic.Clear()` at :630 — so a NIfTI image never carries the key that
    /// every MetaImage does. Upstream quirk, pinned (ledger §2.86).
    #[test]
    fn nii_dictionary_has_no_itk_input_filter_name() {
        let mut img = Image::from_vec(&[3, 2], vec![0.0f32; 6]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        let path = tmp_path("dict.nii");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.meta_data("ITK_InputFilterName"), None);
        assert_eq!(back.meta_data("nifti_type"), Some("1"));
        assert_eq!(back.meta_data("datatype"), Some("16"));
        assert_eq!(back.meta_data("bitpix"), Some("32"));
        assert_eq!(back.meta_data("vox_offset"), Some("352"));
        assert_eq!(back.meta_data("xyzt_units"), Some("10"));
        assert_eq!(back.meta_data("scl_slope"), Some("1"));
        assert_eq!(back.meta_data("qform_code"), Some("1"));
        assert_eq!(
            back.meta_data("qform_code_name"),
            Some("NIFTI_XFORM_SCANNER_ANAT")
        );
        assert_eq!(back.meta_data("ITK_sform_corrected"), Some("NO"));
        assert_eq!(back.meta_data("pixdim[1]"), Some("0.5"));
        // `srow_x[3]` is `-(origin[0] as f32)` and `origin[0]` is `0.0`, so the
        // field genuinely holds `-0.0`; double-conversion's `ToShortest` — and
        // Rust's `Display` — both render that as `-0`.
        assert_eq!(back.meta_data("srow_x"), Some("-0.5 0 0 -0"));
    }

    /// `aux_file` and `ITK_FileNotes` feed fixed-width header fields and are
    /// rejected when they overflow (itkNiftiImageIO.cxx:1478, :1493); a
    /// non-numeric `qform_code` is rejected by `itk::StringToInt32` even though
    /// the value it produces is then thrown away at :2077.
    #[test]
    fn nii_write_rejects_unusable_dictionary_values() {
        let base = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();

        let mut img = base.clone();
        img.set_meta_data("aux_file", &"x".repeat(24));
        let result = write_image(&img, tmp_path("aux.nii"));
        assert!(
            matches!(&result, Err(IoError::InvalidNiftiMetaData(m)) if m.contains("aux_file")),
            "{result:?}"
        );

        let mut img = base.clone();
        img.set_meta_data("ITK_FileNotes", &"x".repeat(80));
        let result = write_image(&img, tmp_path("notes.nii"));
        assert!(
            matches!(&result, Err(IoError::InvalidNiftiMetaData(m)) if m.contains("ITK_FileNotes")),
            "{result:?}"
        );

        let mut img = base.clone();
        img.set_meta_data("qform_code", "not a number");
        let result = write_image(&img, tmp_path("qform.nii"));
        assert!(
            matches!(&result, Err(IoError::InvalidNiftiMetaData(m)) if m.contains("qform_code")),
            "{result:?}"
        );
    }

    /// `descrip` and `aux_file` round-trip; `ITK_FileNotes` is `descrip` under
    /// another name (itkNiftiImageIO.cxx:751, :1112).
    #[test]
    fn nii_descrip_and_aux_file_roundtrip() {
        let mut img = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        img.set_meta_data("ITK_FileNotes", "acquired on a rainy Tuesday");
        img.set_meta_data("aux_file", "sidecar.txt");

        let path = tmp_path("notes_roundtrip.nii");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(
            back.meta_data("descrip"),
            Some("acquired on a rainy Tuesday")
        );
        assert_eq!(
            back.meta_data("ITK_FileNotes"),
            Some("acquired on a rainy Tuesday")
        );
        assert_eq!(back.meta_data("aux_file"), Some("sidecar.txt"));
    }

    /// An Analyze-7.5 header — `sizeof_hdr == 348`, no NIfTI magic — is read
    /// with identity direction and zero origin: `nifti_convert_nhdr2nim` forces
    /// both xform codes to `NIFTI_XFORM_UNKNOWN` for a non-NIfTI header
    /// (nifti1_io.c:3773, :3843), and `SetImageIOOrientationFromNIfTI` then
    /// returns early (itkNiftiImageIO.cxx:1591-1610) because the compiled-in
    /// `Analyze75Flavor` is `AnalyzeITK4Warning`. `scl_slope` is not copied out
    /// of a non-NIfTI header either (:3861-3878), so no rescale happens whatever
    /// the bytes at offset 112 say.
    ///
    /// The reported `nifti_type` still comes out `1`, not `0`:
    /// `nifti_set_type_from_names` sees one file doing both jobs and coerces the
    /// type to `NIFTI_FTYPE_NIFTI1_1` regardless of the missing magic
    /// (nifti1_io.c:3452-3454). Upstream quirk, pinned (ledger §2.92).
    #[test]
    fn analyze75_single_file_reads_with_identity_geometry_and_a_coerced_nifti_type() {
        let mut img = Image::from_vec(&[3, 2], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-2.0, 4.0]).unwrap();
        let path = patched_nii("analyze.nii", &img, |b| {
            b[344..348].copy_from_slice(&[0, 0, 0, 0]); // drop the magic
            patch_f32(b, 112, 8.0); // scl_slope, which a non-NIfTI header ignores
        });

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::Float32);
        assert_eq!(back.spacing(), &[0.5, 2.0]);
        assert_eq!(back.origin(), &[0.0, 0.0]);
        assert_eq!(back.direction(), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            back.scalar_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(back.meta_data("nifti_type"), Some("1"));
        assert_eq!(back.meta_data("scl_slope"), Some("0"));
        assert_eq!(back.meta_data("qform_code"), Some("0"));
        assert_eq!(back.meta_data("sform_code"), Some("0"));
    }

    /// A magic-less `.hdr`/`.img` pair keeps `nifti_type == NIFTI_FTYPE_ANALYZE`,
    /// because `nifti_set_type_from_names` only coerces when the two names are
    /// the same.
    #[test]
    fn analyze75_two_file_pair_keeps_nifti_type_zero() {
        let img = Image::from_vec(&[3, 2], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let hdr = tmp_path("analyze_pair.hdr");
        write_image(&img, &hdr).unwrap();
        let mut bytes = std::fs::read(&hdr).unwrap();
        bytes[344..348].copy_from_slice(&[0, 0, 0, 0]);
        std::fs::write(&hdr, bytes).unwrap();

        let back = read_image(&hdr).unwrap();
        std::fs::remove_file(&hdr).ok();
        std::fs::remove_file(tmp_path("analyze_pair.img")).ok();

        assert_eq!(back.meta_data("nifti_type"), Some("0"));
        assert_eq!(back.origin(), &[0.0, 0.0]);
        assert_eq!(back.direction(), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            back.scalar_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    /// A `NIFTI_INTENT_SYMMATRIX` file loads as
    /// `IOPixelEnum::SYMMETRICSECONDRANKTENSOR`, which
    /// `GetPixelIDFromImageIO` has no branch for: it falls to the final `else`
    /// and throws "Unknown PixelType" (sitkImageReaderBase.cxx:238). SimpleITK
    /// simply cannot open such a file (ledger §3.32).
    #[test]
    fn nii_symmatrix_is_unreadable_through_the_simpleitk_pixel_id_mapping() {
        let img = Image::from_vec_vector::<f32>(&[2, 2], 3, vec![0.0; 12]).unwrap();
        let path = patched_nii("symmatrix.nii", &img, |b| patch_i16(b, 68, 1005));
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::UnsupportedNiftiFeature(m)) if m.contains("SYMMATRIX")),
            "{result:?}"
        );
    }

    /// `NIFTI_INTENT_GENMATRIX` is rejected by ITK itself
    /// (itkNiftiImageIO.cxx:806-810).
    #[test]
    fn nii_genmatrix_is_rejected() {
        let img = Image::from_vec_vector::<f32>(&[2, 2], 4, vec![0.0; 16]).unwrap();
        let path = patched_nii("genmatrix.nii", &img, |b| patch_i16(b, 68, 1004));
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::UnsupportedNiftiFeature(m)) if m.contains("GENMATRIX")),
            "{result:?}"
        );
    }

    /// An axis longer than `SHRT_MAX` cannot be expressed in `dim[i]`, and
    /// `WriteImageInformation` says so before it allocates anything
    /// (itkNiftiImageIO.cxx:1143-1150).
    #[test]
    fn nii_write_rejects_an_axis_longer_than_shrt_max() {
        let img = Image::new(&[32768, 1], PixelId::UInt8);
        let result = write_image(&img, tmp_path("too_wide.nii"));
        assert!(
            matches!(&result, Err(IoError::NiftiWriteRejected(m)) if m.contains("32767")),
            "{result:?}"
        );
    }

    /// A vector image of more than four dimensions has no room for `dim[5]`
    /// (itkNiftiImageIO.cxx:1279-1283).
    #[test]
    fn nii_write_rejects_a_five_dimensional_vector_image() {
        let img = Image::from_vec_vector::<u8>(&[2, 2, 2, 2, 2], 3, vec![0u8; 96]).unwrap();
        let result = write_image(&img, tmp_path("vector_5d.nii"));
        assert!(
            matches!(&result, Err(IoError::NiftiWriteRejected(m)) if m.contains("Dimension=5")),
            "{result:?}"
        );
    }

    /// `Read` casts the on-disk integers into a buffer sized `numElts *
    /// sizeof(float)` and then copies `numElts * numComponents * sizeof(float)`
    /// bytes out of it (itkNiftiImageIO.cxx:386, :447). For a rescaled vector
    /// image that is a heap overflow — there is no defined upstream behaviour to
    /// reproduce, so the read is refused (ledger §1.49, §4.59).
    #[test]
    fn nii_rescaled_integer_vector_image_is_refused_rather_than_overflowing() {
        let img = Image::from_vec_vector::<i16>(&[2, 2], 3, (0..12).collect()).unwrap();
        let path = patched_nii("rescale_vector.nii", &img, |b| {
            patch_f32(b, 112, 2.0); // scl_slope
        });
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::UnsupportedNiftiFeature(m)) if m.contains("heap overflow")),
            "{result:?}"
        );
    }

    /// `nifti_find_file_extension` accepts an all-uppercase extension
    /// (`allow_upper_fext` defaults to `1`, nifti1_io.c:2607-2618) but not a
    /// mixed one — it lowercases nothing, it compares against both spellings.
    ///
    /// So `CanWriteFile` — which is exactly `nifti_is_complete_filename` — says
    /// yes to `IMG.NII`, the registry hands the file to the NIfTI writer, and
    /// `WriteImageInformation` then compares the extension against `".nii"`
    /// with `operator==` (itkNiftiImageIO.cxx:1175-1201) and falls through to
    /// `itkExceptionMacro("Bad Nifti file name: ...")`. An uppercase extension
    /// is writable in the factory's judgement and unwritable in the writer's.
    /// Pinned, not fixed (ledger §2.91).
    #[test]
    fn uppercase_nii_is_claimed_for_writing_and_then_refused() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();

        let shouty = tmp_path("shouty.NII");
        let result = write_image(&img, &shouty);
        assert!(
            matches!(&result, Err(IoError::NiftiWriteRejected(m)) if m.starts_with("Bad Nifti file name:")),
            "{result:?}"
        );
        assert!(!shouty.exists());
        assert!(!tmp_path("shouty.nii").exists());

        // A mixed-case extension is not an extension at all, so no IO claims it.
        let mixed = tmp_path("mixed.Nii");
        assert!(matches!(
            write_image(&img, &mixed),
            Err(IoError::NoWriterFound(_))
        ));
    }

    /// Extraction is component-aware: a vector image keeps its channels.
    #[test]
    fn extract_preserves_vector_components() {
        let data: Vec<u8> = (0..27).collect();
        let img = Image::from_vec_vector::<u8>(&[3, 3], 3, data).unwrap();
        let path = tmp_path("extract_vector.mha");
        write_image(&img, &path).unwrap();

        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        reader.set_extract_size(&[2, 2]).set_extract_index(&[1, 1]);
        let out = reader.execute().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(out.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(out.number_of_components_per_pixel(), 3);
        // Pixels (1,1), (2,1), (1,2), (2,2) -> component offsets 12, 15, 21, 24.
        assert_eq!(
            out.component_slice::<u8>().unwrap(),
            &[12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26]
        );
    }

    // ---- NRRD ------------------------------------------------------------

    /// The header bytes of `NrrdImageIO::Write` for a 3-D scalar image, pinned.
    ///
    /// Field order is teem's `nrrdField` enum order, which is what
    /// `formatNRRD_write` loops over; the two comment lines come from
    /// `nrrd__FormatURLLine0/1` (formatNRRD.c:149-150); the magic is `NRRD0004`
    /// because `nrrd__FormatNRRD_whichVersion` bumps to 4 as soon as `space`
    /// is set, which ITK always does. `endian:` appears only because the
    /// element size exceeds one byte (`nrrd__FieldInteresting`, write.c).
    #[test]
    fn nrrd_header_pins_bytes_for_a_scalar_image() {
        let mut img = Image::from_vec(&[4, 3, 2], (0..24).map(|i| i as f32).collect()).unwrap();
        img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
        img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();

        let path = tmp_path("pin_scalar.nrrd");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let expected = "NRRD0004\n\
                        # Complete NRRD file format specification at:\n\
                        # http://teem.sourceforge.net/nrrd/format.html\n\
                        type: float\n\
                        dimension: 3\n\
                        space: left-posterior-superior\n\
                        sizes: 4 3 2\n\
                        space directions: (0.5,0,0) (0,1.25,0) (0,0,3)\n\
                        kinds: domain domain domain\n\
                        endian: little\n\
                        encoding: raw\n\
                        space origin: (-2,4,7.5)\n\n";
        assert_eq!(&bytes[..expected.len()], expected.as_bytes());
        assert_eq!(bytes.len(), expected.len() + 24 * 4);
    }

    /// A vector image gets a leading `vector` axis, `sizes` grows by one, its
    /// space direction is `none`, and the space becomes a bare `space
    /// dimension: 2` because ITK only names LPS at three domain axes
    /// (itkNrrdImageIO.cxx:1362-1365). `endian:` is absent: `unsigned char`
    /// has element size one.
    #[test]
    fn nrrd_header_pins_bytes_for_a_vector_image() {
        let img = Image::from_vec_vector::<u8>(&[3, 2], 3, (0..18).collect()).unwrap();
        let path = tmp_path("pin_vector.nrrd");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let expected = "NRRD0004\n\
                        # Complete NRRD file format specification at:\n\
                        # http://teem.sourceforge.net/nrrd/format.html\n\
                        type: unsigned char\n\
                        dimension: 3\n\
                        space dimension: 2\n\
                        sizes: 3 3 2\n\
                        space directions: none (1,0) (0,1)\n\
                        kinds: vector domain domain\n\
                        encoding: raw\n\
                        space origin: (0,0)\n\n";
        assert_eq!(&bytes[..expected.len()], expected.as_bytes());
        assert_eq!(bytes.len(), expected.len() + 18);
    }

    #[test]
    fn nrrd_roundtrip_preserves_buffer_and_geometry() {
        let data: Vec<i16> = (0..24).map(|i| i as i16 - 5).collect();
        let mut img = Image::from_vec(&[4, 3, 2], data.clone()).unwrap();
        img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
        img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();
        img.set_direction(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();

        let path = tmp_path("roundtrip.nrrd");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.size(), img.size());
        assert_eq!(back.pixel_id(), PixelId::Int16);
        assert_eq!(back.spacing(), img.spacing());
        assert_eq!(back.origin(), img.origin());
        assert_eq!(back.direction(), img.direction());
        assert_eq!(back.scalar_slice::<i16>().unwrap(), data.as_slice());
        assert_eq!(without_metadata(back), img);
    }

    #[test]
    fn nrrd_roundtrip_all_scalar_types_2d_and_3d() {
        macro_rules! case {
            ($ty:ty, $size:expr, $name:expr) => {{
                let count: usize = $size.iter().product();
                let data: Vec<$ty> = (0..count as u32).map(|i| i as $ty).collect();
                let img = Image::from_vec(&$size, data.clone()).unwrap();
                let path = tmp_path($name);
                write_image(&img, &path).unwrap();
                let back = read_image(&path).unwrap();
                std::fs::remove_file(&path).ok();
                assert_eq!(back.scalar_slice::<$ty>().unwrap(), data.as_slice(), $name);
                assert_eq!(back.size(), &$size[..], $name);
            }};
        }
        macro_rules! both {
            ($ty:ty, $stem:expr) => {{
                case!($ty, [4usize, 2], concat!($stem, "_2d.nrrd"));
                case!($ty, [4usize, 2, 3], concat!($stem, "_3d.nrrd"));
            }};
        }
        both!(u8, "u8");
        both!(i8, "i8");
        both!(u16, "u16");
        both!(i16, "i16");
        both!(u32, "u32");
        both!(i32, "i32");
        both!(u64, "u64");
        both!(i64, "i64");
        both!(f32, "f32");
        both!(f64, "f64");
    }

    #[test]
    fn nrrd_roundtrip_vector_float32() {
        let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25 - 4.0).collect();
        let mut img = Image::from_vec_vector::<f32>(&[4, 3], 3, data.clone()).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        img.set_direction(&[0.0, 1.0, -1.0, 0.0]).unwrap();

        let path = tmp_path("vector_f32.nrrd");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.component_slice::<f32>().unwrap(), data.as_slice());
        assert_eq!(without_metadata(back), img);
    }

    /// Unlike MetaImage, NRRD records the complex-ness in `kinds`, so
    /// `ComplexFloat32` round-trips to itself: `nrrdKindComplex` maps to
    /// `IOPixelEnum::COMPLEX` (itkNrrdImageIO.cxx:750-753), which
    /// `GetPixelIDFromImageIO`'s third branch turns back into the complex pixel
    /// id (sitkImageReaderBase.cxx:234-238).
    #[test]
    fn nrrd_roundtrips_complex_as_complex() {
        macro_rules! case {
            ($ty:ty, $id:expr, $name:expr) => {{
                let data: Vec<Complex<$ty>> = (0..6)
                    .map(|i| Complex::new(i as $ty, -(i as $ty) * 0.5))
                    .collect();
                let mut img = Image::from_vec_complex::<$ty>(&[3, 2], data.clone()).unwrap();
                img.set_spacing(&[0.5, 2.0]).unwrap();
                let path = tmp_path($name);
                write_image(&img, &path).unwrap();
                let raw = std::fs::read(&path).unwrap();
                let end = raw.windows(2).position(|w| w == b"\n\n").unwrap();
                let header = String::from_utf8_lossy(&raw[..end]).to_string();
                let back = read_image(&path).unwrap();
                std::fs::remove_file(&path).ok();

                assert!(header.contains("kinds: complex domain domain"), "{header}");
                assert_eq!(back.pixel_id(), $id, $name);
                assert_eq!(back.number_of_components_per_pixel(), 1, $name);
                assert_eq!(without_metadata(back), img, $name);
            }};
        }
        case!(f32, PixelId::ComplexFloat32, "complex32.nrrd");
        case!(f64, PixelId::ComplexFloat64, "complex64.nrrd");
    }

    /// `nrrdSave` turns a `.nhdr` filename into a detached header naming
    /// `<stem>.<encoding suffix>`, always header-relative. The header has no
    /// blank-line terminator, because there is no attached data to separate.
    #[test]
    fn nhdr_writes_separate_raw_and_reads_back() {
        let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let img = Image::from_vec(&[3, 2], data.clone()).unwrap();
        let path = tmp_path("pair.nhdr");
        let raw = path.with_file_name(format!("sitk_io_test_{}_pair.raw", std::process::id()));

        write_image(&img, &path).unwrap();
        let header = std::fs::read_to_string(&path).unwrap();
        assert!(raw.exists());
        assert!(
            header.ends_with(
                "data file: sitk_io_test_{pid}_pair.raw\n"
                    .replace("{pid}", &std::process::id().to_string())
                    .as_str()
            ),
            "{header}"
        );
        assert_eq!(std::fs::read(&raw).unwrap().len(), 24);

        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&raw).ok();
        assert_eq!(back.scalar_slice::<f32>().unwrap(), data.as_slice());
        assert_eq!(without_metadata(back), img);
    }

    /// `nrrdSpacingCalculate`'s `Direction` status: the spacing is the norm of
    /// the space-direction vector and the direction column is that vector
    /// normalised (axis.c:946-949, itkNrrdImageIO.cxx:807-818).
    #[test]
    fn nrrd_space_directions_decompose_into_spacing_and_direction() {
        // Two orthonormal columns rotated 3-4-5, scaled by 2 and 10.
        let path = tmp_path("skew.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: unsigned char\n\
              dimension: 2\n\
              space dimension: 2\n\
              sizes: 2 2\n\
              space directions: (1.2,1.6) (-8,6)\n\
              kinds: domain domain\n\
              encoding: raw\n\
              space origin: (3,-4)\n\
              \n\
              \x01\x02\x03\x04",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.spacing(), &[2.0, 10.0]);
        assert_eq!(img.direction(), &[0.6, -0.8, 0.8, 0.6]);
        assert_eq!(img.origin(), &[3.0, -4.0]);
        assert_eq!(img.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
    }

    /// `space: right-anterior-superior` flips the first two axis directions and
    /// the first two origin coefficients, and the dictionary reports the space
    /// as `left-posterior-superior` because the conversion happened
    /// (itkNrrdImageIO.cxx:767-786, 1022-1029).
    #[test]
    fn nrrd_ras_space_is_converted_to_lps() {
        let path = tmp_path("ras.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: unsigned char\n\
              dimension: 3\n\
              space: right-anterior-superior\n\
              sizes: 1 1 1\n\
              space directions: (2,0,0) (0,3,0) (0,0,4)\n\
              kinds: domain domain domain\n\
              encoding: raw\n\
              space origin: (10,20,30)\n\
              \n\
              \x07",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.spacing(), &[2.0, 3.0, 4.0]);
        assert_eq!(
            img.direction(),
            &[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(img.origin(), &[-10.0, -20.0, 30.0]);
        assert_eq!(img.meta_data("NRRD_space"), Some("left-posterior-superior"));
    }

    /// `scanner-xyz` has no well-defined LPS conversion, so
    /// `ReadImageInformation`'s `switch` falls to `default:` and the direction
    /// vectors survive unconverted — the space is *not* rejected. Ledger §2.82.
    #[test]
    fn nrrd_scanner_xyz_space_is_left_unconverted() {
        let path = tmp_path("scanner.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: unsigned char\n\
              dimension: 3\n\
              space: scanner-xyz\n\
              sizes: 1 1 1\n\
              space directions: (1,0,0) (0,1,0) (0,0,1)\n\
              kinds: domain domain domain\n\
              encoding: raw\n\
              space origin: (10,20,30)\n\
              \n\
              \x07",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.origin(), &[10.0, 20.0, 30.0]);
        assert_eq!(img.meta_data("NRRD_space"), Some("scanner-xyz"));
    }

    /// `kinds: domain domain vector` puts the pixel axis last, so
    /// `GetAxisOrderForFileReading` reports `needPermutation` and `Read`
    /// permutes it to axis 0 (itkNrrdImageIO.cxx:1146-1170).
    #[test]
    fn nrrd_permutes_a_trailing_pixel_axis_to_the_front() {
        // sizes 2 2 3: axis 0 fastest, so the on-disk order is
        // (x, y, component). Component c of pixel (x,y) is at x + 2*y + 4*c.
        let mut data = Vec::new();
        for c in 0..3u8 {
            for y in 0..2u8 {
                for x in 0..2u8 {
                    data.push(100 * c + 10 * y + x);
                }
            }
        }
        let path = tmp_path("permute.nrrd");
        let mut bytes = b"NRRD0004\n\
              type: unsigned char\n\
              dimension: 3\n\
              space dimension: 2\n\
              sizes: 2 2 3\n\
              space directions: (1,0) (0,1) none\n\
              kinds: domain domain vector\n\
              encoding: raw\n\
              space origin: (0,0)\n\n"
            .to_vec();
        bytes.extend_from_slice(&data);
        std::fs::write(&path, bytes).unwrap();

        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(img.size(), &[2, 2]);
        assert_eq!(img.number_of_components_per_pixel(), 3);
        // Interleaved: pixel (0,0) is [0, 100, 200], pixel (1,0) is [1, 101, 201].
        assert_eq!(
            img.component_slice::<u8>().unwrap(),
            &[0, 100, 200, 1, 101, 201, 10, 110, 210, 11, 111, 211]
        );
        assert_eq!(img.meta_data("NRRD_pixel_original_axis"), Some("2"));
    }

    /// `kinds: list domain domain` has no non-list range axis, so
    /// `UseAnyRangeAxisAsPixel` takes the list axis as the pixel component axis
    /// and the image is a vector image, not a 3-D scalar one
    /// (itkNrrdImageIO.cxx:78-82, 731-736).
    #[test]
    fn nrrd_leading_list_axis_becomes_the_pixel_axis() {
        let path = tmp_path("list_kind.nrrd");
        let mut bytes = b"NRRD0004\n\
              type: unsigned char\n\
              dimension: 3\n\
              space dimension: 2\n\
              sizes: 2 2 2\n\
              space directions: none (1,0) (0,1)\n\
              kinds: list domain domain\n\
              encoding: raw\n\
              space origin: (0,0)\n\n"
            .to_vec();
        bytes.extend_from_slice(&(0..8u8).collect::<Vec<_>>());
        std::fs::write(&path, bytes).unwrap();

        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(img.size(), &[2, 2]);
        assert_eq!(img.number_of_components_per_pixel(), 2);
    }

    /// With no `space directions`, `nrrdSpacingCalculate` reports
    /// `ScalarNoSpace` for `spacings` and `nrrdOriginCalculate` derives the
    /// origin from `axis mins` (cell-centered by default, so half a sample in).
    /// `axis mins` / `axis maxs` never touch the spacing.
    #[test]
    fn nrrd_spacings_and_axis_mins_set_spacing_and_origin_separately() {
        let path = tmp_path("mins.nrrd");
        std::fs::write(
            &path,
            b"NRRD0001\n\
              type: unsigned char\n\
              dimension: 2\n\
              sizes: 2 2\n\
              spacings: 2 4\n\
              axis mins: 10 20\n\
              encoding: raw\n\
              \n\
              \x01\x02\x03\x04",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.spacing(), &[2.0, 4.0]);
        assert_eq!(img.origin(), &[11.0, 22.0]);
    }

    /// Without `axis mins` the origin status is `NoMin` and ITK leaves the
    /// origin at zero (itkNrrdImageIO.cxx:905-912).
    #[test]
    fn nrrd_no_axis_mins_leaves_the_origin_at_zero() {
        let path = tmp_path("nomins.nrrd");
        std::fs::write(
            &path,
            b"NRRD0001\n\
              type: unsigned char\n\
              dimension: 2\n\
              sizes: 2 2\n\
              spacings: 2 4\n\
              encoding: raw\n\
              \n\
              \x01\x02\x03\x04",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.origin(), &[0.0, 0.0]);
    }

    /// Fixed §1.47: upstream `nrrdOriginCalculate`'s `gotMin` loop read
    /// `axis[0]->min` on every iteration instead of `axis[ai]->min`, so a NaN
    /// on axis 1 did not produce the `NoMin` status it should have — the NaN
    /// reached the origin instead. This port checks each axis's own `min`, so
    /// a NaN on *any* axis reports `NoMin` and the origin — for every axis,
    /// including the one with a real `min` — is left at zero.
    #[test]
    fn nrrd_axis_mins_missing_on_any_axis_leaves_the_origin_at_zero() {
        let path = tmp_path("nanmin.nrrd");
        std::fs::write(
            &path,
            b"NRRD0001\n\
              type: unsigned char\n\
              dimension: 2\n\
              sizes: 2 2\n\
              spacings: 2 4\n\
              axis mins: 10 nan\n\
              encoding: raw\n\
              \n\
              \x01\x02\x03\x04",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(
            img.origin(),
            &[0.0, 0.0],
            "a NaN min on axis 1 must void the whole origin, not just axis 1"
        );
    }

    /// `encoding: ASCII` (whose header spelling is what teem's
    /// `nrrdEncodingAscii->name` says) is read but never written by ITK.
    /// Values are whitespace-separated and narrow integer types go through
    /// `sscanf("%d")` into an `int` before being C-cast down.
    #[test]
    fn nrrd_reads_ascii_encoding() {
        let path = tmp_path("ascii.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: short\n\
              dimension: 2\n\
              sizes: 2 2\n\
              encoding: ASCII\n\
              \n\
              -1 2\n3 -4\n",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.scalar_slice::<i16>().unwrap(), &[-1, 2, 3, -4]);
    }

    /// `formatNRRD_read` runs `nrrdLineSkip` then `nrrd__ByteSkipSkip` for every
    /// non-compression encoding (formatNRRD.c:577-605), so a positive
    /// `byte skip` advances into the ascii text just as it does into raw bytes.
    #[test]
    fn nrrd_ascii_honours_a_positive_byte_skip() {
        let path = tmp_path("ascii_byteskip.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: short\n\
              dimension: 2\n\
              sizes: 2 2\n\
              encoding: ascii\n\
              byte skip: 6\n\
              \n\
              XXXXXX-1 2\n3 -4\n",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.scalar_slice::<i16>().unwrap(), &[-1, 2, 3, -4]);
    }

    /// `nrrd__ByteSkipSkip` refuses a backwards byte skip for any encoding but
    /// raw (read.c:320-327).
    #[test]
    fn nrrd_ascii_rejects_a_negative_byte_skip() {
        let path = tmp_path("ascii_backskip.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: short\n\
              dimension: 2\n\
              sizes: 2 2\n\
              encoding: ascii\n\
              byte skip: -1\n\
              \n\
              -1 2\n3 -4\n",
        )
        .unwrap();
        let err = read_image(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(err, IoError::MalformedNrrdHeader(ref m) if m.contains("backwards byte skip")),
            "{err}"
        );
    }

    /// `data file: LIST` names one file per line after the field; the file
    /// count must equal the product of the sizes at and above `dataFileDim`
    /// (`nrrdIoDataFNCheck`).
    #[test]
    fn nrrd_reads_a_data_file_list() {
        let path = tmp_path("list.nhdr");
        let a = path.with_file_name(format!("sitk_io_test_{}_list_a.raw", std::process::id()));
        let b = path.with_file_name(format!("sitk_io_test_{}_list_b.raw", std::process::id()));
        std::fs::write(&a, [1u8, 2, 3, 4]).unwrap();
        std::fs::write(&b, [5u8, 6, 7, 8]).unwrap();
        std::fs::write(
            &path,
            format!(
                "NRRD0004\n\
                 type: unsigned char\n\
                 dimension: 3\n\
                 sizes: 2 2 2\n\
                 encoding: raw\n\
                 data file: LIST\n\
                 sitk_io_test_{pid}_list_a.raw\n\
                 sitk_io_test_{pid}_list_b.raw\n",
                pid = std::process::id()
            ),
        )
        .unwrap();

        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
        assert_eq!(img.size(), &[2, 2, 2]);
        assert_eq!(img.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    /// `line skip` consumes whole lines from the front of the data, then
    /// `byte skip` moves forward from there (read.c).
    #[test]
    fn nrrd_honours_line_skip_and_byte_skip() {
        let path = tmp_path("skips.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: unsigned char\n\
              dimension: 1\n\
              sizes: 3\n\
              line skip: 1\n\
              byte skip: 2\n\
              encoding: raw\n\
              \n\
              junk line\nXX\x0a\x14\x1e",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.scalar_slice::<u8>().unwrap(), &[10, 20, 30]);
    }

    /// Big-endian raw data is byte-swapped after reading (`nrrdSwapEndian`).
    #[test]
    fn nrrd_big_endian_raw_is_swapped() {
        let path = tmp_path("bigend.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: short\n\
              dimension: 1\n\
              sizes: 2\n\
              endian: big\n\
              encoding: raw\n\
              \n\
              \x01\x02\x03\x04",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.scalar_slice::<i16>().unwrap(), &[0x0102, 0x0304]);
    }

    /// `bzip2` is still recognised and rejected by name: this workspace takes a
    /// zlib dependency but not a bzip2 one (ledger §5.8). `gzip`/`gz` are read
    /// — see `tests/compression.rs`.
    #[test]
    fn nrrd_rejects_the_bzip2_encoding() {
        let path = tmp_path("compressed_bzip2.nrrd");
        std::fs::write(
            &path,
            "NRRD0004\ntype: unsigned char\ndimension: 1\nsizes: 2\n\
             encoding: bzip2\n\nxx",
        )
        .unwrap();
        let err = read_image(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        match err {
            IoError::UnsupportedNrrdFeature(message) => {
                assert!(message.contains("bzip2"), "{message}");
            }
            other => panic!("expected UnsupportedNrrdFeature, got {other:?}"),
        }
    }

    /// `ReadImageInformation` raises "Cannot currently handle nrrdTypeBlock"
    /// (itkNrrdImageIO.cxx:617-620).
    #[test]
    fn nrrd_rejects_the_block_type() {
        let path = tmp_path("block.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\ntype: block\nblock size: 4\ndimension: 1\nsizes: 2\n\
              endian: little\nencoding: raw\n\nxxxxxxxx",
        )
        .unwrap();
        let err = read_image(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(
            err,
            IoError::UnsupportedNrrdFeature(ref m) if m.contains("nrrdTypeBlock")
        ));
    }

    /// A `3D-symmetric-matrix` pixel axis is `IOPixelEnum::SYMMETRICSECONDRANKTENSOR`,
    /// which falls off the end of `GetPixelIDFromImageIO`'s if-ladder. Ledger §3.31.
    #[test]
    fn nrrd_rejects_a_symmetric_matrix_pixel_axis() {
        let path = tmp_path("tensor.nrrd");
        let mut bytes = b"NRRD0004\n\
              type: float\n\
              dimension: 2\n\
              sizes: 6 2\n\
              kinds: 3D-symmetric-matrix domain\n\
              endian: little\n\
              encoding: raw\n\n"
            .to_vec();
        bytes.extend_from_slice(&[0u8; 48]);
        std::fs::write(&path, bytes).unwrap();
        let err = read_image(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(
            err,
            IoError::UnsupportedNrrdFeature(ref m) if m.contains("Unknown PixelType")
        ));
    }

    /// `nrrd__HeaderCheck` refuses a multi-byte raw type with no `endian`
    /// field (simple.c).
    #[test]
    fn nrrd_requires_endian_for_multibyte_raw() {
        let path = tmp_path("noendian.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\ntype: short\ndimension: 1\nsizes: 2\nencoding: raw\n\n\x01\x02\x03\x04",
        )
        .unwrap();
        let err = read_image(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(err, IoError::MalformedNrrdHeader(_)), "{err:?}");
    }

    /// Key/value lines survive into the dictionary, `airUnescape`d, and a
    /// duplicate non-comment field is an error (formatNRRD.c:475-478).
    #[test]
    fn nrrd_key_value_pairs_and_duplicate_fields() {
        let path = tmp_path("kvp.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\ntype: unsigned char\ndimension: 1\nsizes: 2\nencoding: raw\n\
              patient:=Doe\\nJane\n\nxx",
        )
        .unwrap();
        let img = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(img.meta_data("patient"), Some("Doe\nJane"));

        let path = tmp_path("dup.nrrd");
        std::fs::write(
            &path,
            b"NRRD0004\ntype: unsigned char\ntype: short\ndimension: 1\nsizes: 2\n\
              encoding: raw\n\nxx",
        )
        .unwrap();
        let err = read_image(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(
            err,
            IoError::MalformedNrrdHeader(ref m) if m.contains("already set field")
        ));
    }

    /// `read_information` parses the header and never touches the pixels: this
    /// `.nhdr` names a data file that does not exist.
    #[test]
    fn nrrd_read_information_does_not_need_the_data_file() {
        let path = tmp_path("info.nhdr");
        std::fs::write(
            &path,
            b"NRRD0004\n\
              type: double\n\
              dimension: 3\n\
              space: left-posterior-superior\n\
              sizes: 100000 100000 1000\n\
              space directions: (1,0,0) (0,1,0) (0,0,1)\n\
              kinds: domain domain domain\n\
              endian: little\n\
              encoding: raw\n\
              space origin: (0,0,0)\n\
              data file: absent.raw\n",
        )
        .unwrap();
        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        let info = reader.read_image_information().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(info.pixel_id, PixelId::Float64);
        assert_eq!(info.size, vec![100_000, 100_000, 1000]);
        assert_eq!(info.spacing, vec![1.0, 1.0, 1.0]);
    }

    /// `can_read_file` needs both a supported extension and the `NRRD` magic;
    /// a `.nrrd` file that is not a NRRD is claimed by nobody.
    #[test]
    fn nrrd_extension_alone_does_not_claim_a_file_for_reading() {
        let path = tmp_path("not_really.nrrd");
        std::fs::write(&path, b"this is a text file, not a NRRD\n").unwrap();
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        std::fs::remove_file(&path).ok();
        assert!(!claimed);
    }

    // ======================================================================
    // GIPL — itk::GiplImageIO
    // ======================================================================

    /// Hand-author the 256-byte header `ReadImageInformation` walks, so a test
    /// can move one field at a time.
    fn gipl_header(
        dims: [u16; 4],
        image_type: u16,
        pixdim: [f32; 4],
        origin: [f64; 4],
        magic: u32,
    ) -> Vec<u8> {
        let mut h = Vec::with_capacity(256);
        for d in dims {
            h.extend_from_slice(&d.to_be_bytes());
        }
        h.extend_from_slice(&image_type.to_be_bytes());
        for p in pixdim {
            h.extend_from_slice(&p.to_be_bytes());
        }
        h.resize(h.len() + 80, 0); // line1
        h.resize(h.len() + 98, 0); // matrix, flag1, flag2, min, max
        for o in origin {
            h.extend_from_slice(&o.to_be_bytes());
        }
        h.resize(h.len() + 16, 0); // pixval_offset, pixval_cal, user_def1, user_def2
        h.extend_from_slice(&magic.to_be_bytes());
        assert_eq!(h.len(), gipl::HEADER_SIZE);
        h
    }

    /// Every byte `GiplImageIO::Write` emits before the pixel data
    /// (itkGiplImageIO.cxx:684-991): four big-endian `dims` with the unused
    /// axes padded to `1`, `image_type`, four `pixdim` floats padded to `1.0`,
    /// the fixed `"No Patient Information"` text, 98 zero bytes covering the
    /// discarded `matrix`/`flag1`/`flag2`/`min`/`max`, four `origin` doubles,
    /// 16 more zero bytes, and `GIPL_MAGIC_NUMBER` at offset 252.
    #[test]
    fn gipl_header_pins_bytes_for_a_3d_image() {
        let data: Vec<i16> = (0..24).map(|i| i as i16 - 5).collect();
        let mut img = Image::from_vec(&[4, 3, 2], data).unwrap();
        img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
        img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();

        let path = tmp_path("pin.gipl");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let mut expected = gipl_header(
            [4, 3, 2, 1],
            15, // GIPL_SHORT
            [0.5, 1.25, 3.0, 1.0],
            [-2.0, 4.0, 7.5, 0.0],
            gipl::GIPL_MAGIC_NUMBER,
        );
        expected[26..26 + 22].copy_from_slice(b"No Patient Information");

        assert_eq!(&bytes[..gipl::HEADER_SIZE], &expected[..]);
        assert_eq!(bytes.len(), gipl::HEADER_SIZE + 24 * 2);
        assert_eq!(&bytes[252..256], &[0xef, 0xff, 0xe9, 0xb0]);
        // The pixel data is big-endian: -5 is 0xfffb.
        assert_eq!(&bytes[256..258], &[0xff, 0xfb]);
    }

    /// The six component types `SwapBytesIfNecessary` has an arm for, at 2-D
    /// and 3-D. A 2-D image comes back **3-D** — `Write` pads `dims` with `1`
    /// and `ReadImageInformation` counts every non-zero slot below index 3
    /// (§2.94) — but the pixel values are exact.
    #[test]
    fn gipl_roundtrip_every_supported_scalar_type_2d_and_3d() {
        macro_rules! case {
            ($ty:ty, $size:expr, $expected_size:expr, $name:expr) => {{
                let count: usize = $size.iter().product();
                let data: Vec<$ty> = (0..count as u32).map(|i| i as $ty).collect();
                let img = Image::from_vec(&$size, data.clone()).unwrap();
                let path = tmp_path($name);
                write_image(&img, &path).unwrap();
                let back = read_image(&path).unwrap();
                std::fs::remove_file(&path).ok();
                assert_eq!(back.scalar_slice::<$ty>().unwrap(), data.as_slice(), $name);
                assert_eq!(back.size(), &$expected_size[..], $name);
            }};
        }
        macro_rules! both {
            ($ty:ty, $stem:expr) => {{
                case!($ty, [4usize, 2], [4usize, 2, 1], concat!($stem, "_2d.gipl"));
                case!(
                    $ty,
                    [4usize, 2, 3],
                    [4usize, 2, 3],
                    concat!($stem, "_3d.gipl")
                );
            }};
        }
        both!(u8, "gipl_u8");
        both!(i8, "gipl_i8");
        both!(u16, "gipl_u16");
        both!(i16, "gipl_i16");
        both!(f32, "gipl_f32");
        both!(f64, "gipl_f64");
    }

    /// Spacing and origin survive; the direction matrix does not exist in the
    /// format and reads back as the identity.
    #[test]
    fn gipl_roundtrip_preserves_spacing_and_origin_and_drops_direction() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
        let mut img = Image::from_vec(&[4, 3, 2], data.clone()).unwrap();
        img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
        img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();
        img.set_direction(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();

        let path = tmp_path("geom.gipl");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.size(), &[4, 3, 2]);
        assert_eq!(back.pixel_id(), PixelId::Float32);
        assert_eq!(back.spacing(), &[0.5, 1.25, 3.0]);
        assert_eq!(back.origin(), &[-2.0, 4.0, 7.5]);
        assert_eq!(
            back.direction(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(back.scalar_slice::<f32>().unwrap(), data.as_slice());
        // GIPL itself puts nothing in the dictionary, but the reader's geometry
        // normalization records the (positive, so unflipped) spacing and the
        // identity direction it read back. `pixdim` is `float`, so a spacing not
        // exactly representable in f32 would not survive; these three are.
        assert_eq!(
            back.meta_data_keys(),
            vec!["ITK_original_direction", "ITK_original_spacing"]
        );
        assert_eq!(back.meta_data("ITK_original_spacing"), Some("0.5 1.25 3"));
        assert_eq!(
            back.meta_data("ITK_original_direction"),
            Some("1 0 0 0 1 0 0 0 1")
        );
    }

    /// The unit third axis a 2-D write invents carries spacing `1.0` and origin
    /// `0.0` — `Write`'s `else` arms (itkGiplImageIO.cxx:807, :910).
    #[test]
    fn gipl_two_dimensional_image_reads_back_as_three_dimensional() {
        let mut img = Image::from_vec(&[3, 2], vec![1u8, 2, 3, 4, 5, 6]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[1.0, -1.0]).unwrap();

        let path = tmp_path("two_d.gipl");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.dimension(), 3);
        assert_eq!(back.size(), &[3, 2, 1]);
        assert_eq!(back.spacing(), &[0.5, 2.0, 1.0]);
        assert_eq!(back.origin(), &[1.0, -1.0, 0.0]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    /// Fixed §1.52: `SwapBytesIfNecessary` has no `INT`/`UINT` arm, so upstream's
    /// `Write` throws `"Pixel Type Unknown"` only *after* the full 256-byte
    /// header is on disk. This port now checks swappability before writing
    /// anything at all, so a pre-existing file at the target path is left
    /// completely untouched.
    #[test]
    fn gipl_write_of_int32_is_rejected_before_the_file_is_touched() {
        let img = Image::from_vec(&[2, 2], vec![1i32, 2, 3, 4]).unwrap();
        let path = tmp_path("int32.gipl");
        std::fs::write(&path, b"pre-existing content that must survive").unwrap();

        let result = write_image(&img, &path);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.starts_with("Pixel Type Unknown")),
            "{result:?}"
        );
        assert_eq!(bytes, b"pre-existing content that must survive");
    }

    /// The read side of the missing `SwapBytesIfNecessary` arm is untouched by
    /// §1.52's write-side fix: a hand-built `GIPL_INT` file — the shape no
    /// writer (including this port's own, now) ever produces — still fails to
    /// read.
    #[test]
    fn gipl_read_of_int32_fails_on_the_missing_swap_arm() {
        let path = tmp_path("int32_read.gipl");
        let mut bytes = gipl_header(
            [2, 2, 1, 1],
            32,
            [1.0; 4],
            [0.0; 4],
            gipl::GIPL_MAGIC_NUMBER,
        );
        for v in [1i32, 2, 3, 4] {
            bytes.extend_from_slice(&v.to_be_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();

        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.starts_with("Pixel Type Unknown")),
            "{result:?}"
        );
    }

    /// A 64-bit integer has no `image_type` at all, and `Write`'s own switch
    /// throws `"Invalid type"` after only the four `dims` values are out
    /// (itkGiplImageIO.cxx:759-761) — an 8-byte file.
    #[test]
    fn gipl_int64_write_leaves_eight_bytes_and_then_fails() {
        let img = Image::from_vec(&[2, 2], vec![1i64, 2, 3, 4]).unwrap();
        let path = tmp_path("int64.gipl");
        let result = write_image(&img, &path);
        let written = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.starts_with("Invalid type")),
            "{result:?}"
        );
        assert_eq!(written, [0, 2, 0, 2, 0, 1, 0, 1]);
    }

    /// Fixed §1.52, `.gipl.gz` counterpart of
    /// `gipl_write_of_int32_is_rejected_before_the_file_is_touched`: the
    /// rejection happens before `gzopen` is ever called, so a pre-existing
    /// `.gipl.gz` at the target path is left completely untouched too.
    #[test]
    fn gipl_gz_write_of_int32_is_rejected_before_the_file_is_touched() {
        let img = Image::from_vec(&[2, 2], vec![1i32, 2, 3, 4]).unwrap();
        let path = tmp_path("int32.gipl.gz");
        std::fs::write(&path, b"pre-existing content that must survive").unwrap();

        let result = write_image(&img, &path);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.starts_with("Pixel Type Unknown")),
            "{result:?}"
        );
        assert_eq!(bytes, b"pre-existing content that must survive");
    }

    /// The `.gipl.gz` counterpart of `gipl_int64_write_leaves_eight_bytes_and_then_fails`.
    #[test]
    fn gipl_gz_int64_write_leaves_eight_bytes_and_then_fails() {
        let img = Image::from_vec(&[2, 2], vec![1i64, 2, 3, 4]).unwrap();
        let path = tmp_path("int64.gipl.gz");
        let result = write_image(&img, &path);
        let written =
            crate::compression::gunzip_transparent(&std::fs::read(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.starts_with("Invalid type")),
            "{result:?}"
        );
        assert_eq!(written, [0, 2, 0, 2, 0, 1, 0, 1]);
    }

    /// `CheckExtension` claims `.gipl.gz` for reading and writing, and `write`
    /// now compresses through [`crate::compression`] instead of refusing
    /// (ledger §4.68, closed).
    #[test]
    fn gipl_gz_is_recognised_and_round_trips() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("compressed.gipl.gz");

        assert!(create_image_io(&path, FileMode::Write).is_some());
        write_image(&img, &path).unwrap();
        assert_eq!(&std::fs::read(&path).unwrap()[..2], b"\x1f\x8b");

        assert!(create_image_io(&path, FileMode::Read).is_some());
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        // `.gipl` reads back 2-D as 3-D (§2.94); `.gipl.gz` is no exception.
        assert_eq!(back.size(), &[2, 2, 1]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4]);
    }

    /// The six component types `SwapBytesIfNecessary` has an arm for, at 2-D
    /// and 3-D, through `.gipl.gz` — the gzip-door counterpart of
    /// `gipl_roundtrip_every_supported_scalar_type_2d_and_3d`.
    #[test]
    fn gipl_gz_roundtrip_every_supported_scalar_type_2d_and_3d() {
        macro_rules! case {
            ($ty:ty, $size:expr, $expected_size:expr, $name:expr) => {{
                let count: usize = $size.iter().product();
                let data: Vec<$ty> = (0..count as u32).map(|i| i as $ty).collect();
                let img = Image::from_vec(&$size, data.clone()).unwrap();
                let path = tmp_path($name);
                write_image(&img, &path).unwrap();
                assert_eq!(&std::fs::read(&path).unwrap()[..2], b"\x1f\x8b", $name);
                let back = read_image(&path).unwrap();
                std::fs::remove_file(&path).ok();
                assert_eq!(back.scalar_slice::<$ty>().unwrap(), data.as_slice(), $name);
                assert_eq!(back.size(), &$expected_size[..], $name);
            }};
        }
        macro_rules! both {
            ($ty:ty, $stem:expr) => {{
                case!(
                    $ty,
                    [4usize, 2],
                    [4usize, 2, 1],
                    concat!($stem, "_2d.gipl.gz")
                );
                case!(
                    $ty,
                    [4usize, 2, 3],
                    [4usize, 2, 3],
                    concat!($stem, "_3d.gipl.gz")
                );
            }};
        }
        both!(u8, "gipl_gz_u8");
        both!(i8, "gipl_gz_i8");
        both!(u16, "gipl_gz_u16");
        both!(i16, "gipl_gz_i16");
        both!(f32, "gipl_gz_f32");
        both!(f64, "gipl_gz_f64");
    }

    /// `gzopen(m_FileName.c_str(), "wb")` (itkGiplImageIO.cxx:671) names no
    /// level, so `.gipl.gz` always compresses at zlib's `Z_DEFAULT_COMPRESSION`
    /// (6) — `MTIME = 0`, `XFL = 0`, `OS = 3` — regardless of `WriteOptions`.
    /// Same precedent as NIfTI (ledger §3.40), extended to GIPL at §3.42.
    #[test]
    fn gipl_gz_write_ignores_use_compression_and_compression_level() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("framing.gipl.gz");
        // `use_compression = false` and an explicit level are both ignored: a
        // `.gipl.gz` is always gzipped, and always at level 6.
        write_image_with(&img, &path, false, 9).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(&bytes[..2], b"\x1f\x8b");
        assert_eq!(&bytes[4..10], &[0, 0, 0, 0, 0x00, 0x03], "MTIME, XFL, OS");
    }

    /// zlib's `gz_look` falls back to a transparent byte-for-byte copy when the
    /// gzip magic is absent (ledger §2.113), same as NRRD and NIfTI — extended
    /// to GIPL at §2.118. A `.gipl.gz` holding a plain, uncompressed GIPL file
    /// reads it verbatim.
    #[test]
    fn gipl_gz_over_a_non_gzip_stream_reads_transparently() {
        let plain = tmp_path("plain_for_transparent.gipl");
        let img = Image::from_vec(&[2, 2], vec![9u8, 8, 7, 6]).unwrap();
        write_image(&img, &plain).unwrap();
        let plain_bytes = std::fs::read(&plain).unwrap();
        std::fs::remove_file(&plain).ok();

        let path = tmp_path("misnamed.gipl.gz");
        std::fs::write(&path, &plain_bytes).unwrap();
        assert!(create_image_io(&path, FileMode::Read).is_some());
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[9, 8, 7, 6]);
    }

    /// A `.gipl.gz` this port never wrote — one stored (uncompressed) deflate
    /// block over the whole 256-byte header and the pixels — reads, because
    /// that is what `gzread` would deliver. No byte of the fixture came from
    /// this crate's own encoder.
    #[test]
    fn gipl_gz_reads_a_hand_built_stored_block_gzip_stream() {
        let mut bytes = gipl_header([2, 2, 1, 1], 8, [1.0; 4], [0.0; 4], gipl::GIPL_MAGIC_NUMBER);
        bytes.extend_from_slice(&[7, 8, 9, 10]);
        let fixture = crate::compression::stored_block_gzip(&bytes);

        let path = tmp_path("fixture.gipl.gz");
        std::fs::write(&path, &fixture).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[7, 8, 9, 10]);
    }

    /// `Read`'s compressed-path success check is `success = p != nullptr`
    /// (itkGiplImageIO.cxx:219-223) — the output *pointer*, never the byte
    /// count `gzread` actually delivered — so a short pixel block reports
    /// success upstream (§1.58). This port refuses it, extending §4.69's
    /// closure to the compressed path at §4.79.
    #[test]
    fn gipl_gz_truncated_pixel_data_is_an_error() {
        let mut bytes = gipl_header([4, 4, 1, 1], 8, [1.0; 4], [0.0; 4], gipl::GIPL_MAGIC_NUMBER);
        bytes.extend_from_slice(&[1, 2, 3]); // 16 pixels declared, 3 present
        let fixture = crate::compression::stored_block_gzip(&bytes);

        let path = tmp_path("gz_short.gipl.gz");
        std::fs::write(&path, &fixture).unwrap();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(matches!(result, Err(IoError::TruncatedData)), "{result:?}");
    }

    /// A gzip stream with the right magic but a payload that will not inflate
    /// is `IoError::CorruptCompressedData`, not the vacuously-`true` success
    /// upstream's `p != nullptr` check reports (§1.58). `can_read_file`'s own
    /// gzread-through-the-magic probe fails the same way — it decompresses the
    /// same corrupt bytes — so the registry never reaches `gipl::read` for this
    /// file either, exactly as `gipl_short_header_is_an_error` pins for the
    /// uncompressed short-header case.
    #[test]
    fn gipl_gz_corrupt_stream_is_an_error() {
        let path = tmp_path("corrupt.gipl.gz");
        std::fs::write(&path, b"\x1f\x8b not really gzip").unwrap();
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = gipl::read(&path);
        std::fs::remove_file(&path).ok();
        assert!(!claimed);
        assert!(
            matches!(&result, Err(IoError::CorruptCompressedData(_))),
            "{result:?}"
        );
    }

    /// `numberofdimension` is a population count over the first three `dims`
    /// slots, not the length of their leading non-zero run, while
    /// `m_Dimensions[i] = dims[i]` copies the *first* `NDims` slots
    /// (itkGiplImageIO.cxx:294-312). `[4, 0, 5, 1]` therefore yields a 2-D image
    /// sized `[4, 0]` — the `5` is counted and then never read. §2.94.
    #[test]
    fn gipl_dimension_count_is_a_population_count_not_a_leading_run() {
        let path = tmp_path("popcount.gipl");
        let header = gipl_header(
            [4, 0, 5, 1],
            8, // GIPL_U_CHAR
            [2.0, 3.0, 4.0, 1.0],
            [10.0, 20.0, 30.0, 0.0],
            gipl::GIPL_MAGIC_NUMBER,
        );
        std::fs::write(&path, &header).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.dimension(), 2);
        assert_eq!(back.size(), &[4, 0]);
        assert_eq!(back.spacing(), &[2.0, 3.0]);
        assert_eq!(back.origin(), &[10.0, 20.0]);
        assert_eq!(back.number_of_pixels(), 0);
    }

    /// `CanReadFile` accepts either magic number, at offset 252
    /// (itkGiplImageIO.cxx:135). `ReadImageInformation` re-reads the same four
    /// bytes and never compares them (§2.93), so the second variant reads fine.
    #[test]
    fn gipl_both_magic_numbers_are_accepted_and_a_wrong_one_is_not() {
        for (magic, label) in [
            (gipl::GIPL_MAGIC_NUMBER, "magic1.gipl"),
            (gipl::GIPL_MAGIC_NUMBER2, "magic2.gipl"),
        ] {
            let path = tmp_path(label);
            let mut bytes = gipl_header([2, 2, 1, 1], 8, [1.0; 4], [0.0; 4], magic);
            bytes.extend_from_slice(&[7, 8, 9, 10]);
            std::fs::write(&path, &bytes).unwrap();
            let back = read_image(&path).unwrap();
            std::fs::remove_file(&path).ok();
            assert_eq!(
                back.scalar_slice::<u8>().unwrap(),
                &[7, 8, 9, 10],
                "{label}"
            );
        }

        let path = tmp_path("badmagic.gipl");
        let mut bytes = gipl_header([2, 2, 1, 1], 8, [1.0; 4], [0.0; 4], 0xdead_beef);
        bytes.extend_from_slice(&[7, 8, 9, 10]);
        std::fs::write(&path, &bytes).unwrap();
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(!claimed);
        assert!(
            matches!(result, Err(IoError::NoReaderFound(_))),
            "{result:?}"
        );
    }

    /// `GIPL_MAGIC_NUMBER2` is 719555000 and `GIPL_MAGIC_NUMBER` 4026526128.
    #[test]
    fn gipl_magic_numbers_have_their_documented_decimal_values() {
        assert_eq!(gipl::GIPL_MAGIC_NUMBER, 4_026_526_128);
        assert_eq!(gipl::GIPL_MAGIC_NUMBER2, 719_555_000);
    }

    /// `Write` never consults `m_NumberOfComponents` for the header but
    /// `GetImageSizeInBytes()` does, so a 3-component image writes a scalar
    /// `image_type` and three times the described bytes. Reading it back gives a
    /// scalar image holding the first `numPixels` components (§2.96).
    #[test]
    fn gipl_vector_image_writes_a_scalar_header_and_reads_back_scalar() {
        let data: Vec<u8> = (0..12).collect();
        let img = Image::from_vec_vector::<u8>(&[2, 2], 3, data).unwrap();
        let path = tmp_path("vector.gipl");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(bytes.len(), gipl::HEADER_SIZE + 12);
        assert_eq!(&bytes[8..10], &8u16.to_be_bytes()); // GIPL_U_CHAR, not a vector
        assert_eq!(back.pixel_id(), PixelId::UInt8);
        assert_eq!(back.size(), &[2, 2, 1]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[0, 1, 2, 3]);
    }

    /// Upstream's `success = !m_Ifstream.bad()` accepts a short read and leaves
    /// the buffer tail uninitialised (§1.53); this port refuses it (§4.69).
    #[test]
    fn gipl_truncated_pixel_data_is_an_error() {
        let path = tmp_path("short.gipl");
        let mut bytes = gipl_header([4, 4, 1, 1], 8, [1.0; 4], [0.0; 4], gipl::GIPL_MAGIC_NUMBER);
        bytes.extend_from_slice(&[1, 2, 3]); // 16 pixels declared, 3 present
        std::fs::write(&path, &bytes).unwrap();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(matches!(result, Err(IoError::TruncatedData)), "{result:?}");
    }

    /// A header shorter than 256 bytes leaves upstream's `pixdim` / `origin`
    /// locals indeterminate; refused here (§4.73).
    #[test]
    fn gipl_short_header_is_an_error() {
        let path = tmp_path("stub.gipl");
        let header = gipl_header([2, 2, 1, 1], 8, [1.0; 4], [0.0; 4], gipl::GIPL_MAGIC_NUMBER);
        std::fs::write(&path, &header[..200]).unwrap();
        // `can_read_file` cannot reach offset 252 either.
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = gipl::read(&path);
        std::fs::remove_file(&path).ok();
        assert!(!claimed);
        assert!(matches!(result, Err(IoError::TruncatedData)), "{result:?}");
    }

    /// `read_image_information` touches only the header. This one declares
    /// 65535³ voxels of `double` and carries none of them.
    #[test]
    fn gipl_read_image_information_does_not_load_pixels() {
        let path = tmp_path("huge.gipl");
        let header = gipl_header(
            [65535, 65535, 65535, 1],
            65, // GIPL_DOUBLE
            [1.0; 4],
            [0.0; 4],
            gipl::GIPL_MAGIC_NUMBER,
        );
        std::fs::write(&path, &header).unwrap();

        let mut reader = ImageFileReader::new();
        reader.set_file_name(&path);
        let info = reader.read_image_information().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(info.pixel_id, PixelId::Float64);
        assert_eq!(info.dimension, 3);
        assert_eq!(info.number_of_components, 1);
        assert_eq!(info.size, vec![65535, 65535, 65535]);
        assert!(info.metadata.is_empty());
    }

    /// An `image_type` outside the table leaves `m_ComponentType` unknown and
    /// SimpleITK's `ExecuteInternalReadScalar` reaches its `"Logic error!"`.
    #[test]
    fn gipl_unknown_image_type_is_rejected() {
        let path = tmp_path("weird_type.gipl");
        let header = gipl_header(
            [2, 2, 1, 1],
            200,
            [1.0; 4],
            [0.0; 4],
            gipl::GIPL_MAGIC_NUMBER,
        );
        std::fs::write(&path, &header).unwrap();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.contains("image_type 200")),
            "{result:?}"
        );
    }

    /// `GIPL_BINARY` (1) reads as `UCHAR`, like `GIPL_U_CHAR` (8)
    /// (itkGiplImageIO.cxx:333-341).
    #[test]
    fn gipl_binary_image_type_reads_as_uint8() {
        let path = tmp_path("binary_type.gipl");
        let mut bytes = gipl_header([2, 1, 1, 1], 1, [1.0; 4], [0.0; 4], gipl::GIPL_MAGIC_NUMBER);
        bytes.extend_from_slice(&[0, 1]);
        std::fs::write(&path, &bytes).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::UInt8);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[0, 1]);
    }

    // ======================================================================
    // VTK — itk::VTKImageIO
    // ======================================================================

    fn write_vtk(name: &str, header: &str, data: &[u8]) -> std::path::PathBuf {
        let path = tmp_path(name);
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    const VTK_PREAMBLE: &str = "# vtk DataFile Version 3.0\n\
         VTK File Generated by Insight Segmentation and Registration Toolkit (ITK)\n";

    /// Every byte `WriteImageInformation` emits (itkVTKImageIO.cxx:653-709),
    /// trailing spaces and `%.16e` exponents included.
    #[test]
    fn vtk_header_pins_bytes_for_a_3d_scalar_image() {
        let mut img = Image::from_vec(&[4, 3, 2], (0..24).map(|i| i as f32).collect()).unwrap();
        img.set_spacing(&[0.5, 1.25, 3.0]).unwrap();
        img.set_origin(&[-2.0, 4.0, 7.5]).unwrap();

        let path = tmp_path("pin_scalar.vtk");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let expected = format!(
            "{VTK_PREAMBLE}\
             BINARY\n\
             DATASET STRUCTURED_POINTS\n\
             DIMENSIONS 4 3 2 \n\
             SPACING 5.0000000000000000e-01 1.2500000000000000e+00 3.0000000000000000e+00 \n\
             ORIGIN -2.0000000000000000e+00 4.0000000000000000e+00 7.5000000000000000e+00 \n\
             POINT_DATA 24\n\
             SCALARS scalars float 1\n\
             LOOKUP_TABLE default\n"
        );
        assert_eq!(&bytes[..expected.len()], expected.as_bytes());
        assert_eq!(bytes.len(), expected.len() + 24 * 4);
        // Big-endian data: 1.0f32 is 0x3f800000.
        assert_eq!(
            &bytes[expected.len() + 4..expected.len() + 8],
            &[0x3f, 0x80, 0, 0]
        );
    }

    /// A 2-D image is padded to three `DIMENSIONS` / `SPACING` / `ORIGIN` slots
    /// with `1`, `1.0`, `0.0` — and the reader collapses it back, so a 2-D
    /// image round-trips as 2-D (unlike GIPL, §2.94).
    #[test]
    fn vtk_header_pins_bytes_for_a_2d_image_and_the_padding_collapses_back() {
        let mut img = Image::from_vec(&[3, 2], vec![1u8, 2, 3, 4, 5, 6]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[1.0, -1.0]).unwrap();

        let path = tmp_path("pin_2d.vtk");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let expected = format!(
            "{VTK_PREAMBLE}\
             BINARY\n\
             DATASET STRUCTURED_POINTS\n\
             DIMENSIONS 3 2 1 \n\
             SPACING 5.0000000000000000e-01 2.0000000000000000e+00 1.0000000000000000e+00 \n\
             ORIGIN 1.0000000000000000e+00 -1.0000000000000000e+00 0.0000000000000000e+00 \n\
             POINT_DATA 6\n\
             SCALARS scalars unsigned_char 1\n\
             LOOKUP_TABLE default\n"
        );
        assert_eq!(&bytes[..expected.len()], expected.as_bytes());
        assert_eq!(bytes.len(), expected.len() + 6);

        assert_eq!(back.dimension(), 2);
        assert_eq!(back.size(), &[3, 2]);
        assert_eq!(back.spacing(), &[0.5, 2.0]);
        assert_eq!(back.origin(), &[1.0, -1.0]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    /// A three-component vector image takes the `VECTORS` branch, which carries
    /// no component count and no `LOOKUP_TABLE` line.
    #[test]
    fn vtk_header_pins_bytes_for_a_vector_image() {
        let img =
            Image::from_vec_vector::<f32>(&[2, 2], 3, (0..12).map(|i| i as f32).collect()).unwrap();
        let path = tmp_path("pin_vector.vtk");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let expected = format!(
            "{VTK_PREAMBLE}\
             BINARY\n\
             DATASET STRUCTURED_POINTS\n\
             DIMENSIONS 2 2 1 \n\
             SPACING 1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00 \n\
             ORIGIN 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 \n\
             POINT_DATA 4\n\
             VECTORS vectors float\n"
        );
        assert_eq!(&bytes[..expected.len()], expected.as_bytes());
        assert_eq!(bytes.len(), expected.len() + 12 * 4);
    }

    #[test]
    fn vtk_roundtrip_all_scalar_types_2d_and_3d() {
        macro_rules! case {
            ($ty:ty, $size:expr, $name:expr) => {{
                let count: usize = $size.iter().product();
                let data: Vec<$ty> = (0..count as u32).map(|i| i as $ty).collect();
                let img = Image::from_vec(&$size, data.clone()).unwrap();
                let path = tmp_path($name);
                write_image(&img, &path).unwrap();
                let back = read_image(&path).unwrap();
                std::fs::remove_file(&path).ok();
                assert_eq!(back.scalar_slice::<$ty>().unwrap(), data.as_slice(), $name);
                assert_eq!(back.size(), &$size[..], $name);
            }};
        }
        macro_rules! both {
            ($ty:ty, $stem:expr) => {{
                case!($ty, [4usize, 2], concat!($stem, "_2d.vtk"));
                case!($ty, [4usize, 2, 3], concat!($stem, "_3d.vtk"));
            }};
        }
        both!(u8, "vtk_u8");
        both!(i8, "vtk_i8");
        both!(u16, "vtk_u16");
        both!(i16, "vtk_i16");
        both!(u32, "vtk_u32");
        both!(i32, "vtk_i32");
        both!(u64, "vtk_u64");
        both!(i64, "vtk_i64");
        both!(f32, "vtk_f32");
        both!(f64, "vtk_f64");
    }

    #[test]
    fn vtk_roundtrip_vector_float32_preserves_geometry() {
        let data: Vec<f32> = (0..36).map(|i| i as f32 * 0.25 - 4.0).collect();
        let mut img = Image::from_vec_vector::<f32>(&[4, 3], 3, data.clone()).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();

        let path = tmp_path("vector_f32.vtk");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.spacing(), &[0.5, 2.0]);
        assert_eq!(back.origin(), &[-1.0, 3.0]);
        assert_eq!(back.component_slice::<f32>().unwrap(), data.as_slice());
    }

    /// VTK has no complex type, so a complex image writes `SCALARS scalars
    /// float 2` and reads back as a two-component vector — the same loss
    /// MetaImage has (§2.103).
    #[test]
    fn vtk_complex_roundtrips_as_a_two_component_vector() {
        let data = vec![
            Complex::new(1.0f32, 2.0),
            Complex::new(-3.0, 4.5),
            Complex::new(0.0, -1.0),
            Complex::new(7.25, 0.0),
        ];
        let img = Image::from_vec_complex(&[2, 2], data).unwrap();
        let path = tmp_path("complex.vtk");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(String::from_utf8_lossy(&bytes).contains("SCALARS scalars float 2\n"));
        assert_eq!(back.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(back.number_of_components_per_pixel(), 2);
        assert_eq!(
            back.component_slice::<f32>().unwrap(),
            &[1.0, 2.0, -3.0, 4.5, 0.0, -1.0, 7.25, 0.0]
        );
    }

    /// A one-component vector image writes `SCALARS scalars float 1`, whose
    /// reader arm is `numComp == 1 → SCALAR` (§2.103).
    #[test]
    fn vtk_one_component_vector_image_reads_back_as_scalar() {
        let img = Image::from_vec_vector::<f32>(&[2, 2], 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let path = tmp_path("one_comp.vtk");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(img.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(back.pixel_id(), PixelId::Float32);
        assert_eq!(back.scalar_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// `int64_t` is `long` under LP64, so `MapPixelType` reports `LONG` and
    /// `GetComponentTypeAsString` prints `long` — not `vtktypeint64`, which is
    /// what a host where `int64_t` is `long long` would write (§4.72). Both
    /// spellings read back to `Int64`.
    #[test]
    fn vtk_int64_writes_as_long_and_both_spellings_read_back() {
        let img = Image::from_vec(&[2, 1], vec![-1i64, i64::MAX]).unwrap();
        let path = tmp_path("i64.vtk");
        write_image(&img, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(String::from_utf8_lossy(&bytes).contains("SCALARS scalars long 1\n"));
        assert_eq!(back.scalar_slice::<i64>().unwrap(), &[-1, i64::MAX]);

        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\n\
             DIMENSIONS 2 1 1 \nPOINT_DATA 2\n\
             SCALARS scalars vtktypeint64 1\nLOOKUP_TABLE default\n"
        );
        let mut data = (-1i64).to_be_bytes().to_vec();
        data.extend_from_slice(&i64::MAX.to_be_bytes());
        let path = write_vtk("vtktypeint64.vtk", &header, &data);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::Int64);
        assert_eq!(back.scalar_slice::<i64>().unwrap(), &[-1, i64::MAX]);
    }

    /// `DIMENSIONS x y 1` is 2-D and `DIMENSIONS x 1 z` is 3-D: the surviving
    /// rule is `dims[2] <= 1`, and the `dims[1]` test above it is dead (§2.100).
    #[test]
    fn vtk_only_the_third_dimension_decides_two_versus_three_d() {
        let header = |dims: &str, n: usize| {
            format!(
                "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS {dims} \n\
                 POINT_DATA {n}\nSCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n"
            )
        };

        let path = write_vtk("dims_z1.vtk", &header("4 3 1", 12), &[0u8; 12]);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.size(), &[4, 3]);

        let path = write_vtk("dims_y1.vtk", &header("4 1 3", 12), &[0u8; 12]);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.size(), &[4, 1, 3]);
    }

    /// ASCII data is whitespace-separated decimal, extracted through
    /// `NumericTraits::PrintType` — `int` for the 8-bit types, so `300` lands as
    /// `44` and `-1` as `255`.
    #[test]
    fn vtk_ascii_unsigned_char_data_is_decimal_integers_cast_to_the_pixel_type() {
        let header = format!(
            "{VTK_PREAMBLE}ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 4 1 1 \n\
             POINT_DATA 4\nSCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("ascii_uchar.vtk", &header, b"0 300 -1 255\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::UInt8);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[0, 44, 255, 255]);
    }

    /// ASCII float data, with spacing and origin taken from the header.
    #[test]
    fn vtk_ascii_float_data_is_read_with_geometry() {
        let header = format!(
            "{VTK_PREAMBLE}ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 2 1 \n\
             SPACING 0.5 2 1\nORIGIN -1 3 0\nPOINT_DATA 4\n\
             SCALARS scalars float 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("ascii_float.vtk", &header, b"1.5 -2.25 3e2 0\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::Float32);
        assert_eq!(back.spacing(), &[0.5, 2.0]);
        assert_eq!(back.origin(), &[-1.0, 3.0]);
        assert_eq!(
            back.scalar_slice::<f32>().unwrap(),
            &[1.5, -2.25, 300.0, 0.0]
        );
    }

    /// `NumericTraits<unsigned short>::PrintType` is `unsigned short` itself, and
    /// `num_get` extracts a negative literal into the unsigned representation:
    /// `-1` is `65535` and the stream stays good, while `-70000` overflows to
    /// `65535` and latches `failbit`, taking the rest of the buffer with it
    /// (§2.105, §2.107).
    #[test]
    fn vtk_ascii_negative_literal_for_an_unsigned_type_wraps_instead_of_failing() {
        let header = format!(
            "{VTK_PREAMBLE}ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 4 1 1 \n\
             POINT_DATA 4\nSCALARS scalars unsigned_short 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("ascii_ushort.vtk", &header, b"-1 -65535 7 8\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<u16>().unwrap(), &[65535, 1, 7, 8]);

        let path = write_vtk("ascii_ushort_of.vtk", &header, b"1 -70000 7 8\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(
            back.scalar_slice::<u16>().unwrap(),
            &[1, 65535, 65535, 65535]
        );
    }

    /// `ImageIOBase::ReadBuffer` declares `PrintType temp;` outside the loop, so
    /// once `failbit` latches every remaining component keeps the failing
    /// extraction's value: `0` when the data ran out, the saturated limit when
    /// it overflowed (§2.107).
    #[test]
    fn vtk_ascii_latches_the_failing_extraction_value_into_every_later_component() {
        let header = |ty: &str| {
            format!(
                "{VTK_PREAMBLE}ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 4 1 1 \n\
                 POINT_DATA 4\nSCALARS scalars {ty} 1\nLOOKUP_TABLE default\n"
            )
        };

        // Data runs out after two values: the rest are zero, not an error.
        let path = write_vtk("ascii_short.vtk", &header("int"), b"10 20\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<i32>().unwrap(), &[10, 20, 0, 0]);

        // An out-of-range value saturates `temp` and every later component
        // inherits the saturated value, not zero.
        let path = write_vtk("ascii_overflow.vtk", &header("short"), b"5 99999 7 8\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(
            back.scalar_slice::<i16>().unwrap(),
            &[5, i16::MAX, i16::MAX, i16::MAX]
        );

        // A non-numeric field fails with `temp = 0`.
        let path = write_vtk("ascii_junk.vtk", &header("int"), b"1 2 x 4\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<i32>().unwrap(), &[1, 2, 0, 0]);
    }

    /// The `LOOKUP_TABLE` line is optional: the reader peeks one line and rewinds
    /// when it is not one (itkVTKImageIO.cxx:331-339).
    #[test]
    fn vtk_scalars_without_a_lookup_table_line_starts_the_data_immediately() {
        let header = format!(
            "{VTK_PREAMBLE}ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 3 1 1 \n\
             POINT_DATA 3\nSCALARS scalars int 1\n"
        );
        let path = write_vtk("no_lookup.vtk", &header, b"7 8 9\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<i32>().unwrap(), &[7, 8, 9]);
    }

    /// `COLOR_SCALARS n` never consults the file's component type: `BINARY`
    /// means `UCHAR` and `ASCII` means `FLOAT`. `n == 3` is `RGB`, which
    /// SimpleITK loads as a `VectorUInt8` (itkVTKImageIO.cxx:277-308).
    #[test]
    fn vtk_color_scalars_binary_reads_as_three_component_uint8() {
        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 1 1 \n\
             POINT_DATA 2\nCOLOR_SCALARS color_scalars 3\n"
        );
        let path = write_vtk("color3.vtk", &header, &[1u8, 2, 3, 4, 5, 6]);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.component_slice::<u8>().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    /// `COLOR_SCALARS ... 1` is `SCALAR` with one component, and an `ASCII` file
    /// reads it as `float` — the arm that ignores the declared type most visibly.
    #[test]
    fn vtk_color_scalars_ascii_reads_as_float() {
        let header = format!(
            "{VTK_PREAMBLE}ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 1 1 \n\
             POINT_DATA 2\nCOLOR_SCALARS color_scalars 1\n"
        );
        let path = write_vtk("color1.vtk", &header, b"0.25 0.75\n");
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::Float32);
        assert_eq!(back.scalar_slice::<f32>().unwrap(), &[0.25, 0.75]);
    }

    /// `COLOR_SCALARS ... 4` is `RGBA`; a count outside `{1, 3, 4}` falls into
    /// the `VECTOR` arm. There is no range check (§2.104).
    #[test]
    fn vtk_color_scalars_component_count_is_unclamped() {
        let header = |n: usize, pixels: usize| {
            format!(
                "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS {pixels} 1 1 \n\
                 POINT_DATA {pixels}\nCOLOR_SCALARS color_scalars {n}\n"
            )
        };
        let path = write_vtk("color4.vtk", &header(4, 1), &[1u8, 2, 3, 4]);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.number_of_components_per_pixel(), 4);

        let path = write_vtk("color7.vtk", &header(7, 1), &[1u8, 2, 3, 4, 5, 6, 7]);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(back.number_of_components_per_pixel(), 7);
    }

    /// The attribute dispatch is a substring test in a fixed order, so a
    /// `SCALARS` array *named* `vector_field` is parsed as a `VECTORS` line:
    /// three components, the third token as the type, no `LOOKUP_TABLE` peek
    /// (§2.101).
    #[test]
    fn vtk_scalars_named_vector_field_is_parsed_as_a_vectors_line() {
        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 1 1 \n\
             POINT_DATA 2\nSCALARS vector_field float 1\n"
        );
        let mut data = Vec::new();
        for v in 0..6u32 {
            data.extend_from_slice(&(v as f32).to_be_bytes());
        }
        let path = write_vtk("named_vector.vtk", &header, &data);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.pixel_id(), PixelId::VectorFloat32);
        assert_eq!(back.number_of_components_per_pixel(), 3);
    }

    /// `aspect_ratio` is `spacing`'s legacy spelling and takes the same arm.
    #[test]
    fn vtk_aspect_ratio_is_read_as_spacing() {
        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 2 1 \n\
             ASPECT_RATIO 0.25 4 1\nPOINT_DATA 4\n\
             SCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("aspect.vtk", &header, &[1u8, 2, 3, 4]);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.spacing(), &[0.25, 4.0]);
    }

    /// `VTKImageIO` reads a `TENSORS` file; SimpleITK's `GetPixelIDFromImageIO`
    /// has no `SYMMETRICSECONDRANKTENSOR` arm and throws `"Unknown PixelType"`
    /// (§3.37).
    #[test]
    fn vtk_tensors_are_unreadable_through_the_simpleitk_pixel_id_mapping() {
        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 1 1 1 \n\
             POINT_DATA 1\nTENSORS tensors float\n"
        );
        let path = write_vtk("tensors.vtk", &header, &[0u8; 36]);
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(claimed, "CanReadFile only tests the DATASET line");
        assert!(
            matches!(&result, Err(IoError::UnsupportedVtkFeature(m)) if m.contains("Unknown PixelType")),
            "{result:?}"
        );
    }

    /// `CanReadFile` requires `structured_points` on the fourth line, so a
    /// `STRUCTURED_GRID` dataset is claimed by nobody; reaching the reader
    /// directly gives `InternalReadImageInformation`'s own message.
    #[test]
    fn vtk_rejects_other_dataset_types() {
        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_GRID\nDIMENSIONS 2 2 1 \n\
             POINT_DATA 4\nSCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("grid.vtk", &header, &[1u8, 2, 3, 4]);
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let via_registry = read_image(&path);
        let direct = vtk::read(&path);
        std::fs::remove_file(&path).ok();

        assert!(!claimed);
        assert!(
            matches!(via_registry, Err(IoError::NoReaderFound(_))),
            "{via_registry:?}"
        );
        assert!(
            matches!(&direct, Err(IoError::MalformedVtkHeader(m)) if m == "Not structured points, can't read"),
            "{direct:?}"
        );
    }

    /// A third line that is neither `ASCII` nor `BINARY` is `"Unrecognized
    /// type"`; note `CanReadFile` never looks at it, so the file is still
    /// claimed and then fails in the reader.
    #[test]
    fn vtk_unrecognized_file_type_line_is_an_error_only_in_the_reader() {
        let header = format!(
            "{VTK_PREAMBLE}HEXADECIMAL\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 1 1 \n\
             POINT_DATA 2\nSCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("hex.vtk", &header, &[1u8, 2]);
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(claimed);
        assert!(
            matches!(&result, Err(IoError::MalformedVtkHeader(m)) if m == "Unrecognized type"),
            "{result:?}"
        );
    }

    /// Upstream lets `GetNextLine`'s "Premature EOF" escape `CanReadFile`, so
    /// `CreateImageIO` throws on a `.vtk` shorter than four lines. Here it is
    /// simply not claimed (§4.71).
    #[test]
    fn vtk_can_read_file_does_not_throw_on_a_short_file() {
        let path = tmp_path("stub.vtk");
        std::fs::write(&path, b"# vtk DataFile Version 3.0\ntitle\n").unwrap();
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(!claimed);
        assert!(
            matches!(result, Err(IoError::NoReaderFound(_))),
            "{result:?}"
        );
    }

    /// `GetNextLine`'s guard is `count > 5`, so five consecutive empty lines are
    /// tolerated and the sixth throws — one more than the message claims
    /// (§2.98).
    #[test]
    fn vtk_five_empty_lines_are_tolerated_and_six_are_not() {
        let body = |blanks: usize| {
            format!(
                "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 1 1 \n{}\
                 POINT_DATA 2\nSCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n",
                "\n".repeat(blanks)
            )
        };
        let path = write_vtk("blank5.vtk", &body(5), &[1u8, 2]);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &[1, 2]);

        let path = write_vtk("blank6.vtk", &body(6), &[1u8, 2]);
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::MalformedVtkHeader(m)) if m.contains("empty lines")),
            "{result:?}"
        );
    }

    /// `getline` sets `eofbit` even when it delivered the line's characters, so
    /// a header line with no terminating newline is a "Premature EOF" (§2.99).
    /// Here the `LOOKUP_TABLE` peek is the line that falls off the end.
    #[test]
    fn vtk_a_final_line_without_a_newline_is_a_premature_eof() {
        let header = format!(
            "{VTK_PREAMBLE}ASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 3 1 1 \n\
             POINT_DATA 3\nSCALARS scalars int 1\n"
        );
        let path = write_vtk("no_newline.vtk", &header, b"7 8 9");
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::MalformedVtkHeader(m)) if m == "Premature EOF in reading a line"),
            "{result:?}"
        );
    }

    /// `Read` discards `ReadBufferAsBinary`'s `bool` and leaves the tail of its
    /// buffer uninitialised (§1.54); refused here (§4.69).
    #[test]
    fn vtk_truncated_binary_data_is_an_error() {
        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 4 4 1 \n\
             POINT_DATA 16\nSCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("short_data.vtk", &header, &[1u8, 2, 3]);
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(matches!(result, Err(IoError::TruncatedData)), "{result:?}");
    }

    /// An under-filled `DIMENSIONS` line reads indeterminate `unsigned int`s
    /// upstream (§1.55); refused here (§4.70).
    #[test]
    fn vtk_under_filled_dimensions_line_is_an_error() {
        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 4 \n\
             POINT_DATA 4\nSCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("bad_dims.vtk", &header, &[1u8, 2, 3, 4]);
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::MalformedVtkHeader(m)) if m.contains("three values")),
            "{result:?}"
        );
    }

    /// An under-filled `SPACING` line keeps the `1.0` defaults
    /// `InternalReadImageInformation` seeded (§4.70).
    #[test]
    fn vtk_under_filled_spacing_line_keeps_the_defaults() {
        let header = format!(
            "{VTK_PREAMBLE}BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 2 1 \n\
             SPACING 0.25\nPOINT_DATA 4\n\
             SCALARS scalars unsigned_char 1\nLOOKUP_TABLE default\n"
        );
        let path = write_vtk("short_spacing.vtk", &header, &[1u8, 2, 3, 4]);
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.spacing(), &[0.25, 1.0]);
    }

    /// `WriteImageInformation` refuses anything but 1, 2 or 3 dimensions
    /// (itkVTKImageIO.cxx:647-651).
    #[test]
    fn vtk_write_rejects_four_dimensional_images() {
        let img = Image::new(&[2, 2, 2, 2], PixelId::UInt8);
        let path = tmp_path("four_d.vtk");
        let result = write_image(&img, &path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::UnsupportedVtkFeature(m)) if m.contains("1, 2 or 3-dimensional")),
            "{result:?}"
        );
    }

    /// A `.vtk` extension is necessary but not sufficient: `CanReadFile` also
    /// wants `structured_points`.
    #[test]
    fn vtk_extension_alone_does_not_claim_a_file_for_reading() {
        let path = tmp_path("not_really.vtk");
        std::fs::write(&path, b"this is a text file, not a VTK image\n").unwrap();
        let claimed = create_image_io(&path, FileMode::Read).is_some();
        std::fs::remove_file(&path).ok();
        assert!(!claimed);
    }

    // ---- PNG ---------------------------------------------------------------

    #[test]
    fn png_roundtrip_grayscale_uint8() {
        let data: Vec<u8> = (0..12u32).map(|i| (i * 23) as u8).collect();
        let img = Image::from_vec(&[4, 3], data.clone()).unwrap();
        let path = tmp_path("gray_u8.png");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::UInt8);
        assert_eq!(back.size(), &[4, 3]);
        // No `sCAL` support (§4.85): spacing/origin are always the defaults,
        // whatever the source `Image` carried.
        assert_eq!(back.spacing(), &[1.0, 1.0]);
        assert_eq!(back.origin(), &[0.0, 0.0]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), data.as_slice());
    }

    /// 16-bit samples above 255 pin that the write/read path carries full
    /// 16-bit range, not just the low byte — the endianness swap itself is
    /// pinned independently in `png::tests` against a hand-built fixture.
    #[test]
    fn png_roundtrip_grayscale_uint16() {
        let data: Vec<u16> = (0..12u32).map(|i| (i * 4111 + 29) as u16).collect();
        let img = Image::from_vec(&[4, 3], data.clone()).unwrap();
        let path = tmp_path("gray_u16.png");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::UInt16);
        assert_eq!(back.size(), &[4, 3]);
        assert_eq!(back.scalar_slice::<u16>().unwrap(), data.as_slice());
    }

    #[test]
    fn png_roundtrip_rgb_vector_uint8() {
        let data: Vec<u8> = (0..36u32).map(|i| (i * 7) as u8).collect();
        let img = Image::from_vec_vector::<u8>(&[4, 3], 3, data.clone()).unwrap();
        let path = tmp_path("rgb_u8.png");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.size(), &[4, 3]);
        assert_eq!(back.component_slice::<u8>().unwrap(), data.as_slice());
    }

    #[test]
    fn png_roundtrip_rgb_vector_uint16() {
        let data: Vec<u16> = (0..36u32).map(|i| (i * 4111 + 29) as u16).collect();
        let img = Image::from_vec_vector::<u16>(&[4, 3], 3, data.clone()).unwrap();
        let path = tmp_path("rgb_u16.png");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorUInt16);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.size(), &[4, 3]);
        assert_eq!(back.component_slice::<u16>().unwrap(), data.as_slice());
    }

    #[test]
    fn png_roundtrip_rgba_vector_uint8() {
        let data: Vec<u8> = (0..48u32).map(|i| (i * 5) as u8).collect();
        let img = Image::from_vec_vector::<u8>(&[4, 3], 4, data.clone()).unwrap();
        let path = tmp_path("rgba_u8.png");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(back.number_of_components_per_pixel(), 4);
        assert_eq!(back.size(), &[4, 3]);
        assert_eq!(back.component_slice::<u8>().unwrap(), data.as_slice());
    }

    #[test]
    fn png_roundtrip_rgba_vector_uint16() {
        let data: Vec<u16> = (0..48u32).map(|i| (i * 4111 + 29) as u16).collect();
        let img = Image::from_vec_vector::<u16>(&[4, 3], 4, data.clone()).unwrap();
        let path = tmp_path("rgba_u16.png");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorUInt16);
        assert_eq!(back.number_of_components_per_pixel(), 4);
        assert_eq!(back.size(), &[4, 3]);
        assert_eq!(back.component_slice::<u16>().unwrap(), data.as_slice());
    }

    /// `WriteSlice` takes `height` from `GetDimensions(1)` alone
    /// (itkPNGImageIO.cxx:605) and never consults any axis beyond it, so a
    /// 3-D image's second and later slices are never written. Ledger §2.125.
    #[test]
    fn png_write_of_a_three_dimensional_image_writes_only_the_first_slice() {
        let data: Vec<u8> = (0..12u32).map(|i| i as u8).collect(); // size [2, 2, 3]
        let img = Image::from_vec(&[2, 2, 3], data.clone()).unwrap();
        let path = tmp_path("three_d.png");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.dimension(), 2);
        assert_eq!(back.size(), &[2, 2]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), &data[..4]);
    }

    /// Fixed §1.59: upstream's `WriteSlice` opens the file with
    /// `fopen(fileName, "wb")` — truncating it immediately — before its
    /// component-type switch's `default:` throws "PNG supports unsigned char
    /// and unsigned short" (itkPNGImageIO.cxx:514-553). This port now checks
    /// the component type before creating (and truncating) the output file,
    /// so a pre-existing file at the target path is left untouched.
    #[test]
    fn png_write_of_an_unwritable_component_type_is_rejected_before_the_file_is_touched() {
        let img = Image::from_vec(&[2, 2], vec![1i16, 2, 3, 4]).unwrap();
        let path = tmp_path("unwritable.png");
        std::fs::write(&path, b"pre-existing content that must survive").unwrap();

        let result = write_image(&img, &path);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&result, Err(IoError::UnsupportedPngFeature(m))
                if m.contains("PNG supports unsigned char and unsigned short")),
            "{result:?}"
        );
        assert_eq!(bytes, b"pre-existing content that must survive");
    }

    /// `CanReadFile`'s signature check rejects a `.png`-named file whose
    /// content is not PNG, so `create_image_io` never selects [`png`] for it
    /// and the top-level [`read_image`] reports [`IoError::NoReaderFound`]
    /// rather than [`IoError::MalformedPngHeader`] — that error is only
    /// observable by calling [`png::read`]/[`png::read_information`] directly,
    /// bypassing the registry, exactly as the analogous GIPL/VTK tests do.
    #[test]
    fn png_garbage_bytes_under_a_png_extension_are_not_claimed_for_reading() {
        let path = tmp_path("garbage.png");
        std::fs::write(&path, b"this is not a png file, but is long enough\n").unwrap();

        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        let direct_read = png::read(&path);
        let direct_info = png::read_information(&path);
        std::fs::remove_file(&path).ok();

        assert!(!claimed);
        assert!(
            matches!(result, Err(IoError::NoReaderFound(_))),
            "{result:?}"
        );
        assert!(
            matches!(&direct_read, Err(IoError::MalformedPngHeader(_))),
            "{direct_read:?}"
        );
        assert!(
            matches!(&direct_info, Err(IoError::MalformedPngHeader(_))),
            "{direct_info:?}"
        );
    }

    /// Unlike the garbage-bytes case, `CanReadFile` checks only the 8-byte
    /// signature (itkPNGImageIO.cxx:79-89) — it does not validate the rest of
    /// the stream — so a file with a genuine signature but a truncated `IDAT`
    /// *is* claimed, and the decode failure surfaces through the registry as
    /// [`IoError::PngDecode`].
    #[test]
    fn png_truncated_idat_is_claimed_and_then_a_decode_error() {
        let img = Image::from_vec(&[8, 8], vec![0u8; 64]).unwrap();
        let path = tmp_path("truncated.png");
        write_image(&img, &path).unwrap();

        let full = std::fs::read(&path).unwrap();
        let truncated = &full[..full.len() - 20];
        std::fs::write(&path, truncated).unwrap();

        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();

        assert!(claimed);
        assert!(matches!(&result, Err(IoError::PngDecode(_))), "{result:?}");
    }

    // ---- JPEG ---------------------------------------------------------------

    /// JPEG is lossy, so only dimensions and pixel type round-trip exactly —
    /// not exact bytes.
    #[test]
    fn jpeg_roundtrip_grayscale_uint8() {
        let data: Vec<u8> = (0..(16 * 12))
            .map(|i| ((i * 37 + i * i) % 256) as u8)
            .collect();
        let img = Image::from_vec(&[16, 12], data).unwrap();
        let path = tmp_path("gray_u8.jpg");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::UInt8);
        assert_eq!(back.size(), &[16, 12]);
    }

    #[test]
    fn jpeg_roundtrip_rgb_vector_uint8() {
        let data: Vec<u8> = (0..(20 * 10 * 3)).map(|i| ((i * 53) % 256) as u8).collect();
        let img = Image::from_vec_vector::<u8>(&[20, 10], 3, data).unwrap();
        let path = tmp_path("rgb_u8.jpg");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.size(), &[20, 10]);
    }

    /// `Write` throws outright when `GetNumberOfDimensions() != 2`
    /// (itkJPEGImageIO.cxx:459-463) — unlike PNG, which silently writes only
    /// the first Z-slice (ledger §2.125). No file is left behind. Ledger
    /// §2.135.
    #[test]
    fn jpeg_write_of_a_three_dimensional_image_is_rejected() {
        let img = Image::from_vec(&[2, 2, 3], vec![0u8; 12]).unwrap();
        let path = tmp_path("three_d.jpg");

        let result = write_image(&img, &path);

        assert!(
            matches!(&result, Err(IoError::JpegWriteRejected(m)) if m.contains("2-dimensional")),
            "{result:?}"
        );
        assert!(!path.exists());
    }

    /// `CanReadFile`'s magic check rejects a `.jpg`-named file whose content
    /// is not a JPEG, so `create_image_io` never selects [`jpeg`] for it and
    /// the top-level [`read_image`] reports [`IoError::NoReaderFound`] rather
    /// than [`IoError::JpegDecode`] — that error is only observable by
    /// calling [`jpeg::read`]/[`jpeg::read_information`] directly, bypassing
    /// the registry, exactly as the analogous PNG/GIPL/VTK tests do.
    #[test]
    fn jpeg_garbage_bytes_under_a_jpg_extension_are_not_claimed_for_reading() {
        let path = tmp_path("garbage.jpg");
        std::fs::write(&path, b"this is not a jpeg file, but is long enough\n").unwrap();

        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        let direct_read = jpeg::read(&path);
        let direct_info = jpeg::read_information(&path);
        std::fs::remove_file(&path).ok();

        assert!(!claimed);
        assert!(
            matches!(result, Err(IoError::NoReaderFound(_))),
            "{result:?}"
        );
        assert!(
            matches!(&direct_read, Err(IoError::JpegDecode(_))),
            "{direct_read:?}"
        );
        assert!(
            matches!(&direct_info, Err(IoError::JpegDecode(_))),
            "{direct_info:?}"
        );
    }

    /// Unlike the garbage-bytes case, `CanReadFile` parses the *header*
    /// completely (itkJPEGImageIO.cxx:141-155) but never touches the
    /// entropy-coded scan data that follows it — so a file with a genuine
    /// header but a truncated scan *is* claimed, and the decode failure
    /// surfaces through the registry as [`IoError::JpegDecode`]. The JPEG
    /// counterpart of `png_truncated_idat_is_claimed_and_then_a_decode_error`.
    #[test]
    fn jpeg_truncated_scan_data_is_claimed_and_then_a_decode_error() {
        let img = Image::from_vec(&[32, 32], vec![0u8; 32 * 32]).unwrap();
        let path = tmp_path("truncated.jpg");
        write_image(&img, &path).unwrap();

        let full = std::fs::read(&path).unwrap();
        let truncated = &full[..full.len() - 20];
        std::fs::write(&path, truncated).unwrap();

        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();

        assert!(claimed);
        assert!(matches!(&result, Err(IoError::JpegDecode(_))), "{result:?}");
    }

    // ---- TIFF --------------------------------------------------------------

    /// The 2-D scalar round-trips, one per `SampleFormat` / bit-depth pair the
    /// `component_type` ladder (itkTIFFImageIO.cxx:395-431) can name.
    #[test]
    fn tiff_roundtrip_scalar_uint8() {
        let data: Vec<u8> = (0..12u32).map(|i| (i * 23) as u8).collect();
        let img = Image::from_vec(&[4, 3], data.clone()).unwrap();
        let path = tmp_path("scalar_u8.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::UInt8);
        assert_eq!(back.size(), &[4, 3]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), data.as_slice());
    }

    #[test]
    fn tiff_roundtrip_scalar_int8() {
        let data: Vec<i8> = (0..12i32).map(|i| (i * 17 - 100) as i8).collect();
        let img = Image::from_vec(&[4, 3], data.clone()).unwrap();
        let path = tmp_path("scalar_i8.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::Int8);
        assert_eq!(back.scalar_slice::<i8>().unwrap(), data.as_slice());
    }

    /// Both libtiff and the `tiff` crate hand back samples already in host
    /// order, so a 16-bit round-trip needs no byte swap on either side — unlike
    /// PNG, whose big-endian samples the port swaps by hand.
    #[test]
    fn tiff_roundtrip_scalar_uint16() {
        let data: Vec<u16> = (0..12u32).map(|i| (i * 4111 + 29) as u16).collect();
        let img = Image::from_vec(&[4, 3], data.clone()).unwrap();
        let path = tmp_path("scalar_u16.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::UInt16);
        assert_eq!(back.scalar_slice::<u16>().unwrap(), data.as_slice());
    }

    #[test]
    fn tiff_roundtrip_scalar_int16() {
        let data: Vec<i16> = (0..12i32).map(|i| (i * 4111 - 20_000) as i16).collect();
        let img = Image::from_vec(&[4, 3], data.clone()).unwrap();
        let path = tmp_path("scalar_i16.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::Int16);
        assert_eq!(back.scalar_slice::<i16>().unwrap(), data.as_slice());
    }

    #[test]
    fn tiff_roundtrip_scalar_float32() {
        let data: Vec<f32> = (0..12u32).map(|i| i as f32 * 0.5 - 3.0).collect();
        let img = Image::from_vec(&[4, 3], data.clone()).unwrap();
        let path = tmp_path("scalar_f32.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::Float32);
        assert_eq!(back.scalar_slice::<f32>().unwrap(), data.as_slice());
    }

    #[test]
    fn tiff_roundtrip_rgb_vector_uint8() {
        let data: Vec<u8> = (0..36u32).map(|i| (i * 7) as u8).collect();
        let img = Image::from_vec_vector::<u8>(&[4, 3], 3, data.clone()).unwrap();
        let path = tmp_path("rgb_u8.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(back.number_of_components_per_pixel(), 3);
        assert_eq!(back.component_slice::<u8>().unwrap(), data.as_slice());
    }

    /// Four components become `PHOTOMETRIC_RGB` (itkTIFFImageIO.cxx:731) plus one
    /// `EXTRASAMPLE_ASSOCALPHA` (`:678-690`), and `GetFormat` reads that back as
    /// `RGB_` with four components (`:107-110`, `:441-444`).
    #[test]
    fn tiff_roundtrip_rgba_vector_uint16() {
        let data: Vec<u16> = (0..48u32).map(|i| (i * 4111 + 29) as u16).collect();
        let img = Image::from_vec_vector::<u16>(&[4, 3], 4, data.clone()).unwrap();
        let path = tmp_path("rgba_u16.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.pixel_id(), PixelId::VectorUInt16);
        assert_eq!(back.number_of_components_per_pixel(), 4);
        assert_eq!(back.component_slice::<u16>().unwrap(), data.as_slice());
    }

    /// `InternalWrite` sends every component count other than 1 down the
    /// `PHOTOMETRIC_RGB` arm (itkTIFFImageIO.cxx:725-732), so a two-component
    /// image is written as RGB with `SamplesPerPixel = 2` — a file no TIFF
    /// reader can interpret as colour, and which upstream's own reader takes
    /// back as a *one*-component grayscale of half-rows (§2.140). This port
    /// writes the same bytes and refuses to read them.
    #[test]
    fn tiff_write_of_a_two_component_image_emits_photometric_rgb_with_two_samples() {
        let data: Vec<u8> = (0..24u32).map(|i| i as u8).collect();
        let img = Image::from_vec_vector::<u8>(&[4, 3], 2, data).unwrap();
        let path = tmp_path("two_component.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path);
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&back, Err(IoError::UnsupportedTiffFeature(m)) if m.contains("SamplesPerPixel = 2")),
            "{back:?}"
        );
    }

    /// A 3-D image writes one directory per slice, each tagged
    /// `FILETYPE_PAGE` with a `TIFFTAG_PAGENUMBER` of `(page, total)`
    /// (itkTIFFImageIO.cxx:800-806). `m_SubFiles` counts only `SUBFILETYPE == 0`
    /// directories (itkTIFFReaderInternal.cxx:231-251), so on the way back in every page is `FILETYPE_PAGE`,
    /// `m_SubFiles == 0`, and the depth comes from `m_NumberOfPages`.
    #[test]
    fn tiff_roundtrip_three_dimensional_volume() {
        let data: Vec<u8> = (0..24u32).map(|i| (i * 3) as u8).collect();
        let img = Image::from_vec(&[4, 2, 3], data.clone()).unwrap();
        let path = tmp_path("volume_u8.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.dimension(), 3);
        assert_eq!(back.size(), &[4, 2, 3]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), data.as_slice());
    }

    /// `WriteImageInformation` writes `25.4 / spacing` as an inch resolution and
    /// `ReadImageInformation` divides it back out — but libtiff's
    /// `TIFFTAG_XRESOLUTION` field is a `float`, so a spacing survives only to
    /// `f32` precision, and never exactly. Ledger §2.142.
    #[test]
    fn tiff_spacing_round_trips_only_to_float_precision() {
        let mut img = Image::from_vec(&[2, 2], vec![0u8; 4]).unwrap();
        img.set_spacing(&[0.5, 0.25]).unwrap();
        let path = tmp_path("spacing.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // 25.4/0.5 = 50.8 and 25.4/0.25 = 101.6 are both exact in `f32`'s
        // mantissa only to within an ulp; the recovered spacing is off by one.
        let spacing = back.spacing();
        assert!((spacing[0] - 0.5).abs() < 1e-7, "{spacing:?}");
        assert!((spacing[1] - 0.25).abs() < 1e-7, "{spacing:?}");
        assert_ne!(spacing, &[0.5, 0.25]);
    }

    /// A zero spacing writes no resolution tags at all
    /// (itkTIFFImageIO.cxx:792), and the reader's `m_ResolutionUnit > 0` guard (`:375`)
    /// then leaves the spacing at its `1.0` seed.
    #[test]
    fn tiff_unit_spacing_is_the_default_when_no_resolution_is_written() {
        let img = Image::from_vec(&[2, 2], vec![0u8; 4]).unwrap();
        let path = tmp_path("no_resolution.tif");
        write_image(&img, &path).unwrap();
        let back = read_image(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // Spacing 1.0 → resolution 25.4 → spacing 1.0000000150184933.
        assert!((back.spacing()[0] - 1.0).abs() < 1e-7);
        assert_eq!(back.origin(), &[0.0, 0.0]);
    }

    /// `m_UseCompression` picks between `COMPRESSION_NONE` and this port's only
    /// reachable compressor, `COMPRESSION_PACKBITS`
    /// (itkTIFFImageIO.cxx:214, :259-265, :694-712). The image must survive the
    /// round-trip through multi-row strips — upstream sizes strips at
    /// `1 MiB / scanlinesize` rows regardless of the compressor
    /// (itkTIFFImageIO.cxx:784), where the `tiff` crate's own `PackBits` default
    /// would be one row per strip.
    #[test]
    fn tiff_packbits_compression_round_trips_through_multi_row_strips() {
        // 64x64 of a low-entropy pattern: PackBits must actually shrink it, and
        // 64 rows of 64 bytes are one 4 KiB strip, well under the 1 MiB target.
        let data: Vec<u8> = (0..64 * 64u32).map(|i| (i / 97) as u8).collect();
        let img = Image::from_vec(&[64, 64], data.clone()).unwrap();

        let plain = tmp_path("packbits_off.tif");
        let packed = tmp_path("packbits_on.tif");
        write_image_with(&img, &plain, false, -1).unwrap();
        write_image_with(&img, &packed, true, -1).unwrap();

        let plain_len = std::fs::metadata(&plain).unwrap().len();
        let packed_len = std::fs::metadata(&packed).unwrap().len();
        let back = read_image(&packed).unwrap();
        std::fs::remove_file(&plain).ok();
        std::fs::remove_file(&packed).ok();

        assert!(packed_len < plain_len, "{packed_len} !< {plain_len}");
        assert_eq!(back.size(), &[64, 64]);
        assert_eq!(back.scalar_slice::<u8>().unwrap(), data.as_slice());
    }

    /// `SetCompressionLevel` is TIFF's *JPEG quality*
    /// (itkTIFFImageIO.cxx:213, itkTIFFImageIO.h:171-179), and the only site that
    /// reads it is the unreachable `JPEG` arm (`:749`) —
    /// so it changes nothing about the bytes written. Ledger §3.52.
    #[test]
    fn tiff_compression_level_does_not_change_the_written_bytes() {
        let data: Vec<u8> = (0..64 * 64u32).map(|i| (i / 97) as u8).collect();
        let img = Image::from_vec(&[64, 64], data).unwrap();

        let default_level = tmp_path("level_default.tif");
        let max_level = tmp_path("level_max.tif");
        write_image_with(&img, &default_level, true, -1).unwrap();
        write_image_with(&img, &max_level, true, 100).unwrap();

        let a = std::fs::read(&default_level).unwrap();
        let b = std::fs::read(&max_level).unwrap();
        std::fs::remove_file(&default_level).ok();
        std::fs::remove_file(&max_level).ok();

        assert_eq!(a, b);
    }

    /// `TIFFImageIO::CanWriteFile` lower-cases nothing — it is
    /// `HasSupportedWriteExtension(name, false)` (itkTIFFImageIO.cxx:542-553),
    /// whose `false` is `ignoreCase`, so it compares the extension against
    /// `.tif`/`.TIF`/`.tiff`/`.TIFF` verbatim, so a `.Tif` file finds no writer.
    #[test]
    fn tiff_mixed_case_extension_finds_no_writer() {
        let img = Image::from_vec(&[2, 2], vec![0u8; 4]).unwrap();
        let path = tmp_path("mixed_case.Tif");
        let result = write_image(&img, &path);
        std::fs::remove_file(&path).ok();

        assert!(
            matches!(&result, Err(IoError::NoWriterFound(_))),
            "{result:?}"
        );
    }

    /// `TIFFImageIO::CanReadFile` ignores the file name entirely (beyond rejecting
    /// an empty one) and asks libtiff to open the file
    /// (itkTIFFImageIO.cxx:30-50), so a TIFF under
    /// any extension is claimed for reading — where `PNGImageIO` and the rest
    /// gate on the extension first.
    #[test]
    fn tiff_content_is_claimed_for_reading_under_any_extension() {
        let data: Vec<u8> = (0..12u32).map(|i| i as u8).collect();
        let img = Image::from_vec(&[4, 3], data.clone()).unwrap();
        let written = tmp_path("named.tif");
        write_image(&img, &written).unwrap();

        let renamed = tmp_path("named.unknown-extension");
        std::fs::rename(&written, &renamed).unwrap();
        let io = create_image_io(&renamed, FileMode::Read);
        let back = read_image(&renamed).unwrap();
        std::fs::remove_file(&renamed).ok();

        assert_eq!(io.map(|io| io.name()), Some("TIFFImageIO"));
        assert_eq!(back.scalar_slice::<u8>().unwrap(), data.as_slice());
    }

    /// The other side of that coin: a non-TIFF under a `.tif` name is not
    /// claimed, because the claim is the successful open.
    #[test]
    fn tiff_garbage_bytes_under_a_tiff_extension_are_not_claimed_for_reading() {
        let path = tmp_path("garbage.tif");
        std::fs::write(&path, b"this is not a tiff file, but is long enough\n").unwrap();

        let claimed = create_image_io(&path, FileMode::Read).is_some();
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();

        assert!(!claimed);
        assert!(
            matches!(&result, Err(IoError::NoReaderFound(_))),
            "{result:?}"
        );
    }
}
