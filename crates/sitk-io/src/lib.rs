//! Image and transform file IO for sitk-rs.
//!
//! Every format is an [`ImageIo`] implementor sitting in one [`registry`];
//! [`ImageFileReader`] and [`ImageFileWriter`] ask the registry which IO
//! handles a path, exactly as SimpleITK's readers and writers ask
//! `itk::ImageIOFactory`. Adding NIfTI, NRRD, PNG or DICOM later is a new
//! module plus one registry entry — no dispatch to extend. See [`image_io`] for
//! the probe order and [`meta_image`] for the only implementor so far.
//!
//! [`read_image`] and [`write_image`] are the procedural shorthand SimpleITK
//! also provides (`itk::simple::ReadImage` / `WriteImage`).
//!
//! Phase 0 supports MetaImage (`.mha` / `.mhd`), ITK's native uncompressed
//! format, which round-trips every scalar, vector, and complex pixel type and
//! the full geometry (see [`meta_image`] for the channel-count caveat that
//! applies to complex images).
//!
//! Transforms have their own reader and writer, [`read_transform`] and
//! [`write_transform`], over the Insight legacy text format (`.tfm` / `.txt`);
//! see [`transform_io`].

pub mod error;
pub mod image_io;
pub mod meta_image;
pub mod reader;
pub mod transform_io;
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
pub use writer::ImageFileWriter;

/// Read an image, letting the [`registry`] pick the format —
/// `itk::simple::ReadImage` (sitkImageFileReader.cxx:70-78).
///
/// The returned image carries the file's meta-data dictionary.
pub fn read_image<P: AsRef<Path>>(path: P) -> Result<Image> {
    let path = path.as_ref();
    image_io::reader_for(path)?.read(path)
}

/// Write an image, letting the [`registry`] pick the format —
/// `itk::simple::WriteImage`.
pub fn write_image<P: AsRef<Path>>(image: &Image, path: P) -> Result<()> {
    let path = path.as_ref();
    image_io::writer_for(path)?.write(image, path)
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

    /// No registered `ImageIo` advertises `.png`, so `CreateImageIO` returns
    /// null and `ImageFileWriter::GetImageIOBase` throws "Unable to determine
    /// ImageIO writer" (sitkImageFileWriter.cxx:207-210).
    #[test]
    fn unknown_extension_errors() {
        let img = Image::new(&[2, 2], PixelId::UInt8);
        assert!(matches!(
            write_image(&img, tmp_path("x.png")),
            Err(IoError::NoWriterFound(_))
        ));
    }

    // ---- registry --------------------------------------------------------

    /// `ImageFileWriter::GetRegisteredImageIOs` lists `GetNameOfClass`, not
    /// extensions (sitkImageIOUtilities.cxx:59-77).
    #[test]
    fn registry_lists_the_meta_image_io_by_class_name() {
        assert_eq!(registered_image_ios(), vec!["MetaImageIO"]);
        assert_eq!(
            image_io_by_name("MetaImageIO").unwrap().name(),
            "MetaImageIO"
        );
        assert!(matches!(
            image_io_by_name("NiftiImageIO"),
            Err(IoError::UnknownImageIo(name)) if name == "NiftiImageIO"
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

        writer.set_image_io(Some("NiftiImageIO"));
        assert!(matches!(
            writer.execute(&img),
            Err(IoError::UnknownImageIo(_))
        ));
        assert_eq!(writer.registered_image_ios(), vec!["MetaImageIO"]);
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

        assert_eq!(
            img.meta_data_keys(),
            vec![
                "ITK_ExperimentDate",
                "ITK_InputFilterName",
                "ITK_VoxelUnits",
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

        assert_eq!(
            back.meta_data_keys(),
            vec!["ITK_InputFilterName", "Modality"]
        );
        assert_eq!(back.meta_data("Modality"), Some("MET_MOD_UNKNOWN"));
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

    /// A zero-size axis collapses. The output direction is the file direction's
    /// submatrix over the retained axes (`SetDirectionCollapseToSubmatrix`,
    /// sitkImageFileReader.cxx:403), and the origin is shifted by the retained
    /// axes' index through that submatrix (`FixNonZeroIndex`, :39-67). The
    /// collapsed axis's own index selects the slice but never shifts the origin
    /// (itkExtractImageFilter.hxx:162-179).
    #[test]
    fn extract_collapses_a_zero_size_axis_and_keeps_the_direction_submatrix() {
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
        assert_eq!(img.direction(), &[0.0, -1.0, 1.0, 0.0]);
        // origin + D * (spacing .* index) = [10, 20] + [[0,-1],[1,0]] * [1, 2]
        assert_eq!(img.origin(), &[8.0, 21.0]);
        assert_eq!(img.scalar_slice::<i16>().unwrap(), &[13, 14, 16, 17]);
        // The dictionary rides along (sitkImageFileReader.cxx:453).
        assert_eq!(img.meta_data("ITK_InputFilterName"), Some("MetaImageIO"));
    }

    /// The *other* pipeline. With no zero entry the extract size's length
    /// equals the output dimension, so SimpleITK reads the file straight into a
    /// lower-dimensional `itk::Image` (sitkImageFileReader.cxx:362-379) — and
    /// `itk::ImageFileReader` then throws the file's direction cosines away for
    /// `GetDefaultDirection`, the identity (itkImageFileReader.hxx:155-162).
    /// The trailing axis is read at index `0`, so `extract_index[2]` is ignored.
    ///
    /// Same file, same index, one fewer `0` in the size: different direction,
    /// different origin, different pixels than
    /// [`extract_collapses_a_zero_size_axis_and_keeps_the_direction_submatrix`].
    #[test]
    fn extract_without_a_zero_axis_gets_the_identity_direction_and_ignores_the_trailing_index() {
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
        assert_eq!(img.scalar_slice::<i16>().unwrap(), &[4, 5, 7, 8]);
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

    /// `DIRECTIONCOLLAPSETOSUBMATRIX` throws when the retained axes' submatrix
    /// is singular (itkExtractImageFilter.hxx:194-200). A direction that maps
    /// the two retained axes onto the same physical axis does that.
    #[test]
    fn extract_rejects_a_singular_collapsed_direction() {
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
        let result = reader.execute();
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(result, Err(IoError::SingularCollapsedDirection)),
            "{result:?}"
        );
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
}
