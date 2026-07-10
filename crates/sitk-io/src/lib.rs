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
//! Three uncompressed formats are supported:
//!
//! * [`meta_image`] — MetaImage (`.mha`, `.mhd` + `.raw`), ITK's native format.
//!   Round-trips every scalar and vector pixel type and the full geometry; a
//!   complex image survives as a two-channel vector image (see that module for
//!   the upstream quirk).
//! * [`nrrd`] — NRRD (`.nrrd` / `.nhdr`), raw encoding only, which does
//!   round-trip a complex image because its `kinds` field records the
//!   distinction.
//! * [`nifti`] — NIfTI-1 (`.nii`, `.hdr` + `.img`). Round-trips every scalar
//!   pixel type, vector images, and complex images as complex. `.nii.gz` is
//!   recognised and rejected; this workspace has no zlib.
//!
//! Transforms have their own reader and writer, [`read_transform`] and
//! [`write_transform`], over the Insight legacy text format (`.tfm` / `.txt`);
//! see [`transform_io`].

pub mod error;
pub mod gipl;
pub mod image_io;
pub mod meta_image;
pub mod nifti;
pub mod nrrd;
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
    fn registry_lists_each_image_io_by_class_name() {
        assert_eq!(
            registered_image_ios(),
            vec!["MetaImageIO", "NrrdImageIO", "NiftiImageIO", "GiplImageIO"]
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
        assert!(matches!(
            image_io_by_name("PNGImageIO"),
            Err(IoError::UnknownImageIo(name)) if name == "PNGImageIO"
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

        writer.set_image_io(Some("PNGImageIO"));
        assert!(matches!(
            writer.execute(&img),
            Err(IoError::UnknownImageIo(_))
        ));
        assert_eq!(
            writer.registered_image_ios(),
            vec!["MetaImageIO", "NrrdImageIO", "NiftiImageIO", "GiplImageIO"]
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

    /// `RescaleFunction(buffer, slope, inter, numElts)` is handed the *voxel*
    /// count, not the component count (itkNiftiImageIO.cxx:513-548), so on a
    /// complex image only the first `numElts` of the `2 * numElts` interleaved
    /// floats are rescaled: the first half of the pixels get both parts scaled,
    /// the second half get neither. Upstream bug, reproduced (ledger §1.50).
    #[test]
    fn nii_rescale_of_a_complex_image_only_touches_the_first_half_of_the_buffer() {
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
            // components 0..5 scaled by 10, components 6..11 untouched
            &[
                10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0
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

    /// This build has no zlib, so a gzipped NIfTI is recognised and refused with
    /// a message that names what is missing.
    #[test]
    fn nii_gz_is_recognised_and_rejected() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("compressed.nii.gz");

        let written = write_image(&img, &path);
        assert!(
            matches!(&written, Err(IoError::UnsupportedNiftiFeature(m))
                if m.contains("zlib") && m.contains("5.8")),
            "{written:?}"
        );

        std::fs::write(&path, b"\x1f\x8b not really gzip").unwrap();
        assert!(create_image_io(&path, FileMode::Read).is_some());
        let read = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&read, Err(IoError::UnsupportedNiftiFeature(m))
                if m.contains("zlib") && m.contains("5.8")),
            "{read:?}"
        );
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

    /// Upstream bug §1.47: `nrrdOriginCalculate`'s `gotMin` loop reads
    /// `axis[0]->min` on every iteration, so a NaN on axis 1 does not produce
    /// the `NoMin` status it should — the NaN reaches the origin instead.
    #[test]
    fn nrrd_origin_calculate_only_checks_the_first_axis_min() {
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
        assert_eq!(img.origin()[0], 11.0);
        assert!(img.origin()[1].is_nan(), "upstream leaks the NaN min");
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

    /// The compressed encodings are recognised and rejected by name, since this
    /// workspace takes no compression dependency (ledger §5.8).
    #[test]
    fn nrrd_rejects_compressed_encodings() {
        for (encoding, needle) in [("gzip", "gzip"), ("gz", "gzip"), ("bzip2", "bzip2")] {
            let path = tmp_path(&format!("compressed_{encoding}.nrrd"));
            std::fs::write(
                &path,
                format!(
                    "NRRD0004\ntype: unsigned char\ndimension: 1\nsizes: 2\n\
                     encoding: {encoding}\n\nxx"
                ),
            )
            .unwrap();
            let err = read_image(&path).unwrap_err();
            std::fs::remove_file(&path).ok();
            match err {
                IoError::UnsupportedNrrdFeature(message) => {
                    assert!(message.contains(needle), "{message}");
                    assert!(message.contains("compression"), "{message}");
                }
                other => panic!("expected UnsupportedNrrdFeature, got {other:?}"),
            }
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
        // `pixdim` is `float`, so a spacing that is not exactly representable in
        // f32 does not survive; these three are.
        assert!(back.meta_data_keys().is_empty());
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

    /// `SwapBytesIfNecessary` has no `INT`/`UINT` arm, so `Write` throws
    /// `"Pixel Type Unknown"` — *after* the full 256-byte header is on disk
    /// (§1.52). The half-written file is left behind, exactly as upstream's
    /// truncating `OpenFileForWriting` plus unwinding `ofstream` destructor
    /// leaves it.
    #[test]
    fn gipl_int32_write_leaves_the_header_and_then_fails() {
        let img = Image::from_vec(&[2, 2], vec![1i32, 2, 3, 4]).unwrap();
        let path = tmp_path("int32.gipl");
        let result = write_image(&img, &path);
        let written = std::fs::read(&path).unwrap();

        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.starts_with("Pixel Type Unknown")),
            "{result:?}"
        );
        assert_eq!(written.len(), gipl::HEADER_SIZE);
        assert_eq!(&written[8..10], &32u16.to_be_bytes()); // GIPL_INT was written

        // And reading such a file fails on the same missing swap arm.
        let read_back = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&read_back, Err(IoError::UnsupportedGiplFeature(m)) if m.starts_with("Pixel Type Unknown")),
            "{read_back:?}"
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

    /// `CheckExtension` claims `.gipl.gz` for reading and writing; upstream then
    /// reaches for zlib. This workspace has none (§5.8), so both fail naming it.
    #[test]
    fn gipl_gz_is_claimed_and_then_refused_for_the_missing_zlib() {
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let path = tmp_path("compressed.gipl.gz");

        assert!(create_image_io(&path, FileMode::Write).is_some());
        let result = write_image(&img, &path);
        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.contains("zlib")),
            "{result:?}"
        );
        assert!(!path.exists(), "write must not create the file");

        std::fs::write(&path, b"\x1f\x8b not really gzip").unwrap();
        assert!(create_image_io(&path, FileMode::Read).is_some());
        let result = read_image(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(&result, Err(IoError::UnsupportedGiplFeature(m)) if m.contains("zlib")),
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
    /// the buffer tail uninitialised (§1.53); this port refuses it (§4.65).
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
    /// locals indeterminate; refused here (§4.69).
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
}
