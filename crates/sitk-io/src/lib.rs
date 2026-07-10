//! Image file IO for sitk-rs.
//!
//! Phase 0 supports MetaImage (`.mha` / `.mhd`), ITK's native uncompressed
//! format, which round-trips every scalar, vector, and complex pixel type and
//! the full geometry (see [`meta_image`] for the channel-count caveat that
//! applies to complex images). The [`read_image`] / [`write_image`] entry
//! points dispatch on file extension, so adding a format later (PNG via
//! `image`, NIfTI, DICOM) is a new match arm plus a module.

pub mod error;
pub mod meta_image;

use std::path::Path;

pub use error::{IoError, Result};
use sitk_core::Image;

/// Read an image, dispatching on the file extension.
pub fn read_image<P: AsRef<Path>>(path: P) -> Result<Image> {
    let path = path.as_ref();
    match extension_lower(path).as_deref() {
        Some("mha") | Some("mhd") => meta_image::read(path),
        other => Err(IoError::UnknownExtension(other.unwrap_or("").to_string())),
    }
}

/// Write an image, dispatching on the file extension.
pub fn write_image<P: AsRef<Path>>(image: &Image, path: P) -> Result<()> {
    let path = path.as_ref();
    match extension_lower(path).as_deref() {
        Some("mha") | Some("mhd") => meta_image::write(image, path),
        other => Err(IoError::UnknownExtension(other.unwrap_or("").to_string())),
    }
}

fn extension_lower(path: &Path) -> Option<String> {
    path.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
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
        assert_eq!(back, img);
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
        assert_eq!(back, img);
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
        assert_eq!(back, img);
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

    #[test]
    fn unknown_extension_errors() {
        let img = Image::new(&[2, 2], PixelId::UInt8);
        assert!(matches!(
            write_image(&img, tmp_path("x.png")),
            Err(IoError::UnknownExtension(_))
        ));
    }
}
