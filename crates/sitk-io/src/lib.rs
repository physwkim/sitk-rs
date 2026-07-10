//! Image file IO for sitk-rs.
//!
//! Phase 0 supports MetaImage (`.mha` / `.mhd`), ITK's native uncompressed
//! format, which round-trips every scalar pixel type and the full geometry. The
//! [`read_image`] / [`write_image`] entry points dispatch on file extension, so
//! adding a format later (PNG via `image`, NIfTI, DICOM) is a new match arm plus
//! a module.

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
    use sitk_core::{Image, PixelId};

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

    /// The header hard-codes `ElementNumberOfChannels = 1`, so anything whose
    /// buffer holds more than one component per pixel must be refused before a
    /// file is created. Complex is the case a `is_vector()` blacklist would
    /// have let through: upstream it is a *basic* pixel type, yet its buffer
    /// carries two components per pixel.
    #[test]
    fn write_rejects_every_non_scalar_pixel_type() {
        for id in [PixelId::ComplexFloat32, PixelId::ComplexFloat64] {
            let img = Image::new(&[2, 2], id);
            let path = tmp_path("nonscalar.mha");
            assert!(matches!(
                write_image(&img, &path),
                Err(IoError::Unsupported(_))
            ));
            assert!(!path.exists(), "{id:?}: rejected write left a file behind");
        }
        let img = Image::new_vector(&[2, 2], PixelId::VectorFloat32, 3).unwrap();
        assert!(matches!(
            write_image(&img, tmp_path("nonscalar.mha")),
            Err(IoError::Unsupported(_))
        ));
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
