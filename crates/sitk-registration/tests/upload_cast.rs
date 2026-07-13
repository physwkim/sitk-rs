//! `DeviceImage::upload` takes a native scalar image and casts it on the device.
//!
//! Two claims, both asserted here rather than assumed:
//!
//! 1. **The device cast is the host cast, to the last bit.** For every scalar pixel
//!    type, uploading the native image and bringing it back must give exactly the
//!    voxels `sitk_filters::cast(img, Float32)` gives. The 64-bit integer cases are
//!    the ones with teeth: the host casts through `f64`, so a value above 2⁵³ is
//!    rounded *twice*, and a device kernel doing a single `(float)x` would disagree.
//!    The kernel therefore goes through `double` too.
//! 2. **A type with no device path is still refused by name.** The convenience of
//!    an automatic cast must not become a silent host conversion for the types the
//!    device cannot take.
#![cfg(feature = "cuda")]

use sitk_core::{Image, PixelId, Scalar};
use sitk_cuda::{CudaError, DeviceImage, backend};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

/// Upload `img` (native type), bring it back, and compare with the host cast —
/// bit for bit.
fn assert_device_cast_matches_host(img: &Image, what: &str) {
    let host = sitk_filters::cast(img, PixelId::Float32).expect("host cast");
    let device = DeviceImage::upload(img)
        .unwrap_or_else(|e| panic!("{what}: upload refused: {e}"))
        .to_host()
        .expect("to_host");

    let h = host.scalar_slice::<f32>().unwrap();
    let d = device.scalar_slice::<f32>().unwrap();
    assert_eq!(h.len(), d.len(), "{what}: length");
    let differ = h
        .iter()
        .zip(d.iter())
        .filter(|&(&a, &b)| a.to_bits() != b.to_bits())
        .count();
    let first = h
        .iter()
        .zip(d.iter())
        .enumerate()
        .find(|&(_, (&a, &b))| a.to_bits() != b.to_bits());
    assert_eq!(
        differ,
        0,
        "{what}: {differ}/{} voxels differ; first at {:?}",
        h.len(),
        first.map(|(i, (&a, &b))| (i, a, b))
    );
    println!("{what}: {}/{} voxels bit-identical", h.len(), h.len());
}

fn image_of<T: Scalar>(vals: Vec<T>) -> Image {
    let n = vals.len();
    Image::from_vec(&[n, 1, 1], vals).unwrap()
}

#[test]
fn every_scalar_type_casts_on_the_device_exactly_as_the_host_filter_casts() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }

    assert_device_cast_matches_host(&image_of(vec![0u8, 1, 127, 128, 254, 255]), "UInt8");
    assert_device_cast_matches_host(&image_of(vec![-128i8, -1, 0, 1, 126, 127]), "Int8");
    assert_device_cast_matches_host(&image_of(vec![0u16, 1, 1000, 4095, 32768, 65535]), "UInt16");
    assert_device_cast_matches_host(
        &image_of(vec![-32768i16, -1024, 0, 1, 4095, 32767]),
        "Int16",
    );

    // 32-bit integers: values with more than 24 significant bits, so the f32
    // mantissa must round and the two paths must round the same way.
    assert_device_cast_matches_host(
        &image_of(vec![
            0u32,
            1,
            16_777_216,
            16_777_217,
            16_777_219,
            2_147_483_647,
            4_294_967_295,
        ]),
        "UInt32",
    );
    assert_device_cast_matches_host(
        &image_of(vec![
            i32::MIN,
            -16_777_217,
            -1,
            0,
            16_777_217,
            i32::MAX - 1,
            i32::MAX,
        ]),
        "Int32",
    );

    // The double-rounding cases. Above 2^53 the host's `native -> f64 -> f32` and a
    // naive device `native -> f32` can land on different f32 values; these inputs
    // straddle that boundary.
    assert_device_cast_matches_host(
        &image_of(vec![
            0u64,
            1,
            (1u64 << 53) - 1,
            1u64 << 53,
            (1u64 << 53) + 1,
            (1u64 << 63) + (1u64 << 39),
            u64::MAX,
        ]),
        "UInt64",
    );
    assert_device_cast_matches_host(
        &image_of(vec![
            i64::MIN,
            -(1i64 << 53) - 1,
            -1,
            0,
            (1i64 << 53) + 1,
            i64::MAX - 512,
            i64::MAX,
        ]),
        "Int64",
    );

    // f64 -> f32: subnormal, huge (overflows to inf), and a value that needs the
    // round-to-nearest-even tie rule.
    assert_device_cast_matches_host(
        &image_of(vec![
            0.0f64,
            -0.0,
            1.0,
            1.0 / 3.0,
            f64::MIN_POSITIVE,
            1e-45,
            3.402_823_5e38,
            1e39,
            -1e39,
            std::f64::consts::PI,
        ]),
        "Float64",
    );

    // Float32 is the device type: nothing to cast, but the path must still be exact.
    assert_device_cast_matches_host(
        &image_of(vec![0.0f32, -0.0, 1.0, -1.5, f32::MAX, f32::MIN_POSITIVE]),
        "Float32",
    );
}

/// Needs no GPU: the refusal precedes the driver, as it always did.
#[test]
fn a_pixel_type_with_no_device_path_is_still_refused_by_name() {
    // A vector image: there is no device cast for it, and `upload` must say so
    // rather than converting it on the host behind the caller's back.
    let scalar = Image::from_vec(&[4, 4, 4], vec![1.0f32; 64]).unwrap();
    let vector = sitk_filters::compose(&[&scalar, &scalar, &scalar]).expect("vector image");
    assert_ne!(vector.pixel_id(), PixelId::Float32);

    match DeviceImage::upload(&vector) {
        Err(CudaError::UnsupportedPixelType(id)) => {
            assert_eq!(id, vector.pixel_id());
            println!("refused by name: {}", CudaError::UnsupportedPixelType(id));
        }
        Err(e) => panic!("refused, but not by name: {e}"),
        Ok(_) => panic!("a vector image was uploaded; the refusal is gone"),
    }
}
