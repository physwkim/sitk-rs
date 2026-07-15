//! `DeviceImage::upload` takes a native scalar image and casts it on the device.
//!
//! Two claims, both asserted here rather than assumed:
//!
//! 1. **The device cast is the host cast, to the last bit.** For every scalar pixel
//!    type, uploading the native image and bringing it back must give exactly the
//!    voxels `sitk::filters::cast(img, Float32)` gives. The 64-bit integer cases are
//!    the ones with teeth: the host casts through `f64`, so a value above 2⁵³ is
//!    rounded *twice*, and a device kernel doing a single `(float)x` would disagree.
//!    The kernel therefore goes through `double` too.
//! 2. **A type with no device path is still refused by name.** The convenience of
//!    an automatic cast must not become a silent host conversion for the types the
//!    device cannot take.
#![cfg(feature = "cuda")]

use sitk::core::{Image, PixelId, Scalar};
use sitk::cuda::{CudaError, DeviceImage, backend};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

/// Upload `img` (native type), bring it back, and compare with the host cast —
/// bit for bit.
fn assert_device_cast_matches_host(img: &Image, what: &str) {
    let host = sitk::filters::cast(img, PixelId::Float32).expect("host cast");
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

/// What `upload` must do with a pixel type. Exactly two outcomes — there is no
/// third, and in particular no "converts it on the host for you".
#[derive(Clone, Copy, Debug, PartialEq)]
enum Expect {
    /// A device cast exists, and it is bit-identical to `sitk::filters::cast`.
    Casts,
    /// No device path: `upload` fails with the pixel type in the error.
    RefusedByName,
}

/// The **exhaustive** verdict. A `PixelId` variant added upstream stops this
/// match — and therefore this test — from compiling, which is the point: someone
/// has to decide which of the two outcomes it gets, in this file, on purpose.
fn expect(id: PixelId) -> Expect {
    match id {
        PixelId::UInt8
        | PixelId::Int8
        | PixelId::UInt16
        | PixelId::Int16
        | PixelId::UInt32
        | PixelId::Int32
        | PixelId::UInt64
        | PixelId::Int64
        | PixelId::Float32
        | PixelId::Float64 => Expect::Casts,

        // A `DeviceImage` is one `f32` per voxel; a complex or multi-component
        // pixel does not fit in it, and no device op consumes one.
        PixelId::ComplexFloat32
        | PixelId::ComplexFloat64
        | PixelId::VectorUInt8
        | PixelId::VectorInt8
        | PixelId::VectorUInt16
        | PixelId::VectorInt16
        | PixelId::VectorUInt32
        | PixelId::VectorInt32
        | PixelId::VectorUInt64
        | PixelId::VectorInt64
        | PixelId::VectorFloat32
        | PixelId::VectorFloat64 => Expect::RefusedByName,
    }
}

/// Every `PixelId`, in discriminant order. Kept beside [`expect`] so the compile
/// error a new variant produces there lands next to the list it also belongs in;
/// `every_pixel_id_is_either_uploadable_or_refused_by_name` checks the alignment.
const ALL: [PixelId; 22] = [
    PixelId::UInt8,
    PixelId::Int8,
    PixelId::UInt16,
    PixelId::Int16,
    PixelId::UInt32,
    PixelId::Int32,
    PixelId::UInt64,
    PixelId::Int64,
    PixelId::Float32,
    PixelId::Float64,
    PixelId::ComplexFloat32,
    PixelId::ComplexFloat64,
    PixelId::VectorUInt8,
    PixelId::VectorInt8,
    PixelId::VectorUInt16,
    PixelId::VectorInt16,
    PixelId::VectorUInt32,
    PixelId::VectorInt32,
    PixelId::VectorUInt64,
    PixelId::VectorInt64,
    PixelId::VectorFloat32,
    PixelId::VectorFloat64,
];

/// One 4×4×4 image of pixel type `id`, values chosen only to be distinguishable.
fn sample(id: PixelId) -> Image {
    const N: usize = 64;
    let size = [4usize, 4, 4];
    let ramp = |i: usize| i as f64 - 7.0;
    match id {
        PixelId::UInt8 => Image::from_vec(&size, (0..N).map(|i| i as u8).collect()),
        PixelId::Int8 => Image::from_vec(&size, (0..N).map(|i| ramp(i) as i8).collect()),
        PixelId::UInt16 => Image::from_vec(&size, (0..N).map(|i| (i * 601) as u16).collect()),
        PixelId::Int16 => Image::from_vec(&size, (0..N).map(|i| (ramp(i) * 91.0) as i16).collect()),
        PixelId::UInt32 => Image::from_vec(&size, (0..N).map(|i| (i as u32) << 20).collect()),
        PixelId::Int32 => Image::from_vec(&size, (0..N).map(|i| (ramp(i) as i32) << 20).collect()),
        PixelId::UInt64 => Image::from_vec(&size, (0..N).map(|i| (i as u64) << 50).collect()),
        PixelId::Int64 => Image::from_vec(&size, (0..N).map(|i| (ramp(i) as i64) << 50).collect()),
        PixelId::Float32 => Image::from_vec(&size, (0..N).map(|i| ramp(i) as f32 / 3.0).collect()),
        PixelId::Float64 => Image::from_vec(&size, (0..N).map(|i| ramp(i) / 3.0).collect()),

        PixelId::ComplexFloat32 => Image::from_vec_complex(
            &size,
            (0..N)
                .map(|i| sitk::core::Complex::new(i as f32, -(i as f32)))
                .collect(),
        ),
        PixelId::ComplexFloat64 => Image::from_vec_complex(
            &size,
            (0..N)
                .map(|i| sitk::core::Complex::new(i as f64, -(i as f64)))
                .collect(),
        ),
        PixelId::VectorUInt8 => Image::from_vec_vector(&size, 2, vec![1u8; 2 * N]),
        PixelId::VectorInt8 => Image::from_vec_vector(&size, 2, vec![1i8; 2 * N]),
        PixelId::VectorUInt16 => Image::from_vec_vector(&size, 2, vec![1u16; 2 * N]),
        PixelId::VectorInt16 => Image::from_vec_vector(&size, 2, vec![1i16; 2 * N]),
        PixelId::VectorUInt32 => Image::from_vec_vector(&size, 2, vec![1u32; 2 * N]),
        PixelId::VectorInt32 => Image::from_vec_vector(&size, 2, vec![1i32; 2 * N]),
        PixelId::VectorUInt64 => Image::from_vec_vector(&size, 2, vec![1u64; 2 * N]),
        PixelId::VectorInt64 => Image::from_vec_vector(&size, 2, vec![1i64; 2 * N]),
        PixelId::VectorFloat32 => Image::from_vec_vector(&size, 2, vec![1.0f32; 2 * N]),
        PixelId::VectorFloat64 => Image::from_vec_vector(&size, 2, vec![1.0f64; 2 * N]),
    }
    .unwrap_or_else(|e| panic!("{id:?}: could not build a sample image: {e}"))
}

/// The closure property, over the whole type list: for **every** `PixelId` a user
/// can hold in an `Image`, `upload` either produces the host cast exactly or names
/// the type it will not take. Nothing is silently converted, and nothing is
/// silently refused.
///
/// The refused half needs no GPU (the refusal precedes the driver); the cast half
/// is skipped when there is no device.
#[test]
fn every_pixel_id_is_either_uploadable_or_refused_by_name() {
    for (i, &id) in ALL.iter().enumerate() {
        assert_eq!(id as i8 as usize, i, "{id:?} is out of discriminant order");
    }

    let have_device = !no_device();
    if !have_device {
        println!("no CUDA device: the cast half is skipped, the refusal half still runs");
    }

    for &id in &ALL {
        let img = sample(id);
        assert_eq!(img.pixel_id(), id, "sample({id:?}) built the wrong type");

        match expect(id) {
            Expect::Casts => {
                if have_device {
                    assert_device_cast_matches_host(&img, id.as_str());
                }
            }
            Expect::RefusedByName => match DeviceImage::upload(&img) {
                Err(CudaError::UnsupportedPixelType(named)) => {
                    assert_eq!(named, id, "refused, but named the wrong type");
                    println!("{}: refused by name", id.as_str());
                }
                Err(e) => panic!("{id:?}: refused, but not by name: {e}"),
                Ok(_) => panic!("{id:?}: uploaded; a type with no device path got through"),
            },
        }
    }
}

/// Needs no GPU: the refusal precedes the driver, as it always did.
#[test]
fn a_pixel_type_with_no_device_path_is_still_refused_by_name() {
    // A vector image: there is no device cast for it, and `upload` must say so
    // rather than converting it on the host behind the caller's back.
    let scalar = Image::from_vec(&[4, 4, 4], vec![1.0f32; 64]).unwrap();
    let vector = sitk::filters::compose(&[&scalar, &scalar, &scalar]).expect("vector image");
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
