//! The metric reads its **fixed** samples as `f32` and its **moving** image as `f64`
//! on the device-resident path, against `f64`/`f64` on the host-producer path, and
//! the claim is that this changes **nothing**.
//!
//! `ResidentMetric::from_device` narrows the fixed volume (one load per sample) and
//! leaves the moving volume wide (eight loads per sample, the trilinear corners),
//! which banks half the memory saving for a conversion cost that does not show above
//! the noise — see `from_device`'s docs for the three-way measurement. Whatever the
//! layout, the kernel widens each load where it reduces, and every multiply, add and
//! accumulator stays `double`.
//!
//! That is only sound if `(double)x` is exact for every `f32`, which it is: `f64`
//! has strictly more exponent range and mantissa bits than `f32`, so the widening
//! is a lossless re-encoding of the same real number and the arithmetic that
//! follows sees the identical operands. This test asserts it rather than trusting
//! it — the same voxels, through both instantiations of the kernel, must produce
//! moments that agree **to the last bit**. This is the gate on the layout: any
//! future change to which volume is narrowed has to keep passing it.
#![cfg(feature = "cuda")]

use sitk::core::{Image, PixelId};
use sitk::cuda::{
    CudaError, DeviceImage, FixedPoints, Moments, MovingGeometry, PointStage, ResidentMetric,
    backend,
};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

/// A deterministic `f32` volume with structure (so the trilinear taps and the
/// gradient are not all equal) and a wide dynamic range (so the reduction has
/// something to round).
fn volume(n: usize, shift: f64) -> Image {
    let c = n as f64 / 2.0;
    let mut v = Vec::with_capacity(n * n * n);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (x as f64 - shift, y as f64, z as f64);
                let r = ((fx - c).powi(2) + (fy - c).powi(2) + (fz - c).powi(2)).sqrt();
                let s = 2000.0 * (-(r * r) / (0.18 * n as f64).powi(2)).exp()
                    + 137.0 * (0.4 * r).sin()
                    + 0.5;
                v.push(s as f32);
            }
        }
    }
    Image::from_vec(&[n, n, n], v).unwrap()
}

fn f32_slice(img: &Image) -> &[f32] {
    assert_eq!(img.pixel_id(), PixelId::Float32);
    img.scalar_slice::<f32>().unwrap()
}

/// Unit spacing, zero origin, identity direction — so `index_to_physical` and
/// `physical_to_index` are both the identity and the geometry contributes no
/// rounding of its own to what is being compared.
fn geometry(n: usize) -> ([usize; 3], [usize; 3], [f64; 3], [f64; 9]) {
    (
        [n, n, n],
        [1, n, n * n],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
}

/// The point map of a small rotation + translation: generic enough that every
/// sample lands off-grid and all eight trilinear corners contribute.
const A: [f64; 9] = [
    0.999_950_000_416_665_3,
    -0.009_999_833_334_166_664,
    0.0,
    0.009_999_833_334_166_664,
    0.999_950_000_416_665_3,
    0.0,
    0.0,
    0.0,
    1.0,
];
const B: [f64; 3] = [1.25, -0.75, 0.5];

/// The point map as the metric takes it: one stage of `mat_vec(matrix, p) + offset`, which
/// is what a single matrix-offset transform hands over.
const MAP: [PointStage; 1] = [PointStage {
    matrix: A,
    offset: B,
}];

fn moments_f64(fixed: &Image, moving: &Image, n: usize) -> Moments {
    let (size, strides, origin, mat) = geometry(n);
    let fvals = f32_slice(fixed).to_vec();
    let mvals = f32_slice(moving).to_vec();
    let mg = MovingGeometry {
        len: n * n * n,
        size: &size,
        strides: &strides,
        origin: &origin,
        phys_to_index: &mat,
        mask: None,
    };
    let mut m = ResidentMetric::new(
        n * n * n,
        |start, out: &mut [f64]| {
            for (k, o) in out.iter_mut().enumerate() {
                *o = f64::from(fvals[start + k]);
            }
        },
        FixedPoints::Grid {
            size: &size,
            origin: &origin,
            idx_to_phys: &mat,
        },
        &mg,
        |start, out: &mut [f64]| {
            for (k, o) in out.iter_mut().enumerate() {
                *o = f64::from(mvals[start + k]);
            }
        },
    )
    .unwrap();
    m.evaluate(&MAP).unwrap()
}

fn moments_split(fixed: &Image, moving: &Image, n: usize) -> Moments {
    let (size, strides, origin, mat) = geometry(n);
    let mg = MovingGeometry {
        len: n * n * n,
        size: &size,
        strides: &strides,
        origin: &origin,
        phys_to_index: &mat,
        mask: None,
    };
    let mut m = ResidentMetric::from_device(
        &DeviceImage::upload(fixed).unwrap(),
        FixedPoints::Grid {
            size: &size,
            origin: &origin,
            idx_to_phys: &mat,
        },
        &DeviceImage::upload(moving).unwrap(),
        &mg,
    )
    .unwrap();
    m.evaluate(&MAP).unwrap()
}

#[test]
fn the_split_volumes_give_bit_identical_moments_to_the_f64_volumes() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 48;
    let (fixed, moving) = (volume(n, 0.0), volume(n, 2.5));

    let wide = moments_f64(&fixed, &moving, n);
    let narrow = moments_split(&fixed, &moving, n);

    println!("f64 volumes: sq {:.17e}  count {}", wide.sq, wide.count);
    println!(
        "f32 fixed / f64 moving: sq {:.17e}  count {}",
        narrow.sq, narrow.count
    );
    assert_eq!(
        wide.count, narrow.count,
        "the two kernels disagreed about which samples are inside"
    );
    assert_eq!(
        wide.sq.to_bits(),
        narrow.sq.to_bits(),
        "Σdiff² moved: f64/f64 {:.17e} vs split {:.17e}",
        wide.sq,
        narrow.sq
    );
    for d in 0..3 {
        assert_eq!(
            wide.s0[d].to_bits(),
            narrow.s0[d].to_bits(),
            "S0[{d}] moved: {:.17e} vs {:.17e}",
            wide.s0[d],
            narrow.s0[d]
        );
        for e in 0..3 {
            assert_eq!(
                wide.s1[d][e].to_bits(),
                narrow.s1[d][e].to_bits(),
                "S1[{d}][{e}] moved: {:.17e} vs {:.17e}",
                wide.s1[d][e],
                narrow.s1[d][e]
            );
        }
    }
}

#[test]
fn the_split_metric_is_bit_identical_run_to_run() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 48;
    let (fixed, moving) = (volume(n, 0.0), volume(n, 2.5));

    let first = moments_split(&fixed, &moving, n);
    for run in 1..4 {
        let again = moments_split(&fixed, &moving, n);
        assert_eq!(first.sq.to_bits(), again.sq.to_bits(), "run {run}: Σdiff²");
        assert_eq!(first.count, again.count, "run {run}: count");
        for d in 0..3 {
            assert_eq!(
                first.s0[d].to_bits(),
                again.s0[d].to_bits(),
                "run {run}: S0"
            );
            for e in 0..3 {
                assert_eq!(
                    first.s1[d][e].to_bits(),
                    again.s1[d][e].to_bits(),
                    "run {run}: S1"
                );
            }
        }
    }
}
