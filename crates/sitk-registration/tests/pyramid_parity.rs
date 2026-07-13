//! The three device pyramid ops against the three CPU filters they transcribe.
//!
//! The contract is **bit-identity**, not a tolerance. A registration pyramid level
//! is built by smoothing, shrinking to get the coarse grid, and resampling onto it;
//! if any of the three drifts from the host, the device's coarse level is a
//! different image and every claim about "landing where `execute` lands" becomes a
//! claim about how much drift is tolerable. The kernels are transcriptions, they
//! compile with multiply-add contraction off, and these tests hold them to it.
//!
//! `shrink` and `resample` are also asserted to be *different operations* — equal
//! at factor 1, disagreeing at factor 2 — because collapsing them into one is the
//! mistake that introduces a sub-voxel translation bias into a pyramid.
#![cfg(feature = "cuda")]

use sitk_core::Image;
use sitk_cuda::{CudaError, DeviceImage, Geometry, backend};
use sitk_registration::metric::{FixedSamples, MovingImage};
use sitk_registration::{CpuBackend, DeviceMeanSquaresMetric, MeanSquaresMetric};
use sitk_transform::{AffineTransform, Euler3DTransform, Interpolator, ResampleImageFilter};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

/// A volume with structure at several scales, so smoothing and interpolation have
/// something to disagree about, plus a non-trivial spacing and origin.
fn volume(n: usize) -> Image {
    let c = n as f64 / 2.0;
    let mut v = Vec::with_capacity(n * n * n);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (x as f64, y as f64, z as f64);
                let r = ((fx - c).powi(2) + (fy - c).powi(2) + (fz - c).powi(2)).sqrt();
                let s = 2000.0 * (-(r * r) / (0.18 * n as f64).powi(2)).exp()
                    + 200.0 * (0.4 * r).sin()
                    + 37.0 * ((fx * 0.9).sin() + (fy * 1.3).cos() + (fz * 0.7).sin())
                    + 400.0;
                v.push(s as f32);
            }
        }
    }
    let mut img = Image::from_vec(&[n, n, n], v).unwrap();
    img.set_spacing(&[0.7, 1.1, 1.3]).unwrap();
    img.set_origin(&[-12.0, 5.5, 3.25]).unwrap();
    img
}

/// Compare two `Float32` images voxel for voxel, on the bits, and their geometry.
fn assert_bit_identical(host: &Image, device: &Image, what: &str) {
    assert_eq!(host.size(), device.size(), "{what}: size");
    assert_eq!(host.spacing(), device.spacing(), "{what}: spacing");
    assert_eq!(host.origin(), device.origin(), "{what}: origin");
    assert_eq!(host.direction(), device.direction(), "{what}: direction");

    let h = host.scalar_slice::<f32>().unwrap();
    let d = device.scalar_slice::<f32>().unwrap();
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
        first.map(|(i, (&a, &b))| (i, a, b, (a - b).abs()))
    );
    println!("{what}: {}/{} voxels bit-identical", h.len(), h.len());
}

#[test]
fn the_device_recursive_gaussian_is_bit_identical_to_the_host_filter() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(48);

    for sigma in [
        vec![2.0, 2.0, 2.0],
        vec![1.0, 0.0, 3.5],  // a zero axis is skipped, as on the host
        vec![0.0, 0.0, 0.0],  // no smoothing at all: a copy
        vec![0.31, 4.7, 0.9], // anisotropic, and small enough to stress the poles
    ] {
        let host = sitk_filters::recursive_gaussian(&img, &sigma).expect("host gaussian");
        let device = sitk_cuda::recursive_gaussian(&DeviceImage::upload(&img).unwrap(), &sigma)
            .expect("device gaussian")
            .to_host()
            .unwrap();
        assert_bit_identical(&host, &device, &format!("recursive_gaussian {sigma:?}"));
    }
}

#[test]
fn the_device_shrink_is_bit_identical_to_the_host_filter() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    // 48 is divisible by 2, 3, 4 and not by 5 or 7 — so the last group covers the
    // ragged case, where `out_size` truncates and the sampling offset is clamped.
    let img = volume(48);

    for factors in [
        vec![1usize, 1, 1],
        vec![2, 2, 2],
        vec![4, 2, 1],
        vec![3, 5, 7],
    ] {
        let host = sitk_filters::shrink(&img, &factors).expect("host shrink");
        let device = sitk_cuda::shrink(&DeviceImage::upload(&img).unwrap(), &factors)
            .expect("device shrink")
            .to_host()
            .unwrap();
        assert_bit_identical(&host, &device, &format!("shrink {factors:?}"));
    }
}

#[test]
fn the_device_resample_is_bit_identical_to_the_host_filter() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(48);

    // The grids a pyramid actually resamples onto: the shrunk grids. Plus the
    // image's own grid, where the resample must be the identity.
    for factors in [vec![1usize, 1, 1], vec![2, 2, 2], vec![3, 5, 7]] {
        let grid = sitk_filters::shrink(&img, &factors).expect("host shrink");

        let mut resampler = ResampleImageFilter::new();
        resampler
            .set_reference_image(&grid)
            .set_interpolator(Interpolator::Linear)
            .set_default_pixel_value(0.0);
        let host = resampler
            .execute(&img, &AffineTransform::identity(3))
            .expect("host resample");

        let device = sitk_cuda::resample_linear(
            &DeviceImage::upload(&img).unwrap(),
            &Geometry::of(&grid),
            0.0,
        )
        .expect("device resample")
        .to_host()
        .unwrap();

        assert_bit_identical(&host, &device, &format!("resample onto shrink {factors:?}"));
    }
}

/// The two ops are **not** interchangeable, and the device keeps them apart. They
/// agree at factor 1 (every output voxel sits exactly on an input voxel) and
/// disagree everywhere at factor 2 (the coarse voxel centers fall between input
/// voxels, and the sampling offset is the rounded shift while the origin carries
/// the unrounded one).
#[test]
fn the_device_shrink_and_the_device_resample_are_different_operations() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(48);
    let src = DeviceImage::upload(&img).unwrap();

    for (factors, expect_equal) in [(vec![1usize, 1, 1], true), (vec![2, 2, 2], false)] {
        let shrunk = sitk_cuda::shrink(&src, &factors).unwrap();
        let grid = shrunk.geometry().clone();
        let resampled = sitk_cuda::resample_linear(&src, &grid, 0.0).unwrap();

        let a = shrunk.to_host().unwrap();
        let b = resampled.to_host().unwrap();
        let (a, b) = (
            a.scalar_slice::<f32>().unwrap().to_vec(),
            b.scalar_slice::<f32>().unwrap().to_vec(),
        );
        let same = a
            .iter()
            .zip(b.iter())
            .filter(|&(&x, &y)| x.to_bits() == y.to_bits())
            .count();

        if expect_equal {
            assert_eq!(same, a.len(), "factor 1: shrink and resample must agree");
            println!("factor 1: shrink == resample on all {} voxels", a.len());
        } else {
            assert!(
                same * 100 < a.len(),
                "factor 2: shrink and resample agreed on {same}/{} voxels — \
                 they are being collapsed into one operation",
                a.len()
            );
            println!(
                "factor 2: shrink and resample agree on only {same}/{} voxels, as they must not",
                a.len()
            );
        }
    }
}

/// **The level-for-level gate.** Build every level of a real pyramid schedule on
/// both paths — smooth, shrink for the grid, resample onto it — and compare the
/// *metric* the optimizer would see at that level, at several transforms.
///
/// This is the claim `execute_on_device` rests on. The end-to-end walk cannot carry
/// it: a gradient descent that branches discretely turns a 1e-12 metric difference
/// into a different trajectory (§2.157), so an end-to-end comparison measures the
/// optimizer's conditioning and not the pyramid. What the pyramid must guarantee is
/// that *the objective at each level is the same objective*, and that is what is
/// asserted here — to 1e-9 relative, the band the device metric is already gated at.
#[test]
fn every_pyramid_level_is_the_same_objective_on_both_paths() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let fixed = volume(n);
    let moving = {
        // The same field, translated — so the metric has a real gradient.
        let mut m = volume(n);
        let o = m.origin().to_vec();
        m.set_origin(&[o[0] + 2.1, o[1] - 1.4, o[2] + 0.8]).unwrap();
        m
    };

    let d_fixed = DeviceImage::upload(&fixed).unwrap();
    let d_moving = DeviceImage::upload(&moving).unwrap();

    // The schedule `execute` would run: coarsest first, physical sigma = level sigma
    // times the fixed image's spacing (the default, `smoothing_sigmas_in_physical_
    // units == false`).
    let schedule = [(4usize, 2.0f64), (2, 1.0), (1, 0.0)];
    let c = n as f64 / 2.0;
    let transforms = [
        Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c]),
        Euler3DTransform::new(0.05, -0.03, 0.02, [1.5, -0.9, 0.4], [c, c, c]),
        Euler3DTransform::new(-0.11, 0.07, -0.05, [-2.2, 1.7, -1.1], [c, c, c]),
    ];

    for (factor, level_sigma) in schedule {
        let factors = vec![factor; 3];
        let sigma: Vec<f64> = fixed.spacing().iter().map(|&s| level_sigma * s).collect();

        // ---- the host level, exactly as `prepare_level` builds it ----
        let host_fixed_smoothed = if level_sigma == 0.0 {
            fixed.clone()
        } else {
            sitk_filters::recursive_gaussian(&fixed, &sigma).unwrap()
        };
        let host_moving_level = if level_sigma == 0.0 {
            moving.clone()
        } else {
            sitk_filters::recursive_gaussian(&moving, &sigma).unwrap()
        };
        let host_fixed_level = if factor == 1 {
            host_fixed_smoothed.clone()
        } else {
            let grid = sitk_filters::shrink(&host_fixed_smoothed, &factors).unwrap();
            let mut r = ResampleImageFilter::new();
            r.set_reference_image(&grid)
                .set_interpolator(Interpolator::Linear)
                .set_default_pixel_value(0.0);
            r.execute(&host_fixed_smoothed, &AffineTransform::identity(3))
                .unwrap()
        };

        // ---- the device level, exactly as `prepare_level_on_device` builds it ----
        let dev_fixed_smoothed = if level_sigma == 0.0 {
            None
        } else {
            Some(sitk_cuda::recursive_gaussian(&d_fixed, &sigma).unwrap())
        };
        let dev_moving_level = if level_sigma == 0.0 {
            None
        } else {
            Some(sitk_cuda::recursive_gaussian(&d_moving, &sigma).unwrap())
        };
        let sf = dev_fixed_smoothed.as_ref().unwrap_or(&d_fixed);
        let dev_fixed_level = if factor == 1 {
            None
        } else {
            let coarse = sitk_cuda::shrink(sf, &factors).unwrap();
            Some(sitk_cuda::resample_linear(sf, coarse.geometry(), 0.0).unwrap())
        };

        // The level images themselves, first: they are the input to everything else.
        assert_bit_identical(
            &host_fixed_level,
            &dev_fixed_level.as_ref().unwrap_or(sf).to_host().unwrap(),
            &format!("level (factor {factor}, sigma {level_sigma}) fixed image"),
        );
        assert_bit_identical(
            &host_moving_level,
            &dev_moving_level
                .as_ref()
                .unwrap_or(&d_moving)
                .to_host()
                .unwrap(),
            &format!("level (factor {factor}, sigma {level_sigma}) moving image"),
        );

        // ---- and the metric each path's optimizer would actually see ----
        let host_metric = MeanSquaresMetric::from_samples(
            FixedSamples::from_image(&host_fixed_level).unwrap(),
            MovingImage::from_image(&host_moving_level).unwrap(),
        )
        .unwrap();
        let device_metric = DeviceMeanSquaresMetric::from_device(
            dev_fixed_level.as_ref().unwrap_or(sf),
            dev_moving_level.as_ref().unwrap_or(&d_moving),
        )
        .unwrap();

        for t in &transforms {
            let h = host_metric.evaluate(t, &CpuBackend);
            let d = device_metric.evaluate(t).unwrap();

            assert_eq!(
                d.valid_points, h.valid_points,
                "factor {factor}: the two paths sampled different points"
            );
            assert!(h.valid_points > 0);

            let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
            let v = rel(d.value, h.value);
            let g = d
                .derivative
                .iter()
                .zip(h.derivative.iter())
                .map(|(&x, &y)| rel(x, y))
                .fold(0.0f64, f64::max);
            println!(
                "factor {factor}, sigma {level_sigma}: {} valid, value rel {v:e}, \
                 derivative rel {g:e}",
                h.valid_points
            );
            assert!(
                v <= 1e-9,
                "factor {factor}: value rel err {v:e} exceeds 1e-9"
            );
            assert!(
                g <= 1e-9,
                "factor {factor}: derivative rel err {g:e} exceeds 1e-9"
            );
            assert!(
                h.derivative.iter().any(|d| d.abs() > 1e-6),
                "the derivative is ~zero here, so the comparison proves nothing"
            );
        }
    }
}

/// The refusals, by name. Needs no GPU for the shape checks that precede the
/// driver — but `upload` needs one, so this is skipped without a device.
#[test]
fn the_pyramid_ops_refuse_a_shape_they_have_no_kernel_for() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }

    // Three voxels on an axis the recursion must smooth: the fourth-order
    // recursion needs four, and the CPU filter refuses this too.
    let tiny = Image::from_vec(&[3usize, 8, 8], vec![1.0f32; 3 * 8 * 8]).unwrap();
    let d = DeviceImage::upload(&tiny).unwrap();
    match sitk_cuda::recursive_gaussian(&d, &[1.0, 1.0, 1.0]) {
        Err(CudaError::UnsupportedGeometry(why)) => println!("refused: {why}"),
        other => panic!("a 3-voxel axis was smoothed: {:?}", other.map(|_| ())),
    }
    assert!(sitk_filters::recursive_gaussian(&tiny, &[1.0, 1.0, 1.0]).is_err());

    // ... but a sigma of zero on that axis smooths nothing, and is fine — on both.
    assert!(sitk_cuda::recursive_gaussian(&d, &[0.0, 1.0, 1.0]).is_ok());
    assert!(sitk_filters::recursive_gaussian(&tiny, &[0.0, 1.0, 1.0]).is_ok());

    match sitk_cuda::shrink(&d, &[2, 0, 2]) {
        Err(CudaError::UnsupportedGeometry(why)) => println!("refused: {why}"),
        other => panic!("a zero shrink factor was accepted: {:?}", other.map(|_| ())),
    }

    let two_d = Image::from_vec(&[8usize, 8], vec![1.0f32; 64]).unwrap();
    let d2 = DeviceImage::upload(&two_d).unwrap();
    match sitk_cuda::shrink(&d2, &[2, 2]) {
        Err(CudaError::UnsupportedGeometry(why)) => println!("refused: {why}"),
        other => panic!("a 2-D image was shrunk: {:?}", other.map(|_| ())),
    }
}
