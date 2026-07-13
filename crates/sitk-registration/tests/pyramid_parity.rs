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
use sitk_transform::{
    AffineTransform, ComposeScaleSkewVersor3DTransform, Euler3DTransform, Interpolator,
    ResampleImageFilter, ScaleSkewVersor3DTransform, ScaleVersor3DTransform, Similarity3DTransform,
    Transform, TranslationTransform, VersorRigid3DTransform, VersorTransform,
};

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
/// A 0/1 ball on `img`'s grid — the payload the nearest resample exists for, and the
/// one where an index error is invisible in the values.
fn binary_mask(img: &Image) -> Image {
    let n = img.size()[0];
    let mut v = vec![0.0f32; n * n * n];
    for (s, x) in v.iter_mut().enumerate() {
        let (i, j, k) = (s % n, (s / n) % n, s / (n * n));
        let c = n as f64 / 2.0;
        let r = ((i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2)).sqrt();
        *x = if r < 0.3 * n as f64 { 1.0 } else { 0.0 };
    }
    let mut m = Image::from_vec(&[n, n, n], v).unwrap();
    m.set_spacing(img.spacing()).unwrap();
    m.set_origin(img.origin()).unwrap();
    m.set_direction(img.direction()).unwrap();
    m
}

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

/// The nearest-neighbour resample, pinned **before** it is wired to anything.
///
/// A mask's values are 0 and 1, so an arithmetic error in this op is invisible in
/// the values it produces — the failure is entirely in the index arithmetic, and it
/// surfaces only later, as a metric valid-sample count that differs by a shell of
/// boundary voxels. So this pins the op directly, and against a **textured** volume
/// as well as a binary one: on a smooth volume an off-by-one index changes the
/// value and is caught, where on a 0/1 mask it usually would not be.
///
/// The third grid is the one that matters. It is offset by exactly **half a voxel**,
/// so every continuous index is an exact half-integer — the tie. Round-half-to-even
/// (`rint`) would answer differently from ITK's `RoundHalfIntegerUp` at every single
/// sample of it, and no other grid in this file would notice.
#[test]
fn the_device_nearest_resample_is_bit_identical_to_the_host_filter() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(48);

    // A binary mask on the same grid — the payload this op exists for.
    let mask = binary_mask(&img);

    // The grid whose every sample lands on a half-integer continuous index.
    let half_voxel = {
        let mut g = img.clone();
        let o: Vec<f64> = img
            .origin()
            .iter()
            .zip(img.spacing().iter())
            .map(|(&o, &sp)| o + 0.5 * sp)
            .collect();
        g.set_origin(&o).unwrap();
        g
    };

    let grids = [
        (
            "shrink [1,1,1]",
            sitk_filters::shrink(&img, &[1, 1, 1]).unwrap(),
        ),
        (
            "shrink [2,2,2]",
            sitk_filters::shrink(&img, &[2, 2, 2]).unwrap(),
        ),
        (
            "shrink [3,5,7]",
            sitk_filters::shrink(&img, &[3, 5, 7]).unwrap(),
        ),
        ("half-voxel offset (every index is a tie)", half_voxel),
    ];

    for (name, grid) in &grids {
        for (payload, src) in [("textured", &img), ("binary mask", &mask)] {
            let mut resampler = ResampleImageFilter::new();
            resampler
                .set_reference_image(grid)
                .set_interpolator(Interpolator::NearestNeighbor)
                .set_default_pixel_value(0.0);
            let host = resampler
                .execute(src, &AffineTransform::identity(3))
                .expect("host nearest resample");

            let device = sitk_cuda::resample_nearest(
                &DeviceImage::upload(src).unwrap(),
                &Geometry::of(grid),
                0.0,
            )
            .expect("device nearest resample")
            .to_host()
            .unwrap();

            assert_bit_identical(&host, &device, &format!("nearest {payload} onto {name}"));
        }
    }
}

/// Nearest and linear are different operations, and the device keeps them apart:
/// they agree only where every sample lands exactly on an input voxel.
#[test]
fn the_device_nearest_and_linear_resamples_are_different_operations() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(48);
    let src = DeviceImage::upload(&img).unwrap();

    // On the image's own grid every sample is a voxel center: the two agree.
    let same = Geometry::of(&img);
    let nn = sitk_cuda::resample_nearest(&src, &same, 0.0)
        .unwrap()
        .to_host()
        .unwrap();
    let lin = sitk_cuda::resample_linear(&src, &same, 0.0)
        .unwrap()
        .to_host()
        .unwrap();
    assert_bit_identical(&nn, &lin, "nearest vs linear on the identity grid");

    // Half a voxel off, the linear resample interpolates and the nearest one picks:
    // they must disagree almost everywhere.
    let mut shifted = img.clone();
    let o: Vec<f64> = img
        .origin()
        .iter()
        .zip(img.spacing().iter())
        .map(|(&o, &sp)| o + 0.5 * sp)
        .collect();
    shifted.set_origin(&o).unwrap();
    let g = Geometry::of(&shifted);
    let nn = sitk_cuda::resample_nearest(&src, &g, 0.0)
        .unwrap()
        .to_host()
        .unwrap();
    let lin = sitk_cuda::resample_linear(&src, &g, 0.0)
        .unwrap()
        .to_host()
        .unwrap();
    let (a, b) = (
        nn.scalar_slice::<f32>().unwrap(),
        lin.scalar_slice::<f32>().unwrap(),
    );
    let differ = a
        .iter()
        .zip(b.iter())
        .filter(|&(&x, &y)| x.to_bits() != y.to_bits())
        .count();
    println!(
        "half-voxel grid: nearest and linear differ at {differ}/{} voxels",
        a.len()
    );
    assert!(
        differ * 2 > a.len(),
        "nearest and linear agreed at {}/{} voxels on a half-voxel grid; one of them \
         is not doing what its name says",
        a.len() - differ,
        a.len()
    );
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

/// **D2's pin, and the load-bearing one of the whole fixed-initial-transform round.**
///
/// The device resample now carries a point map, and the claim is not "close": it is
/// that `resample_*_through` is **bit-identical** to `ResampleImageFilter::execute(input,
/// transform)` for every transform `Transform::matrix_offset_map` accepts. Everything
/// downstream rests on it — the in-buffer predicate is 0/1, so one ulp in the mapped
/// point flips a shell of border voxels and moves the valid-point count the device path
/// pins as *exactly* equal to the host's.
///
/// If this ever fails, the honest answer is to refuse the offending variant in
/// `matrix_offset_map`, not to relax the assertion to a tolerance.
///
/// Both interpolators, and both payloads: a textured volume (where a wrong index shows
/// up in the value) and a binary mask (where it does not, which is the whole point).
#[test]
fn the_device_resample_through_a_transform_is_bit_identical_to_the_host_filter() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(48);
    let mask = binary_mask(&img);

    // Every accepted transform class that can reach a 3-D resample, each with a
    // non-trivial centre so `offset != translation` and a fold would be visible.
    let c = [24.0, 24.0, 24.0];
    let t = [3.5, -2.25, 7.125];
    let transforms: Vec<(&str, Transform)> = vec![
        (
            "Translation",
            Transform::Translation(TranslationTransform::new(vec![2.5, -1.25, 3.75])),
        ),
        // Half a voxel on every axis: every mapped continuous index is an exact
        // half-integer, so `floor(c + 0.5)` sits on a **tie** at every single sample
        // and `floor(c)` sits on an integer boundary. This is what makes the pin
        // sensitive to the *order* of the additions inside the point map, not merely
        // to its result: a 1-ulp difference in `mapped` — the kind a reassociated
        // accumulator produces — flips the nearest index and the linear weights here,
        // where on a generic grid it would be absorbed. The metric kernel's 34%
        // derivative bug was exactly a reassociated accumulator (offset seeded into
        // the sum instead of added last), and a pin that a reassociation can pass is
        // not pinning the arithmetic.
        (
            "Translation by exactly half a voxel (every index is a tie)",
            Transform::Translation(TranslationTransform::new(vec![0.35, 0.55, 0.65])),
        ),
        (
            "Affine",
            Transform::Affine(AffineTransform::new(
                3,
                vec![0.97, -0.21, 0.11, 0.19, 0.95, -0.24, -0.14, 0.22, 0.96],
                t.to_vec(),
                c.to_vec(),
            )),
        ),
        (
            "Euler3D",
            Transform::Euler3D(Euler3DTransform::new(0.31, -0.17, 0.44, t, c)),
        ),
        (
            "VersorRigid3D",
            Transform::VersorRigid3D(VersorRigid3DTransform::new(0.11, -0.23, 0.07, t, c)),
        ),
        (
            "Versor",
            Transform::Versor(VersorTransform::new(0.11, -0.23, 0.07, c)),
        ),
        (
            "Similarity3D",
            Transform::Similarity3D(Similarity3DTransform::new(1.37, 0.11, -0.23, 0.07, t, c)),
        ),
        (
            "ScaleVersor3D",
            Transform::ScaleVersor3D(ScaleVersor3DTransform::new(
                [1.1, 0.9, 1.3],
                0.11,
                -0.23,
                0.07,
                t,
                c,
            )),
        ),
        (
            "ScaleSkewVersor3D",
            Transform::ScaleSkewVersor3D(ScaleSkewVersor3DTransform::new(
                [1.1, 0.9, 1.3],
                [0.02, -0.03, 0.05, 0.01, -0.04, 0.06],
                0.11,
                -0.23,
                0.07,
                t,
                c,
            )),
        ),
        (
            "ComposeScaleSkewVersor3D",
            Transform::ComposeScaleSkewVersor3D(ComposeScaleSkewVersor3DTransform::new(
                [1.1, 0.9, 1.3],
                [0.02, -0.03, 0.05],
                0.11,
                -0.23,
                0.07,
                t,
                c,
            )),
        ),
    ];

    // The grids the level actually resamples onto, including the half-voxel-offset one
    // where every continuous index is a tie.
    let grids: Vec<(String, Image)> = vec![
        (
            "shrink [1,1,1]".into(),
            sitk_filters::shrink(&img, &[1, 1, 1]).unwrap(),
        ),
        (
            "shrink [2,2,2]".into(),
            sitk_filters::shrink(&img, &[2, 2, 2]).unwrap(),
        ),
        (
            "shrink [3,5,7]".into(),
            sitk_filters::shrink(&img, &[3, 5, 7]).unwrap(),
        ),
    ];

    for (tname, transform) in &transforms {
        let map = transform
            .matrix_offset_map()
            .unwrap_or_else(|| panic!("{tname}: matrix_offset_map refused an accepted variant"));

        for (gname, grid) in &grids {
            for (payload, src, interp) in [
                ("textured/linear", &img, Interpolator::Linear),
                ("mask/nearest", &mask, Interpolator::NearestNeighbor),
            ] {
                let mut resampler = ResampleImageFilter::new();
                resampler
                    .set_reference_image(grid)
                    .set_interpolator(interp)
                    .set_default_pixel_value(0.0);
                let host = resampler.execute(src, transform).expect("host resample");

                let d = DeviceImage::upload(src).unwrap();
                let g = Geometry::of(grid);
                let device = match interp {
                    Interpolator::Linear => {
                        sitk_cuda::resample_linear_through(&d, &g, 0.0, &map.matrix, &map.offset)
                    }
                    _ => sitk_cuda::resample_nearest_through(&d, &g, 0.0, &map.matrix, &map.offset),
                }
                .expect("device resample through a transform")
                .to_host()
                .unwrap();

                assert_bit_identical(&host, &device, &format!("{tname} {payload} onto {gname}"));
            }
        }
    }
}

/// Anti-vacuity for the pin above: a transform that maps every point to itself would
/// make it pass while proving nothing. Each transform must actually *move* the output
/// — and it must move it enough to push part of the input off the grid, which is where
/// the in-buffer predicate lives and where a wrong map does its damage.
#[test]
fn the_transforms_in_that_pin_actually_move_the_resample() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(48);
    let grid = sitk_filters::shrink(&img, &[2, 2, 2]).unwrap();
    let d = DeviceImage::upload(&img).unwrap();
    let g = Geometry::of(&grid);

    let identity = sitk_cuda::resample_linear(&d, &g, 0.0)
        .unwrap()
        .to_host()
        .unwrap();
    let identity = identity.scalar_slice::<f32>().unwrap().to_vec();

    let euler = Transform::Euler3D(Euler3DTransform::new(
        0.31,
        -0.17,
        0.44,
        [3.5, -2.25, 7.125],
        [24.0, 24.0, 24.0],
    ));
    let map = euler.matrix_offset_map().unwrap();
    let mapped = sitk_cuda::resample_linear_through(&d, &g, 0.0, &map.matrix, &map.offset)
        .unwrap()
        .to_host()
        .unwrap();
    let mapped = mapped.scalar_slice::<f32>().unwrap();

    let moved = identity
        .iter()
        .zip(mapped.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        moved > identity.len() / 10,
        "the transform moved only {moved} of {} voxels; the bit-identity pin above would \
         be passing on a near-identity map and proving nothing",
        identity.len()
    );

    // ...and it pushes part of the input off the grid: the map must produce voxels that
    // fall outside the input buffer and take the default value, which is the shell the
    // in-buffer predicate is about.
    let outside = mapped.iter().filter(|&&v| v == 0.0).count();
    assert!(
        outside > 0,
        "the transform kept every output voxel inside the input buffer; nothing here \
         exercises the border where a wrong point map flips the predicate"
    );
    println!(
        "moved {moved}/{} voxels, {outside} outside the buffer",
        identity.len()
    );
}

/// The identity path did not move. `resample_linear`/`resample_nearest` now pack
/// `M = I, b = 0` and run the same multiply the transform path runs, instead of
/// skipping it — so this asserts the change was arithmetically inert, which is what
/// `mat_vec(I, p) + 0 == p` bitwise claims. (The existing bit-identity pins above
/// already cover it against the *host*; this covers it against the *device's own*
/// through-form with an identity map, which is the substitution D3 will make.)
#[test]
fn the_identity_map_is_the_identity_resample_on_the_device() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(32);
    let grid = sitk_filters::shrink(&img, &[2, 2, 2]).unwrap();
    let d = DeviceImage::upload(&img).unwrap();
    let g = Geometry::of(&grid);

    let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let zero = [0.0, 0.0, 0.0];

    for (name, plain, through) in [
        (
            "linear",
            sitk_cuda::resample_linear(&d, &g, 0.0).unwrap(),
            sitk_cuda::resample_linear_through(&d, &g, 0.0, &eye, &zero).unwrap(),
        ),
        (
            "nearest",
            sitk_cuda::resample_nearest(&d, &g, 0.0).unwrap(),
            sitk_cuda::resample_nearest_through(&d, &g, 0.0, &eye, &zero).unwrap(),
        ),
    ] {
        assert_bit_identical(
            &plain.to_host().unwrap(),
            &through.to_host().unwrap(),
            &format!("{name}: identity map vs no map"),
        );
    }
}

/// A map of the wrong shape is refused, not silently padded.
#[test]
fn a_malformed_point_map_is_refused() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(16);
    let d = DeviceImage::upload(&img).unwrap();
    let g = Geometry::of(&img);

    match sitk_cuda::resample_linear_through(&d, &g, 0.0, &[1.0, 0.0, 0.0, 1.0], &[0.0; 3]) {
        Err(CudaError::UnsupportedGeometry(why)) => println!("refused: {why}"),
        other => panic!("a 2x2 map was accepted: {:?}", other.map(|_| ())),
    }
    match sitk_cuda::resample_nearest_through(
        &d,
        &g,
        0.0,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[0.0; 2],
    ) {
        Err(CudaError::UnsupportedGeometry(why)) => println!("refused: {why}"),
        other => panic!("a 2-vector offset was accepted: {:?}", other.map(|_| ())),
    }
}

/// **The point map's addition *order*, pinned — not merely its result.**
///
/// The pin above compares output bits, and output bits cannot see a 1-ulp difference
/// in the mapped point *unless a sample sits within 1 ulp of a boundary*. On a generic
/// grid they never do, so a reassociated accumulator — the metric kernel's historical
/// 34% derivative bug, where the offset was seeded into the sum instead of added last —
/// passes it. Measured: seeding the offset into the accumulator leaves every voxel of
/// that test bit-identical.
///
/// So this builds the grid where it *is* visible. For a rotation `M` and offset `b`,
/// the output grid is rotated by `Mᵀ` and shifted half a voxel, so
/// `mapped = M·(Mᵀ(o + h − b) + Mᵀ·i) + b` is `o + h + i` — i.e. **every continuous
/// index is a half-integer**, up to the last bits. `floor(c + 0.5)` is then a tie at
/// every sample, and any reassociation of `acc + b` flips a large fraction of them.
///
/// This is the test that says the device does the host's arithmetic in the host's
/// order, and not merely something numerically close to it.
#[test]
fn the_point_map_addition_order_is_pinned_not_just_its_result() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    // Isotropic, so a rotation of the grid stays a rotation of the index lattice.
    let n = 32usize;
    let img = {
        let mut v = volume(n);
        v.set_spacing(&[1.0, 1.0, 1.0]).unwrap();
        v
    };
    let o = img.origin().to_vec();

    let euler = Transform::Euler3D(Euler3DTransform::new(
        0.0,
        0.0,
        0.37,
        [2.5, -1.25, 0.0],
        [0.0, 0.0, 0.0],
    ));
    let map = euler.matrix_offset_map().unwrap();
    let (m, b) = (&map.matrix, &map.offset);

    // Mᵀ, and the origin that puts every mapped index on a half-integer.
    let mt: Vec<f64> = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .map(|(r, c)| m[c * 3 + r])
        .collect();
    let h = [0.5, 0.5, 0.0];
    let shifted: Vec<f64> = (0..3).map(|d| o[d] + h[d] - b[d]).collect();
    let out_origin = sitk_core::matrix::mat_vec(&mt, &shifted, 3);

    let mut grid = img.clone();
    grid.set_direction(&mt).unwrap();
    grid.set_origin(&out_origin).unwrap();

    let mask = binary_mask(&img);
    for (payload, src, interp) in [
        ("textured/linear", &img, Interpolator::Linear),
        ("mask/nearest", &mask, Interpolator::NearestNeighbor),
    ] {
        let mut resampler = ResampleImageFilter::new();
        resampler
            .set_reference_image(&grid)
            .set_interpolator(interp)
            .set_default_pixel_value(0.0);
        let host = resampler.execute(src, &euler).expect("host resample");

        let d = DeviceImage::upload(src).unwrap();
        let g = Geometry::of(&grid);
        let device = match interp {
            Interpolator::Linear => sitk_cuda::resample_linear_through(&d, &g, 0.0, m, b),
            _ => sitk_cuda::resample_nearest_through(&d, &g, 0.0, m, b),
        }
        .expect("device resample through a transform")
        .to_host()
        .unwrap();

        assert_bit_identical(&host, &device, &format!("tie grid, {payload}"));
    }

    // Anti-vacuity: the construction must actually produce ties, or this test is the
    // generic grid again under a longer comment. Every sample's continuous index must
    // be within a hair of a half-integer.
    let inv = sitk_core::matrix::invert(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3).unwrap();
    let mut ties = 0usize;
    for i in [[0usize, 0, 0], [5, 7, 3], [17, 2, 29], [31, 31, 31]] {
        let idx: Vec<f64> = i.iter().map(|&x| x as f64).collect();
        let phys = sitk_core::matrix::mat_vec(&mt, &idx, 3);
        let phys: Vec<f64> = (0..3).map(|d| out_origin[d] + phys[d]).collect();
        let mapped = sitk_core::matrix::mat_vec(m, &phys, 3);
        let mapped: Vec<f64> = (0..3).map(|d| mapped[d] + b[d]).collect();
        let diff: Vec<f64> = (0..3).map(|d| mapped[d] - o[d]).collect();
        let c = sitk_core::matrix::mat_vec(&inv, &diff, 3);
        for (d, &cd) in c.iter().enumerate().take(2) {
            let frac = (cd - cd.floor() - 0.5).abs();
            assert!(
                frac < 1e-9,
                "index {i:?} axis {d}: continuous index {cd} is not a tie (frac off by \
                 {frac:e}); the construction is broken and this test pins nothing about ordering"
            );
            ties += 1;
        }
    }
    println!("{ties} sampled indices are exact half-integer ties");
}
