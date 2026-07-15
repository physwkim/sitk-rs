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

use sitk::core::Image;
use sitk::cuda::{CudaError, DeviceImage, Geometry, PointStage, backend};
use sitk::registration::metric::{FixedSamples, MovingImage};
use sitk::registration::{CpuBackend, DeviceMeanSquaresMetric, MeanSquaresMetric};
use sitk::transform::{
    AffineTransform, ComposeScaleSkewVersor3DTransform, CompositeTransform, Euler3DTransform,
    Interpolator, ResampleImageFilter, ScaleSkewVersor3DTransform, ScaleVersor3DTransform,
    Similarity3DTransform, Transform, TransformBase, TranslationTransform, VersorRigid3DTransform,
    VersorTransform,
};

fn no_device() -> bool {
    matches!(backend(), Err(CudaError::NoDevice(_)))
}

/// The transform's own point-map stages, in the device's fixed-size form — the *only*
/// thing `resample_*_through` may be handed.
///
/// One conversion, no arithmetic: whatever `point_map_stages` reports is what the kernel
/// replays. A test that folded a composite here, or probed an affine out of the
/// transform, would be pinning the device against a map the host does not evaluate.
fn stages_of(t: &Transform) -> Vec<PointStage> {
    let maps = t
        .point_map_stages()
        .unwrap_or_else(|| panic!("{:?} reports no bitwise point map", t.kind()));
    maps.iter()
        .map(|m| {
            let mut matrix = [0.0f64; 9];
            let mut offset = [0.0f64; 3];
            matrix.copy_from_slice(&m.matrix);
            offset.copy_from_slice(&m.offset);
            PointStage { matrix, offset }
        })
        .collect()
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
        let host = sitk::filters::recursive_gaussian(&img, &sigma).expect("host gaussian");
        let device = sitk::cuda::recursive_gaussian(&DeviceImage::upload(&img).unwrap(), &sigma)
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
        let host = sitk::filters::shrink(&img, &factors).expect("host shrink");
        let device = sitk::cuda::shrink(&DeviceImage::upload(&img).unwrap(), &factors)
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
        let grid = sitk::filters::shrink(&img, &factors).expect("host shrink");

        let mut resampler = ResampleImageFilter::new();
        resampler
            .set_reference_image(&grid)
            .set_interpolator(Interpolator::Linear)
            .set_default_pixel_value(0.0);
        let host = resampler
            .execute(&img, &AffineTransform::identity(3))
            .expect("host resample");

        let device = sitk::cuda::resample_linear(
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
            sitk::filters::shrink(&img, &[1, 1, 1]).unwrap(),
        ),
        (
            "shrink [2,2,2]",
            sitk::filters::shrink(&img, &[2, 2, 2]).unwrap(),
        ),
        (
            "shrink [3,5,7]",
            sitk::filters::shrink(&img, &[3, 5, 7]).unwrap(),
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

            let device = sitk::cuda::resample_nearest(
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
    let nn = sitk::cuda::resample_nearest(&src, &same, 0.0)
        .unwrap()
        .to_host()
        .unwrap();
    let lin = sitk::cuda::resample_linear(&src, &same, 0.0)
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
    let nn = sitk::cuda::resample_nearest(&src, &g, 0.0)
        .unwrap()
        .to_host()
        .unwrap();
    let lin = sitk::cuda::resample_linear(&src, &g, 0.0)
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
        let shrunk = sitk::cuda::shrink(&src, &factors).unwrap();
        let grid = shrunk.geometry().clone();
        let resampled = sitk::cuda::resample_linear(&src, &grid, 0.0).unwrap();

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
            sitk::filters::recursive_gaussian(&fixed, &sigma).unwrap()
        };
        let host_moving_level = if level_sigma == 0.0 {
            moving.clone()
        } else {
            sitk::filters::recursive_gaussian(&moving, &sigma).unwrap()
        };
        let host_fixed_level = if factor == 1 {
            host_fixed_smoothed.clone()
        } else {
            let grid = sitk::filters::shrink(&host_fixed_smoothed, &factors).unwrap();
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
            Some(sitk::cuda::recursive_gaussian(&d_fixed, &sigma).unwrap())
        };
        let dev_moving_level = if level_sigma == 0.0 {
            None
        } else {
            Some(sitk::cuda::recursive_gaussian(&d_moving, &sigma).unwrap())
        };
        let sf = dev_fixed_smoothed.as_ref().unwrap_or(&d_fixed);
        let dev_fixed_level = if factor == 1 {
            None
        } else {
            let coarse = sitk::cuda::shrink(sf, &factors).unwrap();
            Some(sitk::cuda::resample_linear(sf, coarse.geometry(), 0.0).unwrap())
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
    match sitk::cuda::recursive_gaussian(&d, &[1.0, 1.0, 1.0]) {
        Err(CudaError::UnsupportedGeometry(why)) => println!("refused: {why}"),
        other => panic!("a 3-voxel axis was smoothed: {:?}", other.map(|_| ())),
    }
    assert!(sitk::filters::recursive_gaussian(&tiny, &[1.0, 1.0, 1.0]).is_err());

    // ... but a sigma of zero on that axis smooths nothing, and is fine — on both.
    assert!(sitk::cuda::recursive_gaussian(&d, &[0.0, 1.0, 1.0]).is_ok());
    assert!(sitk::filters::recursive_gaussian(&tiny, &[0.0, 1.0, 1.0]).is_ok());

    match sitk::cuda::shrink(&d, &[2, 0, 2]) {
        Err(CudaError::UnsupportedGeometry(why)) => println!("refused: {why}"),
        other => panic!("a zero shrink factor was accepted: {:?}", other.map(|_| ())),
    }

    let two_d = Image::from_vec(&[8usize, 8], vec![1.0f32; 64]).unwrap();
    let d2 = DeviceImage::upload(&two_d).unwrap();
    match sitk::cuda::shrink(&d2, &[2, 2]) {
        Err(CudaError::UnsupportedGeometry(why)) => println!("refused: {why}"),
        other => panic!("a 2-D image was shrunk: {:?}", other.map(|_| ())),
    }
}

/// **D2's pin, and the load-bearing one of the whole fixed-initial-transform round.**
///
/// The device resample carries the transform's own point-map stages, and the claim is
/// not "close": it is that `resample_*_through` is **bit-identical** to
/// `ResampleImageFilter::execute(input, transform)` for every transform
/// `point_map_stages` accepts — including a **multi-stage composite**, which is replayed
/// stage by stage rather than folded. Everything downstream rests on it: the in-buffer
/// predicate is 0/1, so one ulp in the mapped point flips a shell of border voxels and
/// moves the valid-point count the device path pins as *exactly* equal to the host's.
///
/// If this ever fails, the honest answer is to refuse the offending variant in
/// `point_map_stages`, not to relax the assertion to a tolerance.
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
        // The one that could not reach a device resample before this: a composite of
        // THREE maps. `ResampleImageFilter` evaluates its `transform_point`, which
        // applies the members one after another, each rounding on its own; the device
        // replays the same three stages in the same order. A folded `M₃·M₂·M₁` is the
        // same map in exact arithmetic and rounds ONCE — and the fold is exactly what a
        // "just multiply them together" shortcut would do here.
        ("Composite of three maps", {
            let mut composite = CompositeTransform::new(3);
            composite
                .add_transform(Transform::Euler3D(Euler3DTransform::new(
                    0.31, -0.17, 0.44, t, c,
                )))
                .unwrap();
            composite
                .add_transform(Transform::Translation(TranslationTransform::new(vec![
                    1.5, -0.75, 2.25,
                ])))
                .unwrap();
            composite
                .add_transform(Transform::Affine(AffineTransform::new(
                    3,
                    vec![0.98, -0.13, 0.06, 0.12, 0.97, -0.15, -0.07, 0.14, 0.99],
                    vec![-1.25, 0.5, -2.0],
                    c.to_vec(),
                )))
                .unwrap();
            Transform::Composite(composite)
        }),
    ];

    // The grids the level actually resamples onto, including the half-voxel-offset one
    // where every continuous index is a tie.
    let grids: Vec<(String, Image)> = vec![
        (
            "shrink [1,1,1]".into(),
            sitk::filters::shrink(&img, &[1, 1, 1]).unwrap(),
        ),
        (
            "shrink [2,2,2]".into(),
            sitk::filters::shrink(&img, &[2, 2, 2]).unwrap(),
        ),
        (
            "shrink [3,5,7]".into(),
            sitk::filters::shrink(&img, &[3, 5, 7]).unwrap(),
        ),
    ];

    for (tname, transform) in &transforms {
        let stages = stages_of(transform);

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
                        sitk::cuda::resample_linear_through(&d, &g, 0.0, &stages)
                    }
                    _ => sitk::cuda::resample_nearest_through(&d, &g, 0.0, &stages),
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
    let grid = sitk::filters::shrink(&img, &[2, 2, 2]).unwrap();
    let d = DeviceImage::upload(&img).unwrap();
    let g = Geometry::of(&grid);

    let identity = sitk::cuda::resample_linear(&d, &g, 0.0)
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
    let mapped = sitk::cuda::resample_linear_through(&d, &g, 0.0, &stages_of(&euler))
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

/// The identity path did not move. `resample_linear`/`resample_nearest` pack a single
/// `M = I, b = 0` stage and run the same multiply the transform path runs, instead of
/// skipping it — so this asserts that packing is arithmetically inert, which is what
/// `mat_vec(I, p) + 0 == p` bitwise claims.
#[test]
fn the_identity_map_is_the_identity_resample_on_the_device() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(32);
    let grid = sitk::filters::shrink(&img, &[2, 2, 2]).unwrap();
    let d = DeviceImage::upload(&img).unwrap();
    let g = Geometry::of(&grid);

    let eye = [PointStage {
        matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        offset: [0.0; 3],
    }];

    for (name, plain, through) in [
        (
            "linear",
            sitk::cuda::resample_linear(&d, &g, 0.0).unwrap(),
            sitk::cuda::resample_linear_through(&d, &g, 0.0, &eye).unwrap(),
        ),
        (
            "nearest",
            sitk::cuda::resample_nearest(&d, &g, 0.0).unwrap(),
            sitk::cuda::resample_nearest_through(&d, &g, 0.0, &eye).unwrap(),
        ),
    ] {
        assert_bit_identical(
            &plain.to_host().unwrap(),
            &through.to_host().unwrap(),
            &format!("{name}: identity map vs no map"),
        );
    }
}

/// A stage list the device cannot replay is refused **by name**, not silently padded,
/// truncated, or treated as the identity.
///
/// The empty list is the one that matters. A zero-stage replay *is* the identity map —
/// it would resample without the transform, produce a perfectly plausible image, and be
/// wrong. The identity is spelled `resample_linear` / `resample_nearest`; an empty stage
/// list is a caller bug and says so.
#[test]
fn a_stage_list_the_device_cannot_replay_is_refused() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let img = volume(16);
    let d = DeviceImage::upload(&img).unwrap();
    let g = Geometry::of(&img);

    let one = PointStage {
        matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        offset: [1.0, 2.0, 3.0],
    };
    let too_many = vec![one; sitk::cuda::MAX_STAGES + 1];

    for (what, got) in [
        (
            "empty",
            sitk::cuda::resample_linear_through(&d, &g, 0.0, &[]),
        ),
        (
            "over the cap",
            sitk::cuda::resample_nearest_through(&d, &g, 0.0, &too_many),
        ),
    ] {
        match got {
            Err(CudaError::PointMapStageCount { stages, max }) => {
                println!("{what} stage list refused: {stages} stages, the device replays 1..={max}")
            }
            other => panic!("a {what} stage list was accepted: {:?}", other.map(|_| ())),
        }
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
    let out_origin = sitk::core::matrix::mat_vec(&mt, &shifted, 3);

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
            Interpolator::Linear => {
                sitk::cuda::resample_linear_through(&d, &g, 0.0, &stages_of(&euler))
            }
            _ => sitk::cuda::resample_nearest_through(&d, &g, 0.0, &stages_of(&euler)),
        }
        .expect("device resample through a transform")
        .to_host()
        .unwrap();

        assert_bit_identical(&host, &device, &format!("tie grid, {payload}"));
    }

    // Anti-vacuity: the construction must actually produce ties, or this test is the
    // generic grid again under a longer comment. Every sample's continuous index must
    // be within a hair of a half-integer.
    let inv =
        sitk::core::matrix::invert(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3).unwrap();
    let mut ties = 0usize;
    for i in [[0usize, 0, 0], [5, 7, 3], [17, 2, 29], [31, 31, 31]] {
        let idx: Vec<f64> = i.iter().map(|&x| x as f64).collect();
        let phys = sitk::core::matrix::mat_vec(&mt, &idx, 3);
        let phys: Vec<f64> = (0..3).map(|d| out_origin[d] + phys[d]).collect();
        let mapped = sitk::core::matrix::mat_vec(m, &phys, 3);
        let mapped: Vec<f64> = (0..3).map(|d| mapped[d] + b[d]).collect();
        let diff: Vec<f64> = (0..3).map(|d| mapped[d] - o[d]).collect();
        let c = sitk::core::matrix::mat_vec(&inv, &diff, 3);
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

/// **The composite pin, at the tie — the load-bearing test of this round.**
///
/// A `CompositeTransform` fixed-initial transform was refused by the device path until
/// now: the resample took one matrix and one offset, and a composite is not one matrix.
/// The device replays its stages, so it is accepted — and the claim is bit-identity with
/// `ResampleImageFilter::execute(input, composite)`, not a band.
///
/// It is pinned **at the straddle**, not away from it. The grid is built from the
/// composite's folded map so that every continuous index is an exact half-integer:
/// `floor(c + 0.5)` is then a tie at every sample, and `resample_nearest3` turns a
/// last-bit difference into a *different voxel* — a whole intensity, not a limit.
///
/// The anti-vacuity check is the second half, and it is what makes the stage list
/// load-bearing rather than ceremony: resampling through the **folded** map — `M₂·M₁`,
/// `M₂·b₁ + b₂`, the same transform in exact arithmetic, one rounding instead of two —
/// is measured against the same host output, and it must *differ*. If the fold agreed,
/// there would be no reason to replay.
#[test]
fn a_composite_is_replayed_stage_by_stage_and_at_a_tie_the_fold_is_visible() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    // Isotropic with an identity direction, so a rotation of the grid stays a rotation
    // of the index lattice and the tie construction below is exact.
    let n = 32usize;
    let img = {
        let mut v = volume(n);
        v.set_spacing(&[1.0, 1.0, 1.0]).unwrap();
        v
    };
    let o = img.origin().to_vec();

    // Two rigid maps, so the fold is still a rotation and the tie grid is constructible.
    let composite = {
        let mut c = CompositeTransform::new(3);
        c.add_transform(Transform::Euler3D(Euler3DTransform::new(
            0.0,
            0.0,
            0.37,
            [2.5, -1.25, 0.0],
            [0.0, 0.0, 0.0],
        )))
        .unwrap();
        c.add_transform(Transform::Euler3D(Euler3DTransform::new(
            0.0,
            0.0,
            -0.19,
            [-1.75, 0.5, 0.0],
            [3.0, -2.0, 0.0],
        )))
        .unwrap();
        Transform::Composite(c)
    };
    let stages = stages_of(&composite);
    assert_eq!(
        stages.len(),
        2,
        "the composite must hand over two stages, or this test is the single-map pin again"
    );

    // The fold: q = M₂(M₁p + b₁) + b₂ = (M₂M₁)p + (M₂b₁ + b₂). Algebraically the composite;
    // arithmetically one rounding where the composite does two.
    let (folded_m, folded_b) = {
        let mut m = sitk::core::matrix::identity(3);
        let mut b = vec![0.0; 3];
        for s in &stages {
            m = sitk::core::matrix::matmul(&s.matrix, &m, 3);
            let mb = sitk::core::matrix::mat_vec(&s.matrix, &b, 3);
            b = (0..3).map(|d| mb[d] + s.offset[d]).collect();
        }
        (m, b)
    };
    let folded = [PointStage {
        matrix: [
            folded_m[0],
            folded_m[1],
            folded_m[2],
            folded_m[3],
            folded_m[4],
            folded_m[5],
            folded_m[6],
            folded_m[7],
            folded_m[8],
        ],
        offset: [folded_b[0], folded_b[1], folded_b[2]],
    }];

    // The grid on which every mapped continuous index is a half-integer: direction Mᵀ,
    // origin Mᵀ(o + h − b), so `mapped = M·(Mᵀ(o + h − b) + Mᵀ·i) + b = o + h + i`.
    let mt: Vec<f64> = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .map(|(r, c)| folded_m[c * 3 + r])
        .collect();
    let h = [0.5, 0.5, 0.0];
    let shifted: Vec<f64> = (0..3).map(|d| o[d] + h[d] - folded_b[d]).collect();
    let out_origin = sitk::core::matrix::mat_vec(&mt, &shifted, 3);

    let mut grid = img.clone();
    grid.set_direction(&mt).unwrap();
    grid.set_origin(&out_origin).unwrap();
    let g = Geometry::of(&grid);

    // Anti-vacuity for the geometry: the samples must actually sit on ties.
    {
        let mut ties = 0usize;
        for i in [[0usize, 0, 0], [5, 7, 3], [17, 2, 29], [31, 31, 31]] {
            let idx: Vec<f64> = i.iter().map(|&x| x as f64).collect();
            let phys = sitk::core::matrix::mat_vec(&mt, &idx, 3);
            let phys: Vec<f64> = (0..3).map(|d| out_origin[d] + phys[d]).collect();
            let mapped = composite.transform_point(&phys);
            for d in 0..2 {
                let c = mapped[d] - o[d];
                let frac = (c - c.floor() - 0.5).abs();
                assert!(
                    frac < 1e-9,
                    "index {i:?} axis {d}: continuous index {c} is not a tie (off by {frac:e}); \
                     the construction is broken and this pins nothing about the replay"
                );
                ties += 1;
            }
        }
        println!("{ties} sampled indices are exact half-integer ties");
    }

    let mask = binary_mask(&img);
    let mut fold_diffs = 0usize;
    let mut fold_total = 0usize;
    for (payload, src, interp) in [
        ("textured/linear", &img, Interpolator::Linear),
        ("mask/nearest", &mask, Interpolator::NearestNeighbor),
    ] {
        let mut resampler = ResampleImageFilter::new();
        resampler
            .set_reference_image(&grid)
            .set_interpolator(interp)
            .set_default_pixel_value(0.0);
        let host = resampler.execute(src, &composite).expect("host resample");

        let d = DeviceImage::upload(src).unwrap();
        let through = |st: &[PointStage]| {
            match interp {
                Interpolator::Linear => sitk::cuda::resample_linear_through(&d, &g, 0.0, st),
                _ => sitk::cuda::resample_nearest_through(&d, &g, 0.0, st),
            }
            .expect("device resample through a composite")
            .to_host()
            .unwrap()
        };

        // The claim: replaying the composite's own stages is the host, bit for bit.
        assert_bit_identical(
            &host,
            &through(&stages),
            &format!("composite tie grid, {payload}"),
        );

        // The counter-claim: folding them is not.
        let folded_out = through(&folded);
        let a = host.scalar_slice::<f32>().unwrap();
        let b = folded_out.scalar_slice::<f32>().unwrap();
        let differ = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        println!(
            "composite tie grid, {payload}: the fold differs from the host at {differ}/{} voxels",
            a.len()
        );
        fold_diffs += differ;
        fold_total += a.len();
    }
    assert!(
        fold_diffs > 0,
        "folding the composite's {} stages into one matrix reproduced the host at every one \
         of {fold_total} voxels, on a grid where every index is a tie. Then the stage replay \
         is buying nothing here and this test is not pinning what it claims to pin.",
        stages.len()
    );
}

/// **The physical→index matrix is the inverse of the *composed* `Direction·diag(spacing)`,
/// not the direction-only inverse — pinned on an oblique geometry at a tie.**
///
/// The two inverses are algebraically identical — `inverse(D·diag(s)) =
/// diag(1/s)·inverse(D)` — so for a diagonal direction they are bit-identical and no
/// identity/axis-aligned test can tell them apart. For an **oblique** direction the two
/// factorings round differently in the last bits of the matrix, and a physical point that
/// lands on a nearest-neighbour tie is resolved to different voxels by the two. The host
/// filter now inverts the composed matrix (`sitk::core::coord::physical_to_index_matrix`);
/// the device `affines()` must too, or the coarse mask carried to a level under an oblique
/// input geometry is a different image.
///
/// The grid is the input's own geometry shifted half a voxel in index space, so every
/// output voxel's continuous index into the input is an exact half-integer `i + 0.5` — a
/// tie at `floor(c + 0.5)`. The test asserts host↔device bit-identity, then proves it is
/// load-bearing two ways: the composed inverse differs in bits from the direction-only
/// inverse for this geometry, and resolving the tie with the direction-only inverse flips
/// at least one voxel away from the composed answer (what the pre-fix device kernel did).
#[test]
fn the_physical_to_index_matrix_inverts_the_composed_matrix_not_direction_only() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    use sitk::core::coord;
    use sitk::core::matrix::{invert, mat_vec, matmul};

    // A dense, fully oblique orthonormal direction: Rz(0.6)·Ry(0.5)·Rx(0.4). Every entry
    // is non-zero, so `inverse(D·diag(s))` and `diag(1/s)·inverse(D)` genuinely differ in
    // the last bits (a diagonal or single-axis rotation would leave them bit-identical).
    let (rx, ry, rz) = (0.4f64, 0.5f64, 0.6f64);
    let mx = [
        1.0,
        0.0,
        0.0,
        0.0,
        rx.cos(),
        -rx.sin(),
        0.0,
        rx.sin(),
        rx.cos(),
    ];
    let my = [
        ry.cos(),
        0.0,
        ry.sin(),
        0.0,
        1.0,
        0.0,
        -ry.sin(),
        0.0,
        ry.cos(),
    ];
    let mz = [
        rz.cos(),
        -rz.sin(),
        0.0,
        rz.sin(),
        rz.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    let dir = matmul(&mz, &matmul(&my, &mx, 3), 3);
    let spacing = [0.7f64, 1.1, 1.3];
    let in_origin = [-12.0f64, 5.5, 3.25];

    let n = 16usize;
    let mut src = volume(n);
    src.set_spacing(&spacing).unwrap();
    src.set_origin(&in_origin).unwrap();
    src.set_direction(&dir).unwrap();
    let mask = binary_mask(&src); // inherits src's oblique geometry

    // The output grid: the same oblique geometry, its origin shifted so output voxel `i`
    // maps to input continuous index `i + 0.5` on every axis. `out_origin = in_origin +
    // (D·diag(s))·[½,½,½]` (origin-first), so `phys(i) − in_origin = i2p·(i + ½)` and the
    // composed inverse sends it back to `i + ½` up to the last bits.
    let i2p = coord::index_to_physical_matrix(&dir, &spacing, 3);
    let out_origin = coord::index_to_physical_point_f64(&i2p, &in_origin, &[0.5, 0.5, 0.5], 3);
    let mut grid = src.clone();
    grid.set_origin(&out_origin).unwrap();
    let g = Geometry::of(&grid);

    let identity = AffineTransform::identity(3);
    for (payload, image, interp) in [
        ("textured/linear", &src, Interpolator::Linear),
        ("mask/nearest", &mask, Interpolator::NearestNeighbor),
    ] {
        let mut resampler = ResampleImageFilter::new();
        resampler
            .set_reference_image(&grid)
            .set_interpolator(interp)
            .set_default_pixel_value(0.0);
        let host = resampler.execute(image, &identity).expect("host resample");

        let d = DeviceImage::upload(image).unwrap();
        let device = match interp {
            Interpolator::Linear => sitk::cuda::resample_linear(&d, &g, 0.0),
            _ => sitk::cuda::resample_nearest(&d, &g, 0.0),
        }
        .expect("device resample")
        .to_host()
        .unwrap();

        assert_bit_identical(&host, &device, &format!("oblique tie grid, {payload}"));
    }

    // Non-vacuity A: the composed inverse differs in bits from the direction-only inverse
    // for this geometry. If it did not, the fix would be invisible and this test vacuous.
    let p2i_composed = coord::physical_to_index_matrix(&dir, &spacing, 3).unwrap();
    let dinv = invert(&dir, 3).unwrap();
    let mut p2i_dir_only = vec![0.0f64; 9];
    for r in 0..3 {
        for c in 0..3 {
            p2i_dir_only[r * 3 + c] = dinv[r * 3 + c] / spacing[r];
        }
    }
    let matrix_bits_differ = p2i_composed
        .iter()
        .zip(p2i_dir_only.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        matrix_bits_differ,
        "on this oblique geometry the composed inverse equals the direction-only inverse on \
         the bits, so nothing distinguishes the fix — pick a more oblique direction"
    );

    // Non-vacuity B (the load-bearing flip): resolving each output voxel's tie with the
    // direction-only inverse the pre-fix kernel used lands on a different voxel than the
    // composed inverse for at least one sample. This is the whole point of site 2.
    let mut flips = 0usize;
    for flat in 0..(n * n * n) {
        let idx_f = [
            (flat % n) as f64,
            ((flat / n) % n) as f64,
            (flat / (n * n)) as f64,
        ];
        let phys = coord::index_to_physical_point_f64(&i2p, &out_origin, &idx_f, 3);
        let diff: Vec<f64> = (0..3).map(|k| phys[k] - in_origin[k]).collect();
        let c_composed = mat_vec(&p2i_composed, &diff, 3);
        let c_dir_only = mat_vec(&p2i_dir_only, &diff, 3);
        for d in 0..3 {
            if coord::round_half_integer_up(c_composed[d])
                != coord::round_half_integer_up(c_dir_only[d])
            {
                flips += 1;
            }
        }
    }
    assert!(
        flips > 0,
        "the direction-only inverse resolved every tie to the same voxel as the composed \
         inverse ({} voxels), so this grid does not exercise site 2 — the latent fix would \
         be unverified",
        n * n * n
    );
    println!("oblique tie grid: direction-only inverse would flip {flips} nearest-voxel ties");
}
