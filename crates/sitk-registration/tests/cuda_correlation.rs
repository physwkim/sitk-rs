//! [`DeviceCorrelationMetric`] against the host [`CorrelationMetric`] — the host is
//! the reference, not ITK.
//!
//! The bands here are **stated, not discovered**. The host sums N per-sample terms
//! left to right in one thread; the device folds a fixed shared-memory tree in
//! block-index order. No parallel reduction reproduces a serial sum, so the two
//! differ by reduction rounding (~√N·ε on each moment) and the pins are tolerance
//! pins. What is *not* banded: `valid_points` (an integer — the two must walk the
//! same sample set exactly) and run-to-run determinism (bit equality).
//!
//! The **value** band is absolute. NCC's value is `−sfm²/(sff·smm)` ∈ [−1, 0]: it is
//! dimensionless and O(1) by construction, and it passes through zero at any pose
//! where the samples decorrelate. A relative band there divides by a number that is
//! *supposed* to be able to reach zero, so it would either blow up at an
//! uninteresting pose or have to be papered over with `max(|v|, 1)` — which is an
//! absolute band wearing a relative band's clothes. So: absolute, and said out loud.
//!
//! The **derivative** band is relative, because the derivative carries the
//! transform's units (an intensity per radian is not an intensity per mm) and its
//! components differ by orders of magnitude across a rigid parameter vector.
//!
//! Only compiled with the `cuda` feature.
#![cfg(feature = "cuda")]

mod support;

use sitk_core::Image;
use sitk_cuda::DeviceImage;
use sitk_registration::metric::{FixedSamples, MetricValue, MovingImage};
use sitk_registration::{CorrelationMetric, DeviceCorrelationMetric, DeviceMetricError};
use sitk_transform::{
    DisplacementFieldTransform, Euler3DTransform, ParametricTransform, TranslationTransform,
};
use support::{cell_boundary_straddles, no_device, on_cell_wall};

/// The same textured volume the mean-squares pins use: three Gaussian blobs plus a
/// low-frequency sine texture, so the moments are conditioned like a real image and
/// the derivative is nonzero away from the blobs.
fn volume(n: usize, shift: [f64; 3]) -> Image {
    let c = n as f64 / 2.0;
    let blobs = [
        (0.0, 0.0, 0.0, n as f64 / 5.0, 120.0),
        (
            n as f64 / 6.0,
            -(n as f64) / 8.0,
            n as f64 / 7.0,
            n as f64 / 9.0,
            80.0,
        ),
        (
            -(n as f64) / 5.0,
            n as f64 / 6.0,
            -(n as f64) / 9.0,
            n as f64 / 8.0,
            60.0,
        ),
    ];
    let mut v = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (
                    i as f64 - c + shift[0],
                    j as f64 - c + shift[1],
                    k as f64 - c + shift[2],
                );
                let mut s = 0.0;
                for &(bx, by, bz, sig, amp) in &blobs {
                    let d2 = (x - bx).powi(2) + (y - by).powi(2) + (z - bz).powi(2);
                    s += amp * (-d2 / (2.0 * sig * sig)).exp();
                }
                s += 10.0 * (x / 7.0).sin() * (y / 9.0).cos() * (z / 11.0).sin();
                v.push(s as f32);
            }
        }
    }
    let mut img = Image::from_vec(&[n, n, n], v).unwrap();
    img.set_spacing(&[1.0, 1.0, 1.0]).unwrap();
    img
}

fn pair(n: usize) -> (Image, Image) {
    (volume(n, [0.0, 0.0, 0.0]), volume(n, [3.0, -2.0, 1.5]))
}

fn host(fixed: &Image, moving: &Image) -> CorrelationMetric {
    CorrelationMetric::from_samples(
        FixedSamples::from_image(fixed).unwrap(),
        MovingImage::from_image(moving).unwrap(),
    )
    .unwrap()
}

fn device(fixed: &Image, moving: &Image) -> DeviceCorrelationMetric {
    DeviceCorrelationMetric::from_device(
        &DeviceImage::upload(fixed).unwrap(),
        &DeviceImage::upload(moving).unwrap(),
    )
    .unwrap()
}

/// Four poses: the identity, a small rotation, the shift that aligns the two volumes, and
/// a large displacement that pushes part of the fixed grid outside the moving buffer so the
/// valid set is a strict subset (and `sfm` there is weakest — the worst-conditioned pose).
///
/// Every translation here is a *non*-integral number of voxels, deliberately: an integral
/// one puts the rotation-centre voxel exactly on a moving-grid cell boundary, which is a
/// different measurement entirely — see
/// [`a_sample_on_a_cell_boundary_no_longer_costs_a_derivative_component`], which is the
/// pose that does it on purpose. Pin 1 still asserts the absence of a *straddle* (two paths,
/// two cells) rather than assuming it, though the stage replay now makes it impossible.
fn poses(n: usize) -> Vec<(&'static str, Euler3DTransform)> {
    let c = n as f64 / 2.0;
    vec![
        (
            "identity",
            Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]),
        ),
        (
            "small rotation",
            Euler3DTransform::new(0.06, -0.04, 0.03, [2.5, -1.5, 0.75], [c, c, c]),
        ),
        (
            "aligned",
            Euler3DTransform::new(0.0, 0.0, 0.0, [3.0, -2.0, 1.5], [c, c, c]),
        ),
        (
            "partly outside",
            Euler3DTransform::new(0.15, 0.1, -0.12, [12.3, -9.4, 7.1], [c, c, c]),
        ),
    ]
}

/// `Σ|f₁·m₁| / |Σ f₁·m₁|` — the conditioning of `sfm`, the one moment whose terms can
/// cancel (`sff` and `smm` are sums of squares and cannot). It is the factor by which
/// a relative rounding error in the per-sample products is amplified in `sfm`, and
/// hence the honest scale for the value band: at a pose where the samples decorrelate,
/// `sfm → 0` while `Σ|f₁·m₁|` does not, and this number diverges.
///
/// Computed here from an independent sampler (unit spacing, identity direction, so the
/// fixed point *is* the index), not from either metric — a diagnostic that shared code
/// with the thing it diagnoses would be worthless. Its valid count is asserted against
/// the metric's, so a diverging inside-rule cannot silently make it lie.
fn sfm_conditioning(fixed: &Image, moving: &Image, t: &dyn ParametricTransform) -> (f64, usize) {
    let f = fixed.to_f64_vec().unwrap();
    let m = moving.to_f64_vec().unwrap();
    let size = fixed.size().to_vec();
    let msize = moving.size().to_vec();
    let mstride = [1usize, msize[0], msize[0] * msize[1]];

    let sample = |p: &[f64]| -> Option<f64> {
        let c = [p[0], p[1], p[2]];
        for (d, &cd) in c.iter().enumerate() {
            if !(cd >= -0.5 && cd < msize[d] as f64 - 0.5) {
                return None;
            }
        }
        let mut value = 0.0f64;
        for corner in 0..8usize {
            let mut offset = 0usize;
            let mut weight = 1.0f64;
            for (d, &cd) in c.iter().enumerate() {
                let base = cd.floor();
                let frac = cd - base;
                let bit = (corner >> d) & 1;
                weight *= if bit == 1 { frac } else { 1.0 - frac };
                let idx = (base as isize + bit as isize).clamp(0, msize[d] as isize - 1) as usize;
                offset += idx * mstride[d];
            }
            value += weight * m[offset];
        }
        Some(value)
    };

    let mut pairs = Vec::new();
    for k in 0..size[2] {
        for j in 0..size[1] {
            for i in 0..size[0] {
                let p = t.transform_point(&[i as f64, j as f64, k as f64]);
                if let Some(mv) = sample(&p) {
                    pairs.push((f[i + j * size[0] + k * size[0] * size[1]], mv));
                }
            }
        }
    }
    let n = pairs.len() as f64;
    let fbar = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mbar = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let mut sfm = 0.0f64;
    let mut abs = 0.0f64;
    for (fv, mv) in &pairs {
        let prod = (fv - fbar) * (mv - mbar);
        sfm += prod;
        abs += prod.abs();
    }
    (abs / sfm.abs(), pairs.len())
}

/// The bands, stated.
///
/// **Value: 1e-12 absolute.** Measured worst over the four poses: 4.7e-14.
/// **Derivative: 1e-12 relative.** Measured worst over the four poses: 6.7e-15.
///
/// Both sit ~2 decades above what reduction rounding produces and many decades below
/// what a dropped term, a wrong Jacobian column or a mis-scaled moment would produce.
/// The measured numbers are printed at every pose, so a change that pushes them up names
/// the pose it happened at.
///
/// These bands measure the reduction. They used to hold only where no sample landed on a
/// moving-grid cell wall — a sample there cost the x-translation column 2.9e-7, four
/// decades outside them — and the device's stage replay closed that (see
/// [`cell_boundary_straddles`] and the pin below). Pin 1 keeps asserting there is no
/// straddle at its poses: the assertion is now a guard against a regression rather than a
/// precondition that can fail on a pose choice.
const VALUE_BAND: f64 = 1e-12;
const DERIV_BAND: f64 = 1e-12;

/// Pin 1: the device NCC against the host NCC, at four poses, no straddles.
#[test]
fn the_device_ncc_is_the_hosts_ncc() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = pair(n);
    let h = host(&fixed, &moving);
    let d = device(&fixed, &moving);

    for (name, t) in poses(n) {
        let hv: MetricValue = h.evaluate(&t);
        let dv: MetricValue = d.evaluate(&t).unwrap();

        let (cond, cond_count) = sfm_conditioning(&fixed, &moving, &t);
        assert_eq!(
            cond_count, hv.valid_points,
            "{name}: the conditioning probe walked a different sample set than the metric, \
             so the number it prints is not this metric's conditioning"
        );

        let straddles = cell_boundary_straddles(&fixed, &moving, &t);
        assert!(
            straddles.is_empty(),
            "{name}: {} sample(s) land on a moving-grid cell boundary ({straddles:?}) --- the \
             bands below would then be measuring a gradient discontinuity, not the reduction",
            straddles.len()
        );

        assert_eq!(
            dv.valid_points, hv.valid_points,
            "{name}: the device walked a different valid set than the host"
        );
        assert!(hv.valid_points > 0, "{name}: nothing maps inside");

        let v_err = (dv.value - hv.value).abs();
        let d_err = dv
            .derivative
            .iter()
            .zip(hv.derivative.iter())
            .map(|(&g, &c)| (g - c).abs() / (1.0 + c.abs()))
            .fold(0.0f64, f64::max);

        println!(
            "{name:15} valid {:>7}  cond(sfm) {cond:8.2}  host value {:.17e}  \
             |Δvalue| {v_err:.3e}  deriv rel {d_err:.3e}",
            hv.valid_points, hv.value
        );

        assert!(
            hv.derivative.iter().any(|d| d.abs() > 1e-9),
            "{name}: the host derivative is ~zero, so comparing to it proves nothing"
        );
        assert!(
            v_err <= VALUE_BAND,
            "{name}: |Δvalue| {v_err:e} exceeds the {VALUE_BAND:e} absolute band"
        );
        assert!(
            d_err <= DERIV_BAND,
            "{name}: derivative rel err {d_err:e} exceeds the {DERIV_BAND:e} band"
        );
    }
}

/// Pin 7: **a sample on a cell boundary no longer costs a derivative component.** The
/// §2.158 exposure, at the pose that produced it, with the assertions flipped.
///
/// What this pin measured before, at 64³ — a Euler transform whose centre is the fixed
/// volume's centre *voxel* and whose translation is a whole number of voxels, so the centre
/// voxel maps exactly onto an integer index and is the only sample that does:
///
/// * straddling samples: **1** (the centre voxel, (32,32,32)) — the two paths put it in
///   **different cells**
/// * `|Δ∂M/∂x|` at that sample: **7.2e-1**, an O(1) jump
/// * `|Δderivative|`: **2.9e-7** on the x-translation component, ≤1e-14 on the other five
/// * `|Δvalue|`: 4.3e-15 — the interpolant is continuous, so the value never saw it
///
/// It was not a Correlation defect and not a reduction error: it was the device's point map
/// (`p = A·x + b`, with `A` *probed*) differing from the host's evaluated `R·(x − c) + c + t`
/// in the last ulp, at a sample where that ulp decides which cell the sample is in. The
/// trilinear interpolant is continuous across a cell wall; its gradient is not.
///
/// The device is now handed the transform's own **stages** and replays them, so its
/// continuous index is the host's bit for bit, both paths pick the same cell, and both take
/// the same one-sided limit of `∂M/∂x`. The pose is unchanged — the sample is still on the
/// wall, `on_cell_wall` asserts it — and the x-translation column is back inside the
/// ordinary reduction band.
///
/// Mean squares had the same exposure at 1000× the size (it normalizes by nothing), and
/// `cuda_mean_squares.rs` pins the same flip there.
#[test]
fn a_sample_on_a_cell_boundary_no_longer_costs_a_derivative_component() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = pair(n);
    let c = n as f64 / 2.0;
    // Whole-voxel translation about a whole-voxel centre: the centre voxel maps exactly
    // onto an integer index.
    let t = Euler3DTransform::new(0.15, 0.1, -0.12, [12.0, -9.0, 7.0], [c, c, c]);

    // The construction still holds: the sample is still on the wall.
    let walls = on_cell_wall(&fixed, &moving, &t);
    println!("samples on a cell wall: {walls:?}");
    assert!(
        !walls.is_empty(),
        "no sample of this pose lands on a cell wall, so the pin has gone vacuous"
    );
    assert!(
        walls.iter().all(|(s, _)| s.index == [32, 32, 32]),
        "the sample on the wall is the rotation-centre voxel"
    );

    // And the two paths now put it in the same cell.
    let straddles = cell_boundary_straddles(&fixed, &moving, &t);
    assert!(
        straddles.is_empty(),
        "{} sample(s) land in different moving-grid cells on the two paths ({straddles:?})",
        straddles.len()
    );

    let hv = host(&fixed, &moving).evaluate(&t);
    let dv = device(&fixed, &moving).evaluate(&t).unwrap();
    assert_eq!(dv.valid_points, hv.valid_points);

    let v_err = (dv.value - hv.value).abs();
    let d_err: Vec<f64> = dv
        .derivative
        .iter()
        .zip(hv.derivative.iter())
        .map(|(&g, &c)| (g - c).abs())
        .collect();
    println!("|Δvalue|      = {v_err:.3e}");
    println!("|Δderivative| = {d_err:?}   (x-translation was 2.9e-7)");

    assert!(
        v_err <= VALUE_BAND,
        "the value moved by {v_err:e} --- it did not even under the straddle"
    );
    for (k, &e) in d_err.iter().enumerate() {
        assert!(
            e <= 1e-12,
            "param {k} moved by {e:e}. The x-translation column (param 3) is the one this \
             pose used to blow, at 2.9e-7; every column must now sit inside the ordinary \
             reduction band"
        );
    }
    assert!(
        hv.derivative.iter().any(|d| d.abs() > 1e-9),
        "the host derivative is ~zero here, so the comparison proves nothing"
    );
}

/// Pin 2: run-to-run **bit identity**, asserted exactly, across fresh backends and a
/// reused one. Two device passes mean two reductions and one host division between
/// them; every one of those must be order-fixed, or the optimizer's feedback loop
/// turns the last ulp into a different pose (the mechanism D4 measured).
#[test]
fn the_device_ncc_is_bit_identical_run_to_run() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 48;
    let (fixed, moving) = pair(n);
    let t = poses(n)[1].1.clone();

    let first = device(&fixed, &moving).evaluate(&t).unwrap();
    for run in 1..5 {
        let again = device(&fixed, &moving).evaluate(&t).unwrap();
        assert_eq!(
            again.value.to_bits(),
            first.value.to_bits(),
            "run {run}: value differs in its bits"
        );
        assert_eq!(
            again
                .derivative
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            first
                .derivative
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "run {run}: derivative differs in its bits"
        );
        assert_eq!(again.valid_points, first.valid_points);
    }
    let resident = device(&fixed, &moving);
    let a = resident.evaluate(&t).unwrap();
    let b = resident.evaluate(&t).unwrap();
    assert_eq!(a.value.to_bits(), b.value.to_bits());
    assert_eq!(a.value.to_bits(), first.value.to_bits());
    println!(
        "4 fresh metrics + 2 reused: bit-identical, value = {first:?}",
        first = first.value
    );
}

/// Pin 3: the device derivative is the derivative *of the device value*.
///
/// Central differences on the device's own value, in the device's own parameters. This
/// catches a wrong Jacobian contraction that the host comparison cannot: if the host and
/// the device made the *same* contraction mistake, pin 1 would pass and this fails.
///
/// # The valid set must not move, and the geometry here is what makes it not move
///
/// The analytic derivative — the host's and ITK's alike — differentiates the moments
/// over a *frozen* sample set. It does not, and cannot, carry a term for a sample
/// crossing the moving image's boundary: that is a step, not a slope. So a finite
/// difference over a pose where samples enter or leave the moving buffer does not
/// measure the analytic derivative at all, it measures the analytic derivative plus a
/// difference quotient of a staircase — and at 48³ with a rotation about the centre, a
/// 1e-5 rad step moves the corners ~4e-4 voxels and flips a handful of boundary samples,
/// which is enough to swamp the slope entirely (measured: analytic +2.17e-2, central
/// difference −6.97e-2, on the first rotation parameter).
///
/// This is not a defect and it is not banded away. The fixed grid here is a **24³ block
/// sitting strictly inside a 64³ moving volume**, so every sample stays inside under
/// every perturbation, and `valid_points` is *asserted constant* across all thirteen
/// evaluations. Only then is the finite difference a measurement of the slope.
#[test]
fn the_device_derivative_is_the_finite_difference_of_the_device_value() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let moving = volume(64, [3.0, -2.0, 1.5]);
    // The fixed grid: 24³ voxels, placed at physical [20,20,20] — 20 voxels of clearance
    // on every side of the 64³ moving volume, against perturbations that move a sample
    // by ~1e-3 voxels.
    let mut fixed = volume(24, [-8.0, -8.0, -8.0]);
    fixed.set_origin(&[20.0, 20.0, 20.0]).unwrap();

    let d = device(&fixed, &moving);
    let base = Euler3DTransform::new(0.06, -0.04, 0.03, [2.5, -1.5, 0.75], [32.0, 32.0, 32.0]);

    let at = d.evaluate(&base).unwrap();
    let analytic = at.derivative;
    let interior = 24 * 24 * 24;
    assert_eq!(
        at.valid_points, interior,
        "the fixed block is supposed to sit strictly inside the moving volume"
    );

    // Rotations (params 0..3) and translations (3..6) are in different units, so the
    // step is per-block, not global.
    let steps = [1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3];
    let p0 = base.parameters().to_vec();

    for (k, &h) in steps.iter().enumerate() {
        let shifted = |delta: f64| {
            let mut p = p0.clone();
            p[k] += delta;
            let mut t = base.clone();
            t.set_parameters(&p).unwrap();
            let v = d.evaluate(&t).unwrap();
            assert_eq!(
                v.valid_points, interior,
                "param {k}: the perturbed pose moved a sample out of the moving volume, so \
                 the finite difference would be measuring a staircase, not a slope"
            );
            v.value
        };
        let fd = (shifted(h) - shifted(-h)) / (2.0 * h);
        let rel = (fd - analytic[k]).abs() / (1.0 + analytic[k].abs());
        println!(
            "param {k}: analytic {:+.9e}  fd {fd:+.9e}  rel {rel:.2e}",
            analytic[k]
        );
        // Central differences are O(h²)-truncated and the value is O(1e-1) carried in
        // f64, so the floor here is ~1e-8, not the metric's own precision. This band
        // tests the *contraction*, not the reduction.
        assert!(
            rel <= 1e-6,
            "param {k}: analytic {:e} vs central difference {fd:e} (rel {rel:e})",
            analytic[k]
        );
    }
}

/// Pin 4: NCC is invariant under an affine rescale of the moving intensities — that is
/// the whole point of the metric, and the mean subtraction plus the `sfm²/(sff·smm)`
/// normalization is what buys it. A device that dropped the mean subtraction (the
/// one-pass form's failure mode, in its extreme) fails here loudly.
///
/// Band: absolute on the value, and *looser* than pin 1's — this is not a
/// reduction-order comparison, it is the same reduction over a genuinely different set
/// of floats (`3.5·m + 40` rounds differently than `m`), so the two agree to the
/// metric's conditioning, not to its determinism.
#[test]
fn an_affine_rescale_of_the_moving_intensities_leaves_the_value() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 48;
    let (fixed, moving) = pair(n);
    let v: Vec<f32> = moving
        .to_f64_vec()
        .unwrap()
        .iter()
        .map(|&x| (3.5 * x + 40.0) as f32)
        .collect();
    let mut rescaled = Image::from_vec(&[n, n, n], v).unwrap();
    rescaled.set_spacing(&[1.0, 1.0, 1.0]).unwrap();

    let t = poses(n)[1].1.clone();
    let plain = device(&fixed, &moving).evaluate(&t).unwrap();
    let scaled = device(&fixed, &rescaled).evaluate(&t).unwrap();

    let err = (scaled.value - plain.value).abs();
    println!(
        "value {:.17e} -> {:.17e} under m -> 3.5m + 40 ; |Δ| = {err:.3e}",
        plain.value, scaled.value
    );
    assert_eq!(scaled.valid_points, plain.valid_points);
    assert!(
        err <= 1e-6,
        "the value moved by {err:e} under an affine intensity rescale --- NCC is supposed \
         to be invariant to it"
    );
}

/// Pin 5: the degenerate branch, at its boundary, is **the host's branch**.
///
/// A constant moving volume makes `smm = 0`, so `m2f2 = smm·sff = 0 ≤ ε` and the host
/// returns `f64::MAX` with a zero derivative. The device does not decide this: the host
/// contraction evaluates the *same* product in the *same* order against the *same*
/// constant (`f64::EPSILON`) on the device's moments. So the two agree exactly, and
/// they agree on the *branch*, not merely on a number.
///
/// The branch is a discontinuity: on one side the value is O(1e-1), on the other it is
/// 1.8e308. Near `smm ≈ ε` a rounding difference between host and device moments could
/// put them on opposite sides — this test walks a moving volume whose contrast shrinks
/// through that boundary and reports whether the two ever straddle it. It does not widen
/// a band to hide a straddle; if the two disagree on the branch, it says so and fails.
#[test]
fn the_degenerate_branch_is_the_hosts_branch_at_its_boundary() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 32;
    let fixed = volume(n, [0.0, 0.0, 0.0]);
    let t = TranslationTransform::new(vec![0.0, 0.0, 0.0]);

    // Exactly constant: smm == 0, unambiguously inside the branch.
    let flat = Image::from_vec(&[n, n, n], vec![7.0f32; n * n * n]).unwrap();
    let hv = host(&fixed, &flat).evaluate(&t);
    let dv = device(&fixed, &flat).evaluate(&t).unwrap();
    println!(
        "constant moving: host {:e}, device {:e}",
        hv.value, dv.value
    );
    assert_eq!(
        dv.value.to_bits(),
        hv.value.to_bits(),
        "a constant moving volume must take the host's degenerate branch on both paths"
    );
    assert_eq!(hv.value, f64::MAX);
    assert!(dv.derivative.iter().all(|&x| x == 0.0));
    assert_eq!(dv.valid_points, hv.valid_points);

    // And walk *through* the boundary: contrast c gives smm ∝ c², so smm crosses
    // f64::EPSILON somewhere in this sweep. Report every straddle.
    let mut straddles = Vec::new();
    for e in 0..14i32 {
        let c = 10f64.powi(-e);
        let v: Vec<f32> = (0..n * n * n)
            .map(|i| (7.0 + c * ((i % 17) as f64 - 8.0)) as f32)
            .collect();
        let mut m = Image::from_vec(&[n, n, n], v).unwrap();
        m.set_spacing(&[1.0, 1.0, 1.0]).unwrap();

        let hv = host(&fixed, &m).evaluate(&t).value;
        let dv = device(&fixed, &m).evaluate(&t).unwrap().value;
        let h_degenerate = hv == f64::MAX;
        let d_degenerate = dv == f64::MAX;
        println!(
            "contrast 1e-{e:<2}: host {}  device {}  ({hv:+.6e} / {dv:+.6e})",
            if h_degenerate {
                "DEGENERATE"
            } else {
                "value     "
            },
            if d_degenerate {
                "DEGENERATE"
            } else {
                "value     "
            },
        );
        if h_degenerate != d_degenerate {
            straddles.push(e);
        }
    }
    assert!(
        straddles.is_empty(),
        "host and device took DIFFERENT branches at contrast 1e-{straddles:?} --- this is a \
         straddle of the degenerate boundary, reported and not widened away"
    );
}

/// Pin 6: a local-support transform is refused **by name**, not by falling through to
/// the affine probe. `CorrelationMetric::check_transform` refuses it on the host
/// (mirroring ITK's constructor, which throws); the device names the same rule.
#[test]
fn a_displacement_field_is_refused_by_name() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 16;
    let (fixed, moving) = pair(n);
    let d = device(&fixed, &moving);

    let t = DisplacementFieldTransform::new(
        3,
        &[n, n, n],
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, 1.0],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    assert!(t.has_local_support());

    let Err(err) = d.evaluate(&t) else {
        panic!("a displacement field must be refused");
    };
    assert!(
        matches!(err, DeviceMetricError::RequiresGlobalTransform),
        "refused, but as {err:?} --- the correlation metric's rule is that it is \
         global-transform-only, and the refusal must say so"
    );
    println!("refused: {err}");
    // The host's own precondition, on the same transform, for the same reason.
    assert!(host(&fixed, &moving).check_transform(&t).is_err());
}
