//! The CUDA mean-squares metric: correctness against the CPU, run-to-run
//! determinism, the fallback contract, and an end-to-end registration.
//!
//! Only compiled with the `cuda` feature; with it off this file contributes no
//! tests and the CPU suite is untouched.
#![cfg(feature = "cuda")]

mod support;

use sitk_core::Image;
use sitk_registration::metric::{FixedSamples, MovingImage};
use sitk_registration::{
    CpuBackend, CudaMetricBackend, ImageRegistrationMethod, MeanSquaresMetric, MetricBackend,
    MetricValue,
};
use sitk_transform::{BSplineTransform, Euler3DTransform, ParametricTransform};
use support::cell_boundary_straddles;

/// A smooth, textured volume: three Gaussian blobs plus a low-frequency sine
/// texture. Smooth so the metric has a real minimum; textured so the gradient is
/// nonzero away from the blobs and the derivative comparison is not vacuous.
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

/// A rigid transform displaced from the identity, so both the value and every
/// derivative component are nonzero and a wrong sign or a dropped term shows up.
fn probe_transform(n: usize) -> Euler3DTransform {
    let c = n as f64 / 2.0;
    Euler3DTransform::new(0.06, -0.04, 0.03, [2.5, -1.5, 0.75], [c, c, c])
}

/// True when the device is absent — a supported configuration, and the reason the
/// fallback exists. `CudaMetricBackend` cannot report it (it silently runs the CPU
/// path, which is the whole point), so probe the crate underneath it.
fn no_device() -> bool {
    matches!(sitk_cuda::backend(), Err(sitk_cuda::CudaError::NoDevice(_)))
}

fn metric(n: usize) -> MeanSquaresMetric {
    let (fixed, moving) = pair(n);
    MeanSquaresMetric::from_samples(
        FixedSamples::from_image(&fixed).unwrap(),
        MovingImage::from_image(&moving).unwrap(),
    )
    .unwrap()
}

/// Test (a): the GPU's value and derivative against the CPU's, on the 64³ pair.
///
/// Tolerance: **1e-9 relative**. The GPU cannot be bit-identical to the CPU here
/// and it is not supposed to be — the CPU sums N per-sample terms left to right,
/// and no parallel reduction reproduces that order. The divergence is
/// reduction-rounding, bounded by ~N·ε ≈ 4e-9 in the worst case and ~√N·ε ≈ 1e-12
/// in practice. 1e-9 sits between "rounding" and "a real modelling difference", so
/// a dropped term or a wrong Jacobian contraction fails this test loudly rather
/// than hiding under a loose band. The measured error is printed.
///
/// # The precondition, named
///
/// That band means *reduction rounding* only where no sample of this pose lands on a
/// moving-grid cell wall. Where one does, the two paths take **different one-sided limits
/// of a discontinuous ∂M/∂x**, and one such sample — out of 262144 — moves this metric's
/// derivative by **5.7e-6 relative**: 5,700× the band below, and nothing to do with the
/// reduction (ledger §2.158, and
/// [`a_sample_on_a_cell_boundary_costs_one_derivative_component`] below, which is that
/// pose on purpose, with the number measured).
///
/// So the absence of a straddle is *asserted*, not assumed. Without the assertion this
/// pin's 1e-9 is a claim about the reduction that happens to hold because of the
/// geometry — and an edit to `probe_transform` that put one sample on an integer index
/// would blow it by four decades while looking like a kernel regression.
#[test]
fn gpu_value_and_derivative_match_the_cpu() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let m = metric(n);
    let t = probe_transform(n);

    let (fixed, moving) = pair(n);
    let straddles = cell_boundary_straddles(&fixed, &moving, &t);
    assert!(
        straddles.is_empty(),
        "{} sample(s) of this pose land on a moving-grid cell wall ({straddles:?}) --- the \
         1e-9 band below would then be measuring a gradient discontinuity, not the reduction",
        straddles.len()
    );

    let cpu: MetricValue = m.evaluate(&t, &CpuBackend);
    let gpu: MetricValue = m.evaluate(&t, &CudaMetricBackend::new());

    assert_eq!(
        gpu.valid_points, cpu.valid_points,
        "the GPU must walk the same valid-sample set as the CPU"
    );
    assert!(
        cpu.valid_points > 0,
        "the probe transform maps nothing inside"
    );

    let rel = |g: f64, c: f64| (g - c).abs() / (1.0 + c.abs());
    let v_err = rel(gpu.value, cpu.value);
    let d_err = gpu
        .derivative
        .iter()
        .zip(cpu.derivative.iter())
        .map(|(&g, &c)| rel(g, c))
        .fold(0.0f64, f64::max);

    println!("valid_points   = {} (both)", cpu.valid_points);
    println!("cpu value      = {:.17e}", cpu.value);
    println!("gpu value      = {:.17e}", gpu.value);
    println!("value rel err  = {v_err:e}");
    println!(
        "deriv rel err  = {d_err:e}  (max over {} params)",
        cpu.derivative.len()
    );
    println!("cpu derivative = {:?}", cpu.derivative);
    println!("gpu derivative = {:?}", gpu.derivative);

    assert!(v_err <= 1e-9, "value rel err {v_err:e} exceeds 1e-9");
    assert!(d_err <= 1e-9, "derivative rel err {d_err:e} exceeds 1e-9");

    // A derivative of all zeros would pass any tolerance test against a CPU that
    // also returned zeros. It must actually be nonzero, or the comparison is vacuous.
    assert!(
        cpu.derivative.iter().any(|d| d.abs() > 1e-6),
        "the CPU derivative is ~zero here, so the comparison proves nothing"
    );
}

/// **One sample on a moving-grid cell wall costs mean squares one derivative component
/// — and leaves its value untouched.** The exposure §2.158 records for Correlation,
/// pinned where it also exists: in the metric that was written first and never showed it.
///
/// The mechanism is not the metric's. It is the device's point map (`p = A·x + b`, with
/// `A` *probed* as `T(e_e) − T(0)`) differing from the host's (`R·(x − c) + c + t`,
/// evaluated) in the last ulp, at a sample where that ulp decides which **cell** of the
/// moving grid the sample sits in. The trilinear interpolant is continuous across a cell
/// wall; its gradient is not. So any metric whose derivative consumes `∇M` sees it, and
/// mean squares' derivative does: `∂/∂p = (2/N)·Σ (m − f)·∇M·J`.
///
/// The pose puts it there on purpose: a Euler transform whose centre is the volume's
/// centre *voxel* and whose translation is a whole number of voxels. The centre voxel is
/// fixed by the rotation, so it maps to `centre + t` — exactly integral, exactly on a
/// cell wall — and it is the only such sample. Measured, at 64³:
///
/// * straddling samples: **1** — the centre voxel (32, 32, 32); `|Δ∂M/∂x|` there is
///   **0.7156**, an O(1) jump, as a second difference of the image should be
/// * `|Δvalue|`: **3.5e-15 relative** — the interpolant is continuous, so the value does
///   not see it
/// * `|Δderivative|`: **2.996e-4** on the x-translation column (**5.7e-6 relative**),
///   **≤1.1e-11** on the other five
///
/// The size of that one number is the whole reason this pin exists. It is exactly one
/// sample's worth of the `(2/N)·Σ (m − f)·∇M·J` sum: `2/262144 · 55 · 0.7156 ≈ 3e-4`,
/// with 55 the residual at that voxel — an ordinary one for intensities of O(100). It is
/// **1000× what the same straddle costs Correlation** (2.9e-7), because NCC divides by
/// `sff·smm` and mean squares divides by nothing. A metric with no normalization has no
/// protection from this, which is why the loudest version of the exposure lives here.
///
/// # What is asserted, and why each assertion is the interesting one
///
/// * **The value does not move.** This is the assertion that proves the pin is about
///   `∇M` and not about the sampler in general: mean squares' *value* is built from the
///   interpolated intensity alone, which is continuous across the wall, so it must stay
///   inside the ordinary reduction band. If the value ever moves, the cause is not this.
/// * **Five of the six derivative components do not move.** The straddling sample *is*
///   the rotation centre, so the rotation columns of its Jacobian vanish, and the
///   gradient jump is along x alone — leaving only the x-translation column exposed.
///   The mechanism predicts the signature exactly, and the signature is asserted.
/// * **The x-translation column is bounded BELOW as well as above.** Below, because a
///   collapse to rounding would mean the device's point map had become bit-identical to
///   the host's and this whole pin's story had changed; above, because a wrong
///   contraction is orders of magnitude larger than one sample's worth of gradient jump.
#[test]
fn a_sample_on_a_cell_boundary_costs_one_derivative_component() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = pair(n);
    let m = metric(n);
    let c = n as f64 / 2.0;
    let t = Euler3DTransform::new(0.15, 0.1, -0.12, [12.0, -9.0, 7.0], [c, c, c]);

    let straddles = cell_boundary_straddles(&fixed, &moving, &t);
    println!("straddling samples: {straddles:?}");
    assert_eq!(
        straddles.len(),
        1,
        "the pose is constructed so exactly one sample lands on a cell wall"
    );
    let ([i, j, k], axis, jump) = straddles[0];
    assert_eq!(
        [i, j, k],
        [32, 32, 32],
        "and it is the rotation-centre voxel"
    );
    assert_eq!(axis, 0, "the gradient jump is along x");
    assert!(
        jump > 1e-2,
        "|Δ∂M/∂x| = {jump:e} --- the jump across a cell wall is O(1), not O(ε)"
    );

    let cpu: MetricValue = m.evaluate(&t, &CpuBackend);
    let gpu: MetricValue = m.evaluate(&t, &CudaMetricBackend::new());
    assert_eq!(
        gpu.valid_points, cpu.valid_points,
        "a cell-wall straddle must not move the valid set: the sample is inside on both \
         paths, it is only interpolated differently"
    );

    // The same `rel` the ordinary pin bands at 1e-9, so the two numbers are comparable and
    // the claim "this straddle would fail that pin" is checkable rather than asserted.
    let rel = |g: f64, c: f64| (g - c).abs() / (1.0 + c.abs());
    let v_err = rel(gpu.value, cpu.value);
    let d_abs: Vec<f64> = gpu
        .derivative
        .iter()
        .zip(cpu.derivative.iter())
        .map(|(&g, &c)| (g - c).abs())
        .collect();
    let d_rel: Vec<f64> = gpu
        .derivative
        .iter()
        .zip(cpu.derivative.iter())
        .map(|(&g, &c)| rel(g, c))
        .collect();
    println!("value rel err  = {v_err:.3e}");
    println!("cpu derivative = {:?}", cpu.derivative);
    println!("|Δderivative|  = {d_abs:?}");
    println!(
        "x-translation column: |Δ| = {:.4e}, rel = {:.4e}",
        d_abs[3], d_rel[3]
    );

    assert!(
        v_err <= 1e-12,
        "the straddle moved the value by {v_err:e} relative --- it must not: mean squares' \
         value never touches ∇M, and the interpolant is continuous across the wall the \
         straddle is about. If the value moves, the cause is not this"
    );
    for p in [0usize, 1, 2, 4, 5] {
        assert!(
            d_rel[p] <= 1e-9,
            "param {p} moved by {:e} relative --- the other five columns must stay inside the \
             ordinary reduction band: only the x-translation column can see this straddle, \
             because the straddling sample IS the rotation centre and its rotation Jacobian \
             is zero",
            d_rel[p]
        );
    }
    // The punchline, and the reason `gpu_value_and_derivative_match_the_cpu` now asserts
    // the absence of a straddle instead of assuming it: this pose fails that pin's band by
    // ~3.5 decades, on one sample out of 262144.
    assert!(
        d_rel[3] > 1e-9,
        "the x-translation column moved by only {:e} relative --- one straddling sample is \
         supposed to BLOW the 1e-9 band the ordinary pin bands at, and that is the whole \
         reason that pin has to assert there is no straddle",
        d_rel[3]
    );
    assert!(
        (1e-6..1e-2).contains(&d_abs[3]),
        "the x-translation column moved by {:e} absolute, outside the measured cost of one \
         straddling sample ((2/N)·|m − f|·|Δ∂M/∂x| ≈ 3e-4). Bounded above because a wrong \
         contraction is orders of magnitude larger; bounded BELOW because a collapse to \
         rounding means the device's point map has become the host's, and then this pin must \
         be rewritten rather than passed",
        d_abs[3]
    );
}

/// Test (b): **run-to-run bit-identity**, asserted exactly.
///
/// This is a correctness property, not a performance one. The optimizer is a
/// feedback loop: a metric that varies in its last ulp between runs makes the
/// registration *result* vary between runs. The kernel therefore uses a fixed
/// grid, a fixed shared-memory reduction tree, no `atomicAdd`, and a host-side
/// fold in block-index order — so every run performs the same additions in the
/// same order. Equality here is `==` on the bits, with no tolerance at all.
#[test]
fn gpu_result_is_bit_identical_run_to_run() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let m = metric(n);
    let t = probe_transform(n);

    let first = m.evaluate(&t, &CudaMetricBackend::new());
    for run in 1..8 {
        // A fresh backend each time, so the residency upload is redone and the
        // reduction cannot be accidentally memoized.
        let again = m.evaluate(&t, &CudaMetricBackend::new());
        assert_eq!(
            again.value.to_bits(),
            first.value.to_bits(),
            "run {run}: value differs in its bits ({:.17e} vs {:.17e})",
            again.value,
            first.value
        );
        assert_eq!(
            again
                .derivative
                .iter()
                .map(|d| d.to_bits())
                .collect::<Vec<_>>(),
            first
                .derivative
                .iter()
                .map(|d| d.to_bits())
                .collect::<Vec<_>>(),
            "run {run}: derivative differs in its bits"
        );
        assert_eq!(again.valid_points, first.valid_points);
    }
    // And within one resident backend, across repeated evaluations.
    let backend = CudaMetricBackend::new();
    let a = m.evaluate(&t, &backend);
    let b = m.evaluate(&t, &backend);
    assert_eq!(a.value.to_bits(), b.value.to_bits());
    assert_eq!(a.value.to_bits(), first.value.to_bits());
    println!(
        "8 fresh backends + 2 reused: all bit-identical, value = {:.17e}",
        first.value
    );
}

/// Test (c): the fallback contract, exercised through the *mathematics* rather
/// than a type list.
///
/// A B-spline's point map and Jacobian are not affine in the point, so it fails
/// the linearity probe and the backend declines to the CPU. The result must be the
/// CPU's, bit-for-bit — the GPU backend is not allowed to change the answer when
/// it declines, and it is not allowed to fail.
///
/// The B-spline is the sharpest case available for this, because it is *not*
/// caught by any of the backend's cheap structural guards: this port follows ITK
/// in reporting `has_local_support() == false` for a B-spline
/// (`GetTransformCategory()` returns `BSpline`, not `DisplacementField`), the
/// dimension is 3, and the interpolator is linear. Every structural gate passes.
/// The only thing left that can decline it is the linearity probe itself — which
/// is exactly the claim under test.
#[test]
fn a_non_affine_transform_falls_back_to_the_cpu_bit_for_bit() {
    let n = 32;
    let m = metric(n);
    let mut t = BSplineTransform::new(
        3,
        &[0.0, 0.0, 0.0],
        &[n as f64, n as f64, n as f64],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[4, 4, 4],
    )
    .unwrap();
    // Bend it: with all-zero coefficients a B-spline's *point map* degenerates to
    // the identity, which is affine. A deformed one is nonlinear in the point on
    // both counts, so this tests the probe on the point map as well as the Jacobian.
    let np = t.number_of_parameters();
    let coeffs: Vec<f64> = (0..np)
        .map(|k| 3.0 * ((k as f64) * 0.7).sin() + 1.5 * ((k as f64) * 0.13).cos())
        .collect();
    t.set_parameters(&coeffs).unwrap();

    assert!(
        !t.has_local_support(),
        "ITK reports a B-spline as BSpline, not DisplacementField --- if this ever \
         flips, the backend's cheap structural guard would catch the B-spline first \
         and this test would stop proving that the *math* is what declines it"
    );

    let cpu = m.evaluate(&t, &CpuBackend);
    let gpu = m.evaluate(&t, &CudaMetricBackend::new());

    assert_eq!(gpu.value.to_bits(), cpu.value.to_bits());
    assert_eq!(gpu.valid_points, cpu.valid_points);
    assert_eq!(
        gpu.derivative
            .iter()
            .map(|d| d.to_bits())
            .collect::<Vec<_>>(),
        cpu.derivative
            .iter()
            .map(|d| d.to_bits())
            .collect::<Vec<_>>(),
    );

    // Value-only path too: the gradient-free optimizers take it.
    let cpu_v = m.value(&t, &CpuBackend);
    let gpu_v = m.value(&t, &CudaMetricBackend::new());
    assert_eq!(gpu_v.to_bits(), cpu_v.to_bits());

    println!("B-spline declined to the CPU, bit-for-bit (value = {cpu_v:.17e})");
}

/// Where the time actually goes: the one-off upload, the per-iteration
/// evaluation, and what the resident buffers save by not re-uploading.
///
/// `#[ignore]`d — it is a measurement, not an assertion, and at 256³ it runs the
/// CPU metric several times. Run it with:
///
/// ```text
/// cargo test -p sitk-registration --features cuda --release -- --ignored --nocapture
/// ```
#[test]
#[ignore = "measurement, not an assertion; minutes at 256^3"]
fn perf_upload_once_then_iterate() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let ms = |f: &mut dyn FnMut()| {
        let t = std::time::Instant::now();
        f();
        t.elapsed().as_secs_f64() * 1e3
    };

    for &n in &[64usize, 128, 256] {
        let m = metric(n);
        let t = probe_transform(n);
        let gpu = CudaMetricBackend::new();

        // The first GPU evaluation pays the upload; every later one does not. The
        // difference between them IS what residency buys, per iteration.
        let first = ms(&mut || {
            std::hint::black_box(m.evaluate(&t, &gpu));
        });
        let warm: Vec<f64> = (0..5)
            .map(|_| {
                ms(&mut || {
                    std::hint::black_box(m.evaluate(&t, &gpu));
                })
            })
            .collect();
        let warm = median(warm);

        let cpu: Vec<f64> = (0..3)
            .map(|_| {
                ms(&mut || {
                    std::hint::black_box(m.evaluate(&t, &CpuBackend));
                })
            })
            .collect();
        let cpu = median(cpu);

        println!(
            "{n}^3 ({} samples): upload+first {first:.2} ms | gpu/iter {warm:.3} ms | \
             cpu/iter {cpu:.1} ms | per-iter {:.0}x | residency saves {:.2} ms/iter",
            n * n * n,
            cpu / warm,
            first - warm,
        );
    }
}

/// Test (d): end-to-end. A registration driven by the GPU backend must land on the
/// same transform as one driven by the CPU backend.
///
/// This is the test that would catch a metric that is individually plausible but
/// steers the optimizer somewhere else — the feedback loop makes small errors
/// compound, so agreement here is a much stronger statement than agreement on one
/// evaluation.
#[test]
fn a_registration_driven_by_the_gpu_lands_where_the_cpu_lands() {
    if no_device() {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 64;
    let (fixed, moving) = pair(n);
    let c = n as f64 / 2.0;

    let run = |backend: Box<dyn MetricBackend>| {
        let mut reg = ImageRegistrationMethod::new();
        reg.set_metric_as_mean_squares();
        reg.set_optimizer_as_regular_step_gradient_descent(1.0, 1e-4, 25, 1e-8);
        reg.set_metric_backend(backend);
        let initial = Euler3DTransform::new(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], [c, c, c]);
        let t = std::time::Instant::now();
        let r = reg.execute(&fixed, &moving, initial).unwrap();
        (r, t.elapsed().as_secs_f64() * 1e3)
    };

    let (cpu, cpu_ms) = run(Box::new(CpuBackend));
    let (gpu, gpu_ms) = run(Box::new(CudaMetricBackend::new()));

    let cp = cpu.transform.parameters();
    let gp = gpu.transform.parameters();
    println!(
        "cpu: {} iters, {cpu_ms:.1} ms, metric {:.9}",
        cpu.iterations, cpu.metric_value
    );
    println!(
        "gpu: {} iters, {gpu_ms:.1} ms, metric {:.9}",
        gpu.iterations, gpu.metric_value
    );
    println!("cpu params = {cp:?}");
    println!("gpu params = {gp:?}");
    println!("speedup    = {:.1}x on the whole run", cpu_ms / gpu_ms);

    assert_eq!(
        gpu.iterations, cpu.iterations,
        "the two runs took different paths"
    );
    for (k, (&g, &c)) in gp.iter().zip(cp.iter()).enumerate() {
        let rel = (g - c).abs() / (1.0 + c.abs());
        assert!(
            rel <= 1e-6,
            "param {k}: gpu {g:e} vs cpu {c:e} (rel {rel:e}) --- the GPU metric steered the \
             optimizer somewhere else"
        );
    }
}
