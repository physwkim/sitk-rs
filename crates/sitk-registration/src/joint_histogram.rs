//! Joint-histogram mutual-information metric
//! (`itk::JointHistogramMutualInformationImageToImageMetricv4`, the
//! Viola–Wells style estimator described in \cite thevenaz2000).
//!
//! Like [`MattesMutualInformationMetric`](crate::MattesMutualInformationMetric)
//! this measures the statistical dependence between the fixed image `F` and the
//! transformed moving image `M(T(x))` from their joint intensity distribution,
//! so it registers *multi-modality* pairs (any invertible intensity relation,
//! not just `M ≈ F`). It differs from Mattes in how the joint density is built
//! and how the derivative is formed:
//!
//! ```text
//! MI = (1/ln 2) Σ_{f,m} p(f,m) · ln( p(f,m) / ( p_F(f) · p_M(m) ) )
//! value = −MI                     (minimized; ln 2 converts nats → bits)
//! ```
//!
//! The joint density `p(f,m)` is a **hard-binned, then Gaussian-smoothed**
//! `bins × bins` histogram: every sample's (normalized fixed, normalized
//! moving) intensity pair increments exactly *one* bin (nearest-bin, round-half
//! -up — no per-sample Parzen window), the whole `bins × bins` array is then
//! blurred by the **discrete Gaussian** operator (`VarianceForJointPDFSmoothing`,
//! a caller-supplied [`JointHistogramMutualInformationMetric::new`] parameter,
//! ITK/SimpleITK default `1.5` via
//! [`DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING`](JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING);
//! ITK's `itk::DiscreteGaussianImageFilter` — the true discrete analogue of
//! the continuous Gaussian, via modified Bessel functions, not a
//! truncated-and-renormalized sampled Gaussian), and the marginals are column/
//! row sums of the *smoothed* array. The derivative treats this smoothed array
//! as a frozen, continuously-interpolated density field and — for each sample
//! — differentiates the field along the moving axis only (the fixed image's
//! own value does not move with the transform), central-difference style, at
//! the sample's own bilinearly-interpolated location:
//!
//! ```text
//! scalingfactor = ln2 · (∂p_M/∂m)/p_M(m) · p(f,m)/p_M(m)  −  (∂p/∂m) · ln(p(f,m)/p_M(m))
//! ∂value/∂pₖ    = Σ_samples scalingfactor · ( ∇M(T(x))·J_T(x) )ₖ   (÷ N, sign per below)
//! ```
//!
//! This is the standard Viola–Wells "frozen density" gradient estimate: because
//! the histogram is *hard*-binned (not a smooth Parzen kernel), `value` is
//! technically a step function of the transform parameters — as a sample's
//! transformed position crosses a bin boundary it *rebins* into a different
//! histogram cell — so no classical analytic gradient of `value` exists
//! everywhere. The formula above is a smooth surrogate that deliberately
//! excludes the rebinning contribution: it differentiates the already-built,
//! already-smoothed density surface as if it stayed fixed while only the
//! sample's own evaluation point moves, matching ITK's literal
//! `ProcessPoint`. Empirically (see the test module), this makes the
//! surrogate's *magnitude* diverge from a literal finite difference of
//! `value` — and the gap widens, not narrows, with more histogram bins,
//! confirming it is the (structurally excluded) rebinning term rather than a
//! resolution artifact. ITK's own unit test for this metric does not check
//! the derivative against a finite difference either. What the surrogate
//! does preserve is *direction*: it reliably points in the direction that
//! reduces `value`, which is what `gradient_descent_recovers_a_translated_blob`
//! demonstrates and is what actually matters for gradient-based optimization.
//!
//! ## Parity notes vs ITK
//!
//! * **No B-spline Parzen window.** Despite the header
//!   (`itkJointHistogramMutualInformationImageToImageMetricv4.h`) including
//!   `itkBSplineDerivativeKernelFunction.h`, neither the metric's own `.hxx` nor
//!   either threader (`ComputeJointPDFThreader`,
//!   `GetValueAndDerivativeThreader`) ever calls it — that include is dead. The
//!   actual binning is a hard nearest-bin histogram (see below), unlike Mattes'
//!   cubic B-spline Parzen window on the moving axis. This is a genuinely
//!   different algorithm from Mattes, not just a naming variant.
//! * **Marginal-array content is swapped in the literal ITK source — not
//!   reproduced here.** `itkJointHistogramMutualInformationImageToImageMetricv4
//!   .hxx`'s `InitializeForIteration` computes the two marginals via
//!   `ImageLinearIteratorWithIndex` with `SetDirection(0)`/`SetDirection(1)`.
//!   Tracing the iterator's actual index semantics (confirmed against
//!   `itkImageLinearConstIteratorWithIndex.h`'s `NextLine()`, which advances the
//!   *other* index) shows `m_FixedImageMarginalPDF` is populated with the sum
//!   over the **fixed** axis (i.e. the **moving** marginal), and
//!   `m_MovingImageMarginalPDF` with the sum over the **moving** axis (i.e. the
//!   **fixed** marginal) — the two arrays hold content swapped relative to
//!   their names. The source's own comment right above the first loop even says
//!   "Compute moving image marginal PDF by summing over fixed image bins" while
//!   storing the result into `m_FixedImageMarginalPDF`. `ComputeValue` then
//!   pairs `m_FixedImageMarginalPDF[ii]` with the joint pdf's fixed-axis index
//!   `ii` and `m_MovingImageMarginalPDF[jj]` with its moving-axis index `jj` —
//!   i.e. by name, not content — which does **not** reduce to the standard MI
//!   formula unless the fixed and moving marginals happen to be identical
//!   (verified by hand: under true independence with differently-shaped fixed/
//!   moving marginals, the literal ITK formula does not evaluate to zero, so it
//!   is not a valid divergence in general). This crate computes both marginals
//!   directly and correctly (`fixed_marginal[f] = Σ_m jp[f,m]`,
//!   `moving_marginal[m] = Σ_f jp[f,m]`) rather than reproducing the swap, which
//!   is required for the multi-modality test below (differently-shaped
//!   marginals) to have a well-defined optimum at alignment. Note the
//!   derivative threader (`GetValueAndDerivativeThreader::ProcessPoint`) only
//!   ever reads `m_MovingImageMarginalPDF` (never the "fixed" array), so this
//!   correction affects only `ComputeValue`'s pairing, not the derivative
//!   formula's structure — the derivative here uses the crate's correctly-
//!   computed `moving_marginal` throughout, consistent with `ComputeValue`.
//! * **`term2`'s `ln2` factor is asymmetric with `ComputeValue`'s nats→bits
//!   conversion — reproduced here, not corrected, unlike the marginal swap
//!   above.** `itkJointHistogramMutualInformationGetValueAndDerivativeThreader
//!   .hxx:147-151` computes `term2 = m_Log2 * dMmPDF * jointPDFValue /
//!   movingImagePDFValue` (`m_Log2 = std::log(2.0)`, ctor-initialized at
//!   `itkJointHistogramMutualInformationImageToImageMetricv4.hxx:53`) and
//!   `scalingfactor = term2 - term1`, while that same class's `ComputeValue`
//!   divides the *whole* summed value by `m_Log2` only at the very end
//!   (`.hxx:374`: `return -1.0 * total_mi.GetSum() / this->m_Log2;`) — i.e.
//!   `value`'s nats→bits conversion divides everything by `ln2`, but the
//!   derivative multiplies only `term2` by `ln2`, not the whole
//!   `scalingfactor`. This is provably not an exact frozen-density gradient
//!   of anything: writing `phi(J, mm)` for a scalar potential with
//!   `∂phi/∂J = ln J − ln mm` (to match `term1`) and `∂phi/∂mm = −ln2·J/mm`
//!   (to match `term2`), Clairaut's theorem requires the mixed partials to
//!   agree — `∂/∂mm[ln J − ln mm] = −1/mm` must equal
//!   `∂/∂J[−ln2·J/mm] = −ln2/mm` — which is false whenever `ln2 ≠ 1`, so no
//!   such `phi` exists (a proof, not an unobserved coincidence). Dropping the
//!   `ln2` *would* make `term1 − term2` exactly the gradient of
//!   `phi(J,mm) = J·(ln J − ln mm) − J` (see
//!   `frozen_density_identity_matches_finite_difference` in the test module,
//!   which checks exactly this ln2-free identity). But unlike the marginal
//!   swap, which has one demonstrably correct answer whenever the two
//!   marginals differ in shape, there is no unambiguous fix here: no
//!   per-sample frozen-density formula is the exact gradient of the true
//!   bin-summed MI regardless of the `ln2` factor, so removing it would trade
//!   ITK's inconsistency for a different heuristic rather than fixing a bug,
//!   and would silently change every derivative value a SimpleITK-parity user
//!   compares against. This port therefore keeps the literal, `ln2`-laden
//!   `term2` in [`evaluate`](JointHistogramMutualInformationMetric::evaluate).
//! * **`ComputeFixedImageMarginalPDFDerivative` is dead code, not ported.** It
//!   is declared in the threader header but never called from `ProcessPoint`
//!   (only `ComputeMovingImageMarginalPDFDerivative` is; the fixed image's own
//!   coordinate does not depend on the transform, so its marginal's derivative
//!   is never needed).
//! * **No local-support (`DisplacementFieldTransform`) memory optimization.**
//!   ITK's threader code for this metric is identical for every transform
//!   category (it always calls the moving transform's — possibly internally
//!   sparse — Jacobian through the same dense-looking loop; the local-vs-global
//!   memory specialization Mattes hand-writes does not exist for this metric in
//!   ITK). This port likewise uses one path
//!   ([`ParametricTransform::jacobian_wrt_parameters`], the dense/global
//!   Jacobian) for every transform, which is correct for any transform
//!   (verified for Mattes: the dense and local-support paths agree exactly),
//!   just not memory-optimal for a huge per-pixel displacement field.
//! * **Full sampling, exact interpolant gradient.** Same as Mattes/mean
//!   squares: every fixed pixel is sampled (SimpleITK's default), and `∇M` is
//!   the exact gradient of the linear interpolant
//!   (`MovingImage::value_and_physical_gradient`), so the analytic derivative
//!   is checked against a finite difference of the *interpolated* value.
//!
//! [`MattesMutualInformationMetric`]: crate::MattesMutualInformationMetric

use sitk_core::Image;
use sitk_transform::ParametricTransform;

use crate::error::{RegistrationError, Result};
use crate::metric::{FixedSamples, MetricValue, MovingImage};
use crate::scales::PhysicalShiftScales;

/// Bins of padding at each histogram-axis end. ITK's `m_Padding(2)` ctor
/// initializer in `itkJointHistogramMutualInformationImageToImageMetricv4`.
/// Independently defined here (not `pub` in that ITK class either) rather than
/// imported from [`crate::mattes`], which has its own private `PADDING` for
/// its (unrelated, cubic-B-spline-window) Mattes binning scheme.
const PADDING: usize = 2;

/// Minimum histogram bins. ITK's `NumberOfHistogramBins` setter clamps to a
/// floor of 5 (`itkSetClampMacro(..., 5, ...)`), but that floor makes this
/// metric's own bin spacing `1 / (bins − 2·padding − 1)` divide by zero (ITK
/// clamps to 5 without checking this — a second, narrower defect in the same
/// class). The smallest bin count that keeps the spacing finite and positive
/// is 6. This constructor shares
/// [`RegistrationError::TooFewHistogramBins`](crate::error::RegistrationError::TooFewHistogramBins)
/// with the Mattes metric, whose message text hardcodes "at least 5" — for
/// this metric that wording understates the actual minimum (6) enforced here;
/// the `bins` value in the error is still correct.
const MIN_BINS: usize = 2 * PADDING + 2;

/// Default discrete-Gaussian maximum truncation error (ITK's `SetMaximumError
/// (.01f)` call in `InitializeForIteration`).
const SMOOTHING_MAX_ERROR: f64 = 0.01;
/// Default discrete-Gaussian maximum kernel width (ITK
/// `DiscreteGaussianImageFilter`'s ctor default).
const SMOOTHING_MAX_KERNEL_WIDTH: usize = 32;

/// Round half-integer up (ties round toward `+∞`), matching
/// `itk::Math::RoundHalfIntegerUp` (`1.5 → 2`, `2.5 → 3`, `−1.5 → −1`) — the
/// rounding `itk::ImageBase::TransformPhysicalPointToIndex` uses to turn the
/// joint-PDF point into a bin index.
fn round_half_up(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

/// Modified Bessel function `I₀(y)`. Verbatim port (Abramowitz & Stegun
/// 9.8.1/9.8.2 polynomial approximations) of
/// `itk::GaussianOperator::ModifiedBesselI0`.
fn modified_bessel_i0(y: f64) -> f64 {
    let d = y.abs();
    if d < 3.75 {
        let mut m = y / 3.75;
        m *= m;
        1.0 + m
            * (3.5156229
                + m * (3.0899424
                    + m * (1.2067492 + m * (0.2659732 + m * (0.360768e-1 + m * 0.45813e-2)))))
    } else {
        let m = 3.75 / d;
        (d.exp() / d.sqrt())
            * (0.39894228
                + m * (0.1328592e-1
                    + m * (0.225319e-2
                        + m * (-0.157565e-2
                            + m * (0.916281e-2
                                + m * (-0.2057706e-1
                                    + m * (0.2635537e-1
                                        + m * (-0.1647633e-1 + m * 0.392377e-2))))))))
    }
}

/// Modified Bessel function `I₁(y)`. Verbatim port of
/// `itk::GaussianOperator::ModifiedBesselI1`.
fn modified_bessel_i1(y: f64) -> f64 {
    let d = y.abs();
    let accumulator = if d < 3.75 {
        let mut m = y / 3.75;
        m *= m;
        d * (0.5
            + m * (0.87890594
                + m * (0.51498869
                    + m * (0.15084934 + m * (0.2658733e-1 + m * (0.301532e-2 + m * 0.32411e-3))))))
    } else {
        let m = 3.75 / d;
        let mut acc = 0.2282967e-1 + m * (-0.2895312e-1 + m * (0.1787654e-1 - m * 0.420059e-2));
        acc = 0.39894228
            + m * (-0.3988024e-1
                + m * (-0.362018e-2 + m * (0.163801e-2 + m * (-0.1031555e-1 + m * acc))));
        acc * (d.exp() / d.sqrt())
    };
    if y < 0.0 { -accumulator } else { accumulator }
}

/// Modified Bessel function `Iₙ(y)`, `n ≥ 2`. Verbatim port (Numerical
/// Recipes-style downward recurrence) of `itk::GaussianOperator::ModifiedBesselI`.
fn modified_bessel_i(n: i32, y: f64) -> f64 {
    debug_assert!(n >= 2);
    if y == 0.0 {
        return 0.0;
    }
    const ACCURACY: f64 = 40.0;
    let toy = 2.0 / y.abs();
    let mut qip = 0.0f64;
    let mut qi = 1.0f64;
    let mut accumulator = 0.0f64;
    let mut j = 2 * (n + (ACCURACY * n as f64).sqrt() as i32);
    while j > 0 {
        let qim = qip + j as f64 * toy * qi;
        qip = qi;
        qi = qim;
        if qi.abs() > 1.0e10 {
            accumulator *= 1.0e-10;
            qi *= 1.0e-10;
            qip *= 1.0e-10;
        }
        if j == n {
            accumulator = qip;
        }
        j -= 1;
    }
    accumulator *= modified_bessel_i0(y) / qi;
    if y < 0.0 && (n & 1) != 0 {
        -accumulator
    } else {
        accumulator
    }
}

/// Generate the symmetric discrete-Gaussian kernel (`itk::GaussianOperator::
/// GenerateCoefficients`, `itk::DiscreteGaussianImageFilter`'s true discrete
/// Gaussian via modified Bessel functions), truncated once the coefficients'
/// area covers `1 − max_error`, capped at `max_kernel_width` taps.
fn discrete_gaussian_kernel(variance: f64, max_error: f64, max_kernel_width: usize) -> Vec<f64> {
    let et = (-variance).exp();
    let mut coeff = vec![et * modified_bessel_i0(variance)];
    let mut sum = coeff[0];
    coeff.push(et * modified_bessel_i1(variance));
    sum += coeff[1] * 2.0;

    let cap = 1.0 - max_error;
    let mut i = 2i32;
    while sum < cap {
        let c = et * modified_bessel_i(i, variance);
        coeff.push(c);
        sum += c * 2.0;
        if c <= 0.0 {
            break;
        }
        if coeff.len() > max_kernel_width {
            break;
        }
        i += 1;
    }
    for c in coeff.iter_mut() {
        *c /= sum;
    }

    let k = coeff.len();
    let mut kernel = vec![0.0; 2 * k - 1];
    for (idx, &c) in coeff.iter().enumerate() {
        kernel[k - 1 + idx] = c;
        kernel[k - 1 - idx] = c;
    }
    kernel
}

/// Separable 1-D convolution of a `rows × cols` array along `axis` (0 = rows,
/// 1 = columns) with `kernel` (odd length, centred), clamping out-of-range
/// taps to the nearest edge sample — `itk::ZeroFluxNeumannBoundaryCondition`,
/// `DiscreteGaussianImageFilter`'s default boundary condition.
fn convolve_axis(data: &[f64], rows: usize, cols: usize, axis: usize, kernel: &[f64]) -> Vec<f64> {
    let radius = (kernel.len() / 2) as isize;
    let clamp = |i: isize, len: usize| -> usize {
        if i < 0 {
            0
        } else if i as usize >= len {
            len - 1
        } else {
            i as usize
        }
    };
    let mut out = vec![0.0; rows * cols];
    if axis == 0 {
        for m in 0..cols {
            for f in 0..rows {
                let mut acc = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    let src = clamp(f as isize + k as isize - radius, rows);
                    acc += w * data[src * cols + m];
                }
                out[f * cols + m] = acc;
            }
        }
    } else {
        for f in 0..rows {
            for m in 0..cols {
                let mut acc = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    let src = clamp(m as isize + k as isize - radius, cols);
                    acc += w * data[f * cols + src];
                }
                out[f * cols + m] = acc;
            }
        }
    }
    out
}

/// The joint-histogram mutual-information metric. Holds the precomputed fixed
/// samples, moving image, and the histogram geometry (bin count, true
/// intensity ranges, bin spacing) derived once at construction.
/// [`evaluate`](Self::evaluate) returns `value = −MI` (in bits) plus its
/// parameter-derivative for a given transform.
pub struct JointHistogramMutualInformationMetric {
    fixed: FixedSamples,
    moving: MovingImage,
    num_bins: usize,
    fixed_true_min: f64,
    fixed_true_max: f64,
    moving_true_min: f64,
    moving_true_max: f64,
    /// Joint-PDF axis spacing: `1 / (bins − 2·padding − 1)`, so a normalized
    /// intensity in `[0, 1]` maps to the bin range `[padding, bins−1−padding]`.
    spacing: f64,
    /// `VarianceForJointPDFSmoothing` for the discrete-Gaussian blur of the
    /// hard-binned histogram (see [`Self::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING`]).
    variance_for_joint_pdf_smoothing: f64,
}

impl JointHistogramMutualInformationMetric {
    /// ITK's `VarianceForJointPDFSmoothing` ctor default (SimpleITK's
    /// `SetMetricAsJointHistogramMutualInformation`'s `varianceForJointPDFSmoothing`
    /// default), for callers that want ITK/SimpleITK parity rather than a
    /// deliberately chosen value.
    pub const DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING: f64 = 1.5;

    /// Build the metric from a fixed and moving image, a histogram bin
    /// count, and the joint-PDF smoothing variance
    /// (`VarianceForJointPDFSmoothing`, e.g.
    /// [`Self::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING`] for ITK/SimpleITK
    /// parity). Fails if dimensions disagree, the moving direction matrix is
    /// singular, fewer than `MIN_BINS` (6) bins are requested, or either image
    /// is constant (MI is then undefined).
    pub fn new(
        fixed: &Image,
        moving: &Image,
        number_of_histogram_bins: usize,
        variance_for_joint_pdf_smoothing: f64,
    ) -> Result<Self> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        let fixed_samples = FixedSamples::from_image(fixed);
        let moving_image = MovingImage::from_image(moving)?;
        Self::from_samples(
            fixed_samples,
            moving_image,
            number_of_histogram_bins,
            variance_for_joint_pdf_smoothing,
        )
    }

    /// Build the metric from already-prepared fixed samples and moving image
    /// (e.g. from a caller applying its own sampling strategy, interpolator, or
    /// mask before handing off to the metric). Fails if fewer than
    /// `MIN_BINS` (6) bins are requested or either image is constant (MI is then
    /// undefined).
    pub fn from_samples(
        fixed: FixedSamples,
        moving: MovingImage,
        number_of_histogram_bins: usize,
        variance_for_joint_pdf_smoothing: f64,
    ) -> Result<Self> {
        if number_of_histogram_bins < MIN_BINS {
            return Err(RegistrationError::TooFewHistogramBins {
                bins: number_of_histogram_bins,
            });
        }

        let (fixed_min, fixed_max) = fixed.value_range();
        let (moving_min, moving_max) = moving.value_range();
        if fixed_max - fixed_min <= f64::EPSILON {
            return Err(RegistrationError::ConstantIntensity { which: "fixed" });
        }
        if moving_max - moving_min <= f64::EPSILON {
            return Err(RegistrationError::ConstantIntensity { which: "moving" });
        }

        let spacing = 1.0 / ((number_of_histogram_bins - 2 * PADDING - 1) as f64);

        Ok(Self {
            fixed,
            moving,
            num_bins: number_of_histogram_bins,
            fixed_true_min: fixed_min,
            fixed_true_max: fixed_max,
            moving_true_min: moving_min,
            moving_true_max: moving_max,
            spacing,
            variance_for_joint_pdf_smoothing,
        })
    }

    /// Number of fixed sample points.
    pub fn sample_count(&self) -> usize {
        self.fixed.len()
    }

    /// Build a physical-shift scale/learning-rate estimator for `transform`
    /// over this metric's fixed sample points (shared with the other metrics).
    pub fn physical_shift_scales(
        &self,
        transform: &dyn ParametricTransform,
    ) -> PhysicalShiftScales {
        self.fixed.physical_shift_scales(transform)
    }

    /// Normalized joint-PDF-point coordinate of a raw intensity `value` on the
    /// axis with true range `[true_min, true_max]`: `(value − true_min) /
    /// (true_max − true_min)`, ITK's `ComputeJointPDFPoint`. Always in `[0,1]`
    /// for `value` within range.
    fn normalize(value: f64, true_min: f64, true_max: f64) -> f64 {
        (value - true_min) / (true_max - true_min)
    }

    /// Nearest histogram bin (round-half-up, then `+padding`) for a normalized
    /// `[0,1]` coordinate, or `None` if it falls outside `[0, bins)` (should
    /// not happen for in-range values, but mirrors ITK's `IsInside` reject
    /// rather than a silent clamp).
    fn bin_index(&self, normalized: f64) -> Option<usize> {
        let idx = round_half_up(normalized / self.spacing) + PADDING as i64;
        if idx >= 0 && (idx as usize) < self.num_bins {
            Some(idx as usize)
        } else {
            None
        }
    }

    /// Bilinearly interpolate the `bins × bins` joint PDF at normalized point
    /// `(a, b)`.
    fn interp_joint(&self, jp: &[f64], a: f64, b: f64) -> f64 {
        let bins = self.num_bins;
        let last = (bins - 1) as f64;
        let ca = (a / self.spacing + PADDING as f64).clamp(0.0, last);
        let cb = (b / self.spacing + PADDING as f64).clamp(0.0, last);
        let fa0 = ca.floor() as usize;
        let fa1 = (fa0 + 1).min(bins - 1);
        let fa = ca - fa0 as f64;
        let fb0 = cb.floor() as usize;
        let fb1 = (fb0 + 1).min(bins - 1);
        let fb = cb - fb0 as f64;
        let v00 = jp[fa0 * bins + fb0];
        let v01 = jp[fa0 * bins + fb1];
        let v10 = jp[fa1 * bins + fb0];
        let v11 = jp[fa1 * bins + fb1];
        let v0 = v00 * (1.0 - fb) + v01 * fb;
        let v1 = v10 * (1.0 - fb) + v11 * fb;
        v0 * (1.0 - fa) + v1 * fa
    }

    /// Linearly interpolate a 1-D marginal (length `bins`) at normalized point
    /// `t`.
    fn interp_marginal(&self, marginal: &[f64], t: f64) -> f64 {
        let bins = self.num_bins;
        let last = (bins - 1) as f64;
        let c = (t / self.spacing + PADDING as f64).clamp(0.0, last);
        let i0 = c.floor() as usize;
        let i1 = (i0 + 1).min(bins - 1);
        let frac = c - i0 as f64;
        marginal[i0] * (1.0 - frac) + marginal[i1] * frac
    }

    /// Central-difference derivative of the (interpolated) joint PDF along the
    /// moving axis at `(a, b)`, `b`'s window clamped into `[spacing, 1.0]`
    /// exactly as ITK's `ComputeJointPDFDerivative`.
    fn joint_derivative_wrt_moving(&self, jp: &[f64], a: f64, b: f64) -> f64 {
        let offset = 0.5 * self.spacing;
        let eps = self.spacing;
        let mut left = b - offset;
        let mut right = b + offset;
        if left < eps {
            left = eps;
        }
        if right < eps {
            right = eps;
        }
        if left > 1.0 {
            left = 1.0;
        }
        if right > 1.0 {
            right = 1.0;
        }
        let delta = right - left;
        if delta > 0.0 {
            (self.interp_joint(jp, a, right) - self.interp_joint(jp, a, left)) / delta
        } else {
            0.0
        }
    }

    /// Central-difference derivative of the (interpolated) 1-D marginal at
    /// `t`, matching ITK's `ComputeMovingImageMarginalPDFDerivative`.
    fn marginal_derivative(&self, marginal: &[f64], t: f64) -> f64 {
        let offset = 0.5 * self.spacing;
        let eps = self.spacing;
        let mut left = t - offset;
        let mut right = t + offset;
        if left < eps {
            left = eps;
        }
        if right < eps {
            right = eps;
        }
        if left > 1.0 {
            left = 1.0;
        }
        if right > 1.0 {
            right = 1.0;
        }
        let delta = right - left;
        if delta > 0.0 {
            (self.interp_marginal(marginal, right) - self.interp_marginal(marginal, left)) / delta
        } else {
            0.0
        }
    }

    /// Build the smoothed, normalized joint PDF and its marginals from every
    /// fixed sample under `transform` (ITK's `ComputeJointPDFThreader` pass +
    /// `InitializeForIteration`'s smoothing/marginal step). Returns
    /// `(joint_pdf, fixed_marginal, moving_marginal, valid_points)`.
    fn compute_joint_pdf(
        &self,
        transform: &dyn ParametricTransform,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let dim = self.fixed.dim;
        let bins = self.num_bins;
        let n = self.fixed.len();

        let mut hist = vec![0.0f64; bins * bins];
        let mut valid = 0usize;

        for s in 0..n {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let fv = self.fixed.values[s];

            let mp = transform.transform_point(fp);
            let mv = match self.moving.value_and_physical_gradient(&mp) {
                Some((v, _)) => v,
                None => continue,
            };
            if mv < self.moving_true_min || mv > self.moving_true_max {
                continue;
            }

            let a = Self::normalize(fv, self.fixed_true_min, self.fixed_true_max);
            let b = Self::normalize(mv, self.moving_true_min, self.moving_true_max);
            let (Some(fi), Some(mi)) = (self.bin_index(a), self.bin_index(b)) else {
                continue;
            };
            hist[fi * bins + mi] += 1.0;
            valid += 1;
        }

        if valid == 0 {
            return (vec![0.0; bins * bins], vec![0.0; bins], vec![0.0; bins], 0);
        }

        let inv = 1.0 / valid as f64;
        for h in hist.iter_mut() {
            *h *= inv;
        }

        let kernel = discrete_gaussian_kernel(
            self.variance_for_joint_pdf_smoothing,
            SMOOTHING_MAX_ERROR,
            SMOOTHING_MAX_KERNEL_WIDTH,
        );
        let smoothed_once = convolve_axis(&hist, bins, bins, 0, &kernel);
        let jp = convolve_axis(&smoothed_once, bins, bins, 1, &kernel);

        let mut fixed_marginal = vec![0.0f64; bins];
        let mut moving_marginal = vec![0.0f64; bins];
        for f in 0..bins {
            for m in 0..bins {
                let v = jp[f * bins + m];
                fixed_marginal[f] += v;
                moving_marginal[m] += v;
            }
        }

        (jp, fixed_marginal, moving_marginal, valid)
    }

    /// `value = −MI` (in bits) from the smoothed joint PDF and marginals
    /// (ITK's `ComputeValue`).
    fn compute_value(
        bins: usize,
        jp: &[f64],
        fixed_marginal: &[f64],
        moving_marginal: &[f64],
    ) -> f64 {
        let eps = f64::EPSILON;
        let mut total = 0.0f64;
        for i in 0..bins {
            let px = fixed_marginal[i];
            for j in 0..bins {
                let py = moving_marginal[j];
                let denom = px * py;
                let pxy = jp[i * bins + j];
                if denom.abs() > eps && pxy / denom > eps {
                    total += pxy * (pxy / denom).ln();
                }
            }
        }
        -total / std::f64::consts::LN_2
    }

    /// Evaluate `value = −MI` and its parameter-derivative for `transform`.
    ///
    /// Two passes over the fixed samples, exactly as ITK's two threaders: the
    /// first (the private `compute_joint_pdf`) builds the
    /// smoothed joint PDF and marginals (no gradients needed — only sample
    /// intensities); the second walks the samples again, this time with the
    /// moving image's physical gradient, to accumulate each sample's local
    /// Viola–Wells score against the (now fixed) smoothed density field.
    pub fn evaluate(&self, transform: &dyn ParametricTransform) -> MetricValue {
        let bins = self.num_bins;
        let nparams = transform.number_of_parameters();
        let (jp, fixed_marginal, moving_marginal, valid) = self.compute_joint_pdf(transform);

        if valid == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }

        let value = Self::compute_value(bins, &jp, &fixed_marginal, &moving_marginal);

        // Value-validity threshold for the derivative pass: ITK's `constexpr
        // InternalComputationValueType eps{ 1.e-16 };` in
        // `GetValueAndDerivativeThreader::ProcessPoint` (distinct from the
        // `f64::EPSILON` used in `compute_value`, which mirrors ITK's separate
        // `NumericTraits<double>::epsilon()` use in `ComputeValue`).
        const DERIVATIVE_VALUE_EPS: f64 = 1.0e-16;

        let dim = self.fixed.dim;
        let n = self.fixed.len();
        let mut derivative = vec![0.0f64; nparams];

        for s in 0..n {
            let fp = &self.fixed.points[s * dim..(s + 1) * dim];
            let fv = self.fixed.values[s];

            let mp = transform.transform_point(fp);
            let (mv, grad_phys) = match self.moving.value_and_physical_gradient(&mp) {
                Some(vg) => vg,
                None => continue,
            };
            if mv < self.moving_true_min || mv > self.moving_true_max {
                continue;
            }

            let a = Self::normalize(fv, self.fixed_true_min, self.fixed_true_max);
            let b = Self::normalize(mv, self.moving_true_min, self.moving_true_max);

            let jp_val = self.interp_joint(&jp, a, b);
            let mm_val = self.interp_marginal(&moving_marginal, b);
            if jp_val <= DERIVATIVE_VALUE_EPS || mm_val <= DERIVATIVE_VALUE_EPS {
                continue;
            }

            let d_jpdf = self.joint_derivative_wrt_moving(&jp, a, b);
            let d_mm = self.marginal_derivative(&moving_marginal, b);
            let p_ratio = jp_val.ln() - mm_val.ln();
            let term1 = d_jpdf * p_ratio;
            let term2 = std::f64::consts::LN_2 * d_mm * jp_val / mm_val;
            // Sign vs ITK: ITK's v4 optimizers ADD the returned derivative
            // (metrics store the descent direction, −∇value); this crate's
            // optimizers SUBTRACT, so every metric stores the true gradient
            // +∇value instead — the same convention flip documented in
            // `mattes.rs`'s `n_factor` note. Hence the negation here relative
            // to ITK's literal `term2 − term1`.
            let scalingfactor = -(term2 - term1);

            let jac = transform.jacobian_wrt_parameters(fp);
            for (k, dk) in derivative.iter_mut().enumerate() {
                let mut inner = 0.0;
                for (d, &g) in grad_phys.iter().enumerate() {
                    inner += jac[d * nparams + k] * g;
                }
                *dk += scalingfactor * inner;
            }
        }

        let inv = 1.0 / valid as f64;
        for d in derivative.iter_mut() {
            *d *= inv;
        }

        MetricValue {
            value,
            derivative,
            valid_points: valid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::{Transform, TranslationTransform};

    /// A 2-D Gaussian blob of amplitude `amp` and width `sigma`, centred at
    /// `(cx, cy)` in physical (== index, unit spacing) coordinates, on a small
    /// constant pedestal so the background is not a single degenerate bin.
    fn gaussian(w: usize, h: usize, cx: f64, cy: f64, sigma: f64, amp: f64) -> Image {
        let mut v = vec![0.0f64; w * h];
        let s2 = 2.0 * sigma * sigma;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                v[y * w + x] = amp * (-(dx * dx + dy * dy) / s2).exp() + 0.05;
            }
        }
        Image::from_vec(&[w, h], v).unwrap()
    }

    #[test]
    fn discrete_gaussian_kernel_is_normalized_and_symmetric() {
        let kernel = discrete_gaussian_kernel(1.5, 0.01, 32);
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "kernel sum {sum}");
        let radius = kernel.len() / 2;
        for i in 0..=radius {
            assert!(
                (kernel[radius - i] - kernel[radius + i]).abs() < 1e-12,
                "kernel not symmetric at offset {i}"
            );
        }
    }

    #[test]
    fn larger_variance_measurably_smooths_the_joint_pdf() {
        // `variance_for_joint_pdf_smoothing` is now a caller-supplied
        // constructor parameter (previously hard-coded); confirm it actually
        // changes the smoothing behavior in the expected direction: a larger
        // discrete-Gaussian variance spreads the (initially sharply peaked,
        // hard-binned) histogram's mass out further, lowering its peak bin
        // value (the joint PDF sums to 1 either way, so spreading mass out
        // necessarily lowers the peak).
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let moving = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let identity = TranslationTransform::new(vec![0.0, 0.0]);

        let narrow = JointHistogramMutualInformationMetric::new(&fixed, &moving, 32, 0.5).unwrap();
        let wide = JointHistogramMutualInformationMetric::new(&fixed, &moving, 32, 6.0).unwrap();

        let (jp_narrow, _, _, _) = narrow.compute_joint_pdf(&identity);
        let (jp_wide, _, _, _) = wide.compute_joint_pdf(&identity);

        let peak_narrow = jp_narrow.iter().cloned().fold(0.0f64, f64::max);
        let peak_wide = jp_wide.iter().cloned().fold(0.0f64, f64::max);

        assert!(
            peak_wide < peak_narrow,
            "larger variance should spread the joint PDF's mass out, lowering its peak: \
             narrow (variance 0.5) peak {peak_narrow} vs wide (variance 6.0) peak {peak_wide}"
        );
    }

    #[test]
    fn too_few_bins_is_rejected() {
        const V: f64 =
            JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING;
        let a = gaussian(10, 10, 5.0, 5.0, 2.0, 1.0);
        assert!(matches!(
            JointHistogramMutualInformationMetric::new(&a, &a, 5, V),
            Err(RegistrationError::TooFewHistogramBins { bins: 5 })
        ));
        assert!(JointHistogramMutualInformationMetric::new(&a, &a, 6, V).is_ok());
    }

    #[test]
    fn constant_image_is_rejected() {
        const V: f64 =
            JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING;
        let flat = Image::from_vec(&[8, 8], vec![3.0; 64]).unwrap();
        let varied = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        assert!(matches!(
            JointHistogramMutualInformationMetric::new(&flat, &varied, 20, V),
            Err(RegistrationError::ConstantIntensity { which: "fixed" })
        ));
        assert!(matches!(
            JointHistogramMutualInformationMetric::new(&varied, &flat, 20, V),
            Err(RegistrationError::ConstantIntensity { which: "moving" })
        ));
    }

    #[test]
    fn identical_images_are_optimal_at_identity_with_near_zero_derivative() {
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let img = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let metric = JointHistogramMutualInformationMetric::new(
            &img,
            &img,
            32,
            JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING,
        )
        .unwrap();

        let at = |dx: f64, dy: f64| {
            metric
                .evaluate(&TranslationTransform::new(vec![dx, dy]))
                .value
        };
        let aligned = at(0.0, 0.0);
        let shifted = at(5.0, -4.0);
        assert!(
            aligned < shifted,
            "aligned {aligned} should be below shifted {shifted}"
        );

        // A radially symmetric blob's aggregate registration objective is flat
        // to first order at the identity shift, so the derivative there
        // should be near zero (only the hard-binning discretization keeps it
        // from being exactly zero).
        let identity = metric.evaluate(&TranslationTransform::new(vec![0.0, 0.0]));
        assert!(
            identity.derivative[0].abs() < 5e-2,
            "d/dtx at identity {}",
            identity.derivative[0]
        );
        assert!(
            identity.derivative[1].abs() < 5e-2,
            "d/dty at identity {}",
            identity.derivative[1]
        );
    }

    #[test]
    fn multi_modality_nonlinear_remap_is_optimal_at_zero_shift() {
        // Moving is a nonlinear (non-affine), invertible remap of the fixed
        // blob's intensity: squared then rescaled back into a comparable
        // range. This gives fixed/moving marginals genuinely different
        // shapes (unlike a simple contrast inversion), which mean squares
        // could never align (M != F pointwise) and which would also expose
        // the ITK marginal-array swap described in the module docs if it
        // were reproduced literally.
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let s2 = 2.0 * sigma * sigma;
        let mut mv = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 - 20.0, y as f64 - 20.0);
                let f = (-(dx * dx + dy * dy) / s2).exp() + 0.05;
                mv[y * w + x] = f * f;
            }
        }
        let moving = Image::from_vec(&[w, h], mv).unwrap();
        let metric = JointHistogramMutualInformationMetric::new(
            &fixed,
            &moving,
            32,
            JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING,
        )
        .unwrap();

        let at = |dx: f64, dy: f64| {
            metric
                .evaluate(&TranslationTransform::new(vec![dx, dy]))
                .value
        };
        let aligned = at(0.0, 0.0);
        for (dx, dy) in [(5.0, 0.0), (-4.0, 3.0), (0.0, -6.0)] {
            let shifted = at(dx, dy);
            assert!(
                aligned < shifted,
                "aligned {aligned} should be below shift ({dx},{dy}) = {shifted}"
            );
        }
    }

    #[test]
    fn derivative_matches_finite_difference_direction() {
        // Fixed and moving are the same blob; evaluate at a generic
        // translation (off pixel and bin boundaries) and compare the
        // analytic derivative's *direction* to a central finite difference
        // of the value.
        //
        // Unlike Mattes (continuous cubic B-spline Parzen window on both
        // axes, so `value` is continuously differentiable and the analytic
        // derivative matches a finite difference to within ~1e-3 at
        // `h=1e-3`), this metric's histogram is *hard*-binned: a sample's
        // bin membership is a step function of the transform parameters, so
        // `value` is a piecewise-constant step function of the transform
        // parameters in the classical sense, and the analytic formula ported
        // here (`itkJointHistogramMutualInformationGetValueAndDerivativeThreader
        // ::ProcessPoint`) is deliberately an *approximate* Viola–Wells
        // "frozen density" gradient — it differentiates the smoothed density
        // surface at each sample's own location, holding that density fixed,
        // and explicitly does not account for samples re-binning into
        // different histogram cells as the transform changes (that
        // "rebinning" contribution is exactly what a literal finite
        // difference of `value` also captures, since it rebuilds the
        // histogram from scratch at every perturbed parameter).
        //
        // This was verified empirically, not assumed: sweeping the bin
        // count from 10 to 256 at a fixed step (`h=0.02`) shows the finite
        // difference growing roughly 20x (0.015 → 0.33, more bins ⇒ finer
        // rebinning-boundary crossings ⇒ bigger rebinning contribution)
        // while the analytic derivative stays essentially flat (~0.0005 to
        // ~0.0013, since the frozen-density surface's shape is set by
        // `VarianceForJointPDFSmoothing`, not bin count) — the gap widens
        // with resolution rather than narrowing, which is the signature of
        // an intentionally-excluded rebinning term, not a transcription bug.
        // ITK's own unit test for this metric
        // (`itkJointHistogramMutualInformationImageToImageMetricv4Test.cxx`)
        // does not check the derivative against a finite difference either.
        //
        // What *does* hold, and is what actually matters for gradient
        // descent (see `gradient_descent_recovers_a_translated_blob` below,
        // which is the stronger, practical version of this check): the
        // analytic derivative's *sign* — i.e. which way to step to reduce
        // `value` — agrees with the finite difference's sign.
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let moving = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let metric = JointHistogramMutualInformationMetric::new(
            &fixed,
            &moving,
            32,
            JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING,
        )
        .unwrap();

        let p0 = [1.3f64, -0.7];
        let eval = |p: &[f64]| metric.evaluate(&TranslationTransform::new(p.to_vec()));
        let analytic = eval(&p0).derivative;

        let step = 0.02;
        for k in 0..2 {
            let mut pp = p0;
            pp[k] += step;
            let mut pm = p0;
            pm[k] -= step;
            let fd = (eval(&pp).value - eval(&pm).value) / (2.0 * step);
            assert!(
                fd.signum() == analytic[k].signum() && analytic[k].abs() > 1e-6,
                "param {k}: fd {fd} and analytic {} should agree in sign and be non-negligible",
                analytic[k]
            );
        }
    }

    #[test]
    fn frozen_density_identity_matches_finite_difference() {
        // Pins every piece of machinery the analytic derivative rides on —
        // the joint-PDF field derivative, the marginal derivative, the
        // bilinear interpolators, and the sign — against a finite difference
        // of a closed-form potential. This checks the *ln2-free* half of
        // ITK's formula: see the module docs' `ln2` parity note for why the
        // full, literal, ln2-laden `term2` has no exact potential at all (a
        // Clairaut/mixed-partials argument, not a numerical gap) — that is
        // the reason this test targets `term1 − term2_no_ln2` rather than
        // `evaluate()`'s literal accumulated derivative (see
        // `evaluate_matches_an_independent_reimplementation_of_the_literal_formula`
        // below for a check of the literal, ln2-laden formula instead).
        //
        // For one sample with fixed (frozen, non-moving) bin coordinate `a`
        // and moving bin coordinate `b`, define
        //
        //   phi(a, b) = J(a,b)·(ln J(a,b) − ln mm(b)) − J(a,b)
        //
        // where `J`/`mm` are the joint-PDF/moving-marginal FIELDS frozen at
        // `p0` (never rebuilt as `b` is perturbed below — contrast with
        // `derivative_matches_finite_difference_direction`, which finite-
        // differences `evaluate().value` and therefore rebuilds the
        // histogram, capturing a rebinning effect this identity deliberately
        // excludes). By the product rule,
        //
        //   dphi/db = dJ·(ln J − ln mm) + J·(dJ/J − dmm/mm) − dJ
        //           = dJ·(ln J − ln mm) − J·dmm/mm
        //           = term1 − term2_no_ln2
        //
        // exactly (`term1 = dJPDF·pRatio`, `term2_no_ln2 = dMmPDF·J/mm`,
        // i.e. `term2` WITHOUT the `ln2` factor) — no constant, no
        // approximation.
        //
        // Point choice matters: `joint_derivative_wrt_moving`/
        // `marginal_derivative` are central differences over a FIXED
        // half-bin-width window (`±0.5·spacing`), and the bilinear PDF-field
        // interpolators have slope KINKS at integer bin-grid vertices. Pick
        // `a`/`b` at a bin-grid CELL MIDPOINT (array coordinate `x.5`, not an
        // integer): the coarse window's `±0.5·spacing` offsets then land
        // EXACTLY on the cell's two flanking grid vertices, and since
        // `J(a,·)` and `mm(·)` are exactly linear in `b` across that whole
        // cell (no kink strictly inside it), the coarse formula computes the
        // EXACT analytic slope there, not merely a first-order approximation
        // of it — so a fine central difference of `phi(b)` (step kept well
        // inside the cell's `0.5·spacing` half-width, so it never crosses
        // into a neighboring cell) should agree with `term1 −
        // term2_no_ln2` to floating-point precision, not just to some
        // FD-truncation-limited tolerance. Verified: at step 1e-4 (cell
        // half-width ≈0.0185 for the 32 bins used here — ~185x margin), the
        // measured relative error was ≈2e-11; the assertion below uses a
        // 1e-6 tolerance, ~5 orders of magnitude of margin over that
        // measurement.
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let moving = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let metric = JointHistogramMutualInformationMetric::new(
            &fixed,
            &moving,
            32,
            JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING,
        )
        .unwrap();
        let base = TranslationTransform::new(vec![1.3, -0.7]);
        let (jp, _fixed_marginal, moving_marginal, _valid) = metric.compute_joint_pdf(&base);

        // Cell midpoint (array coordinate 16.5) on both axes.
        let a = (16.5 - PADDING as f64) * metric.spacing;
        let b = (16.5 - PADDING as f64) * metric.spacing;

        let phi = |bb: f64| -> f64 {
            let jp_val = metric.interp_joint(&jp, a, bb);
            let mm_val = metric.interp_marginal(&moving_marginal, bb);
            jp_val * (jp_val.ln() - mm_val.ln()) - jp_val
        };

        let d_jpdf = metric.joint_derivative_wrt_moving(&jp, a, b);
        let d_mm = metric.marginal_derivative(&moving_marginal, b);
        let jp_val = metric.interp_joint(&jp, a, b);
        let mm_val = metric.interp_marginal(&moving_marginal, b);
        let p_ratio = jp_val.ln() - mm_val.ln();
        let term1 = d_jpdf * p_ratio;
        let term2_no_ln2 = d_mm * jp_val / mm_val;
        let coarse = term1 - term2_no_ln2;

        let step = 1e-4;
        let fine_fd = (phi(b + step) - phi(b - step)) / (2.0 * step);

        assert!(
            (coarse - fine_fd).abs() < 1e-6 * coarse.abs(),
            "coarse (term1 - term2_no_ln2) {coarse} vs fine FD of phi {fine_fd} \
             (relative error {})",
            (coarse - fine_fd).abs() / coarse.abs()
        );
    }

    #[test]
    fn evaluate_matches_an_independent_reimplementation_of_the_literal_formula() {
        // Complements `frozen_density_identity_matches_finite_difference`
        // above, which pins a single machinery piece (one sample, no
        // aggregation, no transform chain rule) but stops short of the full,
        // literal, ln2-laden formula. This test instead independently
        // re-implements `evaluate()`'s FULL accumulation — ITK's literal
        // `scalingfactor = term2 − term1` (with `ln2`), the moving-image
        // gradient, the transform Jacobian contraction, this crate's sign
        // flip, and the `1/valid_points` normalization — in a hand-written
        // loop that does not call `evaluate()` for the derivative
        // computation, only for the value being checked against. It exists
        // to catch accumulation/normalization/valid-point/sign regressions
        // in `evaluate()` that a single-sample identity test cannot see; it
        // is not a test of whether the formula is an exact gradient of
        // anything (it isn't — see the module docs' `ln2` parity note).
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let moving = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let metric = JointHistogramMutualInformationMetric::new(
            &fixed,
            &moving,
            32,
            JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING,
        )
        .unwrap();

        let transform = TranslationTransform::new(vec![1.3, -0.7]);
        let expected = metric.evaluate(&transform).derivative;

        let (jp, _fixed_marginal, moving_marginal, valid) = metric.compute_joint_pdf(&transform);
        let nparams = transform.number_of_parameters();
        let dim = metric.fixed.dim;
        let n = metric.fixed.len();
        let mut derivative = vec![0.0f64; nparams];
        const DERIVATIVE_VALUE_EPS: f64 = 1.0e-16;

        for s in 0..n {
            let fp = &metric.fixed.points[s * dim..(s + 1) * dim];
            let fv = metric.fixed.values[s];
            let mp = transform.transform_point(fp);
            let (mv, grad_phys) = match metric.moving.value_and_physical_gradient(&mp) {
                Some(vg) => vg,
                None => continue,
            };
            if mv < metric.moving_true_min || mv > metric.moving_true_max {
                continue;
            }
            let a = JointHistogramMutualInformationMetric::normalize(
                fv,
                metric.fixed_true_min,
                metric.fixed_true_max,
            );
            let b = JointHistogramMutualInformationMetric::normalize(
                mv,
                metric.moving_true_min,
                metric.moving_true_max,
            );
            let jp_val = metric.interp_joint(&jp, a, b);
            let mm_val = metric.interp_marginal(&moving_marginal, b);
            if jp_val <= DERIVATIVE_VALUE_EPS || mm_val <= DERIVATIVE_VALUE_EPS {
                continue;
            }
            let d_jpdf = metric.joint_derivative_wrt_moving(&jp, a, b);
            let d_mm = metric.marginal_derivative(&moving_marginal, b);
            let p_ratio = jp_val.ln() - mm_val.ln();
            let term1 = d_jpdf * p_ratio;
            let term2 = std::f64::consts::LN_2 * d_mm * jp_val / mm_val;
            let scalingfactor = -(term2 - term1); // crate's +grad convention, matching evaluate()

            let jac = transform.jacobian_wrt_parameters(fp);
            for (k, dk) in derivative.iter_mut().enumerate() {
                let mut inner = 0.0;
                for (d, &gr) in grad_phys.iter().enumerate() {
                    inner += jac[d * nparams + k] * gr;
                }
                *dk += scalingfactor * inner;
            }
        }
        for d in derivative.iter_mut() {
            *d /= valid as f64;
        }

        for k in 0..nparams {
            assert!(
                (derivative[k] - expected[k]).abs() < 1e-10 * expected[k].abs().max(1e-10),
                "param {k}: independent reimplementation {} vs evaluate() {}",
                derivative[k],
                expected[k]
            );
        }
    }

    #[test]
    fn gradient_descent_recovers_a_translated_blob() {
        let (w, h, sigma) = (40usize, 40usize, 6.0);
        let fixed = gaussian(w, h, 20.0, 20.0, sigma, 1.0);
        let moving = gaussian(w, h, 23.0, 20.0, sigma, 1.0); // true shift (3, 0)
        let metric = JointHistogramMutualInformationMetric::new(
            &fixed,
            &moving,
            32,
            JointHistogramMutualInformationMetric::DEFAULT_VARIANCE_FOR_JOINT_PDF_SMOOTHING,
        )
        .unwrap();

        // Learning rate tuned empirically (not the crate's usual estimated
        // scales — see the module docs' local-support note): the analytic
        // derivative here is the Viola-Wells frozen-density gradient, whose
        // magnitude is set by the smoothing bandwidth rather than by the
        // metric's value scale, so it is far smaller than e.g. Mattes' at a
        // comparable image/blob size. `lr=20, 500 iterations` converges
        // cleanly to within 0.01px on this test's fixed geometry.
        let opt = crate::optimizer::GradientDescentOptimizer::new(20.0, 500);
        let result = opt.optimize(vec![0.0, 0.0], |p| {
            let t = TranslationTransform::new(p.to_vec());
            let r = metric.evaluate(&t);
            (r.value, r.derivative)
        });

        assert!(
            (result.parameters[0] - 3.0).abs() < 0.5,
            "recovered dx {} (expected ~3.0)",
            result.parameters[0]
        );
        assert!(
            result.parameters[1].abs() < 0.5,
            "recovered dy {} (expected ~0.0)",
            result.parameters[1]
        );
    }
}
