//! The **demons** (intensity-difference-force) similarity metric
//! (`itk::DemonsImageToImageMetricv4`, formula from `itkDemonsRegistrationFunction`).
//!
//! Unlike mean squares or Mattes MI, Demons is not a smooth objective whose
//! derivative is optimized by descent — its "derivative" is **Thirion's demons
//! force**, a self-normalizing update that approximates one Newton step of the
//! optical-flow equation `∇M · u ≈ F − M` rather than the analytic gradient of
//! any scalar value. ITK's v4 framework repurposes the same
//! `GetValueAndDerivative` contract to carry it so the existing v4 optimizers
//! can drive it; see [`DemonsImageToImageMetricv4GetValueAndDerivativeThreader::ProcessPoint`]
//! for the source this is ported from.
//!
//! ```text
//! value          = (F − M)²
//! force          = (F − M) · ∇ / ( |∇|² + (F − M)² / normalizer )
//! normalizer     = mean square spacing of the gradient-source image
//! ```
//!
//! `∇` is the **fixed**-image gradient by default (ITK's
//! `GRADIENT_SOURCE_FIXED`, `DemonsImageToImageMetricv4`'s constructor default
//! and the only source [`sitkImageRegistrationMethod::SetMetricAsDemons`]
//! exposes), sampled at the *fixed* sample point — never at the point the
//! transform maps to. `normalizer` compensates for a mismatch in units between
//! the `(F − M)²` term (intensity²) and `|∇|²` (intensity²/mm²) when the fixed
//! image does not have unit spacing; it is the mean square fixed-image spacing,
//! computed once and fixed for the metric's lifetime.
//!
//! Two thresholds gate the force to zero for a sample (but never exclude it
//! from the value sum or the valid-point count — see
//! [Threshold semantics](#threshold-semantics) below):
//!
//! * `intensity_difference_threshold` (ITK default `0.001`, the only knob
//!   [`SetMetricAsDemons`] exposes): below this, `F` and `M` are considered
//!   equal and contribute no force.
//! * `denominator_threshold` (ITK: fixed at `1e-9` in the constructor, with
//!   only a `Get`, no `Set`, in the public API — so this port hardcodes it
//!   too): below this, the denominator is considered degenerate (flat regions
//!   where both the intensity difference and the gradient vanish).
//!
//! ## Sign convention vs ITK
//!
//! ITK's `ProcessPoint` computes `speedValue = fixedImageValue −
//! movingImageValue` (`F − M`) and returns `force = speedValue · gradient /
//! denominator` for its optimizers to **add**. This crate's optimizers
//! **subtract** the returned derivative (`p −= lr · derivative`), the same
//! convention [`mattes`](crate::mattes) and [`metric`](crate::metric) already
//! document and correct for: [`metric::CpuBackend::mean_squares`] computes its
//! `diff` as `M − F`, the mirror image of ITK's own `F − M` mean-squares
//! difference, so that its returned derivative is `+∇value` and subtracting it
//! descends. This module does the same swap — `speed = M − F` here, vs ITK's
//! `F − M` — so `evaluate`'s returned force, when subtracted, moves the
//! transform in the direction ITK's own optimizer would reach by *adding* its
//! `F − M`-based force. This was verified by hand: at a point left of a 1-D
//! ramp's minimum, ITK's `(F−M)·∇/denom` is positive and their optimizer adds
//! it (moving toward the minimum); this crate's `(M−F)·∇/denom` is the
//! negation, and subtracting it produces the identical step.
//!
//! ## Threshold semantics
//!
//! Note precisely what the two thresholds do, since it differs from a plain
//! reading of "threshold ... skips a sample": in ITK's `ProcessPoint` the
//! value `(F − M)²` is assigned *before* either threshold is tested, and
//! `ProcessPoint` unconditionally `return`s `true` — so a thresholded sample
//! still counts toward the value sum and the valid-point count exactly like
//! any other sample; only its **derivative** contribution is forced to zero.
//! [`MetricValue::valid_points`] here retains the meaning it has everywhere
//! else in this crate — the count of fixed samples that mapped inside the
//! moving buffer (and, for this local-support-only metric, landed inside the
//! transform's local-support domain) — and is **not** reduced by either
//! threshold. The tests below demonstrate the thresholds by their actual,
//! ITK-faithful effect: a guarded sample's contribution to the accumulated
//! derivative drops to zero while `valid_points` is unchanged.
//!
//! ## Local support only
//!
//! `DemonsImageToImageMetricv4::Initialize` throws unless
//! `GetTransformCategory() == DisplacementField`
//! (`itkDemonsImageToImageMetricv4.hxx`: `"The moving transform must be a
//! displacement field transform"`) — it does not fall back to a dense/global
//! path the way [`MattesMutualInformationMetric`](crate::mattes) does for a
//! B-spline. This port matches that with [`DemonsMetric::check_transform`],
//! which a caller checks **once** per optimization — the (metric, transform)
//! pairing is fixed for the whole run, so it is not re-verified on every one
//! of the tens of thousands of [`evaluate`](DemonsMetric::evaluate) calls a
//! typical optimization makes. [`evaluate`](DemonsMetric::evaluate) itself
//! `assert!`s the identical condition, calling `check_transform` as the single
//! source of truth, and panics if it does not hold (see its `# Panics`
//! section). Otherwise `evaluate` always takes the per-pixel
//! [`local_support_block`](crate::metric) path — there is no dense branch in
//! this module at all. ITK's `ProcessPoint` does not multiply by a Jacobian
//! either; it writes `speed · gradient[p] / denominator` directly into each of
//! the `NumberOfLocalParameters` local slots, which is only correct when that
//! local Jacobian is the identity — true of every displacement-field local
//! block (`DisplacementFieldTransform::sparse_jacobian_wrt_parameters` returns
//! unit columns, and a displacement field is the only `has_local_support`
//! transform this crate has). This port does the same: `local_support_block` is
//! called only for its parameter-block `offset`, and the gradient components
//! are written in directly, unprojected.
//!
//! [`DemonsImageToImageMetricv4GetValueAndDerivativeThreader::ProcessPoint`]:
//! <https://github.com/InsightSoftwareConsortium/ITK/blob/master/Modules/Registration/Metricsv4/include/itkDemonsImageToImageMetricv4GetValueAndDerivativeThreader.hxx>
//! [`sitkImageRegistrationMethod::SetMetricAsDemons`]:
//! <https://github.com/SimpleITK/SimpleITK/blob/main/Code/Registration/include/sitkImageRegistrationMethod.h>
//! [`SetMetricAsDemons`]:
//! <https://github.com/SimpleITK/SimpleITK/blob/main/Code/Registration/include/sitkImageRegistrationMethod.h>
//! [`metric::CpuBackend::mean_squares`]: crate::metric::CpuBackend

use sitk_core::Image;
use sitk_transform::ParametricTransform;

use crate::error::{RegistrationError, Result};
use crate::metric::{FixedSamples, MetricValue, MovingImage, local_support_block};
use crate::scales::{ScalesEstimator, ScalesEstimatorKind};

/// Threshold below which the denominator `|∇|² + (F−M)²/normalizer` is treated
/// as degenerate. ITK: `m_DenominatorThreshold`, fixed at this value in the
/// constructor with no public setter (only `GetDenominatorThreshold`).
const DENOMINATOR_THRESHOLD: f64 = 1e-9;

/// The Demons (intensity-difference-force) image-to-image metric. See the
/// [module docs](self) for the value/force formulas and the local-support-only
/// restriction. Holds the precomputed fixed samples, a fixed-image gradient
/// sampler (the default gradient source), and the moving image.
pub struct DemonsMetric {
    fixed: FixedSamples,
    /// Linear-interpolation sampler over the **fixed** image, used only to
    /// evaluate its gradient at each fixed sample point — ITK's default
    /// `GRADIENT_SOURCE_FIXED`. [`MovingImage`] is a generic value+gradient
    /// sampler despite its name; nothing here restricts it to the image being
    /// transformed.
    fixed_gradient: MovingImage,
    moving: MovingImage,
    /// `itkGetConstMacro`/`itkSetMacro(IntensityDifferenceThreshold)`. Below
    /// this, `F` and `M` are considered equal and contribute no force.
    intensity_difference_threshold: f64,
    /// Mean square fixed-image spacing (`itk::DemonsImageToImageMetricv4`'s
    /// `m_Normalizer`), computed once from the gradient-source image.
    normalizer: f64,
}

impl DemonsMetric {
    /// Build the metric from a **pre-built** sample set and moving image —
    /// the entry point a registration driver uses once it has applied its
    /// own metric sampling strategy to build `fixed` (see [`FixedSamples`]).
    /// [`new`](Self::new) is the convenience wrapper that builds `fixed` from
    /// a raw image with full (dense) sampling.
    ///
    /// `fixed_image` — the same image `fixed` was sampled from — is **also**
    /// required, in addition to `fixed`, for two things this metric owns and
    /// [`new`](Self::new) already derives from it rather than accepting from
    /// a caller: the **fixed-image gradient sampler** (ITK's default
    /// `GRADIENT_SOURCE_FIXED`) needs a dense interpolator over the *whole*
    /// fixed image, which [`FixedSamples`] — a flat, possibly-sparse list of
    /// sampled points/values — cannot supply; and **`normalizer`** is the
    /// mean square of the fixed image's per-axis spacing (see the
    /// [module docs](self)), a scalar this metric derives once from the raw
    /// image geometry so a second caller cannot get it wrong. Under a
    /// reduced (regular/random) sampling strategy, `fixed` is a strict subset
    /// of the fixed image's voxel grid while `fixed_gradient` stays dense
    /// over the whole image — exactly ITK's sparse-threader behaviour, which
    /// re-samples the dense fixed gradient image at each sparse point rather
    /// than reducing the gradient sampler itself.
    ///
    /// `fixed_image` and `fixed` are debug-asserted to agree in dimension.
    /// `moving`'s dimension cannot be cross-checked here: `MovingImage` does
    /// not currently expose its dimension outside `metric.rs` in this crate,
    /// so a mismatched `moving` will surface as an out-of-bounds panic or a
    /// nonsensical mapped point downstream rather than a
    /// `DimensionMismatch` error.
    ///
    /// Fails if `fixed_image`'s direction matrix is singular.
    pub fn from_samples(
        fixed_image: &Image,
        fixed: FixedSamples,
        moving: MovingImage,
        intensity_difference_threshold: f64,
    ) -> Result<Self> {
        debug_assert_eq!(
            fixed_image.dimension(),
            fixed.dim,
            "from_samples: fixed_image and fixed samples have different dimension"
        );
        if fixed.dim != moving.dim() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dim,
                moving: moving.dim(),
            });
        }
        let dim = fixed_image.dimension();
        let normalizer = fixed_image
            .spacing()
            .iter()
            .take(dim)
            .map(|s| s * s)
            .sum::<f64>()
            / dim as f64;

        Ok(Self {
            fixed,
            fixed_gradient: MovingImage::from_image(fixed_image)?,
            moving,
            intensity_difference_threshold,
            normalizer,
        })
    }

    /// Build the metric from a fixed and moving image and the intensity
    /// difference threshold below which two intensities are considered equal
    /// (ITK/SimpleITK default `0.001`, `SetMetricAsDemons`'s only parameter).
    /// Fails if dimensions disagree or either image's direction matrix is
    /// singular. Delegates to [`from_samples`](Self::from_samples) with full
    /// (dense) sampling.
    pub fn new(fixed: &Image, moving: &Image, intensity_difference_threshold: f64) -> Result<Self> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        Self::from_samples(
            fixed,
            FixedSamples::from_image(fixed)?,
            MovingImage::from_image(moving)?,
            intensity_difference_threshold,
        )
    }

    /// Number of fixed sample points.
    pub fn sample_count(&self) -> usize {
        self.fixed.len()
    }

    /// The configured intensity-difference threshold (`GetIntensityDifferenceThreshold`).
    pub fn intensity_difference_threshold(&self) -> f64 {
        self.intensity_difference_threshold
    }

    /// The denominator threshold (`GetDenominatorThreshold`). Fixed at `1e-9`;
    /// ITK exposes no setter for it, and neither does this port.
    pub fn denominator_threshold(&self) -> f64 {
        DENOMINATOR_THRESHOLD
    }

    /// Build a scale/learning-rate estimator of `kind` for `transform` over
    /// this metric's virtual domain (shared with the mean-squares and Mattes
    /// metrics).
    pub fn scales_estimator(
        &self,
        transform: &dyn ParametricTransform,
        kind: ScalesEstimatorKind,
    ) -> ScalesEstimator {
        self.fixed.scales_estimator(transform, &self.moving, kind)
    }

    /// Check that `transform` can be used with this metric: Demons requires a
    /// transform with local support (a displacement field). Call this once
    /// per optimization, before the loop that repeatedly calls
    /// [`evaluate`](Self::evaluate) — the (metric, transform) pairing is
    /// fixed for the whole run, so it is a precondition to verify up front,
    /// not on every evaluation.
    ///
    /// Matches `DemonsImageToImageMetricv4::Initialize`
    /// (`itkDemonsImageToImageMetricv4.hxx`), which throws `"The moving
    /// transform must be a displacement field transform"` when
    /// `GetTransformCategory() != DisplacementField` — unlike the Mattes MI
    /// metric, Demons does not fall back to a dense/global path.
    pub fn check_transform(&self, transform: &dyn ParametricTransform) -> Result<()> {
        if transform.has_local_support() {
            Ok(())
        } else {
            Err(RegistrationError::RequiresLocalSupportTransform { metric: "Demons" })
        }
    }

    /// Evaluate `value = (F − M)²` and the demons force for `transform`.
    ///
    /// See the [module docs](self) for the sign-convention and
    /// threshold-semantics notes.
    ///
    /// # Panics
    ///
    /// Panics unless `transform` has local support — call
    /// [`check_transform`](Self::check_transform) once per optimization
    /// beforehand rather than relying on this. Matches
    /// `DemonsImageToImageMetricv4::Initialize`
    /// (`itkDemonsImageToImageMetricv4.hxx`), which throws for the identical
    /// condition once, at initialization, not on every per-point evaluation.
    /// The metric value alone at `transform`: `mean((M − F)²)` over the samples
    /// that are geometrically valid. Skips the force (derivative) arithmetic,
    /// but still runs the two validity checks that decide which samples count —
    /// the fixed-image gradient must exist at the fixed point, and a local
    /// parameter block must govern it — so the sample set is identical to
    /// [`evaluate`](Self::evaluate)'s.
    ///
    /// Neither of the two thresholds appears here: both gate the *force* only,
    /// never the value or the valid-point count (see the module docs,
    /// [Threshold semantics](self#threshold-semantics)).
    ///
    /// # Panics
    ///
    /// As [`evaluate`](Self::evaluate): this metric is local-support-only.
    pub fn value(&self, transform: &dyn ParametricTransform) -> f64 {
        assert!(
            self.check_transform(transform).is_ok(),
            "DemonsMetric requires a transform with local support; call check_transform first"
        );

        let n = self.fixed.len();
        let mut value_sum = 0.0f64;
        let mut valid = 0usize;

        let mut scratch = self.fixed.scratch();
        for s in 0..n {
            let fp = self.fixed.point(s, &mut scratch);
            let mp = transform.transform_point(fp);
            let mv = match self.moving.value_at(&mp) {
                Some(v) => v,
                None => continue,
            };
            if self
                .fixed_gradient
                .value_and_physical_gradient(fp)
                .is_none()
            {
                continue;
            }
            if local_support_block(transform, fp).is_none() {
                continue;
            }
            let speed = mv - self.fixed.value(s);
            value_sum += speed * speed;
            valid += 1;
        }

        if valid == 0 {
            return f64::MAX;
        }
        value_sum / valid as f64
    }

    pub fn evaluate(&self, transform: &dyn ParametricTransform) -> MetricValue {
        assert!(
            self.check_transform(transform).is_ok(),
            "DemonsMetric requires a transform with local support; call check_transform first"
        );

        let nparams = transform.number_of_parameters();
        let num_local = transform.number_of_local_parameters();
        let n = self.fixed.len();

        let mut value_sum = 0.0;
        let mut derivative = vec![0.0; nparams];
        let mut valid = 0usize;

        let mut scratch = self.fixed.scratch();
        for s in 0..n {
            let fp = self.fixed.point(s, &mut scratch);
            let fv = self.fixed.value(s);

            let mp = transform.transform_point(fp);
            let mv = match self.moving.value_at(&mp) {
                Some(v) => v,
                None => continue, // maps outside the moving buffer
            };
            // Fixed-image gradient at the *fixed* point (never the mapped
            // point) — ITK's default GRADIENT_SOURCE_FIXED.
            let grad = match self.fixed_gradient.value_and_physical_gradient(fp) {
                Some((_, g)) => g,
                None => continue,
            };
            // The parameter block owning this fixed sample (indexed by the
            // *fixed*/virtual point, matching ITK's
            // ComputeParameterOffsetFromVirtualIndex); its local Jacobian is
            // ignored per the module docs' local-support note.
            let (offset, _local_jac) = match local_support_block(transform, fp) {
                Some(oj) => oj,
                None => continue, // no local region governs this point
            };

            // Geometrically valid from here on: counts toward value and
            // valid_points regardless of the thresholds below (see module
            // docs, "Threshold semantics").
            //
            // speed = M − F: the mirror of ITK's F − M, so the returned force
            // is the negation of ITK's (see module docs, "Sign convention").
            let speed = mv - fv;
            let sqr_speed = speed * speed;
            value_sum += sqr_speed;
            valid += 1;

            let grad_sq: f64 = grad.iter().map(|g| g * g).sum();
            let denom = sqr_speed / self.normalizer + grad_sq;

            if speed.abs() < self.intensity_difference_threshold || denom < DENOMINATOR_THRESHOLD {
                continue; // force forced to zero; value already counted above.
            }

            for mu in 0..num_local {
                derivative[offset + mu] += speed * grad[mu] / denom;
            }
        }

        if valid == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }
        let inv = 1.0 / valid as f64;
        MetricValue {
            value: value_sum * inv,
            derivative: derivative.iter().map(|d| d * inv).collect(),
            valid_points: valid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::GradientDescentOptimizer;
    use sitk_transform::{DisplacementFieldTransform, TranslationTransform};

    /// A 2-D Gaussian blob of amplitude `amp` and width `sigma`, centred at
    /// `(cx, cy)` in physical (== index, unit spacing) coordinates.
    fn gaussian(w: usize, h: usize, cx: f64, cy: f64, sigma: f64, amp: f64) -> Image {
        let mut v = vec![0.0f64; w * h];
        let s2 = 2.0 * sigma * sigma;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                v[y * w + x] = amp * (-(dx * dx + dy * dy) / s2).exp();
            }
        }
        Image::from_vec(&[w, h], v).unwrap()
    }

    /// A zero (identity) displacement field over the same grid as `image`.
    fn zero_field(image: &Image) -> DisplacementFieldTransform {
        DisplacementFieldTransform::new(
            image.dimension(),
            image.size(),
            image.origin(),
            image.spacing(),
            image.direction(),
        )
        .unwrap()
    }

    #[test]
    fn identical_images_at_identity_have_zero_value_and_derivative() {
        let img = gaussian(16, 16, 8.0, 8.0, 3.0, 1.0);
        let metric = DemonsMetric::new(&img, &img, 0.001).unwrap();
        let field = zero_field(&img);

        let r = metric.evaluate(&field);
        assert!(r.value.abs() < 1e-12, "value {}", r.value);
        assert!(
            r.derivative.iter().all(|&d| d == 0.0),
            "expected all-zero derivative, got a nonzero entry"
        );
        assert_eq!(r.valid_points, 16 * 16);
    }

    #[test]
    fn regular_sampling_reaches_the_metric_and_forces_only_the_sampled_blocks() {
        use crate::metric::SamplingStrategy;

        // A quarter-density regular sample set: stride ceil(1/0.25) = 4 over the
        // 256 voxels in scan-line order, so 64 samples remain. ITK's sparse
        // threader evaluates the same per-point force at exactly the sampled
        // positions, and a displacement field's parameter block is per-pixel, so
        // only the sampled pixels' blocks may receive a nonzero force.
        let fixed = gaussian(16, 16, 8.0, 8.0, 3.0, 1.0);
        let moving = gaussian(16, 16, 9.0, 8.0, 3.0, 1.0);
        let samples =
            FixedSamples::from_image_with(&fixed, SamplingStrategy::Regular, 0.25, 0, None)
                .unwrap();
        assert_eq!(samples.len(), 64);

        let metric = DemonsMetric::from_samples(
            &fixed,
            samples,
            MovingImage::from_image(&moving).unwrap(),
            0.001,
        )
        .unwrap();
        assert_eq!(metric.sample_count(), 64);

        let field = zero_field(&fixed);
        let r = metric.evaluate(&field);
        assert_eq!(r.valid_points, 64);
        assert!(
            r.derivative.iter().any(|&d| d != 0.0),
            "sparse sampling produced an all-zero force"
        );

        // A 2-D field's parameters are [dx, dy] per pixel in scan-line order, so
        // flat voxel `f` owns parameters `2f` and `2f + 1`. Regular sampling
        // takes every 4th flat voxel; every other pixel's block must stay zero.
        for f in 0..256 {
            if f % 4 == 0 {
                continue;
            }
            assert_eq!(
                (r.derivative[2 * f], r.derivative[2 * f + 1]),
                (0.0, 0.0),
                "unsampled voxel {f} received a force"
            );
        }
    }

    #[test]
    fn from_samples_rejects_a_moving_image_of_a_different_dimension() {
        let fixed = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let moving_3d = Image::from_vec(&[4, 4, 4], vec![1.0f64; 64]).unwrap();
        let result = DemonsMetric::from_samples(
            &fixed,
            FixedSamples::from_image(&fixed).unwrap(),
            MovingImage::from_image(&moving_3d).unwrap(),
            0.001,
        );
        let Err(err) = result else {
            panic!("a 3-D moving image against a 2-D fixed image must be rejected");
        };
        assert!(
            matches!(
                err,
                RegistrationError::DimensionMismatch {
                    fixed: 2,
                    moving: 3
                }
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn global_transform_is_rejected() {
        let img = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let metric = DemonsMetric::new(&img, &img, 0.001).unwrap();
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        assert!(matches!(
            metric.check_transform(&t),
            Err(RegistrationError::RequiresLocalSupportTransform { metric: "Demons" })
        ));
    }

    #[test]
    #[should_panic(expected = "requires a transform with local support")]
    fn evaluate_panics_on_global_transform() {
        let img = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let metric = DemonsMetric::new(&img, &img, 0.001).unwrap();
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let _ = metric.evaluate(&t);
    }

    #[test]
    #[should_panic(expected = "requires a transform with local support")]
    fn value_panics_on_global_transform() {
        let img = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let metric = DemonsMetric::new(&img, &img, 0.001).unwrap();
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let _ = metric.value(&t);
    }

    #[test]
    fn force_formula_matches_hand_computation_on_a_ramp() {
        // Fixed and moving are linear ramps with exactly analytic gradients
        // (mirrors mean_squares.rs's ramp test), so the demons force can be
        // hand-computed from the same closed-form slopes and compared to the
        // code's output directly — a formula-correctness check, not a finite
        // difference of the value (which the module docs explain does NOT
        // hold for demons: the force is Thirion's self-normalized update, not
        // the SSD's analytic gradient).
        fn ramp(w: usize, h: usize, ax: f64, ay: f64, b: f64) -> Image {
            let mut v = vec![0.0f64; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = b + ax * x as f64 + ay * y as f64;
                }
            }
            Image::from_vec(&[w, h], v).unwrap()
        }
        let (w, h) = (12usize, 12usize);
        let (fax, fay, fb) = (3.0, 5.0, 0.0);
        let (max, may, mb) = (2.0, -1.0, 4.0);
        let fixed = ramp(w, h, fax, fay, fb);
        let moving = ramp(w, h, max, may, mb);
        let metric = DemonsMetric::new(&fixed, &moving, 0.0).unwrap();

        let mut field = zero_field(&fixed);
        let n = field.number_of_parameters();
        // Off-lattice-in-effect: a uniform nonzero displacement (every pixel's
        // local block gets the same (px, py) shift), chosen so no sample maps
        // outside the image.
        let (px, py) = (1.3, -0.7);
        let mut params = vec![0.0; n];
        for i in 0..field.number_of_pixels() {
            params[i * 2] = px;
            params[i * 2 + 1] = py;
        }
        field.set_parameters(&params).unwrap();

        let got = metric.evaluate(&field);

        // Hand-compute the expected per-pixel force at an interior pixel
        // (away from the border, where the mapped point stays in-bounds and
        // the ramp's gradient is the constant analytic slope everywhere).
        let (x, y) = (5.0f64, 6.0f64);
        let fv = fb + fax * x + fay * y;
        let mv = mb + max * (x + px) + may * (y + py);
        let speed = mv - fv; // this crate's M − F convention
        let grad = [fax, fay]; // fixed-image gradient (default source)
        let grad_sq = grad[0] * grad[0] + grad[1] * grad[1];
        let normalizer = 1.0; // unit spacing, dim 2: (1+1)/2
        let denom = speed * speed / normalizer + grad_sq;
        // Every parameter block is written by exactly one sample (its own
        // pixel), but the whole derivative array — local blocks included — is
        // divided by the metric-wide `valid_points` count, mirroring the
        // uniform `n_factor`/`inv` normalization mattes.rs and mean_squares.rs
        // apply across every parameter regardless of local vs. dense support
        // (confirmed by mattes.rs's own
        // `local_support_reproduces_the_global_support_derivative` test).
        let inv = 1.0 / got.valid_points as f64;
        let expected = [speed * grad[0] / denom * inv, speed * grad[1] / denom * inv];

        // Note: valid_points < w*h here — a uniform positive shift pushes the
        // far-border pixels' mapped points outside the moving image, so those
        // samples are (correctly) excluded. The interior pixel checked below
        // is unaffected.
        assert!(got.valid_points > 0 && got.valid_points <= w * h);

        // Every interior pixel's contribution is identical (uniform shift,
        // uniform ramp gradient), so the averaged derivative at any interior
        // pixel's block equals the hand-computed single-sample force, scaled
        // by the same metric-wide 1/valid_points factor.
        let pixel = y as usize * w + x as usize;
        let (dx, dy) = (got.derivative[pixel * 2], got.derivative[pixel * 2 + 1]);
        assert!(
            (dx - expected[0]).abs() < 1e-9,
            "dx {dx} vs expected {}",
            expected[0]
        );
        assert!(
            (dy - expected[1]).abs() < 1e-9,
            "dy {dy} vs expected {}",
            expected[1]
        );

        // Sanity: a small step in the descent direction (subtracting the
        // derivative, this crate's convention) strictly decreases the value —
        // the property an FD test would normally certify, verified directly
        // since FD-of-value does not hold for this metric's force (see module
        // docs).
        let lr = 1e-3;
        let mut stepped = params.clone();
        for (p, d) in stepped.iter_mut().zip(&got.derivative) {
            *p -= lr * d;
        }
        let mut stepped_field = field.clone();
        stepped_field.set_parameters(&stepped).unwrap();
        let after = metric.evaluate(&stepped_field);
        assert!(
            after.value < got.value,
            "expected descent: before {} after {}",
            got.value,
            after.value
        );
    }

    #[test]
    fn intensity_difference_threshold_zeroes_the_guarded_samples_derivative_without_dropping_valid_points()
     {
        // A small but nonzero uniform intensity offset, and a shift small
        // enough to keep every sample in-bounds. With the threshold below the
        // offset every derivative entry is nonzero; raising the threshold
        // above the offset must zero every derivative entry while
        // valid_points (mapped-inside count) is unchanged — per the module
        // docs, ITK's ProcessPoint always returns true and only forces the
        // derivative to zero, it never excludes the sample from the count.
        let (w, h) = (10usize, 10usize);
        // Give the fixed image a tiny spatial gradient (so the denominator
        // guard is not also triggered — flat images have zero gradient), and
        // build moving as fixed plus a small CONSTANT offset, so the
        // intensity difference is uniform across every pixel regardless of
        // the spatial gradient.
        const OFFSET: f64 = 0.0005;
        let mut fv = vec![0.0f64; w * h];
        let mut mv = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                fv[y * w + x] = 1.0 + 0.01 * x as f64;
                mv[y * w + x] = fv[y * w + x] + OFFSET;
            }
        }
        let fixed = Image::from_vec(&[w, h], fv).unwrap();
        let moving = Image::from_vec(&[w, h], mv).unwrap();
        let field = zero_field(&fixed);

        let below = DemonsMetric::new(&fixed, &moving, 0.0001).unwrap();
        let r_below = below.evaluate(&field);
        assert_eq!(r_below.valid_points, w * h);
        assert!(
            r_below.derivative.iter().any(|&d| d != 0.0),
            "expected some nonzero force below the threshold"
        );

        let above = DemonsMetric::new(&fixed, &moving, 0.01).unwrap();
        let r_above = above.evaluate(&field);
        assert_eq!(
            r_above.valid_points,
            w * h,
            "threshold must not drop valid_points"
        );
        assert!(
            r_above.derivative.iter().all(|&d| d == 0.0),
            "expected the intensity-difference threshold to zero every force"
        );
        // The value itself (computed before either threshold is applied) is
        // identical regardless of the threshold.
        assert!((r_below.value - r_above.value).abs() < 1e-12);
    }

    #[test]
    fn denominator_threshold_zeroes_the_guarded_samples_derivative_on_a_flat_region() {
        // A perfectly flat pair of images: zero gradient everywhere, so the
        // denominator |grad|^2 + speed^2/normalizer is exactly 0 whenever the
        // images also match exactly (speed = 0) or nearly so — below
        // DENOMINATOR_THRESHOLD regardless of the (disabled, threshold = 0)
        // intensity-difference guard.
        let (w, h) = (6usize, 6usize);
        let fixed = Image::from_vec(&[w, h], vec![2.0f64; w * h]).unwrap();
        let moving = Image::from_vec(&[w, h], vec![2.0f64; w * h]).unwrap();
        let field = zero_field(&fixed);

        // Disable the intensity-difference guard (threshold 0) so only the
        // denominator guard can be responsible for zeroing the derivative.
        let metric = DemonsMetric::new(&fixed, &moving, 0.0).unwrap();
        let r = metric.evaluate(&field);
        assert_eq!(
            r.valid_points,
            w * h,
            "threshold must not drop valid_points"
        );
        assert!(
            r.derivative.iter().all(|&d| d == 0.0),
            "flat images: denominator is exactly 0, every force must be zeroed"
        );
        assert!(r.value.abs() < 1e-12);
    }

    #[test]
    fn displacement_field_recovers_a_known_uniform_shift_via_gradient_descent() {
        // Fixed is a blob; moving is the same blob shifted by a small known
        // amount. A zero-initialized DisplacementFieldTransform driven
        // directly by GradientDescentOptimizer (no ImageRegistrationMethod)
        // should recover that shift at pixels near the blob's edge, where the
        // intensity gradient is strong enough to clear both thresholds. (The
        // exact centre pixel is unsuitable to check: it sits at the Gaussian
        // peak, a stationary point of its own gradient, so it carries no
        // force signal at all.)
        let (w, h, sigma) = (32usize, 32usize, 5.0);
        let (cx, cy) = (16.0, 16.0);
        let (shift_x, shift_y) = (1.2, -0.8);
        let fixed = gaussian(w, h, cx, cy, sigma, 1.0);
        let moving = gaussian(w, h, cx + shift_x, cy + shift_y, sigma, 1.0);
        let metric = DemonsMetric::new(&fixed, &moving, 0.001).unwrap();

        let field = zero_field(&fixed);
        // Every pixel's own force is divided by the metric-wide valid_points
        // count (~w*h here — see the module docs and the ramp test above), so
        // an individually Newton-sized force needs a learning rate scaled up
        // by roughly that count to move a meaningful fraction of a pixel per
        // iteration. Tuned empirically against this fixture.
        let optimizer = GradientDescentOptimizer::new(50.0, 100);
        let result = optimizer.optimize(field.parameters(), |p| {
            let mut t = field.clone();
            t.set_parameters(p).unwrap();
            let mv = metric.evaluate(&t);
            (mv.value, mv.derivative)
        });

        let before = metric.evaluate(&field).value;
        assert!(
            result.value < before * 0.1,
            "expected large value decrease: before {before} after {}",
            result.value
        );

        // Probe pixels offset one sigma along each axis from the centre,
        // where that axis's gradient dominates and the other's is ~0.
        let px_pixel = (cy as usize) * w + (cx as usize + sigma as usize);
        let py_pixel = (cy as usize + sigma as usize) * w + (cx as usize);
        let (px_dx, px_dy) = (
            result.parameters[px_pixel * 2],
            result.parameters[px_pixel * 2 + 1],
        );
        let (py_dx, py_dy) = (
            result.parameters[py_pixel * 2],
            result.parameters[py_pixel * 2 + 1],
        );
        assert!(
            (px_dx - shift_x).abs() < 0.3,
            "recovered dx {px_dx} at the x-probe pixel vs expected {shift_x}"
        );
        assert!(
            (py_dy - shift_y).abs() < 0.3,
            "recovered dy {py_dy} at the y-probe pixel vs expected {shift_y}"
        );
        // Cross-axis components should stay small at each single-axis probe.
        assert!(px_dy.abs() < 0.3, "unexpected cross-axis dy {px_dy}");
        assert!(py_dx.abs() < 0.3, "unexpected cross-axis dx {py_dx}");
    }

    #[test]
    fn value_agrees_with_evaluate() {
        // The two thresholds gate only the force, so `value` ignores them —
        // it must still land on `evaluate`'s value, over the same sample set.
        use sitk_transform::ParametricTransform;

        let fixed = gaussian(16, 16, 8.0, 8.0, 4.0, 1.0);
        let moving = gaussian(16, 16, 9.5, 7.0, 4.0, 1.0);
        let metric = DemonsMetric::new(&fixed, &moving, 0.001).unwrap();

        let mut field = zero_field(&fixed);
        let full_at_identity = metric.evaluate(&field).value;
        assert!(
            (full_at_identity - metric.value(&field)).abs()
                <= 1e-12 * full_at_identity.abs().max(1.0),
            "at identity: evaluate {full_at_identity} vs value {}",
            metric.value(&field)
        );

        let mut params = field.parameters();
        for (i, p) in params.iter_mut().enumerate() {
            *p = if i % 2 == 0 { 0.6 } else { -0.4 };
        }
        field.set_parameters(&params).unwrap();
        let full = metric.evaluate(&field).value;
        let value_only = metric.value(&field);
        assert!(
            (full - value_only).abs() <= 1e-12 * full.abs().max(1.0),
            "at a nonzero field: evaluate {full} vs value {value_only}"
        );
    }
}
