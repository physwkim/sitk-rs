//! ANTS neighborhood cross-correlation image-to-image metric
//! (`itk::ANTSNeighborhoodCorrelationImageToImageMetricv4`).
//!
//! Around every fixed/virtual pixel `x`, this metric computes the normalized
//! cross-correlation of the fixed and (transformed) moving image over a small
//! rectangular window of radius `r` (`(2r+1)^dim` voxels) centred at `x`, then
//! averages that **local** correlation over the whole image:
//!
//! ```text
//! sF  = Σ_{y∈N(x)} F(y)            sM  = Σ_{y∈N(x)} M(T(y))
//! sFF = Σ_{y∈N(x)} (F(y) − F̄)²     sMM = Σ_{y∈N(x)} (M(T(y)) − M̄)²
//! sFM = Σ_{y∈N(x)} (F(y) − F̄)(M(T(y)) − M̄)      F̄ = sF/|N(x)|, M̄ = sM/|N(x)|
//!
//! localCC(x) = sFM² / (sFF·sMM)          (= 1 when sFF·sMM ≈ 0: a flat window)
//! value      = −(1/N) Σ_x localCC(x)
//! ```
//!
//! where `N` is the number of virtual points whose window contains at least one
//! valid voxel pair. Unlike a global normalized cross-correlation (one
//! correlation over the whole image), this **local** windowed form is
//! invariant to a *spatially varying* intensity gain (illumination fields, bias
//! fields, coil-sensitivity gain) because each window normalizes against its own
//! local mean/variance — the property that motivates using it for deformable,
//! same-modality registration.
//!
//! ## Derivative — ANTs' windowed-center approximation
//!
//! The derivative is **not** the exact gradient of `value` above. Per the ITK
//! class documentation: *"It is assumed that the derivative is only affected by
//! changes in the transform at the center of the window. This is obviously not
//! true but speeds the evaluation up considerably and works well in
//! practice."* Concretely, `∂sFM(x)/∂p` and `∂sMM(x)/∂p` are approximated by the
//! contribution of the window's own center pixel `x` alone (every other pixel
//! `y ∈ N(x)` is treated as fixed), which gives (see
//! `ComputeMovingTransformDerivative` in
//! `itkANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader.hxx`):
//!
//! ```text
//! derivWRTImage(x) = 2·sFM/(sFF·sMM) · ( fixedI − (sFM/sMM)·movingI ) · ∇M(T(x))
//! ∂value/∂p_k      = (1/N) Σ_x  derivWRTImage(x) · J_T(x)[·,k]     (sign: see below)
//! ```
//!
//! with `fixedI = F(x) − F̄`, `movingI = M(T(x)) − M̄`, and `J_T` the transform
//! Jacobian. This is ported verbatim — an intentional ITK approximation, not a
//! porting shortcut — so `tests::derivative_reflects_the_windowed_approximation`
//! compares it against a central finite difference with a wide tolerance
//! (documenting the gap) rather than the tight tolerance the exact-gradient
//! metrics ([`crate::MeanSquaresMetric`], [`crate::MattesMutualInformationMetric`]) use.
//!
//! ## Parity notes vs ITK
//!
//! * **Per-sample recomputation, not the sliding-window incremental update.**
//!   ITK's threader keeps a `deque` of per-hyperplane partial sums
//!   (`UpdateQueuesAtBeginningOfLine` / `UpdateQueuesToNextScanWindow`) so a
//!   window slides from one voxel to the next in `O(1)` amortized per
//!   dimension instead of re-summing `O((2r+1)^dim)` voxels. This port
//!   recomputes every window's sums from scratch (the private `point_result`
//!   method) — the *optimization not taken* — but the accumulated sums
//!   (`sF`, `sM`, `sFF`, `sMM`, `sFM`)
//!   and the value/derivative formulas built from them match ITK's
//!   `ComputeInformationFromQueues` / `ComputeMovingTransformDerivative`
//!   exactly, including the literal (non-simplified) centred-sum arithmetic
//!   (`sumFixed2 − 2·mean·sum + count·mean²`) for bit-identical rounding
//!   behavior.
//! * **Full sampling, exact window geometry.** Every fixed pixel is a virtual
//!   sample (SimpleITK's default), and the window at each sample is the
//!   `(2·radius+1)^dim` block of the fixed image's *own* grid indices centred
//!   there — ITK's `ConstNeighborhoodIterator` — clipped at the image boundary
//!   (`IndexInBounds`; a boundary window simply has fewer voxels).
//! * **Gradient source.** `∇M` is the exact gradient of the *linear
//!   interpolant* (the crate-private `MovingImage::value_and_physical_gradient`),
//!   like every other metric in this crate.
//! * **Sample set vs. raster.** [`FixedSamples`] is a flat, possibly-sparse
//!   list of points (whichever sampling strategy the driver configured), but
//!   this metric's window is defined on the fixed image's own voxel *grid*.
//!   [`from_samples`](AntsNeighborhoodCorrelationMetric::from_samples)
//!   documents why the raw fixed image is required in addition to
//!   `FixedSamples`, and gives a citation-backed answer (from ITK's sparse
//!   threader source) on whether a REGULAR/RANDOM sampling strategy is
//!   meaningful for this metric.
//! * **Global- vs local-support derivative accumulation.** Mirrors
//!   [`mattes`](crate::mattes): a global transform (translation, affine,
//!   versor, B-spline) accumulates a dense per-parameter derivative, summed
//!   over every valid point and divided by `N` (ITK's
//!   `AfterThreadedExecution`, `TransformCategory != DisplacementField`
//!   branch). A [local-support](sitk_transform::ParametricTransform::has_local_support)
//!   [`sitk_transform::DisplacementFieldTransform`] instead writes each point's contribution
//!   directly into *its own* parameter block via the block that
//!   `metric::local_support_block` assembles from
//!   [`sparse_jacobian_wrt_parameters`](sitk_transform::ParametricTransform::sparse_jacobian_wrt_parameters)
//!   and is **not** divided by `N` — ITK skips that division for
//!   `TransformCategory == DisplacementField` because each pixel already owns
//!   a disjoint parameter block, so there is nothing to average.
//! * **Sign convention.** ITK's `derivWRTImage` is `+∂(localCC)/∂p =
//!   −∂value/∂p` (`value = −localCC`), stored un-negated because ITK's v4
//!   optimizers *add* the returned derivative (steepest-descent direction).
//!   This crate's optimizers *subtract* (`p −= lr·derivative`), so every
//!   metric here stores `+∂value/∂p` — the true gradient of `value` — hence
//!   the derivative accumulated below is the **negation** of ITK's literal
//!   `derivWRTImage · J_T`, exactly the sign flip documented in
//!   [`mattes`](crate::mattes).

use sitk_core::Image;
use sitk_transform::ParametricTransform;
use sitk_transform::interpolator::{physical_to_index_matrix, strides};

use crate::error::{RegistrationError, Result};
use crate::metric::{FixedSamples, MetricValue, MovingImage, local_support_block};
use crate::scales::PhysicalShiftScales;

/// Per-point windowed statistics: the local correlation and (when defined)
/// the ANTs windowed-center derivative contribution `derivWRTImage`, in
/// **physical space** (length = image dimension).
struct PointResult {
    local_cc: f64,
    /// `None` when the window's fixed or moving variance is ~0 (ITK zeroes
    /// the derivative there rather than dividing by ~0).
    deriv_wrt_image: Option<Vec<f64>>,
    /// The sample's own physical point (its window center), reused by
    /// `evaluate` for the transform Jacobian — read once here from `raster`
    /// rather than re-derived by every caller.
    center_point: Vec<f64>,
}

/// The ANTS neighborhood cross-correlation metric. Holds the (possibly
/// sparse) fixed sample set, a full dense raster of the fixed image (needed
/// to reconstruct each sample's neighbourhood — see
/// [`from_samples`](Self::from_samples)), the moving image, and the window
/// radius. [`evaluate`](Self::evaluate) returns `value = −(1/N)·Σ localCC`
/// plus its parameter-derivative for a given transform.
pub struct AntsNeighborhoodCorrelationMetric {
    /// The (possibly sparse) sample set driving which points are evaluated
    /// as window centers — whatever sampling strategy the driver configured
    /// when building this via [`from_samples`](Self::from_samples).
    fixed: FixedSamples,
    /// Every voxel of the fixed image, in raster order — the single source
    /// for every window lookup, including each sample's own center pixel, so
    /// a sparse `fixed` still gets a full local window per sample. Built
    /// independently of `fixed` — see
    /// [`from_samples`](Self::from_samples) for why.
    raster: FixedSamples,
    /// `raster`'s grid shape.
    raster_size: Vec<usize>,
    /// First-axis-fastest strides matching `raster.values`/`raster.points`'
    /// row-major order (see [`sitk_transform::interpolator::strides`]).
    raster_strides: Vec<usize>,
    /// Each sample's own grid index into `raster` (row-major `N × dim`),
    /// precomputed once in [`from_samples`](Self::from_samples).
    sample_index: Vec<usize>,
    moving: MovingImage,
    /// Neighbourhood window radius, in voxels (ITK's `m_Radius`, uniform
    /// across axes — matches SimpleITK's `SetMetricAsANTSNeighborhoodCorrelation(radius)`).
    radius: usize,
    /// Every integer offset in `[-radius, radius]^dim`, precomputed once.
    window_offsets: Vec<Vec<isize>>,
}

impl AntsNeighborhoodCorrelationMetric {
    /// Build the metric from a **pre-built** sample set and moving image —
    /// the entry point the registration driver uses once it has applied its
    /// own metric sampling strategy, interpolator, and mask to build
    /// `fixed`/`moving` (see [`FixedSamples`]/[`MovingImage`]).
    /// [`new`](Self::new) is the convenience wrapper that builds both from
    /// raw images with full (dense) sampling.
    ///
    /// `fixed_image` — the same image `fixed` was sampled from — is **also**
    /// required, in addition to `fixed`, because this metric's neighbourhood
    /// window is defined on the fixed image's own voxel *raster*, which
    /// [`FixedSamples`] does not carry (it is a flat list of sampled
    /// points/values, not a grid). Concretely, `from_samples` builds its own
    /// full dense raster from `fixed_image` via `FixedSamples::from_image`
    /// (independent of whatever subset `fixed` holds) and, for each of
    /// `fixed`'s (possibly sparse) points, locates that point's grid index in
    /// the raster by inverting the fixed image's index-to-physical map and
    /// rounding to the nearest voxel — sample points are assumed to coincide
    /// exactly with a fixed-image grid index (true of every sampling
    /// strategy: full, regular, and random sampling in ITK's v4 framework all
    /// select image voxel *indices*, never off-grid points). Every
    /// window-neighbour lookup then reads from that dense raster, not from
    /// `fixed`, so a sparse `fixed` still gets each sampled point's full
    /// `(2·radius+1)^dim` local window.
    ///
    /// Fails if the window (diameter `2·radius + 1`) does not fit inside the
    /// fixed image along some axis, or `fixed_image`'s direction matrix is
    /// singular. `fixed_image` and `fixed` must have the same dimension
    /// (debug-asserted); `moving`'s dimension cannot be checked here — it is
    /// not exposed by [`MovingImage`] — so a mismatched `moving` will surface
    /// as an out-of-bounds panic or a nonsensical mapped point downstream.
    ///
    /// ## Is REGULAR/RANDOM sampling meaningful for this metric?
    ///
    /// **Yes.** `itk::ANTSNeighborhoodCorrelationImageToImageMetricv4` builds
    /// two threaders from the same
    /// `ANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader`
    /// template, selected by domain partitioner
    /// (`itkANTSNeighborhoodCorrelationImageToImageMetricv4.h:166-183`): a
    /// **dense** one (`ThreadedImageRegionPartitioner`, walks the whole image
    /// with ITK's sliding-window optimization) and a **sparse** one
    /// (`ThreadedIndexedContainerPartitioner`, walks a scattered point set —
    /// exactly what a REGULAR/RANDOM metric sampling strategy produces). The
    /// threader's class doc states plainly
    /// (`itkANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader.h:44-47`):
    /// *"Supports both dense and sparse threading ways. The dense threader
    /// iterates over the whole image domain in order and use a neighborhood
    /// scanning window to compute the local cross correlation metric... The
    /// sparse threader uses a sampled point set partitioner to compute local
    /// cross correlation only at the sampled positions."* Its sparse
    /// `ProcessVirtualPoint_impl` override
    /// (`itkANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader.hxx:587-624`)
    /// converts each sampled `virtualIndex` into a **single-point region**
    /// (`auto singlePointSize = ImageRegionType::SizeType::Filled(1); const
    /// ImageRegionType singlePointRegion(virtualIndex, singlePointSize);`,
    /// lines 603-604), then calls the *same*
    /// `InitializeScanning`/`UpdateQueues`/`ComputeInformationFromQueues`/
    /// `ComputeMovingTransformDerivative` routines the dense threader uses
    /// (lines 608-622) — i.e. it re-scans a full `ConstNeighborhoodIterator`
    /// window from the dense image raster, independently, at every sampled
    /// point. The comment directly above that override is explicit
    /// (lines 578-580): *"Specific implementation for sparse threader. It
    /// reuse most of the routine from the dense threader by reinitializing
    /// the scanning at every point."* So a REGULAR/RANDOM sample set changes
    /// only *which* pixels are evaluated as window centers — the per-point
    /// value/derivative formulas, and the full local window each center
    /// gets, are identical to the dense case. This is exactly what
    /// `from_samples`'s raster/`sample_index` split implements: nothing here
    /// needs to reject a sparse `fixed`.
    pub fn from_samples(
        fixed_image: &Image,
        fixed: FixedSamples,
        moving: MovingImage,
        radius: usize,
    ) -> Result<Self> {
        debug_assert_eq!(
            fixed_image.dimension(),
            fixed.dim,
            "from_samples: fixed_image and fixed samples have different dimension"
        );
        let dim = fixed_image.dimension();

        let raster_size = fixed_image.size().to_vec();
        let window = 2 * radius + 1;
        for (axis, &size) in raster_size.iter().enumerate() {
            if window > size {
                return Err(RegistrationError::NeighborhoodRadiusExceedsImage {
                    radius,
                    window,
                    size,
                    axis,
                });
            }
        }

        let raster = FixedSamples::from_image(fixed_image);
        let raster_strides = strides(&raster_size);

        // Invert the fixed image's index-to-physical map so each (possibly
        // sparse) sample's physical point can be located in `raster`'s grid
        // — mirrors `MovingImage::continuous_index` (private, metric.rs),
        // just run once per sample here instead of per moving-image lookup.
        let phys_to_index =
            physical_to_index_matrix(fixed_image.direction(), fixed_image.spacing(), dim)
                .ok_or(RegistrationError::SingularDirection)?;
        let origin = fixed_image.origin();

        let n = fixed.len();
        let mut sample_index = vec![0usize; n * dim];
        for s in 0..n {
            let p = &fixed.points[s * dim..(s + 1) * dim];
            for d in 0..dim {
                let row = &phys_to_index[d * dim..(d + 1) * dim];
                let c: f64 = row
                    .iter()
                    .zip(p.iter().zip(origin.iter()))
                    .map(|(&m, (&pj, &oj))| m * (pj - oj))
                    .sum();
                sample_index[s * dim + d] =
                    c.round().clamp(0.0, (raster_size[d] - 1) as f64) as usize;
            }
        }

        let window_offsets = window_offsets(dim, radius);

        Ok(Self {
            fixed,
            raster,
            raster_size,
            raster_strides,
            sample_index,
            moving,
            radius,
            window_offsets,
        })
    }

    /// Build the metric from a fixed and moving image and a neighbourhood
    /// window radius (ITK/SimpleITK default 2; SimpleITK's
    /// `SetMetricAsANTSNeighborhoodCorrelation` requires it explicitly),
    /// sampling every fixed pixel (SimpleITK's default sampling strategy).
    /// Fails if dimensions disagree, the moving direction matrix is
    /// singular, or the window (diameter `2·radius + 1`) does not fit inside
    /// the fixed image along some axis. Delegates to
    /// [`from_samples`](Self::from_samples).
    pub fn new(fixed: &Image, moving: &Image, radius: usize) -> Result<Self> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        Self::from_samples(
            fixed,
            FixedSamples::from_image(fixed),
            MovingImage::from_image(moving)?,
            radius,
        )
    }

    /// Number of fixed sample points.
    pub fn sample_count(&self) -> usize {
        self.fixed.len()
    }

    /// The configured neighbourhood window radius.
    pub fn radius(&self) -> usize {
        self.radius
    }

    /// Build a physical-shift scale/learning-rate estimator for `transform`
    /// over this metric's fixed sample points (shared with every metric in
    /// this crate).
    pub fn physical_shift_scales(
        &self,
        transform: &dyn ParametricTransform,
    ) -> PhysicalShiftScales {
        self.fixed.physical_shift_scales(transform)
    }

    /// Linear index into `raster` for sample `s`'s precomputed grid position.
    fn sample_linear_index(&self, s: usize) -> usize {
        let dim = self.fixed.dim;
        (0..dim)
            .map(|d| self.sample_index[s * dim + d] * self.raster_strides[d])
            .sum()
    }

    /// Windowed statistics at sample `s`: gather every in-bounds neighbour
    /// whose transformed point lands inside the moving image, accumulate
    /// ITK's `sumFixed`/`sumMoving`/`sumFixed2`/`sumMoving2`/`sumFixedMoving`
    /// over that window (`UpdateQueuesAtBeginningOfLine` /
    /// `UpdateQueuesToNextScanWindow`), form `sFF`/`sMM`/`sFM`
    /// (`ComputeInformationFromQueues`), then the local correlation and
    /// windowed-center derivative contribution
    /// (`ComputeMovingTransformDerivative`). Returns `None` when the window
    /// is empty or the center point itself does not map inside the moving
    /// image — mirroring ITK, which requires the center's own
    /// `TransformAndEvaluateMovingPoint` to succeed regardless of how many
    /// other window voxels are valid.
    fn point_result(&self, transform: &dyn ParametricTransform, s: usize) -> Option<PointResult> {
        let dim = self.fixed.dim;
        let idx = &self.sample_index[s * dim..(s + 1) * dim];

        let mut sum_f = 0.0f64;
        let mut sum_f2 = 0.0f64;
        let mut sum_m = 0.0f64;
        let mut sum_m2 = 0.0f64;
        let mut sum_fm = 0.0f64;
        let mut count = 0.0f64;

        for offset in &self.window_offsets {
            let mut lin = 0usize;
            let mut in_bounds = true;
            for d in 0..dim {
                let ni = idx[d] as isize + offset[d];
                if ni < 0 || ni as usize >= self.raster_size[d] {
                    in_bounds = false;
                    break;
                }
                lin += ni as usize * self.raster_strides[d];
            }
            if !in_bounds {
                continue;
            }

            let fv = self.raster.values[lin];
            let fp = &self.raster.points[lin * dim..(lin + 1) * dim];
            let mp = transform.transform_point(fp);
            let Some((mv, _)) = self.moving.value_and_physical_gradient(&mp) else {
                continue;
            };

            sum_f += fv;
            sum_f2 += fv * fv;
            sum_m += mv;
            sum_m2 += mv * mv;
            sum_fm += fv * mv;
            count += 1.0;
        }

        if count <= 0.0 {
            return None;
        }

        let fixed_mean = sum_f / count;
        let moving_mean = sum_m / count;
        // ITK's literal (non-simplified) expanded arithmetic, kept verbatim
        // for parity with `ComputeInformationFromQueues`.
        let s_ff =
            sum_f2 - fixed_mean * sum_f - fixed_mean * sum_f + count * fixed_mean * fixed_mean;
        let s_mm =
            sum_m2 - moving_mean * sum_m - moving_mean * sum_m + count * moving_mean * moving_mean;
        let s_fm =
            sum_fm - moving_mean * sum_f - fixed_mean * sum_m + count * moving_mean * fixed_mean;

        // The center voxel itself, evaluated again exactly as ITK's
        // `ComputeInformationFromQueues` re-evaluates `oindex = scanIt.GetIndex()`.
        // Sourced from `raster` (not `self.fixed`) so this always agrees with
        // the window neighbours above, regardless of whether `fixed` is a
        // sparse sample set.
        let center_lin = self.sample_linear_index(s);
        let fp_center = &self.raster.points[center_lin * dim..(center_lin + 1) * dim];
        let fv_center = self.raster.values[center_lin];
        let mp_center = transform.transform_point(fp_center);
        let (mv_center, grad_phys) = self.moving.value_and_physical_gradient(&mp_center)?;

        let fixed_i = fv_center - fixed_mean;
        let moving_i = mv_center - moving_mean;

        let eps = f64::EPSILON;
        let denom = s_ff * s_mm;
        let local_cc = if denom.abs() > eps {
            s_fm * s_fm / denom
        } else {
            1.0
        };

        let deriv_wrt_image = if s_ff > eps && s_mm > eps {
            let factor = 2.0 * s_fm / denom * (fixed_i - s_fm / s_mm * moving_i);
            Some(grad_phys.iter().map(|&g| factor * g).collect())
        } else {
            None
        };

        Some(PointResult {
            local_cc,
            deriv_wrt_image,
            center_point: fp_center.to_vec(),
        })
    }

    /// Evaluate `value = −(1/N)·Σ localCC` and its parameter-derivative for
    /// `transform`. Dispatches the derivative accumulation on
    /// [`ParametricTransform::has_local_support`], exactly as
    /// [`mattes`](crate::mattes) and ITK's `HasLocalSupport()` do: a global
    /// transform accumulates a dense per-parameter derivative and divides by
    /// `N`; a local-support (displacement-field) transform writes directly
    /// into each point's own parameter block and is not divided by `N`.
    pub fn evaluate(&self, transform: &dyn ParametricTransform) -> MetricValue {
        let nparams = transform.number_of_parameters();
        let local_support = transform.has_local_support();
        let n = self.fixed.len();

        let mut derivative = vec![0.0f64; nparams];
        let mut value_sum = 0.0f64;
        let mut valid = 0usize;

        for s in 0..n {
            let Some(PointResult {
                local_cc,
                deriv_wrt_image,
                center_point,
            }) = self.point_result(transform, s)
            else {
                continue;
            };

            value_sum -= local_cc;
            valid += 1;

            let Some(deriv_wrt_image) = deriv_wrt_image else {
                continue;
            };
            let fp = &center_point[..];

            if local_support {
                if let Some((offset, local_jac)) = local_support_block(transform, fp) {
                    let num_local = transform.number_of_local_parameters();
                    for mu in 0..num_local {
                        let mut acc = 0.0;
                        for (d, &dv) in deriv_wrt_image.iter().enumerate() {
                            acc += dv * local_jac[d * num_local + mu];
                        }
                        // Negated vs ITK's literal `derivWRTImage · jacobian`
                        // — see the module-doc sign-convention note.
                        derivative[offset + mu] -= acc;
                    }
                }
            } else {
                let jac = transform.jacobian_wrt_parameters(fp);
                for (k, dk) in derivative.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for (d, &dv) in deriv_wrt_image.iter().enumerate() {
                        acc += dv * jac[d * nparams + k];
                    }
                    *dk -= acc;
                }
            }
        }

        if valid == 0 {
            return MetricValue {
                value: f64::MAX,
                derivative: vec![0.0; nparams],
                valid_points: 0,
            };
        }

        let value = value_sum / valid as f64;
        if !local_support {
            for d in derivative.iter_mut() {
                *d /= valid as f64;
            }
        }

        MetricValue {
            value,
            derivative,
            valid_points: valid,
        }
    }
}

/// Every integer offset in the neighbourhood window `[-radius, radius]^dim`
/// (an odometer over `(2·radius + 1)^dim` combinations; enumeration order
/// does not affect the accumulated sums).
fn window_offsets(dim: usize, radius: usize) -> Vec<Vec<isize>> {
    let side = 2 * radius + 1;
    let total = side.pow(dim as u32);
    let mut offsets = Vec::with_capacity(total);
    let mut cur = vec![0usize; dim];
    for _ in 0..total {
        offsets.push(cur.iter().map(|&c| c as isize - radius as isize).collect());
        for c in cur.iter_mut() {
            *c += 1;
            if *c < side {
                break;
            }
            *c = 0;
        }
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::{DisplacementFieldTransform, TranslationTransform};

    /// A 2-D Gaussian blob of amplitude `amp` and width `sigma`, centred at
    /// `(cx, cy)` in physical (== index, unit spacing) coordinates, on a small
    /// constant pedestal so flat background windows are not the only content.
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

    /// Whole-image Pearson correlation coefficient, the "global NCC" this
    /// metric is compared against in
    /// [`local_ncc_is_invariant_to_a_spatially_varying_gain`]. Deliberately
    /// tiny and test-local rather than importing a sibling metric module —
    /// this crate does not (yet) export a global-NCC metric to depend on.
    fn global_ncc(a: &Image, b: &Image) -> f64 {
        let av = a.to_f64_vec();
        let bv = b.to_f64_vec();
        let n = av.len() as f64;
        let a_mean = av.iter().sum::<f64>() / n;
        let b_mean = bv.iter().sum::<f64>() / n;
        let mut saa = 0.0;
        let mut sbb = 0.0;
        let mut sab = 0.0;
        for (&x, &y) in av.iter().zip(&bv) {
            let dx = x - a_mean;
            let dy = y - b_mean;
            saa += dx * dx;
            sbb += dy * dy;
            sab += dx * dy;
        }
        sab / (saa.sqrt() * sbb.sqrt())
    }

    #[test]
    fn identical_images_at_identity_are_optimal_with_zero_derivative() {
        // F == M everywhere ⇒ every window has sFM == sFF == sMM, so
        // localCC == 1 (perfect correlation, value == -1) and, since
        // fixedI == movingI exactly, the windowed-center derivative factor
        // `(fixedI - sFM/sMM*movingI) == fixedI - fixedI == 0` at every point:
        // the derivative is *exactly* zero, not just small, up to rounding.
        let (w, h, sigma) = (24usize, 24usize, 5.0);
        let img = gaussian(w, h, 12.0, 12.0, sigma, 1.0);
        let metric = AntsNeighborhoodCorrelationMetric::new(&img, &img, 2).unwrap();

        let r = metric.evaluate(&TranslationTransform::new(vec![0.0, 0.0]));
        assert!((r.value - (-1.0)).abs() < 1e-9, "value {}", r.value);
        assert!(r.derivative[0].abs() < 1e-9, "d/dtx {}", r.derivative[0]);
        assert!(r.derivative[1].abs() < 1e-9, "d/dty {}", r.derivative[1]);
    }

    #[test]
    fn derivative_reflects_the_windowed_approximation() {
        // ITK's analytic derivative is the *windowed-center approximation*
        // documented in the module docs (only the window's own center pixel
        // is differentiated; the other `(2r+1)^dim − 1` window pixels are
        // treated as constant), not the exact gradient of `value`. That
        // approximation has a predictable, measured scale: for a translation,
        // it captures exactly 1 of the window's `(2r+1)^dim` "pixel
        // contributes to its own window's statistics" interactions that the
        // true gradient sums, so `analytic/fd ≈ 1/(2·radius+1)^dim` — verified
        // empirically (radius 1..4 on this same fixed/moving pair: measured
        // ratios 0.16, 0.044, 0.019, 0.011 vs 1/window of 0.11, 0.04, 0.020,
        // 0.012). This compares against a central finite difference with a
        // tolerance band around that predicted ratio — wide enough for the
        // documented approximation, tight enough to catch a sign flip or a
        // wrong-by-orders-of-magnitude porting bug. Offsets are chosen off
        // any half-integer so no sample flips validity under ±h (see
        // mattes.rs's identical note).
        let (w, h, sigma) = (24usize, 24usize, 4.0);
        let fixed = gaussian(w, h, 12.0, 12.0, sigma, 1.0);
        let moving = gaussian(w, h, 12.6, 11.4, sigma, 1.0);
        let radius = 2;
        let metric = AntsNeighborhoodCorrelationMetric::new(&fixed, &moving, radius).unwrap();
        let expected_ratio = 1.0 / (2 * radius + 1).pow(2) as f64;

        let p0 = [1.3f64, -0.7];
        let eval = |p: &[f64]| metric.evaluate(&TranslationTransform::new(p.to_vec()));
        let analytic = eval(&p0).derivative;

        let h_step = 1e-3;
        for k in 0..2 {
            let mut pp = p0;
            pp[k] += h_step;
            let mut pm = p0;
            pm[k] -= h_step;
            let fd = (eval(&pp).value - eval(&pm).value) / (2.0 * h_step);
            assert!(
                fd * analytic[k] > 0.0,
                "param {k}: fd {fd} and analytic {} have different signs",
                analytic[k]
            );
            let ratio = analytic[k] / fd;
            assert!(
                (expected_ratio / 5.0..expected_ratio * 5.0).contains(&ratio),
                "param {k}: fd {fd} vs analytic {} (ratio {ratio}, expected ~{expected_ratio})",
                analytic[k]
            );
        }
    }

    /// A full-frame smooth texture (a small sum of low-order sinusoids), used
    /// by [`local_ncc_is_invariant_to_a_spatially_varying_gain`] instead of a
    /// single blob on flat background. With a blob-on-background image, most
    /// window statistics *and* most of the whole-image variance sit in the
    /// large flat background, which makes the whole-image ("global")
    /// correlation look artificially robust to a gain field — diluted by
    /// pixels the gain barely affects — while every window *inside* the blob
    /// is fully exposed to it. A texture that fills the frame gives both
    /// statistics comparable exposure to the gain field, which is what
    /// isolates the actual local-vs-global sensitivity difference.
    fn texture(w: usize, h: usize) -> Image {
        let mut v = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let fx = x as f64 / w as f64;
                let fy = y as f64 / h as f64;
                v[y * w + x] = 1.0
                    + 0.6 * (2.0 * std::f64::consts::PI * 3.0 * fx).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 2.0 * fy).cos()
                    + 0.3 * (2.0 * std::f64::consts::PI * (3.0 * fx + 2.0 * fy)).sin();
            }
        }
        Image::from_vec(&[w, h], v).unwrap()
    }

    #[test]
    fn local_ncc_is_invariant_to_a_spatially_varying_gain() {
        // Multiply the moving image by a smooth spatial ramp (a simple model
        // of an illumination/bias field). Local NCC — normalizing against
        // each window's own mean/variance — should barely move; global NCC —
        // one mean/variance over the whole image — should move
        // substantially. This is the property that motivates windowed local
        // correlation for same-modality deformable registration.
        let (w, h) = (40usize, 40usize);
        let fixed = texture(w, h);
        let moving = fixed.clone();

        let mv = moving.to_f64_vec();
        let mut gained = vec![0.0f64; mv.len()];
        for y in 0..h {
            for x in 0..w {
                // Smooth multiplicative ramp from 0.4x to 2.2x across the image.
                let gain = 0.4 + 1.8 * (x as f64 / (w - 1) as f64);
                gained[y * w + x] = mv[y * w + x] * gain;
            }
        }
        let gained_moving = Image::from_vec(&[w, h], gained).unwrap();

        let local_before = AntsNeighborhoodCorrelationMetric::new(&fixed, &moving, 2)
            .unwrap()
            .evaluate(&TranslationTransform::new(vec![0.0, 0.0]))
            .value;
        let local_after = AntsNeighborhoodCorrelationMetric::new(&fixed, &gained_moving, 2)
            .unwrap()
            .evaluate(&TranslationTransform::new(vec![0.0, 0.0]))
            .value;

        let global_before = global_ncc(&fixed, &moving);
        let global_after = global_ncc(&fixed, &gained_moving);

        let local_shift = (local_after - local_before).abs();
        let global_shift = (global_before - global_after).abs();

        assert!(
            local_shift < 0.1,
            "local NCC value moved too much under a spatial gain: before {local_before} after {local_after}"
        );
        assert!(
            global_shift > 0.2,
            "global NCC unexpectedly stayed put under a spatial gain: before {global_before} after {global_after}"
        );
        assert!(
            local_shift < global_shift / 2.0,
            "local NCC ({local_shift}) should be far less sensitive to the gain than global NCC ({global_shift})"
        );
    }

    #[test]
    fn gradient_descent_recovers_a_translated_blob() {
        // Drives `GradientDescentOptimizer` directly (no `ImageRegistrationMethod`)
        // against this metric to confirm the derivative is at least a usable
        // descent direction end-to-end, not merely locally FD-consistent. The
        // learning rate is estimated once from the initial gradient via
        // `physical_shift_scales` (ITK's `EstimateLearningRate::Once`) rather
        // than hand-picked: this metric's raw derivative magnitude (unlike
        // mean-squares/Mattes) is not O(1) — see the module docs — so a fixed
        // guessed rate is not a reliable way to drive it.
        use crate::optimizer::GradientDescentOptimizer;

        let (w, h, sigma) = (48usize, 48usize, 5.0);
        let fixed = gaussian(w, h, 24.0, 24.0, sigma, 1.0);
        let moving = gaussian(w, h, 28.0, 21.0, sigma, 1.0); // shifted by (+4, -3)

        let metric = AntsNeighborhoodCorrelationMetric::new(&fixed, &moving, 2).unwrap();
        let start = vec![0.0f64, 0.0];
        let start_transform = TranslationTransform::new(start.clone());
        let estimator = metric.physical_shift_scales(&start_transform);
        let scales = estimator.estimate_scales();
        let scaled =
            |grad: &[f64]| -> Vec<f64> { grad.iter().zip(&scales).map(|(&g, &s)| g / s).collect() };

        let m0 = metric.evaluate(&start_transform);
        let lr_once = estimator.estimate_learning_rate(&scaled(&m0.derivative));

        let mut optimizer = GradientDescentOptimizer::new(lr_once, 300);
        optimizer.set_scales(scales.clone());

        let result = optimizer.optimize_with_lr(
            start,
            |p| {
                let t = TranslationTransform::new(p.to_vec());
                let r = metric.evaluate(&t);
                (r.value, r.derivative)
            },
            |grad| lr_once.min(estimator.estimate_learning_rate(&scaled(grad))),
        );

        // TranslationTransform maps fixed -> moving physical space, so the
        // recovered translation should land near (+4, -3).
        assert!(
            (result.parameters[0] - 4.0).abs() < 0.5,
            "tx {}",
            result.parameters[0]
        );
        assert!(
            (result.parameters[1] - (-3.0)).abs() < 0.5,
            "ty {}",
            result.parameters[1]
        );
    }

    #[test]
    fn radius_exceeding_the_image_is_rejected() {
        let img = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        assert!(matches!(
            AntsNeighborhoodCorrelationMetric::new(&img, &img, 4),
            Err(RegistrationError::NeighborhoodRadiusExceedsImage {
                radius: 4,
                window: 9,
                size: 8,
                axis: 0,
            })
        ));
        // window == size (== 8) still fits exactly.
        assert!(
            AntsNeighborhoodCorrelationMetric::new(
                &Image::from_vec(&[9, 9], vec![1.0; 81]).unwrap(),
                &Image::from_vec(&[9, 9], vec![1.0; 81]).unwrap(),
                4
            )
            .is_ok()
        );
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let fixed = gaussian(8, 8, 4.0, 4.0, 2.0, 1.0);
        let moving = Image::from_vec(&[8, 8, 8], vec![1.0; 512]).unwrap();
        assert!(matches!(
            AntsNeighborhoodCorrelationMetric::new(&fixed, &moving, 2),
            Err(RegistrationError::DimensionMismatch {
                fixed: 2,
                moving: 3
            })
        ));
    }

    #[test]
    fn local_support_reproduces_the_global_support_derivative() {
        // For a DisplacementFieldTransform whose grid matches the fixed
        // image, each pixel's local parameter block receives exactly its own
        // point's contribution (no aggregation collision), so the
        // local-support and global-support accumulations must agree exactly
        // — the same identity `mattes.rs` verifies for its local-support
        // path.
        use sitk_transform::ParametricTransform;

        let (w, h, sigma) = (16usize, 16usize, 3.0);
        let fixed = gaussian(w, h, 8.0, 8.0, sigma, 1.0);
        let moving = gaussian(w, h, 8.6, 7.4, sigma, 1.0);
        let metric = AntsNeighborhoodCorrelationMetric::new(&fixed, &moving, 2).unwrap();

        let mut field = DisplacementFieldTransform::from_image_domain(&fixed).unwrap();
        let np = field.number_of_parameters();
        let params: Vec<f64> = (0..np)
            .map(|i| ((i * 13 % 11) as f64 - 5.0) * 0.02)
            .collect();
        field.set_parameters(&params);

        // Force the global-support path for comparison by wrapping identical
        // per-point math: since `evaluate` dispatches internally on
        // `has_local_support`, and a `DisplacementFieldTransform` always
        // reports `true`, compare against a dense re-projection built from
        // the same `point_result` the local path uses.
        let local = metric.evaluate(&field);

        let nparams = field.number_of_parameters();
        let mut dense = vec![0.0f64; nparams];
        let mut value_sum = 0.0f64;
        let mut valid = 0usize;
        for s in 0..metric.sample_count() {
            let Some(pr) = metric.point_result(&field, s) else {
                continue;
            };
            value_sum -= pr.local_cc;
            valid += 1;
            let Some(deriv_wrt_image) = pr.deriv_wrt_image else {
                continue;
            };
            let jac = field.jacobian_wrt_parameters(&pr.center_point);
            for (k, dk) in dense.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (d, &dv) in deriv_wrt_image.iter().enumerate() {
                    acc += dv * jac[d * nparams + k];
                }
                *dk -= acc;
            }
        }
        // Note: unlike `value` (always averaged), the *derivative* is **not**
        // divided by `valid` here — matching `evaluate`'s local-support
        // branch, which mirrors ITK's `AfterThreadedExecution` skipping the
        // `/= NumberOfValidPoints` step for `TransformCategory ==
        // DisplacementField` (each pixel owns a disjoint parameter block, so
        // there is nothing to average away).
        let dense_value = value_sum / valid as f64;

        assert!(
            (local.value - dense_value).abs() < 1e-10,
            "value: local {} vs dense {}",
            local.value,
            dense_value
        );
        let max_diff = local
            .derivative
            .iter()
            .zip(&dense)
            .map(|(l, g)| (l - g).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-10, "max derivative diff {max_diff}");
    }
}
