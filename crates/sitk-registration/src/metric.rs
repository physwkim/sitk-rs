//! Image-to-image similarity metrics and their compute backend.
//!
//! Phase-0 registration ships one metric, **mean squares**
//! (`itk::MeanSquaresImageToImageMetricv4`):
//!
//! ```text
//! value = (1/N) Σ ( M(T(xᵢ)) − F(xᵢ) )²
//! ```
//!
//! over the fixed-image sample points `xᵢ` (the *virtual domain*) that map,
//! under transform `T`, inside the moving image `M`. `F` is the fixed image.
//! The derivative with respect to the transform parameters `p` is
//!
//! ```text
//! ∂value/∂pₖ = (2/N) Σ diffᵢ · ( ∇M(T(xᵢ)) · J_T(xᵢ) )ₖ
//! ```
//!
//! where `diffᵢ = M(T(xᵢ)) − F(xᵢ)`, `∇M` is the moving image's spatial
//! gradient, and `J_T` is the transform Jacobian
//! ([`ParametricTransform::jacobian_wrt_parameters`]).
//!
//! `∇M` here is the **exact gradient of the linear interpolant**
//! ([`linear_value_and_gradient`]), so the metric derivative is the true
//! gradient of the (interpolated) metric value — the optimizer's finite
//! difference of the value reproduces it. This is a documented, deliberate
//! difference from ITK, whose `ImageToImageMetricv4` defaults to a
//! Gaussian-smoothed gradient image (or a raw central-difference
//! `CentralDifferenceImageFunction` when `SetUseMovingImageGradientFilter` is
//! off); both are gradient *estimates* not consistent with the interpolated
//! value.
//!
//! ## GPU seam
//!
//! The per-sample reduction is isolated behind [`MetricBackend`]. [`CpuBackend`]
//! runs it on the host and is the only backend that compiles on a machine
//! without a GPU (this one). A future CUDA (`cudarc`) or portable
//! `wgpu`/Metal backend implements the same trait — marshalling the sample
//! arrays, moving buffer, and transform parameters to the device — without any
//! change to [`MeanSquaresMetric`] or the registration method above it.

use sitk_core::Image;
use sitk_transform::ParametricTransform;
use sitk_transform::interpolator::{
    index_to_physical_matrix, linear_value_and_gradient, physical_to_index_matrix, strides,
};

use crate::error::{RegistrationError, Result};
use crate::scales::PhysicalShiftScales;

/// The fixed image reduced to its sample set (the registration *virtual
/// domain*): every pixel's value and its physical point, precomputed once.
pub struct FixedSamples {
    dim: usize,
    /// One value per sample, length `N`.
    values: Vec<f64>,
    /// Physical points, row-major `N × dim`.
    points: Vec<f64>,
    /// Minimum fixed-image spacing (the maximum physical step for optimization).
    min_spacing: f64,
}

impl FixedSamples {
    /// Reduce a fixed image to its full sample set (sampling strategy = None:
    /// every pixel, matching SimpleITK's default).
    pub fn from_image(fixed: &Image) -> Self {
        let dim = fixed.dimension();
        let size = fixed.size().to_vec();
        let values = fixed.to_f64_vec();
        let n = values.len();

        // point = origin + (D · diag(spacing)) · index
        let idx_to_phys = index_to_physical_matrix(fixed.direction(), fixed.spacing(), dim);
        let origin = fixed.origin();

        let mut points = vec![0.0; n * dim];
        let mut index = vec![0usize; dim];
        for s in 0..n {
            for r in 0..dim {
                let mut acc = origin[r];
                for (c, &idx) in index.iter().enumerate() {
                    acc += idx_to_phys[r * dim + c] * idx as f64;
                }
                points[s * dim + r] = acc;
            }
            increment(&mut index, &size);
        }

        let min_spacing = fixed
            .spacing()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        Self {
            dim,
            values,
            points,
            min_spacing,
        }
    }

    /// Number of samples `N`.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The moving image as an `f64` buffer plus the geometry needed to map a
/// physical point to a continuous index and to convert an index-space gradient
/// to a physical-space gradient.
pub struct MovingImage {
    dim: usize,
    buf: Vec<f64>,
    size: Vec<usize>,
    strides: Vec<usize>,
    origin: Vec<f64>,
    /// `diag(1/spacing) · D⁻¹`, row-major `dim × dim`: maps a physical
    /// displacement from the origin to a continuous index.
    phys_to_index: Vec<f64>,
}

impl MovingImage {
    /// Prepare a moving image. Fails if its direction matrix is singular.
    pub fn from_image(moving: &Image) -> Result<Self> {
        let dim = moving.dimension();
        let size = moving.size().to_vec();
        let phys_to_index = physical_to_index_matrix(moving.direction(), moving.spacing(), dim)
            .ok_or(RegistrationError::SingularDirection)?;
        Ok(Self {
            dim,
            buf: moving.to_f64_vec(),
            strides: strides(&size),
            size,
            origin: moving.origin().to_vec(),
            phys_to_index,
        })
    }

    /// Continuous index of physical point `p`: `M · (p − origin)`.
    fn continuous_index(&self, p: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let mut c = vec![0.0; dim];
        for (r, cr) in c.iter_mut().enumerate() {
            let row = &self.phys_to_index[r * dim..(r + 1) * dim];
            *cr = row
                .iter()
                .zip(p.iter().zip(self.origin.iter()))
                .map(|(&m, (&pj, &oj))| m * (pj - oj))
                .sum();
        }
        c
    }

    /// Linear sample and its exact index-space gradient at continuous index
    /// `c`, or `None` if outside the buffer.
    fn value_and_gradient(&self, c: &[f64]) -> Option<(f64, Vec<f64>)> {
        linear_value_and_gradient(&self.buf, &self.size, &self.strides, c)
    }
}

/// The value and parameter-derivative of a metric at one transform.
#[derive(Clone, Debug)]
pub struct MetricValue {
    /// Metric value (lower is better for mean squares).
    pub value: f64,
    /// `∂value/∂pₖ`, length = number of transform parameters.
    pub derivative: Vec<f64>,
    /// How many fixed samples mapped inside the moving image.
    pub valid_points: usize,
}

/// Compute backend for the mean-squares metric: the isolated, parallelizable
/// per-sample reduction. See the [module docs](self#gpu-seam) for the GPU seam.
pub trait MetricBackend {
    /// Accumulate the mean-squares value and its parameter-derivative over all
    /// fixed samples for the given transform.
    fn mean_squares(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> MetricValue;
}

/// Host (CPU) implementation of [`MetricBackend`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl MetricBackend for CpuBackend {
    fn mean_squares(
        &self,
        fixed: &FixedSamples,
        moving: &MovingImage,
        transform: &dyn ParametricTransform,
    ) -> MetricValue {
        let dim = fixed.dim;
        let nparams = transform.number_of_parameters();
        let n = fixed.values.len();

        let mut value_sum = 0.0;
        let mut deriv = vec![0.0; nparams];
        let mut valid = 0usize;

        for s in 0..n {
            let fp = &fixed.points[s * dim..(s + 1) * dim];
            let fv = fixed.values[s];

            let mp = transform.transform_point(fp);
            let cidx = moving.continuous_index(&mp);
            let (mv, grad_index) = match moving.value_and_gradient(&cidx) {
                Some(vg) => vg,
                None => continue,
            };

            let diff = mv - fv;
            value_sum += diff * diff;

            // Convert the index-space gradient to a physical-space gradient. With
            // cindex = M·(p − origin),
            // ∂M(value)/∂p_d = Σ_j (∂value/∂cindex_j) · M[j][d].
            let mut grad_phys = vec![0.0; dim];
            for (d, gp) in grad_phys.iter_mut().enumerate() {
                *gp = grad_index
                    .iter()
                    .enumerate()
                    .map(|(j, &gj)| gj * moving.phys_to_index[j * dim + d])
                    .sum();
            }

            // deriv_k += 2·diff · Σ_d grad_phys[d] · J[d][k].
            let jac = transform.jacobian_wrt_parameters(fp);
            for (k, dk) in deriv.iter_mut().enumerate() {
                let mut g = 0.0;
                for (d, &gp) in grad_phys.iter().enumerate() {
                    g += gp * jac[d * nparams + k];
                }
                *dk += 2.0 * diff * g;
            }

            valid += 1;
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
            derivative: deriv.iter().map(|d| d * inv).collect(),
            valid_points: valid,
        }
    }
}

/// The mean-squares image-to-image metric. Holds the precomputed fixed samples
/// and moving image; [`evaluate`](Self::evaluate) returns value + derivative for
/// a given transform through the chosen backend.
pub struct MeanSquaresMetric {
    fixed: FixedSamples,
    moving: MovingImage,
}

impl MeanSquaresMetric {
    /// Build the metric from a fixed and moving image. Fails if dimensions
    /// disagree or the moving direction matrix is singular.
    pub fn new(fixed: &Image, moving: &Image) -> Result<Self> {
        if fixed.dimension() != moving.dimension() {
            return Err(RegistrationError::DimensionMismatch {
                fixed: fixed.dimension(),
                moving: moving.dimension(),
            });
        }
        Ok(Self {
            fixed: FixedSamples::from_image(fixed),
            moving: MovingImage::from_image(moving)?,
        })
    }

    /// Number of fixed sample points.
    pub fn sample_count(&self) -> usize {
        self.fixed.len()
    }

    /// Build a physical-shift scale/learning-rate estimator for `transform`
    /// over this metric's fixed sample points (ITK
    /// `RegistrationParameterScalesFromPhysicalShift`).
    pub fn physical_shift_scales(
        &self,
        transform: &dyn ParametricTransform,
    ) -> PhysicalShiftScales {
        PhysicalShiftScales::new(
            &self.fixed.points,
            self.fixed.dim,
            transform,
            self.fixed.min_spacing,
        )
    }

    /// Evaluate value + derivative for `transform` using `backend`.
    pub fn evaluate(
        &self,
        transform: &dyn ParametricTransform,
        backend: &dyn MetricBackend,
    ) -> MetricValue {
        backend.mean_squares(&self.fixed, &self.moving, transform)
    }
}

/// Increment a multi-index in place (first index fastest).
fn increment(index: &mut [usize], size: &[usize]) {
    for d in 0..index.len() {
        index[d] += 1;
        if index[d] < size[d] {
            return;
        }
        index[d] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::TranslationTransform;

    // A separable ramp f(x,y) = 3x + 5y makes the mean-squares gradient exactly
    // analytic, so we can check the derivative sign and magnitude precisely.
    fn ramp(w: usize, h: usize, ax: f64, ay: f64) -> Image {
        let mut v = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = ax * x as f64 + ay * y as f64;
            }
        }
        Image::from_vec(&[w, h], v).unwrap()
    }

    #[test]
    fn identity_on_equal_images_is_zero_with_zero_gradient() {
        let img = ramp(8, 8, 3.0, 5.0);
        let metric = MeanSquaresMetric::new(&img, &img).unwrap();
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let r = metric.evaluate(&t, &CpuBackend);
        assert!(r.value.abs() < 1e-9, "value {}", r.value);
        assert!(r.derivative[0].abs() < 1e-6, "d/dtx {}", r.derivative[0]);
        assert!(r.derivative[1].abs() < 1e-6, "d/dty {}", r.derivative[1]);
        assert_eq!(r.valid_points, 64);
    }

    #[test]
    fn derivative_matches_finite_difference() {
        // Fixed is the ramp; moving is the same ramp. Evaluate the metric as a
        // function of the translation parameters at a nonzero point and compare
        // the analytic derivative to a central finite difference.
        let fixed = ramp(12, 12, 3.0, 5.0);
        let moving = ramp(12, 12, 3.0, 5.0);
        let metric = MeanSquaresMetric::new(&fixed, &moving).unwrap();

        // Offsets chosen off any half-integer so no sample sits on the
        // is_inside boundary (which would flip validity under ±h and break the
        // finite difference).
        let p0 = [0.3f64, -0.4];
        let eval = |p: &[f64]| {
            let t = TranslationTransform::new(p.to_vec());
            metric.evaluate(&t, &CpuBackend)
        };
        let analytic = eval(&p0).derivative;

        let h = 1e-4;
        for k in 0..2 {
            let mut pp = p0;
            pp[k] += h;
            let mut pm = p0;
            pm[k] -= h;
            let fd = (eval(&pp).value - eval(&pm).value) / (2.0 * h);
            assert!(
                (fd - analytic[k]).abs() < 1e-3,
                "param {k}: fd {fd} vs analytic {}",
                analytic[k]
            );
        }
    }
}
