//! Image resampling: `itk::ResampleImageFilter`.
//!
//! For each output voxel the output continuous index is mapped to a physical
//! point, the [`Transform`] maps that point into the input's physical space, and
//! the input is interpolated there. Points that fall outside the input buffer
//! take the default pixel value.

use sitk_core::{Image, PixelId, matrix};

use crate::error::{Result, TransformError};
use crate::interpolator::{
    SincWindow, affine_apply, bspline_coefficients, bspline_value_and_gradient,
    gaussian_value_and_gradient, index_to_physical_matrix, linear_at, nearest_at,
    physical_to_index_matrix, strides, windowed_sinc_value_and_gradient,
};
use crate::transform::Transform;

/// Interpolation kernel used when sampling the input image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolator {
    /// Nearest-neighbour (round-half-up, matching ITK's `RoundHalfIntegerUp`).
    NearestNeighbor,
    /// N-linear interpolation.
    Linear,
    /// Cubic (order-3) B-spline, matching SimpleITK's `sitkBSpline` /
    /// `sitkBSpline3` default (`itk::BSplineInterpolateImageFunction`).
    /// Interpolating: reproduces the original samples exactly at integer
    /// indices.
    BSpline,
    /// Gaussian-weighted local average
    /// (`itk::GaussianInterpolateImageFunction`), fixed at SimpleITK's
    /// `sitkGaussian` preset width ([`interpolator::GAUSSIAN_SIGMA`] /
    /// [`interpolator::GAUSSIAN_ALPHA`], in continuous-index units). Unlike
    /// the other three kernels this is *not* interpolating — it smooths
    /// rather than reproducing samples exactly.
    ///
    /// [`interpolator::GAUSSIAN_SIGMA`]: crate::interpolator::GAUSSIAN_SIGMA
    /// [`interpolator::GAUSSIAN_ALPHA`]: crate::interpolator::GAUSSIAN_ALPHA
    Gaussian,
    /// Windowed sinc, Hamming window (`sitkHammingWindowedSinc`,
    /// `itk::WindowedSincInterpolateImageFunction` with
    /// `itk::Function::HammingWindowFunction`), fixed at SimpleITK's radius-5
    /// preset ([`interpolator::WINDOWED_SINC_RADIUS`]). Interpolating.
    ///
    /// [`interpolator::WINDOWED_SINC_RADIUS`]: crate::interpolator::WINDOWED_SINC_RADIUS
    HammingWindowedSinc,
    /// Windowed sinc, Cosine window (`sitkCosineWindowedSinc`) — see
    /// [`HammingWindowedSinc`](Self::HammingWindowedSinc) for the shared
    /// radius/kernel notes.
    CosineWindowedSinc,
    /// Windowed sinc, Welch window (`sitkWelchWindowedSinc`) — see
    /// [`HammingWindowedSinc`](Self::HammingWindowedSinc) for the shared
    /// radius/kernel notes.
    WelchWindowedSinc,
    /// Windowed sinc, Lanczos window (`sitkLanczosWindowedSinc`) — see
    /// [`HammingWindowedSinc`](Self::HammingWindowedSinc) for the shared
    /// radius/kernel notes.
    LanczosWindowedSinc,
    /// Windowed sinc, Blackman window (`sitkBlackmanWindowedSinc`) — see
    /// [`HammingWindowedSinc`](Self::HammingWindowedSinc) for the shared
    /// radius/kernel notes.
    BlackmanWindowedSinc,
}

/// `itk::ResampleImageFilter`: build an output grid and sample the input through
/// a [`Transform`].
///
/// Reference geometry (size / spacing / origin / direction) defaults to the
/// input image's own grid when not overridden. Output pixel type defaults to the
/// input's.
pub struct ResampleImageFilter {
    size: Option<Vec<usize>>,
    spacing: Option<Vec<f64>>,
    origin: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
    interpolator: Interpolator,
    default_value: f64,
    output_pixel_type: Option<PixelId>,
}

impl Default for ResampleImageFilter {
    fn default() -> Self {
        Self {
            size: None,
            spacing: None,
            origin: None,
            direction: None,
            interpolator: Interpolator::Linear,
            default_value: 0.0,
            output_pixel_type: None,
        }
    }
}

impl ResampleImageFilter {
    /// A filter with default settings (linear interpolation, default value 0,
    /// output grid and type inherited from the input at `execute` time).
    pub fn new() -> Self {
        Self::default()
    }

    /// Take the whole output grid (size, spacing, origin, direction) from a
    /// reference image.
    pub fn set_reference_image(&mut self, reference: &Image) -> &mut Self {
        self.size = Some(reference.size().to_vec());
        self.spacing = Some(reference.spacing().to_vec());
        self.origin = Some(reference.origin().to_vec());
        self.direction = Some(reference.direction().to_vec());
        self
    }

    /// Override the output size.
    pub fn set_size(&mut self, size: Vec<usize>) -> &mut Self {
        self.size = Some(size);
        self
    }

    /// Override the output spacing.
    pub fn set_output_spacing(&mut self, spacing: Vec<f64>) -> &mut Self {
        self.spacing = Some(spacing);
        self
    }

    /// Override the output origin.
    pub fn set_output_origin(&mut self, origin: Vec<f64>) -> &mut Self {
        self.origin = Some(origin);
        self
    }

    /// Override the output direction (row-major `dim x dim`).
    pub fn set_output_direction(&mut self, direction: Vec<f64>) -> &mut Self {
        self.direction = Some(direction);
        self
    }

    /// Choose the interpolation kernel.
    pub fn set_interpolator(&mut self, interpolator: Interpolator) -> &mut Self {
        self.interpolator = interpolator;
        self
    }

    /// Value written where the mapped point falls outside the input buffer.
    pub fn set_default_pixel_value(&mut self, value: f64) -> &mut Self {
        self.default_value = value;
        self
    }

    /// Force the output pixel type (default: same as input).
    pub fn set_output_pixel_type(&mut self, id: PixelId) -> &mut Self {
        self.output_pixel_type = Some(id);
        self
    }

    /// Resample `input` through `transform`.
    pub fn execute<T: Transform>(&self, input: &Image, transform: &T) -> Result<Image> {
        let dim = input.dimension();

        let out_size = self.size.clone().unwrap_or_else(|| input.size().to_vec());
        let out_spacing = self
            .spacing
            .clone()
            .unwrap_or_else(|| input.spacing().to_vec());
        let out_origin = self
            .origin
            .clone()
            .unwrap_or_else(|| input.origin().to_vec());
        let out_direction = self
            .direction
            .clone()
            .unwrap_or_else(|| input.direction().to_vec());
        let out_type = self.output_pixel_type.unwrap_or_else(|| input.pixel_id());

        if out_size.len() != dim
            || out_spacing.len() != dim
            || out_origin.len() != dim
            || out_direction.len() != dim * dim
            || transform.dimension() != dim
        {
            return Err(TransformError::DimensionMismatch);
        }

        // Precompute the two affines once, instead of inverting per voxel.
        // Output index -> physical:  p = out_origin + (D_out · diag(out_spacing)) · index
        let out_index_to_phys = index_to_physical_matrix(&out_direction, &out_spacing, dim);
        // Input physical -> continuous index: idx = diag(1/in_spacing) · D_in⁻¹ · (p − in_origin)
        let in_phys_to_index = physical_to_index_matrix(input.direction(), input.spacing(), dim)
            .ok_or(TransformError::SingularDirection)?;
        let in_origin = input.origin().to_vec();

        let in_buf = input.to_f64_vec()?;
        let in_size = input.size().to_vec();
        let in_strides = strides(&in_size);
        // Coefficient decomposition is global (mixes the whole line via the
        // IIR recursion), so it is computed once up front rather than per
        // output voxel.
        let bspline_coeffs = matches!(self.interpolator, Interpolator::BSpline)
            .then(|| bspline_coefficients(&in_buf, &in_size, &in_strides));

        let n_out: usize = out_size.iter().product();
        let mut out_vals = vec![0.0f64; n_out];
        let mut index = vec![0usize; dim];
        for out_val in out_vals.iter_mut() {
            let index_f: Vec<f64> = index.iter().map(|&i| i as f64).collect();
            let phys = affine_apply(&out_index_to_phys, &index_f, &out_origin, dim);
            let mapped = transform.transform_point(&phys);
            let diff: Vec<f64> = (0..dim).map(|d| mapped[d] - in_origin[d]).collect();
            let cindex = matrix::mat_vec(&in_phys_to_index, &diff, dim);

            *out_val = match self.interpolator {
                Interpolator::NearestNeighbor => {
                    nearest_at(&in_buf, &in_size, &in_strides, &cindex)
                        .unwrap_or(self.default_value)
                }
                Interpolator::Linear => {
                    linear_at(&in_buf, &in_size, &in_strides, &cindex).unwrap_or(self.default_value)
                }
                Interpolator::BSpline => bspline_value_and_gradient(
                    bspline_coeffs.as_ref().expect("computed above for BSpline"),
                    &in_size,
                    &in_strides,
                    &cindex,
                )
                .map(|(v, _)| v)
                .unwrap_or(self.default_value),
                Interpolator::Gaussian => {
                    gaussian_value_and_gradient(&in_buf, &in_size, &in_strides, &cindex)
                        .map(|(v, _)| v)
                        .unwrap_or(self.default_value)
                }
                Interpolator::HammingWindowedSinc => windowed_sinc_value_and_gradient(
                    &in_buf,
                    &in_size,
                    &in_strides,
                    &cindex,
                    SincWindow::Hamming,
                )
                .map(|(v, _)| v)
                .unwrap_or(self.default_value),
                Interpolator::CosineWindowedSinc => windowed_sinc_value_and_gradient(
                    &in_buf,
                    &in_size,
                    &in_strides,
                    &cindex,
                    SincWindow::Cosine,
                )
                .map(|(v, _)| v)
                .unwrap_or(self.default_value),
                Interpolator::WelchWindowedSinc => windowed_sinc_value_and_gradient(
                    &in_buf,
                    &in_size,
                    &in_strides,
                    &cindex,
                    SincWindow::Welch,
                )
                .map(|(v, _)| v)
                .unwrap_or(self.default_value),
                Interpolator::LanczosWindowedSinc => windowed_sinc_value_and_gradient(
                    &in_buf,
                    &in_size,
                    &in_strides,
                    &cindex,
                    SincWindow::Lanczos,
                )
                .map(|(v, _)| v)
                .unwrap_or(self.default_value),
                Interpolator::BlackmanWindowedSinc => windowed_sinc_value_and_gradient(
                    &in_buf,
                    &in_size,
                    &in_strides,
                    &cindex,
                    SincWindow::Blackman,
                )
                .map(|(v, _)| v)
                .unwrap_or(self.default_value),
            };

            increment(&mut index, &out_size);
        }

        // Cast f64 results to the requested output pixel type.
        let mut result = build_output(out_type, &out_size, out_vals)?;
        result
            .set_spacing(&out_spacing)
            .map_err(TransformError::Core)?;
        result
            .set_origin(&out_origin)
            .map_err(TransformError::Core)?;
        result
            .set_direction(&out_direction)
            .map_err(TransformError::Core)?;
        Ok(result)
    }
}

fn build_output(id: PixelId, size: &[usize], vals: Vec<f64>) -> Result<Image> {
    use sitk_core::{Scalar, dispatch_scalar};
    fn make<T: Scalar>(size: &[usize], vals: &[f64]) -> sitk_core::Result<Image> {
        let out: Vec<T> = vals.iter().map(|&v| T::from_f64(v)).collect();
        Image::from_vec(size, out)
    }
    dispatch_scalar!(id, make, size, &vals).map_err(TransformError::Core)
}

/// Increment a multi-index in place (first index fastest). Wraps silently on
/// the final overflow, which the caller never reads.
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
    use crate::transform::{AffineTransform, TranslationTransform};

    fn ramp_2d(w: usize, h: usize) -> Image {
        let data: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        Image::from_vec(&[w, h], data).unwrap()
    }

    #[test]
    fn identity_transform_reproduces_input() {
        let img = ramp_2d(5, 4);
        let t = AffineTransform::identity(2);
        let out = ResampleImageFilter::new()
            .set_reference_image(&img)
            .execute(&img, &t)
            .unwrap();
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            img.scalar_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn integer_translation_shifts_pixels() {
        // Output(p) samples Input(transform(p)); a +1x transform shifts content
        // toward −x by one pixel, with the exposed column taking the default.
        let img = ramp_2d(4, 1); // values 0,1,2,3
        let t = TranslationTransform::new(vec![1.0, 0.0]);
        let out = ResampleImageFilter::new()
            .set_reference_image(&img)
            .set_default_pixel_value(-1.0)
            .set_interpolator(Interpolator::NearestNeighbor)
            .execute(&img, &t)
            .unwrap();
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, -1.0]);
    }

    #[test]
    fn linear_interpolation_halfway() {
        // Sampling at x+0.5 averages neighbours.
        let img = ramp_2d(4, 1); // 0,1,2,3
        let t = TranslationTransform::new(vec![0.5, 0.0]);
        let out = ResampleImageFilter::new()
            .set_reference_image(&img)
            .set_default_pixel_value(0.0)
            .set_interpolator(Interpolator::Linear)
            .execute(&img, &t)
            .unwrap();
        // x=0 ->0.5 :0.5 ; x=1->1.5:1.5 ; x=2->2.5:2.5 ; x=3->3.5 outside (>=3.5) -> default 0
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[0.5, 1.5, 2.5, 0.0]);
    }

    #[test]
    fn output_pixel_type_override() {
        let img = ramp_2d(3, 1);
        let t = AffineTransform::identity(2);
        let out = ResampleImageFilter::new()
            .set_reference_image(&img)
            .set_output_pixel_type(PixelId::UInt8)
            .execute(&img, &t)
            .unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 1, 2]);
    }

    #[test]
    fn resample_to_larger_grid_via_spacing() {
        // Halve the spacing -> twice as many samples along x, linearly interpolated.
        let img = ramp_2d(3, 1); // 0,1,2 at spacing 1
        let t = AffineTransform::identity(2);
        let out = ResampleImageFilter::new()
            .set_size(vec![5, 1])
            .set_output_spacing(vec![0.5, 1.0])
            .set_output_origin(vec![0.0, 0.0])
            .set_output_direction(vec![1.0, 0.0, 0.0, 1.0])
            .set_interpolator(Interpolator::Linear)
            .execute(&img, &t)
            .unwrap();
        // output indices 0..4 -> physical x 0,0.5,1,1.5,2 -> values 0,0.5,1,1.5,2
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[0.0, 0.5, 1.0, 1.5, 2.0]
        );
    }

    #[test]
    fn bspline_interpolation_reproduces_ramp_on_identity() {
        // Smoke test for the BSpline dispatch branch: on an identity
        // transform every output sample lands on an integer input index,
        // which the interpolating B-spline kernel reproduces up to
        // coefficient round-trip floating-point noise (see the tighter,
        // tolerance-based exact-reproduction check in `interpolator`'s own
        // tests).
        let img = ramp_2d(6, 1); // 0,1,2,3,4,5
        let t = AffineTransform::identity(2);
        let out = ResampleImageFilter::new()
            .set_reference_image(&img)
            .set_interpolator(Interpolator::BSpline)
            .execute(&img, &t)
            .unwrap();
        for (got, want) in out
            .scalar_slice::<f32>()
            .unwrap()
            .iter()
            .zip(img.scalar_slice::<f32>().unwrap())
        {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }

    #[test]
    fn gaussian_interpolation_smooths_toward_default_outside() {
        // Smoke test for the Gaussian dispatch branch: a point mapped well
        // outside the input buffer (beyond the kernel's own cutoff radius,
        // so `gaussian_value_and_gradient` returns `None`) falls back to the
        // default pixel value, exactly like the other three kernels.
        let img = ramp_2d(4, 1);
        let t = TranslationTransform::new(vec![100.0, 0.0]);
        let out = ResampleImageFilter::new()
            .set_reference_image(&img)
            .set_default_pixel_value(-7.0)
            .set_interpolator(Interpolator::Gaussian)
            .execute(&img, &t)
            .unwrap();
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[-7.0, -7.0, -7.0, -7.0]
        );
    }

    #[test]
    fn windowed_sinc_interpolation_reproduces_ramp_on_identity_for_every_window() {
        // Smoke test for all five windowed-sinc dispatch branches: on an
        // identity transform every output sample lands on an integer input
        // index, which every window reproduces exactly (ITK's delta-weight
        // branch at `distance == 0`; bit-exact check lives in
        // `interpolator`'s own tests).
        let img = ramp_2d(6, 1); // 0,1,2,3,4,5
        let t = AffineTransform::identity(2);
        for interp in [
            Interpolator::HammingWindowedSinc,
            Interpolator::CosineWindowedSinc,
            Interpolator::WelchWindowedSinc,
            Interpolator::LanczosWindowedSinc,
            Interpolator::BlackmanWindowedSinc,
        ] {
            let out = ResampleImageFilter::new()
                .set_reference_image(&img)
                .set_interpolator(interp)
                .execute(&img, &t)
                .unwrap();
            assert_eq!(
                out.scalar_slice::<f32>().unwrap(),
                img.scalar_slice::<f32>().unwrap(),
                "{interp:?}"
            );
        }
    }

    #[test]
    fn windowed_sinc_interpolation_falls_back_to_default_outside_for_every_window() {
        // Mirrors `gaussian_interpolation_smooths_toward_default_outside`: a
        // point mapped well outside the input buffer makes
        // `windowed_sinc_value_and_gradient` return `None` (via the shared
        // `is_inside` gate), falling back to the default pixel value.
        let img = ramp_2d(4, 1);
        let t = TranslationTransform::new(vec![100.0, 0.0]);
        for interp in [
            Interpolator::HammingWindowedSinc,
            Interpolator::CosineWindowedSinc,
            Interpolator::WelchWindowedSinc,
            Interpolator::LanczosWindowedSinc,
            Interpolator::BlackmanWindowedSinc,
        ] {
            let out = ResampleImageFilter::new()
                .set_reference_image(&img)
                .set_default_pixel_value(-7.0)
                .set_interpolator(interp)
                .execute(&img, &t)
                .unwrap();
            assert_eq!(
                out.scalar_slice::<f32>().unwrap(),
                &[-7.0, -7.0, -7.0, -7.0],
                "{interp:?}"
            );
        }
    }
}
