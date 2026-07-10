//! The two `itk::ImageFunction`s the Demons PDE evaluates on the fixed and
//! moving images: `LinearInterpolateImageFunction` and
//! `CentralDifferenceImageFunction`.
//!
//! Both are ported against a scalar image widened to `f64`, which is what
//! `DemonsRegistrationFunction` asks of them (`static_cast<double>` on every
//! read, `CoordinateType = double`).

use sitk_core::{Error, Image, PixelId, matrix};

use crate::Result;

/// A scalar image widened to `f64`, with its geometry and the precomputed
/// inverse of its direction matrix.
///
/// The inverse is hoisted out of the per-pixel loops; `Image`'s own
/// `physical_point_to_continuous_index` inverts on every call and returns a
/// `Result`, neither of which belongs inside `ComputeUpdate`.
pub(crate) struct RealImage {
    data: Vec<f64>,
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    /// Row-major `dim x dim`, as `Image::direction`.
    direction: Vec<f64>,
    /// Row-major inverse of `direction`.
    inverse_direction: Vec<f64>,
    strides: Vec<usize>,
    /// The image's own (scalar) pixel type, kept because
    /// `ESMDemonsRegistrationFunction` warps the moving image into an
    /// `itk::Image<MovingPixelType>` and both quantises to, and sentinels on,
    /// that type.
    pixel_id: PixelId,
}

impl RealImage {
    /// Widen a scalar image. Errors with `RequiresScalarPixelType` on a vector
    /// image (through `Image::to_f64_vec`'s guard) and with `SingularDirection`
    /// when the direction cosine matrix cannot be inverted — the same condition
    /// `TransformPhysicalPointToContinuousIndex` relies on.
    pub(crate) fn new(image: &Image) -> Result<Self> {
        let dim = image.dimension();
        let data = image.to_f64_vec()?;
        let inverse_direction =
            matrix::invert(image.direction(), dim).ok_or(Error::SingularDirection)?;
        let mut strides = vec![1usize; dim];
        for d in 1..dim {
            strides[d] = strides[d - 1] * image.size()[d - 1];
        }
        Ok(RealImage {
            data,
            size: image.size().to_vec(),
            spacing: image.spacing().to_vec(),
            origin: image.origin().to_vec(),
            direction: image.direction().to_vec(),
            inverse_direction,
            strides,
            pixel_id: image.pixel_id(),
        })
    }

    pub(crate) fn dimension(&self) -> usize {
        self.size.len()
    }

    pub(crate) fn spacing(&self) -> &[f64] {
        &self.spacing
    }

    pub(crate) fn size(&self) -> &[usize] {
        &self.size
    }

    pub(crate) fn pixel_id(&self) -> PixelId {
        self.pixel_id
    }

    /// The pixel at an in-bounds multi-index.
    pub(crate) fn at(&self, index: &[usize]) -> f64 {
        let offset: usize = index.iter().zip(&self.strides).map(|(&i, &s)| i * s).sum();
        self.data[offset]
    }

    fn at_signed(&self, index: &[i64]) -> f64 {
        let offset: usize = index
            .iter()
            .zip(&self.strides)
            .map(|(&i, &s)| i as usize * s)
            .sum();
        self.data[offset]
    }

    /// Row `i` of a row-major `dim x dim` matrix.
    fn row(matrix: &[f64], i: usize, dim: usize) -> &[f64] {
        &matrix[i * dim..i * dim + dim]
    }

    /// `ImageBase::TransformIndexToPhysicalPoint`:
    /// `p = origin + Direction * (spacing ⊙ index)`.
    pub(crate) fn index_to_physical_point(&self, index: &[usize]) -> Vec<f64> {
        let dim = self.dimension();
        self.origin
            .iter()
            .enumerate()
            .map(|(i, &origin)| {
                let rotated: f64 = Self::row(&self.direction, i, dim)
                    .iter()
                    .zip(index)
                    .zip(&self.spacing)
                    .map(|((&d, &idx), &spacing)| d * idx as f64 * spacing)
                    .sum();
                origin + rotated
            })
            .collect()
    }

    /// `ImageBase::TransformPhysicalPointToContinuousIndex`.
    pub(crate) fn physical_point_to_continuous_index(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        let diff: Vec<f64> = point
            .iter()
            .zip(&self.origin)
            .map(|(&p, &o)| p - o)
            .collect();
        self.spacing
            .iter()
            .enumerate()
            .map(|(i, &spacing)| {
                let unrotated: f64 = Self::row(&self.inverse_direction, i, dim)
                    .iter()
                    .zip(&diff)
                    .map(|(&m, &d)| m * d)
                    .sum();
                unrotated / spacing
            })
            .collect()
    }

    /// `ImageFunction::IsInsideBuffer(PointType)` (itkImageFunction.h:176-184),
    /// which forwards to the continuous-index overload at line 159-170.
    ///
    /// The buffer's continuous bounds are `[start - 0.5, end + 0.5)` per
    /// `ImageFunction::SetInputImage` (itkImageFunction.hxx:63-64), with `end =
    /// size - 1`. Note the asymmetry: the lower bound is inclusive and the
    /// upper is *exclusive*, so a point exactly half a pixel past the last
    /// pixel centre is outside while its mirror at the first is inside. The
    /// comparison is written as the negation of a positive test so that a
    /// `NaN` coordinate reports outside.
    pub(crate) fn is_inside_buffer(&self, point: &[f64]) -> bool {
        let cindex = self.physical_point_to_continuous_index(point);
        (0..self.dimension()).all(|d| {
            let end = self.size[d] as f64 - 1.0;
            cindex[d] >= -0.5 && cindex[d] < end + 0.5
        })
    }

    /// `LinearInterpolateImageFunction::Evaluate(PointType)`.
    ///
    /// The caller must have established [`RealImage::is_inside_buffer`]; ITK
    /// makes the same assumption ("no validity checking ... is done").
    ///
    /// ITK dispatches to `EvaluateOptimized` for `ImageDimension <= 3` — the
    /// only dimensions SimpleITK instantiates these filters at
    /// (itkLinearInterpolateImageFunction.h:126-300) — which folds one axis at
    /// a time as `a + (b - a) * t`. This is that fold, generalised to N
    /// dimensions; for `N <= 3` it is arithmetically identical, including
    /// floating-point association.
    ///
    /// Two clamps make the fold match the optimized path's early returns:
    ///
    /// * The far neighbour is clamped down to the last pixel, so an axis whose
    ///   `basei + 1` would pass `m_EndIndex` contributes nothing — the optimized
    ///   path's `if (basei[d] > m_EndIndex[d]) return ...` branches.
    /// * The fractional distance is clamped up to `0`. Inside the lower
    ///   half-pixel `basei` is clamped to the first pixel, leaving a *negative*
    ///   distance; ITK's `if (distance <= 0.) return val0` then drops that axis
    ///   rather than extrapolating below the first pixel value.
    ///
    /// # Deviation for `N > 3`
    ///
    /// At four dimensions and up ITK falls back to `EvaluateUnoptimized`
    /// (itkLinearInterpolateImageFunction.hxx:29-95), which has no `distance <=
    /// 0` guard: it weights the clamped corners by `1 - distance` and
    /// `distance` directly, so a negative distance *extrapolates* below the
    /// first pixel. SimpleITK never instantiates these filters above 3D. This
    /// port keeps the optimized path's clamp at every dimension, since the
    /// extrapolation is an artefact of the unoptimized fallback rather than
    /// intended behaviour.
    pub(crate) fn linear_interpolate(&self, point: &[f64]) -> f64 {
        let dim = self.dimension();
        let cindex = self.physical_point_to_continuous_index(point);

        let mut base = vec![0i64; dim];
        let mut distance = vec![0.0f64; dim];
        for d in 0..dim {
            base[d] = (cindex[d].floor() as i64).max(0);
            distance[d] = (cindex[d] - base[d] as f64).max(0.0);
        }

        // The 2^dim corner values, corner bit `d` selecting the far neighbour
        // along axis `d`.
        let corners = 1usize << dim;
        let mut values = Vec::with_capacity(corners);
        let mut neighbor = vec![0i64; dim];
        for corner in 0..corners {
            for d in 0..dim {
                neighbor[d] = if corner & (1 << d) != 0 {
                    (base[d] + 1).min(self.size[d] as i64 - 1)
                } else {
                    base[d]
                };
            }
            values.push(self.at_signed(&neighbor));
        }

        // Fold axis 0 first, as ITK does. Axis `d` is always bit 0 of the
        // current packing, so its two corners are adjacent.
        let mut width = corners;
        for &t in &distance {
            let half = width / 2;
            for k in 0..half {
                let lo = values[2 * k];
                let hi = values[2 * k + 1];
                values[k] = lo + (hi - lo) * t;
            }
            width = half;
        }
        values[0]
    }

    /// `CentralDifferenceImageFunction::EvaluateAtIndex`
    /// (itkCentralDifferenceImageFunction.hxx:106-154), the scalar
    /// specialisation, with `UseImageDirection` at its default `true`.
    ///
    /// A pixel on the first or last slice along an axis gets a zero derivative
    /// along that axis rather than a one-sided difference; an axis of extent
    /// `1` is therefore zero everywhere. The index-space derivative is then
    /// rotated into physical space by the direction matrix.
    pub(crate) fn central_difference_at_index(&self, index: &[usize]) -> Vec<f64> {
        self.local_vector_to_physical_vector(&self.central_difference_at_index_local(index))
    }

    /// `CentralDifferenceImageFunction::EvaluateAtIndex` with
    /// `UseImageDirectionOff()`, which leaves the derivative in index space —
    /// the setting `ESMDemonsRegistrationFunction` uses for both of its gradient
    /// calculators (itkESMDemonsRegistrationFunction.hxx:50, :53) because it
    /// rotates the summed gradient once, at the end, by the *fixed* image's
    /// direction.
    pub(crate) fn central_difference_at_index_local(&self, index: &[usize]) -> Vec<f64> {
        let dim = self.dimension();
        let mut derivative = vec![0.0f64; dim];
        let mut neighbor: Vec<i64> = index.iter().map(|&i| i as i64).collect();

        for d in 0..dim {
            // `index[dim] < start + 1 || index[dim] > start + size - 2`, in
            // signed arithmetic so that `size == 1` yields `-1` for the bound.
            let last = self.size[d] as i64 - 2;
            if neighbor[d] < 1 || neighbor[d] > last {
                derivative[d] = 0.0;
                continue;
            }
            neighbor[d] += 1;
            let plus = self.at_signed(&neighbor);
            neighbor[d] -= 2;
            let minus = self.at_signed(&neighbor);
            neighbor[d] += 1;
            derivative[d] = (plus - minus) * 0.5 / self.spacing[d];
        }

        derivative
    }

    /// `CentralDifferenceImageFunction::Evaluate(PointType)`, the scalar
    /// specialisation (itkCentralDifferenceImageFunction.hxx:250-317), with
    /// `UseImageDirection` at its default `true`, in which case the result is
    /// returned unrotated.
    ///
    /// # Upstream quirk reproduced here
    ///
    /// The two sample points are offset along the *physical* axis `d` by
    /// `0.5 * spacing[d]` — the spacing of *index* axis `d`. When the direction
    /// matrix is not a permutation of the identity these are different axes, so
    /// the step length along a physical axis is borrowed from an unrelated
    /// index axis. ITK's own comment concedes only that "the image direction
    /// may swap dimensions". Reproduced as written.
    ///
    /// A sample point that leaves the buffer zeroes that component, matching
    /// `EvaluateAtIndex`'s boundary rule rather than falling back to a one-sided
    /// difference.
    pub(crate) fn central_difference_at_point(&self, point: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        let mut derivative = vec![0.0f64; dim];
        let mut lower = point.to_vec();
        let mut upper = point.to_vec();

        for d in 0..dim {
            let offset = 0.5 * self.spacing[d];

            lower[d] = point[d] - offset;
            if !self.is_inside_buffer(&lower) {
                derivative[d] = 0.0;
                lower[d] = point[d];
                upper[d] = point[d];
                continue;
            }
            upper[d] = point[d] + offset;
            if !self.is_inside_buffer(&upper) {
                derivative[d] = 0.0;
                lower[d] = point[d];
                upper[d] = point[d];
                continue;
            }

            let delta = upper[d] - lower[d];
            derivative[d] = if delta > 10.0 * f64::EPSILON {
                (self.linear_interpolate(&upper) - self.linear_interpolate(&lower)) / delta
            } else {
                0.0
            };

            lower[d] = point[d];
            upper[d] = point[d];
        }

        // `UseImageDirection` defaults to true, so the derivative — already in
        // physical axes — is returned as is.
        derivative
    }

    /// `CentralDifferenceImageFunction::Evaluate(PointType)` with
    /// `UseImageDirectionOff()`: the physical-space derivative is rotated back
    /// into index space by the inverse direction
    /// (itkCentralDifferenceImageFunction.hxx:311-316). This is what
    /// `ESMDemonsRegistrationFunction`'s `MappedMoving` gradient evaluates on
    /// the *moving* image before the *fixed* image's direction is applied to the
    /// result.
    pub(crate) fn central_difference_at_point_local(&self, point: &[f64]) -> Vec<f64> {
        self.physical_vector_to_local_vector(&self.central_difference_at_point(point))
    }

    /// `ImageBase::TransformLocalVectorToPhysicalVector`
    /// (itkImageBase.h:636-654): `out = Direction * in`.
    pub(crate) fn local_vector_to_physical_vector(&self, local: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        (0..dim)
            .map(|i| {
                Self::row(&self.direction, i, dim)
                    .iter()
                    .zip(local)
                    .map(|(&d, &l)| d * l)
                    .sum()
            })
            .collect()
    }

    /// `ImageBase::TransformPhysicalVectorToLocalVector`
    /// (itkImageBase.h:685-702): `out = Direction⁻¹ * in`.
    fn physical_vector_to_local_vector(&self, physical: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        (0..dim)
            .map(|i| {
                Self::row(&self.inverse_direction, i, dim)
                    .iter()
                    .zip(physical)
                    .map(|(&m, &p)| m * p)
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_2d() -> RealImage {
        // 3x3, value == x + 10*y.
        let mut data = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                data.push((x + 10 * y) as f64);
            }
        }
        RealImage::new(&Image::from_vec(&[3, 3], data).unwrap()).unwrap()
    }

    #[test]
    fn at_reads_first_index_fastest() {
        let img = ramp_2d();
        assert_eq!(img.at(&[0, 0]), 0.0);
        assert_eq!(img.at(&[2, 0]), 2.0);
        assert_eq!(img.at(&[0, 2]), 20.0);
        assert_eq!(img.at(&[1, 2]), 21.0);
    }

    /// The continuous buffer bounds are `[-0.5, size - 0.5)`: inclusive below,
    /// exclusive above.
    #[test]
    fn is_inside_buffer_is_half_open_at_the_upper_bound() {
        let img = ramp_2d();
        // size 3 → end index 2 → bounds [-0.5, 2.5)
        assert!(img.is_inside_buffer(&[-0.5, 0.0]));
        assert!(!img.is_inside_buffer(&[-0.5001, 0.0]));
        assert!(img.is_inside_buffer(&[2.4999, 0.0]));
        assert!(!img.is_inside_buffer(&[2.5, 0.0]));
    }

    #[test]
    fn is_inside_buffer_rejects_nan() {
        let img = ramp_2d();
        assert!(!img.is_inside_buffer(&[f64::NAN, 0.0]));
    }

    #[test]
    fn linear_interpolate_at_pixel_centers_is_exact() {
        let img = ramp_2d();
        assert_eq!(img.linear_interpolate(&[1.0, 2.0]), 21.0);
        assert_eq!(img.linear_interpolate(&[0.0, 0.0]), 0.0);
    }

    /// A ramp is reproduced exactly by bilinear interpolation.
    #[test]
    fn linear_interpolate_on_a_ramp() {
        let img = ramp_2d();
        assert!((img.linear_interpolate(&[0.5, 0.0]) - 0.5).abs() < 1e-12);
        assert!((img.linear_interpolate(&[0.0, 0.5]) - 5.0).abs() < 1e-12);
        assert!((img.linear_interpolate(&[0.5, 0.5]) - 5.5).abs() < 1e-12);
        assert!((img.linear_interpolate(&[1.25, 1.5]) - 16.25).abs() < 1e-12);
    }

    /// Past the last pixel centre the far neighbour clamps onto the last pixel,
    /// so the value is held rather than extrapolated — the optimized path's
    /// `basei[d] > m_EndIndex[d]` early return.
    #[test]
    fn linear_interpolate_holds_the_value_past_the_last_pixel_center() {
        let img = ramp_2d();
        assert_eq!(img.linear_interpolate(&[2.25, 0.0]), 2.0);
        assert_eq!(img.linear_interpolate(&[0.0, 2.25]), 20.0);
    }

    /// Below the first pixel centre `basei` clamps to `0`, leaving a negative
    /// distance that ITK's `if (distance <= 0.) return val0` drops. The value is
    /// held at the first pixel, *not* extrapolated to `-0.25`.
    #[test]
    fn linear_interpolate_holds_the_value_below_the_first_pixel_center() {
        let img = ramp_2d();
        assert_eq!(img.linear_interpolate(&[-0.25, 0.0]), 0.0);
        assert_eq!(img.linear_interpolate(&[0.0, -0.25]), 0.0);
        // Mixed: x extrapolation suppressed, y still interpolates.
        assert_eq!(img.linear_interpolate(&[-0.25, 0.5]), 5.0);
    }

    /// `d/dx (x + 10y) = 1`, `d/dy = 10`, with zeros on the border slices.
    #[test]
    fn central_difference_at_index_on_a_ramp() {
        let img = ramp_2d();
        assert_eq!(img.central_difference_at_index(&[1, 1]), vec![1.0, 10.0]);
        // border along x → x-derivative zero, y-derivative still valid
        assert_eq!(img.central_difference_at_index(&[0, 1]), vec![0.0, 10.0]);
        assert_eq!(img.central_difference_at_index(&[2, 1]), vec![0.0, 10.0]);
        // border along y
        assert_eq!(img.central_difference_at_index(&[1, 0]), vec![1.0, 0.0]);
        // corner
        assert_eq!(img.central_difference_at_index(&[0, 0]), vec![0.0, 0.0]);
    }

    /// An axis of extent 1 is a boundary at every index.
    #[test]
    fn central_difference_at_index_zeroes_a_degenerate_axis() {
        let img = RealImage::new(&Image::from_vec(&[1, 3], vec![0.0, 1.0, 2.0]).unwrap()).unwrap();
        assert_eq!(img.central_difference_at_index(&[0, 1]), vec![0.0, 1.0]);
    }

    /// Non-unit spacing divides the difference.
    #[test]
    fn central_difference_at_index_divides_by_spacing() {
        let mut image = Image::from_vec(&[3, 3], (0..9).map(f64::from).collect()).unwrap();
        image.set_spacing(&[2.0, 4.0]).unwrap();
        let img = RealImage::new(&image).unwrap();
        // value == x + 3y; d/dx = 1/2, d/dy = 3/4
        assert_eq!(img.central_difference_at_index(&[1, 1]), vec![0.5, 0.75]);
    }

    /// With a 90-degree direction matrix the index-space derivative is rotated
    /// into physical space.
    #[test]
    fn central_difference_at_index_applies_the_direction_matrix() {
        let mut image = Image::from_vec(&[3, 3], (0..9).map(f64::from).collect()).unwrap();
        // Rotate index axes: physical x = -index y, physical y = index x.
        image.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let img = RealImage::new(&image).unwrap();
        // local derivative is (1, 3); Direction * (1,3) == (-3, 1)
        assert_eq!(img.central_difference_at_index(&[1, 1]), vec![-3.0, 1.0]);
    }

    /// On a ramp with unit spacing the physical-point central difference
    /// reproduces the analytic gradient at an interior point.
    #[test]
    fn central_difference_at_point_on_a_ramp() {
        let img = ramp_2d();
        let g = img.central_difference_at_point(&[1.0, 1.0]);
        assert!((g[0] - 1.0).abs() < 1e-12);
        assert!((g[1] - 10.0).abs() < 1e-12);
    }

    /// A sample point that leaves the buffer zeroes that component. At `x = 2.4`
    /// the upper sample sits at `2.9`, past the `2.5` bound.
    #[test]
    fn central_difference_at_point_zeroes_an_out_of_buffer_axis() {
        let img = ramp_2d();
        let g = img.central_difference_at_point(&[2.4, 1.0]);
        assert_eq!(g[0], 0.0);
        assert!((g[1] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn index_to_physical_point_applies_origin_spacing_and_direction() {
        let mut image = Image::from_vec(&[3, 3], vec![0.0; 9]).unwrap();
        image.set_spacing(&[2.0, 3.0]).unwrap();
        image.set_origin(&[10.0, 20.0]).unwrap();
        let img = RealImage::new(&image).unwrap();
        assert_eq!(img.index_to_physical_point(&[1, 2]), vec![12.0, 26.0]);
    }

    #[test]
    fn physical_point_to_continuous_index_inverts_index_to_physical_point() {
        let mut image = Image::from_vec(&[3, 3], vec![0.0; 9]).unwrap();
        image.set_spacing(&[2.0, 3.0]).unwrap();
        image.set_origin(&[10.0, 20.0]).unwrap();
        image.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let img = RealImage::new(&image).unwrap();
        let p = img.index_to_physical_point(&[1, 2]);
        let c = img.physical_point_to_continuous_index(&p);
        assert!((c[0] - 1.0).abs() < 1e-12);
        assert!((c[1] - 2.0).abs() < 1e-12);
    }

    /// `UseImageDirectionOff()` leaves `EvaluateAtIndex`'s derivative in index
    /// space, so the same image that rotates to `(-3, 1)` reports `(1, 3)`.
    #[test]
    fn central_difference_at_index_local_skips_the_direction_matrix() {
        let mut image = Image::from_vec(&[3, 3], (0..9).map(f64::from).collect()).unwrap();
        image.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let img = RealImage::new(&image).unwrap();
        assert_eq!(
            img.central_difference_at_index_local(&[1, 1]),
            vec![1.0, 3.0]
        );
    }

    /// `UseImageDirectionOff()` on the point overload rotates the physical
    /// derivative back by `Direction⁻¹`, recovering the index-space derivative.
    #[test]
    fn central_difference_at_point_local_undoes_the_direction_matrix() {
        let mut image = Image::from_vec(&[3, 3], (0..9).map(f64::from).collect()).unwrap();
        // physical x = -index y, physical y = index x; the pixel centre of
        // index (1, 1) is then at physical (-1, 1).
        image.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let img = RealImage::new(&image).unwrap();
        let g = img.central_difference_at_point_local(&[-1.0, 1.0]);
        assert!((g[0] - 1.0).abs() < 1e-12, "{g:?}");
        assert!((g[1] - 3.0).abs() < 1e-12, "{g:?}");
    }

    #[test]
    fn real_image_rejects_a_vector_image() {
        let v = Image::from_vec_vector(&[2], 2, vec![0.0f64; 4]).unwrap();
        assert!(RealImage::new(&v).is_err());
    }
}
