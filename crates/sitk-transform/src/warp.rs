//! `itk::WarpImageFilter`
//! (`Modules/Filtering/ImageGrid/include/itkWarpImageFilter.h(.hxx)`): warp a
//! scalar image by a dense displacement field.
//!
//! The mapping is *inverse*: for each output pixel, its physical point `p` is
//! displaced by the field's value there and the **input** image is interpolated
//! at `p + d`, so `p_in = p_out + d` (`WarpImageFilter.yaml`'s
//! detaileddescription). Nothing is scattered forward, so the output has no
//! holes.
//!
//! # Why this lives in `sitk-transform` and not `sitk-filters`
//!
//! `WarpImageFilter` takes no [`Transform`](crate::Transform) — its second
//! input is a displacement-field *image*. On that count it would belong in
//! `sitk-filters`. But it needs `sitkCreateInterpolator.hxx`'s full
//! `InterpolatorEnum` (`WarpImageFilter.yaml`'s `Interpolator` member), which
//! in this workspace is [`Interpolator`] plus the kernels in
//! [`crate::interpolator`]; and `sitk-filters` must not depend on
//! `sitk-transform` (the dependency runs the other way round, and
//! `sitk-registration` depends on both). Duplicating nine interpolation kernels
//! to honour a crate boundary would be the worse trade, so warping sits beside
//! [`ResampleImageFilter`](crate::ResampleImageFilter) and shares its
//! `InterpolatedImage` sampler verbatim.

use sitk_core::{Error, Image, matrix};

use crate::error::{Result, TransformError};
use crate::interpolator::{
    affine_apply, index_to_physical_matrix, physical_to_index_matrix, strides,
};
use crate::resample::{InterpolatedImage, Interpolator, build_output, increment};

/// `WarpImageFilter`: resample `image` at `p + d(p)` for every output physical
/// point `p`, where `d` is a displacement-field image.
///
/// # Output grid
///
/// Unlike [`ResampleImageFilter`](crate::ResampleImageFilter), whose reference
/// geometry defaults to the *input image*'s, this filter's defaults are ITK's
/// hardcoded ones (itkWarpImageFilter.hxx:43-48): unit spacing, zero origin,
/// identity direction — none of them borrowed from either input. The size
/// defaults to the displacement field's ("The LargestPossibleRegion for the
/// output is inherited from the input displacement field",
/// `WarpImageFilter.yaml`). `SetOutputParametersFromImage` /
/// [`set_output_parameters_from_image`](Self::set_output_parameters_from_image)
/// is how a caller opts into a reference image's geometry.
///
/// ITK keys "size unset" on `m_OutputSize[0] == 0`
/// (itkWarpImageFilter.hxx:428) — only the *first* axis — so an explicit
/// `[0, 5, 5]` also falls back to the field's whole size. That quirk is
/// reproduced.
///
/// # Out-of-domain points
///
/// `p + d` mapped outside the input image's buffer takes
/// [`EdgePaddingValue`](Self::set_edge_padding_value) (default `0`). Inside is
/// ITK's `InterpolateImageFunction::IsInsideBuffer`, the `[-0.5, size - 0.5)`
/// pixel-centred coverage that [`crate::interpolator::is_inside`] implements.
/// Note that this test is on the *input* image, not on the field: a point whose
/// displacement had to be extrapolated off the edge of the field is not
/// specially marked.
pub struct WarpImageFilter {
    interpolator: Interpolator,
    output_size: Option<Vec<usize>>,
    output_spacing: Option<Vec<f64>>,
    output_origin: Option<Vec<f64>>,
    output_direction: Option<Vec<f64>>,
    edge_padding_value: f64,
}

impl Default for WarpImageFilter {
    fn default() -> Self {
        Self {
            interpolator: Interpolator::Linear,
            output_size: None,
            output_spacing: None,
            output_origin: None,
            output_direction: None,
            edge_padding_value: 0.0,
        }
    }
}

impl WarpImageFilter {
    /// A filter with ITK's defaults: linear interpolation, zero edge padding,
    /// unit spacing, zero origin, identity direction, and the displacement
    /// field's size.
    pub fn new() -> Self {
        Self::default()
    }

    /// Choose the interpolation kernel applied to the *input image*
    /// (`sitkLinear` by default). The displacement field itself is always
    /// sampled with ITK's own edge-clamping linear interpolation, which is
    /// hardcoded in `EvaluateDisplacementAtPhysicalPoint` and not configurable
    /// upstream either.
    pub fn set_interpolator(&mut self, interpolator: Interpolator) -> &mut Self {
        self.interpolator = interpolator;
        self
    }

    /// Override the output size (default: the displacement field's).
    pub fn set_output_size(&mut self, size: Vec<usize>) -> &mut Self {
        self.output_size = Some(size);
        self
    }

    /// Override the output spacing (default: 1 per axis).
    pub fn set_output_spacing(&mut self, spacing: Vec<f64>) -> &mut Self {
        self.output_spacing = Some(spacing);
        self
    }

    /// Override the output origin (default: 0 per axis).
    pub fn set_output_origin(&mut self, origin: Vec<f64>) -> &mut Self {
        self.output_origin = Some(origin);
        self
    }

    /// Override the output direction (row-major `dim x dim`; default:
    /// identity).
    pub fn set_output_direction(&mut self, direction: Vec<f64>) -> &mut Self {
        self.output_direction = Some(direction);
        self
    }

    /// Value written where `p + d(p)` falls outside the input image's buffer.
    /// SimpleITK's member is a `double` cast to the input's pixel type
    /// (`WarpImageFilter.yaml`: `type: double`, `pixeltype: Input`).
    pub fn set_edge_padding_value(&mut self, value: f64) -> &mut Self {
        self.edge_padding_value = value;
        self
    }

    /// Take the output size, origin, spacing, and direction from a reference
    /// image — SimpleITK's `SetOutputParameteresFromImage` custom method
    /// (whose upstream name carries that typo).
    pub fn set_output_parameters_from_image(&mut self, reference: &Image) -> &mut Self {
        self.output_size = Some(reference.size().to_vec());
        self.output_spacing = Some(reference.spacing().to_vec());
        self.output_origin = Some(reference.origin().to_vec());
        self.output_direction = Some(reference.direction().to_vec());
        self
    }

    /// Warp `image` by `displacement_field`.
    ///
    /// `image` must be scalar ([`Image::to_f64_vec`] is the guard);
    /// `displacement_field` must be a vector image of the same dimension with
    /// exactly `image.dimension()` components per pixel, which is
    /// `VerifyInputInformation`'s check (itkWarpImageFilter.hxx:103-109).
    ///
    /// SimpleITK's `pixel_types2: RealVectorPixelIDTypeList`
    /// (sitkPixelIDTypeLists.h:143) narrows the field further, to
    /// `VectorFloat32` and `VectorFloat64`. That is an *instantiation* list, not
    /// a run-time check: ITK's own template accepts any vector component type,
    /// and this port reads the field through `components_to_f64_vec`, so an
    /// integer-component field warps by the same arithmetic. Only the
    /// vector-ness is enforced.
    pub fn execute(&self, image: &Image, displacement_field: &Image) -> Result<Image> {
        let dim = image.dimension();
        let field = displacement_field;

        if field.dimension() != dim {
            return Err(TransformError::DimensionMismatch);
        }
        if !field.pixel_id().is_vector() {
            return Err(TransformError::Core(Error::RequiresVectorPixelType(
                field.pixel_id(),
            )));
        }
        if field.number_of_components_per_pixel() != dim {
            return Err(TransformError::DisplacementFieldComponentMismatch {
                expected: dim,
                got: field.number_of_components_per_pixel(),
            });
        }
        if field.number_of_pixels() == 0 {
            return Err(TransformError::InvalidDisplacementFieldDomain);
        }

        // `m_OutputSize[0] == 0` — the first axis alone — means "inherit the
        // field's LargestPossibleRegion" (itkWarpImageFilter.hxx:426-436).
        let out_size = match &self.output_size {
            Some(s) if s.first() != Some(&0) => s.clone(),
            _ => field.size().to_vec(),
        };
        let out_spacing = self
            .output_spacing
            .clone()
            .unwrap_or_else(|| vec![1.0; dim]);
        let out_origin = self.output_origin.clone().unwrap_or_else(|| vec![0.0; dim]);
        let out_direction = self
            .output_direction
            .clone()
            .unwrap_or_else(|| matrix::identity(dim));

        if out_size.len() != dim
            || out_spacing.len() != dim
            || out_origin.len() != dim
            || out_direction.len() != dim * dim
        {
            return Err(TransformError::DimensionMismatch);
        }

        let out_index_to_phys = index_to_physical_matrix(&out_direction, &out_spacing, dim);
        let in_phys_to_index = physical_to_index_matrix(image.direction(), image.spacing(), dim)
            .ok_or(TransformError::SingularDirection)?;
        let in_origin = image.origin().to_vec();
        let sampler = InterpolatedImage::new(image, self.interpolator)?;

        let field_buf = field.components_to_f64_vec();
        let field_reader = FieldReader {
            buf: &field_buf,
            strides: strides(field.size()),
            size: field.size().to_vec(),
            dim,
        };
        let field_phys_to_index = physical_to_index_matrix(field.direction(), field.spacing(), dim)
            .ok_or(TransformError::SingularDirection)?;
        let field_origin = field.origin().to_vec();

        let same_information =
            same_information(&out_size, &out_spacing, &out_origin, &out_direction, field);

        let n_out: usize = out_size.iter().product();
        let mut out_vals = vec![0.0f64; n_out];
        let mut index = vec![0usize; dim];
        let mut displacement = vec![0.0f64; dim];

        for (pixel, out_val) in out_vals.iter_mut().enumerate() {
            let index_f: Vec<f64> = index.iter().map(|&i| i as f64).collect();
            let mut point = affine_apply(&out_index_to_phys, &index_f, &out_origin, dim);

            if same_information {
                // `m_DefFieldSameInformation`: output index == field index, so
                // the field pixel is read straight out, unsampled.
                displacement.copy_from_slice(&field_buf[pixel * dim..(pixel + 1) * dim]);
            } else {
                let diff: Vec<f64> = (0..dim).map(|d| point[d] - field_origin[d]).collect();
                let cindex = matrix::mat_vec(&field_phys_to_index, &diff, dim);
                field_reader.evaluate(&cindex, &mut displacement);
            }

            for (d, p) in point.iter_mut().enumerate() {
                *p += displacement[d];
            }

            let diff: Vec<f64> = (0..dim).map(|d| point[d] - in_origin[d]).collect();
            let cindex = matrix::mat_vec(&in_phys_to_index, &diff, dim);
            *out_val = sampler.sample(&cindex).unwrap_or(self.edge_padding_value);

            increment(&mut index, &out_size);
        }

        // `TOutputImage` is `TInputImage` for SimpleITK's instantiation, so the
        // output — the interpolated value and the edge padding alike — is a
        // `static_cast<PixelType>` of a `double`.
        let mut result = build_output(image.pixel_id(), &out_size, out_vals)?;
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

/// `GenerateInputRequestedRegion`'s `m_DefFieldSameInformation`
/// (itkWarpImageFilter.hxx:384-392): output and field agree on origin, spacing,
/// and direction to within ITK's tolerances, so each output pixel's
/// displacement is the field pixel at the same index — no interpolation, no
/// physical-point round trip.
///
/// # Deviation: the size is compared too
///
/// ITK compares only the three geometry vectors. When they agree but the output
/// is *larger* than the field, the fast path constructs
/// `ImageRegionConstIterator fieldIt(fieldPtr, outputRegionForThread)`
/// (itkWarpImageFilter.hxx:294) over a region exceeding the field's buffer —
/// `GenerateInputRequestedRegion` has already fallen back to the field's
/// largest possible region when the output's requested region failed to verify
/// (hxx:407-410), silencing the pipeline's own check. ITK does **not** read out
/// of bounds there: the iterator constructor's containment assert
/// (itkImageConstIterator.h:207-212) throws in release builds and aborts in
/// debug. The upstream defect is therefore a spurious failure — a valid
/// configuration dies with an internal iterator error instead of being warped.
/// Adding the size to the test costs nothing on the path ITK actually intends
/// (where the output *is* the field's region, the sizes are equal by
/// construction) and sends the mismatched case down the general path, where the
/// field is sampled at physical points and clamped at its border — so this port
/// **produces output where ITK throws**. Reported upstream (ITK #6575, item
/// B34).
fn same_information(
    out_size: &[usize],
    out_spacing: &[f64],
    out_origin: &[f64],
    out_direction: &[f64],
    field: &Image,
) -> bool {
    // itkImageBase.h:52-53: DefaultImageCoordinateTolerance and
    // DefaultImageDirectionTolerance are both 1e-6; the coordinate tolerance is
    // scaled by the output's first-axis spacing (hxx:384-385).
    const TOL: f64 = 1e-6;
    let coordinate_tol = TOL * out_spacing[0];
    let close = |a: &[f64], b: &[f64], tol: f64| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol);

    out_size == field.size()
        && close(out_origin, field.origin(), coordinate_tol)
        && close(out_spacing, field.spacing(), coordinate_tol)
        && close(out_direction, field.direction(), TOL)
}

/// The displacement field, sampled by `EvaluateDisplacementAtPhysicalPoint`
/// (itkWarpImageFilter.hxx:181-267).
struct FieldReader<'a> {
    buf: &'a [f64],
    size: Vec<usize>,
    strides: Vec<usize>,
    dim: usize,
}

impl FieldReader<'_> {
    /// N-linear interpolation of the field at a continuous index, **clamped**
    /// rather than bounded: an index below the field's start or at/above its
    /// last pixel collapses that axis onto the border pixel with zero
    /// fractional distance, so the field is extended by edge replication
    /// instead of being reported as out of domain. ITK has no inside-buffer
    /// test here at all — the output pixel is still produced, and only the
    /// subsequent input-image probe can send it to the edge padding value.
    ///
    /// The corner accumulation reproduces upstream exactly, including the
    /// `if (overlap)` guard that both skips the zero-weight corners and keeps
    /// the `baseIndex + 1` neighbour of a clamped axis from ever being read,
    /// and the `totalOverlap == 1.0` early exit.
    fn evaluate(&self, cindex: &[f64], out: &mut [f64]) {
        let dim = self.dim;
        let mut base = vec![0usize; dim];
        let mut distance = vec![0.0f64; dim];

        for d in 0..dim {
            // `m_StartIndex` is 0 and `m_EndIndex` is `size - 1`
            // (hxx:153-157); this crate's images have no non-zero start index.
            let end = self.size[d] - 1;
            let floor = cindex[d].floor();
            // `floor >= 0.0` is false for NaN too, which then clamps to the
            // start index. C++'s `Math::Floor<IndexValueType>` of a NaN or of a
            // value past `IndexValueType`'s range is undefined; clamping in the
            // float domain, before the cast, is defined and agrees with
            // upstream on every input upstream defines.
            if floor >= 0.0 {
                if floor < end as f64 {
                    base[d] = floor as usize;
                    distance[d] = cindex[d] - floor;
                } else {
                    base[d] = end;
                }
            }
        }

        out.fill(0.0);
        let mut total_overlap = 0.0;
        for counter in 0..(1usize << dim) {
            let mut overlap = 1.0;
            let mut offset = 0usize;
            for d in 0..dim {
                if (counter >> d) & 1 == 1 {
                    offset += (base[d] + 1) * self.strides[d];
                    overlap *= distance[d];
                } else {
                    offset += base[d] * self.strides[d];
                    overlap *= 1.0 - distance[d];
                }
            }

            if overlap != 0.0 {
                let pixel = &self.buf[offset * dim..(offset + 1) * dim];
                for (k, o) in out.iter_mut().enumerate() {
                    *o += overlap * pixel[k];
                }
                total_overlap += overlap;
            }

            if total_overlap == 1.0 {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    /// A 4x4 ramp: pixel (i, j) holds `i + 4 * j`.
    fn ramp() -> Image {
        Image::from_vec(&[4, 4], (0..16).map(|v| v as f32).collect::<Vec<f32>>()).unwrap()
    }

    /// A field of `size` pixels, every pixel holding `d`.
    fn constant_field(size: &[usize], d: &[f64]) -> Image {
        let n: usize = size.iter().product();
        let data: Vec<f64> = std::iter::repeat_n(d.iter().copied(), n)
            .flatten()
            .collect();
        Image::from_vec_vector(size, d.len(), data).unwrap()
    }

    /// Zero displacement over the input's own grid is the identity: warp
    /// reproduces the input exactly. (The input's grid *is* the default output
    /// grid here — unit spacing, zero origin, identity direction.)
    #[test]
    fn zero_field_reproduces_the_input() {
        let img = ramp();
        let field = constant_field(&[4, 4], &[0.0, 0.0]);
        let out = WarpImageFilter::new().execute(&img, &field).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.size(), &[4, 4]);
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            img.scalar_slice::<f32>().unwrap()
        );
    }

    /// `p_in = p_out + d`: a `+1` x-displacement pulls the pixel one to the
    /// *right* of each output pixel, shifting the image content left. The last
    /// column probes `x = 4`, outside `[-0.5, 3.5)`, and takes the padding.
    #[test]
    fn constant_translation_field_shifts_the_image() {
        let img = ramp();
        let field = constant_field(&[4, 4], &[1.0, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_edge_padding_value(-1.0);
        let out = f.execute(&img, &field).unwrap();
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[
                1.0, 2.0, 3.0, -1.0, //
                5.0, 6.0, 7.0, -1.0, //
                9.0, 10.0, 11.0, -1.0, //
                13.0, 14.0, 15.0, -1.0,
            ]
        );
    }

    /// The same shift along y.
    #[test]
    fn constant_translation_field_shifts_along_y() {
        let img = ramp();
        let field = constant_field(&[4, 4], &[0.0, -1.0]);
        let mut f = WarpImageFilter::new();
        f.set_edge_padding_value(-1.0);
        let out = f.execute(&img, &field).unwrap();
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[
                -1.0, -1.0, -1.0, -1.0, // y = 0 probes y = -1
                0.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0,
            ]
        );
    }

    /// A half-pixel displacement interpolates: output (i, j) = input (i + 0.5,
    /// j) = mean of its two x-neighbours. Pixel (3, j) probes x = 3.5, which is
    /// outside `[-0.5, 3.5)`.
    #[test]
    fn half_pixel_field_interpolates_linearly() {
        let img = Image::from_vec(&[4, 1], vec![0.0f32, 10.0, 20.0, 30.0]).unwrap();
        let field = constant_field(&[4, 1], &[0.5, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_edge_padding_value(-1.0);
        let out = f.execute(&img, &field).unwrap();
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[5.0, 15.0, 25.0, -1.0]);
    }

    /// The edge padding value is cast to the input's pixel type, like every
    /// other output pixel.
    #[test]
    fn edge_padding_value_is_cast_to_the_input_pixel_type() {
        let img = Image::from_vec(&[2, 1], vec![7u8, 8]).unwrap();
        let field = constant_field(&[2, 1], &[10.0, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_edge_padding_value(300.0);
        let out = f.execute(&img, &field).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
        // Everything probes far outside; 300 saturates to u8::MAX (C++'s
        // out-of-range static_cast is undefined; this port saturates).
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[255, 255]);
    }

    /// Zero edge padding is the default.
    #[test]
    fn default_edge_padding_is_zero() {
        let img = ramp();
        let field = constant_field(&[4, 4], &[100.0, 0.0]);
        let out = WarpImageFilter::new().execute(&img, &field).unwrap();
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[0.0; 16]);
    }

    /// The output size defaults to the field's, not the input's.
    #[test]
    fn output_size_defaults_to_the_displacement_fields_size() {
        let img = ramp();
        let field = constant_field(&[2, 3], &[0.0, 0.0]);
        let out = WarpImageFilter::new().execute(&img, &field).unwrap();
        assert_eq!(out.size(), &[2, 3]);
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[0.0, 1.0, 4.0, 5.0, 8.0, 9.0]
        );
    }

    /// ITK keys "unset" on `m_OutputSize[0] == 0` alone, so `[0, 5]` also falls
    /// back to the field's size — the `5` is silently discarded.
    #[test]
    fn a_zero_first_axis_output_size_falls_back_to_the_field_size() {
        let img = ramp();
        let field = constant_field(&[2, 3], &[0.0, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_output_size(vec![0, 5]);
        assert_eq!(f.execute(&img, &field).unwrap().size(), &[2, 3]);
    }

    /// An explicit output size larger than the field: the field's geometry
    /// still matches the output's, but its extent does not, so the general
    /// (physical-point) path runs and clamps the field at its border. Here the
    /// field is constant, so the clamped displacement is the same everywhere.
    #[test]
    fn output_larger_than_the_field_clamps_the_field_at_its_border() {
        let img = ramp();
        let field = constant_field(&[2, 2], &[1.0, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_output_size(vec![4, 4]).set_edge_padding_value(-1.0);
        let out = f.execute(&img, &field).unwrap();
        assert_eq!(out.size(), &[4, 4]);
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[
                1.0, 2.0, 3.0, -1.0, //
                5.0, 6.0, 7.0, -1.0, //
                9.0, 10.0, 11.0, -1.0, //
                13.0, 14.0, 15.0, -1.0,
            ]
        );
    }

    /// The general path with a non-constant field: the field has *twice* the
    /// output's spacing, so output pixel `i` samples field continuous index
    /// `i / 2` and interpolates between field pixels.
    ///
    /// Field (spacing 2): d(0) = 0, d(1) = 2 at physical x = 0 and 2.
    /// Output pixel i sits at x = i, i.e. field cindex i/2, so
    /// `d = i`. Then `p_in = i + i = 2i`, and the input ramp gives `2i`.
    #[test]
    fn general_path_interpolates_the_field_between_pixels() {
        let img = Image::from_vec(&[4, 1], vec![0.0f32, 1.0, 2.0, 3.0]).unwrap();
        let mut field = Image::from_vec_vector(&[2, 1], 2, vec![0.0f64, 0.0, 2.0, 0.0]).unwrap();
        field.set_spacing(&[2.0, 1.0]).unwrap();

        let mut f = WarpImageFilter::new();
        f.set_output_size(vec![2, 1]).set_edge_padding_value(-1.0);
        let out = f.execute(&img, &field).unwrap();
        // i = 0: d = 0 -> input(0) = 0.  i = 1: d = 1 -> input(2) = 2.
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[0.0, 2.0]);
    }

    /// Off the end of the field, `EvaluateDisplacementAtPhysicalPoint` clamps
    /// to the border pixel rather than reporting the point out of domain: the
    /// output pixel is still produced, with the border displacement.
    #[test]
    fn the_field_is_extended_by_edge_replication() {
        let img = Image::from_vec(&[4, 1], vec![0.0f32, 10.0, 20.0, 30.0]).unwrap();
        // Field covers only x = 0 and x = 1; d(0) = 0, d(1) = 1.
        let field = Image::from_vec_vector(&[2, 1], 2, vec![0.0f64, 0.0, 1.0, 0.0]).unwrap();

        let mut f = WarpImageFilter::new();
        f.set_output_size(vec![4, 1]).set_edge_padding_value(-1.0);
        let out = f.execute(&img, &field).unwrap();
        // x = 0: d = 0 -> in(0) = 0.       x = 1: d = 1 -> in(2) = 20.
        // x = 2: clamped d = 1 -> in(3) = 30.
        // x = 3: clamped d = 1 -> in(4), outside -> padding.
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[0.0, 20.0, 30.0, -1.0]);
    }

    /// Below the field's start index the same clamp applies, from the other
    /// side.
    #[test]
    fn the_field_is_edge_replicated_below_its_start_too() {
        let img = Image::from_vec(&[4, 1], vec![0.0f32, 10.0, 20.0, 30.0]).unwrap();
        let mut field = Image::from_vec_vector(&[2, 1], 2, vec![1.0f64, 0.0, 0.0, 0.0]).unwrap();
        // Field starts at x = 2, so output x = 0 and 1 are below its start.
        field.set_origin(&[2.0, 0.0]).unwrap();

        let mut f = WarpImageFilter::new();
        f.set_output_size(vec![3, 1]).set_edge_padding_value(-1.0);
        let out = f.execute(&img, &field).unwrap();
        // x = 0: clamped to field pixel 0 -> d = 1 -> in(1) = 10.
        // x = 1: clamped -> d = 1 -> in(2) = 20.
        // x = 2: field pixel 0 exactly -> d = 1 -> in(3) = 30.
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[10.0, 20.0, 30.0]);
    }

    /// The output grid's own spacing/origin enter through the physical point.
    /// With output spacing 0.5 the output samples the input's physical x at
    /// `0, 0.5, 1, 1.5`, plus the zero field.
    #[test]
    fn output_spacing_and_origin_place_the_output_grid() {
        let img = Image::from_vec(&[4, 1], vec![0.0f32, 10.0, 20.0, 30.0]).unwrap();
        let field = constant_field(&[4, 1], &[0.0, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_output_spacing(vec![0.5, 1.0]);
        let out = f.execute(&img, &field).unwrap();
        assert_eq!(out.spacing(), &[0.5, 1.0]);
        // The field is *not* the same information any more (its spacing is 1),
        // so the general path samples it -- it is zero everywhere regardless.
        assert_eq!(out.scalar_slice::<f32>().unwrap(), &[0.0, 5.0, 10.0, 15.0]);
    }

    /// `set_output_parameters_from_image` is how a caller opts into a reference
    /// geometry; without it the defaults are unit/zero/identity, *not* the
    /// input image's.
    #[test]
    fn output_parameters_from_image_copies_the_reference_grid() {
        let mut reference = Image::new(&[2, 2], PixelId::Float32);
        reference.set_spacing(&[3.0, 4.0]).unwrap();
        reference.set_origin(&[-1.0, 5.0]).unwrap();
        reference.set_direction(&[0.0, 1.0, 1.0, 0.0]).unwrap();

        let img = ramp();
        let field = constant_field(&[2, 2], &[0.0, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_output_parameters_from_image(&reference);
        let out = f.execute(&img, &field).unwrap();

        assert_eq!(out.size(), &[2, 2]);
        assert_eq!(out.spacing(), &[3.0, 4.0]);
        assert_eq!(out.origin(), &[-1.0, 5.0]);
        assert_eq!(out.direction(), &[0.0, 1.0, 1.0, 0.0]);
    }

    /// Warping with a zero field over an output grid that matches the input's
    /// is the identity, and equals resampling with an identity transform — the
    /// two filters' shared interpolation seam agrees.
    #[test]
    fn zero_field_warp_equals_identity_resample() {
        use crate::resample::ResampleImageFilter;
        use crate::transform::AffineTransform;

        let img = ramp();
        let field = constant_field(&[4, 4], &[0.0, 0.0]);
        let warped = WarpImageFilter::new().execute(&img, &field).unwrap();
        let resampled = ResampleImageFilter::new()
            .set_reference_image(&img)
            .execute(&img, &AffineTransform::identity(2))
            .unwrap();
        assert_eq!(
            warped.scalar_slice::<f32>().unwrap(),
            resampled.scalar_slice::<f32>().unwrap()
        );
    }

    /// A constant field is exactly a translation transform, so warping matches
    /// resampling through `TranslationTransform` pixel for pixel — including
    /// the out-of-domain column, where `EdgePaddingValue` plays
    /// `DefaultPixelValue`'s role.
    #[test]
    fn constant_field_warp_equals_translation_resample() {
        use crate::resample::ResampleImageFilter;
        use crate::transform::TranslationTransform;

        let img = ramp();
        let field = constant_field(&[4, 4], &[1.5, -0.5]);
        let mut f = WarpImageFilter::new();
        f.set_edge_padding_value(-9.0);
        let warped = f.execute(&img, &field).unwrap();

        let resampled = ResampleImageFilter::new()
            .set_reference_image(&img)
            .set_default_pixel_value(-9.0)
            .execute(&img, &TranslationTransform::new(vec![1.5, -0.5]))
            .unwrap();
        assert_eq!(
            warped.scalar_slice::<f32>().unwrap(),
            resampled.scalar_slice::<f32>().unwrap()
        );
    }

    /// A `TransformToDisplacementFieldFilter` field of a translation, fed back
    /// into `warp`, reproduces resampling by that translation: the two filters
    /// of this port compose.
    #[test]
    fn warp_of_a_transform_to_displacement_field_matches_resample() {
        use crate::resample::ResampleImageFilter;
        use crate::transform::TranslationTransform;
        use crate::transform_to_displacement_field::TransformToDisplacementFieldFilter;

        let img = ramp();
        let t = TranslationTransform::new(vec![1.0, 1.0]);
        let field = TransformToDisplacementFieldFilter::new()
            .set_reference_image(&img)
            .execute(&t)
            .unwrap();

        let mut f = WarpImageFilter::new();
        f.set_edge_padding_value(-9.0);
        let warped = f.execute(&img, &field).unwrap();

        let resampled = ResampleImageFilter::new()
            .set_reference_image(&img)
            .set_default_pixel_value(-9.0)
            .execute(&img, &t)
            .unwrap();
        assert_eq!(
            warped.scalar_slice::<f32>().unwrap(),
            resampled.scalar_slice::<f32>().unwrap()
        );
    }

    /// Every interpolator reproduces the input under a zero field, since each
    /// output pixel lands on an integer input index.
    #[test]
    fn every_interpolating_kernel_reproduces_the_input_under_a_zero_field() {
        let img = ramp();
        let field = constant_field(&[4, 4], &[0.0, 0.0]);
        for interp in [
            Interpolator::NearestNeighbor,
            Interpolator::Linear,
            Interpolator::HammingWindowedSinc,
            Interpolator::CosineWindowedSinc,
            Interpolator::WelchWindowedSinc,
            Interpolator::LanczosWindowedSinc,
            Interpolator::BlackmanWindowedSinc,
        ] {
            let mut f = WarpImageFilter::new();
            f.set_interpolator(interp);
            let out = f.execute(&img, &field).unwrap();
            assert_eq!(
                out.scalar_slice::<f32>().unwrap(),
                img.scalar_slice::<f32>().unwrap(),
                "{interp:?}"
            );
        }
    }

    /// The B-spline kernel reproduces the input to coefficient round-trip
    /// noise, as it does in `resample`.
    #[test]
    fn bspline_kernel_reproduces_the_input_under_a_zero_field() {
        let img = ramp();
        let field = constant_field(&[4, 4], &[0.0, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_interpolator(Interpolator::BSpline);
        let out = f.execute(&img, &field).unwrap();
        for (got, want) in out
            .scalar_slice::<f32>()
            .unwrap()
            .iter()
            .zip(img.scalar_slice::<f32>().unwrap())
        {
            assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        }
    }

    /// `VerifyInputInformation`: the field's component count must equal the
    /// image dimension.
    #[test]
    fn a_field_with_the_wrong_component_count_is_rejected() {
        let img = ramp();
        let field = Image::from_vec_vector(&[4, 4], 3, vec![0.0f64; 48]).unwrap();
        assert_eq!(
            WarpImageFilter::new().execute(&img, &field),
            Err(TransformError::DisplacementFieldComponentMismatch {
                expected: 2,
                got: 3
            })
        );
    }

    /// `pixel_types2: RealVectorPixelIDTypeList` — a scalar image is not a
    /// displacement field, even in one dimension where the component count
    /// would line up.
    #[test]
    fn a_scalar_displacement_field_is_rejected() {
        let img = Image::from_vec(&[4], vec![0.0f32; 4]).unwrap();
        let field = Image::from_vec(&[4], vec![0.0f64; 4]).unwrap();
        assert_eq!(
            WarpImageFilter::new().execute(&img, &field),
            Err(TransformError::Core(Error::RequiresVectorPixelType(
                PixelId::Float64
            )))
        );
    }

    /// A vector *input image* has no scalar reading, and the scalar guard says
    /// so.
    #[test]
    fn a_vector_input_image_is_rejected() {
        let img = Image::from_vec_vector(&[4, 4], 2, vec![0.0f32; 32]).unwrap();
        let field = constant_field(&[4, 4], &[0.0, 0.0]);
        assert_eq!(
            WarpImageFilter::new().execute(&img, &field),
            Err(TransformError::Core(Error::RequiresScalarPixelType(
                PixelId::VectorFloat32
            )))
        );
    }

    #[test]
    fn a_field_of_the_wrong_dimension_is_rejected() {
        let img = ramp();
        let field = Image::from_vec_vector(&[4, 4, 4], 3, vec![0.0f64; 192]).unwrap();
        assert_eq!(
            WarpImageFilter::new().execute(&img, &field),
            Err(TransformError::DimensionMismatch)
        );
    }

    #[test]
    fn an_output_size_of_the_wrong_length_is_rejected() {
        let img = ramp();
        let field = constant_field(&[4, 4], &[0.0, 0.0]);
        let mut f = WarpImageFilter::new();
        f.set_output_size(vec![4, 4, 4]);
        assert_eq!(
            f.execute(&img, &field),
            Err(TransformError::DimensionMismatch)
        );
    }

    /// A three-dimensional warp on the same-information path.
    #[test]
    fn three_dimensional_warp() {
        // 2x2x2 ramp, values 0..7.
        let img =
            Image::from_vec(&[2, 2, 2], (0..8).map(|v| v as f32).collect::<Vec<f32>>()).unwrap();
        // Shift by one along z: p_in = p_out + (0, 0, 1).
        let field = constant_field(&[2, 2, 2], &[0.0, 0.0, 1.0]);
        let mut f = WarpImageFilter::new();
        f.set_edge_padding_value(-1.0);
        let out = f.execute(&img, &field).unwrap();
        assert_eq!(
            out.scalar_slice::<f32>().unwrap(),
            &[4.0, 5.0, 6.0, 7.0, -1.0, -1.0, -1.0, -1.0]
        );
    }

    /// The general path in three dimensions, where all eight corners of
    /// `EvaluateDisplacementAtPhysicalPoint` carry weight.
    ///
    /// The field has spacing 2 and holds `d = (p/4, 0, 0)` at its `p`-th pixel
    /// (`p = i + 2j + 4k`), so an output pixel at unit spacing lands on field
    /// continuous index `(i/2, j/2, k/2)` and its displacement is the mean of
    /// the `2^n` corners its half-integer coordinates straddle. The input is
    /// `v(i, j, k) = i`, whose linear interpolant returns `p_in.x` exactly, so
    /// each output pixel reads back its own interpolated displacement plus its
    /// own x.
    #[test]
    fn three_dimensional_warp_on_the_general_path() {
        let mut img = Image::new(&[4, 2, 2], PixelId::Float64);
        {
            let buf = img.scalar_vec_mut::<f64>().unwrap();
            for k in 0..2 {
                for j in 0..2 {
                    for i in 0..4 {
                        buf[i + 4 * j + 8 * k] = i as f64;
                    }
                }
            }
        }

        let data: Vec<f64> = (0..8).flat_map(|p| [p as f64 / 4.0, 0.0, 0.0]).collect();
        let mut field = Image::from_vec_vector(&[2, 2, 2], 3, data).unwrap();
        field.set_spacing(&[2.0, 2.0, 2.0]).unwrap();

        let mut f = WarpImageFilter::new();
        f.set_output_size(vec![2, 2, 2]);
        let out = f.execute(&img, &field).unwrap();

        // d_x at output (i, j, k) is the mean of the corners of the unit cube
        // its half-integer field index straddles, all values scaled by 1/4:
        //   (0,0,0): {0}                -> 0.000   (1,0,0): {0,1}       -> 0.125
        //   (0,1,0): {0,2}              -> 0.250   (1,1,0): {0,1,2,3}   -> 0.375
        //   (0,0,1): {0,4}              -> 0.500   (1,0,1): {0,1,4,5}   -> 0.625
        //   (0,1,1): {0,2,4,6}          -> 0.750   (1,1,1): {0..7}      -> 0.875
        // and the output value is `i + d_x`.
        assert_eq!(
            out.scalar_slice::<f64>().unwrap(),
            &[
                0.0, 1.125, // k = 0, j = 0
                0.25, 1.375, // k = 0, j = 1
                0.5, 1.625, // k = 1, j = 0
                0.75, 1.875, // k = 1, j = 1
            ]
        );
    }

    /// `m_DefFieldSameInformation` compares the geometry within ITK's
    /// tolerances (1e-6, scaled by the output's first-axis spacing for the
    /// coordinate vectors), not exactly. A field origin off by 1e-9 still takes
    /// the fast path, which reads the field pixel at the same index *verbatim*;
    /// the general path would have interpolated across that 1e-9 and returned
    /// something else. Asserting exact equality is what distinguishes them.
    #[test]
    fn a_sub_tolerance_geometry_difference_still_takes_the_same_information_path() {
        // Input is a wide ramp so nothing lands out of domain.
        let img = Image::from_vec(&[8, 1], (0..8).map(|v| v as f64).collect::<Vec<f64>>()).unwrap();
        // Field d_x = pixel index, so it varies pixel to pixel.
        let data: Vec<f64> = (0..4).flat_map(|p| [p as f64, 0.0]).collect();
        let mut field = Image::from_vec_vector(&[4, 1], 2, data).unwrap();
        field.set_origin(&[1e-9, 0.0]).unwrap();

        let out = WarpImageFilter::new().execute(&img, &field).unwrap();
        // Output pixel i reads field pixel i verbatim: d_x = i, so p_in.x = 2i
        // and the ramp gives exactly 2i. Any interpolation of the field across
        // the 1e-9 origin shift would perturb the last three values.
        assert_eq!(out.scalar_slice::<f64>().unwrap(), &[0.0, 2.0, 4.0, 6.0]);
    }

    /// The other side of the same boundary: an origin shift of 1e-5, past the
    /// `1e-6 * spacing[0]` coordinate tolerance, drops to the general path,
    /// which interpolates the field and no longer returns the field pixel
    /// exactly.
    #[test]
    fn an_over_tolerance_geometry_difference_takes_the_general_path() {
        let img = Image::from_vec(&[8, 1], (0..8).map(|v| v as f64).collect::<Vec<f64>>()).unwrap();
        let data: Vec<f64> = (0..4).flat_map(|p| [p as f64, 0.0]).collect();
        let mut field = Image::from_vec_vector(&[4, 1], 2, data).unwrap();
        field.set_origin(&[1e-5, 0.0]).unwrap();

        let out = WarpImageFilter::new().execute(&img, &field).unwrap();
        let got = out.scalar_slice::<f64>().unwrap();
        // Pixel 0 clamps below the field's start, so it is still exactly 0.
        assert_eq!(got[0], 0.0);
        // Pixels 1..3 interpolate the field: d_x = i - 1e-5, so the value is
        // 2i - 1e-5 -- close to the fast path's 2i, but not equal to it.
        for (i, &g) in got.iter().enumerate().take(4).skip(1) {
            let want = 2.0 * i as f64 - 1e-5;
            assert!((g - want).abs() < 1e-12, "pixel {i}: {g}");
            assert_ne!(g, 2.0 * i as f64);
        }
    }
}
