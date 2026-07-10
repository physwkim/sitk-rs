//! `ESMDemonsRegistrationFunction` (itkESMDemonsRegistrationFunction.hxx) and
//! the `WarpImageFilter` pass it runs each iteration.
//!
//! This is the PDE behind *two* of the family's filters —
//! `FastSymmetricForcesDemonsRegistrationFilter` and
//! `DiffeomorphicDemonsRegistrationFilter` both name it as their
//! `DemonsRegistrationFunctionType` (itkFastSymmetricForcesDemonsRegistrationFilter.h:120-121,
//! whose own comment asks "FIXME: Why is this the only permissible function ?").
//! The similarly named `FastSymmetricForcesDemonsRegistrationFunction` is
//! reachable from no filter and is not ported.
//!
//! # The warped moving image
//!
//! Each iteration warps the moving image onto the fixed image's grid through the
//! current displacement field, with an *edge padding value* of
//! `NumericTraits<MovingPixelType>::max()`
//! (itkESMDemonsRegistrationFunction.hxx:63). `ComputeUpdate` then treats a
//! warped pixel equal to that value as "mapped outside the moving image" and
//! returns a zero update without touching the metric.
//!
//! Two consequences are reproduced here rather than repaired:
//!
//! * The warped image is stored at the *moving image's* pixel type, so
//!   interpolated values are quantised — truncated toward zero for the integer
//!   pixel types, rounded to `f32` for `Float32`
//!   (`static_cast<PixelType>(m_Interpolator->Evaluate(point))`,
//!   itkWarpImageFilter.hxx:314).
//! * A moving image that legitimately contains its own type's maximum is
//!   therefore indistinguishable from an out-of-buffer sample. A `UInt8` moving
//!   image with a `255` pixel silently drops every pixel that lands on it.
//!   Pinned by `a_moving_pixel_equal_to_the_type_maximum_reads_as_out_of_buffer`.
//!
//! # Deviation: a degenerate axis
//!
//! The hand-rolled warped-moving gradient
//! (itkESMDemonsRegistrationFunction.hxx:210-300) bounds-checks `index[dim]`
//! against the fixed image's region and then, at `index[dim] ==
//! FirstIndex[dim]`, reads `GetPixel(index + e_dim)` for the forward difference.
//! When the axis has extent `1` that neighbour is outside the buffer and ITK
//! reads past it — `Image::GetPixel` does no bounds checking. This port yields a
//! zero derivative along an axis of extent `1`, which is what the *other* three
//! gradient types already do there (`CentralDifferenceImageFunction`'s boundary
//! rule).

use sitk_core::{PixelId, Scalar, dispatch_scalar};

use super::field::Field;
use super::image_function::RealImage;

/// `ESMDemonsRegistrationFunctionEnums::Gradient`
/// (itkESMDemonsRegistrationFunction.h:40-46): which image supplies the demons
/// force's gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EsmGradient {
    /// `∇f + ∇(m∘φ)`, the symmetric force. The default.
    #[default]
    Symmetric,
    /// `2·∇f`, Thirion's original force.
    Fixed,
    /// `2·∇(m∘φ)`, the gradient of the *warped* moving image, taken by one-sided
    /// differences at the borders of the fixed image's region.
    WarpedMoving,
    /// `2·∇m` evaluated at the mapped point on the *unwarped* moving image, by
    /// `CentralDifferenceImageFunction::Evaluate`.
    MappedMoving,
}

/// `NumericTraits<T>::max()` for the scalar type `id` names, widened to `f64` —
/// `WarpImageFilter`'s edge padding value here.
pub(crate) fn scalar_max(id: PixelId) -> f64 {
    match id.component_id() {
        PixelId::UInt8 => f64::from(u8::MAX),
        PixelId::Int8 => f64::from(i8::MAX),
        PixelId::UInt16 => f64::from(u16::MAX),
        PixelId::Int16 => f64::from(i16::MAX),
        PixelId::UInt32 => f64::from(u32::MAX),
        PixelId::Int32 => f64::from(i32::MAX),
        PixelId::UInt64 => u64::MAX as f64,
        PixelId::Int64 => i64::MAX as f64,
        PixelId::Float32 => f64::from(f32::MAX),
        // `component_id()` returns only the ten scalar ids, so this is `Float64`.
        _ => f64::MAX,
    }
}

fn narrow_and_widen<T: Scalar>(v: f64) -> f64 {
    T::from_f64(v).as_f64()
}

/// `static_cast<PixelType>(double)`: round-trip through the scalar type `id`
/// names, so the result is exactly a value that type can hold.
pub(crate) fn quantize(id: PixelId, v: f64) -> f64 {
    dispatch_scalar!(id, narrow_and_widen, v)
}

/// The moving image resampled onto the fixed image's grid, at the moving image's
/// pixel type, with out-of-buffer samples set to [`WarpedImage::edge_padding`].
pub(crate) struct WarpedImage {
    data: Vec<f64>,
    size: Vec<usize>,
    strides: Vec<usize>,
    /// `NumericTraits<MovingPixelType>::max()`.
    edge_padding: f64,
}

impl WarpedImage {
    fn at(&self, index: &[usize]) -> f64 {
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

    /// Whether a warped sample carries the "mapped outside" sentinel.
    fn is_outside(&self, value: f64) -> bool {
        value == self.edge_padding
    }
}

/// `WarpImageFilter` as `ESMDemonsRegistrationFunction::InitializeIteration`
/// configures it (lines 149-156): output geometry taken from the fixed image,
/// linear interpolation of the moving image, edge padding at the moving pixel
/// type's maximum.
///
/// The displacement field is read pixel-for-pixel against the fixed image's
/// grid — `WarpImageFilter`'s `m_DefFieldSameInformation` fast path
/// (itkWarpImageFilter.hxx:291-325), which is the only path the rest of the PDE
/// is consistent with: `ComputeUpdate` reads the fixed image, the field and the
/// warped image at one shared index.
pub(crate) fn warp_moving(fixed: &RealImage, moving: &RealImage, field: &Field) -> WarpedImage {
    let dim = fixed.dimension();
    let edge_padding = scalar_max(moving.pixel_id());
    let pixels = field.number_of_pixels();

    let mut strides = vec![1usize; dim];
    for d in 1..dim {
        strides[d] = strides[d - 1] * fixed.size()[d - 1];
    }

    let mut data = Vec::with_capacity(pixels);
    let mut index = vec![0usize; dim];
    for pixel in 0..pixels {
        field.multi_index(pixel, &mut index);
        let mut point = fixed.index_to_physical_point(&index);
        for (coordinate, &displacement) in point.iter_mut().zip(field.vector_at(pixel)) {
            *coordinate += displacement;
        }
        data.push(if moving.is_inside_buffer(&point) {
            quantize(moving.pixel_id(), moving.linear_interpolate(&point))
        } else {
            edge_padding
        });
    }

    WarpedImage {
        data,
        size: fixed.size().to_vec(),
        strides,
        edge_padding,
    }
}

/// `ESMDemonsRegistrationFunction`'s per-iteration state.
///
/// The warped moving image is *not* a member: `initialize_iteration` returns it
/// and `compute_update` takes it, so there is no state in which the function
/// holds a warp that does not match the field it was asked to differentiate.
pub(crate) struct EsmFunction<'a> {
    fixed: &'a RealImage,
    moving: &'a RealImage,
    use_gradient_type: EsmGradient,
    maximum_update_step_length: f64,
    intensity_difference_threshold: f64,
    /// `m_DenominatorThreshold`, hard-coded to `1e-9` in the constructor
    /// (itkESMDemonsRegistrationFunction.hxx:38) and exposed by no setter.
    denominator_threshold: f64,
    /// `m_Normalizer`, or `-1.0` for the unrestricted-step-length special case.
    normalizer: f64,

    sum_of_squared_difference: f64,
    number_of_pixels_processed: u64,
    sum_of_squared_change: f64,
    pub(crate) metric: f64,
    pub(crate) rms_change: f64,
}

impl<'a> EsmFunction<'a> {
    pub(crate) fn new(
        fixed: &'a RealImage,
        moving: &'a RealImage,
        use_gradient_type: EsmGradient,
        maximum_update_step_length: f64,
        intensity_difference_threshold: f64,
    ) -> Self {
        EsmFunction {
            fixed,
            moving,
            use_gradient_type,
            maximum_update_step_length,
            intensity_difference_threshold,
            denominator_threshold: 1e-9,
            normalizer: -1.0,
            sum_of_squared_difference: 0.0,
            number_of_pixels_processed: 0,
            sum_of_squared_change: 0.0,
            metric: f64::MAX,
            rms_change: f64::MAX,
        }
    }

    /// `ESMDemonsRegistrationFunction::InitializeIteration` (lines 113-164).
    ///
    /// The normalizer is `(Σ_k spacing_k²) · step² / dim`, and a
    /// non-positive `maximum_update_step_length` sets it to `-1.0` to mean
    /// "unrestricted": the demons denominator then drops its intensity term.
    pub(crate) fn initialize_iteration(&mut self, field: &Field) -> WarpedImage {
        let spacing = self.fixed.spacing();
        self.normalizer = if self.maximum_update_step_length > 0.0 {
            let sum_of_squares: f64 = spacing.iter().map(|s| s * s).sum();
            sum_of_squares * self.maximum_update_step_length * self.maximum_update_step_length
                / spacing.len() as f64
        } else {
            -1.0
        };

        self.sum_of_squared_difference = 0.0;
        self.number_of_pixels_processed = 0;
        self.sum_of_squared_change = 0.0;

        warp_moving(self.fixed, self.moving, field)
    }

    /// The warped moving image's gradient, in index space, divided by the
    /// *fixed* image's spacing (itkESMDemonsRegistrationFunction.hxx:208-300).
    ///
    /// Unlike `CentralDifferenceImageFunction`, this falls back to a one-sided
    /// difference at the borders of the fixed image's region and wherever the
    /// neighbour carries the out-of-buffer sentinel, and only reports zero when
    /// both neighbours are unusable.
    fn warped_moving_gradient(
        &self,
        warped: &WarpedImage,
        index: &[usize],
        center: f64,
    ) -> Vec<f64> {
        let dim = self.fixed.dimension();
        let mut gradient = vec![0.0f64; dim];
        let mut neighbor: Vec<i64> = index.iter().map(|&i| i as i64).collect();

        for (d, derivative) in gradient.iter_mut().enumerate() {
            let extent = warped.size[d] as i64;
            // Upstream reads out of the buffer here; see the module docs.
            if extent < 2 {
                continue;
            }
            let here = neighbor[d];
            let spacing = self.fixed.spacing()[d];

            if here == 0 {
                neighbor[d] = 1;
                let forward = warped.at_signed(&neighbor);
                neighbor[d] = here;
                if !warped.is_outside(forward) {
                    *derivative = (forward - center) / spacing;
                }
                continue;
            }
            if here == extent - 1 {
                neighbor[d] = here - 1;
                let backward = warped.at_signed(&neighbor);
                neighbor[d] = here;
                if !warped.is_outside(backward) {
                    *derivative = (center - backward) / spacing;
                }
                continue;
            }

            neighbor[d] = here + 1;
            let forward = warped.at_signed(&neighbor);
            neighbor[d] = here - 1;
            let backward = warped.at_signed(&neighbor);
            neighbor[d] = here;

            *derivative = match (warped.is_outside(forward), warped.is_outside(backward)) {
                (true, true) => 0.0,
                (true, false) => (center - backward) / spacing,
                (false, true) => (forward - center) / spacing,
                (false, false) => (forward - backward) * 0.5 / spacing,
            };
        }

        gradient
    }

    /// `ESMDemonsRegistrationFunction::ComputeUpdate` (lines 166-391). The
    /// neighbourhood radius is zero, so only the field's centre pixel is read.
    pub(crate) fn compute_update(
        &mut self,
        warped: &WarpedImage,
        index: &[usize],
        displacement: &[f64],
        update: &mut [f64],
    ) {
        update.fill(0.0);

        let fixed_value = self.fixed.at(index);
        let moving_value = warped.at(index);
        // Mapped outside the moving image: zero update, and the metric never
        // sees this pixel.
        if warped.is_outside(moving_value) {
            return;
        }

        // `usedOrientFreeGradientTimes2`, in index space.
        let gradient_times_2: Vec<f64> = match self.use_gradient_type {
            EsmGradient::Symmetric => {
                let moving = self.warped_moving_gradient(warped, index, moving_value);
                let fixed = self.fixed.central_difference_at_index_local(index);
                fixed.iter().zip(&moving).map(|(f, m)| f + m).collect()
            }
            EsmGradient::WarpedMoving => self
                .warped_moving_gradient(warped, index, moving_value)
                .iter()
                .map(|w| w + w)
                .collect(),
            EsmGradient::Fixed => self
                .fixed
                .central_difference_at_index_local(index)
                .iter()
                .map(|f| f + f)
                .collect(),
            EsmGradient::MappedMoving => {
                let mut mapped_point = self.fixed.index_to_physical_point(index);
                for (coordinate, &offset) in mapped_point.iter_mut().zip(displacement) {
                    *coordinate += offset;
                }
                self.moving
                    .central_difference_at_point_local(&mapped_point)
                    .iter()
                    .map(|m| m + m)
                    .collect()
            }
        };

        // Every gradient above is index-space; the *fixed* image's direction
        // rotates the sum into physical space — even the `MappedMoving` one,
        // which was differentiated on the moving image's grid.
        let used = self
            .fixed
            .local_vector_to_physical_vector(&gradient_times_2);
        let squared_magnitude: f64 = used.iter().map(|g| g * g).sum();

        let speed_value = fixed_value - moving_value;
        if speed_value.abs() >= self.intensity_difference_threshold {
            let denominator = if self.normalizer > 0.0 {
                squared_magnitude + speed_value * speed_value / self.normalizer
            } else {
                squared_magnitude
            };
            if denominator >= self.denominator_threshold {
                let factor = 2.0 * speed_value / denominator;
                for (component, &gradient) in update.iter_mut().zip(&used) {
                    *component = factor * gradient;
                }
            }
        }

        // Accumulated for every pixel that mapped inside, thresholded or not —
        // and, as ITK's own comment says, "without taking into account the
        // current update step".
        self.sum_of_squared_difference += speed_value * speed_value;
        self.number_of_pixels_processed += 1;
        self.sum_of_squared_change += update.iter().map(|u| u * u).sum::<f64>();
    }

    /// `ESMDemonsRegistrationFunction::ReleaseGlobalDataPointer` (lines
    /// 393-408). Both measurements keep their previous value — initially
    /// `f64::MAX` — when no pixel mapped inside the moving image.
    pub(crate) fn finish_iteration(&mut self) {
        if self.number_of_pixels_processed != 0 {
            let n = self.number_of_pixels_processed as f64;
            self.metric = self.sum_of_squared_difference / n;
            self.rms_change = (self.sum_of_squared_change / n).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::Image;

    #[test]
    fn scalar_max_is_the_pixel_types_maximum() {
        assert_eq!(scalar_max(PixelId::UInt8), 255.0);
        assert_eq!(scalar_max(PixelId::Int8), 127.0);
        assert_eq!(scalar_max(PixelId::UInt16), 65535.0);
        assert_eq!(scalar_max(PixelId::Float32), f64::from(f32::MAX));
        assert_eq!(scalar_max(PixelId::Float64), f64::MAX);
        // A vector id resolves through `component_id()`.
        assert_eq!(scalar_max(PixelId::VectorUInt8), 255.0);
    }

    /// `static_cast<PixelType>` truncates toward zero for the integer types and
    /// rounds to nearest for `Float32`.
    #[test]
    fn quantize_round_trips_through_the_pixel_type() {
        assert_eq!(quantize(PixelId::UInt8, 2.9), 2.0);
        assert_eq!(quantize(PixelId::Int8, -2.9), -2.0);
        assert_eq!(quantize(PixelId::Float64, 2.9), 2.9);
        // 0.1 is not representable in binary32.
        assert_ne!(quantize(PixelId::Float32, 0.1), 0.1);
        assert_eq!(quantize(PixelId::Float32, 0.1), f64::from(0.1f32));
    }

    /// A half-pixel shift interpolates to `x + 0.5`, which the `UInt8` warp
    /// truncates back down; the last pixel maps to `4.5`, past the buffer's
    /// half-open `4.5` bound, and takes the edge padding value.
    #[test]
    fn warping_quantizes_to_the_moving_pixel_type_and_pads_outside() {
        let fixed = RealImage::new(&Image::from_vec(&[5, 1], vec![0u8; 5]).unwrap()).unwrap();
        let moving =
            RealImage::new(&Image::from_vec(&[5, 1], vec![0u8, 1, 2, 3, 4]).unwrap()).unwrap();
        let field = Field {
            data: vec![0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],
            size: vec![5, 1],
        };
        let warped = warp_moving(&fixed, &moving, &field);
        assert_eq!(warped.data, vec![0.0, 1.0, 2.0, 3.0, 255.0]);
        assert!(warped.is_outside(255.0));
    }

    /// The same shift on a `Float64` moving image keeps the fractional values,
    /// and nothing is mistaken for the sentinel.
    #[test]
    fn warping_a_float_image_keeps_the_interpolated_value() {
        let fixed = RealImage::new(&Image::from_vec(&[5, 1], vec![0.0f64; 5]).unwrap()).unwrap();
        let moving =
            RealImage::new(&Image::from_vec(&[5, 1], vec![0.0f64, 1.0, 2.0, 3.0, 4.0]).unwrap())
                .unwrap();
        let field = Field {
            data: vec![0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],
            size: vec![5, 1],
        };
        let warped = warp_moving(&fixed, &moving, &field);
        assert_eq!(warped.data, vec![0.5, 1.5, 2.5, 3.5, f64::MAX]);
    }

    /// A zero field over identical grids warps the moving image onto itself.
    #[test]
    fn a_zero_field_warps_the_moving_image_onto_itself() {
        let fixed = RealImage::new(&Image::from_vec(&[3, 2], vec![0.0f64; 6]).unwrap()).unwrap();
        let values: Vec<f64> = (0..6).map(f64::from).collect();
        let moving = RealImage::new(&Image::from_vec(&[3, 2], values.clone()).unwrap()).unwrap();
        let warped = warp_moving(&fixed, &moving, &Field::zeros(&[3, 2]));
        assert_eq!(warped.data, values);
    }
}
