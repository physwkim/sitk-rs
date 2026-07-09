//! ITK's mathematical-morphology family: flat structuring elements,
//! grayscale and binary erosion/dilation, opening/closing, and white/black
//! top-hat.
//!
//! Verified against ITK's `Modules/Filtering/{MathematicalMorphology,
//! BinaryMathematicalMorphology}/include/`: `itkFlatStructuringElement.h` /
//! `.hxx` (structuring elements), `itkEllipsoidInteriorExteriorSpatialFunction.hxx`
//! and `itkFloodFilledSpatialFunctionConditionalConstIterator.hxx` (the
//! ball's exact rasterization rule), `itkMorphologyImageFilter.h`,
//! `itkBasicErodeImageFilter.h` / `.hxx`, `itkBasicDilateImageFilter.h` /
//! `.hxx` (grayscale erode/dilate), `itkGrayscaleMorphologicalOpeningImageFilter.h`
//! / `.hxx`, `itkGrayscaleMorphologicalClosingImageFilter.h` / `.hxx`
//! (grayscale opening/closing), `itkBinaryMorphologyImageFilter.h` / `.hxx`,
//! `itkBinaryErodeImageFilter.h` / `.hxx`, `itkBinaryDilateImageFilter.h` /
//! `.hxx` (binary erode/dilate), `itkBinaryMorphologicalOpeningImageFilter.h`
//! / `.hxx`, `itkBinaryMorphologicalClosingImageFilter.h` / `.hxx` (binary
//! opening/closing), `itkWhiteTopHatImageFilter.h` / `.hxx`,
//! `itkBlackTopHatImageFilter.h` / `.hxx` (top-hats).
//!
//! ## Structuring elements
//!
//! [`StructuringElement`] mirrors `FlatStructuringElement<VDimension>`'s
//! `Box`/`Cross`/`Ball` factories, always with the non-parametric radius
//! convention SimpleITK uses (`radiusIsParametric = false`,
//! `sitkCreateKernel.h`'s `Ball`):
//!
//! - [`StructuringElement::box_`]: every offset in the `(2r+1)`-per-axis
//!   window is on (`FlatStructuringElement::Box`).
//! - [`StructuringElement::cross`]: an offset is on iff at most one of its
//!   axes is nonzero — the union of the `2*radius[d]+1`-long line through
//!   the center along each axis (`Cross`).
//! - [`StructuringElement::ball`]: derived from
//!   `EllipsoidInteriorExteriorSpatialFunction::Evaluate` composed with
//!   `FloodFilledSpatialFunctionConditionalConstIterator::IsPixelIncluded`'s
//!   default "Center" strategy (case 1): the ellipsoid is centered at
//!   `radius[d] + 0.5` in each axis (`Ball`'s `center[i] =
//!   res.GetRadius(i) + 0.5`) and sampled at `index[d] + 0.5`, so for offset
//!   `o = index - radius` the `+0.5`s cancel and an offset is on iff
//!   `Σ_d (o[d] / (radius[d] + 0.5))² ≤ 1` (`axes[d] = GetSize(d) =
//!   2*radius[d]+1` in the non-parametric case, `Evaluate`'s
//!   `distanceSquared <= 1`). This is evaluated directly per offset rather
//!   than literally flood-filled: the region is an ellipsoid (convex, hence
//!   connected) around the seed, so per-offset evaluation and a flood fill
//!   from the center agree pixel-for-pixel.
//!
//! [`StructuringElement::from_mask`] builds an arbitrary structuring element
//! directly from a dimension-0-fastest bool mask (matching
//! [`sitk_core::Neighborhood`]'s own layout), which is how the tests below
//! exercise the "empty structuring element" (every position off) case.
//!
//! ## Grayscale erode / dilate
//!
//! `BasicErodeImageFilter::Evaluate` / `BasicDilateImageFilter::Evaluate`: a
//! plain min/max over the neighborhood positions where the kernel is on,
//! each with its own default `ConstantBoundaryCondition` (set in each
//! filter's constructor): erode uses `NumericTraits<T>::max()` (so an
//! out-of-image neighbor can never win the min), dilate uses
//! `NumericTraits<T>::NonpositiveMin()` (so it can never win the max). An
//! empty kernel therefore yields exactly that sentinel for every pixel (no
//! neighbor ever participates in the reduction) — ITK's own degenerate
//! behavior, not a special case added here.
//!
//! ## Grayscale opening / closing
//!
//! Both default `SafeBorder = true`
//! (`GrayscaleMorphologicalOpeningImageFilter`'s `m_SafeBorder{ true }`
//! member default; `GrayscaleMorphologicalClosingImageFilter`'s constructor
//! sets `m_SafeBorder(true)` explicitly). With it on, `GenerateData`'s
//! `BASIC` branch pads the image by the kernel radius with the *first* op's
//! own boundary sentinel (`max()` before erode for opening, before dilate
//! for closing it's `NonpositiveMin()`), runs both ops, then crops the
//! padding back off. This changes results versus a naive unpadded compose
//! specifically near the image border, where the pad supplies the second
//! op's neighborhood with real data instead of that op's own default
//! boundary condition. Opening is `dilate(erode(f))`; closing is
//! `erode(dilate(f))`.
//!
//! ## Binary erode / dilate
//!
//! `BinaryErodeImageFilter`/`BinaryDilateImageFilter` (via
//! `BinaryMorphologyImageFilter`) implement a fast Vincent-1991
//! border-tracking algorithm; this port evaluates the same net per-pixel
//! result directly over [`sitk_core::NeighborhoodIterator`] instead, traced
//! from each `GenerateData`'s three stages (initial fill, Minkowski-set
//! painting, final restore) end to end in both `.hxx` files:
//!
//! - Erode: a pixel survives (output = `foreground`) iff *every* on-kernel
//!   offset's neighbor (boundary value = `foreground` if
//!   `boundary_to_foreground` else `background`, per `m_BoundaryToForeground`)
//!   equals `foreground`. A pixel that doesn't survive keeps its *original*
//!   value when that value wasn't `foreground` — `GenerateData`'s final
//!   pass, `if (outValue == backgroundValue && inValue != foregroundValue)
//!   outIt.Set(inValue)` — so non-foreground labels on a multi-valued input
//!   pass through erosion untouched; otherwise it becomes `background`.
//! - Dilate: a pixel is painted (output = `foreground`) iff *some*
//!   on-kernel offset's neighbor equals `foreground` (same boundary rule).
//!   An unpainted pixel keeps its original value unless that value was
//!   itself `foreground` (`GenerateData`'s initial fill, `outIt.Set(value ==
//!   foreground ? background : value)`), in which case it becomes
//!   `background` — an isolated foreground pixel not self-painted by a
//!   kernel that excludes the origin; never observed for `box_`/`cross`/
//!   `ball`, which always include the center offset, but reachable via
//!   `from_mask`.
//!
//! Defaults follow `Code/BasicFilters/yaml/BinaryErodeImageFilter.yaml` /
//! `BinaryDilateImageFilter.yaml`: `foreground_value` =
//! `NumericTraits<T>::max()`, `background_value` =
//! `NumericTraits<T>::NonpositiveMin()` (`itkBinaryMorphologyImageFilter.h`'s
//! `m_ForegroundValue`/`m_BackgroundValue` member defaults).
//! `boundary_to_foreground` defaults per filter (each `.hxx` constructor):
//! `true` for erode, `false` for dilate.
//!
//! ## Binary opening / closing
//!
//! `BinaryMorphologicalOpeningImageFilter`: plain `dilate(erode(f))`, no
//! padding — `erode`'s `background_value` is the caller's `background`.
//!
//! `BinaryMorphologicalClosingImageFilter`: `erode(dilate(f))`, with
//! `SafeBorder = true` by default. The internal `background_value` used for
//! `erode`/padding is `0`, unless `foreground == 0`, in which case it's
//! `NumericTraits<T>::max()` (so the pad sentinel never collides with a real
//! foreground value of `0`). After erode and crop, every output pixel that
//! isn't `foreground` is overwritten with the *original* input's value at
//! that position (`GenerateData`'s final loop) — this is what makes closing
//! safe for multi-valued label inputs even though the internal pipeline's
//! own background bookkeeping is single-valued.
//!
//! ## White / black top-hat
//!
//! `WhiteTopHat = input - opening(input)`; `BlackTopHat = closing(input) -
//! input` (`itkWhiteTopHatImageFilter.hxx`/`itkBlackTopHatImageFilter.hxx`,
//! both via `SubtractImageFilter`, both `SafeBorder = true` by default,
//! propagated to the inner opening/closing).

use crate::error::{FilterError, Result};
use crate::subtract;
use sitk_core::{
    ConstantBoundaryCondition, Image, NeighborhoodIterator, PixelId, Scalar, dispatch_scalar,
};

// ---- NumericTraits<T>::max() / NonpositiveMin() ----------------------------

/// `NumericTraits<T>::max()` / `NumericTraits<T>::NonpositiveMin()`: the
/// sentinel values ITK's grayscale erode/dilate default boundary conditions
/// use, and the binary erode/dilate `ForegroundValue`/`BackgroundValue`
/// defaults. For every integer type `NonpositiveMin() == MIN`; only
/// floating point differs, where it's the most negative *finite* value
/// (`-MAX`), not `MIN` (the smallest positive normal float).
trait Bounds: Scalar {
    const MAX_VALUE: Self;
    const NONPOSITIVE_MIN: Self;
}

macro_rules! impl_bounds_int {
    ($($t:ty),+ $(,)?) => {$(
        impl Bounds for $t {
            const MAX_VALUE: Self = <$t>::MAX;
            const NONPOSITIVE_MIN: Self = <$t>::MIN;
        }
    )+};
}

macro_rules! impl_bounds_float {
    ($($t:ty),+ $(,)?) => {$(
        impl Bounds for $t {
            const MAX_VALUE: Self = <$t>::MAX;
            const NONPOSITIVE_MIN: Self = -<$t>::MAX;
        }
    )+};
}

impl_bounds_int!(u8, i8, u16, i16, u32, i32, u64, i64);
impl_bounds_float!(f32, f64);

fn bounds_typed<T: Bounds>() -> (f64, f64) {
    (T::MAX_VALUE.as_f64(), T::NONPOSITIVE_MIN.as_f64())
}

/// `(NumericTraits<T>::max(), NumericTraits<T>::NonpositiveMin())` for
/// whichever concrete type `id` names, round-tripped through `f64`. Exact
/// even at the integer extremes: Rust's saturating float-to-int cast (see
/// [`sitk_core::Scalar::from_f64`]) clamps back to the true `MAX`/`MIN` for
/// any value `as_f64()` rounds up past them (`u64::MAX`, `i64::MAX`).
fn bounds_for(id: PixelId) -> (f64, f64) {
    dispatch_scalar!(id, bounds_typed)
}

// ---- structuring elements ---------------------------------------------

/// A flat (binary) structuring element: an N-D window of on/off offsets
/// around a center, matching `itk::FlatStructuringElement<VDimension>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuringElement {
    radius: Vec<usize>,
    // Dimension-0-fastest, matching `sitk_core::Neighborhood`'s layout.
    on: Vec<bool>,
}

impl StructuringElement {
    /// Per-dimension radius (`FlatStructuringElement::GetRadius`).
    pub fn radius(&self) -> &[usize] {
        &self.radius
    }

    fn on(&self) -> &[bool] {
        &self.on
    }

    /// Builds a structuring element directly from a dimension-0-fastest
    /// on/off `mask` (matching [`sitk_core::Neighborhood::values`]'s own
    /// layout). Errors if `mask.len() != Π (2*radius[d]+1)`.
    pub fn from_mask(radius: &[usize], mask: Vec<bool>) -> Result<Self> {
        let expected = window_len(radius);
        if mask.len() != expected {
            return Err(FilterError::MaskLengthMismatch {
                expected,
                got: mask.len(),
            });
        }
        Ok(Self {
            radius: radius.to_vec(),
            on: mask,
        })
    }

    /// `FlatStructuringElement::Box`: every offset in the window is on.
    pub fn box_(radius: &[usize]) -> Self {
        Self {
            radius: radius.to_vec(),
            on: vec![true; window_len(radius)],
        }
    }

    /// `FlatStructuringElement::Cross`: an offset is on iff at most one axis
    /// is nonzero (the union of the per-axis line through the center).
    pub fn cross(radius: &[usize]) -> Self {
        let on = window_offsets(radius)
            .iter()
            .map(|o| o.iter().filter(|&&x| x != 0).count() <= 1)
            .collect();
        Self {
            radius: radius.to_vec(),
            on,
        }
    }

    /// `FlatStructuringElement::Ball` (non-parametric radius, SimpleITK's
    /// default). See the module docs for the derivation of this predicate.
    pub fn ball(radius: &[usize]) -> Self {
        let on = window_offsets(radius)
            .iter()
            .map(|o| {
                o.iter()
                    .zip(radius)
                    .map(|(&x, &r)| {
                        let half_axis = r as f64 + 0.5;
                        (x as f64 / half_axis).powi(2)
                    })
                    .sum::<f64>()
                    <= 1.0
            })
            .collect();
        Self {
            radius: radius.to_vec(),
            on,
        }
    }
}

fn window_len(radius: &[usize]) -> usize {
    radius.iter().map(|&r| 2 * r + 1).product()
}

/// Per-offset ND coordinates for a `radius`-sized window, dimension-0-fastest
/// — the same enumeration `NeighborhoodIterator::new` builds internally
/// (itkNeighborhood.hxx:41-67, `ComputeNeighborhoodOffsetTable`), duplicated
/// here so a [`StructuringElement`]'s `on` mask lines up index-for-index
/// with a same-radius `Neighborhood::values()`.
fn window_offsets(radius: &[usize]) -> Vec<Vec<i64>> {
    let dim = radius.len();
    let n = window_len(radius);
    let mut offsets = Vec::with_capacity(n);
    let mut offset: Vec<i64> = radius.iter().map(|&r| -(r as i64)).collect();
    for _ in 0..n {
        offsets.push(offset.clone());
        for d in 0..dim {
            offset[d] += 1;
            if offset[d] > radius[d] as i64 {
                offset[d] = -(radius[d] as i64);
            } else {
                break;
            }
        }
    }
    offsets
}

// ---- grayscale erode / dilate ------------------------------------------

fn grayscale_erode_typed<T: Bounds>(img: &Image, kernel: &StructuringElement) -> Result<Image> {
    let boundary = ConstantBoundaryCondition::new(T::MAX_VALUE);
    let iter = NeighborhoodIterator::new(img, kernel.radius(), boundary)?;
    let mut out = Vec::with_capacity(img.number_of_pixels());
    for (_, nb) in iter {
        let mut min = T::MAX_VALUE;
        for (&on, &v) in kernel.on().iter().zip(nb.values()) {
            if on && v < min {
                min = v;
            }
        }
        out.push(min);
    }
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

fn grayscale_dilate_typed<T: Bounds>(img: &Image, kernel: &StructuringElement) -> Result<Image> {
    let boundary = ConstantBoundaryCondition::new(T::NONPOSITIVE_MIN);
    let iter = NeighborhoodIterator::new(img, kernel.radius(), boundary)?;
    let mut out = Vec::with_capacity(img.number_of_pixels());
    for (_, nb) in iter {
        let mut max = T::NONPOSITIVE_MIN;
        for (&on, &v) in kernel.on().iter().zip(nb.values()) {
            if on && v > max {
                max = v;
            }
        }
        out.push(max);
    }
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `BasicErodeImageFilter` (via `GrayscaleErodeImageFilter`): grayscale
/// erosion — min over the kernel-on neighborhood positions, boundary
/// `NumericTraits<T>::max()` (see module docs).
pub fn grayscale_erode(img: &Image, kernel: &StructuringElement) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), grayscale_erode_typed, img, kernel)
}

/// `BasicDilateImageFilter` (via `GrayscaleDilateImageFilter`): grayscale
/// dilation — max over the kernel-on neighborhood positions, boundary
/// `NumericTraits<T>::NonpositiveMin()` (see module docs).
pub fn grayscale_dilate(img: &Image, kernel: &StructuringElement) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), grayscale_dilate_typed, img, kernel)
}

// ---- pad / crop (SafeBorder) -------------------------------------------

fn pad_typed<T: Scalar>(img: &Image, radius: &[usize], value: f64) -> Result<Image> {
    let pad_value = T::from_f64(value);
    let src = img.scalar_slice::<T>()?;
    let in_size = img.size();
    let dim = in_size.len();
    let out_size: Vec<usize> = in_size
        .iter()
        .zip(radius)
        .map(|(&s, &r)| s + 2 * r)
        .collect();

    let mut in_strides = vec![1usize; dim];
    for d in 1..dim {
        in_strides[d] = in_strides[d - 1] * in_size[d - 1];
    }

    let n_out: usize = out_size.iter().product();
    let mut out = vec![pad_value; n_out];
    let mut coord = vec![0usize; dim];
    for (out_p, out_val) in out.iter_mut().enumerate() {
        let mut rem = out_p;
        for d in 0..dim {
            coord[d] = rem % out_size[d];
            rem /= out_size[d];
        }
        let mut inside = true;
        let mut in_p = 0usize;
        for d in 0..dim {
            if coord[d] < radius[d] || coord[d] >= radius[d] + in_size[d] {
                inside = false;
                break;
            }
            in_p += (coord[d] - radius[d]) * in_strides[d];
        }
        if inside {
            *out_val = src[in_p];
        }
    }

    let mut result = Image::from_vec(&out_size, out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// Pads `img` by `radius` on every side of every axis with `value`
/// (`ConstantPadImageFilter`, as used by `GrayscaleMorphologicalOpeningImageFilter`
/// / `...ClosingImageFilter` / `BinaryMorphologicalClosingImageFilter` when
/// `SafeBorder = true`).
fn pad_constant(img: &Image, radius: &[usize], value: f64) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), pad_typed, img, radius, value)
}

fn crop_typed<T: Scalar>(img: &Image, radius: &[usize]) -> Result<Image> {
    let src = img.scalar_slice::<T>()?;
    let in_size = img.size();
    let dim = in_size.len();
    let out_size: Vec<usize> = in_size
        .iter()
        .zip(radius)
        .map(|(&s, &r)| s - 2 * r)
        .collect();

    let mut in_strides = vec![1usize; dim];
    for d in 1..dim {
        in_strides[d] = in_strides[d - 1] * in_size[d - 1];
    }

    let n_out: usize = out_size.iter().product();
    let mut out = Vec::with_capacity(n_out);
    let mut coord = vec![0usize; dim];
    for out_p in 0..n_out {
        let mut rem = out_p;
        for d in 0..dim {
            coord[d] = rem % out_size[d];
            rem /= out_size[d];
        }
        let mut in_p = 0usize;
        for d in 0..dim {
            in_p += (coord[d] + radius[d]) * in_strides[d];
        }
        out.push(src[in_p]);
    }

    let mut result = Image::from_vec(&out_size, out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// Crops `radius` pixels off every side of every axis (`CropImageFilter`
/// with `SetUpperBoundaryCropSize`/`SetLowerBoundaryCropSize` both set to
/// the kernel radius, undoing [`pad_constant`]).
fn crop_border(img: &Image, radius: &[usize]) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), crop_typed, img, radius)
}

// ---- grayscale opening / closing ---------------------------------------

/// `GrayscaleMorphologicalOpeningImageFilter` (`SafeBorder = true`, ITK's
/// default — see module docs): `dilate(erode(f))`, with `f` padded by the
/// kernel radius using erode's own boundary sentinel before the compose,
/// cropped back after.
pub fn grayscale_morphological_opening(img: &Image, kernel: &StructuringElement) -> Result<Image> {
    let (max_value, _) = bounds_for(img.pixel_id());
    let padded = pad_constant(img, kernel.radius(), max_value)?;
    let eroded = grayscale_erode(&padded, kernel)?;
    let dilated = grayscale_dilate(&eroded, kernel)?;
    crop_border(&dilated, kernel.radius())
}

/// `GrayscaleMorphologicalClosingImageFilter` (`SafeBorder = true`, ITK's
/// default — see module docs): `erode(dilate(f))`, with `f` padded by the
/// kernel radius using dilate's own boundary sentinel before the compose,
/// cropped back after.
pub fn grayscale_morphological_closing(img: &Image, kernel: &StructuringElement) -> Result<Image> {
    let (_, nonpositive_min) = bounds_for(img.pixel_id());
    let padded = pad_constant(img, kernel.radius(), nonpositive_min)?;
    let dilated = grayscale_dilate(&padded, kernel)?;
    let eroded = grayscale_erode(&dilated, kernel)?;
    crop_border(&eroded, kernel.radius())
}

// ---- binary erode / dilate ----------------------------------------------

fn binary_erode_typed<T: Bounds>(
    img: &Image,
    kernel: &StructuringElement,
    foreground: f64,
    background: f64,
    boundary_to_foreground: bool,
) -> Result<Image> {
    let foreground = T::from_f64(foreground);
    let background = T::from_f64(background);
    let boundary_value = if boundary_to_foreground {
        foreground
    } else {
        background
    };
    let iter = NeighborhoodIterator::new(
        img,
        kernel.radius(),
        ConstantBoundaryCondition::new(boundary_value),
    )?;
    let mut out = Vec::with_capacity(img.number_of_pixels());
    for (_, nb) in iter {
        let survives = kernel
            .on()
            .iter()
            .zip(nb.values())
            .all(|(&on, &v)| !on || v == foreground);
        let input_value = nb.center_value();
        out.push(if survives {
            foreground
        } else if input_value != foreground {
            input_value
        } else {
            background
        });
    }
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

fn binary_dilate_typed<T: Bounds>(
    img: &Image,
    kernel: &StructuringElement,
    foreground: f64,
    background: f64,
    boundary_to_foreground: bool,
) -> Result<Image> {
    let foreground = T::from_f64(foreground);
    let background = T::from_f64(background);
    let boundary_value = if boundary_to_foreground {
        foreground
    } else {
        background
    };
    let iter = NeighborhoodIterator::new(
        img,
        kernel.radius(),
        ConstantBoundaryCondition::new(boundary_value),
    )?;
    let mut out = Vec::with_capacity(img.number_of_pixels());
    for (_, nb) in iter {
        let painted = kernel
            .on()
            .iter()
            .zip(nb.values())
            .any(|(&on, &v)| on && v == foreground);
        let input_value = nb.center_value();
        out.push(if painted {
            foreground
        } else if input_value == foreground {
            background
        } else {
            input_value
        });
    }
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `BinaryErodeImageFilter`: binary erosion of `foreground`, generalized to
/// multi-valued label inputs (see module docs for the exact restore-original
/// semantics traced from `GenerateData`). Defaults per SimpleITK's yaml:
/// `foreground` = `NumericTraits<T>::max()`, `background` =
/// `NumericTraits<T>::NonpositiveMin()`, `boundary_to_foreground` = `true`.
pub fn binary_erode(
    img: &Image,
    kernel: &StructuringElement,
    foreground: f64,
    background: f64,
    boundary_to_foreground: bool,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        binary_erode_typed,
        img,
        kernel,
        foreground,
        background,
        boundary_to_foreground
    )
}

/// `BinaryDilateImageFilter`: binary dilation of `foreground`, generalized to
/// multi-valued label inputs (see module docs). Defaults per SimpleITK's
/// yaml: `foreground` = `NumericTraits<T>::max()`, `background` =
/// `NumericTraits<T>::NonpositiveMin()`, `boundary_to_foreground` = `false`.
pub fn binary_dilate(
    img: &Image,
    kernel: &StructuringElement,
    foreground: f64,
    background: f64,
    boundary_to_foreground: bool,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        binary_dilate_typed,
        img,
        kernel,
        foreground,
        background,
        boundary_to_foreground
    )
}

// ---- binary opening / closing --------------------------------------------

/// `BinaryMorphologicalOpeningImageFilter`: `dilate(erode(f))`, no padding.
/// Each internal op keeps its own class default `boundary_to_foreground`
/// (erode `true`, dilate `false`) — the ITK minipipeline never overrides it.
pub fn binary_morphological_opening(
    img: &Image,
    kernel: &StructuringElement,
    foreground: f64,
    background: f64,
) -> Result<Image> {
    let eroded = binary_erode(img, kernel, foreground, background, true)?;
    binary_dilate(&eroded, kernel, foreground, background, false)
}

fn restore_non_foreground_typed<T: Scalar>(
    computed: &Image,
    original: &Image,
    foreground: f64,
) -> Result<Image> {
    let foreground = T::from_f64(foreground);
    let computed_vals = computed.scalar_slice::<T>()?;
    let original_vals = original.scalar_slice::<T>()?;
    let restored: Vec<T> = computed_vals
        .iter()
        .zip(original_vals)
        .map(|(&c, &o)| if c != foreground { o } else { c })
        .collect();
    let mut result = Image::from_vec(computed.size(), restored)?;
    result.copy_geometry_from(computed);
    Ok(result)
}

/// `BinaryMorphologicalClosingImageFilter`'s final pass: every pixel that
/// isn't `foreground` is overwritten with `original`'s value there.
fn restore_non_foreground(computed: &Image, original: &Image, foreground: f64) -> Result<Image> {
    dispatch_scalar!(
        computed.pixel_id(),
        restore_non_foreground_typed,
        computed,
        original,
        foreground
    )
}

/// `BinaryMorphologicalClosingImageFilter`: `erode(dilate(f))`,
/// `SafeBorder = true` by default, with a final restore of every
/// non-`foreground` output pixel to `img`'s original value (see module
/// docs). Each internal op keeps its own class default
/// `boundary_to_foreground` (dilate `false`, erode `true`).
pub fn binary_morphological_closing(
    img: &Image,
    kernel: &StructuringElement,
    foreground: f64,
) -> Result<Image> {
    let (max_value, _) = bounds_for(img.pixel_id());
    let background = if foreground == 0.0 { max_value } else { 0.0 };
    let padded = pad_constant(img, kernel.radius(), background)?;
    let dilated = binary_dilate(&padded, kernel, foreground, background, false)?;
    let eroded = binary_erode(&dilated, kernel, foreground, background, true)?;
    let cropped = crop_border(&eroded, kernel.radius())?;
    restore_non_foreground(&cropped, img, foreground)
}

// ---- white / black top-hat ------------------------------------------------

/// `WhiteTopHatImageFilter`: `input - opening(input)` (`SafeBorder = true`,
/// via [`grayscale_morphological_opening`]).
pub fn white_top_hat(img: &Image, kernel: &StructuringElement) -> Result<Image> {
    let opened = grayscale_morphological_opening(img, kernel)?;
    subtract(img, &opened)
}

/// `BlackTopHatImageFilter`: `closing(input) - input` (`SafeBorder = true`,
/// via [`grayscale_morphological_closing`]).
pub fn black_top_hat(img: &Image, kernel: &StructuringElement) -> Result<Image> {
    let closed = grayscale_morphological_closing(img, kernel)?;
    subtract(&closed, img)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn complement_u8(img: &Image) -> Image {
        let v: Vec<u8> = img
            .scalar_slice::<u8>()
            .unwrap()
            .iter()
            .map(|&x| 255 - x)
            .collect();
        let mut out = Image::from_vec(img.size(), v).unwrap();
        out.copy_geometry_from(img);
        out
    }

    // ---- structuring elements ----

    #[test]
    fn from_mask_length_mismatch_errors() {
        let err = StructuringElement::from_mask(&[1, 1], vec![true; 5]).unwrap_err();
        assert_eq!(
            err,
            FilterError::MaskLengthMismatch {
                expected: 9,
                got: 5
            }
        );
    }

    #[test]
    fn box_cross_ball_radius0_all_reduce_to_single_center() {
        for se in [
            StructuringElement::box_(&[0, 0]),
            StructuringElement::cross(&[0, 0]),
            StructuringElement::ball(&[0, 0]),
        ] {
            assert_eq!(se.on(), &[true]);
        }
    }

    #[test]
    fn ball_radius2_2d_excludes_corner_but_includes_near_corner() {
        // A radius-1 non-parametric ball is the full 3x3 (see distance.rs's
        // module docs), so use radius 2 to get a genuinely round footprint:
        // corner offset (2,2) is excluded, (2,1)/(1,2) are included.
        let se = StructuringElement::ball(&[2, 2]);
        // window is 5x5, dimension-0-fastest; index of offset (dx,dy) from
        // center (2,2) is (2+dx) + 5*(2+dy).
        let idx = |dx: i64, dy: i64| ((2 + dx) + 5 * (2 + dy)) as usize;
        assert!(se.on()[idx(0, 0)]);
        assert!(se.on()[idx(2, 1)]);
        assert!(se.on()[idx(1, 2)]);
        assert!(!se.on()[idx(2, 2)]);
        assert!(!se.on()[idx(-2, -2)]);
    }

    #[test]
    fn cross_2d_excludes_diagonal_but_includes_axes() {
        let se = StructuringElement::cross(&[1, 1]);
        let idx = |dx: i64, dy: i64| ((1 + dx) + 3 * (1 + dy)) as usize;
        assert!(se.on()[idx(0, 0)]);
        assert!(se.on()[idx(1, 0)]);
        assert!(se.on()[idx(0, 1)]);
        assert!(se.on()[idx(-1, 0)]);
        assert!(se.on()[idx(0, -1)]);
        assert!(!se.on()[idx(1, 1)]);
        assert!(!se.on()[idx(-1, -1)]);
    }

    // ---- grayscale erode/dilate: empty kernel ----

    #[test]
    fn grayscale_erode_dilate_empty_kernel_yields_bounds_sentinel_everywhere() {
        let se = StructuringElement::from_mask(&[1], vec![false, false, false]).unwrap();
        let f = img_u8(&[3], vec![10, 200, 5]);
        let eroded = grayscale_erode(&f, &se).unwrap();
        assert_eq!(eroded.scalar_slice::<u8>().unwrap(), &[u8::MAX; 3]);
        let dilated = grayscale_dilate(&f, &se).unwrap();
        assert_eq!(dilated.scalar_slice::<u8>().unwrap(), &[0u8; 3]);
    }

    // ---- grayscale erode/dilate: radius 0 is identity ----

    #[test]
    fn grayscale_erode_dilate_radius0_is_identity() {
        let se = StructuringElement::box_(&[0]);
        let f = img_u8(&[3], vec![3, 7, 2]);
        assert_eq!(
            grayscale_erode(&f, &se)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[3, 7, 2]
        );
        assert_eq!(
            grayscale_dilate(&f, &se)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[3, 7, 2]
        );
    }

    // ---- grayscale erode/dilate duality ----

    #[test]
    fn grayscale_erode_dilate_duality_holds_with_matching_boundaries() {
        let se = StructuringElement::ball(&[1, 1]);
        let f = img_u8(&[4, 3], vec![10, 250, 0, 5, 90, 1, 3, 200, 40, 60, 0, 255]);
        let eroded = grayscale_erode(&f, &se).unwrap();
        let dilated_of_complement = grayscale_dilate(&complement_u8(&f), &se).unwrap();
        let complement_of_dilated = complement_u8(&dilated_of_complement);
        assert_eq!(
            eroded.scalar_slice::<u8>().unwrap(),
            complement_of_dilated.scalar_slice::<u8>().unwrap()
        );
    }

    // ---- grayscale opening: SafeBorder vs a naive unpadded compose ----

    #[test]
    fn grayscale_opening_safe_border_differs_from_naive_unpadded_compose_at_the_edge() {
        let se = StructuringElement::box_(&[1]);
        let f = img_u8(&[3], vec![5, 1, 1]);

        let opened = grayscale_morphological_opening(&f, &se).unwrap();
        assert_eq!(opened.scalar_slice::<u8>().unwrap(), &[5, 1, 1]);

        // Naive: erode then dilate directly, no pad/crop.
        let naive = grayscale_dilate(&grayscale_erode(&f, &se).unwrap(), &se).unwrap();
        assert_eq!(naive.scalar_slice::<u8>().unwrap(), &[1, 1, 1]);

        assert_ne!(
            opened.scalar_slice::<u8>().unwrap(),
            naive.scalar_slice::<u8>().unwrap()
        );
    }

    // ---- grayscale opening idempotence ----

    #[test]
    fn grayscale_opening_is_idempotent() {
        let se = StructuringElement::box_(&[1, 1]);
        let f = img_u8(&[4, 3], vec![10, 250, 0, 5, 90, 1, 3, 200, 40, 60, 0, 255]);
        let once = grayscale_morphological_opening(&f, &se).unwrap();
        let twice = grayscale_morphological_opening(&once, &se).unwrap();
        assert_eq!(
            once.scalar_slice::<u8>().unwrap(),
            twice.scalar_slice::<u8>().unwrap()
        );
    }

    // ---- white / black top-hat ----

    #[test]
    fn white_top_hat_recovers_a_spike_narrower_than_the_kernel() {
        let se = StructuringElement::box_(&[1]);
        let f = img_u8(&[5], vec![0, 0, 9, 0, 0]);
        let wth = white_top_hat(&f, &se).unwrap();
        assert_eq!(wth.scalar_slice::<u8>().unwrap(), &[0, 0, 9, 0, 0]);
    }

    #[test]
    fn black_top_hat_recovers_a_notch_narrower_than_the_kernel() {
        let se = StructuringElement::box_(&[1]);
        let f = img_u8(&[5], vec![9, 9, 0, 9, 9]);
        let bth = black_top_hat(&f, &se).unwrap();
        assert_eq!(bth.scalar_slice::<u8>().unwrap(), &[0, 0, 9, 0, 0]);
    }

    // ---- binary erode/dilate: empty kernel (vacuous truth) ----

    #[test]
    fn binary_erode_dilate_empty_kernel_matches_vacuous_semantics() {
        let se = StructuringElement::from_mask(&[1], vec![false, false, false]).unwrap();
        let f = img_u8(&[3], vec![1, 0, 1]);
        let eroded = binary_erode(&f, &se, 1.0, 0.0, true).unwrap();
        assert_eq!(eroded.scalar_slice::<u8>().unwrap(), &[1, 1, 1]);
        let dilated = binary_dilate(&f, &se, 1.0, 0.0, false).unwrap();
        assert_eq!(dilated.scalar_slice::<u8>().unwrap(), &[0, 0, 0]);
    }

    // ---- binary erode/dilate: radius 0 is identity ----

    #[test]
    fn binary_erode_dilate_radius0_is_identity() {
        let se = StructuringElement::box_(&[0]);
        let f = img_u8(&[3], vec![1, 0, 1]);
        assert_eq!(
            binary_erode(&f, &se, 1.0, 0.0, true)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[1, 0, 1]
        );
        assert_eq!(
            binary_dilate(&f, &se, 1.0, 0.0, false)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[1, 0, 1]
        );
    }

    // ---- binary erode/dilate duality ----

    #[test]
    fn binary_erode_dilate_duality_holds_on_a_pure_binary_image() {
        let se = StructuringElement::box_(&[1, 1]);
        let f = img_u8(&[4, 3], vec![0, 255, 0, 0, 255, 255, 0, 0, 0, 255, 0, 0]);
        let eroded = binary_erode(&f, &se, 255.0, 0.0, true).unwrap();
        let dilated_of_complement =
            binary_dilate(&complement_u8(&f), &se, 255.0, 0.0, false).unwrap();
        let complement_of_dilated = complement_u8(&dilated_of_complement);
        assert_eq!(
            eroded.scalar_slice::<u8>().unwrap(),
            complement_of_dilated.scalar_slice::<u8>().unwrap()
        );
    }

    // ---- binary erode/dilate: label preservation ----

    #[test]
    fn binary_dilate_preserves_a_non_foreground_label_outside_the_kernel_reach() {
        // fg=1 at idx0, other-label 5 at idx2 (distance 2, outside a
        // radius-1 kernel's reach from idx0), background elsewhere.
        let se = StructuringElement::box_(&[1]);
        let f = img_u8(&[5], vec![1, 0, 5, 0, 0]);
        let dilated = binary_dilate(&f, &se, 1.0, 0.0, false).unwrap();
        // idx1 gets painted foreground (adjacent to idx0); idx2's label 5
        // is untouched since dilation from idx0 doesn't reach it.
        assert_eq!(dilated.scalar_slice::<u8>().unwrap(), &[1, 1, 5, 0, 0]);
    }

    #[test]
    fn binary_erode_preserves_a_non_foreground_label_at_a_removed_pixel() {
        // idx1's neighbor idx2 (label 5) isn't foreground, so idx1 doesn't
        // survive and reverts to background; idx2's own label passes
        // through untouched regardless of its own survival.
        let se = StructuringElement::box_(&[1]);
        let f = img_u8(&[5], vec![1, 1, 5, 0, 0]);
        let eroded = binary_erode(&f, &se, 1.0, 0.0, true).unwrap();
        assert_eq!(eroded.scalar_slice::<u8>().unwrap(), &[1, 0, 5, 0, 0]);
    }

    // ---- binary opening idempotence ----

    #[test]
    fn binary_opening_is_idempotent() {
        let se = StructuringElement::box_(&[1, 1]);
        let f = img_u8(&[4, 3], vec![0, 255, 0, 0, 255, 255, 0, 0, 0, 255, 0, 0]);
        let once = binary_morphological_opening(&f, &se, 255.0, 0.0).unwrap();
        let twice = binary_morphological_opening(&once, &se, 255.0, 0.0).unwrap();
        assert_eq!(
            once.scalar_slice::<u8>().unwrap(),
            twice.scalar_slice::<u8>().unwrap()
        );
    }

    // ---- binary closing: border-touching foreground + label restore ----

    #[test]
    fn binary_closing_at_the_border_preserves_a_distant_label_and_fills_no_gap() {
        let se = StructuringElement::box_(&[1]);
        let f = img_u8(&[5], vec![1, 0, 5, 0, 0]);
        let closed = binary_morphological_closing(&f, &se, 1.0).unwrap();
        assert_eq!(closed.scalar_slice::<u8>().unwrap(), &[1, 0, 5, 0, 0]);
    }
}
