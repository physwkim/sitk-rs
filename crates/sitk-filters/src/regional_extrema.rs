//! Valued and binary regional extrema.
//!
//! Ports of (ITK `Modules/Filtering/MathematicalMorphology/include/`):
//!
//! - [`valued_regional_maxima`] / [`valued_regional_minima`] —
//!   `itkValuedRegionalMaximaImageFilter.h` / `itkValuedRegionalMinimaImageFilter.h`,
//!   both thin instantiations of `itkValuedRegionalExtremaImageFilter.hxx`
//!   (`TFunction1 = TFunction2 = std::greater` for maxima, `std::less` for
//!   minima) exactly as [`crate::reconstruction`]'s erosion/dilation pair
//!   instantiate `itkReconstructionImageFilter.hxx` — [`ExtremaKind`] plays
//!   the same role here as `ReconstructionKind` does there.
//! - [`regional_maxima`] — `itkRegionalMaximaImageFilter.hxx`: delegates
//!   entirely to [`valued_regional_maxima`] (it has no flooding logic of its
//!   own), then either fills the whole output from `flat_is_maxima` (if the
//!   input was flat) or thresholds the valued output at its marker value.
//! - [`regional_minima`] — `itkRegionalMinimaImageFilter.hxx`: the mirror of
//!   [`regional_maxima`], built the same way: delegates to
//!   [`valued_regional_minima`], then either fills from `flat_is_minima` (if
//!   flat) or thresholds the valued output at its marker value
//!   (`NumericTraits<T>::max()` for minima, so *this* filter's threshold
//!   check is `v == marker -> background_value` -- a non-minimum pixel still
//!   holding the flood value -- `else -> foreground_value`, the same
//!   `BinaryThresholdImageFilter` shape `RegionalMaximaImageFilter.hxx` uses,
//!   just built on the minima marker). A private, non-parity `regional_minima`
//!   helper already exists in [`crate::watershed`], restricted to that
//!   module's own boolean flooding use case (see this module's own docs
//!   above on why [`crate::watershed`]'s helper stays a separate,
//!   unrefactored implementation); this is the unrelated, public,
//!   SimpleITK-parity port of the real `RegionalMinimaImageFilter`.
//!
//! ## The flooding algorithm
//!
//! `itkValuedRegionalExtremaImageFilter.hxx` visits pixels in raster order;
//! for each unvisited pixel it checks whether any neighbor disqualifies it
//! (a strictly greater neighbor for maxima, strictly lesser for minima) and,
//! if so, flood-fills every same-value neighbor reachable from it (its flat
//! zone) to `MarkerValue` -- `NumericTraits<T>::NonpositiveMin()` for maxima,
//! `NumericTraits<T>::max()` for minima. It tracks "already visited" by
//! overwriting the output pixel with `MarkerValue` and later testing
//! `compareOut(V, MarkerValue)`, which also happens to skip any *real* input
//! pixel whose own value already equals the type's extreme in the
//! disqualifying direction.
//!
//! That aliasing cannot change the observable result, though: a flat zone
//! sitting exactly at the type's extreme value (`NonpositiveMin` for maxima)
//! can never have a neighbor *past* that extreme, so in a non-flat image
//! (the flat case is handled separately, see below) such a zone always
//! borders a strictly different -- hence disqualifying -- neighbor, meaning
//! the "correct" flooded value and the aliasing shortcut's do-nothing value
//! are identical (`MarkerValue` either way). This port therefore tracks
//! "visited" in a separate `marked` buffer instead of aliasing it onto a
//! pixel value (matching [`crate::watershed`]'s private `regional_minima`
//! helper, which predates this module and stays a separate, unrefactored
//! implementation restricted to that module's own boolean use case), and
//! reads neighbor values from the untouched input throughout, which is
//! provably equivalent for every input.
//!
//! `ConstantBoundaryCondition`s of `MarkerValue` sit on both the input and
//! output shaped iterators. As in [`crate::reconstruction`] and
//! [`crate::watershed`], a boundary neighbor's assumed value can never win
//! its comparison (`NonpositiveMin` is never `>` anything for maxima;
//! `max()` is never `<` anything for minima), so this port simply skips
//! out-of-bounds neighbors rather than materializing the boundary.
//!
//! A completely flat image (every pixel equal) is reported specially both
//! here (the `flat` field of [`ValuedRegionalExtremaResult`], mirroring
//! `GetFlat()`) and in [`regional_maxima`] (`flat_is_maxima` decides the
//! fill value directly, bypassing the marker-value threshold entirely).
//!
//! `ValuedRegionalMaximaImageFilter`/`ValuedRegionalMinimaImageFilter.yaml`
//! expose only `FullyConnected`; `MarkerValue` is set internally by each
//! filter's constructor and is not user-configurable through SimpleITK.
//! `RegionalMaximaImageFilter.yaml` gives `ForegroundValue`/`BackgroundValue`
//! SimpleITK-level defaults of `1`/`0` (its own `pixeltype: Output` members),
//! which is *not* the raw ITK class's own default of
//! `NumericTraits<OutputPixelType>::max()`/`NonpositiveMin()` -- SimpleITK's
//! generated wrapper always calls `SetForegroundValue`/`SetBackgroundValue`
//! explicitly. Output pixel type is fixed `uint32_t`
//! (`output_pixel_type: uint32_t`), so `foreground_value`/`background_value`
//! are plain `u32` here, matching this crate's convention for other
//! fixed-output-type "pixeltype: Output" parameters (e.g.
//! [`crate::binary_threshold`]'s `inside`/`outside`).

use crate::error::Result;
use crate::image_from_f64;
use crate::morphology::bounds_for;
use crate::reconstruction::{Half, NeighborWalker};
use sitk_core::{Image, PixelId};

/// `TFunction1`/`TFunction2` in `itkValuedRegionalExtremaImageFilter.hxx`:
/// `std::greater` for `ValuedRegionalMaximaImageFilter`, `std::less` for
/// `ValuedRegionalMinimaImageFilter`.
#[derive(Clone, Copy)]
enum ExtremaKind {
    Maxima,
    Minima,
}

impl ExtremaKind {
    /// `compareIn`/`compareOut(a, b)`: `true` when `a` disqualifies a center
    /// pixel valued `b` from being an extremum.
    fn compare(self, a: f64, b: f64) -> bool {
        match self {
            ExtremaKind::Maxima => a > b,
            ExtremaKind::Minima => a < b,
        }
    }

    /// `MarkerValue`'s constructor default: `NumericTraits<T>::NonpositiveMin()`
    /// for maxima, `NumericTraits<T>::max()` for minima.
    fn marker_value(self, id: PixelId) -> f64 {
        let (max_value, nonpositive_min) = bounds_for(id);
        match self {
            ExtremaKind::Maxima => nonpositive_min,
            ExtremaKind::Minima => max_value,
        }
    }
}

/// The flooding engine shared by [`valued_regional_maxima`] and
/// [`valued_regional_minima`] -- see the module docs for why a separate
/// `marked` buffer is provably equivalent to the `.hxx`'s marker-value
/// aliasing. Returns `(values, flat)`.
fn valued_regional_extrema(
    vals: &[f64],
    size: &[usize],
    fully_connected: bool,
    kind: ExtremaKind,
    marker_value: f64,
) -> (Vec<f64>, bool) {
    let total = vals.len();
    if total == 0 || vals.iter().all(|&v| v == vals[0]) {
        return (vals.to_vec(), true);
    }

    let mut walker = NeighborWalker::new(size, fully_connected, Half::Full);
    let mut marked = vec![false; total];
    let mut stack: Vec<usize> = Vec::new();

    for f in 0..total {
        if marked[f] {
            continue;
        }
        let v = vals[f];
        let disqualified = walker.at(f, size).iter().any(|&g| kind.compare(vals[g], v));
        if !disqualified {
            continue;
        }
        marked[f] = true;
        stack.push(f);
        while let Some(i) = stack.pop() {
            for &g in walker.at(i, size) {
                if !marked[g] && vals[g] == v {
                    marked[g] = true;
                    stack.push(g);
                }
            }
        }
    }

    let out = vals
        .iter()
        .zip(&marked)
        .map(|(&v, &m)| if m { marker_value } else { v })
        .collect();
    (out, false)
}

/// Result of [`valued_regional_maxima`] / [`valued_regional_minima`],
/// mirroring `ValuedRegionalExtremaImageFilter`'s image output alongside its
/// `GetFlat()` measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct ValuedRegionalExtremaResult {
    pub image: Image,
    pub flat: bool,
}

fn valued_regional_extrema_image(
    image: &Image,
    fully_connected: bool,
    kind: ExtremaKind,
) -> Result<ValuedRegionalExtremaResult> {
    let size = image.size();
    let vals = image.to_f64_vec()?;
    let marker = kind.marker_value(image.pixel_id());
    let (out, flat) = valued_regional_extrema(&vals, size, fully_connected, kind, marker);
    let img = image_from_f64(image.pixel_id(), size, image, &out)?;
    Ok(ValuedRegionalExtremaResult { image: img, flat })
}

/// `ValuedRegionalMaximaImageFilter`: pixels belonging to a regional maximum
/// (a flat zone all of whose neighbors are strictly lower) keep their value;
/// every other pixel is set to `NumericTraits<T>::NonpositiveMin()`. A
/// completely flat image is one big maximum (`flat: true` in the result,
/// image unchanged).
pub fn valued_regional_maxima(
    image: &Image,
    fully_connected: bool,
) -> Result<ValuedRegionalExtremaResult> {
    valued_regional_extrema_image(image, fully_connected, ExtremaKind::Maxima)
}

/// `ValuedRegionalMinimaImageFilter`: the dual of [`valued_regional_maxima`]
/// -- surviving pixels are regional minima, non-surviving pixels are set to
/// `NumericTraits<T>::max()`.
pub fn valued_regional_minima(
    image: &Image,
    fully_connected: bool,
) -> Result<ValuedRegionalExtremaResult> {
    valued_regional_extrema_image(image, fully_connected, ExtremaKind::Minima)
}

/// `RegionalMaximaImageFilter`: a `UInt32` binary image where `foreground_value`
/// marks the regional maxima of `image` and `background_value` marks
/// everything else.
///
/// Delegates entirely to [`valued_regional_maxima`] (`GenerateData` builds no
/// flooding of its own): if the input is flat, the whole output is filled
/// from `flat_is_maxima` (`true` -> `foreground_value`, `false` ->
/// `background_value`); otherwise every pixel still holding the maxima
/// filter's marker value (`NonpositiveMin`, i.e. not a maximum) becomes
/// `background_value` and every other pixel becomes `foreground_value`.
pub fn regional_maxima(
    image: &Image,
    fully_connected: bool,
    flat_is_maxima: bool,
    foreground_value: u32,
    background_value: u32,
) -> Result<Image> {
    let result = valued_regional_maxima(image, fully_connected)?;
    let size = image.size();
    let total: usize = size.iter().product();

    let out: Vec<u32> = if result.flat {
        vec![
            if flat_is_maxima {
                foreground_value
            } else {
                background_value
            };
            total
        ]
    } else {
        let marker = ExtremaKind::Maxima.marker_value(image.pixel_id());
        result
            .image
            .to_f64_vec()?
            .iter()
            .map(|&v| {
                if v == marker {
                    background_value
                } else {
                    foreground_value
                }
            })
            .collect()
    };

    let mut out_image = Image::from_vec(size, out)?;
    out_image.copy_geometry_from(image);
    Ok(out_image)
}

/// `RegionalMinimaImageFilter`: a `UInt32` binary image where `foreground_value`
/// marks the regional minima of `image` and `background_value` marks
/// everything else. The mirror of [`regional_maxima`] -- see this module's
/// docs for the exact threshold shape shared by both.
pub fn regional_minima(
    image: &Image,
    fully_connected: bool,
    flat_is_minima: bool,
    foreground_value: u32,
    background_value: u32,
) -> Result<Image> {
    let result = valued_regional_minima(image, fully_connected)?;
    let size = image.size();
    let total: usize = size.iter().product();

    let out: Vec<u32> = if result.flat {
        vec![
            if flat_is_minima {
                foreground_value
            } else {
                background_value
            };
            total
        ]
    } else {
        let marker = ExtremaKind::Minima.marker_value(image.pixel_id());
        result
            .image
            .to_f64_vec()?
            .iter()
            .map(|&v| {
                if v == marker {
                    background_value
                } else {
                    foreground_value
                }
            })
            .collect()
    };

    let mut out_image = Image::from_vec(size, out)?;
    out_image.copy_geometry_from(image);
    Ok(out_image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_i32(size: &[usize], data: Vec<i32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- valued_regional_maxima / valued_regional_minima ----

    /// A raised plateau `[1,5,5,5,1]`: the plateau's neighbors (the two ends)
    /// are lower, so the whole 3-pixel plateau survives as one maximum; the
    /// two ends each have a higher neighbor and are flooded to
    /// `NonpositiveMin` (`i32::MIN`).
    #[test]
    fn valued_regional_maxima_keeps_peak_floors_the_rest_to_nonpositive_min() {
        let image = img_i32(&[5, 1], vec![1, 5, 5, 5, 1]);
        let result = valued_regional_maxima(&image, false).unwrap();
        assert!(!result.flat);
        assert_eq!(
            result.image.scalar_slice::<i32>().unwrap(),
            &[i32::MIN, 5, 5, 5, i32::MIN]
        );
    }

    /// Dual of the maxima case: a valley `[5,1,1,1,5]` keeps its 3-pixel
    /// minimum plateau, flooding the two higher ends to `max()`.
    #[test]
    fn valued_regional_minima_keeps_valley_floors_the_rest_to_max() {
        let image = img_i32(&[5, 1], vec![5, 1, 1, 1, 5]);
        let result = valued_regional_minima(&image, false).unwrap();
        assert!(!result.flat);
        assert_eq!(
            result.image.scalar_slice::<i32>().unwrap(),
            &[i32::MAX, 1, 1, 1, i32::MAX]
        );
    }

    /// `GetFlat()`: a completely constant image is reported flat and passes
    /// through unchanged, for both maxima and minima.
    #[test]
    fn flat_image_is_reported_flat_and_unchanged() {
        let image = img_i32(&[3, 3], vec![7; 9]);
        let maxima = valued_regional_maxima(&image, false).unwrap();
        assert!(maxima.flat);
        assert_eq!(maxima.image.scalar_slice::<i32>().unwrap(), &[7; 9]);

        let minima = valued_regional_minima(&image, false).unwrap();
        assert!(minima.flat);
        assert_eq!(minima.image.scalar_slice::<i32>().unwrap(), &[7; 9]);
    }

    /// A single pixel (value 0) whose only *diagonal* neighbor (9) would
    /// disqualify it as a maximum, while every *face* neighbor (-1) is
    /// lower: face connectivity misses the diagonal disqualifier and keeps
    /// it a maximum; full connectivity sees it and floods it away. The outer
    /// ring of -1 pixels is disqualified either way (pixel (1,0) face-touches
    /// the 9 directly), and floods through the whole face-connected ring.
    #[test]
    fn fully_connected_changes_a_diagonal_disqualifier() {
        #[rustfmt::skip]
        let image = img_i32(&[3, 3], vec![
            9, -1, -1,
            -1, 0, -1,
            -1, -1, -1,
        ]);

        let face = valued_regional_maxima(&image, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.image.scalar_slice::<i32>().unwrap(), &[
            9, i32::MIN, i32::MIN,
            i32::MIN, 0, i32::MIN,
            i32::MIN, i32::MIN, i32::MIN,
        ]);

        let full = valued_regional_maxima(&image, true).unwrap();
        #[rustfmt::skip]
        assert_eq!(full.image.scalar_slice::<i32>().unwrap(), &[
            9, i32::MIN, i32::MIN,
            i32::MIN, i32::MIN, i32::MIN,
            i32::MIN, i32::MIN, i32::MIN,
        ]);
    }

    // ---- regional_maxima ----

    /// The same raised-plateau fixture as the valued test above: the
    /// plateau becomes `foreground_value`, the two ends `background_value`.
    #[test]
    fn regional_maxima_thresholds_the_valued_output() {
        let image = img_i32(&[5, 1], vec![1, 5, 5, 5, 1]);
        let out = regional_maxima(&image, false, true, 1, 0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt32);
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[0, 1, 1, 1, 0]);
    }

    /// Non-default foreground/background values plumb straight through the
    /// threshold step.
    #[test]
    fn regional_maxima_custom_foreground_and_background_values() {
        let image = img_i32(&[5, 1], vec![1, 5, 5, 5, 1]);
        let out = regional_maxima(&image, false, true, 255, 7).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[7, 255, 255, 255, 7]);
    }

    /// `FlatIsMaxima` on a constant image: `true` fills every pixel with
    /// `foreground_value`, bypassing the marker-value threshold entirely.
    #[test]
    fn regional_maxima_flat_is_maxima_true_fills_foreground() {
        let image = img_i32(&[3, 1], vec![4, 4, 4]);
        let out = regional_maxima(&image, false, true, 1, 0).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 1]);
    }

    /// `FlatIsMaxima = false` on the same constant image fills every pixel
    /// with `background_value` instead.
    #[test]
    fn regional_maxima_flat_is_maxima_false_fills_background() {
        let image = img_i32(&[3, 1], vec![4, 4, 4]);
        let out = regional_maxima(&image, false, false, 1, 0).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[0, 0, 0]);
    }

    /// Diagonal connectivity change propagates through to the thresholded
    /// binary output, reusing the diagonal-disqualifier fixture.
    #[test]
    fn regional_maxima_fully_connected_changes_the_result() {
        #[rustfmt::skip]
        let image = img_i32(&[3, 3], vec![
            9, -1, -1,
            -1, 0, -1,
            -1, -1, -1,
        ]);

        let face = regional_maxima(&image, false, true, 1, 0).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.scalar_slice::<u32>().unwrap(), &[
            1, 0, 0,
            0, 1, 0,
            0, 0, 0,
        ]);

        let full = regional_maxima(&image, true, true, 1, 0).unwrap();
        #[rustfmt::skip]
        assert_eq!(full.scalar_slice::<u32>().unwrap(), &[
            1, 0, 0,
            0, 0, 0,
            0, 0, 0,
        ]);
    }

    // ---- regional_minima ----

    /// The dual of `regional_maxima_thresholds_the_valued_output`: a valley
    /// `[5,1,1,1,5]` keeps its 3-pixel minimum plateau as `foreground_value`,
    /// the two higher ends become `background_value`.
    #[test]
    fn regional_minima_thresholds_the_valued_output() {
        let image = img_i32(&[5, 1], vec![5, 1, 1, 1, 5]);
        let out = regional_minima(&image, false, true, 1, 0).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt32);
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[0, 1, 1, 1, 0]);
    }

    /// Mirror parity: negating `regional_maxima`'s own raised-plateau fixture
    /// turns the maximum plateau into a minimum plateau (surrounded by
    /// strictly *higher* neighbors instead of lower ones), so
    /// `regional_minima` on the negated image reproduces the exact same
    /// foreground/background pattern `regional_maxima` computed on the
    /// original.
    #[test]
    fn regional_minima_mirrors_regional_maxima_on_negated_fixture() {
        let image = img_i32(&[5, 1], vec![1, 5, 5, 5, 1]);
        let maxima = regional_maxima(&image, false, true, 1, 0).unwrap();

        let negated = img_i32(&[5, 1], vec![-1, -5, -5, -5, -1]);
        let minima = regional_minima(&negated, false, true, 1, 0).unwrap();

        assert_eq!(
            minima.scalar_slice::<u32>().unwrap(),
            maxima.scalar_slice::<u32>().unwrap()
        );
    }

    /// `FlatIsMinima` on a constant image: `true` fills every pixel with
    /// `foreground_value`, bypassing the marker-value threshold entirely.
    #[test]
    fn regional_minima_flat_is_minima_true_fills_foreground() {
        let image = img_i32(&[3, 1], vec![4, 4, 4]);
        let out = regional_minima(&image, false, true, 1, 0).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[1, 1, 1]);
    }

    /// `FlatIsMinima = false` on the same constant image fills every pixel
    /// with `background_value` instead.
    #[test]
    fn regional_minima_flat_is_minima_false_fills_background() {
        let image = img_i32(&[3, 1], vec![4, 4, 4]);
        let out = regional_minima(&image, false, false, 1, 0).unwrap();
        assert_eq!(out.scalar_slice::<u32>().unwrap(), &[0, 0, 0]);
    }
}
