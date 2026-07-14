//! ITK's binary-morphology utility filters: hole filling/peak grinding on a
//! foreground indicator, neighborhood voting, a binary-optimized median, and
//! 2-D skeleton thinning.
//!
//! Verified against ITK's `Modules/Filtering/LabelMap/include/`,
//! `Modules/Segmentation/LabelVoting/include/`, and
//! `Modules/Filtering/BinaryMathematicalMorphology/include/`:
//!
//! - [`binary_fillhole`] / [`binary_grind_peak`] —
//!   `itkBinaryFillholeImageFilter.h`/`.hxx`,
//!   `itkBinaryGrindPeakImageFilter.h`/`.hxx`. ITK's own `.hxx` builds these
//!   from `BinaryImageToShapeLabelMapFilter` +
//!   `ShapeOpeningLabelMapFilter(NUMBER_OF_PIXELS_ON_BORDER, Lambda=1)` +
//!   `LabelMapMaskImageFilter`/`LabelMapToBinaryImageFilter` —
//!   connected-component/label-map machinery this crate doesn't carry, and
//!   each class's own doc comment names the equivalence this port takes
//!   instead (`\sa GrayscaleFillholeImageFilter` /
//!   `\sa GrayscaleGrindPeakImageFilter`): binary fillhole/grindpeak on
//!   `image == foreground_value`'s 0/1 indicator is exactly
//!   [`crate::reconstruction::grayscale_fillhole`] /
//!   [`crate::reconstruction::grayscale_grindpeak`] on that indicator — a
//!   background pixel is an unreachable "hole" iff it isn't connected to the
//!   image border through other 0s, which is precisely what
//!   `grayscale_fillhole`'s interior-minimum reconstruction raises, and dually
//!   for grindpeak's interior-maximum suppression. Tracing the label-map
//!   pipeline end to end (`LabelMapMaskImageFilter`/`LabelMapToBinaryImageFilter`
//!   `.hxx`) confirms both filters' final pixel values reduce to:
//!   fillhole — `foreground_value` wherever the indicator's fillhole
//!   reconstruction is `1`, else the original pixel value (label-preserving,
//!   like [`crate::morphology`]'s binary erode/dilate); grindpeak —
//!   `foreground_value` wherever the indicator's grindpeak reconstruction is
//!   `1`, `background_value` wherever the original pixel *was*
//!   `foreground_value` but the reconstruction zeroed it (an interior,
//!   non-border-connected foreground island), else the original value. This
//!   reuses `reconstruction.rs`'s engine rather than re-deriving connected
//!   components or duplicating its raster/anti-raster/FIFO loop.
//! - [`voting_binary`] — `itkVotingBinaryImageFilter.h`/`.hxx`: a box
//!   neighborhood vote (`ZeroFluxNeumannBoundaryCondition`, replicate-edge
//!   padding) counting neighborhood positions — *including the center* —
//!   equal to `foreground_value`; a `background_value` center becomes
//!   foreground when that count is `>= birth_threshold`, a `foreground_value`
//!   center becomes background when it's `< survival_threshold` (the center's
//!   own foreground-ness counts toward its own survival count), any other
//!   center value (or a threshold miss) passes through unchanged.
//! - [`voting_binary_hole_filling`] / [`voting_binary_iterative_hole_filling`] —
//!   `itkVotingBinaryHoleFillingImageFilter.h`/`.hxx` (a single pass) and
//!   `itkVotingBinaryIterativeHoleFillingImageFilter.h`/`.hxx` (which runs the
//!   former in a loop). The pass fixes `survival_threshold = 0` (a foreground
//!   center never dies) and derives its birth threshold from
//!   `majority_threshold` (`BeforeThreadedGenerateData`: `birth_threshold =
//!   (Π(2·radius[d]+1) - 1) / 2 + majority_threshold`, integer division — the
//!   box neighbor count excluding the center, halved, plus the caller's
//!   margin over 50%); unlike `voting_binary`, every non-`background_value`
//!   center (not just an exact `foreground_value` one) is unconditionally
//!   stamped to `foreground_value` each pass, and its boundary condition is
//!   `ZeroFluxNeumannBoundaryCondition` (replicated, not cropped, unlike
//!   [`crate::rank::rank`]'s neighborhood). [`voting_binary_hole_filling`]
//!   runs this pass exactly once and reports `GetNumberOfPixelsChanged()`
//!   directly; [`voting_binary_iterative_hole_filling`]'s outer loop reruns
//!   it, input := previous output, until either
//!   `maximum_number_of_iterations` passes have run or a pass changes zero
//!   pixels (`m_NumberOfPixelsChanged == 0`, counted only for birth-triggered
//!   flips, exactly as the `.hxx` does). `VotingBinaryHoleFillingImageFilter.yaml`
//!   restricts `pixel_types` to `IntegerPixelIDTypeList`;
//!   [`voting_binary_hole_filling`] enforces this via
//!   [`crate::logic::require_integer_pixel_type`], but [`voting_binary`] and
//!   [`voting_binary_iterative_hole_filling`] declare the same restriction in
//!   their own yamls without enforcing it — a pre-existing gap in this
//!   port, outside this change's scope.
//! - [`binary_median`] — `itkBinaryMedianImageFilter.h`/`.hxx`: a
//!   `ZeroFluxNeumannBoundaryCondition` box neighborhood vote, like
//!   [`voting_binary`], but with a fixed rule and no birth/survival split:
//!   `count` is the number of neighborhood positions (including the center)
//!   equal to `foreground_value`; the output is `foreground_value` when
//!   `count > neighborhoodSize / 2` (integer division — for the always-odd
//!   `Π(2·radius[d]+1)` window this is the true majority, but the `.hxx`
//!   itself computes it as a plain truncating division, not a parity-aware
//!   median rule), else `background_value` unconditionally. Unlike
//!   [`voting_binary`], a center pixel equal to neither `foreground_value`
//!   nor `background_value` does *not* pass through: the `.hxx` only ever
//!   counts foreground matches and always writes one of the two output
//!   values (`DynamicThreadedGenerateData`'s `if (count > medianPosition) ...
//!   else ...` has no third branch). The center's own value only costs one
//!   potential foreground vote — with a foreground-majority neighborhood
//!   such a pixel still becomes `foreground_value`.
//! - [`binary_thinning`] — `itkBinaryThinningImageFilter.h`/`.hxx`: the
//!   Gonzalez–Woods sequential thinning algorithm. ITK only wraps this filter
//!   for 2-D images (`itkBinaryThinningImageFilter.wrap`'s
//!   `itk_wrap_image_filter(..., 2, 2)`) — `ComputeThinImage`'s neighbor
//!   offsets are hardcoded 2-element `OffsetType`s, so a higher-dimensional
//!   instantiation would not even compile in C++; this port returns
//!   [`FilterError::UnsupportedThinningDimension`] instead. Any nonzero input
//!   pixel is treated as foreground (`PrepareData`'s `if (it.Get())`, not
//!   compared against a settable foreground value); output is `0`/`1`. Each
//!   outer round runs four sub-passes (`step` 1..=4) over the 8-neighborhood
//!   (Gonzalez & Woods' `p2..p9`, clockwise from north, `ZeroFluxNeumann`
//!   boundary — ITK's default `NeighborhoodIterator` boundary condition,
//!   since `ComputeThinImage` never overrides it); a foreground pixel is
//!   marked for deletion when all four hold: `testA` (2–6 of its 8 neighbors
//!   are on — not an endpoint, not an interior fill pixel), `testB` (exactly
//!   one 0→1 transition walking the neighbors cyclically — deleting it
//!   wouldn't disconnect a 1-pixel-wide stroke), and a `testC`/`testD` pair
//!   that rotates which cardinal neighbors must be off across the four steps
//!   (`step`s 1/2 clear east/south-boundary and north/west-boundary points,
//!   `step`s 3/4 clear the complementary west/north and south/east ones,
//!   together covering every boundary orientation over a full round).
//!   Deletions found within one sub-pass are collected and applied only after
//!   that sub-pass finishes scanning (so every read in a sub-pass sees the
//!   state from before it), and outer rounds repeat until a full round of
//!   four sub-passes deletes nothing.

use crate::error::{FilterError, Result};
use crate::reconstruction::{grayscale_fillhole, grayscale_grindpeak};
use sitk_core::{
    Image, NeighborhoodIterator, Scalar, ZeroFluxNeumannBoundaryCondition, dispatch_scalar,
};

// ---- binary_fillhole / binary_grind_peak -----------------------------------

fn indicator_typed<T: Scalar>(image: &Image, foreground: f64) -> Result<Image> {
    let foreground = T::from_f64(foreground);
    let one = T::from_f64(1.0);
    let zero = T::from_f64(0.0);
    let out: Vec<T> = image
        .scalar_slice::<T>()?
        .iter()
        .map(|&v| if v == foreground { one } else { zero })
        .collect();
    let mut result = Image::from_vec(image.size(), out)?;
    result.copy_geometry_from(image);
    Ok(result)
}

/// The 0/1 indicator of `image == foreground_value`, same pixel type and
/// geometry as `image` (see module docs).
fn indicator_image(image: &Image, foreground: f64) -> Result<Image> {
    dispatch_scalar!(image.pixel_id(), indicator_typed, image, foreground)
}

fn fillhole_restore_typed<T: Scalar>(
    original: &Image,
    filled_indicator: &Image,
    foreground: f64,
) -> Result<Image> {
    let foreground = T::from_f64(foreground);
    let out: Vec<T> = original
        .scalar_slice::<T>()?
        .iter()
        .zip(filled_indicator.scalar_slice::<T>()?)
        .map(|(&o, &i)| if i.as_f64() != 0.0 { foreground } else { o })
        .collect();
    let mut result = Image::from_vec(original.size(), out)?;
    result.copy_geometry_from(original);
    Ok(result)
}

/// `foreground_value` wherever `filled_indicator` is `1`, else `original`'s
/// own value (label-preserving on the untouched background, matching the
/// `.hxx`'s final `LabelMapMaskImageFilter` pass — see module docs).
fn fillhole_restore(original: &Image, filled_indicator: &Image, foreground: f64) -> Result<Image> {
    dispatch_scalar!(
        original.pixel_id(),
        fillhole_restore_typed,
        original,
        filled_indicator,
        foreground
    )
}

fn grindpeak_restore_typed<T: Scalar>(
    original: &Image,
    grinded_indicator: &Image,
    foreground: f64,
    background: f64,
) -> Result<Image> {
    let foreground = T::from_f64(foreground);
    let background = T::from_f64(background);
    let out: Vec<T> = original
        .scalar_slice::<T>()?
        .iter()
        .zip(grinded_indicator.scalar_slice::<T>()?)
        .map(|(&o, &i)| {
            if i.as_f64() != 0.0 {
                foreground
            } else if o == foreground {
                background
            } else {
                o
            }
        })
        .collect();
    let mut result = Image::from_vec(original.size(), out)?;
    result.copy_geometry_from(original);
    Ok(result)
}

/// `foreground_value` wherever `grinded_indicator` is `1`; `background_value`
/// wherever `original` was `foreground_value` but got grinded away; else
/// `original`'s own value (see module docs).
fn grindpeak_restore(
    original: &Image,
    grinded_indicator: &Image,
    foreground: f64,
    background: f64,
) -> Result<Image> {
    dispatch_scalar!(
        original.pixel_id(),
        grindpeak_restore_typed,
        original,
        grinded_indicator,
        foreground,
        background
    )
}

/// `BinaryFillholeImageFilter` (`itkBinaryFillholeImageFilter.hxx`): flip
/// every background pixel not reachable from the image border through other
/// background pixels to `foreground_value`; every already-foreground pixel,
/// and every border-connected background pixel, keeps its original value.
/// See the module docs for the reconstruction-engine equivalence this port
/// uses in place of ITK's label-map minipipeline.
pub fn binary_fillhole(
    image: &Image,
    foreground_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    let indicator = indicator_image(image, foreground_value)?;
    let filled = grayscale_fillhole(&indicator, fully_connected)?;
    fillhole_restore(image, &filled, foreground_value)
}

/// `BinaryGrindPeakImageFilter` (`itkBinaryGrindPeakImageFilter.hxx`): the
/// dual of [`binary_fillhole`] — every foreground object not connected to the
/// image border is reduced to `background_value`; border-connected foreground
/// and every other pixel keep their original value. See the module docs.
pub fn binary_grind_peak(
    image: &Image,
    foreground_value: f64,
    background_value: f64,
    fully_connected: bool,
) -> Result<Image> {
    let indicator = indicator_image(image, foreground_value)?;
    let grinded = grayscale_grindpeak(&indicator, fully_connected)?;
    grindpeak_restore(image, &grinded, foreground_value, background_value)
}

// ---- voting_binary ----------------------------------------------------------

fn voting_binary_typed<T: Scalar>(
    img: &Image,
    radius: &[usize],
    birth_threshold: u32,
    survival_threshold: u32,
    foreground: f64,
    background: f64,
) -> Result<Image> {
    let foreground = T::from_f64(foreground);
    let background = T::from_f64(background);
    let iter = NeighborhoodIterator::<T, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)?;
    let center_slot = iter.len() / 2;
    // Parallel over output voxels. Every output pixel is a function of its own
    // window alone — the neighbor count is an *equality* count over the borrowed
    // window in its own order, and the birth/survival test is unchanged — so no
    // value crosses voxels and no thread count can reach the result.
    let out: Vec<T> = iter.par_map_window(|_, w| {
        let inpixel = w.get(center_slot);
        let count = w.iter().filter(|&v| v == foreground).count() as u32;
        if inpixel == background && count >= birth_threshold {
            foreground
        } else if inpixel == foreground && count < survival_threshold {
            background
        } else {
            inpixel
        }
    });
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `VotingBinaryImageFilter` (`itkVotingBinaryImageFilter.hxx`): a
/// birth/survival vote over a box neighborhood of the given per-axis
/// `radius`, `ZeroFluxNeumannBoundaryCondition` at the image border (see
/// module docs for the exact rule, including the center pixel's own
/// contribution to its survival count). Pixels equal to neither
/// `foreground_value` nor `background_value` are left unchanged, matching the
/// class doc's own note.
pub fn voting_binary(
    img: &Image,
    radius: &[usize],
    birth_threshold: u32,
    survival_threshold: u32,
    foreground_value: f64,
    background_value: f64,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        voting_binary_typed,
        img,
        radius,
        birth_threshold,
        survival_threshold,
        foreground_value,
        background_value
    )
}

// ---- voting_binary_iterative_hole_filling ------------------------------------

fn voting_binary_hole_filling_pass_typed<T: Scalar>(
    img: &Image,
    radius: &[usize],
    birth_threshold: u32,
    foreground: f64,
    background: f64,
) -> Result<(Image, u32)> {
    let foreground = T::from_f64(foreground);
    let background = T::from_f64(background);
    let iter = NeighborhoodIterator::<T, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)?;
    let center_slot = iter.len() / 2;

    // Parallel over output voxels, each carrying its own `changed` flag rather
    // than incrementing a shared counter.
    //
    // `m_NumberOfPixelsChanged` is the one quantity here that crosses voxels, and
    // it is deliberately NOT accumulated inside the parallel pass. It is recovered
    // afterwards by counting the flags, in voxel order, on one thread. That keeps
    // the count a pure function of the output — it is a `usize` tally of a
    // predicate, so it is exact and could not have been re-associated into a
    // different answer anyway, but tallying it here rather than in the workers
    // means there is no shared accumulator to reason about at all.
    let flagged: Vec<(T, bool)> = iter.par_map_window(|_, w| {
        let inpixel = w.get(center_slot);
        if inpixel == background {
            let count = w.iter().filter(|&v| v == foreground).count() as u32;
            if count >= birth_threshold {
                (foreground, true)
            } else {
                (background, false)
            }
        } else {
            // itkVotingBinaryHoleFillingImageFilter.hxx's unconditional
            // `else { it.Set(foregroundValue); }`: any non-background center
            // is stamped to foreground_value, not just an exact-foreground
            // one, and this branch never contributes to the changed count.
            // Tracked in the upstream-findings ledger, §2.70.
            (foreground, false)
        }
    });

    let changed = flagged.iter().filter(|(_, c)| *c).count() as u32;
    let out: Vec<T> = flagged.into_iter().map(|(v, _)| v).collect();

    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok((result, changed))
}

/// One `VotingBinaryHoleFillingImageFilter` pass at a precomputed
/// `birth_threshold` (see module docs); returns the pass's output and its
/// `m_NumberOfPixelsChanged`.
fn voting_binary_hole_filling_pass(
    img: &Image,
    radius: &[usize],
    birth_threshold: u32,
    foreground: f64,
    background: f64,
) -> Result<(Image, u32)> {
    dispatch_scalar!(
        img.pixel_id(),
        voting_binary_hole_filling_pass_typed,
        img,
        radius,
        birth_threshold,
        foreground,
        background
    )
}

/// [`voting_binary_hole_filling`]'s result: the filled image plus
/// `VotingBinaryHoleFillingImageFilter::GetNumberOfPixelsChanged()`.
#[derive(Clone, Debug, PartialEq)]
pub struct VotingBinaryHoleFillingResult {
    /// The image after one hole-filling pass.
    pub image: Image,
    /// Number of background-valued pixels this pass flipped to
    /// `foreground_value` (`AfterThreadedGenerateData`'s summed per-thread
    /// count) — never counts the unconditional
    /// non-background-to-foreground stamp (see the module docs).
    pub number_of_pixels_changed: u32,
}

/// `VotingBinaryHoleFillingImageFilter`
/// (`itkVotingBinaryHoleFillingImageFilter.h(.hxx)`): a single application of
/// the same per-pass rule [`voting_binary_iterative_hole_filling`] loops —
/// computes `birth_threshold` from `majority_threshold` and `radius`
/// (`BeforeThreadedGenerateData`, see module docs), then runs
/// [`voting_binary_hole_filling_pass`] exactly once, unlike the iterative
/// filter's convergence loop.
///
/// Requires an integer pixel type (`VotingBinaryHoleFillingImageFilter.yaml`'s
/// `IntegerPixelIDTypeList`), unlike its sibling functions in this module.
pub fn voting_binary_hole_filling(
    img: &Image,
    radius: &[usize],
    majority_threshold: u32,
    foreground_value: f64,
    background_value: f64,
) -> Result<VotingBinaryHoleFillingResult> {
    crate::logic::require_integer_pixel_type(img)?;
    let neighborhood_size: usize = radius.iter().map(|&r| 2 * r + 1).product();
    let birth_threshold = ((neighborhood_size - 1) / 2) as u32 + majority_threshold;

    let (image, number_of_pixels_changed) = voting_binary_hole_filling_pass(
        img,
        radius,
        birth_threshold,
        foreground_value,
        background_value,
    )?;
    Ok(VotingBinaryHoleFillingResult {
        image,
        number_of_pixels_changed,
    })
}

/// `VotingBinaryIterativeHoleFillingImageFilter`
/// (`itkVotingBinaryIterativeHoleFillingImageFilter.hxx`): repeatedly apply
/// [`voting_binary_hole_filling_pass`]'s per-iteration rule, input := previous
/// output, until either `maximum_number_of_iterations` passes have run or a
/// pass changes no pixels. See the module docs for `birth_threshold`'s
/// derivation from `majority_threshold` and `radius`.
pub fn voting_binary_iterative_hole_filling(
    img: &Image,
    radius: &[usize],
    majority_threshold: u32,
    maximum_number_of_iterations: u32,
    foreground_value: f64,
    background_value: f64,
) -> Result<Image> {
    let neighborhood_size: usize = radius.iter().map(|&r| 2 * r + 1).product();
    let birth_threshold = ((neighborhood_size - 1) / 2) as u32 + majority_threshold;

    let mut current = img.clone();
    for _ in 0..maximum_number_of_iterations {
        let (next, changed) = voting_binary_hole_filling_pass(
            &current,
            radius,
            birth_threshold,
            foreground_value,
            background_value,
        )?;
        current = next;
        if changed == 0 {
            break;
        }
    }
    Ok(current)
}

// ---- binary_median ------------------------------------------------------------

fn binary_median_typed<T: Scalar>(
    img: &Image,
    radius: &[usize],
    foreground: f64,
    background: f64,
) -> Result<Image> {
    let foreground = T::from_f64(foreground);
    let background = T::from_f64(background);
    let iter = NeighborhoodIterator::<T, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)?;
    let median_position = iter.len() / 2;
    // Parallel over output voxels: the majority vote is a count over one voxel's
    // own window, so nothing crosses voxels.
    let out: Vec<T> = iter.par_map_window(|_, w| {
        let count = w.iter().filter(|&v| v == foreground).count();
        if count > median_position {
            foreground
        } else {
            background
        }
    });
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `BinaryMedianImageFilter` (`itkBinaryMedianImageFilter.hxx`): the binary
/// majority-vote median over a box neighborhood of the given per-axis
/// `radius`, `ZeroFluxNeumannBoundaryCondition` at the image border (see
/// module docs for the exact majority rule and the neither-value case).
pub fn binary_median(
    img: &Image,
    radius: &[usize],
    foreground_value: f64,
    background_value: f64,
) -> Result<Image> {
    let dim = img.dimension();
    if radius.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: radius.len(),
        });
    }
    dispatch_scalar!(
        img.pixel_id(),
        binary_median_typed,
        img,
        radius,
        foreground_value,
        background_value
    )
}

// ---- binary_thinning ---------------------------------------------------------

fn thinning_indicator_typed<T: Scalar>(image: &Image) -> Result<Vec<u8>> {
    Ok(image
        .scalar_slice::<T>()?
        .iter()
        .map(|&v| u8::from(v.as_f64() != 0.0))
        .collect())
}

/// `BinaryThinningImageFilter::PrepareData`: any nonzero input pixel is
/// foreground (not compared against a settable foreground value).
fn thinning_indicator(image: &Image) -> Result<Vec<u8>> {
    dispatch_scalar!(image.pixel_id(), thinning_indicator_typed, image)
}

/// The pixel at `(x, y)`, `ZeroFluxNeumannBoundaryCondition`-clamped to
/// `[0, w) x [0, h)`.
fn clamped_get(data: &[u8], w: i64, h: i64, x: i64, y: i64) -> u8 {
    let cx = x.clamp(0, w - 1) as usize;
    let cy = y.clamp(0, h - 1) as usize;
    data[cx + w as usize * cy]
}

/// `BinaryThinningImageFilter::ComputeThinImage`: the Gonzalez–Woods
/// sequential thinning algorithm (see module docs for `testA`..`testD` and
/// the four-sub-pass structure of one round).
fn compute_thin_image(indicator: &[u8], size: &[usize]) -> Vec<u8> {
    let w = size[0] as i64;
    let h = size[1] as i64;
    let mut data = indicator.to_vec();

    let mut no_change = false;
    while !no_change {
        no_change = true;
        for step in 1..=4 {
            let mut to_delete = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let idx = (x + w * y) as usize;
                    if data[idx] == 0 {
                        continue;
                    }
                    let p2 = clamped_get(&data, w, h, x, y - 1);
                    let p3 = clamped_get(&data, w, h, x + 1, y - 1);
                    let p4 = clamped_get(&data, w, h, x + 1, y);
                    let p5 = clamped_get(&data, w, h, x + 1, y + 1);
                    let p6 = clamped_get(&data, w, h, x, y + 1);
                    let p7 = clamped_get(&data, w, h, x - 1, y + 1);
                    let p8 = clamped_get(&data, w, h, x - 1, y);
                    let p9 = clamped_get(&data, w, h, x - 1, y - 1);

                    let number_of_on_neighbors = p2 as i32
                        + p3 as i32
                        + p4 as i32
                        + p5 as i32
                        + p6 as i32
                        + p7 as i32
                        + p8 as i32
                        + p9 as i32;
                    let test_a = number_of_on_neighbors > 1 && number_of_on_neighbors < 7;

                    let transitions = ((p3 as i32 - p2 as i32).abs()
                        + (p4 as i32 - p3 as i32).abs()
                        + (p5 as i32 - p4 as i32).abs()
                        + (p6 as i32 - p5 as i32).abs()
                        + (p7 as i32 - p6 as i32).abs()
                        + (p8 as i32 - p7 as i32).abs()
                        + (p9 as i32 - p8 as i32).abs()
                        + (p2 as i32 - p9 as i32).abs())
                        / 2;
                    let test_b = transitions == 1;

                    // testC and testD are always set identically in the
                    // .hxx; collapsed to one boolean here.
                    let test_cd = match step {
                        1 => p4 == 0 || p6 == 0,
                        2 => p2 == 0 && p8 == 0,
                        3 => p2 == 0 || p8 == 0,
                        4 => p4 == 0 && p6 == 0,
                        _ => unreachable!("step is always in 1..=4"),
                    };

                    if test_a && test_b && test_cd {
                        to_delete.push(idx);
                        no_change = false;
                    }
                }
            }
            for idx in to_delete {
                data[idx] = 0;
            }
        }
    }
    data
}

/// `BinaryThinningImageFilter` (`itkBinaryThinningImageFilter.hxx`):
/// one-pixel-wide skeleton of a binary image, 2-D only (see module docs).
/// Output pixel type matches `image`'s; values are `0`/`1`.
pub fn binary_thinning(image: &Image) -> Result<Image> {
    let size = image.size();
    if size.len() != 2 {
        return Err(FilterError::UnsupportedThinningDimension(size.len()));
    }
    let indicator = thinning_indicator(image)?;
    let thinned = compute_thin_image(&indicator, size);
    let vals: Vec<f64> = thinned.iter().map(|&v| v as f64).collect();
    crate::image_from_f64(image.pixel_id(), size, image, &vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    use sitk_core::PixelId;

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- binary_fillhole ----

    #[test]
    fn binary_fillhole_fills_enclosed_hole_but_not_border_touching_one() {
        #[rustfmt::skip]
        let image = img_u8(&[5, 5], vec![
            255, 255, 255, 255, 255,
            255, 255, 255, 255, 255,
            255, 255,   0, 255, 255,
            255, 255, 255, 255, 255,
              0, 255, 255, 255, 255,
        ]);
        let out = binary_fillhole(&image, 255.0, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            255, 255, 255, 255, 255,
            255, 255, 255, 255, 255,
            255, 255, 255, 255, 255,
            255, 255, 255, 255, 255,
              0, 255, 255, 255, 255,
        ]);
    }

    #[test]
    fn binary_fillhole_default_foreground_value_one() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            1, 1, 1,
            1, 0, 1,
            1, 1, 1,
        ]);
        let out = binary_fillhole(&image, 1.0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            &[1, 1, 1, 1, 1, 1, 1, 1, 1]
        );
    }

    /// Only the 3x3 center pixel is interior; the corner-to-corner diagonal
    /// hole only reaches the border under full connectivity.
    #[test]
    fn binary_fillhole_fully_connected_changes_whether_a_diagonal_leak_reaches_the_border() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            0, 255, 255,
            255,   0, 255,
            255, 255,   0,
        ]);
        let face = binary_fillhole(&image, 255.0, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.scalar_slice::<u8>().unwrap(), &[
            0, 255, 255,
            255, 255, 255,
            255, 255,   0,
        ]);

        let full = binary_fillhole(&image, 255.0, true).unwrap();
        assert_eq!(
            full.scalar_slice::<u8>().unwrap(),
            image.scalar_slice::<u8>().unwrap()
        );
    }

    // ---- binary_grind_peak ----

    #[test]
    fn binary_grind_peak_removes_interior_island_but_not_border_touching_one() {
        #[rustfmt::skip]
        let image = img_u8(&[5, 5], vec![
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 255, 0, 0,
            0, 0, 0, 0, 0,
            255, 255, 0, 0, 0,
        ]);
        let out = binary_grind_peak(&image, 255.0, 0.0, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            255, 255, 0, 0, 0,
        ]);
    }

    #[test]
    fn binary_grind_peak_uses_background_value_for_grinded_pixels() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            0, 0, 0,
            0, 9, 0,
            0, 0, 0,
        ]);
        let out = binary_grind_peak(&image, 9.0, 3.0, false).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            &[0, 0, 0, 0, 3, 0, 0, 0, 0]
        );
    }

    #[test]
    fn binary_grind_peak_fully_connected_changes_whether_a_diagonal_island_reaches_the_border() {
        #[rustfmt::skip]
        let image = img_u8(&[3, 3], vec![
            255,   0,   0,
              0, 255,   0,
              0,   0, 255,
        ]);
        let face = binary_grind_peak(&image, 255.0, 0.0, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(face.scalar_slice::<u8>().unwrap(), &[
            255,   0,   0,
              0,   0,   0,
              0,   0, 255,
        ]);

        let full = binary_grind_peak(&image, 255.0, 0.0, true).unwrap();
        assert_eq!(
            full.scalar_slice::<u8>().unwrap(),
            image.scalar_slice::<u8>().unwrap()
        );
    }

    // ---- voting_binary ----

    /// 1-D radius-1 neighborhood (3 pixels incl. center): a background center
    /// with exactly `birth_threshold` foreground neighbors flips; one fewer
    /// does not.
    #[test]
    fn voting_binary_birth_threshold_is_exact() {
        // center=0 (background), neighbors [1,1]: count=2.
        let two_on = img_u8(&[3], vec![1, 0, 1]);
        let flips = voting_binary(&two_on, &[1], 2, 1, 1.0, 0.0).unwrap();
        assert_eq!(flips.scalar_slice::<u8>().unwrap()[1], 1);

        // center=0 (background), neighbors [1,0]: count=1 < birth_threshold=2.
        let one_on = img_u8(&[3], vec![1, 0, 0]);
        let stays = voting_binary(&one_on, &[1], 2, 1, 1.0, 0.0).unwrap();
        assert_eq!(stays.scalar_slice::<u8>().unwrap()[1], 0);
    }

    /// Survival counts the center's own foreground contribution: with
    /// `survival_threshold = 2`, a foreground center with one foreground
    /// neighbor (count = 2, including itself) survives; zero neighbors
    /// (count = 1) dies.
    #[test]
    fn voting_binary_survival_threshold_includes_the_center() {
        let one_neighbor = img_u8(&[3], vec![1, 1, 0]);
        let survives = voting_binary(&one_neighbor, &[1], 1, 2, 1.0, 0.0).unwrap();
        assert_eq!(survives.scalar_slice::<u8>().unwrap()[1], 1);

        let no_neighbor = img_u8(&[3], vec![0, 1, 0]);
        let dies = voting_binary(&no_neighbor, &[1], 1, 2, 1.0, 0.0).unwrap();
        assert_eq!(dies.scalar_slice::<u8>().unwrap()[1], 0);
    }

    #[test]
    fn voting_binary_leaves_other_values_unchanged() {
        let image = img_u8(&[3], vec![1, 5, 1]);
        let out = voting_binary(&image, &[1], 1, 1, 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 5, 1]);
    }

    #[test]
    fn voting_binary_zero_flux_neumann_clamps_at_the_border() {
        // index0 is the left edge; ZeroFluxNeumann clamps the out-of-bounds
        // offset -1 back to index0 itself, so index0's radius-1 window is
        // [self, self, index1] = [1, 1, 0]: its own foreground value is
        // counted twice.
        let image = img_u8(&[3], vec![1, 0, 0]);
        let survives = voting_binary(&image, &[1], 1, 2, 1.0, 0.0).unwrap();
        assert_eq!(survives.scalar_slice::<u8>().unwrap()[0], 1);
        let dies = voting_binary(&image, &[1], 1, 3, 1.0, 0.0).unwrap();
        assert_eq!(dies.scalar_slice::<u8>().unwrap()[0], 0);
    }

    // ---- voting_binary_hole_filling ----

    /// 1-D radius-1 neighborhood (3 pixels incl. center), `MajorityThreshold`
    /// left at its yaml default of 1: `birth_threshold = (3-1)/2 + 1 = 2`. A
    /// background center with both neighbors on flips; one fewer does not --
    /// same shape as [`voting_binary`]'s own birth-threshold test, since this
    /// is the same `birth_threshold` derivation with `majority_threshold`
    /// baked to 1.
    #[test]
    fn voting_binary_hole_filling_yaml_default_majority_threshold_birth_threshold_is_exact() {
        let two_on = img_u8(&[3], vec![1, 0, 1]);
        let flips = voting_binary_hole_filling(&two_on, &[1], 1, 1.0, 0.0).unwrap();
        assert_eq!(flips.image.scalar_slice::<u8>().unwrap()[1], 1);
        assert_eq!(flips.number_of_pixels_changed, 1);

        let one_on = img_u8(&[3], vec![1, 0, 0]);
        let stays = voting_binary_hole_filling(&one_on, &[1], 1, 1.0, 0.0).unwrap();
        assert_eq!(stays.image.scalar_slice::<u8>().unwrap()[1], 0);
        assert_eq!(stays.number_of_pixels_changed, 0);
    }

    /// A center pixel equal to neither `foreground_value` nor
    /// `background_value` is unconditionally stamped to `foreground_value`
    /// regardless of its neighbor count (module docs) -- unlike
    /// [`voting_binary`], there is no pass-through branch.
    #[test]
    fn voting_binary_hole_filling_stamps_any_non_background_value_to_foreground() {
        let image = img_u8(&[3], vec![0, 5, 0]); // center=5, zero on-neighbors
        let out = voting_binary_hole_filling(&image, &[1], 1, 1.0, 0.0).unwrap();
        assert_eq!(out.image.scalar_slice::<u8>().unwrap()[1], 1);
        assert_eq!(out.number_of_pixels_changed, 0); // the unconditional stamp never counts as "changed"
    }

    /// `GetNumberOfPixelsChanged()` counts only birth-triggered
    /// background-to-foreground flips, never the unconditional
    /// non-background-to-foreground stamp -- both occur in the same pass
    /// here, and only one should be counted.
    #[test]
    fn voting_binary_hole_filling_changed_count_excludes_unconditional_stamps() {
        #[rustfmt::skip]
        let image = img_u8(&[5, 5], vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 5, 1,
        ]);
        let result = voting_binary_hole_filling(&image, &[1, 1], 1, 1.0, 0.0).unwrap();
        let got = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(got[12], 1); // the former lone hole: a real birth-count flip
        assert_eq!(got[23], 1); // the "neither" pixel: an unconditional stamp
        assert_eq!(result.number_of_pixels_changed, 1);
    }

    /// `voting_binary_hole_filling` is exactly
    /// [`voting_binary_iterative_hole_filling`]'s inner pass run once, not a
    /// convergence loop: a 3x3 hole too large for the birth threshold to
    /// fill from its own neighbors alone needs a second pass, so composing
    /// two manual calls must match the iterative filter capped at 2
    /// iterations, while a single call must NOT already have converged.
    #[test]
    fn voting_binary_hole_filling_is_a_single_pass_not_iterative() {
        #[rustfmt::skip]
        let image = img_u8(&[5, 5], vec![
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
        ]);
        let pass1 = voting_binary_hole_filling(&image, &[1, 1], 1, 1.0, 0.0).unwrap();
        // Center of the 3x3 hole (index 12): its whole 3x3 neighborhood is
        // still background after only one pass, so it cannot have flipped
        // yet -- proof this call didn't loop to convergence internally.
        assert_eq!(pass1.image.scalar_slice::<u8>().unwrap()[12], 0);

        let pass2 = voting_binary_hole_filling(&pass1.image, &[1, 1], 1, 1.0, 0.0).unwrap();
        let iterative =
            voting_binary_iterative_hole_filling(&image, &[1, 1], 1, 2, 1.0, 0.0).unwrap();
        assert_eq!(
            pass2.image.scalar_slice::<u8>().unwrap(),
            iterative.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn voting_binary_hole_filling_rejects_a_floating_point_pixel_type() {
        let image = Image::from_vec(&[3, 1], vec![1.0f32, 0.0, 1.0]).unwrap();
        let err = voting_binary_hole_filling(&image, &[1], 1, 1.0, 0.0).unwrap_err();
        assert_eq!(err, FilterError::RequiresIntegerPixelType(PixelId::Float32));
    }

    #[test]
    fn voting_binary_hole_filling_rejects_a_complex_pixel_type() {
        let image = Image::new(&[3, 1], PixelId::ComplexFloat32);
        let err = voting_binary_hole_filling(&image, &[1], 1, 1.0, 0.0).unwrap_err();
        assert_eq!(
            err,
            FilterError::RequiresIntegerPixelType(PixelId::ComplexFloat32)
        );
    }

    // ---- voting_binary_iterative_hole_filling ----

    #[test]
    fn iterative_hole_filling_converges_before_max_iterations_on_a_small_hole() {
        #[rustfmt::skip]
        let image = img_u8(&[5, 5], vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ]);
        let few = voting_binary_iterative_hole_filling(&image, &[1, 1], 1, 2, 1.0, 0.0).unwrap();
        let many = voting_binary_iterative_hole_filling(&image, &[1, 1], 1, 50, 1.0, 0.0).unwrap();
        assert_eq!(
            few.scalar_slice::<u8>().unwrap(),
            many.scalar_slice::<u8>().unwrap()
        );
        assert_eq!(few.scalar_slice::<u8>().unwrap(), &[1u8; 25]);
    }

    #[test]
    fn iterative_hole_filling_zero_iterations_is_identity() {
        let image = img_u8(&[3, 3], vec![1, 1, 1, 1, 0, 1, 1, 1, 1]);
        let out = voting_binary_iterative_hole_filling(&image, &[1, 1], 1, 0, 1.0, 0.0).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            image.scalar_slice::<u8>().unwrap()
        );
    }

    // ---- binary_median ----

    /// 1-D radius-1 window (3 pixels incl. center), `neighborhoodSize = 3`,
    /// `medianPosition = 3 / 2 = 1`: `count > 1` i.e. `count >= 2` is required
    /// for foreground, so exactly 2-of-3 flips a background center but 1-of-3
    /// does not.
    #[test]
    fn binary_median_tie_rule_is_strict_majority_not_tie() {
        // center=0, neighbors=[1,1]: count=2 > medianPosition(1) -> foreground.
        let two_on = img_u8(&[3], vec![1, 0, 1]);
        let out = binary_median(&two_on, &[1], 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap()[1], 1);

        // center=0, neighbors=[1,0]: count=1, not > medianPosition(1) -> background.
        let one_on = img_u8(&[3], vec![1, 0, 0]);
        let out = binary_median(&one_on, &[1], 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap()[1], 0);
    }

    #[test]
    fn binary_median_removes_a_lone_salt_and_pepper_pixel() {
        let image = img_u8(&[7, 1], vec![5, 5, 5, 99, 5, 5, 5]);
        let out = binary_median(&image, &[1, 0], 99.0, 5.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[5, 5, 5, 5, 5, 5, 5]);
    }

    /// A pixel equal to neither `foreground_value` nor `background_value`
    /// contributes nothing to `count` and is always overwritten -- there is
    /// no pass-through branch (see module docs).
    #[test]
    fn binary_median_neither_value_is_overwritten_to_background() {
        let image = img_u8(&[3], vec![1, 5, 1]);
        let out = binary_median(&image, &[1], 1.0, 0.0).unwrap();
        // center=5 (neither); neighbors=[1,1]: count=2 > medianPosition(1)
        // -> foreground, even though the center itself was never foreground.
        assert_eq!(out.scalar_slice::<u8>().unwrap()[1], 1);
    }

    #[test]
    fn binary_median_zero_flux_neumann_clamps_at_the_border() {
        // index0's radius-1 window clamps to [self, self, index1] = [1,1,0]:
        // its own foreground value counts twice -> count=2 > medianPosition(1).
        let image = img_u8(&[3], vec![1, 0, 0]);
        let out = binary_median(&image, &[1], 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap()[0], 1);
    }

    #[test]
    fn binary_median_radius_zero_is_identity_relabeled_to_fg_bg() {
        // neighborhoodSize=1, medianPosition=0: count>0 iff the pixel itself
        // is foreground, so this just relabels foreground/background exactly.
        let image = img_u8(&[4], vec![1, 0, 1, 1]);
        let out = binary_median(&image, &[0], 1.0, 0.0).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 0, 1, 1]);
    }

    #[test]
    fn binary_median_wrong_radius_length_is_rejected() {
        let image = img_u8(&[3, 3], vec![0; 9]);
        assert!(matches!(
            binary_median(&image, &[1], 1.0, 0.0),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    // ---- binary_thinning ----

    #[test]
    fn binary_thinning_thick_bar_becomes_a_centered_one_pixel_line() {
        // A 5x3 bar with a 1-pixel background border (a bar filling the
        // whole image would touch the image edge on every side, and
        // `ZeroFluxNeumannBoundaryCondition` would then replicate those
        // border pixels as foreground too, making every edge pixel look
        // interior — `numberOfOnNeighbors == 8` fails testA's `< 7` bound
        // and nothing is ever deletable. Expected output hand-traced
        // against `itkBinaryThinningImageFilter.hxx`'s testA/testB/testC
        // rules: round 1 deletes the four corners at step 2 (testC:
        // `p2 == 0 && p8 == 0`), which then exposes the rest of the top
        // and bottom rows and the two side columns to testB's
        // single-transition rule across the remaining steps, eroding
        // everything but the row-2 pixels x=2..4; round 2 finds nothing
        // left to delete.
        #[rustfmt::skip]
        let image = img_u8(&[7, 5], vec![
            0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]);
        let out = binary_thinning(&image).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]);
    }

    #[test]
    fn binary_thinning_cross_keeps_connectivity() {
        #[rustfmt::skip]
        let image = img_u8(&[5, 5], vec![
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ]);
        let out = binary_thinning(&image).unwrap();
        // A 1-pixel-wide cross is already thin; thinning is a fixed point.
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            image.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn binary_thinning_nonzero_input_is_rescaled_to_one() {
        let image = img_u8(&[3, 1], vec![7, 7, 7]);
        let out = binary_thinning(&image).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 1, 1]);
    }

    #[test]
    fn binary_thinning_rejects_non_2d_input() {
        let image = img_u8(&[2, 2, 2], vec![1; 8]);
        assert_eq!(
            binary_thinning(&image).unwrap_err(),
            FilterError::UnsupportedThinningDimension(3)
        );
    }
}

/// Thread-count parity pins for the three window passes in this module.
///
/// All three are **counting** stencils — an equality count over one voxel's own
/// window — so, like the grayscale morphology pins, their non-vacuity guard is
/// slot-dependence, not fold order: a `usize` count is exact and order-insensitive,
/// and asserting that reordering it moves the answer would be asserting a
/// falsehood.
///
/// The one quantity here that genuinely crossed voxels is
/// [`voting_binary_hole_filling`]'s `changed` counter, which the serial pass
/// incremented inside the loop. It is pinned too: the parallel pass returns a
/// per-voxel flag and the count is tallied afterwards on one thread, so the pin
/// below asserts both the image *and* the count.
///
/// `-0.0` does not arise anywhere in this module: every value written is one of
/// the two caller-supplied labels, and the pixel type is required to be integer
/// ([`crate::logic::require_integer_pixel_type`]), so there is no signed zero to
/// preserve.
#[cfg(test)]
mod stencil_thread_parity {
    use super::*;
    use crate::stencil_test_support::{THREADS, binary_volume};
    use sitk_core::parallel;

    const FG: f64 = 1.0;
    const BG: f64 = 0.0;

    // ---- the serial references: the exact loops that were deleted -----------

    fn voting_binary_serial(img: &Image, radius: &[usize], birth: u32, survival: u32) -> Vec<u8> {
        let foreground = u8::from_f64(FG);
        let background = u8::from_f64(BG);
        let iter =
            NeighborhoodIterator::<u8, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        iter.map(|(_, nb)| {
            let inpixel = nb.center_value();
            let count = nb.values().iter().filter(|&&v| v == foreground).count() as u32;
            if inpixel == background && count >= birth {
                foreground
            } else if inpixel == foreground && count < survival {
                background
            } else {
                inpixel
            }
        })
        .collect()
    }

    /// The hole-filling pass *and* its `changed` counter, as the serial loop
    /// produced them: one shared `changed += 1` inside the walk.
    fn hole_filling_serial(img: &Image, radius: &[usize], birth: u32) -> (Vec<u8>, u32) {
        let foreground = u8::from_f64(FG);
        let background = u8::from_f64(BG);
        let iter =
            NeighborhoodIterator::<u8, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        let mut out = Vec::with_capacity(img.number_of_pixels());
        let mut changed = 0u32;
        for (_, nb) in iter {
            let inpixel = nb.center_value();
            let value = if inpixel == background {
                let count = nb.values().iter().filter(|&&v| v == foreground).count() as u32;
                if count >= birth {
                    changed += 1;
                    foreground
                } else {
                    background
                }
            } else {
                foreground
            };
            out.push(value);
        }
        (out, changed)
    }

    fn binary_median_serial(img: &Image, radius: &[usize]) -> Vec<u8> {
        let foreground = u8::from_f64(FG);
        let background = u8::from_f64(BG);
        let iter =
            NeighborhoodIterator::<u8, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        let median_position = iter.len() / 2;
        iter.map(|(_, nb)| {
            let count = nb.values().iter().filter(|&&v| v == foreground).count();
            if count > median_position {
                foreground
            } else {
                background
            }
        })
        .collect()
    }

    fn pixels(img: &Image) -> Vec<u8> {
        img.scalar_slice::<u8>().unwrap().to_vec()
    }

    // ---- non-vacuity --------------------------------------------------------

    /// These pins are worth nothing unless each output voxel actually depends on
    /// *which* window slots it counts. Shown two ways: the median moves a large
    /// share of the voxels (so the volume is not trivially all-foreground or
    /// all-background), and a `[1,1,1]` window gives a different answer than a
    /// `[1,0,0]` one — the two differ only in which slots are in the count, so a
    /// mis-addressed window would be caught.
    #[test]
    fn the_output_depends_on_which_window_slots_are_counted() {
        let img = binary_volume();
        let input = pixels(&img);

        let full = binary_median(&img, &[1, 1, 1], FG, BG).unwrap();
        let thin = binary_median(&img, &[1, 0, 0], FG, BG).unwrap();

        let moved = pixels(&full)
            .iter()
            .zip(&input)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            moved > input.len() / 100,
            "the binary median moved only {moved}/{} voxels — the volume is too uniform to \
             pin anything",
            input.len()
        );

        let differ = pixels(&full)
            .iter()
            .zip(pixels(&thin))
            .filter(|(a, b)| **a != *b)
            .count();
        assert!(
            differ > input.len() / 100,
            "a [1,1,1] window and a [1,0,0] window produced the same median on all but \
             {differ} voxels — the output barely depends on which slots are counted, so \
             these pins could not catch a mis-addressed window"
        );
    }

    /// The hole-filling counter is only pinned meaningfully if the pass actually
    /// flips pixels — a count of zero would match a broken parallel tally by
    /// accident.
    #[test]
    fn the_hole_filling_pass_actually_changes_pixels() {
        let img = binary_volume();
        let (_, changed) = hole_filling_serial(&img, &[2, 2, 2], 14);
        assert!(
            changed > 100,
            "the serial hole-filling pass flipped only {changed} pixels — too few for the \
             counter pin to mean anything"
        );
    }

    // ---- the pins -----------------------------------------------------------

    #[test]
    fn voting_binary_is_bit_identical_at_every_thread_count() {
        let img = binary_volume();
        for (radius, birth, survival) in [(vec![1usize, 1, 1], 12u32, 10u32), (vec![2, 1, 1], 8, 6)]
        {
            let expected = voting_binary_serial(&img, &radius, birth, survival);
            for threads in THREADS {
                let got = parallel::with_threads(threads, || {
                    voting_binary(&img, &radius, birth, survival, FG, BG)
                })
                .unwrap();
                assert_eq!(
                    pixels(&got),
                    expected,
                    "voting_binary({radius:?}, birth={birth}, survival={survival}) moved with \
                     {threads} threads"
                );
            }
        }
    }

    #[test]
    fn voting_binary_hole_filling_image_and_count_are_identical_at_every_thread_count() {
        let img = binary_volume();
        for (radius, majority) in [(vec![2usize, 2, 2], 1u32), (vec![1, 1, 1], 2)] {
            let neighborhood: usize = radius.iter().map(|&r| 2 * r + 1).product();
            let birth = ((neighborhood - 1) / 2) as u32 + majority;
            let (expected_image, expected_changed) = hole_filling_serial(&img, &radius, birth);

            for threads in THREADS {
                let got = parallel::with_threads(threads, || {
                    voting_binary_hole_filling(&img, &radius, majority, FG, BG)
                })
                .unwrap();
                assert_eq!(
                    pixels(&got.image),
                    expected_image,
                    "voting_binary_hole_filling({radius:?}) image moved with {threads} threads"
                );
                assert_eq!(
                    got.number_of_pixels_changed, expected_changed,
                    "voting_binary_hole_filling({radius:?}) changed-count moved with {threads} \
                     threads: {} vs serial {expected_changed}",
                    got.number_of_pixels_changed
                );
            }
        }
    }

    #[test]
    fn binary_median_is_bit_identical_at_every_thread_count() {
        let img = binary_volume();
        for radius in [vec![1usize, 1, 1], vec![2, 1, 3]] {
            let expected = binary_median_serial(&img, &radius);
            for threads in THREADS {
                let got = parallel::with_threads(threads, || binary_median(&img, &radius, FG, BG))
                    .unwrap();
                assert_eq!(
                    pixels(&got),
                    expected,
                    "binary_median({radius:?}) moved with {threads} threads"
                );
            }
        }
    }
}
