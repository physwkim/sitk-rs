//! Seeded region-growing segmentation.
//!
//! Ports of:
//!
//! - `itk::ConnectedThresholdImageFilter`
//!   (`itkConnectedThresholdImageFilter.h` / `.hxx`).
//! - `itk::NeighborhoodConnectedImageFilter`
//!   (`itkNeighborhoodConnectedImageFilter.h` / `.hxx`, backed by
//!   `itkNeighborhoodBinaryThresholdImageFunction.h` / `.hxx`).
//! - `itk::ConfidenceConnectedImageFilter`
//!   (`itkConfidenceConnectedImageFilter.h` / `.hxx`).
//! - `itk::IsolatedConnectedImageFilter`
//!   (`itkIsolatedConnectedImageFilter.h` / `.hxx`).
//!
//! All four grow a region outward from user-supplied seed *indices* (not
//! physical points — SimpleITK's yaml for every filter here declares its
//! seed members as index vectors) via a breadth-first flood fill, matching
//! ITK's `FloodFilledFunctionConditionalConstIterator` /
//! `ShapedFloodFilledFunctionConditionalConstIterator`
//! (`itkFloodFilledFunctionConditionalConstIterator.h` / `.hxx`,
//! `itkShapedFloodFilledFunctionConditionalConstIterator.h` / `.hxx`): an
//! explicit work queue, never recursion, so a large volume cannot blow the
//! stack. [`flood_fill`] is the shared core every filter below builds on.
//!
//! ## Seed admission
//!
//! Both flood-iterator base classes gate every seed the same way in
//! `GoToBegin()`: `region.IsInside(seed) && IsPixelIncluded(seed)`. A seed
//! outside the image, or whose own value fails the filter's inclusion test,
//! is silently dropped — it never starts a flood and is never an error. An
//! all-dropped seed list (including an empty one) yields the pre-zeroed
//! output unchanged. [`flood_fill`] reproduces this exactly.
//!
//! ## Connectivity
//!
//! `ConnectedThresholdImageFilter` is the only one of the four with a
//! `Connectivity` option (`ConnectivityEnum::FaceConnectivity`, the default,
//! vs `FullConnectivity`), selecting between the two flood-iterator classes
//! above. Face connectivity offsets are the `2 * dim` axis-aligned unit
//! steps; full connectivity is every nonzero offset in `{-1,0,1}^dim`
//! (`3^dim - 1` of them) — `itkConnectedComponentAlgorithm.h`'s
//! `setConnectivity`. The other three filters use the unshaped iterator
//! unconditionally, i.e. always face connectivity.
//!
//! ## `neighborhood_connected`
//!
//! `NeighborhoodBinaryThresholdImageFunction::EvaluateAtIndex` requires
//! *every* pixel in a `radius`-sized box around a candidate — not just the
//! candidate itself — to satisfy `[lower, upper]`, walked with a
//! zero-flux-Neumann boundary condition at the image edge (explicit in the
//! ITK source comment).
//!
//! ## `confidence_connected`
//!
//! Iteratively re-estimates `[mean - multiplier*sigma, mean +
//! multiplier*sigma]` from the segmented region and re-floods, for
//! `number_of_iterations` rounds (`number_of_iterations = 0` performs only
//! the initial segmentation). The initial mean/variance instead come from a
//! `initial_neighborhood_radius`-sized box around every in-bounds seed
//! (`itkConfidenceConnectedImageFilter.hxx`'s `ShapedImageNeighborhoodRange`
//! pass, likewise zero-flux-Neumann at the edge); radius 0 degenerates to
//! exactly the seed pixel, unifying the .hxx's two branches
//! (`if m_InitialNeighborhoodRadius > 0`) into one. Every iteration's
//! `[lower, upper]` is widened, if needed, to cover the seeds' own
//! intensity range and clamped to the pixel type's valid range — guaranteeing
//! the flood is never empty. If a round's variance is (almost) zero the loop
//! stops early rather than re-flooding on a degenerate interval
//! (`itk::Math::AlmostEquals`, `itkMath.h`). This port recomputes each
//! round's statistics by summing the already-known segmented mask directly,
//! rather than retracing a second flood over the output image as
//! `SecondIteratorType` does in the .hxx — a flood fill's final visited set
//! doesn't depend on the order pixels were visited in, so both give the same
//! sample.
//!
//! ## `isolated_connected`
//!
//! Bisects for the tightest single threshold bound (upper or lower,
//! depending on `find_upper_threshold`) that still separates `seeds1`'s
//! flood from `seeds2`. This port always runs each bisection round's flood
//! to completion, whereas `itkIsolatedConnectedImageFilter.hxx` exits the
//! instant it reaches `Seeds2.front()`: reaching `Seeds2.front()` already
//! forces the "are any of `seeds2` included" sum to be nonzero regardless of
//! what the rest of the flood would visit, and when `Seeds2.front()` is
//! never reached the reference flood runs to completion anyway — so both
//! give the same separating-threshold decision every round.
//!
//! The bisection arithmetic runs in `NumericTraits<T>::AccumulateType`
//! (`itkNumericTraits.h`) — a wider integer for integer pixel types (so
//! `(upper + lower) / 2` truncates like the reference's integer division),
//! `f64` for float pixel types. This port always widens integers to `i128`
//! rather than replicating each type's exact narrower `AccumulateType`
//! (e.g. ITK's own `uint32_t` `AccumulateType` is `unsigned int`, not
//! widened, an ITK quirk that could theoretically overflow near
//! `UINT_MAX`); every operand here originates from a caller-supplied `f64`
//! narrowed through [`sitk_core::Scalar::from_f64`], so magnitudes never
//! approach the extremes where the two would diverge.
//!
//! `replace_value == 0` degenerates the seeds2-inclusion test to always
//! "not included" (every possible sum, included or not, is 0), matching
//! `itkIsolatedConnectedImageFilter.hxx`'s own `seedIntensitySum` arithmetic
//! rather than special-casing it away.
//!
//! `itk::Math::AlmostEquals`/`NotAlmostEquals` (`itkMath.h`) use a
//! ULP-and-epsilon float comparison; this port uses a small fixed absolute
//! tolerance ([`VARIANCE_ALMOST_ZERO`], and `1e-9` for the sum comparisons
//! in `isolated_connected`) instead of replicating ULP counting bit for bit
//! — both exist only to absorb floating-point rounding noise around exact
//! zero, and the sums/variances here never approach magnitudes where the
//! two tolerances would disagree.

use crate::error::{FilterError, Result};
use crate::morphology::bounds_for;
use sitk_core::{
    Image, NeighborhoodIterator, PixelId, Scalar, ZeroFluxNeumannBoundaryCondition, dispatch_scalar,
};
use std::collections::VecDeque;

// ---- shared seed / flood-fill helpers --------------------------------

/// `RegionType::IsInside(seed)`: maps a signed ITK-style seed index onto the
/// image, or `None` if any axis falls outside `[0, size[d])`.
fn in_bounds_index(seed: &[i64], size: &[usize]) -> Option<Vec<usize>> {
    let mut idx = Vec::with_capacity(size.len());
    for (&s, &sz) in seed.iter().zip(size) {
        if s < 0 || s as usize >= sz {
            return None;
        }
        idx.push(s as usize);
    }
    Some(idx)
}

/// Errors if any seed's dimensionality doesn't match the image's.
fn validate_seed_dims(seeds: &[Vec<i64>], dim: usize) -> Result<()> {
    for seed in seeds {
        if seed.len() != dim {
            return Err(FilterError::DimensionLength {
                expected: dim,
                got: seed.len(),
            });
        }
    }
    Ok(())
}

/// Per-axis unit-offset connectivity (`itkConnectedComponentAlgorithm.h`'s
/// `setConnectivity`): `fully_connected = false` is face connectivity (`±1`
/// along exactly one axis, `2 * dim` offsets); `true` is full connectivity
/// (every nonzero offset in `{-1,0,1}^dim`, `3^dim - 1` offsets).
fn neighbor_offsets(dim: usize, fully_connected: bool) -> Vec<Vec<i64>> {
    let total = 3usize.pow(dim as u32);
    let mut offsets = Vec::new();
    for code in 0..total {
        let mut rem = code;
        let mut offset = vec![0i64; dim];
        let mut nonzero_axes = 0usize;
        for d in offset.iter_mut() {
            let digit = rem % 3;
            rem /= 3;
            *d = digit as i64 - 1; // 0,1,2 -> -1,0,1
            if *d != 0 {
                nonzero_axes += 1;
            }
        }
        if nonzero_axes == 0 || (!fully_connected && nonzero_axes > 1) {
            continue;
        }
        offsets.push(offset);
    }
    offsets
}

/// The shared flood-fill core behind every filter in this module —
/// `FloodFilledFunctionConditionalConstIterator` /
/// `ShapedFloodFilledFunctionConditionalConstIterator`'s BFS via an explicit
/// `std::queue`, ported here as an explicit [`VecDeque`] (never recursion,
/// so a large volume cannot blow the stack).
///
/// `seeds` outside the image are dropped; an in-bounds seed is queued only
/// if `include` accepts it — together this is exactly `GoToBegin`'s combined
/// `IsInside(seed) && IsPixelIncluded(seed)` gate, so a seed whose own value
/// fails `include` never starts a flood. Every candidate pixel is tested by
/// `include` at most once, mirroring the reference's tri-state temporary
/// image (untested / excluded / included).
fn flood_fill(
    img: &Image,
    seeds: &[Vec<i64>],
    offsets: &[Vec<i64>],
    mut include: impl FnMut(&[usize]) -> bool,
) -> Vec<bool> {
    let size = img.size();
    let dim = size.len();
    let total = img.number_of_pixels();
    let mut visited = vec![false; total];
    if total == 0 {
        return visited;
    }

    // 0 = untested, 1 = tested and excluded, 2 = included.
    let mut state = vec![0u8; total];
    let mut queue: VecDeque<Vec<usize>> = VecDeque::new();

    for seed in seeds {
        let Some(idx) = in_bounds_index(seed, size) else {
            continue;
        };
        let f = img.linear_index(&idx);
        if state[f] != 0 {
            continue;
        }
        if include(&idx) {
            state[f] = 2;
            visited[f] = true;
            queue.push_back(idx);
        } else {
            state[f] = 1;
        }
    }

    while let Some(cur) = queue.pop_front() {
        for offset in offsets {
            let mut nb = Vec::with_capacity(dim);
            let mut inside = true;
            for d in 0..dim {
                let v = cur[d] as i64 + offset[d];
                if v < 0 || v as usize >= size[d] {
                    inside = false;
                    break;
                }
                nb.push(v as usize);
            }
            if !inside {
                continue;
            }
            let f = img.linear_index(&nb);
            if state[f] != 0 {
                continue;
            }
            if include(&nb) {
                state[f] = 2;
                visited[f] = true;
                queue.push_back(nb);
            } else {
                state[f] = 1;
            }
        }
    }

    visited
}

fn mask_to_image(size: &[usize], geom: &Image, mask: &[bool], replace_value: u8) -> Result<Image> {
    let out: Vec<u8> = mask
        .iter()
        .map(|&m| if m { replace_value } else { 0 })
        .collect();
    let mut result = Image::from_vec(size, out)?;
    result.copy_geometry_from(geom);
    Ok(result)
}

// ---- connected_threshold -----------------------------------------------

fn connected_threshold_typed<T: Scalar>(
    img: &Image,
    seeds: &[Vec<i64>],
    lower: f64,
    upper: f64,
    replace_value: u8,
    fully_connected: bool,
) -> Result<Image> {
    let dim = img.dimension();
    validate_seed_dims(seeds, dim)?;
    let pixels = img.scalar_slice::<T>()?;
    let lower_t = T::from_f64(lower);
    let upper_t = T::from_f64(upper);
    let offsets = neighbor_offsets(dim, fully_connected);

    let mask = flood_fill(img, seeds, &offsets, |idx| {
        let v = pixels[img.linear_index(idx)];
        v >= lower_t && v <= upper_t
    });
    mask_to_image(img.size(), img, &mask, replace_value)
}

/// `ConnectedThresholdImageFilter`: flood-fills outward from `seeds`,
/// admitting a pixel iff `lower <= v <= upper` (inclusive,
/// `BinaryThresholdImageFunction`). `fully_connected = false` is face
/// connectivity (the filter's default `ConnectivityEnum::FaceConnectivity`);
/// `true` is full connectivity. Output pixel type is `UInt8`, matching
/// SimpleITK's `ConnectedThresholdImageFilter.yaml`
/// (`output_pixel_type: uint8_t`).
pub fn connected_threshold(
    img: &Image,
    seeds: &[Vec<i64>],
    lower: f64,
    upper: f64,
    replace_value: u8,
    fully_connected: bool,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        connected_threshold_typed,
        img,
        seeds,
        lower,
        upper,
        replace_value,
        fully_connected
    )
}

// ---- neighborhood_connected ----------------------------------------------

fn neighborhood_connected_typed<T: Scalar>(
    img: &Image,
    seeds: &[Vec<i64>],
    lower: f64,
    upper: f64,
    radius: &[usize],
    replace_value: u8,
) -> Result<Image> {
    let dim = img.dimension();
    validate_seed_dims(seeds, dim)?;
    let lower_t = T::from_f64(lower);
    let upper_t = T::from_f64(upper);
    let nb_iter = NeighborhoodIterator::<T, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)?;
    // `NeighborhoodConnectedImageFilter` has no `Connectivity` option: it
    // always uses the unshaped flood iterator, i.e. face connectivity.
    let offsets = neighbor_offsets(dim, false);

    let mask = flood_fill(img, seeds, &offsets, |idx| {
        nb_iter
            .neighborhood_at(idx)
            .values()
            .iter()
            .all(|&v| v >= lower_t && v <= upper_t)
    });
    mask_to_image(img.size(), img, &mask, replace_value)
}

/// `NeighborhoodConnectedImageFilter`: like [`connected_threshold`], but a
/// candidate pixel is admitted only if *every* pixel in the `radius`-sized
/// box around it satisfies `[lower, upper]`
/// (`NeighborhoodBinaryThresholdImageFunction::EvaluateAtIndex`), walked
/// with a zero-flux-Neumann boundary condition at the image edge. Always
/// face-connected (this filter has no `Connectivity` option). Output pixel
/// type is `UInt8`, matching SimpleITK's
/// `NeighborhoodConnectedImageFilter.yaml`.
pub fn neighborhood_connected(
    img: &Image,
    seeds: &[Vec<i64>],
    lower: f64,
    upper: f64,
    radius: &[usize],
    replace_value: u8,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        neighborhood_connected_typed,
        img,
        seeds,
        lower,
        upper,
        radius,
        replace_value
    )
}

// ---- confidence_connected -------------------------------------------------

/// `itk::Math::AlmostEquals(variance, 0.0)`'s tolerance
/// (`itkMath.h`'s `FloatAlmostEqual`, default `maxUlps = 4` plus a small
/// absolute epsilon) is, for `f64`, effectively "exactly zero, give or take
/// the rounding noise a variance computation itself accumulates." This fixed
/// absolute tolerance is loose enough to absorb that noise without needing
/// to replicate ULP-distance comparison.
const VARIANCE_ALMOST_ZERO: f64 = 1e-12;

#[allow(clippy::too_many_arguments)]
fn confidence_connected_typed<T: Scalar>(
    img: &Image,
    seeds: &[Vec<i64>],
    number_of_iterations: u32,
    multiplier: f64,
    initial_neighborhood_radius: usize,
    replace_value: u8,
) -> Result<Image> {
    let dim = img.dimension();
    validate_seed_dims(seeds, dim)?;
    let size = img.size();
    let total = img.number_of_pixels();
    let pixels = img.scalar_slice::<T>()?;
    let (max_v, min_v) = bounds_for(T::PIXEL_ID);

    // Everything below that reads a seed's *value* — the initial stats, the
    // seed-intensity clamp — only ever considers in-bounds seeds
    // (`region.IsInside(*si)` in the .hxx); an out-of-bounds seed simply
    // never contributes.
    let in_bounds_seeds: Vec<Vec<usize>> = seeds
        .iter()
        .filter_map(|s| in_bounds_index(s, size))
        .collect();
    if in_bounds_seeds.is_empty() {
        return mask_to_image(size, img, &vec![false; total], replace_value);
    }

    // Initial mean/variance from the `initial_neighborhood_radius`-box
    // around every in-bounds seed; radius 0 degenerates to just the seed
    // pixel.
    let radius_vec = vec![initial_neighborhood_radius; dim];
    let nb_iter =
        NeighborhoodIterator::<T, _>::new(img, &radius_vec, ZeroFluxNeumannBoundaryCondition)?;
    let window_len = nb_iter.len() as f64;
    let mut mean = 0f64;
    let mut sum_sq = 0f64;
    for seed in &in_bounds_seeds {
        let mut neighborhood_sum = 0f64;
        for &v in nb_iter.neighborhood_at(seed).values() {
            let f = v.as_f64();
            neighborhood_sum += f;
            sum_sq += f * f;
        }
        mean += neighborhood_sum / window_len;
    }
    let num = in_bounds_seeds.len() as f64;
    mean /= num;
    let total_num = num * window_len;
    let mut variance = if total_num > 1.0 {
        (sum_sq - mean * mean * total_num) / (total_num - 1.0)
    } else {
        0.0
    };

    // Fixed for the whole run: the range spanned by the seeds' own
    // intensities, and the pixel type's valid range. Every iteration's
    // `[lower, upper]` is widened to cover both, guaranteeing the flood
    // never excludes a seed's own pixel.
    let mut lowest_seed = f64::INFINITY;
    let mut highest_seed = f64::NEG_INFINITY;
    for seed in &in_bounds_seeds {
        let v = pixels[img.linear_index(seed)].as_f64();
        lowest_seed = lowest_seed.min(v);
        highest_seed = highest_seed.max(v);
    }
    let clamp_bounds = |mut lower: f64, mut upper: f64| -> (f64, f64) {
        if lower > lowest_seed {
            lower = lowest_seed;
        }
        if upper < highest_seed {
            upper = highest_seed;
        }
        (lower.max(min_v), upper.min(max_v))
    };

    let (lo0, hi0) = clamp_bounds(
        mean - multiplier * variance.sqrt(),
        mean + multiplier * variance.sqrt(),
    );
    let mut lower_t = T::from_f64(lo0);
    let mut upper_t = T::from_f64(hi0);

    // `FloodFilledImageFunctionConditionalIterator` (unshaped): always face
    // connectivity, like `neighborhood_connected` — no `Connectivity` option
    // on this filter either.
    let offsets = neighbor_offsets(dim, false);
    let mut mask = flood_fill(img, seeds, &offsets, |idx| {
        let v = pixels[img.linear_index(idx)];
        v >= lower_t && v <= upper_t
    });

    for _ in 0..number_of_iterations {
        // Recompute mean/variance directly from the already-known segmented
        // mask, rather than retracing a second flood over the output image
        // as `SecondIteratorType` does: a flood fill's final visited set
        // doesn't depend on traversal order, so both give the same sample.
        let mut n2 = 0f64;
        let mut s2 = 0f64;
        let mut sq2 = 0f64;
        for (i, &included) in mask.iter().enumerate() {
            if included {
                let f = pixels[i].as_f64();
                n2 += 1.0;
                s2 += f;
                sq2 += f * f;
            }
        }
        mean = s2 / n2;
        variance = if n2 > 1.0 {
            (sq2 - s2 * s2 / n2) / (n2 - 1.0)
        } else {
            0.0
        };
        if variance.abs() < VARIANCE_ALMOST_ZERO {
            break;
        }
        let (lo, hi) = clamp_bounds(
            mean - multiplier * variance.sqrt(),
            mean + multiplier * variance.sqrt(),
        );
        lower_t = T::from_f64(lo);
        upper_t = T::from_f64(hi);
        mask = flood_fill(img, seeds, &offsets, |idx| {
            let v = pixels[img.linear_index(idx)];
            v >= lower_t && v <= upper_t
        });
    }

    mask_to_image(size, img, &mask, replace_value)
}

/// `ConfidenceConnectedImageFilter`: iteratively estimates `[mean -
/// multiplier*sigma, mean + multiplier*sigma]` from the currently segmented
/// region and re-floods, for `number_of_iterations` rounds
/// (`number_of_iterations = 0` performs only the initial segmentation). The
/// initial mean/variance come from a `initial_neighborhood_radius`-sized box
/// around every in-bounds seed. Output pixel type is `UInt8`, matching
/// SimpleITK's `ConfidenceConnectedImageFilter.yaml`.
#[allow(clippy::too_many_arguments)]
pub fn confidence_connected(
    img: &Image,
    seeds: &[Vec<i64>],
    number_of_iterations: u32,
    multiplier: f64,
    initial_neighborhood_radius: usize,
    replace_value: u8,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        confidence_connected_typed,
        img,
        seeds,
        number_of_iterations,
        multiplier,
        initial_neighborhood_radius,
        replace_value
    )
}

// ---- isolated_connected -----------------------------------------------

/// `NumericTraits<T>::AccumulateType`-flavored bisection arithmetic: an
/// integer accumulator (truncating division, like the reference's integer
/// `AccumulateType`) for integer pixel types, a float accumulator for float
/// pixel types. See the module doc for why `i128` stands in for every
/// integer type's own (narrower) `AccumulateType`.
#[derive(Clone, Copy)]
enum Acc {
    Int(i128),
    Float(f64),
}

impl Acc {
    fn from_scalar<T: Scalar>(v: T, is_float: bool) -> Acc {
        if is_float {
            Acc::Float(v.as_f64())
        } else {
            Acc::Int(v.as_f64() as i128)
        }
    }

    fn add(self, other: Acc) -> Acc {
        match (self, other) {
            (Acc::Int(a), Acc::Int(b)) => Acc::Int(a + b),
            (Acc::Float(a), Acc::Float(b)) => Acc::Float(a + b),
            _ => unreachable!("Acc variants must match within one bisection"),
        }
    }

    fn sub(self, other: Acc) -> Acc {
        match (self, other) {
            (Acc::Int(a), Acc::Int(b)) => Acc::Int(a - b),
            (Acc::Float(a), Acc::Float(b)) => Acc::Float(a - b),
            _ => unreachable!("Acc variants must match within one bisection"),
        }
    }

    fn midpoint(self, other: Acc) -> Acc {
        match (self, other) {
            (Acc::Int(a), Acc::Int(b)) => Acc::Int((a + b) / 2),
            (Acc::Float(a), Acc::Float(b)) => Acc::Float((a + b) / 2.0),
            _ => unreachable!("Acc variants must match within one bisection"),
        }
    }

    fn lt(self, other: Acc) -> bool {
        match (self, other) {
            (Acc::Int(a), Acc::Int(b)) => a < b,
            (Acc::Float(a), Acc::Float(b)) => a < b,
            _ => unreachable!("Acc variants must match within one bisection"),
        }
    }

    fn to_f64(self) -> f64 {
        match self {
            Acc::Int(v) => v as f64,
            Acc::Float(v) => v,
        }
    }
}

/// Result of [`isolated_connected`], mirroring
/// `IsolatedConnectedImageFilter`'s `GetIsolatedValue()` /
/// `GetThresholdingFailed()` outputs alongside the segmented image.
#[derive(Clone, Debug, PartialEq)]
pub struct IsolatedConnectedResult {
    pub image: Image,
    /// The separating threshold the bisection converged to.
    pub isolated_value: f64,
    /// `true` if the search could not find a threshold that includes every
    /// `seeds1` member while excluding every `seeds2` member.
    pub thresholding_failed: bool,
}

#[allow(clippy::too_many_arguments)]
fn isolated_connected_typed<T: Scalar>(
    img: &Image,
    seeds1: &[Vec<i64>],
    seeds2: &[Vec<i64>],
    lower: f64,
    upper: f64,
    isolated_value_tolerance: f64,
    find_upper_threshold: bool,
    replace_value: u8,
) -> Result<IsolatedConnectedResult> {
    let dim = img.dimension();
    validate_seed_dims(seeds1, dim)?;
    validate_seed_dims(seeds2, dim)?;
    let size = img.size();
    let pixels = img.scalar_slice::<T>()?;
    let is_float = matches!(T::PIXEL_ID, PixelId::Float32 | PixelId::Float64);

    let lower_t = T::from_f64(lower);
    let upper_t = T::from_f64(upper);
    let tol_acc = Acc::from_scalar(T::from_f64(isolated_value_tolerance), is_float);

    // `FloodFilledImageFunctionConditionalIterator` (unshaped): always face
    // connectivity.
    let offsets = neighbor_offsets(dim, false);

    // Floods `seeds1` under `[lo, hi]` and reports whether the resulting
    // mask includes `seeds2` — via the same summed-intensity comparison
    // `itkIsolatedConnectedImageFilter.hxx` makes (`seedIntensitySum`), not
    // a simple "is any seeds2 member in the mask" check, so a
    // `replace_value == 0` call degrades exactly the way the reference's own
    // arithmetic does (every possible sum, included or not, is zero).
    // Runs each flood to completion rather than exiting the instant
    // `Seeds2.front()` is reached, as the reference does — see the module
    // doc for why that yields the same separating-threshold decision.
    let seeds2_included = |lo: T, hi: T| -> bool {
        let mask = flood_fill(img, seeds1, &offsets, |idx| {
            let v = pixels[img.linear_index(idx)];
            v >= lo && v <= hi
        });
        let sum: f64 = seeds2
            .iter()
            .map(|s| {
                in_bounds_index(s, size)
                    .map(|idx| {
                        if mask[img.linear_index(&idx)] {
                            replace_value as f64
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0)
            })
            .sum();
        sum != 0.0
    };

    let isolated_value_t = if find_upper_threshold {
        let mut lo = Acc::from_scalar(lower_t, is_float);
        let hi_bound = Acc::from_scalar(upper_t, is_float);
        let mut hi = hi_bound;
        let mut guess = hi;
        while lo.add(tol_acc).lt(guess) {
            let guess_t = T::from_f64(guess.to_f64());
            if seeds2_included(lower_t, guess_t) {
                hi = guess;
            } else {
                lo = guess;
            }
            guess = lo.midpoint(hi);
        }
        T::from_f64(lo.to_f64())
    } else {
        let lo_bound = Acc::from_scalar(lower_t, is_float);
        let mut lo = lo_bound;
        let mut hi = Acc::from_scalar(upper_t, is_float);
        let mut guess = lo;
        while guess.lt(hi.sub(tol_acc)) {
            let guess_t = T::from_f64(guess.to_f64());
            if seeds2_included(guess_t, upper_t) {
                lo = guess;
            } else {
                hi = guess;
            }
            guess = lo.midpoint(hi);
        }
        T::from_f64(hi.to_f64())
    };

    let (final_lo, final_hi) = if find_upper_threshold {
        (lower_t, isolated_value_t)
    } else {
        (isolated_value_t, upper_t)
    };
    let mask = flood_fill(img, seeds1, &offsets, |idx| {
        let v = pixels[img.linear_index(idx)];
        v >= final_lo && v <= final_hi
    });

    let seed_sum = |seeds: &[Vec<i64>]| -> f64 {
        seeds
            .iter()
            .map(|s| {
                in_bounds_index(s, size)
                    .map(|idx| {
                        if mask[img.linear_index(&idx)] {
                            replace_value as f64
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0)
            })
            .sum()
    };
    let seed1_sum = seed_sum(seeds1);
    let seed2_sum = seed_sum(seeds2);
    let target = replace_value as f64 * seeds1.len() as f64;
    let thresholding_failed = (seed1_sum - target).abs() > 1e-9 || seed2_sum != 0.0;

    let image = mask_to_image(size, img, &mask, replace_value)?;
    Ok(IsolatedConnectedResult {
        image,
        isolated_value: isolated_value_t.as_f64(),
        thresholding_failed,
    })
}

/// `IsolatedConnectedImageFilter`: bisects for the tightest single threshold
/// bound — the upper bound if `find_upper_threshold`, else the lower — that,
/// combined with the fixed other bound (`lower` or `upper`), still separates
/// a `seeds1` flood from `seeds2`. Errors if either seed list is empty
/// (`itkIsolatedConnectedImageFilter.hxx` raises an exception in that case,
/// rather than silently producing an empty output as the other three
/// filters in this module do). Output pixel type is `UInt8`, matching
/// SimpleITK's `IsolatedConnectedImageFilter.yaml`.
#[allow(clippy::too_many_arguments)]
pub fn isolated_connected(
    img: &Image,
    seeds1: &[Vec<i64>],
    seeds2: &[Vec<i64>],
    lower: f64,
    upper: f64,
    isolated_value_tolerance: f64,
    find_upper_threshold: bool,
    replace_value: u8,
) -> Result<IsolatedConnectedResult> {
    if seeds1.is_empty() {
        return Err(FilterError::EmptySeeds { which: "seeds1" });
    }
    if seeds2.is_empty() {
        return Err(FilterError::EmptySeeds { which: "seeds2" });
    }
    dispatch_scalar!(
        img.pixel_id(),
        isolated_connected_typed,
        img,
        seeds1,
        seeds2,
        lower,
        upper,
        isolated_value_tolerance,
        find_upper_threshold,
        replace_value
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_u8_2d(w: usize, h: usize, data: Vec<u8>) -> Image {
        Image::from_vec(&[w, h], data).unwrap()
    }

    // ---- connected_threshold -------------------------------------------

    #[test]
    fn floods_uniform_region_and_stops_at_intensity_step() {
        // 5x1: a plateau of 10s from x=1..=3, background 0 elsewhere.
        let img = Image::from_vec(&[5, 1], vec![0u8, 10, 10, 10, 0]).unwrap();
        let out = connected_threshold(&img, &[vec![2, 0]], 5.0, 15.0, 1, false).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 1, 1, 1, 0]);
    }

    #[test]
    fn diagonal_touch_differs_by_connectivity() {
        // Two single pixels touching only at a corner.
        // . X
        // X .
        let img = img_u8_2d(2, 2, vec![0, 10, 10, 0]);
        let face = connected_threshold(&img, &[vec![1, 0]], 5.0, 15.0, 1, false).unwrap();
        assert_eq!(face.scalar_slice::<u8>().unwrap(), &[0, 1, 0, 0]);

        let full = connected_threshold(&img, &[vec![1, 0]], 5.0, 15.0, 1, true).unwrap();
        assert_eq!(full.scalar_slice::<u8>().unwrap(), &[0, 1, 1, 0]);
    }

    #[test]
    fn seed_value_outside_interval_yields_empty_region() {
        let img = img_u8_2d(3, 1, vec![0, 100, 0]);
        let out = connected_threshold(&img, &[vec![1, 0]], 5.0, 15.0, 1, false).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0]);
    }

    #[test]
    fn seed_on_image_border_still_floods() {
        let img = img_u8_2d(3, 1, vec![10, 10, 0]);
        let out = connected_threshold(&img, &[vec![0, 0]], 5.0, 15.0, 1, false).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 1, 0]);
    }

    #[test]
    fn out_of_bounds_seed_is_dropped_not_errored() {
        let img = img_u8_2d(3, 1, vec![10, 10, 10]);
        let out = connected_threshold(&img, &[vec![99, 0]], 5.0, 15.0, 1, false).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0]);
    }

    #[test]
    fn empty_seed_list_yields_all_zero() {
        let img = img_u8_2d(3, 1, vec![10, 10, 10]);
        let out = connected_threshold(&img, &[], 5.0, 15.0, 1, false).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0]);
    }

    #[test]
    fn seed_dimension_mismatch_errors() {
        let img = img_u8_2d(3, 1, vec![10, 10, 10]);
        let err = connected_threshold(&img, &[vec![1]], 5.0, 15.0, 1, false).unwrap_err();
        assert_eq!(
            err,
            FilterError::DimensionLength {
                expected: 2,
                got: 1
            }
        );
    }

    // ---- neighborhood_connected -----------------------------------------

    #[test]
    fn neighborhood_connected_requires_whole_box_in_range() {
        // 5x1, all pixels 10 except index 3 which is an outlier (100).
        // A radius-1 box centered at index 2 covers indices 1..=3, so the
        // outlier at 3 excludes index 2 itself from the flood, which in turn
        // cuts the flood off from ever reaching index 4.
        let img = Image::from_vec(&[5, 1], vec![10u8, 10, 10, 100, 10]).unwrap();
        let out = neighborhood_connected(&img, &[vec![1, 0]], 5.0, 15.0, &[1, 0], 1).unwrap();
        // Index 0's box (clamped -1..=1) and index 1's box (0..=2) are both
        // entirely in-range -> included. Index 2's box (1..=3) contains the
        // outlier -> excluded.
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 1, 0, 0, 0]);
    }

    #[test]
    fn neighborhood_connected_zero_flux_at_border() {
        // radius-1 box at the left edge clamps its missing neighbor to the
        // edge pixel itself (zero-flux Neumann), so a uniform strip still
        // includes the border seed.
        let img = Image::from_vec(&[3, 1], vec![10u8, 10, 10]).unwrap();
        let out = neighborhood_connected(&img, &[vec![0, 0]], 5.0, 15.0, &[1, 0], 1).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[1, 1, 1]);
    }

    #[test]
    fn neighborhood_connected_empty_seeds_is_all_zero() {
        let img = Image::from_vec(&[3, 1], vec![10u8, 10, 10]).unwrap();
        let out = neighborhood_connected(&img, &[], 5.0, 15.0, &[1, 0], 1).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0]);
    }

    // ---- confidence_connected --------------------------------------------

    #[test]
    fn confidence_connected_zero_iterations_matches_initial_segmentation() {
        // Uniform block of 100 with two outliers just outside 3*sigma; with
        // radius 0 the initial stats come from the seed pixel alone
        // (mean=100, variance=0 -> lower=upper=100), so with 0 iterations
        // only the exact seed value should ever be admitted.
        let img = Image::from_vec(&[5, 1], vec![0u8, 100, 100, 100, 0]).unwrap();
        let out = confidence_connected(&img, &[vec![2, 0]], 0, 2.5, 0, 1).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 1, 1, 1, 0]);
    }

    #[test]
    fn confidence_connected_iteration_narrows_region_from_initial_pass() {
        // The seed's radius-1 neighborhood box includes a mildly different
        // neighbor (102 vs the seed's 100), inflating the initial variance
        // estimate just enough that round 0 floods straight through that
        // neighbor and the uniform run beyond it. Recomputing statistics
        // from that now-larger (mostly-100) region on round 1 gives a
        // materially tighter variance, which excludes the 102 neighbor and,
        // since this is a 1-D chain, cuts off everything past it too.
        let img = Image::from_vec(&[8, 1], vec![100u8, 102, 100, 100, 100, 100, 100, 130]).unwrap();

        let zero_iterations = confidence_connected(&img, &[vec![0, 0]], 0, 2.0, 1, 1).unwrap();
        assert_eq!(
            zero_iterations.scalar_slice::<u8>().unwrap(),
            &[1, 1, 1, 1, 1, 1, 1, 0],
            "round 0 alone floods straight through the 102 neighbor"
        );

        let iterated = confidence_connected(&img, &[vec![0, 0]], 4, 2.0, 1, 1).unwrap();
        assert_eq!(
            iterated.scalar_slice::<u8>().unwrap(),
            &[1, 0, 0, 0, 0, 0, 0, 0],
            "iterating recomputes a tighter variance that excludes the 102 neighbor"
        );
    }

    #[test]
    fn confidence_connected_empty_seeds_is_all_zero() {
        let img = Image::from_vec(&[3, 1], vec![10u8, 10, 10]).unwrap();
        let out = confidence_connected(&img, &[], 2, 2.5, 1, 1).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0]);
    }

    #[test]
    fn confidence_connected_out_of_bounds_seed_is_all_zero() {
        let img = Image::from_vec(&[3, 1], vec![10u8, 10, 10]).unwrap();
        let out = confidence_connected(&img, &[vec![99, 0]], 2, 2.5, 1, 1).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 0, 0]);
    }

    // ---- isolated_connected -----------------------------------------------

    #[test]
    fn isolated_connected_separates_two_blobs() {
        // Two plateaus (value 10 at index 0-1, value 50 at index 4-5)
        // joined by a bridge (value 20 at index 2-3) that stays connected to
        // seeds1 at every threshold from here up. Searching upward from
        // seeds1, the tightest separating upper bound is the largest value
        // that still excludes the 50-plateau: 49 (bisecting [0, 255] with
        // tolerance 1.0 converges there exactly).
        let img = Image::from_vec(&[6, 1], vec![10u8, 10, 20, 20, 50, 50]).unwrap();
        let result =
            isolated_connected(&img, &[vec![0, 0]], &[vec![5, 0]], 0.0, 255.0, 1.0, true, 1)
                .unwrap();
        assert!(!result.thresholding_failed);
        let mask = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(mask[0], 1);
        assert_eq!(mask[5], 0);
        assert_eq!(result.isolated_value, 49.0);
    }

    #[test]
    fn isolated_connected_cannot_separate_reports_failure() {
        // A single uniform plateau: any interval that includes seeds1 also
        // includes seeds2, so no bisection result can separate them.
        let img = Image::from_vec(&[4, 1], vec![10u8, 10, 10, 10]).unwrap();
        let result =
            isolated_connected(&img, &[vec![0, 0]], &[vec![3, 0]], 0.0, 255.0, 1.0, true, 1)
                .unwrap();
        assert!(result.thresholding_failed);
    }

    #[test]
    fn isolated_connected_empty_seeds1_errors() {
        let img = Image::from_vec(&[3, 1], vec![10u8, 10, 10]).unwrap();
        let err =
            isolated_connected(&img, &[], &[vec![2, 0]], 0.0, 255.0, 1.0, true, 1).unwrap_err();
        assert_eq!(err, FilterError::EmptySeeds { which: "seeds1" });
    }

    #[test]
    fn isolated_connected_empty_seeds2_errors() {
        let img = Image::from_vec(&[3, 1], vec![10u8, 10, 10]).unwrap();
        let err =
            isolated_connected(&img, &[vec![0, 0]], &[], 0.0, 255.0, 1.0, true, 1).unwrap_err();
        assert_eq!(err, FilterError::EmptySeeds { which: "seeds2" });
    }

    #[test]
    fn isolated_connected_find_lower_threshold() {
        // Mirror of the separates-two-blobs case, but searching downward:
        // seeds1 on the high plateau, seeds2 on the low one. The tightest
        // separating lower bound is the smallest value that still excludes
        // the 10-plateau while keeping the 20-bridge (and hence seeds1)
        // connected: 11 (bisecting [0, 255] with tolerance 1.0 converges
        // there exactly).
        let img = Image::from_vec(&[6, 1], vec![10u8, 10, 20, 20, 50, 50]).unwrap();
        let result = isolated_connected(
            &img,
            &[vec![5, 0]],
            &[vec![0, 0]],
            0.0,
            255.0,
            1.0,
            false,
            1,
        )
        .unwrap();
        assert!(!result.thresholding_failed);
        let mask = result.image.scalar_slice::<u8>().unwrap();
        assert_eq!(mask[5], 1);
        assert_eq!(mask[0], 0);
        assert_eq!(result.isolated_value, 11.0);
    }
}
