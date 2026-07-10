//! Multi-rater label-fusion filters: [`staple`], [`label_voting`] and
//! [`multi_label_staple`].
//!
//! Verified against ITK v6's
//! `Modules/Filtering/ImageCompare/include/itkSTAPLEImageFilter.h`/`.hxx`,
//! `Modules/Segmentation/LabelVoting/include/itkLabelVotingImageFilter.h`/`.hxx`
//! and `.../itkMultiLabelSTAPLEImageFilter.h`/`.hxx`, plus SimpleITK's
//! `Code/BasicFilters/yaml/{STAPLE,LabelVoting,MultiLabelSTAPLE}ImageFilter.yaml`
//! for the exposed parameters, their defaults, and the measurement accessors.
//!
//! All three take an arbitrary number of same-size, same-pixel-type input
//! images (`SetInput(i, ...)` upstream), so the Rust API takes a `&[&Image]`
//! slice, matching [`crate::grid_utility::tile`].
//!
//! ## `staple`
//!
//! Binary EM. Each input is thresholded to an indicator `D_ij ∈ {0,1}` by
//! `in.Get() > fg - 1e-10 && in.Get() < fg + 1e-10` — an epsilon window, not
//! an equality test, reproduced verbatim. `foreground_value` is a `double` at
//! the SimpleITK level but `InputPixelType` at the ITK level (`pixeltype:
//! Input` in the yaml), so it is cast to the input pixel type before the
//! comparison.
//!
//! There is **no** `p = q = 0.99999` initialisation in the `.hxx`: `W` is
//! seeded with the per-voxel mean of the indicators, and the first E-step
//! derives `p`/`q` from that `W`. The prior `g_t` is the mean of that seeded
//! `W` over the whole image, scaled by `confidence_weight`; it is computed
//! **once**, before the loop, and never updated.
//!
//! Convergence is `Σ`-free: the loop breaks on the first iteration after the
//! zeroth where **every** rater satisfies `(p_i - last_p_i)^2 <= 1e-14` and
//! `(q_i - last_q_i)^2 <= 1e-14` (`min_rms_error`, "7 digits of precision").
//! `elapsed_iterations` is the loop index at the break, or
//! `maximum_iterations` if the iteration cap was hit first.
//! `maximum_iterations` is clamped up to `1` because ITK declares it with
//! `itkSetClampMacro(MaximumIterations, unsigned int, 1, ...)`; with a cap of
//! zero the upstream filter would publish the uninitialised `p`/`q` scratch
//! arrays (`make_unique_for_overwrite`).
//!
//! **Fixed here (upstream bug §1.11):** upstream leaves the E-step ratios
//! `p_i = p_num / p_denom` and `q_i = q_num / q_denom` unguarded, so an
//! all-background input set (`p_denom == 0`) gives sensitivity `p_i = NaN`,
//! and its dual, an all-foreground set (`q_denom == 0`), gives specificity
//! `q_i = NaN`; the `NaN` then floods the output `W` through the M-step. A
//! zero denominator means the rater faced no trials of that class, so this
//! port takes the rate to be vacuously `1` (universal quantification over the
//! empty set) at both sites. `1` is correct at both, but the two sites justify
//! it differently — only the `p`-side is value-independent:
//!
//! - **`p`-side (all-background, `p_denom == 0`) is value-independent.** An
//!   all-background seed makes the prior `g_t == 0`, and the M-step
//!   `W = g_t·α / (g_t·α + (1−g_t)·β)` then zeros the numerator, so `W == 0`
//!   *regardless of `p`*. Any finite sensitivity yields the identical (correct
//!   all-zeros) fused output; `1` is merely the natural vacuous-truth choice.
//! - **`q`-side (all-foreground, `q_denom == 0`) needs `1` specifically.** An
//!   all-foreground seed makes `g_t == confidence_weight`, not `0`. In the
//!   M-step every rater is foreground, so `β = Π(1 − q_i)`, and only `q_i == 1`
//!   drives `β` to `0` and forces the correct fused output `W == 1`. When
//!   `confidence_weight != 1` (so `g_t != 1`) a *different* finite specificity
//!   leaks the `(1 − g_t)·β` term and changes `W` — e.g. `confidence_weight =
//!   0.5`, one rater: `q_i = 1` gives `W = 1`, but `q_i = 0.5` gives
//!   `W = 0.5 / (0.5 + 0.5·0.5) = 0.667`. So on the `q`-side `1` is the unique
//!   vacuous-specificity value that yields the correct all-foreground fusion,
//!   not an interchangeable finite choice. (Only when `confidence_weight == 1`,
//!   `g_t == 1`, does the `q`-side also become value-independent.)
//!
//! ## `label_voting`
//!
//! Per-voxel plurality vote over `0 ..= max_label`. The tie rule is the
//! `.hxx`'s exact scan, quirks included: the winner starts at label `0` with
//! `max_votes = votes[0]`, then for `l = 1 ..`, a **strictly greater** count
//! takes the lead (and raises `max_votes`), while a count merely *equal* to
//! `max_votes` sets the winner to `label_for_undecided_pixels` **without**
//! raising `max_votes` (a no-op — the tying count already equals it). A
//! later, strictly-larger count overrides an earlier tie, so the scan marks
//! a voxel undecided exactly when the global maximum is non-unique.
//!
//! When `label_for_undecided_pixels` is unset the label is `max_label + 1`.
//! Upstream `static_cast`s both the default and a caller-supplied value to the
//! output pixel type (`pixeltype: Output` in the yaml), which **wraps**: for a
//! `UInt8` input using all 256 label values, `max_label + 1 == 256` becomes
//! `0`, so every undecided voxel is silently relabelled as the real label `0`
//! while ITK only emits a warning ("No new label for undecided pixels, using
//! zero"). **Fixed here (upstream bug §1.15):** an undecided label that does
//! not fit the output pixel type is refused with
//! [`FilterError::UndecidedLabelNotRepresentable`] rather than wrapped onto a
//! label that means something else. [`multi_label_staple`] performs the same
//! check, for its own undecided label *and* for the one its internal voting
//! pass needs (`itkMultiLabelSTAPLEImageFilter.hxx:224` has the identical
//! `static_cast`, without even the warning).
//!
//! `.hxx` guards the vote with `NumericTraits<InputPixelType>::IsNonnegative`;
//! SimpleITK restricts this filter to the unsigned integer pixel types, where
//! that guard is vacuously true, so this port requires an unsigned integer
//! pixel type and drops the branch.
//!
//! ## `multi_label_staple`
//!
//! Multi-label EM over one `(max_label + 2) × (max_label + 1)` confusion
//! matrix per rater — one extra *row* for "reject" classifications. The
//! weights type is `float` (`itk::MultiLabelSTAPLEImageFilter<In, Out,
//! float>` in the yaml's `filter_type`), so every accumulation here is `f32`,
//! not `f64`.
//!
//! Confusion matrices are seeded from a plain [`label_voting`] pass over the
//! same inputs (with *its* default undecided label, never the caller's), then
//! **row**-normalised. The EM loop's M-step **column**-normalises the updated
//! matrices, so rows sum to one only at initialisation. Termination is
//! `max |updated - current| < termination_update_threshold` over every matrix
//! entry; with `maximum_number_of_iterations = None` the loop is unbounded,
//! exactly as upstream (`!m_HasMaximumNumberOfIterations` disables the test).
//!
//! **Fixed here (upstream bug §1.10):** upstream's seeding loop does
//! `++m_ConfusionMatrixArray[k][in.Get()][out.Get()]` where `out.Get()` is the
//! voting output, which may be the undecided label `max_label + 1` — one past
//! the matrix's last *column*. `Array2D` is a flat row-major buffer with
//! unchecked `operator[]`, so that write lands on `[in.Get() + 1][0]`, and
//! since there is a spare reject row the index stays inside the allocation;
//! ties in the seeding vote silently add counts to the *next* input label's
//! "decided label 0" cell. This port skips voting-undecided pixels instead —
//! a tie carries no evidence about any rater's confusion between two real
//! labels — matching upstream fix PR InsightSoftwareConsortium/ITK#6579.
//!
//! Prior probabilities default to the relative label frequencies across all
//! inputs (an array of length `max_label + 2` whose last entry stays zero;
//! only the first `max_label + 1` entries are normalised and read). A
//! caller-supplied array is used as-is — **not** renormalised — and must have
//! at least `max_label + 1` entries, the bound `InitializePriorProbabilities`
//! actually checks.
//!
//! The final labelling repeats the E-step and takes the arg-max, with the
//! `.hxx`'s `else if (!(W[ci] < winningLabelW))` branch: a tie (including the
//! initial `W[ci] == 0 == winningLabelW`) yields the undecided label, so an
//! all-zero weight vector labels the voxel undecided.
//!
//! ## Precision note
//!
//! Label values are read through [`sitk_core::Image::to_f64_vec`] and rounded
//! to `u64`, exact for every label below `2^53`. `UInt64` label values above
//! that would round; no SimpleITK test exercises them, and the crate's other
//! label filters ([`crate::overlap`], [`crate::label`]) take the same route.

use crate::error::{FilterError, Result};
use crate::{image_from_f64, quantize_to_pixel_type, require_same_shape};
use sitk_core::{Image, PixelId};
use std::cmp::Ordering;

/// `epsilon` in `STAPLEImageFilter::GenerateData` — the half-width of the
/// window used to test a voxel against `ForegroundValue`.
const STAPLE_EPSILON: f64 = 1.0e-10;

/// `min_rms_error` in `STAPLEImageFilter::GenerateData`: the squared-change
/// bound on every `p_i`/`q_i` that declares convergence ("7 digits of
/// precision").
const STAPLE_MIN_RMS_ERROR: f64 = 1.0e-14;

// ---- shared input handling ------------------------------------------------

/// Every filter here needs at least one input, and ITK requires the inputs to
/// share a `RequestedRegion` (and, being one template instantiation, a pixel
/// type).
fn require_inputs(images: &[&Image]) -> Result<()> {
    let Some((first, rest)) = images.split_first() else {
        return Err(FilterError::EmptyImageList);
    };
    for img in rest {
        require_same_shape(first, img)?;
    }
    Ok(())
}

/// `LabelVotingImageFilter` and `MultiLabelSTAPLEImageFilter` are generated by
/// SimpleITK only for the unsigned integer pixel types.
fn require_unsigned_integer(img: &Image) -> Result<()> {
    match img.pixel_id() {
        PixelId::UInt8 | PixelId::UInt16 | PixelId::UInt32 | PixelId::UInt64 => Ok(()),
        id => Err(FilterError::RequiresUnsignedIntegerPixelType(id)),
    }
}

/// The undecided-pixel label must be representable in the output pixel type,
/// or it is not a label at all: upstream `static_cast`s it, and C++ narrows
/// modulo `2^bits`, so `max_label + 1 == 256` becomes the perfectly valid
/// label `0` for a `UInt8` image. Fixed here (upstream bug §1.15) by refusing
/// the conversion rather than wrapping it onto a real label.
fn checked_undecided_label(id: PixelId, label: u64) -> Result<u64> {
    let maximum = match id {
        PixelId::UInt8 => u64::from(u8::MAX),
        PixelId::UInt16 => u64::from(u16::MAX),
        PixelId::UInt32 => u64::from(u32::MAX),
        // `UInt64` holds every `u64`; the other tags are rejected by
        // `require_unsigned_integer` before reaching here.
        _ => u64::MAX,
    };
    if label > maximum {
        return Err(FilterError::UndecidedLabelNotRepresentable {
            label,
            pixel_id: id,
            maximum,
        });
    }
    Ok(label)
}

/// Read every input's buffer as `u64` label values.
fn label_buffers(images: &[&Image]) -> Result<Vec<Vec<u64>>> {
    images
        .iter()
        .map(|img| Ok(img.to_f64_vec()?.iter().map(|&v| v as u64).collect()))
        .collect()
}

/// `ComputeMaximumInputValue()`: the largest label across all inputs.
fn maximum_input_value(inputs: &[Vec<u64>]) -> u64 {
    inputs
        .iter()
        .flat_map(|buf| buf.iter().copied())
        .max()
        .unwrap_or(0)
}

// ---- STAPLEImageFilter ----------------------------------------------------

/// Result of [`staple`]: the fuzzy ground-truth image plus the three
/// measurements SimpleITK exposes (`GetElapsedIterations`,
/// `GetSensitivity`, `GetSpecificity`).
#[derive(Clone, Debug, PartialEq)]
pub struct StapleResult {
    /// The per-voxel posterior `W` that a voxel belongs to the segmented
    /// object, as a `Float64` image (`NumericTraits<InputPixelType>::RealType`
    /// is `double` for every integer input type SimpleITK instantiates).
    pub image: Image,
    /// `GetElapsedIterations()`: the loop index the E-M algorithm broke at,
    /// or `maximum_iterations` if it never converged.
    pub elapsed_iterations: u32,
    /// `GetSensitivity()`: the true-positive fraction `p_i` per rater.
    pub sensitivity: Vec<f64>,
    /// `GetSpecificity()`: the true-negative fraction `q_i` per rater.
    pub specificity: Vec<f64>,
}

/// `STAPLEImageFilter`: Simultaneous Truth And Performance Level Estimation
/// over `images`, a set of binary expert segmentations sharing a size and an
/// integer pixel type.
///
/// SimpleITK defaults: `foreground_value = 1.0`, `maximum_iterations =
/// u32::MAX`, `confidence_weight = 1.0`. See the module docs for the EM
/// recurrence and its convergence test.
pub fn staple(
    images: &[&Image],
    foreground_value: f64,
    maximum_iterations: u32,
    confidence_weight: f64,
) -> Result<StapleResult> {
    require_inputs(images)?;
    let pixel_id = images[0].pixel_id();
    if pixel_id.is_floating_point() {
        return Err(FilterError::RequiresIntegerPixelType(pixel_id));
    }

    // `itkSetClampMacro(MaximumIterations, unsigned int, 1, max)`.
    let maximum_iterations = maximum_iterations.max(1);
    // `itkSetMacro(ForegroundValue, InputPixelType)` behind SimpleITK's
    // `double` parameter.
    let foreground_value = quantize_to_pixel_type(pixel_id, foreground_value);

    let raters: Vec<Vec<f64>> = images
        .iter()
        .map(|img| Ok(img.to_f64_vec()?))
        .collect::<Result<_>>()?;
    let indicator =
        |v: f64| v > foreground_value - STAPLE_EPSILON && v < foreground_value + STAPLE_EPSILON;

    let number_of_pixels = raters[0].len();
    let number_of_raters = raters.len();

    // Seed `W` with the per-voxel mean of the indicators.
    let mut w = vec![0.0f64; number_of_pixels];
    for rater in &raters {
        for (acc, &v) in w.iter_mut().zip(rater) {
            if indicator(v) {
                *acc += 1.0;
            }
        }
    }

    // ... then take its whole-image mean as the prior `g_t`. `N` is the pixel
    // count as a `double`; an empty image divides by zero, as upstream.
    let mut g_t = 0.0f64;
    for acc in w.iter_mut() {
        *acc /= number_of_raters as f64;
        g_t += *acc;
    }
    g_t = (g_t / number_of_pixels as f64) * confidence_weight;

    let mut p = vec![0.0f64; number_of_raters];
    let mut q = vec![0.0f64; number_of_raters];
    let mut last_p = vec![-10.0f64; number_of_raters];
    let mut last_q = vec![-10.0f64; number_of_raters];

    let mut iteration = 0u32;
    while iteration < maximum_iterations {
        // E-step: sensitivity and specificity of each rater against `W`.
        for (i, rater) in raters.iter().enumerate() {
            let (mut p_num, mut p_denom, mut q_num, mut q_denom) = (0.0, 0.0, 0.0, 0.0);
            for (&v, &wi) in rater.iter().zip(&w) {
                if indicator(v) {
                    p_num += wi;
                } else {
                    q_num += 1.0 - wi;
                }
                p_denom += wi;
                q_denom += 1.0 - wi;
            }
            // A zero denominator means the rater faced no trials of that class
            // (all-background empties `p_denom`, all-foreground empties
            // `q_denom`). The rate is then vacuously 1 — universal
            // quantification over the empty set — rather than upstream's
            // unguarded `0/0 = NaN` (§1.11). `1` is correct at both duals but
            // for different reasons (see the module doc): the `p`-side is
            // value-independent (the prior `g_t == 0` zeros the M-step
            // numerator), whereas on the `q`-side `1` is the *unique*
            // specificity that yields the correct fused `W` unless
            // `confidence_weight == 1`.
            p[i] = if p_denom == 0.0 { 1.0 } else { p_num / p_denom };
            q[i] = if q_denom == 0.0 { 1.0 } else { q_num / q_denom };
        }

        // M-step: rebuild `W` from the new `p`s and `q`s.
        for (pix, wi) in w.iter_mut().enumerate() {
            let mut alpha1 = 1.0f64;
            let mut beta1 = 1.0f64;
            for (i, rater) in raters.iter().enumerate() {
                if indicator(rater[pix]) {
                    alpha1 *= p[i];
                    beta1 *= 1.0 - q[i];
                } else {
                    alpha1 *= 1.0 - p[i];
                    beta1 *= q[i];
                }
            }
            *wi = g_t * alpha1 / (g_t * alpha1 + (1.0 - g_t) * beta1);
        }

        // Convergence is never declared on the zeroth iteration, where
        // `last_p`/`last_q` still hold their `-10.0` sentinels.
        let converged = iteration != 0
            && p.iter().zip(&last_p).all(|(&pi, &lpi)| {
                let d = pi - lpi;
                d * d <= STAPLE_MIN_RMS_ERROR
            })
            && q.iter().zip(&last_q).all(|(&qi, &lqi)| {
                let d = qi - lqi;
                d * d <= STAPLE_MIN_RMS_ERROR
            });
        last_p.copy_from_slice(&p);
        last_q.copy_from_slice(&q);
        if converged {
            break;
        }
        iteration += 1;
    }

    Ok(StapleResult {
        image: image_from_f64(PixelId::Float64, images[0].size(), images[0], &w)?,
        elapsed_iterations: iteration,
        sensitivity: p,
        specificity: q,
    })
}

// ---- LabelVotingImageFilter -----------------------------------------------

/// `DynamicThreadedGenerateData`'s per-voxel scan, shared with
/// [`multi_label_staple`]'s confusion-matrix seeding.
fn voting_labels(inputs: &[Vec<u64>], total_label_count: usize, undecided: u64) -> Vec<u64> {
    let mut votes = vec![0u32; total_label_count];
    let mut out = vec![0u64; inputs[0].len()];

    for (pix, o) in out.iter_mut().enumerate() {
        votes.fill(0);
        for input in inputs {
            votes[input[pix] as usize] += 1;
        }

        // Note `max_votes` is *not* raised on a tie, so a later strictly
        // larger count still wins and an equal count against a stale
        // `max_votes` still marks the voxel undecided.
        let mut winner = 0u64;
        let mut max_votes = votes[0];
        for (label, &count) in votes.iter().enumerate().skip(1) {
            if count > max_votes {
                max_votes = count;
                winner = label as u64;
            } else if count == max_votes {
                winner = undecided;
            }
        }
        *o = winner;
    }
    out
}

/// `LabelVotingImageFilter`: per-voxel plurality vote across `images`, which
/// must share a size and an unsigned integer pixel type. The output has the
/// input's pixel type.
///
/// `label_for_undecided_pixels` is SimpleITK's `LabelForUndecidedPixels`,
/// whose `u64::MAX` sentinel default ("leave unset") is spelled `None` here;
/// unset means `max_label + 1`. Either way the value must fit the output pixel
/// type, or the call errors rather than wrapping onto a real label — see the
/// module docs.
pub fn label_voting(images: &[&Image], label_for_undecided_pixels: Option<u64>) -> Result<Image> {
    require_inputs(images)?;
    let pixel_id = images[0].pixel_id();
    require_unsigned_integer(images[0])?;

    let inputs = label_buffers(images)?;
    let total_label_count = maximum_input_value(&inputs) as usize + 1;
    let undecided = checked_undecided_label(
        pixel_id,
        label_for_undecided_pixels.unwrap_or(total_label_count as u64),
    )?;

    let out = voting_labels(&inputs, total_label_count, undecided);
    let vals: Vec<f64> = out.iter().map(|&v| v as f64).collect();
    image_from_f64(pixel_id, images[0].size(), images[0], &vals)
}

// ---- MultiLabelSTAPLEImageFilter ------------------------------------------

/// Result of [`multi_label_staple`]: the fused label image plus the
/// measurements SimpleITK exposes (`GetElapsedNumberOfIterations`,
/// `GetPriorProbabilities`, `GetConfusionMatrix(i)`).
#[derive(Clone, Debug, PartialEq)]
pub struct MultiLabelStapleResult {
    /// The fused label image, with the inputs' pixel type.
    pub image: Image,
    /// `GetElapsedNumberOfIterations()`.
    pub elapsed_number_of_iterations: u32,
    /// `GetPriorProbabilities()`: the caller's array verbatim, or the
    /// estimated relative label frequencies (length `total_label_count + 1`,
    /// last entry zero).
    pub prior_probabilities: Vec<f32>,
    /// `GetConfusionMatrix(i)` for each rater `i`, flattened row-major as
    /// `(total_label_count + 1)` rows of `total_label_count` columns. Row
    /// `total_label_count` is the "reject" row.
    pub confusion_matrices: Vec<Vec<f32>>,
    /// `max_label + 1`: the number of labels, i.e. the confusion matrices'
    /// column count.
    pub total_label_count: usize,
}

/// `MultiLabelSTAPLEImageFilter`: multi-label EM with a per-rater confusion
/// matrix over `images`, which must share a size and an unsigned integer
/// pixel type.
///
/// SimpleITK defaults: `label_for_undecided_pixels` unset (`max_label + 1`),
/// `termination_update_threshold = 1e-5`, `maximum_number_of_iterations`
/// unset (unbounded), `prior_probabilities` unset (relative label
/// frequencies). See the module docs for the EM recurrence, the confusion
/// matrix layout and its reject-column quirk.
pub fn multi_label_staple(
    images: &[&Image],
    label_for_undecided_pixels: Option<u64>,
    termination_update_threshold: f32,
    maximum_number_of_iterations: Option<u32>,
    prior_probabilities: Option<&[f32]>,
) -> Result<MultiLabelStapleResult> {
    require_inputs(images)?;
    let pixel_id = images[0].pixel_id();
    require_unsigned_integer(images[0])?;

    let inputs = label_buffers(images)?;
    let n_labels = maximum_input_value(&inputs) as usize + 1;
    let n_pixels = inputs[0].len();
    let undecided = checked_undecided_label(
        pixel_id,
        label_for_undecided_pixels.unwrap_or(n_labels as u64),
    )?;

    // `AllocateConfusionMatrixArray`: `(n_labels + 1) x n_labels`, one spare
    // "reject" row, flat and row-major.
    let matrix_len = (n_labels + 1) * n_labels;
    let mut confusion: Vec<Vec<f32>> = vec![vec![0.0; matrix_len]; inputs.len()];

    // `InitializeConfusionMatrixArrayFromVoting`: a fresh `LabelVotingImageFilter`
    // over the same inputs, with *its* own default undecided label rather than
    // this filter's.
    let voting_undecided = checked_undecided_label(pixel_id, n_labels as u64)?;
    let vote = voting_labels(&inputs, n_labels, voting_undecided);
    for (matrix, input) in confusion.iter_mut().zip(&inputs) {
        for (&observed, &fused) in input.iter().zip(&vote) {
            // A voting tie has no column: `fused == n_labels` is one past the
            // last one. Such a pixel says nothing about how this rater confuses
            // two real labels, so it contributes no count (upstream instead
            // writes it into row `observed + 1`, column 0).
            if (fused as usize) < n_labels {
                matrix[observed as usize * n_labels + fused as usize] += 1.0;
            }
        }
    }
    // Normalize matrix rows to unit probability sum.
    for matrix in &mut confusion {
        for row in matrix.chunks_mut(n_labels) {
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for cell in row {
                    *cell /= sum;
                }
            }
        }
    }

    // `InitializePriorProbabilities`.
    let priors: Vec<f32> = match prior_probabilities {
        Some(given) => {
            if given.len() < n_labels {
                return Err(FilterError::InvalidPriorProbabilities {
                    got: given.len(),
                    expected: n_labels,
                });
            }
            given.to_vec()
        }
        None => {
            let mut priors = vec![0.0f32; n_labels + 1];
            for input in &inputs {
                for &label in input {
                    priors[label as usize] += 1.0;
                }
            }
            // `total_prob_mass` is the total labeled-pixel count across all
            // inputs; a zero-pixel image leaves it `0`, so this divides by
            // zero (`0/0 == NaN` for every label), exactly as upstream
            // `itkMultiLabelSTAPLEImageFilter.hxx:210` does
            // (`m_PriorProbabilities[l] /= totalProbMass`, unguarded). Left
            // undocumented before; documented here to match the `staple`
            // sibling's established rule for empty-image divisions (§1.11
            // family, the `g_t = (g_t / number_of_pixels) * ...` division
            // above), which reproduces upstream's 0/0 rather than guarding it.
            let total_prob_mass: f32 = priors[..n_labels].iter().sum();
            for prior in &mut priors[..n_labels] {
                *prior /= total_prob_mass;
            }
            priors
        }
    };

    let mut updated: Vec<Vec<f32>> = vec![vec![0.0; matrix_len]; inputs.len()];
    let mut w = vec![0.0f32; n_labels];

    let mut iteration = 0u32;
    loop {
        if maximum_number_of_iterations.is_some_and(|max| iteration >= max) {
            break;
        }
        for matrix in &mut updated {
            matrix.fill(0.0);
        }

        for pix in 0..n_pixels {
            // E step.
            w.copy_from_slice(&priors[..n_labels]);
            for (matrix, input) in confusion.iter().zip(&inputs) {
                let row = &matrix[input[pix] as usize * n_labels..][..n_labels];
                for (wc, &cell) in w.iter_mut().zip(row) {
                    *wc *= cell;
                }
            }

            // M step.
            let mut sum_w = w[0];
            for &wc in &w[1..] {
                sum_w += wc;
            }
            if sum_w != 0.0 {
                for wc in &mut w {
                    *wc /= sum_w;
                }
            }
            for (matrix, input) in updated.iter_mut().zip(&inputs) {
                let row = &mut matrix[input[pix] as usize * n_labels..][..n_labels];
                for (cell, &wc) in row.iter_mut().zip(&w) {
                    *cell += wc;
                }
            }
        }

        // Normalize each updated matrix's columns with the sum over all expert
        // decisions (every row, reject row included).
        for matrix in &mut updated {
            for ci in 0..n_labels {
                let mut sum_w = matrix[ci];
                for j in 1..=n_labels {
                    sum_w += matrix[j * n_labels + ci];
                }
                if sum_w != 0.0 {
                    for j in 0..=n_labels {
                        matrix[j * n_labels + ci] /= sum_w;
                    }
                }
            }
        }

        // Apply the update, recording the largest single parameter change.
        let mut maximum_update = 0.0f32;
        for (matrix, update) in confusion.iter_mut().zip(&updated) {
            for (cell, &new) in matrix.iter_mut().zip(update) {
                let change = (new - *cell).abs();
                if change > maximum_update {
                    maximum_update = change;
                }
                *cell = new;
            }
        }

        if maximum_update < termination_update_threshold {
            break;
        }
        iteration += 1;
    }

    // Build the combined output by repeating the E step against the estimated
    // confusion matrices.
    let mut out = vec![0u64; n_pixels];
    for (pix, o) in out.iter_mut().enumerate() {
        w.copy_from_slice(&priors[..n_labels]);
        for (matrix, input) in confusion.iter().zip(&inputs) {
            let row = &matrix[input[pix] as usize * n_labels..][..n_labels];
            for (wc, &cell) in w.iter_mut().zip(row) {
                *wc *= cell;
            }
        }

        // `if (W[ci] > winningLabelW) ... else if (!(W[ci] < winningLabelW))`:
        // a tie *or* an unordered comparison (`NaN`) yields the undecided
        // label. Spelled with `partial_cmp` so the `NaN` arm keeps the C++
        // meaning that `>=` would lose.
        let mut winning_label = undecided;
        let mut winning_w = 0.0f32;
        for (ci, &wc) in w.iter().enumerate() {
            match wc.partial_cmp(&winning_w) {
                Some(Ordering::Greater) => {
                    winning_w = wc;
                    winning_label = ci as u64;
                }
                Some(Ordering::Less) => {}
                _ => winning_label = undecided,
            }
        }
        *o = winning_label;
    }

    let vals: Vec<f64> = out.iter().map(|&v| v as f64).collect();
    Ok(MultiLabelStapleResult {
        image: image_from_f64(pixel_id, images[0].size(), images[0], &vals)?,
        elapsed_number_of_iterations: iteration,
        prior_probabilities: priors,
        confusion_matrices: confusion,
        total_label_count: n_labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img<T: sitk_core::Scalar>(size: &[usize], data: Vec<T>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- staple -----------------------------------------------------------

    #[test]
    fn staple_two_perfect_raters_converge_to_one() {
        // Both raters agree exactly: W seeds to the indicator itself, g_t =
        // 0.5, the first E-step gives p = q = 1 for both, and the rebuilt W is
        // the indicator again. Iteration 1 sees no change and breaks.
        let a = img(&[4], vec![1u8, 1, 0, 0]);
        let b = img(&[4], vec![1u8, 1, 0, 0]);
        let r = staple(&[&a, &b], 1.0, u32::MAX, 1.0).unwrap();

        assert_eq!(r.sensitivity, vec![1.0, 1.0]);
        assert_eq!(r.specificity, vec![1.0, 1.0]);
        assert_eq!(r.image.to_f64_vec().unwrap(), vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(r.elapsed_iterations, 1);
        assert_eq!(r.image.pixel_id(), PixelId::Float64);
    }

    #[test]
    fn staple_adversarial_rater_is_down_weighted() {
        // Two agreeing raters plus one that inverts every voxel. The inverted
        // rater must end with strictly lower sensitivity and specificity.
        let a = img(&[8], vec![1u8, 1, 1, 1, 0, 0, 0, 0]);
        let b = img(&[8], vec![1u8, 1, 1, 1, 0, 0, 0, 0]);
        let bad = img(&[8], vec![0u8, 0, 0, 0, 1, 1, 1, 1]);
        let r = staple(&[&a, &b, &bad], 1.0, u32::MAX, 1.0).unwrap();

        assert_eq!(r.sensitivity[0], r.sensitivity[1]);
        assert!(
            r.sensitivity[2] < r.sensitivity[0],
            "adversarial p {} not below good p {}",
            r.sensitivity[2],
            r.sensitivity[0]
        );
        assert!(
            r.specificity[2] < r.specificity[0],
            "adversarial q {} not below good q {}",
            r.specificity[2],
            r.specificity[0]
        );
        // The two honest raters carry the posterior to their own indicator.
        let w = r.image.to_f64_vec().unwrap();
        for (pix, &wi) in w.iter().enumerate() {
            let expected = if pix < 4 { 1.0 } else { 0.0 };
            assert!(
                (wi - expected).abs() < 1e-6,
                "W[{pix}] = {wi}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn staple_foreground_value_selects_the_label() {
        // Label 2 is the foreground; labels 0 and 1 are both background.
        let a = img(&[4], vec![2u8, 2, 1, 0]);
        let b = img(&[4], vec![2u8, 2, 0, 1]);
        let r = staple(&[&a, &b], 2.0, u32::MAX, 1.0).unwrap();
        assert_eq!(r.image.to_f64_vec().unwrap(), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn staple_maximum_iterations_caps_elapsed() {
        // The three-rater fixture above converges at iteration 4; capping at 2
        // stops the loop by the `iter < m_MaximumIterations` test instead, and
        // `GetElapsedIterations()` reports the cap.
        let a = img(&[8], vec![1u8, 1, 1, 1, 0, 0, 0, 0]);
        let b = img(&[8], vec![1u8, 1, 1, 1, 0, 0, 0, 0]);
        let bad = img(&[8], vec![0u8, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(
            staple(&[&a, &b, &bad], 1.0, u32::MAX, 1.0)
                .unwrap()
                .elapsed_iterations,
            4
        );
        let capped = staple(&[&a, &b, &bad], 1.0, 2, 1.0).unwrap();
        assert_eq!(capped.elapsed_iterations, 2);
    }

    #[test]
    fn staple_two_opposed_raters_are_a_fixed_point_at_one_half() {
        // One rater and its exact inverse: W seeds to 0.5 everywhere, so
        // p = q = 0.5 for both and the rebuilt W is 0.5 again. Iteration 1
        // sees no change and breaks.
        let a = img(&[8], vec![1u8, 1, 1, 1, 0, 0, 0, 0]);
        let bad = img(&[8], vec![0u8, 0, 0, 0, 1, 1, 1, 1]);
        let r = staple(&[&a, &bad], 1.0, u32::MAX, 1.0).unwrap();
        assert_eq!(r.sensitivity, vec![0.5, 0.5]);
        assert_eq!(r.specificity, vec![0.5, 0.5]);
        assert_eq!(r.image.to_f64_vec().unwrap(), vec![0.5; 8]);
        assert_eq!(r.elapsed_iterations, 1);
    }

    #[test]
    fn staple_maximum_iterations_clamped_up_to_one() {
        // `itkSetClampMacro(MaximumIterations, unsigned int, 1, max)`: a cap of
        // zero still runs one iteration, so p/q are defined.
        let a = img(&[4], vec![1u8, 1, 0, 0]);
        let b = img(&[4], vec![1u8, 1, 0, 0]);
        let r = staple(&[&a, &b], 1.0, 0, 1.0).unwrap();
        assert_eq!(r.elapsed_iterations, 1);
        assert_eq!(r.sensitivity, vec![1.0, 1.0]);
    }

    #[test]
    fn staple_all_foreground_rater_has_vacuous_specificity_one() {
        // An all-foreground rater has no background trials, so `q_denom == 0`.
        // Upstream divides 0/0 and reports specificity NaN; this port takes the
        // rate as vacuously 1 (§1.11). Sensitivity is the ordinary p = 1: every
        // foreground pixel is correctly called foreground. The seeded W is the
        // indicator (all 1), g_t = mean(W)*cw = 1*0.5 = 0.5, and the M-step
        // rebuilds W = g_t*p / (g_t*p + (1-g_t)*(1-q)) = 0.5 / (0.5 + 0.5*0) = 1
        // everywhere.
        let a = img(&[4], vec![1u8, 1, 1, 1]);
        let r = staple(&[&a], 1.0, 1, 0.5).unwrap();
        assert_eq!(r.sensitivity, vec![1.0]);
        assert_eq!(r.specificity, vec![1.0]);
        assert_eq!(r.image.to_f64_vec().unwrap(), vec![1.0; 4]);
    }

    #[test]
    fn staple_all_background_input_has_vacuous_sensitivity_one_and_zero_truth() {
        // The dual of the test above: an all-background input set gives every
        // rater no foreground trials, so `p_denom == 0`. Upstream reports
        // sensitivity NaN and floods W to NaN through the M-step; this port
        // takes p = 1 vacuously (§1.11). Specificity is the ordinary q = 1
        // (every background pixel correctly called background). The prior
        // g_t = mean(W)*cw = 0 (W seeds to all-zero), so the M-step gives
        // W = 0*1 / (0*1 + 1*0) ... = g_t*alpha / (g_t*alpha + (1-g_t)*beta):
        // with g_t = 0 and beta = q = 1 finite, W = 0 / (0 + 1*1) = 0 -- the
        // correct all-background fused truth, no NaN.
        let a = img(&[4], vec![0u8, 0, 0, 0]);
        let b = img(&[4], vec![0u8, 0, 0, 0]);
        let r = staple(&[&a, &b], 1.0, u32::MAX, 1.0).unwrap();
        assert_eq!(r.sensitivity, vec![1.0, 1.0]);
        assert_eq!(r.specificity, vec![1.0, 1.0]);
        assert_eq!(r.image.to_f64_vec().unwrap(), vec![0.0; 4]);
    }

    #[test]
    fn staple_rejects_float_input() {
        let a = img(&[2], vec![1.0f32, 0.0]);
        assert_eq!(
            staple(&[&a], 1.0, 10, 1.0),
            Err(FilterError::RequiresIntegerPixelType(PixelId::Float32))
        );
    }

    #[test]
    fn staple_rejects_empty_input_list() {
        assert_eq!(staple(&[], 1.0, 10, 1.0), Err(FilterError::EmptyImageList));
    }

    #[test]
    fn staple_rejects_size_mismatch() {
        let a = img(&[4], vec![1u8, 1, 0, 0]);
        let b = img(&[2], vec![1u8, 0]);
        assert_eq!(
            staple(&[&a, &b], 1.0, 10, 1.0),
            Err(FilterError::SizeMismatch {
                a: vec![4],
                b: vec![2]
            })
        );
    }

    #[test]
    fn staple_rejects_type_mismatch() {
        let a = img(&[2], vec![1u8, 0]);
        let b = img(&[2], vec![1u16, 0]);
        assert_eq!(
            staple(&[&a, &b], 1.0, 10, 1.0),
            Err(FilterError::TypeMismatch {
                a: PixelId::UInt8,
                b: PixelId::UInt16
            })
        );
    }

    // ---- label_voting -----------------------------------------------------

    #[test]
    fn label_voting_majority_wins() {
        let a = img(&[3], vec![1u16, 2, 3]);
        let b = img(&[3], vec![1u16, 2, 2]);
        let c = img(&[3], vec![1u16, 3, 2]);
        let out = label_voting(&[&a, &b, &c], None).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![1.0, 2.0, 2.0]);
        assert_eq!(out.pixel_id(), PixelId::UInt16);
    }

    #[test]
    fn label_voting_tie_hits_default_undecided_label() {
        // max label 2 => undecided = 3. Voxel 0 ties 1-vs-2; voxel 1 is a
        // three-way tie among 0, 1, 2.
        let a = img(&[2], vec![1u16, 0]);
        let b = img(&[2], vec![2u16, 1]);
        let c = img(&[2], vec![1u16, 2]);
        // Voxel 0: votes = [0, 2, 1] -> label 1 wins outright.
        let out = label_voting(&[&a, &b, &c], None).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![1.0, 3.0]);
    }

    #[test]
    fn label_voting_two_way_tie_is_undecided() {
        let a = img(&[1], vec![1u16]);
        let b = img(&[1], vec![2u16]);
        let out = label_voting(&[&a, &b], None).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![3.0]);
    }

    #[test]
    fn label_voting_explicit_undecided_label() {
        let a = img(&[1], vec![1u16]);
        let b = img(&[1], vec![2u16]);
        let out = label_voting(&[&a, &b], Some(9)).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![9.0]);
    }

    #[test]
    fn label_voting_later_strict_winner_overrides_earlier_tie() {
        // votes = [1, 1, 2]: label 1 ties label 0 at one vote (marking the
        // voxel undecided), then label 2's two votes take the lead. `max_votes`
        // is still 1 when label 2 is scanned, so `2 > 1` wins.
        let a = img(&[1], vec![0u16]);
        let b = img(&[1], vec![1u16]);
        let c = img(&[1], vec![2u16]);
        let d = img(&[1], vec![2u16]);
        let out = label_voting(&[&a, &b, &c, &d], None).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![2.0]);
    }

    #[test]
    fn label_voting_tie_against_stale_max_votes_is_undecided() {
        // votes = [0, 2, 0, 2]. Scan: max_votes = votes[0] = 0. l=1: 2 > 0,
        // winner = 1, max_votes = 2. l=2: 0 != 2, no change. l=3: 2 == 2,
        // winner = undecided (= 4).
        let a = img(&[1], vec![1u16]);
        let b = img(&[1], vec![1u16]);
        let c = img(&[1], vec![3u16]);
        let d = img(&[1], vec![3u16]);
        let out = label_voting(&[&a, &b, &c, &d], None).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![4.0]);
    }

    #[test]
    fn label_voting_rejects_an_undecided_label_that_does_not_fit_the_output_type() {
        // max label 255 => total_label_count 256, which does not fit a `u8`.
        // ITK's `static_cast<unsigned char>(256)` is 0, so voxel 0 (a 1-1 tie
        // between labels 0 and 255) would come back labelled 0 — an actual
        // label, indistinguishable from an agreed vote for 0. Refuse instead.
        let a = img(&[2], vec![255u8, 7]);
        let b = img(&[2], vec![0u8, 7]);
        assert_eq!(
            label_voting(&[&a, &b], None),
            Err(FilterError::UndecidedLabelNotRepresentable {
                label: 256,
                pixel_id: PixelId::UInt8,
                maximum: 255,
            })
        );

        // The same labels in a `u16` image have room for the 256 label.
        let a = img(&[2], vec![255u16, 7]);
        let b = img(&[2], vec![0u16, 7]);
        let out = label_voting(&[&a, &b], None).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![256.0, 7.0]);

        // 255 labels (max label 254) still fit: 255 is representable, and the
        // guard is `>`, not `>=`.
        let a = img(&[2], vec![254u8, 7]);
        let b = img(&[2], vec![0u8, 7]);
        let out = label_voting(&[&a, &b], None).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![255.0, 7.0]);
        assert_eq!(out.pixel_id(), PixelId::UInt8);
    }

    #[test]
    fn label_voting_rejects_an_explicit_undecided_label_that_does_not_fit() {
        // A caller-supplied label goes through the same `static_cast` upstream:
        // 300 & 0xff == 44, another perfectly ordinary label.
        let a = img(&[1], vec![1u8]);
        let b = img(&[1], vec![2u8]);
        assert_eq!(
            label_voting(&[&a, &b], Some(300)),
            Err(FilterError::UndecidedLabelNotRepresentable {
                label: 300,
                pixel_id: PixelId::UInt8,
                maximum: 255,
            })
        );
    }

    #[test]
    fn label_voting_single_input_is_the_identity() {
        let a = img(&[4], vec![0u16, 1, 2, 3]);
        let out = label_voting(&[&a], None).unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn label_voting_rejects_signed_and_float_input() {
        let a = img(&[2], vec![1i16, 0]);
        assert_eq!(
            label_voting(&[&a], None),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Int16
            ))
        );
        let b = img(&[2], vec![1.0f64, 0.0]);
        assert_eq!(
            label_voting(&[&b], None),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Float64
            ))
        );
    }

    #[test]
    fn label_voting_rejects_empty_input_list() {
        assert_eq!(label_voting(&[], None), Err(FilterError::EmptyImageList));
    }

    #[test]
    fn label_voting_rejects_size_mismatch() {
        let a = img(&[4], vec![1u8, 1, 0, 0]);
        let b = img(&[2], vec![1u8, 0]);
        assert_eq!(
            label_voting(&[&a, &b], None),
            Err(FilterError::SizeMismatch {
                a: vec![4],
                b: vec![2]
            })
        );
    }

    #[test]
    fn label_voting_rejects_type_mismatch() {
        let a = img(&[2], vec![1u8, 0]);
        let b = img(&[2], vec![1u16, 0]);
        assert_eq!(
            label_voting(&[&a, &b], None),
            Err(FilterError::TypeMismatch {
                a: PixelId::UInt8,
                b: PixelId::UInt16
            })
        );
    }

    // ---- multi_label_staple -----------------------------------------------

    /// `maximum_number_of_iterations = Some(0)` skips the EM loop entirely, so
    /// the confusion matrices are exactly the row-normalised voting seed.
    #[test]
    fn multi_label_staple_seed_confusion_matrix_rows_sum_to_one() {
        let a = img(&[6], vec![0u16, 0, 1, 1, 2, 2]);
        let b = img(&[6], vec![0u16, 0, 1, 1, 2, 2]);
        let c = img(&[6], vec![0u16, 1, 1, 2, 2, 0]);
        let r = multi_label_staple(&[&a, &b, &c], None, 1e-5, Some(0), None).unwrap();

        assert_eq!(r.elapsed_number_of_iterations, 0);
        assert_eq!(r.total_label_count, 3);
        for (k, matrix) in r.confusion_matrices.iter().enumerate() {
            assert_eq!(matrix.len(), 4 * 3);
            for (j, row) in matrix.chunks(3).enumerate() {
                let sum: f32 = row.iter().sum();
                // A label that never appears as this rater's decision leaves an
                // all-zero row, which upstream leaves un-normalised.
                if row.iter().any(|&v| v != 0.0) {
                    assert!(
                        (sum - 1.0).abs() < 1e-6,
                        "rater {k} row {j} sums to {sum}, not 1"
                    );
                } else {
                    assert_eq!(sum, 0.0, "rater {k} row {j}");
                }
            }
        }
    }

    #[test]
    fn multi_label_staple_perfect_raters_reproduce_the_input() {
        let a = img(&[6], vec![0u16, 0, 1, 1, 2, 2]);
        let b = img(&[6], vec![0u16, 0, 1, 1, 2, 2]);
        let r = multi_label_staple(&[&a, &b], None, 1e-5, None, None).unwrap();
        assert_eq!(
            r.image.to_f64_vec().unwrap(),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        );
        assert_eq!(r.image.pixel_id(), PixelId::UInt16);
    }

    #[test]
    fn multi_label_staple_down_weights_an_adversarial_rater() {
        // Two honest raters and one that shifts every label by one. The fused
        // labelling must follow the honest majority.
        let a = img(&[6], vec![0u16, 0, 1, 1, 2, 2]);
        let b = img(&[6], vec![0u16, 0, 1, 1, 2, 2]);
        let bad = img(&[6], vec![1u16, 1, 2, 2, 0, 0]);
        let r = multi_label_staple(&[&a, &b, &bad], None, 1e-5, Some(50), None).unwrap();
        assert_eq!(
            r.image.to_f64_vec().unwrap(),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        );

        // The honest raters' confusion matrices are near-diagonal; the
        // adversarial rater's mass sits off the diagonal.
        let honest = &r.confusion_matrices[0];
        let adversarial = &r.confusion_matrices[2];
        for label in 0..3usize {
            assert!(
                honest[label * 3 + label] > 0.9,
                "honest diagonal [{label}][{label}] = {}",
                honest[label * 3 + label]
            );
            assert!(
                adversarial[label * 3 + label] < 0.1,
                "adversarial diagonal [{label}][{label}] = {}",
                adversarial[label * 3 + label]
            );
        }
    }

    #[test]
    fn multi_label_staple_default_priors_are_relative_label_frequencies() {
        // 2 raters x 4 voxels = 8 label draws: six 0s and two 1s.
        let a = img(&[4], vec![0u8, 0, 0, 1]);
        let b = img(&[4], vec![0u8, 0, 0, 1]);
        let r = multi_label_staple(&[&a, &b], None, 1e-5, Some(0), None).unwrap();
        // Length is total_label_count + 1; the trailing reject entry stays 0.
        assert_eq!(r.prior_probabilities, vec![0.75, 0.25, 0.0]);
    }

    #[test]
    fn multi_label_staple_uses_caller_priors_verbatim() {
        let a = img(&[2], vec![0u8, 1]);
        let b = img(&[2], vec![0u8, 1]);
        // Deliberately un-normalised: upstream never renormalises a supplied
        // array, it only length-checks it.
        let r = multi_label_staple(&[&a, &b], None, 1e-5, Some(0), Some(&[3.0, 7.0])).unwrap();
        assert_eq!(r.prior_probabilities, vec![3.0, 7.0]);
    }

    #[test]
    fn multi_label_staple_rejects_short_prior_probabilities() {
        let a = img(&[3], vec![0u8, 1, 2]);
        assert_eq!(
            multi_label_staple(&[&a], None, 1e-5, Some(1), Some(&[0.5, 0.5])),
            Err(FilterError::InvalidPriorProbabilities {
                got: 2,
                expected: 3
            })
        );
    }

    #[test]
    fn multi_label_staple_all_zero_weights_yield_the_undecided_label() {
        // Priors of zero on every label drive every `W[ci]` to zero, so the
        // arg-max scan's `!(W[ci] < winningLabelW)` branch fires on the first
        // label and the voxel stays undecided (= max_label + 1 = 2).
        let a = img(&[2], vec![0u8, 1]);
        let r = multi_label_staple(&[&a], None, 1e-5, Some(0), Some(&[0.0, 0.0])).unwrap();
        assert_eq!(r.image.to_f64_vec().unwrap(), vec![2.0, 2.0]);
    }

    #[test]
    fn multi_label_staple_explicit_undecided_label() {
        let a = img(&[2], vec![0u8, 1]);
        let r = multi_label_staple(&[&a], Some(7), 1e-5, Some(0), Some(&[0.0, 0.0])).unwrap();
        assert_eq!(r.image.to_f64_vec().unwrap(), vec![7.0, 7.0]);
    }

    #[test]
    fn multi_label_staple_rejects_an_undecided_label_that_does_not_fit_uint8() {
        // total_label_count = 256 does not fit a `u8`, so upstream's undecided
        // label wraps to 0 just as in `label_voting`.
        let a = img(&[2], vec![0u8, 255]);
        assert_eq!(
            multi_label_staple(&[&a], None, 1e-5, Some(0), Some(&[0.0; 256])),
            Err(FilterError::UndecidedLabelNotRepresentable {
                label: 256,
                pixel_id: PixelId::UInt8,
                maximum: 255,
            })
        );

        // The internal voting pass needs `max_label + 1` as *its* undecided
        // label too, so an in-range caller label does not rescue the call.
        assert_eq!(
            multi_label_staple(&[&a], Some(7), 1e-5, Some(0), Some(&[0.0; 256])),
            Err(FilterError::UndecidedLabelNotRepresentable {
                label: 256,
                pixel_id: PixelId::UInt8,
                maximum: 255,
            })
        );

        // In a `u16` image the 256 label fits, and the all-zero priors send
        // both voxels to it.
        let a = img(&[2], vec![0u16, 255]);
        let r = multi_label_staple(&[&a], None, 1e-5, Some(0), Some(&[0.0; 256])).unwrap();
        assert_eq!(r.image.to_f64_vec().unwrap(), vec![256.0, 256.0]);
        assert_eq!(r.total_label_count, 256);
    }

    #[test]
    fn multi_label_staple_seed_vote_ties_contribute_no_counts() {
        // Two raters that never agree: every voxel is a voting tie, so the
        // seeding vote is the undecided label 2 = n_labels everywhere, which
        // has no confusion-matrix column. Every count is skipped, so both
        // 3x2 matrices stay all-zero and row normalisation (`sum > 0`) leaves
        // them alone.
        //
        // Upstream instead increments column `n_labels` (= 2), which the flat
        // row-major buffer aliases onto row `observed + 1`, column 0, giving
        // both matrices [0,0, 1,0, 1,0] — a fabricated certainty that input
        // label 1 means output label 0.
        let a = img(&[2], vec![0u8, 1]);
        let b = img(&[2], vec![1u8, 0]);
        let r = multi_label_staple(&[&a, &b], None, 1e-5, Some(0), None).unwrap();

        assert_eq!(r.total_label_count, 2);
        assert_eq!(r.confusion_matrices[0], vec![0.0; 6]);
        assert_eq!(r.confusion_matrices[1], vec![0.0; 6]);
    }

    #[test]
    fn multi_label_staple_seed_keeps_decided_voxels_when_some_voxels_tie() {
        // a = [0, 0, 1, 0], b = [0, 1, 1, 1]; n_labels = 2, so undecided = 2.
        // Per-voxel votes over the two raters, and the `.hxx` scan:
        //   v0: votes [2, 0] -> winner 0
        //   v1: votes [1, 1] -> label 1 ties votes[0] -> undecided (2)
        //   v2: votes [0, 2] -> winner 1
        //   v3: votes [1, 1] -> undecided (2)
        // Seeding counts, skipping the two undecided voxels:
        //   rater a: v0 (observed 0, fused 0), v2 (observed 1, fused 1)
        //   rater b: v0 (observed 0, fused 0), v2 (observed 1, fused 1)
        // Both raw matrices are rows [1,0], [0,1], [0,0]; row normalisation is
        // the identity on them.
        //
        // Upstream's out-of-column write would instead put rater a's two
        // undecided voxels (observed 0) at row 1 column 0, giving row 1 =
        // [2, 1] -> [0.667, 0.333], and rater b's (observed 1) at row 2
        // column 0, giving reject row [1, 0].
        let a = img(&[4], vec![0u8, 0, 1, 0]);
        let b = img(&[4], vec![0u8, 1, 1, 1]);
        let r = multi_label_staple(&[&a, &b], None, 1e-5, Some(0), None).unwrap();

        assert_eq!(r.total_label_count, 2);
        let expected = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(r.confusion_matrices[0], expected);
        assert_eq!(r.confusion_matrices[1], expected);
    }

    #[test]
    fn multi_label_staple_rejects_signed_input() {
        let a = img(&[2], vec![1i32, 0]);
        assert_eq!(
            multi_label_staple(&[&a], None, 1e-5, Some(1), None),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Int32
            ))
        );
    }

    #[test]
    fn multi_label_staple_rejects_empty_input_list() {
        assert_eq!(
            multi_label_staple(&[], None, 1e-5, Some(1), None),
            Err(FilterError::EmptyImageList)
        );
    }

    #[test]
    fn multi_label_staple_rejects_size_mismatch() {
        let a = img(&[4], vec![1u8, 1, 0, 0]);
        let b = img(&[2], vec![1u8, 0]);
        assert_eq!(
            multi_label_staple(&[&a, &b], None, 1e-5, Some(1), None),
            Err(FilterError::SizeMismatch {
                a: vec![4],
                b: vec![2]
            })
        );
    }

    #[test]
    fn multi_label_staple_rejects_type_mismatch() {
        let a = img(&[2], vec![1u8, 0]);
        let b = img(&[2], vec![1u16, 0]);
        assert_eq!(
            multi_label_staple(&[&a, &b], None, 1e-5, Some(1), None),
            Err(FilterError::TypeMismatch {
                a: PixelId::UInt8,
                b: PixelId::UInt16
            })
        );
    }
}
