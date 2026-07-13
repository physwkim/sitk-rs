//! Automatic optimizer parameter-scale and learning-rate estimation
//! (`itk::RegistrationParameterScalesEstimator` and its three concrete
//! subclasses, `Modules/Numerics/Optimizersv4`).
//!
//! A gradient-descent step `p ← p − lr·(grad ⊘ scales)` needs two things a user
//! should not have to hand-tune: **scales** that make a unit change in each
//! parameter produce a comparable effect (so matrix and translation parameters
//! are optimized together), and a **learning rate** that bounds the first step
//! to about one voxel. ITK derives both from how far a parameter change moves
//! sample points drawn from the *virtual domain*.
//!
//! # The three estimators
//!
//! All three share [`RegistrationParameterScalesEstimator`]'s sampling
//! (`itkRegistrationParameterScalesEstimator.hxx`) and differ only in what they
//! measure at each sample. `δ` below is the small parameter variation
//! (`m_SmallParameterVariation`, default `0.01`), `J(x)` is the transform's
//! parameter Jacobian at `x`, and `N` is the number of sample points.
//!
//! | Kind | `EstimateScales` | `EstimateStepScale(step)` |
//! |---|---|---|
//! | [`PhysicalShift`] | `scaleᵢ = (maxₓ‖J(x)·δeᵢ‖ / δ)²` | `maxₓ‖J(x)·step‖`, linearized |
//! | [`IndexShift`] | same, with the shift measured in moving-image continuous-index units | same |
//! | [`Jacobian`] | `scaleᵢ = (1/N)·Σₓ ‖J(x)ᵢ‖²` | `(1/N)·Σₓ ‖J(x)·step‖` |
//!
//! [`PhysicalShift`]: ScalesEstimatorKind::PhysicalShift
//! [`IndexShift`]: ScalesEstimatorKind::IndexShift
//! [`Jacobian`]: ScalesEstimatorKind::Jacobian
//! [`RegistrationParameterScalesEstimator`]: self
//!
//! Note the **max** vs **mean**: the two shift estimators
//! (`itkRegistrationParameterScalesFromShiftBase.hxx:194-210`,
//! `ComputeMaximumVoxelShift`) take the worst sample; the Jacobian estimator
//! (`itkRegistrationParameterScalesFromJacobian.hxx:44-60`, `:76-82`) averages
//! over samples. Two estimators fed the *same* sample points therefore return
//! different scales, which is why the sample set is part of the port and not an
//! implementation detail — see *Sampling* below.
//!
//! `IndexShift`'s shift is a continuous-index distance while
//! [`ScalesEstimator::max_step_size`] (`EstimateMaximumStepSize`,
//! `itkRegistrationParameterScalesEstimator.hxx:48-64`) stays the minimum
//! *physical* virtual spacing, so the learning rate it produces mixes units.
//! That is upstream behavior, reproduced here (ledger §2.114).
//!
//! # Sampling
//!
//! ITK's estimator samples the virtual domain itself; it does **not** reuse the
//! metric's sample points, its sampling percentage, or its fixed mask
//! (`SetScalesSamplingStrategy` / `SetStepScaleSamplingStrategy`,
//! `itkRegistrationParameterScalesEstimator.hxx:346-393`):
//!
//! | Transform | `EstimateScales` samples | `EstimateStepScale` samples |
//! |---|---|---|
//! | local support (displacement field) | central region, radius `m_CentralRegionRadius` | full domain |
//! | linear (`IsLinear()`) | the `2ᵈ` domain corners | the `2ᵈ` domain corners |
//! | otherwise | `SizeOfSmallDomain`-scaled random draw | same random draw |
//!
//! Corner sampling is exact rather than an approximation for a linear
//! transform: `‖J(x)·Δ‖` is a norm of an affine function of `x`, hence convex,
//! so its maximum over the domain box is attained at a vertex. That is why the
//! *shift* estimators lose nothing by looking only at corners — and why the
//! *Jacobian* estimator, which averages, gets a genuinely different number from
//! corners than it would from the full domain.
//!
//! `central_region_radius` therefore has **no observable effect** for any
//! transform this crate registers: a linear transform never reaches
//! central-region sampling, and a displacement field's local Jacobian is the
//! identity at every grid-aligned sample, so its scales are `1` whatever the
//! radius selects. Upstream has the same property; the argument is carried,
//! honored where ITK honors it, and pinned by `VirtualGrid`'s own
//! central-region tests (ledger §2.115).
//!
//! # Local-support transforms (`DisplacementFieldTransform`)
//!
//! A dense transform's parameter count is small (a handful to a few dozen), so
//! probing every parameter and caching a `dim × nparams` Jacobian per sample is
//! cheap. A [`DisplacementFieldTransform`](sitk_transform::DisplacementFieldTransform)
//! has one parameter vector *per pixel*: for a modest 3-D volume `nparams` is in
//! the millions, so the dense probe (`O(nparams)` perturbations, each scanning
//! `O(nsamples)` points, each needing an `O(nparams)`-wide Jacobian row) is
//! intractable — exactly the dense-Jacobian problem the Mattes metric's
//! `evaluate_local_support` (`mattes.rs`) solves for the metric derivative.
//!
//! ITK keys this off `HasLocalSupport()`/`GetTransformCategory() ==
//! DisplacementField` (this crate's [`ParametricTransform::has_local_support`])
//! and switches to a fundamentally different algorithm in
//! `itkRegistrationParameterScalesFromShiftBase.hxx`:
//!
//! - **`EstimateScales`** (lines 34–119) sizes its output to
//!   `GetNumberOfLocalParameters()`, *not* the full parameter count (line 43:
//!   `parameterScales.SetSize(numLocalPara)`), and probes only those
//!   `numLocalPara` axes at the parameter block owning the virtual domain's
//!   central index (lines 54–86, `ComputeParameterOffsetFromVirtualIndex`)
//!   instead of every one of the `nparams` parameters. For
//!   [`DisplacementFieldTransform`](sitk_transform::DisplacementFieldTransform)
//!   the local Jacobian at a grid-aligned sample is always the `dim × dim`
//!   identity (`sparse_jacobian_wrt_parameters` in `sitk-transform/src/displacement.rs`
//!   — the interpolation weight of a pixel at its own grid index is exactly 1,
//!   every other weight 0), so every probe shifts by exactly `δ` and the
//!   ITK formula (lines 102–116: `scaleᵢ = maxShiftᵢ² / δ²`) reduces to
//!   `scaleᵢ = 1`: "every local parameter is a displacement in physical units
//!   already and needs no rebalancing."
//! - **`EstimateStepScale`** (lines 124–155) short-circuits *before* the
//!   dense-path linearization: `if (TransformHasLocalSupportForScalesEstimation())
//!   return this->ComputeMaximumVoxelShift(step);` (lines 132–135) — the full,
//!   unnormalized `step` vector is applied as-is and the shift is the max over
//!   *all* sample points of `‖T_{p+step}(x) − T_p(x)‖`. Local support means each
//!   sample's shift depends only on its own owning parameter block, so this is
//!   `maxₓ ‖local_jacobian(x) · step[offset(x)..offset(x)+numLocal]‖` — no
//!   `nparams`-wide structure, just one small dot product per sample.
//!
//! **A wrinkle this crate does not port.** ITK's optimizer
//! (`itkObjectToObjectOptimizerBase.h`, `m_Scales` doc comment: "Size is
//! expected to be == metric->GetNumberOfLocalParameters()"; enforced in
//! `itkObjectToObjectOptimizerBase.cxx`'s scales-validation block) stores the
//! *short* `numberOfLocalParameters()`-length scales array and broadcasts it
//! itself when scaling each pixel's local gradient block. This crate's
//! optimizers (`optimizer.rs`) do not have that broadcast: they index `scales`
//! 1:1 against the full gradient (`assert_eq!(s.len(), n)`, `n =
//! initial.len()`). So [`ScalesEstimator::estimate_scales`] returns the
//! ITK-derived per-local-parameter scale *tiled* across every parameter block —
//! a full `nparams`-length vector whose entries are numerically identical to
//! what a broadcast-aware optimizer would read out of ITK's short array (every
//! entry is the same value, so indexing by absolute position or by
//! local-block-relative position gives the same number). Giving
//! `optimizer.rs` genuine broadcast support (so this can return the literal
//! `numberOfLocalParameters()`-length array) is out of this module's scope.
//!
//! [`ParametricTransform::has_local_support`]: sitk_transform::ParametricTransform::has_local_support

use sitk_transform::ParametricTransform;

use crate::metric::{SplitMix64, local_support_block};

/// ITK's `m_SmallParameterVariation` default
/// (`itkRegistrationParameterScalesFromShiftBase.hxx:28`).
pub const DEFAULT_SMALL_PARAMETER_VARIATION: f64 = 0.01;

/// ITK's `m_CentralRegionRadius` default
/// (`itkRegistrationParameterScalesEstimator.hxx:35`). SimpleITK re-declares it
/// as the `centralRegionRadius = 5` default argument of every
/// `SetOptimizerScalesFrom*` overload.
pub const DEFAULT_CENTRAL_REGION_RADIUS: usize = 5;

/// ITK's `SizeOfSmallDomain` (`itkRegistrationParameterScalesEstimator.h:358`),
/// the random-sampling budget.
const SIZE_OF_SMALL_DOMAIN: usize = 1000;

/// Seed for [`Sampling::Random`]. ITK draws through
/// `ImageRandomConstIteratorWithIndex`'s global Mersenne Twister, which this
/// crate does not port; a fixed seed keeps the draw reproducible instead
/// (ledger §4.77, and the same deviation `FixedSamples` already documents for
/// `SamplingStrategy::Random`).
const RANDOM_SAMPLING_SEED: u64 = 0;

/// Which upstream estimator to run — SimpleITK's `m_OptimizerScalesType`
/// (`sitkImageRegistrationMethod.h:461-467`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalesEstimatorKind {
    /// `itk::RegistrationParameterScalesFromPhysicalShift`: the shift a
    /// parameter variation produces, in physical units.
    PhysicalShift {
        central_region_radius: usize,
        small_parameter_variation: f64,
    },
    /// `itk::RegistrationParameterScalesFromIndexShift`: the same shift,
    /// measured in the moving image's continuous-index units.
    IndexShift {
        central_region_radius: usize,
        small_parameter_variation: f64,
    },
    /// `itk::RegistrationParameterScalesFromJacobian`: the mean squared
    /// Jacobian column norm. Takes no `small_parameter_variation` — it never
    /// probes with a parameter variation.
    Jacobian { central_region_radius: usize },
}

impl ScalesEstimatorKind {
    fn central_region_radius(&self) -> usize {
        match *self {
            Self::PhysicalShift {
                central_region_radius,
                ..
            }
            | Self::IndexShift {
                central_region_radius,
                ..
            }
            | Self::Jacobian {
                central_region_radius,
            } => central_region_radius,
        }
    }

    /// `δ`. The Jacobian estimator has none; its value is unused there.
    fn small_parameter_variation(&self) -> f64 {
        match *self {
            Self::PhysicalShift {
                small_parameter_variation,
                ..
            }
            | Self::IndexShift {
                small_parameter_variation,
                ..
            } => small_parameter_variation,
            Self::Jacobian { .. } => DEFAULT_SMALL_PARAMETER_VARIATION,
        }
    }
}

impl Default for ScalesEstimatorKind {
    /// SimpleITK's constructor default (`sitkImageRegistrationMethod.cxx:70`,
    /// `m_OptimizerScalesType(Jacobian)`) is overridden by every
    /// `SetOptimizerScalesFrom*` call; this crate's own default is the
    /// physical-shift estimator, which is what `set_optimizer_scales_from_physical_shift`
    /// and the learning-rate estimator use.
    fn default() -> Self {
        Self::PhysicalShift {
            central_region_radius: DEFAULT_CENTRAL_REGION_RADIUS,
            small_parameter_variation: DEFAULT_SMALL_PARAMETER_VARIATION,
        }
    }
}

/// The virtual domain as a grid: everything needed to turn a voxel index into
/// the physical point ITK's `SampleVirtualDomain*` would emit.
#[derive(Clone, Debug)]
pub struct VirtualGrid {
    dim: usize,
    size: Vec<usize>,
    origin: Vec<f64>,
    /// Row-major `dim × dim`, `direction · diag(spacing)`.
    idx_to_phys: Vec<f64>,
}

impl VirtualGrid {
    /// The grid as `(size, origin, index_to_physical)` — everything a device
    /// backend needs to derive a sample's physical point from its index, instead
    /// of being sent the whole precomputed points array.
    #[cfg(feature = "cuda")]
    pub(crate) fn parts(&self) -> (&[usize], &[f64], &[f64]) {
        (&self.size, &self.origin, &self.idx_to_phys)
    }

    pub(crate) fn new(
        dim: usize,
        size: Vec<usize>,
        origin: Vec<f64>,
        idx_to_phys: Vec<f64>,
    ) -> Self {
        Self {
            dim,
            size,
            origin,
            idx_to_phys,
        }
    }

    fn num_pixels(&self) -> usize {
        self.size.iter().product()
    }

    /// `origin + idx_to_phys · index`.
    fn point(&self, index: &[usize], out: &mut Vec<f64>) {
        for r in 0..self.dim {
            let row = &self.idx_to_phys[r * self.dim..(r + 1) * self.dim];
            let mut v = self.origin[r];
            for (c, &m) in row.iter().enumerate() {
                v += m * index[c] as f64;
            }
            out.push(v);
        }
    }

    /// ITK `GetVirtualDomainCentralIndex`
    /// (`itkRegistrationParameterScalesEstimator.hxx:429-445`):
    /// `(lower + upper) / 2.0` truncated to an integer. The region's lower
    /// index is always zero here, so this is `(size − 1) / 2`.
    pub(crate) fn central_index(&self) -> Vec<usize> {
        self.size.iter().map(|&s| s.saturating_sub(1) / 2).collect()
    }

    /// Every point of the axis-aligned box `[lower, upper]` (inclusive), first
    /// index fastest — ITK's `SampleVirtualDomainWithRegion`.
    fn region_points(&self, lower: &[usize], upper: &[usize]) -> Vec<f64> {
        let counts: Vec<usize> = (0..self.dim).map(|d| upper[d] + 1 - lower[d]).collect();
        let total: usize = counts.iter().product();
        let mut points = Vec::with_capacity(total * self.dim);
        let mut index = lower.to_vec();
        for _ in 0..total {
            self.point(&index, &mut points);
            for d in 0..self.dim {
                index[d] += 1;
                if index[d] <= upper[d] {
                    break;
                }
                index[d] = lower[d];
            }
        }
        points
    }

    /// ITK `SampleVirtualDomainFully`.
    fn full_domain_points(&self) -> Vec<f64> {
        let lower = vec![0usize; self.dim];
        let upper: Vec<usize> = self.size.iter().map(|&s| s - 1).collect();
        self.region_points(&lower, &upper)
    }

    /// ITK `SampleVirtualDomainWithCentralRegion` +
    /// `GetVirtualDomainCentralRegion`
    /// (`itkRegistrationParameterScalesEstimator.hxx:447-484`): the domain
    /// clipped to `centralIndex ± radius` on every axis.
    pub(crate) fn central_region_points(&self, radius: usize) -> Vec<f64> {
        let central = self.central_index();
        let lower: Vec<usize> = (0..self.dim)
            .map(|d| central[d].saturating_sub(radius))
            .collect();
        let upper: Vec<usize> = (0..self.dim)
            .map(|d| (central[d] + radius).min(self.size[d] - 1))
            .collect();
        self.region_points(&lower, &upper)
    }

    /// ITK `SampleVirtualDomainWithCorners`
    /// (`itkRegistrationParameterScalesEstimator.hxx:512-538`): the `2ᵈ`
    /// vertices, corner `i` taking axis `d`'s upper index when bit `d` of `i`
    /// is set.
    fn corner_points(&self) -> Vec<f64> {
        let corner_number = 1usize << self.dim;
        let mut points = Vec::with_capacity(corner_number * self.dim);
        let mut index = vec![0usize; self.dim];
        for i in 0..corner_number {
            for (d, (idx, &size)) in index.iter_mut().zip(self.size.iter()).enumerate() {
                let bit = usize::from(i & (1 << d) != 0);
                *idx = bit * (size - 1);
            }
            self.point(&index, &mut points);
        }
        points
    }

    /// ITK `SampleVirtualDomainRandomly`
    /// (`itkRegistrationParameterScalesEstimator.hxx:539-583`): `total` draws
    /// when the domain is small, else `SizeOfSmallDomain · (1 + ln(total /
    /// SizeOfSmallDomain))` draws, capped at `total`. Uniform *with*
    /// replacement, as ITK's random iterator is.
    fn random_points(&self) -> Vec<f64> {
        let total = self.num_pixels();
        let count = if total <= SIZE_OF_SMALL_DOMAIN {
            total
        } else {
            let ratio = 1.0 + ((total as f64) / (SIZE_OF_SMALL_DOMAIN as f64)).ln();
            ((SIZE_OF_SMALL_DOMAIN as f64 * ratio) as usize).min(total)
        };

        let mut rng = SplitMix64::new(RANDOM_SAMPLING_SEED);
        let mut points = Vec::with_capacity(count * self.dim);
        let mut index = vec![0usize; self.dim];
        for _ in 0..count {
            let mut flat = rng.next_below(total.max(1));
            for (idx, &size) in index.iter_mut().zip(self.size.iter()) {
                *idx = flat % size;
                flat /= size;
            }
            self.point(&index, &mut points);
        }
        points
    }
}

/// The sampling strategy ITK selects for one estimation call
/// (`SetScalesSamplingStrategy` / `SetStepScaleSamplingStrategy`). The
/// `VirtualDomainPointSetSampling` branch is unreachable here: it is chosen
/// only when a virtual-domain *point set* is attached to the estimator, which
/// is a point-set-metric feature this crate does not have.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Sampling {
    Corner,
    CentralRegion(usize),
    FullDomain,
    Random,
}

impl Sampling {
    fn points(self, grid: &VirtualGrid) -> Vec<f64> {
        match self {
            Self::Corner => grid.corner_points(),
            Self::CentralRegion(r) => grid.central_region_points(r),
            Self::FullDomain => grid.full_domain_points(),
            Self::Random => grid.random_points(),
        }
    }
}

/// ITK `SetScalesSamplingStrategy`.
fn scales_sampling(transform: &dyn ParametricTransform, radius: usize) -> Sampling {
    if transform.has_local_support() {
        Sampling::CentralRegion(radius)
    } else if transform.is_linear() {
        Sampling::Corner
    } else {
        Sampling::Random
    }
}

/// ITK `SetStepScaleSamplingStrategy`. Only the local-support row differs from
/// [`scales_sampling`]: a step scale must see the whole domain, since each
/// pixel's parameter block only moves its own neighborhood — which is also why
/// `central_region_radius` is absent here.
fn step_scale_sampling(transform: &dyn ParametricTransform) -> Sampling {
    if transform.has_local_support() {
        Sampling::FullDomain
    } else if transform.is_linear() {
        Sampling::Corner
    } else {
        Sampling::Random
    }
}

/// The transform-Jacobian data a [`SampleSet`] holds, chosen by
/// [`ParametricTransform::has_local_support`]. This is the structural guard for
/// the local-support fast path: a local-support transform's state (`Local`) has
/// no field that could hold a `dim × nparams` array, so a future method cannot
/// silently fall back to the dense probe it would be intractable to build.
enum JacobianStore {
    /// Global transform: the per-sample `dim × nparams` Jacobian, evaluated
    /// once and concatenated row-major over samples.
    Dense(Vec<f64>),
    /// Local-support transform (`DisplacementFieldTransform`): one
    /// `(parameter-block offset, dim × numberOfLocalParameters local Jacobian)`
    /// pair per sample that falls inside the transform's domain. Samples
    /// outside are simply absent — ITK's local support means they contribute
    /// no shift no matter what the parameters do. Size is `O(nsamples)`,
    /// independent of `nparams`.
    Local {
        num_local: usize,
        blocks: Vec<(usize, Vec<f64>)>,
    },
}

/// One of the estimator's two sample sets (scales, step scale), with the
/// transform Jacobian evaluated once at every point.
///
/// The Jacobians are cached at construction because they do not depend on the
/// parameters for the transforms registration optimizes: `T_{p+Δ}(x) − T_p(x) =
/// J(x)·Δ` exactly, so ITK's "apply Δ, re-transform, restore Δ" dance
/// (`ComputeSampleShifts`) is a Jacobian product this can take directly.
struct SampleSet {
    /// `m_SamplePoints.size()` — the divisor the Jacobian estimator averages
    /// by, which counts *every* sample, including local-support samples that
    /// fell outside the field and so carry no block.
    n: usize,
    jac: JacobianStore,
}

impl SampleSet {
    /// Evaluate `transform`'s Jacobian at every point of `points` (row-major
    /// `N × dim`). `metric` is an optional row-major `dim × dim` matrix
    /// left-multiplied into each Jacobian: `None` measures shifts in physical
    /// space ([`ScalesEstimatorKind::PhysicalShift`], [`ScalesEstimatorKind::Jacobian`]),
    /// `Some(phys_to_index)` measures them in the moving image's
    /// continuous-index units ([`ScalesEstimatorKind::IndexShift`]).
    ///
    /// The index-shift substitution is exact, not an approximation: a
    /// continuous index is an affine function `M·(y − origin)` of the physical
    /// point `y`, so `index(T_{p+Δ}(x)) − index(T_p(x)) = M·(T_{p+Δ}(x) −
    /// T_p(x)) = (M·J(x))·Δ`. The origin cancels, which is why ITK's
    /// `TransformPointToContinuousIndex` round trip reduces to a matrix product
    /// here.
    fn new(
        points: &[f64],
        dim: usize,
        transform: &dyn ParametricTransform,
        metric: Option<&[f64]>,
    ) -> Self {
        let nparams = transform.number_of_parameters();
        let n = points.len().checked_div(dim).unwrap_or(0);

        let jac = if transform.has_local_support() {
            let num_local = transform.number_of_local_parameters();
            let mut blocks = Vec::new();
            for s in 0..n {
                let p = &points[s * dim..(s + 1) * dim];
                if let Some((offset, block)) = local_support_block(transform, p) {
                    blocks.push((offset, apply_metric(&block, dim, num_local, metric)));
                }
            }
            JacobianStore::Local { num_local, blocks }
        } else {
            let stride = dim * nparams;
            let mut jacobians = vec![0.0; n * stride];
            for s in 0..n {
                let p = &points[s * dim..(s + 1) * dim];
                let j = transform.jacobian_wrt_parameters(p);
                jacobians[s * stride..(s + 1) * stride]
                    .copy_from_slice(&apply_metric(&j, dim, nparams, metric));
            }
            JacobianStore::Dense(jacobians)
        };

        Self { n, jac }
    }
}

/// `metric · block` for a row-major `dim × cols` block and a row-major
/// `dim × dim` `metric`. `None` is the identity.
fn apply_metric(block: &[f64], dim: usize, cols: usize, metric: Option<&[f64]>) -> Vec<f64> {
    let Some(m) = metric else {
        return block.to_vec();
    };
    let mut out = vec![0.0; dim * cols];
    for r in 0..dim {
        for c in 0..cols {
            let mut v = 0.0;
            for d in 0..dim {
                v += m[r * dim + d] * block[d * cols + c];
            }
            out[r * cols + c] = v;
        }
    }
    out
}

/// Squared euclidean norm of `jac · delta` for a row-major `dim × cols` block.
fn squared_shift(jac: &[f64], dim: usize, cols: usize, delta: &[f64]) -> f64 {
    let mut sq = 0.0;
    for r in 0..dim {
        let row = &jac[r * cols..(r + 1) * cols];
        let dot: f64 = row.iter().zip(delta.iter()).map(|(&j, &x)| j * x).sum();
        sq += dot * dot;
    }
    sq
}

/// Optimizer parameter-scale and learning-rate estimator: ITK's
/// `RegistrationParameterScalesEstimator` hierarchy, selected by
/// [`ScalesEstimatorKind`].
///
/// Both of ITK's sample sets (one for `EstimateScales`, one for
/// `EstimateStepScale`) are drawn and their Jacobians cached at construction,
/// so [`estimate_learning_rate`](Self::estimate_learning_rate) stays cheap
/// enough to call every iteration.
pub struct ScalesEstimator {
    kind: ScalesEstimatorKind,
    dim: usize,
    nparams: usize,
    /// Maximum physical step size (ITK `EstimateMaximumStepSize`: the minimum
    /// virtual spacing).
    max_step_size: f64,
    scales: SampleSet,
    step: SampleSet,
    /// For a local-support transform, the parameter-block offset of the virtual
    /// domain's central index — the block ITK probes in `EstimateScales`
    /// (`ComputeParameterOffsetFromVirtualIndex`). `None` for a global
    /// transform, and also when the central index falls outside the transform's
    /// own field.
    central_offset: Option<usize>,
}

impl ScalesEstimator {
    /// Build the estimator for `transform` over the virtual domain `grid`.
    ///
    /// `moving_phys_to_index` is the moving image's row-major `dim × dim`
    /// physical-to-continuous-index matrix, used only by
    /// [`ScalesEstimatorKind::IndexShift`]. `max_step_size` is the minimum
    /// virtual spacing.
    pub fn new(
        grid: &VirtualGrid,
        transform: &dyn ParametricTransform,
        moving_phys_to_index: &[f64],
        max_step_size: f64,
        kind: ScalesEstimatorKind,
    ) -> Self {
        let dim = grid.dim;
        let radius = kind.central_region_radius();
        let metric = match kind {
            ScalesEstimatorKind::IndexShift { .. } => Some(moving_phys_to_index),
            ScalesEstimatorKind::PhysicalShift { .. } | ScalesEstimatorKind::Jacobian { .. } => {
                None
            }
        };

        let scales_points = scales_sampling(transform, radius).points(grid);
        let step_points = step_scale_sampling(transform).points(grid);

        let central_offset = transform.has_local_support().then(|| {
            let mut central_point = Vec::with_capacity(dim);
            grid.point(&grid.central_index(), &mut central_point);
            local_support_block(transform, &central_point).map(|(offset, _)| offset)
        });

        Self {
            kind,
            dim,
            nparams: transform.number_of_parameters(),
            max_step_size,
            scales: SampleSet::new(&scales_points, dim, transform, metric),
            step: SampleSet::new(&step_points, dim, transform, metric),
            central_offset: central_offset.flatten(),
        }
    }

    /// Estimate per-parameter scales (ITK `EstimateScales`).
    ///
    /// The returned vector is always `number_of_parameters()` long; for a
    /// local-support transform it is the `numberOfLocalParameters()`-long array
    /// ITK returns, tiled across every parameter block (see the [module
    /// docs](self)).
    pub fn estimate_scales(&self) -> Vec<f64> {
        match self.kind {
            ScalesEstimatorKind::Jacobian { .. } => self.jacobian_scales(),
            ScalesEstimatorKind::PhysicalShift { .. } | ScalesEstimatorKind::IndexShift { .. } => {
                self.shift_scales()
            }
        }
    }

    /// `RegistrationParameterScalesFromJacobian::EstimateScales`
    /// (`.hxx:26-61`): the mean over samples of each Jacobian column's squared
    /// norm. With no samples ITK leaves the array at its `Fill(1.0)` default.
    fn jacobian_scales(&self) -> Vec<f64> {
        if self.scales.n == 0 {
            return vec![1.0; self.nparams];
        }
        match &self.scales.jac {
            JacobianStore::Dense(jacobians) => {
                let np = self.nparams;
                let mut norms = vec![0.0; np];
                for s in 0..self.scales.n {
                    let jac = &jacobians[s * self.dim * np..(s + 1) * self.dim * np];
                    for p in 0..np {
                        for d in 0..self.dim {
                            let v = jac[d * np + p];
                            norms[p] += v * v;
                        }
                    }
                }
                norms.iter().map(|&x| x / self.scales.n as f64).collect()
            }
            JacobianStore::Local { num_local, blocks } => {
                let mut norms = vec![0.0; *num_local];
                for (_, jac) in blocks {
                    for p in 0..*num_local {
                        for d in 0..self.dim {
                            let v = jac[d * num_local + p];
                            norms[p] += v * v;
                        }
                    }
                }
                let local: Vec<f64> = norms.iter().map(|&x| x / self.scales.n as f64).collect();
                self.broadcast(&local)
            }
        }
    }

    /// `RegistrationParameterScalesFromShiftBase::EstimateScales`
    /// (`.hxx:32-119`), shared by the physical-shift and index-shift
    /// estimators — they differ only in the space the shift is measured in,
    /// which is already baked into the cached Jacobians.
    fn shift_scales(&self) -> Vec<f64> {
        let d = self.kind.small_parameter_variation();
        let eps = f64::EPSILON;

        // maxShiftᵢ = maxₓ ‖J(x) · δeᵢ‖ = δ · maxₓ ‖J(x) column i‖. For a
        // local-support transform only the central block responds to a probe at
        // the central offset, since the blocks are disjoint per pixel.
        let (num_local_para, shifts) = match &self.scales.jac {
            JacobianStore::Dense(jacobians) => {
                let np = self.nparams;
                let mut shifts = vec![0.0f64; np];
                for s in 0..self.scales.n {
                    let jac = &jacobians[s * self.dim * np..(s + 1) * self.dim * np];
                    for (p, shift) in shifts.iter_mut().enumerate() {
                        let mut sq = 0.0;
                        for r in 0..self.dim {
                            let v = jac[r * np + p];
                            sq += v * v;
                        }
                        *shift = shift.max(d * sq.sqrt());
                    }
                }
                (np, shifts)
            }
            JacobianStore::Local { num_local, blocks } => {
                let mut shifts = vec![0.0f64; *num_local];
                let central = self
                    .central_offset
                    .and_then(|off| blocks.iter().find(|(o, _)| *o == off));
                if let Some((_, jac)) = central {
                    for (p, shift) in shifts.iter_mut().enumerate() {
                        let mut sq = 0.0;
                        for r in 0..self.dim {
                            let v = jac[r * num_local + p];
                            sq += v * v;
                        }
                        *shift = d * sq.sqrt();
                    }
                }
                (*num_local, shifts)
            }
        };

        let min_nonzero = shifts
            .iter()
            .copied()
            .filter(|&s| s > eps)
            .fold(f64::INFINITY, f64::min);

        if !min_nonzero.is_finite() {
            // ITK: "Variation in any parameter won't change a voxel position.
            // The default scales (1.0) are used to avoid division-by-zero."
            return vec![1.0; self.nparams];
        }

        let inv_d2 = 1.0 / (d * d);
        let local: Vec<f64> = shifts
            .iter()
            .map(|&ms| {
                let base = if ms <= eps {
                    min_nonzero * min_nonzero
                } else {
                    ms * ms
                };
                base * inv_d2
            })
            .collect();

        debug_assert_eq!(local.len(), num_local_para);
        self.broadcast(&local)
    }

    /// Tile a `numberOfLocalParameters()`-long array across every parameter
    /// block (a no-op for a global transform, where the array is already
    /// `nparams` long). See the [module-level wrinkle note](self).
    fn broadcast(&self, local: &[f64]) -> Vec<f64> {
        let mut out = vec![1.0; self.nparams];
        for chunk in out.chunks_mut(local.len()) {
            chunk.copy_from_slice(&local[..chunk.len()]);
        }
        out
    }

    /// Estimate the shift per unit `step` (ITK `EstimateStepScale`). `step` has
    /// length `nparams`.
    pub fn estimate_step_scale(&self, step: &[f64]) -> f64 {
        match self.kind {
            ScalesEstimatorKind::Jacobian { .. } => self.jacobian_step_scale(step),
            ScalesEstimatorKind::PhysicalShift { .. } | ScalesEstimatorKind::IndexShift { .. } => {
                self.shift_step_scale(step)
            }
        }
    }

    /// `RegistrationParameterScalesFromJacobian::EstimateStepScale`
    /// (`.hxx:63-83` + `ComputeSampleStepScales`, `.hxx:117-172`): the **mean**
    /// of `‖J(x)·step‖` over the samples, with no linearization — the Jacobian
    /// product is already linear in `step`.
    fn jacobian_step_scale(&self, step: &[f64]) -> f64 {
        if self.step.n == 0 {
            return 0.0;
        }
        let sum: f64 = match &self.step.jac {
            JacobianStore::Dense(jacobians) => {
                let np = self.nparams;
                (0..self.step.n)
                    .map(|s| {
                        let jac = &jacobians[s * self.dim * np..(s + 1) * self.dim * np];
                        squared_shift(jac, self.dim, np, step).sqrt()
                    })
                    .sum()
            }
            JacobianStore::Local { num_local, blocks } => blocks
                .iter()
                .map(|(offset, jac)| {
                    let local_step = &step[*offset..*offset + num_local];
                    squared_shift(jac, self.dim, *num_local, local_step).sqrt()
                })
                .sum(),
        };
        sum / self.step.n as f64
    }

    /// `RegistrationParameterScalesFromShiftBase::EstimateStepScale`
    /// (`.hxx:124-155`): the **max** shift over the samples. A local-support
    /// transform applies `step` as-is (line 132-135); a global one linearizes
    /// around a scaled-down step, since ITK's shift is nonlinear in `step` in
    /// general.
    fn shift_step_scale(&self, step: &[f64]) -> f64 {
        match &self.step.jac {
            JacobianStore::Local { .. } => self.max_shift(step),
            JacobianStore::Dense(_) => {
                let max_step = step.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
                if max_step <= f64::EPSILON {
                    return 0.0;
                }
                let factor = self.kind.small_parameter_variation() / max_step;
                let small: Vec<f64> = step.iter().map(|&v| v * factor).collect();
                self.max_shift(&small) / factor
            }
        }
    }

    /// ITK `ComputeMaximumVoxelShift`: `maxₓ ‖J(x)·delta‖` over the step-scale
    /// samples.
    fn max_shift(&self, delta: &[f64]) -> f64 {
        let max_sq = match &self.step.jac {
            JacobianStore::Dense(jacobians) => {
                let np = self.nparams;
                (0..self.step.n)
                    .map(|s| {
                        let jac = &jacobians[s * self.dim * np..(s + 1) * self.dim * np];
                        squared_shift(jac, self.dim, np, delta)
                    })
                    .fold(0.0f64, f64::max)
            }
            JacobianStore::Local { num_local, blocks } => blocks
                .iter()
                .map(|(offset, jac)| {
                    let local_delta = &delta[*offset..*offset + num_local];
                    squared_shift(jac, self.dim, *num_local, local_delta)
                })
                .fold(0.0f64, f64::max),
        };
        max_sq.sqrt()
    }

    /// Estimate the learning rate for a step along `modified_gradient` (the
    /// gradient after weights and scales, `g·w ⊘ s`): the rate that moves
    /// samples by at most `max_step_size`. Returns `1.0` when the step scale is
    /// ~0 — ITK's `GradientDescentOptimizerv4::EstimateLearningRate` guard.
    pub fn estimate_learning_rate(&self, modified_gradient: &[f64]) -> f64 {
        let step_scale = self.estimate_step_scale(modified_gradient);
        if step_scale <= f64::EPSILON {
            1.0
        } else {
            self.max_step_size / step_scale
        }
    }

    /// The maximum physical step size (minimum virtual spacing) — ITK
    /// `EstimateMaximumStepSize`.
    pub fn max_step_size(&self) -> f64 {
        self.max_step_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_transform::{AffineTransform, DisplacementFieldTransform, TranslationTransform};

    fn identity_dir(dim: usize) -> Vec<f64> {
        let mut d = vec![0.0; dim * dim];
        for i in 0..dim {
            d[i * dim + i] = 1.0;
        }
        d
    }

    /// A `n × n` unit-spacing grid at the origin: index `i` is physical `i`.
    fn grid(n: usize) -> VirtualGrid {
        VirtualGrid::new(2, vec![n, n], vec![0.0, 0.0], identity_dir(2))
    }

    fn grid_3d(n: usize) -> VirtualGrid {
        VirtualGrid::new(3, vec![n, n, n], vec![0.0; 3], identity_dir(3))
    }

    fn physical(radius: usize) -> ScalesEstimatorKind {
        ScalesEstimatorKind::PhysicalShift {
            central_region_radius: radius,
            small_parameter_variation: DEFAULT_SMALL_PARAMETER_VARIATION,
        }
    }

    fn index_shift(radius: usize) -> ScalesEstimatorKind {
        ScalesEstimatorKind::IndexShift {
            central_region_radius: radius,
            small_parameter_variation: DEFAULT_SMALL_PARAMETER_VARIATION,
        }
    }

    fn jacobian(radius: usize) -> ScalesEstimatorKind {
        ScalesEstimatorKind::Jacobian {
            central_region_radius: radius,
        }
    }

    /// `AffineTransform::new(dim, matrix, translation, center)`, identity
    /// matrix, no translation.
    fn affine_at(center: [f64; 2]) -> AffineTransform {
        AffineTransform::new(2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], center.to_vec())
    }

    // ---- VirtualGrid sampling -------------------------------------------

    #[test]
    fn central_index_is_the_truncated_midpoint_of_the_index_range() {
        // ITK: centralIndex[d] = (IndexValueType)((lower + upper) / 2.0), lower
        // = 0. A 5-wide axis (0..=4) centers on 2; a 6-wide axis (0..=5)
        // truncates 2.5 down to 2.
        assert_eq!(grid(5).central_index(), vec![2, 2]);
        assert_eq!(
            VirtualGrid::new(2, vec![6, 7], vec![0.0, 0.0], identity_dir(2)).central_index(),
            vec![2, 3]
        );
    }

    #[test]
    fn central_region_is_the_domain_clipped_to_the_radius_box() {
        // 11×11 grid, central index (5,5). Radius 2 clips to [3,7]² — 25
        // points, each visited once, first index fastest.
        let pts = grid(11).central_region_points(2);
        assert_eq!(pts.len(), 25 * 2);
        assert_eq!(&pts[0..2], &[3.0, 3.0]);
        assert_eq!(&pts[2..4], &[4.0, 3.0]);
        assert_eq!(&pts[48..50], &[7.0, 7.0]);
    }

    #[test]
    fn central_region_radius_larger_than_the_domain_clips_to_the_whole_domain() {
        // ITK clips lower/upper against the region's own bounds, so radius 5 on
        // a 5×5 grid selects all 25 pixels, not a 11×11 box.
        let pts = grid(5).central_region_points(DEFAULT_CENTRAL_REGION_RADIUS);
        assert_eq!(pts.len(), 25 * 2);
        assert_eq!(&pts[0..2], &[0.0, 0.0]);
        assert_eq!(&pts[48..50], &[4.0, 4.0]);
    }

    #[test]
    fn central_region_radius_selects_a_smaller_box_than_the_full_domain() {
        // The plumbing test for `central_region_radius`: on a 21×21 domain,
        // radius 1 selects 3×3 = 9 points and radius 3 selects 7×7 = 49, while
        // the full domain has 441. (The radius is nevertheless unobservable in
        // every *scale* this crate can produce — see the module docs and
        // `central_region_radius_does_not_change_a_displacement_fields_unit_scales`.)
        assert_eq!(grid(21).central_region_points(1).len(), 9 * 2);
        assert_eq!(grid(21).central_region_points(3).len(), 49 * 2);
        assert_eq!(grid(21).full_domain_points().len(), 441 * 2);
    }

    #[test]
    fn corner_sampling_visits_every_vertex_once_with_bit_d_selecting_axis_d() {
        // ITK: corner[d] = firstCorner[d] + ((i & (1<<d)) != 0) * (size[d]-1).
        let pts = grid(5).corner_points();
        assert_eq!(pts.len(), 4 * 2);
        assert_eq!(&pts[0..2], &[0.0, 0.0]); // i=0
        assert_eq!(&pts[2..4], &[4.0, 0.0]); // i=1 → bit 0 → x
        assert_eq!(&pts[4..6], &[0.0, 4.0]); // i=2 → bit 1 → y
        assert_eq!(&pts[6..8], &[4.0, 4.0]); // i=3
        assert_eq!(grid_3d(3).corner_points().len(), 8 * 3);
    }

    #[test]
    fn random_sampling_draws_the_whole_domain_when_it_is_small() {
        // total = 25 <= SizeOfSmallDomain = 1000, so ITK sets
        // m_NumberOfRandomSamples = total.
        assert_eq!(grid(5).random_points().len(), 25 * 2);
    }

    #[test]
    fn random_sampling_of_a_large_domain_uses_the_log_scaled_budget() {
        // total = 40³ = 64000 > 1000, so the budget is
        // 1000·(1 + ln(64)) = 1000·5.1589… → truncated to 5158 draws.
        let total = 64_000.0f64;
        let expected = (1000.0 * (1.0 + (total / 1000.0).ln())) as usize;
        assert_eq!(expected, 5158);
        assert_eq!(grid_3d(40).random_points().len(), expected * 3);
    }

    // ---- PhysicalShift ---------------------------------------------------

    #[test]
    fn translation_physical_shift_scales_are_unit() {
        // J = I, so maxShiftᵢ = δ and scaleᵢ = (δ/δ)² = 1.
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let est = ScalesEstimator::new(&grid(41), &t, &identity_dir(2), 1.0, physical(5));
        assert_eq!(est.estimate_scales(), vec![1.0, 1.0]);
    }

    #[test]
    fn affine_physical_shift_scales_are_the_squared_max_offset_from_the_center() {
        // 41×41 domain (0..=40), center (20,20). A matrix parameter mᵣc shifts
        // by δ·|xc − centerc|, maximized over the corners at 20. So
        // scale = (20δ/δ)² = 400; a translation parameter shifts by δ → 1.
        let a = affine_at([20.0, 20.0]);
        let est = ScalesEstimator::new(&grid(41), &a, &identity_dir(2), 1.0, physical(5));
        let scales = est.estimate_scales();
        assert_eq!(scales.len(), 6);
        for s in &scales[0..4] {
            assert!((s - 400.0).abs() < 1e-6, "matrix scale {s} != 400");
        }
        assert!((scales[4] - 1.0).abs() < 1e-9);
        assert!((scales[5] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn learning_rate_bounds_a_translation_step_to_one_voxel() {
        // For a translation, stepScale(g) = ‖g‖, so lr = maxStepSize / ‖g‖.
        // With maxStepSize = 1 and g = (3,4), lr = 1/5 and the first step is
        // ‖lr·g‖ = 1 voxel.
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let est = ScalesEstimator::new(&grid(41), &t, &identity_dir(2), 1.0, physical(5));
        let lr = est.estimate_learning_rate(&[3.0, 4.0]);
        assert!((lr - 0.2).abs() < 1e-9, "lr {lr} != 0.2");
    }

    // ---- IndexShift ------------------------------------------------------

    #[test]
    fn index_shift_scales_divide_the_physical_shift_by_the_moving_spacing() {
        // Moving image spacing (2, 1) with identity direction ⇒ phys_to_index =
        // diag(1/2, 1). A unit translation along x moves the moving-image
        // continuous index by 1/2, along y by 1. So
        //   scale_x = ((δ·½)/δ)² = 0.25,  scale_y = ((δ·1)/δ)² = 1.
        // The physical-shift estimator reports (1, 1) for the same transform,
        // which is what makes this test the index-shift discriminator.
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let phys_to_index = vec![0.5, 0.0, 0.0, 1.0];
        let est = ScalesEstimator::new(&grid(5), &t, &phys_to_index, 1.0, index_shift(5));
        let scales = est.estimate_scales();
        assert!(
            (scales[0] - 0.25).abs() < 1e-12,
            "scale_x {} != 0.25",
            scales[0]
        );
        assert!(
            (scales[1] - 1.0).abs() < 1e-12,
            "scale_y {} != 1",
            scales[1]
        );

        let phys = ScalesEstimator::new(&grid(5), &t, &phys_to_index, 1.0, physical(5));
        assert_eq!(phys.estimate_scales(), vec![1.0, 1.0]);
    }

    #[test]
    fn index_shift_affine_matrix_scale_folds_the_spacing_into_the_extent() {
        // 5×5 domain, center (2,2); corners give |x−2| = |y−2| = 2.
        // phys_to_index = diag(1/2, 1).
        //   m00 (∂Tx/∂m00 = x−cx): physical (δ·2, 0) → index (δ·1, 0) → scale 1.
        //   m01 (∂Tx/∂m01 = y−cy): physical (δ·2, 0) → index (δ·1, 0) → scale 1.
        //   m10 (∂Ty/∂m10 = x−cx): physical (0, δ·2) → index (0, δ·2) → scale 4.
        //   m11:                    physical (0, δ·2) → index (0, δ·2) → scale 4.
        //   tx: index (δ·½, 0) → 0.25.   ty: index (0, δ) → 1.
        let a = affine_at([2.0, 2.0]);
        let phys_to_index = vec![0.5, 0.0, 0.0, 1.0];
        let est = ScalesEstimator::new(&grid(5), &a, &phys_to_index, 1.0, index_shift(5));
        let s = est.estimate_scales();
        let expected = [1.0, 1.0, 4.0, 4.0, 0.25, 1.0];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "scales[{i}] = {got} != {want}");
        }
    }

    // ---- Jacobian --------------------------------------------------------

    #[test]
    fn jacobian_scales_are_the_mean_squared_column_norm_over_the_corners() {
        // 5×5 domain, corners (0,0),(4,0),(0,4),(4,4); affine center (1,1), so
        // (x−1, y−1) ∈ {(−1,−1),(3,−1),(−1,3),(3,3)}.
        //   scale(m00) = mean (x−1)² = (1+9+1+9)/4 = 5.   Likewise m01 uses
        //   (y−1)² = 5, m10 uses (x−1)² = 5, m11 uses (y−1)² = 5.
        //   scale(tx) = scale(ty) = mean 1² = 1.
        // Full-domain sampling would give mean (x−1)² = 3 instead of 5, so this
        // also pins that the Jacobian estimator samples the corners.
        let a = affine_at([1.0, 1.0]);
        let est = ScalesEstimator::new(&grid(5), &a, &identity_dir(2), 1.0, jacobian(5));
        let s = est.estimate_scales();
        let expected = [5.0, 5.0, 5.0, 5.0, 1.0, 1.0];
        for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "scales[{i}] = {got} != {want}");
        }
    }

    #[test]
    fn jacobian_scales_differ_from_physical_shift_scales_mean_versus_max() {
        // Same transform and domain as above. The shift estimator takes the
        // *max* over corners: max|x−1| = 3 ⇒ scale = 3² = 9, against the
        // Jacobian estimator's mean of 5.
        let a = affine_at([1.0, 1.0]);
        let phys = ScalesEstimator::new(&grid(5), &a, &identity_dir(2), 1.0, physical(5));
        assert!((phys.estimate_scales()[0] - 9.0).abs() < 1e-12);

        let jac = ScalesEstimator::new(&grid(5), &a, &identity_dir(2), 1.0, jacobian(5));
        assert!((jac.estimate_scales()[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn jacobian_step_scale_is_the_mean_shift_and_the_shift_estimators_is_the_max() {
        // Affine center (1,1) on a 5×5 domain, step = e(m00). ‖J(x)·step‖ =
        // |x − 1|, which is 1 at two corners and 3 at the other two.
        //   Jacobian:      mean = (1+3+1+3)/4 = 2
        //   PhysicalShift: max  = 3
        let a = affine_at([1.0, 1.0]);
        let mut step = vec![0.0; 6];
        step[0] = 1.0;

        let jac = ScalesEstimator::new(&grid(5), &a, &identity_dir(2), 1.0, jacobian(5));
        assert!((jac.estimate_step_scale(&step) - 2.0).abs() < 1e-12);

        let phys = ScalesEstimator::new(&grid(5), &a, &identity_dir(2), 1.0, physical(5));
        assert!((phys.estimate_step_scale(&step) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn jacobian_translation_scales_are_unit() {
        // J = I at every sample ⇒ every column has squared norm 1 ⇒ mean 1.
        let t = TranslationTransform::new(vec![0.0, 0.0]);
        let est = ScalesEstimator::new(&grid(5), &t, &identity_dir(2), 1.0, jacobian(5));
        assert_eq!(est.estimate_scales(), vec![1.0, 1.0]);
    }

    // ---- central-region radius quirk -------------------------------------

    #[test]
    fn central_region_radius_is_ignored_for_a_linear_transform() {
        // ITK reaches CentralRegionSampling only when the transform has local
        // support; a linear transform always samples the corners, so the radius
        // cannot move any of the three estimators' answers.
        let a = affine_at([1.0, 1.0]);
        let dir = identity_dir(2);
        for kind in [physical(0), index_shift(0), jacobian(0)] {
            let tight = ScalesEstimator::new(&grid(21), &a, &dir, 1.0, kind);
            let wide = match kind {
                ScalesEstimatorKind::PhysicalShift { .. } => physical(9),
                ScalesEstimatorKind::IndexShift { .. } => index_shift(9),
                ScalesEstimatorKind::Jacobian { .. } => jacobian(9),
            };
            let wide = ScalesEstimator::new(&grid(21), &a, &dir, 1.0, wide);
            assert_eq!(
                tight.estimate_scales(),
                wide.estimate_scales(),
                "radius changed {kind:?} scales for a linear transform"
            );
        }
    }

    #[test]
    fn central_region_radius_does_not_change_a_displacement_fields_unit_scales() {
        // The radius *does* select a different sample set here (central region
        // vs. central region), but a displacement field's local Jacobian is the
        // identity at every grid-aligned sample, so every probe shifts by δ and
        // the scales are 1 for any radius.
        let field =
            DisplacementFieldTransform::new(2, &[9, 9], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let g = grid(9);
        let dir = identity_dir(2);
        let tight = ScalesEstimator::new(&g, &field, &dir, 1.0, physical(1)).estimate_scales();
        let wide = ScalesEstimator::new(&g, &field, &dir, 1.0, physical(4)).estimate_scales();
        assert_eq!(tight, wide);
        for (i, &s) in tight.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-9, "scales[{i}] = {s} != 1");
        }
    }

    // ---- local support ---------------------------------------------------

    #[test]
    fn displacement_field_takes_the_local_support_path() {
        let field =
            DisplacementFieldTransform::new(2, &[6, 6], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let est = ScalesEstimator::new(&grid(6), &field, &identity_dir(2), 1.0, physical(5));
        assert!(matches!(est.scales.jac, JacobianStore::Local { .. }));
        assert!(matches!(est.step.jac, JacobianStore::Local { .. }));
    }

    #[test]
    fn displacement_field_scales_are_unit_and_broadcast_to_every_parameter() {
        // ITK's own EstimateScales returns a numberOfLocalParameters()-length
        // unit array for a displacement field (see the module docs); this
        // crate's optimizer has no broadcast support, so estimate_scales tiles
        // that unit value across the full nparams-length vector — every entry
        // is 1.0, matching what a broadcast-aware optimizer would read.
        let field =
            DisplacementFieldTransform::new(2, &[6, 6], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let nparams = field.number_of_parameters();
        let est = ScalesEstimator::new(&grid(6), &field, &identity_dir(2), 1.0, physical(5));
        let scales = est.estimate_scales();
        assert_eq!(scales.len(), nparams);
        for (i, &s) in scales.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-9, "scales[{i}] = {s} != 1");
        }
    }

    #[test]
    fn displacement_field_step_scale_matches_itks_local_formula() {
        // ITK's EstimateStepScale for a local-support transform is exactly
        // ComputeMaximumVoxelShift(step) — no linearization — which for a
        // displacement field (identity local Jacobian, disjoint per-pixel
        // support) is the max over pixels of that pixel's own step-block norm.
        // The step-scale sample set is the *full* domain, so every pixel is
        // seen.
        let field =
            DisplacementFieldTransform::new(2, &[4, 4], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let est = ScalesEstimator::new(&grid(4), &field, &identity_dir(2), 1.0, physical(5));

        let mut step = vec![0.0; field.number_of_parameters()];
        // Pixel (1,1): raster 1 + 1*4 = 5, offset 10. Step (3,4), norm 5.
        step[10] = 3.0;
        step[11] = 4.0;
        // Pixel (2,2): raster 2 + 2*4 = 10, offset 20. Step (1,0), norm 1.
        step[20] = 1.0;

        let step_scale = est.estimate_step_scale(&step);
        assert!(
            (step_scale - 5.0).abs() < 1e-12,
            "step scale {step_scale} != 5 (max pixel-block norm)"
        );
    }

    #[test]
    fn displacement_field_jacobian_step_scale_is_the_mean_over_the_full_domain() {
        // Same field and step as above, but the Jacobian estimator averages:
        // 16 pixels, two of which carry a step (norms 5 and 1), so the mean is
        // 6/16 = 0.375.
        let field =
            DisplacementFieldTransform::new(2, &[4, 4], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let est = ScalesEstimator::new(&grid(4), &field, &identity_dir(2), 1.0, jacobian(5));

        let mut step = vec![0.0; field.number_of_parameters()];
        step[10] = 3.0;
        step[11] = 4.0;
        step[20] = 1.0;

        let got = est.estimate_step_scale(&step);
        assert!(
            (got - 0.375).abs() < 1e-12,
            "mean step scale {got} != 0.375"
        );
    }

    #[test]
    fn local_support_construction_avoids_dense_per_parameter_allocation() {
        // A field with hundreds of thousands of parameters. The dense algorithm
        // would have allocated `n * dim * nparams` floats; the local path must
        // allocate only `O(n)` — one small local-Jacobian block per sample,
        // never touching `nparams`.
        let size = [64usize, 64, 64];
        let field = DisplacementFieldTransform::new(
            3,
            &size,
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            &identity_dir(3),
        )
        .unwrap();
        let nparams = field.number_of_parameters();
        assert!(nparams > 700_000, "test needs a large nparams: {nparams}");

        // Radius 1 keeps the scales sample set at 3³ = 27 points.
        let g = VirtualGrid::new(3, size.to_vec(), vec![0.0; 3], identity_dir(3));
        let est = ScalesEstimator::new(&g, &field, &identity_dir(3), 1.0, physical(1));
        match &est.scales.jac {
            JacobianStore::Local { num_local, blocks } => {
                assert_eq!(*num_local, 3);
                assert_eq!(blocks.len(), 27);
                for (_, jac) in blocks {
                    assert_eq!(
                        jac.len(),
                        3 * num_local,
                        "local block must be dim*num_local, not proportional to nparams"
                    );
                }
            }
            JacobianStore::Dense(_) => {
                panic!("displacement field must take the local-support path")
            }
        }
    }

    #[test]
    fn local_scales_probe_the_central_index_block() {
        // ITK probes deltaParameters at ComputeParameterOffsetFromVirtualIndex(
        // centralIndex, numLocalPara). A 9×9 field's central index is (4,4),
        // raster 4 + 4*9 = 40, so the probed offset is 80.
        let field =
            DisplacementFieldTransform::new(2, &[9, 9], &[0.0, 0.0], &[1.0, 1.0], &identity_dir(2))
                .unwrap();
        let est = ScalesEstimator::new(&grid(9), &field, &identity_dir(2), 1.0, physical(2));
        assert_eq!(est.central_offset, Some(80));
    }
}
