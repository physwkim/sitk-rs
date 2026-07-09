//! `itk::CoherenceEnhancingDiffusionImageFilter` and the
//! `AnisotropicDiffusionLBR` module it sits on: Weickert's coherence-enhancing
//! (CED) and edge-enhancing (EED) diffusion, discretized by Jean-Marie
//! Mirebeau's *lattice basis reduction* (LBR) scheme.
//!
//! The filter integrates `∂ₜu = div(D ∇u)` with Neumann boundary conditions,
//! where the diffusion tensor `D` is rebuilt from the evolving image every few
//! time steps. Four ITK classes stack up, and this module ports all of them:
//!
//! 1. **`StructureTensorImageFilter`** — `S := K_ρ * (∇u_σ ⊗ ∇u_σ)`, the
//!    Gaussian-smoothed outer product of the gradient of a Gaussian-smoothed
//!    image. `σ` is `noise_scale`, `ρ` is `feature_scale`.
//! 2. **`AnisotropicDiffusionLBRImageFilter::DiffusionTensorFunctor`** —
//!    eigendecompose `S`, map its eigenvalues through `EigenValuesTransform`,
//!    and rebuild a tensor on the same eigenvectors.
//! 3. **`CoherenceEnhancingDiffusionImageFilter::EigenValuesTransform`** — the
//!    five transfer functions of [`Enhancement`].
//! 4. **`LinearAnisotropicDiffusionLBRImageFilter`** — the LBR stencil and an
//!    explicit Euler loop.
//!
//! # The LBR stencil (the point of the filter)
//!
//! A naive finite-difference discretization of `div(D ∇u)` loses the maximum
//! principle once `D` is strongly anisotropic: the off-diagonal cross terms go
//! negative and the scheme oscillates. Mirebeau's answer is to pick the stencil
//! *per pixel*, from `D`, using **Selling's algorithm**.
//!
//! A *superbase* of `ℤ^d` is `d+1` integer vectors `b₀ … b_d` with
//! `b₀ + ⋯ + b_d = 0` whose first `d` members form a basis. It is **obtuse**
//! for `D` when `⟨D bᵢ, bⱼ⟩ ≤ 0` for every `i ≠ j`. Selling's algorithm reaches
//! one from the canonical superbase by repeatedly *flipping* the first pair
//! with a positive product — a lattice basis reduction, and the source of the
//! module's name. Once obtuse, `D` decomposes with **non-negative** weights:
//!
//! - **2-D**: `D = Σᵢ₌₀² ρᵢ eᵢ eᵢᵀ`, with `ρᵢ = -⟨D b_{i+1}, b_{i+2}⟩` and
//!   `eᵢ = bᵢ^⊥` (the 90°-rotated superbase vector).
//! - **3-D**: the six offsets are the rows of the cofactor matrix of
//!   `(b₀, b₁, b₂)` and their three pairwise differences; the six weights are
//!   `-⟨D bᵢ, bⱼ⟩` over the six unordered pairs.
//!
//! The scheme is then the non-negative sum
//! `div(D∇u)(x) ≈ Σᵢ ρᵢ (u(x+eᵢ) − 2u(x) + u(x−eᵢ))`, which is monotone, hence
//! stable and maximum-principle-preserving at any anisotropy.
//!
//! ITK folds a factor of two through this: `Stencil` returns
//! `cᵢ = −½⟨D b_j, b_k⟩ = ρᵢ/2`, and `ImageUpdate` then visits each unordered
//! pixel pair *twice* (once from each endpoint, `out[x] += c·in[y]` **and**
//! `out[y] += c·in[x]`), so the assembled matrix carries `cᵢ(x) + cᵢ(y) ≈ ρᵢ`.
//! The halving and the double visit cancel, and the doubled visit is what makes
//! the assembled operator symmetric when `D` varies in space. The diagonal is
//! the row sum of the same accumulation, so every row of the operator sums to
//! zero and a constant image is a fixed point exactly.
//!
//! # Faithfully-reproduced upstream behaviors, rather than "fixed"
//!
//! - **`feature_scale` smooths along axis 0 only.**
//!   `StructureTensorImageFilter::GenerateData` smooths the outer-product image
//!   with `itk::RecursiveGaussianImageFilter`, which is the *single-direction*
//!   `RecursiveSeparableImageFilter` (its `m_Direction` defaults to `0` and the
//!   filter never sets it), not the all-axes
//!   `SmoothingRecursiveGaussianImageFilter`. So `K_ρ` in the structure-tensor
//!   definition is applied along the first image axis and no other. This port
//!   reproduces that. (`noise_scale`'s `K_σ` *is* isotropic: it goes through
//!   `GradientRecursiveGaussianImageFilter`, which does smooth every axis.)
//! - **The class doc's CED/EED formulas are swapped** relative to the code, in
//!   both `itkCoherenceEnhancingDiffusionImageFilter.h` and SimpleITK's yaml.
//!   The code is authoritative and is what this port follows:
//!   [`Enhancement::Ced`] uses `g_CED(μ_max − μᵢ)` and [`Enhancement::Eed`]
//!   uses `g_EED(μᵢ − μ_min)`. See [`Enhancement`] for each formula.
//! - **`Adimensionize` on a constant image divides by zero.** It turns on
//!   `RescaleForUnitMaximumTrace`, whose scaling is `1 / max(trace S)`; a
//!   constant image has `S ≡ 0`, so the scaling is `+∞` and every tensor
//!   becomes `0 · ∞ = NaN`. The `NaN`s then propagate to the diagonal
//!   coefficients, where ITK's `MinimumMaximumImageCalculator` — which seeds
//!   its maximum with `NonpositiveMin()` and keeps it under `value > max`,
//!   a comparison `NaN` always loses — returns `-DBL_MAX`. The resulting step
//!   count is negative, the Euler loop runs zero times, and the input is
//!   returned unchanged. This port reproduces the whole chain, so a constant
//!   image is still a fixed point; see `constant_image_is_returned_unchanged`.
//! - **Selling's algorithm gives up silently after 200 flips**, printing to
//!   `std::cerr` and continuing with a possibly non-obtuse superbase (which
//!   can yield a negative stencil weight). This port also continues, without
//!   the print — a library has no business writing to stderr.
//! - **The eigenvalue sort in `DiffusionTensorFunctor` is dead code.**
//!   `SymmetricSecondRankTensor::ComputeEigenAnalysis` already returns
//!   eigenvalues in ascending order (`SymmetricEigenAnalysis`'s default
//!   `OrderByValue`), so the functor's `std::sort` and its `order` permutation
//!   are the identity. This port relies on the ascending order directly.
//!
//! # Deviations
//!
//! ITK's `TScalar` is `NumericTraits<PixelType>::RealType`, so a `Float32`
//! image runs the entire pipeline — structure tensor, eigenanalysis, Selling
//! decomposition, Euler steps — in single precision. This port computes in
//! `f64` throughout and narrows once, at the output. Bit-exact agreement is out
//! of reach either way: ITK's `SymmetricEigenAnalysis` is a different
//! eigensolver from the cyclic Jacobi in [`crate::linalg`], and the two agree
//! only to solver accuracy.
//!
//! A zero `noise_scale` or `feature_scale` is the identity here, because
//! [`recursive_gaussian`] leaves an axis with `sigma == 0` untouched. ITK
//! instead feeds the zero to `RecursiveGaussianImageFilter::SetUp`, which
//! divides by it. The limit this port takes is the sane one, but it is a
//! deviation, not a reproduction.
//!
//! `pixel_types: RealPixelIDTypeList`: the input must be `Float32` or
//! `Float64`, and the output takes the input's pixel type and geometry. Only
//! 2-D and 3-D are supported — ITK's `Stencil` has no other overload.

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use crate::linalg::{MAX_DIM, Mat, symmetric_eigen};
use crate::recursive_gaussian::{GaussianOrder, recursive_gaussian, recursive_gaussian_with_order};
use sitk_core::{Image, PixelId};

/// `HalfStencilSize` in 3-D, and so the width of the fixed-size stencil
/// arrays. 2-D uses the first three slots.
const MAX_HALF_STENCIL: usize = 6;

/// `constexpr int maxIter = 200` in both `Stencil` overloads.
const SELLING_MAX_ITER: usize = 200;

/// `CoherenceEnhancingDiffusionImageFilter::EnhancementType`: which transfer
/// function maps the structure-tensor eigenvalues `μ₀ ≤ ⋯ ≤ μ_{d-1}` to the
/// diffusion-tensor eigenvalues `λᵢ`.
///
/// Both building blocks are monotone functions of one scalar `s`, sharing
/// `alpha` (the saturation level), `lambda` (the contrast scale) and
/// `exponent` (`m`):
///
/// - `g_CED(s) = α + (1−α)·exp(−(λ/s)^m)` for `s > 0`, and `α` for `s ≤ 0`.
///   Limits: `g_CED(0) = α`, `g_CED(∞) = 1`.
/// - `g_EED(s) = 1 − (1−α)·exp(−(λ/s)^m)` for `s > 0`, and `1` for `s ≤ 0`.
///   Limits: `g_EED(0) = 1`, `g_EED(∞) = α`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enhancement {
    /// Weickert's coherence-enhancing diffusion: `λᵢ = g_CED(μ_max − μᵢ)`.
    /// The most-coherent direction (largest gap from `μ_max`) diffuses at
    /// rate 1, the least-coherent at `α`, so smoothing runs *along* contours.
    Ced,
    /// A variant requiring stronger coherence:
    /// `λᵢ = g_CED((μ_max − μᵢ) / (1 + μᵢ/λ))`. The denominator suppresses
    /// diffusion where `μᵢ` is itself large.
    CCed,
    /// Weickert's edge-enhancing diffusion: `λᵢ = g_EED(μᵢ − μ_min)`. The
    /// direction of least variation diffuses at rate 1 and the gradient
    /// direction at `α`, so smoothing stops *across* contours.
    Eed,
    /// A variant promoting diffusion in at least one direction at each point:
    /// `λᵢ = g_EED(μᵢ)`.
    CEed,
    /// Isotropic tensors, close to Perona-Malik: every `λᵢ = g_EED(μ_max)`.
    Isotropic,
}

/// Parameters of [`coherence_enhancing_diffusion`], defaulting to SimpleITK's
/// `CoherenceEnhancingDiffusionImageFilter.yaml`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoherenceEnhancingDiffusionSettings {
    /// Total time `T` in `∂ₜu = div(D ∇u)`, `0 ≤ t ≤ T`. Non-positive values
    /// return the input unchanged (ITK's `while (remainingTime > 0)`).
    pub diffusion_time: f64,
    /// Contrast scale `λ` of the transfer functions. See [`Enhancement`].
    pub lambda: f64,
    /// Which transfer function to use.
    pub enhancement: Enhancement,
    /// `σ`: the Gaussian scale at which the gradient is measured, in physical
    /// units. Isotropic (all axes).
    pub noise_scale: f64,
    /// `ρ`: the Gaussian scale at which the gradient outer product is
    /// averaged, in physical units. **Applied along axis 0 only**, reproducing
    /// ITK's use of the single-direction `RecursiveGaussianImageFilter`.
    pub feature_scale: f64,
    /// Exponent `m` of the transfer functions. See [`Enhancement`].
    pub exponent: f64,
    /// Saturation level `α` of the transfer functions, the diffusivity the
    /// suppressed direction retains. See [`Enhancement`].
    pub alpha: f64,
    /// The explicit time step, as a fraction of the largest stable one
    /// (`1 / max diagonal coefficient`). Must lie in `(0, 1]`.
    pub ratio_to_max_stable_time_step: f64,
    /// How many explicit steps may run before the diffusion tensors are
    /// recomputed from the evolved image. Must be positive.
    pub max_time_steps_between_tensor_updates: u8,
    /// Rescale the spacing so its minimum is 1, and rescale the structure
    /// tensors so their maximum trace is 1. Makes `lambda`, `noise_scale`,
    /// `feature_scale` and `diffusion_time` dimensionless.
    pub adimensionize: bool,
}

impl Default for CoherenceEnhancingDiffusionSettings {
    /// SimpleITK's yaml defaults: `DiffusionTime = 1.0`, `Lambda = 0.05`,
    /// `Enhancement = CED`, `NoiseScale = 0.5`, `FeatureScale = 2.0`,
    /// `Exponent = 2.0`, `Alpha = 0.01`, `RatioToMaxStableTimeStep = 0.7`,
    /// `MaxTimeStepsBetweenTensorUpdates = 5`, `Adimensionize = true`.
    fn default() -> Self {
        Self {
            diffusion_time: 1.0,
            lambda: 0.05,
            enhancement: Enhancement::Ced,
            noise_scale: 0.5,
            feature_scale: 2.0,
            exponent: 2.0,
            alpha: 0.01,
            ratio_to_max_stable_time_step: 0.7,
            max_time_steps_between_tensor_updates: 5,
            adimensionize: true,
        }
    }
}

// ---- eigenvalue transfer functions ----------------------------------------

/// `g_CED(s) = s <= 0 ? α : α + (1−α)·exp(−(λ/s)^m)`.
fn g_ced(s: f64, lambda: f64, exponent: f64, alpha: f64) -> f64 {
    if s <= 0.0 {
        alpha
    } else {
        alpha + (1.0 - alpha) * (-((lambda / s).powf(exponent))).exp()
    }
}

/// `g_EED(s) = s <= 0 ? 1 : 1 − (1−α)·exp(−(λ/s)^m)`.
fn g_eed(s: f64, lambda: f64, exponent: f64, alpha: f64) -> f64 {
    if s <= 0.0 {
        1.0
    } else {
        1.0 - (1.0 - alpha) * (-((lambda / s).powf(exponent))).exp()
    }
}

/// `CoherenceEnhancingDiffusionImageFilter::EigenValuesTransform`. `ev0` holds
/// the structure-tensor eigenvalues in ascending order, so `ev0[0]` is `μ_min`
/// and `ev0[dim-1]` is `μ_max`.
fn eigen_values_transform(
    ev0: &[f64; MAX_DIM],
    dim: usize,
    s: &CoherenceEnhancingDiffusionSettings,
) -> [f64; MAX_DIM] {
    let (lambda, m, alpha) = (s.lambda, s.exponent, s.alpha);
    let ev_min = ev0[0];
    let ev_max = ev0[dim - 1];

    let mut ev = [0.0; MAX_DIM];
    for (i, out) in ev.iter_mut().enumerate().take(dim) {
        *out = match s.enhancement {
            Enhancement::Ced => g_ced(ev_max - ev0[i], lambda, m, alpha),
            Enhancement::CCed => g_ced(
                (ev_max - ev0[i]) / (1.0 + ev0[i] / lambda),
                lambda,
                m,
                alpha,
            ),
            Enhancement::Eed => g_eed(ev0[i] - ev_min, lambda, m, alpha),
            Enhancement::CEed => g_eed(ev0[i], lambda, m, alpha),
            Enhancement::Isotropic => g_eed(ev_max, lambda, m, alpha),
        };
    }
    ev
}

// ---- Selling's algorithm and the LBR stencil -------------------------------

/// `LinearAnisotropicDiffusionLBRImageFilter::ScalarProduct`: `uᵀ M v` for a
/// symmetric `M`, summed in ITK's order (diagonal first, then the off-diagonal
/// pairs).
fn scalar_product(m: &Mat, u: &[f64; MAX_DIM], v: &[f64; MAX_DIM], dim: usize) -> f64 {
    let mut result = 0.0;
    for i in 0..dim {
        result += m[i][i] * u[i] * v[i];
    }
    for i in 0..dim {
        for j in i + 1..dim {
            result += m[i][j] * (u[i] * v[j] + u[j] * v[i]);
        }
    }
    result
}

/// One pixel's stencil: `half` offsets `eᵢ` with weights `cᵢ = ρᵢ/2`. Each
/// offset stands for the *pair* `x ± eᵢ`, hence "half".
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stencil {
    offsets: [[i64; MAX_DIM]; MAX_HALF_STENCIL],
    coeffs: [f64; MAX_HALF_STENCIL],
}

/// `StencilFunctor::Stencil(Dispatch<2>, ...)`.
fn stencil_2d(d: &Mat) -> Stencil {
    // Canonical superbase of Z^2: e0, e1, -(e0 + e1).
    let mut sb = [[0.0; MAX_DIM]; 4];
    sb[0][0] = 1.0;
    sb[1][1] = 1.0;
    sb[2][0] = -1.0;
    sb[2][1] = -1.0;

    // Selling's algorithm: flip the first pair with a positive product.
    for _ in 0..SELLING_MAX_ITER {
        let mut same = true;
        'outer: for i in 1..=2 {
            for j in 0..i {
                if scalar_product(d, &sb[i], &sb[j], 2) > 0.0 {
                    let (u, v) = (sb[i], sb[j]);
                    for k in 0..2 {
                        sb[0][k] = v[k] - u[k];
                        sb[1][k] = u[k];
                        sb[2][k] = -v[k];
                    }
                    same = false;
                    break 'outer;
                }
            }
        }
        if same {
            break;
        }
    }

    let mut stencil = Stencil {
        offsets: [[0; MAX_DIM]; MAX_HALF_STENCIL],
        coeffs: [0.0; MAX_HALF_STENCIL],
    };
    for i in 0..3 {
        stencil.coeffs[i] = -0.5 * scalar_product(d, &sb[(i + 1) % 3], &sb[(i + 2) % 3], 2);
        // The 90-degree rotation e_i = b_i^perp.
        stencil.offsets[i][0] = -sb[i][1] as i64;
        stencil.offsets[i][1] = sb[i][0] as i64;
    }
    stencil
}

/// `StencilFunctor::Stencil(Dispatch<3>, ...)`.
fn stencil_3d(d: &Mat) -> Stencil {
    // Canonical superbase of Z^3: e0, e1, e2, -(e0 + e1 + e2).
    let mut sb = [[0.0; MAX_DIM]; 4];
    sb[0][0] = 1.0;
    sb[1][1] = 1.0;
    sb[2][2] = 1.0;
    sb[3] = [-1.0, -1.0, -1.0];

    for _ in 0..SELLING_MAX_ITER {
        let mut same = true;
        'outer: for i in 1..=3 {
            for j in 0..i {
                if scalar_product(d, &sb[i], &sb[j], 3) > 0.0 {
                    let (u, v) = (sb[i], sb[j]);
                    // The two superbase vectors other than i and j, each
                    // shifted by u, land in slots 0 and 1 -- in ITK's `k`
                    // order, which the snapshot preserves.
                    let old = sb;
                    let mut l = 0;
                    for (k, ob) in old.iter().enumerate() {
                        if k != i && k != j {
                            for c in 0..3 {
                                sb[l][c] = ob[c] + u[c];
                            }
                            l += 1;
                        }
                    }
                    for c in 0..3 {
                        sb[2][c] = -u[c];
                    }
                    sb[3] = v;
                    same = false;
                    break 'outer;
                }
            }
        }
        if same {
            break;
        }
    }

    // Weights over the six unordered superbase pairs.
    let mut weights = [[0.0; 4]; 4];
    for i in 1..4 {
        for j in 0..i {
            let w = -0.5 * scalar_product(d, &sb[i], &sb[j], 3);
            weights[i][j] = w;
            weights[j][i] = w;
        }
    }

    let mut stencil = Stencil {
        offsets: [[0; MAX_DIM]; MAX_HALF_STENCIL],
        coeffs: [0.0; MAX_HALF_STENCIL],
    };

    // The dual basis: the cofactor matrix of the rows (sb[0], sb[1], sb[2]).
    for i in 0..3 {
        for j in 0..3 {
            stencil.offsets[i][j] = (sb[(i + 1) % 3][(j + 1) % 3] * sb[(i + 2) % 3][(j + 2) % 3]
                - sb[(i + 2) % 3][(j + 1) % 3] * sb[(i + 1) % 3][(j + 2) % 3])
                as i64;
        }
    }
    for j in 0..3 {
        stencil.offsets[3][j] = stencil.offsets[0][j] - stencil.offsets[1][j];
        stencil.offsets[4][j] = stencil.offsets[0][j] - stencil.offsets[2][j];
        stencil.offsets[5][j] = stencil.offsets[1][j] - stencil.offsets[2][j];
    }

    for (c, w) in stencil.coeffs.iter_mut().zip(weights.iter()).take(3) {
        *c = w[3];
    }
    stencil.coeffs[3] = weights[0][1];
    stencil.coeffs[4] = weights[0][2];
    stencil.coeffs[5] = weights[1][2];
    stencil
}

// ---- structure tensor ------------------------------------------------------

fn scalar_image(size: &[usize], spacing: &[f64], vals: Vec<f64>) -> Result<Image> {
    let mut img = Image::from_vec(size, vals)?;
    img.set_spacing(spacing)?;
    Ok(img)
}

/// `MinimumMaximumImageCalculator::ComputeMaximum`: seed with
/// `NumericTraits<double>::NonpositiveMin()` and keep under `value > max`, so
/// a `NaN` never wins.
fn itk_maximum(values: impl Iterator<Item = f64>) -> f64 {
    let mut max = f64::MIN;
    for v in values {
        if v > max {
            max = v;
        }
    }
    max
}

/// `StructureTensorImageFilter::GenerateData` for a scalar image:
/// `K_ρ * (∇u_σ ⊗ ∇u_σ)`, optionally rescaled for unit maximum trace.
///
/// `K_σ` is isotropic (`GradientRecursiveGaussianImageFilter`); `K_ρ` runs
/// along axis 0 only (`RecursiveGaussianImageFilter`, `m_Direction == 0`).
fn structure_tensor(
    data: &[f64],
    size: &[usize],
    spacing: &[f64],
    direction: &[f64],
    noise_scale: f64,
    feature_scale: f64,
    rescale_for_unit_maximum_trace: bool,
) -> Result<Vec<Mat>> {
    let dim = size.len();
    let npix = data.len();
    let base = scalar_image(size, spacing, data.to_vec())?;

    // Gradient of the sigma-smoothed image, in physical (world) coordinates.
    // `recursive_gaussian_with_order` reparametrizes as sigma/spacing[d], so
    // its derivative is index-space; dividing by spacing[d] matches ITK's
    // `it.Get() / spacing`.
    let sigmas = vec![noise_scale; dim];
    let mut grads: Vec<Vec<f64>> = Vec::with_capacity(dim);
    for d in 0..dim {
        let mut orders = vec![GaussianOrder::ZeroOrder; dim];
        orders[d] = GaussianOrder::FirstOrder;
        let g = recursive_gaussian_with_order(&base, &sigmas, &orders, false)?;
        let sp = spacing[d];
        grads.push(g.to_f64_vec().iter().map(|v| v / sp).collect());
    }

    // `TransformLocalVectorToPhysicalVector` (m_UseImageDirection is on by
    // default in GradientRecursiveGaussianImageFilter), then the outer product.
    let mut tensors = vec![[[0.0; MAX_DIM]; MAX_DIM]; npix];
    for (p, t) in tensors.iter_mut().enumerate() {
        let mut g = [0.0; MAX_DIM];
        for (i, gi) in g.iter_mut().enumerate().take(dim) {
            for (j, grad) in grads.iter().enumerate().take(dim) {
                *gi += direction[i * dim + j] * grad[p];
            }
        }
        for i in 0..dim {
            for j in 0..dim {
                t[i][j] = g[i] * g[j];
            }
        }
    }

    // `K_rho`, along axis 0 only. Componentwise, as
    // `RecursiveGaussianImageFilter<TensorImageType>` is.
    let mut rho = vec![0.0; dim];
    rho[0] = feature_scale;
    for i in 0..dim {
        for j in i..dim {
            let comp: Vec<f64> = tensors.iter().map(|t| t[i][j]).collect();
            let smoothed = recursive_gaussian(&scalar_image(size, spacing, comp)?, &rho)?;
            for (t, v) in tensors.iter_mut().zip(smoothed.to_f64_vec()) {
                t[i][j] = v;
                t[j][i] = v;
            }
        }
    }

    if rescale_for_unit_maximum_trace {
        let max_trace = itk_maximum(tensors.iter().map(|t| (0..dim).map(|i| t[i][i]).sum()));
        let scaling = 1.0 / max_trace;
        for t in tensors.iter_mut() {
            for row in t.iter_mut().take(dim) {
                for cell in row.iter_mut().take(dim) {
                    *cell *= scaling;
                }
            }
        }
    }
    Ok(tensors)
}

/// `AnisotropicDiffusionLBRImageFilter::ComputeDiffusionTensors`: the structure
/// tensor with its eigenvalues mapped through [`eigen_values_transform`],
/// rebuilt on the same eigenvectors (`D.Rotate(eigenVectors.GetTranspose())`,
/// i.e. `Σₖ λₖ vₖ vₖᵀ`).
fn diffusion_tensors(
    data: &[f64],
    size: &[usize],
    spacing: &[f64],
    direction: &[f64],
    s: &CoherenceEnhancingDiffusionSettings,
) -> Result<Vec<Mat>> {
    let dim = size.len();
    let mut tensors = structure_tensor(
        data,
        size,
        spacing,
        direction,
        s.noise_scale,
        s.feature_scale,
        s.adimensionize,
    )?;

    for t in tensors.iter_mut() {
        let (mu, vectors) = symmetric_eigen(t, dim);
        let ev = eigen_values_transform(&mu, dim, s);
        for i in 0..dim {
            for j in 0..dim {
                t[i][j] = (0..dim)
                    .map(|k| vectors[i][k] * ev[k] * vectors[j][k])
                    .sum();
            }
        }
    }
    Ok(tensors)
}

// ---- linear anisotropic diffusion (the LBR scheme) -------------------------

/// The assembled sparse operator: per pixel, `half` weights and `2 * half`
/// neighbor indices (`-1` for a neighbor outside the image, ITK's
/// `OutsideBufferIndex()`, giving Neumann boundary conditions), plus the
/// row-sum diagonal.
struct Operator {
    half: usize,
    coeffs: Vec<f64>,
    neighbors: Vec<i64>,
    diagonal: Vec<f64>,
}

/// `LinearAnisotropicDiffusionLBRImageFilter::GenerateStencils`.
fn build_operator(tensors: &[Mat], size: &[usize], spacing: &[f64]) -> Operator {
    let dim = size.len();
    let half = if dim == 2 { 3 } else { MAX_HALF_STENCIL };
    let npix = tensors.len();
    let inv_spacing: Vec<f64> = spacing.iter().map(|s| 1.0 / s).collect();

    let mut strides = vec![1usize; dim];
    for d in 1..dim {
        strides[d] = strides[d - 1] * size[d - 1];
    }

    let mut coeffs = vec![0.0; npix * half];
    let mut neighbors = vec![-1i64; npix * 2 * half];

    let mut x = vec![0i64; dim];
    for p in 0..npix {
        let mut rem = p;
        for d in 0..dim {
            x[d] = (rem % size[d]) as i64;
            rem /= size[d];
        }

        // "Diffusion tensors are homogeneous to the inverse of norms, and are
        // thus rescaled with an inverse spacing."
        let mut d_scaled = [[0.0; MAX_DIM]; MAX_DIM];
        for i in 0..dim {
            for j in 0..dim {
                d_scaled[i][j] = tensors[p][i][j] * inv_spacing[i] * inv_spacing[j];
            }
        }
        let stencil = if dim == 2 {
            stencil_2d(&d_scaled)
        } else {
            stencil_3d(&d_scaled)
        };

        for k in 0..half {
            coeffs[p * half + k] = stencil.coeffs[k];
            for orientation in 0..2 {
                let mut linear = 0i64;
                let mut inside = true;
                for d in 0..dim {
                    let off = stencil.offsets[k][d];
                    let y = if orientation == 1 {
                        x[d] - off
                    } else {
                        x[d] + off
                    };
                    if y < 0 || y >= size[d] as i64 {
                        inside = false;
                        break;
                    }
                    linear += y * strides[d] as i64;
                }
                if inside {
                    neighbors[p * 2 * half + 2 * k + orientation] = linear;
                }
            }
        }
    }

    // The diagonal is the row sum of the same symmetric accumulation the
    // update performs, so every row of `A - diag` sums to zero.
    let mut diagonal = vec![0.0; npix];
    for p in 0..npix {
        for i in 0..2 * half {
            let y = neighbors[p * 2 * half + i];
            if y >= 0 {
                let c = coeffs[p * half + i / 2];
                diagonal[p] += c;
                diagonal[y as usize] += c;
            }
        }
    }

    Operator {
        half,
        coeffs,
        neighbors,
        diagonal,
    }
}

/// One explicit Euler step: `next = δ·(A·prev) + (1 − δ·diag)·prev`, ITK's
/// `ImageUpdate` plus its `FunctorType`.
fn image_update(op: &Operator, prev: &[f64], delta: f64) -> Vec<f64> {
    let npix = prev.len();
    let mut out = vec![0.0; npix];
    let two_half = 2 * op.half;

    for p in 0..npix {
        for i in 0..two_half {
            let y = op.neighbors[p * two_half + i];
            if y >= 0 {
                let y = y as usize;
                let c = op.coeffs[p * op.half + i / 2];
                out[p] += c * prev[y];
                out[y] += c * prev[p];
            }
        }
    }
    for p in 0..npix {
        out[p] = out[p] * delta + prev[p] * (1.0 - delta * op.diagonal[p]);
    }
    out
}

/// `LinearAnisotropicDiffusionLBRImageFilter::ImageUpdateLoop`. Returns the
/// diffused image and the *effective* diffusion time, which falls short of
/// `max_time` when `max_steps` binds.
fn linear_diffusion(
    prev: Vec<f64>,
    tensors: &[Mat],
    size: &[usize],
    spacing: &[f64],
    max_time: f64,
    ratio: f64,
    max_steps: i64,
) -> (Vec<f64>, f64) {
    let op = build_operator(tensors, size, spacing);

    // `delta = MaxStableTimeStep() * m_RatioToMaxStableTimeStep`, where
    // `MaxStableTimeStep()` is `1 / max(diagonal)`. Kept as two operations, in
    // ITK's order, rather than the algebraically equal `ratio / max`.
    let mut delta = (1.0 / itk_maximum(op.diagonal.iter().copied())) * ratio;

    // `int n = ceil(m_DiffusionTime / delta)`. C++ leaves the cast of a
    // non-finite or out-of-range double to `int` undefined; Rust defines it as
    // a saturating cast (NaN -> 0), which is what lets the degenerate
    // Adimensionize-on-a-constant-image path fall through to zero steps.
    let mut n = (max_time / delta).ceil() as i64;
    let effective_time;
    if n > max_steps {
        n = max_steps;
        effective_time = n as f64 * delta;
    } else {
        // `n == 0` gives `delta = inf`; the loop below then runs zero times.
        delta = max_time / n as f64;
        effective_time = max_time;
    }

    let mut image = prev;
    for _ in 0..n.max(0) {
        image = image_update(&op, &image, delta);
    }
    (image, effective_time)
}

// ---- public entry point ----------------------------------------------------

/// `itk::CoherenceEnhancingDiffusionImageFilter`: coherence- or edge-enhancing
/// anisotropic diffusion of `img`, discretized with Mirebeau's lattice-basis-
/// reduction stencil.
///
/// The output has the input's pixel type, size and geometry. A non-positive
/// `diffusion_time` returns the input unchanged.
///
/// Errors with:
/// - [`FilterError::RequiresRealPixelType`] for a non-`Float32`, non-`Float64`
///   input (`pixel_types: RealPixelIDTypeList`);
/// - [`FilterError::UnsupportedLbrDimension`] for an image that is not 2-D or
///   3-D (ITK's `Stencil` has no other overload);
/// - [`FilterError::InvalidTimeStepRatio`] when
///   `ratio_to_max_stable_time_step` is outside `(0, 1]`
///   (`SetRatioToMaxStableTimeStep`);
/// - [`FilterError::ZeroMaxTimeSteps`] when
///   `max_time_steps_between_tensor_updates` is `0` (`SetMaxNumberOfTimeSteps`);
/// - [`FilterError::InvalidSigma`] for a negative `noise_scale` or
///   `feature_scale`, and [`FilterError::AxisTooShortForRecursion`] when an
///   axis has fewer than the four pixels the recursive Gaussian needs.
pub fn coherence_enhancing_diffusion(
    img: &Image,
    settings: &CoherenceEnhancingDiffusionSettings,
) -> Result<Image> {
    let pixel_id = img.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }
    let dim = img.dimension();
    if dim != 2 && dim != 3 {
        return Err(FilterError::UnsupportedLbrDimension(dim));
    }
    let ratio = settings.ratio_to_max_stable_time_step;
    if ratio <= 0.0 || ratio > 1.0 {
        return Err(FilterError::InvalidTimeStepRatio(ratio));
    }
    if settings.max_time_steps_between_tensor_updates == 0 {
        return Err(FilterError::ZeroMaxTimeSteps);
    }
    let max_steps = i64::from(settings.max_time_steps_between_tensor_updates);

    let size = img.size().to_vec();
    let reference_spacing = img.spacing().to_vec();
    let direction = img.direction().to_vec();

    // `Adimensionize` divides the spacing by its minimum, so the finest axis
    // has unit spacing and the scale parameters become dimensionless.
    let spacing = if settings.adimensionize {
        let min_spacing = reference_spacing
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        reference_spacing.iter().map(|s| s / min_spacing).collect()
    } else {
        reference_spacing.clone()
    };

    let mut data = img.to_f64_vec();
    let mut remaining = settings.diffusion_time;
    while remaining > 0.0 {
        let tensors = diffusion_tensors(&data, &size, &spacing, &direction, settings)?;
        let (next, effective) =
            linear_diffusion(data, &tensors, &size, &spacing, remaining, ratio, max_steps);
        data = next;
        remaining -= effective;
    }

    // `image_from_f64` copies `img`'s geometry, matching ITK's restoration of
    // the reference spacing on the way out.
    image_from_f64(pixel_id, &size, img, &data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat2(a: f64, b: f64, c: f64) -> Mat {
        let mut m = [[0.0; MAX_DIM]; MAX_DIM];
        m[0][0] = a;
        m[0][1] = b;
        m[1][0] = b;
        m[1][1] = c;
        m
    }

    /// `Σᵢ 2cᵢ eᵢ eᵢᵀ`, which the doubled assembly makes the operator's
    /// effective tensor. Must equal the `D` the stencil was built from.
    fn reconstruct(stencil: &Stencil, half: usize, dim: usize) -> Mat {
        let mut d = [[0.0; MAX_DIM]; MAX_DIM];
        for k in 0..half {
            for (i, row) in d.iter_mut().enumerate().take(dim) {
                for (j, cell) in row.iter_mut().enumerate().take(dim) {
                    *cell += 2.0
                        * stencil.coeffs[k]
                        * stencil.offsets[k][i] as f64
                        * stencil.offsets[k][j] as f64;
                }
            }
        }
        d
    }

    fn assert_mat_close(got: &Mat, want: &Mat, dim: usize, tol: f64) {
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (got[i][j] - want[i][j]).abs() < tol,
                    "({i},{j}): {} vs {}",
                    got[i][j],
                    want[i][j]
                );
            }
        }
    }

    // ---- Selling / LBR stencil, hand-derived ----

    #[test]
    fn stencil_2d_of_the_identity_is_the_five_point_laplacian() {
        // Canonical superbase is already obtuse for D = I:
        //   c0 = -1/2<b1,b2> = 1/2,  e0 = b0^perp = (0, 1)
        //   c1 = -1/2<b2,b0> = 1/2,  e1 = b1^perp = (-1, 0)
        //   c2 = -1/2<b0,b1> = 0,    e2 = b2^perp = (1, -1)
        let s = stencil_2d(&mat2(1.0, 0.0, 1.0));
        assert_eq!(&s.coeffs[..3], &[0.5, 0.5, 0.0]);
        assert_eq!(s.offsets[0][..2], [0, 1]);
        assert_eq!(s.offsets[1][..2], [-1, 0]);
        assert_eq!(s.offsets[2][..2], [1, -1]);
        // The zero weight on the diagonal offset is what makes it 5-point.
        assert_mat_close(&reconstruct(&s, 3, 2), &mat2(1.0, 0.0, 1.0), 2, 1e-15);
    }

    #[test]
    fn stencil_2d_of_a_diagonal_tensor_keeps_the_canonical_superbase() {
        // D = diag(a, b) with a, b > 0: <D b_i, b_j> <= 0 already.
        //   c0 = b/2 on (0, 1);  c1 = a/2 on (-1, 0);  c2 = 0.
        let s = stencil_2d(&mat2(3.0, 0.0, 7.0));
        assert_eq!(&s.coeffs[..3], &[3.5, 1.5, 0.0]);
        assert_mat_close(&reconstruct(&s, 3, 2), &mat2(3.0, 0.0, 7.0), 2, 1e-15);
    }

    #[test]
    fn stencil_2d_of_a_rotated_tensor_flips_the_superbase_once() {
        // D = [[2, 1], [1, 2]].  <D b1, b0> = 1 > 0, so Selling flips with
        // u = b1 = (0,1), v = b0 = (1,0):
        //   b0 <- v - u = (1, -1),  b1 <- u = (0, 1),  b2 <- -v = (-1, 0).
        // The flipped superbase is obtuse (all three products equal -1), so
        //   c_i = -1/2 * (-1) = 1/2 for every i, and
        //   e0 = (1, 1),  e1 = (-1, 0),  e2 = (0, -1).
        let d = mat2(2.0, 1.0, 2.0);
        let s = stencil_2d(&d);
        assert_eq!(&s.coeffs[..3], &[0.5, 0.5, 0.5]);
        assert_eq!(s.offsets[0][..2], [1, 1]);
        assert_eq!(s.offsets[1][..2], [-1, 0]);
        assert_eq!(s.offsets[2][..2], [0, -1]);
        assert_mat_close(&reconstruct(&s, 3, 2), &d, 2, 1e-15);
    }

    #[test]
    fn stencil_2d_weights_are_non_negative_and_reconstruct_strong_anisotropy() {
        // A tensor with condition number 400, rotated by 30 degrees.
        let (c, sn) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let (l0, l1) = (0.005, 2.0);
        let d = mat2(
            l0 * c * c + l1 * sn * sn,
            (l0 - l1) * c * sn,
            l0 * sn * sn + l1 * c * c,
        );
        let s = stencil_2d(&d);
        for k in 0..3 {
            assert!(s.coeffs[k] >= -1e-15, "coeff {k} = {}", s.coeffs[k]);
        }
        assert_mat_close(&reconstruct(&s, 3, 2), &d, 2, 1e-12);
    }

    #[test]
    fn stencil_3d_of_the_identity_is_the_seven_point_laplacian() {
        let mut d = [[0.0; MAX_DIM]; MAX_DIM];
        for (i, row) in d.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let s = stencil_3d(&d);
        assert_eq!(s.coeffs, [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(s.offsets[0], [1, 0, 0]);
        assert_eq!(s.offsets[1], [0, 1, 0]);
        assert_eq!(s.offsets[2], [0, 0, 1]);
        assert_eq!(s.offsets[3], [1, -1, 0]);
        assert_eq!(s.offsets[4], [1, 0, -1]);
        assert_eq!(s.offsets[5], [0, 1, -1]);
        assert_mat_close(&reconstruct(&s, 6, 3), &d, 3, 1e-15);
    }

    #[test]
    fn stencil_3d_reconstructs_an_anisotropic_tensor() {
        // Rotate diag(0.01, 0.5, 3) about the (1,1,1) axis by 40 degrees.
        let (l0, l1, l2) = (0.01, 0.5, 3.0);
        let a = 40f64.to_radians();
        let (c, sn) = (a.cos(), a.sin());
        let k = 1.0 / 3f64.sqrt();
        // Rodrigues rotation about (k, k, k).
        let mut r = [[0.0; MAX_DIM]; MAX_DIM];
        let axis = [k, k, k];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = if i == j { c } else { 0.0 } + (1.0 - c) * axis[i] * axis[j];
            }
        }
        r[0][1] -= sn * axis[2];
        r[0][2] += sn * axis[1];
        r[1][0] += sn * axis[2];
        r[1][2] -= sn * axis[0];
        r[2][0] -= sn * axis[1];
        r[2][1] += sn * axis[0];
        let lam = [l0, l1, l2];
        let mut d = [[0.0; MAX_DIM]; MAX_DIM];
        for i in 0..3 {
            for j in 0..3 {
                d[i][j] = (0..3).map(|m| r[i][m] * lam[m] * r[j][m]).sum();
            }
        }
        let s = stencil_3d(&d);
        for k in 0..6 {
            assert!(s.coeffs[k] >= -1e-14, "coeff {k} = {}", s.coeffs[k]);
        }
        assert_mat_close(&reconstruct(&s, 6, 3), &d, 3, 1e-12);
    }

    #[test]
    fn selling_reaches_an_obtuse_superbase_for_extreme_anisotropy() {
        // Condition number 10^4. Obtuseness is what guarantees non-negative
        // weights, hence the maximum principle.
        let (c, sn) = (0.3f64.cos(), 0.3f64.sin());
        let (l0, l1) = (1e-4, 1.0);
        let d = mat2(
            l0 * c * c + l1 * sn * sn,
            (l0 - l1) * c * sn,
            l0 * sn * sn + l1 * c * c,
        );
        let s = stencil_2d(&d);
        for k in 0..3 {
            assert!(s.coeffs[k] >= 0.0, "coeff {k} = {}", s.coeffs[k]);
        }
        assert_mat_close(&reconstruct(&s, 3, 2), &d, 2, 1e-12);
    }

    // ---- eigenvalue transfer functions, pinned at analytic points ----

    #[test]
    fn g_ced_hits_its_limit_values() {
        let (lambda, m, alpha) = (0.05, 2.0, 0.01);
        // s <= 0 branch, and the s -> 0+ limit it continues.
        assert_eq!(g_ced(0.0, lambda, m, alpha), alpha);
        assert_eq!(g_ced(-1.0, lambda, m, alpha), alpha);
        assert!((g_ced(1e-6, lambda, m, alpha) - alpha).abs() < 1e-12);
        // s -> inf: exp(0) = 1, so g -> alpha + (1 - alpha) = 1.
        assert!((g_ced(1e12, lambda, m, alpha) - 1.0).abs() < 1e-12);
        // s = lambda: exp(-1).
        let want = alpha + (1.0 - alpha) * (-1.0f64).exp();
        assert!((g_ced(lambda, lambda, m, alpha) - want).abs() < 1e-15);
    }

    #[test]
    fn g_eed_hits_its_limit_values() {
        let (lambda, m, alpha) = (0.05, 2.0, 0.01);
        assert_eq!(g_eed(0.0, lambda, m, alpha), 1.0);
        assert_eq!(g_eed(-1.0, lambda, m, alpha), 1.0);
        assert!((g_eed(1e-6, lambda, m, alpha) - 1.0).abs() < 1e-12);
        assert!((g_eed(1e12, lambda, m, alpha) - alpha).abs() < 1e-12);
        let want = 1.0 - (1.0 - alpha) * (-1.0f64).exp();
        assert!((g_eed(lambda, lambda, m, alpha) - want).abs() < 1e-15);
    }

    #[test]
    fn the_exponent_sharpens_the_transition() {
        let (lambda, alpha) = (1.0, 0.0);
        // At s = 2 lambda, (lambda/s)^m = 2^-m, so g_CED = exp(-2^-m).
        for m in [1.0, 2.0, 4.0] {
            let want = (-(0.5f64).powf(m)).exp();
            assert!(
                (g_ced(2.0, lambda, m, alpha) - want).abs() < 1e-15,
                "m = {m}"
            );
        }
        // Larger m => g closer to 1 above lambda: a sharper switch.
        assert!(g_ced(2.0, lambda, 4.0, alpha) > g_ced(2.0, lambda, 1.0, alpha));
    }

    fn transform(mu: [f64; 2], e: Enhancement) -> [f64; MAX_DIM] {
        let s = CoherenceEnhancingDiffusionSettings {
            enhancement: e,
            ..Default::default()
        };
        let mut ev0 = [0.0; MAX_DIM];
        ev0[0] = mu[0];
        ev0[1] = mu[1];
        eigen_values_transform(&ev0, 2, &s)
    }

    #[test]
    fn ced_gives_alpha_across_the_contour_and_one_along_it() {
        // An ideal straight edge: mu = (0, mu_max). The eigenvector for
        // mu_max is the gradient (across), for 0 the tangent (along).
        let (lambda, m, alpha) = (0.05, 2.0, 0.01);
        let ev = transform([0.0, 1.0], Enhancement::Ced);
        // i = 1 is mu_max: g_CED(0) = alpha  ->  no diffusion across.
        assert_eq!(ev[1], alpha);
        // i = 0 is mu_min: g_CED(mu_max) ~ 1  ->  full diffusion along.
        assert!((ev[0] - g_ced(1.0, lambda, m, alpha)).abs() < 1e-15);
        assert!(ev[0] > 0.99);
    }

    #[test]
    fn ced_is_isotropic_where_the_structure_tensor_is() {
        let ev = transform([0.7, 0.7], Enhancement::Ced);
        assert_eq!(ev[0], 0.01);
        assert_eq!(ev[1], 0.01);
    }

    #[test]
    fn eed_gives_one_along_the_contour_and_alpha_across_it() {
        let (lambda, m, alpha) = (0.05, 2.0, 0.01);
        let ev = transform([0.0, 1.0], Enhancement::Eed);
        // i = 0 is mu_min: g_EED(0) = 1  ->  full diffusion along.
        assert_eq!(ev[0], 1.0);
        // i = 1 is mu_max: g_EED(mu_max - mu_min) = 1 - 0.99*exp(-0.0025)
        //                                          = 0.0124719088265145, just
        // above alpha -- the s -> inf limit is approached from above.
        assert!((ev[1] - g_eed(1.0, lambda, m, alpha)).abs() < 1e-15);
        assert!((ev[1] - 0.012_471_908_826_514_5).abs() < 1e-15, "{}", ev[1]);
        assert!(ev[1] > alpha);
    }

    #[test]
    fn ceed_uses_the_raw_eigenvalue_so_a_flat_region_diffuses_fully() {
        // cEED's g_EED(mu_i) does not subtract mu_min, so where the whole
        // structure tensor is large, every direction is damped -- unlike EED,
        // which always leaves one direction at 1.
        let ev = transform([1.0, 1.0], Enhancement::CEed);
        // g_EED(1) = 0.0124719088265145 for both.
        assert!((ev[0] - 0.012_471_908_826_514_5).abs() < 1e-15, "{}", ev[0]);
        assert_eq!(ev[0], ev[1]);
        let ev = transform([1.0, 1.0], Enhancement::Eed);
        assert_eq!([ev[0], ev[1]], [1.0, 1.0]);
    }

    #[test]
    fn cced_damps_where_the_eigenvalue_itself_is_large() {
        // cCED divides the CED argument by (1 + mu_i/lambda), so a large mu_i
        // shrinks the argument and pushes g_CED back toward alpha.
        let ced = transform([0.0, 1.0], Enhancement::Ced);
        let cced = transform([0.0, 1.0], Enhancement::CCed);
        // mu_0 = 0: the divisor is 1, so the two agree exactly.
        assert_eq!(cced[0], ced[0]);
        // mu_1 = mu_max: the numerator is 0 either way.
        assert_eq!(cced[1], ced[1]);

        // With mu_0 > 0 the divisor bites.
        let ced = transform([0.5, 1.0], Enhancement::Ced);
        let cced = transform([0.5, 1.0], Enhancement::CCed);
        assert!(cced[0] < ced[0], "{} vs {}", cced[0], ced[0]);
    }

    #[test]
    fn isotropic_ignores_the_index_entirely() {
        let (lambda, m, alpha) = (0.05, 2.0, 0.01);
        let ev = transform([0.0, 0.3], Enhancement::Isotropic);
        let want = g_eed(0.3, lambda, m, alpha);
        assert_eq!(ev[0], want);
        assert_eq!(ev[1], want);
    }

    // ---- structure tensor ----

    /// A ramp `u(x, y) = x`. `∇u = (1, 0)`, so `S = [[1, 0], [0, 0]]`.
    ///
    /// The grid is 64 wide because `K_ρ` smooths *along x* (the upstream
    /// axis-0-only quirk) and so drags the two boundary columns' bad
    /// derivative estimates toward the centre; 64 puts the centre 16 `ρ` away
    /// and the residual under 1e-9. See
    /// `the_feature_scale_smooths_along_axis_zero_only`, which pins the same
    /// contamination on a narrow grid.
    #[test]
    fn structure_tensor_of_a_ramp_is_the_hand_computed_outer_product() {
        let (nx, ny) = (64, 16);
        let data: Vec<f64> = (0..nx * ny).map(|p| (p % nx) as f64).collect();
        let spacing = [1.0, 1.0];
        let direction = [1.0, 0.0, 0.0, 1.0];
        let t = structure_tensor(&data, &[nx, ny], &spacing, &direction, 0.5, 2.0, false).unwrap();
        let center = (ny / 2) * nx + nx / 2;
        assert_mat_close(&t[center], &mat2(1.0, 0.0, 0.0), 2, 1e-8);
        // The y-derivative of a function of x alone vanishes: the recursion is
        // separable, so `S_01` and `S_11` are pure rounding noise everywhere,
        // not just near the centre.
        for m in &t {
            assert!(m[0][1].abs() < 1e-14, "S_01 = {}", m[0][1]);
            assert!(m[1][1].abs() < 1e-25, "S_11 = {}", m[1][1]);
        }
    }

    #[test]
    fn structure_tensor_of_a_ramp_along_y_is_the_transposed_outer_product() {
        // Only 32 wide: `K_ρ` smooths along x, where `S_11` is constant, so
        // no boundary error is dragged in and the centre is exact.
        let (nx, ny) = (32, 32);
        let data: Vec<f64> = (0..nx * ny).map(|p| (p / nx) as f64).collect();
        let t = structure_tensor(
            &data,
            &[nx, ny],
            &[1.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            0.5,
            2.0,
            false,
        )
        .unwrap();
        let center = (ny / 2) * nx + nx / 2;
        assert_mat_close(&t[center], &mat2(0.0, 0.0, 1.0), 2, 1e-12);
    }

    #[test]
    fn structure_tensor_scales_with_spacing() {
        // u(x) = x in index units is u = X / spacing in physical units, so
        // |grad u| = 1 / spacing and S_00 = 1 / spacing^2.
        let (nx, ny) = (32, 32);
        let data: Vec<f64> = (0..nx * ny).map(|p| (p % nx) as f64).collect();
        let t = structure_tensor(
            &data,
            &[nx, ny],
            &[2.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            0.5,
            2.0,
            false,
        )
        .unwrap();
        let center = (ny / 2) * nx + nx / 2;
        assert!((t[center][0][0] - 0.25).abs() < 1e-8, "{}", t[center][0][0]);
    }

    #[test]
    fn rescale_for_unit_maximum_trace_normalizes_the_largest_trace_to_one() {
        let (nx, ny) = (16, 16);
        let data: Vec<f64> = (0..nx * ny).map(|p| 5.0 * (p % nx) as f64).collect();
        let t = structure_tensor(
            &data,
            &[nx, ny],
            &[1.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            0.5,
            2.0,
            true,
        )
        .unwrap();
        let max_trace = itk_maximum(t.iter().map(|m| m[0][0] + m[1][1]));
        assert!((max_trace - 1.0).abs() < 1e-12, "{max_trace}");
    }

    #[test]
    fn the_feature_scale_smooths_along_axis_zero_only() {
        // Upstream quirk: `K_rho` goes through the single-direction
        // `RecursiveGaussianImageFilter` (`m_Direction == 0`), so it smooths
        // along axis 0 and no other.
        //
        // Take a 16-wide ramp along x and its exact transpose, a ramp along y.
        // Both have one nonzero structure-tensor component, equal to 1 in the
        // interior and wrong in the two boundary lines, where the replicating
        // recursion cannot see the ramp continue.
        //
        //   * The x-ramp's `S_00` varies along x, the axis `K_rho` smooths, so
        //     the boundary error is dragged inward: at the centre it is off by
        //     1.4e-3.
        //   * The y-ramp's `S_11` varies along y, which `K_rho` never touches,
        //     so at the centre it retains the bare 3.3e-9 error of a recursive
        //     first derivative eight pixels from a replicating boundary.
        //
        // An isotropic `K_rho` would contaminate both alike and the two errors
        // would match. Instead they differ by five orders of magnitude, and
        // that asymmetry *is* the quirk.
        let n = 16;
        let dx: Vec<f64> = (0..n * n).map(|p| (p % n) as f64).collect();
        let dy: Vec<f64> = (0..n * n).map(|p| (p / n) as f64).collect();
        let sp = [1.0, 1.0];
        let dir = [1.0, 0.0, 0.0, 1.0];
        let tx = structure_tensor(&dx, &[n, n], &sp, &dir, 0.5, 2.0, false).unwrap();
        let ty = structure_tensor(&dy, &[n, n], &sp, &dir, 0.5, 2.0, false).unwrap();
        let c = (n / 2) * n + n / 2;

        let smeared = (tx[c][0][0] - 1.0).abs();
        let untouched = (ty[c][1][1] - 1.0).abs();
        assert!(
            smeared > 1e-3,
            "x-ramp should be smeared by K_rho: {smeared}"
        );
        assert!(
            untouched < 1e-8,
            "y-ramp must be untouched by K_rho: {untouched}"
        );
        assert!(
            smeared > 1e5 * untouched,
            "K_rho looks isotropic: {smeared} vs {untouched}"
        );

        // Concretely: the y-ramp's `S_11` is the *unsmoothed* squared gradient,
        // recovered to rounding by the axis-0 pass over a signal constant in x.
        let g = recursive_gaussian_with_order(
            &scalar_image(&[n, n], &sp, dy.clone()).unwrap(),
            &[0.5, 0.5],
            &[GaussianOrder::ZeroOrder, GaussianOrder::FirstOrder],
            false,
        )
        .unwrap()
        .to_f64_vec();
        assert!((ty[c][1][1] - g[c] * g[c]).abs() < 1e-15);
    }

    // ---- the assembled operator ----

    fn ident_tensors(npix: usize, dim: usize) -> Vec<Mat> {
        let mut t = [[0.0; MAX_DIM]; MAX_DIM];
        for (i, row) in t.iter_mut().enumerate().take(dim) {
            row[i] = 1.0;
        }
        vec![t; npix]
    }

    #[test]
    fn operator_rows_sum_to_zero_so_a_constant_is_a_fixed_point() {
        let size = [5, 4];
        let tensors = ident_tensors(20, 2);
        let op = build_operator(&tensors, &size, &[1.0, 1.0]);
        let prev = vec![3.0; 20];
        let next = image_update(&op, &prev, 0.1);
        for v in next {
            assert!((v - 3.0).abs() < 1e-14, "{v}");
        }
    }

    #[test]
    fn the_scheme_conserves_the_pixel_sum() {
        // A symmetric operator with zero row sums also has zero column sums,
        // so sum(next) == sum(prev): the scheme is conservative.
        let size = [6, 5];
        let tensors = ident_tensors(30, 2);
        let op = build_operator(&tensors, &size, &[1.0, 1.0]);
        let prev: Vec<f64> = (0..30).map(|i| (i as f64 * 0.37).sin()).collect();
        let before: f64 = prev.iter().sum();
        let next = image_update(&op, &prev, 0.05);
        let after: f64 = next.iter().sum();
        assert!((after - before).abs() < 1e-12, "{before} vs {after}");
    }

    #[test]
    fn the_identity_tensor_reproduces_the_five_point_laplacian() {
        // With D = I and unit spacing the assembled off-diagonal weight to each
        // of the four axis neighbors is 2 * 0.5 = 1, so
        //   next = prev + delta * (sum of 4 neighbors - 4 * prev).
        let size = [5, 5];
        let tensors = ident_tensors(25, 2);
        let op = build_operator(&tensors, &size, &[1.0, 1.0]);
        // Interior pixel (2, 2) = index 12: all four neighbors exist.
        assert!((op.diagonal[12] - 4.0).abs() < 1e-14, "{}", op.diagonal[12]);
        // Corner pixel (0, 0) = index 0: two neighbors.
        assert!((op.diagonal[0] - 2.0).abs() < 1e-14, "{}", op.diagonal[0]);

        let mut prev = vec![0.0; 25];
        prev[12] = 1.0;
        let next = image_update(&op, &prev, 0.1);
        assert!((next[12] - (1.0 - 0.4)).abs() < 1e-14, "{}", next[12]);
        for n in [7, 11, 13, 17] {
            assert!((next[n] - 0.1).abs() < 1e-14, "{n}: {}", next[n]);
        }
        // Diagonal neighbors get nothing: the stencil's third offset has
        // weight zero.
        for n in [6, 8, 16, 18] {
            assert_eq!(next[n], 0.0);
        }
    }

    #[test]
    fn max_stable_time_step_is_the_reciprocal_of_the_largest_diagonal() {
        let size = [5, 5];
        let tensors = ident_tensors(25, 2);
        let op = build_operator(&tensors, &size, &[1.0, 1.0]);
        let max_diag = itk_maximum(op.diagonal.iter().copied());
        assert!((max_diag - 4.0).abs() < 1e-14);
        // delta = ratio / max_diag keeps 1 - delta * diag >= 0 for every pixel,
        // which is the maximum principle.
        for &ratio in &[0.1, 0.7, 1.0] {
            let delta = ratio / max_diag;
            for &d in &op.diagonal {
                assert!(1.0 - delta * d >= -1e-15);
            }
        }
    }

    #[test]
    fn spacing_rescales_the_tensor_by_the_inverse_spacing_product() {
        // D(i, j) * invSpacing[i] * invSpacing[j]: doubling spacing along x
        // quarters the x-direction weight.
        let size = [5, 5];
        let tensors = ident_tensors(25, 2);
        let op = build_operator(&tensors, &size, &[2.0, 1.0]);
        // D_scaled = diag(1/4, 1), so c0 = 1/2 on (0,1) and c1 = 1/8 on (-1,0).
        assert!((op.coeffs[0] - 0.5).abs() < 1e-15, "{}", op.coeffs[0]);
        assert!((op.coeffs[1] - 0.125).abs() < 1e-15, "{}", op.coeffs[1]);
    }

    // ---- time stepping ----

    #[test]
    fn the_step_count_is_capped_and_shortens_the_effective_time() {
        let size = [5, 5];
        let tensors = ident_tensors(25, 2);
        // max diag = 4, ratio = 1 => delta = 0.25.  n = ceil(10 / 0.25) = 40,
        // capped to 3, so the effective time is 3 * 0.25 = 0.75.
        let (_, effective) =
            linear_diffusion(vec![0.0; 25], &tensors, &size, &[1.0, 1.0], 10.0, 1.0, 3);
        assert!((effective - 0.75).abs() < 1e-14, "{effective}");
    }

    #[test]
    fn an_uncapped_run_spends_exactly_the_requested_time() {
        let size = [5, 5];
        let tensors = ident_tensors(25, 2);
        // delta = 0.25, n = ceil(0.6 / 0.25) = 3 <= 5, so delta becomes 0.2 and
        // the whole 0.6 is spent.
        let (_, effective) =
            linear_diffusion(vec![0.0; 25], &tensors, &size, &[1.0, 1.0], 0.6, 1.0, 5);
        assert_eq!(effective, 0.6);
    }

    #[test]
    fn a_zero_diagonal_yields_an_infinite_stable_step_and_zero_iterations() {
        // Zero tensors => zero diagonal => MaxStableTimeStep = +inf =>
        // n = ceil(t / inf) = 0. The image is returned unchanged and the whole
        // time is charged as effective, so the outer loop terminates.
        let size = [5, 5];
        let tensors = vec![[[0.0; MAX_DIM]; MAX_DIM]; 25];
        let prev: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let (next, effective) =
            linear_diffusion(prev.clone(), &tensors, &size, &[1.0, 1.0], 1.0, 0.7, 5);
        assert_eq!(next, prev);
        assert_eq!(effective, 1.0);
    }

    // ---- the full filter ----

    fn f64_image(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn constant_image_is_returned_unchanged() {
        // Adimensionize on a constant image is the 1/0 -> NaN path documented
        // above; it must still be a fixed point.
        let img = f64_image(&[12, 12], vec![7.0; 144]);
        let out = coherence_enhancing_diffusion(&img, &Default::default()).unwrap();
        for v in out.to_f64_vec() {
            assert_eq!(v, 7.0);
        }
    }

    #[test]
    fn constant_image_is_returned_unchanged_without_adimensionize() {
        // The finite path: S = 0, so every eigenvalue is 0 and CED gives an
        // isotropic alpha * I. Diffusing a constant with any tensor is a no-op.
        let img = f64_image(&[12, 12], vec![7.0; 144]);
        let s = CoherenceEnhancingDiffusionSettings {
            adimensionize: false,
            ..Default::default()
        };
        let out = coherence_enhancing_diffusion(&img, &s).unwrap();
        for v in out.to_f64_vec() {
            assert!((v - 7.0).abs() < 1e-12, "{v}");
        }
    }

    #[test]
    fn a_non_positive_diffusion_time_returns_the_input() {
        let data: Vec<f64> = (0..144).map(|i| (i as f64 * 0.1).sin()).collect();
        let img = f64_image(&[12, 12], data.clone());
        for t in [0.0, -1.0] {
            let s = CoherenceEnhancingDiffusionSettings {
                diffusion_time: t,
                ..Default::default()
            };
            assert_eq!(
                coherence_enhancing_diffusion(&img, &s)
                    .unwrap()
                    .to_f64_vec(),
                data
            );
        }
    }

    #[test]
    fn diffusion_obeys_the_maximum_principle() {
        let data: Vec<f64> = (0..256).map(|i| ((i * 7) % 13) as f64).collect();
        let img = f64_image(&[16, 16], data.clone());
        let out = coherence_enhancing_diffusion(&img, &Default::default())
            .unwrap()
            .to_f64_vec();
        let lo = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for v in out {
            assert!(v >= lo - 1e-9 && v <= hi + 1e-9, "{v} outside [{lo}, {hi}]");
        }
    }

    /// A vertical edge: `u = 0` left of `x = 8`, `u = 1` at and right of it.
    fn vertical_edge(nx: usize, ny: usize) -> Vec<f64> {
        (0..nx * ny)
            .map(|p| if p % nx < nx / 2 { 0.0 } else { 1.0 })
            .collect()
    }

    #[test]
    fn ced_smooths_along_an_edge_while_preserving_the_cross_edge_profile() {
        let (nx, ny) = (24, 24);
        let mut data = vertical_edge(nx, ny);
        // Perturb along the edge so there is something to smooth away.
        for y in 0..ny {
            let jitter = if y % 2 == 0 { 0.15 } else { -0.15 };
            data[y * nx + nx / 2] += jitter;
            data[y * nx + nx / 2 - 1] += jitter;
        }
        let img = f64_image(&[nx, ny], data.clone());
        let s = CoherenceEnhancingDiffusionSettings {
            diffusion_time: 4.0,
            enhancement: Enhancement::Ced,
            ..Default::default()
        };
        let out = coherence_enhancing_diffusion(&img, &s)
            .unwrap()
            .to_f64_vec();

        // (1) Variance *along* the edge (down the two columns at the jump)
        //     must fall: that is the coherence-enhancing smoothing.
        let column_variance = |v: &[f64], x: usize| -> f64 {
            let col: Vec<f64> = (0..ny).map(|y| v[y * nx + x]).collect();
            let mean = col.iter().sum::<f64>() / ny as f64;
            col.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / ny as f64
        };
        let before = column_variance(&data, nx / 2);
        let after = column_variance(&out, nx / 2);
        assert!(
            after < before * 0.5,
            "along-edge variance {before} -> {after}"
        );

        // (2) The cross-edge jump survives: the mean step across x = 11 -> 12
        //     stays within a small tolerance of its original height.
        let row_mean = |v: &[f64], x: usize| -> f64 {
            (0..ny).map(|y| v[y * nx + x]).sum::<f64>() / ny as f64
        };
        let jump_before = row_mean(&data, nx / 2) - row_mean(&data, nx / 2 - 1);
        let jump_after = row_mean(&out, nx / 2) - row_mean(&out, nx / 2 - 1);
        assert!(
            jump_after > 0.9 * jump_before,
            "edge jump {jump_before} -> {jump_after}"
        );
    }

    #[test]
    fn isotropic_enhancement_blurs_the_edge_more_than_ced_does() {
        let (nx, ny) = (24, 24);
        let data = vertical_edge(nx, ny);
        let img = f64_image(&[nx, ny], data.clone());
        let run = |e: Enhancement| {
            let s = CoherenceEnhancingDiffusionSettings {
                diffusion_time: 4.0,
                enhancement: e,
                ..Default::default()
            };
            coherence_enhancing_diffusion(&img, &s)
                .unwrap()
                .to_f64_vec()
        };
        let row_mean = |v: &[f64], x: usize| -> f64 {
            (0..ny).map(|y| v[y * nx + x]).sum::<f64>() / ny as f64
        };
        let jump = |v: &[f64]| row_mean(v, nx / 2) - row_mean(v, nx / 2 - 1);

        let ced = run(Enhancement::Ced);
        let iso = run(Enhancement::Isotropic);
        assert!(
            jump(&ced) > jump(&iso),
            "CED {} should preserve the edge better than Isotropic {}",
            jump(&ced),
            jump(&iso)
        );
    }

    #[test]
    fn a_three_dimensional_volume_diffuses() {
        let (nx, ny, nz) = (8, 8, 8);
        let data: Vec<f64> = (0..nx * ny * nz)
            .map(|p| if p % nx < 4 { 0.0 } else { 1.0 })
            .collect();
        let img = f64_image(&[nx, ny, nz], data.clone());
        let out = coherence_enhancing_diffusion(&img, &Default::default())
            .unwrap()
            .to_f64_vec();
        assert_eq!(out.len(), data.len());
        assert!(out.iter().all(|v| v.is_finite()));
        // Conservative scheme: the total is preserved.
        let before: f64 = data.iter().sum();
        let after: f64 = out.iter().sum();
        assert!((after - before).abs() < 1e-8, "{before} vs {after}");
    }

    #[test]
    fn output_pixel_type_and_geometry_follow_the_input() {
        let mut img = Image::from_vec(&[12, 12], vec![1.0f32; 144]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        let out = coherence_enhancing_diffusion(&img, &Default::default()).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.spacing(), [0.5, 2.0]);
        assert_eq!(out.origin(), [-1.0, 3.0]);
    }

    // ---- error paths ----

    #[test]
    fn a_non_real_pixel_type_is_rejected() {
        let img = Image::from_vec(&[8, 8], vec![1i16; 64]).unwrap();
        assert_eq!(
            coherence_enhancing_diffusion(&img, &Default::default()),
            Err(FilterError::RequiresRealPixelType(PixelId::Int16))
        );
    }

    #[test]
    fn one_and_four_dimensional_images_are_rejected() {
        let img = f64_image(&[16], vec![0.0; 16]);
        assert_eq!(
            coherence_enhancing_diffusion(&img, &Default::default()),
            Err(FilterError::UnsupportedLbrDimension(1))
        );
        let img = f64_image(&[4, 4, 4, 4], vec![0.0; 256]);
        assert_eq!(
            coherence_enhancing_diffusion(&img, &Default::default()),
            Err(FilterError::UnsupportedLbrDimension(4))
        );
    }

    #[test]
    fn a_ratio_outside_zero_to_one_is_rejected() {
        let img = f64_image(&[8, 8], vec![0.0; 64]);
        for ratio in [0.0, -0.5, 1.0001] {
            let s = CoherenceEnhancingDiffusionSettings {
                ratio_to_max_stable_time_step: ratio,
                ..Default::default()
            };
            assert_eq!(
                coherence_enhancing_diffusion(&img, &s),
                Err(FilterError::InvalidTimeStepRatio(ratio))
            );
        }
        // The closed upper end is valid.
        let s = CoherenceEnhancingDiffusionSettings {
            ratio_to_max_stable_time_step: 1.0,
            ..Default::default()
        };
        assert!(coherence_enhancing_diffusion(&img, &s).is_ok());
    }

    #[test]
    fn zero_max_time_steps_is_rejected() {
        let img = f64_image(&[8, 8], vec![0.0; 64]);
        let s = CoherenceEnhancingDiffusionSettings {
            max_time_steps_between_tensor_updates: 0,
            ..Default::default()
        };
        assert_eq!(
            coherence_enhancing_diffusion(&img, &s),
            Err(FilterError::ZeroMaxTimeSteps)
        );
    }

    #[test]
    fn a_negative_scale_is_rejected_by_the_recursive_gaussian() {
        let img = f64_image(&[8, 8], vec![0.0; 64]);
        let s = CoherenceEnhancingDiffusionSettings {
            noise_scale: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            coherence_enhancing_diffusion(&img, &s),
            Err(FilterError::InvalidSigma(_))
        ));
    }

    #[test]
    fn an_axis_shorter_than_the_recursion_is_rejected() {
        let img = f64_image(&[3, 8], vec![0.0; 24]);
        assert!(matches!(
            coherence_enhancing_diffusion(&img, &Default::default()),
            Err(FilterError::AxisTooShortForRecursion { .. })
        ));
    }
}
