//! The B-spline machinery `N4BiasFieldCorrectionImageFilter` is built on,
//! scoped to the subset N4 actually reaches.
//!
//! Three ITK classes are folded in here:
//!
//! - `itkCoxDeBoorBSplineKernelFunction.h(.hxx)` — the piecewise-polynomial
//!   B-spline kernel and the `GetShapeFunctionsInZeroToOneInterval()` matrix
//!   that seeds the lattice-refinement coefficients ([`Kernel`],
//!   [`refined_lattice_coefficients`]).
//! - `itkBSplineScatteredDataPointSetToImageFilter.h(.hxx)` — the
//!   Lee/Wolberg multilevel B-spline approximation of scattered data
//!   ([`fit`]).
//! - `itkBSplineControlPointImageFilter.h(.hxx)` — sampling a control-point
//!   lattice onto an image grid ([`reconstruct`]) and dyadic refinement of a
//!   lattice ([`refine`]).
//!
//! **Scope.** N4 drives the scattered-data filter with
//! `SetNumberOfLevels(1)` (`itkN4BiasFieldCorrectionImageFilter.hxx`'s
//! `UpdateBiasFieldEstimate` fills `numberOfFittingLevels` with 1) and never
//! touches `SetCloseDimension` or `SetGenerateOutputImage(true)`, so [`fit`]
//! implements only the single-level, open-dimension, lattice-only path.
//! N4's multi-level behaviour instead comes from
//! `BSplineControlPointImageFilter::RefineControlPointLattice`, which is
//! ported in full as [`refine`]. The residual-update pass
//! (`ThreadedGenerateDataForUpdatingResidualValues`) is only reachable when
//! the scattered-data filter itself is multilevel, so it is absent.
//!
//! **Precision.** ITK's N4 pins `RealType = float` and the point set's
//! coordinate type to `float`. This port computes in `f64` throughout,
//! matching the workspace's `to_f64_vec` idiom. Every formula is otherwise
//! transcribed literally.

use crate::error::{FilterError, Result};

/// `m_BSplineEpsilon{ 1e-3 }` — `itkBSplineScatteredDataPointSetToImageFilter.h:378`
/// and `itkBSplineControlPointImageFilter.h:261`.
const BSPLINE_EPSILON: f64 = 1e-3;

// ---- vnl_real_polynomial --------------------------------------------------

/// A polynomial in `vnl_real_polynomial`'s layout: `coeffs[0]` is the
/// highest-degree coefficient, `coeffs[n-1]` the constant term. Degree is
/// `len - 1`; leading zeros are never trimmed, exactly as vnl leaves them.
type Poly = Vec<f64>;

/// `vnl_real_polynomial::operator*`: coefficient convolution, degree `d1 + d2`.
fn poly_mul(a: &[f64], b: &[f64]) -> Poly {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// `vnl_real_polynomial::operator+`: align at the constant term, so the
/// shorter operand is zero-padded at the *front* under this layout.
fn poly_add(a: &[f64], b: &[f64]) -> Poly {
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    for (i, &x) in a.iter().enumerate() {
        out[n - a.len() + i] += x;
    }
    for (i, &y) in b.iter().enumerate() {
        out[n - b.len() + i] += y;
    }
    out
}

/// `vnl_real_polynomial::evaluate`: Horner from the highest-degree end.
fn poly_eval(p: &[f64], x: f64) -> f64 {
    p.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// `itk::Math::AlmostEquals(d, 0.0)`. `itkMath.h`'s `FloatAlmostEqual`
/// combines a ULP check with an absolute-difference check; against a literal
/// `0.0` comparand the ULP branch never fires, so it collapses to
/// `|d| <= 0.1 * epsilon`. The knot differences this tests are either exactly
/// `0.0` (repeated knot) or a whole number of knot spacings, so the tight
/// threshold separates them cleanly.
fn is_almost_zero(d: f64) -> bool {
    d.abs() <= 0.1 * f64::EPSILON
}

/// `CoxDeBoorBSplineKernelFunction::CoxDeBoor` — the Cox-de Boor recursion
/// yielding the polynomial piece `whichPiece` of basis function
/// `whichBasisFunction`.
fn cox_de_boor(order: usize, knots: &[f64], which_basis: usize, which_piece: usize) -> Poly {
    let p = order - 1;
    let i = which_basis;

    if p == 0 && which_basis == which_piece {
        return vec![1.0];
    }

    // Term 1. When `p == 0` and the basis function is not the requested
    // piece, both denominators below are exactly zero and this returns the
    // constant zero polynomial — the recursion's other base case.
    let den = knots[i + p] - knots[i];
    let poly1 = if is_almost_zero(den) {
        vec![0.0]
    } else {
        poly_mul(
            &[1.0 / den, -knots[i] / den],
            &cox_de_boor(order - 1, knots, i, which_piece),
        )
    };

    // Term 2.
    let den = knots[i + p + 1] - knots[i + 1];
    let poly2 = if is_almost_zero(den) {
        vec![0.0]
    } else {
        poly_mul(
            &[-1.0 / den, knots[i + p + 1] / den],
            &cox_de_boor(order - 1, knots, i + 1, which_piece),
        )
    };

    poly_add(&poly1, &poly2)
}

/// `CoxDeBoorBSplineKernelFunction::GenerateBSplineShapeFunctions`: the
/// `ceil(0.5 * (order + 1))` polynomial pieces of the single basis function
/// centered at zero, for positive parametric values.
fn centered_shape_functions(spline_order: usize) -> Vec<Poly> {
    let order = spline_order + 1;
    let number_of_pieces = (0.5 * (order + 1) as f64) as usize;
    let knots: Vec<f64> = (0..=order)
        .map(|i| -0.5 * order as f64 + i as f64)
        .collect();
    (0..number_of_pieces)
        .map(|i| cox_de_boor(order, &knots, 0, (0.5 * order as f64) as usize + i))
        .collect()
}

/// `CoxDeBoorBSplineKernelFunction::GetShapeFunctionsInZeroToOneInterval`:
/// the `spline_order + 1` basis functions restricted to `[0, 1]`, one
/// polynomial per row.
fn shape_functions_in_zero_to_one_interval(spline_order: usize) -> Vec<Poly> {
    let order = spline_order + 1;
    let knots: Vec<f64> = (0..2 * order)
        .map(|i| -(spline_order as f64) + i as f64)
        .collect();
    (0..order)
        .map(|i| cox_de_boor(order, &knots, i, order - 1))
        .collect()
}

// ---- kernel ---------------------------------------------------------------

/// The B-spline basis function of a given order, evaluated at a parametric
/// offset from its center.
///
/// `BSplineScatteredDataPointSetToImageFilter` and
/// `BSplineControlPointImageFilter` both dispatch orders 0-3 to the closed
/// forms in `itkBSplineKernelFunction.h` and everything else to the
/// Cox-de Boor kernel; the closed forms agree with Cox-de Boor for orders
/// 1-3, and order 0 is rejected up front by both filters, so this reproduces
/// the same values on every reachable path.
struct Kernel {
    spline_order: usize,
    /// Populated only for `spline_order >= 4`, where the closed forms run out.
    shape_functions: Vec<Poly>,
}

impl Kernel {
    fn new(spline_order: usize) -> Self {
        Self {
            spline_order,
            shape_functions: if spline_order >= 4 {
                centered_shape_functions(spline_order)
            } else {
                Vec::new()
            },
        }
    }

    fn evaluate(&self, u: f64) -> f64 {
        let a = u.abs();
        match self.spline_order {
            // `BSplineKernelFunction<1>::Evaluate`.
            1 => {
                if a < 1.0 {
                    1.0 - a
                } else {
                    0.0
                }
            }
            // `BSplineKernelFunction<2>::Evaluate`.
            2 => {
                if a < 0.5 {
                    0.75 - a * a
                } else if a < 1.5 {
                    (9.0 - 12.0 * a + 4.0 * a * a) * 0.125
                } else {
                    0.0
                }
            }
            // `BSplineKernelFunction<3>::Evaluate`.
            3 => {
                if a < 1.0 {
                    (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0
                } else if a < 2.0 {
                    (8.0 - 12.0 * a + 6.0 * a * a - a * a * a) / 6.0
                } else {
                    0.0
                }
            }
            // `CoxDeBoorBSplineKernelFunction::Evaluate`.
            order => {
                let which = if order % 2 == 0 {
                    (a + 0.5) as usize
                } else {
                    a as usize
                };
                match self.shape_functions.get(which) {
                    Some(piece) => poly_eval(piece, a),
                    None => 0.0,
                }
            }
        }
    }
}

// ---- lattice refinement coefficients --------------------------------------

/// Solve `a * x = b` for `x` by Gauss-Jordan elimination with partial
/// pivoting. `a` is consumed. Returns `None` if `a` is singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    for c in 0..n {
        let pivot = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .expect("column range is non-empty");
        a.swap(c, pivot);
        b.swap(c, pivot);
        let p = a[c][c];
        if p == 0.0 {
            return None;
        }
        for v in a[c].iter_mut() {
            *v /= p;
        }
        for v in b[c].iter_mut() {
            *v /= p;
        }
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = a[r][c];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                a[r][k] -= f * a[c][k];
                b[r][k] -= f * b[c][k];
            }
        }
    }
    Some(b)
}

/// `m_RefinedLatticeCoefficients[i]`, the `2 x (spline_order + 1)` matrix that
/// maps a run of coarse control points onto the two fine control points a
/// dyadic refinement inserts (`itkBSplineControlPointImageFilter.hxx`'s
/// `SetSplineOrder`).
///
/// ITK forms it as `pinv(R) * S` with a rank-preserving SVD (`rcond = 0`);
/// `R` is `C` with its columns scaled by powers of two, transposed and
/// row-flipped, and `C` is the `[0, 1]`-interval shape-function matrix, whose
/// rows are `spline_order + 1` linearly independent polynomials of degree
/// `<= spline_order`. `C` is therefore square and nonsingular, the column
/// scaling and row flip are nonsingular, and so `pinv(R) == R^-1` exactly.
/// This solves `R * X = S` by elimination instead of building an SVD; the
/// result is the same matrix.
fn refined_lattice_coefficients(spline_order: usize) -> Vec<Vec<f64>> {
    let c = shape_functions_in_zero_to_one_interval(spline_order);
    let n = spline_order + 1;

    // `S = flipud(C^T)`, i.e. `S[a][b] == C[b][n - 1 - a]`.
    // `R = flipud((C * diag(2^(n - 1 - j)))^T)`, so `R[a][b] == S[a][b] * 2^a`.
    let s: Vec<Vec<f64>> = (0..n)
        .map(|a| (0..n).map(|b| c[b][n - 1 - a]).collect())
        .collect();
    let r: Vec<Vec<f64>> = (0..n)
        .map(|a| s[a].iter().map(|&v| v * 2f64.powi(a as i32)).collect())
        .collect();

    let x = solve(r, s).expect("the [0,1] shape-function matrix is nonsingular by construction");
    x.into_iter().take(2).collect()
}

// ---- lattice --------------------------------------------------------------

/// A scalar control-point lattice: `itk::Image<itk::Vector<RealType, 1>, D>`
/// in ITK, first index fastest here as everywhere in this workspace.
///
/// The lattice's origin/spacing/direction (`SetPhiLatticeParametricDomainParameters`)
/// are pure metadata: [`reconstruct`] derives the parametric mapping from the
/// target grid's size and the lattice's *size*, never from its pose. N4 in
/// turn overrides the pose on every reconstruction. So this type carries no
/// geometry.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Lattice {
    pub(super) size: Vec<usize>,
    data: Vec<f64>,
}

impl Lattice {
    fn zeroed(size: Vec<usize>) -> Self {
        let n = size.iter().product();
        Self {
            size,
            data: vec![0.0; n],
        }
    }

    fn linear_index(&self, index: &[usize]) -> usize {
        let mut offset = 0;
        let mut stride = 1;
        for (&i, &s) in index.iter().zip(&self.size) {
            offset += i * stride;
            stride *= s;
        }
        offset
    }

    /// `AddImageFilter` over two control-point lattices
    /// (`N4BiasFieldCorrectionImageFilter::UpdateBiasFieldEstimate`).
    pub(super) fn add_assign(&mut self, other: &Lattice) {
        debug_assert_eq!(self.size, other.size);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }
}

/// `BSplineScatteredDataPointSetToImageFilter::NumberToIndex`.
fn number_to_index(number: usize, size: &[usize]) -> Vec<usize> {
    let dim = size.len();
    let mut k = vec![1usize; dim];
    for i in 1..dim {
        k[i] = size[dim - i - 1] * k[i - 1];
    }
    let mut rem = number;
    let mut index = vec![0usize; dim];
    for i in 0..dim {
        index[dim - i - 1] = rem / k[dim - i - 1];
        rem %= k[dim - i - 1];
    }
    index
}

/// Unravel a linear offset into a first-index-fastest multi-index over a
/// hypercube of extent `extent` in every one of `dim` axes.
fn unravel_cube(mut linear: usize, extent: usize, dim: usize) -> Vec<usize> {
    (0..dim)
        .map(|_| {
            let i = linear % extent;
            linear /= extent;
            i
        })
        .collect()
}

// ---- scattered-data fitting -----------------------------------------------

/// Everything `fit` needs about the parametric domain and the samples.
pub(super) struct FitInput<'a> {
    /// Size of the image whose grid defines the parametric domain.
    pub(super) size: &'a [usize],
    pub(super) spacing: &'a [f64],
    /// Physical origin of that grid (`bspliner->SetOrigin(parametricOrigin)`).
    pub(super) origin: &'a [f64],
    pub(super) spline_order: usize,
    pub(super) number_of_control_points: &'a [usize],
    /// Sample coordinates, `points[n * dim + d]`, in the same physical frame
    /// as `origin`/`spacing`.
    pub(super) points: &'a [f64],
    pub(super) values: &'a [f64],
    /// `SetPointWeights` — one confidence weight per sample.
    pub(super) weights: &'a [f64],
}

/// Reparameterize each sample and validate the parametric domain, shared by
/// [`fit`] (`ThreadedGenerateDataForFitting`) and [`reconstruct`]
/// (`DynamicThreadedGenerateData`), which clamp identically.
fn clamp_to_parametric_domain(
    mut value: f64,
    axis: usize,
    spans: f64,
    epsilon: f64,
) -> Result<f64> {
    if (value - spans).abs() <= epsilon {
        value = spans - epsilon;
    }
    if value < 0.0 && value.abs() <= epsilon {
        value = 0.0;
    }
    if value < 0.0 || value >= spans {
        return Err(FilterError::BSplineParametricDomain { axis, value, spans });
    }
    Ok(value)
}

/// Per-axis reparameterization scale `r[d]` and the tolerance `epsilon[d]`
/// both filters build from it.
fn parametric_scale(
    spans: &[usize],
    size: &[usize],
    spacing: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    if size.iter().any(|&s| s < 2) {
        return Err(FilterError::BSplineAxisTooShort(size.to_vec()));
    }
    let r: Vec<f64> = (0..size.len())
        .map(|d| spans[d] as f64 / ((size[d] - 1) as f64 * spacing[d]))
        .collect();
    let epsilon: Vec<f64> = (0..size.len())
        .map(|d| r[d] * spacing[d] * BSPLINE_EPSILON)
        .collect();
    Ok((r, epsilon))
}

/// `BSplineScatteredDataPointSetToImageFilter`'s single-level, open-dimension
/// fit: `ThreadedGenerateDataForFitting` accumulating the delta/omega
/// lattices, then `AfterThreadedGenerateData` dividing them into the phi
/// lattice.
///
/// This is the Lee/Wolberg multilevel B-spline approximation with one level:
/// each sample distributes `w * B^3 / sum(B^2)` into `delta` and `w * B^2`
/// into `omega`, and `phi = delta / omega` wherever `omega` is nonzero.
///
/// `AfterThreadedGenerateData` gates the division on
/// `Math::NotAlmostEquals(omega, 0)`, a 4-ULP window around zero that only
/// subnormal accumulations can fall inside; this uses `omega != 0.0`, and the
/// non-finite guard ITK applies to the quotient catches the rest.
pub(super) fn fit(input: &FitInput<'_>) -> Result<Lattice> {
    let dim = input.size.len();
    let order = input.spline_order;
    if order == 0 {
        return Err(FilterError::InvalidSplineOrder);
    }
    for d in 0..dim {
        if input.number_of_control_points[d] < order + 1 {
            return Err(FilterError::InvalidControlPointCount {
                axis: d,
                control_points: input.number_of_control_points[d],
                spline_order: order,
            });
        }
    }

    let spans: Vec<usize> = (0..dim)
        .map(|d| input.number_of_control_points[d] - order)
        .collect();
    let (r, epsilon) = parametric_scale(&spans, input.size, input.spacing)?;

    let kernel = Kernel::new(order);
    let mut lattice = Lattice::zeroed(input.number_of_control_points.to_vec());
    let mut omega = vec![0.0; lattice.data.len()];
    let mut delta = vec![0.0; lattice.data.len()];

    let support = order + 1;
    let support_total = support.pow(dim as u32);
    let mut neighborhood_weights = vec![0.0; support_total];
    let mut p = vec![0.0; dim];

    for n in 0..input.values.len() {
        for d in 0..dim {
            p[d] = clamp_to_parametric_domain(
                (input.points[n * dim + d] - input.origin[d]) * r[d],
                d,
                spans[d] as f64,
                epsilon[d],
            )?;
        }

        let mut squared_weight_sum = 0.0;
        for (k, w) in neighborhood_weights.iter_mut().enumerate() {
            let idx = unravel_cube(k, support, dim);
            let mut b = 1.0;
            for d in 0..dim {
                // `p[i] - static_cast<unsigned int>(p[i]) - idx[i] + 0.5 * (order - 1)`.
                let u = p[d] - p[d].trunc() - idx[d] as f64 + 0.5 * (order as f64 - 1.0);
                b *= kernel.evaluate(u);
            }
            *w = b;
            squared_weight_sum += b * b;
        }

        let confidence = input.weights[n];
        let value = input.values[n];
        for (k, &t) in neighborhood_weights.iter().enumerate() {
            let idx = unravel_cube(k, support, dim);
            let lattice_index: Vec<usize> = (0..dim).map(|d| idx[d] + p[d] as usize).collect();
            let lin = lattice.linear_index(&lattice_index);
            omega[lin] += confidence * t * t;
            delta[lin] += value * (t * t * t * confidence / squared_weight_sum);
        }
    }

    for (phi, (&o, &d)) in lattice.data.iter_mut().zip(omega.iter().zip(&delta)) {
        if o != 0.0 {
            let q = d / o;
            if q.is_finite() {
                *phi = q;
            }
        }
    }
    Ok(lattice)
}

// ---- reconstruction -------------------------------------------------------

/// `CollapsePhiLattice` applied along one axis at a time, with ITK's
/// raster-order cache: `buf[j]` holds the lattice collapsed along axes
/// `j..dim`, so advancing only the fastest coordinate re-collapses only axis 0.
struct Collapser<'a> {
    kernel: &'a Kernel,
    spline_order: usize,
    lattice_size: &'a [usize],
    /// `buf[j].len() == prod(lattice_size[..j])`; `buf[dim]` is the lattice.
    buf: Vec<Vec<f64>>,
    current_u: Vec<f64>,
}

impl<'a> Collapser<'a> {
    fn new(lattice: &'a Lattice, kernel: &'a Kernel, spline_order: usize) -> Collapser<'a> {
        let dim = lattice.size.len();
        let mut buf: Vec<Vec<f64>> = (0..dim)
            .map(|j| vec![0.0; lattice.size[..j].iter().product()])
            .collect();
        buf.push(lattice.data.clone());
        Collapser {
            kernel,
            spline_order,
            lattice_size: &lattice.size,
            buf,
            current_u: vec![-1.0; dim],
        }
    }

    /// `CollapsePhiLattice(buf[j + 1], buf[j], u, j)`. `buf[j]`'s linear index
    /// `l` runs over axes `0..j`, so the source element for lattice index
    /// `base + i` along axis `j` sits at `l + buf[j].len() * (base + i)`.
    fn collapse(&mut self, j: usize, u: f64) {
        let base = u as usize;
        let b: Vec<f64> = (0..=self.spline_order)
            .map(|i| {
                let idx = base + i;
                let v = u - idx as f64 + 0.5 * (self.spline_order as f64 - 1.0);
                self.kernel.evaluate(v)
            })
            .collect();

        let (dst_side, src_side) = self.buf.split_at_mut(j + 1);
        let dst = &mut dst_side[j];
        let src = &src_side[0];
        let dst_len = dst.len();

        for (l, out) in dst.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, &bi) in b.iter().enumerate() {
                acc += src[l + dst_len * (base + i)] * bi;
            }
            *out = acc;
        }
    }

    /// The lattice evaluated at parametric coordinate `u`.
    fn value(&mut self, u: &[f64]) -> f64 {
        debug_assert_eq!(u.len(), self.lattice_size.len());
        for i in (0..u.len()).rev() {
            if u[i] != self.current_u[i] {
                for j in (0..=i).rev() {
                    self.collapse(j, u[j]);
                    self.current_u[j] = u[j];
                }
                break;
            }
        }
        self.buf[0][0]
    }
}

/// `BSplineControlPointImageFilter::DynamicThreadedGenerateData`: sample the
/// control-point lattice onto a grid of `size` voxels with `spacing`.
///
/// The parametric coordinate of voxel `idx` is
/// `U[d] = spans[d] * idx[d] / (size[d] - 1)`, so neither the grid's origin
/// nor its direction enters the sampled values; `spacing` only scales the
/// domain-edge tolerance `epsilon`.
pub(super) fn reconstruct(
    lattice: &Lattice,
    spline_order: usize,
    size: &[usize],
    spacing: &[f64],
) -> Result<Vec<f64>> {
    let dim = size.len();
    if spline_order == 0 {
        return Err(FilterError::InvalidSplineOrder);
    }
    for d in 0..dim {
        if lattice.size[d] < spline_order + 1 {
            return Err(FilterError::InvalidControlPointCount {
                axis: d,
                control_points: lattice.size[d],
                spline_order,
            });
        }
    }
    let spans: Vec<usize> = (0..dim).map(|d| lattice.size[d] - spline_order).collect();
    let (_, epsilon) = parametric_scale(&spans, size, spacing)?;

    let kernel = Kernel::new(spline_order);
    let mut collapser = Collapser::new(lattice, &kernel, spline_order);

    let total: usize = size.iter().product();
    let mut out = vec![0.0; total];
    let mut u = vec![0.0; dim];
    let mut idx = vec![0usize; dim];
    for (lin, slot) in out.iter_mut().enumerate() {
        let mut rem = lin;
        for d in 0..dim {
            idx[d] = rem % size[d];
            rem /= size[d];
        }
        for d in 0..dim {
            u[d] = clamp_to_parametric_domain(
                spans[d] as f64 * idx[d] as f64 / (size[d] - 1) as f64,
                d,
                spans[d] as f64,
                epsilon[d],
            )?;
        }
        *slot = collapser.value(&u);
    }
    Ok(out)
}

// ---- dyadic refinement ----------------------------------------------------

/// `BSplineControlPointImageFilter::RefineControlPointLattice`, restricted to
/// open dimensions.
///
/// `number_of_levels[d] == 2` doubles the control-point count along axis `d`
/// (`2 * n - spline_order`); `1` leaves it alone. The refined lattice
/// represents exactly the same spline.
///
/// Two upstream quirks are preserved. First, the psi bound check compares
/// against `input->GetLargestPossibleRegion().GetSize()` — the size of the
/// lattice passed in — rather than against the size of the current `psi`
/// lattice, so it is only correct while `psi` still has the input's size, i.e.
/// for the first refinement level. N4 only ever asks for two levels, so it
/// only ever reaches `m == 1`. Second, the iteration visits only all-even
/// refined indices and writes the `2^dim` neighbours of each; along an axis
/// that is *not* being refined this still steps by two, which would leave
/// holes for a non-uniform `number_of_levels`. N4 always passes a uniform
/// array.
pub(super) fn refine(
    lattice: &Lattice,
    spline_order: usize,
    number_of_levels: &[usize],
) -> Lattice {
    let dim = lattice.size.len();
    let maximum_number_of_levels = number_of_levels.iter().copied().max().unwrap_or(1);
    if maximum_number_of_levels <= 1 {
        return lattice.clone();
    }

    let coefficients = refined_lattice_coefficients(spline_order);
    let support = spline_order + 1;
    let support_total = support.pow(dim as u32);
    let corner_total = 1usize << dim;
    let corner_extent = vec![2usize; dim];
    let support_extent = vec![support; dim];

    let mut psi = lattice.clone();
    for m in 1..maximum_number_of_levels {
        let new_control_points: Vec<usize> = (0..dim)
            .map(|d| {
                if m < number_of_levels[d] {
                    2 * psi.size[d] - spline_order
                } else {
                    psi.size[d]
                }
            })
            .collect();
        let mut refined = Lattice::zeroed(new_control_points.clone());

        let even_counts: Vec<usize> = refined.size.iter().map(|&s| s.div_ceil(2)).collect();
        let even_total: usize = even_counts.iter().product();
        for e in 0..even_total {
            let mut rem = e;
            let mut idx = vec![0usize; dim];
            for d in 0..dim {
                idx[d] = 2 * (rem % even_counts[d]);
                rem /= even_counts[d];
            }
            let idx_psi: Vec<usize> = (0..dim)
                .map(|d| {
                    if m < number_of_levels[d] {
                        idx[d] / 2
                    } else {
                        idx[d]
                    }
                })
                .collect();

            for i in 0..corner_total {
                let off = number_to_index(i, &corner_extent);
                let mut tmp = vec![0usize; dim];
                let mut out_of_boundary = false;
                for j in 0..dim {
                    tmp[j] = idx[j] + off[j];
                    if tmp[j] >= new_control_points[j] {
                        out_of_boundary = true;
                        break;
                    }
                }
                if out_of_boundary {
                    continue;
                }

                let mut sum = 0.0;
                for j in 0..support_total {
                    let off_psi = number_to_index(j, &support_extent);
                    let mut tmp_psi = vec![0usize; dim];
                    let mut out_of_boundary = false;
                    for k in 0..dim {
                        tmp_psi[k] = idx_psi[k] + off_psi[k];
                        // Upstream compares against the *input* lattice size.
                        if tmp_psi[k] >= lattice.size[k] {
                            out_of_boundary = true;
                            break;
                        }
                    }
                    if out_of_boundary {
                        continue;
                    }
                    let mut coeff = 1.0;
                    for k in 0..dim {
                        coeff *= coefficients[off[k]][off_psi[k]];
                    }
                    sum += psi.data[psi.linear_index(&tmp_psi)] * coeff;
                }
                let lin = refined.linear_index(&tmp);
                refined.data[lin] = sum;
            }
        }
        psi = refined;
    }
    psi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol})");
    }

    /// `GetShapeFunctionsInZeroToOneInterval()` for the cubic kernel, in exact
    /// rationals (rows are polynomials, highest degree first).
    #[test]
    fn cubic_shape_functions_in_zero_to_one_interval_match_cox_de_boor() {
        let c = shape_functions_in_zero_to_one_interval(3);
        let expected = [
            [-1.0 / 6.0, 1.0 / 2.0, -1.0 / 2.0, 1.0 / 6.0],
            [1.0 / 2.0, -1.0, 0.0, 2.0 / 3.0],
            [-1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 / 6.0],
            [1.0 / 6.0, 0.0, 0.0, 0.0],
        ];
        assert_eq!(c.len(), 4);
        for (row, want) in c.iter().zip(&expected) {
            assert_eq!(row.len(), 4);
            for (&got, &w) in row.iter().zip(want) {
                approx(got, w, 1e-15);
            }
        }
    }

    /// The refinement coefficients are the classical Lee/Wolberg subdivision
    /// masks: cubic gives `1/2 [1, 1]` and `1/8 [1, 6, 1]`.
    #[test]
    fn refined_lattice_coefficients_are_the_subdivision_masks() {
        approx_matrix(
            &refined_lattice_coefficients(1),
            &[&[1.0, 0.0], &[0.5, 0.5]],
        );
        approx_matrix(
            &refined_lattice_coefficients(2),
            &[&[0.75, 0.25, 0.0], &[0.25, 0.75, 0.0]],
        );
        approx_matrix(
            &refined_lattice_coefficients(3),
            &[&[0.5, 0.5, 0.0, 0.0], &[0.125, 0.75, 0.125, 0.0]],
        );
        approx_matrix(
            &refined_lattice_coefficients(4),
            &[
                &[5.0 / 16.0, 5.0 / 8.0, 1.0 / 16.0, 0.0, 0.0],
                &[1.0 / 16.0, 5.0 / 8.0, 5.0 / 16.0, 0.0, 0.0],
            ],
        );
    }

    fn approx_matrix(got: &[Vec<f64>], want: &[&[f64]]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.len(), w.len());
            for (&a, &b) in g.iter().zip(w.iter()) {
                approx(a, b, 1e-12);
            }
        }
    }

    /// Orders 1-3 take the closed-form branch and orders >= 4 the Cox-de Boor
    /// branch; both must reproduce a partition of unity on the integers.
    #[test]
    fn kernels_form_a_partition_of_unity() {
        for order in 1..=5usize {
            let kernel = Kernel::new(order);
            for step in 0..8 {
                let x = 0.125 * step as f64;
                let sum: f64 = (-6i32..=6).map(|k| kernel.evaluate(x - k as f64)).sum();
                approx(sum, 1.0, 1e-12);
            }
        }
    }

    /// The Cox-de Boor branch and the closed forms agree where both are
    /// defined (orders 1-3), which is what lets `Kernel` dispatch on order.
    #[test]
    fn cox_de_boor_agrees_with_the_closed_forms() {
        for order in 1..=3usize {
            let closed = Kernel::new(order);
            let generic = Kernel {
                spline_order: order,
                shape_functions: centered_shape_functions(order),
            };
            for step in -400..=400 {
                let u = 0.01 * step as f64;
                approx(closed.evaluate(u), generic.evaluate(u), 1e-12);
            }
        }
    }

    fn line_lattice(n: usize, a: f64, b: f64) -> Lattice {
        Lattice {
            size: vec![n],
            data: (0..n).map(|m| a * (m as f64 - 1.0) + b).collect(),
        }
    }

    /// A cubic B-spline whose control points sample `a*(m-1) + b` reproduces
    /// the line `a*u + b` exactly, because `sum_n B3(x-n) == 1` and
    /// `sum_n n*B3(x-n) == x`. This pins the parametric mapping
    /// `U[d] = spans[d] * idx[d] / (size[d] - 1)` and the collapse cascade.
    #[test]
    fn reconstruct_reproduces_a_linear_ramp() {
        let lattice = line_lattice(7, 2.0, -0.5);
        let size = [9usize];
        let spacing = [1.0];
        let out = reconstruct(&lattice, 3, &size, &spacing).unwrap();
        let spans = (7 - 3) as f64;
        for (i, &v) in out.iter().enumerate() {
            // The last voxel is nudged inside the domain by `epsilon`.
            let mut u = spans * i as f64 / (size[0] - 1) as f64;
            let eps = spans / (size[0] - 1) as f64 * BSPLINE_EPSILON;
            if (u - spans).abs() <= eps {
                u = spans - eps;
            }
            approx(v, 2.0 * u - 0.5, 1e-12);
        }
    }

    /// Constant control points reconstruct to that constant (partition of
    /// unity), in 1-D, 2-D and 3-D.
    #[test]
    fn reconstruct_of_a_constant_lattice_is_that_constant() {
        for dim in 1..=3usize {
            let lattice = Lattice {
                size: vec![5; dim],
                data: vec![1.25; 5usize.pow(dim as u32)],
            };
            let size = vec![7usize; dim];
            let spacing = vec![0.5; dim];
            for v in reconstruct(&lattice, 3, &size, &spacing).unwrap() {
                approx(v, 1.25, 1e-12);
            }
        }
    }

    /// Dyadic refinement is exact: the refined lattice represents the same
    /// spline, so sampling it on the same grid reproduces the same values.
    #[test]
    fn refinement_preserves_the_spline() {
        for order in 1..=3usize {
            for dim in 1..=2usize {
                let n = order + 3;
                let count = n.pow(dim as u32);
                let lattice = Lattice {
                    size: vec![n; dim],
                    data: (0..count).map(|i| (i as f64 * 0.37).sin()).collect(),
                };
                let size = vec![11usize; dim];
                let spacing = vec![1.0; dim];
                let coarse = reconstruct(&lattice, order, &size, &spacing).unwrap();

                let refined = refine(&lattice, order, &vec![2; dim]);
                for d in 0..dim {
                    assert_eq!(refined.size[d], 2 * n - order);
                }
                let fine = reconstruct(&refined, order, &size, &spacing).unwrap();
                for (a, b) in coarse.iter().zip(&fine) {
                    approx(*a, *b, 1e-9);
                }
            }
        }
    }

    /// `number_of_levels` of all ones is the identity.
    #[test]
    fn refinement_with_one_level_is_a_copy() {
        let lattice = Lattice {
            size: vec![4, 4],
            data: (0..16).map(|i| i as f64).collect(),
        };
        assert_eq!(refine(&lattice, 3, &[1, 1]), lattice);
    }

    /// A single sample at parametric coordinate `p` gives a lattice with the
    /// closed form `phi_k = value * B_k / sum_j B_j^2`, independent of the
    /// sample's weight (the weight cancels between `delta` and `omega`).
    #[test]
    fn single_sample_fit_matches_the_closed_form() {
        let size = [5usize];
        let spacing = [1.0];
        let origin = [0.0];
        // Voxel 1 of 5 with 4 control points, order 3 => one span, p = 0.25.
        let points = [1.0];
        let values = [3.0];
        let weights = [0.75];
        let lattice = fit(&FitInput {
            size: &size,
            spacing: &spacing,
            origin: &origin,
            spline_order: 3,
            number_of_control_points: &[4],
            points: &points,
            values: &values,
            weights: &weights,
        })
        .unwrap();

        let kernel = Kernel::new(3);
        let p = 0.25f64;
        let b: Vec<f64> = (0..4)
            .map(|k| kernel.evaluate(p - p.trunc() - k as f64 + 1.0))
            .collect();
        let b2: f64 = b.iter().map(|x| x * x).sum();
        for (&phi, &bk) in lattice.data.iter().zip(&b) {
            approx(phi, 3.0 * bk / b2, 1e-12);
        }
    }

    /// A zero-weighted sample contributes nothing to either accumulator, so
    /// dropping it leaves the lattice bit-identical.
    #[test]
    fn zero_weight_samples_are_ignored() {
        let size = [5usize, 5];
        let spacing = [1.0, 1.0];
        let origin = [0.0, 0.0];
        let base_points = [1.0, 1.0, 2.0, 3.0, 3.0, 2.0];
        let base_values = [1.0, -2.0, 0.5];
        let base_weights = [1.0, 1.0, 1.0];

        let mut points = base_points.to_vec();
        points.extend_from_slice(&[2.0, 2.0]);
        let mut values = base_values.to_vec();
        values.push(1.0e6);
        let mut weights = base_weights.to_vec();
        weights.push(0.0);

        let with_zero = fit(&FitInput {
            size: &size,
            spacing: &spacing,
            origin: &origin,
            spline_order: 3,
            number_of_control_points: &[4, 4],
            points: &points,
            values: &values,
            weights: &weights,
        })
        .unwrap();
        let without = fit(&FitInput {
            size: &size,
            spacing: &spacing,
            origin: &origin,
            spline_order: 3,
            number_of_control_points: &[4, 4],
            points: &base_points,
            values: &base_values,
            weights: &base_weights,
        })
        .unwrap();
        assert_eq!(with_zero, without);
    }

    /// Weights scale a sample's influence: doubling every weight leaves
    /// `delta / omega` unchanged.
    #[test]
    fn uniformly_scaling_the_weights_leaves_the_lattice_unchanged() {
        let size = [6usize];
        let spacing = [1.0];
        let origin = [0.0];
        let points = [0.0, 2.0, 4.0];
        let values = [1.0, 2.0, -1.0];
        let unit = fit(&FitInput {
            size: &size,
            spacing: &spacing,
            origin: &origin,
            spline_order: 2,
            number_of_control_points: &[5],
            points: &points,
            values: &values,
            weights: &[1.0, 1.0, 1.0],
        })
        .unwrap();
        let doubled = fit(&FitInput {
            size: &size,
            spacing: &spacing,
            origin: &origin,
            spline_order: 2,
            number_of_control_points: &[5],
            points: &points,
            values: &values,
            weights: &[2.0, 2.0, 2.0],
        })
        .unwrap();
        for (a, b) in unit.data.iter().zip(&doubled.data) {
            approx(*a, *b, 1e-13);
        }
    }

    /// `fit` is linear in the sample values.
    #[test]
    fn fit_is_linear_in_the_sample_values() {
        let size = [8usize, 6];
        let spacing = [1.0, 2.0];
        let origin = [-3.0, 4.0];
        let points: Vec<f64> = (0..8)
            .flat_map(|i| [origin[0] + i as f64 * spacing[0], origin[1] + spacing[1]])
            .collect();
        let values: Vec<f64> = (0..8).map(|i| (i as f64 * 0.9).cos()).collect();
        let scaled: Vec<f64> = values.iter().map(|v| 3.0 * v).collect();
        let weights = vec![1.0; 8];
        let mk = |vals: &[f64]| {
            fit(&FitInput {
                size: &size,
                spacing: &spacing,
                origin: &origin,
                spline_order: 3,
                number_of_control_points: &[4, 4],
                points: &points,
                values: vals,
                weights: &weights,
            })
            .unwrap()
        };
        let a = mk(&values);
        let b = mk(&scaled);
        for (x, y) in a.data.iter().zip(&b.data) {
            approx(3.0 * x, *y, 1e-12);
        }
    }

    /// Control points with no sample support keep `phi == 0` rather than
    /// dividing by a zero `omega`.
    #[test]
    fn unsupported_control_points_stay_zero() {
        let size = [20usize];
        let spacing = [1.0];
        let origin = [0.0];
        // Everything sits in the first span of a 3-span lattice.
        let lattice = fit(&FitInput {
            size: &size,
            spacing: &spacing,
            origin: &origin,
            spline_order: 1,
            number_of_control_points: &[4],
            points: &[0.0, 1.0],
            values: &[1.0, 1.0],
            weights: &[1.0, 1.0],
        })
        .unwrap();
        assert_eq!(lattice.data[3], 0.0);
        assert!(lattice.data[0] != 0.0);
    }

    #[test]
    fn spline_order_zero_is_rejected() {
        let err = fit(&FitInput {
            size: &[4],
            spacing: &[1.0],
            origin: &[0.0],
            spline_order: 0,
            number_of_control_points: &[4],
            points: &[0.0],
            values: &[0.0],
            weights: &[1.0],
        })
        .unwrap_err();
        assert_eq!(err, FilterError::InvalidSplineOrder);
        assert_eq!(
            reconstruct(&Lattice::zeroed(vec![4]), 0, &[4], &[1.0]).unwrap_err(),
            FilterError::InvalidSplineOrder
        );
    }

    #[test]
    fn too_few_control_points_is_rejected() {
        let err = fit(&FitInput {
            size: &[4],
            spacing: &[1.0],
            origin: &[0.0],
            spline_order: 3,
            number_of_control_points: &[3],
            points: &[0.0],
            values: &[0.0],
            weights: &[1.0],
        })
        .unwrap_err();
        assert_eq!(
            err,
            FilterError::InvalidControlPointCount {
                axis: 0,
                control_points: 3,
                spline_order: 3,
            }
        );
    }

    #[test]
    fn a_single_pixel_axis_is_rejected() {
        let err = reconstruct(&Lattice::zeroed(vec![4, 4]), 3, &[5, 1], &[1.0, 1.0]).unwrap_err();
        assert_eq!(err, FilterError::BSplineAxisTooShort(vec![5, 1]));
    }

    /// A sample beyond the parametric domain is an error, not a silent clamp;
    /// only offsets within `epsilon` of an edge are pulled back in.
    #[test]
    fn samples_outside_the_parametric_domain_are_rejected() {
        let err = fit(&FitInput {
            size: &[5],
            spacing: &[1.0],
            origin: &[0.0],
            spline_order: 3,
            number_of_control_points: &[4],
            points: &[5.0], // one voxel past the grid's last index
            values: &[1.0],
            weights: &[1.0],
        })
        .unwrap_err();
        assert!(matches!(
            err,
            FilterError::BSplineParametricDomain { axis: 0, .. }
        ));
    }

    /// The grid's far edge maps exactly onto `spans` and is pulled back inside
    /// by `epsilon`, so the last voxel fits rather than throwing.
    #[test]
    fn the_far_domain_edge_is_pulled_inside_by_epsilon() {
        let lattice = fit(&FitInput {
            size: &[5],
            spacing: &[1.0],
            origin: &[0.0],
            spline_order: 3,
            number_of_control_points: &[4],
            points: &[4.0],
            values: &[1.0],
            weights: &[1.0],
        })
        .unwrap();
        assert!(lattice.data[3] != 0.0);
    }

    #[test]
    fn number_to_index_walks_the_first_axis_fastest() {
        assert_eq!(number_to_index(0, &[2, 2]), vec![0, 0]);
        assert_eq!(number_to_index(1, &[2, 2]), vec![1, 0]);
        assert_eq!(number_to_index(2, &[2, 2]), vec![0, 1]);
        assert_eq!(number_to_index(3, &[2, 2]), vec![1, 1]);
        assert_eq!(number_to_index(5, &[4, 4, 4]), vec![1, 1, 0]);
    }
}
