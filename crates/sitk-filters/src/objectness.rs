//! `ObjectnessMeasureImageFilter`: Antiga's generalization of Frangi's
//! vesselness, enhancing `M`-dimensional objects in `N`-dimensional images.
//!
//! Ported from the composite
//! `ITKSimpleITKFilters/include/itkObjectnessMeasureImageFilter.h(.hxx)`,
//! whose `GenerateData` (`.hxx:52-91`) is a two-filter mini-pipeline:
//!
//! 1. `itk::HessianImageFilter` (`ITKSimpleITKFilters/include/itkHessianImageFilter.hxx`)
//!    — a *discrete central-difference* Hessian. **Not**
//!    `HessianRecursiveGaussianImageFilter`: no Gaussian is convolved here at
//!    all. The class doc (`itkObjectnessMeasureImageFilter.h:37-40`) says so
//!    explicitly — "Internally, it computes a Hessian via discrete central
//!    differences. Before applying this filter it is expected that a Gaussian
//!    smoothing filter at an appropriate scale (sigma) was applied to the
//!    input image." So this port needs none of the crate's
//!    [`crate::recursive_gaussian`] machinery; callers who want the
//!    multi-scale Frangi behavior smooth the input themselves first.
//! 2. `itk::HessianToObjectnessMeasureImageFilter`
//!    (`ITK/Modules/Filtering/ImageFeature/include/itkHessianToObjectnessMeasureImageFilter.hxx`)
//!    — the eigenvalue analysis and the `R_A` / `R_B` / `S` formula.
//!
//! The composite forwards `Alpha`, `Beta`, `Gamma`, `ScaleObjectnessMeasure`,
//! `ObjectDimension` and `BrightObject` straight through
//! (`itkObjectnessMeasureImageFilter.hxx:75-80`) and holds no state of its
//! own beyond those six defaults, which match
//! `ObjectnessMeasureImageFilter.yaml` exactly — see
//! [`ObjectnessMeasureSettings::default`].
//!
//! # The Hessian
//!
//! `HessianImageFilter::DynamicThreadedGenerateData`
//! (`itkHessianImageFilter.hxx:198-277`) fills a symmetric tensor per pixel:
//!
//! ```text
//! H(i,i) = ( f(x + e_i) + f(x - e_i) - 2 f(x) ) / s_i^2          (hxx:256-257)
//! H(i,j) = ( f(x - e_i - e_j) - f(x - e_i + e_j)
//!          - f(x + e_i - e_j) + f(x + e_i + e_j) )
//!          / (4 s_i s_j)                                          (hxx:265-267)
//! ```
//!
//! Spacing is *always* applied — the filter has no `UseImageSpacing` toggle.
//!
//! Out-of-image samples come from the default boundary condition of
//! `ConstNeighborhoodIterator`, which is `ZeroFluxNeumannBoundaryCondition`
//! (`itkConstNeighborhoodIterator.h:52`); it is switched on precisely when
//! the radius-1 neighborhood pokes outside the buffered region
//! (`itkConstNeighborhoodIterator.hxx:398-418`), which for this filter is the
//! whole image because `ObjectnessMeasureImageFilter::EnlargeOutputRequestedRegion`
//! (`.hxx:47-51`) forces the largest possible region. Zero-flux Neumann
//! clamps each out-of-range coordinate to the nearest in-image one, so at a
//! face `f(x - e_i)` collapses to `f(x)` and `H(i,i)` degenerates to a
//! one-sided first difference.
//!
//! # Precision
//!
//! `HessianImageFilter`'s output pixel is
//! `SymmetricSecondRankTensor<NumericTraits<PixelType>::RealType, N>`
//! (`itkHessianImageFilter.h:43-44`), and `NumericTraits<float>::RealType` is
//! `double` (`itkNumericTraits.h:1356`), so the tensor is `double` for both
//! `Float32` and `Float64` inputs. The *arithmetic that fills it* is not
//! uniformly `double`, though:
//!
//! - `H(i,i)`: `it.GetPixel(+) + it.GetPixel(-)` is a `float + float` add for
//!   a `Float32` input, rounded to `float` before `- 2.0 * it.GetPixel(x)`
//!   promotes the expression to `double` (`2.0` is a `double` literal).
//! - `H(i,j)`: all four `GetPixel` values are `float`, and `a - b - c + d`
//!   evaluates left-to-right entirely in `float` (three roundings) before the
//!   `double` division by `4 s_i s_j`.
//!
//! [`hessian_at`] reproduces both roundings exactly (`narrow_f32`), rather
//! than computing throughout in `f64`: `f32 + f32` and `f32 - f32` are exact
//! in `f64`, so rounding the `f64` result once to `f32` equals the `f32`
//! operation.
//!
//! # Eigenvalue ordering
//!
//! `HessianToObjectnessMeasureImageFilter` uses
//! `SymmetricEigenAnalysisFixedDimension<N, InputPixelType, EigenValueArrayType>`
//! with `EigenValueType = double` (`.h:75-76`) and its default
//! `m_OrderEigenValues = OrderByValue` (`itkSymmetricEigenAnalysis.h:871`),
//! i.e. **ascending by signed value**. It then re-sorts *by magnitude,
//! retaining sign* with `std::sort(..., AbsLessCompare())` (`.hxx:245-246`,
//! comparator at `.h:145-152`), producing `|e_1| <= ... <= |e_N|`.
//!
//! This port calls `linalg::symmetric_eigen` (ascending by value, same as
//! `OrderByValue`) and then a *stable* sort by `|.|`. See "Upstream findings"
//! below for why that is the right reading of a `std::sort`.
//!
//! # The measure
//!
//! With `d = ObjectDimension`, `N = ImageDimension`, and `a_0 <= ... <=
//! a_{N-1}` the magnitude-sorted absolute eigenvalues (`.hxx:268-334`):
//!
//! ```text
//! sign constraint:  for i in d..N:  bright => e_i <= 0,  dark => e_i >= 0
//!                   (violated => output 0)
//!
//! V = 1
//! if d < N-1:  R_A = a_d / (prod_{j=d+1..N-1} a_j)^(1/(N-d-1))
//!              V *= 1 - exp(-R_A^2 / (2 alpha^2))
//! if d > 0:    R_B = a_{d-1} / (prod_{j=d..N-1} a_j)^(1/(N-d))
//!              V *= exp(-R_B^2 / (2 beta^2))
//! always:      S^2 = sum_i a_i^2
//!              V *= 1 - exp(-S^2 / (2 gamma^2))
//! if ScaleObjectnessMeasure: V *= a_{N-1}
//! ```
//!
//! For `N = 3, d = 1` (Frangi's vesselness) this is
//! `R_A = a_1/a_2`, `R_B = a_0/sqrt(a_1 a_2)`, `S = ||H||_F`.
//!
//! `BrightObject` never negates anything: it only flips which sign the
//! `N - d` largest-magnitude eigenvalues must have. The magnitudes drive the
//! formula either way.
//!
//! # Upstream findings
//!
//! 1. **The class doc's formula is wrong, in ITK and in the SimpleITK yaml.**
//!    `itkObjectnessMeasureImageFilter.h:47-52` (copied verbatim into
//!    `ObjectnessMeasureImageFilter.yaml`'s `detaileddescription`) prints
//!    `R_B = |λ₂| / |λ₂ λ₃|` and `V = (1 - e^{-R_A²/2α²}) · e^{R_B²/2β²} ·
//!    (1 - e^{-S²/2γ²})`. Two errors: `R_B`'s numerator is `|λ₁|` and its
//!    denominator is `sqrt(|λ₂ λ₃|)` (`.hxx:302-311`), and the `R_B`
//!    exponential is *negated* — `std::exp(-0.5 * sqr(rB) / sqr(m_Beta))`
//!    (`.hxx:312`). As printed, `V` would grow without bound in `R_B`.
//!    The code, not the doc, is what this port implements.
//!
//! 2. **`Alpha == 0` and `Beta == 0` are handled asymmetrically.** With a
//!    nonzero denominator, `alpha == 0` skips the `R_A` factor entirely
//!    (`.hxx:288-292` nests the `alpha` test *inside* the denominator test),
//!    leaving it at `1` — which is the correct limit,
//!    `lim_{α→0} 1 - e^{-R_A²/2α²} = 1` for `R_A > 0`. But `beta == 0` is
//!    OR-ed into the denominator test (`.hxx:308`) and drives the whole
//!    measure to `0` — correct as a limit for `R_B > 0`, wrong for
//!    `R_B == 0`, whose limit is `1`. Reproduced as written.
//!
//! 3. **`std::sort` is not stable, so a magnitude tie is unspecified.** At a
//!    pixel with `e_i = -c` and `e_j = +c`, `.hxx:246`'s `std::sort` may
//!    leave them in either order, and `.hxx:252` then reads
//!    `sortedEigenValues[i]` to test the sign constraint — so whether the
//!    pixel outputs `0` or a positive measure depends on the standard
//!    library. libstdc++'s `std::sort` dispatches to `__insertion_sort` for
//!    ranges of at most 16 elements, which *is* stable, so in practice the
//!    ascending-by-value order coming out of `SymmetricEigenAnalysis`
//!    survives the tie and the negative eigenvalue lands first. This port
//!    pins that behavior with a stable sort ([`sort_by_magnitude`]).
//!
//! 4. **The sign constraint admits zero eigenvalues.** `.hxx:252` tests
//!    `sortedEigenValues[i] > 0.0` (bright) / `< 0.0` (dark), so an exactly
//!    zero eigenvalue satisfies *both*. The class doc's `λ₂ < 0 and λ₃ < 0`
//!    is therefore also inexact.
//!
//! 5. **The composite validates nothing.** `ObjectDimension >= ImageDimension`
//!    is caught by the *inner* filter's `VerifyPreconditions`
//!    (`itkHessianToObjectnessMeasureImageFilter.hxx:210-217`), i.e. only at
//!    `Update()` time. This port raises
//!    [`FilterError::InvalidObjectDimension`] up front, which is observably
//!    the same.
//!
//! 6. **Not streamable.** `ObjectnessMeasureImageFilter::EnlargeOutputRequestedRegion`
//!    (`.hxx:47-51`) unconditionally requests the largest possible region,
//!    even though `HessianImageFilter` is tagged `\ingroup Streamed` and pads
//!    its input requested region by 1 (`itkHessianImageFilter.hxx:144-193`).

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use crate::linalg::{MAX_DIM, Mat, symmetric_eigen};
use sitk_core::{Image, PixelId};

/// Parameters of [`objectness_measure`], defaulting to
/// `ObjectnessMeasureImageFilter.yaml`'s members and to
/// `itkObjectnessMeasureImageFilter.hxx:29-36`'s constructor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectnessMeasureSettings {
    /// Weight of `R_A`, the ratio of the smallest eigenvalue that has to be
    /// large to the larger ones. Default `0.5`.
    pub alpha: f64,
    /// Weight of `R_B`, the ratio of the largest eigenvalue that has to be
    /// small to the larger ones. Default `0.5`.
    pub beta: f64,
    /// Weight of `S`, the Frobenius norm of the Hessian (second-order
    /// structureness). Default `5.0`.
    pub gamma: f64,
    /// Multiply the measure by the largest absolute eigenvalue. Default
    /// `true`.
    pub scale_objectness_measure: bool,
    /// Dimensionality of the sought object: `0` blobs, `1` vessels, `2`
    /// plates, `3` hyper-plates. Must be less than the image dimension.
    /// Default `1`.
    pub object_dimension: u32,
    /// Enhance bright structures on a dark background (`true`, the vesselness
    /// convention) or the opposite. Default `true`.
    pub bright_object: bool,
}

impl Default for ObjectnessMeasureSettings {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
            gamma: 5.0,
            scale_objectness_measure: true,
            object_dimension: 1,
            bright_object: true,
        }
    }
}

/// `float`-rounds `v` when the input image's pixel type is `Float32`,
/// reproducing a C++ `float`-typed intermediate.
fn narrow_f32(v: f64, narrow: bool) -> f64 {
    if narrow { v as f32 as f64 } else { v }
}

/// One pixel's central-difference Hessian, with zero-flux Neumann sampling
/// outside the image. `idx` is the pixel's per-axis index; `strides[0] == 1`.
///
/// `narrow` reproduces the `float`-typed intermediates of a `Float32` input;
/// see the module docs.
fn hessian_at(
    vals: &[f64],
    size: &[usize],
    strides: &[usize],
    spacing: &[f64],
    idx: &[usize],
    narrow: bool,
) -> Mat {
    let dim = size.len();
    // ZeroFluxNeumannBoundaryCondition: clamp each coordinate into the image.
    let at = |off: &[isize]| -> f64 {
        let mut lin = 0usize;
        for d in 0..dim {
            let c = (idx[d] as isize + off[d]).clamp(0, size[d] as isize - 1) as usize;
            lin += c * strides[d];
        }
        vals[lin]
    };

    let mut off = [0isize; MAX_DIM];
    let center = at(&off[..dim]);

    let mut h: Mat = [[0.0; MAX_DIM]; MAX_DIM];
    for i in 0..dim {
        off[i] = 1;
        let plus = at(&off[..dim]);
        off[i] = -1;
        let minus = at(&off[..dim]);
        off[i] = 0;
        let sum = narrow_f32(plus + minus, narrow);
        h[i][i] = (sum - 2.0 * center) / (spacing[i] * spacing[i]);
    }

    for i in 0..dim.saturating_sub(1) {
        for j in i + 1..dim {
            let mut sample = |a: isize, b: isize| {
                off[i] = a;
                off[j] = b;
                let v = at(&off[..dim]);
                off[i] = 0;
                off[j] = 0;
                v
            };
            let mm = sample(-1, -1);
            let mp = sample(-1, 1);
            let pm = sample(1, -1);
            let pp = sample(1, 1);
            let mut t = narrow_f32(mm - mp, narrow);
            t = narrow_f32(t - pm, narrow);
            t = narrow_f32(t + pp, narrow);
            let v = t / (4.0 * spacing[i] * spacing[j]);
            h[i][j] = v;
            h[j][i] = v;
        }
    }
    h
}

/// Sorts eigenvalues by `|.|`, retaining sign — `std::sort(..., AbsLessCompare())`.
///
/// Stable, which pins the tie order upstream leaves unspecified; see upstream
/// finding 3 in the module docs. `eigen` arrives ascending by signed value, so
/// a `-c` / `+c` tie keeps `-c` first.
fn sort_by_magnitude(eigen: &[f64]) -> Vec<f64> {
    let mut sorted = eigen.to_vec();
    sorted.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    sorted
}

/// The per-pixel objectness of `HessianToObjectnessMeasureImageFilter::DynamicThreadedGenerateData`
/// (`.hxx:237-341`), given the magnitude-sorted *signed* eigenvalues.
fn objectness_from_sorted_eigenvalues(sorted: &[f64], settings: &ObjectnessMeasureSettings) -> f64 {
    let n = sorted.len();
    let d = settings.object_dimension as usize;

    // .hxx:249-266 — sign constraint; note the strict comparisons let an
    // exactly zero eigenvalue satisfy both the bright and the dark test.
    for &e in &sorted[d..n] {
        if (settings.bright_object && e > 0.0) || (!settings.bright_object && e < 0.0) {
            return 0.0;
        }
    }

    let abs: Vec<f64> = sorted.iter().map(|e| e.abs()).collect();
    let mut measure = 1.0f64;

    // .hxx:278-298 — R_A.
    if d < n - 1 {
        let mut ra = abs[d];
        let mut denominator = 1.0f64;
        for &a in &abs[d + 1..n] {
            denominator *= a;
        }
        if denominator.abs() > 0.0 {
            // Nested, not AND-ed: alpha == 0 leaves the R_A factor at 1.
            if settings.alpha.abs() > 0.0 {
                ra /= denominator.powf(1.0 / (n - d - 1) as f64);
                measure *= 1.0 - (-0.5 * ra * ra / (settings.alpha * settings.alpha)).exp();
            }
        } else {
            measure = 0.0;
        }
    }

    // .hxx:300-318 — R_B. beta == 0 collapses the whole measure.
    if d > 0 {
        let mut rb = abs[d - 1];
        let mut denominator = 1.0f64;
        for &a in &abs[d..n] {
            denominator *= a;
        }
        if denominator.abs() > 0.0 && settings.beta.abs() > 0.0 {
            rb /= denominator.powf(1.0 / (n - d) as f64);
            measure *= (-0.5 * rb * rb / (settings.beta * settings.beta)).exp();
        } else {
            measure = 0.0;
        }
    }

    // .hxx:320-328 — second-order structureness.
    if settings.gamma.abs() > 0.0 {
        let mut frobenius_squared = 0.0f64;
        for &a in &abs {
            frobenius_squared += a * a;
        }
        measure *= 1.0 - (-0.5 * frobenius_squared / (settings.gamma * settings.gamma)).exp();
    }

    // .hxx:330-334.
    if settings.scale_objectness_measure {
        measure *= abs[n - 1];
    }

    measure
}

/// `ObjectnessMeasureImageFilter`: enhances `object_dimension`-dimensional
/// structures in `img`.
///
/// The output takes `img`'s pixel type and geometry.
///
/// `img` is expected to be *pre-smoothed* at the scale of interest: the
/// Hessian is a plain central difference, with no Gaussian convolution. See
/// the module docs.
///
/// Errors with:
/// - [`FilterError::RequiresRealPixelType`] for a non-`Float32`,
///   non-`Float64` input (`pixel_types: RealPixelIDTypeList`);
/// - [`FilterError::UnsupportedObjectnessDimension`] outside 2-D and 3-D;
/// - [`FilterError::InvalidObjectDimension`] when
///   `object_dimension >= img.dimension()`
///   (`itkHessianToObjectnessMeasureImageFilter.hxx:210-217`).
pub fn objectness_measure(img: &Image, settings: &ObjectnessMeasureSettings) -> Result<Image> {
    let pixel_id = img.pixel_id();
    if !matches!(pixel_id, PixelId::Float32 | PixelId::Float64) {
        return Err(FilterError::RequiresRealPixelType(pixel_id));
    }
    let dim = img.dimension();
    if dim != 2 && dim != 3 {
        return Err(FilterError::UnsupportedObjectnessDimension(dim));
    }
    let object_dimension = settings.object_dimension as usize;
    if object_dimension >= dim {
        return Err(FilterError::InvalidObjectDimension {
            object_dimension,
            image_dimension: dim,
        });
    }

    let size = img.size().to_vec();
    let spacing = img.spacing().to_vec();
    let vals = img.to_f64_vec()?;
    let narrow = pixel_id == PixelId::Float32;

    let mut strides = vec![1usize; dim];
    for d in 1..dim {
        strides[d] = strides[d - 1] * size[d - 1];
    }

    let n_pixels = img.number_of_pixels();
    let mut out = vec![0.0f64; n_pixels];
    let mut idx = vec![0usize; dim];
    for (lin, o) in out.iter_mut().enumerate() {
        let mut rest = lin;
        for d in 0..dim {
            idx[d] = rest % size[d];
            rest /= size[d];
        }
        let h = hessian_at(&vals, &size, &strides, &spacing, &idx, narrow);
        let (eigen, _) = symmetric_eigen(&h, dim);
        let sorted = sort_by_magnitude(&eigen[..dim]);
        *o = objectness_from_sorted_eigenvalues(&sorted, settings);
    }

    image_from_f64(pixel_id, &size, img, &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> ObjectnessMeasureSettings {
        ObjectnessMeasureSettings::default()
    }

    #[test]
    fn defaults_match_the_yaml() {
        let s = ObjectnessMeasureSettings::default();
        assert_eq!(s.alpha, 0.5);
        assert_eq!(s.beta, 0.5);
        assert_eq!(s.gamma, 5.0);
        assert!(s.scale_objectness_measure);
        assert_eq!(s.object_dimension, 1);
        assert!(s.bright_object);
    }

    /// 3-D, ObjectDimension = 1 — Frangi's vesselness. Eigenvalues
    /// `(-1, -2, -4)`: `R_A = 2/4 = 0.5`, `R_B = 1/sqrt(8)`,
    /// `S^2 = 1 + 4 + 16 = 21`, scale `= 4`. Expected value computed
    /// independently from `(1 - e^{-R_A^2/2α^2}) e^{-R_B^2/2β^2}
    /// (1 - e^{-S^2/2γ^2}) · 4`.
    #[test]
    fn frangi_vesselness_3d() {
        let v = objectness_from_sorted_eigenvalues(&[-1.0, -2.0, -4.0], &settings());
        assert!((v - 0.42037037523733084).abs() < 1e-12, "{v}");
    }

    /// 2-D blob (`ObjectDimension = 0`): the `R_B` factor is skipped
    /// (`d > 0` is false), `R_A = a_0 / a_1^{1/(2-0-1)} = 1/4`.
    #[test]
    fn blob_2d_skips_r_b() {
        let s = ObjectnessMeasureSettings {
            object_dimension: 0,
            ..settings()
        };
        let v = objectness_from_sorted_eigenvalues(&[-1.0, -4.0], &s);
        assert!((v - 0.13547151936974275).abs() < 1e-12, "{v}");
    }

    /// 2-D vessel (`ObjectDimension = 1`): the `R_A` factor is skipped
    /// (`d < N-1` is false), `R_B = a_0 / a_1^{1/(2-1)} = 1/4`.
    #[test]
    fn vessel_2d_skips_r_a() {
        let v = objectness_from_sorted_eigenvalues(&[-1.0, -4.0], &settings());
        let s = ObjectnessMeasureSettings {
            object_dimension: 0,
            ..settings()
        };
        let blob = objectness_from_sorted_eigenvalues(&[-1.0, -4.0], &s);
        // Same R = 1/4 in both, but one factor is (1 - e^{-R²/2α²}) and the
        // other e^{-R²/2β²}; with alpha == beta they are distinct.
        assert!((v - 1.0174471895798187).abs() < 1e-12, "{v}");
        assert!(v > blob);
    }

    /// The sign constraint tests `sortedEigenValues[i]` for `i in d..N`, not
    /// the magnitudes: a bright object needs the `N-d` largest-magnitude
    /// eigenvalues non-positive.
    #[test]
    fn sign_constraint_rejects_wrong_signed_eigenvalues() {
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[-1.0, 2.0, -4.0], &settings()),
            0.0
        );
        let dark = ObjectnessMeasureSettings {
            bright_object: false,
            ..settings()
        };
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[1.0, -2.0, 4.0], &dark),
            0.0
        );
        // Dark objects: the same magnitudes with flipped signs give exactly
        // the bright answer, since only |e_i| enters the formula.
        let bright = objectness_from_sorted_eigenvalues(&[-1.0, -2.0, -4.0], &settings());
        let flipped = objectness_from_sorted_eigenvalues(&[1.0, 2.0, 4.0], &dark);
        assert_eq!(bright, flipped);
    }

    /// `.hxx:252` uses strict comparisons, so a zero eigenvalue passes both
    /// the bright and the dark constraint. Feeding an unsorted array whose
    /// `a_2 = 0` (which magnitude sorting would never produce, but which
    /// isolates the branch) shows it is the `R_A` denominator that zeroes the
    /// measure, not the sign test — a `+0.0` would have failed the bright
    /// sign test had `>= 0.0` been used.
    #[test]
    fn zero_eigenvalue_passes_the_sign_constraint_but_zeroes_the_denominator() {
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[-1.0, -2.0, 0.0], &settings()),
            0.0
        );
        let dark = ObjectnessMeasureSettings {
            bright_object: false,
            ..settings()
        };
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[1.0, 2.0, 0.0], &dark),
            0.0
        );
        // An all-zero Hessian: sign constraint passes under both flags, and
        // R_A's denominator (a_2 = 0) is what forces the 0.
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[0.0, 0.0, 0.0], &settings()),
            0.0
        );
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[0.0, 0.0, 0.0], &dark),
            0.0
        );
    }

    /// Upstream finding 2: `alpha == 0` leaves the `R_A` factor at `1`
    /// (its limit), while `beta == 0` drives the measure to `0`.
    #[test]
    fn alpha_zero_keeps_the_factor_but_beta_zero_zeroes_the_measure() {
        let no_alpha = ObjectnessMeasureSettings {
            alpha: 0.0,
            ..settings()
        };
        let v = objectness_from_sorted_eigenvalues(&[-1.0, -2.0, -4.0], &no_alpha);
        // 1 · e^{-R_B²/2β²} · (1 - e^{-S²/2γ²}) · 4, with the R_A factor gone.
        assert!((v - 1.0683688211394498).abs() < 1e-12, "{v}");
        assert!(v > objectness_from_sorted_eigenvalues(&[-1.0, -2.0, -4.0], &settings()));

        let no_beta = ObjectnessMeasureSettings {
            beta: 0.0,
            ..settings()
        };
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[-1.0, -2.0, -4.0], &no_beta),
            0.0
        );
        // Even for R_B == 0, whose true limit is 1.
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[0.0, -2.0, -4.0], &no_beta),
            0.0
        );
    }

    /// `gamma == 0` skips the structureness factor entirely.
    #[test]
    fn gamma_zero_skips_the_structureness_factor() {
        let no_gamma = ObjectnessMeasureSettings {
            gamma: 0.0,
            ..settings()
        };
        let v = objectness_from_sorted_eigenvalues(&[-1.0, -2.0, -4.0], &no_gamma);
        // (1 - e^{-0.5}) · e^{-0.25} · 4.
        assert!((v - 1.2257369213215608).abs() < 1e-12, "{v}");
    }

    /// `ScaleObjectnessMeasure` multiplies by the largest absolute eigenvalue.
    #[test]
    fn scale_objectness_measure_multiplies_by_the_largest_magnitude() {
        let scaled = objectness_from_sorted_eigenvalues(&[-1.0, -2.0, -4.0], &settings());
        let unscaled = objectness_from_sorted_eigenvalues(
            &[-1.0, -2.0, -4.0],
            &ObjectnessMeasureSettings {
                scale_objectness_measure: false,
                ..settings()
            },
        );
        assert!((scaled - unscaled * 4.0).abs() < 1e-15);
    }

    /// Upstream finding 3: a `|e| == |e'|` tie keeps the ascending-by-value
    /// order (negative first), which is what libstdc++'s insertion-sort
    /// `std::sort` does for ranges of at most 16 elements. That decides the
    /// sign constraint: with `d = 1`, `sorted[1] = -3` passes for a bright
    /// object where `+3` would have failed.
    #[test]
    fn magnitude_tie_keeps_the_negative_eigenvalue_first() {
        assert_eq!(sort_by_magnitude(&[-3.0, 1.0, 3.0]), vec![1.0, -3.0, 3.0]);
        // sorted[1] = -3 <= 0, sorted[2] = +3 > 0 => bright constraint fails.
        assert_eq!(
            objectness_from_sorted_eigenvalues(&[1.0, -3.0, 3.0], &settings()),
            0.0
        );
    }

    // ---- Hessian -----------------------------------------------------------

    fn hessian_of(img: &Image, idx: &[usize]) -> Mat {
        let dim = img.dimension();
        let size = img.size().to_vec();
        let mut strides = vec![1usize; dim];
        for d in 1..dim {
            strides[d] = strides[d - 1] * size[d - 1];
        }
        hessian_at(
            &img.to_f64_vec().unwrap(),
            &size,
            &strides,
            img.spacing(),
            idx,
            img.pixel_id() == PixelId::Float32,
        )
    }

    /// Central differences with anisotropic spacing, hand-computed. Values are
    /// laid out `v[y * 3 + x]`.
    #[test]
    fn central_differences_at_an_interior_pixel() {
        #[rustfmt::skip]
        let v = vec![
            1.0f64, 2.0,  4.0,
            8.0,   16.0, 32.0,
            64.0, 128.0, 256.0,
        ];
        let mut img = Image::from_vec(&[3, 3], v).unwrap();
        img.set_spacing(&[1.0, 2.0]).unwrap();

        let h = hessian_of(&img, &[1, 1]);
        // H(0,0) = (32 + 8 - 2*16) / 1^2 = 8
        assert_eq!(h[0][0], 8.0);
        // H(1,1) = (128 + 2 - 2*16) / 2^2 = 98/4 = 24.5
        assert_eq!(h[1][1], 24.5);
        // H(0,1) = (v(0,0) - v(0,2) - v(2,0) + v(2,2)) / (4*1*2)
        //        = (1 - 64 - 4 + 256) / 8 = 189/8 = 23.625
        assert_eq!(h[0][1], 23.625);
        assert_eq!(h[1][0], h[0][1]);
    }

    /// Zero-flux Neumann: at a corner the out-of-image samples clamp to the
    /// nearest in-image pixel, so `H(i,i)` degenerates to a one-sided first
    /// difference and the cross term samples the 2x2 corner block.
    #[test]
    fn zero_flux_neumann_at_a_corner() {
        #[rustfmt::skip]
        let v = vec![
            1.0f64, 2.0,  4.0,
            8.0,   16.0, 32.0,
            64.0, 128.0, 256.0,
        ];
        let img = Image::from_vec(&[3, 3], v).unwrap();

        let h = hessian_of(&img, &[0, 0]);
        // H(0,0) = f(1,0) + f(-1 -> 0, 0) - 2 f(0,0) = 2 + 1 - 2 = 1
        assert_eq!(h[0][0], 1.0);
        // H(1,1) = f(0,1) + f(0,-1 -> 0) - 2 f(0,0) = 8 + 1 - 2 = 7
        assert_eq!(h[1][1], 7.0);
        // H(0,1) = (f(0,0) - f(0,1) - f(1,0) + f(1,1)) / 4 = (1 - 8 - 2 + 16)/4
        assert_eq!(h[0][1], 7.0 / 4.0);
    }

    /// `Float32` inputs round `f(x + e_i) + f(x - e_i)` to `float` before the
    /// `double` subtraction, per `itkHessianImageFilter.hxx:256`. `1e8 + 1`
    /// is `1e8` in `float` and `100000001` in `double`.
    #[test]
    fn float32_input_rounds_the_diagonal_sum_to_float() {
        #[rustfmt::skip]
        let v32 = vec![
            0.0f32, 0.0, 0.0,
            1.0,    0.0, 1e8,
            0.0,    0.0, 0.0,
        ];
        let img32 = Image::from_vec(&[3, 3], v32).unwrap();
        assert_eq!(hessian_of(&img32, &[1, 1])[0][0], 1e8);

        let v64: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1e8, 0.0, 0.0, 0.0];
        let img64 = Image::from_vec(&[3, 3], v64).unwrap();
        assert_eq!(hessian_of(&img64, &[1, 1])[0][0], 100000001.0);
    }

    // ---- end to end --------------------------------------------------------

    /// A constant image has a zero Hessian everywhere: every eigenvalue is 0,
    /// the sign constraint passes, and `R_A`'s denominator zeroes the measure.
    #[test]
    fn constant_image_is_all_zero() {
        let img = Image::from_vec(&[4, 4], vec![7.0f64; 16]).unwrap();
        let out = objectness_measure(&img, &settings()).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float64);
        assert!(out.scalar_slice::<f64>().unwrap().iter().all(|&v| v == 0.0));
    }

    /// A bright ridge along y (a 1-D object in 2-D) scores at its crest and
    /// nowhere in the flat background. Cross-checked at the crest against
    /// [`objectness_from_sorted_eigenvalues`] fed the hand-computed Hessian.
    #[test]
    fn bright_ridge_is_enhanced_at_its_crest() {
        // 5x3, bright column x = 2.
        #[rustfmt::skip]
        let v = vec![
            0.0f64, 0.0, 1.0, 0.0, 0.0,
            0.0,    0.0, 1.0, 0.0, 0.0,
            0.0,    0.0, 1.0, 0.0, 0.0,
        ];
        let img = Image::from_vec(&[5, 3], v).unwrap();
        let out = objectness_measure(&img, &settings()).unwrap();
        let o = out.scalar_slice::<f64>().unwrap();

        // Crest at (2, 1): H = [[0 + 0 - 2*1, 0], [1 + 1 - 2*1, 0]] -> diag(-2, 0),
        // magnitude-sorted (0, -2). d = 1: no R_A; R_B = 0 / 2 = 0 -> factor 1;
        // S^2 = 4; scale 2.
        let crest = o[7];
        let expected = objectness_from_sorted_eigenvalues(&[0.0, -2.0], &settings());
        assert!((crest - expected).abs() < 1e-12);
        assert!(crest > 0.0);

        // Flat background column x = 0 has a zero Hessian.
        assert_eq!(o[5], 0.0);
    }

    /// The dark twin of the ridge test: `bright_object = false` scores the
    /// trough and the bright ridge scores nothing.
    #[test]
    fn bright_object_flag_selects_the_sign() {
        #[rustfmt::skip]
        let v = vec![
            0.0f64, 0.0, 1.0, 0.0, 0.0,
            0.0,    0.0, 1.0, 0.0, 0.0,
            0.0,    0.0, 1.0, 0.0, 0.0,
        ];
        let img = Image::from_vec(&[5, 3], v).unwrap();
        let dark = ObjectnessMeasureSettings {
            bright_object: false,
            ..settings()
        };
        let o = objectness_measure(&img, &dark).unwrap();
        let o = o.scalar_slice::<f64>().unwrap();
        // The crest's largest-magnitude eigenvalue is -2 < 0: dark rejects it.
        assert_eq!(o[7], 0.0);
        // Its flanks (x = 1, 3) curve upward: H(0,0) = 0 + 1 - 0 = 1 > 0.
        assert!(o[6] > 0.0);
    }

    // ---- errors ------------------------------------------------------------

    #[test]
    fn integer_pixel_types_are_rejected() {
        let img = Image::from_vec(&[2, 2], vec![0i16; 4]).unwrap();
        assert_eq!(
            objectness_measure(&img, &settings()),
            Err(FilterError::RequiresRealPixelType(PixelId::Int16))
        );
    }

    #[test]
    fn one_and_four_dimensional_images_are_rejected() {
        let img = Image::from_vec(&[4], vec![0.0f64; 4]).unwrap();
        assert_eq!(
            objectness_measure(&img, &settings()),
            Err(FilterError::UnsupportedObjectnessDimension(1))
        );
        let img = Image::from_vec(&[2, 2, 2, 2], vec![0.0f64; 16]).unwrap();
        assert_eq!(
            objectness_measure(&img, &settings()),
            Err(FilterError::UnsupportedObjectnessDimension(4))
        );
    }

    /// `VerifyPreconditions`: `ObjectDimension` must be strictly less than
    /// `ImageDimension`, so `2` is rejected in 2-D while the default `1`
    /// is accepted.
    #[test]
    fn object_dimension_must_be_lower_than_image_dimension() {
        let img = Image::from_vec(&[2, 2], vec![0.0f64; 4]).unwrap();
        let s = ObjectnessMeasureSettings {
            object_dimension: 2,
            ..settings()
        };
        assert_eq!(
            objectness_measure(&img, &s),
            Err(FilterError::InvalidObjectDimension {
                object_dimension: 2,
                image_dimension: 2,
            })
        );
        let s = ObjectnessMeasureSettings {
            object_dimension: 1,
            ..settings()
        };
        assert!(objectness_measure(&img, &s).is_ok());
    }

    #[test]
    fn preserves_geometry_and_pixel_type() {
        let mut img = Image::from_vec(&[3, 3], vec![1.0f32; 9]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[3.0, -1.0]).unwrap();
        let out = objectness_measure(&img, &settings()).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
    }
}
