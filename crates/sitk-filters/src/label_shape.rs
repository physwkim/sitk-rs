//! `LabelShapeStatisticsImageFilter`: per-label shape attributes of an
//! integer label image.
//!
//! Port of SimpleITK's `LabelShapeStatisticsImageFilter`, which is
//! `itk::LabelImageToShapeLabelMapFilter` (i.e.
//! `itk::LabelImageToLabelMapFilter` followed by
//! `itk::ShapeLabelMapFilter`) with the attributes of
//! `itk::ShapeLabelObject` exposed as measurements.
//!
//! Sources:
//!
//! - `itkLabelImageToLabelMapFilter.hxx` — run-length encodes each scanline
//!   (a maximal run of equal, non-background pixels along axis 0) into a
//!   per-label list of lines, in raster order. Ported as
//!   [`sitk_core::LabelMap::from_label_image`], whose raster ordering is a
//!   documented postcondition; `compute_perimeter` below depends on it.
//! - `itkShapeLabelMapFilter.hxx` — `ThreadedProcessLabelObject` computes
//!   every un-gated attribute in a single pass over those lines;
//!   `ComputeFeretDiameter`, `ComputePerimeter` and
//!   `ComputeOrientedBoundingBox` are gated on the corresponding
//!   `ComputeXX` flag.
//! - `itkShapeLabelObject.h` — `GetOrientedBoundingBoxDirection` (an alias
//!   for the principal axes) and `GetOrientedBoundingBoxVertices`.
//! - `itkGeometryUtilities.cxx` — the hyper-sphere volume/perimeter/radius
//!   helpers behind `EquivalentSphericalRadius`,
//!   `EquivalentSphericalPerimeter` and Crofton's constant.
//!
//! ## Perimeter
//!
//! `Perimeter` is *not* a boundary-pixel count. ITK estimates it with a
//! Crofton formula: it counts, for each of the `2^D - 1` distinct
//! unsigned lattice directions, how many times the object's boundary is
//! crossed by lines in that direction ("intercepts"), then weights each
//! direction's count by the area of its Voronoi cell on the unit sphere.
//! The intercept counting in `ComputePerimeter` works directly on the RLE
//! lines: the lines are bucketed into a `D-1` dimensional image indexed by
//! `idx[1..]`, and each bucket is compared against its `3^(D-1) - 1`
//! neighbouring buckets by walking the two sorted line lists in lockstep
//! and counting the overlap of each line with the *gaps between* the
//! neighbour's lines. `PerimeterFromInterceptCount` then applies the
//! direction weights: for 2-D these are exact (`pi/4` times the
//! `1/spacing` weights), for 3-D they are ITK's hard-coded "magical
//! numbers" `c1..c7`, the Voronoi-partition areas of the 26 unit-cube
//! directions on the unit sphere. Those 3-D constants assume isotropic
//! spacing — ITK carries a `TODO - recompute those values if the spacing
//! is non isotropic` there, and this port reproduces the same behaviour
//! rather than correcting it.
//!
//! `Roundness` is `EquivalentSphericalPerimeter / Perimeter` and
//! `PerimeterOnBorderRatio` is `PerimeterOnBorder / Perimeter`, so both are
//! only defined when the perimeter was computed.
//!
//! ## Deliberate divergences from the C++
//!
//! - ITK obtains the determinant of the principal-axes matrix as the
//!   product of the (complex) eigenvalues returned by
//!   `itk::RealEigenDecomposition`. This port computes the determinant
//!   directly by LU decomposition. The two agree exactly in exact
//!   arithmetic (the product of a matrix's eigenvalues *is* its
//!   determinant) and the value is only used through `std::real(det)`,
//!   which for an orthogonal matrix is `±1`.
//! - The symmetric eigendecomposition of the second-order central moments
//!   uses cyclic Jacobi rather than `vnl_symmetric_eigensystem`'s QL
//!   implicit-shift. Both return eigenvalues in ascending order; the sign
//!   of each eigenvector is left as the algorithm produces it, exactly as
//!   ITK does (ITK only fixes the overall determinant, by negating the last
//!   row of the axes matrix, so individual axis signs are not canonical in
//!   ITK either).
//! - SimpleITK returns `0` / a default-constructed value for the gated
//!   attributes when their `ComputeXX` flag is off. Here they are
//!   [`Option`]s, so "not requested" is distinguishable from "computed and
//!   happens to be zero".
//!
//! Only 2-D and 3-D images are supported, matching the dimensions
//! SimpleITK instantiates the filter for; `ComputePerimeter`'s
//! direction weights are dimension-specific overloads for exactly those
//! two dimensions in `itkShapeLabelMapFilter.hxx`.

use std::collections::{BTreeMap, HashMap};

use sitk_core::{Image, LabelMap, LabelObjectLine, coord};

// `MAX_DIM` is the maximum image dimension this filter supports, sized so that
// fixed-size arrays can stand in for ITK's `Index`/`Offset`/`Matrix` types.
use crate::linalg::{MAX_DIM, Mat, symmetric_eigen};
use crate::{FilterError, Result};

/// `itk::ImageRegion` as SimpleITK's `GetBoundingBox` reports it: the
/// per-axis start index followed by the per-axis extent. SimpleITK flattens
/// the two into one `[x, y, width, height]` array; they are kept apart here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundingBox {
    /// Inclusive lowest index along each axis.
    pub index: Vec<i64>,
    /// Number of pixels spanned along each axis.
    pub size: Vec<u64>,
}

/// `itk::ShapeLabelObject`'s oriented bounding box: the tightest box aligned
/// with the object's principal axes that contains every pixel *including the
/// pixels' own extent*, not just their centers.
#[derive(Clone, Debug, PartialEq)]
pub struct OrientedBoundingBox {
    /// Edge lengths, in physical units, along each principal axis.
    pub size: Vec<f64>,
    /// Physical position of the box corner with the minimum coordinate in
    /// every principal-axis direction.
    pub origin: Vec<f64>,
    /// Row-major `dim × dim` matrix; row `i` is the `i`-th principal axis.
    /// `GetOrientedBoundingBoxDirection` is an alias for the principal axes.
    pub direction: Vec<f64>,
    /// The `2^dim` corners, in physical space. Corner `i`'s binary digits
    /// select min (`0`) or max (`1`) along each axis, most significant bit
    /// first: in 2-D the order is `[minX,minY], [minX,maxY], [maxX,minY],
    /// [maxX,maxY]` in the principal-axis basis.
    pub vertices: Vec<Vec<f64>>,
}

/// Shape attributes of one label, as `itk::ShapeLabelObject` stores them.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeStatistics {
    /// Number of pixels carrying this label.
    pub number_of_pixels: u64,
    /// `number_of_pixels` times the physical volume of one pixel.
    pub physical_size: f64,
    /// Center of mass, in physical space.
    pub centroid: Vec<f64>,
    /// Axis-aligned bounding box, in index space.
    pub bounding_box: BoundingBox,
    /// Pixels of this label that touch the edge of the image.
    pub number_of_pixels_on_border: u64,
    /// Physical size of the object's boundary that lies on the image edge.
    pub perimeter_on_border: f64,
    /// Eigenvalues of the second-order central moments, ascending.
    pub principal_moments: Vec<f64>,
    /// Row-major `dim × dim`; row `i` is the eigenvector for
    /// `principal_moments[i]`. The last row is negated if needed to make the
    /// matrix a proper rotation (determinant `+1`).
    pub principal_axes: Vec<f64>,
    /// `sqrt(pm[dim-1] / pm[dim-2])`, or `0` if that denominator is zero.
    pub elongation: f64,
    /// `sqrt(pm[1] / pm[0])`, or `0` if `pm[0]` is zero.
    pub flatness: f64,
    /// Radius of the `dim`-sphere whose volume equals `physical_size`.
    pub equivalent_spherical_radius: f64,
    /// Surface area of that same `dim`-sphere.
    pub equivalent_spherical_perimeter: f64,
    /// Axis lengths of the ellipsoid with the object's principal moments and
    /// `physical_size`.
    pub equivalent_ellipsoid_diameter: Vec<f64>,
    /// Largest distance between two boundary pixel centers. `None` unless
    /// [`LabelShapeStatisticsSettings::compute_feret_diameter`] was set.
    pub feret_diameter: Option<f64>,
    /// Crofton estimate of the object's boundary size. `None` unless
    /// [`LabelShapeStatisticsSettings::compute_perimeter`] was set.
    pub perimeter: Option<f64>,
    /// `equivalent_spherical_perimeter / perimeter`. `None` iff
    /// `perimeter` is.
    pub roundness: Option<f64>,
    /// `perimeter_on_border / perimeter`. `None` iff `perimeter` is.
    pub perimeter_on_border_ratio: Option<f64>,
    /// `None` unless
    /// [`LabelShapeStatisticsSettings::compute_oriented_bounding_box`] was
    /// set.
    pub oriented_bounding_box: Option<OrientedBoundingBox>,
}

/// The four settings SimpleITK's `LabelShapeStatisticsImageFilter` exposes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelShapeStatisticsSettings {
    /// Pixel value that is *not* part of any object. Cast to the image's
    /// integer pixel type before comparison, as
    /// `itkLabelImageToLabelMapFilter.hxx` does. SimpleITK defaults this to
    /// `0` (ITK's own default is `NumericTraits<PixelType>::NonpositiveMin()`).
    pub background_value: f64,
    /// Off by default in SimpleITK "because of the high computation time
    /// required" — it is quadratic in the number of boundary pixels.
    pub compute_feret_diameter: bool,
    /// On by default in SimpleITK.
    pub compute_perimeter: bool,
    /// Off by default in SimpleITK "because of potential memory consumption
    /// issues with sparse labels".
    pub compute_oriented_bounding_box: bool,
}

impl Default for LabelShapeStatisticsSettings {
    fn default() -> Self {
        Self {
            background_value: 0.0,
            compute_feret_diameter: false,
            compute_perimeter: true,
            compute_oriented_bounding_box: false,
        }
    }
}

/// Compute the shape attributes of every label in `img`.
///
/// `img` must have an integer pixel type (SimpleITK's yaml declares
/// `pixel_types: IntegerPixelIDTypeList`) and be 2-D or 3-D. Every pixel
/// value other than `settings.background_value` starts an object; the
/// returned map is keyed by label value, ascending, matching the order
/// `itk::LabelMap::GetLabels` reports.
pub fn label_shape_statistics(
    img: &Image,
    settings: &LabelShapeStatisticsSettings,
) -> Result<BTreeMap<i64, ShapeStatistics>> {
    if !img.pixel_id().is_integer_scalar() {
        return Err(FilterError::RequiresIntegerPixelType(img.pixel_id()));
    }
    let dim = img.dimension();
    if dim != 2 && dim != 3 {
        return Err(FilterError::UnsupportedShapeDimension(dim));
    }

    let size = img.size();
    let spacing = img.spacing();
    let origin = img.origin();
    let direction = img.direction();

    let background = settings.background_value as i64;
    // `LabelMap::from_label_image` *is* `itkLabelImageToLabelMapFilter`, and it
    // emits the lines of each label in raster order — `idx[1..]` first, then
    // `idx[0]` — which `compute_perimeter` relies on.
    let label_map = LabelMap::from_label_image(img, background)?;

    // `compute_feret_diameter` is the only consumer of the dense label image,
    // and it is off by default.
    let labels: Vec<i64> = if settings.compute_feret_diameter {
        img.to_f64_vec()?
            .iter()
            .map(|&v| v.round() as i64)
            .collect()
    } else {
        Vec::new()
    };

    let mut out = BTreeMap::new();
    for object in label_map.label_objects() {
        let label = object.label();
        let lines = object.lines();
        let mut stats = shape_attributes(lines, dim, size, spacing, origin, direction);

        if settings.compute_feret_diameter {
            stats.feret_diameter = Some(compute_feret_diameter(
                &labels, size, dim, spacing, label, lines,
            ));
        }
        if settings.compute_perimeter {
            let perimeter = compute_perimeter(lines, &stats.bounding_box, dim, spacing);
            stats.roundness = Some(stats.equivalent_spherical_perimeter / perimeter);
            stats.perimeter_on_border_ratio = Some(stats.perimeter_on_border / perimeter);
            stats.perimeter = Some(perimeter);
        }
        if settings.compute_oriented_bounding_box {
            stats.oriented_bounding_box = Some(compute_oriented_bounding_box(
                lines,
                dim,
                spacing,
                origin,
                direction,
                &stats.centroid,
                &stats.principal_axes,
            ));
        }
        out.insert(label, stats);
    }
    Ok(out)
}

// ---- geometry helpers -----------------------------------------------------

/// `Image::TransformContinuousIndexToPhysicalPoint`:
/// `p = origin + Direction * (spacing ⊙ index)`.
/// `Image::TransformContinuousIndexToPhysicalPoint` — the continuous method,
/// origin **last** (itkImageBase.h:558-572), via the shared [`sitk_core::coord`]
/// primitive. This is the fold ITK uses for the fractional shape centroid
/// (`itkShapeLabelMapFilter.hxx:298`); it differs from [`index_to_physical`]
/// (integer method, origin-first) only at large origins. No heap allocation: the
/// `IndexToPhysicalPoint` matrix is composed into a stack buffer.
fn continuous_index_to_physical(
    idx: &[f64; MAX_DIM],
    dim: usize,
    spacing: &[f64],
    origin: &[f64],
    direction: &[f64],
) -> [f64; MAX_DIM] {
    let mut i2p = [0.0; MAX_DIM * MAX_DIM];
    coord::index_to_physical_matrix_into(&direction[..dim * dim], &spacing[..dim], &mut i2p, dim);
    let mut p = [0.0; MAX_DIM];
    coord::continuous_index_to_physical_point_into(
        &i2p[..dim * dim],
        &origin[..dim],
        &idx[..dim],
        &mut p,
        dim,
    );
    p
}

/// `Image::TransformIndexToPhysicalPoint` — the integer method, origin **first**
/// (itkImageBase.h:592-604), via the shared [`sitk_core::coord`] primitive. ITK's
/// second-moment and center-of-gravity accumulations use this integer fold. No
/// heap allocation.
pub(crate) fn index_to_physical(
    idx: &[i64; MAX_DIM],
    dim: usize,
    spacing: &[f64],
    origin: &[f64],
    direction: &[f64],
) -> [f64; MAX_DIM] {
    let mut i2p = [0.0; MAX_DIM * MAX_DIM];
    coord::index_to_physical_matrix_into(&direction[..dim * dim], &spacing[..dim], &mut i2p, dim);
    let mut widened = [0.0; MAX_DIM];
    for d in 0..dim {
        widened[d] = idx[d] as f64;
    }
    let mut p = [0.0; MAX_DIM];
    coord::index_to_physical_point_f64_into(
        &i2p[..dim * dim],
        &origin[..dim],
        &widened[..dim],
        &mut p,
        dim,
    );
    p
}

// ---- itkGeometryUtilities.cxx ---------------------------------------------

fn factorial(n: i64) -> f64 {
    if n < 1 {
        1.0
    } else {
        n as f64 * factorial(n - 1)
    }
}

fn double_factorial(n: i64) -> f64 {
    if n < 2 {
        1.0
    } else {
        n as f64 * double_factorial(n - 2)
    }
}

/// `Gamma(n/2 + 1)`.
fn gamma_n2p1(n: i64) -> f64 {
    if n % 2 == 0 {
        factorial(n / 2)
    } else {
        std::f64::consts::PI.sqrt() * double_factorial(n) / 2f64.powf((n + 1) as f64 / 2.0)
    }
}

fn hyper_sphere_volume(dim: i64, radius: f64) -> f64 {
    let d = dim as f64;
    std::f64::consts::PI.powf(d * 0.5) * radius.powf(d) / gamma_n2p1(dim)
}

fn hyper_sphere_perimeter(dim: i64, radius: f64) -> f64 {
    dim as f64 * hyper_sphere_volume(dim, radius) / radius
}

fn hyper_sphere_radius_from_volume(dim: i64, volume: f64) -> f64 {
    (volume * gamma_n2p1(dim) / std::f64::consts::PI.powf(dim as f64 * 0.5)).powf(1.0 / dim as f64)
}

// ---- linear algebra -------------------------------------------------------

/// `itk::Math::AlmostEquals(x, 0.0)`, specialized to a comparison against
/// exactly zero. `itkMath.h`'s `FloatAlmostEqual<double>` combines a ULP check
/// (`maxUlps = 4`) with an absolute-difference check whose
/// `maxAbsoluteDifference` defaults to `0.1 * NumericTraits<double>::epsilon()`
/// (`itkMath.h:334-335`, `~2.22e-17`). That near-zero absolute path is tested
/// *first* (`itkMath.h:339-343`) and, against a literal `0.0` comparand, is the
/// dominant term — the ULP arm never fires — so the comparison collapses to
/// `|x| <= 0.1 * epsilon`. (A prior version of this helper used a 4-ULP bit
/// window `~2e-323`, understating the real window by ~10^306 and dividing where
/// upstream skips the ratio; §2.154.)
pub(crate) fn is_almost_zero(x: f64) -> bool {
    x.abs() <= 0.1 * f64::EPSILON
}

/// Determinant by LU with partial pivoting.
///
/// ITK instead builds `RealEigenDecomposition(principalAxes)` and multiplies
/// the complex eigenvalues; that product *is* the determinant.
pub(crate) fn determinant(m: &Mat, n: usize) -> f64 {
    let mut a = *m;
    let mut det = 1.0;
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            a.swap(pivot, col);
            det = -det;
        }
        det *= a[col][col];
        for r in col + 1..n {
            // col < r, so the pivot row and the row being eliminated are
            // disjoint borrows.
            let (head, tail) = a.split_at_mut(r);
            let pivot_row = &head[col];
            let cur = &mut tail[0];
            let f = cur[col] / pivot_row[col];
            for (x, &p) in cur.iter_mut().zip(pivot_row.iter()).take(n).skip(col) {
                *x -= f * p;
            }
        }
    }
    det
}

// ---- the single-pass attributes -------------------------------------------

/// `ShapeLabelMapFilter::ThreadedProcessLabelObject` up to (but excluding)
/// the three `ComputeXX`-gated attributes.
fn shape_attributes(
    lines: &[LabelObjectLine],
    dim: usize,
    size: &[usize],
    spacing: &[f64],
    origin: &[f64],
    direction: &[f64],
) -> ShapeStatistics {
    let mut size_per_pixel = 1.0;
    for &s in spacing.iter().take(dim) {
        size_per_pixel *= s;
    }
    let size_per_pixel_per_dimension: Vec<f64> = (0..dim)
        .map(|i| size_per_pixel / spacing[i])
        .collect::<Vec<_>>();

    // The image's largest possible region always starts at index 0 here.
    let border_min = [0i64; MAX_DIM];
    let mut border_max = [0i64; MAX_DIM];
    for i in 0..dim {
        border_max[i] = size[i] as i64 - 1;
    }

    let mut nb_of_pixels: u64 = 0;
    let mut centroid = [0.0f64; MAX_DIM];
    let mut mins = [i64::MAX; MAX_DIM];
    let mut maxs = [i64::MIN; MAX_DIM];
    let mut nb_of_pixels_on_border: u64 = 0;
    let mut perimeter_on_border = 0.0f64;
    let mut central_moments: Mat = [[0.0; MAX_DIM]; MAX_DIM];

    for line in lines {
        let idx = line.index();
        let length = line.length();
        let lf = length as f64;

        nb_of_pixels += length as u64;

        for i in 1..dim {
            centroid[i] += lf * idx[i] as f64;
        }
        centroid[0] += (idx[0] * length) as f64 + (length * (length - 1)) as f64 / 2.0;

        for i in 0..dim {
            mins[i] = mins[i].min(idx[i]);
            maxs[i] = maxs[i].max(idx[i]);
        }
        // Must fix the max for the axis 0.
        if idx[0] + length > maxs[0] {
            maxs[0] = idx[0] + length - 1;
        }

        // Object is on a border?
        let mut is_on_border = false;
        for i in 1..dim {
            if idx[i] == border_min[i] || idx[i] == border_max[i] {
                is_on_border = true;
                break;
            }
        }
        if is_on_border {
            // The line touches a border on a dimension other than 0, so the
            // whole line touches a border.
            nb_of_pixels_on_border += length as u64;
        } else {
            let mut is_on_border_0 = false;
            if idx[0] == border_min[0] {
                nb_of_pixels_on_border += 1;
                is_on_border_0 = true;
            }
            if (!is_on_border_0 || length > 1) && idx[0] + length - 1 == border_max[0] {
                nb_of_pixels_on_border += 1;
            }
        }

        // Physical size on border: axis 0 first, then the others.
        if idx[0] == border_min[0] {
            perimeter_on_border += size_per_pixel_per_dimension[0];
        }
        if idx[0] + length - 1 == border_max[0] {
            perimeter_on_border += size_per_pixel_per_dimension[0];
        }
        for i in 1..dim {
            if idx[i] == border_min[i] {
                perimeter_on_border += size_per_pixel_per_dimension[i] * lf;
            }
            if idx[i] == border_max[i] {
                perimeter_on_border += size_per_pixel_per_dimension[i] * lf;
            }
        }

        // Second-order moments, accumulated about the physical origin. The
        // `length <= 2` branch is the straightforward per-pixel sum; the
        // other is the same sum in closed form over the run.
        if length <= 2 {
            let end0 = idx[0] + length;
            let mut iidx = idx;
            while iidx[0] < end0 {
                let pp = index_to_physical(&iidx, dim, spacing, origin, direction);
                for i in 0..dim {
                    central_moments[i][i] += pp[i] * pp[i];
                    for j in i + 1..dim {
                        let cm = pp[i] * pp[j];
                        central_moments[i][j] += cm;
                        central_moments[j][i] += cm;
                    }
                }
                iidx[0] += 1;
            }
        } else {
            let pp = index_to_physical(&idx, dim, spacing, origin, direction);
            // The physical step of one index step along axis 0.
            let mut scale = [0.0f64; MAX_DIM];
            for (i, si) in scale.iter_mut().enumerate().take(dim) {
                *si = spacing[0] * direction[i * dim];
            }

            let lcoff_1 = (lf - 1.0) / 2.0;
            let lcoff_2 = (2.0 * lf - 1.0) / 3.0;

            for i in 0..dim {
                central_moments[i][i] += lf
                    * (pp[i] * pp[i]
                        + lcoff_1 * (2.0 * pp[i] * scale[i] + lcoff_2 * scale[i] * scale[i]));
                for j in i + 1..dim {
                    let cm = lf
                        * (pp[i] * pp[j]
                            + lcoff_1
                                * (pp[i] * scale[j]
                                    + scale[i] * pp[j]
                                    + lcoff_2 * scale[i] * scale[j]));
                    central_moments[j][i] += cm;
                    central_moments[i][j] += cm;
                }
            }
        }
    }

    let n = nb_of_pixels as f64;
    let mut bounding_box_size = [0u64; MAX_DIM];
    for (i, c) in centroid.iter_mut().enumerate().take(dim) {
        *c /= n;
        bounding_box_size[i] = (maxs[i] - mins[i] + 1) as u64;
    }
    for row in central_moments.iter_mut().take(dim) {
        for m in row.iter_mut().take(dim) {
            *m /= n;
        }
    }
    let physical_centroid =
        continuous_index_to_physical(&centroid, dim, spacing, origin, direction);

    // Center the second order moments.
    for i in 0..dim {
        for j in 0..dim {
            central_moments[i][j] -= physical_centroid[i] * physical_centroid[j];
        }
    }

    let (eigenvalues, eigenvectors) = symmetric_eigen(&central_moments, dim);
    let mut principal_moments = [0.0f64; MAX_DIM];
    for i in 0..dim {
        // Clamp to zero: near-zero negative eigenvalues from numerical
        // precision cause an FPE in `std::pow(edet, ...)` in the C++.
        principal_moments[i] = eigenvalues[i].max(0.0);
    }

    // principalAxes = V^T, so row i is the eigenvector of principal_moments[i].
    let mut principal_axes: Mat = [[0.0; MAX_DIM]; MAX_DIM];
    for i in 0..dim {
        for j in 0..dim {
            principal_axes[i][j] = eigenvectors[j][i];
        }
    }
    // Add a final reflection if needed for a proper rotation, by multiplying
    // the last row by the determinant.
    let det = determinant(&principal_axes, dim);
    for v in principal_axes[dim - 1].iter_mut().take(dim) {
        *v *= det;
    }

    let mut elongation = 0.0;
    let mut flatness = 0.0;
    if !is_almost_zero(principal_moments[0]) {
        let flatness_ratio = principal_moments[1] / principal_moments[0];
        if flatness_ratio > 0.0 {
            flatness = flatness_ratio.sqrt();
        }
    }
    if !is_almost_zero(principal_moments[dim - 2]) {
        let elongation_ratio = principal_moments[dim - 1] / principal_moments[dim - 2];
        if elongation_ratio > 0.0 {
            elongation = elongation_ratio.sqrt();
        }
    }

    let physical_size = n * size_per_pixel;
    let equivalent_radius = hyper_sphere_radius_from_volume(dim as i64, physical_size);
    let equivalent_perimeter = hyper_sphere_perimeter(dim as i64, equivalent_radius);

    let mut edet = 1.0;
    for &pm in principal_moments.iter().take(dim) {
        edet *= pm;
    }
    edet = edet.powf(1.0 / dim as f64);
    let mut ellipsoid_diameter = vec![0.0f64; dim];
    for i in 0..dim {
        if edet != 0.0 && principal_moments[i] / edet > 0.0 {
            ellipsoid_diameter[i] = 2.0 * equivalent_radius * (principal_moments[i] / edet).sqrt();
        }
    }

    ShapeStatistics {
        number_of_pixels: nb_of_pixels,
        physical_size,
        centroid: physical_centroid[..dim].to_vec(),
        bounding_box: BoundingBox {
            index: mins[..dim].to_vec(),
            size: bounding_box_size[..dim].to_vec(),
        },
        number_of_pixels_on_border: nb_of_pixels_on_border,
        perimeter_on_border,
        principal_moments: principal_moments[..dim].to_vec(),
        principal_axes: (0..dim)
            .flat_map(|i| (0..dim).map(move |j| (i, j)))
            .map(|(i, j)| principal_axes[i][j])
            .collect(),
        elongation,
        flatness,
        equivalent_spherical_radius: equivalent_radius,
        equivalent_spherical_perimeter: equivalent_perimeter,
        equivalent_ellipsoid_diameter: ellipsoid_diameter,
        feret_diameter: None,
        perimeter: None,
        roundness: None,
        perimeter_on_border_ratio: None,
        oriented_bounding_box: None,
    }
}

// ---- ComputeFeretDiameter -------------------------------------------------

/// `ShapeLabelMapFilter::ComputeFeretDiameter`: collect every pixel of the
/// object with at least one `3^dim` neighbour whose label differs, then take
/// the largest physical distance between any two of them.
///
/// ITK runs this over a label image regenerated from the label map with a
/// `ConstantBoundaryCondition` of `label + 1`, so out-of-image neighbours
/// always differ and edge pixels always count as boundary. That regenerated
/// image agrees pixel-for-pixel with the input label image, so `labels` is
/// used directly.
///
/// Distances use `index difference * spacing` — the direction cosines are
/// deliberately not applied, matching the C++.
fn compute_feret_diameter(
    labels: &[i64],
    size: &[usize],
    dim: usize,
    spacing: &[f64],
    label: i64,
    lines: &[LabelObjectLine],
) -> f64 {
    let offsets = neighborhood_offsets(dim);

    let mut strides = [1usize; MAX_DIM];
    for d in 1..dim {
        strides[d] = strides[d - 1] * size[d - 1];
    }
    let flat =
        |idx: &[i64; MAX_DIM]| -> usize { (0..dim).map(|d| idx[d] as usize * strides[d]).sum() };

    let mut border: Vec<[i64; MAX_DIM]> = Vec::new();
    for line in lines {
        let mut idx = line.index();
        for _ in 0..line.length() {
            let on_border = offsets.iter().any(|off| {
                let mut nidx = [0i64; MAX_DIM];
                for d in 0..dim {
                    nidx[d] = idx[d] + off[d];
                    if nidx[d] < 0 || nidx[d] >= size[d] as i64 {
                        // Outside the image: the boundary condition supplies
                        // `label + 1`, which is never equal to `label`.
                        return true;
                    }
                }
                labels[flat(&nidx)] != label
            });
            if on_border {
                border.push(idx);
            }
            idx[0] += 1;
        }
    }

    let mut feret_diameter_sq = 0.0f64;
    for (i, a) in border.iter().enumerate() {
        for b in &border[i + 1..] {
            let mut length = 0.0;
            for d in 0..dim {
                let diff = (a[d] - b[d]) as f64 * spacing[d];
                length += diff * diff;
            }
            if feret_diameter_sq < length {
                feret_diameter_sq = length;
            }
        }
    }
    feret_diameter_sq.sqrt()
}

/// Every offset in `{-1, 0, 1}^dim`, center included — `ConstNeighborhoodIterator`
/// iterates its full `3^dim` neighborhood and the center pixel can never
/// differ from the label.
fn neighborhood_offsets(dim: usize) -> Vec<[i64; MAX_DIM]> {
    let mut out = vec![[0i64; MAX_DIM]];
    for d in 0..dim {
        let mut next = Vec::with_capacity(out.len() * 3);
        for base in &out {
            for delta in [-1i64, 0, 1] {
                let mut o = *base;
                o[d] = delta;
                next.push(o);
            }
        }
        out = next;
    }
    out
}

// ---- ComputePerimeter -----------------------------------------------------

/// Every offset in `{-1, 0, 1}^rest` except the origin — what
/// `setConnectivity(&lIt, true)` activates on the `dim - 1` dimensional
/// image of line lists.
fn rest_offsets(rest: usize) -> Vec<[i64; 2]> {
    let mut out: Vec<[i64; 2]> = Vec::new();
    if rest == 1 {
        for a in [-1i64, 1] {
            out.push([a, 0]);
        }
    } else {
        for a in [-1i64, 0, 1] {
            for b in [-1i64, 0, 1] {
                if a != 0 || b != 0 {
                    out.push([a, b]);
                }
            }
        }
    }
    out
}

/// `ShapeLabelMapFilter::ComputePerimeter`.
///
/// The `dim - 1` dimensional "line image" of `itkShapeLabelMapFilter.hxx` is
/// a map from `idx[1..]` to that bucket's lines. ITK pads the image by one
/// so out-of-bounding-box neighbours read back an empty list; a missing map
/// key does the same thing here.
fn compute_perimeter(
    lines: &[LabelObjectLine],
    bb: &BoundingBox,
    dim: usize,
    spacing: &[f64],
) -> f64 {
    let rest = dim - 1;

    let key_of = |idx: &[i64; MAX_DIM]| -> [i64; 2] {
        let mut k = [0i64; 2];
        k[..rest].copy_from_slice(&idx[1..=rest]);
        k
    };

    let mut line_image: HashMap<[i64; 2], Vec<LabelObjectLine>> = HashMap::new();
    for line in lines {
        line_image
            .entry(key_of(&line.index()))
            .or_default()
            .push(*line);
    }
    let empty: Vec<LabelObjectLine> = Vec::new();

    let offsets = rest_offsets(rest);
    let mut intercepts: HashMap<[i64; MAX_DIM], u64> = HashMap::new();

    // Iterate the bounding box in the non-scanline axes.
    let extent: Vec<i64> = (0..rest).map(|i| bb.size[i + 1] as i64).collect();
    let n_centers: i64 = extent.iter().product();
    for c in 0..n_centers {
        let mut center = [0i64; 2];
        let mut t = c;
        for i in 0..rest {
            center[i] = bb.index[i + 1] + t % extent[i];
            t /= extent[i];
        }

        let ls = line_image.get(&center).unwrap_or(&empty);

        // There are two intercepts on the 0 axis for each line.
        let mut no = [0i64; MAX_DIM];
        no[0] = 1;
        *intercepts.entry(no).or_insert(0) += 2 * ls.len() as u64;

        for off in &offsets {
            let mut neighbor = center;
            for i in 0..rest {
                neighbor[i] += off[i];
            }
            let ns = line_image.get(&neighbor).unwrap_or(&empty);

            let mut no = [0i64; MAX_DIM];
            for i in 0..rest {
                no[i + 1] = off[i].abs();
            }
            let mut dno = no;
            dno[0] = 1;

            if ns.is_empty() {
                // No line in the neighbor: all the lines in `ls` are on the
                // contour.
                for l in ls {
                    *intercepts.entry(no).or_insert(0) += l.length() as u64;
                    *intercepts.entry(dno).or_insert(0) += (l.length() * 2) as u64;
                }
                continue;
            }

            // Walk both sorted line lists, intersecting each line of `ls`
            // with the *gaps* between the neighbor's lines.
            let mut li = 0usize;
            let mut ni = 0usize;
            let mut n_min = i64::MIN + 1;
            let mut n_max = ns[0].index()[0] - 1;

            while li < ls.len() {
                let l_min = ls[li].index()[0];
                let l_max = l_min + ls[li].length() - 1;

                let straight = (l_max.min(n_max) - l_min.max(n_min) + 1).max(0);
                *intercepts.entry(no).or_insert(0) += straight as u64;

                let left = (l_max.min(n_max + 1) - l_min.max(n_min + 1) + 1).max(0);
                let right = (l_max.min(n_max - 1) - l_min.max(n_min - 1) + 1).max(0);
                *intercepts.entry(dno).or_insert(0) += (left + right) as u64;

                if n_max <= l_max {
                    n_min = ns[ni].index()[0] + ns[ni].length();
                    ni += 1;
                    n_max = if ni < ns.len() {
                        ns[ni].index()[0] - 1
                    } else {
                        i64::MAX - 1
                    };
                } else {
                    li += 1;
                }
            }
        }
    }

    perimeter_from_intercept_count(&intercepts, dim, spacing)
}

fn intercept(intercepts: &HashMap<[i64; MAX_DIM], u64>, key: [i64; MAX_DIM]) -> f64 {
    *intercepts.get(&key).unwrap_or(&0) as f64
}

/// `ShapeLabelMapFilter::PerimeterFromInterceptCount`, the 2-D and 3-D
/// overloads that `ITK_DO_NOT_USE_PERIMETER_SPECIALIZATION` leaves enabled.
fn perimeter_from_intercept_count(
    intercepts: &HashMap<[i64; MAX_DIM], u64>,
    dim: usize,
    spacing: &[f64],
) -> f64 {
    if dim == 2 {
        let dx = spacing[0];
        let dy = spacing[1];
        let norm = (dx * dx + dy * dy).sqrt();

        let mut perimeter = 0.0;
        perimeter += dy * intercept(intercepts, [1, 0, 0]) / 2.0;
        perimeter += dx * intercept(intercepts, [0, 1, 0]) / 2.0;
        perimeter += dx * dy / norm * intercept(intercepts, [1, 1, 0]) / 2.0;
        perimeter * (std::f64::consts::PI / 4.0)
    } else {
        let dx = spacing[0];
        let dy = spacing[1];
        let dz = spacing[2];
        let dxy = (dx * dx + dy * dy).sqrt();
        let dxz = (dx * dx + dz * dz).sqrt();
        let dyz = (dy * dy + dz * dz).sqrt();
        let dxyz = (dx * dx + dy * dy + dz * dz).sqrt();
        let vol = dx * dy * dz;

        // 'magical numbers', corresponding to the area of the Voronoi
        // partition on the unit sphere when the germs are the 26 directions
        // on the unit cube. Sum of (c1+c2+c3 + c4*2+c5*2+c6*2 + c7*4) is 1.
        // ITK's `TODO - recompute those values if the spacing is non
        // isotropic` is reproduced, not fixed.
        let c1 = 0.045_777_891_204_76 * 2.0; // Ox
        let c2 = 0.045_777_891_204_76 * 2.0; // Oy
        let c3 = 0.045_777_891_204_76 * 2.0; // Oz
        let c4 = 0.036_980_627_876_08 * 2.0; // Oxy
        let c5 = 0.036_980_627_876_08 * 2.0; // Oxz
        let c6 = 0.036_980_627_876_08 * 2.0; // Oyz
        let c7 = 0.035_195_639_782_32 * 2.0; // Oxyz

        let mut perimeter = 0.0;
        perimeter += vol / dx * intercept(intercepts, [1, 0, 0]) / 2.0 * c1;
        perimeter += vol / dy * intercept(intercepts, [0, 1, 0]) / 2.0 * c2;
        perimeter += vol / dz * intercept(intercepts, [0, 0, 1]) / 2.0 * c3;
        perimeter += vol / dxy * intercept(intercepts, [1, 1, 0]) / 2.0 * c4;
        perimeter += vol / dxz * intercept(intercepts, [1, 0, 1]) / 2.0 * c5;
        perimeter += vol / dyz * intercept(intercepts, [0, 1, 1]) / 2.0 * c6;
        perimeter += vol / dxyz * intercept(intercepts, [1, 1, 1]) / 2.0 * c7;
        perimeter * 4.0
    }
}

// ---- ComputeOrientedBoundingBox -------------------------------------------

/// `ShapeLabelMapFilter::ComputeOrientedBoundingBox`, plus
/// `ShapeLabelObject::GetOrientedBoundingBoxVertices`.
fn compute_oriented_bounding_box(
    lines: &[LabelObjectLine],
    dim: usize,
    spacing: &[f64],
    origin: &[f64],
    direction: &[f64],
    centroid: &[f64],
    principal_axes: &[f64],
) -> OrientedBoundingBox {
    // Row `i` of the basis matrix is the `i`-th principal axis.
    let axis = |i: usize, j: usize| principal_axes[i * dim + j];

    // Physical points of the start and end of each RLE line, relative to the
    // centroid, projected onto the principal axes.
    let mut min_pa = [f64::INFINITY; MAX_DIM];
    let mut max_pa = [f64::NEG_INFINITY; MAX_DIM];
    for line in lines {
        let mut idx = line.index();
        for _ in 0..2 {
            let pt = index_to_physical(&idx, dim, spacing, origin, direction);
            for i in 0..dim {
                let mut v = 0.0;
                for (j, &c) in centroid.iter().enumerate().take(dim) {
                    v += axis(i, j) * (pt[j] - c);
                }
                min_pa[i] = min_pa[i].min(v);
                max_pa[i] = max_pa[i].max(v);
            }
            idx[0] = line.index()[0] + line.length() - 1;
        }
    }

    // The extrema so far run from pixel center to pixel center. Widen them by
    // the offset from a pixel's center to each of its 2^dim corners, measured
    // in the principal-axis basis.
    let mut adj_min = min_pa;
    let mut adj_max = max_pa;
    for p in 0..(1usize << dim) {
        let mut spacing_axis = [0.0f64; MAX_DIM];
        for (i, sa) in spacing_axis.iter_mut().enumerate().take(dim) {
            *sa = spacing[i] * 0.5;
            if p & (1usize << i) != 0 {
                *sa = -*sa;
            }
        }
        // TransformLocalVectorToPhysicalVector: Direction * v, no spacing.
        let mut physical_offset = [0.0f64; MAX_DIM];
        for (i, po) in physical_offset.iter_mut().enumerate().take(dim) {
            for (j, &sa) in spacing_axis.iter().enumerate().take(dim) {
                *po += direction[i * dim + j] * sa;
            }
        }
        for i in 0..dim {
            let mut pa_offset = 0.0;
            for (j, &po) in physical_offset.iter().enumerate().take(dim) {
                pa_offset += axis(i, j) * po;
            }
            adj_min[i] = adj_min[i].min(min_pa[i] + pa_offset);
            adj_max[i] = adj_max[i].max(max_pa[i] + pa_offset);
        }
    }

    let size: Vec<f64> = (0..dim).map(|i| (adj_max[i] - adj_min[i]).abs()).collect();

    // Rotate the minimum corner back into physical space.
    let obb_origin: Vec<f64> = (0..dim)
        .map(|i| (0..dim).map(|j| axis(j, i) * adj_min[j]).sum::<f64>() + centroid[i])
        .collect();

    let msb = 1usize << (dim - 1);
    let vertices: Vec<Vec<f64>> = (0..(1usize << dim))
        .map(|v| {
            let offset: Vec<f64> = (0..dim)
                .map(|j| if v & (msb >> j) != 0 { size[j] } else { 0.0 })
                .collect();
            (0..dim)
                .map(|k| obb_origin[k] + (0..dim).map(|j| axis(j, k) * offset[j]).sum::<f64>())
                .collect()
        })
        .collect();

    OrientedBoundingBox {
        size,
        origin: obb_origin,
        direction: principal_axes.to_vec(),
        vertices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    // L3 (coord-rounding-port-map.md §5): the fractional shape centroid is
    // converted with ITK's CONTINUOUS method (origin last), while second-moment
    // accumulation uses the INTEGER method (origin first). At a large origin with
    // a shear direction (so one row gets two terms) the two folds differ; before
    // the fix the centroid shared the integer origin-first fold and diverged from
    // ITK's TransformContinuousIndexToPhysicalPoint.
    #[test]
    fn centroid_uses_continuous_origin_last_fold_unlike_integer_moments() {
        let dim = 2;
        let spacing = [1.0, 1.0];
        let origin = [1e16, 0.0];
        let direction = [1.0, 1.0, 0.0, 1.0]; // shear
        let mut cidx = [0.0; MAX_DIM];
        cidx[0] = 1.0;
        cidx[1] = 1.0;
        let mut iidx = [0i64; MAX_DIM];
        iidx[0] = 1;
        iidx[1] = 1;
        let centroid = continuous_index_to_physical(&cidx, dim, &spacing, &origin, &direction);
        let moment = index_to_physical(&iidx, dim, &spacing, &origin, &direction);
        assert_eq!(centroid[0], 1.0000000000000002e16); // ((1 + 1) + origin)
        assert_eq!(moment[0], 1e16); // ((origin + 1) + 1)
        // Non-vacuity: identical bits would mean the centroid kept the integer
        // origin-first fold.
        assert_ne!(centroid[0], moment[0]);
    }

    fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs expected {b}");
    }

    fn assert_slice_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "{what}[{i}]: {x} vs expected {y}");
        }
    }

    /// A `w × h × ...` axis-aligned box of `label` with its lowest corner at
    /// `index`, inside an otherwise-background image of `size`.
    fn box_image(size: &[usize], index: &[i64], extent: &[i64], label: i64) -> Image {
        let n: usize = size.iter().product();
        let mut data = vec![0i32; n];
        let dim = size.len();
        let mut strides = vec![1usize; dim];
        for d in 1..dim {
            strides[d] = strides[d - 1] * size[d - 1];
        }
        let total: i64 = extent.iter().product();
        for k in 0..total {
            let mut t = k;
            let mut flat = 0usize;
            for d in 0..dim {
                let i = index[d] + t % extent[d];
                t /= extent[d];
                flat += i as usize * strides[d];
            }
            data[flat] = label as i32;
        }
        Image::from_vec(size, data).unwrap()
    }

    // ---- 2-D: reproduces SimpleITK's own LabelShapeStatistics test ---------
    //
    // `LabelShapeStatisticsImageFilter.yaml`'s `SimpleLabelB` case has label
    // 50 covering exactly its 17 x 31 bounding box (527 = 17 * 31 pixels), so
    // it is an axis-aligned rectangle at index (17, 12) and every expected
    // value in the yaml applies verbatim to the rectangle below.
    #[test]
    fn rectangle_matches_simpleitk_expected_values() {
        let img = box_image(&[64, 64], &[17, 12], &[17, 31], 50);
        let settings = LabelShapeStatisticsSettings {
            compute_oriented_bounding_box: true,
            ..Default::default()
        };
        let stats = label_shape_statistics(&img, &settings).unwrap();
        assert_eq!(stats.keys().copied().collect::<Vec<_>>(), vec![50]);
        let s = &stats[&50];

        assert_eq!(s.number_of_pixels, 527);
        assert_close(s.physical_size, 527.0, 0.0, "physical_size");
        assert_eq!(s.bounding_box.index, vec![17, 12]);
        assert_eq!(s.bounding_box.size, vec![17, 31]);
        assert_slice_close(&s.centroid, &[25.0, 27.0], 1e-12, "centroid");
        assert_close(s.elongation, 1.825_741_858_350_553_8, 1e-8, "elongation");
        assert_close(s.flatness, 1.825_741_858_350_553_8, 1e-8, "flatness");
        assert_slice_close(
            &s.equivalent_ellipsoid_diameter,
            &[19.170_819_6, 35.000_967_8],
            1e-6,
            "equivalent_ellipsoid_diameter",
        );
        assert_close(
            s.equivalent_spherical_perimeter,
            81.378_604_766_654_04,
            1e-8,
            "equivalent_spherical_perimeter",
        );
        assert_close(
            s.equivalent_spherical_radius,
            12.951_807_210_534_664,
            1e-8,
            "equivalent_spherical_radius",
        );
        assert_eq!(s.number_of_pixels_on_border, 0);
        assert_close(
            s.perimeter.unwrap(),
            89.902_986_366_438_31,
            1e-8,
            "perimeter",
        );
        assert_close(s.perimeter_on_border, 0.0, 0.0, "perimeter_on_border");
        assert_close(
            s.perimeter_on_border_ratio.unwrap(),
            0.0,
            0.0,
            "perimeter_on_border_ratio",
        );
        assert_slice_close(
            &s.principal_axes,
            &[1.0, 0.0, 0.0, 1.0],
            1e-12,
            "principal_axes",
        );
        assert_slice_close(
            &s.principal_moments,
            &[24.0, 80.0],
            1e-9,
            "principal_moments",
        );
        assert_close(
            s.roundness.unwrap(),
            0.905_182_442_271_278,
            1e-8,
            "roundness",
        );

        let obb = s.oriented_bounding_box.as_ref().unwrap();
        assert_slice_close(&obb.size, &[17.0, 31.0], 1e-9, "obb size");
        assert_slice_close(&obb.origin, &[16.5, 11.5], 1e-9, "obb origin");
        assert_slice_close(
            &obb.direction,
            &[1.0, 0.0, 0.0, 1.0],
            1e-12,
            "obb direction",
        );
        let flat: Vec<f64> = obb.vertices.iter().flatten().copied().collect();
        assert_slice_close(
            &flat,
            &[16.5, 11.5, 16.5, 42.5, 33.5, 11.5, 33.5, 42.5],
            1e-9,
            "obb vertices",
        );
    }

    // ---- Perimeter: closed-form intercept counts for a box ----------------
    //
    // For a `w x h` (2-D) or `w x h x d` (3-D) box strictly inside the image,
    // the intercept counts of `ComputePerimeter` can be written down by hand:
    // every bucket of the line image holds exactly one line of length `w`, a
    // present neighbour contributes 2 diagonal intercepts and 0 straight ones,
    // and an absent neighbour contributes `w` straight and `2w` diagonal.

    fn expected_perimeter_2d(w: f64, h: f64, dx: f64, dy: f64) -> f64 {
        let nx = 2.0 * h;
        let ny = 2.0 * w;
        let nxy = 4.0 * w + 4.0 * (h - 1.0);
        let norm = (dx * dx + dy * dy).sqrt();
        (dy * nx / 2.0 + dx * ny / 2.0 + dx * dy / norm * nxy / 2.0) * std::f64::consts::PI / 4.0
    }

    fn expected_perimeter_3d(w: f64, h: f64, d: f64, dx: f64, dy: f64, dz: f64) -> f64 {
        let nx = 2.0 * h * d;
        let ny = 2.0 * w * d;
        let nz = 2.0 * w * h;
        let nyz = 4.0 * w * (h + d - 1.0);
        let nxy = 4.0 * w * d + 4.0 * d * (h - 1.0);
        let nxz = 4.0 * w * h + 4.0 * h * (d - 1.0);
        let nxyz = 8.0 * w * (h + d - 1.0) + 8.0 * (h - 1.0) * (d - 1.0);

        let dxy = (dx * dx + dy * dy).sqrt();
        let dxz = (dx * dx + dz * dz).sqrt();
        let dyz = (dy * dy + dz * dz).sqrt();
        let dxyz = (dx * dx + dy * dy + dz * dz).sqrt();
        let vol = dx * dy * dz;
        let c1 = 0.045_777_891_204_76 * 2.0;
        let c4 = 0.036_980_627_876_08 * 2.0;
        let c7 = 0.035_195_639_782_32 * 2.0;

        4.0 * (vol / dx * nx / 2.0 * c1
            + vol / dy * ny / 2.0 * c1
            + vol / dz * nz / 2.0 * c1
            + vol / dxy * nxy / 2.0 * c4
            + vol / dxz * nxz / 2.0 * c4
            + vol / dyz * nyz / 2.0 * c4
            + vol / dxyz * nxyz / 2.0 * c7)
    }

    #[test]
    fn perimeter_2d_box_matches_closed_form() {
        for (w, h) in [(1i64, 1i64), (1, 5), (5, 1), (3, 7), (17, 31)] {
            let img = box_image(&[40, 40], &[4, 5], &[w, h], 1);
            let stats = label_shape_statistics(&img, &Default::default()).unwrap();
            let got = stats[&1].perimeter.unwrap();
            let want = expected_perimeter_2d(w as f64, h as f64, 1.0, 1.0);
            assert_close(got, want, 1e-9, &format!("perimeter of {w}x{h}"));
        }
    }

    #[test]
    fn perimeter_3d_box_matches_closed_form() {
        for (w, h, d) in [
            (1i64, 1i64, 1i64),
            (3, 1, 1),
            (1, 4, 1),
            (1, 1, 6),
            (3, 4, 5),
        ] {
            let img = box_image(&[12, 12, 12], &[2, 3, 4], &[w, h, d], 1);
            let stats = label_shape_statistics(&img, &Default::default()).unwrap();
            let got = stats[&1].perimeter.unwrap();
            let want = expected_perimeter_3d(w as f64, h as f64, d as f64, 1.0, 1.0, 1.0);
            assert_close(got, want, 1e-9, &format!("perimeter of {w}x{h}x{d}"));
        }
    }

    /// The Crofton weights are also exercised through the anisotropic-spacing
    /// path: the intercept counts do not depend on spacing, only the weights
    /// do.
    #[test]
    fn perimeter_2d_anisotropic_spacing_matches_closed_form() {
        let mut img = box_image(&[40, 40], &[4, 5], &[3, 7], 1);
        img.set_spacing(&[2.0, 0.5]).unwrap();
        let stats = label_shape_statistics(&img, &Default::default()).unwrap();
        assert_close(
            stats[&1].perimeter.unwrap(),
            expected_perimeter_2d(3.0, 7.0, 2.0, 0.5),
            1e-9,
            "anisotropic perimeter",
        );
    }

    /// A single voxel exercises the "no neighbour has any line" branch on
    /// every one of the eight neighbours of the 2-D line image.
    #[test]
    fn single_voxel_3d_perimeter() {
        let img = box_image(&[5, 5, 5], &[2, 2, 2], &[1, 1, 1], 7);
        let stats = label_shape_statistics(&img, &Default::default()).unwrap();
        let s = &stats[&7];
        assert_eq!(s.number_of_pixels, 1);
        // c1*3 + (2/sqrt(2))*c4*3 + (4/sqrt(3))*c7, all times 4.
        let c1 = 0.045_777_891_204_76 * 2.0;
        let c4 = 0.036_980_627_876_08 * 2.0;
        let c7 = 0.035_195_639_782_32 * 2.0;
        let want = 4.0 * (3.0 * c1 + 3.0 * (2.0 / 2f64.sqrt()) * c4 + (4.0 / 3f64.sqrt()) * c7);
        assert_close(s.perimeter.unwrap(), want, 1e-12, "single voxel perimeter");
        // Degenerate object: all central moments vanish.
        assert_slice_close(&s.principal_moments, &[0.0, 0.0, 0.0], 1e-12, "moments");
        assert_close(s.elongation, 0.0, 0.0, "elongation");
        assert_close(s.flatness, 0.0, 0.0, "flatness");
    }

    // ---- Principal moments / axes -----------------------------------------

    /// Variance of `w` unit-spaced samples is `(w^2 - 1)/12`; with spacing
    /// `s` it scales by `s^2`. The cross-moment of an axis-aligned box is
    /// zero, so the principal moments are those two variances, sorted
    /// ascending, and the axes are the coordinate axes in that order — with
    /// the last row negated when the swap makes the matrix a reflection.
    #[test]
    fn principal_moments_axis_aligned_box_isotropic() {
        // 3 wide, 5 tall: var_x = 8/12 = 2/3 < var_y = 24/12 = 2.
        let img = box_image(&[20, 20], &[3, 4], &[3, 5], 1);
        let s = &label_shape_statistics(&img, &Default::default()).unwrap()[&1];
        assert_slice_close(&s.principal_moments, &[2.0 / 3.0, 2.0], 1e-12, "moments");
        assert_slice_close(&s.principal_axes, &[1.0, 0.0, 0.0, 1.0], 1e-12, "axes");
        assert_close(determinant_of(&s.principal_axes, 2), 1.0, 1e-12, "det");
        assert_close(s.elongation, 3f64.sqrt(), 1e-12, "elongation");
        assert_slice_close(&s.centroid, &[4.0, 6.0], 1e-12, "centroid");
    }

    #[test]
    fn principal_moments_axis_aligned_box_anisotropic_spacing() {
        // 3 wide at spacing 2: var_x = 4 * 8/12 = 8/3.
        // 5 tall at spacing 1: var_y = 24/12 = 2 < 8/3, so the axes swap.
        let mut img = box_image(&[20, 20], &[3, 4], &[3, 5], 1);
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let s = &label_shape_statistics(&img, &Default::default()).unwrap()[&1];
        assert_slice_close(&s.principal_moments, &[2.0, 8.0 / 3.0], 1e-12, "moments");
        // Row 0 = y axis, row 1 = x axis; the swap has determinant -1, so the
        // last row is negated to restore a proper rotation.
        assert_slice_close(&s.principal_axes, &[0.0, 1.0, -1.0, 0.0], 1e-12, "axes");
        assert_close(determinant_of(&s.principal_axes, 2), 1.0, 1e-12, "det");
        assert_close(s.elongation, (4.0f64 / 3.0).sqrt(), 1e-12, "elongation");
        assert_close(s.physical_size, 3.0 * 5.0 * 2.0, 1e-12, "physical_size");
        assert_slice_close(&s.centroid, &[8.0, 6.0], 1e-12, "centroid");
    }

    #[test]
    fn principal_moments_3d_box() {
        // 3 x 5 x 7 at spacing 1: variances 2/3, 2, 4.
        let img = box_image(&[12, 12, 14], &[1, 2, 3], &[3, 5, 7], 1);
        let s = &label_shape_statistics(&img, &Default::default()).unwrap()[&1];
        assert_slice_close(
            &s.principal_moments,
            &[2.0 / 3.0, 2.0, 4.0],
            1e-11,
            "moments",
        );
        assert_slice_close(
            &s.principal_axes,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            1e-11,
            "axes",
        );
        // flatness = sqrt(pm1/pm0), elongation = sqrt(pm2/pm1).
        assert_close(s.flatness, 3f64.sqrt(), 1e-11, "flatness");
        assert_close(s.elongation, 2f64.sqrt(), 1e-11, "elongation");
        assert_close(s.physical_size, 105.0, 1e-12, "physical_size");
        assert_close(
            s.equivalent_spherical_radius,
            (105.0 * 3.0 / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0),
            1e-12,
            "equivalent_spherical_radius",
        );
        assert_close(
            s.equivalent_spherical_perimeter,
            4.0 * std::f64::consts::PI * s.equivalent_spherical_radius.powi(2),
            1e-9,
            "equivalent_spherical_perimeter",
        );
    }

    fn determinant_of(m: &[f64], n: usize) -> f64 {
        let mut a: Mat = [[0.0; MAX_DIM]; MAX_DIM];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = m[i * n + j];
            }
        }
        determinant(&a, n)
    }

    /// An object elongated along the `(1, 1)` diagonal: its principal axes are
    /// the diagonals, not the coordinate axes, so this exercises the Jacobi
    /// rotation itself rather than the already-diagonal shortcut.
    #[test]
    fn principal_axes_of_a_diagonally_elongated_object() {
        let mut data = vec![0i32; 21 * 21];
        // |u| + 3|v| <= 9 with u = dx + dy (long diagonal), v = dx - dy.
        for y in 0..21i64 {
            for x in 0..21i64 {
                let (dx, dy) = (x - 10, y - 10);
                if (dx + dy).abs() + 3 * (dx - dy).abs() <= 9 {
                    data[(y * 21 + x) as usize] = 3;
                }
            }
        }
        let img = Image::from_vec(&[21, 21], data).unwrap();
        let s = &label_shape_statistics(&img, &Default::default()).unwrap()[&3];
        let a = &s.principal_axes;

        assert_close(determinant_of(a, 2), 1.0, 1e-12, "det");
        assert_close(a[0] * a[0] + a[1] * a[1], 1.0, 1e-12, "row0 norm");
        assert_close(a[2] * a[2] + a[3] * a[3], 1.0, 1e-12, "row1 norm");
        assert_close(a[0] * a[2] + a[1] * a[3], 0.0, 1e-12, "orthogonal");
        assert!(s.principal_moments[0] < s.principal_moments[1]);

        // The smallest moment's axis is the short diagonal (1, -1)/sqrt(2);
        // the largest moment's axis is the long diagonal (1, 1)/sqrt(2).
        let r = std::f64::consts::FRAC_1_SQRT_2;
        assert_close((a[0] * r - a[1] * r).abs(), 1.0, 1e-9, "row0 along (1,-1)");
        assert_close((a[2] * r + a[3] * r).abs(), 1.0, 1e-9, "row1 along (1,1)");

        // The moment matrix is symmetric under (dx, dy) -> (dy, dx), so the two
        // diagonal moments are equal and the eigenvalues are Mxx -/+ Mxy.
        assert_close(s.centroid[0], 10.0, 1e-12, "centroid x");
        assert_close(s.centroid[1], 10.0, 1e-12, "centroid y");
    }

    // ---- Border counts ----------------------------------------------------

    #[test]
    fn border_counts_for_a_corner_box() {
        // 3 x 4 box at the origin of a 10 x 10 image: its left column and top
        // row lie on the image border.
        let img = box_image(&[10, 10], &[0, 0], &[3, 4], 1);
        let s = &label_shape_statistics(&img, &Default::default()).unwrap()[&1];
        // Lines y=0 (on border, all 3 pixels) and y=1..3 (x=0 only).
        assert_eq!(s.number_of_pixels_on_border, 3 + 3);
        // perimeterOnBorder: each of the 4 lines starts at x=0 -> +1 each.
        // Lines with y == 0 add `length` for the y border.
        assert_close(
            s.perimeter_on_border,
            4.0 + 3.0,
            1e-12,
            "perimeter_on_border",
        );
        assert_close(
            s.perimeter_on_border_ratio.unwrap(),
            7.0 / s.perimeter.unwrap(),
            1e-12,
            "perimeter_on_border_ratio",
        );
    }

    /// A one-pixel-wide image along axis 0: `idx[0]` is simultaneously
    /// `borderMin[0]` and `borderMax[0]`, and ITK's `!isOnBorder0 || length > 1`
    /// guard makes the pixel count once, not twice.
    #[test]
    fn border_count_single_column_counts_once() {
        let img = box_image(&[1, 4], &[0, 1], &[1, 2], 1);
        let s = &label_shape_statistics(&img, &Default::default()).unwrap()[&1];
        assert_eq!(s.number_of_pixels, 2);
        assert_eq!(s.number_of_pixels_on_border, 2);
        // But perimeterOnBorder's two `if`s are unconditional, so each line
        // contributes twice along axis 0.
        assert_close(s.perimeter_on_border, 4.0, 1e-12, "perimeter_on_border");
    }

    /// A line of length > 1 whose start is on `borderMin[0]` and whose end is
    /// on `borderMax[0]` counts both endpoints.
    #[test]
    fn border_count_full_width_line_counts_both_ends() {
        let img = box_image(&[4, 4], &[0, 1], &[4, 1], 1);
        let s = &label_shape_statistics(&img, &Default::default()).unwrap()[&1];
        assert_eq!(s.number_of_pixels, 4);
        assert_eq!(s.number_of_pixels_on_border, 2);
    }

    // ---- Feret diameter ---------------------------------------------------

    #[test]
    fn feret_diameter_of_a_rectangle_is_its_diagonal() {
        let img = box_image(&[20, 20], &[3, 4], &[3, 5], 1);
        let settings = LabelShapeStatisticsSettings {
            compute_feret_diameter: true,
            ..Default::default()
        };
        let s = &label_shape_statistics(&img, &settings).unwrap()[&1];
        assert_close(
            s.feret_diameter.unwrap(),
            (4.0f64 + 16.0).sqrt(),
            1e-12,
            "feret_diameter",
        );
    }

    #[test]
    fn feret_diameter_uses_spacing_not_direction() {
        let mut img = box_image(&[20, 20], &[3, 4], &[3, 5], 1);
        img.set_spacing(&[2.0, 3.0]).unwrap();
        // A 90-degree rotation of the direction cosines must not change it.
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let settings = LabelShapeStatisticsSettings {
            compute_feret_diameter: true,
            ..Default::default()
        };
        let s = &label_shape_statistics(&img, &settings).unwrap()[&1];
        assert_close(
            s.feret_diameter.unwrap(),
            ((2.0f64 * 2.0).powi(2) + (4.0f64 * 3.0).powi(2)).sqrt(),
            1e-12,
            "feret_diameter",
        );
    }

    #[test]
    fn feret_diameter_single_pixel_is_zero() {
        let img = box_image(&[5, 5], &[2, 2], &[1, 1], 1);
        let settings = LabelShapeStatisticsSettings {
            compute_feret_diameter: true,
            ..Default::default()
        };
        let s = &label_shape_statistics(&img, &settings).unwrap()[&1];
        assert_close(s.feret_diameter.unwrap(), 0.0, 0.0, "feret_diameter");
    }

    // ---- Gates ------------------------------------------------------------

    #[test]
    fn gated_attributes_default_to_none() {
        let img = box_image(&[10, 10], &[2, 2], &[3, 4], 1);
        let s = &label_shape_statistics(&img, &Default::default()).unwrap()[&1];
        assert!(s.feret_diameter.is_none());
        assert!(s.oriented_bounding_box.is_none());
        // ComputePerimeter defaults to true in SimpleITK.
        assert!(s.perimeter.is_some());
        assert!(s.roundness.is_some());
        assert!(s.perimeter_on_border_ratio.is_some());
    }

    #[test]
    fn perimeter_gate_off_suppresses_perimeter_derived_values() {
        let img = box_image(&[10, 10], &[2, 2], &[3, 4], 1);
        let settings = LabelShapeStatisticsSettings {
            compute_perimeter: false,
            ..Default::default()
        };
        let s = &label_shape_statistics(&img, &settings).unwrap()[&1];
        assert!(s.perimeter.is_none());
        assert!(s.roundness.is_none());
        assert!(s.perimeter_on_border_ratio.is_none());
        // Un-gated attributes are still there.
        assert_eq!(s.number_of_pixels, 12);
    }

    // ---- Labels and background --------------------------------------------

    #[test]
    fn multiple_labels_are_keyed_ascending_and_background_is_excluded() {
        let mut data = vec![0i32; 8 * 8];
        data[0] = 5; // (0,0)
        data[9] = 2; // (1,1)
        data[10] = 2; // (2,1)
        let img = Image::from_vec(&[8, 8], data).unwrap();
        let stats = label_shape_statistics(&img, &Default::default()).unwrap();
        assert_eq!(stats.keys().copied().collect::<Vec<_>>(), vec![2, 5]);
        assert_eq!(stats[&2].number_of_pixels, 2);
        assert_eq!(stats[&5].number_of_pixels, 1);
    }

    #[test]
    fn non_zero_background_value_relabels_what_counts_as_object() {
        let mut data = vec![0i32; 6 * 6];
        data[7] = 4; // (1,1)
        data[8] = 4; // (2,1)
        let img = Image::from_vec(&[6, 6], data).unwrap();
        let settings = LabelShapeStatisticsSettings {
            background_value: 4.0,
            ..Default::default()
        };
        let stats = label_shape_statistics(&img, &settings).unwrap();
        // Now 0 is an object and 4 is background.
        assert_eq!(stats.keys().copied().collect::<Vec<_>>(), vec![0]);
        assert_eq!(stats[&0].number_of_pixels, 34);
    }

    // ---- Errors -----------------------------------------------------------

    #[test]
    fn floating_point_input_is_rejected() {
        let img = Image::new(&[4, 4], PixelId::Float32);
        let err = label_shape_statistics(&img, &Default::default()).unwrap_err();
        assert!(matches!(err, FilterError::RequiresIntegerPixelType(_)));
    }

    #[test]
    fn one_dimensional_input_is_rejected() {
        let img = Image::new(&[4], PixelId::UInt8);
        let err = label_shape_statistics(&img, &Default::default()).unwrap_err();
        assert!(matches!(err, FilterError::UnsupportedShapeDimension(1)));
    }

    // ---- GeometryUtilities ------------------------------------------------

    #[test]
    fn hyper_sphere_helpers_match_the_closed_forms() {
        // 1-D "sphere" of radius 1 is the segment [-1, 1], volume 2. Used as
        // the denominator of the 2-D Crofton constant.
        assert_close(hyper_sphere_volume(1, 1.0), 2.0, 1e-12, "volume 1d");
        assert_close(
            hyper_sphere_volume(2, 3.0),
            std::f64::consts::PI * 9.0,
            1e-12,
            "volume 2d",
        );
        assert_close(
            hyper_sphere_volume(3, 2.0),
            4.0 / 3.0 * std::f64::consts::PI * 8.0,
            1e-12,
            "volume 3d",
        );
        assert_close(
            hyper_sphere_perimeter(2, 3.0),
            2.0 * std::f64::consts::PI * 3.0,
            1e-12,
            "perimeter 2d",
        );
        assert_close(
            hyper_sphere_perimeter(3, 2.0),
            4.0 * std::f64::consts::PI * 4.0,
            1e-12,
            "perimeter 3d",
        );
        assert_close(
            hyper_sphere_radius_from_volume(2, std::f64::consts::PI * 9.0),
            3.0,
            1e-12,
            "radius 2d",
        );
        assert_close(
            hyper_sphere_radius_from_volume(3, 4.0 / 3.0 * std::f64::consts::PI * 8.0),
            2.0,
            1e-12,
            "radius 3d",
        );
    }

    #[test]
    fn is_almost_zero_matches_the_zero_point_one_epsilon_window() {
        // `AlmostEquals(x, 0.0)` collapses to `|x| <= 0.1 * eps(double)`
        // (~2.22e-17), so every subnormal and every value up to that bound is
        // almost-zero, and anything above it is not.
        assert!(is_almost_zero(0.0));
        assert!(is_almost_zero(-0.0));
        assert!(is_almost_zero(f64::from_bits(4)));
        // Both were `false` under the old 4-ULP bit window; upstream's
        // 0.1*eps window makes them almost-zero (§2.154).
        assert!(is_almost_zero(f64::from_bits(5)));
        assert!(is_almost_zero(1e-300));
        // Upper edge of the window: `0.1*eps` is inside, `eps` is outside.
        assert!(is_almost_zero(0.1 * f64::EPSILON));
        assert!(!is_almost_zero(f64::EPSILON));
    }

    // ---- Oriented bounding box --------------------------------------------

    /// With a non-identity direction matrix the OBB must still hug the object:
    /// its size is spacing-scaled extents, and its origin sits half a pixel
    /// outside the extreme pixel centers along each principal axis.
    #[test]
    fn oriented_bounding_box_of_a_box_with_anisotropic_spacing() {
        let mut img = box_image(&[20, 20], &[3, 4], &[3, 5], 1);
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let settings = LabelShapeStatisticsSettings {
            compute_oriented_bounding_box: true,
            ..Default::default()
        };
        let s = &label_shape_statistics(&img, &settings).unwrap()[&1];
        let obb = s.oriented_bounding_box.as_ref().unwrap();
        // Principal axes are (row0 = y, row1 = -x), so the OBB extents come
        // out in that order: 5 * 1 along y, 3 * 2 along x.
        assert_slice_close(&obb.size, &[5.0, 6.0], 1e-9, "obb size");
        // 2^dim vertices, and the centroid is the mean of all of them.
        assert_eq!(obb.vertices.len(), 4);
        let mean_x: f64 = obb.vertices.iter().map(|v| v[0]).sum::<f64>() / 4.0;
        let mean_y: f64 = obb.vertices.iter().map(|v| v[1]).sum::<f64>() / 4.0;
        assert_close(mean_x, s.centroid[0], 1e-9, "obb center x");
        assert_close(mean_y, s.centroid[1], 1e-9, "obb center y");
    }

    #[test]
    fn oriented_bounding_box_3d_of_an_axis_aligned_box() {
        let img = box_image(&[12, 12, 14], &[1, 2, 3], &[3, 5, 7], 1);
        let settings = LabelShapeStatisticsSettings {
            compute_oriented_bounding_box: true,
            ..Default::default()
        };
        let s = &label_shape_statistics(&img, &settings).unwrap()[&1];
        let obb = s.oriented_bounding_box.as_ref().unwrap();
        assert_slice_close(&obb.size, &[3.0, 5.0, 7.0], 1e-9, "obb size");
        assert_slice_close(&obb.origin, &[0.5, 1.5, 2.5], 1e-9, "obb origin");
        assert_eq!(obb.vertices.len(), 8);
        assert_slice_close(&obb.vertices[0], &[0.5, 1.5, 2.5], 1e-9, "vertex 0");
        assert_slice_close(&obb.vertices[7], &[3.5, 6.5, 9.5], 1e-9, "vertex 7");
    }
}
