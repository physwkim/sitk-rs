//! ITK's distance-map filters: `SignedMaurerDistanceMapImageFilter`,
//! `DanielssonDistanceMapImageFilter`, `SignedDanielssonDistanceMapImageFilter`,
//! `IsoContourDistanceImageFilter`, `ApproximateSignedDistanceMapImageFilter`.
//!
//! Verified against ITK's `Modules/Filtering/DistanceMap/include/`:
//! `itkSignedMaurerDistanceMapImageFilter.hxx`,
//! `itkDanielssonDistanceMapImageFilter.hxx`,
//! `itkSignedDanielssonDistanceMapImageFilter.hxx`,
//! `itkReflectiveImageRegionConstIterator.hxx`,
//! `itkIsoContourDistanceImageFilter.hxx`,
//! `itkFastChamferDistanceImageFilter.hxx`,
//! `itkApproximateSignedDistanceMapImageFilter.hxx`.
//!
//! All of them always produce a `Float64` output image. SimpleITK's yaml fixes
//! the procedural output at `Float32`; we use `f64` throughout (including the
//! internal accumulation) so callers get true double precision instead of
//! ITK's `float` rounding. Two knock-on consequences for the iso-contour /
//! chamfer pair below, both deliberate:
//!
//! - `IsoContourDistanceImageFilter::ComputeValue` guards on
//!   `NumericTraits<PixelRealType>::min()`, the smallest positive *normal*
//!   value of its real type. We use [`f64::MIN_POSITIVE`], so the guard trips
//!   at `2.2e-308` rather than ITK's `1.18e-38` (`float`) for integer inputs.
//! - `FastChamferDistanceImageFilter`'s weights are `float` literals and its
//!   running distance is a `float`; ours are `f64`, so the accumulated chamfer
//!   distance is the `f64` value of the same decimal weights rather than
//!   ITK's `float`-rounded one.
//!
//! ## Maurer
//!
//! `SignedMaurerDistanceMapImageFilter::GenerateData` first builds a 0/∞ seed
//! image: 0 at object pixels that are fully-connected-adjacent to a
//! background pixel (the object's boundary, as produced by
//! `BinaryThresholdImageFilter` + `BinaryContourImageFilter` in the original),
//! `NumericTraits::max()` (here `f64::INFINITY`) everywhere else. It then runs
//! one 1-D exact-EDT pass per dimension (`Voronoi`/`Remove` in the `.hxx`):
//! each pass treats the existing (possibly already-signed) value at a pixel as
//! an unsigned partial squared distance via `Math::Absolute`, combines it with
//! the new dimension using the lower envelope of parabolas, and re-applies the
//! sign from the *original* input classification. `Remove` is the classic
//! parabola-domination test from Maurer et al. 2003.
//!
//! ## Danielsson
//!
//! `DanielssonDistanceMapImageFilter` propagates an integer offset-to-nearest-
//! object vector per pixel using a "reflective" (forward-then-backward, per
//! axis) traversal (`ReflectiveImageRegionConstIterator`): every background
//! pixel compares its current offset against its neighbor's offset plus the
//! step taken to reach that neighbor, keeping whichever has smaller norm
//! (`UpdateLocalDistance`). We replicate the exact visitation order (axis 0
//! fastest, each axis's forward sweep `1..size-1` immediately followed by its
//! backward sweep `size-2..=0`) since the update is Gauss-Seidel (order-
//! dependent), not Jacobi.
//!
//! `SignedDanielssonDistanceMapImageFilter` runs the unsigned filter twice —
//! once on the input, once on the input's complement dilated by one
//! fully-connected pixel (`BinaryBallStructuringElement` at radius 1 is, by
//! ITK's own non-parametric ellipsoid test, the full 3×3×...×3 cube, i.e.
//! plain full connectivity) — and subtracts.
//!
//! `DanielssonDistanceMapImageFilter::InputIsBinary` is intentionally not
//! exposed: it only affects the `VoronoiMap` output (verified in
//! `PrepareData`/`ComputeVoronoiMap`), which this port does not expose (see
//! `danielsson_distance_map` below), so passing it through would be a dead
//! parameter.
//!
//! ## Iso-contour distance
//!
//! `IsoContourDistanceImageFilter` seeds the output with `±FarValue` by which
//! side of `LevelSetValue` each input pixel falls on (exactly *on* the level
//! set gives `0`), then, for every pixel that has a neighbor on the other
//! side along some axis `n`, replaces both endpoints of that crossing edge
//! with a first-order estimate of their signed distance to the interpolated
//! iso-contour. The estimate divides the endpoint's level-set offset by the
//! central-difference gradient magnitude averaged over the two endpoints (the
//! `alpha0 = alpha1 = 0.5` interpolation in `ComputeValue`). A pixel touched
//! from several axes keeps the smallest-magnitude estimate.
//!
//! The neighborhood reads use ITK's default `ZeroFluxNeumannBoundaryCondition`
//! (out-of-bounds coordinates clamp to the nearest in-bounds pixel), which is
//! what `ConstNeighborhoodIterator` applies. That clamping is also why the
//! `+1` neighbor is always in bounds inside the crossing branch: at the far
//! edge of axis `n` the clamped neighbor *is* the center, so `val1 == val0`
//! and the two can never differ in sign.
//!
//! ITK's narrow-band path (`ThreadedGenerateDataBand`) is not ported —
//! `m_NarrowBanding` defaults to `false` and SimpleITK never sets a narrow
//! band. The `.hxx` runs the two passes as separate `ClassicMultiThread`
//! calls, i.e. with a full barrier between initialization and computation, and
//! the second pass is order-dependent under its mutex; we run it single-
//! threaded in raster order, which is the deterministic one-work-unit case.
//!
//! ## Approximate signed distance map
//!
//! `ApproximateSignedDistanceMapImageFilter` is a mini-pipeline of
//! `IsoContourDistanceImageFilter` followed by
//! `FastChamferDistanceImageFilter` — *not* fast marching. It derives
//! `maximumDistance = (SizeValueType) sqrt(Σ size[i]²)` (an integer truncation),
//! hands the iso-contour filter `FarValue = maximumDistance + 1` and
//! `LevelSetValue = (InsideValue + OutsideValue) / 2`, lets the chamfer filter
//! sweep signed distances outward with `MaximumDistance = maximumDistance`,
//! and finally negates the whole image when `InsideValue > OutsideValue` so
//! that "inside an object" always reads negative.
//!
//! `FastChamferDistanceImageFilter::GenerateDataND` is a two-sweep Gauss-Seidel
//! chamfer transform over the in-place output: a forward raster scan updating
//! only the neighbors *after* the center in the 3^dim neighborhood, then a
//! backward scan updating only those *before* it. Both sweeps skip any center
//! whose current value has already saturated (`|value| >= MaximumDistance`),
//! and out-of-bounds neighbor writes are silently dropped
//! (`SetPixel(i, v, status)` "quietly ignores out-of-bounds attempts").
//!
//! Two upstream quirks reproduced verbatim:
//!
//! - The backward scan is `for (it.GoToEnd(), --it; !it.IsAtBegin(); --it)`,
//!   so **pixel 0 is never visited as a center**. This turns out to change no
//!   output: an offset with neighborhood index below the center has, by the
//!   base-3 digit ordering, at least one `-1` component, so every neighbor the
//!   backward scan would relax from the all-zero coordinate is out of bounds.
//!   Pixel 0 is still *written* — as a neighbor of pixel 1 and of its own
//!   diagonal successors.
//! - The chamfer sweep also relaxes the iso-contour's own sub-pixel values: a
//!   center at `-0.5` writes `-0.5 + 0.92644 = 0.42644` over an adjacent
//!   `+0.5`, because `0.42644 < 0.5`. The seeded crossing values are not
//!   pinned.

use crate::error::FilterError;
use crate::{Result, quantize_to_pixel_type};
use sitk_core::Image;

// ---- shared N-D index geometry ---------------------------------------------

/// First-index-fastest strides for a size vector (same convention as
/// `shrink.rs`/`smoothing.rs`/`recursive_gaussian.rs`).
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// Resolved N-D geometry: dimensionality, size, strides, and an *effective*
/// per-axis spacing (real spacing if `use_image_spacing`, else all `1.0`) so
/// callers never need to branch on the spacing flag themselves.
struct Geometry {
    dim: usize,
    size: Vec<usize>,
    strides: Vec<usize>,
    spacing: Vec<f64>,
}

impl Geometry {
    fn new(img: &Image, use_image_spacing: bool) -> Self {
        let size = img.size().to_vec();
        let dim = size.len();
        let strides = strides(&size);
        let spacing = if use_image_spacing {
            img.spacing().to_vec()
        } else {
            vec![1.0; dim]
        };
        Geometry {
            dim,
            size,
            strides,
            spacing,
        }
    }

    fn n(&self) -> usize {
        self.size.iter().product()
    }

    fn coords_of(&self, p: usize) -> Vec<i64> {
        (0..self.dim)
            .map(|d| ((p / self.strides[d]) % self.size[d]) as i64)
            .collect()
    }

    fn flat_index(&self, coord: &[i64]) -> usize {
        coord
            .iter()
            .zip(&self.strides)
            .map(|(&c, &s)| c as usize * s)
            .sum()
    }

    /// The flat index of `coord + offset`, or `None` if it falls outside the
    /// image (out-of-bounds neighbors are simply skipped, never treated as a
    /// virtual foreground/background value — matching `ScanlineFilterCommon`'s
    /// bounds-checked neighbor lookup and `BinaryDilateImageFilter`'s
    /// constant-zero boundary condition).
    fn neighbor_index(&self, coord: &[i64], offset: &[i64]) -> Option<usize> {
        let mut idx = 0usize;
        for d in 0..self.dim {
            let c = coord[d] + offset[d];
            if c < 0 || c as usize >= self.size[d] {
                return None;
            }
            idx += c as usize * self.strides[d];
        }
        Some(idx)
    }

    /// The flat index of `coord + offset` with each component clamped into the
    /// image — ITK's `ZeroFluxNeumannBoundaryCondition`, the default boundary
    /// condition of `ConstNeighborhoodIterator`.
    fn clamped_index(&self, coord: &[i64], offset: &[i64]) -> usize {
        let mut idx = 0usize;
        for d in 0..self.dim {
            let c = (coord[d] + offset[d]).clamp(0, self.size[d] as i64 - 1);
            idx += c as usize * self.strides[d];
        }
        idx
    }
}

/// All `3^dim - 1` full-connectivity neighbor offsets (`{-1,0,1}^dim` minus
/// the zero vector) — the connectivity `BinaryContourImageFilter` uses with
/// `SetFullyConnected(true)`, and what a radius-1 `BinaryBallStructuringElement`
/// turns out to be (see module docs).
fn full_connectivity_offsets(dim: usize) -> Vec<Vec<i64>> {
    let mut offsets = vec![Vec::new()];
    for _ in 0..dim {
        let mut next = Vec::with_capacity(offsets.len() * 3);
        for prefix in &offsets {
            for delta in [-1i64, 0, 1] {
                let mut v = prefix.clone();
                v.push(delta);
                next.push(v);
            }
        }
        offsets = next;
    }
    offsets.retain(|v| v.iter().any(|&x| x != 0));
    offsets
}

// ---- Signed Maurer distance map --------------------------------------------

/// `Ot.Set(InsideIsPositive ? mag : -mag)` for object pixels, the mirror image
/// for background — matching the sign logic repeated in `Voronoi` (every
/// pass) and in the final `sqrt` step of `ThreadedGenerateData`.
#[inline]
fn signed_value(magnitude: f64, is_object: bool, inside_is_positive: bool) -> f64 {
    if is_object == inside_is_positive {
        magnitude
    } else {
        -magnitude
    }
}

/// The parabola-domination test from `SignedMaurerDistanceMapImageFilter::Remove`.
fn maurer_remove(d1: f64, d2: f64, df: f64, x1: f64, x2: f64, xf: f64) -> bool {
    let a = x2 - x1;
    let b = xf - x2;
    let c = xf - x1;
    c * d2.abs() - b * d1.abs() - a * df.abs() - a * b * c > 0.0
}

/// One dimension's worth of `SignedMaurerDistanceMapImageFilter::Voronoi`,
/// applied to every line along axis `d`. `buf` holds the running signed
/// squared distance (or `f64::INFINITY` for not-yet-reached pixels).
fn maurer_voronoi_pass(
    buf: &mut [f64],
    is_object: &[bool],
    geo: &Geometry,
    d: usize,
    inside_is_positive: bool,
) {
    let nd = geo.size[d];
    let stride = geo.strides[d];
    let spacing_d = geo.spacing[d];
    let mut g = vec![0.0f64; nd];
    let mut h = vec![0.0f64; nd];

    for p0 in 0..buf.len() {
        if (p0 / stride) % nd != 0 {
            continue;
        }

        // Build the lower envelope of parabolas from the finite (non-sentinel)
        // samples along this line.
        let mut l: isize = -1;
        for i in 0..nd {
            let p = p0 + i * stride;
            let di = buf[p];
            if di == f64::INFINITY {
                continue;
            }
            let iw = i as f64 * spacing_d;
            if l >= 1 {
                while l >= 1
                    && maurer_remove(
                        g[(l - 1) as usize],
                        g[l as usize],
                        di,
                        h[(l - 1) as usize],
                        h[l as usize],
                        iw,
                    )
                {
                    l -= 1;
                }
            }
            l += 1;
            g[l as usize] = di;
            h[l as usize] = iw;
        }

        if l == -1 {
            continue; // no source on this line yet; leave as sentinel.
        }
        let ns = l;

        // Walk the envelope, writing the combined signed squared distance.
        let mut lw: isize = 0;
        for i in 0..nd {
            let iw = i as f64 * spacing_d;
            let mut d1 = g[lw as usize].abs() + (h[lw as usize] - iw).powi(2);
            while lw < ns {
                let d2 = g[(lw + 1) as usize].abs() + (h[(lw + 1) as usize] - iw).powi(2);
                if d1 <= d2 {
                    break;
                }
                lw += 1;
                d1 = d2;
            }
            let p = p0 + i * stride;
            buf[p] = signed_value(d1, is_object[p], inside_is_positive);
        }
    }
}

/// `SignedMaurerDistanceMapImageFilter`: exact Euclidean (or squared
/// Euclidean) signed distance transform, N-dimensional.
///
/// Parameters and defaults follow
/// `Code/BasicFilters/yaml/SignedMaurerDistanceMapImageFilter.yaml`:
/// `inside_is_positive` (default `false`), `squared_distance` (default
/// `true`), `use_image_spacing` (default `false`), `background_value`
/// (default `0.0`).
pub fn signed_maurer_distance_map(
    img: &Image,
    inside_is_positive: bool,
    squared_distance: bool,
    use_image_spacing: bool,
    background_value: f64,
) -> Result<Image> {
    let geo = Geometry::new(img, use_image_spacing);
    let n = geo.n();
    let input_vals = img.to_f64_vec()?;
    let is_object: Vec<bool> = input_vals.iter().map(|&v| v != background_value).collect();
    let offsets = full_connectivity_offsets(geo.dim);

    // Seed: 0 at object pixels adjacent (full connectivity) to a background
    // pixel, +inf elsewhere. Mirrors BinaryThresholdImageFilter (background ->
    // max, object -> 0) followed by BinaryContourImageFilter(foreground=0,
    // background=max, fully connected).
    let mut buf = vec![f64::INFINITY; n];
    for (p, is_boundary) in buf.iter_mut().enumerate() {
        if !is_object[p] {
            continue;
        }
        let coord = geo.coords_of(p);
        let on_boundary = offsets.iter().any(|off| {
            geo.neighbor_index(&coord, off)
                .is_some_and(|np| !is_object[np])
        });
        if on_boundary {
            *is_boundary = 0.0;
        }
    }

    for d in 0..geo.dim {
        maurer_voronoi_pass(&mut buf, &is_object, &geo, d, inside_is_positive);
    }

    let out_vals: Vec<f64> = if squared_distance {
        buf
    } else {
        buf.iter()
            .zip(&is_object)
            .map(|(&v, &obj)| signed_value(v.abs().sqrt(), obj, inside_is_positive))
            .collect()
    };

    let mut out = Image::from_vec(&geo.size, out_vals)?;
    out.copy_geometry_from(img);
    Ok(out)
}

// ---- Danielsson distance map ------------------------------------------------

/// `DanielssonDistanceMapImageFilter::UpdateLocalDistance`: compare `here`'s
/// current offset-to-nearest-object against the candidate obtained by routing
/// through `here + offset*axis` (i.e. `there`'s offset plus the step just
/// taken), keeping whichever has the smaller (spacing-weighted) norm.
fn update_local_distance(
    comp: &mut [Vec<i64>],
    coord: &[i64],
    geo: &Geometry,
    axis: usize,
    offset: i64,
) {
    let here = geo.flat_index(coord);
    let mut there_coord = coord.to_vec();
    there_coord[axis] += offset;
    let there = geo.flat_index(&there_coord);

    let mut norm1 = 0.0f64;
    let mut norm2 = 0.0f64;
    for (d, (comp_d, &spacing_d)) in comp.iter().zip(&geo.spacing).enumerate() {
        let extra = if d == axis { offset } else { 0 };
        let v1 = comp_d[here] as f64 * spacing_d;
        let v2 = (comp_d[there] + extra) as f64 * spacing_d;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }

    if norm1 > norm2 {
        for (d, comp_d) in comp.iter_mut().enumerate() {
            let extra = if d == axis { offset } else { 0 };
            comp_d[here] = comp_d[there] + extra;
        }
    }
}

/// Propagate the offset-to-nearest-object-pixel vector field for `is_object`,
/// replicating `ReflectiveImageRegionConstIterator`'s exact traversal: for
/// each axis with `size > 1`, a forward sweep over indices `1..size-1`
/// immediately followed by a backward sweep over `size-2..=0`, nested with
/// axis 0 fastest-varying (matching the `for (in = 0; ...)` order in
/// `operator++`). At every visited position, *every* active axis is updated
/// using its current sweep direction (not just the axis that just moved) —
/// see `GenerateData`'s per-pixel loop over all dimensions.
fn danielsson_propagate(is_object: &[bool], geo: &Geometry) -> Vec<Vec<i64>> {
    let dim = geo.dim;
    let n = is_object.len();
    let max_len = *geo.size.iter().max().unwrap_or(&0) as i64;
    let sentinel = 2 * max_len;

    let mut comp = vec![vec![0i64; n]; dim];
    for (p, &obj) in is_object.iter().enumerate() {
        if !obj {
            for axis in comp.iter_mut() {
                axis[p] = sentinel;
            }
        }
    }

    let active_dims: Vec<usize> = (0..dim).filter(|&d| geo.size[d] > 1).collect();
    if active_dims.is_empty() {
        return comp;
    }

    // Per-axis (position, is_reflected) sequence: forward 1..size-1, then
    // backward size-2..=0.
    let sequences: Vec<Vec<(i64, bool)>> = active_dims
        .iter()
        .map(|&d| {
            let sz = geo.size[d] as i64;
            let mut seq = Vec::with_capacity(2 * (sz as usize - 1));
            seq.extend((1..sz).map(|i| (i, false)));
            seq.extend((0..sz - 1).rev().map(|i| (i, true)));
            seq
        })
        .collect();
    let seq_lens: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
    let total: usize = seq_lens.iter().product();

    let mut coord = vec![0i64; dim];
    let mut reflected = vec![false; active_dims.len()];

    for counter in 0..total {
        let mut c = counter;
        for (idx, &d) in active_dims.iter().enumerate() {
            let len = seq_lens[idx];
            let (pos, refl) = sequences[idx][c % len];
            c /= len;
            coord[d] = pos;
            reflected[idx] = refl;
        }

        let here = geo.flat_index(&coord);
        if is_object[here] {
            continue;
        }
        for (idx, &axis) in active_dims.iter().enumerate() {
            let offset = if reflected[idx] { 1 } else { -1 };
            update_local_distance(&mut comp, &coord, geo, axis, offset);
        }
    }

    comp
}

/// `ComputeVoronoiMap`'s distance computation: turn a final offset vector
/// field into per-pixel (squared) distance.
fn danielsson_finalize(comp: &[Vec<i64>], geo: &Geometry, squared: bool) -> Vec<f64> {
    let n = geo.n();
    let mut out = vec![0.0f64; n];
    for (p, out_p) in out.iter_mut().enumerate() {
        let mut d = 0.0f64;
        for (comp_axis, &spacing_axis) in comp.iter().zip(&geo.spacing) {
            let v = comp_axis[p] as f64 * spacing_axis;
            d += v * v;
        }
        *out_p = if squared { d } else { d.sqrt() };
    }
    out
}

/// Dilate `mask` by one fully-connected pixel: `true` where `mask` is `true`
/// or any full-connectivity neighbor (within bounds) is `true`. This is what
/// `BinaryDilateImageFilter` with a radius-1 `BinaryBallStructuringElement`
/// computes (see module docs for why radius 1 is full connectivity, not a
/// face-connected cross).
fn dilate_full_connectivity(mask: &[bool], geo: &Geometry) -> Vec<bool> {
    let offsets = full_connectivity_offsets(geo.dim);
    let mut out = mask.to_vec();
    for (p, out_p) in out.iter_mut().enumerate() {
        if mask[p] {
            continue;
        }
        let coord = geo.coords_of(p);
        *out_p = offsets
            .iter()
            .any(|off| geo.neighbor_index(&coord, off).is_some_and(|np| mask[np]));
    }
    out
}

/// `DanielssonDistanceMapImageFilter`: unsigned distance-to-nearest-nonzero-
/// pixel transform, N-dimensional, approximate-but-exact-for-unobstructed-
/// sources (see module docs).
///
/// Only the primary distance map is exposed. SimpleITK's `VoronoiMap` and
/// `VectorDistanceMap` are getter-only "measurements" on the object-oriented
/// filter (not procedural-function parameters/returns — confirmed against
/// `ExpandTemplateGenerator/templates/ProceduralAPI.h.jinja`, which only ever
/// returns `Execute()`'s single output), and this crate is procedural-only for
/// now; `InputIsBinary` is dropped for the reason in the module docs.
///
/// Parameters follow
/// `Code/BasicFilters/yaml/DanielssonDistanceMapImageFilter.yaml`:
/// `squared_distance` (default `false`), `use_image_spacing` (default
/// `false`).
pub fn danielsson_distance_map(
    img: &Image,
    squared_distance: bool,
    use_image_spacing: bool,
) -> Result<Image> {
    let geo = Geometry::new(img, use_image_spacing);
    let vals = img.to_f64_vec()?;
    let is_object: Vec<bool> = vals.iter().map(|&v| v != 0.0).collect();

    let comp = danielsson_propagate(&is_object, &geo);
    let out_vals = danielsson_finalize(&comp, &geo, squared_distance);

    let mut out = Image::from_vec(&geo.size, out_vals)?;
    out.copy_geometry_from(img);
    Ok(out)
}

/// `SignedDanielssonDistanceMapImageFilter`: runs the unsigned Danielsson
/// transform twice — once on the input, once on the input's complement
/// dilated by one fully-connected pixel — and subtracts, per
/// `SignedDanielssonDistanceMapImageFilter::GenerateData`.
///
/// Same output-exposure decision as [`danielsson_distance_map`] (distance map
/// only). Parameters follow
/// `Code/BasicFilters/yaml/SignedDanielssonDistanceMapImageFilter.yaml`:
/// `inside_is_positive` (default `false`), `squared_distance` (default
/// `false`), `use_image_spacing` (default `false`).
pub fn signed_danielsson_distance_map(
    img: &Image,
    inside_is_positive: bool,
    squared_distance: bool,
    use_image_spacing: bool,
) -> Result<Image> {
    let geo = Geometry::new(img, use_image_spacing);
    let vals = img.to_f64_vec()?;
    let is_object: Vec<bool> = vals.iter().map(|&v| v != 0.0).collect();

    let comp1 = danielsson_propagate(&is_object, &geo);
    let d1 = danielsson_finalize(&comp1, &geo, squared_distance);

    let inverted: Vec<bool> = is_object.iter().map(|&o| !o).collect();
    let dilated = dilate_full_connectivity(&inverted, &geo);
    let comp2 = danielsson_propagate(&dilated, &geo);
    let d2 = danielsson_finalize(&comp2, &geo, squared_distance);

    let out_vals: Vec<f64> = d1
        .iter()
        .zip(&d2)
        .map(|(&a, &b)| if inside_is_positive { b - a } else { a - b })
        .collect();

    let mut out = Image::from_vec(&geo.size, out_vals)?;
    out.copy_geometry_from(img);
    Ok(out)
}

// ---- Iso-contour distance ---------------------------------------------------

/// `IsoContourDistanceImageFilter::ComputeValue` for the pixel at `coord`
/// (flat index `center`). `input` is read with zero-flux-Neumann clamping;
/// `out` is read *and* written in place, so the raster sweep is Gauss-Seidel
/// exactly as the `.hxx` (its per-pixel `m_Mutex` serializes the same
/// read-compare-write).
fn iso_contour_compute_value(
    input: &[f64],
    out: &mut [f64],
    geo: &Geometry,
    coord: &[i64],
    center: usize,
    level_set_value: f64,
) -> Result<()> {
    let dim = geo.dim;
    let val0 = input[center] - level_set_value;
    let sign = val0 > 0.0;

    // `off` is the running neighborhood offset from `coord`; every borrow
    // resets it to all-zero before returning.
    let mut off = vec![0i64; dim];

    // grad0[ng] = GetNext(ng, 1) - GetPrevious(ng, 1)
    let mut grad0 = vec![0.0f64; dim];
    for (ng, g) in grad0.iter_mut().enumerate() {
        off[ng] = 1;
        let plus = input[geo.clamped_index(coord, &off)];
        off[ng] = -1;
        let minus = input[geo.clamped_index(coord, &off)];
        off[ng] = 0;
        *g = plus - minus;
    }

    for n in 0..dim {
        // `GetPixel(center + stride[n])`. When that neighbor is out of bounds
        // the boundary condition clamps it back onto the center, so `val1`
        // would equal `val0` and the sign test below could never fire.
        off[n] = 1;
        let next = geo.neighbor_index(coord, &off);
        off[n] = 0;
        let Some(next) = next else { continue };

        let val1 = input[next] - level_set_value;
        if sign == (val1 > 0.0) {
            continue;
        }

        // grad1[ng] = in(c + e_n + e_ng) - in(c + e_n - e_ng)
        let mut grad1 = vec![0.0f64; dim];
        off[n] = 1;
        for (ng, g) in grad1.iter_mut().enumerate() {
            off[ng] += 1;
            let plus = input[geo.clamped_index(coord, &off)];
            off[ng] -= 2;
            let minus = input[geo.clamped_index(coord, &off)];
            off[ng] += 1;
            *g = plus - minus;
        }
        off[n] = 0;

        let diff = if sign { val0 - val1 } else { val1 - val0 };
        if diff < f64::MIN_POSITIVE {
            return Err(FilterError::IsoContourDegenerateDifference(diff));
        }

        // alpha0 == alpha1 == 0.5: average the two endpoints' central
        // differences, then convert to a physical-space gradient.
        let mut norm = 0.0f64;
        let mut grad = vec![0.0f64; dim];
        for ng in 0..dim {
            grad[ng] = (grad0[ng] * 0.5 + grad1[ng] * 0.5) / (2.0 * geo.spacing[ng]);
            norm += grad[ng] * grad[ng];
        }
        norm = norm.sqrt();

        if norm <= f64::MIN_POSITIVE {
            return Err(FilterError::IsoContourZeroGradient);
        }

        let val = grad[n].abs() * geo.spacing[n] / norm / diff;
        let val_new0 = val0 * val;
        let val_new1 = val1 * val;
        if val_new0.abs() < out[center].abs() {
            out[center] = val_new0;
        }
        if val_new1.abs() < out[next].abs() {
            out[next] = val_new1;
        }
    }
    Ok(())
}

/// The two passes of `IsoContourDistanceImageFilter::GenerateData`, on raw
/// `f64` buffers so `approximate_signed_distance_map` can feed the result
/// straight into the chamfer sweep.
fn iso_contour_values(
    input: &[f64],
    geo: &Geometry,
    level_set_value: f64,
    far_value: f64,
) -> Result<Vec<f64>> {
    // Pass 1 (`ThreadedGenerateData`): ±FarValue by side, 0 exactly on the
    // level set.
    let mut out: Vec<f64> = input
        .iter()
        .map(|&v| {
            if v > level_set_value {
                far_value
            } else if v < level_set_value {
                -far_value
            } else {
                0.0
            }
        })
        .collect();

    // Pass 2 (`ThreadedGenerateDataFull`), raster order.
    for p in 0..out.len() {
        let coord = geo.coords_of(p);
        iso_contour_compute_value(input, &mut out, geo, &coord, p, level_set_value)?;
    }
    Ok(out)
}

/// `IsoContourDistanceImageFilter`: signed distance from each grid point near
/// the `level_set_value` iso-contour to that interpolated contour, and
/// `±far_value` everywhere else. See the module docs for the scheme and for
/// the boundary condition.
///
/// Parameters and defaults follow
/// `Code/BasicFilters/yaml/IsoContourDistanceImageFilter.yaml`:
/// `level_set_value` (default `0.0`), `far_value` (default `10.0`).
///
/// Uses the input's spacing unconditionally — the ITK filter reads
/// `GetInput()->GetSpacing()` in `BeforeThreadedGenerateData` and exposes no
/// `UseImageSpacing` switch.
///
/// # Errors
///
/// [`FilterError::IsoContourDegenerateDifference`] and
/// [`FilterError::IsoContourZeroGradient`] mirror the two
/// `itkGenericExceptionMacro`/`itkExceptionStringMacro` throws in
/// `ComputeValue`: a level-set crossing whose endpoint values are separated by
/// less than the real type's smallest positive normal, and a crossing whose
/// interpolated gradient has no magnitude ("Gradient norm is lower than pixel
/// precision").
pub fn iso_contour_distance(img: &Image, level_set_value: f64, far_value: f64) -> Result<Image> {
    let geo = Geometry::new(img, true);
    let input = img.to_f64_vec()?;
    let out_vals = iso_contour_values(&input, &geo, level_set_value, far_value)?;

    let mut out = Image::from_vec(&geo.size, out_vals)?;
    out.copy_geometry_from(img);
    Ok(out)
}

// ---- Approximate signed distance map ----------------------------------------

/// The state `FastChamferDistanceImageFilter` precomputes before its two
/// sweeps: the constructor's per-neighbor-type weights, the radius-1
/// neighborhood's offsets, and `neighbor_type` (the `-1 + Σ_n (offset[n] !=
/// 0)` table `GenerateDataND` builds for each half of the neighborhood).
struct ChamferKernel {
    /// `m_Weights`: hard-coded for 1/2/3-D, `sqrt(i)` for `i in 1..=dim`
    /// otherwise (the `default:` arm, which also warns upstream).
    weights: Vec<f64>,
    /// The `3^dim` offsets of a radius-1 `itk::Neighborhood`, in its own index
    /// order (first dimension fastest, each component running `-1, 0, 1`).
    /// Index `3^dim / 2` is the center.
    offsets: Vec<Vec<i64>>,
    /// One less than the offset's number of non-zero components — i.e. which
    /// chamfer weight that neighbor takes. Never read at the center, whose
    /// count would underflow.
    neighbor_type: Vec<usize>,
    /// `m_MaximumDistance`.
    maximum_distance: f64,
}

impl ChamferKernel {
    fn new(dim: usize, maximum_distance: f64) -> Self {
        let weights = match dim {
            1 => vec![0.92644],
            2 => vec![0.92644, 1.34065],
            3 => vec![0.92644, 1.34065, 1.65849],
            _ => (1..=dim).map(|i| (i as f64).sqrt()).collect(),
        };
        let count = 3usize.pow(dim as u32);
        let offsets: Vec<Vec<i64>> = (0..count)
            .map(|i| {
                (0..dim)
                    .map(|n| (i / 3usize.pow(n as u32)) as i64 % 3 - 1)
                    .collect()
            })
            .collect();
        let neighbor_type = offsets
            .iter()
            .map(|o| o.iter().filter(|&&c| c != 0).count().saturating_sub(1))
            .collect();
        ChamferKernel {
            weights,
            offsets,
            neighbor_type,
            maximum_distance,
        }
    }

    fn center(&self) -> usize {
        self.offsets.len() / 2
    }

    /// One sweep of `GenerateDataND`: relax every neighbor in `neighbors` from
    /// each center in `centers`, in place.
    fn sweep(
        &self,
        vals: &mut [f64],
        geo: &Geometry,
        neighbors: &[usize],
        centers: impl Iterator<Item = usize>,
    ) {
        for p in centers {
            let center_value = vals[p];
            if center_value >= self.maximum_distance || center_value <= -self.maximum_distance {
                continue;
            }
            let coord = geo.coords_of(p);

            // Out-of-bounds neighbors are skipped: upstream reads them through
            // the boundary condition but writes them with `SetPixel(i, v,
            // status)`, which "quietly ignores out-of-bounds attempts", so the
            // comparison's outcome is discarded either way.
            if center_value > -self.weights[0] {
                for &i in neighbors {
                    let val = center_value + self.weights[self.neighbor_type[i]];
                    if let Some(q) = geo.neighbor_index(&coord, &self.offsets[i])
                        && val < vals[q]
                    {
                        vals[q] = val;
                    }
                }
            }
            if center_value < self.weights[0] {
                for &i in neighbors {
                    let val = center_value - self.weights[self.neighbor_type[i]];
                    if let Some(q) = geo.neighbor_index(&coord, &self.offsets[i])
                        && val > vals[q]
                    {
                        vals[q] = val;
                    }
                }
            }
        }
    }
}

/// `FastChamferDistanceImageFilter::GenerateDataND`, in place on `vals` (which
/// `GenerateData` has already initialized by copying the input).
fn fast_chamfer_distance(vals: &mut [f64], geo: &Geometry, maximum_distance: f64) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let kernel = ChamferKernel::new(geo.dim, maximum_distance);
    let center = kernel.center();
    let forward: Vec<usize> = (center + 1..kernel.offsets.len()).collect();
    let backward: Vec<usize> = (0..center).collect();

    kernel.sweep(vals, geo, &forward, 0..n);
    // `for (it.GoToEnd(), --it; !it.IsAtBegin(); --it)`: the backward scan
    // stops *before* pixel 0, so index 0 is never a center. Reproduced.
    kernel.sweep(vals, geo, &backward, (1..n).rev());
}

/// `ApproximateSignedDistanceMapImageFilter`: signed chamfer distance to the
/// boundary of the objects in a binary image, negative inside them.
///
/// Parameters and defaults follow
/// `Code/BasicFilters/yaml/ApproximateSignedDistanceMapImageFilter.yaml`:
/// `inside_value` (default `1`), `outside_value` (default `0`). Both are
/// declared `pixeltype: Input` there, so they are cast to the input's pixel
/// type before the ITK filter sees them; we do the same via
/// [`quantize_to_pixel_type`]. That cast is load-bearing — it is the cast
/// values that decide `InsideValue > OutsideValue`, hence the final sign flip.
///
/// SimpleITK only instantiates this wrapper for `IntegerPixelIDTypeList`; the
/// ITK filter itself is generic over the input pixel type, and so is this
/// port.
///
/// # Errors
///
/// Propagates the two [`iso_contour_distance`] errors from the internal
/// iso-contour stage.
pub fn approximate_signed_distance_map(
    img: &Image,
    inside_value: f64,
    outside_value: f64,
) -> Result<Image> {
    let geo = Geometry::new(img, true);
    let inside = quantize_to_pixel_type(img.pixel_id(), inside_value);
    let outside = quantize_to_pixel_type(img.pixel_id(), outside_value);

    // maximumDistance = (OutputSizeValueType) sqrt(Σ size[i]²) — an integer
    // truncation of the image's corner-to-corner diagonal.
    let squared_diagonal: u64 = geo.size.iter().map(|&s| (s as u64) * (s as u64)).sum();
    let maximum_distance = (squared_diagonal as f64).sqrt() as u64;

    let level_set_value = (inside + outside) / 2.0;
    let far_value = (maximum_distance + 1) as f64;

    let input = img.to_f64_vec()?;
    let mut vals = iso_contour_values(&input, &geo, level_set_value, far_value)?;
    fast_chamfer_distance(&mut vals, &geo, maximum_distance as f64);

    // The mini-pipeline assumes "inside" is the side *below* the iso-contour.
    // When it is not, every distance came out with the opposite sign.
    if inside > outside {
        for v in &mut vals {
            *v = -*v;
        }
    }

    let mut out = Image::from_vec(&geo.size, vals)?;
    out.copy_geometry_from(img);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_voxel(size: &[usize], on: &[usize]) -> Image {
        let n: usize = size.iter().product();
        let strides = strides(size);
        let mut data = vec![0u8; n];
        let idx: usize = on.iter().zip(&strides).map(|(&c, &s)| c * s).sum();
        data[idx] = 1;
        Image::from_vec(size, data).unwrap()
    }

    fn analytic_sq_dist(coord: &[usize], on: &[usize], spacing: &[f64]) -> f64 {
        coord
            .iter()
            .zip(on)
            .zip(spacing)
            .map(|((&c, &o), &s)| {
                let d = (c as f64 - o as f64) * s;
                d * d
            })
            .sum()
    }

    #[test]
    fn maurer_single_voxel_2d_matches_analytic_squared_distance() {
        let size = [7usize, 7];
        let on = [3usize, 3];
        let img = single_voxel(&size, &on);
        let out = signed_maurer_distance_map(&img, false, true, false, 0.0).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();
        let strides = strides(&size);

        for y in 0..size[1] {
            for x in 0..size[0] {
                let p = x * strides[0] + y * strides[1];
                let expected = analytic_sq_dist(&[x, y], &on, &[1.0, 1.0]);
                let is_object = (x, y) == (on[0], on[1]);
                let signed_expected = if is_object { -expected } else { expected };
                assert!(
                    (vals[p] - signed_expected).abs() < 1e-12,
                    "pixel ({x},{y}): got {}, expected {}",
                    vals[p],
                    signed_expected
                );
            }
        }
    }

    #[test]
    fn maurer_single_voxel_3d_matches_analytic_squared_distance() {
        let size = [5usize, 5, 5];
        let on = [2usize, 2, 2];
        let img = single_voxel(&size, &on);
        let out = signed_maurer_distance_map(&img, false, true, false, 0.0).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();
        let strides = strides(&size);

        for z in 0..size[2] {
            for y in 0..size[1] {
                for x in 0..size[0] {
                    let p = x * strides[0] + y * strides[1] + z * strides[2];
                    let expected = analytic_sq_dist(&[x, y, z], &on, &[1.0, 1.0, 1.0]);
                    let is_object = (x, y, z) == (on[0], on[1], on[2]);
                    let signed_expected = if is_object { -expected } else { expected };
                    assert!(
                        (vals[p] - signed_expected).abs() < 1e-12,
                        "pixel ({x},{y},{z}): got {}, expected {}",
                        vals[p],
                        signed_expected
                    );
                }
            }
        }
    }

    #[test]
    fn maurer_anisotropic_spacing_matches_analytic_formula() {
        let size = [6usize, 6];
        let on = [2usize, 2];
        let mut img = single_voxel(&size, &on);
        img.set_spacing(&[1.0, 3.0]).unwrap();
        let out = signed_maurer_distance_map(&img, false, true, true, 0.0).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();
        let strides = strides(&size);

        for y in 0..size[1] {
            for x in 0..size[0] {
                let p = x * strides[0] + y * strides[1];
                let expected = analytic_sq_dist(&[x, y], &on, &[1.0, 3.0]);
                let is_object = (x, y) == (on[0], on[1]);
                let signed_expected = if is_object { -expected } else { expected };
                assert!(
                    (vals[p] - signed_expected).abs() < 1e-9,
                    "pixel ({x},{y}): got {}, expected {}",
                    vals[p],
                    signed_expected
                );
            }
        }

        // Sanity: spacing must actually matter (not silently ignored).
        let isotropic = signed_maurer_distance_map(&img, false, true, false, 0.0).unwrap();
        assert_ne!(
            isotropic.scalar_slice::<f64>().unwrap(),
            out.scalar_slice::<f64>().unwrap()
        );
    }

    #[test]
    fn maurer_inside_is_positive_flips_sign_only() {
        let size = [7usize, 7];
        let on = [3usize, 3];
        let img = single_voxel(&size, &on);
        let neg = signed_maurer_distance_map(&img, false, true, false, 0.0).unwrap();
        let pos = signed_maurer_distance_map(&img, true, true, false, 0.0).unwrap();
        let neg_vals = neg.scalar_slice::<f64>().unwrap();
        let pos_vals = pos.scalar_slice::<f64>().unwrap();

        for (&n, &p) in neg_vals.iter().zip(pos_vals) {
            assert!((n.abs() - p.abs()).abs() < 1e-12);
            if n != 0.0 {
                assert!(
                    (n + p).abs() < 1e-12,
                    "expected exact sign flip: {n} vs {p}"
                );
            }
        }
    }

    #[test]
    fn maurer_squared_false_is_sqrt_of_squared_true() {
        let size = [7usize, 7];
        let on = [3usize, 3];
        let img = single_voxel(&size, &on);
        let sq = signed_maurer_distance_map(&img, false, true, false, 0.0).unwrap();
        let lin = signed_maurer_distance_map(&img, false, false, false, 0.0).unwrap();
        let sq_vals = sq.scalar_slice::<f64>().unwrap();
        let lin_vals = lin.scalar_slice::<f64>().unwrap();

        for (&s, &l) in sq_vals.iter().zip(lin_vals) {
            let expected = if s >= 0.0 { s.sqrt() } else { -((-s).sqrt()) };
            assert!((l - expected).abs() < 1e-9, "got {l}, expected {expected}");
        }
    }

    fn filled_square(size: &[usize], lo: usize, hi: usize) -> Image {
        let n: usize = size.iter().product();
        let strides = strides(size);
        let mut data = vec![0u8; n];
        for y in lo..=hi {
            for x in lo..=hi {
                let p = x * strides[0] + y * strides[1];
                data[p] = 1;
            }
        }
        Image::from_vec(size, data).unwrap()
    }

    #[test]
    fn maurer_and_danielsson_agree_on_filled_square() {
        let size = [11usize, 11];
        let img = filled_square(&size, 4, 6); // 3x3 block, has a true interior pixel at (5,5)

        let maurer = signed_maurer_distance_map(&img, false, false, false, 0.0).unwrap();
        let danielsson = signed_danielsson_distance_map(&img, false, false, false).unwrap();

        let m = maurer.scalar_slice::<f64>().unwrap();
        let d = danielsson.scalar_slice::<f64>().unwrap();
        for (i, (&mv, &dv)) in m.iter().zip(d).enumerate() {
            assert!(
                (mv - dv).abs() < 1e-9,
                "pixel {i}: maurer={mv}, danielsson={dv}"
            );
        }
    }

    #[test]
    fn danielsson_unsigned_matches_maurer_abs_on_background() {
        let size = [7usize, 7];
        let on = [3usize, 3];
        let img = single_voxel(&size, &on);

        let maurer = signed_maurer_distance_map(&img, false, false, false, 0.0).unwrap();
        let danielsson = danielsson_distance_map(&img, false, false).unwrap();
        let m = maurer.scalar_slice::<f64>().unwrap();
        let d = danielsson.scalar_slice::<f64>().unwrap();

        for (&mv, &dv) in m.iter().zip(d) {
            assert!((mv.abs() - dv).abs() < 1e-9);
        }
    }

    #[test]
    fn danielsson_squared_false_is_sqrt_of_squared_true() {
        let size = [7usize, 7];
        let on = [3usize, 3];
        let img = single_voxel(&size, &on);
        let sq = danielsson_distance_map(&img, true, false).unwrap();
        let lin = danielsson_distance_map(&img, false, false).unwrap();
        let sq_vals = sq.scalar_slice::<f64>().unwrap();
        let lin_vals = lin.scalar_slice::<f64>().unwrap();

        for (&s, &l) in sq_vals.iter().zip(lin_vals) {
            assert!((l - s.sqrt()).abs() < 1e-9);
        }
    }

    // ---- iso-contour distance ----------------------------------------------

    /// `f64` image whose rows all equal `row`, repeated `rows` times.
    fn rows_f64(row: &[f64], rows: usize) -> Image {
        let data: Vec<f64> = row.iter().copied().cycle().take(row.len() * rows).collect();
        Image::from_vec(&[row.len(), rows], data).unwrap()
    }

    /// `u8` image whose rows all equal `row`, repeated `rows` times.
    fn rows_u8(row: &[u8], rows: usize) -> Image {
        let data: Vec<u8> = row.iter().copied().cycle().take(row.len() * rows).collect();
        Image::from_vec(&[row.len(), rows], data).unwrap()
    }

    fn assert_rows_eq(out: &Image, expected: &[f64], rows: usize) {
        let vals = out.scalar_slice::<f64>().unwrap();
        let w = expected.len();
        assert_eq!(vals.len(), w * rows);
        for y in 0..rows {
            for x in 0..w {
                assert!(
                    (vals[x + y * w] - expected[x]).abs() < 1e-12,
                    "pixel ({x},{y}): got {}, expected {}",
                    vals[x + y * w],
                    expected[x]
                );
            }
        }
    }

    /// A unit-slope ramp crossing the level set halfway between two pixels:
    /// both endpoints of the crossing edge sit exactly `0.5` from the
    /// interpolated contour, and no other pixel is touched.
    #[test]
    fn iso_contour_unit_ramp_puts_crossing_endpoints_at_half_a_pixel() {
        let img = rows_f64(&[0.0, 1.0, 2.0, 3.0, 4.0], 3);
        let out = iso_contour_distance(&img, 2.5, 10.0).unwrap();
        assert_rows_eq(&out, &[-10.0, -10.0, -0.5, 0.5, 10.0], 3);
    }

    /// The distance is measured in physical space: doubling the spacing along
    /// the crossing axis doubles the reported distance.
    #[test]
    fn iso_contour_scales_with_image_spacing() {
        let mut img = rows_f64(&[0.0, 1.0, 2.0, 3.0, 4.0], 3);
        img.set_spacing(&[2.0, 1.0]).unwrap();
        let out = iso_contour_distance(&img, 2.5, 10.0).unwrap();
        assert_rows_eq(&out, &[-10.0, -10.0, -1.0, 1.0, 10.0], 3);
    }

    /// Boundary: a crossing on the first and last column. The `-1` neighbor of
    /// column 0 is out of bounds, so the zero-flux-Neumann clamp must make the
    /// central difference one-sided rather than index out of the buffer; the
    /// `+1` neighbor of the last column is out of bounds, so its (clamped)
    /// value equals the center's and the crossing branch cannot fire there.
    #[test]
    fn iso_contour_crossing_at_both_image_edges() {
        let img = rows_f64(&[0.0, 1.0], 2);
        let out = iso_contour_distance(&img, 0.5, 10.0).unwrap();
        assert_rows_eq(&out, &[-0.5, 0.5], 2);
    }

    /// Boundary: `level_set_value` exactly equal to every pixel. No pixel is
    /// on either side, so the seed is `0` everywhere and no crossing exists.
    #[test]
    fn iso_contour_pixels_exactly_on_the_level_set_seed_to_zero() {
        let img = rows_f64(&[3.0, 3.0, 3.0], 2);
        let out = iso_contour_distance(&img, 3.0, 10.0).unwrap();
        assert_rows_eq(&out, &[0.0, 0.0, 0.0], 2);
    }

    /// Boundary: no crossing anywhere leaves the whole image at `+far_value`,
    /// and `far_value` is signed by side.
    #[test]
    fn iso_contour_without_a_crossing_is_signed_far_value() {
        let img = rows_f64(&[1.0, 2.0, 3.0], 2);
        assert_rows_eq(
            &iso_contour_distance(&img, 0.0, 7.5).unwrap(),
            &[7.5, 7.5, 7.5],
            2,
        );
        assert_rows_eq(
            &iso_contour_distance(&img, 9.0, 7.5).unwrap(),
            &[-7.5, -7.5, -7.5],
            2,
        );
    }

    /// `ComputeValue`'s "Gradient norm is lower than pixel precision" throw: a
    /// `1, 0, 1, 0` ramp crosses the level set at every edge, and at pixel 1
    /// both endpoints' central differences vanish.
    #[test]
    fn iso_contour_zero_gradient_at_a_crossing_is_an_error() {
        let img = rows_f64(&[1.0, 0.0, 1.0, 0.0], 3);
        assert_eq!(
            iso_contour_distance(&img, 0.5, 10.0),
            Err(FilterError::IsoContourZeroGradient)
        );
    }

    /// `ComputeValue`'s `diff < NumericTraits<PixelRealType>::min()` throw: a
    /// crossing whose two endpoints are a single subnormal apart.
    #[test]
    fn iso_contour_subnormal_crossing_difference_is_an_error() {
        let tiny = f64::from_bits(1); // 5e-324, the smallest positive subnormal
        let img = rows_f64(&[0.0, tiny, 0.0], 2);
        assert_eq!(
            iso_contour_distance(&img, 0.0, 10.0),
            Err(FilterError::IsoContourDegenerateDifference(tiny))
        );
    }

    // ---- approximate signed distance map -----------------------------------

    /// A vertical step edge between columns 2 and 3 of a 7x3 image.
    ///
    /// `maximumDistance = (SizeValueType) sqrt(7² + 3²) = 7`, so
    /// `FarValue = 8`. The iso-contour stage seeds
    /// `[-8, -8, -0.5, 0.5, 8, 8, 8]`, then the chamfer sweep propagates the
    /// axial weight `0.92644` outward from both crossing endpoints — including
    /// *over* the seeded `+0.5`, which the center at `-0.5` overwrites with
    /// `-0.5 + 0.92644 = 0.42644`. Finally `inside_value > outside_value`
    /// flips every sign, putting the object (columns 3..6) on the negative
    /// side.
    #[test]
    fn asdm_step_edge_matches_the_chamfer_sweep_pixel_for_pixel() {
        let img = rows_u8(&[0, 0, 0, 1, 1, 1, 1], 3);
        let out = approximate_signed_distance_map(&img, 1.0, 0.0).unwrap();
        let w = 0.92644f64;
        assert_rows_eq(
            &out,
            &[
                0.5 + 2.0 * w,
                0.5 + w,
                0.5,
                -(w - 0.5),
                -(2.0 * w - 0.5),
                -(3.0 * w - 0.5),
                -(4.0 * w - 0.5),
            ],
            3,
        );
    }

    /// Swapping `inside_value` and `outside_value` leaves `LevelSetValue =
    /// (inside + outside) / 2` untouched and only toggles the final negation,
    /// so the result is an exact (bit-for-bit) sign flip.
    #[test]
    fn asdm_swapping_inside_and_outside_is_an_exact_sign_flip() {
        let img = rows_u8(&[0, 0, 0, 1, 1, 1, 1], 3);
        let a = approximate_signed_distance_map(&img, 1.0, 0.0).unwrap();
        let b = approximate_signed_distance_map(&img, 0.0, 1.0).unwrap();
        for (&x, &y) in a
            .scalar_slice::<f64>()
            .unwrap()
            .iter()
            .zip(b.scalar_slice::<f64>().unwrap())
        {
            assert_eq!(x, -y);
        }
    }

    /// `inside_value == outside_value` puts the level set *on* the single
    /// value the image takes, so every pixel seeds to `0` and no chamfer
    /// center saturates. Also the boundary of the negation test: `1 > 1` is
    /// false, so no flip.
    #[test]
    fn asdm_equal_inside_and_outside_values_yield_an_all_zero_map() {
        let img = rows_u8(&[1, 1, 1, 1], 2);
        let out = approximate_signed_distance_map(&img, 1.0, 1.0).unwrap();
        assert_rows_eq(&out, &[0.0, 0.0, 0.0, 0.0], 2);
    }

    /// Boundary: an image with no object at all has no crossing, so every
    /// chamfer center is saturated at `|-FarValue| >= maximumDistance` and is
    /// skipped. The output is the constant `maximumDistance + 1`, negated.
    /// For 7x3 that is `(SizeValueType) sqrt(58) = 7`, hence `8`.
    #[test]
    fn asdm_uniform_image_saturates_at_the_truncated_diagonal_plus_one() {
        let background = rows_u8(&[0, 0, 0, 0, 0, 0, 0], 3);
        assert_rows_eq(
            &approximate_signed_distance_map(&background, 1.0, 0.0).unwrap(),
            &[8.0; 7],
            3,
        );

        let object = rows_u8(&[1, 1, 1, 1, 1, 1, 1], 3);
        assert_rows_eq(
            &approximate_signed_distance_map(&object, 1.0, 0.0).unwrap(),
            &[-8.0; 7],
            3,
        );
    }

    /// `inside_value`/`outside_value` are `pixeltype: Input` in the yaml, i.e.
    /// cast to the input pixel type before use. On a `u8` image `255.5` casts
    /// to `255` and `-3.0` casts to `0`, so `LevelSetValue` is `127.5` and the
    /// sign flip still fires (`255 > 0`) — a caller passing the exact `255`
    /// and `0` must get the identical image.
    #[test]
    fn asdm_inside_outside_are_cast_to_the_input_pixel_type() {
        let img = rows_u8(&[0, 0, 0, 255, 255, 255, 255], 3);
        let cast = approximate_signed_distance_map(&img, 255.5, -3.0).unwrap();
        let exact = approximate_signed_distance_map(&img, 255.0, 0.0).unwrap();
        assert_eq!(
            cast.scalar_slice::<f64>().unwrap(),
            exact.scalar_slice::<f64>().unwrap()
        );

        // And the level set really did land at 127.5: same geometry as the 0/1
        // step edge, so the same distances.
        let binary =
            approximate_signed_distance_map(&rows_u8(&[0, 0, 0, 1, 1, 1, 1], 3), 1.0, 0.0).unwrap();
        assert_eq!(
            exact.scalar_slice::<f64>().unwrap(),
            binary.scalar_slice::<f64>().unwrap()
        );
    }

    /// The chamfer sweep is 3-D-aware: the axial weight along the crossing
    /// axis is unchanged, and the diagonal weights (1.34065, 1.65849) never
    /// beat it on a flat step edge.
    #[test]
    fn asdm_step_edge_in_3d_keeps_the_axial_chamfer_weight() {
        let size = [5usize, 3, 3];
        let data: Vec<u8> = (0..45).map(|p| u8::from(p % 5 >= 3)).collect();
        let img = Image::from_vec(&size, data).unwrap();
        let out = approximate_signed_distance_map(&img, 1.0, 0.0).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();

        let w = 0.92644f64;
        let expected = [0.5 + 2.0 * w, 0.5 + w, 0.5, -(w - 0.5), -(2.0 * w - 0.5)];
        for (p, &v) in vals.iter().enumerate() {
            assert!(
                (v - expected[p % 5]).abs() < 1e-12,
                "pixel {p}: got {v}, expected {}",
                expected[p % 5]
            );
        }
    }

    /// The iso-contour stage of the mini-pipeline is the public
    /// `iso_contour_distance` with `LevelSetValue = (inside + outside) / 2` and
    /// `FarValue = maximumDistance + 1`, so its errors surface unchanged.
    #[test]
    fn asdm_propagates_the_iso_contour_gradient_error() {
        let img = rows_f64(&[1.0, 0.0, 1.0, 0.0], 3);
        assert_eq!(
            approximate_signed_distance_map(&img, 1.0, 0.0),
            Err(FilterError::IsoContourZeroGradient)
        );
    }
}
