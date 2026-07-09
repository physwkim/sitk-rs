//! ITK's distance-map filters: `SignedMaurerDistanceMapImageFilter`,
//! `DanielssonDistanceMapImageFilter`, `SignedDanielssonDistanceMapImageFilter`.
//!
//! Verified against ITK's `Modules/Filtering/DistanceMap/include/`:
//! `itkSignedMaurerDistanceMapImageFilter.hxx`,
//! `itkDanielssonDistanceMapImageFilter.hxx`,
//! `itkSignedDanielssonDistanceMapImageFilter.hxx`,
//! `itkReflectiveImageRegionConstIterator.hxx`.
//!
//! All three always produce a `Float64` output image. SimpleITK's yaml fixes
//! the procedural output at `Float32`; we use `f64` throughout (including the
//! internal accumulation) so callers get true double precision instead of
//! ITK's `float` rounding.
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

use crate::Result;
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
    let input_vals = img.to_f64_vec();
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
    let vals = img.to_f64_vec();
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
    let vals = img.to_f64_vec();
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
}
