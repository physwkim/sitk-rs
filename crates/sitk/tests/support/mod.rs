//! The straddle probes, shared by every device test — one copy, so the two metrics
//! cannot drift apart in what they mean by "no sample sits on a boundary".
//!
//! # What a straddle is, and why these probes now expect to find none
//!
//! The host evaluates the transform's own expression. The device is handed the
//! transform's own **stages** ([`ParametricTransform::point_map_stages`]) — its stored
//! `matrix`/`offset` pairs, one per map it applies, in the order it applies them — and
//! replays them, one rounded `mat_vec` plus one rounded add per stage. Same operations,
//! same order, so the mapped point and the continuous index are the host's **bit for
//! bit**.
//!
//! They were not always. The device used to be handed an affine form *probed* out of the
//! transform (`b = T(0)`, `A[d][e] = T(e_e)[d] − b[d]`), which is the same map in exact
//! arithmetic and differs in the last ulp in f64, because recovering `A[d][e]` from
//! `T(e_e) − T(0)` subtracts the offset back off. That ulp was invisible in every
//! *continuous* output and decisive in every *discrete* one, and this module has one probe
//! per discrete decision the sampler makes:
//!
//! * [`cell_boundary_straddles`] — `floor(c)`, which cell of the moving grid the sample
//!   sits in. The trilinear interpolant is continuous across a cell wall; **its gradient
//!   is not**, so a sample on `c_d = integer` gets one of two O(1)-different gradients
//!   depending on which side the last ulp lands it. Ledger §2.158.
//! * [`in_buffer_straddles`] — `is_inside`, i.e. `c_d ∈ [−0.5, size_d − 0.5)`. Decides
//!   whether the sample is a sample at all, and therefore `valid_points`.
//! * [`moving_mask_straddles`] — `round(c_d)`, which mask voxel gates the sample. Same
//!   consequence: the sample exists or it does not.
//!
//! Under the stage list all three are **empty by construction**, for every transform the
//! device accepts: two chains that perform the same operations on the same bits in the
//! same order cannot disagree about a branch. The probes are kept, and the pins that call
//! them assert emptiness rather than banding a disagreement — they are the standing
//! evidence that the property holds, and the thing that fails loudly if a future change
//! reintroduces a second arithmetic for the point map.
//!
//! # Why these reproduce the paths rather than approximating them
//!
//! Both matrices come from the same `sitk_transform` helpers the metric itself calls
//! (`index_to_physical_matrix`, `physical_to_index_matrix`); the device point map is the
//! stage replay the kernel performs ([`replay_stages`], whose per-stage arithmetic is
//! `mat_vec` then `+ offset`, which is what the kernel's `fmadd_rn` chain and its final
//! `__dadd_rn` are); and both accumulator chains are written in the order the two paths
//! write them (`VirtualGrid::write_point`; `resident.rs`'s `fmadd_rn`, a rounded multiply
//! and a rounded add, not a fused one). A probe that computed a third arithmetic of its
//! own would answer a question nobody asked.
//!
//! What these probes model is the kernel's chain, not the kernel itself — the device's `c`
//! is not observable from the host. What *is* observable is every discrete decision taken
//! on it: the valid-sample count and the metric moments. Those are pinned against the CPU
//! path on the device itself (`cuda_boundary.rs`, `cuda_mean_squares.rs`), and it is the
//! agreement of those that closes the loop the model opens.
//!
//! 3-D only, and the direction matrix must be the identity — asserted, not assumed. Every
//! test image in this crate is built that way, and a probe that silently mis-handled an
//! oblique direction would report "no straddles" for the wrong reason.
#![allow(dead_code)] // each test binary uses a different subset of these probes

use sitk::core::Image;
use sitk::transform::ParametricTransform;
use sitk::transform::interpolator::{index_to_physical_matrix, physical_to_index_matrix};
use sitk::transform::matrix_offset::replay_stages;

/// True when the device is absent — a supported configuration, and the reason the
/// fallback exists.
pub fn no_device() -> bool {
    matches!(
        sitk::cuda::backend(),
        Err(sitk::cuda::CudaError::NoDevice(_))
    )
}

/// One fixed sample, mapped to a moving continuous index down both paths.
#[derive(Clone, Copy, Debug)]
pub struct Mapped {
    /// The fixed sample's grid voxel.
    pub index: [usize; 3],
    /// The continuous moving index the **host** computes for it.
    pub host_c: [f64; 3],
    /// The continuous moving index the **device** computes for it.
    pub dev_c: [f64; 3],
}

impl Mapped {
    /// The largest per-axis gap between the two paths' continuous indices.
    pub fn gap(&self) -> f64 {
        (0..3)
            .map(|d| (self.host_c[d] - self.dev_c[d]).abs())
            .fold(0.0f64, f64::max)
    }
}

fn identity_direction(img: &Image) -> bool {
    img.direction() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].as_slice()
}

/// Walk every fixed voxel, handing the caller the sample's continuous moving index as
/// each path computes it.
fn for_each_sample(
    fixed: &Image,
    moving: &Image,
    t: &dyn ParametricTransform,
    mut f: impl FnMut(Mapped),
) {
    assert_eq!(fixed.dimension(), 3, "the probes are 3-D");
    assert!(
        identity_direction(fixed) && identity_direction(moving),
        "the probes reproduce the two paths' arithmetic for an identity direction only"
    );
    let size = fixed.size().to_vec();
    let forigin = fixed.origin().to_vec();
    let fmat = index_to_physical_matrix(fixed.direction(), fixed.spacing(), 3);
    let msize = moving.size().to_vec();
    let morigin = moving.origin().to_vec();
    let mmat = physical_to_index_matrix(moving.direction(), moving.spacing(), 3)
        .expect("singular moving direction");
    let _ = msize;

    // The point map the device is given: the transform's own stages, exactly as
    // `cuda::affine_form` takes them. A transform with no stages is not a transform the
    // device runs, so a probe of one would be measuring nothing.
    let stages = t
        .point_map_stages()
        .expect("the probes model the device path, which only accepts a transform with stages");

    // c = M·(p − origin), accumulated from zero, left to right — `MovingImage::
    // continuous_index` and the kernel's `c` loop write it in this order.
    let cindex = |p: &[f64; 3]| -> [f64; 3] {
        let mut c = [0.0f64; 3];
        for (r, cr) in c.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for j in 0..3 {
                acc += mmat[r * 3 + j] * (p[j] - morigin[j]);
            }
            *cr = acc;
        }
        c
    };

    for k in 0..size[2] {
        for j in 0..size[1] {
            for i in 0..size[0] {
                // The sample's physical point: `VirtualGrid::write_point`, which the
                // kernel reimplements in the same order. Both paths read the same `x`.
                let idx = [i, j, k];
                let mut x = [0.0f64; 3];
                for (r, xr) in x.iter_mut().enumerate() {
                    let mut acc = forigin[r];
                    for (c, &id) in idx.iter().enumerate() {
                        acc += fmat[r * 3 + c] * id as f64;
                    }
                    *xr = acc;
                }

                let ph = t.transform_point(&x);
                let host_p = [ph[0], ph[1], ph[2]];

                // The kernel's chain: one stage at a time, `mat_vec` then `+ offset`,
                // rounding once per stage — not the stages folded into one matrix.
                let pd = replay_stages(&stages, &x, 3);
                let dev_p = [pd[0], pd[1], pd[2]];

                f(Mapped {
                    index: idx,
                    host_c: cindex(&host_p),
                    dev_c: cindex(&dev_p),
                });
            }
        }
    }
}

/// `sitk::transform::is_inside`, and the kernel's `is_inside`, which is the same rule:
/// `c_d ∈ [−0.5, size_d − 0.5)` on every axis.
fn inside(c: &[f64; 3], size: &[usize]) -> bool {
    (0..3).all(|d| c[d] >= -0.5 && c[d] < size[d] as f64 - 0.5)
}

/// `MovingImage::mask_allows`, and the kernel's copy of it: round to nearest voxel
/// (half away from zero on both paths), reject if that voxel is outside the buffer or
/// zero in the mask.
fn mask_allows(c: &[f64; 3], size: &[usize], strides: &[usize], mask: &[bool]) -> bool {
    let mut flat = 0usize;
    for d in 0..3 {
        let r = c[d].round();
        if r < 0.0 || r as usize >= size[d] {
            return false;
        }
        flat += r as usize * strides[d];
    }
    mask[flat]
}

/// A sample whose moving-grid **cell** differs between the two paths — the ledger
/// §2.158 straddle. Returned: `(fixed index, axis, |Δ∂M/∂axis|)`, one entry per axis
/// whose interpolated gradient jumps by more than `1e-6`.
///
/// This is a *precondition* probe: a pin that bands a ∇M-consuming quantity is
/// measuring reduction rounding only where this returns empty. Where it does not, the
/// two paths return two different one-sided limits of a discontinuous derivative and the
/// difference is O(1)·(sample weight), not O(ε).
pub fn cell_boundary_straddles(
    fixed: &Image,
    moving: &Image,
    t: &dyn ParametricTransform,
) -> Vec<([usize; 3], usize, f64)> {
    let m = moving.to_f64_vec().unwrap();
    let msize = moving.size().to_vec();
    let mstride = [1usize, msize[0], msize[0] * msize[1]];

    // The gradient of the trilinear interpolant at a continuous index, in index space —
    // the quantity whose discontinuity across a cell wall is the whole subject.
    let grad = |c: &[f64; 3]| -> Option<[f64; 3]> {
        if !inside(c, &msize) {
            return None;
        }
        let mut g = [0.0f64; 3];
        for corner in 0..8usize {
            let mut offset = 0usize;
            let mut dw = [1.0f64; 3];
            for (d, &cd) in c.iter().enumerate() {
                let base = cd.floor();
                let frac = cd - base;
                let bit = (corner >> d) & 1;
                let wd = if bit == 1 { frac } else { 1.0 - frac };
                let dwd = if bit == 1 { 1.0 } else { -1.0 };
                for (e, dwe) in dw.iter_mut().enumerate() {
                    *dwe *= if e == d { dwd } else { wd };
                }
                let idx = (base as isize + bit as isize).clamp(0, msize[d] as isize - 1) as usize;
                offset += idx * mstride[d];
            }
            for (e, ge) in g.iter_mut().enumerate() {
                *ge += dw[e] * m[offset];
            }
        }
        Some(g)
    };

    let mut out = Vec::new();
    for_each_sample(fixed, moving, t, |s| {
        if let (Some(gh), Some(gd)) = (grad(&s.host_c), grad(&s.dev_c)) {
            for (e, (&h, &d)) in gh.iter().zip(gd.iter()).enumerate() {
                if (h - d).abs() > 1e-6 {
                    out.push((s.index, e, (h - d).abs()));
                }
            }
        }
    });
    out
}

/// A sample the two paths **disagree about the existence of**: inside the moving buffer
/// on one path and outside on the other. Every such sample is a `valid_points`
/// disagreement, and `valid_points` is asserted with `assert_eq!` everywhere.
pub fn in_buffer_straddles(
    fixed: &Image,
    moving: &Image,
    t: &dyn ParametricTransform,
) -> Vec<Mapped> {
    let msize = moving.size().to_vec();
    let mut out = Vec::new();
    for_each_sample(fixed, moving, t, |s| {
        if inside(&s.host_c, &msize) != inside(&s.dev_c, &msize) {
            out.push(s);
        }
    });
    out
}

/// Samples sitting **exactly on** the in-buffer predicate's boundary on the host path:
/// `c_d == −0.5` (the closed end — inside) or `c_d == size_d − 0.5` (the open end —
/// outside). Returned with the axis that lands on it.
///
/// These are the samples where a 1-ulp difference between the two paths flips the
/// predicate. A geometry that produces none of them cannot exercise
/// [`in_buffer_straddles`], so a pin that reports "no disagreement" over such a geometry
/// has proved nothing — which is why the construction test asserts this is non-empty
/// before it measures anything.
pub fn on_buffer_boundary(
    fixed: &Image,
    moving: &Image,
    t: &dyn ParametricTransform,
) -> Vec<(Mapped, usize)> {
    let msize = moving.size().to_vec();
    let mut out = Vec::new();
    for_each_sample(fixed, moving, t, |s| {
        for (d, &size) in msize.iter().enumerate() {
            let hi = size as f64 - 0.5;
            if s.host_c[d] == hi || s.host_c[d] == -0.5 {
                out.push((s, d));
            }
        }
    });
    out
}

/// A sample the two paths **gate differently through the moving mask**: `round(c_d)`
/// picks one mask voxel on the host path and another on the device path, and the two
/// voxels do not agree about whether the sample is in.
///
/// `mask` is the moving mask on the moving image's own grid, in the same
/// nonzero-is-inside convention as `MovingImage::with_moving_mask`.
pub fn moving_mask_straddles(
    fixed: &Image,
    moving: &Image,
    mask: &Image,
    t: &dyn ParametricTransform,
) -> Vec<Mapped> {
    assert_eq!(mask.size(), moving.size(), "the mask is on the moving grid");
    let bits = mask_bits(mask);
    let msize = moving.size().to_vec();
    let mstride = [1usize, msize[0], msize[0] * msize[1]];

    let mut out = Vec::new();
    for_each_sample(fixed, moving, t, |s| {
        // The mask gate runs *before* the in-buffer test on both paths, but a sample
        // outside the buffer is dropped either way and is not a mask straddle.
        if !inside(&s.host_c, &msize) && !inside(&s.dev_c, &msize) {
            return;
        }
        let h = mask_allows(&s.host_c, &msize, &mstride, &bits);
        let d = mask_allows(&s.dev_c, &msize, &mstride, &bits);
        if h != d {
            out.push(s);
        }
    });
    out
}

/// Samples whose host continuous index lands **exactly on a rounding tie** —
/// `c_d == n + 0.5` for an integer `n`, where `round` (half away from zero on both
/// paths) jumps to the next mask voxel and a 1-ulp difference picks the other one.
///
/// The counterpart of [`on_buffer_boundary`] for the mask gate, and used the same way:
/// it proves the constructed geometry actually puts samples where the disagreement
/// could happen.
pub fn on_round_tie(
    fixed: &Image,
    moving: &Image,
    t: &dyn ParametricTransform,
) -> Vec<(Mapped, usize)> {
    let mut out = Vec::new();
    for_each_sample(fixed, moving, t, |s| {
        for d in 0..3 {
            if (s.host_c[d] - s.host_c[d].floor()) == 0.5 {
                out.push((s, d));
            }
        }
    });
    out
}

/// **The invariant.** Samples whose continuous index differs between the two paths *in any
/// bit* — `to_bits()` equality, not a tolerance.
///
/// Every other probe here reports a *consequence* (a different cell, a different side of
/// the buffer wall, a different mask voxel), and each of them can be empty by luck of the
/// pose while the underlying indices still differ. This is the property itself, and for
/// every transform the device accepts it must be empty at every sample of every pose.
pub fn index_bit_mismatches(
    fixed: &Image,
    moving: &Image,
    t: &dyn ParametricTransform,
) -> Vec<Mapped> {
    let mut out = Vec::new();
    for_each_sample(fixed, moving, t, |s| {
        if (0..3).any(|d| s.host_c[d].to_bits() != s.dev_c[d].to_bits()) {
            out.push(s);
        }
    });
    out
}

/// Samples whose **host** continuous index lands exactly on a moving-grid cell wall —
/// `c_d == floor(c_d)` — where `floor` picks the cell and the trilinear gradient is
/// discontinuous. Returned: `(sample, axis)` per axis that lands on a wall.
///
/// The counterpart of [`on_buffer_boundary`] and [`on_round_tie`] for the cell wall, and
/// used the same way: [`cell_boundary_straddles`] reports where the two paths *disagree*,
/// which is now empty by construction, so a pin that asserts agreement needs this to prove
/// the pose actually puts a sample on the wall — otherwise it agrees about nothing.
pub fn on_cell_wall(
    fixed: &Image,
    moving: &Image,
    t: &dyn ParametricTransform,
) -> Vec<(Mapped, usize)> {
    let msize = moving.size().to_vec();
    let mut out = Vec::new();
    for_each_sample(fixed, moving, t, |s| {
        if !inside(&s.host_c, &msize) {
            return;
        }
        for d in 0..3 {
            if s.host_c[d] == s.host_c[d].floor() {
                out.push((s, d));
            }
        }
    });
    out
}

/// A moving mask's voxels as the metric reads them: nonzero is inside.
pub fn mask_bits(mask: &Image) -> Vec<bool> {
    mask.to_f64_vec()
        .unwrap()
        .iter()
        .map(|&v| v != 0.0)
        .collect()
}
