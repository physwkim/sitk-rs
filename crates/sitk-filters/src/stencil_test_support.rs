//! Shared fixtures for the `thread_parity` pins that guard the parallel stencils.
//!
//! Every stencil filter in this crate that was converted from a serial
//! `iter().map().collect()` to a [`sitk_core::NeighborhoodIterator`] window pass
//! carries a pin: its output must be bit-identical to a hand-kept copy of the
//! exact serial loop it replaced, at every thread count. Those pins all need the
//! same two things — a volume big enough that the parallel path actually runs,
//! and a way to say *why* the pin is not vacuous — so they live here rather than
//! being copied into eight modules.
//!
//! # Two kinds of stencil, two kinds of non-vacuity
//!
//! A pin proves nothing unless it *could* fail. What makes it able to fail
//! depends on what the stencil does, and the two cases need different guards:
//!
//! * **Stencils that sum** (`canny`, the separable Gaussian-derivative pass,
//!   `anisotropic_diffusion`, `min_max_curvature_flow`) accumulate `f64` terms
//!   over a window. `f64` addition is not associative, so re-ordering the terms
//!   moves the bits — *if* the values are rich enough to round.
//!   [`window_sum_order_is_observable`] measures exactly that, on the volume the
//!   pin actually uses, and it is the guard those pins rely on.
//!
//! * **Stencils that compare** (`morphology`'s grayscale erode/dilate,
//!   `binary_morphology`, `geodesic_morphology`, `contour`) take a min, a max, or
//!   an equality count. Those are order-*insensitive*: reversing the window
//!   changes nothing, and asserting otherwise would be asserting a falsehood. The
//!   thing that can go wrong in them is a **wrong window slot** — reading the
//!   neighbor at the wrong offset — so their guard is
//!   [`a_perturbed_window_slot_changes_the_output`]: it shows the output really
//!   does depend on which slot is read, so a mis-addressed window would be caught.
//!
//! Writing a fold-order guard for a min/max filter would look rigorous and assert
//! nothing. Saying which guard applies, per filter, is the point.

use sitk_core::{Image, PixelId};

/// A 32³ volume — 32 768 voxels, above `sitk_core::parallel`'s 16 384 serial
/// threshold, so a window pass really runs on rayon instead of taking the serial
/// fast path and pinning nothing.
///
/// The values are irregular (not a smooth ramp, whose differences vanish
/// identically and would hide a wrong tap) and are built in `f64` first, then
/// narrowed, so the two pixel types carry the *same* underlying signal:
///
/// * `Float64` carries full 53-bit mantissas, so window sums genuinely round and
///   their term order is observable. This is the volume with teeth.
/// * `Float32` carries 24-bit mantissas and exercises the widening-per-access
///   path that replaced the deleted `f64` volume copies.
pub(crate) fn volume(pixel: PixelId) -> Image {
    let n = 32usize;
    let mut data = vec![0.0f64; n * n * n];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (i as f64, j as f64, k as f64);
                data[(k * n + j) * n + i] = (0.7 * x).sin() * 40.0
                    + (0.3 * y).cos() * 25.0
                    + (x * y * 0.01 + z * 0.9).sin() * 13.0
                    + ((i * 37 + j * 11 + k * 7) % 29) as f64;
            }
        }
    }
    assert!(
        n * n * n > 1 << 14,
        "the volume must exceed parallel's serial threshold, or the parallel path never runs"
    );
    let mut img = match pixel {
        PixelId::Float64 => Image::from_vec(&[n, n, n], data).unwrap(),
        PixelId::Float32 => {
            let d: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            Image::from_vec(&[n, n, n], d).unwrap()
        }
        other => panic!("volume() does not build {other:?}"),
    };
    img.set_spacing(&[1.0, 0.75, 1.3]).unwrap();
    img
}

/// A binary 32³ volume with blobs, for the morphology stencils — a mask whose
/// foreground is neither everything nor nothing, so an erode and a dilate both
/// actually move pixels.
pub(crate) fn binary_volume() -> Image {
    let n = 32usize;
    let src = volume(PixelId::Float64);
    let vals = src.to_f64_vec().unwrap();
    let data: Vec<u8> = vals.iter().map(|&v| u8::from(v > 30.0)).collect();
    let fg = data.iter().filter(|&&v| v == 1).count();
    assert!(
        fg > data.len() / 10 && fg < data.len() * 9 / 10,
        "binary_volume is degenerate: {fg}/{} foreground",
        data.len()
    );
    Image::from_vec(&[n, n, n], data).unwrap()
}

/// The thread counts every pin walks.
pub(crate) const THREADS: [usize; 4] = [1, 4, 48, 96];

/// Both pixel types, in the order the pins walk them.
pub(crate) const PIXELS: [PixelId; 2] = [PixelId::Float64, PixelId::Float32];

/// Assert two `f64` slices are equal **bit for bit**, naming the first voxel that
/// moved and the context it moved in.
pub(crate) fn assert_bits_eq(got: &[f64], expected: &[f64], context: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{context}: length changed ({} vs {})",
        got.len(),
        expected.len()
    );
    for (i, (a, b)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{context}: moved at voxel {i}: {a:?} ({:#x}) vs serial {b:?} ({:#x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

/// Does the order of a window sum change its bits on this image?
///
/// Walks a `radius` window over `img` and, at each voxel, sums the window's
/// values forward and again in reverse. Returns true as soon as one voxel's two
/// sums differ bit for bit.
///
/// This is the non-vacuity guard for every summing stencil. If it returns false,
/// the input is too benign for the pin to mean anything: `f64` addition would be
/// associative *on this data*, so a filter that re-associated its window sum
/// would still produce identical bits and the pin could not fail. That is not a
/// hypothetical — on an `f32` volume, a Sobel window's power-of-two weights make
/// every partial sum exactly representable in `f64`, and a pin there asserts
/// nothing.
pub(crate) fn window_sum_order_is_observable(img: &Image, radius: &[usize]) -> bool {
    use sitk_core::{NeighborhoodIterator, ZeroFluxNeumannBoundaryCondition};
    let iter = NeighborhoodIterator::<f64, _>::new(img, radius, ZeroFluxNeumannBoundaryCondition)
        .expect("float64 volume");
    iter.par_map_window(|_, w| {
        let values: Vec<f64> = w.iter_f64().collect();
        let forward: f64 = values.iter().sum();
        let reverse: f64 = values.iter().rev().sum();
        forward.to_bits() != reverse.to_bits()
    })
    .into_iter()
    .any(|differs| differs)
}
