//! Fused single-pass image transforms: read the native pixel type, compute in
//! `f64`, write the native output type — with **no intermediate `Vec<f64>`**.
//!
//! # Why this module exists
//!
//! The port's elementwise filters were all written as
//!
//! ```text
//! let vals = img.to_f64_vec()?;              // alloc #1: n * 8 bytes
//! let out  = parallel::map_slice(&vals, f);  // alloc #2: n * 8 bytes
//! image_from_f64(id, size, img, &out)        // alloc #3: n * size_of::<T>()
//! ```
//!
//! For a 256³ `Float32` image that is 67 MB in, ~335 MB allocated, and four
//! passes over memory where two would do. The dominant cost is **not** the
//! arithmetic and not `malloc`'s bookkeeping — it is the kernel zeroing every
//! page of those intermediates on first touch. Measured on a 256³
//! `rescale_intensity` (`out = a*in + b`, the simplest op there is): the staged
//! form takes 438 829 minor page faults and 395 MB of RSS; the fused form takes
//! 98 403 and 133 MB, and runs 2.4–4× faster.
//!
//! The widening itself is not the problem — `f32 -> f64` is *lossless*, so
//! computing in `f64` costs nothing but the ALU. Only **materializing** the
//! widened volume costs. So [`map_pixels`] keeps the `f64` arithmetic exactly as
//! it was and deletes the buffers: it widens per element in-register, applies
//! the caller's `f64 -> f64` closure, and narrows on store straight into the
//! output buffer.
//!
//! # Bit-parity
//!
//! Every value the closure sees is the same `f64` it saw before (`Scalar::as_f64`
//! is the same lossless cast `to_f64_vec` performed), and every value stored is
//! `Scalar::from_f64` of the same `f64`, exactly as `image_from_f64` did. Only
//! the *buffers between them* are gone. The pass is a [`crate::parallel`] map
//! over independent output elements — element `i` is written by exactly one task
//! from input `i` alone — so it introduces no reduction and nothing is
//! re-associated. The result is bit-identical at any thread count.
//!
//! # Stencils
//!
//! This module is for the **elementwise** family only, where the output pixel is
//! a function of the input pixel at the same index. A sliding-window filter must
//! not route through here: it needs [`crate::neighborhood::WindowView`], which
//! solves a different (and, measured, much larger) problem.

use crate::error::{Error, Result};
use crate::image::Image;
use crate::parallel;
use crate::pixel::{PixelId, Scalar};

/// `out[i] = f(src[i] as f64)`, stored as `target`'s pixel type.
///
/// One pass, one allocation (the output buffer). The intermediate `f64` volume
/// that `to_f64_vec()` + `image_from_f64()` would have materialized never
/// exists. Output geometry is copied from `src`.
///
/// `f` is the *same* `f64 -> f64` closure the staged form applied, so the result
/// is bit-identical — see the module docs.
///
/// # Errors
///
/// If `src` is not a scalar image, or `target` is not a scalar pixel type.
pub fn map_pixels<F>(src: &Image, target: PixelId, f: F) -> Result<Image>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    // Both tags must be scalar. `dispatch_scalar!` resolves a vector tag to its
    // *component* type, which would quietly produce a scalar image of that
    // component instead of erroring — the same trap `image_from_f64` guards.
    if !src.pixel_id().is_scalar() {
        return Err(Error::RequiresScalarPixelType(src.pixel_id()));
    }
    if !target.is_scalar() {
        return Err(Error::RequiresScalarPixelType(target));
    }

    // Two runtime tags, so the pass is monomorphized over the (input, output)
    // pair: the inner loop then reads a plain `&[I]` and writes a `Vec<O>` with
    // no dynamic dispatch and no bounds check.
    let mut out = dispatch_in(src, target, &f)?;
    out.copy_geometry_from(src);
    Ok(out)
}

fn dispatch_in<F>(src: &Image, target: PixelId, f: &F) -> Result<Image>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    fn inner<I: Scalar, F>(src: &Image, target: PixelId, f: &F) -> Result<Image>
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        let pixels = src.scalar_slice::<I>()?;
        dispatch_out(pixels, src.size(), target, f)
    }
    crate::dispatch_scalar_infer!([, _] src.pixel_id(), inner, src, target, f)
}

fn dispatch_out<I, F>(pixels: &[I], size: &[usize], target: PixelId, f: &F) -> Result<Image>
where
    I: Scalar,
    F: Fn(f64) -> f64 + Sync + Send,
{
    fn inner<O: Scalar, I: Scalar, F>(pixels: &[I], size: &[usize], f: &F) -> Result<Image>
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        // The whole point: widen, compute, narrow — all in registers. The only
        // buffer that exists is the output one.
        let out: Vec<O> = parallel::map_slice(pixels, |&x| O::from_f64(f(x.as_f64())));
        Image::from_vec(size, out)
    }
    crate::dispatch_scalar_infer!([, _, _] target, inner, pixels, size, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pass must equal the staged `to_f64_vec` -> map -> `image_from_f64`
    /// form it replaces, bit for bit, for every input/output type pair that
    /// matters — including the narrowing casts, where `Scalar::from_f64`'s
    /// saturation is the behaviour being pinned.
    fn assert_matches_staged(src: &Image, target: PixelId, f: impl Fn(f64) -> f64 + Sync + Send) {
        let staged: Vec<f64> = src.to_f64_vec().unwrap().iter().map(|&v| f(v)).collect();
        let fused = map_pixels(src, target, &f).unwrap();
        assert_eq!(fused.pixel_id(), target);
        assert_eq!(fused.to_f64_vec().unwrap().len(), staged.len());

        // Compare after the same narrowing the staged form would have applied.
        fn narrow<T: Scalar>(vals: &[f64]) -> Vec<f64> {
            vals.iter().map(|&v| T::from_f64(v).as_f64()).collect()
        }
        let expected = crate::dispatch_scalar!(target, narrow, &staged);
        assert_eq!(fused.to_f64_vec().unwrap(), expected);
    }

    #[test]
    fn float32_in_float32_out_matches_the_staged_form() {
        let img = Image::from_vec(&[4, 3], (0..12).map(|i| i as f32 * 1.5).collect()).unwrap();
        assert_matches_staged(&img, PixelId::Float32, |v| v * 2.0 + 1.0);
    }

    #[test]
    fn uint8_in_float64_out_matches_the_staged_form() {
        let img = Image::from_vec(&[4, 3], (0..12u8).collect()).unwrap();
        assert_matches_staged(&img, PixelId::Float64, |v| v / 3.0);
    }

    /// The narrowing exit saturates rather than wrapping; a fused pass must
    /// reproduce that, not sidestep it.
    #[test]
    fn float64_in_uint8_out_saturates_like_the_staged_form() {
        let img = Image::from_vec(&[4], vec![-5.0f64, 3.7, 300.0, 254.9]).unwrap();
        assert_matches_staged(&img, PixelId::UInt8, |v| v);
        let out = map_pixels(&img, PixelId::UInt8, |v| v).unwrap();
        assert_eq!(out.scalar_slice::<u8>().unwrap(), &[0, 3, 255, 254]);
    }

    /// Above `parallel`'s serial threshold the pass runs on the pool; it must
    /// still equal the sequential map.
    #[test]
    fn a_large_image_matches_the_staged_form_on_the_parallel_path() {
        let n = 1 << 16;
        let img = Image::from_vec(&[n], (0..n).map(|i| (i % 251) as f32).collect()).unwrap();
        assert_matches_staged(&img, PixelId::Float32, |v| (v - 12.0) * 0.25);
    }

    #[test]
    fn geometry_is_carried_across() {
        let mut img = Image::from_vec(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-3.0, 7.0]).unwrap();
        let out = map_pixels(&img, PixelId::Float32, |v| v).unwrap();
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
    }

    #[test]
    fn a_vector_target_is_rejected() {
        let img = Image::from_vec(&[2, 2], vec![1.0f32; 4]).unwrap();
        assert!(map_pixels(&img, PixelId::VectorFloat64, |v| v).is_err());
    }
}
