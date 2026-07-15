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
//! the *buffers between them* are gone. The pass is a [`crate::core::parallel`] map
//! over independent output elements — element `i` is written by exactly one task
//! from input `i` alone — so it introduces no reduction and nothing is
//! re-associated. The result is bit-identical at any thread count.
//!
//! # Stencils
//!
//! This module is for the **elementwise** family only, where the output pixel is
//! a function of the input pixel at the same index. A sliding-window filter must
//! not route through here: it needs [`crate::core::neighborhood::WindowView`], which
//! solves a different (and, measured, much larger) problem.

use crate::core::error::{Error, Result};
use crate::core::image::Image;
use crate::core::parallel;
use crate::core::pixel::{PixelId, Scalar};

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
    // Checked here so a vector `target` is rejected before it is allocated.
    if !src.pixel_id().is_scalar() {
        return Err(Error::RequiresScalarPixelType(src.pixel_id()));
    }
    if !target.is_scalar() {
        return Err(Error::RequiresScalarPixelType(target));
    }

    let mut dst = Image::new(src.size(), target);
    map_pixels_into(src, &mut dst, f)?;
    Ok(dst)
}

/// [`map_pixels`] writing into a destination image the **caller owns**: the same
/// single pass, but the output volume outlives the call.
///
/// `dst`'s pixel type *is* the target type — there is no separate `target`
/// argument, because a destination that already exists has already answered that
/// question. On success `dst` holds `f` applied to every pixel of `src`, and
/// `dst`'s geometry (spacing, origin, direction) has been overwritten with
/// `src`'s, exactly as [`map_pixels`] sets it on the image it returns.
///
/// # Why this exists
///
/// [`map_pixels`] deleted the *intermediate* buffers; it still allocates the
/// output one, and a fresh output volume costs a page fault per 4 KiB on first
/// touch (see [`crate::core::alloc`]). A caller that runs the pass repeatedly — a
/// registration loop, a pyramid level, an iterative filter — can now hoist that
/// out: allocate `dst` once, call this in the loop, and the pages are faulted
/// once for the whole loop instead of once per iteration. Nothing else can
/// remove that cost, because a function returning `Image` by value has nowhere
/// to put a reused buffer.
///
/// [`map_pixels`] is this function plus an allocation, so both share one loop
/// body.
///
/// # Errors — the destination is checked, never adjusted
///
/// - [`Error::RequiresScalarPixelType`] if `src` or `dst` is not scalar.
/// - [`Error::DestinationSizeMismatch`] if `dst.size() != src.size()`.
///
/// A mismatched destination is a **caller error, and is never silently
/// repaired**. Resizing `dst` to fit would make its size mean two different
/// things — "the size I asked for" on one path and "whatever the last call left
/// behind" on the other — and a caller that got the size wrong has, far more
/// likely, passed the wrong buffer than asked for a resize. It would also defeat
/// the entire purpose: the buffer exists to be *reused*, and a call that
/// reallocates it silently is the fresh allocation this API was built to
/// eliminate, wearing a destination's clothes.
///
/// Geometry is the one thing not checked, because it is an **output**, not an
/// input: the operation defines the result's geometry to be `src`'s, so `dst`'s
/// previous spacing/origin/direction carry no information and are overwritten.
/// Requiring the caller to pre-match them would be asking them to guess a value
/// this function is about to set.
pub fn map_pixels_into<F>(src: &Image, dst: &mut Image, f: F) -> Result<()>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    if !src.pixel_id().is_scalar() {
        return Err(Error::RequiresScalarPixelType(src.pixel_id()));
    }
    if !dst.pixel_id().is_scalar() {
        return Err(Error::RequiresScalarPixelType(dst.pixel_id()));
    }
    if dst.size() != src.size() {
        return Err(Error::DestinationSizeMismatch {
            expected: src.size().to_vec(),
            actual: dst.size().to_vec(),
        });
    }

    // Two runtime tags, so the pass is monomorphized over the (input, output)
    // pair: the inner loop then reads a plain `&[I]` and writes a `&mut [O]`
    // with no dynamic dispatch and no bounds check.
    dispatch_in(src, dst, &f)?;
    dst.copy_geometry_from(src);
    Ok(())
}

fn dispatch_in<F>(src: &Image, dst: &mut Image, f: &F) -> Result<()>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    fn inner<I: Scalar, F>(src: &Image, dst: &mut Image, f: &F) -> Result<()>
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        let pixels = src.scalar_slice::<I>()?;
        dispatch_out(pixels, dst, f)
    }
    crate::core::dispatch_scalar_infer!([, _] src.pixel_id(), inner, src, dst, f)
}

fn dispatch_out<I, F>(pixels: &[I], dst: &mut Image, f: &F) -> Result<()>
where
    I: Scalar,
    F: Fn(f64) -> f64 + Sync + Send,
{
    fn inner<O: Scalar, I: Scalar, F>(pixels: &[I], dst: &mut Image, f: &F) -> Result<()>
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        // The whole point: widen, compute, narrow — all in registers. No buffer
        // between the input and the caller's output.
        let out = dst.scalar_vec_mut::<O>()?;
        parallel::map_slice_into(pixels, out, |&x| O::from_f64(f(x.as_f64())));
        Ok(())
    }
    let target = dst.pixel_id();
    crate::core::dispatch_scalar_infer!([, _, _] target, inner, pixels, dst, f)
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
        let expected = crate::core::dispatch_scalar!(target, narrow, &staged);
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

    /// The two forms must be the same pass. If they ever drift, this is what
    /// catches it: same input, same closure, bit-identical output.
    #[test]
    fn the_into_form_and_the_allocating_form_agree_bit_for_bit() {
        let n = 1 << 16;
        let img = Image::from_vec(&[n], (0..n).map(|i| (i % 251) as f32).collect()).unwrap();
        let f = |v: f64| (v - 12.0) * 0.25;

        let allocated = map_pixels(&img, PixelId::Float64, f).unwrap();
        let mut dst = Image::new(&[n], PixelId::Float64);
        map_pixels_into(&img, &mut dst, f).unwrap();

        assert_eq!(
            dst.scalar_slice::<f64>().unwrap(),
            allocated.scalar_slice::<f64>().unwrap()
        );
    }

    /// The point of the API: a destination survives the call and can be written
    /// again. The second pass must leave no trace of the first — every pixel is
    /// overwritten, not merged.
    #[test]
    fn a_destination_can_be_written_repeatedly() {
        let img = Image::from_vec(&[4], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
        let mut dst = Image::new(&[4], PixelId::Float64);

        map_pixels_into(&img, &mut dst, |v| v * 10.0).unwrap();
        assert_eq!(
            dst.scalar_slice::<f64>().unwrap(),
            &[10.0, 20.0, 30.0, 40.0]
        );

        map_pixels_into(&img, &mut dst, |v| v + 0.5).unwrap();
        assert_eq!(dst.scalar_slice::<f64>().unwrap(), &[1.5, 2.5, 3.5, 4.5]);
    }

    /// A wrong-sized destination is an error, never a silent resize — including
    /// the case where the pixel *count* is right and only the shape is wrong,
    /// which a length check alone would wave through.
    #[test]
    fn a_wrong_sized_destination_is_rejected_and_left_untouched() {
        let img = Image::from_vec(&[2, 3], vec![1.0f64; 6]).unwrap();

        let mut too_small = Image::new(&[2, 2], PixelId::Float64);
        let err = map_pixels_into(&img, &mut too_small, |v| v).unwrap_err();
        assert!(matches!(err, Error::DestinationSizeMismatch { .. }));

        // Same 6 pixels, transposed shape: still rejected.
        let mut reshaped = Image::new(&[3, 2], PixelId::Float64);
        let err = map_pixels_into(&img, &mut reshaped, |v| v).unwrap_err();
        assert!(matches!(err, Error::DestinationSizeMismatch { .. }));
        assert_eq!(
            reshaped.size(),
            &[3, 2],
            "a rejected destination is untouched"
        );
        assert_eq!(reshaped.scalar_slice::<f64>().unwrap(), &[0.0; 6]);
    }

    /// The destination's pixel type *is* the target type, so a destination whose
    /// type is not scalar has no meaning — and must not be reinterpreted as its
    /// component type.
    #[test]
    fn a_non_scalar_destination_is_rejected() {
        let img = Image::from_vec(&[2, 2], vec![1.0f32; 4]).unwrap();
        let mut dst = Image::new(&[2, 2], PixelId::VectorFloat64);
        let err = map_pixels_into(&img, &mut dst, |v| v).unwrap_err();
        assert!(matches!(err, Error::RequiresScalarPixelType(_)));
    }

    /// Geometry is an output, not an input: the destination's own spacing and
    /// origin carry no information into the call and are replaced by the
    /// source's, exactly as the allocating form sets them.
    #[test]
    fn the_destination_geometry_is_overwritten_with_the_sources() {
        let mut img = Image::from_vec(&[2, 2], vec![1.0f32; 4]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-3.0, 7.0]).unwrap();

        let mut dst = Image::new(&[2, 2], PixelId::Float32);
        dst.set_spacing(&[9.0, 9.0]).unwrap();
        map_pixels_into(&img, &mut dst, |v| v).unwrap();

        assert_eq!(dst.spacing(), img.spacing());
        assert_eq!(dst.origin(), img.origin());
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
