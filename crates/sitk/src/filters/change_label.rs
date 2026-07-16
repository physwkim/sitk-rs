//! `ChangeLabelImageFilter`: remap sets of label values.
//!
//! Port of ITK `Modules/Filtering/ImageLabel/include/itkChangeLabelImageFilter.h`:
//! a `UnaryFunctorImageFilter` around `Functor::ChangeLabel`, whose
//! `operator()` looks a pixel's value up in a `std::map<InputPixelType,
//! OutputPixelType>` and returns the mapped value if present, otherwise the
//! original value unchanged. Every pixel is looked up exactly once against
//! the *original* input, so a chain like `1 -> 2, 2 -> 3` in the map is
//! **not** transitive: an input pixel valued `1` becomes `2`, not `3`.
//!
//! `ChangeLabelImageFilter.yaml`'s `ChangeMap` member is `std::map<double,
//! double>`, cast per-entry to `InputPixelType`/`OutputPixelType`
//! (`static_cast`, i.e. truncating for integer types, matching
//! `crate::filters::quantize_to_pixel_type`) before being handed to the filter.
//! `std::map` is keyed and iterated by the *raw, uncast* `double`, so two
//! distinct raw keys that happen to truncate to the same pixel-type value
//! (e.g. `1.2` and `1.4` both truncating to `1` for an integer pixel type)
//! resolve by upstream's ascending-raw-key iteration order, last write
//! wins: `1.4`'s entry (the larger raw key) overwrites `1.2`'s. This port
//! reproduces that exactly, sorting `change_map` by raw key (a stable sort,
//! so a repeated *identical* raw key also resolves last-wins, matching
//! `std::map::operator[]` assignment) before quantizing.
//!
//! No `output_pixel_type` in the yaml: output pixel type matches the input.
//! `pixel_types: IntegerPixelIDTypeList` restricts this filter to integer
//! pixel types, matching [`crate::filters::modulus`]'s `IntegerPixelIDTypeList` gate.

use crate::core::{Image, PixelId, Scalar};
use crate::filters::error::Result;
use crate::filters::logic::require_integer_pixel_type;
use std::collections::HashMap;
use std::hash::Hash;

/// `ChangeLabelImageFilter::SetChangeMap`: `change_map` is a list of
/// `(original, result)` pairs, matching `std::map<double, double>`'s
/// raw-key ordering and overwrite semantics (see the module docs). A pixel
/// whose value matches no `original` entry passes through unchanged.
pub fn change_label(image: &Image, change_map: &[(f64, f64)]) -> Result<Image> {
    require_integer_pixel_type(image)?;

    let mut pairs: Vec<(f64, f64)> = change_map.to_vec();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Dispatch on the (integer) component type and build the map natively, so an
    // un-remapped `UInt64`/`Int64` pixel above `2^53` passes through bit-for-bit
    // rather than being collapsed by a `to_f64_vec` round-trip (`2^53 + 1 ->
    // 2^53`). `require_integer_pixel_type` has already rejected floating-point
    // images, so the float arm is unreachable; a *vector* integer image resolves
    // to its component type here and then errors in `scalar_slice`, exactly as
    // the old `to_f64_vec` did.
    match image.pixel_id().component_id() {
        PixelId::UInt8 => change_label_native::<u8>(image, &pairs),
        PixelId::Int8 => change_label_native::<i8>(image, &pairs),
        PixelId::UInt16 => change_label_native::<u16>(image, &pairs),
        PixelId::Int16 => change_label_native::<i16>(image, &pairs),
        PixelId::UInt32 => change_label_native::<u32>(image, &pairs),
        PixelId::Int32 => change_label_native::<i32>(image, &pairs),
        PixelId::UInt64 => change_label_native::<u64>(image, &pairs),
        PixelId::Int64 => change_label_native::<i64>(image, &pairs),
        other => unreachable!(
            "change_label: require_integer_pixel_type rejects non-integer pixel \
             types (got component {other:?})"
        ),
    }
}

/// The native remap for one integer pixel type: quantize each `(original,
/// result)` pair to `T` (the `f64`-sourced keys/values collapse to `T` via
/// [`Scalar::from_f64`], the same quantization the public `f64` API always
/// implied), then map every pixel through a `HashMap<T, T>`, passing an
/// unmapped pixel through unchanged. The pass-through is a plain `T` copy, so it
/// is exact for `T = u64`/`i64` above `2^53`.
///
/// The sorted `pairs` give `std::map<double, double>`'s ascending-raw-key,
/// last-write-wins resolution: two raw keys that quantize to the same `T`
/// collide in the map and the larger raw key (inserted later) wins.
fn change_label_native<T: Scalar + Eq + Hash>(
    image: &Image,
    pairs: &[(f64, f64)],
) -> Result<Image> {
    let mut lookup: HashMap<T, T> = HashMap::with_capacity(pairs.len());
    for &(original, result) in pairs {
        lookup.insert(T::from_f64(original), T::from_f64(result));
    }

    let vals = image.scalar_slice::<T>()?;
    let out: Vec<T> = vals
        .iter()
        .map(|&v| lookup.get(&v).copied().unwrap_or(v))
        .collect();

    let mut img = Image::from_vec(image.size(), out)?;
    img.copy_geometry_from(image);
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;
    use crate::filters::error::FilterError;

    fn img_i32(size: &[usize], data: Vec<i32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    /// Unmapped values pass through unchanged; mapped values are rewritten.
    #[test]
    fn maps_listed_values_passes_through_the_rest() {
        let image = img_i32(&[5, 1], vec![1, 2, 3, 4, 5]);
        let out = change_label(&image, &[(2.0, 20.0), (4.0, 40.0)]).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[1, 20, 3, 40, 5]);
    }

    /// Empty change map (the yaml's `defaults` test fixture) is identity.
    #[test]
    fn empty_change_map_is_identity() {
        let image = img_i32(&[3, 1], vec![7, 8, 9]);
        let out = change_label(&image, &[]).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[7, 8, 9]);
    }

    /// `1 -> 2` and `2 -> 3` in the same map is not transitive: a pixel
    /// valued 1 becomes 2 (not 3), and a pixel valued 2 becomes 3 -- each
    /// pixel is looked up exactly once against the original input.
    #[test]
    fn chained_mapping_is_not_transitive() {
        let image = img_i32(&[2, 1], vec![1, 2]);
        let out = change_label(&image, &[(1.0, 2.0), (2.0, 3.0)]).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[2, 3]);
    }

    /// Two raw double keys that truncate to the same integer pixel-type key
    /// (`1.2` and `1.4` both truncate to `1`) resolve by ascending-raw-key
    /// order, last write wins -- `1.4`'s entry (100) survives over `1.2`'s
    /// (99), matching `std::map<double,double>`'s iteration order, not the
    /// caller's slice order (the pairs are listed with `1.4` first here).
    #[test]
    fn colliding_quantized_keys_resolve_by_ascending_raw_key() {
        let image = img_i32(&[1, 1], vec![1]);
        let out = change_label(&image, &[(1.4, 100.0), (1.2, 99.0)]).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[100]);
    }

    /// A repeated identical raw key also resolves last-wins.
    #[test]
    fn repeated_identical_key_last_wins() {
        let image = img_i32(&[1, 1], vec![1]);
        let out = change_label(&image, &[(1.0, 5.0), (1.0, 6.0)]).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[6]);
    }

    /// Output pixel type matches the input (no `output_pixel_type` in the
    /// yaml).
    #[test]
    fn output_pixel_type_matches_input() {
        let image = img_i32(&[2, 1], vec![1, 2]);
        let out = change_label(&image, &[(1.0, 9.0)]).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Int32);
    }

    /// An un-remapped `UInt64` pixel above `2^53` passes through bit-for-bit.
    /// `2^53 + 1` is not `f64`-representable, so the old `to_f64_vec` path
    /// collapsed it to `2^53`; the native `HashMap<u64, u64>` pass-through does
    /// not. A remapped small label is still rewritten.
    #[test]
    fn unmapped_u64_above_2_53_passes_through_losslessly() {
        let hard = (1u64 << 53) + 1; // 9_007_199_254_740_993
        let image = Image::from_vec(&[3, 1], vec![1u64, hard, u64::MAX]).unwrap();
        let out = change_label(&image, &[(1.0, 7.0)]).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt64);
        assert_eq!(out.scalar_slice::<u64>().unwrap(), &[7, hard, u64::MAX]);
    }

    /// `pixel_types: IntegerPixelIDTypeList` -- a floating-point image is
    /// rejected, matching `crate::filters::modulus`'s same-shaped gate.
    #[test]
    fn rejects_float_pixel_type() {
        let image = Image::from_vec(&[1, 1], vec![1.0f32]).unwrap();
        assert_eq!(
            change_label(&image, &[]),
            Err(FilterError::RequiresIntegerPixelType(image.pixel_id()))
        );
    }
}
