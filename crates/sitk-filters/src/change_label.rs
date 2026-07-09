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
//! [`crate::quantize_to_pixel_type`]) before being handed to the filter.
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
//! pixel types, matching [`crate::modulus`]'s `IntegerPixelIDTypeList` gate.

use crate::error::Result;
use crate::logic::require_integer_pixel_type;
use crate::{image_from_f64, quantize_to_pixel_type};
use sitk_core::Image;
use std::collections::HashMap;

/// `ChangeLabelImageFilter::SetChangeMap`: `change_map` is a list of
/// `(original, result)` pairs, matching `std::map<double, double>`'s
/// raw-key ordering and overwrite semantics (see the module docs). A pixel
/// whose value matches no `original` entry passes through unchanged.
pub fn change_label(image: &Image, change_map: &[(f64, f64)]) -> Result<Image> {
    require_integer_pixel_type(image)?;
    let id = image.pixel_id();

    let mut pairs: Vec<(f64, f64)> = change_map.to_vec();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut lookup: HashMap<u64, f64> = HashMap::with_capacity(pairs.len());
    for (original, result) in pairs {
        let key = quantize_to_pixel_type(id, original);
        let value = quantize_to_pixel_type(id, result);
        lookup.insert(key.to_bits(), value);
    }

    let vals = image.to_f64_vec()?;
    let out: Vec<f64> = vals
        .iter()
        .map(|&v| lookup.get(&v.to_bits()).copied().unwrap_or(v))
        .collect();

    image_from_f64(id, image.size(), image, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FilterError;
    use sitk_core::PixelId;

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

    /// `pixel_types: IntegerPixelIDTypeList` -- a floating-point image is
    /// rejected, matching `crate::modulus`'s same-shaped gate.
    #[test]
    fn rejects_float_pixel_type() {
        let image = Image::from_vec(&[1, 1], vec![1.0f32]).unwrap();
        assert_eq!(
            change_label(&image, &[]),
            Err(FilterError::RequiresIntegerPixelType(image.pixel_id()))
        );
    }
}
