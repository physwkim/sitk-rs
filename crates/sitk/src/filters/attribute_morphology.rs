//! Morphological opening/closing by attribute (area).
//!
//! Ports of (ITK `Modules/Nonunit/Review/include/`):
//!
//! - [`area_opening`] -- `itkAreaOpeningImageFilter.h`, a thin instantiation
//!   of `itkAttributeMorphologyBaseImageFilter.hxx` with `TFunction =
//!   std::greater<InputPixelType>` (and `TAttribute` = ITK's spacing value
//!   type, always `double`).
//! - [`area_closing`] -- `itkAreaClosingImageFilter.h`, the same base class
//!   with `TFunction = std::less<InputPixelType>`.
//!
//! `AttributeKind` plays the same role here as `ExtremaKind` does in
//! [`crate::filters::regional_extrema`] and `ReconstructionKind` does in
//! [`crate::filters::reconstruction`].
//!
//! ## The sweep
//!
//! `itkAttributeMorphologyBaseImageFilter.hxx`'s `GenerateData()`:
//!
//! 1. If `Lambda <= 0.0`, casts the input straight through -- a documented
//!    fast path, not an error. `Lambda`'s SimpleITK default is `0.0`, so
//!    **[`area_opening`]/[`area_closing`] are no-ops unless the caller
//!    raises `lambda` above zero.**
//! 2. Otherwise, sorts every pixel's flat index by raw pixel value
//!    (descending for opening/`std::greater`, ascending for closing/
//!    `std::less`), ties broken by ascending flat index (`std::stable_sort`
//!    preserves the identity-initialized order for ties).
//! 3. Sweeps the sorted order once, building a max-tree (opening) or
//!    min-tree (closing) via union-find: each pixel starts as its own
//!    one-pixel component (`MakeSet`: attribute = 1, or with
//!    `UseImageSpacing`, the product of the image spacing). Each of its
//!    neighbors that either ties its value or was already swept gets
//!    unioned in: `Criterion` accepts the merge (accumulating the
//!    neighbor's attribute into this pixel's) when the two share a raw
//!    value or the neighbor's accumulated attribute is still under
//!    `Lambda`; otherwise the neighbor's component has already proven
//!    itself a permanent feature, the merge is refused, and this pixel's
//!    attribute is pinned to `Lambda`.
//! 4. Resolves every non-root pixel's final value from its parent in
//!    reverse sweep order -- a parent always sweeps later than its
//!    children, so a single backward pass suffices.
//!
//! ## Fixed upstream bug: the last raster pixel was exempt from the sort
//!
//! `GenerateData()` called `std::stable_sort(&m_SortPixels[0],
//! &m_SortPixels[buffsize - 1], ...)`: the *exclusive* end iterator was
//! `&m_SortPixels[buffsize - 1]`, one slot short of
//! `&m_SortPixels[buffsize]`. Since `m_SortPixels` is identity-initialized
//! (`m_SortPixels[pos] = pos`) before the sort, this off-by-one left array
//! slot `buffsize - 1` completely untouched: it always held flat index
//! `buffsize - 1` (the image's last pixel in raster order) and was
//! therefore **always swept last, regardless of that pixel's actual
//! value.**
//!
//! This was not merely a reordering effect: `FindRoot` treats "never
//! visited" (`m_Parent[x] == INACTIVE == -1`) and "a genuine root"
//! (`m_Parent[x] == ACTIVE == -2`) identically -- both are simply `< 0`. If
//! some other pixel was swept earlier and, following the normal sweep rule,
//! unioned against flat index `buffsize - 1` *before that pixel's own
//! turn*, `FindRoot` silently treated the not-yet-visited pixel as an
//! already-resolved root with its sentinel `AuxData` of `-1`, which
//! satisfies `Criterion`'s `AuxData[r] < Lambda` for essentially any
//! positive `Lambda` and got merged in -- corrupting the absorbing
//! component's accumulated attribute by `-1`. The premature parent link
//! itself was silently overwritten once flat index `buffsize - 1` finally
//! reached its own turn (`MakeSet` unconditionally resets it to `ACTIVE`),
//! but the corruption already applied to the *other* component's
//! `AuxData` was not undone. Because that corrupted delta is a fixed offset
//! unrelated to any real pixel value, it could flip a `Criterion` decision
//! that would otherwise land exactly on the `Lambda` boundary, changing
//! which components survive (see this module's tests for a fully
//! hand-verified case: an isolated single-pixel peak sitting at the last
//! flat index used to survive `area_opening` even though the same peak
//! elsewhere in the same image is correctly removed).
//!
//! Fix PR InsightSoftwareConsortium/ITK#6581 (item B19 of #6575) widens the
//! sort to cover the full buffer: `std::stable_sort(m_SortPixels.get(),
//! m_SortPixels.get() + buffsize, ...)`. This port matches that fix by
//! sorting the whole `sort_pixels` array rather than `[..total - 1]`. Once
//! every pixel is sorted by value, the sweep's own invariant -- a neighbor
//! is only looked up via `find_root` once `kind.compare` (or the tie-break)
//! proves it sorts no later than the current pixel, which is exactly when
//! it has already been swept -- holds for *every* flat index, so
//! `find_root` never observes an `INACTIVE` (`-1`) node in practice; the
//! `parent[x] < 0` conflation in `find_root` is dead code once the sort is
//! correct, not a latent defect of its own.
//!
//! `UseImageSpacing` scales each pixel's starting attribute contribution by
//! the product of [`Image::spacing`], matching `psize = Π spacing[i]`; when
//! `false`, each pixel simply contributes `1.0`. `FullyConnected` selects
//! `Half::Full` neighbor connectivity, exactly as `SetupOffsetVec`'s
//! `setConnectivity(&It, m_FullyConnected)` does. Output pixel type always
//! matches the input (`AreaOpeningImageFilter.yaml`/
//! `AreaClosingImageFilter.yaml` have no `output_pixel_type`).

use crate::core::Image;
use crate::filters::error::Result;
use crate::filters::image_from_f64;
use crate::filters::reconstruction::{Half, NeighborWalker};
use std::cmp::Ordering;

/// `TFunction` in `itkAttributeMorphologyBaseImageFilter.hxx`: `std::greater`
/// for `AreaOpeningImageFilter`, `std::less` for `AreaClosingImageFilter`.
#[derive(Clone, Copy)]
enum AttributeKind {
    Opening,
    Closing,
}

impl AttributeKind {
    /// `compare(a, b)`: `true` when `a` should sweep strictly before `b`.
    fn compare(self, a: f64, b: f64) -> bool {
        match self {
            AttributeKind::Opening => a > b,
            AttributeKind::Closing => a < b,
        }
    }
}

/// `m_Parent[x] = ACTIVE` and `m_AuxData[x] = attribute_value_per_pixel`.
fn make_set(x: usize, attribute_value_per_pixel: f64, parent: &mut [i64], aux_data: &mut [f64]) {
    parent[x] = -2; // ACTIVE
    aux_data[x] = attribute_value_per_pixel;
}

/// `FindRoot`, iterative with full path compression. `parent[x] < 0` covers
/// both `INACTIVE` (`-1`, never swept) and `ACTIVE` (`-2`, a genuine root) --
/// upstream's `FindRoot` does not distinguish them either, but with the
/// sort fixed (see the module docs) the sweep never calls this on an
/// `INACTIVE` node, so the conflation is unreachable in practice.
fn find_root(parent: &mut [i64], x: usize) -> usize {
    let mut root = x;
    while parent[root] >= 0 {
        root = parent[root] as usize;
    }
    let mut cur = x;
    while parent[cur] >= 0 && parent[cur] as usize != root {
        let next = parent[cur] as usize;
        parent[cur] = root as i64;
        cur = next;
    }
    root
}

/// `MakeSet`/`FindRoot`/`Criterion`/`Union` and the resolving pass, run over
/// `vals` in flat raster order. See the module docs for the upstream
/// off-by-one this fixes.
fn attribute_morphology(
    vals: &[f64],
    size: &[usize],
    fully_connected: bool,
    lambda: f64,
    attribute_value_per_pixel: f64,
    kind: AttributeKind,
) -> Vec<f64> {
    let total = vals.len();
    if total == 0 {
        return Vec::new();
    }

    let mut sort_pixels: Vec<usize> = (0..total).collect();
    sort_pixels.sort_by(|&a, &b| {
        if kind.compare(vals[a], vals[b]) {
            Ordering::Less
        } else if kind.compare(vals[b], vals[a]) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });

    let mut parent = vec![-1i64; total]; // INACTIVE
    let mut aux_data = vec![-1.0f64; total]; // "invalid value"

    make_set(
        sort_pixels[0],
        attribute_value_per_pixel,
        &mut parent,
        &mut aux_data,
    );

    let mut walker = NeighborWalker::new(size, fully_connected, Half::Full);

    for &this_pos in &sort_pixels[1..] {
        let this_pix = vals[this_pos];
        make_set(
            this_pos,
            attribute_value_per_pixel,
            &mut parent,
            &mut aux_data,
        );

        for &neigh in walker.at(this_pos, size) {
            let neigh_pix = vals[neigh];
            if kind.compare(neigh_pix, this_pix) || (this_pix == neigh_pix && neigh < this_pos) {
                let r = find_root(&mut parent, neigh);
                if r != this_pos {
                    let criterion = vals[r] == vals[this_pos] || aux_data[r] < lambda;
                    if criterion {
                        aux_data[this_pos] += aux_data[r];
                        parent[r] = this_pos as i64;
                    } else {
                        aux_data[this_pos] = lambda;
                    }
                }
            }
        }
    }

    let mut resolved = vals.to_vec();
    for pos in (0..total).rev() {
        let r_pos = sort_pixels[pos];
        if parent[r_pos] >= 0 {
            resolved[r_pos] = resolved[parent[r_pos] as usize];
        }
    }
    resolved
}

fn attribute_morphology_image(
    image: &Image,
    lambda: f64,
    use_image_spacing: bool,
    fully_connected: bool,
    kind: AttributeKind,
) -> Result<Image> {
    if lambda <= 0.0 {
        return crate::filters::cast(image, image.pixel_id());
    }

    let size = image.size();
    let vals = image.to_f64_vec()?;
    let attribute_value_per_pixel = if use_image_spacing {
        image.spacing().iter().product()
    } else {
        1.0
    };

    let out = attribute_morphology(
        &vals,
        size,
        fully_connected,
        lambda,
        attribute_value_per_pixel,
        kind,
    );
    image_from_f64(image.pixel_id(), size, image, &out)
}

/// `AreaOpeningImageFilter`: trims blobs brighter than their surroundings
/// (`std::greater`) whose accumulated attribute (pixel count, or physical
/// area with `use_image_spacing`) is below `lambda` down to their
/// surrounding gray level; blobs at or above `lambda` are left unchanged.
/// `lambda <= 0.0` -- SimpleITK's default -- is a documented fast path that
/// leaves the image completely unchanged (see the module docs).
pub fn area_opening(
    image: &Image,
    lambda: f64,
    use_image_spacing: bool,
    fully_connected: bool,
) -> Result<Image> {
    attribute_morphology_image(
        image,
        lambda,
        use_image_spacing,
        fully_connected,
        AttributeKind::Opening,
    )
}

/// `AreaClosingImageFilter`: the dual of [`area_opening`] (`std::less`) --
/// fills valleys darker than their surroundings whose accumulated attribute
/// is below `lambda`.
pub fn area_closing(
    image: &Image,
    lambda: f64,
    use_image_spacing: bool,
    fully_connected: bool,
) -> Result<Image> {
    attribute_morphology_image(
        image,
        lambda,
        use_image_spacing,
        fully_connected,
        AttributeKind::Closing,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_i32(size: &[usize], data: Vec<i32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- area_opening ----

    /// Hand-derived two-blob fixture: an area-1 peak (index 3) and an
    /// area-3 blob (indices 7..=9) separated by background. `Lambda = 2`
    /// (pixel-count attribute, no spacing) removes the area-1 peak (its
    /// accumulated attribute 1 < 2) and keeps the area-3 blob (3 >= 2).
    #[test]
    fn area_opening_removes_small_peak_keeps_large_blob() {
        #[rustfmt::skip]
        let image = img_i32(&[13, 1], vec![
            0, 0, 0, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0,
        ]);
        let out = area_opening(&image, 2.0, false, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0,
        ]);
    }

    /// Same fixture, `Lambda = 5.5`: plain pixel counting (1 and 3) puts
    /// both blobs under threshold, collapsing the whole image to
    /// background. With `use_image_spacing` and `spacing = [2.0, 1.0]`
    /// each pixel instead contributes `2.0` (`psize = 2.0 * 1.0`), so the
    /// area-3 blob's accumulated attribute (`3 * 2.0 = 6.0`) clears 5.5
    /// while the area-1 peak's (`2.0`) still does not -- same `lambda`,
    /// different survivor, pinning that `UseImageSpacing` scales the
    /// threshold rather than merely being a cosmetic flag.
    #[test]
    fn area_opening_use_image_spacing_scales_the_area_threshold() {
        #[rustfmt::skip]
        let data = vec![
            0, 0, 0, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0,
        ];
        let mut no_spacing = img_i32(&[13, 1], data.clone());
        let unscaled = area_opening(&no_spacing, 5.5, false, false).unwrap();
        assert_eq!(unscaled.scalar_slice::<i32>().unwrap(), &[0i32; 13]);

        no_spacing.set_spacing(&[2.0, 1.0]).unwrap();
        let scaled = area_opening(&no_spacing, 5.5, true, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(scaled.scalar_slice::<i32>().unwrap(), &[
            0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0,
        ]);
    }

    /// Two single-pixel peaks that touch only diagonally: under face
    /// connectivity each is an isolated area-1 component (both removed by
    /// `Lambda = 2`); under full connectivity they merge into one area-2
    /// component at the moment they're compared (both swept before any
    /// background pixel), which meets `Lambda = 2` and survives whole.
    #[test]
    fn area_opening_fully_connected_merges_diagonal_peaks() {
        #[rustfmt::skip]
        let image = img_i32(&[3, 3], vec![
            5, 0, 0,
            0, 5, 0,
            0, 0, 0,
        ]);

        let face = area_opening(&image, 2.0, false, false).unwrap();
        assert_eq!(face.scalar_slice::<i32>().unwrap(), &[0i32; 9]);

        let full = area_opening(&image, 2.0, false, true).unwrap();
        #[rustfmt::skip]
        assert_eq!(full.scalar_slice::<i32>().unwrap(), &[
            5, 0, 0,
            0, 5, 0,
            0, 0, 0,
        ]);
    }

    /// `Lambda <= 0.0` (SimpleITK's default) is a fast path that leaves the
    /// image completely unchanged, even though every blob here is well
    /// under any positive area threshold that would otherwise remove them.
    #[test]
    fn area_opening_non_positive_lambda_is_identity() {
        #[rustfmt::skip]
        let image = img_i32(&[13, 1], vec![
            0, 0, 0, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0,
        ]);
        let out = area_opening(&image, 0.0, false, false).unwrap();
        assert_eq!(
            out.scalar_slice::<i32>().unwrap(),
            image.scalar_slice::<i32>().unwrap()
        );
    }

    /// Fixed upstream bug (see module docs, ledger §1.19): the sort now
    /// covers the whole `sort_pixels` array, so the last raster pixel is
    /// swept in its correct value order like every other pixel. An isolated
    /// area-1 peak placed at the image's last flat index therefore merges
    /// into its background neighbor at that neighbor's own turn, exactly as
    /// the identical peak swept in the middle of an array does (previous
    /// test) -- both are removed by `Lambda = 2`.
    #[test]
    fn area_opening_removes_a_small_peak_at_the_last_raster_pixel_too() {
        let image = img_i32(&[3, 1], vec![0, 0, 5]);
        let out = area_opening(&image, 2.0, false, false).unwrap();
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[0, 0, 0]);
    }

    // ---- area_closing ----

    /// Dual of `area_opening_removes_small_peak_keeps_large_blob`: an
    /// area-1 valley (index 3) filled in, an area-3 valley (indices 7..=9)
    /// left alone.
    #[test]
    fn area_closing_fills_small_valley_keeps_large_valley() {
        #[rustfmt::skip]
        let image = img_i32(&[13, 1], vec![
            5, 5, 5, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5,
        ]);
        let out = area_closing(&image, 2.0, false, false).unwrap();
        #[rustfmt::skip]
        assert_eq!(out.scalar_slice::<i32>().unwrap(), &[
            5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 5, 5, 5,
        ]);
    }

    /// `Lambda <= 0.0` is a no-op for closing too.
    #[test]
    fn area_closing_non_positive_lambda_is_identity() {
        #[rustfmt::skip]
        let image = img_i32(&[13, 1], vec![
            5, 5, 5, 0, 5, 5, 5, 0, 0, 0, 5, 5, 5,
        ]);
        let out = area_closing(&image, 0.0, false, false).unwrap();
        assert_eq!(
            out.scalar_slice::<i32>().unwrap(),
            image.scalar_slice::<i32>().unwrap()
        );
    }
}
