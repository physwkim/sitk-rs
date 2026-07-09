//! Toboggan (gradient-descent) over-segmentation.
//!
//! Port of `itk::TobogganImageFilter`
//! (`Modules/Segmentation/Watersheds/include/itkTobogganImageFilter.h` /
//! `.hxx`).
//!
//! Every pixel slides down a steepest-descent path to a local minimum; each
//! newly discovered minimum gets a fresh label, and the whole trajectory that
//! reached it inherits that label. SimpleITK's `TobogganImageFilter.yaml`
//! declares `output_pixel_type: uint32_t` and no members, so [`toboggan`]
//! takes the input image and nothing else.
//!
//! ## The algorithm, exactly as `GenerateData` writes it
//!
//! The output buffer doubles as the label map *and* the visit bookkeeping,
//! with three meanings per value:
//!
//! - `0` — unlabeled, never touched;
//! - `1` — touched by the slide/flood currently in progress (a scratch mark);
//! - `> 1` — a final region label. `CurrentLabel` starts at **2** and
//!   increments once per newly discovered minimum, so the first region label
//!   is `2`, not `1`. The value `1` is never a final label because every
//!   scratch mark is overwritten before the outer scan moves on.
//!
//! The outer scan is raster order over the flat buffer (`ImageRegionConstIterator`
//! on the output, first index fastest). The first unlabeled pixel encountered
//! in that order starts the first region, so **label numbering is
//! first-encountered-in-raster-order**.
//!
//! Per unlabeled pixel `p`:
//!
//! 1. **Slide.** `MinimumNeighborValue` is seeded once with `input[p]` and is
//!    only ever lowered — it is *not* reset per step, so the descent is
//!    strictly monotone in value. Each step marks the current position `1`,
//!    then scans the **face-connected** neighbors (`2 * dim` axis-aligned unit
//!    steps; the yaml's brief calls the result "a 4 connected labeled map"),
//!    skipping any neighbor already marked `1` (that is the path itself, and
//!    skipping it is what prevents a cycle). A neighbor replaces the running
//!    minimum only on a **strict** `<`, so ties keep the earlier candidate:
//!    the visit order is dimension `0`, `1`, … and within a dimension
//!    `+1` **before** `-1` (`for (int t = 1; t >= -1; t = t - 2)`). Hence on a
//!    plateau, and among equal-valued neighbors generally, the first strictly
//!    smaller neighbor in that order wins and no later equal value displaces
//!    it.
//!
//!    The slide stops when no neighbor is strictly smaller (a local minimum,
//!    whose class is the scratch `1`) or when the step lands on a pixel whose
//!    class is already `> 1` (it slid into an existing region — that pixel is
//!    appended to the visited list but is *not* marked `1`).
//!
//! 2. **Flood** (only when the slide ended on a scratch-marked local minimum).
//!    A LIFO flood from the minimum absorbs every unlabeled neighbor whose
//!    input value is `<= ` the popped seed's value, marking it `1` and adding
//!    it to the visited list. Whenever the flood touches a neighbor whose
//!    class is `> 1` (again gated on `<=`), `MinimumNeighborClass` is
//!    overwritten with it — last write wins, in the flood's own traversal
//!    order, which visits `-1` **before** `+1` per dimension (the reverse of
//!    the slide's order: `for (int t = -1; t <= 1; t = t + 2)`).
//!
//! 3. **Label.** If the class is still `1`, this was a brand-new minimum:
//!    take `CurrentLabel` and increment it. If it is `> 1`, the region merged
//!    into an existing one and takes that label. Then every visited index is
//!    written with the chosen label.
//!
//! ## Upstream quirks reproduced verbatim
//!
//! - **The flood is `<=`, not `==`.** The `.hxx` comment says "Connect any
//!   pixels having the same value as the minimum we found", but the test is
//!   `inputImage->GetPixel(NeighborIndex) <= SeedValue`. The seed of the
//!   flood is a local minimum *among its own non-scratch neighbors*, so its
//!   direct neighbors can only be equal — but a pixel absorbed at equal value
//!   can itself have a strictly smaller neighbor that the slide never
//!   examined, and the flood then descends into it. This port keeps `<=`.
//!
//! - **The flood re-pushes its seed.** `CurrentPositionIndex` is already in
//!   `Visited` when the flood starts, and the flood's first pop pushes it
//!   again. Harmless (the final relabel loop is idempotent), and reproduced
//!   rather than deduplicated.
//!
//! - **`CurrentPositionIndex = MinimumNeighborIndex`** in the merge branch of
//!   the `.hxx` is dead — nothing reads it afterwards — and has no analogue
//!   here.
//!
//! Labels are handed out as `u32` (`OutputImagePixelType`), and the output
//! carries the input's geometry.

use crate::error::Result;
use sitk_core::Image;

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// The face-connected neighbor of `flat` one step along axis `d` in direction
/// `step` (`+1` or `-1`), or `None` when that step leaves the image —
/// `outputImage->GetRequestedRegion().IsInside(NeighborIndex)`.
fn neighbor(flat: usize, d: usize, step: i8, size: &[usize], strides: &[usize]) -> Option<usize> {
    let coord = (flat / strides[d]) % size[d];
    if step > 0 {
        (coord + 1 < size[d]).then(|| flat + strides[d])
    } else {
        (coord > 0).then(|| flat - strides[d])
    }
}

/// `TobogganImageFilter`: label each pixel with the local minimum its
/// steepest-descent path reaches.
///
/// The output is a `UInt32` label image with the input's geometry. Labels
/// start at `2` and are assigned in raster order of the first pixel that
/// discovers each minimum; `0` and `1` never survive in the output (see the
/// module docs for why `1` is reserved as a scratch mark). An image with a
/// single minimum comes out uniformly labeled `2`.
///
/// Neighbors are face-connected only (4-connected in 2-D, 6 in 3-D).
pub fn toboggan(image: &Image) -> Result<Image> {
    let size = image.size();
    let dim = size.len();
    let total: usize = size.iter().product();
    let strides = strides(size);
    let vals = image.to_f64_vec();

    // "Zero the output" — the buffer is the label map and the scratch marks.
    let mut out = vec![0u32; total];
    let mut current_label: u32 = 2;

    for p in 0..total {
        if out[p] != 0 {
            continue;
        }

        // Seeded once, outside the descent loop, and only ever lowered.
        let mut minimum_neighbor_value = vals[p];
        let mut current = p;
        let mut visited = vec![p];
        let mut minimum_neighbor_class;

        // Search along a steepest descent path to a local minimum.
        loop {
            out[current] = 1;
            let mut minimum_neighbor_index = current;
            for d in 0..dim {
                // `for (int t = 1; t >= -1; t = t - 2)`: +1 before -1.
                for step in [1i8, -1] {
                    let Some(n) = neighbor(current, d, step, size, &strides) else {
                        continue;
                    };
                    // Class 1 is the path we are currently walking; ignore it.
                    if out[n] != 1 && vals[n] < minimum_neighbor_value {
                        minimum_neighbor_value = vals[n];
                        minimum_neighbor_index = n;
                    }
                }
            }

            let mut found_minimum = false;
            if minimum_neighbor_index != current {
                visited.push(minimum_neighbor_index);
                current = minimum_neighbor_index;
            } else {
                found_minimum = true;
            }
            minimum_neighbor_class = out[minimum_neighbor_index];
            // We slid into a different class.
            if minimum_neighbor_class > 1 {
                found_minimum = true;
            }
            if found_minimum {
                break;
            }
        }

        if minimum_neighbor_class == 1 {
            // Flood fill from the minimum, connecting pixels whose value is
            // `<=` the popped seed's (see the module docs on this `<=`).
            let mut open = vec![current];
            while let Some(seed) = open.pop() {
                visited.push(seed);
                let seed_value = vals[seed];
                for d in 0..dim {
                    // `for (int t = -1; t <= 1; t = t + 2)`: -1 before +1.
                    for step in [-1i8, 1] {
                        let Some(n) = neighbor(seed, d, step, size, &strides) else {
                            continue;
                        };
                        if vals[n] <= seed_value {
                            let neighbor_class = out[n];
                            if neighbor_class == 0 {
                                open.push(n);
                                out[n] = 1;
                            }
                            if neighbor_class > 1 {
                                minimum_neighbor_class = neighbor_class;
                            }
                        }
                    }
                }
            }
        }

        // MinimumNeighborClass is always >= 1 here.
        let label_for_region = if minimum_neighbor_class == 1 {
            let label = current_label;
            current_label += 1;
            label
        } else {
            // Bumped into another region: equivalent to finding its minimum.
            minimum_neighbor_class
        };

        for &v in &visited {
            out[v] = label_for_region;
        }
    }

    let mut result = Image::from_vec(size, out)?;
    result.copy_geometry_from(image);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(size: &[usize], data: Vec<f32>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn labels(image: &Image) -> Vec<u32> {
        image.scalar_slice::<u32>().unwrap().to_vec()
    }

    /// Output pixel type is `uint32_t` per the yaml.
    #[test]
    fn output_is_uint32_with_input_geometry() {
        let mut input = img(&[3, 1], vec![1.0, 0.0, 1.0]);
        input.set_spacing(&[2.0, 3.0]).unwrap();
        let out = toboggan(&input).unwrap();
        assert_eq!(out.pixel_id(), sitk_core::PixelId::UInt32);
        assert_eq!(out.spacing(), &[2.0, 3.0]);
    }

    /// A single V-shaped basin: every pixel slides to index 3, one label, and
    /// the first label handed out is 2 (not 1, not 0).
    #[test]
    fn single_minimum_is_one_region_labeled_two() {
        let out = toboggan(&img(&[7, 1], vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0]));
        assert_eq!(labels(&out.unwrap()), vec![2; 7]);
    }

    /// Two basins separated by a ridge at index 4. Raster order labels the
    /// left basin 2 and the right basin 3; the ridge pixel joins the right
    /// basin because the slide checks `+1` before `-1`. Full trace inline.
    #[test]
    fn two_basins_1d_hand_derived() {
        // values: 3 2 1 2 3 2 1 2 3   (minima at 2 and 6, ridge at 4)
        let out = toboggan(&img(
            &[9, 1],
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0],
        ));
        // p=0: slide 0(3) -> 1(2) -> 2(1); minimum at 2, new label 2.
        //      flood from 2 absorbs nothing (both neighbors have value 2 > 1).
        //      visited = {0,1,2,2} -> label 2.
        // p=3: neighbors of 3 are 4(3) and 2(label 2). step +1 first: 4 has
        //      value 3 > 2, no. step -1: 2 has class 2 (!= 1) and value 1 < 2,
        //      so it becomes the minimum. Step onto 2, class 2 > 1 -> merge.
        //      visited = {3,2} -> label 2.
        // p=4: +1 -> 5 (value 2 < 3) wins; step to 5. From 5: +1 -> 6 (value
        //      1 < 2) wins; step to 6. From 6: neighbors 7(2) and 5(class 1,
        //      skipped); no strict decrease -> local minimum, class 1.
        //      New label 3. visited = {4,5,6,6} -> label 3.
        // p=7: +1 -> 8 (3 > 2) no; -1 -> 6 (value 1 < 2, class 3) -> merge 3.
        // p=8: +1 out; -1 -> 7 (value 2 < 3, class 3) -> merge 3.
        assert_eq!(labels(&out.unwrap()), vec![2, 2, 2, 2, 3, 3, 3, 3, 3]);
    }

    /// The ridge pixel of an *even*-width symmetric double basin: the ridge
    /// sits between two equal-valued neighbors and the slide's `+1`-before-`-1`
    /// order plus the strict `<` sends it to the **higher** index, i.e. the
    /// right basin. This pins the tie-break direction.
    #[test]
    fn plateau_tie_break_prefers_the_positive_step() {
        // p=1 is the ridge; both neighbors (0 and 2) have value 0.
        // The slide checks +1 (index 2) first: 0 < 1, so index 2 wins and
        // index 0 (equal value, checked second) does not displace it.
        let out = toboggan(&img(&[3, 1], vec![0.0, 1.0, 0.0]));
        // p=0: local minimum immediately (neighbor 1 has value 1 > 0), new
        //      label 2; flood from 0: neighbor 1 has value 1 > 0, not
        //      absorbed. -> {0} = 2.
        // p=1: +1 -> 2 (0 < 1) wins; -1 -> 0 (0 < 0 is false) does not.
        //      Step onto 2, class 0 -> continue. From 2: -1 neighbor is 1
        //      (class 1, skipped); no strict decrease -> minimum, class 1.
        //      New label 3, flood from 2 absorbs nothing. -> {1,2} = 3.
        assert_eq!(labels(&out.unwrap()), vec![2, 3, 3]);
    }

    /// A completely flat image: pixel 0 is an immediate local minimum, and its
    /// flood (`<=`) swallows the entire image. One label, `2`.
    #[test]
    fn flat_image_is_one_region() {
        let out = toboggan(&img(&[4, 3], vec![5.0; 12]));
        assert_eq!(labels(&out.unwrap()), vec![2; 12]);
    }

    /// 2-D, two wells with a ridge column between them. Row-major raster order
    /// discovers the left well first (label 2), then the right (label 3).
    #[test]
    fn two_basins_2d_hand_derived() {
        #[rustfmt::skip]
        let input = img(&[5, 3], vec![
            2.0, 1.0, 3.0, 1.0, 2.0,
            1.0, 0.0, 3.0, 0.0, 1.0,
            2.0, 1.0, 3.0, 1.0, 2.0,
        ]);
        let out = labels(&toboggan(&input).unwrap());
        // Column 2 (indices 2, 7, 12) is the ridge, value 3 everywhere.
        // Left well minimum is index 6 (value 0), right well is index 8.
        // Raster scan hits index 0 first -> left well gets label 2.
        // Index 3 (value 1, row 0) is the first pixel that descends into the
        // right well -> label 3.
        // Ridge pixels descend into whichever side the +1-first order picks:
        // index 2 checks +1 (index 3, value 1 < 3) before -1, so it joins the
        // right basin; likewise 7 and 12.
        #[rustfmt::skip]
        assert_eq!(out, vec![
            2, 2, 3, 3, 3,
            2, 2, 3, 3, 3,
            2, 2, 3, 3, 3,
        ]);
    }

    /// Labels are assigned first-encountered-in-raster-order: mirroring the
    /// image horizontally swaps which basin is found first, hence swaps 2 and
    /// 3.
    #[test]
    fn label_numbering_follows_raster_order() {
        let a = labels(&toboggan(&img(&[5, 1], vec![0.0, 1.0, 2.0, 1.0, 0.5])).unwrap());
        let b = labels(&toboggan(&img(&[5, 1], vec![0.5, 1.0, 2.0, 1.0, 0.0])).unwrap());
        // `a`: index 0 is the deepest-left minimum -> label 2; index 4 -> 3.
        assert_eq!(a, vec![2, 2, 3, 3, 3]);
        // `b`: index 0 is still visited first, so it still gets label 2 even
        // though its minimum (0.5) is the shallower one.
        assert_eq!(b, vec![2, 2, 3, 3, 3]);
    }

    /// Zero-pixel image: no regions, empty output, no panic.
    #[test]
    fn empty_image() {
        let out = toboggan(&img(&[0, 0], vec![])).unwrap();
        assert_eq!(labels(&out), Vec::<u32>::new());
    }

    /// The flood's `<=` (not `==`) lets it descend below the slide's minimum.
    /// Pinning the quirk: pixel 0 (value 1) slides to pixel 1 (value 0),
    /// whose flood absorbs pixel 2 (value 0, equal) and then, from pixel 2,
    /// pixel 3 (value -1, strictly smaller than pixel 2's value).
    #[test]
    fn flood_descends_below_the_minimum() {
        let out = labels(&toboggan(&img(&[5, 1], vec![1.0, 0.0, 0.0, -1.0, 5.0])).unwrap());
        // Slide from 0: +1 -> 1 (0 < 1) wins. From 1: +1 -> 2 (0 < 0 false);
        // -1 -> 0 (class 1, skipped). Local minimum at 1, class 1.
        // Flood: pop 1 (value 0) -> neighbor 0 (value 1 > 0, no), neighbor 2
        // (value 0 <= 0, class 0) absorbed. pop 2 (value 0) -> neighbor 1
        // (class 1, but `<=` holds and class is 1, not 0 and not > 1: no-op),
        // neighbor 3 (value -1 <= 0, class 0) absorbed. pop 3 (value -1) ->
        // neighbor 2 (0 <= -1 false), neighbor 4 (5 <= -1 false).
        // visited = {0,1,1,2,3} -> label 2. Pixel 4 then slides into 3.
        assert_eq!(out, vec![2, 2, 2, 2, 2]);
    }
}
