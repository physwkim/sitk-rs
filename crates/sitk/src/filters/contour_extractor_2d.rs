//! `ContourExtractor2DImageFilter`: marching-squares iso-line extraction from a
//! 2-D image, verified against
//! `Modules/Filtering/Path/include/itkContourExtractor2DImageFilter.h`/`.hxx`.
//!
//! Unlike every other filter in this crate the output is not an image: it is a
//! list of polylines ([`Contour`]). The coordinates are **continuous indices**,
//! not physical points — ITK's output is a `PolyLineParametricPath<2>`, whose
//! `VertexType` is `ContinuousIndex<double, 2>`, and SimpleITK hands that vertex
//! list straight to the caller (`ContourExtractor2DImageFilter.yaml`'s
//! `itk_get: f->GetOutput(n)->GetVertexList()` under `no_return_image: true`).
//! Nothing in either wrapper applies the image's origin/spacing/direction.
//!
//! ## The algorithm, stage by stage
//!
//! A 2x2 square is associated with the pixel at its top-left corner, and the
//! top-left corner sweeps the image in raster order (x fastest), skipping the
//! bottom row and right column so that every square is whole. The four corners
//!
//! ```text
//! v0 v1
//! v2 v3
//! ```
//!
//! are each classified "high" or "low" and packed into
//! `square_case = v0 + 2·v1 + 4·v2 + 8·v3`, which selects one of sixteen
//! configurations. "High" means `> contour_value` normally, and `== label` in
//! `label_contours` mode. Each configuration contributes zero, one, or (the two
//! saddles) two directed segments, whose endpoints sit on the square's sides at
//! the linearly interpolated crossing `t = (contour_value − from) / (to − from)`
//! — fixed at `t = 0.5` in `label_contours` mode, where the values are labels
//! and interpolating between them is meaningless.
//!
//! Segments are stitched by [`ContourData::add_segment`], which keeps two hash
//! maps from vertex to growing contour: one keyed by each contour's first
//! vertex, one by its last. A new segment `from → to` therefore either starts a
//! contour, extends one at either end, joins two distinct contours, or — when
//! the contour it would extend at both ends is the *same* contour — closes it by
//! repeating the first vertex at the end. **Endpoint matching is exact `f64`
//! equality**, mirroring ITK's `std::unordered_map<VertexType, …, VertexHash>`
//! keyed on `ContinuousIndex`'s exact `operator==`. This works because every
//! vertex is produced by the same expression `from_index + t · to_offset`
//! evaluated identically by both squares that share a side, so the two bit
//! patterns agree bit for bit. It is not a tolerance-based merge, and this port
//! does not make it one.
//!
//! When two distinct contours are joined, the one with the **lower** creation
//! number survives and keeps its position in the master list; the newer one is
//! spliced into it and deleted. That is what makes the returned order stable
//! (contours come back in creation order, which is the raster order of the
//! square that first opened them), and it is why this port stores contours in a
//! creation-indexed `Vec<Option<…>>` rather than a `Vec` it would have to
//! re-index on every merge.
//!
//! ## The saddles
//!
//! `square_case` 6 (`v1`, `v2` high) and 9 (`v0`, `v3` high) are the ambiguous
//! ones: the two high corners touch only at the square's centre. By default the
//! *low* pixels are vertex-connected, i.e. the two segments are placed so the
//! low-valued diagonal stays one region and the high-valued pixels are split.
//! `vertex_connect_high_pixels` swaps which diagonal is joined. Under
//! `label_contours`, "high" means "is this label" and every label is traced
//! separately, so the default leaves all four pixels in separate contours.
//!
//! ## Orientation
//!
//! Segments are emitted so that, walking from a segment's tail to its head,
//! values below `contour_value` are on the left and values above are on the
//! right (`.hxx`: "recall that we draw the lines so that (moving from tail to
//! head) the lower-valued pixels are on the left of the line"). In an index
//! frame with `y` increasing downward, that makes contours run clockwise around
//! bright blobs. `reverse_contour_orientation` reverses each emitted point list,
//! which flips this.
//!
//! ## Closedness
//!
//! A contour that runs into the image border cannot close, and is returned open;
//! everything else closes. ITK exposes no flag for this and documents "test
//! whether the beginning point is the same as the end point"; [`Contour`] does
//! exactly that test, so [`Contour::is_closed`] is derived, not independent
//! information.
//!
//! ## Degenerate arcs
//!
//! `AddSegment` drops any segment whose two endpoints are equal. That happens
//! exactly when a square has one corner sitting *on* `contour_value` and the
//! other three strictly above it: both interpolations land on that corner, and
//! the arc has zero length. Such a square contributes nothing at all.
//!
//! ## Deviations from upstream
//!
//! * **`RequestedRegion` is not exposed.** ITK lets a caller restrict the sweep
//!   with `SetRequestedRegion`, and `GenerateDataForLabels` copies the region
//!   into a scratch image so the constant boundary applies at the region edge.
//!   SimpleITK's yaml declares no such member, so this port always sweeps the
//!   whole image, which is ITK's own default (`m_UseCustomRegion == false`).
//! * **`m_UnusedLabel` is not materialized.** In `label_contours` mode ITK picks
//!   the smallest pixel value absent from the image and uses it as the
//!   out-of-image constant, purely so that out-of-image samples compare unequal
//!   to every real label. This port samples out-of-image as "not this label"
//!   directly, which is the same thing. The one *observable* consequence of the
//!   search is its failure mode — if the image's labels exhaust the pixel type's
//!   entire value range there is no unused value and ITK throws — and that is
//!   reproduced as [`FilterError::ContourExtractorNoUnusedLabel`].
//! * `contour_value` is `static_cast` to the input pixel type before use
//!   (`pixeltype: Input` in the yaml), so `0.5` on an integer image is `0`.

use std::collections::HashMap;
use std::collections::VecDeque;

use crate::core::{Image, PixelId, Scalar, dispatch_scalar};
use crate::filters::error::{FilterError, Result};
use crate::filters::quantize_to_pixel_type;

/// One extracted iso-line, as a `PolyLineParametricPath<2>`'s vertex list.
#[derive(Clone, Debug, PartialEq)]
pub struct Contour {
    /// Continuous-index coordinates `[x, y]`, in path order. A closed contour
    /// repeats its first point as its last.
    pub points: Vec<[f64; 2]>,
    /// `points.first() == points.last()`, the test ITK's class docs prescribe.
    pub is_closed: bool,
}

type Vertex = [f64; 2];

/// Exact-equality hash key for a vertex. `f64` is not `Hash`/`Eq`, so the bit
/// patterns stand in — the same exact-equality semantics as ITK's
/// `std::unordered_map<ContinuousIndex<double, 2>, …>`. `-0.0` is folded onto
/// `0.0` so that the key agrees with `==`, as `std::hash<double>` does. (No
/// vertex this filter builds is actually `-0.0`: `from_index + t·to_offset`
/// adds `±0.0` to an integer, and IEEE `0.0 + -0.0` is `+0.0`. The fold makes
/// the key's contract independent of that argument.)
type VertexKey = [u64; 2];

fn vertex_key(v: Vertex) -> VertexKey {
    [
        (if v[0] == 0.0 { 0.0 } else { v[0] }).to_bits(),
        (if v[1] == 0.0 { 0.0 } else { v[1] }).to_bits(),
    ]
}

/// `ContourExtractor2DImageFilter::ContourData`: the growing contours plus the
/// two endpoint indexes into them.
///
/// `contours[i]` is the contour whose `m_ContourNumber` is `i`, or `None` once
/// it has been merged away. ITK keeps a `std::list` in creation order and
/// erases merged nodes; iterating this `Vec`'s live entries in index order
/// reproduces that list exactly, because merges always delete the
/// *higher*-numbered contour.
#[derive(Default)]
struct ContourData {
    contours: Vec<Option<VecDeque<Vertex>>>,
    starts: HashMap<VertexKey, usize>,
    ends: HashMap<VertexKey, usize>,
}

impl ContourData {
    fn get(&mut self, i: usize) -> &mut VecDeque<Vertex> {
        self.contours[i]
            .as_mut()
            .expect("endpoint maps only ever name live contours")
    }

    /// `ContourExtractor2DImageFilter::AddSegment`, branch for branch.
    fn add_segment(&mut self, from: Vertex, to: Vertex) {
        if from == to {
            // Degenerate arc: a square with exactly one corner *on* the contour
            // value and the rest above it. The point is picked up by the
            // neighbouring squares.
            return;
        }

        let new_tail = self.starts.get(&vertex_key(to)).copied();
        let new_head = self.ends.get(&vertex_key(from)).copied();

        match (new_tail, new_head) {
            // The arc joins a contour that starts at `to` with one that ends at
            // `from`.
            (Some(tail), Some(head)) => {
                if head == tail {
                    // Closing a loop: repeat the first vertex at the end, and
                    // retire both endpoints.
                    self.get(head).push_back(to);
                    self.starts.remove(&vertex_key(to));
                    self.ends.remove(&vertex_key(from));
                } else if tail > head {
                    // `tail` is the newer contour: copy it onto the end of
                    // `head` and delete it.
                    let tail_contour = self.contours[tail].take().expect("live contour");
                    let tail_back = *tail_contour.back().expect("contours hold >= 2 vertices");
                    self.get(head).extend(tail_contour);

                    self.starts.remove(&vertex_key(to));
                    self.ends.remove(&vertex_key(tail_back));
                    self.ends.remove(&vertex_key(from));
                    let head_back = *self.get(head).back().expect("just extended");
                    self.ends.insert(vertex_key(head_back), head);
                } else {
                    // `head` is the newer contour: copy it onto the front of
                    // `tail` and delete it.
                    let head_contour = self.contours[head].take().expect("live contour");
                    let head_front = *head_contour.front().expect("contours hold >= 2 vertices");
                    let tail_contour = self.get(tail);
                    for v in head_contour.iter().rev() {
                        tail_contour.push_front(*v);
                    }

                    self.ends.remove(&vertex_key(from));
                    self.starts.remove(&vertex_key(head_front));
                    self.starts.remove(&vertex_key(to));
                    let tail_front = *self.get(tail).front().expect("just prepended");
                    self.starts.insert(vertex_key(tail_front), tail);
                }
            }
            // No contour to attach to: open a new one.
            (None, None) => {
                let number = self.contours.len();
                self.contours.push(Some(VecDeque::from([from, to])));
                self.starts.insert(vertex_key(from), number);
                self.ends.insert(vertex_key(to), number);
            }
            // Prepend the arc to the contour that starts at `to`.
            (Some(tail), None) => {
                self.get(tail).push_front(from);
                self.starts.remove(&vertex_key(to));
                self.starts.insert(vertex_key(from), tail);
            }
            // Append the arc to the contour that ends at `from`.
            (None, Some(head)) => {
                self.get(head).push_back(to);
                self.ends.remove(&vertex_key(from));
                self.ends.insert(vertex_key(to), head);
            }
        }
    }

    /// The surviving contours in creation order — ITK's `m_Contours` list.
    fn into_contours(self) -> Vec<VecDeque<Vertex>> {
        self.contours.into_iter().flatten().collect()
    }
}

/// `CreateSingleContour`: sweep the 2x2 squares whose top-left corner runs over
/// `[x_first, x_last] x [y_first, y_last]` (inclusive), in raster order.
///
/// `label` selects the mode: `Some(l)` is `LabelContours` tracing label `l`
/// (samples outside the image are "not `l`", standing in for ITK's
/// `m_UnusedLabel` constant boundary), `None` compares against `contour_value`.
fn create_single_contour<T: Scalar>(
    data: &[T],
    size: [usize; 2],
    label: Option<T>,
    contour_value: f64,
    vertex_connect_high_pixels: bool,
    (x_first, x_last): (i64, i64),
    (y_first, y_last): (i64, i64),
) -> Vec<VecDeque<Vertex>> {
    let value = |x: i64, y: i64| -> f64 {
        let inside = x >= 0 && y >= 0 && (x as usize) < size[0] && (y as usize) < size[1];
        match label {
            Some(l) => {
                if inside && data[y as usize * size[0] + x as usize] == l {
                    1.0
                } else {
                    0.0
                }
            }
            None => {
                debug_assert!(inside, "value mode never samples outside the image");
                data[y as usize * size[0] + x as usize].as_f64()
            }
        }
    };
    // "High" is `== label` in label mode, `> contour_value` otherwise.
    let is_high = |v: f64| {
        if label.is_some() {
            v != 0.0
        } else {
            v > contour_value
        }
    };
    // `InterpolateContourPosition`: linear crossing between two adjacent
    // samples, or the midpoint when the values are labels.
    let interpolate =
        |from_value: f64, to_value: f64, from: [i64; 2], offset: [i64; 2]| -> Vertex {
            let t = if label.is_some() {
                0.5
            } else {
                (contour_value - from_value) / (to_value - from_value)
            };
            [
                from[0] as f64 + t * offset[0] as f64,
                from[1] as f64 + t * offset[1] as f64,
            ]
        };

    const RIGHT: [i64; 2] = [1, 0];
    const DOWN: [i64; 2] = [0, 1];

    let mut contour_data = ContourData::default();
    for y in y_first..=y_last {
        for x in x_first..=x_last {
            let (v0, v1) = (value(x, y), value(x + 1, y));
            let (v2, v3) = (value(x, y + 1), value(x + 1, y + 1));
            let square_case = u8::from(is_high(v0))
                + 2 * u8::from(is_high(v1))
                + 4 * u8::from(is_high(v2))
                + 8 * u8::from(is_high(v3));

            // The four side crossings, evaluated only where the case needs them
            // (an unused one would divide by zero).
            let top = || interpolate(v0, v1, [x, y], RIGHT);
            let bottom = || interpolate(v2, v3, [x, y + 1], RIGHT);
            let left = || interpolate(v0, v2, [x, y], DOWN);
            let right = || interpolate(v1, v3, [x + 1, y], DOWN);

            match square_case {
                0 | 15 => {}
                1 => contour_data.add_segment(top(), left()),
                2 => contour_data.add_segment(right(), top()),
                3 => contour_data.add_segment(right(), left()),
                4 => contour_data.add_segment(left(), bottom()),
                5 => contour_data.add_segment(top(), bottom()),
                6 => {
                    if vertex_connect_high_pixels {
                        contour_data.add_segment(left(), top());
                        contour_data.add_segment(right(), bottom());
                    } else {
                        contour_data.add_segment(right(), top());
                        contour_data.add_segment(left(), bottom());
                    }
                }
                7 => contour_data.add_segment(right(), bottom()),
                8 => contour_data.add_segment(bottom(), right()),
                9 => {
                    if vertex_connect_high_pixels {
                        contour_data.add_segment(top(), right());
                        contour_data.add_segment(bottom(), left());
                    } else {
                        contour_data.add_segment(top(), left());
                        contour_data.add_segment(bottom(), right());
                    }
                }
                10 => contour_data.add_segment(bottom(), top()),
                11 => contour_data.add_segment(bottom(), left()),
                12 => contour_data.add_segment(left(), right()),
                13 => contour_data.add_segment(top(), right()),
                14 => contour_data.add_segment(left(), top()),
                _ => unreachable!("square_case is four bits"),
            }
        }
    }

    contour_data.into_contours()
}

/// How many distinct values the pixel type can represent, or `None` for the
/// floating-point types, whose count is far beyond any image's pixel count.
/// Used only by [`require_unused_label`].
fn distinct_value_count(pixel_id: PixelId) -> Option<u128> {
    match pixel_id {
        PixelId::UInt8 | PixelId::Int8 | PixelId::VectorUInt8 | PixelId::VectorInt8 => Some(1 << 8),
        PixelId::UInt16 | PixelId::Int16 | PixelId::VectorUInt16 | PixelId::VectorInt16 => {
            Some(1 << 16)
        }
        PixelId::UInt32 | PixelId::Int32 | PixelId::VectorUInt32 | PixelId::VectorInt32 => {
            Some(1 << 32)
        }
        PixelId::UInt64 | PixelId::Int64 | PixelId::VectorUInt64 | PixelId::VectorInt64 => {
            Some(1 << 64)
        }
        PixelId::Float32
        | PixelId::ComplexFloat32
        | PixelId::VectorFloat32
        | PixelId::Float64
        | PixelId::ComplexFloat64
        | PixelId::VectorFloat64 => None,
    }
}

/// `GenerateDataForLabels`' `itkAssertOrThrowMacro(m_UnusedLabel !=
/// allLabels.front(), "Need at least one unused value in the space of labels")`.
///
/// ITK walks up from the pixel type's minimum looking for the first value the
/// image does not use; since `labels` is the image's *distinct* values, that
/// search fails exactly when `labels` is the whole value range.
fn require_unused_label(pixel_id: PixelId, distinct_labels: usize) -> Result<()> {
    if distinct_value_count(pixel_id).is_some_and(|n| distinct_labels as u128 == n) {
        return Err(FilterError::ContourExtractorNoUnusedLabel(pixel_id));
    }
    Ok(())
}

fn contour_extractor_2d_typed<T: Scalar>(
    image: &Image,
    contour_value: f64,
    reverse_contour_orientation: bool,
    vertex_connect_high_pixels: bool,
    label_contours: bool,
) -> Result<Vec<Contour>> {
    let size = [image.size()[0], image.size()[1]];
    let data = image.scalar_slice::<T>()?;

    let raw = if label_contours {
        label_contours_of(data, size, vertex_connect_high_pixels)?
    } else if size[0] < 2 || size[1] < 2 {
        // `shrunkSize = {size[0] - 1, size[1] - 1}` leaves no whole square.
        Vec::new()
    } else {
        create_single_contour(
            data,
            size,
            None,
            contour_value,
            vertex_connect_high_pixels,
            (0, size[0] as i64 - 2),
            (0, size[1] as i64 - 2),
        )
    };

    Ok(raw
        .into_iter()
        .map(|c| {
            let points: Vec<Vertex> = if reverse_contour_orientation {
                c.into_iter().rev().collect()
            } else {
                c.into()
            };
            Contour {
                is_closed: points.first() == points.last(),
                points,
            }
        })
        .collect())
}

/// `GenerateDataForLabels`: trace every distinct label separately, each over its
/// own bounding box grown by one pixel, and concatenate the results in ascending
/// label order.
fn label_contours_of<T: Scalar>(
    data: &[T],
    size: [usize; 2],
    vertex_connect_high_pixels: bool,
) -> Result<Vec<VecDeque<Vertex>>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut labels = data.to_vec();
    labels.sort_by(|a, b| a.partial_cmp(b).expect("NaN label"));
    labels.dedup();
    require_unused_label(T::PIXEL_ID, labels.len())?;

    // [min_x, min_y, max_x, max_y] per label, inclusive.
    let mut boxes = vec![[i64::MAX, i64::MAX, i64::MIN, i64::MIN]; labels.len()];
    for (p, v) in data.iter().enumerate() {
        let (x, y) = ((p % size[0]) as i64, (p / size[0]) as i64);
        let i = labels
            .binary_search_by(|probe| probe.partial_cmp(v).expect("NaN label"))
            .expect("every pixel value is a label");
        let b = &mut boxes[i];
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x);
        b[3] = b[3].max(y);
    }

    let mut out = Vec::new();
    for (&label, b) in labels.iter().zip(&boxes) {
        // `extendedIndex = bbox.min - 1`, `extendedSize = bbox.max - bbox.min +
        // 2`: the top-left corner sweeps `[min - 1, max]`, so the squares cover
        // `[min - 1, max + 1]` and every label pixel is fully surrounded.
        out.extend(create_single_contour(
            data,
            size,
            Some(label),
            0.0,
            vertex_connect_high_pixels,
            (b[0] - 1, b[2]),
            (b[1] - 1, b[3]),
        ));
    }
    Ok(out)
}

/// `ContourExtractor2DImageFilter`: the marching-squares iso-lines of a 2-D
/// image at `contour_value`, as continuous-index polylines. See the module docs
/// for the algorithm, the saddle rule, the orientation convention, and the two
/// places this port departs from ITK.
///
/// * `contour_value` — the iso-value, `static_cast` to the input pixel type
///   first. Ignored when `label_contours`.
/// * `reverse_contour_orientation` — reverse every returned point list.
/// * `vertex_connect_high_pixels` — in the two ambiguous saddle squares, join
///   the high-valued diagonal instead of the low-valued one.
/// * `label_contours` — treat pixel values as labels, trace each distinct label
///   separately (crossings at the midpoint, never interpolated), and return all
///   of them in ascending label order.
///
/// Errors on a non-2-D image, and — under `label_contours` — on an image whose
/// labels exhaust the pixel type's value range.
///
/// Panics if a `label_contours` run is given an image containing `NaN`, which
/// has no order and cannot be sorted into a label list; ITK's `std::sort` is
/// equally undefined there.
pub fn contour_extractor_2d(
    image: &Image,
    contour_value: f64,
    reverse_contour_orientation: bool,
    vertex_connect_high_pixels: bool,
    label_contours: bool,
) -> Result<Vec<Contour>> {
    if image.dimension() != 2 {
        return Err(FilterError::UnsupportedContourExtractorDimension(
            image.dimension(),
        ));
    }
    let contour_value = quantize_to_pixel_type(image.pixel_id(), contour_value);
    dispatch_scalar!(
        image.pixel_id(),
        contour_extractor_2d_typed,
        image,
        contour_value,
        reverse_contour_orientation,
        vertex_connect_high_pixels,
        label_contours
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_f64(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    fn img_u8(size: &[usize], data: Vec<u8>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    /// A single bright pixel at the centre of a 3x3, iso 0.5. Hand-traced
    /// through all four squares in raster order:
    ///
    /// * square (0,0) is case 8 -> bottom(0.5,1) -> right(1,0.5)  [new contour]
    /// * square (1,0) is case 4 -> left(1,0.5) -> bottom(1.5,1)   [append]
    /// * square (0,1) is case 2 -> right(1,1.5) -> top(0.5,1)     [prepend]
    /// * square (1,1) is case 1 -> top(1.5,1) -> left(1,1.5)      [close]
    ///
    /// giving one closed diamond through the four edge midpoints.
    #[test]
    fn single_bright_pixel_yields_one_closed_diamond() {
        let image = img_f64(&[3, 3], vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let contours = contour_extractor_2d(&image, 0.5, false, false, false).unwrap();
        assert_eq!(contours.len(), 1);
        assert!(contours[0].is_closed);
        assert_eq!(
            contours[0].points,
            vec![[1.0, 1.5], [0.5, 1.0], [1.0, 0.5], [1.5, 1.0], [1.0, 1.5],]
        );
    }

    /// The orientation convention, read off the diamond above: walking
    /// `(1,1.5) -> (0.5,1)` the bright pixel `(1,1)` lies to the *right* of the
    /// direction of travel (rotate the direction right by 90 degrees in this
    /// y-down index frame and it points at the pixel). Equivalently, the loop's
    /// shoelace sum is positive in a y-up reading, i.e. clockwise on screen.
    #[test]
    fn contours_circle_hills_with_the_high_side_on_the_right() {
        let image = img_f64(&[3, 3], vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let points = contour_extractor_2d(&image, 0.5, false, false, false).unwrap()[0]
            .points
            .clone();

        let [ax, ay] = points[0];
        let [bx, by] = points[1];
        let (dx, dy) = (bx - ax, by - ay);
        // Right of the travel direction, in a y-down frame.
        let (rx, ry) = (-dy, dx);
        let (mx, my) = ((ax + bx) / 2.0, (ay + by) / 2.0);
        // The bright pixel sits at index (1,1); it should lie along +right.
        let (tx, ty) = (1.0 - mx, 1.0 - my);
        assert!(rx * tx + ry * ty > 0.0, "high side is not on the right");

        let area: f64 = points
            .windows(2)
            .map(|w| w[0][0] * w[1][1] - w[1][0] * w[0][1])
            .sum();
        assert!((area - 1.0).abs() < 1e-12, "shoelace sum {area}");
    }

    /// `ReverseContourOrientation` reverses each point list; a closed loop stays
    /// closed and its winding flips.
    #[test]
    fn reverse_contour_orientation_flips_the_point_order() {
        let image = img_f64(&[3, 3], vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let forward = contour_extractor_2d(&image, 0.5, false, false, false).unwrap();
        let reversed = contour_extractor_2d(&image, 0.5, true, false, false).unwrap();
        assert_eq!(reversed.len(), 1);
        assert!(reversed[0].is_closed);

        let mut expected = forward[0].points.clone();
        expected.reverse();
        assert_eq!(reversed[0].points, expected);
        assert_eq!(
            reversed[0].points,
            vec![[1.0, 1.5], [1.5, 1.0], [1.0, 0.5], [0.5, 1.0], [1.0, 1.5],]
        );
    }

    /// `ContourValue` sets where on each edge the crossing lands. With a centre
    /// of 4, a background of 0 and an iso of 1, every crossing is a quarter of
    /// the way from the low side: `t = (1 - 0) / (4 - 0) = 0.25` on edges that
    /// run low->high, and `t = (1 - 4) / (0 - 4) = 0.75` on edges that run
    /// high->low. The diamond is the same shape, pulled in toward the centre.
    #[test]
    fn contour_value_sets_the_interpolated_crossing_position() {
        let image = img_f64(&[3, 3], vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let contours = contour_extractor_2d(&image, 1.0, false, false, false).unwrap();
        assert_eq!(contours.len(), 1);
        assert_eq!(
            contours[0].points,
            vec![
                [1.0, 1.75],
                [0.25, 1.0],
                [1.0, 0.25],
                [1.75, 1.0],
                [1.0, 1.75],
            ]
        );
    }

    /// A contour that reaches the image border cannot close: a bright column at
    /// `x = 0` produces one open polyline running down the `x = 0.5` line, with
    /// the bright side on its right.
    #[test]
    fn a_contour_touching_the_image_border_is_left_open() {
        let image = img_f64(&[3, 3], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let contours = contour_extractor_2d(&image, 0.5, false, false, false).unwrap();
        assert_eq!(contours.len(), 1);
        assert!(!contours[0].is_closed);
        assert_eq!(contours[0].points, vec![[0.5, 0.0], [0.5, 1.0], [0.5, 2.0]]);
    }

    /// Saddle case 6 (`v1`, `v2` high) on the classic 2x2 checkerboard. With
    /// `vertex_connect_high_pixels` off the low diagonal is joined -- segments
    /// are right->top and left->bottom; with it on they become left->top and
    /// right->bottom. Two open contours either way, but paired differently.
    #[test]
    fn saddle_case_six_pairs_endpoints_by_vertex_connect_high_pixels() {
        let image = img_f64(&[2, 2], vec![0.0, 1.0, 1.0, 0.0]);

        let low = contour_extractor_2d(&image, 0.5, false, false, false).unwrap();
        assert_eq!(low.len(), 2);
        assert!(low.iter().all(|c| !c.is_closed));
        assert_eq!(low[0].points, vec![[1.0, 0.5], [0.5, 0.0]]);
        assert_eq!(low[1].points, vec![[0.0, 0.5], [0.5, 1.0]]);

        let high = contour_extractor_2d(&image, 0.5, false, true, false).unwrap();
        assert_eq!(high.len(), 2);
        assert_eq!(high[0].points, vec![[0.0, 0.5], [0.5, 0.0]]);
        assert_eq!(high[1].points, vec![[1.0, 0.5], [0.5, 1.0]]);
    }

    /// Saddle case 9 (`v0`, `v3` high), the other checkerboard phase:
    /// top->left / bottom->right by default, top->right / bottom->left when
    /// high pixels are vertex-connected.
    #[test]
    fn saddle_case_nine_pairs_endpoints_by_vertex_connect_high_pixels() {
        let image = img_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);

        let low = contour_extractor_2d(&image, 0.5, false, false, false).unwrap();
        assert_eq!(low.len(), 2);
        assert_eq!(low[0].points, vec![[0.5, 0.0], [0.0, 0.5]]);
        assert_eq!(low[1].points, vec![[0.5, 1.0], [1.0, 0.5]]);

        let high = contour_extractor_2d(&image, 0.5, false, true, false).unwrap();
        assert_eq!(high.len(), 2);
        assert_eq!(high[0].points, vec![[0.5, 0.0], [1.0, 0.5]]);
        assert_eq!(high[1].points, vec![[0.5, 1.0], [0.0, 0.5]]);
    }

    /// A "U" shape: the right arm opens its own contour (square (2,0)) before
    /// square (2,1) joins it to the one opened at (0,0). That is the
    /// `tail > head` merge branch -- the newer contour is appended onto the
    /// older, which keeps contour 0's list slot. One closed 13-point loop.
    #[test]
    fn distinct_contours_merge_newer_into_older_when_the_tail_is_newer() {
        #[rustfmt::skip]
        let image = img_f64(&[5, 4], vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let contours = contour_extractor_2d(&image, 0.5, false, false, false).unwrap();
        assert_eq!(contours.len(), 1);
        assert!(contours[0].is_closed);
        assert_eq!(
            contours[0].points,
            vec![
                [3.0, 2.5],
                [2.0, 2.5],
                [1.0, 2.5],
                [0.5, 2.0],
                [0.5, 1.0],
                [1.0, 0.5],
                [1.5, 1.0],
                [2.0, 1.5],
                [2.5, 1.0],
                [3.0, 0.5],
                [3.5, 1.0],
                [3.5, 2.0],
                [3.0, 2.5],
            ]
        );
    }

    /// An upside-down "U": square (1,1) opens contour 1 inside the notch, and
    /// square (1,2) later finds contour 1 *ending* at `from` while contour 0
    /// *starts* at `to`. Since the head is the newer one, that is the
    /// `tail < head` branch -- the newer contour is prepended onto the older.
    #[test]
    fn distinct_contours_merge_newer_into_older_when_the_head_is_newer() {
        #[rustfmt::skip]
        let image = img_f64(&[5, 4], vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let contours = contour_extractor_2d(&image, 0.5, false, false, false).unwrap();
        assert_eq!(contours.len(), 1);
        assert!(contours[0].is_closed);
        assert_eq!(
            contours[0].points,
            vec![
                [3.0, 2.5],
                [2.5, 2.0],
                [2.0, 1.5],
                [1.5, 2.0],
                [1.0, 2.5],
                [0.5, 2.0],
                [0.5, 1.0],
                [1.0, 0.5],
                [2.0, 0.5],
                [3.0, 0.5],
                [3.5, 1.0],
                [3.5, 2.0],
                [3.0, 2.5],
            ]
        );
    }

    /// A square with exactly one corner *on* the contour value and the rest
    /// above it is case 14 (`left -> top`), and both interpolations land on that
    /// corner: `t = (1 - 1) / (2 - 1) = 0`. The arc is degenerate and dropped,
    /// so nothing is emitted at all.
    #[test]
    fn a_degenerate_zero_length_arc_is_dropped() {
        let image = img_f64(&[2, 2], vec![1.0, 2.0, 2.0, 2.0]);
        let contours = contour_extractor_2d(&image, 1.0, false, false, false).unwrap();
        assert!(contours.is_empty());
    }

    /// A uniform image has no crossing anywhere. So does an image with fewer
    /// than two rows or columns, where no whole 2x2 square exists.
    #[test]
    fn images_without_a_crossing_or_a_whole_square_yield_nothing() {
        assert!(
            contour_extractor_2d(&img_f64(&[4, 4], vec![7.0; 16]), 0.5, false, false, false)
                .unwrap()
                .is_empty()
        );
        assert!(
            contour_extractor_2d(
                &img_f64(&[1, 4], vec![0.0, 1.0, 0.0, 1.0]),
                0.5,
                false,
                false,
                false
            )
            .unwrap()
            .is_empty()
        );
        assert!(
            contour_extractor_2d(
                &img_f64(&[4, 1], vec![0.0, 1.0, 0.0, 1.0]),
                0.5,
                false,
                false,
                false
            )
            .unwrap()
            .is_empty()
        );
    }

    /// `ContourValue` is `static_cast` to the input pixel type first, so on a
    /// `UInt8` image every value in `[0, 1)` becomes `0`. That changes both the
    /// `> contour_value` test *and* the interpolation: with the cast value each
    /// crossing solves `t = (0 - 0) / (1 - 0) = 0`, collapsing onto the
    /// background end of its edge. The diamond therefore runs through the four
    /// background pixels neighbouring the bright one rather than through the
    /// edge midpoints. A value of `1.0` selects nothing, since no pixel exceeds
    /// it.
    #[test]
    fn contour_value_is_cast_to_the_input_pixel_type() {
        let image = img_u8(&[3, 3], vec![0, 0, 0, 0, 1, 0, 0, 0, 0]);
        let half = contour_extractor_2d(&image, 0.5, false, false, false).unwrap();
        let nine_tenths = contour_extractor_2d(&image, 0.9, false, false, false).unwrap();
        let zero = contour_extractor_2d(&image, 0.0, false, false, false).unwrap();
        assert_eq!(half, zero);
        assert_eq!(half, nine_tenths);
        assert_eq!(half.len(), 1);
        assert!(half[0].is_closed);
        assert_eq!(
            half[0].points,
            vec![[1.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0],]
        );

        assert!(
            contour_extractor_2d(&image, 1.0, false, false, false)
                .unwrap()
                .is_empty()
        );
    }

    /// `LabelContours` on a uniform 2x2: the single label's bounding box is the
    /// whole image, grown by one pixel, so the sweep runs over top-left corners
    /// `[-1, 1]^2`. Out-of-image samples are "not this label", crossings are
    /// fixed at the midpoint, and the result is the block's boundary -- with
    /// negative coordinates where it runs outside the image.
    #[test]
    fn label_contours_traces_a_block_boundary_at_midpoints() {
        let image = img_u8(&[2, 2], vec![0, 0, 0, 0]);
        let contours = contour_extractor_2d(&image, 0.0, false, false, true).unwrap();
        assert_eq!(contours.len(), 1);
        assert!(contours[0].is_closed);
        assert_eq!(
            contours[0].points,
            vec![
                [1.0, 1.5],
                [0.0, 1.5],
                [-0.5, 1.0],
                [-0.5, 0.0],
                [0.0, -0.5],
                [1.0, -0.5],
                [1.5, 0.0],
                [1.5, 1.0],
                [1.0, 1.5],
            ]
        );
    }

    /// Every distinct label is traced separately, in ascending label order, and
    /// `contour_value` is ignored. A lone `1` inside a field of `0` gives label
    /// `0`'s contours first, then label `1`'s midpoint diamond.
    #[test]
    fn label_contours_returns_every_label_in_ascending_order() {
        let image = img_u8(&[3, 3], vec![0, 0, 0, 0, 1, 0, 0, 0, 0]);
        let contours = contour_extractor_2d(&image, 12345.0, false, false, true).unwrap();

        // Label 1 occupies one pixel; its contour is the last one emitted and is
        // the same diamond the iso-value mode finds at 0.5.
        let last = contours.last().unwrap();
        assert!(last.is_closed);
        assert_eq!(
            last.points,
            vec![[1.0, 1.5], [0.5, 1.0], [1.0, 0.5], [1.5, 1.0], [1.0, 1.5],]
        );
        // Label 0 is the ring around it: an outer boundary and an inner one.
        assert_eq!(contours.len(), 3);
        assert!(contours[0].is_closed && contours[1].is_closed);
    }

    /// A `UInt8` image using all 256 values leaves ITK no unused label to use as
    /// its out-of-image constant, and it throws.
    #[test]
    fn label_contours_rejects_an_image_that_exhausts_the_label_space() {
        let image = img_u8(&[16, 16], (0..=255u8).collect());
        assert_eq!(
            contour_extractor_2d(&image, 0.0, false, false, true).unwrap_err(),
            FilterError::ContourExtractorNoUnusedLabel(PixelId::UInt8)
        );
        // One value short of exhaustion is fine.
        let mut data: Vec<u8> = (0..=255u8).collect();
        data[255] = 0;
        assert!(contour_extractor_2d(&img_u8(&[16, 16], data), 0.0, false, false, true).is_ok());
    }

    #[test]
    fn rejects_non_2d_input() {
        let image = img_f64(&[2, 2, 2], vec![0.0; 8]);
        assert_eq!(
            contour_extractor_2d(&image, 0.5, false, false, false).unwrap_err(),
            FilterError::UnsupportedContourExtractorDimension(3)
        );
        let image = img_f64(&[4], vec![0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            contour_extractor_2d(&image, 0.5, false, false, false).unwrap_err(),
            FilterError::UnsupportedContourExtractorDimension(1)
        );
    }
}
