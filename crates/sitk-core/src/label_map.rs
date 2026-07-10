//! `itk::LabelMap`: a sparse, run-length-encoded image of labelled objects.
//!
//! Port of `Modules/Filtering/LabelMap/include/`:
//!
//! - `itkLabelObjectLine.h` / `.hxx` — `{ IndexType m_Index; LengthType m_Length; }`,
//!   a run of `m_Length` pixels starting at `m_Index` and extending along axis 0.
//! - `itkLabelObject.h` / `.hxx` — one label's lines, in a `std::deque`.
//! - `itkLabelMap.h` / `.hxx` — `std::map<LabelType, LabelObjectPointer>` plus a
//!   background value, deriving from `itk::ImageBase` (geometry, no pixel
//!   container).
//!
//! ## Why this is a separate type, not an `Image`
//!
//! `itk::LabelMap` derives from `ImageBase`, not `Image` (`itkLabelMap.h:69`):
//! it carries size/spacing/origin/direction and has **no `PixelContainer`**.
//! SimpleITK nonetheless forces it into its single `sitk::Image` class, and
//! pays for it with a runtime throw in every buffer accessor
//! (`sitkPimpleImageBase.hxx:829-832`):
//!
//! ```cpp
//! if constexpr (IsLabel<ImageType>::Value)
//! { sitkExceptionMacro("This method is not supported for LabelMaps.") }
//! ```
//!
//! That is upstream runtime-rejecting a state a type could have made
//! unrepresentable. Here [`LabelMap`] is its own type, so it can never be
//! handed to a filter that expects pixel data — no guard, no throw, no
//! `Option<PixelBuffer>` branch under every `match PixelId`. Consequently
//! `PixelId` gains no `Label*` variants; a `LabelMap` instead *stores* the
//! [`PixelId`] of the label image it came from (and will round-trip back to),
//! which is what `itk::LabelObject`'s `TLabel` template parameter is.
//!
//! ## The "optimized" invariant
//!
//! In ITK, `LabelObject::Optimize()` (`itkLabelObject.h:196-200`,
//! `itkLabelObject.hxx:299-361`) sorts the lines, merges touching ones and
//! removes double coverage — and it is **opt-in**. Until it is called, lines may
//! overlap, so `Size()` (`itkLabelObject.h:153-161`) carries the warning
//! *"To get an accurate result, you need to make sure there is no duplication
//! in the line container."*
//!
//! This port maintains that optimized state as an **invariant of
//! [`LabelObject`]**, restored by every mutator:
//!
//! ```text
//! every line has length >= 1
//! lines are sorted by `itk::Functor::LabelObjectLineComparator`
//!   (reverse-dimension lexicographic on the start index)
//! no two lines on the same row touch or overlap:
//!   prev.index[0] + prev.length < next.index[0]
//! ```
//!
//! [`LabelObject::add_line`] is the single owner of that invariant: it is the
//! only way a line enters the container, and it restores the three rules on
//! exit. There is therefore **no public `optimize()`**, and
//! [`LabelObject::size`] is always exact.
//!
//! This is a deliberate strengthening over upstream, recorded in
//! `doc/upstream-findings.md` §4. It is safe because ITK's opt-in design exists
//! for a reason Rust does not have: `LabelObject` is a `LightObject` shared by
//! raw pointer, so many writers append lines before any one of them can decide
//! the object is complete. A `&mut LabelObject` has exactly one writer at a
//! time, and re-establishing the invariant per line is `O(log n)` amortized
//! when lines arrive in raster order — which is how every producer in this
//! crate emits them.
//!
//! Two upstream behaviours are *not* reproduced as a result, both strictly
//! narrower than what this type can represent:
//!
//! - `AddLine` "without any check" (`itkLabelObject.hxx:182-190`,
//!   `itkLabelMap.h:216-222`) can leave a pixel covered twice within one
//!   object. Here it cannot.
//! - `LabelObject::Size()` accumulates into an `int` (`itkLabelObject.hxx:217`),
//!   overflowing above 2^31 pixels. [`LabelObject::size`] returns `u64`.
//!
//! Double coverage *between* objects of different labels is still
//! representable, because ITK's `LabelUniqueLabelMapFilter` exists precisely to
//! remove it and `LabelMap::GetPixel` (`itkLabelMap.hxx:155-170`) is specified
//! in terms of it ("If the given index is contained in several objects, only
//! the smallest label of those objects is returned").

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::image::Image;
use crate::pixel::{PixelId, Scalar};
use crate::{dispatch_scalar, matrix};

/// The largest image dimension a [`LabelMap`] supports.
///
/// `itk::LabelObjectLine` is templated on the dimension and stores an
/// `itk::Index<VImageDimension>`; SimpleITK instantiates its LabelMap filters
/// for 2-D and 3-D. A fixed-size index keeps [`LabelObjectLine`] `Copy` and
/// allocation-free, which the run-length inner loops depend on.
pub const MAX_DIM: usize = 3;

/// `itk::LabelObjectLine`: `length` consecutive pixels starting at `index` and
/// running along axis 0.
///
/// The start index is zero-padded to [`MAX_DIM`]; the padding is what makes the
/// reverse-dimension comparison below dimension-agnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelObjectLine {
    index: [i64; MAX_DIM],
    length: i64,
}

impl LabelObjectLine {
    /// The line's start index, zero-padded to [`MAX_DIM`].
    pub fn index(&self) -> [i64; MAX_DIM] {
        self.index
    }

    /// Number of pixels in the run. Always `>= 1`.
    pub fn length(&self) -> i64 {
        self.length
    }

    /// One past the last index along axis 0.
    fn end(&self) -> i64 {
        self.index[0] + self.length
    }

    /// `itkLabelObjectLine.hxx:64-77` — is `idx` one of this line's pixels?
    ///
    /// `idx` shorter than [`MAX_DIM`] is compared against the zero padding, so a
    /// 2-D index never matches a 3-D line at `z != 0`.
    pub fn has_index(&self, idx: &[i64]) -> bool {
        if !same_row(self.index, pad_index(idx)) {
            return false;
        }
        idx[0] >= self.index[0] && idx[0] < self.end()
    }
}

/// Zero-pad an index of any length `<= MAX_DIM` to `MAX_DIM`.
///
/// Panics if `idx` is longer than [`MAX_DIM`]; every caller has already passed
/// through a dimension check.
fn pad_index(idx: &[i64]) -> [i64; MAX_DIM] {
    let mut out = [0i64; MAX_DIM];
    out[..idx.len()].copy_from_slice(idx);
    out
}

/// Do two start indices name the same axis-0 row?
fn same_row(a: [i64; MAX_DIM], b: [i64; MAX_DIM]) -> bool {
    a[1..] == b[1..]
}

/// `itk::Functor::LabelObjectLineComparator`
/// (`itkLabelObjectLineComparator.h:41-48`): reverse-dimension lexicographic on
/// the start index, breaking ties by length.
///
/// Under the [`LabelObject`] invariant no two lines share a start index, so the
/// length tiebreak is unreachable; it is kept because it is what makes this
/// function a strict weak ordering on arbitrary lines, which
/// [`LabelObject::add_line`]'s `partition_point` needs while a duplicate start
/// index is momentarily possible.
fn cmp_lines(a: &LabelObjectLine, b: &LabelObjectLine) -> Ordering {
    for i in (0..MAX_DIM).rev() {
        match a.index[i].cmp(&b.index[i]) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    a.length.cmp(&b.length)
}

/// `itk::LabelObject`: one label's pixels, run-length encoded.
///
/// The lines always satisfy the optimized invariant described in the [module
/// docs](self); [`LabelObject::add_line`] is its single owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelObject {
    label: i64,
    dimension: usize,
    lines: Vec<LabelObjectLine>,
}

impl LabelObject {
    /// An empty object carrying `label`, for a `dimension`-dimensional map.
    ///
    /// Returns [`Error::UnsupportedLabelMapDimension`] unless
    /// `1 <= dimension <= MAX_DIM`.
    pub fn new(label: i64, dimension: usize) -> Result<Self> {
        if dimension == 0 || dimension > MAX_DIM {
            return Err(Error::UnsupportedLabelMapDimension(dimension));
        }
        Ok(LabelObject {
            label,
            dimension,
            lines: Vec::new(),
        })
    }

    /// The label this object carries.
    pub fn label(&self) -> i64 {
        self.label
    }

    /// Number of spatial dimensions.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The object's lines, in `LabelObjectLineComparator` order.
    pub fn lines(&self) -> &[LabelObjectLine] {
        &self.lines
    }

    /// `itkLabelObject.hxx:213-224` — the number of pixels in the object.
    ///
    /// Exact, unconditionally: the invariant forbids double coverage, so the
    /// sum of the line lengths *is* the pixel count. Upstream's `Size()` needs
    /// a prior `Optimize()` for the same guarantee, and accumulates into an
    /// `int`.
    pub fn size(&self) -> u64 {
        self.lines.iter().map(|l| l.length as u64).sum()
    }

    /// `itkLabelObject.hxx:226-231`.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// `itkLabelObject.hxx:77-93` — is `idx` one of this object's pixels?
    pub fn has_index(&self, idx: &[i64]) -> bool {
        idx.len() == self.dimension && self.lines.iter().any(|l| l.has_index(idx))
    }

    /// Every pixel index of the object, in raster order.
    pub fn indices(&self) -> impl Iterator<Item = Vec<i64>> + '_ {
        let dim = self.dimension;
        self.lines.iter().flat_map(move |line| {
            (0..line.length).map(move |k| {
                let mut idx = line.index;
                idx[0] += k;
                idx[..dim].to_vec()
            })
        })
    }

    /// Add a run of `length` pixels starting at `index`, restoring the
    /// optimized invariant.
    ///
    /// This is the **single owner** of the invariant: it is the only way a line
    /// enters the container. The result is exactly what upstream's
    /// `AddLine` followed by `Optimize()` (`itkLabelObject.hxx:299-361`) would
    /// produce, including upstream's merge predicate
    /// `currentIdx[0] + currentLength >= idx[0]` — lines that merely *touch*
    /// merge, they need not overlap.
    ///
    /// Errors on a wrong-dimension `index` or on `length < 1`; upstream stores
    /// a zero-length line, which the invariant makes unrepresentable.
    pub fn add_line(&mut self, index: &[i64], length: i64) -> Result<()> {
        if index.len() != self.dimension {
            return Err(Error::GeometryMismatch {
                dimension: self.dimension,
            });
        }
        if length < 1 {
            return Err(Error::NonPositiveLineLength(length));
        }
        let line = LabelObjectLine {
            index: pad_index(index),
            length,
        };

        // Producers emit lines in raster order, which is exactly ascending
        // `cmp_lines` order, so the common case is an append.
        let pos = match self.lines.last() {
            Some(last) if cmp_lines(last, &line) == Ordering::Less => self.lines.len(),
            None => 0,
            _ => self
                .lines
                .partition_point(|l| cmp_lines(l, &line) == Ordering::Less),
        };
        self.lines.insert(pos, line);

        // Only the new line can bridge two previously non-touching lines, so a
        // single left merge followed by a right cascade restores the invariant.
        let mut i = pos;
        if i > 0 && merge_touching(&mut self.lines, i - 1) {
            i -= 1;
        }
        while i + 1 < self.lines.len() && merge_touching(&mut self.lines, i) {}
        Ok(())
    }

    /// `itkLabelObject.hxx:146-160` — add a single pixel.
    pub fn add_index(&mut self, index: &[i64]) -> Result<()> {
        self.add_line(index, 1)
    }

    /// `itkLabelObject.hxx:375-380`.
    pub fn clear(&mut self) {
        self.lines.clear();
    }
}

/// Merge `lines[i]` and `lines[i + 1]` if they are on the same row and touch or
/// overlap. Returns whether a merge happened.
///
/// `lines` is `cmp_lines`-sorted, so `lines[i].index[0] <= lines[i+1].index[0]`
/// whenever the two share a row.
fn merge_touching(lines: &mut Vec<LabelObjectLine>, i: usize) -> bool {
    let (a, b) = (lines[i], lines[i + 1]);
    if !same_row(a.index, b.index) || a.end() < b.index[0] {
        return false;
    }
    lines[i].length = a.end().max(b.end()) - a.index[0];
    lines.remove(i + 1);
    true
}

/// `itk::LabelMap`: label objects keyed by label, plus a background value and
/// `itk::ImageBase` geometry.
///
/// Two invariants hold by construction, both owned by this type's mutators:
///
/// ```text
/// objects[k].label() == k          for every key k
/// background not in objects.keys()
/// ```
///
/// The second is why [`LabelMap::set_background`] returns the object it had to
/// evict: upstream's `ChangeLabelLabelMapFilter` performs exactly that eviction
/// by hand before every `SetBackgroundValue`
/// (`itkChangeLabelLabelMapFilter.hxx:107-127`).
#[derive(Clone, Debug, PartialEq)]
pub struct LabelMap {
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
    pixel_id: PixelId,
    background: i64,
    objects: BTreeMap<i64, LabelObject>,
}

impl LabelMap {
    /// An empty map over `size`, whose labels round-trip to a `pixel_id` label
    /// image.
    ///
    /// Geometry defaults to unit spacing, zero origin and an identity
    /// direction, as `itk::ImageBase` does.
    pub fn new(size: &[usize], pixel_id: PixelId, background: i64) -> Result<Self> {
        let dim = size.len();
        if dim == 0 || dim > MAX_DIM {
            return Err(Error::UnsupportedLabelMapDimension(dim));
        }
        if !pixel_id.is_integer_scalar() {
            return Err(Error::RequiresIntegerPixelType(pixel_id));
        }
        Ok(LabelMap {
            size: size.to_vec(),
            spacing: vec![1.0; dim],
            origin: vec![0.0; dim],
            direction: matrix::identity(dim),
            pixel_id,
            background,
            objects: BTreeMap::new(),
        })
    }

    /// `itk::LabelImageToLabelMapFilter::ThreadedGenerateData`
    /// (`itkLabelImageToLabelMapFilter.hxx:84-124`): walk the image in raster
    /// order, emitting one line per maximal run of equal, non-background pixels
    /// along axis 0.
    ///
    /// The lines of each object therefore come out ordered by `idx[1..]` first,
    /// then by `idx[0]` — which is `LabelObjectLineComparator` order, so the
    /// [`LabelObject`] invariant costs nothing here and no run ever merges with
    /// another (two maximal runs on one row are separated by at least one
    /// background pixel).
    ///
    /// `background` is compared against the *integer* pixel values, matching
    /// upstream's `value != static_cast<InputImagePixelType>(m_BackgroundValue)`.
    pub fn from_label_image(img: &Image, background: i64) -> Result<Self> {
        if !img.pixel_id().is_integer_scalar() {
            return Err(Error::RequiresIntegerPixelType(img.pixel_id()));
        }
        let size = img.size();
        let dim = size.len();
        if dim > MAX_DIM {
            return Err(Error::UnsupportedLabelMapDimension(dim));
        }

        let mut map = LabelMap::new(size, img.pixel_id(), background)?;
        map.spacing = img.spacing().to_vec();
        map.origin = img.origin().to_vec();
        map.direction = img.direction().to_vec();

        let labels: Vec<i64> = img
            .to_f64_vec()?
            .iter()
            .map(|&v| v.round() as i64)
            .collect();
        let nx = size[0];
        let n_rest: usize = size[1..].iter().product();

        for rest in 0..n_rest {
            let mut idx = [0i64; MAX_DIM];
            let mut t = rest;
            for (d, &sz) in size.iter().enumerate().take(dim).skip(1) {
                idx[d] = (t % sz) as i64;
                t /= sz;
            }
            let base = rest * nx;

            let mut x = 0usize;
            while x < nx {
                let value = labels[base + x];
                if value == background {
                    x += 1;
                    continue;
                }
                let start = x;
                x += 1;
                while x < nx && labels[base + x] == value {
                    x += 1;
                }
                idx[0] = start as i64;
                map.set_line(&idx[..dim], (x - start) as i64, value)?;
            }
        }
        Ok(map)
    }

    /// The pixel type of the label image this map round-trips to.
    pub fn pixel_id(&self) -> PixelId {
        self.pixel_id
    }

    /// `itkLabelMap.h:288-293`.
    pub fn background(&self) -> i64 {
        self.background
    }

    /// Set the background value, evicting and returning the object whose label
    /// collides with it, if any.
    ///
    /// The eviction is what keeps `background ∉ objects.keys()` true by
    /// construction. Upstream spells the same sequence out at
    /// `itkChangeLabelLabelMapFilter.hxx:114-126`.
    pub fn set_background(&mut self, value: i64) -> Option<LabelObject> {
        self.background = value;
        self.objects.remove(&value)
    }

    /// `itkLabelMap.hxx:148-153`.
    ///
    /// Upstream's declaration claims *"If the label is the background one, true
    /// is also returned"* (`itkLabelMap.h:159-163`); the implementation is a
    /// plain `find(label) != end()` and returns `false` for the background.
    /// The implementation is reproduced, not the comment — and here the
    /// background can never be a key at all.
    pub fn has_label(&self, label: i64) -> bool {
        self.objects.contains_key(&label)
    }

    /// `itkLabelMap.hxx:108-125`. `None` where upstream throws.
    pub fn label_object(&self, label: i64) -> Option<&LabelObject> {
        self.objects.get(&label)
    }

    /// The labels present, ascending — `itkLabelMap.hxx:477-489`.
    pub fn labels(&self) -> impl Iterator<Item = i64> + '_ {
        self.objects.keys().copied()
    }

    /// The label objects, in ascending label order — `itkLabelMap.hxx:491-503`.
    pub fn label_objects(&self) -> impl Iterator<Item = &LabelObject> + '_ {
        self.objects.values()
    }

    /// `itkLabelMap.h:269-274`.
    pub fn number_of_label_objects(&self) -> usize {
        self.objects.len()
    }

    /// `itkLabelMap.hxx:367-375` — insert `object`, overriding any object that
    /// already carries its label.
    ///
    /// Rejects the background label, which upstream admits into the container
    /// and then throws on at every `GetLabelObject`/`RemoveLabel`
    /// (`itkLabelMap.hxx:110-116`, `:453-459`).
    pub fn add_label_object(&mut self, object: LabelObject) -> Result<()> {
        if object.dimension != self.dimension() {
            return Err(Error::GeometryMismatch {
                dimension: self.dimension(),
            });
        }
        if object.label == self.background {
            return Err(Error::LabelIsBackground(object.label));
        }
        self.objects.insert(object.label, object);
        Ok(())
    }

    /// `itkLabelMap.hxx:451-460`.
    pub fn remove_label(&mut self, label: i64) -> Option<LabelObject> {
        self.objects.remove(&label)
    }

    /// `itkLabelMap.hxx:465-475`.
    pub fn clear_labels(&mut self) {
        self.objects.clear();
    }

    /// `itkLabelMap.hxx:321-347` — add a run to `label`'s object, creating it if
    /// absent.
    ///
    /// A run whose label is the background is dropped, as upstream does.
    /// Unlike upstream, the run cannot leave the object double-covered: see the
    /// [module docs](self).
    pub fn set_line(&mut self, index: &[i64], length: i64, label: i64) -> Result<()> {
        if label == self.background {
            return Ok(());
        }
        let dim = self.dimension();
        if index.len() != dim {
            return Err(Error::GeometryMismatch { dimension: dim });
        }
        match self.objects.get_mut(&label) {
            Some(object) => object.add_line(index, length),
            None => {
                let mut object = LabelObject::new(label, dim)?;
                object.add_line(index, length)?;
                self.objects.insert(label, object);
                Ok(())
            }
        }
    }

    /// `itkLabelMap.hxx:155-170` — the label at `idx`, or the background value.
    ///
    /// When several objects cover `idx`, the **smallest** label wins, because
    /// `std::map` iterates ascending. Reproduced by `BTreeMap`.
    pub fn get_pixel(&self, idx: &[i64]) -> i64 {
        self.objects
            .values()
            .find(|o| o.has_index(idx))
            .map_or(self.background, |o| o.label)
    }

    /// `itk::LabelMapToLabelImageFilter` (`itkLabelMapToLabelImageFilter.hxx:28-52`):
    /// fill with the background value, then paint each object's pixels with its
    /// label.
    ///
    /// Objects are painted in ascending label order, so where two objects
    /// overlap the **larger** label wins — the opposite of [`LabelMap::get_pixel`],
    /// and exactly what upstream's `LabelMapFilter` iteration order produces.
    pub fn to_label_image(&self) -> Result<Image> {
        let total: usize = self.size.iter().product();
        let mut values = vec![self.background as f64; total];
        let dim = self.dimension();

        let mut strides = [1usize; MAX_DIM];
        for d in 1..dim {
            strides[d] = strides[d - 1] * self.size[d - 1];
        }

        for object in self.objects.values() {
            let label = object.label as f64;
            for line in &object.lines {
                let base: usize = (1..dim).map(|d| line.index[d] as usize * strides[d]).sum();
                let start = base + line.index[0] as usize;
                values[start..start + line.length as usize].fill(label);
            }
        }

        let mut img = dispatch_scalar!(self.pixel_id, build_label_image, &self.size, &values)?;
        self.apply_geometry_to(&mut img)?;
        Ok(img)
    }

    /// Per-dimension size of the map.
    pub fn size(&self) -> &[usize] {
        &self.size
    }

    /// Number of spatial dimensions.
    pub fn dimension(&self) -> usize {
        self.size.len()
    }

    /// Physical size of one pixel along each axis.
    pub fn spacing(&self) -> &[f64] {
        &self.spacing
    }

    /// Physical position of index zero.
    pub fn origin(&self) -> &[f64] {
        &self.origin
    }

    /// Row-major `dimension x dimension` direction cosine matrix.
    pub fn direction(&self) -> &[f64] {
        &self.direction
    }

    /// Copy `img`'s geometry, which must have this map's dimension.
    pub fn copy_geometry_from(&mut self, img: &Image) -> Result<()> {
        if img.dimension() != self.dimension() {
            return Err(Error::GeometryMismatch {
                dimension: self.dimension(),
            });
        }
        self.spacing = img.spacing().to_vec();
        self.origin = img.origin().to_vec();
        self.direction = img.direction().to_vec();
        Ok(())
    }

    /// Apply this map's geometry to `img`, which must have this map's dimension.
    pub fn apply_geometry_to(&self, img: &mut Image) -> Result<()> {
        img.set_spacing(&self.spacing)?;
        img.set_origin(&self.origin)?;
        img.set_direction(&self.direction)?;
        Ok(())
    }
}

fn build_label_image<T: Scalar>(size: &[usize], values: &[f64]) -> Result<Image> {
    let data: Vec<T> = values.iter().map(|&v| T::from_f64(v)).collect();
    Image::from_vec(size, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object_with(lines: &[([i64; 2], i64)]) -> LabelObject {
        let mut o = LabelObject::new(7, 2).unwrap();
        for (idx, len) in lines {
            o.add_line(idx, *len).unwrap();
        }
        o
    }

    fn line_tuples(o: &LabelObject) -> Vec<([i64; MAX_DIM], i64)> {
        o.lines().iter().map(|l| (l.index(), l.length())).collect()
    }

    #[test]
    fn add_line_keeps_lines_in_raster_order_whatever_the_insertion_order() {
        let o = object_with(&[([5, 2], 1), ([0, 0], 1), ([3, 1], 1), ([0, 2], 1)]);
        assert_eq!(
            line_tuples(&o),
            vec![
                ([0, 0, 0], 1),
                ([3, 1, 0], 1),
                ([0, 2, 0], 1),
                ([5, 2, 0], 1),
            ]
        );
    }

    #[test]
    fn touching_lines_merge() {
        // itkLabelObject.hxx:341 merges on `>=`, so a zero gap is enough.
        let o = object_with(&[([0, 0], 3), ([3, 0], 2)]);
        assert_eq!(line_tuples(&o), vec![([0, 0, 0], 5)]);
        assert_eq!(o.size(), 5);
    }

    #[test]
    fn a_one_pixel_gap_does_not_merge() {
        let o = object_with(&[([0, 0], 3), ([4, 0], 2)]);
        assert_eq!(line_tuples(&o), vec![([0, 0, 0], 3), ([4, 0, 0], 2)]);
        assert_eq!(o.size(), 5);
    }

    #[test]
    fn overlapping_lines_merge_and_size_counts_each_pixel_once() {
        let o = object_with(&[([0, 0], 5), ([2, 0], 5)]);
        assert_eq!(line_tuples(&o), vec![([0, 0, 0], 7)]);
        assert_eq!(o.size(), 7);
    }

    #[test]
    fn a_contained_line_leaves_the_container_unchanged() {
        let o = object_with(&[([0, 0], 10), ([3, 0], 2)]);
        assert_eq!(line_tuples(&o), vec![([0, 0, 0], 10)]);
        assert_eq!(o.size(), 10);
    }

    #[test]
    fn an_identical_line_added_twice_is_not_double_counted() {
        let o = object_with(&[([2, 1], 4), ([2, 1], 4)]);
        assert_eq!(line_tuples(&o), vec![([2, 1, 0], 4)]);
        assert_eq!(o.size(), 4);
    }

    #[test]
    fn a_bridging_line_cascades_through_every_line_it_joins() {
        let mut o = object_with(&[([0, 0], 2), ([4, 0], 2), ([8, 0], 2)]);
        assert_eq!(o.lines().len(), 3);
        o.add_line(&[2, 0], 6).unwrap();
        assert_eq!(line_tuples(&o), vec![([0, 0, 0], 10)]);
        assert_eq!(o.size(), 10);
    }

    #[test]
    fn lines_on_different_rows_never_merge() {
        let o = object_with(&[([0, 0], 3), ([0, 1], 3)]);
        assert_eq!(line_tuples(&o), vec![([0, 0, 0], 3), ([0, 1, 0], 3)]);
        assert_eq!(o.size(), 6);
    }

    #[test]
    fn lines_sort_by_the_slowest_axis_first() {
        // The x offsets are two apart so no two lines on a shared row touch;
        // this test is about the comparator, not the merge.
        let mut o = LabelObject::new(1, 3).unwrap();
        for idx in [[0, 0, 1], [0, 1, 0], [2, 0, 0], [0, 0, 0]] {
            o.add_line(&idx, 1).unwrap();
        }
        assert_eq!(
            line_tuples(&o),
            vec![
                ([0, 0, 0], 1),
                ([2, 0, 0], 1),
                ([0, 1, 0], 1),
                ([0, 0, 1], 1),
            ]
        );
    }

    #[test]
    fn add_line_rejects_a_non_positive_length_and_a_wrong_dimension_index() {
        let mut o = LabelObject::new(1, 2).unwrap();
        assert_eq!(o.add_line(&[0, 0], 0), Err(Error::NonPositiveLineLength(0)));
        assert_eq!(
            o.add_line(&[0, 0], -3),
            Err(Error::NonPositiveLineLength(-3))
        );
        assert_eq!(
            o.add_line(&[0, 0, 0], 1),
            Err(Error::GeometryMismatch { dimension: 2 })
        );
        assert!(o.is_empty());
    }

    #[test]
    fn label_object_new_rejects_an_unsupported_dimension() {
        assert_eq!(
            LabelObject::new(1, 0),
            Err(Error::UnsupportedLabelMapDimension(0))
        );
        assert_eq!(
            LabelObject::new(1, 4),
            Err(Error::UnsupportedLabelMapDimension(4))
        );
    }

    #[test]
    fn has_index_and_indices_agree_with_the_lines() {
        let o = object_with(&[([1, 0], 2), ([0, 1], 1)]);
        assert!(o.has_index(&[1, 0]));
        assert!(o.has_index(&[2, 0]));
        assert!(!o.has_index(&[0, 0]));
        assert!(!o.has_index(&[3, 0]));
        assert!(o.has_index(&[0, 1]));
        assert_eq!(
            o.indices().collect::<Vec<_>>(),
            vec![vec![1, 0], vec![2, 0], vec![0, 1]]
        );
    }

    // ---- LabelMap ---------------------------------------------------------

    #[test]
    fn from_label_image_run_length_encodes_in_raster_order() {
        // 4x2, labels: 1 1 0 2 / 2 2 2 0
        let img = Image::from_vec(&[4, 2], vec![1u8, 1, 0, 2, 2, 2, 2, 0]).unwrap();
        let map = LabelMap::from_label_image(&img, 0).unwrap();
        assert_eq!(map.labels().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(
            line_tuples(map.label_object(1).unwrap()),
            vec![([0, 0, 0], 2)]
        );
        assert_eq!(
            line_tuples(map.label_object(2).unwrap()),
            vec![([3, 0, 0], 1), ([0, 1, 0], 3)]
        );
        assert_eq!(map.label_object(2).unwrap().size(), 4);
        assert_eq!(map.pixel_id(), PixelId::UInt8);
        assert_eq!(map.background(), 0);
    }

    #[test]
    fn from_label_image_honours_a_non_zero_background() {
        let img = Image::from_vec(&[3, 1], vec![0u8, 5, 5]).unwrap();
        let map = LabelMap::from_label_image(&img, 5).unwrap();
        assert_eq!(map.labels().collect::<Vec<_>>(), vec![0]);
        assert_eq!(map.label_object(0).unwrap().size(), 1);
    }

    #[test]
    fn from_label_image_rejects_a_float_image() {
        let img = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            LabelMap::from_label_image(&img, 0),
            Err(Error::RequiresIntegerPixelType(PixelId::Float32))
        );
    }

    #[test]
    fn from_label_image_rejects_a_four_dimensional_image() {
        let img = Image::from_vec(&[2, 2, 2, 2], vec![0u8; 16]).unwrap();
        assert_eq!(
            LabelMap::from_label_image(&img, 0),
            Err(Error::UnsupportedLabelMapDimension(4))
        );
    }

    #[test]
    fn to_label_image_round_trips_from_label_image() {
        let img = Image::from_vec(&[4, 2], vec![1u8, 1, 0, 2, 2, 2, 2, 0]).unwrap();
        let map = LabelMap::from_label_image(&img, 0).unwrap();
        assert_eq!(map.to_label_image().unwrap(), img);
    }

    #[test]
    fn to_label_image_paints_the_larger_label_over_an_overlap() {
        let mut map = LabelMap::new(&[3, 1], PixelId::UInt8, 0).unwrap();
        map.set_line(&[0, 0], 2, 1).unwrap();
        map.set_line(&[1, 0], 2, 2).unwrap();
        // get_pixel reports the smallest label, the image the largest.
        assert_eq!(map.get_pixel(&[1, 0]), 1);
        assert_eq!(
            map.to_label_image().unwrap().scalar_slice::<u8>().unwrap(),
            &[1, 2, 2]
        );
    }

    #[test]
    fn to_label_image_carries_the_geometry_across() {
        let mut img = Image::from_vec(&[2, 2], vec![0u8, 1, 1, 0]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        img.set_direction(&[0.0, -1.0, 1.0, 0.0]).unwrap();
        let map = LabelMap::from_label_image(&img, 0).unwrap();
        assert_eq!(map.spacing(), &[0.5, 2.0]);
        let back = map.to_label_image().unwrap();
        assert_eq!(back, img);
    }

    #[test]
    fn set_line_drops_a_background_run() {
        let mut map = LabelMap::new(&[3, 1], PixelId::UInt8, 0).unwrap();
        map.set_line(&[0, 0], 3, 0).unwrap();
        assert_eq!(map.number_of_label_objects(), 0);
    }

    #[test]
    fn add_label_object_rejects_the_background_label_and_a_wrong_dimension() {
        let mut map = LabelMap::new(&[3, 3], PixelId::UInt8, 4).unwrap();
        let bg = LabelObject::new(4, 2).unwrap();
        assert_eq!(map.add_label_object(bg), Err(Error::LabelIsBackground(4)));
        let wrong_dim = LabelObject::new(1, 3).unwrap();
        assert_eq!(
            map.add_label_object(wrong_dim),
            Err(Error::GeometryMismatch { dimension: 2 })
        );
    }

    #[test]
    fn set_background_evicts_the_colliding_object() {
        let mut map = LabelMap::new(&[3, 1], PixelId::UInt8, 0).unwrap();
        map.set_line(&[0, 0], 1, 1).unwrap();
        map.set_line(&[1, 0], 1, 2).unwrap();
        let evicted = map.set_background(2).unwrap();
        assert_eq!(evicted.label(), 2);
        assert_eq!(map.background(), 2);
        assert_eq!(map.labels().collect::<Vec<_>>(), vec![1]);
        assert!(!map.has_label(2));
    }

    #[test]
    fn get_pixel_returns_the_background_outside_every_object() {
        let mut map = LabelMap::new(&[3, 1], PixelId::UInt8, 9).unwrap();
        map.set_line(&[0, 0], 1, 1).unwrap();
        assert_eq!(map.get_pixel(&[0, 0]), 1);
        assert_eq!(map.get_pixel(&[2, 0]), 9);
    }

    #[test]
    fn new_rejects_a_float_pixel_id_and_an_unsupported_dimension() {
        assert_eq!(
            LabelMap::new(&[2, 2], PixelId::Float32, 0),
            Err(Error::RequiresIntegerPixelType(PixelId::Float32))
        );
        assert_eq!(
            LabelMap::new(&[2, 2, 2, 2], PixelId::UInt8, 0),
            Err(Error::UnsupportedLabelMapDimension(4))
        );
    }
}
