//! The label-image ⇄ [`LabelMap`] converters, and the filters that rewrite a
//! [`LabelMap`] in place.
//!
//! Ports of `Modules/Filtering/LabelMap/include/`:
//!
//! - `itkLabelImageToLabelMapFilter.h` / `.hxx`
//! - `itkLabelMapToLabelImageFilter.h` / `.hxx`
//! - `itkBinaryImageToLabelMapFilter.h` / `.hxx`
//! - `itkAggregateLabelMapFilter.h` / `.hxx`
//! - `itkChangeLabelLabelMapFilter.h` / `.hxx`
//! - `itkMergeLabelMapFilter.h` / `.hxx`
//! - `itkRelabelLabelMapFilter.h` → `itkAttributeRelabelLabelMapFilter.hxx`
//! - `itkLabelUniqueLabelMapFilter.h` → `itkAttributeUniqueLabelMapFilter.hxx`
//!
//! All five map filters are `InPlaceLabelMapFilter`s whose SimpleITK wrappers
//! call `InPlaceOff()`, so each function here takes `&LabelMap` and returns a
//! fresh one. `InPlaceLabelMapFilter::AllocateOutputs`
//! (`itkInPlaceLabelMapFilter.hxx:83-105`) copies the input's background value
//! and objects into the output, which is what `map.clone()` stands for below.
//!
//! Two upstream defects are relevant here, and they are handled differently
//! because only one of them is representable in this port. `MergeLabelMapFilter`'s
//! never-cleared deferred deque is **not** reproduced — see
//! [`merge_label_map`]; `AttributeUniqueLabelMapFilter`'s inverted
//! empty-object-removal guard **is** — see [`label_unique_label_map`].
//!
//! The first two are one-liners over [`LabelMap::from_label_image`] and
//! [`LabelMap::to_label_image`], which is where the run-length encoding and the
//! `LabelObject` invariant live. What these wrappers add is SimpleITK's
//! pixel-type gating and its `double`-typed parameter casts.
//!
//! ## `binary_image_to_label_map`
//!
//! `itkBinaryImageToLabelMapFilter` derives from the same `ScanlineFilterCommon`
//! as `itkConnectedComponentImageFilter`, so it shares
//! [`crate::label::scanline_components`] and [`crate::label::create_consecutive`]
//! with [`crate::label::connected_component`]. The three differences:
//!
//! 1. **Foreground is an equality test, not "nonzero".** `.hxx:167` compares
//!    `pixelValue == this->m_InputForegroundValue`. SimpleITK casts the `double`
//!    it exposes to the input pixel type first (`pixeltype: Input`), so a
//!    foreground of `1.5` on a `UInt8` image matches pixels valued `1`.
//!
//! 2. **The label numbering skips the output background.**
//!    `CreateConsecutive(m_OutputBackgroundValue)`
//!    (`itkBinaryImageToLabelMapFilter.hxx:117`,
//!    `itkScanlineFilterCommon.h:199-228`) starts its counter at `0` and bumps
//!    it once, on the single assignment where it would equal the background. So
//!    with the default background `0` the labels are `1, 2, 3, …`; with a
//!    background of `3` they are `0, 1, 2, 4, 5, …`. This differs from
//!    `connected_component`, whose background is fixed at `0`.
//!
//! 3. **The output is a `LabelMap`, not an image.** Each run becomes one
//!    `SetLine` (`.hxx:128-141`), in raster order, so no line ever merges with
//!    another and the [`LabelObject`](sitk_core::LabelObject) invariant costs
//!    nothing.
//!
//! `BinaryImageToLabelMapFilter.yaml` fixes the label type to `uint32_t`
//! (`filter_type: itk::BinaryImageToLabelMapFilter<InputImageType,
//! itk::LabelMap< itk::LabelObject< uint32_t, ... > > >`), so the returned map's
//! [`LabelMap::pixel_id`] is always [`PixelId::UInt32`] regardless of the input's.
//!
//! ### Defaults
//!
//! ITK's constructor (`.hxx:33-35`) defaults `m_InputForegroundValue` to
//! `NumericTraits<InputPixelType>::max()` and `m_OutputBackgroundValue` to
//! `NumericTraits<OutputPixelType>::NonpositiveMin()`, and the yaml's
//! `detaileddescriptionSet` still says so. SimpleITK overrides both: the yaml's
//! declared defaults are `1.0` and `0.0`. [`BinaryImageToLabelMapSettings::default`]
//! follows the yaml, which is the behaviour a SimpleITK caller sees.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};

use sitk_core::{Image, LabelMap, LabelObject, LabelObjectLine, MAX_DIM, PixelId};

use crate::error::{FilterError, Result};
use crate::label::{create_consecutive, scanline_components};
use crate::quantize_to_pixel_type;

/// `itk::LabelImageToLabelMapFilter`: run-length encode an integer label image.
///
/// `LabelImageToLabelMapFilter.yaml` declares
/// `pixel_types: UnsignedIntegerPixelIDTypeList`, so a signed, floating-point or
/// vector image is rejected. `background_value` is a `pixeltype: Output` member
/// — SimpleITK casts it to the label type, which for this filter *is* the input
/// pixel type (`itk::LabelObject< typename InputImageType::PixelType, ... >`) —
/// before ITK compares it against any pixel.
pub fn label_image_to_label_map(img: &Image, background_value: f64) -> Result<LabelMap> {
    if !img.pixel_id().is_integer_scalar() || img.pixel_id().is_signed() {
        return Err(FilterError::RequiresUnsignedIntegerPixelType(
            img.pixel_id(),
        ));
    }
    let background = quantize_to_pixel_type(img.pixel_id(), background_value) as i64;
    Ok(LabelMap::from_label_image(img, background)?)
}

/// `itk::LabelMapToLabelImageFilter`: paint every object's pixels with its
/// label, over a background-filled image.
///
/// The output pixel type is the map's own ([`LabelMap::pixel_id`]), matching the
/// yaml's `output_image_type: itk::Image<typename InputImageType::LabelType, …>`.
pub fn label_map_to_label_image(map: &LabelMap) -> Result<Image> {
    Ok(map.to_label_image()?)
}

/// The three settings `BinaryImageToLabelMapFilter.yaml` exposes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryImageToLabelMapSettings {
    /// Face connectivity when `false` (4-connected in 2-D, 6-connected in 3-D);
    /// face+edge+vertex connectivity when `true`.
    pub fully_connected: bool,
    /// The input value that counts as foreground, compared for **equality**
    /// after a cast to the input pixel type. SimpleITK's default is `1.0`;
    /// ITK's own is `NumericTraits<InputPixelType>::max()`.
    pub input_foreground_value: f64,
    /// The `LabelMap`'s background value, and the label the consecutive
    /// numbering skips. Cast to the `uint32_t` label type. SimpleITK's default
    /// is `0.0`; ITK's own is `NumericTraits<OutputPixelType>::NonpositiveMin()`.
    pub output_background_value: f64,
}

impl Default for BinaryImageToLabelMapSettings {
    fn default() -> Self {
        Self {
            fully_connected: false,
            input_foreground_value: 1.0,
            output_background_value: 0.0,
        }
    }
}

/// `itk::BinaryImageToLabelMapFilter`: label the connected components of the
/// pixels equal to `settings.input_foreground_value`.
///
/// Returns the map and its `NumberOfObjects` measurement.
///
/// `BinaryImageToLabelMapFilter.yaml` declares
/// `pixel_types: IntegerPixelIDTypeList`, so a floating-point or vector image is
/// rejected.
pub fn binary_image_to_label_map(
    img: &Image,
    settings: &BinaryImageToLabelMapSettings,
) -> Result<(LabelMap, u64)> {
    if !img.pixel_id().is_integer_scalar() {
        return Err(FilterError::RequiresIntegerPixelType(img.pixel_id()));
    }
    let size = img.size();
    // `itk::LabelObject<uint32_t>` is the label type the yaml pins.
    let background =
        quantize_to_pixel_type(PixelId::UInt32, settings.output_background_value) as i64;
    let mut map = LabelMap::new(size, PixelId::UInt32, background)?;
    map.copy_geometry_from(img)?;

    let total: usize = size.iter().product();
    if total == 0 {
        return Ok((map, 0));
    }

    let foreground = quantize_to_pixel_type(img.pixel_id(), settings.input_foreground_value);
    let is_fg: Vec<bool> = img.to_f64_vec()?.iter().map(|&v| v == foreground).collect();

    let mut components = scanline_components(&is_fg, size, settings.fully_connected);
    let (root_to_output, number_of_objects) = create_consecutive(&mut components, background);

    let dim = size.len();
    let mut idx = vec![0i64; dim];
    for (line, runs) in components.line_map.iter().enumerate() {
        if runs.is_empty() {
            continue;
        }
        let mut t = line;
        for d in 1..dim {
            idx[d] = (t % size[d]) as i64;
            t /= size[d];
        }
        for run in runs {
            let root = components.uf.find(run.label);
            idx[0] = run.start as i64;
            map.set_line(&idx, run.len as i64, root_to_output[root])?;
        }
    }
    Ok((map, number_of_objects))
}

// ---- the pure label-map filters ----------------------------------------

/// Add every line of `src` to `map`'s object for `target`, creating it if
/// absent.
///
/// This is upstream's `while (!lit.IsAtEnd()) { mainLo->AddLine(lit.GetLine()); }`
/// followed by `mainLo->Optimize()`, which four of the filters below spell out
/// identically. Here the merge happens inside [`LabelMap::set_line`]'s
/// `add_line`, so there is nothing left to optimize afterwards.
fn add_lines_to_label(map: &mut LabelMap, src: &LabelObject, target: i64) -> Result<()> {
    let dim = map.dimension();
    for line in src.lines() {
        map.set_line(&line.index()[..dim], line.length(), target)?;
    }
    Ok(())
}

/// `itk::AggregateLabelMapFilter` (`itkAggregateLabelMapFilter.hxx:27-60`):
/// collapse every object into the first one.
///
/// "First" is the map's iteration order, which is `std::map`'s — so the
/// **smallest** label wins and every other object's pixels move into it. The
/// output has exactly one object, or none for an empty input.
pub fn aggregate_label_map(map: &LabelMap) -> Result<LabelMap> {
    let mut out = map.clone();
    let labels: Vec<i64> = out.labels().collect();
    let Some((&main, rest)) = labels.split_first() else {
        return Ok(out);
    };
    for &label in rest {
        let object = out
            .remove_label(label)
            .expect("label came from out.labels()");
        add_lines_to_label(&mut out, &object, main)?;
    }
    Ok(out)
}

/// `itk::ChangeLabelLabelMapFilter` (`itkChangeLabelLabelMapFilter.hxx:69-168`):
/// relabel objects, merging collisions.
///
/// `change_map` is a list of `(original, result)` pairs mirroring the yaml's
/// `std::map<double, double>` member, with the same raw-key ordering and
/// last-write-wins overwrite semantics [`crate::change_label`] documents: the
/// pairs are sorted by *raw* key, then each key and value is cast to the map's
/// label type, so `1.2` and `1.4` collapse onto key `1` with `1.4`'s value
/// winning.
///
/// Upstream runs three passes, and the observable semantics all follow from the
/// first one pulling every affected object *out* of the map before any of them
/// is put back:
///
/// - **no chaining.** `{1 -> 2, 2 -> 3}` sends object 1 to label 2 and object 2
///   to label 3. Object 1 is not then re-read as a `2` and forwarded to `3`,
///   matching the scalar [`crate::change_label`].
/// - **collisions merge.** `{1 -> 3, 2 -> 3}` unions both objects' pixels into
///   label 3; so does `{1 -> 5}` when label 5 already exists and is untouched.
/// - **the background can be relabelled.** A `(background, new)` entry moves the
///   background value to `new` (`.hxx:107-127`), destroying any object that
///   already carried `new` — its pixels become background.
/// - **objects relabelled onto the background disappear** (`.hxx:139`), tested
///   against the *new* background, so a `(background, new)` entry silently
///   deletes every object relabelled to `new` in the same call.
pub fn change_label_label_map(map: &LabelMap, change_map: &[(f64, f64)]) -> Result<LabelMap> {
    let id = map.pixel_id();
    let mut pairs: Vec<(f64, f64)> = change_map.to_vec();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut cmap: BTreeMap<i64, i64> = BTreeMap::new();
    for (old, new) in pairs {
        cmap.insert(
            quantize_to_pixel_type(id, old) as i64,
            quantize_to_pixel_type(id, new) as i64,
        );
    }

    let mut out = map.clone();

    // Pass 1: pull every object whose label is a key out of the map.
    let mut pulled: Vec<LabelObject> = Vec::new();
    for &old in cmap.keys() {
        if old != out.background() {
            if let Some(object) = out.remove_label(old) {
                pulled.push(object);
            }
        }
    }

    // Pass 2: move the background, evicting whatever object already sat on the
    // new background value.
    if let Some(&new_background) = cmap.get(&out.background()) {
        if new_background != out.background() {
            out.set_background(new_background)?;
        }
    }

    // Pass 3: put the pulled objects back under their new labels.
    for mut object in pulled {
        let new_label = cmap[&object.label()];
        if new_label == out.background() {
            continue;
        }
        if out.has_label(new_label) {
            add_lines_to_label(&mut out, &object, new_label)?;
        } else {
            object.set_label(new_label);
            out.add_label_object(object)?;
        }
    }
    Ok(out)
}

/// `itk::MergeLabelMapFilter`'s `ChoiceMethodEnum`
/// (`itkMergeLabelMapFilter.h:43-46`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergeLabelMapMethod {
    /// Keep each input object's own label where it is free; renumber the rest
    /// with [`LabelMap::push_label_object`].
    #[default]
    Keep,
    /// Union the pixels of same-labelled objects.
    Aggregate,
    /// Renumber every object consecutively, discarding the input labels.
    Pack,
    /// Refuse any collision.
    Strict,
}

/// `itk::MergeLabelMapFilter` (`itkMergeLabelMapFilter.hxx:32-245`): merge
/// `maps[1..]` into a copy of `maps[0]`.
///
/// The output's geometry, background value and label type are `maps[0]`'s. Every
/// input must share `maps[0]`'s size; upstream relies on `ProcessObject`'s
/// region check for that.
///
/// # Upstream defect in [`MergeLabelMapMethod::Keep`], not reproduced
///
/// `MergeWithKeep` declares its deferred-object deque *outside* the per-input
/// loop (`itkMergeLabelMapFilter.hxx:74`) and drains it *inside*
/// (`:103-108`) without ever clearing it. With three or more inputs, an object
/// deferred while merging input 1 is therefore pushed again while merging input
/// 2: `PushLabelObject` calls `SetLabel(L2)` on the object *already stored* at
/// key `L1` and inserts the same pointer at key `L2`. Upstream's container then
/// holds two keys aliasing one `LabelObject`; `GetNumberOfLabelObjects()` counts
/// both, and `LabelMapToLabelImage` paints the object twice, at `L2` each time,
/// so the pixels end up labelled `L2` and `L1` never appears in the image.
///
/// That aliased state is precisely what this port's `objects[k].label() == k`
/// invariant makes unrepresentable. This port drains the deque once per input,
/// which is what the surrounding code plainly intends: a deferred object gets
/// exactly one new label. The two differ only when there are three or more
/// inputs *and* input 1 contributed a deferred object — where upstream yields
/// the object at `L2` plus a stale alias at `L1`, and this port yields it at
/// `L1`. Pinned by
/// `merge_keep_gives_a_deferred_object_one_label_across_three_inputs`.
pub fn merge_label_map(maps: &[LabelMap], method: MergeLabelMapMethod) -> Result<LabelMap> {
    let Some((first, rest)) = maps.split_first() else {
        return Err(FilterError::EmptyLabelMapList);
    };
    for map in rest {
        if map.size() != first.size() {
            return Err(FilterError::SizeMismatch {
                a: first.size().to_vec(),
                b: map.size().to_vec(),
            });
        }
    }

    let mut out = first.clone();
    match method {
        MergeLabelMapMethod::Keep => {
            let mut deferred: Vec<LabelObject> = Vec::new();
            for map in rest {
                for object in map.label_objects() {
                    if out.background() != object.label() && !out.has_label(object.label()) {
                        out.add_label_object(object.clone())?;
                    } else {
                        deferred.push(object.clone());
                    }
                }
                // Upstream never clears this deque; see the module-level note.
                for object in deferred.drain(..) {
                    out.push_label_object(object)?;
                }
            }
        }
        MergeLabelMapMethod::Strict => {
            for (i, map) in rest.iter().enumerate() {
                for object in map.label_objects() {
                    let label = object.label();
                    if label == out.background() {
                        return Err(FilterError::MergeLabelIsBackground {
                            label,
                            input: i + 1,
                        });
                    }
                    if out.has_label(label) {
                        return Err(FilterError::MergeLabelInUse {
                            label,
                            input: i + 1,
                        });
                    }
                    out.add_label_object(object.clone())?;
                }
            }
        }
        MergeLabelMapMethod::Aggregate => {
            for map in rest {
                for object in map.label_objects() {
                    let label = object.label();
                    if out.has_label(label) {
                        add_lines_to_label(&mut out, object, label)?;
                    } else if label != out.background() {
                        out.add_label_object(object.clone())?;
                    }
                }
            }
        }
        MergeLabelMapMethod::Pack => {
            let objects: Vec<LabelObject> = out.label_objects().cloned().collect();
            out.clear_labels();
            for object in objects {
                out.push_label_object(object)?;
            }
            for map in rest {
                for object in map.label_objects() {
                    out.push_label_object(object.clone())?;
                }
            }
        }
    }
    Ok(out)
}

/// `itk::RelabelLabelMapFilter` = `AttributeRelabelLabelMapFilter` with
/// `LabelLabelObjectAccessor` (`itkRelabelLabelMapFilter.h:45-49`): renumber the
/// objects `0, 1, 2, …`, skipping the background value once.
///
/// The attribute sorted on is the label itself. `AttributeRelabelLabelMapFilter`'s
/// default `Comparator` is *descending* by attribute and its `ReverseComparator`
/// is ascending (`itkAttributeRelabelLabelMapFilter.h:105-127`), so
/// `reverse_ordering = true` — which `RelabelLabelMapFilter`'s constructor
/// forces, and which the yaml exposes as the default — numbers the objects in
/// ascending original-label order.
///
/// Upstream's class docs claim it "reassigns them to the output by calling the
/// PushLabelObject method"; the implementation
/// (`itkAttributeRelabelLabelMapFilter.hxx:64-79`) calls `SetLabel` +
/// `AddLabelObject` against its own counter and never touches `PushLabelObject`.
/// The counter's `if (label == GetBackgroundValue()) ++label` runs each
/// iteration but the counter only rises, so the background is skipped exactly
/// once — the same shape as
/// [`create_consecutive`](crate::label::create_consecutive).
pub fn relabel_label_map(map: &LabelMap, reverse_ordering: bool) -> Result<LabelMap> {
    let mut objects: Vec<LabelObject> = map.label_objects().cloned().collect();
    if reverse_ordering {
        objects.sort_by_key(|o| o.label());
    } else {
        objects.sort_by_key(|o| std::cmp::Reverse(o.label()));
    }

    let mut out = map.clone();
    out.clear_labels();
    let mut label: i64 = 0;
    for mut object in objects {
        if label == out.background() {
            label += 1;
        }
        object.set_label(label);
        out.add_label_object(object)?;
        label += 1;
    }
    Ok(out)
}

/// One line together with the label of the object it came from — upstream's
/// `LineOfLabelObject` (`itkAttributeUniqueLabelMapFilter.h:111-121`), with the
/// object pointer replaced by its label, which identifies it uniquely inside a
/// [`LabelMap`] and is also the only attribute `LabelUniqueLabelMapFilter` reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LineOfLabelObject {
    line: LabelObjectLine,
    label: i64,
}

impl Ord for LineOfLabelObject {
    /// `LineOfLabelObjectComparator` (`itkAttributeUniqueLabelMapFilter.h:123-142`)
    /// orders by the start index alone, slowest axis first — raster order.
    ///
    /// Upstream's comparator reports *equivalence* for two lines sharing a start
    /// index, leaving `std::priority_queue`'s order between them unspecified.
    /// This port breaks that tie by `(length, label)` so the pop order is
    /// deterministic.
    fn cmp(&self, other: &Self) -> Ordering {
        let (a, b) = (self.line.index(), other.line.index());
        for i in (0..MAX_DIM).rev() {
            match a[i].cmp(&b[i]) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
        (self.line.length(), self.label).cmp(&(other.line.length(), other.label))
    }
}

impl PartialOrd for LineOfLabelObject {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// `itk::LabelUniqueLabelMapFilter` = `AttributeUniqueLabelMapFilter` with
/// `LabelLabelObjectAccessor` (`itkLabelUniqueLabelMapFilter.h:46-51`): remove
/// every overlap, keeping one object per pixel.
///
/// All lines of all objects go into one raster-ordered priority queue
/// (`itkAttributeUniqueLabelMapFilter.hxx:29-63`). Each popped line is compared
/// with the previously kept line on the same row; where they overlap, the
/// object with the **larger label** wins when `reverse_ordering` is `false` (the
/// yaml's default), the smaller when it is `true`. The loser is truncated, and
/// any part of it left dangling past the winner is pushed back onto the queue.
///
/// # Upstream defect, reproduced
///
/// The final "remove objects without lines" pass is written
/// `while (it.IsAtEnd())` (`itkAttributeUniqueLabelMapFilter.hxx:239`) where it
/// means `while (!it.IsAtEnd())`, so its body never runs. An object whose pixels
/// were entirely claimed by others therefore survives in the map with **zero
/// lines**, inflating `GetNumberOfLabelObjects()`. This port keeps the empty
/// objects, pinned by
/// `label_unique_keeps_a_fully_overlapped_object_as_an_empty_one`.
pub fn label_unique_label_map(map: &LabelMap, reverse_ordering: bool) -> Result<LabelMap> {
    let dim = map.dimension();

    let mut queue: BinaryHeap<Reverse<LineOfLabelObject>> = BinaryHeap::new();
    let mut objects: BTreeMap<i64, LabelObject> = BTreeMap::new();
    for object in map.label_objects() {
        for &line in object.lines() {
            queue.push(Reverse(LineOfLabelObject {
                line,
                label: object.label(),
            }));
        }
        // `lo->Clear()` — the lines are read back from `kept` at the end.
        objects.insert(object.label(), LabelObject::new(object.label(), dim)?);
    }

    let mut out = map.clone();
    out.clear_labels();

    let mut kept: Vec<LineOfLabelObject> = Vec::new();
    if let Some(Reverse(head)) = queue.pop() {
        kept.push(head);
    }
    while let Some(Reverse(mut current)) = queue.pop() {
        let idx = current.line.index();
        let prev = *kept.last().expect("kept is seeded and never fully drained");
        let prev_idx = prev.line.index();

        if idx[1..] != prev_idx[1..] {
            kept.push(current);
        } else {
            let prev_length = prev.line.length();
            let length = current.line.length();
            if prev_idx[0] + prev_length > idx[0] {
                // Overlap. `attr` is the label, so `attr == prev_attr` would
                // mean two lines of one object overlap, which the optimized
                // invariant forbids; upstream needs the tie-break because an
                // `AttributeUniqueLabelMapFilter` on any other attribute can
                // hit it.
                let keep_current = if current.label > prev.label {
                    !reverse_ordering
                } else {
                    reverse_ordering
                };
                if keep_current {
                    if prev_idx[0] + prev_length > idx[0] + length {
                        // The previous line outlives the current one on the
                        // right; queue its tail.
                        let mut tail = idx;
                        tail[0] = idx[0] + length;
                        queue.push(Reverse(LineOfLabelObject {
                            line: LabelObjectLine::new(
                                &tail[..dim],
                                prev_idx[0] + prev_length - tail[0],
                            )?,
                            label: prev.label,
                        }));
                    }
                    let truncated = idx[0] - prev_idx[0];
                    if truncated != 0 {
                        kept.pop();
                        kept.push(LineOfLabelObject {
                            line: LabelObjectLine::new(&prev_idx[..dim], truncated)?,
                            label: prev.label,
                        });
                    } else {
                        kept.pop();
                    }
                    kept.push(current);
                } else if prev_idx[0] + prev_length < idx[0] + length {
                    // Keep the previous line; requeue the current one's tail.
                    // (A current line the previous fully covers is discarded.)
                    let mut tail = idx;
                    tail[0] = prev_idx[0] + prev_length;
                    current.line = LabelObjectLine::new(&tail[..dim], idx[0] + length - tail[0])?;
                    queue.push(Reverse(current));
                }
            } else {
                kept.push(current);
            }
        }
    }

    for entry in &kept {
        let object = objects
            .get_mut(&entry.label)
            .expect("every label came from the input map");
        object.add_line(&entry.line.index()[..dim], entry.line.length())?;
    }
    // The empty objects go back in too — see the module note on `.hxx:239`.
    for (_, object) in objects {
        out.add_label_object(object)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::Error as CoreError;

    fn labels_of(map: &LabelMap) -> Vec<i64> {
        map.labels().collect()
    }

    /// Every pixel of the map, as a dense label image, for readable assertions.
    fn dense(map: &LabelMap) -> Vec<u32> {
        map.to_label_image()
            .unwrap()
            .scalar_slice::<u32>()
            .unwrap()
            .to_vec()
    }

    // ---- label_image_to_label_map ----------------------------------------

    #[test]
    fn label_image_to_label_map_encodes_runs_and_keeps_the_input_pixel_type() {
        let img = Image::from_vec(&[4, 2], vec![1u16, 1, 0, 2, 2, 2, 2, 0]).unwrap();
        let map = label_image_to_label_map(&img, 0.0).unwrap();
        assert_eq!(labels_of(&map), vec![1, 2]);
        assert_eq!(map.pixel_id(), PixelId::UInt16);
        assert_eq!(map.label_object(1).unwrap().lines().len(), 1);
        assert_eq!(map.label_object(2).unwrap().lines().len(), 2);
        assert_eq!(map.label_object(2).unwrap().size(), 4);
    }

    #[test]
    fn label_image_to_label_map_casts_the_background_to_the_input_pixel_type() {
        // 2.7 truncates to 2 under `static_cast<uint8_t>`, so label 2 is the
        // background and only label 1 survives.
        let img = Image::from_vec(&[3, 1], vec![1u8, 2, 2]).unwrap();
        let map = label_image_to_label_map(&img, 2.7).unwrap();
        assert_eq!(labels_of(&map), vec![1]);
        assert_eq!(map.background(), 2);
    }

    #[test]
    fn label_image_to_label_map_rejects_signed_float_and_vector_images() {
        let signed = Image::from_vec(&[2, 2], vec![0i16; 4]).unwrap();
        assert_eq!(
            label_image_to_label_map(&signed, 0.0),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Int16
            ))
        );
        let float = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            label_image_to_label_map(&float, 0.0),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::Float32
            ))
        );
        let vector = Image::from_vec_vector(&[2, 2], 2, vec![0u8; 8]).unwrap();
        assert_eq!(
            label_image_to_label_map(&vector, 0.0),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::VectorUInt8
            ))
        );
    }

    #[test]
    fn label_image_to_label_map_rejects_a_complex_image() {
        let img = Image::new(&[2, 2], PixelId::ComplexFloat32);
        assert_eq!(
            label_image_to_label_map(&img, 0.0),
            Err(FilterError::RequiresUnsignedIntegerPixelType(
                PixelId::ComplexFloat32
            ))
        );
    }

    #[test]
    fn label_image_to_label_map_rejects_a_four_dimensional_image() {
        let img = Image::from_vec(&[2, 2, 2, 2], vec![0u8; 16]).unwrap();
        assert_eq!(
            label_image_to_label_map(&img, 0.0),
            Err(FilterError::Core(CoreError::UnsupportedLabelMapDimension(
                4
            )))
        );
    }

    // ---- label_map_to_label_image ----------------------------------------

    #[test]
    fn label_map_to_label_image_round_trips() {
        let img = Image::from_vec(&[4, 3], vec![0u8, 1, 1, 0, 0, 0, 2, 2, 3, 3, 0, 0]).unwrap();
        let map = label_image_to_label_map(&img, 0.0).unwrap();
        assert_eq!(label_map_to_label_image(&map).unwrap(), img);
    }

    #[test]
    fn label_map_to_label_image_fills_with_the_maps_background() {
        let img = Image::from_vec(&[3, 1], vec![7u8, 1, 7]).unwrap();
        let map = label_image_to_label_map(&img, 7.0).unwrap();
        assert_eq!(
            label_map_to_label_image(&map)
                .unwrap()
                .scalar_slice::<u8>()
                .unwrap(),
            &[7, 1, 7]
        );
    }

    // ---- binary_image_to_label_map ---------------------------------------

    #[test]
    fn binary_image_to_label_map_labels_face_connected_components_from_one() {
        // 1 . 1
        // 1 . 1
        let img = Image::from_vec(&[3, 2], vec![1u8, 0, 1, 1, 0, 1]).unwrap();
        let (map, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 2);
        assert_eq!(labels_of(&map), vec![1, 2]);
        assert_eq!(map.pixel_id(), PixelId::UInt32);
        assert_eq!(map.background(), 0);
        assert_eq!(dense(&map), vec![1, 0, 2, 1, 0, 2]);
    }

    #[test]
    fn binary_image_to_label_map_full_connectivity_joins_a_diagonal() {
        // 1 .
        // . 1
        let img = Image::from_vec(&[2, 2], vec![1u8, 0, 0, 1]).unwrap();
        let face = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(face.1, 2);

        let settings = BinaryImageToLabelMapSettings {
            fully_connected: true,
            ..Default::default()
        };
        let (map, n) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(n, 1);
        assert_eq!(labels_of(&map), vec![1]);
    }

    #[test]
    fn binary_image_to_label_map_foreground_is_an_equality_test_not_nonzero() {
        // `connected_component` would treat both 1 and 5 as foreground; this
        // filter only accepts pixels equal to `input_foreground_value`.
        let img = Image::from_vec(&[3, 1], vec![1u8, 5, 1]).unwrap();
        let (map, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 2);
        assert_eq!(dense(&map), vec![1, 0, 2]);

        let settings = BinaryImageToLabelMapSettings {
            input_foreground_value: 5.0,
            ..Default::default()
        };
        let (map, n) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(n, 1);
        assert_eq!(dense(&map), vec![0, 1, 0]);
    }

    #[test]
    fn binary_image_to_label_map_casts_the_foreground_to_the_input_pixel_type() {
        // `static_cast<uint8_t>(1.9)` is 1.
        let img = Image::from_vec(&[2, 1], vec![1u8, 0]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            input_foreground_value: 1.9,
            ..Default::default()
        };
        assert_eq!(binary_image_to_label_map(&img, &settings).unwrap().1, 1);
    }

    #[test]
    fn binary_image_to_label_map_numbering_skips_the_output_background_once() {
        // Three components with `output_background_value = 2`: CreateConsecutive
        // hands out 0, 1, then bumps past 2 to 3.
        let img = Image::from_vec(&[5, 1], vec![1u8, 0, 1, 0, 1]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            output_background_value: 2.0,
            ..Default::default()
        };
        let (map, n) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(n, 3);
        assert_eq!(labels_of(&map), vec![0, 1, 3]);
        assert_eq!(map.background(), 2);
        assert_eq!(dense(&map), vec![0, 2, 1, 2, 3]);
    }

    #[test]
    fn binary_image_to_label_map_numbering_starts_at_zero_for_a_non_zero_background() {
        let img = Image::from_vec(&[3, 1], vec![1u8, 0, 1]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            output_background_value: 9.0,
            ..Default::default()
        };
        let (map, _) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(labels_of(&map), vec![0, 1]);
    }

    #[test]
    fn binary_image_to_label_map_negative_background_saturates_to_zero() {
        // `static_cast<uint32_t>(-1.0)` is C++ UB (an out-of-range float→int
        // conversion). This port routes the value through the same
        // `quantize_to_pixel_type`/`Scalar::from_f64` saturating cast every
        // other `pixeltype:` member uses, so it lands on 0.
        let img = Image::from_vec(&[2, 1], vec![1u8, 0]).unwrap();
        let settings = BinaryImageToLabelMapSettings {
            output_background_value: -1.0,
            ..Default::default()
        };
        let (map, _) = binary_image_to_label_map(&img, &settings).unwrap();
        assert_eq!(map.background(), 0);
        assert_eq!(labels_of(&map), vec![1]);
    }

    #[test]
    fn binary_image_to_label_map_on_an_all_background_image_has_no_objects() {
        let img = Image::from_vec(&[3, 2], vec![0u8; 6]).unwrap();
        let (map, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 0);
        assert_eq!(map.number_of_label_objects(), 0);
        assert_eq!(dense(&map), vec![0; 6]);
    }

    #[test]
    fn binary_image_to_label_map_labels_in_raster_order_of_first_appearance() {
        // The right-hand component's first pixel appears before the left-hand
        // one's, so it takes label 1.
        //   . 1
        //   1 1
        let img = Image::from_vec(&[2, 2], vec![0u8, 1, 1, 1]).unwrap();
        let (map, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 1);
        assert_eq!(dense(&map), vec![0, 1, 1, 1]);
    }

    #[test]
    fn binary_image_to_label_map_copies_geometry() {
        let mut img = Image::from_vec(&[2, 2], vec![1u8, 0, 0, 0]).unwrap();
        img.set_spacing(&[0.5, 2.0]).unwrap();
        img.set_origin(&[-1.0, 3.0]).unwrap();
        let (map, _) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(map.spacing(), &[0.5, 2.0]);
        assert_eq!(map.origin(), &[-1.0, 3.0]);
        assert_eq!(map.to_label_image().unwrap().spacing(), &[0.5, 2.0]);
    }

    #[test]
    fn binary_image_to_label_map_rejects_a_float_image() {
        let img = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            binary_image_to_label_map(&img, &Default::default()),
            Err(FilterError::RequiresIntegerPixelType(PixelId::Float32))
        );
    }

    #[test]
    fn binary_image_to_label_map_rejects_a_complex_image() {
        let img = Image::new(&[2, 2], PixelId::ComplexFloat32);
        assert_eq!(
            binary_image_to_label_map(&img, &Default::default()),
            Err(FilterError::RequiresIntegerPixelType(
                PixelId::ComplexFloat32
            ))
        );
    }

    #[test]
    fn binary_image_to_label_map_3d_face_connectivity_crosses_slices() {
        let mut data = vec![0u8; 8];
        data[0] = 1; // (0,0,0)
        data[4] = 1; // (0,0,1)
        let img = Image::from_vec(&[2, 2, 2], data).unwrap();
        let (_, n) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn binary_image_to_label_map_agrees_with_connected_component_on_a_zero_one_image() {
        let img = Image::from_vec(&[4, 3], vec![1u8, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1]).unwrap();
        let (map, _) = binary_image_to_label_map(&img, &Default::default()).unwrap();
        let cc = crate::label::connected_component(&img, false).unwrap();
        assert_eq!(dense(&map), cc.scalar_slice::<u32>().unwrap());
    }

    // ---- shared fixtures for the pure-map filters -------------------------

    /// A 2-D line: `(start index, length)`.
    type Line2 = ([i64; 2], i64);

    /// A 2-D map over `size` whose objects are given as `(label, &[(index, length)])`.
    fn map_of(size: &[usize], background: i64, objects: &[(i64, &[Line2])]) -> LabelMap {
        let mut map = LabelMap::new(size, PixelId::UInt8, background).unwrap();
        for (label, lines) in objects {
            for (index, length) in *lines {
                map.set_line(index, *length, *label).unwrap();
            }
        }
        map
    }

    /// `(label, [(index, length), ...])` for every object, ascending by label.
    fn shape_of(map: &LabelMap) -> Vec<(i64, Vec<Line2>)> {
        map.label_objects()
            .map(|o| {
                (
                    o.label(),
                    o.lines()
                        .iter()
                        .map(|l| ([l.index()[0], l.index()[1]], l.length()))
                        .collect(),
                )
            })
            .collect()
    }

    // ---- aggregate_label_map ---------------------------------------------

    #[test]
    fn aggregate_collapses_every_object_into_the_smallest_label() {
        let map = map_of(
            &[6, 2],
            0,
            &[
                (3, &[([0, 0], 2)]),
                (1, &[([3, 0], 1)]),
                (7, &[([0, 1], 4)]),
            ],
        );
        let out = aggregate_label_map(&map).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![(1, vec![([0, 0], 2), ([3, 0], 1), ([0, 1], 4)])]
        );
        assert_eq!(out.label_object(1).unwrap().size(), 7);
    }

    #[test]
    fn aggregate_merges_the_lines_it_collapses_onto_one_row() {
        // Labels 1 and 2 sit side by side; the collapsed object holds one line.
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 2)]), (2, &[([2, 0], 2)])]);
        let out = aggregate_label_map(&map).unwrap();
        assert_eq!(shape_of(&out), vec![(1, vec![([0, 0], 4)])]);
    }

    #[test]
    fn aggregate_on_an_empty_map_produces_an_empty_map() {
        let map = map_of(&[4, 1], 0, &[]);
        assert_eq!(aggregate_label_map(&map).unwrap(), map);
    }

    #[test]
    fn aggregate_on_a_single_object_is_the_identity() {
        let map = map_of(&[4, 1], 0, &[(5, &[([0, 0], 2)])]);
        assert_eq!(aggregate_label_map(&map).unwrap(), map);
    }

    // ---- change_label_label_map ------------------------------------------

    #[test]
    fn change_label_relabels_and_does_not_chain() {
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)]), (2, &[([2, 0], 1)])]);
        let out = change_label_label_map(&map, &[(1.0, 2.0), (2.0, 3.0)]).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![(2, vec![([0, 0], 1)]), (3, vec![([2, 0], 1)])]
        );
    }

    #[test]
    fn change_label_merges_a_collision() {
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)]), (2, &[([2, 0], 1)])]);
        let out = change_label_label_map(&map, &[(1.0, 3.0), (2.0, 3.0)]).unwrap();
        assert_eq!(shape_of(&out), vec![(3, vec![([0, 0], 1), ([2, 0], 1)])]);
        assert_eq!(out.label_object(3).unwrap().size(), 2);
    }

    #[test]
    fn change_label_merges_into_an_untouched_object() {
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)]), (5, &[([2, 0], 1)])]);
        let out = change_label_label_map(&map, &[(1.0, 5.0)]).unwrap();
        assert_eq!(shape_of(&out), vec![(5, vec![([0, 0], 1), ([2, 0], 1)])]);
    }

    #[test]
    fn change_label_dropping_an_object_onto_the_background_deletes_it() {
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)]), (2, &[([2, 0], 1)])]);
        let out = change_label_label_map(&map, &[(1.0, 0.0)]).unwrap();
        assert_eq!(shape_of(&out), vec![(2, vec![([2, 0], 1)])]);
    }

    #[test]
    fn change_label_moves_the_background_and_destroys_the_object_in_its_way() {
        // Background 0 -> 2. Object 2's pixels become background; object 1 stays.
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)]), (2, &[([2, 0], 1)])]);
        let out = change_label_label_map(&map, &[(0.0, 2.0)]).unwrap();
        assert_eq!(out.background(), 2);
        assert_eq!(shape_of(&out), vec![(1, vec![([0, 0], 1)])]);
    }

    #[test]
    fn change_label_relabelling_onto_the_new_background_deletes_the_object() {
        // {0 -> 2, 1 -> 2}: the background becomes 2, so object 1 -- pulled out in
        // pass 1 and tested against the *new* background in pass 3 -- vanishes.
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)]), (3, &[([2, 0], 1)])]);
        let out = change_label_label_map(&map, &[(0.0, 2.0), (1.0, 2.0)]).unwrap();
        assert_eq!(out.background(), 2);
        assert_eq!(shape_of(&out), vec![(3, vec![([2, 0], 1)])]);
    }

    #[test]
    fn change_label_a_background_entry_mapping_to_itself_changes_nothing() {
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)])]);
        let out = change_label_label_map(&map, &[(0.0, 0.0)]).unwrap();
        assert_eq!(out, map);
    }

    #[test]
    fn change_label_resolves_two_raw_keys_that_truncate_alike_last_wins() {
        // 1.2 and 1.4 both cast to label 1; std::map iterates by raw key, so 1.4's
        // value (3) overwrites 1.2's (2).
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)])]);
        let out = change_label_label_map(&map, &[(1.4, 3.0), (1.2, 2.0)]).unwrap();
        assert_eq!(shape_of(&out), vec![(3, vec![([0, 0], 1)])]);
    }

    #[test]
    fn change_label_with_an_empty_map_is_the_identity() {
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)])]);
        assert_eq!(change_label_label_map(&map, &[]).unwrap(), map);
    }

    #[test]
    fn change_label_ignores_a_label_that_is_not_present() {
        let map = map_of(&[4, 1], 0, &[(1, &[([0, 0], 1)])]);
        let out = change_label_label_map(&map, &[(9.0, 4.0)]).unwrap();
        assert_eq!(out, map);
    }

    // ---- merge_label_map -------------------------------------------------

    #[test]
    fn merge_keep_renumbers_only_the_colliding_objects() {
        let a = map_of(&[8, 1], 0, &[(1, &[([0, 0], 1)]), (2, &[([1, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(2, &[([2, 0], 1)]), (5, &[([3, 0], 1)])]);
        let out = merge_label_map(&[a, b], MergeLabelMapMethod::Keep).unwrap();
        // 5 is free, so it is kept; 2 collides and PushLabelObject gives it
        // last + 1 == 6.
        assert_eq!(
            shape_of(&out),
            vec![
                (1, vec![([0, 0], 1)]),
                (2, vec![([1, 0], 1)]),
                (5, vec![([3, 0], 1)]),
                (6, vec![([2, 0], 1)]),
            ]
        );
    }

    #[test]
    fn merge_keep_gives_a_deferred_object_one_label_across_three_inputs() {
        // Upstream's never-cleared deque (itkMergeLabelMapFilter.hxx:74, :103-108)
        // would push b's object again while merging c: it would end at label 7
        // with a stale alias at 6, and GetNumberOfLabelObjects() would report 5.
        // Here it is pushed once, at 6.
        let a = map_of(&[8, 1], 0, &[(1, &[([0, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(1, &[([1, 0], 1)]), (5, &[([2, 0], 1)])]);
        let c = map_of(&[8, 1], 0, &[(3, &[([3, 0], 1)])]);
        let out = merge_label_map(&[a, b, c], MergeLabelMapMethod::Keep).unwrap();
        assert_eq!(out.number_of_label_objects(), 4);
        assert_eq!(
            shape_of(&out),
            vec![
                (1, vec![([0, 0], 1)]),
                (3, vec![([3, 0], 1)]),
                (5, vec![([2, 0], 1)]),
                (6, vec![([1, 0], 1)]),
            ]
        );
    }

    #[test]
    fn merge_keep_defers_an_object_whose_label_is_the_background() {
        let a = map_of(&[8, 1], 7, &[(1, &[([0, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(7, &[([1, 0], 1)])]);
        let out = merge_label_map(&[a, b], MergeLabelMapMethod::Keep).unwrap();
        assert_eq!(out.background(), 7);
        // PushLabelObject: last == 1, last + 1 == 2 != background.
        assert_eq!(
            shape_of(&out),
            vec![(1, vec![([0, 0], 1)]), (2, vec![([1, 0], 1)])]
        );
    }

    #[test]
    fn merge_aggregate_unions_same_labelled_objects() {
        let a = map_of(&[8, 1], 0, &[(1, &[([0, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(1, &[([1, 0], 1)]), (2, &[([4, 0], 1)])]);
        let out = merge_label_map(&[a, b], MergeLabelMapMethod::Aggregate).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![(1, vec![([0, 0], 2)]), (2, vec![([4, 0], 1)])]
        );
    }

    #[test]
    fn merge_aggregate_drops_an_object_whose_label_is_the_background() {
        let a = map_of(&[8, 1], 7, &[(1, &[([0, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(7, &[([1, 0], 1)])]);
        let out = merge_label_map(&[a, b], MergeLabelMapMethod::Aggregate).unwrap();
        assert_eq!(shape_of(&out), vec![(1, vec![([0, 0], 1)])]);
    }

    #[test]
    fn merge_pack_renumbers_everything_consecutively() {
        let a = map_of(&[8, 1], 0, &[(4, &[([0, 0], 1)]), (9, &[([1, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(4, &[([2, 0], 1)])]);
        let out = merge_label_map(&[a, b], MergeLabelMapMethod::Pack).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![
                (1, vec![([0, 0], 1)]),
                (2, vec![([1, 0], 1)]),
                (3, vec![([2, 0], 1)]),
            ]
        );
    }

    #[test]
    fn merge_pack_starts_at_zero_when_the_background_is_not_zero() {
        // PushLabelObject on an empty map: background != 0 -> label 0.
        let a = map_of(&[8, 1], 7, &[(4, &[([0, 0], 1)]), (9, &[([1, 0], 1)])]);
        let out = merge_label_map(&[a], MergeLabelMapMethod::Pack).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![(0, vec![([0, 0], 1)]), (1, vec![([1, 0], 1)])]
        );
    }

    #[test]
    fn merge_pack_skips_the_background_while_numbering() {
        // background 2. Empty map -> 0; then last + 1 == 1; then last + 1 == 2 is
        // the background, so PushLabelObject's second branch gives last + 2 == 3.
        let a = map_of(
            &[8, 1],
            2,
            &[
                (4, &[([0, 0], 1)]),
                (9, &[([1, 0], 1)]),
                (11, &[([2, 0], 1)]),
            ],
        );
        let out = merge_label_map(&[a], MergeLabelMapMethod::Pack).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![
                (0, vec![([0, 0], 1)]),
                (1, vec![([1, 0], 1)]),
                (3, vec![([2, 0], 1)]),
            ]
        );
    }

    #[test]
    fn merge_strict_rejects_a_reused_label() {
        let a = map_of(&[8, 1], 0, &[(1, &[([0, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(1, &[([1, 0], 1)])]);
        assert_eq!(
            merge_label_map(&[a, b], MergeLabelMapMethod::Strict),
            Err(FilterError::MergeLabelInUse { label: 1, input: 1 })
        );
    }

    #[test]
    fn merge_strict_rejects_the_output_background_as_a_label() {
        let a = map_of(&[8, 1], 7, &[(1, &[([0, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(7, &[([1, 0], 1)])]);
        assert_eq!(
            merge_label_map(&[a, b], MergeLabelMapMethod::Strict),
            Err(FilterError::MergeLabelIsBackground { label: 7, input: 1 })
        );
    }

    #[test]
    fn merge_strict_accepts_disjoint_labels() {
        let a = map_of(&[8, 1], 0, &[(1, &[([0, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(2, &[([1, 0], 1)])]);
        let out = merge_label_map(&[a, b], MergeLabelMapMethod::Strict).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![(1, vec![([0, 0], 1)]), (2, vec![([1, 0], 1)])]
        );
    }

    #[test]
    fn merge_keeps_the_first_maps_background_and_geometry() {
        let a = map_of(&[8, 1], 7, &[(1, &[([0, 0], 1)])]);
        let b = map_of(&[8, 1], 0, &[(2, &[([1, 0], 1)])]);
        let out = merge_label_map(&[a, b], MergeLabelMapMethod::Keep).unwrap();
        assert_eq!(out.background(), 7);
        assert_eq!(out.size(), &[8, 1]);
    }

    #[test]
    fn merge_rejects_an_empty_input_list_and_a_size_mismatch() {
        assert_eq!(
            merge_label_map(&[], MergeLabelMapMethod::Keep),
            Err(FilterError::EmptyLabelMapList)
        );
        let a = map_of(&[8, 1], 0, &[]);
        let b = map_of(&[4, 1], 0, &[]);
        assert_eq!(
            merge_label_map(&[a, b], MergeLabelMapMethod::Keep),
            Err(FilterError::SizeMismatch {
                a: vec![8, 1],
                b: vec![4, 1]
            })
        );
    }

    #[test]
    fn merge_a_single_map_is_the_identity_for_keep_aggregate_and_strict() {
        let a = map_of(&[8, 1], 0, &[(4, &[([0, 0], 1)])]);
        for method in [
            MergeLabelMapMethod::Keep,
            MergeLabelMapMethod::Aggregate,
            MergeLabelMapMethod::Strict,
        ] {
            assert_eq!(
                merge_label_map(std::slice::from_ref(&a), method).unwrap(),
                a
            );
        }
    }

    // ---- relabel_label_map -----------------------------------------------

    #[test]
    fn relabel_numbers_ascending_by_original_label_when_reversed() {
        // ReverseOrdering = true is RelabelLabelMapFilter's (and the yaml's) default.
        let map = map_of(
            &[8, 1],
            0,
            &[
                (4, &[([0, 0], 1)]),
                (9, &[([1, 0], 1)]),
                (6, &[([2, 0], 1)]),
            ],
        );
        let out = relabel_label_map(&map, true).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![
                (1, vec![([0, 0], 1)]), // was 4
                (2, vec![([2, 0], 1)]), // was 6
                (3, vec![([1, 0], 1)]), // was 9
            ]
        );
    }

    #[test]
    fn relabel_numbers_descending_by_original_label_when_not_reversed() {
        let map = map_of(
            &[8, 1],
            0,
            &[
                (4, &[([0, 0], 1)]),
                (9, &[([1, 0], 1)]),
                (6, &[([2, 0], 1)]),
            ],
        );
        let out = relabel_label_map(&map, false).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![
                (1, vec![([1, 0], 1)]), // was 9
                (2, vec![([2, 0], 1)]), // was 6
                (3, vec![([0, 0], 1)]), // was 4
            ]
        );
    }

    #[test]
    fn relabel_starts_at_zero_and_skips_a_non_zero_background_exactly_once() {
        let map = map_of(
            &[8, 1],
            1,
            &[
                (4, &[([0, 0], 1)]),
                (9, &[([1, 0], 1)]),
                (6, &[([2, 0], 1)]),
            ],
        );
        let out = relabel_label_map(&map, true).unwrap();
        assert_eq!(out.labels().collect::<Vec<_>>(), vec![0, 2, 3]);
    }

    #[test]
    fn relabel_on_an_empty_map_produces_an_empty_map() {
        let map = map_of(&[8, 1], 0, &[]);
        assert_eq!(relabel_label_map(&map, true).unwrap(), map);
    }

    // ---- label_unique_label_map ------------------------------------------

    #[test]
    fn label_unique_gives_the_overlap_to_the_larger_label_by_default() {
        // 1: [0, 4)   2: [2, 6)
        let map = map_of(&[8, 1], 0, &[(1, &[([0, 0], 4)]), (2, &[([2, 0], 4)])]);
        let out = label_unique_label_map(&map, false).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![(1, vec![([0, 0], 2)]), (2, vec![([2, 0], 4)])]
        );
    }

    #[test]
    fn label_unique_gives_the_overlap_to_the_smaller_label_when_reversed() {
        let map = map_of(&[8, 1], 0, &[(1, &[([0, 0], 4)]), (2, &[([2, 0], 4)])]);
        let out = label_unique_label_map(&map, true).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![(1, vec![([0, 0], 4)]), (2, vec![([4, 0], 2)])]
        );
    }

    #[test]
    fn label_unique_splits_a_line_straddled_by_a_shorter_winner() {
        // 1: [0, 6)   2: [2, 2)  -- 2 wins the middle, 1 keeps both flanks.
        let map = map_of(&[8, 1], 0, &[(1, &[([0, 0], 6)]), (2, &[([2, 0], 2)])]);
        let out = label_unique_label_map(&map, false).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![(1, vec![([0, 0], 2), ([4, 0], 2)]), (2, vec![([2, 0], 2)]),]
        );
        assert_eq!(out.label_object(1).unwrap().size(), 4);
    }

    #[test]
    fn label_unique_keeps_a_fully_overlapped_object_as_an_empty_one() {
        // Upstream's `while (it.IsAtEnd())` (itkAttributeUniqueLabelMapFilter.hxx:239)
        // never removes it; neither do we.
        let map = map_of(&[8, 1], 0, &[(1, &[([2, 0], 2)]), (2, &[([0, 0], 6)])]);
        let out = label_unique_label_map(&map, false).unwrap();
        assert_eq!(out.number_of_label_objects(), 2);
        assert!(out.label_object(1).unwrap().is_empty());
        assert_eq!(shape_of(&out)[1], (2, vec![([0, 0], 6)]));
    }

    #[test]
    fn label_unique_leaves_non_overlapping_objects_alone() {
        let map = map_of(&[8, 2], 0, &[(1, &[([0, 0], 2)]), (2, &[([4, 1], 2)])]);
        assert_eq!(label_unique_label_map(&map, false).unwrap(), map);
    }

    #[test]
    fn label_unique_never_merges_across_rows() {
        // Same x-range, different y: no overlap.
        let map = map_of(&[8, 2], 0, &[(1, &[([0, 0], 4)]), (2, &[([0, 1], 4)])]);
        assert_eq!(label_unique_label_map(&map, false).unwrap(), map);
    }

    #[test]
    fn label_unique_resolves_a_three_way_overlap_by_label() {
        // 1: [0, 6)  2: [1, 4)  3: [2, 2)
        let map = map_of(
            &[8, 1],
            0,
            &[
                (1, &[([0, 0], 6)]),
                (2, &[([1, 0], 4)]),
                (3, &[([2, 0], 2)]),
            ],
        );
        let out = label_unique_label_map(&map, false).unwrap();
        assert_eq!(
            shape_of(&out),
            vec![
                (1, vec![([0, 0], 1), ([5, 0], 1)]),
                (2, vec![([1, 0], 1), ([4, 0], 1)]),
                (3, vec![([2, 0], 2)]),
            ]
        );
        // Every pixel is claimed exactly once.
        assert_eq!(
            out.label_objects().map(|o| o.size()).sum::<u64>(),
            6,
            "the union covers [0, 6) with no double coverage"
        );
    }

    #[test]
    fn label_unique_pixel_count_is_the_union_of_the_inputs() {
        let map = map_of(&[8, 1], 0, &[(1, &[([0, 0], 5)]), (2, &[([3, 0], 5)])]);
        let out = label_unique_label_map(&map, false).unwrap();
        assert_eq!(out.label_objects().map(|o| o.size()).sum::<u64>(), 8);
    }

    #[test]
    fn label_unique_on_an_empty_map_produces_an_empty_map() {
        let map = map_of(&[8, 1], 0, &[]);
        assert_eq!(label_unique_label_map(&map, false).unwrap(), map);
    }

    // ---- push_label_object numbering (exercised through pack) -------------

    #[test]
    fn push_label_object_fills_a_gap_below_the_first_label_at_the_type_ceiling() {
        // last == u8::MAX, so branches 2 and 3 are unreachable; first - 1 == 3.
        let mut map = LabelMap::new(&[8, 1], PixelId::UInt8, 0).unwrap();
        map.set_line(&[0, 0], 1, 4).unwrap();
        map.set_line(&[1, 0], 1, 255).unwrap();
        let mut object = LabelObject::new(9, 2).unwrap();
        object.add_line(&[2, 0], 1).unwrap();
        map.push_label_object(object).unwrap();
        assert_eq!(map.labels().collect::<Vec<_>>(), vec![3, 4, 255]);
    }
}
