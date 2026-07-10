//! `LabelSetDilateImageFilter` and `LabelSetErodeImageFilter`: Beare's
//! separable, per-label morphology by parabolic distance transform.
//!
//! Ported from `ITK/Modules/Filtering/LabelErodeDilate/include/` —
//! `itkLabelSetMorphBaseImageFilter.h(.hxx)` (the shared driver),
//! `itkLabelSetDilateImageFilter.hxx`, `itkLabelSetErodeImageFilter.hxx`, and
//! `itkLabelSetUtils.h` (the four line routines that do all the work).
//!
//! # How it works
//!
//! Both filters run one 1-D pass per axis (`itkLabelSetMorphBaseImageFilter.hxx:364-375`)
//! over a `float` *distance image* — `DistanceImageType = Image<RealType, D>`
//! with `RealType = NumericTraits<PixelType>::FloatType`, and `FloatType` is
//! `float` for every integer pixel type (`itkNumericTraits.h:617-619`, and the
//! base class's own comment at `.h:106-108`: "RealType is usually 'double' in
//! NumericTraits. Here we prefer float in order to save memory."). **All the
//! parabolic arithmetic below therefore happens in `f32`**, including the exact
//! `lineBuf[j] == BaseSigma` equality test that decides erosion's output. This
//! port matches that precision element for element.
//!
//! `GenerateData` (`.hxx:300-376`) turns the per-axis radius into a scale:
//!
//! ```text
//! use_image_spacing:  scale[p] = 0.5 * r[p]^2
//! otherwise:          scale[p] = 0.5 * r[p]^2 + 1      // "a little bit of a margin"
//!
//! firstval  = first p with r[p] != 0   (0 if every radius is 0)
//! BaseSigma = scale[firstval]
//! scale[p] /= scale[firstval]   for p > firstval        // elliptical support
//! ```
//!
//! Then axes run in order `0..D`, skipping any axis whose `scale` is `0`. The
//! first axis that *does* run takes a specialised "first pass" that builds the
//! distance image straight from the binary label mask; later axes run the
//! general contact-point parabola.
//!
//! ## Dilation
//!
//! First pass (`doOneDimensionDilateFirstPass`, `itkLabelSetUtils.h:371-453`)
//! seeds `lineBuf[i] = sigma` on labelled pixels and `0` on background, then
//! `DoLineDilateFirstPass` (`.h:57-117`) propagates the parabola
//! `h(k) = sigma - magnitude * k^2` outward, carrying the label with it.
//! Later passes (`doOneDimensionDilate`, `.h:596-678`) run
//! `DoLineLabelProp<..., true>` (`.h:165-231`) on the distance image, reading
//! labels from the *output* image built by the previous pass.
//!
//! Background is a plateau at `0`, so a label reaches a pixel only while its
//! parabola stays **strictly** above `0`. With `use_image_spacing` that means
//! `sum_p (s_p k_p / r_p)^2 < 1` — an *open* ellipsoid. A voxel exactly `r`
//! away in physical units is **not** dilated into.
//!
//! ## Erosion
//!
//! First pass (`doOneDimensionErodeFirstPass`, `.h:233-369`) run-length encodes
//! the line into maximal runs of one label value and, for each run, writes
//! `min(sigma, magnitude*(distance to just past the run's end)^2)` from either
//! side (`DoLineErodeFirstPass`, `.h:29-55`). A run that touches the image
//! border gets `sigma` for that end, so the border never erodes. Later passes
//! (`doOneDimensionErode`, `.h:455-594`) re-derive the runs from the
//! **original input** labels — never from the output — and run `DoLine<..., false>`
//! (`.h:119-163`) over the run padded by one "outside" cell on each side.
//!
//! The output is written only on the *last* axis (`lastpass`): a pixel keeps
//! its label exactly when its accumulated distance is still pinned at the cap,
//! `lineBuf[j] == BaseSigma` (`.h:578-581`). That means it survives iff no
//! pixel of a different label (or background, or a different run of the *same*
//! label) lies strictly inside the same open ellipsoid — which is why the
//! filter separates touching labels.
//!
//! # Upstream findings
//!
//! 1. **`SimpleITK`'s `UseImageSpacing` default disagrees with ITK's.**
//!    `LabelSetMorphBaseImageFilter`'s constructor sets
//!    `m_UseImageSpacing = false` (`.hxx:206`), but both
//!    `LabelSetDilateImageFilter.yaml` and `LabelSetErodeImageFilter.yaml`
//!    declare `default: 'true'` and the generated wrapper always calls the
//!    setter. This port follows the yaml, since that is what a SimpleITK
//!    caller observes.
//!
//! 2. **The SimpleITK default is an identity on unit-spacing images.** With
//!    `Radius = (1,1,1)` and `UseImageSpacing = true`, `BaseSigma = 0.5` and
//!    the first-pass `magnitude` is `spacing^2 / 2 = 0.5`, so a neighbour one
//!    voxel away sits at parabola height `0.5 - 0.5 = 0` exactly — and the
//!    strict `thisval > lineBuf[pos]` (`.h:82`) / `T >= baseVal` with
//!    `krange == 0` evaluated last (`.h:190-195`) both hand the tie to the
//!    pixel's own value. Dilation adds nothing; symmetrically, erosion removes
//!    nothing (`lineBuf[j] == BaseSigma` still holds one pixel in from the
//!    boundary). ITK's own tests only ever use radius 3, 5 and 41
//!    (`Modules/Filtering/LabelErodeDilate/test/CMakeLists.txt`), so this is
//!    untested upstream. The `+ 1` margin of the non-spacing branch
//!    (`.hxx:329`) exists precisely to avoid it. Pinned by
//!    [`tests::spacing_mode_radius_one_is_an_identity`].
//!
//! 3. **`firstval` and "the first pass" are not the same axis.** `firstval` is
//!    the first *nonzero radius* (`.hxx:339-347`), but the first axis actually
//!    processed is the first with `scale[d] > 0` (`.hxx:364-375` and
//!    `itkLabelSetDilateImageFilter.hxx:196`). In the non-spacing branch every
//!    `scale[d]` is `>= 1`, so axis `0` always runs — even when `r[0] == 0`.
//!    Consequences, both reproduced here:
//!    - Dilation with `Radius = (0, k)` and `UseImageSpacing = false` still
//!      dilates one voxel along axis 0, because that axis runs the first pass
//!      with `sigma = scale[0] = 1` and `magnitude = 0.5`.
//!    - Erosion with the same settings caps the distance image at `sigma = 1`
//!      while the last pass compares against `BaseSigma = 0.5*k^2 + 1 > 1`, so
//!      **the output is entirely background** for any `k >= 1`. Pinned by
//!      [`tests::erode_zero_leading_radius_without_spacing_blanks_the_output`].
//!
//! 4. **Erosion can leave the output image uninitialized.** The label output is
//!    written only when `lastpass` is true (`itkLabelSetUtils.h:345`, `:572`),
//!    i.e. on axis `D-1` — but that axis is skipped when `scale[D-1] == 0`,
//!    which `UseImageSpacing = true` produces for `r[D-1] == 0`.
//!    `LabelSetMorphBaseImageFilter::GenerateData` only calls
//!    `AllocateOutputs()` (`.hxx:309`), which does not fill, so upstream reads
//!    uninitialized memory. Likewise, if *no* axis runs (every radius `0`,
//!    spacing mode) neither filter writes the output at all. C++ UB; this port
//!    defines both cases as all-background (`0`), the value-initialized pixel.
//!    Pinned by [`tests::erode_zero_trailing_radius_with_spacing_yields_background`].
//!
//! 5. **`doOneDimensionErodeFirstPass` never clears `lineBuf` at background
//!    positions.** `lineBuf` is allocated once per axis (`.h:260`) and the
//!    scan-line copy only writes labelled positions —
//!    `if (labBuf[i]) { lineBuf[i] = 1.0; }` with no `else`
//!    (`.h:281-288`), unlike the dilate twin which has one (`.h:424-430`). So
//!    a background pixel's distance is inherited from the *previous scan
//!    line*, and `outputIterator.Set(lineBuf[j++])` (`.h:341`) writes that
//!    stale value into the distance image. It is benign: later passes only
//!    read the distance image inside runs, and the label output is gated on
//!    `labBuf[j2]`, which is `0` there. Reproduced anyway, so the distance
//!    image matches upstream byte for byte.
//!
//! 6. **Run detection compares labels as `float`.** `RealType val = labBuf[idx]`
//!    (`.h:297`, `:516`) narrows the label to `float` before
//!    `val != labBuf[idxend]` promotes each subsequent label back to `float`.
//!    Two distinct labels that round to the same `float` — e.g. `16777216` and
//!    `16777217` for `Int32` — merge into one run and erode as a single
//!    object. Reproduced by [`same_run`].
//!
//! 7. **`DoLineDilateFirstPass`'s right pass reads the *original* labels at
//!    the contact point** (`labBuf[lastcontact]`, `.h:108`), not the labels the
//!    left pass just propagated (`NewLabBuf`). The contact point is always a
//!    local maximum of `tmpLineBuf`, which for a binary seeding can only sit on
//!    a labelled pixel, so the two agree; the code is fragile rather than
//!    wrong.
//!
//! # Tie-breaking under dilation
//!
//! Two labels equidistant from a pixel: `DoLineLabelProp`'s negative half scans
//! `krange` **upward** to `0` and its positive half scans **downward** to `0`,
//! and the comparison is `T >= baseVal` — so the *last* candidate examined wins
//! a tie, and that is always `krange == 0`. In the negative half that means the
//! nearest left neighbour; in the positive half it means `tmpLabelBuf[pos]`,
//! i.e. whatever the negative half decided. **The lower index along the current
//! axis wins**, regardless of which label value is numerically larger.
//! `DoLineDilateFirstPass` reaches the same rule through strict `>` tests
//! (`.h:82`, `.h:106`). Pinned by [`tests::dilate_tie_goes_to_the_lower_index`].

use crate::error::{FilterError, Result};
use crate::image_from_f64;
use crate::logic::require_integer_pixel_type;
use sitk_core::Image;

/// `RealType val = labBuf[idx]; ... val != labBuf[idxend]` — the run-detection
/// comparison of `itkLabelSetUtils.h:297,304` and `:516,523`, in `f32`.
/// See upstream finding 6.
fn same_run(a: f64, b: f64) -> bool {
    a as f32 == b as f32
}

/// `DoLineErodeFirstPass` (`itkLabelSetUtils.h:29-55`), over one run.
fn do_line_erode_first_pass(
    buf: &mut [f32],
    leftend: f32,
    rightend: f32,
    magnitude: f32,
    sigma: f32,
) {
    let line_length = buf.len();
    for (pos, b) in buf.iter_mut().enumerate() {
        let offset = (line_length - pos) as f32;
        let from_left = (pos + 1) as f32;
        let left = leftend - magnitude * from_left * from_left;
        let right = rightend - magnitude * offset * offset;
        *b = left.min(right).min(sigma);
    }
}

/// `DoLineDilateFirstPass` (`itkLabelSetUtils.h:57-117`).
fn do_line_dilate_first_pass(
    line_buf: &mut [f32],
    tmp_line_buf: &mut [f32],
    lab_buf: &[f64],
    new_lab_buf: &mut [f64],
    magnitude: f32,
) {
    let line_length = line_buf.len();

    let mut lastcontact = 0usize;
    let mut lastval = line_buf[0];
    for pos in 0..line_length {
        let krange = (pos - lastcontact) as f32;
        let thisval = lastval - magnitude * krange * krange;

        if line_buf[pos] >= line_buf[lastcontact] {
            lastcontact = pos;
            lastval = line_buf[pos];
        }
        tmp_line_buf[pos] = line_buf[pos].max(thisval);
        new_lab_buf[pos] = if thisval > line_buf[pos] {
            lab_buf[lastcontact]
        } else {
            lab_buf[pos]
        };
    }

    let mut lastcontact = line_length - 1;
    let mut lastval = tmp_line_buf[lastcontact];
    for pos in (0..line_length).rev() {
        let krange = (lastcontact - pos) as f32;
        let thisval = lastval - magnitude * krange * krange;

        if tmp_line_buf[pos] >= tmp_line_buf[lastcontact] {
            lastcontact = pos;
            lastval = tmp_line_buf[pos];
        }
        line_buf[pos] = tmp_line_buf[pos].max(thisval);
        // No `else` branch upstream: the left pass already filled every slot.
        if thisval > tmp_line_buf[pos] {
            new_lab_buf[pos] = lab_buf[lastcontact];
        }
    }
}

/// `DoLine<..., false>` (`itkLabelSetUtils.h:119-163`), the contact-point
/// parabolic *erosion*. Only the `doDilate == false` instantiation exists
/// upstream; dilation goes through [`do_line_label_prop`] instead.
fn do_line_erode(line_buf: &mut [f32], tmp_line_buf: &mut [f32], magnitude: f32, extreme: f32) {
    let line_length = line_buf.len() as isize;

    let mut koffset: isize = 0;
    let mut newcontact: isize = 0;
    for pos in 0..line_length {
        let mut base_val = extreme;
        let mut krange = koffset;
        while krange <= 0 {
            let k = krange as f32;
            let t = line_buf[(pos + krange) as usize] - magnitude * k * k;
            if t <= base_val {
                base_val = t;
                newcontact = krange;
            }
            krange += 1;
        }
        tmp_line_buf[pos as usize] = base_val;
        koffset = newcontact - 1;
    }

    koffset = 0;
    newcontact = 0;
    for pos in (0..line_length).rev() {
        let mut base_val = extreme;
        let mut krange = koffset;
        while krange >= 0 {
            let k = krange as f32;
            let t = tmp_line_buf[(pos + krange) as usize] - magnitude * k * k;
            if t <= base_val {
                base_val = t;
                newcontact = krange;
            }
            krange -= 1;
        }
        line_buf[pos as usize] = base_val;
        koffset = newcontact + 1;
    }
}

/// `DoLineLabelProp<..., true>` (`itkLabelSetUtils.h:165-231`), the
/// contact-point parabolic *dilation* with label propagation. Only the
/// `doDilate == true` instantiation exists upstream.
fn do_line_label_prop(
    line_buf: &mut [f32],
    tmp_line_buf: &mut [f32],
    label_buf: &mut [f64],
    tmp_label_buf: &mut [f64],
    magnitude: f32,
    extreme: f32,
) {
    let line_length = line_buf.len() as isize;

    let mut koffset: isize = 0;
    let mut newcontact: isize = 0;
    for pos in 0..line_length {
        let mut base_val = extreme;
        let mut base_lab = label_buf[pos as usize];
        let mut krange = koffset;
        // Ascending to 0, so `krange == 0` is examined last and takes ties.
        while krange <= 0 {
            let k = krange as f32;
            let t = line_buf[(pos + krange) as usize] - magnitude * k * k;
            if t >= base_val {
                base_val = t;
                newcontact = krange;
                base_lab = label_buf[(pos + krange) as usize];
            }
            krange += 1;
        }
        tmp_line_buf[pos as usize] = base_val;
        tmp_label_buf[pos as usize] = base_lab;
        koffset = newcontact - 1;
    }

    koffset = 0;
    newcontact = 0;
    for pos in (0..line_length).rev() {
        let mut base_val = extreme;
        let mut base_lab = tmp_label_buf[pos as usize];
        let mut krange = koffset;
        // Descending to 0, so `krange == 0` — the negative half's answer —
        // takes ties against anything to the right.
        while krange >= 0 {
            let k = krange as f32;
            let t = tmp_line_buf[(pos + krange) as usize] - magnitude * k * k;
            if t >= base_val {
                base_val = t;
                newcontact = krange;
                base_lab = tmp_label_buf[(pos + krange) as usize];
            }
            krange -= 1;
        }
        line_buf[pos as usize] = base_val;
        label_buf[pos as usize] = base_lab;
        koffset = newcontact + 1;
    }
}

/// Maximal runs of one nonzero label value, `[first, last]` inclusive
/// (`itkLabelSetUtils.h:295-313`, `:514-532`).
fn label_runs(lab_buf: &[f64]) -> Vec<(usize, usize)> {
    let line_length = lab_buf.len();
    let mut runs = Vec::new();
    let mut idx = 0;
    while idx < line_length {
        let val = lab_buf[idx];
        if val as f32 != 0.0 {
            let mut idxend = idx;
            while idxend < line_length && same_run(val, lab_buf[idxend]) {
                idxend += 1;
            }
            runs.push((idx, idxend - 1));
            idx = idxend - 1;
        }
        idx += 1;
    }
    runs
}

/// The linear index of the first pixel of every line running along `axis`,
/// in the order `ImageLinearIteratorWithIndex::NextLine` visits them (an
/// odometer over the other axes, lowest axis fastest).
fn line_starts(size: &[usize], strides: &[usize], axis: usize, n_pixels: usize) -> Vec<usize> {
    (0..n_pixels)
        .filter(|lin| (lin / strides[axis]) % size[axis] == 0)
        .collect()
}

/// Shared driver of `LabelSetMorphBaseImageFilter::GenerateData`
/// (`.hxx:300-376`) plus the dilate/erode `ThreadedGenerateData` bodies.
fn label_set_morph(
    img: &Image,
    radius: &[usize],
    use_image_spacing: bool,
    dilate: bool,
) -> Result<Image> {
    require_integer_pixel_type(img)?;
    let dim = img.dimension();
    if radius.len() != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: radius.len(),
        });
    }

    let labels = img.to_f64_vec()?;
    let size = img.size().to_vec();
    let spacing = img.spacing().to_vec();
    let n_pixels = labels.len();

    let mut strides = vec![1usize; dim];
    for d in 1..dim {
        strides[d] = strides[d - 1] * size[d - 1];
    }

    // .hxx:316-331 — radius to scale.
    let mut scale = vec![0.0f64; dim];
    for d in 0..dim {
        let r = radius[d] as f64;
        scale[d] = if use_image_spacing {
            0.5 * r * r
        } else {
            0.5 * r * r + 1.0
        };
    }
    // .hxx:339-352 — the first nonzero *radius* normalizes the later axes.
    let firstval = radius.iter().position(|&r| r != 0).unwrap_or(0);
    let first_scale = scale[firstval];
    for s in scale.iter_mut().skip(firstval + 1) {
        *s /= first_scale;
    }
    let base_sigma = first_scale as f32;

    // .hxx:196-205 — `m_Extreme` and `m_MagnitudeSign`. `NonpositiveMin` for a
    // float is `-max`.
    let extreme = if dilate { -f32::MAX } else { f32::MAX };
    let magnitude_sign = if dilate { 1.0f32 } else { -1.0f32 };

    let mut dist = vec![0.0f32; n_pixels];
    // `AllocateOutputs()` does not fill; see upstream finding 4. Defined as 0.
    let mut out = vec![0.0f64; n_pixels];
    let mut first_pass_done = false;

    for axis in 0..dim {
        // `.hxx:368` / `itkLabelSetDilateImageFilter.hxx:196` — a *positive*
        // test, not `<= 0`. When every radius is 0 in spacing mode,
        // `scale[firstval]` is 0 and the normalization above leaves the later
        // axes at `0.0 / 0.0 = NaN`, which fails `> 0` (and would pass a
        // negated `<= 0`).
        if scale[axis] > 0.0 {
            let sigma = scale[axis] as f32;
            let iscale = if use_image_spacing {
                spacing[axis] as f32
            } else {
                1.0f32
            };
            let last_pass = axis == dim - 1;
            let line_length = size[axis];
            let stride = strides[axis];
            let starts = line_starts(&size, &strides, axis, n_pixels);

            // `(magnitudeSign * iscale * iscale)` is a `float`; the division is by
            // a `double` (`2.0` / `2.0 * sigma`) and narrows back on assignment.
            let numerator = (magnitude_sign * iscale * iscale) as f64;
            let magnitude = if first_pass_done {
                (numerator / (2.0 * sigma as f64)) as f32
            } else {
                (numerator / 2.0) as f32
            };

            let mut line_buf = vec![0.0f32; line_length];
            let mut tmp_line_buf = vec![0.0f32; line_length];
            let mut lab_buf = vec![0.0f64; line_length];
            let mut new_lab_buf = vec![0.0f64; line_length];

            for start in starts {
                let offsets = |i: usize| start + i * stride;

                match (dilate, first_pass_done) {
                    // doOneDimensionDilateFirstPass — itkLabelSetUtils.h:371-453
                    (true, false) => {
                        for i in 0..line_length {
                            lab_buf[i] = labels[offsets(i)];
                            line_buf[i] = if lab_buf[i] != 0.0 { sigma } else { 0.0 };
                        }
                        do_line_dilate_first_pass(
                            &mut line_buf,
                            &mut tmp_line_buf,
                            &lab_buf,
                            &mut new_lab_buf,
                            magnitude,
                        );
                        for i in 0..line_length {
                            dist[offsets(i)] = line_buf[i];
                            out[offsets(i)] = new_lab_buf[i];
                        }
                    }
                    // doOneDimensionDilate — itkLabelSetUtils.h:596-678. The labels
                    // come from the *output* image the previous pass wrote.
                    (true, true) => {
                        for i in 0..line_length {
                            line_buf[i] = dist[offsets(i)];
                            lab_buf[i] = out[offsets(i)];
                        }
                        do_line_label_prop(
                            &mut line_buf,
                            &mut tmp_line_buf,
                            &mut lab_buf,
                            &mut new_lab_buf,
                            magnitude,
                            extreme,
                        );
                        for i in 0..line_length {
                            dist[offsets(i)] = line_buf[i];
                            out[offsets(i)] = lab_buf[i];
                        }
                    }
                    // doOneDimensionErodeFirstPass — itkLabelSetUtils.h:233-369.
                    // `line_buf` is deliberately *not* cleared at background
                    // positions; see upstream finding 5.
                    (false, false) => {
                        for i in 0..line_length {
                            lab_buf[i] = labels[offsets(i)];
                            if lab_buf[i] != 0.0 {
                                line_buf[i] = 1.0;
                            }
                        }
                        for (first, last) in label_runs(&lab_buf) {
                            let leftend = if first == 0 { sigma } else { 0.0 };
                            let rightend = if last == line_length - 1 { sigma } else { 0.0 };
                            do_line_erode_first_pass(
                                &mut line_buf[first..=last],
                                leftend,
                                rightend,
                                magnitude,
                                sigma,
                            );
                        }
                        for i in 0..line_length {
                            dist[offsets(i)] = line_buf[i];
                        }
                        if last_pass {
                            for i in 0..line_length {
                                out[offsets(i)] = if line_buf[i] == sigma {
                                    lab_buf[i]
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                    // doOneDimensionErode — itkLabelSetUtils.h:455-594. The runs
                    // come from the *input* labels on every pass.
                    (false, true) => {
                        for i in 0..line_length {
                            line_buf[i] = dist[offsets(i)];
                            lab_buf[i] = labels[offsets(i)];
                        }
                        for (first, last) in label_runs(&lab_buf) {
                            let sll = last - first + 1;
                            let leftend = if first == 0 { base_sigma } else { 0.0 };
                            let rightend = if last == line_length - 1 {
                                base_sigma
                            } else {
                                0.0
                            };
                            let mut short = vec![0.0f32; sll + 2];
                            let mut tmp_short = vec![0.0f32; sll + 2];
                            short[0] = leftend;
                            short[sll + 1] = rightend;
                            short[1..=sll].copy_from_slice(&line_buf[first..=last]);

                            do_line_erode(&mut short, &mut tmp_short, magnitude, extreme);
                            line_buf[first..=last].copy_from_slice(&short[1..=sll]);
                        }
                        for i in 0..line_length {
                            dist[offsets(i)] = line_buf[i];
                        }
                        if last_pass {
                            for i in 0..line_length {
                                out[offsets(i)] = if line_buf[i] == base_sigma {
                                    lab_buf[i]
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                }
            }
            first_pass_done = true;
        }
    }

    image_from_f64(img.pixel_id(), &size, img, &out)
}

/// `LabelSetDilateImageFilter`: dilates every label of `img` independently,
/// by an axis-aligned open ellipsoid of semi-axes `radius`.
///
/// `radius` holds one entry per axis (SimpleITK's `dim_vec` `Radius`, an
/// `unsigned int` per axis defaulting to `1`; ITK stores them as `double`s but
/// SimpleITK exposes only integers). `use_image_spacing` defaults to `true` in
/// `LabelSetDilateImageFilter.yaml` — note that ITK's own constructor defaults
/// it to `false`.
///
/// Where two labels reach a pixel with equal parabola height, the one at the
/// lower index along the axis being processed wins; see the module docs.
///
/// The output takes `img`'s pixel type and geometry.
///
/// Errors with [`FilterError::RequiresIntegerPixelType`] for a floating-point
/// input (`pixel_types: IntegerPixelIDTypeList`) and
/// [`FilterError::DimensionLength`] when `radius` is not one value per axis.
pub fn label_set_dilate(img: &Image, radius: &[usize], use_image_spacing: bool) -> Result<Image> {
    label_set_morph(img, radius, use_image_spacing, true)
}

/// `LabelSetErodeImageFilter`: erodes every label of `img` independently, by an
/// axis-aligned open ellipsoid of semi-axes `radius`.
///
/// A pixel keeps its label iff no pixel of a *different* label — background
/// included, and including a disconnected run of the same label along some axis
/// — lies strictly inside the ellipsoid centred on it. Touching labels are
/// therefore separated. The image border does not erode.
///
/// Arguments and errors are as for [`label_set_dilate`].
pub fn label_set_erode(img: &Image, radius: &[usize], use_image_spacing: bool) -> Result<Image> {
    label_set_morph(img, radius, use_image_spacing, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    /// SimpleITK's `Radius` default is `std::vector<unsigned int>(3, 1)`,
    /// truncated to the image's dimension, and `UseImageSpacing` is `true`.
    fn defaults(dim: usize) -> (Vec<usize>, bool) {
        (vec![1; dim], true)
    }

    fn img(size: &[usize], v: Vec<i32>) -> Image {
        Image::from_vec(size, v).unwrap()
    }

    fn labels_of(out: &Image) -> Vec<i32> {
        out.scalar_slice::<i32>().unwrap().to_vec()
    }

    #[test]
    fn defaults_match_the_yamls() {
        let (radius, use_image_spacing) = defaults(3);
        assert_eq!(radius, vec![1, 1, 1]);
        assert!(use_image_spacing);
    }

    // ---- dilation ----------------------------------------------------------

    /// Two labels three voxels apart, radius 3: every pixel is reached, and the
    /// middle pixel is equidistant (2 voxels from each). The tie goes to the
    /// **lower index**, not the smaller label value — swapping the labels
    /// swaps the answer.
    #[test]
    fn dilate_tie_goes_to_the_lower_index() {
        let a = img(&[5, 1], vec![1, 0, 0, 0, 2]);
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[3, 3], true).unwrap()),
            [1, 1, 1, 2, 2]
        );

        let b = img(&[5, 1], vec![2, 0, 0, 0, 1]);
        assert_eq!(
            labels_of(&label_set_dilate(&b, &[3, 3], true).unwrap()),
            [2, 2, 2, 1, 1]
        );
    }

    /// Radius 2, spacing 1: the parabola reaches height `0` exactly two voxels
    /// out, which loses the tie against background's `0`. So each label grows
    /// by one voxel and the middle pixel stays background — the open-ball rule.
    #[test]
    fn dilate_is_an_open_ball() {
        let a = img(&[5, 1], vec![1, 0, 0, 0, 2]);
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[2, 2], true).unwrap()),
            [1, 1, 0, 2, 2]
        );
    }

    /// Upstream finding 2: `Radius = 1` with `UseImageSpacing` on unit spacing
    /// is the identity, because the neighbour's parabola height is exactly `0`.
    /// Turning spacing off adds the `+1` margin and the dilation happens.
    #[test]
    fn spacing_mode_radius_one_is_an_identity() {
        let a = img(&[5, 1], vec![0, 0, 3, 0, 0]);
        let (radius, use_image_spacing) = defaults(2);
        assert_eq!(
            labels_of(&label_set_dilate(&a, &radius, use_image_spacing).unwrap()),
            [0, 0, 3, 0, 0]
        );
        assert_eq!(
            labels_of(&label_set_dilate(&a, &radius, false).unwrap()),
            [0, 3, 3, 3, 0]
        );

        // ... and erosion is equally inert, for the mirrored reason.
        let b = img(&[5, 1], vec![0, 4, 4, 4, 0]);
        assert_eq!(
            labels_of(&label_set_erode(&b, &radius, use_image_spacing).unwrap()),
            [0, 4, 4, 4, 0]
        );
        assert_eq!(
            labels_of(&label_set_erode(&b, &radius, false).unwrap()),
            [0, 0, 4, 0, 0]
        );
    }

    /// Halving the spacing along an axis halves a voxel's physical size, so a
    /// radius of 2 physical units reaches three voxels out along it
    /// (`0.5 k < 2` for `k <= 3`) rather than the one voxel it reaches at unit
    /// spacing. Ignoring spacing instead applies the `+1` margin rule
    /// (`k^2 < r^2 + 2 = 6`, i.e. `k <= 2`) with the voxel grid as the metric.
    #[test]
    fn dilate_honours_image_spacing() {
        let mut a = img(&[7, 1], vec![0, 0, 0, 5, 0, 0, 0]);
        a.set_spacing(&[0.5, 1.0]).unwrap();
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[2, 2], true).unwrap()),
            [5, 5, 5, 5, 5, 5, 5]
        );
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[2, 2], false).unwrap()),
            [0, 5, 5, 5, 5, 5, 0]
        );
    }

    /// A 2-D dilation is a genuine ellipse, not a per-axis box. Radius `(2, 2)`
    /// with the `+1` margin admits `k0^2 + k1^2 < 6`: `(2,1)` is in (`5 < 6`)
    /// but the corners `(2,2)` are out (`8`). With spacing on there is no
    /// margin and the rule is `k0^2 + k1^2 < 4`, a 3x3 block.
    #[test]
    fn dilate_2d_is_elliptical_not_separable() {
        #[rustfmt::skip]
        let a = img(&[5, 5], vec![
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ]);
        #[rustfmt::skip]
        let margin = vec![
            0, 1, 1, 1, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            0, 1, 1, 1, 0,
        ];
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[2, 2], false).unwrap()),
            margin
        );
        #[rustfmt::skip]
        let strict = vec![
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ];
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[2, 2], true).unwrap()),
            strict
        );
    }

    /// Dilation writes its output on every pass, so a zero *trailing* radius is
    /// well defined: it simply dilates along axis 0 only. Contrast
    /// [`erode_zero_trailing_radius_with_spacing_yields_background`].
    #[test]
    fn dilate_zero_trailing_radius_with_spacing_dilates_axis_zero_only() {
        let a = img(&[5, 3], vec![9; 15]);
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[2, 0], true).unwrap()),
            vec![9; 15]
        );
    }

    /// Upstream finding 3, dilation half: `Radius = (0, 2)` without spacing
    /// still dilates one voxel along axis 0, because `scale[0] = 1 > 0` makes
    /// axis 0 the "first pass" even though its radius is zero.
    #[test]
    fn dilate_zero_leading_radius_without_spacing_still_grows_axis_zero() {
        let a = img(&[5, 1], vec![0, 0, 7, 0, 0]);
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[0, 2], false).unwrap()),
            [0, 7, 7, 7, 0]
        );
        // With spacing on, `scale[0] == 0` and axis 0 is genuinely skipped.
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[0, 2], true).unwrap()),
            [0, 0, 7, 0, 0]
        );
    }

    // ---- erosion -----------------------------------------------------------

    /// Radius 2, spacing 1: a pixel survives iff every background pixel is at
    /// physical distance `>= 2`. The two pixels adjacent to background die.
    #[test]
    fn erode_against_background() {
        let a = img(&[7, 1], vec![0, 1, 1, 1, 1, 1, 0]);
        assert_eq!(
            labels_of(&label_set_erode(&a, &[2, 2], true).unwrap()),
            [0, 0, 1, 1, 1, 0, 0]
        );
    }

    /// Touching labels separate: each run erodes against the other as if the
    /// other were background.
    #[test]
    fn erode_separates_touching_labels() {
        let a = img(&[6, 1], vec![1, 1, 1, 2, 2, 2]);
        assert_eq!(
            labels_of(&label_set_erode(&a, &[2, 2], true).unwrap()),
            [1, 1, 0, 0, 2, 2]
        );
    }

    /// The image border is treated as "more of the same label": a run touching
    /// it gets `sigma` at that end (`itkLabelSetUtils.h:324-331`), so the two
    /// pixels at the edges survive while the ones flanking the interior
    /// boundary do not.
    #[test]
    fn erode_does_not_eat_the_image_border() {
        let a = img(&[6, 1], vec![1, 1, 1, 1, 0, 0]);
        assert_eq!(
            labels_of(&label_set_erode(&a, &[2, 2], true).unwrap()),
            [1, 1, 1, 0, 0, 0]
        );
    }

    /// Upstream finding 3, erosion half: with `Radius = (0, 2)` and no image
    /// spacing, axis 0 runs the first pass and caps the distance image at
    /// `sigma = scale[0] = 1`, while the last pass compares against
    /// `BaseSigma = 0.5*4 + 1 = 3`. Nothing can equal it: the output is
    /// entirely background.
    #[test]
    fn erode_zero_leading_radius_without_spacing_blanks_the_output() {
        #[rustfmt::skip]
        let a = img(&[3, 7], vec![
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        ]);
        assert_eq!(
            labels_of(&label_set_erode(&a, &[0, 2], false).unwrap()),
            vec![0; 21]
        );
        // With spacing on, axis 0 is skipped outright and axis 1 is both the
        // first and the last pass, so `sigma == BaseSigma` and the border rule
        // keeps every pixel of this all-foreground image.
        assert_eq!(
            labels_of(&label_set_erode(&a, &[0, 2], true).unwrap()),
            vec![1; 21]
        );
    }

    /// Upstream finding 4: `scale[D-1] == 0` skips the only pass that writes
    /// the label output, which upstream leaves uninitialized. Defined here as
    /// all-background.
    #[test]
    fn erode_zero_trailing_radius_with_spacing_yields_background() {
        let a = img(&[5, 3], vec![9; 15]);
        assert_eq!(
            labels_of(&label_set_erode(&a, &[2, 0], true).unwrap()),
            vec![0; 15]
        );
    }

    /// Every radius zero with spacing on: no axis runs at all. Upstream returns
    /// uninitialized memory for both filters; this port returns background.
    #[test]
    fn all_zero_radii_with_spacing_run_no_pass() {
        let a = img(&[4, 4], vec![3; 16]);
        assert_eq!(
            labels_of(&label_set_erode(&a, &[0, 0], true).unwrap()),
            vec![0; 16]
        );
        assert_eq!(
            labels_of(&label_set_dilate(&a, &[0, 0], true).unwrap()),
            vec![0; 16]
        );
    }

    /// Upstream finding 6: run detection narrows labels to `float`, so
    /// `16777216` and `16777217` (adjacent `Int32` values that share a `f32`
    /// representation) form a single run and erode as one object instead of
    /// separating.
    #[test]
    fn f32_label_collision_merges_two_runs() {
        const A: i32 = 16_777_216;
        const B: i32 = 16_777_217;
        assert_eq!(A as f32, B as f32);

        // One six-long run spanning the whole line: both ends touch the image
        // border, so nothing erodes and the interface is never found.
        let merged = img(&[6, 1], vec![A, A, A, B, B, B]);
        assert_eq!(
            labels_of(&label_set_erode(&merged, &[2, 2], true).unwrap()),
            [A, A, A, B, B, B]
        );

        // Two labels that do *not* collide separate as usual.
        let distinct = img(&[6, 1], vec![1, 1, 1, 2, 2, 2]);
        assert_eq!(
            labels_of(&label_set_erode(&distinct, &[2, 2], true).unwrap()),
            [1, 1, 0, 0, 2, 2]
        );
    }

    /// A 2-D erosion is elliptical for the same reason dilation is: with
    /// `radius = (2, 2)` and unit spacing, every edge pixel of a 3x3 block has
    /// a background face neighbour one voxel away (`1 < 2`) and dies, leaving
    /// only the centre, whose nearest background pixel is two voxels away.
    #[test]
    fn erode_2d_block() {
        #[rustfmt::skip]
        let a = img(&[5, 5], vec![
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ]);
        #[rustfmt::skip]
        let expected = vec![
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ];
        assert_eq!(
            labels_of(&label_set_erode(&a, &[2, 2], true).unwrap()),
            expected
        );
    }

    // ---- errors and geometry -----------------------------------------------

    #[test]
    fn floating_point_pixel_types_are_rejected() {
        let a = Image::from_vec(&[2, 2], vec![0.0f32; 4]).unwrap();
        assert_eq!(
            label_set_dilate(&a, &[1, 1], true),
            Err(FilterError::RequiresIntegerPixelType(PixelId::Float32))
        );
        assert_eq!(
            label_set_erode(&a, &[1, 1], true),
            Err(FilterError::RequiresIntegerPixelType(PixelId::Float32))
        );
    }

    #[test]
    fn radius_needs_one_value_per_axis() {
        let a = img(&[2, 2], vec![0; 4]);
        assert_eq!(
            label_set_dilate(&a, &[1, 1, 1], true),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 3
            })
        );
    }

    #[test]
    fn preserves_geometry_and_pixel_type() {
        let mut a = Image::from_vec(&[3, 3], vec![1u8; 9]).unwrap();
        a.set_spacing(&[0.5, 2.0]).unwrap();
        a.set_origin(&[3.0, -1.0]).unwrap();
        let out = label_set_dilate(&a, &[2, 2], true).unwrap();
        assert_eq!(out.pixel_id(), PixelId::UInt8);
        assert_eq!(out.spacing(), a.spacing());
        assert_eq!(out.origin(), a.origin());
    }
}
