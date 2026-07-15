# Port-side boundary-condition map

*What value does the port substitute when a filter reads a neighbor that lies
outside the image?* One row per port filter that touches a boundary, read from
the code (not inferred from the filter's name). This is the port half of a
two-sided parity audit; the ITK half is built separately and merged against
this.

Candidate rules, named consistently below:

- **clamp / replicate** — out-of-bounds index clamped to the nearest in-bounds
  voxel per axis (`i.clamp(0, size-1)`). This is what ITK calls
  `ZeroFluxNeumannBoundaryCondition`, and it is edge-pixel *replication*.
- **constant(c)** — a fixed value `c` for any out-of-bounds index.
- **wrap / periodic** — `i.rem_euclid(size)`.
- **mirror / reflect** — reflect with edge-repeat, period `2*size`.
- **skip** — out-of-bounds neighbor is *not read and not substituted*; it is
  dropped from the computation entirely (a genuinely different rule from
  clamp — see §7).
- **default-pixel** — a whole sample maps outside the input; a caller-set
  background value is written (resampling flavour, not a neighbor rule).

---

## 0. The shared primitive — `sitk-core/src/boundary.rs`

Every filter routed through `NeighborhoodIterator` picks one of these four by
the 3rd constructor argument. The rule each applies, read from code:

| BC type | rule | exact code | line |
|---|---|---|---|
| `ZeroFluxNeumannBoundaryCondition` | clamp / replicate | `remapped(index, image, \|i, size\| i.clamp(0, size as i64 - 1) as usize)` | boundary.rs:67 |
| `ConstantBoundaryCondition<T>(c)` | constant(c) | out-of-bounds → `self.constant`; in-bounds reads through | boundary.rs:97-108 |
| `PeriodicBoundaryCondition` | wrap | `\|i, size\| i.rem_euclid(size as i64) as usize` | boundary.rs:119 |
| `MirrorBoundaryCondition` | mirror | period `2*size`, `if m < size { m } else { period-1-m }` | boundary.rs:140-144 |

The choice of BC is made *at `NeighborhoodIterator::new` call sites*; it is
**per-call only where a filter threads a parameter into that argument** (§3,
§2 binary). Every other primitive-anchored filter passes a literal unit
struct, i.e. hard-codes the rule.

---

## 1. `NeighborhoodIterator` + hard-coded `ZeroFluxNeumann` (clamp / replicate)

All of these construct `NeighborhoodIterator::<_, _>::new(img, radius,
ZeroFluxNeumannBoundaryCondition)` with a literal unit struct — **hard-coded**,
no per-call boundary parameter. Rule for every row: **clamp / replicate**.

| filter (file) | boundary-decision site(s) |
|---|---|
| gradient — all 5 derivative/gradient filters | gradient.rs:165, 290, 339, 437, 742 |
| canny (2nd-derivative, gate, zero-crossing) | canny.rs:88, 155, 234 |
| geodesic morphology (erode/dilate marker) | geodesic_morphology.rs:142, 176 |
| iso-contour / grayscale contour | contour.rs:613 |
| binary morphology — vote-based dilate/erode/open/close, 2-D connectivity | binary_morphology.rs:259, 319, 471 |
| normalized correlation | normalized_correlation.rs:298 |
| denoise — curvature/gradient-anisotropic, bilateral, discrete-gaussian, mean/median | denoise.rs:130, 183, 626, 1166 |
| (grad/curvature) anisotropic diffusion | anisotropic_diffusion.rs:117 |
| min/max curvature flow (solver + kernel) | min_max_curvature_flow.rs:412, 566 |
| noise estimate | noise_estimate.rs:75, 174 |
| connected/confidence region growing | region_growing.rs:433, 519 |
| level-set update function (Δ, ∇, curvature) | level_set/function.rs:787, 795 |
| sharpening (Laplacian) | via convolution (§3), `ZeroFluxNeumannPad` default |
| stochastic fractal dimension | stochastic_fractal_dimension.rs (NeighborhoodIterator, ZeroFluxNeumann) |
| fast-marching topology check (`label_at`) — **hand-rolled clamp**, not the primitive | fast_marching_base.rs:751 `(coords[d]+offset[d]).clamp(0, size[d]-1)` |

Non-vacuity guard for this group (any bit-pin): the summing stencils
(gradient, discrete-gaussian, mean, Laplacian, correlation) need a
**fold-order** guard; the comparison stencils here are none — those are §2/§5.
A clamp-vs-skip vs clamp-vs-zero pin must exercise an **edge voxel whose
clamped neighbor differs from a zeroed/skipped one**, else the rule is
invisible.

---

## 2. `NeighborhoodIterator` + `Constant` — grayscale/binary morphology (`morphology.rs`)

| filter | site | rule | per-call? |
|---|---|---|---|
| grayscale erode | morphology.rs:309 | `constant(T::MAX_VALUE)` | hard-coded |
| grayscale dilate | morphology.rs:333 | `constant(T::NONPOSITIVE_MIN)` | hard-coded |
| binary erode | morphology.rs:537-540 | `constant(boundary_value)` | **per-call** — `boundary_to_foreground: bool` selects foreground vs background (morphology.rs:532-536) |
| binary dilate | morphology.rs:580-583 | `constant(boundary_value)` | **per-call** — same bool |
| grayscale morphological closing | morphology.rs:515 | pads with `NonpositiveMin` sentinel before compose, crops after | hard-coded |

The constant is chosen so the sentinel never wins the min/max — the
comparison-stencil analogue of the flooding filters in §5. Non-vacuity guard:
a **slot-dependence** guard (min/max is order-insensitive; a fold-order guard
would be asserting a falsehood).

---

## 3. Convolution / FFT / deconvolution — **per-call** `ConvolutionBoundaryCondition`

Enum `{ ZeroPad, ZeroFluxNeumannPad (#[default]), PeriodicPad }`
(convolution.rs:77-85). The caller passes it; it is a required positional
argument (no defaulting in the fn body — the `#[default]` is on the enum).

| variant | maps to | site |
|---|---|---|
| `ZeroPad` | `constant(0.0)` | convolution.rs:341 (spatial), :447 (FFT pad) |
| `ZeroFluxNeumannPad` | clamp / replicate | convolution.rs:349, :454 |
| `PeriodicPad` | wrap | convolution.rs:357, :457 |

Entry points, all **per-call** with default `ZeroFluxNeumannPad`:

| filter | code path | decision site |
|---|---|---|
| `convolution` | `NeighborhoodIterator` (spatial) | convolution.rs:334-359 |
| `fft_convolution` | padded scratch buffer (`pad_input`→`sample_region`→`boundary.get_pixel`) | convolution.rs:441-459 |
| `inverse_/wiener_/tikhonov_/landweber_/projected_landweber_/richardson_lucy_deconvolution` | shared `convolution::pad_input` | deconvolution.rs:144-150 → convolution.rs:441-459 |
| `fft_pad` | delegates to §4 pad filters | fourier.rs:400-406 |

Note: `OutputRegionMode::Valid` shrinks the output to the boundary-free
interior, so the BC is constructed but never exercised — the rule is unchanged,
only whether it fires.

---

## 4. Pad filters (`geometry.rs`) — hard-coded per filter (rule *is* the filter)

All route through `pad_fill`→`boundary.get_pixel`.

| filter | rule | site | per-call? |
|---|---|---|---|
| `constant_pad` | `constant(c)` | geometry.rs:320 | value `c` per-call; rule hard-coded |
| `mirror_pad` | mirror | geometry.rs:330 | hard-coded |
| `wrap_pad` | wrap | geometry.rs:341 | hard-coded |
| `zero_flux_neumann_pad` | clamp / replicate | geometry.rs:405 | hard-coded |

`forward_fft`/`inverse_fft`/`crop`/`extract`/`flip`/`permute_axes` read only
in-bounds — no BC (confirmed: `fourier.rs:19` "none of them pads").

---

## 5. Flooding / connected-component / extrema — hand-rolled **skip**

These do NOT use `NeighborhoodIterator`. They hand-roll a neighbor walk that
**skips** any out-of-image neighbor (`if v < 0 || v as usize >= size[d] {
continue }`), documented as *equivalent to* an ITK `Constant(sentinel)` pad
because the sentinel can never win the filter's comparison. Rule: **skip**
(no value substituted). All **hard-coded**.

| filter | walk / site | ITK-equivalent sentinel |
|---|---|---|
| reconstruction by erosion/dilation | shared `Connectivity::collect`, reconstruction.rs:159-170 | `NumericTraits<T>::max()` |
| morphological watershed-from-markers, regional minima | `NeighborWalker`, watershed.rs (via reconstruction.rs) | `NumericTraits<Label>::max()` |
| `scalar_connected_component` | `NeighborWalker` (Half::Previous), scalar_connected_component.rs:125-133 | `Constant(0)` |
| `regional_maxima` / `regional_minima` | `NeighborWalker`, regional_extrema.rs:87 | `MarkerValue` (`NonpositiveMin`/`max`) |
| `connected_component` (label.rs) | shared `NeighborWalker` family | `Constant(0)` |
| `morphology_reconstruction` | via `reconstruct` | as reconstruction |
| toboggan | hand-rolled `neighbor()`→`Option`, toboggan.rs:99-102 | `IsInside` skip |
| classic watershed | **padded retaining-wall** (distinct — §12) | `NumericTraits<T>::max()` border |
| `object_morphology` | hand-rolled skip, object_morphology.rs:159 | `UseBoundaryCondition=false` → skip |
| `reinitialize_level_set` | hand-rolled skip, reinitialize_level_set.rs:253-257 | face-connected, OOB skipped |
| label-shape boundary/perimeter count | hand-rolled short-circuit, label_shape.rs:684-687 | `Constant(label+1)` → edge always counts as boundary |

Non-vacuity guard: **slot-dependence** (order-insensitive comparisons). A pin
must include a component/basin that **touches the image edge**, else the skip
is never taken.

---

## 6. Hand-rolled inline `ZeroFluxNeumann` clamp (edge replication)

Arithmetically identical to the shared primitive (`i.clamp(0, size-1)`) but
inlined rather than routed through `NeighborhoodIterator`. All **hard-coded**.

| filter | site |
|---|---|
| demons field Gaussian smoothing | demons/field.rs:124-125 |
| objectness / Hessian (`hessian_at`) | objectness.rs:228-232 |
| level-set grid (`clamped_index`) | level_set/grid.rs:52-57 (sibling `in_bounds_index`:64 instead returns `None` = skip) |
| displacement-field Jacobian determinant | jacobian_determinant.rs:241-247 (`(i+1).min(last)` / `i.saturating_sub(1)`) |
| chan-vese level-set update | chan_vese.rs:444 |
| fast-marching `label_at` | fast_marching_base.rs:751 (also listed §1) |

These are the sites a primitive-anchored search would miss — they touch no
shared BC symbol. Included by a separate `.clamp(0,` sweep.

---

## 7. TRUE zero-flux Neumann (drop the tap) — **distinct from the clamp**

A naming collision worth flagging to the ITK side: these implement genuine
zero-flux (the out-of-image tap is *excluded* / contributes zero flux across
the border), which is **not** the edge-replication that ITK's
`ZeroFluxNeumannBoundaryCondition` class actually performs. Different bits on
every boundary voxel.

| filter | rule | site |
|---|---|---|
| coherence-enhancing diffusion | OOB neighbor → index `-1`, tap dropped from row-sum and update | coherence_enhancing_diffusion.rs:646-654, 665, 691 |
| adaptive histogram equalization | OOB window offsets excluded from the local histogram; denominator shrinks (`MovingHistogram` Add/RemoveBoundary) | adaptive_histogram_equalization.rs:56-66 |
| patch-based denoising | `ImageBoundaryFacesCalculator` interior-first; boundary faces clamped (patch_based_denoising.rs:621) but doc states "no boundary value ever reaches the arithmetic" (:70-71) | patch_based_denoising.rs:64-79 |

All **hard-coded**.

---

## 8. Distance transforms — mostly **no neighbor substitution**

| filter | rule | site |
|---|---|---|
| `signed_maurer_distance_map` | pure EDT (parabola lower-envelope); only the boundary *seed* reads a neighbor, via **skip** (`neighbor_index`→`None`) | distance.rs:194-198, 378-381 |
| `danielsson_distance_map` | structural — sweeps never generate an OOB read (forward `1..size` off −1, backward `size-2..=0` off +1) | distance.rs:466-502 |
| `signed_danielsson_distance_map` | propagation as above (none); the extra dilation drops OOB = `constant-zero` background | distance.rs:528-540 |
| `iso_contour_distance` | **both**: clamp for the central-difference gradient (`clamped_index`, :209-216), **skip** for the level-set crossing neighbor (:653-655) | distance.rs:209, 653 |
| `approximate_signed_distance_map` | iso-contour clamp (above) + chamfer sweep that **drops** OOB neighbor writes | distance.rs:846-864 |

All **hard-coded**. Two distinct policies coexist in `distance.rs`
(`neighbor_index` skip vs `clamped_index` clamp); `iso_contour_distance` uses
both — the most likely place to conflate them.

---

## 9. Recursive / IIR seeding — the seam differs by filter

The "boundary" of an IIR filter is how the causal/anti-causal recursion is
seeded at the first/last sample. **These are not all the same rule:**

| filter | seed rule | site |
|---|---|---|
| `recursive_gaussian` / `_with_order` / `smoothing_recursive_gaussian` | **clamp / replicate** — border value `data[0]`/`data[ln-1]` scaled by `bn*`/`bm*` boundary coefficients ("border extends to infinity") | recursive_gaussian.rs:476, 484, 496, 505 |
| Deriche coefficients (`Coefficients::new`) | computes the `bn1..4`/`bm1..4` consumed above | deriche.rs:271-278 |
| `smooth_gaussian` (FIR, separable) | **clamp / replicate** | smoothing.rs:93-94 |
| `bspline_decomposition` | **mirror** (whole-point symmetric) — closed-form mirror seed, *not* clamp | bspline_decomposition.rs:114-146, 151-154 |

All **hard-coded**. Flag: recursive Gaussian/Deriche seed = clamp; B-spline
decomposition seed = mirror. Do not assume the IIR filters share a seam.

---

## 10. Resampling / interpolation (`sitk-transform`) — two-level

Two independent decisions, both present:

**(a) Whole sample outside input** — `default-pixel`, mostly **per-call**:

| entry point | rule | per-call? | site |
|---|---|---|---|
| `ResampleImageFilter` | `sample(..).unwrap_or(default_value)` | **per-call** `set_default_pixel_value`, default 0.0 | resample.rs:273 |
| `WarpImageFilter` | `unwrap_or(edge_padding_value)` | **per-call** `set_edge_padding_value`, default 0.0 | warp.rs:262 |
| Warp displacement-field read | clamp / replicate (no inside test at all) | hard-coded | warp.rs:351-378 |
| `BSplineTransform::transform_point` | identity `T(x)=x` outside valid region `[1,gridSize-2)` | hard-coded | bspline.rs:564-565 |
| `expand` (ExpandImageFilter) | clamp only; no whole-sample-outside path (interior by construction) | hard-coded | expand.rs:189-192 |
| BSpline re-grid / N4 fit | `0.0` outside old grid / domain error+ε-clamp | hard-coded | bspline.rs:470-471, n4_bias_field/bspline.rs:391-407 |

Shared inside test: `c >= -0.5 && c < size-0.5` (interpolator.rs:29-34).

**(b) Per-tap edge of the interpolator kernel** (once the sample is inside) —
all **hard-coded**, and the rule differs by kernel:

| kernel | per-tap rule | site |
|---|---|---|
| nearest, linear, linear+grad | clamp / replicate | interpolator.rs:50, 83, 129 |
| windowed-sinc (all 5 windows) | clamp / replicate | interpolator.rs:697 |
| B-spline | **mirror** fold (`-i`, `2(n-1)-i`) | interpolator.rs:303-312 |
| Gaussian | **region truncation + renormalize** (third distinct rule — neither clamp nor mirror) | interpolator.rs:419-420, 480 |
| expand linear / nearest | clamp / replicate | expand.rs:70, 88 |
| N4 kernel taps | **skip** out-of-lattice taps; `evaluate`→0 outside support | n4_bias_field/bspline.rs:720-724 |

Gaussian's truncate+renormalize is a genuinely third edge behavior — flag to
the ITK side alongside §7.

---

## 11. Demons deformable-registration warp — `is_inside_buffer` → edge-padding

The moving-image sample / gradient reads gate on `is_inside_buffer`
(`c >= -0.5 && c < size-0.5`); outside, an **edge-padding** value is used or the
tap dropped. All **hard-coded**.

| filter | rule | site |
|---|---|---|
| symmetric-forces demons | `is_inside_buffer` gate, else drop/skip | demons/symmetric.rs:194-262 |
| level-set-motion demons | `is_inside_buffer` gate on moving/smoothed | demons/level_set_motion.rs:252-283 |
| ESM demons | out-of-buffer → `edge_padding = scalar_max(moving)` | demons/esm.rs:15, 139, 155 |
| central-difference image function | OOB difference tap → EvaluateAtIndex boundary rule (not one-sided) | demons/image_function.rs:219-238 |

---

## 12. Classic watershed — **padded retaining-wall** (a padded scratch buffer)

Distinct from every path above: `watershed_classic` allocates a one-pixel
border around the volume and fills it with `NumericTraits<T>::max()` (the
"retaining wall"), then reads face-connected neighbors of the *padded* buffer
with no per-read bounds check. Rule = **constant(type max)** materialized as a
pad, not consulted per-read. Hard-coded.

- pad build: watershed_classic.rs:878 (`Padded::new`), :906 (`BuildRetainingWall`)
- wall value: watershed_classic.rs:936-948 (`NumericTraits<T>::max()` per pixel type)

---

## Per-call vs hard-coded — summary

**Per-call boundary behavior** (caller can change the substitution):

- convolution / fft_convolution / all 6 deconvolutions / fft_pad —
  `ConvolutionBoundaryCondition` (§3), default `ZeroFluxNeumannPad`.
- binary erode / binary dilate — `boundary_to_foreground: bool` picks the
  `Constant` sentinel (§2).
- `constant_pad` — the constant *value* (rule fixed to constant) (§4).
- `ResampleImageFilter` / `WarpImageFilter` — default/edge-padding pixel value
  (§10a).
- `invert_displacement_field` — `enforce_boundary_condition: bool`, yaml
  default `true` (zeroes the outer lattice plane; invert.rs:234-237).

**Everything else is hard-coded** to match its ITK filter's fixed default.

---

## Bound — what was enumerated, and where the map could still be blind

This claim ("the port has no other boundary substitution") is only worth its
enumeration:

**Anchors swept, workspace-wide (`rg`):**
`NeighborhoodIterator::new`, `BoundaryCondition` and all four impl names,
`ConvolutionBoundaryCondition`, `ConstantBoundaryCondition::new`,
`.clamp(0,`, `rem_euclid`, `.min(..size)`, `saturating_sub`, `is_inside` /
`IsInside`, and hand-rolled `if v < 0 || v >= size[d]` neighbor guards. Every
hit across `sitk-filters`, `sitk-core`, `sitk-transform` was read and
classified above.

**Where a primitive-anchored search is structurally blind** (and thus was
covered only by the `.clamp(0,` / skip-guard sweep, §5-§12): a filter that
inlines its own edge handling touches no shared symbol. The confirmed inliners
are demons/field, objectness, level_set/grid, jacobian_determinant, chan_vese,
fast_marching, the entire flooding/CC/extrema family, distance.rs, the IIR
filters, watershed_classic, and the interpolators. The main panel's predicted
hiding spots were correct: `signed_maurer_distance_map` (§8, skip/EDT), the
recursive Gaussians (§9, clamp seed), FFT pad (§3/§4), and manual `if i==0`
edge cases (§5, §6).

**Residual not exhaustively read** (matched a value-clamp anchor but appear to
be intensity/index clamps, not neighbor OOB — classified by inspection, not
full read): `intensity.rs`, `scalar_to_rgb_colormap.rs`, `kmeans.rs`,
`label_map_mask.rs`, `label_map_overlay.rs`, `grid_utility.rs`, `slice.rs`,
`shrink.rs`, `slic.rs`, `clamp.rs`, `math.rs`, `logic.rs`, `threshold.rs`.
These clamp pixel *values* or generate output *indices*; none was found to
substitute a *neighbor*. If the ITK side flags any of these as a
neighbor-reading filter, they need a targeted re-read — that is the honest edge
of this enumeration.

**Two rules the ITK side must not conflate with "ZeroFluxNeumann":**
1. §7 CED / AHE and §10b Gaussian implement *drop-the-tap* / *truncate*, not
   edge replication.
2. §9 B-spline decomposition and §10b B-spline interpolation seed with
   *mirror*, not clamp.
