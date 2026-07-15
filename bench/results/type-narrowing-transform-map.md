# Type-narrowing census — `sitk-transform` + `sitk-registration` + the DEVICE path

Report-only sweep. Domain: the transform crate, the registration crate, and the
CUDA device path (`sitk-cuda` + the `sitk-registration` device driver). Sister
panel owns `sitk-core`/`sitk-filters`.

**Method.** One row per site where the port's intermediate compute type differs
from ITK's — or where the host and device compute the same value at different
precision. Both sides read firsthand: port `file:line` and ITK
`.hxx`/`.h:line` (or host `file:line` vs device `file:line`). ITK reference at
`/home/stevek/work/ITK`, read against the **v4** metric/optimizer family the port
mirrors (not the v3 `ImageToImageMetric` family).

**The discriminator** (the point of the sweep). For each precision difference,
does it feed a **discrete decision** — a threshold/convergence stop, a branch
(`< 0`, `== 0`), an index/round-to-int, a saturating cast?
- Feeds a discrete decision → **CANDIDATE**: name the separating host-vs-ITK (or
  host-vs-device) input that takes a different discrete path.
- Only ULP-shifts a continuous value → **§4 keep-more-precise / faithful**.

---

## Headline verdict

**This is a nearly-all-faithful sweep, like the boundary sweep — not the
coordinate sweep.** Zero sites where the port is *narrower* than ITK (no
direction-2 bug). The ITK **v4** family the port mirrors computes in `double` end
to end (`TInternalComputationValueType = double`, `PDFValueType = double`,
`TCoordRep = double`), so at almost every accumulator, term, index, and threshold
the port's `f64` neither narrows below ITK nor exceeds it.

**Two things break the "all-faithful" symmetry and are the findings of this
sweep:**

1. **One port-*wider* site (direction 1): the mean-squares difference.** ITK
   MeanSquaresv4 computes `const FixedImagePixelType diff = fixedImageValue −
   movingImageValue` **in the image pixel type**
   (`itkMeanSquaresImageToImageMetricv4GetValueAndDerivativeThreader.hxx:44`,
   read firsthand) — a **`float`** subtraction for the common
   `sitkFloat32`-registration case, with catastrophic cancellation *before* the
   promotion to `double` at `:50`. The port subtracts in `f64`
   (`metric.rs:1104,1143,1209,1405,1437`). Port is more precise; the gap is
   data-dependent (float cancellation, can exceed a ULP-of-result), and it feeds
   the continuous metric value → the convergence stop. **Not bit-pinnable to ITK
   → §4 keep-more-precise, but flagged for the verify phase** (row A9). The device
   mean-squares kernel also subtracts in `double` (`resident.rs:134`), so it
   matches the *port*, not ITK — no host↔device gap, but both diverge from ITK
   for a float image.

2. **One host-`f64` vs device-`f32` narrowing (direction 3): the `f32` device
   image.** `DeviceImage::upload` casts the input to `f32`; for a `Float64` input
   the host `execute` keeps `f64` while the device holds `f32`, and this feeds the
   Mattes Parzen bin (discrete). It is **by-design and documented** (§5.12 /
   §2.162): the device reproduces `execute` on the **Float32 cast**, exact for
   integer/`Float32` inputs, divergent from the `f64` host only for `Float64`
   inputs. Detailed in §D (rows D1a/D1b) with its separating input.

The **third** substantive both-ends divergence is **host↔device reduction *order***
(both sides `f64`, not a narrowing): the device metric value/derivative is summed
in a fixed parallel-tree order that no left-to-right host `f64` accumulation
reproduces. It feeds the convergence stop and the RegularStep relaxation branch,
so it *can* change the device optimizer's endpoint. It is measured and documented
as **`doc/upstream-findings.md` §2.157** and is an accepted band, not a bug — and
it is not a *type* narrowing. Detailed in §D below with its separating input.

**Device-metric-value → host-convergence round-trip verdict: `f64`-faithful.**
The device returns `MetricValue { value: f64, derivative: Vec<f64> }`
(`device.rs:37`, `metric.rs` `MetricValue`) into the optimizer's `Objective`,
which is `f64` (`optimizer.rs`). The convergence window is `f64`
(`convergence.rs:35-39`). There is **no `f32` anywhere on the value→stop path.**
The Mattes value is *bit-identical* to the host (deterministic counting-sort
histogram); mean-squares/correlation values differ only by reduction rounding.

---

## §A — Host metric precision (`mattes`, `joint_histogram`, `correlation`, `ants_correlation`, `demons`, `metric`)

ITK v4 metrics are `double` throughout — confirmed at the source:
`itkMattesMutualInformationImageToImageMetricv4.h:88` (`TInternalComputationValueType = double`),
`:156` (`using PDFValueType = TInternalComputationValueType`),
`:159` (`JointPDFType = Image<PDFValueType,2>`);
`itkANTSNeighborhoodCorrelationImageToImageMetricv4.h:86` (`= double`).

| # | port file:line | ITK .hxx:line | port type | ITK type | discrete? | class | note / separating input |
|---|---|---|---|---|---|---|---|
| A1 | `mattes.rs:193-197` (`value/bin_size - normalized_min` → `term as isize`) | `...Metricv4GetValueAndDerivativeThreader.hxx:245-250` (`... movingImageParzenWindowTerm` → `static_cast<OffsetValueType>`) | `f64`→`isize` trunc | `double`→`OffsetValueType` trunc | **YES** — Parzen bin index | **faithful** | Both truncate a `double` term. Port comment (`:195`) matches ITK: term ≥ padding ≥ 0 so trunc==floor. Clamp `[2, nbins-3]` (port `:198-204`) mirrors ITK `:253-262`. No narrowing. |
| A2 | `mattes.rs:140-145,183-184` (`fixed_bin_size`, `moving_bin_size`, `*_normalized_min`) | `...Metricv4.hxx` `m_*BinSize`/`m_*NormalizedMin` | `f64` | `double` | feeds A1 | **faithful** | Histogram geometry all `f64`. |
| A3 | `joint_histogram.rs:533-536` (`round_half_up(normalized/spacing)+PADDING`) | `itkJointHistogramMutualInformationImageToImageMetricv4` (`GetValueAndDerivative` threader) | `f64`→`i64` round-half-up | `double`→index | **YES** — bin index | **faithful** | `round_half_up = (x+0.5).floor() as i64` (`:209-210`) = ITK `RoundHalfIntegerUp`. All `f64`. |
| A4 | `joint_histogram.rs:209-210` `round_half_up` | `itkMath::RoundHalfIntegerUp` | `f64`→`i64` | `double`→int | **YES** — voxel/bin round | **faithful** | Half-up not half-even; matches the coordinate-sweep primitive. |
| A5 | `ants_correlation.rs:435-440` (`sum_f, sum_f2, sum_m, sum_m2, sum_fm, count` all `0.0f64`) | ANTS-NC `QueueRealType`/scan sums (`= double`) | `f64` | `double` | denominator-zero guard (`eps` `:384`) | **faithful** | Local-CC sums `f64` both sides. `sff*smm` zero-guard at `f64` `EPSILON`. |
| A6 | `ants_correlation.rs:272-278` (continuous index → nearest voxel, ITK index rounding) | ANTS-NC threader `TransformPhysicalPointToIndex` | `f64`→int | `double`→`IndexType` | **YES** — center-voxel round | **faithful** | Port comment `:278`: ITK index rounding, not half-away `f64::round`. |
| A7 | `correlation.rs` sums (`sf, sm, sff, smm, sfm`) | `itkCorrelationImageToImageMetricv4HelperThreader.hxx` (`= double`) | `f64` | `double` | `sff*smm != 0` guard | **faithful** | Both `f64`. |
| A8 | `metric.rs:225-236` `SampleValues::Float32(f32)` (+ every native pixel variant) | ITK reads pixel, `static_cast<RealType=double>` | native pixel `f32`/int | native pixel | widened before compute | **faithful (storage, not compute)** | Samples held in native pixel type, widened to `f64` before any metric arithmetic — exactly ITK reading a pixel into `RealType=double`. Not a narrowing. |
| **A9** | `metric.rs:1104,1143,1209,1405,1437` `let diff = mv - fixed.value(s)` (**`f64`** subtraction) | `itkMeanSquaresImageToImageMetricv4GetValueAndDerivativeThreader.hxx:44` `const FixedImagePixelType diff = fixedImageValue - movingImageValue` | **`f64`** | **`FixedImagePixelType`** (= `float` for a `sitkFloat32` image) | convergence stop / relaxation (via continuous value) | **§4 keep-more-precise — PORT WIDER (direction 1); flagged for verify** | ITK subtracts in the image pixel type: for a `float` image the diff is a **float** subtraction (~6.1e-2 granularity near intensity 1000), promoted to `double` only at `:50` for the square. Port keeps the low bits in `f64`. **Separating input:** fixed `1000.0001_f32`, moving `1000.0002` → ITK diff is float-rounding-dominated, port diff `1.0e-4` exact; summed over the image and fed to the optimizer the two metric values differ above ULP-of-result, so near a convergence-window edge the host can stop at a different iteration than ITK. Not bit-pinnable (continuous value, no floor/round/index below it). Verify-phase decision: match ITK's float subtraction for bit-parity, or keep `f64` for host/device determinism (the `metric.rs` module docs already argue for `f64`). Note: the **device** MS kernel subtracts in `double` (`resident.rs:134`) → matches the port, not ITK. |

**§A CANDIDATE-BUG rows: 0 (0 port-narrower). Port-wider (direction-1, §4-flagged): 1 (A9). §4-faithful rows: 8.**

---

## §B — Optimizer value/gradient/step precision + convergence (`optimizer`, `convergence`, `lbfgs2`, `lbfgsb`, `gradient_free`, `scales`)

ITK v4 optimizers are `TInternalComputationValueType = double`. The two discrete
stops the brief flagged as highest-value both use **compensated summation on both
sides**.

| # | port file:line | ITK .hxx:line | port type | ITK type | discrete? | class | note / separating input |
|---|---|---|---|---|---|---|---|
| B1 | `optimizer.rs:399,402` (`gradient_magnitude = compensated_sum(g*g).sqrt()`; `< gradient_magnitude_tolerance`) | `itkRegularStepGradientDescentOptimizerv4.hxx:108-113,115` (`compensatedSummation` … `std::sqrt`; `< m_GradientMagnitudeTolerance`) | `f64` compensated | `double` `CompensatedSummation` | **YES** — stationary-point stop | **faithful** | Both compensated `f64`. The precise stop I expected to diverge does **not**: ITK uses the same compensated sum, so no separating input exists at the precision level. |
| B2 | `optimizer.rs:413,419` (`scalar_product = compensated_sum(...)`; `< 0.0` → relax) | `itkRegularStepGradientDescentOptimizerv4.hxx:127-133,136` (`compensatedSummation` … `scalarProduct < 0` → `*= m_RelaxationFactor`) | `f64` compensated | `double` `CompensatedSummation` | **YES** — direction-reversal relaxation branch | **faithful** | Same compensated `f64` on both sides; sign of the same dot product. |
| B3 | `optimizer.rs:423-424` (`step_length = relaxation*lr`; `< minimum_step_length`) | `...Optimizerv4` step-halving stop | `f64` | `double` | **YES** — step-too-small stop | **faithful** | All `f64`. |
| B4 | `convergence.rs:56-93` (`total_energy += |v|`; line-fit `c0-c1`) | `itkWindowConvergenceMonitoringFunction` (`TScalar` via `TInternalComputationValueType = double`) | `f64` | `double` | feeds `cv <= min_cv` stop (`optimizer.rs:220-221`) | **faithful** | Closed-form re-derivation of ITK's order-1 B-spline fit, all `f64`. Sum *order* may differ from ITK's `BSplineScatteredDataPointSetToImageFilter`, but both `f64` → §4 ULP, not narrowing. |
| B5 | `optimizer.rs:220-221` (`cv <= min_cv`) | ITK `m_ConvergenceValue <= m_MinimumConvergenceValue` | `f64` | `double` | **YES** — convergence stop | **faithful** | Value and window both `f64`; no `f32` at the stop. |
| B6 | `optimizer.rs:244` analog (`stepScale <= epsilon`) / `EstimateLearningRate` | `itkRegularStepGradientDescentOptimizerv4.hxx:244` (`stepScale <= epsilon()`) | `f64` | `double` | **YES** — degenerate-scale branch | **faithful** | Both `f64` epsilon. |
| B7 | `lbfgsb.rs`, `lbfgs2.rs` core | ITK delegates to netlib/liblbfgs `double` core | `f64` | `double` | projected-gradient / line-search stops | **faithful** | Port re-implements the same `double` core; no `float` in either. (Deep line-search parity is an algorithm-order question, not a type-narrowing one — out of this sweep's scope.) |

**§B CANDIDATE-BUG rows: 0. §4-faithful rows: 7.**

---

## §C — Transform / interpolation / resample precision (`transform`, `matrix_offset`, `resample`, `interpolator`, `bspline`, `displacement`, `warp`, `composite`)

ITK interpolators/resamplers are `TCoordRep = double` and
`TInterpolatorPrecisionType = double`; `NumericTraits<pixel>::RealType = double`
for every input pixel type. (The index↔physical rounding primitive was audited in
the prior coordinate sweep — `sitk-core::coord`, `RoundHalfIntegerUp` — and is not
re-derived here.)

| # | port file:line | ITK .h/.hxx:line | port type | ITK type | discrete? | class | note / separating input |
|---|---|---|---|---|---|---|---|
| C1 | `interpolator.rs:29-33` `is_inside(cindex: &[f64], size)` (`[-0.5, size-0.5)`) | `itkInterpolateImageFunction`/`itkImageFunction` `IsInsideBuffer` on `ContinuousIndex<double>` | `f64` | `double` | **YES** — sample-validity branch | **faithful** | Continuous index `f64`; matches ITK bound. Device mirrors bit-for-bit (`resident.rs:90-95`). |
| C2 | `interpolator.rs:49` nearest `(cindex[d]+0.5).floor() as isize` | ITK `NearestNeighborInterpolateImageFunction` (round of `ContinuousIndex<double>`) | `f64`→`isize` | `double`→`IndexType` | **YES** — nearest-voxel pick | **faithful** | Half-up round of `f64` index. |
| C3 | `interpolator.rs:299` B-spline `cindex[d].floor() as isize - 1` (support start) | `itkBSplineInterpolateImageFunction.hxx` (`std::floor` support region) | `f64`→`isize` | `double` | **YES** — support-region index | **faithful** | Knot base from `f64` floor. |
| C4 | `interpolator.rs:64-...,109,290,407,648` linear/cubic/gaussian/sinc values | ITK `EvaluateAtContinuousIndex` in `RealType=double` | `f64` | `double` | continuous | **faithful** | Interpolated value `f64` both sides; the value is continuous in `c` → §4. |
| C5 | `matrix_offset.rs` `transform_point` / `mat_vec` | `itkMatrixOffsetTransformBase.hxx` (`ScalarType`/`TParametersValueType = double`) | `f64` | `double` | feeds C1-C3 | **faithful** | Point map `f64`. Device replays these stored `f64` matrices per stage (`resident.rs:237-249`). |
| C6 | `resample.rs` output cast (interpolated `f64` → output pixel, clamp+round for int output) | `itkResampleImageFilter.hxx` `CastPixelWithBoundsChecking` (`double`→output, clamp+round) | `f64`→output pixel | `double`→output pixel | **YES** — saturating cast (int output) | **faithful** | Both cast a `double` interpolant to the output pixel with the same clamp/round; the cast type is the *output* pixel on both sides, not an intermediate the port narrowed. |

**§C CANDIDATE-BUG rows: 0. §4-faithful rows: 6.**

---

## §D — HOST `f64` vs DEVICE (`sitk-cuda` kernels + `sitk-registration` device driver) — the primary target

**Established up front (not narrowings):**
- The **Mattes kernel computes in `double` end-to-end** (`ops/mattes.rs`: 55×`double`, 0×`float`); `ops/histogram.rs` all `double`.
- **`DeviceImage` stores `f32`** — the storage type. `(double)x` of an `f32` is
  *exact*, so widening on load is lossless **relative to the `f32`-cast reference**;
  the narrowing that matters happened earlier, at `upload`, and is visible only vs an
  `f64`-input host (row D1b).
- Every `float` token in a kernel is either an **image-buffer pointer**
  (`const float* in`) or a **terminal `(float)` narrowing** storing a finished
  `double` result back into an `f32` `DeviceImage` — never an intermediate the
  device computed at reduced width. Enumerated: `pyramid.rs:201,601,641`,
  `rescale_intensity.rs:84`, `gaussian.rs:73` (all `y = (float)(double expr)`),
  `mean_squares.rs:155` / `correlation.rs:224` / `resident.rs:399` (`FSCALAR`
  narrow *fixed* volume — see D1).

### D — the two things that could have narrowed, and why they don't

| # | device file:line | host file:line | device op | host op | discrete? | class | separating input |
|---|---|---|---|---|---|---|---|
| D1a | `resident.rs:367-401` (`Volumes::Split`: `fvals: f32`, `mbuf: f64`) vs a **device pipeline** whose images came from `DeviceImage::upload` | host `execute` on the **Float32 cast** of the same inputs (`method.rs:2643-2652`) | fixed/moving stored `f32`, widened `(double)fvals` (`resident.rs:331`) | same `f32`-cast values, `f64` load | feeds `diff` (MS/corr) and Parzen bin (Mattes) | **faithful** — reference = the f32-cast host | `(double)f32` is exact, so widening equals the host's `f64` load of the same `f32` pixel bit-for-bit. `∀ f32 x, (double)x` is exact. For **integer / `Float32`** inputs the f32-cast host *is* the host (`prepare_level` promotes integers to `Float32` once, §5.12), so there is no separating input at all. |
| **D1b** | `image.rs` `DeviceImage::upload` casts input to **`f32`**; `resident.rs:367-401` then stores `f32`/widens; `mattes.rs:227` truncates `mv` into the Parzen bin | host `execute` on the **original `Float64`** image keeps `f64` at every level (`prepare_level` `method.rs:2447-2452`: `recursive_gaussian` narrows to the *input* pixel type, so a `Float64` input stays `Float64`) | interpolated `mv` from **`f32`** storage | interpolated `mv` from **`f64`** storage | **YES** — Mattes bin `(long long)(mv/binSize − nmin)` (`mattes.rs:227`) | **CANDIDATE (host `f64` vs device `f32`) — by-design & documented §5.12 / §2.162, NOT a defect** | a **`Float64`** moving image with a voxel whose value, after `f32` rounding, crosses a Parzen bin boundary: host-`f64` lands bin *k*, device-`f32` lands bin *k±1* → ½-unit of Parzen mass moves → different `−MI` on the bits. MS/correlation only *sum* the `f32`-rounded value (continuous, §4). Same root cause reaches the **pyramid**: for a `Float64` input the host smooths/shrinks in `f64` while the device narrows each level to `f32` (`pyramid.rs:601,641`) — measured 1.4% coarse-level metric / 5.5e-4 worst-param before the §5.12 integer-promotion fix, which closed the *integer* case; the `Float64` case is the residual **by contract** (device reproduces the Float32-cast). Refuse-or-accept is the caller's one decision at `upload`. |
| D2 | sampler `resident.rs:90-333` (`fmadd_rn` = `__dadd_rn(acc,__dmul_rn(a,b))`, `floor`, `is_inside`, `round`, Parzen trunc) | `interpolator.rs:29-49`, `mattes.rs:193`, host sampler | `double`, mul+add rounded **separately** | `f64`, Rust unfused mul+add | **YES** — floor cell, `is_inside`, mask `round`, Parzen bin | **faithful (pinned bit-identical)** | `fmadd_rn` exists precisely so NVRTC's default FMA does not contract where Rust doesn't. Before the pin, value agreed to 1e-15 and the derivative was off **7%** (`resident.rs:12-14,106-107`). Every discrete decision is now bit-identical host↔device. Resample-through (`pyramid.rs:562-642`) and pyramid Gaussian (`pyramid.rs:217-293`) compiled `function_exact` (no FMA) for the same reason (`pyramid.rs:27,268,839`). |

### D3 — the ONE real host↔device divergence: reduction ORDER (both `f64`, NOT a narrowing)

| aspect | detail |
|---|---|
| **device site** | `resident.rs:335-353` `emit_partials` (fixed shared-mem tree) + `:760-772` `Partials::fold` (host fold in block-index order); `mean_squares.rs:425-438`; `correlation.rs` two-pass moments |
| **host site** | `metric.rs` / `mattes.rs` / `correlation.rs` — sequential left-to-right `f64` accumulation over N samples |
| **types** | **both `f64`** — this is reduction *order*, not a precision *type* difference |
| **discrete?** | **YES, indirectly** — the reduced metric value feeds `convergence_value()` → `cv <= min_cv` (`optimizer.rs:220`), and the reduced gradient feeds `scalarProduct < 0` (RegularStep relaxation, `optimizer.rs:419`) |
| **class** | **ACCEPTED / DOCUMENTED band — `doc/upstream-findings.md` §2.157**, not a candidate bug and not a narrowing |
| **separating input** | Measured at 256³ vs `execute`, same start transform, **unit (ill-conditioned) scales**: params agree 4e-13 (iter 1), 2e-10 (iter 2), 5e-8 (iter 3), ~500×/step, and the two runs stop (both `StepTooSmall`) at **different local minima 7.5e-3 apart** in the worst parameter. With physical-shift scales: **same 33 iterations, same 16 580 608 valid points**, worst param **7.7e-14** (the rounding floor). So the separating input is *an ill-conditioned parameter space*, where a ~√N·ε value/derivative difference flips a discrete RegularStep branch and the feedback loop amplifies it — chaotic for the host too; the host run is simply "the one you saw first" (`method.rs:2657-2696`). |
| **Mattes exception** | The Mattes **value** does **not** live in this band: the joint histogram is a **deterministic counting sort** accumulating per-bin `f64`, bit-identical to the host loop (`histogram.rs:1-46,444`), and handed to the host's own `mattes_tail` (`device.rs:576-628`). Only the Mattes *derivative* is banded (affine-Jacobian probe, 1e-9 relative), and the derivative feeds no discrete decision inside the metric (`device.rs:604-628`). |

### D4 — the `f32`-underflow trap the brief flagged: guarded, not a candidate

| aspect | detail |
|---|---|
| concern | an `f64` fixed-mask voxel holding e.g. `1e-320` would underflow to `f32` 0 if it rode a `DeviceImage`, silently dropping a sample the host keeps (`method.rs:369-370,3064-3065`) |
| resolution | the user fixed mask is thresholded to a **binary 0/1 volume on the host in `f64`** (`binary_volume(m)`) **before** upload — `method.rs:2757-2768` — so nonzero-but-tiny never reaches `f32`. `DeviceMask::upload` likewise does `to_f64_vec()` then `v != 0.0` host-side (`mask.rs:93-106`). The device `threshold_nonzero` (`mask.rs:44-51`) only ever sees a nearest-neighbour-resampled exact-`{0.0,1.0}` predicate, where `x != 0.0f` and `(double)x != 0.0` decide identically (`mask.rs:34-42`). |
| class | **guarded / faithful** — the discrete validity decision is made in `f64` on the host; the `f32` storage never gates. No separating input survives the guard. |

**§D rows: 1 host↔device narrowing feeding a discrete decision (D1b), by-design &
documented (§5.12/§2.162), not a defect; 0 undocumented candidate bugs.** D3 is a
documented accepted band and is reduction *order*, not a type narrowing.
**§4/faithful/guarded/documented rows: 5 (D1a, D1b, D2, D3, D4).** No device kernel
narrows a discrete-decision input to `f32` in *compute*; the only device `f32` that
reaches a discrete decision is image *storage* (D1b), and only vs an `f64`-input
host.

---

## Totals

| slice | port-narrower BUG (dir 2) | port-wider §4 (dir 1) | host-f64/device-f32 (dir 3) | faithful / §4 rows |
|---|---|---|---|---|
| A — host metrics | 0 | **1 (A9, mean-squares diff, verify-flagged)** | — | 8 |
| B — optimizer/convergence | 0 | 0 | — | 7 |
| C — transform/interp/resample | 0 | 0 (B-spline decomp keeps more precision, §4) | — | 6 |
| D — host↔device | 0 | 0 | **1 (D1b, `f32` image → Mattes bin; by-design §5.12)** | 5 (incl. §2.157 reduction-order band) |
| **total** | **0** | **1** | **1** | **26** |

**Zero port-narrower bugs (no direction-2 defect).** Two substantive both-ends
findings, both surfaced above and neither a fixable narrowing:

- **A9 (direction 1, port wider):** the port subtracts the mean-squares difference
  in `f64` where ITK MeanSquaresv4 subtracts in the image pixel type (`float` for a
  `sitkFloat32` image, `hxx:44`). Port is more accurate; the gap is data-dependent
  and feeds the convergence stop. **Flagged for the verify phase** — a policy call
  (bit-parity with ITK vs `f64` determinism), not a bug to fix blind.
- **D1b (direction 3, host `f64` vs device `f32`):** `DeviceImage` storage is `f32`;
  for a `Float64` input the host keeps `f64` while the device narrows, and the
  narrowed value truncates into the Mattes Parzen bin. **By-design and documented
  (§5.12 / §2.162):** the device reproduces `execute` on the Float32 cast — exact
  for integer/`Float32` inputs, divergent from an `f64`-input host only.

Everything else matches: the ITK v4 family is `double` throughout, the port matches
it in `f64`, and the device is pinned bit-identical to the host at every discrete
decision (`fmadd_rn` / `function_exact` / deterministic counting sort). The only
other host↔device value difference is reduction *order* (both `f64`, §2.157) — a
reduction-rounding band that can move an ill-conditioned optimizer's endpoint, not
a precision narrowing.
