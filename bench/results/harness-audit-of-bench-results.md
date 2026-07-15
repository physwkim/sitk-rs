# Audit of `doc/bench-results.md` against the measurement protocol

I do not edit `doc/bench-results.md`; this is the audit, for the main panel to
apply. The protocol and its proof are in `harness-instability-result.md`; the two
facts that drive every verdict below are:

- **The old harness put the box's ~2 s warm-up ramp inside the measurement window**
  (criterion warm-up was 500 ms), and its cell order ran a seconds-long `t1` leg
  immediately before every `tN` leg — which is what cools the box. At 64³ this
  reported ops **up to 2.02× slower than they are**. At 256³ it is not resolvable
  (0.89–0.99). At 512³ **I did not measure it.**
- **The op-set inside a process flips short 64³ cells between two modes 2× apart.**
  Solo, the same cells are stable to 1.05–1.12×.

**Noise floor under the protocol: 1.13× at 64³, 1.08× at 256³, 1.15× at 512³.**
**A ratio below 1.15× is not a result** — that verdict alone strikes several rows
below, independent of the ramp.

## §"small (64³)" — the whole table: RETRACT

Every cell is a `tN` number from a **twelve-op** process under the **500 ms**
warm-up, i.e. both defects at once, at the size where both are largest. Measured
inflation on the same code, paired, 3 rounds: `gmrg` **2.02×**, `signed_maurer`
**1.87×**, `discrete_gaussian` **1.86×**, `gradient_magnitude` **1.47×**, `mean`
1.26×, `binary_dilate` 1.21×, `median` 1.17×. And the ITK column is not a control:
`bench/cpp/main.cxx:466` warms up with **one call**, so ITK's 64³ rows carry the
same defect, unquantified.

Named claims that do not survive:

- **"`mean` still loses at 64³, by 2.82×"** — not reproducible. The rust number
  (14.4 ms) was taken pre-cost-class-split *and* with the ramp inside the window;
  under the protocol on merged main `mean` 64³/`tN` is **1.73–1.75 ms** across two
  campaigns. The ITK half is unverified, so I claim no new ratio — only that this one
  cannot stand.
- **"`otsu_threshold` crossed to a win, 0.68×"** and **"`gradient_magnitude` is now a
  tie, 1.05×"** — both are inside or near the noise floor *and* taken with the ramp
  inside the measurement. `gradient_magnitude` at 64³ cannot be certified at all: its
  wall is at the pool wake-up floor and it fails its own spread test (1.22×).
- The **"was"** column (6.62×, 4.52×, 2.91×) is from the same defective harness.
  The *direction* of the grain-seam win is corroborated by the ABBA controls below;
  the *magnitudes* are not.

**Action: retake the whole 64³ table with `bench/run_protocol.py`, and retake the
ITK column with a C++ harness whose warm-up covers the ramp.** Until then, no 64³
rust/itk ratio in this repository is quotable.

## §"medium (256³)" — mostly survives; three rows do not

The warm-up defect is not resolvable at this size, so these numbers are not
*inflated*. But they are single legs with no error bar, and the protocol's floor is
1.08×, which strikes rows on its own:

| row | verdict |
|---|---|
| `binary_dilate` 0.03×, `connected_component` 0.25×, `signed_maurer` 0.30×, `median` 0.38×, `rescale_intensity` 0.42×, `gmrg` 0.47×, `gradient_magnitude` 0.64×, `mean` 0.80×, `fft_convolution` 0.87× | **survive** — far outside the floor; retake for an error bar, but the claims hold |
| `smoothing_recursive_gaussian` **1.02×** | **NOT A RESULT** — inside the noise floor. It is not "parity", it is unresolved |
| `otsu_threshold` 0.57× | **soft** — the two rust rounds themselves disagree by **1.27×** (46.5 / 36.7), above the floor. The ratio is directionally safe, the number is not |
| `discrete_gaussian` 0.76× | **soft** — rust rounds disagree by 1.16× (114.0 / 131.8), above the floor |
| all `t1` columns | **unverified** — I measured the `t1` warm-up defect only at 64³, where it reaches 1.45×. The medium `t1` cells were not re-measured |

## §"large (512³)" — two "ties" are not ties

| row | verdict |
|---|---|
| `binary_dilate` 0.02×, `connected_component` 0.27×, `median` 0.39×, `rescale_intensity` 0.41×, `signed_maurer` 0.46×, `mean` 0.73×, `fft_convolution` 0.85×, `smoothing_recursive_gaussian` **1.75×** | **survive** — outside the 1.15× floor |
| `discrete_gaussian` 0.91×, `otsu_threshold` 1.37× | **survive, marginally** — outside the floor but by less than 2× the floor; retake |
| `gradient_magnitude` **1.02×**, `gmrg` **1.01×** | **NOT RESULTS.** The document calls these "ties at large, not wins" and declines to claim them, which was the right instinct — but 1.02× and 1.01× are *inside the noise floor* and cannot establish parity either. They are unresolved, not tied |
| the whole size | **the warm-up defect at 512³ is unmeasured.** A `large` cell gets a 2 s measurement window and the ramp is ~2 s of work, so the defect could be *larger* here than at 64³, not smaller. I did not test it and I will not assume it |

## §"The noise floor at `large`, measured" — the conclusion stands, the numbers do not

The ABBA twin's suspect cells (`gradient_magnitude` medium **0.91×**, `gradient_magnitude`
large **0.98×**, `discrete_gaussian` large **1.02×**) are all **inside the noise
floor** — which is exactly the document's own conclusion ("every suspect cell's range
overlaps… a regression the regressing code cannot reproduce is not a regression").
That conclusion is correct and is *strengthened* by this round. The control cells
(**0.34×**, **0.44×**) are far outside the floor and survive as evidence that the
64³ direction is real, though their absolute ms are ramp-inflated.

**"Process shape moves a cell by ~60%"** — this is the op-set mechanism, and it is
now measured rather than observed: in a twelve-op process, short 64³ `tN` cells flip
bimodally by **2×**, and solo they are stable to 1.05–1.12×. The document's rule
("a number is comparable only within the same sweep shape") is too weak: **numbers
from the same sweep shape are not comparable either**, because the mode flips between
runs of the identical binary. One op per process is the only shape that reproduces.

## §1–§2 (GPU, device pipeline) — outside my instrument, flagged

Not measured under this protocol; the CPU/host columns come from `device_pipeline`,
not `bench_ops`. The document already flags its host column as bimodal (`smooth`
flipping between ~1,540 and ~2,300 ms) and already refuses to quote a point estimate
("quote this as ~70×"). That instinct is now explained: a bimodal host column between
runs of an identical binary is the same second mechanism. **The GPU ratios' *host*
denominators are unverified**, so the speedups (62–81×, 88–111×) carry the host's
instability; the device columns are stable in the document's own data.

## What this costs the document

Struck outright: the entire 64³ table (12 rust/itk ratios), plus `smoothing_recursive_gaussian`
medium 1.02×, `gradient_magnitude` large 1.02×, `gmrg` large 1.01×. Marked soft:
`otsu_threshold` and `discrete_gaussian` medium; every `t1` column; every 512³ row's
freedom from the warm-up defect; the GPU sections' host denominators.

That is a shorter document. It is also one where a number means something.
