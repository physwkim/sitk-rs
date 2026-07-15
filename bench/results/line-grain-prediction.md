# Prediction: what the line-pass grain is worth, written before it was measured

Committed on the timing-free round, ahead of any run of the harness on this
change (`3b65049`). It exists to be checked against the box, and to be wrong in
public if it is wrong. The model is stated so a failure identifies *which* step
was false, not merely that the number missed.

## The model

Equal tasks, `P` workers, total work `W`:

    makespan(T) = (W / T) * ceil(T / P)

A task count below `P` leaves workers idle; a task count just above `P` runs a
second, mostly empty wave and buys **nothing**. On this box `P = 96`:

| T | waves | makespan |
|---|---|---|
| 35 | 1 | W/35 |
| 43 | 1 | W/43 |
| 64 | 1 | W/64 |
| 70 | 1 | W/70 |
| 86 | 1 | W/86 |
| **128** | **2** | **W/64 — identical to T=64** |
| 164 | 2 | W/82 |

So doubling a task count only pays when it stays **inside one wave**. `64 → 128`
is a wash on a 96-worker box; `35 → 70` and `43 → 86` are 2× on the pass.

## The predictions

The binding probe (all twelve ops, all three sizes, on the real path) says the
new grain binds on exactly six shapes, all at 64³. Applying the model to each:

**P1 — nothing at 256³ or 512³ moves at all.** Not "moves a little": the emitted
block runs are the same integers, so the decomposition is bit-for-bit the same
work on the same threads. If any `medium` or `large` cell moves outside the
noise band, the integer no-op proof is *wrong* and this change must be reverted,
not tuned.

**P2 — `smoothing_recursive_gaussian`, `gradient_magnitude_recursive_gaussian`
and `signed_maurer_distance_map` at 64³ do not improve. 0%, within noise.** Their
one binding shape is axis 0, `64 → 128` tasks, which the model says is a wash at
`P = 96`. These are the two cells this task was aimed at, and I predict the
change does not move them. If they *do* improve, my model is wrong — most likely
because tasks are not equal-cost or rayon's adaptive splitting was never reaching
64 leaves in the first place.

**P3 — `fft_convolution` at 64³ is the only cell that improves.** Its padded
36×70×70 passes go `43 → 86` (axis 0) and `35 → 70` (axis 1), both inside one
wave: those passes should roughly halve. They are only part of the op — the
axis-2 column-path passes and the pointwise spectrum multiply are untouched — so
the op-level gain is bounded well under 2×. **Central estimate: 15%. Band: >5%
and <2×.** If it comes in under 5%, the line passes are a smaller share of that
op than I think.

**P4 — no checksum moves anywhere.** Already discharged, not predicted:
`bit_parity` 18/18 at 1/4/48/96 (its expected values are pinned constants) and
`cargo nextest run --workspace` 3414/3414.

## What this change is *not* worth

It does not close the cells it was sent at. On this box it is a structural fix —
one grain rule for every pass in the module instead of a second, fixed policy —
that pays only on `fft_convolution` small, plus on pool widths this box does not
have (`P = 48`: utilisation 0.67 → 0.89; `P = 128`: 0.50 → 1.00; `P = 256`: 0.25
→ 0.50). If the panel's bar is "a cell must move", this change does not clear it
on merit and should be judged on the structure and on the two pins it adds.
