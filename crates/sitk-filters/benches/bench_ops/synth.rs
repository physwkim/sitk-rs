//! Deterministic input generator, `doc/bench-spec.md` §"Inputs".
//!
//! The xorshift64* update rule, its "seeded per-volume" scoping (state resets
//! to `seed` once per volume, not once per voxel and not once for the whole
//! suite), and the numeric seed itself all come from the spec, which fixes
//! `SEED = 42` for every volume/size/harness (amended after the first C++
//! run — the original left the seed unstated, voiding `input_checksum`
//! equality). The spec's own reference: seed 42's first five voxels are `59,
//! 641, 384, 121, 923`; [`synth`] is checked against that literally in
//! `tests/bench_correctness_gate.rs`.
pub const SEED: u64 = 42;

/// `xorshift64*` mapped into `[0, 1000)`, one call per synthesized volume.
///
/// Voxels are produced in flat order `0..n`; that is first-index-fastest
/// (dimension 0 fastest), the same layout `sitk_core::Image::from_vec`
/// expects, so the returned buffer can be handed to it directly with no
/// transposition.
pub fn synth(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) % 1000) as f32
        })
        .collect()
}

/// `doc/bench-spec.md` §"The three input variants": `mask_u8` — `base >=
/// 500.0 ? 1 : 0`, `UInt8`. Ops 8 (`binary_dilate`) and 10
/// (`connected_component`).
pub fn threshold_u8(raw: &[f32]) -> Vec<u8> {
    raw.iter().map(|&v| u8::from(v >= 500.0)).collect()
}

/// `doc/bench-spec.md` §"The three input variants": `mask_f32` — the same
/// `>= 500.0` threshold, kept as `Float32` 0.0/1.0 (binary *content* in a
/// Float32 *type*). Op 9 (`signed_maurer_distance_map`) only, with
/// `background_value = 0.0`.
pub fn threshold_f32(raw: &[f32]) -> Vec<f32> {
    raw.iter()
        .map(|&v| if v >= 500.0 { 1.0 } else { 0.0 })
        .collect()
}
