//! Deterministic RNG primitives used by [`crate::noise`], ported operation-
//! for-operation from ITK's `itk::Statistics` random variate generators so
//! that a given seed reproduces ITK's own pseudo-random *stream* (see the
//! module's own divergence note on thread decomposition).
//!
//! Two independent generators are ported, matching which one each noise
//! filter's `.hxx` actually instantiates:
//!
//! - [`MersenneTwister`] — `itkMersenneTwisterRandomVariateGenerator.h`/`.cxx`
//!   (Matsumoto/Nishimura/Wagner's MT19937 variant, `IntegerType = uint32_t`).
//!   Used directly by `SaltAndPepperNoiseImageFilter`, `ShotNoiseImageFilter`,
//!   and `SpeckleNoiseImageFilter`.
//! - [`NormalVariateGenerator`] — `itkNormalVariateGenerator.h`/`.cxx`, C.S.
//!   Wallace's pooled-rotation "FastNorm" generator. This is a *different*
//!   algorithm from the Mersenne Twister's own `GetNormalVariate` (a
//!   Box-Muller transform of two MT uniform draws): `AdditiveGaussianNoiseImageFilter`
//!   and `ShotNoiseImageFilter` both construct a
//!   `Statistics::NormalVariateGenerator` for their Gaussian draws, never the
//!   Mersenne Twister's `GetNormalVariate`, so Wallace's algorithm — not
//!   Box-Muller — is the one ported here.
//!
//! `FastNorm`'s C++ source is a goto-driven state machine (`renormalize` /
//! `startpass` / four near-identical `matrixN` rotation passes / `endpass`);
//! [`NormalVariateGenerator::fast_norm`] restructures the same states into
//! plain Rust control flow (loops instead of gotos) without changing any
//! arithmetic step, and keeps the four `matrixN` passes textually separate
//! (mirroring the source's own duplication) rather than unifying them, since
//! each differs from the others only in sign pattern and assignment target —
//! exactly the kind of small per-branch divergence a "clever" unification
//! would risk transcribing incorrectly.

// ---------------------------------------------------------------------
// Mersenne Twister — itkMersenneTwisterRandomVariateGenerator.h / .cxx
// ---------------------------------------------------------------------

const STATE_LEN: usize = 624;
const PERIOD_M: usize = 397;

fn hi_bit(u: u32) -> u32 {
    u & 0x8000_0000
}
fn lo_bit(u: u32) -> u32 {
    u & 0x0000_0001
}
fn lo_bits(u: u32) -> u32 {
    u & 0x7fff_ffff
}
fn mix_bits(u: u32, v: u32) -> u32 {
    hi_bit(u) | lo_bits(v)
}
fn twist(m: u32, s0: u32, s1: u32) -> u32 {
    m ^ (mix_bits(s0, s1) >> 1) ^ ((-(lo_bit(s1) as i32)) as u32 & 0x9908_b0df)
}

/// `itk::Statistics::MersenneTwisterRandomVariateGenerator`, ported from
/// `itkMersenneTwisterRandomVariateGenerator.cxx`. `IntegerType` is `uint32_t`.
pub(crate) struct MersenneTwister {
    state: [u32; STATE_LEN],
    next: usize,
    left: i32,
}

impl MersenneTwister {
    /// `SetSeed`/`InitializeWithoutMutexLocking`: seed the state array with
    /// Knuth's TAOCP Vol.2 LCG, then [`Self::reload`] once so the first draw
    /// needs no separate priming step.
    pub(crate) fn new(seed: u32) -> Self {
        let mut state = [0u32; STATE_LEN];
        state[0] = seed;
        for i in 1..STATE_LEN {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        let mut mt = MersenneTwister {
            state,
            next: 0,
            left: 0,
        };
        mt.reload();
        mt
    }

    /// `reload()`: re-twist the full state array. `p` walks the same path
    /// as the original's raw pointer, split into the same two loops plus the
    /// one final wrap-around element (the last iteration reads `m_State[0]`
    /// instead of `p[1]`, since `p[1]` would run one past the array).
    fn reload(&mut self) {
        const OFFSET: usize = STATE_LEN - PERIOD_M;
        let mut p = 0usize;
        for _ in 0..(STATE_LEN - PERIOD_M) {
            self.state[p] = twist(self.state[p + PERIOD_M], self.state[p], self.state[p + 1]);
            p += 1;
        }
        for _ in 0..(PERIOD_M - 1) {
            self.state[p] = twist(self.state[p - OFFSET], self.state[p], self.state[p + 1]);
            p += 1;
        }
        self.state[STATE_LEN - 1] = twist(
            self.state[STATE_LEN - 1 - OFFSET],
            self.state[STATE_LEN - 1],
            self.state[0],
        );

        self.left = STATE_LEN as i32;
        self.next = 0;
    }

    /// `GetIntegerVariate()`: a tempered `uint32_t` in `[0, 2^32-1]`.
    fn get_integer_variate(&mut self) -> u32 {
        if self.left == 0 {
            self.reload();
        }
        self.left -= 1;

        let mut s1 = self.state[self.next];
        self.next += 1;
        s1 ^= s1 >> 11;
        s1 ^= (s1 << 7) & 0x9d2c_5680;
        s1 ^= (s1 << 15) & 0xefc6_0000;
        s1 ^ (s1 >> 18)
    }

    /// `GetVariateWithClosedRange()` (also `GetVariate()`'s override): uniform
    /// in `[0, 1]`.
    pub(crate) fn get_variate(&mut self) -> f64 {
        self.get_integer_variate() as f64 * (1.0 / u32::MAX as f64)
    }

    /// `GetVariateWithOpenUpperRange()`: uniform in `[0, 1)`.
    pub(crate) fn get_variate_open_upper(&mut self) -> f64 {
        self.get_integer_variate() as f64 / 4_294_967_296.0
    }
}

// ---------------------------------------------------------------------
// NormalVariateGenerator (Wallace's FastNorm) — itkNormalVariateGenerator.h / .cxx
// ---------------------------------------------------------------------

const ELEN: i32 = 7;
const LEN: i32 = 128;
const LMASK: i32 = 4 * (LEN - 1);
const TLEN: usize = 8 * LEN as usize;

const SCALE: f64 = 30_000_000.0;
const RSCALE: f64 = 1.0 / SCALE;
const RCONS: f64 = 1.0 / (2.0 * 1024.0 * 1024.0 * 1024.0);

/// `NormalVariateGenerator::SignedShiftXOR`: the source's own comment notes
/// this assumes two's-complement wraparound, which is exactly what `as u32`
/// followed by a plain `<<` gives in Rust.
fn signed_shift_xor(irs: i32) -> i32 {
    let uirs = irs as u32;
    let shifted = uirs << 1;
    (if irs <= 0 {
        shifted ^ 333_556_017
    } else {
        shifted
    }) as i32
}

/// `m_Lseed = 69069 * (long)m_Lseed + 33331`, truncated back to 32 bits —
/// the multiply is widened to 64 bits first only to avoid relying on 32-bit
/// signed overflow, matching the source's own `long`-widened computation.
fn lcg_step(lseed: i32) -> i32 {
    (69_069i64 * lseed as i64 + 33_331) as i32
}

/// `(long)a + (long)b` truncated back to 32 bits.
fn add_trunc32(a: i32, b: i32) -> i32 {
    (a as i64 + b as i64) as i32
}

/// `itk::Statistics::NormalVariateGenerator`, ported from
/// `itkNormalVariateGenerator.cxx` (C.S. Wallace's "FastNorm").
pub(crate) struct NormalVariateGenerator {
    vec1: [i32; TLEN],
    nslew: i32,
    irs: i32,
    lseed: i32,
    chic1: f64,
    chic2: f64,
    actual_rsd: f64,
    gaussfaze: i32,
    gscale: f64,
}

impl NormalVariateGenerator {
    /// `Initialize(int randomSeed)`.
    pub(crate) fn new(seed: i32) -> Self {
        let fake = 1.0 + 0.125 / TLEN as f64;
        let chic2 = (2.0 * TLEN as f64 - fake * fake).sqrt() / fake;
        let chic1 = fake * (0.5 / TLEN as f64).sqrt();
        NormalVariateGenerator {
            vec1: [0; TLEN],
            nslew: 0,
            irs: seed,
            lseed: seed,
            chic1,
            chic2,
            actual_rsd: 0.0,
            gaussfaze: 1,
            gscale: RSCALE,
        }
    }

    /// `GetVariate()`: pull the next value out of the pool of `TLEN` saved
    /// deviates, refilling with [`Self::fast_norm`] once the pool is spent.
    pub(crate) fn get_variate(&mut self) -> f64 {
        self.gaussfaze -= 1;
        if self.gaussfaze != 0 {
            return self.gscale * self.vec1[self.gaussfaze as usize] as f64;
        }
        self.fast_norm()
    }

    /// `FastNorm()`'s top-level dispatch: `renormalize` (periodically, every
    /// 256 pool refills) then `startpass` (always).
    fn fast_norm(&mut self) -> f64 {
        if self.nslew & 0xFF == 0 {
            self.renormalize();
        }
        self.startpass()
    }

    /// `FastNorm`'s `renormalize:`/`nextpair:` labels: every 65536 refills,
    /// rebuild the whole pool from scratch as ordinary Box-Muller normals
    /// (rejecting draws outside the unit disk), rescaled so its sum of
    /// squares matches a `Chi-Sq(TLEN)` variate; every 256 refills (including
    /// every 65536th), recompute the correction factor in
    /// [`Self::recalcsumsq`].
    fn renormalize(&mut self) {
        if self.nslew & 0xFFFF != 0 {
            self.recalcsumsq();
            return;
        }

        let mut ts = 0.0f64;
        let mut p = 0usize;
        loop {
            self.lseed = lcg_step(self.lseed);
            self.irs = signed_shift_xor(self.irs);
            let r1 = add_trunc32(self.irs, self.lseed);
            let tx = RCONS * r1 as f64;

            self.lseed = lcg_step(self.lseed);
            self.irs = signed_shift_xor(self.irs);
            let r2 = add_trunc32(self.irs, self.lseed);
            let ty = RCONS * r2 as f64;

            let tr = tx * tx + ty * ty;
            if !(0.1..=1.0).contains(&tr) {
                continue;
            }

            self.lseed = lcg_step(self.lseed);
            self.irs = signed_shift_xor(self.irs);
            let mut r3 = add_trunc32(self.irs, self.lseed);
            if r3 < 0 {
                r3 = !r3;
            }
            let tz_sq = -2.0 * ((r3 as f64 + 0.5) * RCONS).ln();
            ts += tz_sq;
            let tz = (tz_sq / tr).sqrt();
            self.vec1[p] = (SCALE * tx * tz) as i32;
            p += 1;
            self.vec1[p] = (SCALE * ty * tz) as i32;
            p += 1;
            if p >= TLEN {
                break;
            }
        }

        let correction = TLEN as f64 / ts;
        let tr = correction.sqrt();
        for slot in self.vec1.iter_mut() {
            let tx = *slot as f64 * tr;
            *slot = if tx < 0.0 {
                (tx - 0.5) as i32
            } else {
                (tx + 0.5) as i32
            };
        }
        self.recalcsumsq();
    }

    /// `FastNorm`'s `recalcsumsq:` label: recompute `m_ActualRSD`, the
    /// reciprocal actual standard deviation of the current pool.
    fn recalcsumsq(&mut self) {
        let mut ts = 0.0f64;
        for &v in self.vec1.iter() {
            let tx = v as f64;
            ts += tx * tx;
        }
        ts = (ts / (SCALE * SCALE * TLEN as f64)).sqrt();
        self.actual_rsd = 1.0 / ts;
    }

    /// `FastNorm`'s `startpass:` label: derive this round's `(skew, stride,
    /// mtype, stype)` from the LCG/shift-XOR stream, run the matching
    /// `matrixN` rotation pass over the pool, then `endpass:`'s correction.
    fn startpass(&mut self) -> f64 {
        self.nslew += 1;
        self.gaussfaze = TLEN as i32 - 1;

        self.lseed = lcg_step(self.lseed);
        self.irs = signed_shift_xor(self.irs);
        let mut t = add_trunc32(self.irs, self.lseed);
        if t < 0 {
            t = !t;
        }
        t >>= 29 - 2 * ELEN;
        let mut skew = (LEN - 1) & t;
        t >>= ELEN;
        skew *= 4;
        let mut stride = (LEN / 2 - 1) & t;
        t >>= ELEN - 1;
        stride = 8 * stride + 4;
        let mtype = t & 3;
        let stype = self.nslew & 3;

        // `pa`/`pb`/`pc`/`pd`/`p0`/`pe` mirror the source's raw `int *`
        // pointers as `i32` offsets rather than `usize`: the matching
        // `matrixN` pass below decrements one of them one step past its own
        // base on its very last iteration (a value that pointer arithmetic
        // in C++ can legally form without dereferencing, but that `usize`
        // cannot represent), so these stay signed until the point of use.
        let inc: i32;
        let mask: i32;
        let (mut pa, mut pb, mut pc, mut pd, p0): (i32, i32, i32, i32, i32);
        let len = LEN;
        match stype {
            0 => {
                inc = 1;
                mask = LMASK;
                pa = 0;
                pb = pa + len;
                pc = pb + len;
                pd = pc + len;
                p0 = 4 * len;
            }
            1 => {
                inc = 1;
                mask = LMASK;
                pa = 4 * len;
                pb = pa + len;
                pc = pb + len;
                pd = pc + len;
                p0 = 0;
            }
            2 => {
                inc = 2;
                mask = 2 * LMASK;
                skew *= 2;
                stride *= 2;
                pa = 1;
                pb = pa + 2 * len;
                pc = pb + 2 * len;
                pd = pc + 2 * len;
                p0 = 0;
            }
            _ => {
                inc = 2;
                mask = 2 * LMASK;
                skew *= 2;
                stride *= 2;
                pa = 0;
                pb = pa + 2 * len;
                pc = pb + 2 * len;
                pd = pc + 2 * len;
                p0 = 1;
            }
        }

        match mtype {
            0 => {
                pa += inc * (len - 1);
                let mut i = LEN;
                loop {
                    skew = (skew + stride) & mask;
                    let mut pe = p0 + skew;
                    let mut p = -self.vec1[pa as usize];
                    let mut q = -self.vec1[pb as usize];
                    let mut r = self.vec1[pc as usize];
                    let mut s = self.vec1[pd as usize];
                    let mut t = (p + q + r + s) >> 1;
                    p = t - p;
                    q = t - q;
                    r = t - r;
                    s = t - s;

                    t = -self.vec1[pe as usize];
                    self.vec1[pe as usize] = p;
                    pe += inc;
                    p = self.vec1[pe as usize];
                    self.vec1[pe as usize] = q;
                    pe += inc;
                    q = -self.vec1[pe as usize];
                    self.vec1[pe as usize] = r;
                    pe += inc;
                    r = self.vec1[pe as usize];
                    self.vec1[pe as usize] = s;

                    s = (p + q + r + t) >> 1;
                    self.vec1[pa as usize] = s - p;
                    pa -= inc;
                    self.vec1[pb as usize] = s - q;
                    pb += inc;
                    self.vec1[pc as usize] = s - r;
                    pc += inc;
                    self.vec1[pd as usize] = s - t;
                    pd += inc;

                    i -= 1;
                    if i == 0 {
                        break;
                    }
                }
            }
            1 => {
                pb += inc * (len - 1);
                let mut i = LEN;
                loop {
                    skew = (skew + stride) & mask;
                    let mut pe = p0 + skew;
                    let mut p = -self.vec1[pa as usize];
                    let mut q = self.vec1[pb as usize];
                    let mut r = self.vec1[pc as usize];
                    let mut s = -self.vec1[pd as usize];
                    let mut t = (p + q + r + s) >> 1;
                    p = t - p;
                    q = t - q;
                    r = t - r;
                    s = t - s;

                    t = self.vec1[pe as usize];
                    self.vec1[pe as usize] = p;
                    pe += inc;
                    p = -self.vec1[pe as usize];
                    self.vec1[pe as usize] = q;
                    pe += inc;
                    q = -self.vec1[pe as usize];
                    self.vec1[pe as usize] = r;
                    pe += inc;
                    r = self.vec1[pe as usize];
                    self.vec1[pe as usize] = s;

                    s = (p + q + r + t) >> 1;
                    self.vec1[pa as usize] = s - p;
                    pa += inc;
                    self.vec1[pb as usize] = s - t;
                    pb -= inc;
                    self.vec1[pc as usize] = s - q;
                    pc += inc;
                    self.vec1[pd as usize] = s - r;
                    pd += inc;

                    i -= 1;
                    if i == 0 {
                        break;
                    }
                }
            }
            2 => {
                pc += inc * (len - 1);
                let mut i = LEN;
                loop {
                    skew = (skew + stride) & mask;
                    let mut pe = p0 + skew;
                    let mut p = self.vec1[pa as usize];
                    let mut q = -self.vec1[pb as usize];
                    let mut r = self.vec1[pc as usize];
                    let mut s = -self.vec1[pd as usize];
                    let mut t = (p + q + r + s) >> 1;
                    p = t - p;
                    q = t - q;
                    r = t - r;
                    s = t - s;

                    t = self.vec1[pe as usize];
                    self.vec1[pe as usize] = p;
                    pe += inc;
                    p = self.vec1[pe as usize];
                    self.vec1[pe as usize] = q;
                    pe += inc;
                    q = -self.vec1[pe as usize];
                    self.vec1[pe as usize] = r;
                    pe += inc;
                    r = -self.vec1[pe as usize];
                    self.vec1[pe as usize] = s;

                    s = (p + q + r + t) >> 1;
                    self.vec1[pa as usize] = s - r;
                    pa += inc;
                    self.vec1[pb as usize] = s - p;
                    pb += inc;
                    self.vec1[pc as usize] = s - q;
                    pc -= inc;
                    self.vec1[pd as usize] = s - t;
                    pd += inc;

                    i -= 1;
                    if i == 0 {
                        break;
                    }
                }
            }
            _ => {
                pd += inc * (len - 1);
                let mut i = LEN;
                loop {
                    skew = (skew + stride) & mask;
                    let mut pe = p0 + skew;
                    let mut p = self.vec1[pa as usize];
                    let mut q = self.vec1[pb as usize];
                    let mut r = -self.vec1[pc as usize];
                    let mut s = -self.vec1[pd as usize];
                    let mut t = (p + q + r + s) >> 1;
                    p = t - p;
                    q = t - q;
                    r = t - r;
                    s = t - s;

                    t = -self.vec1[pe as usize];
                    self.vec1[pe as usize] = p;
                    pe += inc;
                    p = self.vec1[pe as usize];
                    self.vec1[pe as usize] = q;
                    pe += inc;
                    q = self.vec1[pe as usize];
                    self.vec1[pe as usize] = r;
                    pe += inc;
                    r = -self.vec1[pe as usize];
                    self.vec1[pe as usize] = s;

                    s = (p + q + r + t) >> 1;
                    self.vec1[pa as usize] = s - q;
                    pa += inc;
                    self.vec1[pb as usize] = s - r;
                    pb += inc;
                    self.vec1[pc as usize] = s - t;
                    pc += inc;
                    self.vec1[pd as usize] = s - p;
                    pd -= inc;

                    i -= 1;
                    if i == 0 {
                        break;
                    }
                }
            }
        }

        // endpass:
        let ts = self.chic1 * (self.chic2 + self.gscale * self.vec1[TLEN - 1] as f64);
        self.gscale = RSCALE * ts * self.actual_rsd;
        self.gscale * self.vec1[0] as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_twister_same_seed_is_deterministic() {
        let mut a = MersenneTwister::new(42);
        let mut b = MersenneTwister::new(42);
        let seq_a: Vec<f64> = (0..2000).map(|_| a.get_variate()).collect();
        let seq_b: Vec<f64> = (0..2000).map(|_| b.get_variate()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn mersenne_twister_different_seeds_diverge() {
        let mut a = MersenneTwister::new(1);
        let mut b = MersenneTwister::new(2);
        let seq_a: Vec<f64> = (0..16).map(|_| a.get_variate()).collect();
        let seq_b: Vec<f64> = (0..16).map(|_| b.get_variate()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn mersenne_twister_closed_range_stays_in_0_1() {
        let mut mt = MersenneTwister::new(7);
        for _ in 0..5000 {
            let v = mt.get_variate();
            assert!((0.0..=1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn mersenne_twister_open_upper_range_never_reaches_1() {
        let mut mt = MersenneTwister::new(7);
        for _ in 0..5000 {
            let v = mt.get_variate_open_upper();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn mersenne_twister_survives_multiple_reloads() {
        // STATE_LEN draws exhausts one reload's worth; pull several times
        // that many to exercise `reload()` being called again mid-stream.
        let mut mt = MersenneTwister::new(99);
        let draws: Vec<f64> = (0..(STATE_LEN * 3 + 5)).map(|_| mt.get_variate()).collect();
        assert!(draws.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn normal_variate_generator_same_seed_is_deterministic() {
        let mut a = NormalVariateGenerator::new(42);
        let mut b = NormalVariateGenerator::new(42);
        let seq_a: Vec<f64> = (0..3000).map(|_| a.get_variate()).collect();
        let seq_b: Vec<f64> = (0..3000).map(|_| b.get_variate()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn normal_variate_generator_different_seeds_diverge() {
        let mut a = NormalVariateGenerator::new(1);
        let mut b = NormalVariateGenerator::new(2);
        let seq_a: Vec<f64> = (0..16).map(|_| a.get_variate()).collect();
        let seq_b: Vec<f64> = (0..16).map(|_| b.get_variate()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn normal_variate_generator_survives_pool_exhaustion_and_periodic_renormalize() {
        // TLEN draws exhausts the pool once; 257*TLEN crosses the
        // `nslew & 0xFF == 0` renormalize boundary at least once.
        let mut mt = NormalVariateGenerator::new(123);
        let n = TLEN * 257 + 17;
        let draws: Vec<f64> = (0..n).map(|_| mt.get_variate()).collect();
        // A crude sanity bound: standard-normal draws essentially never
        // exceed +-10 in any finite sample of this size.
        assert!(draws.iter().all(|v| v.is_finite() && v.abs() < 10.0));
    }

    #[test]
    fn normal_variate_generator_is_approximately_standard_normal() {
        let mut mt = NormalVariateGenerator::new(2024);
        let n = 20_000;
        let draws: Vec<f64> = (0..n).map(|_| mt.get_variate()).collect();
        let mean: f64 = draws.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        assert!(mean.abs() < 0.1, "mean drifted too far from 0: {mean}");
        assert!(
            (variance - 1.0).abs() < 0.15,
            "variance drifted too far from 1: {variance}"
        );
    }
}
