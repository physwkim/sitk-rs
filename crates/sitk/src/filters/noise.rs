//! ITK's image-noise family, ported from
//! `Modules/Filtering/ImageNoise/include/` (`itkNoiseBaseImageFilter.h`/`.hxx`,
//! `itkAdditiveGaussianNoiseImageFilter.h`/`.hxx`,
//! `itkSaltAndPepperNoiseImageFilter.h`/`.hxx`,
//! `itkShotNoiseImageFilter.h`/`.hxx`, `itkSpeckleNoiseImageFilter.h`/`.hxx`),
//! with parameter names/defaults from `SimpleITK/Code/BasicFilters/yaml/`'s
//! definitions of the same four filters.
//!
//! Every filter here draws from one of the two RNGs in [`crate::filters::random`]:
//! [`additive_gaussian_noise`] and [`shot_noise`] use
//! `Statistics::NormalVariateGenerator` (C.S. Wallace's "FastNorm", *not* the
//! Mersenne Twister's own Box-Muller `GetNormalVariate` — see
//! [`crate::filters::random`]'s module doc comment for why that distinction matters);
//! [`salt_and_pepper_noise`], [`shot_noise`]'s Poisson branch, and
//! [`speckle_noise`] draw uniform variates from
//! `Statistics::MersenneTwisterRandomVariateGenerator` directly. This matches
//! each `.hxx`'s own `ThreadedGenerateData` exactly: `AdditiveGaussianNoiseImageFilter`
//! and `ShotNoiseImageFilter` each construct both a
//! `MersenneTwisterRandomVariateGenerator` (`ShotNoiseImageFilter` only) and a
//! `Statistics::NormalVariateGenerator`, while `SaltAndPepperNoiseImageFilter`
//! and `SpeckleNoiseImageFilter` construct only the Mersenne Twister.
//!
//! **Seeding.** `NoiseBaseImageFilter::Hash(a, b) = (a + b) * 2654435761u`
//! (Knuth's multiplicative hash) reseeds a fresh generator per output region,
//! `b` being the sum of that region's starting index components
//! (`indSeed`). ITK runs one thread per region, each with its own `indSeed`,
//! so a multi-threaded run reseeds many times across an image. This port has
//! no thread decomposition — the whole image is one region starting at index
//! zero, so `indSeed == 0` always and every filter here reseeds with
//! `Hash(seed, 0)` exactly once. This reproduces ITK's own **single-threaded**
//! output bit-for-bit (a one-region run also has `indSeed == 0`), but not
//! ITK's default **multi-threaded** output, whose per-region reseeding
//! produces a different stream than one continuous region does; this
//! divergence is intentional and reported under UNFIXED rather than chasing
//! ITK's thread decomposition.
//!
//! **Clamping.** `NoiseBaseImageFilter::ClampCast` clamps the computed
//! `double` to `[NumericTraits<T>::NonpositiveMin(), NumericTraits<T>::max()]`,
//! then for integer `T` rounds half-up (`Math::Round` ==
//! `RoundHalfIntegerUp`, i.e. `floor(value + 0.5)`) rather than truncating;
//! [`clamp_cast`] mirrors this exactly (unlike [`crate::core::Scalar::from_f64`]'s
//! plain saturating-truncating `as` cast used elsewhere in this crate).

use crate::core::{Image, Scalar, dispatch_scalar};
use crate::filters::error::Result;
use crate::filters::random::{MersenneTwister, NormalVariateGenerator};

/// `NumericTraits<T>::max()` / `NonpositiveMin()`, the bounds
/// [`clamp_cast`] clamps against.
trait ClampBounds: Scalar {
    const IS_INTEGER: bool;
    fn clamp_max() -> Self;
    fn clamp_nonpositive_min() -> Self;
}

macro_rules! impl_clamp_bounds_unsigned {
    ($($t:ty),+ $(,)?) => {$(
        impl ClampBounds for $t {
            const IS_INTEGER: bool = true;
            fn clamp_max() -> Self { <$t>::MAX }
            fn clamp_nonpositive_min() -> Self { 0 }
        }
    )+};
}

macro_rules! impl_clamp_bounds_signed {
    ($($t:ty),+ $(,)?) => {$(
        impl ClampBounds for $t {
            const IS_INTEGER: bool = true;
            fn clamp_max() -> Self { <$t>::MAX }
            fn clamp_nonpositive_min() -> Self { <$t>::MIN }
        }
    )+};
}

macro_rules! impl_clamp_bounds_float {
    ($($t:ty),+ $(,)?) => {$(
        impl ClampBounds for $t {
            const IS_INTEGER: bool = false;
            fn clamp_max() -> Self { <$t>::MAX }
            // NumericTraits<float>::NonpositiveMin() is the most negative
            // finite value (`lowest()`), which is exactly `MIN` for Rust
            // floats (unlike integer `MIN`, which is the smallest value).
            fn clamp_nonpositive_min() -> Self { <$t>::MIN }
        }
    )+};
}

impl_clamp_bounds_unsigned!(u8, u16, u32, u64);
impl_clamp_bounds_signed!(i8, i16, i32, i64);
impl_clamp_bounds_float!(f32, f64);

/// `NoiseBaseImageFilter::ClampCast`: clamp to `[NonpositiveMin, max]`, then
/// round-half-up for integer output types or cast directly for
/// floating-point ones.
fn clamp_cast<T: ClampBounds>(value: f64) -> T {
    let max = T::clamp_max().as_f64();
    let min = T::clamp_nonpositive_min().as_f64();
    if value >= max {
        return T::clamp_max();
    }
    if value <= min {
        return T::clamp_nonpositive_min();
    }
    if T::IS_INTEGER {
        T::from_f64((value + 0.5).floor())
    } else {
        T::from_f64(value)
    }
}

/// `NoiseBaseImageFilter::Hash(a, b) = (a + b) * 2654435761u`, specialized to
/// `b == 0` (this port's single, whole-image region) per the module doc
/// comment.
fn thread_seed(seed: u32) -> u32 {
    seed.wrapping_mul(2_654_435_761)
}

// ---- additive_gaussian_noise -----------------------------------------

fn additive_gaussian_noise_typed<T: ClampBounds>(
    img: &Image,
    mean: f64,
    standard_deviation: f64,
    seed: u32,
) -> Result<Image> {
    let input = img.scalar_slice::<T>()?;
    let mut randn = NormalVariateGenerator::new(thread_seed(seed) as i32);

    let out: Vec<T> = input
        .iter()
        .map(|&v| clamp_cast(v.as_f64() + mean + standard_deviation * randn.get_variate()))
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `AdditiveGaussianNoiseImageFilter`: `I = I0 + mean + standard_deviation *
/// N(0, 1)`, `N(0, 1)` drawn from [`NormalVariateGenerator`]. Output keeps
/// `img`'s pixel type, narrowed via [`clamp_cast`].
pub fn additive_gaussian_noise(
    img: &Image,
    mean: f64,
    standard_deviation: f64,
    seed: u32,
) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        additive_gaussian_noise_typed,
        img,
        mean,
        standard_deviation,
        seed
    )
}

// ---- salt_and_pepper_noise ---------------------------------------------

fn salt_and_pepper_noise_typed<T: ClampBounds>(
    img: &Image,
    probability: f64,
    seed: u32,
) -> Result<Image> {
    let input = img.scalar_slice::<T>()?;
    let mut rand = MersenneTwister::new(thread_seed(seed));
    let salt = T::clamp_max();
    let pepper = T::clamp_nonpositive_min();

    let out: Vec<T> = input
        .iter()
        .map(|&v| {
            if rand.get_variate() < probability {
                if rand.get_variate() < 0.5 {
                    salt
                } else {
                    pepper
                }
            } else {
                v
            }
        })
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `SaltAndPepperNoiseImageFilter`: with probability `probability`, replace a
/// pixel with `NumericTraits<T>::max()` (salt, 50% of triggers) or
/// `NumericTraits<T>::NonpositiveMin()` (pepper, the other 50%); otherwise
/// pass it through unchanged. `SaltValue`/`PepperValue` are always ITK's
/// type-derived defaults here, matching the SimpleITK procedural surface
/// (which never exposes them as settable parameters).
pub fn salt_and_pepper_noise(img: &Image, probability: f64, seed: u32) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        salt_and_pepper_noise_typed,
        img,
        probability,
        seed
    )
}

// ---- shot_noise ----------------------------------------------------------

fn shot_noise_typed<T: ClampBounds>(img: &Image, scale: f64, seed: u32) -> Result<Image> {
    let input = img.scalar_slice::<T>()?;
    let s = thread_seed(seed);
    let mut rand = MersenneTwister::new(s);
    let mut randn = NormalVariateGenerator::new(s as i32);

    let out: Vec<T> = input
        .iter()
        .map(|&v| {
            let scaled = scale * v.as_f64();
            // The Poisson/Gaussian-approximation switchover: `scaled < 50` is
            // ITK's own hardcoded threshold (`itkShotNoiseImageFilter.hxx`'s
            // comment: "the lambda value ... where a Gaussian ... is a good
            // approximation of the Poisson").
            if scaled < 50.0 {
                let acceptance = (-scaled).exp();
                let mut k: i64 = 0;
                let mut p = 1.0;
                loop {
                    k += 1;
                    p *= rand.get_variate();
                    if p <= acceptance {
                        break;
                    }
                }
                clamp_cast((k - 1) as f64 / scale)
            } else {
                let gaussian_approx = scaled + scaled.sqrt() * randn.get_variate();
                clamp_cast(gaussian_approx / scale)
            }
        })
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `ShotNoiseImageFilter`: `I = N(I0 * scale) / scale`, `N(lambda)` a
/// Poisson-distributed variate of mean `lambda` — sampled by the classic
/// repeated-uniform-product rejection method when `lambda < 50`, or by its
/// `lambda + sqrt(lambda) * N(0,1)` Gaussian approximation once `lambda`
/// grows too large for the exact method to stay efficient. Output keeps
/// `img`'s pixel type, narrowed via [`clamp_cast`].
pub fn shot_noise(img: &Image, scale: f64, seed: u32) -> Result<Image> {
    dispatch_scalar!(img.pixel_id(), shot_noise_typed, img, scale, seed)
}

// ---- speckle_noise --------------------------------------------------------

fn speckle_noise_typed<T: ClampBounds>(
    img: &Image,
    standard_deviation: f64,
    seed: u32,
) -> Result<Image> {
    let input = img.scalar_slice::<T>()?;
    let mut rand = MersenneTwister::new(thread_seed(seed));

    // Gamma(shape = 1/theta, scale = theta) sampling via Ahrens & Dieter's
    // GD algorithm, exactly as `itkSpeckleNoiseImageFilter.hxx` transcribes
    // it (see https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables).
    let theta = standard_deviation * standard_deviation;
    let k = 1.0 / theta;
    let floork = k.floor();
    let delta = k - floork;
    let v0 = std::f64::consts::E / (std::f64::consts::E + delta);
    let floork_count = floork as i64;

    let out: Vec<T> = input
        .iter()
        .map(|&v| {
            let xi = loop {
                let v1 = 1.0 - rand.get_variate_open_upper();
                let v2 = 1.0 - rand.get_variate_open_upper();
                let v3 = 1.0 - rand.get_variate_open_upper();
                let (xi, nu) = if v1 <= v0 {
                    let xi = v2.powf(1.0 / delta);
                    let nu = v3 * xi.powf(delta - 1.0);
                    (xi, nu)
                } else {
                    let xi = 1.0 - v2.ln();
                    let nu = v3 * (-xi).exp();
                    (xi, nu)
                };
                if nu <= (-xi).exp() * xi.powf(delta - 1.0) {
                    break xi;
                }
            };

            let mut gamma = xi;
            for _ in 0..floork_count {
                gamma -= (1.0 - rand.get_variate_open_upper()).ln();
            }
            gamma *= theta;

            clamp_cast(gamma * v.as_f64())
        })
        .collect();
    let mut result = Image::from_vec(img.size(), out)?;
    result.copy_geometry_from(img);
    Ok(result)
}

/// `SpeckleNoiseImageFilter`: multiplicative `I = I0 * G`, `G` a
/// gamma-distributed variate of mean 1 and variance `standard_deviation^2`
/// (shape `1/standard_deviation^2`, scale `standard_deviation^2`). Output
/// keeps `img`'s pixel type, narrowed via [`clamp_cast`].
///
/// `standard_deviation` at or extremely near `0.0` makes the shape parameter
/// `1/standard_deviation^2` diverge; ITK's own algorithm has no guard against
/// this (`itkSpeckleNoiseImageFilter.hxx`'s inner correction loop runs
/// `floor(1/standard_deviation^2)` times per pixel), so neither does this
/// port — see this crate's noise-porting report for the UNFIXED note.
pub fn speckle_noise(img: &Image, standard_deviation: f64, seed: u32) -> Result<Image> {
    dispatch_scalar!(
        img.pixel_id(),
        speckle_noise_typed,
        img,
        standard_deviation,
        seed
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

    fn flat_u8(size: &[usize], value: u8) -> Image {
        let n: usize = size.iter().product();
        Image::from_vec(size, vec![value; n]).unwrap()
    }

    fn flat_f64(size: &[usize], value: f64) -> Image {
        let n: usize = size.iter().product();
        Image::from_vec(size, vec![value; n]).unwrap()
    }

    // ---- additive_gaussian_noise ----

    #[test]
    fn additive_gaussian_zero_mean_zero_std_is_identity() {
        let img = Image::from_vec(&[4, 4], (0u8..16).collect()).unwrap();
        let out = additive_gaussian_noise(&img, 0.0, 0.0, 123).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            img.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn additive_gaussian_same_seed_is_deterministic() {
        let img = flat_u8(&[8, 8], 128);
        let a = additive_gaussian_noise(&img, 0.0, 20.0, 7).unwrap();
        let b = additive_gaussian_noise(&img, 0.0, 20.0, 7).unwrap();
        assert_eq!(
            a.scalar_slice::<u8>().unwrap(),
            b.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn additive_gaussian_different_seeds_diverge() {
        let img = flat_u8(&[8, 8], 128);
        let a = additive_gaussian_noise(&img, 0.0, 20.0, 1).unwrap();
        let b = additive_gaussian_noise(&img, 0.0, 20.0, 2).unwrap();
        assert_ne!(
            a.scalar_slice::<u8>().unwrap(),
            b.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn additive_gaussian_clamps_at_u8_extremes() {
        let low = flat_u8(&[6, 6], 0);
        let out_low = additive_gaussian_noise(&low, -1000.0, 1.0, 5).unwrap();
        assert!(
            out_low
                .scalar_slice::<u8>()
                .unwrap()
                .iter()
                .all(|&v| v == 0)
        );

        let high = flat_u8(&[6, 6], 255);
        let out_high = additive_gaussian_noise(&high, 1000.0, 1.0, 5).unwrap();
        assert!(
            out_high
                .scalar_slice::<u8>()
                .unwrap()
                .iter()
                .all(|&v| v == 255)
        );
    }

    #[test]
    fn additive_gaussian_sample_mean_and_variance_match_within_tolerance() {
        let img = flat_f64(&[64, 64], 1000.0);
        let out = additive_gaussian_noise(&img, 0.0, 10.0, 99).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();
        let n = vals.len() as f64;
        let mean: f64 = vals.iter().sum::<f64>() / n;
        let variance: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert!(
            (mean - 1000.0).abs() < 1.0,
            "sample mean {mean} too far from 1000.0"
        );
        assert!(
            (variance - 100.0).abs() < 20.0,
            "sample variance {variance} too far from 100.0"
        );
    }

    // ---- salt_and_pepper_noise ----

    #[test]
    fn salt_and_pepper_zero_probability_is_identity() {
        let img = Image::from_vec(&[5, 5], (0u8..25).collect()).unwrap();
        let out = salt_and_pepper_noise(&img, 0.0, 42).unwrap();
        assert_eq!(
            out.scalar_slice::<u8>().unwrap(),
            img.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn salt_and_pepper_same_seed_is_deterministic() {
        let img = flat_u8(&[10, 10], 100);
        let a = salt_and_pepper_noise(&img, 0.3, 11).unwrap();
        let b = salt_and_pepper_noise(&img, 0.3, 11).unwrap();
        assert_eq!(
            a.scalar_slice::<u8>().unwrap(),
            b.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn salt_and_pepper_different_seeds_diverge() {
        let img = flat_u8(&[10, 10], 100);
        let a = salt_and_pepper_noise(&img, 0.3, 1).unwrap();
        let b = salt_and_pepper_noise(&img, 0.3, 2).unwrap();
        assert_ne!(
            a.scalar_slice::<u8>().unwrap(),
            b.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn salt_and_pepper_probability_one_hits_only_extremes() {
        let img = flat_u8(&[10, 10], 100);
        let out = salt_and_pepper_noise(&img, 1.0, 3).unwrap();
        assert!(
            out.scalar_slice::<u8>()
                .unwrap()
                .iter()
                .all(|&v| v == 0 || v == 255)
        );
    }

    #[test]
    fn salt_and_pepper_uses_pixel_type_extremes_for_salt_and_pepper_values() {
        let img = Image::from_vec(&[20, 20], vec![50u8; 400]).unwrap();
        let out = salt_and_pepper_noise(&img, 1.0, 9).unwrap();
        let vals = out.scalar_slice::<u8>().unwrap();
        assert!(vals.contains(&u8::MAX));
        assert!(vals.contains(&u8::MIN));
    }

    // ---- shot_noise ----

    #[test]
    fn shot_noise_same_seed_is_deterministic() {
        let img = flat_u8(&[8, 8], 40);
        let a = shot_noise(&img, 1.0, 21).unwrap();
        let b = shot_noise(&img, 1.0, 21).unwrap();
        assert_eq!(
            a.scalar_slice::<u8>().unwrap(),
            b.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn shot_noise_different_seeds_diverge() {
        let img = flat_u8(&[8, 8], 40);
        let a = shot_noise(&img, 1.0, 1).unwrap();
        let b = shot_noise(&img, 1.0, 2).unwrap();
        assert_ne!(
            a.scalar_slice::<u8>().unwrap(),
            b.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn shot_noise_poisson_branch_below_switchover_stays_near_input() {
        // scale=1, pixel=10 => scaled=10 < 50: Poisson branch, mean 10.
        let img = flat_u8(&[32, 32], 10);
        let out = shot_noise(&img, 1.0, 55).unwrap();
        let vals = out.scalar_slice::<u8>().unwrap();
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        assert!((mean - 10.0).abs() < 3.0, "poisson mean drifted: {mean}");
    }

    #[test]
    fn shot_noise_gaussian_branch_at_and_above_switchover_stays_near_input() {
        // scale=1, pixel=60 => scaled=60 >= 50: Gaussian-approximation branch.
        let img = flat_f64(&[32, 32], 60.0);
        let out = shot_noise(&img, 1.0, 55).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            (mean - 60.0).abs() < 5.0,
            "gaussian-approx mean drifted: {mean}"
        );
    }

    #[test]
    fn shot_noise_clamps_at_u8_max() {
        // scale=0.1, pixel=255 => scaled=25.5 < 50: Poisson branch with mean
        // 25.5 divided back by scale (*10). The spread is wide enough that,
        // over a large sample, some draws divide back to above u8::MAX and
        // must clamp rather than wrap. Note the mean itself cannot be forced
        // past 255 (shot noise preserves the input's mean), so unlike
        // additive_gaussian's fixed-offset test this can only assert that
        // clamping fires for at least one pixel, not for all of them.
        let img = flat_u8(&[20, 20], 255);
        let out = shot_noise(&img, 0.1, 5).unwrap();
        assert!(out.scalar_slice::<u8>().unwrap().contains(&255));
    }

    // ---- speckle_noise ----

    #[test]
    fn speckle_same_seed_is_deterministic() {
        let img = flat_u8(&[10, 10], 100);
        let a = speckle_noise(&img, 1.0, 17).unwrap();
        let b = speckle_noise(&img, 1.0, 17).unwrap();
        assert_eq!(
            a.scalar_slice::<u8>().unwrap(),
            b.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn speckle_different_seeds_diverge() {
        let img = flat_u8(&[10, 10], 100);
        let a = speckle_noise(&img, 1.0, 1).unwrap();
        let b = speckle_noise(&img, 1.0, 2).unwrap();
        assert_ne!(
            a.scalar_slice::<u8>().unwrap(),
            b.scalar_slice::<u8>().unwrap()
        );
    }

    #[test]
    fn speckle_sample_mean_multiplier_is_near_one() {
        let img = flat_f64(&[64, 64], 1000.0);
        let out = speckle_noise(&img, 0.5, 2024).unwrap();
        let vals = out.scalar_slice::<f64>().unwrap();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        // Gamma noise has mean 1, so the output's mean should track the
        // constant input value, not drift toward 0 or blow up.
        assert!(
            (mean - 1000.0).abs() < 100.0,
            "speckle sample mean {mean} too far from input 1000.0"
        );
    }

    #[test]
    fn speckle_clamps_at_u8_max() {
        let img = flat_u8(&[10, 10], 255);
        let out = speckle_noise(&img, 5.0, 3).unwrap();
        // A large standard_deviation with an already-saturated input should
        // push at least some multiplicative draws above 255.
        assert!(out.scalar_slice::<u8>().unwrap().contains(&255));
    }

    // ---- output type / geometry ----

    #[test]
    fn noise_filters_preserve_pixel_type_and_geometry() {
        let mut img = Image::from_vec(&[4, 4], vec![10.0f32; 16]).unwrap();
        img.set_spacing(&[2.0, 3.0]).unwrap();
        img.set_origin(&[1.0, -1.0]).unwrap();

        let out = additive_gaussian_noise(&img, 0.0, 1.0, 1).unwrap();
        assert_eq!(out.pixel_id(), PixelId::Float32);
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
    }
}
