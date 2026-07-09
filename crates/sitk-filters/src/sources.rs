//! Image-source generators: filters with no input image, ported from
//! `Modules/Filtering/ImageSources/include/itkGaussianImageSource.h`/`.hxx`
//! ([`gaussian_source`]), `itkGaborImageSource.h`/`.hxx` ([`gabor_source`]),
//! and `itkGridImageSource.h`/`.hxx` ([`grid_source`]).
//!
//! `PhysicalPointImageSource` (`PhysicalPointImageSource.yaml`) is **not**
//! ported: its `pixel_types` is `VectorPixelIDTypeList` and its default
//! output pixel type is `sitkVectorFloat32` — it produces a vector image with
//! one component per dimension. This crate's [`Image`]/`PixelBuffer` model is
//! scalar-only (ten scalar variants, see `sitk_core::pixel::PixelId`), so
//! there is no representation for a multi-component pixel to port it onto.
//!
//! Every source in this module shares the same physical-space placement
//! surface ([`SourceGeometry`]): `size` fixes the output dimension and pixel
//! grid; `origin`/`spacing`/`direction` place it in physical space exactly as
//! the ported filter's own members do, there being no input image to inherit
//! geometry from. `direction` follows SimpleITK's own documented shorthand
//! (`Code/Common/include/sitkTemplateFunctions.h`'s `sitkSTLToITKDirection`):
//! an empty vector defaults to the identity matrix; any other length that
//! isn't exactly `dim*dim` is an error. Unlike SimpleITK's generated
//! wrappers, which truncate a too-long per-axis vector (`sigma`, `mean`, ...)
//! via `sitkSTLVectorToITK` and only error when one is too short, this port
//! requires every per-axis vector's length to equal the working dimension
//! exactly, matching this crate's own `require_dim`/`Image::set_spacing`
//! convention (see `geometry.rs`) rather than reintroducing that truncation
//! rule.

use crate::error::{FilterError, Result};
use sitk_core::{Image, PixelId, Scalar, dispatch_scalar, matrix};

// ---- shared geometry -------------------------------------------------

/// Physical-space placement shared by every source in this module —
/// `Origin`/`Spacing`/`Direction` are identical members (down to their yaml
/// defaults) across the ported filters.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceGeometry {
    /// Physical coordinate of index zero. Must have one entry per dimension.
    pub origin: Vec<f64>,
    /// Physical spacing between pixels along each axis. Must have one entry
    /// per dimension.
    pub spacing: Vec<f64>,
    /// Row-major `dim x dim` direction cosine matrix. Empty defaults to the
    /// identity matrix (`sitkSTLToITKDirection`'s documented shorthand for a
    /// zero-sized array); any other length must equal `dim*dim` exactly.
    pub direction: Vec<f64>,
}

impl Default for SourceGeometry {
    /// SimpleITK's yaml defaults, sized for a 3-D image: `Origin = [0,0,0]`,
    /// `Spacing = [1,1,1]`, `Direction = []` (identity). A caller targeting a
    /// different dimension must supply dimension-matched `origin`/`spacing`.
    fn default() -> Self {
        Self {
            origin: vec![0.0; 3],
            spacing: vec![1.0; 3],
            direction: Vec::new(),
        }
    }
}

/// Build a geometry-only scratch image: a zero-filled `Float64` buffer whose
/// sole purpose is validating `geometry` against `size`'s dimension (reusing
/// `Image::set_spacing`/`set_origin`/`set_direction`'s own length checks) and
/// then supplying `continuous_index_to_physical_point` to the generators
/// below. The pixel type and buffer contents are discarded; only geometry
/// and dimension are used.
fn geometry_image(size: &[usize], geometry: &SourceGeometry) -> Result<Image> {
    let dim = size.len();
    let mut geo = Image::new(size, PixelId::Float64);
    geo.set_spacing(&geometry.spacing)?;
    geo.set_origin(&geometry.origin)?;
    let direction = if geometry.direction.is_empty() {
        matrix::identity(dim)
    } else {
        geometry.direction.clone()
    };
    geo.set_direction(&direction)?;
    Ok(geo)
}

fn require_dim(len: usize, dim: usize) -> Result<()> {
    if len != dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: len,
        });
    }
    Ok(())
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

fn build_typed_image<T: Scalar>(size: &[usize], vals: &[f64]) -> Result<Image> {
    let out: Vec<T> = vals.iter().map(|&v| T::from_f64(v)).collect();
    Ok(Image::from_vec(size, out)?)
}

// ---- gaussian_source ---------------------------------------------------

/// `GaussianImageSource`'s own parameters (everything but geometry and
/// output pixel type).
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianSourceSettings {
    /// Standard deviation in each direction. One entry per dimension.
    pub sigma: Vec<f64>,
    /// Mean (peak location) in each direction. One entry per dimension.
    pub mean: Vec<f64>,
    /// Scale factor multiplying the true Gaussian value.
    pub scale: f64,
    /// Whether to normalize the Gaussian (sum over infinite space is 1.0).
    pub normalized: bool,
}

impl Default for GaussianSourceSettings {
    /// SimpleITK's yaml defaults, sized for a 3-D image: `Sigma = [16,16,16]`,
    /// `Mean = [32,32,32]`, `Scale = 255`, `Normalized = false`.
    fn default() -> Self {
        Self {
            sigma: vec![16.0; 3],
            mean: vec![32.0; 3],
            scale: 255.0,
            normalized: false,
        }
    }
}

/// `GaussianImageSource`: an n-dimensional Gaussian, evaluated via
/// `GaussianSpatialFunction` at every pixel's physical point
/// (`itkGaussianImageSource.hxx::GenerateData`).
///
/// `value(p) = scale * (1/prefix) * exp(-sum_d (p[d]-mean[d])^2 / (2*sigma[d]^2))`,
/// where `prefix = 1` unless `normalized`, in which case
/// `prefix = product_d (sigma[d] * sqrt(2*pi))`
/// (`itkGaussianSpatialFunction.hxx::Evaluate`). The result is narrowed to
/// the output pixel type with `static_cast`/truncating semantics
/// ([`Scalar::from_f64`]), matching the `.hxx`'s
/// `static_cast<PixelType>(value)`.
///
/// Errors if `sigma`/`mean`/`geometry.origin`/`geometry.spacing` don't have
/// one entry per dimension of `size`, or `geometry.direction` is non-empty
/// and not exactly `dim*dim`.
pub fn gaussian_source(
    pixel_id: PixelId,
    size: &[usize],
    settings: &GaussianSourceSettings,
    geometry: &SourceGeometry,
) -> Result<Image> {
    let dim = size.len();
    require_dim(settings.sigma.len(), dim)?;
    require_dim(settings.mean.len(), dim)?;
    let sigma = &settings.sigma;
    let mean = &settings.mean;

    let geo = geometry_image(size, geometry)?;

    let prefix_denom = if settings.normalized {
        let sqrt_two_pi = (2.0 * std::f64::consts::PI).sqrt();
        sigma.iter().fold(1.0, |acc, &s| acc * s * sqrt_two_pi)
    } else {
        1.0
    };

    let axis_strides = strides(size);
    let count: usize = size.iter().product();
    let mut vals = vec![0.0f64; count];
    for (o, slot) in vals.iter_mut().enumerate() {
        let idx_f: Vec<f64> = (0..dim)
            .map(|d| ((o / axis_strides[d]) % size[d]) as f64)
            .collect();
        let point = geo.continuous_index_to_physical_point(&idx_f);
        let suffix_exp: f64 = (0..dim)
            .map(|d| {
                let diff = point[d] - mean[d];
                diff * diff / (2.0 * sigma[d] * sigma[d])
            })
            .sum();
        *slot = settings.scale * (1.0 / prefix_denom) * (-suffix_exp).exp();
    }

    let mut out = dispatch_scalar!(pixel_id, build_typed_image, size, &vals)?;
    out.copy_geometry_from(&geo);
    Ok(out)
}

// ---- gabor_source -------------------------------------------------------

/// `GaborImageSource`'s own parameters (everything but geometry and output
/// pixel type). `PhaseOffset` is deliberately absent: SimpleITK's yaml never
/// exposes it (`GaborImageSource.yaml` has no `PhaseOffset` member), so the
/// underlying `GaborKernelFunction`'s phase offset stays at its own default
/// of `0.0` in every call this port can make.
#[derive(Clone, Debug, PartialEq)]
pub struct GaborSourceSettings {
    /// Standard deviation in each direction; `sigma[0]` is also the Gabor
    /// kernel's own envelope width along axis 0. One entry per dimension.
    pub sigma: Vec<f64>,
    /// Mean (center) in each direction. One entry per dimension.
    pub mean: Vec<f64>,
    /// Modulation frequency of the sine/cosine component along axis 0.
    pub frequency: f64,
    /// `false` evaluates the cosine (real/symmetric) part, `true` the sine
    /// (imaginary/antisymmetric) part.
    pub calculate_imaginary_part: bool,
}

impl Default for GaborSourceSettings {
    /// SimpleITK's yaml defaults, sized for a 3-D image: `Sigma = [16,16,16]`,
    /// `Mean = [32,32,32]`, `Frequency = 0.4`,
    /// `CalculateImaginaryPart = false`.
    fn default() -> Self {
        Self {
            sigma: vec![16.0; 3],
            mean: vec![32.0; 3],
            frequency: 0.4,
            calculate_imaginary_part: false,
        }
    }
}

/// `GaborImageSource`: a Gabor filter oriented along axis 0 — a
/// `GaborKernelFunction` sinusoid-in-Gaussian-envelope along axis 0,
/// multiplied by a non-normalized 1-D Gaussian envelope along every other
/// axis (`itkGaborImageSource.hxx::GenerateData`):
///
/// `value(p) = exp(-0.5 * sum_{d=1..} ((p[d]-mean[d])/sigma[d])^2) * kernel(p[0]-mean[0])`
///
/// where, with `u = p[0]-mean[0]`
/// (`itkGaborKernelFunction.h::Evaluate`, `PhaseOffset` fixed at `0`):
///
/// `kernel(u) = exp(-0.5*(u/sigma[0])^2) * cos_or_sin(2*pi*frequency*u)`
///
/// (`cos` unless `calculate_imaginary_part`, then `sin`). The result is
/// narrowed to the output pixel type with truncating semantics
/// ([`Scalar::from_f64`]), matching the `.hxx`'s `static_cast<PixelType>`.
///
/// Errors under the same conditions as [`gaussian_source`].
pub fn gabor_source(
    pixel_id: PixelId,
    size: &[usize],
    settings: &GaborSourceSettings,
    geometry: &SourceGeometry,
) -> Result<Image> {
    let dim = size.len();
    require_dim(settings.sigma.len(), dim)?;
    require_dim(settings.mean.len(), dim)?;
    let sigma = &settings.sigma;
    let mean = &settings.mean;

    let geo = geometry_image(size, geometry)?;

    let axis_strides = strides(size);
    let count: usize = size.iter().product();
    let mut vals = vec![0.0f64; count];
    for (o, slot) in vals.iter_mut().enumerate() {
        let idx_f: Vec<f64> = (0..dim)
            .map(|d| ((o / axis_strides[d]) % size[d]) as f64)
            .collect();
        let point = geo.continuous_index_to_physical_point(&idx_f);

        let envelope_rest: f64 = (1..dim)
            .map(|d| {
                let z = (point[d] - mean[d]) / sigma[d];
                z * z
            })
            .sum();
        let gaussian_rest = (-0.5 * envelope_rest).exp();

        let u = point[0] - mean[0];
        let envelope0 = (-0.5 * (u / sigma[0]).powi(2)).exp();
        let phase = 2.0 * std::f64::consts::PI * settings.frequency * u;
        let trig = if settings.calculate_imaginary_part {
            phase.sin()
        } else {
            phase.cos()
        };

        *slot = gaussian_rest * envelope0 * trig;
    }

    let mut out = dispatch_scalar!(pixel_id, build_typed_image, size, &vals)?;
    out.copy_geometry_from(&geo);
    Ok(out)
}

// ---- grid_source --------------------------------------------------------

/// `GridImageSource`'s own parameters (everything but geometry and output
/// pixel type). The kernel is fixed to `GaussianKernelFunction<double>`
/// (`itkGridImageSource.hxx`'s constructor — not the B-spline kernel — see
/// [`grid_source`]'s doc for why the choice is numerically irrelevant here).
#[derive(Clone, Debug, PartialEq)]
pub struct GridSourceSettings {
    /// Standard deviation of the per-axis Gaussian pulses. One entry per
    /// dimension.
    pub sigma: Vec<f64>,
    /// Spacing between grid lines along each axis. One entry per dimension.
    pub grid_spacing: Vec<f64>,
    /// Offset of the first grid line along each axis (clamped to at most
    /// `grid_spacing[d]`, matching `BeforeThreadedGenerateData`'s
    /// `std::min(GridOffset, GridSpacing)`). One entry per dimension.
    pub grid_offset: Vec<f64>,
    /// Which axes are gridded; an axis with `false` is left constant (a
    /// factor of `1`) rather than pulsed. One entry per dimension.
    pub which_dimensions: Vec<bool>,
    /// Scale factor multiplying the grid value.
    pub scale: f64,
}

impl Default for GridSourceSettings {
    /// SimpleITK's yaml defaults, sized for a 3-D image:
    /// `Sigma = [0.5,0.5,0.5]`, `GridSpacing = [4,4,4]`,
    /// `GridOffset = [0,0,0]`, `WhichDimensions = [true,true,true]`,
    /// `Scale = 255`.
    fn default() -> Self {
        Self {
            sigma: vec![0.5; 3],
            grid_spacing: vec![4.0; 3],
            grid_offset: vec![0.0; 3],
            which_dimensions: vec![true; 3],
            scale: 255.0,
        }
    }
}

/// `GaussianKernelFunction<double>::Evaluate` (`itkGaussianKernelFunction.h`):
/// the standard normal density, `exp(-0.5*x^2) / sqrt(2*pi)`.
fn gaussian_kernel(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// `GridImageSource`: a separable grid of pulses, one per gridded axis,
/// multiplied together (`itkGridImageSource.hxx`).
///
/// For each gridded axis `i` (`which_dimensions[i]`), `BeforeThreadedGenerateData`
/// builds a 1-D response by summing shifted Gaussian-kernel pulses spaced
/// `grid_spacing[i]` apart and inverting it so grid lines read as troughs
/// (`pixels = 1 - pixels/pixels.max()`); a non-gridded axis is left at a
/// constant `1`. The final pixel is `scale` times the product of each axis's
/// 1-D response at that pixel's own index along the axis.
///
/// Two upstream quirks are reproduced exactly here, both consequences of
/// `BeforeThreadedGenerateData` computing each axis's response along a
/// single `ImageLinearIteratorWithIndex` line with every *other* index held
/// at the region's start index (always `0` in this crate's `Image` model):
///
/// - **`Origin` does not affect the grid pattern.** The `.hxx` computes
///   `point[i] - origin[i] - ...`, and `point[i]` (from
///   `TransformIndexToPhysicalPoint` with every other index at `0`) is itself
///   `origin[i] + direction[i][i]*spacing[i]*index[i]` — so `origin[i]`
///   cancels algebraically. This port computes the already-cancelled form
///   (`direction[i][i]*spacing[i]*index[i] - ...`) rather than transcribing
///   the literal subtraction, avoiding needless floating-point cancellation
///   error for a large `origin`; the two are mathematically identical.
///   `origin` still becomes the output image's own origin metadata, it just
///   never influences *where the grid lines fall*.
/// - **Only `direction`'s diagonal feeds the pattern**, not its full row:
///   with every other index at `0`, `TransformIndexToPhysicalPoint`'s
///   `sum_k direction[i][k]*spacing[k]*index[k]` reduces to its `k == i`
///   term. An off-diagonal direction entry is faithfully inert here — again
///   because `BeforeThreadedGenerateData` itself only ever walks that one
///   line, not because this port dropped a term it should have used.
///
/// The kernel is always `GaussianKernelFunction`, matching the constructor's
/// hardcoded choice (`itkGridImageSource.hxx`; the class documents itself as
/// pluggable via `SetKernelFunction`, but SimpleITK's yaml never exposes that
/// setter). Its `1/sqrt(2*pi)` normalization constant cancels out of
/// `pixels/pixels.max()` regardless, so it is carried here for fidelity to
/// the ported formula rather than because it changes the result.
///
/// Errors if `sigma`/`grid_spacing`/`grid_offset`/`which_dimensions`/
/// `geometry.origin`/`geometry.spacing` don't have one entry per dimension
/// of `size`, or `geometry.direction` is non-empty and not exactly `dim*dim`.
pub fn grid_source(
    pixel_id: PixelId,
    size: &[usize],
    settings: &GridSourceSettings,
    geometry: &SourceGeometry,
) -> Result<Image> {
    let dim = size.len();
    require_dim(settings.sigma.len(), dim)?;
    require_dim(settings.grid_spacing.len(), dim)?;
    require_dim(settings.grid_offset.len(), dim)?;
    require_dim(settings.which_dimensions.len(), dim)?;

    let geo = geometry_image(size, geometry)?;
    let spacing = geo.spacing().to_vec();
    let direction = geo.direction().to_vec();

    let mut axis_pixels: Vec<Vec<f64>> = Vec::with_capacity(dim);
    for i in 0..dim {
        if !settings.which_dimensions[i] {
            axis_pixels.push(vec![1.0; size[i]]);
            continue;
        }
        let offset = settings.grid_offset[i].min(settings.grid_spacing[i]);
        let number_of_gaussians =
            (size[i] as f64 * spacing[i] / settings.grid_spacing[i]).ceil() as u32 + 4;
        let direction_diag = direction[i * dim + i];

        let mut pixels = vec![0.0f64; size[i]];
        for (idx, slot) in pixels.iter_mut().enumerate() {
            let point_i = direction_diag * spacing[i] * idx as f64;
            let mut val = 0.0;
            for j in 0..number_of_gaussians {
                let num = point_i - (j as f64 - 2.0) * settings.grid_spacing[i] - offset;
                val += gaussian_kernel(num / settings.sigma[i]);
            }
            *slot = val;
        }
        let max_val = pixels.iter().cloned().fold(f64::MIN, f64::max);
        for p in pixels.iter_mut() {
            *p = 1.0 - *p / max_val;
        }
        axis_pixels.push(pixels);
    }

    let axis_strides = strides(size);
    let count: usize = size.iter().product();
    let mut vals = vec![0.0f64; count];
    for (o, slot) in vals.iter_mut().enumerate() {
        let mut val = 1.0;
        for (i, axis) in axis_pixels.iter().enumerate() {
            let ii = (o / axis_strides[i]) % size[i];
            val *= axis[ii];
        }
        *slot = settings.scale * val;
    }

    let mut out = dispatch_scalar!(pixel_id, build_typed_image, size, &vals)?;
    out.copy_geometry_from(&geo);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_geometry(dim: usize) -> SourceGeometry {
        SourceGeometry {
            origin: vec![0.0; dim],
            spacing: vec![1.0; dim],
            direction: matrix::identity(dim),
        }
    }

    // ---- gaussian_source ----

    #[test]
    fn gaussian_hand_derived_values_1d() {
        let settings = GaussianSourceSettings {
            sigma: vec![2.0],
            mean: vec![2.0],
            scale: 1.0,
            normalized: false,
        };
        let img =
            gaussian_source(PixelId::Float64, &[5], &settings, &identity_geometry(1)).unwrap();
        let v = img.scalar_slice::<f64>().unwrap();
        // peak exactly at index 2 (physical point == mean): exp(0) == 1.
        assert!((v[2] - 1.0).abs() < 1e-12);
        // index 0 and 4 are both |diff|==2 from the mean: exp(-4/(2*4)) = exp(-0.5).
        let expected_2away = (-0.5f64).exp();
        assert!((v[0] - expected_2away).abs() < 1e-12);
        assert!((v[4] - expected_2away).abs() < 1e-12);
        // index 1 and 3 are both |diff|==1: exp(-1/8).
        let expected_1away = (-0.125f64).exp();
        assert!((v[1] - expected_1away).abs() < 1e-12);
        assert!((v[3] - expected_1away).abs() < 1e-12);
    }

    #[test]
    fn gaussian_normalized_peak_matches_1_over_sigma_sqrt_2pi() {
        let sigma = 2.0;
        let settings = GaussianSourceSettings {
            sigma: vec![sigma],
            mean: vec![2.0],
            scale: 1.0,
            normalized: true,
        };
        let img =
            gaussian_source(PixelId::Float64, &[5], &settings, &identity_geometry(1)).unwrap();
        let v = img.scalar_slice::<f64>().unwrap();
        let expected_peak = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        assert!((v[2] - expected_peak).abs() < 1e-12);
    }

    #[test]
    fn gaussian_2d_peak_lands_at_mean_under_rotated_direction() {
        // A non-identity direction and non-trivial origin/spacing: the peak
        // must occur at whatever index maps (through the full affine
        // transform) to `mean`, proving the formula actually consults
        // `continuous_index_to_physical_point` rather than raw index.
        let theta = std::f64::consts::FRAC_PI_6;
        let geometry = SourceGeometry {
            origin: vec![1.0, -2.0],
            spacing: vec![1.5, 0.5],
            direction: vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
        };
        let geo = geometry_image(&[6, 6], &geometry).unwrap();
        let peak_index = [3.0, 2.0];
        let mean = geo.continuous_index_to_physical_point(&peak_index);

        let settings = GaussianSourceSettings {
            sigma: vec![1.0, 1.0],
            mean,
            scale: 1.0,
            normalized: false,
        };
        let img = gaussian_source(PixelId::Float64, &[6, 6], &settings, &geometry).unwrap();
        let v = img.scalar_slice::<f64>().unwrap();
        let peak_offset = img.linear_index(&[3, 2]);
        assert!((v[peak_offset] - 1.0).abs() < 1e-12);
        // Every other pixel must be strictly less than the peak (isotropic
        // sigma + orthonormal rotation preserves distance ordering).
        for (o, &val) in v.iter().enumerate() {
            if o != peak_offset {
                assert!(val < 1.0, "pixel {o} = {val} should be < peak");
            }
        }
    }

    #[test]
    fn gaussian_integer_pixel_type_truncates_not_rounds() {
        // static_cast<uint8>(254.9) truncates to 254, it does not round to 255.
        let settings = GaussianSourceSettings {
            sigma: vec![1.0],
            mean: vec![0.0],
            scale: 254.9,
            normalized: false,
        };
        let img = gaussian_source(PixelId::UInt8, &[1], &settings, &identity_geometry(1)).unwrap();
        assert_eq!(img.scalar_slice::<u8>().unwrap(), &[254]);
    }

    #[test]
    fn gaussian_propagates_geometry() {
        let geometry = SourceGeometry {
            origin: vec![3.0, -1.0],
            spacing: vec![2.0, 0.5],
            direction: vec![0.0, 1.0, -1.0, 0.0],
        };
        let img = gaussian_source(
            PixelId::Float32,
            &[4, 4],
            &GaussianSourceSettings {
                sigma: vec![16.0, 16.0],
                mean: vec![32.0, 32.0],
                scale: 255.0,
                normalized: false,
            },
            &geometry,
        )
        .unwrap();
        assert_eq!(img.origin(), geometry.origin.as_slice());
        assert_eq!(img.spacing(), geometry.spacing.as_slice());
        assert_eq!(img.direction(), geometry.direction.as_slice());
    }

    #[test]
    fn gaussian_dimension_mismatch_errors() {
        let settings = GaussianSourceSettings {
            sigma: vec![1.0, 1.0, 1.0],
            mean: vec![0.0, 0.0],
            scale: 1.0,
            normalized: false,
        };
        assert_eq!(
            gaussian_source(PixelId::Float64, &[2, 2], &settings, &identity_geometry(2)),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 3
            })
        );
    }

    #[test]
    fn gaussian_bad_direction_length_errors() {
        let settings = GaussianSourceSettings {
            sigma: vec![1.0, 1.0],
            mean: vec![0.0, 0.0],
            scale: 1.0,
            normalized: false,
        };
        let geometry = SourceGeometry {
            origin: vec![0.0, 0.0],
            spacing: vec![1.0, 1.0],
            direction: vec![1.0, 0.0, 0.0], // 3 entries, needs 4 for 2-D.
        };
        assert!(matches!(
            gaussian_source(PixelId::Float64, &[2, 2], &settings, &geometry),
            Err(FilterError::Core(_))
        ));
    }

    // ---- gabor_source ----

    #[test]
    fn gabor_real_and_imaginary_parts_at_dc() {
        let base = GaborSourceSettings {
            sigma: vec![2.0],
            mean: vec![4.0],
            frequency: 0.2,
            calculate_imaginary_part: false,
        };
        let real = gabor_source(PixelId::Float64, &[9], &base, &identity_geometry(1)).unwrap();
        // u == 0 at the mean: envelope(1) * cos(0) == 1.
        assert!((real.scalar_slice::<f64>().unwrap()[4] - 1.0).abs() < 1e-12);

        let imaginary = GaborSourceSettings {
            calculate_imaginary_part: true,
            ..base
        };
        let img = gabor_source(PixelId::Float64, &[9], &imaginary, &identity_geometry(1)).unwrap();
        // u == 0 at the mean: envelope(1) * sin(0) == 0.
        assert!(img.scalar_slice::<f64>().unwrap()[4].abs() < 1e-12);
    }

    #[test]
    fn gabor_real_part_is_even_and_imaginary_part_is_odd_about_mean() {
        let real_settings = GaborSourceSettings {
            sigma: vec![5.0],
            mean: vec![4.0],
            frequency: 0.2,
            calculate_imaginary_part: false,
        };
        let real = gabor_source(
            PixelId::Float64,
            &[9],
            &real_settings,
            &identity_geometry(1),
        )
        .unwrap();
        let rv = real.scalar_slice::<f64>().unwrap();
        for d in 1..=4usize {
            assert!(
                (rv[4 - d] - rv[4 + d]).abs() < 1e-12,
                "real part not even at offset {d}"
            );
        }

        let imaginary_settings = GaborSourceSettings {
            calculate_imaginary_part: true,
            ..real_settings
        };
        let imaginary = gabor_source(
            PixelId::Float64,
            &[9],
            &imaginary_settings,
            &identity_geometry(1),
        )
        .unwrap();
        let iv = imaginary.scalar_slice::<f64>().unwrap();
        for d in 1..=4usize {
            assert!(
                (iv[4 - d] + iv[4 + d]).abs() < 1e-12,
                "imaginary part not odd at offset {d}"
            );
        }
    }

    #[test]
    fn gabor_frequency_pins_zero_crossing_spacing() {
        // frequency = 0.25 => period 4, so cos(2*pi*0.25*u) == cos(pi/2 * u)
        // crosses zero at odd u and alternates sign at even u. sigma is huge
        // so the envelope is ~constant over this small domain and does not
        // mask the sign pattern.
        let settings = GaborSourceSettings {
            sigma: vec![100.0],
            mean: vec![4.0],
            frequency: 0.25,
            calculate_imaginary_part: false,
        };
        let img = gabor_source(PixelId::Float64, &[9], &settings, &identity_geometry(1)).unwrap();
        let v = img.scalar_slice::<f64>().unwrap();
        for &idx in &[1usize, 3, 5, 7] {
            assert!(
                v[idx].abs() < 1e-9,
                "expected zero crossing at {idx}: {}",
                v[idx]
            );
        }
        let signs: Vec<f64> = [0usize, 2, 4, 6, 8]
            .iter()
            .map(|&i| v[i].signum())
            .collect();
        assert_eq!(signs, vec![1.0, -1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn gabor_dimension_mismatch_errors() {
        let settings = GaborSourceSettings {
            sigma: vec![1.0],
            mean: vec![0.0, 0.0],
            frequency: 0.4,
            calculate_imaginary_part: false,
        };
        assert_eq!(
            gabor_source(PixelId::Float64, &[2, 2], &settings, &identity_geometry(2)),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        );
    }

    // ---- grid_source ----

    fn grid_1d_settings(grid_offset: f64) -> GridSourceSettings {
        GridSourceSettings {
            sigma: vec![0.5],
            grid_spacing: vec![4.0],
            grid_offset: vec![grid_offset],
            which_dimensions: vec![true],
            scale: 1.0,
        }
    }

    #[test]
    fn grid_hand_derived_troughs_and_off_peak_1d() {
        let settings = grid_1d_settings(0.0);
        let img = grid_source(PixelId::Float64, &[8], &settings, &identity_geometry(1)).unwrap();
        let v = img.scalar_slice::<f64>().unwrap();
        // Grid lines (troughs) at multiples of grid_spacing == 0, 4.
        assert!(v[0].abs() < 1e-9, "trough at 0: {}", v[0]);
        assert!(v[4].abs() < 1e-9, "trough at 4: {}", v[4]);
        // Hand-derived off-peak value at index 2 (midway between grid
        // lines): the two dominant kernel terms are at |num/sigma| == 4,
        // giving kernel sum ~= 2 * exp(-8)/sqrt(2*pi) against a peak kernel
        // value of 1/sqrt(2*pi), so 1 - sum/peak ~= 1 - 2*exp(-8) =~ 0.9993276.
        let expected_off_peak = 1.0 - 2.0 * (-8.0f64).exp();
        assert!(
            (v[2] - expected_off_peak).abs() < 1e-4,
            "off-peak value {} vs expected {expected_off_peak}",
            v[2]
        );
        // idx 6 is symmetric to idx 2 around the trough at idx 4.
        assert!((v[6] - v[2]).abs() < 1e-9);
    }

    #[test]
    fn grid_offset_shifts_trough_positions() {
        let settings = grid_1d_settings(2.0);
        let img = grid_source(PixelId::Float64, &[8], &settings, &identity_geometry(1)).unwrap();
        let v = img.scalar_slice::<f64>().unwrap();
        // Troughs move from {0, 4} to {2, 6}.
        assert!(v[2].abs() < 1e-9, "trough at 2: {}", v[2]);
        assert!(v[6].abs() < 1e-9, "trough at 6: {}", v[6]);
        // 0 and 4, no longer trough positions, land back on the same
        // off-peak value hand-derived above.
        let expected_off_peak = 1.0 - 2.0 * (-8.0f64).exp();
        assert!((v[0] - expected_off_peak).abs() < 1e-4);
        assert!((v[4] - expected_off_peak).abs() < 1e-4);
    }

    #[test]
    fn grid_which_dimensions_masks_axis_constant() {
        let settings = GridSourceSettings {
            sigma: vec![0.5, 0.5],
            grid_spacing: vec![4.0, 4.0],
            grid_offset: vec![0.0, 0.0],
            which_dimensions: vec![true, false],
            scale: 1.0,
        };
        let img = grid_source(PixelId::Float64, &[6, 8], &settings, &identity_geometry(2)).unwrap();
        let v = img.scalar_slice::<f64>().unwrap();
        // axis 1 is masked off: fixing x, every y must give the same value.
        for x in 0..6 {
            let base = v[img.linear_index(&[x, 0])];
            for y in 1..8 {
                let val = v[img.linear_index(&[x, y])];
                assert!(
                    (val - base).abs() < 1e-12,
                    "axis 1 should be constant at x={x}: y=0 -> {base}, y={y} -> {val}"
                );
            }
        }
        // axis 0 is still gridded: it must actually vary.
        let x0 = v[img.linear_index(&[0, 0])];
        let x2 = v[img.linear_index(&[2, 0])];
        assert!((x0 - x2).abs() > 1e-6);
    }

    #[test]
    fn grid_scale_factor_is_linear() {
        let base = GridSourceSettings {
            scale: 1.0,
            ..grid_1d_settings(0.0)
        };
        let scaled = GridSourceSettings {
            scale: 7.5,
            ..grid_1d_settings(0.0)
        };
        let a = grid_source(PixelId::Float64, &[8], &base, &identity_geometry(1)).unwrap();
        let b = grid_source(PixelId::Float64, &[8], &scaled, &identity_geometry(1)).unwrap();
        let av = a.scalar_slice::<f64>().unwrap();
        let bv = b.scalar_slice::<f64>().unwrap();
        for (x, y) in av.iter().zip(bv.iter()) {
            assert!((y - 7.5 * x).abs() < 1e-9);
        }
    }

    #[test]
    fn grid_origin_does_not_affect_pattern_only_metadata() {
        let settings = grid_1d_settings(0.0);
        let near_origin = SourceGeometry {
            origin: vec![0.0],
            spacing: vec![1.0],
            direction: vec![1.0],
        };
        let far_origin = SourceGeometry {
            origin: vec![500.0],
            spacing: vec![1.0],
            direction: vec![1.0],
        };
        let a = grid_source(PixelId::Float64, &[8], &settings, &near_origin).unwrap();
        let b = grid_source(PixelId::Float64, &[8], &settings, &far_origin).unwrap();
        // Bit-for-bit identical pixel pattern...
        assert_eq!(
            a.scalar_slice::<f64>().unwrap(),
            b.scalar_slice::<f64>().unwrap()
        );
        // ...even though the requested origin still lands in the metadata.
        assert_eq!(a.origin(), &[0.0]);
        assert_eq!(b.origin(), &[500.0]);
    }

    #[test]
    fn grid_direction_only_diagonal_feeds_the_pattern() {
        let settings = GridSourceSettings {
            sigma: vec![0.5, 0.5],
            grid_spacing: vec![4.0, 4.0],
            grid_offset: vec![0.0, 0.0],
            which_dimensions: vec![true, true],
            scale: 1.0,
        };
        let identity = identity_geometry(2);
        // Off-diagonal-only shear: diagonal entries unchanged (1, 1), only
        // the off-diagonal term differs from identity.
        let sheared = SourceGeometry {
            origin: vec![0.0, 0.0],
            spacing: vec![1.0, 1.0],
            direction: vec![1.0, 0.5, 0.0, 1.0],
        };
        let a = grid_source(PixelId::Float64, &[8, 8], &settings, &identity).unwrap();
        let b = grid_source(PixelId::Float64, &[8, 8], &settings, &sheared).unwrap();
        assert_eq!(
            a.scalar_slice::<f64>().unwrap(),
            b.scalar_slice::<f64>().unwrap()
        );

        // Changing the diagonal itself, by contrast, does change the
        // pattern: doubling axis 0's diagonal entry doubles its effective
        // position, so index 2 (previously off-peak) becomes a trough.
        let doubled_diag = SourceGeometry {
            origin: vec![0.0, 0.0],
            spacing: vec![1.0, 1.0],
            direction: vec![2.0, 0.0, 0.0, 1.0],
        };
        let c = grid_source(PixelId::Float64, &[8, 8], &settings, &doubled_diag).unwrap();
        let cv = c.scalar_slice::<f64>().unwrap();
        assert!(cv[c.linear_index(&[2, 0])].abs() < 1e-9);
    }

    #[test]
    fn grid_propagates_geometry() {
        let geometry = SourceGeometry {
            origin: vec![3.0, -1.0],
            spacing: vec![2.0, 0.5],
            direction: vec![0.0, 1.0, -1.0, 0.0],
        };
        let settings = GridSourceSettings {
            sigma: vec![0.5, 0.5],
            grid_spacing: vec![4.0, 4.0],
            grid_offset: vec![0.0, 0.0],
            which_dimensions: vec![true, true],
            scale: 255.0,
        };
        let img = grid_source(PixelId::Float32, &[6, 6], &settings, &geometry).unwrap();
        assert_eq!(img.origin(), geometry.origin.as_slice());
        assert_eq!(img.spacing(), geometry.spacing.as_slice());
        assert_eq!(img.direction(), geometry.direction.as_slice());
    }

    #[test]
    fn grid_dimension_mismatch_errors() {
        let settings = GridSourceSettings {
            sigma: vec![0.5],
            grid_spacing: vec![4.0, 4.0],
            grid_offset: vec![0.0, 0.0],
            which_dimensions: vec![true, true],
            scale: 255.0,
        };
        assert_eq!(
            grid_source(PixelId::Float64, &[4, 4], &settings, &identity_geometry(2)),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        );
    }
}
