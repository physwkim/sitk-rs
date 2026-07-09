//! Image-source generators: filters with no input image. Ported so far:
//! `GaussianImageSource` (`itkGaussianImageSource.h`/`.hxx`,
//! `Modules/Filtering/ImageSources/include`; [`gaussian_source`]).
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
}
