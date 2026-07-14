//! The level-set PDE terms, ported from `itkLevelSetFunction.h/.hxx`
//! (`ComputeUpdate`, `ComputeMeanCurvature`, `ComputeGlobalTimeStep`) with the
//! speed/advection sampling of `itkSegmentationLevelSetFunction.h/.hxx`
//! (`PropagationSpeed`, `AdvectionField`).
//!
//! ITK solves
//!
//! ```text
//! phi_t + alpha A(x)·grad(phi) + beta P(x)|grad(phi)| = gamma Z(x) kappa |grad(phi)|
//! ```
//!
//! and `ComputeUpdate` returns `curvature - propagation - advection` (the
//! Laplacian-smoothing term is dropped: none of the concrete functions ported
//! here ever gives it a non-zero weight). Inside the surface is negative,
//! outside is positive.
//!
//! `Z(x)` is `CurvatureSpeed`. `LevelSetFunction::CurvatureSpeed` returns `1`
//! (itkLevelSetFunction.h:216-220) and `SegmentationLevelSetFunction` does not
//! override it, so the threshold, Laplacian and Canny functions all curve at
//! unit speed. `GeodesicActiveContourLevelSetFunction` and
//! `ShapeDetectionLevelSetFunction` *do* override it to return
//! `PropagationSpeed` (itkGeodesicActiveContourLevelSetFunction.h:115-120,
//! itkShapeDetectionLevelSetFunction.h:104-109), so for those two the [`speed`]
//! buffer feeds both terms. [`CurvatureSpeed`] selects between the two.
//!
//! [`speed`]: LevelSetFunction::speed

use super::grid::Grid;
use crate::denoise::curvature_flow_update_at;

/// The `FiniteDifferenceFunction` a `SparseFieldLevelSetImageFilter` subclass
/// installs with `SetDifferenceFunction`.
///
/// The five segmentation filters install a `SegmentationLevelSetFunction`
/// ([`LevelSetFunction`]); `AntiAliasBinaryImageFilter` installs a
/// `CurvatureFlowFunction` ([`CurvatureFlowFunction`]) instead.
pub(super) enum DifferenceFunction {
    LevelSet(LevelSetFunction),
    CurvatureFlow(CurvatureFlowFunction),
}

impl DifferenceFunction {
    /// `FiniteDifferenceFunction::ComputeNeighborhoodScales()`, read by
    /// `SparseFieldLevelSetImageFilter::InitializeActiveLayerValues`.
    pub(super) fn neighborhood_scales(&self) -> &[f64] {
        match self {
            DifferenceFunction::LevelSet(f) => &f.neighborhood_scales,
            DifferenceFunction::CurvatureFlow(f) => &f.neighborhood_scales,
        }
    }

    pub(super) fn compute_update(
        &self,
        phi: &[f64],
        grid: &Grid,
        index: usize,
        gd: &mut GlobalData,
    ) -> f64 {
        match self {
            DifferenceFunction::LevelSet(f) => f.compute_update(phi, grid, index, gd),
            DifferenceFunction::CurvatureFlow(f) => f.compute_update(phi, grid, index),
        }
    }

    pub(super) fn compute_global_time_step(&self, gd: &mut GlobalData) -> f64 {
        match self {
            DifferenceFunction::LevelSet(f) => f.compute_global_time_step(gd),
            DifferenceFunction::CurvatureFlow(f) => f.time_step,
        }
    }
}

/// `CurvatureFlowFunction` (itkCurvatureFlowFunction.h/.hxx): pure `kappa *
/// |grad(phi)|` at a constant time step. `ComputeGlobalTimeStep` returns
/// `m_TimeStep` regardless of the global data, and `ComputeUpdate` ignores it
/// entirely.
pub(super) struct CurvatureFlowFunction {
    /// As [`LevelSetFunction::neighborhood_scales`]. The function's own radius
    /// is `1` on every axis, so this is `ScaleCoefficients` verbatim.
    pub(super) neighborhood_scales: Vec<f64>,
    /// `m_TimeStep`, whose constructor default is `0.05`. No
    /// `SparseFieldLevelSetImageFilter` subclass in this crate changes it.
    pub(super) time_step: f64,
}

/// `CurvatureFlowFunction`'s constructor default for `m_TimeStep`
/// (itkCurvatureFlowFunction.hxx:38): `m_TimeStep = 0.05f`, a `float` literal
/// widened into a `double` member (`FiniteDifferenceFunction::TimeStepType`).
/// The widening is not exact, so the step is `0.05000000074505806`, not `0.05`.
pub(super) const CURVATURE_FLOW_TIME_STEP: f64 = 0.05f32 as f64;

impl CurvatureFlowFunction {
    pub(super) fn new(neighborhood_scales: Vec<f64>) -> Self {
        CurvatureFlowFunction {
            neighborhood_scales,
            time_step: CURVATURE_FLOW_TIME_STEP,
        }
    }

    /// `CurvatureFlowFunction::ComputeUpdate` (hxx:50-127), shared with
    /// [`crate::denoise::curvature_flow`] through
    /// [`curvature_flow_update_at`]. Neighbor reads clamp at the image border,
    /// matching the `ZeroFluxNeumannBoundaryCondition` of ITK's
    /// `NeighborhoodIterator`.
    fn compute_update(&self, phi: &[f64], grid: &Grid, index: usize) -> f64 {
        let dim = grid.dim();
        let center = grid.coord(index);
        let mut coord = vec![0i64; dim];
        curvature_flow_update_at(dim, &self.neighborhood_scales, |offset| {
            for d in 0..dim {
                coord[d] = center[d] + offset[d];
            }
            phi[grid.clamped_index(&coord)]
        })
    }
}

/// Which `CurvatureSpeed` override the concrete `SegmentationLevelSetFunction`
/// subclass uses for the `Z(x)` factor of the curvature term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CurvatureSpeed {
    /// `LevelSetFunction::CurvatureSpeed`, the un-overridden base: constant `1`.
    Unit,
    /// Overridden to return `PropagationSpeed`, i.e. the speed image.
    Propagation,
}

/// `LevelSetFunction::GlobalDataStruct`: the derivatives cached by
/// `ComputeUpdate` for the term helpers, plus the per-iteration maxima that
/// `ComputeGlobalTimeStep` turns into a stable time step.
pub(super) struct GlobalData {
    pub(super) max_advection_change: f64,
    pub(super) max_propagation_change: f64,
    pub(super) max_curvature_change: f64,
    /// Central first derivatives, one per axis.
    pub(super) dx: Vec<f64>,
    /// One-sided first derivatives, one per axis.
    pub(super) dx_forward: Vec<f64>,
    pub(super) dx_backward: Vec<f64>,
    /// Hessian, row-major `dim x dim`.
    pub(super) dxy: Vec<f64>,
    /// `1.0e-6` plus the squared central-difference gradient magnitude; the
    /// floor keeps the curvature term finite where the gradient vanishes.
    pub(super) grad_mag_sqr: f64,
}

impl GlobalData {
    pub(super) fn new(dim: usize) -> Self {
        GlobalData {
            max_advection_change: 0.0,
            max_propagation_change: 0.0,
            max_curvature_change: 0.0,
            dx: vec![0.0; dim],
            dx_forward: vec![0.0; dim],
            dx_backward: vec![0.0; dim],
            dxy: vec![0.0; dim * dim],
            grad_mag_sqr: 0.0,
        }
    }

    fn dxy(&self, i: usize, j: usize, dim: usize) -> f64 {
        self.dxy[i * dim + j]
    }
}

/// The segmentation level-set function: the three term weights plus the
/// pre-generated speed and advection images they sample.
pub(super) struct LevelSetFunction {
    /// `SegmentationLevelSetFunction::m_SpeedImage`, sampled by
    /// `PropagationSpeed` — and by `CurvatureSpeed` too when
    /// [`curvature_speed`](Self::curvature_speed) is
    /// [`CurvatureSpeed::Propagation`].
    pub(super) speed: Vec<f64>,
    /// `SegmentationLevelSetFunction::m_AdvectionImage`, split into one `f64`
    /// buffer per axis so no vector pixel type is needed. Empty when the
    /// advection weight is zero (ITK never allocates the image then either —
    /// `SegmentationLevelSetImageFilter::GenerateData`).
    pub(super) advection: Vec<Vec<f64>>,
    /// `alpha`, `beta`, `gamma`.
    pub(super) advection_weight: f64,
    pub(super) propagation_weight: f64,
    pub(super) curvature_weight: f64,
    /// The `Z(x)` factor of the curvature term.
    pub(super) curvature_speed: CurvatureSpeed,
    /// `FiniteDifferenceFunction::ComputeNeighborhoodScales()`:
    /// `ScaleCoefficients[d] / Radius[d]`, and with `UseImageSpacing` on (the
    /// ITK default) `ScaleCoefficients[d] == 1 / spacing[d]` and the radius is
    /// `1` on every axis.
    pub(super) neighborhood_scales: Vec<f64>,
    /// `max_d ScaleCoefficients[d]`, the divisor of `ComputeGlobalTimeStep`.
    pub(super) max_scale_coefficient: f64,
    /// `LevelSetFunction::m_WaveDT` and `m_DT`, both `1 / (2 * dim)`.
    pub(super) wave_dt: f64,
    pub(super) dt: f64,
}

impl LevelSetFunction {
    pub(super) fn new(
        speed: Vec<f64>,
        advection: Vec<Vec<f64>>,
        advection_weight: f64,
        propagation_weight: f64,
        curvature_weight: f64,
        curvature_speed: CurvatureSpeed,
        spacing: &[f64],
    ) -> Self {
        let dim = spacing.len();
        let scale_coefficients: Vec<f64> = spacing.iter().map(|&s| 1.0 / s).collect();
        let max_scale_coefficient = scale_coefficients.iter().copied().fold(0.0, f64::max);
        LevelSetFunction {
            speed,
            advection,
            advection_weight,
            propagation_weight,
            curvature_weight,
            curvature_speed,
            neighborhood_scales: scale_coefficients,
            max_scale_coefficient,
            wave_dt: 1.0 / (2.0 * dim as f64),
            dt: 1.0 / (2.0 * dim as f64),
        }
    }

    /// The first block of `ComputeUpdate` (itkLevelSetFunction.hxx:283-312):
    /// central, forward, backward and mixed second derivatives at `coord`,
    /// each scaled into physical units by `neighborhood_scales`.
    ///
    /// Neighbor reads clamp at the image border, matching the
    /// `ZeroFluxNeumannBoundaryCondition` of ITK's `NeighborhoodIterator`.
    pub(super) fn compute_derivatives(
        &self,
        phi: &[f64],
        grid: &Grid,
        coord: &mut [i64],
        gd: &mut GlobalData,
    ) {
        let dim = grid.dim();
        let scales = &self.neighborhood_scales;

        gd.grad_mag_sqr = 1.0e-6;
        let center = phi[grid.clamped_index(coord)];

        for i in 0..dim {
            coord[i] += 1;
            let forward = phi[grid.clamped_index(coord)];
            coord[i] -= 2;
            let backward = phi[grid.clamped_index(coord)];
            coord[i] += 1;

            gd.dx[i] = 0.5 * (forward - backward) * scales[i];
            gd.dxy[i * dim + i] = (forward + backward - 2.0 * center) * scales[i] * scales[i];
            gd.dx_forward[i] = (forward - center) * scales[i];
            gd.dx_backward[i] = (center - backward) * scales[i];
            gd.grad_mag_sqr += gd.dx[i] * gd.dx[i];

            for j in (i + 1)..dim {
                coord[i] -= 1;
                coord[j] -= 1;
                let mm = phi[grid.clamped_index(coord)];
                coord[j] += 2;
                let mp = phi[grid.clamped_index(coord)];
                coord[i] += 2;
                let pp = phi[grid.clamped_index(coord)];
                coord[j] -= 2;
                let pm = phi[grid.clamped_index(coord)];
                coord[i] -= 1;
                coord[j] += 1;

                let mixed = 0.25 * (mm - mp - pm + pp) * scales[i] * scales[j];
                gd.dxy[i * dim + j] = mixed;
                gd.dxy[j * dim + i] = mixed;
            }
        }
    }

    /// `LevelSetFunction::ComputeMeanCurvature` (itkLevelSetFunction.hxx:152-172).
    ///
    /// This is `kappa * |grad(phi)|`, not `kappa` alone — for a signed distance
    /// function (`|grad(phi)| == 1`) the two coincide, which is what the tests
    /// below exercise. `m_UseMinimalCurvature` is `false` by default and
    /// neither ported filter turns it on, so `ComputeCurvatureTerm` always
    /// lands here.
    pub(super) fn mean_curvature(gd: &GlobalData, dim: usize) -> f64 {
        let mut curvature_term = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                if j != i {
                    curvature_term -= gd.dx[i] * gd.dx[j] * gd.dxy(i, j, dim);
                    curvature_term += gd.dxy(j, j, dim) * gd.dx[i] * gd.dx[i];
                }
            }
        }
        curvature_term / gd.grad_mag_sqr
    }

    /// `LevelSetFunction::ComputeUpdate` (itkLevelSetFunction.hxx:275-409),
    /// evaluated at the active-layer pixel `index` of the level-set image
    /// `phi`.
    ///
    /// `FloatOffsetType offset` is always zero here: both
    /// `GeodesicActiveContourLevelSetImageFilter` and
    /// `ShapeDetectionLevelSetImageFilter` call `InterpolateSurfaceLocationOff()`
    /// in their constructors, so `SparseFieldLevelSetImageFilter::CalculateChange`
    /// takes its `else // Don't do interpolation` branch and `PropagationSpeed`
    /// / `AdvectionField` sample the speed and advection images at the pixel
    /// index itself.
    pub(super) fn compute_update(
        &self,
        phi: &[f64],
        grid: &Grid,
        index: usize,
        gd: &mut GlobalData,
    ) -> f64 {
        let dim = grid.dim();
        let mut coord = grid.coord(index);
        self.compute_derivatives(phi, grid, &mut coord, gd);

        // Curvature: gamma * Z(x) * kappa * |grad(phi)|.
        let mut curvature_term = 0.0;
        if self.curvature_weight != 0.0 {
            let z = match self.curvature_speed {
                CurvatureSpeed::Unit => 1.0,
                CurvatureSpeed::Propagation => self.speed[index],
            };
            curvature_term = Self::mean_curvature(gd, dim) * self.curvature_weight * z;
            gd.max_curvature_change = gd.max_curvature_change.max(curvature_term.abs());
        }

        // Advection: alpha * A(x)·grad(phi), upwinded per axis on the sign of
        // the (weighted) advective force.
        let mut advection_term = 0.0;
        if self.advection_weight != 0.0 {
            for i in 0..dim {
                let field = self.advection[i][index];
                let x_energy = self.advection_weight * field;
                if x_energy > 0.0 {
                    advection_term += field * gd.dx_backward[i];
                } else {
                    advection_term += field * gd.dx_forward[i];
                }
                gd.max_advection_change = gd.max_advection_change.max(x_energy.abs());
            }
            advection_term *= self.advection_weight;
        }

        // Propagation: beta * P(x) * |grad(phi)|, with the Godunov upwind
        // gradient magnitude chosen on the sign of the propagation term
        // (Sethian, Ch. 6).
        let mut propagation_term = 0.0;
        if self.propagation_weight != 0.0 {
            propagation_term = self.propagation_weight * self.speed[index];

            let mut propagation_gradient = 0.0;
            if propagation_term > 0.0 {
                for i in 0..dim {
                    propagation_gradient +=
                        gd.dx_backward[i].max(0.0).powi(2) + gd.dx_forward[i].min(0.0).powi(2);
                }
            } else {
                for i in 0..dim {
                    propagation_gradient +=
                        gd.dx_backward[i].min(0.0).powi(2) + gd.dx_forward[i].max(0.0).powi(2);
                }
            }

            gd.max_propagation_change = gd.max_propagation_change.max(propagation_term.abs());
            propagation_term *= propagation_gradient.sqrt();
        }

        curvature_term - propagation_term - advection_term
    }

    /// `LevelSetFunction::ComputeGlobalTimeStep` (itkLevelSetFunction.hxx:207-251).
    /// Consumes the per-iteration maxima and resets them, as ITK does.
    pub(super) fn compute_global_time_step(&self, gd: &mut GlobalData) -> f64 {
        gd.max_advection_change += gd.max_propagation_change;

        let mut dt = if gd.max_curvature_change.abs() > 0.0 {
            if gd.max_advection_change > 0.0 {
                (self.wave_dt / gd.max_advection_change).min(self.dt / gd.max_curvature_change)
            } else {
                self.dt / gd.max_curvature_change
            }
        } else if gd.max_advection_change > 0.0 {
            self.wave_dt / gd.max_advection_change
        } else {
            0.0
        };

        dt /= self.max_scale_coefficient;

        gd.max_advection_change = 0.0;
        gd.max_propagation_change = 0.0;
        gd.max_curvature_change = 0.0;

        dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A function with unit spacing, no advection, and the given weights.
    /// `CurvatureSpeed::Propagation`, as the geodesic active contour and shape
    /// detection functions use.
    fn function(
        n: usize,
        advection_weight: f64,
        propagation_weight: f64,
        curvature_weight: f64,
    ) -> LevelSetFunction {
        LevelSetFunction::new(
            vec![1.0; n],
            Vec::new(),
            advection_weight,
            propagation_weight,
            curvature_weight,
            CurvatureSpeed::Propagation,
            &[1.0, 1.0],
        )
    }

    /// The same, on a grid whose spacing is `spacing`.
    fn spaced(n: usize, propagation_weight: f64, spacing: &[f64]) -> LevelSetFunction {
        LevelSetFunction::new(
            vec![1.0; n],
            Vec::new(),
            0.0,
            propagation_weight,
            0.0,
            CurvatureSpeed::Propagation,
            spacing,
        )
    }

    /// `phi[x + 5*y] = f(x)` on a 5x5 grid: constant along `y`, so only the
    /// `x` axis contributes to any term.
    fn along_x(profile: [f64; 5]) -> Vec<f64> {
        let mut phi = vec![0.0; 25];
        for y in 0..5 {
            for (x, &v) in profile.iter().enumerate() {
                phi[x + 5 * y] = v;
            }
        }
        phi
    }

    /// The center pixel `(2, 2)` of a 5x5 grid.
    const CENTER: usize = 2 + 5 * 2;

    // ---- Godunov upwind switch on the sign of the propagation term --------

    /// `phi` rises only on the forward side of the center: `dx_backward == 0`,
    /// `dx_forward == 1`. A *positive* propagation term must then select
    /// `max(dx_backward, 0)^2 + min(dx_forward, 0)^2 == 0` — an outward-moving
    /// front looks backward, and there is no slope behind it.
    #[test]
    fn positive_propagation_upwinds_to_the_backward_difference() {
        let phi = along_x([0.0, 0.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);
        let mut gd = GlobalData::new(2);

        let f = function(25, 0.0, 1.0, 0.0);
        // update == -propagation_term == -(+1 * sqrt(0)) == 0
        assert_eq!(f.compute_update(&phi, &grid, CENTER, &mut gd), 0.0);
        assert_eq!(gd.dx_backward[0], 0.0);
        assert_eq!(gd.dx_forward[0], 1.0);
    }

    /// Same field, negative speed: the switch flips to
    /// `min(dx_backward, 0)^2 + max(dx_forward, 0)^2 == 1`, so the update is
    /// `-(-1 * 1) == +1`.
    #[test]
    fn negative_propagation_upwinds_to_the_forward_difference() {
        let phi = along_x([0.0, 0.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);
        let mut gd = GlobalData::new(2);

        let mut f = function(25, 0.0, 1.0, 0.0);
        f.speed = vec![-1.0; 25];
        assert_eq!(f.compute_update(&phi, &grid, CENTER, &mut gd), 1.0);
    }

    /// The propagation term's sign — not the speed's — drives the switch, so a
    /// negative weight against a positive speed behaves like a negative speed.
    #[test]
    fn the_propagation_switch_reads_the_weighted_term_not_the_raw_speed() {
        let phi = along_x([0.0, 0.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);
        let mut gd = GlobalData::new(2);

        let f = function(25, 0.0, -1.0, 0.0);
        assert_eq!(f.compute_update(&phi, &grid, CENTER, &mut gd), 1.0);
    }

    // ---- Curvature --------------------------------------------------------

    /// A straight interface: `phi = x - 2`. Every second derivative vanishes,
    /// so the mean curvature is exactly zero.
    #[test]
    fn curvature_of_a_straight_interface_is_zero() {
        let phi = along_x([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);
        let mut gd = GlobalData::new(2);

        let f = function(25, 0.0, 0.0, 1.0);
        f.compute_derivatives(&phi, &grid, &mut grid.coord(CENTER), &mut gd);
        assert_eq!(LevelSetFunction::mean_curvature(&gd, 2), 0.0);
    }

    /// A signed distance to a circle of radius `r` (negative inside) has
    /// `|grad(phi)| == 1`, so `ComputeMeanCurvature` returns `kappa == 1 / r`
    /// on the interface. Sampled on the `+x` axis at distance `r` from the
    /// center, for two radii.
    #[test]
    fn curvature_of_a_circle_is_one_over_the_radius() {
        let n = 41usize;
        let grid = Grid::new(&[n, n]);
        let c = 20.0;

        for radius in [6.0f64, 12.0] {
            let mut phi = vec![0.0; n * n];
            for y in 0..n {
                for x in 0..n {
                    let dx = x as f64 - c;
                    let dy = y as f64 - c;
                    phi[x + n * y] = (dx * dx + dy * dy).sqrt() - radius;
                }
            }

            let f = function(n * n, 0.0, 0.0, 1.0);
            let index = (c as usize + radius as usize) + n * c as usize;
            let mut gd = GlobalData::new(2);
            f.compute_derivatives(&phi, &grid, &mut grid.coord(index), &mut gd);

            let kappa = LevelSetFunction::mean_curvature(&gd, 2);
            let expected = 1.0 / radius;
            assert!(
                (kappa - expected).abs() < 0.1 * expected,
                "radius {radius}: kappa {kappa} vs expected {expected}"
            );
        }
    }

    /// `CurvatureSpeed::Unit` ignores the speed image; `CurvatureSpeed::
    /// Propagation` multiplies the curvature term by it. On a circle of radius
    /// `6` sampled on the interface, `kappa == 1/6`, so a speed of `3` triples
    /// the term only in the `Propagation` case. The update is `+curvature`
    /// because both other weights are zero.
    #[test]
    fn curvature_speed_selects_between_unit_and_the_speed_image() {
        let n = 41usize;
        let grid = Grid::new(&[n, n]);
        let c = 20.0;
        let radius = 6.0;
        let mut phi = vec![0.0; n * n];
        for y in 0..n {
            for x in 0..n {
                let dx = x as f64 - c;
                let dy = y as f64 - c;
                phi[x + n * y] = (dx * dx + dy * dy).sqrt() - radius;
            }
        }
        let index = 26 + n * 20;

        let mut unit = LevelSetFunction::new(
            vec![3.0; n * n],
            Vec::new(),
            0.0,
            0.0,
            1.0,
            CurvatureSpeed::Unit,
            &[1.0, 1.0],
        );
        let mut gd = GlobalData::new(2);
        let plain = unit.compute_update(&phi, &grid, index, &mut gd);

        unit.curvature_speed = CurvatureSpeed::Propagation;
        let mut gd = GlobalData::new(2);
        let scaled = unit.compute_update(&phi, &grid, index, &mut gd);

        assert!(
            (plain - 1.0 / radius).abs() < 0.01,
            "unit curvature {plain}"
        );
        assert!((scaled - 3.0 * plain).abs() < 1e-12, "scaled {scaled}");
    }

    // ---- Advection upwinding ----------------------------------------------

    /// `phi` has a steep backward slope (5) and a shallow forward slope (1).
    /// A positive `x_energy` (`weight * field > 0`) selects `dx_backward`, so
    /// the advection term is `field * 5 * weight` and the update is its
    /// negation.
    #[test]
    fn positive_advective_force_upwinds_to_the_backward_difference() {
        let phi = along_x([-10.0, -5.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);
        let mut gd = GlobalData::new(2);

        let mut f = function(25, 1.0, 0.0, 0.0);
        f.advection = vec![vec![1.0; 25], vec![0.0; 25]];
        assert_eq!(f.compute_update(&phi, &grid, CENTER, &mut gd), -5.0);
    }

    /// The same field with the advective force reversed selects `dx_forward`
    /// (slope 1), giving `-(-1 * 1 * 1) == +1`.
    #[test]
    fn negative_advective_force_upwinds_to_the_forward_difference() {
        let phi = along_x([-10.0, -5.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);
        let mut gd = GlobalData::new(2);

        let mut f = function(25, 1.0, 0.0, 0.0);
        f.advection = vec![vec![-1.0; 25], vec![0.0; 25]];
        assert_eq!(f.compute_update(&phi, &grid, CENTER, &mut gd), 1.0);
    }

    /// `x_energy` is `weight * field`, so flipping the weight flips the
    /// upwind side even though the field is unchanged. With `weight == -1` and
    /// `field == +1`, `x_energy < 0` selects `dx_forward == 1`; the term is
    /// `1 * 1 * (-1) == -1` and the update `+1`.
    #[test]
    fn the_advection_switch_reads_the_weighted_force_not_the_raw_field() {
        let phi = along_x([-10.0, -5.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);
        let mut gd = GlobalData::new(2);

        let mut f = function(25, -1.0, 0.0, 0.0);
        f.advection = vec![vec![1.0; 25], vec![0.0; 25]];
        assert_eq!(f.compute_update(&phi, &grid, CENTER, &mut gd), 1.0);
    }

    // ---- ComputeGlobalTimeStep --------------------------------------------

    /// No motion at all: every maximum is zero and so is the time step.
    #[test]
    fn time_step_is_zero_when_nothing_changes() {
        let f = function(1, 0.0, 0.0, 0.0);
        let mut gd = GlobalData::new(2);
        assert_eq!(f.compute_global_time_step(&mut gd), 0.0);
    }

    /// Propagation only: `dt = m_WaveDT / (max_advection + max_propagation)`.
    #[test]
    fn time_step_folds_propagation_into_the_advection_bound() {
        let f = function(1, 0.0, 1.0, 0.0);
        let mut gd = GlobalData::new(2);
        gd.max_propagation_change = 2.0;
        // wave_dt == 1/4 for 2-D; max_advection_change becomes 0 + 2.
        assert_eq!(f.compute_global_time_step(&mut gd), 0.125);
    }

    /// Curvature only: `dt = m_DT / max_curvature`.
    #[test]
    fn time_step_uses_the_curvature_bound_alone_when_there_is_no_wave_motion() {
        let f = function(1, 0.0, 0.0, 1.0);
        let mut gd = GlobalData::new(2);
        gd.max_curvature_change = 0.5;
        assert_eq!(f.compute_global_time_step(&mut gd), 0.5);
    }

    /// Both bounds active: the smaller wins.
    #[test]
    fn time_step_takes_the_tighter_of_the_two_bounds() {
        let f = function(1, 1.0, 1.0, 1.0);
        let mut gd = GlobalData::new(2);
        gd.max_advection_change = 10.0; // wave bound: 0.25 / 10 == 0.025
        gd.max_curvature_change = 0.5; //  curvature bound: 0.25 / 0.5 == 0.5
        assert_eq!(f.compute_global_time_step(&mut gd), 0.025);
    }

    /// `dt` is divided by the largest scale coefficient (`1 / min spacing`),
    /// so a half-pixel grid halves the step.
    #[test]
    fn time_step_is_divided_by_the_largest_scale_coefficient() {
        let f = spaced(1, 1.0, &[0.5, 1.0]);
        let mut gd = GlobalData::new(2);
        gd.max_propagation_change = 1.0;
        // wave_dt / 1.0 == 0.25, then / max(2.0, 1.0)
        assert_eq!(f.compute_global_time_step(&mut gd), 0.125);
    }

    /// The maxima are consumed: a second call with no new motion yields zero.
    #[test]
    fn time_step_resets_the_accumulated_maxima() {
        let f = function(1, 0.0, 1.0, 0.0);
        let mut gd = GlobalData::new(2);
        gd.max_propagation_change = 2.0;
        f.compute_global_time_step(&mut gd);
        assert_eq!(f.compute_global_time_step(&mut gd), 0.0);
    }

    // ---- Derivative scaling ------------------------------------------------

    /// `neighborhood_scales[d] == 1 / spacing[d]` converts index-space
    /// differences into physical units, so halving the spacing doubles every
    /// first derivative.
    #[test]
    fn derivatives_are_scaled_by_the_inverse_spacing() {
        let phi = along_x([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);

        let unit = spaced(25, 0.0, &[1.0, 1.0]);
        let mut gd = GlobalData::new(2);
        unit.compute_derivatives(&phi, &grid, &mut grid.coord(CENTER), &mut gd);
        assert_eq!(gd.dx[0], 1.0);

        let half = spaced(25, 0.0, &[0.5, 1.0]);
        let mut gd = GlobalData::new(2);
        half.compute_derivatives(&phi, &grid, &mut grid.coord(CENTER), &mut gd);
        assert_eq!(gd.dx[0], 2.0);
        assert_eq!(gd.dx_forward[0], 2.0);
        assert_eq!(gd.dx_backward[0], 2.0);
    }

    /// A border pixel's out-of-image neighbor clamps back onto itself
    /// (`ZeroFluxNeumannBoundaryCondition`), so the one-sided difference across
    /// the border is zero.
    #[test]
    fn out_of_image_neighbors_clamp_to_the_border_pixel() {
        let phi = along_x([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let grid = Grid::new(&[5, 5]);
        let f = function(25, 0.0, 0.0, 0.0);

        let mut gd = GlobalData::new(2);
        let left_edge = 5 * 2; // (0, 2)
        f.compute_derivatives(&phi, &grid, &mut grid.coord(left_edge), &mut gd);
        assert_eq!(gd.dx_backward[0], 0.0);
        assert_eq!(gd.dx_forward[0], 1.0);
        assert_eq!(gd.dx[0], 0.5);
    }
}

/// The level-set solver is the **other consumer** of the curvature-flow update,
/// and the one a signature change can break silently: it holds no window at all
/// — it reads a bare `f64` buffer through a clamping closure — so it does not go
/// through `Stencil`, and nothing about it fails to compile if the two sides of
/// the seam ever drift apart.
///
/// This pins them together. `CurvatureFlowFunction::compute_update` and
/// [`crate::denoise::curvature_flow_update`] must return **bit-identical** `f64`
/// for the same field at the same voxel — the solver reaching the body through
/// its `FnMut` closure, the filter reaching it through a borrowed
/// [`sitk_core::WindowView`] and through an owned [`sitk_core::Neighborhood`].
/// Three routes into one body; if any of them ever computes something different,
/// this fails.
#[cfg(test)]
mod seam_parity {
    use super::*;
    use crate::denoise::curvature_flow_update;
    use crate::level_set::grid::Grid;
    use sitk_core::{
        Image, NeighborhoodIterator, Stencil, ZeroFluxNeumannBoundaryCondition, parallel,
    };

    /// A field with real curvature — a smooth ramp would make every second
    /// derivative vanish and the update identically zero, and the pin would hold
    /// for the wrong reason.
    fn field(size: &[usize]) -> Vec<f64> {
        let n: usize = size.iter().product();
        (0..n)
            .map(|i| {
                let (x, y, z) = (
                    (i % size[0]) as f64,
                    ((i / size[0]) % size[1]) as f64,
                    (i / (size[0] * size[1])) as f64,
                );
                (0.6 * x).sin() * 9.0 + (0.4 * y).cos() * 5.0 + (0.3 * z + 0.2 * x).sin() * 3.0
            })
            .collect()
    }

    #[test]
    fn the_solvers_update_is_the_filters_update_bit_for_bit() {
        let size = [9usize, 8, 7];
        let phi = field(&size);
        let grid = Grid::new(&size);
        let scales = vec![1.0f64, 1.0, 1.0];
        let solver = CurvatureFlowFunction::new(scales.clone());

        let mut img = Image::from_vec(&size, phi.clone()).unwrap();
        img.set_spacing(&[1.0, 1.0, 1.0]).unwrap();
        let radius = vec![1usize; 3];
        let iter =
            NeighborhoodIterator::<f64, _>::new(&img, &radius, ZeroFluxNeumannBoundaryCondition)
                .unwrap();

        // Route 1: the filter's borrowed window (the code this refactor ships).
        let borrowed: Vec<f64> = iter.par_map_window(|_, w| curvature_flow_update(&w, 3, &scales));

        // Route 2: the filter's owned window (the code it replaced).
        let owned_iter =
            NeighborhoodIterator::<f64, _>::new(&img, &radius, ZeroFluxNeumannBoundaryCondition)
                .unwrap();
        let owned: Vec<f64> = owned_iter
            .map(|(_, nb)| curvature_flow_update(&nb, 3, &scales))
            .collect();

        // Route 3: the solver's closure — no window of any kind, clamping reads
        // straight out of the buffer.
        let solved: Vec<f64> = (0..phi.len())
            .map(|i| solver.compute_update(&phi, &grid, i))
            .collect();

        // Non-vacuity: an all-zero update field would make all three agree no
        // matter how badly one of them was broken.
        let nonzero = solved.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > solved.len() / 2,
            "only {nonzero}/{} updates are non-zero — the field is too flat for this pin to \
             mean anything",
            solved.len()
        );

        for i in 0..phi.len() {
            assert_eq!(
                borrowed[i].to_bits(),
                owned[i].to_bits(),
                "voxel {i}: the borrowed window and the owned window disagree ({:?} vs {:?})",
                borrowed[i],
                owned[i]
            );
            assert_eq!(
                solved[i].to_bits(),
                owned[i].to_bits(),
                "voxel {i}: the level-set solver and the filter disagree ({:?} vs {:?})",
                solved[i],
                owned[i]
            );
        }

        // And the borrowed route does not depend on the thread count.
        for threads in [1usize, 4, 48, 96] {
            let got: Vec<f64> = parallel::with_threads(threads, || {
                iter.par_map_window(|_, w| curvature_flow_update(&w, 3, &scales))
            });
            for (i, (a, b)) in got.iter().zip(&borrowed).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "voxel {i} moved with {threads} threads"
                );
            }
        }

        // The `Stencil` trait's own two routes to the same value, at one voxel:
        // by offset, and by slot. `compute_threshold` reads windows both ways.
        let center = [4usize, 4, 3];
        let nb = iter.neighborhood_at(&center);
        assert_eq!(
            Stencil::center(&nb).to_bits(),
            Stencil::at(&nb, &[0i64, 0, 0]).to_bits(),
            "the center slot and the zero offset name different values"
        );
    }
}
