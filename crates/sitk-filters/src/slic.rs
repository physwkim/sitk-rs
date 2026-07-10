//! `SLICImageFilter`: Simple Linear Iterative Clustering super-pixel
//! segmentation.
//!
//! Port of `itk::SLICImageFilter` (`itkSLICImageFilter.h` / `.hxx`) as
//! wrapped by SimpleITK's `SLICImageFilter.yaml` (`output_pixel_type:
//! uint32_t`, `SuperGridSize` default `[50, 50, 50]`,
//! `SpatialProximityWeight` `10.0`, `MaximumNumberOfIterations` `5`,
//! `EnforceConnectivity` / `InitializationPerturbation` `true`).
//!
//! SLIC is k-means in the joint domain of index space and pixel value. A
//! cluster is a vector `[value, i₀, …, i_{D-1}]`; the distance between a
//! cluster and a pixel at index `p` is
//!
//! ```text
//! d(c, v, p) = (c.value − v)² + Σ_d ((c.i_d − p_d) · scale_d)²
//! scale_d    = SpatialProximityWeight / SuperGridSize[d]
//! ```
//!
//! so the spatial term is normalised by the super-grid size — a pixel one
//! full grid cell away contributes exactly `SpatialProximityWeight²` per
//! axis. ITK evaluates this in `TDistancePixel` (`float` by default, which
//! is what SimpleITK instantiates), and this port does the same: the
//! per-term differences are formed in `f64` and narrowed to `f32` before
//! squaring and accumulating.
//!
//! # Algorithm
//!
//! 1. **Grid initialisation.** Cluster centres come from shrinking the input
//!    by `SuperGridSize` (`itk::ShrinkImageFilter`; see [`crate::shrink`]).
//!    The centre of grid cell `j` sits at continuous index
//!    `j·f + δ` with `δ_d = (size_d−1)/2 − f_d·(outSize_d−1)/2`, and takes
//!    the value of input pixel `j·f + round(δ)`.
//! 2. **Perturbation** (`InitializationPerturbation`, auto-disabled when any
//!    `SuperGridSize[d] < 3`). Each centre moves to the index of minimum
//!    gradient magnitude (Frobenius norm of the central-difference Jacobian,
//!    scaled by spacing) within its `3^D` neighbourhood, taking that pixel's
//!    value. Out-of-image reads use zero-flux Neumann (clamp) boundaries,
//!    matching ITK's default `ConstNeighborhoodIterator`.
//! 3. **Iteration** (`MaximumNumberOfIterations` times). The distance image is
//!    reset to `+∞`; each cluster `i`, in ascending order, scans the box of
//!    radius `SuperGridSize` around its rounded centre and claims every pixel
//!    whose stored distance is *strictly* greater than `d(c_i, v, p)` — so
//!    ties go to the lower cluster index. Then every cluster is replaced by
//!    the mean value/index of the pixels labelled with it, and
//!    `average_residual = sqrt(Σ_i d(c_i, c_i^old)) / (numClusters·(D+1))`.
//! 4. **Connectivity** (`EnforceConnectivity`). See below.
//!
//! # Enforce-connectivity pass
//!
//! `minSuperSize = Π SuperGridSize[d] / 4`. First, for each cluster `i` in
//! ascending order, the pixel at the rounded centre is checked for label `i`;
//! if it does not carry it, the `SuperGridSize/2`-radius neighbourhood is
//! scanned in raster order for the first pixel that does. If found, that
//! pixel's face-connected component of label `i` is marked; if the component
//! is smaller than `minSuperSize` the marks are removed again. Clusters whose
//! label is nowhere in that neighbourhood are simply skipped.
//!
//! Then a raster scan visits every unmarked pixel: its face-connected
//! component (of its current label) is relabelled to `nextLabel`, starting at
//! `numClusters`. If the component reaches `minSuperSize` it keeps that new
//! label and `nextLabel` advances; otherwise the whole component is
//! overwritten with `prevLabel` — **the label of the most recently visited
//! marked pixel in raster order**, which is ITK's exact merge-target choice
//! and need not be a spatial neighbour. `prevLabel` starts at `numClusters`,
//! an id no cluster owns, so an image whose very first pixel is unmarked and
//! whose component is undersized is labelled with that unused id. Both
//! quirks are reproduced.
//!
//! # Labels
//!
//! There is no final relabel step in ITK. Output labels are **cluster indices
//! starting at 0**, plus the `≥ numClusters` ids minted by the connectivity
//! pass. They are therefore *not* guaranteed contiguous: a cluster whose
//! pixels are all claimed by neighbours leaves a hole, and undersized
//! components consume no id. [`slic`] reproduces this; run
//! [`crate::relabel_component`] afterwards if contiguity is needed.
//!
//! # Restrictions relative to ITK
//!
//! - **Scalar input only.** ITK templates over scalar *and* vector/RGB images
//!   (`numberOfComponents > 1` widens the cluster vector and the intensity
//!   term of the distance). This crate has no vector pixel type, so only the
//!   `numberOfComponents == 1` instantiation is ported. The distance,
//!   perturbation gradient and cluster update are all written for one
//!   component.
//! - **Single-threaded.** ITK reduces per-thread cluster maps and runs the
//!   per-cluster connectivity marking with `ParallelizeArray`; both are
//!   order-independent for the marking phase, and the reduction is a sum, so
//!   the sequential evaluation here differs only in floating-point summation
//!   order.
//! - ITK leaves the output buffer uninitialised before the first labelling
//!   sweep; a pixel outside every cluster's search box would read garbage.
//!   Here it reads 0.
//! - An empty cluster divides its accumulator by zero. ITK's
//!   `vnl_vector /= 0` yields `NaN`, then rounds the `NaN` centre to an index
//!   (undefined behaviour in C++). This port keeps the `NaN` and defines the
//!   `f64 → i64` conversion as Rust's saturating cast (`NaN → 0`); the `NaN`
//!   centre then loses every distance comparison, so the cluster stays empty.
//! - Perturbation indices are clamped to the image before they are read or
//!   stored. ITK reads them unclamped, which is out of bounds only on axes
//!   shorter than the `3`-pixel perturbation window.

use crate::error::{FilterError, Result};
use sitk_core::Image;

/// Parameters of [`slic`], mirroring SimpleITK's `SLICImageFilter` members.
#[derive(Debug, Clone, PartialEq)]
pub struct SlicSettings {
    /// Requested super-pixel size per axis; also the shrink factor used to
    /// seed the grid and the radius of each cluster's search box. Must hold
    /// at least one entry per image dimension (extras are ignored, as in
    /// SimpleITK's `sitkSTLVectorToITK`) and every entry must be `>= 1`.
    pub super_grid_size: Vec<u32>,
    /// Weight of the spatial term in the joint distance. Larger values give
    /// more regular super-pixel shapes.
    pub spatial_proximity_weight: f64,
    /// Number of k-means sweeps.
    pub maximum_number_of_iterations: u32,
    /// Run the post-processing pass that makes every label spatially
    /// connected.
    pub enforce_connectivity: bool,
    /// Move each initial centre to the local gradient minimum. Automatically
    /// disabled when any `super_grid_size[d] < 3`.
    pub initialization_perturbation: bool,
}

impl Default for SlicSettings {
    /// SimpleITK's defaults: `SuperGridSize = [50, 50, 50]`,
    /// `SpatialProximityWeight = 10.0`, `MaximumNumberOfIterations = 5`,
    /// `EnforceConnectivity = true`, `InitializationPerturbation = true`.
    fn default() -> Self {
        Self {
            super_grid_size: vec![50, 50, 50],
            spatial_proximity_weight: 10.0,
            maximum_number_of_iterations: 5,
            enforce_connectivity: true,
            initialization_perturbation: true,
        }
    }
}

/// Output of [`slic`].
#[derive(Debug, Clone)]
pub struct SlicResult {
    /// `UInt32` label image; see the module docs for the labelling scheme.
    pub labels: Image,
    /// ITK's `AverageResidual` measurement after the final iteration:
    /// `sqrt(Σ_i d(c_i, c_i^old)) / (numClusters · (D+1))`. Stays `f64::MAX`
    /// (ITK's `NumericTraits<double>::max()`) when no iteration ran.
    pub average_residual: f64,
}

/// `itk::Math::RoundHalfIntegerUp`: `floor(x + 0.5)`.
///
/// The `as i64` cast saturates, so `NaN` (an empty cluster's centre) maps to
/// `0` instead of C++'s undefined behaviour.
fn round_half_integer_up(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

/// First-index-fastest strides for a size vector.
fn strides(size: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; size.len()];
    for d in 1..size.len() {
        s[d] = s[d - 1] * size[d - 1];
    }
    s
}

/// An image's index space: extent per axis plus the first-index-fastest
/// strides that flatten it.
struct Lattice {
    size: Vec<usize>,
    stride: Vec<usize>,
}

impl Lattice {
    fn new(size: &[usize]) -> Self {
        Self {
            size: size.to_vec(),
            stride: strides(size),
        }
    }

    fn dim(&self) -> usize {
        self.size.len()
    }

    /// Coordinate along axis `d` of a flat index.
    fn coord(&self, flat: usize, d: usize) -> usize {
        (flat / self.stride[d]) % self.size[d]
    }

    fn flatten(&self, idx: &[usize]) -> usize {
        idx.iter().zip(&self.stride).map(|(&i, &s)| i * s).sum()
    }

    /// Flat index of a signed coordinate, or `None` when it leaves the image.
    fn flatten_signed(&self, coord: &[i64]) -> Option<usize> {
        let mut flat = 0usize;
        for (d, &c) in coord.iter().enumerate() {
            if c < 0 || c >= self.size[d] as i64 {
                return None;
            }
            flat += c as usize * self.stride[d];
        }
        Some(flat)
    }

    /// The two face neighbours of `flat` along axis `d`, in ITK's
    /// `{center + stride, center - stride}` order, skipping those outside.
    fn face_neighbors(&self, flat: usize, d: usize) -> impl Iterator<Item = usize> {
        let coord = self.coord(flat, d);
        let stride = self.stride[d];
        [
            (coord + 1 < self.size[d]).then(|| flat + stride),
            (coord > 0).then(|| flat - stride),
        ]
        .into_iter()
        .flatten()
    }
}

/// Simple Linear Iterative Clustering super-pixel segmentation.
///
/// Returns a `UInt32` label image (labels start at 0, are *not* necessarily
/// contiguous) and the final `AverageResidual`. See the [module docs] for the
/// algorithm, the scalar-only restriction, and the reproduced ITK quirks.
///
/// Errors when `super_grid_size` has fewer entries than the image has
/// dimensions, or when any of the first `dim` entries is zero.
///
/// [module docs]: self
pub fn slic(img: &Image, settings: &SlicSettings) -> Result<SlicResult> {
    let dim = img.dimension();
    if settings.super_grid_size.len() < dim {
        return Err(FilterError::DimensionLength {
            expected: dim,
            got: settings.super_grid_size.len(),
        });
    }
    let grid: Vec<usize> = settings.super_grid_size[..dim]
        .iter()
        .map(|&g| g as usize)
        .collect();
    if grid.contains(&0) {
        return Err(FilterError::InvalidSuperGridSize(
            settings.super_grid_size.clone(),
        ));
    }

    let size = img.size().to_vec();
    let total: usize = size.iter().product();
    if total == 0 {
        let mut labels = Image::from_vec(&size, Vec::<u32>::new())?;
        labels.copy_geometry_from(img);
        return Ok(SlicResult {
            labels,
            average_residual: f64::MAX,
        });
    }

    let lat = Lattice::new(&size);
    let vals = img.to_f64_vec()?;
    let spacing = img.spacing().to_vec();

    // ---- grid initialisation (itk::ShrinkImageFilter geometry) ------------
    //
    // Shrinking by `grid` produces `out_size` samples per axis; the shrunk
    // pixel `j` sits at continuous index `j·f + δ` of the input and carries
    // the value of input pixel `j·f + round(δ)`. ITK's SLIC recovers exactly
    // that continuous index via TransformIndexToPhysicalPoint ∘
    // TransformPhysicalPointToContinuousIndex, which is spacing/direction
    // independent.
    let mut out_size = vec![0usize; dim];
    let mut delta = vec![0.0f64; dim];
    let mut offset = vec![0usize; dim];
    for d in 0..dim {
        let f = grid[d];
        out_size[d] = (size[d] / f).max(1);
        delta[d] = (size[d] as f64 - 1.0) / 2.0 - f as f64 * (out_size[d] as f64 - 1.0) / 2.0;
        offset[d] = (delta[d] + 0.5).floor().max(0.0) as usize;
        offset[d] = offset[d].min(size[d] - 1);
    }
    let num_clusters: usize = out_size.iter().product();
    let out_stride = strides(&out_size);

    // Cluster layout: `[value, i₀, …, i_{D-1}]` per cluster, one component
    // because the input is scalar.
    let ncomp = 1 + dim;
    let mut clusters = vec![0.0f64; num_clusters * ncomp];
    for c in 0..num_clusters {
        let mut flat = 0usize;
        for d in 0..dim {
            let j = (c / out_stride[d]) % out_size[d];
            flat += (j * grid[d] + offset[d]).min(size[d] - 1) * lat.stride[d];
            clusters[c * ncomp + 1 + d] = (j * grid[d]) as f64 + delta[d];
        }
        clusters[c * ncomp] = vals[flat];
    }

    let scales: Vec<f64> = (0..dim)
        .map(|d| settings.spatial_proximity_weight / grid[d] as f64)
        .collect();

    // ---- perturbation -----------------------------------------------------
    if settings.initialization_perturbation && grid.iter().all(|&g| g >= 3) {
        for c in 0..num_clusters {
            perturb_cluster(
                &mut clusters[c * ncomp..(c + 1) * ncomp],
                &vals,
                &lat,
                &spacing,
            );
        }
    }

    // ---- main loop --------------------------------------------------------
    let mut labels = vec![0u32; total];
    let mut distance = vec![0.0f32; total];
    let mut old_clusters = vec![0.0f64; clusters.len()];
    let mut counts = vec![0usize; num_clusters];
    let mut average_residual = f64::MAX;

    for _ in 0..settings.maximum_number_of_iterations {
        distance.fill(f32::MAX);

        for c in 0..num_clusters {
            let cluster = &clusters[c * ncomp..(c + 1) * ncomp];
            let Some((lo, hi)) = search_box(cluster, &grid, &lat.size) else {
                continue;
            };
            for_each_index(&lo, &hi, &mut |idx: &[usize]| {
                let flat = lat.flatten(idx);
                let d = pixel_distance(cluster, vals[flat], idx, &scales);
                if d < distance[flat] {
                    distance[flat] = d;
                    labels[flat] = c as u32;
                }
            });
        }

        // Replace every cluster by the mean of its members. ITK reduces
        // per-thread maps into a zeroed array, then divides by the count.
        std::mem::swap(&mut clusters, &mut old_clusters);
        clusters.fill(0.0);
        counts.fill(0);
        for (flat, &label) in labels.iter().enumerate() {
            let c = label as usize;
            counts[c] += 1;
            clusters[c * ncomp] += vals[flat];
            for d in 0..dim {
                clusters[c * ncomp + 1 + d] += lat.coord(flat, d) as f64;
            }
        }

        let mut l1_residual = 0.0f64;
        for c in 0..num_clusters {
            let n = counts[c] as f64;
            for k in 0..ncomp {
                clusters[c * ncomp + k] /= n;
            }
            l1_residual += f64::from(cluster_distance(
                &clusters[c * ncomp..(c + 1) * ncomp],
                &old_clusters[c * ncomp..(c + 1) * ncomp],
                &scales,
            ));
        }
        average_residual = l1_residual.sqrt() / clusters.len() as f64;
    }

    // ---- enforce connectivity ---------------------------------------------
    if settings.enforce_connectivity {
        enforce_connectivity(&clusters, ncomp, &grid, &lat, &mut labels);
    }

    let mut out = Image::from_vec(&size, labels)?;
    out.copy_geometry_from(img);
    Ok(SlicResult {
        labels: out,
        average_residual,
    })
}

/// `d1 + d2` of `SLICImageFilter::Distance(cluster, v, pt)`, evaluated in
/// `TDistancePixel = float` exactly as ITK does.
fn pixel_distance(cluster: &[f64], v: f64, idx: &[usize], scales: &[f64]) -> f32 {
    let d = (cluster[0] - v) as f32;
    let mut acc = d * d;
    for (j, &scale) in scales.iter().enumerate() {
        let d = ((cluster[1 + j] - idx[j] as f64) * scale) as f32;
        acc += d * d;
    }
    acc
}

/// `SLICImageFilter::Distance(cluster1, cluster2)`, the residual metric.
fn cluster_distance(a: &[f64], b: &[f64], scales: &[f64]) -> f32 {
    let d = (a[0] - b[0]) as f32;
    let mut acc = d * d;
    for (j, &scale) in scales.iter().enumerate() {
        let d = ((a[1 + j] - b[1 + j]) * scale) as f32;
        acc += d * d;
    }
    acc
}

/// The cluster's search box: the rounded centre padded by `grid` and cropped
/// to the image. `None` when the crop is empty (ITK's `Region::Crop` returning
/// false).
fn search_box(cluster: &[f64], grid: &[usize], size: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
    let dim = size.len();
    let mut lo = vec![0usize; dim];
    let mut hi = vec![0usize; dim];
    for d in 0..dim {
        let centre = round_half_integer_up(cluster[1 + d]);
        let l = centre - grid[d] as i64;
        let h = centre + grid[d] as i64;
        if h < 0 || l >= size[d] as i64 {
            return None;
        }
        lo[d] = l.max(0) as usize;
        hi[d] = h.min(size[d] as i64 - 1) as usize;
    }
    Some((lo, hi))
}

/// Visit every index of the inclusive box `[lo, hi]` in raster order (axis 0
/// fastest), matching ITK's scanline iteration.
fn for_each_index(lo: &[usize], hi: &[usize], f: &mut dyn FnMut(&[usize])) {
    let dim = lo.len();
    let mut idx = lo.to_vec();
    loop {
        f(&idx);
        let mut d = 0;
        loop {
            if d == dim {
                return;
            }
            if idx[d] < hi[d] {
                idx[d] += 1;
                break;
            }
            idx[d] = lo[d];
            d += 1;
        }
    }
}

/// `SLICImageFilter::ThreadedPerturbClusters` for one cluster: move the centre
/// to the minimum-gradient index of its `3^D` neighbourhood and adopt that
/// pixel's value.
fn perturb_cluster(cluster: &mut [f64], vals: &[f64], lat: &Lattice, spacing: &[f64]) {
    let dim = lat.dim();
    // Zero-flux Neumann boundary (ITK's `ConstNeighborhoodIterator` default).
    let at = |coord: &[i64]| -> f64 {
        let flat: usize = (0..dim)
            .map(|d| coord[d].clamp(0, lat.size[d] as i64 - 1) as usize * lat.stride[d])
            .sum();
        vals[flat]
    };

    let centre: Vec<i64> = (0..dim)
        .map(|d| round_half_integer_up(cluster[1 + d]))
        .collect();
    let lo: Vec<i64> = centre.iter().map(|&c| c - 1).collect();
    let hi: Vec<i64> = centre.iter().map(|&c| c + 1).collect();

    let mut min_g = f64::MAX;
    let mut min_idx = centre.clone();
    let mut p = lo.clone();
    let mut probe = vec![0i64; dim];
    loop {
        // Frobenius norm² of the central-difference Jacobian.
        let mut g_norm = 0.0f64;
        for d in 0..dim {
            probe.copy_from_slice(&p);
            probe[d] = p[d] + 1;
            let forward = at(&probe);
            probe[d] = p[d] - 1;
            let backward = at(&probe);
            let j = (forward - backward) / (2.0 * spacing[d]);
            g_norm += j * j;
        }
        if g_norm < min_g {
            min_g = g_norm;
            min_idx.copy_from_slice(&p);
        }

        let mut d = 0;
        loop {
            if d == dim {
                // Clamp before reading/storing: ITK reads `min_idx` unclamped,
                // which is out of bounds only on axes shorter than 3.
                for d in 0..dim {
                    min_idx[d] = min_idx[d].clamp(0, lat.size[d] as i64 - 1);
                    cluster[1 + d] = min_idx[d] as f64;
                }
                cluster[0] = at(&min_idx);
                return;
            }
            if p[d] < hi[d] {
                p[d] += 1;
                break;
            }
            p[d] = lo[d];
            d += 1;
        }
    }
}

/// `ThreadedConnectivity` over every cluster followed by
/// `SingleThreadedConnectivity`.
fn enforce_connectivity(
    clusters: &[f64],
    ncomp: usize,
    grid: &[usize],
    lat: &Lattice,
    labels: &mut [u32],
) {
    let num_clusters = clusters.len() / ncomp;
    let min_super_size = grid.iter().product::<usize>() / 4;
    let mut marker = vec![0u8; labels.len()];
    let mut stack = Vec::new();

    for c in 0..num_clusters {
        let label = c as u32;
        let centre: Vec<i64> = (0..lat.dim())
            .map(|d| round_half_integer_up(clusters[c * ncomp + 1 + d]))
            .collect();

        let seed = lat
            .flatten_signed(&centre)
            .filter(|&flat| labels[flat] == label)
            .or_else(|| search_neighborhood(&centre, label, grid, lat, labels));

        // A cluster whose label is nowhere in its search neighbourhood is
        // skipped entirely (ITK only warns, in debug builds).
        if let Some(seed) = seed {
            relabel_connected_region(seed, label, label, lat, labels, &mut marker, &mut stack);
            if stack.len() < min_super_size {
                for &flat in &stack {
                    marker[flat] = 0;
                }
            }
        }
    }

    // Relabel every component not reached above. Undersized components take
    // `prev_label`: the label of the last marked pixel seen in raster order,
    // seeded with `num_clusters` (an id no cluster owns).
    let mut next_label = num_clusters as u32;
    let mut prev_label = num_clusters as u32;
    for flat in 0..labels.len() {
        if marker[flat] == 0 {
            let required = labels[flat];
            relabel_connected_region(
                flat,
                required,
                next_label,
                lat,
                labels,
                &mut marker,
                &mut stack,
            );
            if stack.len() >= min_super_size {
                prev_label = next_label;
                next_label += 1;
            } else {
                for &s in &stack {
                    labels[s] = prev_label;
                }
            }
        } else {
            prev_label = labels[flat];
        }
    }
}

/// Raster-order scan of the `grid/2`-radius neighbourhood around `centre` for
/// the first pixel carrying `label`. Out-of-image neighbours read ITK's
/// constant boundary value (`NumericTraits<uint32>::max()`), which no cluster
/// label can equal.
fn search_neighborhood(
    centre: &[i64],
    label: u32,
    grid: &[usize],
    lat: &Lattice,
    labels: &[u32],
) -> Option<usize> {
    let radius: Vec<i64> = grid.iter().map(|&g| (g / 2) as i64).collect();
    let mut p: Vec<i64> = (0..lat.dim()).map(|d| centre[d] - radius[d]).collect();
    loop {
        if let Some(flat) = lat.flatten_signed(&p) {
            if labels[flat] == label {
                return Some(flat);
            }
        }
        let mut d = 0;
        loop {
            if d == lat.dim() {
                return None;
            }
            if p[d] < centre[d] + radius[d] {
                p[d] += 1;
                break;
            }
            p[d] = centre[d] - radius[d];
            d += 1;
        }
    }
}

/// `SLICImageFilter::RelabelConnectedRegion`: face-connected flood fill from
/// `seed` over pixels labelled `required` that are not yet marked, marking
/// them and (when `required != output`) relabelling them to `output`. The
/// visited flat indices are left in `stack`.
fn relabel_connected_region(
    seed: usize,
    required: u32,
    output: u32,
    lat: &Lattice,
    labels: &mut [u32],
    marker: &mut [u8],
    stack: &mut Vec<usize>,
) {
    stack.clear();
    stack.push(seed);
    marker[seed] = 1;
    if required != output {
        labels[seed] = output;
    }

    let mut head = 0usize;
    while head < stack.len() {
        let flat = stack[head];
        head += 1;
        for d in 0..lat.dim() {
            for neighbor in lat.face_neighbors(flat, d) {
                if labels[neighbor] == required && marker[neighbor] == 0 {
                    stack.push(neighbor);
                    marker[neighbor] = 1;
                    if required != output {
                        labels[neighbor] = output;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    fn settings(grid: &[u32], weight: f64, iters: u32) -> SlicSettings {
        SlicSettings {
            super_grid_size: grid.to_vec(),
            spatial_proximity_weight: weight,
            maximum_number_of_iterations: iters,
            enforce_connectivity: false,
            initialization_perturbation: false,
        }
    }

    fn labels_of(result: &SlicResult) -> Vec<u32> {
        result
            .labels
            .to_f64_vec()
            .unwrap()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }

    /// Number of face-connected components carrying each label.
    fn components_per_label(labels: &[u32], size: &[usize]) -> HashMap<u32, usize> {
        let lat = Lattice::new(size);
        let mut seen = vec![false; labels.len()];
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for start in 0..labels.len() {
            if seen[start] {
                continue;
            }
            *counts.entry(labels[start]).or_insert(0) += 1;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(flat) = stack.pop() {
                for d in 0..lat.dim() {
                    for n in lat.face_neighbors(flat, d) {
                        if !seen[n] && labels[n] == labels[start] {
                            seen[n] = true;
                            stack.push(n);
                        }
                    }
                }
            }
        }
        counts
    }

    /// 12×12, grid 4 → centres at 1.5/5.5/9.5 on each axis, so the pure
    /// spatial partition is 4×4 blocks labelled `bx + 3·by`.
    fn expected_grid_labels_12x12() -> Vec<u32> {
        let mut expected = vec![0u32; 144];
        for y in 0..12 {
            for x in 0..12 {
                expected[y * 12 + x] = (x / 4 + 3 * (y / 4)) as u32;
            }
        }
        expected
    }

    #[test]
    fn constant_image_yields_the_pure_grid() {
        let img = Image::from_vec(&[12, 12], vec![7.0f64; 144]).unwrap();
        let out = slic(&img, &settings(&[4, 4], 10.0, 1)).unwrap();
        assert_eq!(out.labels.pixel_id(), sitk_core::PixelId::UInt32);
        assert_eq!(labels_of(&out), expected_grid_labels_12x12());
    }

    #[test]
    fn two_blobs_separate_into_distinct_superpixels() {
        // 16×8, left half 0 / right half 200; grid 8 → exactly two clusters,
        // seeded at (3.5, 3.5) with value 0 and (11.5, 3.5) with value 200.
        // The intensity term (200² = 40000) dwarfs the spatial term, so the
        // labelling reproduces the blob boundary at x = 8.
        let data: Vec<f64> = (0..128)
            .map(|i| if i % 16 < 8 { 0.0 } else { 200.0 })
            .collect();
        let img = Image::from_vec(&[16, 8], data).unwrap();
        let out = slic(&img, &settings(&[8, 8], 10.0, 1)).unwrap();
        let labels = labels_of(&out);
        for y in 0..8 {
            for x in 0..16 {
                let want = u32::from(x >= 8);
                assert_eq!(labels[y * 16 + x], want, "at ({x}, {y})");
            }
        }
    }

    /// 12×12, value 0 for `x < 6` and 100 otherwise. Cluster centres sample
    /// (2,·) → 0 and (6,·)/(10,·) → 100.
    fn step_image_12x12() -> Image {
        let data: Vec<f64> = (0..144)
            .map(|i| if i % 12 < 6 { 0.0 } else { 100.0 })
            .collect();
        Image::from_vec(&[12, 12], data).unwrap()
    }

    #[test]
    fn huge_spatial_weight_is_grid_like() {
        let out = slic(&step_image_12x12(), &settings(&[4, 4], 1.0e6, 1)).unwrap();
        assert_eq!(labels_of(&out), expected_grid_labels_12x12());
    }

    #[test]
    fn tiny_spatial_weight_is_intensity_dominated() {
        let img = step_image_12x12();
        let out = slic(&img, &settings(&[4, 4], 1.0e-6, 1)).unwrap();
        let labels = labels_of(&out);
        let vals = img.to_f64_vec().unwrap();
        // Every pixel joins a cluster seeded with its own intensity, i.e. the
        // super-pixel boundary follows the step, not the grid.
        for (flat, &label) in labels.iter().enumerate() {
            let seed_x = 2 + 4 * (label as usize % 3);
            let cluster_value = if seed_x < 6 { 0.0 } else { 100.0 };
            assert_eq!(cluster_value, vals[flat], "at flat index {flat}");
        }
        // The grid label would have put x = 4, 5 (value 0) in column-1
        // clusters, whose seed value is 100.
        assert_eq!(labels[4] % 3, 0);
        assert_eq!(labels[5] % 3, 0);
    }

    /// 12×12 of 100s with a 0 at the cluster-0 seed (2,2) and an isolated 0 at
    /// (5,5) — the latter still inside cluster 0's `grid`-radius search box
    /// [0,6]², but not a grid seed point. With a negligible spatial weight both
    /// zero pixels join cluster 0, which is therefore disconnected before the
    /// connectivity pass. `min_super_size` is 4·4/4 = 4, so both singletons are
    /// undersized.
    fn island_image() -> Image {
        let mut data = vec![100.0f64; 144];
        data[2 * 12 + 2] = 0.0;
        data[5 * 12 + 5] = 0.0;
        Image::from_vec(&[12, 12], data).unwrap()
    }

    #[test]
    fn without_connectivity_a_label_can_have_two_components() {
        let out = slic(&island_image(), &settings(&[4, 4], 1.0e-6, 1)).unwrap();
        let labels = labels_of(&out);
        assert_eq!(labels[2 * 12 + 2], 0);
        assert_eq!(labels[5 * 12 + 5], 0);
        // Three components of label 0: the two zero-valued singletons, plus the
        // 2×2 corner block [0,1]² that no other cluster's search box reaches,
        // so cluster 0 claims it despite the 100² intensity penalty.
        assert_eq!(components_per_label(&labels, &[12, 12])[&0], 3);
    }

    #[test]
    fn enforce_connectivity_removes_the_island() {
        let mut s = settings(&[4, 4], 1.0e-6, 1);
        s.enforce_connectivity = true;
        let out = slic(&island_image(), &s).unwrap();
        let labels = labels_of(&out);

        // The two orphan pixels no longer share a label ...
        assert_ne!(labels[2 * 12 + 2], labels[5 * 12 + 5]);
        // ... and every surviving label is a single connected component.
        for (label, count) in components_per_label(&labels, &[12, 12]) {
            assert_eq!(count, 1, "label {label} has {count} components");
        }
    }

    #[test]
    fn labels_start_at_zero_and_are_contiguous_for_the_pure_grid() {
        let img = Image::from_vec(&[12, 12], vec![7.0f64; 144]).unwrap();
        let mut s = settings(&[4, 4], 10.0, 1);
        s.enforce_connectivity = true;
        let out = slic(&img, &s).unwrap();
        let distinct: BTreeSet<u32> = labels_of(&out).into_iter().collect();
        // 9 clusters, each a 4×4 block ≥ min_super_size = 4: nothing merges,
        // nothing new is minted. ITK has no final relabel step; the ids are
        // the cluster indices themselves.
        assert_eq!(distinct, (0..9).collect::<BTreeSet<u32>>());
    }

    #[test]
    fn labels_need_not_be_contiguous() {
        // Cluster 0 owns only the two singleton zeros, both undersized, so it
        // vanishes in the connectivity pass while its neighbours keep their
        // ids: label 0 is absent from the output even though labels start at 0.
        let mut s = settings(&[4, 4], 1.0e-6, 1);
        s.enforce_connectivity = true;
        let out = slic(&island_image(), &s).unwrap();
        let distinct: BTreeSet<u32> = labels_of(&out).into_iter().collect();
        assert!(!distinct.contains(&0), "got {distinct:?}");
    }

    #[test]
    fn perturbation_moves_centres_to_the_gradient_minimum() {
        // A constant image has zero gradient everywhere, so the minimum-gradient
        // search keeps the first neighbourhood index it visits: centre − 1 on
        // every axis. Grid 3 on a 12-axis seeds centres at 1, 4, 7, 10; after
        // perturbation they sit at 0, 3, 6, 9.
        let img = Image::from_vec(&[12, 12], vec![7.0f64; 144]).unwrap();

        let plain = labels_of(&slic(&img, &settings(&[3, 3], 10.0, 1)).unwrap());
        let mut s = settings(&[3, 3], 10.0, 1);
        s.initialization_perturbation = true;
        let perturbed = labels_of(&slic(&img, &s).unwrap());

        // Pixel (2,2): nearest unperturbed centre is 1 on both axes → cluster
        // (0,0); nearest perturbed centre is 3 on both axes → cluster (1,1).
        assert_eq!(plain[2 * 12 + 2], 0);
        assert_eq!(perturbed[2 * 12 + 2], 1 + 4);
    }

    #[test]
    fn perturbation_is_disabled_below_grid_size_three() {
        let img = Image::from_vec(&[12, 12], vec![7.0f64; 144]).unwrap();
        let mut s = settings(&[2, 2], 10.0, 1);
        s.initialization_perturbation = true;
        let on = labels_of(&slic(&img, &s).unwrap());
        let off = labels_of(&slic(&img, &settings(&[2, 2], 10.0, 1)).unwrap());
        assert_eq!(on, off);
    }

    #[test]
    fn zero_iterations_leaves_the_residual_at_double_max() {
        let img = Image::from_vec(&[12, 12], vec![7.0f64; 144]).unwrap();
        let out = slic(&img, &settings(&[4, 4], 10.0, 0)).unwrap();
        assert_eq!(out.average_residual, f64::MAX);
        assert_eq!(labels_of(&out), vec![0u32; 144]);
    }

    #[test]
    fn residual_of_a_converged_constant_image_is_finite() {
        let img = Image::from_vec(&[12, 12], vec![7.0f64; 144]).unwrap();
        let out = slic(&img, &settings(&[4, 4], 10.0, 3)).unwrap();
        assert!(out.average_residual.is_finite());
        assert!(out.average_residual >= 0.0);
    }

    #[test]
    fn defaults_match_simpleitk() {
        let s = SlicSettings::default();
        assert_eq!(s.super_grid_size, vec![50, 50, 50]);
        assert_eq!(s.spatial_proximity_weight, 10.0);
        assert_eq!(s.maximum_number_of_iterations, 5);
        assert!(s.enforce_connectivity);
        assert!(s.initialization_perturbation);
    }

    #[test]
    fn default_grid_of_three_entries_works_on_a_two_d_image() {
        // SimpleITK's `sitkSTLVectorToITK` ignores entries beyond the image
        // dimension.
        let img = Image::from_vec(&[8, 8], vec![1.0f64; 64]).unwrap();
        let s = SlicSettings {
            enforce_connectivity: false,
            ..SlicSettings::default()
        };
        let out = slic(&img, &s).unwrap();
        // One cluster: grid 50 > every axis, so out_size is 1 on both axes.
        assert_eq!(labels_of(&out), vec![0u32; 64]);
    }

    #[test]
    fn an_image_smaller_than_min_super_size_takes_the_unused_prev_label() {
        // 8×8 with the default grid 50 gives one cluster and
        // min_super_size = 50·50/4 = 625. The single component is undersized,
        // so ThreadedConnectivity un-marks it and the raster pass rewrites it
        // with `prev_label`, which is still at its seed value `num_clusters`
        // (= 1) — an id no cluster owns. ITK does exactly this.
        let img = Image::from_vec(&[8, 8], vec![1.0f64; 64]).unwrap();
        let out = slic(&img, &SlicSettings::default()).unwrap();
        assert_eq!(labels_of(&out), vec![1u32; 64]);
    }

    #[test]
    fn too_few_grid_entries_is_rejected() {
        let img = Image::new(&[8, 8], sitk_core::PixelId::Float64);
        assert!(matches!(
            slic(&img, &settings(&[4], 10.0, 1)),
            Err(FilterError::DimensionLength {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn zero_grid_size_is_rejected() {
        let img = Image::new(&[8, 8], sitk_core::PixelId::Float64);
        assert!(matches!(
            slic(&img, &settings(&[0, 4], 10.0, 1)),
            Err(FilterError::InvalidSuperGridSize(_))
        ));
    }

    #[test]
    fn empty_image_yields_an_empty_label_image() {
        let img = Image::from_vec(&[0, 4], Vec::<f64>::new()).unwrap();
        let out = slic(&img, &settings(&[4, 4], 10.0, 1)).unwrap();
        assert_eq!(out.labels.number_of_pixels(), 0);
        assert_eq!(out.average_residual, f64::MAX);
    }
}
