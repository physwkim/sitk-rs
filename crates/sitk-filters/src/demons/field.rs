//! The displacement field and the separable Gaussian smoother
//! `PDEDeformableRegistrationFilter` applies to it.

use crate::denoise::gaussian_operator_kernel;

/// A displacement field: one `dim`-component vector per pixel, components
/// interleaved and pixels in first-index-fastest order — the same layout
/// `sitk_core::Image` gives a `VectorFloat64` image, so the two convert without
/// a shuffle.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Field {
    /// Length `number_of_pixels * dim`.
    pub(crate) data: Vec<f64>,
    pub(crate) size: Vec<usize>,
}

impl Field {
    /// A zero field over `size`, with `size.len()` components per pixel.
    ///
    /// This is `PDEDeformableRegistrationFilter::CopyInputToOutput`'s
    /// no-initial-field branch (itkPDEDeformableRegistrationFilter.hxx:162-179),
    /// which fills the output with the zero vector.
    pub(crate) fn zeros(size: &[usize]) -> Self {
        let n: usize = size.iter().product();
        Field {
            data: vec![0.0; n * size.len()],
            size: size.to_vec(),
        }
    }

    pub(crate) fn dimension(&self) -> usize {
        self.size.len()
    }

    pub(crate) fn number_of_pixels(&self) -> usize {
        self.size.iter().product()
    }

    /// The `dim` components of the pixel at linear offset `pixel`.
    pub(crate) fn vector_at(&self, pixel: usize) -> &[f64] {
        let dim = self.dimension();
        &self.data[pixel * dim..pixel * dim + dim]
    }

    /// Decode a linear pixel offset into a multi-index, first index fastest.
    pub(crate) fn multi_index(&self, mut pixel: usize, index: &mut [usize]) {
        for (component, &extent) in index.iter_mut().zip(&self.size) {
            *component = pixel % extent;
            pixel /= extent;
        }
    }
}

/// The Gaussian smoothing parameters shared by
/// `PDEDeformableRegistrationFilter::SmoothDisplacementField` and
/// `::SmoothUpdateField`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Smoothing {
    /// One standard deviation per axis, in **pixel** units — no spacing is
    /// applied ("The values are set with respect to pixel coordinates").
    pub(crate) standard_deviations: Vec<f64>,
    pub(crate) maximum_error: f64,
    pub(crate) maximum_kernel_width: u32,
}

/// `PDEDeformableRegistrationFilter::SmoothDisplacementField`
/// (itkPDEDeformableRegistrationFilter.hxx:256-313) and `::SmoothUpdateField`
/// (lines 315-361), which differ only in which field and which standard
/// deviations they use — both run the same sequence of `ImageDimension`
/// directional `GaussianOperator`s through a
/// `VectorNeighborhoodOperatorImageFilter`, each axis consuming the previous
/// axis's output.
///
/// The operator's variance is `sqr(standard_deviations[axis])`. Every vector
/// component is convolved independently
/// (`VectorNeighborhoodInnerProduct`), and out-of-buffer taps take the
/// `ZeroFluxNeumannBoundaryCondition` that `ConstNeighborhoodIterator` defaults
/// to (itkConstNeighborhoodIterator.h:52) — i.e. the index is clamped into the
/// buffer, repeating the edge pixel.
pub(crate) fn smooth_field(field: &mut Field, smoothing: &Smoothing) {
    let dim = field.dimension();
    let pixels = field.number_of_pixels();

    // Strides in *pixels*, first index fastest.
    let mut strides = vec![1usize; dim];
    for d in 1..dim {
        strides[d] = strides[d - 1] * field.size[d - 1];
    }

    let mut scratch = vec![0.0f64; field.data.len()];
    let mut index = vec![0usize; dim];

    for axis in 0..dim {
        // `oper.SetVariance(itk::Math::sqr(m_StandardDeviations[j]))`.
        let variance = smoothing.standard_deviations[axis] * smoothing.standard_deviations[axis];
        let kernel = gaussian_operator_kernel(
            variance,
            smoothing.maximum_error,
            smoothing.maximum_kernel_width,
        );
        let radius = (kernel.len() / 2) as isize;
        let extent = field.size[axis] as isize;
        let stride = strides[axis] as isize;

        for pixel in 0..pixels {
            field.multi_index(pixel, &mut index);
            let center = index[axis] as isize;

            for component in 0..dim {
                let mut sum = 0.0;
                for (tap, &weight) in kernel.iter().enumerate() {
                    // ZeroFluxNeumann: clamp the sample into the buffer.
                    let sampled = (center + tap as isize - radius).clamp(0, extent - 1);
                    let neighbor = pixel.wrapping_add_signed((sampled - center) * stride);
                    sum += weight * field.data[neighbor * dim + component];
                }
                scratch[pixel * dim + component] = sum;
            }
        }

        field.data.copy_from_slice(&scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zero-variance kernel is `[0, 1, 0]`, so smoothing is the identity.
    #[test]
    fn zero_standard_deviation_leaves_the_field_unchanged() {
        let mut field = Field {
            data: (0..12).map(f64::from).collect(),
            size: vec![3, 2],
        };
        let before = field.clone();
        smooth_field(
            &mut field,
            &Smoothing {
                standard_deviations: vec![0.0, 0.0],
                maximum_error: 0.1,
                maximum_kernel_width: 30,
            },
        );
        assert_eq!(field, before);
    }

    /// The kernel is normalised, and the boundary condition repeats the edge,
    /// so a constant field is a fixed point of the smoother — including at the
    /// border, where a zero-padding boundary would pull the value down.
    #[test]
    fn a_constant_field_is_unchanged_including_at_the_border() {
        let mut field = Field {
            data: vec![7.0; 5 * 4 * 2],
            size: vec![5, 4],
        };
        smooth_field(
            &mut field,
            &Smoothing {
                standard_deviations: vec![1.5, 1.5],
                maximum_error: 0.1,
                maximum_kernel_width: 30,
            },
        );
        for &v in &field.data {
            assert!((v - 7.0).abs() < 1e-12, "got {v}");
        }
    }

    /// The smoother is a convex combination of clamped samples, so it never
    /// leaves the input's range; and it genuinely diffuses — a central impulse
    /// loses height to its neighbours.
    #[test]
    fn smoothing_spreads_an_impulse_without_leaving_the_input_range() {
        // 3x3, two components; the impulse is component 0 of the centre pixel.
        let mut data = vec![0.0; 9 * 2];
        let center = 4; // (x=1, y=1)
        data[center * 2] = 1.0;
        let mut field = Field {
            data,
            size: vec![3, 3],
        };
        smooth_field(
            &mut field,
            &Smoothing {
                standard_deviations: vec![1.0, 1.0],
                maximum_error: 0.1,
                maximum_kernel_width: 30,
            },
        );
        for &v in &field.data {
            assert!((0.0..=1.0).contains(&v), "got {v}");
        }
        assert!(field.data[center * 2] < 1.0);
        let left = 3; // (x=0, y=1)
        assert!(field.data[left * 2] > 0.0);
        // The untouched component stays zero.
        for pixel in 0..9 {
            assert_eq!(field.data[pixel * 2 + 1], 0.0);
        }
    }

    /// Each vector component is convolved independently: a field whose first
    /// component varies and whose second is constant keeps the second constant.
    #[test]
    fn components_are_smoothed_independently() {
        let mut data = Vec::new();
        for x in 0..5 {
            data.push(f64::from(x));
            data.push(3.0);
        }
        let mut field = Field {
            data,
            size: vec![5, 1],
        };
        smooth_field(
            &mut field,
            &Smoothing {
                standard_deviations: vec![1.0, 1.0],
                maximum_error: 0.1,
                maximum_kernel_width: 30,
            },
        );
        for pixel in 0..5 {
            assert!((field.data[pixel * 2 + 1] - 3.0).abs() < 1e-12);
        }
    }

    /// Separability: smoothing a rank-one field `f(x) g(y)` along both axes
    /// equals the product of the 1-D smoothings. Checked against an
    /// independently computed 1-D convolution with the same kernel and clamp.
    #[test]
    fn separable_smoothing_matches_two_one_dimensional_convolutions() {
        let (nx, ny) = (5usize, 4usize);
        let fx: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let gy: Vec<f64> = vec![2.0, 1.0, 4.0, 3.0];

        let mut data = Vec::new();
        for &g in &gy {
            for &f in &fx {
                data.push(f * g);
                data.push(0.0);
            }
        }
        let mut field = Field {
            data,
            size: vec![nx, ny],
        };
        let smoothing = Smoothing {
            standard_deviations: vec![1.0, 2.0],
            maximum_error: 0.1,
            maximum_kernel_width: 30,
        };
        smooth_field(&mut field, &smoothing);

        let convolve = |signal: &[f64], variance: f64| -> Vec<f64> {
            let kernel = gaussian_operator_kernel(variance, 0.1, 30);
            let radius = (kernel.len() / 2) as i64;
            let n = signal.len() as i64;
            (0..signal.len())
                .map(|i| {
                    kernel
                        .iter()
                        .enumerate()
                        .map(|(tap, &w)| {
                            let s = (i as i64 + tap as i64 - radius).clamp(0, n - 1);
                            w * signal[s as usize]
                        })
                        .sum()
                })
                .collect()
        };
        let sx = convolve(&fx, 1.0);
        let sy = convolve(&gy, 4.0);

        for (y, &sy_y) in sy.iter().enumerate() {
            for (x, &sx_x) in sx.iter().enumerate() {
                let got = field.data[(y * nx + x) * 2];
                let want = sx_x * sy_y;
                assert!((got - want).abs() < 1e-12, "at ({x},{y}): {got} vs {want}");
            }
        }
    }
}
