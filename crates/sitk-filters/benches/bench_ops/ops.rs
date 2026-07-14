//! The 12 ops from `doc/bench-spec.md`'s op table, each pinned to the exact
//! parameters that table specifies. One free `fn(&Image) -> Result<Image>`
//! per op (no closures) so every entry in [`OPS`] shares one function-pointer
//! type regardless of the wrapped op's own parameter list.
use sitk_core::Image;
use sitk_filters::{
    ConvolutionBoundaryCondition, OutputRegionMode, Result, StructuringElement, binary_dilate,
    connected_component, discrete_gaussian, fft_convolution, gradient_magnitude,
    gradient_magnitude_recursive_gaussian, mean, median, otsu_threshold, rescale_intensity_cpu,
    signed_maurer_distance_map, smoothing_recursive_gaussian,
};

/// `doc/bench-spec.md` §"The three input variants, and which op takes
/// which": which derived buffer of the shared synthesized volume an op
/// reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InputKind {
    /// `base_f32`: the raw synth values in `[0, 1000)`, as `Float32`. Ops 1,
    /// 2, 3, 4, 5, 6, 7, 11, 12.
    BaseF32,
    /// `mask_u8`: `base >= 500.0 ? 1 : 0`, `UInt8`. Ops 8 (`binary_dilate`)
    /// and 10 (`connected_component`).
    MaskU8,
    /// `mask_f32`: the same `>= 500.0` threshold, kept as `Float32` 0.0/1.0
    /// (binary *content* in a Float32 *type*). Op 9
    /// (`signed_maurer_distance_map`) only, with `background_value = 0.0` --
    /// a signed distance map is only meaningful on binary content, but the
    /// port's filter takes a Float32 image.
    MaskF32,
}

pub struct OpSpec {
    pub key: &'static str,
    pub input: InputKind,
    pub run: fn(&Image) -> Result<Image>,
}

pub const OPS: &[OpSpec] = &[
    OpSpec {
        key: "rescale_intensity",
        input: InputKind::BaseF32,
        run: run_rescale_intensity,
    },
    OpSpec {
        key: "smoothing_recursive_gaussian",
        input: InputKind::BaseF32,
        run: run_smoothing_recursive_gaussian,
    },
    OpSpec {
        key: "discrete_gaussian",
        input: InputKind::BaseF32,
        run: run_discrete_gaussian,
    },
    OpSpec {
        key: "median",
        input: InputKind::BaseF32,
        run: run_median,
    },
    OpSpec {
        key: "mean",
        input: InputKind::BaseF32,
        run: run_mean,
    },
    OpSpec {
        key: "gradient_magnitude",
        input: InputKind::BaseF32,
        run: run_gradient_magnitude,
    },
    OpSpec {
        key: "gradient_magnitude_recursive_gaussian",
        input: InputKind::BaseF32,
        run: run_gradient_magnitude_recursive_gaussian,
    },
    OpSpec {
        key: "binary_dilate",
        input: InputKind::MaskU8,
        run: run_binary_dilate,
    },
    OpSpec {
        key: "signed_maurer_distance_map",
        input: InputKind::MaskF32,
        run: run_signed_maurer_distance_map,
    },
    OpSpec {
        key: "connected_component",
        input: InputKind::MaskU8,
        run: run_connected_component,
    },
    OpSpec {
        key: "otsu_threshold",
        input: InputKind::BaseF32,
        run: run_otsu_threshold,
    },
    OpSpec {
        key: "fft_convolution",
        input: InputKind::BaseF32,
        run: run_fft_convolution,
    },
];

/// `doc/bench-spec.md` op 1 parameters, named so `benches/bench_ops.rs`'s GPU
/// row calls `sitk_cuda::rescale_intensity_gpu` with the exact same values
/// instead of a second, possibly-drifting literal.
pub const RESCALE_OUTPUT_MIN: f64 = 0.0;
pub const RESCALE_OUTPUT_MAX: f64 = 255.0;

fn run_rescale_intensity(img: &Image) -> Result<Image> {
    // Always the CPU path, regardless of the `cuda` feature: `t1`/`tN` and the
    // correctness-gate reference must stay comparable across builds and must
    // never silently run on GPU (`rescale_intensity`'s public dispatcher
    // tries GPU first when the feature is on; `rescale_intensity_cpu` is the
    // guaranteed-CPU entry point it falls back to).
    rescale_intensity_cpu(img, RESCALE_OUTPUT_MIN, RESCALE_OUTPUT_MAX)
}

fn run_smoothing_recursive_gaussian(img: &Image) -> Result<Image> {
    smoothing_recursive_gaussian(img, &[2.0, 2.0, 2.0], false)
}

fn run_discrete_gaussian(img: &Image) -> Result<Image> {
    discrete_gaussian(img, &[4.0, 4.0, 4.0], &[0.01, 0.01, 0.01], 32, true)
}

fn run_median(img: &Image) -> Result<Image> {
    median(img, &[2, 2, 2])
}

fn run_mean(img: &Image) -> Result<Image> {
    mean(img, &[2, 2, 2])
}

fn run_gradient_magnitude(img: &Image) -> Result<Image> {
    gradient_magnitude(img, true)
}

fn run_gradient_magnitude_recursive_gaussian(img: &Image) -> Result<Image> {
    gradient_magnitude_recursive_gaussian(img, 2.0, false)
}

fn run_binary_dilate(img: &Image) -> Result<Image> {
    let kernel = StructuringElement::ball(&[3, 3, 3]);
    binary_dilate(img, &kernel, 1.0, 0.0, false)
}

fn run_signed_maurer_distance_map(img: &Image) -> Result<Image> {
    signed_maurer_distance_map(img, false, false, true, 0.0)
}

fn run_connected_component(img: &Image) -> Result<Image> {
    connected_component(img, None, false)
}

fn run_otsu_threshold(img: &Image) -> Result<Image> {
    otsu_threshold(img, 128, false, 1, 0, None).map(|(image, _threshold)| image)
}

fn run_fft_convolution(img: &Image) -> Result<Image> {
    let kernel = Image::from_vec(&[7, 7, 7], vec![1.0f32; 343]).expect("7^3 box kernel");
    fft_convolution(
        img,
        &kernel,
        true,
        ConvolutionBoundaryCondition::default(),
        OutputRegionMode::default(),
    )
}
