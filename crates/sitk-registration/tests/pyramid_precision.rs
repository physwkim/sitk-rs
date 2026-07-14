//! The pyramid does not re-quantize an integer volume (ledger §5.12).
//!
//! `prepare_level` smooths with `recursive_gaussian`, which narrows back to its
//! *input's* pixel type. On a `UInt16` CT that rounded the smoothed intensities to
//! `UInt16` at every level — and then the resample onto the shrunk grid interpolated
//! and rounded a second time. The level pipeline now runs in floating point, so
//! registering a `UInt16` volume is registering its `Float32` cast, exactly.
//!
//! Why exactly and not merely closely: the promoted run *is* the float run. `cast`
//! from `UInt16` to `Float32` is lossless (every `u16` is a representable `f32`), and
//! after the cast the two runs execute the same code on the same buffers. Anything
//! short of bit-identical would mean the promotion is not where I think it is.

use sitk_core::{Image, PixelId};
use sitk_registration::ImageRegistrationMethod;
use sitk_transform::{Euler3DTransform, ParametricTransform};

/// A `UInt16` volume, as a CT arrives from disk: values in the thousands, where a
/// unit of quantization is ~1e-4 of the dynamic range and a smoothed intensity
/// almost never lands on an integer.
fn ct(n: usize, shift: f64) -> Image {
    let c = n as f64 / 2.0;
    let mut v = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (x, y, z) = (i as f64 - shift, j as f64, k as f64);
                let r = ((x - c).powi(2) + (y - c).powi(2) + (z - c).powi(2)).sqrt();
                let s = 2000.0 * (-(r * r) / (0.18 * n as f64).powi(2)).exp()
                    + 200.0 * (0.4 * r).sin()
                    + 400.0;
                v.push(s.clamp(0.0, 65535.0) as u16);
            }
        }
    }
    let mut img = Image::from_vec(&[n, n, n], v).unwrap();
    img.set_spacing(&[0.8, 0.9, 1.1]).unwrap();
    img
}

fn method() -> ImageRegistrationMethod {
    let mut reg = ImageRegistrationMethod::new();
    reg.set_metric_as_mean_squares();
    reg.set_optimizer_as_regular_step_gradient_descent(1.0, 1e-5, 100, 1e-8);
    reg.set_optimizer_scales_from_physical_shift();
    reg.set_shrink_factors_per_level(vec![2, 1]);
    reg.set_smoothing_sigmas_per_level(vec![1.0, 0.0]);
    reg
}

fn initial(n: usize) -> Euler3DTransform {
    let c = n as f64 / 2.0;
    Euler3DTransform::new(0.0, 0.0, 0.0, [0.0; 3], [c, c, c])
}

/// Registering a `UInt16` pair is registering its `Float32` cast — bit for bit,
/// level for level. Before the promotion the two disagreed, because the `UInt16`
/// run rounded every smoothed level back to integers.
#[test]
fn an_integer_pyramid_is_the_pyramid_of_its_float_cast() {
    let n = 32;
    let (fixed, moving) = (ct(n, 0.0), ct(n, 3.0));
    let ffix = sitk_filters::cast(&fixed, PixelId::Float32).unwrap();
    let fmov = sitk_filters::cast(&moving, PixelId::Float32).unwrap();

    let reg = method();
    let int = reg.execute(&fixed, &moving, initial(n)).unwrap();
    let flt = reg.execute(&ffix, &fmov, initial(n)).unwrap();

    assert_eq!(
        int.levels.len(),
        2,
        "the schedule has two levels and both must be reported"
    );
    for (a, b) in int.levels.iter().zip(flt.levels.iter()) {
        assert_eq!(
            a.iterations, b.iterations,
            "level {} took a different number of steps on the integer volume",
            a.level
        );
        assert_eq!(
            a.valid_points, b.valid_points,
            "level {} sampled a different number of points on the integer volume",
            a.level
        );
        assert_eq!(
            a.metric_value.to_bits(),
            b.metric_value.to_bits(),
            "level {} converged to a different metric value on the integer volume: \
             {} vs {}",
            a.level,
            a.metric_value,
            b.metric_value
        );
    }
    assert_eq!(
        int.transform.parameters(),
        flt.transform.parameters(),
        "the integer run and the float run must be the same run"
    );

    // And the registration actually did something: a run that recovered nothing
    // would also agree with itself.
    assert!(
        (int.transform.parameters()[3] - 3.0 * 0.8).abs() < 0.2,
        "expected ~2.4 mm of x translation, got {:?}",
        int.transform.parameters()
    );
}

/// The same volume, on the device. The host pyramid used to re-quantize where the
/// device pyramid (which holds `f32`) did not, so the two paths *disagreed* on
/// integer inputs by construction — logged in `execute_on_device`'s doc comment,
/// never measured. With the host promoted, they agree to the same band a `Float32`
/// input already agreed to.
#[cfg(feature = "cuda")]
#[test]
fn the_device_pyramid_agrees_with_the_host_pyramid_on_an_integer_volume() {
    use sitk_cuda::{CudaError, DeviceImage};

    if matches!(sitk_cuda::backend(), Err(CudaError::NoDevice(_))) {
        println!("SKIPPED: no CUDA device");
        return;
    }
    let n = 32;
    let (fixed, moving) = (ct(n, 0.0), ct(n, 3.0));
    let reg = method();

    let host = reg.execute(&fixed, &moving, initial(n)).unwrap();
    let device = reg
        .execute_on_device(
            &DeviceImage::upload(&fixed).unwrap(),
            &DeviceImage::upload(&moving).unwrap(),
            initial(n),
        )
        .unwrap();

    assert_eq!(host.levels.len(), device.levels.len());
    for (h, d) in host.levels.iter().zip(device.levels.iter()) {
        println!(
            "level {}: host {} iters / {} valid / {:.12} | device {} iters / {} valid / {:.12}",
            h.level,
            h.iterations,
            h.valid_points,
            h.metric_value,
            d.iterations,
            d.valid_points,
            d.metric_value
        );
        assert_eq!(
            d.iterations, h.iterations,
            "level {} took a different number of steps",
            h.level
        );
        assert_eq!(
            d.valid_points, h.valid_points,
            "level {} sampled a different number of points",
            h.level
        );
    }
    let worst = device
        .transform
        .parameters()
        .iter()
        .zip(host.transform.parameters().iter())
        .map(|(&d, &h)| (d - h).abs() / (1.0 + h.abs()))
        .fold(0.0f64, f64::max);
    println!("worst parameter disagreement on a UInt16 pyramid: {worst:e}");
    assert!(
        worst <= 1e-9,
        "the host no longer re-quantizes, so an integer pyramid must land where the \
         device pyramid lands; worst {worst:e}"
    );
}
