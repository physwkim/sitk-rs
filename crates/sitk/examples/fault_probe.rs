//! One op, one 256³ volume, one process — so `/usr/bin/time -v`'s minor-fault
//! count belongs to that op and (apart from an identical input volume, built the
//! same way in every build) to nothing else.
//!
//! ```text
//! /usr/bin/time -v ./target/release/examples/fault_probe rescale_intensity
//! ```

use std::time::Instant;

use sitk::core::Image;
use sitk::filters::{StructuringElement, binary_dilate, rescale_intensity};

fn main() {
    let op = std::env::args().nth(1).expect("op");
    let n = 256usize;
    let count = n * n * n;

    // Both inputs are built by a plain `collect` — never through the primitives
    // under test — so the setup's page faults are the same constant in every
    // build being compared.
    let t = Instant::now();
    match op.as_str() {
        "rescale_intensity" => {
            let img = Image::from_vec(&[n, n, n], (0..count).map(|i| (i % 251) as f32).collect())
                .unwrap();
            let t = Instant::now();
            let out = rescale_intensity(&img, 0.0, 255.0).unwrap();
            println!("{op}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&out);
        }
        "binary_dilate" => {
            let mask = Image::from_vec(
                &[n, n, n],
                (0..count).map(|i| u8::from(i % 251 > 128)).collect(),
            )
            .unwrap();
            let t = Instant::now();
            let out = binary_dilate(
                &mask,
                &StructuringElement::box_(&[1, 1, 1]),
                1.0,
                0.0,
                false,
            )
            .unwrap();
            println!("{op}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&out);
        }
        other => panic!("unknown op {other}"),
    }
    std::hint::black_box(t);
}
