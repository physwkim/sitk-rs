//! End-to-end Phase-0 acceptance test: the full read → filter → resample → write
//! pipeline, proving the core model, IO, filters, and resampling compose.

use sitk::filters;
use sitk::io::{read_image, write_image};
use sitk::transform::{AffineTransform, Interpolator, ResampleImageFilter};
use sitk::{Image, PixelId};

fn tmp(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sitk_pipeline_{}_{name}", std::process::id()));
    p
}

#[test]
fn read_filter_resample_write_roundtrip() {
    // 1. Build a 4x3 UInt8 image with non-trivial geometry and write it.
    let data: Vec<u8> = (0..12).map(|i| (i * 10) as u8).collect();
    let mut src = Image::from_vec(&[4, 3], data).unwrap();
    src.set_spacing(&[2.0, 0.5]).unwrap();
    src.set_origin(&[-1.0, 3.0]).unwrap();
    let src_path = tmp("src.mha");
    write_image(&src, &src_path).unwrap();

    // 2. Read it back and confirm geometry survived the file.
    let img = read_image(&src_path).unwrap();
    assert_eq!(img.spacing(), &[2.0, 0.5]);
    assert_eq!(img.origin(), &[-1.0, 3.0]);
    assert_eq!(img.pixel_id(), PixelId::UInt8);

    // 3. Cast to f32, scale, add a constant.
    let f = filters::cast(&img, PixelId::Float32).unwrap();
    let scaled = filters::multiply_constant(&f, 0.5).unwrap();
    let biased = filters::add_constant(&scaled, 1.0).unwrap();
    assert_eq!(biased.pixel_id(), PixelId::Float32);
    // pixel 0: (0*0.5)+1 = 1 ; pixel 11: (110*0.5)+1 = 56
    let vals = biased.scalar_slice::<f32>().unwrap();
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[11], 56.0);

    // 4. Resample under the identity transform onto the same grid: unchanged.
    let t = AffineTransform::identity(2);
    let resampled = ResampleImageFilter::new()
        .set_reference_image(&biased)
        .set_interpolator(Interpolator::Linear)
        .execute(&biased, &t)
        .unwrap();
    assert_eq!(
        resampled.scalar_slice::<f32>().unwrap(),
        biased.scalar_slice::<f32>().unwrap()
    );
    assert_eq!(resampled.spacing(), biased.spacing());
    assert_eq!(resampled.origin(), biased.origin());

    // 5. Threshold to a UInt8 mask and write it out, then read and verify.
    let mask = filters::binary_threshold(&resampled, 10.0, 40.0, 1, 0).unwrap();
    let mask_path = tmp("mask.mha");
    write_image(&mask, &mask_path).unwrap();
    let mut mask_back = read_image(&mask_path).unwrap();

    assert_eq!(mask_back.pixel_id(), PixelId::UInt8);
    // A read installs the ImageIO's meta-data dictionary
    // (itkMetaImageIO.cxx:270-278) plus the reader's geometry-normalization
    // records (`ITK_original_*`, itkImageFileReader.hxx:216-239); a filter
    // output has none. Compare the rest.
    assert_eq!(
        mask_back.meta_data_keys(),
        vec![
            "ITK_InputFilterName",
            "ITK_original_direction",
            "ITK_original_spacing",
            "Modality",
        ]
    );
    // The mask's spacing is positive, so nothing flipped; the records are the
    // raw grid the reader saw.
    assert_eq!(mask_back.meta_data("ITK_original_spacing"), Some("2 0.5"));
    assert_eq!(
        mask_back.meta_data("ITK_original_direction"),
        Some("1 0 0 1")
    );
    for key in ["ITK_InputFilterName", "Modality"] {
        mask_back.erase_meta_data(key);
    }
    mask_back.erase_meta_data("ITK_original_direction");
    mask_back.erase_meta_data("ITK_original_spacing");
    assert_eq!(mask_back, mask);
    // Values in [10,40]: biased pixels 21..56 step... check a couple.
    // biased[d] = d*5 + 1 for d in 0..12 -> 1,6,11,16,21,26,31,36,41,46,51,56
    // inside [10,40]: indices 2..7 (11,16,21,26,31,36) -> 1, else 0.
    let expected: Vec<u8> = (0..12u8)
        .map(|d| {
            let v = d as f32 * 5.0 + 1.0;
            if (10.0..=40.0).contains(&v) { 1 } else { 0 }
        })
        .collect();
    assert_eq!(mask_back.scalar_slice::<u8>().unwrap(), expected.as_slice());

    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&mask_path).ok();
}
