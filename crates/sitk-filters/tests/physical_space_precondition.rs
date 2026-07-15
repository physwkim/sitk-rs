//! The physical-space precondition, both directions, per audited group.
//!
//! ITK's `ImageToImageFilter::VerifyInputInformation` throws "Inputs do not
//! occupy the same physical space!" when an extra input's origin, spacing or
//! direction differs from the primary's. This port routes every filter that
//! *inherits* that check through one owner (`geometry::require_same_physical_space`,
//! surfaced as `FilterError::PhysicalSpaceMismatch`). The classification of who
//! inherits it — and, crucially, who ITK *exempts* — is `doc/physical-space-audit.md`.
//!
//! Every group below pins **both** directions, because a suite that only proves
//! the verifiers throw would pass a patch that routed everything, silently
//! breaking the filters ITK does not check:
//!
//! * a VERIFIES filter **refuses** a shifted-origin second input, and
//! * an EXEMPT / NOT-AN-INPUT filter in reach **still accepts** one.
//!
//! The accepting-direction pins are the load-bearing ones; they are the guard
//! against a global fold.

use sitk_core::Image;
use sitk_filters::FilterError;

/// A 4×4 `f64` ramp, on the default identity grid (origin 0, spacing 1).
fn base() -> Image {
    Image::from_vec(&[4, 4], (0..16).map(|k| k as f64).collect()).unwrap()
}

/// The same buffer, moved to a different physical grid by an origin shift — the
/// cheapest mismatch `VerifyInputInformation` rejects.
fn shifted_origin() -> Image {
    let mut img = base();
    img.set_origin(&[10.0, 0.0]).unwrap();
    img
}

/// A `u8` copy of `base`'s grid, for filters that want an integer second input.
fn base_u8() -> Image {
    Image::from_vec(&[4, 4], (0..16u8).collect()).unwrap()
}

fn shifted_u8() -> Image {
    let mut img = base_u8();
    img.set_origin(&[10.0, 0.0]).unwrap();
    img
}

fn is_space_mismatch<T>(r: Result<T, FilterError>) -> bool {
    matches!(r, Err(FilterError::PhysicalSpaceMismatch { index: 1 }))
}

/// Index-aware variant: filters with more than a second input report *which*
/// input was off-grid (ternary's `c` is input 2; an nary's k-th extra input is
/// input k). The reported index is part of the contract, so the pins assert it.
fn is_space_mismatch_at<T>(r: Result<T, FilterError>, want: usize) -> bool {
    matches!(r, Err(FilterError::PhysicalSpaceMismatch { index }) if index == want)
}

// ===========================================================================
// Group 1 — the functor family (binary_apply / binary_apply_in_place /
// comparison_apply engines): arithmetic, logic, comparisons, mask.
// ===========================================================================

mod group1_functors {
    use super::*;
    use sitk_filters::{add, and, equal, greater, mask, modulus, multiply, subtract};

    #[test]
    fn arithmetic_refuses_a_shifted_second_operand() {
        // Same buffer and size; only the grid differs. On one grid these add
        // fine; shifted, ITK throws and so does the port.
        assert!(add(&base(), &base()).is_ok());
        assert!(is_space_mismatch(add(&base(), &shifted_origin())));
        assert!(is_space_mismatch(subtract(&base(), &shifted_origin())));
        assert!(is_space_mismatch(multiply(&base(), &shifted_origin())));
    }

    #[test]
    fn modulus_refuses_a_shifted_second_operand() {
        let a = Image::from_vec(&[4, 4], (1..17i32).collect()).unwrap();
        let mut b = Image::from_vec(&[4, 4], vec![3i32; 16]).unwrap();
        assert!(modulus(&a, &b).is_ok());
        b.set_origin(&[10.0, 0.0]).unwrap();
        assert!(is_space_mismatch(modulus(&a, &b)));
    }

    #[test]
    fn logic_refuses_a_shifted_second_operand() {
        assert!(and(&base_u8(), &base_u8()).is_ok());
        assert!(is_space_mismatch(and(&base_u8(), &shifted_u8())));
    }

    #[test]
    fn comparison_refuses_a_shifted_second_operand() {
        assert!(equal(&base(), &base(), 1, 0).is_ok());
        assert!(is_space_mismatch(equal(&base(), &shifted_origin(), 1, 0)));
        assert!(is_space_mismatch(greater(&base(), &shifted_origin(), 1, 0)));
    }

    #[test]
    fn mask_refuses_a_shifted_mask() {
        // `mask` here is the MaskImageFilter functor (logic.rs), a binary
        // functor — distinct from the uint8 mask *input* filters. It routes
        // through the same engine.
        // The MaskImageFilter functor requires the two operands to share a
        // pixel type (`require_same_shape`), so both are f64 here.
        assert!(mask(&base(), &base(), 0.0, 0.0).is_ok());
        assert!(is_space_mismatch(mask(
            &base(),
            &shifted_origin(),
            0.0,
            0.0
        )));
    }

    /// The accepting direction: `histogram_matching` inherits nothing — ITK's
    /// `HistogramMatchingImageFilter` overrides `VerifyInputInformation` to an
    /// empty body (`itkHistogramMatchingImageFilter.h:202-209`), so a reference
    /// image on a *different* grid is accepted, not rejected. If the check had
    /// been folded somewhere global, this would start throwing.
    #[test]
    fn exempt_histogram_matching_still_accepts_a_shifted_reference() {
        let reference = shifted_origin();
        let out = sitk_filters::histogram_matching(&base(), &reference, 64, 8, false);
        assert!(
            out.is_ok(),
            "histogram_matching is EXEMPT and must accept any grid"
        );
    }
}

// ===========================================================================
// Group 2 — the f64-compute math family (math.rs): the binary
// `two_image_f64*` helpers, the `nary_reduce_f64` reduction, and the
// `three_image_f64*` ternary helpers. Every entry point routes through one of
// these five shared helpers, so pinning one representative per helper pins the
// whole family; the ternary and nary pins additionally assert the reported
// input index (2 for ternary's `c`, k for an nary's k-th extra input).
// ===========================================================================

mod group2_math {
    use super::*;
    use sitk_filters::{
        atan2, divide_real, nary_add, paste, squared_difference, squared_difference_in_place,
        ternary_add, ternary_add_in_place,
    };

    #[test]
    fn binary_refuses_a_shifted_second_operand() {
        // Route: `two_image_f64_with_output`. All seven binary entry points and
        // `divide_real` share it, so one shifted-operand pin per distinct
        // functor covers the seam.
        assert!(squared_difference(&base(), &base()).is_ok());
        assert!(is_space_mismatch(squared_difference(
            &base(),
            &shifted_origin()
        )));
        assert!(is_space_mismatch(atan2(&base(), &shifted_origin())));
        assert!(is_space_mismatch(divide_real(&base(), &shifted_origin())));
    }

    #[test]
    fn binary_in_place_refuses_a_shifted_second_operand() {
        // Route: `two_image_f64_in_place`. The in-place variant consumes `a`,
        // so it gets its own fresh buffers.
        assert!(squared_difference_in_place(base(), &base()).is_ok());
        assert!(is_space_mismatch(squared_difference_in_place(
            base(),
            &shifted_origin()
        )));
    }

    #[test]
    fn nary_refuses_a_shifted_input_and_names_its_index() {
        // Route: `require_inputs`. The reduction reports which extra input is
        // off-grid: the k-th input past the first is index k.
        assert!(nary_add(&[&base(), &base(), &base()]).is_ok());
        assert!(is_space_mismatch_at(
            nary_add(&[&base(), &shifted_origin(), &base()]),
            1
        ));
        assert!(is_space_mismatch_at(
            nary_add(&[&base(), &base(), &shifted_origin()]),
            2
        ));
    }

    #[test]
    fn ternary_refuses_either_shifted_operand_by_index() {
        // Route: `three_image_f64`. `b` is input 1, `c` is input 2 — the
        // index-2 pin is the one Group 1 could not reach.
        assert!(ternary_add(&base(), &base(), &base()).is_ok());
        assert!(is_space_mismatch_at(
            ternary_add(&base(), &shifted_origin(), &base()),
            1
        ));
        assert!(is_space_mismatch_at(
            ternary_add(&base(), &base(), &shifted_origin()),
            2
        ));
    }

    #[test]
    fn ternary_in_place_refuses_either_shifted_operand_by_index() {
        // Route: `three_image_f64_in_place`.
        assert!(ternary_add_in_place(base(), &base(), &base()).is_ok());
        assert!(is_space_mismatch_at(
            ternary_add_in_place(base(), &shifted_origin(), &base()),
            1
        ));
        assert!(is_space_mismatch_at(
            ternary_add_in_place(base(), &base(), &shifted_origin()),
            2
        ));
    }

    /// The accepting direction: `PasteImageFilter` is EXEMPT — it composites by
    /// region index, not physical space, and ITK never verifies the source and
    /// destination share a grid. A source shifted off the destination's grid
    /// must still paste. A global fold would break exactly this.
    #[test]
    fn exempt_paste_still_accepts_a_shifted_source() {
        let out = paste(&base_u8(), &shifted_u8(), &[0, 0], &[2, 2], &[0, 0]);
        assert!(out.is_ok(), "paste is EXEMPT and must accept any grid");
    }
}

// ===========================================================================
// Group 3 — the ComposeImageFilter family: complex compose
// (real_and_imaginary_to_complex / magnitude_and_phase_to_complex, routed
// through complex.rs's `complex_binary`) and vector `compose` (an &[&Image]
// slice, routed in vector.rs). All are `ComposeImageFilter : ImageToImageFilter`
// with no override, so every indexed input is verified.
// ===========================================================================

mod group3_compose {
    use super::*;
    use sitk_filters::complex::real_and_imaginary_to_complex;
    use sitk_filters::{ConvolutionBoundaryCondition, OutputRegionMode, compose, convolution};

    #[test]
    fn complex_compose_refuses_a_shifted_second_operand() {
        // Route: complex.rs `complex_binary`. Both operands must be real-typed;
        // base()/shifted_origin() are f64.
        assert!(real_and_imaginary_to_complex(&base(), &base()).is_ok());
        assert!(is_space_mismatch(real_and_imaginary_to_complex(
            &base(),
            &shifted_origin()
        )));
    }

    #[test]
    fn vector_compose_refuses_a_shifted_component_by_index() {
        // Route: vector.rs `compose`. Each component past the first is an
        // indexed input; the reported index is its slice position.
        assert!(compose(&[&base(), &base(), &base()]).is_ok());
        assert!(is_space_mismatch_at(
            compose(&[&base(), &shifted_origin(), &base()]),
            1
        ));
        assert!(is_space_mismatch_at(
            compose(&[&base(), &base(), &shifted_origin()]),
            2
        ));
    }

    /// The accepting direction: `ConvolutionImageFilter` is EXEMPT — its base
    /// overrides `VerifyInputInformation` to an empty body because the kernel is
    /// a stencil, not a co-registered image (`itkConvolutionImageFilterBase.h:161-162`).
    /// A kernel on a different grid must still convolve.
    #[test]
    fn exempt_convolution_still_accepts_a_shifted_kernel() {
        let mut kernel = Image::from_vec(&[3, 3], vec![1.0f64 / 9.0; 9]).unwrap();
        kernel.set_origin(&[10.0, 0.0]).unwrap();
        let out = convolution(
            &base(),
            &kernel,
            false,
            ConvolutionBoundaryCondition::default(),
            OutputRegionMode::default(),
        );
        assert!(
            out.is_ok(),
            "convolution is EXEMPT and must accept any kernel grid"
        );
    }
}

// ===========================================================================
// Group 4 — grayscale/binary morphological reconstruction and geodesic
// morphology (marker = input 0, mask = input 1). All derive from
// ImageToImageFilter with no override; both inputs are itkSetInput'd, so the
// inherited verifier compares them.
// ===========================================================================

mod group4_reconstruction {
    use super::*;
    use sitk_filters::{
        binary_reconstruction_by_dilation, grayscale_geodesic_dilate, reconstruction_by_dilation,
        tile,
    };

    #[test]
    fn grayscale_reconstruction_refuses_a_shifted_mask() {
        // Route: reconstruction.rs `reconstruct_images`. marker == mask, so the
        // marker<=mask precondition holds; the grid check fires first when the
        // mask is shifted.
        assert!(reconstruction_by_dilation(&base(), &base(), false).is_ok());
        assert!(is_space_mismatch(reconstruction_by_dilation(
            &base(),
            &shifted_origin(),
            false
        )));
    }

    #[test]
    fn geodesic_morphology_refuses_a_shifted_mask() {
        // Route: geodesic_morphology.rs (two sites, one per direction).
        assert!(grayscale_geodesic_dilate(&base(), &base(), false, false).is_ok());
        assert!(is_space_mismatch(grayscale_geodesic_dilate(
            &base(),
            &shifted_origin(),
            false,
            false
        )));
    }

    #[test]
    fn binary_reconstruction_refuses_a_shifted_mask() {
        // Route: morphology_reconstruction.rs. Values need not be binary for the
        // grid check, which fires before component labelling.
        assert!(binary_reconstruction_by_dilation(&base(), &base(), 15.0, 0.0, false).is_ok());
        assert!(is_space_mismatch(binary_reconstruction_by_dilation(
            &base(),
            &shifted_origin(),
            15.0,
            0.0,
            false
        )));
    }

    /// The accepting direction: `TileImageFilter` is EXEMPT — its override
    /// deliberately does not call the superclass (`itkTileImageFilter.hxx:357-364`,
    /// *"the images are going to be tiled ... which likely has no physical
    /// location meaning"*). It is the slice filter a "route every N-ary filter"
    /// shortcut would have wrongly checked, so this pin is the sharpest guard in
    /// the suite against that fold.
    #[test]
    fn exempt_tile_still_accepts_a_shifted_input() {
        let out = tile(&[&base(), &shifted_origin()], &[2, 1], 0.0);
        assert!(out.is_ok(), "tile is EXEMPT and must accept any grid");
    }
}

// ===========================================================================
// Group 5 — labels and statistics: LabelStatistics (intensity=input 0,
// label=input 1), LabelIntensityStatistics (label=input 0, feature=input 1),
// MorphologicalWatershedFromMarkers (image=input 0, marker=input 1) and
// LabelOverlay (image=input 0, label=input 1). All inherit the verifier
// (LabelStatistics/LabelOverlap via ImageSink's identical comparison).
// ===========================================================================

mod group5_labels {
    use super::*;
    use sitk_filters::deconvolution::inverse_deconvolution;
    use sitk_filters::label_intensity::{
        LabelIntensityStatisticsSettings, label_intensity_statistics,
    };
    use sitk_filters::{
        ConvolutionBoundaryCondition, OutputRegionMode, label_overlay, label_statistics,
        morphological_watershed_from_markers,
    };

    #[test]
    fn label_statistics_refuses_a_shifted_label_image() {
        assert!(label_statistics(&base(), &base()).is_ok());
        assert!(is_space_mismatch(label_statistics(
            &base(),
            &shifted_origin()
        )));
    }

    #[test]
    fn label_intensity_statistics_refuses_a_shifted_feature_image() {
        // The label image must be integer-typed; the feature may be anything.
        let settings = LabelIntensityStatisticsSettings::default();
        assert!(label_intensity_statistics(&base_u8(), &base(), &settings).is_ok());
        assert!(is_space_mismatch(label_intensity_statistics(
            &base_u8(),
            &shifted_origin(),
            &settings
        )));
    }

    #[test]
    fn watershed_from_markers_refuses_a_shifted_marker() {
        assert!(morphological_watershed_from_markers(&base(), &base(), false, false).is_ok());
        assert!(is_space_mismatch(morphological_watershed_from_markers(
            &base(),
            &shifted_origin(),
            false,
            false
        )));
    }

    #[test]
    fn label_overlay_refuses_a_shifted_label_image() {
        // The base must be non-float; both inputs are u8 here.
        let colormap = [255u8, 0, 0, 0, 255, 0];
        assert!(label_overlay(&base_u8(), &base_u8(), 0.5, 0.0, &colormap).is_ok());
        assert!(is_space_mismatch(label_overlay(
            &base_u8(),
            &shifted_u8(),
            0.5,
            0.0,
            &colormap
        )));
    }

    /// The accepting direction: `InverseDeconvolutionImageFilter` is EXEMPT via
    /// the shared `ConvolutionImageFilterBase` empty override — the kernel is a
    /// PSF stencil, not a co-registered image. A shifted kernel must still
    /// deconvolve.
    #[test]
    fn exempt_deconvolution_still_accepts_a_shifted_kernel() {
        let mut kernel = Image::from_vec(&[3, 3], vec![1.0f64 / 9.0; 9]).unwrap();
        kernel.set_origin(&[10.0, 0.0]).unwrap();
        let out = inverse_deconvolution(
            &base(),
            &kernel,
            1e-6,
            false,
            ConvolutionBoundaryCondition::default(),
            OutputRegionMode::default(),
        );
        assert!(
            out.is_ok(),
            "inverse_deconvolution is EXEMPT and must accept any kernel grid"
        );
    }
}

// ===========================================================================
// Group 6 — the N-ary label-fusion filters (staple / label_voting /
// multi_label_staple), all &[&Image] slices routed through label_fusion.rs's
// `require_inputs`. Each is `: ImageToImageFilter` with no override and walks
// GetInput(i) over every indexed input.
// ===========================================================================

mod group6_label_fusion {
    use super::*;
    use sitk_filters::label_fusion::{label_voting, multi_label_staple, staple};
    use sitk_filters::tile;

    #[test]
    fn label_voting_refuses_a_shifted_input_by_index() {
        // Route: `require_inputs`. Inputs are unsigned-integer label maps.
        assert!(label_voting(&[&base_u8(), &base_u8(), &base_u8()], None).is_ok());
        assert!(is_space_mismatch_at(
            label_voting(&[&base_u8(), &shifted_u8(), &base_u8()], None),
            1
        ));
        assert!(is_space_mismatch_at(
            label_voting(&[&base_u8(), &base_u8(), &shifted_u8()], None),
            2
        ));
    }

    #[test]
    fn staple_refuses_a_shifted_input() {
        // Same shared helper; the grid check fires before the EM loop.
        assert!(is_space_mismatch(staple(
            &[&base_u8(), &shifted_u8()],
            1.0,
            10,
            1.0
        )));
    }

    #[test]
    fn multi_label_staple_refuses_a_shifted_input() {
        assert!(is_space_mismatch(multi_label_staple(
            &[&base_u8(), &shifted_u8()],
            None,
            1e-5,
            None,
            None,
        )));
    }

    /// The accepting direction: `TileImageFilter` is the sibling *slice* filter
    /// that is EXEMPT while these three VERIFY — the exact discrimination this
    /// group turns on. A "route every &[&Image] filter" fold would break it.
    #[test]
    fn exempt_tile_still_accepts_a_shifted_slice_input() {
        let out = tile(&[&base_u8(), &shifted_u8()], &[2, 1], 0.0);
        assert!(out.is_ok(), "tile is EXEMPT and must accept any grid");
    }
}

// ===========================================================================
// Group 7 — the overlap/distance measures (label_overlap_measures,
// directed_hausdorff_distance, hausdorff_distance, similarity_index). Both
// segmentations are pipeline inputs; the inherited verifier (ImageSink's for
// label_overlap_measures) compares them. hausdorff_distance is covered
// transitively — it calls directed_hausdorff_distance in both orderings.
// ===========================================================================

mod group7_overlap {
    use super::*;
    use sitk_filters::{
        directed_hausdorff_distance, hausdorff_distance, label_overlap_measures, paste,
        similarity_index,
    };

    #[test]
    fn label_overlap_measures_refuses_a_shifted_second_input() {
        // Integer-typed segmentations; u8 here.
        assert!(label_overlap_measures(&base_u8(), &base_u8()).is_ok());
        assert!(is_space_mismatch(label_overlap_measures(
            &base_u8(),
            &shifted_u8()
        )));
    }

    #[test]
    fn directed_hausdorff_distance_refuses_a_shifted_second_input() {
        assert!(directed_hausdorff_distance(&base(), &base()).is_ok());
        assert!(is_space_mismatch(directed_hausdorff_distance(
            &base(),
            &shifted_origin()
        )));
    }

    #[test]
    fn hausdorff_distance_refuses_a_shifted_second_input() {
        // Transitive: the first directed_hausdorff_distance(image1, image2) call
        // fires the check.
        assert!(hausdorff_distance(&base(), &base()).is_ok());
        assert!(is_space_mismatch(hausdorff_distance(
            &base(),
            &shifted_origin()
        )));
    }

    #[test]
    fn similarity_index_refuses_a_shifted_second_input() {
        assert!(similarity_index(&base(), &base()).is_ok());
        assert!(is_space_mismatch(similarity_index(
            &base(),
            &shifted_origin()
        )));
    }

    /// The accepting direction: `PasteImageFilter` is EXEMPT (region-index
    /// composite, no physical-space meaning), a global-fold sentinel for this
    /// group whose own files hold no exempt filter.
    #[test]
    fn exempt_paste_still_accepts_a_shifted_source() {
        let out = paste(&base_u8(), &shifted_u8(), &[0, 0], &[2, 2], &[0, 0]);
        assert!(out.is_ok(), "paste is EXEMPT and must accept any grid");
    }
}

// ===========================================================================
// Group 8 — the segmentation level sets. The five
// SegmentationLevelSetImageFilter derivatives register the feature image as a
// *named* ProcessObject input, walked by the same iterator the verifier uses,
// so its grid IS compared (all route through level_set's check_same_size).
// scalar_chan_and_vese is the NOT-AN-INPUT counter-case: its initial level set
// is deep-copied through SetLevelSet, never registered as an input, so the one
// remaining input (the feature) has nothing to compare against and any grid
// pairing is accepted.
// ===========================================================================

mod group8_level_sets {
    use super::*;
    use sitk_filters::{
        ChanAndVeseParams, geodesic_active_contour_level_set, scalar_chan_and_vese_dense_level_set,
        threshold_segmentation_level_set,
    };

    #[test]
    fn geodesic_active_contour_refuses_a_shifted_feature() {
        // Route: check_same_size. advection_scaling = 0 keeps the aligned run
        // cheap; one iteration suffices to reach a result.
        assert!(
            geodesic_active_contour_level_set(&base(), &base(), 0.02, 1.0, 1.0, 0.0, 1, false)
                .is_ok()
        );
        assert!(is_space_mismatch(geodesic_active_contour_level_set(
            &base(),
            &shifted_origin(),
            0.02,
            1.0,
            1.0,
            0.0,
            1,
            false
        )));
    }

    #[test]
    fn threshold_segmentation_refuses_a_shifted_feature() {
        // A second entry point through the same shared check_same_size helper.
        assert!(is_space_mismatch(threshold_segmentation_level_set(
            &base(),
            &shifted_origin(),
            0.0,
            1.0,
            0.02,
            1.0,
            1.0,
            1,
            false
        )));
    }

    /// The accepting direction — the NOT-AN-INPUT case that the confirmations
    /// singled out: `ScalarChanAndVeseDenseLevelSetImageFilter` deep-copies the
    /// initial level set through `SetLevelSet`
    /// (`itkMultiphaseDenseFiniteDifferenceImageFilter.h:285-296`); only the
    /// feature is an input, so there is nothing for the verifier to compare and
    /// a feature on a different grid is accepted. The port must add NO check
    /// here — this pin fails the instant one is folded in.
    #[test]
    fn not_an_input_chan_vese_accepts_a_shifted_feature() {
        let params = ChanAndVeseParams::default();
        let out = scalar_chan_and_vese_dense_level_set(&base(), &shifted_origin(), &params);
        assert!(
            out.is_ok(),
            "scalar_chan_and_vese is NOT-AN-INPUT and must accept any grid pairing"
        );
    }
}

// ===========================================================================
// Group 9 — the demons split. PDEDeformableRegistrationFilter takes fixed,
// moving and initial field through itkSetInputMacro
// (itkPDEDeformableRegistrationFilter.h:119,125,131), so its four siblings —
// SymmetricForcesDemons, FastSymmetricForcesDemons, DiffeomorphicDemons,
// LevelSetMotion — inherit the base verifier and compare fixed vs moving.
// DemonsRegistrationFilter alone overrides it away
// (itkDemonsRegistrationFilter.h:148-149) and accepts a mismatched pair. Not a
// rule to rationalize — reproduced as an upstream inconsistency. The split is
// encoded structurally: the four siblings call validate_verifying_image_pair,
// demons_registration calls the plain validate_image_pair.
// ===========================================================================

mod group9_demons {
    use super::*;
    use sitk_filters::{
        DemonsParams, DiffeomorphicDemonsParams, FastSymmetricForcesDemonsParams,
        LevelSetMotionParams, SymmetricForcesDemonsParams, demons_registration,
        diffeomorphic_demons_registration, fast_symmetric_forces_demons_registration,
        level_set_motion_registration, symmetric_forces_demons_registration,
    };

    #[test]
    fn the_four_siblings_refuse_a_shifted_moving_image() {
        // The grid check fires in validate_verifying_image_pair before any
        // iteration, so the default (large) iteration counts never run.
        assert!(is_space_mismatch(symmetric_forces_demons_registration(
            &base(),
            &shifted_origin(),
            None,
            &SymmetricForcesDemonsParams::default()
        )));
        assert!(is_space_mismatch(
            fast_symmetric_forces_demons_registration(
                &base(),
                &shifted_origin(),
                None,
                &FastSymmetricForcesDemonsParams::default()
            )
        ));
        assert!(is_space_mismatch(diffeomorphic_demons_registration(
            &base(),
            &shifted_origin(),
            None,
            &DiffeomorphicDemonsParams::default()
        )));
        assert!(is_space_mismatch(level_set_motion_registration(
            &base(),
            &shifted_origin(),
            None,
            &LevelSetMotionParams::default()
        )));
    }

    #[test]
    fn a_sibling_still_accepts_an_aligned_pair() {
        // Same grid ⇒ the check passes and the registration runs. One iteration.
        let params = SymmetricForcesDemonsParams {
            number_of_iterations: 1,
            ..Default::default()
        };
        assert!(symmetric_forces_demons_registration(&base(), &base(), None, &params).is_ok());
    }

    /// The EXEMPT half of the split: `DemonsRegistrationFilter` overrides the
    /// verifier away, so a moving image on a *different* grid is accepted and
    /// the registration runs. This pin is the anti-fold guard for the split — if
    /// demons_registration were switched to the verifying helper it would start
    /// throwing here.
    #[test]
    fn exempt_demons_registration_accepts_a_shifted_moving_image() {
        let params = DemonsParams {
            number_of_iterations: 1,
            ..Default::default()
        };
        let out = demons_registration(&base(), &shifted_origin(), None, &params);
        assert!(
            out.is_ok(),
            "demons_registration is EXEMPT and must accept a mismatched fixed/moving pair"
        );
    }
}

// ===========================================================================
// Group 10 — grids, displacement fields, and the FFT correlations.
// checker_board (image2 = input 1) and invert_displacement_field
// (InverseFieldInitialEstimate = input 1) inherit the plain verifier.
// masked_fft/fft_normalized_correlation are VERIFIES-AND-ADDS: the superclass
// compares every input (moving and both masks) against the fixed image's grid,
// and the subclass additionally requires each mask to match its image's size —
// which this group EXTENDS, not replaces.
// ===========================================================================

mod group10_grids_fft {
    use super::*;
    use sitk_filters::{
        InvertDisplacementFieldSettings, checker_board, fft_normalized_correlation,
        invert_displacement_field, masked_fft_normalized_correlation,
    };

    fn field() -> Image {
        Image::from_vec_vector(&[4, 4], 2, vec![0.0f64; 32]).unwrap()
    }

    fn shifted_field() -> Image {
        let mut f = field();
        f.set_origin(&[10.0, 0.0]).unwrap();
        f
    }

    #[test]
    fn checker_board_refuses_a_shifted_second_image() {
        assert!(checker_board(&base(), &base(), &[2, 2]).is_ok());
        assert!(is_space_mismatch(checker_board(
            &base(),
            &shifted_origin(),
            &[2, 2]
        )));
    }

    #[test]
    fn invert_displacement_field_refuses_a_shifted_estimate() {
        let settings = InvertDisplacementFieldSettings::default();
        assert!(invert_displacement_field(&field(), Some(&field()), &settings).is_ok());
        assert!(is_space_mismatch(invert_displacement_field(
            &field(),
            Some(&shifted_field()),
            &settings
        )));
    }

    #[test]
    fn fft_correlation_refuses_a_shifted_moving_image() {
        assert!(fft_normalized_correlation(&base(), &base(), 0, 0.0).is_ok());
        assert!(is_space_mismatch(fft_normalized_correlation(
            &base(),
            &shifted_origin(),
            0,
            0.0
        )));
    }

    #[test]
    fn masked_fft_refuses_a_shifted_moving_and_a_shifted_mask_by_index() {
        // Superclass check: moving at index 1, fixed mask at index 2.
        assert!(is_space_mismatch_at(
            masked_fft_normalized_correlation(&base(), &shifted_origin(), None, None, 0, 0.0),
            1
        ));
        assert!(is_space_mismatch_at(
            masked_fft_normalized_correlation(
                &base(),
                &base(),
                Some(&shifted_origin()),
                None,
                0,
                0.0
            ),
            2
        ));
    }

    /// The ADD half survives: the subclass's own mask-size check still fires as
    /// a SizeMismatch, distinct from the physical-space check — the routing
    /// extended VerifyInputInformation, it did not replace the added checks.
    #[test]
    fn masked_fft_keeps_its_added_mask_size_check() {
        let small_mask = Image::from_vec(&[3, 3], vec![1.0f64; 9]).unwrap();
        let out =
            masked_fft_normalized_correlation(&base(), &base(), Some(&small_mask), None, 0, 0.0);
        assert!(
            matches!(out, Err(FilterError::SizeMismatch { .. })),
            "the added mask-size check must still fire, got {out:?}"
        );
    }
}

// ===========================================================================
// Group 11 — N4 bias field correction's confidence image. The image's mask
// already routes through mask_input (index 1). The confidence image is a second
// named input (itkSetInputMacro(ConfidenceImage, …),
// itkN4BiasFieldCorrectionImageFilter.h:198) whose grid the inherited verifier
// also walks — previously only its size was checked.
// ===========================================================================

mod group11_n4_confidence {
    use super::*;
    use sitk_filters::{N4BiasFieldCorrectionSettings, n4_bias_field_correction, paste};

    #[test]
    fn n4_refuses_a_shifted_confidence_image() {
        // The grid check runs in N4::new before the iterative fit; a single
        // fitting iteration keeps the check reachable while staying cheap.
        let settings = N4BiasFieldCorrectionSettings {
            maximum_number_of_iterations: vec![1],
            ..Default::default()
        };
        assert!(is_space_mismatch(n4_bias_field_correction(
            &base(),
            None,
            Some(&shifted_origin()),
            &settings
        )));
    }

    /// The accepting direction: `PasteImageFilter` is EXEMPT — the global-fold
    /// sentinel for this group, whose own inputs all verify.
    #[test]
    fn exempt_paste_still_accepts_a_shifted_source() {
        let out = paste(&base_u8(), &shifted_u8(), &[0, 0], &[2, 2], &[0, 0]);
        assert!(out.is_ok(), "paste is EXEMPT and must accept any grid");
    }
}
