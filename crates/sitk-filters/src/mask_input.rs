//! The preconditions on an **optional mask image input**, in one place.
//!
//! Every ITK filter that takes a mask takes it as a *pipeline input*
//! (`itkSetInputMacro(MaskImage, ...)`, `SetInput2`, `SetNthInput(1, ...)`), never as a
//! plain parameter. That has three consequences, and they are the same three for every
//! such filter:
//!
//! 1. **The mask's pixel type is fixed.** SimpleITK instantiates these filters with the
//!    mask template argument nailed to `itk::Image<uint8_t, Dim>` (e.g.
//!    `ConnectedComponentImageFilter.yaml`, `MaskedAssignImageFilter.yaml`,
//!    `StochasticFractalDimensionImageFilter.yaml`), and the wrapper feeds it through
//!    `CastImageToITK<MaskImageType>` — a `dynamic_cast`, not a value cast — so any other
//!    pixel type throws rather than being converted
//!    ([`FilterError::RequiresUInt8MaskPixelType`]).
//! 2. **The mask must have the image's size**, or the pipeline cannot iterate them together
//!    ([`FilterError::SizeMismatch`]).
//! 3. **The mask must sit on the image's grid.** `ImageToImageFilter::VerifyInputInformation`
//!    (`itkImageToImageFilter.hxx:148-223`) walks *every* input DataObject and compares
//!    origin / spacing / direction against the first, throwing "Inputs do not occupy the same
//!    physical space!" on a mismatch ([`FilterError::PhysicalSpaceMismatch`] with `index: 1`).
//!    ITK never resamples a mask, and never accepts an index-aligned mask that lives
//!    somewhere else in physical space.
//!
//! One owner, because these three were being enforced à la carte: a mask input that skips any
//! of them accepts a call upstream would have thrown on.
//!
//! # The mask's *value* convention is NOT owned here, because ITK has two
//!
//! What the values mean is the caller's filter's business, and the two conventions in this
//! crate come from two different upstream classes:
//!
//! * **`MaskImageFilter`** (`itkMaskImageFilter.h:55`): keep where `mask != m_MaskingValue`,
//!   replace with the outside value where it *equals* it — and `m_MaskingValue` defaults to
//!   `TMask{}`, i.e. **0**. This is the convention of [`crate::label::connected_component`]
//!   (which runs its input through `MaskImageFilter` verbatim,
//!   `itkConnectedComponentImageFilter.hxx:79-92`) and of
//!   [`crate::scalar_connected_component`] (whose functor filter open-codes the same test,
//!   `itkConnectedComponentFunctorImageFilter.hxx:97-110`).
//! * **`MaskedImageToHistogramFilter`** (via `HistogramThresholdImageFilter`): admit to the
//!   histogram where `mask == m_MaskValue`, and `m_MaskValue` defaults to
//!   `NumericTraits<MaskPixelType>::max()`, i.e. **255**. This is the convention of
//!   [`crate::histogram::ThresholdMask`].
//!
//! They are opposite polarities with opposite defaults, and they are both reachable from
//! SimpleITK: a mask of all `1`s keeps every voxel in `connected_component` and admits *no*
//! voxel to a threshold's histogram. Unifying them would invent a third behaviour and call it
//! parity; see ledger §2.175.

use crate::error::{FilterError, Result};
use crate::geometry::same_physical_space;
use sitk_core::{Image, PixelId};

/// Validate a mask image input against its filter's primary input and hand back its voxels.
///
/// Enforces all three preconditions above — pixel type, size, physical space — in that order,
/// and is the only way a filter in this crate should reach a mask's data.
pub(crate) fn uint8_mask_voxels<'a>(image: &Image, mask: &'a Image) -> Result<&'a [u8]> {
    if mask.pixel_id() != PixelId::UInt8 {
        return Err(FilterError::RequiresUInt8MaskPixelType(mask.pixel_id()));
    }
    if mask.size() != image.size() {
        return Err(FilterError::SizeMismatch {
            a: image.size().to_vec(),
            b: mask.size().to_vec(),
        });
    }
    if !same_physical_space(image, mask) {
        return Err(FilterError::PhysicalSpaceMismatch { index: 1 });
    }
    Ok(mask.scalar_slice::<u8>()?)
}
