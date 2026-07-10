//! DICOM (`.dcm`, `.dicom`) reader — `itk::GDCMImageIO`, on the pure-Rust
//! `dicom-object` crate (ledger §5.8(a): `cargo tree -p sitk-io` shows no
//! `*-sys` crate).
//!
//! Upstream is `itkGDCMImageIO.cxx`, but almost none of the observable
//! behaviour lives there: `ReadImageInformation` and `Read` are thin wrappers
//! over `gdcm::ImageReader`, and every geometry rule, pixel-type rule and
//! meta-data rule is GDCM's. So the citations below are to both trees:
//!
//! * `itkGDCMImageIO.cxx` — ITK's own decisions (`CanReadFile`'s sniffing,
//!   the component-type switch, the ultrasound spacing override, the
//!   direction orthogonalisation, the meta-data dictionary loop).
//! * `Modules/ThirdParty/GDCM/src/gdcm/Source/…` — `gdcm::ImageHelper`,
//!   `gdcm::Rescaler`, `gdcm::PixelFormat`, `gdcm::StringFilter`,
//!   `gdcm::MediaStorage`, `gdcm::ImageCodec`, `gdcm::LookupTable`.
//!
//! # This module reads; it does not write
//!
//! `GDCMImageIO::Write` (itkGDCMImageIO.cxx:832-1416) rebuilds a DICOM file
//! from the meta-data dictionary, mints Study/Series/FrameOfReference UIDs from
//! a global `gdcm::UIDGenerator`, and inverse-rescales the buffer. None of that
//! is ported. [`DicomImageIo::can_write_file`] answers `true` for the
//! extensions upstream advertises — as `GDCMImageIO::CanWriteFile` does, since
//! it is nothing but `HasSupportedWriteExtension(name, false)`
//! (itkGDCMImageIO.cxx:814-826) — and [`write`] then refuses. Ledger §5.30
//! carries the open decision.
//!
//! # Encapsulated (compressed) transfer syntaxes are refused
//!
//! `GDCMImageIO::Read` decompresses through `gdcm::ImageChangeTransferSyntax`
//! (itkGDCMImageIO.cxx:295-305), which dispatches to GDCM's own JPEG (libijg),
//! JPEG-LS (CharLS), JPEG 2000 (OpenJPEG) and RLE codecs. Their pixel output is
//! those libraries' bit-for-bit — a different IDCT, a different chroma
//! upsampler, or a different `RequestPlanarConfiguration` dance produces
//! different pixels, not merely a different implementation. Substituting
//! `jpeg-decoder` would be approximating, so every encapsulated transfer syntax
//! is refused with [`IoError::UnsupportedDicomFeature`]. Ledger §4.106.
//!
//! Native syntaxes are read: Implicit VR LE, Explicit VR LE, Explicit VR BE,
//! and Deflated Explicit VR LE.
//!
//! # Sequence-driven SOP classes are refused
//!
//! `ImageHelper::GetSpacingValue` / `GetOriginValue` /
//! `GetDirectionCosinesValue` / `GetRescaleInterceptSlopeValue` take their
//! values out of the Shared / Per-frame Functional Groups Sequences
//! `(5200,9229)` / `(5200,9230)` for the enhanced multi-frame SOP classes, and
//! out of `(0054,0022)` for Nuclear Medicine (gdcmImageHelper.cxx:542-574,
//! :676-724, :1087-1149, :1489-1527). That traversal is not ported;
//! [`read_information`] refuses those SOP classes. Ledger §4.107.
//!
//! # What *is* reproduced
//!
//! * `CanReadFile`'s two-offset `DICM` sniff plus the `readNoPreambleDicom`
//!   heuristic (itkGDCMImageIO.cxx:196-261, :121-182).
//! * The dimension is **always 3**. `GDCMImageIO`'s constructor calls
//!   `SetNumberOfDimensions(3)` (itkGDCMImageIO.cxx:94) and
//!   `InternalReadImageInformation` never lowers it, so a single 2-D slice
//!   loads as a 3-D image of size `[cols, rows, 1]`.
//! * The component type comes from `(0028,0100)` Bits Allocated and
//!   `(0028,0103)` Pixel Representation through `PixelFormat::GetScalarType`
//!   (gdcmPixelFormat.cxx:130-199) — **not** from Bits Stored.
//! * Rescale Slope / Intercept promote the component type through
//!   `Rescaler::ComputeInterceptSlopePixelType` (gdcmRescaler.cxx:196-219),
//!   whose `ComputeBestFit` keys off `PixelFormat::GetMin`/`GetMax`, which *do*
//!   read Bits Stored (gdcmPixelFormat.cxx:222-256).
//! * Spacing is `(0028,0030)` Pixel Spacing (or `(0018,1164)`, `(3002,0011)`,
//!   `(0018,2010)`, `(0028,0034)` — the tag depends on the SOP class), **with
//!   its two values swapped**, and Z spacing is `(0018,0088)` Spacing Between
//!   Slices or `(3004,000c)` (gdcmImageHelper.cxx:1298-1479, :1609, :1645).
//! * Origin is `(0020,0032)` Image Position (Patient), direction is
//!   `(0020,0037)` Image Orientation (Patient), re-orthogonalised through two
//!   cross products (itkGDCMImageIO.cxx:724-744).
//! * The meta-data dictionary keys are `"gggg|eeee"`, lowercase hex, zero
//!   padded (gdcmTag.cxx:80-89); values are `gdcm::StringFilter::ToString`'s,
//!   with binary VRs base64-encoded (itkGDCMImageIO.cxx:746-805).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use dicom_core::dictionary::{DataDictionary, VirtualVr};
use dicom_core::header::{HasLength, Header as _};
use dicom_core::value::Value;
use dicom_core::{Tag, VR};
use dicom_dictionary_std::StandardDataDictionary;
use dicom_object::mem::InMemElement;
use dicom_object::{FileDicomObject, InMemDicomObject, OpenFileOptions};

use sitk_core::{Image, PixelBuffer, PixelId};

use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, ImageIo};
use crate::writer::WriteOptions;

/// The parsed file, as `dicom-object` hands it back.
type Obj = FileDicomObject<InMemDicomObject<StandardDataDictionary>>;

/// `itk::DefaultImageCoordinateTolerance`, the threshold
/// `InternalReadImageInformation` uses to decide that a Z spacing is
/// effectively zero (itkGDCMImageIO.cxx:704, itkMath.h).
const COORDINATE_TOLERANCE: f64 = 1.0e-6;

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

const SOP_CLASS_UID: Tag = Tag(0x0008, 0x0016);
const MODALITY: Tag = Tag(0x0008, 0x0060);
const SPACING_BETWEEN_SLICES: Tag = Tag(0x0018, 0x0088);
const IMAGER_PIXEL_SPACING: Tag = Tag(0x0018, 0x1164);
const NOMINAL_SCANNED_PIXEL_SPACING: Tag = Tag(0x0018, 0x2010);
const IMAGE_POSITION_PATIENT: Tag = Tag(0x0020, 0x0032);
const IMAGE_ORIENTATION_PATIENT: Tag = Tag(0x0020, 0x0037);
const SAMPLES_PER_PIXEL: Tag = Tag(0x0028, 0x0002);
const PHOTOMETRIC_INTERPRETATION: Tag = Tag(0x0028, 0x0004);
const PLANAR_CONFIGURATION: Tag = Tag(0x0028, 0x0006);
const NUMBER_OF_FRAMES: Tag = Tag(0x0028, 0x0008);
const ROWS: Tag = Tag(0x0028, 0x0010);
const COLUMNS: Tag = Tag(0x0028, 0x0011);
const PIXEL_SPACING: Tag = Tag(0x0028, 0x0030);
const PIXEL_ASPECT_RATIO: Tag = Tag(0x0028, 0x0034);
const BITS_ALLOCATED: Tag = Tag(0x0028, 0x0100);
const BITS_STORED: Tag = Tag(0x0028, 0x0101);
const HIGH_BIT: Tag = Tag(0x0028, 0x0102);
const PIXEL_REPRESENTATION: Tag = Tag(0x0028, 0x0103);
const RESCALE_INTERCEPT: Tag = Tag(0x0028, 0x1052);
const RESCALE_SLOPE: Tag = Tag(0x0028, 0x1053);
const SEGMENTED_RED_PALETTE_DATA: Tag = Tag(0x0028, 0x1221);
const IMAGE_PLANE_PIXEL_SPACING: Tag = Tag(0x3002, 0x0011);
const GRID_FRAME_OFFSET_VECTOR: Tag = Tag(0x3004, 0x000c);
const PIXEL_DATA: Tag = Tag(0x7fe0, 0x0010);

// ---------------------------------------------------------------------------
// gdcm::MediaStorage
// ---------------------------------------------------------------------------

/// The `gdcm::MediaStorage::MSType` values `GDCMImageIO` and
/// `gdcm::ImageHelper` branch on. Everything else is [`MediaStorage::Other`],
/// which is GDCM's `MS_END` / `default:` arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MediaStorage {
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
    DigitalMammographyImageStorageForPresentation,
    DigitalMammographyImageStorageForProcessing,
    DigitalIntraoralXrayImageStorageForPresentation,
    DigitalIntraoralXRayImageStorageForProcessing,
    CtImageStorage,
    EnhancedCtImageStorage,
    UltrasoundImageStorageRetired,
    UltrasoundImageStorage,
    UltrasoundMultiFrameImageStorageRetired,
    UltrasoundMultiFrameImageStorage,
    MrImageStorage,
    EnhancedMrImageStorage,
    SecondaryCaptureImageStorage,
    MultiframeSingleBitSecondaryCaptureImageStorage,
    MultiframeGrayscaleByteSecondaryCaptureImageStorage,
    MultiframeGrayscaleWordSecondaryCaptureImageStorage,
    MultiframeTrueColorSecondaryCaptureImageStorage,
    XRayAngiographicImageStorage,
    XRayRadiofluoroscopingImageStorage,
    XRayAngiographicBiPlaneImageStorageRetired,
    NuclearMedicineImageStorage,
    PetImageStorage,
    RtImageStorage,
    RtDoseStorage,
    HardcopyGrayscaleImageStorage,
    HardcopyColorImageStorage,
    Philips3D,
    VideoEndoscopicImageStorage,
    GeneralElectricMagneticResonanceImageStorage,
    GePrivate3DModelStorage,
    PhilipsPrivateMrSyntheticImageStorage,
    VlPhotographicImageStorage,
    VlMicroscopicImageStorage,
    SegmentationStorage,
    XRay3DAngiographicImageStorage,
    XRay3DCraniofacialImageStorage,
    EnhancedUsVolumeStorage,
    BreastTomosynthesisImageStorage,
    BreastProjectionXRayImageStorageForPresentation,
    BreastProjectionXRayImageStorageForProcessing,
    OphthalmicTomographyImageStorage,
    EnhancedPetImageStorage,
    EnhancedMrColorImageStorage,
    IvoctForPresentation,
    IvoctForProcessing,
    LegacyConvertedEnhancedCtImageStorage,
    LegacyConvertedEnhancedMrImageStorage,
    LegacyConvertedEnhancedPetImageStorage,
    FujiPrivateMammoCrImageStorage,
    /// GDCM's `MS_END`: an unrecognised SOP Class UID, or none at all.
    Other,
}

/// The `MSStrings` table (gdcmMediaStorage.cxx:26-155), restricted to the rows
/// [`MediaStorage`] carries.
const MEDIA_STORAGE_UIDS: &[(&str, MediaStorage)] = &[
    (
        "1.2.840.10008.5.1.1.29",
        MediaStorage::HardcopyGrayscaleImageStorage,
    ),
    (
        "1.2.840.10008.5.1.1.30",
        MediaStorage::HardcopyColorImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.1",
        MediaStorage::ComputedRadiographyImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.1.1",
        MediaStorage::DigitalXRayImageStorageForPresentation,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.1.1.1",
        MediaStorage::DigitalXRayImageStorageForProcessing,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.1.2",
        MediaStorage::DigitalMammographyImageStorageForPresentation,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.1.2.1",
        MediaStorage::DigitalMammographyImageStorageForProcessing,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.1.3",
        MediaStorage::DigitalIntraoralXrayImageStorageForPresentation,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.1.3.1",
        MediaStorage::DigitalIntraoralXRayImageStorageForProcessing,
    ),
    ("1.2.840.10008.5.1.4.1.1.2", MediaStorage::CtImageStorage),
    (
        "1.2.840.10008.5.1.4.1.1.2.1",
        MediaStorage::EnhancedCtImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.2.2",
        MediaStorage::LegacyConvertedEnhancedCtImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.3",
        MediaStorage::UltrasoundMultiFrameImageStorageRetired,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.3.1",
        MediaStorage::UltrasoundMultiFrameImageStorage,
    ),
    ("1.2.840.10008.5.1.4.1.1.4", MediaStorage::MrImageStorage),
    (
        "1.2.840.10008.5.1.4.1.1.4.1",
        MediaStorage::EnhancedMrImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.4.3",
        MediaStorage::EnhancedMrColorImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.4.4",
        MediaStorage::LegacyConvertedEnhancedMrImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.6",
        MediaStorage::UltrasoundImageStorageRetired,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.6.1",
        MediaStorage::UltrasoundImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.6.2",
        MediaStorage::EnhancedUsVolumeStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.7",
        MediaStorage::SecondaryCaptureImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.7.1",
        MediaStorage::MultiframeSingleBitSecondaryCaptureImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.7.2",
        MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.7.3",
        MediaStorage::MultiframeGrayscaleWordSecondaryCaptureImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.7.4",
        MediaStorage::MultiframeTrueColorSecondaryCaptureImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.12.1",
        MediaStorage::XRayAngiographicImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.12.2",
        MediaStorage::XRayRadiofluoroscopingImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.12.3",
        MediaStorage::XRayAngiographicBiPlaneImageStorageRetired,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.13.1.1",
        MediaStorage::XRay3DAngiographicImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.13.1.2",
        MediaStorage::XRay3DCraniofacialImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.13.1.3",
        MediaStorage::BreastTomosynthesisImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.13.1.4",
        MediaStorage::BreastProjectionXRayImageStorageForPresentation,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.13.1.5",
        MediaStorage::BreastProjectionXRayImageStorageForProcessing,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.14.1",
        MediaStorage::IvoctForPresentation,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.14.2",
        MediaStorage::IvoctForProcessing,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.20",
        MediaStorage::NuclearMedicineImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.66.4",
        MediaStorage::SegmentationStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.77.1.1.1",
        MediaStorage::VideoEndoscopicImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.77.1.2",
        MediaStorage::VlMicroscopicImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.77.1.4",
        MediaStorage::VlPhotographicImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.77.1.5.4",
        MediaStorage::OphthalmicTomographyImageStorage,
    ),
    ("1.2.840.10008.5.1.4.1.1.128", MediaStorage::PetImageStorage),
    (
        "1.2.840.10008.5.1.4.1.1.128.1",
        MediaStorage::LegacyConvertedEnhancedPetImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.130",
        MediaStorage::EnhancedPetImageStorage,
    ),
    (
        "1.2.840.10008.5.1.4.1.1.481.1",
        MediaStorage::RtImageStorage,
    ),
    ("1.2.840.10008.5.1.4.1.1.481.2", MediaStorage::RtDoseStorage),
    ("1.2.840.113543.6.6.1.3.10002", MediaStorage::Philips3D),
    (
        "1.2.840.113619.4.2",
        MediaStorage::GeneralElectricMagneticResonanceImageStorage,
    ),
    ("1.2.840.113619.4.26", MediaStorage::GePrivate3DModelStorage),
    (
        "1.3.46.670589.5.0.10",
        MediaStorage::PhilipsPrivateMrSyntheticImageStorage,
    ),
    (
        "1.2.392.200036.9125.1.1.4",
        MediaStorage::FujiPrivateMammoCrImageStorage,
    ),
];

/// `MediaStorage::GetMSType` (gdcmMediaStorage.cxx:157-183) over the table.
fn media_storage_from_uid(uid: &str) -> MediaStorage {
    // `GetFromDataSetOrHeader` truncates a UI at the first space
    // (gdcmMediaStorage.cxx:417-422); the trailing NUL pad is stripped by the
    // caller.
    let uid = match uid.find(' ') {
        Some(i) => &uid[..i],
        None => uid,
    };
    MEDIA_STORAGE_UIDS
        .iter()
        .find(|(candidate, _)| *candidate == uid)
        .map_or(MediaStorage::Other, |(_, ms)| *ms)
}

/// `MediaStorage::SetFromFile` (gdcmMediaStorage.cxx:544-619).
///
/// Both `(0002,0002)` Media Storage SOP Class UID and `(0008,0016)` SOP Class
/// UID are read; when they disagree the *dataset*'s wins. Only when both are
/// absent does the `(0008,0060)` Modality guess run — and this port stops
/// there, because `GuessFromModality`'s table is only consulted for ACR-NEMA
/// files that `dicom-object` cannot parse anyway (no file meta group). A file
/// with neither UID resolves to [`MediaStorage::Other`], which is GDCM's
/// `SetFromModality` failure arm, except that GDCM then defaults to
/// `SecondaryCaptureImageStorage` when Pixel Data is present
/// (gdcmMediaStorage.cxx:532-538) — reproduced here.
fn media_storage_from_file(obj: &Obj) -> MediaStorage {
    let header_uid = obj
        .meta()
        .media_storage_sop_class_uid
        .trim_end_matches('\0');
    let dataset_uid = string_value(obj, SOP_CLASS_UID).unwrap_or_default();
    let dataset_uid = dataset_uid.trim_end_matches(['\0', ' ']);

    if !header_uid.is_empty() && !dataset_uid.is_empty() && header_uid == dataset_uid {
        return media_storage_from_uid(header_uid);
    }
    if !dataset_uid.is_empty() {
        return media_storage_from_uid(dataset_uid);
    }
    if !header_uid.is_empty() {
        return media_storage_from_uid(header_uid);
    }
    // Neither UID: GDCM falls back to Modality, and — failing that — to
    // SecondaryCaptureImageStorage when the dataset carries Pixel Data.
    if find(obj, MODALITY).is_some() || find(obj, PIXEL_DATA).is_some() {
        return MediaStorage::SecondaryCaptureImageStorage;
    }
    MediaStorage::Other
}

impl MediaStorage {
    /// Whether `ImageHelper` would read this SOP class's geometry out of a
    /// functional-groups sequence rather than out of the top-level dataset —
    /// the union of the guard lists at gdcmImageHelper.cxx:542-561, :676-694,
    /// :1087-1099 and :1489-1508, plus `NuclearMedicineImageStorage`'s
    /// `(0054,0022)` path (:575-601, :725-755).
    fn is_sequence_driven(self) -> bool {
        matches!(
            self,
            Self::EnhancedCtImageStorage
                | Self::EnhancedMrImageStorage
                | Self::EnhancedMrColorImageStorage
                | Self::EnhancedPetImageStorage
                | Self::OphthalmicTomographyImageStorage
                | Self::MultiframeSingleBitSecondaryCaptureImageStorage
                | Self::MultiframeGrayscaleByteSecondaryCaptureImageStorage
                | Self::MultiframeGrayscaleWordSecondaryCaptureImageStorage
                | Self::MultiframeTrueColorSecondaryCaptureImageStorage
                | Self::XRay3DAngiographicImageStorage
                | Self::XRay3DCraniofacialImageStorage
                | Self::SegmentationStorage
                | Self::IvoctForProcessing
                | Self::IvoctForPresentation
                | Self::BreastTomosynthesisImageStorage
                | Self::BreastProjectionXRayImageStorageForPresentation
                | Self::BreastProjectionXRayImageStorageForProcessing
                | Self::LegacyConvertedEnhancedMrImageStorage
                | Self::LegacyConvertedEnhancedCtImageStorage
                | Self::LegacyConvertedEnhancedPetImageStorage
                | Self::NuclearMedicineImageStorage
        )
    }

    /// Whether `InternalReadImageInformation`'s own `switch(ms)` claims the
    /// spacing, bypassing `image.GetSpacing()` (itkGDCMImageIO.cxx:650-693).
    fn itk_overrides_spacing(self) -> bool {
        matches!(
            self,
            Self::HardcopyGrayscaleImageStorage
                | Self::GePrivate3DModelStorage
                | Self::Philips3D
                | Self::VideoEndoscopicImageStorage
                | Self::UltrasoundMultiFrameImageStorage
                | Self::UltrasoundImageStorage
                | Self::UltrasoundImageStorageRetired
                | Self::UltrasoundMultiFrameImageStorageRetired
        )
    }

    /// `ImageHelper::GetSpacingTagFromMediaStorage` (gdcmImageHelper.cxx:1298-1402),
    /// with `ForcePixelSpacing` false (ITK never sets it). `None` is GDCM's
    /// `Tag(0xffff,0xffff)`.
    ///
    /// The `SecondaryCaptureImageStorage` arm is not reachable through here:
    /// `GetSpacingValue` handles it inline (`:1552-1565`).
    fn spacing_tag(self) -> Option<Tag> {
        match self {
            Self::EnhancedUsVolumeStorage
            | Self::CtImageStorage
            | Self::MrImageStorage
            | Self::RtDoseStorage
            | Self::NuclearMedicineImageStorage
            | Self::PetImageStorage
            | Self::GeneralElectricMagneticResonanceImageStorage
            | Self::PhilipsPrivateMrSyntheticImageStorage
            | Self::VlPhotographicImageStorage
            | Self::VlMicroscopicImageStorage
            | Self::IvoctForProcessing
            | Self::IvoctForPresentation => Some(PIXEL_SPACING),

            Self::ComputedRadiographyImageStorage
            | Self::DigitalXRayImageStorageForPresentation
            | Self::DigitalXRayImageStorageForProcessing
            | Self::DigitalMammographyImageStorageForPresentation
            | Self::DigitalMammographyImageStorageForProcessing
            | Self::DigitalIntraoralXrayImageStorageForPresentation
            | Self::DigitalIntraoralXRayImageStorageForProcessing
            | Self::XRayAngiographicImageStorage
            | Self::XRayRadiofluoroscopingImageStorage
            | Self::XRayAngiographicBiPlaneImageStorageRetired
            | Self::FujiPrivateMammoCrImageStorage => Some(IMAGER_PIXEL_SPACING),

            Self::RtImageStorage => Some(IMAGE_PLANE_PIXEL_SPACING),

            // `SecondaryCaptureImagePlaneModule` is on, so SC reads (0028,0030).
            Self::SecondaryCaptureImageStorage => Some(PIXEL_SPACING),

            Self::MultiframeSingleBitSecondaryCaptureImageStorage
            | Self::MultiframeGrayscaleByteSecondaryCaptureImageStorage
            | Self::MultiframeGrayscaleWordSecondaryCaptureImageStorage
            | Self::MultiframeTrueColorSecondaryCaptureImageStorage
            | Self::HardcopyGrayscaleImageStorage
            | Self::HardcopyColorImageStorage => Some(NOMINAL_SCANNED_PIXEL_SPACING),

            Self::UltrasoundImageStorage
            | Self::UltrasoundImageStorageRetired
            | Self::UltrasoundMultiFrameImageStorageRetired => Some(PIXEL_ASPECT_RATIO),

            // GePrivate3DModelStorage, Philips3D, VideoEndoscopicImageStorage,
            // UltrasoundMultiFrameImageStorage and every unlisted class:
            // `Tag(0xffff,0xffff)`.
            _ => None,
        }
    }

    /// `ImageHelper::GetZSpacingTagFromMediaStorage` (gdcmImageHelper.cxx:1404-1478)
    /// with `SecondaryCaptureImagePlaneModule` **on** — ITK sets it
    /// (itkGDCMImageIO.cxx:451) — and `ForcePixelSpacing` off.
    ///
    /// Note the fall-through in upstream's `switch`: CT, PET, RT Image, every
    /// projection-radiography class, every ultrasound class *and* Secondary
    /// Capture all land in the same arm, so with the plane module on they all
    /// take `(0018,0088)`.
    fn z_spacing_tag(self) -> Option<Tag> {
        match self {
            Self::EnhancedUsVolumeStorage
            | Self::MrImageStorage
            | Self::NuclearMedicineImageStorage
            | Self::GeneralElectricMagneticResonanceImageStorage
            | Self::PetImageStorage
            | Self::CtImageStorage
            | Self::RtImageStorage
            | Self::ComputedRadiographyImageStorage
            | Self::DigitalXRayImageStorageForPresentation
            | Self::DigitalXRayImageStorageForProcessing
            | Self::DigitalMammographyImageStorageForPresentation
            | Self::DigitalMammographyImageStorageForProcessing
            | Self::DigitalIntraoralXrayImageStorageForPresentation
            | Self::DigitalIntraoralXRayImageStorageForProcessing
            | Self::XRayAngiographicImageStorage
            | Self::XRayRadiofluoroscopingImageStorage
            | Self::XRayAngiographicBiPlaneImageStorageRetired
            | Self::UltrasoundImageStorage
            | Self::UltrasoundMultiFrameImageStorage
            | Self::UltrasoundImageStorageRetired
            | Self::UltrasoundMultiFrameImageStorageRetired
            | Self::SecondaryCaptureImageStorage => Some(SPACING_BETWEEN_SLICES),

            Self::RtDoseStorage => Some(GRID_FRAME_OFFSET_VECTOR),

            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// gdcm::PixelFormat
// ---------------------------------------------------------------------------

/// `gdcm::PixelFormat::ScalarType` (gdcmPixelFormat.h:51-67).
///
/// The discriminants are upstream's, because `GDCMImageIO` compares two of
/// them with `>` (itkGDCMImageIO.cxx:529-545) through
/// `PixelFormat::operator ScalarType()` (gdcmPixelFormat.h:86).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ScalarType {
    UInt8 = 0,
    Int8 = 1,
    UInt12 = 2,
    Int12 = 3,
    UInt16 = 4,
    Int16 = 5,
    UInt32 = 6,
    Int32 = 7,
    UInt64 = 8,
    Int64 = 9,
    Float16 = 10,
    Float32 = 11,
    Float64 = 12,
    SingleBit = 13,
    Unknown = 14,
}

/// `gdcm::PixelFormat` (gdcmPixelFormat.h:243-253).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PixelFormat {
    samples_per_pixel: u16,
    bits_allocated: u16,
    bits_stored: u16,
    high_bit: u16,
    pixel_representation: u16,
}

impl PixelFormat {
    /// `PixelFormat::SetBitsAllocated` (gdcmPixelFormat.h:99-121): three
    /// bit-mask spellings some scanners emit are read as the value they meant,
    /// and Bits Stored / High Bit are reset to follow.
    fn set_bits_allocated(&mut self, ba: u16) {
        if ba == 0 {
            return;
        }
        let ba = match ba {
            0xffff => 16,
            0x0fff => 12,
            0x00ff => 8,
            other => other,
        };
        self.bits_allocated = ba;
        self.bits_stored = ba;
        self.high_bit = ba - 1;
    }

    /// `PixelFormat::SetBitsStored` (gdcmPixelFormat.h:134-149): the same three
    /// bit-mask spellings some scanners emit (FUJIFILM CR + MONO1) are read as
    /// the value they meant, then guarded to `bs <= BitsAllocated && bs` before
    /// Bits Stored / High Bit follow. `SetHighBit(bs - 1)` collapses to
    /// `high_bit = bs - 1` here: the guard forces `1 <= bs <= BitsAllocated`, so
    /// `bs - 1` never hits `SetHighBit`'s own bitmask remaps (:159-167).
    fn set_bits_stored(&mut self, bs: u16) {
        let bs = match bs {
            0xffff => 16,
            0x0fff => 12,
            0x00ff => 8,
            other => other,
        };
        if bs <= self.bits_allocated && bs != 0 {
            self.bits_stored = bs;
            self.high_bit = bs - 1;
        }
    }

    /// `PixelFormat::GetScalarType` (gdcmPixelFormat.cxx:130-199).
    ///
    /// Bits **Allocated** picks the width; Pixel Representation picks the sign
    /// by stepping one place up the [`ScalarType`] enum. Bits Stored plays no
    /// part — a 16/12/11 unsigned pixel is `UInt16`.
    fn scalar_type(self) -> ScalarType {
        let base = match self.bits_allocated {
            0 => return ScalarType::Unknown,
            1 => ScalarType::SingleBit,
            8 => ScalarType::UInt8,
            12 => ScalarType::UInt12,
            16 => ScalarType::UInt16,
            32 => ScalarType::UInt32,
            64 => ScalarType::UInt64,
            // "This is illegal in DICOM, assuming a RGB image" (:157-160).
            24 => ScalarType::UInt8,
            _ => return ScalarType::Unknown,
        };
        match self.pixel_representation {
            0 => base,
            1 => match base {
                ScalarType::SingleBit => ScalarType::Unknown,
                ScalarType::UInt8 => ScalarType::Int8,
                ScalarType::UInt12 => ScalarType::Int12,
                ScalarType::UInt16 => ScalarType::Int16,
                ScalarType::UInt32 => ScalarType::Int32,
                ScalarType::UInt64 => ScalarType::Int64,
                _ => ScalarType::Unknown,
            },
            // The "secret codes" GDCM writes for float pixel data (:99-113).
            2 if self.bits_allocated == 16 => ScalarType::Float16,
            3 if self.bits_allocated == 32 => ScalarType::Float32,
            4 if self.bits_allocated == 64 => ScalarType::Float64,
            _ => ScalarType::Unknown,
        }
    }

    /// `PixelFormat::GetMin` (gdcmPixelFormat.cxx:222-238) — keyed off Bits
    /// **Stored**, not Bits Allocated.
    fn min(self) -> i64 {
        match self.pixel_representation {
            1 => !(((1u64 << self.bits_stored) - 1) >> 1) as i64,
            _ => 0,
        }
    }

    /// `PixelFormat::GetMax` (gdcmPixelFormat.cxx:240-256).
    fn max(self) -> i64 {
        match self.pixel_representation {
            1 => (((1u64 << self.bits_stored) - 1) >> 1) as i64,
            _ => ((1u64 << self.bits_stored) - 1) as i64,
        }
    }

    /// `PixelFormat::GetPixelSize` (gdcmPixelFormat.cxx:206-220): bytes per
    /// pixel, all samples included. A 12-bit pixel "fakes a short value".
    fn pixel_size(self) -> usize {
        let per_sample = if self.bits_allocated == 12 {
            2
        } else {
            (self.bits_allocated / 8) as usize
        };
        per_sample * self.samples_per_pixel as usize
    }
}

/// `ImageHelper::GetPixelFormatValue` (gdcmImageHelper.cxx:847-894). Every
/// attribute defaults as GDCM's `Attribute<...> at = { 0 }` does — and Samples
/// per Pixel to `1`.
fn pixel_format_from_dataset(obj: &Obj) -> PixelFormat {
    let mut pf = PixelFormat {
        samples_per_pixel: 1,
        bits_allocated: 0,
        bits_stored: 0,
        high_bit: 0,
        pixel_representation: 0,
    };
    pf.set_bits_allocated(u16_value(obj, BITS_ALLOCATED).unwrap_or(0));
    if let Some(bs) = u16_value(obj, BITS_STORED) {
        pf.set_bits_stored(bs);
    }
    if let Some(hb) = u16_value(obj, HIGH_BIT) {
        pf.high_bit = hb;
    }
    pf.pixel_representation = u16_value(obj, PIXEL_REPRESENTATION).unwrap_or(0);
    pf.samples_per_pixel = u16_value(obj, SAMPLES_PER_PIXEL).unwrap_or(1);
    pf
}

// ---------------------------------------------------------------------------
// gdcm::Rescaler
// ---------------------------------------------------------------------------

/// `ComputeBestFit` (gdcmRescaler.cxx:123-194).
fn compute_best_fit(pf: PixelFormat, intercept: f64, slope: f64) -> ScalarType {
    let (pfmin, pfmax) = if slope >= 0.0 {
        (pf.min() as f64, pf.max() as f64)
    } else {
        (pf.max() as f64, pf.min() as f64)
    };
    let min = slope * pfmin + intercept;
    let max = slope * pfmax + intercept;

    if min >= 0.0 {
        if max <= f64::from(u8::MAX) {
            ScalarType::UInt8
        } else if max <= f64::from(u16::MAX) {
            ScalarType::UInt16
        } else if max <= f64::from(u32::MAX) {
            ScalarType::UInt32
        } else if max <= u64::MAX as f64 {
            ScalarType::Float64
        } else {
            ScalarType::Unknown
        }
    } else if max <= f64::from(i8::MAX) && min >= f64::from(i8::MIN) {
        ScalarType::Int8
    } else if max <= f64::from(i16::MAX) && min >= f64::from(i16::MIN) {
        ScalarType::Int16
    } else if max <= f64::from(i32::MAX) && min >= f64::from(i32::MIN) {
        ScalarType::Int32
    } else if max <= i64::MAX as f64 && min >= i64::MIN as f64 {
        ScalarType::Float64
    } else {
        ScalarType::Unknown
    }
}

/// `Rescaler::ComputeInterceptSlopePixelType` (gdcmRescaler.cxx:196-219).
///
/// A non-integral slope or intercept forces `Float64` outright; otherwise the
/// output type is the narrowest one that holds `slope * [min, max] + intercept`.
fn compute_intercept_slope_pixel_type(pf: PixelFormat, intercept: f64, slope: f64) -> ScalarType {
    if pf.samples_per_pixel != 1 {
        return pf.scalar_type();
    }
    if pf.scalar_type() == ScalarType::SingleBit {
        return ScalarType::SingleBit;
    }
    if slope != slope.trunc() || intercept != intercept.trunc() {
        return ScalarType::Float64;
    }
    compute_best_fit(pf, intercept, slope)
}

/// The `ptLarger` guard (itkGDCMImageIO.cxx:523-551): for an unsigned output
/// type the *signed* counterpart is the yardstick, so a signed input may widen
/// into an unsigned output.
fn pixel_type_larger_than_output(pixel: ScalarType, output: ScalarType) -> bool {
    match output {
        ScalarType::UInt8 => pixel > ScalarType::Int8,
        ScalarType::UInt12 => pixel > ScalarType::Int12,
        ScalarType::UInt16 => pixel > ScalarType::Int16,
        ScalarType::UInt32 => pixel > ScalarType::Int32,
        ScalarType::UInt64 => pixel > ScalarType::Int64,
        other => pixel > other,
    }
}

/// `InternalReadImageInformation`'s two `switch(pixeltype)` ladders
/// (itkGDCMImageIO.cxx:466-507, :559-596), which agree on every arm they share.
fn component_pixel_id(st: ScalarType) -> Result<PixelId> {
    Ok(match st {
        ScalarType::Int8 => PixelId::Int8,
        ScalarType::UInt8 => PixelId::UInt8,
        ScalarType::Int12 | ScalarType::Int16 => PixelId::Int16,
        ScalarType::UInt12 | ScalarType::UInt16 => PixelId::UInt16,
        ScalarType::Int32 => PixelId::Int32,
        ScalarType::UInt32 => PixelId::UInt32,
        ScalarType::Float32 => PixelId::Float32,
        ScalarType::Float64 => PixelId::Float64,
        other => {
            return Err(IoError::MalformedDicom(format!(
                "Unhandled PixelFormat: {other:?}"
            )));
        }
    })
}

// ---------------------------------------------------------------------------
// Photometric interpretation
// ---------------------------------------------------------------------------

/// The `gdcm::PhotometricInterpretation` values the read path branches on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Photometric {
    Monochrome1,
    Monochrome2,
    PaletteColor,
    Rgb,
    YbrFull,
    YbrFull422,
    YbrIct,
    YbrRct,
    Other,
}

impl Photometric {
    fn parse(s: &str) -> Self {
        match s.trim_end_matches(['\0', ' ']) {
            "MONOCHROME1" => Self::Monochrome1,
            "MONOCHROME2" => Self::Monochrome2,
            "PALETTE COLOR" => Self::PaletteColor,
            "RGB" => Self::Rgb,
            "YBR_FULL" => Self::YbrFull,
            "YBR_FULL_422" => Self::YbrFull422,
            "YBR_ICT" => Self::YbrIct,
            "YBR_RCT" => Self::YbrRct,
            _ => Self::Other,
        }
    }
}

/// `PixmapReader::ReadImageInternal` (gdcmPixmapReader.cxx:779-849): the
/// element when present, else the Samples-per-Pixel default.
fn photometric_from_dataset(obj: &Obj, samples_per_pixel: u16) -> Photometric {
    match string_value(obj, PHOTOMETRIC_INTERPRETATION) {
        Some(s) if !s.trim_end_matches(['\0', ' ']).is_empty() => Photometric::parse(&s),
        _ if samples_per_pixel == 3 => Photometric::Rgb,
        _ => Photometric::Monochrome2,
    }
}

// ---------------------------------------------------------------------------
// Dataset accessors
// ---------------------------------------------------------------------------

/// `DataSet::FindDataElement` — present, even if zero-length.
fn find(obj: &Obj, tag: Tag) -> Option<&InMemElement<StandardDataDictionary>> {
    obj.element_opt(tag).ok().flatten()
}

/// `DataElement::IsEmpty` — a zero value length.
fn is_empty(e: &InMemElement<StandardDataDictionary>) -> bool {
    e.length() == dicom_core::Length(0)
}

/// The element's raw value bytes, as `gdcm::ByteValue::GetPointer` would give
/// them. `None` for a sequence or an encapsulated pixel-data fragment list.
fn value_bytes(e: &InMemElement<StandardDataDictionary>) -> Option<Cow<'_, [u8]>> {
    match e.value() {
        Value::Primitive(p) => Some(p.to_bytes()),
        _ => None,
    }
}

/// The element's value as text, undecoded past ISO-8859-1.
fn string_value(obj: &Obj, tag: Tag) -> Option<String> {
    let e = find(obj, tag)?;
    let bytes = value_bytes(e)?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// `Attribute<gggg,eeee>` over a `US` element: the value, or `None` when the
/// element is absent or empty (upstream then keeps the attribute's default).
fn u16_value(obj: &Obj, tag: Tag) -> Option<u16> {
    let e = find(obj, tag)?;
    if is_empty(e) {
        return None;
    }
    e.value().to_int::<u16>().ok()
}

/// `Element<VR::DS, VM::VM1_n>::Read` over at most `max` values.
///
/// GDCM reads with `std::istream::operator>>(double)`, which skips leading
/// whitespace and stops at the first character that cannot extend the number.
/// A token that parses to nothing leaves the slot untouched — an uninitialised
/// `double` — so this port stops at the first unparsable token instead, and its
/// callers check the count.
fn parse_decimal_strings(bytes: &[u8], max: usize) -> Vec<f64> {
    let text = String::from_utf8_lossy(bytes);
    let mut out = Vec::new();
    for token in text.split('\\').take(max) {
        match token.trim().trim_end_matches('\0').trim().parse::<f64>() {
            Ok(v) => out.push(v),
            Err(_) => break,
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// `ImageHelper::GetDimensionsValue` (gdcmImageHelper.cxx:899-971), minus the
/// ACR-NEMA `(0028,0005)` legacy arm, which needs an ACR-NEMA parser this crate
/// does not have.
fn dimensions(obj: &Obj) -> [usize; 3] {
    let cols = u16_value(obj, COLUMNS).unwrap_or(0) as usize;
    let rows = u16_value(obj, ROWS).unwrap_or(0) as usize;
    let frames = find(obj, NUMBER_OF_FRAMES)
        .filter(|e| !is_empty(e))
        .and_then(|e| e.value().to_int::<i64>().ok())
        .unwrap_or(0);
    // "theReturn[2] = 1; if (numberofframes > 1) theReturn[2] = at.GetValue();"
    let depth = if frames > 1 { frames as usize } else { 1 };
    [cols, rows, depth]
}

/// The 2-D part of `ImageHelper::GetSpacingValue` (gdcmImageHelper.cxx:1570-1658).
///
/// Reproduces two upstream quirks verbatim:
///
/// * a `DS` value of `0` becomes `1.0` ("Cannot have a spacing of 0", `:1605`);
/// * the two values are **swapped** (`:1609`), so `spacing[0]` is Pixel
///   Spacing's *column* spacing and `spacing[1]` its *row* spacing.
///
/// A single-valued `DS` (no backslash) is duplicated and **not** swapped
/// (`:1611-1621`).
fn in_plane_spacing(obj: &Obj, tag: Option<Tag>) -> Result<[f64; 2]> {
    let Some(tag) = tag else {
        return Ok([1.0, 1.0]);
    };
    let Some(e) = find(obj, tag).filter(|e| !is_empty(e)) else {
        return Ok([1.0, 1.0]);
    };
    let bytes = value_bytes(e)
        .ok_or_else(|| IoError::MalformedDicom(format!("{tag} holds a sequence, not a spacing")))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();

    if text.contains('\\') {
        let values = parse_decimal_strings(&bytes, 2);
        if values.len() != 2 {
            return Err(IoError::MalformedDicom(format!(
                "{tag} has {} parsable value(s), expected 2",
                values.len()
            )));
        }
        let fix = |v: f64| if v == 0.0 { 1.0 } else { v };
        Ok([fix(values[1]), fix(values[0])])
    } else {
        let single = text
            .trim()
            .trim_end_matches('\0')
            .trim()
            .parse::<f64>()
            .map_err(|_| IoError::MalformedDicom(format!("{tag} is not a decimal string")))?;
        let single = if single == 0.0 { 1.0 } else { single };
        Ok([single, single])
    }
}

/// The Z part of `ImageHelper::GetSpacingValue` (gdcmImageHelper.cxx:1663-1769).
///
/// The `(0028,0009)` Frame Increment Pointer fallback is not ported: it is only
/// reachable when the SOP class has no Z spacing tag at all, and every such
/// class either lands in [`MediaStorage::itk_overrides_spacing`] or in
/// [`MediaStorage::is_sequence_driven`]. Ledger §4.108.
fn z_spacing(obj: &Obj, ms: MediaStorage) -> Result<f64> {
    let Some(tag) = ms.z_spacing_tag() else {
        return Ok(1.0);
    };
    let Some(e) = find(obj, tag) else {
        return Ok(1.0);
    };
    if is_empty(e) {
        return Ok(1.0);
    }
    let bytes =
        value_bytes(e).ok_or_else(|| IoError::MalformedDicom(format!("{tag} holds a sequence")))?;

    if tag == GRID_FRAME_OFFSET_VECTOR {
        // VM2_n: the spacing is the first offset step (:1716-1719).
        let values = parse_decimal_strings(&bytes, usize::MAX);
        if values.len() < 2 {
            return Err(IoError::MalformedDicom(
                "(3004,000c) Grid Frame Offset Vector needs at least two values".into(),
            ));
        }
        return Ok(values[1] - values[0]);
    }

    // VM1: `el.Read(ss)` then `sp.push_back(el.GetValue(i))` for each value read.
    let values = parse_decimal_strings(&bytes, 1);
    match values.first() {
        Some(&v) => Ok(v),
        // GDCM pushes nothing, leaving `sp` two long, and the caller then reads
        // `sp[2]` out of bounds (gdcmImageHelper.cxx:1771 asserts it away in a
        // debug build only). No deterministic outcome to reproduce. Ledger §1.70.
        None => Err(IoError::MalformedDicom(format!(
            "{tag} has no parsable decimal value"
        ))),
    }
}

/// `InternalReadImageInformation`'s ultrasound/hardcopy spacing arm
/// (itkGDCMImageIO.cxx:650-693): read `(0028,0030)` directly, swap, and punt
/// the Z spacing to `1.0`.
///
/// Upstream asserts `m_El.GetLength() == 2` and then reads both slots; a
/// single-valued `(0028,0030)` leaves the second slot uninitialised, so this
/// port refuses instead. Ledger §1.71.
fn itk_override_spacing(obj: &Obj) -> Result<[f64; 3]> {
    let Some(e) = find(obj, PIXEL_SPACING).filter(|e| !is_empty(e)) else {
        return Ok([1.0, 1.0, 1.0]);
    };
    let bytes = value_bytes(e).ok_or_else(|| {
        IoError::MalformedDicom("(0028,0030) holds a sequence, not a spacing".into())
    })?;
    let values = parse_decimal_strings(&bytes, 2);
    if values.len() != 2 {
        return Err(IoError::MalformedDicom(format!(
            "(0028,0030) has {} parsable value(s), expected 2",
            values.len()
        )));
    }
    // `std::swap(sp[0], sp[1])`; note this arm does *not* rewrite a zero.
    Ok([values[1], values[0], 1.0])
}

/// `ImageHelper::GetOriginValue`'s default arm (gdcmImageHelper.cxx:602-623).
fn origin(obj: &Obj) -> [f64; 3] {
    let mut ori = [0.0; 3];
    if let Some(e) = find(obj, IMAGE_POSITION_PATIENT) {
        if let Some(bytes) = value_bytes(e) {
            for (slot, value) in ori.iter_mut().zip(parse_decimal_strings(&bytes, 3)) {
                *slot = value;
            }
        }
    }
    ori
}

/// `DirectionCosines::IsValid` (gdcmDirectionCosines.cxx:53-72): both rows unit
/// length and mutually orthogonal, to `1e-3`.
fn direction_cosines_valid(d: &[f64; 6]) -> bool {
    const EPSILON: f64 = 1e-3;
    let norm_v1 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let norm_v2 = d[3] * d[3] + d[4] * d[4] + d[5] * d[5];
    let dot = d[0] * d[3] + d[1] * d[4] + d[2] * d[5];
    (norm_v1 - 1.0).abs() < EPSILON && (norm_v2 - 1.0).abs() < EPSILON && dot.abs() < EPSILON
}

/// `DirectionCosines::Normalize` (gdcmDirectionCosines.cxx:119-131), applied to
/// each row; a zero row is left alone.
fn normalize_direction_cosines(d: &mut [f64; 6]) {
    for row in 0..2 {
        let s = &mut d[row * 3..row * 3 + 3];
        let norm = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        if norm != 0.0 {
            for v in s.iter_mut() {
                *v /= norm;
            }
        }
    }
}

/// `ImageHelper::GetDirectionCosinesFromDataSet` (gdcmImageHelper.cxx:626-667)
/// followed by the default arm of `GetDirectionCosinesValue` (`:757-766`):
/// an absent, unreadable or unfixable `(0020,0037)` yields the identity pair.
fn direction_cosines(obj: &Obj) -> [f64; 6] {
    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let Some(e) = find(obj, IMAGE_ORIENTATION_PATIENT) else {
        return IDENTITY;
    };
    let Some(bytes) = value_bytes(e) else {
        return IDENTITY;
    };
    // `Attribute<0x0020,0x0037> at = {{1,0,0,0,1,0}}` — the default survives an
    // empty element.
    let mut dircos = IDENTITY;
    for (slot, value) in dircos.iter_mut().zip(parse_decimal_strings(&bytes, 6)) {
        *slot = value;
    }
    if direction_cosines_valid(&dircos) {
        return dircos;
    }
    normalize_direction_cosines(&mut dircos);
    if direction_cosines_valid(&dircos) {
        dircos
    } else {
        IDENTITY
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm == 0.0 {
        v
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

/// `InternalReadImageInformation`'s re-orthogonalisation (itkGDCMImageIO.cxx:724-744),
/// flattened into ITK's row-major direction matrix.
///
/// The direction cosines are the matrix's **columns**
/// (itkImageFileReader.hxx:181-188), so `direction[j * 3 + i]` is axis `i`'s
/// `j`-th component.
fn direction_matrix(dircos: [f64; 6]) -> Vec<f64> {
    let row = [dircos[0], dircos[1], dircos[2]];
    let column = [dircos[3], dircos[4], dircos[5]];

    let slice = normalized(cross(row, column));
    let row = normalized(cross(column, slice));
    let column = cross(slice, row);

    let axes = [row, column, slice];
    let mut m = vec![0.0; 9];
    for (i, axis) in axes.iter().enumerate() {
        for (j, &v) in axis.iter().enumerate() {
            m[j * 3 + i] = v;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Rescale
// ---------------------------------------------------------------------------

/// `GetRescaleInterceptSlopeValueFromDataSet` (gdcmImageHelper.cxx:812-845).
///
/// `ImageHelper::GetRescaleInterceptSlopeValue`'s SOP-class ladder collapses
/// under ITK, which sets `ForceRescaleInterceptSlope(true)`
/// (itkGDCMImageIO.cxx:445): the `|| ForceRescaleInterceptSlope` disjunct at
/// gdcmImageHelper.cxx:1163 always fires, so the Philips-private MR arm and the
/// Real World Value Mapping arm below it are dead for every non-enhanced SOP
/// class.
fn rescale_intercept_slope(obj: &Obj) -> Result<(f64, f64)> {
    let mut intercept = 0.0;
    let mut slope = 1.0;

    if let Some(e) = find(obj, RESCALE_INTERCEPT).filter(|e| !is_empty(e)) {
        let bytes = value_bytes(e)
            .ok_or_else(|| IoError::MalformedDicom("(0028,1052) holds a sequence".into()))?;
        if let Some(&v) = parse_decimal_strings(&bytes, 1).first() {
            intercept = v;
        }
    }
    if let Some(e) = find(obj, RESCALE_SLOPE).filter(|e| !is_empty(e)) {
        let bytes = value_bytes(e)
            .ok_or_else(|| IoError::MalformedDicom("(0028,1053) holds a sequence".into()))?;
        if let Some(&v) = parse_decimal_strings(&bytes, 1).first() {
            // "Cannot have slope == 0. Defaulting to 1.0 instead" (:832-837).
            slope = if v == 0.0 { 1.0 } else { v };
        }
    }
    Ok((intercept, slope))
}

// ---------------------------------------------------------------------------
// Meta-data dictionary
// ---------------------------------------------------------------------------

/// `Tag::PrintAsPipeSeparatedString` (gdcmTag.cxx:80-89) — lowercase hex, four
/// digits each, zero padded.
fn pipe_separated(tag: Tag) -> String {
    format!("{:04x}|{:04x}", tag.group(), tag.element())
}

/// Whether ITK routes this VR through the base64 branch
/// (itkGDCMImageIO.cxx:768-769).
fn is_binary_vr(vr: VR) -> bool {
    matches!(
        vr,
        VR::OB | VR::OD | VR::OF | VR::OL | VR::OV | VR::OW | VR::SQ | VR::UN
    )
}

/// `VR::IsASCII` (gdcmVR.h:280-312).
fn is_ascii_vr(vr: VR) -> bool {
    matches!(
        vr,
        VR::AE
            | VR::AS
            | VR::CS
            | VR::DA
            | VR::DS
            | VR::DT
            | VR::IS
            | VR::LO
            | VR::LT
            | VR::PN
            | VR::SH
            | VR::ST
            | VR::TM
            | VR::UC
            | VR::UI
            | VR::UR
            | VR::UT
    )
}

/// `DataSetHelper::ComputeVR` (gdcmDataSetHelper.cxx:79-297), restricted to the
/// public tags ITK actually keeps.
///
/// The dictionary VR wins whenever it is known; the element's own VR is
/// consulted only when the dictionary has nothing (or says `UN`). GDCM's dual
/// VRs arrive here as `dicom-rs`'s [`VirtualVr`] variants: `Xs` is `US_SS`,
/// `Px` / `Ox` are `OB_OW`, `Lt` is `US_OW`.
fn compute_vr(e: &InMemElement<StandardDataDictionary>, pixel_representation: u16) -> VR {
    let tag = e.tag();
    let dict_vr = StandardDataDictionary.by_tag(tag).map(|entry| entry.vr);

    match dict_vr {
        None | Some(VirtualVr::Exact(VR::UN)) => {
            // `dicom-rs` never reports `VR::INVALID`; an implicit-VR dataset
            // hands back the dictionary's relaxed VR, and an unknown tag `UN`.
            let devr = e.header().vr();
            if devr != VR::UN {
                return devr;
            }
            if e.length().is_undefined() {
                // CP-246: `UN` with an undefined length is really a sequence.
                return VR::SQ;
            }
            VR::UN
        }
        // `US_SS` resolves against Pixel Representation, except (0028,0071)
        // which is always `US` (gdcmDataSetHelper.cxx:138-187).
        Some(VirtualVr::Xs) => {
            if tag == Tag(0x0028, 0x0071) || pixel_representation != 1 {
                VR::US
            } else {
                VR::SS
            }
        }
        // `OB_OW`: Pixel Data and overlay data are `OW` unless encapsulated
        // (:221-230). This module refuses encapsulated files, so `OW` it is —
        // and ITK skips (7fe0,0010) anyway.
        Some(VirtualVr::Px) | Some(VirtualVr::Ox) => VR::OW,
        // `US_OW` -> `OW` (:278-281).
        Some(VirtualVr::Lt) => VR::OW,
        Some(VirtualVr::Exact(vr)) => vr,
        // `VirtualVr` is `#[non_exhaustive]`; any future context-dependent VR
        // falls back to its unambiguous form, else the element's own VR.
        Some(other) => other.exact().unwrap_or_else(|| e.header().vr()),
    }
}

/// `std::ostream::operator<<(double)` with the default precision of 6
/// significant digits — i.e. C's `%g` with precision 6.
///
/// Rust's `{}` prints the shortest round-tripping representation, which is a
/// different string for almost every value with more than six significant
/// digits, so it cannot be used for a `FL` / `FD` meta-data value.
fn format_default_float(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if !v.is_finite() {
        // libstdc++ prints "inf"/"-inf"/"nan".
        return if v.is_nan() {
            "nan".to_string()
        } else if v > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }

    const PRECISION: i32 = 6;
    let exponent = v.abs().log10().floor() as i32;
    // `%g` re-derives the exponent from the *rounded* value, so 9.9999999 with
    // precision 6 prints as "10" (exponent 1), not "10.0000" (exponent 0).
    let exponent = {
        let scaled = format!("{:.*e}", (PRECISION - 1) as usize, v);
        scaled
            .rsplit_once('e')
            .and_then(|(_, e)| e.parse::<i32>().ok())
            .unwrap_or(exponent)
    };

    let mut s = if !(-4..PRECISION).contains(&exponent) {
        let mantissa = format!("{:.*e}", (PRECISION - 1) as usize, v);
        let (mantissa, exp) = mantissa.rsplit_once('e').expect("`{:e}` always emits `e`");
        let exp: i32 = exp
            .parse()
            .expect("`{:e}` always emits an integer exponent");
        let mantissa = trim_trailing_zeros(mantissa);
        // libstdc++ prints at least two exponent digits, with a sign.
        format!(
            "{mantissa}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    } else {
        let decimals = (PRECISION - 1 - exponent).max(0) as usize;
        trim_trailing_zeros(&format!("{v:.decimals$}"))
    };
    if s.is_empty() {
        s.push('0');
    }
    s
}

/// `%g` strips trailing zeros in the fraction, and a bare trailing point.
fn trim_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `gdcm::StringFilter::ToString` (gdcmStringFilter.cxx:338-476).
fn string_filter_to_string(e: &InMemElement<StandardDataDictionary>, vr: VR) -> String {
    if vr == VR::UN || vr == VR::SQ {
        // `ToStringPairInternal` bails to "" for UN (:402-406) and warns its way
        // to "" for SQ (:461-467).
        return String::new();
    }
    let Some(bytes) = value_bytes(e) else {
        return String::new();
    };
    if bytes.is_empty() {
        return String::new();
    }

    if is_ascii_vr(vr) {
        // The raw bytes, truncated at the first NUL (:418-421). Trailing spaces
        // — DICOM's even-length padding — are *kept*.
        let text = String::from_utf8_lossy(&bytes);
        let end = text.find('\0').unwrap_or(text.len());
        return text[..end].to_string();
    }

    /// `StringFilterCase` (gdcmStringFilter.cxx:77-87): values joined with `\`.
    macro_rules! join {
        ($width:expr, $from:expr, $fmt:expr) => {{
            let mut out = String::new();
            for chunk in bytes.chunks_exact($width) {
                if !out.is_empty() {
                    out.push('\\');
                }
                let _ = write!(out, "{}", $fmt($from(chunk)));
            }
            out
        }};
    }

    match vr {
        VR::US => join!(2, |c: &[u8]| u16::from_le_bytes([c[0], c[1]]), |v: u16| v
            .to_string()),
        VR::SS => join!(2, |c: &[u8]| i16::from_le_bytes([c[0], c[1]]), |v: i16| v
            .to_string()),
        VR::UL => join!(
            4,
            |c: &[u8]| u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            |v: u32| v.to_string()
        ),
        VR::SL => join!(
            4,
            |c: &[u8]| i32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            |v: i32| v.to_string()
        ),
        VR::FL => join!(
            4,
            |c: &[u8]| f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            |v: f32| format_default_float(f64::from(v))
        ),
        VR::FD => join!(
            8,
            |c: &[u8]| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
            format_default_float
        ),
        // `Tag::operator<<` (gdcmTag.h:287-294).
        VR::AT => join!(
            4,
            |c: &[u8]| (
                u16::from_le_bytes([c[0], c[1]]),
                u16::from_le_bytes([c[2], c[3]])
            ),
            |(g, el): (u16, u16)| format!("({g:04x},{el:04x})")
        ),
        // Every other VR hits `default: gdcm_assert(0)`, which is a no-op in a
        // release build and leaves the value empty. Ledger §2.148.
        _ => String::new(),
    }
}

/// `itksysBase64_Encode(input, len, output, 0)` (KWSys Base64.c:91-128) —
/// standard base64, `=`-padded, no end marker.
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        out.push(ALPHABET[usize::from(b[0] >> 2)] as char);
        out.push(ALPHABET[usize::from(((b[0] << 4) & 0x30) | (b[1] >> 4))] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[usize::from(((b[1] << 2) & 0x3c) | (b[2] >> 6))] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[usize::from(b[2] & 0x3f)] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// The meta-data dictionary loop (itkGDCMImageIO.cxx:746-805).
///
/// `m_LoadPrivateTags` is `false` by default (itkGDCMImageIO.h:289) and
/// SimpleITK's `ImageFileReader` never turns it on for a plain `ReadImage`, so
/// odd-group tags are dropped wholesale.
fn metadata(obj: &Obj, pixel_representation: u16) -> BTreeMap<String, String> {
    let mut dict = BTreeMap::new();
    for e in obj.iter() {
        let tag = e.tag();
        // `Tag::IsPublic()` — an even group (gdcmTag.h:150).
        if tag.group() % 2 != 0 {
            continue;
        }
        let vr = compute_vr(e, pixel_representation);
        if is_binary_vr(vr) {
            if vr == VR::SQ || tag == PIXEL_DATA {
                continue;
            }
            // `ref.GetByteValue()` is null for a sequence of fragments.
            if let Some(bytes) = value_bytes(e) {
                dict.insert(pipe_separated(tag), base64_encode(&bytes));
            }
        } else {
            dict.insert(pipe_separated(tag), string_filter_to_string(e, vr));
        }
    }
    dict
}

// ---------------------------------------------------------------------------
// CanReadFile
// ---------------------------------------------------------------------------

/// `readNoPreambleDicom` (itkGDCMImageIO.cxx:121-182).
///
/// Walks group-2 elements from offset 0 until a non-group-2 element appears;
/// the file is DICOM-like when that element's group is `0x0008`.
fn read_no_preamble_dicom(file: &mut BufReader<File>) -> bool {
    const EXPLICIT_VRS: [&[u8; 2]; 20] = [
        b"AE", b"AS", b"AT", b"CS", b"DA", b"DS", b"DT", b"FL", b"FD", b"IS", b"LO", b"PN", b"SH",
        b"SL", b"SS", b"ST", b"TM", b"UI", b"UL", b"US",
    ];

    loop {
        let mut tag = [0u8; 4];
        if file.read_exact(&mut tag).is_err() {
            return false;
        }
        let group = u16::from_le_bytes([tag[0], tag[1]]);
        if group != 0x0002 && group != 0x0008 {
            return false;
        }
        let mut vrcode = [0u8; 2];
        if file.read_exact(&mut vrcode).is_err() {
            return false;
        }

        let length: u64 = if EXPLICIT_VRS.contains(&&vrcode) {
            let mut len = [0u8; 2];
            if file.read_exact(&mut len).is_err() {
                return false;
            }
            u64::from(u16::from_le_bytes(len))
        } else {
            // Implicit VR: the two VR bytes are the low half of a 32-bit length.
            let mut rest = [0u8; 2];
            if file.read_exact(&mut rest).is_err() {
                return false;
            }
            u64::from(u32::from_le_bytes([vrcode[0], vrcode[1], rest[0], rest[1]]))
        };
        if length == 0 {
            return false;
        }
        // `file.ignore(length); if (file.eof()) return false;`
        match std::io::copy(&mut file.by_ref().take(length), &mut std::io::sink()) {
            Ok(n) if n < length => return false,
            Ok(_) => {}
            Err(_) => return false,
        }
        if group != 0x0002 {
            return true;
        }
    }
}

/// `GDCMImageIO::CanReadFile` (itkGDCMImageIO.cxx:196-261).
///
/// Sniffs `DICM` at offset 128 **and then at 0** — upstream's loop probes both
/// and never breaks early — then falls back on [`read_no_preamble_dicom`]. Note
/// the ordering quirk: the offset-128 read happens first, so a file shorter
/// than 132 bytes fails the `file.fail()` check and is rejected outright, even
/// if it is a perfectly good preamble-less DICOM stream.
fn can_read_file(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    let mut file = BufReader::new(file);

    let mut dicomsig = false;
    for offset in [128u64, 0] {
        if file.seek(SeekFrom::Start(offset)).is_err() {
            return false;
        }
        let mut buf = [0u8; 4];
        if file.read_exact(&mut buf).is_err() {
            return false;
        }
        if &buf == b"DICM" {
            dicomsig = true;
        }
    }
    if !dicomsig {
        if file.seek(SeekFrom::Start(0)).is_err() {
            return false;
        }
        dicomsig = read_no_preamble_dicom(&mut file);
    }
    if !dicomsig {
        return false;
    }

    // `gdcm::ImageReader::Read()`: the file must parse *and* hold an image.
    // `dicom-object` only parses, so the image test is spelled out here — a
    // parseable DICOM with no Pixel Data (an RTSTRUCT, say) is not readable.
    // Ledger §4.109.
    match open(path) {
        Ok(obj) => {
            let [cols, rows, _] = dimensions(&obj);
            cols > 0 && rows > 0 && find(&obj, PIXEL_DATA).is_some()
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

fn open(path: &Path) -> Result<Obj> {
    OpenFileOptions::new()
        .open_file(path)
        .map_err(|e| IoError::DicomRead(e.to_string()))
}

/// Everything `InternalReadImageInformation` leaves on the IO for `Read` to
/// pick up (itkGDCMImageIO.cxx:430-806).
struct ImageHeader {
    obj: Obj,
    /// `m_Dimensions`, always three long.
    size: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    /// Row-major 3×3 direction matrix.
    direction: Vec<f64>,
    /// The file's stored pixel format — `image.GetPixelFormat()`.
    pixel_format: PixelFormat,
    photometric: Photometric,
    /// `m_ComponentType`, after the rescale promotion.
    component: PixelId,
    /// `m_NumberOfComponents`, after `PALETTE_COLOR` expands to three.
    number_of_components: usize,
    intercept: f64,
    slope: f64,
    /// `m_SingleBit`.
    single_bit: bool,
    metadata: BTreeMap<String, String>,
}

fn read_header(path: &Path) -> Result<ImageHeader> {
    let obj = open(path)?;

    let ts = obj
        .meta()
        .transfer_syntax
        .trim_end_matches(['\0', ' '])
        .to_owned();
    if !is_native_transfer_syntax(&ts) {
        return Err(IoError::UnsupportedDicomFeature(format!(
            "encapsulated transfer syntax {ts}: GDCM decodes it with libijg / CharLS / \
             OpenJPEG, whose pixel output no pure-Rust codec reproduces bit-for-bit"
        )));
    }

    let ms = media_storage_from_file(&obj);
    if ms.is_sequence_driven() {
        return Err(IoError::UnsupportedDicomFeature(format!(
            "SOP class {ms:?} keeps its geometry in a functional-groups sequence, \
             which this port does not traverse"
        )));
    }

    let pf = pixel_format_from_dataset(&obj);
    let stored = pf.scalar_type();

    // `InternalReadImageInformation`'s first switch (itkGDCMImageIO.cxx:466-507).
    let single_bit = stored == ScalarType::SingleBit;
    if matches!(stored, ScalarType::UInt12 | ScalarType::Int12) {
        return Err(IoError::UnsupportedDicomFeature(
            "12 bits allocated: GDCM unpacks the packed 12-bit stream in its codec layer, \
             which this port does not implement"
                .into(),
        ));
    }
    if !single_bit {
        // Rejects UINT64 / INT64 / FLOAT16 / UNKNOWN, as upstream's
        // `itkExceptionMacro("Unhandled PixelFormat: ...")` does.
        component_pixel_id(stored)?;
    }

    let photometric = photometric_from_dataset(&obj, pf.samples_per_pixel);
    if photometric == Photometric::YbrFull422 {
        return Err(IoError::UnsupportedDicomFeature(
            "native YBR_FULL_422: GDCM's RAWCodec upsamples the 4:2:2 chroma through \
             DecodeByStreams, which this port does not implement"
                .into(),
        ));
    }
    // `DoInvertMonochrome` is a silent no-op for MONOCHROME1 unless Bits
    // Allocated is 8 or 16 (gdcmImageCodec.cxx:367-430); reproducing the no-op
    // would emit an un-inverted MONOCHROME1 as if it were MONOCHROME2. Refused.
    if photometric == Photometric::Monochrome1 && !matches!(pf.bits_allocated, 8 | 16) {
        return Err(IoError::UnsupportedDicomFeature(format!(
            "MONOCHROME1 with {} bits allocated: gdcm::ImageCodec::DoInvertMonochrome \
             only inverts 8- or 16-bit pixels",
            pf.bits_allocated
        )));
    }

    // `outputpt` (itkGDCMImageIO.cxx:509-556).
    let (intercept, slope) = if single_bit {
        (0.0, 1.0)
    } else {
        rescale_intercept_slope(&obj)?
    };
    let output = if single_bit {
        ScalarType::UInt8
    } else if slope != 1.0 || intercept != 0.0 {
        let promoted = compute_intercept_slope_pixel_type(pf, intercept, slope);
        if pixel_type_larger_than_output(stored, promoted) {
            return Err(IoError::MalformedDicom(
                "Pixel type larger than output type".into(),
            ));
        }
        promoted
    } else {
        stored
    };
    let component = component_pixel_id(output)?;

    // `m_NumberOfComponents` (itkGDCMImageIO.cxx:598-634).
    let mut number_of_components = usize::from(pf.samples_per_pixel);
    if photometric == Photometric::PaletteColor {
        number_of_components = 3;
    }

    let size = dimensions(&obj);
    let spacing = if ms.itk_overrides_spacing() {
        itk_override_spacing(&obj)?
    } else {
        let plane = in_plane_spacing(&obj, ms.spacing_tag())?;
        let z = z_spacing(&obj, ms)?;
        // "Spacing may be negative at this point, will be fixed below"
        // (itkGDCMImageIO.cxx:703-704).
        let z = if z.abs() < COORDINATE_TOLERANCE {
            1.0
        } else {
            z
        };
        [plane[0], plane[1], z]
    };

    let origin = origin(&obj);
    let direction = direction_matrix(direction_cosines(&obj));
    let metadata = metadata(&obj, pf.pixel_representation);

    Ok(ImageHeader {
        obj,
        size,
        spacing,
        origin,
        direction,
        pixel_format: pf,
        photometric,
        component,
        number_of_components,
        intercept,
        slope,
        single_bit,
        metadata,
    })
}

/// The DICOM transfer syntaxes whose Pixel Data is stored natively (not
/// encapsulated) — `gdcm::RAWCodec::CanDecode` (gdcmRAWCodec.cxx:53-60), minus
/// `ImplicitVRBigEndianPrivateGE`, which `dicom-object` does not parse.
fn is_native_transfer_syntax(uid: &str) -> bool {
    matches!(
        uid,
        "1.2.840.10008.1.2"          // Implicit VR Little Endian
            | "1.2.840.10008.1.2.1"  // Explicit VR Little Endian
            | "1.2.840.10008.1.2.1.99" // Deflated Explicit VR Little Endian
            | "1.2.840.10008.1.2.2" // Explicit VR Big Endian
    )
}

// ---------------------------------------------------------------------------
// Pixel decode — `GDCMImageIO::Read` (itkGDCMImageIO.cxx:263-427)
// ---------------------------------------------------------------------------

/// `image.GetBuffer()` for a native transfer syntax: the raw Pixel Data bytes,
/// little-endian on every host (`dicom-object` has already byte-swapped an
/// Explicit VR Big Endian stream while parsing).
fn pixel_data_bytes(obj: &Obj) -> Result<Vec<u8>> {
    let e = find(obj, PIXEL_DATA)
        .ok_or_else(|| IoError::MalformedDicom("no Pixel Data element (7fe0,0010)".into()))?;
    let bytes = value_bytes(e).ok_or_else(|| {
        IoError::UnsupportedDicomFeature(
            "Pixel Data is an encapsulated fragment sequence, not a native buffer".into(),
        )
    })?;
    Ok(bytes.into_owned())
}

/// Bytes per stored sample: `PixelFormat::GetPixelSize` for one sample.
fn bytes_per_sample(pf: PixelFormat) -> usize {
    (pf.bits_allocated as usize) / 8
}

/// `GDCMImageIO::Read`, minus the transfer-syntax change (refused upstream of
/// here) and the debug asserts.
fn decode_pixels(h: &ImageHeader) -> Result<PixelBuffer> {
    let pf = h.pixel_format;
    let [cols, rows, frames] = h.size;
    let frames = frames.max(1);

    let raw = pixel_data_bytes(&h.obj)?;

    if h.single_bit {
        return decode_single_bit(&raw, cols, rows, frames);
    }

    // GetBufferLength for the ordinary path: every dimension times the pixel
    // size (gdcmBitmap.cxx:310-315). GetBuffer copies exactly this many bytes.
    let expected = cols
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(frames))
        .and_then(|v| v.checked_mul(pf.pixel_size()))
        .ok_or_else(|| IoError::MalformedDicom("image dimensions overflow".into()))?;
    if raw.len() < expected {
        return Err(IoError::TruncatedData);
    }
    let mut bytes = raw;
    bytes.truncate(expected);

    // Planar Configuration 1 → 0: ITK always requests interleaved
    // (itkGDCMImageIO.cxx:307-317).
    let planar = u16_value(&h.obj, PLANAR_CONFIGURATION).unwrap_or(0);
    if pf.samples_per_pixel == 3 && planar == 1 {
        bytes = deinterleave_planes(&bytes, cols, rows, frames, bytes_per_sample(pf))?;
    }

    // The per-photometric buffer transform (itkGDCMImageIO.cxx:325-360).
    match h.photometric {
        Photometric::PaletteColor => {
            if h.slope != 1.0 || h.intercept != 0.0 {
                return Err(IoError::UnsupportedDicomFeature(
                    "PALETTE_COLOR with a Rescale Slope/Intercept: ITK rescales the \
                     already-expanded RGB buffer against the index pixel type, which this \
                     port does not reproduce"
                        .into(),
                ));
            }
            bytes = apply_palette(&h.obj, &bytes, pf)?;
        }
        Photometric::Monochrome1 => bytes = invert_monochrome(&bytes, pf),
        _ => {}
    }

    // Y'CbCr → RGB gate (itkGDCMImageIO.cxx:404-417).
    let stored = pf.scalar_type();
    let ybr = h.number_of_components == 3
        && h.photometric == Photometric::YbrFull
        && matches!(stored, ScalarType::UInt8 | ScalarType::Int8);

    if h.slope != 1.0 || h.intercept != 0.0 {
        if ybr {
            // ITK rescales in place, then reinterprets the rescaled buffer as
            // `unsigned char` for YCbCr_to_RGB — a size/type mismatch when the
            // output type widens. Refused rather than reproduced.
            return Err(IoError::UnsupportedDicomFeature(
                "YBR_FULL combined with a Rescale Slope/Intercept: ITK's YCbCr_to_RGB \
                 reinterprets the rescaled buffer as unsigned char"
                    .into(),
            ));
        }
        let output = compute_intercept_slope_pixel_type(pf, h.intercept, h.slope);
        bytes = rescale_stream(&bytes, stored, output, h.intercept, h.slope);
    }

    if ybr {
        ycbcr_to_rgb(&mut bytes)?;
    }

    Ok(bytes_to_pixel_buffer(bytes, h.component))
}

/// SINGLEBIT expansion (itkGDCMImageIO.cxx:320-338, :362-378): each packed bit
/// becomes a 0 / 255 byte, LSB first.
///
/// ITK's expansion loop runs `len / 8` times where `len = cols * rows *
/// frames`, ignoring the per-row byte padding GDCM added when `cols % 8 != 0`
/// (gdcmBitmap.cxx:288-294). Reproducing that would read the wrong bytes for a
/// padded row, so a non-byte-aligned width is refused. Ledger §1.72.
fn decode_single_bit(raw: &[u8], cols: usize, rows: usize, frames: usize) -> Result<PixelBuffer> {
    if cols % 8 != 0 {
        return Err(IoError::UnsupportedDicomFeature(format!(
            "SINGLEBIT image with Columns = {cols} (not a multiple of 8): ITK's bit-expansion \
             loop ignores the per-row byte padding"
        )));
    }
    let expanded = cols
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(frames))
        .ok_or_else(|| IoError::MalformedDicom("image dimensions overflow".into()))?;
    let packed = expanded / 8;
    if raw.len() < packed {
        return Err(IoError::TruncatedData);
    }
    let mut out = vec![0u8; expanded];
    for (i, &c) in raw[..packed].iter().enumerate() {
        for bit in 0..8 {
            out[i * 8 + bit] = if c & (1 << bit) != 0 { 255 } else { 0 };
        }
    }
    Ok(PixelBuffer::UInt8(out))
}

/// `ImageChangePlanarConfiguration::RGBPlanesToRGBPixels` per frame
/// (gdcmImageChangePlanarConfiguration.cxx:82-99): `[R…R G…G B…B]` → `[RGB …]`.
fn deinterleave_planes(
    raw: &[u8],
    cols: usize,
    rows: usize,
    frames: usize,
    bps: usize,
) -> Result<Vec<u8>> {
    let plane = cols * rows * bps;
    let framesize = plane * 3;
    if raw.len() < framesize * frames {
        return Err(IoError::TruncatedData);
    }
    let mut out = vec![0u8; framesize * frames];
    for z in 0..frames {
        let base = z * framesize;
        for i in 0..(cols * rows) {
            for s in 0..3 {
                let src = base + s * plane + i * bps;
                let dst = base + (i * 3 + s) * bps;
                out[dst..dst + bps].copy_from_slice(&raw[src..src + bps]);
            }
        }
    }
    Ok(out)
}

/// `ImageCodec::DoInvertMonochrome` (gdcmImageCodec.cxx:367-430). Only the
/// 8- and 16-bit arms exist; other widths were refused in [`read_header`].
fn invert_monochrome(raw: &[u8], pf: PixelFormat) -> Vec<u8> {
    let mut out = raw.to_vec();
    match pf.bits_allocated {
        8 => {
            for b in &mut out {
                *b = 255u8.wrapping_sub(*b);
            }
        }
        16 => {
            if pf.pixel_representation == 1 {
                for chunk in out.chunks_exact_mut(2) {
                    let c = u16::from_le_bytes([chunk[0], chunk[1]]);
                    chunk.copy_from_slice(&0xffffu16.wrapping_sub(c).to_le_bytes());
                }
            } else {
                // `mask = 2^BitsStored - 1`, built the same way GDCM builds it
                // so BitsStored ≥ 16 saturates to 0xffff rather than overflowing.
                let mask: u16 = if pf.bits_stored >= 16 {
                    0xffff
                } else {
                    (1u16 << pf.bits_stored) - 1
                };
                for chunk in out.chunks_exact_mut(2) {
                    let mut c = u16::from_le_bytes([chunk[0], chunk[1]]);
                    if c > mask {
                        c = mask;
                    }
                    chunk.copy_from_slice(&(mask - c).to_le_bytes());
                }
            }
        }
        _ => {}
    }
    out
}

/// The palette lookup table `PixmapReader::ReadImageInternal` builds
/// (gdcmPixmapReader.cxx:259-336) and `LookupTable::Decode` applies
/// (gdcmLookupTable.cxx:506-561).
struct LookupTable {
    /// 8 or 16 — `pf.GetBitsAllocated()`, the width of the pixel index.
    bit_sample: u16,
    /// `Internal->RGB`, interleaved `[R,G,B]`. `u16` entries for the 16-bit
    /// table are stored little-endian, two bytes each.
    rgb: Vec<u8>,
    /// `Internal->Length[type]`, the entry count per channel.
    length: [usize; 3],
}

/// `Element<VR::US, VM::VM3>::SetFromDataElement` over `(0028,1101+i)`.
fn palette_descriptor(obj: &Obj, tag: Tag) -> Result<[u16; 3]> {
    let e = find(obj, tag)
        .ok_or_else(|| IoError::MalformedDicom(format!("missing palette descriptor {tag}")))?;
    let values = e
        .value()
        .to_multi_int::<u16>()
        .map_err(|_| IoError::MalformedDicom(format!("{tag} is not a US descriptor")))?;
    if values.len() < 3 {
        return Err(IoError::MalformedDicom(format!(
            "{tag} has {} values, expected 3",
            values.len()
        )));
    }
    Ok([values[0], values[1], values[2]])
}

/// `ImageApplyLookupTable::Apply` (gdcmImageApplyLookupTable.cxx:37-131) for the
/// non-RGB8 path, followed by the `LookupTable::Decode` over the index buffer.
///
/// Segmented palette LUTs `(0028,1221-1223)` are refused: GDCM routes them
/// through `SegmentedPaletteColorLookupTable` after a
/// `gdcm_assert(0 && "Please report this image")` (gdcmPixmapReader.cxx:263-267),
/// so its own author treats them as untested. Ledger §4.110.
fn apply_palette(obj: &Obj, index: &[u8], pf: PixelFormat) -> Result<Vec<u8>> {
    let bit_sample = pf.bits_allocated;
    if bit_sample != 8 && bit_sample != 16 {
        return Err(IoError::UnsupportedDicomFeature(format!(
            "PALETTE_COLOR with {bit_sample} bits allocated: only 8- and 16-bit indices are read"
        )));
    }
    if find(obj, SEGMENTED_RED_PALETTE_DATA).is_some() {
        return Err(IoError::UnsupportedDicomFeature(
            "segmented palette colour lookup table (0028,1221): GDCM's own reader flags it \
             untested"
                .into(),
        ));
    }

    let entry_bytes = (bit_sample as usize) / 8;
    let mut lut = LookupTable {
        bit_sample,
        rgb: vec![
            0u8;
            if bit_sample == 8 {
                256 * 3
            } else {
                65536 * 3 * 2
            }
        ],
        length: [0; 3],
    };

    for i in 0..3u16 {
        let desc = palette_descriptor(obj, Tag(0x0028, 0x1101 + i))?;
        // `InitializeLUT`: a length of 0 means 65536 (gdcmLookupTable.cxx:98-101).
        let length = if desc[0] == 0 {
            65536
        } else {
            desc[0] as usize
        };
        let bitsize = desc[2];
        // `InitializeLUT` silently does nothing for any other entry width
        // (gdcmLookupTable.cxx:91-94), which would then read an all-zero table.
        if bitsize != 8 && bitsize != 16 {
            return Err(IoError::MalformedDicom(format!(
                "palette descriptor (0028,{:04x}) declares {bitsize}-bit entries",
                0x1101 + i
            )));
        }
        lut.length[i as usize] = length;

        let data_tag = Tag(0x0028, 0x1201 + i);
        let data = find(obj, data_tag).and_then(value_bytes).ok_or_else(|| {
            // GDCM warns "Icon Sequence is incomplete. Giving up" and clears
            // the pixel data (gdcmPixmapReader.cxx:329-334), so the read
            // fails downstream.
            IoError::MalformedDicom(format!("missing palette data {data_tag}"))
        })?;
        lut.set_channel(i as usize, bitsize, &data)?;
    }

    lut.decode(index, entry_bytes)
}

impl LookupTable {
    /// `LookupTable::SetLUT` (gdcmLookupTable.cxx:186-244).
    fn set_channel(&mut self, channel: usize, bitsize: u16, data: &[u8]) -> Result<()> {
        let length = self.length[channel];
        if self.bit_sample == 8 {
            let mult = (bitsize as usize) / 8; // bytes per stored entry
            let expected = length * mult;
            // "Length*mult == data" (single byte per entry after the offset) vs
            // the `mult2 == 2` fallback for two-byte-per-entry tables.
            let (stride, offset) = if data.len() == expected || data.len() == expected + 1 {
                (mult, if mult == 2 { 1 } else { 0 })
            } else if length != 0 && data.len() == length * 2 {
                (2, 0)
            } else {
                return Err(IoError::MalformedDicom(format!(
                    "palette channel {channel} has {} data bytes, expected {expected}",
                    data.len()
                )));
            };
            for i in 0..length {
                let src = i * stride + offset;
                if src >= data.len() {
                    return Err(IoError::MalformedDicom(
                        "palette data shorter than declared length".into(),
                    ));
                }
                self.rgb[3 * i + channel] = data[src];
            }
        } else {
            // 16-bit table: `Length * 2 == data` (gdcmLookupTable.cxx:225).
            if data.len() != length * 2 {
                return Err(IoError::MalformedDicom(format!(
                    "16-bit palette channel {channel} has {} data bytes, expected {}",
                    data.len(),
                    length * 2
                )));
            }
            for i in 0..length {
                let value = [data[2 * i], data[2 * i + 1]];
                let dst = 2 * (3 * i + channel);
                self.rgb[dst..dst + 2].copy_from_slice(&value);
            }
        }
        Ok(())
    }

    /// `LookupTable::Decode` (gdcmLookupTable.cxx:506-561): each index maps to
    /// its `[R,G,B]` triple.
    fn decode(&self, index: &[u8], entry_bytes: usize) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(index.len() / entry_bytes * 3 * entry_bytes);
        if self.bit_sample == 8 {
            for &idx in index {
                let base = 3 * idx as usize;
                out.extend_from_slice(&self.rgb[base..base + 3]);
            }
        } else {
            if index.len() % 2 != 0 {
                return Err(IoError::MalformedDicom(
                    "16-bit palette index buffer has an odd byte length".into(),
                ));
            }
            for pair in index.chunks_exact(2) {
                let idx = u16::from_le_bytes([pair[0], pair[1]]) as usize;
                let base = 2 * 3 * idx;
                out.extend_from_slice(&self.rgb[base..base + 6]);
            }
        }
        Ok(out)
    }
}

/// `YCbCr_to_RGB` (itkGDCMImageIO.cxx:185-192) →
/// `ImageChangePhotometricInterpretation::YBR2RGB` with `storedbits = 8`
/// (gdcmImageChangePhotometricInterpretation.h:93-106).
fn ycbcr_to_rgb(bytes: &mut [u8]) -> Result<()> {
    if bytes.len() % 3 != 0 {
        return Err(IoError::MalformedDicom(format!(
            "Buffer size {} is not valid for a 3-sample YBR image",
            bytes.len()
        )));
    }
    // `Round(x) = (int)(x + 0.5)` truncates toward zero, as the C cast does.
    let round = |x: f64| (x + 0.5) as i32;
    let clamp = |v: i32| -> u8 { v.clamp(0, 255) as u8 };
    const HALF: f64 = 128.0;
    for px in bytes.chunks_exact_mut(3) {
        let y = f64::from(px[0]);
        let cb = f64::from(px[1]);
        let cr = f64::from(px[2]);
        let r = round(y + 1.402 * (cr - HALF));
        let g = round(y - (0.114 * 1.772 * (cb - HALF) + 0.299 * 1.402 * (cr - HALF)) / 0.587);
        let b = round(y + 1.772 * (cb - HALF));
        px[0] = clamp(r);
        px[1] = clamp(g);
        px[2] = clamp(b);
    }
    Ok(())
}

/// `Rescaler::Rescale` → `RescaleFunction` (gdcmRescaler.cxx:24-42, :362-417):
/// `out = (TOut)(slope * in + intercept)` over the whole buffer read as a
/// stream of `stored`-typed scalars.
fn rescale_stream(
    input: &[u8],
    stored: ScalarType,
    output: ScalarType,
    intercept: f64,
    slope: f64,
) -> Vec<u8> {
    let in_width = scalar_width(stored);
    let mut out = Vec::with_capacity(input.len() / in_width * scalar_width(output));
    for chunk in input.chunks_exact(in_width) {
        let v = read_scalar(chunk, stored);
        write_scalar(&mut out, slope * v + intercept, output);
    }
    out
}

/// Bytes per scalar of a [`ScalarType`] this port can rescale to or from.
fn scalar_width(st: ScalarType) -> usize {
    match st {
        ScalarType::UInt8 | ScalarType::Int8 => 1,
        ScalarType::UInt16 | ScalarType::Int16 => 2,
        ScalarType::UInt32 | ScalarType::Int32 | ScalarType::Float32 => 4,
        ScalarType::Float64 => 8,
        // Never reached: 12-bit, 64-bit integer, Float16, SingleBit and Unknown
        // are all refused before rescale.
        _ => 1,
    }
}

/// Read one little-endian scalar as `f64` — the widening `in[i]` promotion in
/// `RescaleFunction`.
fn read_scalar(bytes: &[u8], st: ScalarType) -> f64 {
    match st {
        ScalarType::UInt8 => f64::from(bytes[0]),
        ScalarType::Int8 => f64::from(bytes[0] as i8),
        ScalarType::UInt16 => f64::from(u16::from_le_bytes([bytes[0], bytes[1]])),
        ScalarType::Int16 => f64::from(i16::from_le_bytes([bytes[0], bytes[1]])),
        ScalarType::UInt32 => {
            f64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        ScalarType::Int32 => {
            f64::from(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        ScalarType::Float32 => {
            f64::from(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        ScalarType::Float64 => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        _ => 0.0,
    }
}

/// Write `v` as a little-endian scalar of type `st` — the `(TOut)` cast in
/// `RescaleFunction`. A Rust `as` cast saturates rather than wrapping, which
/// matches C truncation for the in-range values `ComputeBestFit` guarantees.
fn write_scalar(out: &mut Vec<u8>, v: f64, st: ScalarType) {
    match st {
        ScalarType::UInt8 => out.push(v as u8),
        ScalarType::Int8 => out.push(v as i8 as u8),
        ScalarType::UInt16 => out.extend_from_slice(&(v as u16).to_le_bytes()),
        ScalarType::Int16 => out.extend_from_slice(&(v as i16).to_le_bytes()),
        ScalarType::UInt32 => out.extend_from_slice(&(v as u32).to_le_bytes()),
        ScalarType::Int32 => out.extend_from_slice(&(v as i32).to_le_bytes()),
        ScalarType::Float32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
        ScalarType::Float64 => out.extend_from_slice(&v.to_le_bytes()),
        _ => {}
    }
}

/// Reinterpret the finished little-endian byte buffer as the image's component
/// type.
fn bytes_to_pixel_buffer(bytes: Vec<u8>, component: PixelId) -> PixelBuffer {
    match component {
        PixelId::UInt8 => PixelBuffer::UInt8(bytes),
        PixelId::Int8 => PixelBuffer::Int8(bytes.iter().map(|&b| b as i8).collect()),
        PixelId::UInt16 => PixelBuffer::UInt16(
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
        ),
        PixelId::Int16 => PixelBuffer::Int16(
            bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect(),
        ),
        PixelId::UInt32 => PixelBuffer::UInt32(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        PixelId::Int32 => PixelBuffer::Int32(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        PixelId::Float32 => PixelBuffer::Float32(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        PixelId::Float64 => PixelBuffer::Float64(
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        ),
        // component_pixel_id only ever yields the scalar types above.
        other => unreachable!("unexpected DICOM component type {other:?}"),
    }
}

/// The `ImageIo` for DICOM files.
#[derive(Clone, Copy, Debug, Default)]
pub struct DicomImageIo;

impl ImageIo for DicomImageIo {
    /// `itkOverrideGetNameOfClassMacro(GDCMImageIO)` (itkGDCMImageIO.h:138) —
    /// the string `ioutils::CreateImageIOByName` accepts and
    /// `GetRegisteredImageIOs` lists. Upstream's class name is `GDCMImageIO`,
    /// not `DICOMImageIO`.
    fn name(&self) -> &'static str {
        "GDCMImageIO"
    }

    /// `.dcm`, `.DCM`, `.dicom`, `.DICOM` (itkGDCMImageIO.cxx:105-111).
    fn supported_read_extensions(&self) -> &'static [&'static str] {
        &[".dcm", ".DCM", ".dicom", ".DICOM"]
    }

    fn supported_write_extensions(&self) -> &'static [&'static str] {
        &[".dcm", ".DCM", ".dicom", ".DICOM"]
    }

    fn can_read_file(&self, path: &Path) -> bool {
        can_read_file(path)
    }

    /// `CanWriteFile` is `HasSupportedWriteExtension(name, false)` —
    /// case-**sensitive** (itkGDCMImageIO.cxx:814-826), like `TIFFImageIO`'s.
    fn can_write_file(&self, path: &Path) -> bool {
        crate::image_io::has_supported_extension(path, self.supported_write_extensions(), false)
    }

    fn read_information(&self, path: &Path) -> Result<ImageInformation> {
        read_information(path)
    }

    fn read(&self, path: &Path) -> Result<Image> {
        read(path)
    }

    fn write(&self, _image: &Image, _path: &Path, _options: &WriteOptions) -> Result<()> {
        Err(IoError::UnsupportedDicomFeature(
            "writing DICOM files is not implemented".into(),
        ))
    }
}

/// Read the header — `GDCMImageIO::ReadImageInformation`.
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    let h = read_header(path)?;
    Ok(ImageInformation {
        pixel_id: pixel_id_for(h.component, h.number_of_components),
        // `SetNumberOfDimensions(3)` in the constructor, never lowered.
        dimension: 3,
        number_of_components: h.number_of_components,
        size: h.size.to_vec(),
        spacing: h.spacing.to_vec(),
        origin: h.origin.to_vec(),
        direction: h.direction,
        metadata: h.metadata,
    })
}

/// `ImageReaderBase::GetPixelIDFromImageIO` (sitkImageReaderBase.cxx:215-227):
/// one component loads as a scalar, anything else as a vector image.
fn pixel_id_for(component: PixelId, number_of_components: usize) -> PixelId {
    if number_of_components == 1 {
        component
    } else {
        component.vector_id()
    }
}

/// Read the image — `GDCMImageIO::Read`.
pub fn read(path: &Path) -> Result<Image> {
    let h = read_header(path)?;
    let buffer = decode_pixels(&h)?;
    let mut image = if h.number_of_components == 1 {
        Image::from_parts(
            buffer,
            h.size.to_vec(),
            h.spacing.to_vec(),
            h.origin.to_vec(),
            h.direction.clone(),
        )?
    } else {
        Image::from_parts_vector(
            buffer,
            h.number_of_components,
            h.size.to_vec(),
            h.spacing.to_vec(),
            h.origin.to_vec(),
            h.direction.clone(),
        )?
    };
    for (key, value) in &h.metadata {
        image.set_meta_data(key, value);
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // --- Byte-level DICOM fixture builder (Explicit VR Little Endian) --------

    /// VRs encoded with a 2-byte reserved field and a 32-bit length.
    fn is_long_vr(vr: &[u8; 2]) -> bool {
        matches!(
            vr,
            b"OB" | b"OW" | b"OF" | b"OD" | b"OL" | b"OV" | b"SQ" | b"UC" | b"UR" | b"UT" | b"UN"
        )
    }

    /// One Explicit VR Little Endian data element, value auto-padded to even
    /// length (`\0` for UI/binary, space for text).
    fn elem(group: u16, element: u16, vr: &[u8; 2], value: &[u8]) -> Vec<u8> {
        let mut v = value.to_vec();
        if v.len() % 2 == 1 {
            v.push(if matches!(vr, b"UI" | b"OB" | b"OW" | b"UN") {
                0x00
            } else {
                b' '
            });
        }
        let mut out = Vec::new();
        out.extend_from_slice(&group.to_le_bytes());
        out.extend_from_slice(&element.to_le_bytes());
        out.extend_from_slice(vr);
        if is_long_vr(vr) {
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
        } else {
            out.extend_from_slice(&(v.len() as u16).to_le_bytes());
        }
        out.extend_from_slice(&v);
        out
    }

    fn us(v: u16) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    /// The (0002) file-meta group with its computed group length.
    fn meta_group(ts: &str, sop_class: &str) -> Vec<u8> {
        let mut g = Vec::new();
        g.extend(elem(0x0002, 0x0001, b"OB", &[0x00, 0x01]));
        g.extend(elem(0x0002, 0x0002, b"UI", sop_class.as_bytes()));
        g.extend(elem(0x0002, 0x0003, b"UI", b"1.2.3.4.5"));
        g.extend(elem(0x0002, 0x0010, b"UI", ts.as_bytes()));
        let mut out = elem(0x0002, 0x0000, b"UL", &(g.len() as u32).to_le_bytes());
        out.extend(g);
        out
    }

    /// A full file: 128-byte preamble, `DICM`, meta group, then the dataset.
    fn dicom_file(ts: &str, sop_class: &str, dataset: &[u8]) -> Vec<u8> {
        let mut f = vec![0u8; 128];
        f.extend_from_slice(b"DICM");
        f.extend(meta_group(ts, sop_class));
        f.extend_from_slice(dataset);
        f
    }

    fn write_temp(bytes: &[u8], name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("sitk_dicom_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
        path
    }

    const EVR_LE: &str = "1.2.840.10008.1.2.1";
    const CT: &str = "1.2.840.10008.5.1.4.1.1.2";
    const SC: &str = "1.2.840.10008.5.1.4.1.1.7";
    const ENHANCED_CT: &str = "1.2.840.10008.5.1.4.1.1.2.1";
    const JPEG_BASELINE: &str = "1.2.840.10008.1.2.4.50";

    /// A grayscale 16-bit CT slice: `cols`×`rows`, MONOCHROME2, the given pixel
    /// words, Pixel Spacing `0.5\0.75`, position `1\2\3`, slice spacing `2.5`.
    fn ct_dataset(cols: u16, rows: u16, pixels: &[u16]) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", CT.as_bytes()));
        d.extend(elem(0x0008, 0x0060, b"CS", b"CT"));
        d.extend(elem(0x0018, 0x0088, b"DS", b"2.5"));
        d.extend(elem(0x0020, 0x0032, b"DS", b"1\\2\\3"));
        d.extend(elem(0x0020, 0x0037, b"DS", b"1\\0\\0\\0\\1\\0"));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(rows)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(cols)));
        d.extend(elem(0x0028, 0x0030, b"DS", b"0.5\\0.75"));
        d.extend(elem(0x0028, 0x0100, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(15)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        let mut pd = Vec::new();
        for &p in pixels {
            pd.extend_from_slice(&p.to_le_bytes());
        }
        d.extend(elem(0x7fe0, 0x0010, b"OW", &pd));
        d
    }

    #[test]
    fn set_bits_stored_remaps_fujifilm_bitmask_and_guards() {
        // FUJIFILM CR + MONO1 emit BitsStored as a bitmask; GDCM reads the value
        // they meant (gdcmPixelFormat.h:134-149). Without the remap, min()/max()
        // compute `1u64 << 0xffff` (dicom.rs:691,699,700) — a debug panic /
        // release garbage that mis-drives pixel-type promotion.
        let mut pf = PixelFormat {
            samples_per_pixel: 1,
            bits_allocated: 0,
            bits_stored: 0,
            high_bit: 0,
            pixel_representation: 0,
        };
        pf.set_bits_allocated(0xffff); // -> 16
        pf.set_bits_stored(0xffff); // -> 16, guarded by BitsAllocated
        assert_eq!(pf.bits_stored, 16);
        assert_eq!(pf.high_bit, 15);
        assert_eq!(pf.max(), (1i64 << 16) - 1);
        assert_eq!(pf.min(), 0);

        // Guard: a BitsStored exceeding BitsAllocated is dropped outright.
        let mut pf2 = PixelFormat {
            samples_per_pixel: 1,
            bits_allocated: 8,
            bits_stored: 8,
            high_bit: 7,
            pixel_representation: 0,
        };
        pf2.set_bits_stored(16);
        assert_eq!(pf2.bits_stored, 8, "bs > BitsAllocated is rejected");
        assert_eq!(pf2.high_bit, 7);

        // Zero is dropped (unknown / absent).
        let mut pf3 = pf2;
        pf3.set_bits_stored(0);
        assert_eq!(pf3.bits_stored, 8);
    }

    #[test]
    fn read_information_ct_slice_is_three_dimensional() {
        let bytes = dicom_file(EVR_LE, CT, &ct_dataset(4, 2, &[0; 8]));
        let path = write_temp(&bytes, "ct_info.dcm");
        let info = read_information(&path).unwrap();

        assert_eq!(info.dimension, 3, "GDCMImageIO forces 3 dimensions");
        assert_eq!(info.size, vec![4, 2, 1]);
        assert_eq!(info.pixel_id, PixelId::UInt16);
        assert_eq!(info.number_of_components, 1);
        // Pixel Spacing 0.5\0.75 is stored row\column and swapped to column\row.
        assert_eq!(info.spacing, vec![0.75, 0.5, 2.5]);
        assert_eq!(info.origin, vec![1.0, 2.0, 3.0]);
        assert_eq!(
            info.direction,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn read_information_multiframe_sets_depth() {
        let mut d = ct_dataset(2, 2, &[0; 12]);
        // Re-declare with NumberOfFrames = 3 and 3 frames of pixel data.
        d = {
            let mut nd = Vec::new();
            nd.extend(elem(0x0008, 0x0016, b"UI", CT.as_bytes()));
            nd.extend(elem(0x0028, 0x0008, b"IS", b"3"));
            nd.extend(elem(0x0028, 0x0002, b"US", &us(1)));
            nd.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
            nd.extend(elem(0x0028, 0x0010, b"US", &us(2)));
            nd.extend(elem(0x0028, 0x0011, b"US", &us(2)));
            nd.extend(elem(0x0028, 0x0100, b"US", &us(16)));
            nd.extend(elem(0x0028, 0x0101, b"US", &us(16)));
            nd.extend(elem(0x0028, 0x0102, b"US", &us(15)));
            nd.extend(elem(0x0028, 0x0103, b"US", &us(0)));
            let mut pd = Vec::new();
            for _ in 0..12 {
                pd.extend_from_slice(&0u16.to_le_bytes());
            }
            nd.extend(elem(0x7fe0, 0x0010, b"OW", &pd));
            let _ = &d;
            nd
        };
        let bytes = dicom_file(EVR_LE, CT, &d);
        let path = write_temp(&bytes, "ct_multiframe.dcm");
        let info = read_information(&path).unwrap();
        assert_eq!(info.size, vec![2, 2, 3]);
    }

    #[test]
    fn read_ct_pixels_verbatim_without_rescale() {
        let pixels: Vec<u16> = vec![100, 200, 300, 400];
        let bytes = dicom_file(EVR_LE, CT, &ct_dataset(2, 2, &pixels));
        let path = write_temp(&bytes, "ct_pixels.dcm");
        let image = read(&path).unwrap();
        match image.buffer() {
            PixelBuffer::UInt16(v) => assert_eq!(v, &pixels),
            other => panic!("expected UInt16 buffer, got {other:?}"),
        }
    }

    #[test]
    fn rescale_promotes_component_type_and_applies() {
        // Bits Stored 12, unsigned; intercept -1024 pushes the range negative,
        // so ComputeBestFit lands on Int16.
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", CT.as_bytes()));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(2)));
        d.extend(elem(0x0028, 0x0100, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(12)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(11)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        d.extend(elem(0x0028, 0x1052, b"DS", b"-1024"));
        d.extend(elem(0x0028, 0x1053, b"DS", b"1"));
        let mut pd = Vec::new();
        pd.extend_from_slice(&1024u16.to_le_bytes());
        pd.extend_from_slice(&2000u16.to_le_bytes());
        d.extend(elem(0x7fe0, 0x0010, b"OW", &pd));

        let bytes = dicom_file(EVR_LE, CT, &d);
        let path = write_temp(&bytes, "ct_rescale.dcm");

        let info = read_information(&path).unwrap();
        assert_eq!(info.pixel_id, PixelId::Int16);

        let image = read(&path).unwrap();
        match image.buffer() {
            PixelBuffer::Int16(v) => assert_eq!(v, &vec![0i16, 976i16]),
            other => panic!("expected Int16 buffer, got {other:?}"),
        }
    }

    #[test]
    fn monochrome1_eight_bit_is_inverted() {
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", CT.as_bytes()));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME1"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(2)));
        d.extend(elem(0x0028, 0x0100, b"US", &us(8)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(8)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(7)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        d.extend(elem(0x7fe0, 0x0010, b"OB", &[10, 200]));

        let bytes = dicom_file(EVR_LE, CT, &d);
        let path = write_temp(&bytes, "mono1.dcm");
        let image = read(&path).unwrap();
        match image.buffer() {
            // 255 - value.
            PixelBuffer::UInt8(v) => assert_eq!(v, &vec![245u8, 55u8]),
            other => panic!("expected UInt8 buffer, got {other:?}"),
        }
    }

    fn rgb_dataset(planar: u16, pixel_data: &[u8]) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", SC.as_bytes()));
        d.extend(elem(0x0028, 0x0002, b"US", &us(3)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"RGB"));
        d.extend(elem(0x0028, 0x0006, b"US", &us(planar)));
        d.extend(elem(0x0028, 0x0010, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(2)));
        d.extend(elem(0x0028, 0x0100, b"US", &us(8)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(8)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(7)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        d.extend(elem(0x7fe0, 0x0010, b"OB", pixel_data));
        d
    }

    #[test]
    fn rgb_interleaved_loads_as_vector() {
        // 2×1 RGB, planar 0: [R0,G0,B0, R1,G1,B1].
        let bytes = dicom_file(EVR_LE, SC, &rgb_dataset(0, &[10, 20, 30, 40, 50, 60]));
        let path = write_temp(&bytes, "rgb0.dcm");
        let info = read_information(&path).unwrap();
        assert_eq!(info.number_of_components, 3);
        assert_eq!(info.pixel_id, PixelId::VectorUInt8);

        let image = read(&path).unwrap();
        assert_eq!(image.number_of_components_per_pixel(), 3);
        match image.buffer() {
            PixelBuffer::UInt8(v) => assert_eq!(v, &vec![10, 20, 30, 40, 50, 60]),
            other => panic!("expected UInt8 buffer, got {other:?}"),
        }
    }

    #[test]
    fn rgb_planar_configuration_one_is_deinterleaved() {
        // Same pixels stored plane-by-plane: [R0,R1, G0,G1, B0,B1].
        let bytes = dicom_file(EVR_LE, SC, &rgb_dataset(1, &[10, 40, 20, 50, 30, 60]));
        let path = write_temp(&bytes, "rgb1.dcm");
        let image = read(&path).unwrap();
        match image.buffer() {
            PixelBuffer::UInt8(v) => assert_eq!(v, &vec![10, 20, 30, 40, 50, 60]),
            other => panic!("expected UInt8 buffer, got {other:?}"),
        }
    }

    #[test]
    fn metadata_dictionary_keys_and_values() {
        let mut d = ct_dataset(2, 1, &[0, 0]);
        // Append a public OB element to exercise the base64 branch, and a
        // multi-valued US to exercise StringFilter's backslash join.
        d.extend(elem(0x0028, 0x2000, b"OB", &[1, 2, 3, 4])); // ICC Profile
        let bytes = dicom_file(EVR_LE, CT, &d);
        let path = write_temp(&bytes, "meta.dcm");
        let info = read_information(&path).unwrap();

        // "gggg|eeee" lowercase keys. `StringFilter::ToString` truncates an
        // ASCII value only at the first NUL, so the even-length space padding
        // DICOM adds to odd-length values survives (gdcmStringFilter.cxx:418-421).
        assert_eq!(
            info.metadata.get("0028|0011").map(String::as_str),
            Some("2")
        );
        assert_eq!(
            info.metadata.get("0028|0004").map(String::as_str),
            Some("MONOCHROME2 ")
        );
        assert_eq!(
            info.metadata.get("0020|0032").map(String::as_str),
            Some("1\\2\\3 ")
        );
        // Binary VR → base64.
        assert_eq!(
            info.metadata.get("0028|2000").map(String::as_str),
            Some("AQIDBA==")
        );
        // Pixel Data is never emitted.
        assert!(!info.metadata.contains_key("7fe0|0010"));
    }

    #[test]
    fn can_read_valid_file_and_rejects_garbage() {
        let bytes = dicom_file(EVR_LE, CT, &ct_dataset(2, 1, &[0, 0]));
        let path = write_temp(&bytes, "canread.dcm");
        assert!(DicomImageIo.can_read_file(&path));

        let junk = write_temp(&[0u8; 64], "junk.bin");
        assert!(!DicomImageIo.can_read_file(&junk));
    }

    #[test]
    fn encapsulated_transfer_syntax_is_refused() {
        // A JPEG-baseline file with an encapsulated (undefined-length) Pixel Data.
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", CT.as_bytes()));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0100, b"US", &us(8)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(8)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(7)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        // Pixel Data (7fe0,0010) OB, undefined length, one fragment.
        d.extend_from_slice(&[0xe0, 0x7f, 0x10, 0x00]);
        d.extend_from_slice(b"OB");
        d.extend_from_slice(&[0, 0]);
        d.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        // Basic offset table item, empty.
        d.extend_from_slice(&[0xfe, 0xff, 0x00, 0xe0, 0, 0, 0, 0]);
        // Fragment item, 2 bytes.
        d.extend_from_slice(&[0xfe, 0xff, 0x00, 0xe0, 0x02, 0, 0, 0, 0xaa, 0xbb]);
        // Sequence delimiter.
        d.extend_from_slice(&[0xfe, 0xff, 0xdd, 0xe0, 0, 0, 0, 0]);

        let bytes = dicom_file(JPEG_BASELINE, CT, &d);
        let path = write_temp(&bytes, "jpeg.dcm");
        assert!(matches!(
            read_information(&path),
            Err(IoError::UnsupportedDicomFeature(_))
        ));
    }

    #[test]
    fn sequence_driven_sop_class_is_refused() {
        // The dataset's own SOP Class UID must name the enhanced class — it wins
        // over the file-meta UID in `MediaStorage::SetFromFile`.
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", ENHANCED_CT.as_bytes()));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(2)));
        d.extend(elem(0x0028, 0x0100, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(15)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        d.extend(elem(0x7fe0, 0x0010, b"OW", &[0, 0, 0, 0]));

        let bytes = dicom_file(EVR_LE, ENHANCED_CT, &d);
        let path = write_temp(&bytes, "enhanced_ct.dcm");
        assert!(matches!(
            read_information(&path),
            Err(IoError::UnsupportedDicomFeature(_))
        ));
    }

    #[test]
    fn z_spacing_tag_present_but_unparsable_is_refused() {
        // A CT with (0018,0088) Spacing Between Slices present but non-numeric.
        // GDCM's `el.Read` pushes nothing, leaving `sp` two long, then reads
        // `sp[2]` out of bounds — no deterministic value to reproduce (§1.70).
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", CT.as_bytes()));
        d.extend(elem(0x0018, 0x0088, b"DS", b"abc"));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0100, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(15)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        d.extend(elem(0x7fe0, 0x0010, b"OW", &[0, 0]));

        let bytes = dicom_file(EVR_LE, CT, &d);
        let path = write_temp(&bytes, "zspacing_unparsable.dcm");
        assert!(matches!(
            read_information(&path),
            Err(IoError::MalformedDicom(_))
        ));
    }

    #[test]
    fn ultrasound_single_valued_pixel_spacing_is_refused() {
        // UltrasoundImageStorage takes ITK's ultrasound spacing override, which
        // asserts (0028,0030) has two values and reads both; a single value
        // leaves the second slot uninitialised, so this port refuses (§1.71).
        const US_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.6.1";
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", US_STORAGE.as_bytes()));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0030, b"DS", b"0.5"));
        d.extend(elem(0x0028, 0x0100, b"US", &us(8)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(8)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(7)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        d.extend(elem(0x7fe0, 0x0010, b"OW", &[0, 0]));

        let bytes = dicom_file(EVR_LE, US_STORAGE, &d);
        let path = write_temp(&bytes, "us_single_spacing.dcm");
        assert!(matches!(
            read_information(&path),
            Err(IoError::MalformedDicom(_))
        ));
    }

    #[test]
    fn single_bit_non_byte_aligned_width_is_refused() {
        // BitsAllocated = 1 → SINGLEBIT. Columns = 3 is not a multiple of 8, so
        // ITK's bit-expansion loop would read past the per-row byte padding GDCM
        // adds — the wrong bytes for the row. Refused at decode time (§1.72).
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", CT.as_bytes()));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0011, b"US", &us(3)));
        d.extend(elem(0x0028, 0x0100, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(0)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        d.extend(elem(0x7fe0, 0x0010, b"OB", &[0b0000_0101]));

        let bytes = dicom_file(EVR_LE, CT, &d);
        let path = write_temp(&bytes, "singlebit_unaligned.dcm");
        // The header reads fine (a single-bit image reports UInt8); the refusal
        // is at pixel decode.
        assert!(read_information(&path).is_ok());
        assert!(matches!(
            read(&path),
            Err(IoError::UnsupportedDicomFeature(_))
        ));
    }

    #[test]
    fn write_is_refused() {
        let bytes = dicom_file(EVR_LE, CT, &ct_dataset(2, 1, &[0, 0]));
        let path = write_temp(&bytes, "for_write.dcm");
        let image = read(&path).unwrap();
        let out = write_temp(&[], "out.dcm");
        assert!(matches!(
            DicomImageIo.write(&image, &out, &WriteOptions::default()),
            Err(IoError::UnsupportedDicomFeature(_))
        ));
    }

    #[test]
    fn io_reports_gdcm_class_name_and_extensions() {
        assert_eq!(DicomImageIo.name(), "GDCMImageIO");
        assert_eq!(
            DicomImageIo.supported_read_extensions(),
            &[".dcm", ".DCM", ".dicom", ".DICOM"]
        );
    }

    // --- Unit tests for the arithmetic / encoding helpers -------------------

    #[test]
    fn native_transfer_syntaxes_only() {
        assert!(is_native_transfer_syntax("1.2.840.10008.1.2"));
        assert!(is_native_transfer_syntax("1.2.840.10008.1.2.1"));
        assert!(is_native_transfer_syntax("1.2.840.10008.1.2.1.99"));
        assert!(is_native_transfer_syntax("1.2.840.10008.1.2.2"));
        assert!(!is_native_transfer_syntax("1.2.840.10008.1.2.4.50"));
        assert!(!is_native_transfer_syntax("1.2.840.10008.1.2.4.90"));
        assert!(!is_native_transfer_syntax("1.2.840.10008.1.2.5"));
    }

    #[test]
    fn media_storage_uid_lookup() {
        assert_eq!(
            media_storage_from_uid("1.2.840.10008.5.1.4.1.1.2"),
            MediaStorage::CtImageStorage
        );
        assert_eq!(
            media_storage_from_uid("1.2.840.10008.5.1.4.1.1.2.1"),
            MediaStorage::EnhancedCtImageStorage
        );
        // Trailing space is trimmed before the lookup.
        assert_eq!(
            media_storage_from_uid("1.2.840.10008.5.1.4.1.1.4 "),
            MediaStorage::MrImageStorage
        );
        assert_eq!(media_storage_from_uid("9.9.9"), MediaStorage::Other);
    }

    #[test]
    fn best_fit_matches_gdcm_for_ct_rescale() {
        let pf = PixelFormat {
            samples_per_pixel: 1,
            bits_allocated: 16,
            bits_stored: 12,
            high_bit: 11,
            pixel_representation: 0,
        };
        // slope 1, intercept -1024 over [0, 4095] → [-1024, 3071] → Int16.
        assert_eq!(
            compute_intercept_slope_pixel_type(pf, -1024.0, 1.0),
            ScalarType::Int16
        );
        // A non-integral slope forces Float64.
        assert_eq!(
            compute_intercept_slope_pixel_type(pf, 0.0, 2.5),
            ScalarType::Float64
        );
    }

    #[test]
    fn pixel_type_larger_than_output_uses_signed_yardstick() {
        // A signed input may widen into an unsigned output of the same width.
        assert!(!pixel_type_larger_than_output(
            ScalarType::Int16,
            ScalarType::UInt16
        ));
        assert!(pixel_type_larger_than_output(
            ScalarType::UInt16,
            ScalarType::UInt8
        ));
    }

    #[test]
    fn default_float_formatting_matches_ostream() {
        assert_eq!(format_default_float(0.0), "0");
        assert_eq!(format_default_float(1.0), "1");
        assert_eq!(format_default_float(1.5), "1.5");
        assert_eq!(format_default_float(0.75), "0.75");
        // Six significant digits, trailing zeros stripped.
        assert_eq!(format_default_float(1234.5678), "1234.57");
        assert_eq!(format_default_float(0.123456789), "0.123457");
        assert_eq!(format_default_float(100000.0), "100000");
        // Beyond six significant digits switches to exponential.
        assert_eq!(format_default_float(1_000_000.0), "1e+06");
        assert_eq!(format_default_float(0.0001), "0.0001");
        assert_eq!(format_default_float(0.00001), "1e-05");
        assert_eq!(format_default_float(-2.5), "-2.5");
    }

    #[test]
    fn base64_matches_kwsys() {
        assert_eq!(base64_encode(&[1, 2, 3, 4]), "AQIDBA==");
        assert_eq!(base64_encode(b"M"), "TQ==");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn scalar_type_from_bits_and_representation() {
        let mut pf = PixelFormat {
            samples_per_pixel: 1,
            bits_allocated: 16,
            bits_stored: 12,
            high_bit: 11,
            pixel_representation: 0,
        };
        // Bits Stored is ignored: 16/12 unsigned is UInt16, not UInt12.
        assert_eq!(pf.scalar_type(), ScalarType::UInt16);
        pf.pixel_representation = 1;
        assert_eq!(pf.scalar_type(), ScalarType::Int16);
    }

    #[test]
    fn ycbcr_to_rgb_black_and_grey() {
        // Y=0, Cb=Cr=128 → black.
        let mut buf = vec![0u8, 128, 128];
        ycbcr_to_rgb(&mut buf).unwrap();
        assert_eq!(buf, vec![0, 0, 0]);
        // Y=128, Cb=Cr=128 → mid grey (R=G=B=128).
        let mut buf = vec![128u8, 128, 128];
        ycbcr_to_rgb(&mut buf).unwrap();
        assert_eq!(buf, vec![128, 128, 128]);
    }

    #[test]
    fn read_image_flips_a_negative_z_spacing_dicom_and_records_the_originals() {
        // A CT slice whose Spacing Between Slices (0018,0088) is negative. GDCM
        // preserves that sign at the IO layer ("Spacing may be negative at this
        // point, will be fixed below", itkGDCMImageIO.cxx:703-704), leaving the
        // reader's `normalize_reader_geometry` to flip it. This is the
        // end-to-end pin: a real DICOM negative Z-spacing must survive through
        // `read_image` to a positive spacing with the Z direction column
        // negated and the raw values recorded under `ITK_original_*`. Reverting
        // the IO-layer sign preservation (the negative `z` kept in
        // `read_header`) makes this fail.
        let mut d = Vec::new();
        d.extend(elem(0x0008, 0x0016, b"UI", CT.as_bytes()));
        d.extend(elem(0x0008, 0x0060, b"CS", b"CT"));
        d.extend(elem(0x0018, 0x0088, b"DS", b"-2.5")); // negative Spacing Between Slices
        d.extend(elem(0x0020, 0x0032, b"DS", b"1\\2\\3"));
        d.extend(elem(0x0020, 0x0037, b"DS", b"1\\0\\0\\0\\1\\0"));
        d.extend(elem(0x0028, 0x0002, b"US", &us(1)));
        d.extend(elem(0x0028, 0x0004, b"CS", b"MONOCHROME2"));
        d.extend(elem(0x0028, 0x0010, b"US", &us(2))); // rows
        d.extend(elem(0x0028, 0x0011, b"US", &us(4))); // cols
        d.extend(elem(0x0028, 0x0030, b"DS", b"0.5\\0.75"));
        d.extend(elem(0x0028, 0x0100, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0101, b"US", &us(16)));
        d.extend(elem(0x0028, 0x0102, b"US", &us(15)));
        d.extend(elem(0x0028, 0x0103, b"US", &us(0)));
        d.extend(elem(0x7fe0, 0x0010, b"OW", &[0u8; 16])); // 4×2 × 2 bytes
        let bytes = dicom_file(EVR_LE, CT, &d);
        let path = write_temp(&bytes, "negative_z_spacing.dcm");

        let image = crate::read_image(&path).unwrap();

        // Flipped positive: |−2.5| with the Z direction *column* negated.
        assert_eq!(image.spacing(), &[0.75, 0.5, 2.5]);
        assert_eq!(
            image.direction(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        );
        // The raw negative sign is preserved in the recorded originals.
        assert_eq!(
            image.meta_data("ITK_original_spacing"),
            Some("0.75 0.5 -2.5")
        );
        assert_eq!(
            image.meta_data("ITK_original_direction"),
            Some("1 0 0 0 1 0 0 0 1")
        );
    }
}
