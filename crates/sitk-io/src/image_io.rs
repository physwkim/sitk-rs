//! The `ImageIo` seam: one trait per file format plus the registry that picks
//! which one handles a given path.
//!
//! This is ITK's `itk::ImageIOFactory` protocol, minus the object factory.
//! Upstream, every format ships an `itk::ImageIOBase` subclass and registers a
//! factory; `ImageIOFactory::CreateImageIO(path, mode)` instantiates all of
//! them and asks each whether it can handle the file
//! (itkImageIOFactory.cxx:59-108). SimpleITK's `ImageFileReader` /
//! `ImageFileWriter` then delegate to whichever one answered
//! (sitkImageReaderBase.cxx:74-115, sitkImageFileWriter.cxx:195-219), and
//! `ioutils::GetRegisteredImageIOs` lists them by class name
//! (sitkImageIOUtilities.cxx:59-77).
//!
//! Here the "factory" is a static slice ([`registry`]) because the set of
//! formats is fixed at compile time. Everything else — the two-phase probe
//! order, the class-name lookup, the "can read" / "can write" split — matches
//! upstream.
//!
//! # Probe order: extension first, content always
//!
//! ITK 6's `CreateImageIO` runs two phases (itkImageIOFactory.cxx:70-107):
//!
//! 1. every IO whose *advertised extension list* matches the path is asked
//!    `CanReadFile` / `CanWriteFile`; the first one that says yes wins. An IO
//!    that matches by extension but answers no is struck off.
//! 2. the IOs left over — those that never advertised the extension — are asked
//!    the same question, so a format with no extension of its own can still
//!    rescue the file.
//!
//! So the extension only *orders* the probe: it never authorises an IO on its
//! own. A `.mhd`-named file whose content is not a MetaImage header is claimed
//! by nobody and the read fails with [`IoError::NoReaderFound`]. Reading is
//! content-checked, writing is extension-checked (`MetaImageIO::CanWriteFile`
//! is exactly `HasSupportedWriteExtension`, itkMetaImageIO.cxx:370-380).
//!
//! Phase 2 cannot currently rescue anything, because [`MetaImageIo`]'s own
//! `can_read_file` re-checks the extension itself — see
//! [`crate::meta_image`]'s module docs for that upstream quirk.
//!
//! [`MetaImageIo`]: crate::meta_image::MetaImageIo

use std::collections::BTreeMap;
use std::path::Path;

use sitk_core::{Image, PixelId};

use crate::error::{IoError, Result};
use crate::gipl::GiplImageIo;
use crate::meta_image::MetaImageIo;
use crate::nifti::NiftiImageIo;
use crate::nrrd::NrrdImageIo;
use crate::png::PngImageIo;
use crate::vtk::VtkImageIo;
use crate::writer::WriteOptions;

/// Which of [`ImageIo::can_read_file`] / [`ImageIo::can_write_file`] the
/// registry probe should use. `itk::IOFileModeEnum`
/// (itkImageIOBase.h, `IOFileModeEnum::ReadMode` / `WriteMode`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileMode {
    /// Probe with [`ImageIo::can_read_file`].
    Read,
    /// Probe with [`ImageIo::can_write_file`].
    Write,
}

/// The image information an [`ImageIo`] can report without loading pixels.
///
/// This is SimpleITK's post-`ReadImageInformation` accessor set
/// (sitkImageFileReader.h:126-141) plus the meta-data dictionary
/// (sitkImageFileReader.cxx:165). Every field comes from the `itk::ImageIO`,
/// so `number_of_components` is the *file's* component count — for a MetaImage
/// that is `ElementNumberOfChannels`, which is `2` for a file holding complex
/// samples even though SimpleITK's `Image` would report `1` component per
/// pixel. `pixel_id` is already translated into this crate's [`PixelId`], as
/// upstream's `GetPixelID` is (sitkImageFileReader.h:120-124).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageInformation {
    /// The pixel type the file will load as.
    pub pixel_id: PixelId,
    /// Number of image axes.
    pub dimension: usize,
    /// The `itk::ImageIO`'s component count, *not* `Image`'s.
    pub number_of_components: usize,
    /// Size along each axis, `dimension` entries.
    pub size: Vec<usize>,
    /// Spacing along each axis, `dimension` entries.
    pub spacing: Vec<f64>,
    /// Origin, `dimension` entries.
    pub origin: Vec<f64>,
    /// Row-major direction cosines, `dimension * dimension` entries.
    pub direction: Vec<f64>,
    /// The file's meta-data dictionary, flattened to strings exactly as
    /// SimpleITK flattens `itk::MetaDataDictionary`.
    pub metadata: BTreeMap<String, String>,
}

impl ImageInformation {
    /// The dictionary keys, in ascending byte order — `GetMetaDataKeys`.
    pub fn meta_data_keys(&self) -> Vec<&str> {
        self.metadata.keys().map(String::as_str).collect()
    }

    /// Whether `key` is present — `HasMetaDataKey`.
    pub fn has_meta_data_key(&self, key: &str) -> bool {
        self.metadata.contains_key(key)
    }

    /// The value stored under `key` — `GetMetaData`, which throws on an absent
    /// key where this returns `None` (as [`sitk_core::Image::meta_data`] does).
    pub fn meta_data(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// One image file format: `itk::ImageIOBase`.
///
/// Implementors are zero-sized and live in [`registry`] for the lifetime of the
/// program, mirroring the singleton `itk::ImageIOBase` instances the object
/// factory hands out.
pub trait ImageIo: Sync {
    /// The IO's class name, as `GetNameOfClass` reports it (`"MetaImageIO"`).
    /// This is the string [`registered_image_ios`] lists and
    /// [`crate::ImageFileWriter::set_image_io`] accepts.
    fn name(&self) -> &'static str;

    /// Extensions this IO advertises for reading, each including the dot and
    /// spelled lowercase — `GetSupportedReadExtensions`. Used only to *order*
    /// the registry probe; [`ImageIo::can_read_file`] is what decides.
    fn supported_read_extensions(&self) -> &'static [&'static str];

    /// Extensions this IO advertises for writing — `GetSupportedWriteExtensions`.
    fn supported_write_extensions(&self) -> &'static [&'static str];

    /// Whether this IO can read `path`, judged by opening the file and looking
    /// at its content — `CanReadFile`. An IO may additionally require an
    /// extension of its own; `MetaImageIO` does.
    fn can_read_file(&self, path: &Path) -> bool;

    /// Whether this IO can write `path`, judged by extension only —
    /// `CanWriteFile`. The default is `HasSupportedWriteExtension(path, true)`,
    /// which is `MetaImageIO::CanWriteFile` verbatim
    /// (itkMetaImageIO.cxx:370-380).
    fn can_write_file(&self, path: &Path) -> bool {
        has_supported_extension(path, self.supported_write_extensions(), true)
    }

    /// Read the header: geometry, pixel type, and meta-data dictionary, with no
    /// pixel data — `ReadImageInformation`.
    fn read_information(&self, path: &Path) -> Result<ImageInformation>;

    /// Read the whole image, dictionary included.
    fn read(&self, path: &Path) -> Result<Image>;

    /// Write `image` to `path`.
    ///
    /// `options` carries the `m_UseCompression` / `m_CompressionLevel` an
    /// `itk::ImageIOBase` would hold as member state; an IO that cannot
    /// compress ignores both, as `SetUseCompression` is only ever a request
    /// (sitkImageFileWriter.h:80-85).
    fn write(&self, image: &Image, path: &Path, options: &WriteOptions) -> Result<()>;
}

/// `ImageIOBase::HasSupportedReadExtension` / `...WriteExtension`: does `path`
/// end with any of `extensions`?
///
/// `ignore_case` mirrors upstream's flag; `CreateImageIO` always passes `true`
/// (itkImageIOFactory.cxx:62).
pub fn has_supported_extension(path: &Path, extensions: &[&str], ignore_case: bool) -> bool {
    let name = path.as_os_str().to_string_lossy();
    extensions.iter().any(|ext| {
        if ignore_case {
            name.len() >= ext.len() && name[name.len() - ext.len()..].eq_ignore_ascii_case(ext)
        } else {
            name.ends_with(ext)
        }
    })
}

static META_IMAGE_IO: MetaImageIo = MetaImageIo;
static NRRD_IMAGE_IO: NrrdImageIo = NrrdImageIo;
static NIFTI_IMAGE_IO: NiftiImageIo = NiftiImageIo;
static GIPL_IMAGE_IO: GiplImageIo = GiplImageIo;
static VTK_IMAGE_IO: VtkImageIo = VtkImageIo;
static PNG_IMAGE_IO: PngImageIo = PngImageIo;

/// Every registered [`ImageIo`], in registration order.
///
/// `ObjectFactoryBase::CreateAllInstance("itkImageIOBase")`'s result, which
/// both `CreateImageIO` and `GetRegisteredImageIOs` iterate. Probe order is
/// this order, so an earlier entry wins a tie.
///
/// [`MetaImageIo`], [`NrrdImageIo`], [`NiftiImageIo`], [`VtkImageIo`] and
/// [`PngImageIo`] advertise disjoint extension sets, so their relative order
/// decides nothing for a named file. It does matter in phase 2 of
/// [`create_image_io`], where an extension-less path is offered to every IO in
/// turn: `MetaImageIo::can_read_file` re-checks the extension and declines,
/// `NrrdImageIo::can_read_file` probes the magic and may claim it, and
/// `NiftiImageIo::can_read_file` then resolves the file through
/// `nifti_findhdrname` and may claim it.
///
/// [`GiplImageIo`] advertises *no* extensions — `itk::GiplImageIO`'s
/// constructor registers none — so it is reachable only from phase 2, where its
/// own `CheckExtension` gate makes it claim `.gipl` and `.gipl.gz` and nothing
/// else.
///
/// [`VtkImageIo`]: crate::vtk::VtkImageIo
/// [`GiplImageIo`]: crate::gipl::GiplImageIo
/// [`PngImageIo`]: crate::png::PngImageIo
pub fn registry() -> &'static [&'static dyn ImageIo] {
    const IOS: &[&dyn ImageIo] = &[
        &META_IMAGE_IO,
        &NRRD_IMAGE_IO,
        &NIFTI_IMAGE_IO,
        &GIPL_IMAGE_IO,
        &VTK_IMAGE_IO,
        &PNG_IMAGE_IO,
    ];
    IOS
}

/// The class names of every registered [`ImageIo`] —
/// `ImageFileWriter::GetRegisteredImageIOs` / `ioutils::GetRegisteredImageIOs`
/// (sitkImageIOUtilities.cxx:59-77).
pub fn registered_image_ios() -> Vec<&'static str> {
    registry().iter().map(|io| io.name()).collect()
}

/// Look an [`ImageIo`] up by class name — `ioutils::CreateImageIOByName`
/// (sitkImageIOUtilities.cxx:79-96), which throws on an unknown name.
pub fn image_io_by_name(name: &str) -> Result<&'static dyn ImageIo> {
    registry()
        .iter()
        .copied()
        .find(|io| io.name() == name)
        .ok_or_else(|| IoError::UnknownImageIo(name.to_string()))
}

/// Pick the [`ImageIo`] for `path` — `itk::ImageIOFactory::CreateImageIO`
/// (itkImageIOFactory.cxx:59-108).
///
/// Phase 1 probes the IOs that advertise a matching extension, phase 2 probes
/// the rest. `None` means no IO claimed the file; the caller turns that into
/// [`IoError::NoReaderFound`] or [`IoError::NoWriterFound`], because upstream's
/// two call sites raise different messages.
pub fn create_image_io(path: &Path, mode: FileMode) -> Option<&'static dyn ImageIo> {
    let can_handle = |io: &'static dyn ImageIo| match mode {
        FileMode::Read => io.can_read_file(path),
        FileMode::Write => io.can_write_file(path),
    };
    let extension_match = |io: &'static dyn ImageIo| {
        let extensions = match mode {
            FileMode::Read => io.supported_read_extensions(),
            FileMode::Write => io.supported_write_extensions(),
        };
        has_supported_extension(path, extensions, true)
    };

    let ios = registry();
    // Phase 1: extension-matching IOs, in registration order. One that matches
    // by extension but cannot handle the file is struck off so phase 2 does not
    // re-probe it (upstream nulls the pointer, :89).
    let mut struck_off = vec![false; ios.len()];
    for (slot, &io) in struck_off.iter_mut().zip(ios) {
        if !extension_match(io) {
            continue;
        }
        if can_handle(io) {
            return Some(io);
        }
        *slot = true;
    }

    // Phase 2: everything not already probed.
    for (&struck, &io) in struck_off.iter().zip(ios) {
        if !struck && can_handle(io) {
            return Some(io);
        }
    }
    None
}

/// Resolve the reader for `path`, reproducing
/// `ImageReaderBase::GetImageIOBase`'s error ladder
/// (sitkImageReaderBase.cxx:74-100): a missing file is reported as such before
/// "unable to determine ImageIO reader".
pub(crate) fn reader_for(path: &Path) -> Result<&'static dyn ImageIo> {
    if let Some(io) = create_image_io(path, FileMode::Read) {
        return Ok(io);
    }
    if !path.exists() {
        return Err(IoError::FileNotFound(path.to_path_buf()));
    }
    std::fs::File::open(path)?;
    Err(IoError::NoReaderFound(path.to_path_buf()))
}

/// Resolve the writer for `path` — `ImageFileWriter::GetImageIOBase`
/// (sitkImageFileWriter.cxx:195-219).
pub(crate) fn writer_for(path: &Path) -> Result<&'static dyn ImageIo> {
    create_image_io(path, FileMode::Write).ok_or_else(|| IoError::NoWriterFound(path.to_path_buf()))
}
