//! [`ImageFileWriter`] — SimpleITK's `itk::simple::ImageFileWriter`
//! (sitkImageFileWriter.h:77-184, sitkImageFileWriter.cxx).

use std::path::{Path, PathBuf};

use sitk_core::Image;

use crate::compression::{MAX_COMPRESSION_LEVEL, MIN_COMPRESSION_LEVEL};
use crate::error::Result;
use crate::image_io::{image_io_by_name, registered_image_ios, writer_for};

/// The two compression knobs SimpleITK's writer hands to `itk::ImageIOBase`,
/// as one value, because this crate's [`ImageIo`](crate::ImageIo) implementors
/// are stateless singletons in a static registry where upstream's are
/// per-write objects carrying `m_UseCompression` / `m_CompressionLevel`.
///
/// [`WriteOptions::default`] is SimpleITK's default: no compression, and a
/// level of `-1` meaning "leave the `ImageIO` on its own default".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteOptions {
    /// `SetUseCompression` — a *request*, honoured only by formats that
    /// compress (sitkImageFileWriter.h:80-99).
    pub use_compression: bool,
    /// `SetCompressionLevel` — `-1` (the default) means the `ImageIO` keeps its
    /// own level, because `itk::ImageFileWriter::GenerateData` forwards the
    /// value only when it is non-negative (itkImageFileWriter.hxx:199-201).
    pub compression_level: i32,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            use_compression: false,
            compression_level: -1,
        }
    }
}

impl WriteOptions {
    /// The level an `ImageIo` should deflate at, given the level it would use
    /// on its own.
    ///
    /// A negative [`WriteOptions::compression_level`] is never forwarded, so
    /// `io_default` stands. Anything else passes through
    /// `itkSetClampMacro(CompressionLevel, int, 1, GetMaximumCompressionLevel())`
    /// (itkImageIOBase.h:288) — so `0` becomes `1` and `100` becomes `9`.
    pub(crate) fn resolved_level(&self, io_default: i32) -> i32 {
        if self.compression_level < 0 {
            io_default
        } else {
            self.compression_level
                .clamp(MIN_COMPRESSION_LEVEL, MAX_COMPRESSION_LEVEL)
        }
    }
}

/// Write an image, choosing the format by file name or by an explicit
/// [`ImageIo`](crate::ImageIo) name.
///
/// ```no_run
/// # use sitk_core::{Image, PixelId};
/// # use sitk_io::ImageFileWriter;
/// # let image = Image::new(&[4, 4], PixelId::UInt8);
/// let mut writer = ImageFileWriter::new();
/// writer.set_file_name("out.mha").set_use_compression(true);
/// writer.execute(&image)?;
/// # Ok::<(), sitk_io::IoError>(())
/// ```
///
/// `SetCompressor` and the DICOM-only `SetKeepOriginalImageUID` are not
/// exposed. Upstream's compressor string selects between `gzip` and `bzip2`
/// for NRRD and does nothing for the other two formats; the default — the
/// empty string, which `NrrdImageIO::InternalSetCompressor` resolves to `gzip`
/// (itkNrrdImageIO.cxx:380-392) — is the only setting this port implements
/// (ledger §6).
#[derive(Clone, Debug, Default)]
pub struct ImageFileWriter {
    file_name: PathBuf,
    image_io_name: Option<String>,
    options: WriteOptions,
}

impl ImageFileWriter {
    /// A writer with no file name, letting the registry pick the format.
    pub fn new() -> Self {
        Self::default()
    }

    /// `SetFileName`.
    pub fn set_file_name<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.file_name = path.as_ref().to_path_buf();
        self
    }

    /// `GetFileName`.
    pub fn file_name(&self) -> &Path {
        &self.file_name
    }

    /// `SetUseCompression` (sitkImageFileWriter.h:87). A request: a format that
    /// cannot compress ignores it.
    pub fn set_use_compression(&mut self, use_compression: bool) -> &mut Self {
        self.options.use_compression = use_compression;
        self
    }

    /// `GetUseCompression`.
    pub fn use_compression(&self) -> bool {
        self.options.use_compression
    }

    /// `UseCompressionOn`.
    pub fn use_compression_on(&mut self) -> &mut Self {
        self.set_use_compression(true)
    }

    /// `UseCompressionOff`.
    pub fn use_compression_off(&mut self) -> &mut Self {
        self.set_use_compression(false)
    }

    /// `SetCompressionLevel` (sitkImageFileWriter.h:111). `-1`, the default,
    /// leaves each format on its own level: `2` for MetaImage and NRRD, and —
    /// for NIfTI, which never sees this value — zlib's default of `6`.
    ///
    /// The value is clamped to `1..=9` by the `ImageIO`, not here, so
    /// [`ImageFileWriter::compression_level`] reports back what was set.
    pub fn set_compression_level(&mut self, level: i32) -> &mut Self {
        self.options.compression_level = level;
        self
    }

    /// `GetCompressionLevel`.
    pub fn compression_level(&self) -> i32 {
        self.options.compression_level
    }

    /// Override the automatically detected [`ImageIo`](crate::ImageIo) by class
    /// name, e.g. `"MetaImageIO"`. `None` (the default) restores automatic
    /// detection.
    ///
    /// `SetImageIO` (sitkImageFileWriter.h:128-141). An unknown name is
    /// reported by [`ImageFileWriter::execute`] as
    /// [`IoError::UnknownImageIo`](crate::IoError::UnknownImageIo), matching
    /// upstream, where `CreateImageIOByName` throws inside `Execute`.
    pub fn set_image_io(&mut self, name: Option<&str>) -> &mut Self {
        self.image_io_name = name.map(str::to_string);
        self
    }

    /// `GetImageIO` — the empty string upstream, `None` here.
    pub fn image_io(&self) -> Option<&str> {
        self.image_io_name.as_deref()
    }

    /// The class names of every registered [`ImageIo`](crate::ImageIo) —
    /// `GetRegisteredImageIOs` (sitkImageFileWriter.h:75-77).
    pub fn registered_image_ios(&self) -> Vec<&'static str> {
        registered_image_ios()
    }

    /// Write `image` to the configured file name.
    ///
    /// `Execute` (sitkImageFileWriter.cxx:222-249). The IO is the one named by
    /// [`ImageFileWriter::set_image_io`] if any, else the first registered IO
    /// whose `can_write_file` accepts the path — which, for every format, means
    /// the path's extension.
    pub fn execute(&self, image: &Image) -> Result<()> {
        let io = match &self.image_io_name {
            Some(name) => image_io_by_name(name)?,
            None => writer_for(&self.file_name)?,
        };
        io.write(image, &self.file_name, &self.options)
    }

    /// `Execute(image, fileName, useCompression, compressionLevel)`
    /// (sitkImageFileWriter.cxx:161-167).
    ///
    /// Upstream's overload is `SetFileName` + `SetUseCompression` +
    /// `SetCompressionLevel` on `this`, then the one-argument `Execute` — so all
    /// three *persist* on the writer. `&mut self` here says the same thing.
    pub fn execute_with<P: AsRef<Path>>(
        &mut self,
        image: &Image,
        path: P,
        use_compression: bool,
        compression_level: i32,
    ) -> Result<()> {
        self.set_file_name(path)
            .set_use_compression(use_compression)
            .set_compression_level(compression_level);
        self.execute(image)
    }
}
