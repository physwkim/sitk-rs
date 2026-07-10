//! [`ImageFileWriter`] — SimpleITK's `itk::simple::ImageFileWriter`
//! (sitkImageFileWriter.h:77-184, sitkImageFileWriter.cxx).

use std::path::{Path, PathBuf};

use sitk_core::Image;

use crate::error::Result;
use crate::image_io::{image_io_by_name, registered_image_ios, writer_for};

/// Write an image, choosing the format by file name or by an explicit
/// [`ImageIo`](crate::ImageIo) name.
///
/// ```no_run
/// # use sitk_core::{Image, PixelId};
/// # use sitk_io::ImageFileWriter;
/// # let image = Image::new(&[4, 4], PixelId::UInt8);
/// let mut writer = ImageFileWriter::new();
/// writer.set_file_name("out.mha");
/// writer.execute(&image)?;
/// # Ok::<(), sitk_io::IoError>(())
/// ```
///
/// Compression (`SetUseCompression` / `SetCompressionLevel` / `SetCompressor`)
/// and the DICOM-only `SetKeepOriginalImageUID` are not exposed: no format in
/// this crate compresses yet.
#[derive(Clone, Debug, Default)]
pub struct ImageFileWriter {
    file_name: PathBuf,
    image_io_name: Option<String>,
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
        io.write(image, &self.file_name)
    }
}
