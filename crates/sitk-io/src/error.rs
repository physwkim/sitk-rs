//! IO error type.

use std::path::PathBuf;

/// Errors produced while reading or writing image files.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Underlying filesystem error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// No registered [`ImageIo`](crate::ImageIo) claimed the file for reading,
    /// and the file exists and is readable. `ImageReaderBase::GetImageIOBase`
    /// throws `"Unable to determine ImageIO reader for ..."` here
    /// (sitkImageReaderBase.cxx:99).
    #[error("unable to determine ImageIO reader for {0}")]
    NoReaderFound(PathBuf),

    /// No registered [`ImageIo`](crate::ImageIo) claimed the file for writing.
    /// `ImageFileWriter::GetImageIOBase` throws `"Unable to determine ImageIO
    /// writer for ..."` here (sitkImageFileWriter.cxx:209).
    #[error("unable to determine ImageIO writer for {0}")]
    NoWriterFound(PathBuf),

    /// A read was requested for a path that does not exist. Checked before
    /// [`IoError::NoReaderFound`], as in sitkImageReaderBase.cxx:89-92.
    #[error("the file {0} does not exist")]
    FileNotFound(PathBuf),

    /// `set_image_io` named an IO that is not registered.
    /// `ioutils::CreateImageIOByName` throws `"Unable to create ImageIO: ..."`
    /// (sitkImageIOUtilities.cxx:92).
    #[error("unable to create ImageIO: {0}")]
    UnknownImageIo(String),

    /// A MetaImage header could not be parsed.
    #[error("malformed MetaImage header")]
    MalformedHeader,

    /// A MetaIO `ElementType` value is not a supported scalar pixel type.
    #[error("unsupported MetaImage ElementType: {0}")]
    UnsupportedElementType(String),

    /// A MetaImage feature not yet implemented.
    #[error("unsupported MetaImage feature: {0}")]
    Unsupported(String),

    /// A NIfTI-1 header could not be parsed, or is internally inconsistent.
    /// `nifti_convert_nhdr2nim` rejects `bad dim[0]`, `bad sizeof_hdr`,
    /// `bad dim[1]` and `bad datatype` (nifti1_io.c:3654-3752); ITK itself
    /// rejects a file with no orthonormal direction cosines
    /// (itkNiftiImageIO.cxx:1847).
    #[error("malformed NIfTI header: {0}")]
    MalformedNiftiHeader(String),

    /// A NIfTI-1 `datatype` value ITK's `ReadImageInformation` leaves as
    /// `UNKNOWNCOMPONENTTYPE` (itkNiftiImageIO.cxx:924) — `DT_FLOAT128`,
    /// `DT_COMPLEX256`, or an unassigned code.
    #[error("unsupported NIfTI datatype: {0}")]
    UnsupportedNiftiDatatype(i16),

    /// A NIfTI-1 feature this port does not implement, or that SimpleITK's
    /// wrapping layer cannot represent.
    #[error("unsupported NIfTI feature: {0}")]
    UnsupportedNiftiFeature(String),

    /// A value in the meta-data dictionary is not usable in the NIfTI-1 header
    /// field it feeds — `itk::StringToInt32` on `qform_code`, or an over-long
    /// `aux_file` / `ITK_FileNotes`.
    #[error("invalid NIfTI meta-data: {0}")]
    InvalidNiftiMetaData(String),

    /// `NiftiImageIO::WriteImageInformation` refused the image or the file
    /// name — an axis longer than `SHRT_MAX`, an unusable extension, or a
    /// vector image of more than four dimensions.
    #[error("cannot write NIfTI file: {0}")]
    NiftiWriteRejected(String),

    /// The pixel data was shorter than the header's declared size.
    #[error("pixel data is truncated")]
    TruncatedData,

    /// A path lacked a usable stem/filename.
    #[error("invalid image path: {0}")]
    InvalidPath(PathBuf),

    /// An extraction region set on [`ImageFileReader`](crate::ImageFileReader)
    /// is not contained in the file's region.
    #[error(
        "the requested extraction region (index {index:?}, size {size:?}) \
         is not contained within the file's region {file_size:?}"
    )]
    ExtractRegionOutOfBounds {
        /// The requested start index, per internal axis.
        index: Vec<usize>,
        /// The requested size, per internal axis (`0` collapses the axis).
        size: Vec<usize>,
        /// The file's own size, padded to the internal dimension.
        file_size: Vec<usize>,
    },

    /// The extraction region's non-zero axis count is not a legal output
    /// dimension. `ImageFileReader::Execute` throws `"The extraction region has
    /// unsupported output dimension of ..."` (sitkImageFileReader.cxx:319-324).
    #[error("the extraction region has unsupported output dimension of {0}")]
    ExtractOutputDimension(usize),

    /// The direction submatrix left by collapsing the zero-size axes is
    /// singular. `ExtractImageFilter`'s `DIRECTIONCOLLAPSETOSUBMATRIX` throws
    /// `"Invalid submatrix extracted for collapsed direction."`
    /// (itkExtractImageFilter.hxx:196-199).
    #[error("invalid submatrix extracted for collapsed direction")]
    SingularCollapsedDirection,

    /// The file's own dimension is below the minimum SimpleITK will load.
    /// `ImageFileReader::Execute` throws `"The file has unsupported image
    /// dimension of ..."` (sitkImageFileReader.cxx:302-307).
    #[error("the file has unsupported image dimension of {0}")]
    UnsupportedImageDimension(usize),

    /// A core image error surfaced during assembly.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for IO results.
pub type Result<T> = std::result::Result<T, IoError>;
