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

    /// A NRRD header could not be parsed. Carries the message NrrdIO's `biff`
    /// stack would have produced.
    #[error("malformed NRRD header: {0}")]
    MalformedNrrdHeader(String),

    /// A NRRD feature this port does not implement — a compressed encoding, a
    /// `block` element type, or a `data file:` form with no reader here.
    #[error("unsupported NRRD feature: {0}")]
    UnsupportedNrrdFeature(String),

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

    /// A zlib or gzip stream could not be inflated. Upstream has no such error:
    /// `MET_PerformUncompression` returns `true` after printing "Uncompress
    /// failed" (metaUtils.cxx:883), leaving the caller's buffer uninitialised.
    /// See [`crate::compression`] and ledger §4.64.
    #[error("corrupt compressed stream: {0}")]
    CorruptCompressedData(String),

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

    /// A transform rejected the parameters read from a transform file, or a
    /// composite rejected a sub-transform of another dimension.
    #[error(transparent)]
    Transform(#[from] sitk_transform::TransformError),

    /// The path's extension is not one an Insight legacy transform reader
    /// handles (`TxtTransformIO::CanReadFile` accepts only `.txt` and `.tfm`);
    /// `TransformFileReader::Update` then throws `"Could not create Transform IO
    /// object for reading file ..."` (itkTransformFileReader.cxx:83).
    #[error("could not create Transform IO object for reading file {0}")]
    NoTransformReaderFound(PathBuf),

    /// As [`IoError::NoTransformReaderFound`], for writing
    /// (itkTransformFileWriter.hxx: `"Can't Create IO object for file ..."`).
    #[error("could not create Transform IO object for writing file {0}")]
    NoTransformWriterFound(PathBuf),

    /// The transform file parsed but yielded no transform.
    /// `TransformFileReader::Update` throws `"failed to read file: ..."`
    /// (itkTransformFileReader.cxx:113-118), and `itk::simple::ReadTransform`
    /// throws `"there appears to be not transform in the file!"`
    /// (sitkTransform.cxx:676-680).
    #[error("read transform file {0}, but there appears to be no transform in the file")]
    NoTransformInFile(PathBuf),

    /// An `#Insight Transform File` line broke the format —
    /// e.g. `"Tags must be delimited by :"` (itkTxtTransformIO.cxx:152).
    #[error("malformed transform file: {0}")]
    MalformedTransformFile(String),

    /// A `Transform:` line named a transform this crate cannot construct.
    /// `TransformIOBase::CreateTransform` throws `"Unregistered transform type:
    /// ..."`.
    #[error("unregistered transform type: {0}")]
    UnknownTransformType(String),

    /// The transform read is neither 2D nor 3D. `itk::simple::ReadTransform`
    /// throws `"Unable to transform with InputSpaceDimension: ..."`
    /// (sitkTransform.cxx:718-722).
    #[error("unable to read a transform of dimension {0}: only 2D and 3D are supported")]
    UnsupportedTransformDimension(usize),
}

/// Convenience alias for IO results.
pub type Result<T> = std::result::Result<T, IoError>;
