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

    /// A GIPL feature this port does not implement, or that `GiplImageIO`
    /// itself refuses. Two sites, both upstream's own: the
    /// `"Pixel Type Unknown"` `SwapBytesIfNecessary` raises for a
    /// 32-bit integer image (itkGiplImageIO.cxx:648-651), and the
    /// `"Invalid type"` `Write` raises for a 64-bit one (`:759-761`).
    #[error("unsupported GIPL feature: {0}")]
    UnsupportedGiplFeature(String),

    /// A legacy VTK header could not be parsed. Carries `VTKImageIO`'s own
    /// message: `"Premature EOF in reading a line"`, `"Unrecognized type"`,
    /// `"Not structured points, can't read"`, `"No dimensions defined"`, or
    /// `"Unrecognized pixel type"` (itkVTKImageIO.cxx:46-193, :142).
    #[error("malformed VTK header: {0}")]
    MalformedVtkHeader(String),

    /// A legacy VTK feature SimpleITK's wrapping layer cannot represent — a
    /// `TENSORS` attribute (§3.37) — or a write `VTKImageIO` refuses, namely an
    /// image of more than three dimensions (itkVTKImageIO.cxx:647-651).
    #[error("unsupported VTK feature: {0}")]
    UnsupportedVtkFeature(String),

    /// A PNG file's 8-byte signature was missing or did not match, discovered
    /// while parsing a header (rather than by [`ImageIo::can_read_file`]
    /// (crate::ImageIo), which gates the normal registry path). Two upstream
    /// messages collapse into this one variant:
    /// `"PNGImageIO failed to read header for file: ..."` when fewer than 8
    /// bytes are present, and `"File is not png type: ..."` when 8 bytes are
    /// present but do not match (itkPNGImageIO.cxx:130-143, :325-338).
    #[error("malformed PNG header: {0}")]
    MalformedPngHeader(String),

    /// A PNG feature `PNGImageIO` itself refuses, or that SimpleITK's wrapping
    /// layer cannot represent. Two sites, both upstream's own: `"PNG supports
    /// unsigned char and unsigned short"` on write for any other component
    /// type (itkPNGImageIO.cxx:550), and `GetPixelIDFromImageIO`'s
    /// `"Unknown PixelType"` for a 2-channel (gray + alpha) PNG, which
    /// `png_get_channels` never turns into `RGB`/`RGBA` so `m_PixelType` stays
    /// `SCALAR` with `NumberOfComponents == 2`
    /// (itkPNGImageIO.cxx:452-461, sitkImageReaderBase.cxx:215-238). See
    /// [`crate::png`] and ledger §3.
    #[error("unsupported PNG feature: {0}")]
    UnsupportedPngFeature(String),

    /// A PNG file failed to decode — a bad signature the `png` crate itself
    /// caught, a truncated/corrupt IDAT stream, or a malformed chunk. Upstream
    /// has no single equivalent: libpng's error callback longjmps out of
    /// either `png_read_info` (`"PNG critical error in ..."`) or
    /// `png_read_image` (`"Error while reading file: ..."`)
    /// (itkPNGImageIO.cxx:164-168, :248-252).
    #[error("png decoding error: {0}")]
    PngDecode(#[from] png::DecodingError),

    /// A PNG image failed to encode — a bad bit-depth/color-type combination,
    /// or an IO failure inside the `png` crate's writer.
    #[error("png encoding error: {0}")]
    PngEncode(#[from] png::EncodingError),

    /// A zlib or gzip stream could not be inflated. Upstream has no such error:
    /// `MET_PerformUncompression` returns `true` after printing "Uncompress
    /// failed" (metaUtils.cxx:883), leaving the caller's buffer uninitialised.
    /// See [`crate::compression`] and ledger §4.75.
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

    /// The HDF5 layer refused a transform file. Upstream wraps every
    /// `H5::Exception` in an `itkExceptionMacro` carrying `getCDetailMsg()`
    /// (itkHDF5TransformIO.cxx:341-344, :420-423).
    #[error("hdf5 error: {0}")]
    Hdf5(#[from] rust_hdf5::Hdf5Error),

    /// An HDF5 transform file this port refuses but `itk::HDF5TransformIO`
    /// reads, because libhdf5 would convert the stored elements on the way out
    /// and [`rust_hdf5`] hands back the stored bytes — a big-endian parameter
    /// dataset, or a float element neither 4 nor 8 bytes wide.
    /// See [`crate::transform_hdf5`] and ledger §4.81, §4.82.
    #[error("unsupported HDF5 transform file: {0}")]
    UnsupportedHdf5Transform(String),

    /// The transform read is neither 2D nor 3D. `itk::simple::ReadTransform`
    /// throws `"Unable to transform with InputSpaceDimension: ..."`
    /// (sitkTransform.cxx:718-722).
    #[error("unable to read a transform of dimension {0}: only 2D and 3D are supported")]
    UnsupportedTransformDimension(usize),
}

/// Convenience alias for IO results.
pub type Result<T> = std::result::Result<T, IoError>;
