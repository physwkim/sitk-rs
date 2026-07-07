//! IO error type.

use std::path::PathBuf;

/// Errors produced while reading or writing image files.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Underlying filesystem error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The file extension is not a recognised image format.
    #[error("unrecognised image file extension: {0}")]
    UnknownExtension(String),

    /// A MetaImage header could not be parsed.
    #[error("malformed MetaImage header")]
    MalformedHeader,

    /// A MetaIO `ElementType` value is not a supported scalar pixel type.
    #[error("unsupported MetaImage ElementType: {0}")]
    UnsupportedElementType(String),

    /// A MetaImage feature not yet implemented.
    #[error("unsupported MetaImage feature: {0}")]
    Unsupported(String),

    /// The pixel data was shorter than the header's declared size.
    #[error("pixel data is truncated")]
    TruncatedData,

    /// A path lacked a usable stem/filename.
    #[error("invalid image path: {0}")]
    InvalidPath(PathBuf),

    /// A core image error surfaced during assembly.
    #[error(transparent)]
    Core(#[from] sitk_core::Error),
}

/// Convenience alias for IO results.
pub type Result<T> = std::result::Result<T, IoError>;
