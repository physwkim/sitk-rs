//! MATLAB Level-4 transform files (`.mat`) — a port of
//! `itk::MatlabTransformIOTemplate<double>`
//! (`Modules/IO/TransformMatlab/src/itkMatlabTransformIO.cxx`), reached
//! through [`crate::io::read_transform`] / [`crate::io::write_transform`] like
//! [`crate::io::transform_io`] and [`crate::io::transform_hdf5`].
//!
//! The format is `vnl_matlab_write`'s (`Modules/ThirdParty/VNL/src/vxl/core/
//! vnl/vnl_matlab_write.h`/`.cxx`) — a MATLAB Level-4 `.mat` file holds a flat
//! sequence of *records*, one per stored vnl vector/matrix, each:
//!
//! ```text
//! i32  type     -- precision + storage + byte order, see [`parse_header`]
//! i32  rows
//! i32  cols
//! i32  imag     -- 0: real; non-zero: an imaginary block follows (never written, refused on read)
//! i32  namlen   -- variable-name length, including the trailing NUL
//! u8[namlen]    -- the variable name
//! T[rows*cols]  -- column-major data, T = f32 (`type` single) or f64 (double)
//! ```
//!
//! `MatlabTransformIOTemplate<double>::Write` (`:125-142`) writes each
//! transform as exactly **two** consecutive records: the parameters, named
//! the transform's own `GetTransformTypeAsString()`, then the fixed
//! parameters, always named the literal `"fixed"` — never the transform's
//! name suffixed, and never validated on read (ledger §2.143). Both records
//! are always `rows`-long **column vectors** (`cols == 1`); `Read()`
//! (`:75-123`) rejects any other shape with `"Only vector parameters
//! supported"`.
//!
//! # No composite flattening
//!
//! Unlike [`crate::io::transform_io`] and [`crate::io::transform_hdf5`], a
//! [`crate::transform::Transform::Composite`] is **not** expanded into its
//! queue here. `TransformFileWriterTemplate::Update`'s composite-flattening
//! `CompositeTransformIOHelper` runs only on the float↔double
//! precision-*mismatch* write path (`itkTransformFileWriterSpecializations.cxx:
//! 276-323`, the generic `AddToTransformList<TIn, TOut>` template); the
//! common double→double path this crate always takes has its own full
//! specialization (`AddToTransformList<TransformBaseTemplate<double>,
//! TransformBaseTemplate<double>>`, `:328-335`) that pushes the transform onto
//! `m_TransformList` **whole**. `MatlabTransformIO::Write` therefore writes a
//! composite as one opaque record pair: its own aggregate
//! `GetParameters()`/`GetFixedParameters()` (`itkCompositeTransform.hxx:
//! 594-741`), under the name `CompositeTransform_double_D_D`.
//!
//! Reading that back cannot reconstruct the sub-transform queue — the file
//! carries no boundary between sub-transforms, only one concatenated
//! parameter vector — so upstream's `Read` always creates a **fresh, empty**
//! `CompositeTransform` for that name and then calls `SetFixedParameters` /
//! `SetParametersByValue` on it, exactly as for any other transform. An empty
//! composite expects zero parameters and zero fixed parameters
//! (`GetNumberOfParameters`/`GetNumberOfLocalParameters` over an empty
//! `GetTransformsToOptimizeQueue()`, `itkCompositeTransform.hxx:594-620`), so
//! that round-trips; a **non-empty** composite's aggregate vectors are
//! (almost always) non-empty, so on read `SetFixedParameters` fails a
//! parameter-count check — a file only upstream's own writer produces and its
//! own reader cannot load. ITK's own test suite for this IO
//! (`itkIOTransformMatlabGTest.cxx`) exercises exactly one composite case,
//! `WriteAndReadEmptyCompositeTransform`, and no non-empty one.
//!
//! This port does **not** reproduce that write-then-fail asymmetry (§5.29,
//! option b): `write_transform` refuses a non-empty composite up front with
//! [`IoError::UnsupportedMatlabTransformFeature`], so the failure surfaces at
//! write rather than on a later read, and no unreadable file is produced. An
//! empty composite still writes and round-trips through
//! `read_transform_list`, whose fresh empty `CompositeTransform` accepts the
//! empty aggregate vectors. See ledger §1.69/§2.144.
//!
//! The same guard covers nesting: because a composite containing a composite
//! makes the outer composite non-empty, `write_transform` refuses it too —
//! where upstream's `Write` never special-cases `CompositeTransform` at all
//! and so places **no restriction on composite nesting**, unlike `.tfm`
//! (`"Composite Transform can only be 1st transform in a file"`) and `.h5`
//! (the same message). See ledger §2.145.
//!
//! # CanReadFile / CanWriteFile
//!
//! `MatlabTransformIOTemplate::CanReadFile` and `CanWriteFile`
//! (`itkMatlabTransformIO.cxx:30-48`) are textually identical: both check only
//! `itksys::SystemTools::GetFilenameLastExtension(filename) == ".mat"`,
//! case-sensitively. No content probing, no read/write asymmetry — unlike
//! HDF5 (§2.119/§4.80), this format has none to reproduce.
//!
//! # Fidelity notes
//!
//! Byte-swap detection is modelled on `vnl_matlab_readhdr::read_hdr`'s
//! `switch` over "recognized native" `type` values
//! (`vnl_matlab_read.cxx:124-150`), but recognizes all 8 legal
//! `precision | storage | byte_order` combinations `vnl_matlab_header.h`'s
//! `type_t` can compose, not just the 7 upstream's own switch lists —
//! upstream is missing `1010` (big-endian, column-wise, single-precision),
//! so a genuinely big-endian single-precision `.mat` file misclassifies as
//! needing a swap it does not need. Fixed in this port (ledger §1.69): `1010`
//! is exactly as native as its seven siblings, and recognizing it costs
//! nothing (the writer only ever emits `type = 0`, so this is purely a
//! foreign-file read-path correction).
//!
//! A complex-valued record (`imag != 0`) is refused outright
//! ([`IoError::UnsupportedMatlabTransformFeature`]) rather than reproduced.
//! Upstream's `ReadMat` (`:50-73`) ignores `vnl_matlab_readhdr::read_data`'s
//! `bool` return; `read_data` on a complex header fails `type_chck` and
//! returns `false` **without reading the data blocks**, leaving the stream
//! desynchronized so every subsequent record decodes as garbage relative to
//! whatever bytes happen to follow. That is not a "quirk" with stable,
//! reproducible semantics — it is stream corruption — so this port closes it
//! at the point of detection instead (ledger §4.104).
//!
//! A short read of the header that immediately **precedes** a transform (the
//! loop's `if (!mathdr) break;`, `:90-93`) ends the file cleanly, matching
//! `std::istream::operator bool()`'s `good() && !eof()` — any read that
//! cannot fill the full 20 requested bytes sets both `failbit` and `eofbit`,
//! regardless of how many of the 20 it actually got. A short read of the
//! *second* header of a pair (the "fixed" one, checked with no such gate at
//! `:109-114`) is **not** distinguished by how many bytes were obtained: a
//! clean end-of-file leaves the header's `std::memset`-zeroed fields in
//! place, so `cols()` reads `0` and upstream's ungated `cols() != 1` check
//! throws `"Only vector parameters supported"`; this port raises the same
//! error uniformly for *any* short second-header read, since a partial (not
//! clean-EOF) truncation lands on uninitialized C++ stack memory upstream and
//! has no single reproducible outcome to match (ledger §4.104). A truncated
//! variable name or a truncated data block is likewise not attempted
//! bit-for-bit and surfaces as a plain IO error.

use std::io::Read;
use std::path::Path;

use crate::transform::{ParametricTransform, Transform};

use crate::io::error::{IoError, Result};
use crate::io::transform_io::{correct_transform_precision_type, create_transform};

/// `MatlabTransformIOTemplate::CanReadFile` (`itkMatlabTransformIO.cxx:30-38`):
/// the `.mat` extension, case-sensitive.
pub fn can_read_file(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("mat")
}

/// `MatlabTransformIOTemplate::CanWriteFile` (`itkMatlabTransformIO.cxx:40-48`)
/// — textually identical to [`can_read_file`].
pub fn can_write_file(path: &Path) -> bool {
    can_read_file(path)
}

// ---------------------------------------------------------------------------
// Header parsing — vnl_matlab_readhdr::read_hdr / vnl_matlab_header
// ---------------------------------------------------------------------------

/// A parsed `vnl_matlab_header` (`vnl_matlab_header.h:35-56`) plus the
/// variable name `read_hdr` reads immediately after it (`vnl_matlab_read.cxx:
/// 152-159`).
struct MatHeader {
    rows: i32,
    cols: i32,
    imag: i32,
    is_single: bool,
    need_swap: bool,
    name: String,
}

/// The header a short, clean end-of-file synthesizes: upstream's
/// `std::memset(&hdr, 0, sizeof hdr)` (`vnl_matlab_read.cxx:118`) before the
/// raw read that fails to fill it.
fn zeroed_header() -> MatHeader {
    MatHeader {
        rows: 0,
        cols: 0,
        imag: 0,
        is_single: false,
        need_swap: false,
        name: String::new(),
    }
}

/// Reads up to `buf.len()` bytes, stopping at the first zero-length read
/// (i.e. end of file). Returns how many bytes were actually obtained — never
/// more than `buf.len()`, possibly fewer on a short file.
fn fill_up_to(reader: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        let n = reader.read(&mut buf[got..])?;
        if n == 0 {
            break;
        }
        got += n;
    }
    Ok(got)
}

/// The leading header of a (parameters, fixed) pair — `Read()`'s
/// `if (!mathdr) break;` (`itkMatlabTransformIO.cxx:90-93`). `None` on a
/// clean end of file, ending the read loop with no error.
fn read_leading_header(reader: &mut impl Read) -> Result<Option<MatHeader>> {
    let mut buf = [0u8; 20];
    if fill_up_to(reader, &mut buf)? < 20 {
        return Ok(None);
    }
    Ok(Some(parse_header(reader, &buf)?))
}

/// The trailing ("fixed") header of a pair — read with no `operator bool()`
/// gate at all (`itkMatlabTransformIO.cxx:95-114`). A short read synthesizes
/// [`zeroed_header`], whose `cols() == 0` then fails the caller's shared
/// `cols != 1` check exactly as upstream's clean-EOF case does.
fn read_trailing_header(reader: &mut impl Read) -> Result<MatHeader> {
    let mut buf = [0u8; 20];
    if fill_up_to(reader, &mut buf)? < 20 {
        return Ok(zeroed_header());
    }
    parse_header(reader, &buf)
}

/// `vnl_matlab_readhdr::read_hdr` (`vnl_matlab_read.cxx:118-171`) from a
/// filled 20-byte header buffer: byte-swap detection, then the variable name.
fn parse_header(reader: &mut impl Read, buf: &[u8; 20]) -> Result<MatHeader> {
    let mut ty = i32::from_le_bytes(buf[0..4].try_into().unwrap());
    let mut rows = i32::from_le_bytes(buf[4..8].try_into().unwrap());
    let mut cols = i32::from_le_bytes(buf[8..12].try_into().unwrap());
    let mut imag = i32::from_le_bytes(buf[12..16].try_into().unwrap());
    let mut namlen = i32::from_le_bytes(buf[16..20].try_into().unwrap());

    // `need_swap` is false for `type == 0` (native little-endian double) or
    // for any of the 8 legal `precision | storage | byte_order` combinations
    // built from `vnl_matlab_header.h`'s `type_t` (`SINGLE=10`,
    // `ROW_WISE=100`, `BIG_ENDIAN=1000`) — upstream's own switch
    // (`vnl_matlab_read.cxx:124-150`) lists only 7 of the 8, omitting `1010`
    // (big-endian, column-wise, single-precision); this port recognizes all
    // eight, since `1010` is exactly as native as its seven siblings
    // (ledger §1.69, fixed in this port).
    let need_swap = !matches!(ty, 0 | 10 | 100 | 110 | 1000 | 1010 | 1100 | 1110);
    if need_swap {
        ty = ty.swap_bytes();
        rows = rows.swap_bytes();
        cols = cols.swap_bytes();
        imag = imag.swap_bytes();
        namlen = namlen.swap_bytes();
    }

    // `is_single()`: `(type % (10 * SINGLE)) >= SINGLE`, i.e. `(type % 100) >=
    // 10` (`vnl_matlab_read.cxx:96-99` via `vnl_matlab_header.h`'s `type_t`).
    // `rem_euclid` keeps this defined even for a corrupted negative `type`,
    // where C++'s truncating `%` would differ from Rust's `%` on the sign.
    let is_single = ty.rem_euclid(100) >= 10;

    let namlen = usize::try_from(namlen).map_err(|_| {
        IoError::MalformedTransformFile("negative variable name length in .mat header".to_string())
    })?;
    let mut name_buf = vec![0u8; namlen];
    reader.read_exact(&mut name_buf)?;
    // `varname[hdr.namlen] = '\0'` (`vnl_matlab_read.cxx:161`) guarantees a
    // NUL after whatever `namlen` bytes were read; `std::string(name())`
    // therefore stops at the first NUL, even an internal one.
    let end = name_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(name_buf.len());
    let name = String::from_utf8_lossy(&name_buf[..end]).into_owned();

    Ok(MatHeader {
        rows,
        cols,
        imag,
        is_single,
        need_swap,
        name,
    })
}

/// Both headers of a pair must describe a column vector — `Read()`'s
/// `if (mathdr.cols() != 1)` / `if (mathdr2.cols() != 1)`
/// (`itkMatlabTransformIO.cxx:97-99`, `:109-111`), both raising the same
/// message.
fn require_column_vector(header: &MatHeader) -> Result<()> {
    if header.cols != 1 {
        return Err(IoError::MalformedTransformFile(
            "Only vector parameters supported".to_string(),
        ));
    }
    Ok(())
}

/// `ReadMat<T>` (`itkMatlabTransformIO.cxx:50-73`): `header.rows` elements,
/// `f32` when [`MatHeader::is_single`] else `f64`, each byte-swapped if the
/// header needed a swap, widened to `f64` either way.
fn read_vector(reader: &mut impl Read, header: &MatHeader) -> Result<Vec<f64>> {
    if header.imag != 0 {
        return Err(IoError::UnsupportedMatlabTransformFeature(
            "complex-valued .mat variable".to_string(),
        ));
    }
    let rows = usize::try_from(header.rows).map_err(|_| {
        IoError::MalformedTransformFile("negative row count in .mat header".to_string())
    })?;
    let mut out = Vec::with_capacity(rows);
    if header.is_single {
        let mut buf = [0u8; 4];
        for _ in 0..rows {
            reader.read_exact(&mut buf)?;
            if header.need_swap {
                buf.reverse();
            }
            out.push(f64::from(f32::from_le_bytes(buf)));
        }
    } else {
        let mut buf = [0u8; 8];
        for _ in 0..rows {
            reader.read_exact(&mut buf)?;
            if header.need_swap {
                buf.reverse();
            }
            out.push(f64::from_le_bytes(buf));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// `MatlabTransformIOTemplate<double>::Read` (`itkMatlabTransformIO.cxx:75-123`)
/// — the flat `ReadTransformList`, before `TransformFileReader::Update`'s
/// composite fold (which, for this format, only ever has something to do on
/// an empty composite; see the module doc).
pub(crate) fn read_transform_list(path: &Path) -> Result<Vec<Transform>> {
    let bytes = std::fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            IoError::FileNotFound(path.to_path_buf())
        } else {
            IoError::Io(e)
        }
    })?;
    let mut cursor = std::io::Cursor::new(bytes);
    let mut list = Vec::new();

    while let Some(header) = read_leading_header(&mut cursor)? {
        require_column_vector(&header)?;
        let parameters = read_vector(&mut cursor, &header)?;

        let type_name = correct_transform_precision_type(&header.name)?;
        let mut transform = create_transform(&type_name)?;

        let fixed_header = read_trailing_header(&mut cursor)?;
        require_column_vector(&fixed_header)?;
        let fixed_parameters = read_vector(&mut cursor, &fixed_header)?;

        // `itkMatlabTransformIO.cxx:117-118`: fixed parameters set before
        // parameters.
        transform.set_fixed_parameters(&fixed_parameters)?;
        transform.set_parameters(&parameters)?;
        list.push(transform);
    }
    Ok(list)
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// `MatlabTransformIOTemplate<double>::Write`
/// (`itkMatlabTransformIO.cxx:125-142`) over `vnl_matlab_write`'s 1-D array
/// overload (`vnl_matlab_write.cxx:163-183`).
///
/// No flattening: see the module doc's "No composite flattening" section.
/// Every non-composite transform, and an *empty* composite, is written through
/// the same generic `parameters()`/`fixed_parameters()` calls. A **non-empty**
/// composite is refused up front (see below).
pub(crate) fn write_transform(transform: &Transform, path: &Path) -> Result<()> {
    // §2.144/§2.145 (§5.29, option b): upstream's `Write` writes a non-empty
    // composite as one opaque record pair — its aggregate parameter vectors,
    // with no sub-transform boundary anywhere in the file — and
    // [`read_transform_list`] can only rebuild a fresh *empty* composite from
    // that name, so the parameter counts mismatch and the file never reads
    // back. Rather than reproduce that asymmetry (a file only this IO's own
    // writer can produce and its own reader cannot load), refuse the write, so
    // the failure surfaces here instead of on a later read. An empty composite
    // still writes and round-trips. A *nested* composite is caught by the same
    // guard: the outer composite is non-empty (§2.145).
    if let Transform::Composite(composite) = transform
        && !composite.transforms().is_empty()
    {
        return Err(IoError::UnsupportedMatlabTransformFeature(format!(
            "a non-empty CompositeTransform cannot be written to a .mat file and read \
                 back — MatlabTransformIO records one concatenated parameter vector with no \
                 sub-transform boundary (doc/upstream-findings.md §2.144/§2.145), so the {} \
                 sub-transform(s) here would be unrecoverable",
            composite.transforms().len()
        )));
    }

    let mut out = Vec::new();
    write_record(
        &mut out,
        &transform.itk_transform_type_name(),
        &transform.parameters(),
    );
    write_record(&mut out, "fixed", &transform.fixed_parameters());
    std::fs::write(path, out)?;
    Ok(())
}

/// `vnl_matlab_write(ostream&, const T*, unsigned, const char*)`
/// (`vnl_matlab_write.cxx:163-180`) specialized to `T = double`, the only
/// precision this port's writer ever produces: `hdr.type` is always the fully
/// native sentinel `0` (little-endian, column-wise, double —
/// `vnl_matlab_header::vnl_none`, `vnl_matlab_header.h:44`).
fn write_record(out: &mut Vec<u8>, name: &str, values: &[f64]) {
    out.extend_from_slice(&0i32.to_le_bytes()); // type
    // `static_cast<vxl_int_32>(n)` (`:172`) is an unchecked, silently
    // truncating cast in upstream too; `as i32` matches it.
    out.extend_from_slice(&(values.len() as i32).to_le_bytes()); // rows
    out.extend_from_slice(&1i32.to_le_bytes()); // cols
    out.extend_from_slice(&0i32.to_le_bytes()); // imag: always real
    let namlen = (name.len() + 1) as i32; // includes the trailing NUL
    out.extend_from_slice(&namlen.to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::transform::{
        AffineTransform, BSplineTransform, ComposeScaleSkewVersor3DTransform, CompositeTransform,
        DisplacementFieldTransform, Euler2DTransform, Euler3DTransform, ScaleLogarithmicTransform,
        ScaleSkewVersor3DTransform, ScaleTransform, ScaleVersor3DTransform, Similarity2DTransform,
        Similarity3DTransform, TranslationTransform, VersorRigid3DTransform, VersorTransform,
    };

    use super::*;
    use crate::io::{read_transform, write_transform as dispatch_write};

    fn tmp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("sitk_rs_transform_matlab_{name}"));
        path
    }

    /// Every concrete `Transform` variant except `Composite`, with parameters
    /// that are not the identity's, so a dropped field cannot pass.
    fn every_transform() -> Vec<Transform> {
        vec![
            TranslationTransform::new(vec![1.0, -2.5]).into(),
            ScaleTransform::new(vec![2.0, 3.0], vec![1.0, 1.0]).into(),
            ScaleLogarithmicTransform::new(vec![2.0, 3.0, 4.0], vec![1.0, 0.0, -1.0]).into(),
            Euler2DTransform::new(0.3, [1.0, 2.0], [0.5, -0.5]).into(),
            Euler3DTransform::new(0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [0.5, 0.0, -0.5]).into(),
            Similarity2DTransform::new(1.5, 0.3, [1.0, 2.0], [0.5, -0.5]).into(),
            Similarity3DTransform::new(1.5, 0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [0.5, 0.0, -0.5])
                .into(),
            VersorTransform::new(0.1, 0.2, 0.3, [0.5, 0.0, -0.5]).into(),
            VersorRigid3DTransform::new(0.1, 0.2, 0.3, [1.0, 2.0, 3.0], [0.5, 0.0, -0.5]).into(),
            ScaleVersor3DTransform::new(
                [1.1, 1.2, 1.3],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [0.5, 0.0, -0.5],
            )
            .into(),
            ScaleSkewVersor3DTransform::new(
                [1.1, 1.2, 1.3],
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [0.5, 0.0, -0.5],
            )
            .into(),
            ComposeScaleSkewVersor3DTransform::new(
                [1.1, 1.2, 1.3],
                [0.01, 0.02, 0.03],
                0.1,
                0.2,
                0.3,
                [1.0, 2.0, 3.0],
                [0.5, 0.0, -0.5],
            )
            .into(),
            AffineTransform::new(
                2,
                vec![1.0, 0.2, -0.3, 1.1],
                vec![4.0, 5.0],
                vec![0.5, -0.5],
            )
            .into(),
            displacement_field().into(),
            bspline().into(),
        ]
    }

    fn displacement_field() -> DisplacementFieldTransform {
        let mut field = DisplacementFieldTransform::new(
            2,
            &[3, 2],
            &[1.0, 2.0],
            &[0.5, 0.25],
            &[0.0, -1.0, 1.0, 0.0],
        )
        .unwrap();
        let n = field.number_of_parameters();
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.125).collect();
        field.set_parameters(&values).unwrap();
        field
    }

    fn bspline() -> BSplineTransform {
        let mut transform =
            BSplineTransform::new(2, &[0.0, 0.0], &[4.0, 4.0], &[1.0, 0.0, 0.0, 1.0], &[2, 2])
                .unwrap();
        let n = transform.number_of_parameters();
        let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        transform.set_parameters(&values).unwrap();
        transform
    }

    fn composite() -> Transform {
        let mut composite = CompositeTransform::new(2);
        composite
            .add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        composite
            .add_transform(Euler2DTransform::new(0.25, [3.0, 4.0], [0.5, 0.5]).into())
            .unwrap();
        composite.into()
    }

    // -- CanReadFile / CanWriteFile -----------------------------------------

    #[test]
    fn can_read_and_write_agree_on_the_extension() {
        assert!(can_read_file(Path::new("t.mat")));
        assert!(can_write_file(Path::new("t.mat")));
        assert!(!can_read_file(Path::new("t.MAT")));
        assert!(!can_write_file(Path::new("t.MAT")));
        assert!(!can_read_file(Path::new("t.tfm")));
        assert!(!can_read_file(Path::new("t")));
    }

    // -- round trips ----------------------------------------------------------

    #[test]
    fn every_transform_round_trips_through_a_mat_file() {
        for (i, transform) in every_transform().into_iter().enumerate() {
            let path = tmp_path(&format!("roundtrip_{i}.mat"));
            dispatch_write(&transform, &path).unwrap();
            let read = read_transform(&path).unwrap();
            let _ = std::fs::remove_file(&path);

            assert_eq!(
                read.itk_transform_type_name(),
                transform.itk_transform_type_name()
            );
            assert_eq!(
                read.parameters(),
                transform.parameters(),
                "parameters of {}",
                transform.itk_transform_type_name()
            );
            assert_eq!(
                read.fixed_parameters(),
                transform.fixed_parameters(),
                "fixed parameters of {}",
                transform.itk_transform_type_name()
            );
        }
    }

    #[test]
    fn an_empty_composite_round_trips_through_a_mat_file() {
        let transform: Transform = CompositeTransform::new(2).into();
        let path = tmp_path("empty_composite.mat");
        dispatch_write(&transform, &path).unwrap();
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read, transform);
    }

    /// §2.144 / §5.29: a non-empty composite would write as one opaque record
    /// pair that can never be read back, so this port refuses it at write —
    /// the failure surfaces here, not on a later read — and produces no file.
    #[test]
    fn a_non_empty_composite_is_rejected_at_write() {
        let transform = composite();
        let path = tmp_path("non_empty_composite.mat");
        let result = dispatch_write(&transform, &path);
        assert!(
            matches!(result, Err(IoError::UnsupportedMatlabTransformFeature(_))),
            "{result:?}"
        );
        assert!(!path.exists());
    }

    /// §2.145: a composite nested inside another is caught by the same
    /// non-empty guard (the outer composite is non-empty), so it is refused at
    /// write here too — unlike upstream, whose `Write` never special-cases
    /// `CompositeTransform` and would write the same unreadable file.
    #[test]
    fn a_nested_composite_is_rejected_at_write() {
        let mut inner = CompositeTransform::new(2);
        inner
            .add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        let mut outer = CompositeTransform::new(2);
        outer.add_transform(inner.into()).unwrap();
        let transform: Transform = outer.into();

        let path = tmp_path("nested_composite.mat");
        let result = dispatch_write(&transform, &path);
        assert!(
            matches!(result, Err(IoError::UnsupportedMatlabTransformFeature(_))),
            "{result:?}"
        );
        assert!(!path.exists());
    }

    #[test]
    fn a_transform_crosses_between_mat_tfm_and_h5() {
        for (i, transform) in every_transform().into_iter().enumerate() {
            let mat = tmp_path(&format!("cross_{i}.mat"));
            let tfm = tmp_path(&format!("cross_{i}.tfm"));
            let h5 = tmp_path(&format!("cross_{i}.h5"));
            let name = transform.itk_transform_type_name();

            dispatch_write(&transform, &mat).unwrap();
            let from_mat = read_transform(&mat).unwrap();
            dispatch_write(&from_mat, &tfm).unwrap();
            let from_tfm = read_transform(&tfm).unwrap();
            dispatch_write(&from_tfm, &h5).unwrap();
            let from_h5 = read_transform(&h5).unwrap();
            dispatch_write(&from_h5, &mat).unwrap();
            let round = read_transform(&mat).unwrap();

            for (label, other) in [
                ("mat", &from_mat),
                ("tfm", &from_tfm),
                ("h5", &from_h5),
                ("round", &round),
            ] {
                assert_eq!(other.itk_transform_type_name(), name, "{label}");
                assert_eq!(
                    other.parameters(),
                    transform.parameters(),
                    "{name} via {label}"
                );
                assert_eq!(
                    other.fixed_parameters(),
                    transform.fixed_parameters(),
                    "{name} via {label}"
                );
            }

            let _ = std::fs::remove_file(&mat);
            let _ = std::fs::remove_file(&tfm);
            let _ = std::fs::remove_file(&h5);
        }
    }

    // -- on-disk layout -------------------------------------------------------

    #[test]
    fn the_second_record_is_always_named_fixed_verbatim() {
        let transform: Transform = TranslationTransform::new(vec![1.0, 2.0]).into();
        let path = tmp_path("layout.mat");
        dispatch_write(&transform, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        // First record: type=0, rows=2, cols=1, imag=0, namlen=len("TranslationTransform_double_2_2")+1.
        let name = "TranslationTransform_double_2_2";
        assert_eq!(&bytes[0..4], &0i32.to_le_bytes());
        assert_eq!(&bytes[4..8], &2i32.to_le_bytes());
        assert_eq!(&bytes[8..12], &1i32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0i32.to_le_bytes());
        assert_eq!(&bytes[16..20], &((name.len() + 1) as i32).to_le_bytes());
        assert_eq!(&bytes[20..20 + name.len()], name.as_bytes());
        assert_eq!(bytes[20 + name.len()], 0);

        // Second record starts right after the first's name NUL + 2 f64 values.
        let second = 20 + name.len() + 1 + 2 * 8;
        assert_eq!(&bytes[second..second + 4], &0i32.to_le_bytes());
        assert_eq!(&bytes[second + 4..second + 8], &0i32.to_le_bytes()); // rows: no fixed params
        assert_eq!(&bytes[second + 8..second + 12], &1i32.to_le_bytes());
        assert_eq!(&bytes[second + 16..second + 20], &6i32.to_le_bytes()); // "fixed\0"
        assert_eq!(&bytes[second + 20..second + 25], b"fixed");
        assert_eq!(bytes.len(), second + 25 + 1);
    }

    // -- quirks -----------------------------------------------------------

    /// The second record's own name is written as `"fixed"` but never
    /// checked on read — any name works.
    #[test]
    fn the_fixed_record_name_is_never_validated_on_read() {
        let mut out = Vec::new();
        // Euler2DTransform: 3 parameters (angle, translation), 2 fixed (center).
        write_record(&mut out, "Euler2DTransform_double_2_2", &[0.1, 1.0, 2.0]);
        write_record(&mut out, "not_fixed_at_all", &[0.5, -0.5]);
        let path = tmp_path("unvalidated_fixed_name.mat");
        std::fs::write(&path, out).unwrap();
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read.fixed_parameters(), vec![0.5, -0.5]);
    }

    /// `is_single()` is decoupled from the transform's own precision suffix;
    /// a `f32`-precision record still reads back widened to `f64`.
    #[test]
    fn a_float_precision_file_reads_as_double() {
        let mut out = Vec::new();
        // type = 10 (single, column-wise, little-endian).
        out.extend_from_slice(&10i32.to_le_bytes());
        out.extend_from_slice(&2i32.to_le_bytes());
        out.extend_from_slice(&1i32.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        let name = "TranslationTransform_float_2_2";
        out.extend_from_slice(&((name.len() + 1) as i32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(&1.0f32.to_le_bytes());
        out.extend_from_slice(&2.0f32.to_le_bytes());
        write_record(&mut out, "fixed", &[]);

        let path = tmp_path("float_precision.mat");
        std::fs::write(&path, out).unwrap();
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            read.itk_transform_type_name(),
            "TranslationTransform_double_2_2"
        );
        assert_eq!(read.parameters(), vec![1.0, 2.0]);
    }

    /// `type = 1010` (single-precision, column-wise, the "big-endian" tag
    /// bit set) is one of the 8 legal `vnl_matlab_header::type_t`
    /// combinations, but upstream's own `switch` in `read_hdr` omits it —
    /// the one genuinely-native encoding its own list is missing (ledger
    /// §1.69, fixed in this port). Before the fix this fell to `need_swap =
    /// true`, byte-swapping `rows`/`cols` into huge garbage and failing the
    /// read; this pins that it is now recognized as native, exactly like its
    /// seven siblings (`a_float_precision_file_reads_as_double` for `10`).
    #[test]
    fn type_1010_is_recognized_as_native_and_needs_no_swap() {
        let mut out = Vec::new();
        out.extend_from_slice(&1010i32.to_le_bytes());
        out.extend_from_slice(&2i32.to_le_bytes());
        out.extend_from_slice(&1i32.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        let name = "TranslationTransform_float_2_2";
        out.extend_from_slice(&((name.len() + 1) as i32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(&1.0f32.to_le_bytes());
        out.extend_from_slice(&2.0f32.to_le_bytes());
        write_record(&mut out, "fixed", &[]);

        let path = tmp_path("type_1010.mat");
        std::fs::write(&path, out).unwrap();
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            read.itk_transform_type_name(),
            "TranslationTransform_double_2_2"
        );
        assert_eq!(read.parameters(), vec![1.0, 2.0]);
    }

    #[test]
    fn a_complex_valued_record_is_rejected() {
        let mut out = Vec::new();
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&2i32.to_le_bytes());
        out.extend_from_slice(&1i32.to_le_bytes());
        out.extend_from_slice(&1i32.to_le_bytes()); // imag != 0
        let name = "TranslationTransform_double_2_2";
        out.extend_from_slice(&((name.len() + 1) as i32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(&1.0f64.to_le_bytes());
        out.extend_from_slice(&2.0f64.to_le_bytes());

        let path = tmp_path("complex.mat");
        std::fs::write(&path, out).unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            result,
            Err(IoError::UnsupportedMatlabTransformFeature(_))
        ));
    }

    #[test]
    fn a_non_vector_first_header_is_rejected() {
        let mut out = Vec::new();
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&2i32.to_le_bytes());
        out.extend_from_slice(&2i32.to_le_bytes()); // cols = 2, not a vector
        out.extend_from_slice(&0i32.to_le_bytes());
        let name = "TranslationTransform_double_2_2";
        out.extend_from_slice(&((name.len() + 1) as i32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        for v in [1.0f64, 2.0, 3.0, 4.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }

        let path = tmp_path("non_vector.mat");
        std::fs::write(&path, out).unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(result, Err(IoError::MalformedTransformFile(m)) if m == "Only vector parameters supported")
        );
    }

    /// A file that ends right after the first (parameters) record — no
    /// second header at all — hits the ungated `cols() != 1` check against a
    /// zeroed header.
    #[test]
    fn a_missing_second_header_is_rejected() {
        let mut out = Vec::new();
        write_record(&mut out, "TranslationTransform_double_2_2", &[1.0, 2.0]);
        let path = tmp_path("missing_second_header.mat");
        std::fs::write(&path, out).unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(result, Err(IoError::MalformedTransformFile(m)) if m == "Only vector parameters supported")
        );
    }

    #[test]
    fn an_empty_file_has_no_transform() {
        let path = tmp_path("empty.mat");
        std::fs::write(&path, []).unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::NoTransformInFile(_))));
    }

    #[test]
    fn an_unknown_transform_type_string_is_rejected() {
        let mut out = Vec::new();
        write_record(&mut out, "NoSuchTransform_double_2_2", &[1.0, 2.0]);
        write_record(&mut out, "fixed", &[]);
        let path = tmp_path("unknown_type.mat");
        std::fs::write(&path, out).unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(result, Err(IoError::UnknownTransformType(name)) if name == "NoSuchTransform_double_2_2")
        );
    }

    #[test]
    fn a_parameter_count_that_does_not_match_the_transform_is_an_error() {
        let mut out = Vec::new();
        write_record(&mut out, "TranslationTransform_double_2_2", &[1.0]);
        write_record(&mut out, "fixed", &[]);
        let path = tmp_path("short_parameters.mat");
        std::fs::write(&path, out).unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            result,
            Err(IoError::Transform(
                crate::transform::TransformError::InvalidParameters {
                    got: 1,
                    expected: 2
                }
            ))
        ));
    }
}
