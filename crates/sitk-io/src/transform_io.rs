//! Insight legacy transform files (`.tfm` / `.txt`) — a line-by-line port of
//! `itk::TxtTransformIOTemplate<double>`
//! (`Modules/IO/TransformInsightLegacy/src/itkTxtTransformIO.cxx`), wrapped by
//! [`read_transform`] / [`write_transform`], SimpleITK's `ReadTransform` /
//! `WriteTransform` (`sitkTransform.cxx:668-737`).
//!
//! The format is:
//!
//! ```text
//! #Insight Transform File V1.0
//! #Transform 0
//! Transform: AffineTransform_double_2_2
//! Parameters: 1 0 0 1 3 4
//! FixedParameters: 0 0
//! ```
//!
//! Only the `double` precision variant exists here, matching SimpleITK, which
//! instantiates `TransformFileReader`/`Writer` on `double` alone. `.h5` /
//! `.hdf5` live in [`crate::transform_hdf5`], which [`read_transform`] and
//! [`write_transform`] dispatch to; the MATLAB `.mat` format is out of scope
//! (ledger §5.8).
//!
//! # Fidelity notes
//!
//! Numbers are emitted through the exact ECMAScript shortest-round-trip
//! algorithm ITK uses (`itk::NumberToString<double>` → double-conversion's
//! `EcmaScriptConverter().ToShortest()`), so output is byte-identical to ITK's;
//! see [`convert_number_to_string`].
//!
//! The reader reproduces the upstream parser's quirks rather than tightening
//! them — the header line is never validated, parameter lists truncate silently
//! at the first unparseable token, and a `Parameters:` line binds to the
//! most-recently-created transform, not to the one it syntactically follows.
//! Ledger rows §1.45, §2.78–§2.81, §3.30, §4.47 and §4.50 record each of these.

use std::fmt::Write as _;
use std::path::Path;

use sitk_transform::{
    AffineTransform, BSplineTransform, ComposeScaleSkewVersor3DTransform, CompositeTransform,
    DisplacementFieldTransform, Euler2DTransform, Euler3DTransform, ParametricTransform,
    ScaleLogarithmicTransform, ScaleSkewVersor3DTransform, ScaleTransform, ScaleVersor3DTransform,
    Similarity2DTransform, Similarity3DTransform, Transform, TransformBase, TranslationTransform,
    VersorRigid3DTransform, VersorTransform,
};

use crate::error::{IoError, Result};
use crate::transform_hdf5;

/// How deep `ComponentTransformFile:` references may nest before the reader
/// gives up. ITK has no such limit and recurses until the C++ stack is
/// exhausted on a self-referencing file; see ledger §4.50.
const MAX_COMPONENT_DEPTH: usize = 32;

// ---------------------------------------------------------------------------
// Number formatting — itk::NumberToString<double>
// ---------------------------------------------------------------------------

/// `itk::ConvertNumberToString(double)` — the shortest decimal string that
/// round-trips to `value`, formatted exactly as ECMAScript's
/// `Number.prototype.toString` (`itkNumberToString.cxx:56-72`, which calls
/// double-conversion's `DoubleToStringConverter::EcmaScriptConverter()` with
/// `ToShortest`).
///
/// That converter is configured with `UNIQUE_ZERO | EMIT_POSITIVE_EXPONENT_SIGN`,
/// the symbols `"Infinity"` / `"NaN"`, exponent character `'e'`, and the
/// decimal-notation window `-6 < decimal_point <= 21`. Outside that window the
/// exponential form `d.ddde±X` is used.
///
/// ```text
///  1.0        -> "1"              0.001 -> "0.001"
///  1e20       -> "100000000000000000000"
///  1e21       -> "1e+21"          1e-7  -> "1e-7"
/// -0.0        -> "0"              1e-6  -> "0.000001"
/// ```
///
/// Rust's `{:e}` already produces the shortest round-tripping digit string, so
/// the only work here is re-laying it out the way double-conversion does.
pub(crate) fn convert_number_to_string(value: f64) -> String {
    // DoubleToStringConverter::HandleSpecialValues: NaN prints unsigned.
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-Infinity" } else { "Infinity" }.to_string();
    }
    // UNIQUE_ZERO: the sign of -0.0 is dropped.
    if value == 0.0 {
        return "0".to_string();
    }

    let negative = value < 0.0;
    let (digits, decimal_point) = shortest_digits(value.abs());
    let len = digits.len() as i32;

    let mut out = String::with_capacity(digits.len() + 8);
    if negative {
        out.push('-');
    }

    if -6 < decimal_point && decimal_point <= 21 {
        // CreateDecimalRepresentation, with digits_after_point = max(0, len - decimal_point).
        if decimal_point <= 0 {
            out.push_str("0.");
            for _ in 0..-decimal_point {
                out.push('0');
            }
            out.push_str(&digits);
        } else if decimal_point >= len {
            out.push_str(&digits);
            for _ in 0..decimal_point - len {
                out.push('0');
            }
        } else {
            let split = decimal_point as usize;
            out.push_str(&digits[..split]);
            out.push('.');
            out.push_str(&digits[split..]);
        }
    } else {
        // CreateExponentialRepresentation.
        let exponent = decimal_point - 1;
        out.push_str(&digits[..1]);
        if len > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let _ = write!(out, "{}", exponent.abs());
    }
    out
}

/// The shortest round-tripping decimal digits of a finite, strictly positive
/// `value`, plus double-conversion's `decimal_point`: the number of digits that
/// precede the decimal separator (`value = 0.<digits> * 10^decimal_point`).
fn shortest_digits(value: f64) -> (String, i32) {
    // `{:e}` is Rust's shortest-round-trip scientific form: "d[.ddd]e<exp>".
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("{:e} always emits an exponent");
    let exponent: i32 = exponent.parse().expect("{:e} exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    (digits, exponent + 1)
}

// ---------------------------------------------------------------------------
// Number parsing — vnl_vector<double>::read_ascii
// ---------------------------------------------------------------------------

/// `std::istringstream >> itk::Array<double>` — i.e. `vnl_vector::read_ascii`
/// on a zero-sized vector (`vnl_vector.hxx:266-278`):
///
/// ```cpp
/// while (s >> value) { allvals.push_back(value); }
/// ```
///
/// `operator>>(istream&, double&)` consumes the longest prefix of the next
/// whitespace-delimited run that forms a number, so `"1 2 3junk 4"` yields
/// `[1, 2, 3]` — the `"junk"` left in the stream fails the next extraction and
/// the loop stops. Nothing is reported: the list is silently truncated
/// (ledger §2.78).
fn parse_doubles(value: &str) -> Vec<f64> {
    let mut out = Vec::new();
    for token in value.split_ascii_whitespace() {
        match longest_f64_prefix(token) {
            Some((v, consumed)) => {
                out.push(v);
                if consumed < token.len() {
                    // The stream stopped inside this token; the next extraction
                    // sees the unconsumed tail and fails.
                    break;
                }
            }
            None => break,
        }
    }
    out
}

/// The longest prefix of `token` that parses as an `f64`, with its byte length.
fn longest_f64_prefix(token: &str) -> Option<(f64, usize)> {
    for end in (1..=token.len()).rev() {
        if !token.is_char_boundary(end) {
            continue;
        }
        if let Ok(v) = token[..end].parse::<f64>() {
            return Some((v, end));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Transform type names and the factory
// ---------------------------------------------------------------------------

/// `TransformIOBaseTemplate<double>::CorrectTransformPrecisionType`
/// (`itkTransformIOBase.h:197-207`): a name that does not already mention
/// `double` has its first `float` rewritten to `double`.
///
/// Upstream calls `std::string::replace(npos, 5, "double")` when the name
/// mentions neither, which throws `std::out_of_range` — an `#Insight Transform
/// File` naming, say, `TranslationTransform_2_2` aborts the read with a
/// std-library exception rather than an ITK one (ledger §1.45). Here that is a
/// plain [`IoError::UnknownTransformType`].
pub(crate) fn correct_transform_precision_type(name: &str) -> Result<String> {
    if name.contains("double") {
        return Ok(name.to_string());
    }
    match name.find("float") {
        Some(begin) => Ok(format!("{}double{}", &name[..begin], &name[begin + 5..])),
        None => Err(IoError::UnknownTransformType(name.to_string())),
    }
}

/// Split `"AffineTransform_double_2_2"` into `("AffineTransform", 2)`, rejecting
/// anything that is not a `double`, square-dimensioned ITK transform name.
fn split_transform_type_name(name: &str) -> Result<(&str, usize)> {
    let bad = || IoError::UnknownTransformType(name.to_string());
    let (head, output_dim) = name.rsplit_once('_').ok_or_else(bad)?;
    let (head, input_dim) = head.rsplit_once('_').ok_or_else(bad)?;
    let (class_name, precision) = head.rsplit_once('_').ok_or_else(bad)?;
    let input_dim: usize = input_dim.parse().map_err(|_| bad())?;
    let output_dim: usize = output_dim.parse().map_err(|_| bad())?;
    if precision != "double" || input_dim != output_dim || class_name.is_empty() {
        return Err(bad());
    }
    Ok((class_name, input_dim))
}

fn identity_matrix(dim: usize) -> Vec<f64> {
    let mut m = vec![0.0; dim * dim];
    for i in 0..dim {
        m[i * dim + i] = 1.0;
    }
    m
}

/// `TransformIOBaseTemplate::CreateTransform` — build the identity instance of
/// the named transform, whose parameters the caller then overwrites.
///
/// ITK looks the name up in `TransformFactoryBase`'s registry and throws
/// `"Unregistered transform type"` on a miss; the registry holds the same
/// families this crate implements, restricted here to the 2D/3D instantiations
/// SimpleITK exposes.
pub(crate) fn create_transform(type_name: &str) -> Result<Transform> {
    let (class_name, dim) = split_transform_type_name(type_name)?;
    let unsupported = || IoError::UnknownTransformType(type_name.to_string());

    let zeros2 = [0.0; 2];
    let zeros3 = [0.0; 3];

    Ok(match (class_name, dim) {
        ("TranslationTransform", d) => TranslationTransform::new(vec![0.0; d]).into(),
        ("ScaleTransform", d) => ScaleTransform::new(vec![1.0; d], vec![0.0; d]).into(),
        ("ScaleLogarithmicTransform", d) => ScaleLogarithmicTransform::identity(d).into(),
        ("AffineTransform", d) => {
            AffineTransform::new(d, identity_matrix(d), vec![0.0; d], vec![0.0; d]).into()
        }
        ("Euler2DTransform", 2) => Euler2DTransform::new(0.0, zeros2, zeros2).into(),
        ("Similarity2DTransform", 2) => Similarity2DTransform::new(1.0, 0.0, zeros2, zeros2).into(),
        ("Euler3DTransform", 3) => Euler3DTransform::new(0.0, 0.0, 0.0, zeros3, zeros3).into(),
        ("Similarity3DTransform", 3) => {
            Similarity3DTransform::new(1.0, 0.0, 0.0, 0.0, zeros3, zeros3).into()
        }
        ("VersorTransform", 3) => VersorTransform::new(0.0, 0.0, 0.0, zeros3).into(),
        ("VersorRigid3DTransform", 3) => {
            VersorRigid3DTransform::new(0.0, 0.0, 0.0, zeros3, zeros3).into()
        }
        ("ScaleVersor3DTransform", 3) => {
            ScaleVersor3DTransform::new([1.0; 3], 0.0, 0.0, 0.0, zeros3, zeros3).into()
        }
        ("ScaleSkewVersor3DTransform", 3) => {
            ScaleSkewVersor3DTransform::new([1.0; 3], [0.0; 6], 0.0, 0.0, 0.0, zeros3, zeros3)
                .into()
        }
        ("ComposeScaleSkewVersor3DTransform", 3) => ComposeScaleSkewVersor3DTransform::new(
            [1.0; 3], [0.0; 3], 0.0, 0.0, 0.0, zeros3, zeros3,
        )
        .into(),
        ("CompositeTransform", d) => CompositeTransform::new(d).into(),
        ("DisplacementFieldTransform", d) => DisplacementFieldTransform::new(
            d,
            &vec![1; d],
            &vec![0.0; d],
            &vec![1.0; d],
            &identity_matrix(d),
        )?
        .into(),
        ("BSplineTransform", d) => BSplineTransform::new(
            d,
            &vec![0.0; d],
            &vec![1.0; d],
            &identity_matrix(d),
            &vec![1; d],
        )?
        .into(),
        _ => return Err(unsupported()),
    })
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// `TxtTransformIOTemplate::trim(source, " \t\r\n")` (`itkTxtTransformIO.cxx:60-79`).
fn trim(source: &str) -> &str {
    source.trim_matches([' ', '\t', '\r', '\n'])
}

/// `TxtTransformIOTemplate::Read` (`itkTxtTransformIO.cxx:107-224`) — parse a
/// file into ITK's flat `ReadTransformList`, before the composite folding
/// `TransformFileReader::Update` applies.
fn read_transform_list(path: &Path, depth: usize) -> Result<Vec<Transform>> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            IoError::FileNotFound(path.to_path_buf())
        } else {
            IoError::Io(e)
        }
    })?;

    let mut list: Vec<Transform> = Vec::new();
    // ITK's `transform` local: the most recently *created* transform, which is
    // what every later `Parameters:` / `FixedParameters:` line binds to — even
    // one belonging, by reading order, to a different transform (ledger §2.79).
    let mut current: Option<usize> = None;
    let mut tmp_parameters: Vec<f64> = Vec::new();
    let mut tmp_fixed_parameters: Vec<f64> = Vec::new();
    let mut have_parameters = false;
    let mut have_fixed_parameters = false;

    for raw_line in text.lines() {
        let line = trim(raw_line);
        if line.is_empty() || line.starts_with('#') {
            // ITK also skips all-whitespace lines here; `trim` has already made
            // those empty. The header line is never checked (ledger §2.80).
            continue;
        }

        let Some((name, value)) = line.split_once(':') else {
            return Err(IoError::MalformedTransformFile(
                "Tags must be delimited by :".to_string(),
            ));
        };
        let name = trim(name);
        let value = trim(value);

        match name {
            "Transform" => {
                let type_name = correct_transform_precision_type(value)?;
                list.push(create_transform(&type_name)?);
                current = Some(list.len() - 1);
            }
            "ComponentTransformFile" => {
                if depth >= MAX_COMPONENT_DEPTH {
                    return Err(IoError::MalformedTransformFile(format!(
                        "ComponentTransformFile nesting exceeded {MAX_COMPONENT_DEPTH} levels"
                    )));
                }
                list.push(read_component_file(path, value, depth + 1)?);
                // Upstream does *not* update `transform` here: parameter lines
                // after a ComponentTransformFile still bind to whatever
                // `Transform:` line preceded it (ledger §2.79).
            }
            "Parameters" | "FixedParameters" => {
                let buffer = parse_doubles(value);
                if name == "Parameters" {
                    tmp_parameters = buffer;
                    if have_fixed_parameters {
                        apply(&mut list, current, &tmp_parameters, &tmp_fixed_parameters)?;
                        tmp_parameters = Vec::new();
                        tmp_fixed_parameters = Vec::new();
                        have_fixed_parameters = false;
                        have_parameters = false;
                    } else {
                        have_parameters = true;
                    }
                } else {
                    tmp_fixed_parameters = buffer;
                    if current.is_none() {
                        return Err(IoError::MalformedTransformFile(
                            "Please set the transform before parametersor fixed parameters"
                                .to_string(),
                        ));
                    }
                    if have_parameters {
                        apply(&mut list, current, &tmp_parameters, &tmp_fixed_parameters)?;
                        tmp_parameters = Vec::new();
                        tmp_fixed_parameters = Vec::new();
                        have_fixed_parameters = false;
                        have_parameters = false;
                    } else {
                        have_fixed_parameters = true;
                    }
                }
            }
            // Unknown tags are ignored, as upstream's if/else-if chain does.
            _ => {}
        }
    }

    Ok(list)
}

/// The two `SetFixedParameters` / `SetParametersByValue` calls ITK makes once
/// both lines have been seen — fixed parameters **first**, because they carry
/// the centre and grid geometry the parameters are interpreted against
/// (`itkTxtTransformIO.cxx:186-217`).
///
/// The parameter count is [`ParametricTransform::set_parameters`]'s own check
/// (ledger §5.16): ITK's own checks are per-class and inconsistent —
/// `MatrixOffsetTransformBase` and `TranslationTransform` throw only when the
/// vector is *shorter* than needed and silently ignore trailing values,
/// `VersorTransform` checks nothing at all, and `BSplineTransform` demands
/// exact equality. This port demands exact equality everywhere (ledger §4.47),
/// uniformly, inside `set_parameters` itself.
fn apply(
    list: &mut [Transform],
    current: Option<usize>,
    parameters: &[f64],
    fixed_parameters: &[f64],
) -> Result<()> {
    let index = current.ok_or_else(|| {
        IoError::MalformedTransformFile(
            "Please set the transform before parametersor fixed parameters".to_string(),
        )
    })?;
    let transform = &mut list[index];
    transform.set_fixed_parameters(fixed_parameters)?;
    transform.set_parameters(parameters)?;
    Ok(())
}

/// `TxtTransformIOTemplate::ReadComponentFile` (`itkTxtTransformIO.cxx:82-104`):
/// the component file name is resolved against the master file's directory and
/// read with a fresh reader; only its *first* transform is taken.
fn read_component_file(master: &Path, value: &str, depth: usize) -> Result<Transform> {
    let directory = master.parent().unwrap_or_else(|| Path::new(""));
    let full_path = directory.join(value);
    let list = read_and_fold(&full_path, depth)?;
    list.into_iter()
        .next()
        .ok_or(IoError::NoTransformInFile(full_path))
}

/// `TransformFileReaderTemplate::Update` (`itkTransformFileReader.cxx:129-160`)
/// minus the `KernelTransform` special case: if the first transform read is a
/// `CompositeTransform`, every following transform is added to it and it alone
/// is returned.
///
/// `TransformIOFactory::CreateTransformIO` polls every registered IO's
/// `CanReadFile` in registration order. Only two are ported: the Insight legacy
/// text IO, which looks at the extension alone, and
/// [`crate::transform_hdf5`], which looks at the *content* alone. This port
/// asks the text IO first, so an HDF5 file misnamed `.tfm` is parsed as text
/// and fails, where ITK's answer depends on its factory registration order
/// (ledger §4.76).
fn read_and_fold(path: &Path, depth: usize) -> Result<Vec<Transform>> {
    let list = if can_handle(path) {
        read_transform_list(path, depth)?
    } else if transform_hdf5::can_read_file(path) {
        transform_hdf5::read_transform_list(path)?
    } else {
        return Err(IoError::NoTransformReaderFound(path.to_path_buf()));
    };
    if list.is_empty() {
        return Err(IoError::NoTransformInFile(path.to_path_buf()));
    }

    let mut iter = list.into_iter();
    let first = iter.next().expect("checked non-empty");
    let Transform::Composite(mut composite) = first else {
        return Ok(std::iter::once(first).chain(iter).collect());
    };
    for transform in iter {
        composite.add_transform(transform)?;
    }
    Ok(vec![composite.into()])
}

/// `TxtTransformIOTemplate::CanReadFile` / `CanWriteFile`
/// (`itkTxtTransformIO.cxx:37-56`): the `.txt` and `.tfm` extensions.
fn can_handle(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("txt") | Some("tfm")
    )
}

/// Read a transform from an Insight legacy transform file (`.tfm` / `.txt`) or
/// an HDF5 one (any file holding a `/TransformGroup`, conventionally `.h5` /
/// `.hdf5`) — `itk::simple::ReadTransform` (`sitkTransform.cxx:668-723`).
///
/// A file may hold several transforms; as upstream, only the first is returned
/// (SimpleITK prints a warning here, which this port has no channel for —
/// ledger §3.30). A `CompositeTransform` at the head of the file absorbs the
/// rest and is returned whole.
///
/// Only 2D and 3D transforms are supported, matching `ReadTransform`'s final
/// `sitkExceptionMacro`.
///
/// ```no_run
/// use sitk_transform::TransformBase;
///
/// let transform = sitk_io::read_transform("affine.tfm")?;
/// println!("{}", transform.dimension());
/// # Ok::<(), sitk_io::IoError>(())
/// ```
pub fn read_transform<P: AsRef<Path>>(path: P) -> Result<Transform> {
    let path = path.as_ref();
    let list = read_and_fold(path, 0)?;
    let transform = list.into_iter().next().expect("read_and_fold is non-empty");
    match transform.dimension() {
        2 | 3 => Ok(transform),
        dimension => Err(IoError::UnsupportedTransformDimension(dimension)),
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// `itk_impl_details::print_vector` (`itkTxtTransformIO.cxx:227-239`): values
/// separated by a single space, with no trailing space — so an empty vector
/// contributes nothing, leaving the `"FixedParameters: "` prefix's own trailing
/// space at the end of the line (ledger §2.81).
fn print_vector(out: &mut String, values: &[f64]) {
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&convert_number_to_string(*value));
    }
}

/// Write a transform to an Insight legacy transform file (`.tfm` / `.txt`) —
/// `itk::simple::WriteTransform` (`sitkTransform.cxx:731-737`) over
/// `TxtTransformIOTemplate::Write` (`itkTxtTransformIO.cxx:242-296`) — or to an
/// HDF5 one, for the eight extensions [`crate::transform_hdf5`] claims.
///
/// A [`CompositeTransform`] is written as a `CompositeTransform` line with no
/// parameters, followed by each of its sub-transforms in queue order — the
/// expansion `CompositeTransformIOHelper::GetTransformList` performs. A
/// composite nested *inside* a composite is rejected, as upstream's
/// `"Composite Transform can only be 1st transform in a file"`.
///
/// ```no_run
/// use sitk_transform::{Transform, TranslationTransform};
///
/// let transform: Transform = TranslationTransform::new(vec![1.0, 2.0]).into();
/// sitk_io::write_transform(&transform, "translation.tfm")?;
/// # Ok::<(), sitk_io::IoError>(())
/// ```
pub fn write_transform<P: AsRef<Path>>(transform: &Transform, path: P) -> Result<()> {
    let path = path.as_ref();
    if can_handle(path) {
        std::fs::write(path, serialize(transform)?)?;
        Ok(())
    } else if transform_hdf5::can_write_file(path) {
        transform_hdf5::write_transform(transform, path)
    } else {
        Err(IoError::NoTransformWriterFound(path.to_path_buf()))
    }
}

/// The exact byte content [`write_transform`] puts on disk.
fn serialize(transform: &Transform) -> Result<String> {
    // CompositeTransformIOHelper::GetTransformList: the composite itself first,
    // then its transform queue front-to-back.
    let list: Vec<&Transform> = match transform {
        Transform::Composite(composite) => std::iter::once(transform)
            .chain(composite.transforms())
            .collect(),
        _ => vec![transform],
    };

    let mut out = String::from("#Insight Transform File V1.0\n");
    for (count, transform) in list.iter().enumerate() {
        let _ = writeln!(out, "#Transform {count}");
        let _ = writeln!(out, "Transform: {}", transform.itk_transform_type_name());
        if matches!(transform, Transform::Composite(_)) {
            if count > 0 {
                return Err(IoError::MalformedTransformFile(
                    "Composite Transform can only be 1st transform in a file".to_string(),
                ));
            }
            continue;
        }
        out.push_str("Parameters: ");
        print_vector(&mut out, &transform.parameters());
        out.push('\n');
        out.push_str("FixedParameters: ");
        print_vector(&mut out, &transform.fixed_parameters());
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("sitk_rs_transform_io_{name}"));
        path
    }

    // -- itk::NumberToString<double> ---------------------------------------

    #[test]
    fn ecmascript_shortest_matches_javascript_number_to_string() {
        // Values chosen to hit each branch of DoubleToStringConverter::ToShortest.
        let cases: &[(f64, &str)] = &[
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (-1.0, "-1"),
            (100.0, "100"),
            (1.5, "1.5"),
            (0.5, "0.5"),
            (0.001, "0.001"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (1e20, "100000000000000000000"),
            (1e21, "1e+21"),
            (1.23e22, "1.23e+22"),
            (f64::MAX, "1.7976931348623157e+308"),
            (-f64::MAX, "-1.7976931348623157e+308"),
            (-3e-5 / 9.0, "-0.0000033333333333333333"),
            (1.0 / 3.0, "0.3333333333333333"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (f64::NAN, "NaN"),
        ];
        for &(value, expected) in cases {
            assert_eq!(convert_number_to_string(value), expected, "for {value:?}");
        }
    }

    #[test]
    fn every_written_number_round_trips_exactly() {
        let values = [
            std::f64::consts::PI,
            1.0 / 7.0,
            -2.220446049250313e-16,
            5e-324,
            1.7976931348623157e308,
        ];
        for value in values {
            let text = convert_number_to_string(value);
            assert_eq!(text.parse::<f64>().unwrap(), value, "{text}");
        }
    }

    // -- vnl_vector<double>::read_ascii ------------------------------------

    #[test]
    fn parameters_parse_until_the_first_bad_token() {
        assert_eq!(parse_doubles("1 2 3"), vec![1.0, 2.0, 3.0]);
        assert_eq!(parse_doubles(""), Vec::<f64>::new());
        assert_eq!(parse_doubles("1 junk 3"), vec![1.0]);
        // "3junk" extracts 3, then the leftover "junk" fails the next read.
        assert_eq!(parse_doubles("1 2 3junk 4"), vec![1.0, 2.0, 3.0]);
        assert_eq!(parse_doubles("1e2 -3.5"), vec![100.0, -3.5]);
    }

    // -- CorrectTransformPrecisionType -------------------------------------

    #[test]
    fn float_transform_names_are_rewritten_to_double() {
        assert_eq!(
            correct_transform_precision_type("AffineTransform_float_2_2").unwrap(),
            "AffineTransform_double_2_2"
        );
        assert_eq!(
            correct_transform_precision_type("AffineTransform_double_2_2").unwrap(),
            "AffineTransform_double_2_2"
        );
        // Upstream throws std::out_of_range here (ledger §1.45).
        assert!(correct_transform_precision_type("AffineTransform_2_2").is_err());
    }

    // -- round trips --------------------------------------------------------

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

    #[test]
    fn every_transform_round_trips_through_a_tfm_file() {
        for (i, transform) in every_transform().into_iter().enumerate() {
            let path = tmp_path(&format!("roundtrip_{i}.tfm"));
            write_transform(&transform, &path).unwrap();
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
    fn affine_2d_output_is_byte_pinned() {
        let transform: Transform = AffineTransform::new(
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![3.0, 4.0],
            vec![0.5, -0.25],
        )
        .into();
        assert_eq!(
            serialize(&transform).unwrap(),
            "#Insight Transform File V1.0\n\
             #Transform 0\n\
             Transform: AffineTransform_double_2_2\n\
             Parameters: 1 0 0 1 3 4\n\
             FixedParameters: 0.5 -0.25\n"
        );
    }

    #[test]
    fn a_transform_with_no_fixed_parameters_leaves_a_trailing_space() {
        let transform: Transform = TranslationTransform::new(vec![1.0, -2.5]).into();
        assert_eq!(
            serialize(&transform).unwrap(),
            "#Insight Transform File V1.0\n\
             #Transform 0\n\
             Transform: TranslationTransform_double_2_2\n\
             Parameters: 1 -2.5\n\
             FixedParameters: \n"
        );
    }

    #[test]
    fn composite_round_trips_with_its_queue_in_order() {
        let mut composite = CompositeTransform::new(2);
        composite
            .add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        composite
            .add_transform(Euler2DTransform::new(0.25, [3.0, 4.0], [0.5, 0.5]).into())
            .unwrap();
        let transform: Transform = composite.into();

        let path = tmp_path("composite.tfm");
        write_transform(&transform, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(
            text,
            "#Insight Transform File V1.0\n\
             #Transform 0\n\
             Transform: CompositeTransform_double_2_2\n\
             #Transform 1\n\
             Transform: TranslationTransform_double_2_2\n\
             Parameters: 1 2\n\
             FixedParameters: \n\
             #Transform 2\n\
             Transform: Euler2DTransform_double_2_2\n\
             Parameters: 0.25 3 4\n\
             FixedParameters: 0.5 0.5\n"
        );
        assert_eq!(read, transform);
    }

    #[test]
    fn a_composite_may_not_be_nested() {
        let mut inner = CompositeTransform::new(2);
        inner
            .add_transform(TranslationTransform::new(vec![1.0, 2.0]).into())
            .unwrap();
        let mut outer = CompositeTransform::new(2);
        outer.add_transform(inner.into()).unwrap();
        let transform: Transform = outer.into();
        assert!(matches!(
            serialize(&transform),
            Err(IoError::MalformedTransformFile(_))
        ));
    }

    #[test]
    fn a_line_without_a_colon_is_an_error() {
        let path = tmp_path("no_colon.tfm");
        std::fs::write(&path, "#Insight Transform File V1.0\nnonsense\n").unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::MalformedTransformFile(_))));
    }

    #[test]
    fn fixed_parameters_before_any_transform_is_an_error() {
        let path = tmp_path("early_fixed.tfm");
        std::fs::write(&path, "FixedParameters: 0 0\n").unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::MalformedTransformFile(_))));
    }

    /// `.mat` is `MatlabTransformIO`'s, which this crate does not port; no other
    /// IO claims it, so both directions fail as `TransformFileReader` /
    /// `Writer` do when `CreateTransformIO` returns null.
    #[test]
    fn an_unknown_extension_is_rejected() {
        let transform: Transform = TranslationTransform::new(vec![1.0, 2.0]).into();
        assert!(matches!(
            write_transform(&transform, tmp_path("bad.mat")),
            Err(IoError::NoTransformWriterFound(_))
        ));
        assert!(matches!(
            read_transform(tmp_path("bad.mat")),
            Err(IoError::NoTransformReaderFound(_))
        ));
    }

    /// A `.h5` path that does not exist is not readable — `H5Fis_hdf5` fails and
    /// `HDF5TransformIO::CanReadFile`'s `catch (...)` returns false — but it is
    /// writable, since `CanWriteFile` looks only at the extension.
    #[test]
    fn a_missing_hdf5_file_is_not_readable() {
        assert!(matches!(
            read_transform(tmp_path("missing.h5")),
            Err(IoError::NoTransformReaderFound(_))
        ));
    }

    #[test]
    fn an_empty_file_has_no_transform() {
        let path = tmp_path("empty.tfm");
        std::fs::write(&path, "#Insight Transform File V1.0\n").unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(IoError::NoTransformInFile(_))));
    }

    #[test]
    fn a_float_precision_file_reads_as_double() {
        let path = tmp_path("float_precision.tfm");
        std::fs::write(
            &path,
            "#Insight Transform File V1.0\n\
             Transform: TranslationTransform_float_2_2\n\
             Parameters: 1 2\n\
             FixedParameters: \n",
        )
        .unwrap();
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read.parameters(), vec![1.0, 2.0]);
    }

    /// Verbatim `ITK/Examples/Data/IdentityTransform.tfm`, which is hand-written
    /// rather than writer-produced: `"# Transform 0"` has a space the writer never
    /// emits, values carry redundant `".0"` suffixes, and both value lists have
    /// runs of extra spaces.
    #[test]
    fn itks_own_identity_transform_example_reads() {
        let path = tmp_path("identity_example.tfm");
        std::fs::write(
            &path,
            "#Insight Transform File V1.0\n\
             # Transform 0\n\
             Transform: AffineTransform_double_3_3\n\
             Parameters: 1 0 0 0 1.0 0.0 0 0.0 1.0  0  0  0\n\
             FixedParameters:  0.0 0.0 0.0\n",
        )
        .unwrap();
        let read = read_transform(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(read.itk_transform_type_name(), "AffineTransform_double_3_3");
        assert_eq!(
            read.parameters(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(read.fixed_parameters(), vec![0.0, 0.0, 0.0]);
        assert_eq!(
            read.transform_point(&[3.0, -1.0, 2.0]),
            vec![3.0, -1.0, 2.0]
        );
    }

    #[test]
    fn a_parameter_count_that_does_not_match_the_transform_is_an_error() {
        let path = tmp_path("short_parameters.tfm");
        std::fs::write(
            &path,
            "#Insight Transform File V1.0\n\
             Transform: TranslationTransform_double_2_2\n\
             Parameters: 1\n\
             FixedParameters: \n",
        )
        .unwrap();
        let result = read_transform(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            result,
            Err(IoError::Transform(
                sitk_transform::TransformError::InvalidParameters {
                    got: 1,
                    expected: 2
                }
            ))
        ));
    }

    #[test]
    fn parameters_bind_to_the_most_recently_created_transform() {
        // Ledger §2.79: the `Parameters:` line below belongs, by reading order,
        // to the first transform; ITK applies it to the second.
        let path = tmp_path("misbinding.tfm");
        std::fs::write(
            &path,
            "#Insight Transform File V1.0\n\
             Transform: TranslationTransform_double_2_2\n\
             Parameters: 7 8\n\
             Transform: TranslationTransform_double_2_2\n\
             FixedParameters: \n",
        )
        .unwrap();
        let list = read_transform_list(&path, 0).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].parameters(), vec![0.0, 0.0]);
        assert_eq!(list[1].parameters(), vec![7.0, 8.0]);
    }

    #[test]
    fn a_component_transform_file_is_read_from_the_master_directory() {
        let component = tmp_path("component_child.tfm");
        std::fs::write(
            &component,
            "#Insight Transform File V1.0\n\
             Transform: TranslationTransform_double_2_2\n\
             Parameters: 5 6\n\
             FixedParameters: \n",
        )
        .unwrap();
        let master = tmp_path("component_master.tfm");
        std::fs::write(
            &master,
            "#Insight Transform File V1.0\n\
             Transform: CompositeTransform_double_2_2\n\
             ComponentTransformFile: sitk_rs_transform_io_component_child.tfm\n",
        )
        .unwrap();

        let read = read_transform(&master).unwrap();
        let _ = std::fs::remove_file(&master);
        let _ = std::fs::remove_file(&component);

        let Transform::Composite(composite) = read else {
            panic!("expected a composite transform");
        };
        assert_eq!(composite.transforms().len(), 1);
        assert_eq!(composite.transforms()[0].parameters(), vec![5.0, 6.0]);
    }
}
