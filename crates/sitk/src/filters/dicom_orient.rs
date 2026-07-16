//! `DICOMOrientImageFilter`: permute and flip a 3-D image's axes to reach a
//! desired DICOM-style patient orientation (e.g. `"LPS"`, `"RAS"`), updating
//! its direction cosines and origin to match, without touching the physical
//! location of any pixel.
//!
//! Ported from `itkDICOMOrientImageFilter.h` / `.hxx` (the filter proper),
//! `itkDICOMOrientation.h` / `itkDICOMOrientation.cxx` (the orientation-code
//! parsing/permutation machinery), and `sitkDICOMOrientImageFilter_Support.cxx`
//! (the two public statics `GetOrientationFromDirectionCosines` /
//! `GetDirectionCosinesFromOrientation`, ported as free functions
//! [`get_orientation_from_direction_cosines`] /
//! [`get_direction_cosines_from_orientation`]).
//!
//! `DICOMOrientImageFilter.yaml` registers this filter for 3-D images only
//! (`custom_register: factory.RegisterMemberFunctions<PixelIDTypeList, 3>()`),
//! over `typelist2::append<BasicPixelIDTypeList, VectorPixelIDTypeList>` —
//! both scalar and vector pixel types. [`dicom_orient`] rejects non-3-D input
//! with [`FilterError::UnsupportedDicomOrientDimension`], matching
//! `itkDICOMOrientImageFilter.h:142`'s `static_assert(ImageDimension == 3,
//! ...)`.
//!
//! ## Orientation model: terms, not a packed code
//!
//! Upstream's `DICOMOrientation::OrientationEnum` packs three axis terms
//! (`CoordinateEnum`, `itkDICOMOrientation.h:47-56`) into one `uint32_t`, and
//! a 48-entry `std::map<std::string, OrientationEnum>`
//! (`itkDICOMOrientation.cxx:74-163`) both parses a 3-letter code and prints
//! one back. This port drops the packed code and the table: `Orientation`
//! is just `Option<[Coordinate; 3]>` (`None` for `INVALID`), `Coordinate` is
//! one of `{Right, Left, Anterior, Posterior, Inferior, Superior}`, and the
//! string form is computed directly from the terms' single-letter labels in
//! both directions. This is algebraically equivalent to the table (every one
//! of the 48 valid codes is exactly one letter from each of the three axis
//! families `{L,R}`/`{A,P}`/`{I,S}`, in some order and with some sign — 3! ×
//! 2³ = 48 — which is exactly what parsing-then-validating produces), pinned
//! by `orientation_string_round_trips_every_upstream_code` below against the
//! literal 48-string list from `itkDICOMOrientation.cxx:112-160`. The one
//! behavior a naive term-by-term parse would get wrong — an unrecognized
//! character silently becoming a term that then passes the axis-family check
//! — is closed by rejecting any string whose length isn't exactly 3, or that
//! contains a character outside `RLAPIS`, before ever constructing terms
//! (`Orientation::from_str_ci`).
//!
//! ## Upstream findings
//!
//! - **`GetDirectionCosinesFromOrientation` never validates its string.**
//!   `sitkDICOMOrientImageFilter_Support.cxx:40-48` builds a `DICOMOrientation`
//!   directly from the caller's string and returns `GetAsDirection()` with no
//!   check at all; an unparseable string silently yields an all-zero 3x3
//!   matrix (`OrientationToDirectionCosines`'s `switch` on
//!   `CoordinateEnum::UNKNOWN` assigns nothing, `itkDICOMOrientation.cxx:
//!   258-259`). This is distinct from the *filter's* path
//!   (`SetDesiredCoordinateOrientation`), which does eventually reject an
//!   invalid string via `VerifyPreconditions` — the static utility function
//!   has no such gate. Reproduced as-is in
//!   [`get_direction_cosines_from_orientation`].
//! - **`GetOrientationFromDirectionCosines` treats an empty vector as
//!   identity.** The conversion goes through `sitkSTLToITKDirection`
//!   (`sitkTemplateFunctions.h:187-207`), whose empty-input branch calls
//!   `SetIdentity()` rather than raising the length-mismatch exception it uses
//!   for every other wrong length. An empty `direction` therefore resolves to
//!   `"LPS"`, not an error. Reproduced in
//!   [`get_orientation_from_direction_cosines`]; any other length but 9
//!   errors with [`FilterError::InvalidDirectionCosinesLength`].
//! - **The filter's own invalid-string path is two-stage.**
//!   `SetDesiredCoordinateOrientation(const std::string&)`
//!   (`itkDICOMOrientImageFilter.hxx:156-167`) warns (`itkWarningMacro`) on an
//!   unparseable string but still *applies* the resulting `INVALID` value; the
//!   actual failure is deferred to `VerifyPreconditions()`
//!   (`itkDICOMOrientImageFilter.hxx:296-306`), which only runs when the
//!   pipeline updates. This port has no persistent filter object to warn
//!   through and no delayed `Update()` — [`dicom_orient`] collapses both
//!   stages into one upfront [`FilterError::InvalidDesiredOrientation`].
//! - **`DirectionCosinesToOrientation`'s tie-break is insertion order, not
//!   magnitude.** The greedy nearest-axis search
//!   (`itkDICOMOrientation.cxx:166-228`) inserts all nine `(|value|, row,
//!   col)` triples into a `std::multimap` in row-major insertion order and
//!   repeatedly takes `rbegin()`; `std::multimap` orders equal keys by
//!   insertion order (guaranteed since C++11), so a tie among the largest
//!   remaining magnitudes resolves to the *last-inserted* (highest row, then
//!   highest column) entry. This port builds the same `(row, col)` list in
//!   the same order and picks the maximum with `Iterator::max_by`, which the
//!   standard library documents as returning the last of equal elements — the
//!   same tie-break, reproduced rather than coincidental.
//!
//! ## Vector images
//!
//! Permute and flip only reorder pixels and rewrite geometry; they never
//! touch a pixel's own component values. For a vector image this port
//! extracts each component as a scalar image
//! ([`crate::core::Image::extract_component`]), reorients each one through the
//! same scalar path ([`crate::filters::geometry::permute_axes`] /
//! [`crate::filters::geometry::flip`]), and recomposes
//! ([`crate::core::Image::from_component_images`]) — every component receives
//! an identical geometry transform, so the recomposed image's geometry is
//! exactly that of any one component.

use crate::core::Image;

use crate::filters::error::{FilterError, Result};
use crate::filters::geometry::{flip, permute_axes};

const DIMENSION: usize = 3;

const IDENTITY_DIRECTION: [f64; DIMENSION * DIMENSION] =
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// `DICOMOrientImageFilter.yaml`'s `DesiredCoordinateOrientation` default.
pub const DEFAULT_ORIENTATION: &str = "LPS";

/// `DICOMOrientation::CoordinateEnum` (`itkDICOMOrientation.h:47-56`), minus
/// `UNKNOWN` — an unrecognized character is rejected before a `Coordinate` is
/// ever constructed (see the module doc's parsing note).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coordinate {
    Right,
    Left,
    Anterior,
    Posterior,
    Inferior,
    Superior,
}

impl Coordinate {
    fn from_char(c: char) -> Option<Self> {
        match c {
            'R' => Some(Coordinate::Right),
            'L' => Some(Coordinate::Left),
            'A' => Some(Coordinate::Anterior),
            'P' => Some(Coordinate::Posterior),
            'I' => Some(Coordinate::Inferior),
            'S' => Some(Coordinate::Superior),
            _ => None,
        }
    }

    fn as_char(self) -> char {
        match self {
            Coordinate::Right => 'R',
            Coordinate::Left => 'L',
            Coordinate::Anterior => 'A',
            Coordinate::Posterior => 'P',
            Coordinate::Inferior => 'I',
            Coordinate::Superior => 'S',
        }
    }

    /// The direction-matrix row this term occupies: 0 for the Left/Right
    /// pair, 1 for Anterior/Posterior, 2 for Inferior/Superior
    /// (`itkDICOMOrientation.cxx`'s `case max_c`/row-index switches).
    fn axis_family(self) -> u8 {
        match self {
            Coordinate::Right | Coordinate::Left => 0,
            Coordinate::Anterior | Coordinate::Posterior => 1,
            Coordinate::Inferior | Coordinate::Superior => 2,
        }
    }

    /// `CodeAxisDir` (`0x1`, `itkDICOMOrientImageFilter.hxx:66`): `true` for
    /// the terms that are the "positive" direction of an identity (LPS)
    /// matrix.
    fn is_positive(self) -> bool {
        matches!(
            self,
            Coordinate::Left | Coordinate::Posterior | Coordinate::Superior
        )
    }
}

fn same_orientation_axes(a: Coordinate, b: Coordinate) -> bool {
    a.axis_family() == b.axis_family()
}

/// `DICOMOrientation::OrientationEnum`, represented as its three axis terms
/// (fastest-moving axis first) or `None` for `INVALID`. See the module doc
/// for why this replaces upstream's packed code and string table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Orientation(Option<[Coordinate; DIMENSION]>);

impl Orientation {
    const INVALID: Orientation = Orientation(None);

    /// `DICOMOrientation(CoordinateEnum, CoordinateEnum, CoordinateEnum)`
    /// (`itkDICOMOrientation.cxx:28-39`): `INVALID` if any two terms share an
    /// axis family.
    fn from_terms(primary: Coordinate, secondary: Coordinate, tertiary: Coordinate) -> Self {
        if same_orientation_axes(primary, secondary)
            || same_orientation_axes(primary, tertiary)
            || same_orientation_axes(secondary, tertiary)
        {
            Orientation::INVALID
        } else {
            Orientation(Some([primary, secondary, tertiary]))
        }
    }

    /// `DICOMOrientation(std::string)` (`itkDICOMOrientation.cxx:41-51`):
    /// case-insensitive; anything that isn't exactly 3 characters from
    /// `RLAPIS`, with no axis-family collision, resolves to `INVALID`.
    fn from_str_ci(s: &str) -> Self {
        let chars: Vec<char> = s.chars().map(|c| c.to_ascii_uppercase()).collect();
        if chars.len() != DIMENSION {
            return Orientation::INVALID;
        }
        let terms: Option<Vec<Coordinate>> =
            chars.iter().map(|&c| Coordinate::from_char(c)).collect();
        match terms {
            Some(t) => Orientation::from_terms(t[0], t[1], t[2]),
            None => Orientation::INVALID,
        }
    }

    /// `DICOMOrientation::GetAsString` (`itkDICOMOrientation.cxx:54-65`).
    fn as_string(self) -> String {
        match self.0 {
            None => "INVALID".to_string(),
            Some(terms) => terms.iter().map(|t| t.as_char()).collect(),
        }
    }

    /// The three axis terms. Panics on `INVALID` — every caller validates
    /// non-`INVALID`-ness first (`dicom_orient` checks `desired` up front,
    /// and [`direction_cosines_to_orientation`] never returns `INVALID`).
    fn terms(self) -> [Coordinate; DIMENSION] {
        self.0
            .expect("Orientation::terms called on INVALID; callers must validate first")
    }
}

/// `DICOMOrientation::DirectionCosinesToOrientation`
/// (`itkDICOMOrientation.cxx:166-228`): the closest orientation for a 3x3
/// row-major direction cosine matrix, found by greedily assigning the
/// largest remaining `|value|` to its `(row, col)` slot and removing every
/// other entry in that row or column, three times. Never returns `INVALID`:
/// each of the 3 iterations claims a distinct row (physical axis family) and
/// a distinct column (image axis), so the resulting terms can never collide.
///
/// `dir` must have exactly 9 elements; callers ([`dicom_orient`], on an
/// already dimension-checked image, and
/// [`get_orientation_from_direction_cosines`] after its own length check)
/// guarantee this.
fn direction_cosines_to_orientation(dir: &[f64]) -> Orientation {
    debug_assert_eq!(dir.len(), DIMENSION * DIMENSION);

    let mut entries: Vec<(usize, usize)> = (0..DIMENSION)
        .flat_map(|row| (0..DIMENSION).map(move |col| (row, col)))
        .collect();

    let mut terms = [Coordinate::Right; DIMENSION];
    for _ in 0..DIMENSION {
        // `Iterator::max_by` returns the *last* of equal-maximum elements,
        // matching `std::multimap`'s insertion-ordered ties (see the module
        // doc's upstream-findings note).
        let &(row, col) = entries
            .iter()
            .max_by(|a, b| {
                let av = dir[a.0 * DIMENSION + a.1].abs();
                let bv = dir[b.0 * DIMENSION + b.1].abs();
                av.partial_cmp(&bv)
                    .expect("direction cosines must not be NaN")
            })
            .expect("each of the 3 iterations leaves at least one candidate (row, col)");

        let value = dir[row * DIMENSION + col];
        let term = match row {
            0 if value > 0.0 => Coordinate::Left,
            0 => Coordinate::Right,
            1 if value > 0.0 => Coordinate::Posterior,
            1 => Coordinate::Anterior,
            2 if value > 0.0 => Coordinate::Superior,
            2 => Coordinate::Inferior,
            _ => unreachable!("row is always 0..DIMENSION"),
        };
        terms[col] = term;
        entries.retain(|&(r, c)| r != row && c != col);
    }

    Orientation::from_terms(terms[0], terms[1], terms[2])
}

/// `DICOMOrientation::OrientationToDirectionCosines`
/// (`itkDICOMOrientation.cxx:231-263`): the row-major direction cosine matrix
/// for an orientation. `INVALID` maps every term to `UNKNOWN`, which the
/// upstream `switch` does not assign at all, leaving an all-zero matrix.
fn orientation_to_direction_cosines(o: Orientation) -> [f64; DIMENSION * DIMENSION] {
    let mut direction = [0.0f64; DIMENSION * DIMENSION];
    let Some(terms) = o.0 else {
        return direction;
    };
    for (i, term) in terms.into_iter().enumerate() {
        let sign = if term.is_positive() { 1.0 } else { -1.0 };
        let row = term.axis_family() as usize;
        direction[row * DIMENSION + i] = sign;
    }
    direction
}

/// `DICOMOrientImageFilter::DeterminePermutationsAndFlips`
/// (`itkDICOMOrientImageFilter.hxx:57-132`): the permute order and flip axes
/// that carry `given` into `desired`. Both orientations must already be
/// non-`INVALID`.
fn determine_permutations_and_flips(
    desired: Orientation,
    given: Orientation,
) -> ([usize; DIMENSION], [bool; DIMENSION]) {
    let desired_codes = desired.terms();
    let given_codes = given.terms();

    let mut permute_order = [0usize, 1, 2];

    for i in 0..DIMENSION - 1 {
        if same_orientation_axes(given_codes[i], desired_codes[i]) {
            continue;
        }
        for j in 0..DIMENSION {
            if !same_orientation_axes(given_codes[i], desired_codes[j]) {
                continue;
            }
            if i == j {
                continue;
            }
            if same_orientation_axes(given_codes[j], desired_codes[i]) {
                // Cyclic (i j); the remaining axis is stationary.
                permute_order[i] = j;
                permute_order[j] = i;
            } else {
                for k in 0..DIMENSION {
                    if same_orientation_axes(given_codes[j], desired_codes[k]) {
                        permute_order[i] = k;
                        permute_order[j] = i;
                        permute_order[k] = j;
                        break;
                    }
                }
            }
            break;
        }
    }

    let mut flip_axes = [false; DIMENSION];
    for (i, &j) in permute_order.iter().enumerate() {
        if given_codes[j].is_positive() != desired_codes[i].is_positive() {
            flip_axes[i] = true;
        }
    }

    (permute_order, flip_axes)
}

/// `PermuteAxesImageFilter` then `FlipImageFilter`, only running each when it
/// would change anything (`DICOMOrientImageFilter::NeedToPermute` /
/// `NeedToFlip`, `itkDICOMOrientImageFilter.hxx:169-195`), with
/// `FlipAboutOriginOff()` (`itkDICOMOrientImageFilter.hxx:241`).
fn reorient_scalar(
    img: &Image,
    permute_order: &[usize; DIMENSION],
    flip_axes: &[bool; DIMENSION],
) -> Result<Image> {
    let mut out = img.clone();
    if permute_order.iter().enumerate().any(|(j, &o)| o != j) {
        out = permute_axes(&out, permute_order)?;
    }
    if flip_axes.iter().any(|&f| f) {
        out = flip(&out, flip_axes, false)?;
    }
    Ok(out)
}

/// The vector-image path: see the module doc's "Vector images" section.
fn reorient_vector(
    img: &Image,
    permute_order: &[usize; DIMENSION],
    flip_axes: &[bool; DIMENSION],
) -> Result<Image> {
    let n = img.number_of_components_per_pixel();
    let mut components = Vec::with_capacity(n);
    for c in 0..n {
        let component = img.extract_component(c)?;
        components.push(reorient_scalar(&component, permute_order, flip_axes)?);
    }
    let refs: Vec<&Image> = components.iter().collect();
    Ok(Image::from_component_images(&refs)?)
}

/// Measurements `DICOMOrientImageFilter.yaml` exposes (`FlipAxes`,
/// `PermuteOrder`), computed during execution, alongside the reoriented
/// image. `InputCoordinateOrientation` is not included: the yaml exposes it
/// nowhere (only `DesiredCoordinateOrientation` is a member, and only
/// `FlipAxes`/`PermuteOrder` are measurements).
pub struct DicomOrientResult {
    pub image: Image,
    /// `GetFlipAxes()`: one flag per *output* axis (applied after
    /// permuting).
    pub flip_axes: Vec<bool>,
    /// `GetPermuteOrder()`: output axis `i` is sourced from input axis
    /// `permute_order[i]`.
    pub permute_order: Vec<u32>,
}

/// `DICOMOrientImageFilter`: permute and flip `img`'s axes so its direction
/// cosines read as `desired_coordinate_orientation` (case-insensitive; e.g.
/// `"LPS"`, `"RAS"`), recomputing origin and direction to match. 3-D images
/// only ([`FilterError::UnsupportedDicomOrientDimension`]); scalar and vector
/// pixel types alike (see the module doc's "Vector images" section).
///
/// `desired_coordinate_orientation` must parse to one of the 48 valid
/// 3-letter orientation codes, or this returns
/// [`FilterError::InvalidDesiredOrientation`] (see the module doc's
/// upstream-findings note on the two-stage upstream failure this collapses
/// into one).
pub fn dicom_orient(
    img: &Image,
    desired_coordinate_orientation: &str,
) -> Result<DicomOrientResult> {
    if img.dimension() != DIMENSION {
        return Err(FilterError::UnsupportedDicomOrientDimension(
            img.dimension(),
        ));
    }

    let desired = Orientation::from_str_ci(desired_coordinate_orientation);
    if desired == Orientation::INVALID {
        return Err(FilterError::InvalidDesiredOrientation(
            desired_coordinate_orientation.to_string(),
        ));
    }

    let given = direction_cosines_to_orientation(img.direction());
    let (permute_order, flip_axes) = determine_permutations_and_flips(desired, given);

    let out_image = if img.pixel_id().is_vector() {
        reorient_vector(img, &permute_order, &flip_axes)?
    } else {
        reorient_scalar(img, &permute_order, &flip_axes)?
    };

    Ok(DicomOrientResult {
        image: out_image,
        flip_axes: flip_axes.to_vec(),
        permute_order: permute_order.iter().map(|&o| o as u32).collect(),
    })
}

/// `DICOMOrientImageFilter::GetOrientationFromDirectionCosines`
/// (`sitkDICOMOrientImageFilter_Support.cxx:30-38`): the closest 3-letter
/// orientation code for a row-major 3x3 direction cosine matrix.
///
/// `direction` may be empty (treated as identity, i.e. `"LPS"` —
/// `sitkSTLToITKDirection`'s special case) or exactly 9 elements; any other
/// length is [`FilterError::InvalidDirectionCosinesLength`].
pub fn get_orientation_from_direction_cosines(direction: &[f64]) -> Result<String> {
    if direction.is_empty() {
        return Ok(direction_cosines_to_orientation(&IDENTITY_DIRECTION).as_string());
    }
    if direction.len() != DIMENSION * DIMENSION {
        return Err(FilterError::InvalidDirectionCosinesLength(direction.len()));
    }
    Ok(direction_cosines_to_orientation(direction).as_string())
}

/// `DICOMOrientImageFilter::GetDirectionCosinesFromOrientation`
/// (`sitkDICOMOrientImageFilter_Support.cxx:40-48`): the row-major 3x3
/// direction cosine matrix for a 3-letter orientation code.
///
/// Unlike [`dicom_orient`], this never errors on an invalid `orientation`:
/// upstream's static utility has no validation at all (see the module doc's
/// upstream-findings note), so an unparseable string silently yields an
/// all-zero matrix.
pub fn get_direction_cosines_from_orientation(orientation: &str) -> Vec<f64> {
    orientation_to_direction_cosines(Orientation::from_str_ci(orientation)).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PixelId;

    /// The literal 48-string list from `itkDICOMOrientation.cxx:112-160`
    /// (`CreateCodeToString`), so the term-based model is checked against
    /// upstream's own table rather than a self-generated one.
    const ALL_48_CODES: [&str; 48] = [
        "RIP", "LIP", "RSP", "LSP", "RIA", "LIA", "RSA", "LSA", "IRP", "ILP", "SRP", "SLP", "IRA",
        "ILA", "SRA", "SLA", "RPI", "LPI", "RAI", "LAI", "RPS", "LPS", "RAS", "LAS", "PRI", "PLI",
        "ARI", "ALI", "PRS", "PLS", "ARS", "ALS", "IPR", "SPR", "IAR", "SAR", "IPL", "SPL", "IAL",
        "SAL", "PIR", "PSR", "AIR", "ASR", "PIL", "PSL", "AIL", "ASL",
    ];

    fn img_f64(size: &[usize], data: Vec<f64>) -> Image {
        Image::from_vec(size, data).unwrap()
    }

    // ---- Orientation string parsing -----------------------------------

    #[test]
    fn orientation_string_round_trips_every_upstream_code() {
        for &code in &ALL_48_CODES {
            let o = Orientation::from_str_ci(code);
            assert_ne!(o, Orientation::INVALID, "{code} parsed as INVALID");
            assert_eq!(o.as_string(), code, "{code} did not round-trip");
            // Case-insensitive, matching `std::transform(..., ::toupper)`.
            assert_eq!(Orientation::from_str_ci(&code.to_lowercase()), o);
        }
    }

    #[test]
    fn invalid_orientation_strings_are_rejected() {
        // Wrong length.
        assert_eq!(Orientation::from_str_ci("LP"), Orientation::INVALID);
        assert_eq!(Orientation::from_str_ci("LPSA"), Orientation::INVALID);
        assert_eq!(Orientation::from_str_ci(""), Orientation::INVALID);
        // Unrecognized character.
        assert_eq!(Orientation::from_str_ci("LPX"), Orientation::INVALID);
        // Axis-family collision: L and R are the same family.
        assert_eq!(Orientation::from_str_ci("LRS"), Orientation::INVALID);
        // The literal "INVALID" string itself (7 chars) is also invalid.
        assert_eq!(Orientation::from_str_ci("INVALID"), Orientation::INVALID);
    }

    // ---- direction cosines <-> orientation -----------------------------

    #[test]
    fn identity_direction_is_lps() {
        assert_eq!(
            direction_cosines_to_orientation(&IDENTITY_DIRECTION).as_string(),
            "LPS"
        );
    }

    #[test]
    fn ras_direction_round_trips() {
        // RAS: Right/Anterior/Superior negate the LR and AP axes relative to
        // the LPS identity.
        let ras = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(direction_cosines_to_orientation(&ras).as_string(), "RAS");
        assert_eq!(
            orientation_to_direction_cosines(Orientation::from_str_ci("RAS")),
            ras
        );
    }

    #[test]
    fn invalid_orientation_direction_cosines_are_all_zero() {
        // `OrientationToDirectionCosines(INVALID)`: every term is UNKNOWN,
        // and the switch assigns nothing.
        assert_eq!(
            orientation_to_direction_cosines(Orientation::INVALID),
            [0.0; 9]
        );
    }

    // ---- get_orientation_from_direction_cosines / get_direction_cosines_from_orientation ----

    #[test]
    fn public_statics_round_trip() {
        assert_eq!(
            get_orientation_from_direction_cosines(&IDENTITY_DIRECTION).unwrap(),
            "LPS"
        );
        assert_eq!(
            get_direction_cosines_from_orientation("LPS"),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            get_direction_cosines_from_orientation("RAS"),
            &[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn empty_direction_vector_defaults_to_identity() {
        // `sitkSTLToITKDirection`'s special case: empty -> SetIdentity().
        assert_eq!(get_orientation_from_direction_cosines(&[]).unwrap(), "LPS");
    }

    #[test]
    fn wrong_length_direction_vector_errors() {
        assert_eq!(
            get_orientation_from_direction_cosines(&[1.0, 2.0, 3.0, 4.0]),
            Err(FilterError::InvalidDirectionCosinesLength(4))
        );
    }

    #[test]
    fn invalid_orientation_string_yields_all_zero_direction_with_no_error() {
        // `GetDirectionCosinesFromOrientation` has no validation at all.
        assert_eq!(
            get_direction_cosines_from_orientation("not-an-orientation"),
            vec![0.0; 9]
        );
    }

    // ---- dicom_orient: permute/flip determination, pinned against the
    // SimpleITK yaml's own RAS/RIP test fixtures (`DICOMOrientImageFilter.yaml`,
    // tags "RAS" and "RIP"), whose "default" tag proves the fixture input
    // image (RA-Short.nrrd) already has an LPS (identity) direction: with the
    // default DesiredCoordinateOrientation ("LPS") the yaml's own expected
    // FlipAxes/PermuteOrder are [0,0,0]/[0,1,2] -- i.e. given == desired ==
    // LPS. An identity-direction synthetic image is therefore the same
    // "given" orientation as that fixture.

    #[test]
    fn lps_to_ras_flips_without_permuting() {
        // yaml tag "RAS": FlipAxes [1,1,0], PermuteOrder [0,1,2].
        let img = img_f64(&[2, 3, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = dicom_orient(&img, "RAS").unwrap();
        assert_eq!(out.permute_order, vec![0, 1, 2]);
        assert_eq!(out.flip_axes, vec![true, true, false]);
        assert_eq!(out.image.size(), &[2, 3, 1]);
        assert_eq!(
            out.image.direction(),
            &[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        );
        let expected_origin = img.continuous_index_to_physical_point(&[1.0, 2.0, 0.0]);
        assert_eq!(out.image.origin(), expected_origin.as_slice());
        // Reversing both in-plane axes of a row-major (x-fastest) sequence.
        assert_eq!(
            out.image.to_f64_vec().unwrap(),
            vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        );
    }

    #[test]
    fn lps_to_rip_permutes_and_flips() {
        // yaml tag "RIP": FlipAxes [1,1,0], PermuteOrder [0,2,1].
        let size = [2usize, 3, 4];
        let mut data = vec![0.0f64; 24];
        for z in 0..4 {
            for y in 0..3 {
                for x in 0..2 {
                    data[z * 6 + y * 2 + x] = (x + 10 * y + 100 * z) as f64;
                }
            }
        }
        let img = img_f64(&size, data);
        let out = dicom_orient(&img, "RIP").unwrap();
        assert_eq!(out.permute_order, vec![0, 2, 1]);
        assert_eq!(out.flip_axes, vec![true, true, false]);
        assert_eq!(out.image.size(), &[2, 4, 3]);

        // Hand-derived closed form for permute_order=[0,2,1] then
        // flip_axes=[true,true,false] on value(x,y,z) = x + 10y + 100z (see
        // the module's `determine_permutations_and_flips`/`reorient_scalar`
        // composition): final(a,b,c) = (1-a) + 10*c + 100*(3-b).
        let out_size = out.image.size().to_vec();
        let mut expected = vec![0.0f64; 24];
        for c in 0..out_size[2] {
            for b in 0..out_size[1] {
                for a in 0..out_size[0] {
                    let idx = c * out_size[0] * out_size[1] + b * out_size[0] + a;
                    expected[idx] =
                        (1 - a as i64) as f64 + 10.0 * c as f64 + 100.0 * (3 - b as i64) as f64;
                }
            }
        }
        assert_eq!(out.image.to_f64_vec().unwrap(), expected);
    }

    #[test]
    fn ras_moves_u64_pixels_losslessly() {
        // `dicom_orient` reindexes through `permute_axes`/`flip`, both now
        // native; a `UInt64` value above 2^53 whose bits cannot survive an
        // f64 round-trip proves the pixel path never widened. RAS reverses the
        // two in-plane axes of a 2x3x1 row-major sequence.
        const HI: u64 = (1 << 53) + 1;
        // Non-vacuity guard: this value must genuinely differ from its f64
        // round-trip, else recovering it exactly would prove nothing.
        assert_ne!(HI, (HI as f64) as u64);
        let seq: Vec<u64> = (0..6).map(|k| HI + k).collect();
        let img = Image::from_vec(&[2, 3, 1], seq).unwrap();
        let out = dicom_orient(&img, "RAS").unwrap();
        assert_eq!(out.flip_axes, vec![true, true, false]);
        assert_eq!(
            out.image.scalar_slice::<u64>().unwrap(),
            &[HI + 5, HI + 4, HI + 3, HI + 2, HI + 1, HI]
        );
    }

    #[test]
    fn given_equals_desired_is_a_no_op() {
        // yaml "default" tag: FlipAxes [0,0,0], PermuteOrder [0,1,2].
        let img = img_f64(&[2, 2, 2], (0..8).map(|v| v as f64).collect());
        let out = dicom_orient(&img, DEFAULT_ORIENTATION).unwrap();
        assert_eq!(out.permute_order, vec![0, 1, 2]);
        assert_eq!(out.flip_axes, vec![false, false, false]);
        assert_eq!(out.image.to_f64_vec().unwrap(), img.to_f64_vec().unwrap());
        assert_eq!(out.image.origin(), img.origin());
        assert_eq!(out.image.direction(), img.direction());
    }

    #[test]
    fn invalid_desired_orientation_errors() {
        let img = img_f64(&[2, 2, 2], vec![0.0; 8]);
        assert_eq!(
            dicom_orient(&img, "XYZ").err(),
            Some(FilterError::InvalidDesiredOrientation("XYZ".to_string()))
        );
    }

    #[test]
    fn non_3d_image_is_rejected() {
        let img = img_f64(&[2, 2], vec![0.0; 4]);
        assert_eq!(
            dicom_orient(&img, "LPS").err(),
            Some(FilterError::UnsupportedDicomOrientDimension(2))
        );
    }

    // ---- vector images --------------------------------------------------

    #[test]
    fn vector_image_reorients_every_component_identically() {
        let size = [2usize, 3, 1];
        // Two components: component 0 mirrors the scalar RAS test's data,
        // component 1 is offset by 1000 per pixel, interleaved per pixel.
        let mut interleaved = Vec::with_capacity(12);
        for base in [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0] {
            interleaved.push(base);
            interleaved.push(base + 1000.0);
        }
        let img = Image::from_vec_vector(&size, 2, interleaved).unwrap();

        let out = dicom_orient(&img, "RAS").unwrap();
        assert_eq!(out.permute_order, vec![0, 1, 2]);
        assert_eq!(out.flip_axes, vec![true, true, false]);
        assert_eq!(out.image.pixel_id(), PixelId::VectorFloat64);
        assert_eq!(out.image.number_of_components_per_pixel(), 2);
        assert_eq!(
            out.image.direction(),
            &[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        );

        // Same pixel-order reversal as the scalar RAS test, but every
        // component moves together.
        let component_0 = out.image.extract_component(0).unwrap();
        let component_1 = out.image.extract_component(1).unwrap();
        assert_eq!(
            component_0.to_f64_vec().unwrap(),
            vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        );
        assert_eq!(
            component_1.to_f64_vec().unwrap(),
            vec![1005.0, 1004.0, 1003.0, 1002.0, 1001.0, 1000.0]
        );
    }

    #[test]
    fn vector_image_permutation_matches_scalar_per_component() {
        let size = [2usize, 3, 4];
        let n = 2usize;
        let mut interleaved = vec![0.0f64; 24 * n];
        for z in 0..4 {
            for y in 0..3 {
                for x in 0..2 {
                    let pixel = z * 6 + y * 2 + x;
                    let base = (x + 10 * y + 100 * z) as f64;
                    interleaved[pixel * n] = base;
                    interleaved[pixel * n + 1] = base + 10000.0;
                }
            }
        }
        let vector_img = Image::from_vec_vector(&size, n, interleaved).unwrap();

        let mut scalar_data = vec![0.0f64; 24];
        for z in 0..4 {
            for y in 0..3 {
                for x in 0..2 {
                    scalar_data[z * 6 + y * 2 + x] = (x + 10 * y + 100 * z) as f64;
                }
            }
        }
        let scalar_img = img_f64(&size, scalar_data);

        let vector_out = dicom_orient(&vector_img, "RIP").unwrap();
        let scalar_out = dicom_orient(&scalar_img, "RIP").unwrap();

        assert_eq!(vector_out.permute_order, scalar_out.permute_order);
        assert_eq!(vector_out.flip_axes, scalar_out.flip_axes);
        assert_eq!(vector_out.image.size(), scalar_out.image.size());
        assert_eq!(vector_out.image.origin(), scalar_out.image.origin());
        assert_eq!(vector_out.image.direction(), scalar_out.image.direction());
        assert_eq!(
            vector_out
                .image
                .extract_component(0)
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            scalar_out.image.to_f64_vec().unwrap()
        );
    }
}
