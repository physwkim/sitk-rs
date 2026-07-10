//! [`ImageSeriesReader`] — SimpleITK's `itk::simple::ImageSeriesReader`
//! (sitkImageSeriesReader.h:69-283, sitkImageSeriesReader.cxx), wrapping
//! `itk::ImageSeriesReader<TOutputImage>` (itkImageSeriesReader.hxx).
//!
//! DICOM series file-name enumeration (`GetGDCMSeriesFileNames`,
//! `GetGDCMSeriesIDs`) is out of scope; callers supply [`file_names`]
//! directly, as every other caller of this reader must.
//!
//! [`file_names`]: ImageSeriesReader::set_file_names

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sitk_core::{Complex, Image, PixelBuffer, PixelId, matrix};

use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, reader_for};
use crate::reader::normalize_reader_geometry;

/// `SimpleITK_MAX_DIMENSION` (CMake `SimpleITK_MAX_DIMENSION_DEFAULT`), the
/// ceiling `ImageSeriesReader::Execute` enforces on the promoted dimension
/// (sitkImageSeriesReader.cxx:231).
const SITK_MAX_DIMENSION: usize = 5;

/// Read a stack of `N`-dimensional slice files into one `N+1`-dimensional
/// image, or (when the promotion collapses, see [`ImageSeriesReader::execute`])
/// an `N`-dimensional one.
///
/// ```no_run
/// # use sitk_io::ImageSeriesReader;
/// let mut reader = ImageSeriesReader::new();
/// reader.set_file_names(&["slice0.png", "slice1.png", "slice2.png"]);
/// let volume = reader.execute()?;
/// assert_eq!(volume.dimension(), 3);
/// # Ok::<(), sitk_io::IoError>(())
/// ```
///
/// Every slice is read through the *same* [`ImageIo`](crate::ImageIo) —
/// whichever one [`file_names`](ImageSeriesReader::set_file_names)`[0]`
/// resolves to — even the "first" and "last" files used for geometry, which
/// are a different file from `[0]` under [`ImageSeriesReader::reverse_order`].
/// This reproduces `itk::ImageSeriesReader<TOutputImage>` reusing the one
/// `itk::ImageIOBase` its owning `itk::simple::ImageSeriesReader` resolved
/// from `m_FileNames.front()` (sitkImageSeriesReader.cxx:208,260;
/// itkImageSeriesReader.hxx:110-114): a series whose files are not all the
/// same format fails or misreads through the wrong parser, exactly as
/// upstream would.
///
/// `SetOutputPixelType` and `SetImageIO` are not exposed, matching
/// [`crate::ImageFileReader`]'s own scope (ledger §4): this crate has no
/// pixel-casting layer, and every slice must already share the pixel type
/// [`ImageSeriesReader::execute`] deduces from the first file.
#[derive(Clone, Debug)]
pub struct ImageSeriesReader {
    file_names: Vec<PathBuf>,
    spacing_warning_rel_threshold: f64,
    force_orthogonal_direction: bool,
    reverse_order: bool,
    meta_data_dictionary_array_update: bool,
    slice_metadata: Vec<BTreeMap<String, String>>,
}

impl Default for ImageSeriesReader {
    fn default() -> Self {
        Self {
            file_names: Vec::new(),
            spacing_warning_rel_threshold: 1e-4,
            force_orthogonal_direction: true,
            reverse_order: false,
            meta_data_dictionary_array_update: false,
            slice_metadata: Vec::new(),
        }
    }
}

impl ImageSeriesReader {
    /// A reader with no file names.
    pub fn new() -> Self {
        Self::default()
    }

    /// `SetFileNames`.
    pub fn set_file_names<P: AsRef<Path>>(&mut self, names: &[P]) -> &mut Self {
        self.file_names = names.iter().map(|p| p.as_ref().to_path_buf()).collect();
        self
    }

    /// `GetFileNames`.
    pub fn file_names(&self) -> &[PathBuf] {
        &self.file_names
    }

    /// `SetSpacingWarningRelThreshold` (sitkImageSeriesReader.h:172-176,
    /// default `1e-4`). Stored for parity but otherwise inert: it only ever
    /// gates `itkWarningMacro`'s non-uniform-sampling warning
    /// (itkImageSeriesReader.hxx:486-490), and this crate has no logging
    /// infrastructure to emit it, matching the established convention
    /// elsewhere in this crate (e.g. [`crate::dicom_orient`](crate) and
    /// `threshold.rs`'s dropped warnings).
    pub fn set_spacing_warning_rel_threshold(&mut self, threshold: f64) -> &mut Self {
        self.spacing_warning_rel_threshold = threshold;
        self
    }

    /// `GetSpacingWarningRelThreshold`.
    pub fn spacing_warning_rel_threshold(&self) -> f64 {
        self.spacing_warning_rel_threshold
    }

    /// `SetForceOrthogonalDirection` (sitkImageSeriesReader.h:178-190, default
    /// `true`). See [`ImageSeriesReader::execute`] for exactly what this
    /// changes.
    pub fn set_force_orthogonal_direction(&mut self, force: bool) -> &mut Self {
        self.force_orthogonal_direction = force;
        self
    }

    /// `GetForceOrthogonalDirection`.
    pub fn force_orthogonal_direction(&self) -> bool {
        self.force_orthogonal_direction
    }

    /// `ForceOrthogonalDirectionOn`.
    pub fn force_orthogonal_direction_on(&mut self) -> &mut Self {
        self.set_force_orthogonal_direction(true)
    }

    /// `ForceOrthogonalDirectionOff`.
    pub fn force_orthogonal_direction_off(&mut self) -> &mut Self {
        self.set_force_orthogonal_direction(false)
    }

    /// `SetReverseOrder` (sitkImageSeriesReader.h:192-203, default `false`).
    /// Reverses which *file* fills which series slot; slot `0` of the output
    /// is still slot `0`; it is file `N-1` that lands there instead of file
    /// `0`.
    pub fn set_reverse_order(&mut self, reverse: bool) -> &mut Self {
        self.reverse_order = reverse;
        self
    }

    /// `GetReverseOrder`.
    pub fn reverse_order(&self) -> bool {
        self.reverse_order
    }

    /// `ReverseOrderOn`.
    pub fn reverse_order_on(&mut self) -> &mut Self {
        self.set_reverse_order(true)
    }

    /// `ReverseOrderOff`.
    pub fn reverse_order_off(&mut self) -> &mut Self {
        self.set_reverse_order(false)
    }

    /// `SetMetaDataDictionaryArrayUpdate` (sitkImageSeriesReader.h:90-104,
    /// default `false`).
    ///
    /// When `false`, [`ImageSeriesReader::execute`] can still turn collection
    /// on partway through the series: `itk::ImageSeriesReader::GenerateData`
    /// flips its own `needToUpdateMetaDataDictionaryArray` the moment it
    /// detects non-uniform slice spacing, *unconditionally*, regardless of
    /// this setting (itkImageSeriesReader.hxx:439-452) — see
    /// [`ImageSeriesReader::execute`]'s docs for what that does to the slice
    /// indices [`ImageSeriesReader::meta_data_keys`] and friends accept.
    pub fn set_meta_data_dictionary_array_update(&mut self, update: bool) -> &mut Self {
        self.meta_data_dictionary_array_update = update;
        self
    }

    /// `GetMetaDataDictionaryArrayUpdate`.
    pub fn meta_data_dictionary_array_update(&self) -> bool {
        self.meta_data_dictionary_array_update
    }

    /// `MetaDataDictionaryArrayUpdateOn`.
    pub fn meta_data_dictionary_array_update_on(&mut self) -> &mut Self {
        self.set_meta_data_dictionary_array_update(true)
    }

    /// `MetaDataDictionaryArrayUpdateOff`.
    pub fn meta_data_dictionary_array_update_off(&mut self) -> &mut Self {
        self.set_meta_data_dictionary_array_update(false)
    }

    /// Read the series.
    ///
    /// `Execute` (sitkImageSeriesReader.cxx:195-244) plus
    /// `itk::ImageSeriesReader<TOutputImage>`'s `GenerateOutputInformation`
    /// and `GenerateData` (itkImageSeriesReader.hxx:81-237, 257-504).
    ///
    /// # Dimension promotion
    ///
    /// The output dimension is the *first* file's own dimension plus one,
    /// except that a promotion to `4` collapses back to `3` when that first
    /// file's own third axis has size `1` (sitkImageSeriesReader.cxx:219-229) —
    /// so three 100x100x1 files promote to a 3-D `100x100x3` volume, not a 4-D
    /// one. The promoted dimension must land in `2..=5`
    /// ([`SITK_MAX_DIMENSION`]).
    ///
    /// # Which axis the slices stack along
    ///
    /// Usually the newly-promoted axis, but not always:
    /// `ComputeMovingDimensionIndex` (itkImageSeriesReader.hxx:54-79) starts
    /// at `min(first file's own dimension, output dimension - 1)` and then
    /// walks *down* while the (padding-extended) size at the axis just below
    /// is `1` — so a single-row or single-column first file walks the moving
    /// axis down further still. A lone file (`file_names` of length 1) uses
    /// `min(first file's own dimension, output dimension)` instead, with no
    /// walk-down at all — that series has no moving axis if the single file
    /// already fills every output axis.
    ///
    /// # Geometry
    ///
    /// With more than one file, inter-slice spacing and the direction column
    /// at the moving axis come from the vector between the *first* and
    /// *last* files' origins (in series order, honouring
    /// [`ImageSeriesReader::reverse_order`]) — not from adjacent files. Both
    /// origins can be overridden per file by an `"ITK_ImageOrigin"`
    /// meta-data value, itself only honoured when it tokenizes into exactly
    /// as many `f64`s as the output has dimensions (a malformed override is
    /// silently ignored rather than reproducing the out-of-bounds read a
    /// wrong-length `itk::Array` assignment would cause upstream — ledger
    /// §1). The image's own `origin`, however, is always the first file's
    /// *un*-overridden origin (itkImageSeriesReader.hxx:121,226) — the
    /// override only ever feeds the spacing/direction computation.
    ///
    /// If the first/last origins coincide, spacing at the moving axis is `1.0`
    /// and the direction column is left untouched.
    /// Otherwise, [`ImageSeriesReader::force_orthogonal_direction`] (the
    /// default) keeps the existing direction column but flips its sign to
    /// agree with the first-to-last direction; turning it off replaces the
    /// column with that direction, normalized.
    ///
    /// A single file's geometry is simply its own, unmodified.
    ///
    /// # Per-slice checks
    ///
    /// Every slice (including the first and last, re-read fresh here exactly
    /// as `itk::ImageSeriesReader::GenerateData` re-reads them,
    /// itkImageSeriesReader.hxx:308-328) must match the series' pixel type —
    /// [`IoError::SeriesPixelTypeMismatch`], a deliberate divergence this
    /// crate has no casting layer to avoid — and, once padded to the output
    /// dimension with the moving axis forced to `1`, the first file's own
    /// size — [`IoError::SeriesSizeMismatch`], upstream's own check
    /// (itkImageSeriesReader.hxx:354-361).
    ///
    /// # Meta-data dictionary collection and its index quirk
    ///
    /// With [`ImageSeriesReader::meta_data_dictionary_array_update`] off (the
    /// default), no per-slice dictionary is collected — *unless* upstream's
    /// non-uniform-sampling detector fires partway through the series
    /// (comparing each slice's own origin to the *previous* slice's, in
    /// series order, against the spacing derived above), which turns
    /// collection on unconditionally from that slice onward
    /// (itkImageSeriesReader.hxx:294-295,439-452). Collected dictionaries are
    /// pushed in the order they are collected, so
    /// [`ImageSeriesReader::meta_data_keys`]`(slice)`'s `slice` indexes into
    /// that (possibly shorter, possibly shifted) array — not necessarily the
    /// series position — exactly as `GetMetaDataKeysCustomCast::CustomCast`'s
    /// `mda.at(i)` does (sitkMetaDataDictionaryCustomCast.hxx:38-46).
    ///
    /// The image itself never carries any slice's dictionary. It gets exactly
    /// one meta-data value, `"ITK_non_uniform_sampling_deviation"`, set
    /// whenever any deviation was detected at all — regardless of
    /// [`ImageSeriesReader::spacing_warning_rel_threshold`], which upstream
    /// only gates the (here, dropped) warning, not this assignment
    /// (itkImageSeriesReader.hxx:486-496).
    pub fn execute(&mut self) -> Result<Image> {
        self.slice_metadata.clear();

        if self.file_names.is_empty() {
            return Err(IoError::EmptySeriesFileNames);
        }
        let n = self.file_names.len();

        let io = reader_for(&self.file_names[0])?;
        let dispatch_info = io.read_information(&self.file_names[0])?;

        let file_dim = dispatch_info.dimension;
        let mut out_dim = file_dim + 1;
        if out_dim == 4 && dispatch_info.size[2] == 1 {
            out_dim -= 1;
        }
        if !(2..=SITK_MAX_DIMENSION).contains(&out_dim) {
            return Err(IoError::UnsupportedSeriesDimension(out_dim - 1));
        }
        let pixel_id = dispatch_info.pixel_id;

        let first_idx = if self.reverse_order { n - 1 } else { 0 };
        let last_idx = if self.reverse_order { 0 } else { n - 1 };

        let mut first_info = io.read_information(&self.file_names[first_idx])?;
        check_pixel_type(&self.file_names[first_idx], &first_info, pixel_id)?;
        // Upstream derives the first/last geometry through the same
        // `ImageFileReader` that normalizes negative spacing to positive
        // (itkImageSeriesReader.hxx:120-122); this port reads raw geometry via
        // `read_information`, so it must normalize before deriving.
        normalize_info_geometry(&mut first_info);
        let (first_size, first_spacing, first_origin, first_direction) =
            padded_info_geometry(&first_info, out_dim);

        let moving_dim;
        let out_size;
        let out_spacing;
        let out_origin;
        let out_direction;
        let spacing_defined;

        if n == 1 {
            moving_dim = first_info.dimension.min(out_dim);
            out_size = first_size.clone();
            out_spacing = first_spacing;
            out_origin = first_origin;
            out_direction = first_direction;
            spacing_defined = false;
        } else {
            let mut moving = first_info.dimension.min(out_dim - 1);
            while moving > 0 && first_size[moving - 1] == 1 {
                moving -= 1;
            }
            moving_dim = moving;

            let mut last_info = io.read_information(&self.file_names[last_idx])?;
            check_pixel_type(&self.file_names[last_idx], &last_info, pixel_id)?;
            normalize_info_geometry(&mut last_info);
            let (_, _, last_padded_origin, _) = padded_info_geometry(&last_info, out_dim);

            let position1 = itk_image_origin_override(&first_info.metadata, out_dim)
                .unwrap_or_else(|| first_origin.clone());
            let position_n = itk_image_origin_override(&last_info.metadata, out_dim)
                .unwrap_or(last_padded_origin);

            let dir_n: Vec<f64> = (0..out_dim).map(|j| position_n[j] - position1[j]).collect();
            let dir_n_norm = norm(&dir_n);

            let mut spacing = first_spacing.clone();
            let mut direction = first_direction.clone();
            if almost_zero(dir_n_norm) {
                spacing[moving_dim] = 1.0;
                spacing_defined = false;
            } else {
                spacing[moving_dim] = dir_n_norm / (n as f64 - 1.0);
                spacing_defined = true;
                if self.force_orthogonal_direction {
                    let dot: f64 = (0..out_dim)
                        .map(|j| dir_n[j] * direction[j * out_dim + moving_dim])
                        .sum();
                    if dot < 0.0 {
                        for j in 0..out_dim {
                            direction[j * out_dim + moving_dim] *= -1.0;
                        }
                    }
                } else {
                    for j in 0..out_dim {
                        direction[j * out_dim + moving_dim] = dir_n[j] / dir_n_norm;
                    }
                }
            }

            let mut size = first_size.clone();
            if moving_dim != out_dim {
                size[moving_dim] = n;
            }
            out_size = size;
            out_spacing = spacing;
            out_origin = first_origin;
            out_direction = direction;
        }

        let components = first_info.number_of_components;
        let stride = if pixel_id.is_complex() {
            2
        } else if pixel_id.is_vector() {
            components
        } else {
            1
        };

        let mut expected_size = first_size.clone();
        if moving_dim != out_dim {
            expected_size[moving_dim] = 1;
        }

        let total_pixels: usize = out_size.iter().product();
        let mut buffer = PixelBuffer::zeroed(pixel_id, total_pixels * stride);

        let mut collect_metadata = self.meta_data_dictionary_array_update;
        let mut max_spacing_deviation = 0.0_f64;
        let mut prev_slice_origin: Option<Vec<f64>> = None;
        let mut collected = Vec::new();

        for i in 0..n {
            let file_idx = if self.reverse_order { n - 1 - i } else { i };
            let path = self.file_names[file_idx].clone();
            let mut slice_image = io.read(&path)?;
            // Every slice goes through the same `ImageFileReader` normalization
            // upstream applies per file (itkImageSeriesReader.hxx:106-109,327):
            // negative spacing flips positive and the raw geometry is recorded
            // under `ITK_original_*`, which the collected per-slice dictionaries
            // then copy (itkImageFileReader.hxx:219-221,
            // itkImageSeriesReader.hxx:468).
            normalize_reader_geometry(&mut slice_image)?;
            if slice_image.pixel_id() != pixel_id {
                return Err(IoError::SeriesPixelTypeMismatch {
                    file: path,
                    pixel_type: slice_image.pixel_id().as_str(),
                    expected: pixel_id.as_str(),
                });
            }

            let (slice_size, _, slice_origin, _) = padded_image_geometry(&slice_image, out_dim);
            if slice_size != expected_size {
                return Err(IoError::SeriesSizeMismatch {
                    file: path,
                    size: slice_size,
                    expected: expected_size.clone(),
                    reference_file: self.file_names[first_idx].clone(),
                });
            }

            let mut non_uniform = false;
            let mut deviation = 0.0;
            if let Some(prev) = &prev_slice_origin {
                let dir_n: Vec<f64> = (0..out_dim).map(|j| slice_origin[j] - prev[j]).collect();
                let dir_n_norm = norm(&dir_n);
                if spacing_defined && !almost_equal(dir_n_norm, out_spacing[moving_dim]) {
                    non_uniform = true;
                    deviation = (out_spacing[moving_dim] - dir_n_norm).abs();
                    if deviation > max_spacing_deviation {
                        max_spacing_deviation = deviation;
                    }
                    collect_metadata = true;
                }
            }
            prev_slice_origin = Some(slice_origin);

            if collect_metadata {
                let mut dict = image_metadata(&slice_image);
                if non_uniform {
                    dict.insert(
                        "ITK_non_uniform_sampling_deviation".to_string(),
                        deviation.to_string(),
                    );
                }
                collected.push(dict);
            }

            place_slice(
                &mut buffer,
                &slice_image,
                i,
                moving_dim,
                out_dim,
                &out_size,
                stride,
            );
        }

        self.slice_metadata = collected;

        let mut image = assemble_image(
            buffer,
            pixel_id,
            components,
            out_size,
            out_spacing,
            out_origin,
            out_direction,
        )?;
        if max_spacing_deviation > 0.0 {
            image.set_meta_data(
                "ITK_non_uniform_sampling_deviation",
                &max_spacing_deviation.to_string(),
            );
        }
        Ok(image)
    }

    /// `GetMetaDataKeys(slice)` (sitkImageSeriesReader.h:208-226). See
    /// [`ImageSeriesReader::execute`]'s docs for what `slice` indexes when
    /// [`ImageSeriesReader::meta_data_dictionary_array_update`] is off but
    /// collection activated partway through the series anyway.
    pub fn meta_data_keys(&self, slice: usize) -> Result<Vec<&str>> {
        self.slice_metadata
            .get(slice)
            .map(|m| m.keys().map(String::as_str).collect())
            .ok_or(IoError::SeriesSliceIndexOutOfRange {
                slice,
                len: self.slice_metadata.len(),
            })
    }

    /// `HasMetaDataKey(slice, key)` (sitkImageSeriesReader.h:228-234).
    pub fn has_meta_data_key(&self, slice: usize, key: &str) -> Result<bool> {
        self.slice_metadata
            .get(slice)
            .map(|m| m.contains_key(key))
            .ok_or(IoError::SeriesSliceIndexOutOfRange {
                slice,
                len: self.slice_metadata.len(),
            })
    }

    /// `GetMetaData(slice, key)` (sitkImageSeriesReader.h:236-248). Upstream
    /// throws when `key` is absent from the slice's dictionary; this returns
    /// `None`, matching [`ImageInformation::meta_data`]'s own convention.
    pub fn meta_data(&self, slice: usize, key: &str) -> Result<Option<&str>> {
        self.slice_metadata
            .get(slice)
            .map(|m| m.get(key).map(String::as_str))
            .ok_or(IoError::SeriesSliceIndexOutOfRange {
                slice,
                len: self.slice_metadata.len(),
            })
    }
}

fn check_pixel_type(file: &Path, info: &ImageInformation, expected: PixelId) -> Result<()> {
    if info.pixel_id != expected {
        return Err(IoError::SeriesPixelTypeMismatch {
            file: file.to_path_buf(),
            pixel_type: info.pixel_id.as_str(),
            expected: expected.as_str(),
        });
    }
    Ok(())
}

/// Pad geometry to `out_dim` axes: size `1`, spacing `1`, origin `0`, and an
/// identity direction row for every axis the file lacks — `itk::
/// ImageFileReader`'s own padding rule (itkImageFileReader.hxx:155-192),
/// applied by the internal `itk::ImageFileReader<TOutputImage>` every
/// `itk::ImageSeriesReader` constructs per file (itkImageSeriesReader.hxx:
/// 106-109,253,327).
fn pad_axes(
    file_dim: usize,
    out_dim: usize,
    size: &[usize],
    spacing: &[f64],
    origin: &[f64],
    direction: &[f64],
) -> (Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let padded_size: Vec<usize> = (0..out_dim)
        .map(|i| if i < file_dim { size[i] } else { 1 })
        .collect();
    let padded_spacing: Vec<f64> = (0..out_dim)
        .map(|i| if i < file_dim { spacing[i] } else { 1.0 })
        .collect();
    let padded_origin: Vec<f64> = (0..out_dim)
        .map(|i| if i < file_dim { origin[i] } else { 0.0 })
        .collect();
    let mut padded_direction = matrix::identity(out_dim);
    for row in 0..out_dim.min(file_dim) {
        for col in 0..out_dim.min(file_dim) {
            padded_direction[row * out_dim + col] = direction[row * file_dim + col];
        }
    }
    (padded_size, padded_spacing, padded_origin, padded_direction)
}

/// Apply `itk::ImageFileReader`'s positive-spacing normalization to the raw
/// geometry `read_information` reports, so the series reader derives its
/// inter-slice spacing/direction from the same normalized geometry upstream's
/// per-file `ImageFileReader` produces (itkImageSeriesReader.hxx:120-122). Only
/// the sign-flip half applies here — the `ITK_original_*` recording lives on
/// the per-slice [`Image`] reads, which are what the collected dictionaries
/// copy.
fn normalize_info_geometry(info: &mut ImageInformation) {
    crate::reader::flip_negative_spacing(info.dimension, &mut info.spacing, &mut info.direction);
}

fn padded_info_geometry(
    info: &ImageInformation,
    out_dim: usize,
) -> (Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>) {
    pad_axes(
        info.dimension,
        out_dim,
        &info.size,
        &info.spacing,
        &info.origin,
        &info.direction,
    )
}

fn padded_image_geometry(
    image: &Image,
    out_dim: usize,
) -> (Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>) {
    pad_axes(
        image.dimension(),
        out_dim,
        image.size(),
        image.spacing(),
        image.origin(),
        image.direction(),
    )
}

/// `ExposeMetaData<Array<double>>(dict, "ITK_ImageOrigin", position)`
/// (itkImageSeriesReader.hxx:160,173). A present, wrong-length array would
/// resize `position` itself — `itk::Array` is a `vnl_vector` alias whose
/// `operator=` reallocates to the source's length — and then be indexed at
/// every one of `out_dim` axes regardless, an out-of-bounds read in a real
/// build. This port only honours an override whose whitespace-separated
/// token count exactly equals `out_dim` (every token parsing as `f64`),
/// silently falling back to the real origin otherwise — ledger §1.
fn itk_image_origin_override(
    metadata: &BTreeMap<String, String>,
    out_dim: usize,
) -> Option<Vec<f64>> {
    let raw = metadata.get("ITK_ImageOrigin")?;
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.len() != out_dim {
        return None;
    }
    tokens.iter().map(|t| t.parse::<f64>().ok()).collect()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `itk::Math::AlmostEquals(a, 0.0)` — the zero-distance branch of
/// `FloatAlmostEqual`'s default `maxAbsoluteDifference = 0.1 * epsilon`
/// (itkMath.h). Not a bit-exact ULP reproduction; see [`almost_equal`].
fn almost_zero(a: f64) -> bool {
    a.abs() <= 0.1 * f64::EPSILON
}

/// A faithful-but-not-bit-exact stand-in for `itk::Math::AlmostEquals(a, b)`
/// (itkMath.h's `FloatAlmostEqual`, default `maxAbsoluteDifference = 0.1 *
/// epsilon`, `maxUlps = 4`) — this crate has no ULP-based comparison, so this
/// uses the same relative-tolerance style already established at
/// `sitk-transform`'s `bspline.rs:511`.
fn almost_equal(a: f64, b: f64) -> bool {
    let diff = (a - b).abs();
    diff <= 0.1 * f64::EPSILON || diff <= 4.0 * f64::EPSILON * a.abs().max(b.abs())
}

fn image_metadata(image: &Image) -> BTreeMap<String, String> {
    image
        .meta_data_keys()
        .into_iter()
        .map(|k| {
            (
                k.to_string(),
                image
                    .meta_data(k)
                    .expect("key came from meta_data_keys")
                    .to_string(),
            )
        })
        .collect()
}

/// Scatter one slice's pixels into the output buffer at series slot `i`.
///
/// Every output axis `j` maps to the slice's own axis `j` when `j` is one of
/// the slice's own axes and `j != moving_dim`; axis `moving_dim` is fixed at
/// the series slot `i`, overriding whatever the slice's own coordinate there
/// would have been — relevant only when the axis walk-down in
/// [`ImageSeriesReader::execute`]'s moving-dimension computation pulled
/// `moving_dim` below the slice's own dimension; every other output axis is
/// fixed at `0`, the same padding rule [`pad_axes`] applies to geometry.
/// `moving_dim == out_dim` (the single-file, no-moving-axis case) falls out
/// of these same comparisons with no separate branch: no output axis ever
/// equals it, so every axis maps straight through.
fn place_slice(
    buffer: &mut PixelBuffer,
    slice: &Image,
    i: usize,
    moving_dim: usize,
    out_dim: usize,
    out_size: &[usize],
    stride: usize,
) {
    let file_dim = slice.dimension();
    let file_size = slice.size();
    let mut file_strides = vec![1usize; file_dim];
    for d in 1..file_dim {
        file_strides[d] = file_strides[d - 1] * file_size[d - 1];
    }
    let mut out_strides = vec![1usize; out_dim];
    for d in 1..out_dim {
        out_strides[d] = out_strides[d - 1] * out_size[d - 1];
    }

    let slice_pixel_count: usize = (0..out_dim)
        .filter(|&d| d != moving_dim)
        .map(|d| out_size[d])
        .product();

    let mut coord = vec![0usize; out_dim];
    let mut in_offsets = Vec::with_capacity(slice_pixel_count);
    let mut out_offsets = Vec::with_capacity(slice_pixel_count);
    for _ in 0..slice_pixel_count {
        let mut in_flat = 0usize;
        for (k, &stride_k) in file_strides.iter().enumerate() {
            let c = if k == moving_dim { 0 } else { coord[k] };
            in_flat += c * stride_k;
        }
        in_offsets.push(in_flat);

        let mut out_flat = 0usize;
        for (d, &stride_d) in out_strides.iter().enumerate() {
            let c = if d == moving_dim { i } else { coord[d] };
            out_flat += c * stride_d;
        }
        out_offsets.push(out_flat);

        for d in 0..out_dim {
            if d == moving_dim {
                continue;
            }
            coord[d] += 1;
            if coord[d] < out_size[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    macro_rules! copy {
        ($src:ident, $variant:ident) => {{
            let PixelBuffer::$variant(dst) = buffer else {
                unreachable!("pixel type is validated equal before place_slice runs")
            };
            for (&o, &s) in out_offsets.iter().zip(&in_offsets) {
                dst[o * stride..(o + 1) * stride]
                    .copy_from_slice(&$src[s * stride..(s + 1) * stride]);
            }
        }};
    }
    match slice.buffer() {
        PixelBuffer::UInt8(v) => copy!(v, UInt8),
        PixelBuffer::Int8(v) => copy!(v, Int8),
        PixelBuffer::UInt16(v) => copy!(v, UInt16),
        PixelBuffer::Int16(v) => copy!(v, Int16),
        PixelBuffer::UInt32(v) => copy!(v, UInt32),
        PixelBuffer::Int32(v) => copy!(v, Int32),
        PixelBuffer::UInt64(v) => copy!(v, UInt64),
        PixelBuffer::Int64(v) => copy!(v, Int64),
        PixelBuffer::Float32(v) => copy!(v, Float32),
        PixelBuffer::Float64(v) => copy!(v, Float64),
    }
}

/// Wrap a completed component buffer into an [`Image`], dispatching on
/// scalar / vector / complex exactly as [`crate::nrrd`]'s `build_image` does:
/// `Image::assemble` is private, so a complex image is built through
/// `from_vec_complex` and then given its geometry.
fn assemble_image(
    buffer: PixelBuffer,
    pixel_id: PixelId,
    components: usize,
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
) -> Result<Image> {
    if !pixel_id.is_complex() {
        return if components <= 1 {
            Image::from_parts(buffer, size, spacing, origin, direction).map_err(IoError::Core)
        } else {
            Image::from_parts_vector(buffer, components, size, spacing, origin, direction)
                .map_err(IoError::Core)
        };
    }

    let mut image = match &buffer {
        PixelBuffer::Float32(v) => Image::from_vec_complex(
            &size,
            v.chunks_exact(2)
                .map(|c| Complex::new(c[0], c[1]))
                .collect(),
        ),
        PixelBuffer::Float64(v) => Image::from_vec_complex(
            &size,
            v.chunks_exact(2)
                .map(|c| Complex::new(c[0], c[1]))
                .collect(),
        ),
        _ => unreachable!("a complex PixelId always backs a Float32/Float64 buffer"),
    }
    .map_err(IoError::Core)?;
    image.set_spacing(&spacing).map_err(IoError::Core)?;
    image.set_origin(&origin).map_err(IoError::Core)?;
    image.set_direction(&direction).map_err(IoError::Core)?;
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write_image;

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "sitk_io_series_reader_test_{}_{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Hand-build a 2-D MetaImage with an arbitrary extra header line —
    /// `write_image` cannot express a custom meta-data field (`meta_image::
    /// write` never emits `img.meta_data_keys()`), so the `ITK_ImageOrigin`
    /// override tests need the header text directly.
    fn hand_built_2d_mha(
        size: [usize; 2],
        origin: [f64; 2],
        extra: &str,
        element_type: &str,
        data: &[u8],
    ) -> Vec<u8> {
        let header = format!(
            "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             BinaryDataByteOrderMSB = False\n\
             CompressedData = False\n\
             TransformMatrix = 1 0 0 1\n\
             Offset = {} {}\n\
             ElementSpacing = 1 1\n\
             DimSize = {} {}\n\
             {extra}\
             ElementType = {element_type}\n\
             ElementDataFile = LOCAL\n",
            origin[0], origin[1], size[0], size[1],
        );
        let mut bytes = header.into_bytes();
        bytes.extend_from_slice(data);
        bytes
    }

    /// A 2-D MetaImage with an explicit (possibly negative) `ElementSpacing` —
    /// `write_image` refuses a negative spacing (`Image::set_spacing` rejects
    /// non-positive), but `Image::from_parts` (the reader's own path) accepts
    /// it, so the header text is written directly.
    fn mha_2d_with_spacing(
        size: [usize; 2],
        spacing: [f64; 2],
        origin: [f64; 2],
        data: &[u8],
    ) -> Vec<u8> {
        let header = format!(
            "ObjectType = Image\n\
             NDims = 2\n\
             BinaryData = True\n\
             BinaryDataByteOrderMSB = False\n\
             CompressedData = False\n\
             TransformMatrix = 1 0 0 1\n\
             Offset = {} {}\n\
             ElementSpacing = {} {}\n\
             DimSize = {} {}\n\
             ElementType = MET_UCHAR\n\
             ElementDataFile = LOCAL\n",
            origin[0], origin[1], spacing[0], spacing[1], size[0], size[1],
        );
        let mut bytes = header.into_bytes();
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn series_reader_flips_negative_spacing_and_records_the_originals() {
        // Two 2-D slices carrying a negative Y spacing — the sign a raw read
        // preserves (dicom.rs) but that every read path must flip positive.
        // The series reader is the one path that used to bypass
        // `normalize_reader_geometry` (itkImageSeriesReader.hxx:120-122,468).
        let dir = tmp_dir("negative_spacing_series");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        std::fs::write(
            &p0,
            mha_2d_with_spacing([2, 2], [2.0, -3.0], [0.0, 0.0], &[0, 1, 2, 3]),
        )
        .unwrap();
        std::fs::write(
            &p1,
            mha_2d_with_spacing([2, 2], [2.0, -3.0], [0.0, 0.0], &[10, 11, 12, 13]),
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader
            .set_file_names(&[&p0, &p1])
            .set_meta_data_dictionary_array_update(true);
        let image = reader.execute().unwrap();

        // The derived volume geometry is flipped positive: the negative Y
        // spacing becomes positive and the Y direction *column* is negated.
        assert_eq!(image.dimension(), 3);
        assert_eq!(image.spacing(), &[2.0, 3.0, 1.0]);
        assert_eq!(
            image.direction(),
            &[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        );
        // Pixels are untouched by the sign flip.
        assert_eq!(
            image.scalar_slice::<u8>().unwrap(),
            &[0, 1, 2, 3, 10, 11, 12, 13]
        );

        // Every collected per-slice dictionary carries the raw geometry under
        // `ITK_original_*`, exactly as upstream's per-file `ImageFileReader`
        // records and the series reader copies.
        for slice in 0..2 {
            assert_eq!(
                reader.meta_data(slice, "ITK_original_spacing").unwrap(),
                Some("2 -3"),
                "slice {slice} ITK_original_spacing"
            );
            assert_eq!(
                reader.meta_data(slice, "ITK_original_direction").unwrap(),
                Some("1 0 0 1"),
                "slice {slice} ITK_original_direction"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_2d_files_stack_into_a_3d_volume_in_series_order() {
        let dir = tmp_dir("basic_stack");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        write_image(&Image::from_vec(&[2, 2], vec![0u8, 1, 2, 3]).unwrap(), &p0).unwrap();
        write_image(
            &Image::from_vec(&[2, 2], vec![10u8, 11, 12, 13]).unwrap(),
            &p1,
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]);
        let image = reader.execute().unwrap();

        assert_eq!(image.dimension(), 3);
        assert_eq!(image.size(), &[2, 2, 2]);
        assert_eq!(
            image.scalar_slice::<u8>().unwrap(),
            &[0, 1, 2, 3, 10, 11, 12, 13]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reverse_order_swaps_which_file_fills_which_series_slot() {
        let dir = tmp_dir("reverse_order");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        write_image(&Image::from_vec(&[2, 2], vec![0u8, 1, 2, 3]).unwrap(), &p0).unwrap();
        write_image(
            &Image::from_vec(&[2, 2], vec![10u8, 11, 12, 13]).unwrap(),
            &p1,
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]).set_reverse_order(true);
        let image = reader.execute().unwrap();

        assert_eq!(
            image.scalar_slice::<u8>().unwrap(),
            &[10, 11, 12, 13, 0, 1, 2, 3]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_2d_file_promotes_with_one_z_slice_and_unmodified_geometry() {
        let dir = tmp_dir("single_2d");
        let path = dir.join("s0.mha");
        let mut img = Image::from_vec(&[3, 2], vec![0u8, 1, 2, 3, 4, 5]).unwrap();
        img.set_spacing(&[2.0, 3.0]).unwrap();
        img.set_origin(&[10.0, 20.0]).unwrap();
        img.set_direction(&[0.0, 1.0, -1.0, 0.0]).unwrap();
        write_image(&img, &path).unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&path]);
        let image = reader.execute().unwrap();

        assert_eq!(image.dimension(), 3);
        assert_eq!(image.size(), &[3, 2, 1]);
        assert_eq!(image.spacing(), &[2.0, 3.0, 1.0]);
        assert_eq!(image.origin(), &[10.0, 20.0, 0.0]);
        assert_eq!(
            image.direction(),
            &[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(image.scalar_slice::<u8>().unwrap(), &[0, 1, 2, 3, 4, 5]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_3d_file_with_a_unit_third_axis_collapses_and_keeps_its_own_geometry() {
        let dir = tmp_dir("single_3d_collapse");
        let path = dir.join("s0.mha");
        let mut img = Image::from_vec(&[3, 2, 1], (0u8..6).collect()).unwrap();
        img.set_spacing(&[1.5, 2.5, 4.0]).unwrap();
        img.set_origin(&[1.0, 2.0, 3.0]).unwrap();
        write_image(&img, &path).unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&path]);
        let image = reader.execute().unwrap();

        // A 4-D promotion of a 3-D file collapses back to 3-D when that
        // file's own third axis has size 1 (sitkImageSeriesReader.cxx:219-229).
        assert_eq!(image.dimension(), 3);
        assert_eq!(image.size(), &[3, 2, 1]);
        assert_eq!(image.spacing(), &[1.5, 2.5, 4.0]);
        assert_eq!(image.origin(), &[1.0, 2.0, 3.0]);
        assert_eq!(image.scalar_slice::<u8>().unwrap(), &[0, 1, 2, 3, 4, 5]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn three_3d_files_promote_to_a_4d_volume_when_the_z_axis_is_not_unit() {
        let dir = tmp_dir("promote_4d");
        let paths: Vec<_> = (0..3).map(|i| dir.join(format!("s{i}.mha"))).collect();
        for (i, path) in paths.iter().enumerate() {
            let value = (i as u8) * 10;
            write_image(&Image::from_vec(&[2, 2, 2], vec![value; 8]).unwrap(), path).unwrap();
        }

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&paths);
        let image = reader.execute().unwrap();

        assert_eq!(image.dimension(), 4);
        assert_eq!(image.size(), &[2, 2, 2, 3]);
        let mut expected = Vec::new();
        expected.extend(std::iter::repeat_n(0u8, 8));
        expected.extend(std::iter::repeat_n(10u8, 8));
        expected.extend(std::iter::repeat_n(20u8, 8));
        assert_eq!(image.scalar_slice::<u8>().unwrap(), expected.as_slice());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn size_mismatch_between_slices_is_reported_with_the_reference_file() {
        let dir = tmp_dir("size_mismatch");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        write_image(&Image::from_vec(&[2, 2], vec![0u8, 1, 2, 3]).unwrap(), &p0).unwrap();
        write_image(
            &Image::from_vec(&[3, 2], vec![0u8, 1, 2, 3, 4, 5]).unwrap(),
            &p1,
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]);
        let err = reader.execute().unwrap_err();
        match err {
            IoError::SeriesSizeMismatch {
                file,
                size,
                expected,
                reference_file,
            } => {
                assert_eq!(file, p1);
                assert_eq!(size, vec![3, 2, 1]);
                assert_eq!(expected, vec![2, 2, 1]);
                assert_eq!(reference_file, p0);
            }
            other => panic!("expected SeriesSizeMismatch, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pixel_type_mismatch_at_a_middle_slice_is_reported() {
        let dir = tmp_dir("pixel_mismatch");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        let p2 = dir.join("s2.mha");
        write_image(&Image::from_vec(&[2, 2], vec![0u8, 1, 2, 3]).unwrap(), &p0).unwrap();
        write_image(&Image::from_vec(&[2, 2], vec![0u16, 1, 2, 3]).unwrap(), &p1).unwrap();
        write_image(&Image::from_vec(&[2, 2], vec![0u8, 1, 2, 3]).unwrap(), &p2).unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1, &p2]);
        let err = reader.execute().unwrap_err();
        match err {
            IoError::SeriesPixelTypeMismatch {
                file,
                pixel_type,
                expected,
            } => {
                assert_eq!(file, p1);
                assert_eq!(pixel_type, PixelId::UInt16.as_str());
                assert_eq!(expected, PixelId::UInt8.as_str());
            }
            other => panic!("expected SeriesPixelTypeMismatch, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_file_names_is_rejected() {
        let mut reader = ImageSeriesReader::new();
        assert!(matches!(
            reader.execute().unwrap_err(),
            IoError::EmptySeriesFileNames
        ));
    }

    #[test]
    fn a_5d_input_file_exceeds_sitk_max_dimension() {
        let dir = tmp_dir("too_many_dims");
        let path = dir.join("s0.mha");
        write_image(
            &Image::from_vec(&[2, 2, 2, 2, 2], vec![0u8; 32]).unwrap(),
            &path,
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&path]);
        assert!(matches!(
            reader.execute().unwrap_err(),
            IoError::UnsupportedSeriesDimension(5)
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn itk_image_origin_override_feeds_spacing_but_not_the_output_origin() {
        let dir = tmp_dir("origin_override_spacing");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        std::fs::write(
            &p0,
            hand_built_2d_mha(
                [2, 2],
                [5.0, 5.0],
                "ITK_ImageOrigin = 0 0 0\n",
                "MET_UCHAR",
                &[0, 1, 2, 3],
            ),
        )
        .unwrap();
        std::fs::write(
            &p1,
            hand_built_2d_mha(
                [2, 2],
                [5.0, 5.0],
                "ITK_ImageOrigin = 0 0 8\n",
                "MET_UCHAR",
                &[10, 11, 12, 13],
            ),
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]);
        let image = reader.execute().unwrap();

        // The override feeds the spacing/direction derivation (8 / (2-1)).
        assert_eq!(image.spacing(), &[1.0, 1.0, 8.0]);
        // But the image's own origin is always the first file's *real*,
        // un-overridden origin (itkImageSeriesReader.hxx:121,226).
        assert_eq!(image.origin(), &[5.0, 5.0, 0.0]);
        assert_eq!(
            image.direction(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_wrong_length_itk_image_origin_override_is_ignored() {
        let dir = tmp_dir("origin_override_wrong_length");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        std::fs::write(
            &p0,
            hand_built_2d_mha(
                [2, 2],
                [5.0, 5.0],
                "ITK_ImageOrigin = 1 2\n",
                "MET_UCHAR",
                &[0, 1, 2, 3],
            ),
        )
        .unwrap();
        std::fs::write(
            &p1,
            hand_built_2d_mha(
                [2, 2],
                [5.0, 5.0],
                "ITK_ImageOrigin = 1 2\n",
                "MET_UCHAR",
                &[10, 11, 12, 13],
            ),
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]);
        let image = reader.execute().unwrap();

        // The malformed (wrong-token-count) override falls back to the real
        // (identical, here) origins, so the derived vector is zero.
        assert_eq!(image.spacing(), &[1.0, 1.0, 1.0]);
        // Falling back to spacing_defined = false means the rolling
        // non-uniform-sampling check never runs, so nothing is collected.
        assert!(matches!(
            reader.meta_data_keys(0).unwrap_err(),
            IoError::SeriesSliceIndexOutOfRange { slice: 0, len: 0 }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn force_orthogonal_direction_on_flips_sign_to_agree_with_the_derived_direction() {
        let dir = tmp_dir("force_orthogonal_flip");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        std::fs::write(
            &p0,
            hand_built_2d_mha(
                [2, 2],
                [5.0, 5.0],
                "ITK_ImageOrigin = 0 0 0\n",
                "MET_UCHAR",
                &[0, 1, 2, 3],
            ),
        )
        .unwrap();
        std::fs::write(
            &p1,
            hand_built_2d_mha(
                [2, 2],
                [5.0, 5.0],
                "ITK_ImageOrigin = 0 -3 -4\n",
                "MET_UCHAR",
                &[10, 11, 12, 13],
            ),
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]);
        let image = reader.execute().unwrap();

        // dirN = [0,-3,-4], dot with the identity's existing column [0,0,1]
        // is -4 < 0, so the existing column is kept but sign-flipped.
        assert_eq!(
            image.direction(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn force_orthogonal_direction_off_replaces_the_column_with_the_derived_direction() {
        let dir = tmp_dir("force_orthogonal_replace");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        std::fs::write(
            &p0,
            hand_built_2d_mha(
                [2, 2],
                [5.0, 5.0],
                "ITK_ImageOrigin = 0 0 0\n",
                "MET_UCHAR",
                &[0, 1, 2, 3],
            ),
        )
        .unwrap();
        std::fs::write(
            &p1,
            hand_built_2d_mha(
                [2, 2],
                [5.0, 5.0],
                "ITK_ImageOrigin = 0 3 4\n",
                "MET_UCHAR",
                &[10, 11, 12, 13],
            ),
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader
            .set_file_names(&[&p0, &p1])
            .set_force_orthogonal_direction(false);
        let image = reader.execute().unwrap();

        // dirN = [0,3,4], norm 5: the moving-axis column is replaced outright
        // with [0, 0.6, 0.8], not merely sign-agreed with the existing one.
        assert_eq!(
            image.direction(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.8]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn meta_data_dictionary_array_update_off_collects_nothing_when_spacing_is_uniform() {
        let dir = tmp_dir("mdda_off");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        write_image(&Image::from_vec(&[2, 2], vec![0u8, 1, 2, 3]).unwrap(), &p0).unwrap();
        write_image(
            &Image::from_vec(&[2, 2], vec![10u8, 11, 12, 13]).unwrap(),
            &p1,
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]);
        reader.execute().unwrap();

        assert!(matches!(
            reader.meta_data_keys(0).unwrap_err(),
            IoError::SeriesSliceIndexOutOfRange { slice: 0, len: 0 }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn meta_data_dictionary_array_update_on_collects_every_slice_from_the_start() {
        let dir = tmp_dir("mdda_on");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        write_image(&Image::from_vec(&[2, 2], vec![0u8, 1, 2, 3]).unwrap(), &p0).unwrap();
        write_image(
            &Image::from_vec(&[2, 2], vec![10u8, 11, 12, 13]).unwrap(),
            &p1,
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader
            .set_file_names(&[&p0, &p1])
            .set_meta_data_dictionary_array_update(true);
        reader.execute().unwrap();

        assert!(!reader.meta_data_keys(0).unwrap().is_empty());
        assert!(reader.has_meta_data_key(0, "ITK_InputFilterName").unwrap());
        assert!(reader.has_meta_data_key(1, "ITK_InputFilterName").unwrap());
        assert!(matches!(
            reader.meta_data_keys(2).unwrap_err(),
            IoError::SeriesSliceIndexOutOfRange { slice: 2, len: 2 }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn non_uniform_sampling_activates_collection_mid_series_and_shifts_the_slice_index() {
        let dir = tmp_dir("non_uniform");
        let paths: Vec<_> = (0..4).map(|i| dir.join(format!("s{i}.mha"))).collect();
        for (i, path) in paths.iter().enumerate() {
            let extra = match i {
                0 => "ITK_ImageOrigin = 0 0 0\nSliceTag = 0\n".to_string(),
                3 => "ITK_ImageOrigin = 0 0 6\nSliceTag = 3\n".to_string(),
                n => format!("SliceTag = {n}\n"),
            };
            std::fs::write(
                path,
                hand_built_2d_mha([2, 2], [7.0, 7.0], &extra, "MET_UCHAR", &[i as u8; 4]),
            )
            .unwrap();
        }

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&paths);
        let image = reader.execute().unwrap();

        // Uniform derived spacing is 6 / (4-1) = 2.0, but every file's own
        // (un-overridden) origin is identical, so every step after the first
        // registers as a full deviation and collection turns on at series
        // slot 1 and stays on (itkImageSeriesReader.hxx:439-452).
        assert_eq!(
            image.meta_data("ITK_non_uniform_sampling_deviation"),
            Some("2")
        );
        assert!(!reader.meta_data_keys(0).unwrap().is_empty());
        // Collected index 0 is series slot 1, not slot 0 — the index-shift
        // quirk the upstream `.at(i)` reproduces (sitkMetaDataDictionaryCustomCast.hxx:38-46).
        assert_eq!(reader.meta_data(0, "SliceTag").unwrap(), Some("1"));
        assert_eq!(reader.meta_data(1, "SliceTag").unwrap(), Some("2"));
        assert_eq!(reader.meta_data(2, "SliceTag").unwrap(), Some("3"));
        assert!(matches!(
            reader.meta_data_keys(3).unwrap_err(),
            IoError::SeriesSliceIndexOutOfRange { slice: 3, len: 3 }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn vector_pixel_slices_scatter_by_component() {
        let dir = tmp_dir("vector_scatter");
        let p0 = dir.join("s0.mha");
        let p1 = dir.join("s1.mha");
        let d0: Vec<u8> = vec![0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32];
        let d1: Vec<u8> = d0.iter().map(|v| v + 100).collect();
        write_image(
            &Image::from_vec_vector::<u8>(&[2, 2], 3, d0.clone()).unwrap(),
            &p0,
        )
        .unwrap();
        write_image(
            &Image::from_vec_vector::<u8>(&[2, 2], 3, d1.clone()).unwrap(),
            &p1,
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]);
        let image = reader.execute().unwrap();

        assert_eq!(image.pixel_id(), PixelId::VectorUInt8);
        assert_eq!(image.number_of_components_per_pixel(), 3);
        assert_eq!(image.size(), &[2, 2, 2]);
        let mut expected = d0;
        expected.extend(d1);
        assert_eq!(image.component_slice::<u8>().unwrap(), expected.as_slice());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn complex_pixel_slices_round_trip_through_nrrd() {
        let dir = tmp_dir("complex_scatter");
        let p0 = dir.join("s0.nrrd");
        let p1 = dir.join("s1.nrrd");
        let d0: Vec<Complex<f32>> = (0..4)
            .map(|i| Complex::new(i as f32 + 1.0, -(i as f32) - 1.0))
            .collect();
        let d1: Vec<Complex<f32>> = (0..4)
            .map(|i| Complex::new(i as f32 + 10.0, -(i as f32) - 10.0))
            .collect();
        write_image(
            &Image::from_vec_complex::<f32>(&[2, 2], d0.clone()).unwrap(),
            &p0,
        )
        .unwrap();
        write_image(
            &Image::from_vec_complex::<f32>(&[2, 2], d1.clone()).unwrap(),
            &p1,
        )
        .unwrap();

        let mut reader = ImageSeriesReader::new();
        reader.set_file_names(&[&p0, &p1]);
        let image = reader.execute().unwrap();

        assert_eq!(image.pixel_id(), PixelId::ComplexFloat32);
        assert_eq!(image.size(), &[2, 2, 2]);
        let flatten =
            |v: &[Complex<f32>]| -> Vec<f32> { v.iter().flat_map(|c| [c.re, c.im]).collect() };
        let mut expected = flatten(&d0);
        expected.extend(flatten(&d1));
        assert_eq!(image.component_slice::<f32>().unwrap(), expected.as_slice());
        std::fs::remove_dir_all(&dir).ok();
    }
}
