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

        let first_info = io.read_information(&self.file_names[first_idx])?;
        check_pixel_type(&self.file_names[first_idx], &first_info, pixel_id)?;
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

            let last_info = io.read_information(&self.file_names[last_idx])?;
            check_pixel_type(&self.file_names[last_idx], &last_info, pixel_id)?;
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
            let slice_image = io.read(&path)?;
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
