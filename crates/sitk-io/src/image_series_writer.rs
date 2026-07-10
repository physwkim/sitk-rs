//! [`ImageSeriesWriter`] — SimpleITK's `itk::simple::ImageSeriesWriter`
//! (sitkImageSeriesWriter.h:37-207, sitkImageSeriesWriter.cxx), wrapping
//! `itk::ImageSeriesWriter<TInputImage, TOutputImage>`
//! (itkImageSeriesWriter.hxx).

use std::path::{Path, PathBuf};

use sitk_core::{Complex, Image, PixelBuffer, PixelId, matrix};

use crate::error::{IoError, Result};
use crate::image_io::{image_io_by_name, registered_image_ios, writer_for};
use crate::writer::WriteOptions;

/// Slice a 3-D image into `size()[2]` numbered 2-D files.
///
/// ```no_run
/// # use sitk_core::{Image, PixelId};
/// # use sitk_io::ImageSeriesWriter;
/// let image = Image::new(&[4, 4, 3], PixelId::UInt8);
/// let mut writer = ImageSeriesWriter::new();
/// writer.set_file_names(&["s0.mha", "s1.mha", "s2.mha"]);
/// writer.execute(&image)?;
/// # Ok::<(), sitk_io::IoError>(())
/// ```
///
/// Only a 3-D image is accepted: `ImageSeriesWriter` dispatches through the
/// same `MemberFunctionFactory` machinery as [`crate::ImageFileWriter`], but
/// registers only dimension 3 (sitkImageSeriesWriter.cxx:43-55) — see
/// [`ImageSeriesWriter::execute`].
///
/// `SetCompressor` is not exposed, matching [`crate::ImageFileWriter`]'s own
/// scope decision (ledger §6): the empty string — which resolves to each
/// format's own default compressor — is the only setting this port
/// implements.
#[derive(Clone, Debug, Default)]
pub struct ImageSeriesWriter {
    file_names: Vec<PathBuf>,
    image_io_name: Option<String>,
    options: WriteOptions,
}

impl ImageSeriesWriter {
    /// A writer with no file names, letting the registry pick the format from
    /// the first file name at [`ImageSeriesWriter::execute`] time.
    pub fn new() -> Self {
        Self::default()
    }

    /// `SetFileNames`. The number of names must equal the image's Z size.
    pub fn set_file_names<P: AsRef<Path>>(&mut self, names: &[P]) -> &mut Self {
        self.file_names = names.iter().map(|p| p.as_ref().to_path_buf()).collect();
        self
    }

    /// `GetFileNames`.
    pub fn file_names(&self) -> &[PathBuf] {
        &self.file_names
    }

    /// `SetUseCompression` (sitkImageSeriesWriter.h:95-116). A request: a
    /// format that cannot compress ignores it.
    pub fn set_use_compression(&mut self, use_compression: bool) -> &mut Self {
        self.options.use_compression = use_compression;
        self
    }

    /// `GetUseCompression`.
    pub fn use_compression(&self) -> bool {
        self.options.use_compression
    }

    /// `UseCompressionOn`.
    pub fn use_compression_on(&mut self) -> &mut Self {
        self.set_use_compression(true)
    }

    /// `UseCompressionOff`.
    pub fn use_compression_off(&mut self) -> &mut Self {
        self.set_use_compression(false)
    }

    /// `SetCompressionLevel` (sitkImageSeriesWriter.h:118-128). `-1`, the
    /// default, leaves each format on its own level.
    pub fn set_compression_level(&mut self, level: i32) -> &mut Self {
        self.options.compression_level = level;
        self
    }

    /// `GetCompressionLevel`.
    pub fn compression_level(&self) -> i32 {
        self.options.compression_level
    }

    /// Override the automatically detected [`ImageIo`](crate::ImageIo) by
    /// class name. `None` (the default) restores automatic detection from the
    /// first file name.
    ///
    /// `SetImageIO` (sitkImageSeriesWriter.h:70-86). An unknown name is
    /// reported by [`ImageSeriesWriter::execute`] as
    /// [`IoError::UnknownImageIo`], matching upstream, where
    /// `CreateImageIOByName` throws inside `GetImageIOBase`.
    pub fn set_image_io(&mut self, name: Option<&str>) -> &mut Self {
        self.image_io_name = name.map(str::to_string);
        self
    }

    /// `GetImageIO` — the empty string upstream, `None` here.
    pub fn image_io(&self) -> Option<&str> {
        self.image_io_name.as_deref()
    }

    /// The class names of every registered [`ImageIo`](crate::ImageIo) —
    /// `GetRegisteredImageIOs` (sitkImageSeriesWriter.h:65-68).
    pub fn registered_image_ios(&self) -> Vec<&'static str> {
        registered_image_ios()
    }

    /// Write `image`'s Z slices to [`ImageSeriesWriter::set_file_names`]'s
    /// paths, in order.
    ///
    /// `Execute` (sitkImageSeriesWriter.cxx:200-209,211-269) plus
    /// `itk::ImageSeriesWriter::WriteFiles` (itkImageSeriesWriter.hxx:
    /// 167-343). Checks run in this order, matching upstream's own dispatch
    /// and `ExecuteInternal`/`WriteFiles` sequence:
    ///
    /// 1. `image.dimension() != 3` —
    ///    [`IoError::SeriesWriterUnsupportedDimension`], the same
    ///    `MemberFunctionFactory` rejection [`crate::ImageFileWriter`] would
    ///    hit for an unregistered pixel type/dimension pair
    ///    (sitkMemberFunctionFactory.hxx).
    /// 2. no file names — [`IoError::EmptySeriesWriterFileNames`].
    /// 3. every file name checked for a DICOM extension —
    ///    [`IoError::SeriesWriterDicomRejected`]. The check is `fn.substr(
    ///    fn.find_last_of(".") + 1)` on the *whole path string*
    ///    (sitkImageSeriesWriter.cxx:228), not just the file name: with no
    ///    `.` anywhere in the path, C++'s `std::string::npos + 1` wraps to
    ///    `0`, so the "extension" becomes the entire path — a file literally
    ///    named `dcm` or `dicom` (no extension, no directory prefix) is
    ///    rejected — and a `.` in a *parent directory* name can feed this
    ///    computation too, since the search is not confined to the file name
    ///    component. This port reproduces both quirks by operating on the raw
    ///    path string rather than [`Path::extension`].
    /// 4. the `ImageIo` is resolved once, from
    ///    [`ImageSeriesWriter::set_image_io`] or else the first file name.
    /// 5. `file_names.len() != image.size()[2]` —
    ///    [`IoError::SeriesFileNameCountMismatch`], including the upstream
    ///    string's own trailing space (itkImageSeriesWriter.hxx:240-244) —
    ///    upstream throws rather than truncating either list.
    /// 6. each slice is written in turn.
    ///
    /// # Per-slice geometry
    ///
    /// 2-D origin and spacing are the literal first two components of the
    /// 3-D image's own origin/spacing — an index truncation, not a physical
    /// re-projection. 2-D direction is the literal top-left 2x2 submatrix of
    /// the 3-D direction matrix, replaced with the identity if that submatrix
    /// is exactly singular (`vnl_determinant(...) == 0.0`,
    /// itkImageSeriesWriter.hxx:210-220) — an arbitrary but upstream's own
    /// choice, since a 2x2 matrix cannot represent an arbitrary 3x3
    /// orientation anyway.
    ///
    /// # Per-slice meta-data
    ///
    /// Every slice's `ImageIo` dictionary gets `ITK_Origin` (the *full* 3-D
    /// physical point of index `(0, 0, slice)`, not the truncated 2-D
    /// origin), `ITK_Spacing` (the full 3-D spacing, constant across
    /// slices), `ITK_NumberOfDimensions` (`3`, constant), and `ITK_ZDirection`
    /// (the full 3x3 direction matrix, *transposed*) —
    /// itkImageSeriesWriter.hxx:293-333. Most of this crate's `ImageIo`
    /// writers do not persist a dictionary at all (see [`crate::meta_image`]);
    /// [`crate::image_hdf5`] is the one format here that round-trips it.
    pub fn execute(&self, image: &Image) -> Result<()> {
        if image.dimension() != 3 {
            return Err(IoError::SeriesWriterUnsupportedDimension {
                pixel_type: image.pixel_id().as_str(),
                dimension: image.dimension(),
            });
        }
        if self.file_names.is_empty() {
            return Err(IoError::EmptySeriesWriterFileNames);
        }
        for name in &self.file_names {
            if is_dicom_extension(name) {
                return Err(IoError::SeriesWriterDicomRejected);
            }
        }

        let io = match &self.image_io_name {
            Some(name) => image_io_by_name(name)?,
            None => writer_for(&self.file_names[0])?,
        };

        let z_size = image.size()[2];
        if self.file_names.len() != z_size {
            return Err(IoError::SeriesFileNameCountMismatch {
                actual: self.file_names.len(),
                expected: z_size,
            });
        }

        let (size2, spacing2, origin2, direction2) = project_2d_geometry(image);
        let pixel_id = image.pixel_id();
        let components = image.number_of_components_per_pixel();

        for (slice, path) in self.file_names.iter().enumerate() {
            let buffer = extract_slice_buffer(image, slice);
            let mut slice_image = assemble_2d_image(
                buffer,
                pixel_id,
                components,
                size2.clone(),
                spacing2.clone(),
                origin2.clone(),
                direction2.clone(),
            )?;
            for (key, value) in slice_metadata(image, slice) {
                slice_image.set_meta_data(&key, &value);
            }
            io.write(&slice_image, path, &self.options)?;
        }
        Ok(())
    }

    /// `Execute(image, fileNames, useCompression, compressionLevel)`
    /// (sitkImageSeriesWriter.cxx:188-198). As with
    /// [`crate::ImageFileWriter::execute_with`], upstream's overload sets
    /// `SetFileNames` / `SetUseCompression` / `SetCompressionLevel` on `this`
    /// and then calls the one-argument `Execute` — so all three persist.
    pub fn execute_with<P: AsRef<Path>>(
        &mut self,
        image: &Image,
        file_names: &[P],
        use_compression: bool,
        compression_level: i32,
    ) -> Result<()> {
        self.set_file_names(file_names)
            .set_use_compression(use_compression)
            .set_compression_level(compression_level);
        self.execute(image)
    }
}

/// `itk::simple::WriteImage(image, fileNames, useCompression,
/// compressionLevel)` (sitkImageSeriesWriter.cxx:36-41).
pub fn write_image_series<P: AsRef<Path>>(
    image: &Image,
    file_names: &[P],
    use_compression: bool,
    compression_level: i32,
) -> Result<()> {
    let mut writer = ImageSeriesWriter::new();
    writer.execute_with(image, file_names, use_compression, compression_level)
}

/// `fn.substr(fn.find_last_of(".") + 1)`, lower-cased, compared to `"dcm"` /
/// `"dicom"` (sitkImageSeriesWriter.cxx:227-234) — see
/// [`ImageSeriesWriter::execute`] for the no-dot wraparound this reproduces.
fn is_dicom_extension(path: &Path) -> bool {
    let full = path.as_os_str().to_string_lossy();
    let ext = match full.rfind('.') {
        Some(pos) => &full[pos + 1..],
        None => &full[..],
    };
    ext.eq_ignore_ascii_case("dcm") || ext.eq_ignore_ascii_case("dicom")
}

/// Project the 3-D image's geometry down to 2 axes — `WriteFiles`
/// (itkImageSeriesWriter.hxx:196-220).
fn project_2d_geometry(image: &Image) -> (Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let size = image.size();
    let spacing = image.spacing();
    let origin = image.origin();
    let direction = image.direction();

    let size2 = vec![size[0], size[1]];
    let spacing2 = vec![spacing[0], spacing[1]];
    let origin2 = vec![origin[0], origin[1]];

    let mut direction2 = vec![0.0; 4];
    for row in 0..2 {
        for col in 0..2 {
            direction2[row * 2 + col] = direction[row * 3 + col];
        }
    }
    if matrix::determinant_magnitude(&direction2, 2) == 0.0 {
        direction2 = matrix::identity(2);
    }
    (size2, spacing2, origin2, direction2)
}

/// The full 3-D physical point of index `(0, 0, slice)` —
/// `inputImage->TransformIndexToPhysicalPoint(inIndex, origin2)`
/// (itkImageSeriesWriter.hxx:303).
fn slice_physical_origin(image: &Image, slice: usize) -> Vec<f64> {
    let spacing = image.spacing();
    let origin = image.origin();
    let direction = image.direction();
    let scaled = [0.0, 0.0, spacing[2] * slice as f64];
    let rotated = matrix::mat_vec(direction, &scaled, 3);
    vec![
        origin[0] + rotated[0],
        origin[1] + rotated[1],
        origin[2] + rotated[2],
    ]
}

/// `ITK_Origin` / `ITK_Spacing` / `ITK_NumberOfDimensions` / `ITK_ZDirection`
/// (itkImageSeriesWriter.hxx:293-333).
fn slice_metadata(image: &Image, slice: usize) -> Vec<(String, String)> {
    let origin = slice_physical_origin(image, slice);
    let spacing = image.spacing();
    let direction = image.direction();

    // ITK_ZDirection is the 3x3 direction matrix, transposed
    // (`directionMatrix[j][i] = direction2[i][j]`, itkImageSeriesWriter.hxx:329).
    let mut z_direction = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            z_direction[j * 3 + i] = direction[i * 3 + j];
        }
    }

    vec![
        ("ITK_Origin".to_string(), fmt_f64_list(&origin)),
        ("ITK_Spacing".to_string(), fmt_f64_list(spacing)),
        ("ITK_NumberOfDimensions".to_string(), "3".to_string()),
        ("ITK_ZDirection".to_string(), fmt_f64_list(&z_direction)),
    ]
}

fn fmt_f64_list(values: &[f64]) -> String {
    values
        .iter()
        .map(f64::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Copy Z-slice `slice`'s pixels out of `image`'s buffer. Contiguous: `x`
/// varies fastest, then `y`, then `z`, so one Z-slice is one contiguous
/// component range.
fn extract_slice_buffer(image: &Image, slice: usize) -> PixelBuffer {
    let size = image.size();
    let stride = image.buffer_stride();
    let per_slice = size[0] * size[1] * stride;
    let start = slice * per_slice;
    let end = start + per_slice;

    macro_rules! take {
        ($v:ident, $variant:ident) => {
            PixelBuffer::$variant($v[start..end].to_vec())
        };
    }
    match image.buffer() {
        PixelBuffer::UInt8(v) => take!(v, UInt8),
        PixelBuffer::Int8(v) => take!(v, Int8),
        PixelBuffer::UInt16(v) => take!(v, UInt16),
        PixelBuffer::Int16(v) => take!(v, Int16),
        PixelBuffer::UInt32(v) => take!(v, UInt32),
        PixelBuffer::Int32(v) => take!(v, Int32),
        PixelBuffer::UInt64(v) => take!(v, UInt64),
        PixelBuffer::Int64(v) => take!(v, Int64),
        PixelBuffer::Float32(v) => take!(v, Float32),
        PixelBuffer::Float64(v) => take!(v, Float64),
    }
}

/// Wrap a completed component buffer into an [`Image`], dispatching on
/// scalar / vector / complex exactly as [`crate::image_series_reader`]'s own
/// `assemble_image` does (itself matching [`crate::nrrd`]'s `build_image`):
/// `Image::assemble` is private, so a complex image is built through
/// `from_vec_complex` and then given its geometry.
fn assemble_2d_image(
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
    use crate::read_image;

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "sitk_io_series_writer_test_{}_{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn basic_3d_image_writes_one_file_per_z_slice() {
        let dir = tmp_dir("basic_write");
        let paths: Vec<_> = (0..3).map(|i| dir.join(format!("s{i}.mha"))).collect();
        let data: Vec<u8> = (0..12).collect();
        let image = Image::from_vec(&[2, 2, 3], data).unwrap();

        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&paths);
        writer.execute(&image).unwrap();

        for (i, path) in paths.iter().enumerate() {
            let slice = read_image(path).unwrap();
            assert_eq!(slice.dimension(), 2);
            assert_eq!(slice.size(), &[2, 2]);
            let expected: Vec<u8> = (0..4).map(|j| (i * 4 + j) as u8).collect();
            assert_eq!(slice.scalar_slice::<u8>().unwrap(), expected.as_slice());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn origin_and_spacing_truncate_not_project() {
        let dir = tmp_dir("truncate");
        let paths: Vec<_> = (0..2).map(|i| dir.join(format!("s{i}.mha"))).collect();
        let mut image = Image::from_vec(&[2, 2, 2], vec![0u8; 8]).unwrap();
        image.set_spacing(&[2.0, 3.0, 5.0]).unwrap();
        image.set_origin(&[10.0, 20.0, 30.0]).unwrap();
        // A non-orthonormal top-left 2x2 submatrix, which a physical
        // re-projection would never produce — proving this is a literal
        // truncation, not a projection.
        image
            .set_direction(&[2.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap();

        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&paths);
        writer.execute(&image).unwrap();

        let slice0 = read_image(&paths[0]).unwrap();
        assert_eq!(slice0.spacing(), &[2.0, 3.0]);
        assert_eq!(slice0.origin(), &[10.0, 20.0]);
        assert_eq!(slice0.direction(), &[2.0, 0.0, 0.0, 3.0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn singular_2x2_submatrix_falls_back_to_identity() {
        let dir = tmp_dir("singular");
        let paths: Vec<_> = (0..2).map(|i| dir.join(format!("s{i}.mha"))).collect();
        let mut image = Image::from_vec(&[2, 2, 2], vec![0u8; 8]).unwrap();
        // Top-left 2x2 is [[1,0],[2,0]], determinant exactly 0.
        image
            .set_direction(&[1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();

        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&paths);
        writer.execute(&image).unwrap();

        let slice0 = read_image(&paths[0]).unwrap();
        assert_eq!(slice0.direction(), &[1.0, 0.0, 0.0, 1.0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dot_extension_dcm_is_rejected() {
        let image = Image::from_vec(&[2, 2, 1], vec![0u8; 4]).unwrap();
        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&["out.dcm"]);
        assert!(matches!(
            writer.execute(&image).unwrap_err(),
            IoError::SeriesWriterDicomRejected
        ));
    }

    #[test]
    fn dot_extension_dicom_is_rejected_case_insensitively() {
        let image = Image::from_vec(&[2, 2, 1], vec![0u8; 4]).unwrap();
        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&["out.DICOM"]);
        assert!(matches!(
            writer.execute(&image).unwrap_err(),
            IoError::SeriesWriterDicomRejected
        ));
    }

    /// `fn.substr(fn.find_last_of(".") + 1)` with no `.` anywhere wraps
    /// (`std::string::npos + 1 == 0`) to the *whole path string*
    /// (sitkImageSeriesWriter.cxx:227-234): a bare, dot-less file named
    /// literally `dcm` is rejected even though it has no extension at all.
    #[test]
    fn a_bare_dotless_filename_named_dcm_is_rejected_via_the_wraparound_quirk() {
        let image = Image::from_vec(&[2, 2, 1], vec![0u8; 4]).unwrap();
        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&["dcm"]);
        assert!(matches!(
            writer.execute(&image).unwrap_err(),
            IoError::SeriesWriterDicomRejected
        ));
    }

    #[test]
    fn non_3d_image_is_rejected_with_the_dispatch_message() {
        let image = Image::from_vec(&[2, 2], vec![0u8; 4]).unwrap();
        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&["out.mha"]);
        match writer.execute(&image).unwrap_err() {
            IoError::SeriesWriterUnsupportedDimension {
                pixel_type,
                dimension,
            } => {
                assert_eq!(pixel_type, PixelId::UInt8.as_str());
                assert_eq!(dimension, 2);
            }
            other => panic!("expected SeriesWriterUnsupportedDimension, got {other:?}"),
        }
    }

    #[test]
    fn empty_file_names_is_rejected() {
        let image = Image::from_vec(&[2, 2, 1], vec![0u8; 4]).unwrap();
        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names::<&str>(&[]);
        assert!(matches!(
            writer.execute(&image).unwrap_err(),
            IoError::EmptySeriesWriterFileNames
        ));
    }

    #[test]
    fn filename_count_mismatch_is_a_hard_error() {
        let dir = tmp_dir("count_mismatch");
        let paths: Vec<_> = (0..2).map(|i| dir.join(format!("s{i}.mha"))).collect();
        let image = Image::from_vec(&[2, 2, 3], vec![0u8; 12]).unwrap();

        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&paths);
        match writer.execute(&image).unwrap_err() {
            IoError::SeriesFileNameCountMismatch { actual, expected } => {
                assert_eq!(actual, 2);
                assert_eq!(expected, 3);
            }
            other => panic!("expected SeriesFileNameCountMismatch, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn per_slice_metadata_round_trips_through_hdf5() {
        let dir = tmp_dir("metadata_hdf5");
        let paths: Vec<_> = (0..2).map(|i| dir.join(format!("s{i}.h5"))).collect();
        let mut image = Image::from_vec(&[2, 2, 2], vec![0u8; 8]).unwrap();
        image.set_spacing(&[1.5, 2.5, 4.0]).unwrap();
        image.set_origin(&[10.0, 20.0, 30.0]).unwrap();

        let mut writer = ImageSeriesWriter::new();
        writer.set_file_names(&paths);
        writer.execute(&image).unwrap();

        let slice0 = read_image(&paths[0]).unwrap();
        assert_eq!(slice0.meta_data("ITK_Origin"), Some("10 20 30"));
        assert_eq!(slice0.meta_data("ITK_Spacing"), Some("1.5 2.5 4"));
        assert_eq!(slice0.meta_data("ITK_NumberOfDimensions"), Some("3"));
        assert_eq!(
            slice0.meta_data("ITK_ZDirection"),
            Some("1 0 0 0 1 0 0 0 1")
        );

        let slice1 = read_image(&paths[1]).unwrap();
        assert_eq!(slice1.meta_data("ITK_Origin"), Some("10 20 34"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_image_series_is_execute_with_the_defaults() {
        let dir = tmp_dir("free_function");
        let paths: Vec<_> = (0..2).map(|i| dir.join(format!("s{i}.mha"))).collect();
        let image = Image::from_vec(&[2, 2, 2], vec![0u8, 1, 2, 3, 4, 5, 6, 7]).unwrap();

        write_image_series(&image, &paths, false, -1).unwrap();

        let slice1 = read_image(&paths[1]).unwrap();
        assert_eq!(slice1.scalar_slice::<u8>().unwrap(), &[4, 5, 6, 7]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
