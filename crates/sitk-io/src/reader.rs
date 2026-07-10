//! [`ImageFileReader`] — SimpleITK's `itk::simple::ImageFileReader`
//! (sitkImageFileReader.h:95-219, sitkImageFileReader.cxx).

use std::path::{Path, PathBuf};

use sitk_core::{Image, PixelBuffer, matrix};

use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, reader_for};

/// Read an image file, optionally extracting a sub-region, and expose the
/// file's header information without loading pixels.
///
/// ```no_run
/// # use sitk_io::ImageFileReader;
/// let mut reader = ImageFileReader::new();
/// reader.set_file_name("volume.mha");
///
/// // Size, pixel type and meta-data keys, with no pixel data read:
/// let info = reader.read_image_information()?;
/// assert_eq!(info.dimension, 3);
///
/// // A 2-D slice out of the 3-D file: the zero-size axis collapses.
/// reader.set_extract_size(&[0, 0, 0]);
/// # Ok::<(), sitk_io::IoError>(())
/// ```
#[derive(Clone, Debug, Default)]
pub struct ImageFileReader {
    file_name: PathBuf,
    extract_size: Vec<usize>,
    extract_index: Vec<usize>,
    information: Option<ImageInformation>,
}

impl ImageFileReader {
    /// A reader with no file name and no extraction region.
    pub fn new() -> Self {
        Self::default()
    }

    /// `SetFileName`.
    pub fn set_file_name<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.file_name = path.as_ref().to_path_buf();
        self
    }

    /// `GetFileName`.
    pub fn file_name(&self) -> &Path {
        &self.file_name
    }

    /// Size of the region to extract from the file, per file axis.
    ///
    /// Empty (the default) loads the whole image. Otherwise the output image
    /// has one axis per **non-zero** entry: a `0` collapses that axis to the
    /// single slice at [`ImageFileReader::set_extract_index`]'s value, so
    /// `[10, 20, 0]` reads a 2-D `10x20` slice out of a 3-D file. Axes past the
    /// end of `size` are collapsed too. See [`ImageFileReader::execute`] for
    /// the two geometry rules this selects between.
    ///
    /// `SetExtractSize` (sitkImageFileReader.h:172-197).
    pub fn set_extract_size(&mut self, size: &[usize]) -> &mut Self {
        self.extract_size = size.to_vec();
        self
    }

    /// `GetExtractSize`.
    pub fn extract_size(&self) -> &[usize] {
        &self.extract_size
    }

    /// Starting index of the region to extract. Missing axes are `0`.
    ///
    /// `SetExtractIndex` (sitkImageFileReader.h:200-211). Upstream's index is
    /// `std::vector<int>`; a negative entry can only ever fail the "region is
    /// inside the file's region" check, so this port takes `usize` and makes
    /// that state unrepresentable.
    pub fn set_extract_index(&mut self, index: &[usize]) -> &mut Self {
        self.extract_index = index.to_vec();
        self
    }

    /// `GetExtractIndex`.
    pub fn extract_index(&self) -> &[usize] {
        &self.extract_index
    }

    /// Read the file's header — pixel type, dimension, component count,
    /// geometry, and the meta-data dictionary — without loading pixel data.
    ///
    /// `ReadImageInformation` (sitkImageFileReader.h:107-113,
    /// sitkImageFileReader.cxx:235-241). The result is cached and also
    /// available from [`ImageFileReader::information`] after an
    /// [`ImageFileReader::execute`].
    ///
    /// The reported `number_of_components` is the *file's*, so a MetaImage
    /// holding complex samples reports `2` here while the loaded [`Image`]
    /// reports one component per pixel.
    pub fn read_image_information(&mut self) -> Result<&ImageInformation> {
        let io = reader_for(&self.file_name)?;
        self.information = Some(io.read_information(&self.file_name)?);
        Ok(self
            .information
            .as_ref()
            .expect("information was just stored"))
    }

    /// The information from the last [`ImageFileReader::read_image_information`]
    /// or [`ImageFileReader::execute`], if either has run.
    ///
    /// Upstream's accessors read default-constructed state before the first
    /// call — `GetPixelID()` is `sitkUnknown` and `GetSize()` is empty. This
    /// port has no `sitkUnknown`, so "not read yet" is `None`.
    pub fn information(&self) -> Option<&ImageInformation> {
        self.information.as_ref()
    }

    /// Read the image, applying the extraction region if one is set.
    ///
    /// `Execute` (sitkImageFileReader.cxx:287-342). The file's own dimension
    /// must be at least 2; upstream additionally caps it at
    /// `SITK_IO_INPUT_MAX_DIMENSION` (5), a ceiling this port does not impose
    /// anywhere. When an extraction region is set, the output dimension — the
    /// number of non-zero entries in [`ImageFileReader::set_extract_size`] —
    /// must also be at least 2.
    ///
    /// # Which geometry the extracted image gets
    ///
    /// Upstream splits into two pipelines on the extract size's *length*
    /// (sitkImageFileReader.cxx:362-389) that disagree on geometry: reading
    /// `[10, 20]` versus `[10, 20, 0]` from the same oblique 3-D file gives an
    /// identity direction and a dropped `extract_index[2]` in the first case,
    /// but the file direction's submatrix and an honoured `extract_index[2]` in
    /// the second — the same requested slice, two different directions and two
    /// different pixel sets. Only the first matches the header's documented
    /// contract, "when the dimension of the image is reduced, the direction
    /// cosine matrix will be set to the identity ... the matrix from the file
    /// can still be obtained by `GetDirection`" (sitkImageFileReader.h:176-186).
    /// Ledger §3.27; that divergence is **not reproduced here**.
    ///
    /// This port runs a single pipeline that always honours the documented
    /// contract, equivalent to ITK's `SetDirectionCollapseToIdentity` rather
    /// than `…ToSubmatrix`:
    ///
    /// * The output direction is the **identity** whenever the extraction
    ///   reduces the dimension (a zero-size axis, or a short `extract_size`
    ///   whose missing trailing entries are taken as `0`); a pure crop that
    ///   keeps every axis keeps the file direction. A singular file direction
    ///   is therefore never an error — the collapse no longer inverts a
    ///   submatrix.
    /// * Every `extract_index` entry applies, on any axis, collapsed or not
    ///   (`GetExtractIndex`'s "missing dimensions are treated the same as 0",
    ///   sitkImageFileReader.h:210-213). So `[10, 20]` and `[10, 20, 0]` select
    ///   the same slice and return the same image.
    ///
    /// The origin comes from `ExtractImageFilter::GenerateOutputInformation`
    /// plus SimpleITK's `FixNonZeroIndex` (itkExtractImageFilter.hxx:156-180,
    /// sitkImageFileReader.cxx:39-67): the retained axes' origin shifted by the
    /// retained axes' own index, through the output (identity, when reduced)
    /// direction and spacing. A collapsed axis's index selects its slice but
    /// never shifts the origin — but note this reader has already discarded the
    /// file's oblique frame for a reduced dimension (identity collapse, §3.27),
    /// so there is no oblique world frame here for that shift to be "correct"
    /// in; the collapsed-axis origin drop is a consequence of the §3.27
    /// identity contract, not the §2.75 quirk. The §2.75 fix — taking the
    /// retained physical components of the full `TransformIndexToPhysicalPoint`
    /// so an oblique slice `k` lands at its true world corner — lives in the
    /// direction-preserving `ExtractImageFilter` path
    /// (`sitk_filters::geometry::extract`), which keeps the oblique submatrix
    /// and so has a coherent frame to be correct in.
    pub fn execute(&mut self) -> Result<Image> {
        let io = reader_for(&self.file_name)?;
        let info = io.read_information(&self.file_name)?;

        if info.dimension < 2 {
            return Err(IoError::UnsupportedImageDimension(info.dimension));
        }
        let out_dim = self.extract_size.iter().filter(|&&s| s != 0).count();
        if !self.extract_size.is_empty() && out_dim < 2 {
            return Err(IoError::ExtractOutputDimension(out_dim));
        }

        let image = io.read(&self.file_name)?;
        self.information = Some(info);

        if self.extract_size.is_empty() {
            return Ok(image);
        }
        self.execute_extract(&image, out_dim)
    }

    fn execute_extract(&self, image: &Image, out_dim: usize) -> Result<Image> {
        let file_dim = image.dimension();

        // One pipeline: read at the file's own dimension (padded up to the
        // extract size's length), so `[10, 20]` and `[10, 20, 0]` describe the
        // same region — the trailing entry missing from `extract_size` is `0`,
        // per the header. Ledger §3.27.
        let internal_dim = file_dim.max(self.extract_size.len());

        // itk::ImageFileReader pads the axes the file does not have with size
        // 1, spacing 1, origin 0 and an identity direction row
        // (itkImageFileReader.hxx:172-193).
        let at = |v: &[f64], i: usize, pad: f64| if i < file_dim { v[i] } else { pad };
        let internal_size: Vec<usize> = (0..internal_dim)
            .map(|i| if i < file_dim { image.size()[i] } else { 1 })
            .collect();
        let internal_spacing: Vec<f64> = (0..internal_dim)
            .map(|i| at(image.spacing(), i, 1.0))
            .collect();
        let internal_origin: Vec<f64> = (0..internal_dim)
            .map(|i| at(image.origin(), i, 0.0))
            .collect();
        let mut internal_direction = matrix::identity(internal_dim);
        for row in 0..file_dim {
            for col in 0..file_dim {
                internal_direction[row * internal_dim + col] =
                    image.direction()[row * file_dim + col];
            }
        }

        let sizes: Vec<usize> = (0..internal_dim)
            .map(|i| self.extract_size.get(i).copied().unwrap_or(0))
            .collect();
        let indices: Vec<usize> = (0..internal_dim)
            .map(|i| self.extract_index.get(i).copied().unwrap_or(0))
            .collect();

        // `largestRegion.IsInside(index) && largestRegion.IsInside(upperIndex)`,
        // where a zero-size axis's upper index is its own index
        // (sitkImageFileReader.cxx:430-444).
        for i in 0..internal_dim {
            let upper = indices[i] + sizes[i].saturating_sub(1);
            if indices[i] >= internal_size[i] || upper >= internal_size[i] {
                return Err(IoError::ExtractRegionOutOfBounds {
                    index: indices,
                    size: sizes,
                    file_size: internal_size,
                });
            }
        }

        let retained: Vec<usize> = (0..internal_dim).filter(|&i| sizes[i] != 0).collect();
        debug_assert_eq!(retained.len(), out_dim);

        let out_size: Vec<usize> = retained.iter().map(|&i| sizes[i]).collect();
        let out_spacing: Vec<f64> = retained.iter().map(|&i| internal_spacing[i]).collect();
        // The documented contract: a reduced dimension collapses the direction
        // to the identity (`SetDirectionCollapseToIdentity`); a pure crop that
        // keeps every axis keeps the file direction. Ledger §3.27.
        let out_direction = if out_dim < internal_dim {
            matrix::identity(out_dim)
        } else {
            let mut m = vec![0.0; out_dim * out_dim];
            for (a, &i) in retained.iter().enumerate() {
                for (b, &j) in retained.iter().enumerate() {
                    m[a * out_dim + b] = internal_direction[i * internal_dim + j];
                }
            }
            m
        };

        // FixNonZeroIndex: origin += out_direction * (out_spacing .* index).
        let scaled: Vec<f64> = retained
            .iter()
            .enumerate()
            .map(|(a, &i)| indices[i] as f64 * out_spacing[a])
            .collect();
        let rotated = matrix::mat_vec(&out_direction, &scaled, out_dim);
        let out_origin: Vec<f64> = retained
            .iter()
            .enumerate()
            .map(|(a, &i)| internal_origin[i] + rotated[a])
            .collect();

        let buffer = gather(image, &sizes, &indices, &retained, &out_size)?;
        let out = if image.buffer_stride() == 1 {
            Image::from_parts(buffer, out_size, out_spacing, out_origin, out_direction)
        } else {
            Image::from_parts_vector(
                buffer,
                image.buffer_stride(),
                out_size,
                out_spacing,
                out_origin,
                out_direction,
            )
        }?;
        Ok(copy_metadata(image, out))
    }
}

/// `extractor->GetOutput()->SetMetaDataDictionary(itkImage->GetMetaDataDictionary())`
/// (sitkImageFileReader.cxx:453).
fn copy_metadata(from: &Image, mut to: Image) -> Image {
    for key in from.meta_data_keys() {
        let value = from
            .meta_data(key)
            .expect("key came from meta_data_keys")
            .to_string();
        to.set_meta_data(key, &value);
    }
    to
}

/// Copy the extraction region's pixels out of `image`'s interleaved buffer.
///
/// `sizes` / `indices` are per *internal* axis; `retained` names the internal
/// axes that survive, in order. Internal axes at or past the file's dimension
/// contribute nothing (their only legal coordinate is `0`), which is exactly
/// how the file's trailing axes are read at index `0` when the extract size is
/// shorter than the file's dimension.
fn gather(
    image: &Image,
    sizes: &[usize],
    indices: &[usize],
    retained: &[usize],
    out_size: &[usize],
) -> Result<PixelBuffer> {
    if image.pixel_id().is_complex() {
        // MetaIO cannot represent a complex element type, so no ImageIo in this
        // crate produces one; `from_parts_vector` would silently relabel it.
        return Err(IoError::Unsupported(
            "extracting a region from a complex image".into(),
        ));
    }
    let file_dim = image.dimension();
    let file_size = image.size();
    let stride = image.buffer_stride();

    let mut file_strides = vec![1usize; file_dim];
    for d in 1..file_dim {
        file_strides[d] = file_strides[d - 1] * file_size[d - 1];
    }

    // The collapsed axes' fixed contribution to the input offset.
    let mut base = 0usize;
    for (d, &stride_d) in file_strides.iter().enumerate() {
        if sizes.get(d).copied().unwrap_or(0) == 0 {
            base += indices.get(d).copied().unwrap_or(0) * stride_d;
        }
    }

    let out_count: usize = out_size.iter().product();
    let mut picks = Vec::with_capacity(out_count);
    let mut coord = vec![0usize; retained.len()];
    for _ in 0..out_count {
        let mut flat = base;
        for (a, &i) in retained.iter().enumerate() {
            if i < file_dim {
                flat += (indices[i] + coord[a]) * file_strides[i];
            }
        }
        picks.push(flat);

        for a in 0..retained.len() {
            coord[a] += 1;
            if coord[a] < out_size[a] {
                break;
            }
            coord[a] = 0;
        }
    }

    macro_rules! pick {
        ($v:expr, $variant:ident) => {
            PixelBuffer::$variant(
                picks
                    .iter()
                    .flat_map(|&p| $v[p * stride..(p + 1) * stride].iter().copied())
                    .collect(),
            )
        };
    }
    Ok(match image.buffer() {
        PixelBuffer::UInt8(v) => pick!(v, UInt8),
        PixelBuffer::Int8(v) => pick!(v, Int8),
        PixelBuffer::UInt16(v) => pick!(v, UInt16),
        PixelBuffer::Int16(v) => pick!(v, Int16),
        PixelBuffer::UInt32(v) => pick!(v, UInt32),
        PixelBuffer::Int32(v) => pick!(v, Int32),
        PixelBuffer::UInt64(v) => pick!(v, UInt64),
        PixelBuffer::Int64(v) => pick!(v, Int64),
        PixelBuffer::Float32(v) => pick!(v, Float32),
        PixelBuffer::Float64(v) => pick!(v, Float64),
    })
}
