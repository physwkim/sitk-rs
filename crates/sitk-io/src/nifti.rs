//! NIfTI-1 (`.nii`, `.hdr` + `.img`) reader and writer — `itk::NiftiImageIO`.
//!
//! This is a port of `itkNiftiImageIO.cxx` **plus** the parts of the vendored
//! `nifti1_io.c` that class relies on (`nifti_image_read`,
//! `nifti_convert_nhdr2nim`, `nifti_convert_nim2nhdr`,
//! `nifti_image_write_engine`, `nifti_quatern_to_mat44`,
//! `nifti_mat44_to_quatern`, `nifti_make_orthog_mat44`, `nifti_mat33_polar`,
//! `is_nifti_file`, `nifti_findhdrname`, `nifti_findimgname`,
//! `nifti_find_file_extension`). Behaviour is pinned to what those two files
//! do, not to the NIfTI-1 specification.
//!
//! # Coordinate systems: NIfTI is RAS+, ITK is LPS
//!
//! Both `qto_xyz` and `sto_xyz` map voxel indices to **RAS+** millimetres
//! (`+x` right, `+y` anterior, `+z` superior). ITK works in **LPS**. The
//! conversion `itkNiftiImageIO` applies is a sign flip on the first two rows
//! *only*, and it is spelled out twice:
//!
//! * read (`SetImageIOOrientationFromNIfTI`, itkNiftiImageIO.cxx:1854-1909):
//!   `origin[0] = -m[0][3]`, `origin[1] = -m[1][3]`, `origin[2] = +m[2][3]`;
//!   each direction column `d` takes `m[i][d]` with `i < 2` negated, then is
//!   normalised.
//! * write (`SetNIfTIOrientationFromImageIO`, :1968-2047): every direction
//!   column is negated whole (`dirx[i] = -GetDirection(0)[i]`), and then — only
//!   when the image has three or more dimensions — `dirx[2]`, `diry[2]` and
//!   `dirz[2]` are negated *again*, which restores the `+z` row. A 1-D or 2-D
//!   image has no third row to restore, and `dirz` is set to `(0,0,1)`
//!   outright. Origin: `m[0][3] = -origin[0]`, `m[1][3] = -origin[1]`,
//!   `m[2][3] = +origin[2]`.
//!
//! So the transform is `diag(-1,-1,1)` applied on the left, in both directions —
//! an involution, and the round trip is exact up to `float` rounding.
//!
//! # qform vs. sform
//!
//! `nifti_convert_nhdr2nim` (nifti1_io.c:3772-3860) already decides what the
//! two matrices *are*: `qform_code <= 0` (or a non-NIfTI header) leaves
//! `qto_xyz = diag(dx, dy, dz)` and forces `qform_code` to
//! `NIFTI_XFORM_UNKNOWN`; `sform_code <= 0` leaves `sto_xyz` **all zero** (the
//! struct is `calloc`-ed) and forces `sform_code` to `NIFTI_XFORM_UNKNOWN`.
//!
//! On top of that `SetImageIOOrientationFromNIfTI` picks one
//! (itkNiftiImageIO.cxx:1591-1850):
//!
//! 1. **both codes `UNKNOWN`** → origin `0`, direction identity, and return.
//!    (The Analyze-7.5 `analyze75_orient` branch is skipped under the default
//!    `Analyze75Flavor::AnalyzeITK4Warning`, so it is not ported — see
//!    [`the_mat`].)
//! 2. otherwise `prefer_sform_over_qform` starts as "the two matrices agree
//!    element by element", and is then refined: if the sform is an invertible
//!    affine whose 3×3 normalises to an orthonormal matrix, the sform wins when
//!    the qform is `UNKNOWN`, or when `sform_code == NIFTI_XFORM_SCANNER_ANAT`,
//!    or (for the remaining `sform_code` values 2/3/4 with a known qform) when
//!    an SVD comparison says the two are very similar.
//! 3. if the sform is *not* orthonormal and `ITK_NIFTI_SFORM_PERMISSIVE` is on,
//!    the sform's 3×3 is replaced by its polar factor `U·Vᵀ` and
//!    `ITK_sform_corrected` is reported as `YES`.
//! 4. failing all that, the qform is used if its code is known; otherwise the
//!    read throws.
//!
//! Because `SetNIfTIOrientationFromImageIO` writes `qform_code = sform_code =
//! NIFTI_XFORM_SCANNER_ANAT` unconditionally (:2077-2078), every file this
//! module writes takes branch 2's `sform_code == SCANNER_ANAT` arm on read.
//!
//! # `scl_slope` / `scl_inter`
//!
//! `ReadImageInformation` (:988-1017) forces slope `1` / intercept `0` for an
//! Analyze-7.5 file, and otherwise takes them from the header, mapping a slope
//! whose magnitude is below `DBL_EPSILON` to `1`. `MustRescale()` is then
//! "slope differs from 1, or intercept differs from 0". When it holds and the
//! on-disk component type is an **integer**, the type reported to the caller is
//! promoted to `float` — so `PixelId::Int16` on disk becomes
//! [`PixelId::Float32`] in memory. `float`/`double` on disk keep their type.
//! The rescale itself, `RescaleFunction` (:239-247), covers every component of
//! every voxel — `numElts * GetNumberOfComponents()` elements. Upstream bug
//! §1.50, fixed in this port, passed only `numElts` (the *voxel* count), so a
//! multi-component image (e.g. `COMPLEX64`, `2·numElts` floats) had its tail
//! left unrescaled.
//!
//! # Vector images and `intent_code`
//!
//! `NIFTI_INTENT_VECTOR` / `NIFTI_INTENT_DISPVECT` / `NIFTI_INTENT_SYMMATRIX`
//! put the component count in `dim[5]`, and ITK then derives the *spatial*
//! dimension from `dim[4]`/`dim[3]`/`dim[2]` alone (:786-805) — never from
//! `dim[0]`, and never from `dim[1]`. A scalar image instead uses `dim[0]`
//! with trailing `1`s trimmed while the index stays above 3 (:820-824).
//!
//! `NIFTI_INTENT_DISPVECT` additionally turns on RAS↔LPS *vector* conversion,
//! because `m_ConvertRASDisplacementVectors` defaults to `true`
//! (itkNiftiImageIO.h:289) — the first two components of every 3-vector are
//! negated on the way in and on the way out. Upstream's own guard names "3-component
//! vector or point" in its exception text but never actually checks the count
//! (:565-571 read, :2177-2183 write) before applying a hard-coded stride-3
//! walk; bug §1.51, fixed here on both read and write by rejecting a
//! `NIFTI_INTENT_DISPVECT` image whose component count is not 3.
//!
//! `NIFTI_INTENT_GENMATRIX` is rejected by ITK itself (:806-810).
//! `NIFTI_INTENT_SYMMATRIX` loads as `SYMMETRICSECONDRANKTENSOR`, which
//! SimpleITK's `GetPixelIDFromImageIO` cannot represent (it throws "Unknown
//! PixelType"); this port **implements** the read (ledger §3.32) as a vector
//! image of the tensor's unique matrix entries, reordered from NIfTI's
//! lower-triangular to ITK's upper-triangular component order
//! (`UpperToLowerOrder`, :83-119) so the vector matches what `itk::Image` would
//! hold. The component count must be a triangular number.
//!
//! # Compression
//!
//! `.nii.gz`, `.hdr.gz` and `.img.gz` are read and written. Compression is
//! decided by the file *name* and nothing else: `znzopen(path, mode,
//! nifti_is_gzfile(path))` (znzlib.c:48-82) gzips iff the name ends in `.gz`,
//! so `SetUseCompression` and `SetCompressionLevel` are both dead for this
//! format — writing always deflates at zlib's default level of 6, and writing a
//! `.nii` never compresses however the writer is configured (ledger §3.40).
//!
//! `znzread` on a gz file falls back to a transparent byte copy when the gzip
//! magic is missing, so a `.nii.gz` holding a plain header reads fine
//! (ledger §2.113). Conversely a gzip stream named `.nii` is *not* gunzipped.
//!
//! One `znzFile` is one gzip stream: a `.nii.gz` is a single stream over
//! header, extender and pixels, while `.hdr.gz`/`.img.gz` are two independent
//! streams (nifti1_io.c:5958-5971).
//!
//! # What this module does not do
//!
//! * `.nia` — the NIfTI ASCII single-file variant (`NIFTI_FTYPE_ASCII`)
//!   (ledger §4.62).
//! * `NiftiImageIO::Read`'s streaming sub-region path
//!   (`nifti_read_subregion_image`); [`read`] always loads the whole image, as
//!   [`crate::meta_image::read`] does.
//! * `SetLegacyAnalyze75Mode` / `SetUseLegacyModeForTwoFileWriting`: SimpleITK
//!   exposes neither, so only the compiled-in defaults
//!   (`Analyze75Flavor::AnalyzeITK4Warning`, two-file writing as NIfTI-1 `ni1`)
//!   are implemented. Reading an Analyze-7.5 header *is* supported, because
//!   `nifti_convert_nhdr2nim`'s `is_nifti` gate makes it nearly free.
//! * Big-endian **hosts**. Reading a big-endian *file* on a little-endian host
//!   works (the header carries `dim[0]` out of range, which is upstream's swap
//!   signal); writing always emits little-endian, as upstream does on a
//!   little-endian host (ledger §4.58).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sitk_core::{Complex, Image, PixelBuffer, PixelId};

use crate::compression::{
    ZLIB_DEFAULT_COMPRESSION_LEVEL, gunzip_transparent, gunzip_transparent_prefix, gzip_compress,
};
use crate::error::{IoError, Result};
use crate::image_io::{ImageInformation, ImageIo};
use crate::writer::WriteOptions;

// ---------------------------------------------------------------------------
// nifti1.h constants
// ---------------------------------------------------------------------------

const NIFTI_TYPE_UINT8: i16 = 2;
const NIFTI_TYPE_INT16: i16 = 4;
const NIFTI_TYPE_INT32: i16 = 8;
const NIFTI_TYPE_FLOAT32: i16 = 16;
const NIFTI_TYPE_COMPLEX64: i16 = 32;
const NIFTI_TYPE_FLOAT64: i16 = 64;
const NIFTI_TYPE_RGB24: i16 = 128;
const NIFTI_TYPE_INT8: i16 = 256;
const NIFTI_TYPE_UINT16: i16 = 512;
const NIFTI_TYPE_UINT32: i16 = 768;
const NIFTI_TYPE_INT64: i16 = 1024;
const NIFTI_TYPE_UINT64: i16 = 1280;
const NIFTI_TYPE_FLOAT128: i16 = 1536;
const NIFTI_TYPE_COMPLEX128: i16 = 1792;
const NIFTI_TYPE_COMPLEX256: i16 = 2048;
const NIFTI_TYPE_RGBA32: i16 = 2304;

const NIFTI_INTENT_GENMATRIX: i16 = 1004;
const NIFTI_INTENT_SYMMATRIX: i16 = 1005;
const NIFTI_INTENT_DISPVECT: i16 = 1006;
const NIFTI_INTENT_VECTOR: i16 = 1007;

const NIFTI_XFORM_UNKNOWN: i16 = 0;
const NIFTI_XFORM_SCANNER_ANAT: i16 = 1;
const NIFTI_XFORM_ALIGNED_ANAT: i16 = 2;
const NIFTI_XFORM_TALAIRACH: i16 = 3;
const NIFTI_XFORM_MNI_152: i16 = 4;

const NIFTI_UNITS_METER: u8 = 1;
const NIFTI_UNITS_MM: u8 = 2;
const NIFTI_UNITS_MICRON: u8 = 3;
const NIFTI_UNITS_SEC: u8 = 8;
const NIFTI_UNITS_MSEC: u8 = 16;
const NIFTI_UNITS_USEC: u8 = 24;

/// `NIFTI_FTYPE_ANALYZE` — an Analyze-7.5 header, no NIfTI magic.
const FTYPE_ANALYZE: i32 = 0;
/// `NIFTI_FTYPE_NIFTI1_1` — single `.nii` file, magic `n+1`.
const FTYPE_NIFTI1_1: i32 = 1;
/// `NIFTI_FTYPE_NIFTI1_2` — `.hdr` + `.img` pair, magic `ni1`.
const FTYPE_NIFTI1_2: i32 = 2;

/// `sizeof(struct nifti_1_header)`.
pub const HEADER_SIZE: usize = 348;

/// `nifti_set_iname_offset`'s single-file answer: `348 + 4` (the extender),
/// already 16-byte aligned (nifti1_io.c:5696-5705).
const SINGLE_FILE_VOX_OFFSET: i32 = 352;

// ---------------------------------------------------------------------------
// Small dense linear algebra: nifti's mat33/mat44, plus the SVD ITK needs
// ---------------------------------------------------------------------------

/// nifti's `mat44`: a row-major 4×4 of `float`.
type Mat44 = [[f32; 4]; 4];
/// nifti's `mat33`.
type Mat33 = [[f32; 3]; 3];

const ZERO44: Mat44 = [[0.0; 4]; 4];

/// `nifti_mat33_determ` (nifti1_io.c:1838-1848).
fn mat33_determ(r: &Mat33) -> f32 {
    let (r11, r12, r13) = (r[0][0] as f64, r[0][1] as f64, r[0][2] as f64);
    let (r21, r22, r23) = (r[1][0] as f64, r[1][1] as f64, r[1][2] as f64);
    let (r31, r32, r33) = (r[2][0] as f64, r[2][1] as f64, r[2][2] as f64);
    (r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23
        - r31 * r22 * r13) as f32
}

/// `nifti_mat33_rownorm` — the largest row's `L1` norm (nifti1_io.c:1853-1863).
fn mat33_rownorm(a: &Mat33) -> f32 {
    (0..3)
        .map(|i| (a[i][0].abs() + a[i][1].abs() + a[i][2].abs()) as f64)
        .fold(f64::NEG_INFINITY, f64::max) as f32
}

/// `nifti_mat33_colnorm` (nifti1_io.c:1868-1878).
fn mat33_colnorm(a: &Mat33) -> f32 {
    (0..3)
        .map(|j| (a[0][j].abs() + a[1][j].abs() + a[2][j].abs()) as f64)
        .fold(f64::NEG_INFINITY, f64::max) as f32
}

/// `nifti_mat33_inverse` (nifti1_io.c:1806-1833). A singular input yields an
/// all-zero matrix, as upstream's `deti = 0` fallthrough does.
fn mat33_inverse(r: &Mat33) -> Mat33 {
    let (r11, r12, r13) = (r[0][0] as f64, r[0][1] as f64, r[0][2] as f64);
    let (r21, r22, r23) = (r[1][0] as f64, r[1][1] as f64, r[1][2] as f64);
    let (r31, r32, r33) = (r[2][0] as f64, r[2][1] as f64, r[2][2] as f64);

    let mut deti =
        r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23
            - r31 * r22 * r13;
    if deti != 0.0 {
        deti = 1.0 / deti;
    }
    [
        [
            (deti * (r22 * r33 - r32 * r23)) as f32,
            (deti * (-r12 * r33 + r32 * r13)) as f32,
            (deti * (r12 * r23 - r22 * r13)) as f32,
        ],
        [
            (deti * (-r21 * r33 + r31 * r23)) as f32,
            (deti * (r11 * r33 - r31 * r13)) as f32,
            (deti * (-r11 * r23 + r21 * r13)) as f32,
        ],
        [
            (deti * (r21 * r32 - r31 * r22)) as f32,
            (deti * (-r11 * r32 + r31 * r12)) as f32,
            (deti * (r11 * r22 - r21 * r12)) as f32,
        ],
    ]
}

/// `nifti_mat33_polar` (nifti1_io.c:1902-1951): Higham's iteration for the
/// orthogonal factor of `A`'s polar decomposition. A singular `A` is perturbed
/// along the diagonal until its determinant is non-zero, exactly as upstream.
fn mat33_polar(a: &Mat33) -> Mat33 {
    let mut x = *a;
    let mut z = x;
    let mut dif = 1.0f32;

    let mut gam = mat33_determ(&x);
    while gam == 0.0 {
        gam = (0.00001 * (0.001 + mat33_rownorm(&x) as f64)) as f32;
        x[0][0] += gam;
        x[1][1] += gam;
        x[2][2] += gam;
        gam = mat33_determ(&x);
    }

    let mut k = 0;
    loop {
        let y = mat33_inverse(&x);
        let (gam, gmi) = if dif > 0.3 {
            let alp = ((mat33_rownorm(&x) as f64) * (mat33_colnorm(&x) as f64)).sqrt();
            let bet = ((mat33_rownorm(&y) as f64) * (mat33_colnorm(&y) as f64)).sqrt();
            let gam = (bet / alp).sqrt() as f32;
            (gam, (1.0 / gam as f64) as f32)
        } else {
            (1.0, 1.0)
        };
        // Note the transposed access into Y: upstream writes `Y.m[j][i]` where
        // the X term uses `X.m[i][j]`.
        for i in 0..3 {
            for j in 0..3 {
                z[i][j] =
                    (0.5 * (gam as f64 * x[i][j] as f64 + gmi as f64 * y[j][i] as f64)) as f32;
            }
        }
        dif = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (z[i][j] - x[i][j]).abs() as f64)
            .sum::<f64>() as f32;

        k += 1;
        if k > 100 || dif < 3.0e-6 {
            break;
        }
        x = z;
    }
    z
}

/// `nifti_make_orthog_mat44` (nifti1_io.c:1748-1801). Each row is normalised in
/// place — a zero row #1 becomes `ex`, a zero row #2 becomes `ey`, and a zero
/// row #3 becomes the cross product of the two rows already normalised — after
/// which the 3×3 is replaced by its polar factor.
fn make_orthog_mat44(rows: [[f32; 3]; 3]) -> Mat44 {
    let mut q = rows;
    let norm = |row: &[f32; 3]| -> f64 {
        (row[0] as f64).powi(2) + (row[1] as f64).powi(2) + (row[2] as f64).powi(2)
    };

    for (i, fallback) in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]]
        .into_iter()
        .enumerate()
    {
        let val = norm(&q[i]);
        if val > 0.0 {
            let s = (1.0 / val.sqrt()) as f32;
            q[i] = [q[i][0] * s, q[i][1] * s, q[i][2] * s];
        } else {
            q[i] = fallback;
        }
    }
    let val = norm(&q[2]);
    q[2] = if val > 0.0 {
        let s = (1.0 / val.sqrt()) as f32;
        [q[2][0] * s, q[2][1] * s, q[2][2] * s]
    } else {
        [
            q[0][1] * q[1][2] - q[0][2] * q[1][1],
            q[0][2] * q[1][0] - q[0][0] * q[1][2],
            q[0][0] * q[1][1] - q[0][1] * q[1][0],
        ]
    };

    let p = mat33_polar(&q);
    let mut out = ZERO44;
    for i in 0..3 {
        out[i][..3].copy_from_slice(&p[i]);
    }
    out[3][3] = 1.0;
    out
}

/// `mat44_transpose` (itkNiftiImageIO.cxx:1120-1133).
fn mat44_transpose(m: &Mat44) -> Mat44 {
    let mut out = ZERO44;
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = m[j][i];
        }
    }
    out
}

/// `nifti_quatern_to_mat44` (nifti1_io.c:1478-1523).
#[allow(clippy::too_many_arguments)]
fn quatern_to_mat44(
    qb: f32,
    qc: f32,
    qd: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    qfac: f32,
) -> Mat44 {
    let (mut b, mut c, mut d) = (qb as f64, qc as f64, qd as f64);
    let mut a = 1.0 - (b * b + c * c + d * d);
    if a < 1.0e-7 {
        a = 1.0 / (b * b + c * c + d * d).sqrt();
        b *= a;
        c *= a;
        d *= a;
        a = 0.0;
    } else {
        a = a.sqrt();
    }

    let xd = if dx > 0.0 { dx as f64 } else { 1.0 };
    let yd = if dy > 0.0 { dy as f64 } else { 1.0 };
    let mut zd = if dz > 0.0 { dz as f64 } else { 1.0 };
    if qfac < 0.0 {
        zd = -zd;
    }

    let mut r = ZERO44;
    r[0][0] = ((a * a + b * b - c * c - d * d) * xd) as f32;
    r[0][1] = (2.0 * (b * c - a * d) * yd) as f32;
    r[0][2] = (2.0 * (b * d + a * c) * zd) as f32;
    r[1][0] = (2.0 * (b * c + a * d) * xd) as f32;
    r[1][1] = ((a * a + c * c - b * b - d * d) * yd) as f32;
    r[1][2] = (2.0 * (c * d - a * b) * zd) as f32;
    r[2][0] = (2.0 * (b * d - a * c) * xd) as f32;
    r[2][1] = (2.0 * (c * d + a * b) * yd) as f32;
    r[2][2] = ((a * a + d * d - c * c - b * b) * zd) as f32;
    r[0][3] = qx;
    r[1][3] = qy;
    r[2][3] = qz;
    r[3][3] = 1.0;
    r
}

/// The quaternion and offset parameters `nifti_mat44_to_quatern` extracts
/// (nifti1_io.c:1549-1661). `dx`/`dy`/`dz` (the column lengths) are discarded
/// by the only caller, `SetNIfTIOrientationFromImageIO`, which passes `nullptr`.
struct Quatern {
    b: f32,
    c: f32,
    d: f32,
    x: f32,
    y: f32,
    z: f32,
    qfac: f32,
}

/// `nifti_mat44_to_quatern` (nifti1_io.c:1549-1661).
fn mat44_to_quatern(r: &Mat44) -> Quatern {
    let (qx, qy, qz) = (r[0][3], r[1][3], r[2][3]);

    let (mut r11, mut r12, mut r13) = (r[0][0] as f64, r[0][1] as f64, r[0][2] as f64);
    let (mut r21, mut r22, mut r23) = (r[1][0] as f64, r[1][1] as f64, r[1][2] as f64);
    let (mut r31, mut r32, mut r33) = (r[2][0] as f64, r[2][1] as f64, r[2][2] as f64);

    let mut xd = (r11 * r11 + r21 * r21 + r31 * r31).sqrt();
    let mut yd = (r12 * r12 + r22 * r22 + r32 * r32).sqrt();
    let mut zd = (r13 * r13 + r23 * r23 + r33 * r33).sqrt();

    if xd == 0.0 {
        r11 = 1.0;
        r21 = 0.0;
        r31 = 0.0;
        xd = 1.0;
    }
    if yd == 0.0 {
        r22 = 1.0;
        r12 = 0.0;
        r32 = 0.0;
        yd = 1.0;
    }
    if zd == 0.0 {
        r33 = 1.0;
        r13 = 0.0;
        r23 = 0.0;
        zd = 1.0;
    }

    r11 /= xd;
    r21 /= xd;
    r31 /= xd;
    r12 /= yd;
    r22 /= yd;
    r32 /= yd;
    r13 /= zd;
    r23 /= zd;
    r33 /= zd;

    let q: Mat33 = [
        [r11 as f32, r12 as f32, r13 as f32],
        [r21 as f32, r22 as f32, r23 as f32],
        [r31 as f32, r32 as f32, r33 as f32],
    ];
    let p = mat33_polar(&q);
    let (r11, r12, mut r13) = (p[0][0] as f64, p[0][1] as f64, p[0][2] as f64);
    let (r21, r22, mut r23) = (p[1][0] as f64, p[1][1] as f64, p[1][2] as f64);
    let (r31, r32, mut r33) = (p[2][0] as f64, p[2][1] as f64, p[2][2] as f64);

    let det =
        r11 * r22 * r33 - r11 * r32 * r23 - r21 * r12 * r33 + r21 * r32 * r13 + r31 * r12 * r23
            - r31 * r22 * r13;
    let qfac = if det > 0.0 {
        1.0f32
    } else {
        r13 = -r13;
        r23 = -r23;
        r33 = -r33;
        -1.0f32
    };

    let (b, c, d);
    let mut a = r11 + r22 + r33 + 1.0;
    if a > 0.5 {
        a = 0.5 * a.sqrt();
        b = 0.25 * (r32 - r23) / a;
        c = 0.25 * (r13 - r31) / a;
        d = 0.25 * (r21 - r12) / a;
    } else {
        let xd = 1.0 + r11 - (r22 + r33);
        let yd = 1.0 + r22 - (r11 + r33);
        let zd = 1.0 + r33 - (r11 + r22);
        let (mut bb, mut cc, mut dd);
        if xd > 1.0 {
            bb = 0.5 * xd.sqrt();
            cc = 0.25 * (r12 + r21) / bb;
            dd = 0.25 * (r13 + r31) / bb;
            a = 0.25 * (r32 - r23) / bb;
        } else if yd > 1.0 {
            cc = 0.5 * yd.sqrt();
            bb = 0.25 * (r12 + r21) / cc;
            dd = 0.25 * (r23 + r32) / cc;
            a = 0.25 * (r13 - r31) / cc;
        } else {
            dd = 0.5 * zd.sqrt();
            bb = 0.25 * (r13 + r31) / dd;
            cc = 0.25 * (r23 + r32) / dd;
            a = 0.25 * (r21 - r12) / dd;
        }
        if a < 0.0 {
            bb = -bb;
            cc = -cc;
            dd = -dd;
        }
        b = bb;
        c = cc;
        d = dd;
    }

    Quatern {
        b: b as f32,
        c: c as f32,
        d: d as f32,
        x: qx,
        y: qy,
        z: qz,
        qfac,
    }
}

/// One-sided Jacobi SVD of a small square matrix: `A = U · diag(W) · Vᵀ` with
/// `W` sorted descending and non-negative.
///
/// ITK reaches for `itk::Math::SVD` (an Eigen `JacobiSVD`) in three places on
/// the NIfTI read path — [`is_affine`]'s condition number,
/// [`sform_and_qform_are_very_similar`]'s comparison of the two matrices'
/// left-singular bases, and the `ITK_NIFTI_SFORM_PERMISSIVE` polar factor.
/// This is a dependency-free stand-in. `W` and `U·Vᵀ` are canonical, so the
/// condition-number and polar-factor uses agree with Eigen exactly (up to
/// rounding); `U` alone is only canonical up to the sign of each column, which
/// is why ledger §4.57 records that the "very similar" comparison could disagree
/// with Eigen's on inputs that are *not* near-equal — the case where both
/// answers are "not similar" anyway.
fn jacobi_svd<const N: usize>(a: [[f64; N]; N]) -> ([[f64; N]; N], [f64; N], [[f64; N]; N]) {
    let mut u = a;
    let mut v = [[0.0f64; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _ in 0..60 {
        let mut converged = true;
        for p in 0..N.saturating_sub(1) {
            for q in (p + 1)..N {
                let alpha: f64 = (0..N).map(|i| u[i][p] * u[i][p]).sum();
                let beta: f64 = (0..N).map(|i| u[i][q] * u[i][q]).sum();
                let gamma: f64 = (0..N).map(|i| u[i][p] * u[i][q]).sum();
                if gamma == 0.0 || gamma.abs() <= 1e-300 * (alpha * beta).sqrt() {
                    continue;
                }
                if gamma.abs() > f64::EPSILON * (alpha * beta).sqrt() {
                    converged = false;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..N {
                    let (up, uq) = (u[i][p], u[i][q]);
                    u[i][p] = c * up - s * uq;
                    u[i][q] = s * up + c * uq;
                    let (vp, vq) = (v[i][p], v[i][q]);
                    v[i][p] = c * vp - s * vq;
                    v[i][q] = s * vp + c * vq;
                }
            }
        }
        if converged {
            break;
        }
    }

    let mut w = [0.0f64; N];
    for j in 0..N {
        w[j] = (0..N).map(|i| u[i][j] * u[i][j]).sum::<f64>().sqrt();
        if w[j] > 0.0 {
            for row in u.iter_mut() {
                row[j] /= w[j];
            }
        }
    }

    // Selection sort, descending, permuting U's and V's columns with W.
    for i in 0..N {
        let mut best = i;
        for (j, &wj) in w.iter().enumerate().skip(i + 1) {
            if wj > w[best] {
                best = j;
            }
        }
        if best != i {
            w.swap(i, best);
            for row in u.iter_mut() {
                row.swap(i, best);
            }
            for row in v.iter_mut() {
                row.swap(i, best);
            }
        }
    }
    (u, w, v)
}

/// `IsAffine` (itkNiftiImageIO.cxx:1534-1583) — is `nifti_mat` an invertible
/// affine transform?
///
/// Upstream tests three things: that the bottom row is `(0,0,0,1)` to within
/// `float` epsilon; that the SVD condition number `σ_min/σ_max` exceeds
/// `double` epsilon; and that the SVD pseudo-inverse's top-left 3×3 and
/// translation agree, to `1e-2`, with the analytic affine inverse
/// `[[T⁻¹, −T⁻¹v], [0, 1]]`.
///
/// The third test is vacuous given the first two: for a matrix whose bottom row
/// *is* `(0,0,0,1)`, the true inverse **is** the analytic affine inverse, and
/// the pseudo-inverse of a well-conditioned matrix is its true inverse. Only
/// the first two are computed here; the divergence is bounded to matrices whose
/// condition number lands between `DBL_EPSILON` and vnl's own rank-truncation
/// tolerance, where upstream would zero a singular value and answer `false`
/// (ledger §4.55).
fn is_affine(nifti_mat: &Mat44) -> bool {
    let mut mat = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            mat[i][j] = nifti_mat[i][j] as f64;
        }
    }

    let mut bottom_row_error = (mat[3][3] - 1.0).abs();
    for x in mat[3].iter().take(3) {
        bottom_row_error += x.abs();
    }
    if bottom_row_error > f32::EPSILON as f64 {
        return false;
    }

    let (_, w, _) = jacobi_svd(mat);
    let condition = w[3] / w[0];
    condition > f64::EPSILON
}

/// `is_identity(tol)` — vnl's element-wise test (vnl_matrix_fixed.hxx:661-673).
fn is_identity<const N: usize>(m: &[[f32; N]; N], tol: f64) -> bool {
    for (i, row) in m.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            let dev = if i == j { (x - 1.0).abs() } else { x.abs() };
            if dev as f64 > tol {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// The 348-byte header
// ---------------------------------------------------------------------------

/// `struct nifti_1_header` (nifti1.h:150-206), field for field.
#[derive(Clone, Debug, PartialEq)]
struct RawHeader {
    sizeof_hdr: i32,
    dim_info: u8,
    dim: [i16; 8],
    intent_p1: f32,
    intent_p2: f32,
    intent_p3: f32,
    intent_code: i16,
    datatype: i16,
    bitpix: i16,
    slice_start: i16,
    pixdim: [f32; 8],
    vox_offset: f32,
    scl_slope: f32,
    scl_inter: f32,
    slice_end: i16,
    slice_code: u8,
    xyzt_units: u8,
    cal_max: f32,
    cal_min: f32,
    slice_duration: f32,
    toffset: f32,
    descrip: [u8; 80],
    aux_file: [u8; 24],
    qform_code: i16,
    sform_code: i16,
    quatern_b: f32,
    quatern_c: f32,
    quatern_d: f32,
    qoffset_x: f32,
    qoffset_y: f32,
    qoffset_z: f32,
    srow_x: [f32; 4],
    srow_y: [f32; 4],
    srow_z: [f32; 4],
    intent_name: [u8; 16],
    magic: [u8; 4],
}

impl Default for RawHeader {
    /// `memset(&nhdr, 0, sizeof(nhdr))` — every `nifti_convert_nim2nhdr` starts
    /// here (nifti1_io.c:5478).
    fn default() -> Self {
        RawHeader {
            sizeof_hdr: 0,
            dim_info: 0,
            dim: [0; 8],
            intent_p1: 0.0,
            intent_p2: 0.0,
            intent_p3: 0.0,
            intent_code: 0,
            datatype: 0,
            bitpix: 0,
            slice_start: 0,
            pixdim: [0.0; 8],
            vox_offset: 0.0,
            scl_slope: 0.0,
            scl_inter: 0.0,
            slice_end: 0,
            slice_code: 0,
            xyzt_units: 0,
            cal_max: 0.0,
            cal_min: 0.0,
            slice_duration: 0.0,
            toffset: 0.0,
            descrip: [0; 80],
            aux_file: [0; 24],
            qform_code: 0,
            sform_code: 0,
            quatern_b: 0.0,
            quatern_c: 0.0,
            quatern_d: 0.0,
            qoffset_x: 0.0,
            qoffset_y: 0.0,
            qoffset_z: 0.0,
            srow_x: [0.0; 4],
            srow_y: [0.0; 4],
            srow_z: [0.0; 4],
            intent_name: [0; 16],
            magic: [0; 4],
        }
    }
}

fn rd_i16(b: &[u8], off: usize, swap: bool) -> i16 {
    let a = [b[off], b[off + 1]];
    if swap {
        i16::from_be_bytes(a)
    } else {
        i16::from_le_bytes(a)
    }
}

fn rd_i32(b: &[u8], off: usize, swap: bool) -> i32 {
    let a = [b[off], b[off + 1], b[off + 2], b[off + 3]];
    if swap {
        i32::from_be_bytes(a)
    } else {
        i32::from_le_bytes(a)
    }
}

fn rd_f32(b: &[u8], off: usize, swap: bool) -> f32 {
    let a = [b[off], b[off + 1], b[off + 2], b[off + 3]];
    if swap {
        f32::from_be_bytes(a)
    } else {
        f32::from_le_bytes(a)
    }
}

/// `NIFTI_VERSION` (nifti1.h:1495-1499): the NIfTI version digit, or `0` for a
/// header with no NIfTI magic (an Analyze-7.5 header).
fn nifti_version(magic: &[u8; 4]) -> u8 {
    if magic[0] == b'n'
        && magic[3] == 0
        && (magic[1] == b'i' || magic[1] == b'+')
        && magic[2].is_ascii_digit()
        && magic[2] != b'0'
    {
        magic[2] - b'0'
    } else {
        0
    }
}

/// `NIFTI_ONEFILE` (nifti1.h:1506).
fn nifti_onefile(magic: &[u8; 4]) -> bool {
    magic[1] == b'+'
}

/// `need_nhdr_swap` (nifti1_io.c:4143-4176): `Some(swap)`, or `None` when
/// neither `dim[0]` nor `sizeof_hdr` makes sense either way round.
fn need_nhdr_swap(dim0: i16, hdrsize: i32) -> Option<bool> {
    if dim0 != 0 {
        if (1..=7).contains(&dim0) {
            return Some(false);
        }
        if (1..=7).contains(&dim0.swap_bytes()) {
            return Some(true);
        }
        return None;
    }
    if hdrsize == HEADER_SIZE as i32 {
        return Some(false);
    }
    if hdrsize.swap_bytes() == HEADER_SIZE as i32 {
        return Some(true);
    }
    None
}

impl RawHeader {
    /// Decode the 348 bytes, choosing the byte order the way
    /// `nifti_convert_nhdr2nim` does: from `dim[0]`, falling back to
    /// `sizeof_hdr`. Also returns the Analyze-7.5 `orient` byte, which lives at
    /// `qform_code`'s address and must be read *before* any swap
    /// (nifti1_io.c:3675-3688).
    fn parse(b: &[u8]) -> Result<(Self, bool)> {
        if b.len() < HEADER_SIZE {
            return Err(IoError::MalformedNiftiHeader(format!(
                "header is {} bytes, need {HEADER_SIZE}",
                b.len()
            )));
        }
        let dim0_native = i16::from_le_bytes([b[40], b[41]]);
        let hdrsize_native = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let swap = match need_nhdr_swap(dim0_native, hdrsize_native) {
            Some(s) => s,
            None if dim0_native != 0 => {
                return Err(IoError::MalformedNiftiHeader("bad dim[0]".into()));
            }
            None => return Err(IoError::MalformedNiftiHeader("bad sizeof_hdr".into())),
        };

        let mut dim = [0i16; 8];
        for (i, d) in dim.iter_mut().enumerate() {
            *d = rd_i16(b, 40 + 2 * i, swap);
        }
        let mut pixdim = [0f32; 8];
        for (i, p) in pixdim.iter_mut().enumerate() {
            *p = rd_f32(b, 76 + 4 * i, swap);
        }
        let mut srow = [[0f32; 4]; 3];
        for (r, row) in srow.iter_mut().enumerate() {
            for (c, x) in row.iter_mut().enumerate() {
                *x = rd_f32(b, 280 + 16 * r + 4 * c, swap);
            }
        }
        let mut descrip = [0u8; 80];
        descrip.copy_from_slice(&b[148..228]);
        let mut aux_file = [0u8; 24];
        aux_file.copy_from_slice(&b[228..252]);
        let mut intent_name = [0u8; 16];
        intent_name.copy_from_slice(&b[328..344]);
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&b[344..348]);

        Ok((
            RawHeader {
                sizeof_hdr: rd_i32(b, 0, swap),
                dim_info: b[39],
                dim,
                intent_p1: rd_f32(b, 56, swap),
                intent_p2: rd_f32(b, 60, swap),
                intent_p3: rd_f32(b, 64, swap),
                intent_code: rd_i16(b, 68, swap),
                datatype: rd_i16(b, 70, swap),
                bitpix: rd_i16(b, 72, swap),
                slice_start: rd_i16(b, 74, swap),
                pixdim,
                vox_offset: rd_f32(b, 108, swap),
                scl_slope: rd_f32(b, 112, swap),
                scl_inter: rd_f32(b, 116, swap),
                slice_end: rd_i16(b, 120, swap),
                slice_code: b[122],
                xyzt_units: b[123],
                cal_max: rd_f32(b, 124, swap),
                cal_min: rd_f32(b, 128, swap),
                slice_duration: rd_f32(b, 132, swap),
                toffset: rd_f32(b, 136, swap),
                descrip,
                aux_file,
                qform_code: rd_i16(b, 252, swap),
                sform_code: rd_i16(b, 254, swap),
                quatern_b: rd_f32(b, 256, swap),
                quatern_c: rd_f32(b, 260, swap),
                quatern_d: rd_f32(b, 264, swap),
                qoffset_x: rd_f32(b, 268, swap),
                qoffset_y: rd_f32(b, 272, swap),
                qoffset_z: rd_f32(b, 276, swap),
                srow_x: srow[0],
                srow_y: srow[1],
                srow_z: srow[2],
                intent_name,
                magic,
            },
            swap,
        ))
    }

    /// The little-endian on-disk image of the struct, which is what
    /// `znzwrite(&nhdr, 1, sizeof(nhdr), fp)` produces on a little-endian host
    /// (nifti1_io.c:5939). `data_type`, `db_name`, `extents`, `session_error`,
    /// `glmax` and `glmin` stay zero, as `nifti_convert_nim2nhdr` leaves them;
    /// `regular` is `'r'` ("for some stupid reason", :5484).
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..4].copy_from_slice(&self.sizeof_hdr.to_le_bytes());
        b[38] = b'r';
        b[39] = self.dim_info;
        for (i, d) in self.dim.iter().enumerate() {
            b[40 + 2 * i..42 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }
        b[56..60].copy_from_slice(&self.intent_p1.to_le_bytes());
        b[60..64].copy_from_slice(&self.intent_p2.to_le_bytes());
        b[64..68].copy_from_slice(&self.intent_p3.to_le_bytes());
        b[68..70].copy_from_slice(&self.intent_code.to_le_bytes());
        b[70..72].copy_from_slice(&self.datatype.to_le_bytes());
        b[72..74].copy_from_slice(&self.bitpix.to_le_bytes());
        b[74..76].copy_from_slice(&self.slice_start.to_le_bytes());
        for (i, p) in self.pixdim.iter().enumerate() {
            b[76 + 4 * i..80 + 4 * i].copy_from_slice(&p.to_le_bytes());
        }
        b[108..112].copy_from_slice(&self.vox_offset.to_le_bytes());
        b[112..116].copy_from_slice(&self.scl_slope.to_le_bytes());
        b[116..120].copy_from_slice(&self.scl_inter.to_le_bytes());
        b[120..122].copy_from_slice(&self.slice_end.to_le_bytes());
        b[122] = self.slice_code;
        b[123] = self.xyzt_units;
        b[124..128].copy_from_slice(&self.cal_max.to_le_bytes());
        b[128..132].copy_from_slice(&self.cal_min.to_le_bytes());
        b[132..136].copy_from_slice(&self.slice_duration.to_le_bytes());
        b[136..140].copy_from_slice(&self.toffset.to_le_bytes());
        b[148..228].copy_from_slice(&self.descrip);
        b[228..252].copy_from_slice(&self.aux_file);
        b[252..254].copy_from_slice(&self.qform_code.to_le_bytes());
        b[254..256].copy_from_slice(&self.sform_code.to_le_bytes());
        b[256..260].copy_from_slice(&self.quatern_b.to_le_bytes());
        b[260..264].copy_from_slice(&self.quatern_c.to_le_bytes());
        b[264..268].copy_from_slice(&self.quatern_d.to_le_bytes());
        b[268..272].copy_from_slice(&self.qoffset_x.to_le_bytes());
        b[272..276].copy_from_slice(&self.qoffset_y.to_le_bytes());
        b[276..280].copy_from_slice(&self.qoffset_z.to_le_bytes());
        for (r, row) in [self.srow_x, self.srow_y, self.srow_z].iter().enumerate() {
            for (c, x) in row.iter().enumerate() {
                b[280 + 16 * r + 4 * c..284 + 16 * r + 4 * c].copy_from_slice(&x.to_le_bytes());
            }
        }
        b[328..344].copy_from_slice(&self.intent_name);
        b[344..348].copy_from_slice(&self.magic);
        b
    }
}

/// `nifti_datatype_sizes` (nifti1_io.c:1426-1456): `(nbyper, swapsize)`;
/// `(0, 0)` for a datatype the library does not know.
fn datatype_sizes(datatype: i16) -> (usize, usize) {
    match datatype {
        NIFTI_TYPE_INT8 | NIFTI_TYPE_UINT8 => (1, 0),
        NIFTI_TYPE_INT16 | NIFTI_TYPE_UINT16 => (2, 2),
        NIFTI_TYPE_RGB24 => (3, 0),
        NIFTI_TYPE_RGBA32 => (4, 0),
        NIFTI_TYPE_INT32 | NIFTI_TYPE_UINT32 | NIFTI_TYPE_FLOAT32 => (4, 4),
        NIFTI_TYPE_COMPLEX64 => (8, 4),
        NIFTI_TYPE_FLOAT64 | NIFTI_TYPE_INT64 | NIFTI_TYPE_UINT64 => (8, 8),
        NIFTI_TYPE_FLOAT128 => (16, 16),
        NIFTI_TYPE_COMPLEX128 => (16, 8),
        NIFTI_TYPE_COMPLEX256 => (32, 16),
        _ => (0, 0),
    }
}

/// How many *components* one on-disk element of `datatype` holds: `2` for the
/// complex types, `3`/`4` for RGB/RGBA, `1` otherwise. Upstream never names
/// this; it is implied by `SetNumberOfComponents` in
/// `ReadImageInformation`'s datatype switch (itkNiftiImageIO.cxx:900-923).
fn components_per_element(datatype: i16) -> usize {
    match datatype {
        NIFTI_TYPE_COMPLEX64 | NIFTI_TYPE_COMPLEX128 => 2,
        NIFTI_TYPE_RGB24 => 3,
        NIFTI_TYPE_RGBA32 => 4,
        _ => 1,
    }
}

/// `FIXED_FLOAT(x)` (nifti1_io.h:565): non-finite becomes `0`.
///
/// Active because `<math.h>` is included before the `#ifdef isfinite` guard and
/// glibc defines `isfinite` as a macro — so this is what an ITK built on Linux
/// does. Ledger §2.
fn fixed_f32(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

// ---------------------------------------------------------------------------
// nifti_image: the struct nifti_convert_nhdr2nim builds
// ---------------------------------------------------------------------------

/// The subset of `nifti_image` that `itkNiftiImageIO` reads.
///
/// `qto_ijk` / `sto_ijk` (the `nifti_mat44_inverse`s) are not modelled: nothing
/// on ITK's read path consults them.
#[derive(Clone, Debug)]
struct NiftiImage {
    nifti_type: i32,
    ndim: usize,
    dim: [i64; 8],
    pixdim: [f32; 8],
    nvox: u64,
    datatype: i16,
    nbyper: usize,
    /// `true` when the file's byte order is not the host's. `nifti_image`'s
    /// `byteorder` field, reduced to the only thing `nifti_read_buffer` uses it
    /// for; the element-wise swap width comes from [`datatype_sizes`].
    swapped: bool,
    qform_code: i16,
    sform_code: i16,
    quatern_b: f32,
    quatern_c: f32,
    quatern_d: f32,
    qoffset_x: f32,
    qoffset_y: f32,
    qoffset_z: f32,
    qto_xyz: Mat44,
    sto_xyz: Mat44,
    scl_slope: f32,
    scl_inter: f32,
    intent_code: i16,
    intent_p1: f32,
    intent_p2: f32,
    intent_p3: f32,
    intent_name: String,
    toffset: f32,
    xyz_units: u8,
    time_units: u8,
    freq_dim: u8,
    phase_dim: u8,
    slice_dim: u8,
    slice_code: i32,
    slice_start: i32,
    slice_end: i32,
    slice_duration: f32,
    cal_min: f32,
    cal_max: f32,
    descrip: String,
    aux_file: String,
    iname_offset: i32,
    /// The file the pixel bytes live in — `.nii` itself, or the sibling `.img`.
    iname: PathBuf,
}

/// Trim a NUL-terminated fixed char array to a `String`, as `std::ostringstream
/// << char[N]` does.
fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// `nifti_convert_nhdr2nim` (nifti1_io.c:3644-3926), plus `nifti_set_filenames`
/// and `nifti_set_type_from_names` for the `iname` it derives.
fn convert_nhdr2nim(nhdr: &RawHeader, swap: bool, hdr_path: &Path) -> Result<NiftiImage> {
    let mut nhdr = nhdr.clone();

    if nhdr.datatype == 0 || nhdr.datatype == 1 {
        return Err(IoError::MalformedNiftiHeader("bad datatype".into()));
    }
    if nhdr.dim[1] <= 0 {
        return Err(IoError::MalformedNiftiHeader("bad dim[1]".into()));
    }

    let is_nifti = nifti_version(&nhdr.magic) != 0;
    let dim0 = nhdr.dim[0] as usize;

    for ii in 2..=dim0.min(7) {
        if nhdr.dim[ii] <= 0 {
            nhdr.dim[ii] = 1;
        }
    }
    for ii in (dim0 + 1)..=7 {
        if nhdr.dim[ii] != 1 && nhdr.dim[ii] != 0 {
            nhdr.dim[ii] = 1;
        }
    }
    for ii in 1..=dim0.min(7) {
        if nhdr.pixdim[ii] == 0.0 || !nhdr.pixdim[ii].is_finite() {
            nhdr.pixdim[ii] = 1.0;
        }
    }

    let is_onefile = is_nifti && nifti_onefile(&nhdr.magic);
    let mut nifti_type = if is_nifti {
        if is_onefile {
            FTYPE_NIFTI1_1
        } else {
            FTYPE_NIFTI1_2
        }
    } else {
        FTYPE_ANALYZE
    };

    let mut nvox: u64 = 1;
    for ii in 1..=dim0.min(7) {
        nvox = nvox
            .checked_mul(nhdr.dim[ii] as u64)
            .ok_or_else(|| IoError::MalformedNiftiHeader("voxel count overflows".into()))?;
    }

    let (nbyper, _swapsize) = datatype_sizes(nhdr.datatype);
    if nbyper == 0 {
        return Err(IoError::MalformedNiftiHeader("bad datatype".into()));
    }

    // qto_xyz. `nim->qfac` itself is never read again on the ITK path — only
    // the matrix it went into.
    let (qform_code, qto_xyz, quatern, qoffset);
    if !is_nifti || nhdr.qform_code <= 0 {
        let mut m = ZERO44;
        m[0][0] = nhdr.pixdim[1];
        m[1][1] = nhdr.pixdim[2];
        m[2][2] = nhdr.pixdim[3];
        m[3][3] = 1.0;
        qform_code = NIFTI_XFORM_UNKNOWN;
        qto_xyz = m;
        quatern = (0.0, 0.0, 0.0);
        qoffset = (0.0, 0.0, 0.0);
    } else {
        let qb = fixed_f32(nhdr.quatern_b);
        let qc = fixed_f32(nhdr.quatern_c);
        let qd = fixed_f32(nhdr.quatern_d);
        let qx = fixed_f32(nhdr.qoffset_x);
        let qy = fixed_f32(nhdr.qoffset_y);
        let qz = fixed_f32(nhdr.qoffset_z);
        let qfac = if nhdr.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        qto_xyz = quatern_to_mat44(
            qb,
            qc,
            qd,
            qx,
            qy,
            qz,
            nhdr.pixdim[1],
            nhdr.pixdim[2],
            nhdr.pixdim[3],
            qfac,
        );
        qform_code = nhdr.qform_code;
        quatern = (qb, qc, qd);
        qoffset = (qx, qy, qz);
    }

    // sto_xyz: left all-zero (calloc) when the sform is absent.
    let (sform_code, sto_xyz);
    if !is_nifti || nhdr.sform_code <= 0 {
        sform_code = NIFTI_XFORM_UNKNOWN;
        sto_xyz = ZERO44;
    } else {
        let mut m = ZERO44;
        m[0] = nhdr.srow_x;
        m[1] = nhdr.srow_y;
        m[2] = nhdr.srow_z;
        m[3][3] = 1.0;
        sform_code = nhdr.sform_code;
        sto_xyz = m;
    }

    let iname_offset = if is_onefile {
        (nhdr.vox_offset as i32).max(HEADER_SIZE as i32)
    } else {
        nhdr.vox_offset as i32
    };

    // nifti_set_filenames + nifti_set_type_from_names (nifti1_io.c:3088-3120,
    // 3424-3457). `nifti_image_read` passes the *found header path*, extension
    // and all, so `.hdr` yields the sibling `.img` and `.nii` yields itself. If
    // the two names coincide the type is forced to single-file, and a
    // single-file type with distinct names is forced to the pair type.
    let prefix = hdr_path.to_string_lossy().into_owned();
    let comp = is_gz(hdr_path);
    let fname = make_hdrname(&prefix, nifti_type, comp);
    let iname = make_imgname(&prefix, nifti_type, comp);
    if fname == iname {
        nifti_type = FTYPE_NIFTI1_1;
    } else if nifti_type == FTYPE_NIFTI1_1 {
        nifti_type = FTYPE_NIFTI1_2;
    }

    let mut dim = [0i64; 8];
    for (out, raw) in dim.iter_mut().zip(nhdr.dim.iter()) {
        *out = *raw as i64;
    }

    Ok(NiftiImage {
        nifti_type,
        ndim: dim0,
        dim,
        pixdim: nhdr.pixdim,
        nvox,
        datatype: nhdr.datatype,
        nbyper,
        swapped: swap,
        qform_code,
        sform_code,
        quatern_b: quatern.0,
        quatern_c: quatern.1,
        quatern_d: quatern.2,
        qoffset_x: qoffset.0,
        qoffset_y: qoffset.1,
        qoffset_z: qoffset.2,
        qto_xyz,
        sto_xyz,
        // The NIfTI-only fields stay zero for an Analyze header.
        scl_slope: if is_nifti {
            fixed_f32(nhdr.scl_slope)
        } else {
            0.0
        },
        scl_inter: if is_nifti {
            fixed_f32(nhdr.scl_inter)
        } else {
            0.0
        },
        intent_code: if is_nifti { nhdr.intent_code } else { 0 },
        intent_p1: if is_nifti {
            fixed_f32(nhdr.intent_p1)
        } else {
            0.0
        },
        intent_p2: if is_nifti {
            fixed_f32(nhdr.intent_p2)
        } else {
            0.0
        },
        intent_p3: if is_nifti {
            fixed_f32(nhdr.intent_p3)
        } else {
            0.0
        },
        intent_name: if is_nifti {
            cstr(&nhdr.intent_name)
        } else {
            String::new()
        },
        toffset: if is_nifti {
            fixed_f32(nhdr.toffset)
        } else {
            0.0
        },
        xyz_units: if is_nifti { nhdr.xyzt_units & 0x07 } else { 0 },
        time_units: if is_nifti { nhdr.xyzt_units & 0x38 } else { 0 },
        freq_dim: if is_nifti { nhdr.dim_info & 0x03 } else { 0 },
        phase_dim: if is_nifti {
            (nhdr.dim_info >> 2) & 0x03
        } else {
            0
        },
        slice_dim: if is_nifti {
            (nhdr.dim_info >> 4) & 0x03
        } else {
            0
        },
        slice_code: if is_nifti { nhdr.slice_code as i32 } else { 0 },
        slice_start: if is_nifti { nhdr.slice_start as i32 } else { 0 },
        slice_end: if is_nifti { nhdr.slice_end as i32 } else { 0 },
        slice_duration: if is_nifti {
            fixed_f32(nhdr.slice_duration)
        } else {
            0.0
        },
        cal_min: fixed_f32(nhdr.cal_min),
        cal_max: fixed_f32(nhdr.cal_max),
        descrip: cstr(&nhdr.descrip),
        aux_file: cstr(&nhdr.aux_file),
        iname_offset,
        iname,
    })
}

// ---------------------------------------------------------------------------
// Filenames
// ---------------------------------------------------------------------------

/// The extensions ITK's `NiftiImageIO` constructor advertises
/// (itkNiftiImageIO.cxx:180).
const SUPPORTED_EXTENSIONS: &[&str] = &[".nia", ".nii", ".nii.gz", ".hdr", ".img", ".img.gz"];

/// The four bare extensions `nifti_find_file_extension` knows
/// (nifti1_io.c:2595-2598).
const BARE_EXTENSIONS: [&str; 4] = [".nii", ".hdr", ".img", ".nia"];

fn is_mixed_case(s: &str) -> bool {
    s.chars().any(|c| c.is_ascii_lowercase()) && s.chars().any(|c| c.is_ascii_uppercase())
}

/// `nifti_find_file_extension` (nifti1_io.c:2590-2651), with `HAVE_ZLIB` on
/// (ITK builds the vendored library against its own zlib) and
/// `allow_upper_fext` at its default `1`, so `.NII` is accepted but `.Nii` is
/// not.
fn find_file_extension(name: &str) -> Option<&str> {
    if name.len() < 4 {
        return None;
    }
    let ext = &name[name.len() - 4..];
    if BARE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()) {
        return if is_mixed_case(ext) { None } else { Some(ext) };
    }
    if name.len() < 7 {
        return None;
    }
    let ext = &name[name.len() - 7..];
    let lower = ext.to_ascii_lowercase();
    if [".nii.gz", ".hdr.gz", ".img.gz"].contains(&lower.as_str()) {
        return if is_mixed_case(ext) { None } else { Some(ext) };
    }
    None
}

/// `nifti_validfilename` (nifti1_io.c:2556-2576): a name is valid unless it is
/// *nothing but* an extension. A name with no recognised extension is valid.
fn valid_filename(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    !matches!(find_file_extension(name), Some(ext) if ext.len() == name.len())
}

/// `nifti_is_complete_filename` (nifti1_io.c:2512-2536), which is
/// `NiftiImageIO::CanWriteFile` verbatim (itkNiftiImageIO.cxx:217-223).
fn is_complete_filename(name: &str) -> bool {
    match find_file_extension(name) {
        None => false,
        Some(ext) => ext.len() != name.len(),
    }
}

/// `nifti_makebasename` (nifti1_io.c:2687-2701): the name with any recognised
/// extension chopped off.
fn make_basename(path: &Path) -> String {
    let name = path.to_string_lossy().into_owned();
    match find_file_extension(&name) {
        Some(ext) => name[..name.len() - ext.len()].to_string(),
        None => name,
    }
}

/// Is `s` uppercase-with-no-lowercase (`is_uppercase`, nifti1_io.c:3295-3308)?
fn is_uppercase(s: &str) -> bool {
    !s.is_empty()
        && !s.chars().any(|c| c.is_ascii_lowercase())
        && s.chars().any(|c| c.is_ascii_uppercase())
}

fn cased(ext: &str, upper: bool) -> String {
    if upper {
        ext.to_ascii_uppercase()
    } else {
        ext.to_string()
    }
}

/// The extension `nifti_makehdrname` / `nifti_makeimgname` invent when `prefix`
/// carries none.
fn invented_extension(nifti_type: i32, header: bool) -> &'static str {
    match (nifti_type, header) {
        (FTYPE_NIFTI1_1, _) => ".nii",
        (3, _) => ".nia",
        (_, true) => ".hdr",
        (_, false) => ".img",
    }
}

/// `nifti_makehdrname` (nifti1_io.c:2946-2999) and `nifti_makeimgname`
/// (:3016-3069) — one function, since they differ only in which extension they
/// rewrite and which one they invent.
///
/// When `prefix` already ends in a recognised extension, the type argument is
/// ignored: `.img` becomes `.hdr` (or, for the image name, `.hdr` becomes
/// `.img`), and every other extension is left alone. `nifti_image_read` passes
/// the full header path here, so both names come out right; but
/// `WriteImageInformation` passes `nifti_makebasename(FName)`, which has no
/// extension, so the write path always takes the "make one up" arm.
///
/// The invented extension is always lowercase, so a write through an uppercase
/// name would land on a lowercase file. It never gets that far:
/// `WriteImageInformation` compares the extension case-sensitively and refuses
/// `IMG.NII` before reaching here (ledger §2.91).
///
/// `comp` is upstream's fourth argument: `comp && (!ext || !strstr(iname, ".gz"))`
/// appends `.gz` (nifti1_io.c:2984). `nifti_set_filenames` derives it from
/// `nifti_is_gzfile(prefix)`, so on read it is redundant — the prefix *is* the
/// found header path, whose `.gz` is already there. On write the prefix is
/// `nifti_makebasename(FName)`, which has no extension, so `comp` is the only
/// thing that puts the `.gz` back.
fn make_name(prefix: &str, nifti_type: i32, header: bool, comp: bool) -> PathBuf {
    let mut iname = prefix.to_string();
    // `extgz` is uppercased along with the other four when the extension is;
    // with no extension it stays lowercase, and upstream's `!ext` short-circuit
    // means the `strstr` never runs.
    let ext_gz = match find_file_extension(&iname) {
        Some(ext) => {
            let upper = is_uppercase(ext);
            let (from, to) = if header {
                (cased(".img", upper), cased(".hdr", upper))
            } else {
                (cased(".hdr", upper), cased(".img", upper))
            };
            if ext[..4] == from {
                let start = iname.len() - ext.len();
                iname.replace_range(start..start + 4, &to);
            }
            Some(cased(".gz", upper))
        }
        None => {
            iname.push_str(invented_extension(nifti_type, header));
            None
        }
    };
    if comp {
        match &ext_gz {
            Some(gz) if iname.contains(gz.as_str()) => {}
            Some(gz) => iname.push_str(gz),
            None => iname.push_str(".gz"),
        }
    }
    PathBuf::from(iname)
}

fn make_hdrname(prefix: &str, nifti_type: i32, comp: bool) -> PathBuf {
    make_name(prefix, nifti_type, true, comp)
}

fn make_imgname(prefix: &str, nifti_type: i32, comp: bool) -> PathBuf {
    make_name(prefix, nifti_type, false, comp)
}

/// `nifti_findhdrname` (nifti1_io.c:2746-2842): return `fname` itself if it
/// exists and is not an `.img`; otherwise probe `base + ext` for `ext` in
/// `.nii, .nii.gz, .hdr, .hdr.gz` — or, when the input *was* an `.img`, in
/// `.hdr, .hdr.gz, .nii, .nii.gz`.
fn find_hdr_name(path: &Path) -> Option<PathBuf> {
    let name = path.to_string_lossy().into_owned();
    if !valid_filename(&name) {
        return None;
    }
    let base = make_basename(path);
    let ext = find_file_extension(&name);
    let upper = ext.map(is_uppercase).unwrap_or(false);

    let mut hdr_first = false;
    if let Some(ext) = ext {
        if path.exists() {
            // `fileext_n_compare(ext, ".img", 4)` (nifti1_io.c:2770) looks at the
            // first four characters only, so `.img.gz` is an image name too and
            // sends the search to `.hdr`/`.hdr.gz` first. Upstream's compare
            // matches all-lowercase or all-uppercase; `find_file_extension` has
            // already rejected anything mixed, so ASCII-insensitive is the same
            // predicate over the reachable inputs.
            if !ext[..4].eq_ignore_ascii_case(".img") {
                return Some(path.to_path_buf());
            }
            hdr_first = true;
        }
    }

    let order: [&str; 2] = if hdr_first {
        [".hdr", ".nii"]
    } else {
        [".nii", ".hdr"]
    };
    for stem in order {
        for suffix in ["", ".gz"] {
            let candidate = PathBuf::from(format!(
                "{base}{}{}",
                cased(stem, upper),
                cased(suffix, upper)
            ));
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// `nifti_findimgname` (nifti1_io.c:2861-2929): `.nii` first for a single-file
/// type, `.img` first otherwise.
fn find_img_name(iname: &Path, nifti_type: i32) -> Option<PathBuf> {
    let name = iname.to_string_lossy().into_owned();
    if !valid_filename(&name) {
        return None;
    }
    let base = make_basename(iname);
    let upper = find_file_extension(&name)
        .map(is_uppercase)
        .unwrap_or(false);
    let order: [&str; 2] = if nifti_type == FTYPE_NIFTI1_1 {
        [".nii", ".img"]
    } else {
        [".img", ".nii"]
    };
    for stem in order {
        for suffix in ["", ".gz"] {
            let candidate = PathBuf::from(format!(
                "{base}{}{}",
                cased(stem, upper),
                cased(suffix, upper)
            ));
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// `nifti_is_gzfile` (nifti1_io.c:2656-2668): does the name end in `.gz`?
/// A pure name test — the file's content is never consulted.
fn is_gz(path: &Path) -> bool {
    path.to_string_lossy().to_ascii_lowercase().ends_with(".gz")
}

/// `is_nifti_file` (nifti1_io.c:3483-3529): `2` for a `.hdr`/`.img` NIfTI pair,
/// `1` for a single-file NIfTI, `0` for an Analyze-7.5 header, `-1` otherwise.
fn is_nifti_file(path: &Path) -> i32 {
    let Some(hdr) = find_hdr_name(path) else {
        return -1;
    };
    let Ok(bytes) = read_prefix(&hdr, HEADER_SIZE) else {
        return -1;
    };
    if bytes.len() < HEADER_SIZE {
        return -1;
    }
    let magic: [u8; 4] = bytes[344..348].try_into().expect("slice is 4 bytes");
    if nifti_version(&magic) != 0 {
        return if nifti_onefile(&magic) { 1 } else { 2 };
    }
    let sizeof_hdr = i32::from_le_bytes(bytes[0..4].try_into().expect("slice is 4 bytes"));
    if sizeof_hdr == HEADER_SIZE as i32 || sizeof_hdr.swap_bytes() == HEADER_SIZE as i32 {
        return 0;
    }
    -1
}

/// `znzread` of `n` bytes from `znzopen(path, "rb", nifti_is_gzfile(path))`.
///
/// Whether the file is gunzipped is decided by its *name*, never its content —
/// so a gzip stream called `.nii` is read as raw bytes, and a plain file called
/// `.nii.gz` is read transparently by zlib's `gz_look` (ledger §2.113).
fn read_prefix(path: &Path, n: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    if is_gz(path) {
        return gunzip_transparent_prefix(&std::fs::read(path)?, n);
    }
    let file = std::fs::File::open(path)?;
    let mut buf = Vec::with_capacity(n);
    file.take(n as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

/// The whole of `path` through the same `znzFile`, for the image data.
fn read_all(path: &Path) -> Result<Vec<u8>> {
    let raw = std::fs::read(path)?;
    if is_gz(path) {
        return gunzip_transparent(&raw);
    }
    Ok(raw)
}

// ---------------------------------------------------------------------------
// Orientation: SetImageIOOrientationFromNIfTI
// ---------------------------------------------------------------------------

/// `ITK_NIFTI_SFORM_PERMISSIVE` (itkNiftiImageIO.cxx:188-192). Upstream reads
/// the variable once, in the constructor; this reads it per call (ledger §4.60), because the
/// [`NiftiImageIo`] in the registry is a zero-sized singleton. The compiled-in
/// default `ITK_NIFTI_IO_SFORM_PERMISSIVE_DEFAULT` is `OFF` (ITK's
/// CMakeLists.txt:504-509), and its `if constexpr (0)` guard is dead.
///
/// Note that *any* value other than `NO`/`OFF`/`FALSE` — the empty string
/// included — turns permissive mode on, because `GetEnv` succeeds for an empty
/// variable.
fn sform_permissive() -> bool {
    match std::env::var("ITK_NIFTI_SFORM_PERMISSIVE") {
        Err(_) => false,
        Ok(v) => {
            let v = v.to_ascii_uppercase();
            v != "NO" && v != "OFF" && v != "FALSE"
        }
    }
}

fn mat44_3x3(m: &Mat44) -> Mat33 {
    [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]
}

fn mat33_to_f64(m: &Mat33) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[i][j] as f64;
        }
    }
    out
}

fn mat33_mul(a: &Mat33, b: &Mat33) -> Mat33 {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn mat33_transpose(a: &Mat33) -> Mat33 {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[j][i];
        }
    }
    c
}

/// `qform_sform_are_similar` (itkNiftiImageIO.cxx:1669-1692).
fn qform_sform_are_similar(nim: &NiftiImage) -> bool {
    let (s, q) = (&nim.sto_xyz, &nim.qto_xyz);
    for i in 0..3 {
        for j in 0..3 {
            if (s[i][j] - q[i][j]).abs() as f64 > 1e-5 {
                return false;
            }
        }
    }
    let col3: f64 = (0..4).map(|i| (s[i][3] - q[i][3]).abs() as f64).sum();
    if col3 > 1e-7 {
        return false;
    }
    let row3: f64 = (0..4).map(|j| (s[3][j] - q[3][j]).abs() as f64).sum();
    row3 <= 1e-7
}

/// `sform_decomposable_without_skew` (itkNiftiImageIO.cxx:1699-1740). Upstream
/// also emits a warning when the sform's column lengths disagree with
/// `pixdim`; a warning has no observable effect here, so only the return value
/// is ported.
fn sform_decomposable_without_skew(nim: &NiftiImage) -> bool {
    if !is_affine(&nim.sto_xyz) {
        return false;
    }
    let mut rotation = mat44_3x3(&nim.sto_xyz);
    for j in 0..3 {
        let norm = ((rotation[0][j] as f64).powi(2)
            + (rotation[1][j] as f64).powi(2)
            + (rotation[2][j] as f64).powi(2))
        .sqrt() as f32;
        if norm != 0.0 {
            for row in rotation.iter_mut() {
                row[j] /= norm;
            }
        }
    }
    let candidate = mat33_mul(&rotation, &mat33_transpose(&rotation));
    is_identity(&candidate, 1.0e-4)
}

/// `sform_and_qform_are_very_similar` (itkNiftiImageIO.cxx:1764-1790).
fn sform_and_qform_are_very_similar(nim: &NiftiImage) -> bool {
    let (su, sw, _) = jacobi_svd(mat33_to_f64(&mat44_3x3(&nim.sto_xyz)));
    let (qu, qw, _) = jacobi_svd(mat33_to_f64(&mat44_3x3(&nim.qto_xyz)));

    let mut candidate = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            candidate[i][j] = (0..3).map(|k| su[i][k] * qu[j][k]).sum::<f64>() as f32;
        }
    }
    let spacing_similar = (0..3).all(|i| (sw[i] - qw[i]).abs() <= 1.0e-4);
    let rotation_similar = is_identity(&candidate, 1.0e-4);
    let offsets_similar =
        (0..3).all(|i| (nim.sto_xyz[i][3] - nim.qto_xyz[i][3]).abs() as f64 <= 1.0e-4);
    let perspectives_similar =
        (0..4).all(|j| (nim.sto_xyz[3][j] - nim.qto_xyz[3][j]).abs() as f64 <= 1.0e-4);

    rotation_similar && offsets_similar && perspectives_similar && spacing_similar
}

/// The lambda that chooses `theMat` (itkNiftiImageIO.cxx:1662-1850). Returns
/// the matrix and `m_SFORM_Corrected`.
fn the_mat(nim: &NiftiImage) -> Result<(Mat44, bool)> {
    let mut prefer_sform_over_qform = qform_sform_are_similar(nim);

    if !prefer_sform_over_qform || nim.sform_code != NIFTI_XFORM_UNKNOWN {
        if sform_decomposable_without_skew(nim) {
            let qform_known = nim.qform_code != NIFTI_XFORM_UNKNOWN;
            let sform_known = nim.sform_code != NIFTI_XFORM_UNKNOWN;
            if (!qform_known && sform_known) || nim.sform_code == NIFTI_XFORM_SCANNER_ANAT {
                // Either there is no qform to fall back on, or the sform is
                // already expressed in the scanner's own frame. Upstream splits
                // these into two arms with the same body (:1752-1760).
                prefer_sform_over_qform = true;
            } else if qform_known && sform_known {
                prefer_sform_over_qform = sform_and_qform_are_very_similar(nim);
            }
        } else if sform_permissive() {
            let (u, w, v) = jacobi_svd(mat33_to_f64(&mat44_3x3(&nim.sto_xyz)));
            if w[2] > 1e-8 {
                let mut m = ZERO44;
                for i in 0..3 {
                    for j in 0..3 {
                        m[i][j] = (0..3).map(|k| u[i][k] * v[j][k]).sum::<f64>() as f32;
                    }
                }
                for (i, row) in m.iter_mut().enumerate().take(3) {
                    row[3] = nim.sto_xyz[i][3];
                }
                m[3] = nim.sto_xyz[3];
                return Ok((m, true));
            }
        }
    }

    if prefer_sform_over_qform {
        return Ok((nim.sto_xyz, false));
    }
    if nim.qform_code != NIFTI_XFORM_UNKNOWN {
        return Ok((nim.qto_xyz, false));
    }
    Err(IoError::MalformedNiftiHeader(
        "ITK only supports orthonormal direction cosines.  No orthonormal definition found!".into(),
    ))
}

/// `Normalize` (itkNiftiImageIO.cxx:1506-1524): scale to unit `L2` norm, or
/// leave an all-zero vector alone.
fn normalize(x: &mut [f64]) {
    let sum: f64 = x.iter().map(|v| v * v).sum();
    if sum == 0.0 {
        return;
    }
    let sum = sum.sqrt();
    for v in x.iter_mut() {
        *v /= sum;
    }
}

fn identity_direction(dims: usize) -> Vec<f64> {
    let mut m = vec![0.0; dims * dims];
    for i in 0..dims {
        m[i * dims + i] = 1.0;
    }
    m
}

/// `SetImageIOOrientationFromNIfTI` (itkNiftiImageIO.cxx:1586-1910).
///
/// Returns `(origin, direction, sform_corrected)`. `direction` is row-major,
/// which is SimpleITK's layout: `direction[j * dims + i]` is the `j`-th
/// component of the `i`-th axis, i.e. `GetDirection(i)[j]` in ITK's
/// `ImageIOBase` (itkImageFileReader.hxx:187).
fn image_io_orientation(
    nim: &NiftiImage,
    dims: usize,
    spacingscale: f64,
    timingscale: f64,
) -> Result<(Vec<f64>, Vec<f64>, bool)> {
    let mut origin = vec![0.0; dims];
    let mut direction = identity_direction(dims);

    if nim.qform_code == NIFTI_XFORM_UNKNOWN && nim.sform_code == NIFTI_XFORM_UNKNOWN {
        // The Analyze-7.5 `analyze75_orient` branch is gated on a
        // `LegacyAnalyze75Mode` other than `AnalyzeITK4`/`AnalyzeITK4Warning`
        // (:1603-1605), and `AnalyzeITK4Warning` is the compiled-in default
        // that SimpleITK cannot change — so it never runs. Origin stays zero
        // and the direction stays `ImageIOBase`'s identity.
        return Ok((origin, direction, false));
    }

    let (mat, corrected) = the_mat(nim)?;

    origin[0] = -(mat[0][3] as f64) * spacingscale;
    if dims > 1 {
        origin[1] = -(mat[1][3] as f64) * spacingscale;
    }
    if dims > 2 {
        origin[2] = mat[2][3] as f64 * spacingscale;
    }
    if dims > 3 {
        origin[3] = nim.toffset as f64 * timingscale;
    }

    let max_defined = dims.min(3);
    for d in 0..max_defined {
        let mut axis = vec![0.0f64; dims];
        for (i, a) in axis.iter_mut().enumerate().take(max_defined) {
            *a = mat[i][d] as f64;
            if i < 2 {
                *a *= -1.0;
            }
        }
        normalize(&mut axis);
        for (j, a) in axis.iter().enumerate() {
            direction[j * dims + d] = *a;
        }
    }
    Ok((origin, direction, corrected))
}

// ---------------------------------------------------------------------------
// ReadImageInformation
// ---------------------------------------------------------------------------

/// Which `itk::IOPixelEnum` `ReadImageInformation` settles on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PixelKind {
    Scalar,
    Complex,
    Rgb,
    Vector,
}

/// The full result of `NiftiImageIO::ReadImageInformation`.
struct Info {
    nim: NiftiImage,
    dims: usize,
    /// `GetNumberOfComponents()` — `2` for complex, `3`/`4` for RGB/RGBA,
    /// `dim[5]` for a vector image, `1` for a scalar.
    components: usize,
    kind: PixelKind,
    /// `m_ComponentType`, after the `MustRescale` promotion to `float`.
    component: PixelId,
    /// `m_OnDiskComponentType`.
    on_disk: PixelId,
    rescale_slope: f64,
    rescale_intercept: f64,
    convert_ras: bool,
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
    metadata: BTreeMap<String, String>,
}

impl Info {
    /// `MustRescale()` (itkNiftiImageIO.cxx:226-231).
    fn must_rescale(&self) -> bool {
        self.rescale_slope.abs() > f64::EPSILON
            && ((self.rescale_slope - 1.0).abs() > f64::EPSILON
                || self.rescale_intercept.abs() > f64::EPSILON)
    }

    /// `ImageReaderBase::GetPixelIDFromImageIO` (sitkImageReaderBase.cxx:
    /// 200-239): one component of a scalar/complex pixel type stays scalar;
    /// RGB/RGBA/VECTOR load as a vector image; a two-component COMPLEX loads as
    /// complex. `SYMMETRICSECONDRANKTENSOR` reaches the `else` and throws in
    /// SimpleITK — so [`read_info`] instead loads a `NIFTI_INTENT_SYMMATRIX`
    /// image as a `Vector`, whose components this branch maps with `vector_id`
    /// (ledger §3.32).
    fn pixel_id(&self) -> PixelId {
        match self.kind {
            PixelKind::Scalar => self.component,
            PixelKind::Complex => match self.component {
                PixelId::Float64 => PixelId::ComplexFloat64,
                _ => PixelId::ComplexFloat32,
            },
            PixelKind::Rgb | PixelKind::Vector => self.component.vector_id(),
        }
    }
}

/// `itk::NumberToString`'s shortest round-trip decimal. Rust's `Display` for
/// `f32`/`f64` is also shortest-round-trip, but never switches to exponential
/// notation, where double-conversion does outside `1e-6 .. 1e21` (ledger §4.63).
fn num_to_string<T: std::fmt::Display>(v: T) -> String {
    v.to_string()
}

/// `str_xform` (itkNiftiImageIO.cxx:60-77).
fn str_xform(code: i16) -> &'static str {
    match code {
        NIFTI_XFORM_SCANNER_ANAT => "NIFTI_XFORM_SCANNER_ANAT",
        NIFTI_XFORM_ALIGNED_ANAT => "NIFTI_XFORM_ALIGNED_ANAT",
        NIFTI_XFORM_TALAIRACH => "NIFTI_XFORM_TALAIRACH",
        NIFTI_XFORM_MNI_152 => "NIFTI_XFORM_MNI_152",
        _ => "NIFTI_XFORM_UNKNOWN",
    }
}

/// `str_xform2code` (itkNiftiImageIO.cxx:37-58).
fn str_xform2code(name: &str) -> i16 {
    match name {
        "NIFTI_XFORM_SCANNER_ANAT" => NIFTI_XFORM_SCANNER_ANAT,
        "NIFTI_XFORM_ALIGNED_ANAT" => NIFTI_XFORM_ALIGNED_ANAT,
        "NIFTI_XFORM_TALAIRACH" => NIFTI_XFORM_TALAIRACH,
        "NIFTI_XFORM_MNI_152" => NIFTI_XFORM_MNI_152,
        _ => NIFTI_XFORM_UNKNOWN,
    }
}

/// `SetImageIOMetadataFromNIfTI` (itkNiftiImageIO.cxx:624-754), followed by the
/// `ITK_FileNotes` entry `ReadImageInformation` adds afterwards (:1112).
///
/// `ITK_InputFilterName` is **not** here: `ReadImageInformation` encapsulates it
/// at :1102 and `SetImageIOMetadataFromNIfTI` then calls `thisDic.Clear()` at
/// :630, so a NIfTI image never carries it. Ledger §2.86.
///
/// `qfac` and `qto_xyz` are omitted: upstream stores them as `float` and
/// `Matrix<float,4,4>` rather than `std::string`, and SimpleITK stringifies
/// those through `MetaDataObject::Print` (a trailing newline, and a multi-line
/// block for the matrix). This port's dictionary is `String`-valued. Ledger §4.56.
fn build_metadata(nim: &NiftiImage, sform_corrected: bool) -> BTreeMap<String, String> {
    let mut md = BTreeMap::new();
    let mut put = |k: &str, v: String| {
        md.insert(k.to_string(), v);
    };

    put(
        "ITK_sform_corrected",
        if sform_corrected { "YES" } else { "NO" }.to_string(),
    );
    put("nifti_type", nim.nifti_type.to_string());
    let dim_info = (nim.freq_dim as i32 & 0x03)
        | ((nim.phase_dim as i32 & 0x03) << 2)
        | ((nim.slice_dim as i32 & 0x03) << 4);
    put("dim_info", dim_info.to_string());
    for idx in 0..8 {
        put(&format!("dim[{idx}]"), nim.dim[idx].to_string());
    }
    put("intent_p1", num_to_string(nim.intent_p1));
    put("intent_p2", num_to_string(nim.intent_p2));
    put("intent_p3", num_to_string(nim.intent_p3));
    put("intent_code", nim.intent_code.to_string());
    put("datatype", nim.datatype.to_string());
    put("bitpix", (8 * nim.nbyper).to_string());
    put("slice_start", nim.slice_start.to_string());
    for idx in 0..8 {
        put(&format!("pixdim[{idx}]"), num_to_string(nim.pixdim[idx]));
    }
    put("vox_offset", nim.iname_offset.to_string());
    put("scl_slope", num_to_string(nim.scl_slope));
    put("scl_inter", num_to_string(nim.scl_inter));
    put("slice_end", nim.slice_end.to_string());
    put("slice_code", nim.slice_code.to_string());
    let xyzt_units = (nim.xyz_units as i32 & 0x07) | (nim.time_units as i32 & 0x38);
    put("xyzt_units", xyzt_units.to_string());
    put("cal_max", num_to_string(nim.cal_max));
    put("cal_min", num_to_string(nim.cal_min));
    put("slice_duration", num_to_string(nim.slice_duration));
    put("toffset", num_to_string(nim.toffset));
    put("descrip", nim.descrip.clone());
    put("aux_file", nim.aux_file.clone());
    put("qform_code", nim.qform_code.to_string());
    put("qform_code_name", str_xform(nim.qform_code).to_string());
    put("sform_code", nim.sform_code.to_string());
    put("sform_code_name", str_xform(nim.sform_code).to_string());
    put("quatern_b", num_to_string(nim.quatern_b));
    put("quatern_c", num_to_string(nim.quatern_c));
    put("quatern_d", num_to_string(nim.quatern_d));
    put("qoffset_x", num_to_string(nim.qoffset_x));
    put("qoffset_y", num_to_string(nim.qoffset_y));
    put("qoffset_z", num_to_string(nim.qoffset_z));
    for (row, name) in ["srow_x", "srow_y", "srow_z"].iter().enumerate() {
        let s = (0..4)
            .map(|col| num_to_string(nim.sto_xyz[row][col]))
            .collect::<Vec<_>>()
            .join(" ");
        put(name, s);
    }
    put("intent_name", nim.intent_name.clone());
    put("ITK_FileNotes", nim.descrip.clone());
    md
}

/// Read the header of `path` — `nifti_image_read(fname, /*read_data=*/false)`
/// followed by `NiftiImageIO::ReadImageInformation`.
fn read_info(path: &Path) -> Result<Info> {
    let hdr_path = find_hdr_name(path).ok_or_else(|| {
        IoError::MalformedNiftiHeader(format!(
            "{} is not recognized as a NIFTI file",
            path.display()
        ))
    })?;
    let bytes = read_prefix(&hdr_path, HEADER_SIZE)?;
    let (raw, swap) = RawHeader::parse(&bytes)?;
    let nim = convert_nhdr2nim(&raw, swap, &hdr_path)?;

    // Dimension and component count from the intent code (:786-837).
    let intent_vectorish = matches!(
        nim.intent_code,
        NIFTI_INTENT_DISPVECT | NIFTI_INTENT_VECTOR | NIFTI_INTENT_SYMMATRIX
    );
    if nim.intent_code == NIFTI_INTENT_GENMATRIX {
        return Err(IoError::UnsupportedNiftiFeature(format!(
            "{} has an intent code of NIFTI_INTENT_GENMATRIX which is not yet implemented in ITK",
            path.display()
        )));
    }
    let (dims, mut components) = if intent_vectorish {
        let dims = if nim.dim[4] > 1 {
            4
        } else if nim.dim[3] > 1 {
            3
        } else if nim.dim[2] > 1 {
            2
        } else {
            1
        };
        (dims, nim.dim[5].max(0) as usize)
    } else {
        let mut realdim = nim.ndim;
        while realdim > 3 && nim.dim[realdim] == 1 {
            realdim -= 1;
        }
        (realdim, 1)
    };

    // The datatype switch (:842-926).
    let (mut component, mut kind) = match nim.datatype {
        NIFTI_TYPE_INT8 => (PixelId::Int8, PixelKind::Scalar),
        NIFTI_TYPE_UINT8 => (PixelId::UInt8, PixelKind::Scalar),
        NIFTI_TYPE_INT16 => (PixelId::Int16, PixelKind::Scalar),
        NIFTI_TYPE_UINT16 => (PixelId::UInt16, PixelKind::Scalar),
        NIFTI_TYPE_INT32 => (PixelId::Int32, PixelKind::Scalar),
        NIFTI_TYPE_UINT32 => (PixelId::UInt32, PixelKind::Scalar),
        NIFTI_TYPE_INT64 => (PixelId::Int64, PixelKind::Scalar),
        NIFTI_TYPE_UINT64 => (PixelId::UInt64, PixelKind::Scalar),
        NIFTI_TYPE_FLOAT32 => (PixelId::Float32, PixelKind::Scalar),
        NIFTI_TYPE_FLOAT64 => (PixelId::Float64, PixelKind::Scalar),
        NIFTI_TYPE_COMPLEX64 => {
            components = 2;
            (PixelId::Float32, PixelKind::Complex)
        }
        NIFTI_TYPE_COMPLEX128 => {
            components = 2;
            (PixelId::Float64, PixelKind::Complex)
        }
        NIFTI_TYPE_RGB24 => {
            components = 3;
            (PixelId::UInt8, PixelKind::Rgb)
        }
        NIFTI_TYPE_RGBA32 => {
            components = 4;
            (PixelId::UInt8, PixelKind::Rgb)
        }
        // ITK's `default: break` leaves `m_ComponentType` at
        // `UNKNOWNCOMPONENTTYPE`, on which SimpleITK's reader throws.
        other => return Err(IoError::UnsupportedNiftiDatatype(other)),
    };

    // The intent switch (:931-983).
    let mut convert_ras = false;
    match nim.intent_code {
        NIFTI_INTENT_SYMMATRIX => {
            // `itk::Image` reads this as `SYMMETRICSECONDRANKTENSOR`; SimpleITK's
            // `GetPixelIDFromImageIO` has no arm for it and throws "Unknown
            // PixelType". This port's `Image` CAN hold the data, so the tensor is
            // loaded as a vector image of its unique matrix entries (ledger §3.32).
            // The component count must be a triangular number T(d) = d(d+1)/2 for
            // the NIfTI-lower → ITK-upper reorder (`UpperToLowerOrder`) applied in
            // `read` to be a valid permutation.
            kind = PixelKind::Vector;
            let d = sym_mat_dim(components);
            if d * (d + 1) / 2 != components {
                return Err(IoError::UnsupportedNiftiFeature(format!(
                    "{}: NIFTI_INTENT_SYMMATRIX declares {components} components, which is not a \
                     symmetric-matrix triangular count d·(d+1)/2",
                    path.display()
                )));
            }
        }
        NIFTI_INTENT_DISPVECT => {
            kind = PixelKind::Vector;
            // `m_ConvertRASDisplacementVectors` defaults to true.
            convert_ras = true;
        }
        NIFTI_INTENT_VECTOR => {
            kind = PixelKind::Vector;
            // `m_ConvertRASVectors` defaults to false.
        }
        _ => {}
    }

    // Rescale (:988-1017).
    let (rescale_slope, rescale_intercept) = if nim.nifti_type == FTYPE_ANALYZE {
        (1.0, 0.0)
    } else {
        let mut slope = nim.scl_slope as f64;
        if slope.abs() < f64::EPSILON {
            slope = 1.0;
        }
        (slope, nim.scl_inter as f64)
    };
    let on_disk = component;
    let must_rescale = rescale_slope.abs() > f64::EPSILON
        && ((rescale_slope - 1.0).abs() > f64::EPSILON || rescale_intercept.abs() > f64::EPSILON);
    if must_rescale && !matches!(component, PixelId::Float32 | PixelId::Float64) {
        component = PixelId::Float32;
    }

    // Units (:1020-1045).
    let spacingscale = match nim.xyz_units {
        NIFTI_UNITS_METER => 1e3,
        NIFTI_UNITS_MM => 1e0,
        NIFTI_UNITS_MICRON => 1e-3,
        _ => 1.0,
    };
    let timingscale = match nim.time_units {
        NIFTI_UNITS_SEC => 1.0,
        NIFTI_UNITS_MSEC => 1e-3,
        NIFTI_UNITS_USEC => 1e-6,
        _ => 1.0,
    };

    // Dimensions and spacing (:1050-1094).
    if dims == 0 || dims > 7 {
        return Err(IoError::UnsupportedImageDimension(dims));
    }
    let mut size = Vec::with_capacity(dims);
    let mut spacing = Vec::with_capacity(dims);
    for d in 0..dims {
        size.push(nim.dim[d + 1] as usize);
        let raw = nim.pixdim[d + 1] as f64;
        spacing.push(match d {
            0..=2 => raw * spacingscale,
            3 => raw * timingscale,
            // "Scaling is not defined in this dimension" (:1055).
            _ => raw,
        });
    }

    let (origin, direction, sform_corrected) =
        image_io_orientation(&nim, dims, spacingscale, timingscale)?;
    let metadata = build_metadata(&nim, sform_corrected);

    Ok(Info {
        nim,
        dims,
        components,
        kind,
        component,
        on_disk,
        rescale_slope,
        rescale_intercept,
        convert_ras,
        size,
        spacing,
        origin,
        direction,
        metadata,
    })
}

/// Read the header only — geometry, pixel type, and meta-data dictionary.
pub fn read_information(path: &Path) -> Result<ImageInformation> {
    let info = read_info(path)?;
    Ok(ImageInformation {
        pixel_id: info.pixel_id(),
        dimension: info.dims,
        number_of_components: info.components,
        size: info.size,
        spacing: info.spacing,
        origin: info.origin,
        direction: info.direction,
        metadata: info.metadata,
    })
}

// ---------------------------------------------------------------------------
// Pixel data
// ---------------------------------------------------------------------------

/// Decode `bytes` into a component buffer. `swapped` byte-swaps each
/// `swapsize`-byte group, which is `nifti_read_buffer`'s rule
/// (nifti1_io.c:5030-5034) — for `COMPLEX64` the swap size is `4`, so the real
/// and imaginary halves are swapped independently.
///
/// `FLOAT32`/`FLOAT64`/`COMPLEX64`/`COMPLEX128` are read verbatim. Upstream's
/// `IS_GOOD_FLOAT` block (:5036-5070) maps every NaN or infinity read from disk
/// to `0` — silent, irreversible pixel-data loss active only on glibc (where
/// `<math.h>` defines `isfinite` as a macro). This port **fixes** that at
/// source and preserves the non-finite pixels (ledger §2.90).
fn decode(bytes: &[u8], datatype: i16, swapped: bool) -> PixelBuffer {
    macro_rules! unpack {
        ($ty:ty, $variant:ident) => {{
            const S: usize = std::mem::size_of::<$ty>();
            PixelBuffer::$variant(
                bytes
                    .chunks_exact(S)
                    .map(|c| {
                        let a: [u8; S] = c.try_into().expect("chunks_exact yields S bytes");
                        if swapped {
                            <$ty>::from_be_bytes(a)
                        } else {
                            <$ty>::from_le_bytes(a)
                        }
                    })
                    .collect(),
            )
        }};
    }
    match datatype {
        NIFTI_TYPE_UINT8 | NIFTI_TYPE_RGB24 | NIFTI_TYPE_RGBA32 => {
            PixelBuffer::UInt8(bytes.to_vec())
        }
        NIFTI_TYPE_INT8 => PixelBuffer::Int8(bytes.iter().map(|&b| b as i8).collect()),
        NIFTI_TYPE_INT16 => unpack!(i16, Int16),
        NIFTI_TYPE_UINT16 => unpack!(u16, UInt16),
        NIFTI_TYPE_INT32 => unpack!(i32, Int32),
        NIFTI_TYPE_UINT32 => unpack!(u32, UInt32),
        NIFTI_TYPE_INT64 => unpack!(i64, Int64),
        NIFTI_TYPE_UINT64 => unpack!(u64, UInt64),
        NIFTI_TYPE_FLOAT32 | NIFTI_TYPE_COMPLEX64 => unpack!(f32, Float32),
        NIFTI_TYPE_FLOAT64 | NIFTI_TYPE_COMPLEX128 => unpack!(f64, Float64),
        _ => unreachable!("datatype validated by read_info"),
    }
}

/// `CastCopy<PixelType>` (itkNiftiImageIO.cxx:250-259) — widen an integer
/// buffer to `float` ahead of the rescale.
fn cast_to_f32(buf: &PixelBuffer) -> Vec<f32> {
    match buf {
        PixelBuffer::UInt8(v) => v.iter().map(|&x| x as f32).collect(),
        PixelBuffer::Int8(v) => v.iter().map(|&x| x as f32).collect(),
        PixelBuffer::UInt16(v) => v.iter().map(|&x| x as f32).collect(),
        PixelBuffer::Int16(v) => v.iter().map(|&x| x as f32).collect(),
        PixelBuffer::UInt32(v) => v.iter().map(|&x| x as f32).collect(),
        PixelBuffer::Int32(v) => v.iter().map(|&x| x as f32).collect(),
        PixelBuffer::UInt64(v) => v.iter().map(|&x| x as f32).collect(),
        PixelBuffer::Int64(v) => v.iter().map(|&x| x as f32).collect(),
        PixelBuffer::Float32(v) => v.clone(),
        PixelBuffer::Float64(v) => v.iter().map(|&x| x as f32).collect(),
    }
}

/// `RescaleFunction` (itkNiftiImageIO.cxx:237-247), applied to the first
/// `count` elements of the buffer.
///
/// Fixed §1.50: upstream calls `RescaleFunction(buffer, ..., numElts)` with
/// `numElts` — the **voxel** count — rather than `numElts * GetNumberOfComponents()`,
/// so on a multi-component image (e.g. a `COMPLEX64` buffer, `2·numElts`
/// floats) only a prefix of the buffer is rescaled and the tail keeps its raw
/// on-disk values. `count` here is `numElts * components`, covering every
/// component of every voxel. The integer arms of upstream's dispatch
/// (:518-548) are dead: `ReadImageInformation` has already promoted every
/// integer component type to `float`.
fn rescale(buf: &mut PixelBuffer, slope: f64, intercept: f64, count: usize) {
    match buf {
        PixelBuffer::Float32(v) => {
            for x in v.iter_mut().take(count) {
                *x = (*x as f64 * slope + intercept) as f32;
            }
        }
        PixelBuffer::Float64(v) => {
            for x in v.iter_mut().take(count) {
                *x = *x * slope + intercept;
            }
        }
        _ => unreachable!("rescale runs only after the promotion to float/double"),
    }
}

/// `ConvertRASToFromLPS_CXYZT` (itkNiftiImageIO.cxx:263-274): negate components
/// `0` and `1` of every group of three, over `size / 3` groups.
///
/// Fixed §1.51: upstream checks only the *pixel type* before calling this,
/// never that the component count is three, so a 2- or 4-component
/// `NIFTI_INTENT_DISPVECT` image got a stride-3 walk over an interleaved
/// buffer of a different stride. `read` now rejects a non-3-component vector
/// before calling this, matching what upstream's own exception text already
/// claimed to enforce.
fn convert_ras_cxyzt(buf: &mut PixelBuffer, size: usize) {
    let n = size / 3;
    macro_rules! flip {
        ($v:expr) => {
            for i in 0..n {
                $v[3 * i] = -$v[3 * i];
                $v[3 * i + 1] = -$v[3 * i + 1];
            }
        };
    }
    match buf {
        PixelBuffer::Float32(v) => flip!(v),
        PixelBuffer::Float64(v) => flip!(v),
        _ => unreachable!("guarded by the component-type check in read"),
    }
}

/// `ConvertRASToFromLPS_XYZTC` (itkNiftiImageIO.cxx:279-289): in NIfTI's
/// component-slowest layout the first `size / 3 * 2` components are the whole
/// `x` and `y` planes, so negating that prefix flips `L↔R` and `P↔A`.
///
/// Fixed §1.51 (write side): `write` rejects a non-3-component vector before
/// calling this, mirroring the same missing check in upstream's `Write`
/// (itkNiftiImageIO.cxx:2177-2183) that `Read` has (:565-571).
fn convert_ras_xyztc(buf: &mut [u8], component: PixelId, size: usize) {
    let n = size / 3 * 2;
    match component {
        PixelId::Float32 => {
            for chunk in buf.chunks_exact_mut(4).take(n) {
                let v = -f32::from_le_bytes(chunk.try_into().expect("4 bytes"));
                chunk.copy_from_slice(&v.to_le_bytes());
            }
        }
        PixelId::Float64 => {
            for chunk in buf.chunks_exact_mut(8).take(n) {
                let v = -f64::from_le_bytes(chunk.try_into().expect("8 bytes"));
                chunk.copy_from_slice(&v.to_le_bytes());
            }
        }
        _ => unreachable!("guarded by the component-type check in write"),
    }
}

/// Read the whole image.
///
/// `NiftiImageIO::Read` (itkNiftiImageIO.cxx:292-584) reads a sub-region when
/// the requested `ImageIORegion` is smaller than the file's; this port always
/// takes the whole-block path, since [`ImageIo::read`] has no region parameter
/// and [`crate::ImageFileReader`] extracts after the fact.
pub fn read(path: &Path) -> Result<Image> {
    let info = read_info(path)?;
    let nim = &info.nim;

    let img_path = find_img_name(&nim.iname, nim.nifti_type)
        .ok_or_else(|| IoError::FileNotFound(nim.iname.clone()))?;

    // `nifti_image_load`: `nvox * nbyper` bytes at `iname_offset`.
    let raw = read_all(&img_path)?;
    let ntot = (nim.nvox as usize)
        .checked_mul(nim.nbyper)
        .ok_or(IoError::TruncatedData)?;
    let offset = if nim.iname_offset < 0 {
        raw.len().saturating_sub(ntot)
    } else {
        nim.iname_offset as usize
    };
    if raw.len() < offset + ntot {
        return Err(IoError::TruncatedData);
    }
    let data = &raw[offset..offset + ntot];

    let cpe = components_per_element(nim.datatype);
    let num_elts: usize = info.size.iter().product();
    if info.kind == PixelKind::Vector && info.components == 0 {
        // `dim[5] == 0` on an intent-vector image. Upstream calls
        // `SetNumberOfComponents(0)` and divides by it downstream; refuse
        // (ledger §4.61).
        return Err(IoError::MalformedNiftiHeader(
            "intent-vector image declares dim[5] = 0 components".into(),
        ));
    }

    let mut buffer = decode(data, nim.datatype, nim.swapped);

    // The `float` promotion happens on the nifti-ordered buffer, before the
    // de-interleave (itkNiftiImageIO.cxx:380-436).
    let promoted = info.must_rescale() && info.component != info.on_disk;
    if promoted {
        if info.kind == PixelKind::Vector {
            // Upstream sizes the cast buffer `numElts * sizeof(float)` while
            // copying `numElts * numComponents * sizeof(float)` bytes out of it,
            // and then strides the de-interleave by `numComponents * 4` bytes
            // per *component*. Both overrun the heap (ledger §1.49); there is no
            // well-defined behaviour to port.
            return Err(IoError::UnsupportedNiftiFeature(format!(
                "{}: a vector NIfTI image with an integer datatype and a non-trivial \
                 scl_slope/scl_inter drives itkNiftiImageIO::Read into a heap overflow \
                 (ledger §1.49/§4.59); refusing to guess what it meant",
                path.display()
            )));
        }
        buffer = PixelBuffer::Float32(cast_to_f32(&buffer));
    }
    let component_after_cast = info.component;

    // nifti is `x y z t vec`, ITK is `vec x y z t` (:452-509). Only a genuine
    // vector image is re-ordered: scalar, complex, RGB and RGBA are memcpy'd.
    if info.kind == PixelKind::Vector && info.components > 1 {
        let series = num_elts; // dim[1..4] product == numElts for every vector case
        let needed = series
            .checked_mul(info.components)
            .ok_or(IoError::TruncatedData)?;
        if buffer.len() < needed {
            return Err(IoError::TruncatedData);
        }
        buffer = deinterleave(&buffer, series, info.components);
    } else {
        let keep = num_elts.checked_mul(cpe).ok_or(IoError::TruncatedData)?;
        if buffer.len() < keep {
            return Err(IoError::TruncatedData);
        }
        buffer = truncate(buffer, keep);
    }

    // A symmetric-matrix intent image is stored lower-triangular in NIfTI and
    // upper-triangular in ITK; after the de-interleave above (which is
    // component-order-preserving) reorder the components of every pixel so the
    // vector image matches what `itk::Image` would hold — upstream's `vecOrder`
    // remap in the copy loop (itkNiftiImageIO.cxx:463-501, ledger §3.32).
    if info.nim.intent_code == NIFTI_INTENT_SYMMATRIX && info.components > 1 {
        let order = upper_to_lower_order(sym_mat_dim(info.components));
        buffer = permute_components(buffer, num_elts, info.components, &order);
    }

    if info.must_rescale() {
        rescale(
            &mut buffer,
            info.rescale_slope,
            info.rescale_intercept,
            num_elts * info.components,
        );
    }

    if info.convert_ras {
        // Fixed §1.51: upstream's guard (itkNiftiImageIO.cxx:566-570) checks
        // only the pixel type, never that `numComponents == 3`, even though
        // its own exception text names that count. Enforce it here.
        if info.components != 3 {
            return Err(IoError::UnsupportedNiftiFeature(format!(
                "RAS conversion requires pixel to be 3-component vector or point. \
                 Current pixel type is {}-component VECTOR.",
                info.components
            )));
        }
        if !matches!(component_after_cast, PixelId::Float32 | PixelId::Float64) {
            return Err(IoError::UnsupportedNiftiFeature(format!(
                "RAS conversion of datatype {} is not supported",
                component_after_cast.as_str()
            )));
        }
        convert_ras_cxyzt(&mut buffer, num_elts * info.components);
    }

    let mut image = assemble(
        buffer,
        info.kind,
        info.components,
        info.size.clone(),
        info.spacing.clone(),
        info.origin.clone(),
        info.direction.clone(),
    )?;
    for (key, value) in &info.metadata {
        image.set_meta_data(key, value);
    }
    Ok(image)
}

/// `SymMatDim(count)` (itkNiftiImageIO.cxx:123-135): the rank `d` of a symmetric
/// matrix with `count` unique (triangular) entries, where T(d) = d·(d+1)/2.
fn sym_mat_dim(count: usize) -> usize {
    let mut dim = 0;
    let mut row = 1;
    let mut c = count as isize;
    while c > 0 {
        c -= row;
        dim += 1;
        row += 1;
    }
    dim
}

/// `UpperToLowerOrder(dim)` (itkNiftiImageIO.cxx:83-119): `order[c]` is the ITK
/// (upper-triangular) buffer position that on-disk (NIfTI lower-triangular)
/// component `c` moves to — upstream's `vecOrder`. For `dim = 3` this is
/// `[0, 1, 3, 2, 4, 5]`.
fn upper_to_lower_order(dim: usize) -> Vec<usize> {
    // Linear index of the upper-triangular entry `(r, c)` with `r <= c`, filled
    // row by row: `sum_{k<r}(dim-k) + (c-r)`.
    let ut_index = |r: usize, c: usize| r * dim - r * r.saturating_sub(1) / 2 + (c - r);
    let mut order = Vec::with_capacity(dim * (dim + 1) / 2);
    for i in 0..dim {
        for j in 0..=i {
            order.push(ut_index(j, i));
        }
    }
    order
}

/// Reorder the `components` components of every one of `num_pixels` pixels in an
/// interleaved (component-fastest) buffer: `out[pixel][order[c]] = in[pixel][c]`.
/// `order` is a permutation of `0..components`.
fn permute_components(
    buf: PixelBuffer,
    num_pixels: usize,
    components: usize,
    order: &[usize],
) -> PixelBuffer {
    macro_rules! p {
        ($v:expr, $variant:ident) => {{
            let src = $v;
            let mut out = src.clone();
            for pixel in 0..num_pixels {
                let base = pixel * components;
                for c in 0..components {
                    out[base + order[c]] = src[base + c];
                }
            }
            PixelBuffer::$variant(out)
        }};
    }
    match buf {
        PixelBuffer::UInt8(v) => p!(v, UInt8),
        PixelBuffer::Int8(v) => p!(v, Int8),
        PixelBuffer::UInt16(v) => p!(v, UInt16),
        PixelBuffer::Int16(v) => p!(v, Int16),
        PixelBuffer::UInt32(v) => p!(v, UInt32),
        PixelBuffer::Int32(v) => p!(v, Int32),
        PixelBuffer::UInt64(v) => p!(v, UInt64),
        PixelBuffer::Int64(v) => p!(v, Int64),
        PixelBuffer::Float32(v) => p!(v, Float32),
        PixelBuffer::Float64(v) => p!(v, Float64),
    }
}

fn truncate(buf: PixelBuffer, keep: usize) -> PixelBuffer {
    macro_rules! t {
        ($v:expr, $variant:ident) => {{
            let mut v = $v;
            v.truncate(keep);
            PixelBuffer::$variant(v)
        }};
    }
    match buf {
        PixelBuffer::UInt8(v) => t!(v, UInt8),
        PixelBuffer::Int8(v) => t!(v, Int8),
        PixelBuffer::UInt16(v) => t!(v, UInt16),
        PixelBuffer::Int16(v) => t!(v, Int16),
        PixelBuffer::UInt32(v) => t!(v, UInt32),
        PixelBuffer::Int32(v) => t!(v, Int32),
        PixelBuffer::UInt64(v) => t!(v, UInt64),
        PixelBuffer::Int64(v) => t!(v, Int64),
        PixelBuffer::Float32(v) => t!(v, Float32),
        PixelBuffer::Float64(v) => t!(v, Float64),
    }
}

/// `nifti[c][voxel] -> itk[voxel][c]`.
fn deinterleave(buf: &PixelBuffer, series: usize, components: usize) -> PixelBuffer {
    macro_rules! d {
        ($v:expr, $variant:ident) => {{
            let mut out = Vec::with_capacity(series * components);
            for voxel in 0..series {
                for c in 0..components {
                    out.push($v[c * series + voxel]);
                }
            }
            PixelBuffer::$variant(out)
        }};
    }
    match buf {
        PixelBuffer::UInt8(v) => d!(v, UInt8),
        PixelBuffer::Int8(v) => d!(v, Int8),
        PixelBuffer::UInt16(v) => d!(v, UInt16),
        PixelBuffer::Int16(v) => d!(v, Int16),
        PixelBuffer::UInt32(v) => d!(v, UInt32),
        PixelBuffer::Int32(v) => d!(v, Int32),
        PixelBuffer::UInt64(v) => d!(v, UInt64),
        PixelBuffer::Int64(v) => d!(v, Int64),
        PixelBuffer::Float32(v) => d!(v, Float32),
        PixelBuffer::Float64(v) => d!(v, Float64),
    }
}

/// `itk[voxel][c] -> nifti[c][voxel]` (itkNiftiImageIO.cxx:2151-2175).
fn interleave_to_nifti(buf: &PixelBuffer, series: usize, components: usize) -> Vec<u8> {
    macro_rules! i {
        ($v:expr) => {{
            let mut out = Vec::with_capacity(series * components * std::mem::size_of_val(&$v[0]));
            for c in 0..components {
                for voxel in 0..series {
                    out.extend_from_slice(&$v[voxel * components + c].to_le_bytes());
                }
            }
            out
        }};
    }
    match buf {
        PixelBuffer::UInt8(v) => i!(v),
        PixelBuffer::Int8(v) => {
            let mut out = Vec::with_capacity(series * components);
            for c in 0..components {
                for voxel in 0..series {
                    out.push(v[voxel * components + c] as u8);
                }
            }
            out
        }
        PixelBuffer::UInt16(v) => i!(v),
        PixelBuffer::Int16(v) => i!(v),
        PixelBuffer::UInt32(v) => i!(v),
        PixelBuffer::Int32(v) => i!(v),
        PixelBuffer::UInt64(v) => i!(v),
        PixelBuffer::Int64(v) => i!(v),
        PixelBuffer::Float32(v) => i!(v),
        PixelBuffer::Float64(v) => i!(v),
    }
}

fn to_le_bytes(buf: &PixelBuffer) -> Vec<u8> {
    macro_rules! p {
        ($v:expr) => {{
            let mut out = Vec::with_capacity(std::mem::size_of_val(&$v[..]));
            for x in $v.iter() {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }};
    }
    match buf {
        PixelBuffer::UInt8(v) => v.clone(),
        PixelBuffer::Int8(v) => v.iter().map(|&x| x as u8).collect(),
        PixelBuffer::UInt16(v) => p!(v),
        PixelBuffer::Int16(v) => p!(v),
        PixelBuffer::UInt32(v) => p!(v),
        PixelBuffer::Int32(v) => p!(v),
        PixelBuffer::UInt64(v) => p!(v),
        PixelBuffer::Int64(v) => p!(v),
        PixelBuffer::Float32(v) => p!(v),
        PixelBuffer::Float64(v) => p!(v),
    }
}

/// Build the [`Image`] for the pixel type `ReadImageInformation` settled on.
///
/// `Image::assemble` is private, so a complex image is built from its
/// interleaved buffer via `from_vec_complex` and then given its geometry —
/// which is what `sitk::ImageFileReader::Execute` does through
/// `ImportImageFilter`, only in one step.
fn assemble(
    buffer: PixelBuffer,
    kind: PixelKind,
    components: usize,
    size: Vec<usize>,
    spacing: Vec<f64>,
    origin: Vec<f64>,
    direction: Vec<f64>,
) -> Result<Image> {
    match kind {
        PixelKind::Scalar => Ok(Image::from_parts(buffer, size, spacing, origin, direction)?),
        PixelKind::Rgb | PixelKind::Vector => Ok(Image::from_parts_vector(
            buffer, components, size, spacing, origin, direction,
        )?),
        PixelKind::Complex => {
            let mut image = match buffer {
                PixelBuffer::Float32(v) => Image::from_vec_complex(
                    &size,
                    v.chunks_exact(2)
                        .map(|c| Complex::new(c[0], c[1]))
                        .collect(),
                )?,
                PixelBuffer::Float64(v) => Image::from_vec_complex(
                    &size,
                    v.chunks_exact(2)
                        .map(|c| Complex::new(c[0], c[1]))
                        .collect(),
                )?,
                _ => unreachable!("a complex NIfTI datatype decodes to a float buffer"),
            };
            image.set_spacing(&spacing)?;
            image.set_origin(&origin)?;
            image.set_direction(&direction)?;
            Ok(image)
        }
    }
}

// ---------------------------------------------------------------------------
// WriteImageInformation / Write
// ---------------------------------------------------------------------------

/// `getQFormCodeFromDictionary` / `getSFormCodeFromDictionary`
/// (itkNiftiImageIO.cxx:1912-1947).
///
/// Both results are **dead**: `SetNIfTIOrientationFromImageIO` overwrites
/// `qform_code` and `sform_code` with `NIFTI_XFORM_SCANNER_ANAT` at its last two
/// statements (:2077-2078). They are still evaluated here because
/// `itk::StringToInt32` *throws* on a non-numeric value, so a dictionary
/// carrying `qform_code = "abc"` fails the write (ledger §2.88).
fn xform_code_from_dictionary(image: &Image, name_key: &str, code_key: &str) -> Result<i16> {
    if let Some(name) = image.meta_data(name_key) {
        return Ok(str_xform2code(name));
    }
    if let Some(code) = image.meta_data(code_key) {
        return code.trim().parse::<i32>().map(|v| v as i16).map_err(|_| {
            IoError::InvalidNiftiMetaData(format!("NIfTI metadata '{code_key}': {code}"))
        });
    }
    Ok(NIFTI_XFORM_SCANNER_ANAT)
}

/// What [`write`] needs from `WriteImageInformation` before it can lay out the
/// bytes.
struct WriteInfo {
    header: RawHeader,
    nifti_type: i32,
    components: usize,
    /// `dim[1] * dim[2] * dim[3] * dim[4]`, the vector branch's `seriesdist`.
    series: usize,
    /// `nvox * nbyper`.
    data_bytes: usize,
    convert_ras: bool,
    /// `IsCompressed` (itkNiftiImageIO.cxx:1173-1174) — the file name ended in
    /// a lowercase `.gz`, and so `nifti_makehdrname`/`nifti_makeimgname` put it
    /// back and `znzopen` gzips the stream.
    is_compressed: bool,
}

/// `NiftiImageIO::WriteImageInformation` (itkNiftiImageIO.cxx:1136-1502) and
/// `SetNIfTIOrientationFromImageIO` (:1949-2079), collapsed into the on-disk
/// header they jointly produce via `nifti_convert_nim2nhdr` (nifti1_io.c:5474).
fn write_information(image: &Image, path: &Path) -> Result<WriteInfo> {
    let dims = image.dimension();
    for (i, &d) in image.size().iter().enumerate() {
        if d > i16::MAX as usize {
            return Err(IoError::NiftiWriteRejected(format!(
                "Dimension({i}) = {d} is greater than maximum possible dimension {}",
                i16::MAX
            )));
        }
    }

    let name = path.to_string_lossy().into_owned();
    let ext = find_file_extension(&name).ok_or_else(|| {
        IoError::NiftiWriteRejected(format!(
            "Bad Nifti file name. No extension found for file: {name}"
        ))
    })?;
    // `WriteImageInformation` compares `ExtensionName` with `==` against the
    // lowercase spellings (itkNiftiImageIO.cxx:1175-1201), even though
    // `CanWriteFile` accepted the uppercase one — so `.NII` reaches here and is
    // refused (ledger §2.91). `IsCompressed` likewise looks for a lowercase
    // `.gz` only (:1173).
    let is_compressed = ext.rfind(".gz").is_some();
    let nifti_type = match ext {
        ".nii" | ".nii.gz" => FTYPE_NIFTI1_1,
        ".nia" => {
            return Err(IoError::UnsupportedNiftiFeature(
                "the NIfTI ASCII variant (.nia, NIFTI_FTYPE_ASCII)".into(),
            ));
        }
        ".hdr" | ".img" | ".hdr.gz" | ".img.gz" => FTYPE_NIFTI1_2,
        _ => {
            return Err(IoError::NiftiWriteRejected(format!(
                "Bad Nifti file name: {name}"
            )));
        }
    };

    let components = image.buffer_stride();
    let kind = if image.pixel_id().is_complex() {
        PixelKind::Complex
    } else if components > 1 {
        PixelKind::Vector
    } else {
        PixelKind::Scalar
    };

    let mut nhdr = RawHeader {
        sizeof_hdr: HEADER_SIZE as i32,
        ..RawHeader::default()
    };
    // `nifti_simple_init_nim` leaves dz at 1.0; the switch below only sets the
    // pixdims the image actually has.
    let mut pixdim = [0.0f32; 8];
    pixdim[1] = 1.0;
    pixdim[2] = 1.0;
    pixdim[3] = 1.0;
    let mut dim = [1i16; 8];
    let mut toffset = 0.0f32;
    for d in 0..dims {
        dim[d + 1] = image.size()[d] as i16;
        pixdim[d + 1] = image.spacing()[d] as f32;
    }
    if dims >= 4 {
        toffset = image.origin()[3] as f32;
    }

    let mut intent_code = 0i16;
    if kind == PixelKind::Vector {
        if dims > 4 {
            return Err(IoError::NiftiWriteRejected(format!(
                "Can not store a vector image of more than 4 dimensions in a Nifti file. \
                 Dimension={dims}"
            )));
        }
        dim[0] = 5;
        intent_code = NIFTI_INTENT_VECTOR;
        if let Some(v) = image.meta_data("intent_code") {
            if v.trim().parse::<i32>().ok() == Some(NIFTI_INTENT_DISPVECT as i32) {
                intent_code = NIFTI_INTENT_DISPVECT;
            }
        }
        dim[5] = components as i16;
        for d in dims..4 {
            dim[d + 1] = 1;
        }
    } else {
        dim[0] = dims as i16;
    }

    // datatype / nbyper (:1344-1464).
    let (mut datatype, mut nbyper) = match image.pixel_id().component_id() {
        PixelId::UInt8 => (NIFTI_TYPE_UINT8, 1usize),
        PixelId::Int8 => (NIFTI_TYPE_INT8, 1),
        PixelId::UInt16 => (NIFTI_TYPE_UINT16, 2),
        PixelId::Int16 => (NIFTI_TYPE_INT16, 2),
        PixelId::UInt32 => (NIFTI_TYPE_UINT32, 4),
        PixelId::Int32 => (NIFTI_TYPE_INT32, 4),
        PixelId::UInt64 => (NIFTI_TYPE_UINT64, 8),
        PixelId::Int64 => (NIFTI_TYPE_INT64, 8),
        PixelId::Float32 => (NIFTI_TYPE_FLOAT32, 4),
        PixelId::Float64 => (NIFTI_TYPE_FLOAT64, 8),
        other => unreachable!("component_id never yields {other:?}"),
    };
    if kind == PixelKind::Complex {
        nbyper *= 2;
        datatype = if datatype == NIFTI_TYPE_FLOAT32 {
            NIFTI_TYPE_COMPLEX64
        } else {
            NIFTI_TYPE_COMPLEX128
        };
    }

    // SetNIfTIOrientationFromImageIO. The dictionary codes are computed for
    // their (throwing) side effect and then discarded, exactly as upstream.
    let _ = xform_code_from_dictionary(image, "qform_code_name", "qform_code")?;
    let _ = xform_code_from_dictionary(image, "sform_code_name", "sform_code")?;

    // `GetDirection(i)` is the `i`-th *column* of the direction matrix
    // (itkImageFileWriter.hxx:198-202), and it has `dims` entries — so for
    // `dims < 3` upstream's `if (i < 3) dirx[2] = 0` is a no-op on the
    // already-zeroed tail.
    let negated_column = |i: usize| -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for (j, v) in out.iter_mut().enumerate().take(dims.min(3)) {
            *v = -(image.direction()[j * dims + i] as f32);
        }
        out
    };
    let mut dirx = negated_column(0);
    let mut diry = if dims > 1 {
        negated_column(1)
    } else {
        [0.0; 3]
    };
    let mut dirz = if dims > 2 {
        negated_column(2)
    } else {
        [0.0, 0.0, 1.0]
    };
    if dims > 2 {
        // "Read comments in nifti1.h about interpreting DICOM Image
        // Orientation (Patient)" — this restores the RAS `+z` row.
        dirx[2] = -dirx[2];
        diry[2] = -diry[2];
        dirz[2] = -dirz[2];
    }

    let mut matrix = mat44_transpose(&make_orthog_mat44([dirx, diry, dirz]));
    matrix[0][3] = -image.origin()[0] as f32;
    matrix[1][3] = if dims > 1 {
        -image.origin()[1] as f32
    } else {
        0.0
    };
    // "NOTE: The final dimension is not negated!"
    matrix[2][3] = if dims > 2 {
        image.origin()[2] as f32
    } else {
        0.0
    };

    let q = mat44_to_quatern(&matrix);
    let mut sto_xyz = matrix;
    let sto_limit = dims.min(3);
    for row in sto_xyz.iter_mut().take(sto_limit) {
        for (jj, cell) in row.iter_mut().enumerate().take(sto_limit) {
            *cell *= image.spacing()[jj] as f32;
        }
    }
    pixdim[0] = q.qfac;

    // aux_file / ITK_FileNotes (:1475-1497).
    let mut aux_file = [0u8; 24];
    if let Some(v) = image.meta_data("aux_file") {
        if v.len() > 23 {
            return Err(IoError::InvalidNiftiMetaData(
                "aux_file too long, Nifti limit is 23 characters".into(),
            ));
        }
        aux_file[..v.len()].copy_from_slice(v.as_bytes());
    }
    let mut descrip = [0u8; 80];
    if let Some(v) = image.meta_data("ITK_FileNotes") {
        if v.len() > 79 {
            return Err(IoError::InvalidNiftiMetaData(
                "ITK_FileNotes (Nifti descrip field) too long, Nifti limit is 79 characters".into(),
            ));
        }
        descrip[..v.len()].copy_from_slice(v.as_bytes());
    }

    // `m_ConvertRASVectors` is false and `m_ConvertRASDisplacementVectors` is
    // true by default (itkNiftiImageIO.h:288-289).
    let convert_ras = intent_code == NIFTI_INTENT_DISPVECT;

    // nifti_convert_nim2nhdr (nifti1_io.c:5474-5582).
    nhdr.dim = dim;
    nhdr.pixdim = [
        pixdim[0],
        pixdim[1].abs(),
        pixdim[2].abs(),
        pixdim[3].abs(),
        pixdim[4].abs(),
        pixdim[5].abs(),
        pixdim[6].abs(),
        pixdim[7].abs(),
    ];
    nhdr.datatype = datatype;
    nhdr.bitpix = (8 * nbyper) as i16;
    // `cal_max > cal_min` is false (both zero), so neither is written.
    // `scl_slope != 0` is true: m_RescaleSlope defaults to 1.0.
    nhdr.scl_slope = 1.0;
    nhdr.scl_inter = 0.0;
    nhdr.descrip = descrip;
    nhdr.aux_file = aux_file;
    nhdr.magic = if nifti_type == FTYPE_NIFTI1_1 {
        *b"n+1\0"
    } else {
        *b"ni1\0"
    };
    nhdr.intent_code = intent_code;
    nhdr.vox_offset = if nifti_type == FTYPE_NIFTI1_1 {
        SINGLE_FILE_VOX_OFFSET as f32
    } else {
        0.0
    };
    nhdr.xyzt_units = (NIFTI_UNITS_MM & 0x07) | (NIFTI_UNITS_SEC & 0x38);
    nhdr.toffset = toffset;
    nhdr.qform_code = NIFTI_XFORM_SCANNER_ANAT;
    nhdr.quatern_b = q.b;
    nhdr.quatern_c = q.c;
    nhdr.quatern_d = q.d;
    nhdr.qoffset_x = q.x;
    nhdr.qoffset_y = q.y;
    nhdr.qoffset_z = q.z;
    nhdr.pixdim[0] = if q.qfac >= 0.0 { 1.0 } else { -1.0 };
    nhdr.sform_code = NIFTI_XFORM_SCANNER_ANAT;
    nhdr.srow_x = sto_xyz[0];
    nhdr.srow_y = sto_xyz[1];
    nhdr.srow_z = sto_xyz[2];

    let series = (dim[1] as usize) * (dim[2] as usize) * (dim[3] as usize) * (dim[4] as usize);
    let nvox = if kind == PixelKind::Vector {
        series * components
    } else {
        series
    };

    Ok(WriteInfo {
        header: nhdr,
        nifti_type,
        components,
        series,
        data_bytes: nvox * nbyper,
        convert_ras,
        is_compressed,
    })
}

/// Write `image` as NIfTI-1. `.nii` is a single file (header, the four-byte
/// zero extender, then the pixel data at `vox_offset = 352`); `.hdr`/`.img`
/// write a 352-byte header file and a sibling `.img` holding the pixels from
/// offset `0`.
///
/// A `.gz` on the name gzips each file that gets written — the `.nii.gz` case
/// is one stream over header, extender and pixels; `.hdr.gz` / `.img.gz` are
/// two independent streams, because `nifti_image_write_engine` closes the
/// header's `znzFile` before opening the image's (nifti1_io.c:5958-5971).
///
/// [`WriteOptions`] does **not** reach this format. `NiftiImageIO` never
/// consults `GetUseCompression` — the extension alone decides
/// (itkNiftiImageIO.cxx:1173) — and never passes a level, so `znzopen(..., "wb")`
/// always deflates at zlib's default of 6 (ledger §3.40).
pub fn write(image: &Image, path: &Path) -> Result<()> {
    let info = write_information(image, path)?;
    let kind = if image.pixel_id().is_complex() {
        PixelKind::Complex
    } else if info.components > 1 {
        PixelKind::Vector
    } else {
        PixelKind::Scalar
    };

    let component = image.pixel_id().component_id();
    if info.convert_ras && info.components != 3 {
        // Fixed §1.51: upstream's guard (itkNiftiImageIO.cxx:2177-2183) checks
        // only the pixel type, never that `numComponents == 3`, even though its
        // own exception text names that count. Enforce it here.
        return Err(IoError::NiftiWriteRejected(format!(
            "RAS conversion requires pixel to be 3-component vector or point. \
             Current pixel type is {}-component VECTOR.",
            info.components
        )));
    }
    let mut data = if kind == PixelKind::Vector {
        if info.convert_ras && !matches!(component, PixelId::Float32 | PixelId::Float64) {
            return Err(IoError::NiftiWriteRejected(format!(
                "RAS conversion of datatype {} is not supported",
                component.as_str()
            )));
        }
        interleave_to_nifti(image.buffer(), info.series, info.components)
    } else {
        to_le_bytes(image.buffer())
    };
    if info.convert_ras {
        convert_ras_xyztc(&mut data, component, info.components * info.series);
    }
    debug_assert_eq!(data.len(), info.data_bytes);

    let base = make_basename(path);
    let header_bytes = info.header.to_bytes();

    // `nifti_image_write_engine` (nifti1_io.c:5850-5960): the header, then
    // `nifti_write_extensions` — which, with `num_ext == 0` and
    // `skip_blank_ext == 0`, emits a four-byte zero extender — then the pixels
    // at `iname_offset`.
    let mut header_block = Vec::with_capacity(SINGLE_FILE_VOX_OFFSET as usize);
    header_block.extend_from_slice(&header_bytes);
    header_block.extend_from_slice(&[0u8; 4]);

    let comp = info.is_compressed;
    // One gzip stream per file `znzopen` opens.
    let maybe_gzip = |bytes: Vec<u8>| {
        if comp {
            gzip_compress(&bytes, ZLIB_DEFAULT_COMPRESSION_LEVEL)
        } else {
            bytes
        }
    };

    if info.nifti_type == FTYPE_NIFTI1_1 {
        header_block.extend_from_slice(&data);
        std::fs::write(
            make_hdrname(&base, info.nifti_type, comp),
            maybe_gzip(header_block),
        )?;
    } else {
        std::fs::write(
            make_hdrname(&base, info.nifti_type, comp),
            maybe_gzip(header_block),
        )?;
        std::fs::write(make_imgname(&base, info.nifti_type, comp), maybe_gzip(data))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The ImageIo implementor
// ---------------------------------------------------------------------------

/// `itk::NiftiImageIO`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NiftiImageIo;

impl ImageIo for NiftiImageIo {
    fn name(&self) -> &'static str {
        "NiftiImageIO"
    }

    fn supported_read_extensions(&self) -> &'static [&'static str] {
        SUPPORTED_EXTENSIONS
    }

    fn supported_write_extensions(&self) -> &'static [&'static str] {
        SUPPORTED_EXTENSIONS
    }

    /// `NiftiImageIO::CanReadFile` (itkNiftiImageIO.cxx:602-621): `is_nifti_file`
    /// answering `1`/`2` claims the file; `0` (an Analyze-7.5 header) claims it
    /// too, because the default `Analyze75Flavor` is not `AnalyzeReject`.
    ///
    /// Note that `is_nifti_file` resolves the *header* file itself
    /// (`nifti_findhdrname`), so `can_read_file("brain")` is true when
    /// `brain.nii` exists — the extension is a hint, never a requirement, which
    /// is the opposite of [`crate::meta_image::MetaImageIo`]'s behaviour.
    fn can_read_file(&self, path: &Path) -> bool {
        is_nifti_file(path) >= 0
    }

    /// `NiftiImageIO::CanWriteFile` is `nifti_is_complete_filename`
    /// (itkNiftiImageIO.cxx:217-223) — a pure name test.
    fn can_write_file(&self, path: &Path) -> bool {
        is_complete_filename(&path.to_string_lossy())
    }

    fn read_information(&self, path: &Path) -> Result<ImageInformation> {
        read_information(path)
    }

    fn read(&self, path: &Path) -> Result<Image> {
        read(path)
    }

    /// `options` is ignored: NIfTI compresses iff the file name ends in `.gz`,
    /// and always at zlib's default level. See [`write`].
    fn write(&self, image: &Image, path: &Path, _options: &WriteOptions) -> Result<()> {
        write(image, path)
    }
}
