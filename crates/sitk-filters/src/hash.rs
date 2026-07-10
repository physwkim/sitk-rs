//! `HashImageFilter` / `Hash()`: SHA1 or MD5 digest of an image's raw pixel
//! bytes.
//!
//! Ported from the SimpleITK-facing layer
//! (`Code/BasicFilters/include/sitkHashImageFilter.h:49-52`,
//! `Code/BasicFilters/src/sitkHashImageFilter.cxx:88-129` — the `Hash`
//! free function and `HashImageFilter::Execute`/`ExecuteInternal`) and the
//! filter it drives, which is vendored *inside SimpleITK's own tree* rather
//! than main ITK
//! (`Code/BasicFilters/include/itkHashImageFilter.hxx:59-152` — **not**
//! `ITK/Modules/Core/TestKernel/include/itkTestingHashImageFilter.hxx`, an
//! unrelated test-infrastructure filter that happens to share a name).
//!
//! ## What gets hashed
//!
//! `itk::HashImageFilter::AfterThreadedGenerateData`
//! (`itkHashImageFilter.hxx:71-135`):
//!
//! - `numberOfComponent = sizeof(PixelType) / sizeof(ValueType)` — `1` for a
//!   scalar pixel, `2` for complex (`NumericTraits<complex<T>>::ValueType`
//!   is `T`), and the vector length for a `VectorImage` (special-cased via
//!   `AccessorFunctorType::GetVectorLength`, since a `VectorImage`'s
//!   `PixelType` doesn't carry its length in `sizeof`). This is exactly
//!   [`Image::buffer_stride`], and the raw component buffer it multiplies
//!   against pixel count is [`Image::component_slice`] — both already unify
//!   scalar/complex/vector the same way ITK's `numberOfComponent` does, so
//!   no per-category branch is needed here.
//! - Each component is byte-swapped to little-endian
//!   (`Swapper::SwapRangeFromSystemToLittleEndian`) before hashing — a
//!   no-op in this port, which always serializes explicitly via a private
//!   `ToLeBytes` trait rather than reasoning about host byte order.
//! - The filter hashes raw bytes, not text: both algorithms run over the
//!   `numberOfValues * sizeof(ValueType)`-byte buffer.
//!
//! `sizeof(PixelType) % sizeof(ValueType) != 0` throws `"Unsupported data
//! type for hashing!"` for a non-`VectorImage` pixel type upstream
//! (`itkHashImageFilter.hxx:107-110`); structurally unreachable here, since
//! every pixel category this port supports has an integral component count
//! by construction ([`Image::buffer_stride`] is always a whole number of
//! `T`-sized components — there is no pixel type whose size is a
//! non-integer multiple of its own component size).
//!
//! ## Two defaults, not one
//!
//! ITK's `itk::HashImageFilter` constructor defaults to `MD5`
//! (`itkHashImageFilter.hxx:34` — `this->m_HashFunction = MD5;`), but
//! SimpleITK's facing layer defaults to `SHA1`
//! (`sitkHashImageFilter.h:75` — `HashFunction m_HashFunction{ SHA1 };`)
//! and *always* calls `hasher->SetHashFunction(this->GetHashFunction())`
//! before `Update()` (`sitkHashImageFilter.cxx:120-127`), so the ITK-level
//! `MD5` default never actually reaches a caller through
//! `itk::simple::Hash`. This port only exposes the SimpleITK-level default
//! ([`HashFunction::default`]).
//!
//! ## No `LabelMap` branch
//!
//! `HashImageFilter::ExecuteInternalLabelImage`
//! (`sitkHashImageFilter.cxx:92-107`) casts a `LabelMap`-pixel-typed
//! `Image` to a scalar image of the label's pixel type and recurses. Not
//! needed here: [`sitk_core::LabelMap`] is already a distinct type, not an
//! [`Image`] (`doc/upstream-findings.md` §4.25) — a caller hashes a label
//! map by calling
//! [`LabelMap::to_label_image`](sitk_core::LabelMap::to_label_image) first
//! and passing the result to [`hash_image`].
//!
//! ## Algorithm provenance
//!
//! ITK vendors C implementations of MD5 (`itksys/MD5.h`) and SHA1
//! (`Ancillary/hl_sha1.h`). This port implements both from scratch against
//! their public specifications — RFC 1321 (MD5) and FIPS 180-1 (SHA1) —
//! rather than transliterating the vendored C, and is pinned against the
//! standard known-answer test vectors published in each spec, plus an
//! image-hash test cross-checked independently against the system
//! `md5sum`/`sha1sum` (see the tests below).

use sitk_core::{Image, Scalar, dispatch_scalar};
use std::fmt::Write as _;

/// Which digest [`hash_image`] computes — `sitkHashImageFilter.h:52-56`'s
/// `HashImageFilter::HashFunction` enum, `SHA1` declared first to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashFunction {
    Sha1,
    Md5,
}

/// SimpleITK's facing-layer default (`sitkHashImageFilter.h:75`), **not**
/// `itk::HashImageFilter`'s own `MD5` default — see the module docs.
impl Default for HashFunction {
    fn default() -> Self {
        HashFunction::Sha1
    }
}

/// `itk::simple::Hash`: the SHA1 or MD5 digest of `img`'s raw pixel bytes,
/// as a lowercase hex string. See the module docs for exactly what bytes
/// are hashed and in what order.
///
/// Infallible: [`dispatch_scalar!`] always resolves a `T` matching
/// `img.pixel_id()`'s component type, so [`Image::component_slice`] cannot
/// fail for any `Image` that exists (same precedent as
/// [`crate::stable_time_step_bound`]/[`crate::fast_marching::large_value`]).
pub fn hash_image(img: &Image, function: HashFunction) -> String {
    dispatch_scalar!(img.pixel_id(), hash_image_typed, img, function)
}

fn hash_image_typed<T: Scalar + ToLeBytes>(img: &Image, function: HashFunction) -> String {
    let components = img
        .component_slice::<T>()
        .expect("dispatch_scalar! resolved T to img.pixel_id()'s own component type");
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(components));
    for &v in components {
        v.append_le_bytes(&mut bytes);
    }
    match function {
        HashFunction::Sha1 => sha1_hex(&bytes),
        HashFunction::Md5 => md5_hex(&bytes),
    }
}

/// Serializes one pixel-buffer component to little-endian bytes, matching
/// ITK's `Swapper::SwapRangeFromSystemToLittleEndian` (module docs). Scoped
/// to this module rather than added to [`sitk_core::Scalar`] — hashing is
/// the only consumer that needs a byte-level view of a pixel component.
trait ToLeBytes {
    fn append_le_bytes(self, out: &mut Vec<u8>);
}

macro_rules! impl_to_le_bytes {
    ($($t:ty),+ $(,)?) => {$(
        impl ToLeBytes for $t {
            fn append_le_bytes(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    )+};
}

impl_to_le_bytes!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

fn to_hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").expect("String writer never fails");
    }
    s
}

// ---- MD5 (RFC 1321) --------------------------------------------------------

/// Per-round left-rotate amounts, 4 values repeated 4 times per round
/// (RFC 1321 §3.4, the `S` table via `FF`/`GG`/`HH`/`II`'s literal shift
/// arguments).
const MD5_SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// `T[i] = floor(2^32 * abs(sin(i + 1)))`, `1 <= i <= 64` (RFC 1321 §3.4).
const MD5_K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// Pads `data` per RFC 1321 §3.1-3.2: a `0x80` byte, zeros up to 56 mod 64,
/// then the original bit length as a little-endian `u64`.
fn md5_pad(data: &[u8]) -> Vec<u8> {
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());
    msg
}

/// RFC 1321 MD5, returning the 32-character lowercase hex digest.
fn md5_hex(data: &[u8]) -> String {
    let msg = md5_pad(data);

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            m[i] = u32::from_le_bytes(word.try_into().unwrap());
        }

        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(MD5_K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(MD5_SHIFTS[i]));
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut digest = Vec::with_capacity(16);
    digest.extend_from_slice(&a0.to_le_bytes());
    digest.extend_from_slice(&b0.to_le_bytes());
    digest.extend_from_slice(&c0.to_le_bytes());
    digest.extend_from_slice(&d0.to_le_bytes());
    to_hex_lower(&digest)
}

// ---- SHA1 (FIPS 180-1) ------------------------------------------------------

/// Pads `data` per FIPS 180-1 §5: a `0x80` byte, zeros up to 56 mod 64,
/// then the original bit length as a big-endian `u64`.
fn sha1_pad(data: &[u8]) -> Vec<u8> {
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    msg
}

/// FIPS 180-1 SHA1, returning the 40-character lowercase hex digest.
fn sha1_hex(data: &[u8]) -> String {
    let msg = sha1_pad(data);

    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xefcdab89;
    let mut h2: u32 = 0x98badcfe;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xc3d2e1f0;

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5a827999u32),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
                _ => (b ^ c ^ d, 0xca62c1d6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut digest = Vec::with_capacity(20);
    digest.extend_from_slice(&h0.to_be_bytes());
    digest.extend_from_slice(&h1.to_be_bytes());
    digest.extend_from_slice(&h2.to_be_bytes());
    digest.extend_from_slice(&h3.to_be_bytes());
    digest.extend_from_slice(&h4.to_be_bytes());
    to_hex_lower(&digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sitk_core::PixelId;

    // ---- RFC 1321 §A.5 known-answer vectors --------------------------------

    #[test]
    fn md5_known_answer_vectors() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"a"), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5_hex(b"message digest"),
            "f96b697d7cb7938d525a2f31aaf161d0"
        );
        assert_eq!(
            md5_hex(b"abcdefghijklmnopqrstuvwxyz"),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
        assert_eq!(
            md5_hex(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
        assert_eq!(
            md5_hex(
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            ),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    #[test]
    fn md5_million_a() {
        let data = vec![b'a'; 1_000_000];
        assert_eq!(md5_hex(&data), "7707d6ae4e027c70eea2a935c2296f21");
    }

    // ---- FIPS 180-1 §A known-answer vectors --------------------------------

    #[test]
    fn sha1_known_answer_vectors() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"a"), "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            sha1_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test]
    fn sha1_million_a() {
        let data = vec![b'a'; 1_000_000];
        assert_eq!(sha1_hex(&data), "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    }

    // ---- hash_image ---------------------------------------------------------

    #[test]
    fn hash_image_default_is_sha1() {
        assert_eq!(HashFunction::default(), HashFunction::Sha1);
    }

    #[test]
    fn hash_image_u8_matches_first_principles() {
        // A 2x2 u8 image: component buffer is just the 4 pixel bytes
        // themselves (buffer_stride == 1, no byte-swap needed for u8).
        let img = Image::from_vec(&[2, 2], vec![1u8, 2, 3, 4]).unwrap();
        let expected_sha1 = sha1_hex(&[1, 2, 3, 4]);
        let expected_md5 = md5_hex(&[1, 2, 3, 4]);
        assert_eq!(hash_image(&img, HashFunction::Sha1), expected_sha1);
        assert_eq!(hash_image(&img, HashFunction::Md5), expected_md5);
        // Cross-checked independently against the system `sha1sum`/`md5sum`/
        // `openssl dgst` on the raw byte sequence `\x01\x02\x03\x04`.
        assert_eq!(expected_sha1, "12dada1fff4d4787ade3333147202c3b443e376f");
        assert_eq!(expected_md5, "08d6c05a21512a79a1dfeb9d2a8f262f");
    }

    #[test]
    fn hash_image_u16_serializes_little_endian() {
        // A single u16 pixel, value 0x0102: little-endian bytes are [0x02, 0x01].
        let img = Image::from_vec(&[1, 1], vec![0x0102u16]).unwrap();
        let expected = sha1_hex(&[0x02, 0x01]);
        assert_eq!(hash_image(&img, HashFunction::Sha1), expected);
        // Cross-checked independently against the system `sha1sum`/`openssl dgst`
        // on the raw byte sequence `\x02\x01`.
        assert_eq!(expected, "c92920944247d80c842eaa65fd01efec1c84c342");
    }

    #[test]
    fn hash_image_complex_hashes_both_components() {
        // A single ComplexFloat32 pixel: buffer_stride is 2 (real, imaginary),
        // so the hash covers both f32 components' little-endian bytes.
        let img = Image::new(&[1, 1], PixelId::ComplexFloat32);
        let components = img.component_slice::<f32>().unwrap();
        assert_eq!(components.len(), 2);
        let mut expected_bytes = Vec::new();
        for &c in components {
            expected_bytes.extend_from_slice(&c.to_le_bytes());
        }
        assert_eq!(
            hash_image(&img, HashFunction::Md5),
            md5_hex(&expected_bytes)
        );
    }

    #[test]
    fn hash_image_vector_hashes_all_components() {
        let img = Image::from_vec_vector(&[1, 1], 3, vec![1.0f32, 2.0, 3.0]).unwrap();
        let mut expected_bytes = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            expected_bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            hash_image(&img, HashFunction::Sha1),
            sha1_hex(&expected_bytes)
        );
    }

    #[test]
    fn hash_image_sha1_and_md5_differ() {
        let img = Image::from_vec(&[2, 1], vec![7u8, 9]).unwrap();
        assert_ne!(
            hash_image(&img, HashFunction::Sha1),
            hash_image(&img, HashFunction::Md5)
        );
    }
}
