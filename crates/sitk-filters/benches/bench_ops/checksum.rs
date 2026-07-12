//! FNV-1a 64 over a pixel buffer's little-endian bytes — `doc/bench-spec.md`'s
//! `input_checksum`/`output_checksum`. This is the field the spec calls "the
//! linchpin": the C++ (ITK) harness must reproduce `input_checksum` exactly,
//! so the byte order (little-endian, via `to_le_bytes`) and the hash
//! constants below are not incidental — they are the contract.
use sitk_core::PixelBuffer;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Every `PixelBuffer` variant reduces to the same shape: its little-endian
/// component bytes, in buffer order. `u8`/`i8` included via `to_le_bytes()`
/// too (a one-element array) so every arm is textually identical.
pub fn checksum_buffer(buf: &PixelBuffer) -> u64 {
    match buf {
        PixelBuffer::UInt8(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::Int8(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::UInt16(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::Int16(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::UInt32(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::Int32(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::UInt64(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::Int64(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::Float32(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
        PixelBuffer::Float64(v) => fnv1a64(v.iter().flat_map(|x| x.to_le_bytes())),
    }
}

/// `0x`-prefixed lowercase hex, matching the schema's example literals.
pub fn checksum_hex(value: u64) -> String {
    format!("0x{value:016x}")
}
