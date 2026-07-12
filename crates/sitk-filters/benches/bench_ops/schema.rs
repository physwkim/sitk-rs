//! One row of `doc/bench-spec.md`'s output schema, one object per
//! `(op, size, config)`.
use serde::Serialize;

#[derive(Serialize)]
pub struct Row {
    pub harness: &'static str,
    pub op: &'static str,
    pub size: &'static str,
    pub voxels: u64,
    pub config: &'static str,
    pub threads: u32,
    pub ms_mean: Option<f64>,
    pub ms_median: Option<f64>,
    pub ms_stddev: Option<f64>,
    pub samples: Option<u32>,
    pub input_checksum: Option<String>,
    pub output_checksum: Option<String>,
    /// `doc/bench-spec.md` §"Correctness gate — not optional": GPU rows are
    /// not held to `output_checksum` bit-parity, so the gate reports these
    /// two against the CPU result instead. `None` for every `t1`/`tN` row and
    /// for any `gpu` row that did not produce a result.
    pub max_abs_err: Option<f64>,
    pub max_rel_err: Option<f64>,
    pub skipped: Option<String>,
}
