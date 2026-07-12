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
    pub skipped: Option<String>,
}
