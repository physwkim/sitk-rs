use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{CompileOptions, compile_ptx, compile_ptx_with_opts};

use crate::error::CudaError;

/// Device 0. Multi-GPU is a later wave; every path in this crate names the
/// device through this constant so the widening is one edit.
const DEVICE_ORDINAL: usize = 0;

/// How a kernel source is allowed to round: with NVRTC's default multiply-add
/// contraction, or without it. Part of the module cache key — see
/// [`Backend::function_exact`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Arith {
    /// NVRTC's default: `a*b + c` may become one FMA, rounding once.
    Fused,
    /// `--fmad=false`: every multiply and every add rounds, as on the host.
    Exact,
}

/// A CUDA device, its context, its stream, and its compiled-module cache.
///
/// One per process, reached through [`backend`]. Construction is the *only*
/// place that can fail for environmental reasons (no driver, no device); once
/// a `Backend` exists, the device is known good.
pub struct Backend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// NVRTC is a ~100 ms compile per kernel source. Cache keyed on the source
    /// text itself: the same source is the same PTX on this device, and a
    /// changed source is a different key with no invalidation step to forget.
    modules: Mutex<HashMap<u64, Arc<CudaModule>>>,
}

impl Backend {
    /// The error is the driver's own reason, unwrapped: [`backend`] is the one
    /// place that dresses it as a [`CudaError::NoDevice`], so the reason cannot
    /// pick up that prefix twice on its way to the caller.
    fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(DEVICE_ORDINAL).map_err(|e| e.to_string())?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            modules: Mutex::new(HashMap::new()),
        })
    }

    /// The device's name, e.g. `"NVIDIA RTX 5000 Ada Generation"`.
    pub fn device_name(&self) -> Result<String, CudaError> {
        Ok(self.ctx.name()?)
    }

    /// The device's compute capability as `(major, minor)`.
    pub fn compute_capability(&self) -> Result<(i32, i32), CudaError> {
        use cudarc::driver::sys::CUdevice_attribute as Attr;
        let major = self
            .ctx
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self
            .ctx
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok((major, minor))
    }

    /// The stream every op in this crate uses.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Block until the stream has drained. Timing phases are separated with
    /// this, so a phase's measured cost is its own.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(self.stream.synchronize()?)
    }

    /// Load `name` from `src`, compiling `src` with NVRTC on first use and
    /// serving the cached module afterwards.
    pub fn function(&self, src: &str, name: &str) -> Result<CudaFunction, CudaError> {
        self.compile(src, name, Arith::Fused)
    }

    /// [`function`](Self::function), but compiled with **no
    /// multiply-add contraction** (`--fmad=false`).
    ///
    /// By default NVRTC fuses `a*b + c` into a single FMA, which rounds *once*
    /// where the host rounds twice. That is more accurate and entirely fine for a
    /// kernel whose contract is a tolerance — the metric's is ~1e-12 — but it makes
    /// bit-identity to a host expression impossible.
    ///
    /// The pyramid ops need bit-identity, not accuracy: their job is to land a
    /// level's fixed image where `execute` lands it. With contraction off, a
    /// `double` expression on the device performs the same multiplies and the same
    /// adds, in the same order, with the same rounding as the Rust it was
    /// transcribed from — so `sitk_filters::recursive_gaussian` and its kernel
    /// agree to the last bit rather than to a tolerance that would then have to be
    /// argued about level by level.
    pub fn function_exact(&self, src: &str, name: &str) -> Result<CudaFunction, CudaError> {
        self.compile(src, name, Arith::Exact)
    }

    fn compile(&self, src: &str, name: &str, arith: Arith) -> Result<CudaFunction, CudaError> {
        let mut hasher = DefaultHasher::new();
        src.hash(&mut hasher);
        // Part of the key: the same source compiled both ways is two different
        // modules, and serving one for the other would silently undo the contract
        // `function_exact` exists to provide.
        arith.hash(&mut hasher);
        let key = hasher.finish();

        // Two threads racing on the same uncached source both compile; the
        // second's insert wins and the loser's module is dropped. That costs
        // one redundant NVRTC call in a race and keeps the lock off the
        // compile, which can take ~100 ms.
        if let Some(module) = self
            .modules
            .lock()
            .expect("module cache poisoned")
            .get(&key)
        {
            return Ok(module.load_function(name)?);
        }
        let ptx = match arith {
            Arith::Fused => compile_ptx(src)?,
            Arith::Exact => compile_ptx_with_opts(
                src,
                CompileOptions {
                    fmad: Some(false),
                    ..Default::default()
                },
            )?,
        };
        let module = self.ctx.load_module(ptx)?;
        self.modules
            .lock()
            .expect("module cache poisoned")
            .insert(key, Arc::clone(&module));
        Ok(module.load_function(name)?)
    }
}

/// The process-wide backend, or the reason there is none.
///
/// Initialized once, on first call: a machine with no driver, no device, or a
/// driver too old to load gets [`CudaError::NoDevice`] here — every time, with
/// no retry storm and no panic — and every op falls back to the CPU. The probe
/// is not repeated per call, so the GPU-less case costs one driver query for
/// the life of the process.
pub fn backend() -> Result<&'static Backend, CudaError> {
    static BACKEND: OnceLock<Result<Backend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(Backend::new)
        .as_ref()
        .map_err(|reason| CudaError::NoDevice(reason.clone()))
}
