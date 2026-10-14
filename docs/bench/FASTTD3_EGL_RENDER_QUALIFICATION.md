# Original FastTD3 native EGL render check

This candidate adds one separate phase to `abc_bench.fasttd3_program`:
`original_humanoidbench_egl_render`. Root must release and run it through the
existing canonical GPU0 lease. No native or GPU execution was done during
preparation. The original FastTD3 and HumanoidBench sources stay unchanged.

The original training program forces EGL. Its one-call and twelve-call checks
cannot exercise the default 5,000-call render interval. This small check uses
the installed public HumanoidBench Gym environment. It does not start the
training program or replace any learner. A passing result only proves the
one-world native render boundary. The original 128-world allocation, replay,
AMP, compilation and learning checks remain separate.

The check has fixed inputs: `h1hand-hurdle-v0`, public `rgb_array` rendering,
eight float32 zero actions of shape61, and no optimizer calls. It uses the
original wrapper's public `seed(0)` call, then normal reset behavior. It does
not claim matched random starts. Observations must have shape151. The probe
records all native warning counts, finite INTEGRATION state plus qpos, qvel,
ctrl, act and time at reset and after each control. It records the original
RGB pixels at reset and after control8. Early termination, nonfinite values,
warnings, failed rendering or failed close reject the check. Native rewards
are logged only as boundary observations, not a task score.

Before constructing a native world, the helper reads the existing canonical
ledger and held lock. It requires a live coordinator ancestor, GPU0 and its
single visible UUID. This is a read-only copy of the published ancestry/lock
contract, with EGL in the child instead of the coordinator's `disable`
profile. It does not reserve resources or change the ledger.

The public [EGL device enumeration extension](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_enumeration.txt)
permits device queries before EGL initialization. The helper checks device
index0 before any world or display construction. It uses
[EGL_CUDA_DEVICE_NV](https://registry.khronos.org/EGL/extensions/NV/EGL_NV_device_cuda.txt)
to obtain the CUDA device handle. It queries that handle's UUID and PCI bus
with the public CUDA driver API, then compares both with physical GPU0 from
`nvidia-smi` and the canonical UUID. Unsupported queries and mismatches fail.
There is no alternate device, software renderer or OSMesa fallback.

After each native frame, the helper reads the current display and context.
The public [EGL device query extension](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_query.txt)
maps the current display back to its device. That device must be the same
preflight handle, UUID and PCI bus. This avoids inferring EGL identity from
CUDA visibility alone. The C binding uses pointer-width `EGLAttrib`, as the
extension specifies. The installed PyOpenGL declaration uses an int32 pointer;
the probe does not change that module. It binds only public query functions.
CUDA identity references: [device UUID](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html)
and [PCI bus ID](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html).

The helper is a separate normal distribution, `fasttd3-egl-probe==0.1.0`,
with one Python module. Root installs its reviewed wheel into the idle author
prefix with `--no-deps`. This avoids installing ABC's conflicting simulator
and Torch requirements there. The supervisor checks the distribution version,
module bytes, SHA256, RECORD and normal import origin with standard metadata.
No source path injection, private renderer access or author patch is used.

The new config keeps the original source/runtime fields. It requires
`arguments: []`, `timeout_s <= 300` and an exact `probe_pin` containing
`distribution`, `version`, `sha256` and `bytes`. Its canonical parent bound is
at most360 seconds with at least60 seconds beyond the worker wait. Existing
training and allocation-check limits are unchanged. The supervisor sets EGL,
physical device index0 and PyOpenGL's EGL platform only for this new phase.
It preserves the canonical UUID. It reaps its direct child. The existing
canonical runner owns final process-group cleanup and charging.

Future release and run commands, after root review and an idle GPU0:

```sh
uv pip install --offline --no-deps --python /home/user/aditya/runtimes/fasttd3-original-humanoidbench-229ed59-py310/bin/python /home/user/aditya/RL/builds/abc-fasttd3-egl-render-qualification-20261014/wheels/fasttd3_egl_probe-0.1.0-py3-none-any.whl
ABC_BENCH_REPO=/home/user/aditya/RL/abc /home/user/aditya/RL/abc/.venv/bin/python -I -B -m abc_bench.runner --algorithm author-subset --training-config /home/user/aditya/RL/builds/abc-fasttd3-egl-render-qualification-20261014/egl-render.job.proposal.json --timeout-seconds 360
```

Root first merges and normally installs the supervisor candidate in its
coordinator prefix. The neutral CPU ABC wheel in this build has only27
tracked model assets. It is suitable for CPU CLI tests, not ABC native runs.
The actual Hurdle probe uses all original normally installed HB assets.

Output `author-output/egl-render/EGL_RENDER_RESULT.json` has a distinct render
qualification kind. It includes device evidence, file hashes, native state,
RGB and cleanup status. A failed or partial result stays failed. The process
receipt has no score and keeps `experiment_accepted=false`. Root must inspect
the real receipt and owned-process cleanup before accepting the boundary.
Even a passed boundary leaves `training_ready=false` and `world128_ready=false`.

CPU verification uses installed wheels in a separate neutral prefix. Tests
use explicit query, native and process doubles. They verify import safety,
pointer widths, device failures, held-lock admission, native warning handling,
eight-control lifecycle, failed close, timeout and cancellation. They do not
prove actual EGL callability, device mapping, native pixels or GPU cleanup.

I prepared the code and CPU checks. Root reviews the candidate, releases the
wheels, and runs the canonical job. The runtime agent owns the original
dependency and asset checks. The method reviewer checks the author recipe.
The [diagram](FASTTD3_EGL_RENDER_QUALIFICATION.mmd) shows these boundaries.
This text follows STE guidance. Full ASD-STE100 compliance was not checked.

## Root integration status

Root merged the candidate and built the full ABC wheel from main.
All 1,149 source/wheel/installed payloads match, including 914 assets.
The helper is normally installed in the idle original author prefix.
Its import initializes no native library or simulator. Actual probe metadata
and canonical job admission pass without hardware calls.

The final installed selection passes 86 checks. Five optional Catcher UI DOM
checks also pass. Main-repository Ruff and whitespace checks pass.
The first selected run passed 85 checks and exposed one test assumption:
a timed-out child can exit before writing its own render receipt.
The test now permits that valid failure path and still requires failed status
and a stopped child. The earlier log is retained. No production timeout or
cleanup behavior changed. Historical full-suite failures remain unresolved.

Evidence is in
`/home/user/aditya/RL/builds/fasttd3-egl-root-release-20261014/`.
The full wheel is installed only in the idle regression prefix.
The live canonical prefix remains unchanged while Mulligan owns GPU0.
No native EGL render, 128-world qualification or Fast training score is accepted.
