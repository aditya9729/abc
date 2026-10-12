# Original ABC policy demonstration collection

The collector records commands from the original ABC-VLA policy in the original
sequential ABC simulator. It does not train a policy. It does not add a
SimTaskEnv.obs, render, task evaluation, physics step, policy inference, action
clip, or reset to the rollout. It adds pure native get_state reads for telemetry
and native state/model serialization after reset. Original public call order,
parameters, and commands remain unchanged.
The author source reference is `d0832d12651d1b260a652861a14648dc5f3660c7`.

The code is in `abc_bench/demo_collect.py`. The diagram source is
[DEMONSTRATION_COLLECTION.mmd](DEMONSTRATION_COLLECTION.mmd). The source CPU
tests use explicit doubles and unchanged author rollout function bodies. Native
collection, installation, and physical replay need separate qualification. A
model-generated successful trajectory has no human, Atlas, or dynamics approval.

## Run and recipe

The planner launches this worker through the canonical subset coordinator. A live
GPU0 lease is required before native imports. Direct worker launch does not obtain
a lease. The coordinator supplies finite wall time and cancellation bounds.

The worker entry point is:

```sh
python -I -m abc_bench.demo_collect --config /absolute/config.json --output /absolute/new-worker-output
```

The output can contain only the coordinator's regular `job.json` and `run.log`
before entry. Previous payloads, receipts, directories, and symlink bookkeeping
are rejected. The worker never changes a model download or an installed prefix.

The configuration has six fields:

| Field | Contract |
| --- | --- |
| `schema_version` | Integer1 |
| `checkpoint` | Existing absolute file pin: path, bytes, SHA256 |
| `metadata` | Original checkpoint metadata pin, including the actual model architecture |
| `sim_eval` | All keys from `abc_bench.upstream_eval.RECIPE` |
| `split` | `train`, `validation`, or `development` |
| `max_output_bytes` | Positive integer, at most5GiB |

The author settings remain flow10, loaded policy horizon30, execute15, RTC
prefix4/lead4, stock fast inference with max-autotune, CPU MuJoCo physics,
MJWarp cameras168x224, and a maximum3540 controls. The cohort has1–3 worlds.
Only the cohort size, seed, and disclosed count5 reset request can differ.

Development uses only seed20260511, one world, and original `randomization=None`.
It is a reference for checking the recorder. It is not TRAIN data. The original
default bottle sampler chooses2–6 bottles; this seed's actual scene must still
contain five bottles for admission.

TRAIN and validation use the author-compatible JSON string for:

```json
{"bottle_count":5,"randomize_variants":true,"randomize_scales":true}
```

This is a five-bottle subset of the original distribution. Bottle variants,
scales, object positions, and bin positions remain random. Validation requested
seeds are9100–9102. Their original mapped retries109100–109102 remain validation.
TRAIN excludes both validation families and development20260511. An actual reset
must use its requested seed or the original +100000 fallback. Development cannot
resample its exact reference seed.

The recorder preserves carried arm poses, the original same-seed retry after
`forget_arm_state`, and the original fallback after two placement failures. It
records every attempt and only opens an episode after a successful delegate reset.
Each reset includes the actual initial robot state, not just the random object
layout. Episodes finish before the next reset.

## Observation and action chronology

With `video_every_n_actions=1`, reset supplies raw frame0. The author's existing
post-control video render supplies frame1 through frameT. The cached frame at
control index`t` is the image before command`t`. Numeric state at the same index
is the actual public14D policy state. There are T commands and T+1 states/images.
Raw camera arrays are copied before video encoding. Decoded MP4 frames are not
used as training images.

The14 channels are `left_j1..left_j6,left_grip,right_j1..right_j6,right_grip`.
Joint commands are radians. Grip commands are fractions; the unchanged native
env multiplies them by0.0475m. The collector records the actual float32 command
passed to `step_one`. It does not clip or normalize the command again. The native
float64 `data.ctrl` buffer is also recorded. This buffer is the actuator input;
it is not measured force or a claim about internally limited effective control.
Native actuator limit flags and ranges are included.

RTC samples a policy observation before control11 in the first15-control chunk.
The next request commits the current chunk's actions11–14 as its prefix. The
original RTC manager removes output positions0–3. The next chunk starts at
full sampler output position4, at physical control15. This repeats every15
controls. The recorder links every issued row to the exact original sampler
output row. It retains bounded small output arrays and checks shared memory and
unchanged values. At most two observation objects remain live in recorder memory.

Policy-request RGB, state, prompt, noise, prefix, outputs, observation control
index, and simulation time are recorded separately. The policy can use an image
from an earlier control than the image saved beside the issued command. Those
are distinct observations. The recorder checks their byte fingerprint and does
not replace either one. Internal self-bound compile/RTC warmups stay unchanged
and are excluded from the production request trace. Predictions from an inference
that finishes after success are recorded but are never exported as issued actions.

Simulation time is float64 native `data.time`. Request start/end clocks use a
monotonic wall clock. Creation time is UTC. These clocks are not interchangeable.
The extra file I/O changes wall time. Original per-control video rendering also
changes RTC overlap telemetry. This collection cannot establish real-time latency.

## Episode files and admission

Each manifest uses `sadhana.abc.policy-demonstration/v1` and a globally unique
collection/episode ID. Every artifact pin contains an absolute path, byte size,
and SHA256. Numeric and RGB artifacts also record dtype and shape.

| Artifact | Storage and shape |
| --- | --- |
| `states` | `.npy`, float32 `[T+1,14]` |
| `issued_actions` | `.npy`, float32 `[T,14]` |
| `sim_times` | `.npy`, float64 `[T+1]` |
| `qpos`, `qvel` | `.npy`, float64 `[T+1,Nq]`, `[T+1,Nv]` |
| `actuator_ctrl` | `.npy`, float64 `[T,Nu]` |
| `rgb24` | Raw uint8 RGB `[T+1,168,672,3]`; top, left, right panels |
| `request_rgb24` | Raw uint8 RGB `[requests,168,672,3]` for exact sampler inputs |
| `policy_requests` | JSONL: sampler arguments, outputs, RGB offsets, observation fingerprints |
| `action_provenance` | JSONL: control index, inference ID, output row index |
| `task_judge` | JSONL: T rows `{control_index:1..T,sim_time,task_eval}` |
| `randomization` | Full original RandomizationState JSON |
| `initial_integration` | `.npy`, float64 native `mjSTATE_INTEGRATION` vector |
| `scene_xml`, `compiled_model` | Actual reset scene XML and compiled MJB |

All `.npy` files use `np.save(...,allow_pickle=False)`. Numeric rows are bounded
by the author horizon. RGB is streamed; there is no episode-sized image list.
Duplicate public evaluations at one control do not add judge rows. The initial
task evaluation has its own manifest field. The final task evaluation is the last
original post-control value.

The reset group ID is exactly:

```text
SHA256(ASCII(scene_xml.sha256 + ':' + initial_integration.sha256))
```

This binds the group to native model and integration state. The MJB captures
compiled model fields that XML alone can omit. The integration spec and installed
software versions are recorded. These files make replay possible; they do not
prove deterministic replay. A same-version replay comparison is still required.
See the [MuJoCo state API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-getstate)
and [reproducibility limits](https://mujoco.readthedocs.io/en/stable/computation/index.html#reproducibility).

A manifest is admitted only when the completed original summary agrees with
the captured seed, reset state, T controls, and exact final task evaluation, and
the current final state has all five active bottles in the bin. Historical best
progress alone cannot pass. Termination and full-horizon truncation are separate.
A correct but unsuccessful rollout can have a completed worker and `admitted=false`.
Partial or failed traces cannot be admitted. All approval fields start false.

Producer lineage pins the collection config, checkpoint, checkpoint metadata,
and software/source manifest. It does not refer to the worker receipt. The worker
receipt binds the final episode manifests and original summary. The canonical
parent then binds the worker receipt. This order prevents circular hashes.

Every recorder-owned payload is checked against the byte cap before writing.
Up to128KiB is reserved for manifests and the final receipt. The original FFmpeg
writer and XML API are not intercepted; their files are checked at recorder
boundaries and at final admission. Compiled MJB size is checked before saving.
The cap is therefore not a strict pre-write filesystem quota for external MP4/XML
operations. Any detected overrun fails the worker and cannot publish accepted data.

The planner publishes a new, immutable TRAIN/validation cohort with:

```sh
python -I -m abc_bench.demo_cohort --runs <TRAINrun> <VALrun> --output <newcohort.json>
```

The publisher verifies the canonical parent, all artifact hashes, original
success, sampler linkage, and whole-episode split isolation. Atlas/Engram review
and task training are separate consumers. The EXPO reader consumes the installed
public artifact format. There is no R1 hardware result in these native YAM traces.

## Checks and team handoff

The collector agent owns this recorder, its tests, and this diagram. The planner
owns the cohort publisher, job configs, wheel installation, GPU admission, and
native runs. The EXPO agent owns the dataset reader and learner. The independent
reviewer checks code and evidence. No reviewer approval changes dataset labels.

Run the source CPU checks with the existing ABC prefix:

```sh
CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled .venv/bin/python -m pytest tests/bench/test_demo_collect.py -q
.venv/bin/ruff check abc_bench/demo_collect.py tests/bench/test_demo_collect.py
.venv/bin/ruff format --check abc_bench/demo_collect.py tests/bench/test_demo_collect.py
```

For relocated installed tests, set `ABC_BENCH_REPO` to the explicit pinned ABC
source checkout. This supplies only the unchanged author function bodies for CPU
doubles. Production code never injects a source path. The tests check call parity,
RTC lead/prefix indices, terminal stop, T+1 raw frames, action provenance, reset
retries, split guards, byte-cap failure, resource close, and scoped patch restore.

This explanation uses short sentences and ASD-STE100 guidance. No formal language
compliance review is claimed. Native qualification and task dataset quality remain
separate measured gates.
