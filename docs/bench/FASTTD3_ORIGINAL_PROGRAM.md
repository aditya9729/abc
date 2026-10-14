# Original FastTD3 process supervisor

`abc_bench.fasttd3_program` runs the unchanged author `fast_td3/train.py`.
It owns process bounds and output paths. It implements no model, learner,
replay buffer, update, evaluation, or training loop. Root owns actual execution.

The selected current-release recipe is H1Hand Hurdle, seed1, classic FastTD3.
Fast source is `229ed59bbf43ea2f7a2d5d90d1076314839944d7`.
HumanoidBench source is `cb1189039151c8aadaaa987b442da54383c87fab`.
The default task uses100000 vector calls,128 worlds, batch32768, replay51200
rows per world, two critic updates, delayed actor updates, AMP and compilation.
No historical paper-run configuration or aggregate is reproduced by this claim.

## Canonical command and configuration

Root releases the candidate normally before launch. Use the existing coordinator:

```sh
python -m abc_bench.runner --algorithm author-subset \
--training-config /ABSOLUTE/REVIEWED/full.job.json --timeout-seconds 28860
```

The job selects module `abc_bench.fasttd3_program`, with arguments
`["--config", "/ABSOLUTE/REVIEWED/full.config.json"]` and `max_wall_s=28860`.
The coordinator owns the output and exclusive GPU0 UUID lease.
The full process configuration has exactly these fields:

```json
{
  "schema_version": 1,
  "python": "/home/user/aditya/runtimes/fasttd3-original-humanoidbench-229ed59-py310/bin/python",
  "source_root": "/home/user/aditya/RL/builds/fasttd3-pristine-author-readiness-20261013/source",
  "source_commit": "229ed59bbf43ea2f7a2d5d90d1076314839944d7",
  "humanoid_source_root": "/home/user/aditya/RL/humanoid-bench-author",
  "humanoid_source_commit": "cb1189039151c8aadaaa987b442da54383c87fab",
  "phase": "original_humanoidbench_train",
  "arguments": ["--env-name", "h1hand-hurdle-v0", "--seed", "1"],
  "timeout_s": 28800
}
```

The supervisor permits task, seed1..5 and an optional safe `--exp-name`.
It rejects learner, task, world-count, checkpoint and output-path overrides.
It does not add a stopping rule or change author defaults.
A separate `original_humanoidbench_qualification` phase permits exactly1 or12
original vector calls and a worker bound of1800seconds, with60seconds cleanup.
This phase is allocation or first-update evidence, not a benchmark.
The full job has an8hour worker bound and60seconds cleanup.
These are chosen watchdogs; completion within them is not yet measured.
Other workers retain their existing admission bounds.

## Source, imports and runtime

The worker checks both clean Git revisions and records tracked-file manifests.
It checks every normally installed author payload against source bytes through
public distribution metadata, without importing Torch or a simulator.
The complete Fast wheel has12 unchanged Python files, including its four
environment files. Packaging these files does not change the original source.
HumanoidBench has2780 unique namespace files:66 Python files and2714 other files.
The separate2738asset inventory includes24 Python files already counted above.

The child uses the absolute original interpreter and script:
`python -u -B /.../source/fast_td3/train.py`.
The standard script-directory import behavior supports the author's bare imports.
`-I` would remove this behavior, so it is deliberately absent for that child.
There is no `runpy`, `sys.path` edit, private adapter import, or `PYTHONPATH` injection.
The worker's metadata subprocess uses `-I -B` and normal distribution metadata.
Normal virtual environments can link to an external immutable base Python.
The worker checks their owned launcher directory and actual package prefix.

The original script forces EGL on Linux. The child preserves it and sets physical
`MUJOCO_EGL_DEVICE_ID=0` under the canonical GPU0 lease.
CUDA visibility remains the coordinator's exact single UUID.
OSMesa CPU probing is a separate profile and cannot qualify original EGL rendering.
Root must establish actual GPU0 EGL, native, compile and memory readiness first.
The short1/12-call phase never reaches the original5000-call rendering cadence.

Fast requires Torch2.6.0; HumanoidBench declares Torch2.3.1.
The selected normal runtime uses2.6.0+cu124 and retains that declared conflict.
Payload equality and CPU fixtures do not settle native dependency compatibility.
Runtime provenance also records installed JAX and JAXlib versions.
The authors do not pin these versions. Cached JAX0.4.28 rejected the original
`JAX_DEFAULT_MATMUL_PRECISION=highest` setting before CLI help.
Root owns a declared compatible runtime repair and actual original CLI checks.
The supervisor changes no author environment setting or import.

## Output, evaluation and cleanup

Each command gets a new owned `author-output` directory as its current directory.
Original `models/` paths therefore cannot overwrite source or an earlier run.
Offline W&B remains enabled, including original evaluation metrics and GIF output.
Tracking and cache paths stay under `/home/user/aditya`. The child inherits PATH.

Original periodic evaluation and saving remain5000 calls apart.
The100000-call recipe last evaluates and saves periodically at index95000.
It writes final weights with `global_step=100000`; it has no final evaluation.
Evaluation reuses the128 training worlds. A separate RGB world always exists.
Its checkpoint saves weights, normalizers, arguments and index.
It omits optimizer, scheduler, replay, RNG and native-world state.
`--checkpoint-path` is a warm restart, not exact continuation; it is not admitted here.

The worker publishes process-only receipts and an original stdout log.
Metrics remain empty and `experiment_accepted=false`, even after exit zero.
Root must separately authenticate checkpoint contents, W&B returns, videos and cleanup.
Nonzero exits, cancellation and timeout remain failures. Partial files are preserved.
The worker stops and reaps its direct child. The unchanged canonical runner owns
the process group and kills surviving descendants before terminal publication.
Native detached-service cleanup requires actual root inspection.

## Verification and ownership

Tests use installed wheels and explicit CPU process or metadata fixtures.
They cover direct argv, fresh output, UUID/EGL metadata, unchanged PATH,
path-injection rejection, nonzero exit, timeout, SIGTERM, source changes,
normal payload completeness, and separate full/qualification admission bounds.
They establish no native task score. The first installed test attempt exposed
the normal virtual-environment symlink case; its failure log is preserved.

This isolated Git worktree contains only the supervisor, narrow coordinator
admission, tests and documentation. Its neutral test wheel has27 tracked ABC
assets. It is not a complete ABC native asset release. Root builds the final
combined package from main and preserves its full asset inventory.
No live source, prefix, GPU job, ledger or viewer changed during preparation.

Root reviews, releases and executes the command. The runtime agent qualifies
dependencies and assets. The recipe reviewer checks original task defaults.
This agent implements process supervision and installed CPU checks.
The [architecture](FASTTD3_ORIGINAL_PROGRAM.mmd) shows those boundaries.
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
