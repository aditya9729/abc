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

## Native qualification — 14 October 2026

The original Hurdle runtime now passes GPU0 EGL rendering, one vector collection,
and the first learning update with the original 128-world configuration.
These checks establish execution. They report no convergence or task-success score.

| Check | Actual result | Canonical time |
| --- | --- | ---: |
| GPU0 EGL | Eight controls, nine finite states, two decoded RGB frames, zero warnings, clean close | See native receipt |
| One original vector call | Final index 1; finite saved tensors; observation count 128 | 122.18 s |
| Twelve original vector calls | Final index 12; finite saved tensors; observation count 132,608 | 172.93 s |

Both training checkpoints retain all 61 expected configuration fields. The actor
noise buffer has 128 rows; its input and output dimensions are 151 and 61.
The twelve-call result follows the original first-update branch at index 11.
Source control flow implies two critic calls and one actor call. The original
checkpoint does not save optimizer counters, and scalar loss logging starts
later. Those calls are source-derived, not measured optimizer counters.

The first allocation attempt failed before world construction because the
supervisor's long TMPDIR exceeded the AF_UNIX socket limit. The
[short-path repair](FASTTD3_SHORT_TEMP_REPAIR.md) changes only launcher storage.
A real CPU forkserver regression reproduces the old failure and verifies the
repair. All 63 selected root installed checks pass. Original author sources and
task settings remain unchanged. Both successful native groups have exited and
released the canonical lease. The first EGL attempt also finished cleanly but
missed live process capture; a separate repeat supplies the accepted identity.

Evidence: `builds/fasttd3-original-native-acceptance-20261014/actual-egl-r2-acceptance.json`
and `builds/fasttd3-short-temp-repair-20261014/next-acceptance/actual-{one-r2,twelve}-acceptance.json`.
The [compact index](FASTTD3_NATIVE_QUALIFICATION_20261014.json) pins these receipts.
Full 100,000-call training is a separate experiment. Root launched it on GPU0
after these qualifications and the serial Mulligan evaluations. Its result
remains pending. The original eight-hour worker watchdog remains active.

## Live full run and reference comparison — 14 October 2026

Run `author-subset-20261014T030724-7fc63f` uses the unchanged current-release
recipe above. Its process remains live. The immutable 04:29 UTC history snapshot
contains 597 scalar rows through index 59,700 and 11 evaluations through 55,000.
Mean return rose from 7.44 at index 5,000 to 91.71 at 45,000, then declined to
89.50 at 50,000 and 82.01 at 55,000. These are original periodic measurements,
not a completed benchmark or a success rate.

[Partial reference plot](/home/user/aditya/RL/builds/fasttd3-released-reference-comparison-20261014/reference-comparison.png)
compares these observations with the authors' released Hurdle curve. At the
same nominal 7.04 million transition coordinate, our observed mean is 82.01;
the released aggregate is 377.23. The gap is present in these data. Its cause
has not been isolated. The plot includes every local evaluation and does not
fill the remaining training interval with predictions.

The paper reports three FastTD3 runs on an A100 with 16 CPU cores. This attempt
uses one seed on the available host. The current default ends after 12.8 million
training transitions; the released reference extends to 19.84 million nominal
steps. The released JSON contains no per-seed configurations, generating source
revision or runtime metadata. This descriptive comparison is not a controlled
revision comparison. [Paper setup](https://arxiv.org/html/2505.22642v3#S3),
[pinned released data](https://github.com/younggyoseo/FastTD3/blob/229ed59bbf43ea2f7a2d5d90d1076314839944d7/data/humanoidbench_result.json).

The source history also contains later author fixes: `d2fc7e9` passes configured
exploration bounds into the actor, and `7acc3a3` changes categorical projection
at exact support positions. Both are included in the selected release. They
show why the method name alone cannot identify a historical experiment. They
do not explain this performance gap. The author's README suggests disabling
compilation when reproduction is difficult. We have preserved this run's
original default and have not launched that comparison.

The 04:38:20 UTC stdout snapshot contains **eight printed QACC warnings** and
continues through index 65,325. In MuJoCo 3.1.6, that condition resets internal
physics state. Gym termination is separate, so an episode can continue across
the discontinuity. Warning suppression makes the printed count a lower bound
on detected recoveries. The log does not identify affected worlds or phases.
The warning after the 65,000 evaluation marker could concern evaluation or
the separate render rollout. [Versioned MuJoCo implementation](https://github.com/google-deepmind/mujoco/blob/3.1.6/src/engine/engine_forward.c).

The saved scalar snapshot through 59,700 is finite. Later/full checks remain
pending. Finite tensors and complete history do not establish uninterrupted,
warning-free physics. Keep the warnings visible beside any eventual scores.
No fatal traceback or inspected nonfinite scalar requires an immediate stop;
the original bounded run continues unchanged. This is not evidence that the
recoveries were harmless or that every measurement is invalid.

[Live observation evidence](FASTTD3_LIVE_REFERENCE_20261014.json) pins the local
comparison, original sources and independent stability assessment. Raw source
diffs and the plot are under `builds/fasttd3-released-reference-comparison-20261014/`.
The stability assessment is under `builds/fasttd3-original-stability-observation-20261014/`.
Both paths are relative to this repository's parent. Terminal acceptance and
the final result remain separate pending steps.

## Canonical command and configuration

Root releases the candidate normally before launch. Use the existing coordinator:

```sh
ABC_BENCH_REPO=/home/user/aditya/RL/abc python -m abc_bench.runner --algorithm author-subset \
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
The prepared runtime now uses JAX/JAXlib 0.4.35. The original CLI exits zero.
The original one-world wrapper passes CPU OSMesa reset, eight zero controls, rendering,
and close. These checks do not qualify the original EGL training path.
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

Root integrated the supervisor and runtime provenance changes into main.
The full normal wheel has 1149 authenticated source/wheel/installed payloads,
including 914 assets. It is installed in an idle regression prefix.
All 86 selected installed supervisor, render-probe, admission and viewer tests pass.
Actual source/runtime metadata checks authenticate 12 Fast and 2780 HumanoidBench
payloads. Prepared 1-call, 12-call and full-job configurations pass admission.
Ruff and the patch whitespace check pass. The earlier full-suite failures remain
unresolved; this is not a green full-suite claim.

Root release evidence is under
`/home/user/aditya/RL/builds/fasttd3-original-root-release-20261014/`.
`NORMAL_WHEEL_AUTHENTICATION.json` identifies the wheel and installed prefix.
`ACTUAL_INSTALLED_METADATA.json` records source/runtime checks and three job pins.
That initial release preceded the native qualifications described above.
Mulligan has since released GPU0. Rendering, allocation and first-update checks
passed before the full run started. The canonical and Fast author prefixes now
remain unchanged while that original run is active. Its terminal artifact
acceptance and final report are pending. The separate
[EGL check](FASTTD3_EGL_RENDER_QUALIFICATION.md) records the rendering boundary.

Root reviews, releases and executes the command. The runtime agent qualifies
dependencies and assets. The recipe reviewer checks original task defaults.
This agent implements process supervision and installed CPU checks.
The [architecture](FASTTD3_ORIGINAL_PROGRAM.mmd) shows those boundaries.
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
