# Original Mulligan command supervision

`abc_bench.mulligan_program` owns process lifetime and output paths. The released
Mulligan code owns training, sampling, policy execution, and evaluation.
No learner or rollout loop is copied into ABC.

The supported source is the pristine release
`bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8` at
`/home/user/aditya/RL/mulligan`. Use a normal author installation, such as
`/home/user/aditya/runtimes/mulligan-bc42da1`. Default editable `uv sync` is
insufficient for this binding: install the unchanged author wheel without
changing its locked dependencies. The metadata preflight compares every
installed `mulligan/**/*.py` payload with the pinned source before execution.

The config uses `schema_version=1` and these fields:

| Field | Meaning |
| --- | --- |
| `python` | Absolute author-runtime interpreter. |
| `source_root`, `source_commit` | Pristine checkout and exact released commit. |
| `mode` | `gpu` or separate `cpu-start-design`. |
| `entrypoint` | One original shell entrypoint, CPU start-design module, or pinned visual recorder. |
| `arguments` | Literal argument list for the original CLI. |
| `timeout_s` | Finite worker bound, at most86400 seconds. |
| `environment` | Optional HF paths/offline controls and task-local OSMesa library paths. |
| `binding_pin` | Required installed-file path, bytes and SHA-256 for the visual recorder. |

GPU shell entrypoints are `scripts/sim/eval_cell.sh`, `train_cell.sh`, and
`train_divl_heads.sh`. Use one explicit seed1..5 and an original Square-Narrow
IDQL/DIVL recipe. Preserve the original flags and recipes. Any recipe overrides
remain visible in the recorded argument list and need separate fidelity review.

The worker supplies `PYTHON` and calls the unchanged shell script with an argv
list. It runs from the pristine checkout. It creates a fresh `author-output`
directory and passes that directory through `MULLIGAN_OUTPUT_DIR`. Output/root
overrides, including `--system.checkpoint_dir`, are rejected. A DIVL job using
`--parent local` cannot find a previous job's output. Pass the actual previous
`final_model` directory through the author's explicit `--parent` option.

The optional `abc_bench.mulligan_video` binding belongs to a separate recorder
change. Its installed origin and source hash must match `binding_pin`. Its CLI
retains the original task/policy evaluator and adds declared recording. This
launcher alone does not qualify that native recorder.

For GPU execution, prepare a normal canonical job with
`module=abc_bench.mulligan_program` and `arguments=["--config", absolute_path]`.
Set `max_wall_s` to at least `timeout_s+60`, at most86460 seconds. Supply matching
finite admission credit and an explicit parent timeout through the existing
`python -m abc_bench.runner --algorithm author-subset` interface. Root owns
admission and launches. Existing module bounds are unchanged.

The worker retains the existing GPU0 lease contract and `MUJOCO_GL=disable`.
Only its original-command child gets `MUJOCO_GL=osmesa` and declared task-local
`LD_LIBRARY_PATH`. CUDA visibility remains the reserved GPU UUID. All children
remain in the canonical process group. Canonical cleanup reaps that group.
`IMAGEIO_FFMPEG_NO_PREVENT_SIGINT=1` keeps imageio's encoders in that group.
The worker records only direct-child cleanup; it does not claim descendant or
native cleanup solely from the original shell's exit.

Child W&B is offline. HF, LeRobot, Torch, Triton, and Inductor caches stay under
`/home/user/aditya`. Original shell temporary files stay under the job directory.
No author training hyperparameter changes are applied by the launcher.

CPU start design stays separate from GPU admission. Use the original public CLI:

```sh
CUDA_VISIBLE_DEVICES= /home/user/aditya/runtimes/mulligan-bc42da1/bin/python -I -B \
  -m mulligan.sampling.sim_design \
  --root /home/user/aditya/RL/mulligan \
  --config /home/user/aditya/RL/mulligan/configs/sim/square_narrow/r01_sampler.yaml \
  --out /home/user/aditya/RL/builds/your-fresh-start-design \
  --check
```

The worker's CPU mode admits only this module with `--check`. It supplies the
public `--root` argument because installed packages do not include the release
input tree. CPU mode hides CUDA and cannot enter the GPU job path.

The receipt task is `SquareNarrow`; the embodiment is `franka_panda`. A completed
command still records `experiment_accepted=false`, with empty measured metrics.
Root must authenticate checkpoints, complete native trial outcomes, reset
identities, videos, and recipe fidelity separately. Exit75, timeout, cancellation,
or another nonzero exit remains failed. The launcher never invents task results.

Focused installed CPU process fixtures and source/runtime metadata checks cover
this boundary. They are not simulation or learning evidence. Root owns release
and native qualification. The independent reviewer checks source and process
contracts. [Architecture](MULLIGAN_AUTHOR_PROGRAM.mmd) shows the boundary.
Text follows STE guidance; full ASD-STE100 compliance was not checked.
