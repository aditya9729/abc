# Original R1 Lite WBC program supervisor

`abc_bench.wbc_program` runs the unchanged feedback fitter and gated pose PPO from `r1pro-rl-body-controller` commit `c437c7183aee296b973bb706432b92e8a1ae1f16`. It is a process adapter, not a learner. It also admits the original nominal evaluator and the exported-policy evaluator. The original simulator, reward, action composition, optimizer and export code stay unchanged.

Root launches this worker through `python -m abc_bench.runner --algorithm author-subset --training-config JOB.json`. The existing canonical GPU0 lease, strict occupancy check, ledger and process-group cleanup remain authoritative. Workers cannot acquire or override a lease. Worker deadlines are at most 7,100 seconds; both the job and effective parent deadline must include a further 100 seconds. Root retains job admission and scientific acceptance.

A schema1 worker configuration has exactly these fields:

```json
{
  "schema_version": 1,
  "python": "/home/user/aditya/RL/builds/r1pro-wbc-local-pilot-20261016-01/prefix/bin/python",
  "source_root": "/home/user/aditya/RL/r1pro-rl-body-controller",
  "source_commit": "c437c7183aee296b973bb706432b92e8a1ae1f16",
  "program": "pose-ppo",
  "arguments": ["--config", "/absolute/recipe.toml", "--model", "/absolute/r1_lite.xml", "--references", "/absolute/train", "--nominal-policy", "/absolute/nominal/policy.pt", "--normalization-dataset", "/absolute/teacher/aggregate/dataset.json", "--output", "{output}", "--device", "cuda:0", "--seed", "0"],
  "timeout_s": 3600,
  "input_manifest": "/absolute/INPUTS.json",
  "expected_summary": {"status": "complete", "iterations_this_run": 250, "transitions_this_run": 32768000, "optimizer_steps": 5000, "nominal_state_unchanged": true}
}
```

Paths above are examples; nonexistent inputs refuse admission. `INPUTS.json` has `files`, a list of unique absolute `path` and `sha256` pairs. Pin every original and installed `src/**/*.py`, the interpreter, selected script, and all direct input files. Input directories are checked recursively. Root must also pin external mesh files and the teacher manifest's referenced source and episode files. The original dataset loaders enforce their own provenance checks. The worker authenticates pins before and after execution and requires a clean original checkout. The current runtime retains Torch2.11.0+cu128; the original lock selected Torch2.14.1. It is a tested dependency variant, not a lock reproduction.

Supported programs are `feedback-fit`, `feedback-evaluate`, `pose-ppo`, and `exported-policy-evaluate`. The two training programs require `--device cuda:0`; the evaluators require `--device cpu`. Each uses only its declared original arguments. `--output {output}` becomes the new `author-output` directory inside the canonical result. The original program creates that directory. No shell or arbitrary module is accepted. The exported-policy script retains its normal script directory for its original sibling import; other programs use isolated Python mode.

The worker checks 20GiB minimum free space and a 5GiB combined result/cache ceiling before launch, every 0.2 seconds, and after completion. This is a sampled guard, not an atomic quota. A violation cancels the direct child; the canonical parent reaps the whole process group. Caches remain inside the run. Failed exits, timeouts, input mutation and summary mismatches remain failed receipts. A completed receipt means the original process and expected update accounting completed; it does not assert a control or task improvement.

The R1Lite tracking policy controls nine joint positions and three base velocities at 50Hz. This is a distinct interface from EXPO's fixed-base bilateral14D manipulation policy. Connecting them requires a declared adapter and matched evaluation. The supervisor does not add smoothing, interpolate chunks, or alter either interface.

The focused CPU tests use explicit subprocess fixtures, with no learner, simulator or GPU. They exercise source and input rejection, lease refusal, finite deadlines, parent cleanup margin, actual child exit and timeout, disk refusal, post-execution checks, and summary acceptance.

[Architecture source](R1LITE_WBC_PROGRAM.mmd). Root selects the experiment and owns the GPU job. The adapter runs the original program and records evidence. A separate reviewer checks that evidence. Root accepts the result only after the control comparison is complete. This explanation uses simple technical English; no formal ASD-STE100 compliance check was performed.
