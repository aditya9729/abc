# ABC benchmark campaign

Owner: Codex / Sadhana. All files remain under `~/aditya/RL/abc`.

Use [QF3 native progress](QF3_NATIVE_PROGRESS.md) for the current ABC-VLA implementation and measured results. Use the [CPU comparison reader](vla_comparison.md) to validate retained 50-episode evidence. It requires a real selected checkpoint and a new complete learned evaluation before it reports a comparison. The ABC-DiT examples below describe the earlier smoke and adaptation pilots.

This campaign uses one GPU with a cumulative 10,800-second allowance. The initial allowance was 7,200 seconds; the user added one hour. Preserve prior charges. The additional two-hour request remains pending. The coordinator reserves GPU 0. The global lock and ledger live in `outputs/bench`. They apply even when a run has a different results directory. Job cancellation stops the complete subprocess group. Before a new job, the runner checks GPU 0 occupancy. It refuses active compute or graphics memory, records the physical GPU UUID, and sets child visibility to that UUID. This read-only check does not stop other jobs. The shared lease and coordination reservation still apply; telemetry is a snapshot, not an atomic lock against other projects.

New jobs use the CUDA MJWarp renderer on logical device 0. The child has `MUJOCO_GL=disable` and one CPU thread per numeric library. The worker refuses missing or mismatched physical UUIDs and classic OpenGL. Classic EGL can select devices independently of CUDA visibility. Historical receipts retain their original camera settings and scores. A fresh matched evaluation is required to compare results after a renderer change.

The sustained ResFiT route uses the installed Sadhana package, `nirvana-rl-harness`. Its `nrh.abc_training` module connects the native binding to the existing repeated-update trainer. Install a reviewed harness wheel in this ABC environment with `uv pip install --python .venv/bin/python --no-deps /absolute/path/to/nirvana_rl_harness-0.4.0-py3-none-any.whl`. Run `python -m abc_bench.runner --algorithm sustained-resfit --training-config /absolute/pinned_config.json --timeout-seconds 300`. Configuration and admission are described in Sadhana's `harness/docs/ABC_TRAINING_CLI.md`. Collection exports unreviewed data. Training requires a separate accepted review bound to its exact data and execution evidence. The outer deadline includes model loading and rendering setup. The campaign ledger applies to both stages. Expected trainer budget stops retain partial counts and checkpoint references; they do not count as completed targets.

## Reproduce

```sh
uv sync --frozen --extra dev
uv run prepare.py --sim-task put_plastic_bottles_in_bin --checkpoint
uv run python -m abc_bench.runner --chunks 2 --worlds 1 --video
uv run python -m abc_bench.runner --chunks 236 --worlds 3
uv run python -m abc_bench.runner --algorithm qf3 --timeout-seconds 600
uv run python -m abc_bench.runner --algorithm resfit --timeout-seconds 600
uv run python -m abc_bench.runner --algorithm comparison --qf3-state outputs/bench/<qf3-run>/adaptation.pt --resfit-state outputs/bench/<resfit-run>/adaptation.pt --timeout-seconds 1800
uv run python -m abc_bench.dashboard --results-root outputs/bench --port 8766
uv run python -m abc_bench.dashboard --results-root outputs/bench --export outputs/bench/dashboard.html
```

Use the coordinator command for GPU updates. Do not call the GPU update worker directly. The worker checks its parent's campaign lease. Imports do not initialize hardware or start a job. The dashboard cannot launch training or robot commands.

## Evidence and limits

Native policy evaluation uses the released bottles checkpoint and the upstream evaluation entry point. The default baseline uses ten Euler steps. It disables compilation and RTC. Cold calls remain in the logged latency samples. Terminal chunks can lack a latency record. No real-time deadline claim follows from these measurements.

A short rollout has no published task-success result. A full-horizon pilot has an episode count and success count. The count remains visible. Three episodes do not establish a reliable population success rate. Compare only runs with matched conditions.

QF3 update smoke uses the actual pretrained ABC-DiT head. Its rank-four output adapter and fitted immediate-return critic are declared adaptations. ResFiT update smoke uses frozen features and an invertible action codec. Neither smoke establishes a faithful paper reproduction or convergence. Real-Time EXPO-FT still needs an ABC bridge. See `algorithms.md` and `update_smoke.md`. The matched evaluator loads their saved snapshots and compares them with the frozen policy. It creates a fresh environment for each method and seed, verifies initial joint and object positions, and uses the same noise and action cadence. See `paired_eval.md`.

R1 Lite is the active transfer target. It has a static asset inspection receipt. Its ABC task adapter and compatible learned policy remain unverified. Missing metrics remain unavailable. See `embodiments.md`.

Large checkpoints, downloaded scenes, videos, and raw run receipts stay outside Git. Commit the source pin, dependency lock, verification notes, and compact campaign summary. Each run links its execution log and hashed summary.

A standalone architecture diagram is in [architecture.svg](architecture.svg).

## Architecture

```mermaid
flowchart LR
    C[Codex coordinator: budget controller] -->|Bounded subprocess control| E[ABC native simulation and released policy]
    C -->|Bounded update control| T[QF3 or ResFiT update smoke]
    C -->|Matched evaluation control| M[Frozen and saved adaptation pilots]
    E -->|Measured data| R[Run receipts and artifacts]
    T -->|Measured data| R
    M -->|Measured data| R
    R -->|Read-only data| U[Local dashboard]
    V[Independent reviewer and advisory Jev] -.->|Verification| C
    P[EXPOFT and R1 adapters: planned] -.-> E
```

I coordinate the runtime and measurements. The algorithm agent checks the objectives. The embodiment agent checks model contracts and connects the update smoke. The UI agent displays evidence. The independent reviewer checks failure behavior and claims. Jev supplies advisory judgments.

This explanation uses ASD-STE100 guidance. A full vocabulary and grammar compliance check has not been performed.

Installed runner entry points use `ABC_BENCH_REPO` to select this checkout. The simulator requires its downloaded checkout assets. Lightweight dashboard imports need no GPU runtime.
