# ABC benchmark engineering instructions

Read `/home/user/aditya/AGENTS.md` and the shared coordination board first.

This checkout is the Sadhana-owned ABC benchmark. Use branch `staging_attempt1`. Preserve upstream behavior. Keep changes in `abc_bench`, `tests/bench`, and `docs/bench` unless a reviewed integration requires an upstream change.

Use `uv sync --frozen --extra dev`. Run `CUDA_VISIBLE_DEVICES= uv run python -m pytest tests/bench -q`. Run Ruff on benchmark files. Build the package and check its installed entry points.

The authorized campaign uses GPU 0 only. On 2026-10-08, the user said: "hi you can use gpu 0 for your experiments - please use it for as much time as you need". This removes the earlier cumulative time limit of 10,800 seconds. Preserve all earlier charges and record this authorization. The coordinator can allocate additional finite admission credit in the existing ledger for each bounded job without asking again. Its numeric `limit_seconds` is allocated admission credit under this uncapped authorization, not a new user time cap. Keep per-job watchdogs, exclusive GPU 0 leases and occupancy checks. Run GPU jobs through `python -m abc_bench.runner`. The update worker must hold a coordinator lease. Do not start simultaneous jobs, reset charges, touch other GPUs or disturb Leela's jobs.

Show only executed measurements. Separate smoke tests, method adaptations, benchmarks, and embodiment preflight. Keep missing measurements unavailable. Native YAM results are not R1 results. Do not label candidate selection alone as Real-Time EXPO-FT.

Keep downloaded assets, checkpoints, videos, and raw outputs outside Git. Commit the dependency lock and compact reproducibility evidence. Publish architecture diagrams and team explanations when goals finish. Follow ASD-STE100 guidance and state any missing compliance check.
