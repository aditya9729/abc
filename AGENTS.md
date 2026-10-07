# ABC benchmark engineering instructions

Read `/home/user/aditya/AGENTS.md` and the shared coordination board first.

This checkout is the Sadhana-owned ABC benchmark. Use branch `staging_attempt1`. Preserve upstream behavior. Keep changes in `abc_bench`, `tests/bench`, and `docs/bench` unless a reviewed integration requires an upstream change.

Use `uv sync --frozen --extra dev`. Run `CUDA_VISIBLE_DEVICES= uv run python -m pytest tests/bench -q`. Run Ruff on benchmark files. Build the package and check its installed entry points.

The authorized campaign uses GPU 0 only. Its total budget is 7,200 seconds. Run GPU jobs through `python -m abc_bench.runner`. The update worker must hold a coordinator lease. Do not start simultaneous jobs or reset the ledger to bypass the budget.

Show only executed measurements. Separate smoke tests, method adaptations, benchmarks, and embodiment preflight. Keep missing measurements unavailable. Native YAM results are not R1 results. Do not label candidate selection alone as Real-Time EXPO-FT.

Keep downloaded assets, checkpoints, videos, and raw outputs outside Git. Commit the dependency lock and compact reproducibility evidence. Publish architecture diagrams and team explanations when goals finish. Follow ASD-STE100 guidance and state any missing compliance check.
