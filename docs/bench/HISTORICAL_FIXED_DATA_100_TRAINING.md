# Save the original EXPO checkpoint before analysis

The previous diagnostic100 attempt completed 14 update calls before the planner cancelled it. Its call 10 measurement took 437.48 seconds. Repeated parameter scans and diagnostic forwards threatened the job deadline. No full checkpoint was saved. This was a budget and instrumentation problem, not an observed nonfinite learner or overfitting failure. Its artifacts remain preserved.

The separate `historical_online_fixed_data_100_training` phase removes these probes from the training job. It keeps the original Real-Time EXPO-FT learner, corrected TRAIN pool, success-only actor routing, sampler, RNG streams, batch 64, UTD 20, and full checkpoint API. It starts from RTC2000 and fresh seed 42 heads. It does not resume the cancelled state.

The worker checks initial integrity, restores replay, and performs 100 original update calls. Each call checks finite returned scalar metrics and exact optimizer step counters. The worker then saves the full step 100 checkpoint **before** the final integrity scan and input authentication. A saved checkpoint still needs independent acceptance.

Frozen-bank generation, terminal calibration, and repeated parameter hashes are planned for separate jobs after an accepted save. No such post-save job has run yet. This is an explicit fixed-data schedule adaptation. It does not execute the author's online collection loop.

| Contract | New training phase |
| --- | --- |
| Maximum original update calls | 100 |
| Consumed critic / actor rows | 128,000 / 6,400 |
| Original pending queue | One returned batch plus two queued critic batches |
| Intermediate full checkpoints | None |
| Final full checkpoint | Original `agent` and `params`, step 100 |
| Worker / canonical parent / outer bounds | 9000 / 9100 / 9200 seconds |
| Resource envelope | Same reviewed one-save floor and allowances |

The old 10-call and diagnostic100-call phases retain their schedules and deadlines. No original learner file changed. No VLA chunk smoothing was added.

The source passed 109 focused CPU tests and Ruff. Tests exercise the real job loader, legacy isolation, exact pending-batch retention, and all 100 shared-driver calls with explicit CPU fixtures. Diagnostic helpers must remain unused in that fixture. Another test verifies that a final scan failure preserves the already saved checkpoint. These tests do not establish numerical learning or simulator performance.

The estimated total is 118 minutes; the worker ceiling is 150 minutes. These are budget estimates. There is no automatic extension or validated generalization-based early stopping rule. TRAIN scores do not determine when to stop. Original frozen DEV evaluation must follow an accepted save before any improvement claim.

The planner owns installation, GPU0 admission, launch, and acceptance. The recipe agent owns this adapter and tests. The boundary agent independently reviews source fidelity and claims. The storage agent checks capacity. Installed qualification passed: 109 tests in each new prefix and the actual 9100-second job entrypoints. The GPU launch remains pending resource admission.

See [architecture](HISTORICAL_FIXED_DATA_100_TRAINING.mmd), [source](../../abc_bench/historical_fixed_data_100_training.py), and [frozen preparation](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-100-training-preparation-20261019/PLAN.json). The [sampler reference](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-100-training-preparation-20261019/SAMPLER_REFERENCE.json) contains the first 14 cancelled-run label signatures. Its first 10 exactly match the accepted C10 run. A new run must be compared separately; label hashes do not prove full observation/action tensor identity.

This explanation follows STE guidance. Full ASD-STE100 compliance was not checked.
