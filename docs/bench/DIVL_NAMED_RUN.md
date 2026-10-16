# Full DIVL rolling-save identity

The full R1 DIVL command adds `--wandb.run_name=r1-divl-<canonical run id>`.
The fresh canonical output directory supplies the identity. For example,
`outputs/bench/author-subset-.../training` yields `r1-divl-author-subset-...`.

The original trainer requires a non-null name and a positive resume-save
frequency to enable rolling saves. The full recipe already specifies 5,000
updates. Its missing name previously disabled optimizer, RNG and sampler saves.
The new argument enables that original path, including the final 150,000 state.
The same requirement applied to the earlier
[original local training run](MULLIGAN_LOCAL_TRAINING_20261014.md).

Only full-training mode with stage `divl` receives this argument. IDQL and
short qualification commands keep their previous behavior. A conflicting name
in the full DIVL recipe is refused. The worker retains `WANDB_MODE=offline`.
Learner, sampler, losses, update budget, evaluator and GPU admission are unchanged.
The name does not implement cross-job resume or select an earlier checkpoint.

The production module matches the installed, CPU-qualified named-run release.
Its 106-case qualification checked the original Draccus parser and exact original
rolling-save gate without running training, a model or native physics.
The installed tests passed in 38.82 seconds. All 53 wheel payloads match source.
The new canonical prefix uses 1.4 MiB and borrows existing dependencies read-only.
All 249 payloads and seven dependency metadata records in the old prefix remain unchanged.
[Release receipt](/home/user/aditya/RL/builds/other-methods-plan-20261015/divl-named-run-canonical-release-01/RELEASE.json).
The new repository tests use self-contained command fixtures; they need no
preparation-tree snapshot or original learner installation. Existing recipe and
parser integration suites retain their separately prepared local inputs.

After actual full training, inspect the fixed final model and completed rolling
state. Verify critic/value counters, frozen parent actor/normalizer and original
loader consistency. Process, ledger and monitor acceptance remain separate.
The named-run qualification establishes no full DIVL completion or task score.

[Architecture](DIVL_NAMED_RUN.mmd). This explanation uses STE guidance;
full compliance was not checked.
