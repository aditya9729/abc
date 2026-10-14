# Original Mulligan: full locked-grid evaluation

Run `author-subset-20261014T135652-749c2a` is evaluating the accepted local DIVL
checkpoint on all 8,000 Square-Narrow starts. It launched on GPU0 at 13:56 UTC.
The full result is pending. No further training is running.

The [two original training stages](MULLIGAN_LOCAL_DIVL_20261014.md) each completed
150,000 updates. The earlier frozen comparison covers 50 starts. This run closes
the evaluation-coverage gap for the same local training seed.
It does not establish the paper's five-seed aggregate or a fresh collection loop.

## Exact experiment

The unchanged author source is `bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8`.
All 236 normally installed author Python files match that source.
The original command is:

```sh
bash scripts/sim/eval_cell.sh square-narrow-r01-mulligan \
  --seed 1 --stage divl_head \
  --checkpoint "${ACCEPTED_MULLIGAN_DIVL:?Use the accepted local final_model}" \
  --num-action-samples 32 --eval-seed 42 --env-seed 42
```

The recorded config supplies the absolute accepted checkpoint path.
The harness launches this command through its existing canonical GPU0 runner.
It owns process lifetime and output paths. The author code owns evaluation.

The run omits the pilot's `--grid-file`, `--shard`, and `--n-shards` overrides.
The original script generates and checks its locked grid. The generated manifest
has 8,000 points and the expected hash
`45a9963f5592f495477c44beabe8d3c3e3173e920c2b2ba009c1e69532e53b17`.
It uses the original ten asynchronous environments and 400-control limit.
The explicit evaluation and reset seeds remain 42/42. The upstream defaults are
unseeded, so exact agreement with historical reference outcomes is not assumed.

The worker has a four-hour watchdog. The parent allows another 60 seconds for
cleanup. The prior 50-start runtime projects to about 3.5 hours for 8,000 starts,
including repeated startup cost. This is an estimate, not a completion promise.
Existing GPU admission credit covers the bound; earlier charges are preserved.

## Acceptance and handoff

Accept a full result only after all 8,000 unique locked point identities exist.
Check complete outcomes, finite rewards and lengths, and unchanged checkpoint bytes.
Check original and canonical exits, descendant cleanup, and released GPU ownership.
Compare the first 50 outcomes with the earlier matched pilot. Disclose differences.
Keep this full-grid result separate from the 400-start training monitors.

Configuration, input hashes, source/runtime preflight and live process identity
are under `/home/user/aditya/RL/builds/mulligan-local-full-grid-20261014/`.
The original outputs are under `outputs/bench/author-subset-20261014T135652-749c2a/`.
The live tool session is 64206. A live receipt is not final acceptance evidence.

Root owns execution and result acceptance. The independent reviewer checked the
original grid semantics and the config changes. No learner or simulator code changed.
The [existing architecture](MULLIGAN_AUTHOR_EXPERIMENTS_20261013.mmd) shows this
author-evaluator boundary; its 50-start label describes the earlier published pilot.
STE guidance was used; full ASD-STE100 compliance was not checked.
