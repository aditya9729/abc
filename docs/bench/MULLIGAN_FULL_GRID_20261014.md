# Original Mulligan: full locked-grid evaluation

Run `author-subset-20261014T135652-749c2a` completed all 8,000 Square-Narrow starts:
**7,429 successes, 571 failures, 92.8625% success**. The original evaluator and
canonical supervisor exited 0. GPU0 execution took 9,506.22 seconds, about
2 hours 38 minutes. The frozen checkpoint did not change.

[Open the results viewer](http://127.0.0.1:8770/original-methods/).
The full-grid card is separate from the earlier pilot and training monitors.

The [two original training stages](MULLIGAN_LOCAL_DIVL_20261014.md) each completed
150,000 updates. The earlier frozen comparison covers 50 starts. This run closes
the evaluation-coverage gap for the same local training seed.
It does not establish the paper's five-seed aggregate or a fresh collection loop.

| Measurement | Result |
| --- | --- |
| Local full grid, training seed 1, 32 candidates | 7,429 / 8,000; 92.8625% |
| Author's reported same-recipe seed-1 DIVL rate | 94.7125% |
| Local minus author reference | −1.85 percentage points |
| Mean episode reward / length | 0.928625 / 195.579375 controls |
| Covered grid cells | 80 / 80 |

The [author CSV](https://github.com/ankile/mulligan/blob/bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8/paper/data/sim/sim_significance/divl_frozen_actor_results.csv)
supplies the reference rate. This is a descriptive comparison, not a matched
historical rollout or a numerical-equivalence claim. Our checkpoint was retrained.
Our evaluation fixes policy/reset seeds at 42/42; the author defaults leave them unset.
The full grid includes the earlier 50 development starts. It is not wholly untouched test data.
We did not evaluate the local IDQL parent on this full grid, so this result alone
does not measure the full-grid effect of adding DIVL.

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

The worker had a four-hour watchdog. The parent allowed another 60 seconds for
cleanup. The job finished within that bound. Earlier GPU charges are preserved.

## Acceptance and handoff

The reviewed checker verified all 8,000 ordered, unique locked point identities,
finite outcomes, all 80 cell summaries, and the complete final resume file.
It recomputed aggregate values and verified the original post-close completion event.
All 236 source and installed author Python files and checkpoint files stayed unchanged.
The first 50 complete outcome rows exactly match the earlier pilot, including rewards and lengths.

Root checked all 14 recorded process identities and both owned process groups.
None remained. The GPU budget lock was available, its ledger had no active owner,
and GPU0 had no compute processes. No other GPU was queried or changed.

Configuration, input hashes, source/runtime preflight and live process identity
are under `/home/user/aditya/RL/builds/mulligan-local-full-grid-20261014/`.
The original outputs are under `outputs/bench/author-subset-20261014T135652-749c2a/`.
Session 64206 ended with exit 0. `ACCEPTANCE.json` binds the native data and resource checks.
The [evidence index](MULLIGAN_FULL_GRID_20261014.json) links their exact hashes.

After native acceptance, root installed the prepared normal wheel without dependencies.
All 1,149 source, wheel and installed payloads match, including 933 native assets.
Dependency versions did not change. Publication held the existing viewer lock.
Only the full-grid record, snapshot time and current viewer-release metadata changed.
The prior release metadata and all other results remain intact.

Actual browser checks passed for the new counts, seed and overlap labels, desktop/mobile
layout, existing selectors, and all 19 FastTD3 return rows. Three earlier recordings
passed HTTP byte checks, seeking, playback and refresh. All 42 existing video nodes
and positions survived refresh. Root inspected both screenshots and checked browser
process cleanup. This evaluation adds no rollout video; earlier clips remain pilot evidence.

Root owns execution and result acceptance. The independent reviewer checked the
original grid semantics and the config changes. No learner or simulator code changed.
The [publication architecture](MULLIGAN_FULL_GRID_VIEWER.mmd) shows the completed
author-evaluator, acceptance, package and viewer boundaries. The older
[training diagram](MULLIGAN_AUTHOR_EXPERIMENTS_20261013.mmd) retains its historical pilot label.
STE guidance was used; full ASD-STE100 compliance was not checked.
