# Original Mulligan: completed local DIVL stage

The original DIVL stage completed 150,000 updates on our accepted local IDQL
checkpoint. On the same 50 viewer starts, it scores **46/50 with one candidate
and 47/50 with 32 candidates**. The prior IDQL stage scores 46/50 and 49/50.
This small development cohort shows no improvement from the additional stage.

The [viewer](http://127.0.0.1:8770/original-methods/#mulligan-local-divl-heading)
keeps the local stages separate. It shows the first three fixed starts, selected
before evaluation. These three DIVL recordings all succeed. The full score still
includes the three N32 failures, at starts 6, 9, and 12.

## What changed during training

The unchanged [author implementation](https://github.com/ankile/mulligan) is
pinned at `bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8`. All 236 installed author
Python payloads match. The original `train_divl_heads.sh` command received the
absolute path of the accepted local IDQL final model. This avoids resolving a
different local parent. The selected recipe is `square-narrow-r01-mulligan`, seed 1.

The actor and its normalization data stay fixed. Only the critic and value
models train. Checkpoint inspection confirms 150,000 updates in their Adam
states, 186 finite model tensor leaves, and equality between final and rolling
weights. Actor and normalizer files exactly match the IDQL parent. No learner,
reward, action transform, sample rule, or author training loop was changed.

Producer `author-subset-20261014T010158-7967ea` completed in 4,814.36 seconds,
including its final monitoring evaluations. Original and canonical exits are
zero. Its owned process group is empty. The run ended at the author budget.
A resumed optimizer, replay, or simulator continuation was not tested.

## Measured outcomes

| Cohort and stage | One candidate | 32 candidates |
| --- | ---: | ---: |
| IDQL: original monitoring | 372/400 | 382/400 |
| DIVL: original monitoring | 365/400 | 376/400 |
| IDQL: matched viewer starts | 46/50 | 49/50 |
| DIVL: matched viewer starts | 46/50 | 47/50 |

The monitoring and viewer cohorts are different. Monitoring RNG histories also
differ between training stages. Those 400-episode totals are not a paired test
of the unchanged actor. For the matched viewer cohort, all 50 N1 rows match
IDQL exactly, including rewards and episode lengths.

Within DIVL, both candidate modes succeed on 45 starts. N32 alone succeeds on
two; N1 alone succeeds on one; both fail on two. Candidate count and ranking
change together. This does not isolate the critic effect.

At N32, IDQL and DIVL both succeed on 46 starts. DIVL fixes point 0 but loses
points 6, 9, and 12. The actor, evaluation starts, candidate count, and evaluator
seeds are held fixed across these two stages. Different chosen actions can
produce different subsequent observations and sampling histories. One training
seed and 50 development starts do not establish a reliable method difference.

The original evaluator uses the same first shard of the Sobol8k grid, ten
workers, seeds 42/42, and a 400-control limit. The requested 50-start shard is
complete. The full 8,000-start grid remains partial. N1 and N32 job durations
are 85.48 and 78.33 seconds; these are not per-action inference latency.

## Full recordings and checks

The separate recording pass uses the original synchronous evaluator and the
existing CPU rendering observer. All ten recorded result rows match the original
asynchronous N32 evaluator exactly. Every MP4 passes full CPU decoding.

| Fixed start | Outcome | Controls | Frames | Simulation playback |
| --- | --- | ---: | ---: | ---: |
| 0 | Success | 240 | 241 | 12.05 s |
| 1 | Success | 214 | 215 | 10.75 s |
| 2 | Success | 160 | 161 | 8.05 s |

Each video includes reset and every control frame at 20 simulation frames per
second. These durations do not measure wall-clock real-time execution. Success
ends the author's evaluation episode even when the environment's native `done`
flag is false. This is the original evaluator's rule.

The recording job completed in 188.34 seconds. Original and canonical exits
are zero. Recorded process identities have exited; the owned group is empty.
Checkpoint hashes remain unchanged through all evaluations. An initial launch
omitted the repository environment variable and failed before GPU admission;
its log is retained. The corrected launch uses the same method settings.

Evidence is under
`/home/user/aditya/RL/builds/mulligan-author-experiments-20261013/`:

- `local-divl-review/ACTUAL_COMPLETED_DIVL_CHECK.json`: original training and saved-state checks.
- `local-divl-publication/LOCAL_FROZEN_SCORES.json`: both complete matched cohorts.
- `local-divl-publication/LOCAL_NATIVE_VIDEO.json`: full videos, trace checks, and evaluator agreement.
- `local-divl-publication/FINAL_REVIEW.json`: independent verification and limits.

The [evidence index](MULLIGAN_LOCAL_DIVL_20261014.json) pins the accepted files.
Nine CPU DOM checks pass. The actual GPU-disabled browser verifies video hashes,
HTTP range playback, durations, seeking, refresh, accessible labels, equal local
clip widths, and mobile layout. Root inspected the published screenshot. An
initial browser check incorrectly expected six released video elements at once;
the existing selector displays two policies for one selected start. The corrected
check verifies all three selections against all six original video filenames.
No product change was needed for that test correction. The normally installed
wheel matches all 1,149 source payloads in both regression and canonical prefixes.
The report does not claim the historical full benchmark test suite passes.

## Consequence for the scaling study

The harness now executes both local author training stages and compares their
saved policies on matched starts. Extra value training did not improve this
small cohort. We should test the effect of data collection separately from
training longer on the same data. Recorded operator data are not a fresh
autonomous collection or human intervention policy.

The [scaling plan](DEPLOYMENT_RL_SCALING_PLAN.md) separates data quantity,
collection selection, optimizer work, and supervision cost. The original Panda
task does not establish R1 Lite performance.

Root owns execution and acceptance. The method reviewer checks author recipes
and evidence. The recorder observes the original evaluator. The UI agent shows
accepted scores and videos. The author programs own learning and control.
The [architecture](MULLIGAN_AUTHOR_EXPERIMENTS_20261013.mmd) shows the implemented
path and the proposed R1 Lite boundary. STE guidance was used; full ASD-STE100
compliance was not checked.
