# Author recipe versus executed diagnostics

The original ABC-VLA placed all five bottles in one development scene. The
other zero-success rollouts do not test the papers' training claims. Root used
reduced data and training schedules to qualify integration, then evaluated those
diagnostic checkpoints. That experiment design did not establish task readiness.
The reports must keep that limit clear. Using original learner code alone does
not preserve its training recipe.

## Exact discrepancies

| Condition | Author source or prerequisite | Executed condition | Consequence |
| --- | --- | --- | --- |
| ABC-VLA | Original evaluation: flow 10, RTC prefix/lead 4, original default randomized reset; default five worlds | Same path on one world; five bottles placed at control 819 | The path works. One scene does not reproduce a published success rate. |
| pi0.5 SFT | EXPO ordinary pick example permits up to ten task demonstrations, configures 4,001 steps and selects checkpoint 2,000 for RL | One three-bottle TRAIN episode; one original update on a fixed batch of size one, anchor 200 | A falling fixed TRAIN loss proves optimization only. It does not establish a five-bottle policy. |
| Ordinary EXPO | Batch 64, UTD 20; ten completed online episodes before updates; pick launcher uses episode cadence and three original update calls per eligible episode | Batch 2, UTD 1; 32 controls, zero completed warmup episodes, one update | The collection and update schedule differs materially from the release example. |
| EXPO replay | Original learner must learn from the online replay stream under its declared mixing and success rules | Both sampled actor rows and both sampled critic rows came from the existing demonstration; 32 live rows were available | This update does not show learning from online transitions. |
| FastTD3 | HumanoidBench defaults: 128 worlds, batch 32,768, replay 51,200 slots per world, two critic updates per vector step, actor delay two | One ABC world, same batch and update defaults, 4,000 controls | Updates and sampled replay rows per new transition increase 128 times. This changes the learning regime. |
| FastTD3 reset coverage | ABC transfer requires a declared TRAIN distribution with geometry coverage | TRAIN fixed all shapes to bottle_0 and all scales to one; evaluation randomized both | The frozen rollout changes geometry as well as placement. It does not isolate learner quality. |
| ResFiT | Frozen base plus original residual learner and demonstration-derived action scaler | Single three-bottle demonstration supplied ranges that changed ABC base commands even at zero residual | The result cannot isolate the residual. Base command parity must pass first. |
| QF3 | Exact author release and initialization | Paper-derived local implementation; official implementation unconfirmed | Further QF3 experiments remain on hold. |

The EXPO pick script is an example for another robot and task. Transferring it to
ABC-YAM is an explicit task adaptation. Matching its schedule alone would not
reproduce the paper's reported task results. pi0.5 and ABC-VLA also use different
pretrained backbones. FastTD3 receives privileged state and learns from scratch.

## What the measurements establish

ABC-VLA completed one episode successfully. Frozen FastTD3, pi0.5 and EXPO each
completed the same captured scene with zero bottle placements. These are actual
measurements. They cannot establish that the published methods fail.

The pi0.5 input has one layout and three bottles. The evaluation has five bottles
and different geometry. No separate episode-level validation set was used.
The SFT stopping rule checked a fixed TRAIN batch, not validation performance.
More updates on that same batch would not correct the missing task coverage.

An adapter error also occurred: an extra mesh context prevented the first EXPO
update. The adapter-only fix was tested before the successful retry. Original
learner math remained unchanged. Successful transport, update and checkpoint
tests do not establish task learning.

Action clipping remains an explicit boundary change. pi0.5 clipped 4,566 gripper
coordinates. Post-update EXPO clipped 2,706 gripper and 50 joint coordinates.
These counts warrant trajectory inspection. They do not prove an action-unit
bug or explain the complete zero-success result.

## Correction sequence

1. Preserve the successful original ABC evaluation and its video. Use the
   original policy to collect full, successful five-bottle TRAIN demonstrations
   from the intended reset distribution. Save issued actions, actual state,
   camera frames, task judge outputs and complete randomization states. Label
   these as simulator-policy demonstrations, without human approval claims.
2. Verify the observation/action boundary by replay and tensor inspection. Check
   arm order, radians, gripper fractions, time alignment, image ordering and
   normalization on the tensors that actually reach the learner.
3. Use the original pi0.5 training code with shuffled task data. Separate entire
   demonstration episodes for validation. Save bounded checkpoints and stop on
   validation plateau or budget. Mark early stopping as an explicit schedule
   deviation. Show a successful full validation rollout before online RL.
4. Extend the EXPO adapter to express multiple completed episodes and the
   released ordinary collection/update schedule. Qualify batch 64/UTD 20 on
   GPU0 before a long collection job. Record the replay origin of sampled rows.
   Preserve original success rules and learner math. A reduced regime must have
   a separate name and cannot be called a reproduction.
5. Qualify FastTD3 on an author-supported environment first. Then supply parallel
   ABC collection with the intended replay pressure and reset diversity. The
   current one-world host cannot supply the author 128-world regime. Do not
   extend the deficient scratch pilot or silently add reward shaping.
6. Declare a small matched evaluation cohort before selecting a model. Use
   validation layouts for early stopping and separate final layouts for scores.
   Scene 20260511 has been reused for development. It is not a fresh final test.
   Keep all GPU jobs bounded, serial on GPU0 and outside Leela process groups.

No further GPU training was launched for this audit. The benchmark goal remains
unfinished. The first required result is a task-ready pi0.5 validation rollout,
not another evaluation of the one-update diagnostic checkpoint.

## Source and evidence pins

- Original ABC: `d0832d12651d1b260a652861a14648dc5f3660c7`.
- Original FastTD3: `229ed59bbf43ea2f7a2d5d90d1076314839944d7`;
  `fast_td3/hyperparams.py` and `fast_td3/train.py`. The HumanoidBench Stand
  horizon is 100,000 vector steps, distinct from the BaseArgs 150,000 default.
- Original EXPO: `21fe3d3b7d913c80836496817965932f49e0aedf`;
  `scripts/pick/finetune_droid.sh`, `scripts/pick/run_server.sh`, and
  `train_pi_robo.py` lines 49, 56, 389 and 409.
- Actual SFT: `author-subset-20261012T141113-882427/expoft/receipt.json`.
  The receipt records fixed anchors `[200, 200]`, batch one and one actual step.
- Actual EXPO TRAIN: `author-subset-20261012T161954-d3d16b`.
- Actual EXPO frozen: `author-subset-20261012T164835-f1fba0`.
- [Compact authenticated results](original_native_results_20261012.json).
- [Executed architecture](ORIGINAL_METHODS_ARCHITECTURE.mmd).
- [Actual video viewer](http://127.0.0.1:8770/original-methods/).

Root owns recipe selection, GPU execution and evidence acceptance. Method peers
check author sources and adapters. Independent review checks claims and retained
artifacts. A passed code review does not approve a scientific comparison.
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
