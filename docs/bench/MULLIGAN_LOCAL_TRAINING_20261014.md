# Local training with the original Mulligan recipe

The original Square-Narrow IDQL recipe completed on GPU0. Our saved policy
scores 46/50 with one action candidate and 49/50 with 32 candidates.
The [viewer](http://127.0.0.1:8770/original-methods/#mulligan-local-heading)
shows three complete episodes from this checkpoint, including its failure.

This is one seed on the original Franka Panda simulation task. It establishes
working local training and evaluation. It does not establish R1 Lite performance
or a scaling law.

## What ran

The unchanged [author source](https://github.com/ankile/mulligan) is pinned at
`bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8`. All 236 installed Python files match
that source. The [experiment report](MULLIGAN_AUTHOR_EXPERIMENTS_20261013.md)
records the runtime and three deployment dataset revisions.

The original command was:

```sh
bash scripts/sim/train_cell.sh square-narrow-r01-mulligan --seed 1 \
  --wandb.run_name=mulligan-r1-seed1-original-20261013
```

The extra run name enables the author's rolling checkpoint path. It does not
change the learner. The harness supplies process supervision, owned output paths,
GPU0 visibility, and a six-hour watchdog.

Training kept the original 150,000 updates and batch size 1,024. It started from
scratch on the accumulated round-1 data, rather than loading released R0 weights.
The actor uses 21,015 human-supervised frames. The critic/value view has 57,108
unique frames. Its original sampler mixes human-success and DAgger sources equally.
The recipe retains failures for value learning. It does not imitate all failures.

Run `author-subset-20261013T222807-f6985d` completed in 8,470.99 seconds, including
its final evaluations. Both original and canonical processes exited zero.
The owned process group is empty. Training stopped at the original budget.

## Two evaluation cohorts

Keep these cohorts separate. The first is the author's original training monitor.
The second matches the existing viewer's released-policy experiment.

| Cohort | One candidate | 32 candidates |
| --- | ---: | ---: |
| Original 400 monitoring starts | 372/400 · 93.0% | 382/400 · 95.5% |
| Viewer: fixed first 50 Sobol8k starts | 46/50 · 92% | 49/50 · 98% |

The viewer cohort uses policy/reset seeds 42/42, ten workers, and a 400-control
limit. It is one complete shard of the 8,000-point grid. The author marks the
full-grid result as partial; the requested 50-start shard is complete.

On those 50 starts, both modes succeed on 45. N32 succeeds on four N1 failures.
N1 succeeds on the one N32 failure, point 0. Candidate count and ranking change
together. This comparison does not isolate the value function's contribution.

The original frozen evaluator took 74.86 seconds for N1 and 66.15 seconds for N32.
These job times include different episode lengths and startup work. They are
not per-action inference latency measurements.

The viewer's released R0/R1 scores remain 34/50 and 46/50. Those checkpoints use
the later DIVL stage, while our current local checkpoint is IDQL. Equal start
identities do not make these identical training stages. Local DIVL is now running;
its result remains pending.

## Complete recordings and acceptance

The recording pass uses the original synchronous evaluator with CPU rendering.
All ten outcome rows exactly match the earlier asynchronous N32 evaluation.
The fixed first three starts were selected before observing local outcomes.

| Start | Outcome | Controls | Decoded frames | Playback |
| --- | --- | ---: | ---: | ---: |
| 0 | Failure | 400 | 401 | 20.05 s |
| 1 | Success | 167 | 168 | 8.40 s |
| 2 | Success | 158 | 159 | 7.95 s |

Each MP4 contains its reset frame and every counted control frame. Playback uses
20 simulation frames per second. These durations do not measure real-time control.
The original process and all three observed encoders exited. No owned group remains.

The original public loader restored all 184 finite model tensor leaves.
Actor, critic, and value Adam counters each report 150,000 updates.
Final weights exactly match the completed rolling checkpoint. Model and normalizer
hashes stayed unchanged through evaluation. A resumed training update was not tested.

Six installed export tests pass. The actual browser loaded, played, and sought all
three recordings. Refresh preserved playback. Desktop and mobile layouts passed.
Released-policy, RTC, and Catcher sections remain available. The historical full
benchmark suite still has its prior failures; this publication did not rerun it.

Evidence is under
`/home/user/aditya/RL/builds/mulligan-author-experiments-20261013/`:

- `local-parent-review/ACTUAL_COMPLETED_PARENT_CHECK.json`: original producer and checkpoint acceptance.
- `local-publication/LOCAL_FROZEN_SCORES.json`: native scores, paired outcomes, and original result hashes.
- `local-publication/LOCAL_NATIVE_VIDEO.json`: complete recordings, decoding, parity, and cleanup.
- `local-publication/VIEWER_BROWSER_RESULT.json`: actual browser checks.

The [compact evidence index](MULLIGAN_LOCAL_TRAINING_20261014.json) pins these files.
Root checked acceptance before each serial launch. Prepared job JSON alone does
not enforce that semantic gate. Independent review checked sources, jobs, grid,
seeds, and the viewer change. External review remains waived by the user.

## What this means for scaling

We can now train on recorded deployment data with the author's learner and inspect
the resulting policy. We have not collected a fresh round of human corrections.
We have not isolated targeted collection from additional data or optimizer work.

The original frozen-actor DIVL stage started at 01:01 UTC on 14 October as
`author-subset-20261014T010158-7967ea`. Its log confirms the accepted parent actor,
parent normalization, and critic/value-only updates. It retains the author's
150,000-update budget and final evaluations. Tool session 8594 owns the canonical
GPU0 job with a six-hour worker watchdog. Completion and frozen-actor acceptance
remain pending. No second GPU job is running.

The next comparisons must match their learner, training budget, and evaluation starts.
Record supervision cost as well as episode count. Keep final evaluation starts
outside future collection. The [scaling plan](DEPLOYMENT_RL_SCALING_PLAN.md)
separates SGS reset selection, Mulligan collection, and Real-Time EXPO updates.

Root owns execution and acceptance. The method reviewer checks recipe fidelity.
The launcher and recorder agents own process and video boundaries. Author code
owns learning and control. The [architecture](MULLIGAN_AUTHOR_EXPERIMENTS_20261013.mmd)
marks future DIVL and R1 Lite connections. This text uses STE guidance;
full ASD-STE100 compliance was not checked.
