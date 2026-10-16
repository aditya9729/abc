# Historical-reward Real-Time EXPO-FT: fixed-data A/B

This driver tests whether the original Real-Time EXPO-FT learner can use one
additional autonomous TRAIN episode. It also saves a full final checkpoint for
later evaluation. **This source publication establishes no learning gain.** The
added episode passes the original historical judge but fails the stricter placement
footprint check. It is not an accepted corrective demonstration.

The initial publication copied the qualified canonical runtime and tests, with a
local-artifact availability guard in the historical test. The later
[corrected replay-order condition](HISTORICAL_ONLINE_FIXED_DATA_ORDER.md) changes
two helper files and adds six historical cases. It preserves the old A/B paths.
No original EXPO or OpenPI learner source changes. See the original
[architecture](HISTORICAL_ONLINE_FIXED_DATA.mmd) and the
[corrected-order diagram](HISTORICAL_ONLINE_FIXED_DATA_ORDER.mmd).

## Matched conditions and schedule

| Condition | Restored rows | Added rows | Critic sampleable starts | Actor sampleable starts |
| --- | ---: | ---: | ---: | ---: |
| A: `baseline_16968` | 16,968 | 0 | 16,960 | 7,069 |
| B: `plus_autonomous_349` | 16,968 | 349 | 17,309 | 7,410 |

Both start from RTC checkpoint 2000 and seed 42. The restored data contains
7,069 demonstration rows and 9,899 timeout rows. Both use the original
`RealTimeEXPOFTLearner`: batch 64, UTD 20, horizon 16, replan 8, delay 5,
32 base candidates, 32 edit samples, filter N 32, filter-edit 1 and edit scale 0.1.
Offline ratio is zero. Actor sampling stays success-only, without HIL-only routing.

The named `historical_online_fixed_data_10` phase calls the original update ten
times. This gives 200 critic/filter/encoder updates and 10 base-actor/editor/
temperature updates. It consumes 12,800 critic examples and 640 actor examples.
The original prefetch path generates 16,640 critic and 704 actor examples; the
remainder is not consumed. Measurements use the existing fixed TRAIN windows and
candidate banks at calls 0, 1 and 10.

**Ten calls on fixed data are a schedule adaptation.** This is not the author's
online episode schedule or a reproduction of its reported task result. The
original sampler, update, losses, transforms and one-batch-ahead routing remain
in use. Adding rows also exposes eight previously excluded timeout-tail starts.
Equal seeds and update budgets do not imply equal sampled batches. The report
does not claim to know the consumed rows' source indices.

## Added experience and labels

The source is canonical run `author-subset-20261016T010107-003fa5`: TRAIN seed
21000, policy seed 42, 349 controls and 350 saved RGB/state boundaries. It used
the frozen RTC base-bypass policy. All rows are autonomous, with `is_hil=False`.
Its R1 Pro episode 12/reference 11 lineage is shared with the 21 original
demonstrations. This adds an episode, not an independent task origin.

Row `j` uses measured state and RGB from boundary `j`, then the **applied** command,
reward, mask and terminal flag from boundary `j+1`. The loader checks the complete
causal metadata before insertion and reads images lazily. All 78 clipped controls
use the recorded applied command. Raw actions stay float64 until the original
R1 transform converts them. Object state and lineage do not enter actor inputs.

The original `BatchProcessor.on_episode_start`, `insert_transition` and
`on_episode_done(True)` own insertion, action-chunk backfill and episode success
marking. The last row has reward 1, terminal true and bootstrap mask 0. Earlier
rows have reward 0 and mask 1. A partial append aborts the fresh worker before
sampling. Actor and replay normalization must agree for all added current actions
and states.

The episode's final footprint has known 36.346 mm and 10.080 mm overhangs. Keep
its original historical reward unchanged and keep that geometric failure visible.
The separate strict-teacher admission path remains unchanged and has no admitted
complete correction. Legacy result keys `recovery_data` and
`recovery_inputs_after` contain explicitly labeled historical-policy metadata in
this phase; the key names do not confer teacher status.

## Cumulative change from `staging_attempt1`

The base commit is `ffc7d74d2cb1857e0ca2bd8f72777d88d0d5400a`. This publication
includes qualified changes that previously existed only in build preparations:

- `realtime_update_probe.py` adds closed phase dispatch and transfers ownership
  of the learner state to the fixed-data worker. Its default one-update path stays
  available.
- `fixed_data_learning.py` releases obsolete state and batch references while
  retaining the original pending batch and prefetch queue. It adds data-admission
  hooks and an optional final-checkpoint callback. Its learner math does not change.
- `recovery_fixed_data_learning.py` retains the separately qualified strict
  admission path, including connected-trajectory validation. It is unchanged from
  the qualified historical A/B release.
- `historical_fixed_data_learning.py` adds the causal autonomous episode loader,
  original insertion checks and final checkpoint callback.

The copied tests preserve ownership, causal-data, admission and save-failure
regressions. The existing probe tests are already identical to the qualified copy.
Package metadata, dependency versions, other algorithms and unrelated files stay
unchanged.

## Full checkpoint and evaluation boundary

After updates and prefetch shutdown, the callback uses the original
`initialize_checkpoint_dir` and `realtime_expo_ft.save_checkpoint`. It saves the
full `agent` and `params` items at **step 10**, waits for completion, then closes
the manager. It refuses an existing output directory. An asynchronous save failure
cannot produce a completed phase report.

Step 10 matters: the original `eval_droid_policy.py` treats step 0 as fresh
initialization. A later frozen evaluation must restore step 10 in a separate
process through that original evaluator. This snapshot does not include replay,
sampler or environment restart state, so it is not an exact training-resume claim.

The prepared evaluator uses `only_base_actions=False` and the original task
factory. Fresh initialization, A and B use the same DEV starts 9100–9103, after
discarding constructor reset 9099. These are reused development starts; FINAL
10000–10099 remains reserved. Report historical task scores and strict footprint
diagnostics separately. Losses, Q ranks or a successful checkpoint save do not
establish better task performance.

## Qualification and operation

The [installed CPU release](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-release-01/CPU_RELEASE.json)
passed 91 focused cases and checked source/wheel/installed identity. A separate
[actual transform check](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-transform-qualification-execution-01/ROOT_ACCEPTANCE.json)
passed all 349 rows through original transforms and a fresh small original replay,
including future-action backfill. That check performs no learning and does not
restore the full 16,968-row replay.

Six historical test cases use pinned local capture artifacts or authenticated
author-method ASTs with explicit CPU storage doubles. They skip only when a
required local file is missing. Malformed data, hash mismatches and access errors
still fail. The four metadata-refusal cases and two checkpoint save/wait/close
cases remain runnable without those artifacts. This test-only portability guard
is a new publication change; the prior 91-case qualification predates it.
Root verified all 12 historical cases with local artifacts present. With a
deliberately absent artifact directory, six cases passed and six skipped.
[Publication qualification](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-publication-qualification-01/RESULT.json)
records both checks. The doubles do not establish an actual Orbax restore.

Use the [concrete preparation and configuration templates](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-preparation-20261016/README.md)
and [root command handoff](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-preparation-20261016/ROOT_COMMANDS.json).
Root binds installed payload hashes and fresh checkpoint destinations, installs
normal packages in the existing runtimes, and launches serial bounded jobs through
the canonical GPU 0 runner. No copied runtime or `sys.path` change is needed.
The worker limit is 2,300 seconds; the parent allows 2,400 seconds with cleanup.

An equivalent full author checkpoint occupies 15.156 GiB. Two final checkpoints
with 25% headroom need about 37.89 GiB; this is an estimate, not a hard bound.
Retain only final A/B snapshots and reuse common initialization. Preserve the
original full-save semantics and check free storage before admission.

Publish through the `benchmark` remote (the user fork), not `origin` (amazon-far).

The source agent prepares this integration. The independent reviewer checks its
frozen source. Root owns qualification, runs, checkpoint restore and result
acceptance. This explanation follows STE guidance; full ASD-STE100 compliance
has not been checked.
