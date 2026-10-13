# Original Mulligan experiments

We are testing the released Square-Narrow experiment before adapting its method
to R1 Lite. The author learner, policy, controller, task and evaluation loop
remain unchanged. This experiment uses a Franka Panda and state observations.
It is not a VLA or an R1 Lite result.

## Fixed sources and data

The [author repository](https://github.com/ankile/mulligan) is pinned at
`bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8`. Its clean checkout is
`/home/user/aditya/RL/mulligan`. Its frozen dependencies are installed in
`/home/user/aditya/runtimes/mulligan-bc42da1`. The project was then installed
as an unchanged normal wheel. All 236 installed author Python files match
the source. Python is 3.12.13; Torch is 2.11.0+cu128; MuJoCo is 3.3.7.

The first evaluation uses released seed-1 DIVL checkpoints:

| Policy | Hugging Face revision |
| --- | --- |
| `mulligan/sim-square-narrow-r00-sobol-divl` | `62d773f964724264e2fecbf4fb6a2d2480ada8da` |
| `mulligan/sim-square-narrow-r01-mulligan-divl` | `93356d9ead01cc93ee3e7f1d1cb9ea7e1f51f1f7` |

Both complete checkpoints are downloaded. Their policy files match the
published hashes. Together they occupy about 579 MB.

The original round-1 training uses three complete LeRobot datasets. Their files are
in `/home/user/aditya/.cache/huggingface/hub/datasets--mulligan--*`.

| Source | Revision | Episodes | Frames |
| --- | --- | ---: | ---: |
| `sim-square-narrow-c00-teleop-sobol` | `e01d0970ed8522d3c5c210e88a040dc3864a421b` | 100 | 16,773 |
| `sim-square-narrow-c01-dagger-mulligan` | `76143af3be6192a3c69063f99e590c18db379bdc` | 100 | 18,151 |
| `sim-square-narrow-c01-sobol-policy-rollouts` | `dcdb0d1a7ad234fed48e95209bc55b95f237c48a` | 100 | 22,484 |

These datasets contain state/action Parquet files at 20 Hz, with original
source, validity, success, reward and termination fields. The correction set
also has intervention labels. They contain no images or videos. All recorded
7-dimensional actions and 9-dimensional robot states are finite. Keep the
environment-state features and the author's normalization rules too.
The data total is 13.17 MB. Failed policy episodes remain in the original data.
The author's actor and critic masks determine which samples enter each update.

## Evaluation and training protocol

The first pilot compares the released R0 and R1 policies on the same 50 starts,
with 32 candidate actions. Both use shard 1 of 160 from the original Sobol8k
manifest and explicit policy/reset seeds 42/42. Both have ten environment
workers and the original 400-step horizon. The original R0 default uses a
different grid. Its shared-grid override is deliberate and recorded.

The small video manifests were generated with the author's function and seed.
Their point payloads exactly match the beginning of the full manifest.
The round-0 video pass evaluates 50 starts. The round-1 pass evaluates ten.
Both record complete episodes for points 0, 1 and 2. Both preserve ten workers
and the original batches, seeds and maximum horizon.
The [video observer](MULLIGAN_AUTHOR_VIDEO.md) uses the original synchronous
evaluator and CPU OSMesa rendering. Video duration follows simulation time.
It does not measure wall-clock real-time execution.

This is a one-seed, 50-start pilot. It cannot establish the full paper score.
The author's reference CSV reports 76.525% for R0 and 94.7125% for R1 on their
respective full grids. Those are author measurements, not local measurements.
The R0 and R1 change includes additional data and its collection procedure.
It does not isolate the effect of reset selection or the critic alone.

After frozen evaluation, run the original round-1 actor/critic training entrypoint,
then its original frozen-actor DIVL stage. Each stage has 150,000 updates and
batch size 1,024. Preserve the recipe's losses, masks, optimizer settings and
final evaluation. A timed-out or shortened run remains incomplete. Do not
extend training beyond the recipe because of a disappointing score.

The targeted-versus-coverage comparison uses the two R1 `no-cf` recipes listed
in [the scaling plan](DEPLOYMENT_RL_SCALING_PLAN.md). Real-Time EXPO-FT remains
a separate experiment with its own timing and initialization requirements.

## Measured frozen evaluation

Both original evaluator jobs finished cleanly on GPU0:

| Released policy | Successes | Success rate | Job wall time |
| --- | ---: | ---: | ---: |
| R0 Sobol, seed1, DIVL N32 | 34 / 50 | 68% | 98.72 s |
| R1 Mulligan, seed1, DIVL N32 | 46 / 50 | 92% | 77.94 s |

R1 succeeds on 13 starts where R0 fails. R0 succeeds on one start where R1
fails. Both succeed on 33 starts; both fail on three. The net difference is
24 percentage points on this fixed pilot. The author outcomes confirm equal
start identities and evaluation settings. This is evidence for these released
checkpoints on this subset. It is not a gain from local training.

The canonical runs are `author-subset-20261013T221340-0abcf1` and
`author-subset-20261013T221536-297fe3`, under `RL/abc/outputs/bench/`.
`FROZEN_N32_RESULTS.json` in the experiment directory records the original
point-result paths and hashes. The existing benchmark dashboard now includes
the measured counts and links to all 50 author outcome rows for each policy.

## Complete native videos

The [existing viewer](http://127.0.0.1:8770/original-methods/#mulligan-heading)
now shows both released policies side by side. Select any of the first three
starts. Each MP4 contains the initial frame and every counted control frame.

| Start | Round 0 | Round 1 |
| --- | --- | --- |
| 0 | Failure; 400 controls, 401 frames | Failure; 400 controls, 401 frames |
| 1 | Failure; 400 controls, 401 frames | Success; 148 controls, 149 frames |
| 2 | Success; 162 controls, 163 frames | Success; 160 controls, 161 frames |

All 50 round-0 and ten round-1 video-evaluation rows exactly match the
corresponding earlier asynchronous results. This includes outcomes, rewards,
episode lengths and start metadata. The 50-start scores come from the original
asynchronous evaluations, not the smaller round-1 video pass.

Video runs are `author-subset-20261013T221732-14ae26` and
`author-subset-20261013T222510-5ec8a7`. Both completed cleanly.
`NATIVE_VIDEO_ACCEPTANCE.json` records all six videos and trace hashes.
Each author process and observed encoder process exited.

The original imageio encoder starts a separate process group by default.
Our launcher now sets its documented `IMAGEIO_FFMPEG_NO_PREVENT_SIGINT=1`.
Actual CPU and native checks confirmed that the encoder shares the job group.
The round-0 pass preceded this correction; its observed encoders were checked
individually after completion. No author policy, learner or physics code changed.

## Original training is running

Run `author-subset-20261013T222807-f6985d` started on GPU0 at 22:28 UTC.
It uses the author's `train_cell.sh square-narrow-r01-mulligan --seed 1`.
The only extra flag supplies `wandb.run_name`, which enables rolling resume saves.
All learning settings remain unchanged: 150,000 updates, batch size 1,024,
the original losses, samplers, optimizer, horizons and final evaluation.

The original trainer admitted 57,108 unique critic/value frames and 21,015
actor frames. Its critic sampler uses half human-success and half DAgger data.
Its actor sampler uses human-only data. The input includes 23 state dimensions
and seven action dimensions. It predicts eight actions and executes six.

At 22:34:39 UTC, the original log reported 6,000 updates with finite losses.
Its first rolling checkpoint was saved at 5,000 updates. Warm intervals took
about 46–49 seconds per 1,000 updates. This projects about two hours of training;
it is not a measured completion time. The final author evaluation adds 400
episodes for each of the one-candidate and 32-candidate modes.

No locally trained task score is accepted yet. The job has a six-hour watchdog.
Do not change its installed runtime. Do not extend the fixed recipe based on
final-test outcomes. A saved resume file does not establish tested recovery.
Each canonical job gets a new output root, so matching a run name alone cannot
resume the old job.

The original `config.json` is saved before derived dimensions and data provenance
are attached. Treat it as launch configuration, not a complete data receipt.
Use the input pins and original load/sampler logs as separate evidence.
After IDQL acceptance, pass its actual `final_model` path to the original DIVL
stage. `--parent local` cannot discover a parent in another canonical output root.
The DIVL stage must preserve the trained actor and its normalization.

## Verification and remaining work

The original start-design command reproduced all 279 locked manifest rows.
The relevant author test suite passed 251 tests. The generated full grid
matches the published file and manifest hashes. Native CPU renderer checks
preserved all observed physics, policy inputs and outcomes across 12 zero-action
steps. These checks establish readiness; they do not establish policy success.

Evidence is under
`/home/user/aditya/RL/builds/mulligan-author-experiments-20261013/`.
`ASSET_HASH_CHECKS.json`, `DATA_INSPECTION.json`, `VIDEO_GRID_CHECK.json` and
`runtime/` contain the actual checks. `preparation/COMMANDS.json` contains
the original commands. Training, the subsequent DIVL stage and their measured
evaluations remain incomplete. No R1 Lite performance claim follows from Panda.

The installed recorder checks passed 17 tests. The launcher checks passed 32.
The full installed benchmark suite reported 775 passed, 24 failed and two skipped.
The 24 failures were then checked against the original installed `516a2c1` release.
Both releases passed 23 targeted cases in separate processes with short paths.
Both failed the same historical source-pin test, whose expected pin already
differs from `516a2c1`. The original full-suite failures also include collection
order and Unix-socket path limits. The suite remains failed; no new failure was
identified among these cases. Detailed logs and exact releases are under
`/home/user/aditya/RL/builds/abc-mulligan-full-regression-20261013/`.

The viewer uses the existing publication path. It preserves previous experiment
records and videos. Its installed export checks passed six tests. Native video
playback, seeking and mobile layout passed in a CPU browser. All six clips
loaded and sought to their ends. Both policies played together. Snapshot refresh
preserved playback. The existing RTC clip remained available. See
`VIEWER_BROWSER_RESULT.json` and the compact
[evidence pins](MULLIGAN_AUTHOR_EXPERIMENTS_20261013.json).

An independent results audit reproduced both scores, all paired counts,
source/checkpoint/grid hashes, video hashes and decoded frame counts.
It also checked the unchanged original evaluator and rollout functions.
Its receipt is `results-review/AUDIT.json` in the experiment directory.

Root owns native execution and result acceptance. The launcher agent owns
process supervision. The recorder agent owns video observation. The method
reviewer checks recipe fidelity independently. The original source owns all
learning and control behavior. The [experiment architecture](MULLIGAN_AUTHOR_EXPERIMENTS_20261013.mmd)
shows completed and pending work. The [scaling architecture](DEPLOYMENT_RL_SCALING_PLAN.mmd)
shows the wider plan; its future R1 Lite connections remain proposed.
This text follows STE guidance. A full ASD-STE100 compliance check was not run.
