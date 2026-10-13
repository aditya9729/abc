# RTC command and method audit

The full video does not test online Real-Time EXPO-FT. It tests an RTC base
policy after one supervised update on three demonstrations. Task-policy
readiness has not been established for that checkpoint. The coordinator checked
source compatibility but did not establish policy readiness before showing it.

This audit uses source review and CPU analysis. No new training or simulator
run was launched. Earlier results and the failed process-cleanup status remain
unchanged.

## Required sequence and actual work

The [Real-Time EXPO-FT paper, IV-C and VII-E1](https://arxiv.org/html/2609.18207v1)
requires task-specific, prefix-conditioned imitation training before online RL.
It describes initialization to roughly 30% task success or higher. Its online
method combines delayed base inference, edits from the current observation,
action-value selection, and a noise-value filter for training backups.

The [ordinary EXPO-FT paper, 4.2](https://arxiv.org/html/2605.25477v1)
also checks the starting policy. If needed, task imitation training precedes
online learning, with roughly 40% success as its stated target.

| Stage | Released real-time example | Executed ABC diagnostic |
| --- | --- | --- |
| Task initialization | RTC-SFT, 25 demonstrations, maximum 4,000 updates | RTC-SFT, three successful TRAIN demonstrations, one update |
| Selected initialization | Offline directory `checkpoints/2000/params` | Genuine checkpoint 1; optimizer counter 1 |
| Online learner | `RealTimeEXPOFTLearner` | Not run |
| Online components | Editor, action critic, noise critic, base-policy updates | None trained or used in this clip |
| Evaluation | Restore the online policy | Restore `RTCLearner` checkpoint 1 |
| Task and actions | DROID dynamic picking; seven Cartesian/gripper channels | ABC-YAM; 14 absolute joint/gripper channels |

Source: pinned EXPO commit `21fe3d3b7d913c80836496817965932f49e0aedf`,
[offline launcher](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/scripts/dynamic_pick/offline_train.sh),
[online launcher](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/scripts/dynamic_pick/run_server.sh),
and real-time OpenPI `2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8`.
The offline loop uses zero-based periodic saves. Directory 2000 is not proof
of exactly 2,000 optimizer updates. The example's update count alone is also
not a readiness test.

The released script uses delay 5 and 30 transitions per update call. The
paper's Dynamic Picking settings use delay 3 and 25 transitions per call.
These differences remain explicit. Running the released example is not an
exact reproduction of every paper setting.

## Recorded command problem

The analysis compares 3,540 RTC commands with 3,538 commands from three
successful TRAIN demonstrations. It excludes changes across demonstration
resets. Each change is the largest absolute change among the 12 joint targets.
Grippers are excluded from these joint statistics. These are command targets,
not measured joint motion. The scenes and policies are not a matched benchmark.

| Measurement | RTC checkpoint 1 | Successful demonstrations |
| --- | ---: | ---: |
| Median largest joint-target change per command | 4.17 degrees | 1.05 degrees |
| 95th percentile | 32.58 degrees | 2.52 degrees |
| Maximum | 141.30 degrees | 6.35 degrees |
| Commands with any joint target outside its actuator range | 33.14% | 0% |

At eight-command chunk boundaries, the median largest target jump is
26.32 degrees. Within chunks, it is 3.68 degrees. Thus, the abrupt requests
are concentrated where new predictions replace the previous chunk.
MuJoCo limits effective actuator controls internally. The log records raw
requested controls. Analytically applying those limits still leaves a
23.53-degree median boundary jump; that calculation is not a native replay.

## Checks and limits

Independent checks found no joint-order, sign, radians/degrees, or
absolute/delta conversion error in the inspected path. The native adapter
passes joint targets unchanged and scales each gripper fraction by 0.0475 m.
Across all three demonstrations, action-to-recorded-control mapping is exact.
Recorded joint positions also reproduce the saved float32 observation states.

Installed input/output transforms passed CPU round-trip and camera-routing
checks. Normalization is applied once in each direction. This checks the
coordinate conversion; it does not prove model quality or actual model tensors.

For delay 5 and chunk length 8, the original evaluator passes the prior
normalized tail `[3:8]` as the prefix. It executes the next generated positions
`[5:13]`. The inspected source indices agree. The original sampler code clamps
the five supplied prefix positions; the trace does not prove which tensor
it received. This code imposes no hard bound on the next action. The model
must learn that continuation.

The trace does not contain the actual normalized prefix, raw model output,
or current joint positions. Therefore, runtime prefix values and physical
tracking remain unverified. The prefix also represents requested actions,
which can differ from the controls limited by the simulator.
The original client replaces nonfinite actions with zeros. Finite recorded
commands therefore do not prove that all original model outputs were finite.

Insufficient task and prefix learning is a plausible cause of the motion.
It is not a demonstrated sole cause. There is no evidence from this clip
that the complete Real-Time EXPO-FT method failed: its editor and critics
were absent. `only_base_actions=False` enables RTC prefixes for `RTCLearner`;
it does not create those online components.

## Next experiment requirements

Retain the checkpoint and video as diagnostic evidence. Do not admit checkpoint
1 as a usable online-RL initialization. Resolve client shutdown before another
campaign. Keep the original author loops and losses.

Verify actual model inputs and prefix values in a bounded diagnostic. Establish
task-specific RTC-SFT readiness with genuine task data and validation episodes.
Compare delay 0 and delay 5 under matched conditions if discontinuities remain.
Select initialization from observed task behavior, not training loss alone.
Only then assess the original online learner. No extra clipping or smoothing
has been introduced as an alleged method correction.

The previous rollout also missed its 34 ms control target: median command
spacing was 89.76 ms. Step-driven simulation does not establish continuous
real-time operation. Policy readiness, process shutdown, and timing remain
separate unresolved requirements.

## Evidence and ownership

All new evidence is under
`/home/user/aditya/RL/builds/rtc-action-diagnosis-20261013/`:

- `COMMAND_DIAGNOSTIC.json`: measured command statistics and event checksum.
- `native-review/analyze.py` and `ANALYSIS.json`: independent source, control,
  state, actuator-limit, and numerical checks.
- `data-review/check_boundary.py` and `CPU_BOUNDARY_RESULT.json`: installed
  transforms, camera routing, source identity, and recorded-trace limits.
- `paper-review/REVIEW.md`: paper requirements and exact repository anchors.

The measured event SHA256 is
`8fdf46795138484753be6cdc4a3ab974f3a5e5e01d855f58f78e7b30a17aa872`.
The source run is `author-subset-20261013T193655-d2c6f5`.
[Compact measurements and evidence checksums](RTC_ACTION_DIAGNOSIS_20261013.json).
[Full visual result](RTC_VISUAL_ROLLOUT_20261013.md).
[Architecture and audit flow](RTC_ACTION_DIAGNOSIS_20261013.mmd).

Root owns experiment selection and result claims. The method reviewer checked
the original paper and source. The data reviewer checked model boundaries.
The environment reviewer checked native action and state contracts.
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
