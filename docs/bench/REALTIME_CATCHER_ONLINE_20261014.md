# Catcher: native Real-Time EXPO integration

The short native training run completed. The saved checkpoint contains actual
base-policy, editor, critic, and filter updates. This verifies an integration;
it does not establish an improvement in task success.

[View complete training episodes](http://127.0.0.1:8770/original-methods/#catcher-training).
The viewer also retains the separate original frozen BC/RTC trials.

## Measured result

| Measurement | Observed value |
| --- | --- |
| Native simulator controls | 3,029 |
| After-terminal no-op dispatches | 44 |
| Complete / partial native episodes | 44 / 1 |
| Successful complete training episodes | 8 |
| Original learner update calls | 34 |
| Base policy / editor / temperature updates | 34 each |
| Critic / noise filter / state encoder updates | 680 each |
| CPU process duration | 589.88 seconds |
| Server / client exit codes | 0 / 0 |
| Remaining owned process groups | None |

These outcomes come from changing policies and starting states. They are not
a frozen-policy score. There is no matched before/after improvement result.
The viewer selects episodes 1, 12, and 44 by time, independently of outcome.
All three selected episodes fail. Episode 12 follows the first learner update.
Each clip includes the reset frame and every native control frame at 30 fps.
Simulation-clock playback does not establish wall-clock real-time operation.

The original loop executes indices 0 through 3072. Thus, it dispatches 3,073
commands. The task adapter treats its after-terminal commands as no-ops.
All 45 recordings decoded at native 128 × 128 resolution.
The client log contains an ignored `WeakKeyDictionary` interruption during
interpreter shutdown. The client exited zero; no escalation or surviving owned
process was observed.

## What ran, and what changed

The run uses released EXPO revision `21fe3d3`, RTC policy revision `9296f31`,
and Kinetix revision `cf7453e`. The EXPO update equations and complete original
training-loop suffix remain intact. Explicit state-input branches connect the
released learner to the Catcher flow policy. This is a declared adaptation of
the release; it is not an unchanged, released Catcher EXPO experiment.

The bounded settings were seed 42, delay 4, execution horizon 4, action horizon
8, batch 64, UTD 20, and one update call per completed eligible episode.
The original ten-episode warmup remains. The loader admits the first 128
complete demonstration episodes, which contain 8,638 records.
BC24 supplies the initial actor. Candidate, edit, and filter counts are 32.
The run uses `bc_guidance` and edit scale 0.2.

The adapters provide symbolic observations, state encoding, the flow-policy
interface, data loading, task lifecycle, and transport. They preserve the
released critic, editor, temperature, filter-target, and target-update equations.
The state encoder's two 256-unit hidden layers are an integration choice.
The release does not provide an exact Catcher configuration for that choice.

Two earlier attempts remain preserved. The first failed before simulation
because the original checkpoint helper required the author's `--resume` flag
on the newly created directory. A fresh empty directory still initializes
from BC24; it does not restore prior training. The second attempt collected
747 controls but failed before its first update because logging required images.
The final repair changes one logging function. State batches report 16 common
statistics; image batches retain their original 19. Malformed nonempty image
batches still fail. Four installed logging checks pass.

The combined JAX runtime retains two declared dependency metadata conflicts.
It explicitly uses the original PRNG convention. Persistent compilation caching
is disabled because the author helper hard-codes a path outside the workspace.
No GPU was used; original Mulligan training retained GPU0.

## Acceptance evidence

Replay retains 11,666 records: all 8,638 demonstrations plus 3,028 labelled
native transitions. One final transition remains unlabelled in the original
episode sink. No replay record was evicted. Native traces preserve issued
commands, observation order, termination, masks, and separate success labels.

The original checkpoint helper restores the complete saved learner twice.
Both restores match. Optimizer counters agree with the update schedule.
All dynamic values are finite. All seven parameter groups differ from fresh
initialization. The inference cache is explicitly refreshed after restoration.
All 32 checkpoint files remain unchanged by verification.

This does not test resuming the native world, replay, or asynchronous queues.
Checkpoint directory `3072` denotes a control-loop index, not an optimizer count.

Six installed viewer-export checks pass. Browser checks cover all three frozen
clips and three training clips, seeking, playback, refresh, and mobile layout.
The earlier full benchmark suite remains at 775 passed, 24 failed, and two
skipped. It was not rerun for this focused viewer and evidence publication.

Artifacts are under
`/home/user/aditya/RL/builds/realtime-expo-catcher-integration-20261013/`:

- `ROOT_NATIVE_ONLINE_ACCEPTANCE.json`
- `native-driver/actual-cpu-3072-r2/PROCESS_RESULT.json`
- `native-driver/R2_BOUNDARY_OBSERVATION.json`
- `native-driver-review/ACTUAL_R2_CHECKPOINT.json`
- `native-publication/PUBLICATION.json`
- `native-publication/VIEWER_BROWSER_RESULT.json`

## Scaling decision

The end-to-end loop now executes. Its short schedule cannot answer whether
more deployment experience improves a policy. The paper reports 100,000 online
steps and multiple seeds. This check changes the schedule and data subset.
Prefix-training and filter-target differences between paper and release remain
documented in the [learner review](REALTIME_LEARNER_REUSE_20261013.md).
These must remain visible in any larger experiment.

The [scaling plan](DEPLOYMENT_RL_SCALING_PLAN.md) separates collection allocation,
learner updates, and independent final evaluation. Original Mulligan training
continues on GPU0. Its locally trained result remains pending at this publication.
SGS remains a preparation task while its author implementation is unavailable.
R1 Lite adaptation follows a measured author-task result.

Root owns execution, acceptance, and publication. The task agent owns the
environment and trace checks. The actor agent owns backend and checkpoint checks.
An independent reviewer checks data and result claims. External reviews remain
waived under the user's results-first instruction.

[Evidence pins](REALTIME_CATCHER_ONLINE_20261014.json) and
[architecture](REALTIME_CATCHER_ONLINE_20261014.mmd).
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
The wider original-recipe goal remains active.
