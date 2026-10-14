# Real-Time EXPO-FT: GPU training and frozen Catcher trials

The recovered trained policy succeeds in **3/4** development trials.
The starting RTC-guided BC24 policy succeeds in **1/4** on the same environment seeds.
The learned policy fixes three failures and loses the baseline's one success.
This small result does not establish reliable improvement, generalization, or a scaling law.

[Open the comparison and all eight complete videos](http://127.0.0.1:8770/original-methods/#catcher-gpu-frozen-heading).
These videos show the two-dimensional Catcher task. They do not show R1 Lite.

| Environment seed | Starting RTC-guided BC24 | Trained whole policy |
| --- | --- | --- |
| 100001 | [Success](http://127.0.0.1:8770/original-methods/cb0345b6d7377f82.mp4) — 75 controls | [Failure](http://127.0.0.1:8770/original-methods/bb06df72aa9866cd.mp4) — 58 controls |
| 100002 | [Failure](http://127.0.0.1:8770/original-methods/6e65b9ce40e21a93.mp4) — 59 controls | [Success](http://127.0.0.1:8770/original-methods/f4b155ea2f61360b.mp4) — 66 controls |
| 100003 | [Failure](http://127.0.0.1:8770/original-methods/b75dc66e41d5565e.mp4) — 62 controls | [Success](http://127.0.0.1:8770/original-methods/8f44979dc7abdcaf.mp4) — 65 controls |
| 100004 | [Failure](http://127.0.0.1:8770/original-methods/b0071b60917c1b17.mp4) — 65 controls | [Success](http://127.0.0.1:8770/original-methods/37fee9ff4292bf43.mp4) — 66 controls |

Each video includes the reset frame and every control frame through termination.
Playback uses 30 simulation frames per second. It does not measure wall-clock real-time control.
The four environment seeds use one fixed level and the same initial physical state.
They produce different stochastic trajectories, not four distinct scene layouts.
These development trials were reused. They are not a held-out test set.

## What actually ran

The original GPU learner ran for 1388.52 seconds, including startup and shutdown.
It dispatched 4097 calls: 4038 native controls and 59 terminal no-ops.
There were 59 complete training episodes and one partial episode.
Native terminal success occurred in 32 episodes. Positive terminal reward occurred in 35.
Keep those two measurements separate. Training outcomes come from a changing policy.

The saved checkpoint has these actual optimizer counters:

| Saved group | Update count |
| --- | ---: |
| Base actor, editor, temperature | 3343 each |
| Critic, filter critic, encoder | 66860 each |
| Target critic optimizer | 0; target parameters use Polyak updates |

Training index 4096 is a driver index. It is not the actor update count.
The actor passed its 1,000-call learning-rate warmup. The earlier CPU-R2 run had 34 actor calls.
All seven parameter groups changed from initialization.
Replay retained 1,011,008 prior rows and 4,037 labelled online rows, without eviction.
The original driver leaves the final native transition unlabelled.

## Original code and declared changes

The run uses the author's `RealTimeEXPOFTLearner`, original training driver,
`AsyncChunkSampler`, replay sampling, losses, update functions and checkpoint API.
The original episode schedule uses batch 64, UTD 20 and offline ratio 0.5.
Ten collection episodes precede updates. No local stopping rule replaced that schedule.

The Catcher integration remains an explicit adaptation.
It supplies a 679-value state, six actions, H8 chunks, C4 execution, delay 4 and noise 0.1.
The backend loads the released flow BC24 actor. This is not a locally trained RTC-SFT artifact.
The task/backend hooks and declared asynchronous-launch recheck remain in use.
The exact simulation-paper configuration was not released with the inspected EXPO repository.
This run therefore does not reproduce the complete paper experiment.
See the [fidelity review](REALTIME_LEARNER_REUSE_20261013.md) and [recorded-input comparison](CATCHER_INFERENCE_PARITY_20261014.md).
The runtime still has four declared dependency metadata conflicts. No numerical dependency changed for frozen evaluation.
Only the task's checkpoint-admission code changed after recovery acceptance.

## Shutdown and checkpoint recovery

The author exited 0 after finalizing its checkpoint. The environment client exited −2 on SIGINT during shutdown.
The supervisor sent that signal immediately after author completion.
A late shutdown race is plausible. The exact signal delivery point is unknown.
The original receipts remain unchanged. Clean client completion was not accepted.

Root verified native closure, released resources, unchanged source and retained replay.
A separate CPU reader restored the complete checkpoint twice through the original public API.
Both restores matched. Parameters, optimizer clocks and serialized RNG passed their checks.
All 28 checkpoint files remained unchanged. This does not establish exact environment/replay continuation.
The [supervisor repair](REALTIME_CLIENT_SHUTDOWN_REPAIR.md) adds five seconds for natural client exit.
It also rejects nonzero client completion. It changes no learner code.

All four subsequent frozen evaluations exited 0/0 without cleanup escalation.
Their actor and optimizer state remained fixed. Each restored the same 3,343-update checkpoint.
Each matched its baseline's second reset state and observation.
The learned policy retains its saved RNG; this is not a common-policy-noise ablation.
The complete frozen process times were approximately 121–125 seconds per trial, including startup.

## What this means for scaling and R1 Lite

This run establishes a useful method-integration result and a promising small development comparison.
It does not isolate which training change caused the difference from CPU-R2.
Prior size, update exposure and training device all changed between those runs.

The next scaling study must compare useful new episodes at matched learner budgets.
Use multiple training seeds and a separate evaluation cohort before making a performance claim.
The [deployment-data plan](DEPLOYMENT_RL_SCALING_PLAN.md) separates SGS reset selection,
Mulligan supervised collection, and Real-Time EXPO execution and learning.
The [R1 Lite assessment](R1LITE_AUTHOR_RECIPE_TRANSFER_20261014.md) identifies the reward and reset contracts still needed.
No R1 Lite training or hardware run is claimed here.

## Verification and team

The combined normal wheel matches all 1,149 source and installed payloads.
The installed release passes 61 focused Python checks and 26 DOM checks.
These checks cover process cleanup, device roles, export and retained viewer behavior.
The historical broader repository suite has unresolved failures; it is not certified by these focused checks.
The actual browser played all eight videos and tested seeking. Refresh retained the video players and their positions.
Desktop and mobile checks passed. Earlier Catcher, Mulligan and FastTD3 panels remained unchanged.
The companion evidence file records these checks and their artifact hashes.

Root owns execution, artifact acceptance and publication.
The author implementation owns learning and action selection.
Independent agents reviewed process repair, recovery, frozen admission and presentation.
External Jev and OpenQodex checks remain waived by the user's results-first instruction.

[Architecture source](REALTIME_CATCHER_GPU_20261014.mmd). [Evidence index](REALTIME_CATCHER_GPU_20261014.json).
This explanation uses ASD-STE100 guidance. It has not had a full compliance check.
