# Fresh original ABC collection results

Original ABC-VLA succeeded on all six scenes below. Every final native judge
reported all five active bottles currently in the bin. No policy update ran.
The original policy and simulator Python remain unchanged from
`d0832d12651d1b260a652861a14648dc5f3660c7`. This five-bottle collection is a
small data subset, not a paper reproduction or a post-training comparison.

| Split | Scene seed | Controls | Current bottles | Success |
| --- | --- | --- | --- | --- |
| TRAIN | 9030 | 1410 | 5/5 | Yes |
| TRAIN | 9031 | 1091 | 5/5 | Yes |
| TRAIN | 9032 | 1037 | 5/5 | Yes |
| Validation | 9100 | 1239 | 5/5 | Yes |
| Validation | 9101 | 1072 | 5/5 | Yes |
| Validation | 9102 | 1058 | 5/5 | Yes |

[Watch all six actual videos](http://127.0.0.1:8770/original-abc-collection-20261012/).
All clips played in the browser. Their frame counts equal T+1. Decoded first,
middle and final frames agree with raw camera recordings within compression
tolerance. The viewer reports the source hashes. Earlier zero scores remain
separately labeled integration diagnostics.

The completed TRAIN run is `author-subset-20261012T182237-95ea21`.
The completed validation run is `author-subset-20261012T190246-c4e7d0`.
The canonical parents bind their completed workers. Validation used526.89 GPU0
seconds. The collector preserved flow10, H30, execute15, RTC prefix4/lead4,
original CPU MuJoCo physics and MJWarp cameras. The explicit five-bottle subset,
three-world counts and split seeds are adaptations to the original distribution.

## Dataset and checks

The immutable public cohort is
`/home/user/aditya/RL/builds/demo-collection-20261012/published-cohort-v1/cohort.json`.
Its schema is `sadhana.abc.demonstration-cohort/v1`. Episodes use
`sadhana.abc.policy-demonstration/v1`. Each has T raw float32 physical YAM14
commands and T+1 state/camera frames. Three synchronized RGB panels are168x224.
Native model/integration files, policy requests and task judges remain pinned.
These are simulator judgments. Human, Atlas and dynamics approval are false.

The normally installed EXPO reader authenticated the real cohort and preserved
first/last complete-window source actions and states. It exposes3493 TRAIN H16
windows and3324 validation windows. Original OpenPI RunningStats fitted TRAIN
windows only, including the last partial statistics batch. Statistics are at
`/home/user/aditya/RL/builds/demo-collection-20261012/task-stats-v1/task_stats_manifest.json`.
Independent review authenticated127 unique pins and checked chronology, RTC
issued-row linkage, split isolation and TRAIN statistics means/std. It did not
independently recompute quantiles or qualify deterministic physics replay.

## Remaining comparison work

The new task-SFT adapter passed36 focused installed CPU tests with explicit small
doubles. Original EXPO/OpenPI Python stayed unchanged. Its first B64 capacity
attempt stopped before model initialization: our file validator rejected the
original checkpoint's valid empty `commit_success.txt`. It made zero updates.
This was an adapter defect. Version0.2.6 fixed exact empty checkpoint-object
authentication. Its real retry initialized the original model at update0, then
stopped when compile-only lowering supplied JAX abstract arguments to the typed
TrainState constructor. The original update and B64 forward did not execute.
A scoped capacity-only repair is being qualified. No measured memory-fit or
task-training result follows from either failed attempt.

The old EXPO host cannot recreate these collected validation resets. It uses
default bottle-count sampling and clears carried arm state. A real CPU qualifier
restored all three validation MJB models and full integration vectors twice.
All six initial model/state/judge comparisons passed. All96 replayed controls
matched saved actuator inputs, qpos/qvel/time, policy states and task judgments
exactly. No camera or policy ran. Independent review authenticated the actual
results against source arrays. [Actual CPU receipt](/home/user/aditya/RL/builds/demo-collection-20261012/captured-val-cpu-v1/receipt.json).
A versioned production host and live camera checks remain required before
matched policy rollout comparisons.
The original episode-based EXPO training schedule also needs a separate driver.
FastTD3's original Stand environment passed a small native CPU boundary test.
The extra unused render world explicitly disabled rendering. The actual receipt
is `builds/humanoidbench-author-setup-20261012/native-cpu-003/receipt.json` under
the RL workspace. Its128-world GPU qualification and parallel ABC collection
remain pending. The canonical stage registration now assigns the proposed
HumanoidBench worker to h1hand-stand-v0/unitree_h1_with_hands. It cannot be
reported as ABC bottles/YAM. Fifteen small CPU boundary tests passed. The
new stage still needs normal release and native qualification.

Root owns serial GPU admission, data publication and result claims. Method peers
own adapters and runtime setup. Independent review checks code and artifacts.
[The architecture](DEMONSTRATION_COLLECTION.mmd) separates executed collection
and statistics from pending training and evaluation. This explanation uses STE
guidance; formal ASD-STE100 compliance was not checked. No R1Lite result exists.
