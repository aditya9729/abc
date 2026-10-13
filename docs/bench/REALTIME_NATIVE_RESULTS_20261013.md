# Original RTC integration: measured results

The original RTC-SFT program now trains on ABC data. The original RTC evaluator
restores its checkpoint and runs delayed action chunks in native ABC simulation.
The control path works. The measured control rate does not meet the 34 ms interval.
These are integration checks, not a task-success benchmark or online RL result.

| Check | Measured result |
| --- | --- |
| Original RTC-SFT | One update with batch size 64; prefix delay sampled up to 10; seed 42 |
| Training metrics | Loss 0.0404392; gradient norm 0.511554; both finite |
| Saved state | Actor/Adam/schedule counters 1/1/1 and expected RNG |
| Native control | 32 commands, 544 physics steps, 1.088 simulated seconds |
| Timing | Median dispatch 91.731 ms; p95 297.346 ms; target 34 ms |
| Observation cost | Median 82.5 ms; native physics call median 5.365 ms |
| Cold stalls | First model plan 33.974 s; first prefix replan 25.061 s |
| Cleanup | Both programs stopped; native environment and video closed; GPU 0 lease released |

The video contains 33 frames and plays at simulation speed. It does not show
wall-clock execution speed. The first zero-control reset has a separate clip.
The policy received only one update. Its short truncated episode is not a useful
success-rate measurement. Online `RealTimeEXPOFTLearner` updates remain pending.
Direct GPU overlap was not measured by the original logs.

A later [full visual episode](RTC_VISUAL_ROLLOUT_20261013.md) now replaces the short
clip in the viewer. It covers 120.36 simulated seconds and ends at 0/5 bottles.
The recording is complete, but its job failed the client process cleanup check.

The 32-control check uses two TRAIN resets, delay 5, replan 8 and
`only_base_actions=False`. The last flag is required: True bypasses RTC prefix
conditioning. Native horizon 32 and author loop budget 33 preserve the terminal
observation. These limits are explicit diagnostic settings.

The task adapter changes DROID inputs to ABC-YAM absolute 14-dimensional actions,
14-dimensional state and three camera views. It uses 3538 recorded TRAIN controls from three episodes and
TRAIN-only normalization statistics. No VAL rows train the model. This is not
an R1Lite result or DROID paper reproduction. The original learner, replay,
prefix loss, optimizer and control loops are preserved.
[Exact four-file dispatch patch](REALTIME_AUTHOR_TASK_DISPATCH.patch) and
[compact source pins and results](REALTIME_NATIVE_RESULTS_20261013.json) make the
changes reviewable. Source and dependency proofs are separate from native results.

Full local evidence is in
`/home/user/aditya/RL/builds/realtime-expo-abc-integration-20261013/`:
`ACTUAL_RTC_ONE_STEP_RESULT.json`, `ACTUAL_RTC_TIMING_RESULT.json`,
`RTC_ONE_STEP_METRICS.json`, `RTC_TIMING_MEASUREMENTS.json` and the checkpoint
manifest. The independent review is under
`/home/user/aditya/RL/builds/realtime-expoft-independent-method-review-20261013/`.
Both original programs ran through the canonical GPU 0 coordinator. No other GPU
or Leela job was used. The earlier ordinary SFT/EXPO artifacts are retained.

Next: measure warm training throughput and prepare sufficient RTC-SFT before
online Real-Time EXPO-FT. Profile observation collection before a real-time claim.
Do not reduce the author batch, skip fresh cameras, or replace its update loop
without declaring a new comparison.

Root owns execution and publication. The data agent owns the input contract.
The environment agent owns native ABC operations. The reviewer checks source,
state and measured events. [Architecture](REALTIME_AUTHOR_PROGRAM.mmd).
STE guidance used; full ASD-STE100 compliance was not checked.

Verification: 34 focused installed checks passed; one author-client test was
skipped in the coordinator-only prefix. All 24 client boundary checks passed in
the separate installed client-test prefix. The initial combined benchmark suite
reported 762 passes, 21 failures and one skip. One new RTC test relied on global
import state; it was corrected to check a fresh process and then passed. The
other failures concern historical Fast test import state and an older authenticated
`subset.py` pin. Isolated runs passed 44 compiler tests and 21 reach tests; one
historical reach test still rejects the newer coordinator pin. These results do
not establish a fully passing combined legacy suite. No old run pin was weakened.
