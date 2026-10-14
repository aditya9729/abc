# Catcher readiness with the original RTC evaluator

The frozen author BC24 policy succeeds on **96 of 100 streams without inference
delay**, **15 of 100 with naive delayed inference**, and **54 of 100 with RTC
guidance at the same four-control delay**. These are
completed CPU evaluations. They contain no EXPO updates and show no learning
gain. They establish a larger starting-policy check before the next Real-Time
EXPO experiment.

| Frozen profile | Delay, controls | Successes / streams | Mean return | Mean episode length, controls |
| --- | --- | --- | --- | --- |
| BC | 0 | 96 / 100 | 0.990 | 67.630 |
| Naive delayed BC | 4 | 15 / 100 | -0.260 | 66.590 |
| RTC guidance | 4 | 54 / 100 | 0.340 | 67.190 |

All three rows use the same released BC24 checkpoint, fixed Catcher level, master
seed 20261014, 100 native random streams, action horizon 8, execution horizon 4,
ten flow steps, and original action noise 0.1. These are 100 streams on one
level, not 100 task configurations or training seeds. The original evaluator
returns aggregate first-episode statistics. It does not expose paired
per-stream outcomes here. Success counts are inferred from its binary solved
mean, with floating-point tolerance; they are not counts of positive returns.

The two delayed profiles isolate the inference method at the same configured
delay. RTC guidance has a 39-percentage-point higher success fraction in this
cohort. The means do not expose which individual streams change outcome.
This is a frozen inference comparison, not an RL gain or evidence across tasks.
Four controls equal 133 ms of simulation time in this task. The CPU execution
does not demonstrate wall-clock real-time control.

## What changed from the author implementation

The installed selector calls the original public `eval_flow.eval()` function.
The original function owns resets, policy inference, physics, termination,
scoring, and rendering. Its full command would sweep 96 settings and discard
the returned video. Our selector chooses one setting and saves its metrics and
video. It also exposes the native `num_evals` setting. The naive control uses
selector version 0.1.1; its only behavior change adds the existing naive profile
to the allowed CLI choices. It changes no author
rollout or learner calculation.

The original RTC revision is
`9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`; Kinetix is
`cf7453ea103fa0b77348af1a39f689c658161613`. The author policy, model, evaluator,
training constants, level, and BC24 checkpoint authenticate against the
recorded hashes. The CPU runtime uses JAX 0.4.35, jaxlib 0.4.34, Flax 0.10.2,
and the original legacy PRNG setting. Its dependency check passes.

All three processes exited zero without cleanup escalation. Root confirmed empty
owned process groups, finite original metrics, unchanged source and checkpoint
bytes, and full video decoding. End-to-end times were 315.16 seconds for BC
and 299.61 seconds for RTC. The naive delayed control took 332.48 seconds.
These times include startup, compilation and serialization.
These measurements are not inference-latency benchmarks.

Each recording shows only batch member zero, with all 256 original pre-action
frames at 128 by 128 pixels. It can include automatic resets after the scored
first episode. Playback follows the author's 15 fps serialization: 17.07
seconds for 8.53 simulation seconds, or half speed. It is not a recording of
all 100 streams.

The complete batch-zero recordings are available for
[BC](/home/user/aditya/RL/builds/realtime-expo-catcher-readiness-20261014/actual/bc_delay0_c4/native/bc_delay0_c4.mp4),
[naive delayed BC](/home/user/aditya/RL/builds/realtime-expo-catcher-readiness-naive-control-20261014/actual/naive_delay4_c4/native/naive_delay4_c4.mp4), and
[RTC guidance](/home/user/aditya/RL/builds/realtime-expo-catcher-readiness-20261014/actual/rtc_guidance_delay4_c4/native/rtc_guidance_delay4_c4.mp4).

## What this means for the next RL run

The earlier [four-start EXPO-client comparison](REALTIME_CATCHER_FROZEN_20261014.md)
scored BC24 1/4 and the short R2 policy 0/4. Keep those results. This larger
check uses a different evaluation driver, batch, random-key path and runtime.
It does not invalidate that cohort or prove a client defect. A matched control
through the actual EXPO client is needed before transferring this 54% baseline
to a learning claim in that path.

The next learning profile must use the original Real-Time EXPO learner and
training driver. The prepared profile restores the driver's transition-derived
episode update schedule, batch 64 and UTD 20. It uses the full 1,011,008-record
complete-episode prior and declares an offline replay fraction of 0.5. This is
a new, explicit pilot configuration; the exact simulation release recipe is
still incomplete. The old R2 pilot made only 34 actor updates and remained
inside the 1,000-update learning-rate warmup.

GPU profiling remains pending while original FastTD3 owns GPU0. The task client
must stay on CPU while the learner uses the leased GPU. Learning outcomes,
collection counts, update counts, and frozen evaluation must remain separate.
No result here establishes a scaling law or R1 Lite performance.

## Evidence and ownership

The raw artifacts are under
`/home/user/aditya/RL/builds/realtime-expo-catcher-readiness-20261014/` and
`/home/user/aditya/RL/builds/realtime-expo-catcher-readiness-naive-control-20261014/`.
The [evidence index](CATCHER_READINESS_20261014.json) records the exact files.
The [diagram](CATCHER_READINESS_20261014.mmd) shows executed and planned paths.
The [scaling plan](DEPLOYMENT_RL_SCALING_PLAN.md) places this check alongside
SGS and Mulligan.

Root owns execution, acceptance and reporting. The preparation agent supplies
the small selector and process boundary. An independent agent checks source
preservation and result interpretation. The authors' code owns policy inference
and simulation. This explanation follows STE guidance; full ASD-STE100
compliance was not checked. The broader original-recipe goal remains active.
