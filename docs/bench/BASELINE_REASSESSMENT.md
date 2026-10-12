# Baseline reassessment and original learners

The original ABC-VLA succeeded with the upstream evaluation settings: five
bottles placed at control 819. The earlier short comparison used another recipe.
Its 0/3 result does not establish failure of the published baseline.

The original ABC simulator and policy source remain unchanged at
`d0832d12651d1b260a652861a14648dc5f3660c7`.
Original source alone does not establish an original evaluation recipe.

| Setting | Previous small comparison | New upstream qualification |
| --- | --- | --- |
| Control horizon | 1,000 | 236 chunks × 15 = 3,540 |
| Flow denoising steps | 5 | 10 |
| RTC | Disabled | Enabled; prefix 4, lead 4 |
| Physics | Batched float32 MJWarp, one world | Original sequential CPU MuJoCo |
| Cameras | MJWarp, three 168×224 RGB views | Same backend and dimensions |
| Fast inference | Disabled | Original max-autotune compilation |
| Layout seeds | 9011, 9012, 9013 | 20260511 for first qualification |
| Policy random seed | 901 | Upstream default 0 |
| Cohort | Three development layouts | One layout; upstream default is five |
| Scene reset | Forced five bottles; variants and scales fixed | Original default reset; variants and scales randomized |

RTC conditions new predictions on actions that are already queued for execution.
The new test calls the original `abc_minimal.eval_policy.run_eval` function.
It preserves the original reset behavior. It records any seed resampling in the
original output. One layout qualifies the path; it cannot reproduce a paper score.

The physical scene is `put_plastic_bottles_in_bin`. The explicit prompt remains
`sim throw plastic bottles in bin`. Released shared-mixture metadata maps the
placement scene to this training prompt. The local VLA sidecar lacks this map.
The physical throwing scene is a different task.

The checkpoint is `vla_abc130k_200000_v2.pt`, a generalist VLA parent at step
200,000 of a configured 500,000-step schedule. Its SHA-256 is
`4838acf69279497b917c998039089b5b6a310dfe65fc0b5c438c3339ee25c753`.
The README's specific bottles-success example uses a task-finetuned DiT policy.
That example is not a measured claim for this VLA parent.
The prior host already verified external statistics against embedded statistics.
No normalization mismatch has been established.

## What each method changes

| Method | Learned component | Base policy | Current evidence and limitation |
| --- | --- | --- | --- |
| ResFiT | TD3 action residual | Frozen ABC-VLA | Original learner executed. Demo-derived action scaling also changes base commands. |
| QF3 | Flow policy head through LoRA | ABC-VLA initialization | Paper implementation executed. Exact author implementation and initializer remain unconfirmed. Further experiments are on hold. |
| FastTD3 | Full state policy from scratch | None | The 4,000-control pilot and frozen author-bf16 evaluation completed. The one-world data regime is a material gap; no method ranking is justified. |
| EXPO-FT | Q-guided action editor and distilled VLA | pi0.5 | Real SFT and a full frozen pi0.5 rollout completed. Native EXPO update and post-update evaluation remain pending. |

ResFiT preserves the base network. Its original action scaler maps commands
through ranges fitted from demonstrations. Our single successful demonstration
has three bottles. The evaluation scene has five. Its ranges clipped or changed
base commands even when the learned residual was zero. The previous ResFiT score
therefore does not isolate the learned correction.

QF3 changes the flow policy itself. It does not add an action residual.
The small experiment used paper equations plus explicit local critic choices.
The official project page still had a disabled code link on 2026-10-12.
Keep the source and measured result. Wait for official code before another QF3 run.

The user selected original FastTD3 from scratch. It receives privileged simulator
state, including object poses and velocities. This differs from the VLA's camera
input. It is a separate benchmark condition. Its original networks, replay,
normalization and update mathematics stay intact. The ABC environment boundary
maps normalized actions to physical joint commands and separates success from
time-limit truncation. The first native test uses one world.

The user selected original EXPO-FT with pi0.5. Its author learner is JAX code.
The previous PyTorch port is not used for this condition. The author workflow
requires task-specific supervised finetuning before online RL. ABC uses two arms
and 14 joint/gripper channels. The author's DROID examples use another action
convention. A versioned data and action adapter must establish that boundary.
The source-judged three-bottle TRAIN demonstration is a small diagnostic input.
It is not a human-approved five-bottle expert dataset.

## Execution and evidence

### Measured baseline

Run `author-subset-20261012T134246-730cdb` completed one of one worlds.
The original task judge reported five of five bottles and current success.
The policy completed control 819. The coordinator charged 377.15 seconds,
including cold compilation. [Watch the recorded cameras](http://127.0.0.1:8770/original-methods/).

This result does not identify which recipe change caused the improvement.
Success occurred before the earlier 1,000-control limit. A longer horizon alone
does not explain it. Layout, reset distribution, physics, sampling and RTC changed
together. The saved full randomization state binds later matched evaluations.

The original summary SHA-256 is
`fa8b83a8e801eca5bdfcfee39806c1ec09ecdc15b483411591b9df8cd5877bc6`.
The original worker receipt SHA-256 is
`9042f589a2873a46ccc1800e6180d19f706389992d416af5cdbf7f03c6c70df4`.
Actual browser playback, desktop/mobile layout and video range delivery passed.

### Pretrained pi0.5 qualification

Run `author-subset-20261012T141113-882427` completed original pretrained pi0.5
initialization and one original supervised update. The fixed TRAIN-batch loss
fell from 0.11113274 to 0.06018557. The gradient norm was finite at 12.59335.
Training stopped after meeting the configured 25% loss-reduction target.

This is a tiny-batch training diagnostic. It has no simulator task score or
held-out loss. The one three-bottle demonstration does not establish a trained
five-bottle policy. The separate frozen-policy measurement below does not change
the scope of this SFT diagnostic. Native EXPO-FT RL remains pending.
The original full train-state checkpoint is recorded in `expoft/sft_qualification.json`
within that run. The coordinator charged 237.55 seconds and released its GPU0 lease.

### Original pi0.5 frozen native evaluation

Run `author-subset-20261012T153955-9965f5` completed 3,540 controls on the exact
captured baseline scene. The original task judge reported zero of five bottles
and zero of one successful episodes. All seven parameter groups and optimizer
counters stayed unchanged. No learner update occurred. The host closed with
exit zero. The coordinator charged 504.74 seconds, including initialization,
compilation, parameter hashing and cleanup.

The original `EXPOLearner.sample_actions(..., only_base_actions=True)` draws eight
base candidates and executes candidate zero. It uses H16/C8 and the real one-step
SFT checkpoint. Synchronized warm sampler latency had median 0.13348 seconds
and p95 0.15170 seconds across 442 warm calls; the first cold call took
20.05311 seconds. Physics pauses during inference. These timings do not prove
realtime operation or candidate-selection performance.

The host clipped 4,566 gripper coordinates across 2,921 control steps. Channels
6 and 13 were affected; no joint coordinate was clipped. This declared physical
action boundary differs from the original unbounded simulator command call.
It is shared by the pi0.5 and EXPO conditions. The demonstration has three bottles
and one training layout, while this evaluation has five bottles and different
geometry. One SFT update does not establish a task-trained policy. This result
does not establish a failure of pi0.5 or EXPO-FT and cannot support a method ranking.

The worker receipt SHA256 is
`9029d85fe65abd3f203965a79b5e64997f85735eea5c7f3658d35ae7b53c7e43`.
The actual episode video has 3,541 decoded frames at 672 by 168 pixels.
An initial zero-control reset also produced a video; it is excluded from the
episode count. Root verification is recorded in
`builds/expoft-author-setup-20261012/PI05_NATIVE_FROZEN_RESULT_V1.json`.
The finite parent job is `configs/bench/original_pi05_frozen_job.json`.

The first native EXPO update attempt, `author-subset-20261012T155036-42f91c`,
collected 32 TRAIN controls and failed before completing an update. Our adapter
added an outer OpenPI mesh context around `agent.update`. The original learner's
actor update opens its own context and rejects nesting. The simulator closed
cleanly. The failed attempt has no trained checkpoint or task score. Its original
receipts and clips are preserved. The repair removes the extra adapter wrapper;
original learner source remains unchanged. A new native attempt must verify the
repair after the installed CPU regression and independent review.

### Commands and installation

The baseline job is `configs/bench/upstream_vla_baseline_job.json`.
Its maximum wall time is 1,800 seconds. Run it through the installed coordinator:

```sh
ABC_BENCH_REPO=/home/user/aditya/RL/abc /home/user/aditya/RL/abc/.venv/bin/python -I -u -B -m abc_bench.runner --algorithm author-subset --checkpoint /home/user/aditya/RL/abc/cache/vla_abc130k_200000_v2.pt --training-config /home/user/aditya/RL/abc/configs/bench/upstream_vla_baseline_job.json --timeout-seconds 1800
```

The current run records `TORCHINDUCTOR_COMPILE_THREADS=1` and separate compiler
caches under `RL/builds/abc-baseline-reassessment-20261012`. These are resource
overrides. Stock fast inference remains enabled. RTC video rendering affects
timing. This qualification does not establish hardware latency for deployment.

The new baseline wheel passed 265 installed CPU tests and five subtests.
Its 183 Python files and all 933 original model assets match the saved source.
Evidence is under `RL/builds/abc-baseline-reassessment-20261012`.
Runtime and fixture tests establish execution. They do not establish task success.
Missing native scores remain unavailable.

### Original FastTD3 native qualification and bounded pilot

Run `author-subset-20261012T142454-566f77` completed 64 native controls.
It performed 106 original critic updates and 53 original actor updates.
Three sampled critic-gradient checks and two sampled actor-gradient checks passed.
The original checkpoint export is 15,832,066 bytes. Its SHA-256 is
`2c0c6fffc9903ee5a4f24e5e49cd08502c26cf12cc48ede052a4ae1ad4a66170`.
The coordinator charged 67.71 seconds. No full task episode completed in this smoke.
The original task reward remained zero. This is integration evidence, not a failure rate.

The client now uses one total RPC deadline. An independent slow-response probe
rejected a trickling response after 0.0606 seconds for a 0.06-second timeout.
Installed adapter version 0.1.2 preserves the original update equations.
This transport correction changes no training mathematics.

The bounded scratch pilot uses `configs/bench/fasttd3_scratch_subset_job.json`.
Its maximum is 4,000 controls and 1,800 wall seconds. It starts from scratch.
The batch size remains 32,768 and the replay capacity remains 51,200.
Original compile, AMP, network, optimizer, reward and update defaults are retained.
One training world, disabled tracking and a finite budget are diagnostic overrides.

Training uses fixed bottle variants and scales. The captured evaluation scene has
different variants and scales. Thus the evaluation changes geometry as well as layout.
The state policy sees privileged qpos, qvel and geometry labels; the VLA does not.
Do not attribute a short scratch-policy result to method quality or claim equivalent
training coverage. No reward shaping or pretrained policy was added to FastTD3.

The EXPO RGB/action host passed 29 installed CPU tests. The installed ABC snapshot
contains 186 Python modules and 933 authenticated original model assets.
The frozen pi0.5 rollout is measured above. Native EXPO update and post-update
evaluation evidence remain pending.
[Architecture source](ORIGINAL_METHODS_ARCHITECTURE.mmd) labels these execution states.

The 4,000-control pilot has now completed. It made 7,978 critic updates and
3,989 actor updates. Every placement reward was zero. Its frozen FP32 diagnostic
completed 3,540 controls in the exact captured scene: zero success and zero
bottles placed. Actor and normalizer fingerprints stayed unchanged. No more
training in this regime is scheduled.

The separate author-precision evaluation,
`author-subset-20261012T155412-eb00e3`, completed the same 3,540-control scene
with zero success and zero bottles placed. It used the same exported checkpoint,
saved bf16 AMP, compiled observation normalizer and eager actor. Actual actor
output was bf16; the environment boundary received float32. Policy, normalizer
and CPU/device RNG fingerprints stayed unchanged. Actor and critic updates were
zero. The coordinator charged 258.38 seconds. This corrects the evaluation
precision; it adds no training or evidence of convergence. The earlier FP32
result remains historical. Canonical worker receipt SHA256:
`afb9801ab8419961de945f2c6367509dc9783813262821d5c729ccb61a8fd467`.

[FastTD3 design review](FASTTD3_DESIGN_REVIEW.md) explains a material mismatch.
Changing 128 author-default worlds to one world increases updates and replay
sampling per new transition by 128 times. Original equations do not preserve
the original learning regime. This pilot establishes integration and records
a diagnostic outcome. It does not reproduce the paper's learning results.

Root controls serial GPU0 jobs and accepts the evidence. The baseline peer checks
original evaluation and the state host. The FastTD3 peer checks author mathematics
and the installed learner. The EXPO peer checks JAX, pi0.5 and the task adapter.
Other GPU jobs and Leela environments remain outside our process groups.

This explanation uses STE guidance. It has not had a full ASD-STE100 compliance check.

Sources: [original ABC](https://github.com/amazon-far/abc),
[original FastTD3](https://github.com/younggyoseo/FastTD3),
[QF3 project](https://qf3-rl.github.io/), and the original EXPO-FT checkout
at `21fe3d3b7d913c80836496817965932f49e0aedf`.
