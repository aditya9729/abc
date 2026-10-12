# Baseline reassessment and original learners

The short ABC comparison does not show that the published ABC baseline fails.
Its evaluation recipe differed from upstream defaults. We are checking the
baseline before drawing conclusions about new learners.

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
| FastTD3 | Full state policy from scratch | None | Original learner passed CPU updates and checkpoint tests. Native ABC results are pending. |
| EXPO-FT | Q-guided action editor and distilled VLA | pi0.5 | Original JAX learner passed CPU fixture updates. Native pretrained-policy results are pending. |

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

Root controls serial GPU0 jobs and accepts the evidence. The baseline peer checks
original evaluation and the state host. The FastTD3 peer checks author mathematics
and the installed learner. The EXPO peer checks JAX, pi0.5 and the task adapter.
Other GPU jobs and Leela environments remain outside our process groups.

This explanation uses STE guidance. It has not had a full ASD-STE100 compliance check.

Sources: [original ABC](https://github.com/amazon-far/abc),
[original FastTD3](https://github.com/younggyoseo/FastTD3),
[QF3 project](https://qf3-rl.github.io/), and the original EXPO-FT checkout
at `21fe3d3b7d913c80836496817965932f49e0aedf`.
