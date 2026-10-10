# Actual small ABC comparison

All three conditions ran in the original ABC-Sim MJWarp environment on GPU0.
They used the same released ABC-VLA, five bottles, cameras, sampling settings,
layout seeds and success judges. Each evaluation had three complete episodes.
Training stopped at the specified small budgets. No further training is scheduled.

| Condition | Full-task successes | Bottles entered in layouts 9011 / 9012 / 9013 | Training |
| --- | --- | --- | --- |
| Frozen original ABC-VLA | 0/3 | 3 / 4 / 4 | None |
| ABC-VLA plus original-author ResFiT | 0/3 | 1 / 1 / 0 | 258 warmup controls; 32 training controls; 160 critic and 32 actor updates |
| ABC-VLA with QF3 paper implementation | 1/3 | 4 / 4 / 5 | 1000 warmup controls; 1000 training controls; 128 critic and 16 actor updates |

QF3 solved layout 9013 at control 996. Both placement-history success and the
original simultaneous-success judge were true. Its final evaluation preserved
the recorded learner, replay and training random-number state. ResFiT evaluation made no actor or critic updates.
A native tensor-state fingerprint was not recorded for ResFiT.
This is one observed success in a small development cohort. It cannot establish
reliable improvement or reproduce the paper percentages.

The [viewer](http://127.0.0.1:8770/small-comparison/) contains all nine actual
camera clips. Select layout 9013 to see the QF3 success. All nine clips decoded
and advanced in the browser. Desktop and mobile checks passed. No page errors
or external browser requests were recorded. Videos are hash-checked before copying.

Original ABC policy and simulator code remain unchanged at `d0832d12`.
Original ResFiT learner code remains unchanged at `66262b06`. The adapter supplies
ABC observations, physical controls, real demo replay and a finite training budget.
QF3 adjusts the flow head with rank 4 LoRA. It does not add an action residual.
Its actor equations follow the [paper](https://arxiv.org/pdf/2610.08789v1).
Author code and the exact ABC initializer were unavailable in the checked sources.
[QF3 fidelity notes](/home/user/aditya/RL/sadhana-qf3-early-stopping/harness/docs/QF3_SMALL_COMPARISON.md)
list the local critic, optimizer, LoRA and evaluation choices.

The short training budgets differ. This comparison does not isolate algorithm quality.

The ResFiT result has a measured data confound. The single successful demo has
three bottles. Its action ranges omit some base-policy commands in the five-bottle
scene. The unchanged author scaler then alters those commands even with zero residual.
It changed 801, 972 and 998 controls above tolerance 1e-6 in the three episodes.
For example, the first left_joint2 command changed from 0.48959 to 0.81293 radians
before the residual acted. This result does not isolate the learned residual's effect.
A representative demo pool and an explicit scaling control are needed for that claim.
The demo is source-judged, not Atlas or human expert approved.

The baseline took 339.406 seconds. Original ResFiT training took 96.231 seconds;
its frozen evaluation took 316.753 seconds. QF3 training and evaluation took
516.163 seconds. These times include loading, simulation, inference, learning,
evidence writing and teardown. They are reserved job wall times, not kernel timings.

The native runs completed without nonfinite training failures. Original-author
CPU tests also exercised real updates and checkpoint loading. Fifteen coordinator
and reporter tests passed, with five subtests. Independent review checked final
cohorts, actual seeds, source and artifact identities. OpenQodex and Jev support
code and decision review; they do not establish benchmark correctness.

[Commands and recipe changes](SMALL_AUTHOR_COMPARISON.md) explain how to inspect
and repeat the jobs. [Measured receipt index](small_author_comparison_results.json)
pins the actual runs and browser check. Raw data, weights and videos stay outside Git.
[Architecture source](SMALL_AUTHOR_COMPARISON.mmd) shows the implemented boundaries.
Root coordinates jobs and acceptance. Peers check the host, data, learner, paper
method and report. This text uses STE guidance; no full compliance check was done.
