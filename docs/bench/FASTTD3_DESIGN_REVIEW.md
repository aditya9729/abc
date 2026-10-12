# FastTD3 design review for ABC-Sim

The one-world pilot verifies the original learner integration. It does not
reproduce FastTD3's training regime. Keeping the author equations and batch size
did not preserve the amount or diversity of new data per update. This mismatch
must be corrected before another scratch-training campaign.

## Sources and scope

This review reads [paper section 2.1 and 2.2](https://arxiv.org/html/2505.22642v2)
and the [author repository](https://github.com/younggyoseo/FastTD3/tree/229ed59bbf43ea2f7a2d5d90d1076314839944d7).
The pinned author commit is `229ed59bbf43ea2f7a2d5d90d1076314839944d7`.
It includes later optional extensions. Our checkpoint selects `agent=fasttd3`
and an empty `sim_type`. It uses neither SimbaV2 nor SimNorm.

The paper examines parallel environments, large batches, distributional value
learning, clipped double Q-learning, model size, exploration, update frequency,
and replay capacity. These choices work together. Parallel resets supply varied
experience. Large batches use that experience. Few updates per new transition
limit repeated fitting. GPU replay, mixed precision and compilation reduce cost.
The authors also distinguish task return from useful robot behavior.

## What the installed learner preserves

The exported pilot checkpoint contains these actual settings:

| Choice | Actual setting | Meaning for our adapter |
| --- | --- | --- |
| Actor | ReLU MLP, hidden widths 512/256/128, tanh output | Original state policy; no VLA initializer or added action residual |
| Critics | Two ReLU MLPs, widths 1024/512/256, 101 categorical atoms | Original C51 projection and losses |
| Value support | -250 to 250 | Original generic defaults; not tuned for ABC |
| Double Q | Enabled | Original smaller expected target distribution; actor maximizes the smaller critic expectation |
| Update schedule | Two critic updates per vector step; one actor update | Original delayed-policy rule |
| Targets and returns | tau 0.1; gamma 0.99; one-step return | Original target updates and timeout bootstrap semantics |
| Exploration | Gaussian scale sampled per environment, 0.001 to 0.4 | Only one simultaneous scale in our pilot |
| Target smoothing | Noise scale 0.001; clip 0.5 | Separate from exploration noise |
| Optimizer | AdamW, learning rates 0.0003; weight decay 0.1 | Original defaults |
| Execution | CUDA, bf16 AMP, compile enabled | Original training settings; separate evaluator correction described below |
| Normalization | Empirical observation normalization; reward normalization off | Original implementation; input coverage remains our responsibility |

The local `train.py` boundary accepts an environment factory and read-only
observer. Source seals verify the original network, replay and normalization
files. Four protected function bodies remain identical to the author AST.
This establishes source identity. It does not establish experimental fidelity.

## The quantitative mismatch

Let E be the number of parallel environments, U the critic updates per vector
step, and B the total batch size. One vector step provides E new transitions.
Gradient updates per new transition are U/E. Sampled critic rows per new
transition are U*B/E. These are different quantities.

| Quantity | Author BaseArgs configuration | Our pilot |
| --- | ---: | ---: |
| E | 128 | 1 |
| U | 2 | 2 |
| B | 32,768 | 32,768 |
| U/E | 0.015625 | 2 |
| U*B/E | 512 | 65,536 |
| Replay capacity | 51,200 per world; 6,553,600 total | 51,200 total |

Both update pressure and replay sampling per new transition increased by 128
times. This is an arithmetic comparison with the author BaseArgs configuration,
not a claim that every published task uses 128 environments.

The actual run collected 4,000 transitions and made 7,978 critic updates.
It sampled 261,423,104 critic rows with replacement. That is 65,355.776 sampled
rows per collected transition, including the learning delay. The first critic
update had only 12 stored transitions. A 32,768-row batch therefore repeated
each row about 2,731 times on average. Repetition cannot supply new trajectories.
This analysis identifies excessive reuse. It does not prove measured overfitting.

## Other limits that affect interpretation

- Training completed one 3,540-control episode and 460 controls of the next.
  Every task reward was zero. No positive placement example reached the learner.
  The data cannot yet show whether its value estimates guide useful task learning.
- Five training bottles used variant 0 and unit scales. The captured evaluation
  used variants 18, 16, 9, 7 and 10, with random scales. These are unseen geometries.
  The 95 variant indicators are constant during training. With author epsilon
  0.01 and zero standard deviation, an unseen indicator can normalize to +100;
  a formerly active indicator can become -100. This is a specific input shift.
- The support spacing is 5. Rewards are 0, 0.2, 0.4, 0.6, 0.8 and 1. With gamma
  0.99, nonnegative discounted returns are bounded by 100. The support contains
  that bound, but includes unused negative values. C51 interpolation can encode
  fractional rewards. This spacing does not make the reward mathematically invalid.
- The actor sees all 216 privileged state channels, including object poses,
  velocities and geometry labels. These are not all available on a real R1 Lite.
  The author supports separate actor and critic inputs. Our current adapter does
  not use that feature. Deployment requires a defined observable actor input.
- ABC-VLA starts pretrained. FastTD3 starts from scratch. Their short-run scores
  do not isolate differences between RL learners.

## Actual evidence and current decision

Training run `author-subset-20261012T143734-05cba8` completed 4,000 controls,
7,978 critic updates and 3,989 actor updates. The checkpoint SHA-256 is
`fd28f78a8d0f86d9060610b357009f605fe5b2a8fb32c59c5f49c2f339358f12`.
Sampled gradient checks were finite. This is execution evidence.

Frozen run `author-subset-20261012T144536-ab0669` completed the exact captured
scene and full horizon. It placed zero bottles and had zero success. It made
zero updates. Full actor and normalizer fingerprints stayed equal. Its adapter
used FP32 eager inference. The author evaluator uses configured AMP and, when
enabled, a compiled normalizer. Preserve this run as a labeled diagnostic.
The corrected author-precision evaluation remains pending until a new actual
receipt exists. It must use the same checkpoint and require no more training.

Do not extend the current one-world training run. Before another campaign,
qualify an author-native task and a source-supported parallel ABC setup.
Measure new-transition throughput and reward discovery before scaling updates.
Use TRAIN and validation scenes from the same documented randomization family.
Keep final held-out scenes separate. Any reward shaping, demonstrations,
smaller-batch regime or residual adapter must be a separately named condition.
A 256-row single-world batch would restore the sampling ratio of 512. It would
still leave the update ratio and exploration diversity different from the author
128-world setup. It is not a substitute for a reproduction.

Root checks design assumptions, controls GPU0 jobs and accepts results. The
FastTD3 peer checks source identity and precision. The independent peer checks
the claims and viewer. The EXPO peer prepares the separate pi0.5 experiment.
Leela jobs remain outside our process groups.

[Architecture source](FASTTD3_DESIGN_REVIEW.mmd) shows the executed pilot and
the required training conditions. This text uses STE guidance. It has not had
a full ASD-STE100 compliance check.
