# Scaling RL with deployment experience

The objective is to measure whether additional useful experience improves a
deployed policy. The harness must control collection, learning, and evaluation.
More updates or more simulator instances do not establish that improvement.
This document is a research-backed plan. It contains no new training result.

## What the papers contribute

| Work | Decision it changes | Implication for this harness |
| --- | --- | --- |
| [SGS: A Balanced Data Diet](https://arxiv.org/html/2610.12465v1) | Which configuration starts the next simulated episode | Track recent success per reset configuration; practise configurations with intermediate success while retaining coverage |
| [Real-Time EXPO-FT](https://arxiv.org/html/2609.18207v1) | How to improve and execute a delayed policy | Reuse the released learner, slow candidate inference, fast current-observation edits, critics, and original replay/update rules |
| [Mulligan](https://arxiv.org/html/2610.05882v1) | Where to collect the next supervised deployment episodes | Revisit observed failures, fill unexplored initial-state regions, and collect corrections or complete demonstrations |

SGS is an outer loop around PPO. It selects resets from a fixed bank using
rolling Boolean outcomes. It does not select old replay transitions. Its
manipulation setup includes reaching, stable-grasp, and near-goal resets.
If every configuration has the same zero success estimate, the sampler cannot
identify a useful frontier. The reset bank must contain learnable cases.
The [author site](https://sgs-rl.github.io/) still labels code as coming soon.
We have no exact SGS author implementation to launch at this check.

Mulligan can target hard failures because an operator supplies missing behavior.
It combines failure promotion with coverage selection. Some corrections are
followed by a complete demonstration from the same start; these demonstrations
count toward the collection budget. Its HiL-IDQL variant separates actor
imitation from value learning and reranks sampled actions. The paper trains
the critic on failures as well as successes. Preserve each released task's
actual actor masks and intervention-context rules when reproducing it.
See the [author implementation](https://github.com/ankile/mulligan).

Real-Time EXPO-FT addresses stale observations during slow policy inference.
It needs valid action timing, rewards, and an appropriate initial policy.
Its robot experiments still require human resets. The one-GPU synchronous
training driver retains asynchronous action inference. The prior
[implementation review](REALTIME_LEARNER_REUSE_20261013.md) records the available
code and the missing Kinetix integration. Preserve the released equations and
disclose paper/code differences.

These methods support a shared collection-and-learning workflow. They do not
prove that combining all three produces better policies. SGS with EXPO's
off-policy replay would be a new experiment. Mulligan without human corrections
would change its supervision. We will assess those changes separately.

## Existing infrastructure and missing connections

The inspected repositories provide different parts of the system:

| Existing component | What it provides | Remaining connection |
| --- | --- | --- |
| Legacy `nrh/scaling.py` and `SCALING_AND_GENERATION.md`, Sadhana branch `e59322e` | Nested data-diversity cohorts, fixed training budgets, lineage checks, comparisons | An outcome-driven choice of the next collection start |
| `abc_bench/runner.py` and `author_program.py` | GPU0 job supervision, bounded execution, original EXPO program launches | A narrow launcher for original Mulligan commands; the current EXPO whitelist does not admit them |
| `abc_bench/captured_reset.py` | Captured ABC validation resets | A separate validated TRAIN reset bank; validation captures must not become curriculum data |
| Current Sadhana `mixture.py`, `rollout_artifacts.py`, and `rollout_execution.py`, inspected at `85fe831` | Real-data floor sampling, persistent rollout evidence, execution events | Method-specific collection feedback and an original RL learner connection |
| Existing result viewer | Episode results and full videos | Learning curves against data, compute, and supervision cost; per-start failure maps |

The current real-data floor applies over an epoch. It is not an exact
per-update actor/critic balance. The legacy comparison also requires equal
training budgets; a compute-scaling report needs a separate comparison rule.

The legacy harness remains in its worktree under
`/home/user/aditya/RL/sadhana-realtime-expoft-abc/harness`.
The current SDK is `/home/user/aditya/RL/sadhana`. These are separate versions;
their APIs are not assumed interchangeable. The current SDK describes RL
updates as pending. This review did not rerun its integration tests.

Keep orchestration and reporting in the harness. Keep method updates in author
code. Do not route an original-method experiment through our older reference
TD3, QF3, or candidate-selection substitutes. Do not impose the SFT mixer's
real-data fraction on an author's RL replay recipe without declaring a change.

## Data contract for the feedback loop

Store episode identity, policy revision, task and reset identity, source family,
real/sim origin, timestamps, observed state, proposed and executed actions,
reward, termination, truncation, success, and failure stage. Keep intervention
intervals and operator time when supervision exists. Real-Time EXPO additionally
needs observation age, action prefixes, and inference/execution timing.

Keep valid failures. Corrupt or unverified records remain excluded, but a failed
task is not automatically bad data. Derive actor-eligible and critic-eligible
views using the selected author's recipe. Do not teach imitation to copy all
failed policy actions. Record which sources actually enter each update.

Use three distinct evaluation roles:

1. Collection feedback may choose the next training starts. Mulligan's round
   evaluation belongs here once its failures guide collection.
2. Development evaluation selects checkpoints and stopping decisions.
3. Final evaluation stays outside collection, reconstruction, and tuning.

A feedback start promoted into training is no longer held out for future
rounds. Preserve its history and evaluate later policies on fresh independent
starts. Do not feed final-test failures back into the training bank.

Some released Mulligan simulation rounds explicitly use evaluation-grid
feedback. Preserve and disclose that protocol when reproducing those recipes.
For our later scaling study, reserve a separate untouched final cohort; do not
describe every released grid as an independent final test.

For the R1 Lite pipeline, retain approved real episodes and annotations.
The current SDK consumes published Alaya artifacts; do not bypass that
interface to read another project's private data. Atlas/Engram ownership
remains subject to the team's published contract. Leela supplies validated
physical scenes and reset states. Sadhana owns collection allocation and
post-training. A reconstructed or re-rendered video alone is not an RL
transition; simulation must execute actions and produce aligned outcomes.

## First experiment: original Mulligan on Square-Narrow

Mulligan is the most direct first test of the deployment-data question. Its
release includes the simulator, start manifests, collection data, checkpoints,
and training recipes. This reduces new integration work. It does not replace
the Real-Time EXPO lane or establish performance on R1 Lite.

Use the released policy family first. Do not substitute pi0.5 or an R1 Lite
controller while checking the recipe. Proceed in this order:

1. Reproduce the original start-design check from locked manifests on CPU.
2. Run complete episodes from released initial and improved checkpoints on
   the same 50-start simulation shard. Save full successes and failures.
3. Retrain one released Square-Narrow round with its exact data, actor/critic
   rules, and original entrypoints. Record all recipe deviations, if any.
4. Compare uniform and targeted collection at the same round and supervision
   budget. Compare the original single-candidate and 32-candidate reranking
   modes. Confirm useful results across the paper's five
   training seeds and the full locked evaluation grid before claiming its
   reported performance.

For step 2, pass the same authenticated grid through `--grid-file` for both
rounds, with matching shard and explicit environment/evaluation seeds. The
R0 and R1 recipe defaults use different grids. A matching shard number alone
does not match starts. Document R0's shared-grid override as a pilot change.
The reviewed grid evaluator does not expose a video flag. Add a declared
rendering recorder without changing the original policy or task behavior.

The clean first collection comparison uses these released R1 recipes:

| Collection strategy | Original recipe | Evaluation settings per checkpoint |
| --- | --- | --- |
| Sobol coverage, without counterfactual demonstrations | `square-narrow-r01-sobol-no-cf` | `--stage divl_head --num-action-samples 1` and `32` |
| Mulligan targeting, without counterfactual demonstrations | `square-narrow-r01-mulligan-no-cf` | `--stage divl_head --num-action-samples 1` and `32` |

Both use the same R0 Sobol data, 100 R1 collection episodes, and the same
diagnostic rollout source. This is a clearer test of collection allocation
than the full baseline-versus-Mulligan recipes, which also change the initial
design and counterfactual demonstrations. With one candidate, ranking cannot
change the selected action, but the evaluator still computes Q. Do not call
that timing measurement critic-free. Record the additional computation for
32 candidates. Restore the full Mulligan recipe as a separate comparison.
This tests the combined effect of candidate count and ranking. Isolating the
critic requires a control that selects from the same candidate pool without
Q ranking. Do not invent an author evaluator flag or claim that isolation here.

Reusing recorded deployment data establishes data reuse and retraining. It
does not demonstrate a fresh autonomous collection loop. Replaying recorded
operator segments is also not a new human intervention policy. Label both.
The 50-start run is a visible pilot; it does not reproduce the full grid score.

The original README documents these entrypoints. They were inspected, not run:

```bash
python -m mulligan.sampling.sim_design \
  --config configs/sim/square_narrow/r01_sampler.yaml \
  --out outputs/check/narrow_r01 --check

bash scripts/sim/eval_cell.sh square-narrow-r01-mulligan \
  --seed 1 --shard 1 --n-shards 160

bash scripts/sim/train_cell.sh square-narrow-r01-mulligan --seed 1
bash scripts/sim/train_divl_heads.sh square-narrow-r01-mulligan \
  --seed 1 --parent local
```

The release reports hours per seed, not seconds. Measure this host before a
campaign. Use GPU0 serially through the shared runner after adding its narrow
author-command adapter. Keep caches and outputs under `/home/user/aditya`.
Downloads remain scoped to the selected task, round, and seed.

## What scaling must measure

| Question | Change | Hold fixed | Evidence |
| --- | --- | --- | --- |
| Does deployment experience help? | More collected rounds or independent episodes | Task, method, initialization rule, evaluation distribution | Success versus collection episodes and operator minutes |
| Does data selection help? | Uniform versus targeted collection or reset selection | Learner, budget, available starts, supervision, seeds | Paired success and coverage comparisons |
| Does the released candidate-selection mode help? | One candidate versus 32 candidates with critic ranking | Actor/data/checkpoint and evaluation starts | Combined sampling-and-ranking gain, latency, and candidate count |
| Does extra training help? | More optimizer work on the same frozen data | Dataset and starting checkpoint | Success versus GPU time; repeat-data control |
| Does parallel simulation help? | Number of active worlds | Defined replay/update scaling rule and task distribution | Both success versus transitions and success versus wall time |

First establish sample efficiency on GPU0. A million-world result is not a
requirement for a useful pilot. More worlds can change batch sizes, replay age,
and optimizer updates per new transition. Record those changes; do not repeat
the earlier one-world versus vector-update mismatch. SGS's published scale
study increases environment count and total collected experience; it is not
a fixed-compute comparison.

For Real-Time EXPO, retain the proposed original Kinetix integration as its own
lane. Show success as real transitions accumulate, with matching delay and
complete baseline/post-training episodes. A useful first learning curve can
record 10k, 25k, 50k, and 100k transitions. These are proposed reporting points,
not a changed author update schedule. Compare compute and latency as well as
reward. Do not merge its scores with Mulligan's different task and backbone.

For SGS, prepare reset/outcome interfaces and monitor the author release.
Under the user's original-implementation requirement, this plan adds no
unofficial SGS implementation. The existing [SGS study](SGS_RECIPE_STUDY.md) records
paper details and reset prerequisites.

Use validation for early stopping. Set stopping rules before the run. Preserve
the author's full schedule for an exact reproduction; label an early-stopped
pilot separately. A flat curve is a result to inspect, not a reason to keep
spending compute without a changed hypothesis.

## Deliverable and ownership

The viewer should show success against new transitions, collection episodes,
operator minutes, and GPU hours. Add per-start coverage and failure maps,
source counts entering updates, inference latency, and full episode videos.
Keep author reference scores separate from local measurements. Missing
measurements stay unavailable.

Root owns method integration and experimental decisions. The SGS reviewer
checks reset allocation and scaling assumptions. The Mulligan reviewer checks
supervision and original recipe boundaries. The harness reviewer checks
existing interfaces and evaluation independence. Project owners publish
versioned data and scene artifacts; this plan does not establish new bindings.

Independent method and harness notes are under
`/home/user/aditya/RL/builds/deployment-rl-scaling-study-20261013/`.
The Mulligan source check used revision
`bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8`. Source inspection confirmed its
evaluation controls and public metadata; it did not load models or datasets.

[Architecture source](DEPLOYMENT_RL_SCALING_PLAN.mmd). Solid connections show
existing component relationships. Dashed connections show proposed integration.
No simulator, GPU job, dependency installation, or dataset download was made
for this study. Text uses STE guidance; full ASD-STE100 compliance was not checked.
