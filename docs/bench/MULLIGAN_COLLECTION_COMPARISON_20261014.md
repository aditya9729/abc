# Deployment collection: a matched released-policy comparison

The two original round-1 collection recipes give different results depending
on the deployed candidate-selection setting. On the same 50 starts:

| Released collection recipe | One action candidate | 32 candidates with ranking |
| --- | ---: | ---: |
| Sobol, no counterfactual demonstrations | 42/50 (84%) | 43/50 (86%) |
| Mulligan, no counterfactual demonstrations | 46/50 (92%) | 42/50 (84%) |

Mulligan leads by four successes at N1 and trails by one at N32. These are
one-seed development results. They do not establish a reliable overall winner,
a scaling law, or the paper's full-grid aggregate. Report both settings; do not
select a favorable candidate count after seeing this cohort.

## What is controlled

These runs use the unchanged original evaluator at author revision
`bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8`. They use the same first 50 Sobol8k
starts as the existing viewer, seeds 42/42, ten workers, and a 400-control limit.
The requested shard is complete; the full 8,000-point grid remains partial.
Both released checkpoint files remain unchanged. There is no new local
training, collection, or video pass in this comparison.

The model revisions are `15d7a9bd272d7111e2b76addf75c70ef385da19f`
for Sobol and `61dba7085b601d6e3ce944f76cd67e8c5428918e` for Mulligan.
Each is the released seed-1 DIVL checkpoint. Their recipes have the same
150,000-update IDQL and DIVL budgets, batch 1,024, and learner settings.
Published checkpoint metadata says step 150001, while the training summaries
say 150000. Keep both original fields; they are not fresh optimizer counters.

Both collection arms use the same 100 round-0 teleoperation episodes and 100
diagnostic rollouts. Each adds 100 successful round-1 team episodes. The actors
receive original human-supervised data; critic views retain the original source
and failure rules. The actor and normalizer are fixed during each DIVL stage.

## What is not controlled

Both strategies first promote the same 21 failure starts. Sobol fills the other
79 slots with Sobol starts. Mulligan uses 20 local perturbations and 59 coverage
starts. This compares the remaining allocation after shared failure promotion.
It does not compare failure targeting against pure uniform collection.

The released round-1 views share 21 actual source episodes and have 179 distinct
episodes in their union. Their supervision differs:

| Released metadata | Sobol | Mulligan |
| --- | ---: | ---: |
| New successful team episodes | 100 | 100 |
| Mixed-human successful episodes | 46 | 53 |
| Published human-tagged records | 1,667 | 1,928 |
| Possible valid human-tagged records | 1,567–1,667 | 1,828–1,928 |

The ranges account for 100 invalid records in each view. Mulligan has at least
161 more valid human-tagged records under this metadata accounting. Exact counts
need trajectory payloads, which this comparison did not download. Operator
minutes, resets, and discarded attempts are not measured. Equal successful
episode quotas do not establish equal human effort.

## Paired outcomes

At N1, both policies succeed on 40 starts. Sobol alone succeeds on two, Mulligan
alone on six, and both fail on two. At N32, both succeed on 36; Sobol alone on
seven, Mulligan alone on six, and both fail on one.

Within Sobol, moving from N1 to N32 fixes four starts and loses three. Within
Mulligan, it fixes one and loses five. Candidate count, random sampling, and
ranking change together. This does not isolate the critic's contribution or
explain the failures. A causal critic test would need a separate control that
chooses from the same candidates without Q ranking.

## Execution and verification

All four canonical and author processes exited zero. Each observed owned
process group is empty. Root checked cleanup before each serial GPU0 launch.
The jobs took 90.03, 103.90, 84.35, and 101.66 seconds in table order, N1 then
N32 per arm. Those times include different episode lengths and startup work;
they are not per-action latency measurements.

The data reader validates the original commands and saved configurations,
complete point identities, native rewards and lengths, cell summaries, success
totals, and unchanged checkpoint files. It accepts data only. Root separately
checks process cleanup and limits the scientific claims. The seven reader
checks are arithmetic and corruption checks, not simulated policy tests.

Actual evidence is under
`/home/user/aditya/RL/builds/mulligan-collection-results-20261014/`.
`RESULTS.json` contains the four scores and paired point IDs; the four
`*-cleanup.json` files record root's terminal checks. The
[compact index](MULLIGAN_COLLECTION_COMPARISON_20261014.json) pins the evidence.

The harness can now compare released collection choices under matched
evaluation conditions. A new causal study still needs controlled collection
cost, multiple training seeds, and separate final starts. SGS reset selection
and Real-Time EXPO online updates remain different interventions, as the
[scaling plan](DEPLOYMENT_RL_SCALING_PLAN.md) explains.

Root owns execution and interpretation. The preparation agent authenticates
assets. The results agent checks data and source contracts. The original author
evaluator owns policy and physics execution. The
[diagram](MULLIGAN_COLLECTION_COMPARISON_20261014.mmd) separates these roles.
STE guidance was used; full ASD-STE100 compliance was not checked.
