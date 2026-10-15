# Original Mulligan on R1 Lite

The harness now admits the original Mulligan training and frozen evaluation
programs through `abc_bench.mulligan_program`. The author owns the learner,
loader, evaluator and action sampler. The R1 task and data are explicit
adaptations in separately pinned packages.

Four modes cover native diagnostics, short original-training qualifications,
original training, and frozen evaluation. Admission checks bind the source,
runtime, task, data and root acceptance records. The canonical runner retains
the exclusive GPU0 lease and finite deadlines. These four modes have a
90-second process cleanup allowance; other modes retain their existing limits.

The monitored IDQL profile keeps the original 150,000-update ceiling. It uses
the author's existing configuration for TRAIN evaluation and model saves every
10,000 updates. These evaluations consume the original shared RNG. Their
trajectory differs from the author's end-only evaluation schedule. Root applies
the declared stop rule after five consecutive complete zero-success monitors
for both N1 and N32. Positive results reset the count. This is a compute rule,
not a test for convergence or overfitting.

Frozen evaluation can use a full accepted model or a separately accepted
periodic IDQL policy from a stopped producer. The latter preserves the failed
producer status. Root must inspect and accept the saved policy separately.
An earlier policy does not inherit later optimizer counters. Partial acceptance
permits inference only; it cannot admit training resume or a DIVL parent.
The periodic path requires a stopped producer below 150,000 updates. It does
not admit an earlier checkpoint from a naturally completed full run.

The fixed frozen protocol uses DEV9100–9119, ten native workers, seed42,
and either one candidate or selection among32 candidates. TRAIN monitors are
separate from DEV results. All outcomes require native trace and resource
checks. CPU tests do not establish task performance.

## Verification and operation

The integrated wheel passed **364 installed CPU cases**. These cover the eight
Mulligan suites, shared subset/author admission, and five cancellation and
environment tests. The tests use explicit doubles and owned synthetic child
processes. Original parser tests load configuration with CUDA hidden.
No original model, learner update or simulator runs in this qualification.

Ruff formatting and lint checks passed from the package source root. All249
source, wheel and installed payload files match. The seven production modules
match their previously qualified runtime versions. Nine newer viewer files
were preserved. Six test files received import formatting changes. The two
earlier lint failures remain recorded with their correction.

Use the existing installed runner, with an actual admitted job:

```text
python -I -B -m abc_bench.runner --algorithm author-subset \
  --training-config /absolute/path/to/admitted-job.json \
  --timeout-seconds <accepted-parent-bound>
```

This is a campaign-specific integration. It requires the pinned external
runtimes, manifests and acceptance records. A Git checkout alone does not
supply them. Never copy a receipt to authorize another checkpoint or run.
The live training runtime was not replaced by this source integration.

[Exact test commands and build evidence](/home/user/aditya/RL/builds/other-methods-plan-20261015/abc-mulligan-tracked-integration-release-01/BUILD_EXECUTION.json),
[installed release evidence](/home/user/aditya/RL/builds/other-methods-plan-20261015/abc-mulligan-tracked-integration-release-01/RELEASE.json),
and [current results](/home/user/aditya/RL/sadhana-r1lite-rl-hillclimb/packages/r1lite-rl/docs/rl/R1LITE_RL_COMPARISON.md)
keep software checks and policy measurements separate.

[Architecture source](MULLIGAN_R1_HARNESS_20261015.mmd).
Root integrates, executes and accepts results. Agents prepare source and
review evidence. This explanation follows STE guidance; full compliance has
not been checked.
