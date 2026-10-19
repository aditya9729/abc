# Original EXPO fixed-data learning: 100 calls

The accepted ten-call experiment changed critic scores but did not improve the
selected action in the measured acquisition branch. Fresh heads selected
candidate44. C10 selected candidate41. Both failed the sampled lift/contact
criterion. Candidates12 and14 passed under the same recorded continuation.
These results concern one TRAIN state and a fixed action bank.

This experiment tests whether the learner can use the existing data with more
updates. It initializes from the same RTC2000 policy and fresh original EXPO
heads. It does not resume C10. The sealed phase is
`historical_online_fixed_data_100` in `abc_bench.historical_fixed_data_100`.
The old ten-call phase remains available with its original defaults.

## Recipe and scope

The original `RealTimeEXPOFTLearner` owns the update, losses, target networks,
actor filtering, and full checkpoint writer. Batch size64, UTD20, offline
ratio0, success-only actor sampling, and seed42 remain unchanged. Each call uses
`agent.update(agent, batch, 20, actor_batch)`. The original batch processor
retains its pending Future and two internal critic batches.

The corrected replay order is7069 demonstration rows,349 autonomous success
rows, then9899 historical timeout rows. The original terminal reward remains
unchanged. This provides17309 eligible critic starts and176 positive windows.
The21 demonstration trajectories share one source lineage. More updates do not
add independent corrective experience.

The schedule is a declared adaptation:100 updates on fixed historical TRAIN
data, with no online collection. This is not the author's full deployment loop.
It contains2000 critic updates and100 actor/editor updates. It consumes128000
critic examples and6400 actor examples. Expected positive-window exposure is
about7.39 visits per window; this is not a convergence guarantee.

## Measurements and stopping

Inspect the learner at calls0,10,30,100. Retain the existing terminal-window and
fixed-bank diagnostics. Also score the accepted64-candidate O72 bank twice,
using the current encoder/target critic and original head IDs6/4. These calls do
not sample proposals or edits. The same-process repeated outputs must match.
Historical scores from another compiled process are descriptive comparisons.

Eight added terminal windows describe the single autonomous success event.
Seven include the original buffer's backfilled tails. Do not interpret these as
eight independent successes. Measurement checks protect learner parameters,
optimizer counters, RNG state, and the pending sampling stream.

Stop after100 calls. Abort on nonfinite values, resource failure, cancellation,
or deadline. Do not extend automatically. No generalization-based early
stopping rule is claimed. A TRAIN rank or bootstrapped loss plateau is not a
reliable overfitting signal.

Save one full checkpoint at100 through the original writer. Evaluate it with
the original frozen evaluation path on the separate DEV starts. Keep training
execution, restored-checkpoint validity, and physical task success distinct.
The checkpoint does not include an exact restart of this diagnostic's replay
and sampling process.

## Resources and ownership

The worker has7200 seconds. The parent has7300 seconds. The outer operator has
7400 seconds and inherits the qualified descendant cleanup functions. GPU0
uses the existing exclusive campaign lease. All previous time charges remain.

This separate one-save operator requires29,200,425,701 free bytes at admission.
It retains an8GiB free-space floor,20,342,055,653 bytes for one full checkpoint,
and256MiB for other outputs. These are engineering limits. A five-second
filesystem census is not a write quota. The old64GiB-floor operator is unchanged.

Root owns installation, storage decisions, launch, and result acceptance. The
recipe agent implements the caller and preserves author code. The boundary
agent reviews measurement isolation and process handling. The storage agent
proposes exact retired files while protecting retained models and data.

The preparation artifacts are under
`/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/`:

- `fixed-data-100-preparation-20261019/PLAN.json`: frozen source, tests and phase.
- `fixed-data-100-release-01/`: new normally installed runtimes and bound inputs.
- `fixed-data-100-operator-preparation-20261019/`: resource contract and supervisor.

The first release passed79 tests in each installed prefix. Its first launch
failed before GPU admission. The generic job loader rejected7300 seconds,
although the phase validator accepted that parent limit. The corrected release
adds an exception for this sealed phase only. It passes103 tests per installed
prefix, including the actual outer loader and effective parent validation.
Full input authentication also passes. See `RELEASE_V2.json` and
`CPU_INPUT_AUTHENTICATION_V2.json` in the release directory.

Root launched `author-subset-20261019T044009-4af3af` at04:40UTC on19October2026.
The job uses `fixed-data-100-execution-02/plus_autonomous_349_before_timeouts`.
The first failed admission remains in `fixed-data-100-execution-01`.
Read the current process receipt before interpreting this launch record.
A launch is not a completed learning or task-performance result.

The independent result checker hashes and decodes saved selector tensors. It
checks repeated outputs, the archived candidate bank, and reported scores.
Ten checker tests and seven operator tests pass. Actual training acceptance
requires100 completed calls, finite measurements, a full checkpoint, and cleanup.
No100-call task result is established by this document.

The task uses fixed-base R1 Lite arms and grippers. It does not test whole-body
control. The [architecture](HISTORICAL_FIXED_DATA_100.mmd) marks the pending
physical evaluation. This explanation follows STE guidance. Full ASD-STE100
compliance was not checked.
