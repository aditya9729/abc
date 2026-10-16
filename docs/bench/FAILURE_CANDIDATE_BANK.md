# Earlier-failure candidate coverage and selection

This mode captures an original EXPO proposal bank at **TRAIN21000 after control128**.
It performs no updates or native controls. The original bank capture is now complete;
the later candidate0 branch timed out without recovery. The other three selected
branches remain unrun.

The accepted capture contains the exact reset/replayed commands,129 raw RGB frames,
full simulator states and public observations. At128 the historical judge is false;
the remote is outside the tray and has no finger contact. A previous fresh base-bypass
continuation timed out after772 controls. Historical policy RNG, queued actions and
normalized planned prefix were not retained. This is a physically reproducible origin,
not an exact original-controller restart.

The earlier terminal-window experiment cannot answer this question: all96 generated
proposals succeeded after the object was already released and resting in the judge's
accepted region. The TRAIN20044 identity fixtures also do not supply this bank. Their
delayed observations did not execute the newly generated shared prefix.

## Source and data flow

The existing `abc_bench.policy_path_probe` retains its old identity schema and adds
`sadhana.failure_candidate_bank_canonical/1`. Both use the existing canonical lease,
GPU0 occupancy rules, host-memory guard and1800/1900-second child/parent deadlines.
The new caller is `abc_bench/failure_candidate_bank.py`. Root owns installation and
execution. No new launcher or learner is introduced.

The caller authenticates the accepted O128 RGB/state14 and existing original runtime,
RTC2000 parameters, task, statistics and author source. It reuses the qualified original
`build_agent` initialization and the original output transformations. The actor sees
only RGB and measured joint/jaw state; provenance and object/judge fields are not inputs.

1. Construct the original full learner with seed42 and fresh heads. Save its actual RNG.
2. Call original `sample_pre_cache` with `prefix_padded=None`. Save32 horizon16 proposals.
3. Give the same returned RNG to the original no-edit32 and full32+32 selectors at delay0.
4. Save normalized8-command candidates, target-Q values, selected indices and RNGs.
5. Convert every candidate with original zero-padding/output transforms and final float32 cast.
6. Require bitwise command, planned-prefix and outgoing-RNG parity with public full/bypass calls.
7. Verify unchanged parameter digests, six zero optimizer counters and immutable input files.

The two selector batch shapes can give different Q arithmetic on their shared32 base
candidates. Both score arrays are retained; they are not assumed equal. These are fresh-head
rankings, not the trained two-call-fit snapshot or calibrated probabilities.

## Result contract

The result schema is `sadhana.failure_candidate_bank_result/1`, with completion status
`failure_candidate_bank_complete`. `artifact_accepted` and `benchmark_accepted` remain false.
Root must accept the actual numerical output separately.

| Field | Meaning |
| --- | --- |
| `origin_run_id`, `capture_acceptance`, `capture_result` | Exact accepted physical origin and evidence |
| `arrays.base_normalized_h16` | Pinned float bank[32,16,14] |
| `arrays.full/cand`, `arrays.no_edit/cand` | Normalized windows[64,8,14] and[32,8,14] |
| `arrays.physical64_float32` | Original transformed proposed commands[64,8,14], before native clipping |
| `arrays.full/qs`, `arrays.no_edit/qs`, corresponding `/idx` | Original target-Q rankings and choices |
| `arrays.rng/constructor`, `arrays.rng/selector_start`, public outgoing RNGs | Actual original call states |
| `branches.unique` | Ordered `{branch,candidate_index,physical_float32_sha256,aliases}` |
| `branches.aliases` | All four requested labels, retained even when commands coincide |

The choices are base0, full winner, no-edit winner and one uniform64 draw from a separate
`random.Random(42)`. Deduplication uses the complete proposed float32 command window bytes.
It does not erase labels or claim distinct post-clipping actions. No policy RNG is consumed
by this diagnostic choice.

## Planned physical comparison — outside this caller

Root's separate public task wrapper will replay128, send one selected8-command window,
then return O136 to a **fresh** original base-bypass controller, independently seeded42.
That controller has at most764 controls remaining. It starts with no prefix or pending
future and delay0, then uses original H16/replan8/delay5 behavior. If a native terminal
occurs inside the forced window, it stops there. Proposed/applied actions and clipping
must remain distinct. No learner or sampler patch is needed.

This deliberately resets controller state at136. It is a conditional intervention,
not an exact RTC queue continuation, a full-EXPO rollout, or an on-policy Q validation.
A recovering inspected proposal and a failing full winner would show a local selection
miss under this continuation. Four failures would not show that all64 proposals fail,
that the state is unrecoverable, or that memory caused failure. Preserve the historical
judge and report strict placement separately; do not infer strict success from its score.

Use the bank first, then at most four deduplicated branches. A simple full recovery
rollout would provide only one outcome and would mix selection with later policy behavior.
The proposed comparison better separates the first-window decision, within the stated limit.

The old two-input capture took489.361 seconds; the old772-control continuation took326.232
seconds of canonical wall time. These are cost references, not new timing guarantees.
Four full branches would use at most3,600 physical controls including repeated prefixes,
plus reset settling. Output proposals are capped at64MiB; each future recorded branch
can reuse the existing250MB trace cap. Root schedules each finite lease after Atlas releases GPU0.

## Qualification and handoff

Preparation and exact pins live in
`builds/r1lite-realtime-expo-20261014/failure-candidate-bank-preparation-20261016`.
`JOB.template.json` intentionally leaves the installed canonical interpreter null.
The original numerical interpreter is already pinned in `PAYLOAD_CONFIG.json`; it is
not copied or installed by this change. Root must build/install the small benchmark
closure, bind the actual job interpreter, and authenticate the frozen payload.

Pure CPU fixtures cover capture joins, original-call RNG plumbing, conversion padding,
public parity refusal, deterministic aliases, and unchanged identity-mode dispatch.
They use explicit doubles and establish no original numerical parity or native outcome.
The first test invocation preserved a legacy interpreter-location failure under ABC's
external-target venv; the unchanged cases passed with the existing copied canonical Python.

[Architecture](FAILURE_CANDIDATE_BANK.mmd). Root owns execution/acceptance; this agent
prepares the inference caller; Sadhana owns the separate task wrapper; Leela owns physics.
The text follows STE guidance without a complete compliance check.

## Completed bank capture

Run `author-subset-20261016T171400-2049ea` completed in337.460 canonical seconds.
It saved32 base and32 edited candidates at the accepted O128 input. All10 exact
parity checks, including public-call checks, passed. Eight parameter-tree digests
and all six zero optimizer counters stayed unchanged. No training or simulator
control occurred.

| Choice | Candidate index | Source | Selected target-Q value |
| --- | ---: | --- | ---: |
| Base bypass | 0 | Base proposal0 | Not selected by Q |
| Full selector | 39 | Edited version of base proposal7 | -0.243566811 |
| Selector without edits | 25 | Base proposal25 | -0.272201389 |
| Separate seeded uniform draw | 14 | Base proposal14 | Not selected by Q |

All four physical command windows are distinct. The fresh-head Q values are
ranking outputs, not calibrated success probabilities. Shared base windows are
bitwise equal; their scores differ by up to0.000145853 between the two selector
batch shapes. Keep both score arrays. The full winner proposes a negative right
jaw value in part of its window. Preserve the native clipping and applied-action
records when testing it; do not silently repair the proposal.

The result index has38 entries:36 persisted NPY arrays and two explicit null masks.
The parent retains the old identity supervision/cleanup label. Its worker and
result correctly identify the new failure-bank schema. This is not a rerun of
the old two-input identity experiment.

Root prepared four hash-bound physical window descriptors. The installed Sadhana
consumer accepts them. The earlier native qualification used recorded commands,
so it does not supply outcomes for these new candidates. The later candidate0
branch has now completed900 native controls:128 replay,8 forced and764 fresh-base
controls. It timed out with no recorded remote finger contact. The remote position
stayed unchanged. Native clipping affected0 forced and131 continuation controls
(jaws only). The three other branches remain unrun after GPU0 occupancy refused
the next admission. No optimizer update or recovery improvement occurred.

[Measured branch report](/home/user/aditya/RL/sadhana-r1lite-rl-hillclimb/packages/r1lite-rl/docs/rl/FAILURE_CANDIDATE_BRANCHES.md) ·
[Video viewer](http://127.0.0.1:8770/original-methods/r1lite-recovery/#failure-candidate-branches).

[Bank result](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/failure-candidate-bank-execution-01/bank/RESULT.json) ·
[Root acceptance](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/failure-candidate-bank-execution-01/ROOT_ACCEPTANCE.json) ·
[Branch descriptors](/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/failure-candidate-branches-preparation-20261016/PREPARATION.json).
