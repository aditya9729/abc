# ABC benchmark engineering instructions

Read `/home/user/aditya/AGENTS.md` and the shared coordination board first.

This checkout is the Sadhana-owned ABC benchmark. Use branch `staging_attempt1`. Preserve upstream behavior. Keep changes in `abc_bench`, `tests/bench`, and `docs/bench` unless a reviewed integration requires an upstream change.

Use `uv sync --frozen --extra dev`. Run `CUDA_VISIBLE_DEVICES= uv run python -m pytest tests/bench -q`. Run Ruff on benchmark files. Build the package and check its installed entry points.

The authorized campaign uses GPU 0 only. On 2026-10-08, the user said: "hi you can use gpu 0 for your experiments - please use it for as much time as you need". This removes the earlier cumulative time limit of 10,800 seconds. Preserve all earlier charges and record this authorization. The coordinator can allocate additional finite admission credit in the existing ledger for each bounded job without asking again. Its numeric `limit_seconds` is allocated admission credit under this uncapped authorization, not a new user time cap. Keep per-job watchdogs, exclusive GPU 0 leases and occupancy checks. Run GPU jobs through `python -m abc_bench.runner`. The update worker must hold a coordinator lease. Do not start simultaneous jobs, reset charges, touch other GPUs or disturb Leela's jobs.

Show only executed measurements. Separate smoke tests, method adaptations, benchmarks, and embodiment preflight. Keep missing measurements unavailable. Native YAM results are not R1 results. Do not label candidate selection alone as Real-Time EXPO-FT.

## Current method correction — 13 October 2026

The user requires the exact author implementation and explicitly selected
**Real-Time EXPO-FT**, not standard EXPO-FT. This supersedes the earlier standard
EXPO serial campaign and pending FastTD3 ABC resource/training plans below.
Keep their code and evidence, but do not resume those campaigns automatically.

Use the author's real-time OpenPI branch, `RTCLearner` for prefix-conditioned
SFT, `RealTimeEXPOFTLearner`, and the original training/evaluation drivers with
`AsyncChunkSampler`. The ordinary SFT125 checkpoint is not RTC-SFT initialization.
Do not replace the author loop, replay schedule, loss, filter, or stopping policy
with locally written equivalents. Keep task/transport changes explicit and
outside the learner. GPU0-only authorization still applies; do not execute
author shell scripts that expose GPUs 1–3 unchanged.

First establish FastTD3 on an author-supported task with its task-specific
settings and original entrypoint. A scratch state policy and a pretrained VLA
are different training strategies. Their ABC scores do not isolate algorithm
quality. Preserve actual cancellation and incomplete-run evidence.
For pristine upstream code, the author's direct script entrypoint is permitted.
Do not rewrite imports or training programs just to fit our packaging convention.

Read the current section of `docs/bench/RECIPE_FIDELITY_AUDIT.md` before launch.
Passing adapter tests or matching update function hashes is not recipe fidelity.

Keep downloaded assets, checkpoints, videos, and raw outputs outside Git. Commit the dependency lock and compact reproducibility evidence. Publish architecture diagrams and team explanations when goals finish. Follow ASD-STE100 guidance and state any missing compliance check.

Captured host schema2 is developed separately from the historical clipping host.
Read `docs/bench/captured-reset-v2/CONTRACT.md`. It restores one declared VAL MJB
and full INTEGRATION state and gates actions on exact physics plus live RGB.
Use raw original float32 commands only under the explicit schema2 action profile.
The public CapturedScene helper alone does not admit actions. Root owns native
camera acceptance and frozen-policy jobs; CPU camera doubles are not native proof.

FastTD3 ABC vector qualification uses `abc_bench.fasttd3_vector_stage` through
the canonical runner. Read `docs/bench/FASTTD3_VECTOR_STAGE.md`. Schema1 admits
only128 worlds/64vector calls with a pinned native CPU admission receipt.
Keep learner B32768/replay51200 and original update functions unchanged. The
finite qualification is not convergence or a frozen task-success result.

The separate `abc_bench.fasttd3_scratch_stage` admits only4000 calls after a
completed canonical native128 qualification. Read
`docs/bench/FASTTD3_SCRATCH_STAGE.md`. The original64-stage source stays unchanged.
Root owns the normal release and native GPU0 job. External reviews are not a
prerequisite under the user's current results-first instruction.

The separate `abc_bench.fasttd3_frozen_stage` authenticates a completed4000
training producer before captured VAL9100/9101/9102 evaluation. Read
`docs/bench/FASTTD3_FROZEN_STAGE.md`. Keep both children in the canonical
process group. Admit only full native episodes, video and unchanged
actor/normalizer/RNG evidence. Root owns serial1800s native jobs.

## Verified original RTC path — 13 October 2026

Read `docs/bench/REALTIME_NATIVE_RESULTS_20261013.md` for the two accepted native
checks. Use `abc_bench.author_program` only as process supervision. The author
owns training and control loops. Task/model dispatch is the checked four-file
patch in `docs/bench/REALTIME_AUTHOR_TASK_DISPATCH.patch`.

The one-update RTC checkpoint is a smoke artifact, not a trained benchmark actor.
Read `docs/bench/RTC_ACTION_DIAGNOSIS_20261013.md` before further RTC work.
The full trace has abrupt joint-target changes at chunk boundaries. Source and
CPU checks found no mapping defect, but actual model prefix tensors were not
recorded. Do not admit checkpoint 1 as online-RL initialization. Establish task
readiness with the original RTC-SFT recipe and validation behavior first. The
RTC-only clip contains no EXPO editor or critics and cannot assess their method.
The 32-control TRAIN diagnostic misses 34 ms: median dispatch 91.7 ms. Do not present
its simulation-clock video as wall-clock real-time execution. Original evaluation
needs `only_base_actions=False` for RTC prefixes, two resets, and one extra loop
slot to observe terminal state. Keep the native client dependencies in its own
normally installed prefix; ABC's optional `realtime-expo` extra declares the task
ConfigDict dependency. The original client wheel supplies its own dependencies.

The later full visual rollout uses native horizon 3540 and author bound 3541.
Read `docs/bench/RTC_VISUAL_ROLLOUT_20261013.md`. Its video is complete, but the
job remains failed because client process cleanup required SIGTERM. Native close
and complete video evidence do not establish graceful process exit. Keep these
claims separate. Resolve shutdown before another campaign. The client package
extra pins the observed native `ml-collections==1.1.0`; server dependencies stay separate.

## Original RTC task training — 14 October 2026

R1 Lite uses Sadhana's separately installed task package and the original RTC
entrypoint. Measured batch64 training takes about7.2seconds/update on GPU0.
Only schema1 `train_offline_rtc.py` may use up to36000worker seconds and
36100parent seconds. Both the job and effective CLI timeout must preserve
100seconds for cleanup. Online/evaluation and Catcher limits remain unchanged.
The original4000-step recipe is not split through approximate resume.

Schema1 author receipts honor explicit nonempty task/embodiment fields.
Canonical receipts are in `outputs/bench/runs/<id>.json`; worker receipts stay
in `outputs/bench/<id>/receipt.json`. Their recorded hashes must match.

The R1 qualification completed10updates with verified finite checkpoint,
optimizer counters and learner RNG. This does not admit task success.
Read Sadhana's `packages/r1lite-rl/docs/rl/R1LITE_REALTIME_EXPO.md` in the
`RL/sadhana-r1lite-rl-hillclimb` worktree for task adaptations and current runs.
