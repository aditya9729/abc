The vector boundary runs each ABC world in its own CPU process. Each process
owns a separate original MuJoCo model and data object. The learner uses the
original FastTD3 package. No network, replay, normalization, reward or update
equation was changed.

The training profile has 128 worlds. It uses a batch of 32768, a replay capacity
of 51200 per world, two critic updates and one actor update per pair of critic
updates. Compile and BF16 AMP keep their original defaults. One vector step
collects 128 transitions. A 64-step qualification should collect 8192
transitions and perform 106 critic and 53 actor updates. These counts are
predictions for ABC; the ABC learner has not run through this boundary yet.

Each world uses five bottles with the original random variants and scales.
Ordinary resets carry the arm pose. Placement failure follows the original
driver: retry the same seed from init_q, then try seed +100000. Every attempt
is recorded. Rank r and episode e request seed 200000+r+128e. The ordinary and
fallback banks are disjoint for e=0..780. The bank then fails closed. Development
20260511 and validation 9100..9102/109100..109102 are excluded.

Observations are float32[216]: original qpos, qvel, five active flags, five
19-way variant indicators and six scales. They are privileged state, not
camera observations. Actions are normalized[-1,1] YAM14. The adapter clips
normalized commands and maps them to original joint ranges and grip fractions.
The original environment maps grip fractions to metres. This is scratch RL,
not a VLA residual method. The source-policy ABC path uses its original raw
physical commands and does not use this normalized adapter.

The original physics timestep is .002 seconds with 17 physics steps per
control. Each worker steps once before the parent receives the barrier. On
success or timeout, the final observation is copied before that world resets.
The learner receives the new reset observation plus the raw final observation.
Original FastTD3 uses the raw final observation for replay and bootstraps a
timeout. It does not bootstrap a task success. No reward shaping was added.

CPU acceptance used two actual worlds. A normal step ended neither world.
Explicit native-state fixtures then tested the original success judge and the
original time-limit flag. Those fixtures do not show policy performance or a
full3540-step episode. Both owned children exited0. All12 arm joints were exact
across ordinary reset. One grip fraction changed by one float32 ULP in the
unchanged fraction/metre round-trip. The receipt records that error and the
small grip tolerance. No Torch or Warp module was imported in either worker.

The server checks the canonical lease with the inherited GPU UUID. It then
hides CUDA in its own process before native imports. The learner keeps the
coordinator's GPU visibility. The server uses bounded Unix JSONL frames and
finite worker, RPC and wall deadlines. Cleanup must account for every owned
rank/PID and reject missing, alive or nonzero-exit children. Normal cleanup
failure stays failed on repeated close. Forced shutdown cannot become success.

The public client is `sadhana_fasttd3_abc.vector.VectorRemoteEnv`. Its CLI calls
the unchanged `learner.train`. The public host is
`abc_bench.fasttd3_vector_host`. The CPU command is
`abc_bench.fasttd3_vector_cpu`. Use the normal installed candidates, not a
source-path edit. The exact config and artifact pins are in
`builds/abc-fasttd3-vector-v1/RELEASE_AUDIT.json`.

Root owns the pending canonical supervisor, lease allowlist, normal released
install and GPU execution. Root must measure fresh128-world RAM and GPU
capacity before training. The original replay alone requires11822694400 bytes
for obs216/action14 at128x51200. This excludes networks, optimizer state,
minibatches and compiler memory. The largest measured CPU child is not a
measured128-world memory peak. Original randomizer model caches remain intact.

Captured validation is not admitted here. A future evaluation boundary must
use the published `abc_bench.captured_reset.CapturedScene` API after root
admits its release and native camera gate. This module does not import that
peer's worktree. No R1Lite or method-ranking result is available from these
CPU checks.

Root plans and owns native jobs. The FastTD3 owner builds the world boundary
and client. The EXPO owner owns the captured-scene API. This explanation uses
ASD-STE100 guidance. It has not had a formal compliance check.
