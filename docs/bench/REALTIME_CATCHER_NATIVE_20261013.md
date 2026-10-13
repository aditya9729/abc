# Catcher: original policy videos and authentic-data learner check

The author BC24 policy completes the first episode in each of three native
Catcher trials. The [viewer](http://127.0.0.1:8770/original-methods/#catcher-heading)
shows all three recordings. A separate check feeds authentic expert data through
the original Real-Time EXPO learner. Native online learning remains pending.
These checks establish a working starting point, not an RL improvement.

## Original frozen evaluation

| Profile | Simulated inference delay | First-episode outcome | Controls |
| --- | --- | --- | --- |
| BC | 0 | Success | 67 |
| Naive delayed BC | 4 | Success | 76 |
| RTC guidance | 4 | Success | 68 |

Each row uses one trial, seed 100000, the same BC24 checkpoint, action horizon
8, execution horizon 4, ten flow steps, and original action noise 0.1.
This sample cannot establish a success rate or latency advantage.

The installed `eval_flow.py`, `model.py`, and `train_expert.py` are byte-exact
copies of RTC revision `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`.
Kinetix remains at `cf7453ea103fa0b77348af1a39f689c658161613`.
The selector calls the original public `eval()` function for three profiles.
The original function owns physics, control, rewards, and first-episode scoring.
The original full command sweeps 96 profiles and discards its returned video;
the selector is an explicit entrypoint and output adaptation.

All three videos contain 256 original pre-action frames at native 128 × 128
resolution. Automatic resets can occur after termination. Thus, each recording
contains more than the scored first episode. Playback uses the author's 15 fps
serialization: 17.07 seconds for 8.53 simulation seconds. It runs at half the
simulation rate and provides no wall-clock real-time measurement.

The CPU runtime matches 103 original locked package versions. The process exited
zero in 107.98 seconds, including compilation, evaluation, and serialization.
The checkpoint checksum stayed unchanged. Root verified source and video hashes,
decoded every frame, and inspected representative native frames. Browser checks
passed for playback, end seeking, refresh, and desktop/mobile layout. Existing
Mulligan scores and earlier recordings remain visible.

An earlier serialization attempt failed because our check expected 512 × 512
frames. The original renderer applies downscale 4. We corrected the check and
preserved the failed attempt. No author renderer or policy changed.

## Authentic expert data reaches the released learner

The original Catcher archive contains 1,015,808 records. Complete episodes cover
1,011,008 records in 14,955 episodes; 4,800 trailing records lack completion.
The loader excludes those trailing records explicitly. It preserves stored
success labels independently of returns. The archive has no per-step rewards
or terminal post-action states. Sparse reward reconstruction and absorbing
terminal masks therefore remain declared data assumptions.

The installed CPU check used the first 128 complete episodes: 8,638 records,
126 successful episodes, and two episodes with positive return but false success.
The original `BatchProcessor` retained state, actions, rewards, masks, terminal
flags, intervention flags, and success labels. No admitted demonstration was
evicted. Terminal action chunks preserved their issued actions.

The original `RealTimeEXPOFTLearner` performed two updates with batch size 64
and update-to-data ratio 20. Its final counters were:

- Critic and noise-filter critic: 40 each.
- Action editor, temperature, and base policy: 2 each.

Metrics stayed finite. The first base-policy update used the original zero
warmup learning rate. The second update changed genuine BC24 parameters.
The check uses edit scale 0.1; the prepared task configuration uses the released
robot default 0.2. This difference prevents treating the check as that task's
exact recipe. It executes zero native environment controls and reports no
learned-policy score.

## Integration boundaries and remaining work

The state-policy backend, state replay hooks, and Catcher task adapter connect
the original components. They do not replace the learner. The merged execution
tree contains both state and task changes, so direct author scripts cannot
silently import an incomplete source tree. The normal wheel and installed
payloads match: 38 EXPO Python files and eight task Python files.

Independent source review closed four task-boundary defects: nonfinite actions
were silently replaced, initial connection retries were unbounded, demonstrations
could exceed replay capacity, and disconnects could wait for reconnection.
The task configuration now rejects invalid actions, bounds startup, sizes replay
for admitted data, and fails on disconnect. DROID defaults remain original.
The original complete training loop and shutdown suffix remain byte-exact.

A native boundary test then completed 70 scripted controls through a failure
termination. All 81 native state leaves, observation, reward, and success matched
the original same-key step at tolerance 1e-5. An after-terminal no-op preserved
state and RNG. Rendering, reset, invalid-action rejection, and close also passed.
Four separate client/transport fixture checks passed. These fixtures do not
establish native process cleanup. Two earlier test assumptions about leaf count
and render size were corrected; their failures remain recorded. Production
task code did not change during those tests.

The harness supervisor now permits the task settings path and explicit legacy
JAX PRNG mode. Its child-environment check passes, and `PYTHONPATH` remains
rejected. Thirteen installed supervisor/export checks passed. Ruff and text
diff checks passed. The earlier full benchmark suite remains failed at
775 passed, 24 failed, and two skipped; it was not rerun for this focused change.

The next admission is a bounded native CPU run through `train_pi_robo.py`, with
the original ten-episode warmup and original update functions. That run must
show complete episodes, actual updates, a saved checkpoint, and graceful client
cleanup. It is a short integration schedule, not the full paper experiment.

Two combined-runtime dependency metadata conflicts remain. Paper/code differences
also remain: prefix training, noise-filter targets, and some simulation settings.
The task adapter uses an explicit reset RNG policy; it does not claim complete
trajectory RNG equivalence across original automatic resets. Native controls,
driver dispatches, and after-terminal no-ops must have separate counts.

GPU0 remains exclusively allocated to original Mulligan training. Its last
viewer snapshot records 94,000/150,000 updates. Final local policy performance
is pending. The existing 34/50 versus 46/50 comparison uses released checkpoints.
SGS still has no released author implementation at this check.

## Artifacts and ownership

Paths below are relative to `/home/user/aditya/RL/builds/`:

- `rtc-catcher-frozen-eval-20261013/native_retry1-pilot/RESULT.json`
- `rtc-catcher-frozen-eval-20261013/ROOT_NATIVE_ACCEPTANCE.json`
- `rtc-catcher-frozen-eval-20261013/VIEWER_BROWSER_RESULT.json`
- `realtime-expo-catcher-integration-20261013/combined/ACTUAL_DATA_LEARNER_RESULT.json`
- `realtime-expo-catcher-integration-20261013/data/CALLBACK_HANDOFF.md`
- `realtime-expo-catcher-integration-20261013/EXECUTION_INSTALLED_AUDIT.json`
- `realtime-expo-catcher-integration-20261013/task/NATIVE_BOUNDARY_RESULT.json`
- `realtime-expo-catcher-integration-20261013/task-review/CORRECTED_SOURCE_REVIEW.json`

Root owns execution, acceptance, and publication. One agent prepares the original
evaluation and actor backend. One prepares the task boundary. An independent
agent reviews method and transport claims. External review remains waived under
the user's results-first instruction; no external review pass is claimed.

[Evidence pins](REALTIME_CATCHER_NATIVE_20261013.json),
[architecture](REALTIME_CATCHER_NATIVE_20261013.mmd), and
[scaling plan](DEPLOYMENT_RL_SCALING_PLAN.md). This explanation uses STE guidance.
Full ASD-STE100 compliance was not checked. The original-recipe goal remains active.
