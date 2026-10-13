# ABC environment for the original Real-Time EXPO-FT client

This boundary is implemented in `abc_bench.realtime_expo_env.ABCYamEnv`.
It has CPU contract tests. Native operation is pending.
Root owns the native acceptance run and its GPU0 lease.
The module import does not construct a simulator.

The task config selects `env_type='abc_yam'` and this environment class.
The client must import the class lazily, as the author task configs do.
The learner-side config can set `env=None` when ABC is unavailable.
The original client imports no DROID module for this environment type.
Its factory already calls `task_config.env(**dict(task_config))`.

## Boundary

The class uses existing public `make_original_env` and `reset_world`.
The original ABC simulator, renderer, reset sampler, and task judge stay unchanged.
One process owns one native world.
The class requires the canonical lease before native construction and operations.
It does not launch a host, learner, controller, or physics thread.

Required constructor fields are `episode_seeds`, `example_action`, and `event_log_path`.
Use a finite, unique TRAIN seed sequence.
The existing TRAIN seed check excludes held-out families.
The event path needs an existing absolute parent directory.
The class refuses an existing event file.
Use `env_usage='train'` and `language_instruction='put plastic bottles in bin'`.
The default native horizon is3540 controls.
A shorter explicit horizon is a smoke-test task change, not a paper result.

State and commands use float32[14]: left joints1–6, left gripper, right joints1–6, right gripper.
Joints use radians. Grippers use original ABC policy coordinates.
The adapter adds no clipping, normalization, reordering, or Cartesian conversion.
Three actual RGB images use uint8 HWC[168,224,3].
Their fields are `top_image`, `left_wrist_image`, and `right_wrist_image`.
All camera timestamps must match current native simulation time.
The data agent owns the public model transforms and demonstration loader.

`reset()` preserves original five-bottle variant/scale randomization and arm carry.
The existing placement-error retry remains unchanged.
Each reset consumes one declared seed.
Seed exhaustion fails; the adapter does not cycle or invent seeds.
Include the next reset seed after the last planned episode completion.
The original training loop resets before it checks its next iteration bound.
Native randomization and retry records enter the event log.
Matched captured evaluation is unavailable through this initial TRAIN class.
Random TRAIN resets cannot prove a captured VAL result.

The native full-five-bottle judge supplies success.
The author interface receives sparse success reward1 and terminal mask0.
Partial native bottle reward remains diagnostic evidence.
Time-limit truncation also receives mask0, matching the original DROID environment.
This is an explicit task interface choice, not reward shaping or a changed learner.

## Control order and time

The original learner, `AsyncChunkSampler`, replay, and training loop are outside this module.
Keep delay5 and replan8 under the agreed initial recipe.
One nonterminal `step()` calls original `env.step(action, render_obs=False)` once.
That call advances17 native physics steps at0.002 seconds each.
Its simulation duration is0.034 seconds.
Set `control_hz=1/0.034`, approximately29.4118Hz.
Do not label this setting30Hz real-hardware performance.

Observations, rendering, waiting, and background inference do not advance physics.
Only native controls advance the world.
The author sampler launches pre-cache when five actions remain.
Those action calls can advance native physics during background pre-cache.
Root must measure that overlap from actual sampler events and native control intervals.
Event logging alone does not prove this overlap.
The author `sim_latency_ms` option sleeps in wall time; it does not inject physics steps.
Report actual dispatch rate and missed deadlines separately from simulation rate.

The original training driver sends `example_action` before its replay loop.
This adapter requires that command to equal native initial hold controls.
A zero vector is not an ABC hold.
The adapter validates the declared hold after construction and initial reset.
That boot command advances one genuine native control and0.034 seconds.
The original driver does not save it as a replay transition.
The first episode therefore has one fewer collected transition than native controls.

The original training loop sends one extra command after observing `done`.
The adapter acknowledges that command without advancing the terminal world.
The response includes `executed_action` and `physics_advanced=False`.
The event log marks `terminal_noop`; it is not a native executed control.
This explicit compatibility behavior preserves the original training loop.
The original evaluation loop checks `done` before dispatch and needs no extra command.

## Lifecycle and inherited client limits

The author client sanitizes nonfinite actions to zeros before this adapter sees them.
Absence of adapter errors therefore does not prove finite policy outputs.
Preserve and inspect the original client warnings.
The client reserves all-minus-one commands and skips the environment call.
That command cannot be counted as a native control.

The original client retries connections and can recreate environments after errors.
This adapter refuses a second live native world in the same process.
It does not make ambiguous action RPC retries idempotent.
Root must treat a lost action reply or unexpected reconnect as an incomplete run.
Do not infer a completed trajectory from a learner exit code alone.

The client has no environment-close operation or final cleanup.
Root must close the environment during supervised shutdown, before escalation.
`close()` closes the video writer, native renderer, and event log.
It is idempotent and permits a later clean environment construction.
The class registers `close()` with Python's `atexit`.
Normal exit and SIGINT unwinding can run that cleanup.
Direct SIGTERM, SIGKILL, or a native crash can bypass it.
Root must verify cleanup before accepting completed native evidence.
Successful native close appends a `closed` event.
With `video_dir`, videos use `abc-train-NNNN.mp4` at the native simulation rate.
The directory must exist and be absolute.
Existing videos are refused.
Terminal info closes the current video after the terminal observation is collected.
The supervisor must also close partial videos on cancellation.

The original ABC prefix currently lacks `openpi-client` and `ml-collections`.
The pristine client also needs NumPy, websockets, and tyro.
The original `openpi-client0.1.0` requires NumPy below2.0.
The current ABC prefix contains NumPy2.2.6.
Resolve declared dependencies in the selected normal client installation.
Root owns normal dependency installation into the selected client runtime.
No installation into an existing runtime occurred for this change.

## Source anchors and verification

Author EXPO commit: `21fe3d3b7d913c80836496817965932f49e0aedf`.
`client/run_client.py:128` supplies the environment factory.
Its lines173–179 sanitize actions; lines184–191 handle DROID and the reserved sentinel.
`train_pi_robo.py:310` sends the startup command.
Its lines327–366 observe, label, plan, pace, and dispatch.
Its line398 handles `done` after dispatch.
`expo_ft/utils/loop_utils.py:58` schedules original background pre-cache.
Its lines89–91 implement simulated wall-clock sleep.
`eval_droid_policy.py:399` checks `done` before its step.
`client/envs/droid_env.py:115` defines sparse reward and terminal mask.

CPU tests use an explicit world double, not simulation evidence.
They check the author call sequence, action order, startup refusal, terminal no-op,
native success/mask mapping, synchronized RGB, seed exhaustion, and lease refusal.
Run them against the normally installed wheel with GPUs hidden.
Native physics, rendered camera acceptance, async overlap, and policy quality remain pending.

See [the architecture source](REALTIME_EXPO_ENV.mmd).
This text uses STE guidance. Full ASD-STE100 compliance was not checked.
