# Captured validation reset and original command profile

Host configuration schema 2 selects one declared validation episode from an
immutable `sadhana.abc.demonstration-cohort/v1` cohort. It uses the episode's
actual seed and the completed canonical collector worker/parent lineage.
It does not use a summary seed to resample a scene.

The public helper is `abc_bench.captured_reset.CapturedScene(profile)`.
`restore(env, report)` uses public `MjModel.from_binary_path`,
`env.reload_from_model`, `env.set_task`, `mj_setState(mjSTATE_INTEGRATION)` and
`mj_forward`. `check_physics(env)` returns exact model serialization, full
INTEGRATION, qpos, qvel, float64 time, original float32 state14, judge and
chronology comparisons. `recorded_camera_hashes()` returns comparison hashes;
it never returns recorded images as live observations. These helpers alone do
not admit actions. `CapturedExpoTask.reset` admits actions only after every
physics check and every actual initial MJWarp RGB hash passes. It repeats the
physics checks after rendering. Any failed reset keeps actions disabled.

The profile supports the authenticated current ABC/YAM captured model only:
MuJoCo 3.8.1, float64 MJB ABI `(54321,8,92,3008001,472)`, nq58/nv52/nu14,
INTEGRATION specification 16383 with 531 float64 values, dt0.002/decimation17,
no plugins, sleep, callbacks or Python task runtime state. Runtime package
versions must match the completed collector. Original source pins are checked.
The collector checkpoint identity remains sealed by the collector; the host
has no policy weights and does not reread them.

After restoration, the host never resets, randomizes or forgets the arm state.
A new explicit reset reloads the same captured model and full state. No implicit
resampling or autoreset occurs. `reset-proof-NNNN.json` records completed or
failed checks. The RPC schema/operation keys remain `sadhana.expoft.abc.v1`.

The action profile is `original_float32_no_host_clip_v1`. It forwards every
finite 14D command after the original float32 cast, including gripper commands
outside nominal `[0,1]`. Native actuator limits remain active. Returned
`action_applied14` identifies the submitted float32 command, not measured
physical motion or a hypothetical unclamped actuator force. Host clipping
records are empty; nominal bound violations are counted separately. Legacy
configuration schema 1 retains its explicit clip-then-float32 behavior.

The new paired source and installed CPU tests use a small real MuJoCo model,
explicit task/camera doubles, bounded file authentication and actual host/client
dispatch guards. They do not prove live MJWarp equality or policy performance.
The separate root CPU qualifier already restored all three actual VAL MJB/state
captures twice and matched all 96 first-prefix physics rows exactly. Native
initial RGB equality, no-action camera qualification and policy rollouts remain
coordinator acceptance steps. Initial equality does not prove future trajectory
identity or complete mjData memory equality.

Use only a canonical coordinator lease to launch this host. Root owns release
installation, live cameras, GPU0 policy jobs and result publication.
