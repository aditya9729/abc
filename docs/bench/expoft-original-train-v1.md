# Ordinary EXPO TRAIN host schema3

The installed `abc_bench.expoft_host` supports an additional TRAIN-only schema3.
Schema1 clipping/development behavior and schema2 captured VAL remain unchanged.
The RPC envelope stays `sadhana.expoft.abc.v1` and keeps describe/reset/step/
observe_rgb/close. The coordinator owns the lease, process group and watchdog.

Require one original CPU MuJoCo world, MJWarp three168x224 RGB views, horizon3540,
`original_five_variant_scale_armcarry_v1` reset profile and
`original_float32_no_host_clip_v1` action profile. Specify every requested seed.
Both9100..9102/109100..109102 and development20260511 are reserved. A seed whose
mapped fallback reaches a reserved seed is also rejected.

Normal reset retains the carried arm. On original RandomizationSamplingError,
clear the carried arm and retry the same seed. On a second placement error,
try requested+100000 once. Any other native error stops immediately. This
sequence matches original `abc_minimal.eval_policy.run_sequential`.

Each successful reset reports requested and actual seed, attempts, full
RandomizationState, initial MJB and float64 INTEGRATION/qpos/qvel file pins,
original policy state and live camera hashes. The EXPO client authenticates
those pins and observes actual initial RGB before commands. This is a recorded
TRAIN scene, not equality to a predefined captured validation scene.

Step forwards finite physical14 float32 proposals unchanged. Gripper proposals
can exceed nominal[0,1]. Record nominal bound violations separately from empty
host clipping records. Native actuator constraints remain enabled. Preserve
the terminal world until explicit reset. No automatic reset occurs.

CPU unit tests use explicit reset doubles for conditional failure paths. One
test uses the real installed original MuJoCo factory with cameras disabled.
It checks native reset/arm carry and full initial-state recording. Live cameras,
policy sampling, GPU capacity and task performance need root native proof.
