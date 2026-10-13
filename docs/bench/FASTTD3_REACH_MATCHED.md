# Matched TRAIN reach diagnostic

This stage compares the true initial scratch actor with the final trained actor.
It uses three admitted TRAIN captures, not a held-out score. It keeps the full
five-bottle benchmark and the VAL capture gates separate.

The optional original train boundary exports the actual actor and normalizer
before the first action, transition or update. The original checkpoint writer
and all original update bodies remain in use. The producer must complete a new
reach runtime gate and a bounded reach training job before matched job
preparation can succeed. A historical completed producer without the control-zero
export is rejected. A runtime gate alone is not a training producer.

The camera-free host uses the normal ABC factory, then restores each pinned MJB
and full `mjSTATE_INTEGRATION` state. It verifies serialized model bytes, all 531
state values, qpos, qvel, state14, time, native judge and the shared action/state
schema. It measures the initial distance before an action, checks the geometry
against independent owned FK, and verifies that geometry does not change native
INTEGRATION. Every control checks finite physics, the 0.034-second native step
and zero counts for all seven warnings. No sampler reset or camera is used.

Each pair uses state216, 14 normalized actions and the existing float32 affine
mapping. Both actors run exactly 128 controls. Dense reward is `exp(-d/0.15)`,
where `d` is the shortest world-space distance from two grasp sites to five
bottle centers. Reach requires `d <= 0.12 m` for five consecutive controls.
Qualified reach also requires reset `d > 0.15 m`. Already-near cases stay in the
report but stay out of that denominator. These rules came from admitted TRAIN
geometry and are diagnostic adaptations. Native current five-of-five success,
any-control native success and peak bottle count are separate fields.

The public frozen Fast consumer uses the exported actor and normalizer in eval
mode, no exploration/noise, no normalizer updates and no optimizer calls. It
checks separate actor and normalizer hashes before and after each evaluation.
Both actors must have identical reset proofs, precision, horizon and reward.
The initial hash must match the actual pre-loop export. The comparison measures
the policy and its own frozen normalization together. It does not isolate an
algorithm effect or prove generalization.

The canonical supervisor owns one sequential native host and one consumer in
the existing coordinator process group. It reuses the public lease, CPU
preflight, source checks, child environment, cleanup and memory admission. The
memory helper admits only 64/128 worlds. This stage conservatively uses its
unchanged 64-world bound for one world and records this difference. It does not
widen the helper. Readiness is 180 seconds, consumer work is 850 seconds,
shutdown is 30 seconds and the canonical job is at most 1260 seconds. This new
allowlist entry retains the generic 7200-second cap outside this stage.

Prepare after an actual completed reach TRAIN producer exists:

```sh
python -I -B -m abc_bench.fasttd3_reach_matched_stage --prepare-job \
  --parent /absolute/path/to/completed-parent.json \
  --worker /absolute/path/to/completed-worker/receipt.json \
  --host-config /absolute/path/to/matched-host.json \
  --source-pins /absolute/path/to/MATCHED_SOURCE_PINS.json \
  --output /absolute/path/to/new-prepared-job
```

Root releases the wheels and runs the prepared job through the normal canonical
runner. No native or GPU run was part of preparation. Actual Torch export,
native restore/RPC and initial/final policy results remain pending. Installed
CPU fixtures and source/packaging checks do not prove learning or task success.

Root owns review, release, admission and actual results. The preparation agent
owns these sources and CPU checks. This text uses ASD-STE100 guidance. A full
compliance check is unavailable. See the [architecture](FASTTD3_REACH_MATCHED.mmd).
