# Catcher input and runtime checks

The released Real-Time EXPO learner is the integration target.
This check connects neither a complete task driver nor a training campaign.
The original Mulligan training continues on GPU0 in a separate runtime.
These checks use CPU only. They establish input behavior and runtime compatibility
on specific fixtures. They establish no policy success or learning gain.

## Actor inputs

The original Kinetix flow policy draws noise inside its sampling methods.
EXPO must instead decode the noise selected by its filter.
The [candidate patch](REALTIME_CATCHER_ACTOR_INPUTS.patch) adds optional noise
arguments to `action` and `realtime_action`. It also accepts explicit per-example
prefix lengths in the original flow loss. Defaults preserve the author behavior.
The patch adds 51 lines and removes nine in `src/model.py`.
It does not change the network, parameter schema, solver steps, or RL learner.

The candidate passed 12 CPU checks in each of two runtimes.
Tests check original-versus-candidate defaults, complete gradients, the authentic
BC24 checkpoint, supplied-noise use, clean prefixes, loss masks, shapes, and JIT.
Most gradient tests use small declared numerical fixtures.
They are not task trajectories or installed end-to-end integration tests.
Root verified all ten files in the actor's sealed evidence bundle.

Two inherited choices need explicit experiment settings:

- The original prefix loss divides by unmasked tokens, not action coordinates.
  An explicit zero-length prefix therefore scales the vanilla loss by six for Catcher.
  The patch preserves that reduction. Valid prefix lengths are zero through eight.
  An all-eight batch has zero loss and gradients. Invalid values produce NaN;
  the backend must validate its narrower deployment domain before updates.
- The released learner supplies prefix zero at episode boot, otherwise its configured delay.
  The planned Catcher configuration sets that delay to four.
  The simulation appendix specifies independent uniform prefixes from zero through four.
  This patch does not silently change that schedule or the original inference mode.

Supplying loss prefixes does not switch `simulated_delay` or inference guidance.
The backend must state when it uses original guidance or hard prefix clamping.
The complete backend, symbolic replay, driver, and native acceptance remain pending.

## Runtime findings

The original CPU readiness prefix remains unchanged:
`/home/user/aditya/runtimes/rtc-kinetix-9296f31-catcher-py311`.
It uses JAX 0.4.35 and jaxlib 0.4.34.
The separate integration prefix uses the installed EXPO server's JAX/jaxlib 0.5.3:
`/home/user/aditya/runtimes/realtime-expo-catcher-integration-py311`.
This prefix combines the EXPO server dependencies with missing Kinetix dependencies.
It is an integration runtime, not the unchanged RTC lock environment.
The original server and live training prefixes were not modified.

The combined prefix has 239 packages. Its observed freeze is saved with the evidence.
`uv pip check` still fails with two conflicts:
Kinetix declares JAX 0.4.35; jaxued declares Orbax Checkpoint 0.5.3.
The installed versions are 0.5.3 and 0.11.13 respectively.
The only extra compatibility downgrade so far is decorator 5.3.1 to 4.4.2,
which satisfies the original MoviePy requirement. Metadata conflicts remain visible.
The CPU checks below do not establish a fully supported dependency resolution.

Identical original BC24 weights restore exactly in both model checks.
On equal synthetic inputs, the velocity outputs differ by at most 9.54e-7.
Same-key sampled actions initially differ by up to 0.753 because JAX changed
the default `jax_threefry_partitionable` setting from false to true.
Setting `JAX_THREEFRY_PARTITIONABLE=false` in the integration diagnostic reduces
the action difference to 1.03e-6. This is measured fixture agreement, not universal
equality or task-level equivalence. Record the PRNG setting in later run manifests.

Root's first physics diagnostic failed under JAX 0.5.3.
The local diagnostic nested a dynamic-parameter JIT around an author method
that declares those parameters static. The corrected diagnostic captures constant
`EnvParams`, as the original trainer does. No author physics code changed.
Both runtime checks then completed 32 scripted controls with the original
noise, logging, and replay-reset wrappers. All 5,544 recorded observation/state
arrays agree within absolute and relative tolerances of 1e-5; maximum absolute
error is 2.15e-6. Reward and done rows match exactly.
No terminal state occurred, so terminal handling and automatic reset remain untested.
These controls are scripted diagnostics, not episodes from the learned policy.

## Evidence and next acceptance

Evidence resides under
`/home/user/aditya/RL/builds/realtime-expo-catcher-integration-20261013/`.
The actor candidate is commit `5accbcd202e13352d2e46fb3d222f285753bd3df`.
Its seal has SHA256 `c82b28f7b4a87053f7aafc34bca23f791ce5e416d3226900a9771c524edd22e9`.
[Compact evidence pins](REALTIME_CATCHER_BOUNDARIES_20261013.json) identify the
diagnostic sources, outputs, and version differences.
The actor tests compare each runtime against its own original default behavior.
The separate cross-runtime diagnostics explicitly align the PRNG setting.

Next, test the normally installed combined backend and state adapter.
Verify actual learner updates, original replay chronology, prefix use, termination,
and the delay-four/execution-four schedule before a native learning campaign.
Preserve the original critic, editor, and noise-filter update equations.
Record all task and recipe choices separately from the original implementation.

Root owns integration and native acceptance. The actor agent owns the input hooks
and backend candidate. The state agent owns the symbolic data boundary.
An independent reviewer checks runtime and actor evidence.
The review found no blocking defect in these checks. It reproduced the reported
array comparisons and confirmed unchanged installed Kinetix source files.
It did not accept the complete integration or the conflicting dependency pins.
[Architecture](REALTIME_CATCHER_BOUNDARIES_20261013.mmd).
The prose uses STE guidance; full ASD-STE100 compliance was not checked.
