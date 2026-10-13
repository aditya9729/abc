# Reuse the released Real-Time EXPO-FT learner

The user pointed to the actual algorithm file. Root re-read the README and
traced the learner, driver, sampler, backend, replay, and client boundaries.
The current remote README and learner match the local source byte for byte.
The EXPO-FT revision is `21fe3d3b7d913c80836496817965932f49e0aedf`.

The original method is available. The earlier availability report overstated
the consequence of a missing packaged Kinetix experiment. It is an integration
gap. It is not a reason to rewrite the critic, editor, or filtering algorithm,
or to ask the user to approve a replacement algorithm.

## What is already implemented

Source: [original learner](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/expo_ft/agents/alg/realtime_expo_ft.py).

| Component | Original implementation |
| --- | --- |
| Delayed base candidates and action prefixes | `sample_pre_cache`, line 665 |
| Edits and Q selection using the current observation | `sample_actions`, line 698; `_jitted_fast_select`, line 185 |
| Noise-seed filtering during critic backups | `sample_batch_actions`, line 780 |
| Edit-policy and temperature training | `update_edit_actor`, line 946; `update_temperature`, line 1021 |
| Base-policy training | `update_actor`, line 986 |
| Action-critic and noise-critic training | `update_critic`, line 1036 |
| Combined update sequence and inference-cache refresh | `update`, line 1157; `_update_jit`, line 1168 |

`train_pi_robo.py` imports this learner at line 148, constructs it at line 202,
uses `AsyncChunkSampler` at line 262, and calls its update at line 294.
`realtime_expo_ft_pi_config.py` selects this class. The README documents
RTC-SFT, online RL, and evaluation under “Running Real-Time EXPO-FT.”
These are actual training components, not placeholders.

The README's RTC-SFT initialization sequence describes its DROID robot setup.
The [simulation paper](https://arxiv.org/html/2609.18207v1#S7.SS3) instead starts
from the released per-level BC flow policy.
Do not impose the robot's RTC-SFT prerequisite on that simulation experiment.
The synchronous training driver still overlaps slow inference with actions.
The separate asynchronous updater needs multiple GPUs; it is not required to
use the real-time method on the authorized single GPU.

The [README](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/README.md)
also describes adding task environments and policy backends. `create` accepts
a prebuilt actor. The client constructs `task_config.env`. Those are useful
integration boundaries; they do not establish that every surrounding component
already supports symbolic Kinetix state.

## Required Kinetix integration

| Boundary | Concrete released assumption | Required work |
| --- | --- | --- |
| Policy factory | Driver directly calls `build_pi05` | Wrap the original Kinetix flow policy using the actor interface |
| Critic input | `create` builds a ResNet camera encoder | Supply the paper's symbolic state encoder through an explicit factory hook |
| Replay and batches | Camera fields, pi0.5 transforms, shared padded state/action dimensions | Preserve state and action dimensions, temporal alignment, and chunk rewards in a state-data adapter |
| Augmentation | Both available modes modify images | Supply state-appropriate preprocessing; disabling full augmentation alone still crops |
| Base updates | Released online prefix length is zero at boot, otherwise deployment delay | Account explicitly for the simulation paper's independently randomized training prefix |
| Base training data | Released config selects successful episodes | Use the existing `actor_success_only=False` option for the simulation setting |
| Environment | DROID dataset dispatch and task settings | Add Kinetix reset, observations, rewards, termination, and rendering |
| Action transport | The client skips an all-minus-one action as a sentinel | Make that behavior task-specific; this is a valid Kinetix control vector |

The original flow policy uses JAX/Flax NNX. A different ML framework is not
the main obstacle. Its interface still differs: the backup filter must pass
its selected noise into denoising, and the policy wrapper must honor it.
The original `FlowPolicy.action` and `realtime_action` generate noise internally.
They need an explicit input for the selected noise. Otherwise, a wrapper could
silently discard the filter's decision. Verify selected-noise-to-action identity.
Matching method names alone would not verify that contract. The original
Kinetix prefix-loss schedule also differs from the simulation paper's uniform
prefix lengths from zero through four. Record this recipe change explicitly.

One algorithm detail also differs between paper and release. The released
noise-filter target uses `qs[:, 0]`, the decoded base candidate's value
(`realtime_expo_ft.py:939`). Paper Equation 11 describes the value of the
selected edited-or-base chunk. For an original-implementation comparison,
retain the released target. Do not silently replace it with the paper equation.
Likewise, report the prefix-training difference before claiming paper fidelity.

Preserve original critic, edit, filter, and target-update calculations wherever
the data contract permits. Record any required source changes separately.
Do not claim an unchanged full Kinetix implementation before that integration
is tested. Replacing vector state with artificial camera images would change
the paper's experiment and is not the proposed solution.

The original training loop rechecks inference launch after refilling its plan
(`train_pi_robo.py:346-348`). The standalone evaluator checks only before refill
and then pops an action (`eval_droid_policy.py:324,389-394`). When delay equals
the execution horizon, that loop misses the full-plan launch condition. Source
inspection shows it falls back to synchronous inference at later boundaries.
Use the original sampler's correct scheduling and test this 4/4 case explicitly.
This is a source finding; no new runtime test was performed.
This does not explain the earlier ABC run, whose settings were delay 5/horizon 8.

## Evidence and scope

This was source review. No simulator or GPU was run. Existing one-update RTC
footage still contains no online EXPO editor or critic training. It remains
an integration diagnostic, not a result for this learner.

Evidence is under
`/home/user/aditya/RL/builds/realtime-expo-reuse-check-20261013/`.
`CURRENT_SOURCE_CHECK.json` records the two remote/local byte comparisons.
The independent learner and boundary reviews document exact dependencies.

Root owns the integration and claims. The learner reviewer checks update and
sampling code. The boundary reviewer checks policy, replay, and driver APIs.
[Corrected integration diagram](REALTIME_LEARNER_REUSE_20261013.mmd).
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.

## Later CPU readiness check: original Catcher policy

The original Catcher BC epoch-24 policy is now downloaded and authenticated.
Its 12,290,914 bytes have SHA256
`926d2d3821fb31be8cfa24532711efe52ba10d62786e332a48c5a20431b43c00`.
The original flow-policy implementation restored all 42 parameter leaves exactly.
All 3,072,006 parameters are finite. Original task and velocity shape tracing
confirmed 679 observation dimensions, six action dimensions and horizon eight.
No environment step, sampled episode or optimizer update ran in this check.

The original RTC source is pinned at `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`.
Its Kinetix submodule is pinned at `cf7453ea103fa0b77348af1a39f689c658161613`.
Both checkouts remain clean. The separate CPU prefix is
`/home/user/aditya/runtimes/rtc-kinetix-9296f31-catcher-py311`.
Its 103 installed versions match the original lock. CUDA packages were omitted
explicitly; GPU runtime readiness remains pending. The original training and
evaluation entrypoints both passed their help/import checks.

Pristine Kinetix packaging omitted 14 PNG textures required at import time.
A separate packaging copy adds only their package-data declaration.
The normal installed wheel preserves all 31 original Python and 14 PNG payloads.
No author learning code changed. The top-level project keeps its original
direct-script entrypoints. No editable install or Python path injection was used.

The measured 679-versus-six dimensions confirm that the robot batch helper's
shared state/action padding assumption needs an explicit state-task adapter.
The encoder, supplied-noise, prefix-loss, replay and task boundaries above remain
unfinished. This is initialization evidence, not a Real-Time EXPO benchmark.
The expert dataset remains undownloaded; this check only needed the small policy.

Receipts and the readiness diagram are under
`/home/user/aditya/RL/builds/realtime-expo-next-experiment-20261013/`.
Read `READINESS.md`, `CPU_ACTOR_RESULT.json`, `SOURCE_RUNTIME_AUDIT.json` and
`READINESS_SEAL.json`. Root owns the later native run and its acceptance.
The 21-member seal has SHA256
`1e9baa0c6f9fd96c6a2bdf45f66876435d465fb9ffc4bc7f5e3e5e44a6b85685`.

Later [input and runtime checks](REALTIME_CATCHER_BOUNDARIES_20261013.md)
verify an optional selected-noise/prefix hook candidate and 32 scripted Catcher
controls on CPU. They record the diagnostic JIT correction, PRNG default change,
and unresolved dependency metadata conflicts. Full learner integration and
native policy results remain pending.
