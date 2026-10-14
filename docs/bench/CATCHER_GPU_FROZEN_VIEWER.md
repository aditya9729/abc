# Catcher GPU checkpoint: optional frozen comparison viewer

This candidate adds `catcher.gpu_frozen_comparison` to the packaged static viewer.
It preserves `catcher.frozen_comparison`, which shows the earlier CPU-R2 experiment.
The new panel is hidden when its optional record is absent.
It reports no default score or learner-update count.
Root owns result acceptance and publication. No actual GPU comparison exists in this candidate.

## Accepted record contract

The record uses `schema="catcher.gpu-frozen-comparison/v1"` and
`status="accepted_matched_development"`. It needs exactly four ordered rows,
with seeds `100001`, `100002`, `100003`, and `100004`.
Each row has `rtc_bc24` and `saved_gpu4096` clips.
Root must reuse the previously accepted BC24 cases for these exact environment-seed labels.
Root must first accept every new frozen GPU case and the full checkpoint restore.
Do not publish missing or failed cases as failures in a score denominator.
Omit the optional comparison until all four pairs are accepted.

Each clip uses the existing frozen display shape:

| Field | Required value |
| --- | --- |
| `status` | `accepted_frozen_case` |
| `complete` | `true` |
| `solved` | Actual boolean outcome |
| `controls` | Actual integer, from 1 through 256 |
| `frames` | Actual decoded frame count, equal to `controls + 1` |
| `sha256` | Full lowercase MP4 content hash |
| `video` | First 16 hash characters followed by `.mp4`; local filename only |

The separate `restore` object comes directly from the accepted full restore receipt:

| Field | Source |
| --- | --- |
| `status` | `original_full_checkpoint_restore_verified` |
| `checkpoint_step` | Actual saved training index, exactly `4096` |
| `original_update_calls` | Actual saved actor step; never inferred from index 4096 |
| `counters.actor_train_state.step` | Actor step from that same receipt |
| `counters.critic.step` | Critic step from that same receipt |
| `counters.filter_critic.step` | Filter step from that same receipt |
| `receipt` | Actual restore receipt pin: `{path, bytes, sha256}` |

Displayed counts must be positive safe integers. Actor step equals update calls.
Critic and filter steps each equal 20 times those calls in this selected profile.
The full restore checker owns all seven group checks and checkpoint/source provenance.
The UI checks the displayed subset. It does not authenticate native evidence itself.
The prepared checker is under `builds/realtime-expo-catcher-gpu-frozen-plan-20261014/`.
Its `checkpoint_reference.restore_proof` supplies the accepted receipt pin.

## Behavior and limits

The panel shows the two accepted outcome counts and eight complete episode players.
It shows training index and actual actor, critic, and filter updates separately.
It keeps existing video elements on unchanged refreshes, including seek and playback state.
All labels use text nodes. Media paths must be local content-hash filenames.

These are four seeded development trials on one level. They are not held out.
The original server/template seed stays 42. Environment seeds are 100001 through 100004.
BC24 initializes from seed 42. The learned policy retains its genuine restored checkpoint RNG.
This whole-policy comparison does not impose common policy noise or distinct physical layouts.
This compares the starting base with the learned whole policy, including selection and editing.
It does not isolate a critic effect or reproduce the paper experiment.
The frozen episodes run on CPU. GPU names the learned checkpoint training provenance.
Playback uses the simulation clock. It does not establish wall-clock real-time control.

## Verification and ownership

Node tests use explicitly synthetic DOM objects, outcomes, counts, hashes, and paths.
They do not provide native video, actual browser, training, or frozen-policy evidence.
Installed export checks use a new neutral prefix. Existing result/video files must survive export.
Root owns the normal full-wheel release, accepted result mapping, and actual browser checks.
The candidate adds no framework, learner, sampler, server, or dependency.

See [the architecture](CATCHER_GPU_FROZEN_VIEWER.mmd).
This explanation uses ASD-STE100 guidance. It has not had a full compliance check.
