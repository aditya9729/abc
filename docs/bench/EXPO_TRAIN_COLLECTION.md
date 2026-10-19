# Original base-policy collection for TRAIN

This adapter records complete autonomous TRAIN episodes for the original
EXPO replay format. It calls the existing Sadhana native environment and frame
writer. It does not change the author evaluator, sampler, learner, reward or
action commands. It adds no smoothing. The existing single-seed recovery
recorder remains unchanged.

The code is ready for runtime binding. It has not collected a new native
episode. Starts **21004–21007 are proposed, not reserved**. The coordinator must
reserve them before the descriptor can pass admission. Current EXPO evaluation
and the queued WBC pilot retain priority.

## Data contract

The client executes one unscored startup reset at 20999, followed by one to four
declared TRAIN starts. It saves a reset frame and one frame after every actual
control. An episode with T commands therefore has T+1 lossless frames. The
recorder keeps success, failure and timeout episodes. It rejects a reset during
an unfinished episode and rejects terminal no-ops as experience.

The exporter joins frame t with the command and labels in frame t+1:

| Field | Meaning |
| --- | --- |
| `observations.head_left` | Pre-action uint8 RGB, `[360,640,3]` |
| `observations.state` | Pre-action float32 measured state14 |
| `observations.prompt` | Existing task instruction |
| `actions` | Actual applied float64 physical14 command, radians/metres |
| `rewards` | Unchanged historical task reward |
| `dones` | Termination or time-limit truncation |
| `masks` | Zero on termination, one on time-limit truncation |
| `is_hil` | Always false for these autonomous episodes |
| `is_success` | Initially false; original replay restore reconstructs marks |

Proposed commands remain in capture metadata. They are not substituted for
clipped, actually executed commands. The original `BatchProcessor` calls
`PiReplayBuffer.restore_success_marks()` to mark closed successful episodes.
Do not send these episodes through the synthetic reference-demo loader.

Each episode uses the existing 250 MB frame-output cap. Four episodes plus the
cohort receipt are bounded by 1,002,000,000 bytes. The separate replay export uses
a declared cap, at most 5 GiB, with 4 MB reserved for its manifest. These limits do
not establish live disk admission. The canonical operator must also preserve
the current checkpoint reserve and all other running jobs.

## Binding and execution

Install the normal ABC wheel into new, unused client/export prefixes. Reuse the
qualified dependencies. Do not change an active prefix or use source-path
injection. The client needs the existing installed
`sadhana_r1lite.recovery_continuation` API. The export process needs the installed
original EXPO package. No simulator is imported by the export module.

1. Write a coordinator reservation with schema
   `sadhana.expo_train_seed_reservation/1`, owner `/root`, split `TRAIN`, the exact
   `seeds` list and `startup_seed: 20999`. The validator excludes DEV/FINAL,
   previously used low TRAIN seeds, duplicates, and the existing reserved online
   range 30000–280001. It accepts no fabricated default reservation.
2. Copy the accepted task settings to a new artifact. Change only the reset
   schedule, set `record_video=false`, and clear `event_log_path`. Retain the
   selected 21 demo IDs, cohort/statistics identities and native task configuration.
3. Write the collection descriptor using the fields below. Hash each input with
   `abc_bench.expo_train_collection.pin()`. The installed module paths must match
   the exact pinned files. Use a new capture directory under `~/aditya`.
4. Create a two-line client config shim whose `get_config()` returns
   `make_collection_task_config("/absolute/bound/DESCRIPTOR.json")` from
   `abc_bench.expo_train_collection`. The original server keeps its existing
   Sadhana task/data config; both server and client use the same new task settings.
5. Reuse the original canonical `author_program` evaluator configuration. Keep
   RTC2000 initialization, seed 42, H16/replan8/delay5, N32, assets and normalization.
   Set `only_base_actions=True`, the declared episode count, `save_video=False`,
   and fresh output/checkpoint directories. Bind the new client interpreter and
   client shim. Call `validate_evaluator_arguments(arguments, episodes=N)` on the
   actual arguments, then authenticate the full source/runtime/task closure with
   the existing canonical binder. No live job is supplied in this preparation.

The descriptor has exactly these fields:

| Field | Value |
| --- | --- |
| `schema` | `sadhana.expo_train_collection/1` |
| `split `, `mode` | `TRAIN`, `original_base_evaluator` |
| `seeds`, `startup_seed` | Reserved list of 1–4 starts, 20999 |
| `reservation` | Pin of the coordinator reservation |
| `task_settings`, `native_config` | Pins of the task JSON and native environment JSON |
| `recording_modules` | Module-name→pin map for the five names in `MODULES` |
| `policy_reference` | Pin of the accepted base evaluator config |
| `output_dir` | New absolute capture directory |
| `lineage` | `source_episode:12`, `scene_reference:11`, `new_independent_source_lineage:false` |

The recorder cannot observe remote server weights or arguments. Its result
states that limitation and remains `artifact_accepted=false`. The coordinator
must verify the actual canonical run and its immutable inputs. The argument
check does not replace weight, task, dependency or process authentication.

After a complete collection, create a pin JSON for its `RESULT.json` and a pin JSON
for the qualified original `expo_ft/data/replay_buffer.py`. Run the exporter from
the new normal installed prefix:

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu <export-python> -B -m abc_bench.expo_train_export \
  --descriptor /absolute/bound/DESCRIPTOR.json \
  --result-pin /absolute/bound/COLLECTION_RESULT_PIN.json \
  --serializer-pin /absolute/bound/ORIGINAL_SERIALIZER_PIN.json \
  --output /absolute/new/replay-output \
  --max-bytes <admitted-byte-cap>
```

The exporter calls the original public serializer and verifies its bytes. It
uses the original public import order (`expo_ft.agents` before the replay
module); direct replay-first import exposes an upstream circular import.
It writes `buffers/{step:012d}.pkl` and a provenance manifest. A failed or partial
export remains rejected. Never append it to accepted replay or overwrite it.

## Evidence and limits

CPU tests use the published frame writer with explicit synthetic physics.
They test causal RGB/state/action alignment, all three terminal outcomes,
clipping, resets, split exclusion, capture failure and byte limits. A separate
CPU process uses the installed original save/restore and success-mark APIs.
It proves serialization compatibility; it does not test native physics,
policy inference, real-time throughput, or new task success.

The two existing historical successes at 21001/21002 contain 690 controls, but
their original lossless RGB was not saved. The old replay helper is sealed to
21000/128 controls. This new fresh-collection descriptor intentionally does not
admit those historical IDs. A separate pending salvage condition records those
inputs; it is not a runnable recollection job. Do not decode MP4 as exact policy
observations or claim these episodes are verified recovery demonstrations.

All new reset draws still share source episode 12/reference 11 ancestry. A
historical task success does not prove strict placement, failed-grasp recovery,
or WBC learning. Plain-language wording follows the requested simplified style;
formal ASD-STE100 conformance has not been audited.
