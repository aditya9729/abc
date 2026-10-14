# Original FastTD3 Hurdle viewer

The optional `fasttd3_original` field in `results.json` shows the unchanged author H1Hand Hurdle experiment. The existing `fasttd3` field still describes earlier ABC work. No experiment data or video is included in this change. An absent field leaves the section hidden. Root owns result acceptance, media conversion, the normal package release, and publication.

The curve and table show **mean evaluation return**, not a success rate. They use the original zero-based vector index, not cumulative transitions or the WandB history row number. Original `frame = vector_index * 128`; collection occurred before the log, so that logged point follows `(vector_index + 1) * 128` training transitions. The last evaluation is at 95,000. The final saved weights at 100,000 have no evaluation in the original recipe. The page never creates a 100k point.

This is one seed from the current author release, not a multi-seed paper reproduction. Each evaluation uses the first episode in each of the 128 training worlds, a deterministic actor and a 1000-control limit. The original GIF comes from one separate render world. It starts with a reset image, samples every other control, requests 30 fps playback, and can end with an autoreset image. It is not a per-control trace or a held-out episode evaluation.

## Publication contract

The outer document remains `sadhana.original-methods.progress/v2`. Add only the optional top-level `fasttd3_original` object. Until root accepts the full run, this complete pending value is sufficient:

```json
{
  "schema": "fasttd3.original-hurdle/v1",
  "status": "pending"
}
```

Pending mode has no score, curve, table or media. Optional `evaluations` and `clips` arrays must be absent or empty. It cannot display partial live measurements as an accepted complete run.

For the accepted state, use these exact fields:

| Field | Value or contract |
| --- | --- |
| `schema` | `fasttd3.original-hurdle/v1` |
| `status` | `accepted_complete_run` |
| `author_commit` | `229ed59bbf43ea2f7a2d5d90d1076314839944d7` |
| `task` | `h1hand-hurdle-v0` |
| `seed` | Integer 1 |
| `num_envs` | Integer 128 |
| `completed_vector_calls` | Integer 100000 |
| `simulator_warning_count` | Required nonnegative safe integer; count printed simulator warning messages in the scanned original stdout |
| `evaluations` | Exactly 19 rows in index order, with the fields below |
| `clips` | Exactly three slots in order: 5000, 50000, 95000 |

Each evaluation row has `vector_index` equal to `5000 * (position + 1)`, finite numeric `mean_return`, and finite numeric `mean_length` in [1,1000]. Copy the actual original `eval_avg_return` and `eval_avg_length`; do not substitute training reward, success fraction, or a later checkpoint's score. The table rounds displayed values to three decimals. The full values remain in JSON. Zero and negative mean returns are valid.

Count warning messages from the accepted run's scanned stdout. Do not default a missing count to zero. Missing, negative, fractional, nonnumeric or unsafe counts reject the accepted record. A positive count adds a visible warning about internal physics resets and unmeasured per-world effects. It preserves otherwise accepted return points and recordings. The count measures printed messages, not affected worlds or recovery events. A zero count means only that no warning messages were found in the scanned stdout. It does not prove clean physics. Finite checkpoints and a zero process exit do not establish warning-free simulation.

A pending clip slot contains only its fixed index and status, for example:

```json
{"vector_index": 5000, "status": "pending"}
```

An accepted clip uses these fields:

| Field | Value or contract |
| --- | --- |
| `vector_index` | Its preselected index |
| `status` | `accepted_original_recording` |
| `video` | First 16 characters of `sha256`, followed by `.mp4`; a local sibling file |
| `sha256` | 64 lowercase hexadecimal characters for the published MP4 |
| `source_gif_sha256` | 64 lowercase hexadecimal characters for the original author GIF |
| `playback_timing` | `source_gif_unchanged` |

Root must check the source association and conversion timing before setting the accepted clip status. The browser validates names and metadata; it does not hash video bytes or authenticate a producer. Conversion must retain the recorded GIF timing. Do not retime to the native control rate or infer a full per-control episode from the converted clip. Slots are chosen before the final result and are not replaced by the best-looking rollout. A pending slot has no `video`, `sha256`, or `source_gif_sha256` field.

The recording selector has an accessible label and uses native video controls. It pauses the old clip when the selection changes. An unchanged 30-second refresh preserves selection and playback. Absent, pending, or invalid data hides or clears prior media. Malformed data reports through the existing error area; it cannot leave the section visible as accepted.

## Verification and review

Only the packaged HTML, a CPU DOM fixture test, this document, and the diagram change. The old comparison heading now explicitly says "Earlier ABC comparison", so its five-bottle discussion cannot be read as Hurdle's evaluation. Existing panel data/render functions, original learner code, author outputs, runtimes, data readers, result JSON and the served viewer are unchanged. No browser, native simulator, GPU, or installer ran.

Run the existing Node binary from the isolated worktree:

```sh
/home/user/aditya/runtimes/node-v22.16.0-linux-x64/bin/node --test \
  tests/bench/test_fasttd3_original_viewer.cjs \
  tests/bench/test_mulligan_local_divl_viewer.cjs \
  tests/bench/test_catcher_frozen_viewer.cjs
```

All 18 tests pass: nine FastTD3 tests and nine existing Catcher/Mulligan tests. Inputs are explicitly synthetic DOM/media fixtures. They check the optional/pending states, return/length rendering, negative returns, visible warning counts, invalid-count rejection, fixed clip slots, source/playback metadata rejection, refresh identity, media cleanup, and existing panel preservation. The default accepted fixture supplies an explicit zero count; it is not a native measurement. These tests are not native results or actual browser layout proof. Root will build the normal full-asset wheel and check publication after actual run acceptance.

Root supplies accepted records and media. I prepared and tested the optional viewer. The [diagram](FASTTD3_ORIGINAL_VIEWER_20261014.mmd) shows these roles and the proposed publication boundary. STE guidance was used; full compliance was not checked.
