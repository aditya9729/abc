# Local DIVL full-grid viewer contract

The optional `mulligan.local_divl.full_grid_evaluation` record adds a separate
card. It does not change the existing 50-start scores or videos. This 8,000-start
cohort includes the earlier 50 pilot starts. Its score stays separate from the
pilot score. The 400-episode training monitors remain a separate cohort.
An absent record hides
the card. A pending record shows only a reported completion count. It does not
show a success count, failure count or partial success rate.

| Field | Required value or meaning |
| --- | --- |
| `schema` | `mulligan.local-divl-full-grid/v1` |
| `status` | `pending` or `accepted_complete` |
| `task` | `square_narrow` |
| `run_id` | Canonical `author-subset-YYYYMMDDTHHMMSS-hex6` run ID |
| `training_seed` | `1` |
| `num_action_samples` | `32` |
| `eval_seed`, `env_seed` | Both `42` |
| `episodes` | `8000` |
| `completed_episodes` | Integer from 0 through 8000; exactly 8000 when accepted |
| `successes` | `null` when pending; accepted integer from 0 through 8000 |
| `receipt` | Absent/null when pending; accepted `{path, bytes, sha256}` pin |

The accepted receipt must have a nonempty path, positive safe integer byte count,
and a lowercase SHA256. Root authenticates the actual source, checkpoint, locked
manifest, 8,000 unique points, outcomes, original process exit and cleanup before
it maps an accepted record. The UI validates the display contract. It cannot
authenticate local receipt bytes or grant acceptance. Failures equal 8,000 minus
accepted successes. No default score or new video is introduced.

Root's active run is `author-subset-20261014T135652-749c2a`.
Its score is pending while this candidate is prepared. This is the accepted local
DIVL checkpoint from training seed 1, evaluated with N32 on the complete locked
grid. The explicit evaluation/reset seeds differ from upstream unseeded defaults.
It does not establish the historical five-seed aggregate, exact paper agreement,
or fresh deployment-data collection. The historical seed-1 CSV reference belongs
in the separately authenticated result report; it is not part of this UI schema.

The renderer uses only text and a three-column count table. It adds no media
nodes. It runs before the existing policy/video refresh identity check returns.
The old renderer bodies and video data remain unchanged.

The Node checks use explicit synthetic records. They check hidden/absent and
pending states, complete accepted counts, malformed scope/evidence rejection,
and prior video identity/position across refresh, completion and removal.
The initial export check supplies the candidate HTML as an explicit resource
fixture to the existing normally installed exporter. It verifies exact bytes
and prior fixture data preservation. That fixture does not establish installed
candidate-wheel fidelity.

Root has now built the normal full-asset wheel and installed it in a separate
environment for resource inspection. All 1,149 payloads match source, wheel and
that installation, including 933 native assets. The real installed exporter
produces the exact candidate HTML and preserves existing synthetic result and
media files. Compared with the current canonical installation, only the HTML
payload differs; the other 1,148 payloads match. The separate inspection
environment deliberately has no training dependencies. No model, simulator or
GPU ran for these checks. [Preparation evidence](MULLIGAN_FULL_GRID_VIEWER_PREPARATION.json).

The canonical runtime and live viewer remain unchanged while the original
evaluation runs. After native acceptance, root must install the checked wheel
in the canonical runtime, verify the installed payloads, export the viewer,
publish the actual accepted record and run the actual browser checks. The
prepared wheel and fixture checks do not accept the native result.

The [diagram](MULLIGAN_FULL_GRID_VIEWER.mmd) shows this handoff.
The explanation uses ASD-STE100 guidance. Full compliance was not checked.
