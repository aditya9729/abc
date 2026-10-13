# Packaged original-methods viewer

The package contains the static original-methods results viewer.
The prior released snapshot had SHA256
`3fb17a6910b452e6c1dfd53014c50bc1214b7744b25feeb28de2441c72f1a802`.
The isolated SFT125 candidate changes only this HTML and viewer documentation.
It preserves the existing video identity checks and DOM retention behavior.

Build and install the wheel in an isolated environment:

```sh
uv build --wheel --out-dir /absolute/wheels
python3 -m venv /absolute/neutral-prefix
uv pip install --python /absolute/neutral-prefix/bin/python --no-deps /absolute/wheels/abc_minimal-0.1.0-py3-none-any.whl
env CUDA_VISIBLE_DEVICES= /absolute/neutral-prefix/bin/python -I -B \
  -m abc_bench.export_original_methods --output-dir /absolute/viewer-output
```

Use an explicit absolute output directory. The command writes only `index.html`.
It replaces an existing ordinary HTML file atomically. It preserves existing
`results.json`, videos, and other files. It rejects file and symlink destinations.
It does not import a learner or simulator, start a server, or write results.
The wheel export needs only the Python standard library.

The viewer reads `results.json` and referenced videos from the same directory.
Root must publish authenticated result data and videos separately.
The exporter does not verify task results or make missing results available.
The packaged HTML is a pinned snapshot. Future HTML changes need a new release.
Root owns package integration, version, release, and the served viewer.

The current SFT card prefers `expoft.sft200_training_completed` when present.
It shows actual updates, selected update, held-out loss and early stopping.
The older `sft_completed` actor and its videos remain separate history.

The new cohort reads `matched_frozen_sft125.strategies.pi05_sft125`.
Only accepted native cases contribute to its episode count and success score.
Absent or zero accepted cases display `Pending`, with no default0/3 result.
Future cases must remain unavailable with null success until public acceptance.
There is no matching post-RL125 cohort or RL-effect comparison.

Root publishes `expoft.sft125_frozen_evaluation` only after native acceptance.
It uses the existing `{episodes, successes, planned_episodes, clips}` video shape.
Use null before the first accepted case. Each clip needs a content-hash MP4 name.
The new clips use the same retention function as the historical videos.
`expoft.sft125_frozen_active` supplies the current canonical status and captured seed.

`fasttd3.scene_cache_resource_probe` supplies the completed six-case memory table.
It reports separate process lifetime peaks for ranks0/55 and scene limits12/0/1.
Recorded model/state/output comparisons and raw XML identity have separate scope.
These measurements do not establish aggregate64-world memory or training admission.

The standalone V4 reader is an operational preparation artifact, not a package dependency.
It binds an explicit selected update to genuine export, producer, parent and actor pins.
Its baseline result must use a new cohort field. Never replace the old V3 cohorts.
Root owns the exact reader command, result projection and normal full native release.
CPU DOM doubles do not prove browser playback. Root owns actual browser acceptance.

Run the focused checks against the installed wheel:

```sh
env CUDA_VISIBLE_DEVICES= /absolute/neutral-prefix/bin/python -I -B \
  -m unittest discover -s tests/bench -p test_original_methods_export.py -v
```

See [the architecture](ORIGINAL_METHODS_VIEWER.mmd) for the data flow.
This explanation uses ASD-STE100 guidance. A full compliance check was not performed.
