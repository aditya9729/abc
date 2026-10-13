# Packaged original-methods viewer

The package contains the exact HTML from the current results viewer.
The template snapshot has SHA256
`3fb17a6910b452e6c1dfd53014c50bc1214b7744b25feeb28de2441c72f1a802`
and contains 23825 bytes. Its CSS and JavaScript are unchanged.

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

Run the focused checks against the installed wheel:

```sh
env CUDA_VISIBLE_DEVICES= /absolute/neutral-prefix/bin/python -I -B \
  -m unittest discover -s tests/bench -p test_original_methods_export.py -v
```

See [the architecture](ORIGINAL_METHODS_VIEWER.mmd) for the data flow.
This explanation uses ASD-STE100 guidance. A full compliance check was not performed.
