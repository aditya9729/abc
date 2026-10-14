# Taku in a reconstructed laundry room

This experiment uses Dyna's published Taku visual model and all 869 measured motion samples.
It adds an estimated laundry room with a basket, towels, folding table, washer, dryer, and shelves.
The reference is [Dyna's official workflow poster](https://www.dyna.co/assets/we/workflow-video-poster.jpg).

The recorded motion comes from teleoperation episode 62458843, published in Figure 3.1 of [Dyna-2.1](https://www.dyna.co/dyna-2.1).
Its connection to the separate hour-long autonomous video is unknown.
The room and props are estimates. Towels are static rigid visual objects.
The dryer door follows a scripted schedule. No learned policy or physics stepping runs.
The output is a scene reconstruction with recorded motion, not an autonomous laundry result.

## Reproduce

Use Python 3.12, uv, FFmpeg, and an OSMesa library on Linux.
Create the separate environment and install a normal wheel:

```sh
UV_PROJECT_ENVIRONMENT=runtime/.venv uv sync --locked --group dev --no-install-project
uv build
uv pip install --no-deps --python runtime/.venv/bin/python dist/taku_public_replay-0.2.2-py3-none-any.whl
runtime/.venv/bin/python scripts/fetch_sources.py --output research
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  runtime/.venv/bin/taku-public-replay prepare --research research --output artifacts/robot-prepared
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  runtime/.venv/bin/taku-public-replay prepare-laundry \
  --original artifacts/robot-prepared --research research --output artifacts/laundry-prepared
```

All source URLs and SHA-256 values are pinned in `source_manifest.json`.
The fetch helper rejects changed downloads and conflicting local files.
The original public files retain author ownership. No open redistribution license was found.
Keep fetched assets and generated recordings outside Git.

Pass an OSMesa library directory explicitly to the bounded render wrapper:

```sh
bash scripts/render_laundry_cpu.sh \
  "$PWD/artifacts/laundry-prepared" "$PWD/artifacts/laundry-render" /path/to/osmesa/library-directory
```

The wrapper requires available CPU cores 84 and 85. Change this affinity for another machine.
Use at most two cores with this resource profile.
The runtime uses software rendering, a 6 GiB address-space limit, and a 600-second external deadline.
The local experiment pins Mesa 23.2.1 and MuJoCo 3.3.7. Mesa is separate from the Python environment.
The Python lockfile records all package versions. The font is the system DejaVu Sans font.

## Edit the room

Edit the explicit dimensions and object positions in [laundry.py](src/taku_public_replay/laundry.py).
Run `prepare-laundry` with a new output directory after changes.
`scene_design.json` is a generated evidence file. Editing that file does not regenerate geometry.

The source dryer position, yaw, width, door height, and event times come from the published tracking JSON.
The interpretation as a front-plane frame is inferred from measured reaches.
The local +x axis faces the robot. The cabinet extends along local -x.
The +y edge defines the hinge. Door interpolation between event endpoints is an estimate.
The recorded gap is used as the neighbor gap; that interpretation is also an estimate.
All other object dimensions and placement are estimates from the reference image.

## Verify

```sh
runtime/.venv/bin/python -m pytest -q
runtime/.venv/bin/ruff check src tests scripts
runtime/.venv/bin/ruff format --check src tests scripts
```

Eight focused checks cover decoding, joint transforms, gripper scaling, invalid input, source identity, and door timing.
Native acceptance requires a clean outer exit, software GL, all 869 frames, and complete video decoding.
The sample interval is 1/12 second. The final sample starts at 72.333333 seconds.
The encoded video holds the final frame for one interval and lasts 72.416667 seconds.
No motion samples are removed or accelerated.

Dyna owns the model and recorded motion. This project owns conversion and estimated props.
The coordinator owns execution admission and independent acceptance.
The [architecture diagram](docs/architecture.mmd) shows these boundaries.
This explanation uses STE guidance. Full ASD-STE100 compliance was not checked.
