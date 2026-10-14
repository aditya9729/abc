# Taku laundry-room reconstruction — 14 October 2026

The local viewer shows Dyna's published Taku robot and recorded motion in an estimated laundry room. It includes a basket, towels, folding table, washer, dryer and shelves. [Open the viewer](http://127.0.0.1:8771/). Both recordings contain all 869 samples at 12 frames per second and last 72.416667 seconds.

This is a visual replay. The public material inspected does not supply the learned controller, original room assets, robot dynamics or cloth model needed to reproduce autonomous laundry handling. There is no policy inference, physics stepping or local task-success score.

## Source and fidelity

The requested [post](https://x.com/MarilynLiu97/status/2109808785615458626) discusses Dyna-2.1 and Taku. The [official Dyna article](https://www.dyna.co/dyna-2.1) supplies the robot visual model, joint tree, recorded channels, dryer metadata and room reference. Figure 3.1 identifies the motion as one teleoperation episode. Its connection to the separate hour-long autonomous video is unknown.

| Component | Implemented result | Limit |
| --- | --- | --- |
| Robot | Published Taku model: 105 meshes, 48 joints and 49 links | Visual geometry; no verified inertial or contact model |
| Motion | All 66 recorded channels decoded; measured root and link poses replayed | Recorded episode 62458843, not a learned controller |
| Dryer | Published position, yaw, width, door height and event endpoints | Frame interpretation, gap meaning and door interpolation are estimates |
| Room and props | Basket, towels, folding table, washer, dryer, shelves and walls | Estimated dimensions and placement from the reference image |
| Towels | Static visual objects | No deformation, grasping or folding physics |
| Timing | Original 12 Hz sample sequence; final frame held for one interval | No speedup; no demonstration of a 100 Hz controller |

The author reference image is separate from the local recordings. Its autonomous and playback-speed labels describe the author's video only. The replay preserves measured motion; it does not establish accurate contact with the estimated props.

## Implementation and use

Source is in [benchmarks/taku-public-replay](../../benchmarks/taku-public-replay/README.md). This is a separately locked experiment, added as a reviewed exception to the benchmark folder boundary. It does not change upstream ABC, the original learners, or the active training environments. Version 0.2.2 is installed as a normal wheel in its own runtime.

The source manifest pins four public downloads by URL and SHA-256. The loader decodes the published little-endian int16 channels with their recorded scales. It computes joint transforms with the source rotation order and gripper mapping. MuJoCo 3.3.7 receives those transforms as mocap poses. The room has zero dynamic degrees of freedom, zero actuators and disabled contacts. Rendering calls `mj_forward`, with no `mj_step`.

Follow the package README to fetch sources, prepare the model and render on CPU. Edit object dimensions in `src/taku_public_replay/laundry.py`, then prepare a new output directory. `scene_design.json` is generated evidence; editing it does not regenerate the room. The [viewer runbook](../../benchmarks/taku-public-replay/viewer/README.md) explains local publication and browser checks.

Raw data, runtimes, recordings and native evidence remain outside Git under:

```text
/home/user/aditya/RL/builds/post-scene-reproduction-20261014/
  research/                         # Original source responses and model/log assets
  artifacts/laundry-prepared-v3/     # MuJoCo XML, meshes, pose arrays and scene manifest
  artifacts/laundry-render-v2/       # Complete MP4, frames.jsonl and COMPLETE.json
  root-native-scene-02/              # Independent native acceptance and exit evidence
  independent-review/               # Source, transform and camera reviews
  viewer-public-01/                  # Only the five public viewer files
```

The [JSON evidence index](TAKU_LAUNDRY_RECONSTRUCTION_20261014.json) records exact paths, sizes and hashes. No additional redistribution license was found for the author assets. Large assets and generated media are not committed.

## Verification

Eight focused installed-package tests pass. They cover channel decoding, transforms, gripper scaling, invalid input, source identity and door timing. Ruff checks pass. Independent review authenticated the source inputs and normal installed payloads. It found no blocking transform or decode defect within the visual-replay scope.

The accepted room render took 172.28 seconds on two CPU cores with OSMesa llvmpipe. It exited 0 and left no owned worker processes. Root checked all 869 frame identifiers, timestamps, door angles and finite pose errors. Maximum position error was zero; maximum rotation-matrix error was 7.78e-16. Full video decoding confirms 960 by 640 pixels, 12 fps and 72.416667 seconds. The separate robot-only replay has the same complete sequence at 640 by 480 pixels.

The first room camera put walls in front of the scene. Root cancelled that attempt after 53.91 seconds and kept its evidence. The final camera uses the open side of the room. Root inspected the first, middle and last frames. The repair changes the camera only; robot motion and room geometry stay fixed.

The first browser attempt exposed a transport defect: the basic static server returned whole files to byte-range requests, so video seeking failed. A separate read-only server now supplies byte ranges and exposes only the five published files on localhost. It changes video delivery only. Native recordings and acceptance records remain unchanged.

Six DOM checks and six real HTTP checks with synthetic files pass. The final browser check uses both actual recordings: loading, seeking to 75%, playback, refresh and desktop/mobile layout checks pass, with no page errors. Root inspected both screenshots and confirmed that no owned browser helper remained. The final HTTP check verifies every served file against its local bytes, plus partial-video and invalid-range responses. Exact evidence is indexed in the JSON report.

These focused checks do not resolve historical failures in the broader ABC suite. They do not establish autonomous control, cloth behavior or reproduction of Dyna's RL results.

## Architecture, team and next experiment

The [architecture diagram](TAKU_LAUNDRY_RECONSTRUCTION_20261014.mmd) shows inputs, owners, data flow and execution control. Dyna owns the published model and log. The scene agent prepared their conversion and the estimated props. The independent reviewer checked source identity and transforms. The viewer agent prepared display and transport code. The coordinator admitted CPU runs, checked complete outputs, inspected the view and published the result.

The useful control lesson is to keep task planning, robot motion policy and whole-body control as separate interfaces. This replay checks geometry and recorded pose conversion only. An RL experiment still needs a physical robot model, validated contacts and cloth, a controller/action interface, valid resets, and a defined reward and success test. Those are missing components, not measurements from this scene. They must be supplied and checked before training can test laundry handling or transfer to R1 Lite.

This report follows ASD-STE100 guidance. Full compliance was not checked. The broader original-method RL campaign remains active; this scene does not complete it.
