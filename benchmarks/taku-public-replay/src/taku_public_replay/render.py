"""CPU-only native MuJoCo rendering of measured poses, without physics stepping."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .assets import sha256


def render_recording(prepared: Path, output: Path, wall_seconds: float) -> dict:
    required_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LP_NUM_THREADS": "2",
    }
    for key, expected in required_environment.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"Set {key}={expected!r} before native rendering")
    if len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError("Native rendering requires affinity to at most two CPU cores")
    if not 0 < wall_seconds <= 540:
        raise ValueError("Wall-time limit must be positive and at most 540 seconds")

    import mujoco
    import numpy as np
    from OpenGL import GL
    from PIL import Image, ImageDraw, ImageFont

    started = time.monotonic()
    prepared, output = prepared.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = json.loads((prepared / "preparation.json").read_text())
    design = None
    if "scene_design_sha256" in preparation:
        if sha256(prepared / "scene_design.json") != preparation["scene_design_sha256"]:
            raise ValueError("Scene design changed after preparation")
        design = json.loads((prepared / "scene_design.json").read_text())
    for field, filename in (
        ("model_sha256", "taku_visual_replay.xml"),
        ("trajectory_sha256", "measured_trajectory.npz"),
    ):
        if sha256(prepared / filename) != preparation[field]:
            raise ValueError(f"Prepared {filename} changed after preparation")
    for mesh in preparation["meshes"]:
        if sha256(prepared / "meshes" / mesh["file"]) != mesh["sha256"]:
            raise ValueError(f"Prepared mesh changed: {mesh['file']}")
    trajectory = np.load(prepared / "measured_trajectory.npz", allow_pickle=False)
    poses, timestamps = trajectory["world_from_link"], trajectory["timestamps"]
    links = trajectory["links"].tolist()
    model = mujoco.MjModel.from_xml_path(str(prepared / "taku_visual_replay.xml"))
    if model.nv or model.nu or model.nq:
        raise RuntimeError("The replay must contain no dynamic joints or actuators")
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link) for link in links]
    if min(body_ids) < 0:
        raise RuntimeError("A prepared kinematic link is missing in MuJoCo")
    mocap_ids = model.body_mocapid[body_ids]
    if min(mocap_ids) < 0:
        raise RuntimeError("Every kinematic link must be an explicit mocap body")
    door_mocap_id = None
    if design is not None:
        from .laundry import door_angle, dryer_door_pose

        door_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scene_dryer_door")
        if door_id < 0 or model.body_mocapid[door_id] < 0:
            raise RuntimeError("Reconstructed dryer door has no native mocap body")
        door_mocap_id = model.body_mocapid[door_id]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 135, -18, 3.4
    if design is not None:
        for field in ("azimuth", "elevation", "distance"):
            setattr(camera, field, design["camera"][field])
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False
    width, height = (960, 640) if design is not None else (640, 480)
    video_name = (
        "taku-laundry-reconstruction.mp4" if design is not None else "taku-measured-replay.mp4"
    )
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    encoder_command = [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(preparation["fps"]),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / video_name),
    ]
    max_position_error, max_rotation_error = 0.0, 0.0
    renderer = None
    encoder = None
    completed = 0
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        gl = {
            name: GL.glGetString(enum).decode()
            for name, enum in (
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
            )
        }
        if not any(token in gl["renderer"].lower() for token in ("llvmpipe", "softpipe")):
            raise RuntimeError(f"Software-only renderer required; received {gl['renderer']}")
        (output / "STARTED.json").write_text(
            json.dumps(
                {
                    "renderer": gl,
                    "pid": os.getpid(),
                    "affinity": sorted(os.sched_getaffinity(0)),
                    "environment": required_environment,
                },
                indent=2,
            )
        )
        encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_log = output / "frames.jsonl"
        with frame_log.open("w") as log:
            for index, (sample_poses, timestamp) in enumerate(zip(poses, timestamps, strict=True)):
                if time.monotonic() - started > wall_seconds:
                    raise TimeoutError(f"Replay exceeded its {wall_seconds:g}-second wall limit")
                for mocap_id, pose in zip(mocap_ids, sample_poses, strict=True):
                    data.mocap_pos[mocap_id] = pose[:3, 3]
                    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], pose[:3, :3].flatten())
                scene_angle = None
                if design is not None:
                    door_pos, door_quat = dryer_door_pose(
                        float(timestamp), design["source_scene_metadata"]
                    )
                    data.mocap_pos[door_mocap_id] = door_pos
                    data.mocap_quat[door_mocap_id] = door_quat
                    scene_angle = door_angle(float(timestamp), design["source_scene_metadata"])
                data.time = float(timestamp)
                # Kinematics only. No mj_step, controls, inference, contact, or cloth physics.
                mujoco.mj_forward(model, data)
                position_error = float(np.max(np.abs(data.xpos[body_ids] - sample_poses[:, :3, 3])))
                rotation_error = float(
                    np.max(np.abs(data.xmat[body_ids].reshape(-1, 3, 3) - sample_poses[:, :3, :3]))
                )
                max_position_error = max(max_position_error, position_error)
                max_rotation_error = max(max_rotation_error, rotation_error)
                if position_error > 1e-10 or rotation_error > 1e-10 or data.ncon:
                    raise RuntimeError(
                        "Native poses differ from source transforms or contacts are active"
                    )
                camera.lookat[:] = (
                    design["camera"]["lookat"]
                    if design is not None
                    else [sample_poses[0, 0, 3], sample_poses[0, 1, 3], 0.85]
                )
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width, 61), fill="#172327")
                draw.text(
                    (12, 8),
                    "TAKU  |  Reconstructed laundry room"
                    if design is not None
                    else "TAKU  |  Author measured-state replay",
                    font=title_font,
                    fill="white",
                )
                draw.text(
                    (12, 34),
                    "Published robot + measured log  |  Estimated room, dimensions and props"
                    if design is not None
                    else "Original geometry + logged poses  |  MuJoCo / CPU",
                    font=text_font,
                    fill="#c3ded0",
                )
                draw.rectangle((0, height - 56, width, height), fill="#172327")
                draw.text(
                    (12, height - 49),
                    "Static towels; scripted dryer door. No trained policy, cloth dynamics or task-success claim."
                    if design is not None
                    else "No policy or dynamics. Laundry scene and cloth are not reproduced.",
                    font=text_font,
                    fill="white",
                )
                draw.text(
                    (12, height - 28),
                    f"Episode 62458843  |  {index + 1:03d}/869  |  t = {timestamp:06.3f} s  |  12 Hz",
                    font=text_font,
                    fill="#c3ded0",
                )
                if index in (0, 200, 400, 600, len(poses) - 1):
                    frame.save(output / f"frame-{index:04d}.png")
                encoder.stdin.write(frame.tobytes())
                log.write(
                    json.dumps(
                        {
                            "frame": index,
                            "timestamp_seconds": float(timestamp),
                            "native_position_error": position_error,
                            "native_rotation_error": rotation_error,
                            "scripted_dryer_door_angle_radians": scene_angle,
                        }
                    )
                    + "\n"
                )
                completed += 1
                if completed % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "rendered": completed,
                                "total": len(poses),
                                "elapsed_seconds": round(time.monotonic() - started, 2),
                            }
                        ),
                        flush=True,
                    )
        encoder.stdin.close()
        encoder.wait(timeout=max(1, wall_seconds - (time.monotonic() - started)))
        if encoder.returncode:
            raise RuntimeError(
                f"Video encoder exited {encoder.returncode}: {encoder.stderr.read().decode()}"
            )
        report = {
            "status": "complete",
            "mode": "measured-state kinematic replay; no physics stepping",
            "scene_design": design,
            "frames": completed,
            "fps": preparation["fps"],
            "last_sample_time_seconds": float(timestamps[-1]),
            "video_duration_seconds": completed / preparation["fps"],
            "last_frame_hold_seconds": 1 / preparation["fps"],
            "mujoco_version": mujoco.__version__,
            "opengl": gl,
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nbody": model.nbody,
            "nmocap": model.nmocap,
            "max_position_error_m": max_position_error,
            "max_rotation_matrix_error": max_rotation_error,
            "wall_seconds": time.monotonic() - started,
            "video_name": video_name,
            "video_sha256": sha256(output / video_name),
            "preparation_sha256": sha256(prepared / "preparation.json"),
        }
    except BaseException as error:
        (output / "INCOMPLETE.json").write_text(
            json.dumps({"frames": completed, "error": str(error)}) + "\n"
        )
        raise
    finally:
        if encoder is not None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait(timeout=5)
            encoder.stderr.close()
        if renderer is not None:
            renderer.close()
        trajectory.close()
    # A successful receipt requires resource cleanup as well as complete frame output.
    (output / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
