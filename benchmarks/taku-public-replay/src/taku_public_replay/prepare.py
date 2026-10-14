"""Prepare a visual-only MuJoCo scene. No simulator is imported here."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .assets import decode_meshes, forward_kinematics, load_recording, sha256


def prepare(research: Path, output: Path) -> dict:
    research, output = research.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recording = load_recording(research / "tracking_data.json")
    kinematics = json.loads((research / "dvt1_kin.json").read_text())
    meshes, generator = decode_meshes(research / "dvt1_body.glb.txt", output / "meshes")
    links = list(forward_kinematics(kinematics, recording, 0))
    if set(mesh["link"] for mesh in meshes) - set(links):
        raise ValueError("Some visual meshes have no author-defined kinematic link")
    transforms = np.asarray(
        [
            [poses[link] for link in links]
            for poses in (
                forward_kinematics(kinematics, recording, frame) for frame in range(recording.n)
            )
        ]
    )
    if not np.isfinite(transforms).all():
        raise ValueError("Forward kinematics produced non-finite transforms")
    np.savez_compressed(
        output / "measured_trajectory.npz",
        timestamps=recording.timestamps,
        links=np.asarray(links),
        world_from_link=transforms,
        **recording.channels,
    )
    model = ET.Element("mujoco", model="Taku public measured-state replay — no dynamics")
    ET.SubElement(model, "compiler", angle="radian", meshdir="meshes", inertiafromgeom="false")
    ET.SubElement(model, "option", gravity="0 0 0")
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "quality", shadowsize="1024", offsamples="0")
    asset = ET.SubElement(model, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        width="256",
        height="256",
        rgb1="0.78 0.79 0.76",
        rgb2="0.85 0.86 0.83",
    )
    ET.SubElement(
        asset,
        "material",
        name="grid",
        texture="grid",
        texrepeat="12 12",
        texuniform="true",
        reflectance="0",
    )
    world = ET.SubElement(model, "worldbody")
    ET.SubElement(
        world,
        "light",
        pos="-2 -3 6",
        dir="0 0 -1",
        directional="true",
        diffuse="0.8 0.8 0.8",
        ambient="0.4 0.4 0.4",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "geom",
        name="reference_grid",
        type="plane",
        size="5 5 0.01",
        material="grid",
        contype="0",
        conaffinity="0",
    )
    bodies = {link: ET.SubElement(world, "body", name=link, mocap="true") for link in links}
    for mesh in meshes:
        ET.SubElement(asset, "mesh", name=mesh["name"], file=mesh["file"], inertia="shell")
        ET.SubElement(
            bodies[mesh["link"]],
            "geom",
            name=mesh["name"],
            type="mesh",
            mesh=mesh["name"],
            rgba=" ".join(map(str, mesh["rgba"])),
            contype="0",
            conaffinity="0",
            density="0",
            group="1",
        )
    ET.indent(model)
    ET.ElementTree(model).write(output / "taku_visual_replay.xml", encoding="unicode")
    report = {
        "schema_version": 1,
        "claim": "Native MuJoCo visual replay of author measured states",
        "not_claimed": [
            "dynamic simulation",
            "RL inference",
            "laundry success",
            "same hour-long episode",
            "exact laundry room",
            "physical inertias or contact parameters",
        ],
        "input_files": {
            name: sha256(research / name)
            for name in ["dvt1_body.glb.txt", "dvt1_kin.json", "tracking_data.json"]
        },
        "episode": recording.metadata["episode"],
        "frames": recording.n,
        "fps": recording.fps,
        "declared_duration_seconds": recording.declared_duration,
        "last_sample_time_seconds": float(recording.timestamps[-1]),
        "encoded_duration_seconds": recording.n / recording.fps,
        "last_frame_hold_seconds": 1 / recording.fps,
        "channels": len(recording.channels),
        "links": len(links),
        "meshes": meshes,
        "glb_generator": generator,
        "coordinate_system": "right-handed, z up, metres/radians",
        "joint_origin_rotation": "Rz(yaw) @ Ry(pitch) @ Rx(roll), then joint motion",
        "source_quaternion_order": "target channels xyzw; native mocap wxyz",
        "scene": "Neutral reference grid only, matching the public tracking viewer's scene scope",
        "source_dryer_metadata_used": False,
        "model_sha256": sha256(output / "taku_visual_replay.xml"),
        "trajectory_sha256": sha256(output / "measured_trajectory.npz"),
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
