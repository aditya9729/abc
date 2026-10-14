"""Decode the author's public files without inferring missing physics."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Recording:
    """Measured channels: metres, radians, normalized gripper opening, 12 Hz."""

    fps: float
    n: int
    declared_duration: float
    channels: dict[str, np.ndarray]
    kinds: dict[str, str]
    metadata: dict

    @property
    def timestamps(self) -> np.ndarray:
        return np.arange(self.n, dtype=np.float64) / self.fps


def load_recording(path: Path) -> Recording:
    data = json.loads(path.read_text())
    n, fps = data["n"], data["fps"]
    if not isinstance(n, int) or n < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Recording needs at least two frames and a positive finite frame rate")
    channels, kinds = {}, {}
    for name, (kind, encoded) in data["channels"].items():
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != 2 * n:
            raise ValueError(f"Channel {name} has {len(raw)} bytes; expected {2 * n}")
        scale = data["scale"][kind]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Channel {name} has invalid scale")
        # The website uses Int16Array then Float32Array. Make byte order explicit.
        decoded = (np.frombuffer(raw, dtype="<i2").astype(np.float64) / scale).astype(np.float32)
        decoded.flags.writeable = False
        channels[name], kinds[name] = decoded, kind
    for required in ("robot.x", "robot.y", "robot.yaw", "robot.grip_l", "robot.grip_r"):
        if required not in channels:
            raise ValueError(f"Missing measured channel: {required}")
    duration = float(data["duration"])
    if abs(duration - (n - 1) / fps) > 0.001:
        raise ValueError("Declared recording duration does not match its sample timestamps")
    return Recording(fps, n, duration, channels, kinds, data)


def rotation_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("A moving joint must have a nonzero axis")
    x, y, z = axis / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def joint_origin(joint: dict) -> np.ndarray:
    """Match Three.js Euler(roll,pitch,yaw,'ZYX'): parent to child, right handed."""
    roll, pitch, yaw = joint["rpy"]
    result = np.eye(4)
    result[:3, :3] = (
        rotation_axis([0, 0, 1], yaw)
        @ rotation_axis([0, 1, 0], pitch)
        @ rotation_axis([1, 0, 0], roll)
    )
    result[:3, 3] = joint["xyz"]
    return result


def forward_kinematics(kinematics: dict, recording: Recording, index: int) -> dict[str, np.ndarray]:
    """Return world-from-link transforms for one measured sample; z is up."""
    if not 0 <= index < recording.n:
        raise ValueError("Frame index is outside the recording")
    channels = recording.channels
    root = np.eye(4)
    root[:3, :3] = rotation_axis([0, 0, 1], float(channels["robot.yaw"][index]))
    root[:2, 3] = [channels["robot.x"][index], channels["robot.y"][index]]
    transforms = {kinematics["root"]: root}
    pending = list(kinematics["joints"])
    while pending:
        next_pending = []
        for joint in pending:
            if joint["parent"] not in transforms:
                next_pending.append(joint)
                continue
            if joint["child"] in transforms:
                raise ValueError(f"Link has multiple parents: {joint['child']}")
            name = joint["name"]
            angle = float(channels[f"robot.{name}"][index]) if f"robot.{name}" in channels else 0.0
            for side, suffix in (("left", "l"), ("right", "r")):
                opening = float(channels[f"robot.grip_{suffix}"][index])
                if name == f"{side}_gripper_front_joint":
                    angle = -0.0345 * opening
                elif name == f"{side}_gripper_rear_joint":
                    angle = 0.0345 * opening
            motion = np.eye(4)
            if joint["type"] == "revolute":
                motion[:3, :3] = rotation_axis(joint["axis"], angle)
            elif joint["type"] == "prismatic":
                motion[:3, 3] = np.asarray(joint["axis"]) * angle
            elif joint["type"] != "fixed":
                raise ValueError(f"Unsupported joint type: {joint['type']}")
            transforms[joint["child"]] = transforms[joint["parent"]] @ joint_origin(joint) @ motion
        if len(next_pending) == len(pending):
            raise ValueError("Kinematic tree has a cycle or a missing parent")
        pending = next_pending
    return transforms


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = (
        base64.b64decode(path.read_bytes(), validate=True)
        if path.suffix == ".txt"
        else path.read_bytes()
    )
    magic, version, length = struct.unpack_from("<4sII", raw)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise ValueError("Expected a complete glTF 2 binary asset")
    json_length, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:
        raise ValueError("Missing GLB JSON chunk")
    document = json.loads(raw[20 : 20 + json_length])
    offset = 20 + json_length
    binary_length, binary_type = struct.unpack_from("<II", raw, offset)
    if binary_type != 0x004E4942 or offset + 8 + binary_length != len(raw):
        raise ValueError("Missing or inconsistent GLB binary chunk")
    return document, raw[offset + 8 :]


def decode_meshes(path: Path, destination: Path) -> tuple[list[dict], dict]:
    """Convert the original Draco triangles to OBJ; retain source vertices and normals."""
    import DracoPy

    document, binary = read_glb(path)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = []
    for node in document["nodes"]:
        if "mesh" not in node:
            continue
        if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
            raise ValueError("Unexpected node transform; do not discard the author's transform")
        mesh = document["meshes"][node["mesh"]]
        if len(mesh["primitives"]) != 1:
            raise ValueError("This converter requires the published one-primitive mesh structure")
        primitive = mesh["primitives"][0]
        if primitive.get("mode", 4) != 4:
            raise ValueError("Only triangle primitives are supported")
        extension = primitive["extensions"]["KHR_draco_mesh_compression"]
        view = document["bufferViews"][extension["bufferView"]]
        start = view.get("byteOffset", 0)
        draco = DracoPy.decode(binary[start : start + view["byteLength"]])
        vertices, faces, normals = draco.points, draco.faces, draco.normals
        if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
            raise ValueError("Mesh contains non-finite coordinates")
        if vertices.shape != normals.shape or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("Mesh indices or normals do not match its vertices")
        name = node["name"]
        if not name.replace("_", "").isalnum():
            raise ValueError("Unsafe mesh name")
        output = destination / f"{name}.obj"
        with output.open("w") as file:
            for vertex in vertices:
                file.write("v " + " ".join(f"{float(v):.9g}" for v in vertex) + "\n")
            for normal in normals:
                file.write("vn " + " ".join(f"{float(v):.9g}" for v in normal) + "\n")
            for face in faces:
                file.write("f " + " ".join(f"{int(v) + 1}//{int(v) + 1}" for v in face) + "\n")
        material = document["materials"][primitive["material"]]
        rgba = material.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1])
        metadata.append(
            {
                "name": name,
                "link": name.split("__")[0],
                "file": output.name,
                "rgba": rgba,
                "vertices": len(vertices),
                "triangles": len(faces),
                "sha256": sha256(output),
            }
        )
    return metadata, document["asset"]
