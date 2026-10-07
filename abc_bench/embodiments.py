"""Fail-closed embodiment inspection; no simulator starts at import time."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class URDFInspection:
    """Static asset evidence. This does not establish simulation readiness."""

    path: str
    sha256: str
    robot_name: str
    movable_joints: tuple[str, ...]
    left_arm_joints: tuple[str, ...]
    right_arm_joints: tuple[str, ...]
    torso_joints: tuple[str, ...]
    mesh_count: int
    missing_meshes: tuple[str, ...]
    unresolved_meshes: tuple[str, ...]
    status: str = "requires_adapter"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_urdf(
    path: str | Path, *, package_roots: Mapping[str, str | Path] | None = None
) -> URDFInspection:
    """Inspect source bytes and mesh resolution without private project imports.

    Map ROS package names to explicit package roots. Unknown packages remain
    unresolved. Relative paths resolve against the URDF's parent directory.
    No assets are changed or downloaded. A URDF supplies neither an ABC
    controller nor a validated reward, reset, or camera configuration.
    """
    source = Path(path).resolve(strict=True)
    content = source.read_bytes()
    root = ET.fromstring(content)
    if root.tag != "robot":
        raise ValueError(f"Expected URDF robot root, got {root.tag!r}")
    joints = root.findall("joint")
    names = [joint.get("name") for joint in joints]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("URDF joint names must be nonempty and unique")
    movable = tuple(
        str(joint.get("name")) for joint in joints if joint.get("type") != "fixed"
    )
    missing: list[str] = []
    unresolved: list[str] = []
    meshes = root.findall(".//mesh")
    for mesh in meshes:
        reference = mesh.get("filename", "")
        if reference.startswith("package://"):
            package, separator, relative = reference[10:].partition("/")
            if not separator or package not in (package_roots or {}):
                unresolved.append(reference)
                continue
            candidate = Path((package_roots or {})[package]) / relative
        elif "://" in reference or not reference:
            unresolved.append(reference)
            continue
        else:
            candidate = source.parent / reference
        if not candidate.is_file():
            missing.append(reference)
    return URDFInspection(
        path=str(source),
        sha256=hashlib.sha256(content).hexdigest(),
        robot_name=root.get("name", ""),
        movable_joints=movable,
        left_arm_joints=tuple(
            name for name in movable if name.startswith("left_arm_joint")
        ),
        right_arm_joints=tuple(
            name for name in movable if name.startswith("right_arm_joint")
        ),
        torso_joints=tuple(name for name in movable if name.startswith("torso_joint")),
        mesh_count=len(meshes),
        missing_meshes=tuple(sorted(set(missing))),
        unresolved_meshes=tuple(sorted(set(unresolved))),
    )


def validate_yam_model(model: Any, *, require_cameras: bool = True) -> None:
    """Reject incompatible MuJoCo models before ABC builds unchecked indices.

    The check validates native naming, scalar joints, and joint actuator
    transmission. It does not certify geometry, gains, gripper semantics,
    contact physics, or policy transfer. MuJoCo is imported only on invocation.
    """
    import mujoco

    errors: list[str] = []
    joint_type = mujoco.mjtObj.mjOBJ_JOINT
    actuator_type = mujoco.mjtObj.mjOBJ_ACTUATOR
    scalar_types = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    for side in ("left", "right"):
        bindings = [(f"{side}_joint{i}", f"{side}_joint{i}") for i in range(1, 7)]
        bindings.append((f"{side}_left_finger", f"{side}_gripper"))
        for joint_name, actuator_name in bindings:
            joint_id = mujoco.mj_name2id(model, joint_type, joint_name)
            actuator_id = mujoco.mj_name2id(model, actuator_type, actuator_name)
            if joint_id < 0:
                errors.append(f"missing joint {joint_name}")
            elif int(model.jnt_type[joint_id]) not in scalar_types:
                errors.append(f"non-scalar joint {joint_name}")
            if actuator_id < 0:
                errors.append(f"missing actuator {actuator_name}")
            elif joint_id >= 0 and (
                int(model.actuator_trntype[actuator_id])
                != int(mujoco.mjtTrn.mjTRN_JOINT)
                or int(model.actuator_trnid[actuator_id, 0]) != joint_id
            ):
                errors.append(f"wrong joint transmission for actuator {actuator_name}")
    if require_cameras:
        for camera in ("top", "left", "right"):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                errors.append(f"missing camera {camera}")
    if errors:
        raise ValueError("Incompatible ABC YAM model: " + "; ".join(errors))
