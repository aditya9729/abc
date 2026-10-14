import base64
import json

import numpy as np
import pytest

from taku_public_replay.assets import Recording, forward_kinematics, joint_origin, load_recording


def make_recording(**extra):
    channels = {
        name: np.zeros(2)
        for name in ["robot.x", "robot.y", "robot.yaw", "robot.grip_l", "robot.grip_r"]
    }
    channels.update({name: np.array(values) for name, values in extra.items()})
    return Recording(12, 2, 1 / 12, channels, {}, {})


def test_origin_rotation_uses_zyx_order():
    origin = joint_origin({"rpy": [np.pi / 2, 0, np.pi / 2], "xyz": [1, 2, 3]})
    np.testing.assert_allclose(origin @ [0, 1, 0, 1], [1, 2, 4, 1], atol=1e-12)
    np.testing.assert_allclose(origin @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)


def test_child_translation_follows_parent_rotation_in_unordered_tree():
    kin = {
        "root": "base",
        "joints": [
            {
                "name": "tip",
                "type": "fixed",
                "parent": "arm",
                "child": "tip",
                "xyz": [1, 0, 0],
                "rpy": [0, 0, 0],
                "axis": [0, 0, 1],
            },
            {
                "name": "arm",
                "type": "revolute",
                "parent": "base",
                "child": "arm",
                "xyz": [0, 0, 1],
                "rpy": [0, 0, 0],
                "axis": [0, 0, 1],
            },
        ],
    }
    result = forward_kinematics(kin, make_recording(**{"robot.arm": [np.pi / 2, 0]}), 0)
    np.testing.assert_allclose(result["tip"][:3, 3], [0, 1, 1], atol=1e-12)


def test_public_gripper_opening_mapping_preserves_sign_and_scale():
    kin = {
        "root": "base",
        "joints": [
            {
                "name": f"left_gripper_{finger}_joint",
                "type": "prismatic",
                "parent": "base",
                "child": finger,
                "xyz": [0, 0, 0],
                "rpy": [0, 0, 0],
                "axis": [0, 1, 0],
            }
            for finger in ("front", "rear")
        ],
    }
    result = forward_kinematics(kin, make_recording(**{"robot.grip_l": [1, 0]}), 0)
    np.testing.assert_allclose(result["front"][:3, 3], [0, -0.0345, 0])
    np.testing.assert_allclose(result["rear"][:3, 3], [0, 0.0345, 0])


def test_invalid_tree_fails_instead_of_dropping_missing_links():
    kin = {"root": "base", "joints": [{"parent": "missing", "child": "tip"}]}
    with pytest.raises(ValueError, match="missing parent"):
        forward_kinematics(kin, make_recording(), 0)


def test_log_decode_preserves_signed_little_endian_values_and_rejects_partial_channel(tmp_path):
    encoded = base64.b64encode(np.array([-1234, 2345], dtype="<i2").tobytes()).decode()
    data = {
        "n": 2,
        "fps": 12,
        "duration": 0.083,
        "scale": {"pos": 1000},
        "channels": {key: ["pos", encoded] for key in make_recording().channels},
    }
    path = tmp_path / "log.json"
    path.write_text(json.dumps(data))
    recording = load_recording(path)
    np.testing.assert_array_equal(
        recording.channels["robot.x"], np.array([-1.234, 2.345], np.float32)
    )
    assert not recording.channels["robot.x"].flags.writeable
    data["channels"]["robot.x"][1] = base64.b64encode(b"\0\0").decode()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="expected 4"):
        load_recording(path)
