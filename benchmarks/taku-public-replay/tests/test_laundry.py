import math
import hashlib
import json

import numpy as np
import pytest

from taku_public_replay.laundry import door_angle, dryer_door_pose, prepare_laundry


def scene():
    return {
        "dryer": {"x": 2, "y": 3, "yaw": math.pi / 2},
        "width": 0.6,
        "door_z": 1,
        "door": {"t_open": [10, 14], "t_close": [20, 24], "angle_deg": 105, "hinge": "+y"},
    }


def test_scripted_door_preserves_published_endpoints_and_stays_closed_outside_events():
    data = scene()
    for time in (0, 10, 24, 70):
        assert door_angle(time, data) == 0
    for time in (14, 16, 20):
        assert door_angle(time, data) == math.radians(105)
    assert door_angle(12, data) == math.radians(52.5)
    assert door_angle(22, data) == math.radians(52.5)


def test_door_hinge_stays_at_transformed_plus_y_edge_while_only_orientation_changes():
    first_pos, first_quat = dryer_door_pose(0, scene())
    open_pos, open_quat = dryer_door_pose(16, scene())
    np.testing.assert_allclose(first_pos, [1.7, 3, 1], atol=1e-12)
    np.testing.assert_array_equal(first_pos, open_pos)
    assert not np.allclose(first_quat, open_quat)
    assert abs(np.linalg.norm(open_quat) - 1) < 1e-12


def test_changed_scene_metadata_rejected_before_output_is_created(tmp_path):
    original, research, output = (tmp_path / name for name in ("original", "research", "output"))
    original.mkdir()
    research.mkdir()
    (original / "taku_visual_replay.xml").write_bytes(b"model")
    (original / "measured_trajectory.npz").write_bytes(b"trajectory")
    (research / "tracking_data.json").write_bytes(b"changed recording")
    (original / "preparation.json").write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(b"model").hexdigest(),
                "trajectory_sha256": hashlib.sha256(b"trajectory").hexdigest(),
                "input_files": {
                    "tracking_data.json": hashlib.sha256(b"original recording").hexdigest()
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Raw tracking metadata"):
        prepare_laundry(original, research, output)
    assert not output.exists()
