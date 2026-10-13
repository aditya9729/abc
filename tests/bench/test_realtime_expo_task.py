"""Client metadata checks; no native factory or server adapter import."""

import json
import subprocess
import sys

import numpy as np
import pytest

from abc_bench.realtime_expo_env import ABCYamEnv
from abc_bench.realtime_expo_task import get_config, load_client_task


def snapshot():
    # Explicit synthetic metadata fixture; it is not an accepted cohort or job.
    return {
        "env_type": "abc_yam",
        "env_name": "abc_yam_five_bottle",
        "language_instruction": "put plastic bottles in bin",
        "edit_action_xyzg": False,
        "dataset_loader": "expo_abc_adapter.realtime_data:load_train_dataset",
        "cohort_pin": {
            "path": "/synthetic/cohort.json",
            "bytes": 1,
            "sha256": "0" * 64,
        },
        "control_hz": 1.0 / 0.034,
        "auto_reset_steps": 3540,
        "max_episode_steps": 3540,
        "example_action": [[0.0] * 14],
        "episode_seeds": [],
        "event_log_path": "",
    }


def test_client_snapshot_factory_preserves_server_metadata(tmp_path, monkeypatch):
    path = tmp_path / "task.json"
    raw = snapshot()
    path.write_text(json.dumps(raw))
    monkeypatch.setenv("ABC_REALTIME_EXPO_CLIENT_CONFIG", str(path))
    config = get_config()
    assert config.env is ABCYamEnv
    assert config.example_action.dtype == np.float32
    assert config.example_action.tolist() == raw["example_action"]
    assert dict(config.cohort_pin) == raw["cohort_pin"]
    assert config.episode_seeds == raw["episode_seeds"]
    # Other benchmark tests can import native libraries into this process.
    # Verify this import contract in a fresh installed-package interpreter.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; from abc_bench.realtime_expo_task import get_config; "
                "get_config(); assert not any(name in sys.modules for name in "
                "('jax','mujoco','torch','warp','expo_abc_adapter'))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_missing_snapshot_setting_fails(monkeypatch):
    monkeypatch.delenv("ABC_REALTIME_EXPO_CLIENT_CONFIG", raising=False)
    with pytest.raises(ValueError, match="Set ABC_REALTIME"):
        get_config()


@pytest.mark.parametrize(
    "change", [{"env": "bad"}, {"env_type": "droid"}, {"example_action": [[0] * 7]}]
)
def test_invalid_snapshot_never_constructs_native(tmp_path, change):
    path = tmp_path / "task.json"
    path.write_text(json.dumps(snapshot() | change))
    with pytest.raises(ValueError):
        load_client_task(path)


def test_duplicate_fields_refuse(tmp_path):
    path = tmp_path / "task.json"
    path.write_text('{"env_type":"abc_yam","env_type":"droid"}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_client_task(path)
