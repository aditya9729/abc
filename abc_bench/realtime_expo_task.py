"""Client-only task snapshot loader for the pristine author rollout program.

The coordinator exports JSON metadata from the server's public make_task_config,
excluding its env callable, then supplies explicit seeds and output paths. No
server adapter, OpenPI model, or JAX dependency is imported in this client.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from abc_bench.realtime_expo_env import ABCYamEnv


def load_client_task(path: str | Path) -> ConfigDict:
    """Load an explicit coordinator-pinned JSON snapshot; never construct native."""
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Client task snapshot must be an existing absolute JSON file")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate client task key: {key}")
            value[key] = item
        return value

    def nonfinite(value):
        raise ValueError(f"Nonfinite client task value: {value}")

    raw = json.loads(
        path.read_text(), object_pairs_hook=unique, parse_constant=nonfinite
    )
    if not isinstance(raw, dict) or "env" in raw:
        raise ValueError("Snapshot must contain task metadata without an env callable")
    required = {
        "env_type",
        "env_name",
        "language_instruction",
        "edit_action_xyzg",
        "dataset_loader",
        "cohort_pin",
        "control_hz",
        "auto_reset_steps",
        "max_episode_steps",
        "example_action",
        "episode_seeds",
        "event_log_path",
    }
    if (
        not required <= set(raw)
        or set(raw) - required - {"environment_profile"}
        or raw["env_type"] != "abc_yam"
    ):
        raise ValueError(
            "Client snapshot must match the explicit ABC server task fields"
        )
    example = np.asarray(raw["example_action"], dtype=np.float32)
    if example.shape != (1, 14) or not np.isfinite(example).all():
        raise ValueError("Client example_action must be finite float32[1,14]")
    raw["example_action"] = example
    config = ConfigDict(raw)
    config.env = ABCYamEnv
    return config


def get_config() -> ConfigDict:
    """Author config factory; root pins the snapshot path in the resolved job."""
    path = os.environ.get("ABC_REALTIME_EXPO_CLIENT_CONFIG")
    if not path:
        raise ValueError(
            "Set ABC_REALTIME_EXPO_CLIENT_CONFIG to the pinned task snapshot"
        )
    return load_client_task(path)
