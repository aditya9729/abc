"""CPU protocol tests using the published frame writer and synthetic physics."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sadhana_r1lite import recovery_continuation as api

from abc_bench import expo_train_collection as collect
from abc_bench import expo_train_export as exporter


def write(path, value):
    path.write_text(json.dumps(value))
    return collect.pin(path)


@pytest.fixture
def descriptor(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    seeds = [29001, 29002, 29003]
    reservation = write(
        tmp_path / "reservation.json",
        {
            "schema": "sadhana.expo_train_seed_reservation/1",
            "owner": "/root",
            "split": "TRAIN",
            "seeds": seeds,
            "startup_seed": 20999,
        },
    )
    native = write(tmp_path / "native.json", {"synthetic_test_only": True})
    settings = write(
        tmp_path / "settings.json",
        {
            "schema": "sadhana.r1lite_expo_task/1",
            "episode_seeds": [20999, *seeds],
            "env_config_path": native["path"],
            "event_log_path": "",
            "record_video": False,
            "cohort_pin": native,
            "selected_train_seeds": [20001],
            "example_action": [0.0] * 14,
        },
    )
    arguments = [
        "--only_base_actions=True",
        "--checkpoint_step=0",
        "--seed=42",
        "--replan_steps=8",
        "--delay=5",
        "--sim_latency=0",
        "--config.N=32",
        "--config.n_edit_samples=32",
    ]
    policy = write(
        tmp_path / "reference.json",
        {"program": "eval_droid_policy.py", "arguments": arguments},
    )
    value = {
        "schema": collect.SCHEMA,
        "split": "TRAIN",
        "mode": "original_base_evaluator",
        "seeds": seeds,
        "startup_seed": 20999,
        "reservation": reservation,
        "native_config": native,
        "task_settings": settings,
        "recording_modules": {
            name: collect.pin(importlib.util.find_spec(name).origin)
            for name in collect.MODULES
        },
        "policy_reference": policy,
        "output_dir": str(tmp_path / "capture"),
        "lineage": {
            "source_episode": 12,
            "scene_reference": 11,
            "new_independent_source_lineage": False,
        },
    }
    path = tmp_path / "DESCRIPTOR.json"
    write(path, value)
    return path, collect.load_descriptor(path)


class SyntheticEnv:
    """Only tests protocol alignment; there is no simulation or controller here."""

    def __init__(self, seeds):
        self.seeds, self.resets, self.native_controls = [20999, *seeds], 0, 0
        self.closed = False

    def reset(self):
        self.seed = self.seeds[self.resets]
        self.resets += 1
        self.step_count = 0
        self.state = np.full(14, self.resets / 100, dtype=np.float32)
        self.labels = (False, False, 0.0, 1.0)
        return self.get_observation()

    def step(self, proposed):
        self.native_controls += 1
        self.step_count += 1
        self.state += np.float32(0.01)
        length, ending = {
            29001: (2, "success"),
            29002: (3, "timeout"),
            29003: (1, "failure"),
        }[self.seed]
        done = self.step_count == length
        success = done and ending == "success"
        terminated, truncated = (
            done and ending != "timeout",
            done and ending == "timeout",
        )
        applied = np.clip(proposed, -0.2, 0.2).astype(np.float64)
        self.last_native_info = {
            "step": self.step_count,
            "action_applied": applied,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "failure": done and ending == "failure",
        }
        self.labels = (done, success, float(success), float(not terminated))
        return {"terminal_noop": False, "executed_action": applied}

    def get_info_for_step(self):
        return self.labels

    def get_observation(self):
        return {
            "state": self.state.copy(),
            "head_left": np.full(
                (360, 640, 3), self.resets * 10 + self.step_count, np.uint8
            ),
        }

    def render(self):
        return self.get_observation()["head_left"]

    def close(self):
        self.closed = True


def observe(env, index):
    observation = env.get_observation()
    integration = np.zeros(437, np.float64)
    integration[0] = env.step_count / 15
    return {
        "rgb": observation["head_left"],
        "integration": integration,
        "integration_spec": 8191,
        "public": {
            "index": index,
            "time_s": integration[0],
            "state14": observation["state"].tolist(),
        },
    }


def run_capture(descriptor):
    path, value = descriptor
    env = SyntheticEnv(value["seeds"])
    recorder = collect.TrainRecordingEnv(
        env, value, collect.pin(path), api, observe=observe
    )
    recorder.reset()
    for _ in value["seeds"]:
        recorder.reset()
        while not recorder.terminal:
            recorder.step(np.full(14, 0.7))
    recorder.close()
    return recorder, collect.pin(Path(value["output_dir"]) / "RESULT.json")


def test_real_writer_and_causal_all_outcome_export(descriptor):
    recorder, result_pin = run_capture(descriptor)
    result = collect.read(result_pin)
    rows = list(exporter.collection_rows(descriptor[1], result_pin))
    assert result["status"] == "completed_candidate"
    assert recorder.env.closed and len(rows) == 6
    assert [row[0] for row in rows] == [29001, 29001, 29002, 29002, 29002, 29003]
    assert [int(row[2]["observations"]["head_left"][0, 0, 0]) for row in rows] == [
        20,
        21,
        30,
        31,
        32,
        40,
    ]
    assert [row[2]["dones"] for row in rows] == [False, True, False, False, True, True]
    assert [row[2]["masks"] for row in rows] == [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert [row[2]["rewards"] for row in rows] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    for _, _, row in rows:
        np.testing.assert_array_equal(row["actions"], np.full(14, 0.2))
        assert row["actions"].dtype == np.float64
        assert row["observations"]["state"].dtype == np.float32
        assert row["is_hil"] is row["is_success"] is False
    fixture = os.environ.get("EXPO_COLLECTION_TEST_FIXTURE")
    if fixture:
        # Explicit synthetic fixture for a separate CPU original-serializer test.
        write(
            Path(fixture),
            {
                "descriptor": collect.pin(descriptor[0]),
                "result": result_pin,
                "synthetic_physics": True,
            },
        )


@pytest.mark.parametrize(
    "seeds",
    [
        [9100],
        [10000],
        [20001],
        [21000],
        [21004, 21004],
        [30000],
        [280001],
        [True],
        [],
        [29001, 29002, 29003, 29004, 29005],
    ],
)
def test_split_and_reserved_schedule_exclusion(seeds):
    with pytest.raises(ValueError):
        collect.validate_seeds(seeds, 20999)


def test_requires_actual_coordinator_reservation(descriptor):
    path, value = descriptor
    value["seeds"] = [21004]
    write(path, value)
    with pytest.raises(ValueError, match="reservation"):
        collect.load_descriptor(path)


def test_client_factory_preserves_original_task_fields(descriptor):
    config = collect.make_collection_task_config(descriptor[0])
    assert config.env is collect.make_collection_env
    assert list(config.episode_seeds) == [20999, 29001, 29002, 29003]
    assert (config.control_hz, config.action_horizon, config.auto_reset_steps) == (
        15,
        16,
        901,
    )
    assert config.dataset_loader == "sadhana_r1lite.expo_data:load_train_dataset"


def test_mid_episode_reset_rejected_and_evidence_retained(descriptor):
    path, value = descriptor
    recorder = collect.TrainRecordingEnv(
        SyntheticEnv(value["seeds"]), value, collect.pin(path), api, observe=observe
    )
    recorder.reset()
    recorder.reset()
    recorder.step(np.zeros(14))
    with pytest.raises(ValueError, match="unfinished"):
        recorder.reset()
    result = collect.read(collect.pin(Path(value["output_dir"]) / "RESULT.json"))
    assert result["status"] == "incomplete_or_failed" and recorder.env.closed
    assert collect.read(result["episodes"][0])["complete"] is False


def test_failure_after_native_step_is_not_fabricated_experience(descriptor):
    path, value = descriptor

    def failing_observe(env, index):
        if index:
            raise RuntimeError("camera failed after step")
        return observe(env, index)

    recorder = collect.TrainRecordingEnv(
        SyntheticEnv(value["seeds"]),
        value,
        collect.pin(path),
        api,
        observe=failing_observe,
    )
    recorder.reset()
    recorder.reset()
    with pytest.raises(RuntimeError, match="camera"):
        recorder.step(np.zeros(14))
    result_pin = collect.pin(Path(value["output_dir"]) / "RESULT.json")
    episode = collect.read(collect.read(result_pin)["episodes"][0])
    assert episode["controls"] == 1 and len(episode["files"]) == 1
    assert episode["last_unfinished_proposal"] == [0.0] * 14
    with pytest.raises(ValueError, match="Complete collection"):
        list(exporter.collection_rows(value, result_pin))


def test_refuses_before_step_when_frame_budget_exhausted(descriptor):
    path, value = descriptor
    recorder = collect.TrainRecordingEnv(
        SyntheticEnv(value["seeds"]), value, collect.pin(path), api, observe=observe
    )
    recorder.reset()
    recorder.reset()
    recorder.episode_output.max_bytes = (
        recorder.episode_output.used + recorder.episode_output.reserve
    )
    with pytest.raises(ValueError, match="Storage reserve"):
        recorder.step(np.zeros(14))
    assert recorder.env.native_controls == 0


@pytest.mark.parametrize("change", ["seed", "state_before", "mask", "applied"])
def test_export_rejects_causal_or_label_corruption(descriptor, change):
    _, result_pin = run_capture(descriptor)
    episode = collect.read(collect.read(result_pin)["episodes"][0])
    rgb, previous = exporter.frame(episode["files"][0])
    _, following = exporter.frame(episode["files"][1])
    following = copy.deepcopy(following)
    if change == "seed":
        following["seed"] += 1
    elif change == "state_before":
        following["state_before"][0] += 1
    elif change == "mask":
        following["done_success_reward_mask"][3] = 0.0
    else:
        following["applied"][0] += 1
    with pytest.raises(ValueError):
        exporter.transition(previous, following, rgb)


def test_evaluator_binding_rejects_full_expo_or_wrong_count(descriptor):
    arguments = collect.read(descriptor[1]["policy_reference"])["arguments"] + [
        "--num_episodes=3",
        "--save_video=False",
    ]
    collect.validate_evaluator_arguments(arguments, episodes=3)
    with pytest.raises(ValueError, match="num_episodes"):
        collect.validate_evaluator_arguments(arguments, episodes=4)
    with pytest.raises(ValueError, match="only_base"):
        collect.validate_evaluator_arguments(
            [
                arg.replace("--only_base_actions=True", "--only_base_actions=False")
                for arg in arguments
            ],
            episodes=3,
        )


@pytest.mark.parametrize(
    "extra", [["--only_base_actions", "False"], ["--noonly_base_actions"]]
)
def test_evaluator_binding_rejects_alternate_override(descriptor, extra):
    arguments = collect.read(descriptor[1]["policy_reference"])["arguments"] + [
        "--num_episodes=3",
        "--save_video=False",
        *extra,
    ]
    with pytest.raises(ValueError, match="only_base"):
        collect.validate_evaluator_arguments(arguments, episodes=3)
