"""Small explicit CPU artifact doubles; these are not robot task results."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from abc_bench.demo_cohort import (
    CHANNELS,
    EPISODE_SCHEMA,
    publish_cohort,
    validate_episode,
)
from abc_bench.upstream_eval import artifact


def write_json(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return artifact(path)


def write_array(root: Path, name: str, value: np.ndarray) -> dict:
    path = root / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    return {**artifact(path), "shape": list(value.shape), "dtype": value.dtype.name}


def episode_fixture(root: Path, split: str, seed: int) -> tuple[dict, dict]:
    root.mkdir(parents=True)
    count = 16
    times = np.arange(count + 1, dtype=np.float64) * 0.034
    states = np.zeros((count + 1, 14), dtype=np.float32)
    states[:, 0] = seed / 10000
    actions = states[:-1].copy()
    # Preserve finite original policy commands outside physical gripper limits.
    actions[:, 6] = -0.16
    actions[:, 13] = 1.1
    reset = {"seed": seed, "metadata": {"bottle_count": 5}}
    final = {"success": True, "num_bottles_in_bin": 5, "num_active_bottles": 5}
    files = {
        "states": write_array(root, "states", states),
        "issued_actions": write_array(root, "issued_actions", actions),
        "sim_times": write_array(root, "sim_times", times),
        "qpos": write_array(root, "qpos", np.zeros((count + 1, 20), np.float64)),
        "qvel": write_array(root, "qvel", np.zeros((count + 1, 18), np.float64)),
        "actuator_ctrl": write_array(
            root, "actuator_ctrl", np.zeros((count, 14), np.float64)
        ),
        "initial_integration": write_array(
            root, "integration", np.array([seed], np.float64)
        ),
        "randomization": write_json(root / "randomization.json", reset),
    }
    rgb = root / "rgb24.bin"
    rgb.write_bytes(bytes((count + 1) * 168 * 672 * 3))
    files["rgb24"] = {
        **artifact(rgb),
        "shape": [count + 1, 168, 672, 3],
        "dtype": "uint8",
    }
    scene = root / "scene.xml"
    scene.write_text(f'<mujoco model="explicit-cpu-double-{seed}"/>')
    files["scene_xml"] = artifact(scene)
    judges = root / "task_judge.jsonl"
    judges.write_text(
        "".join(
            json.dumps(
                {
                    "control_index": i + 1,
                    "sim_time": float(times[i + 1]),
                    "task_eval": final,
                }
            )
            + "\n"
            for i in range(count)
        )
    )
    files["task_judge"] = artifact(judges)
    compiled = root / "scene.mjb"
    compiled.write_bytes(b"Explicit CPU model identity double")
    files["compiled_model"] = artifact(compiled)
    request_images = root / "request_rgb24.bin"
    request_images.write_bytes(bytes(2 * 168 * 672 * 3))
    files["request_rgb24"] = {
        **artifact(request_images),
        "shape": [2, 168, 672, 3],
        "dtype": "uint8",
    }
    requests = []
    sampled = np.tile(actions[0], (30, 1))
    for index, observed in enumerate((0, 11)):
        prompt = "sim throw plastic bottles in bin"
        fingerprint = hashlib.sha256(states[observed].tobytes())
        fingerprint.update(bytes(168 * 672 * 3))
        fingerprint.update(prompt.encode())
        requests.append(
            {
                "inference_id": index + 1,
                "observation": {
                    "control_index": observed,
                    "sim_time": float(times[observed]),
                    "fingerprint": fingerprint.hexdigest(),
                },
                "prompt": prompt,
                "state": states[observed].tolist(),
                "actions": sampled.tolist(),
                "image_frame_index": index,
                "image_byte_offset": index * 168 * 672 * 3,
                "image_shape_hwc": [168, 672, 3],
                "image_dtype": "uint8",
                "camera_names": ["top", "left", "right"],
                "prefix_length": 0 if index == 0 else 4,
            }
        )
    trace = root / "policy_requests.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in requests))
    files["policy_requests"] = artifact(trace)
    trace = root / "action_provenance.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "control_index": i + 1,
                    "inference_id": 1 if i < 15 else 2,
                    "inferred_action_index": i if i < 15 else 4,
                    "observation_control_index": 0 if i < 15 else 11,
                    "prefix_length": 0 if i < 15 else 4,
                }
            )
            + "\n"
            for i in range(count)
        )
    )
    files["action_provenance"] = artifact(trace)
    config = {
        "split": split,
        "checkpoint": {"fixture": "not a model"},
        "metadata": {"fixture": True},
    }
    config_pin = write_json(root / "config.json", config)
    producer = {
        "config": config_pin,
        "checkpoint": config["checkpoint"],
        "metadata": config["metadata"],
        "software": write_json(
            root / "software.json", {"fixture": "no native runtime"}
        ),
    }
    group = hashlib.sha256(
        (
            files["scene_xml"]["sha256"] + ":" + files["initial_integration"]["sha256"]
        ).encode()
    ).hexdigest()
    episode = {
        "schema": EPISODE_SCHEMA,
        "episode_id": f"explicit-{seed}",
        "world_index": 0,
        "controls": count,
        "frame_count": count + 1,
        "split": split,
        "seed": seed,
        "requested_seed": seed,
        "reset_group_id": group,
        "task": "put_plastic_bottles_in_bin",
        "prompt": "sim throw plastic bottles in bin",
        "policy_channels": CHANNELS,
        "camera_names": ["top", "left", "right"],
        "image_shape_hwc": [168, 224, 3],
        "producer": producer,
        "physics": {"dt": 0.002, "decimation": 17, "control_hz": 1 / 0.034},
        "artifacts": files,
        "final_task_eval": final,
        "admission": {
            "complete_current_success": True,
            "active_bottles": 5,
            "current_bottles_in_bin": 5,
            "terminated": True,
            "truncated": False,
        },
        "human_expert_approved": False,
        "atlas_approved": False,
        "dynamics_validated": False,
    }
    world = {
        "world_index": 0,
        "world_seed": seed,
        "steps": count,
        "final_success": True,
        "final_task_eval": final,
        "randomization": reset,
    }
    return episode, world


def seal(run: Path, episode: dict, world: dict, *, admitted: bool = True) -> None:
    entry = {
        "manifest": write_json(run / "episode" / "manifest.json", episode),
        "admitted": admitted,
    }
    summary = write_json(run / "summary.json", {"worlds": [world]})
    worker = write_json(
        run / "receipt.json",
        {
            "schema_version": 1,
            "status": "completed",
            "split": episode["split"],
            "phase": "original_policy_demonstration_collection",
            "input_config": episode["producer"]["config"],
            "checkpoint": episode["producer"]["checkpoint"],
            "metadata": episode["producer"]["metadata"],
            "software": episode["producer"]["software"],
            "original_summary": summary,
            "episodes": [entry],
        },
    )
    job = write_json(run / "job.json", {"module": "abc_bench.demo_collect"})
    write_json(
        run.parent / "runs" / f"{run.name}.json",
        {
            "schema_version": 1,
            "run_id": run.name,
            "status": "completed",
            "worker_exit_code": 0,
            "task": "put_plastic_bottles_in_bin",
            "embodiment": "native_yam",
            "resolved_job_sha256": job["sha256"],
            "worker_receipt_sha256": worker["sha256"],
        },
    )


@pytest.fixture
def data(tmp_path):
    runs = [tmp_path / "bench" / split for split in ("train", "validation")]
    episodes = [
        episode_fixture(run / "episode", split, seed)
        for run, split, seed in zip(
            runs, ("train", "validation"), (9030, 9100), strict=True
        )
    ]
    for run, (episode, world) in zip(runs, episodes, strict=True):
        seal(run, episode, world)
    return runs, episodes, tmp_path / "cohort.json"


def test_publishes_authenticated_whole_episodes_without_changing_commands(data):
    runs, episodes, output = data
    cohort = publish_cohort(runs, output)
    assert cohort["splits"] == {
        "train": ["explicit-9030"],
        "validation": ["explicit-9100"],
        "development": [],
    }
    assert json.loads(output.read_text()) == cohort
    assert all(entry["parent_receipt"]["sha256"] for entry in cohort["episodes"])
    commands = np.load(episodes[0][0]["artifacts"]["issued_actions"]["path"])
    assert commands[0, 6] == np.float32(-0.16)
    assert commands[0, 13] == np.float32(1.1)
    with pytest.raises(FileExistsError):
        publish_cohort(runs, output)


@pytest.mark.parametrize("mutation", ["worker", "pending", "job"])
def test_requires_actual_completed_parent_seal(data, mutation):
    runs, _, output = data
    if mutation in {"worker", "job"}:
        (runs[0] / ("receipt.json" if mutation == "worker" else "job.json")).write_text(
            "{}"
        )
    else:
        parent_path = runs[0].parent / "runs" / "train.json"
        parent = json.loads(parent_path.read_text())
        parent["status"] = "running"
        write_json(parent_path, parent)
    with pytest.raises(ValueError):
        publish_cohort(runs, output)
    assert not output.exists()


def test_failed_rollout_cannot_supply_training_data(data):
    runs, episodes, output = data
    episode, world = episodes[0]
    seal(runs[0], episode, world, admitted=False)
    with pytest.raises(ValueError, match="disjoint TRAIN"):
        publish_cohort(runs, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "source_final",
        "terminal_frame",
        "reset_group",
        "development",
        "approval",
        "dtype",
        "clock",
    ],
)
def test_rejects_false_admission_or_misaligned_numeric_contract(data, mutation):
    runs, episodes, _ = data
    episode, world = episodes[0]
    if mutation == "source_final":
        world["final_success"] = False
    elif mutation == "terminal_frame":
        episode["frame_count"] = episode["controls"]
    elif mutation == "reset_group":
        episode["reset_group_id"] = "0" * 64
    elif mutation == "development":
        episode["requested_seed"] = 20260511
    elif mutation == "approval":
        episode["atlas_approved"] = True
    elif mutation == "dtype":
        episode["artifacts"]["sim_times"].pop("dtype")
    elif mutation == "clock":
        times = np.arange(17, dtype=np.float64) * 0.04
        episode["artifacts"]["sim_times"] = write_array(
            runs[0] / "episode", "sim_times", times
        )
    pin = write_json(runs[0] / "episode" / "manifest.json", episode)
    with pytest.raises(ValueError):
        validate_episode(pin, world, "train")


def test_copied_trajectory_cannot_cross_splits(data):
    runs, episodes, output = data
    validation, world = episodes[1]
    for name in (
        "states",
        "issued_actions",
        "rgb24",
        "policy_requests",
        "request_rgb24",
        "action_provenance",
    ):
        validation["artifacts"][name] = episodes[0][0]["artifacts"][name]
    seal(runs[1], validation, world)
    with pytest.raises(ValueError, match="leak"):
        publish_cohort(runs, output)
    assert not output.exists()


def test_unsealed_array_changes_are_rejected(data):
    runs, episodes, output = data
    path = Path(episodes[0][0]["artifacts"]["issued_actions"]["path"])
    actions = np.load(path)
    actions[0, 0] += 1
    np.save(path, actions, allow_pickle=False)
    with pytest.raises(ValueError, match="producer pin"):
        publish_cohort(runs, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_model",
        "missing_requests",
        "wrong_action",
        "future_observation",
        "changed_pixels",
        "rtc_schedule",
    ],
)
def test_sampler_and_model_audit_artifacts_must_be_real_and_aligned(data, mutation):
    runs, episodes, output = data
    episode, world = episodes[0]
    if mutation in {"missing_model", "missing_requests"}:
        name = "compiled_model" if mutation == "missing_model" else "policy_requests"
        Path(episode["artifacts"][name]["path"]).unlink()
    else:
        name = (
            "action_provenance"
            if mutation == "future_observation"
            else "policy_requests"
        )
        path = Path(episode["artifacts"][name]["path"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "future_observation":
            rows[0]["observation_control_index"] = 11
        elif mutation == "rtc_schedule":
            rows[1]["observation"]["control_index"] = 0
            rows[1]["observation"]["sim_time"] = 0.0
        elif mutation == "changed_pixels":
            rgb = Path(episode["artifacts"]["request_rgb24"]["path"])
            with rgb.open("r+b") as stream:
                stream.write(b"\xff")
            old = episode["artifacts"]["request_rgb24"]
            episode["artifacts"]["request_rgb24"] = {**old, **artifact(rgb)}
        else:
            rows[0]["actions"][0][0] += 0.1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        episode["artifacts"][name] = artifact(path)
        seal(runs[0], episode, world)
    with pytest.raises(ValueError):
        publish_cohort(runs, output)
    assert not output.exists()
