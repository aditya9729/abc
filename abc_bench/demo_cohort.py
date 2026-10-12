"""Publish a task-SFT cohort from completed, coordinated policy demonstrations.

Failed or partial rollouts stay outside the cohort. Development episodes remain
visible for provenance but cannot supply TRAIN or validation examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from abc_bench.upstream_eval import artifact
from abc_bench.upstream_report import coordinated_receipt

EPISODE_SCHEMA = "sadhana.abc.policy-demonstration/v1"
COHORT_SCHEMA = "sadhana.abc.demonstration-cohort/v1"
CHANNELS = (
    [f"left_j{i}" for i in range(1, 7)]
    + ["left_grip"]
    + [f"right_j{i}" for i in range(1, 7)]
    + ["right_grip"]
)
SFT_PROMPT = "put plastic bottles in bin"


def verified_pin(pin: dict[str, Any]) -> Path:
    """Hash a producer artifact in bounded blocks, including raw RGB streams."""
    if not isinstance(pin, dict) or not {"path", "bytes", "sha256"} <= pin.keys():
        raise ValueError("An artifact requires path, bytes and SHA256")
    if type(pin["path"]) is not str or type(pin["sha256"]) is not str:
        raise ValueError("Artifact path and SHA256 must be strings")
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Artifacts must be existing absolute files")
    if type(pin["bytes"]) is not int or pin["bytes"] < 0:
        raise ValueError("Artifact bytes must be a nonnegative integer")
    identity = artifact(path)
    if any(identity[key] != pin[key] for key in ("path", "bytes", "sha256")):
        raise ValueError(f"Artifact differs from its producer pin: {path}")
    return path


def pinned_json(pin: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(pin, dict)
        or type(pin.get("bytes")) is not int
        or not 0 < pin["bytes"] <= 1024**2
    ):
        raise ValueError("A JSON manifest must fit the 1 MiB boundary")
    value = json.loads(verified_pin(pin).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("A JSON manifest must contain an object")
    return value


def _array(pin: dict[str, Any], shape: tuple[int, ...], dtype: str):
    import numpy as np

    path = verified_pin(pin)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        array.shape != shape
        or pin.get("shape") != list(shape)
        or array.dtype != np.dtype(dtype)
        or type(pin.get("dtype")) is not str
        or np.dtype(pin["dtype"]) != np.dtype(dtype)
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"Array shape, dtype or finite-coordinate check failed: {path}"
        )
    return array


def _jsonl(pin: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
    if (
        not isinstance(pin, dict)
        or type(pin.get("bytes")) is not int
        or pin["bytes"] > maximum
    ):
        raise ValueError("Trace JSONL exceeds its bounded control-horizon size")
    with verified_pin(pin).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if any(type(row) is not dict for row in rows):
        raise ValueError("Trace JSONL rows must be objects")
    return rows


def _validate_sampler_trace(
    files: dict[str, Any], states, actions, times, prompt: str
) -> None:
    """Bind commands to genuine sampler rows; RTC observations can be earlier."""
    import numpy as np

    requests = _jsonl(files["policy_requests"], 16 * 1024**2)
    controls = _jsonl(files["action_provenance"], 2 * 1024**2)
    count = len(actions)
    if (
        not 1 <= len(requests) <= min(236, math.ceil(count / 15) + 1)
        or len(controls) != count
    ):
        raise ValueError("Sampler and issued-control trace counts differ")
    rgb = files["request_rgb24"]
    rgb_path = verified_pin(rgb)
    if (
        rgb.get("shape") != [len(requests), 168, 672, 3]
        or rgb.get("dtype") != "uint8"
        or rgb["bytes"] != len(requests) * 168 * 672 * 3
    ):
        raise ValueError("Exact sampler RGB frames differ from request count")
    inferred = []
    observation_controls = []
    for i, row in enumerate(requests):
        observation = row.get("observation", {})
        observed = observation.get("control_index")
        if (
            row.get("inference_id") != i + 1
            or type(observed) is not int
            or observed != (0 if i == 0 else 11 + 15 * (i - 1))
            or not 0 <= observed < count
            or observation.get("sim_time") != float(times[observed])
            or row.get("image_frame_index") != i
            or row.get("image_byte_offset") != i * 168 * 672 * 3
            or row.get("image_shape_hwc") != [168, 672, 3]
            or row.get("image_dtype") != "uint8"
            or row.get("camera_names") != ["top", "left", "right"]
            or row.get("prompt") != prompt
            or row.get("prefix_length") != (0 if i == 0 else 4)
            or not np.array_equal(
                np.asarray(row.get("state"), dtype=np.float32), states[observed]
            )
        ):
            raise ValueError("Sampler observation, RTC prefix or RGB index differs")
        with rgb_path.open("rb") as stream:
            stream.seek(i * 168 * 672 * 3)
            pixels = stream.read(168 * 672 * 3)
        fingerprint = hashlib.sha256(np.asarray(row["state"], dtype="<f4").tobytes())
        fingerprint.update(pixels)
        fingerprint.update(prompt.encode("utf-8"))
        if observation.get("fingerprint") != fingerprint.hexdigest():
            raise ValueError(
                "Sampler RGB/state/prompt differ from the recorded observation"
            )
        values = np.asarray(row.get("actions"), dtype=np.float32)
        if values.shape != (30, 14) or not np.isfinite(values).all():
            raise ValueError("Original sampler output must be finite float32[30,14]")
        inferred.append(values)
        observation_controls.append(observed)
    for i, row in enumerate(controls):
        request_id = row.get("inference_id")
        index = row.get("inferred_action_index")
        if (
            row.get("control_index") != i + 1
            or type(request_id) is not int
            or not 1 <= request_id <= len(requests)
            or request_id != i // 15 + 1
            or type(index) is not int
            or index != i % 15 + (0 if request_id == 1 else 4)
        ):
            raise ValueError("Issued action points outside the recorded sampler trace")
        request = requests[request_id - 1]
        if (
            row.get("observation_control_index") != observation_controls[request_id - 1]
            or row["observation_control_index"] > i
            or row.get("prefix_length") != request["prefix_length"]
            or not np.array_equal(actions[i], inferred[request_id - 1][index])
        ):
            raise ValueError("Issued action differs from its genuine sampler output")


def validate_episode(
    pin: dict[str, Any], world: dict[str, Any], split: str
) -> dict[str, Any]:
    """Verify observed chronology and current success without changing labels."""
    import numpy as np

    episode = pinned_json(pin)
    if type(world) is not dict:
        raise ValueError("An original world summary must be an object")
    count = episode.get("controls")
    if (
        episode.get("schema") != EPISODE_SCHEMA
        or type(count) is not int
        or not 16 <= count <= 3540
        or episode.get("frame_count") != count + 1
        or episode.get("split") != split
        or episode.get("task") != "put_plastic_bottles_in_bin"
        or episode.get("policy_channels") != CHANNELS
        or episode.get("camera_names") != ["top", "left", "right"]
        or episode.get("image_shape_hwc") != [168, 224, 3]
        or type(episode.get("episode_id")) is not str
        or not episode["episode_id"]
    ):
        raise ValueError("Episode schema, split or control contract differs")
    seed = episode.get("seed")
    requested = episode.get("requested_seed")
    if type(seed) is not int or type(requested) is not int:
        raise ValueError("Episode seeds must be integers")
    if seed not in {requested, requested + 100000}:
        raise ValueError("Actual seed differs from the original reset or fallback")
    heldout = set(range(9100, 9103)) | set(range(109100, 109103))
    if split == "validation" and requested not in range(9100, 9103):
        raise ValueError("Validation must use its declared requested scene family")
    if split == "train" and (seed in heldout or requested in heldout):
        raise ValueError("TRAIN cannot use validation scenes or their fallback family")
    if split != "development" and {20260511, 20360511} & {seed, requested}:
        raise ValueError("The reused development scene cannot supply task data")
    if split == "development" and (seed != 20260511 or requested != seed):
        raise ValueError("The development reference must preserve its exact scene")
    admission = episode.get("admission")
    if (
        not isinstance(admission, dict)
        or admission.get("complete_current_success") is not True
        or type(admission.get("active_bottles")) is not int
        or admission["active_bottles"] != 5
        or type(admission.get("current_bottles_in_bin")) is not int
        or admission["current_bottles_in_bin"] != 5
        or admission.get("terminated") is not True
        or admission.get("truncated") is not False
    ):
        raise ValueError("Only complete current five-bottle successes are admitted")
    for flag in ("human_expert_approved", "atlas_approved", "dynamics_validated"):
        if episode.get(flag) is not False:
            raise ValueError("Automatic collection cannot claim external approval")
    final = episode.get("final_task_eval")
    if (
        world.get("world_seed") != seed
        or world.get("world_index") != episode.get("world_index")
        or world.get("steps") != count
        or world.get("final_success") is not True
        or not isinstance(final, dict)
        or final != world.get("final_task_eval")
        or final.get("success") is not True
        or final.get("num_bottles_in_bin") != 5
        or final.get("num_active_bottles") != 5
    ):
        raise ValueError("Episode terminal judge differs from the original summary")
    physics = episode.get("physics", {})
    if (
        physics.get("dt") != 0.002
        or physics.get("decimation") != 17
        or type(physics.get("control_hz")) not in (float, int)
        or not math.isclose(physics["control_hz"], 1 / 0.034, rel_tol=1e-12)
    ):
        raise ValueError("Episode must record the native 0.034-second control period")
    producer = episode.get("producer")
    if (
        not isinstance(producer, dict)
        or not {"config", "checkpoint", "metadata", "software"} <= producer.keys()
    ):
        raise ValueError("An episode must record noncircular producer lineage")
    # Checkpoint weights are authenticated once by the collection worker. The
    # lineage records their identity without rereading 8.8 GB for every episode.
    config = pinned_json(producer["config"])
    pinned_json(producer["software"])
    if (
        config.get("split") != split
        or producer["checkpoint"] != config.get("checkpoint")
        or producer["metadata"] != config.get("metadata")
    ):
        raise ValueError("Episode producer differs from its collection config")
    files = episode["artifacts"]
    states = _array(files["states"], (count + 1, 14), "float32")
    actions = _array(files["issued_actions"], (count, 14), "float32")
    times = _array(files["sim_times"], (count + 1,), "float64")
    if not np.allclose(np.diff(times), 0.034, rtol=0, atol=1e-9):
        raise ValueError("Observation times do not advance once per native control")
    rgb = files["rgb24"]
    verified_pin(rgb)
    if (
        rgb.get("shape") != [count + 1, 168, 672, 3]
        or rgb.get("dtype") != "uint8"
        or rgb["bytes"] != (count + 1) * 168 * 672 * 3
    ):
        raise ValueError("RGB stream must have one genuine terminal frame")
    for name in ("qpos", "qvel", "actuator_ctrl"):
        item = files[name]
        shape = item.get("shape")
        rows = count if name == "actuator_ctrl" else count + 1
        if (
            type(shape) is not list
            or len(shape) != 2
            or shape[0] != rows
            or type(shape[1]) is not int
            or not 0 < shape[1] <= 10000
        ):
            raise ValueError(f"Native {name} shape differs")
        _array(item, tuple(shape), "float64")
    reset = pinned_json(files["randomization"])
    if reset != world.get("randomization"):
        raise ValueError("Captured reset differs from the original world")
    verified_pin(files["scene_xml"])
    integration_shape = files["initial_integration"].get("shape")
    if (
        type(integration_shape) is not list
        or len(integration_shape) != 1
        or type(integration_shape[0]) is not int
        or not 0 < integration_shape[0] <= 100000
    ):
        raise ValueError("Initial integration must be a native state vector")
    _array(files["initial_integration"], tuple(integration_shape), "float64")
    group = episode.get("reset_group_id")
    expected_group = hashlib.sha256(
        (
            files["scene_xml"]["sha256"] + ":" + files["initial_integration"]["sha256"]
        ).encode("ascii")
    ).hexdigest()
    if group != expected_group:
        raise ValueError(
            "Reset group must bind the actual scene and initial integration"
        )
    verified_pin(files["compiled_model"])
    _validate_sampler_trace(files, states, actions, times, episode["prompt"])
    rows = _jsonl(files["task_judge"], 4 * 1024**2)
    if (
        len(rows) != count
        or any(row.get("control_index") != i + 1 for i, row in enumerate(rows))
        or rows[-1].get("task_eval") != final
        or any(row.get("sim_time") != float(times[i + 1]) for i, row in enumerate(rows))
    ):
        raise ValueError("Source judge controls or final result differ")
    return episode


def publish_cohort(
    runs: list[Path], output: Path, *, sft_task_prompt: str = SFT_PROMPT
) -> dict[str, Any]:
    """Publish only after real worker seals and whole-episode splits pass."""
    if not runs or len(runs) > 32 or len({p.resolve() for p in runs}) != len(runs):
        raise ValueError("Provide 1 to 32 distinct completed collection runs")
    if not isinstance(sft_task_prompt, str) or not sft_task_prompt.strip():
        raise ValueError("A declared SFT task instruction is required")
    if output.exists():
        raise FileExistsError("Cohort output already exists; publish a new version")
    cohort: dict[str, Any] = {
        "schema": COHORT_SCHEMA,
        "sft_task_prompt": sft_task_prompt,
        "splits": {name: [] for name in ("train", "validation", "development")},
        "episodes": [],
        "excluded": [],
        "claim_scope": "Simulator policy demonstrations; no human or Atlas approval",
    }
    ids: set[str] = set()
    seeds: dict[int, str] = {}
    groups: set[str] = set()
    paths: set[str] = set()
    contents: set[tuple[str, ...]] = set()
    for run in runs:
        run = run.resolve()
        admitted = coordinated_receipt(run)
        if admitted is None:
            raise ValueError(f"Collection coordinator is not completed: {run}")
        worker, worker_pin, parent = admitted
        job = json.loads((run / "job.json").read_text(encoding="utf-8"))
        if job.get("module") != "abc_bench.demo_collect":
            raise ValueError(
                "Only the admitted original-policy collector may produce a cohort"
            )
        if worker.get("phase") != "original_policy_demonstration_collection":
            raise ValueError("Worker is not a demonstration collection stage")
        config = pinned_json(worker["input_config"])
        summary = pinned_json(worker["original_summary"])
        worlds = summary.get("worlds", [])
        entries = worker.get("episodes", [])
        if not entries or len(entries) != len(worlds):
            raise ValueError("Collection receipt and original world count differ")
        parent_pin = artifact(run.parent / "runs" / (parent["run_id"] + ".json"))
        for entry, world in zip(entries, worlds, strict=True):
            pin = entry["manifest"]
            episode = pinned_json(pin)
            split = episode.get("split")
            if (
                split not in cohort["splits"]
                or split != worker.get("split")
                or split != config.get("split")
            ):
                raise ValueError("Collection split is unavailable")
            if entry.get("admitted") is not True:
                cohort["excluded"].append(
                    {
                        "manifest": pin,
                        "reason": "Not a complete current five-bottle success",
                    }
                )
                continue
            episode = validate_episode(pin, world, split)
            expected_producer = {
                "config": worker["input_config"],
                "checkpoint": worker["checkpoint"],
                "metadata": worker["metadata"],
                "software": worker["software"],
            }
            if episode["producer"] != expected_producer:
                raise ValueError(
                    "Episode producer lineage differs from completed worker"
                )
            episode_id = episode["episode_id"]
            seed = episode["seed"]
            group = episode["reset_group_id"]
            trajectory = [
                episode["artifacts"][k] for k in ("states", "issued_actions", "rgb24")
            ]
            signature = tuple(p["sha256"] for p in trajectory)
            if (
                episode_id in ids
                or seed in seeds
                or group in groups
                or any(p["path"] in paths for p in trajectory)
                or signature in contents
            ):
                raise ValueError(
                    "Duplicate episode, reset, seed or trajectory would leak across splits"
                )
            ids.add(episode_id)
            seeds[seed] = split
            groups.add(group)
            paths.update(p["path"] for p in trajectory)
            contents.add(signature)
            cohort["splits"][split].append(episode_id)
            cohort["episodes"].append(
                {
                    "manifest": pin,
                    "worker_receipt": worker_pin,
                    "parent_receipt": parent_pin,
                }
            )
    if not cohort["splits"]["train"] or not cohort["splits"]["validation"]:
        raise ValueError(
            "A task cohort needs successful, disjoint TRAIN and validation episodes"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(cohort, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)
    return cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft-task-prompt", default=SFT_PROMPT)
    args = parser.parse_args()
    publish_cohort(args.runs, args.output, sft_task_prompt=args.sft_task_prompt)


if __name__ == "__main__":
    main()
