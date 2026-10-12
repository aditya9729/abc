"""Publish pinned native measurements; incomplete evaluations have no score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from abc_bench.subset_report import copy_video

# Reviewed adapter 0.1.2 evaluated with FP32 tensors and no author autocast.
# Bind that historical disclosure to its sealed source, not a mutable install.
LEGACY_FP32_LEARNER_SHA256 = (
    "fe3881d2ac5e3371b83bed7da26840dd281222e11696ac262363d1973478ff41"
)


def artifact_bytes(pin: dict) -> bytes:
    """Verify a producer's file identity before consuming its contents."""
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Recorded artifacts must be existing absolute files")
    if type(pin.get("bytes")) is not int or pin["bytes"] < 0:
        raise ValueError("Recorded artifact length must be a nonnegative integer")
    data = path.read_bytes()
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError(f"Recorded artifact differs: {path}")
    return data


def _identity(path: Path, data: bytes) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _json(data: bytes) -> dict:
    value = json.loads(data)

    def finite(item: Any) -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError("Recorded JSON numbers must be finite")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    if type(value) is not dict:
        raise ValueError("Recorded JSON must be an object")
    return value


def pinned_json(pin: dict) -> dict:
    return _json(artifact_bytes(pin))


def _integer(value: Any, name: str, minimum: int = 0, maximum: int = 1000000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}")
    return value


def _number(value: Any, name: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid {name}")
    return float(value)


def _pending(status: str = "pending") -> dict:
    return {"status": status, "successes": None, "episodes": None, "video": None}


def coordinated_receipt(
    run: Path, *, nested: bool = False
) -> tuple[dict, dict, dict] | None:
    """Authenticate a completed worker against the canonical coordinator seal.

    Coordinator receipts lack a declared worker byte count. The verified digest
    binds its complete bytes; this reporter records the measured length too.
    """
    run = run.resolve()
    parent = run.parent if nested and run.name == "expoft" else run
    worker_path = (parent / "expoft" if nested else parent) / "receipt.json"
    coordinator_path = parent.parent / "runs" / (parent.name + ".json")
    if not coordinator_path.is_file():
        return None
    coordinator = _json(coordinator_path.read_bytes())
    if (
        coordinator.get("schema_version") != 1
        or coordinator.get("run_id") != parent.name
    ):
        raise ValueError("Coordinator run identity differs")
    if coordinator.get("status") != "completed":
        return None
    if (
        coordinator.get("worker_exit_code") != 0
        or coordinator.get("task") != "put_plastic_bottles_in_bin"
        or coordinator.get("embodiment") != "native_yam"
    ):
        raise ValueError("Coordinator did not complete the native ABC job")
    job_bytes = (parent / "job.json").read_bytes()
    if hashlib.sha256(job_bytes).hexdigest() != coordinator.get("resolved_job_sha256"):
        raise ValueError("Coordinator job differs from its pin")
    data = worker_path.read_bytes()
    identity = _identity(worker_path, data)
    if identity["sha256"] != coordinator.get("worker_receipt_sha256"):
        raise ValueError("Worker receipt differs from its coordinator pin")
    worker = _json(data)
    if worker.get("schema_version") != 1 or worker.get("status") != "completed":
        raise ValueError("Worker did not complete its declared stage")
    return worker, identity, coordinator


def _baseline(run: Path) -> tuple[dict, dict, dict]:
    admitted = coordinated_receipt(run)
    if admitted is None:
        raise ValueError("A completed original ABC-VLA qualification is required")
    worker, identity, _ = admitted
    if (
        worker.get("algorithm") != "original ABC-VLA"
        or worker.get("phase") != "original_recipe_one_world_qualification"
    ):
        raise ValueError("A completed original ABC-VLA qualification is required")
    summary = pinned_json(worker["original_summary"])
    worlds = summary["worlds"]
    if (
        summary.get("task") != "put_plastic_bottles_in_bin"
        or summary["num_worlds"] != 1
        or len(worlds) != 1
    ):
        raise ValueError("This qualification viewer requires one complete world")
    world = worlds[0]
    success = world["success"]
    if type(success) is not bool or summary["num_success"] != int(success):
        raise ValueError("Original success count and world outcome disagree")
    if type(world["steps"]) is not int or not 0 <= world["steps"] <= 3540:
        raise ValueError("Invalid completed control count")
    video_pin = worker["original_videos"][0]
    if video_pin["path"] != world["video_path"]:
        raise ValueError("Original summary and video identity disagree")
    artifact_bytes(video_pin)
    return worker, identity, summary


def _fasttd3(run: Path | None, *, stage: str) -> tuple[dict, dict, dict, dict] | None:
    if run is None:
        return None
    admitted = coordinated_receipt(run)
    if admitted is None:
        return None
    worker, identity, coordinator = admitted
    if (
        worker.get("algorithm") != "original FastTD3 from scratch"
        or worker.get("phase") != stage
    ):
        raise ValueError("FastTD3 receipt has a different method or stage")
    learner = pinned_json(worker["learner_receipt"])
    host = pinned_json(worker["host_receipt"])
    if any(
        value.get("schema_version") != 1 or value.get("status") != "completed"
        for value in (learner, host)
    ):
        raise ValueError("FastTD3 child receipts must complete")
    if learner.get("condition") != "original_fasttd3_scratch_privileged_state_abc":
        raise ValueError("FastTD3 learner condition differs")
    config = pinned_json(worker["resolved_config"])
    if config.get("stage") != stage:
        raise ValueError("FastTD3 resolved configuration stage differs")
    pinned_json(worker["host_description"])
    # These are historical inventories. Verify their seals, not mutable installed files.
    pinned_json(worker["learner_module_snapshot"])
    pinned_json(host["original_sources_assets"])
    if learner["environment"].get("privileged_state") is not True:
        raise ValueError("FastTD3 must identify its privileged state inputs")
    if worker["metrics"] != coordinator["metrics"]:
        raise ValueError("FastTD3 coordinator and worker metrics disagree")
    return worker, identity, learner, host


def _training(
    admitted: tuple[dict, dict, dict, dict] | None, baseline_seed: int
) -> dict:
    if admitted is None:
        return _pending()
    worker, identity, learner, host = admitted
    controls = _integer(learner["controls"], "training controls", 1)
    actor = _integer(learner["actual_updates"]["actor"], "actor updates", 1)
    critic = _integer(learner["actual_updates"]["critic"], "critic updates", 1)
    if (
        worker["metrics"]
        != {"controls": controls, "actor_updates": actor, "critic_updates": critic}
        or host["metrics"]["controls"] != controls
        or sum(_integer(ep["controls"], "episode controls") for ep in host["episodes"])
        != controls
    ):
        raise ValueError("FastTD3 training counters disagree")
    if any(ep["seed"] == baseline_seed for ep in host["episodes"]):
        raise ValueError("FastTD3 training includes the matched evaluation scene")
    checkpoint = learner["checkpoint"]
    artifact_bytes(checkpoint)
    randomization = learner["environment"].get("reset_options", {}).get("randomization")
    if randomization is not None and (
        type(randomization) is not dict
        or any(
            type(randomization.get(name)) is not bool
            for name in ("randomize_variants", "randomize_scales")
        )
    ):
        raise ValueError("Invalid FastTD3 TRAIN randomization settings")
    episodes = []
    for episode in host["episodes"]:
        video = episode.get("video")
        if video is not None:
            artifact_bytes(video)
        episodes.append(
            {
                "seed": episode["seed"],
                "controls": episode["controls"],
                "completed": episode["completed"],
                "max_native_reward": _number(
                    episode["max_native_reward"], "training reward"
                ),
                "stop_reason": episode["end_reason"],
                "video_identity": video,
                "scene_metadata": episode.get("randomization", {}).get("metadata"),
                "scale_states": episode.get("randomization", {}).get("scale_states"),
            }
        )
    return {
        "status": "completed",
        "controls": controls,
        "actor_updates": actor,
        "critic_updates": critic,
        "elapsed_seconds": _number(worker["elapsed_seconds"], "training elapsed time"),
        "action_clipping_count": _integer(
            host["metrics"]["normalized_action_clipping_count"],
            "clipped action components",
        ),
        "checkpoint_identity": checkpoint,
        "scene_randomization": randomization,
        "worker_receipt_identity": identity,
        "episodes": episodes,
        "successes": None,
        "claim_scope": "Training rollout; no frozen task score or convergence claim",
    }


def _evaluation_precision(worker: dict, learner: dict) -> dict:
    profile = learner.get("evaluation_precision")
    if profile is None:
        inventory = pinned_json(worker["learner_module_snapshot"])
        if not any(
            module.get("sha256") == LEGACY_FP32_LEARNER_SHA256
            and Path(module.get("path", ""))
            .as_posix()
            .endswith("/sadhana_fasttd3_abc/learner.py")
            for module in inventory.get("modules", [])
        ):
            raise ValueError(
                "Evaluation precision is unrecorded for an unknown evaluator"
            )
        return {
            "label": "FP32 legacy diagnostic; author AMP evaluation was not used",
            "autocast_enabled": False,
            "policy_parameter_dtype": ["torch.float32"],
            "policy_action_dtype": ["torch.float32"],
            "environment_action_dtype": "torch.float32",
            "source_basis": "Reviewed historical learner source SHA256",
            "learner_sha256": LEGACY_FP32_LEARNER_SHA256,
        }
    if type(profile) is not dict:
        raise ValueError("Invalid evaluation precision record")
    for name in (
        "autocast_enabled",
        "configured_amp",
        "configured_cuda",
        "cuda_matmul_allow_tf32",
        "cudnn_deterministic",
        "cudnn_allow_tf32",
    ):
        if type(profile.get(name)) is not bool:
            raise ValueError(f"Invalid evaluation precision {name}")
    selector = profile.get("amp_dtype")
    dtype = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}.get(selector)
    if (
        dtype is None
        or profile.get("autocast_dtype") != dtype
        or profile.get("autocast_device_type") not in ("cuda", "cpu")
        or profile.get("float32_matmul_precision") not in ("highest", "high", "medium")
        or profile.get("environment_action_dtype") != "torch.float32"
    ):
        raise ValueError("Invalid evaluation precision dtype or device")
    valid_dtypes = {"torch.float32", "torch.bfloat16", "torch.float16"}
    for name in ("policy_parameter_dtype", "policy_action_dtype"):
        values = profile.get(name)
        if (
            type(values) is not list
            or not values
            or any(
                type(value) is not str or value not in valid_dtypes for value in values
            )
            or values != sorted(set(values))
        ):
            raise ValueError(f"Invalid evaluation precision {name}")
    if profile["autocast_enabled"] and not profile["configured_amp"]:
        raise ValueError("Evaluation autocast disagrees with configured AMP")
    return {
        **profile,
        "label": (
            selector.upper() + " autocast"
            if profile["autocast_enabled"]
            else "FP32 tensors; autocast disabled"
        ),
        "source_basis": "Recorded actual frozen evaluator precision",
    }


def _scene_distribution_changes(train: dict, world: dict) -> list[str]:
    """Describe observed changes; do not treat a held-out scene as in-distribution."""
    settings = train.get("scene_randomization") or {}
    randomization = world["randomization"]
    metadata = randomization.get("metadata") or {}
    train_variants = {
        variant
        for episode in train.get("episodes", [])
        for variant in (episode.get("scene_metadata") or {}).get("bottle_variants", [])
    }
    changes = []
    if (
        train_variants
        and settings.get("randomize_variants") is False
        and any(
            variant not in train_variants
            for variant in metadata.get("bottle_variants", [])
        )
    ):
        changes.append("bottle shapes")
    if settings.get("randomize_scales") is False and any(
        value != 1.0 for value in randomization.get("scale_states", {}).values()
    ):
        changes.append("object scales")
    return changes


def _evaluation(
    admitted: tuple[dict, dict, dict, dict] | None,
    train: dict,
    world: dict,
    summary_pin: dict,
) -> dict:
    if admitted is None:
        return _pending()
    worker, identity, learner, host = admitted
    if train["status"] != "completed":
        raise ValueError(
            "Frozen FastTD3 evaluation requires its completed TRAIN export"
        )
    for name in ("actor_updates", "critic_updates"):
        if type(learner.get(name)) is not int or learner[name] != 0:
            raise ValueError("FastTD3 evaluation performed learner updates")
    before, after = (
        learner.get("policy_state_before"),
        learner.get("policy_state_after"),
    )
    if not isinstance(before, str) or not before or before != after:
        raise ValueError("FastTD3 evaluation has no unchanged policy state fingerprint")
    checkpoint = learner["checkpoint"]
    artifact_bytes(checkpoint)
    artifact_bytes(worker["evaluation_checkpoint"])
    if any(
        pin["sha256"] != train["checkpoint_identity"]["sha256"]
        or pin["bytes"] != train["checkpoint_identity"]["bytes"]
        for pin in (checkpoint, worker["evaluation_checkpoint"])
    ):
        raise ValueError("FastTD3 evaluation checkpoint differs from the TRAIN export")
    config = pinned_json(worker["resolved_config"])
    reference = config["host"]["reference"]
    if (
        config["host"]["max_episode_steps"] != 3540
        or reference is None
        or reference["sha256"] != summary_pin["sha256"]
        or reference["world_index"] != 0
        or learner["evaluation_seeds"] != [world["world_seed"]]
        or len(learner["episodes"]) != 1
        or len(host["episodes"]) != 1
    ):
        raise ValueError(
            "FastTD3 evaluation is not the single full-horizon matched scene"
        )
    episode, actual = learner["episodes"][0], host["episodes"][0]
    if (
        episode.get("completed") is not True
        or type(episode.get("success")) is not bool
        or episode.get("seed") != world["world_seed"]
        or episode.get("scene_match") != "matched_baseline"
        or episode.get("randomization") != world["randomization"]
    ):
        raise ValueError(
            "FastTD3 episode is incomplete or has a different captured scene"
        )
    for name in (
        "seed",
        "controls",
        "completed",
        "success",
        "scene_match",
        "randomization",
        "final_native_task_eval",
        "video",
    ):
        if episode.get(name) != actual.get(name):
            raise ValueError("FastTD3 learner and host episode evidence disagree")
    controls = _integer(episode["controls"], "frozen controls", 1, 3540)
    if not episode["success"] and (
        controls != 3540 or episode["end_reason"] != "timeout"
    ):
        raise ValueError("Failed frozen evaluation must complete the full horizon")
    if (
        host["metrics"]["controls"] != controls
        or host["metrics"]["completed_episodes"] != 1
    ):
        raise ValueError("FastTD3 host did not complete exactly one evaluation")
    expected = {
        "num_worlds": 1,
        "successes": int(episode["success"]),
        "matched_worlds": 1,
        "actor_updates": 0,
        "critic_updates": 0,
    }
    if learner["successes"] != expected["successes"] or worker["metrics"] != expected:
        raise ValueError("FastTD3 frozen score and episode evidence disagree")
    evaluation = episode["final_native_task_eval"]
    if (
        evaluation["success"] != episode["success"]
        or evaluation["num_active_bottles"] != 5
    ):
        raise ValueError("FastTD3 native task judge differs from the scored episode")
    artifact_bytes(episode["video"])
    precision = _evaluation_precision(worker, learner)
    return {
        "status": "completed",
        "successes": expected["successes"],
        "episodes": 1,
        "seed": episode["seed"],
        "controls": controls,
        "bottles_in_bin": evaluation["num_bottles_in_bin"],
        "native_current_success": evaluation["success"],
        "return": _number(episode["return"], "evaluation return"),
        "max_native_reward": _number(episode["max_native_reward"], "evaluation reward"),
        "elapsed_seconds": _number(
            worker["elapsed_seconds"], "evaluation elapsed time"
        ),
        "evaluation_precision": precision,
        "scene_metadata": world["randomization"].get("metadata"),
        "scale_states": world["randomization"].get("scale_states"),
        "scene_distribution_changes": _scene_distribution_changes(train, world),
        "video_identity": episode["video"],
        "worker_receipt_identity": identity,
        "checkpoint_identity": checkpoint,
        "policy_state_before": before,
        "policy_state_after": after,
        "actor_updates": 0,
        "critic_updates": 0,
        "claim_scope": "One frozen matched scene; privileged-state scratch policy, not a method ranking",
    }


def _sft(run: Path | None) -> dict:
    if run is None:
        return _pending()
    admitted = coordinated_receipt(run, nested=True)
    if admitted is None:
        return _pending()
    worker, identity, coordinator = admitted
    if (
        worker.get("phase") != "original_pi05_sft_qualification"
        or worker["metrics"] != coordinator["metrics"]
    ):
        raise ValueError("pi0.5 SFT receipt phase or metrics disagree")
    metrics = worker["metrics"]
    if any(
        type(metrics.get(name)) is not bool
        for name in ("optimization_passed", "tiny_batch_loss_target_met")
    ):
        raise ValueError("Invalid SFT diagnostic qualification flags")
    return {
        "status": "completed",
        "actual_steps": _integer(worker["actual_steps"], "SFT updates", 1),
        "initial_fixed_loss": _number(
            metrics["initial_fixed_loss"], "initial TRAIN loss"
        ),
        "final_fixed_loss": _number(metrics["final_fixed_loss"], "final TRAIN loss"),
        "final_gradient_norm": _number(
            metrics["final_gradient_norm"], "SFT gradient norm"
        ),
        "optimization_passed": metrics["optimization_passed"],
        "tiny_batch_loss_target_met": metrics["tiny_batch_loss_target_met"],
        "stop_reason": worker["early_stopping_reason"],
        "elapsed_seconds": _number(worker["elapsed_s"], "SFT elapsed time"),
        "worker_receipt_identity": identity,
        "successes": None,
        "claim_scope": "Fixed-noise TRAIN loss diagnostic; no held-out validation or simulator task score",
    }


def _sealed_file(pin: dict) -> dict:
    """Hash large trees without loading weights; older producer pins lack lengths."""
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("Native evidence requires ordinary absolute files")
    digest, length = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            length += len(block)
    if digest.hexdigest() != pin["sha256"] or (
        "bytes" in pin
        and _integer(pin["bytes"], "artifact bytes", maximum=10**14) != length
    ):
        raise ValueError(f"Native artifact differs: {path}")
    return {"path": str(path), "bytes": length, "sha256": digest.hexdigest()}


def _sealed_json(pin: dict) -> tuple[dict, dict]:
    identity = _sealed_file(pin)
    return _json(Path(identity["path"]).read_bytes()), identity


def _checkpoint_tree(pin: dict, kind: str) -> tuple[dict, dict]:
    tree, identity = _sealed_json(pin)
    root = Path(tree["root"])
    records = tree["files"]
    if (
        tree.get("schema") != "expo-abc-checkpoint-tree/v1"
        or tree.get("kind") != kind
        or not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or not 1 <= len(records) <= 10000
    ):
        raise ValueError("Native checkpoint tree schema or root differs")
    closure = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Native checkpoint tree contains a symbolic link")
        if path.is_file():
            closure.append(str(path.relative_to(root)))
    if len(records) != len(closure) or {row["path"] for row in records} != set(closure):
        raise ValueError("Native checkpoint file closure differs")
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        digest != tree["tree_sha256"]
        or sum(row["bytes"] for row in records) != tree["total_bytes"]
    ):
        raise ValueError("Native checkpoint tree digest or length differs")
    for row in records:
        _sealed_file({**row, "path": str(root / row["path"])})
    producer, _ = _sealed_json(tree["producer_receipt"])
    if tree["root"] != producer.get("checkpoint_artifact"):
        raise ValueError("Native checkpoint root differs from its sealed producer")
    return tree, identity


def _native_counters(value: dict) -> dict:
    if type(value) is not dict or set(value) != {
        "critic",
        "batch_encoder",
        "edit_actor",
        "temp",
        "actor",
    }:
        raise ValueError("Native optimizer counter groups differ")
    for count in value.values():
        _integer(count, "native optimizer step")
    return value


def _native_admission(run: Path | None, phase: str, sft: dict) -> dict | None:
    if run is None:
        return None
    admitted = coordinated_receipt(run, nested=True)
    if admitted is None:
        return None
    worker, identity, coordinator = admitted
    if worker.get("phase") != phase or worker["metrics"] != coordinator["metrics"]:
        raise ValueError("Native pi0.5/EXPO phase or coordinator metrics differ")
    if worker.get("source_pins") != {
        "expo_ft": "21fe3d3b7d913c80836496817965932f49e0aedf",
        "openpi": "46407a41183b037313a383ff679683f2773b766d",
    }:
        raise ValueError("Native original author source pins differ")
    implementation = worker["implementation_sha256"]
    if not {
        "native.py",
        "original_runtime.py",
        "artifacts.py",
        "host.py",
        "data.py",
        "config.py",
        "replay.py",
        "stage.py",
    } <= set(implementation) or any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
        for digest in implementation.values()
    ):
        raise ValueError("Native implementation source hashes are missing")
    _native_counters(worker["optimizer_counters_before"])
    _native_counters(worker["optimizer_counters_after"])
    inputs, parameters = worker["input_pins"], worker["resolved_parameters"]
    tree, checkpoint_identity = _checkpoint_tree(
        inputs["checkpoint_pin"], "original_pi05_sft"
    )
    if (
        sft["status"] != "completed"
        or tree["producer_receipt"]["sha256"]
        != sft["worker_receipt_identity"]["sha256"]
        or worker["producer_sft_steps"] != sft["actual_steps"]
        or inputs["checkpoint_pin"] != parameters["checkpoint_pin"]
        or inputs["source_manifest"]["path"] != parameters["source_manifest"]
        or inputs["stats_manifest"]["sha256"] != parameters["stats_manifest_sha256"]
    ):
        raise ValueError("Native stage is not bound to the selected SFT/source/stats")
    source, source_identity = _sealed_json(inputs["source_manifest"])
    stats, stats_identity = _sealed_json(inputs["stats_manifest"])
    if (
        source.get("schema") != "abc-author-success-prefix/v1"
        or source.get("split") != "train"
        or source.get("task") != "put_plastic_bottles_in_bin"
        or stats.get("schema") != "expo-abc-yam14-diagnostic/v1"
        or stats.get("split") != "train"
        or stats["source_manifest"]["sha256"] != source_identity["sha256"]
    ):
        raise ValueError("Native source or TRAIN normalization lineage differs")
    for pin in [
        source["source_judge"],
        source["source_labels"],
        *source["raw_inputs"].values(),
    ]:
        _sealed_file(pin)
    stats_file = _sealed_file(stats["stats"])
    saved_stats = [
        row
        for row in tree["files"]
        if row["path"].startswith("assets/")
        and row["path"].endswith("/norm_stats.json")
    ]
    if len(saved_stats) != 1 or saved_stats[0]["sha256"] != stats_file["sha256"]:
        raise ValueError("Native normalization differs from the actual SFT assets")
    lifecycle = worker["host_lifecycle"]
    host, host_identity = _sealed_json(lifecycle["receipt"])
    config, config_identity = _sealed_json(host["config"])
    expected_stage = (
        "train" if phase == "original_expo_update_diagnostic" else "evaluation"
    )
    if (
        lifecycle.get("exit_code") != 0
        or lifecycle.get("cleanup_errors")
        or host != lifecycle["host_receipt"]
        or host.get("schema_version") != 1
        or host.get("status") != "completed"
        or host.get("blockers")
        or host.get("stage") != expected_stage
        or config["stage"] != expected_stage
        or config_identity["sha256"] != lifecycle["host_config"]["sha256"]
        or config_identity["path"] != parameters["host_config"]
    ):
        raise ValueError("Native host lifecycle or configuration differs")
    pinned_json(host["original_sources_assets"])
    if (
        host["metrics"]["controls"] != worker["actual_controls"]
        or sum(ep["controls"] for ep in host["episodes"]) != worker["actual_controls"]
    ):
        raise ValueError("Native host and worker control counts differ")
    native_episodes = [ep for ep in host["episodes"] if ep["controls"] > 0]
    if any(ep["completed"] for ep in host["episodes"] if ep["controls"] == 0):
        raise ValueError("A reset-only host record cannot be a completed episode")
    if len(native_episodes) != len(worker["episodes"]):
        raise ValueError("Native host and worker episode counts differ")
    videos = lifecycle["videos"]
    for pin in videos:
        artifact_bytes(pin)
    episode_videos = [ep["video"] for ep in host["episodes"]]
    if sorted(videos, key=lambda pin: pin["path"]) != sorted(
        episode_videos, key=lambda pin: pin["path"]
    ):
        raise ValueError("Native videos differ from their host lifecycle")
    for episode, actual in zip(worker["episodes"], native_episodes):
        _integer(episode["controls"], "native episode controls", 1, 3540)
        _integer(episode["command_clip_count"], "native clipped components")
        _number(episode["max_native_reward"], "native episode reward")
        if (
            any(
                type(episode[name]) is not bool
                for name in (
                    "full_task_success",
                    "completed_native_episode",
                    "terminated",
                    "truncated",
                )
            )
            or episode["seed"] not in parameters["seeds"]
        ):
            raise ValueError("Native episode flags or seed differ")
        for name, host_name in {
            "seed": "seed",
            "controls": "controls",
            "full_task_success": "success",
            "completed_native_episode": "completed",
            "command_clip_count": "action_clipping_count",
            "max_native_reward": "max_native_reward",
            "final_native_task_eval": "final_native_task_eval",
        }.items():
            if episode[name] != actual[host_name]:
                raise ValueError("Native learner and host episode evidence differs")
        mask = episode["ever_placed_mask"]
        if (
            len(mask) != 5
            or any(type(value) is not bool for value in mask)
            or sum(mask) != episode["ever_placed_count"]
        ):
            raise ValueError("Native ever-placed evidence differs")
    metrics = worker["metrics"]
    expected = {
        "full_task_success_count": sum(
            ep["full_task_success"] for ep in worker["episodes"]
        ),
        "completed_native_episode_count": sum(
            ep["completed_native_episode"] for ep in worker["episodes"]
        ),
        "max_native_reward": max(ep["max_native_reward"] for ep in worker["episodes"]),
        "command_clip_count": sum(
            ep["command_clip_count"] for ep in worker["episodes"]
        ),
        "max_ever_placed_bottles": max(
            ep["ever_placed_count"] for ep in worker["episodes"]
        ),
    }
    if (
        any(metrics[name] != value for name, value in expected.items())
        or host["metrics"]["action_clipping_count"] != metrics["command_clip_count"]
    ):
        raise ValueError("Native aggregate metrics differ from episode evidence")
    timing = metrics["sampler_wall_latency"]
    count = _integer(timing["count"], "native sampler calls", 1)
    if timing["warm_count"] != count - 1:
        raise ValueError("Native latency sample counts differ")
    for name in (
        "first_cold_s",
        "all_mean_s",
        *(["warm_median_s", "warm_p95_s"] if count > 1 else []),
    ):
        _number(timing[name], "native synchronized latency")
    return {
        "worker": worker,
        "worker_identity": identity,
        "host": host,
        "native_episodes": native_episodes,
        "host_identity": host_identity,
        "config": config,
        "input_identities": {
            "sft_checkpoint_pin": checkpoint_identity,
            "source_manifest": source_identity,
            "stats_manifest": stats_identity,
            "norm_stats": stats_file,
        },
    }


def _native_measurements(admitted: dict) -> dict:
    worker = admitted["worker"]
    return {
        "controls": _integer(worker["actual_controls"], "native controls", 1, 3540),
        "actual_steps": _integer(worker["actual_steps"], "native updates"),
        "elapsed_seconds": _number(worker["elapsed_s"], "native stage wall time"),
        "stop_reason": worker["early_stopping_reason"],
        "sampler_wall_latency": worker["metrics"]["sampler_wall_latency"],
        "worker_receipt_identity": admitted["worker_identity"],
        "host_receipt_identity": admitted["host_identity"],
        "input_identities": admitted["input_identities"],
        "action_clipping_count": worker["metrics"]["command_clip_count"],
        "max_native_reward": worker["metrics"]["max_native_reward"],
        "max_ever_placed_bottles": worker["metrics"]["max_ever_placed_bottles"],
        "sampling_recipe": worker["sampling_recipe"],
        "episode_rollouts": [
            {**ep, "video_identity": host["video"]}
            for ep, host in zip(worker["episodes"], admitted["native_episodes"])
        ],
        "successes": None,
        "episodes": None,
        "video": None,
        "claim_scope": "Synchronous native YAM development diagnostic; not realtime EXPO-FT or a paper score",
    }


def _native_training(admitted: dict | None, baseline_seed: int) -> dict:
    if admitted is None:
        return _pending()
    worker = admitted["worker"]
    result = _native_measurements(admitted)
    params = worker["resolved_parameters"]
    _integer(params["batch_size"], "EXPO diagnostic batch size", 2, 2)
    _integer(params["utd_ratio"], "EXPO diagnostic UTD", 1, 1)
    if baseline_seed in params["seeds"]:
        raise ValueError("Native EXPO diagnostic TRAIN domain differs")
    update_budget = _integer(params["max_updates"], "EXPO update budget", 1, 4)
    control_budget = _integer(params["max_controls"], "EXPO control budget", 1, 3540)
    steps = _integer(
        result["actual_steps"], "EXPO diagnostic updates", 1, update_budget
    )
    if result["controls"] > control_budget or len(worker["updates"]) != steps:
        raise ValueError("Native EXPO update/control budget differs")
    counters = worker["optimizer_counters_before"]
    for index, update in enumerate(worker["updates"], 1):
        _native_counters(update["optimizer_counters_before"])
        _native_counters(update["optimizer_counters_after"])
        if update["actual_update"] != index:
            raise ValueError("Native EXPO update sequence differs")
        if update["optimizer_counters_before"] != counters:
            raise ValueError("Native EXPO update counters are discontinuous")
        expected = {name: value + 1 for name, value in counters.items()}
        if update["optimizer_counters_after"] != expected:
            raise ValueError("Native EXPO original update counters differ")
        counters = expected
    if counters != worker["optimizer_counters_after"]:
        raise ValueError("Native EXPO final update counters differ")
    export_path = (
        Path(admitted["worker_identity"]["path"]).parent / "checkpoint-export.json"
    )
    export = _json(export_path.read_bytes())
    tree, checkpoint_identity = _checkpoint_tree(
        export["checkpoint_pin"], "original_expo_learner"
    )
    if (
        tree["root"] != worker["checkpoint_artifact"]
        or tree["producer_receipt"]["sha256"] != admitted["worker_identity"]["sha256"]
        or export["producer_receipt"] != tree["producer_receipt"]
    ):
        raise ValueError("Native EXPO export differs from its producer receipt")
    return {
        **result,
        "status": "completed",
        "batch_size": 2,
        "utd_ratio": 1,
        "control_budget": control_budget,
        "update_budget": update_budget,
        "updates": worker["updates"],
        "diagnostic_changes": worker["diagnostic_changes"],
        "optimizer_counters_before": worker["optimizer_counters_before"],
        "optimizer_counters_after": counters,
        "checkpoint_pin_identity": checkpoint_identity,
    }


def _native_evaluation(
    admitted: dict | None, world: dict, summary_pin: dict, train: dict | None = None
) -> dict:
    if admitted is None:
        return _pending()
    worker, host, config = admitted["worker"], admitted["host"], admitted["config"]
    result = _native_measurements(admitted)
    if (
        result["actual_steps"] != 0
        or worker["updates"]
        or worker["optimizer_counters_before"] != worker["optimizer_counters_after"]
    ):
        raise ValueError("Frozen pi0.5/EXPO evaluation changed optimizer counters")
    before, after = (
        worker["actual_parameter_hashes_before"],
        worker["actual_parameter_hashes_after"],
    )
    groups = {
        "critic",
        "batch_encoder",
        "edit_actor",
        "temp",
        "target_critic",
        "actor",
        "target_actor",
    }
    if (
        set(before) != groups
        or before != after
        or any(
            type(pin.get("bytes")) is not int
            or pin["bytes"] <= 0
            or type(pin.get("leaves")) is not int
            or pin["leaves"] <= 0
            or not isinstance(pin.get("sha256"), str)
            or len(pin["sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in pin["sha256"])
            for pin in before.values()
        )
    ):
        raise ValueError(
            "Frozen pi0.5/EXPO parameter byte hashes are missing or changed"
        )
    if train is not None:
        if train["status"] != "completed":
            raise ValueError(
                "Frozen EXPO evaluation requires its completed TRAIN export"
            )
        pin = worker["expo_checkpoint_pin"]
        tree, identity = _checkpoint_tree(pin, "original_expo_learner")
        if (
            identity != train["checkpoint_pin_identity"]
            or tree["producer_receipt"]["sha256"]
            != train["worker_receipt_identity"]["sha256"]
            or worker["resolved_parameters"]["expo_checkpoint_pin"] != pin
            or admitted["input_identities"] != train["input_identities"]
        ):
            raise ValueError("Frozen EXPO checkpoint or TRAIN lineage differs")
    reference = config["host"]["reference"]
    if (
        config["host"]["max_episode_steps"] != 3540
        or reference["sha256"] != summary_pin["sha256"]
        or reference["world_index"] != 0
        or worker["resolved_parameters"]["seeds"] != [world["world_seed"]]
        or len(admitted["native_episodes"]) != 1
    ):
        raise ValueError(
            "Native frozen evaluation requires the single captured baseline scene"
        )
    episode, actual = worker["episodes"][0], admitted["native_episodes"][0]
    if (
        actual["randomization"] != world["randomization"]
        or actual["scene_match"] != "matched_baseline"
        or actual["seed"] != world["world_seed"]
    ):
        raise ValueError("Native frozen captured RandomizationState differs")
    controls = result["controls"]
    if episode["completed_native_episode"] is not True:
        return {
            **result,
            "status": "partial",
            "actual_parameter_hashes": before,
            "optimizer_counters": worker["optimizer_counters_after"],
        }
    success = episode["full_task_success"]
    if worker["metrics"]["sampler_wall_latency"]["count"] != (controls + 7) // 8:
        raise ValueError(
            "Native full evaluation sampler count differs from C8 prefixes"
        )
    if (
        type(success) is not bool
        or episode["terminated"] is not success
        or episode["truncated"] is not (not success)
        or (not success and controls != 3540)
        or episode["stop_reason"]
        != ("task_success" if success else "native_time_limit")
        or actual["end_reason"] != ("success" if success else "timeout")
        or host["metrics"]["completed_episodes"] != 1
    ):
        raise ValueError("Native frozen completion, horizon or termination differs")
    evaluation = episode["final_native_task_eval"]
    if (
        evaluation["success"] is not success
        or evaluation["num_active_bottles"] != 5
        or success
        != (_integer(evaluation["num_bottles_in_bin"], "native bottles", 0, 5) == 5)
        or worker["metrics"]["full_task_success_count"] != int(success)
        or worker["metrics"]["completed_native_episode_count"] != 1
    ):
        raise ValueError("Native frozen score differs from the original task judge")
    return {
        **result,
        "status": "completed",
        "successes": int(success),
        "episodes": 1,
        "seed": episode["seed"],
        "bottles_in_bin": evaluation["num_bottles_in_bin"],
        "native_current_success": success,
        "actual_parameter_hashes": before,
        "optimizer_counters": worker["optimizer_counters_after"],
        "video_identity": actual["video"],
    }


def _atomic_write(path: Path, contents: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".report-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(
    run: Path,
    output: Path,
    *,
    fasttd3_train: Path | None = None,
    fasttd3_eval: Path | None = None,
    pi05_sft: Path | None = None,
    pi05_eval: Path | None = None,
    expo_train: Path | None = None,
    expo_eval: Path | None = None,
) -> dict:
    """Validate every displayed measurement before replacing the live report."""
    worker, identity, summary = _baseline(run)
    world = summary["worlds"][0]
    training = _training(_fasttd3(fasttd3_train, stage="train"), world["world_seed"])
    evaluation = _evaluation(
        _fasttd3(fasttd3_eval, stage="evaluate"),
        training,
        world,
        worker["original_summary"],
    )
    sft = _sft(pi05_sft)
    pi05_evaluation = _native_evaluation(
        _native_admission(pi05_eval, "original_pi05_frozen_eval", sft),
        world,
        worker["original_summary"],
    )
    expo_training = _native_training(
        _native_admission(expo_train, "original_expo_update_diagnostic", sft),
        world["world_seed"],
    )
    expo_evaluation = _native_evaluation(
        _native_admission(expo_eval, "original_expo_frozen_eval", sft),
        world,
        worker["original_summary"],
        expo_training,
    )
    output.mkdir(parents=True, exist_ok=True)
    for episode in training.get("episodes") or []:
        pin = episode["video_identity"]
        episode["video"] = copy_video(pin, output) if pin is not None else None
    if evaluation["status"] == "completed":
        evaluation["video"] = copy_video(evaluation["video_identity"], output)
    for native in (pi05_evaluation, expo_training, expo_evaluation):
        for episode in native.get("episode_rollouts", []):
            episode["video"] = copy_video(episode["video_identity"], output)
        if native["status"] == "completed" and native.get("video_identity"):
            native["video"] = copy_video(native["video_identity"], output)
    result = {
        "schema_version": 1,
        "condition": "Original ABC-VLA with upstream evaluation settings",
        "successes": int(world["success"]),
        "episodes": 1,
        "seed": world["world_seed"],
        "controls": world["steps"],
        "bottles_in_bin": world["final_task_eval"]["num_bottles_in_bin"],
        "native_current_success": world["final_success"],
        "video": copy_video(worker["original_videos"][0], output),
        "video_identity": worker["original_videos"][0],
        "summary_identity": worker["original_summary"],
        "worker_receipt_sha256": identity["sha256"],
        "configuration": summary["config"],
        "claim_scope": "One original-recipe layout; not a paper score or a causal ablation",
        "methods": {
            "fasttd3": {"training": training, "evaluation": evaluation},
            "pi05": {
                "sft": sft,
                "evaluation": pi05_evaluation,
                "expoft_training": expo_training,
                "expoft_evaluation": expo_evaluation,
            },
        },
    }
    _atomic_write(
        output / "results.json", json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    _atomic_write(output / "index.html", HTML)
    return result


HTML = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Original methods in ABC-Sim</title><style>
body{background:#101720;color:#e8edf3;font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}p{line-height:1.6;color:#c0ccd9}a{color:#9fc8ff}video{width:100%;background:#080c12}.card{background:#1a2532;border:1px solid #3b4b60;border-radius:12px;padding:20px;margin:20px 0}.score{font-size:28px;color:#94dcbb}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.grid .card{margin:0}@media(max-width:700px){.grid{grid-template-columns:1fr}}small{color:#b2bdcb}
</style><h1>Original methods · ABC-Sim</h1>
<p>The original policy and evaluation code ran in native ABC-Sim. The clip shows all three simulator cameras.</p>
<section class="card"><h2>Frozen ABC-VLA</h2><div id="score" class="score">Loading verified results…</div><p id="detail"></p><video id="video" controls preload="metadata"></video></section>
<p>Upstream settings: ten diffusion steps, RTC prefix 4 and lead 4, CPU MuJoCo physics, MJWarp cameras, and a maximum of 3,540 controls. The trained prompt remap is explicit.</p>
<p>This is one layout. The earlier comparison used different layouts, physics, sampling and RTC settings. This result does not isolate which change caused the difference.</p>
<div class="grid"><section class="card"><h2>Original FastTD3</h2><p>Privileged state policy trained from scratch. It receives object pose and velocity.</p><div id="fast-score" class="score">Task score pending</div><p id="fast-training">Training measurements pending.</p><p id="fast-evaluation">Frozen matched evaluation pending.</p><p id="fast-distribution"></p><div id="fast-clips"></div></section><section class="card"><h2>Original EXPO-FT · pi0.5</h2><p>Different pretrained image policy. Its task adaptation uses one source-judged three-bottle TRAIN sample. It is not a verified five-bottle expert dataset.</p><div id="sft-status" class="score">Task adaptation pending</div><p id="sft-detail"></p><h3>Frozen pi0.5</h3><div id="pi05-score" class="score">Task score pending</div><p id="pi05-detail"></p><h3>Ordinary EXPO diagnostic</h3><p id="expo-training">Native updates pending.</p><div id="expo-score" class="score">Frozen task score pending</div><p id="expo-detail"></p><p>Simulation pauses during inference. These runs do not measure realtime EXPO-FT or RTC.</p><div id="pi05-clips"></div></section></div>
<p>These conditions have different observations, backbones and training histories. This page reports the measured runs. It does not rank the methods or reproduce paper percentages.</p>
<p><a href="results.json">Measured result and source identities</a> · <a href="/small-comparison/">Preserved earlier diagnostic</a></p>
<script>
fetch('results.json').then(r=>{if(!r.ok)throw Error('Results unavailable');return r.json()}).then(d=>{
if(d.schema_version!==1||!Number.isInteger(d.successes)||d.episodes!==1||!/^[0-9a-f]{16}\\.mp4$/.test(d.video))throw Error('Invalid result');
document.querySelector('#score').textContent=d.successes+'/'+d.episodes+' successful episode';
document.querySelector('#detail').textContent='Layout '+d.seed+' · '+d.controls+' controls · '+d.bottles_in_bin+'/5 bottles currently in the bin · native success '+d.native_current_success;
document.querySelector('#video').src=d.video;
const methods=d.methods||{},fast=methods.fasttd3||{},train=fast.training||{},ev=fast.evaluation||{},pi=methods.pi05||{},sft=pi.sft||{};
const clips=document.querySelector('#fast-clips');
function clip(label,name,parent=clips){if(!/^[0-9a-f]{16}\\.mp4$/.test(name||''))return;const p=document.createElement('p');p.textContent=label;const v=document.createElement('video');v.controls=true;v.preload='metadata';v.src=name;parent.append(p,v);}
if(train.status==='completed'){
document.querySelector('#fast-training').textContent='TRAIN: '+train.controls+' controls · '+train.critic_updates+' critic updates · '+train.actor_updates+' actor updates · '+train.action_clipping_count+' clipped action components · '+train.elapsed_seconds.toFixed(1)+' seconds. Training clips are not task scores.';
(train.episodes||[]).forEach(e=>clip('TRAIN layout '+e.seed+' · '+e.controls+' controls · maximum native reward '+e.max_native_reward+' · '+e.stop_reason,e.video));
const settings=train.scene_randomization||{};
if(settings.randomize_variants===false&&settings.randomize_scales===false)document.querySelector('#fast-distribution').textContent='TRAIN uses fixed bottle shapes and scales.';
}
if(ev.status==='completed'){
document.querySelector('#fast-score').textContent=ev.successes+'/'+ev.episodes+' frozen matched episode';
const precision=(ev.evaluation_precision||{}).label||'precision not recorded';
document.querySelector('#fast-evaluation').textContent='Layout '+ev.seed+' · '+ev.controls+' controls · '+ev.bottles_in_bin+'/5 bottles currently in the bin · native success '+ev.native_current_success+' · zero learner updates · unchanged policy state · '+precision+' · '+ev.elapsed_seconds.toFixed(1)+' seconds for the complete stage, including simulator work.';
if((ev.scene_distribution_changes||[]).length)document.querySelector('#fast-distribution').textContent+=' The captured evaluation scene has different '+ev.scene_distribution_changes.join(' and ')+' from TRAIN. This changes the state distribution.';
clip('Frozen matched evaluation · '+precision,ev.video);
}
if(sft.status==='completed'){
document.querySelector('#sft-status').textContent=sft.actual_steps+' SFT update'+(sft.actual_steps===1?'':'s');
document.querySelector('#sft-detail').textContent='Fixed-noise TRAIN loss: '+sft.initial_fixed_loss.toFixed(6)+' → '+sft.final_fixed_loss.toFixed(6)+' · gradient norm '+sft.final_gradient_norm.toFixed(4)+' · stop: '+sft.stop_reason+' · '+sft.elapsed_seconds.toFixed(1)+' seconds. This measures training loss; it is not held-out validation or simulator success.';
}
const piClips=document.querySelector('#pi05-clips');
function latency(stage){const t=stage.sampler_wall_latency||{};return t.count?'Sampler wall time: first call '+t.first_cold_s.toFixed(3)+' s'+(t.warm_count?' · warm median '+t.warm_median_s.toFixed(3)+' s · warm p95 '+t.warm_p95_s.toFixed(3)+' s':'')+'. ':'';}
function nativeEvaluation(stage,scoreId,detailId,label){
if(stage.status!=='completed'&&stage.status!=='partial')return;
document.querySelector(scoreId).textContent=stage.status==='completed'?stage.successes+'/'+stage.episodes+' frozen matched episode':'Partial rollout; task score unavailable';
document.querySelector(detailId).textContent=stage.controls+' controls · '+(stage.status==='completed'?stage.bottles_in_bin+'/5 bottles currently in the bin · ':'')+'zero updates · unchanged optimizer counters and parameter byte hashes · stop: '+stage.stop_reason+'. '+latency(stage)+stage.elapsed_seconds.toFixed(1)+' seconds for the complete stage, including model and simulator work.';
(stage.episode_rollouts||[]).forEach(e=>clip(label+' · layout '+e.seed+' · '+e.controls+' controls · '+e.ever_placed_count+'/5 bottles ever placed',e.video,piClips));
}
nativeEvaluation(pi.evaluation||{},'#pi05-score','#pi05-detail','Frozen pi0.5');
const expoTrain=pi.expoft_training||{};
if(expoTrain.status==='completed'){
document.querySelector('#expo-training').textContent='Local TRAIN diagnostic: batch '+expoTrain.batch_size+' · UTD '+expoTrain.utd_ratio+' · '+expoTrain.controls+'/'+expoTrain.control_budget+' controls · '+expoTrain.actual_steps+'/'+expoTrain.update_budget+' original update calls · stop: '+expoTrain.stop_reason+'. '+latency(expoTrain)+'These training rollouts are not task scores or paper performance.';
(expoTrain.episode_rollouts||[]).forEach(e=>clip('EXPO TRAIN · layout '+e.seed+' · '+e.controls+' controls · maximum native reward '+e.max_native_reward,e.video,piClips));
}
nativeEvaluation(pi.expoft_evaluation||{},'#expo-score','#expo-detail','Frozen ordinary EXPO');
}).catch(e=>{document.querySelector('#score').textContent=e.message;});
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fasttd3-train", type=Path)
    parser.add_argument("--fasttd3-eval", type=Path)
    parser.add_argument("--pi05-sft", type=Path)
    parser.add_argument("--pi05-eval", type=Path)
    parser.add_argument("--expo-train", type=Path)
    parser.add_argument("--expo-eval", type=Path)
    args = parser.parse_args()
    publish(
        args.run,
        args.output,
        fasttd3_train=args.fasttd3_train,
        fasttd3_eval=args.fasttd3_eval,
        pi05_sft=args.pi05_sft,
        pi05_eval=args.pi05_eval,
        expo_train=args.expo_train,
        expo_eval=args.expo_eval,
    )


if __name__ == "__main__":
    main()
