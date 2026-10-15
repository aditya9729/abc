"""Publish accepted R1 Lite cohorts and full recordings into the existing local viewer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from abc_bench.export_original_methods import export_viewer


def pin(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def checked(item: dict) -> Path:
    path = Path(item["path"])
    if pin(path) != item:
        raise ValueError(f"Accepted artifact changed: {path}")
    return path


def _visual_review(accepted: dict) -> None:
    visual = accepted["visual_review"]
    if (
        visual["status"] != "viewed_by_root"
        or visual["reviewer"] != "/root"
        or not visual["files"]
        or not visual["seeds"]
        or not visual["selection_reason"]
        or any(
            type(seed) is not int or seed not in range(9100, 9120)
            for seed in visual["seeds"]
        )
    ):
        raise ValueError("Explicit root visual review is required")
    for item in visual["files"]:
        checked(item)


def _clean_execution(accepted: dict, evidence: dict) -> None:
    execution = evidence["execution"]
    canonical = json.loads(checked(execution["canonical"]).read_text())
    worker = json.loads(checked(execution["worker"]).read_text())
    if (
        canonical["run_id"] != accepted["run_id"]
        or canonical["status"] != "completed"
        or canonical["worker_exit_code"] != 0
        or canonical["worker_receipt_sha256"] != execution["worker"]["sha256"]
        or worker["status"] != "completed"
        or worker["author_exit_code"] != 0
        or worker["client_exit_code"] != 0
        or worker["owned_children_stopped"] is not True
        or worker.get("cleanup_errors")
        or worker["author_shutdown"] != "already_stopped"
        or worker["client_shutdown"] != "already_stopped"
        or execution["natural_shutdown"] is not True
    ):
        raise ValueError(
            "Full EXPO requires clean completed canonical and worker receipts"
        )


def _complete_media(evidence: dict) -> None:
    native = evidence["native"]
    schedule = [9001, *range(9100, 9120)]
    if (
        native["status"] != "complete_native_evidence"
        or native["reset_order"]
        != [[index + 1, seed] for index, seed in enumerate(schedule)]
        or native["schedule_valid"] is not True
        or native["errors"]
        or [row["seed"] for row in native["episodes"]] != schedule
        or [row["seed"] for row in evidence["media"]] != schedule
    ):
        raise ValueError("Full EXPO native or media cohort is incomplete")
    for index, (episode, media) in enumerate(
        zip(native["episodes"], evidence["media"])
    ):
        source = checked(media["video"])
        sidecar_path = checked(media["sidecar"])
        sidecar = json.loads(sidecar_path.read_text())
        if (
            episode["scored"] is not (index > 0)
            or media["scored"] is not (index > 0)
            or episode["errors"]
            or str(source) != episode["video_path"]
            or str(sidecar_path) != episode["sidecar_path"]
            or media["full_decode_exit_code"] != 0
            or Fraction(media["probe"]["avg_frame_rate"]) != 15
            or (media["probe"]["width"], media["probe"]["height"]) != (640, 360)
            or int(media["probe"]["nb_read_frames"]) != episode["native_controls"] + 1
            or sidecar["seed"] != episode["seed"]
            or sidecar["native_controls"] != episode["native_controls"]
            or sidecar["frames"] != episode["native_controls"] + 1
            or sidecar["fps"] != 15
            or (index > 0 and sidecar["complete_native_episode"] is not True)
            or (index == 0 and episode["native_controls"] != 0)
        ):
            raise ValueError("Full EXPO native/media evidence disagrees")


def _initialized_evidence(accepted: dict) -> tuple[dict, dict]:
    """Authenticate the complete before-learning cohort without running inference."""
    if (
        accepted.get("schema") != "sadhana.initialized_full_expo_root_acceptance/1"
        or accepted.get("status") != "accepted_initialized_full_expo_dev20"
        or accepted.get("benchmark_accepted") is not True
    ):
        raise ValueError("Root acceptance of initialized full EXPO DEV20 is required")
    evidence = json.loads(checked(accepted["read_only_acceptance"]).read_text())
    observations = evidence["accepted_scalar_observations"]
    if (
        evidence["schema"] != "sadhana.r1lite_initialized_full_expo_dev20_acceptance/1"
        or evidence["status"] != "read_only_evidence_accepted"
        or evidence["run_id"] != accepted["run_id"]
        or evidence["evidence_errors"]
        or evidence["author_native_comparison"] != observations
        or ("observations" in accepted and accepted["observations"] != observations)
        or evidence["checkpoint_files_unchanged"] is not True
        or evidence["processes_after_checks"]["live_owned"]
        or observations["full_agent_restore_marker"] != []
        or observations["successes"] + observations["failures"] != 20
        or observations["async_full_expo_marker"]
        != "RealTimeEXPOFTLearner eval: using async main-actor inference delay=5 env-steps"
    ):
        raise ValueError("Initialized root acceptance and read-only evidence disagree")
    _visual_review(accepted)
    for name in (
        "root_pins",
        "launch_pins",
        "immutable_input_pins",
        "author_source_pins",
        "artifact_pins",
    ):
        if not evidence[name]:
            raise ValueError(f"Initialized evidence lacks {name}")
        for item in evidence[name]:
            checked(item)
    _clean_execution(accepted, evidence)
    initialization = evidence["initialization"]
    frozen = evidence["frozen_scope"]
    if (
        initialization["model_cls"] != "RealTimeEXPOFTLearner"
        or initialization["checkpoint_step"] != 0
        or initialization["seed"] != 42
        or initialization["learning_updates_in_evaluator"] != 0
        or initialization["actor_weight_producer_actual_updates"] != 2001
        or initialization["sampling"]
        != {
            "base_candidates": 32,
            "edited_candidates": 32,
            "total_candidates": 64,
            "filter_N": 32,
            "filter_n_edit": 1,
            "horizon": 16,
            "replan_steps": 8,
            "delay": 5,
        }
        or observations["actor_target_weight_loader_reads"]
        != [initialization["actor_weights"]] * 2
        or frozen["source_calls_update_or_save"]
        or frozen["evaluation_checkpoint_step"] != 0
        or frozen["actor_weight_checkpoint_label"] != 2000
        or frozen["actor_weight_producer_actual_updates"] != 2001
        or frozen["full_agent_restored"] is not False
    ):
        raise ValueError("Initialized EXPO recipe or frozen evaluation scope differs")
    checked(frozen["evaluator"])
    _complete_media(evidence)
    return evidence, observations


def _named_pin(entries: list[dict], name: str) -> dict:
    matches = [item for item in entries if Path(item["path"]).name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one accepted {name}")
    checked(matches[0])
    return matches[0]


def _continued_producer(
    producer: dict, host: dict, evidence: dict, step: int
) -> tuple[int, dict]:
    """Authenticate a separately accepted restart and its complete NEW history segment."""
    schedule = host["schedule"]
    history = host["history"]
    parent = json.loads(checked(host["parent_report"]).read_text())
    calls = schedule["total_update_calls"]
    new_calls = schedule["new_update_calls"]
    if (
        producer.get("status") != "root_accepted_original_online_checkpoint"
        or producer.get("owner") != "/root"
        or producer.get("artifact_accepted") is not True
        or producer.get("run_id") != evidence["producer_run_id"]
        or producer.get("checkpoint") != evidence["accepted_producer_checkpoint"]
        or producer.get("checkpoint_step") != step
        or host.get("run_id") != producer["run_id"]
        or host.get("status") != "resumed_checkpoint_replay_complete_history_verified"
        or host.get("artifact_accepted") is not False
        or host.get("lifecycle_accepted") is not False
        or host.get("all_inspected_files_unchanged") is not True
        or host.get("parent_artifacts_unchanged") is not True
        or host.get("checkpoint") != producer["checkpoint"]
        or schedule["checkpoint_step"] != step
        or type(calls) is not int
        or type(new_calls) is not int
        or new_calls <= 0
        or calls != parent["model"]["counters"]["actor_train_state"]["step"] + new_calls
        or history["raw_history_complete"] is not True
        or history["final_boundary_row_present"] is not True
        or history["last_step"] != step
        or parent.get("raw_history_complete") is not False
        or parent.get("status") != "host_checkpoint_replay_verified_history_incomplete"
    ):
        raise ValueError("Continued EXPO producer or complete-segment history differs")
    final = history["final_row"]
    required = (
        "actor_loss",
        "actor_grad_norm",
        "critic_loss",
        "filter_critic_loss",
        "edit_actor_loss",
        "temperature",
        "temperature_loss",
        "actor_state_step",
        "next_q_nan_ratio",
    )
    if any(
        type(final.get("training/" + key)) not in (int, float)
        or not math.isfinite(final["training/" + key])
        for key in required
    ):
        raise ValueError(
            "Continued final logged update metrics are absent or nonfinite"
        )
    if (
        final["training/actor_state_step"] != calls - 1
        or final["training/next_q_nan_ratio"] != 0
    ):
        raise ValueError("Continued final logged update counters differ")
    inventory = host["checkpoint_files"]
    root = Path(host["checkpoint"])
    if (
        not root.is_absolute()
        or not root.is_dir()
        or root.is_symlink()
        or not inventory
        or len({item["path"] for item in inventory}) != len(inventory)
        or any(
            Path(item["path"]).is_symlink()
            or not Path(item["path"]).resolve().is_relative_to(root.resolve())
            for item in inventory
        )
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ValueError(
            "Continued checkpoint inventory is absent, aliased or outside its root"
        )
    for item in inventory:
        checked(item)
    observed = {str(path) for path in root.rglob("*") if path.is_file()}
    if observed != {item["path"] for item in inventory}:
        raise ValueError("Continued checkpoint inventory has extra or missing files")
    for key in ("history_file", "native_events", "author_log", "source_manifest"):
        checked(host[key])
    return calls, {
        "resumed_segment_complete": True,
        "parent_history_complete": False,
        "metrics_scope": "final_logged_update_only",
        "approximate_restart": True,
        "new_learning_updates": new_calls,
        "parent_report": host["parent_report"],
    }


def _matched_evidence(
    accepted: dict, cohort: str, *, continued: bool = False
) -> tuple[dict, dict, dict]:
    """Bind a completed new full-policy cohort and its restricted producer acceptance."""
    if (
        accepted.get("schema") != "sadhana.r1lite_frozen_full_expo_root_acceptance/1"
        or accepted.get("status") != "accepted_frozen_full_expo_dev20"
        or accepted.get("owner") != "/root"
        or accepted.get("cohort") != cohort
        or accepted.get("benchmark_accepted") is not True
    ):
        raise ValueError(
            "Explicit root acceptance of the new full EXPO cohort is required"
        )
    evidence = json.loads(checked(accepted["read_only_acceptance"]).read_text())
    observations = evidence["accepted_scalar_observations"]
    step = evidence["checkpoint_step"]
    trained = cohort == "trained"
    if (
        evidence.get("schema") != "sadhana.r1lite_frozen_full_expo_dev20_acceptance/1"
        or evidence.get("status") != "read_only_evidence_accepted"
        or evidence.get("cohort") != cohort
        or evidence.get("run_id") != accepted["run_id"]
        or evidence["evidence_errors"]
        or evidence["checkpoint_files_unchanged"] is not True
        or evidence["processes_after_checks"]["live_owned"]
        or evidence["author_native_comparison"] != observations
        or accepted.get("observations") != observations
        or observations["successes"] + observations["failures"] != 20
        or observations["incomplete"] != 0
        or observations["async_full_expo_marker"]
        != "RealTimeEXPOFTLearner eval: using async main-actor inference delay=5 env-steps"
        or type(step) is not int
        or (step < 9009 if trained else step != 0)
        or observations["full_agent_restore_marker"] != ([str(step)] if trained else [])
    ):
        raise ValueError(
            "New full EXPO root acceptance and read-only evidence disagree"
        )
    _visual_review(accepted)
    _clean_execution(accepted, evidence)
    for name in (
        "source_pins",
        "operations_pins",
        "installed_supervisor_pins",
        "producer_artifact_pins",
        "launch_pins",
        "immutable_input_pins",
        "author_source_pins",
        "artifact_pins",
    ):
        if not evidence[name]:
            raise ValueError(f"New full EXPO evidence lacks {name}")
        for item in evidence[name]:
            checked(item)
    frozen = evidence["frozen_scope"]
    if (
        frozen["source_calls_update_or_save"]
        or frozen["evaluation_checkpoint_step"] != step
        or frozen["full_agent_restored"] is not trained
    ):
        raise ValueError("Original frozen restore scope differs")
    checked(frozen["evaluator"])
    _complete_media(evidence)
    intent = json.loads(
        checked(
            _named_pin(evidence["artifact_pins"], "EVAL_EXEC_INTENT.json")
        ).read_text()
    )
    if (
        intent["producer_run_id"] != evidence["producer_run_id"]
        or intent["cohort"] != cohort
        or intent["checkpoint_step"] != step
        or intent["accepted_producer_checkpoint"]
        != evidence["accepted_producer_checkpoint"]
        or any(
            intent[key] not in evidence["launch_pins"]
            for key in ("config", "job", "settings")
        )
        or intent["root_artifact_acceptance"] not in evidence["producer_artifact_pins"]
        or intent["host_inspection"] not in evidence["producer_artifact_pins"]
    ):
        raise ValueError("New cohort intent and accepted producer bindings differ")
    config = json.loads(checked(intent["config"]).read_text())
    settings = json.loads(checked(intent["settings"]).read_text())
    arguments = dict(item.split("=", 1) for item in config["arguments"])
    fixed = {
        "--seed": "42",
        "--num_episodes": "20",
        "--replan_steps": "8",
        "--delay": "5",
        "--only_base_actions": "False",
        "--save_video": "True",
        "--sim_latency": "0",
        "--config.N": "32",
        "--config.n_edit_samples": "32",
        "--config.filter_N": "32",
        "--config.filter_n_edit": "1",
        "--config.edit_scale": "0.1",
        "--checkpoint_step": str(step),
    }
    if (
        config["program"] != "eval_droid_policy.py"
        or len(arguments) != len(config["arguments"])
        or any(arguments.get(key) != value for key, value in fixed.items())
        or settings["episode_seeds"] != [9001, *range(9100, 9120)]
        or settings["record_video"] is not True
        or (
            not trained
            and observations["actor_target_weight_loader_reads"]
            != [arguments["--config.pi05_weight_loader_path"]] * 2
        )
    ):
        raise ValueError("Matched full-policy recipe or episode schedule differs")
    expected_jax = {
        "JAX_DEFAULT_PRNG_IMPL": "threefry2x32",
        "JAX_THREEFRY_PARTITIONABLE": "true",
        "JAX_THREEFRY_GPU_KERNEL_LOWERING": "false",
        "JAX_ENABLE_X64": "false",
        "JAX_PLATFORMS": "cuda",
        "JAX_PLATFORM_NAME": "cuda",
    }
    actual = evidence["execution"]["declared_and_actual_environment"]
    if (
        intent["declared_parent_jax"] != expected_jax
        or set(actual)
        != set(expected_jax)
        | {"XLA_PYTHON_CLIENT_MEM_FRACTION", "CUDA_VISIBLE_DEVICES"}
        or any(actual.get(key) != value for key, value in expected_jax.items())
        or actual["XLA_PYTHON_CLIENT_MEM_FRACTION"] != "0.90"
        or not actual["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
    ):
        raise ValueError("Declared and observed evaluation PRNG settings differ")
    producer = json.loads(checked(intent["root_artifact_acceptance"]).read_text())
    host = json.loads(checked(intent["host_inspection"]).read_text())
    if continued:
        if producer.get("host_inspection") != intent["host_inspection"]:
            raise ValueError(
                "Continued root acceptance does not bind this host inspection"
            )
        calls, continued_scope = _continued_producer(producer, host, evidence, step)
    else:
        if (
            producer.get("status") != "root_accepted_original_online_checkpoint"
            or producer.get("owner") != "/root"
            or producer.get("artifact_accepted") is not True
            or producer.get("run_id") != evidence["producer_run_id"]
            or producer.get("checkpoint") != evidence["accepted_producer_checkpoint"]
            or producer.get("host_inspection") != intent["host_inspection"]
            or host.get("status")
            != "host_checkpoint_replay_verified_history_incomplete"
            or host.get("all_inspected_files_unchanged") is not True
            or host.get("checkpoint") != producer["checkpoint"]
            or (trained and step != producer["checkpoint_step"])
            or any(
                value.get("acceptance_scope") != "frozen_evaluation_only"
                or value.get("frozen_evaluation_only") is not True
                or value.get("raw_history_complete") is not False
                or value.get("training_metrics_available") is not False
                for value in (producer, host)
            )
        ):
            raise ValueError(
                "Producer acceptance must retain its incomplete-history evaluation-only scope"
            )
        calls = host["expected_update_calls"]
    counters = {key: row["step"] for key, row in host["model"]["counters"].items()}
    if (
        type(calls) is not int
        or calls <= 0
        or counters
        != {
            "actor_train_state": calls,
            "edit_actor": calls,
            "temp": calls,
            "critic": 20 * calls,
            "filter_critic": 20 * calls,
            "batch_encoder": 20 * calls,
        }
    ):
        raise ValueError("Accepted original update counters differ")
    binding = {
        "producer_run_id": evidence["producer_run_id"],
        "accepted_producer_checkpoint": evidence["accepted_producer_checkpoint"],
        "evaluator": frozen["evaluator"],
        "recipe_arguments": {
            key: value
            for key, value in arguments.items()
            if key not in ("--checkpoint_dir", "--checkpoint_step", "--client_port")
        },
        "settings": {
            key: value for key, value in settings.items() if key != "event_log_path"
        },
        "declared_parent_jax": expected_jax,
    }
    metadata = {
        "cohort": cohort,
        "evaluation_checkpoint_step": step,
        "learning_updates": calls if trained else 0,
        "producer_update_counters": counters,
        "producer_checkpoint_label": producer["checkpoint_step"],
        "producer_run_id": evidence["producer_run_id"],
        "producer_acceptance": intent["root_artifact_acceptance"],
        "producer_history_scope": continued_scope
        if continued
        else {
            key: producer[key]
            for key in (
                "acceptance_scope",
                "frozen_evaluation_only",
                "raw_history_complete",
                "training_metrics_available",
            )
        },
        "comparison_binding": binding,
        "comparison_binding_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True).encode()
        ).hexdigest(),
        "declared_parent_jax": expected_jax,
    }
    return evidence, observations, metadata


def _publish_locked(
    acceptance: Path, output: Path, receipt: Path, cohort: str = "rtc"
) -> dict:
    """Copy all scored videos and add one result field; preserve historical data."""
    if not all(p.is_absolute() for p in (acceptance, output, receipt)):
        raise ValueError("Use absolute acceptance, viewer and receipt paths")
    if not output.is_dir() or output.is_symlink() or receipt.exists():
        raise ValueError("Require an existing ordinary viewer and a fresh receipt")
    accepted = json.loads(acceptance.read_text())
    metadata = None
    if cohort in (
        "matched_initialized_full_expo",
        "trained_full_expo",
        "continued_full_expo",
    ):
        evidence, observations, metadata = _matched_evidence(
            accepted,
            "initialized" if cohort == "matched_initialized_full_expo" else "trained",
            continued=cohort == "continued_full_expo",
        )
    elif cohort == "initialized_full_expo":
        evidence, observations = _initialized_evidence(accepted)
    elif cohort == "rtc":
        if (
            accepted.get("status") != "accepted_complete_native_rtc_dev20_artifacts"
            or accepted.get("benchmark_accepted") is not True
        ):
            raise ValueError("Root acceptance of complete RTC measurements is required")
        evidence = json.loads(checked(accepted["read_only_acceptance"]).read_text())
        if (
            evidence["status"] != "read_only_evidence_accepted"
            or evidence["run_id"] != accepted["run_id"]
            or evidence["evidence_errors"]
            or evidence["accepted_scalar_observations"] != accepted["observations"]
        ):
            raise ValueError("Root acceptance and complete read-only evidence disagree")
        observations = accepted["observations"]
        if (
            observations["full_agent_restore_marker"] != ["2000"]
            or accepted["canonical_status"] != "failed"
            or accepted["worker_status"] != "failed"
        ):
            raise ValueError(
                "Require the declared RTC2000 cohort and preserved supervision failure"
            )
    else:
        raise ValueError("Unknown R1 Lite publication cohort")
    episodes = observations["episodes"]
    if (
        [row["seed"] for row in episodes] != list(range(9100, 9120))
        or observations["incomplete"] != 0
        or sum(row["success"] for row in episodes) != observations["successes"]
    ):
        raise ValueError("Require the complete ordered DEV20 cohort")
    media = {row["seed"]: row for row in evidence["media"] if row["scored"]}
    native = {
        row["seed"]: row for row in evidence["native"]["episodes"] if row["scored"]
    }
    if set(media) != set(range(9100, 9120)) or set(native) != set(media):
        raise ValueError("Every scored outcome needs its complete native video")
    results = output / "results.json"
    old_bytes = results.read_bytes()
    base = json.loads(old_bytes)
    if base.get("schema") != "sadhana.original-methods.progress/v2":
        raise ValueError("Existing viewer schema differs")
    if cohort == "continued_full_expo" and "r1lite_continued_full_expo" in base:
        raise ValueError(
            "Continued EXPO cohort already exists; publication is append-only"
        )
    if metadata is not None and cohort != "continued_full_expo":
        other_key = (
            "r1lite_matched_initialized_full_expo"
            if cohort == "trained_full_expo"
            else "r1lite_trained_full_expo"
        )
        other = base.get(other_key)
        if (
            other is not None
            and other.get("comparison_binding_sha256")
            != metadata["comparison_binding_sha256"]
        ):
            raise ValueError(
                "Before and after cohorts do not share the accepted matched comparison contract"
            )
    rows = []
    for row in episodes:
        video = media[row["seed"]]
        source = checked(video["video"])
        checked(video["sidecar"])
        if (
            video["full_decode_exit_code"] != 0
            or int(video["probe"]["nb_read_frames"]) != row["native_controls"] + 1
        ):
            raise ValueError("A complete decoded recording is required")
        name = video["video"]["sha256"][:16] + ".mp4"
        target = output / name
        if target.is_symlink():
            raise ValueError("Video destination must not be a symlink")
        if target.exists():
            if pin(target)["sha256"] != video["video"]["sha256"]:
                raise ValueError("Video content identity collision")
        else:
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
        ending = native[row["seed"]]["native_ending"]
        if (
            ending not in ("success", "failure", "timeout")
            or row["success"] != (ending == "success")
            or row["native_controls"] != native[row["seed"]]["native_controls"]
            or row["len"] != row["native_controls"] + 1
            or row["return"] != row["native_return"]
        ):
            raise ValueError("Native ending disagrees with the scored outcome")
        rows.append({**row, "outcome": "native_" + ending, "video": name})
    timestamp = datetime.now(timezone.utc).isoformat()
    cohort_result = {
        "schema": "sadhana.r1lite_viewer/1",
        "snapshot_utc": timestamp,
        "run_id": accepted["run_id"],
        "acceptance": pin(acceptance),
        "successes": observations["successes"],
        "episodes": rows,
        "checkpoint_label": 2000,
        "actual_updates": 2001,
        "telemetry": evidence["telemetry"],
        "supervision": "Failed shutdown supervision; complete task artifacts separately accepted. Author SIGTERM was attempted. Both child exit codes were zero; natural author shutdown is unproven.",
        "scope": "One training seed and one fixed DEV20 cohort. Synthetic training from one source episode. Fixed base and torso. This is RTC-SFT before online EXPO learning.",
    }
    if cohort == "initialized_full_expo":
        cohort_result.update(
            schema="sadhana.r1lite_initialized_full_expo_viewer/1",
            initialization=evidence["initialization"],
            evaluation_checkpoint_step=0,
            learning_updates=0,
            supervision="Clean completed canonical and worker receipts. Author and client exited zero with natural shutdown.",
            scope="Original full Real-Time EXPO-FT policy before online learning: RTC2000 actor weights with freshly initialized editor and critics. N32 plus 32 edited candidates, H16/C8/delay5. One fixed DEV20 cohort, fixed base and torso. This measures initialization, not learning improvement. Later matched full-policy evaluation remains pending.",
        )
        # These legacy names describe the RTC weight producer, not EXPO training.
        cohort_result["actor_weight_checkpoint_label"] = cohort_result.pop(
            "checkpoint_label"
        )
        cohort_result["actor_weight_producer_actual_updates"] = cohort_result.pop(
            "actual_updates"
        )
        base["r1lite_initialized_full_expo"] = cohort_result
    elif cohort == "continued_full_expo":
        cohort_result.pop("checkpoint_label")
        cohort_result.pop("actual_updates")
        cohort_result.update(
            schema="sadhana.r1lite_continued_full_expo_viewer/1",
            **metadata,
            supervision="Clean completed frozen evaluation; author and client exited zero. The resumed training artifact has separate root acceptance.",
            scope="One fixed DEV20 simulation cohort after a one-time approximate training restart. The new segment has complete raw history and finite final logged-update metrics; the earlier parent's incomplete history remains unchanged. Saved learner RNG is restored; policy noise is not identical. Replay sampling, prefetch and environment restart are approximate. No reliable RL improvement or hardware claim.",
        )
        base["r1lite_continued_full_expo"] = cohort_result
    elif metadata is not None:
        cohort_result.pop("checkpoint_label")
        cohort_result.pop("actual_updates")
        cohort_result.update(
            schema="sadhana.r1lite_matched_full_expo_viewer/1",
            **metadata,
            supervision="Clean completed frozen evaluation; author and client exited zero. The training producer's failed cancellation receipt and unknown worker exit remain preserved.",
            scope="One fixed DEV20 simulation cohort with declared JAX PRNG settings. Original initialization or full-checkpoint restore and sampling are retained. The training artifact is accepted for frozen evaluation only: raw training history is incomplete and loss, gradient and Q-target metrics are unavailable. Matched seeds/configuration do not imply identical policy noise or measured equal fresh heads. No multi-seed, held-out or hardware claim.",
        )
        base["r1lite_" + cohort] = cohort_result
    else:
        base["r1lite"] = cohort_result
    if results.read_bytes() != old_bytes:
        raise RuntimeError(
            "Viewer data changed during preparation; retry from current data"
        )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    backup = receipt.with_suffix(".previous-results.json")
    with backup.open("xb") as stream:
        stream.write(old_bytes)
    old_index = output / "index.html"
    with receipt.with_suffix(".previous-index.html").open("xb") as stream:
        stream.write(old_index.read_bytes())
    export_viewer(output)
    base["packaged_viewer_release"] = {
        "previous": base.get("packaged_viewer_release"),
        "template": pin(old_index),
        "scope": (
            "Accepted R1 Lite RTC DEV20 with all20 complete videos"
            if cohort == "rtc"
            else f"Accepted {cohort} DEV20 with all20 complete videos; historical cohorts preserved"
        ),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(base, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, results)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report = {
        "status": f"accepted_{cohort}_results_and_all20_videos_published",
        "at_utc": timestamp,
        "acceptance": pin(acceptance),
        "previous_results": pin(backup),
        "results": pin(results),
        "template": pin(old_index),
        "videos": [pin(output / row["video"]) for row in rows],
        "scope": "Data publication only; browser and video playback checks are separate.",
    }
    with receipt.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def publish(acceptance: Path, output: Path, receipt: Path, cohort: str = "rtc") -> dict:
    """Serialize with the existing viewer publishers and preserve their fields."""
    if not output.is_absolute() or not output.is_dir() or output.is_symlink():
        raise ValueError("Require an existing absolute ordinary viewer directory")
    with (output / ".original-methods-snapshot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _publish_locked(acceptance, output, receipt, cohort)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--cohort",
        choices=(
            "rtc",
            "initialized_full_expo",
            "matched_initialized_full_expo",
            "trained_full_expo",
            "continued_full_expo",
        ),
        default="rtc",
    )
    args = parser.parse_args()
    report = publish(args.acceptance, args.output_dir, args.receipt, args.cohort)
    print(json.dumps({"status": report["status"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
