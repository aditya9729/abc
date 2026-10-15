"""Publish accepted R1 Lite RTC and initialized EXPO cohorts into the local viewer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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
    for item in visual["files"]:
        checked(item)
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
            "Initialized EXPO requires clean completed canonical and worker receipts"
        )
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
        raise ValueError("Initialized native or media cohort is incomplete")
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
            raise ValueError("Initialized native/media evidence disagrees")
    return evidence, observations


def _publish_locked(
    acceptance: Path, output: Path, receipt: Path, cohort: str = "rtc"
) -> dict:
    """Copy all scored videos and add one result field; preserve historical data."""
    if not all(p.is_absolute() for p in (acceptance, output, receipt)):
        raise ValueError("Use absolute acceptance, viewer and receipt paths")
    if not output.is_dir() or output.is_symlink() or receipt.exists():
        raise ValueError("Require an existing ordinary viewer and a fresh receipt")
    accepted = json.loads(acceptance.read_text())
    if cohort == "initialized_full_expo":
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
            else "Accepted initialized full EXPO DEV20 with all20 complete videos; historical cohorts preserved"
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
        "--cohort", choices=("rtc", "initialized_full_expo"), default="rtc"
    )
    args = parser.parse_args()
    report = publish(args.acceptance, args.output_dir, args.receipt, args.cohort)
    print(json.dumps({"status": report["status"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
