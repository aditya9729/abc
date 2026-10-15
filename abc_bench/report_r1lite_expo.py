"""Publish accepted R1 Lite RTC measurements into the existing local viewer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
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


def _publish_locked(acceptance: Path, output: Path, receipt: Path) -> dict:
    """Copy all scored videos and add one result field; preserve historical data."""
    if not all(p.is_absolute() for p in (acceptance, output, receipt)):
        raise ValueError("Use absolute acceptance, viewer and receipt paths")
    if not output.is_dir() or output.is_symlink() or receipt.exists():
        raise ValueError("Require an existing ordinary viewer and a fresh receipt")
    accepted = json.loads(acceptance.read_text())
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
    episodes = observations["episodes"]
    if (
        [row["seed"] for row in episodes] != list(range(9100, 9120))
        or observations["incomplete"] != 0
        or sum(row["success"] for row in episodes) != observations["successes"]
        or observations["full_agent_restore_marker"] != ["2000"]
        or accepted["canonical_status"] != "failed"
        or accepted["worker_status"] != "failed"
    ):
        raise ValueError(
            "Require the declared RTC2000 cohort and preserved supervision failure"
        )
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
        if ending not in ("success", "failure", "timeout") or row["success"] != (
            ending == "success"
        ):
            raise ValueError("Native ending disagrees with the scored outcome")
        rows.append({**row, "outcome": "native_" + ending, "video": name})
    timestamp = datetime.now(timezone.utc).isoformat()
    base["r1lite"] = {
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
        "scope": "Accepted R1 Lite RTC DEV20 with all20 complete videos",
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
        "status": "accepted_rtc_results_and_all20_videos_published",
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


def publish(acceptance: Path, output: Path, receipt: Path) -> dict:
    """Serialize with the existing viewer publishers and preserve their fields."""
    if not output.is_absolute() or not output.is_dir() or output.is_symlink():
        raise ValueError("Require an existing absolute ordinary viewer directory")
    with (output / ".original-methods-snapshot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _publish_locked(acceptance, output, receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = publish(args.acceptance, args.output_dir, args.receipt)
    print(json.dumps({"status": report["status"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
