"""Launch one small author-method job through the shared GPU 0 reservation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench.resources import inspect_reserved_gpu

ALLOWED_MODULES = {
    "abc_author_subset.host",
    "abc_bench.subset_worker",
    "nrh.qf3_training",
}


def require_subset_lease() -> None:
    """Admit only descendants of the live coordinator holding the GPU 0 lock."""
    from abc_bench.runner import RESULTS

    ledger = json.loads((RESULTS / "gpu_budget.json").read_text())
    owner = ledger.get("active_parent_pid")
    if type(owner) is not int or owner <= 1:
        raise RuntimeError("No active ABC subset coordinator")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ledger.get("active_gpu_uuid"):
        raise RuntimeError("GPU visibility differs from the coordinator reservation")
    if os.environ.get("MUJOCO_GL") != "disable" or ledger.get("gpu") != 0:
        raise RuntimeError("The subset requires the reserved GPU 0 and MJWarp")
    current = os.getppid()
    for _ in range(16):
        if current == owner:
            break
        if current <= 1:
            raise RuntimeError("This worker is not a descendant of the coordinator")
        stat = Path(f"/proc/{current}/stat").read_text()
        current = int(stat[stat.rfind(")") + 2 :].split()[1])
    else:
        raise RuntimeError("Coordinator ancestry exceeds the bounded process tree")
    with (RESULTS / ".gpu-budget.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
        raise RuntimeError("The coordinator GPU 0 lease is not held")


def load_job(path: Path) -> dict[str, Any]:
    """Require a finite job and a normal installed, explicitly selected worker."""
    job = json.loads(path.read_text())
    if (
        type(job.get("schema_version")) is not int
        or job["schema_version"] != 1
        or job.get("module") not in ALLOWED_MODULES
    ):
        raise ValueError("Use schema_version 1 and an author-subset worker module")
    python = Path(job["python"])
    if not python.is_absolute() or not python.is_file():
        raise ValueError("python must name an existing absolute interpreter")
    arguments = job.get("arguments")
    if not isinstance(arguments, list) or any(type(x) is not str for x in arguments):
        raise ValueError("arguments must be a list of strings")
    if any(x.startswith("--out") for x in arguments):
        raise ValueError("The coordinator owns the output directory")
    if not isinstance(job.get("label"), str) or not job["label"]:
        raise ValueError("A method and stage label is required")
    maximum = job.get("max_wall_s")
    if (
        type(maximum) not in (int, float)
        or not math.isfinite(maximum)
        or not 0 < maximum <= 7200
    ):
        raise ValueError("max_wall_s must be finite, positive and at most 7200")
    return job


def run_job(args: Any) -> dict[str, Any]:
    """Keep earlier charges; reap the whole owned group before releasing its lock."""
    from abc_bench.runner import RESULTS, execute_command, publish, write_json

    if args.training_config is None:
        raise ValueError("author-subset requires --training-config")
    job = load_job(args.training_config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / ".gpu-budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = RESULTS / "gpu_budget.json"
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("gpu") != 0 or any(
            key in ledger
            for key in ("active_parent_pid", "active_gpu_uuid", "active_visual_lease")
        ):
            raise RuntimeError("GPU 0 has an active or unresolved campaign owner")
        charged, limit = ledger["charged_seconds"], ledger["limit_seconds"]
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0
            for x in (charged, limit)
        ):
            raise ValueError(
                "The original ledger must have finite nonnegative charges and credit"
            )
        timeout = min(float(job["max_wall_s"]), float(args.timeout_seconds))
        if timeout <= 0 or not math.isfinite(timeout) or limit - charged < timeout:
            raise RuntimeError(
                "Insufficient finite GPU 0 admission credit for this job"
            )
        binding = inspect_reserved_gpu(0)
        run_id = (
            "author-subset-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )
        out = RESULTS / run_id
        out.mkdir()
        # The QF3 worker creates its output and refuses a pre-existing directory.
        worker_out = out / "qf3" if job["module"] == "nrh.qf3_training" else out
        command = [
            job["python"],
            "-I",
            "-u",
            "-B",
            "-m",
            job["module"],
            *job["arguments"],
            "--out" if job["module"] == "nrh.qf3_training" else "--output",
            str(worker_out),
        ]
        write_json(out / "job.json", job)
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "algorithm": job["label"],
            "phase": "small_subset",
            "status": "running",
            "task": "put_plastic_bottles_in_bin",
            "embodiment": "native_yam",
            "claim_scope": "small ABC-VLA transfer diagnostic; not full paper reproduction",
            "gpu": binding,
            "command": command,
            "resolved_job_sha256": hashlib.sha256(
                (out / "job.json").read_bytes()
            ).hexdigest(),
            "metrics": {},
            "artifacts": [
                {
                    "label": "Execution log",
                    "path": str((out / "run.log").relative_to(RESULTS)),
                }
            ],
            "blockers": [],
        }
        publish(receipt, RESULTS)
        ledger["charged_seconds"] = charged + timeout
        ledger["active_parent_pid"] = os.getpid()
        ledger["active_gpu_uuid"] = binding["uuid"]
        write_json(ledger_path, ledger)
        started = time.monotonic()
        try:
            code = execute_command(
                command, out / "run.log", timeout, gpu_uuid=binding["uuid"]
            )
            receipt["worker_exit_code"] = code
            if code != 0:
                raise RuntimeError(f"Worker exited {code}; see run.log")
            worker_path = worker_out / "receipt.json"
            worker = json.loads(worker_path.read_text())
            if worker.get("status") != "completed":
                raise RuntimeError("Worker did not finish its declared stage")
            receipt["metrics"] = worker.get("metrics", {})
            receipt["worker_receipt_sha256"] = hashlib.sha256(
                worker_path.read_bytes()
            ).hexdigest()
            receipt["artifacts"].append(
                {
                    "label": "Worker results",
                    "path": str(worker_path.relative_to(RESULTS)),
                }
            )
            receipt["status"] = "completed"
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.TimeoutExpired,
            KeyboardInterrupt,
        ) as error:
            receipt["status"] = "failed"
            receipt["blockers"] = [str(error)]
        finally:
            elapsed = time.monotonic() - started
            ledger["charged_seconds"] = charged + elapsed
            ledger.pop("active_parent_pid", None)
            ledger.pop("active_gpu_uuid", None)
            write_json(ledger_path, ledger)
            receipt["elapsed_seconds"] = elapsed
            publish(receipt, RESULTS)
        return receipt
