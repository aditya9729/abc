"""Run bounded ABC evaluations and publish measured dashboard receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = (
    Path(os.environ.get("ABC_BENCH_REPO", str(Path(__file__).resolve().parents[1])))
    .expanduser()
    .resolve()
)
RESULTS = ROOT / "outputs" / "bench"


MAX_EVAL_HORIZON = 3540


def comparison_plan(horizon: int) -> dict[str, Any]:
    """One resolved horizon controls command, step reservation, and claim label."""
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= MAX_EVAL_HORIZON
    ):
        raise ValueError(
            f"Evaluation horizon must be an integer in [1, {MAX_EVAL_HORIZON}]"
        )
    return {
        "horizon": horizon,
        "maximum_steps": 3 * 3 * horizon,
        "method_fidelity": f"matched {horizon}-step, three-seed adaptation pilot",
    }


class RunCancelled(RuntimeError):
    """The coordinator cancelled a bounded experiment."""


def execute_command(command: list[str], log_path: Path, timeout: float) -> int:
    """Reap the job's process group on timeout, SIGTERM or keyboard cancellation."""

    def cancelled(signum: int, frame: Any) -> None:
        raise RunCancelled(f"Received cancellation signal {signum}")

    old_handler = signal.signal(signal.SIGTERM, cancelled)
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "0",
                    "MUJOCO_GL": "egl",
                    "PYTHONUNBUFFERED": "1",
                },
                start_new_session=True,
            )
            return process.wait(timeout=timeout)
    finally:
        # Block further SIGTERM delivery until cleanup finishes.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # The leader can exit before descendants, including workers which ignore TERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        finally:
            signal.signal(signal.SIGTERM, old_handler)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(summary: dict[str, Any], *, phase: str) -> dict[str, Any]:
    """Keep missing latency unavailable; do not convert missing data to zero."""
    import numpy as np

    worlds = summary["worlds"]
    if not worlds or summary["num_worlds"] != len(worlds):
        raise ValueError("Missing or inconsistent world records")
    successes = sum(bool(w["success"]) for w in worlds)
    if successes != summary["num_success"]:
        raise ValueError("Success count disagrees with world records")
    latency = [
        1000 * float(c["current_chunk_infer_s"])
        for w in worlds
        for c in w.get("chunk_metrics", [])
        if c.get("current_chunk_infer_s") is not None
    ]
    if any(not np.isfinite(x) or x < 0 for x in latency):
        raise ValueError("Invalid inference latency")
    metrics = {
        "successes": successes if phase == "benchmark" else None,
        "episodes": len(worlds) if phase == "benchmark" else None,
        "reward": summary.get("mean_reward"),
        "simulation_steps": sum(int(w["steps"]) for w in worlds),
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)) if latency else None,
            "p95": float(np.percentile(latency, 95)) if latency else None,
        },
        "latency_samples": len(latency),
        "latency_scope": "upstream logged current-chunk inference; includes cold calls; terminal chunks may be omitted",
        "completion_time_seconds": None,
    }
    return metrics


def partial_log_metrics(log: Path) -> dict[str, Any] | None:
    """Recover only fully completed worlds from upstream's explicit stdout records."""
    import numpy as np

    completed: dict[int, tuple[bool, int]] = {}
    latency: dict[int, list[float]] = {}
    for line in log.read_text().splitlines():
        done = re.match(r"world=(\d+) done success=(True|False).* steps=(\d+)$", line)
        if done:
            world = int(done[1])
            if world in completed:
                raise ValueError("Duplicate completed world in evaluation log")
            completed[world] = (done[2] == "True", int(done[3]))
        chunk = re.match(r"world=(\d+) chunk=\d+ infer=(\d+)ms", line)
        if chunk:
            latency.setdefault(int(chunk[1]), []).append(float(chunk[2]))
    if not completed:
        return None
    samples = [value for world in completed for value in latency.get(world, [])]
    return {
        "successes": sum(success for success, steps in completed.values()),
        "episodes": len(completed),
        "simulation_steps": sum(steps for success, steps in completed.values()),
        "reward": None,
        "completion_time_seconds": None,
        "latency_ms": {
            "p50": float(np.percentile(samples, 50)) if samples else None,
            "p95": float(np.percentile(samples, 95)) if samples else None,
        },
        "latency_samples": len(samples),
        "latency_scope": "rounded upstream chunk stdout; completed worlds only; terminal chunks may be omitted",
        "evidence_scope": "partial campaign; completed-world log records only; unfinished worlds excluded",
    }


def publish(receipt: dict[str, Any], results: Path = RESULTS) -> None:
    """Serialize receipt publication so dashboard snapshots cannot lose a writer."""
    import fcntl

    results.mkdir(parents=True, exist_ok=True)
    with (results / ".publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _publish_unlocked(receipt, results)


def _publish_unlocked(receipt: dict[str, Any], results: Path) -> None:
    write_json(results / "runs" / (receipt["run_id"] + ".json"), receipt)
    runs = [
        json.loads(p.read_text()) for p in sorted((results / "runs").glob("*.json"))
    ]
    write_json(
        results / "results.json",
        {
            "schema_version": 1,
            "runs": runs,
            "embodiments": [
                {
                    "id": "native_yam",
                    "label": "Native bimanual YAM",
                    "status": "available",
                    "blockers": [],
                },
                {
                    "id": "r1lite",
                    "label": "Galaxea R1 Lite",
                    "status": "requires_adapter",
                    "blockers": [
                        "Controller, action, camera and task calibration required"
                    ],
                },
                {
                    "id": "r1pro",
                    "label": "Galaxea R1 Pro",
                    "status": "requires_adapter",
                    "blockers": [
                        "16-D dual-arm contract differs from the 14-D native policy"
                    ],
                },
            ],
        },
    )


def execution_provenance() -> dict[str, Any]:
    """Separate the accepted upstream pin from the local benchmark checkout."""
    manifest_path = RESULTS / "runtime_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    upstream = json.loads(manifest_bytes).get("upstream_commit")
    if not isinstance(upstream, str) or re.fullmatch(r"[0-9a-f]{40}", upstream) is None:
        raise ValueError("Campaign manifest requires a pinned upstream commit")
    return {
        "upstream_commit": upstream,
        "runtime_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "harness_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "harness_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "abc_bench").glob("*.py"))
        },
    }


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    import fcntl

    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / ".gpu-budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = RESULTS / "gpu_budget.json"
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else {"limit_seconds": 7200, "charged_seconds": 0.0, "gpu": 0}
        )
        remaining = ledger["limit_seconds"] - ledger["charged_seconds"]
        timeout = min(args.timeout_seconds, remaining)
        if timeout <= 0:
            raise ValueError("Authorized two-hour GPU budget exhausted")
        algorithm = getattr(args, "algorithm", "baseline")
        plan = comparison_plan(getattr(args, "eval_horizon", 1000))
        run_id = (
            algorithm
            + "-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )
        out = results / run_id
        out.mkdir()
        command = [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "eval_policy.py"),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--output-dir",
            str(out),
            "--task",
            args.task,
            "--num-worlds",
            str(args.worlds),
            "--num-chunks",
            str(args.chunks),
            "--seed",
            str(args.seed),
            "--no-fast-inference",
            "--no-rtc",
            "--log-every-chunk",
        ]
        if algorithm == "comparison":
            if args.qf3_state is None or args.resfit_state is None:
                raise ValueError("Comparison requires --qf3-state and --resfit-state")
            command = [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "abc_bench.paired_eval",
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--qf3-state",
                str(args.qf3_state.resolve()),
                "--resfit-state",
                str(args.resfit_state.resolve()),
                "--output",
                str(out / "paired_eval.json"),
                "--task",
                args.task,
                "--horizon",
                str(plan["horizon"]),
                "--seeds",
                "101,102,103",
                "--device",
                "cuda:0",
            ]
        elif algorithm != "baseline":
            timeout = min(timeout, 600.0)
            command = [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "abc_bench.update_smoke",
                "--algorithm",
                algorithm,
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--output",
                str(out / "update_smoke.json"),
                "--task",
                args.task,
                "--device",
                "cuda:0",
                "--seed",
                str(args.seed),
                "--max-seconds",
                str(timeout),
            ]
        if args.video and algorithm == "baseline":
            command += ["--save-video", "--video-every-n-actions", "15"]
        phase = (
            "benchmark"
            if algorithm == "comparison"
            or (algorithm == "baseline" and args.chunks >= 236)
            else "smoke"
        )
        receipt: dict[str, Any] = {
            "run_id": run_id,
            "algorithm": "Frozen ABC-DiT baseline",
            "embodiment": "native_yam",
            "task": args.task,
            "seed": args.seed,
            "phase": phase,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "metrics": {},
            "blockers": [],
            "method_fidelity": "upstream released checkpoint and evaluation entry point",
            "claim_scope": "small pilot; not a converged method comparison or transfer result",
            "budget": {
                "wall_seconds": timeout,
                "steps": args.worlds * args.chunks * 15,
            },
            "command": command,
            "artifacts": [
                {
                    "label": "Execution log",
                    "path": str((out / "run.log").relative_to(results)),
                }
            ],
        }
        if algorithm == "comparison":
            receipt["algorithm"] = "Matched native adaptation evaluation"
            receipt["method_fidelity"] = plan["method_fidelity"]
            receipt["budget"]["steps"] = plan["maximum_steps"]
            receipt["evaluation_horizon_steps"] = plan["horizon"]
            receipt["claim_scope"] = (
                "frozen base and four-update adaptations; not full paper or transfer comparison"
            )
        elif algorithm != "baseline":
            receipt["algorithm"] = (
                "QF3 output-adapter update smoke"
                if algorithm == "qf3"
                else "ResFiT frozen-feature update smoke"
            )
            receipt["method_fidelity"] = "declared adaptation; optimizer smoke only"
            receipt["budget"]["steps"] = 48
            receipt["claim_scope"] = (
                "actual pretrained policy and simulation; no converged method-performance comparison"
            )
        receipt.update(execution_provenance())
        publish(receipt, results)
        start = time.monotonic()
        # Reserve the full timeout before execution: interruption cannot overspend the shared ledger.
        ledger["charged_seconds"] += timeout
        ledger["active_parent_pid"] = os.getpid()
        write_json(ledger_path, ledger)
        try:
            code = execute_command(command, out / "run.log", timeout)
            if code:
                raise RuntimeError(
                    f"ABC evaluation failed with exit code {code}; see run.log"
                )
            summary_path = out / (
                "paired_eval.json"
                if algorithm == "comparison"
                else (
                    "summary.json" if algorithm == "baseline" else "update_smoke.json"
                )
            )
            summary = json.loads(summary_path.read_text())
            if algorithm == "comparison":
                receipt["metrics"] = {
                    "simulation_steps": sum(
                        r["metrics"]["simulation_steps"] for r in summary["runs"]
                    )
                }
                for index, record in enumerate(summary["runs"]):
                    child = {
                        **receipt,
                        **record,
                        "run_id": run_id + "-method-" + str(index),
                        "phase": "benchmark",
                        "status": "completed",
                        "seed": None,
                        "evaluation_seeds": [101, 102, 103],
                        "method_fidelity": plan["method_fidelity"]
                        + "; four-update adaptation; not full paper reproduction",
                        "evaluation_horizon_steps": plan["horizon"],
                    }
                    child["artifacts"] = [
                        {
                            "label": item["label"],
                            "path": str(out.relative_to(results) / item["path"]),
                        }
                        for item in record.get("artifacts", [])
                    ]
                    child["artifacts"].append(
                        {
                            "label": "Matched evaluation receipt",
                            "path": str(summary_path.relative_to(results)),
                        }
                    )
                    publish(child, results)
            elif algorithm == "baseline":
                receipt["metrics"] = summarize(summary, phase=phase)
            else:
                receipt["metrics"] = {
                    "successes": None,
                    "episodes": None,
                    "latency_ms": summary["latency_ms"],
                    "latency_scope": summary["latency_scope"],
                    "simulation_steps": summary["executed_steps"],
                    "gradient_finite": summary["gradient_finite"],
                    "zero_adapter_equivalence": summary["zero_adapter_equivalence"],
                    "parameter_delta_l2": summary["parameter_delta_l2"],
                    "updates": len(summary["updates"]),
                    "completion_time_seconds": None,
                }
                receipt["adaptations"] = summary["adaptations"]
                receipt["artifacts"].append(
                    {
                        "label": "Adaptation optimizer snapshot",
                        "path": str((out / "adaptation.pt").relative_to(results)),
                    }
                )
            receipt["status"] = "completed"
            receipt["summary_sha256"] = hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest()
            receipt["artifacts"].append(
                {
                    "label": "Upstream summary"
                    if algorithm == "baseline"
                    else "Update smoke evidence",
                    "path": str(summary_path.relative_to(results)),
                }
            )
            for video in out.glob("*.mp4"):
                receipt["artifacts"].append(
                    {"label": video.name, "path": str(video.relative_to(results))}
                )
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            subprocess.TimeoutExpired,
            KeyboardInterrupt,
        ) as error:
            receipt["status"] = (
                "cancelled"
                if isinstance(error, (RunCancelled, KeyboardInterrupt))
                else "failed"
            )
            receipt["blockers"] = [str(error)]
            if (
                algorithm == "baseline"
                and phase == "benchmark"
                and (out / "run.log").is_file()
            ):
                partial = partial_log_metrics(out / "run.log")
                if partial is not None:
                    receipt["metrics"] = partial
                    receipt["partial_log_sha256"] = hashlib.sha256(
                        (out / "run.log").read_bytes()
                    ).hexdigest()
                    receipt["status"] = "partial"
                    receipt["claim_scope"] = (
                        "interrupted campaign; only fully completed worlds counted; planned cohort incomplete"
                    )
        finally:
            elapsed = time.monotonic() - start
            ledger["charged_seconds"] += elapsed - timeout
            ledger.pop("active_parent_pid", None)
            write_json(ledger_path, ledger)
            receipt["metrics"]["elapsed_seconds"] = elapsed
            publish(receipt, results)
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "cache/bottles_75k.pt"
    )
    parser.add_argument(
        "--algorithm",
        choices=("baseline", "qf3", "resfit", "comparison"),
        default="baseline",
    )
    parser.add_argument(
        "--eval-horizon",
        type=int,
        default=1000,
        help="Matched comparison action horizon, maximum 3540",
    )
    parser.add_argument("--qf3-state", type=Path)
    parser.add_argument("--resfit-state", type=Path)
    parser.add_argument("--task", default="put_plastic_bottles_in_bin")
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.worlds < 1 or args.chunks < 1 or args.timeout_seconds <= 0:
        parser.error("worlds, chunks and timeout must be positive")
    try:
        comparison_plan(args.eval_horizon)
    except ValueError as error:
        parser.error(str(error))
    if not args.checkpoint.is_file():
        parser.error("checkpoint missing; run prepare.py --checkpoint first")
    receipt = run_baseline(args)
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
