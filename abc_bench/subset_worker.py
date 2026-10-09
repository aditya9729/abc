"""Supervise the original ABC host and separately installed ResFiT learner."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from abc_bench.subset import require_subset_lease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_subset_lease()
    config = json.loads(args.config.read_text())
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Expected subset supervision schema 1")
    python = Path(config["learner_python"])
    if not python.is_absolute() or not python.is_file():
        raise ValueError("An existing absolute learner interpreter is required")
    learner_timeout = config.get("learner_timeout_s", 3000)
    if (
        type(learner_timeout) not in (int, float)
        or not math.isfinite(learner_timeout)
        or not 0 < learner_timeout <= 7200
    ):
        raise ValueError("learner_timeout_s must be finite, positive and at most 7200")
    for name in ("host_arguments", "learner_arguments"):
        if not isinstance(config.get(name), list) or any(
            type(x) is not str for x in config[name]
        ):
            raise ValueError(f"{name} must contain only strings")
        if any(x.startswith(("--output", "--socket")) for x in config[name]):
            raise ValueError("Supervisor owns socket and output paths")
    args.output.mkdir(parents=True, exist_ok=True)
    private = args.output / "rpc"
    private.mkdir(mode=0o700)
    socket = private / "host.sock"
    host_out, learner_out = args.output / "host", args.output / "learner"
    host_out.mkdir()
    learner_out.mkdir()
    host_command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "abc_author_subset.host",
        "--mode",
        "serve",
        *config["host_arguments"],
        "--socket",
        str(socket),
        "--output",
        str(host_out),
    ]
    learner_command = [
        str(python),
        "-I",
        "-B",
        "-m",
        "sadhana_resfit_author_abc.diagnostic",
        *config["learner_arguments"],
        "--socket",
        str(socket),
        "--output",
        str(learner_out),
    ]
    started = time.monotonic()
    host = learner = None
    with (
        (args.output / "host.log").open("w") as host_log,
        (args.output / "learner.log").open("w") as learner_log,
    ):
        try:
            # Both children remain in the root-owned process group.
            host = subprocess.Popen(
                host_command, stdout=host_log, stderr=subprocess.STDOUT
            )
            ready_deadline = started + 300
            while not (host_out / "ready.json").is_file():
                if host.poll() is not None:
                    raise RuntimeError(
                        f"ABC host exited {host.returncode}; see host.log"
                    )
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError(
                        "ABC host did not become ready within 300 seconds"
                    )
                time.sleep(0.1)
            learner = subprocess.Popen(
                learner_command, stdout=learner_log, stderr=subprocess.STDOUT
            )
            code = learner.wait(timeout=learner_timeout)
            if code != 0:
                raise RuntimeError(
                    f"Original ResFiT learner exited {code}; see learner.log"
                )
            host_code = host.wait(timeout=30)
            if host_code != 0:
                raise RuntimeError(f"ABC host exited {host_code}; see host.log")
            result = json.loads((learner_out / "receipt.json").read_text())
            if result.get("status") != "completed":
                raise RuntimeError(
                    "Learner did not complete its declared diagnostic stage"
                )
            result["elapsed_seconds"] = time.monotonic() - started
            result["supervision"] = {
                "host_command": host_command,
                "learner_command": learner_command,
            }
            (args.output / "receipt.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        finally:
            for child in (learner, host):
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()


if __name__ == "__main__":
    main()
