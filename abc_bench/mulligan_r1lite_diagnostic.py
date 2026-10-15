"""Canonical process boundary for the sealed R1 adapter diagnostic; no learner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

MODE = "r1lite-native-diagnostic"
WORKER_SECONDS = 1800
PARENT_SECONDS = 1900
CHILD_CLEANUP_SECONDS = 80
KILL_WAIT_SECONDS = 2
WORKSPACE = Path("/home/user/aditya")
QUALIFICATION_SHA256 = (
    "a0677569bbc52f5e86c2b9721fd857a07248a9bf1cfaa04802528bbd1be15866"
)
TEN_QUALIFICATION_SHA256 = (
    "bbd07cc8ed7897bb91f90382739238c07c782755f956b4aa7c37b9b7fa9dd338"
)
BOUNDARY_SHA256 = "89c87470826b1c6960288d9bf177fc78f662458c80aaa3cc593c15f914743001"
TASK = "r1lite_remote_into_tray"
EMBODIMENT = "r1lite_2025"


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def owned(value: str) -> Path:
    path = Path(value)
    require(
        path.is_absolute() and path.resolve().is_relative_to(WORKSPACE),
        "Diagnostic paths must stay under /home/user/aditya",
    )
    return path


def pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def verify(record: dict) -> Path:
    require(
        set(record) == {"path", "bytes", "sha256"}, "Use a complete diagnostic file pin"
    )
    path = owned(record["path"])
    require(path.is_file() and pin(path) == record, f"Diagnostic input changed: {path}")
    return path


def qualification(config: dict) -> tuple[Path, dict, dict]:
    manifest_path = verify(config["qualification_manifest"])
    require(
        config["qualification_manifest"]["sha256"]
        == (
            TEN_QUALIFICATION_SHA256
            if config["case"] == "ten_slots"
            else QUALIFICATION_SHA256
        ),
        "Only the independently reviewed native diagnostic is admitted",
    )
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Invalid diagnostic source manifest path",
        )
        verify({**entry, "path": str(root / relative)})
    plan = json.loads((root / "QUALIFICATION_PLAN.json").read_text())
    require(
        plan["boundary_source_manifest"]["sha256"] == BOUNDARY_SHA256,
        "Only the CPU-qualified v3 boundary is admitted",
    )
    boundary = json.loads(verify(plan["boundary_source_manifest"]).read_text())
    for item in plan["input_pins"] + [
        plan["boundary_wheel"],
        plan["cpu_qualification"],
    ]:
        verify(item)
    for seed in plan["cases"][config["case"]]["seeds"]:
        entry = plan["oracles"][str(seed)]
        verify({key: entry[key] for key in ("path", "bytes", "sha256")})
    return root, plan, boundary


def load_config(path: Path) -> dict:
    """Validate fixed source and runtime payloads without importing either runtime."""
    config = json.loads(path.read_text())
    require(
        set(config)
        == {
            "schema_version",
            "mode",
            "python",
            "case",
            "timeout_s",
            "qualification_manifest",
            "runtimes",
        },
        "Use only explicit R1 diagnostic config fields",
    )
    require(
        type(config["schema_version"]) is int
        and config["schema_version"] == 1
        and config["mode"] == MODE,
        "Use the exact R1 diagnostic profile",
    )
    require(
        config["case"] in ("one_slot", "two_slots", "ten_slots"),
        "Select one_slot, two_slots or ten_slots",
    )
    require(
        type(config["timeout_s"]) is int and config["timeout_s"] == WORKER_SECONDS,
        "R1 diagnostic worker timeout must be 1800 seconds",
    )
    require(
        set(config["runtimes"]) == {"learner", "native"}, "Pin both isolated runtimes"
    )
    _, _, boundary = qualification(config)
    for runtime in config["runtimes"].values():
        require(
            set(runtime) == {"python", "package_dir"},
            "Pin runtime interpreter and installed package directory",
        )
        verify(runtime["python"])
        package = owned(runtime["package_dir"])
        require(package.is_dir(), "Installed boundary package is absent")
        for entry in boundary["files"]:
            prefix = "src/sadhana_mulligan_boundary/"
            if entry["path"].startswith(prefix):
                verify({**entry, "path": str(package / entry["path"][len(prefix) :])})
    require(
        config["python"] == config["runtimes"]["learner"]["python"]["path"],
        "Worker child must use the pinned learner-side interpreter",
    )
    return config


def check_parent(
    job: dict, config: dict, effective_timeout: float | None = None
) -> None:
    require(
        config.get("mode") == MODE and config.get("timeout_s") == WORKER_SECONDS,
        "Parent cleanup applies only to the fixed R1 diagnostic",
    )
    require(
        job.get("task") == TASK and job.get("embodiment") == EMBODIMENT,
        "R1 diagnostic task and embodiment must be explicit",
    )
    require(
        type(job.get("max_wall_s")) is int and job["max_wall_s"] == PARENT_SECONDS,
        "R1 diagnostic parent timeout must be 1900 seconds",
    )
    if effective_timeout is not None:
        require(
            type(effective_timeout) in (int, float)
            and math.isfinite(effective_timeout)
            and effective_timeout >= PARENT_SECONDS,
            "Effective R1 diagnostic parent timeout must include 100 seconds cleanup",
        )


def verify_runtime(config: dict, plan: dict) -> dict:
    """Inspect installed metadata and locations with bounded stdlib-only children."""
    code = (
        "import json,sys\nfrom importlib.metadata import distribution,version\n"
        "d=distribution('sadhana-mulligan-boundary')\n"
        "names=sys.argv[1:]\n"
        "result={'python':list(sys.version_info[:2]),'prefix':sys.prefix,"
        "'package_dir':str(d.locate_file('sadhana_mulligan_boundary').resolve()),"
        "'versions':{name:version(name) for name in names}}\n"
        "if 'sadhana-r1lite' in names:\n"
        " import hashlib\n"
        " p=distribution('sadhana-r1lite').locate_file('sadhana_r1lite/env.py')\n"
        " result['native_env_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "print(json.dumps(result))\n"
    )
    result = {}
    for name, runtime in config["runtimes"].items():
        expected = (
            {"numpy": "2.2.6"}
            if name == "learner"
            else plan["fixed_settings"]["native_versions"]
        )
        expected = {**expected, "sadhana-mulligan-boundary": "0.1.2"}
        probe = subprocess.run(
            [runtime["python"]["path"], "-I", "-B", "-c", code, *expected],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        observed = json.loads(probe.stdout)
        package = Path(runtime["package_dir"]).resolve()
        require(
            observed["python"] == ([3, 12] if name == "learner" else [3, 11])
            and observed["versions"] == expected
            and Path(observed["package_dir"]) == package
            and package.is_relative_to(Path(observed["prefix"]).resolve()),
            f"{name} runtime is not the pinned normally installed boundary",
        )
        if name == "native":
            require(
                observed["native_env_sha256"]
                == plan["fixed_settings"]["native_env_sha256"],
                "Installed native task environment differs",
            )
        result[name] = observed
    return result


def command_for(config: dict, source: Path, settings: Path, output: Path) -> list[str]:
    return [
        config["python"],
        "-I",
        "-u",
        "-B",
        str(source / "qualify_native.py"),
        "--plan",
        str(source / "QUALIFICATION_PLAN.json"),
        "--case",
        config["case"],
        "--settings",
        str(settings),
        "--output",
        str(output),
    ]


def run(config: dict, output: Path) -> dict:
    from abc_bench.runner import write_json
    from abc_bench.subset import require_subset_lease

    started = time.monotonic()
    # The wrapper retains canonical GPU visibility to prove its lease. Only its
    # stdlib probes and native diagnostic child receive empty GPU visibility.
    require_subset_lease()
    source, plan, _ = qualification(config)
    runtime = verify_runtime(config, plan)
    owned(str(output)).mkdir(parents=True, exist_ok=True)
    diagnostic = output / "diagnostic"
    require(
        not diagnostic.exists(), "Diagnostic output must be a fresh child directory"
    )
    settings = json.loads(
        (
            source
            / (
                "ONE_SLOT_SETTINGS.json"
                if config["case"] == "one_slot"
                else "TEN_SLOTS_SETTINGS.json"
                if config["case"] == "ten_slots"
                else "TWO_SLOTS_SETTINGS.json"
            )
        ).read_text()
    )
    settings.update(
        native_python=config["runtimes"]["native"]["python"]["path"],
        output_dir=str(diagnostic / "native"),
    )
    settings_path = output / "diagnostic-settings.json"
    require(not settings_path.exists(), "Resolved diagnostic settings already exist")
    write_json(settings_path, settings)
    settings_path.chmod(0o444)
    command = command_for(config, source, settings_path, diagnostic)
    result = {
        "schema_version": 1,
        "status": "running",
        "mode": MODE,
        "task": TASK,
        "embodiment": EMBODIMENT,
        "case": config["case"],
        "metrics": {},
        "experiment_accepted": False,
        "claim_scope": "Native adapter diagnostic only; no learner score or scientific acceptance",
        "command": command,
        "runtime": runtime,
        "settings": pin(settings_path),
        "qualification_manifest": config["qualification_manifest"],
        "descendant_cleanup_owner": "canonical runner with explicit R1 cleanup grace",
    }
    write_json(output / "config.json", config)
    receipt = output / "receipt.json"

    def cancelled(signum, frame):
        raise InterruptedError(f"R1 diagnostic cancelled by signal {signum}")

    previous = signal.signal(signal.SIGTERM, cancelled)
    process = None
    cleanup_failure = None
    try:
        write_json(receipt, result)
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        with (output / "diagnostic.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            result["diagnostic_pid"] = process.pid
            write_json(receipt, result)
            code = process.wait(
                timeout=max(0.0, WORKER_SECONDS - (time.monotonic() - started))
            )
        result["diagnostic_exit_code"] = code
        require(code == 0, f"Native diagnostic exited {code}")
        report_path = diagnostic / "RESULT.json"
        report = json.loads(report_path.read_text())
        require(
            report["status"] == "passed_adapter_diagnostic_pending_canonical_acceptance"
            and report["case"] == config["case"]
            and not report["error"]
            and not report["cleanup_error"],
            "Diagnostic did not pass its declared checks",
        )
        require(
            report["settings"] == pin(settings_path)
            and report["operator"] == pin(source / "qualify_native.py")
            and report["plan"] == pin(source / "QUALIFICATION_PLAN.json"),
            "Diagnostic result provenance differs",
        )
        qualification(config)
        result.update(status="completed", diagnostic_result=pin(report_path))
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                result["status"] = "failed"
                result["cleanup_intervention"] = "diagnostic_SIGTERM"
                process.terminate()
                try:
                    process.wait(timeout=CHILD_CLEANUP_SECONDS)
                except subprocess.TimeoutExpired:
                    result["cleanup_intervention"] = "diagnostic_SIGKILL"
                    process.kill()
                    try:
                        process.wait(timeout=KILL_WAIT_SECONDS)
                    except subprocess.TimeoutExpired as error:
                        cleanup_failure = error
                        result["cleanup_error"] = (
                            "Diagnostic survived bounded kill/reap"
                        )
            result["direct_child_stopped"] = (
                process is None or process.poll() is not None
            )
            result["elapsed_s"] = time.monotonic() - started
            write_json(receipt, result)
        finally:
            signal.signal(signal.SIGTERM, previous)
        if cleanup_failure is not None:
            raise RuntimeError(
                "Diagnostic cleanup failed; canonical group cleanup remains required"
            ) from cleanup_failure
    return result
