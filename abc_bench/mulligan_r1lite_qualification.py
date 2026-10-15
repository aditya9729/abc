"""Fixed short original-trainer diagnostics; never a full-training admission."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from abc_bench import mulligan_r1lite_training as training
from abc_bench.mulligan_r1lite_diagnostic import (
    EMBODIMENT,
    TASK,
    owned,
    pin,
    require,
    verify,
)

MODE = "r1lite-original-qualification"
WORKER_SECONDS = 1800
PARENT_SECONDS = 1900
OPERATOR_MANIFEST_PATH = Path(
    "/home/user/aditya/RL/builds/other-methods-plan-20261015/"
    "original-qualification-operator-v2/SOURCE_MANIFEST.json"
)
OPERATOR_MANIFEST_SHA256 = (
    "18c7d3af2c1626e7bb56c131b14ef65010e0186d0af7ee8d04338b10cf72b8a3"
)
PHASES = {
    "update_restore": {"steps": 2, "save_every": 1, "eval_freq": "null"},
    "timing_monitor": {"steps": 2000, "save_every": 1000, "eval_freq": "2000"},
}
GATES = {
    "installed_author",
    "native_one",
    "native_two",
    "native_ten",
    "original_cli_parser",
}


def operator_source() -> Path:
    manifest = pin(OPERATOR_MANIFEST_PATH)
    require(
        manifest["sha256"] == OPERATOR_MANIFEST_SHA256,
        "Qualification inspector source changed",
    )
    for row in json.loads(verify(manifest).read_text())["files"]:
        relative = Path(row["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Invalid inspector source path",
        )
        verify({**row, "path": str(OPERATOR_MANIFEST_PATH.parent / relative)})
    return OPERATOR_MANIFEST_PATH.parent / "inspect_model.py"


def parent_checkpoint(config: dict, inputs: dict) -> Path | None:
    if config["stage"] == "idql":
        require(config["parent"] is None, "Fresh IDQL diagnostic has no parent")
        return None
    receipt = json.loads(verify(config["parent"]).read_text())
    require(
        receipt.get("status") == "accepted_r1_idql_diagnostic_parent"
        and receipt.get("policy_type") == "idql"
        and receipt.get("training_steps") == 2
        and receipt.get("task") == TASK
        and receipt.get("embodiment") == EMBODIMENT,
        "DIVL qualification needs an accepted two-update diagnostic IDQL parent",
    )
    expected = {
        "author_manifest": config["author_manifest"],
        "runtimes": config["runtimes"],
        "data": inputs["data"],
        "task": inputs["task"],
    }
    require(
        receipt["bindings"] == expected,
        "Diagnostic parent source/runtime/data/task differ",
    )
    require(
        set(receipt["evidence"]) == {"model_inspection", "canonical_cleanup"},
        "Diagnostic parent needs actual model and cleanup evidence",
    )
    for row in receipt["evidence"].values():
        verify(row)
    root = owned(receipt["checkpoint_root"])
    require(
        root.is_dir() and root.name == "final_model",
        "Diagnostic parent must be final_model",
    )
    require(bool(receipt["checkpoint_files"]), "Empty diagnostic parent inventory")
    training.verified_tree(root, receipt["checkpoint_files"])
    return root


def verify_admission(config: dict, inputs: dict) -> None:
    receipt = json.loads(verify(config["admission"]).read_text())
    require(
        receipt.get("schema") == "sadhana.mulligan_r1_qualification_admission/1"
        and receipt.get("status") == "accepted_for_short_original_qualification"
        and receipt.get("stage") == config["stage"]
        and receipt.get("phase") == config["phase"]
        and receipt.get("task") == TASK
        and receipt.get("embodiment") == EMBODIMENT
        and receipt.get("timeout_s") == WORKER_SECONDS,
        "Root must separately admit the bounded diagnostic, not full training",
    )
    gates = GATES | (
        {"update_restore"} if config["phase"] == "timing_monitor" else set()
    )
    require(
        set(receipt["evidence"]) == gates,
        "Required diagnostic qualification evidence missing",
    )
    for row in receipt["evidence"].values():
        verify(row)
    expected = {
        "author_manifest": config["author_manifest"],
        "runtimes": config["runtimes"],
        "data": inputs["data"],
        "task": inputs["task"],
        "parent": config["parent"],
        "operator_manifest": pin(OPERATOR_MANIFEST_PATH),
    }
    require(receipt["bindings"] == expected, "Diagnostic admission bindings differ")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    require(
        set(config)
        == {
            "schema_version",
            "mode",
            "stage",
            "phase",
            "python",
            "timeout_s",
            "author_manifest",
            "runtimes",
            "admission",
            "parent",
        },
        "Use only fixed short qualification fields; no arbitrary arguments",
    )
    require(
        type(config["schema_version"]) is int
        and config["schema_version"] == 1
        and config["mode"] == MODE
        and config["stage"] in ("idql", "divl")
        and config["phase"] in PHASES,
        "Unknown original qualification profile",
    )
    require(
        type(config["timeout_s"]) is int and config["timeout_s"] == WORKER_SECONDS,
        "Short qualification has a conservative 1800-second cap",
    )
    root, _, inputs = training.source_inputs(config)
    training.verify_installation_files(config, root / "source", inputs["boundary"])
    operator_source()
    parent_checkpoint(config, inputs)
    verify_admission(config, inputs)
    return config


def check_parent(
    job: dict, config: dict, effective_timeout: float | None = None
) -> None:
    require(
        config.get("mode") == MODE
        and config.get("timeout_s") == WORKER_SECONDS
        and job.get("task") == TASK
        and job.get("embodiment") == EMBODIMENT,
        "Use explicit R1 qualification task and fixed worker bound",
    )
    require(
        type(job.get("max_wall_s")) is int and job["max_wall_s"] == PARENT_SECONDS,
        "Qualification parent must have the fixed 1900-second cap",
    )
    if effective_timeout is not None:
        require(
            type(effective_timeout) in (int, float)
            and math.isfinite(effective_timeout)
            and min(job["max_wall_s"], effective_timeout) >= PARENT_SECONDS,
            "Effective qualification parent must retain 100 seconds cleanup",
        )


def command_for(
    config: dict, recipe: dict, output: Path, parent: Path | None
) -> list[str]:
    """Only diagnostic steps/eval/save and operational names differ from the sealed recipe."""
    phase = PHASES[config["phase"]]
    command = training.command_for(config, recipe, output, parent)
    changes = {
        "--training.training_steps": str(phase["steps"]),
        "--training.periodic_checkpoint_freq": str(phase["save_every"]),
        "--training.resume_checkpoint_freq": str(phase["save_every"]),
        "--training.save_final_model": "True",
        "--eval.freq": phase["eval_freq"],
        "--wandb.run_name": f"r1-diagnostic-{config['stage']}-{config['phase']}-{output.name}",
    }
    arguments = {arg.split("=", 1)[0]: arg.split("=", 1)[1] for arg in command[6:]}
    require(len(arguments) == len(command[6:]), "Duplicate sealed original argument")
    arguments.update(changes)
    return command[:6] + [f"{key}={value}" for key, value in arguments.items()]


def run(config: dict, output: Path) -> dict:
    from abc_bench.runner import write_json
    from abc_bench.subset import require_subset_lease

    started = time.monotonic()
    require_subset_lease()
    require(os.getpgrp() == os.getpid(), "Qualification must lead its canonical group")
    root, recipe, inputs = training.source_inputs(config)
    training.verify_installation_files(config, root / "source", inputs["boundary"])
    inspector = operator_source()
    verify_admission(config, inputs)
    parent = parent_checkpoint(config, inputs)
    runtime = training.verify_runtime(config, inputs)
    owned(str(output)).mkdir(parents=True, exist_ok=True)
    diagnostic = output / "qualification"
    diagnostic.mkdir(exist_ok=False)
    for name in ("tmp", "wandb"):
        (diagnostic / name).mkdir()
    settings = {
        **inputs["settings"],
        "native_python": config["runtimes"]["native"]["python"]["path"],
        "output_dir": str(diagnostic / "native"),
    }
    settings_path = diagnostic / "settings.json"
    write_json(settings_path, settings)
    settings_path.chmod(0o444)
    command = command_for(config, recipe, diagnostic, parent)
    inspect_command = [
        config["python"],
        "-I",
        "-u",
        "-B",
        str(inspector),
        "--root",
        str(diagnostic / "checkpoints"),
        "--stage",
        config["stage"],
        "--phase",
        config["phase"],
        "--author-log",
        str(diagnostic / "author.log"),
        "--output",
        str(diagnostic / "MODEL_INSPECTION.json"),
    ]
    if parent is not None:
        inspect_command.extend(["--parent", str(parent)])
    result = {
        "schema_version": 1,
        "status": "running",
        "mode": MODE,
        "stage": config["stage"],
        "phase": config["phase"],
        "task": TASK,
        "embodiment": EMBODIMENT,
        "metrics": {},
        "experiment_accepted": False,
        "claim_scope": "Short original update/save/loader and optional TRAIN monitor; no benchmark or exact-resume acceptance",
        "runtime": runtime,
        "command": command,
        "inspection_command": inspect_command,
        "settings": pin(settings_path),
        "admission": config["admission"],
        "parent": config["parent"],
        "operator_manifest": pin(OPERATOR_MANIFEST_PATH),
        "exact_resume_executed": False,
        "watchdog_note": "1800 seconds is a conservative diagnostic cap, not a measured R1 duration",
    }
    write_json(output / "config.json", config)
    receipt = output / "receipt.json"
    process = None
    known = {}

    def cancelled(signum, frame):
        raise InterruptedError(f"Original qualification cancelled by signal {signum}")

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        write_json(receipt, result)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONHOME", "WANDB_DISABLED")
        }
        environment.update(
            SADHANA_MULLIGAN_SETTINGS=str(settings_path),
            MUJOCO_GL="osmesa",
            WANDB_MODE="offline",
            WANDB_DIR=str(diagnostic / "wandb"),
            TMPDIR=str(diagnostic / "tmp"),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            IMAGEIO_FFMPEG_NO_PREVENT_SIGINT="1",
        )
        deadline = started + WORKER_SECONDS
        for name, child_command in (
            ("author", command),
            ("inspection", inspect_command),
        ):
            require(
                time.monotonic() < deadline,
                "Diagnostic deadline passed before child launch",
            )
            child_started = time.monotonic()
            child_env = (
                environment
                if name == "author"
                else {**environment, "CUDA_VISIBLE_DEVICES": ""}
            )
            with (diagnostic / f"{name}.log").open("xb") as log:
                process = subprocess.Popen(
                    child_command,
                    cwd=root / "source",
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                result[f"{name}_pid"] = process.pid
                while True:
                    training.observe_descendants(os.getpgrp(), known)
                    result["observed_descendants"] = list(known.values())
                    write_json(receipt, result)
                    code = process.poll()
                    if code is not None:
                        break
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(child_command, WORKER_SECONDS)
                    time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
            result[f"{name}_exit_code"] = code
            result[f"{name}_wall_seconds"] = time.monotonic() - child_started
            require(code == 0, f"Original qualification {name} exited {code}")
        report_path = diagnostic / "MODEL_INSPECTION.json"
        report = json.loads(report_path.read_text())
        require(
            report["status"] == "passed_short_model_inspection_pending_root_acceptance"
            and report["stage"] == config["stage"]
            and report["phase"] == config["phase"]
            and report["steps"] == PHASES[config["phase"]]["steps"]
            and report["artifact_accepted"] is False,
            "Short model inspection did not pass",
        )
        training.source_inputs(config)
        training.verify_installation_files(config, root / "source", inputs["boundary"])
        operator_source()
        parent_checkpoint(config, inputs)
        result.update(status="completed", model_inspection=pin(report_path))
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
            clean = training.cleanup(
                process,
                os.getpgrp(),
                known,
                terminate_group=result["status"] != "completed",
            )
            result["cleanup"] = clean
            if not clean["clean"] or clean["cleanup_intervention"]:
                result["status"] = "failed"
            result["elapsed_s"] = time.monotonic() - started
            write_json(receipt, result)
        finally:
            signal.signal(signal.SIGTERM, previous)
        if result["status"] != "completed" and not result.get("error"):
            raise RuntimeError(
                "Qualification cleanup failed; canonical group cleanup remains required"
            )
    return result
