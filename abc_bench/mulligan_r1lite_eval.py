"""Canonical admission for the installed original Mulligan frozen DEV20 caller."""

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

MODE = "r1lite-frozen-eval"
MAX_WORKER_SECONDS = training.MAX_WORKER_SECONDS
PARENT_MARGIN_SECONDS = training.PARENT_MARGIN_SECONDS
PREPARATION = Path(
    "/home/user/aditya/RL/builds/other-methods-plan-20261015/frozen-eval-caller-preparation-v1"
)
RELEASE = Path(
    "/home/user/aditya/RL/builds/other-methods-plan-20261015/frozen-eval-caller-release-01/CPU_QUALIFICATION.json"
)
RELEASE_SHA256 = "7d2a0d3fe4129b2109c0bee36ceb1bcc361d053db753f8db6c39548b90ebc0cb"
CALLER_SOURCE_SHA256 = (
    "680f1af976c177c28da80b9bd7bacde25cde7ef4f1027ccd2297fb8d427c3ed9"
)
PERIODIC_INSPECTOR_MANIFEST = Path(
    "/home/user/aditya/RL/builds/other-methods-plan-20261015/"
    "mulligan-partial-checkpoint-inspector-preparation-v2-20261015/SOURCE_MANIFEST.json"
)
PERIODIC_INSPECTOR_SHA256 = (
    "fbbaa876d1525aafb92134805f176e4fb002d5feaa748d2ffb0c1b6bc5b87416"
)
GATES = {
    "installed_caller",
    "native_one",
    "native_two",
    "native_ten",
    "original_restore",
    "timing",
}
CALLER_MODULES = {
    "mulligan": (
        "training/evaluation.py",
        "utils/load_pretrained.py",
        "agents/idql.py",
        "training/normalization.py",
        "configs/policy.py",
        "utils/seeding.py",
    ),
    "sadhana_mulligan_boundary": (
        "vector.py",
        "settings.py",
        "protocol.py",
        "lifecycle.py",
        "pidfd.py",
    ),
}


def source_inputs(config: dict) -> dict:
    """Bind fixed source locations, installed caller bytes and original author inputs."""
    record = config["caller_release"]
    require(
        Path(record["path"]) == RELEASE and record["sha256"] == RELEASE_SHA256,
        "Use the exact installed caller CPU release; copied receipts cannot redirect inputs",
    )
    release = json.loads(verify(record).read_text())
    source_record = release["source_manifest"]
    require(
        Path(source_record["path"]) == PREPARATION / "SOURCE_MANIFEST.json"
        and source_record["sha256"] == CALLER_SOURCE_SHA256,
        "Caller source differs",
    )
    source = json.loads(verify(source_record).read_text())
    for row in source["prepared_files"] + source["inputs"]:
        verify(row)
    for row in release["files"]:
        verify(row)
    require(
        config["runtimes"]["learner"]["python"] == release["interpreter"]
        and config["python"] == release["interpreter"]["path"],
        "Use the exact normally installed qualified caller interpreter",
    )
    inventory_record = next(
        row
        for row in release["files"]
        if Path(row["path"]).name == "INSTALLED_SOURCE.json"
    )
    inventory = json.loads(verify(inventory_record).read_text())["files"]
    for row in inventory:
        verify(row)
    site = (
        Path(release["interpreter"]["path"]).parent.parent
        / "lib/python3.12/site-packages"
    )
    learner = config["runtimes"]["learner"]
    require(
        Path(learner["author_package_dir"]) == site / "mulligan"
        and Path(learner["package_dir"]) == site / "sadhana_mulligan_boundary",
        "Caller author and boundary packages must use the qualified installed paths",
    )
    observed = {
        str(p)
        for package in (
            "mulligan",
            "sadhana_mulligan_boundary",
            "sadhana_mulligan_eval",
        )
        for p in (site / package).rglob("*.py")
    }
    require(
        observed == {row["path"] for row in inventory},
        "Caller Python inventory has extra or missing modules",
    )
    author_record = pin(training.AUTHOR_MANIFEST_PATH)
    author_config = {"author_manifest": author_record, "stage": config["method"]}
    author_root, _, author = training.source_inputs(author_config)
    training.verify_installation_files(
        config, author_root / "source", author["boundary"]
    )
    return {
        **author,
        "author_manifest": author_record,
        "installed": inventory,
        "settings": json.loads(
            (PREPARATION / "DEV_SETTINGS.template.json").read_text()
        ),
        "caller_template": json.loads(
            (PREPARATION / "CONFIG.template.json").read_text()
        ),
    }


def accepted_periodic_idql(config: dict, inputs: dict, receipt: dict) -> dict:
    """Admit only a separately accepted, quiescent original IDQL periodic model."""
    step = receipt.get("training_steps")
    require(
        receipt.get("schema") == "sadhana.mulligan_r1_periodic_checkpoint_acceptance/1"
        and receipt.get("status") == "accepted_r1_idql_periodic_checkpoint"
        and receipt.get("owner") == "/root"
        and receipt.get("artifact_accepted") is True
        and receipt.get("scope")
        in ("frozen_evaluation_only", "frozen_policy_artifact_only")
        and config["method"] == receipt.get("policy_type") == "idql"
        and type(step) is int
        and 0 < step < 150000
        and step % 10000 == 0
        and receipt.get("task") == TASK
        and receipt.get("embodiment") == EMBODIMENT
        and config["parent_acceptance"] is None
        and receipt.get("parent") is None
        and receipt.get("producer_status") == "failed",
        "Partial IDQL needs explicit root artifact acceptance; not full training or DIVL",
    )
    expected = {key: inputs[key] for key in ("author_manifest", "data", "task")}
    require(receipt.get("bindings") == expected, "Periodic checkpoint lineage differs")
    root = owned(receipt["checkpoint_root"])
    require(
        root.is_dir() and not root.is_symlink() and root.name == f"checkpoint_{step}",
        "Periodic checkpoint directory must name its actual saved step",
    )
    training.verified_tree(root, receipt["checkpoint_files"])
    require(
        {p.name for p in root.iterdir()}
        == {"metadata.json", "stats.json", "policy.pt"},
        "Use the original three-file periodic checkpoint",
    )
    metadata = json.loads((root / "metadata.json").read_text())
    require(
        type(metadata.get("step")) is int
        and metadata["step"] == step
        and metadata.get("policy_type") == "idql",
        "Actual periodic metadata step/type differs from acceptance",
    )
    inspection = json.loads(verify(receipt["inspection"]).read_text())
    manifest_pin = inspection["source_manifest"]
    require(
        Path(manifest_pin["path"]) == PERIODIC_INSPECTOR_MANIFEST
        and manifest_pin["sha256"] == PERIODIC_INSPECTOR_SHA256,
        "Use the exact reviewed partial inspector source",
    )
    manifest = json.loads(verify(manifest_pin).read_text())
    for row in manifest["files"] + manifest["inputs"]:
        verify(row)
    rolling_step = inspection.get("rolling_step")
    require(
        inspection.get("schema") == "sadhana.mulligan_partial_idql_inspection/1"
        and inspection.get("status")
        == "passed_partial_idql_checkpoint_inspection_pending_root_acceptance"
        and inspection.get("artifact_accepted") is False
        and inspection.get("scope") == receipt["scope"]
        and inspection.get("rolling_completed") is False
        and type(rolling_step) is int
        and step <= rolling_step < 150000
        and rolling_step % 10000 == 0
        and type(inspection.get("finite_model_leaves")) is int
        and inspection["finite_model_leaves"] > 0
        and inspection.get("fixed_input_original_loader_probe")
        == "equal_finite_action_q_v_for_seven_calls"
        and inspection.get("restore_probe_device") == "cpu"
        and inspection.get("cuda_initialized") is False
        and inspection.get("exact_resume_executed") is False,
        "Partial inspection must identify finite original restored policy and actual rolling step",
    )
    for key in (
        "producer_run_id",
        "bindings",
        "training_steps",
        "policy_type",
        "checkpoint_root",
        "checkpoint_files",
        "producer_cleanup",
        "stop_evidence",
    ):
        require(
            inspection.get(key) == receipt[key],
            f"Partial inspection binding differs: {key}",
        )
    if receipt["scope"] == "frozen_policy_artifact_only":
        require(
            inspection.get("optimizer_at_selected_step_available") is False
            and inspection.get("optimizer_counters") == {}
            and "periodic_equals_rolling_at_selected_step" in inspection
            and inspection["periodic_equals_rolling_at_selected_step"] is None
            and "original_rolling_rng_sampler_present" in inspection
            and inspection["original_rolling_rng_sampler_present"] is None,
            "Policy-only inspection must not attribute later optimizer or RNG state to this policy",
        )
    else:
        counters = inspection.get("optimizer_counters", {})
        require(
            rolling_step == step
            and inspection.get("optimizer_at_selected_step_available") is True
            and inspection.get("periodic_equals_rolling_at_selected_step") is True
            and inspection.get("original_rolling_rng_sampler_present") is True
            and isinstance(counters, dict)
            and set(counters) == {"actor", "critic", "value"}
            and all(
                isinstance(row, dict)
                and type(row.get("step")) is int
                and row["step"] == step
                for row in counters.values()
            ),
            "Same-step inspection must prove actual optimizer counters and periodic/rolling equality",
        )
    cleanup = json.loads(verify(receipt["producer_cleanup"]).read_text())
    stop = json.loads(verify(receipt["stop_evidence"]).read_text())
    run_id = receipt["producer_run_id"]
    require(
        isinstance(run_id, str)
        and bool(run_id)
        and cleanup.get("owner") == stop.get("owner") == "/root"
        and cleanup.get("run_id") == stop.get("run_id") == run_id
        and cleanup.get("all_owned_processes_stopped") is True
        and cleanup.get("lease_released") is True
        and cleanup.get("charges_preserved") is True
        and stop.get("status") == "root_stopped_after_committed_periodic_checkpoint"
        and type(stop.get("training_steps")) is int
        and stop["training_steps"] == rolling_step
        and stop.get("checkpoint_root")
        == str(root.parent / f"checkpoint_{rolling_step}"),
        "Partial checkpoint needs matching root stop, cleanup and closed accounting",
    )
    canonical = json.loads(verify(cleanup["canonical"]).read_text())
    worker_path = verify(cleanup["worker"])
    worker = json.loads(worker_path.read_text())
    require(
        canonical.get("run_id") == run_id
        and canonical.get("status") == worker.get("status") == "failed"
        and worker.get("mode") == training.MODE
        and worker.get("stage") == "idql"
        and all(
            row.get("task") == TASK and row.get("embodiment") == EMBODIMENT
            for row in (canonical, worker)
        )
        and worker_path.name == "receipt.json"
        and worker_path.parent.name == run_id
        and root.resolve().is_relative_to(
            (worker_path.parent / "training/checkpoints").resolve()
        ),
        "Preserve the actual stopped IDQL producer and its checkpoint location",
    )
    require(
        bool(receipt["evidence"]),
        "Periodic acceptance needs inspection/execution evidence",
    )
    for row in receipt["evidence"]:
        verify(row)
    return receipt


def accepted_checkpoint(config: dict, inputs: dict) -> dict:
    receipt = json.loads(verify(config["checkpoint_acceptance"]).read_text())
    if receipt.get("status") == "accepted_r1_idql_periodic_checkpoint":
        return accepted_periodic_idql(config, inputs, receipt)
    expected_status = (
        "accepted_r1_idql_parent"
        if config["method"] == "idql"
        else "accepted_r1_divl_checkpoint"
    )
    require(
        receipt.get("status") == expected_status
        and receipt.get("task") == TASK
        and receipt.get("embodiment") == EMBODIMENT
        and receipt.get("training_steps") == 150000
        and receipt.get("policy_type") == config["method"],
        "Frozen evaluation needs the accepted original 150000-step R1 checkpoint",
    )
    expected = {key: inputs[key] for key in ("author_manifest", "data", "task")}
    require(
        receipt.get("bindings") == expected,
        "Checkpoint source/data/task lineage differs",
    )
    root = owned(receipt["checkpoint_root"])
    require(
        root.is_dir() and not root.is_symlink() and root.name == "final_model",
        "Use an accepted final_model directory",
    )
    training.verified_tree(root, receipt["checkpoint_files"])
    require(
        {"metadata.json", "stats.json", "policy.pt"}
        <= {p.name for p in root.iterdir()},
        "Original checkpoint files are absent",
    )
    require(bool(receipt["evidence"]), "Checkpoint acceptance needs producer evidence")
    for row in receipt["evidence"]:
        verify(row)
    if config["method"] == "divl":
        require(
            receipt.get("parent") == config["parent_acceptance"],
            "DIVL parent lineage differs",
        )
        parent_config = {
            **config,
            "method": "idql",
            "checkpoint_acceptance": config["parent_acceptance"],
            "parent_acceptance": None,
        }
        parent = accepted_checkpoint(parent_config, inputs)
        require(
            parent.get("status") == "accepted_r1_idql_parent",
            "DIVL still requires the accepted original 150000-step IDQL parent",
        )
    else:
        require(
            config["parent_acceptance"] is None and receipt.get("parent") is None,
            "IDQL must not declare a DIVL parent",
        )
    return receipt


def verify_admission(config: dict, inputs: dict) -> None:
    receipt = json.loads(verify(config["admission"]).read_text())
    require(
        receipt.get("schema") == "sadhana.mulligan_r1_frozen_eval_admission/1"
        and receipt.get("status") == "accepted_for_r1_frozen_eval"
        and receipt.get("task") == TASK
        and receipt.get("embodiment") == EMBODIMENT
        and receipt.get("method") == config["method"]
        and receipt.get("num_action_samples") == config["num_action_samples"]
        and receipt.get("timeout_s") == config["timeout_s"],
        "Root must accept this method, candidate count and measured timeout",
    )
    expected = {
        key: config[key]
        for key in (
            "caller_release",
            "runtimes",
            "checkpoint_acceptance",
            "parent_acceptance",
        )
    }
    expected.update({key: inputs[key] for key in ("author_manifest", "data", "task")})
    require(
        receipt["bindings"] == expected, "Frozen evaluation admission bindings differ"
    )
    require(
        set(receipt["evidence"]) == GATES,
        "Actual native/restore/timing qualification is required",
    )
    for row in receipt["evidence"].values():
        verify(row)


def validate(config: dict) -> tuple[dict, dict]:
    require(
        set(config)
        == {
            "schema_version",
            "mode",
            "method",
            "num_action_samples",
            "python",
            "timeout_s",
            "caller_release",
            "runtimes",
            "checkpoint_acceptance",
            "parent_acceptance",
            "admission",
        },
        "Use only the fixed frozen evaluation fields; no command or argument override",
    )
    require(
        type(config["schema_version"]) is int
        and config["schema_version"] == 1
        and config["mode"] == MODE
        and config["method"] in ("idql", "divl")
        and type(config["num_action_samples"]) is int
        and config["num_action_samples"] in (1, 32),
        "Select IDQL/DIVL and exactly N1/N32",
    )
    timeout = config["timeout_s"]
    require(
        type(timeout) in (int, float)
        and math.isfinite(timeout)
        and 0 < timeout <= MAX_WORKER_SECONDS,
        "Frozen evaluation timeout must be measured and finite in (0,86400]",
    )
    inputs = source_inputs(config)
    checkpoint = accepted_checkpoint(config, inputs)
    verify_admission(config, inputs)
    return inputs, checkpoint


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    validate(config)
    return config


def check_parent(
    job: dict, config: dict, effective_timeout: float | None = None
) -> None:
    require(
        config.get("mode") == MODE
        and job.get("task") == TASK
        and job.get("embodiment") == EMBODIMENT,
        "Frozen R1 task and embodiment must be explicit",
    )
    maximum = job.get("max_wall_s")
    needed = config["timeout_s"] + PARENT_MARGIN_SECONDS
    require(
        type(maximum) in (int, float)
        and math.isfinite(maximum)
        and needed <= maximum <= MAX_WORKER_SECONDS + PARENT_MARGIN_SECONDS,
        "Frozen evaluation parent must include at least 100 seconds cleanup",
    )
    if effective_timeout is not None:
        require(
            type(effective_timeout) in (int, float)
            and math.isfinite(effective_timeout)
            and min(maximum, effective_timeout) >= needed,
            "Effective frozen evaluation timeout must include at least 100 seconds cleanup",
        )


def caller_config(
    config: dict, inputs: dict, checkpoint: dict, settings_path: Path, destination: Path
) -> dict:
    """Resolve only task/runtime/output/checkpoint substitutions in the sealed caller."""
    inventory = {row["path"]: row for row in inputs["installed"]}
    learner = config["runtimes"]["learner"]
    modules = {}
    for package, names in CALLER_MODULES.items():
        root = Path(
            learner["author_package_dir"]
            if package == "mulligan"
            else learner["package_dir"]
        )
        for name in names:
            modules[f"{package}/{name}"] = inventory[str(root / name)]
    native = config["runtimes"]["native"]
    native_files = [native["python"]]
    prefix = "src/sadhana_mulligan_boundary/"
    for row in inputs["boundary"]["files"]:
        if row["path"].startswith(prefix):
            native_files.append(
                {
                    **row,
                    "path": str(
                        Path(native["package_dir"]) / row["path"][len(prefix) :]
                    ),
                }
            )
    return {
        **inputs["caller_template"],
        "method": config["method"],
        "num_action_samples": config["num_action_samples"],
        "checkpoint_root": checkpoint["checkpoint_root"],
        "checkpoint_files": checkpoint["checkpoint_files"],
        "checkpoint_acceptance": config["checkpoint_acceptance"],
        "parent_acceptance": config["parent_acceptance"],
        "learner_python": learner["python"],
        "installed_modules": modules,
        "native_runtime_files": native_files,
        "settings": pin(settings_path),
        "output_dir": str(destination),
    }


def command_for(config: dict, caller_path: Path) -> list[str]:
    return [
        config["python"],
        "-I",
        "-u",
        "-B",
        "-m",
        "sadhana_mulligan_eval.cli",
        "--config",
        str(caller_path),
    ]


def run(config: dict, output: Path) -> dict:
    from abc_bench.runner import write_json
    from abc_bench.subset import require_subset_lease

    started = time.monotonic()
    require_subset_lease()
    require(
        os.getpgrp() == os.getpid(),
        "Frozen evaluation wrapper must lead its canonical process group",
    )
    inputs, checkpoint = validate(config)
    runtime = training.verify_runtime(config, inputs)
    owned(str(output)).mkdir(parents=True, exist_ok=True)
    owned_output = output / "frozen-eval"
    owned_output.mkdir(exist_ok=False)
    (owned_output / "tmp").mkdir()
    destination = owned_output / "evaluation"
    settings = {
        **inputs["settings"],
        "native_python": config["runtimes"]["native"]["python"]["path"],
        "output_dir": str(destination / "native"),
    }
    settings_path = owned_output / "settings.json"
    write_json(settings_path, settings)
    settings_path.chmod(0o444)
    resolved = caller_config(config, inputs, checkpoint, settings_path, destination)
    caller_path = owned_output / "caller-config.json"
    write_json(caller_path, resolved)
    caller_path.chmod(0o444)
    command = command_for(config, caller_path)
    result = {
        "schema_version": 1,
        "status": "running",
        "mode": MODE,
        "task": TASK,
        "embodiment": EMBODIMENT,
        "method": config["method"],
        "num_action_samples": config["num_action_samples"],
        "metrics": {},
        "experiment_accepted": False,
        "runtime": runtime,
        "command": command,
        "caller_config": pin(caller_path),
        "admission": config["admission"],
        "checkpoint_acceptance": config["checkpoint_acceptance"],
        "claim_scope": "Original frozen evaluator execution only; root must accept native/media/cleanup and score",
    }
    write_json(output / "config.json", config)
    receipt = output / "receipt.json"
    process = None
    known = {}

    def cancelled(signum, frame):
        raise InterruptedError(f"Frozen evaluation cancelled by signal {signum}")

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        write_json(receipt, result)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONHOME")
        }
        environment.update(
            MUJOCO_GL="osmesa",
            WANDB_MODE="offline",
            TMPDIR=str(owned_output / "tmp"),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            IMAGEIO_FFMPEG_NO_PREVENT_SIGINT="1",
        )
        with (owned_output / "caller.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=owned_output,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            result["caller_pid"] = process.pid
            training.observe_descendants(os.getpgrp(), known)
            result["observed_descendants"] = list(known.values())
            write_json(receipt, result)
            deadline = started + config["timeout_s"]
            while True:
                before = len(known)
                training.observe_descendants(os.getpgrp(), known)
                if len(known) != before:
                    result["observed_descendants"] = list(known.values())
                    write_json(receipt, result)
                code = process.poll()
                if code is not None:
                    break
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, config["timeout_s"])
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        result["caller_exit_code"] = code
        require(code == 0, f"Original Mulligan frozen caller exited {code}")
        report_path = destination / "RESULT.json"
        report = json.loads(report_path.read_text())
        require(
            report.get("schema") == "sadhana.mulligan_frozen_eval_result/1"
            and report.get("status") == "completed"
            and report.get("config") == pin(caller_path)
            and report.get("experiment_accepted") is False
            and report.get("method") == config["method"]
            and report.get("num_action_samples") == config["num_action_samples"],
            "Caller did not complete this exact frozen evaluation",
        )
        validate(config)
        verify(result["caller_config"])
        verify(resolved["settings"])
        result.update(status="completed", caller_result=pin(report_path))
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
            cleanup = training.cleanup(
                process,
                os.getpgrp(),
                known,
                terminate_group=result["status"] != "completed",
            )
            result["cleanup"] = cleanup
            if not cleanup["clean"] or cleanup["cleanup_intervention"]:
                result["status"] = "failed"
            result["elapsed_s"] = time.monotonic() - started
            write_json(receipt, result)
        finally:
            signal.signal(signal.SIGTERM, previous)
        if result["status"] != "completed" and not result.get("error"):
            raise RuntimeError(
                "Frozen evaluation cleanup failed; canonical cleanup remains required"
            )
    return result
