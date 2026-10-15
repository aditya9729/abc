"""Admit the sealed R1 IDQL/DIVL recipes; own processes, never learner logic."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from abc_bench.mulligan_r1lite_diagnostic import (
    EMBODIMENT,
    TASK,
    owned,
    pin,
    require,
    verify,
)
from abc_bench.mulligan_r1lite_diagnostic import (
    verify_runtime as verify_boundary_runtime,
)

MODE = "r1lite-original-training"
MAX_WORKER_SECONDS = 86400
PARENT_MARGIN_SECONDS = 100
CHILD_CLEANUP_SECONDS = 80
KILL_WAIT_SECONDS = 2
AUTHOR_MANIFEST_PATH = Path(
    "/home/user/aditya/RL/builds/other-methods-plan-20261015/"
    "author-r1-preparation-v1/SOURCE_MANIFEST.json"
)
AUTHOR_MANIFEST_SHA256 = (
    "c343f3af5915edb5a42d773d3088f015a749e39e24797dd06c482cf130bdebe7"
)
MONITORED_RECIPE_PATH = Path(
    "/home/user/aditya/RL/builds/other-methods-plan-20261015/"
    "mulligan-monitored-preparation-20261015/MONITORED_RECIPE.json"
)
MONITORED_RECIPE_SHA256 = (
    "3ee08efb2fe81d569f32126028331d512691152a719cd8f3168454b87fc2a30b"
)
BOUNDARY_SHA256 = "89c87470826b1c6960288d9bf177fc78f662458c80aaa3cc593c15f914743001"
DATA_SHA256 = "3a892935ffa4aea005540b67f3a53ba0e4889d5aa36e682779ed4b3b04a7a716"
GATES = {
    "installed_author",
    "native_one",
    "native_two",
    "native_ten",
    "update_restore",
    "timing",
}


def verified_tree(root: Path, files: list[dict]) -> None:
    """Authenticate every named file and reject aliases and an incomplete inventory."""
    paths = []
    for row in files:
        path = verify(row)
        require(
            path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),
            "Pinned tree escapes its declared root",
        )
        paths.append(path.resolve())
    require(len(paths) == len(set(paths)), "Duplicate pinned tree file")
    observed = [p for p in root.rglob("*") if p.is_file()]
    require(
        not any(p.is_symlink() for p in root.rglob("*"))
        and {p.resolve() for p in observed} == set(paths),
        "Pinned tree contains extra or missing files",
    )


def source_inputs(config: dict) -> tuple[Path, dict, dict]:
    record = config["author_manifest"]
    require(
        Path(record["path"]) == AUTHOR_MANIFEST_PATH,
        "Use the exact frozen author manifest path; copied manifests cannot bind sibling inputs",
    )
    require(
        record["sha256"] == AUTHOR_MANIFEST_SHA256,
        "Use only the reviewed three-file R1 author patch",
    )
    manifest_path = verify(record)
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["files"]:
        verify(row)
    root = manifest_path.parent
    recipe = json.loads((root / f"{config['stage'].upper()}_ARGV.json").read_text())
    if "monitor_recipe" in config:
        require(config["stage"] == "idql", "Monitored recipe admits fresh IDQL only")
        monitor_record = config["monitor_recipe"]
        require(
            Path(monitor_record["path"]) == MONITORED_RECIPE_PATH
            and monitor_record["sha256"] == MONITORED_RECIPE_SHA256,
            "Use the exact reviewed monitored IDQL recipe path and hash",
        )
        monitored = json.loads(verify(monitor_record).read_text())
        changes = {
            "--eval.freq=150000": "--eval.freq=10000",
            "--training.periodic_checkpoint_freq=0": "--training.periodic_checkpoint_freq=10000",
        }
        require(
            monitored["base_recipe"] == pin(root / "IDQL_ARGV.json")
            and monitored["module"] == recipe["module"]
            and monitored["argv"] == [changes.get(arg, arg) for arg in recipe["argv"]],
            "Monitored recipe may change only evaluation and periodic-save cadence",
        )
        recipe = monitored
    settings = json.loads(
        (root / f"{config['stage'].upper()}_TRAIN_SETTINGS.json").read_text()
    )
    require(
        recipe["module"] == "mulligan.training.train",
        "Only the original trainer is admitted",
    )
    inputs = json.loads((root / "INPUTS.json").read_text())["inputs"]
    boundary_record = next(row for row in inputs if row["sha256"] == BOUNDARY_SHA256)
    boundary = json.loads(verify(boundary_record).read_text())
    data_record = next(row for row in inputs if row["sha256"] == DATA_SHA256)
    data_path = verify(data_record)
    data = json.loads(data_path.read_text())
    for row in data["files"]:
        relative = Path(row["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Data manifest must use contained relative paths",
        )
        verify({**row, "path": str(data_path.parent / relative)})
    task = owned(settings["task_config"])
    require(
        pin(task)["sha256"] == settings["task_config_sha256"],
        "Task configuration changed",
    )
    return (
        root,
        recipe,
        {
            "settings": settings,
            "boundary": boundary,
            "data": data_record,
            "task": pin(task),
        },
    )


def verify_installation_files(config: dict, source: Path, boundary: dict) -> None:
    """Check installed bytes before lease admission; no package imports or probes."""
    require(
        set(config["runtimes"]) == {"learner", "native"}, "Pin both isolated runtimes"
    )
    for name, runtime in config["runtimes"].items():
        fields = {"python", "package_dir"} | (
            {"author_package_dir"} if name == "learner" else set()
        )
        require(set(runtime) == fields, "Use explicit installed runtime bindings")
        verify(runtime["python"])
        package = owned(runtime["package_dir"])
        require(package.is_dir(), "Installed boundary package is absent")
        for row in boundary["files"]:
            prefix = "src/sadhana_mulligan_boundary/"
            if row["path"].startswith(prefix):
                verify({**row, "path": str(package / row["path"][len(prefix) :])})
    require(
        config["python"] == config["runtimes"]["learner"]["python"]["path"],
        "Trainer must use the pinned learner interpreter",
    )
    installed = owned(config["runtimes"]["learner"]["author_package_dir"])
    expected = {
        str(p.relative_to(source / "mulligan")): pin(p)["sha256"]
        for p in (source / "mulligan").rglob("*.py")
    }
    actual = {
        str(p.relative_to(installed)): pin(p)["sha256"] for p in installed.rglob("*.py")
    }
    require(expected and expected == actual, "Installed author Python payloads differ")


def parent_checkpoint(config: dict) -> Path | None:
    if config["stage"] == "idql":
        require(config["parent"] is None, "IDQL starts without a pretrained parent")
        return None
    path = verify(config["parent"])
    receipt = json.loads(path.read_text())
    require(
        receipt.get("status") == "accepted_r1_idql_parent"
        and receipt.get("task") == TASK
        and receipt.get("embodiment") == EMBODIMENT
        and receipt.get("training_steps") == 150000
        and receipt.get("policy_type") == "idql",
        "DIVL needs an accepted R1 IDQL parent",
    )
    root = owned(receipt["checkpoint_root"])
    require(
        root.is_dir() and root.name == "final_model",
        "Use the accepted IDQL final_model directory",
    )
    verified_tree(root, receipt["checkpoint_files"])
    require(bool(receipt["checkpoint_files"]), "Parent checkpoint inventory is empty")
    for row in receipt["evidence"]:
        verify(row)
    require(
        bool(receipt["evidence"]),
        "Parent acceptance must bind actual producer evidence",
    )
    return root


def verify_admission(config: dict, inputs: dict) -> None:
    """A root attestation binds the future actual qualification, not source fixtures."""
    receipt = json.loads(verify(config["admission"]).read_text())
    require(
        receipt.get("schema") == "sadhana.mulligan_r1_training_admission/1"
        and receipt.get("status") == "accepted_for_r1_original_training"
        and receipt.get("stage") == config["stage"]
        and receipt.get("task") == TASK
        and receipt.get("embodiment") == EMBODIMENT
        and receipt.get("training_steps") == 150000
        and receipt.get("timeout_s") == config["timeout_s"],
        "Root must bind qualified inputs and a measured training watchdog",
    )
    require(
        set(receipt["evidence"]) == GATES,
        "All native, update/restore and timing gates are required",
    )
    for row in receipt["evidence"].values():
        verify(row)
    expected = {
        "author_manifest": config["author_manifest"],
        "runtimes": config["runtimes"],
        "data": inputs["data"],
        "task": inputs["task"],
        "parent": config["parent"],
    }
    if "monitor_recipe" in config:
        expected["monitor_recipe"] = config["monitor_recipe"]
    require(
        receipt["bindings"] == expected,
        "Admission bindings differ from this exact training configuration",
    )


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    require(
        set(config)
        == {
            "schema_version",
            "mode",
            "stage",
            "python",
            "timeout_s",
            "author_manifest",
            "runtimes",
            "admission",
            "parent",
        }
        | ({"monitor_recipe"} if "monitor_recipe" in config else set()),
        "Use only explicit R1 original-training fields; no command or argument override",
    )
    require(
        type(config["schema_version"]) is int
        and config["schema_version"] == 1
        and config["mode"] == MODE
        and config["stage"] in ("idql", "divl"),
        "Select the named original R1 IDQL or DIVL training profile",
    )
    timeout = config["timeout_s"]
    require(
        type(timeout) in (int, float)
        and math.isfinite(timeout)
        and 0 < timeout <= MAX_WORKER_SECONDS,
        "Training timeout must be finite in (0,86400]",
    )
    root, _, inputs = source_inputs(config)
    verify_installation_files(config, root / "source", inputs["boundary"])
    parent_checkpoint(config)
    verify_admission(config, inputs)
    return config


def check_parent(
    job: dict, config: dict, effective_timeout: float | None = None
) -> None:
    require(
        config.get("mode") == MODE
        and job.get("task") == TASK
        and job.get("embodiment") == EMBODIMENT,
        "R1 training task and embodiment must be explicit",
    )
    maximum = job.get("max_wall_s")
    needed = config["timeout_s"] + PARENT_MARGIN_SECONDS
    require(
        type(maximum) in (int, float)
        and math.isfinite(maximum)
        and needed <= maximum <= MAX_WORKER_SECONDS + PARENT_MARGIN_SECONDS,
        "R1 training parent must include at least 100 seconds cleanup",
    )
    if effective_timeout is not None:
        require(
            type(effective_timeout) in (int, float)
            and math.isfinite(effective_timeout)
            and min(maximum, effective_timeout) >= needed,
            "Effective R1 training parent must include at least 100 seconds cleanup",
        )


def verify_runtime(config: dict, inputs: dict) -> dict:
    runtime = verify_boundary_runtime(config, {"fixed_settings": inputs["settings"]})
    code = (
        "import json,sys\nfrom importlib.metadata import distribution\n"
        "d=distribution('mulligan')\n"
        "print(json.dumps({'prefix':sys.prefix,'version':d.version,"
        "'package_dir':str(d.locate_file('mulligan').resolve())}))\n"
    )
    probe = subprocess.run(
        [config["python"], "-I", "-B", "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    observed = json.loads(probe.stdout)
    package = Path(config["runtimes"]["learner"]["author_package_dir"]).resolve()
    require(
        observed["version"] == "0.1.0"
        and Path(observed["package_dir"]) == package
        and package.is_relative_to(Path(observed["prefix"]).resolve()),
        "Trainer is not the pinned normal Mulligan installation",
    )
    runtime["author"] = observed
    return runtime


def command_for(
    config: dict, recipe: dict, output: Path, parent: Path | None
) -> list[str]:
    arguments = []
    for arg in recipe["argv"]:
        if arg.startswith("--system.checkpoint_dir="):
            arg = "--system.checkpoint_dir=" + str(output / "checkpoints")
        elif arg.startswith("--pretrained_artifact="):
            require(parent is not None, "DIVL parent remains unresolved")
            arg = "--pretrained_artifact=" + str(parent)
        arguments.append(arg)
    return [
        config["python"],
        "-I",
        "-u",
        "-B",
        "-m",
        "mulligan.training.train",
        *arguments,
    ]


def process_snapshot() -> dict[int, dict]:
    """Read Linux identities without importing a runtime or signalling processes."""
    rows = {}
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        pid = int(path.parent.name)
        if fields[0] not in ("Z", "X"):
            rows[pid] = {
                "boot_id": boot,
                "pid": pid,
                "start_ticks": int(fields[19]),
                "pgid": int(fields[2]),
                "ppid": int(fields[1]),
            }
    return rows


def observe_descendants(pgid: int, known: dict) -> list[dict]:
    """Retain observed identities across exit/reparenting, including W&B sessions."""
    snapshot = process_snapshot()
    live = {
        row["pid"]
        for row in known.values()
        if row["pid"] in snapshot
        and all(
            snapshot[row["pid"]][key] == row[key]
            for key in ("boot_id", "pid", "start_ticks")
        )
    }
    owned_pids = live | {os.getpid()}
    changed = True
    while changed:
        changed = False
        for pid, row in snapshot.items():
            if pid != os.getpid() and (
                row["pgid"] == pgid or row["ppid"] in owned_pids
            ):
                known[(row["pid"], row["start_ticks"])] = row
                if pid not in owned_pids:
                    owned_pids.add(pid)
                    changed = True
    return [
        row
        for row in known.values()
        if row["pid"] in snapshot
        and all(
            snapshot[row["pid"]][key] == row[key]
            for key in ("boot_id", "pid", "start_ticks")
        )
    ]


def cleanup(
    process: Any, pgid: int, known: dict | None = None, *, terminate_group: bool = False
) -> dict:
    """Retain the leader while native peers finish, even after the trainer exits."""
    deadline = time.monotonic() + CHILD_CLEANUP_SECONDS
    result = {
        "cleanup_intervention": None,
        "scope": "Current canonical group and exact previously observed descendants; root separately verifies complete cleanup",
    }
    known = {} if known is None else known
    if terminate_group:
        # run() proved that this wrapper is the canonical group leader. Its
        # finally block ignores TERM while native peers receive cleanup time.
        result["group_sigterm_requested"] = True
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if process is not None and process.poll() is None:
        result["cleanup_intervention"] = "trainer_SIGTERM"
        process.terminate()
    while True:
        if process is not None:
            process.poll()  # Reap the owned direct child when it exits.
        members = observe_descendants(pgid, known)
        if not members or time.monotonic() >= deadline:
            break
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    if process is not None and process.poll() is None:
        result["cleanup_intervention"] = "trainer_SIGKILL"
        process.kill()
        try:
            process.wait(timeout=KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    result.update(
        known_descendants=list(known.values()),
        remaining_known_descendants=observe_descendants(pgid, known),
        direct_child_stopped=process is None or process.poll() is not None,
    )
    result["clean"] = (
        result["direct_child_stopped"] and not result["remaining_known_descendants"]
    )
    return result


def run(config: dict, output: Path) -> dict:
    from abc_bench.runner import write_json
    from abc_bench.subset import require_subset_lease

    started = time.monotonic()
    require_subset_lease()
    require(
        os.getpgrp() == os.getpid(),
        "Training wrapper must lead its canonical process group",
    )
    root, recipe, inputs = source_inputs(config)
    verify_installation_files(config, root / "source", inputs["boundary"])
    verify_admission(config, inputs)
    parent = parent_checkpoint(config)
    runtime = verify_runtime(config, inputs)
    owned(str(output)).mkdir(parents=True, exist_ok=True)
    training = output / "training"
    training.mkdir(exist_ok=False)
    for name in ("tmp", "wandb"):
        (training / name).mkdir()
    settings = {
        **inputs["settings"],
        "native_python": config["runtimes"]["native"]["python"]["path"],
        "output_dir": str(training / "native"),
    }
    settings_path = training / "settings.json"
    write_json(settings_path, settings)
    settings_path.chmod(0o444)
    command = command_for(config, recipe, training, parent)
    result = {
        "schema_version": 1,
        "status": "running",
        "mode": MODE,
        "stage": config["stage"],
        "task": TASK,
        "embodiment": EMBODIMENT,
        "metrics": {},
        "experiment_accepted": False,
        "claim_scope": "Original 150000-step author command with explicit R1/data substitutions; checkpoints and scores require separate acceptance",
        "runtime": runtime,
        "command": command,
        "settings": pin(settings_path),
        "admission": config["admission"],
        "parent": config["parent"],
        "descendant_cleanup_owner": "canonical runner with explicit R1 training grace",
    }
    if "monitor_recipe" in config:
        result.update(
            monitor_recipe=config["monitor_recipe"],
            root_stop_rule=recipe["root_stop_rule"],
            claim_scope=(
                "Original IDQL learner up to150000 updates; original TRAIN N1/N32 "
                "evaluation and periodic models every10000, rolling state every5000. "
                "Root applies the predeclared compute-futility stop rule after committed "
                "monitors; no automatic early stopping, exact end-only RNG trajectory, "
                "convergence or policy-gain claim."
            ),
        )
    write_json(output / "config.json", config)
    receipt = output / "receipt.json"
    process = None
    known = {}

    def cancelled(signum, frame):
        raise InterruptedError(f"R1 original training cancelled by signal {signum}")

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
            WANDB_DIR=str(training / "wandb"),
            TMPDIR=str(training / "tmp"),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            IMAGEIO_FFMPEG_NO_PREVENT_SIGINT="1",
        )
        with (training / "author.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=root / "source",
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            result["author_pid"] = process.pid
            observe_descendants(os.getpgrp(), known)
            result["observed_descendants"] = list(known.values())
            write_json(receipt, result)
            deadline = started + config["timeout_s"]
            while True:
                before = len(known)
                observe_descendants(os.getpgrp(), known)
                if len(known) != before:
                    result["observed_descendants"] = list(known.values())
                    write_json(receipt, result)
                code = process.poll()
                if code is not None:
                    break
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, config["timeout_s"])
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        result["author_exit_code"] = code
        require(code == 0, f"Original R1 Mulligan trainer exited {code}")
        source_inputs(config)
        verify_installation_files(config, root / "source", inputs["boundary"])
        parent_checkpoint(config)
        result["status"] = "completed"
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
            cleanup_result = cleanup(
                process,
                os.getpgrp(),
                known,
                terminate_group=result["status"] != "completed",
            )
            result["cleanup"] = cleanup_result
            if not cleanup_result["clean"] or cleanup_result["cleanup_intervention"]:
                result["status"] = "failed"
            result["elapsed_s"] = time.monotonic() - started
            write_json(receipt, result)
        finally:
            signal.signal(signal.SIGTERM, previous)
        if result["status"] != "completed" and not result.get("error"):
            raise RuntimeError(
                "R1 training cleanup failed; canonical group cleanup remains required"
            )
    return result
