"""Root-pinned matched-evaluator requests and producer receipt validation.

No learner, model, native package or NRH implementation is imported here.
The request is explicit caller trust, not a producer's acceptance flag.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import math
import re
import time
from dataclasses import dataclass
from email.parser import BytesParser
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Any

from abc_bench.visual_evaluation_bootstrap import (
    artifact,
    canonical,
    interpreter_probe,
    read_json,
    require,
    verify_pin,
)

CANONICAL_CAMPAIGN = Path("/home/user/aditya/RL/abc/outputs/bench")
REVISION = "sadhana.visual-methods-matched-evaluation/2"
NRH_VERSION = "0.4.13"
TASK = "put_plastic_bottles_in_bin"
PROTOCOLS = {
    "resfit-frozen-nominal": False,
    "resfit-learned-mean": True,
    "expo-frozen-async-n1": False,
    "expo-full-learned-n8e8": True,
    "qf3-frozen-h30c15-nortc": False,
    "qf3-learned-online-h30c15-nortc": True,
}
MAX_RECEIPT_BYTES = 512 * 1024**2


def count(value: Any, name: str, maximum: int = 50000) -> int:
    require(
        type(value) is int and 0 <= value <= maximum,
        "Invalid typed evaluation count: " + name,
    )
    return value


def positive(value: Any, name: str) -> float:
    require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        "Finite positive evaluation " + name + " required",
    )
    return float(value)


def same(left: Any, right: Any) -> bool:
    return canonical(left) == canonical(right)


def source_binding() -> dict[str, str]:
    return {
        name: artifact(Path(__file__).with_name(name + ".py"))["sha256"]
        for name in (
            "runner",
            "visual_evaluation_dispatch",
            "visual_evaluation_bootstrap",
        )
    }


def _worker(worker: dict, deadline: float) -> tuple[list[dict], dict]:
    require(
        isinstance(worker, dict)
        and set(worker)
        == {
            "python",
            "python_resolved",
            "prefix",
            "version",
            "pyvenv",
            "site_packages",
            "nrh_version",
            "metadata",
            "record",
            "nrh_files",
            "pth",
            "startup",
            "dependency_metadata",
        },
        "Worker binding fields differ",
    )
    prefix = Path(worker["prefix"])
    require(
        prefix.is_absolute()
        and all(
            not prefix.resolve().is_relative_to(Path(root))
            for root in (
                "/home/user/aditya/RL/abc",
                "/home/user/aditya/RL/sadhana-resfit",
            )
        ),
        "Worker must use an isolated environment",
    )
    require(
        worker["python"]["path"] == str(prefix / "bin/python")
        and str(Path(worker["python"]["path"]).resolve()) == worker["python_resolved"],
        "Literal worker interpreter differs",
    )
    verify_pin(worker["python"], 64 * 1024**2)
    require(
        worker["pyvenv"]["path"] == str(prefix / "pyvenv.cfg"),
        "Worker venv config origin differs",
    )
    cfg = verify_pin(worker["pyvenv"]).decode()
    require(
        "include-system-site-packages = false" in cfg,
        "Worker system site must be isolated",
    )
    require(
        isinstance(worker["version"], str)
        and worker["version"]
        and worker["nrh_version"] == NRH_VERSION,
        "Final admitted worker release/version required",
    )
    site = Path(worker["site_packages"])
    require(
        site.is_absolute()
        and site.parent.parent.parent == prefix
        and site.name == "site-packages",
        "Worker site prefix differs",
    )
    pins = [worker["python"], worker["pyvenv"], worker["metadata"], worker["record"]]
    require(
        isinstance(worker["nrh_files"], dict) and worker["nrh_files"],
        "Installed NRH file inventory required",
    )
    files = {str(p.relative_to(site)): p for p in (site / "nrh").rglob("*.py")}
    require(
        set(files) == set(worker["nrh_files"])
        and not list((site / "nrh").rglob("*.pyc")),
        "Installed NRH inventory or unpinned bytecode differs",
    )
    for name, pin in worker["nrh_files"].items():
        require(
            pin["path"] == str(files[name])
            and files[name].resolve().is_relative_to(site.resolve()),
            "Installed NRH source origin differs",
        )
        verify_pin(pin)
        pins.append(pin)
    for name in ("nrh/__init__.py", "nrh/visual_method_evaluation.py"):
        require(name in files, "Installed evaluator module absent")
    metadata = Path(worker["metadata"]["path"])
    record = Path(worker["record"]["path"])
    require(
        metadata.parent == record.parent
        and metadata.parent.parent == site
        and metadata.name == "METADATA"
        and record.name == "RECORD",
        "Worker distribution metadata origin differs",
    )
    values = BytesParser().parsebytes(verify_pin(worker["metadata"]))
    require(
        values["Name"] == "nirvana-rl-harness" and values["Version"] == NRH_VERSION,
        "Installed NRH distribution differs",
    )
    entries = list(csv.reader(io.StringIO(verify_pin(worker["record"]).decode())))
    seen = set()
    for row in entries:
        require(
            len(row) == 3 and row[0] not in seen,
            "Duplicate/malformed worker RECORD entry",
        )
        relative, encoded, size = row
        target = site / relative
        require(
            target.resolve().is_relative_to(site.resolve())
            and relative.startswith(("nrh/", metadata.parent.name + "/")),
            "Worker RECORD path escape",
        )
        seen.add(relative)
        if target == record:
            require(encoded == size == "", "Worker RECORD self hash differs")
            continue
        require(
            encoded.startswith("sha256=") and size.isdigit(),
            "Every nonself worker RECORD file must be hashed; install without bytecode",
        )
        digest = base64.urlsafe_b64decode(
            encoded[7:] + "=" * (-len(encoded[7:]) % 4)
        ).hex()
        pin = {"path": str(target), "bytes": int(size), "sha256": digest}
        verify_pin(pin)
        pins.append(pin)
    require(set(files) <= seen, "Worker RECORD omits NRH source")
    require(
        isinstance(worker["pth"], list)
        and {p["path"] for p in worker["pth"]} == {str(p) for p in site.glob("*.pth")},
        "Unpinned worker startup path file",
    )
    dependency_paths = []
    for pin in worker["pth"]:
        for line in verify_pin(pin).decode().splitlines():
            if line and not line.startswith("#"):
                require(
                    Path(line).is_absolute()
                    and not line.startswith(("import ", "import\t")),
                    "Executable or relative worker .pth is forbidden",
                )
                directory = Path(line)
                require(
                    directory.is_dir(),
                    "Worker dependency path must exist",
                )
                dependency_paths.append(str(directory))
        pins.append(pin)
    require(
        worker["startup"] == [],
        "Worker startup customization is forbidden before lease arming",
    )
    require(
        isinstance(worker["dependency_metadata"], dict),
        "Selected dependency metadata pins required",
    )
    for pin in [*worker["startup"], *worker["dependency_metadata"].values()]:
        verify_pin(pin)
        pins.append(pin)
    probe = interpreter_probe(worker["python"]["path"], deadline)
    metadata = probe["value"]
    require(
        set(metadata) == {"executable", "version", "base_prefix", "paths", "stdlib"}
        and metadata["executable"] == worker["python"]["path"]
        and metadata["version"] == worker["version"]
        and isinstance(metadata["base_prefix"], str)
        and Path(metadata["base_prefix"]).is_absolute()
        and isinstance(metadata["stdlib"], str)
        and Path(metadata["stdlib"]).is_absolute()
        and isinstance(metadata["paths"], list)
        and 1 <= len(metadata["paths"]) <= 16
        and all(
            isinstance(p, str) and len(p) <= 4096 and Path(p).is_absolute()
            for p in metadata["paths"]
        ),
        "Isolated interpreter startup metadata differs",
    )
    paths = [*metadata["paths"], str(site), *dependency_paths]
    for name in ("sitecustomize", "usercustomize"):
        require(
            PathFinder.find_spec(name, paths) is None,
            "Worker startup customization is forbidden before lease arming",
        )
    package = PathFinder.find_spec("nrh", paths)
    require(
        package is not None
        and package.origin == str(files["nrh/__init__.py"])
        and list(package.submodule_search_locations or []) == [str(site / "nrh")]
        and Path(package.origin).resolve() == files["nrh/__init__.py"].resolve(),
        "Installed NRH package import origin differs",
    )
    entry = PathFinder.find_spec(
        "nrh.visual_method_evaluation", package.submodule_search_locations
    )
    require(
        entry is not None
        and entry.origin == str(files["nrh/visual_method_evaluation.py"]),
        "Installed evaluator import origin differs",
    )
    return pins, probe


@dataclass(frozen=True)
class EvaluationRequest:
    pin: dict
    value: dict
    config: dict
    cohort: dict
    identity: dict
    startup_pins: list[dict]
    worker_probe: dict

    @property
    def protocol_id(self) -> str:
        return hashlib.sha256(canonical(self.identity)).hexdigest()


def load_request(
    pin: dict, campaign: Path = CANONICAL_CAMPAIGN, *, deadline: float | None = None
) -> EvaluationRequest:
    if deadline is None:
        deadline = time.monotonic() + 5
    verify_pin(pin)
    value = read_json(pin["path"])
    require(
        set(value)
        == {
            "schema_version",
            "kind",
            "config",
            "identity",
            "worker",
            "software_binding",
            "native_admission",
            "checkpoint_admission",
            "coordinator_sources",
            "campaign_directory",
            "cleanup_s",
            "receipt_max_bytes",
            "native_resources",
        },
        "Root evaluation request fields differ",
    )
    require(
        type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["kind"] == "root_abc_matched_visual_evaluation_request",
        "Root evaluation request schema/kind differs",
    )
    require(
        value["campaign_directory"] == str(campaign.absolute())
        and campaign.resolve() == campaign.absolute(),
        "Original absolute campaign required",
    )
    require(
        value["coordinator_sources"] == source_binding(),
        "Isolated coordinator source binding differs",
    )
    positive(value["cleanup_s"], "cleanup allowance")
    count(value["receipt_max_bytes"], "receipt byte admission", MAX_RECEIPT_BYTES)
    require(
        value["receipt_max_bytes"] > 0,
        "Positive bounded receipt byte admission required",
    )
    resources = value["native_resources"]
    require(
        isinstance(resources, dict)
        and set(resources)
        == {
            "admission",
            "host_rss_limit_bytes",
            "gpu_memory_limit_bytes",
            "virtual_address_limit_bytes",
        },
        "Explicit native resource admission required",
    )
    for key in ("host_rss_limit_bytes", "gpu_memory_limit_bytes"):
        require(
            type(resources[key]) is int and resources[key] > 0,
            "Typed native resident-memory bound required",
        )
    require(
        resources["virtual_address_limit_bytes"] is None
        or type(resources["virtual_address_limit_bytes"]) is int
        and resources["virtual_address_limit_bytes"] > 0,
        "Native virtual-address bound must be explicit or null",
    )
    verify_pin(resources["admission"])
    resource_admission = read_json(resources["admission"]["path"])
    require(
        resource_admission.get("kind") == "root_abc_visual_native_resource_admission"
        and resource_admission.get("status")
        == "accepted_for_serial_native_development_evaluation"
        and same(
            resource_admission.get("limits"),
            {k: v for k, v in resources.items() if k != "admission"},
        ),
        "Root native resident/virtual resource admission differs",
    )
    verify_pin(value["config"])
    config = read_json(value["config"]["path"])
    require(
        set(config)
        == {
            "schema_version",
            "domain",
            "protocol",
            "cohort",
            "construction_config",
            "export",
            "seed",
            "control_cap",
            "max_wall_s",
            "mode",
            "video",
            "resume",
        }
        and type(config["schema_version"]) is int
        and config["schema_version"] == 1
        and config["domain"] == "sim"
        and config["protocol"] in PROTOCOLS,
        "Exact native evaluation config required",
    )
    count(config["seed"], "seed", 2**63 - 1)
    count(config["control_cap"], "control cap", 1000)
    require(config["control_cap"] > 0, "Positive control cap required")
    positive(config["max_wall_s"], "inner wall budget")
    require(
        config["mode"] in {"paused_simulation_fixed_tick", "strict_wall"}
        and (
            config["protocol"].startswith("expo")
            or config["mode"] == "paused_simulation_fixed_tick"
        ),
        "Protocol clock mode differs",
    )
    verify_pin(config["cohort"])
    cohort = read_json(config["cohort"]["path"])
    require(
        set(cohort) == {"schema_version", "kind", "split", "task", "episodes"}
        and type(cohort["schema_version"]) is int
        and cohort["schema_version"] == 1
        and cohort["kind"] == "visual_development_cohort"
        and cohort["split"] == "development"
        and cohort["task"] == TASK
        and isinstance(cohort["episodes"], list)
        and len(cohort["episodes"]) == 50,
        "Exactly 50 development episodes required; finals unsupported",
    )
    ids, seeds = [], []
    for row in cohort["episodes"]:
        require(
            set(row) == {"id", "seed"}
            and isinstance(row["id"], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["id"]) is not None,
            "Safe development episode ID required",
        )
        ids.append(row["id"])
        seeds.append(count(row["seed"], "episode seed", 2**63 - 1))
    require(len(set(ids)) == len(set(seeds)) == 50, "Duplicate development IDs/seeds")
    verify_pin(config["construction_config"])
    require(isinstance(config["resume"], list), "Pinned completed-prefix list required")
    for resume in config["resume"]:
        verify_pin(resume, value["receipt_max_bytes"])
    video = config["video"]
    require(
        isinstance(video, dict)
        and set(video) == {"enabled", "episode_ids"}
        and type(video["enabled"]) is bool
        and isinstance(video["episode_ids"], list)
        and len(video["episode_ids"]) <= 3
        and len(set(video["episode_ids"])) == len(video["episode_ids"])
        and set(video["episode_ids"]) <= set(ids)
        and (video["enabled"] or not video["episode_ids"]),
        "Preselected bounded video IDs differ",
    )
    identity = value["identity"]
    require(
        isinstance(identity, dict)
        and identity.get("revision") == REVISION
        and same(
            identity.get("config"),
            {k: v for k, v in config.items() if k not in {"max_wall_s", "resume"}},
        )
        and identity.get("task_protocol") == "sadhana.qf3-abc-first-placement/1"
        and identity.get("expert_ground_truth") is False,
        "Root resolved identity differs",
    )
    recipe = identity.get("scientific_recipe")
    require(
        isinstance(recipe, dict)
        and recipe.get("author_reproduction") is False
        and recipe.get("author_initializer_confirmed") is False,
        "Explicit nonreproduction recipe required",
    )
    startup_pins, worker_probe = _worker(value["worker"], deadline)
    verify_pin(value["software_binding"])
    software = read_json(value["software_binding"]["path"])
    require(
        software.get("kind") == "root_abc_visual_worker_release_binding"
        and software.get("status") == "accepted_cpu_source_wheel_install"
        and software.get("nrh_version") == NRH_VERSION
        and software.get("nrh_sources")
        == {n: p["sha256"] for n, p in value["worker"]["nrh_files"].items()},
        "Final root worker source/wheel binding required; provisional source cannot launch",
    )
    verify_pin(software["source_admission"])
    evaluator_sources = {
        name: value["worker"]["nrh_files"]["nrh/" + name + ".py"]["sha256"]
        for name in (
            "visual_policy_export",
            "visual_method_evaluation",
            "evaluation_video",
            "qf3_visual_policy",
        )
    }
    require(
        identity.get("evaluator_sources") == evaluator_sources,
        "Resolved installed evaluator sources differ",
    )
    verify_pin(value["native_admission"])
    native = read_json(value["native_admission"]["path"])
    require(
        type(native.get("schema_version")) is int
        and native["schema_version"] == 1
        and native.get("kind") == "visual_method_native_evaluation_admission"
        and native.get("status") == "accepted_for_serial_native_development_evaluation"
        and type(native.get("requested_episode_count")) is int
        and native["requested_episode_count"] == 50
        and native.get("campaign_directory") == value["campaign_directory"]
        and native.get("device") == "cuda:0"
        and native.get("evaluator_sources") == evaluator_sources
        and native.get("protocol_id") == hashlib.sha256(canonical(identity)).hexdigest()
        and native.get("cohort") == config["cohort"]
        and native.get("construction_config") == config["construction_config"]
        and config["protocol"] in native.get("allowed_protocols", [])
        and same(native.get("worker_binding"), value["worker"])
        and same(native.get("coordinator_sources"), value["coordinator_sources"])
        and same(native.get("native_resources"), resources),
        "Root native/runtime/worker admission differs",
    )
    learned = PROTOCOLS[config["protocol"]]
    require(
        (value["checkpoint_admission"] is not None) == learned
        and (config["export"] is not None) == learned,
        "Separate learned checkpoint/export trust required",
    )
    if learned:
        verify_pin(value["checkpoint_admission"])
        verify_pin(config["export"])
    request = EvaluationRequest(
        pin, value, config, cohort, identity, startup_pins, worker_probe
    )
    resume_rows(request)
    return request


def worker_command(
    request: EvaluationRequest, config: Path, out: Path, campaign: Path
) -> list[str]:
    value = request.value
    command = [
        value["worker"]["python"]["path"],
        "-I",
        "-B",
        "-m",
        "nrh.visual_method_evaluation",
        "--config",
        str(config.absolute()),
        "--out",
        str(out.absolute()),
        "--campaign-directory",
        str(campaign.absolute()),
    ]
    for label in ("native", "checkpoint"):
        pin = value[label + "_admission"]
        if pin is not None:
            flag = "--trusted-" + label + "-admission"
            command += [
                flag,
                pin["path"],
                flag + "-sha256",
                pin["sha256"],
                flag + "-bytes",
                str(pin["bytes"]),
            ]
    return command


def _timing_samples(row: dict) -> None:
    """Validate the shared evaluator's retained host-phase samples, not speed."""
    samples = row["control_wall_samples"]
    returned = row["returned_controls"]
    require(
        returned <= len(samples) <= returned + (not row["complete"]),
        "Timing samples must retain every returned control and at most one failed loop",
    )
    phases = ("selection_dispatch", "step", "capture", "whole_loop")
    for index, sample in enumerate(samples, 1):
        require(
            isinstance(sample, dict)
            and set(sample) == {"iteration", "returned_control_index", *phases}
            and type(sample["iteration"]) is int
            and sample["iteration"] == index,
            "Exact typed ordered timing sample required",
        )
        control = sample["returned_control_index"]
        require(
            (type(control) is int and control == index and control <= returned)
            or (
                control is None
                and not row["complete"]
                and index == len(samples) == returned + 1
            ),
            "Timing returned-control causal index differs",
        )
        for phase in phases:
            value = sample[phase]
            require(
                (value is None and not row["complete"] and phase in {"step", "capture"})
                or (
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                ),
                "Finite nonnegative host timing or legitimate partial phase null required",
            )
        require(
            (
                sample["step"] is not None
                or (control is None and sample["capture"] is None)
            )
            and (sample["capture"] is None or control is not None)
            and (control is None or sample["step"] is not None)
            and (
                sample["step"] is None
                or control is not None
                or row["unknown_physical_attempts"] == 1
            ),
            "Timing phase/physical-return causality differs",
        )
        require(
            sum(sample[p] or 0 for p in phases[:-1]) <= sample["whole_loop"] + 1e-8,
            "Host phase timings exceed their enclosing whole loop",
        )


def _rows(request: EvaluationRequest, receipt: dict) -> list[dict]:
    require(
        type(receipt.get("schema_version")) is int
        and receipt["schema_version"] == 1
        and receipt.get("revision") == REVISION
        and receipt.get("domain") == "sim"
        and receipt.get("protocol_id") == request.protocol_id
        and same(receipt.get("identity"), request.identity)
        and same(
            receipt.get("scientific_recipe"), request.identity["scientific_recipe"]
        )
        and receipt.get("author_reproduction") is False
        and receipt.get("expert_ground_truth") is False
        and receipt.get("native_world_restored") is False
        and type(receipt.get("requested_episode_count")) is int
        and receipt["requested_episode_count"] == 50,
        "Worker receipt identity/claims differ",
    )
    rows = receipt.get("episodes")
    require(
        isinstance(rows, list) and len(rows) <= 50, "Bounded episode prefix required"
    )
    for ordinal, row in enumerate(rows):
        expected = request.cohort["episodes"][ordinal]
        require(
            isinstance(row, dict)
            and row.get("id") == expected["id"]
            and type(row.get("requested_seed")) is int
            and row["requested_seed"] == expected["seed"]
            and type(row.get("complete")) is bool
            and type(row.get("policy_unchanged")) is bool,
            "Typed ordered worker episode differs",
        )
        returned = count(
            row.get("returned_controls"), "row returned", request.config["control_cap"]
        )
        captured = count(row.get("captured_judged_controls"), "row captured", returned)
        unknown = count(row.get("unknown_physical_attempts"), "row unknown", 1)
        require(
            isinstance(row.get("commands"), list)
            and len(row["commands"]) <= returned
            and isinstance(row.get("control_wall_samples"), list)
            and len(row["control_wall_samples"]) <= request.config["control_cap"],
            "Bounded worker commands/latency required",
        )
        _timing_samples(row)
        for index, command in enumerate(row["commands"], 1):
            require(
                type(command.get("index")) is int and command["index"] == index,
                "Command index differs",
            )
            for key in ("physical", "u", "z"):
                values = command.get(key)
                require(
                    isinstance(values, list)
                    and len(values) == 14
                    and all(
                        type(x) in (int, float) and math.isfinite(x) for x in values
                    ),
                    "Finite actual command vector required",
                )
        if row["complete"]:
            require(
                row["policy_unchanged"] is True
                and type(row.get("success")) is bool
                and type(row.get("native_success")) is bool
                and returned > 0
                and captured == returned
                and unknown == 0
                and len(row["commands"]) == len(row["control_wall_samples"]) == returned
                and (row["success"] or returned == 1000)
                and row.get("status") == ("success" if row["success"] else "timeout"),
                "Healthy complete episode boundary required",
            )
            if request.config["protocol"].startswith("qf3"):
                policy = row.get("qf3_policy")
                require(
                    isinstance(policy, dict)
                    and policy.get("selection_rng") is None
                    and policy.get("preservation_before")
                    == policy.get("preservation_after")
                    and isinstance(policy.get("preservation_before"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", policy["preservation_before"])
                    is not None,
                    "QF3 no-update row evidence differs",
                )
        else:
            require(ordinal == len(rows) - 1, "Only the last row can be partial")
    return rows


def resume_rows(request: EvaluationRequest) -> list[dict]:
    rows = []
    seen = set()
    for pin in request.config["resume"]:
        require(pin["sha256"] not in seen, "Duplicate resume receipt pin")
        seen.add(pin["sha256"])
        verify_pin(pin, request.value["receipt_max_bytes"])
        receipt = read_json(pin["path"], request.value["receipt_max_bytes"])
        require(
            receipt.get("status") in {"completed", "incomplete"}
            and receipt.get("error") is None
            and receipt.get("secondary_errors") == [],
            "Failed invocation cannot grant resume credit",
        )
        candidates = _rows(request, receipt)
        for row in candidates:
            if row["complete"]:
                require(
                    len(rows) < 50
                    and row["id"] == request.cohort["episodes"][len(rows)]["id"],
                    "Duplicate/noncontiguous resume prefix",
                )
                rows.append(row)
    return rows


def summarize_worker(
    request: EvaluationRequest, receipt: dict, exit_code: int | None
) -> dict:
    rows = _rows(request, receipt)
    status = receipt.get("status")
    require(
        receipt.get("error") is None or isinstance(receipt["error"], str),
        "Typed worker error required",
    )
    require(
        isinstance(receipt.get("secondary_errors"), list)
        and all(isinstance(error, str) for error in receipt["secondary_errors"]),
        "Typed worker secondary errors required",
    )
    require(
        status in {"completed", "incomplete", "failed"}
        and type(exit_code) is int
        and (
            (status in {"completed", "incomplete"} and exit_code == 0)
            or (status == "failed" and exit_code != 0)
        ),
        "Evaluator exit code and status differ",
    )
    clean = receipt.get("error") is None and receipt.get("secondary_errors") == []
    require(
        status == "failed" or clean,
        "Nonfailed invocation has preservation/cleanup errors",
    )
    prefix = resume_rows(request)
    require(
        len(rows) >= len(prefix) and all(same(a, b) for a, b in zip(rows, prefix)),
        "Worker changed completed resume prefix",
    )
    new = rows[len(prefix) :]
    counts = receipt.get("counts")
    require(
        isinstance(counts, dict)
        and set(counts)
        == {
            "attempted_dispatches",
            "returned_controls",
            "captured_judged_controls",
            "unknown_physical_attempts",
        },
        "Exact invocation counter fields required",
    )
    for key, value in counts.items():
        count(value, key)
    returned = sum(r["returned_controls"] for r in new)
    captured = sum(r["captured_judged_controls"] for r in new)
    unknown = sum(r["unknown_physical_attempts"] for r in new)
    require(
        counts
        == {
            "attempted_dispatches": returned + unknown,
            "returned_controls": returned,
            "captured_judged_controls": captured,
            "unknown_physical_attempts": unknown,
        },
        "Invocation physics counters do not reconcile; resumed controls cannot be borrowed",
    )
    if request.config["protocol"].startswith("qf3"):
        require(
            receipt.get("counts_scope")
            == "current invocation; completed resume prefix excluded"
            and type(receipt.get("cumulative_returned_controls")) is int
            and receipt["cumulative_returned_controls"]
            == sum(r["returned_controls"] for r in rows),
            "QF3 invocation/cumulative scope differs",
        )
    completed = [r for r in rows if r["complete"]]
    successes = sum(r["success"] for r in completed)
    native_successes = sum(r["native_success"] for r in completed)
    valid_score = (
        status == "completed"
        and clean
        and len(completed) == 50
        and all(r["policy_unchanged"] for r in completed)
    )
    require(
        status != "completed" or valid_score,
        "Completed receipt lacks healthy full cohort",
    )
    successful = [r for r in completed if r["success"]]
    expected = {
        "requested": 50,
        "completed": len(completed),
        "partial": len(rows) - len(completed),
        "unstarted": 50 - len(rows),
        "successes": successes,
        "complete_success_rate": successes / len(completed) if completed else None,
        "success_definition": "five distinct first placements in episode history",
        "native_current_successes": native_successes,
        "native_current_complete_success_rate": native_successes / len(completed)
        if completed
        else None,
        "mean_controls_success_only": sum(r["returned_controls"] for r in successful)
        / len(successful)
        if successful
        else None,
        "mean_controls_all_complete": sum(r["returned_controls"] for r in completed)
        / len(completed)
        if completed
        else None,
        "success_bounds_over_requested": [
            successes / 50,
            (successes + 50 - len(completed)) / 50,
        ],
        "full_cohort_score": successes / 50 if valid_score else None,
    }
    require(
        same(receipt.get("metrics"), expected),
        "Worker summary metrics disagree with episode evidence",
    )
    completed_controls = sum(r["returned_controls"] for r in new if r["complete"])
    accepted = completed_controls if status != "failed" and clean else 0
    return {
        "status": status,
        "worker_status": status,
        "producer_evaluation_metrics": expected,
        "performance_admission": "pending_independent_reader_and_root",
        "scientific_recipe": receipt["scientific_recipe"],
        "protocol_id": request.protocol_id,
        "reused_completed_episodes": len(prefix),
        "new_completed_episodes": len(completed) - len(prefix),
        "metrics": {
            "simulation_steps": returned,
            "accepted_complete_control_steps": accepted,
            "reported_complete_row_control_steps": completed_controls,
            "discarded_partial_control_steps": returned - completed_controls,
            "captured_judged_controls": captured,
            "returned_uncaptured_controls": returned - captured,
            "unknown_physical_attempts": unknown,
            "attempted_dispatches": returned + unknown,
            "updates": 0,
            "successes": None,
            "episodes": None,
            "latency_ms": {"p50": None, "p95": None},
            "counters_scope": "producer current invocation physics only; complete resume prefix excluded",
        },
    }
