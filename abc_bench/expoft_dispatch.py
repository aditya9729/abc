"""Lightweight coordinator contract for the named ABC/Torch EXPO training port.

Worker values remain producer evidence. This module never loads checkpoints,
imports a learner, initializes a simulator or turns training into evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PROFILE = "abc-vla/released-config-first/1"
REVISION = "sadhana.realtime-expoft-worker/1"
MODES = {"paused_simulation_fixed_tick", "strict_wall"}
STATUSES = {"completed", "budget_stopped", "deadline_missed_partial", "failed"}


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _count(value: Any, name: str) -> int:
    _require(type(value) is int and value >= 0, "Invalid EXPO " + name)
    return value


def read_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate EXPO JSON key: " + key)
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError("Nonfinite EXPO JSON value: " + value)

    result = json.loads(
        path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid
    )
    _require(isinstance(result, dict), "EXPO JSON object required")
    json.dumps(result, allow_nan=False)
    return result


def training_input(path: Path) -> dict[str, Any]:
    config = read_json(path)
    _require(
        type(config.get("schema_version")) is int and config["schema_version"] == 1,
        "EXPO config requires schema1",
    )
    _require(
        config.get("domain") == "sim" and config.get("profile") == PROFILE,
        "EXPO coordinator requires the native ABC/Torch profile",
    )
    _require(
        isinstance(config.get("mode"), str) and config["mode"] in MODES,
        "EXPO clock mode differs",
    )
    _require(
        _count(config.get("episodes"), "episodes") > 0, "EXPO episodes must be positive"
    )
    maximum = config.get("max_wall_s")
    _require(
        type(maximum) in (int, float) and math.isfinite(maximum) and maximum > 0,
        "EXPO config requires finite positive max_wall_s",
    )
    _require(
        isinstance(config.get("artifacts"), dict)
        and isinstance(config["artifacts"].get("checkpoint_path"), str)
        and Path(config["artifacts"]["checkpoint_path"]).is_absolute(),
        "EXPO config requires an absolute candidate checkpoint path",
    )
    # The worker's lightweight resolver validates its complete versioned schema.
    # The native factory then checks the live coordinator lease before model load.
    return config


def summarize_worker(worker: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    _require(
        worker.get("revision") == REVISION and worker.get("domain") == "sim",
        "EXPO worker revision/domain differs",
    )
    status = worker.get("status")
    _require(
        isinstance(status, str) and status in STATUSES, "EXPO worker status differs"
    )
    identity = worker.get("identity")
    _require(
        isinstance(identity, dict)
        and identity.get("revision") == REVISION
        and json.dumps(identity.get("config"), sort_keys=True, allow_nan=False)
        == json.dumps(
            {k: v for k, v in config.items() if k != "max_wall_s"},
            sort_keys=True,
            allow_nan=False,
        ),
        "EXPO worker input identity differs",
    )
    _require(worker.get("clock_mode") == config["mode"], "EXPO worker clock differs")
    # Training receipts are never an independent native or author reproduction.
    _require(
        worker.get("independent_native_admission") is False
        and worker.get("author_reproduction") is False
        and worker.get("wall_time_reactivity_verified") is False,
        "EXPO training worker cannot self-admit performance or deadlines",
    )
    invocation, wrapper = worker.get("invocation"), worker.get("wrapper")
    _require(
        isinstance(invocation, dict) and isinstance(wrapper, dict),
        "EXPO counters missing",
    )
    initial = invocation.get("initial_wrapper")
    _require(isinstance(initial, dict), "EXPO invocation initial counters missing")
    physical = _count(invocation.get("physical_control_steps"), "physical controls")
    accepted = _count(
        invocation.get("accepted_completed_control_steps"), "accepted controls"
    )
    discarded = _count(
        invocation.get("discarded_partial_control_steps"), "discarded controls"
    )
    _require(
        physical == accepted + discarded, "EXPO invocation controls do not reconcile"
    )
    episodes = _count(wrapper.get("completed_episodes"), "completed episodes")
    initial_episodes = _count(initial.get("completed_episodes"), "initial episodes")
    calls = _count(wrapper.get("accepted_update_calls"), "accepted update calls")
    initial_calls = _count(initial.get("accepted_update_calls"), "initial update calls")
    _require(
        initial_episodes <= episodes <= config["episodes"] and initial_calls <= calls,
        "EXPO completed invocation counters regressed",
    )
    _require(
        isinstance(worker.get("accepted_update_groups"), list)
        and len(worker["accepted_update_groups"]) == calls,
        "EXPO accepted update-group count differs",
    )
    for ordinal, group in enumerate(worker["accepted_update_groups"]):
        _require(
            isinstance(group, dict)
            and type(group.get("ordinal")) is int
            and group["ordinal"] == ordinal,
            "EXPO accepted update-group ordinal differs",
        )
    _count(worker.get("successful_base_updates"), "producer base updates")
    _require(
        isinstance(worker.get("learning_status"), str)
        and worker["learning_status"]
        in {"components_exercised_no_gain_claim", "awaiting_success_imitation_data"},
        "EXPO learning status differs",
    )
    if status == "completed":
        _require(
            episodes == config["episodes"]
            and wrapper.get("phase") == "ready"
            and discarded == 0,
            "EXPO completed worker has an incomplete episode/update boundary",
        )
    return {
        "worker_status": status,
        "episode_schedule_completed": status == "completed",
        "independent_admission_required": True,
        "learning_status": worker["learning_status"],
        "clock_mode": worker["clock_mode"],
        "wall_time_reactivity_verified": False,
        "metrics": {
            "simulation_steps": physical,
            "accepted_completed_control_steps": accepted,
            "discarded_partial_control_steps": discarded,
            "completed_training_episodes": episodes - initial_episodes,
            "cumulative_completed_training_episodes": episodes,
            "updates": calls - initial_calls,
            "cumulative_accepted_update_calls": calls,
            "successes": None,
            "episodes": None,
            "latency_ms": {"p50": None, "p95": None},
            "counters_scope": "invocation controls, including warmup; accepted complete update calls only",
        },
    }
