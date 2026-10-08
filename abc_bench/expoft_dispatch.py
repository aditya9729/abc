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
REVISION = "sadhana.realtime-expoft-worker/2"
MODES = {"paused_simulation_fixed_tick", "strict_wall"}
STATUSES = {"completed", "budget_stopped", "deadline_missed_partial", "failed"}
WRAPPER_COUNTS = {
    "completed_episodes",
    "issued_steps",
    "update_debt",
    "accepted_update_calls",
    "checkpoint_serial",
}
LEARNER_COUNTS = {
    "critic",
    "critic_target",
    "noise",
    "editor",
    "temperature",
    "base",
    "auxiliary",
    "update_calls",
    "skipped_base",
}


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
        type(config.get("schema_version")) is int and config["schema_version"] == 2,
        "EXPO coordinator requires canonical schema2",
    )
    storage = config.get("checkpoint_storage")
    _require(
        isinstance(storage, dict)
        and set(storage) == {"format", "schema_version"}
        and storage["format"] == "immutable_episode_blocks"
        and type(storage["schema_version"]) is int
        and storage["schema_version"] == 1,
        "EXPO native coordinator requires immutable episode block storage1",
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


def _wrapper(value: Any, *, initial: bool) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == WRAPPER_COUNTS | {"phase"},
        "EXPO wrapper counter schema differs",
    )
    for key in WRAPPER_COUNTS:
        _count(value[key], key)
    phases = (
        {"ready", "pending_updates"}
        if initial
        else {"ready", "pending_updates", "collecting"}
    )
    _require(
        isinstance(value["phase"], str) and value["phase"] in phases,
        "EXPO wrapper phase differs",
    )
    _require(
        value["phase"] != "ready" or value["update_debt"] < 30,
        "EXPO ready boundary retains due update debt",
    )
    return value


def _learner_counts(value: Any, calls: int, *, failed: bool) -> dict[str, int]:
    _require(
        isinstance(value, dict) and set(value) == LEARNER_COUNTS,
        "EXPO learner counter schema differs",
    )
    for key in LEARNER_COUNTS:
        _count(value[key], "learner " + key)
    expected = {
        **{key: 20 * calls for key in ("critic", "noise", "critic_target")},
        **{key: calls for key in ("editor", "temperature", "update_calls")},
    }
    # An interrupted optimizer group may have phase counts beyond the accepted
    # prefix. Only the wrapper's fully retained groups receive update credit.
    _require(
        all(value[k] >= v if failed else value[k] == v for k, v in expected.items())
        and (
            value["base"] + value["skipped_base"] >= calls
            if failed
            else value["base"] + value["skipped_base"] == calls
        )
        and (
            value["auxiliary"] <= value["base"]
            if failed
            else value["auxiliary"] == value["base"]
        ),
        "EXPO learner and accepted update groups do not reconcile",
    )
    return value


def summarize_worker(
    worker: dict[str, Any], config: dict[str, Any], *, exit_code: int | None = None
) -> dict[str, Any]:
    _require(
        worker.get("revision") == REVISION and worker.get("domain") == "sim",
        "EXPO worker revision/domain differs",
    )
    status = worker.get("status")
    _require(
        isinstance(status, str) and status in STATUSES, "EXPO worker status differs"
    )
    if exit_code is not None:
        _require(
            type(exit_code) is int
            and (
                status == "completed"
                and exit_code == 0
                or status in {"budget_stopped", "deadline_missed_partial"}
                and exit_code == 2
                or status == "failed"
                and exit_code not in {0, 2}
            ),
            "EXPO worker exit code and status differ",
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
    _require(
        identity.get("diagnostic_fixture") is False
        and type(identity.get("learning_starts")) is int
        and identity["learning_starts"] == 10
        and type(identity.get("step_interval")) is int
        and identity["step_interval"] == 30,
        "EXPO native update clock identity differs",
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
    initial = _wrapper(invocation.get("initial_wrapper"), initial=True)
    wrapper = _wrapper(wrapper, initial=False)
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
    issued_delta = wrapper["issued_steps"] - initial["issued_steps"]
    _require(
        issued_delta >= 0
        and accepted <= issued_delta <= physical
        and (status == "failed" or issued_delta == physical)
        and wrapper["checkpoint_serial"] >= initial["checkpoint_serial"],
        "EXPO invocation controls differ from issued wrapper counters",
    )
    episodes = _count(wrapper.get("completed_episodes"), "completed episodes")
    initial_episodes = _count(initial.get("completed_episodes"), "initial episodes")
    calls = _count(wrapper.get("accepted_update_calls"), "accepted update calls")
    initial_calls = _count(initial.get("accepted_update_calls"), "initial update calls")
    _require(
        initial_episodes <= episodes <= config["episodes"] and initial_calls <= calls,
        "EXPO completed invocation counters regressed",
    )
    new_episodes = episodes - initial_episodes
    _require(
        accepted >= new_episodes and (new_episodes > 0 or accepted == 0),
        "EXPO accepted controls differ from completed episode prefix",
    )
    if initial_episodes <= 10:
        _require(
            initial_calls == 0 and initial["update_debt"] == 0,
            "EXPO warmup prefix contains learner update credit",
        )
    warmup = _count(
        invocation.get("accepted_warmup_control_steps"), "accepted warmup controls"
    )
    learned = _count(
        invocation.get("accepted_learning_control_steps"), "accepted learning controls"
    )
    new_warmup_episodes = min(episodes, 10) - min(initial_episodes, 10)
    new_learning_episodes = episodes - max(initial_episodes, min(episodes, 10))
    _require(
        warmup + learned == accepted
        and warmup >= new_warmup_episodes
        and learned >= new_learning_episodes
        and (new_warmup_episodes > 0 or warmup == 0)
        and (new_learning_episodes > 0 or learned == 0),
        "EXPO completed warmup/learning control partition differs",
    )
    learned_delta = (
        wrapper["update_debt"] - initial["update_debt"] + 30 * (calls - initial_calls)
    )
    _require(
        learned_delta == learned,
        "EXPO update debt differs from accepted learning controls",
    )
    if episodes <= 10:
        _require(
            calls == 0 and wrapper["update_debt"] == 0,
            "EXPO warmup boundary contains learner update credit",
        )
    before = _learner_counts(
        invocation.get("initial_learner_counters"), initial_calls, failed=False
    )
    after = _learner_counts(
        worker.get("learner_counters"), calls, failed=status == "failed"
    )
    _require(
        all(after[k] >= before[k] for k in LEARNER_COUNTS),
        "EXPO learner counters regressed",
    )
    _require(
        isinstance(worker.get("accepted_update_groups"), list)
        and len(worker["accepted_update_groups"]) == calls,
        "EXPO accepted update-group count differs",
    )
    for ordinal, group in enumerate(worker["accepted_update_groups"]):
        _require(
            isinstance(group, dict)
            and set(group) == {"ordinal", "pin"}
            and isinstance(group["pin"], dict)
            and type(group.get("ordinal")) is int
            and group["ordinal"] == ordinal,
            "EXPO accepted update-group ordinal differs",
        )
    _require(
        _count(worker.get("successful_base_updates"), "producer base updates")
        == after["base"]
        and _count(worker.get("skipped_base_updates"), "producer skipped base updates")
        == after["skipped_base"],
        "EXPO producer base counters differ from learner",
    )
    expected_learning = (
        "components_exercised_no_gain_claim"
        if after["base"] > 0
        else "awaiting_success_imitation_data"
        if after["skipped_base"] > 0
        else "no_accepted_update_groups"
    )
    _require(
        worker.get("learning_status") == expected_learning,
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
            "accepted_warmup_control_steps": warmup,
            "accepted_learning_control_steps": learned,
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
