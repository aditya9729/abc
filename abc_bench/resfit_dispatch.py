"""Validate producer training receipts without loading ResFiT or native artifacts.

Returned controls, validated replay rows, completed episodes and optimizer work
have different scopes. This reader preserves them without self-admitting any
checkpoint, learning result, real-time deadline or benchmark performance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from abc_bench.expoft_dispatch import read_json as read_training_json

PROFILE = "resfit-abc-vla-visual-paper-mse-unprojected/v1"
DATA_PROFILE = "resfit-abc-vla-visual-online-only/v1"
REVISION = "sadhana.resfit-abc-vla-worker/1"
SCHEDULE = {"warmup_controls": 10000, "critic_warmup_updates": 10000, "batch_size": 256}
WRAPPER_COUNTS = {
    "controls",
    "episodes",
    "critic_warmup_updates",
    "ordinary_critic_updates",
    "actor_updates",
    "debt",
    "checkpoint_serial",
}
LEARNER_COUNTS = {
    "critic_updates",
    "critic_warmup_updates",
    "ordinary_critic_updates",
    "actor_updates",
    "critic_target_updates",
    "actor_target_updates",
}
CONFIG_FIELDS = {
    "schema_version",
    "domain",
    "profile",
    "data_profile",
    "seed",
    "sampler_seed",
    "replay_seed",
    "learner_seed",
    "episodes",
    "target_controls",
    "replay_capacity",
    "max_wall_s",
    "diagnostic",
    "artifacts",
    "sources",
    "native_inventory",
}


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _count(value: Any, name: str) -> int:
    _require(type(value) is int and value >= 0, "Invalid ResFiT " + name)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return read_training_json(path, method="ResFiT")


def _same(left: Any, right: Any) -> bool:
    # JSON distinguishes integer, floating-point and Boolean aliases.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def training_input(path: Path) -> dict[str, Any]:
    config = read_json(path)
    _require(set(config) == CONFIG_FIELDS, "ResFiT coordinator config schema differs")
    _require(
        type(config["schema_version"]) is int and config["schema_version"] == 1,
        "ResFiT coordinator requires schema1",
    )
    _require(
        config["domain"] == "sim"
        and config["profile"] == PROFILE
        and config["data_profile"] == DATA_PROFILE,
        "ResFiT native visual online-only profile required",
    )
    _require(
        config["diagnostic"] is None
        and type(config["replay_capacity"]) is int
        and config["replay_capacity"] == 200000,
        "ResFiT native recipe requires full 200000-control capacity",
    )
    for key in ("seed", "sampler_seed", "replay_seed", "learner_seed"):
        _require(_count(config[key], key) < 2**63, "ResFiT seed bounds differ")
    for key in ("episodes", "target_controls"):
        _require(_count(config[key], key) > 0, "ResFiT run limits must be positive")
    _require(
        config["seed"] + config["episodes"] + 100000 < 2**63,
        "ResFiT reset seed bounds differ",
    )
    maximum = config["max_wall_s"]
    _require(
        type(maximum) in (int, float) and math.isfinite(maximum) and maximum > 0,
        "ResFiT config requires finite positive max_wall_s",
    )
    for key in ("artifacts", "sources", "native_inventory"):
        _require(
            isinstance(config[key], dict), "ResFiT explicit local pin mappings required"
        )
    candidate = config["artifacts"].get("checkpoint_path")
    _require(
        isinstance(candidate, str) and Path(candidate).is_absolute(),
        "ResFiT requires absolute candidate checkpoint path",
    )
    # The worker validates remaining source/artifact schemas and the live lease.
    # The coordinator does not decode weights or silently reduce the recipe.
    return config


def _wrapper(value: Any, *, initial: bool) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == WRAPPER_COUNTS | {"phase"},
        "ResFiT wrapper schema differs",
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
        "ResFiT wrapper phase differs",
    )
    controls, warmup, ordinary = (
        value[k]
        for k in ("controls", "critic_warmup_updates", "ordinary_critic_updates")
    )
    _require(
        warmup <= 10000
        and ordinary <= max(0, controls - 10000) * 4
        and value["debt"] == max(0, controls - 10000) * 4 - ordinary
        and value["actor_updates"] == ordinary // 4,
        "ResFiT wrapper update schedule/debt differs",
    )
    _require(
        (controls >= 10000 or warmup == 0) and (ordinary == 0 or warmup == 10000),
        "ResFiT warmup ordering differs",
    )
    _require(
        value["episodes"]
        <= controls
        <= 1000 * (value["episodes"] + int(value["phase"] == "collecting")),
        "ResFiT replay controls differ from episode boundaries",
    )
    if value["phase"] == "ready":
        _require(
            value["debt"] == 0 and (controls < 10000 or warmup == 10000),
            "ResFiT ready boundary retains due updates",
        )
    return value


def summarize_worker(
    worker: dict[str, Any], config: dict[str, Any], *, exit_code: int | None = None
) -> dict[str, Any]:
    _require(
        type(worker.get("schema_version")) is int
        and worker["schema_version"] == 1
        and worker.get("revision") == REVISION
        and worker.get("domain") == "sim",
        "ResFiT worker schema/revision/domain differs",
    )
    status = worker.get("status")
    _require(
        isinstance(status, str) and status in {"completed", "budget_stopped", "failed"},
        "ResFiT worker status differs",
    )
    if exit_code is not None:
        _require(
            type(exit_code) is int
            and (
                status == "completed"
                and exit_code == 0
                or status == "budget_stopped"
                and exit_code == 2
                or status == "failed"
                and exit_code not in {0, 2}
            ),
            "ResFiT worker exit code and status differ",
        )
    identity = worker.get("identity")
    _require(
        isinstance(identity, dict)
        and identity.get("revision") == REVISION
        and _same(
            identity.get("config"),
            {k: v for k, v in config.items() if k != "max_wall_s"},
        )
        and _same(identity.get("schedule"), SCHEDULE)
        and identity.get("data_profile") == DATA_PROFILE
        and type(identity.get("n_step")) is int
        and identity["n_step"] == 3
        and type(identity.get("gamma")) is float
        and identity["gamma"] == 0.99,
        "ResFiT input identity/schedule differs",
    )
    for flag in (
        "native_independent_admission",
        "native_physics_resume",
        "task_gain",
        "paper_reproduction",
        "matched_evaluation_implemented",
        "expert_ground_truth",
    ):
        _require(
            worker.get(flag) is False,
            "ResFiT training worker cannot self-admit " + flag,
        )
    _require(
        worker.get("full_recipe") is True
        and worker.get("data_profile") == DATA_PROFILE
        and type(worker.get("offline_rows")) is int
        and worker["offline_rows"] == 0
        and type(worker.get("reference_replay_capacity")) is int
        and worker["reference_replay_capacity"] == 200000,
        "ResFiT online-only full recipe differs",
    )
    invocation = worker.get("invocation")
    _require(isinstance(invocation, dict), "ResFiT invocation counts missing")
    initial = _wrapper(invocation.get("initial_wrapper"), initial=True)
    final = _wrapper(worker.get("wrapper"), initial=False)
    _require(
        all(final[k] >= initial[k] for k in WRAPPER_COUNTS - {"debt"}),
        "ResFiT wrapper counters regressed",
    )
    physical = _count(
        invocation.get("physical_control_steps"), "known physical controls"
    )
    accepted = _count(
        invocation.get("accepted_complete_control_steps"), "completed controls"
    )
    discarded = _count(
        invocation.get("discarded_partial_control_steps"), "discarded controls"
    )
    warmup = _count(invocation.get("warmup_control_steps"), "physical warmup controls")
    learned = _count(
        invocation.get("learning_control_steps"), "physical learning controls"
    )
    attempts = _count(invocation.get("control_dispatch_attempts"), "dispatch attempts")
    unknown = _count(
        invocation.get("native_failed_step_calls_with_unknown_physics"),
        "unknown-physics calls",
    )
    replay_delta = final["controls"] - initial["controls"]
    _require(
        physical == accepted + discarded == warmup + learned
        and accepted <= replay_delta <= physical
        and attempts == physical + unknown,
        "ResFiT physical/completed/replay controls do not reconcile",
    )
    _require(
        status == "failed"
        or (replay_delta == physical and attempts == physical and unknown == 0),
        "ResFiT nonfailed invocation has missing returned controls",
    )
    expected_warmup = min(physical, max(0, 10000 - initial["controls"]))
    _require(
        warmup == expected_warmup, "ResFiT physical warmup/learning partition differs"
    )
    new_episodes = final["episodes"] - initial["episodes"]
    _require(
        new_episodes <= accepted <= 1000 * new_episodes
        and (new_episodes > 0 or accepted == 0)
        and final["episodes"] <= config["episodes"],
        "ResFiT completed episode prefix differs",
    )
    if final["phase"] != "collecting":
        _require(
            accepted == replay_delta and discarded == 0,
            "ResFiT complete boundary retains partial replay controls",
        )
    counters = worker.get("learner_counters")
    _require(
        isinstance(counters, dict) and set(counters) == LEARNER_COUNTS,
        "ResFiT learner counter schema differs",
    )
    for key in LEARNER_COUNTS:
        _count(counters[key], "learner " + key)
    expected = {
        "critic_warmup_updates": final["critic_warmup_updates"],
        "ordinary_critic_updates": final["ordinary_critic_updates"],
        "critic_updates": final["critic_warmup_updates"]
        + final["ordinary_critic_updates"],
        "critic_target_updates": final["critic_warmup_updates"]
        + final["ordinary_critic_updates"],
        "actor_updates": final["actor_updates"],
        "actor_target_updates": final["actor_updates"],
    }
    _require(
        all(
            counters[k] >= count if status == "failed" else counters[k] == count
            for k, count in expected.items()
        ),
        "ResFiT learner counters differ from wrapper update prefix",
    )
    if status == "completed":
        _require(
            final["phase"] == "ready"
            and final["controls"] >= config["target_controls"]
            and final["controls"]
            <= max(initial["controls"], config["target_controls"] + 999),
            "ResFiT target has not completed at an episode/update boundary",
        )
    return {
        "worker_status": status,
        "training_target_reached": status == "completed",
        "independent_admission_required": True,
        "metrics": {
            "simulation_steps": physical,
            "accepted_complete_control_steps": accepted,
            "discarded_partial_control_steps": discarded,
            "warmup_physical_control_steps": warmup,
            "learning_physical_control_steps": learned,
            "validated_replay_rows": replay_delta,
            "completed_training_episodes": new_episodes,
            "cumulative_validated_replay_controls": final["controls"],
            "ordinary_critic_updates": final["ordinary_critic_updates"]
            - initial["ordinary_critic_updates"],
            "critic_warmup_updates": final["critic_warmup_updates"]
            - initial["critic_warmup_updates"],
            "actor_updates": final["actor_updates"] - initial["actor_updates"],
            "native_failed_step_calls_with_unknown_physics": unknown,
            "producer_learner_counters": counters,
            "successes": None,
            "episodes": None,
            "latency_ms": {"p50": None, "p95": None},
            "counters_scope": "known invocation physics, validated replay rows and producer optimizer operations; partial work is not admitted checkpoint credit",
        },
    }
