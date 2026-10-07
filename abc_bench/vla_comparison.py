"""Read pinned ABC-VLA development evidence; never construct or run a policy.

This deliberately does not consume the historical ABC-DiT ``paired_eval`` schema.
Artifact hashes authenticate bytes, not producer truth or selection chronology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

PROTOCOL = "sadhana.qf3-abc-first-placement/1"
TASK = "put_plastic_bottles_in_bin"
SEEDS = list(range(1000000, 1000050))
SAMPLER_SEED = 91001
HORIZON = 1000
COMMON_SOURCES = {
    "nrh.qf3": "06cd071b858c5885bd4849ca3a25c8f80b16894a5bc9ec372091125d2d1361c7",
    "nrh.qf3_abc_vla": "ca2f0addce30714081392bb7bc6928ddfa8a6d013841ad96c3111390d1a83eb3",
    "nrh.qf3_learning": "cc1aba70e2de7b9ddd4df295bec35137a8afd1a2b6670f9acdb836849bdb047f",
    "nrh.qf3_replay": "ded57c1cfe7c039160bd563404aa3d519a9e6aeb189c987512addf058ea5242a",
    "nrh.qf3_tasks": "ac29ddfcf6688551f270800f05c2d513f5dfbb4870c7c7cf0fc55824ee596016",
}
SOURCE_PROFILES = {
    "frozen": {
        **COMMON_SOURCES,
        "nrh.qf3_rollouts": "9efbef4a97bba44b408145a19b0f9ec1daf1513c2cd28889cdb936c5e5892cb2",
        "nrh.qf3_training": "2c477508b088c5e73f91f2fb3b5c87d9adef5ae8f1f5bd86a7943d913bde44bc",
    },
    "learned": {
        **COMMON_SOURCES,
        "nrh.qf3_rollouts": "39be24afd59e20c8afffa02370b18b3b9ead3f5fd1b4b4761834aba9966bdef4",
        "nrh.qf3_training": "50008416d361b64e5ed27e93deff47013dcfe92ba399d5b8e0a2479616971f61",
    },
}
SELECTION_RULE = {
    "actor_updates": 200,
    "boundary": "completed_outer",
    "capacity_run_fully_accepted": True,
    "completed_warmup_episodes": 320,
    "freeze_complete_checkpoint_sha_and_selection_manifest_before_development_evaluation": True,
    "full_fresh_validation_complete_episodes": 50,
    "nominal_target16000_may_be_unreached": True,
    "ordinary_critic_updates": 1600,
    "outer_iterations": 1,
    "partial": False,
    "pending_validation": None,
    "warmup_batches": 4,
}
CAPACITY_LEARNER = {
    "actor_learning_rate": 0.000045,
    "actor_loss": {
        "alpha": 0.5,
        "lambda_base": 1.0,
        "lambda_cfm": 0.01,
        "reduction": "mean_elements",
    },
    "actor_q_aggregation": "first",
    "actor_target_factor": 0.05,
    "critic_ensemble": 2,
    "critic_hidden": [256, 256],
    "critic_learning_rate": 0.0003,
    "critic_target_factor": 0.005,
    "gamma": 0.99,
    "gradient_clip": None,
    "optimizer_betas": [0.9, 0.999],
    "target_q_aggregation": "min",
    "timeout_bootstrap": True,
    "weight_decay": 0.0,
}


class ComparisonError(ValueError):
    """The supplied public evidence does not satisfy this comparison contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _same(left: Any, right: Any, message: str) -> None:
    # JSON types matter: True is not the integer 1 in a seed/counter contract.
    _require(_hash(left) == _hash(right), message)


def _artifact_same(left: dict[str, Any], right: dict[str, Any], message: str) -> None:
    # Optional producer labels are not artifact identity.
    _same(
        {key: left.get(key) for key in ("path", "bytes", "sha256")},
        {key: right.get(key) for key in ("path", "bytes", "sha256")},
        message,
    )


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ComparisonError(f"non-finite JSON number: {value}")


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str
    bytes: int | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Artifact:
        _require(isinstance(record, dict), "artifact record missing")
        _require(_integer(record.get("bytes")), "artifact byte size missing/invalid")
        return cls(record.get("path"), record.get("sha256"), record["bytes"])


class _Reader:
    def __init__(self) -> None:
        self.verified: dict[str, dict[str, Any]] = {}

    @staticmethod
    def signature(path: Path) -> tuple[int, int, int, int]:
        state = path.stat()
        return state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns

    def pin(self, item: Artifact) -> dict[str, Any]:
        _require(
            isinstance(item.path, str) and bool(item.path), "artifact path missing"
        )
        _require(_sha(item.sha256), "artifact SHA-256 missing/invalid")
        _require(
            item.bytes is None or _integer(item.bytes), "artifact byte size invalid"
        )
        path = Path(item.path).resolve()
        _require(path.is_file(), f"artifact missing: {path}")
        signature = self.signature(path)
        size = signature[1]
        if item.bytes is not None:
            _require(size == item.bytes, f"artifact size differs: {path}")
        # Re-read referenced binaries. This filesystem can preserve timestamps
        # across immediate same-size writes, so a stat-keyed hash cache is unsafe.
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        prior = {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}
        _require(
            self.signature(path) == signature,
            f"artifact changed during verification: {path}",
        )
        self.verified[str(path)] = prior
        _require(prior["sha256"] == item.sha256, f"artifact SHA-256 differs: {path}")
        return prior

    def json(self, item: Artifact) -> dict[str, Any]:
        _require(
            isinstance(item.path, str) and _sha(item.sha256),
            "JSON artifact path/hash invalid",
        )
        _require(
            item.bytes is None or _integer(item.bytes), "artifact byte size invalid"
        )
        path = Path(item.path).resolve()
        try:
            raw = path.read_bytes()
            _require(
                item.bytes is None or len(raw) == item.bytes,
                f"artifact size differs: {path}",
            )
            _require(
                hashlib.sha256(raw).hexdigest() == item.sha256,
                f"artifact SHA-256 differs: {path}",
            )
            self.verified[str(path)] = {
                "path": str(path),
                "bytes": len(raw),
                "sha256": item.sha256,
            }
            value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ComparisonError(f"invalid JSON artifact: {path}: {exc}") from exc
        _require(isinstance(value, dict), "JSON artifact must contain an object")
        return value

    def record(self, value: dict[str, Any]) -> dict[str, Any]:
        return self.json(Artifact.from_record(value))


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _matrix(value: Any, rows: int, columns: int, name: str) -> None:
    _require(
        isinstance(value, list) and len(value) == rows, f"{name} row shape differs"
    )
    for row in value:
        _require(
            isinstance(row, list)
            and len(row) == columns
            and all(_finite(x) for x in row),
            f"{name} columns/values differ",
        )


def _outcomes(
    tape: dict[str, Any],
    head: str,
    *,
    requested_seeds: list[int] = SEEDS,
    actual_seeds: list[int] | None = None,
    initial_zero: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute the tracker ledger and cross-check redundant per-world summaries."""
    _same(tape.get("schema_version"), 1, "tape schema differs")
    _require(tape.get("domain") == "sim", "non-native tape domain")
    _require(
        tape.get("protocol_id") == PROTOCOL, "legacy or incompatible task protocol"
    )
    stage = tape.get("stage")
    _require(
        isinstance(stage, str) and re.fullmatch(f"evaluate-{head}-[0-9]+", stage),
        "wrong rollout head/stage",
    )
    rollout = tape["rollout"]
    _require(rollout.get("partial") is False, "partial rollout refused")
    worlds = rollout.get("worlds")
    _require(
        isinstance(worlds, list) and len(worlds) == 50, "expected exactly 50 world rows"
    )
    actual_seeds = requested_seeds if actual_seeds is None else actual_seeds
    by_seed = {seed: [] for seed in requested_seeds}
    for decision in rollout.get("decisions", []):
        seed = decision.get("requested_seed")
        _require(
            type(seed) is int and seed in by_seed,
            "decision seed missing or outside cohort",
        )
        _same(
            decision.get("actual_seed"),
            actual_seeds[requested_seeds.index(seed)],
            "decision actual seed differs",
        )
        by_seed[seed].append(decision)
    results = []
    for index, world in enumerate(worlds):
        _same(world.get("world"), index, "missing/duplicate/reordered world index")
        _same(
            world.get("requested_seed"),
            requested_seeds[index],
            "configured world seed order differs",
        )
        _same(
            world.get("actual_seed"),
            actual_seeds[index],
            "actual world seed order differs",
        )
        tracker = world["tracker"]
        _require(
            tracker.get("protocol_revision") == PROTOCOL
            and tracker.get("task_id") == TASK,
            "tracker protocol differs",
        )
        _same(tracker.get("schema_version"), 1, "tracker schema differs")
        names = [f"bottle_{x}" for x in range(1, 6)]
        _same(tracker.get("object_names"), names, "tracker bottle names differ")
        initial = tracker.get("initial_mask")
        _require(
            isinstance(initial, list)
            and len(initial) == 5
            and all(type(x) is bool for x in initial),
            "invalid initial mask",
        )
        if initial_zero:
            _same(initial, [False] * 5, "nonzero initial placement mask")
        ever, step, reward = initial[:], 0, 0
        last_native = False
        decisions = by_seed[requested_seeds[index]]
        for decision_index, decision in enumerate(decisions):
            _same(
                decision.get("decision_index"),
                decision_index,
                "decision order/duplicate differs",
            )
            _require(
                decision.get("episode_id") == f"{stage}-0-world-{index}",
                "decision episode/world identity differs",
            )
            events = decision.get("events")
            _require(
                isinstance(events, list) and 1 <= len(events) <= 15,
                "invalid issued prefix duration",
            )
            if decision_index < len(decisions) - 1:
                _require(len(events) == 15, "short nonterminal action prefix")
            _matrix(
                decision.get("issued_normalized"),
                len(events),
                14,
                "issued normalized actions",
            )
            _matrix([decision.get("next_physical_state")], 1, 14, "next physical state")
            for event in events:
                _require(
                    not all(ever) and step < HORIZON,
                    "events continue beyond task completion",
                )
                step += 1
                _same(event.get("step"), step, "event step sequence differs")
                mask = event.get("ever_placed")
                _require(
                    isinstance(mask, list)
                    and len(mask) == 5
                    and all(type(x) is bool for x in mask),
                    "invalid event history mask",
                )
                _require(
                    all(not old or new for old, new in zip(ever, mask, strict=True)),
                    "placement history reversed",
                )
                newly = [
                    name
                    for name, old, new in zip(names, ever, mask, strict=True)
                    if new and not old
                ]
                _same(
                    event.get("new_objects"), newly, "first-placement events disagree"
                )
                _require(
                    _finite(event.get("reward")) and event["reward"] == len(newly),
                    "first-placement reward disagrees",
                )
                reward += len(newly)
                _same(
                    event.get("paper_success"),
                    all(mask),
                    "event history success disagrees",
                )
                _require(
                    type(event.get("native_success")) is bool,
                    "native current success missing",
                )
                last_native = event["native_success"]
                ever = mask
        _require(
            1 <= step <= HORIZON and (all(ever) or step == HORIZON),
            "incomplete episode refused",
        )
        _same(world.get("completed"), True, "world incomplete")
        _same(tracker.get("done"), True, "tracker incomplete")
        _same(world.get("steps"), step, "world/event length differs")
        _same(tracker.get("steps"), step, "tracker/event length differs")
        _same(tracker.get("ever_placed"), ever, "final tracker/event mask differs")
        _same(
            world.get("paper_success"), all(ever), "world/event history success differs"
        )
        _same(
            world.get("native_success"),
            last_native,
            "world/event native-current success differs",
        )
        _same(world.get("decisions"), len(decisions), "world decision count differs")
        _require(
            _finite(world.get("reward")) and world["reward"] == reward,
            "world/event reward differs",
        )
        results.append(
            {
                "seed": requested_seeds[index],
                "length": step,
                "history_success": all(ever),
                "native_current_success": last_native,
                "reward": reward,
            }
        )
    lengths = [x["length"] for x in results]
    successful = [x["length"] for x in results if x["history_success"]]
    native_lengths = [x["length"] for x in results if x["native_current_success"]]
    ticks, steps = max(lengths), sum(lengths)
    _same(rollout.get("physics_ticks"), ticks, "physics tick count differs")
    _same(rollout.get("simulation_steps"), steps, "active step count differs")
    _same(rollout.get("actualsteps"), steps, "actual step count differs")
    _same(rollout.get("episodes"), 50, "episode count differs")
    _same(
        rollout.get("successes"), len(successful), "rollout success aggregate differs"
    )
    physical = rollout.get("physical_tape")
    _require(
        isinstance(physical, list) and len(physical) == ticks,
        "physical tape tick shape differs",
    )
    for tick in physical:
        _matrix(tick, 50, 14, "submitted physical tape")
    profile = rollout["compiled_command_profile"]
    _require(
        profile.get("available") is True and profile.get("action_modified") is False,
        "compiled command profile unavailable/modified",
    )
    _require(
        isinstance(profile.get("revision"), str)
        and re.fullmatch("native_abc_affine/1:[0-9a-f]{64}", profile["revision"]),
        "command profile revision differs",
    )
    _require(
        str(rollout.get("action_projection", "")).startswith("none added;"),
        "added action projection refused",
    )
    summary = {
        "completed_episodes": 50,
        "successes": len(successful),
        "native_successes": len(native_lengths),
        "success_rate": len(successful) / 50,
        "completed_failures": 50 - len(successful),
        "completed_episode_lengths": lengths,
        "successful_episode_lengths": successful,
        "mean_successful_episode_length": _mean(successful),
        "failure_inclusive_mean_control_steps": _mean(lengths),
        "native_current_success_only_mean_control_steps": _mean(native_lengths),
        "simulation_steps": steps,
        "physics_ticks": ticks,
        "reward_sum": sum(x["reward"] for x in results),
    }
    return results, summary


def _runtime_artifacts(
    reader: _Reader, config: dict[str, Any], identity: dict[str, Any]
) -> None:
    for name in (
        "checkpoint",
        "metadata",
        "norm_stats",
        "prompt_metadata",
        "tokenizer",
    ):
        reader.pin(
            Artifact(config["base"][f"{name}_path"], identity["artifacts"][name])
        )
    native_root = (
        Path(config["source_artifacts"]["abc_minimal.vla"]["path"]).resolve().parents[1]
    )
    for name, record in config["native_inventory"].items():
        manifest = reader.record(record)
        _same(
            record["sha256"],
            identity["native_inventory"][name],
            "runtime inventory pin differs",
        )
        files = manifest["files"]
        _require(isinstance(files, list) and files, "native inventory is empty")
        seen = set()
        for item in files:
            path = (native_root / item["path"]).resolve()
            _require(
                path.is_relative_to(native_root) and str(path) not in seen,
                "invalid/duplicate inventory path",
            )
            seen.add(str(path))
            reader.pin(Artifact(str(path), item["sha256"], item["bytes"]))
    for name, record in config["source_artifacts"].items():
        reader.pin(Artifact.from_record(record))
        _same(
            record["sha256"],
            identity["sources"][name],
            "runtime public source pin differs",
        )


def _evaluation(reader: _Reader, receipt: Artifact, head: str) -> dict[str, Any]:
    worker = reader.json(receipt)
    _same(worker.get("schema_version"), 1, "worker schema differs")
    _require(worker.get("domain") == "sim", "native worker receipt required")
    _require(
        worker.get("stage") == "evaluate" and worker.get("status") == "completed",
        "complete standalone evaluation required",
    )
    _same(
        worker.get("sources"),
        SOURCE_PROFILES[head],
        "unreviewed worker/collector/source profile",
    )
    revision = (
        "sadhana.qf3-native-worker/2"
        if head == "frozen"
        else "sadhana.qf3-native-worker/4"
    )
    _same(worker.get("worker_revision"), revision, "worker revision differs")
    requested = reader.record(worker["input_config"])
    config = reader.record(worker["resolved_config"])
    _same(config.get("schema_version"), 1, "configuration schema differs")
    _require(
        _hash(config) == worker.get("resolved_config_sha256"),
        "resolved configuration semantic hash differs",
    )
    for name in (
        "schema_version",
        "stage",
        "seed",
        "base",
        "evaluation",
        "reset_options",
        "training",
        "learner",
        "replay_capacity",
        "source_artifacts",
        "native_inventory",
        "resume",
        "lora",
    ):
        _same(
            requested.get(name), config.get(name), f"requested/resolved {name} differs"
        )
    _same(config.get("stage"), "evaluate", "configuration stage differs")
    evaluation = config["evaluation"]
    _same(evaluation.get("heads"), [head], "one requested policy head required")
    _same(evaluation.get("seeds"), SEEDS, "configured fixed seed order differs")
    _same(evaluation.get("sampler_seed"), SAMPLER_SEED, "configured noise seed differs")
    _same(evaluation.get("worlds"), 50, "50-world grouping required")
    _require(
        evaluation.get("max_control_steps_per_world") is None,
        "short-horizon evaluation refused",
    )
    _same(config["training"].get("encode_microbatch"), 4, "encoder microbatch differs")
    _same(
        config.get("reset_options"),
        {
            "randomization": {
                "bottle_count": 5,
                "randomize_scales": False,
                "randomize_variants": False,
            }
        },
        "reset/scene protocol differs",
    )
    capture = worker["camera_capture_configuration"]
    _same(
        [capture.get("camera_height"), capture.get("camera_width")],
        [168, 224],
        "capture geometry differs",
    )
    identity = worker["runtime_identity"]
    _require(identity.get("evidence_mode") == "native", "fixture runtime refused")
    _same(
        [
            identity.get("horizon"),
            identity.get("execute_prefix"),
            identity.get("diffusion_steps"),
        ],
        [30, 15, 5],
        "ABC-VLA full30/first15/fiveEuler protocol differs",
    )
    _same(identity.get("task_id"), TASK, "task differs")
    _same(
        identity.get("binding_source"),
        COMMON_SOURCES["nrh.qf3_abc_vla"],
        "binding source differs",
    )
    _same(identity.get("core_source"), COMMON_SOURCES["nrh.qf3"], "core source differs")
    _require(_sha(worker.get("base_id")), "base identity missing")
    _same(
        worker["base_id"],
        _hash(identity),
        "base identity does not hash runtime identity",
    )
    for key in ("task_id", "seed", "upstream_revision"):
        _same(
            config["base"].get(key),
            identity.get(key),
            f"runtime/config base {key} differs",
        )
    _same(
        config.get("lora"), identity.get("lora_config"), "runtime/config LoRA differs"
    )
    _runtime_artifacts(reader, config, identity)
    _require(
        isinstance(worker.get("rollouts"), list) and len(worker["rollouts"]) == 1,
        "exactly one new evaluation tape required",
    )
    tape = reader.record(worker["rollouts"][0])
    _same(tape.get("base_id"), worker["base_id"], "tape base identity differs")
    event_key = (
        "evaluation_events" if head == "frozen" else "invocation_evaluation_events"
    )
    events = worker.get(event_key)
    _require(
        isinstance(events, list) and len(events) == 1,
        "exactly one new invocation event required",
    )
    event = events[0]
    _require(
        event.get("head") == head
        and event.get("reason") == ("baseline" if head == "frozen" else "final"),
        "stale/periodic evaluation event refused",
    )
    all_events = worker.get("evaluation_events")
    _require(
        isinstance(all_events, list) and all_events.count(event) == 1,
        "invocation event is not uniquely retained",
    )
    _same(
        tape.get("stage"),
        f"evaluate-{head}-{all_events.index(event)}",
        "tape does not bind invocation event",
    )
    _require(_integer(event.get("outer_iterations")), "event outer counter invalid")
    rows, summary = _outcomes(tape, head)
    _same(
        tape["rollout"].get("reset_options"),
        config["reset_options"],
        "tape reset protocol differs",
    )
    reported = event["summary"]
    _same(worker["evaluation"].get(head), reported, "event/receipt summary differs")
    for key, value in summary.items():
        if key not in ("native_current_success_only_mean_control_steps", "reward_sum"):
            _same(
                reported.get(key),
                value,
                f"aggregate {key} differs from retained events",
            )
    _same(
        [reported.get("attempted_episodes"), reported.get("requested_episodes")],
        [50, 50],
        "not all50 attempted/requested",
    )
    _same(reported.get("partial"), False, "partial summary refused")
    _same(
        reported.get("partial_episode_lengths"), [], "partial episode evidence refused"
    )
    _same(reported.get("sampler_seed"), SAMPLER_SEED, "event noise seed differs")
    _same(
        reported.get("configured_seed_sha256"),
        _hash(SEEDS),
        "event seed digest differs",
    )
    _same(reported.get("base_id"), worker["base_id"], "event base identity differs")
    before = reported.get("training_state_before")
    _require(
        _sha(before) and before == reported.get("training_state_after"),
        "training state changed across evaluation",
    )
    actor_updates = reported.get("actor_updates")
    _require(
        _integer(actor_updates)
        and (actor_updates == 0 if head == "frozen" else actor_updates > 0),
        "missing/invalid actual actor update count",
    )
    _same(
        worker["metrics"].get("actor_updates"), actor_updates, "actor counter differs"
    )
    if head == "learned":
        _same(
            worker.get("restore_verified"),
            True,
            "learned training restore was not verified",
        )
    phases = worker["phase_counts"]
    _same(
        phases.get("evaluation_steps"),
        summary["simulation_steps"],
        "new invocation steps do not match tape",
    )
    for name in ("warmup_steps", "train_steps", "critic_updates", "actor_updates"):
        _same(phases.get(name), 0, "standalone evaluation modified training")
    reader.pin(Artifact.from_record(worker["checkpoint"]))
    return {
        "worker": worker,
        "config": config,
        "rows": rows,
        "summary": summary,
        "command_profile": tape["rollout"]["compiled_command_profile"],
        "event": event,
        "identity": identity,
        "receipt": reader.pin(receipt),
    }


def _time(value: Any) -> datetime:
    _require(isinstance(value, str), "selection/invocation time missing")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ComparisonError("invalid selection/invocation time") from exc
    _require(
        result.tzinfo is not None, "selection/invocation time must include timezone"
    )
    return result


def _capacity_evidence(
    reader: _Reader,
    capacity: dict[str, Any],
    config: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    expected = {
        "warmup_worlds": 80,
        "warmup_episodes_per_world": 4,
        "train_worlds": 16,
        "max_outer_iterations": 1,
        "target_control_steps": 16000,
        "critic_updates_per_iteration": 1600,
        "critic_batch_size": 64,
        "actor_updates_per_iteration": 200,
        "actor_batch_size": 256,
        "encode_microbatch": 4,
        "max_control_steps_per_world": None,
        "actor_target_every_iterations": 1,
        "critic_target_every_updates": 1,
    }
    for key, value in expected.items():
        _same(config["training"].get(key), value, f"capacity recipe {key} differs")
    _same(config.get("learner"), CAPACITY_LEARNER, "capacity learner recipe differs")
    _same(config.get("seed"), 903, "capacity learner seed differs")
    _same(config.get("replay_capacity"), 50000, "capacity replay size differs")
    _same(config["evaluation"].get("worlds"), 50, "capacity validation worlds differ")
    _require(
        config["evaluation"].get("max_control_steps_per_world") is None,
        "short capacity validation refused",
    )
    _same(
        config["evaluation"].get("layout_sampling"),
        {"mode": "fresh_per_event", "seed_start": 2**40, "seed_stride": 200001},
        "capacity fresh layout protocol differs",
    )
    for key, value in {
        "heads": ["learned"],
        "sampler_seed": 91001,
        "seeds": SEEDS,
        "cadence": {
            "unit": "outer_iterations",
            "every": 1,
            "phase": "after_outer_updates",
        },
    }.items():
        _same(
            config["evaluation"].get(key), value, f"capacity evaluation {key} differs"
        )
    totals, counts, seen = (
        {"warmup": 0, "train": 0, "evaluate": 0},
        {"warmup": 0, "train": 0, "evaluate": 0},
        set(),
    )
    fresh_tape = None
    for record in capacity["rollouts"]:
        tape = reader.record(record)
        _same(tape.get("schema_version"), 1, "capacity tape schema differs")
        _require(
            tape.get("domain") == "sim" and tape.get("protocol_id") == PROTOCOL,
            "capacity tape domain/protocol differs",
        )
        _same(tape.get("base_id"), capacity["base_id"], "capacity tape base differs")
        stage = tape["stage"]
        _require(record["path"] not in seen, "duplicate capacity tape path")
        seen.add(record["path"])
        category = "evaluate" if stage.startswith("evaluate-learned-") else stage
        _require(category in totals, "unrelated capacity rollout")
        count = {"warmup": 80, "train": 16, "evaluate": 50}[category]
        rollout = tape["rollout"]
        _same(rollout.get("partial"), False, "partial capacity rollout")
        rows = rollout["worlds"]
        _require(
            isinstance(rows, list) and len(rows) == count,
            "capacity world count differs",
        )
        lengths = []
        for index, row in enumerate(rows):
            _same(row.get("world"), index, "capacity world order differs")
            _same(row.get("completed"), True, "incomplete capacity world")
            _require(
                _integer(row.get("steps"), 1) and row["steps"] <= HORIZON,
                "capacity world length invalid",
            )
            lengths.append(row["steps"])
        _same(
            rollout.get("episodes"), count, "capacity completed episode count differs"
        )
        _same(
            rollout.get("simulation_steps"),
            sum(lengths),
            "capacity tape step sum differs",
        )
        _same(
            rollout.get("physics_ticks"), max(lengths), "capacity physics ticks differ"
        )
        totals[category] += sum(lengths)
        counts[category] += 1
        if category == "evaluate":
            fresh_tape = tape
    # Warmup stage labels share 'warmup'; distinguish batches by pinned paths.
    # They are validated below by record count, not a fabricated concatenated tape.
    _same(
        len(capacity["rollouts"]),
        6,
        "capacity requires four warmup, one train and one validation tape",
    )
    _same(
        counts,
        {"warmup": 4, "train": 1, "evaluate": 1},
        "capacity tape stage counts differ",
    )
    for name, key in (
        ("warmup", "warmup_steps"),
        ("train", "training_steps"),
        ("evaluate", "evaluation_steps"),
    ):
        _same(
            capacity["metrics"].get(key),
            totals[name],
            f"capacity {name} step count differs",
        )
    periodic = capacity["evaluation_events"][-1]
    validation = periodic["validation"]
    _same(
        selection.get("validation_manifest"),
        validation["manifest"],
        "selected fresh manifest pin differs",
    )
    manifest = reader.record(validation["manifest"])
    _same(manifest.get("schema_version"), 1, "fresh manifest schema differs")
    _require(
        manifest.get("domain") == "sim"
        and manifest.get("revision") == "sadhana.qf3-fresh-validation/1",
        "fresh manifest domain/revision differs",
    )
    _same(manifest.get("base_id"), capacity["base_id"], "fresh manifest base differs")
    _same(
        manifest.get("source_sha256"),
        _hash(capacity["sources"]),
        "fresh manifest source differs",
    )
    _same(
        manifest.get("policy_state_sha256"),
        selection["policy_state_sha256"],
        "selected policy fingerprint differs from capacity evidence",
    )
    _same(manifest.get("actor_updates"), 200, "fresh policy update count differs")
    _same(manifest.get("outer_iterations"), 1, "fresh outer count differs")
    _same(manifest.get("cursor"), 0, "fresh cursor differs")
    _same(
        manifest.get("requested_seeds"),
        [2**40 + j * 200001 for j in range(50)],
        "fresh50 requested seed order differs",
    )
    _same(manifest.get("noise_seed"), 91002, "fresh50 noise differs")
    _same(manifest.get("head"), "learned", "fresh policy head differs")
    _same(manifest.get("protocol_id"), PROTOCOL, "fresh protocol differs")
    _same(
        manifest.get("reset_options"),
        config["reset_options"],
        "fresh reset options differ",
    )
    _same(
        manifest.get("training_control_steps"),
        totals["train"],
        "fresh training counter differs",
    )
    resets = validation.get("reset_evidence")
    _require(
        isinstance(resets, list) and len(resets) == 1, "fresh50 reset evidence missing"
    )
    reset = reader.record(resets[0])
    _same(reset.get("schema_version"), 1, "fresh reset schema differs")
    _require(reset.get("domain") == "sim", "fresh reset domain differs")
    _same(
        reset.get("revision"),
        "sadhana.qf3-fresh-validation/1",
        "fresh reset revision differs",
    )
    _same(
        reset.get("manifest_sha256"),
        validation["manifest"]["sha256"],
        "fresh reset manifest differs",
    )
    _same(reset.get("start"), 0, "fresh reset batch start differs")
    _same(
        [x.get("requested_seed") for x in reset["worlds"]],
        manifest["requested_seeds"],
        "fresh actual reset rows differ",
    )
    actual = [x.get("actual_seed") for x in reset["worlds"]]
    _require(
        all(
            type(value) is int and value in (requested, requested + 100000)
            for requested, value in zip(
                manifest["requested_seeds"], actual, strict=True
            )
        ),
        "fresh actual seed slots differ",
    )
    _, observed = _outcomes(
        fresh_tape,
        "learned",
        requested_seeds=manifest["requested_seeds"],
        actual_seeds=actual,
        initial_zero=False,
    )
    _same(
        fresh_tape["rollout"].get("reset_options"),
        config["reset_options"],
        "fresh tape reset protocol differs",
    )
    _same(periodic.get("outer_iterations"), 1, "fresh event outer counter differs")
    _same(
        periodic["summary"].get("actor_updates"),
        200,
        "fresh summary actor count differs",
    )
    _same(periodic["summary"].get("attempted_episodes"), 50, "fresh attempts differ")
    _same(periodic["summary"].get("requested_episodes"), 50, "fresh requests differ")
    _same(
        periodic["summary"].get("partial_episode_lengths"),
        [],
        "fresh partial lengths differ",
    )
    _same(periodic["summary"].get("sampler_seed"), 91002, "fresh summary noise differs")
    _same(
        periodic["summary"].get("configured_seed_sha256"),
        _hash(manifest["requested_seeds"]),
        "fresh summary seed digest differs",
    )
    _same(
        periodic["summary"].get("base_id"),
        capacity["base_id"],
        "fresh summary base differs",
    )
    _same(
        periodic["summary"].get("simulation_steps"),
        totals["evaluate"],
        "fresh summary steps differ",
    )
    for key, value in observed.items():
        if key not in ("native_current_success_only_mean_control_steps", "reward_sum"):
            _same(periodic["summary"].get(key), value, f"fresh aggregate {key} differs")
    before = periodic["summary"].get("training_state_before")
    _require(
        _sha(before) and before == periodic["summary"].get("training_state_after"),
        "capacity validation changed training state",
    )


def _selection(
    reader: _Reader, item: Artifact, parent_item: Artifact, learned: dict[str, Any]
) -> dict[str, Any]:
    selection, parent = reader.json(item), reader.json(parent_item)
    _same(selection.get("schema_version"), 1, "selection schema differs")
    _require(
        selection.get("kind") == "qf3_development_checkpoint_selection",
        "selection manifest kind differs",
    )
    _require(
        selection.get("status") == "accepted"
        and selection.get("independent_review") is True,
        "independent pre-outcome selection required",
    )
    _same(
        selection.get("selection_rule"),
        SELECTION_RULE,
        "selection boundary/quantity rule differs",
    )
    worker = learned["worker"]
    config = learned["config"]
    selected = selection["checkpoint"]
    reader.pin(Artifact.from_record(selected))
    _same(
        selected.get("kind"),
        "qf3_completed_boundary_checkpoint",
        "selected checkpoint kind differs",
    )
    _artifact_same(
        config.get("resume"),
        selected,
        "learned resume does not bind selected checkpoint",
    )
    _same(
        selection.get("source_sha256"),
        worker["sources"],
        "selected source identity differs",
    )
    _same(selection.get("base_id"), worker["base_id"], "selected base identity differs")
    _require(
        _sha(selection.get("policy_state_sha256")),
        "selected bridge policy state identity missing",
    )
    _same(selection.get("actor_updates"), 200, "selected actor update count differs")
    _same(selection.get("critic_updates"), 1600, "selected critic update count differs")
    _same(
        worker["metrics"].get("actor_updates"),
        200,
        "evaluated actor counter differs from selection",
    )
    _same(
        worker["metrics"].get("critic_updates"),
        1600,
        "evaluated critic counter differs from selection",
    )
    capacity = reader.record(selection["capacity_receipt"])
    _same(capacity.get("schema_version"), 1, "capacity receipt schema differs")
    _require(
        capacity.get("domain") == "sim"
        and capacity.get("stage") == "train"
        and capacity.get("status") in ("completed", "budget_stopped"),
        "capacity run is not accepted native training",
    )
    _same(
        capacity.get("sources"), worker["sources"], "capacity source identity differs"
    )
    _same(capacity.get("base_id"), worker["base_id"], "capacity base identity differs")
    _artifact_same(
        capacity.get("latest_complete"),
        selected,
        "selected checkpoint is not capacity complete boundary",
    )
    completed = capacity.get("completed_checkpoints")
    _require(
        isinstance(completed, list) and selected in completed,
        "selected checkpoint missing from completed boundaries",
    )
    _same(
        capacity.get("runtime_identity"),
        learned["identity"],
        "capacity runtime identity differs",
    )
    _same(
        capacity.get("camera_capture_configuration"),
        worker["camera_capture_configuration"],
        "capacity capture configuration differs",
    )
    _same(
        capacity["metrics"].get("outer_iterations"),
        1,
        "capacity outer boundary differs",
    )
    _same(capacity["metrics"].get("actor_updates"), 200, "capacity actor count differs")
    _same(
        capacity["metrics"].get("critic_updates"), 1600, "capacity critic count differs"
    )
    wrapper = capacity["wrapper"]
    _same(wrapper.get("warmup_batches"), 4, "capacity warmup batch count differs")
    _same(
        wrapper.get("completed_episodes"),
        336,
        "capacity complete warmup/rollout episodes differ",
    )
    _require(
        wrapper.get("boundary") == "completed_outer"
        and wrapper.get("partial") is False,
        "capacity boundary incomplete",
    )
    _require(
        wrapper["evaluation_state"].get("pending_validation") is None,
        "capacity validation pending",
    )
    periodic = capacity["evaluation_events"][-1]
    _require(
        periodic.get("head") == "learned" and periodic.get("reason") == "periodic",
        "capacity fresh validation missing",
    )
    _require(
        periodic["summary"].get("completed_episodes") == 50
        and periodic["summary"].get("partial") is False
        and "validation" in periodic,
        "capacity fresh50 validation incomplete",
    )
    capacity_config = reader.record(capacity["resolved_config"])
    _same(
        capacity.get("resolved_config_sha256"),
        _hash(capacity_config),
        "capacity resolved semantic hash differs",
    )
    capacity_requested = reader.record(capacity["input_config"])
    for key in (
        "schema_version",
        "stage",
        "seed",
        "base",
        "learner",
        "lora",
        "training",
        "evaluation",
        "replay_capacity",
        "reset_options",
        "native_inventory",
        "source_artifacts",
        "resume",
    ):
        _same(
            capacity_requested.get(key),
            capacity_config.get(key),
            f"capacity requested/resolved {key} differs",
        )
    _same(capacity_config.get("stage"), "train", "capacity configuration stage differs")
    _same(
        selection.get("capacity_resolved_config"),
        capacity["resolved_config"],
        "selection config pin differs",
    )
    _same(capacity_config.get("resume"), None, "capacity did not start from candidate")
    for key in (
        "seed",
        "replay_capacity",
        "base",
        "learner",
        "lora",
        "training",
        "reset_options",
        "native_inventory",
        "source_artifacts",
    ):
        _same(
            capacity_config.get(key),
            config.get(key),
            f"selected/evaluated {key} differs",
        )
    _capacity_evidence(reader, capacity, capacity_config, selection)
    _require(
        parent.get("status") == "completed"
        and parent.get("training_stage") == "evaluate"
        and parent.get("worker_status") == "completed",
        "learned invocation parent incomplete",
    )
    _same(
        parent.get("summary_sha256"),
        learned["receipt"]["sha256"],
        "parent worker receipt pin differs",
    )
    _same(
        parent.get("training_config_sha256"),
        worker["input_config"]["sha256"],
        "parent requested input pin differs",
    )
    for key, value in {
        "evaluation_heads": ["learned"],
        "evaluation_seeds": SEEDS,
        "evaluation_sampler_seed": SAMPLER_SEED,
        "evaluation_horizon_steps": HORIZON,
    }.items():
        _same(parent.get(key), value, f"parent {key} differs")
    _require(
        _time(selection.get("selected_at_utc")) < _time(parent.get("created_at")),
        "selection was not recorded before invocation",
    )
    return {
        "manifest": reader.pin(item),
        "invocation_parent": reader.pin(parent_item),
        "checkpoint": selected,
        "policy_state_sha256": selection["policy_state_sha256"],
        "policy_state_scope": "Full bridge state at the capacity fresh-validation boundary, including online/target heads and bridge RNG; not an isolated tensor hash.",
        "chronology_scope": "Independent selection metadata and controller start time; byte hashes alone do not prove chronology.",
    }


def _paired(
    baseline: list[dict[str, Any]], learned: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    buckets: dict[str, list[int]] = {
        name: [] for name in ("win", "loss", "both", "neither")
    }
    changes = []
    for left, right in zip(baseline, learned, strict=True):
        _same(left["seed"], right["seed"], "pairing seed differs")
        label = (
            "both"
            if left[key] and right[key]
            else "win"
            if right[key]
            else "loss"
            if left[key]
            else "neither"
        )
        buckets[label].append(left["seed"])
        if label == "both":
            changes.append(
                {
                    "seed": left["seed"],
                    "baseline_control_steps": left["length"],
                    "learned_control_steps": right["length"],
                    "learned_minus_baseline": right["length"] - left["length"],
                }
            )
    return {
        "counts": {name: len(seeds) for name, seeds in buckets.items()},
        "seeds": buckets,
        "common_success_length_changes": changes,
        "common_success_mean_length_change": _mean(
            [x["learned_minus_baseline"] for x in changes]
        ),
    }


def _bounded(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ComparisonError:
            raise
        except (
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise ComparisonError(f"malformed comparison evidence: {exc}") from exc

    return wrapped


@_bounded
def validate_baseline(receipt: Artifact) -> dict[str, Any]:
    """Validate a pinned, complete historical frozen ABC-VLA development receipt."""
    reader = _Reader()
    baseline = _evaluation(reader, receipt, "frozen")
    return {
        "schema_version": 1,
        "status": "validated_frozen_development_evidence",
        "comparison_executed": False,
        "learned_result": None,
        "baseline": {
            "receipt": baseline["receipt"],
            "summary": baseline["summary"],
            "worlds": baseline["rows"],
        },
        "verified_artifacts": list(reader.verified.values()),
        "claim_scope": "Posthoc artifact validation; no new native execution or paper reproduction.",
    }


@_bounded
def compare(
    baseline_receipt: Artifact,
    learned_receipt: Artifact,
    selection: Artifact,
    learned_parent: Artifact,
) -> dict[str, Any]:
    """Compare fixed development outcomes after independent checkpoint selection."""
    reader = _Reader()
    baseline = _evaluation(reader, baseline_receipt, "frozen")
    learned = _evaluation(reader, learned_receipt, "learned")
    _same(
        baseline["identity"],
        learned["identity"],
        "runtime/base/normalization/prompt identity differs",
    )
    _same(
        baseline["worker"]["base_id"],
        learned["worker"]["base_id"],
        "base identity differs",
    )
    _same(
        baseline["worker"]["camera_capture_configuration"],
        learned["worker"]["camera_capture_configuration"],
        "capture configuration differs",
    )
    _same(
        baseline["command_profile"]["revision"],
        learned["command_profile"]["revision"],
        "compiled actuator profile differs",
    )
    selected = _selection(reader, selection, learned_parent, learned)
    return {
        "schema_version": 1,
        "status": "descriptive_fixed_development_comparison",
        "selection": selected,
        "baseline": {
            "receipt": baseline["receipt"],
            "summary": baseline["summary"],
            "worlds": baseline["rows"],
        },
        "learned": {
            "receipt": learned["receipt"],
            "summary": learned["summary"],
            "worlds": learned["rows"],
        },
        "history_success_pairing": _paired(
            baseline["rows"], learned["rows"], "history_success"
        ),
        "native_current_success_pairing": _paired(
            baseline["rows"], learned["rows"], "native_current_success"
        ),
        "source_version_differences": {
            key: {
                "baseline": baseline["worker"]["sources"][key],
                "learned": learned["worker"]["sources"][key],
            }
            for key in ("nrh.qf3_training", "nrh.qf3_rollouts")
        },
        "command_profiles": {
            "baseline": baseline["command_profile"],
            "learned": learned["command_profile"],
        },
        "verified_artifacts": list(reader.verified.values()),
        "limits": [
            "Development cohort only; no final-test, author-performance or paper reproduction claim.",
            "Success-only means describe each policy's successful subset; differing subsets do not prove speed gain.",
            "Common-success length changes are separate descriptive paired values; no p-values or causal RL gain claim.",
            "Native current success is judged at each history-defined episode end; lengths are not native first-success times.",
            "Protocol and actual seed pairing do not prove equal native object geometry or trajectories.",
            "Training-state preservation uses matching worker-reported hashes; no checkpoint tensors are loaded.",
            "Selection chronology relies on independent metadata and pinned controller provenance, not hash timestamps.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate-baseline", "compare"))
    for name in ("baseline", "learned", "selection", "learned-parent"):
        parser.add_argument(f"--{name}")
        parser.add_argument(f"--{name}-sha256")
    parser.add_argument(
        "--output", help="Create a new JSON file; existing files are refused."
    )
    args = parser.parse_args(argv)
    try:
        _require(
            args.baseline is not None and args.baseline_sha256 is not None,
            "pinned baseline receipt required",
        )
        baseline = Artifact(args.baseline, args.baseline_sha256)
        if args.mode == "validate-baseline":
            _require(
                not any(
                    getattr(args, key)
                    for key in ("learned", "selection", "learned_parent")
                ),
                "baseline-only mode refuses learned inputs",
            )
            report = validate_baseline(baseline)
        else:
            for name in ("learned", "selection", "learned_parent"):
                _require(
                    getattr(args, name) is not None
                    and getattr(args, f"{name}_sha256") is not None,
                    f"pinned {name} required",
                )
            report = compare(
                baseline,
                Artifact(args.learned, args.learned_sha256),
                Artifact(args.selection, args.selection_sha256),
                Artifact(args.learned_parent, args.learned_parent_sha256),
            )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with Path(args.output).open("x") as stream:
                stream.write(text)
        else:
            print(text, end="")
        return 0
    except (ComparisonError, OSError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
