"""Read full three-seed QF3 evidence using the Python standard library only.

Tensor, replay-label and native geometry truth require separately pinned audits.
This reader constructs no policy and creates no checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any

from . import vla_comparison as legacy
from .vla_comparison import Artifact, ComparisonError

PROTOCOL = legacy.PROTOCOL
TASK = legacy.TASK
REVISION = "abc.qf3-full-campaign-comparison/1"
WORKER = "sadhana.qf3-native-worker/5"
STORAGE = {"mode": "immutable_replay_blocks", "schema_version": 1}
PROFILE = {
    **legacy.COMMON_SOURCES,
    "nrh.qf3_rollouts": "39be24afd59e20c8afffa02370b18b3b9ead3f5fd1b4b4761834aba9966bdef4",
    "nrh.qf3_training": "65a62e73c7ad2079c8d6c40262220cb8149323f7f7e7bed8425d2d162645a2c8",
    "nrh.qf3_checkpoint_storage": "bbed2a270430f6c82ca41e8170af9879ede4001fad340a82fe0e5b249e080818",
}
RELEASE_SHA = "55ce6669f0a3bb32fa6f0338f3f169c204dfa4f452c0523949a7d01ee2c004c4"
RELEASE_BYTES = 162986
INSTALL_REVIEW_SHA = "b5e03c6a7e06fa7a100b5a1651c81d95677481ee1b9401247386dd631f5d5f64"
NATIVE_INSTALL_REVIEW_SHA = (
    "95e4f01f036093a1008593e8c2e818765db5357bb87578fafa9b8e8ed5d7294c"
)
ACTUAL_INSTALL_PROOF_SHA = (
    "581fa675ba7953ad5e7b08ac800bafac6ac2089f8fbab9bdf431c0d1c6bf19ee"
)
CANDIDATE_SHA = "4838acf69279497b917c998039089b5b6a310dfe65fc0b5c438c3339ee25c753"
LEGACY_SHA = "54803360e8dfb0ba26834a47d295f46680c3006a31af39a3f0131258ead90206"
NATIVE_SOURCES = {
    "abc_minimal.batched_eval": "4163446dad84ed98c603d3869a212b6b8f7338021686c454bf02fb0ac723dfd5",
    "abc_minimal.checkpointing": "754e87f9083fab6a06f1abca79a53cd544cd9b53fbcabf3b367a255d2fa7847d",
    "abc_minimal.config": "bfb0e3b847ff920a25f9095c8a6050a3b25c23946e8c5bdab43c19b0f0f0996a",
    "abc_minimal.dit": "5d02eb00e3f6974120de606519b18d9dbec293c6223554e0f5912fa0f95527cd",
    "abc_minimal.policy": "529daee67fbd29c4eabeba48c707079fa978db7883a135db0e45039e672d380a",
    "abc_minimal.preprocess": "bc15a01e86461a69126156f1367833d01cd1443f7a1f0796d3162aac2924b190",
    "abc_minimal.vla": "c7ae8f78e03e460099947450163157f89673358e24636aea2992c041b2428010",
    "abc_sim.batched_env": "46cff7da736426b3bcce99f92239deb8ab7419e2eb600f0860665e405f0c535b",
    "abc_sim.config": "d07d5d190605e8a7417ce1a2a465b000568c9afa1606e64980b3b8a62109d833",
    "abc_sim.env": "159501e3509a6b2684ea3edd6ce466ce875e1036a69882bb9c33d0da5a520a40",
}
NATIVE_INVENTORY = {
    "assets": "58c76a493cde59f69c6d4bc7ed1d906d776e246573a6521ad23b1b99ea005776",
    "sources": "b16cf193c7678ffb1e414cd3994a86e53c38927a425e09854966b981b7219e6b",
}
ARTIFACT_HASHES = {
    "checkpoint": CANDIDATE_SHA,
    "metadata": "036e79330027162e5002b7d6ae3798f0d4a450ba987115704343bfce026ce8d7",
    "norm_stats": "a17876c300d337c108f6712435ca4138244c82504baf8b9753cb29c4d9b4d5a3",
    "prompt_metadata": "48f1e86416866a3b004f85eeb8ff7bfaab76f981d4efc1a45b1070f5b9a3bed7",
    "tokenizer": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
}
VERSIONS = {
    "abc-minimal": "0.1.0",
    "torch": "2.11.0+cu128",
    "numpy": "2.2.6",
    "mujoco": "3.8.1",
    "mujoco-warp": "3.8.0.3",
    "warp-lang": "1.13.0",
    "sentencepiece": "0.2.2",
}
FINAL_SEEDS = [2**50 + j * 200001 for j in range(50)]
TARGET = 400000
TRIPLES = [(901, 902, 903), (1901, 1902, 1903), (2901, 2902, 2903)]
RESET = {
    "randomization": {
        "bottle_count": 5,
        "randomize_scales": False,
        "randomize_variants": False,
    }
}
TRAINING = {
    "warmup_worlds": 80,
    "warmup_episodes_per_world": 4,
    "train_worlds": 16,
    "max_outer_iterations": 25000,
    "target_control_steps": TARGET,
    "critic_updates_per_iteration": 1600,
    "critic_batch_size": 64,
    "actor_updates_per_iteration": 200,
    "actor_batch_size": 256,
    "encode_microbatch": 4,
    "max_control_steps_per_world": None,
    "actor_target_every_iterations": 1,
    "critic_target_every_updates": 1,
}
CADENCE = {"unit": "outer_iterations", "every": 1, "phase": "after_outer_updates"}
FINAL_CADENCE = {"unit": "final", "every": None, "phase": "after_outer_updates"}
CONTINUATION_KEYS = (
    "base",
    "source_artifacts",
    "native_inventory",
    "lora",
    "learner",
    "seed",
    "replay_capacity",
    "reset_options",
    "fixed_layout_seed",
    "training",
    "evaluation",
    "checkpoint_storage",
)
AUDIT_CHECKS = (
    "learner_targets_optimizers_rng",
    "canonical_ring_hydration",
    "block_row_selector_semantics",
    "causal_replay_native_dispatch_reward_lineage",
    "episode_evidence_id_lineage",
    "evicted_rows_prior_boundary_coverage",
)
PRESERVATION_CHECKS = (
    "selected_resume_checkpoint",
    "saved_learner_replay_bridge_rng_preservation",
)

_same, _need, _hash = legacy._same, legacy._require, legacy._hash
_integer, _finite, _sha = legacy._integer, legacy._finite, legacy._sha


class _Reader(legacy._Reader):
    def _prior(self, item: Artifact) -> None:
        previous = self.verified.get(str(Path(item.path).resolve()))
        if previous is not None:
            _same(
                previous["sha256"],
                item.sha256,
                "conflicting artifact SHA pins for the same path",
            )
            if item.bytes is not None:
                _same(
                    previous["bytes"],
                    item.bytes,
                    "conflicting artifact byte pins for the same path",
                )

    def pin(self, item: Artifact) -> dict:
        _need(
            isinstance(item.path, str) and Path(item.path).is_absolute(),
            "absolute artifact path required",
        )
        self._prior(item)
        return super().pin(item)

    def json(self, item: Artifact) -> dict:
        _need(
            isinstance(item.path, str) and Path(item.path).is_absolute(),
            "absolute JSON artifact path required",
        )
        self._prior(item)
        path = Path(item.path).resolve()
        try:
            before = self.signature(path)
            value = super().json(item)
            _same(self.signature(path), before, "JSON artifact changed during read")
            return value
        except OSError as exc:
            raise ComparisonError(f"JSON artifact unavailable: {path}: {exc}") from exc


def _fields(value: Any, required: set[str], name: str) -> dict:
    _need(isinstance(value, dict) and required <= set(value), f"missing {name} fields")
    return value


def _pin(reader: legacy._Reader, value: dict) -> dict:
    return reader.pin(Artifact.from_record(value))


def _match_pin(left: dict, right: dict, name: str) -> None:
    legacy._artifact_same(left, right, name)


def _continuation(config: dict) -> str:
    return _hash(
        {
            "revision": WORKER,
            "sources": PROFILE,
            "config": {k: config[k] for k in CONTINUATION_KEYS},
        }
    )


def _noise_rng(value: dict) -> None:
    _need(
        isinstance(value, dict)
        and set(value) == {"bit_generator", "state", "has_uint32", "uinteger"},
        "noise RNG fields differ",
    )
    _same(value.get("bit_generator"), "PCG64", "noise generator differs")
    state = value.get("state")
    _need(
        isinstance(state, dict) and set(state) == {"state", "inc"},
        "noise PCG64 state fields differ",
    )
    _need(
        all(_integer(v) and v < 2**128 for v in state.values())
        and state["inc"] % 2 == 1,
        "noise PCG64 state invalid",
    )
    _need(
        type(value["has_uint32"]) is int
        and value["has_uint32"] in (0, 1)
        and _integer(value["uinteger"])
        and value["uinteger"] < 2**32,
        "noise RNG uint32 cache invalid",
    )


def _config(
    reader: legacy._Reader, worker: dict, index: int, *, head: str | None = None
) -> dict:
    requested = reader.record(worker["input_config"])
    config = reader.record(worker["resolved_config"])
    _same(
        worker["resolved_config_sha256"],
        _hash(config),
        "resolved semantic hash differs",
    )
    for key in (
        "schema_version",
        "stage",
        "base",
        "lora",
        "learner",
        "seed",
        "replay_capacity",
        "training",
        "evaluation",
        "reset_options",
        "source_artifacts",
        "native_inventory",
        "resume",
    ):
        _same(requested.get(key), config.get(key), f"requested/resolved {key} differs")
    _same(
        requested.get("checkpoint_storage", {"mode": "inline", "schema_version": 1}),
        config.get("checkpoint_storage"),
        "requested storage differs",
    )
    _same(config.get("schema_version"), 1, "config schema differs")
    _same(
        config.get("checkpoint_storage"),
        STORAGE,
        "full campaign requires block storage",
    )
    _same(config.get("reset_options"), RESET, "reset protocol differs")
    _same(config.get("fixed_layout_seed"), None, "fixed training layout refused")
    _same(config["base"].get("task_id"), TASK, "configured task differs")
    _same(
        config["base"].get("upstream_revision"),
        "d0832d12651d1b260a652861a14648dc5f3660c7",
        "upstream revision differs",
    )
    _same(
        {k: v.get("sha256") for k, v in config["source_artifacts"].items()},
        NATIVE_SOURCES,
        "native source profile differs",
    )
    _same(
        {k: v.get("sha256") for k, v in config["native_inventory"].items()},
        NATIVE_INVENTORY,
        "native inventory profile differs",
    )
    base_seed, lora_seed, worker_seed = TRIPLES[index]
    _same(config["base"].get("seed"), base_seed, "base seed differs")
    _same(config.get("seed"), worker_seed, "training seed differs")
    _same(
        config.get("lora"),
        {"rank": 4, "scale": 1.0, "init_std": 0.02, "seed": lora_seed},
        "LoRA config differs",
    )
    _same(config.get("replay_capacity"), 50000, "replay capacity differs")
    _same(config.get("learner"), legacy.CAPACITY_LEARNER, "learner recipe differs")
    _same(config.get("training"), TRAINING, "full training recipe differs")
    expected = {
        "heads": ["frozen"] if head == "frozen" else ["learned"],
        "seeds": FINAL_SEEDS,
        "worlds": 50,
        "max_control_steps_per_world": None,
        "sampler_seed": 2**45 + 17 + index * 1000000,
        "cadence": FINAL_CADENCE if head == "frozen" else CADENCE,
    }
    if head != "frozen":
        expected["layout_sampling"] = {
            "mode": "fresh_per_event",
            "seed_start": 2**40 + index * 2**39,
            "seed_stride": 200001,
        }
    _same(
        config.get("evaluation"), expected, "evaluation/cohort/noise protocol differs"
    )
    return config


def _runtime(reader: legacy._Reader, worker: dict, config: dict) -> None:
    _same(worker.get("schema_version"), 1, "worker receipt schema differs")
    _same(worker.get("domain"), "sim", "native simulation receipt required")
    _same(worker.get("worker_revision"), WORKER, "worker revision differs")
    _same(worker.get("stage"), config["stage"], "worker stage differs")
    _same(worker.get("sources"), PROFILE, "unreviewed source profile")
    identity = worker["runtime_identity"]
    _same(worker.get("base_id"), _hash(identity), "base identity differs")
    _same(identity.get("evidence_mode"), "native", "fixture runtime refused")
    _same(identity.get("task_id"), TASK, "task differs")
    for key in ("seed", "upstream_revision"):
        _same(identity.get(key), config["base"].get(key), f"runtime base {key} differs")
    _same(identity.get("lora_config"), config["lora"], "runtime LoRA differs")
    _same(identity.get("binding_source"), PROFILE["nrh.qf3_abc_vla"], "binding differs")
    _same(identity.get("core_source"), PROFILE["nrh.qf3"], "core differs")
    _same(
        [identity.get(k) for k in ("horizon", "execute_prefix", "diffusion_steps")],
        [30, 15, 5],
        "action sampler differs",
    )
    _same(
        identity["artifacts"].get("checkpoint"),
        CANDIDATE_SHA,
        "candidate initializer differs",
    )
    _same(identity.get("artifacts"), ARTIFACT_HASHES, "runtime artifacts differ")
    _same(identity.get("versions"), VERSIONS, "runtime dependency versions differ")
    _same(
        identity.get("prompt"),
        "sim throw plastic bottles in bin",
        "parent prompt differs",
    )
    _same(
        identity.get("checkpoint_role"),
        "declared_shared_candidate",
        "checkpoint role differs",
    )
    _same(
        identity.get("exact_paper_initializer_verified"),
        False,
        "unverified initializer claim",
    )
    capture = worker["camera_capture_configuration"]
    _same(
        [capture.get("camera_height"), capture.get("camera_width")],
        [168, 224],
        "camera capture differs",
    )
    _same(capture.get("camera_gpu_id"), 0, "capture GPU ordering differs")
    _same(
        capture.get("source"),
        "pinned abc_minimal.config.SimEvalConfig defaults",
        "capture public constructor differs",
    )
    legacy._runtime_artifacts(reader, config, identity)


def _parent(
    reader: legacy._Reader,
    record: dict,
    worker_pin: dict,
    worker: dict,
    config: dict,
    *,
    allow_failed: bool = False,
) -> dict:
    parent = reader.record(record)
    _same(parent.get("algorithm"), "qf3-vla", "controller algorithm differs")
    _same(
        parent.get("summary_sha256"), worker_pin["sha256"], "parent worker pin differs"
    )
    _same(
        parent.get("training_config_sha256"),
        worker["input_config"]["sha256"],
        "parent input pin differs",
    )
    _same(parent.get("training_stage"), config["stage"], "parent stage differs")
    _same(parent.get("worker_status"), worker["status"], "parent worker status differs")
    _same(parent.get("seed"), config["seed"], "parent seed differs")
    _same(
        parent.get("policy_seed"), config["base"]["seed"], "parent policy seed differs"
    )
    _same(parent.get("task"), TASK, "parent task differs")
    _need(
        parent.get("status") == "completed"
        or allow_failed
        and parent.get("status") in ("failed", "timed_out"),
        "controller invocation incomplete",
    )
    legacy._time(parent.get("created_at"))
    _need(
        _finite(parent["metrics"].get("elapsed_seconds"))
        and parent["metrics"]["elapsed_seconds"] >= 0,
        "parent actual elapsed time missing",
    )
    for key in (
        "training_steps",
        "warmup_steps",
        "outer_iterations",
        "actor_updates",
        "critic_updates",
        "target_reached",
        "evaluation_steps",
        "physics_ticks_evaluation",
        "physics_ticks_training",
        "completed_training_episodes",
        "replay_total_added",
    ):
        _same(
            parent["metrics"].get(key),
            worker["metrics"].get(key),
            f"parent metric {key} differs",
        )
    if config["stage"] == "evaluate":
        for key, value in (
            ("evaluation_heads", config["evaluation"]["heads"]),
            ("evaluation_seeds", FINAL_SEEDS),
            ("evaluation_sampler_seed", config["evaluation"]["sampler_seed"]),
            ("evaluation_horizon_steps", 1000),
        ):
            _same(parent.get(key), value, f"parent {key} differs")
    return parent


def _tape(
    reader: legacy._Reader,
    record: dict,
    base_id: str,
    stage: str,
    worlds: int,
    *,
    namespace: str,
    requested: list[int] | None = None,
    actual: list[int] | None = None,
    outcomes: bool = True,
) -> dict:
    tape = reader.record(record)
    _same(
        [
            tape.get("schema_version"),
            tape.get("domain"),
            tape.get("protocol_id"),
            tape.get("base_id"),
            tape.get("stage"),
        ],
        [1, "sim", PROTOCOL, base_id, stage],
        "tape identity differs",
    )
    rollout = tape["rollout"]
    _same(rollout.get("partial"), False, "partial accepted tape refused")
    rows = rollout["worlds"]
    _need(isinstance(rows, list) and len(rows) == worlds, "tape world count differs")
    by_episode: dict[str, list[dict]] = {
        f"{namespace}-world-{j}": [] for j in range(worlds)
    }
    for decision in rollout["decisions"]:
        episode = decision.get("episode_id")
        _need(episode in by_episode, "unrelated/duplicate decision episode")
        by_episode[episode].append(decision)
    results, decision_count, evidence = [], 0, []
    for j, row in enumerate(rows):
        _same(row.get("world"), j, "world order differs")
        seed, observed = row.get("requested_seed"), row.get("actual_seed")
        _need(
            _integer(seed) and _integer(observed) and observed in (seed, seed + 100000),
            "actual reset seed differs",
        )
        if not outcomes:
            _need(seed < 2**31, "training reset outside native training seed namespace")
        if requested is not None:
            _same(seed, requested[j], "requested seed order differs")
        if actual is not None:
            _same(observed, actual[j], "actual seed order differs")
        _same(row.get("completed"), True, "incomplete accepted world")
        tracker = row["tracker"]
        names = [f"bottle_{i}" for i in range(1, 6)]
        _same(
            [
                tracker.get("schema_version"),
                tracker.get("protocol_revision"),
                tracker.get("task_id"),
                tracker.get("object_names"),
            ],
            [1, PROTOCOL, TASK, names],
            "tracker identity differs",
        )
        initial = tracker.get("initial_mask")
        _need(
            isinstance(initial, list)
            and len(initial) == 5
            and all(type(v) is bool for v in initial),
            "invalid reset predicate mask",
        )
        ever, steps, reward, last_native = [False] * 5, 0, 0.0, False
        decisions = by_episode[f"{namespace}-world-{j}"]
        for d, decision in enumerate(decisions):
            _same(decision.get("decision_index"), d, "decision order differs")
            _same(
                [decision.get("requested_seed"), decision.get("actual_seed")],
                [seed, observed],
                "decision seed differs",
            )
            events = decision.get("events")
            _need(
                isinstance(events, list) and 1 <= len(events) <= 15,
                "invalid decision duration",
            )
            if d < len(decisions) - 1:
                _same(len(events), 15, "short nonterminal prefix")
            legacy._matrix(
                decision.get("issued_normalized"),
                len(events),
                14,
                "issued normalized commands",
            )
            legacy._matrix(
                [decision.get("next_physical_state")], 1, 14, "next physical state"
            )
            if "evidence_id" in decision:
                body = {k: v for k, v in decision.items() if k != "evidence_id"}
                _same(
                    decision["evidence_id"],
                    _hash(body),
                    "decision evidence identity differs",
                )
                evidence.append(decision["evidence_id"])
            else:
                raise ComparisonError("decision evidence identity missing")
            for event in events:
                _need(
                    steps < 1000 and (not outcomes or not all(ever)),
                    "events after completion",
                )
                steps += 1
                _same(event.get("step"), steps, "event step order differs")
                if outcomes:
                    mask = event.get("ever_placed")
                    _need(
                        isinstance(mask, list)
                        and len(mask) == 5
                        and all(type(v) is bool for v in mask),
                        "invalid history mask",
                    )
                    _need(
                        all(
                            not old or new for old, new in zip(ever, mask, strict=True)
                        ),
                        "placement history reversed",
                    )
                    new = [
                        name
                        for name, a, b in zip(names, ever, mask, strict=True)
                        if b and not a
                    ]
                    _same(
                        event.get("new_objects"),
                        new,
                        "first placement transition differs",
                    )
                    _need(
                        _finite(event.get("reward")) and event["reward"] == len(new),
                        "first placement reward differs",
                    )
                    reward += len(new)
                    _same(
                        event.get("paper_success"),
                        all(mask),
                        "event history success differs",
                    )
                    _need(
                        type(event.get("native_success")) is bool,
                        "native judge missing",
                    )
                    ever, last_native = mask, event["native_success"]
        decision_count += len(decisions)
        _need(
            _integer(row.get("steps"), 1) and row["steps"] == steps <= 1000,
            "episode/decision length differs",
        )
        _same(row.get("decisions"), len(decisions), "episode decision count differs")
        _same(tracker.get("steps"), steps, "tracker steps differ")
        _same(tracker.get("done"), True, "tracker boundary incomplete")
        if outcomes:
            _need(all(ever) or steps == 1000, "unfinished episode")
            _same(tracker.get("ever_placed"), ever, "final tracker history differs")
            _same(row.get("paper_success"), all(ever), "world history success differs")
            _same(
                row.get("native_success"), last_native, "world native success differs"
            )
            _need(
                _finite(row.get("reward")) and row["reward"] == reward,
                "world reward differs",
            )
        else:
            _need(
                type(row.get("paper_success")) is bool
                and type(row.get("native_success")) is bool,
                "training world judges missing",
            )
            _need(
                row["paper_success"] or steps == 1000,
                "incomplete training episode falsely marked complete",
            )
            ever, last_native = [row["paper_success"]] * 5, row["native_success"]
        results.append(
            {
                "seed": seed,
                "actual_seed": observed,
                "length": steps,
                "history_success": all(ever),
                "native_current_success": last_native,
                "reward": reward,
            }
        )
    lengths = [r["length"] for r in results]
    for key, expected in (
        ("episodes", worlds),
        ("simulation_steps", sum(lengths)),
        ("actualsteps", sum(lengths)),
        ("physics_ticks", max(lengths)),
        ("successes", sum(r["history_success"] for r in results)),
    ):
        _same(rollout.get(key), expected, f"tape {key} differs")
    physical = rollout.get("physical_tape")
    _need(
        isinstance(physical, list) and len(physical) == max(lengths),
        "physical tape duration differs",
    )
    for tick in physical:
        legacy._matrix(tick, worlds, 14, "submitted physical commands")
    _same(rollout.get("reset_options"), RESET, "tape reset differs")
    profile = rollout["compiled_command_profile"]
    _need(
        profile.get("available") is True
        and profile.get("action_modified") is False
        and re.fullmatch(
            r"native_abc_affine/1:[0-9a-f]{64}", str(profile.get("revision"))
        )
        is not None,
        "compiled command profile differs",
    )
    _need(
        str(rollout.get("action_projection", "")).startswith("none added;"),
        "added projection",
    )
    successful = [r["length"] for r in results if r["history_success"]]
    native_lengths = [r["length"] for r in results if r["native_current_success"]]
    summary = {
        "completed_episodes": worlds,
        "successes": len(successful),
        "native_successes": len(native_lengths),
        "success_rate": len(successful) / worlds,
        "completed_failures": worlds - len(successful),
        "completed_episode_lengths": lengths,
        "successful_episode_lengths": successful,
        "mean_successful_episode_length": legacy._mean(successful),
        "failure_inclusive_mean_control_steps": legacy._mean(lengths),
        "native_current_success_only_mean_control_steps": legacy._mean(native_lengths),
        "simulation_steps": sum(lengths),
        "physics_ticks": max(lengths),
    }
    return {
        "rows": results,
        "summary": summary,
        "profile": profile,
        "decisions": decision_count,
        "evidence_id_sha256": _hash(evidence),
        "episode_ids": list(by_episode),
    }


def _summary(
    reported: dict, parsed: dict, *, noise: int, seeds: list[int], base_id: str
) -> None:
    for key, value in parsed["summary"].items():
        if key != "native_current_success_only_mean_control_steps":
            _same(reported.get(key), value, f"reported {key} differs from tape")
    for key, value in (
        ("attempted_episodes", 50),
        ("requested_episodes", 50),
        ("partial", False),
        ("partial_episode_lengths", []),
        ("sampler_seed", noise),
        ("configured_seed_sha256", _hash(seeds)),
        ("base_id", base_id),
    ):
        _same(reported.get(key), value, f"summary {key} differs")
    before = reported.get("training_state_before")
    _need(
        _sha(before) and before == reported.get("training_state_after"),
        "evaluation state changed",
    )


def _release(reader: legacy._Reader, request: dict) -> None:
    _need(
        INSTALL_REVIEW_SHA is not None,
        "real independent 0.4.10 installation review pin is not yet bound",
    )
    pin = request["release_wheel"]
    _same(
        [pin.get("sha256"), pin.get("bytes")],
        [RELEASE_SHA, RELEASE_BYTES],
        "release wheel differs",
    )
    _pin(reader, pin)
    review = request["independent_install_review"]
    _same(
        review.get("sha256"),
        INSTALL_REVIEW_SHA,
        "independent installation review differs",
    )
    reader.record(review)
    _need(
        NATIVE_INSTALL_REVIEW_SHA is not None,
        "actual native 0.4.10 installation admission is not yet bound",
    )
    _same(
        request["native_install_review"].get("sha256"),
        NATIVE_INSTALL_REVIEW_SHA,
        "native installation admission differs",
    )
    native_review = reader.record(request["native_install_review"])
    _same(
        native_review["actual_ABC_installed_proof"].get("sha256"),
        ACTUAL_INSTALL_PROOF_SHA,
        "actual installation proof differs",
    )
    reader.record(native_review["actual_ABC_installed_proof"])
    with zipfile.ZipFile(pin["path"]) as archive:
        members = [
            p for p in archive.namelist() if p.startswith("nrh/") and p.endswith(".py")
        ]
        _same(len(members), 40, "release module count differs")
        for name, checksum in PROFILE.items():
            _same(
                hashlib.sha256(
                    archive.read(name.replace(".", "/") + ".py")
                ).hexdigest(),
                checksum,
                "release source differs",
            )
        metadata = archive.read("nirvana_rl_harness-0.4.10.dist-info/METADATA").decode()
        _need("Version: 0.4.10\n" in metadata, "release version differs")
    _same(
        hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
        LEGACY_SHA,
        "legacy helper source changed",
    )
    _pin(reader, pin)


def _accepted_checks(value: dict, names: tuple[str, ...]) -> None:
    _same(value.get("schema_version"), 1, "independent admission schema differs")
    _need(
        value.get("status") == "accepted" and value.get("independent_review") is True,
        "independent accepted admission required",
    )
    for name in names:
        _same(value["checks"].get(name), True, f"independent {name} check missing")


def _input_preflight(
    reader: legacy._Reader, pin: dict, worker: dict, config: dict
) -> None:
    value = reader.record(pin)
    _accepted_checks(
        value, ("actual_installed_resolve", "artifact_native_source_inventory_pins")
    )
    for key, expected in (
        ("kind", "qf3_full_campaign_input_preflight"),
        ("source_sha256", PROFILE),
        ("release_wheel_sha256", RELEASE_SHA),
        ("native_install_review_sha256", NATIVE_INSTALL_REVIEW_SHA),
        ("continuation_id", _continuation(config)),
        ("resolved_config_sha256", _hash(config)),
    ):
        _same(value.get(key), expected, f"full input preflight {key} differs")
    _match_pin(
        value["input_config"],
        worker["input_config"],
        "full input preflight requested config differs",
    )
    _match_pin(
        value["resolved_config"],
        worker["resolved_config"],
        "full input preflight resolved config differs",
    )


def _admission(
    reader: legacy._Reader, record: dict, worker: dict, config: dict
) -> dict:
    admission = reader.record(record)
    _same(
        admission.get("kind"),
        "qf3_full_campaign_checkpoint_admission",
        "training admission kind differs",
    )
    _accepted_checks(admission, AUDIT_CHECKS)
    for key, value in (
        ("source_sha256", PROFILE),
        ("worker_revision", WORKER),
        ("checkpoint_schema", 4),
        ("block_revision", "sadhana.qf3-replay-blocks/1"),
        ("base_id", worker["base_id"]),
        ("protocol_id", PROTOCOL),
        ("continuation_id", _continuation(config)),
    ):
        _same(admission.get(key), value, f"admission {key} differs")
    checkpoint = admission["admitted_checkpoint"]
    _same(
        checkpoint.get("kind"),
        "qf3_completed_boundary_checkpoint",
        "admitted checkpoint role differs",
    )
    _pin(reader, checkpoint)
    _need(
        any(
            _hash({k: p.get(k) for k in ("path", "bytes", "sha256")})
            == _hash({k: checkpoint.get(k) for k in ("path", "bytes", "sha256")})
            for p in worker["completed_checkpoints"]
        ),
        "admitted checkpoint absent from worker completed lineage",
    )
    for key in (
        "policy_state_sha256",
        "canonical_replay_sha256",
        "exposed_state_sha256",
        "evaluation_preservation_state_sha256",
        "ordered_record_selectors_sha256",
        "learner_state_sha256",
        "training_rng_sha256",
    ):
        _need(_sha(admission.get(key)), f"admission {key} missing")
    replay = admission["replay_metadata"]
    for key, value in (
        ("schema_version", 1),
        ("contract", "sadhana.qf3-abc-proposal-issued-prefix/1"),
        ("capacity", 50000),
        ("base_id", worker["base_id"]),
        ("protocol_id", PROTOCOL),
        ("seed", config["seed"]),
    ):
        _same(replay.get(key), value, f"admitted replay {key} differs")
    total, size = replay.get("total_added"), replay.get("size")
    _need(
        _integer(total) and _integer(size) and size == min(total, 50000),
        "admitted ring counters differ",
    )
    _same(replay.get("cursor"), total % 50000, "admitted ring cursor differs")
    blocks = admission["replay_blocks"]
    _need(
        isinstance(blocks, list) and len(blocks) <= size, "admission block refs invalid"
    )
    seen, selected, end = set(), 0, 0
    for block in blocks:
        pin = block["artifact"]
        checksum = pin.get("sha256")
        _need(checksum not in seen, "duplicate admission block")
        seen.add(checksum)
        first, count, selectors = (
            block.get(k) for k in ("first_ordinal", "record_count", "selector_count")
        )
        _need(
            _integer(first)
            and _integer(count, 1)
            and _integer(selectors, 1)
            and selectors <= count <= 50000
            and first >= end
            and first + count <= total,
            "invalid/unused admission block reference",
        )
        end, selected = first + count, selected + selectors
        _same(
            pin.get("kind"),
            "qf3_immutable_replay_block",
            "admission block kind differs",
        )
        _pin(reader, pin)
    _same(selected, size, "admitted selector count differs")
    boundary = admission["boundary"]
    for key in (
        "iteration",
        "warmup_batches",
        "training_steps",
        "warmup_steps",
        "physics_ticks",
        "completed_episodes",
        "evaluation_steps",
        "evaluation_physics_ticks",
        "discarded_evaluation_steps",
        "discarded_evaluation_physics_ticks",
    ):
        _need(_integer(boundary.get(key)), f"invalid accepted boundary {key}")
    _same(boundary.get("partial"), False, "partial admitted boundary")
    pending = boundary["evaluation_state"].get("pending_validation")
    _need(
        pending is None or isinstance(pending, dict) and set(pending) == {"manifest"},
        "invalid admitted pending validation",
    )
    k, batches = boundary["iteration"], boundary["warmup_batches"]
    _need(batches <= 4, "too many accepted warmups")
    _same(
        boundary.get("completed_episodes"),
        80 * batches + 16 * k,
        "admitted completed episodes differ",
    )
    _same(
        boundary.get("boundary"),
        "completed_outer" if k else "completed_warmup_batch",
        "admitted boundary phase differs",
    )
    _same(
        admission["counters"],
        {
            "critic_updates": 1600 * k,
            "actor_updates": 200 * k,
            "critic_target_updates": 1600 * k,
            "actor_target_updates": k,
        },
        "accepted update/target schedule differs",
    )
    _need(not k or batches == 4, "outer before completed warmup")
    return admission


def _fresh(
    reader: legacy._Reader,
    event: dict,
    worker: dict,
    config: dict,
    index: int,
    ordinal: int,
    tape_pin: dict,
    *,
    geometry_seen: set[str],
) -> dict:
    outer = ordinal + 1
    for key, value in (
        ("head", "learned"),
        ("reason", "periodic"),
        ("outer_iterations", outer),
        ("crossed_thresholds", [outer]),
    ):
        _same(event.get(key), value, f"fresh event {key} differs")
    _need(_integer(event.get("training_control_steps"), 1), "fresh xaxis missing")
    validation = event["validation"]
    manifest = reader.record(validation["manifest"])
    seeds = [2**40 + index * 2**39 + (50 * ordinal + j) * 200001 for j in range(50)]
    noise = config["evaluation"]["sampler_seed"] + ordinal + 1
    for key, value in (
        ("schema_version", 1),
        ("revision", "sadhana.qf3-fresh-validation/1"),
        ("domain", "sim"),
        ("continuation_id", _continuation(config)),
        ("source_sha256", _hash(PROFILE)),
        ("base_id", worker["base_id"]),
        ("protocol_id", PROTOCOL),
        ("head", "learned"),
        ("cursor", ordinal),
        ("outer_iterations", outer),
        ("crossed_thresholds", [outer]),
        ("training_control_steps", event["training_control_steps"]),
        ("actor_updates", 200 * outer),
        ("requested_seeds", seeds),
        ("reset_options", RESET),
        ("noise_seed", noise),
    ):
        _same(manifest.get(key), value, f"fresh manifest {key} differs")
    _need(_sha(manifest.get("policy_state_sha256")), "fresh policy hash missing")
    noise_rng = manifest.get("noise_rng")
    _noise_rng(noise_rng)
    reset_pins = validation["reset_evidence"]
    _same(len(reset_pins), 1, "fresh50 reset batch count differs")
    reset = reader.record(reset_pins[0])
    for key, value in (
        ("schema_version", 1),
        ("revision", "sadhana.qf3-fresh-validation/1"),
        ("domain", "sim"),
        ("manifest_sha256", validation["manifest"]["sha256"]),
        ("start", 0),
    ):
        _same(reset.get(key), value, f"fresh reset {key} differs")
    worlds = reset["worlds"]
    _same(len(worlds), 50, "fresh reset worlds differ")
    actual = []
    for requested, row in zip(seeds, worlds, strict=True):
        _same(row.get("requested_seed"), requested, "fresh reset requested differs")
        observed = row.get("actual_seed")
        _need(
            _integer(observed) and observed in (requested, requested + 100000),
            "fresh fallback differs",
        )
        actual.append(observed)
        randomization = row["randomization"]
        _same(randomization.get("seed"), observed, "randomization seed differs")
        objects, scales, metadata = (
            randomization[k] for k in ("object_states", "scale_states", "metadata")
        )
        _need(
            set(objects) == {"bin_joint", *(f"bottle_{i}_joint" for i in range(1, 6))},
            "fresh object names differ",
        )
        _same(scales, {}, "fresh scales differ")
        _same(metadata.get("bottle_count"), 5, "fresh bottle count differs")
        variants = metadata.get("bottle_variants")
        _need(
            isinstance(variants, list)
            and len(variants) == 5
            and all(isinstance(v, str) and v for v in variants),
            "fresh variants invalid",
        )
        _same(metadata.get("bottle_variant"), variants[0], "fresh variant differs")
        for pose in objects.values():
            for key, size in (("pos", 3), ("quat", 4)):
                vector = pose[key]
                _need(
                    isinstance(vector, list)
                    and len(vector) == size
                    and all(_finite(v) for v in vector),
                    "fresh pose invalid",
                )
        geometry = _hash({"object_states": objects, "scale_states": scales})
        _same(row.get("geometry_sha256"), geometry, "fresh geometry hash differs")
        _need(geometry not in geometry_seen, "accepted fresh geometry repeated")
        geometry_seen.add(geometry)
    stage = f"evaluate-learned-{ordinal}"
    parsed = _tape(
        reader,
        tape_pin,
        worker["base_id"],
        stage,
        50,
        namespace=stage + "-0",
        requested=seeds,
        actual=actual,
    )
    _summary(
        event["summary"], parsed, noise=noise, seeds=seeds, base_id=worker["base_id"]
    )
    _same(
        event["summary"].get("actor_updates"),
        200 * outer,
        "fresh actor counter differs",
    )
    return {
        "manifest": validation["manifest"],
        "noise_rng_sha256": _hash(noise_rng),
        "policy_state_sha256": manifest["policy_state_sha256"],
        **parsed,
    }


def _accounting(parsed: dict) -> dict:
    return {
        "episodes": parsed["summary"]["completed_episodes"],
        "simulation_steps": parsed["summary"]["simulation_steps"],
        "physics_ticks": parsed["summary"]["physics_ticks"],
        "decisions": parsed["decisions"],
        "evidence_id_sha256": parsed["evidence_id_sha256"],
        "episode_ids": parsed["episode_ids"],
    }


def _pending(
    reader: legacy._Reader, admission: dict, worker: dict, config: dict, index: int
) -> dict | None:
    state = admission["boundary"]["evaluation_state"]
    pending = state["pending_validation"]
    if pending is None:
        return None
    manifest = reader.record(pending["manifest"])
    ordinal, outer = state["validation_cursor"], admission["boundary"]["iteration"]
    _same(ordinal + 1, outer, "pending validation skipped an outer")
    seeds = [2**40 + index * 2**39 + (ordinal * 50 + j) * 200001 for j in range(50)]
    for key, expected in (
        ("schema_version", 1),
        ("revision", "sadhana.qf3-fresh-validation/1"),
        ("domain", "sim"),
        ("continuation_id", _continuation(config)),
        ("source_sha256", _hash(PROFILE)),
        ("base_id", worker["base_id"]),
        ("protocol_id", PROTOCOL),
        ("head", "learned"),
        ("cursor", ordinal),
        ("outer_iterations", outer),
        ("crossed_thresholds", [outer]),
        ("training_control_steps", admission["boundary"]["training_steps"]),
        ("actor_updates", 200 * outer),
        ("requested_seeds", seeds),
        ("reset_options", RESET),
        ("noise_seed", config["evaluation"]["sampler_seed"] + ordinal + 1),
        ("policy_state_sha256", admission["policy_state_sha256"]),
    ):
        _same(manifest.get(key), expected, f"pending manifest {key} differs")
    _noise_rng(manifest["noise_rng"])
    _same(
        admission.get("pending_noise_rng_sha256"),
        _hash(manifest["noise_rng"]),
        "pending noise state audit differs",
    )
    return pending


def _discarded(reader: legacy._Reader, pin: dict, base_id: str) -> dict:
    """Count attempted controls without admitting incomplete outcomes or labels."""
    tape = reader.record(pin)
    for key, value in (
        ("schema_version", 1),
        ("domain", "sim"),
        ("protocol_id", PROTOCOL),
        ("base_id", base_id),
    ):
        _same(tape.get(key), value, f"discarded tape {key} differs")
    stage, rollout = tape["stage"], tape["rollout"]
    _need(
        stage in ("warmup", "train") or stage.startswith("evaluate-learned-"),
        "unknown discarded stage",
    )
    _need(type(rollout.get("partial")) is bool, "discarded partial type differs")
    rows = rollout["worlds"]
    worlds = 80 if stage == "warmup" else 16 if stage == "train" else 50
    _same(len(rows), worlds, "discarded world count differs")
    lengths, completed = [], 0
    for index, row in enumerate(rows):
        _same(row.get("world"), index, "discarded world order differs")
        _need(
            _integer(row.get("steps")) and row["steps"] <= 1000,
            "discarded length invalid",
        )
        _need(type(row.get("completed")) is bool, "discarded completion invalid")
        _need(
            _integer(row.get("requested_seed"))
            and _integer(row.get("actual_seed"))
            and row["actual_seed"]
            in (row["requested_seed"], row["requested_seed"] + 100000),
            "discarded reset seed invalid",
        )
        lengths.append(row["steps"])
        completed += row["completed"]
    steps, ticks = sum(lengths), max(lengths)
    for key, value in (
        ("simulation_steps", steps),
        ("actualsteps", steps),
        ("physics_ticks", ticks),
        ("episodes", completed),
    ):
        _same(rollout.get(key), value, f"discarded {key} differs")
    _same(
        len(rollout["physical_tape"]), ticks, "discarded physical tape length differs"
    )
    for tick in rollout["physical_tape"]:
        legacy._matrix(tick, worlds, 14, "discarded submitted controls")
    decisions = rollout["decisions"]
    _need(isinstance(decisions, list), "discarded decision evidence missing")
    by_episode = {}
    for decision in decisions:
        episode = decision.get("episode_id")
        _need(isinstance(episode, str) and episode, "discarded episode id missing")
        events = decision.get("events")
        _need(
            isinstance(events, list) and 1 <= len(events) <= 15,
            "discarded decision duration invalid",
        )
        legacy._matrix(
            decision.get("issued_normalized"),
            len(events),
            14,
            "discarded normalized commands",
        )
        legacy._matrix(
            [decision.get("next_physical_state")], 1, 14, "discarded next state"
        )
        _same(
            decision.get("evidence_id"),
            _hash({k: v for k, v in decision.items() if k != "evidence_id"}),
            "discarded evidence identity differs",
        )
        prior = by_episode.setdefault(episode, {"steps": 0, "decisions": 0})
        _same(
            decision.get("decision_index"),
            prior["decisions"],
            "discarded decision order differs",
        )
        for event in events:
            prior["steps"] += 1
            _same(event.get("step"), prior["steps"], "discarded event order differs")
        prior["decisions"] += 1
    _same(
        sum(v["steps"] for v in by_episode.values()),
        steps,
        "discarded decisions do not match attempted controls",
    )
    return {
        "stage": stage,
        "simulation_steps": steps,
        "physics_ticks": ticks,
        "completed_episodes": completed,
        "decisions": len(decisions),
    }


def _updates(worker: dict, previous: dict, accepted: dict) -> dict:
    rows = worker["updates"]
    _need(isinstance(rows, list), "update log missing")
    counts = {"critic": previous["critic_updates"], "actor": previous["actor_updates"]}
    actual = {"critic": 0, "actor": 0}
    expected_iteration = previous["actor_target_updates"]
    in_iteration = {"critic": 0, "actor": 0}
    for row in rows:
        kind = row.get("kind")
        _need(kind in counts, "unknown update kind")
        _same(row.get("iteration"), expected_iteration, "update outer order differs")
        limit = 1600 if kind == "critic" else 200
        _need(in_iteration[kind] < limit, "too many updates in outer")
        _need(
            kind == "critic"
            and not in_iteration["actor"]
            or kind == "actor"
            and in_iteration["critic"] == 1600,
            "actor/critic update ordering differs",
        )
        counts[kind] += 1
        actual[kind] += 1
        in_iteration[kind] += 1
        _same(row.get(kind + "_updates"), counts[kind], "optimizer counter jump/repeat")
        _need(
            all(_finite(v) for k, v in row.items() if k not in ("kind", "iteration")),
            "nonfinite update metric",
        )
        if in_iteration["actor"] == 200:
            expected_iteration += 1
            in_iteration = {"critic": 0, "actor": 0}
    for kind, count in counts.items():
        _need(
            count >= accepted[kind + "_updates"],
            "accepted update missing raw log",
        )
        _same(
            worker["phase_counts"].get(kind + "_updates"),
            actual[kind],
            "invocation update count differs",
        )
        _same(
            worker["metrics"].get(kind + "_updates"),
            count,
            "reported optimizer count differs",
        )
    _same(
        worker["metrics"].get("outer_iterations"),
        expected_iteration,
        "raw outer counter lacks completed update block",
    )
    return {
        "accepted": {
            k: accepted[k + "_updates"] - previous[k + "_updates"] for k in counts
        },
        "discarded": {k: counts[k] - accepted[k + "_updates"] for k in counts},
    }


def _chain(reader: legacy._Reader, records: list[dict], index: int) -> dict:
    _need(isinstance(records, list) and records, "training chain missing")
    zero_counters = {
        "critic_updates": 0,
        "actor_updates": 0,
        "critic_target_updates": 0,
        "actor_target_updates": 0,
    }
    previous = {
        "iteration": 0,
        "warmup_batches": 0,
        "training_steps": 0,
        "warmup_steps": 0,
        "physics_ticks": 0,
        "completed_episodes": 0,
        "evaluation_steps": 0,
        "evaluation_physics_ticks": 0,
        "discarded_evaluation_steps": 0,
        "discarded_evaluation_physics_ticks": 0,
    }
    previous_counters, previous_admission = zero_counters, None
    history, geometry_seen, lineage = [], set(), []
    previous_pending, total_decisions = None, 0
    identity, continuation, selected, config = None, None, None, None
    invocations, costs, parents_seen, policies = [], 0.0, set(), []
    for entry in records:
        worker = reader.record(entry["worker"])
        config = _config(reader, worker, index)
        _input_preflight(reader, entry["input_preflight"], worker, config)
        _same(config.get("stage"), "train", "training chain stage differs")
        _runtime(reader, worker, config)
        parent = _parent(
            reader, entry["parent"], entry["worker"], worker, config, allow_failed=True
        )
        _need(
            worker.get("status") in ("completed", "budget_stopped", "failed"),
            "unknown worker state",
        )
        _need(
            entry["parent"]["sha256"] not in parents_seen,
            "duplicate controller invocation",
        )
        parents_seen.add(entry["parent"]["sha256"])
        costs += parent["metrics"]["elapsed_seconds"]
        if invocations:
            prior = selected["parent"]
            _need(
                legacy._time(parent["created_at"])
                >= legacy._time(prior["created_at"])
                + timedelta(seconds=prior["metrics"]["elapsed_seconds"]),
                "resume invocation precedes prior accepted invocation end",
            )
        admission = _admission(reader, entry["admission"], worker, config)
        for name, pin in (
            ("worker_receipt", entry["worker"]),
            ("controller_receipt", entry["parent"]),
            ("resolved_config", worker["resolved_config"]),
        ):
            _match_pin(admission[name], pin, f"admission {name} differs")
        _same(
            admission.get("predecessor_admission"),
            previous_admission,
            "admission predecessor differs",
        )
        if previous_admission is None:
            _same(
                config.get("resume"),
                None,
                "fresh full campaign required; inherited credit refused",
            )
            identity, continuation = worker["runtime_identity"], _continuation(config)
        else:
            _match_pin(
                config["resume"],
                selected["admission"]["admitted_checkpoint"],
                "resume does not use prior accepted bytes",
            )
            _same(
                worker.get("restore_verified"),
                True,
                "resumed training restore unverified",
            )
            _same(
                worker["runtime_identity"],
                identity,
                "training runtime identity changed",
            )
            _same(_continuation(config), continuation, "training continuation changed")
        boundary, counters = admission["boundary"], admission["counters"]
        b, k = boundary["warmup_batches"], boundary["iteration"]
        _need(
            b >= previous["warmup_batches"] and k >= previous["iteration"],
            "accepted boundary regressed",
        )
        if previous["training_steps"] >= TARGET:
            _same(
                k,
                previous["iteration"],
                "training continued beyond first target while validation pending",
            )
        expected_ordinals = [
            ("warmup", j) for j in range(previous["warmup_batches"], b)
        ] + [("train", j) for j in range(previous["iteration"], k)]
        accepted_tapes = admission["accepted_tapes"]
        _same(
            [[x.get("category"), x.get("ordinal")] for x in accepted_tapes],
            [list(x) for x in expected_ordinals],
            "new accepted tape ordinals skip/repeat",
        )
        deltas = {
            "warmup_steps": 0,
            "training_steps": 0,
            "physics_ticks": 0,
            "completed_episodes": 0,
        }
        consumed, axes = [], {}
        axis = previous["training_steps"]
        for item, (category, ordinal) in zip(
            accepted_tapes, expected_ordinals, strict=True
        ):
            pin = item["artifact"]
            _need(pin["sha256"] not in lineage, "new accepted tape repeated")
            parsed = _tape(
                reader,
                pin,
                worker["base_id"],
                category,
                80 if category == "warmup" else 16,
                namespace=f"{category}-{ordinal}",
                outcomes=False,
            )
            accounting = _accounting(parsed)
            _same(
                item.get("accounting"),
                accounting,
                "audited training tape accounting differs",
            )
            _same(
                item.get("source_sha256"),
                PROFILE,
                "training label audit source differs",
            )
            steps = parsed["summary"]["simulation_steps"]
            deltas["warmup_steps" if category == "warmup" else "training_steps"] += (
                steps
            )
            deltas["physics_ticks"] += parsed["summary"]["physics_ticks"]
            deltas["completed_episodes"] += parsed["summary"]["completed_episodes"]
            total_decisions += parsed["decisions"]
            lineage.append(pin["sha256"])
            consumed.append(pin)
            if category == "train":
                axis += steps
                axes[ordinal] = axis
        for key, delta in deltas.items():
            _same(
                boundary.get(key),
                previous[key] + delta,
                f"accepted {key} delta lacks new tape evidence",
            )
        _same(
            admission.get("covered_training_tape_sha256"),
            lineage,
            "occupied/evicted audit lineage differs",
        )
        _same(
            admission["replay_metadata"].get("total_added"),
            total_decisions,
            "canonical replay total differs from accepted decision lineage",
        )
        state = boundary["evaluation_state"]
        current_history = state["history"]
        _same(
            current_history[: len(history)],
            history,
            "accepted history changed on resume",
        )
        h = len(current_history)
        _need(h in (k, max(0, k - 1)), "fresh validation skipped an accepted outer")
        _need(
            (b, k, h)
            != (previous["warmup_batches"], previous["iteration"], len(history)),
            "accepted boundary repeated",
        )
        _same(state.get("baseline"), None, "frozen development baseline credit refused")
        _same(state.get("validation_cursor"), h, "accepted fresh cursor differs")
        _same(state.get("next_threshold"), h + 1, "accepted threshold differs")
        pending = _pending(reader, admission, worker, config, index)
        _need(
            pending is None or h == k - 1, "pending manifest after completed validation"
        )
        new_fresh = admission["fresh_tapes"]
        _same(
            [r.get("event_index") for r in new_fresh],
            list(range(len(history), h)),
            "fresh event ordinals skip/repeat",
        )
        fresh_steps, fresh_ticks = 0, 0
        for item in new_fresh:
            ordinal = item["event_index"]
            event = current_history[ordinal]
            _same(
                event.get("training_control_steps"),
                axes.get(ordinal, previous["training_steps"]),
                "fresh x-axis differs from its new rollout",
            )
            if previous_pending is not None and ordinal == len(history):
                _match_pin(
                    event["validation"]["manifest"],
                    previous_pending["manifest"],
                    "pending retry replaced its immutable manifest/noise intent",
                )
            _need(
                _integer(event.get("training_control_steps"), 1), "fresh x-axis invalid"
            )
            prior_axis = (
                current_history[ordinal - 1]["training_control_steps"] if ordinal else 0
            )
            _need(
                prior_axis < event["training_control_steps"],
                "fresh x-axis did not advance",
            )
            _need(
                prior_axis < TARGET,
                "accepted boundary after first target crossing refused",
            )
            parsed = _fresh(
                reader,
                event,
                worker,
                config,
                index,
                ordinal,
                item["artifact"],
                geometry_seen=geometry_seen,
            )
            for key in ("noise_rng_sha256", "policy_state_sha256"):
                _same(
                    item.get(key),
                    parsed[key],
                    f"fresh independent {key} binding differs",
                )
            fresh_steps += parsed["summary"]["simulation_steps"]
            fresh_ticks += parsed["summary"]["physics_ticks"]
            policies.append(parsed["policy_state_sha256"])
            consumed.append(item["artifact"])
        if h == k:
            _same(
                boundary["training_steps"],
                current_history[-1]["training_control_steps"] if k else 0,
                "accepted x-axis differs from controls",
            )
        accepted_eval = sum(e["summary"]["simulation_steps"] for e in current_history)
        accepted_ticks = sum(e["summary"]["physics_ticks"] for e in current_history)
        _same(
            boundary["evaluation_steps"],
            accepted_eval + boundary["discarded_evaluation_steps"],
            "accepted/discarded eval controls differ",
        )
        _same(
            boundary["evaluation_physics_ticks"],
            accepted_ticks + boundary["discarded_evaluation_physics_ticks"],
            "accepted/discarded eval ticks differ",
        )
        discarded = {
            "warmup_steps": 0,
            "training_steps": 0,
            "evaluation_steps": 0,
            "physics_ticks": 0,
        }
        discarded_training_episodes, discarded_training_ticks, discarded_eval_ticks = (
            0,
            0,
            0,
        )
        discarded_evaluations = []
        for item in admission["discarded_tapes"]:
            parsed = _discarded(reader, item["artifact"], worker["base_id"])
            _same(
                item.get("accounting"),
                parsed,
                "discarded tape audit accounting differs",
            )
            key = (
                "warmup_steps"
                if parsed["stage"] == "warmup"
                else "training_steps"
                if parsed["stage"] == "train"
                else "evaluation_steps"
            )
            discarded[key] += parsed["simulation_steps"]
            discarded["physics_ticks"] += parsed["physics_ticks"]
            if key == "evaluation_steps":
                discarded_eval_ticks += parsed["physics_ticks"]
                discarded_evaluations.append(
                    (
                        item["artifact"]["sha256"],
                        parsed["simulation_steps"],
                        parsed["physics_ticks"],
                    )
                )
            else:
                discarded_training_episodes += parsed["completed_episodes"]
                discarded_training_ticks += parsed["physics_ticks"]
            consumed.append(item["artifact"])
        retained = admission.get("retained_discarded_evaluation_tape_sha256", [])
        _need(
            isinstance(retained, list)
            and all(_sha(v) for v in retained)
            and len(set(retained)) == len(retained),
            "invalid retained discarded-evaluation tape prefix",
        )
        _same(
            retained,
            [v[0] for v in discarded_evaluations[: len(retained)]],
            "retained discarded-evaluation tapes are not the pinned attempted prefix",
        )
        retained_steps = sum(v[1] for v in discarded_evaluations[: len(retained)])
        retained_ticks = sum(v[2] for v in discarded_evaluations[: len(retained)])
        for key, expected in (
            (
                "discarded_evaluation_steps",
                previous["discarded_evaluation_steps"] + retained_steps,
            ),
            (
                "discarded_evaluation_physics_ticks",
                previous["discarded_evaluation_physics_ticks"] + retained_ticks,
            ),
            (
                "evaluation_steps",
                previous["evaluation_steps"] + fresh_steps + retained_steps,
            ),
            (
                "evaluation_physics_ticks",
                previous["evaluation_physics_ticks"] + fresh_ticks + retained_ticks,
            ),
        ):
            _same(
                boundary.get(key),
                expected,
                f"canonical {key} delta lacks retained tape evidence",
            )
        raw_pins = worker["rollouts"]
        indexed = {(p["category"], p["ordinal"]): p["artifact"] for p in accepted_tapes}
        fresh_indexed = {p["event_index"]: p["artifact"] for p in new_fresh}
        causal_prefix = []
        if len(history) < previous["iteration"]:
            _need(
                len(history) in fresh_indexed,
                "pending fresh event was not completed before new training",
            )
            causal_prefix.append(fresh_indexed[len(history)])
        for ordinal in range(previous["warmup_batches"], b):
            causal_prefix.append(indexed[("warmup", ordinal)])
        for ordinal in range(previous["iteration"], k):
            causal_prefix.append(indexed[("train", ordinal)])
            if ordinal in fresh_indexed:
                causal_prefix.append(fresh_indexed[ordinal])
        expected_order = causal_prefix + [
            p["artifact"] for p in admission["discarded_tapes"]
        ]
        _same(
            [{key: p[key] for key in ("path", "bytes", "sha256")} for p in raw_pins],
            [
                {key: p[key] for key in ("path", "bytes", "sha256")}
                for p in expected_order
            ],
            "raw collection order skips/repeats validation or trains before pending retry",
        )
        _same(
            len({p["sha256"] for p in consumed}), len(consumed), "raw tape used twice"
        )
        _same(
            sorted(
                _hash({key: p[key] for key in ("path", "bytes", "sha256")})
                for p in consumed
            ),
            sorted(
                _hash({key: p[key] for key in ("path", "bytes", "sha256")})
                for p in raw_pins
            ),
            "raw rollouts not exactly accepted/discarded partition",
        )
        for key, expected in (
            ("warmup_steps", deltas["warmup_steps"] + discarded["warmup_steps"]),
            ("train_steps", deltas["training_steps"] + discarded["training_steps"]),
            ("evaluation_steps", fresh_steps + discarded["evaluation_steps"]),
        ):
            _same(
                worker["phase_counts"].get(key), expected, f"invocation {key} differs"
            )
        invocation_events = worker["invocation_evaluation_events"]
        _need(isinstance(invocation_events, list), "invocation event ledger missing")
        _same(
            [
                e
                for e in invocation_events
                if e.get("summary", {}).get("partial") is False
            ],
            current_history[len(history) :],
            "new complete fresh events do not match invocation-only ledger",
        )
        _need(
            all(
                e.get("head") == "learned"
                and e.get("reason") == "periodic"
                and type(e.get("summary", {}).get("partial")) is bool
                for e in invocation_events
            ),
            "training invocation has unrelated final/baseline event",
        )
        _same(
            sum(
                e["summary"]["simulation_steps"]
                for e in invocation_events
                if e["summary"]["partial"]
            ),
            discarded["evaluation_steps"],
            "partial invocation events do not match discarded evaluation controls",
        )
        update_delta = _updates(worker, previous_counters, counters)
        accounting = {
            "accepted": deltas,
            "discarded": discarded,
            "updates": update_delta,
            "accepted_fresh_steps": fresh_steps,
            "accepted_fresh_ticks": fresh_ticks,
        }
        _same(
            admission["invocation_accounting"],
            accounting,
            "independent invocation delta accounting differs",
        )
        for key in ("warmup_steps", "training_steps"):
            _same(
                worker["metrics"].get(key),
                previous[key] + deltas[key] + discarded[key],
                "raw cumulative counter differs",
            )
        for key, expected in (
            (
                "evaluation_steps",
                previous["evaluation_steps"]
                + fresh_steps
                + discarded["evaluation_steps"],
            ),
            (
                "physics_ticks_evaluation",
                previous["evaluation_physics_ticks"]
                + fresh_ticks
                + discarded_eval_ticks,
            ),
            (
                "physics_ticks_training",
                previous["physics_ticks"]
                + deltas["physics_ticks"]
                + discarded_training_ticks,
            ),
            (
                "completed_training_episodes",
                previous["completed_episodes"]
                + deltas["completed_episodes"]
                + discarded_training_episodes,
            ),
            (
                "replay_total_added",
                total_decisions
                + sum(
                    q["accounting"].get("decisions", 0)
                    for q in admission["discarded_tapes"]
                    if q["accounting"]["stage"] in ("warmup", "train")
                ),
            ),
        ):
            _same(
                worker["metrics"].get(key),
                expected,
                f"raw {key} lacks invocation evidence",
            )
        _same(
            worker["metrics"].get("target_reached"),
            worker["metrics"]["training_steps"] >= TARGET,
            "raw target flag differs",
        )
        if boundary["training_steps"] >= TARGET and h == k:
            _same(
                entry,
                records[-1],
                "training chain continued beyond first accepted target boundary",
            )
            _same(
                worker["status"],
                "completed",
                "target selection requires completed training",
            )
            _same(worker["metrics"].get("target_reached"), True, "target flag false")
            _same(pending, None, "selected validation remains pending")
            for key in ("training_steps", "warmup_steps"):
                _same(
                    discarded[key],
                    0,
                    "selected completed invocation has discarded training controls",
                )
            for key in ("critic_updates", "actor_updates"):
                _same(
                    worker["metrics"].get(key),
                    counters[key],
                    "selected optimizer counter differs from completed boundary",
                )
            _same(
                worker["metrics"].get("outer_iterations"),
                k,
                "selected completed outer differs",
            )
        invocations.append(
            {
                "worker": entry["worker"],
                "parent": entry["parent"],
                "admission": entry["admission"],
                "input_preflight": entry["input_preflight"],
                "accepted_boundary": {
                    key: value
                    for key, value in boundary.items()
                    if key != "evaluation_state"
                },
                "accepted_evaluation_state_sha256": _hash(boundary["evaluation_state"]),
                "retained_discarded_evaluation_tape_sha256": retained,
                "accounting": accounting,
                "parent_elapsed_seconds": parent["metrics"]["elapsed_seconds"],
            }
        )
        selected = {
            "worker": worker,
            "worker_pin": entry["worker"],
            "parent": parent,
            "parent_pin": entry["parent"],
            "admission": admission,
            "admission_pin": entry["admission"],
        }
        previous_pending = pending
        previous, previous_counters, previous_admission, history = (
            boundary,
            counters,
            entry["admission"],
            current_history,
        )
    _need(
        previous["warmup_batches"] == 4
        and previous["training_steps"] >= TARGET
        and previous["iteration"] >= 1,
        "full post-warmup target not reached",
    )
    _same(
        len(history),
        previous["iteration"],
        "selected target lacks a new complete fresh50 event",
    )
    _same(previous_pending, None, "selected target validation remains pending")
    _same(
        selected["admission"]["policy_state_sha256"],
        policies[-1],
        "selected full policy differs from accepted final fresh event",
    )
    return {
        "selected": selected,
        "config": config,
        "continuation_id": continuation,
        "runtime_identity": identity,
        "invocations": invocations,
        "parent_elapsed_seconds": costs,
        "training_steps": previous["training_steps"],
        "warmup_steps": previous["warmup_steps"],
        "warmup_plus_training_steps": previous["warmup_steps"]
        + previous["training_steps"],
        "outer_iterations": previous["iteration"],
        "accepted_fresh_events": len(history),
    }


def _selection(reader: legacy._Reader, pin: dict, chain: dict) -> dict:
    value = reader.record(pin)
    _accepted_checks(
        value,
        (
            "first_target_boundary",
            "all_completed_boundary_lineage",
            "fresh_validation_complete",
        ),
    )
    _same(
        value.get("kind"),
        "qf3_full_campaign_checkpoint_selection",
        "selection kind differs",
    )
    selected = chain["selected"]
    for name, expected in (
        ("checkpoint", selected["admission"]["admitted_checkpoint"]),
        ("training_admission", selected["admission_pin"]),
        ("training_receipt", selected["worker_pin"]),
        ("resolved_config", selected["worker"]["resolved_config"]),
    ):
        _match_pin(value[name], expected, f"selection {name} differs")
    for name, expected in (
        ("base_id", selected["worker"]["base_id"]),
        ("protocol_id", PROTOCOL),
        ("source_sha256", PROFILE),
        ("continuation_id", chain["continuation_id"]),
        ("policy_state_sha256", selected["admission"]["policy_state_sha256"]),
    ):
        _same(value.get(name), expected, f"selection {name} differs")
    rule = {
        "kind": "first_completed_outer_at_post_warmup_target",
        "target_control_steps": TARGET,
        "training_steps": chain["training_steps"],
        "outer_iterations": chain["outer_iterations"],
        "fresh_validation_accepted": True,
    }
    _same(
        value.get("rule"),
        rule,
        "selection is not the prespecified first-target boundary",
    )
    selected_at = legacy._time(value.get("selected_at_utc"))
    _need(
        legacy._time(selected["parent"]["created_at"])
        + timedelta(seconds=selected["parent"]["metrics"]["elapsed_seconds"])
        <= selected_at,
        "selection precedes completed training invocation",
    )
    return {"record": value, "pin": pin, "time": selected_at}


def _preservation(
    reader: legacy._Reader,
    pin: dict,
    worker_pin: dict,
    parent_pin: dict,
    worker: dict,
    chain: dict,
    event: dict,
) -> dict:
    value = reader.record(pin)
    _accepted_checks(value, PRESERVATION_CHECKS)
    _same(
        value.get("kind"),
        "qf3_learned_final_preservation_admission",
        "learned preservation admission kind differs",
    )
    selected = chain["selected"]
    for name, expected in (
        ("worker_receipt", worker_pin),
        ("controller_receipt", parent_pin),
        ("resolved_config", worker["resolved_config"]),
        ("selected_admission", selected["admission_pin"]),
        ("selected_checkpoint", selected["admission"]["admitted_checkpoint"]),
        ("final_checkpoint", worker["evidence_checkpoint"]),
    ):
        _match_pin(value[name], expected, f"preservation {name} differs")
    _same(
        value["final_checkpoint"].get("kind"),
        "qf3_evidence_only_checkpoint",
        "final checkpoint role differs",
    )
    _pin(reader, value["final_checkpoint"])
    for name, expected in (
        ("source_sha256", PROFILE),
        ("worker_revision", WORKER),
        ("checkpoint_schema", 4),
        ("base_id", worker["base_id"]),
        ("protocol_id", PROTOCOL),
        ("continuation_id", chain["continuation_id"]),
    ):
        _same(value.get(name), expected, f"preservation {name} differs")
    expected_states = {
        "learner": selected["admission"]["learner_state_sha256"],
        "replay": selected["admission"]["canonical_replay_sha256"],
        "bridge": selected["admission"]["policy_state_sha256"],
        "training_rng": selected["admission"]["training_rng_sha256"],
    }
    _same(
        value.get("before_state_sha256"),
        expected_states,
        "saved selected state audit differs",
    )
    _same(
        value.get("after_state_sha256"),
        expected_states,
        "saved final evaluation state changed",
    )
    _same(
        value.get("exposed_state_sha256"),
        event["summary"]["training_state_before"],
        "evaluation preservation audit hash differs",
    )
    _same(
        value.get("exposed_state_sha256"),
        selected["admission"]["evaluation_preservation_state_sha256"],
        "final preserved state differs from selected learner/replay/training-RNG domain",
    )
    _same(
        value.get("replay_blocks"),
        selected["admission"]["replay_blocks"],
        "final replay block selectors/pins changed",
    )
    for block in value["replay_blocks"]:
        _pin(reader, block["artifact"])
    _same(
        value.get("native_world_restore_claimed"),
        False,
        "unproven native world restore claim",
    )
    return pin


def _evaluation(
    reader: legacy._Reader, entry: dict, chain: dict, index: int, head: str
) -> dict:
    worker = reader.record(entry["worker"])
    config = _config(reader, worker, index, head=head)
    _input_preflight(reader, entry["input_preflight"], worker, config)
    _same(config.get("stage"), "evaluate", "new standalone final invocation required")
    _same(worker.get("status"), "completed", "final evaluation incomplete")
    _runtime(reader, worker, config)
    _same(
        worker["runtime_identity"],
        chain["runtime_identity"],
        "final runtime/base identity differs from training",
    )
    parent = _parent(reader, entry["parent"], entry["worker"], worker, config)
    trained_config = chain["config"]
    if head == "learned":
        _same(
            _continuation(config),
            chain["continuation_id"],
            "learned final continuation differs",
        )
        _match_pin(
            config["resume"],
            chain["selected"]["admission"]["admitted_checkpoint"],
            "learned final resume not selected checkpoint",
        )
        _same(
            worker.get("restore_verified"),
            True,
            "learned final checkpoint restore unverified",
        )
        expected_counts = chain["selected"]["admission"]["counters"]
        expected_steps, expected_warmup, expected_k = (
            chain["training_steps"],
            chain["warmup_steps"],
            chain["outer_iterations"],
        )
    else:
        _same(config.get("resume"), None, "frozen final must start fresh")
        allowed = dict(trained_config)
        allowed["evaluation"] = dict(allowed["evaluation"])
        allowed["evaluation"].pop("layout_sampling")
        allowed["evaluation"]["heads"] = ["frozen"]
        allowed["evaluation"]["cadence"] = FINAL_CADENCE
        _same(
            _continuation(config),
            _continuation(allowed),
            "frozen final has additional semantic changes",
        )
        expected_counts = {k: 0 for k in chain["selected"]["admission"]["counters"]}
        expected_steps, expected_warmup, expected_k = 0, 0, 0
    for key, value in (
        ("training_steps", expected_steps),
        ("warmup_steps", expected_warmup),
        ("outer_iterations", expected_k),
        ("actor_updates", expected_counts["actor_updates"]),
        ("critic_updates", expected_counts["critic_updates"]),
        ("target_reached", head == "learned"),
    ):
        _same(worker["metrics"].get(key), value, f"final {key} differs")
    _same(worker.get("updates"), [], "standalone final invocation trained")
    for key in (
        "warmup_steps",
        "train_steps",
        "critic_updates",
        "actor_updates",
        "stock_inference_checks",
    ):
        _same(
            worker["phase_counts"].get(key),
            0,
            "standalone final has non-evaluation work",
        )
    events = worker["invocation_evaluation_events"]
    _need(
        isinstance(events, list) and len(events) == 1,
        "exactly one new final event required; inherited summaries refused",
    )
    event = events[0]
    for key, value in (
        ("head", head),
        ("reason", "final"),
        ("outer_iterations", expected_k),
        ("training_control_steps", expected_steps),
        ("crossed_thresholds", []),
    ):
        _same(event.get(key), value, f"new final event {key} differs")
    _need("validation" not in event, "final event uses validation namespace")
    all_events = worker["evaluation_events"]
    _need(
        isinstance(all_events, list)
        and all_events
        and _hash(all_events[-1]) == _hash(event),
        "new final event absent from history",
    )
    _same(
        all_events[:-1],
        chain["selected"]["admission"]["boundary"]["evaluation_state"]["history"]
        if head == "learned"
        else [],
        "final inherited event history differs",
    )
    _same(
        worker["evaluation"], {head: event["summary"]}, "stale reported final summary"
    )
    tapes = worker["rollouts"]
    _need(
        isinstance(tapes, list) and len(tapes) == 1,
        "one newly retained final tape required",
    )
    stage = f"evaluate-{head}-{len(all_events) - 1}"
    parsed = _tape(
        reader,
        tapes[0],
        worker["base_id"],
        stage,
        50,
        namespace=stage + "-0",
        requested=FINAL_SEEDS,
    )
    selected_boundary = chain["selected"]["admission"]["boundary"]
    for key, expected in (
        (
            "evaluation_steps",
            (selected_boundary["evaluation_steps"] if head == "learned" else 0)
            + parsed["summary"]["simulation_steps"],
        ),
        (
            "physics_ticks_evaluation",
            (selected_boundary["evaluation_physics_ticks"] if head == "learned" else 0)
            + parsed["summary"]["physics_ticks"],
        ),
        (
            "physics_ticks_training",
            selected_boundary["physics_ticks"] if head == "learned" else 0,
        ),
        (
            "completed_training_episodes",
            selected_boundary["completed_episodes"] if head == "learned" else 0,
        ),
        (
            "replay_total_added",
            chain["selected"]["admission"]["replay_metadata"]["total_added"]
            if head == "learned"
            else 0,
        ),
    ):
        _same(worker["metrics"].get(key), expected, f"final cumulative {key} differs")
    _summary(
        event["summary"],
        parsed,
        noise=config["evaluation"]["sampler_seed"],
        seeds=FINAL_SEEDS,
        base_id=worker["base_id"],
    )
    _same(
        event["summary"].get("actor_updates"),
        expected_counts["actor_updates"],
        "final summary actor count differs",
    )
    _same(
        worker["phase_counts"].get("evaluation_steps"),
        parsed["summary"]["simulation_steps"],
        "new final physical counts differ",
    )
    preservation = (
        _preservation(
            reader,
            entry["preservation_admission"],
            entry["worker"],
            entry["parent"],
            worker,
            chain,
            event,
        )
        if head == "learned"
        else None
    )
    return {
        "worker": entry["worker"],
        "parent": entry["parent"],
        "new_tape": tapes[0],
        "input_preflight": entry["input_preflight"],
        "preservation_admission": preservation,
        "rows": parsed["rows"],
        "summary": parsed["summary"],
        "profile": parsed["profile"],
        "parent_time": legacy._time(parent["created_at"]),
        "parent_elapsed_seconds": parent["metrics"]["elapsed_seconds"],
    }


def _seal(
    reader: legacy._Reader, record: dict, selections: list[dict], finals: list[dict]
) -> None:
    value = reader.record(record)
    _same(value.get("schema_version"), 1, "seal schema differs")
    _same(value.get("kind"), "qf3_full_campaign_selection_seal", "seal kind differs")
    _same(value.get("status"), "accepted", "selection seal not accepted")
    _same(value.get("source_sha256"), PROFILE, "seal source profile differs")
    _same(
        value.get("selections"),
        [s["pin"] for s in selections],
        "seal selections differ",
    )
    _same(
        value.get("final_protocol"),
        {
            "seeds": FINAL_SEEDS,
            "sampler_seeds": [2**45 + 17 + i * 1000000 for i in range(3)],
            "worlds": 50,
            "encode_microbatch": 4,
            "horizon": 1000,
            "reset_options": RESET,
            "camera_height": 168,
            "camera_width": 224,
            "model_pad": 224,
            "execute_prefix": 15,
            "horizon_action": 30,
            "diffusion_steps": 5,
        },
        "seal final protocol differs",
    )
    when = legacy._time(value.get("sealed_at_utc"))
    _need(
        all(s["time"] <= when for s in selections),
        "seal precedes a checkpoint selection",
    )
    _need(
        all(when < f["parent_time"] for f in finals),
        "all three selections must be sealed before earliest of all six final invocations",
    )


def _population(values: list[float | int | None]) -> dict:
    _same(len(values), 3, "exactly three training seeds required")
    defined = [v for v in values if v is not None]
    _need(all(_finite(v) for v in defined), "invalid aggregate value")
    mean = sum(defined) / 3 if len(defined) == 3 else None
    return {
        "per_seed": values,
        "defined_seeds": len(defined),
        "mean": mean,
        "population_std": math.sqrt(sum((v - mean) ** 2 for v in defined) / 3)
        if mean is not None
        else None,
    }


def _judge(rows: list[dict], key: str) -> dict:
    lengths = [r["length"] for r in rows if r[key]]
    return {
        "successes": len(lengths),
        "success_rate": len(lengths) / len(rows),
        "successful_only_mean_control_steps": legacy._mean(lengths),
        "failure_inclusive_mean_control_steps": legacy._mean(
            [r["length"] for r in rows]
        ),
    }


def _unadmitted_attempts(
    reader: legacy._Reader, entries: list[dict], accepted_parent_hashes: set[str]
) -> list[dict]:
    """Retain controller failures/partial attempts as costs, without counter credit."""
    _need(isinstance(entries, list), "discarded invocation list missing")
    result = []
    for entry in entries:
        index = entry.get("seed_index")
        _need(_integer(index) and index < 3, "discarded invocation seed index differs")
        parent = reader.record(entry["parent"])
        checksum = entry["parent"]["sha256"]
        _need(
            checksum not in accepted_parent_hashes,
            "controller receipt duplicated in accepted/discarded invocations",
        )
        accepted_parent_hashes.add(checksum)
        _need(
            parent.get("status") in ("failed", "timed_out", "completed"),
            "discarded controller status invalid",
        )
        _need(
            _finite(parent["metrics"].get("elapsed_seconds"))
            and parent["metrics"]["elapsed_seconds"] >= 0,
            "discarded parent actual elapsed missing",
        )
        legacy._time(parent.get("created_at"))
        config = reader.record(entry["input_config"])
        _same(
            parent.get("training_config_sha256"),
            entry["input_config"]["sha256"],
            "discarded parent config differs",
        )
        _same(config.get("stage"), "train", "discarded attempt must be training")
        _same(config.get("training"), TRAINING, "discarded attempt recipe differs")
        _same(config.get("seed"), TRIPLES[index][2], "discarded seed differs")
        if entry.get("worker") is not None:
            worker = reader.record(entry["worker"])
            _same(
                parent.get("summary_sha256"),
                entry["worker"]["sha256"],
                "discarded worker pin differs",
            )
            _same(worker.get("sources"), PROFILE, "discarded worker source differs")
            for pin in worker.get("rollouts", []):
                reader.record(
                    pin
                )  # Pin and preserve incomplete evidence; admit no outcome.
        _same(
            entry.get("accepted_counter_credit"),
            0,
            "unadmitted attempt cannot supply training credit",
        )
        _need(
            isinstance(entry.get("reason"), str) and bool(entry["reason"]),
            "discarded reason missing",
        )
        result.append(
            {
                "seed_index": index,
                "parent": entry["parent"],
                "worker": entry.get("worker"),
                "input_config": entry["input_config"],
                "reason": entry["reason"],
                "accepted_counter_credit": 0,
                "parent_elapsed_seconds": parent["metrics"]["elapsed_seconds"],
            }
        )
    return result


@legacy._bounded
def compare(request: Artifact) -> dict:
    """Validate an externally selected three-seed campaign. Never select outcomes."""
    reader = _Reader()
    value = reader.json(request)
    _same(value.get("schema_version"), 1, "request schema differs")
    _same(
        value.get("kind"),
        "qf3_full400k_three_seed_comparison_request",
        "request kind differs",
    )
    _release(reader, value)
    entries = value["seeds"]
    _need(
        isinstance(entries, list) and len(entries) == 3,
        "exactly three ordered training seeds required",
    )
    results, selections, finals, runtimes = [], [], [], []
    for index, entry in enumerate(entries):
        _same(entry.get("index"), index, "seed order differs")
        chain = _chain(reader, entry["training_chain"], index)
        selection = _selection(reader, entry["selection"], chain)
        frozen = _evaluation(reader, entry["frozen_final"], chain, index, "frozen")
        learned = _evaluation(reader, entry["learned_final"], chain, index, "learned")
        _same(frozen["profile"], learned["profile"], "paired compiled controls differ")
        _same(
            [r["actual_seed"] for r in frozen["rows"]],
            [r["actual_seed"] for r in learned["rows"]],
            "paired actual fallback seeds differ",
        )
        selections.append(selection)
        finals.extend((frozen, learned))
        runtime = dict(chain["runtime_identity"])
        for key in ("seed", "lora_config"):
            runtime.pop(key)
        runtimes.append(runtime)
        judges = {}
        for key in ("history_success", "native_current_success"):
            left, right = _judge(frozen["rows"], key), _judge(learned["rows"], key)
            judges[key] = {
                "frozen": left,
                "learned": right,
                "paired": legacy._paired(frozen["rows"], learned["rows"], key),
                "learned_minus_frozen_success_rate": right["success_rate"]
                - left["success_rate"],
            }
        results.append(
            {
                "index": index,
                "seed_triple_base_lora_worker": list(TRIPLES[index]),
                "selection": selection["pin"],
                "judges": judges,
                "training_steps_post_warmup": chain["training_steps"],
                "warmup_steps": chain["warmup_steps"],
                "warmup_plus_training_steps": chain["warmup_plus_training_steps"],
                "outer_iterations": chain["outer_iterations"],
                "accepted_fresh_events": chain["accepted_fresh_events"],
                "training_invocations": chain["invocations"],
                "final_events": [
                    {
                        k: v
                        for k, v in f.items()
                        if k not in ("rows", "parent_time", "profile")
                    }
                    for f in (frozen, learned)
                ],
                "parent_elapsed_seconds": chain["parent_elapsed_seconds"]
                + frozen["parent_elapsed_seconds"]
                + learned["parent_elapsed_seconds"],
            }
        )
    _same(runtimes[0], runtimes[1], "cross-seed base/recipe/runtime differs")
    _same(runtimes[0], runtimes[2], "cross-seed base/recipe/runtime differs")
    _same(
        len({f["parent"]["sha256"] for f in finals}),
        6,
        "final controller invocation reused",
    )
    _seal(reader, value["selection_seal"], selections, finals)
    parent_hashes = [f["parent"]["sha256"] for f in finals] + [
        p["parent"]["sha256"] for r in results for p in r["training_invocations"]
    ]
    _same(
        len(set(parent_hashes)),
        len(parent_hashes),
        "controller receipt reused across seeds or stages",
    )
    attempts = _unadmitted_attempts(
        reader, value["discarded_invocations"], set(parent_hashes)
    )
    aggregates = {}
    for key in ("history_success", "native_current_success"):
        per_seed = [r["judges"][key] for r in results]
        metrics = {
            "learned_minus_frozen_success_rate": _population(
                [p["learned_minus_frozen_success_rate"] for p in per_seed]
            ),
            "common_success_mean_length_change": _population(
                [p["paired"]["common_success_mean_length_change"] for p in per_seed]
            ),
        }
        for head in ("frozen", "learned"):
            for name in (
                "success_rate",
                "successful_only_mean_control_steps",
                "failure_inclusive_mean_control_steps",
            ):
                metrics[f"{head}_{name}"] = _population(
                    [p[head][name] for p in per_seed]
                )
        aggregates[key] = metrics
    return {
        "schema_version": 1,
        "revision": REVISION,
        "domain": "sim",
        "status": "accepted_evidence_comparison",
        "worker_revision": WORKER,
        "sources": PROFILE,
        "checkpoint_schema": 4,
        "request": reader.verified[str(Path(request.path).resolve())],
        "selection_seal": value["selection_seal"],
        "per_seed": results,
        "three_seed_population_statistics": aggregates,
        "parent_elapsed_seconds": sum(r["parent_elapsed_seconds"] for r in results)
        + sum(a["parent_elapsed_seconds"] for a in attempts),
        "unadmitted_invocations": attempts,
        "verified_artifacts": list(reader.verified.values()),
        "scope": {
            "exact_paper_reproduction": False,
            "paper_initializer_confirmed": False,
            "tensor_or_replay_semantics_verified_by_reader": False,
            "checkpoint_and_label_semantics": "separately pinned independent admissions",
            "actual_gaussian_noise_regenerated": False,
            "final_geometry_equality_verified": False,
            "access_isolation_proved_by_seal": False,
            "native_world_restore_verified": False,
            "400k_accounting": "declared additional post-warmup controls; author warmup inclusion unresolved",
            "native_judge_length": "native-current outcome at history-defined episode end",
            "variability": "three training-seed population std; no pooled150-trial inference",
            "numerical_repeatability_guaranteed": False,
        },
    }


def write_output(path: Path, value: dict) -> None:
    """Publish once, atomically; preserve any prior report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".qf3-comparison-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(
                (
                    json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ).encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(Artifact(args.request, args.request_sha256))
        write_output(Path(args.output), result)
    except (ComparisonError, OSError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {"status": "accepted_evidence_comparison", "output": args.output},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
