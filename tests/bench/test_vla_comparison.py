"""Synthetic public-schema doubles only; no learned benchmark or native execution."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from abc_bench import vla_comparison as vc


def write(path, value, *, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if binary else json.dumps(value, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def artifact(record):
    return vc.Artifact.from_record(record)


def tape(head, *, lengths=None, successes=None, native=None, event_index=0):
    lengths = lengths or [1] * 50
    successes = successes or [True] * 50
    native = native or successes
    stage = f"evaluate-{head}-{event_index}"
    worlds, decisions = [], []
    for world, (length, success, stock) in enumerate(
        zip(lengths, successes, native, strict=True)
    ):
        events = []
        for step in range(1, length + 1):
            end = success and step == length
            events.append(
                {
                    "step": step,
                    "ever_placed": [end] * 5,
                    "new_objects": [f"bottle_{x}" for x in range(1, 6)] if end else [],
                    "reward": float(5 if end else 0),
                    "paper_success": end,
                    "native_success": stock and step == length,
                }
            )
        for index, start in enumerate(range(0, length, 15)):
            chunk = events[start : start + 15]
            decisions.append(
                {
                    "requested_seed": vc.SEEDS[world],
                    "actual_seed": vc.SEEDS[world],
                    "decision_index": index,
                    "episode_id": f"{stage}-0-world-{world}",
                    "events": chunk,
                    "issued_normalized": [[0.0] * 14 for _ in chunk],
                    "next_physical_state": [0.0] * 14,
                }
            )
        worlds.append(
            {
                "world": world,
                "requested_seed": vc.SEEDS[world],
                "actual_seed": vc.SEEDS[world],
                "completed": True,
                "steps": length,
                "paper_success": success,
                "native_success": stock,
                "reward": float(5 if success else 0),
                "decisions": (length + 14) // 15,
                "tracker": {
                    "schema_version": 1,
                    "protocol_revision": vc.PROTOCOL,
                    "task_id": vc.TASK,
                    "done": True,
                    "steps": length,
                    "object_names": [f"bottle_{x}" for x in range(1, 6)],
                    "initial_mask": [False] * 5,
                    "ever_placed": [success] * 5,
                },
            }
        )
    return {
        "schema_version": 1,
        "domain": "sim",
        "protocol_id": vc.PROTOCOL,
        "stage": stage,
        "rollout": {
            "worlds": worlds,
            "decisions": decisions,
            "partial": False,
            "physical_tape": [
                [[0.0] * 14 for _ in range(50)] for _ in range(max(lengths))
            ],
            "physics_ticks": max(lengths),
            "simulation_steps": sum(lengths),
            "actualsteps": sum(lengths),
            "episodes": 50,
            "successes": sum(successes),
            "action_projection": "none added; fixture commands",
            "compiled_command_profile": {
                "available": True,
                "action_modified": False,
                "revision": "native_abc_affine/1:" + "1" * 64,
            },
        },
    }


@pytest.fixture
def evidence(tmp_path):
    """Make tiny dummy files and explicit synthetic learned-schema fixtures."""
    root = tmp_path / "native"
    public = write(
        root / "abc_minimal/vla.py", b"# fixture, never imported\n", binary=True
    )
    asset = write(root / "abc_sim/models/tiny.xml", b"fixture bytes", binary=True)
    sources = write(
        tmp_path / "sources.json",
        {"schema_version": 1, "files": [{**public, "path": "abc_minimal/vla.py"}]},
    )
    assets = write(
        tmp_path / "assets.json",
        {"schema_version": 1, "files": [{**asset, "path": "abc_sim/models/tiny.xml"}]},
    )
    base = {
        "task_id": vc.TASK,
        "seed": 901,
        "upstream_revision": "fixture-public-revision",
    }
    base_hashes = {}
    for name in (
        "checkpoint",
        "metadata",
        "norm_stats",
        "prompt_metadata",
        "tokenizer",
    ):
        item = write(
            tmp_path / f"{name}.bin", f"synthetic {name}".encode(), binary=True
        )
        base[f"{name}_path"] = item["path"]
        base_hashes[name] = item["sha256"]
    lora = {"rank": 4, "scale": 1.0, "init_std": 0.02, "seed": 902}
    identity = {
        "evidence_mode": "native",
        "task_id": vc.TASK,
        "seed": 901,
        "upstream_revision": base["upstream_revision"],
        "horizon": 30,
        "execute_prefix": 15,
        "diffusion_steps": 5,
        "lora_config": lora,
        "binding_source": vc.COMMON_SOURCES["nrh.qf3_abc_vla"],
        "core_source": vc.COMMON_SOURCES["nrh.qf3"],
        "artifacts": base_hashes,
        "sources": {"abc_minimal.vla": public["sha256"]},
        "native_inventory": {"sources": sources["sha256"], "assets": assets["sha256"]},
        "statistics": "2" * 64,
        "prompt": "fixture only",
        "versions": {"runtime": "synthetic-no-import"},
    }
    config = {
        "schema_version": 1,
        "stage": "evaluate",
        "seed": 903,
        "base": base,
        "lora": lora,
        "source_artifacts": {"abc_minimal.vla": public},
        "native_inventory": {"sources": sources, "assets": assets},
        "reset_options": {
            "randomization": {
                "bottle_count": 5,
                "randomize_scales": False,
                "randomize_variants": False,
            }
        },
        "resume": None,
        "replay_capacity": 50000,
        "learner": deepcopy(vc.CAPACITY_LEARNER),
        "training": {
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
        },
        "evaluation": {
            "heads": ["frozen"],
            "seeds": vc.SEEDS,
            "sampler_seed": 91001,
            "worlds": 50,
            "max_control_steps_per_world": None,
            "cadence": {
                "unit": "outer_iterations",
                "every": 1,
                "phase": "after_outer_updates",
            },
            "layout_sampling": {
                "mode": "fresh_per_event",
                "seed_start": 2**40,
                "seed_stride": 200001,
            },
        },
    }
    base_id = vc._hash(identity)
    selected = write(
        tmp_path / "capacity/checkpoint-complete-000006.pt",
        b"synthetic selected checkpoint",
        binary=True,
    )
    selected["kind"] = "qf3_completed_boundary_checkpoint"
    manifest = {
        "schema_version": 1,
        "domain": "sim",
        "revision": "sadhana.qf3-fresh-validation/1",
        "base_id": base_id,
        "source_sha256": vc._hash(vc.SOURCE_PROFILES["learned"]),
        "policy_state_sha256": "3" * 64,
        "actor_updates": 200,
        "outer_iterations": 1,
        "cursor": 0,
        "requested_seeds": [2**40 + j * 200001 for j in range(50)],
        "noise_seed": 91002,
        "head": "learned",
        "protocol_id": vc.PROTOCOL,
        "reset_options": config["reset_options"],
        "training_control_steps": 16,
    }
    manifest_pin = write(tmp_path / "capacity/validation.json", manifest)
    reset_pin = write(
        tmp_path / "capacity/reset.json",
        {
            "schema_version": 1,
            "domain": "sim",
            "start": 0,
            "revision": "sadhana.qf3-fresh-validation/1",
            "manifest_sha256": manifest_pin["sha256"],
            "worlds": [
                {"requested_seed": s, "actual_seed": s}
                for s in manifest["requested_seeds"]
            ],
        },
    )
    capacity_config = deepcopy(config)
    capacity_config["stage"] = "train"
    capacity_config["evaluation"]["heads"] = ["learned"]
    capacity_rollouts = []
    for index, (stage, count) in enumerate(
        [("warmup", 80)] * 4 + [("train", 16), ("evaluate-learned-0", 50)]
    ):
        capacity_rollouts.append(
            write(
                tmp_path / f"capacity/tape-{index}.json",
                {
                    "schema_version": 1,
                    "domain": "sim",
                    "base_id": base_id,
                    "protocol_id": vc.PROTOCOL,
                    "stage": stage,
                    "rollout": {
                        "partial": False,
                        "worlds": [
                            {"world": i, "completed": True, "steps": 1}
                            for i in range(count)
                        ],
                        "episodes": count,
                        "simulation_steps": count,
                        "physics_ticks": 1,
                    },
                },
            )
        )
    fresh_tape = tape("learned")
    fresh_tape["base_id"] = base_id
    fresh_tape["rollout"]["reset_options"] = config["reset_options"]
    for row in fresh_tape["rollout"]["worlds"]:
        row["requested_seed"] = row["actual_seed"] = manifest["requested_seeds"][
            row["world"]
        ]
    for decision in fresh_tape["rollout"]["decisions"]:
        index = vc.SEEDS.index(decision["requested_seed"])
        decision["requested_seed"] = decision["actual_seed"] = manifest[
            "requested_seeds"
        ][index]
    capacity_rollouts[-1] = write(tmp_path / "capacity/tape-5.json", fresh_tape)
    _, fresh_summary = vc._outcomes(
        fresh_tape, "learned", requested_seeds=manifest["requested_seeds"]
    )
    fresh_summary.update(
        partial=False,
        training_state_before="7" * 64,
        training_state_after="7" * 64,
        actor_updates=200,
        attempted_episodes=50,
        requested_episodes=50,
        partial_episode_lengths=[],
        sampler_seed=91002,
        configured_seed_sha256=vc._hash(manifest["requested_seeds"]),
        base_id=base_id,
    )
    periodic = {
        "head": "learned",
        "reason": "periodic",
        "outer_iterations": 1,
        "summary": fresh_summary,
        "validation": {"manifest": manifest_pin, "reset_evidence": [reset_pin]},
    }
    capacity = {
        "schema_version": 1,
        "domain": "sim",
        "stage": "train",
        "status": "completed",
        "sources": vc.SOURCE_PROFILES["learned"],
        "runtime_identity": identity,
        "camera_capture_configuration": {"camera_height": 168, "camera_width": 224},
        "input_config": write(tmp_path / "capacity/input.json", capacity_config),
        "resolved_config_sha256": vc._hash(capacity_config),
        "completed_checkpoints": [selected],
        "base_id": base_id,
        "latest_complete": selected,
        "metrics": {
            "outer_iterations": 1,
            "actor_updates": 200,
            "critic_updates": 1600,
            "warmup_steps": 320,
            "training_steps": 16,
            "evaluation_steps": 50,
        },
        "wrapper": {
            "warmup_batches": 4,
            "completed_episodes": 336,
            "boundary": "completed_outer",
            "partial": False,
            "evaluation_state": {"pending_validation": None},
        },
        "evaluation_events": [periodic],
        "resolved_config": write(tmp_path / "capacity/resolved.json", capacity_config),
        "rollouts": capacity_rollouts,
    }
    cap_pin = write(tmp_path / "capacity/receipt.json", capacity)
    selection = {
        "schema_version": 1,
        "kind": "qf3_development_checkpoint_selection",
        "status": "accepted",
        "independent_review": True,
        "selection_rule": vc.SELECTION_RULE,
        "selected_at_utc": "2026-10-07T23:00:00+00:00",
        "checkpoint": selected,
        "capacity_receipt": cap_pin,
        "capacity_resolved_config": capacity["resolved_config"],
        "source_sha256": vc.SOURCE_PROFILES["learned"],
        "base_id": base_id,
        "policy_state_sha256": "3" * 64,
        "validation_manifest": manifest_pin,
        "actor_updates": 200,
        "critic_updates": 1600,
    }
    selection_pin = write(tmp_path / "selection.json", selection)
    generated = {
        "root": tmp_path,
        "identity": identity,
        "config": config,
        "selection": selection,
        "selection_pin": selection_pin,
        "capacity": capacity,
        "capacity_config": capacity_config,
        "manifest": manifest,
        "manifest_pin": manifest_pin,
        "base_id": base_id,
    }
    for head in ("frozen", "learned"):
        child_config = deepcopy(config)
        child_config["evaluation"]["heads"] = [head]
        if head == "learned":
            child_config["resume"] = selected
        raw_tape = tape(head, event_index=0 if head == "frozen" else 1)
        raw_tape["base_id"] = base_id
        raw_tape["rollout"]["reset_options"] = config["reset_options"]
        _, summary = vc._outcomes(raw_tape, head)
        summary.pop("native_current_success_only_mean_control_steps")
        summary.pop("reward_sum")
        summary.update(
            {
                "actor_updates": 0 if head == "frozen" else 200,
                "attempted_episodes": 50,
                "requested_episodes": 50,
                "partial": False,
                "partial_episode_lengths": [],
                "sampler_seed": 91001,
                "configured_seed_sha256": vc._hash(vc.SEEDS),
                "base_id": base_id,
                "training_state_before": "4" * 64,
                "training_state_after": "4" * 64,
            }
        )
        event = {
            "head": head,
            "reason": "baseline" if head == "frozen" else "final",
            "outer_iterations": 0 if head == "frozen" else 1,
            "summary": summary,
        }
        folder = tmp_path / head
        worker = {
            "schema_version": 1,
            "domain": "sim",
            "stage": "evaluate",
            "status": "completed",
            "sources": vc.SOURCE_PROFILES[head],
            "worker_revision": "sadhana.qf3-native-worker/2"
            if head == "frozen"
            else "sadhana.qf3-native-worker/4",
            "base_id": base_id,
            "runtime_identity": identity,
            "camera_capture_configuration": {"camera_height": 168, "camera_width": 224},
            "metrics": {
                "actor_updates": summary["actor_updates"],
                "critic_updates": 0 if head == "frozen" else 1600,
            },
            "phase_counts": {
                "evaluation_steps": summary["simulation_steps"],
                "warmup_steps": 0,
                "train_steps": 0,
                "critic_updates": 0,
                "actor_updates": 0,
            },
            "evaluation": {head: summary},
            "evaluation_events": [event] if head == "frozen" else [periodic, event],
            "checkpoint": write(
                folder / "checkpoint.pt", b"synthetic after-state", binary=True
            ),
        }
        if head == "learned":
            worker["invocation_evaluation_events"] = [event]
            worker["restore_verified"] = True
        generated[head] = {
            "config": child_config,
            "tape": raw_tape,
            "worker": worker,
            "folder": folder,
        }
    repack(generated)
    return generated


def repack(data):
    for head in ("frozen", "learned"):
        side = data[head]
        worker, folder = side["worker"], side["folder"]
        worker["input_config"] = write(folder / "input.json", side["config"])
        worker["resolved_config"] = write(folder / "resolved.json", side["config"])
        worker["resolved_config_sha256"] = vc._hash(side["config"])
        worker["rollouts"] = [write(folder / "tape.json", side["tape"])]
        side["pin"] = write(folder / "receipt.json", worker)
    data["selection_pin"] = write(data["root"] / "selection.json", data["selection"])
    data["parent"] = {
        "status": "completed",
        "training_stage": "evaluate",
        "worker_status": "completed",
        "created_at": "2026-10-07T23:01:00+00:00",
        "summary_sha256": data["learned"]["pin"]["sha256"],
        "training_config_sha256": data["learned"]["worker"]["input_config"]["sha256"],
        "evaluation_heads": ["learned"],
        "evaluation_seeds": vc.SEEDS,
        "evaluation_sampler_seed": 91001,
        "evaluation_horizon_steps": 1000,
    }
    data["parent_pin"] = write(data["root"] / "parent.json", data["parent"])


def compare(data):
    return vc.compare(
        artifact(data["frozen"]["pin"]),
        artifact(data["learned"]["pin"]),
        artifact(data["selection_pin"]),
        artifact(data["parent_pin"]),
    )


def repack_capacity(data):
    capacity = data["capacity"]
    capacity["input_config"] = write(
        data["root"] / "capacity/input.json", data["capacity_config"]
    )
    capacity["resolved_config"] = write(
        data["root"] / "capacity/resolved.json", data["capacity_config"]
    )
    capacity["resolved_config_sha256"] = vc._hash(data["capacity_config"])
    data["selection"]["capacity_resolved_config"] = capacity["resolved_config"]
    data["selection"]["capacity_receipt"] = write(
        data["root"] / "capacity/receipt.json", capacity
    )
    repack(data)


def test_synthetic_positive_and_common_success_lengths(evidence):
    report = compare(evidence)
    assert report["history_success_pairing"]["counts"] == {
        "win": 0,
        "loss": 0,
        "both": 50,
        "neither": 0,
    }
    assert report["history_success_pairing"]["common_success_mean_length_change"] == 0
    assert "Full bridge state" in report["selection"]["policy_state_scope"]
    assert set(report["source_version_differences"]) == {
        "nrh.qf3_training",
        "nrh.qf3_rollouts",
    }


def test_baseline_only_requires_no_selection(evidence):
    Path(evidence["selection_pin"]["path"]).unlink()
    report = vc.validate_baseline(artifact(evidence["frozen"]["pin"]))
    assert report["comparison_executed"] is False and report["learned_result"] is None
    assert report["baseline"]["summary"]["simulation_steps"] == 50


def test_changed_success_subsets_keep_common_lengths_separate(evidence):
    patterns = {
        "frozen": (
            [1000, 1, 1, 1000] + [1] * 46,
            [False, True, True, False] + [True] * 46,
        ),
        "learned": (
            [2, 1000, 3, 1000] + [1] * 46,
            [True, False, True, False] + [True] * 46,
        ),
    }
    for head, (lengths, successes) in patterns.items():
        side = evidence[head]
        new = tape(
            head,
            lengths=lengths,
            successes=successes,
            event_index=0 if head == "frozen" else 1,
        )
        new["base_id"] = evidence["base_id"]
        new["rollout"]["reset_options"] = side["config"]["reset_options"]
        _, observed = vc._outcomes(new, head)
        summary = side["worker"]["evaluation"][head]
        summary.update(
            {
                key: value
                for key, value in observed.items()
                if key
                not in ("reward_sum", "native_current_success_only_mean_control_steps")
            }
        )
        side["worker"]["phase_counts"]["evaluation_steps"] = sum(lengths)
        side["tape"] = new
    repack(evidence)
    report = compare(evidence)
    pairing = report["history_success_pairing"]
    assert pairing["counts"] == {"win": 1, "loss": 1, "both": 47, "neither": 1}
    assert pairing["common_success_mean_length_change"] == 2 / 47
    assert report["baseline"]["summary"]["mean_successful_episode_length"] == 1
    assert report["learned"]["summary"]["mean_successful_episode_length"] == 51 / 48
    assert (
        report["learned"]["summary"]["failure_inclusive_mean_control_steps"]
        == 2051 / 50
    )


def test_no_common_success_returns_null_length_mean():
    left = [
        {"seed": 1, "length": 10, "history_success": True},
        {"seed": 2, "length": 1000, "history_success": False},
    ]
    right = [
        {"seed": 1, "length": 1000, "history_success": False},
        {"seed": 2, "length": 20, "history_success": True},
    ]
    report = vc._paired(left, right, "history_success")
    assert report["counts"] == {"win": 1, "loss": 1, "both": 0, "neither": 0}
    assert report["common_success_length_changes"] == []
    assert report["common_success_mean_length_change"] is None


def test_nominal_target_budget_stop_is_only_accepted_at_complete_capacity_boundary(
    evidence,
):
    evidence["capacity"]["status"] = "budget_stopped"
    repack_capacity(evidence)
    assert compare(evidence)["status"] == "descriptive_fixed_development_comparison"
    evidence["capacity"]["wrapper"]["partial"] = True
    repack_capacity(evidence)
    with pytest.raises(vc.ComparisonError, match="boundary incomplete"):
        compare(evidence)


def test_fresh_fallback_seed_slot_requires_matching_tape_and_reset(evidence):
    cap = evidence["capacity"]
    reset_record = cap["evaluation_events"][0]["validation"]["reset_evidence"][0]
    reset = json.loads(Path(reset_record["path"]).read_text())
    reset["worlds"][0]["actual_seed"] += 100000
    cap["evaluation_events"][0]["validation"]["reset_evidence"][0] = write(
        Path(reset_record["path"]), reset
    )
    tape_record = cap["rollouts"][-1]
    fresh = json.loads(Path(tape_record["path"]).read_text())
    fresh["rollout"]["worlds"][0]["actual_seed"] += 100000
    fresh["rollout"]["decisions"][0]["actual_seed"] += 100000
    cap["rollouts"][-1] = write(Path(tape_record["path"]), fresh)
    repack_capacity(evidence)
    assert compare(evidence)["selection"]["policy_state_sha256"] == "3" * 64
    fresh["rollout"]["decisions"][0]["actual_seed"] += 1
    cap["rollouts"][-1] = write(Path(tape_record["path"]), fresh)
    repack_capacity(evidence)
    with pytest.raises(vc.ComparisonError, match="decision actual seed"):
        compare(evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("critic_batch_size", 8),
        ("actor_batch_size", 8),
        ("warmup_episodes_per_world", 1),
        ("train_worlds", 1),
        ("critic_target_every_updates", 10),
        ("max_control_steps_per_world", 15),
    ],
)
def test_reduced_capacity_recipe_cannot_match_itself(evidence, field, value):
    evidence["capacity_config"]["training"][field] = value
    evidence["learned"]["config"]["training"][field] = value
    repack_capacity(evidence)
    with pytest.raises(vc.ComparisonError, match="capacity recipe"):
        compare(evidence)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("created_at", None, "time missing"),
        ("created_at", "2026-10-07T23:01:00", "timezone"),
        ("summary_sha256", "f" * 64, "worker receipt pin"),
        ("training_config_sha256", "f" * 64, "requested input pin"),
        ("evaluation_sampler_seed", 901, "evaluation_sampler_seed"),
        ("evaluation_heads", ["frozen"], "evaluation_heads"),
    ],
)
def test_parent_provenance_must_bind_new_learned_invocation(
    evidence, field, value, match
):
    evidence["parent"][field] = value
    evidence["parent_pin"] = write(evidence["root"] / "parent.json", evidence["parent"])
    with pytest.raises(vc.ComparisonError, match=match):
        compare(evidence)


def test_json_hash_and_parse_use_one_snapshot(tmp_path, monkeypatch):
    record = write(tmp_path / "record.json", {"value": "verified snapshot"})
    original = Path.read_bytes

    def captured_then_changed(path):
        raw = original(path)
        path.write_text('{"value":"different later bytes"}')
        return raw

    monkeypatch.setattr(Path, "read_bytes", captured_then_changed)
    assert vc._Reader().record(record) == {"value": "verified snapshot"}


def test_cached_binary_pin_rejects_same_size_mutation(tmp_path):
    record = write(tmp_path / "binary", b"first", binary=True)
    reader = vc._Reader()
    reader.pin(artifact(record))
    Path(record["path"]).write_bytes(b"other")
    with pytest.raises(vc.ComparisonError, match="SHA-256"):
        reader.pin(artifact(record))


@pytest.mark.parametrize(
    "raw", [b'{"value":1,"value":2}', b'{"value":NaN}', b'{"value":Infinity}']
)
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, raw):
    record = write(tmp_path / "record.json", raw, binary=True)
    with pytest.raises(vc.ComparisonError, match="duplicate JSON|non-finite JSON"):
        vc._Reader().record(record)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda x: x["worker"].update(status="budget_stopped"), "complete standalone"),
        (lambda x: x["worker"].update(domain="fixture"), "native worker"),
        (lambda x: x["worker"].update(schema_version=True), "schema"),
        (
            lambda x: x["worker"].update(invocation_evaluation_events=[]),
            "invocation event",
        ),
        (
            lambda x: x["worker"]["invocation_evaluation_events"][0].update(
                reason="periodic"
            ),
            "stale/periodic",
        ),
        (
            lambda x: x["worker"]["phase_counts"].update(evaluation_steps=0),
            "invocation steps",
        ),
        (
            lambda x: x["worker"]["phase_counts"].update(actor_updates=1),
            "modified training",
        ),
        (
            lambda x: x["worker"]["evaluation"]["learned"].update(
                training_state_after="5" * 64
            ),
            "state changed",
        ),
        (
            lambda x: x["worker"]["evaluation"]["learned"].update(actor_updates=0),
            "actor update",
        ),
        (
            lambda x: x["worker"]["evaluation"]["learned"].update(successes=49),
            "aggregate successes",
        ),
        (lambda x: x["worker"].update(runtime_identity=[]), "malformed"),
        (
            lambda x: x["config"]["evaluation"].update(seeds=[1000000] * 50),
            "seed order",
        ),
        (lambda x: x["config"]["evaluation"].update(sampler_seed=91002), "noise seed"),
        (lambda x: x["config"]["evaluation"].update(worlds=True), "grouping"),
        (lambda x: x["config"]["training"].update(encode_microbatch=5), "microbatch"),
        (
            lambda x: x["worker"]["camera_capture_configuration"].update(
                camera_height=224
            ),
            "capture",
        ),
        (lambda x: x["tape"]["rollout"].update(partial=True), "partial"),
        (lambda x: x["tape"]["rollout"]["worlds"].pop(), "50 world"),
        (
            lambda x: x["tape"]["rollout"]["worlds"][1].update(actual_seed=1000000),
            "actual world",
        ),
        (
            lambda x: x["tape"]["rollout"]["worlds"][0].update(native_success=False),
            "native-current",
        ),
        (
            lambda x: x["tape"]["rollout"]["decisions"][0]["events"][0].update(
                reward=0
            ),
            "reward",
        ),
        (
            lambda x: x["tape"]["rollout"]["decisions"][0]["events"][0].update(
                new_objects=[]
            ),
            "first-placement",
        ),
        (
            lambda x: x["tape"]["rollout"]["decisions"][0].update(
                episode_id="legacy-pilot"
            ),
            "episode/world",
        ),
        (lambda x: x["tape"]["rollout"].update(physical_tape=[]), "physical tape"),
    ],
)
def test_reject_wrong_or_inherited_learned_evidence(evidence, mutation, match):
    mutation(evidence["learned"])
    repack(evidence)
    with pytest.raises(vc.ComparisonError, match=match):
        compare(evidence)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("selected_at_utc", "2026-10-07T23:02:00+00:00", "before invocation"),
        ("selected_at_utc", "2026-10-07T22:00:00", "timezone"),
        ("independent_review", False, "independent"),
        ("policy_state_sha256", "5" * 64, "policy fingerprint"),
        ("actor_updates", True, "actor update"),
    ],
)
def test_reject_invalid_selection(evidence, field, value, match):
    evidence["selection"][field] = value
    repack(evidence)
    with pytest.raises(vc.ComparisonError, match=match):
        compare(evidence)


def test_reject_smaller_matching_capacity_recipe(evidence):
    for config in (evidence["capacity_config"], evidence["learned"]["config"]):
        config["training"]["warmup_worlds"] = 1
    evidence["capacity"]["resolved_config"] = write(
        evidence["root"] / "capacity/resolved.json", evidence["capacity_config"]
    )
    evidence["capacity"]["input_config"] = write(
        evidence["root"] / "capacity/input.json", evidence["capacity_config"]
    )
    evidence["capacity"]["resolved_config_sha256"] = vc._hash(
        evidence["capacity_config"]
    )
    evidence["selection"]["capacity_resolved_config"] = evidence["capacity"][
        "resolved_config"
    ]
    evidence["selection"]["capacity_receipt"] = write(
        evidence["root"] / "capacity/receipt.json", evidence["capacity"]
    )
    repack(evidence)
    with pytest.raises(vc.ComparisonError, match="capacity recipe warmup_worlds"):
        compare(evidence)


@pytest.mark.parametrize(
    "target", ["tape", "checkpoint", "capacity_reset", "base_weights", "public_source"]
)
def test_reject_mutated_referenced_bytes(evidence, target):
    paths = {
        "tape": evidence["learned"]["worker"]["rollouts"][0]["path"],
        "checkpoint": evidence["selection"]["checkpoint"]["path"],
        "capacity_reset": evidence["capacity"]["evaluation_events"][0]["validation"][
            "reset_evidence"
        ][0]["path"],
        "base_weights": evidence["config"]["base"]["checkpoint_path"],
        "public_source": evidence["config"]["source_artifacts"]["abc_minimal.vla"][
            "path"
        ],
    }
    Path(paths[target]).write_bytes(b"mutated bytes")
    with pytest.raises(vc.ComparisonError, match="SHA-256|size"):
        compare(evidence)


def test_invalid_public_artifact_size(evidence):
    record = evidence["frozen"]["pin"]
    with pytest.raises(vc.ComparisonError, match="byte size"):
        vc.validate_baseline(vc.Artifact(record["path"], record["sha256"], True))


def test_no_heavy_imports_cli_failure_or_output_overwrite(evidence, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    args = [
        "validate-baseline",
        "--baseline",
        evidence["frozen"]["pin"]["path"],
        "--baseline-sha256",
        evidence["frozen"]["pin"]["sha256"],
        "--output",
        str(output),
    ]
    assert vc.main(args) == 1 and output.read_text() == "preserve"
    assert (
        vc.main(
            [
                "compare",
                "--baseline",
                evidence["frozen"]["pin"]["path"],
                "--baseline-sha256",
                evidence["frozen"]["pin"]["sha256"],
            ]
        )
        == 1
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import abc_bench.vla_comparison; assert not any(x in sys.modules for x in ('torch','numpy','nrh','abc_sim','warp','mujoco'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
