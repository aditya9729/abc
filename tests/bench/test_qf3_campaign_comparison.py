"""Synthetic CPU metadata fixtures; no native, tensor or campaign proof.

The complete orchestration fixture uses a smaller counter threshold to keep
CPU tests bounded. Production constants and the public 400k recipe stay fixed.
The tape parser itself runs on retained JSON-shaped decisions and controls.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from abc_bench import qf3_campaign_comparison as q


class Store:
    def __init__(self, root):
        self.root, self.n = root, 0
        root.mkdir(parents=True, exist_ok=True)

    def save(self, value, *, kind=None):
        self.n += 1
        path = self.root / f"fixture-{self.n:05d}"
        raw = (
            value
            if isinstance(value, bytes)
            else (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        )
        path.write_bytes(raw)
        pin = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        if kind:
            pin["kind"] = kind
        return pin

    def read(self, pin):
        return json.loads(Path(pin["path"]).read_bytes())

    def replace(self, pin, function):
        value = self.read(pin)
        function(value)
        return self.save(value, kind=pin.get("kind"))


def accepted(kind, checks):
    return {
        "schema_version": 1,
        "kind": kind,
        "status": "accepted",
        "independent_review": True,
        "checks": dict.fromkeys(checks, True),
        "fixture_scope": "synthetic CPU software fixture; no external semantic proof",
    }


def tape(base, stage, namespace, seeds, length=1, *, failures=(), native_false=()):
    worlds, rows, decisions = len(seeds), [], []
    lengths = []
    for j, seed in enumerate(seeds):
        success = j not in failures
        steps = length if success else 1000
        lengths.append(steps)
        for d, begin in enumerate(range(0, steps, 15)):
            events = []
            for step in range(begin + 1, min(begin + 15, steps) + 1):
                done = success and step == steps
                events.append(
                    {
                        "step": step,
                        "ever_placed": [done] * 5,
                        "new_objects": [f"bottle_{i}" for i in range(1, 6)]
                        if done
                        else [],
                        "reward": 5 if done else 0,
                        "paper_success": done,
                        "native_success": done and j not in native_false,
                    }
                )
            body = {
                "episode_id": f"{namespace}-world-{j}",
                "decision_index": d,
                "requested_seed": seed,
                "actual_seed": seed,
                "issued_normalized": [[0.0] * 14 for _ in events],
                "events": events,
                "next_physical_state": [0.0] * 14,
            }
            decisions.append({**body, "evidence_id": q._hash(body)})
        rows.append(
            {
                "world": j,
                "requested_seed": seed,
                "actual_seed": seed,
                "steps": steps,
                "decisions": (steps + 14) // 15,
                "completed": True,
                "paper_success": success,
                "native_success": success and j not in native_false,
                "reward": 5 if success else 0,
                "tracker": {
                    "schema_version": 1,
                    "protocol_revision": q.PROTOCOL,
                    "task_id": q.TASK,
                    "object_names": [f"bottle_{i}" for i in range(1, 6)],
                    "initial_mask": [True] * 5,
                    "steps": steps,
                    "done": True,
                    "ever_placed": [success] * 5,
                },
            }
        )
    return {
        "schema_version": 1,
        "domain": "sim",
        "protocol_id": q.PROTOCOL,
        "base_id": base,
        "stage": stage,
        "fixture_scope": "synthetic CPU metadata",
        "rollout": {
            "partial": False,
            "worlds": rows,
            "decisions": decisions,
            "episodes": worlds,
            "successes": worlds - len(failures),
            "simulation_steps": sum(lengths),
            "actualsteps": sum(lengths),
            "physics_ticks": max(lengths),
            "physical_tape": [[[0.0] * 14 for _ in seeds] for _ in range(max(lengths))],
            "reset_options": q.RESET,
            "compiled_command_profile": {
                "available": True,
                "action_modified": False,
                "revision": "native_abc_affine/1:" + "a" * 64,
            },
            "action_projection": "none added; synthetic fixture",
        },
    }


def summary(parsed, config, base, actors):
    result = {
        k: v
        for k, v in parsed["summary"].items()
        if k != "native_current_success_only_mean_control_steps"
    }
    result.update(
        attempted_episodes=50,
        requested_episodes=50,
        partial=False,
        partial_episode_lengths=[],
        sampler_seed=config["evaluation"]["sampler_seed"],
        configured_seed_sha256=q._hash(config["evaluation"]["seeds"]),
        base_id=base,
        actor_updates=actors,
        training_state_before="b" * 64,
        training_state_after="b" * 64,
    )
    return result


def checkpoint(store, base, config, boundary, total, counters, worker_pin=None):
    value = accepted("qf3_full_campaign_checkpoint_admission", q.AUDIT_CHECKS)
    value.update(
        source_sha256=q.PROFILE,
        worker_revision=q.WORKER,
        checkpoint_schema=4,
        block_revision="sadhana.qf3-replay-blocks/1",
        base_id=base,
        protocol_id=q.PROTOCOL,
        continuation_id=q._continuation(config),
        admitted_checkpoint=store.save(
            b"synthetic checkpoint bytes", kind="qf3_completed_boundary_checkpoint"
        ),
        policy_state_sha256="c" * 64,
        canonical_replay_sha256="d" * 64,
        exposed_state_sha256="2" * 64,
        evaluation_preservation_state_sha256="b" * 64,
        ordered_record_selectors_sha256="e" * 64,
        learner_state_sha256="f" * 64,
        training_rng_sha256="1" * 64,
        replay_metadata={
            "schema_version": 1,
            "contract": "sadhana.qf3-abc-proposal-issued-prefix/1",
            "capacity": 50000,
            "base_id": base,
            "protocol_id": q.PROTOCOL,
            "seed": config["seed"],
            "total_added": total,
            "size": min(total, 50000),
            "cursor": total % 50000,
        },
        replay_blocks=[
            {
                "artifact": store.save(
                    b"synthetic block bytes", kind="qf3_immutable_replay_block"
                ),
                "first_ordinal": 0,
                "record_count": total,
                "selector_count": total,
            }
        ],
        boundary=boundary,
        counters=counters,
    )
    return value


def preflight(store, worker, config):
    value = accepted(
        "qf3_full_campaign_input_preflight",
        ("actual_installed_resolve", "artifact_native_source_inventory_pins"),
    )
    value.update(
        source_sha256=q.PROFILE,
        release_wheel_sha256=q.RELEASE_SHA,
        native_install_review_sha256=q.NATIVE_INSTALL_REVIEW_SHA,
        continuation_id=q._continuation(config),
        resolved_config_sha256=q._hash(config),
        input_config=worker["input_config"],
        resolved_config=worker["resolved_config"],
    )
    return store.save(value)


def parent(store, worker_pin, worker, config, time):
    value = {
        "algorithm": "QF3 ABC-VLA " + config["stage"],
        "status": "completed",
        "worker_status": worker["status"],
        "summary_sha256": worker_pin["sha256"],
        "training_config_sha256": worker["input_config"]["sha256"],
        "training_stage": config["stage"],
        "seed": None if config["stage"] == "evaluate" else config["seed"],
        "policy_seed": config["base"]["seed"],
        "task": q.TASK,
        "created_at": time,
        "metrics": {**worker["metrics"], "elapsed_seconds": 1.0},
    }
    if config["stage"] == "evaluate":
        value.update(
            evaluation_heads=config["evaluation"]["heads"],
            evaluation_seeds=q.FINAL_SEEDS,
            evaluation_sampler_seed=config["evaluation"]["sampler_seed"],
            evaluation_horizon_steps=1000,
        )
    return store.save(value)


def runtime(config, artifacts):
    return {
        "evidence_mode": "native",
        "task_id": q.TASK,
        "seed": config["base"]["seed"],
        "upstream_revision": config["base"]["upstream_revision"],
        "lora_config": config["lora"],
        "binding_source": q.PROFILE["nrh.qf3_abc_vla"],
        "core_source": q.PROFILE["nrh.qf3"],
        "horizon": 30,
        "execute_prefix": 15,
        "diffusion_steps": 5,
        "artifacts": artifacts,
        "versions": q.VERSIONS,
        "prompt": "sim throw plastic bottles in bin",
        "checkpoint_role": "declared_shared_candidate",
        "exact_paper_initializer_verified": False,
        "sources": q.NATIVE_SOURCES,
        "native_inventory": q.NATIVE_INVENTORY,
        "statistics": "fixture-only",
    }


def worker(
    store,
    config,
    identity,
    *,
    metrics,
    rollouts,
    events,
    status="completed",
    resume=False,
):
    return {
        "schema_version": 1,
        "domain": "sim",
        "worker_revision": q.WORKER,
        "stage": config["stage"],
        "sources": q.PROFILE,
        "fixture_scope": "synthetic CPU software fixture; not a native receipt",
        "base_id": q._hash(identity),
        "runtime_identity": identity,
        "camera_capture_configuration": {
            "camera_height": 168,
            "camera_width": 224,
            "camera_gpu_id": 0,
            "source": "pinned abc_minimal.config.SimEvalConfig defaults",
        },
        "input_config": store.save(config),
        "resolved_config": store.save(config),
        "resolved_config_sha256": q._hash(config),
        "status": status,
        "restore_verified": resume,
        "rollouts": rollouts,
        "updates": [],
        "metrics": metrics,
        "evaluation_events": events,
        "invocation_evaluation_events": events,
        "evaluation": {},
        "phase_counts": {
            "warmup_steps": 0,
            "train_steps": 0,
            "evaluation_steps": 0,
            "critic_updates": 0,
            "actor_updates": 0,
            "stock_inference_checks": 0,
        },
        "completed_checkpoints": [],
    }


@pytest.fixture(scope="module")
def campaign_files(tmp_path_factory):
    """Build bounded schema fixtures with 26/28/30 complete outers."""
    store = Store(tmp_path_factory.mktemp("qf3-campaign-fixtures"))
    artifacts = {
        name: store.save(("synthetic " + name).encode()) for name in q.ARTIFACT_HASHES
    }
    root = store.root / "native"
    source_pins = {}
    for name in q.NATIVE_SOURCES:
        path = root / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# explicit synthetic public-source fixture\n")
        source_pins[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    inventory = {
        name: store.save(
            {
                "files": [
                    {
                        "path": "abc_minimal/vla.py",
                        "sha256": source_pins["abc_minimal.vla"]["sha256"],
                        "bytes": source_pins["abc_minimal.vla"]["bytes"],
                    }
                ]
            }
        )
        for name in q.NATIVE_INVENTORY
    }
    fixtures = {
        "store": store,
        "artifacts": artifacts,
        "sources": source_pins,
        "inventory": inventory,
    }
    # Module-scoped generation uses a temporary patch, restored before return.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q, "TARGET", 6080)
        patch.setattr(q, "CANDIDATE_SHA", artifacts["checkpoint"]["sha256"])
        patch.setattr(
            q, "ARTIFACT_HASHES", {k: p["sha256"] for k, p in artifacts.items()}
        )
        patch.setattr(
            q, "NATIVE_SOURCES", {k: p["sha256"] for k, p in source_pins.items()}
        )
        patch.setattr(
            q, "NATIVE_INVENTORY", {k: p["sha256"] for k, p in inventory.items()}
        )
        entries = []
        for index, length in enumerate((15, 14, 13)):
            base_seed, lora_seed, seed = q.TRIPLES[index]
            config = {
                "schema_version": 1,
                "stage": "train",
                "seed": seed,
                "replay_capacity": 50000,
                "checkpoint_storage": q.STORAGE,
                "base": {
                    "seed": base_seed,
                    "task_id": q.TASK,
                    "upstream_revision": "d0832d12651d1b260a652861a14648dc5f3660c7",
                    **{k + "_path": p["path"] for k, p in artifacts.items()},
                },
                "lora": {"rank": 4, "scale": 1.0, "init_std": 0.02, "seed": lora_seed},
                "learner": q.legacy.CAPACITY_LEARNER,
                "training": q.TRAINING,
                "reset_options": q.RESET,
                "fixed_layout_seed": None,
                "resume": None,
                "source_artifacts": source_pins,
                "native_inventory": inventory,
                "evaluation": {
                    "heads": ["learned"],
                    "seeds": q.FINAL_SEEDS,
                    "worlds": 50,
                    "max_control_steps_per_world": None,
                    "sampler_seed": 2**45 + 17 + index * 1000000,
                    "cadence": q.CADENCE,
                    "layout_sampling": {
                        "mode": "fresh_per_event",
                        "seed_start": 2**40 + index * 2**39,
                        "seed_stride": 200001,
                    },
                },
            }
            identity = runtime(config, q.ARTIFACT_HASHES)
            base = q._hash(identity)
            accepted_tapes, fresh_tapes, events, pins = [], [], [], []
            reader = q._Reader()
            for batch in range(4):
                pin = store.save(
                    tape(base, "warmup", f"warmup-{batch}", list(range(80)))
                )
                parsed = q._tape(
                    reader,
                    pin,
                    base,
                    "warmup",
                    80,
                    namespace=f"warmup-{batch}",
                    outcomes=False,
                )
                accepted_tapes.append(
                    {
                        "category": "warmup",
                        "ordinal": batch,
                        "artifact": pin,
                        "accounting": q._accounting(parsed),
                        "source_sha256": q.PROFILE,
                    }
                )
                pins.append(pin)
            k = (6080 + 16 * length - 1) // (16 * length)
            for ordinal in range(k):
                pin = store.save(
                    tape(base, "train", f"train-{ordinal}", list(range(16)), length)
                )
                parsed = q._tape(
                    reader,
                    pin,
                    base,
                    "train",
                    16,
                    namespace=f"train-{ordinal}",
                    outcomes=False,
                )
                accepted_tapes.append(
                    {
                        "category": "train",
                        "ordinal": ordinal,
                        "artifact": pin,
                        "accounting": q._accounting(parsed),
                        "source_sha256": q.PROFILE,
                    }
                )
                pins.append(pin)
                fresh_seeds = [
                    2**40 + index * 2**39 + (ordinal * 50 + j) * 200001
                    for j in range(50)
                ]
                noise = config["evaluation"]["sampler_seed"] + ordinal + 1
                rng = {
                    "bit_generator": "PCG64",
                    "state": {"state": noise, "inc": 1},
                    "has_uint32": 0,
                    "uinteger": 0,
                }
                manifest = {
                    "schema_version": 1,
                    "revision": "sadhana.qf3-fresh-validation/1",
                    "domain": "sim",
                    "continuation_id": q._continuation(config),
                    "source_sha256": q._hash(q.PROFILE),
                    "base_id": base,
                    "protocol_id": q.PROTOCOL,
                    "head": "learned",
                    "cursor": ordinal,
                    "outer_iterations": ordinal + 1,
                    "crossed_thresholds": [ordinal + 1],
                    "training_control_steps": 16 * length * (ordinal + 1),
                    "actor_updates": 200 * (ordinal + 1),
                    "requested_seeds": fresh_seeds,
                    "reset_options": q.RESET,
                    "noise_seed": noise,
                    "noise_rng": rng,
                    "policy_state_sha256": "c" * 64,
                }
                manifest_pin = store.save(manifest)
                reset_rows = []
                for s in fresh_seeds:
                    objects = {
                        name: {
                            "pos": [float(s), 0.0, 0.0],
                            "quat": [1.0, 0.0, 0.0, 0.0],
                        }
                        for name in (
                            "bin_joint",
                            *(f"bottle_{i}_joint" for i in range(1, 6)),
                        )
                    }
                    reset_rows.append(
                        {
                            "requested_seed": s,
                            "actual_seed": s,
                            "randomization": {
                                "seed": s,
                                "object_states": objects,
                                "scale_states": {},
                                "metadata": {
                                    "bottle_count": 5,
                                    "bottle_variant": "fixture",
                                    "bottle_variants": ["fixture"] * 5,
                                },
                            },
                            "geometry_sha256": q._hash(
                                {"object_states": objects, "scale_states": {}}
                            ),
                        }
                    )
                reset_pin = store.save(
                    {
                        "schema_version": 1,
                        "revision": "sadhana.qf3-fresh-validation/1",
                        "domain": "sim",
                        "manifest_sha256": manifest_pin["sha256"],
                        "start": 0,
                        "worlds": reset_rows,
                    }
                )
                stage = f"evaluate-learned-{ordinal}"
                fresh_pin = store.save(tape(base, stage, stage + "-0", fresh_seeds))
                parsed = q._tape(
                    reader,
                    fresh_pin,
                    base,
                    stage,
                    50,
                    namespace=stage + "-0",
                    requested=fresh_seeds,
                )
                temporary = {
                    "evaluation": {"sampler_seed": noise, "seeds": fresh_seeds}
                }
                event = {
                    "head": "learned",
                    "reason": "periodic",
                    "outer_iterations": ordinal + 1,
                    "training_control_steps": 16 * length * (ordinal + 1),
                    "crossed_thresholds": [ordinal + 1],
                    "summary": summary(parsed, temporary, base, 200 * (ordinal + 1)),
                    "validation": {
                        "manifest": manifest_pin,
                        "reset_evidence": [reset_pin],
                    },
                }
                events.append(event)
                fresh_tapes.append(
                    {
                        "event_index": ordinal,
                        "artifact": fresh_pin,
                        "noise_rng_sha256": q._hash(rng),
                        "policy_state_sha256": "c" * 64,
                    }
                )
                pins.append(fresh_pin)
            boundary = {
                "iteration": k,
                "warmup_batches": 4,
                "training_steps": 16 * length * k,
                "warmup_steps": 320,
                "physics_ticks": 4 + length * k,
                "completed_episodes": 320 + 16 * k,
                "evaluation_steps": 50 * k,
                "evaluation_physics_ticks": k,
                "discarded_evaluation_steps": 0,
                "discarded_evaluation_physics_ticks": 0,
                "partial": False,
                "boundary": "completed_outer",
                "evaluation_state": {
                    "pending_validation": None,
                    "baseline": None,
                    "history": events,
                    "validation_cursor": k,
                    "next_threshold": k + 1,
                },
            }
            counters = {
                "critic_updates": 1600 * k,
                "actor_updates": 200 * k,
                "critic_target_updates": 1600 * k,
                "actor_target_updates": k,
            }
            audit = checkpoint(store, base, config, boundary, 320 + 16 * k, counters)
            metrics = {
                "training_steps": boundary["training_steps"],
                "warmup_steps": 320,
                "outer_iterations": k,
                "actor_updates": 200 * k,
                "critic_updates": 1600 * k,
                "target_reached": True,
                "evaluation_steps": 50 * k,
                "physics_ticks_evaluation": k,
                "physics_ticks_training": 4 + length * k,
                "completed_training_episodes": 320 + 16 * k,
                "replay_total_added": 320 + 16 * k,
            }
            report = worker(
                store, config, identity, metrics=metrics, rollouts=pins, events=events
            )
            report["phase_counts"].update(
                warmup_steps=320,
                train_steps=boundary["training_steps"],
                evaluation_steps=50 * k,
                critic_updates=1600 * k,
                actor_updates=200 * k,
            )
            for outer in range(k):
                report["updates"].extend(
                    {
                        "kind": "critic",
                        "iteration": outer,
                        "critic_updates": 1600 * outer + j + 1,
                        "critic_loss": 0.0,
                    }
                    for j in range(1600)
                )
                report["updates"].extend(
                    {
                        "kind": "actor",
                        "iteration": outer,
                        "actor_updates": 200 * outer + j + 1,
                        "actor_loss": 0.0,
                    }
                    for j in range(200)
                )
            report["completed_checkpoints"] = [audit["admitted_checkpoint"]]
            worker_pin = store.save(report)
            parent_pin = parent(
                store, worker_pin, report, config, "2026-10-08T00:00:00+00:00"
            )
            audit.update(
                worker_receipt=worker_pin,
                controller_receipt=parent_pin,
                resolved_config=report["resolved_config"],
                predecessor_admission=None,
                accepted_tapes=accepted_tapes,
                fresh_tapes=fresh_tapes,
                discarded_tapes=[],
                covered_training_tape_sha256=[
                    p["artifact"]["sha256"] for p in accepted_tapes
                ],
                invocation_accounting={
                    "accepted": {
                        "warmup_steps": 320,
                        "training_steps": boundary["training_steps"],
                        "physics_ticks": boundary["physics_ticks"],
                        "completed_episodes": boundary["completed_episodes"],
                    },
                    "discarded": {
                        "warmup_steps": 0,
                        "training_steps": 0,
                        "evaluation_steps": 0,
                        "physics_ticks": 0,
                    },
                    "updates": {
                        "accepted": {"critic": 1600 * k, "actor": 200 * k},
                        "discarded": {"critic": 0, "actor": 0},
                    },
                    "accepted_fresh_steps": 50 * k,
                    "accepted_fresh_ticks": k,
                },
            )
            admission_pin = store.save(audit)
            chain_entry = {
                "worker": worker_pin,
                "parent": parent_pin,
                "admission": admission_pin,
                "input_preflight": preflight(store, report, config),
            }
            selection = accepted(
                "qf3_full_campaign_checkpoint_selection",
                (
                    "first_target_boundary",
                    "all_completed_boundary_lineage",
                    "fresh_validation_complete",
                ),
            )
            selection.update(
                checkpoint=audit["admitted_checkpoint"],
                training_admission=admission_pin,
                training_receipt=worker_pin,
                resolved_config=report["resolved_config"],
                base_id=base,
                protocol_id=q.PROTOCOL,
                source_sha256=q.PROFILE,
                continuation_id=q._continuation(config),
                policy_state_sha256="c" * 64,
                selected_at_utc="2026-10-08T00:01:00+00:00",
                rule={
                    "kind": "first_completed_outer_at_post_warmup_target",
                    "target_control_steps": 6080,
                    "training_steps": boundary["training_steps"],
                    "outer_iterations": k,
                    "fresh_validation_accepted": True,
                },
            )
            entry = {
                "index": index,
                "training_chain": [chain_entry],
                "selection": store.save(selection),
            }
            for head in ("frozen", "learned"):
                final_config = copy.deepcopy(config)
                final_config["stage"] = "evaluate"
                final_config["evaluation"]["heads"] = [head]
                if head == "frozen":
                    final_config["evaluation"].pop("layout_sampling")
                    final_config["evaluation"]["cadence"] = {
                        "unit": "final",
                        "every": None,
                        "phase": "after_outer_updates",
                    }
                else:
                    final_config["resume"] = audit["admitted_checkpoint"]
                inherited = events if head == "learned" else []
                stage = f"evaluate-{head}-{len(inherited)}"
                final_pin = store.save(
                    tape(
                        base,
                        stage,
                        stage + "-0",
                        q.FINAL_SEEDS,
                        length=1 if head == "frozen" else 2,
                        failures=(0,) if index == 2 else (),
                        native_false=(1, 2) if head == "learned" else (1,),
                    )
                )
                parsed = q._tape(
                    reader,
                    final_pin,
                    base,
                    stage,
                    50,
                    namespace=stage + "-0",
                    requested=q.FINAL_SEEDS,
                )
                final_summary = summary(
                    parsed, final_config, base, 200 * k if head == "learned" else 0
                )
                event = {
                    "head": head,
                    "reason": "final",
                    "training_control_steps": boundary["training_steps"]
                    if head == "learned"
                    else 0,
                    "outer_iterations": k if head == "learned" else 0,
                    "crossed_thresholds": [],
                    "summary": final_summary,
                }
                final_metrics = (
                    dict(metrics)
                    if head == "learned"
                    else {
                        "training_steps": 0,
                        "warmup_steps": 0,
                        "outer_iterations": 0,
                        "actor_updates": 0,
                        "critic_updates": 0,
                        "target_reached": False,
                        "evaluation_steps": 0,
                        "physics_ticks_evaluation": 0,
                        "physics_ticks_training": 0,
                        "completed_training_episodes": 0,
                        "replay_total_added": 0,
                    }
                )
                final_metrics["evaluation_steps"] += parsed["summary"][
                    "simulation_steps"
                ]
                final_metrics["physics_ticks_evaluation"] += parsed["summary"][
                    "physics_ticks"
                ]
                final_report = worker(
                    store,
                    final_config,
                    identity,
                    metrics=final_metrics,
                    rollouts=[final_pin],
                    events=inherited + [event],
                    resume=head == "learned",
                )
                final_report["invocation_evaluation_events"] = [event]
                final_report["evaluation"] = {head: final_summary}
                final_report["phase_counts"]["evaluation_steps"] = parsed["summary"][
                    "simulation_steps"
                ]
                final_report["evidence_checkpoint"] = store.save(
                    b"synthetic final checkpoint", kind="qf3_evidence_only_checkpoint"
                )
                final_worker_pin = store.save(final_report)
                final_parent_pin = parent(
                    store,
                    final_worker_pin,
                    final_report,
                    final_config,
                    "2026-10-08T00:03:00+00:00",
                )
                final_entry = {
                    "worker": final_worker_pin,
                    "parent": final_parent_pin,
                    "input_preflight": preflight(store, final_report, final_config),
                }
                if head == "learned":
                    preservation = accepted(
                        "qf3_learned_final_preservation_admission",
                        q.PRESERVATION_CHECKS,
                    )
                    state = {
                        "learner": "f" * 64,
                        "replay": "d" * 64,
                        "bridge": "c" * 64,
                        "training_rng": "1" * 64,
                    }
                    preservation.update(
                        worker_receipt=final_worker_pin,
                        controller_receipt=final_parent_pin,
                        resolved_config=final_report["resolved_config"],
                        selected_admission=admission_pin,
                        selected_checkpoint=audit["admitted_checkpoint"],
                        final_checkpoint=final_report["evidence_checkpoint"],
                        source_sha256=q.PROFILE,
                        worker_revision=q.WORKER,
                        checkpoint_schema=4,
                        base_id=base,
                        protocol_id=q.PROTOCOL,
                        continuation_id=q._continuation(config),
                        before_state_sha256=state,
                        after_state_sha256=state,
                        exposed_state_sha256="b" * 64,
                        replay_blocks=audit["replay_blocks"],
                        native_world_restore_claimed=False,
                    )
                    final_entry["preservation_admission"] = store.save(preservation)
                entry[head + "_final"] = final_entry
            entries.append(entry)
        seal = {
            "schema_version": 1,
            "kind": "qf3_full_campaign_selection_seal",
            "status": "accepted",
            "source_sha256": q.PROFILE,
            "selections": [e["selection"] for e in entries],
            "sealed_at_utc": "2026-10-08T00:02:00+00:00",
            "final_protocol": {
                "seeds": q.FINAL_SEEDS,
                "sampler_seeds": [2**45 + 17 + i * 1000000 for i in range(3)],
                "worlds": 50,
                "encode_microbatch": 4,
                "horizon": 1000,
                "reset_options": q.RESET,
                "camera_height": 168,
                "camera_width": 224,
                "model_pad": 224,
                "execute_prefix": 15,
                "horizon_action": 30,
                "diffusion_steps": 5,
            },
        }
        request = {
            "schema_version": 1,
            "kind": "qf3_full400k_three_seed_comparison_request",
            "seeds": entries,
            "selection_seal": store.save(seal),
            "discarded_invocations": [],
            "release_wheel": {
                "path": "/home/user/aditya/RL/builds/qf3-replay-block-runtime/nirvana_rl_harness-0.4.10-py3-none-any.whl",
                "sha256": q.RELEASE_SHA,
                "bytes": q.RELEASE_BYTES,
            },
            "independent_install_review": {
                "path": "/home/user/aditya/RL/builds/qf3-replay-block-installed-review/INDEPENDENT_INSTALLED_REVIEW.json",
                "sha256": q.INSTALL_REVIEW_SHA,
                "bytes": 7577,
            },
            "native_install_review": {
                "path": "/home/user/aditya/RL/builds/qf3-replay-block-native-preflight/INDEPENDENT_NATIVE_LAUNCH_PREFLIGHT.json",
                "sha256": q.NATIVE_INSTALL_REVIEW_SHA,
                "bytes": 6913,
            },
        }
        fixtures.update(request=request, request_pin=store.save(request))
    return fixtures


@pytest.fixture
def campaign(campaign_files, monkeypatch):
    files = campaign_files
    monkeypatch.setattr(q, "TARGET", 6080)
    monkeypatch.setattr(q, "CANDIDATE_SHA", files["artifacts"]["checkpoint"]["sha256"])
    monkeypatch.setattr(
        q, "ARTIFACT_HASHES", {k: p["sha256"] for k, p in files["artifacts"].items()}
    )
    monkeypatch.setattr(
        q, "NATIVE_SOURCES", {k: p["sha256"] for k, p in files["sources"].items()}
    )
    monkeypatch.setattr(
        q, "NATIVE_INVENTORY", {k: p["sha256"] for k, p in files["inventory"].items()}
    )
    return files


def test_three_complete_pairs_and_population_accounting(campaign):
    result = q.compare(q.Artifact.from_record(campaign["request_pin"]))
    assert [r["outer_iterations"] for r in result["per_seed"]] == [26, 28, 30]
    assert all(
        r["warmup_plus_training_steps"] == r["training_steps_post_warmup"] + 320
        for r in result["per_seed"]
    )
    assert result["parent_elapsed_seconds"] == 9.0
    assert result["scope"]["tensor_or_replay_semantics_verified_by_reader"] is False
    assert result["scope"]["actual_gaussian_noise_regenerated"] is False
    judge = result["three_seed_population_statistics"]["history_success"]
    assert judge["learned_minus_frozen_success_rate"]["population_std"] == 0.0
    assert (
        result["per_seed"][0]["judges"]["native_current_success"]["paired"]["counts"][
            "loss"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(schema_version=1.0),
        lambda x: x["seeds"][0].update(index=True),
        lambda x: x["seeds"].pop(),
        lambda x: x["release_wheel"].update(sha256="0" * 64),
        lambda x: x["independent_install_review"].update(sha256="0" * 64),
    ],
)
def test_request_release_and_seed_refusals(campaign, change):
    pin = campaign["store"].replace(campaign["request_pin"], change)
    with pytest.raises(q.ComparisonError):
        q.compare(q.Artifact.from_record(pin))


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_steps", 6079),
        ("warmup_batches", 3),
        ("iteration", 27),
        ("training_steps", 6240.0),
    ],
)
def test_boundary_counter_jumps_refuse(campaign, field, value):
    store = campaign["store"]
    entry = copy.deepcopy(campaign["request"]["seeds"][0]["training_chain"][0])
    entry["admission"] = store.replace(
        entry["admission"], lambda a: a["boundary"].update({field: value})
    )
    with pytest.raises(q.ComparisonError):
        q._chain(q._Reader(), [entry], 0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["accepted_tapes"].pop(),
        lambda a: a["accepted_tapes"][4].update(ordinal=1),
        lambda a: a["fresh_tapes"].pop(),
        lambda a: a["fresh_tapes"][0].update(noise_rng_sha256="0" * 64),
        lambda a: a["covered_training_tape_sha256"].reverse(),
        lambda a: a["checks"].update(
            causal_replay_native_dispatch_reward_lineage=False
        ),
        lambda a: a["boundary"]["evaluation_state"].update(
            pending_validation={"cursor": 0}
        ),
        lambda a: a["replay_blocks"][0].update(selector_count=0),
        lambda a: a["replay_metadata"].update(cursor=1),
    ],
)
def test_training_and_audit_lineage_refusals(campaign, mutation):
    store = campaign["store"]
    entry = copy.deepcopy(campaign["request"]["seeds"][0]["training_chain"][0])
    entry["admission"] = store.replace(entry["admission"], mutation)
    with pytest.raises(q.ComparisonError):
        q._chain(q._Reader(), [entry], 0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda w: w.update(invocation_evaluation_events=[]),
        lambda w: w["invocation_evaluation_events"][0].update(reason="periodic"),
        lambda w: w["phase_counts"].update(train_steps=1),
        lambda w: w["invocation_evaluation_events"][0]["summary"].update(partial=True),
        lambda w: w["invocation_evaluation_events"][0]["summary"].update(
            training_state_after="0" * 64
        ),
    ],
)
def test_final_refuses_stale_partial_or_training_invocations(
    campaign, monkeypatch, mutation
):
    store = campaign["store"]
    root = copy.deepcopy(campaign["request"]["seeds"][0])
    # The parent pin is deliberately rebuilt so the final-event gate is exercised.
    entry = root["frozen_final"]
    value = store.read(entry["worker"])
    mutation(value)
    entry["worker"] = store.save(value)
    p = store.read(entry["parent"])
    p["summary_sha256"] = entry["worker"]["sha256"]
    entry["parent"] = store.save(p)
    chain = q._chain(q._Reader(), root["training_chain"], 0)
    with pytest.raises(q.ComparisonError):
        q._evaluation(q._Reader(), entry, chain, 0, "frozen")


def test_initial_placements_credit_only_first_executed_step(tmp_path):
    store = Store(tmp_path)
    value = tape("a" * 64, "evaluate", "test", [7])
    pin = store.save(value)
    parsed = q._tape(q._Reader(), pin, "a" * 64, "evaluate", 1, namespace="test")
    assert parsed["rows"][0]["reward"] == 5
    assert parsed["rows"][0]["length"] == 1
    # Reset predicates must not initialize rewarded history: suppressing first
    # step reward/transitions is a malformed tape even though initial_mask=true.
    decision = value["rollout"]["decisions"][0]
    decision["events"][0].update(new_objects=[], reward=0)
    decision["evidence_id"] = q._hash(
        {k: v for k, v in decision.items() if k != "evidence_id"}
    )
    with pytest.raises(q.ComparisonError, match="first placement"):
        q._tape(
            q._Reader(), store.save(value), "a" * 64, "evaluate", 1, namespace="test"
        )


def test_empty_success_lengths_and_population_null(tmp_path):
    store = Store(tmp_path)
    parsed = q._tape(
        q._Reader(),
        store.save(tape("a" * 64, "evaluate", "test", [7], failures=(0,))),
        "a" * 64,
        "evaluate",
        1,
        namespace="test",
    )
    assert parsed["summary"]["mean_successful_episode_length"] is None
    assert parsed["summary"]["failure_inclusive_mean_control_steps"] == 1000
    assert q._population([None, 1, 2]) == {
        "per_seed": [None, 1, 2],
        "defined_seeds": 2,
        "mean": None,
        "population_std": None,
    }
    assert q._population([0, 1, 2])["population_std"] == pytest.approx((2 / 3) ** 0.5)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda t: t["rollout"]["worlds"].reverse(),
        lambda t: t["rollout"]["worlds"][0].update(actual_seed=8),
        lambda t: t["rollout"]["worlds"][0].update(steps=1.0),
        lambda t: t["rollout"]["decisions"][0].update(evidence_id="0" * 64),
        lambda t: t["rollout"]["physical_tape"][0][0].pop(),
        lambda t: t["rollout"].update(partial=True),
    ],
)
def test_tape_order_types_controls_and_pins_refuse(tmp_path, mutation):
    store = Store(tmp_path)
    value = tape("a" * 64, "evaluate", "test", [7, 9])
    mutation(value)
    with pytest.raises(q.ComparisonError):
        q._tape(
            q._Reader(), store.save(value), "a" * 64, "evaluate", 2, namespace="test"
        )


def test_seal_before_all_six_finals(campaign):
    campaign["store"]
    selections = [
        {"pin": e["selection"], "time": q.legacy._time("2026-10-08T00:01:00Z")}
        for e in campaign["request"]["seeds"]
    ]
    finals = [{"parent_time": q.legacy._time("2026-10-08T00:03:00Z")} for _ in range(6)]
    q._seal(q._Reader(), campaign["request"]["selection_seal"], selections, finals)
    finals[0]["parent_time"] = q.legacy._time("2026-10-08T00:01:59Z")
    with pytest.raises(q.ComparisonError, match="earliest"):
        q._seal(q._Reader(), campaign["request"]["selection_seal"], selections, finals)


def test_json_exact_bytes_duplicates_mutation_and_atomic_output(tmp_path):
    store = Store(tmp_path)
    pin = store.save(b'{"schema_version":1,"schema_version":2}')
    with pytest.raises(q.ComparisonError, match="duplicate"):
        q._Reader().record(pin)
    pin = store.save(b'{"x":NaN}')
    with pytest.raises(q.ComparisonError):
        q._Reader().record(pin)
    pin = store.save(b"binary")
    Path(pin["path"]).write_bytes(b"mutate")
    with pytest.raises(q.ComparisonError, match="SHA"):
        q._Reader().pin(q.Artifact.from_record(pin))
    output = tmp_path / "output.json"
    q.write_output(output, {"value": 1})
    with pytest.raises(FileExistsError):
        q.write_output(output, {"value": 2})
    assert json.loads(output.read_bytes()) == {"value": 1}
    assert not list(tmp_path.glob(".qf3-comparison-*"))


def test_light_import_and_legacy_bytes():
    source = "import sys; from abc_bench import qf3_campaign_comparison; assert not any(x.split('.')[0] in {'torch','numpy','nrh','abc_sim','abc_minimal'} for x in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-B", "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        hashlib.sha256(Path(q.legacy.__file__).read_bytes()).hexdigest() == q.LEGACY_SHA
    )


def test_cli_refusal_creates_no_output(tmp_path):
    store = Store(tmp_path)
    pin = store.save({"schema_version": 1, "kind": "wrong"})
    output = tmp_path / "result.json"
    assert (
        q.main(
            [
                "--request",
                pin["path"],
                "--request-sha256",
                pin["sha256"],
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()


def split_chain(campaign, *, pending=False, discard=False):
    """Use the same actual parsed tape rows in two admitted invocation deltas."""
    store = campaign["store"]
    entry = campaign["request"]["seeds"][0]["training_chain"][0]
    original = store.read(entry["admission"])
    original_report = store.read(entry["worker"])
    config = store.read(original_report["resolved_config"])
    k = original["boundary"]["iteration"]
    first_b, first_k, first_h = (4, k, k - 1) if pending else (2, 0, 0)
    nodes, previous = [], None
    for invocation, (b, outer, h) in enumerate(
        ((first_b, first_k, first_h), (4, k, k))
    ):
        audit = copy.deepcopy(original)
        full_tapes = [
            a
            for a in original["accepted_tapes"]
            if a["ordinal"] < (b if a["category"] == "warmup" else outer)
        ]
        previous_tapes = previous["covered_training_tape_sha256"] if previous else []
        audit["accepted_tapes"] = [
            a for a in full_tapes if a["artifact"]["sha256"] not in previous_tapes
        ]
        previous_h = (
            len(previous["boundary"]["evaluation_state"]["history"]) if previous else 0
        )
        audit["fresh_tapes"] = original["fresh_tapes"][previous_h:h]
        audit["discarded_tapes"] = []
        audit["covered_training_tape_sha256"] = [
            a["artifact"]["sha256"] for a in full_tapes
        ]
        boundary = audit["boundary"]
        boundary.update(
            iteration=outer,
            warmup_batches=b,
            boundary="completed_outer" if outer else "completed_warmup_batch",
            warmup_steps=sum(
                a["accounting"]["simulation_steps"]
                for a in full_tapes
                if a["category"] == "warmup"
            ),
            training_steps=sum(
                a["accounting"]["simulation_steps"]
                for a in full_tapes
                if a["category"] == "train"
            ),
            physics_ticks=sum(a["accounting"]["physics_ticks"] for a in full_tapes),
            completed_episodes=80 * b + 16 * outer,
            evaluation_steps=50 * h,
            evaluation_physics_ticks=h,
        )
        boundary["evaluation_state"].update(
            history=original["boundary"]["evaluation_state"]["history"][:h],
            validation_cursor=h,
            next_threshold=h + 1,
            pending_validation=None,
        )
        if pending and invocation == 0:
            manifest_pin = original["boundary"]["evaluation_state"]["history"][-1][
                "validation"
            ]["manifest"]
            boundary["evaluation_state"]["pending_validation"] = {
                "manifest": manifest_pin
            }
            audit["pending_noise_rng_sha256"] = q._hash(
                store.read(manifest_pin)["noise_rng"]
            )
        total = sum(a["accounting"]["decisions"] for a in full_tapes)
        audit["replay_metadata"].update(total_added=total, size=total, cursor=total)
        audit["replay_blocks"][0].update(record_count=total, selector_count=total)
        audit["counters"] = {
            "critic_updates": 1600 * outer,
            "actor_updates": 200 * outer,
            "critic_target_updates": 1600 * outer,
            "actor_target_updates": outer,
        }
        audit["admitted_checkpoint"] = store.save(
            f"synthetic boundary {invocation}".encode(),
            kind="qf3_completed_boundary_checkpoint",
        )
        new_config = copy.deepcopy(config)
        if previous:
            new_config["resume"] = previous["admitted_checkpoint"]
        new_report = copy.deepcopy(original_report)
        new_report.update(
            input_config=store.save(new_config),
            resolved_config=store.save(new_config),
            resolved_config_sha256=q._hash(new_config),
            status="budget_stopped" if invocation == 0 else "completed",
            restore_verified=invocation > 0,
            completed_checkpoints=[audit["admitted_checkpoint"]],
            evaluation_events=boundary["evaluation_state"]["history"],
            invocation_evaluation_events=boundary["evaluation_state"]["history"][
                previous_h:
            ],
        )
        previous_outer = previous["boundary"]["iteration"] if previous else 0
        new_report["updates"] = [
            row
            for row in original_report["updates"]
            if previous_outer <= row["iteration"] < outer
        ]
        consumed = [a["artifact"] for a in audit["accepted_tapes"]] + [
            a["artifact"] for a in audit["fresh_tapes"]
        ]
        delta = {
            "warmup_steps": sum(
                a["accounting"]["simulation_steps"]
                for a in audit["accepted_tapes"]
                if a["category"] == "warmup"
            ),
            "training_steps": sum(
                a["accounting"]["simulation_steps"]
                for a in audit["accepted_tapes"]
                if a["category"] == "train"
            ),
            "physics_ticks": sum(
                a["accounting"]["physics_ticks"] for a in audit["accepted_tapes"]
            ),
            "completed_episodes": sum(
                a["accounting"]["episodes"] for a in audit["accepted_tapes"]
            ),
        }
        discarded = {
            "warmup_steps": 0,
            "training_steps": 0,
            "evaluation_steps": 0,
            "physics_ticks": 0,
        }
        if pending and invocation == 0 and discard:
            value = store.read(original["fresh_tapes"][-1]["artifact"])
            value["rollout"]["partial"] = True
            value["rollout"]["episodes"] = 0
            for row in value["rollout"]["worlds"]:
                row.update(
                    completed=False, paper_success=False, native_success=False, reward=0
                )
            discarded_pin = store.save(value)
            attempted = q._discarded(q._Reader(), discarded_pin, new_report["base_id"])
            audit["discarded_tapes"] = [
                {"artifact": discarded_pin, "accounting": attempted}
            ]
            consumed.append(discarded_pin)
            discarded.update(
                evaluation_steps=attempted["simulation_steps"],
                physics_ticks=attempted["physics_ticks"],
            )
            partial = copy.deepcopy(
                original["boundary"]["evaluation_state"]["history"][-1]
            )
            partial["summary"].update(partial=True, completed_episodes=0)
            new_report["invocation_evaluation_events"].append(partial)
            new_report["evaluation_events"] = boundary["evaluation_state"][
                "history"
            ] + [partial]
        selected_hashes = {p["sha256"] for p in consumed}
        new_report["rollouts"] = [
            p for p in original_report["rollouts"] if p["sha256"] in selected_hashes
        ]
        new_report["rollouts"].extend(
            p
            for p in consumed
            if p["sha256"] not in {p["sha256"] for p in original_report["rollouts"]}
        )
        new_report["metrics"].update(
            training_steps=boundary["training_steps"],
            warmup_steps=boundary["warmup_steps"],
            outer_iterations=outer,
            actor_updates=200 * outer,
            critic_updates=1600 * outer,
            target_reached=boundary["training_steps"] >= q.TARGET,
            evaluation_steps=boundary["evaluation_steps"]
            + discarded["evaluation_steps"],
            physics_ticks_evaluation=boundary["evaluation_physics_ticks"]
            + discarded["physics_ticks"],
            physics_ticks_training=boundary["physics_ticks"],
            completed_training_episodes=boundary["completed_episodes"],
            replay_total_added=total,
        )
        new_report["phase_counts"].update(
            warmup_steps=delta["warmup_steps"],
            train_steps=delta["training_steps"],
            evaluation_steps=50 * (h - previous_h) + discarded["evaluation_steps"],
            critic_updates=1600 * (outer - previous_outer),
            actor_updates=200 * (outer - previous_outer),
        )
        new_worker_pin = store.save(new_report)
        parent_pin = parent(
            store,
            new_worker_pin,
            new_report,
            new_config,
            f"2026-10-08T00:0{2 * invocation}:00+00:00",
        )
        audit.update(
            worker_receipt=new_worker_pin,
            controller_receipt=parent_pin,
            resolved_config=new_report["resolved_config"],
            predecessor_admission=nodes[-1]["admission"] if nodes else None,
            invocation_accounting={
                "accepted": delta,
                "discarded": discarded,
                "updates": {
                    "accepted": {
                        "critic": 1600 * (outer - previous_outer),
                        "actor": 200 * (outer - previous_outer),
                    },
                    "discarded": {"critic": 0, "actor": 0},
                },
                "accepted_fresh_steps": 50 * (h - previous_h),
                "accepted_fresh_ticks": h - previous_h,
            },
        )
        nodes.append(
            {
                "worker": new_worker_pin,
                "parent": parent_pin,
                "admission": store.save(audit),
                "input_preflight": preflight(store, new_report, new_config),
            }
        )
        previous = audit
    return nodes


@pytest.mark.parametrize(
    "pending,discard", [(False, False), (True, False), (True, True)]
)
def test_completed_warmup_and_pending_event_resume_exact_deltas(
    campaign, pending, discard
):
    nodes = split_chain(campaign, pending=pending, discard=discard)
    result = q._chain(q._Reader(), nodes, 0)
    assert result["outer_iterations"] == 26
    assert result["warmup_steps"] == 320
    assert result["training_steps"] == 6240
    assert result["accepted_fresh_events"] == 26
    if pending:
        last = result["invocations"][-1]["accounting"]
        assert last["accepted"]["training_steps"] == 0
        assert last["updates"]["accepted"] == {"critic": 0, "actor": 0}
        assert last["accepted_fresh_steps"] == 50
        assert result["invocations"][0]["accounting"]["discarded"][
            "evaluation_steps"
        ] == (50 if discard else 0)


def test_resume_counter_credit_and_pending_manifest_replacement_refuse(campaign):
    nodes = split_chain(campaign, pending=True)
    store = campaign["store"]
    corrupted = copy.deepcopy(nodes)
    corrupted[1]["admission"] = store.replace(
        corrupted[1]["admission"],
        lambda a: a["invocation_accounting"]["accepted"].update(training_steps=6240),
    )
    with pytest.raises(q.ComparisonError, match="accounting"):
        q._chain(q._Reader(), corrupted, 0)
    # A byte-distinct manifest with the same semantic event is still a different
    # pending artifact, and must not replace the originally durable intent.
    corrupted = copy.deepcopy(nodes)
    audit = store.read(corrupted[1]["admission"])
    event = audit["boundary"]["evaluation_state"]["history"][-1]
    event["validation"]["manifest"] = store.replace(
        event["validation"]["manifest"], lambda m: m.update(extra_fixture_field=True)
    )
    corrupted[1]["admission"] = store.save(audit)
    with pytest.raises(q.ComparisonError, match="pending retry"):
        q._chain(q._Reader(), corrupted, 0)


def test_first_target_not_later_qualified_boundary(campaign):
    nodes = split_chain(campaign)
    store = campaign["store"]
    audit = store.read(nodes[1]["admission"])
    # Earlier accepted event reaches the target. A later target-qualified final
    # boundary must fail even if it is the last linked artifact.
    audit["boundary"]["evaluation_state"]["history"][0]["training_control_steps"] = (
        q.TARGET
    )
    nodes[1]["admission"] = store.save(audit)
    with pytest.raises(q.ComparisonError):
        q._chain(q._Reader(), nodes, 0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["after_state_sha256"].update(training_rng="0" * 64),
        lambda p: p["checks"].update(
            saved_learner_replay_bridge_rng_preservation=False
        ),
        lambda p: p["replay_blocks"][0].update(selector_count=0),
        lambda p: p.update(native_world_restore_claimed=True),
    ],
)
def test_independent_saved_final_preservation_required(campaign, mutation):
    root = copy.deepcopy(campaign["request"]["seeds"][0])
    entry = root["learned_final"]
    entry["preservation_admission"] = campaign["store"].replace(
        entry["preservation_admission"], mutation
    )
    chain = q._chain(q._Reader(), root["training_chain"], 0)
    with pytest.raises(q.ComparisonError):
        q._evaluation(q._Reader(), entry, chain, 0, "learned")


def test_unreached_target_cannot_pass_completed_status(campaign, monkeypatch):
    monkeypatch.setattr(q, "TARGET", 400000)
    with pytest.raises(q.ComparisonError):
        q._chain(q._Reader(), campaign["request"]["seeds"][0]["training_chain"], 0)


def test_json_signature_change_during_read_refuses(tmp_path, monkeypatch):
    store = Store(tmp_path)
    pin = store.save({"x": 1})
    reader = q._Reader()
    calls = iter(((1, 1, 1, 1), (1, 1, 1, 2)))
    monkeypatch.setattr(reader, "signature", lambda path: next(calls))
    with pytest.raises(q.ComparisonError, match="changed during read"):
        reader.record(pin)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["evaluation"].update(worlds=4),
        lambda c: c["evaluation"].update(sampler_seed=91001),
        lambda c: c["evaluation"]["seeds"].reverse(),
        lambda c: c["training"].update(max_outer_iterations=25),
        lambda c: c["training"].update(target_control_steps=16000),
        lambda c: c["lora"].update(seed=902.0),
        lambda c: c["base"].update(seed=901.0),
        lambda c: c.update(checkpoint_storage={"mode": "inline", "schema_version": 1}),
        lambda c: c["source_artifacts"]["abc_minimal.vla"].update(sha256="0" * 64),
        lambda c: c["native_inventory"]["assets"].update(sha256="0" * 64),
        lambda c: c.update(fixed_layout_seed=901),
    ],
)
def test_exact_recipe_grouping_noise_and_native_profile(campaign, mutation):
    store = campaign["store"]
    report = store.read(campaign["request"]["seeds"][0]["training_chain"][0]["worker"])
    config = store.read(report["resolved_config"])
    mutation(config)
    report.update(
        input_config=store.save(config),
        resolved_config=store.save(config),
        resolved_config_sha256=q._hash(config),
    )
    with pytest.raises(q.ComparisonError):
        q._config(q._Reader(), report, 0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda w: w["camera_capture_configuration"].update(camera_height=224),
        lambda w: w["camera_capture_configuration"].update(camera_gpu_id=1),
        lambda w: w["runtime_identity"].update(prompt="put bottles away"),
        lambda w: w["runtime_identity"]["versions"].update(mujoco="3.13.0"),
        lambda w: w["runtime_identity"].update(diffusion_steps=10),
        lambda w: w["runtime_identity"].update(exact_paper_initializer_verified=True),
        lambda w: w["sources"].update(**{"nrh.qf3_training": "0" * 64}),
    ],
)
def test_capture_prompt_versions_and_source_guard(campaign, mutation):
    store = campaign["store"]
    report = store.read(campaign["request"]["seeds"][0]["training_chain"][0]["worker"])
    config = store.read(report["resolved_config"])
    mutation(report)
    report["base_id"] = q._hash(report["runtime_identity"])
    with pytest.raises(q.ComparisonError):
        q._runtime(q._Reader(), report, config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(noise_seed=91001),
        lambda m: m.update(cursor=1.0),
        lambda m: m["requested_seeds"].reverse(),
        lambda m: m["noise_rng"].update(has_uint32=False),
    ],
)
def test_fresh_noise_order_and_strict_metadata(campaign, mutation):
    store = campaign["store"]
    root = campaign["request"]["seeds"][0]["training_chain"][0]
    report, audit = store.read(root["worker"]), store.read(root["admission"])
    config = store.read(report["resolved_config"])
    event = copy.deepcopy(audit["boundary"]["evaluation_state"]["history"][0])
    event["validation"]["manifest"] = store.replace(
        event["validation"]["manifest"], mutation
    )
    with pytest.raises(q.ComparisonError):
        q._fresh(
            q._Reader(),
            event,
            report,
            config,
            0,
            0,
            audit["fresh_tapes"][0]["artifact"],
            geometry_seen=set(),
        )


def test_block_missing_or_changed_bytes_not_admitted(campaign, tmp_path):
    store = campaign["store"]
    root = campaign["request"]["seeds"][0]["training_chain"][0]
    report, audit = store.read(root["worker"]), store.read(root["admission"])
    config = store.read(report["resolved_config"])
    block = tmp_path / "block"
    block.write_bytes(b"original")
    pin = {
        "path": str(block),
        "bytes": 8,
        "sha256": hashlib.sha256(b"original").hexdigest(),
        "kind": "qf3_immutable_replay_block",
    }
    audit["replay_blocks"][0]["artifact"] = pin
    record = store.save(audit)
    block.write_bytes(b"mutation")
    with pytest.raises(q.ComparisonError, match="SHA"):
        q._admission(q._Reader(), record, report, config)
    block.unlink()
    with pytest.raises(q.ComparisonError, match="missing"):
        q._admission(q._Reader(), record, report, config)


def test_preserved_unadmitted_costs_have_no_counter_credit(campaign):
    store = campaign["store"]
    record = campaign["request"]["seeds"][0]["training_chain"][0]
    parent_record = store.read(record["parent"])
    parent_record["status"] = "timed_out"
    report = store.read(record["worker"])
    entry = {
        "seed_index": 0,
        "parent": store.save(parent_record),
        "input_config": report["input_config"],
        "worker": record["worker"],
        "reason": "explicit synthetic discarded attempt",
        "accepted_counter_credit": 0,
    }
    accepted_result = q._unadmitted_attempts(q._Reader(), [entry], set())
    assert accepted_result[0]["parent_elapsed_seconds"] == 1.0
    entry["accepted_counter_credit"] = 6240
    with pytest.raises(q.ComparisonError, match="credit"):
        q._unadmitted_attempts(q._Reader(), [entry], set())


def test_predecessor_and_new_input_preflight_required(campaign):
    nodes = split_chain(campaign)
    store = campaign["store"]
    corrupted = copy.deepcopy(nodes)
    corrupted[1]["admission"] = store.replace(
        corrupted[1]["admission"], lambda a: a.update(predecessor_admission=None)
    )
    with pytest.raises(q.ComparisonError, match="predecessor"):
        q._chain(q._Reader(), corrupted, 0)
    corrupted = copy.deepcopy(nodes)
    corrupted[1]["input_preflight"] = store.replace(
        corrupted[1]["input_preflight"], lambda p: p.update(continuation_id="0" * 64)
    )
    with pytest.raises(q.ComparisonError, match="preflight"):
        q._chain(q._Reader(), corrupted, 0)


def test_conflicting_pins_and_relative_paths_refuse(tmp_path):
    store = Store(tmp_path)
    pin = store.save({"x": 1})
    reader = q._Reader()
    reader.record(pin)
    other = {**pin, "sha256": "0" * 64}
    with pytest.raises(q.ComparisonError, match="conflicting"):
        reader.record(other)
    with pytest.raises(q.ComparisonError, match="absolute"):
        q._Reader().record({**pin, "path": "relative.json"})


def test_production_target_and_source_profile_not_caller_options():
    assert q.TARGET == q.TRAINING["target_control_steps"] == 400000
    assert q.TRAINING["max_outer_iterations"] == 25000
    assert (
        q.PROFILE["nrh.qf3_training"]
        == "65a62e73c7ad2079c8d6c40262220cb8149323f7f7e7bed8425d2d162645a2c8"
    )


@pytest.mark.parametrize(
    "counter,discarded",
    [
        ("evaluation_steps", "discarded_evaluation_steps"),
        ("evaluation_physics_ticks", "discarded_evaluation_physics_ticks"),
    ],
)
def test_canonical_evaluation_credit_requires_retained_tapes(
    campaign, counter, discarded
):
    entry = copy.deepcopy(campaign["request"]["seeds"][0]["training_chain"][0])
    store = campaign["store"]

    def fabricate(value):
        value["boundary"][counter] += 10
        value["boundary"][discarded] += 10

    entry["admission"] = store.replace(entry["admission"], fabricate)
    with pytest.raises(q.ComparisonError, match="retained tape evidence"):
        q._chain(q._Reader(), [entry], 0)


def retained_pending_attempt(campaign):
    """Select the valid after-attempt pending checkpoint, not the earlier intent."""
    store = campaign["store"]
    nodes = split_chain(campaign, pending=True, discard=True)
    first = store.read(nodes[0]["admission"])
    attempted = first["discarded_tapes"][0]
    steps, ticks = (
        attempted["accounting"]["simulation_steps"],
        attempted["accounting"]["physics_ticks"],
    )
    first["retained_discarded_evaluation_tape_sha256"] = [
        attempted["artifact"]["sha256"]
    ]
    for key, value in (
        ("evaluation_steps", steps),
        ("discarded_evaluation_steps", steps),
        ("evaluation_physics_ticks", ticks),
        ("discarded_evaluation_physics_ticks", ticks),
    ):
        first["boundary"][key] += value
    nodes[0]["admission"] = store.save(first)
    second = store.read(nodes[1]["admission"])
    second["predecessor_admission"] = nodes[0]["admission"]
    for key, value in (
        ("evaluation_steps", steps),
        ("discarded_evaluation_steps", steps),
        ("evaluation_physics_ticks", ticks),
        ("discarded_evaluation_physics_ticks", ticks),
    ):
        second["boundary"][key] += value
    report = store.read(nodes[1]["worker"])
    report["metrics"]["evaluation_steps"] += steps
    report["metrics"]["physics_ticks_evaluation"] += ticks
    nodes[1]["worker"] = store.save(report)
    p = store.read(nodes[1]["parent"])
    p["summary_sha256"] = nodes[1]["worker"]["sha256"]
    p["metrics"]["evaluation_steps"] += steps
    p["metrics"]["physics_ticks_evaluation"] += ticks
    nodes[1]["parent"] = store.save(p)
    second.update(
        worker_receipt=nodes[1]["worker"], controller_receipt=nodes[1]["parent"]
    )
    nodes[1]["admission"] = store.save(second)
    return nodes


def test_post_attempt_pending_checkpoint_retains_exact_credit_on_resume(campaign):
    nodes = retained_pending_attempt(campaign)
    chain = q._chain(q._Reader(), nodes, 0)
    selected = chain["selected"]["admission"]["boundary"]
    assert selected["discarded_evaluation_steps"] == 50
    assert selected["discarded_evaluation_physics_ticks"] == 1
    assert selected["evaluation_steps"] == 26 * 50 + 50
    assert chain["invocations"][1]["accounting"]["accepted_fresh_steps"] == 50
    assert chain["invocations"][1]["retained_discarded_evaluation_tape_sha256"] == []


@pytest.mark.parametrize(
    "mode",
    ["unknown", "duplicate", "unretained_counter", "partial_counter", "non_prefix"],
)
def test_retained_attempt_unknown_duplicate_and_partial_credit_refuses(campaign, mode):
    store = campaign["store"]
    nodes = retained_pending_attempt(campaign)
    first = store.read(nodes[0]["admission"])
    pin = first["discarded_tapes"][0]["artifact"]
    if mode == "unknown":
        first["retained_discarded_evaluation_tape_sha256"] = ["0" * 64]
    elif mode == "duplicate":
        first["retained_discarded_evaluation_tape_sha256"] = [pin["sha256"]] * 2
    elif mode == "unretained_counter":
        first["retained_discarded_evaluation_tape_sha256"] = []
    elif mode == "partial_counter":
        first["boundary"]["evaluation_steps"] -= 40
        first["boundary"]["discarded_evaluation_steps"] -= 40
    else:
        first["retained_discarded_evaluation_tape_sha256"] = ["0" * 64, pin["sha256"]]
    nodes[0]["admission"] = store.save(first)
    with pytest.raises(q.ComparisonError, match="retained|prefix"):
        q._chain(q._Reader(), nodes, 0)


@pytest.mark.parametrize(
    "head,value",
    [("learned", False), ("learned", "yes"), ("learned", 1), ("frozen", True)],
)
def test_final_target_flag_must_be_exact_inherited_boolean(campaign, head, value):
    store = campaign["store"]
    root = campaign["request"]["seeds"][0]
    entry = copy.deepcopy(root[head + "_final"])
    report = store.read(entry["worker"])
    report["metrics"]["target_reached"] = value
    entry["worker"] = store.save(report)
    p = store.read(entry["parent"])
    p["summary_sha256"] = entry["worker"]["sha256"]
    p["metrics"]["target_reached"] = value
    entry["parent"] = store.save(p)
    if head == "learned":
        audit = store.read(entry["preservation_admission"])
        audit.update(worker_receipt=entry["worker"], controller_receipt=entry["parent"])
        entry["preservation_admission"] = store.save(audit)
    chain = q._chain(q._Reader(), root["training_chain"], 0)
    with pytest.raises(q.ComparisonError, match="target_reached"):
        q._evaluation(q._Reader(), entry, chain, 0, head)


def test_preservation_domain_is_distinct_and_selected_hash_cannot_change(campaign):
    root = campaign["request"]["seeds"][0]
    chain = q._chain(q._Reader(), root["training_chain"], 0)
    admission = chain["selected"]["admission"]
    assert (
        admission["exposed_state_sha256"]
        != admission["evaluation_preservation_state_sha256"]
    )
    q._evaluation(q._Reader(), root["learned_final"], chain, 0, "learned")
    corrupted = copy.deepcopy(chain)
    corrupted["selected"]["admission"]["evaluation_preservation_state_sha256"] = (
        "0" * 64
    )
    with pytest.raises(q.ComparisonError, match="selected learner/replay"):
        q._evaluation(q._Reader(), root["learned_final"], corrupted, 0, "learned")


def test_coherent_final_hash_replacement_cannot_bypass_selected_domain(campaign):
    store = campaign["store"]
    root = campaign["request"]["seeds"][0]
    entry = copy.deepcopy(root["learned_final"])
    report = store.read(entry["worker"])
    for summary_record in (
        report["invocation_evaluation_events"][0]["summary"],
        report["evaluation_events"][-1]["summary"],
        report["evaluation"]["learned"],
    ):
        summary_record.update(
            training_state_before="9" * 64, training_state_after="9" * 64
        )
    entry["worker"] = store.save(report)
    p = store.read(entry["parent"])
    p["summary_sha256"] = entry["worker"]["sha256"]
    entry["parent"] = store.save(p)
    audit = store.read(entry["preservation_admission"])
    audit.update(
        worker_receipt=entry["worker"],
        controller_receipt=entry["parent"],
        exposed_state_sha256="9" * 64,
    )
    entry["preservation_admission"] = store.save(audit)
    chain = q._chain(q._Reader(), root["training_chain"], 0)
    with pytest.raises(q.ComparisonError, match="selected learner/replay"):
        q._evaluation(q._Reader(), entry, chain, 0, "learned")


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("head", [None, "learned", "frozen"])
def test_three_seed_configs_use_approved_distinct_final_cadences(campaign, index, head):
    root = campaign["request"]["seeds"][index]
    entry = root[head + "_final"] if head else root["training_chain"][0]
    report = campaign["store"].read(entry["worker"])
    config = q._config(q._Reader(), report, index, head=head)
    expected = {
        "unit": "final" if head == "frozen" else "outer_iterations",
        "every": None if head == "frozen" else 1,
        "phase": "after_outer_updates",
    }
    assert config["evaluation"]["cadence"] == expected


@pytest.mark.parametrize("head", [None, "learned", "frozen"])
def test_wrong_training_or_final_cadence_refuses(campaign, head):
    store = campaign["store"]
    root = campaign["request"]["seeds"][0]
    entry = root[head + "_final"] if head else root["training_chain"][0]
    report = store.read(entry["worker"])
    config = store.read(report["resolved_config"])
    config["evaluation"]["cadence"] = {
        "unit": "outer_iterations" if head == "frozen" else "final",
        "every": 1 if head == "frozen" else None,
        "phase": "after_outer_updates",
    }
    report.update(
        input_config=store.save(config),
        resolved_config=store.save(config),
        resolved_config_sha256=q._hash(config),
    )
    with pytest.raises(q.ComparisonError, match="evaluation/cohort/noise protocol"):
        q._config(q._Reader(), report, 0, head=head)


@pytest.mark.parametrize("stage", ["train", "evaluate"])
def test_parent_frozen_controller_stage_label(tmp_path, stage):
    """Producer runner25e3c270 emits this label; other metadata stays strict."""
    store = Store(tmp_path)
    config = {
        "stage": stage,
        "seed": 903,
        "base": {"seed": 901},
        "evaluation": {"heads": ["learned"], "sampler_seed": 902},
    }
    metrics = {
        key: 0
        for key in (
            "training_steps",
            "warmup_steps",
            "outer_iterations",
            "actor_updates",
            "critic_updates",
            "evaluation_steps",
            "physics_ticks_evaluation",
            "physics_ticks_training",
            "completed_training_episodes",
            "replay_total_added",
        )
    }
    metrics["target_reached"] = False
    worker = {
        "status": "completed",
        "input_config": {"sha256": "1" * 64},
        "metrics": metrics,
    }
    worker_pin = store.save(worker)
    record = parent(store, worker_pin, worker, config, "2026-10-09T00:00:00Z")
    assert (
        q._parent(q.legacy._Reader(), record, worker_pin, worker, config)["algorithm"]
        == "QF3 ABC-VLA " + stage
    )
    for wrong in (
        "qf3-vla",
        "QF3 ABC-VLA wrong",
        "QF3 ABC-VLA " + ("evaluate" if stage == "train" else "train"),
    ):
        bad = store.replace(
            record, lambda value, wrong=wrong: value.update(algorithm=wrong)
        )
        with pytest.raises(q.ComparisonError, match="controller algorithm differs"):
            q._parent(q.legacy._Reader(), bad, worker_pin, worker, config)
    bad = store.replace(record, lambda value: value.update(training_stage="collect"))
    with pytest.raises(q.ComparisonError, match="parent stage differs"):
        q._parent(q.legacy._Reader(), bad, worker_pin, worker, config)
    bad = store.replace(record, lambda value: value.update(status="running"))
    with pytest.raises(q.ComparisonError, match="controller invocation incomplete"):
        q._parent(q.legacy._Reader(), bad, worker_pin, worker, config)

    wrong_seed = config["seed"] if stage == "evaluate" else None
    bad = store.replace(record, lambda value: value.update(seed=wrong_seed))
    with pytest.raises(q.ComparisonError, match="parent seed differs"):
        q._parent(q.legacy._Reader(), bad, worker_pin, worker, config)
