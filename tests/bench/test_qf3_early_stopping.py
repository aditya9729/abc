"""Explicit CPU metadata/coordinator doubles; no native ABC task result."""

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from abc_bench import runner


def pin(path, kind):
    body = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "kind": kind,
    }


@pytest.fixture
def stopping_receipt(tmp_path):
    profile = {
        "profile": "user-requested-development-early-stopping/v1",
        "split_role": "development",
        "patience": 3,
        "min_success_improvement": 1,
    }
    config = {
        "schema_version": 1,
        "stage": "train",
        "device": "cuda:0",
        "max_wall_s": 500,
        "early_stopping": profile,
        "base": {
            "task_id": "put_plastic_bottles_in_bin",
            "checkpoint_path": "/fixture/base.pt",
            "seed": 1,
        },
        "training": {
            "warmup_episodes_per_world": 1,
            "target_control_steps": 400000,
            "train_worlds": 16,
        },
        "evaluation": {
            "sampler_seed": 71,
            "layout_sampling": {"seed_start": 2**40, "seed_stride": 200001},
        },
    }
    state = {
        "schema_version": 1,
        "profile": profile["profile"],
        "policy_sha256": hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "checks": [],
        "best": None,
        "stale_checks": 3,
        "stop_reason": "development_success_plateau",
    }
    history = []
    for ordinal, successes in enumerate([3, 3, 2, 3]):
        seeds = [2**40 + (ordinal * 50 + i) * 200001 for i in range(50)]
        policy = str(ordinal + 1) * 64
        manifest_path = tmp_path / f"manifest-{ordinal}.json"
        runner.write_json(
            manifest_path,
            {
                "cursor": ordinal,
                "requested_seeds": seeds,
                "noise_seed": 72 + ordinal,
                "policy_state_sha256": policy,
            },
        )
        manifest = pin(manifest_path, "fixture_manifest")
        reset_path = tmp_path / f"reset-{ordinal}.json"
        runner.write_json(
            reset_path,
            {
                "manifest_sha256": manifest["sha256"],
                "start": 0,
                "worlds": [
                    {
                        "requested_seed": seed,
                        "actual_seed": seed + 100000,
                        "geometry_sha256": hashlib.sha256(
                            str(seed).encode()
                        ).hexdigest(),
                    }
                    for seed in seeds
                ],
            },
        )
        event = {
            "head": "learned",
            "reason": "periodic",
            "summary": {
                "partial": False,
                "requested_episodes": 50,
                "attempted_episodes": 50,
                "completed_episodes": 50,
                "actor_updates": ordinal + 1,
                "successes": successes,
                "training_state_before": policy,
                "training_state_after": policy,
                "configured_seed_sha256": hashlib.sha256(
                    json.dumps(seeds, separators=(",", ":")).encode()
                ).hexdigest(),
            },
            "validation": {
                "manifest": manifest,
                "reset_evidence": [pin(reset_path, "fixture_reset")],
            },
        }
        event_path = tmp_path / f"event-{ordinal}.json"
        runner.write_json(
            event_path, {"schema_version": 1, "domain": "fixture", "event": event}
        )
        checkpoint = tmp_path / f"checkpoint-{ordinal}.pt"
        checkpoint.write_bytes(
            f"EXPLICIT CPU CHECKPOINT FILE DOUBLE {ordinal}".encode()
        )
        check = {
            "ordinal": ordinal,
            "event": pin(event_path, "qf3_accepted_development_evaluation"),
            "checkpoint": pin(checkpoint, "qf3_completed_boundary_checkpoint"),
            "policy_sha256": policy,
            "successes": successes,
            "improved": ordinal == 0,
        }
        state["checks"].append(check)
        history.append(event)
    state["best"] = copy.deepcopy(state["checks"][0])
    return config, {
        "stage": "train",
        "status": "early_stopped",
        "domain": "fixture",
        "stopping_profile": profile,
        "early_stopping": state,
        "latest_complete": state["checks"][-1]["checkpoint"],
        "wrapper": {
            "partial": False,
            "boundary": "completed_outer",
            "warmup_batches": 1,
            "early_stopping": state,
            "evaluation_state": {"history": history, "pending_validation": None},
        },
        "metrics": {
            "actor_updates": 4,
            "critic_updates": 8,
            "simulation_steps": 204,
            "target_reached": False,
        },
    }


def test_qf3_coordinator_publishes_intentional_stop_and_settles_existing_lease(
    stopping_receipt, tmp_path, monkeypatch
):
    config, summary = stopping_receipt
    campaign = tmp_path / "campaign"
    monkeypatch.setattr(runner, "RESULTS", campaign)
    runner.write_json(
        campaign / "gpu_budget.json",
        {"limit_seconds": 7200, "charged_seconds": 7100, "gpu": 0},
    )
    monkeypatch.setattr(
        runner,
        "inspect_reserved_gpu",
        lambda _: {"uuid": "GPU-00000000-0000-0000-0000-000000000000"},
    )
    monkeypatch.setattr(runner, "execution_provenance", lambda: {"fixture_only": True})
    monkeypatch.setattr(runner, "publish", lambda *_: None)
    path = tmp_path / "input.json"
    runner.write_json(path, config)

    def fixture_worker(command, log_path, timeout, *, gpu_uuid):
        assert command[1:3] == ["-m", "nrh.qf3_training"] and timeout == 100
        child = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert child["early_stopping"] == config["early_stopping"]
        runner.write_json(
            Path(command[command.index("--out") + 1]) / "receipt.json", summary
        )
        return 0

    monkeypatch.setattr(runner, "execute_command", fixture_worker)
    args = Namespace(
        results=tmp_path / "results",
        timeout_seconds=500,
        algorithm="qf3-vla",
        training_config=path,
        checkpoint=tmp_path / "unused.pt",
        task="unused",
        worlds=1,
        chunks=2,
        seed=0,
        video=False,
    )
    result = runner.run_baseline(args)
    assert result["status"] == result["worker_status"] == "early_stopped"
    assert (
        result["training_target_reached"] is False
        and result["metrics"]["successes"] is None
    )
    assert result["early_stopping"]["best"] == summary["early_stopping"]["best"]
    ledger = json.loads((campaign / "gpu_budget.json").read_text())
    assert 7100 <= ledger["charged_seconds"] < 7101 and "active_gpu_uuid" not in ledger


@pytest.mark.parametrize(
    "fault",
    [
        "baseline",
        "partial",
        "final",
        "missing_actor",
        "stale",
        "policy",
        "duplicate",
        "checkpoint",
        "schema_bool",
        "stale_float",
        "ordinal_bool",
        "score_float",
        "profile_patience_float",
        "profile_improvement_bool",
        "warmup_bool",
        "actor_bool",
        "episode_float",
        "event_actor_bool",
        "best_score_float",
        "wrapper_schema_bool",
        "wrapper_stale_float",
    ],
)
def test_qf3_coordinator_refuses_false_stop(stopping_receipt, fault):
    config, summary = stopping_receipt
    if fault == "baseline":
        summary["wrapper"]["evaluation_state"]["history"][0]["head"] = "frozen"
    elif fault == "partial":
        summary["wrapper"]["evaluation_state"]["history"][0]["summary"]["partial"] = (
            True
        )
    elif fault == "final":
        config["early_stopping"]["split_role"] = "final"
    elif fault == "missing_actor":
        summary["metrics"]["actor_updates"] = 0
    elif fault == "stale":
        summary["early_stopping"]["stale_checks"] = 2
    elif fault == "policy":
        summary["early_stopping"]["checks"][0]["policy_sha256"] = "0" * 64
    elif fault == "duplicate":
        summary["early_stopping"]["checks"][1]["event"] = summary["early_stopping"][
            "checks"
        ][0]["event"]
    elif fault == "schema_bool":
        summary["early_stopping"]["schema_version"] = True
    elif fault == "stale_float":
        summary["early_stopping"]["stale_checks"] = 3.0
    elif fault == "ordinal_bool":
        summary["early_stopping"]["checks"][0]["ordinal"] = False
    elif fault == "score_float":
        summary["early_stopping"]["checks"][0]["successes"] = 3.0
    elif fault == "profile_patience_float":
        config["early_stopping"]["patience"] = 3.0
    elif fault == "profile_improvement_bool":
        config["early_stopping"]["min_success_improvement"] = True
    elif fault == "warmup_bool":
        summary["wrapper"]["warmup_batches"] = True
    elif fault == "actor_bool":
        summary["metrics"]["actor_updates"] = True
    elif fault == "episode_float":
        summary["wrapper"]["evaluation_state"]["history"][0]["summary"][
            "completed_episodes"
        ] = 50.0
    elif fault == "event_actor_bool":
        summary["wrapper"]["evaluation_state"]["history"][0]["summary"][
            "actor_updates"
        ] = True
    elif fault == "best_score_float":
        summary["early_stopping"]["best"]["successes"] = 3.0
    elif fault in {"wrapper_schema_bool", "wrapper_stale_float"}:
        summary["wrapper"]["early_stopping"] = copy.deepcopy(summary["early_stopping"])
        if fault == "wrapper_schema_bool":
            summary["wrapper"]["early_stopping"]["schema_version"] = True
        else:
            summary["wrapper"]["early_stopping"]["stale_checks"] = 3.0
    else:
        Path(summary["early_stopping"]["best"]["checkpoint"]["path"]).write_bytes(
            b"changed checkpoint"
        )
    with pytest.raises(ValueError):
        runner.qf3_stopping_result(config, summary)
