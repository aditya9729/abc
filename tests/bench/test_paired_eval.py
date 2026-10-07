"""CPU checks for paired evaluation's source and metric contracts."""

import numpy as np
import pytest
import torch
from torch import nn

from abc_bench.paired_eval import (
    HISTORICAL_SNAPSHOTS,
    load_output_adapter,
    make_residual_actor,
    summarize_worlds,
    validate_snapshot,
)
from abc_bench.update_smoke import ActionCodec, OutputAdapter


def test_snapshot_rejects_source_or_action_contract_mismatch():
    snapshot = {
        "algorithm": "qf3",
        "base_checkpoint_sha256": "base",
        "executed_prefix_steps": 8,
    }
    validate_snapshot(snapshot, "qf3", "base")
    with pytest.raises(ValueError, match="mismatch"):
        validate_snapshot(snapshot, "qf3", "other")
    snapshot["executed_prefix_steps"] = 15
    with pytest.raises(ValueError, match="eight-action"):
        validate_snapshot(snapshot, "qf3", "base")


def test_historical_snapshot_requires_external_pins():
    snapshot = {"algorithm": "resfit", "executed_prefix_steps": 8}
    with pytest.raises(ValueError, match="Historical"):
        validate_snapshot(snapshot, "resfit", "base")
    validate_snapshot(
        snapshot,
        "resfit",
        "base",
        artifact_hash=HISTORICAL_SNAPSHOTS["resfit"],
        manifest_base_hash="base",
    )
    with pytest.raises(ValueError, match="Historical"):
        validate_snapshot(
            snapshot,
            "resfit",
            "base",
            artifact_hash=HISTORICAL_SNAPSHOTS["resfit"],
            manifest_base_hash="different",
        )


def test_adapter_rejects_mutated_base_and_wrong_rank():
    adapter = OutputAdapter(nn.Linear(12, 3))
    snapshot = {
        "adapter": {key: value.clone() for key, value in adapter.state_dict().items()}
    }
    load_output_adapter(adapter, snapshot)
    snapshot["adapter"]["base.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="immutable"):
        load_output_adapter(adapter, snapshot)
    snapshot["adapter"] = adapter.state_dict()
    snapshot["adapter"]["down.weight"] = torch.zeros(3, 12)
    with pytest.raises(ValueError, match="Invalid"):
        load_output_adapter(adapter, snapshot)


def test_residual_shape_and_codec_boundary_fail_closed():
    with pytest.raises(ValueError, match="contract"):
        make_residual_actor({"actor": {"wrong": torch.zeros(1)}}, "cpu")
    codec = ActionCodec(np.zeros(14), np.ones(14))
    with pytest.raises(ValueError, match="strictly"):
        codec.decode(np.ones((8, 14)))


def test_failure_does_not_become_successful_completion_time():
    failed = {
        "success": False,
        "steps": 1000,
        "simulation_duration_seconds": 34.0,
        "last_task_score": 0.0,
    }
    metrics = summarize_worlds([failed], [0.1, 0.2])
    assert metrics["completion_time_seconds"] is None
    assert metrics["episode_duration_seconds"]["mean"] == 34.0
    passed = {
        "success": True,
        "steps": 100,
        "simulation_duration_seconds": 3.4,
        "last_task_score": 1.0,
    }
    metrics = summarize_worlds([failed, passed], [0.1, 0.2])
    assert metrics["successes"] == 1
    assert metrics["episodes"] == 2
    assert metrics["completion_time_seconds"] == 3.4
    assert metrics["simulation_steps"] == 1100


def test_deployment_boundary_clamp_is_explicit_and_measured():
    from abc_bench.paired_eval import DEPLOY_CODEC_EPSILON, decode_residual_proposal

    codec = ActionCodec(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32))
    coded = np.array([[1.0, -1.0], [0.2, -0.3]], dtype=np.float32)
    physical, count, delta = decode_residual_proposal(
        codec, coded, np.zeros_like(coded)
    )
    expected = np.arctanh(
        np.clip(coded, -1 + DEPLOY_CODEC_EPSILON, 1 - DEPLOY_CODEC_EPSILON)
    ) * (1 + 1e-6)
    np.testing.assert_allclose(physical, expected, atol=1e-6)
    assert count == 2
    assert delta == float(np.max(np.abs(physical)))
    assert np.isfinite(physical).all()
    with pytest.raises(ValueError, match="finite"):
        decode_residual_proposal(codec, np.array([[np.nan, 0.0]]), np.zeros((1, 2)))


def test_partial_evidence_survives_a_later_method_failure(tmp_path):
    import json

    from abc_bench.paired_eval import persist_partial

    output = tmp_path / "paired_eval.json"
    baseline = {
        "method": "baseline",
        "status": "completed",
        "worlds": [{"seed": 101, "steps": 1000}],
    }
    partial = {
        "method": "qf3",
        "status": "partial",
        "worlds": [{"seed": 101, "steps": 900}],
    }
    persist_partial(output, [baseline], partial, {"base_checkpoint_sha256": "base"})
    saved = json.loads(output.read_text())
    assert saved["status"] == "partial"
    assert saved["paired_initial_states_verified"] is False
    assert saved["runs"] == [baseline, partial]
    assert saved["base_checkpoint_sha256"] == "base"
    assert not output.with_suffix(".tmp").exists()


def test_paired_horizon_shares_runner_bound_and_requires_unique_seeds():
    from abc_bench.paired_eval import validate_evaluation_config

    validate_evaluation_config(2000, [101, 102, 103])
    validate_evaluation_config(3540, [101])
    with pytest.raises(ValueError, match="horizon"):
        validate_evaluation_config(3541, [101])
    with pytest.raises(ValueError, match="unique"):
        validate_evaluation_config(2000, [101, 101])
