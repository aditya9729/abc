"""Independent paired orchestration checks; all runtimes are explicit CPU mocks."""

import copy
import json
import os
from argparse import Namespace
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest
import torch
from torch import nn

from abc_bench import paired_eval, runner
from abc_bench.dashboard import export_snapshot
from abc_bench.update_smoke import OutputAdapter


@pytest.mark.parametrize("horizon", [1000, 2000])
def test_comparison_forwards_task_and_does_not_invent_single_seed(
    monkeypatch, tmp_path, horizon
):
    campaign = tmp_path / "campaign"
    monkeypatch.setattr(runner, "RESULTS", campaign)
    runner.write_json(
        campaign / "runtime_manifest.json",
        {"upstream_commit": "d0832d12651d1b260a652861a14648dc5f3660c7"},
    )
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "synthetic-revision"
    )
    monkeypatch.setattr(
        runner,
        "inspect_reserved_gpu",
        lambda ordinal: {"ordinal": 0, "uuid": "GPU-fixture"},
    )
    published = []
    monkeypatch.setattr(
        runner,
        "publish",
        lambda receipt, results: published.append(copy.deepcopy(receipt)),
    )

    def synthetic_worker(command, log_path, timeout, *, gpu_uuid):
        assert gpu_uuid == "GPU-fixture"
        assert command[command.index("--task") + 1] == "custom-task"
        assert int(command[command.index("--horizon") + 1]) == horizon
        destination = command[command.index("--output") + 1]
        runner.write_json(
            runner.Path(destination),
            {
                "runs": [
                    {
                        "algorithm": "synthetic",
                        "metrics": {"simulation_steps": 8},
                        "artifacts": [],
                    }
                ]
            },
        )
        return 0

    monkeypatch.setattr(runner, "execute_command", synthetic_worker)
    args = Namespace(
        results=tmp_path / "results",
        timeout_seconds=2,
        checkpoint=tmp_path / "base.pt",
        task="custom-task",
        worlds=1,
        chunks=236,
        seed=999,
        video=False,
        algorithm="comparison",
        qf3_state=tmp_path / "q.pt",
        resfit_state=tmp_path / "r.pt",
        eval_horizon=horizon,
    )
    receipt = runner.run_baseline(args)
    assert receipt["status"] == "completed"
    assert receipt["upstream_commit"] == "d0832d12651d1b260a652861a14648dc5f3660c7"
    assert receipt["harness_commit"] == "synthetic-revision"
    assert (
        receipt["source_sha256"]["abc_bench/runner.py"]
        == runner.hashlib.sha256(runner.Path(runner.__file__).read_bytes()).hexdigest()
    )
    child = next(item for item in published if "-method-" in item["run_id"])
    assert child["seed"] is None
    assert child["evaluation_seeds"] == [101, 102, 103]
    assert child["evaluation_horizon_steps"] == horizon
    assert child["budget"]["steps"] == 9 * horizon
    assert f"{horizon}-step" in child["method_fidelity"]


def test_offline_export_links_resolve_from_destination_not_results_root(tmp_path):
    root = tmp_path / "source" / "results"
    root.mkdir(parents=True)
    artifact = root / "clip sample.mp4"
    artifact.write_bytes(b"synthetic video")
    (root / "results.json").write_text(
        json.dumps(
            {"runs": [{"run_id": "synthetic", "artifacts": [{"path": artifact.name}]}]}
        )
    )
    destination = tmp_path / "share" / "dashboard.html"
    export_snapshot(root, destination)
    expected = quote(os.path.relpath(artifact, destination.parent))
    assert '"url": "' + expected + '"' in destination.read_text()


def test_paired_video_failure_closes_env_and_uses_selected_campaign(
    monkeypatch, tmp_path
):
    import imageio.v2 as imageio

    import abc_minimal.eval_policy
    import abc_minimal.policy
    import abc_sim

    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"synthetic checkpoint, never loaded")
    for name in ("q.pt", "r.pt"):
        (tmp_path / name).write_bytes(b"synthetic snapshot, loader is mocked")
    base_hash = paired_eval.sha256(checkpoint)
    campaign = tmp_path / "selected-campaign"
    monkeypatch.setattr(runner, "RESULTS", campaign)
    runner.write_json(
        campaign / "runtime_manifest.json", {"checkpoint": {"sha256": base_hash}}
    )
    base = nn.Linear(2, 14)
    snapshot_adapter = OutputAdapter(base)
    snapshots = {
        "q.pt": {
            "algorithm": "qf3",
            "base_checkpoint_sha256": base_hash,
            "executed_prefix_steps": 8,
            "adapter": snapshot_adapter.state_dict(),
        },
        "r.pt": {
            "algorithm": "resfit",
            "base_checkpoint_sha256": base_hash,
            "executed_prefix_steps": 8,
        },
    }
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: snapshots[path.name])
    actor = nn.Identity()
    monkeypatch.setattr(actor, "to", lambda device: actor)
    monkeypatch.setattr(
        paired_eval, "make_residual_actor", lambda snapshot, device: actor
    )
    monkeypatch.setattr(paired_eval, "require_campaign_lease", lambda device: None)
    model = nn.Module()
    model.final_layer = nn.Module()
    model.final_layer.linear = base
    policy = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        action_dim=14,
        chunk_length=30,
        norm_stats={"actions": {"mean": np.zeros(14), "std": np.ones(14)}},
    )
    monkeypatch.setattr(abc_minimal.policy, "DiTInferencePolicy", lambda *args: policy)
    monkeypatch.setattr(
        abc_minimal.eval_policy, "resolve_prompt", lambda config: "synthetic"
    )
    closed = []
    monkeypatch.setattr(
        abc_sim,
        "make_env",
        lambda **kwargs: SimpleNamespace(close=lambda: closed.append(True)),
    )

    def failed_writer(*args, **kwargs):
        raise ValueError("Synthetic video writer failure")

    monkeypatch.setattr(imageio, "get_writer", failed_writer)
    args = Namespace(
        device="cuda:0",
        checkpoint=checkpoint,
        qf3_state=tmp_path / "q.pt",
        resfit_state=tmp_path / "r.pt",
        task="put_plastic_bottles_in_bin",
        output=tmp_path / "result" / "paired.json",
        seeds=[101],
        no_video=False,
        horizon=1000,
    )
    with pytest.raises(ValueError, match="Synthetic video writer failure"):
        paired_eval.run(args)
    assert closed == [True]
