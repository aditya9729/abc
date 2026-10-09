"""Explicit public-file diagnostic routing with CPU-only protocol fixtures."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from abc_bench import runner


def setup_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", "/home/user/aditya/RL/abc/.venv")
    root = tmp_path / "root"
    root.mkdir()
    campaign = root / "outputs/bench"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RESULTS", campaign)
    monkeypatch.setattr(runner, "HEAD_CHECK_CAMPAIGN", campaign)
    child = tmp_path / "public_child.py"
    child.write_text("# explicit CPU protocol fixture\n")
    monkeypatch.setattr(runner, "HEAD_CHECK_CHILD", child)
    monkeypatch.setattr(
        runner,
        "HEAD_CHECK_CHILD_SHA256",
        hashlib.sha256(child.read_bytes()).hexdigest(),
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "stage": "head_check",
                "device": "cuda:0",
                "campaign_directory": str(campaign),
                "max_wall_s": 100,
            }
        )
    )
    args = argparse.Namespace(
        algorithm="qf3-head-check",
        results=campaign,
        training_config=config,
        head_check_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        timeout_seconds=200,
        checkpoint=tmp_path / "not_read.pt",
        task="put_plastic_bottles_in_bin",
        seed=1,
        worlds=1,
        chunks=1,
        video=False,
        eval_horizon=1000,
    )
    return args, child


def test_exact_public_child_route_and_boundary_refusal(monkeypatch, tmp_path):
    args, child = setup_plan(monkeypatch, tmp_path)
    config, command = runner.head_check_plan(args, 200)
    assert command == [
        sys.executable,
        str(child),
        "--config",
        str(args.training_config),
    ]
    assert config["max_wall_s"] == 100
    args.results = tmp_path / "disconnected"
    with pytest.raises(ValueError, match="canonical"):
        runner.run_baseline(args)
    assert not args.results.exists()
    args.results = runner.RESULTS
    args.head_check_config_sha256 = "0" * 64
    with pytest.raises(ValueError, match="config pin"):
        runner.head_check_plan(args, 200)
    args.head_check_config_sha256 = hashlib.sha256(
        args.training_config.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="cleanup reserve"):
        runner.head_check_plan(args, 159)
    child.write_text("changed")
    with pytest.raises(ValueError, match="child source"):
        runner.head_check_plan(args, 200)


@pytest.mark.parametrize("bad", [False, True])
def test_existing_accounted_controller_consumes_diagnostic_only(
    monkeypatch, tmp_path, bad
):
    args, _ = setup_plan(monkeypatch, tmp_path)
    runner.RESULTS.mkdir(parents=True)
    ledger = runner.RESULTS / "gpu_budget.json"
    ledger.write_text(
        json.dumps({"limit_seconds": 1000, "charged_seconds": 100, "gpu": 0})
    )
    monkeypatch.setattr(
        runner, "inspect_reserved_gpu", lambda _: {"uuid": "fixture-only"}
    )
    monkeypatch.setattr(runner, "execution_provenance", dict)
    monkeypatch.setattr(runner, "publish", lambda *_: None)

    def execute(command, log, timeout, *, gpu_uuid):
        assert command[0] == sys.executable and command[1] == str(
            runner.HEAD_CHECK_CHILD
        )
        assert gpu_uuid == "fixture-only" and timeout == 100
        assert json.loads(ledger.read_text())["charged_seconds"] == 300
        out = Path(command[-1])
        out.mkdir()
        receipt = {
            "kind": "qf3_head_diagnostic",
            "diagnostic_config_sha256": args.head_check_config_sha256,
            "stage": "head_check",
            "status": "completed",
            "child_source_sha256": runner.HEAD_CHECK_CHILD_SHA256,
            "performance_claim": False,
            "metrics": {
                "simulation_steps": 1 if bad else 0,
                "optimizer_steps": 0,
                "critic_updates": 0,
                "actor_updates": 0,
                "worlds_created": 0,
            },
        }
        (out / "receipt.json").write_text(json.dumps(receipt))
        return 0

    monkeypatch.setattr(runner, "execute_command", execute)
    result = runner.run_baseline(args)
    assert result["status"] == ("failed" if bad else "completed")
    assert result["budget"]["steps"] == 0
    if not bad:
        assert result["metrics"]["simulation_steps"] == 0
        assert result["metrics"]["successes"] is None
    assert json.loads(ledger.read_text())["charged_seconds"] >= 100
