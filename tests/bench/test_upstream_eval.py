"""Explicit CPU dispatch doubles; these tests do not produce simulator scores."""

import builtins
import copy
import json
import sys
import types
from dataclasses import asdict, make_dataclass
from pathlib import Path

import pytest

from abc_bench import upstream_eval


@pytest.fixture
def qualification(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"explicit test checkpoint double")
    model = {
        "camera_keys": ["top", "left", "right"],
        "backbone": {
            "checkpoint": None,
            "load_base_checkpoint": True,
            "image_size": 224,
            "fixed_seq_len": 256,
            "feature_layer": -1,
            "train_backbone": True,
            "train_siglip": True,
            "activation_checkpointing": False,
            "proprio_token": True,
        },
        "dit": {
            "hidden_size": 512,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "state_dim": 14,
            "action_dim": 14,
            "chunk_length": 30,
            "num_pool_tokens": 8,
            "pool_num_heads": 8,
            "pool_qk_norm": True,
            "direct_state_conditioning": True,
        },
    }
    checkpoint_pin = upstream_eval.artifact(checkpoint)
    metadata = tmp_path / "fixture.json"
    metadata.write_text(
        json.dumps(
            {
                "format": "abc_checkpoint_metadata/v1",
                "bytes": checkpoint_pin["bytes"],
                "sha256": checkpoint_pin["sha256"],
                "max_action_prefix": 8,
                "model_config": model,
            }
        )
    )
    config = {
        "schema_version": 1,
        "checkpoint": checkpoint_pin,
        "metadata": upstream_eval.artifact(metadata),
        "sim_eval": copy.deepcopy(upstream_eval.RECIPE),
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(config))
    return path, config, model


def native_doubles(monkeypatch, model, evaluator):
    config_module = types.ModuleType("abc_minimal.config")
    config_module.GemmaVLAConfig = make_dataclass(
        "GemmaVLAConfigDouble", [(name, object) for name in model["backbone"]]
    )
    config_module.VLADiTConfig = make_dataclass(
        "VLADiTConfigDouble", [(name, object) for name in model["dit"]]
    )
    config_module.VLAModelConfig = make_dataclass(
        "VLAModelConfigDouble",
        [("camera_keys", tuple), ("backbone", object), ("dit", object)],
    )
    config_module.SimEvalConfig = make_dataclass(
        "SimEvalConfigDouble",
        [("checkpoint", str), ("output_dir", str), ("vla_model", object)]
        + [(name, object) for name in upstream_eval.RECIPE],
    )
    eval_module = types.ModuleType("abc_minimal.eval_policy")
    eval_module.run_eval = evaluator
    package = types.ModuleType("abc_minimal")
    package.config, package.eval_policy = config_module, eval_module
    monkeypatch.setitem(sys.modules, "abc_minimal", package)
    monkeypatch.setitem(sys.modules, "abc_minimal.config", config_module)
    monkeypatch.setitem(sys.modules, "abc_minimal.eval_policy", eval_module)


def forbid_native_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"abc_minimal", "torch", "mujoco", "mujoco_warp"}:
            raise AssertionError(f"Native import occurred before admission: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize(
    "change",
    [
        {"num_worlds": 5},
        {"num_worlds": True},
        {"num_chunks": 67},
        {"num_chunks": float("inf")},
        {"diffusion_steps": 5},
        {"rtc": False},
        {"rtc_prefix_length": 0},
        {"device": "cuda:1"},
        {"camera_backend": "mujoco"},
        {"prompt": "sim put plastic bottles in bin"},
        {"fast_inference": False},
        {"parallel_worlds": 1},
        {"output_dir": "/unexpected"},
    ],
)
def test_invalid_recipe_rejects_before_lease_or_native(
    qualification, monkeypatch, tmp_path, change
):
    path, config, _ = qualification
    config["sim_eval"].update(change)
    path.write_text(json.dumps(config))
    forbid_native_imports(monkeypatch)

    def unexpected_lease():
        raise AssertionError("Invalid configuration reached the lease")

    monkeypatch.setattr(upstream_eval, "require_subset_lease", unexpected_lease)
    with pytest.raises(ValueError):
        upstream_eval.run(path, tmp_path / "out")


def test_no_native_import_occurs_when_lease_rejects(
    qualification, monkeypatch, tmp_path
):
    path, _, _ = qualification
    forbid_native_imports(monkeypatch)

    def rejected_lease():
        raise RuntimeError("No active coordinator fixture")

    monkeypatch.setattr(upstream_eval, "require_subset_lease", rejected_lease)
    with pytest.raises(RuntimeError, match="No active coordinator"):
        upstream_eval.run(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", [True, -1, 1e999])
def test_bad_metadata_dimension_rejects_without_native(qualification, monkeypatch, bad):
    path, config, _ = qualification
    metadata_path = Path(config["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["model_config"]["dit"]["depth"] = bad
    metadata_path.write_text(json.dumps(metadata))
    config["metadata"] = upstream_eval.artifact(metadata_path)
    path.write_text(json.dumps(config))
    forbid_native_imports(monkeypatch)
    with pytest.raises(ValueError):
        upstream_eval.load_config(path)


def test_metadata_hash_mismatch_rejects(qualification):
    path, config, _ = qualification
    config["metadata"]["sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Metadata SHA-256"):
        upstream_eval.load_config(path)


def test_dispatch_preserves_recipe_and_records_original_resolved_config(
    qualification, monkeypatch, tmp_path
):
    path, config, model = qualification
    admitted = []
    monkeypatch.setattr(
        upstream_eval, "require_subset_lease", lambda: admitted.append(True)
    )
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")

    def original_eval_double(sim):
        assert admitted == [True]
        assert sim.num_chunks * sim.execute_chunk_dim == 3540
        assert sim.diffusion_steps == 10 and sim.rtc
        assert sim.rtc_prefix_length == sim.rtc_inference_lead_steps == 4
        assert sim.vla_model.camera_keys == ("top", "left", "right")
        assert asdict(sim.vla_model.dit) == model["dit"]
        assert asdict(sim.vla_model.backbone) == model["backbone"]
        out = Path(sim.output_dir)
        out.mkdir()
        video = out / "world_000.mp4"
        video.write_bytes(b"explicit video artifact double")
        resolved = asdict(sim)
        resolved["prefix_length"] = 0  # Original run_eval resolves this under RTC.
        summary = {
            "config": resolved,
            "num_worlds": 1,
            "num_success": 0,
            "success_rate": 0.0,
            "mean_reward": 0.2,
            "worlds": [{"steps": 3540, "video_path": str(video)}],
        }
        (out / "summary.json").write_text(json.dumps(summary))
        return summary

    native_doubles(monkeypatch, model, original_eval_double)
    out = tmp_path / "out"
    receipt = upstream_eval.run(path, out)
    assert receipt["status"] == "completed"
    assert receipt["checkpoint"] == config["checkpoint"]
    assert receipt["metadata"] == config["metadata"]
    assert receipt["metrics"]["success_rate"] == 0.0
    assert receipt["upstream_default_num_worlds"] == 5
    assert (
        receipt["compile_resource_environment"]["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    )
    assert (
        json.loads((out / "resolved_sim_eval.json").read_text())["prefix_length"] == 0
    )
    assert receipt["original_summary"] == upstream_eval.artifact(
        out / "original_eval" / "summary.json"
    )
    assert receipt["original_videos"][0]["path"].endswith("world_000.mp4")


def test_original_failure_keeps_failed_receipt_and_propagates(
    qualification, monkeypatch, tmp_path
):
    path, _, model = qualification
    monkeypatch.setattr(upstream_eval, "require_subset_lease", lambda: None)

    def failed_eval_double(_):
        raise RuntimeError("Explicit original evaluator failure double")

    native_doubles(monkeypatch, model, failed_eval_double)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="evaluator failure double"):
        upstream_eval.run(path, out)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["metrics"] == {}
    assert "evaluator failure double" in receipt["blockers"][0]
    assert "original_summary" not in receipt


def test_corrupt_checkpoint_does_not_call_original_evaluator(
    qualification, monkeypatch, tmp_path
):
    path, config, model = qualification
    config["checkpoint"]["sha256"] = "0" * 64
    metadata_path = Path(config["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    config["metadata"] = upstream_eval.artifact(metadata_path)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(upstream_eval, "require_subset_lease", lambda: None)

    def unexpected_eval(_):
        raise AssertionError("Corrupt checkpoint reached original evaluator")

    native_doubles(monkeypatch, model, unexpected_eval)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Checkpoint SHA-256"):
        upstream_eval.run(path, out)
    assert json.loads((out / "receipt.json").read_text())["status"] == "failed"


def test_native_import_failure_keeps_a_failed_receipt(
    qualification, monkeypatch, tmp_path
):
    path, _, _ = qualification
    monkeypatch.setattr(upstream_eval, "require_subset_lease", lambda: None)
    original_import = builtins.__import__

    def missing_native(name, *args, **kwargs):
        if name == "abc_minimal":
            raise ImportError("Explicit missing native dependency double")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_native)
    out = tmp_path / "out"
    with pytest.raises(ImportError, match="missing native dependency double"):
        upstream_eval.run(path, out)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["metrics"] == {}
