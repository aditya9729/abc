"""Qualify the original ABC-VLA evaluation recipe under the GPU 0 lease.

This worker calls the unchanged ``abc_minimal.eval_policy.run_eval``. The
qualification keeps the original horizon and inference settings, but reduces
the default five-world cohort to one world. It is not a full published benchmark.
The stock sequential path uses CPU MuJoCo physics and MJWarp cameras. FastTD3's
planned batched MJWarp host is a separate simulator execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from abc_bench.subset import require_subset_lease

RECIPE = {
    "policy": "vla",
    "task": "put_plastic_bottles_in_bin",
    "num_worlds": 1,
    "seed": 20260511,
    "num_chunks": 236,
    "execute_chunk_dim": 15,
    "prefix_length": None,
    "diffusion_steps": 10,
    "policy_seed": 0,
    "camera_height": 168,
    "camera_width": 224,
    "device": "cuda:0",
    "gpu_id": 0,
    "camera_backend": "mjwarp",
    "parallel_worlds": 0,
    "randomization": None,
    "fast_inference": True,
    "fast_compile_mode": "max-autotune",
    "vanilla_physics": False,
    "rtc": True,
    "rtc_prefix_length": 4,
    "rtc_inference_lead_steps": 4,
    "log_every_chunk": True,
    "save_video": True,
    "video_fps": 30,
    "video_every_n_actions": 1,
    "prompt": "sim throw plastic bottles in bin",
}
RESOURCE_ENV = (
    "TORCHINDUCTOR_COMPILE_THREADS",
    "MAX_JOBS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
)


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON values must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _finite(value)
    if type(value) is not dict:
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _pin(value: Any, name: str, maximum: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} needs path, bytes, and sha256")
    if type(value["path"]) is not str:
        raise ValueError(f"{name}.path must be an existing absolute file")
    path = Path(value["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{name}.path must be an existing absolute file")
    size = value["bytes"]
    if type(size) is not int or not 0 < size <= maximum:
        raise ValueError(f"{name}.bytes must be positive and at most {maximum}")
    if path.stat().st_size != size:
        raise ValueError(f"{name} size differs from its pin")
    if type(value["sha256"]) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", value["sha256"]
    ):
        raise ValueError(f"{name}.sha256 must contain 64 lowercase hexadecimal digits")
    return {**value, "path": str(path.resolve())}


def artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _model_config(value: Any) -> dict[str, Any]:
    """Bound metadata architecture before loading any native/model module."""
    if type(value) is not dict or set(value) != {"camera_keys", "backbone", "dit"}:
        raise ValueError(
            "Metadata requires camera_keys, backbone, and dit model_config"
        )
    if value["camera_keys"] != ["top", "left", "right"]:
        raise ValueError("Metadata must specify the original top/left/right cameras")
    backbone, dit = value["backbone"], value["dit"]
    if type(backbone) is not dict or type(dit) is not dict:
        raise ValueError("Metadata backbone and dit must be objects")
    backbone_types = {
        "checkpoint": type(None),
        "load_base_checkpoint": bool,
        "image_size": int,
        "fixed_seq_len": int,
        "feature_layer": int,
        "train_backbone": bool,
        "train_siglip": bool,
        "activation_checkpointing": bool,
        "proprio_token": bool,
        "train_token_embedding": bool,
    }
    dit_types = {
        "hidden_size": int,
        "depth": int,
        "num_heads": int,
        "mlp_ratio": (int, float),
        "state_dim": int,
        "action_dim": int,
        "chunk_length": int,
        "num_pool_tokens": int,
        "pool_num_heads": int,
        "pool_qk_norm": bool,
        "direct_state_conditioning": bool,
    }
    for section, types in ((backbone, backbone_types), (dit, dit_types)):
        required = set(types) - {"train_token_embedding"}
        if not required <= set(section) or set(section) - set(types):
            raise ValueError("Metadata architecture keys differ from the original VLA")
        for name, item in section.items():
            expected = types[name]
            allowed = expected if type(expected) is tuple else (expected,)
            if type(item) not in allowed:
                raise ValueError(f"Metadata model_config.{name} has an invalid type")
            if type(item) in (int, float) and not -128 <= item <= 4096:
                raise ValueError(f"Metadata model_config.{name} exceeds bounded limits")
    if (
        dit["state_dim"] != 14
        or dit["action_dim"] != 14
        or dit["chunk_length"] != 30
        or backbone["image_size"] != 224
        or backbone["fixed_seq_len"] != 256
        or any(dit[name] <= 0 for name in dit if type(dit[name]) in (int, float))
    ):
        raise ValueError(
            "Metadata must describe the original 14D, 30-action, 224px VLA"
        )
    return value


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the finite one-world qualification before native imports."""
    config = _read_json(path)
    if set(config) != {"schema_version", "checkpoint", "metadata", "sim_eval"}:
        raise ValueError("Use schema_version, checkpoint, metadata, and sim_eval only")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Expected original ABC qualification schema_version 1")
    sim = config["sim_eval"]
    if type(sim) is not dict or set(sim) != set(RECIPE):
        raise ValueError(
            "sim_eval must specify the complete original qualification recipe"
        )
    for name, expected in RECIPE.items():
        if type(sim[name]) is not type(expected) or sim[name] != expected:
            raise ValueError(
                f"sim_eval.{name} must be {expected!r} for this qualification"
            )
    config["checkpoint"] = _pin(config["checkpoint"], "checkpoint", 32 * 1024**3)
    config["metadata"] = _pin(config["metadata"], "metadata", 1024**2)
    metadata_path = Path(config["metadata"]["path"])
    if artifact(metadata_path) != config["metadata"]:
        raise ValueError("Metadata SHA-256 differs from its pin")
    metadata = _read_json(metadata_path)
    if (
        metadata.get("format") != "abc_checkpoint_metadata/v1"
        or type(metadata.get("bytes")) is not int
        or metadata["bytes"] != config["checkpoint"]["bytes"]
        or metadata.get("sha256") != config["checkpoint"]["sha256"]
        or type(metadata.get("max_action_prefix")) is not int
        or metadata["max_action_prefix"] <= 4
    ):
        raise ValueError("Metadata does not admit the pinned checkpoint and RTC prefix")
    _model_config(metadata.get("model_config"))
    return config, metadata


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config, metadata = load_config(config_path)
    require_subset_lease()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "receipt.json").exists() or (output / "original_eval").exists():
        raise ValueError("Output already contains an original ABC qualification")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "algorithm": "original ABC-VLA",
        "phase": "original_recipe_one_world_qualification",
        "claim_scope": "Reduced one-world cohort; not the full published benchmark",
        "upstream_default_num_worlds": 5,
        "maximum_controls_per_world": 3540,
        "input_config": artifact(config_path),
        "checkpoint": config["checkpoint"],
        "metadata": config["metadata"],
        "compile_resource_environment": {
            name: os.environ[name] for name in RESOURCE_ENV if name in os.environ
        },
        "compile_resource_note": (
            "Stock fast_inference=True and max-autotune are retained. Listed environment "
            "values are resource overrides. RTC video includes rendering in timing."
        ),
        "metrics": {},
        "blockers": [],
    }
    started = time.monotonic()
    _write_json(output / "receipt.json", receipt)
    try:
        if artifact(Path(config["checkpoint"]["path"])) != config["checkpoint"]:
            raise ValueError("Checkpoint SHA-256 differs from its pin")
        # No native import occurs before configuration, lease, and artifact admission.
        from abc_minimal import config as original_config
        from abc_minimal import eval_policy

        model = metadata["model_config"]
        vla = original_config.VLAModelConfig(
            camera_keys=tuple(model["camera_keys"]),
            backbone=original_config.GemmaVLAConfig(**model["backbone"]),
            dit=original_config.VLADiTConfig(**model["dit"]),
        )
        sim = original_config.SimEvalConfig(
            checkpoint=config["checkpoint"]["path"],
            output_dir=str(output / "original_eval"),
            vla_model=vla,
            **config["sim_eval"],
        )
        _write_json(output / "requested_sim_eval.json", asdict(sim))
        eval_policy.run_eval(sim)
        summary_path = output / "original_eval" / "summary.json"
        summary = _read_json(summary_path)
        if summary.get("num_worlds") != 1 or len(summary.get("worlds", [])) != 1:
            raise RuntimeError(
                "Original summary did not contain the declared one world"
            )
        world = summary["worlds"][0]
        if type(world.get("steps")) is not int or not 0 <= world["steps"] <= 3540:
            raise RuntimeError("Original summary exceeded the declared control horizon")
        _write_json(output / "resolved_sim_eval.json", summary["config"])
        receipt["resolved_config"] = artifact(output / "resolved_sim_eval.json")
        receipt["original_summary"] = artifact(summary_path)
        receipt["original_videos"] = [artifact(Path(world["video_path"]))]
        receipt["metrics"] = {
            name: summary[name]
            for name in (
                "num_success",
                "num_worlds",
                "success_rate",
                "mean_reward",
                "mean_max_progress",
                "mean_max_bottles_in_bin",
            )
            if name in summary
        }
        receipt["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        receipt["status"] = (
            "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        receipt["blockers"] = [f"{type(error).__name__}: {error}"]
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        _write_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
