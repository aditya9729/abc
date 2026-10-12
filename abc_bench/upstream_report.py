"""Publish pinned native measurements; incomplete evaluations have no score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from abc_bench.subset_report import copy_video

# Reviewed adapter 0.1.2 evaluated with FP32 tensors and no author autocast.
# Bind that historical disclosure to its sealed source, not a mutable install.
LEGACY_FP32_LEARNER_SHA256 = (
    "fe3881d2ac5e3371b83bed7da26840dd281222e11696ac262363d1973478ff41"
)


def artifact_bytes(pin: dict) -> bytes:
    """Verify a producer's file identity before consuming its contents."""
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Recorded artifacts must be existing absolute files")
    if type(pin.get("bytes")) is not int or pin["bytes"] < 0:
        raise ValueError("Recorded artifact length must be a nonnegative integer")
    data = path.read_bytes()
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError(f"Recorded artifact differs: {path}")
    return data


def _identity(path: Path, data: bytes) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _json(data: bytes) -> dict:
    value = json.loads(data)

    def finite(item: Any) -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError("Recorded JSON numbers must be finite")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    if type(value) is not dict:
        raise ValueError("Recorded JSON must be an object")
    return value


def pinned_json(pin: dict) -> dict:
    return _json(artifact_bytes(pin))


def _integer(value: Any, name: str, minimum: int = 0, maximum: int = 1000000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}")
    return value


def _number(value: Any, name: str, minimum: float = 0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid {name}")
    return float(value)


def _pending(status: str = "pending") -> dict:
    return {"status": status, "successes": None, "episodes": None, "video": None}


def coordinated_receipt(
    run: Path, *, nested: bool = False
) -> tuple[dict, dict, dict] | None:
    """Authenticate a completed worker against the canonical coordinator seal.

    Coordinator receipts lack a declared worker byte count. The verified digest
    binds its complete bytes; this reporter records the measured length too.
    """
    run = run.resolve()
    parent = run.parent if nested and run.name == "expoft" else run
    worker_path = (parent / "expoft" if nested else parent) / "receipt.json"
    coordinator_path = parent.parent / "runs" / (parent.name + ".json")
    if not coordinator_path.is_file():
        return None
    coordinator = _json(coordinator_path.read_bytes())
    if (
        coordinator.get("schema_version") != 1
        or coordinator.get("run_id") != parent.name
    ):
        raise ValueError("Coordinator run identity differs")
    if coordinator.get("status") != "completed":
        return None
    if (
        coordinator.get("worker_exit_code") != 0
        or coordinator.get("task") != "put_plastic_bottles_in_bin"
        or coordinator.get("embodiment") != "native_yam"
    ):
        raise ValueError("Coordinator did not complete the native ABC job")
    job_bytes = (parent / "job.json").read_bytes()
    if hashlib.sha256(job_bytes).hexdigest() != coordinator.get("resolved_job_sha256"):
        raise ValueError("Coordinator job differs from its pin")
    data = worker_path.read_bytes()
    identity = _identity(worker_path, data)
    if identity["sha256"] != coordinator.get("worker_receipt_sha256"):
        raise ValueError("Worker receipt differs from its coordinator pin")
    worker = _json(data)
    if worker.get("schema_version") != 1 or worker.get("status") != "completed":
        raise ValueError("Worker did not complete its declared stage")
    return worker, identity, coordinator


def _baseline(run: Path) -> tuple[dict, dict, dict]:
    admitted = coordinated_receipt(run)
    if admitted is None:
        raise ValueError("A completed original ABC-VLA qualification is required")
    worker, identity, _ = admitted
    if (
        worker.get("algorithm") != "original ABC-VLA"
        or worker.get("phase") != "original_recipe_one_world_qualification"
    ):
        raise ValueError("A completed original ABC-VLA qualification is required")
    summary = pinned_json(worker["original_summary"])
    worlds = summary["worlds"]
    if (
        summary.get("task") != "put_plastic_bottles_in_bin"
        or summary["num_worlds"] != 1
        or len(worlds) != 1
    ):
        raise ValueError("This qualification viewer requires one complete world")
    world = worlds[0]
    success = world["success"]
    if type(success) is not bool or summary["num_success"] != int(success):
        raise ValueError("Original success count and world outcome disagree")
    if type(world["steps"]) is not int or not 0 <= world["steps"] <= 3540:
        raise ValueError("Invalid completed control count")
    video_pin = worker["original_videos"][0]
    if video_pin["path"] != world["video_path"]:
        raise ValueError("Original summary and video identity disagree")
    artifact_bytes(video_pin)
    return worker, identity, summary


def _fasttd3(run: Path | None, *, stage: str) -> tuple[dict, dict, dict, dict] | None:
    if run is None:
        return None
    admitted = coordinated_receipt(run)
    if admitted is None:
        return None
    worker, identity, coordinator = admitted
    if (
        worker.get("algorithm") != "original FastTD3 from scratch"
        or worker.get("phase") != stage
    ):
        raise ValueError("FastTD3 receipt has a different method or stage")
    learner = pinned_json(worker["learner_receipt"])
    host = pinned_json(worker["host_receipt"])
    if any(
        value.get("schema_version") != 1 or value.get("status") != "completed"
        for value in (learner, host)
    ):
        raise ValueError("FastTD3 child receipts must complete")
    if learner.get("condition") != "original_fasttd3_scratch_privileged_state_abc":
        raise ValueError("FastTD3 learner condition differs")
    config = pinned_json(worker["resolved_config"])
    if config.get("stage") != stage:
        raise ValueError("FastTD3 resolved configuration stage differs")
    pinned_json(worker["host_description"])
    # These are historical inventories. Verify their seals, not mutable installed files.
    pinned_json(worker["learner_module_snapshot"])
    pinned_json(host["original_sources_assets"])
    if learner["environment"].get("privileged_state") is not True:
        raise ValueError("FastTD3 must identify its privileged state inputs")
    if worker["metrics"] != coordinator["metrics"]:
        raise ValueError("FastTD3 coordinator and worker metrics disagree")
    return worker, identity, learner, host


def _training(
    admitted: tuple[dict, dict, dict, dict] | None, baseline_seed: int
) -> dict:
    if admitted is None:
        return _pending()
    worker, identity, learner, host = admitted
    controls = _integer(learner["controls"], "training controls", 1)
    actor = _integer(learner["actual_updates"]["actor"], "actor updates", 1)
    critic = _integer(learner["actual_updates"]["critic"], "critic updates", 1)
    if (
        worker["metrics"]
        != {"controls": controls, "actor_updates": actor, "critic_updates": critic}
        or host["metrics"]["controls"] != controls
        or sum(_integer(ep["controls"], "episode controls") for ep in host["episodes"])
        != controls
    ):
        raise ValueError("FastTD3 training counters disagree")
    if any(ep["seed"] == baseline_seed for ep in host["episodes"]):
        raise ValueError("FastTD3 training includes the matched evaluation scene")
    checkpoint = learner["checkpoint"]
    artifact_bytes(checkpoint)
    randomization = learner["environment"].get("reset_options", {}).get("randomization")
    if randomization is not None and (
        type(randomization) is not dict
        or any(
            type(randomization.get(name)) is not bool
            for name in ("randomize_variants", "randomize_scales")
        )
    ):
        raise ValueError("Invalid FastTD3 TRAIN randomization settings")
    episodes = []
    for episode in host["episodes"]:
        video = episode.get("video")
        if video is not None:
            artifact_bytes(video)
        episodes.append(
            {
                "seed": episode["seed"],
                "controls": episode["controls"],
                "completed": episode["completed"],
                "max_native_reward": _number(
                    episode["max_native_reward"], "training reward"
                ),
                "stop_reason": episode["end_reason"],
                "video_identity": video,
                "scene_metadata": episode.get("randomization", {}).get("metadata"),
                "scale_states": episode.get("randomization", {}).get("scale_states"),
            }
        )
    return {
        "status": "completed",
        "controls": controls,
        "actor_updates": actor,
        "critic_updates": critic,
        "elapsed_seconds": _number(worker["elapsed_seconds"], "training elapsed time"),
        "action_clipping_count": _integer(
            host["metrics"]["normalized_action_clipping_count"],
            "clipped action components",
        ),
        "checkpoint_identity": checkpoint,
        "scene_randomization": randomization,
        "worker_receipt_identity": identity,
        "episodes": episodes,
        "successes": None,
        "claim_scope": "Training rollout; no frozen task score or convergence claim",
    }


def _evaluation_precision(worker: dict, learner: dict) -> dict:
    profile = learner.get("evaluation_precision")
    if profile is None:
        inventory = pinned_json(worker["learner_module_snapshot"])
        if not any(
            module.get("sha256") == LEGACY_FP32_LEARNER_SHA256
            and Path(module.get("path", ""))
            .as_posix()
            .endswith("/sadhana_fasttd3_abc/learner.py")
            for module in inventory.get("modules", [])
        ):
            raise ValueError(
                "Evaluation precision is unrecorded for an unknown evaluator"
            )
        return {
            "label": "FP32 legacy diagnostic; author AMP evaluation was not used",
            "autocast_enabled": False,
            "policy_parameter_dtype": ["torch.float32"],
            "policy_action_dtype": ["torch.float32"],
            "environment_action_dtype": "torch.float32",
            "source_basis": "Reviewed historical learner source SHA256",
            "learner_sha256": LEGACY_FP32_LEARNER_SHA256,
        }
    if type(profile) is not dict:
        raise ValueError("Invalid evaluation precision record")
    for name in (
        "autocast_enabled",
        "configured_amp",
        "configured_cuda",
        "cuda_matmul_allow_tf32",
        "cudnn_deterministic",
        "cudnn_allow_tf32",
    ):
        if type(profile.get(name)) is not bool:
            raise ValueError(f"Invalid evaluation precision {name}")
    selector = profile.get("amp_dtype")
    dtype = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}.get(selector)
    if (
        dtype is None
        or profile.get("autocast_dtype") != dtype
        or profile.get("autocast_device_type") not in ("cuda", "cpu")
        or profile.get("float32_matmul_precision") not in ("highest", "high", "medium")
        or profile.get("environment_action_dtype") != "torch.float32"
    ):
        raise ValueError("Invalid evaluation precision dtype or device")
    valid_dtypes = {"torch.float32", "torch.bfloat16", "torch.float16"}
    for name in ("policy_parameter_dtype", "policy_action_dtype"):
        values = profile.get(name)
        if (
            type(values) is not list
            or not values
            or any(
                type(value) is not str or value not in valid_dtypes for value in values
            )
            or values != sorted(set(values))
        ):
            raise ValueError(f"Invalid evaluation precision {name}")
    if profile["autocast_enabled"] and not profile["configured_amp"]:
        raise ValueError("Evaluation autocast disagrees with configured AMP")
    return {
        **profile,
        "label": (
            selector.upper() + " autocast"
            if profile["autocast_enabled"]
            else "FP32 tensors; autocast disabled"
        ),
        "source_basis": "Recorded actual frozen evaluator precision",
    }


def _scene_distribution_changes(train: dict, world: dict) -> list[str]:
    """Describe observed changes; do not treat a held-out scene as in-distribution."""
    settings = train.get("scene_randomization") or {}
    randomization = world["randomization"]
    metadata = randomization.get("metadata") or {}
    train_variants = {
        variant
        for episode in train.get("episodes", [])
        for variant in (episode.get("scene_metadata") or {}).get("bottle_variants", [])
    }
    changes = []
    if (
        train_variants
        and settings.get("randomize_variants") is False
        and any(
            variant not in train_variants
            for variant in metadata.get("bottle_variants", [])
        )
    ):
        changes.append("bottle shapes")
    if settings.get("randomize_scales") is False and any(
        value != 1.0 for value in randomization.get("scale_states", {}).values()
    ):
        changes.append("object scales")
    return changes


def _evaluation(
    admitted: tuple[dict, dict, dict, dict] | None,
    train: dict,
    world: dict,
    summary_pin: dict,
) -> dict:
    if admitted is None:
        return _pending()
    worker, identity, learner, host = admitted
    if train["status"] != "completed":
        raise ValueError(
            "Frozen FastTD3 evaluation requires its completed TRAIN export"
        )
    for name in ("actor_updates", "critic_updates"):
        if type(learner.get(name)) is not int or learner[name] != 0:
            raise ValueError("FastTD3 evaluation performed learner updates")
    before, after = (
        learner.get("policy_state_before"),
        learner.get("policy_state_after"),
    )
    if not isinstance(before, str) or not before or before != after:
        raise ValueError("FastTD3 evaluation has no unchanged policy state fingerprint")
    checkpoint = learner["checkpoint"]
    artifact_bytes(checkpoint)
    artifact_bytes(worker["evaluation_checkpoint"])
    if any(
        pin["sha256"] != train["checkpoint_identity"]["sha256"]
        or pin["bytes"] != train["checkpoint_identity"]["bytes"]
        for pin in (checkpoint, worker["evaluation_checkpoint"])
    ):
        raise ValueError("FastTD3 evaluation checkpoint differs from the TRAIN export")
    config = pinned_json(worker["resolved_config"])
    reference = config["host"]["reference"]
    if (
        config["host"]["max_episode_steps"] != 3540
        or reference is None
        or reference["sha256"] != summary_pin["sha256"]
        or reference["world_index"] != 0
        or learner["evaluation_seeds"] != [world["world_seed"]]
        or len(learner["episodes"]) != 1
        or len(host["episodes"]) != 1
    ):
        raise ValueError(
            "FastTD3 evaluation is not the single full-horizon matched scene"
        )
    episode, actual = learner["episodes"][0], host["episodes"][0]
    if (
        episode.get("completed") is not True
        or type(episode.get("success")) is not bool
        or episode.get("seed") != world["world_seed"]
        or episode.get("scene_match") != "matched_baseline"
        or episode.get("randomization") != world["randomization"]
    ):
        raise ValueError(
            "FastTD3 episode is incomplete or has a different captured scene"
        )
    for name in (
        "seed",
        "controls",
        "completed",
        "success",
        "scene_match",
        "randomization",
        "final_native_task_eval",
        "video",
    ):
        if episode.get(name) != actual.get(name):
            raise ValueError("FastTD3 learner and host episode evidence disagree")
    controls = _integer(episode["controls"], "frozen controls", 1, 3540)
    if not episode["success"] and (
        controls != 3540 or episode["end_reason"] != "timeout"
    ):
        raise ValueError("Failed frozen evaluation must complete the full horizon")
    if (
        host["metrics"]["controls"] != controls
        or host["metrics"]["completed_episodes"] != 1
    ):
        raise ValueError("FastTD3 host did not complete exactly one evaluation")
    expected = {
        "num_worlds": 1,
        "successes": int(episode["success"]),
        "matched_worlds": 1,
        "actor_updates": 0,
        "critic_updates": 0,
    }
    if learner["successes"] != expected["successes"] or worker["metrics"] != expected:
        raise ValueError("FastTD3 frozen score and episode evidence disagree")
    evaluation = episode["final_native_task_eval"]
    if (
        evaluation["success"] != episode["success"]
        or evaluation["num_active_bottles"] != 5
    ):
        raise ValueError("FastTD3 native task judge differs from the scored episode")
    artifact_bytes(episode["video"])
    precision = _evaluation_precision(worker, learner)
    return {
        "status": "completed",
        "successes": expected["successes"],
        "episodes": 1,
        "seed": episode["seed"],
        "controls": controls,
        "bottles_in_bin": evaluation["num_bottles_in_bin"],
        "native_current_success": evaluation["success"],
        "return": _number(episode["return"], "evaluation return"),
        "max_native_reward": _number(episode["max_native_reward"], "evaluation reward"),
        "elapsed_seconds": _number(
            worker["elapsed_seconds"], "evaluation elapsed time"
        ),
        "evaluation_precision": precision,
        "scene_metadata": world["randomization"].get("metadata"),
        "scale_states": world["randomization"].get("scale_states"),
        "scene_distribution_changes": _scene_distribution_changes(train, world),
        "video_identity": episode["video"],
        "worker_receipt_identity": identity,
        "checkpoint_identity": checkpoint,
        "policy_state_before": before,
        "policy_state_after": after,
        "actor_updates": 0,
        "critic_updates": 0,
        "claim_scope": "One frozen matched scene; privileged-state scratch policy, not a method ranking",
    }


def _sft(run: Path | None) -> dict:
    if run is None:
        return _pending()
    admitted = coordinated_receipt(run, nested=True)
    if admitted is None:
        return _pending()
    worker, identity, coordinator = admitted
    if (
        worker.get("phase") != "original_pi05_sft_qualification"
        or worker["metrics"] != coordinator["metrics"]
    ):
        raise ValueError("pi0.5 SFT receipt phase or metrics disagree")
    metrics = worker["metrics"]
    if any(
        type(metrics.get(name)) is not bool
        for name in ("optimization_passed", "tiny_batch_loss_target_met")
    ):
        raise ValueError("Invalid SFT diagnostic qualification flags")
    return {
        "status": "completed",
        "actual_steps": _integer(worker["actual_steps"], "SFT updates", 1),
        "initial_fixed_loss": _number(
            metrics["initial_fixed_loss"], "initial TRAIN loss"
        ),
        "final_fixed_loss": _number(metrics["final_fixed_loss"], "final TRAIN loss"),
        "final_gradient_norm": _number(
            metrics["final_gradient_norm"], "SFT gradient norm"
        ),
        "optimization_passed": metrics["optimization_passed"],
        "tiny_batch_loss_target_met": metrics["tiny_batch_loss_target_met"],
        "stop_reason": worker["early_stopping_reason"],
        "elapsed_seconds": _number(worker["elapsed_s"], "SFT elapsed time"),
        "worker_receipt_identity": identity,
        "successes": None,
        "claim_scope": "Fixed-noise TRAIN loss diagnostic; no held-out validation or simulator task score",
    }


def _atomic_write(path: Path, contents: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".report-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(
    run: Path,
    output: Path,
    *,
    fasttd3_train: Path | None = None,
    fasttd3_eval: Path | None = None,
    pi05_sft: Path | None = None,
) -> dict:
    """Validate every displayed measurement before replacing the live report."""
    worker, identity, summary = _baseline(run)
    world = summary["worlds"][0]
    training = _training(_fasttd3(fasttd3_train, stage="train"), world["world_seed"])
    evaluation = _evaluation(
        _fasttd3(fasttd3_eval, stage="evaluate"),
        training,
        world,
        worker["original_summary"],
    )
    sft = _sft(pi05_sft)
    output.mkdir(parents=True, exist_ok=True)
    for episode in training.get("episodes") or []:
        pin = episode["video_identity"]
        episode["video"] = copy_video(pin, output) if pin is not None else None
    if evaluation["status"] == "completed":
        evaluation["video"] = copy_video(evaluation["video_identity"], output)
    result = {
        "schema_version": 1,
        "condition": "Original ABC-VLA with upstream evaluation settings",
        "successes": int(world["success"]),
        "episodes": 1,
        "seed": world["world_seed"],
        "controls": world["steps"],
        "bottles_in_bin": world["final_task_eval"]["num_bottles_in_bin"],
        "native_current_success": world["final_success"],
        "video": copy_video(worker["original_videos"][0], output),
        "video_identity": worker["original_videos"][0],
        "summary_identity": worker["original_summary"],
        "worker_receipt_sha256": identity["sha256"],
        "configuration": summary["config"],
        "claim_scope": "One original-recipe layout; not a paper score or a causal ablation",
        "methods": {
            "fasttd3": {"training": training, "evaluation": evaluation},
            "pi05": {
                "sft": sft,
                "evaluation": _pending(),
                "expoft_training": _pending(),
                "expoft_evaluation": _pending(),
            },
        },
    }
    _atomic_write(
        output / "results.json", json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    _atomic_write(output / "index.html", HTML)
    return result


HTML = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Original methods in ABC-Sim</title><style>
body{background:#101720;color:#e8edf3;font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}p{line-height:1.6;color:#c0ccd9}a{color:#9fc8ff}video{width:100%;background:#080c12}.card{background:#1a2532;border:1px solid #3b4b60;border-radius:12px;padding:20px;margin:20px 0}.score{font-size:28px;color:#94dcbb}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.grid .card{margin:0}@media(max-width:700px){.grid{grid-template-columns:1fr}}small{color:#b2bdcb}
</style><h1>Original methods · ABC-Sim</h1>
<p>The original policy and evaluation code ran in native ABC-Sim. The clip shows all three simulator cameras.</p>
<section class="card"><h2>Frozen ABC-VLA</h2><div id="score" class="score">Loading verified results…</div><p id="detail"></p><video id="video" controls preload="metadata"></video></section>
<p>Upstream settings: ten diffusion steps, RTC prefix 4 and lead 4, CPU MuJoCo physics, MJWarp cameras, and a maximum of 3,540 controls. The trained prompt remap is explicit.</p>
<p>This is one layout. The earlier comparison used different layouts, physics, sampling and RTC settings. This result does not isolate which change caused the difference.</p>
<div class="grid"><section class="card"><h2>Original FastTD3</h2><p>Privileged state policy trained from scratch. It receives object pose and velocity.</p><div id="fast-score" class="score">Task score pending</div><p id="fast-training">Training measurements pending.</p><p id="fast-evaluation">Frozen matched evaluation pending.</p><p id="fast-distribution"></p><div id="fast-clips"></div></section><section class="card"><h2>Original EXPO-FT · pi0.5</h2><p>Different pretrained image policy. Its task adaptation uses the local TRAIN sample.</p><div id="sft-status" class="score">Task adaptation pending</div><p id="sft-detail"></p><p>Frozen pi0.5 task evaluation and native EXPO-FT training are pending. No task score is available.</p></section></div>
<p>These conditions have different observations, backbones and training histories. This page reports the measured runs. It does not rank the methods or reproduce paper percentages.</p>
<p><a href="results.json">Measured result and source identities</a> · <a href="/small-comparison/">Preserved earlier diagnostic</a></p>
<script>
fetch('results.json').then(r=>{if(!r.ok)throw Error('Results unavailable');return r.json()}).then(d=>{
if(d.schema_version!==1||!Number.isInteger(d.successes)||d.episodes!==1||!/^[0-9a-f]{16}\\.mp4$/.test(d.video))throw Error('Invalid result');
document.querySelector('#score').textContent=d.successes+'/'+d.episodes+' successful episode';
document.querySelector('#detail').textContent='Layout '+d.seed+' · '+d.controls+' controls · '+d.bottles_in_bin+'/5 bottles currently in the bin · native success '+d.native_current_success;
document.querySelector('#video').src=d.video;
const methods=d.methods||{},fast=methods.fasttd3||{},train=fast.training||{},ev=fast.evaluation||{},sft=(methods.pi05||{}).sft||{};
const clips=document.querySelector('#fast-clips');
function clip(label,name){if(!/^[0-9a-f]{16}\\.mp4$/.test(name||''))return;const p=document.createElement('p');p.textContent=label;const v=document.createElement('video');v.controls=true;v.preload='metadata';v.src=name;clips.append(p,v);}
if(train.status==='completed'){
document.querySelector('#fast-training').textContent='TRAIN: '+train.controls+' controls · '+train.critic_updates+' critic updates · '+train.actor_updates+' actor updates · '+train.action_clipping_count+' clipped action components · '+train.elapsed_seconds.toFixed(1)+' seconds. Training clips are not task scores.';
(train.episodes||[]).forEach(e=>clip('TRAIN layout '+e.seed+' · '+e.controls+' controls · maximum native reward '+e.max_native_reward+' · '+e.stop_reason,e.video));
const settings=train.scene_randomization||{};
if(settings.randomize_variants===false&&settings.randomize_scales===false)document.querySelector('#fast-distribution').textContent='TRAIN uses fixed bottle shapes and scales.';
}
if(ev.status==='completed'){
document.querySelector('#fast-score').textContent=ev.successes+'/'+ev.episodes+' frozen matched episode';
const precision=(ev.evaluation_precision||{}).label||'precision not recorded';
document.querySelector('#fast-evaluation').textContent='Layout '+ev.seed+' · '+ev.controls+' controls · '+ev.bottles_in_bin+'/5 bottles currently in the bin · native success '+ev.native_current_success+' · zero learner updates · unchanged policy state · '+precision+' · '+ev.elapsed_seconds.toFixed(1)+' seconds for the complete stage, including simulator work.';
if((ev.scene_distribution_changes||[]).length)document.querySelector('#fast-distribution').textContent+=' The captured evaluation scene has different '+ev.scene_distribution_changes.join(' and ')+' from TRAIN. This changes the state distribution.';
clip('Frozen matched evaluation · '+precision,ev.video);
}
if(sft.status==='completed'){
document.querySelector('#sft-status').textContent=sft.actual_steps+' SFT update'+(sft.actual_steps===1?'':'s');
document.querySelector('#sft-detail').textContent='Fixed-noise TRAIN loss: '+sft.initial_fixed_loss.toFixed(6)+' → '+sft.final_fixed_loss.toFixed(6)+' · gradient norm '+sft.final_gradient_norm.toFixed(4)+' · stop: '+sft.stop_reason+' · '+sft.elapsed_seconds.toFixed(1)+' seconds. This measures training loss; it is not held-out validation or simulator success.';
}
}).catch(e=>{document.querySelector('#score').textContent=e.message;});
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fasttd3-train", type=Path)
    parser.add_argument("--fasttd3-eval", type=Path)
    parser.add_argument("--pi05-sft", type=Path)
    args = parser.parse_args()
    publish(
        args.run,
        args.output,
        fasttd3_train=args.fasttd3_train,
        fasttd3_eval=args.fasttd3_eval,
        pi05_sft=args.pi05_sft,
    )


if __name__ == "__main__":
    main()
