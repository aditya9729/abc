"""Matched native-YAM evaluation of saved four-update adaptation pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from abc_bench.update_smoke import ActionCodec, OutputAdapter, require_campaign_lease

METHODS = {
    "baseline": "Frozen ABC-DiT matched pilot",
    "qf3": "QF3 output-adapter matched pilot",
    "resfit": "ResFiT frozen-feature matched pilot",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


HISTORICAL_SNAPSHOTS = {
    "qf3": "5f9ebf32d4703605d9d852600e477a0bb237b2d64a6585ad4295d8e038f1da56",
    "resfit": "6bb9d9be21642175912fc905c31c2510be505ac3f15754a790363bc69e5bdd4b",
}


def validate_snapshot(
    snapshot: dict[str, Any],
    algorithm: str,
    base_hash: str,
    *,
    artifact_hash: str | None = None,
    manifest_base_hash: str | None = None,
) -> None:
    """Reject the wrong method, base checkpoint, or executed action contract."""
    if snapshot.get("algorithm") != algorithm:
        raise ValueError(f"Expected {algorithm} adaptation snapshot")
    embedded_hash = snapshot.get("base_checkpoint_sha256")
    if embedded_hash is None:
        if (
            artifact_hash != HISTORICAL_SNAPSHOTS.get(algorithm)
            or manifest_base_hash != base_hash
        ):
            raise ValueError(
                "Historical snapshot requires the accepted artifact and campaign manifest pins"
            )
    elif embedded_hash != base_hash:
        raise ValueError("Adaptation base checkpoint SHA-256 mismatch")
    if snapshot.get("executed_prefix_steps") != 8:
        raise ValueError("Matched evaluation requires an eight-action prefix snapshot")


def load_output_adapter(adapter: OutputAdapter, snapshot: dict[str, Any]) -> None:
    """Load only compatible adaptation weights; reject changed immutable base."""
    saved = snapshot["adapter"]
    expected = adapter.state_dict()
    if saved.keys() != expected.keys():
        raise ValueError("QF3 adapter state keys differ")
    for key, value in expected.items():
        if saved[key].shape != value.shape or not torch.isfinite(saved[key]).all():
            raise ValueError(f"Invalid QF3 adapter tensor {key}")
        if key.startswith("base.") and not torch.equal(saved[key].cpu(), value.cpu()):
            raise ValueError(f"QF3 snapshot changed immutable {key}")
    adapter.load_state_dict(saved, strict=True)
    adapter.requires_grad_(False)


def make_residual_actor(snapshot: dict[str, Any], device: str) -> nn.Sequential:
    """Restore the accepted 1550-feature, 112-action, hidden-64 actor only."""
    actor = nn.Sequential(
        nn.Linear(1550 + 112, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Linear(64, 112),
    )
    saved = snapshot["actor"]
    expected = actor.state_dict()
    if saved.keys() != expected.keys() or any(
        saved[key].shape != value.shape or not torch.isfinite(saved[key]).all()
        for key, value in expected.items()
    ):
        raise ValueError("ResFiT actor does not match the 1550/112/64 contract")
    actor.load_state_dict(saved, strict=True)
    return actor.to(device).eval().requires_grad_(False)


def summarize_worlds(
    worlds: list[dict[str, Any]], latencies: list[float]
) -> dict[str, Any]:
    """Keep success-only completion distinct from all-world episode duration."""
    if (
        not worlds
        or not latencies
        or any(not np.isfinite(x) or x < 0 for x in latencies)
    ):
        raise ValueError("Evaluation requires worlds and finite proposal latencies")
    successful = [
        world["simulation_duration_seconds"] for world in worlds if world["success"]
    ]
    durations = [world["simulation_duration_seconds"] for world in worlds]
    return {
        "successes": sum(bool(world["success"]) for world in worlds),
        "episodes": len(worlds),
        "simulation_steps": sum(world["steps"] for world in worlds),
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50) * 1000),
            "p95": float(np.percentile(latencies, 95) * 1000),
        },
        "latency_samples": len(latencies),
        "latency_scope": "whole action proposal, including ResFiT feature encoding; synchronized CUDA; cold calls included",
        "completion_time_seconds": float(np.mean(successful)) if successful else None,
        "completion_time_scope": "mean simulation time of successful episodes only",
        "episode_duration_seconds": {
            "mean": float(np.mean(durations)),
            "values": durations,
            "scope": "all episodes, including horizon failures",
        },
        "reward": float(np.mean([world["last_task_score"] for world in worlds])),
        "reward_scope": "mean terminal task score, not summed training return",
    }


DEPLOY_CODEC_EPSILON = 1e-4


def decode_residual_proposal(
    codec: ActionCodec,
    coded: np.ndarray,
    nominal: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Explicit new deployment saturation; the underlying codec stays strict."""
    if (
        coded.shape != nominal.shape
        or not np.isfinite(coded).all()
        or not np.isfinite(nominal).all()
    ):
        raise ValueError(
            "Residual proposal and nominal must have matching finite values"
        )
    bound = 1 - DEPLOY_CODEC_EPSILON
    saturated = int(np.count_nonzero(np.abs(coded) > bound))
    physical = codec.decode(np.clip(coded, -bound, bound))
    if not np.isfinite(physical).all():
        raise ValueError("Decoded residual proposal is non-finite")
    return physical, saturated, float(np.max(np.abs(physical - nominal)))


def persist_partial(
    output: Path,
    completed_runs: list[dict[str, Any]],
    current_run: dict[str, Any] | None,
    provenance: dict[str, Any],
) -> None:
    """Preserve measured completed worlds without claiming campaign acceptance."""
    receipt = {
        **provenance,
        "schema_version": 1,
        "status": "partial",
        "paired_initial_states_verified": False,
        "runs": completed_runs + ([current_run] if current_run is not None else []),
        "claim_scope": "Partial actual evaluation; only listed worlds completed; campaign may fail later",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def validate_evaluation_config(horizon: int, seeds: list[int]) -> None:
    """Share the runner horizon bound and reject duplicate paired episodes."""
    from abc_bench.runner import comparison_plan

    comparison_plan(horizon)
    if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Require one to three unique evaluation seeds")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_evaluation_config(args.horizon, args.seeds)
    if args.device != "cuda:0":
        raise RuntimeError("Rendered evaluation workers require leased cuda:0")
    # A worker cannot bypass the one-GPU campaign budget by direct invocation.
    require_campaign_lease(args.device)
    import imageio.v2 as imageio

    from abc_minimal.config import SimEvalConfig
    from abc_minimal.eval_policy import resolve_prompt
    from abc_minimal.policy import DiTInferencePolicy
    from abc_minimal.preprocess import normalize, resize_pad_normalize
    from abc_sim import make_env

    start = time.monotonic()
    base_hash = sha256(args.checkpoint)
    snapshots = {
        "qf3": torch.load(args.qf3_state, map_location="cpu", weights_only=True),
        "resfit": torch.load(args.resfit_state, map_location="cpu", weights_only=True),
    }
    from abc_bench.runner import RESULTS

    manifest = RESULTS / "runtime_manifest.json"
    manifest_content = json.loads(manifest.read_text())
    if manifest_content["checkpoint"]["sha256"] != base_hash:
        raise ValueError("Current base checkpoint differs from campaign manifest")
    for method, snapshot in snapshots.items():
        snapshot_path = args.qf3_state if method == "qf3" else args.resfit_state
        validate_snapshot(
            snapshot,
            method,
            base_hash,
            artifact_hash=sha256(snapshot_path),
            manifest_base_hash=manifest_content["checkpoint"]["sha256"],
        )
    # Validate the fixed residual actor shape before loading the large base model.
    actor = make_residual_actor(snapshots["resfit"], "cpu")
    config = SimEvalConfig(checkpoint=str(args.checkpoint), task=args.task)
    config.prompt = resolve_prompt(config)
    torch.manual_seed(0)
    policy = DiTInferencePolicy(args.checkpoint, config, args.device)
    model = policy.model.eval().requires_grad_(False)
    adapter = OutputAdapter(model.final_layer.linear)
    load_output_adapter(adapter, snapshots["qf3"])
    model.final_layer.linear = adapter
    actor = actor.to(args.device)
    codec = ActionCodec(
        **{key: policy.norm_stats["actions"][key] for key in ("mean", "std")}
    )
    if policy.action_dim != 14 or policy.chunk_length != 30:
        raise ValueError("Matched pilot requires ABC 30-by-14 action chunks")

    def synchronize() -> None:
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)

    deployment_saturated_elements = 0
    deployment_max_nominal_delta = 0.0

    @torch.no_grad()
    def propose(obs: dict[str, Any], method: str, noise: np.ndarray) -> np.ndarray:
        nonlocal deployment_saturated_elements, deployment_max_nominal_delta
        adapter.enabled = method == "qf3"
        action = policy.infer(obs, noise=noise)[:8].copy()
        if method != "resfit":
            return action
        state = torch.as_tensor(
            normalize(obs["state"], policy.norm_stats["state"]), device=policy.device
        ).reshape(1, -1)
        images = {
            camera: resize_pad_normalize(
                obs["images"][camera], preset=policy.norm_preset
            )
            .unsqueeze(0)
            .to(policy.device)
            for camera in model.camera_keys
        }
        features = torch.cat((state, model.build_vision_tokens(images).mean(1)), -1)
        nominal = torch.as_tensor(codec.encode(action), device=policy.device).reshape(
            1, -1
        )
        coded = (
            nominal + 0.2 * actor(torch.cat((features, nominal), -1)).tanh()
        ).clamp(-1, 1)
        physical, saturated, delta = decode_residual_proposal(
            codec,
            coded.cpu().numpy().reshape(8, 14),
            action,
        )
        deployment_saturated_elements += saturated
        deployment_max_nominal_delta = max(deployment_max_nominal_delta, delta)
        return physical

    runs = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint_sha256": base_hash,
        "seeds": args.seeds,
        "horizon_steps": args.horizon,
        "horizon_selection_scope": "exploratory longer-horizon diagnostic after the initial 1000-action pilot; not an untouched held-out evaluation"
        if args.horizon > 1000
        else "initial matched adaptation pilot",
        "runtime_manifest_sha256": sha256(manifest),
        "adaptation_artifacts": {
            method: {"path": str(path.resolve()), "sha256": sha256(path)}
            for method, path in (("qf3", args.qf3_state), ("resfit", args.resfit_state))
        },
        "resfit_deployment_codec_epsilon": DEPLOY_CODEC_EPSILON,
    }
    persist_partial(args.output, runs, None, provenance)
    for method, label in METHODS.items():
        method_start = time.monotonic()
        worlds = []
        latencies = []
        artifacts = []
        deployment_saturated_elements = 0
        deployment_max_nominal_delta = 0.0
        for seed_index, seed in enumerate(args.seeds):
            # Native reset preserves prior arm state. A fresh instance prevents leakage.
            env = make_env(
                task=args.task,
                prompt=config.prompt,
                render_cameras=True,
                camera_backend="mjwarp",
                camera_gpu_id=0,
                camera_height=224,
                camera_width=224,
                physics_dt=0.002,
                control_decimation=17,
                max_episode_steps=args.horizon,
                terminate_on_success=True,
            )
            writer = None
            try:
                if seed_index == 0 and not args.no_video:
                    video = args.output.parent / f"{method}-seed-{seed}.mp4"
                    writer = imageio.get_writer(
                        video, fps=1 / (0.002 * 17 * 8), codec="libx264"
                    )
                    artifacts.append(
                        {"label": f"{method} first-seed rollout", "path": video.name}
                    )
                if env.model.opt.timestep != 0.002 or env._control_decimation != 17:
                    raise ValueError(
                        "Resolved simulator timing differs from matched contract"
                    )
                obs, _ = env.reset(seed=seed)
                initial_state = env.get_state().tolist()
                initial_qpos_sha256 = hashlib.sha256(
                    env.capture_state()["qpos"].tobytes()
                ).hexdigest()
                rng = np.random.default_rng(seed)
                steps = 0
                chunk_index = 0
                success = False
                terminated = truncated = False
                task_score = 0.0
                world_start = time.monotonic()
                while not (terminated or truncated):
                    noise = rng.standard_normal((30, 14)).astype(np.float32)
                    synchronize()
                    proposal_start = time.monotonic()
                    action = propose(obs, method, noise)
                    synchronize()
                    latencies.append(time.monotonic() - proposal_start)
                    if action.shape != (8, 14) or not np.isfinite(action).all():
                        raise ValueError(
                            "Non-finite or incompatible proposed physical action"
                        )
                    if writer is not None:
                        writer.append_data(obs["images"]["top"].transpose(1, 2, 0))
                    for command in action:
                        obs, task_score, terminated, truncated, info = env.step(command)
                        steps += 1
                        success = bool(info.get("task_success", False))
                        if terminated or truncated:
                            break
                    chunk_index += 1
                worlds.append(
                    {
                        "seed": seed,
                        "success": success,
                        "steps": steps,
                        "chunks": chunk_index,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "last_task_score": float(task_score),
                        "simulation_duration_seconds": float(env.data.time),
                        "wall_seconds": time.monotonic() - world_start,
                        "initial_policy_state": initial_state,
                        "initial_qpos_sha256": initial_qpos_sha256,
                    }
                )
            finally:
                try:
                    if writer is not None:
                        writer.close()
                finally:
                    env.close()
            partial_metrics = summarize_worlds(worlds, latencies)
            partial_metrics["elapsed_seconds"] = time.monotonic() - method_start
            persist_partial(
                args.output,
                runs,
                {
                    "algorithm": label,
                    "method": method,
                    "status": "partial",
                    "embodiment": "native_yam",
                    "task": args.task,
                    "metrics": partial_metrics,
                    "worlds": worlds,
                    "artifacts": artifacts,
                    "proposal_latency_seconds": latencies,
                    "deployment_saturated_elements": deployment_saturated_elements,
                    "deployment_max_physical_nominal_delta": deployment_max_nominal_delta,
                },
                provenance,
            )
        metrics = summarize_worlds(worlds, latencies)
        metrics["elapsed_seconds"] = time.monotonic() - method_start
        runs.append(
            {
                "algorithm": label,
                "method": method,
                "status": "completed",
                "deployment_saturated_elements": deployment_saturated_elements,
                "deployment_max_physical_nominal_delta": deployment_max_nominal_delta,
                "embodiment": "native_yam",
                "task": args.task,
                "metrics": metrics,
                "worlds": worlds,
                "artifacts": artifacts,
                "proposal_latency_seconds": latencies,
            }
        )
        persist_partial(args.output, runs, None, provenance)
    # Check the paired initial states, including randomized task object poses.
    for index in range(len(args.seeds)):
        initial = {result["worlds"][index]["initial_qpos_sha256"] for result in runs}
        if len(initial) != 1:
            raise ValueError("Paired initial simulator states differ across methods")
    return {
        "schema_version": 1,
        "status": "completed",
        "paired_initial_states_verified": True,
        "resfit_deployment_codec_epsilon": DEPLOY_CODEC_EPSILON,
        "resfit_deployment_scope": "new interior clamp adaptation; differs from strict smoke at codec boundaries",
        "runs": runs,
        "seeds": args.seeds,
        "horizon_steps": args.horizon,
        "horizon_selection_scope": "exploratory longer-horizon diagnostic after the initial 1000-action pilot; not an untouched held-out evaluation"
        if args.horizon > 1000
        else "initial matched adaptation pilot",
        "physics_dt": 0.002,
        "control_decimation": 17,
        "camera_resolution": [224, 224],
        "execute_chunk_dim": 8,
        "prefix_conditioning": False,
        "noise_scope": "identical per-seed NumPy Gaussian sequence by proposal index for each method",
        "checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint_sha256": base_hash,
        "snapshot_base_binding": "embedded hash when present; accepted artifact hash plus external campaign manifest for historical snapshots",
        "adaptation_artifacts": {
            method: {"path": str(path.resolve()), "sha256": sha256(path)}
            for method, path in (("qf3", args.qf3_state), ("resfit", args.resfit_state))
        },
        "runtime_manifest": str(manifest),
        "runtime_manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        "elapsed_seconds": time.monotonic() - start,
        "claim_scope": "Matched evaluation of four-update adaptation pilots; not full-paper reproduction, converged training, or R1 transfer",
        "baseline_comparison_scope": "This baseline uses the same eight-action cadence, explicit noise, unprefixed sampler, cameras, and reset contract as both adaptations",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qf3-state", type=Path, required=True)
    parser.add_argument("--resfit-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--seeds",
        type=lambda text: [int(seed) for seed in text.split(",")],
        default=[101, 102, 103],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="put_plastic_bottles_in_bin")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    try:
        validate_evaluation_config(args.horizon, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if any(
        not path.is_file()
        for path in (args.checkpoint, args.qf3_state, args.resfit_state)
    ):
        parser.error("Checkpoint or adaptation snapshot is missing")
    receipt = run(args)
    receipt["code_sha256"] = sha256(Path(__file__))
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
