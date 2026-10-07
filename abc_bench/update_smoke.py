"""Bounded actual-policy optimizer smoke; never a converged method benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from abc_bench.algorithms import ResFiTFeatureLearner, qf3_actor_loss


class OutputAdapter(nn.Module):
    """Zero-start rank-four output adapter around an immutable linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 4):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.up = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        nn.init.zeros_(self.up.weight)
        self.enabled = True

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.base(inputs)
        return result + self.up(self.down(inputs)) if self.enabled else result


@dataclass(frozen=True)
class ActionCodec:
    """Tanh of checkpoint z-score; exact inverse away from floating-point edges."""

    mean: np.ndarray
    std: np.ndarray

    def encode(self, physical: np.ndarray) -> np.ndarray:
        result = np.tanh((physical - self.mean) / (self.std + 1e-6))
        if not np.isfinite(result).all() or (np.abs(result) >= 1).any():
            raise ValueError("Physical action saturated the invertible tanh codec")
        return result.astype(np.float32)

    def decode(self, coded: np.ndarray) -> np.ndarray:
        if not np.isfinite(coded).all() or (np.abs(coded) >= 1).any():
            raise ValueError("Coded actions must be finite and strictly inside (-1,1)")
        return (np.arctanh(coded) * (self.std + 1e-6) + self.mean).astype(np.float32)


def discounted_return(rewards: list[float], gamma: float) -> float:
    return float(sum(gamma**i * reward for i, reward in enumerate(rewards)))


def require_campaign_lease(device: str) -> None:
    """GPU smoke workers must be spawned by the budget-owning coordinator."""
    if not device.startswith("cuda"):
        return
    import fcntl
    import os

    from abc_bench.runner import RESULTS

    campaign = RESULTS
    ledger = json.loads((campaign / "gpu_budget.json").read_text())
    if ledger.get("active_parent_pid") != os.getppid():
        raise RuntimeError(
            "Use python -m abc_bench.runner --algorithm qf3|resfit for GPU smoke"
        )
    with (campaign / ".gpu-budget.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise RuntimeError("GPU campaign lease is not held")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from abc_minimal.config import SimEvalConfig
    from abc_minimal.eval_policy import resolve_prompt
    from abc_minimal.policy import DiTInferencePolicy
    from abc_minimal.preprocess import normalize, resize_pad_normalize
    from abc_sim import make_env

    start = time.monotonic()
    deadline = start + args.max_seconds

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("Update smoke deadline reached")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = SimEvalConfig(checkpoint=str(args.checkpoint), task=args.task)
    prompt = resolve_prompt(config)
    config.prompt = prompt
    with args.checkpoint.open("rb") as checkpoint_file:
        base_checkpoint_sha256 = hashlib.file_digest(
            checkpoint_file, "sha256"
        ).hexdigest()
    policy = DiTInferencePolicy(args.checkpoint, config, args.device)
    model = policy.model.eval().requires_grad_(False)
    adapter = OutputAdapter(model.final_layer.linear)
    model.final_layer.linear = adapter
    adapter.down.requires_grad_(True)
    adapter.up.requires_grad_(True)
    codec = ActionCodec(
        **{key: policy.norm_stats["actions"][key] for key in ("mean", "std")}
    )
    env = make_env(
        task=args.task,
        prompt=prompt,
        render_cameras=True,
        camera_height=224,
        camera_width=224,
        max_episode_steps=None,
        terminate_on_success=False,
    )
    records = []
    infer_times = []
    executed_steps = 0

    @torch.no_grad()
    def conditioning(obs: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
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
        tokens = model.build_vision_tokens(images).detach()
        return state, tokens, policy.task_vec.detach().clone()

    @torch.no_grad()
    def nominal(obs: dict[str, Any]) -> np.ndarray:
        check_deadline()
        adapter.enabled = False
        try:
            torch.cuda.synchronize() if policy.device.type == "cuda" else None
            before = time.monotonic()
            result = policy.infer(obs)
            torch.cuda.synchronize() if policy.device.type == "cuda" else None
            infer_times.append(time.monotonic() - before)
            return result[: args.executed_steps].copy()
        finally:
            adapter.enabled = True

    try:
        obs, _ = env.reset(seed=args.seed)
        action = nominal(obs)
        current = conditioning(obs)
        for _ in range(args.collect_chunks):
            rewards = []
            applied = []
            for command in action:
                check_deadline()
                obs, reward, terminated, truncated, _ = env.step(command)
                applied.append(command.copy())
                rewards.append(float(reward))
                executed_steps += 1
                if terminated or truncated:
                    raise RuntimeError(
                        "Unexpected early boundary; incomplete chunk is not fabricated"
                    )
            next_condition = conditioning(obs)
            next_action = nominal(obs)
            records.append(
                (
                    current,
                    np.stack(applied),
                    discounted_return(rewards, args.gamma),
                    args.gamma ** len(applied),
                    next_condition,
                    next_action.copy(),
                )
            )
            current, action = next_condition, next_action

        state = torch.cat([record[0][0] for record in records])
        tokens = torch.cat([record[0][1] for record in records])
        task = torch.cat([record[0][2] for record in records])
        features = torch.cat((state, tokens.mean(1)), -1).detach()
        next_features = torch.cat(
            [torch.cat((record[4][0], record[4][1].mean(1)), -1) for record in records]
        ).detach()
        physical = np.stack([record[1] for record in records])
        normalized = torch.as_tensor(
            normalize(physical, policy.norm_stats["actions"]), device=policy.device
        )
        next_physical = np.stack([record[5] for record in records])
        rewards = torch.tensor([record[2] for record in records], device=policy.device)
        discounts = torch.tensor(
            [record[3] for record in records], device=policy.device
        )
        updates = []
        zero_equivalence = True
        gradient_finite = True
        if args.algorithm == "qf3":
            critic = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(features.shape[1] + normalized[0].numel(), 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    )
                    for _ in range(2)
                ]
            ).to(policy.device)
            critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
            actor_opt = torch.optim.Adam(
                [*adapter.down.parameters(), *adapter.up.parameters()], lr=1e-4
            )
            initial = adapter.up.weight.detach().clone()
            for _ in range(args.updates):
                check_deadline()
                # Explicit one-step fitted critic smoke: no unverified future-Q bootstrap.
                q = torch.cat(
                    [
                        c(torch.cat((features, normalized.flatten(1)), -1))
                        for c in critic
                    ],
                    -1,
                )
                critic_loss = (q - rewards[:, None]).square().mean()
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_opt.step()
                critic_opt.zero_grad(set_to_none=True)
                critic.requires_grad_(False)
                noise = torch.randn_like(normalized)
                t = torch.rand(len(records), 1, 1, device=policy.device)
                interpolation = normalized * (1 - t) + noise * t
                with torch.no_grad():
                    cond = model.compute_cond(state, task, t.reshape(-1))
                    adapter.enabled = False
                    base_velocity = model.predict_velocity(
                        interpolation, cond, tokens
                    ).detach()
                    adapter.enabled = True
                velocity = model.predict_velocity(interpolation, cond, tokens)
                if not updates:
                    zero_equivalence = bool(
                        torch.equal(velocity.detach(), base_velocity)
                    )

                def conservative(endpoint: Tensor) -> Tensor:
                    return (
                        torch.cat(
                            [
                                c(torch.cat((features, endpoint.flatten(1)), -1))
                                for c in critic
                            ],
                            -1,
                        )
                        .min(-1)
                        .values
                    )

                loss, metrics = qf3_actor_loss(
                    normalized,
                    noise,
                    velocity,
                    t,
                    conservative,
                    base_velocity=base_velocity,
                    base_weight=1.0,
                    reverse_time=True,
                )
                actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [p.grad for p in adapter.parameters() if p.requires_grad]
                gradient_finite &= all(
                    g is not None and bool(torch.isfinite(g).all()) for g in gradients
                )
                if not gradient_finite or not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite QF3 adapter update")
                actor_opt.step()
                critic.requires_grad_(True)
                metrics["critic_loss"] = float(critic_loss.detach())
                updates.append(metrics)
            parameter_delta = float((adapter.up.weight.detach() - initial).norm())

            def updated(obs: dict[str, Any]) -> np.ndarray:
                return policy.infer(obs)[: args.executed_steps]
        else:
            coded = codec.encode(physical)
            next_coded = codec.encode(next_physical)
            learner = ResFiTFeatureLearner(
                features.shape[1],
                coded[0].size,
                hidden=64,
                device=str(policy.device),
                seed=args.seed,
            )
            batch = {
                "obs": features,
                "next_obs": next_features,
                "action": torch.as_tensor(coded, device=policy.device).flatten(1),
                "nominal": torch.as_tensor(coded, device=policy.device).flatten(1),
                "next_nominal": torch.as_tensor(
                    next_coded, device=policy.device
                ).flatten(1),
                "reward": rewards,
                "discount": discounts,
            }
            initial = learner.actor[-1].weight.detach().clone()
            zero_equivalence = bool(
                torch.equal(
                    learner.action(features, batch["nominal"]), batch["nominal"]
                )
            )
            for _ in range(args.updates):
                check_deadline()
                metrics = learner.update(batch)
                available_gradients = [
                    parameter.grad
                    for module in (learner.actor, learner.critics)
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ]
                gradient_finite &= bool(available_gradients) and all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in available_gradients
                )
                if not gradient_finite:
                    raise RuntimeError("Non-finite ResFiT gradient")
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise RuntimeError("Non-finite ResFiT update")
                updates.append(metrics)
            parameter_delta = float(
                (learner.actor[-1].weight.detach() - initial).norm()
            )

            def updated(obs: dict[str, Any]) -> np.ndarray:
                raw = nominal(obs)
                st, tok, _ = conditioning(obs)
                feature = torch.cat((st, tok.mean(1)), -1)
                proposal = learner.action(
                    feature,
                    torch.as_tensor(codec.encode(raw), device=policy.device).reshape(
                        1, -1
                    ),
                )
                coded_action = proposal.cpu().numpy().reshape(raw.shape)
                # Learner's [-1,1] clamp can hit edges; fail closed instead of fake inverse.
                return codec.decode(coded_action)

        if not zero_equivalence or parameter_delta <= 0:
            raise RuntimeError(
                "Zero-adapter equivalence or actual parameter update failed"
            )
        adaptation_path = args.output.parent / "adaptation.pt"
        adaptation_path.parent.mkdir(parents=True, exist_ok=True)
        if args.algorithm == "qf3":
            state_artifact = {
                "adapter": adapter.state_dict(),
                "critic": critic.state_dict(),
                "actor_optimizer": actor_opt.state_dict(),
                "critic_optimizer": critic_opt.state_dict(),
            }
        else:
            state_artifact = {
                "actor": learner.actor.state_dict(),
                "critics": learner.critics.state_dict(),
                "target_actor": learner.target_actor.state_dict(),
                "target_critics": learner.target_critics.state_dict(),
                "actor_optimizer": learner.actor_optimizer.state_dict(),
                "critic_optimizer": learner.critic_optimizer.state_dict(),
                "steps": learner.steps,
                "learner_rng": learner.rng.get_state(),
            }
        state_artifact.update(
            {
                "algorithm": args.algorithm,
                "checkpoint": str(args.checkpoint.resolve()),
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "seed": args.seed,
                "executed_prefix_steps": args.executed_steps,
                "torch_rng": torch.get_rng_state(),
                "scope": "optimizer snapshot; environment/replay state absent; not exact continuation",
            }
        )
        temporary = adaptation_path.with_suffix(".tmp")
        torch.save(state_artifact, temporary)
        temporary.replace(adaptation_path)
        post_rewards = []
        for _ in range(args.eval_chunks):
            check_deadline()
            for command in updated(obs):
                obs, reward, terminated, truncated, _ = env.step(command)
                post_rewards.append(float(reward))
                executed_steps += 1
                if terminated or truncated:
                    raise RuntimeError("Unexpected post-update boundary")
        return {
            "schema_version": 1,
            "phase": "update_smoke",
            "algorithm": args.algorithm,
            "status": "completed",
            "embodiment": "native_yam",
            "seed": args.seed,
            "task": args.task,
            "prompt": prompt,
            "executed_steps": executed_steps,
            "collected_chunks": len(records),
            "executed_prefix_steps": args.executed_steps,
            "updates": updates,
            "zero_adapter_equivalence": zero_equivalence,
            "gradient_finite": gradient_finite,
            "parameter_delta_l2": parameter_delta,
            "collection_discounted_returns": [record[2] for record in records],
            "post_update_task_scores": post_rewards,
            "latency_ms": {
                "p50": float(np.percentile(infer_times, 50) * 1000),
                "p95": float(np.percentile(infer_times, 95) * 1000),
            },
            "latency_scope": "frozen nominal full-chunk inference; cold calls included",
            "elapsed_seconds": time.monotonic() - start,
            "checkpoint": str(args.checkpoint.resolve()),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "objective_config": {
                "qf3_clip": 1.0,
                "qf3_cfm_weight": 1.0,
                "qf3_base_weight": 1.0,
            }
            if args.algorithm == "qf3"
            else {"residual_scale": learner.scale, "actor_delay": 2},
            "camera_resolution": [224, 224],
            "adaptation_checkpoint": str(adaptation_path.resolve()),
            "adaptation_checkpoint_sha256": hashlib.sha256(
                adaptation_path.read_bytes()
            ).hexdigest(),
            "checkpoint_scope": "optimizer snapshot; not exact resume",
            "claim_scope": "optimizer and execution smoke only; no convergence or method superiority",
            "adaptations": [
                "executed prefix only",
                "frozen mean vision-token/state features",
                "no demonstration mixture",
                "QF3 rank-four final-layer adapter and immediate-return fitted critic"
                if args.algorithm == "qf3"
                else "ResFiT tanh z-score codec and small feature learner",
            ],
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("qf3", "resfit"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="put_plastic_bottles_in_bin")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--collect-chunks", type=int, default=4)
    parser.add_argument("--executed-steps", type=int, default=8)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--eval-chunks", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if not (
        1 <= args.collect_chunks <= 8
        and 1 <= args.executed_steps <= 8
        and 1 <= args.updates <= 8
        and 1 <= args.eval_chunks <= 2
        and 0 < args.gamma <= 1
        and 0 < args.max_seconds <= 600
    ):
        parser.error(
            "Smoke bounds: chunks/steps/updates <=8, eval <=2, seconds <=600, gamma in (0,1]"
        )
    if not args.checkpoint.is_file():
        parser.error("checkpoint does not exist")
    require_campaign_lease(args.device)
    receipt = run(args)
    receipt["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
