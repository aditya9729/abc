"""Auditable algorithm components. These are adaptations, not official trainers.

Actions use one declared normalized codec. Replay actions are executed full
actions, never residuals. No simulator or hardware is initialized on import.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn


def bellman_target(reward: Tensor, discount: Tensor, next_q: Tensor) -> Tensor:
    """Use an already composed gamma**ticks/nonterminal discount, shape (B,)."""
    if reward.shape != discount.shape or reward.shape != next_q.shape:
        raise ValueError("reward, discount and next_q shapes must match")
    if not all(torch.isfinite(x).all() for x in (reward, discount, next_q)):
        raise ValueError("Bellman inputs must be finite")
    if (discount < 0).any() or (discount > 1).any():
        raise ValueError("discount must be in [0, 1]")
    return (reward + discount * next_q).detach()


def qf3_endpoint(
    action: Tensor,
    noise: Tensor,
    velocity: Tensor,
    time: Tensor,
    clip: float,
    *,
    reverse_time: bool = False,
) -> Tensor:
    """Eq. 3, with ABC reverse-time support: v_ABC=noise-action, x0=x-t*v.

    time must broadcast over action (B,H,A), e.g. (B,1,1). The clamp
    preserves the actual autograd mask. It is not an actuator safety clamp.
    """
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("velocity clip must be positive and finite")
    if action.shape != noise.shape or action.shape != velocity.shape:
        raise ValueError("action, noise and velocity shapes must match")
    if not all(torch.isfinite(x).all() for x in (action, noise, velocity, time)):
        raise ValueError("flow inputs must be finite")
    if (time < 0).any() or (time > 1).any():
        raise ValueError("flow time must be in [0, 1]")
    target = noise - action if reverse_time else action - noise
    delta = (velocity - target).clamp(-clip, clip)
    return action - time * delta if reverse_time else action + (1 - time) * delta


def qf3_actor_loss(
    action: Tensor,
    noise: Tensor,
    velocity: Tensor,
    time: Tensor,
    critic: Callable[[Tensor], Tensor],
    *,
    clip: float = 1.0,
    cfm_weight: float = 1.0,
    base_velocity: Tensor | None = None,
    base_weight: float = 0.0,
    reverse_time: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Paper Eq. 4 and Sec. 5.2 anchor; caller must freeze critic parameters.

    Mean reduction is a declared adaptation. The critic callable consumes the
    one-step action estimate. Sampler gradients are not required by this loss.
    """
    if any(not math.isfinite(w) or w < 0 for w in (cfm_weight, base_weight)):
        raise ValueError("loss weights must be finite and nonnegative")
    endpoint = qf3_endpoint(
        action, noise, velocity, time, clip, reverse_time=reverse_time
    )
    target = noise - action if reverse_time else action - noise
    cfm = (velocity - target).square().mean()
    anchor = velocity.new_zeros(())
    if base_weight:
        if base_velocity is None or base_velocity.shape != velocity.shape:
            raise ValueError("base anchor requires matching frozen base velocity")
        anchor = (velocity - base_velocity.detach()).square().mean()
    q = critic(endpoint).mean()
    loss = -q + cfm_weight * cfm + base_weight * anchor
    return loss, {
        "actor_loss": float(loss.detach()),
        "cfm_loss": float(cfm.detach()),
        "base_anchor_loss": float(anchor.detach()),
        "masked_fraction": float(
            ((velocity - target).abs() > clip).float().mean().detach()
        ),
    }


def select_candidates(candidates: Tensor, ensemble_q: Tensor) -> tuple[Tensor, Tensor]:
    """EXPO-style deterministic argmax of conservative Q, not full EXPO-FT.

    candidates (B,N,H,A); ensemble_q (B,N,K). Caller must construct raw and
    edited candidates from the correct observation and use the same backup.
    """
    if candidates.ndim != 4 or ensemble_q.ndim != 3:
        raise ValueError("expected candidates (B,N,H,A), Q (B,N,K)")
    if candidates.shape[:2] != ensemble_q.shape[:2] or ensemble_q.shape[-1] < 1:
        raise ValueError("candidate and Q dimensions differ")
    if not torch.isfinite(candidates).all() or not torch.isfinite(ensemble_q).all():
        raise ValueError("candidates and Q must be finite")
    index = ensemble_q.min(-1).values.argmax(-1)
    return candidates[torch.arange(len(index), device=index.device), index], index


class ResFiTFeatureLearner:
    """ResFiT-style frozen-feature adaptation, with full-action critics.

    Reuses the paper recipe (ensemble10, random-pair target, ensemble-mean
    actor, bounded residual), but replaces learned visual encoders with caller
    features. This class does not establish paper-level reproduction.
    Inputs: obs (B,O), action/nominal (B,A), reward/discount (B,).
    Caller owns n-step accumulation and equal offline/online batch sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        residual_scale: float = 0.2,
        hidden: int = 128,
        seed: int = 0,
        device: str = "cpu",
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        target_noise: float = 0.025,
        noise_clip: float = 0.3,
    ):
        if min(obs_dim, action_dim, hidden) <= 0 or not 0 < residual_scale <= 1:
            raise ValueError("positive dimensions and residual_scale in (0,1] required")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.device = torch.device(device)
        self.scale, self.steps = residual_scale, 0
        if any(not math.isfinite(v) or v < 0 for v in (target_noise, noise_clip)):
            raise ValueError(
                "normalized residual noise parameters must be finite and nonnegative"
            )
        self.target_noise, self.noise_clip = target_noise, noise_clip

        def mlp(inputs: int, outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(inputs, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, outputs),
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = mlp(obs_dim + action_dim, action_dim).to(self.device)
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
            self.critics = nn.ModuleList(
                [mlp(obs_dim + action_dim, 1) for _ in range(10)]
            ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=critic_lr
        )
        self.rng = torch.Generator(device=self.device).manual_seed(seed)

    def _full(self, obs: Tensor, nominal: Tensor, actor: nn.Module) -> Tensor:
        return (
            nominal + self.scale * actor(torch.cat((obs, nominal), -1)).tanh()
        ).clamp(-1, 1)

    @torch.no_grad()
    def _smoothed_target_action(self, obs: Tensor, nominal: Tensor) -> Tensor:
        """Noise is in unit residual coordinates, before residual scaling."""
        residual = self.target_actor(torch.cat((obs, nominal), -1)).tanh()
        smoothing = (
            self.target_noise
            * torch.randn(residual.shape, generator=self.rng, device=self.device)
        ).clamp(-self.noise_clip, self.noise_clip)
        residual = (residual + smoothing).clamp(-1, 1)
        return (nominal + self.scale * residual).clamp(-1, 1)

    @torch.no_grad()
    def action(self, obs: Tensor, nominal: Tensor) -> Tensor:
        return self._full(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device),
            torch.as_tensor(nominal, dtype=torch.float32, device=self.device),
            self.actor,
        )

    def update(
        self, batch: Mapping[str, Tensor], *, actor_enabled: bool = True
    ) -> dict[str, float]:
        b = {
            k: torch.as_tensor(batch[k], dtype=torch.float32, device=self.device)
            for k in (
                "obs",
                "action",
                "nominal",
                "next_obs",
                "next_nominal",
                "reward",
                "discount",
            )
        }
        n = len(b["reward"])
        shapes = {
            "obs": (n, self.obs_dim),
            "next_obs": (n, self.obs_dim),
            "action": (n, self.action_dim),
            "nominal": (n, self.action_dim),
            "next_nominal": (n, self.action_dim),
            "reward": (n,),
            "discount": (n,),
        }
        if not n or any(
            tuple(b[k].shape) != shape or not torch.isfinite(b[k]).all()
            for k, shape in shapes.items()
        ):
            raise ValueError("invalid transition batch shape or values")
        if any((b[k].abs() > 1).any() for k in ("action", "nominal", "next_nominal")):
            raise ValueError("actions must use normalized [-1,1] codec")
        with torch.no_grad():
            a = self._smoothed_target_action(b["next_obs"], b["next_nominal"])
            subset = torch.randperm(10, generator=self.rng, device=self.device)[
                :2
            ].tolist()
            q = (
                torch.cat(
                    [
                        self.target_critics[i](torch.cat((b["next_obs"], a), -1))
                        for i in subset
                    ],
                    -1,
                )
                .min(-1)
                .values
            )
            target = bellman_target(b["reward"], b["discount"], q)
        q = torch.cat(
            [critic(torch.cat((b["obs"], b["action"]), -1)) for critic in self.critics],
            -1,
        )
        loss = (q - target[:, None]).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), 10)
        self.critic_optimizer.step()
        metrics = {"critic_loss": float(loss.detach())}
        if actor_enabled and (self.steps + 1) % 2 == 0:
            self.critics.requires_grad_(False)
            try:
                a = self._full(b["obs"], b["nominal"], self.actor)
                actor_loss = -torch.cat(
                    [c(torch.cat((b["obs"], a), -1)) for c in self.critics], -1
                ).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                self.actor_optimizer.step()
                metrics["actor_loss"] = float(actor_loss.detach())
            finally:
                self.critics.requires_grad_(True)
        with torch.no_grad():
            for model, target_model in (
                (self.actor, self.target_actor),
                (self.critics, self.target_critics),
            ):
                for source, destination in zip(
                    model.parameters(), target_model.parameters(), strict=True
                ):
                    destination.lerp_(source, 0.005)
        self.steps += 1
        return metrics
