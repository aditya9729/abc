import pytest
import torch

from abc_bench.algorithms import (
    ResFiTFeatureLearner,
    bellman_target,
    qf3_actor_loss,
    qf3_endpoint,
    select_candidates,
)


def test_bootstrap_terminal_and_timeout():
    result = bellman_target(
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 0.99]),
        torch.tensor([5.0, 5.0], requires_grad=True),
    )
    assert torch.allclose(result, torch.tensor([1.0, 5.95]))
    assert not result.requires_grad


def test_qf3_filters_q_gradient_but_preserves_cfm():
    v = torch.tensor([[0.2, 2.0]], requires_grad=True)
    z = torch.zeros_like(v)
    endpoint = qf3_endpoint(z, z, v, torch.tensor([[0.5]]), 1.0)
    endpoint.sum().backward()
    assert torch.allclose(v.grad, torch.tensor([[0.5, 0.0]]))
    v.grad = None
    loss, _ = qf3_actor_loss(
        z, z, v, torch.tensor([[0.5]]), lambda a: a.sum(-1), clip=1.0, cfm_weight=1.0
    )
    loss.backward()
    assert v.grad[0, 1] > 0  # unclipped CFM repairs the saturated coordinate


def test_reverse_time_matches_forward_time():
    action = torch.tensor([[0.3, -0.2]])
    noise = torch.tensor([[-0.5, 0.7]])
    forward = torch.tensor([[0.9, -0.5]], requires_grad=True)
    reverse = -forward.detach().clone().requires_grad_(True)
    a = qf3_endpoint(action, noise, forward, torch.tensor([[0.2]]), 0.7)
    b = qf3_endpoint(
        action, noise, reverse, torch.tensor([[0.8]]), 0.7, reverse_time=True
    )
    assert torch.allclose(a, b)


def test_base_anchor_does_not_train_base():
    velocity = torch.tensor([[0.2]], requires_grad=True)
    base = torch.tensor([[0.1]], requires_grad=True)
    loss, _ = qf3_actor_loss(
        torch.zeros_like(velocity),
        torch.zeros_like(velocity),
        velocity,
        torch.tensor([[0.5]]),
        lambda a: a.sum(-1),
        base_velocity=base,
        base_weight=1.0,
    )
    loss.backward()
    assert base.grad is None


def test_candidate_selection_uses_pessimistic_ensemble():
    candidates = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    q = torch.tensor([[[10.0, -1.0], [2.0, 3.0], [1.0, 1.0]]])
    action, index = select_candidates(candidates, q)
    assert index.item() == 1
    assert action.item() == 2


def test_resfit_full_action_gradient_and_targets():
    learner = ResFiTFeatureLearner(2, 1, hidden=8, seed=12)
    nominal = torch.full((4, 1), 0.4)
    obs = torch.zeros(4, 2)
    assert torch.equal(learner.action(obs, nominal), nominal)
    before = learner.actor[-1].weight.detach().clone()
    batch = {
        "obs": obs,
        "action": nominal,
        "nominal": nominal,
        "next_obs": obs,
        "next_nominal": nominal,
        "reward": torch.ones(4),
        "discount": torch.zeros(4),
    }
    first = learner.update(batch)
    assert "actor_loss" not in first
    assert torch.equal(before, learner.actor[-1].weight)
    metrics = learner.update(batch)
    assert "actor_loss" in metrics
    third = learner.update(batch)
    assert "actor_loss" not in third
    fourth = learner.update(batch)
    assert "actor_loss" in fourth
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert not torch.equal(before, learner.actor[-1].weight)
    assert all(p.grad is None for p in learner.target_critics.parameters())
    assert (learner.action(obs, torch.ones_like(nominal)).abs() <= 1).all()


def test_invalid_flow_time_and_discount():
    with pytest.raises(ValueError, match="time"):
        qf3_endpoint(
            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.tensor([1.1]), 1.0
        )
    with pytest.raises(ValueError, match="discount"):
        bellman_target(torch.zeros(1), torch.tensor([-1.0]), torch.zeros(1))


def test_target_smoothing_stays_within_residual_bound():
    learner = ResFiTFeatureLearner(
        2,
        1,
        hidden=8,
        seed=12,
        residual_scale=0.01,
        target_noise=100.0,
        noise_clip=100.0,
    )
    obs = torch.zeros(512, 2)
    nominal = torch.full((512, 1), 0.4)
    smooth = learner._smoothed_target_action(obs, nominal)
    assert (smooth - nominal).abs().max() <= 0.010001
    assert torch.unique(smooth).numel() > 1
    # Saturated actor plus positive smoothing still respects unit residual limits.
    with torch.no_grad():
        learner.target_actor[-1].bias.fill_(100.0)
    smooth = learner._smoothed_target_action(obs, nominal)
    assert (smooth - nominal).abs().max() <= 0.010001
