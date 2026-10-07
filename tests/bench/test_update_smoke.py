"""Small CPU regressions for smoke adaptation and action provenance math."""

import numpy as np
import pytest
import torch
from torch import nn

from abc_bench.algorithms import qf3_actor_loss
from abc_bench.update_smoke import ActionCodec, OutputAdapter, discounted_return


def test_zero_adapter_and_frozen_base_then_actual_update():
    torch.manual_seed(7)
    base = nn.Linear(12, 3)
    adapter = OutputAdapter(base)
    x = torch.randn(2, 4, 12)
    expected = base(x).detach()
    assert torch.equal(adapter(x), expected)
    before = base.weight.detach().clone()
    target = torch.randn(2, 4, 3)
    noise = torch.randn_like(target)
    time = torch.full((2, 1, 1), 0.5)
    critic = nn.Linear(12, 1).requires_grad_(False)
    optimizer = torch.optim.Adam(
        [*adapter.down.parameters(), *adapter.up.parameters()], lr=1e-3
    )
    loss, _ = qf3_actor_loss(
        target,
        noise,
        adapter(x),
        time,
        lambda action: critic(action.flatten(1)),
        base_velocity=expected,
        base_weight=1.0,
        reverse_time=True,
    )
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in adapter.parameters() if p.requires_grad
    )
    optimizer.step()
    assert torch.equal(base.weight, before)
    assert base.weight.grad is None
    assert critic.weight.grad is None
    assert not torch.equal(adapter(x), expected)
    adapter.enabled = False
    assert torch.equal(adapter(x), expected)


def test_codec_round_trip_preserves_full_physical_commands():
    codec = ActionCodec(
        np.array([0.3, -0.2], dtype=np.float32), np.array([0.4, 0.7], dtype=np.float32)
    )
    physical = np.array([[0.2, 0.9], [1.0, -0.8]], dtype=np.float32)
    np.testing.assert_allclose(
        codec.decode(codec.encode(physical)), physical, atol=2e-6
    )
    with pytest.raises(ValueError, match="strictly inside"):
        codec.decode(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="saturated"):
        codec.encode(np.array([[1e5, 0.0]]))


def test_return_counts_executed_ticks_only():
    assert discounted_return([2.0, 3.0], 0.5) == 3.5
