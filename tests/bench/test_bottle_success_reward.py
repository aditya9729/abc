"""CPU mathematical and explicit-double contracts; no native runtime claim."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench.bottle_success_reward import (
    BOTTLE_NAMES,
    GAMMA,
    PBRS_PROFILE,
    POTENTIAL_BOUND,
    SPARSE_PROFILE,
    BottlePotentialGeometry,
    GeometrySnapshot,
    potential,
    reward_config_sha256,
    reward_spec,
    transition_reward,
)


def judge(count=0):
    return {
        "num_active_bottles": 5,
        "num_bottles_in_bin": count,
        "bottle_in_bin_mask": [index < count for index in range(5)],
        "success": count == 5,
        "reward": float(np.float32(count / 5)),
        "bottle_names": list(BOTTLE_NAMES),
        "bottle_center_offsets": [[0.0, 0.0, 0.03] for _ in range(5)],
        "bottle_reaches": [0.1] * 5,
        "bin_bottom_y": -0.02,
        "bin_top_y": 0.2,
        "ever_success": True,
        "max_bottles_in_bin_so_far": 5,
    }


def snapshot(*, centers=None, grasps=None, mask=None):
    return GeometrySnapshot(
        bottle_centers=np.tile([0.5, 0.0, 0.78], (5, 1))
        if centers is None
        else centers,
        grasp_positions=[[0, 0, 0.78], [0, 1, 0.78]] if grasps is None else grasps,
        bin_opening=[0.8, 0, 1.0],
        table_point=[0.0, 0.0, 0.75],
        table_normal=[0, 0, 1],
        bottle_reaches=[0.1] * 5,
        in_bin=[False] * 5 if mask is None else mask,
    )


def reward(previous, final, *, count=0, timeout=False):
    return transition_reward(
        PBRS_PROFILE,
        judge(count),
        terminated=count == 5,
        truncated=timeout,
        phi_pre=previous,
        phi_final=final,
    )


def test_stationary_partial_placement_has_no_positive_income():
    phi = potential(snapshot(mask=[True, False, False, False, False]))["phi"]
    results = [reward(phi, phi, count=1)["learner_reward"] for _ in range(100)]
    assert phi > 0
    assert all(value < 0 for value in results)
    assert sum(GAMMA**t * value for t, value in enumerate(results)) == pytest.approx(
        (GAMMA**100 - 1) * phi
    )


def test_out_and_back_cycle_telescopes_to_nonpositive_return():
    values = [0.02, 0.21, 0.12, 0.02]
    rewards = [reward(a, b)["learner_reward"] for a, b in pairwise(values)]
    assert rewards[0] > 0  # Progress gives a dense learning signal.
    assert sum(GAMMA**t * value for t, value in enumerate(rewards)) == pytest.approx(
        (GAMMA**3 - 1) * values[0]
    )


def test_success_potential_zero_preserves_base_objective_and_positive_payment():
    result = reward(0.25, 0.25, count=5)
    assert result["base_reward"] == 1
    assert result["phi_final_actual"] == 0.25
    assert result["phi_final_used"] == 0
    assert result["learner_reward"] == 0.75
    # A complete discounted return differs from the sparse return only by Phi(s0).
    values = [0.01, 0.03, 0.17, 0.25]
    results = [reward(a, b) for a, b in pairwise(values[:-1])]
    results.append(reward(values[-2], values[-1], count=5))
    actual = sum(GAMMA**t * row["learner_reward"] for t, row in enumerate(results))
    assert actual == pytest.approx(GAMMA**2 - values[0])


def test_timeout_retains_final_potential_and_bootstrap_cancels_boundary():
    result = reward(0.1, 0.2, timeout=True)
    assert result["phi_final_used"] == 0.2
    sparse_value_final = 0.7
    shaped_value_final = sparse_value_final - 0.2
    assert result["learner_reward"] + GAMMA * shaped_value_final == pytest.approx(
        GAMMA * sparse_value_final - 0.1
    )


def test_reset_potential_is_not_part_of_the_previous_transition():
    final = reward(0.10, 0.20, timeout=True)
    next_episode = reward(0.02, 0.03)
    assert final["learner_reward"] == pytest.approx(0.098)
    assert next_episode["learner_reward"] == pytest.approx(0.0097)
    assert final["learner_reward"] != pytest.approx(GAMMA * 0.02 - 0.10)


@pytest.mark.parametrize("count", range(6))
def test_sparse_reference_ignores_historical_success_and_partial_fraction(count):
    result = transition_reward(
        SPARSE_PROFILE, judge(count), terminated=count == 5, truncated=False
    )
    assert result["learner_reward"] == float(count == 5)
    assert result["shaping_reward"] == 0
    assert result["native_fraction"] == float(np.float32(count / 5))


def test_dense_reach_clearance_transport_and_bounds():
    base = snapshot()
    near = snapshot(grasps=[[0.5, 0, 0.78], [0, 1, 0.78]])
    lifted = snapshot(centers=np.tile([0.5, 0, 1.0], (5, 1)))
    transported = snapshot(centers=np.tile([0.8, 0, 1.0], (5, 1)))
    assert potential(near)["phi"] > potential(base)["phi"]
    assert potential(base)["lift_clearance"] == [0.0] * 5
    assert potential(lifted)["lift_clearance"] == [1.0] * 5
    assert potential(transported)["transport"][0] > potential(lifted)["transport"][0]
    assert potential(snapshot(mask=[True] * 5))["phi"] == POTENTIAL_BOUND
    for height in (0.0, 0.7, 0.9, 1.2, 100.0):
        value = potential(snapshot(centers=np.tile([0.5, 0, height], (5, 1))))
        assert 0 <= value["phi"] <= POTENTIAL_BOUND
        assert max(value["bottle_progress"]) <= 0.75


def test_geometry_snapshot_detaches_caller_arrays():
    centers = np.tile([0.5, 0, 0.78], (5, 1))
    value = snapshot(centers=centers)
    centers[:] = 100
    assert value.bottle_centers[0, 0] == 0.5
    assert not value.bottle_centers.flags.writeable


def test_world_rigid_transform_preserves_the_geometric_potential():
    original = snapshot(centers=np.tile([0.5, 0, 1.0], (5, 1)))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([1.0, -2.0, 0.3])
    transformed = GeometrySnapshot(
        bottle_centers=original.bottle_centers @ rotation.T + translation,
        grasp_positions=original.grasp_positions @ rotation.T + translation,
        bin_opening=rotation @ original.bin_opening + translation,
        table_point=rotation @ original.table_point + translation,
        table_normal=rotation @ original.table_normal,
        bottle_reaches=original.bottle_reaches,
        in_bin=original.in_bin,
    )
    assert potential(transformed)["phi"] == pytest.approx(potential(original)["phi"])


def test_collision_extent_not_wrapper_height_sets_conservative_clearance():
    original = snapshot(centers=np.tile([0.5, 0, 0.9], (5, 1)))
    large = GeometrySnapshot(
        bottle_centers=original.bottle_centers,
        grasp_positions=original.grasp_positions,
        bin_opening=original.bin_opening,
        table_point=original.table_point,
        table_normal=original.table_normal,
        bottle_reaches=[0.2] * 5,
        in_bin=original.in_bin,
    )
    assert potential(original)["lift_clearance"][0] > 0
    assert potential(large)["lift_clearance"] == [0.0] * 5


@pytest.mark.parametrize(
    "change",
    [
        {"num_active_bottles": 4},
        {"num_active_bottles": True},
        {"num_bottles_in_bin": 2},
        {"success": True},
        {"bottle_in_bin_mask": [0] * 5},
        {"reward": 0.2},
        {"reward": float("nan")},
    ],
)
def test_inconsistent_native_judges_fail_closed(change):
    value = judge()
    value.update(change)
    with pytest.raises(ValueError):
        transition_reward(SPARSE_PROFILE, value, terminated=False, truncated=False)


@pytest.mark.parametrize(
    "previous,final", [(-0.1, 0), (0, 0.26), (True, 0), (0, np.nan)]
)
def test_invalid_potential_rejected(previous, final):
    with pytest.raises((TypeError, ValueError)):
        reward(previous, final)


def test_profile_and_terminal_semantics_fail_closed():
    with pytest.raises(ValueError):
        reward_spec("unknown")
    with pytest.raises(ValueError):
        transition_reward(SPARSE_PROFILE, judge(5), terminated=False, truncated=True)
    with pytest.raises(ValueError):
        transition_reward(SPARSE_PROFILE, judge(), terminated=True, truncated=False)
    with pytest.raises(ValueError):
        transition_reward(SPARSE_PROFILE, judge(5), terminated=True, truncated=True)
    with pytest.raises(ValueError):
        transition_reward(
            SPARSE_PROFILE, judge(), terminated=False, truncated=False, phi_pre=0
        )


def fk_double(monkeypatch):
    calls = []
    names = {
        name: index for index, name in enumerate(BOTTLE_NAMES + ("bin_container",))
    }
    model = SimpleNamespace(
        nq=2,
        nmocap=0,
        opt=SimpleNamespace(gravity=np.array([0.0, 0.0, -9.81])),
        geom_type=[0],
        body=lambda name: SimpleNamespace(id=names[name]),
        site=lambda name: SimpleNamespace(
            id=["left_grasp_site", "right_grasp_site"].index(name)
        ),
        geom=lambda name: SimpleNamespace(id=0 if name == "table_plane" else -1),
    )
    scratch = SimpleNamespace(
        qpos=np.zeros(2),
        xpos=np.tile([0.5, 0, 0.8], (6, 1)),
        xmat=np.tile(np.eye(3).reshape(9), (6, 1)),
        xipos=np.full((6, 3), -100.0),  # Massless wrapper COM must never be used.
        site_xpos=np.zeros((2, 3)),
        geom_xpos=np.array([[0.0, 0.0, 0.75]]),
        geom_xmat=np.eye(3).reshape(1, 9),
    )
    native = SimpleNamespace(
        qpos=np.array([0.2, 0.3]), site_xpos=np.full((2, 3), -999.0)
    )

    def kinematics(actual_model, data):
        assert actual_model is model and data is scratch
        calls.append(data.qpos.copy())
        data.site_xpos[:] = [[data.qpos[0], 0, 0.8], [data.qpos[1], 0, 0.8]]
        # Actual bin local y is its height axis, rotated to world z.
        data.xmat[5] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).reshape(9)

    def data_factory(actual_model):
        assert actual_model is model
        return scratch

    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(
            MjData=data_factory,
            mj_kinematics=kinematics,
            mjtGeom=SimpleNamespace(mjGEOM_PLANE=0),
        ),
    )
    return model, native, scratch, calls


def test_fk_reader_uses_final_qpos_separate_scratch_real_frames_and_com_offsets(
    monkeypatch,
):
    model, native, scratch, calls = fk_double(monkeypatch)
    original_qpos, original_stale_sites = native.qpos.copy(), native.site_xpos.copy()
    reader = BottlePotentialGeometry(model, judge())
    value = reader.snapshot(model, native, judge())
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], original_qpos)
    np.testing.assert_array_equal(native.qpos, original_qpos)
    np.testing.assert_array_equal(native.site_xpos, original_stale_sites)
    np.testing.assert_allclose(value.grasp_positions[:, 0], [0.2, 0.3])
    np.testing.assert_allclose(value.bottle_centers[0], [0.5, 0, 0.83])
    np.testing.assert_allclose(value.bin_opening, [0.5, 0, 1.05])
    scratch.xpos[:] = 100
    assert value.bottle_centers[0, 0] == 0.5
    reader.close()
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.snapshot(model, native, judge())


def test_fk_rebinding_and_metadata_changes_fail_closed(monkeypatch):
    model, native, _, _ = fk_double(monkeypatch)
    reader = BottlePotentialGeometry(model, judge())
    with pytest.raises(ValueError, match="model changed"):
        reader.snapshot(deepcopy(model), native, judge())
    changed = judge()
    changed["bottle_reaches"][0] = 0.11
    with pytest.raises(ValueError, match="metadata changed"):
        reader.snapshot(model, native, changed)


def test_fk_bottle_center_uses_native_body_rotation_of_scale_aware_offset(monkeypatch):
    model, native, scratch, _ = fk_double(monkeypatch)
    scratch.xmat[0] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).reshape(9)
    reader = BottlePotentialGeometry(model, judge())
    actual = reader.snapshot(model, native, judge())
    np.testing.assert_allclose(actual.bottle_centers[0], [0.53, 0, 0.8])


def test_geometry_contract_binds_actual_caller_model_geometry_and_exact_config(
    monkeypatch,
):
    model, _, _, _ = fk_double(monkeypatch)
    reader = BottlePotentialGeometry(model, judge())
    first = reader.contract(PBRS_PROFILE, native_model_sha256="a" * 64)
    second = reader.contract(PBRS_PROFILE, native_model_sha256="b" * 64)
    assert first["config_sha256"] == reward_config_sha256(PBRS_PROFILE)
    assert first["geometry_sha256"] == second["geometry_sha256"]
    assert first["contract_sha256"] != second["contract_sha256"]
    declaration = dict(first)
    seal = declaration.pop("contract_sha256")
    encoded = json.dumps(
        declaration, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert seal == hashlib.sha256(encoded.encode("ascii")).hexdigest()
    changed = judge()
    changed["bottle_center_offsets"][0][0] = 0.02
    other = BottlePotentialGeometry(model, changed).contract(
        PBRS_PROFILE, native_model_sha256="a" * 64
    )
    assert other["geometry_sha256"] != first["geometry_sha256"]
    with pytest.raises(ValueError):
        reader.contract(PBRS_PROFILE, native_model_sha256="not-a-pin")
    with pytest.raises(ValueError):
        reader.contract(SPARSE_PROFILE, native_model_sha256="a" * 64)


def test_fk_rejects_mocap_or_nonvertical_table(monkeypatch):
    model, native, scratch, _ = fk_double(monkeypatch)
    model.nmocap = 1
    with pytest.raises(ValueError, match="mocap"):
        BottlePotentialGeometry(model, judge())
    model.nmocap = 0
    reader = BottlePotentialGeometry(model, judge())
    scratch.geom_xmat[0] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).reshape(9)
    with pytest.raises(ValueError, match="gravity vertical"):
        reader.snapshot(model, native, judge())
