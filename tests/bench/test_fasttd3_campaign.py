"""Focused installed CPU contracts with explicit native and process doubles."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import fasttd3_campaign_host as host
from abc_bench import fasttd3_campaign_stage as stage
from abc_bench.bottle_success_reward import PBRS_PROFILE, SPARSE_PROFILE, reward_spec
from abc_bench.fasttd3_campaign_contract import PROFILE, PROFILE64, campaign_spec


def config(phase="runtime_gate", profile=PBRS_PROFILE, num_envs=128):
    return {
        "schema_version": 1,
        "profile": PROFILE if num_envs == 128 else PROFILE64,
        "phase": phase,
        "num_envs": num_envs,
        "seed_base": 200000,
        "seed_stride": 128,
        "max_episode_steps": 3540,
        "vector_steps": 64 if phase == "runtime_gate" else 42480,
        "reward": reward_spec(profile),
        "campaign": campaign_spec(phase),
        "timeouts": {
            "ready_s": 600,
            "step_s": 180,
            "shutdown_s": 30,
            "wall_s": 2400 if phase == "runtime_gate" else 12600,
            "rpc_s": 180,
        },
    }


@pytest.mark.parametrize("phase", ["runtime_gate", "training"])
@pytest.mark.parametrize("num_envs", [64, 128])
def test_separate_phases_preserve_declared_task_futility_contract(phase, num_envs):
    value = config(phase, num_envs=num_envs)
    assert host.validate_config(value) is value
    assert value["campaign"]["loss_based_stopping"] is False
    assert value["campaign"]["validation_seeds_used_for_stopping"] == []
    if phase == "training":
        assert value["campaign"]["minimum_calls"] == 21240
        assert value["campaign"]["minimum_completed_episodes_per_rank"] == 6
        assert value["campaign"]["patience_windows"] == 3


@pytest.mark.parametrize(
    "profile,num_envs",
    [
        (PROFILE, 64),
        (PROFILE64, 128),
        (PROFILE64, 64.0),
        (PROFILE64, True),
        (PROFILE64, 2),
    ],
)
def test_world_count_must_match_explicit_profile(profile, num_envs):
    value = config()
    value.update(profile=profile, num_envs=num_envs)
    with pytest.raises(ValueError, match="world count"):
        host.validate_config(value)


def test_64_worlds_preserve_per_rank_reset_sequences_and_original_updates():
    original, smaller = config(), config(num_envs=64)
    for rank in (0, 63):
        for episode in (0, 6, 780):
            assert host.native.seed_for(
                original, rank, episode
            ) == host.native.seed_for(smaller, rank, episode)
    assert stage.counters(64, num_envs=64) == {
        "vector_steps": 64,
        "transitions": 4096,
        "critic_updates": 106,
        "actor_updates": 53,
    }
    assert stage.counters(42480, num_envs=64)["transitions"] == 2718720
    assert stage.recipe(config("training", num_envs=64))["batch_size"] == 32768
    assert stage.recipe(config("training", num_envs=64))["buffer_size"] == 51200


@pytest.mark.parametrize(
    "field,value", [("vector_steps", 4000), ("vector_steps", True), ("num_envs", 2)]
)
def test_cannot_silently_shorten_long_campaign(field, value):
    changed = config("training")
    changed[field] = value
    with pytest.raises(ValueError):
        host.validate_config(changed)


def test_threshold_and_gamma_are_versioned():
    for section, field, value in [
        ("campaign", "minimum_calls", 64),
        ("reward", "gamma", 0.999),
    ]:
        changed = config("training")
        changed[section] = copy.deepcopy(changed[section])
        changed[section][field] = value
        with pytest.raises(ValueError):
            host.validate_config(changed)


def physics(time=1.0):
    return {
        "time": time,
        "time_finite": True,
        "finite": {name: True for name in ("qpos", "qvel", "qacc", "ctrl")},
        "warnings": {
            name: {"number": 0, "lastinfo": 0}
            for name in (
                "INERTIA",
                "CONTACTFULL",
                "CNSTRFULL",
                "VGEOMFULL",
                "BADQPOS",
                "BADQVEL",
                "BADQACC",
                "BADCTRL",
            )
        },
    }


@pytest.mark.parametrize(
    "failure",
    ["BADQPOS", "BADQVEL", "BADQACC", "BADCTRL", "nonfinite", "internal_reset"],
)
def test_numerical_failure_preserves_original_model_and_exact_replay_state(
    tmp_path, monkeypatch, failure
):
    world = object.__new__(host.RewardWorld)
    world.env = SimpleNamespace(model=object(), data=object())
    world.output, world.rank, world.actual_seed = tmp_path, 7, 200007
    world.episode_index, world.controls = 3, 11200
    state = np.arange(8, dtype=np.float64)
    world.integration = lambda: state.copy()
    before, after = physics(), physics(1.034)
    if failure.startswith("BAD"):
        after["warnings"][failure] = {"number": 1, "lastinfo": 30}
    elif failure == "nonfinite":
        after["finite"]["qacc"] = False
    else:
        after["time"] = 0.034
    monkeypatch.setattr(host, "physics_snapshot", lambda _data: after)
    saved = []

    def save_model(model, path):
        assert model is world.env.model
        saved.append(path)
        Path(path).write_bytes(b"explicit original native MJB double")

    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(mj_saveModel=save_model))
    with pytest.raises(FloatingPointError):
        world.check_physics(before, pre_state=state, action=[0.0] * 14)
    evidence = json.loads(next(tmp_path.glob("physics-failure-*.json")).read_text())
    assert (evidence["rank"], evidence["seed"], evidence["attempted_control"]) == (
        7,
        200007,
        11201,
    )
    assert evidence["before"] == before and evidence["after"] == after
    assert evidence["physical_action14"] == [0.0] * 14
    assert len(saved) == 1
    assert stage.pin(evidence["compiled_model"]).is_file()
    for row in evidence["integration"].values():
        np.testing.assert_array_equal(
            np.load(stage.pin(row), allow_pickle=False), state
        )


def test_benign_warning_is_reported_without_failure_or_disk_logs(tmp_path, monkeypatch):
    world = object.__new__(host.RewardWorld)
    world.env, world.output = SimpleNamespace(data=object()), tmp_path
    after = physics(1.034)
    after["warnings"]["CONTACTFULL"]["number"] = 1
    monkeypatch.setattr(host, "physics_snapshot", lambda _data: after)
    assert world.check_physics(physics()) == after
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("success", [False, True])
def test_reward_uses_native_final_potential_before_autoreset(
    tmp_path, monkeypatch, success
):
    world = object.__new__(host.RewardWorld)
    world.config, world.reward_profile = config(), PBRS_PROFILE
    world.closed, world.controls, world.episode_steps = False, 3539, 3539
    world.rank, world.episode_index = 0, 0
    world.requested_seed, world.actual_seed, world.reset_attempts = 200000, 200000, []
    world.output, world.phi, world.initial_phi = tmp_path, 0.1, 0.01
    world.reward_totals = dict.fromkeys(
        ("native_fraction", "base_reward", "shaping_reward", "learner_reward"), 0.0
    )
    world.peak_bottles = 0
    world.initial_physics = physics()
    world.final_progress = dict.fromkeys(
        ("phi", "reach", "clearance", "lift_transport"), 0.0
    )
    world.peak_progress = world.final_progress.copy()
    world.schema = {
        "action_schema": {"policy_low": [-2.0] * 14, "policy_high": [2.0] * 14}
    }
    world.check_physics = lambda *_args, **_kwargs: physics(1.034)
    world.integration = lambda: np.zeros(8)
    world.observation = lambda: [1.0] * 216
    count = 5 if success else 0
    judge = {
        "num_active_bottles": 5,
        "num_bottles_in_bin": count,
        "bottle_in_bin_mask": [success] * 5,
        "success": success,
        "reward": float(success),
    }
    world.env = SimpleNamespace(
        model=object(),
        data=SimpleNamespace(ctrl=np.zeros(14)),
        get_state=lambda: [0.0] * 14,
        step=lambda *_args, **_kwargs: (
            None,
            float(success),
            success,
            not success,
            {"task_eval": judge},
        ),
    )
    world.geometry = SimpleNamespace(
        snapshot=lambda *_args: "actual final native geometry"
    )
    monkeypatch.setattr(
        host,
        "potential",
        lambda snapshot: {
            "phi": 0.2,
            "reach": [0.4] * 5,
            "lift_clearance": [0.3] * 5,
            "transport": [0.2] * 5,
        },
    )
    order = []

    def reset():
        episode = json.loads((tmp_path / "episode-0000.json").read_text())
        order.append(episode)
        world.phi = 0.001
        return {"obs": [2.0] * 216}

    world.reset = reset
    result = world.step([0.0] * 14)
    assert len(order) == 1
    final = order[0]["reward_v2"]["final_transition"]
    assert final["phi_final_actual"] == 0.2
    assert final["phi_final_used"] == (0.0 if success else 0.2)
    assert result["reward"] == pytest.approx(0.9 if success else 0.098)
    assert result["obs"] != result["terminal_obs"]
    assert order[0]["reward_v2"]["final_phi"] == 0.2


def test_lease_is_checked_before_config_reads_output_or_children(tmp_path, monkeypatch):
    def reject():
        raise RuntimeError("explicit missing canonical lease")

    monkeypatch.setattr(stage, "require_subset_lease", reject)
    with pytest.raises(RuntimeError, match="missing canonical lease"):
        stage.run(tmp_path / "does-not-exist.json", tmp_path / "must-not-exist")
    assert list(tmp_path.iterdir()) == []


def test_original_counts_are_dynamic_and_no_new_learner_math():
    assert stage.counters(64) == {
        "vector_steps": 64,
        "transitions": 8192,
        "critic_updates": 106,
        "actor_updates": 53,
    }
    assert stage.counters(42480) == {
        "vector_steps": 42480,
        "transitions": 5437440,
        "critic_updates": 84938,
        "actor_updates": 42469,
    }
    assert stage.recipe(config("training"))["batch_size"] == 32768
    assert stage.recipe(config("training"), actual=31860)["total_timesteps"] == 31860
    assert stage.recipe(config("training"))["save_interval"] == 0


def test_sparse_profile_is_a_separate_declared_objective():
    value = config(profile=SPARSE_PROFILE)
    assert host.validate_config(value)["reward"]["potential_bound"] == 0
