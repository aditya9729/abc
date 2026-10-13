"""CPU fixtures for diagnostic semantics. These tests do not execute native physics."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import fasttd3_reach_host as host
from abc_bench import fasttd3_reach_stage as stage
from abc_bench import subset
from abc_bench.fasttd3_campaign_host import validate_config as validate_campaign
from abc_bench.fasttd3_reach_contract import (
    HORIZON,
    ReachEpisode,
    nearest_distance,
    validate_config,
)

BUILD = Path("/home/user/aditya/RL/builds/abc-fasttd3-reach-diagnostic-20261013")


def configuration():
    return json.loads((BUILD / "runtime_gate-host.json").read_text())


def test_geometry_nearest_any_site_and_bottle():
    centers = np.array(
        [[10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0], [40.0, 0, 0], [0.0, 0.11, 0]]
    )
    assert nearest_distance(centers, [[1.0, 0, 0], [0.0, 0, 0]]) == 0.11
    assert nearest_distance(centers + [0, 0, 1], [[1.0, 0, 1], [0.0, 0, 1]]) == 0.11
    with pytest.raises(ValueError):
        nearest_distance(centers[:4], np.zeros((2, 3)))
    centers[4, 0] = np.nan
    with pytest.raises(FloatingPointError):
        nearest_distance(centers, np.zeros((2, 3)))


def test_reward_hold_and_reset_geometry_eligibility():
    episode = ReachEpisode(0.4)
    assert episode.step(0.3) == pytest.approx(np.exp(-2))
    for _ in range(4):
        episode.step(0.12)
    assert not episode.evidence()["reach_achieved"]
    episode.step(0.1200001)
    for _ in range(4):
        episode.step(0.12)
    assert not episode.evidence()["reach_achieved"]
    assert episode.step(0.12) == pytest.approx(np.exp(-0.8))
    assert episode.evidence()["qualified_reach_achieved"] is True
    near = ReachEpisode(0.05)
    for _ in range(5):
        near.step(0.01)
    assert near.evidence()["reach_achieved"] is True
    assert near.evidence()["qualified_reach_achieved"] is False
    assert near.evidence()["initially_inside_criterion"] is True
    boundary = ReachEpisode(0.15)
    assert boundary.evidence()["eligible_initial_distance"] is False


def test_fixed_horizon_and_invalid_distance():
    episode = ReachEpisode(0.5)
    for _ in range(HORIZON):
        assert 0 <= episode.step(0.0) <= 1
    assert episode.evidence()["reward_sum"] == HORIZON
    with pytest.raises(RuntimeError):
        episode.step(0.1)
    for bad in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            ReachEpisode(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_envs", 128),
        ("max_episode_steps", 3540),
        ("seed_stride", 64),
        ("vector_steps", 64),
        ("phase", "historical"),
        ("schema_version", True),
    ],
)
def test_new_profile_rejects_old_recipe_mutations(field, value):
    config = configuration()
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_campaign_predicate_still_rejects_reach():
    with pytest.raises(ValueError):
        validate_campaign(configuration())


def physics():
    return {
        "time": 1.0,
        "time_finite": True,
        "warnings": {
            name: {"number": 0, "lastinfo": 0}
            for name in (
                "INERTIA",
                "CONTACTFULL",
                "CNSTRFULL",
                "BADQPOS",
                "BADQVEL",
                "BADQACC",
                "BADCTRL",
            )
        },
        "finite": {name: True for name in ("qpos", "qvel", "qacc", "ctrl")},
    }


@pytest.mark.parametrize(
    "warning",
    ("INERTIA", "CONTACTFULL", "CNSTRFULL", "BADQPOS", "BADQVEL", "BADQACC", "BADCTRL"),
)
def test_every_warning_fails_with_partial_evidence(monkeypatch, tmp_path, warning):
    world = host.ReachWorld.__new__(host.ReachWorld)
    world.env, world.output, world.rank, world.controls = (
        SimpleNamespace(data=None),
        tmp_path,
        0,
        7,
    )
    snapshot = physics()
    snapshot["warnings"][warning]["number"] = 1
    monkeypatch.setattr(host, "physics_snapshot", lambda data: snapshot)
    with pytest.raises(FloatingPointError):
        world.check_physics()
    recorded = json.loads((tmp_path / "physics-failure-00000007.json").read_text())
    assert recorded["after"]["warnings"][warning]["number"] == 1


def test_time_reset_fails(monkeypatch, tmp_path):
    world = host.ReachWorld.__new__(host.ReachWorld)
    world.env, world.output, world.rank, world.controls = (
        SimpleNamespace(data=None),
        tmp_path,
        0,
        4,
    )
    monkeypatch.setattr(host, "physics_snapshot", lambda data: physics())
    with pytest.raises(FloatingPointError):
        world.check_physics({"time": 1.0})


class NativeBoundaryFixture:
    """Explicit step double. This class supplies no native physics evidence."""

    def __init__(self, steps):
        self.controls = steps
        self.calls = []
        self.model = object()
        self.data = SimpleNamespace(ctrl=np.zeros(14), time=1.0)
        self.judge = {
            "num_active_bottles": 5,
            "num_bottles_in_bin": 5,
            "bottle_in_bin_mask": [True] * 5,
            "success": True,
        }

    def step(self, actions, *, render_obs):
        self.calls.append((actions.copy(), render_obs))
        self.controls += 1
        return (
            None,
            1.0,
            False,
            self.controls == HORIZON,
            {"task_eval": self.judge.copy()},
        )

    def get_state(self):
        return np.zeros(14)


def make_fixture_world(tmp_path, steps):
    world = host.ReachWorld.__new__(host.ReachWorld)
    world.env = NativeBoundaryFixture(steps)
    world.closed, world.controls, world.episode_steps = False, steps, steps
    world.config, world.rank, world.output = configuration(), 0, tmp_path
    world.episode_index, world.requested_seed, world.actual_seed = 0, 200000, 200000
    world.reset_attempts, world.completed_episodes, world.peak_native_bottles = [], 0, 0
    world.schema = {
        "action_schema": {"policy_low": [-2.0] * 14, "policy_high": [2.0] * 14}
    }
    world.initial_physics = physics()
    world.reaching = ReachEpisode(0.3)
    for _ in range(steps):
        world.reaching.step(0.3)
    world.geometry = SimpleNamespace(
        snapshot=lambda *args: SimpleNamespace(
            bottle_centers=np.tile([0.1, 0, 0], (5, 1)),
            grasp_positions=np.zeros((2, 3)),
        )
    )
    world.integration = lambda: np.arange(5, dtype=np.float64)
    world.check_physics = lambda *args: dict(physics(), time=1.034)
    world.observation = lambda: [7.0] * 216
    world.trace = io.StringIO()
    world.reset = lambda: {
        "obs": [99.0] * 216,
        "rank": 0,
        "requested_seed": 200128,
        "actual_seed": 200128,
    }
    return world


def test_native_success_does_not_replace_diagnostic_timeout(tmp_path):
    world = make_fixture_world(tmp_path, 0)
    row = world.step(np.zeros(14, dtype=np.float32))
    assert row["terminated"] is False and row["truncated"] is False
    assert row["episodes"] == []
    assert world.env.calls[0][1] is False
    trace = json.loads(world.trace.getvalue())
    assert trace["native_five_bottle_eval"]["success"] is True
    assert trace["reward"] == pytest.approx(np.exp(-0.1 / 0.15))


def test_terminal_observation_stays_before_autoreset(tmp_path):
    world = make_fixture_world(tmp_path, HORIZON - 1)
    row = world.step(np.full(14, 3.0, dtype=np.float32))
    assert row["obs"] == [99.0] * 216
    assert row["terminal_obs"] == [7.0] * 216
    assert row["terminated"] is False and row["truncated"] is True
    assert row["physical_action"] == [2.0] * 14
    assert row["normalized_clipped"] == 14
    episode = row["episodes"][0]
    assert "success" not in episode
    assert episode["native_full_success"] is True
    assert episode["reach_diagnostic"]["reach_achieved"] is False
    assert episode["controls"] == 128


def test_bad_actions_never_advance_fixture(tmp_path):
    world = make_fixture_world(tmp_path, 0)
    for value in ([0.0] * 13, [float("nan")] * 14, [1e300] * 14):
        with pytest.raises(ValueError):
            world.step(value)
    assert world.env.calls == [] and world.controls == 0


def test_all_rank_actions_validate_before_transport():
    pool = host.ReachVectorPool.__new__(host.ReachVectorPool)
    pool.initial_pending, pool.vector_steps, pool.config = False, 0, configuration()
    advanced = []
    pool._call = lambda requests: advanced.append(requests)
    actions = [[0.0] * 14 for _ in range(64)]
    actions[-1][-1] = float("inf")
    with pytest.raises(ValueError):
        pool.step(actions)
    assert not advanced and pool.vector_steps == 0


def test_lease_guard_runs_before_missing_input_or_mutation(monkeypatch, tmp_path):
    def rejected():
        raise RuntimeError("explicit absent lease fixture")

    monkeypatch.setattr(stage, "require_subset_lease", rejected)
    output = tmp_path / "absent"
    with pytest.raises(RuntimeError, match="absent lease"):
        stage.run(tmp_path / "missing.json", output)
    assert not output.exists()


def test_canonical_reach_allowlist_does_not_expand_generic_bound(tmp_path):
    original = json.loads((BUILD / "runtime_gate-job.json").read_text())
    path = tmp_path / "job.json"
    path.write_text(json.dumps(original))
    assert subset.load_job(path)["module"] == "abc_bench.fasttd3_reach_stage"
    original["max_wall_s"] = 7201
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="7200"):
        subset.load_job(path)


def test_job_inputs_are_bounded_but_unexecuted():
    config, evidence = stage.load_config(
        Path(
            "/home/user/aditya/RL/builds/abc-fasttd3-reach-matched-20261013/runtime_gate-config.json"
        )
    )
    assert len(evidence["worker_resources"]) == 64
    assert config["runtime_gate_parent"] is None
    assert stage.counters(256) == {
        "vector_steps": 256,
        "transitions": 16384,
        "critic_updates": 490,
        "actor_updates": 245,
    }
    assert stage.counters(2048)["critic_updates"] == 4074


def test_long_preparation_rejects_absent_new_gate(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        stage.prepare_training_config(
            BUILD / "runtime_gate-config.json",
            BUILD / "training-host.json",
            tmp_path / "absent-parent.json",
            tmp_path / "absent-worker.json",
        )
