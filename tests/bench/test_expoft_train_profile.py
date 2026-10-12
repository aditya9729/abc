"""TRAIN reset/raw action contracts. Doubles are explicit; no policy/GPU."""

import copy
import json
from types import SimpleNamespace

import pytest

from abc_bench import expoft_host as host
from abc_bench import expoft_train_profile as profile


class PlacementFailure(Exception):
    pass


class ResetDouble:
    def __init__(self, failures=0):
        self.failures, self.calls = failures, []

    def reset(self, *, seed, options, randomize):
        self.calls.append(("reset", seed, copy.deepcopy(options), randomize))
        if self.failures:
            self.failures -= 1
            raise PlacementFailure("Explicit placement-error fixture")
        return {}, {
            "randomization": SimpleNamespace(
                to_dict=lambda: {"seed": seed, "metadata": {"bottle_count": 5}}
            )
        }

    def forget_arm_state(self):
        self.calls.append(("forget_arm_state",))


@pytest.mark.parametrize(
    "failures,expected",
    [
        (0, [("reset", 9200)]),
        (1, [("reset", 9200), ("forget_arm_state",), ("reset", 9200)]),
        (
            2,
            [
                ("reset", 9200),
                ("forget_arm_state",),
                ("reset", 9200),
                ("reset", 109200),
            ],
        ),
    ],
)
def test_exact_conditional_author_reset_sequence(failures, expected):
    env = ResetDouble(failures)
    _, _, actual, attempts = profile.reset_world(
        env, 9200, sampling_error=PlacementFailure
    )
    assert [call[:2] for call in env.calls] == expected
    assert actual == (109200 if failures == 2 else 9200)
    assert [item["operation"] for item in attempts] == [call[0] for call in expected]
    for call in env.calls:
        if call[0] == "reset":
            assert call[2:] == (profile.OPTIONS, True)


def test_other_native_failure_never_clears_or_retries():
    env = ResetDouble()
    env.reset = lambda **_: (_ for _ in ()).throw(RuntimeError("native failure"))
    with pytest.raises(RuntimeError, match="native failure"):
        profile.reset_world(env, 9200, sampling_error=PlacementFailure)
    assert env.calls == []


@pytest.mark.parametrize(
    "seed", [9100, 9101, 9102, 109100, 109101, 109102, 20260511, 20160511, True]
)
def test_heldout_and_mapped_families_rejected(seed):
    with pytest.raises(ValueError):
        profile.check_seed(seed)


def test_schema3_is_separate_and_rejects_wrong_action_profile(tmp_path):
    value = {
        "schema_version": 3,
        "stage": "train",
        "host": {
            "num_envs": 1,
            "initial_seed": 9200,
            "allowed_seeds": [9200],
            "max_episode_steps": 3540,
            "save_video": False,
            "reset_profile": profile.PROFILE,
            "action_profile": profile.ACTION_PROFILE,
        },
        "timeouts": {"host_wall_s": 1800, "rpc_s": 120, "max_requests": 100000},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    assert host.load_config(path) == value
    value["host"]["action_profile"] = "clipped"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        host.load_config(path)


def test_actual_original_cpu_reset_preserves_carried_arm_and_native_profile(tmp_path):
    """Actual native CPU reset; no camera, VLA or GPU is initialized."""
    import numpy as np
    from abc_sim import make_env

    env = make_env(
        "put_plastic_bottles_in_bin",
        render_cameras=False,
        enable_task_randomizer=True,
        max_episode_steps=3540,
        terminate_on_success=True,
    )
    try:
        _, _, actual, attempts = profile.reset_world(env, 9200)
        assert actual in (9200, 109200)
        assert attempts[0]["operation"] == "reset"
        proposal = env.get_state().copy()
        proposal[0] += np.float32(0.001)
        env.step(proposal, render_obs=False)
        carried = env.get_state().copy()
        _, state, actual, attempts = profile.reset_world(env, 9201)
        assert state["metadata"]["bottle_count"] == 5
        if len(attempts) == 1:
            expected = carried.copy()
            for index in (6, 13):
                # Original reset converts measured float32 grip to metres;
                # original get_state projects it back to float32 policy units.
                physical = np.float32(carried[index]) * 0.0475
                expected[index] = np.float32(
                    np.clip(float(physical) / 0.0475, 0.0, 1.0)
                )
            np.testing.assert_array_equal(env.get_state(), expected)
        pins = profile.record_initial(env, tmp_path / "initial", randomization=state)
        assert pins["model"] == {"nq": 58, "nv": 52, "nu": 14}
        assert pins["native_time_s"] == 0.0
        assert pins["artifacts"]["integration"]["shape"] == [531]
    finally:
        env.close()
