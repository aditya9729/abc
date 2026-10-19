"""CPU admission and diagnostic-routing tests; no learning claims from doubles."""

import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import historical_fixed_data_100 as hundred
from abc_bench import historical_fixed_data_learning as historical
from abc_bench import realtime_update_probe as probe

PREPARATION = Path(
    "/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-100-preparation-20261019"
)


def prepared_config():
    path = PREPARATION / "PHASE_SPEC.json"
    if not path.exists():
        pytest.skip("Local sealed100 specification is unavailable")
    return {
        "phase": hundred.PHASE,
        "fixed_data": {
            "spec": probe.file_pin(path),
            "python": probe.file_pin(Path(sys.executable).resolve()),
            "condition": historical.ORDERED_CONDITION,
        },
        "installed_modules": {
            name: {}
            for name in (
                hundred.MODULE,
                historical.MODULE,
                "abc_bench.fixed_data_learning",
                "expo_ft.agents.alg.agent",
            )
        },
    }


def test_sealed_spec_preserves_original_recipe_and_published_bank():
    config = prepared_config()
    spec = hundred.load_spec(config)
    parent = json.loads(probe.authenticate(spec["parent_spec"]).read_text())
    for field in (
        "checkpoint_parent",
        "settings",
        "replay_acceptance",
        "replay_inspection",
        "episode",
        "runtime_versions",
        "reward_view",
        "assembly_order",
    ):
        assert spec[field] == parent[field]
    assert spec["measure_at"] == [0, 10, 30, 100]
    assert spec["stopping"]["generalization_based_early_stopping"] is False
    accepted = hundred.authenticate_bank(spec)
    assert accepted["arrays"]["full/cand"]["shape"] == [64, 8, 14]
    assert hundred.prepare_before_restore is historical.prepare_before_restore
    assert hundred.prepare_replay is historical.prepare_replay
    assert probe.fixed_phase_module(config) is hundred


@pytest.mark.parametrize(
    "field,value",
    [
        ("update_calls", 101),
        ("measure_at", [0, 100]),
        ("final_checkpoint_step", 10),
        ("reward_view", "shaped"),
    ],
)
def test_spec_refuses_unbounded_or_changed_recipe(tmp_path, monkeypatch, field, value):
    config = prepared_config()
    spec = json.loads(Path(config["fixed_data"]["spec"]["path"]).read_text())
    spec[field] = value
    changed = tmp_path / "spec.json"
    changed.write_text(json.dumps(spec))
    config["fixed_data"]["spec"] = probe.file_pin(changed)
    monkeypatch.setattr(hundred, "SPEC_SHA256", config["fixed_data"]["spec"]["sha256"])
    with pytest.raises(ValueError, match="bounded100"):
        hundred.load_spec(config)


def test_old_phase_still_refuses_hundred_spec():
    config = prepared_config()
    config["phase"] = historical.PHASE
    with pytest.raises(ValueError, match="specification differs"):
        historical.load_spec(config)


def test_added_terminal_windows_use_isolated_original_views(monkeypatch):
    indices, prepared_args = [], []
    buffer = SimpleNamespace(rng=891)

    class View:
        def __init__(self, index):
            self.index, self.rng = index, None

        def sample_jax(self, count):
            assert count == 1 and self.rng == (2804, self.index)
            return {"fixture_index": self.index}

        def _convert_to_openpi_format(self, value):
            return value

    def fixed_window_view(actual, index):
        assert actual is buffer
        indices.append(index)
        return View(index)

    def prepare(raw, *args):
        prepared_args.append(args)
        return raw

    contracts = SimpleNamespace(
        fixed_window_view=fixed_window_view,
        check_terminal_target=lambda raw, offset, np: {
            "reward": 0.99**offset,
            "mask": 0,
        },
        batch_fingerprint=lambda prepared, np: str(prepared["fixture_index"]),
    )
    monkeypatch.setitem(sys.modules, "sadhana_terminal_critic.contracts", contracts)
    monkeypatch.setitem(
        sys.modules,
        "expo_ft.agents.alg.batch_utils",
        SimpleNamespace(prepare_critic_batch=prepare),
    )
    jax = SimpleNamespace(
        random=SimpleNamespace(PRNGKey=lambda x: x, fold_in=lambda key, i: (key, i))
    )
    rows = hundred.added_terminal_batches(
        SimpleNamespace(critic_camera_keys=("base_0_rgb",)), buffer, jax, np
    )
    assert buffer.rng == 891
    assert indices == list(range(7410, 7418))
    assert prepared_args == [(32, 14, 32, 16, 8, ("base_0_rgb",))] * 8
    assert [row["actual_controls_until_terminal"] for row in rows] == list(
        range(8, 0, -1)
    )
    assert [row["includes_author_backfilled_tail"] for row in rows] == [False] + [
        True
    ] * 7
    np.testing.assert_allclose(
        [row["target"]["reward"] for row in rows], 0.99 ** np.arange(7, -1, -1)
    )


@pytest.mark.parametrize("repeat_changes", [False, True])
def test_extra_hook_scores_current_cache_twice_without_retaining_old_agent(
    tmp_path, monkeypatch, repeat_changes
):
    calls, agents = [], []
    fixture = np.arange(4, dtype=np.float32)
    path = tmp_path / "input.npy"
    np.save(path, fixture)
    accepted = {
        "arrays": {
            "fixture": {**probe.file_pin(path), "shape": [4], "dtype": "float32"}
        }
    }
    monkeypatch.setattr(hundred, "authenticate_bank", lambda spec: accepted)
    monkeypatch.setattr(hundred, "added_terminal_batches", lambda *args: ["terminal"])
    monkeypatch.setattr(
        hundred,
        "terminal_calibration",
        lambda agent, rows, *args: {"current_version": agent.version},
    )
    monkeypatch.setattr(
        hundred, "model_digests", lambda agent, *args: {"model": agent.version}
    )
    jax = SimpleNamespace(device_get=lambda x: x, numpy=np)
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    api = object()
    monkeypatch.setitem(sys.modules, "expo_ft", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "expo_ft.agents", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "expo_ft.agents.alg", SimpleNamespace(realtime_expo_ft=api)
    )

    class Agent:
        def __init__(self, version, cache="stale"):
            self.version, self.cache = version, cache

        def replace(self, *, _infer_cache):
            assert _infer_cache is None
            return Agent(self.version, None)

        def cache_infer_params(self):
            assert self.cache is None
            return Agent(self.version, self.version)

    class Recorder:
        def __init__(self, *args):
            self.arrays, self.comparisons = {}, []

        def tree(self, name, value):
            self.arrays[name] = list(value)

        def array(self, name, value):
            self.arrays[name] = value.tolist()
            return value

    def score(agent, arrays, transformed, actual_api, *args):
        assert agent.cache == agent.version and actual_api is api
        np.testing.assert_array_equal(arrays["fixture"], fixture)
        calls.append(agent.version)
        qs = np.arange(64, dtype=np.float32) + agent.version
        if repeat_changes and len(calls) % 2 == 0:
            qs[0] += 1
        return {"qs": qs, "idx": np.asarray(63)}, np.asarray([6, 4])

    def parity(left, right, *args):
        if not np.array_equal(left["qs"], right["qs"]):
            raise ValueError("repeat changed")

    scorer = SimpleNamespace(
        rebuild_transformed=lambda *args: ({"input": "same"}, {}),
        score_fixed_bank=score,
        validate_selection=lambda selected, arrays, np: None,
        require_repeat_parity=parity,
    )
    bank = SimpleNamespace(
        require_parity=lambda record, name, a, b: np.testing.assert_array_equal(a, b)
    )
    published = {
        "scorer": scorer,
        "bank_helpers": bank,
        "identity_helpers": SimpleNamespace(Recorder=Recorder),
    }
    monkeypatch.setattr(hundred, "load_published", lambda name: published[name])
    spec = {"o72_diagnostic": {name: name for name in published}}
    hook = hundred.diagnostic_hook(
        spec, SimpleNamespace(replay_buffer=object()), tmp_path
    )
    if repeat_changes:
        with pytest.raises(ValueError, match="repeat changed"):
            hook(Agent(0), 0)
        failure = json.loads(
            (tmp_path / "extra-diagnostic-000/RESULT.json").read_text()
        )
        assert failure["complete"] is False and "fixed64" in failure["arrays"]
        return
    for call in (0, 10):
        agent = Agent(call)
        agents.append(weakref.ref(agent))
        result = hook(agent, call)
        assert result["complete"] and result["winner"] == 63
        assert result["fixed_scoring_calls"] == 2
        assert (
            result["candidate_generation_calls"] == result["edit_sampling_calls"] == 0
        )
        assert result["terminal_calibration"]["current_version"] == call
        del agent
    assert calls == [0, 0, 10, 10]
    assert all(ref() is None for ref in agents)


def test_full_save_wrapper_forwards_only_step100(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        historical,
        "save_final_checkpoint",
        lambda *args, **kwargs: seen.append((args, kwargs)),
    )
    agent, config = object(), {}
    hundred.save_final_checkpoint(agent, config, tmp_path)
    assert seen == [((agent, config, tmp_path), {"step": 100})]


def test_long_run_dispatch_does_not_change_shared_defaults(monkeypatch, tmp_path):
    from abc_bench import fixed_data_learning as fixed

    seen = []
    monkeypatch.setattr(hundred, "load_spec", lambda config: {})
    monkeypatch.setattr(hundred, "diagnostic_hook", lambda *args: "read_only_hook")
    monkeypatch.setattr(
        probe,
        "checked_import",
        lambda name, config: SimpleNamespace(
            run=lambda *args, **kwargs: seen.append((args, kwargs))
        ),
    )
    args = [[], object(), object(), object(), {}, tmp_path, object(), object(), 1.0]
    hundred.run(*args)
    forwarded, kwargs = seen[0]
    assert list(forwarded) == args
    assert kwargs["calls"] == 100 and kwargs["measure_at"] == (0, 10, 30, 100)
    assert (
        kwargs["recovery"] is hundred
        and kwargs["checkpoint_writer"] is hundred.save_final_checkpoint
    )
    assert kwargs["extra_measure"] == "read_only_hook"
    assert fixed.CALLS == historical.CALLS == 10
    assert fixed.MEASURE_AT == historical.MEASURE_AT == (0, 1, 10)
