"""CPU checks for the training-only phase; no policy or native execution."""

import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import fixed_data_learning as fixed
from abc_bench import historical_fixed_data_100 as diagnostic
from abc_bench import historical_fixed_data_100_training as training
from abc_bench import historical_fixed_data_learning as historical
from abc_bench import realtime_update_probe as probe
from abc_bench import subset

PREPARATION = Path("/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-data-100-training-preparation-20261019")


def initial():
    return {
        "optimizers": {key: {"step": 0, "counts": {"count": 0}, "finite_leaves": 1, "finite_bytes": 4}
                       for key in probe.COUNTERS},
        "parameters": {key: {"finite_leaves": 1, "finite_bytes": 4}
                       for key in (*probe.COUNTERS, "target_critic", "target_actor_params")},
    }


def config():
    return {"phase": training.PHASE, "fixed_data": {
        "spec": probe.file_pin(PREPARATION / "PHASE_SPEC.json"),
        "python": probe.file_pin(Path(sys.executable).resolve()),
        "condition": historical.ORDERED_CONDITION,
    }, "installed_modules": {name: {} for name in (
        training.MODULE, historical.MODULE, "abc_bench.fixed_data_learning", "expo_ft.agents.alg.agent")}}


def test_sealed_training_spec_and_legacy_dispatch():
    spec = training.load_spec(config())
    parent = json.loads(probe.authenticate(spec["parent_spec"]).read_text())
    for key in ("checkpoint_parent", "settings", "replay_acceptance", "replay_inspection",
                "episode", "runtime_versions", "reward_view", "assembly_order"):
        assert spec[key] == parent[key]
    assert spec["measure_at"] == [] and "o72_diagnostic" not in spec
    assert probe.fixed_phase_module(config()) is training
    assert training.prepare_before_restore is historical.prepare_before_restore
    assert training.prepare_replay is historical.prepare_replay
    assert probe.phase_deadlines(config()) == (9000, 9100)
    assert probe.phase_deadlines({"phase": diagnostic.PHASE}) == (7200, 7300)
    assert probe.phase_deadlines({"phase": historical.PHASE}) == (2300, 2400)
    assert diagnostic.MEASURE_AT == (0, 10, 30, 100)
    assert fixed.CALLS == historical.CALLS == 10
    assert fixed.MEASURE_AT == historical.MEASURE_AT == (0, 1, 10)


@pytest.mark.parametrize("key,value", [("measure_at", [0, 100]), ("update_calls", 101),
    ("worker_seconds", 9001), ("diagnostics_in_training", True),
    ("checkpoint_before_final_integrity", False), ("reward_view", "shaped")])
def test_changed_training_contract_rejected(tmp_path, monkeypatch, key, value):
    cfg = config()
    spec = json.loads(Path(cfg["fixed_data"]["spec"]["path"]).read_text())
    spec[key] = value
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    cfg["fixed_data"]["spec"] = probe.file_pin(path)
    monkeypatch.setattr(training, "SPEC_SHA256", cfg["fixed_data"]["spec"]["sha256"])
    with pytest.raises(ValueError, match="bounded100"):
        training.load_spec(cfg)


def test_driver_reuses_initial_inventory_without_measurement_hook(tmp_path, monkeypatch):
    before = initial()
    (tmp_path / "before.json").write_text(json.dumps(before))
    seen = []
    monkeypatch.setattr(probe, "checked_import", lambda *args: SimpleNamespace(
        run=lambda *args, **kwargs: seen.append(kwargs)))
    training.run([], object(), object(), object(), config(), tmp_path, object(), object(), 1.)
    assert seen[0]["collect_diagnostics"] is False and seen[0]["measure_at"] == ()
    assert seen[0]["initial_integrity"] == before and "extra_measure" not in seen[0]
    assert seen[0]["calls"] == 100
    assert seen[0]["recovery"] is training


def test_nonzero_initial_optimizer_rejected(tmp_path):
    before = initial()
    before["optimizers"]["critic"]["step"] = 1
    (tmp_path / "before.json").write_text(json.dumps(before))
    with pytest.raises(ValueError, match="fresh"):
        training.initial_integrity(tmp_path)


def test_original_full_checkpoint_saver_receives_step100(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(historical, "save_final_checkpoint", lambda *args, **kwargs: seen.append((args, kwargs)))
    agent, cfg = object(), {}
    training.save_final_checkpoint(agent, cfg, tmp_path)
    assert seen == [((agent, cfg, tmp_path), {"step": 100})]


@pytest.mark.parametrize("save_first", [False, True])
def test_save_order_preserves_legacy_and_checkpoint_survives_final_scan_failure(tmp_path, save_first):
    events = []
    path = tmp_path / "checkpoint.json"

    def save(agent):
        events.append("save")
        path.write_text("full saver fixture")
        return {"step": 100}

    def optimizers(agent):
        events.append("optimizer_inventory")
        raise ValueError("final inventory fixture failure")

    with pytest.raises(ValueError, match="inventory fixture"):
        fixed.finalize_training(object(), optimizers, None, save, calls=100, save_first=save_first)
    assert path.exists() is save_first
    assert events == (["save", "optimizer_inventory"] if save_first else ["optimizer_inventory"])


def sealed_job_fixture(tmp_path):
    # Existing qualified paths are used only by the actual read-only admission.
    old = PREPARATION.parent / "fixed-data-100-release-01/JOB_V2.json"
    job = json.loads(old.read_text())
    cfg = json.loads(Path(job["arguments"][1]).read_text())
    cfg.update(phase=training.PHASE, worker_seconds=9000, parent_seconds=9100)
    cfg["fixed_data"]["spec"] = probe.file_pin(PREPARATION / "PHASE_SPEC.json")
    cfg["installed_modules"][training.MODULE] = probe.file_pin(Path(training.__file__))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    job.update(max_wall_s=9100, arguments=["--config", str(path)])
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    return job_path


def test_actual_loader_then_effective_parent_for_sealed_training_phase(tmp_path, monkeypatch):
    path = sealed_job_fixture(tmp_path)
    job = subset.load_job(path)
    monkeypatch.setattr(probe, "host_available", lambda: probe.MIN_START_BYTES)
    probe.validate_job(job, 9100)
    with pytest.raises(ValueError, match="cleanup"):
        probe.validate_job(job, 9099)


@pytest.mark.parametrize("change", [{"max_wall_s": 9101}, {"max_wall_s": 9100.0},
    {"max_wall_s": 7300}, {"module": "abc_author_subset.host"}, {"python": "/usr/bin/python3"}])
def test_training_parent_exception_rejects_misbound_jobs(tmp_path, change):
    path = sealed_job_fixture(tmp_path)
    job = json.loads(path.read_text())
    job.update(change)
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError):
        subset.load_job(path)


def test_legacy_hundred_job_cannot_take_training9100_exception(tmp_path):
    old = PREPARATION.parent / "fixed-data-100-release-01/JOB_V2.json"
    job = json.loads(old.read_text())
    job["max_wall_s"] = 9100
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="parent deadline"):
        subset.load_job(path)


def test_full_shared_training_route_never_accesses_diagnostic_dataset_or_helpers(tmp_path, monkeypatch):
    """Exercise all100 orchestration calls with explicit CPU doubles, not a learner."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Expensive diagnostic must not run")

    jax = SimpleNamespace(numpy=np, device_get=lambda x: x, device_put=lambda x, *_: x,
                         block_until_ready=lambda x: x, random=SimpleNamespace(PRNGKey=lambda x: x))
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    monkeypatch.setitem(sys.modules, "expo_ft.agents.alg.batch_utils", SimpleNamespace(prepare_critic_batch=forbidden))
    monkeypatch.setitem(sys.modules, "expo_ft.agents.alg.realtime_expo_ft", SimpleNamespace(_jitted_fast_select=forbidden, batch_encode=forbidden))
    monkeypatch.setitem(sys.modules, "expo_ft.agents.vla.pi05", SimpleNamespace(_jitted_sample_with_prefix=forbidden))
    monkeypatch.setitem(sys.modules, "sadhana_terminal_critic.contracts", SimpleNamespace(
        batch_fingerprint=forbidden, check_terminal_target=forbidden, fixed_window_view=forbidden, terminal_starts=forbidden))
    monkeypatch.setattr(probe, "authenticate_inputs", lambda *_: {})
    events = []
    stats = SimpleNamespace(mean=np.array(0), std=np.array(1), q01=np.array(-1), q99=np.array(1))
    data_config = SimpleNamespace(norm_stats={"state": stats, "actions": stats}, use_quantile_norm=True)

    class Buffer:
        rows = 7069
        _data_config = data_config

        def __len__(self):
            return self.rows

    class Processor:
        def __init__(self):
            self.replay_buffer, self.offline_replay_buffer = Buffer(), []
            self.replay_iterator = self.iterator()
            self.calls = 0

        def iterator(self):
            queue = deque(self.batch(1280, i) for i in range(1, 4))
            next_id = 4
            while queue:
                yield queue.popleft()
                queue.append(self.batch(1280, next_id))
                next_id += 1

        def restore(self, parent, up_to_step):
            assert up_to_step == 9908
            self.replay_buffer.rows += 9899

        def next_batch(self, rng):
            self.calls += 1
            critic = next(self.replay_iterator)
            return critic, self.batch(64, critic["fixture_id"]), rng+1

        @staticmethod
        def batch(n, cid):
            return {"actions": np.zeros((n, 1)), "rewards": np.zeros(n), "masks": np.ones(n),
                    "dones": np.zeros(n), "valids": np.ones(n), "fixture_id": cid}

    class Agent:
        actor = SimpleNamespace(data_config=data_config, replicated_sharding="fixture")

        def __init__(self, step=0):
            self.step, self.rng = step, step+42
            for key, mult in probe.COUNTERS.items():
                setattr(self, key, SimpleNamespace(step=step*mult))

        def replace(self, **kwargs):
            assert kwargs == {"rng": self.rng}
            return self

        def update(self, actual, critic, utd, actor):
            assert actual is self and utd == 20
            assert critic["fixture_id"] == actor["fixture_id"] == self.step+1
            events.append(f"update{self.step+1}")
            return Agent(self.step+1), {"next_q_nan_ratio": 0.}

    class Dataset:
        @property
        def episodes(self):
            return forbidden()

    processor = Processor()

    def prepare_before(*args):
        processor.replay_buffer.rows += 349
        return {"append": {"rows": 349}}

    recovery = SimpleNamespace(
        PHASE=training.PHASE, STATUS=training.STATUS,
        load_spec=lambda _: {"checkpoint_parent": "fixture"},
        prepare_before_restore=prepare_before, prepare_replay=lambda *args, **kwargs: {},
        authenticate_data=lambda _: {"files": [], "replay_bytes": 0, "recovery_inputs": {}},
    )

    def save(agent, cfg, output):
        assert agent.step == 100
        events.append("save100")
        return {"step": 100}

    def optimizer(agent):
        events.append("final_optimizer_inventory")
        return {key: {"step": 100*mult, "counts": {"count": 100*mult}}
                for key, mult in probe.COUNTERS.items()}

    def parameters(agent):
        events.append("final_parameter_inventory")
        return {"finite_fixture": True}

    result = fixed.run([Agent()], processor, Dataset(), None, {"phase": training.PHASE}, tmp_path,
                       optimizer, parameters, 0., recovery=recovery, checkpoint_writer=save,
                       calls=100, measure_at=(), collect_diagnostics=False, initial_integrity=initial())
    assert events == [*(f"update{i}" for i in range(1, 101)), "save100",
                      "final_optimizer_inventory", "final_parameter_inventory"]
    assert result["measurements"] == {} and result["arrays"] == []
    assert result["sampling"]["consumed_critic_rows"] == 128000
    assert result["sampling"]["consumed_actor_rows"] == 6400
    assert processor.calls == 101
    assert not list(tmp_path.glob("measurement*")) and not (tmp_path / "frozen-inputs.json").exists()
