"""Installed CPU boundary tests. Native doubles below do not prove MuJoCo replay."""

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import fasttd3_reach_matched as matched
from abc_bench import fasttd3_reach_matched_stage as stage
from abc_bench.fasttd3_reach_contract import ReachEpisode
from abc_bench.subset import load_job

BUILD = Path("/home/user/aditya/RL/builds/abc-fasttd3-reach-matched-20261013")


def config():
    return json.loads((BUILD / "matched-host.json").read_text())


def test_actual_published_train_sources_are_pinned_and_load_without_native_import():
    value, episodes = matched.load_config(BUILD / "matched-host.json")
    assert value == config()
    assert [episode["seed"] for episode in episodes] == [9030, 9031, 9032]
    assert all(episode["split"] == "train" for episode in episodes)
    for episode in episodes:
        arrays = matched.array_inputs(episode)
        assert arrays["initial_integration"].shape == (531,)
        assert arrays["initial_integration"].dtype == np.float64
    assert "mujoco" not in sys.modules and "torch" not in sys.modules


@pytest.mark.parametrize(
    "invalid", ["VAL", "duplicate", "reward", "horizon", "watchdog", "model_pin"]
)
def test_bad_capture_selection_rejected_before_native_loading(tmp_path, invalid):
    value = config()
    if invalid == "VAL":
        cohort = matched.checked_json(value["cohort"])
        value["cases"][0] = cohort["episodes"][3]
    elif invalid == "duplicate":
        value["cases"][1] = value["cases"][0]
    elif invalid in {"reward", "horizon"}:
        value["reach"]["scale_m" if invalid == "reward" else "horizon_controls"] = 0.2
    elif invalid == "watchdog":
        value["timeouts"]["max_requests"] = 801
    else:
        value["cases"][0]["manifest"]["sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        matched.load_config(path)
    assert "mujoco" not in sys.modules


def world_double(tmp_path, *, native_success=False):
    world = matched.MatchedWorld.__new__(matched.MatchedWorld)
    world.closed, world.active, world.controls, world.resets = False, True, 0, 1
    world.episode_steps, world.peak_native_bottles, world.native_ever_success = (
        0,
        0,
        False,
    )
    world.actual_seed, world.output, world.trace, world.episodes = (
        9030,
        tmp_path,
        io.StringIO(),
        [],
    )
    world.schema = {
        "action_schema": {
            "policy_low": [-2.0] * 6 + [0.0] + [-2.0] * 6 + [0.0],
            "policy_high": [2.0] * 6 + [1.0] + [2.0] * 6 + [1.0],
        }
    }
    world.reset_proof = {"explicit_CPU_fixture": True}
    world.reaching = ReachEpisode(0.4)
    world.initial_physics = {"explicit_CPU_fixture": True}
    world.check_physics = lambda before=None: {
        "time": (world.episode_steps + (before is not None)) * 0.034
    }
    world.integration = lambda: np.zeros(531)
    world.observation = lambda: [float(world.episode_steps)] * 216
    world.geometry = SimpleNamespace(
        snapshot=lambda *args: SimpleNamespace(
            bottle_centers=np.array([[0.1, 0.0, 0.0]] * 5),
            grasp_positions=np.zeros((2, 3)),
        )
    )

    class NativeEnvDouble:
        model, data = object(), object()
        calls = 0

        def step(self, physical, *, render_obs):
            assert render_obs is False
            self.calls += 1
            self.last_physical = physical
            count = 5 if native_success else 0
            judge = {
                "num_active_bottles": 5,
                "num_bottles_in_bin": count,
                "success": native_success,
                "bottle_in_bin_mask": [native_success] * 5,
            }
            return None, count / 5, False, self.calls == 128, {"task_eval": judge}

    world.env = NativeEnvDouble()
    return world


def test_fixed128_timeout_no_autoreset_and_reach_does_not_set_native_success(tmp_path):
    world = world_double(tmp_path)
    for control in range(1, 129):
        result = world.step([[0.0] * 14])
        assert result["terminated"] == [False]
        assert result["truncated"] == [control == 128]
        assert result["obs"] == result["terminal_obs"]
        assert result["autoresets"] == [None]
    episode = result["episodes"][0]
    assert episode["reach_diagnostic"]["qualified_reach_achieved"] is True
    assert episode["success"] is episode["native_full_success"] is False
    assert world.env.calls == 128 and world.resets == 1
    with pytest.raises(RuntimeError, match="active"):
        world.step([[0.0] * 14])


def test_native_five_success_does_not_shorten_reach_diagnostic(tmp_path):
    world = world_double(tmp_path, native_success=True)
    result = world.step([[0.0] * 14])
    assert result["episodes"] == [] and world.active is True
    for _ in range(127):
        result = world.step([[0.0] * 14])
    assert result["episodes"][0]["success"] is True
    assert result["episodes"][0]["native_full_success_any_control"] is True


@pytest.mark.parametrize(
    "action", [[[float("nan")] * 14], [[1e300] * 14], [0.0] * 14, [[0.0] * 13]]
)
def test_invalid_actions_do_not_advance_native_fixture(tmp_path, action):
    world = world_double(tmp_path)
    with pytest.raises(ValueError):
        world.step(action)
    assert world.env.calls == 0 and world.controls == 0


def test_mapping_matches_existing_float32_action_boundary(tmp_path):
    world = world_double(tmp_path)
    action = np.linspace(-2, 2, 14, dtype=np.float32)
    world.step([action.tolist()])
    schema = world.schema["action_schema"]
    low, high = np.asarray(schema["policy_low"]), np.asarray(schema["policy_high"])
    expected = (low + (np.clip(action, -1, 1) + 1) * 0.5 * (high - low)).astype(
        np.float32
    )
    assert world.env.last_physical.tobytes() == expected.tobytes()


def test_host_and_stage_refuse_without_lease_before_mutation(tmp_path, monkeypatch):
    def reject():
        raise RuntimeError("explicit CPU fixture: no canonical lease")

    monkeypatch.setattr(stage, "require_subset_lease", reject)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="no canonical lease"):
        stage.run(tmp_path / "absent-config.json", output)
    assert not output.exists()
    monkeypatch.setattr(matched, "require_subset_lease", reject)
    monkeypatch.setattr(matched, "load_config", lambda p: (config(), []))
    with pytest.raises(RuntimeError, match="no canonical lease"):
        matched.serve(tmp_path / "absent", tmp_path / "env.sock", output)
    assert not output.exists()


def test_existing_full_task_gate_cannot_prepare_reach_job(tmp_path):
    # Genuine historical full-task files are not positive reach evidence.
    parent = Path(
        "/home/user/aditya/RL/abc/outputs/bench/runs/author-subset-20261013T005851-79817b.json"
    )
    worker = Path(
        "/home/user/aditya/RL/abc/outputs/bench/author-subset-20261013T005851-79817b/receipt.json"
    )
    with pytest.raises(ValueError, match="complete.*reach TRAIN producer"):
        matched.prepare_binding(
            parent, worker, BUILD / "matched-host.json", tmp_path / "binding.json"
        )
    assert not (tmp_path / "binding.json").exists()


def test_old_reach_config_requires_new_initial_capture_inventory():
    from abc_bench import fasttd3_reach_stage

    prior = (
        BUILD.parent / "abc-fasttd3-reach-diagnostic-20261013/runtime_gate-config.json"
    )
    with pytest.raises(ValueError, match="Incomplete diagnostic source inventory"):
        fasttd3_reach_stage.load_config(prior)


def test_separate_allowlist_keeps_generic7200_wall_bound(tmp_path):
    job = {
        "schema_version": 1,
        "module": "abc_bench.fasttd3_reach_matched_stage",
        "python": str(stage.HOST_PYTHON),
        "arguments": ["--config", str(tmp_path / "future.json")],
        "label": "TRAIN reach diagnostic",
        "max_wall_s": 1260,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert load_job(path) == job
    job["max_wall_s"] = 7201
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError):
        load_job(path)


def test_source_inventory_and_watchdogs_are_specific_and_bounded():
    sources = json.loads((BUILD / "MATCHED_SOURCE_PINS.json").read_text())
    assert stage.validate_sources(sources) == sources
    invalid = copy.deepcopy(sources)
    invalid["fast_adapter"].pop()
    with pytest.raises(ValueError, match="inventory"):
        stage.validate_sources(invalid)
    assert sum(stage.LIMITS.values()) + 90 < 1260
    assert "abc_bench/fasttd3_reach_matched_stage.py" in stage.ABC_PAYLOADS
    assert "sadhana_fasttd3_abc/reach_matched.py" in stage.FAST_PAYLOADS


def test_memory_failure_records_partial_receipt_and_launches_no_children(
    tmp_path, monkeypatch
):
    sources = json.loads((BUILD / "MATCHED_SOURCE_PINS.json").read_text())
    config = {
        "source_pins": sources,
        "binding": {"explicit_CPU_fixture": True},
        "learner_python": str(stage.LEARNER_PYTHON),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    monkeypatch.setattr(stage, "require_subset_lease", lambda: None)
    monkeypatch.setattr(
        stage, "load_config", lambda path: (config, {}, {"explicit_CPU_fixture": True})
    )

    def preflight_double(*args, **kwargs):
        (tmp_path / "result/learner-preflight.json").write_text(
            '{"explicit_CPU_fixture":true}'
        )

    def memory_double(evidence, *, count):
        assert count == 64 and evidence == {"explicit_CPU_fixture": True}
        raise MemoryError("explicit CPU fixture: insufficient RAM")

    def no_children(*args, **kwargs):
        pytest.fail("Memory rejection must precede any native/actor child")

    monkeypatch.setattr(stage.subprocess, "run", preflight_double)
    monkeypatch.setattr(stage.subprocess, "Popen", no_children)
    monkeypatch.setattr(stage.qualification, "memory_admission", memory_double)
    with pytest.raises(MemoryError, match="insufficient RAM"):
        stage.run(config_path, tmp_path / "result")
    receipt = json.loads((tmp_path / "result/receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["child_exit_codes"] == {"host": None, "learner": None}
    assert receipt["metrics"] == {}


def test_nonreach_producer_cannot_prepare_canonical_job_or_output(tmp_path):
    parent = Path(
        "/home/user/aditya/RL/abc/outputs/bench/runs/author-subset-20261013T005851-79817b.json"
    )
    worker = Path(
        "/home/user/aditya/RL/abc/outputs/bench/author-subset-20261013T005851-79817b/receipt.json"
    )
    sources = json.loads((BUILD / "MATCHED_SOURCE_PINS.json").read_text())
    with pytest.raises(ValueError, match="reach TRAIN producer"):
        stage.prepare_job(
            parent, worker, BUILD / "matched-host.json", sources, tmp_path / "job"
        )
    assert not (tmp_path / "job").exists()


def test_native_constructor_failure_records_no_completion_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(matched, "load_config", lambda path: (config(), []))
    monkeypatch.setattr(matched, "require_subset_lease", lambda: None)

    def failing_native_double(*args):
        raise ValueError("explicit CPU native-constructor fixture failure")

    monkeypatch.setattr(matched, "MatchedTask", failing_native_double)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    config_path = tmp_path / "fixture-config.json"
    config_path.write_text("{}")
    with pytest.raises(ValueError, match="fixture failure"):
        matched.serve(config_path, tmp_path / "env.sock", tmp_path / "result")
    result = json.loads((tmp_path / "result/receipt.json").read_text())
    assert result["status"] == "failed" and "controls" not in result
    assert not (tmp_path / "env.sock").exists()
    assert "mujoco" not in sys.modules and "torch" not in sys.modules
