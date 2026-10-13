"""Installed CPU boundary checks; all native/process doubles are explicit."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from abc_bench import fasttd3_compiler_cache_diagnostic as stage
from abc_bench.subset import ALLOWED_MODULES, load_job
from abc_bench.upstream_eval import artifact

BUILD = Path(
    "/home/user/aditya/RL/builds/abc-fasttd3-compiler-cache-diagnostic-20261013"
)


def test_installed_real_metadata_preparation_is_not_native_or_training():
    config, host, evidence = stage.load_config(
        BUILD / "prepared-cpu-metadata/config.json"
    )
    job = load_job(BUILD / "prepared-cpu-metadata/job.json")
    assert host["num_envs"] == 64 and config["ranks"] == [0, 55]
    assert len(evidence["worker_resources"]) == 64
    assert job["module"] == stage.MODULE and job["max_wall_s"] == 4200
    assert all(name not in sys.modules for name in ("mujoco", "torch", "jax", "warp"))
    assert stage.MODULE in ALLOWED_MODULES


@pytest.mark.parametrize(
    "kind",
    ["rank", "mode", "resets", "controls", "action", "limits", "source", "python"],
)
def test_config_changes_rejected_before_native_import(tmp_path, kind):
    value = json.loads((BUILD / "prepared-cpu-metadata/config.json").read_text())
    if kind == "rank":
        value["ranks"] = [0, 1]
    elif kind == "mode":
        value["cache_modes"] = ["default"]
    elif kind in {"resets", "controls"}:
        value["resets_per_case" if kind == "resets" else "controls_per_reset"] = True
    elif kind == "action":
        value["normalized_action14"][0] = 1
    elif kind == "limits":
        value["limits"]["parent_seconds"] = 7201
    elif kind == "source":
        value["source_pins"][0]["sha256"] = "0" * 64
    else:
        value["python"] = "/usr/bin/python"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        stage.load_config(path)
    assert "mujoco" not in sys.modules


def test_live_lease_required_before_output_or_child(monkeypatch, tmp_path):
    def rejected():
        raise RuntimeError("explicit lease rejection fixture")

    monkeypatch.setattr(stage, "require_subset_lease", rejected)
    with pytest.raises(RuntimeError, match="lease rejection"):
        stage.run(BUILD / "prepared-cpu-metadata/config.json", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_internal_case_requires_owned_cpu_parent_before_native(tmp_path):
    with pytest.raises(RuntimeError, match="owned CPU-only"):
        stage.run_case(
            BUILD / "prepared-cpu-metadata/config.json",
            tmp_path / "out",
            55,
            "disabled",
        )
    assert not (tmp_path / "out").exists() and "mujoco" not in sys.modules


def test_one_child_floor_is_explicit_and_does_not_change_public_guard(tmp_path):
    evidence = {"worker_resources": [{"measurement": {"maxrss_kib": 3828776}}]}
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 70606218 kB\n")
    report = stage.diagnostic_memory_guard(evidence, meminfo)
    assert report["required_kib"] == 70606218
    assert "UNVALIDATED" in report["scope"]
    assert report["production_memory_admission_changed"] is False
    meminfo.write_text("MemAvailable: 70606217 kB\n")
    with pytest.raises(MemoryError):
        stage.diagnostic_memory_guard(evidence, meminfo)
    from abc_bench.fasttd3_vector_stage import memory_admission

    with pytest.raises(ValueError, match="64/128"):
        memory_admission(evidence, count=1, meminfo=meminfo)


@pytest.mark.parametrize("mode", ["default", "disabled"])
def test_explicit_api_double_sets_capacity_only_before_factory(mode):
    calls, cache = [], object()
    capacity = [500 * 1024**2]

    def set_capacity(actual, value):
        assert actual is cache
        calls.append(("set", value))
        capacity[0] = value
        return value

    api = SimpleNamespace(
        mj_getCache=lambda: cache,
        mj_getCacheCapacity=lambda actual: capacity[0],
        mj_getCacheSize=lambda actual: 0,
        mj_setCacheCapacity=set_capacity,
    )
    setup = stage.set_cache_mode(api, mode)
    calls.append(("explicit_world_factory_double", capacity[0]))
    assert setup["initial_occupied_bytes"] == 0
    assert calls == (
        [("set", 0), ("explicit_world_factory_double", 0)]
        if mode == "disabled"
        else [("explicit_world_factory_double", 500 * 1024**2)]
    )


@pytest.mark.parametrize("problem", ["capacity", "occupancy", "disable"])
def test_nonfresh_or_failed_public_cache_disable_rejected(problem):
    api = SimpleNamespace(
        mj_getCache=lambda: object(),
        mj_getCacheCapacity=lambda cache: 1 if problem == "capacity" else 500 * 1024**2,
        mj_getCacheSize=lambda cache: 1 if problem == "occupancy" else 0,
        mj_setCacheCapacity=lambda cache, size: 1,
    )
    with pytest.raises(RuntimeError):
        stage.set_cache_mode(api, "disabled")


@pytest.mark.parametrize("terminal", [False, True])
def test_public_step_observer_calls_once_returns_identical_tuple_before_autoreset(
    terminal,
):
    events = []
    returned = (object(), 0.0, terminal, False, {"task_eval": {"success": terminal}})

    def original(action, *, render_obs):
        events.append(("original", action, render_obs))
        return returned

    world = SimpleNamespace(env=SimpleNamespace(step=original))
    stage.install_step_observer(
        world, lambda result: events.append(("observe", result is returned))
    )
    result = world.env.step([0.0] * 14, render_obs=False)
    if terminal:
        events.append(("explicit_original_world_autoreset_double",))
    assert result is returned
    assert events[:2] == [("original", [0.0] * 14, False), ("observe", True)]
    assert sum(e[0] == "original" for e in events) == 1


def test_failed_original_step_is_never_retried_or_observed():
    events = []

    def original(*args, **kwargs):
        events.append("original")
        raise FloatingPointError("explicit native step failure double")

    world = SimpleNamespace(env=SimpleNamespace(step=original))
    stage.install_step_observer(world, lambda result: events.append("observe"))
    with pytest.raises(FloatingPointError):
        world.env.step([0.0] * 14)
    assert events == ["original"]


def physics():
    return {
        "time_finite": True,
        "finite": {"qpos": True},
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
    }


@pytest.mark.parametrize("warning", list(physics()["warnings"]))
def test_every_native_warning_count_is_fatal(warning):
    value = physics()
    value["warnings"][warning]["number"] = 1
    with pytest.raises(FloatingPointError):
        stage.valid_physics(value)


@pytest.mark.parametrize(
    "changed", [None, "integration", "model", "reward", "judge", "ctrl"]
)
def test_explicit_byte_fixtures_compare_exact_native_components_without_admission(
    tmp_path, changed
):
    state_a, state_b, xml = (
        tmp_path / "a.bin",
        tmp_path / "b.bin",
        tmp_path / "scene.xml",
    )
    state_a.write_bytes(b"explicit CPU fixture integration")
    state_b.write_bytes(
        b"different" if changed == "integration" else state_a.read_bytes()
    )
    xml.write_bytes(b"<explicit_CPU_fixture/>")
    left = {
        "label": "explicit_CPU_fixture",
        "physics": physics(),
        "integration": artifact(state_a),
        "reset_scene_xml": artifact(xml),
        "model": "source digest fixture",
        "reward": 0.0,
        "judge": False,
        "ctrl": [0.0] * 14,
    }
    right = copy.deepcopy(left)
    right["integration"] = artifact(state_b)
    if changed not in {None, "integration"}:
        right[changed] = "changed explicit fixture"
    differences = stage.compare_records([left], [right])
    assert bool(differences) == (changed is not None)
    # This pure function returns differences only, not a producer/qualification.
    assert isinstance(differences, list)


def test_declared_pin_is_not_enough_when_actual_bytes_change(tmp_path):
    path = tmp_path / "state.bin"
    path.write_bytes(b"explicit fixture")
    original = artifact(path)
    path.write_bytes(b"changed explicit fixture")
    row = {
        "label": "fixture",
        "physics": physics(),
        "integration": original,
        "reset_scene_xml": original,
    }
    with pytest.raises(ValueError):
        stage.compare_records([row], [row])


@pytest.mark.parametrize(
    "problem", ["partial", "fixture", "wrong_kind", "missing_cache", "wrong_capacity"]
)
def test_failed_or_fixture_case_cannot_be_accepted_as_equivalent(problem):
    # Explicit receipt shape double; no native success was fabricated or run.
    value = {
        "schema": stage.SCHEMA,
        "kind": "native_case_resource_only",
        "status": "completed",
        "cache_mode": "default",
        "cpu_visibility": "",
        "renderer": "disable",
        "training_updates": 0,
        "qualification": False,
        "records": [{}] * 39,
    }
    if problem == "partial":
        value["status"] = "failed"
    elif problem == "fixture":
        value["explicit_CPU_fixture"] = True
    elif problem == "wrong_kind":
        value["kind"] = "training_producer"
    elif problem == "wrong_capacity":
        value["resources"] = [{"point": "before_model", "cache_capacity_bytes": 0}] * 41
    with pytest.raises(ValueError):
        stage.compare_cases(value, copy.deepcopy(value))


@pytest.mark.parametrize("failure", ["exit", "timeout", "cancel"])
def test_owned_failed_process_is_reaped_and_failed_receipt_published(
    monkeypatch, tmp_path, failure
):
    from abc_bench import fasttd3_vector_stage as cleanup

    config = {"python": sys.executable, "source_pins": []}  # Explicit metadata double.
    monkeypatch.setattr(stage, "require_subset_lease", lambda: None)
    monkeypatch.setattr(stage, "load_config", lambda path: (config, {}, {}))
    monkeypatch.setattr(
        stage,
        "diagnostic_memory_guard",
        lambda evidence: {"explicit_CPU_estimate_double": True},
    )
    monkeypatch.setattr(stage, "process_start_ticks", lambda pid: 123)
    monkeypatch.setattr(stage.os, "getpgid", lambda pid: stage.os.getpgrp())
    events = []

    class ExplicitOwnedProcessDouble:
        pid = 999999
        returncode = None

        def __init__(self, *args, **kwargs):
            events.append("spawn_double")
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""

        def wait(self, timeout):
            if self.returncode is not None:
                return self.returncode
            if failure == "timeout":
                raise subprocess.TimeoutExpired("explicit child double", timeout)
            if failure == "cancel":
                raise InterruptedError("explicit cancellation double")
            self.returncode = 7
            return 7

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate_owned_double")
            self.returncode = -15

    monkeypatch.setattr(stage.subprocess, "Popen", ExplicitOwnedProcessDouble)
    # The actual published cleanup function runs against this explicit double.
    assert callable(cleanup.stop_process)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired, InterruptedError)):
        stage.run(config_path, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["qualification"] is False
    assert (
        receipt["benchmark_score"] is None
        and receipt["full64_memory_sufficient"] is None
    )
    assert receipt["cases"][0]["owned_child_reaped"] is True
    assert events.count("spawn_double") == 1
    assert events.count("terminate_owned_double") == (0 if failure == "exit" else 1)


def test_generic_allowlist_timeout_bounds_unchanged(tmp_path):
    job = json.loads((BUILD / "prepared-cpu-metadata/job.json").read_text())
    job["max_wall_s"] = 7201
    path = tmp_path / "overlong-job.json"
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="7200"):
        load_job(path)


def test_actual_diagnostic_configuration_cannot_admit_full_training():
    from abc_bench import fasttd3_campaign_stage as full

    with pytest.raises(ValueError):
        full.load_config(BUILD / "prepared-cpu-metadata/config.json")
    assert "mujoco" not in sys.modules and "torch" not in sys.modules
