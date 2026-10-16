"""Explicit CPU doubles; no original numerical stack, model, or simulator."""

import copy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import failure_candidate_bank as bank
from abc_bench import policy_path_probe as worker


def pin(path):
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return pin(path)


def fixture_capture(tmp_path):
    root = tmp_path / "capture"
    rgb = root / "rgb/0128.npy"
    rgb.parent.mkdir(parents=True)
    np.save(rgb, np.zeros((360, 640, 3), dtype=np.uint8))
    integration = root / "integration/0128.npy"
    integration.parent.mkdir()
    np.save(integration, np.zeros(437, dtype=np.float64))
    obs = {
        "index": 128,
        "time_s": 128 / 15,
        "state14": [0.0] * 14,
        "judge": {"success": False, "all_ok": False},
        "native_observation": {
            "left_arm_q": [0.0] * 6,
            "left_gripper_width": [0.0],
            "right_arm_q": [0.0] * 6,
            "right_gripper_width": [0.0],
        },
    }
    obs_pin = put(root / "observations/0128.json", obs)
    result_pin = put(
        root / "RESULT.json",
        {
            "status": "two_fresh_prefix_captures_equal",
            "origin_run_id": bank.ORIGIN,
            "seed": 21000,
            "boundary": {"time_s": 128 / 15},
            "files": [pin(rgb), pin(integration), obs_pin],
        },
    )
    accepted = put(
        tmp_path / "ACCEPTED.json",
        {
            "status": "accepted_reproducible_prefix_capture_only",
            "artifact_accepted": True,
            "origin_run_id": bank.ORIGIN,
            "seed": 21000,
            "controls_per_pass": 128,
            "capture_directory": str(root),
            "evidence": [result_pin],
        },
    )
    return {
        "capture_acceptance": accepted,
        "capture_result": result_pin,
        "boundary_rgb": pin(rgb),
        "boundary_observation": obs_pin,
        "boundary_integration": pin(integration),
    }


def test_causal_capture_input_has_only_rgb_and_measured_state(tmp_path):
    config = fixture_capture(tmp_path)
    raw = bank.capture_observation(config, np)
    assert set(raw) == {"head_left", "state"}
    assert raw["state"].dtype == np.float32 and raw["head_left"].dtype == np.uint8


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_seed", 20044),
        ("episode_control", 127),
        ("initialization_seed", 43),
        ("boot_delay", False),
        ("boot_delay", 5),
        ("prefix_padded", []),
        ("n_edit_samples", 0),
        ("branch_random_seed", 43),
    ],
)
def test_fixed_context_cannot_change(field, value):
    config = {"schema": bank.INPUT_SCHEMA, "fixed": copy.deepcopy(bank.FIXED)}
    bank.validate_fixed(config)
    config["fixed"][field] = value
    with pytest.raises(ValueError):
        bank.validate_fixed(config)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_origin",
        "unaccepted",
        "unjoined_result",
        "unjoined_rgb",
        "tamper",
        "state",
        "terminal",
    ],
)
def test_bad_capture_refused(tmp_path, mutation):
    config = fixture_capture(tmp_path)
    if mutation in ("wrong_origin", "unaccepted", "unjoined_result"):
        p = Path(config["capture_acceptance"]["path"])
        value = json.loads(p.read_text())
        value[
            {
                "wrong_origin": "origin_run_id",
                "unaccepted": "artifact_accepted",
                "unjoined_result": "evidence",
            }[mutation]
        ] = {"wrong_origin": "other", "unaccepted": False, "unjoined_result": []}[
            mutation
        ]
        config["capture_acceptance"] = put(p, value)
    elif mutation == "unjoined_rgb":
        other = tmp_path / "other.npy"
        other.write_bytes(Path(config["boundary_rgb"]["path"]).read_bytes())
        config["boundary_rgb"] = pin(other)
    elif mutation == "tamper":
        Path(config["boundary_rgb"]["path"]).write_bytes(b"changed")
    else:
        p = Path(config["boundary_observation"]["path"])
        value = json.loads(p.read_text())
        if mutation == "state":
            value["state14"][0] = 1.0
        else:
            value["judge"]["all_ok"] = True
        old = config["boundary_observation"]
        config["boundary_observation"] = put(p, value)
        result_path = Path(config["capture_result"]["path"])
        result = json.loads(result_path.read_text())
        result["files"] = [
            config["boundary_observation"] if x == old else x for x in result["files"]
        ]
        config["capture_result"] = put(result_path, result)
        acceptance_path = Path(config["capture_acceptance"]["path"])
        acceptance = json.loads(acceptance_path.read_text())
        acceptance["evidence"] = [config["capture_result"]]
        config["capture_acceptance"] = put(acceptance_path, acceptance)
    with pytest.raises(ValueError):
        bank.capture_observation(config, np)


def test_original_output_shape_zero_padding_and_final_float32():
    observed = []

    def transform(value):
        observed.append(value.copy())
        return value.astype(np.float64) * 0.12345

    jnp = SimpleNamespace(
        concatenate=np.concatenate,
        zeros=np.zeros,
        asarray=lambda x: np.asarray(x, dtype=np.float32),
    )
    windows = np.ones((64, 8, 14), dtype=np.float32)
    value = bank.physical_windows(
        SimpleNamespace(process_transformed_outputs=transform), windows, jnp
    )
    assert observed[0].shape == (64, 16, 14)
    assert not observed[0][:, 8:].any()
    assert value.shape == windows.shape and value.dtype == np.float32


def test_branch_order_rng_isolation_and_physical_duplicate_aliases():
    windows = np.arange(64 * 8 * 14, dtype=np.float32).reshape(64, 8, 14)
    before = random.getstate()
    result = bank.branch_set(windows, 35, 2, np)
    assert random.getstate() == before
    assert [x["candidate_index"] for x in result["aliases"]] == [0, 35, 2, 14]
    windows[35] = windows[0]
    windows[14] = windows[2]
    result = bank.branch_set(windows, 35, 2, np)
    assert len(result["unique"]) == 2
    assert result["unique"][0]["aliases"] == ["base0", "full_winner"]
    assert result["unique"][1]["aliases"] == ["no_edit_winner", "seeded_uniform64"]
    for bad in (windows.astype(np.float64), windows[:32], windows * np.nan):
        with pytest.raises(ValueError):
            bank.branch_set(bad, 35, 2, np)


class RecordDouble:
    def __init__(self):
        self.values, self.comparisons = {}, []

    def array(self, name, value):
        value = np.asarray(value)
        assert np.isfinite(value).all()
        self.values[name] = value.copy()
        return value

    def tree(self, name, value):
        self.values[name] = copy.deepcopy(value)

    def compare(self, name, left, right, exact):
        a, b = np.asarray(left), np.asarray(right)
        result = {
            "name": name,
            "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "bitwise_equal": a.shape == b.shape
            and a.dtype == b.dtype
            and a.tobytes() == b.tobytes(),
        }
        self.comparisons.append(result)
        return result


def inference_double(bad_public=False):
    calls = []
    cached = np.arange(32 * 16 * 14, dtype=np.float32).reshape(32, 16, 14)
    constructor = np.array([101, 202], dtype=np.uint32)
    selector = constructor + 7
    jax = SimpleNamespace(
        random=SimpleNamespace(split=lambda key: np.stack([key + 1, key + 2]))
    )
    actor = SimpleNamespace(
        process_raw_inputs=lambda raw, dim, size: {
            "image": {"head": raw["head_left"]},
            "state": raw["state"],
        },
        process_transformed_outputs=lambda x: x * np.float32(0.001),
    )
    selected_full = None

    def select(rng, cache, images, state, *params, **kwargs):
        nonlocal selected_full
        calls.append((rng.copy(), kwargs.copy()))
        candidates = cache[:, :8]
        edits = kwargs["n_edit_samples"]
        if edits:
            candidates = np.concatenate([candidates, candidates + np.float32(0.25)])
        index = 35 if edits else 2
        out = {
            "cand": candidates,
            "idx": np.array(index),
            "selected": candidates[index],
            "executed_padded": np.pad(candidates[index], ((0, 0), (0, 18))),
            "qs": np.arange(len(candidates), dtype=np.float32),
            "rng": rng + 10,
            "enc_all": np.zeros((len(candidates), 1)),
            "q_all": candidates.reshape(len(candidates), -1),
            "states_all": np.zeros((len(candidates), 32)),
        }
        if edits:
            selected_full = out
        return out

    def public(raw, *, precached, delay, only_base_actions):
        assert delay == 0 and precached is cached
        value = cached[0, :8] if only_base_actions else selected_full["selected"]
        command = bank.physical_windows(actor, value[None], np)[0]
        if bad_public:
            command = command + np.float32(0.5)
        rng = jax.random.split(selector if only_base_actions else selected_full["rng"])[
            0
        ]
        return (
            command,
            SimpleNamespace(rng=rng),
            {"executed_padded": np.pad(value, ((0, 0), (0, 18)))},
        )

    def precache(raw, *, prefix_padded):
        assert prefix_padded is None
        return cached, SimpleNamespace(rng=selector, sample_actions=public)

    agent = SimpleNamespace(
        rng=constructor,
        sample_pre_cache=precache,
        actor=actor,
        _infer_cache={
            "batch_encoder_params": {},
            "edit_actor_params": {},
            "target_critic_params": {},
        },
        batch_encoder=SimpleNamespace(apply_fn=None),
        edit_actor=SimpleNamespace(apply_fn=None),
        target_critic=SimpleNamespace(apply_fn=None),
        num_qs=10,
        num_min_qs=2,
        freeze_encoder=True,
        critic_camera_keys=("head",),
    )
    return agent, SimpleNamespace(_jitted_fast_select=select), jax, calls


def test_actual_constructor_and_shared_post_precache_rng_parity():
    agent, api, jax, calls = inference_double()
    record = RecordDouble()
    physical, full, noedit = bank.capture_candidates(
        agent,
        {"head_left": np.zeros(3), "state": np.zeros(14)},
        record,
        api,
        jax,
        np,
        np,
    )
    assert (full, noedit, physical.shape) == (35, 2, (64, 8, 14))
    assert [x[1]["n_edit_samples"] for x in calls] == [0, 32]
    assert np.array_equal(calls[0][0], calls[1][0])
    assert np.array_equal(calls[0][0], agent.rng + 7)
    assert np.array_equal(record.values["rng/constructor"], agent.rng)
    assert len(record.comparisons) == 10 and all(
        x["bitwise_equal"] for x in record.comparisons
    )


def test_public_discrepancy_refuses_completed_bank():
    agent, api, jax, _ = inference_double(bad_public=True)
    with pytest.raises(ValueError, match="public_physical"):
        bank.capture_candidates(
            agent,
            {"head_left": np.zeros(3), "state": np.zeros(14)},
            RecordDouble(),
            api,
            jax,
            np,
            np,
        )


def test_new_canonical_dispatch_is_fixed_and_does_not_change_old_mode(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(worker, "BUILDS", tmp_path)
    observed = []

    def payload(*args):
        observed.append(args)
        return {}, {"schema": bank.INPUT_SCHEMA, "fixed": copy.deepcopy(bank.FIXED)}

    monkeypatch.setattr(worker, "payload", payload)
    p = tmp_path / "config.json"
    for schema in (worker.SCHEMA, bank.CONFIG_SCHEMA):
        p.write_text(json.dumps({"schema": schema, "output": str(tmp_path / "fresh")}))
        worker.load_config(p)
    assert observed == [(), (worker.BANK_PAYLOAD_MANIFEST,)]
    command = worker.diagnostic_command(
        {"schema": bank.CONFIG_SCHEMA, "output": "/future"},
        {"caller": {"path": "/fixed/bank.py"}},
        {"python": {"path": "/fixed/python"}},
    )
    assert command[6] == worker.BANK_PAYLOAD_MANIFEST["path"]


def test_result_does_not_admit_task_success_or_updates():
    value = {
        "schema": bank.RESULT_SCHEMA,
        "status": "failure_candidate_bank_complete",
        "fixed": bank.FIXED,
        "origin_run_id": bank.ORIGIN,
        "artifact_accepted": False,
        "benchmark_accepted": False,
        "optimizer_steps_performed": 0,
        "native_controls": 0,
        "parity_passed": True,
        "parameters_unchanged": True,
        "counters_before": dict.fromkeys("abcdef", 0),
        "counters_after": dict.fromkeys("abcdef", 0),
        "parameters_before": dict.fromkeys("abcdefgh", "hash"),
        "parameters_after": dict.fromkeys("abcdefgh", "hash"),
        "branches": bank.branch_set(np.zeros((64, 8, 14), dtype=np.float32), 1, 2, np),
    }
    bank.validate_result(value)
    for key, changed in (
        ("artifact_accepted", True),
        ("optimizer_steps_performed", True),
        ("native_controls", 1),
        ("parity_passed", False),
        ("parameters_after", {}),
    ):
        with pytest.raises(ValueError):
            bank.validate_result({**value, key: changed})
