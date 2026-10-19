"""CPU-only admission and data-path doubles; no learner or simulator runtime."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import fixed_bank_scoring as scorer


def pin(path):
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_authenticated_helper_and_tamper(tmp_path):
    path = tmp_path / "helper.py"
    path.write_text("MARKER = 73\n")
    item = pin(path)
    assert scorer.load_helper(item).MARKER == 73
    path.write_text("MARKER = 74\n")
    with pytest.raises(ValueError, match="Input pin differs"):
        scorer.load_helper(item)


@pytest.mark.parametrize(
    "field,value",
    [
        ("N", 32),
        ("n_edit_samples", 32),
        ("episode_control", 128),
        ("checkpoint_step", 9909),
        ("delay", False),
    ],
)
def test_fixed_context_refuses_other_experiments(field, value):
    config = {"schema": scorer.INPUT_SCHEMA, "fixed": copy.deepcopy(scorer.FIXED)}
    scorer.validate_fixed(config)
    config["fixed"][field] = value
    with pytest.raises(ValueError):
        scorer.validate_fixed(config)


def arrays():
    candidates = np.arange(64 * 8 * 14, dtype=np.float32).reshape(64, 8, 14)
    return {
        "full/cand": candidates,
        "full/states_all": np.zeros((64, 32), np.float32),
        "rng/selector_start": np.asarray([42, 73], np.uint32),
        "transformed/O72/image/base_0_rgb": np.zeros((1, 224, 224, 3), np.float32),
        "transformed/O72/state": np.zeros((1, 32), np.float32),
    }


def selected(values):
    qs = np.arange(64, dtype=np.float32)
    return {
        "cand": values["full/cand"].copy(),
        "qs": qs,
        "idx": np.asarray(63, np.int32),
        "selected": values["full/cand"][63].copy(),
        "q_all": values["full/cand"].reshape(64, 112).copy(),
        "states_all": values["full/states_all"].copy(),
        "executed_padded": np.concatenate(
            [values["full/cand"][63], np.zeros((8, 18), np.float32)], axis=1
        ),
    }


def test_original_selector_receives_only_frozen_windows_and_same_head_key():
    values = arrays()
    calls = []
    agent = SimpleNamespace(
        action_dim=14,
        state_dim=32,
        replan_steps=8,
        num_qs=10,
        num_min_qs=2,
        freeze_encoder=True,
        critic_camera_keys=("base_0_rgb",),
        _infer_cache={
            "batch_encoder_params": "current_encoder",
            "edit_actor_params": "unused_edit",
            "target_critic_params": "current_target",
        },
        batch_encoder=SimpleNamespace(apply_fn="original_encoder"),
        edit_actor=SimpleNamespace(apply_fn="unused_edit_fn"),
        target_critic=SimpleNamespace(apply_fn="original_target"),
        actor=SimpleNamespace(infer_sharding="original_inference_device"),
    )
    transformed = {
        "image": {
            key: np.zeros((1, 224, 224, 3), np.float32)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "state": np.zeros((1, 32), np.float32),
    }

    def fast_select(rng, bank, images, state, encoder, editor, target, **kwargs):
        assert np.array_equal(rng, values["rng/selector_start"])
        assert bank.shape == (64, 16, 14) and np.array_equal(
            bank[:, :8], values["full/cand"]
        )
        assert (bank[:, 8:] == 0).all()
        assert encoder == "current_encoder" and target == "current_target"
        assert images["base_0_rgb"].shape == (1, 224, 224, 3) and state.shape == (1, 32)
        assert images is transformed["image"] and set(images) == {
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        }
        assert kwargs["N"] == 64 and kwargs["n_edit_samples"] == kwargs["delay"] == 0
        assert kwargs["num_min_qs"] == 2 and kwargs["num_qs"] == 10
        assert (
            kwargs["encoder_fn"] == "original_encoder"
            and kwargs["critic_fn"] == "original_target"
        )
        calls.append("original_selector")
        return selected(values)

    def subsample(key, params, num_sample, num_qs):
        assert key == "same_head_key" and num_sample == 2 and num_qs == 10
        assert np.array_equal(params["head_ids"], np.arange(10))
        calls.append("original_subsample")
        return {"head_ids": np.asarray([2, 8])}

    api = SimpleNamespace(
        _jitted_fast_select=fast_select, subsample_image_ensemble=subsample
    )
    placements = []

    def device_put(value, sharding):
        assert sharding == "original_inference_device"
        placements.append(value.shape)
        return value

    jax = SimpleNamespace(
        random=SimpleNamespace(split=lambda rng: ("same_head_key", "unused")),
        device_put=device_put,
    )
    metadata = {}
    actual, heads = scorer.score_fixed_bank(
        agent, values, transformed, api, jax, np, metadata
    )
    assert calls == ["original_selector", "original_subsample"]
    assert np.array_equal(heads, [2, 8])
    scorer.validate_selection(actual, values, np)
    assert placements == [(64, 16, 14), (2,), (2,)]
    assert set(metadata["images"]) == set(transformed["image"])


@pytest.mark.parametrize(
    "mutation",
    ["nan", "index", "candidate", "selected", "q_all", "states_all", "executed_padded"],
)
def test_invalid_score_outputs_fail_closed(mutation):
    values = arrays()
    result = selected(values)
    if mutation == "nan":
        result["qs"][1] = np.nan
    elif mutation == "index":
        result["idx"] = np.asarray(0, np.int32)
    elif mutation == "candidate":
        result["cand"][0, 0, 0] += 1
    elif mutation == "selected":
        result["selected"][0, 0] += 1
    else:
        result[mutation][0, 0] += 1
    with pytest.raises(ValueError):
        scorer.validate_selection(result, values, np)


def parity_helpers():
    def parity(record, name, left, right):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes() != right.tobytes()
        ):
            raise ValueError(name)

    def compare(name, left, right, exact):
        diff = left.astype(float) - right.astype(float)
        return {
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "finite": bool(np.isfinite(diff).all()),
            "max_abs": float(np.abs(diff).max()),
            "rms": float(np.sqrt(np.mean(diff**2))),
            "bitwise_equal": left.tobytes() == right.tobytes(),
        }

    return SimpleNamespace(compare=compare), SimpleNamespace(require_parity=parity)


def replay_fixture():
    candidates = np.ones((64, 8, 14), np.float32)
    reference = {
        "cand": candidates,
        "q_all": candidates.reshape(64, -1),
        "qs": np.arange(64, dtype=np.float32) / 100,
        "enc_all": np.ones((64, 512), np.float32),
        "idx": np.asarray(63, np.int32),
        "selected": candidates[63],
        "executed_padded": np.ones((8, 32), np.float32),
        "states_all": np.ones((64, 32), np.float32),
        "rng": np.asarray([1, 2], np.uint32),
    }
    return {"full/" + k: v.copy() for k, v in reference.items()}, {
        k: v.copy() for k, v in reference.items()
    }


def test_numerical_original_replay_is_not_bitwise_or_a_replacement_bank():
    values, outputs = replay_fixture()
    record, bank = parity_helpers()
    outputs["qs"][20] += 0.001
    outputs["enc_all"][0, 0] += 0.002
    outputs["cand"][40, 0, 0] += 0.0001
    result = scorer.describe_original_replay(outputs, values, record, bank, np)
    assert result["errors"]["cand"]["bitwise_equal"] is False
    assert result["errors"]["qs"]["max_abs"] > 0
    assert values["full/cand"][40, 0, 0] == 1


@pytest.mark.parametrize("mutation", ["base", "rng", "state", "idx", "nan", "dtype"])
def test_original_replay_rejects_input_or_current_output_inconsistency(mutation):
    values, outputs = replay_fixture()
    record, bank = parity_helpers()
    if mutation == "base":
        outputs["cand"][0, 0, 0] = np.nextafter(np.float32(1), np.float32(2))
    elif mutation == "rng":
        outputs["rng"][0] += 1
    elif mutation == "state":
        outputs["states_all"][0, 0] += 0.0001
    elif mutation == "idx":
        outputs["idx"] = np.asarray(62, np.int32)
    elif mutation == "nan":
        outputs["qs"][20] = np.nan
    else:
        outputs["enc_all"] = outputs["enc_all"].astype(np.float64)
    with pytest.raises(ValueError):
        scorer.describe_original_replay(outputs, values, record, bank, np)


def test_historical_score_and_rank_changes_are_diagnostic_not_gates():
    values, outputs = replay_fixture()
    record, bank = parity_helpers()
    outputs["qs"][12] += 2
    outputs["idx"] = np.asarray(12, np.int32)
    outputs["enc_all"][0, 0] += 0.01
    result = scorer.describe_original_replay(outputs, values, record, bank, np)
    assert result["current_winner"] == 12 and result["reference_winner"] == 63
    assert result["tested_candidate_order_preserved"] is False
    assert result["errors"]["qs"]["max_abs"] > 1
    fixed = scorer.describe_fixed_score_history(outputs, values, record, bank, np)
    assert fixed["current_winner"] == 12


def test_same_process_repeat_rejects_even_one_bit_difference():
    _, outputs = replay_fixture()
    repeated = {k: v.copy() for k, v in outputs.items()}
    record, bank = parity_helpers()
    scorer.require_repeat_parity(outputs, repeated, record, bank, "fresh")
    repeated["qs"][20] = np.nextafter(repeated["qs"][20], np.float32(2))
    with pytest.raises(ValueError, match="fresh/fixed64_repeat/qs"):
        scorer.require_repeat_parity(outputs, repeated, record, bank, "fresh")


def complete_result():
    stages = []
    for name in ("fresh", "c10"):
        counts = (
            {k: 0 for k in scorer.C10_COUNTERS}
            if name == "fresh"
            else dict(scorer.C10_COUNTERS)
        )
        params = {str(i): {"sha256": "fixture", "leaf_count": 1} for i in range(8)}
        stages.append(
            {
                "stage": name,
                "winner": 44,
                "head_ids": [6, 4],
                "counters_before": counts,
                "counters_after": counts.copy(),
                "parameters_before": params,
                "parameters_after": copy.deepcopy(params),
                "rng_unchanged": True,
                "fixed_scoring_calls": 2,
                "fixed_scoring_repeat_bitwise_equal": True,
            }
        )
    return {
        "schema": scorer.RESULT_SCHEMA,
        "status": "fixed_bank_scoring_complete",
        "fixed": scorer.FIXED.copy(),
        "artifact_accepted": False,
        "benchmark_accepted": False,
        "historical_comparisons_diagnostic_only": True,
        "fresh_parameter_identity_passed": True,
        "transformed_inputs_exact": True,
        "optimizer_steps_performed": 0,
        "native_controls": 0,
        "candidate_generation_calls": 0,
        "edit_sampling_calls": 1,
        "checkpoint_writes": 0,
        "stages": stages,
    }


@pytest.mark.parametrize(
    "mutation",
    ["counters", "parameters", "head_ids", "rng", "parity", "learning", "repeat"],
)
def test_invalid_stage_receipts_are_rejected(mutation):
    result = complete_result()
    scorer.validate_result(result)
    if mutation == "counters":
        result["stages"][1]["counters_after"]["critic"] = 201
    elif mutation == "parameters":
        result["stages"][1]["parameters_after"]["0"]["sha256"] = "changed"
    elif mutation == "head_ids":
        result["stages"][1]["head_ids"] = [3, 7]
    elif mutation == "rng":
        result["stages"][1]["rng_unchanged"] = False
    elif mutation == "parity":
        result["historical_comparisons_diagnostic_only"] = False
    elif mutation == "repeat":
        result["stages"][1]["fixed_scoring_repeat_bitwise_equal"] = False
    else:
        result["optimizer_steps_performed"] = 1
    with pytest.raises(ValueError):
        scorer.validate_result(result)


def test_unaccepted_bank_stops_before_checkpoint_reads(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"schema": "bank"}))
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "status": "accepted_original_acquisition_bank",
                "artifact_accepted": False,
                "bank_result": pin(path),
            }
        )
    )
    config = {"bank_result": pin(path), "bank_acceptance": pin(acceptance)}
    bank = SimpleNamespace(
        validate_result=lambda r: None, ACQUISITION_RESULT_SCHEMA="bank"
    )
    with pytest.raises(ValueError, match="Require accepted original acquisition bank"):
        scorer.accepted_inputs(config, bank, np)


def test_historical_fixed_score_report_still_requires_exact_candidate_inputs():
    values, outputs = replay_fixture()
    record, bank = parity_helpers()
    outputs["cand"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="fresh/fixed64/cand"):
        scorer.describe_fixed_score_history(outputs, values, record, bank, np)


@pytest.mark.parametrize(
    "mutation", [None, "image", "state", "prompt", "mask", "missing_camera", "optional"]
)
def test_original_preprocessing_requires_complete_exact_recorded_tree(mutation):
    tree = {
        "image": {
            key: np.zeros((1, 2, 2, 3), np.float32)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "image_mask": {
            "base_0_rgb": np.asarray([True]),
            "left_wrist_0_rgb": np.asarray([False]),
            "right_wrist_0_rgb": np.asarray([False]),
        },
        "state": np.zeros((1, 32), np.float32),
        "tokenized_prompt": np.asarray([[1, 2]], np.int32),
        "token_ar_mask": None,
    }
    values = {
        "raw/O72/head_left": np.zeros((2, 2, 3), np.uint8),
        "raw/O72/state": np.zeros(14, np.float32),
    }

    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(prefix + "/" + key, child)
        else:
            values[prefix] = None if value is None else value.copy()

    flatten("transformed/O72", tree)
    if mutation == "image":
        tree["image"]["left_wrist_0_rgb"][0, 0, 0, 0] = 1
    elif mutation == "state":
        tree["state"][0, 0] = 1
    elif mutation == "prompt":
        tree["tokenized_prompt"][0, 0] = 3
    elif mutation == "mask":
        tree["image_mask"]["left_wrist_0_rgb"][0] = True
    elif mutation == "missing_camera":
        del tree["image"]["right_wrist_0_rgb"]
    elif mutation == "optional":
        tree["token_ar_mask"] = np.asarray([False])

    def process(raw, action_dim, resize_size):
        assert set(raw) == {"head_left", "state"} and (action_dim, resize_size) == (
            14,
            224,
        )
        return tree

    def parity(record, name, left, right):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes() != right.tobytes()
        ):
            raise ValueError(name)

    agent = SimpleNamespace(actor=SimpleNamespace(process_raw_inputs=process))
    record = SimpleNamespace(tree=lambda name, value: None)
    bank = SimpleNamespace(require_parity=parity)
    if mutation is not None:
        with pytest.raises(ValueError):
            scorer.rebuild_transformed(agent, values, record, bank, "fresh")
    else:
        transformed, metadata = scorer.rebuild_transformed(
            agent, values, record, bank, "fresh"
        )
        assert transformed is tree
        assert metadata["transformed/O72/image/right_wrist_0_rgb"]["shape"] == [
            1,
            2,
            2,
            3,
        ]
