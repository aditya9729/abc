"""Score an accepted O72 bank with original fresh and restored C10 critics.

One original edit replay validates the fresh execution path. The fixed-bank
comparison never edits candidates, updates a learner, writes a checkpoint,
samples base actions, or operates a simulator. Numerical imports follow admission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import time
from pathlib import Path

CONFIG_SCHEMA = "sadhana.fixed_acquisition_bank_scoring_canonical/5"
INPUT_SCHEMA = "sadhana.fixed_acquisition_bank_scoring_inputs/5"
RESULT_SCHEMA = "sadhana.fixed_acquisition_bank_scoring_result/5"
PROFILE = "r1lite-fixed-acquisition-bank-scoring"
FIXED = {
    "train_seed": 21000,
    "episode_control": 72,
    "checkpoint_step": 10,
    "N": 64,
    "H": 16,
    "replan": 8,
    "delay": 0,
    "n_edit_samples": 0,
    "num_qs": 10,
    "num_min_qs": 2,
    "validation_N": 32,
    "validation_n_edit_samples": 32,
}
C10_COUNTERS = {
    "actor_train_state": 10,
    "edit_actor": 10,
    "temp": 10,
    "critic": 200,
    "filter_critic": 200,
    "batch_encoder": 200,
}
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticate(item):
    path = Path(item["path"])
    require(
        path.is_absolute() and path.is_file() and not path.is_symlink(),
        "Regular absolute input required",
    )
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require(
        (before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns),
        "Input changed while reading",
    )
    require(
        after.st_size == item["bytes"] and digest.hexdigest() == item["sha256"],
        "Input pin differs",
    )
    return path


def read_json(item):
    return json.loads(authenticate(item).read_text())


def load_helper(item):
    spec = importlib.util.spec_from_file_location(
        "fixed_bank_authenticated_helper", authenticate(item)
    )
    require(
        spec is not None and spec.loader is not None, "Missing pinned helper loader"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_fixed(config):
    require(
        config.get("schema") == INPUT_SCHEMA and config.get("fixed") == FIXED,
        "Only fixed O72/C10 bank scoring is supported",
    )
    for key, value in FIXED.items():
        require(
            type(config["fixed"][key]) is type(value), f"Wrong fixed field type: {key}"
        )


def accepted_inputs(config, bank, np):
    result = read_json(config["bank_result"])
    bank.validate_result(result)
    acceptance = read_json(config["bank_acceptance"])
    require(
        result["schema"] == bank.ACQUISITION_RESULT_SCHEMA
        and acceptance.get("status") == "accepted_original_acquisition_bank"
        and acceptance.get("artifact_accepted") is True
        and acceptance.get("bank_result") == config["bank_result"],
        "Require accepted original acquisition bank",
    )
    arrays = {}
    for name, item in result["arrays"].items():
        if item is None:
            arrays[name] = None
            continue
        value = np.load(authenticate(item), allow_pickle=False)
        require(
            list(value.shape) == item["shape"]
            and str(value.dtype) == item["dtype"]
            and np.isfinite(value).all(),
            f"Invalid bank array: {name}",
        )
        arrays[name] = value
    require(
        arrays["full/cand"].shape == (64, 8, 14)
        and arrays["full/cand"].dtype == np.float32
        and arrays["base_normalized_h16"].shape == (32, 16, 14)
        and arrays["base_normalized_h16"].dtype == np.float32
        and arrays["rng/selector_start"].shape == (2,)
        and arrays["rng/selector_start"].dtype == np.uint32,
        "Wrong fixed bank or selector RNG",
    )
    checkpoint = read_json(config["checkpoint_receipt"])
    accepted_checkpoint = read_json(config["checkpoint_acceptance"])
    require(
        accepted_checkpoint.get("status")
        == "original_fixed_data_training_and_final_checkpoint_verified"
        and accepted_checkpoint.get("condition")
        == "plus_autonomous_349_before_timeouts"
        and accepted_checkpoint.get("checkpoint_step") == 10
        and config["checkpoint_receipt"] in accepted_checkpoint.get("receipt_pins", []),
        "Require accepted corrected C10 checkpoint",
    )
    require(
        checkpoint.get("schema") == "sadhana.original_expo_final_checkpoint/1"
        and checkpoint.get("step") == 10
        and checkpoint.get("condition") == "plus_autonomous_349_before_timeouts"
        and checkpoint.get("original_full_saver") is True
        and checkpoint.get("wait_completed") is True
        and checkpoint.get("manager_closed") is True
        and checkpoint.get("directory")
        == accepted_checkpoint.get("checkpoint_directory"),
        "Wrong original C10 checkpoint",
    )
    root = Path(checkpoint["directory"])
    require(
        root.is_absolute()
        and root == root.resolve()
        and root.is_relative_to("/home/user/aditya"),
        "C10 checkpoint must remain under aditya",
    )
    require(
        len(checkpoint["files"])
        == accepted_checkpoint["full_checkpoint_files_verified"],
        "Incomplete C10 checkpoint receipt",
    )
    for item in checkpoint["files"]:
        require(
            Path(item["path"]).is_relative_to(root / "10"),
            "C10 file outside restored step",
        )
        authenticate(item)
    return result, arrays, checkpoint


def tensor_metadata(value):
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sharding": str(getattr(value, "sharding", None)),
        "committed": bool(getattr(value, "committed", False)),
    }


def rebuild_transformed(agent, arrays, record, bank, stage):
    """Use original preprocessing; require its complete recorded tensor tree."""
    raw = {
        "head_left": arrays["raw/O72/head_left"].copy(),
        "state": arrays["raw/O72/state"].copy(),
    }
    transformed = agent.actor.process_raw_inputs(raw, 14, 224)
    record.tree(f"{stage}/transformed_rebuilt", transformed)
    leaves = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(prefix + "/" + str(key), child)
        else:
            leaves[prefix] = value

    visit("transformed/O72", transformed)
    expected = {
        key: value
        for key, value in arrays.items()
        if key.startswith("transformed/O72/")
    }
    require(set(leaves) == set(expected), "Original transformed input tree differs")
    for key, value in leaves.items():
        if value is None or expected[key] is None:
            require(
                value is None and expected[key] is None,
                f"Optional transformed field differs: {key}",
            )
        else:
            bank.require_parity(
                record, f"{stage}/transformed_exact/{key}", value, expected[key]
            )
    return transformed, {key: tensor_metadata(value) for key, value in leaves.items()}


def original_selector_call(
    agent, arrays, transformed, api, jax, jnp, *, replay_original, metadata=None
):
    """Reuse the author selector for its exact replay or fixed-bank diagnostic."""
    require(
        (
            agent.action_dim,
            agent.state_dim,
            agent.replan_steps,
            agent.num_qs,
            agent.num_min_qs,
        )
        == (14, 32, 8, 10, 2)
        and agent.freeze_encoder is True,
        "Original critic dimensions differ",
    )
    if replay_original:
        precached = jnp.asarray(arrays["base_normalized_h16"])
    else:
        windows = jnp.asarray(arrays["full/cand"])
        precached = jnp.concatenate(
            [windows, jnp.zeros((64, 8, 14), dtype=windows.dtype)], axis=1
        )
    precached = jax.device_put(precached, agent.actor.infer_sharding)
    images, state = transformed["image"], transformed["state"]
    rng = jax.device_put(
        jnp.asarray(arrays["rng/selector_start"]), agent.actor.infer_sharding
    )
    if metadata is not None:
        metadata.update(
            precached=tensor_metadata(precached),
            rng=tensor_metadata(rng),
            state=tensor_metadata(state),
            images={key: tensor_metadata(value) for key, value in images.items()},
        )
    cache = agent._infer_cache
    require(cache is not None, "Fresh inference cache required")
    return api._jitted_fast_select(
        rng,
        precached,
        images,
        state,
        cache["batch_encoder_params"],
        cache["edit_actor_params"],
        cache["target_critic_params"],
        encoder_fn=agent.batch_encoder.apply_fn,
        edit_fn=agent.edit_actor.apply_fn,
        critic_fn=agent.target_critic.apply_fn,
        N=32 if replay_original else 64,
        n_edit_samples=32 if replay_original else 0,
        num_qs=10,
        num_min_qs=2,
        replan_steps=8,
        action_dim=14,
        state_dim=32,
        padded_action_dim=32,
        delay=0,
        edit_scale=0.1,
        edit_action_xyzg=False,
        freeze_encoder=True,
        only_base_actions=False,
        critic_camera_keys=agent.critic_camera_keys,
    )


def score_fixed_bank(agent, arrays, transformed, api, jax, jnp, metadata=None):
    """Score immutable candidates; no edit or base sampling call occurs here."""
    selected = original_selector_call(
        agent,
        arrays,
        transformed,
        api,
        jax,
        jnp,
        replay_original=False,
        metadata=metadata,
    )
    rng = jax.device_put(
        jnp.asarray(arrays["rng/selector_start"]), agent.actor.infer_sharding
    )
    key, _ = jax.random.split(rng)
    # The author's same subsampling utility selects IDs as well as model leaves.
    indices = api.subsample_image_ensemble(key, {"head_ids": jnp.arange(10)}, 2, 10)[
        "head_ids"
    ]
    return selected, indices


def score_order_evidence(current, reference, np):
    tested = [0, 44, 12, 14]
    old_ranks, new_ranks = np.empty(64, int), np.empty(64, int)
    old_ranks[np.argsort(reference, kind="stable")] = np.arange(64)
    new_ranks[np.argsort(current, kind="stable")] = np.arange(64)
    return {
        "reference_winner": int(reference.argmax()),
        "current_winner": int(current.argmax()),
        "reference_winner_gap": float(np.sort(reference)[-1] - np.sort(reference)[-2]),
        "current_winner_gap": float(np.sort(current)[-1] - np.sort(current)[-2]),
        "rank_positions_changed": int(np.count_nonzero(old_ranks != new_ranks)),
        "maximum_rank_displacement": int(np.abs(old_ranks - new_ranks).max()),
        "tested_candidate_order_preserved": bool(
            np.array_equal(
                np.argsort(current[tested], kind="stable"),
                np.argsort(reference[tested], kind="stable"),
            )
        ),
    }


def historical_comparisons(selected, arrays, record, np, label):
    """Report cross-process drift; numerical differences are not acceptance gates."""
    errors = {}
    for name, value in selected.items():
        comparison = record.compare(
            f"fresh/{label}/{name}", value, arrays[f"full/{name}"], exact=False
        )
        require(
            comparison["left_shape"] == comparison["right_shape"]
            and comparison["left_dtype"] == comparison["right_dtype"]
            and comparison["finite"],
            f"Malformed historical comparison tensor: {name}",
        )
        errors[name] = {
            key: comparison[key] for key in ("max_abs", "rms", "bitwise_equal")
        }
    qs, index = np.asarray(selected["qs"]), np.asarray(selected["idx"])
    require(
        qs.shape == (64,)
        and qs.dtype == np.float32
        and index.shape == ()
        and np.issubdtype(index.dtype, np.integer)
        and int(index) == int(qs.argmax()),
        "Historical diagnostic selector index differs from current argmax",
    )
    require(
        np.array_equal(
            np.asarray(selected["selected"]), np.asarray(selected["cand"])[int(index)]
        ),
        "Historical diagnostic selected command differs from current candidate",
    )
    return {
        "errors": errors,
        **score_order_evidence(qs, arrays["full/qs"], np),
        "scope": "Cross-process differences are diagnostic only. No numerical or rank equivalence is claimed. Only the accepted immutable bank is used for fresh/C10 scoring.",
    }


def describe_original_replay(selected, arrays, record, bank, np):
    for name in ("states_all", "rng"):
        bank.require_parity(
            record,
            f"fresh/original_replay/{name}",
            selected[name],
            arrays[f"full/{name}"],
        )
    bank.require_parity(
        record,
        "fresh/original_replay/base32",
        selected["cand"][:32],
        arrays["full/cand"][:32],
    )
    return historical_comparisons(selected, arrays, record, np, "original_replay")


def require_repeat_parity(selected, repeated, record, bank, stage):
    require(set(selected) == set(repeated), "Repeated selector output tree differs")
    for name in selected:
        bank.require_parity(
            record, f"{stage}/fixed64_repeat/{name}", selected[name], repeated[name]
        )


def describe_fixed_score_history(selected, arrays, record, bank, np):
    for name in ("cand", "q_all", "states_all"):
        bank.require_parity(
            record, f"fresh/fixed64/{name}", selected[name], arrays[f"full/{name}"]
        )
    # The no-edit scorer consumes fewer RNG splits than the original edited selector.
    comparable = {name: value for name, value in selected.items() if name != "rng"}
    return historical_comparisons(comparable, arrays, record, np, "fixed64")


def validate_selection(selected, arrays, np):
    for name, value in selected.items():
        require(
            np.isfinite(np.asarray(value)).all(), f"Nonfinite scoring output: {name}"
        )
    qs, index = np.asarray(selected["qs"]), np.asarray(selected["idx"])
    require(
        qs.shape == (64,)
        and qs.dtype == np.float32
        and index.shape == ()
        and np.issubdtype(index.dtype, np.integer),
        "Wrong score or index shape/dtype",
    )
    index = int(index)
    require(
        index == int(qs.argmax()) and 0 <= index < 64,
        "Selector index differs from Q argmax",
    )
    candidates = np.asarray(selected["cand"])
    require(
        candidates.shape == (64, 8, 14)
        and candidates.dtype == np.float32
        and candidates.tobytes() == arrays["full/cand"].tobytes(),
        "Fixed candidate bank changed",
    )
    require(
        np.array_equal(np.asarray(selected["selected"]), candidates[index]),
        "Selected command differs from indexed candidate",
    )
    for name, expected in (
        ("q_all", arrays["full/cand"].reshape(64, 112)),
        ("states_all", arrays["full/states_all"]),
    ):
        actual = np.asarray(selected[name])
        require(
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and actual.tobytes() == expected.tobytes(),
            f"Fixed scoring input differs: {name}",
        )
    padded = np.asarray(selected["executed_padded"])
    require(
        padded.shape == (8, 32)
        and padded.dtype == np.float32
        and np.array_equal(padded[:, :14], candidates[index])
        and np.count_nonzero(padded[:, 14:]) == 0,
        "Padded selected command differs",
    )


def validate_result(result):
    validate_fixed({"schema": INPUT_SCHEMA, "fixed": result.get("fixed")})
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "fixed_bank_scoring_complete"
        and result.get("artifact_accepted") is False
        and result.get("benchmark_accepted") is False
        and result.get("historical_comparisons_diagnostic_only") is True
        and result.get("fresh_parameter_identity_passed") is True
        and result.get("transformed_inputs_exact") is True,
        "Incomplete fixed-bank scoring result",
    )
    for key in (
        "optimizer_steps_performed",
        "native_controls",
        "candidate_generation_calls",
        "checkpoint_writes",
    ):
        require(
            type(result.get(key)) is int and result[key] == 0,
            f"Forbidden operation: {key}",
        )
    require(
        type(result.get("edit_sampling_calls")) is int
        and result["edit_sampling_calls"] == 1,
        "Require one validation-only original edit replay",
    )
    require(
        [x.get("stage") for x in result.get("stages", [])] == ["fresh", "c10"],
        "Both fixed scoring stages required",
    )
    for stage in result["stages"]:
        expected = (
            {key: 0 for key in C10_COUNTERS}
            if stage["stage"] == "fresh"
            else C10_COUNTERS
        )
        require(
            stage["counters_before"] == stage["counters_after"] == expected,
            "Wrong or changed update counters",
        )
        require(
            stage["parameters_before"] == stage["parameters_after"]
            and len(stage["parameters_before"]) == 8
            and stage["rng_unchanged"] is True,
            "Scoring changed model or RNG state",
        )
        require(
            type(stage["winner"]) is int and 0 <= stage["winner"] < 64,
            "Invalid fixed-bank winner",
        )
        require(
            type(stage.get("fixed_scoring_calls")) is int
            and stage["fixed_scoring_calls"] == 2
            and stage.get("fixed_scoring_repeat_bitwise_equal") is True,
            "Require two exactly repeated fixed scoring calls",
        )
    require(
        result["stages"][0]["head_ids"] == result["stages"][1]["head_ids"]
        and result["stages"][0]["head_ids"] == [6, 4],
        "Target-head selection changed",
    )


def run(config, output):
    validate_fixed(config)
    bank = load_helper(config["bank_helpers"])
    helper = bank.load_helper(config["identity_helpers"])
    pins = config["inputs"] + config["installed_sources"] + config["checkpoint_files"]
    for item in pins:
        authenticate(item)
    for name, version in config["runtime_versions"].items():
        require(
            importlib.metadata.version(name) == version,
            f"Runtime version differs: {name}",
        )
    for name, value in config["environment"].items():
        require(os.environ.get(name) == value, f"Declared environment differs: {name}")
    from sadhana_terminal_critic.contracts import require_canonical_lease

    require_canonical_lease(config["lease_guard"])
    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    from expo_ft.agents.alg import realtime_expo_ft as api
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.utils.train_utils import load_task_dataset
    from sadhana_r1lite.expo_config import make_task_config
    from sadhana_terminal_critic.diagnostic import (
        build_agent,
        counters,
        load_model_config,
    )

    require(
        len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu",
        "One leased GPU required",
    )
    accepted_bank, arrays, checkpoint = accepted_inputs(config, bank, np)
    online, settings = (
        read_json(config["evaluation_config"]),
        read_json(config["settings"]),
    )
    model_config = load_model_config(config, online)
    task = make_task_config(config["settings"]["path"])
    dataset = load_task_dataset(settings["cohort_pin"]["path"], task, num_data=21)
    buffer = create_replay_buffer(
        config=model_config,
        example_action=task.example_action,
        capacity=1802,
        task_description=task.language_instruction,
        replan_steps=8,
        seed=42,
        critic_camera_keys=tuple(task.critic_camera_keys),
    )
    buffer.insert_dataset(dataset[:1])
    agent = build_agent(
        model_config,
        task,
        buffer,
        "initialized",
        Path("/unused-no-restore"),
        jax,
        np,
        ocp,
    ).cache_infer_params()
    record = helper.Recorder(output, np, jax)

    def parameters(value):
        trees = {
            name: getattr(value, name).params
            for name in (
                "actor_train_state",
                "critic",
                "target_critic",
                "batch_encoder",
                "filter_critic",
                "edit_actor",
                "temp",
            )
        }
        trees["target_actor_params"] = value.target_actor_params
        return {
            name: {"sha256": digest["sha256"], "leaf_count": len(digest["leaves"])}
            for name, tree in trees.items()
            for digest in [helper.params_digest(tree, jax, np)]
        }

    stages = []
    progress = {
        "schema": "sadhana.fixed_bank_scoring_progress/5",
        "phase": "fresh",
        "stages": stages,
        "arrays": record.arrays,
        "comparisons": record.comparisons,
        "edit_sampling_calls": 0,
        "c10_restore_started": False,
    }

    def persist_progress():
        (output / "PROGRESS.json").write_text(
            json.dumps(progress, indent=2, allow_nan=False) + "\n"
        )

    try:
        for name in ("fresh", "c10"):
            progress["phase"] = name
            if name == "c10":
                progress["c10_restore_started"] = True
                persist_progress()
                with ocp.CheckpointManager(
                    Path(checkpoint["directory"]),
                    item_handlers={
                        "agent": ocp.PyTreeCheckpointHandler(),
                        "params": ocp.PyTreeCheckpointHandler(),
                    },
                    options=ocp.CheckpointManagerOptions(read_only=True, create=False),
                ) as manager:
                    agent = api.restore_checkpoint(
                        manager, agent.replace(_infer_cache=None), step=10
                    )
                agent = agent.cache_infer_params()
            before, before_parameters = counters(agent, jax, np), parameters(agent)
            stage = {
                "stage": name,
                "counters_before": before,
                "parameters_before": before_parameters,
            }
            stages.append(stage)
            persist_progress()
            expected = (
                {key: 0 for key in C10_COUNTERS} if name == "fresh" else C10_COUNTERS
            )
            require(before == expected, "Restored counters differ")
            rng_before = record.array(f"{name}/agent_rng_before", agent.rng)
            transformed, transformed_metadata = rebuild_transformed(
                agent, arrays, record, bank, name
            )
            stage.update(
                transformed_inputs_exact=True, transformed_metadata=transformed_metadata
            )
            persist_progress()
            if name == "fresh":
                require(
                    before_parameters == accepted_bank["parameters_before"],
                    "Fresh parameter identity differs from accepted bank",
                )
                progress["fresh_parameter_identity_passed"] = True
                progress["edit_sampling_calls"] = 1
                persist_progress()
                stage["original_replay_inputs"] = {}
                replay = original_selector_call(
                    agent,
                    arrays,
                    transformed,
                    api,
                    jax,
                    jnp,
                    replay_original=True,
                    metadata=stage["original_replay_inputs"],
                )
                for key, value in replay.items():
                    record.array(f"fresh/original_replay/{key}", value)
                progress["historical_replay_diagnostics"] = describe_original_replay(
                    replay, arrays, record, bank, np
                )
                progress["historical_comparisons_diagnostic_only"] = True
                persist_progress()
            stage["fixed64_inputs"] = {}
            selected, heads = score_fixed_bank(
                agent, arrays, transformed, api, jax, jnp, stage["fixed64_inputs"]
            )
            for key, value in selected.items():
                record.array(f"{name}/{key}", value)
            stage["fixed_scoring_calls"] = 1
            persist_progress()
            repeated, repeated_heads = score_fixed_bank(
                agent, arrays, transformed, api, jax, jnp
            )
            stage["fixed_scoring_calls"] = 2
            for key, value in repeated.items():
                record.array(f"{name}/fixed64_repeat/{key}", value)
            record.array(f"{name}/fixed64_repeat/head_ids", repeated_heads)
            require_repeat_parity(selected, repeated, record, bank, name)
            bank.require_parity(
                record, f"{name}/fixed64_repeat/head_ids", heads, repeated_heads
            )
            stage["fixed_scoring_repeat_bitwise_equal"] = True
            validate_selection(selected, arrays, np)
            head_ids = record.array(f"{name}/head_ids", heads).tolist()
            require(head_ids == [6, 4], "Original fixed-bank target-head IDs differ")
            if name == "fresh":
                progress["historical_fixed64_diagnostics"] = (
                    describe_fixed_score_history(selected, arrays, record, bank, np)
                )
            bank.require_parity(
                record,
                f"{name}/fixed_candidates",
                selected["cand"],
                arrays["full/cand"],
            )
            bank.require_parity(
                record, f"{name}/agent_rng_unchanged", rng_before, agent.rng
            )
            after, after_parameters = counters(agent, jax, np), parameters(agent)
            require(
                before == after and before_parameters == after_parameters,
                "Scoring changed parameters or counters",
            )
            scores = np.asarray(selected["qs"])
            stage.update(
                winner=int(np.asarray(selected["idx"])),
                head_ids=head_ids,
                counters_after=after,
                parameters_after=after_parameters,
                rng_unchanged=True,
                ranking_descending=np.argsort(-scores, kind="stable").tolist(),
                top_two_gap=float(np.sort(scores)[-1] - np.sort(scores)[-2]),
            )
            persist_progress()
        progress["phase"] = "complete"
    finally:
        persist_progress()
    for item in pins + checkpoint["files"]:
        authenticate(item)
    require(
        sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        < MAX_OUTPUT_BYTES - 2 * 1024 * 1024,
        "Scoring artifact cap exceeded",
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": "fixed_bank_scoring_complete",
        "fixed": FIXED,
        "artifact_accepted": False,
        "benchmark_accepted": False,
        "historical_comparisons_diagnostic_only": True,
        "historical_replay_diagnostics": progress["historical_replay_diagnostics"],
        "fresh_parameter_identity_passed": True,
        "transformed_inputs_exact": True,
        "historical_fixed64_diagnostics": progress["historical_fixed64_diagnostics"],
        "optimizer_steps_performed": 0,
        "native_controls": 0,
        "candidate_generation_calls": 0,
        "edit_sampling_calls": 1,
        "checkpoint_writes": 0,
        "stages": stages,
        "arrays": record.arrays,
        "comparisons": record.comparisons,
        "bank_result": config["bank_result"],
        "bank_acceptance": config["bank_acceptance"],
        "checkpoint_receipt": config["checkpoint_receipt"],
        "checkpoint_acceptance": config["checkpoint_acceptance"],
        "runtime": {"jax": jax.__version__, "devices": [str(x) for x in jax.devices()]},
        "scope": "Historical replay and scoring comparisons are descriptive only after cross-process numerical equivalence failed. Fresh/C10 each repeat the same fixed64 scoring graph with exact tensor stability, authenticated parameters and inputs, and an unchanged accepted candidate bank. No base proposals, learning, rollout or checkpoint writes; this is not the original online trained selector or full-rank equivalence.",
    }
    validate_result(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        hashlib.sha256(args.manifest.read_bytes()).hexdigest() == args.manifest_sha256,
        "Payload seal differs",
    )
    manifest = json.loads(args.manifest.read_text())
    for item in manifest["files"]:
        authenticate(item)
    require(
        manifest["caller"]["path"] == str(Path(__file__).resolve()),
        "Wrong caller location",
    )
    config = read_json(manifest["config"])
    bank = load_helper(config["bank_helpers"])
    helper = bank.load_helper(config["identity_helpers"])
    output = helper.fresh_output(args.output)
    output.mkdir(parents=True)
    started = time.monotonic()
    result = {"status": "failed", "artifact_accepted": False, "native_controls": 0}
    try:
        result = run(config, output)
        for item in manifest["files"]:
            authenticate(item)
        result["source_manifest"] = helper.pin(args.manifest)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["source_manifest"] = helper.pin(args.manifest)
        if (output / "PROGRESS.json").is_file():
            result["progress"] = helper.pin(output / "PROGRESS.json")
        data = json.dumps(result, indent=2, allow_nan=False).encode() + b"\n"
        require(len(data) <= 2 * 1024 * 1024, "Result receipt cap exceeded")
        with (output / "RESULT.json").open("xb") as stream:
            stream.write(data)


if __name__ == "__main__":
    main()
