"""Capture original EXPO proposals at the accepted TRAIN21000 control128 boundary.

Numerical imports occur only inside run(), after fixed input and lease checks.
No training, simulator operation, or policy continuation is implemented here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import time
from pathlib import Path

CONFIG_SCHEMA = "sadhana.failure_candidate_bank_canonical/1"
INPUT_SCHEMA = "sadhana.failure_candidate_bank_inputs/1"
RESULT_SCHEMA = "sadhana.failure_candidate_bank_result/1"
PROFILE = "r1lite-failure-candidate-bank"
ORIGIN = "author-subset-20261015T210006-cbb8d7"
FIXED = {
    "train_seed": 21000,
    "episode_control": 128,
    "initialization_seed": 42,
    "branch_random_seed": 42,
    "N": 32,
    "H": 16,
    "replan": 8,
    "boot_delay": 0,
    "continuation_delay": 5,
    "n_edit_samples": 32,
    "edit_scale": 0.1,
    "prefix_padded": None,
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
    digest = hashlib.sha256()
    before = path.stat()
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


def validate_fixed(config):
    require(
        config["schema"] == INPUT_SCHEMA and config["fixed"] == FIXED,
        "Only accepted O128 fresh boot is supported",
    )
    for key, expected in FIXED.items():
        require(
            type(config["fixed"][key]) is type(expected),
            f"Wrong fixed field type: {key}",
        )


def capture_observation(config, np):
    acceptance = read_json(config["capture_acceptance"])
    capture = read_json(config["capture_result"])
    require(
        acceptance["status"] == "accepted_reproducible_prefix_capture_only"
        and acceptance["artifact_accepted"] is True
        and acceptance["origin_run_id"] == ORIGIN
        and acceptance["seed"] == 21000
        and acceptance["controls_per_pass"] == 128,
        "Wrong accepted failure origin",
    )
    require(
        config["capture_result"] in acceptance["evidence"],
        "Capture result is not root accepted",
    )
    require(
        capture["status"] == "two_fresh_prefix_captures_equal"
        and capture["origin_run_id"] == ORIGIN
        and capture["seed"] == 21000,
        "Wrong capture result",
    )
    root = Path(acceptance["capture_directory"])
    for key, relative in (
        ("boundary_rgb", "rgb/0128.npy"),
        ("boundary_observation", "observations/0128.json"),
        ("boundary_integration", "integration/0128.npy"),
    ):
        item = config[key]
        require(
            item in capture["files"] and Path(item["path"]) == root / relative,
            f"Boundary file is not the accepted control128: {key}",
        )
        authenticate(item)
    observation = read_json(config["boundary_observation"])
    require(
        observation["index"] == 128
        and observation["time_s"] == capture["boundary"]["time_s"]
        and observation["judge"]["success"] is False
        and observation["judge"]["all_ok"] is False,
        "Wrong failure-state timing or judge",
    )
    image = np.load(config["boundary_rgb"]["path"], allow_pickle=False)
    state = np.asarray(observation["state14"], dtype=np.float32)
    public = observation["native_observation"]
    measured = np.concatenate(
        [
            public[key]
            for key in (
                "left_arm_q",
                "left_gripper_width",
                "right_arm_q",
                "right_gripper_width",
            )
        ]
    ).astype(np.float32)
    require(
        image.shape == (360, 640, 3)
        and image.dtype == np.uint8
        and state.shape == (14,)
        and np.isfinite(state).all()
        and state.tobytes() == measured.tobytes(),
        "Wrong raw image or measured state14",
    )
    return {"head_left": image.copy(), "state": state.copy()}


def physical_windows(actor, windows, jnp):
    """Exact original sample_actions._unnorm convention, including final cast."""
    padded = jnp.concatenate(
        [windows, jnp.zeros((windows.shape[0], 8, 14), dtype=windows.dtype)], axis=1
    )
    return jnp.asarray(actor.process_transformed_outputs(padded))[:, :8, :]


def branch_set(physical, full_index, no_edit_index, np):
    require(
        physical.shape == (64, 8, 14)
        and physical.dtype == np.float32
        and np.isfinite(physical).all(),
        "Require64 finite physical float32 windows",
    )
    require(
        type(full_index) is int
        and 0 <= full_index < 64
        and type(no_edit_index) is int
        and 0 <= no_edit_index < 32,
        "Invalid selector index",
    )
    random_index = random.Random(FIXED["branch_random_seed"]).randrange(64)
    choices = [
        ("base0", 0),
        ("full_winner", full_index),
        ("no_edit_winner", no_edit_index),
        ("seeded_uniform64", random_index),
    ]
    unique, aliases, by_bytes = [], [], {}
    for label, index in choices:
        data = np.ascontiguousarray(physical[index]).tobytes()
        if data not in by_bytes:
            by_bytes[data] = len(unique)
            unique.append(
                {
                    "branch": len(unique),
                    "candidate_index": index,
                    "physical_float32_sha256": hashlib.sha256(data).hexdigest(),
                    "aliases": [],
                }
            )
        branch = by_bytes[data]
        unique[branch]["aliases"].append(label)
        aliases.append({"label": label, "candidate_index": index, "branch": branch})
    return {
        "random_draw": {
            "method": "random.Random(42).randrange(64)",
            "seed": 42,
            "index": random_index,
            "policy_rng_consumed": False,
        },
        "aliases": aliases,
        "unique": unique,
    }


def require_parity(record, name, left, right):
    result = record.compare(name, left, right, exact=True)
    require(
        result["bitwise_equal"] and result["finite"],
        f"Original public parity failed: {name}",
    )


def validate_result(value):
    validate_fixed({"schema": INPUT_SCHEMA, "fixed": value.get("fixed")})
    require(
        value.get("schema") == RESULT_SCHEMA
        and value.get("status") == "failure_candidate_bank_complete"
        and value.get("origin_run_id") == ORIGIN
        and value.get("fixed") == FIXED
        and value.get("artifact_accepted") is False
        and value.get("benchmark_accepted") is False
        and type(value.get("optimizer_steps_performed")) is int
        and value["optimizer_steps_performed"] == 0
        and type(value.get("native_controls")) is int
        and value["native_controls"] == 0
        and value.get("parity_passed") is True
        and value.get("parameters_unchanged") is True
        and value.get("counters_before") == value.get("counters_after")
        and len(value.get("counters_before", {})) == 6
        and all(type(x) is int and x == 0 for x in value["counters_before"].values()),
        "Incomplete or mismatched inference-only bank result",
    )
    require(
        value.get("parameters_before") == value.get("parameters_after")
        and len(value.get("parameters_before", {})) == 8,
        "Parameter identity missing",
    )
    require(
        1 <= len(value.get("branches", {}).get("unique", [])) <= 4
        and [x["label"] for x in value["branches"]["aliases"]]
        == ["base0", "full_winner", "no_edit_winner", "seeded_uniform64"],
        "Wrong branch schedule",
    )


def load_helper(item):
    path = authenticate(item)
    spec = importlib.util.spec_from_file_location("failure_bank_identity_helpers", path)
    require(
        spec is not None and spec.loader is not None, "Missing pinned helper loader"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def capture_candidates(agent, raw, record, full_api, jax, jnp, np):
    """Original calls, with explicit dependencies for CPU boundary doubles."""
    record.array("rng/constructor", agent.rng)
    bank, selector_agent = agent.sample_pre_cache(
        copy.deepcopy(raw), prefix_padded=None
    )
    require(bank.shape == (32, 16, 14), "Wrong public pre-cache shape")
    bank_before = record.array("base_normalized_h16", bank).copy()
    record.array("rng/selector_start", selector_agent.rng)
    transformed = agent.actor.process_raw_inputs(copy.deepcopy(raw), 14, 224)
    record.tree("transformed/O128", transformed)
    cache = agent._infer_cache
    selections = {}
    for label, edits in (("no_edit", 0), ("full", 32)):
        selected = full_api._jitted_fast_select(
            selector_agent.rng,
            bank,
            transformed["image"],
            transformed["state"],
            cache["batch_encoder_params"],
            cache["edit_actor_params"],
            cache["target_critic_params"],
            encoder_fn=agent.batch_encoder.apply_fn,
            edit_fn=agent.edit_actor.apply_fn,
            critic_fn=agent.target_critic.apply_fn,
            N=32,
            n_edit_samples=edits,
            num_qs=agent.num_qs,
            num_min_qs=agent.num_min_qs,
            replan_steps=8,
            action_dim=14,
            state_dim=32,
            padded_action_dim=32,
            delay=0,
            edit_scale=0.1,
            edit_action_xyzg=False,
            freeze_encoder=agent.freeze_encoder,
            only_base_actions=False,
            critic_camera_keys=agent.critic_camera_keys,
        )
        for name in (
            "selected",
            "executed_padded",
            "qs",
            "idx",
            "cand",
            "rng",
            "enc_all",
            "q_all",
            "states_all",
        ):
            record.array(f"{label}/{name}", selected[name])
        selections[label] = selected
    full, no_edit = selections["full"], selections["no_edit"]
    require(
        full["cand"].shape == (64, 8, 14) and no_edit["cand"].shape == (32, 8, 14),
        "Wrong candidate cardinality",
    )
    require_parity(record, "shared_base_windows", full["cand"][:32], no_edit["cand"])
    physical = physical_windows(agent.actor, full["cand"], jnp)
    physical_np = record.array("physical64_float32", physical)
    record.array("edit_normalized_delta", full["cand"][32:] - full["cand"][:32])
    for label, only_base in (("full", False), ("base0", True)):
        command, returned, info = selector_agent.sample_actions(
            copy.deepcopy(raw), precached=bank, delay=0, only_base_actions=only_base
        )
        index = int(np.asarray(full["idx"])) if not only_base else 0
        selected_window = full["cand"][index]
        padded = jnp.pad(selected_window, ((0, 0), (0, 18)))
        # Use the public single-window batch shape for bitwise parity. Report bulk arithmetic separately.
        single = physical_windows(agent.actor, selected_window[None], jnp)[0]
        require_parity(record, f"{label}/public_physical", command, single)
        require_parity(
            record, f"{label}/public_prefix", info["executed_padded"], padded
        )
        expected_rng = jax.random.split(
            full["rng"] if not only_base else selector_agent.rng
        )[0]
        require_parity(record, f"{label}/public_rng", returned.rng, expected_rng)
        require_parity(record, f"{label}/bulk_physical", physical[index], command)
        record.array(f"{label}/public_outgoing_rng", returned.rng)
    require_parity(record, "base_bank_not_mutated", bank, bank_before)
    return physical_np, int(np.asarray(full["idx"])), int(np.asarray(no_edit["idx"]))


def run(config, output):
    validate_fixed(config)
    helper = load_helper(config["identity_helpers"])
    pins = config["inputs"] + config["installed_sources"] + config["checkpoint_files"]
    for expected in pins:
        authenticate(expected)
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
    from expo_ft.agents.alg import realtime_expo_ft as full_api
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
    raw = capture_observation(config, np)
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
    )
    agent = agent.cache_infer_params()
    require(
        (
            agent.N,
            agent.action_horizon,
            agent.replan_steps,
            agent.action_dim,
            agent.state_dim,
            agent.n_edit_samples,
            agent.edit_scale,
        )
        == (32, 16, 8, 14, 32, 32, 0.1)
        and agent.edit_action_xyzg is False
        and agent.actor.freeze_pi05_encoder is True,
        "Resolved original learner differs",
    )
    before = counters(agent, jax, np)
    require(all(x == 0 for x in before.values()), "Fresh initialized learner required")

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

    params_before = parameters(agent)
    record = helper.Recorder(output, np, jax)
    record.tree("raw/O128", raw)
    physical_np, full_index, no_edit_index = capture_candidates(
        agent, raw, record, full_api, jax, jnp, np
    )
    after = counters(agent, jax, np)
    params_after = parameters(agent)
    require(
        before == after and params_before == params_after,
        "Inference changed parameters or counters",
    )
    for expected in pins:
        authenticate(expected)
    require(
        sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        < MAX_OUTPUT_BYTES - 2 * 1024 * 1024,
        "Bank output exceeds bounded artifact cap",
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": "failure_candidate_bank_complete",
        "fixed": FIXED,
        "origin_run_id": ORIGIN,
        "artifact_accepted": False,
        "benchmark_accepted": False,
        "optimizer_steps_performed": 0,
        "native_controls": 0,
        "parity_passed": True,
        "parameters_unchanged": True,
        "parameters_before": params_before,
        "parameters_after": params_after,
        "counters_before": before,
        "counters_after": after,
        "arrays": record.arrays,
        "comparisons": record.comparisons,
        "capture_acceptance": config["capture_acceptance"],
        "capture_result": config["capture_result"],
        "branches": branch_set(physical_np, full_index, no_edit_index, np),
        "runtime": {"jax": jax.__version__, "devices": [str(x) for x in jax.devices()]},
        "scope": "One fresh initialized inference bank at a physically captured TRAIN failure. No outcome or trained-policy claim.",
        "continuation_proposal": {
            "implemented_here": False,
            "forced_controls": 8,
            "fresh_boot_at": 136,
            "remaining_controls": 764,
            "seed": 42,
            "only_base_actions": True,
        },
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
    helper = load_helper(config["identity_helpers"])
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
        data = json.dumps(result, indent=2, allow_nan=False).encode() + b"\n"
        require(len(data) <= 2 * 1024 * 1024, "Result receipt cap exceeded")
        with (output / "RESULT.json").open("xb") as stream:
            stream.write(data)


if __name__ == "__main__":
    main()
