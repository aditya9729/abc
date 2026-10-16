"""Fixed TRAIN fit/rank diagnostic; called only by the existing capacity worker."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PHASE = "fixed_data_10"
SPEC_SHA256 = "9e42561f6fe20549fe126017bef0ee142552839ce22c324a72317323d52c45df"
MEASURE_AT = (0, 1, 10)
CALLS = 10
STATUS = "original_fixed_data_ten_calls_completed"
MODULE = "abc_bench.fixed_data_learning"


def require(value, message):
    if not value:
        raise ValueError(message)


def load_spec(config):
    from abc_bench.realtime_update_probe import authenticate

    require(config.get("phase") == PHASE, "Unknown fixed-data phase")
    binding = config["fixed_data"]
    require(set(binding) == {"spec", "python"}, "Unexpected fixed-data binding")
    require(
        binding["spec"]["sha256"] == SPEC_SHA256, "Fixed-data specification differs"
    )
    spec = json.loads(authenticate(binding["spec"]).read_text())
    authenticate(binding["python"])
    require(
        spec["phase"] == PHASE and spec["update_calls"] == CALLS,
        "Only the bounded ten-call phase is admitted",
    )
    require(spec["measure_at"] == list(MEASURE_AT), "Measurement schedule differs")
    require(
        MODULE in config["installed_modules"],
        "Fixed-data helper must be installed and pinned",
    )
    return spec


def numeric_inventory(checkpoint_parent, expected, cutoff=9908, count=9899):
    """Match the original restore selection before loading any pickle."""
    root = Path(checkpoint_parent) / "buffers"
    require(root.is_dir() and not root.is_symlink(), "Missing real replay directory")
    actual = [
        p for p in root.glob("*.pkl") if p.stem.isdigit() and int(p.stem) <= cutoff
    ]
    require(
        len(actual) == count == len(expected), "Replay numeric inventory count differs"
    )
    require(
        all(not p.is_symlink() and p.is_file() for p in actual),
        "Replay file is not regular",
    )
    labels = [int(p.stem) for p in actual]
    require(len(set(labels)) == count, "Duplicate numeric replay label")
    expected_paths = [Path(p["path"]) for p in expected]
    require(
        len(set(expected_paths)) == count and set(actual) == set(expected_paths),
        "Replay numeric filename set differs from accepted inventory",
    )
    require(
        all(p.parent == root for p in expected_paths),
        "Replay pin escapes original parent",
    )
    return sorted(actual, key=lambda p: int(p.stem))


def authenticate_data(config, *, spec=None):
    from abc_bench.realtime_update_probe import authenticate

    spec = load_spec(config) if spec is None else spec
    for name, version in spec["runtime_versions"].items():
        require(
            importlib.metadata.version(name) == version,
            f"Original numerical runtime differs: {name}",
        )
    for expected in spec["inputs"]:
        authenticate(expected)
    accepted = json.loads(authenticate(spec["replay_acceptance"]).read_text())
    report = json.loads(authenticate(spec["replay_inspection"]).read_text())
    require(
        accepted["owner"] == "/root" and accepted["artifact_accepted"] is True,
        "Accepted original replay provenance required",
    )
    require(
        accepted["host_inspection"] == spec["replay_inspection"],
        "Replay acceptance link differs",
    )
    require(
        report["checkpoint"] == str(Path(spec["checkpoint_parent"]) / "9909"),
        "Original replay parent differs",
    )
    require(
        report["replay"]["transitions"] == 9899
        and report["replay"]["physical_action_and_pre_action_state_equal"] is True
        and report["replay"]["reward_done_mask_equal"] is True,
        "Accepted replay semantics differ",
    )
    files = report["replay"]["files"]
    numeric_inventory(spec["checkpoint_parent"], files)
    for expected in files:
        authenticate(expected)
    return {
        "spec": spec,
        "files": files,
        "replay_bytes": sum(p["bytes"] for p in files),
    }


def accounting(consumed_calls, returned_batches):
    require(
        type(consumed_calls) is int and 0 <= consumed_calls <= CALLS,
        "Invalid consumed calls",
    )
    require(
        type(returned_batches) is int and returned_batches == consumed_calls + 1,
        "The exact next batch must remain pending",
    )
    return {
        "consumed_critic_rows": consumed_calls * 1280,
        "consumed_actor_rows": consumed_calls * 64,
        "generated_critic_batches": returned_batches + 2,
        "generated_critic_rows": (returned_batches + 2) * 1280,
        "generated_actor_batches": returned_batches,
        "generated_actor_rows": returned_batches * 64,
        "unconsumed_returned_critic_batches": 1,
        "unconsumed_internal_critic_batches": 2,
        "unconsumed_actor_batches": 1,
    }


def retained_queue(iterator):
    """Read the paused original generator's queue; never advance or replace it."""
    require(
        iterator.gi_frame is not None, "Original replay iterator unexpectedly ended"
    )
    queue = iterator.gi_frame.f_locals.get("queue")
    require(
        isinstance(queue, deque) and len(queue) == 2,
        "Original three-batch queue differs",
    )
    return tuple(queue)


def isolated_measure(agent, measure, snapshot, pending, iterator):
    before = snapshot(agent, pending)
    retained = () if iterator is None else retained_queue(iterator)
    result = measure(agent)
    require(
        before == snapshot(agent, pending),
        "Measurement changed agent/replay/combine RNG or counters",
    )
    require(
        iterator is None
        or all(a is b for a, b in zip(retained, retained_queue(iterator), strict=True)),
        "Measurement replaced an original pending critic batch",
    )
    return {
        "rng_and_counters_before_after": before,
        "unchanged": True,
        "result": result,
    }


def update_sequence(
    agent_holder, processor, combine_rng, update, measure, snapshot, labels, emit
):
    """Original next_batch/one-ahead/update routing, with explicit quiescent measurements."""
    agent = (
        agent_holder.pop()
    )  # Transfer ownership; callers must not retain old states.
    measurements = {
        0: isolated_measure(
            agent,
            lambda a: measure(a, 0),
            lambda a, p: snapshot(a, combine_rng, p),
            None,
            None,
        )
    }
    emit("measurement-00.json", measurements[0])
    consumed = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(processor.next_batch, combine_rng)
        for call in range(1, CALLS + 1):
            batch, actor_batch, combine_rng = pending.result()
            pending = pool.submit(processor.next_batch, combine_rng)
            consumed.append(
                {
                    "call": call,
                    "critic": labels(batch, 1280),
                    "actor": labels(actor_batch, 64),
                }
            )
            agent, detail = update(agent, batch, actor_batch, call)
            emit(f"update-{call:02d}.json", {**consumed[-1], **detail})
            if call in MEASURE_AT:
                exact_pending = (
                    pending.result()
                )  # Quiesce; retain this Future/result for next call.
                measurements[call] = isolated_measure(
                    agent,
                    lambda a, call=call: measure(a, call),
                    lambda a, p, combine_rng=combine_rng: snapshot(a, combine_rng, p),
                    exact_pending,
                    processor.replay_iterator,
                )
                measurements[call]["sampling"] = accounting(call, call + 1)
                emit(f"measurement-{call:02d}.json", measurements[call])
                del exact_pending  # The original Future retains the exact next batch.
        exact_pending = pending.result()
        internal = retained_queue(processor.replay_iterator)
        unconsumed = {
            "returned_critic": labels(exact_pending[0], 1280),
            "returned_actor": labels(exact_pending[1], 64),
            "internal_critics": [labels(x, 1280) for x in internal],
        }
    return agent, measurements, consumed, unconsumed


def rank_record(values, reference_index=31):
    require(
        len(values) == 32 and all(math.isfinite(x) for x in values),
        "Invalid fixed-bank Q values",
    )
    best = max(
        range(32), key=lambda i: values[i]
    )  # Original argmax: lowest index on ties.
    reference = values[reference_index]
    rank = 1 + sum(
        x > reference or (x == reference and i < reference_index)
        for i, x in enumerate(values)
        if i != reference_index
    )
    return {
        "argmax": best,
        "reference_rank_with_original_tie_rule": rank,
        "reference_top1": best == reference_index,
        "reference_margin": reference
        - max(x for i, x in enumerate(values) if i != reference_index),
        "q": values,
        "tie_rule": "first maximum / lowest candidate index",
    }


def run(
    agent_holder,
    processor,
    dataset,
    task,
    config,
    output,
    optimizer_state,
    parameters_state,
    construction_seconds,
    *,
    recovery=None,
    checkpoint_writer=None,
):
    """No model implementation here: original builders, sampler, update and Q helpers only."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from expo_ft.agents.alg.batch_utils import prepare_critic_batch
    from expo_ft.agents.alg.realtime_expo_ft import _jitted_fast_select, batch_encode
    from expo_ft.agents.vla.pi05 import _jitted_sample_with_prefix
    from sadhana_terminal_critic.contracts import (
        batch_fingerprint,
        check_terminal_target,
        fixed_window_view,
        terminal_starts,
    )

    from abc_bench import realtime_update_probe as probe

    agent = agent_holder.pop()
    started = time.monotonic()
    spec = load_spec(config) if recovery is None else recovery.load_spec(config)
    buffer = processor.replay_buffer
    require(
        len(buffer) == 7069 and len(processor.offline_replay_buffer) == 0,
        "Original demos must be seeded exactly once before restore",
    )
    restore_start = time.monotonic()
    processor.restore(spec["checkpoint_parent"], up_to_step=9908)
    require(len(buffer) == 16968, "Original restore did not append exactly9899 rows")
    restore_seconds = time.monotonic() - restore_start
    print(
        json.dumps(
            {
                "phase": "restored_fixed_replay",
                "rows": len(buffer),
                "seconds": restore_seconds,
            }
        ),
        flush=True,
    )
    require(
        inspect.getgeneratorstate(processor.replay_iterator) == inspect.GEN_CREATED,
        "Training iterator advanced before original replay restore",
    )
    require(not hasattr(buffer, "rng"), "Training sampler RNG already advanced")
    # Normalization is loaded by the original actor and replay pipelines from the pinned asset.
    for field in ("state", "actions"):
        for attr in ("mean", "std", "q01", "q99"):
            require(
                np.array_equal(
                    getattr(agent.actor.data_config.norm_stats[field], attr),
                    getattr(buffer._data_config.norm_stats[field], attr),
                ),
                "Actor/replay normalization differs",
            )
    require(
        agent.actor.data_config.use_quantile_norm
        == buffer._data_config.use_quantile_norm,
        "Actor/replay normalization mode differs",
    )

    recovery_data = None
    if recovery is not None:
        recovery_data = recovery.prepare_replay(agent, processor, config, spec)
        probe.write_json(output / "recovery-data.json", recovery_data)

    arrays = []

    def save_array(name, value):
        value = np.asarray(jax.device_get(value))
        require(np.isfinite(value).all(), f"Nonfinite diagnostic tensor: {name}")
        path = output / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        entry = {
            **probe.file_pin(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        arrays.append(entry)
        return entry

    def array_digest(value):
        a = np.ascontiguousarray(np.asarray(jax.device_get(value)))
        return hashlib.sha256(a.tobytes()).hexdigest()

    banks, terminal_batches = [], []
    offset = 0
    freeze_start = time.monotonic()
    freeze_rng = array_digest(agent.rng)
    for ordinal, episode in enumerate(dataset.episodes):
        t = episode.steps - 8
        require(t >= 5, "Logged window lacks the preceding five actions")
        slow = agent.actor.process_raw_inputs(
            copy.deepcopy(dataset[offset + t - 5]["observations"]), 14, 224
        )
        current = agent.actor.process_raw_inputs(
            copy.deepcopy(dataset[offset + t]["observations"]), 14, 224
        )
        # Original input transforms normalize physical applied commands, then pad to32.
        physical = episode.arrays["actions"][t - 5 : t + 8]
        physical_h16 = np.concatenate([physical, np.repeat(physical[-1:], 3, axis=0)])
        raw = copy.deepcopy(dataset[offset + t - 5]["observations"])
        raw["actions"] = physical_h16.copy()
        normalized = agent.actor.input_model_transforms(
            agent.actor.input_transforms(raw)
        )["actions"]
        normalized = np.asarray(normalized, dtype=buffer.dataset_dict["actions"].dtype)
        require(
            normalized.shape == (16, 32) and np.isfinite(normalized).all(),
            "Normalized logged shape differs",
        )
        prefix, logged = normalized[:5].copy(), normalized[5:13, :14].copy()
        require(
            np.array_equal(prefix, buffer.dataset_dict["actions"][offset + t - 5, :5]),
            "Physical logged prefix and original replay normalization differ",
        )
        require(
            np.array_equal(logged, buffer.dataset_dict["actions"][offset + t, :8, :14]),
            "Physical logged execution window and replay normalization differ",
        )
        noise_key = jax.random.fold_in(jax.random.PRNGKey(2701), ordinal)
        select_key = jax.random.fold_in(jax.random.PRNGKey(2702), ordinal)
        helper_key = jax.random.fold_in(jax.random.PRNGKey(2703), ordinal)
        noise = jax.random.normal(noise_key, (1, 32, 16, 32))
        padded = _jitted_sample_with_prefix(
            slow,
            agent.actor_train_state,
            helper_key,
            jnp.asarray(prefix)[None],
            32,
            noise=noise,
        )
        bank = agent.actor._unpad_actions(padded)
        require(bank.shape == (32, 16, 14), "Original bank shape differs")
        augmented = bank.at[31, 5:13].set(jnp.asarray(logged))
        frozen = {
            "bank": bank,
            "augmented": augmented,
            "current": current,
            "select_key": select_key,
            "seed": episode.seed,
            "t": t,
        }
        frozen["bank_digest"] = array_digest(bank)
        frozen["augmented_digest"] = array_digest(augmented)
        banks.append(frozen)
        for name, value in (
            ("noise", noise),
            ("noise-key", noise_key),
            ("selection-key", select_key),
            ("helper-key", helper_key),
            ("prefix", prefix),
            ("bank", bank),
            ("augmented-bank", augmented),
            ("logged-physical", physical[5:13]),
            ("logged-normalized", logged),
            ("current-state", current["state"]),
            ("current-image", current["image"]["base_0_rgb"]),
            ("slow-state", slow["state"]),
            ("slow-image", slow["image"]["base_0_rgb"]),
        ):
            save_array(f"seed-{episode.seed}-{name}", value)
        for local in terminal_starts(episode.steps):
            view = fixed_window_view(buffer, offset + local)
            view.rng = jax.random.fold_in(jax.random.PRNGKey(2704), offset + local)
            raw_batch = view._convert_to_openpi_format(view.sample_jax(1))
            target = check_terminal_target(raw_batch, episode.steps - 1 - local, np)
            prepared = prepare_critic_batch(
                raw_batch, 32, 14, 32, 16, 8, agent.critic_camera_keys
            )
            terminal_batches.append(
                {
                    "seed": episode.seed,
                    "step": local,
                    "target": target,
                    "fingerprint": batch_fingerprint(prepared, np),
                    "observations": prepared["observations"],
                    "states": prepared["critic_states"],
                    "actions": prepared["actions"],
                }
            )

        offset += episode.steps
        print(
            json.dumps(
                {"phase": "frozen_bank", "episode": ordinal + 1, "seed": episode.seed}
            ),
            flush=True,
        )
    require(
        len(banks) == 21 and len(terminal_batches) == 168,
        "Fixed measurement pool differs",
    )
    require(
        not hasattr(buffer, "rng") and array_digest(agent.rng) == freeze_rng,
        "Frozen-input preparation advanced training RNG",
    )
    freeze_seconds = time.monotonic() - freeze_start
    probe.write_json(
        output / "frozen-inputs.json",
        {
            "arrays": arrays,
            "seconds": freeze_seconds,
            "terminal_batches": [
                {
                    k: v
                    for k, v in x.items()
                    if k not in ("observations", "states", "actions")
                }
                for x in terminal_batches
            ],
        },
    )

    def counter_steps(a):
        return {
            name: int(np.asarray(jax.device_get(getattr(a, name).step)))
            for name in probe.COUNTERS
        }

    def snapshot(a, combine, pending):
        def key(value):
            return None if value is None else np.asarray(jax.device_get(value)).tolist()

        return {
            "agent_rng": key(a.rng),
            "replay_rng": key(getattr(buffer, "rng", None)),
            "combine_rng": key(combine),
            "pending_combine_rng": key(None if pending is None else pending[2]),
            "optimizer_steps": counter_steps(a),
        }

    def labels(batch, rows):
        require(
            batch is not None and batch["actions"].shape[0] == rows,
            "Original training batch shape differs",
        )
        values = {
            k: np.asarray(jax.device_get(batch[k]))
            for k in ("rewards", "masks", "dones", "valids")
        }
        require(
            all(x.shape == (rows,) and np.isfinite(x).all() for x in values.values()),
            "Invalid sampled labels",
        )
        return {
            "rows": rows,
            "positive_return": int((values["rewards"] > 0).sum()),
            "terminal_mask_zero": int((values["masks"] == 0).sum()),
            "done": int(values["dones"].sum()),
            "critic_weight_zero": int((values["valids"] == 0).sum()),
            "label_sha256": {k: array_digest(x) for k, x in values.items()},
        }

    def select(a, bank, frozen):
        return _jitted_fast_select(
            frozen["select_key"],
            bank,
            frozen["current"]["image"],
            frozen["current"]["state"],
            a.batch_encoder.params,
            a.edit_actor.params,
            a.target_critic.params,
            encoder_fn=a.batch_encoder.apply_fn,
            edit_fn=a.edit_actor.apply_fn,
            critic_fn=a.target_critic.apply_fn,
            N=32,
            n_edit_samples=0,
            num_qs=a.num_qs,
            num_min_qs=a.num_min_qs,
            replan_steps=8,
            action_dim=14,
            state_dim=32,
            padded_action_dim=32,
            delay=5,
            edit_scale=0.1,
            edit_action_xyzg=False,
            freeze_encoder=a.freeze_encoder,
            only_base_actions=False,
            critic_camera_keys=a.critic_camera_keys,
        )

    def measure(a, call):
        a = (
            a.replace()
        )  # Immutable state copy; original helpers receive explicit fixed keys.
        start = time.monotonic()
        rows, rank_rows = [], []
        for frozen in banks:
            pair = jax.random.choice(
                jax.random.split(frozen["select_key"])[0],
                jnp.arange(a.num_qs),
                (a.num_min_qs,),
                replace=False,
            )
            rank = {
                "seed": frozen["seed"],
                "current_step": frozen["t"],
                "fixed_target_heads": np.asarray(pair).tolist(),
            }
            for kind in ("bank", "augmented"):
                require(
                    array_digest(frozen[kind]) == frozen[kind + "_digest"],
                    "Frozen bank changed",
                )
                selection = select(a, frozen[kind], frozen)
                q = np.asarray(jax.device_get(selection["qs"]), dtype=np.float64)
                record = rank_record(q.tolist())
                if kind == "bank":
                    record = {
                        key.replace("reference_", "slot31_"): value
                        for key, value in record.items()
                    }
                    record["slot31_role"] = "unlabelled_generated_candidate"
                else:
                    record["slot31_role"] = (
                        "logged_successful_execution_chunk_diagnostic_insertion"
                    )
                rank[kind] = record
                require(
                    int(np.asarray(selection["idx"])) == rank[kind]["argmax"],
                    "Original argmax differs",
                )
                save_array(f"call-{call:02d}-seed-{frozen['seed']}-{kind}-q", q)
            rank_rows.append(rank)
        pair = jax.random.choice(
            jax.random.PRNGKey(2705),
            jnp.arange(a.num_qs),
            (a.num_min_qs,),
            replace=False,
        )
        pair_host = np.asarray(pair).tolist()
        for frozen in terminal_batches:
            encoded = batch_encode(
                a.batch_encoder.apply_fn,
                a.batch_encoder.params,
                frozen["observations"],
                stop_gradient=True,
            )
            row = {
                "seed": frozen["seed"],
                "step": frozen["step"],
                "target": frozen["target"]["reward"],
            }
            for name in ("critic", "target_critic"):
                critic = getattr(a, name)
                q = np.asarray(
                    jax.device_get(
                        critic.apply_fn(
                            {"params": critic.params},
                            encoded,
                            frozen["actions"],
                            False,
                            p=frozen["states"],
                        )
                    ),
                    dtype=np.float64,
                )
                require(
                    q.shape == (10, 1) and np.isfinite(q).all(),
                    "Invalid original terminal Q",
                )
                row[name] = q[:, 0].tolist()
            row["fixed_two_target_min"] = min(
                row["target_critic"][i] for i in pair_host
            )
            rows.append(row)
        summaries = {}
        for name in ("critic", "target_critic"):
            q = np.asarray([x[name] for x in rows])
            target = np.asarray([x["target"] for x in rows])[:, None]
            summaries[name] = {
                "mse": float(((q - target) ** 2).mean()),
                "bias": float((q - target).mean()),
                "mean_prediction_mse": float(
                    ((q.mean(axis=1) - target[:, 0]) ** 2).mean()
                ),
                "head_variance": float(q.var(axis=1).mean()),
                "negative_fraction": float((q < 0).mean()),
                "mse_by_head": ((q - target) ** 2).mean(axis=0).tolist(),
            }
        qmin = np.asarray([x["fixed_two_target_min"] for x in rows])
        target = np.asarray([x["target"] for x in rows])
        summaries["fixed_two_target_min"] = {
            "heads": pair_host,
            "mse": float(((qmin - target) ** 2).mean()),
            "bias": float((qmin - target).mean()),
            "negative_fraction": float((qmin < 0).mean()),
        }
        optimizers = optimizer_state(a)
        parameters = parameters_state(a)
        return {
            "call": call,
            "terminal_windows": rows,
            "calibration": summaries,
            "ranking": rank_rows,
            "optimizers": optimizers,
            "parameters": parameters,
            "seconds": time.monotonic() - start,
        }

    update_seconds = []

    def update(a, batch, actor_batch, call):
        a = a.replace(rng=jax.device_put(a.rng, a.actor.replicated_sharding))
        start = time.monotonic()
        a, info = a.update(a, batch, 20, actor_batch)
        jax.block_until_ready((a, info))
        elapsed = time.monotonic() - start
        metrics = {k: float(np.asarray(jax.device_get(v))) for k, v in info.items()}
        require(
            metrics and all(math.isfinite(x) for x in metrics.values()),
            "Missing/nonfinite returned update metrics",
        )
        require(
            metrics.get("next_q_nan_ratio") == 0,
            "Final critic minibatch returned nonfinite target Q",
        )
        require(
            counter_steps(a) == {k: call * v for k, v in probe.COUNTERS.items()},
            "Original step counts differ",
        )
        update_seconds.append(elapsed)
        print(
            json.dumps(
                {"phase": "original_update_returned", "call": call, "seconds": elapsed}
            ),
            flush=True,
        )
        return a, {
            "update_seconds": elapsed,
            "returned_metrics_last_critic_minibatch_only": metrics,
        }

    agent_holder.append(agent)
    del agent
    agent, measurements, consumed, unconsumed = update_sequence(
        agent_holder,
        processor,
        jax.random.PRNGKey(142),
        update,
        measure,
        snapshot,
        labels,
        lambda name, value: probe.write_json(output / name, value),
    )
    after = optimizer_state(agent)
    for name, multiplier in probe.COUNTERS.items():
        require(
            after[name]["step"] == CALLS * multiplier
            and after[name]["counts"]
            and all(x == CALLS * multiplier for x in after[name]["counts"].values()),
            "Final optimizer counters differ",
        )
    final_parameters = parameters_state(agent)
    checkpoint = (
        None
        if checkpoint_writer is None
        else checkpoint_writer(agent, config, output)
    )
    immutability_started = time.monotonic()
    authenticated = (
        authenticate_data(config)
        if recovery is None
        else recovery.authenticate_data(config)
    )
    probe.authenticate_inputs(config)
    result = {
        "status": STATUS,
        "phase": PHASE,
        "artifact_accepted": False,
        "benchmark_accepted": False,
        "online_schedule_executed": False,
        "native_controls": 0,
        "update_calls": CALLS,
        "batch_size": 64,
        "utd_ratio": 20,
        "before": measurements[0]["result"]["optimizers"],
        "after": after,
        "parameters_after": final_parameters,
        "measurements": measurements,
        "consumed_batches": consumed,
        "unconsumed_batches": unconsumed,
        "sampling": accounting(10, 11),
        "arrays": arrays,
        "update_seconds": sum(update_seconds),
        "update_seconds_each": update_seconds,
        "construction_seconds": construction_seconds,
        "restore_seconds": restore_seconds,
        "freeze_seconds": freeze_seconds,
        "phase_seconds": time.monotonic() - started,
        "final_immutability_seconds": time.monotonic() - immutability_started,
        "replay_files": len(authenticated["files"]),
        "replay_bytes": authenticated["replay_bytes"],
        "scope": "Fresh original learner; fixed TRAIN fit/rank only; no collection, resume, recovery or task-success claim",
        "limits": [
            "21 successes share one source lineage; timeout rows are not negative-action labels.",
            "Logged action insertion is a diagnostic bank augmentation; other candidates are unlabelled.",
            "168 terminal windows include original backfilled tails; ranking windows are fully executed.",
            "Original update metrics expose only the last of20 critic minibatches.",
            "Frozen banks do not measure changes in the updated actor proposal distribution.",
            "Timing includes compilation and measurements; no automatic extension.",
        ],
    }
    if recovery is not None:
        result.update(
            status=recovery.STATUS,
            phase=recovery.PHASE,
            recovery_data=recovery_data,
            recovery_inputs_after=authenticated["recovery_inputs"],
        )
    if checkpoint is not None:
        result["checkpoint"] = checkpoint
    probe.write_json(output / "update.json", result)
    return result
