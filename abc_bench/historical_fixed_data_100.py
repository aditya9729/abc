"""Bounded original EXPO updates on the corrected fixed TRAIN pool.

This is an explicit fixed-data schedule adaptation, not the author's online
collection loop. No training reward, sampler, loss or update is replaced.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from abc_bench import historical_fixed_data_learning as historical
from abc_bench.fixed_data_learning import require

MODULE = "abc_bench.historical_fixed_data_100"
PHASE = "historical_online_fixed_data_100"
STATUS = "original_historical_online_hundred_calls_checkpoint_completed"
CALLS = 100
MEASURE_AT = (0, 10, 30, 100)
SPEC_SHA256 = "6bd1eb6c41f42c35c6d21ca93872a5cdb9d4128db989f23b462e2b7a2de5d18d"

# These are the same task/data adapters used by the accepted ordered C10 run.
prepare_before_restore = historical.prepare_before_restore
prepare_replay = historical.prepare_replay


def load_spec(config):
    from abc_bench.realtime_update_probe import authenticate

    require(config.get("phase") == PHASE, "Unknown hundred-call phase")
    binding = config["fixed_data"]
    require(set(binding) == {"spec", "python", "condition"}, "Unexpected phase binding")
    require(
        binding["condition"] == historical.ORDERED_CONDITION,
        "Require the corrected C10 pool",
    )
    require(
        binding["spec"]["sha256"] == SPEC_SHA256, "Hundred-call specification differs"
    )
    spec = json.loads(authenticate(binding["spec"]).read_text())
    authenticate(binding["python"])
    require(
        spec["schema"] == "sadhana.historical_online_fixed_data_100_spec/1"
        and spec["phase"] == PHASE
        and spec["update_calls"] == CALLS
        and spec["measure_at"] == list(MEASURE_AT)
        and spec["final_checkpoint_step"] == CALLS
        and spec["assembly_condition"] == historical.ORDERED_CONDITION
        and spec["assembly_order"] == list(historical.ORDER)
        and spec["reward_view"] == historical.REWARD_VIEW,
        "Require original rewards, ordered pool, bounded100 calls and one final step100 save",
    )
    require(
        {
            MODULE,
            historical.MODULE,
            "abc_bench.fixed_data_learning",
            "expo_ft.agents.alg.agent",
        }
        <= set(config["installed_modules"]),
        "Install and pin the new phase, existing drivers and original checkpoint helper",
    )
    return spec


def load_published(item):
    """Load an authenticated, versioned published artifact, never a mutable checkout."""
    from abc_bench.realtime_update_probe import authenticate

    path = authenticate(item)
    spec = importlib.util.spec_from_file_location(
        "fixed_data_published_" + path.stem, path
    )
    require(
        spec is not None and spec.loader is not None, "Missing published helper loader"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_bank(spec):
    from abc_bench.realtime_update_probe import authenticate

    contract = spec["o72_diagnostic"]
    for item in contract.values():
        authenticate(item)
    result = json.loads(authenticate(contract["bank_result"]).read_text())
    acceptance = json.loads(authenticate(contract["bank_acceptance"]).read_text())
    bank = load_published(contract["bank_helpers"])
    bank.validate_result(result)
    require(
        result["schema"] == bank.ACQUISITION_RESULT_SCHEMA
        and acceptance.get("status") == "accepted_original_acquisition_bank"
        and acceptance.get("artifact_accepted") is True
        and acceptance.get("bank_result") == contract["bank_result"],
        "Require the accepted immutable O72 bank",
    )
    for item in result["arrays"].values():
        if item is not None:
            authenticate(item)
    return result


def authenticate_data(config):
    spec = load_spec(config)
    result = historical.authenticate_data(config, spec=spec)
    authenticate_bank(spec)
    return result


def added_terminal_batches(agent, buffer, jax, np):
    """Read isolated original replay windows without advancing the training iterator."""
    from expo_ft.agents.alg.batch_utils import prepare_critic_batch
    from sadhana_terminal_critic.contracts import (
        batch_fingerprint,
        check_terminal_target,
        fixed_window_view,
    )

    rows = []
    for local in range(341, 349):
        index = historical.DEMO_ROWS + local
        view = fixed_window_view(buffer, index)
        view.rng = jax.random.fold_in(jax.random.PRNGKey(2804), index)
        raw = view._convert_to_openpi_format(view.sample_jax(1))
        target = check_terminal_target(raw, 348 - local, np)
        prepared = prepare_critic_batch(
            raw, 32, 14, 32, 16, 8, agent.critic_camera_keys
        )
        rows.append(
            {
                "replay_index": index,
                "episode_step": local,
                "actual_controls_until_terminal": 349 - local,
                "includes_author_backfilled_tail": local > 341,
                "target": target,
                "fingerprint": batch_fingerprint(prepared, np),
                "prepared": prepared,
            }
        )
    return rows


def terminal_calibration(agent, batches, jax, np):
    from expo_ft.agents.alg.realtime_expo_ft import batch_encode

    rows = []
    for frozen in batches:
        prepared = frozen["prepared"]
        encoded = batch_encode(
            agent.batch_encoder.apply_fn,
            agent.batch_encoder.params,
            prepared["observations"],
            stop_gradient=True,
        )
        row = {key: value for key, value in frozen.items() if key != "prepared"}
        for name in ("critic", "target_critic"):
            critic = getattr(agent, name)
            q = np.asarray(
                jax.device_get(
                    critic.apply_fn(
                        {"params": critic.params},
                        encoded,
                        prepared["actions"],
                        False,
                        p=prepared["critic_states"],
                    )
                ),
                dtype=np.float64,
            )
            require(
                q.shape == (10, 1) and np.isfinite(q).all(), "Invalid added terminal Q"
            )
            row[name] = q[:, 0].tolist()
        rows.append(row)
    targets = np.asarray([row["target"]["reward"] for row in rows])[:, None]
    summary = {}
    for name in ("critic", "target_critic"):
        q = np.asarray([row[name] for row in rows])
        summary[name] = {
            "mse": float(((q - targets) ** 2).mean()),
            "bias": float((q - targets).mean()),
            "negative_fraction": float((q < 0).mean()),
        }
    return {
        "windows": rows,
        "calibration": summary,
        "scope": "Eight overlapping TRAIN windows from one historical terminal; seven include author backfilled tails",
    }


def diagnostic_hook(spec, processor, output):
    """Retain only frozen inputs; never capture an earlier learner or training RNG."""
    state = {}

    def measure(agent, call):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from expo_ft.agents.alg import realtime_expo_ft as api

        from abc_bench.realtime_update_probe import write_json

        destination = Path(output) / f"extra-diagnostic-{call:03d}"
        destination.mkdir()
        if not state:
            result = authenticate_bank(spec)
            contract = spec["o72_diagnostic"]
            arrays = {}
            for name, item in result["arrays"].items():
                value = (
                    None if item is None else np.load(item["path"], allow_pickle=False)
                )
                if item is not None:
                    require(
                        list(value.shape) == item["shape"]
                        and str(value.dtype) == item["dtype"]
                        and np.isfinite(value).all(),
                        "Invalid archived bank tensor",
                    )
                arrays[name] = value
            state.update(
                arrays=arrays,
                scorer=load_published(contract["scorer"]),
                bank=load_published(contract["bank_helpers"]),
                helper=load_published(contract["identity_helpers"]),
                terminals=added_terminal_batches(
                    agent, processor.replay_buffer, jax, np
                ),
            )
        scorer, bank, arrays = state["scorer"], state["bank"], state["arrays"]
        record = state["helper"].Recorder(destination, np, jax)
        result = {"call": call, "complete": False}
        try:
            before = model_digests(agent, state["helper"], jax, np)
            transformed, metadata = scorer.rebuild_transformed(
                agent, arrays, record, bank, f"call{call}"
            )
            # Author updates refresh this cache. Clearing only the local immutable
            # measurement copy also makes call0/current parameter selection explicit.
            current = agent.replace(_infer_cache=None).cache_infer_params()
            selected, heads = scorer.score_fixed_bank(
                current, arrays, transformed, api, jax, jnp
            )
            repeated, repeat_heads = scorer.score_fixed_bank(
                current, arrays, transformed, api, jax, jnp
            )
            record.tree("fixed64", selected)
            record.tree("fixed64_repeat", repeated)
            scorer.validate_selection(selected, arrays, np)
            scorer.require_repeat_parity(
                selected, repeated, record, bank, f"call{call}"
            )
            bank.require_parity(record, "heads_repeat", heads, repeat_heads)
            head_ids = record.array("head_ids", heads).tolist()
            require(head_ids == [6, 4], "Original fixed-bank head IDs differ")
            scores = np.asarray(jax.device_get(selected["qs"]))
            calibration = terminal_calibration(agent, state["terminals"], jax, np)
            after = model_digests(agent, state["helper"], jax, np)
            require(before == after, "Extra diagnostic changed model parameters")
            result.update(
                winner=int(np.asarray(jax.device_get(selected["idx"]))),
                head_ids=head_ids,
                scores=scores.tolist(),
                ranking_descending=np.argsort(-scores, kind="stable").tolist(),
                top_two_gap=float(np.sort(scores)[-1] - np.sort(scores)[-2]),
                fixed_scoring_repeat_bitwise_equal=True,
                fixed_scoring_calls=2,
                candidate_generation_calls=0,
                edit_sampling_calls=0,
                transformed_metadata=metadata,
                terminal_calibration=calibration,
                model_digests_before=before,
                model_digests_after=after,
                model_parameters_unchanged=True,
                complete=True,
                scope="Fixed TRAIN bank under current target critic/encoder; not online actor selection, a stop criterion, full-task or WBC performance",
            )
            return result
        finally:
            result.update(arrays=record.arrays, comparisons=record.comparisons)
            write_json(destination / "RESULT.json", result)

    return measure


def model_digests(agent, helper, jax, np):
    from abc_bench.realtime_update_probe import COUNTERS

    trees = {name: getattr(agent, name).params for name in (*COUNTERS, "target_critic")}
    trees["target_actor_params"] = agent.target_actor_params
    return {
        name: helper.params_digest(tree, jax, np)["sha256"]
        for name, tree in trees.items()
    }


def save_final_checkpoint(agent, config, output):
    return historical.save_final_checkpoint(agent, config, output, step=CALLS)


def run(*args):
    from abc_bench import historical_fixed_data_100  # noqa: PLW0406
    from abc_bench.realtime_update_probe import checked_import

    config, output = args[4:6]
    spec = load_spec(config)
    shared = checked_import("abc_bench.fixed_data_learning", config)
    return shared.run(
        *args,
        recovery=historical_fixed_data_100,
        checkpoint_writer=save_final_checkpoint,
        calls=CALLS,
        measure_at=MEASURE_AT,
        extra_measure=diagnostic_hook(spec, args[1], output),
    )
