"""Admitted recovery data for the original, bounded fixed-TRAIN diagnostic.

This named phase adds data, not a learner. It cannot create teacher acceptance.
The historical fixed_data_10 phase and its specification remain separate.
"""

from __future__ import annotations

import copy
import inspect
import json

from abc_bench.fixed_data_learning import CALLS, MEASURE_AT, require

MODULE = "abc_bench.recovery_fixed_data_learning"
ADAPTER = "sadhana_r1lite.recovery_expo_data"
CONNECTED_ADAPTER = "sadhana_r1lite.connected_recovery_expo_data"
ADMISSION_SCHEMAS = (
    "sadhana.recovery_expo_admission/1",
    "sadhana.recovery_expo_admission/2",
)
PHASE = "recovery_fixed_data_10"
STATUS = "original_recovery_fixed_data_ten_calls_completed"
SPEC_SHA256 = "4f99cb27dee929ef4a6708a561cba9181f5e5dfd6805628b610e556e70300f3d"
CONDITIONS = ("baseline_16968", "plus_accepted_correction")
ORIGINAL = "original_leela_judge3_first_terminal"
BASE_ROWS = 16968


def admission(config, spec):
    """Authenticate the root envelope; the installed adapter checks all trace rows."""
    from abc_bench.realtime_update_probe import authenticate

    pin = config["fixed_data"]["ingestion_admission"]
    require(isinstance(pin, dict), "A real pinned teacher admission is required")
    record = json.loads(authenticate(pin).read_text())
    require(
        record.get("schema") in ADMISSION_SCHEMAS
        and record.get("owner") == "/root"
        and record.get("status") == "accepted_complete_recovery_episode_for_ingestion"
        and record.get("artifact_accepted") is True
        and record.get("training_authorized") is False,
        "Root must first accept a complete teacher; ingestion is not training admission",
    )
    require(
        record.get("reward_view") == ORIGINAL, "Strict rewards cannot enter this phase"
    )
    require(
        record.get("normalization_manifest") == spec["normalization_manifest"],
        "Keep the original 21-demo normalization asset",
    )
    require(
        record.get("task_config") == spec["task_config"],
        "Correction must use the unchanged reference11 task",
    )
    return record


def load_spec(config):
    from abc_bench.realtime_update_probe import authenticate

    require(config.get("phase") == PHASE, "Unknown recovery fixed-data phase")
    binding = config["fixed_data"]
    require(
        set(binding) == {"spec", "python", "condition", "ingestion_admission"},
        "Unexpected recovery fixed-data binding",
    )
    require(binding["condition"] in CONDITIONS, "Unknown matched condition")
    require(binding["spec"]["sha256"] == SPEC_SHA256, "Recovery specification differs")
    spec = json.loads(authenticate(binding["spec"]).read_text())
    authenticate(binding["python"])
    require(
        spec["phase"] == PHASE
        and spec["update_calls"] == CALLS
        and spec["measure_at"] == list(MEASURE_AT),
        "Preserve the original ten-call measurement schedule",
    )
    require(
        {MODULE, "abc_bench.fixed_data_learning", ADAPTER, CONNECTED_ADAPTER}
        <= set(config["installed_modules"]),
        "Both fixed-data helpers and both adapters must be installed and pinned",
    )
    release = json.loads(authenticate(spec["adapter_release"]).read_text())
    require(
        release["status"] == "installed_CPU_fixtures_qualified_only"
        and release["installed_tests_passed"] == spec["adapter_cpu_cases"],
        "Bind the actual qualified ingestion release",
    )
    for name in (
        "source_manifest",
        "parent_review",
        "derivation_AST_equality",
        "wheel",
        "installed_identity",
        "test_result",
        "test_xml",
    ):
        authenticate(release[name])
    identity = json.loads(authenticate(release["installed_identity"]).read_text())
    require(
        identity["source_wheel_installed_match"] is True,
        "Require actual source/wheel/installed ingestion identity",
    )
    for module in (ADAPTER, CONNECTED_ADAPTER):
        suffix = "/" + module.replace(".", "/") + ".py"
        qualified = [row for row in identity["files"] if row["path"].endswith(suffix)]
        installed = config["installed_modules"][module]
        require(
            len(qualified) == 1
            and all(installed[k] == qualified[0][k] for k in ("bytes", "sha256")),
            "Installed ingestion payload differs from qualified release",
        )
        authenticate(installed)
    admission(config, spec)
    return spec


def authenticate_data(config):
    from abc_bench import fixed_data_learning as fixed
    from abc_bench.realtime_update_probe import authenticate, checked_import

    spec = load_spec(config)
    result = fixed.authenticate_data(config, spec=spec)
    record = admission(config, spec)
    if record["schema"] == ADMISSION_SCHEMAS[1]:
        connected = checked_import(CONNECTED_ADAPTER, config)
        pins = connected.connected_input_pins(
            config["fixed_data"]["ingestion_admission"]
        )
        for pin in pins:
            authenticate(pin)
    else:
        pins = []
        for name in (
            "prefix_acceptance",
            "prefix_result",
            "teacher_result",
            "normalization_manifest",
            "producer_source_manifest",
            "producer_release",
            "origin_policy_evidence",
            "task_config",
            "strict_contract_source",
        ):
            pin = record[name]
            authenticate(pin)
            pins.append(pin)
        for pin in record["qualification_evidence"]:
            authenticate(pin)
            pins.append(pin)
        for name in ("prefix_result", "teacher_result"):
            inventory = json.loads(authenticate(record[name]).read_text())["files"]
            require(bool(inventory), "Complete captured frame inventories required")
            for pin in inventory:
                authenticate(pin)
                pins.append(pin)
    result["recovery_inputs"] = {
        "admission": config["fixed_data"]["ingestion_admission"],
        "admission_schema": record["schema"],
        "input_files": pins,
        "authenticated_file_references": len(pins),
        "bytes_in_file_references": sum(pin["bytes"] for pin in pins),
        "scope": "Pinned capture and complete teacher inventories; no model or pickle load",
    }
    return result


def check_routing(processor):
    require(
        processor.batch_size == 64
        and processor.utd_ratio == 20
        and processor.offline_ratio == 0
        and processor.actor_success_only is True
        and processor.use_dagger_hil_sampling is False,
        "Require B64/UTD20, offline_ratio=0 and success-only non-HIL actor routing",
    )
    require(
        processor.offline_iterator is None
        and processor.hil_iterator is None
        and len(processor.offline_replay_buffer) == 0,
        "No offline or dagger-HIL sampler is permitted",
    )
    require(
        inspect.getgeneratorstate(processor.replay_iterator) == inspect.GEN_CREATED
        and not hasattr(processor.replay_buffer, "rng"),
        "Prepare all data before the first original sample",
    )


def source_pools(buffer, correction_rows, np):
    """Eligible start counts, not sampled indices or consumed source coverage."""
    end = len(buffer) - buffer._replan_steps
    groups = [
        ("historical_demo", 0, 7069),
        ("historical_online_timeout", 7069, BASE_ROWS),
    ]
    if correction_rows:
        groups += [
            ("correction_policy_prefix", BASE_ROWS, BASE_ROWS + 128),
            ("correction_teacher_suffix", BASE_ROWS + 128, len(buffer)),
        ]
    rows = []
    for source, start, stop in groups:
        sample_stop = max(start, min(stop, end))
        rows.append(
            {
                "source": source,
                "start": start,
                "stop_exclusive": stop,
                "stored_rows": stop - start,
                "critic_sampleable_starts": sample_stop - start,
                "actor_sampleable_starts": int(
                    np.count_nonzero(
                        buffer.dataset_dict["is_success"][start:sample_stop]
                    )
                ),
            }
        )
    return rows


def verify_normalized_actions(actor, buffer, correction, start, np):
    """Use the original actor transforms; compare stored current actions and state.

    Backfilled future chunks retain the original replay behavior. This check
    does not independently recompute normalization, padding or future chunks.
    """
    for index in range(len(correction)):
        transition = correction[index]
        obs = transition["observations"]
        require(
            set(obs) == {"state", "head_left", "prompt"},
            "No privileged lineage may enter actor observations",
        )
        physical = np.asarray(transition["actions"])
        require(
            physical.shape == (14,)
            and physical.dtype == np.float32
            and np.isfinite(physical).all(),
            "Require finite float32 physical14",
        )
        raw = copy.deepcopy(obs)
        raw["actions"] = np.tile(physical[None, :], (16, 1))
        transformed = actor.input_model_transforms(actor.input_transforms(raw))
        expected = np.asarray(
            transformed["actions"], dtype=buffer.dataset_dict["actions"].dtype
        )
        state = np.asarray(
            transformed["state"], dtype=buffer.dataset_dict["state"].dtype
        )
        require(
            expected.shape == (16, 32)
            and np.isfinite(expected).all()
            and np.isfinite(state).all(),
            "Original transformed tensors differ",
        )
        require(
            np.array_equal(
                expected[0], buffer.dataset_dict["actions"][start + index, 0]
            )
            and np.array_equal(state, buffer.dataset_dict["state"][start + index]),
            "Applied-action/state actor and replay normalization disagree",
        )
    return {
        "rows_checked": len(correction),
        "physical_action_dim": 14,
        "normalized_action_dim": 32,
        "original_actor_transforms_used": True,
        "compared": "current action and state only; original future backfill retained",
    }


def prepare_replay(agent, processor, config, spec):
    import numpy as np

    from abc_bench.realtime_update_probe import checked_import

    check_routing(processor)
    buffer = processor.replay_buffer
    require(len(buffer) == BASE_ROWS, "Restore the exact old replay before append")
    adapter = checked_import(ADAPTER, config)
    record = admission(config, spec)
    if record["schema"] == ADMISSION_SCHEMAS[1]:
        connected = checked_import(CONNECTED_ADAPTER, config)
        correction = connected.ConnectedRecoveryTransitions(
            config["fixed_data"]["ingestion_admission"]
        )
    else:
        correction = adapter.RecoveryTransitions(
            config["fixed_data"]["ingestion_admission"]
        )
    require(
        correction.admission == admission(config, spec), "Admission changed during load"
    )
    condition = config["fixed_data"]["condition"]
    require(condition in CONDITIONS, "Unknown matched condition")
    append_report, normalized = None, None
    if condition == "plus_accepted_correction":
        # Failure aborts the caller. Never sample a partially appended replay.
        append_report = adapter.append_after_restore(
            buffer, correction, expected_rows=BASE_ROWS
        )
        normalized = verify_normalized_actions(
            agent.actor, buffer, correction, BASE_ROWS, np
        )
    require(
        len(buffer) == BASE_ROWS + (len(correction) if append_report else 0),
        "Matched replay count differs",
    )
    check_routing(processor)
    pools = source_pools(buffer, len(correction) if append_report else 0, np)
    by_source = {row["source"]: row for row in pools}
    require(
        by_source["historical_demo"]["actor_sampleable_starts"] == 7069
        and by_source["historical_online_timeout"]["actor_sampleable_starts"] == 0,
        "Historical success eligibility changed",
    )
    if append_report:
        require(
            by_source["correction_policy_prefix"]["actor_sampleable_starts"] == 0,
            "Failed policy prefix became actor eligible",
        )
    return {
        "condition": condition,
        "admission": correction.admission_pin,
        "reward_view": ORIGINAL,
        "origin_group": correction.admission["origin_group"],
        "rows_before": BASE_ROWS,
        "rows_after": len(buffer),
        "append": append_report,
        "normalization_check": normalized,
        "source_pools": pools,
        "source_ranges_are_not_policy_inputs": True,
        "clipped_policy_prefix_controls": correction.clipped_prefix_controls,
        "first_historical_terminal_control": correction.first_historical_terminal,
        "complete_teacher_controls": correction.full_controls,
        "original_view_rows": len(correction),
        "strict_terminal_reward_view_executed": False,
        "actor_eligibility": "is_success; synthetic is_hil is not human provenance",
        "sample_indices_exposed_by_original_sampler": False,
        "consumed_source_coverage": None,
        "sampling_limit": "Report eligible pools and total budgets only; exact consumed source indices unavailable",
        "comparison_limit": "Pool sizes change sampling; same seeds do not give matched batch identities",
        "training_authorized": False,
        "task_success_claim": False,
    }


def run(*args):
    from abc_bench.realtime_update_probe import checked_import

    config = args[4]
    shared = checked_import("abc_bench.fixed_data_learning", config)
    # Fixed named module, never an arbitrary callback supplied by a config.
    from abc_bench import recovery_fixed_data_learning  # noqa: PLW0406

    return shared.run(*args, recovery=recovery_fixed_data_learning)
