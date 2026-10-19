"""Original fixed-data updates and full save; expensive diagnostics run separately."""

from __future__ import annotations

import json
from pathlib import Path

from abc_bench import historical_fixed_data_learning as historical
from abc_bench.fixed_data_learning import require

MODULE = "abc_bench.historical_fixed_data_100_training"
PHASE = "historical_online_fixed_data_100_training"
STATUS = "original_historical_online_hundred_calls_training_checkpoint_completed"
CALLS = 100
MEASURE_AT = ()
SPEC_SHA256 = "8dfc35c3e58a8a521a63c8c37b3277efe503614133d47102214a14dc712a6031"

prepare_before_restore = historical.prepare_before_restore
prepare_replay = historical.prepare_replay


def load_spec(config):
    from abc_bench.realtime_update_probe import authenticate

    require(config.get("phase") == PHASE, "Unknown training-only phase")
    binding = config["fixed_data"]
    require(set(binding) == {"spec", "python", "condition"}, "Unexpected phase binding")
    require(binding["condition"] == historical.ORDERED_CONDITION, "Require corrected C10 pool")
    require(binding["spec"]["sha256"] == SPEC_SHA256, "Training-only specification differs")
    spec = json.loads(authenticate(binding["spec"]).read_text())
    authenticate(binding["python"])
    require(
        spec["schema"] == "sadhana.historical_online_fixed_data_100_training_spec/1"
        and spec["phase"] == PHASE
        and spec["update_calls"] == CALLS
        and spec["measure_at"] == []
        and spec["final_checkpoint_step"] == CALLS
        and spec["diagnostics_in_training"] is False
        and spec["checkpoint_before_final_integrity"] is True
        and (spec["worker_seconds"], spec["parent_seconds"], spec["outer_seconds"]) == (9000, 9100, 9200)
        and spec["assembly_condition"] == historical.ORDERED_CONDITION
        and spec["assembly_order"] == list(historical.ORDER)
        and spec["reward_view"] == historical.REWARD_VIEW,
        "Require bounded100 original updates, unchanged data and save before separate diagnostics",
    )
    require(
        {MODULE, historical.MODULE, "abc_bench.fixed_data_learning", "expo_ft.agents.alg.agent"}
        <= set(config["installed_modules"]),
        "Install and pin phase, shared driver and original checkpoint helper",
    )
    return spec


def authenticate_data(config):
    return historical.authenticate_data(config, spec=load_spec(config))


def initial_integrity(output):
    """Read the single full inventory already produced by the original capacity driver."""
    from abc_bench.realtime_update_probe import COUNTERS

    before = json.loads((Path(output) / "before.json").read_text())
    require(set(before) == {"optimizers", "parameters"}, "Missing initial integrity inventory")
    require(set(before["optimizers"]) == set(COUNTERS), "Initial optimizer inventory differs")
    require(set(before["parameters"]) == set(COUNTERS) | {"target_critic", "target_actor_params"},
            "Initial parameter inventory differs")
    for row in before["optimizers"].values():
        require(row["step"] == 0 and row["counts"] and all(x == 0 for x in row["counts"].values()),
                "Require fresh original optimizer counters")
    for row in (*before["optimizers"].values(), *before["parameters"].values()):
        require(row["finite_leaves"] > 0 and row["finite_bytes"] > 0, "Missing finite initial inventory")
    return before


def save_final_checkpoint(agent, config, output):
    return historical.save_final_checkpoint(agent, config, output, step=CALLS)


def run(*args):
    from abc_bench import historical_fixed_data_100_training  # noqa: PLW0406
    from abc_bench.realtime_update_probe import checked_import

    config, output = args[4:6]
    load_spec(config)
    before = initial_integrity(output)
    shared = checked_import("abc_bench.fixed_data_learning", config)
    return shared.run(
        *args,
        recovery=historical_fixed_data_100_training,
        checkpoint_writer=save_final_checkpoint,
        calls=CALLS,
        measure_at=MEASURE_AT,
        collect_diagnostics=False,
        initial_integrity=before,
    )
