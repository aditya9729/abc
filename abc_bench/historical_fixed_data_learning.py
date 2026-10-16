"""Original-reward autonomous TRAIN data and a full final EXPO checkpoint.

This fixed-data driver does not admit a strict correction or change the learner.
The installed original processor owns transforms, insertion and success marks.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import time
from pathlib import Path

from abc_bench.fixed_data_learning import CALLS, MEASURE_AT, require

MODULE = "abc_bench.historical_fixed_data_learning"
PHASE = "historical_online_fixed_data_10"
STATUS = "original_historical_online_ten_calls_checkpoint_completed"
SPEC_SHA256 = "0f108bb4ad2624877a32d84191438495548cd2516db77355eeb16a5d548a28a6"
CONDITIONS = ("baseline_16968", "plus_autonomous_349")
BASE_ROWS = 16968
ROWS = 349
PROMPT = "put the remote in the organizer"
REWARD_VIEW = "original_leela_judge3_first_terminal"


def load_spec(config):
    from abc_bench.realtime_update_probe import authenticate

    require(config.get("phase") == PHASE, "Unknown historical-data phase")
    binding = config["fixed_data"]
    require(set(binding) == {"spec", "python", "condition"}, "Unexpected phase binding")
    require(binding["condition"] in CONDITIONS, "Unknown matched condition")
    require(isinstance(binding["python"], dict), "Root must bind the installed numerical interpreter")
    require(binding["spec"]["sha256"] == SPEC_SHA256, "Historical specification differs")
    spec = json.loads(authenticate(binding["spec"]).read_text())
    authenticate(binding["python"])
    require(
        spec["phase"] == PHASE
        and spec["update_calls"] == CALLS
        and spec["measure_at"] == list(MEASURE_AT)
        and spec["final_checkpoint_step"] == CALLS
        and spec["reward_view"] == REWARD_VIEW,
        "Keep original rewards, ten calls and nonzero final checkpoint label",
    )
    require(
        {MODULE, "abc_bench.fixed_data_learning", "expo_ft.agents.alg.agent"}
        <= set(config["installed_modules"]),
        "Install and pin the phase, shared driver and original checkpoint helper",
    )
    return spec


def episode_inputs(spec):
    """Authenticate the fixed public capture; no array, pickle or model imports."""
    from abc_bench.realtime_update_probe import authenticate

    episode = spec["episode"]
    acceptance = json.loads(authenticate(episode["root_acceptance"]).read_text())
    result_path = authenticate(episode["result"])
    result = json.loads(result_path.read_text())
    require(
        acceptance["owner"] == "/root"
        and acceptance["status"] == "accepted_completed_normal_existing_judge_success_strict_unverified"
        and acceptance["historical_success"] is True
        and acceptance["strict_placement_verified"] is False
        and acceptance["teacher_data"] is False
        and acceptance["policy_run_verified"] is True,
        "Require accepted historical policy experience, never strict teacher admission",
    )
    require(
        result["schema"] == "sadhana.recovery_continuation_pilot/1"
        and result["status"] == "completed_candidate_only"
        and result["role"] == "normal"
        and result["seed"] == 21000
        and result["policy_seed"] == 42
        and result["replay_controls"] == 0
        and result["closed"] is True
        and result["terminal"] is True
        and result["error"] is None
        and result["teacher_data"] is False
        and result["last_unfinished_proposal"] is None
        and result["unfinished_step_may_have_advanced_physics"] is False
        and all(result[k] == ROWS for k in (
            "policy_controls", "native_controls", "recorded_through_control",
            "controls_with_successful_step_reply",
        )),
        "Require the complete autonomous 349-control TRAIN episode",
    )
    pins = result["files"]
    require(len(pins) == ROWS + 3, "Captured file count differs")
    paths = [Path(p["path"]) for p in pins]
    require(len(set(paths)) == len(paths), "Duplicate captured input")
    expected = [result_path.parent / "frames" / f"{i:04d}.npz" for i in range(ROWS + 1)]
    require(
        set(paths) == set(expected + [result_path.parent / name for name in ("INPUTS.json", "STARTUP.json")]),
        "Capture inventory differs from the declared causal frame series",
    )
    for pin in pins:
        require(not Path(pin["path"]).is_symlink(), "Captured input must be regular")
        authenticate(pin)
    by_path = {pin["path"]: pin for pin in pins}
    return [by_path[str(path)] for path in expected], pins


def authenticate_data(config):
    from abc_bench import fixed_data_learning as fixed

    spec = load_spec(config)
    result = fixed.authenticate_data(config, spec=spec)
    _, pins = episode_inputs(spec)
    result["recovery_inputs"] = {
        "kind": "historical_autonomous_policy_episode_not_strict_correction",
        "episode_result": spec["episode"]["result"],
        "input_files": pins,
        "authenticated_file_references": len(pins),
        "bytes_in_file_references": sum(pin["bytes"] for pin in pins),
    }
    return result


def validate_metadata(meta, index, previous, np):
    """Check a causal boundary and its original post-control reward record."""
    require(
        meta["episode_control"] == index
        and meta["role"] == "normal"
        and meta["seed"] == 21000
        and meta["is_hil"] is False
        and meta["teacher_data"] is False
        and meta["integration_spec"] == 16383
        and meta["public"]["index"] == index,
        "Frame identity or autonomous provenance differs",
    )
    state = np.asarray(meta["public"]["state14"], dtype=np.float64)
    require(state.shape == (14,) and np.isfinite(state).all(), "Invalid measured state14")
    require(
        np.array_equal(state, state.astype(np.float32).astype(np.float64)),
        "Captured policy state no longer represents original float32 observations",
    )
    require(
        math.isclose(meta["public"]["time_s"], index / 15, abs_tol=1e-9, rel_tol=0),
        "Control timestamp differs",
    )
    judge = meta["public"]["judge"]
    require(
        judge["judge_version"] == "leela.judge/3"
        and judge["success"] is (index == ROWS)
        and judge["failure"] is False,
        "Historical judge labels differ",
    )
    if index == 0:
        require(meta["kind"] == "policy_boundary" and previous is None, "Invalid first boundary")
        return False
    require(meta["kind"] == "policy_control" and meta["policy_control"] == index, "Invalid control row")
    require(meta["state_before"] == previous["public"]["state14"], "Pre-action state is not previous boundary")
    applied = np.asarray(meta["applied"], dtype=np.float64)
    proposed = np.asarray(meta["proposed"], dtype=np.float64)
    require(
        applied.shape == proposed.shape == (14,)
        and np.isfinite(applied).all() and np.isfinite(proposed).all(),
        "Invalid proposed/applied physical14 command",
    )
    info = meta["native_info"]
    terminal = index == ROWS
    require(
        meta["applied"] == info["action_applied"]
        and info["step"] == index
        and info["terminated"] is terminal
        and info["truncated"] is False
        and info["success"] is terminal
        and info["failure"] is False
        and meta["done_success_reward_mask"] == [terminal, terminal, float(terminal), float(not terminal)],
        "Applied command or original terminal/reward/bootstrap labels differ",
    )
    clipped = info["clipped"]
    mask = np.asarray(clipped["mask"], dtype=bool)
    require(
        mask.shape == (14,)
        and np.array_equal(mask, proposed != applied)
        and clipped["any"] is bool(mask.any()),
        "Clipping evidence differs; never train on proposed commands",
    )
    return bool(clipped["any"])


def read_frame(pin, np):
    from abc_bench.realtime_update_probe import authenticate

    with np.load(authenticate(pin), allow_pickle=False) as data:
        require(set(data.files) == {"rgb", "integration", "metadata_utf8"}, "Unknown captured arrays")
        rgb, state, encoded = data["rgb"], data["integration"], data["metadata_utf8"]
        require(rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8, "RGB format differs")
        require(state.shape == (437,) and state.dtype == np.float64 and np.isfinite(state).all(), "Invalid saved integration")
        require(encoded.ndim == 1 and encoded.dtype == np.uint8, "Invalid metadata encoding")
        meta = json.loads(encoded.tobytes())
        require(math.isclose(state[0], meta["public"]["time_s"], abs_tol=1e-12, rel_tol=0), "Integration time differs")
        return rgb, meta


def transition(previous, following, rgb, np):
    """Causal pre-action observation, applied command and post-action labels."""
    done, _, reward, mask = following["done_success_reward_mask"]
    return {
        "observations": {
            "state": np.asarray(previous["public"]["state14"], dtype=np.float32),
            "head_left": rgb,
            "prompt": PROMPT,
        },
        "actions": np.asarray(following["applied"], dtype=np.float64),
        "rewards": reward, "masks": mask, "dones": done,
        "is_hil": False, "is_success": False,
    }


class HistoricalEpisode:
    """Read-only causal metadata; read each RGB lazily, without privileged inputs."""

    def __init__(self, spec):
        import numpy as np

        self.pins, _ = episode_inputs(spec)
        self.metadata = []
        clipped = 0
        for index, pin in enumerate(self.pins):
            _, meta = read_frame(pin, np)
            clipped += validate_metadata(meta, index, self.metadata[-1] if index else None, np)
            self.metadata.append(meta)
        require(clipped == spec["episode"]["clipped_controls"] == 78, "Clipped control count differs")
        self.clipped_controls = clipped

    def __len__(self):
        return ROWS

    def __getitem__(self, index):
        import numpy as np

        if not isinstance(index, int) or not 0 <= index < ROWS:
            raise IndexError(index)
        rgb, meta = read_frame(self.pins[index], np)
        require(meta == self.metadata[index], "Causal metadata changed after validation")
        return transition(meta, self.metadata[index + 1], rgb, np)


def check_routing(processor):
    require(
        processor.batch_size == 64 and processor.utd_ratio == 20
        and processor.offline_ratio == 0 and processor.actor_success_only is True
        and processor.use_dagger_hil_sampling is False,
        "Require original B64/UTD20/offline0/success-only non-HIL routing",
    )
    require(
        processor.offline_iterator is None and processor.hil_iterator is None
        and len(processor.offline_replay_buffer) == 0
        and inspect.getgeneratorstate(processor.replay_iterator) == inspect.GEN_CREATED
        and not hasattr(processor.replay_buffer, "rng"),
        "Append only before the first original sample with no separate actor dataset",
    )


def append_episode(processor, episode, np):
    """Use the author's online API; any partial append error aborts the fresh run."""
    buffer = processor.replay_buffer
    require(
        len(buffer) == buffer._insert_index == BASE_ROWS
        and buffer._capacity > BASE_ROWS + ROWS
        and buffer._action_horizon == 16 and buffer._replan_steps == 8 and buffer._delay == 5
        and bool(buffer.dataset_dict["dones"][BASE_ROWS - 1])
        and buffer._ep_step_counter == 0 and not buffer._prev_is_hil,
        "Require exact restored replay, closed prior episode and no circular overwrite",
    )
    require(len(episode) == ROWS, "Wrong autonomous episode length")
    processor.on_episode_start()
    for index in range(ROWS):
        processor.insert_transition(episode[index])
    processor.on_episode_done(True)
    added = slice(BASE_ROWS, BASE_ROWS + ROWS)
    require(len(buffer) == buffer._insert_index == BASE_ROWS + ROWS, "Original insertion count differs")
    require(
        not np.any(buffer.dataset_dict["is_hil"][added])
        and not np.any(buffer.dataset_dict["hil_chunk"][added])
        and np.all(buffer.dataset_dict["is_success"][added]),
        "All autonomous rows must remain non-HIL and receive original episode success marks",
    )
    require(
        np.array_equal(buffer.dataset_dict["episode_step"][added], np.arange(ROWS))
        and buffer._ep_step_counter == 0 and not buffer._prev_is_hil,
        "Episode history must reset only at the recorded terminal",
    )
    terminal = np.arange(ROWS) == ROWS - 1
    require(
        np.array_equal(buffer.dataset_dict["dones"][added], terminal)
        and np.array_equal(buffer.dataset_dict["rewards"][added], terminal.astype(float))
        and np.array_equal(buffer.dataset_dict["masks"][added], (~terminal).astype(float)),
        "Original rewards, termination and bootstrap masks changed during insertion",
    )
    return {"rows": ROWS, "is_hil_rows": 0, "original_success_marked_rows": ROWS,
            "partial_append_policy": "abort worker and discard replay; never continue training"}


def verify_normalization(actor, buffer, episode, np):
    """Original actor transform must equal original replay's stored current row."""
    for index in range(ROWS):
        row = episode[index]
        raw = copy.deepcopy(row["observations"])
        raw["actions"] = np.tile(row["actions"][None, :], (16, 1))
        expected = actor.input_model_transforms(actor.input_transforms(raw))
        action = np.asarray(expected["actions"], dtype=buffer.dataset_dict["actions"].dtype)
        state = np.asarray(expected["state"], dtype=buffer.dataset_dict["state"].dtype)
        require(
            action.shape == (16, 32) and np.isfinite(action).all() and np.isfinite(state).all()
            and np.array_equal(action[0], buffer.dataset_dict["actions"][BASE_ROWS + index, 0])
            and np.array_equal(state, buffer.dataset_dict["state"][BASE_ROWS + index]),
            "Actor and replay current applied-action/state normalization disagree",
        )
    return {"rows_checked": ROWS, "raw_action_dtype": "float64", "normalized_action_dim": 32,
            "original_transforms_used": True, "future_chunk_backfill": "unchanged author insertion"}


def source_pools(buffer, np):
    """Original eligible starts; these are not observed sampling indices."""
    groups = [("historical_demo", 0, 7069), ("historical_timeout", 7069, BASE_ROWS)]
    if len(buffer) > BASE_ROWS:
        groups.append(("autonomous_historical_success", BASE_ROWS, len(buffer)))
    result = []
    for name, start, stop in groups:
        end = max(start, min(stop, len(buffer) - buffer._replan_steps))
        result.append({"source": name, "start": start, "stop_exclusive": stop,
                       "stored_rows": stop - start, "critic_sampleable_starts": end - start,
                       "actor_sampleable_starts": int(np.count_nonzero(buffer.dataset_dict["is_success"][start:end]))})
    return result


def prepare_replay(agent, processor, config, spec):
    import numpy as np

    check_routing(processor)
    buffer = processor.replay_buffer
    require(len(buffer) == BASE_ROWS, "Require exact original restored replay")
    episode = HistoricalEpisode(spec)
    appended, normalized = None, None
    if config["fixed_data"]["condition"] == CONDITIONS[1]:
        appended = append_episode(processor, episode, np)
        normalized = verify_normalization(agent.actor, buffer, episode, np)
    check_routing(processor)
    pools = source_pools(buffer, np)
    require(pools[0]["actor_sampleable_starts"] == 7069 and pools[1]["actor_sampleable_starts"] == 0, "Old success eligibility changed")
    require(len(buffer) == BASE_ROWS + (ROWS if appended else 0), "Condition size differs")
    if appended:
        require(pools[2]["actor_sampleable_starts"] == ROWS - 8, "New actor pool differs")
    return {
        "condition": config["fixed_data"]["condition"], "reward_view": REWARD_VIEW,
        "episode": spec["episode"], "rows_before": BASE_ROWS, "rows_after": len(buffer),
        "append": appended, "normalization_check": normalized, "source_pools": pools,
        "causal_frames": episode.pins, "causal_pair_rule": "row j uses frame j observation and frame j+1 applied action/reward; j=0..348",
        "teacher_data": False, "strict_correction_admitted": False,
        "strict_placement_verified": False, "strict_footprint_pass": False,
        "new_independent_origin": False, "sample_indices_available": False,
        "consumed_source_coverage": None,
        "sampling_limit": "Adding rows exposes eight previous tail starts; seeds do not match batch identities",
    }


def save_final_checkpoint(agent, config, output):
    """Original full saver, completed before the worker reports a finished phase."""
    import jax

    from abc_bench.realtime_update_probe import checked_import, file_pin, write_json

    base = checked_import("expo_ft.agents.alg.agent", config)
    learner = checked_import("expo_ft.agents.alg.realtime_expo_ft", config)
    destination = Path(output) / "learned-checkpoints"
    require(not destination.exists(), "Never overwrite an earlier checkpoint")
    started = time.monotonic()
    jax.block_until_ready(agent)
    manager, resuming = base.initialize_checkpoint_dir(destination, keep_period=None, overwrite=False, resume=False)
    try:
        require(not resuming, "Final snapshot is not a training resume")
        learner.save_checkpoint(manager, agent, CALLS)
        manager.wait_until_finished()
        require(tuple(manager.all_steps()) == (CALLS,), "Original save did not complete at nonzero step10")
        require(all((destination / str(CALLS) / name).is_dir() for name in ("agent", "params")), "Missing full original agent/params items")
    finally:
        manager.close()
    paths = sorted(destination.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Unexpected checkpoint symlink")
    files = [file_pin(path) for path in paths if path.is_file()]
    require(bool(files), "Empty completed checkpoint")
    result = {
        "schema": "sadhana.original_expo_final_checkpoint/1", "directory": str(destination),
        "step": CALLS, "condition": config["fixed_data"]["condition"],
        "original_full_saver": True, "wait_completed": True, "manager_closed": True,
        "files": files, "bytes": sum(p["bytes"] for p in files),
        "elapsed_seconds": time.monotonic() - started,
        "separate_process_restore_verified": False, "task_performance_measured": False,
        "scope": "Full author agent and params for frozen evaluation; replay, sampler and environment restart state not saved",
    }
    write_json(Path(output) / "checkpoint.json", result)
    return result


def run(*args):
    from abc_bench.realtime_update_probe import checked_import

    shared = checked_import("abc_bench.fixed_data_learning", args[4])
    from abc_bench import historical_fixed_data_learning  # noqa: PLW0406

    return shared.run(*args, recovery=historical_fixed_data_learning, checkpoint_writer=save_final_checkpoint)
