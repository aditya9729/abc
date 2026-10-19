"""Installed CPU contracts; explicit doubles do not qualify numerical learning."""

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import historical_fixed_data_learning as historical
from abc_bench import realtime_update_probe as probe

PREPARATION = Path("/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/historical-online-fixed-data-preparation-20261016")


def local_artifact(path):
    """Skip only unavailable local evidence; corrupt evidence must still fail."""
    path = Path(path)
    try:
        path.stat()
    except FileNotFoundError:
        pytest.skip(f"Local historical EXPO artifact is unavailable: {path}")
    return path


def original_methods(relative_path, class_name, names):
    """Transplant pinned author AST only; avoid importing JAX/native packages."""
    spec = json.loads(local_artifact(PREPARATION / "PHASE_SPEC.json").read_text())
    pin = next(p for p in spec["inputs"] if p["path"].endswith("/" + relative_path))
    source = local_artifact(pin["path"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == pin["sha256"]
    module = ast.parse(source)
    nodes = module.body if class_name is None else next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [copy.deepcopy(n) for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(tree), pin["path"], "exec"), scope)
    return scope


def processor_double():
    """Actual author insertion/backfill/marking over explicit CPU storage doubles."""
    source = "expo_ft/data/replay_buffer.py"
    helpers = original_methods(source, None, {"_insert_recursively"})
    methods = original_methods(source, "PiReplayBuffer", {"insert", "mark_episode_success"})
    methods["insert"].__globals__["_insert_recursively"] = helpers["_insert_recursively"]

    class Buffer:
        insert = methods["insert"]
        mark_episode_success = methods["mark_episode_success"]

        def __init__(self):
            self._capacity = historical.BASE_ROWS + historical.ROWS + 1
            self._size = self._insert_index = historical.BASE_ROWS
            self._action_horizon, self._replan_steps, self._delay = 16, 8, 5
            self._ep_step_counter, self._prev_is_hil = 0, False
            n = self._capacity
            self.dataset_dict = {k: np.zeros(n, dtype=bool) for k in ("dones", "is_hil", "hil_chunk", "is_success")}
            self.dataset_dict.update(actions=np.zeros((n, 16, 32), np.float32), state=np.zeros((n, 32), np.float32),
                                     rewards=np.zeros(n, np.float32), masks=np.ones(n, np.float32), episode_step=np.zeros(n, np.int32))
            self.dataset_dict["dones"][historical.BASE_ROWS - 1] = True
            self.dataset_dict["is_success"][:7069] = True

        def __len__(self):
            return self._size

        def _preprocess_single_transition(self, row):
            # Deliberately small storage transform double. Original numerical
            # transforms must additionally pass the real runtime check.
            result = {k: row[k] for k in ("rewards", "masks", "dones", "is_hil", "hil_chunk", "is_success", "episode_step")}
            result["actions"] = np.pad(row["actions"].astype(np.float32), ((0, 0), (0, 18)))
            result["state"] = np.pad(row["state"].astype(np.float32), (0, 18))
            return result

    methods = original_methods("expo_ft/data/batch_processor.py", "BatchProcessor", {"insert_transition", "on_episode_start", "on_episode_done"})

    class Processor:
        insert_transition = methods["insert_transition"]
        on_episode_start = methods["on_episode_start"]
        on_episode_done = methods["on_episode_done"]

    processor = Processor()
    processor.replay_buffer = Buffer()
    processor._ep_buffer_start = historical.BASE_ROWS
    return processor


class EpisodeDouble:
    def __len__(self):
        return historical.ROWS

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        done = index == len(self) - 1
        return {"observations": {"state": np.full(14, index, np.float32), "head_left": np.zeros((1, 1, 3), np.uint8), "prompt": historical.PROMPT},
                "actions": np.full(14, index + 0.123456789, np.float64), "rewards": float(done),
                "masks": float(not done), "dones": done, "is_hil": False, "is_success": False}


def test_actual_saved_episode_causal_applied_and_terminal_contract():
    spec = json.loads(local_artifact(PREPARATION / "PHASE_SPEC.json").read_text())
    local_artifact(spec["episode"]["root_acceptance"]["path"])
    result = json.loads(local_artifact(spec["episode"]["result"]["path"]).read_text())
    for pin in result["files"]:
        local_artifact(pin["path"])
    episode = historical.HistoricalEpisode(spec)
    assert len(episode) == 349 and episode.clipped_controls == 78
    for index in (0, 173, 348):
        row = episode[index]
        before, after = episode.metadata[index:index + 2]
        np.testing.assert_array_equal(row["actions"], after["applied"])
        np.testing.assert_array_equal(row["observations"]["state"], before["public"]["state14"])
        assert set(row["observations"]) == {"state", "head_left", "prompt"}
        assert row["actions"].dtype == np.float64
        assert row["is_hil"] is False and row["is_success"] is False
        assert row["dones"] is (index == 348)
    clipped = next(i for i, m in enumerate(episode.metadata) if i and m["native_info"]["clipped"]["any"])
    assert not np.array_equal(episode[clipped - 1]["actions"], episode.metadata[clipped]["proposed"])


@pytest.mark.parametrize("change", ["state_before", "terminal", "human", "clipping"])
def test_metadata_rejects_noncausal_or_mislabeled_rows(change):
    before = {"public": {"state14": [0.0] * 14}}
    meta = {"episode_control": 1, "policy_control": 1, "role": "normal", "seed": 21000,
            "is_hil": False, "teacher_data": False, "integration_spec": 16383, "kind": "policy_control",
            "public": {"index": 1, "state14": [0.0] * 14, "time_s": 1 / 15,
                       "judge": {"judge_version": "leela.judge/3", "success": False, "failure": False}},
            "state_before": [0.0] * 14, "applied": [0.0] * 14, "proposed": [0.0] * 14,
            "done_success_reward_mask": [False, False, 0.0, 1.0],
            "native_info": {"action_applied": [0.0] * 14, "step": 1, "terminated": False,
                            "truncated": False, "success": False, "failure": False,
                            "clipped": {"mask": [False] * 14, "any": False}}}
    if change == "state_before":
        meta["state_before"][0] = 1.0
    elif change == "terminal":
        meta["done_success_reward_mask"] = [True, True, 1.0, 0.0]
    elif change == "human":
        meta["is_hil"] = True
    else:
        meta["proposed"][0] = 1.0
    with pytest.raises(ValueError):
        historical.validate_metadata(meta, 1, before, np)


def test_original_insert_marks_episode_and_preserves_boundary_backfill():
    processor = processor_double()
    buffer = processor.replay_buffer
    old_tail = buffer.dataset_dict["actions"][historical.BASE_ROWS - 1].copy()
    report = historical.append_episode(processor, EpisodeDouble(), np)
    assert report["original_success_marked_rows"] == 349
    np.testing.assert_array_equal(buffer.dataset_dict["actions"][historical.BASE_ROWS - 1], old_tail)
    # Author backfills future action chunks inside the new episode only.
    np.testing.assert_array_equal(buffer.dataset_dict["actions"][historical.BASE_ROWS, 1, :14], EpisodeDouble()[1]["actions"].astype(np.float32))
    assert buffer.dataset_dict["rewards"][historical.BASE_ROWS:].sum() == 1
    assert buffer.dataset_dict["masks"][historical.BASE_ROWS + 348] == 0
    pools = historical.source_pools(buffer, np)
    assert [r["critic_sampleable_starts"] for r in pools] == [7069, 9899, 341]
    assert [r["actor_sampleable_starts"] for r in pools] == [7069, 0, 341]


def test_partial_append_error_never_marks_success_or_continues():
    processor = processor_double()
    insert = processor.insert_transition
    calls = []

    def fail_on_second(row):
        calls.append(row)
        if len(calls) == 2:
            raise RuntimeError("explicit storage refusal")
        insert(row)

    processor.insert_transition = fail_on_second
    with pytest.raises(RuntimeError, match="storage refusal"):
        historical.append_episode(processor, EpisodeDouble(), np)
    assert len(processor.replay_buffer) == historical.BASE_ROWS + 1
    assert not processor.replay_buffer.dataset_dict["is_success"][historical.BASE_ROWS]


@pytest.mark.parametrize("wrong", ["prior_done", "horizon"])
def test_append_refuses_wrong_original_boundary(wrong):
    processor = processor_double()
    if wrong == "prior_done":
        processor.replay_buffer.dataset_dict["dones"][historical.BASE_ROWS - 1] = False
    else:
        processor.replay_buffer._action_horizon = 8
    with pytest.raises(ValueError):
        historical.append_episode(processor, EpisodeDouble(), np)
    assert len(processor.replay_buffer) == historical.BASE_ROWS


def test_actor_and_replay_current_transform_check_detects_wrong_action():
    processor = processor_double()
    episode = EpisodeDouble()
    historical.append_episode(processor, episode, np)

    def transform(raw):
        return {"actions": np.pad(raw["actions"].astype(np.float32), ((0, 0), (0, 18))),
                "state": np.pad(raw["state"].astype(np.float32), (0, 18))}

    actor = SimpleNamespace(input_transforms=transform, input_model_transforms=lambda x: x)
    assert historical.verify_normalization(actor, processor.replay_buffer, episode, np)["rows_checked"] == 349
    processor.replay_buffer.dataset_dict["actions"][historical.BASE_ROWS + 12, 0, 0] += 1
    with pytest.raises(ValueError, match="normalization"):
        historical.verify_normalization(actor, processor.replay_buffer, episode, np)


@pytest.mark.parametrize("wait_failure", [False, True])
@pytest.mark.parametrize("step", [10, 100])
def test_final_checkpoint_uses_full_author_save_wait_close_nonzero(tmp_path, monkeypatch, wait_failure, step):
    events = []
    agent = object()
    checkpoint = tmp_path / "learned-checkpoints"

    class Manager:
        def wait_until_finished(self):
            events.append("wait")
            if wait_failure:
                raise RuntimeError("original asynchronous save failed")
            for name in ("agent", "params"):
                item = checkpoint / str(step) / name
                item.mkdir(parents=True)
                (item / "_METADATA").write_text("explicit test double")

        def all_steps(self):
            return [step]

        def close(self):
            events.append("close")

    manager = Manager()

    def initialize(path, **kwargs):
        assert path == checkpoint and kwargs == {"keep_period": None, "overwrite": False, "resume": False}
        events.append("initialize")
        return manager, False

    def save(actual_manager, actual_agent, actual_step):
        assert actual_manager is manager and actual_agent is agent and actual_step == step
        events.append("save")

    modules = {"expo_ft.agents.alg.agent": SimpleNamespace(initialize_checkpoint_dir=initialize),
               "expo_ft.agents.alg.realtime_expo_ft": SimpleNamespace(save_checkpoint=save)}
    monkeypatch.setattr(probe, "checked_import", lambda name, config: modules[name])
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(block_until_ready=lambda actual: events.append("block") if actual is agent else None))
    config = {"fixed_data": {"condition": historical.CONDITIONS[0]}}
    kwargs = {} if step == 10 else {"step": step}
    if wait_failure:
        with pytest.raises(RuntimeError, match="save failed"):
            historical.save_final_checkpoint(agent, config, tmp_path, **kwargs)
        assert not (tmp_path / "checkpoint.json").exists()
    else:
        result = historical.save_final_checkpoint(agent, config, tmp_path, **kwargs)
        assert result["step"] == step and result["manager_closed"]
        assert not result["separate_process_restore_verified"]
        assert len(result["files"]) == 2
    assert events == ["block", "initialize", "save", "wait", "close"]


def ordered_processor_double():
    """Original methods over CPU doubles; rows here are fixture data, not a dataset."""
    processor = processor_double()
    buffer = processor.replay_buffer
    buffer._size = buffer._insert_index = processor._ep_buffer_start = historical.DEMO_ROWS
    buffer.dataset_dict["dones"][:] = False
    buffer.dataset_dict["dones"][historical.DEMO_ROWS - 1] = True
    buffer.dataset_dict["rewards"][historical.DEMO_ROWS - 1] = 1
    buffer._valids_keep_terminal_windows = True
    buffer._seed, buffer._discount = 42, 0.99
    buffer._gather_storage = []
    buffer._gather = lambda name, values, indices: values[indices].copy()
    processor.batch_size, processor.utd_ratio, processor.offline_ratio = 64, 20, 0
    processor.actor_success_only, processor.use_dagger_hil_sampling = True, False
    processor.offline_iterator = processor.hil_iterator = None
    processor.offline_replay_buffer = []
    processor.replay_iterator = (row for row in [])
    methods = original_methods("expo_ft/data/replay_buffer.py", "PiReplayBuffer",
                               {"restore_success_marks", "_find_episode_boundaries_from_dones"})
    buffer._find_episode_boundaries_from_dones = lambda dones, size: methods[
        "_find_episode_boundaries_from_dones"
    ](buffer, dones, size)
    buffer.restore_success_marks = lambda: methods["restore_success_marks"](buffer)
    return processor


def restore_fixture_timeouts(processor):
    """Run original processor.restore with a declared raw-timeout loader double."""
    methods = original_methods("expo_ft/data/batch_processor.py", "BatchProcessor", {"restore"})

    def load(parent, buffer, *, up_to_step):
        assert parent == "explicit-fixture-parent" and up_to_step == 9908
        assert len(buffer) == historical.DEMO_ROWS + historical.ROWS
        for index in range(9899):
            done = (index + 1) % 900 == 0 or index == 9898
            buffer.insert({"observations": {"state": np.zeros(14, np.float32)},
                           "actions": np.full(14, -0.25, np.float64),
                           "rewards": 0.0, "masks": 1.0, "dones": done,
                           "is_hil": False, "is_success": False})

    methods["restore"].__globals__["restore_replay_buffer"] = load
    methods["restore"](processor, "explicit-fixture-parent", up_to_step=9908)


def original_sample_at(buffer, selected):
    """Unchanged sample_jax AST; explicit deterministic RNG/device doubles."""
    method = original_methods("expo_ft/data/replay_buffer.py", "PiReplayBuffer", {"sample_jax"})["sample_jax"]

    def randint(key, shape, *, minval, maxval):
        assert shape == (len(selected),) and minval == 0
        assert all(minval <= index < maxval for index in selected)
        return np.asarray(selected, dtype=np.int32)

    method.__globals__.update(
        jnp=np,
        jax=SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda seed: seed,
                                                  split=lambda key: (key, key + 1),
                                                  randint=randint),
                            block_until_ready=lambda value: value),
    )
    return method(buffer, len(selected), keys=["rewards", "masks", "dones", "state", "actions"])


def test_ordered_hook_original_restore_and_sampler_expose_terminal_reward(monkeypatch):
    processor = ordered_processor_double()
    buffer = processor.replay_buffer
    episode = EpisodeDouble()
    monkeypatch.setattr(historical, "HistoricalEpisode", lambda spec: episode)
    config = {"fixed_data": {"condition": historical.ORDERED_CONDITION}}
    before = historical.prepare_before_restore(None, processor, config, {})
    assert before["append"]["rows"] == 349 and before["episode"] is episode
    assert len(buffer) == 7418 and not hasattr(buffer, "rng")
    last_success_chunk = buffer.dataset_dict["actions"][7417].copy()
    restore_fixture_timeouts(processor)
    np.testing.assert_array_equal(buffer.dataset_dict["actions"][7417], last_success_chunk)
    episode.pins = []  # No files in this explicitly synthetic storage fixture.

    def transform(raw):
        return {"actions": np.pad(raw["actions"].astype(np.float32), ((0, 0), (0, 18))),
                "state": np.pad(raw["state"].astype(np.float32), (0, 18))}

    actor = SimpleNamespace(input_transforms=transform, input_model_transforms=lambda value: value)
    prepared = historical.prepare_replay(SimpleNamespace(actor=actor), processor, config,
                                         {"episode": {}}, before_restore=before)
    assert prepared["rows_before"] == 7069 and prepared["rows_after"] == 17317
    assert prepared["normalization_check"]["rows_checked"] == 349
    report = prepared["eligibility"]
    assert report["positive_reward_eligible_starts"] == list(range(7410, 7418))
    assert report["terminal_row"] == 7417 and report["max_start_exclusive"] == 17309
    pools = historical.source_pools(buffer, np, before_timeouts=True)
    assert [row["critic_sampleable_starts"] for row in pools] == [7069, 349, 9891]
    assert [row["actor_sampleable_starts"] for row in pools] == [7069, 349, 0]
    assert not hasattr(buffer, "rng")
    batch = original_sample_at(buffer, report["positive_reward_eligible_starts"])
    np.testing.assert_allclose(batch["rewards"], 0.99 ** np.arange(7, -1, -1), rtol=1e-6)
    np.testing.assert_array_equal(batch["masks"], np.zeros(8))
    np.testing.assert_array_equal(batch["dones"], np.ones(8, dtype=bool))
    np.testing.assert_array_equal(batch["valids"], np.ones(8))


def test_original_tail_append_still_has_no_sampleable_added_terminal_reward():
    processor = processor_double()
    buffer = processor.replay_buffer
    historical.append_episode(processor, EpisodeDouble(), np)
    buffer._valids_keep_terminal_windows = True
    buffer._seed, buffer._discount = 42, 0.99
    buffer._gather_storage = []
    buffer._gather = lambda name, values, indices: values[indices].copy()
    batch = original_sample_at(buffer, list(range(16968, 17309)))
    np.testing.assert_array_equal(batch["rewards"], np.zeros(341))
    assert buffer.dataset_dict["rewards"][17316] == 1


@pytest.mark.parametrize("condition", historical.CONDITIONS[:2])
def test_old_conditions_skip_pre_restore_without_touching_processor(condition):
    assert historical.prepare_before_restore(None, object(),
                                              {"fixed_data": {"condition": condition}}, {}) is None


def test_ordered_hook_requires_unadvanced_original_iterator(monkeypatch):
    processor = ordered_processor_double()
    next(processor.replay_iterator, None)
    monkeypatch.setattr(historical, "HistoricalEpisode", lambda spec: pytest.fail("must refuse first"))
    with pytest.raises(ValueError, match="before the first original sample"):
        historical.prepare_before_restore(None, processor,
                                          {"fixed_data": {"condition": historical.ORDERED_CONDITION}}, {})
    assert len(processor.replay_buffer) == 7069


def test_ordered_eligibility_refuses_changed_terminal_labels():
    processor = ordered_processor_double()
    historical.append_episode(processor, EpisodeDouble(), np, start=historical.DEMO_ROWS)
    restore_fixture_timeouts(processor)
    processor.replay_buffer.dataset_dict["rewards"][7417] = 0
    with pytest.raises(ValueError, match="labels or timeout reward"):
        historical.verify_ordered_eligibility(processor.replay_buffer, np)
