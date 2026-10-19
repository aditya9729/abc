"""Pure boundary and orchestration tests; explicit batches, no numerical imports."""

import json
import subprocess
import sys
import weakref
from collections import deque
from pathlib import Path

import pytest

from abc_bench import fixed_data_learning as fixed
from abc_bench import realtime_update_probe as probe


def test_exact_numeric_inventory_preserves_gaps_and_ignores_later(tmp_path):
    root = tmp_path / "checkpoints"
    buffers = root / "buffers"
    buffers.mkdir(parents=True)
    files = []
    for label in (0, 2):
        path = buffers / f"{label:012d}.pkl"
        path.write_bytes(b"no pickle execution")
        files.append(probe.file_pin(path))
    (buffers / "000000000003.pkl").write_bytes(b"later")
    assert fixed.numeric_inventory(root, files, cutoff=2, count=2) == [
        Path(p["path"]) for p in files
    ]
    # Passing buffers instead of its checkpoint parent must fail.
    with pytest.raises(ValueError, match="directory"):
        fixed.numeric_inventory(buffers, files, cutoff=2, count=2)


@pytest.mark.parametrize(
    "mutation", ["extra", "missing", "same_numeric_label", "symlink", "wrong_pin_set"]
)
def test_numeric_inventory_refuses_changed_restore_selection(tmp_path, mutation):
    root = tmp_path / "checkpoints"
    buffers = root / "buffers"
    buffers.mkdir(parents=True)
    first = buffers / "000000000000.pkl"
    last = buffers / "000000000002.pkl"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    expected = [probe.file_pin(first), probe.file_pin(last)]
    if mutation == "extra":
        (buffers / "000000000001.pkl").write_bytes(b"extra")
    elif mutation == "missing":
        last.unlink()
    elif mutation == "same_numeric_label":
        last.rename(buffers / "0.pkl")
    elif mutation == "symlink":
        last.unlink()
        last.symlink_to(first)
    else:
        expected[1] = probe.file_pin(first)
    with pytest.raises(ValueError):
        fixed.numeric_inventory(root, expected, cutoff=2, count=2)


@pytest.mark.parametrize("calls,critic,actor", [(1, 5120, 128), (10, 16640, 704)])
def test_generated_and_consumed_are_distinct(calls, critic, actor):
    result = fixed.accounting(calls, calls + 1)
    assert result["generated_critic_rows"] == critic
    assert result["generated_actor_rows"] == actor
    assert result["consumed_critic_rows"] == 1280 * calls
    assert result["consumed_actor_rows"] == 64 * calls
    assert result["unconsumed_internal_critic_batches"] == 2


@pytest.mark.parametrize("calls,returned", [(True, 2), (11, 12), (10, 10), (-1, 0)])
def test_accounting_refuses_missing_prefetch_or_extra_updates(calls, returned):
    with pytest.raises(ValueError):
        fixed.accounting(calls, returned)


class ExplicitProcessor:
    """The original queue shape, with numbered strings rather than model tensors."""

    def __init__(self):
        self.generated = 0
        self.actor_generated = 0
        self.replay_rng = 0
        self.replay_iterator = self.iterator()
        self.returned = []

    def iterator(self):
        queue = deque()

        def append():
            self.generated += 1
            self.replay_rng += 1
            queue.append({"critic_id": self.generated})

        for _ in range(3):
            append()
        while queue:
            yield queue.popleft()
            append()

    def next_batch(self, rng):
        critic = next(self.replay_iterator)
        self.actor_generated += 1
        self.replay_rng += 1
        result = (critic, {"actor_id": self.actor_generated}, rng + 1)
        self.returned.append(result)
        return result


@pytest.mark.parametrize("calls,schedule", [(10, (0, 1, 10)), (100, (0, 10, 30, 100)), (100, ())])
def test_actual_orchestration_retains_pending_and_quiesces_measurement(calls, schedule):
    processor = ExplicitProcessor()
    updates, measurements, emitted = [], [], {}

    def snapshot(agent, combine, pending):
        return (
            agent,
            processor.replay_rng,
            combine,
            None if pending is None else pending[2],
        )

    def measure(agent, call):
        measurements.append((call, processor.generated, processor.actor_generated))
        if call:
            assert processor.returned[call][0] is not processor.returned[call - 1][0]
            assert len(fixed.retained_queue(processor.replay_iterator)) == 2
        return {"call": call, "agent": agent}

    def update(agent, critic, actor, call):
        # Each exact pending object is consumed once; measurement never regenerates it.
        assert critic is processor.returned[call - 1][0]
        assert actor is processor.returned[call - 1][1]
        updates.append((critic["critic_id"], actor["actor_id"]))
        return agent + 1, {"completed": call}

    result = fixed.update_sequence(
        [0],
        processor,
        142,
        update,
        measure,
        snapshot,
        lambda batch, rows: {**batch, "rows": rows},
        emitted.__setitem__,
        **({} if calls == 10 else {"calls": calls, "measure_at": schedule}),
    )
    assert result[0] == calls
    assert updates == [(i, i) for i in range(1, calls + 1)]
    assert [row[0] for row in measurements] == list(schedule)
    if schedule:
        assert measurements[-1] == (calls, calls + 3, calls + 1)
    assert result[3]["returned_critic"]["critic_id"] == calls + 1
    assert [x["critic_id"] for x in result[3]["internal_critics"]] == [calls + 2, calls + 3]
    assert len(result[2]) == calls and len(emitted) == calls + len(schedule)


@pytest.mark.parametrize("calls,schedule", [(101, (0, 10, 30, 101)), (100, (0, 1, 100)), (True, (0, 1, 10))])
def test_unsupported_schedule_fails_before_agent_or_sampler_access(calls, schedule):
    with pytest.raises(ValueError, match="schedule"):
        fixed.update_sequence(None, None, None, None, None, None, None, None,
                              calls=calls, measure_at=schedule)


def test_hundred_call_accounting_preserves_three_unconsumed_critic_batches():
    result = fixed.accounting(100, 101, calls=100)
    assert result["consumed_critic_rows"] == 128000
    assert result["consumed_actor_rows"] == 6400
    assert result["generated_critic_batches"] == 103
    assert result["generated_actor_batches"] == 101
    with pytest.raises(ValueError):
        fixed.accounting(101, 102, calls=100)


def test_hundred_call_probes_preserve_stream_against_serial_no_probe_reference():
    reference, measured = ExplicitProcessor(), ExplicitProcessor()
    expected, combine = [], 142
    # Reference consumes the same stream without invoking any measurement.
    for call in range(101):
        critic, actor, combine = reference.next_batch(combine)
        if call < 100:
            expected.append((critic["critic_id"], actor["actor_id"]))
    actual, measurements = [], []

    def update(agent, critic, actor, call):
        actual.append((critic["critic_id"], actor["actor_id"]))
        return call, {}

    fixed.update_sequence(
        [0], measured, 142, update,
        lambda agent, call: measurements.append(call),
        lambda agent, combine, pending: (agent, measured.replay_rng, combine),
        lambda batch, rows: dict(batch), lambda *_: None,
        calls=100, measure_at=(0, 10, 30, 100),
    )
    assert actual == expected
    assert measurements == [0, 10, 30, 100]
    assert measured.replay_rng == reference.replay_rng
    assert measured.generated == reference.generated == 103
    assert measured.actor_generated == reference.actor_generated == 101


def test_measurement_rng_mutation_is_refused():
    state = {"rng": 1}

    def bad_measure(_):
        state["rng"] += 1

    with pytest.raises(ValueError, match="Measurement changed"):
        fixed.isolated_measure(0, bad_measure, lambda *_: state["rng"], None, None)


@pytest.mark.parametrize("calls,schedule", [(10, (0, 1, 10)), (100, (0, 10, 30, 100)), (100, ())])
def test_consumed_states_and_batches_are_released_without_losing_prefetch(calls, schedule):
    class Tracked:
        def __init__(self, kind, number):
            self.kind = kind
            self.number = number

    class WeakProcessor:
        """Retain only the original queue; diagnostic records use weak references."""

        def __init__(self):
            self.generated = {"critic": 0, "actor": 0}
            self.references = {"critic": {}, "actor": {}}
            self.returned = []
            self.replay_iterator = self.iterator()

        def make(self, kind):
            self.generated[kind] += 1
            number = self.generated[kind]
            batch = Tracked(kind, number)
            self.references[kind][number] = weakref.ref(batch)
            return batch

        def iterator(self):
            queue = deque()
            for _ in range(3):
                queue.append(self.make("critic"))
            while queue:
                yield queue.popleft()
                queue.append(self.make("critic"))

        def next_batch(self, rng):
            critic = next(self.replay_iterator)
            actor = self.make("actor")
            self.returned.append((weakref.ref(critic), weakref.ref(actor), rng + 1))
            return critic, actor, rng + 1

    holder = [Tracked("agent", 0)]
    initial = weakref.ref(holder[0])
    processor = WeakProcessor()
    updates, measurements, emitted = [], [], {}

    def snapshot(agent, combine, pending):
        return (
            agent.number,
            tuple(processor.generated.items()),
            combine,
            None if pending is None else pending[2],
        )

    def measure(agent, call):
        assert not holder
        if call:
            # The exact pending result remains live through its original Future.
            assert processor.returned[call][0]() is not None
            assert processor.returned[call][1]() is not None
            queued = fixed.retained_queue(processor.replay_iterator)
            assert [batch.number for batch in queued] == [call + 2, call + 3]
            assert processor.references["critic"][call + 2]() is queued[0]
            assert processor.references["critic"][call + 3]() is queued[1]
        measurements.append(call)
        return {"call": call, "agent_number": agent.number}

    def update(agent, critic, actor, call):
        assert agent.number == call - 1
        assert critic is processor.returned[call - 1][0]()
        assert actor is processor.returned[call - 1][1]()
        if call >= 2:
            assert initial() is None
        if call >= 3:
            # Measurement 1 must not retain the batch consumed by update 2.
            assert processor.returned[1][0]() is None
            assert processor.returned[1][1]() is None
        updates.append((critic.number, actor.number))
        return Tracked("agent", call), {"completed": call}

    result = fixed.update_sequence(
        holder,
        processor,
        142,
        update,
        measure,
        snapshot,
        lambda batch, rows: {
            "kind": batch.kind,
            "number": batch.number,
            "rows": rows,
        },
        emitted.__setitem__,
        **({} if calls == 10 else {"calls": calls, "measure_at": schedule}),
    )
    assert not holder and initial() is None
    assert result[0].number == calls
    assert updates == [(i, i) for i in range(1, calls + 1)]
    assert measurements == list(schedule)
    assert processor.generated == {"critic": calls + 3, "actor": calls + 1}
    assert result[3]["returned_critic"]["number"] == calls + 1
    assert [batch["number"] for batch in result[3]["internal_critics"]] == [calls + 2, calls + 3]
    assert len(result[2]) == calls and len(emitted) == calls + len(schedule)


def test_measurement_pending_queue_replacement_is_refused():
    processor = ExplicitProcessor()
    pending = processor.next_batch(142)
    queue = processor.replay_iterator.gi_frame.f_locals["queue"]

    def bad_measure(_):
        queue[0] = dict(queue[0])

    with pytest.raises(ValueError, match="replaced"):
        fixed.isolated_measure(
            0, bad_measure, lambda *_: 1, pending, processor.replay_iterator
        )


def test_original_argmax_ties_do_not_inflate_reference_top1():
    tied = fixed.rank_record([1.0] * 32)
    assert tied["argmax"] == 0 and tied["reference_rank_with_original_tie_rule"] == 32
    assert tied["reference_top1"] is False and tied["reference_margin"] == 0
    values = [0.0] * 32
    values[31] = 1.0
    assert fixed.rank_record(values)["reference_top1"] is True


@pytest.mark.parametrize(
    "values", [[0.0] * 31, [float("nan")] * 32, [float("inf")] * 32]
)
def test_nonfinite_or_incomplete_ranking_refused(values):
    with pytest.raises(ValueError):
        fixed.rank_record(values)


def test_fixed_phase_requires_exact_spec_and_installed_helper(tmp_path, monkeypatch):
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps({"phase": fixed.PHASE, "update_calls": 10, "measure_at": [0, 1, 10]})
    )
    binding = {
        "phase": fixed.PHASE,
        "fixed_data": {
            "spec": probe.file_pin(path),
            "python": probe.file_pin(Path(sys.executable).resolve()),
        },
        "installed_modules": {fixed.MODULE: {}},
    }
    monkeypatch.setattr(fixed, "SPEC_SHA256", binding["fixed_data"]["spec"]["sha256"])
    assert fixed.load_spec(binding)["update_calls"] == 10
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="identity"):
        fixed.load_spec(binding)
    binding["fixed_data"]["spec"] = probe.file_pin(path)
    with pytest.raises(ValueError, match="specification"):
        fixed.load_spec(binding)


def test_module_import_has_no_numerical_or_runtime_side_effects():
    # Inspect only these installed modules in a fresh isolated interpreter.
    code = (
        "import sys; from abc_bench import fixed_data_learning, realtime_update_probe; "
        "assert not ({'jax','torch','mujoco','expo_ft','sadhana_terminal_critic'} & set(sys.modules))"
    )
    subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True, timeout=30)


def test_failed_measurement_stops_before_any_later_update():
    processor = ExplicitProcessor()
    calls = []

    def measure(_agent, call):
        if call == 1:
            raise ValueError("explicit failed measurement")
        return {}

    def update(agent, _critic, _actor, call):
        calls.append(call)
        return agent + 1, {}

    with pytest.raises(ValueError, match="failed measurement"):
        fixed.update_sequence(
            [0],
            processor,
            142,
            update,
            measure,
            lambda a, key, pending: (a, key, processor.replay_rng),
            lambda *_: {},
            lambda *_: None,
        )
    assert calls == [1]
    assert (
        processor.actor_generated == 2
    )  # Exact one-ahead batch exists; no third fetch.


def test_existing_one_update_config_and_new_phase_remain_separate(
    tmp_path, monkeypatch
):
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    config = {
        "schema": probe.SCHEMA,
        "online_config": probe.file_pin(inputs),
        "source_manifest": probe.file_pin(inputs),
        "source_root": str(tmp_path),
        "runtime_inputs": probe.file_pin(inputs),
        "initialization_acceptance": probe.file_pin(inputs),
        "installed_modules": {probe.MODULE: {}},
        "memory_environment": probe.MEMORY_ENV,
        "worker_seconds": 2300,
        "parent_seconds": 2400,
        "minimum_start_bytes": probe.MIN_START_BYTES,
        "minimum_run_bytes": probe.MIN_RUN_BYTES,
    }
    monkeypatch.setattr(probe, "ONLINE_SHA", config["online_config"]["sha256"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert "phase" not in probe.load_config(path)
    config["phase"] = "arbitrary-update-count"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fields"):
        probe.load_config(path)
    config["fixed_data"] = {}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="phase"):
        probe.load_config(path)
