"""CPU integration with the installed original serializer; explicit fixture only."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from abc_bench import expo_train_export as exporter
from abc_bench.expo_train_collection import pin


@pytest.fixture
def captured():
    path = Path(os.environ["EXPO_COLLECTION_TEST_FIXTURE"])
    value = json.loads(path.read_text())
    assert value["synthetic_physics"] is True
    return value


def test_original_serializer_and_restore_keep_executed_data(captured, tmp_path):
    serializer_pin = pin(
        Path(importlib.util.find_spec("expo_ft").origin).parent
        / "data/replay_buffer.py"
    )
    save = exporter.original_serializer(serializer_pin)
    from expo_ft.data import replay_buffer as original

    assert save is original.save_replay_buffer_transition
    output = tmp_path / "export"
    result = exporter.export(
        captured["descriptor"]["path"],
        captured["result"],
        serializer_pin,
        output,
        12_000_000,
    )
    assert result["status"] == "completed_candidate_replay"
    assert [row["controls"] for row in result["episodes"]] == [2, 3, 1]
    assert [row["reward"] for row in result["episodes"]] == [1.0, 0.0, 0.0]
    assert [row["mask"] for row in result["episodes"]] == [0.0, 1.0, 0.0]

    class Rows:
        def __init__(self):
            self.rows = []

        def insert(self, row):
            self.rows.append(row)

    restored = original.restore_replay_buffer(output, Rows())
    assert len(restored.rows) == 6
    assert [
        int(row["observations"]["head_left"][0, 0, 0]) for row in restored.rows
    ] == [20, 21, 30, 31, 32, 40]
    for row in restored.rows:
        np.testing.assert_array_equal(row["actions"], np.full(14, 0.2))
        assert row["is_hil"] is False
    # Exercise the original mark reconstruction without constructing a model or buffer.
    buffer = object.__new__(original.PiReplayBuffer)
    buffer.dataset_dict = {
        "is_success": np.zeros(6, bool),
        "dones": np.asarray([row["dones"] for row in restored.rows]),
        "rewards": np.asarray([row["rewards"] for row in restored.rows]),
    }
    buffer._size = 6
    buffer.restore_success_marks()
    np.testing.assert_array_equal(
        buffer.dataset_dict["is_success"], [True, True, False, False, False, False]
    )


def test_export_refuses_existing_output(captured, tmp_path):
    serializer_pin = pin(
        Path(importlib.util.find_spec("expo_ft").origin).parent
        / "data/replay_buffer.py"
    )
    with pytest.raises(FileExistsError):
        exporter.export(
            captured["descriptor"]["path"],
            captured["result"],
            serializer_pin,
            tmp_path,
            12_000_000,
        )


def test_export_budget_abort_retains_failed_receipt(captured, tmp_path):
    serializer_pin = pin(
        Path(importlib.util.find_spec("expo_ft").origin).parent
        / "data/replay_buffer.py"
    )
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="byte cap"):
        exporter.export(
            captured["descriptor"]["path"],
            captured["result"],
            serializer_pin,
            output,
            4_100_000,
        )
    result = json.loads((output / "RESULT.json").read_text())
    assert result["status"] == "incomplete_or_failed" and result["files"] == []
