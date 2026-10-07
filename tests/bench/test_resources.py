"""Explicit CPU telemetry doubles verify refusal without touching GPU jobs."""

import subprocess

import pytest

from abc_bench.resources import inspect_reserved_gpu

ZERO = "GPU-00000000-0000-0000-0000-000000000000"
ONE = "GPU-11111111-1111-1111-1111-111111111111"


def telemetry(monkeypatch, inventory, processes):
    values = iter([inventory, processes])
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: next(values)
    )


def test_other_devices_can_run_while_reserved_device_is_free(monkeypatch):
    telemetry(
        monkeypatch,
        f"0, {ZERO}, A100, 580.1, 0\n1, {ONE}, A100, 580.1, 19000\n",
        f"123, {ONE}\n",
    )
    result = inspect_reserved_gpu()
    assert result["uuid"] == ZERO
    assert result["ordinal"] == 0


def test_occupied_reserved_device_is_refused_without_process_control(monkeypatch):
    telemetry(monkeypatch, f"0, {ZERO}, A100, 580.1, 1234\n", f"987, {ZERO}\n")
    with pytest.raises(RuntimeError, match="Existing jobs remain untouched"):
        inspect_reserved_gpu()


def test_graphics_or_unknown_memory_is_also_refused(monkeypatch):
    telemetry(monkeypatch, f"0, {ZERO}, A100, 580.1, 40\n", "")
    with pytest.raises(RuntimeError, match="occupied"):
        inspect_reserved_gpu()


def test_missing_device_and_malformed_inventory_fail_closed(monkeypatch):
    for inventory in (
        "",
        f"0, {ZERO}, A100, 580.1, N/A\n",
        "malformed\n",
        "0, GPU-invalid, A100, 580.1, 0\n",
        f"0, {ZERO}, A100, 580.1, -1\n",
        f"0, {ZERO}, A100, 580.1, 0\n1, {ZERO}, A100, 580.1, 0\n",
    ):
        telemetry(monkeypatch, inventory, "")
        with pytest.raises(RuntimeError, match="Invalid GPU inventory"):
            inspect_reserved_gpu()


def test_unavailable_telemetry_fails_closed(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("synthetic telemetry unavailable")

    monkeypatch.setattr(subprocess, "check_output", unavailable)
    with pytest.raises(RuntimeError, match="no job started"):
        inspect_reserved_gpu()
