"""Explicit CPU telemetry doubles verify refusal without touching GPU jobs."""

import hashlib
import subprocess

import pytest

from abc_bench.resources import inspect_reserved_gpu

ZERO = "GPU-00000000-0000-0000-0000-000000000000"
ONE = "GPU-11111111-1111-1111-1111-111111111111"


def telemetry(monkeypatch, inventory, processes, xml=None):
    values = iter([inventory, processes] + ([] if xml is None else [xml]))
    calls = []

    def query(command, *, text, timeout):
        assert text is True
        assert timeout == 5
        calls.append(command)
        return next(values)

    monkeypatch.setattr(subprocess, "check_output", query)
    return calls


def idle_xml(uuid=ZERO):
    """Explicit minimal nvidia-smi telemetry; no native GPU is queried."""
    return f"""<?xml version="1.0" ?>
<nvidia_smi_log>
  <gpu id="00000000:05:00.0">
    <uuid>{uuid}</uuid>
    <display_active>Disabled</display_active>
    <fb_memory_usage><used>1 MiB</used></fb_memory_usage>
    <utilization><gpu_util>0 %</gpu_util><memory_util>0 %</memory_util></utilization>
    <processes>
    </processes>
  </gpu>
</nvidia_smi_log>
"""


def test_other_devices_can_run_while_reserved_device_is_free(monkeypatch):
    calls = telemetry(
        monkeypatch,
        f"0, {ZERO}, A100, 580.1, 0\n1, {ONE}, A100, 580.1, 19000\n",
        f"123, {ONE}\n",
    )
    result = inspect_reserved_gpu()
    assert result["uuid"] == ZERO
    assert result["ordinal"] == 0
    assert "idle_memory_allowance" not in result
    assert len(calls) == 2


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


def test_one_mib_idle_allowance_requires_complete_matching_xml(monkeypatch):
    xml = idle_xml()
    calls = telemetry(
        monkeypatch,
        f"0, {ZERO}, A100, 580.1, 1\n1, {ONE}, A100, 580.1, 19000\n",
        f"123, {ONE}\n",
        xml,
    )
    result = inspect_reserved_gpu()
    evidence = result["idle_memory_allowance"]
    assert result["memory_used_mib"] == 1
    assert evidence["allowed_memory_used_mib"] == 1
    assert evidence["uuid"] == ZERO
    assert evidence["xml_memory_used_mib"] == result["memory_used_mib"]
    assert evidence["gpu_utilization_percent"] == 0
    assert evidence["memory_utilization_percent"] == 0
    assert evidence["display_active"] == "Disabled"
    assert evidence["process_table"] == "present_empty_compute_graphics_other"
    assert evidence["memory_cause"] == "unproven"
    assert evidence["xml_sha256"] == hashlib.sha256(xml.encode("utf-8")).hexdigest()
    assert evidence["xml_utf8_bytes"] == len(xml.encode("utf-8"))
    assert evidence["xml_query"] == ["nvidia-smi", "-q", "-x", "-i", ZERO]
    assert evidence["xml_query_timeout_s"] == 5
    assert calls[-1] == evidence["xml_query"]
    assert len(calls) == 3


def test_one_mib_matching_uuid_query_uses_requested_physical_device(monkeypatch):
    calls = telemetry(
        monkeypatch,
        f"0, {ZERO}, A100, 580.1, 19000\n1, {ONE}, A100, 580.1, 1\n",
        f"123, {ZERO}\n",
        idle_xml(ONE),
    )
    result = inspect_reserved_gpu(1)
    assert result["uuid"] == ONE
    assert calls[-1] == ["nvidia-smi", "-q", "-x", "-i", ONE]


@pytest.mark.parametrize("memory", [0, 1, 2])
def test_compute_pid_never_uses_idle_allowance(monkeypatch, memory):
    calls = telemetry(
        monkeypatch, f"0, {ZERO}, A100, 580.1, {memory}\n", f"987, {ZERO}\n"
    )
    with pytest.raises(RuntimeError, match="compute PIDs \\[987\\]"):
        inspect_reserved_gpu()
    assert len(calls) == 2


@pytest.mark.parametrize("memory", [2, 40, 19000])
def test_more_than_one_mib_without_compute_pid_is_refused(monkeypatch, memory):
    calls = telemetry(monkeypatch, f"0, {ZERO}, A100, 580.1, {memory}\n", "")
    with pytest.raises(RuntimeError, match="occupied"):
        inspect_reserved_gpu()
    assert len(calls) == 2


@pytest.mark.parametrize("process_type", ["C", "G", "C+G", "M", "O", "unknown"])
def test_any_combined_xml_process_prevents_allowance(monkeypatch, process_type):
    xml = idle_xml().replace(
        "<processes>",
        "<processes><process_info><pid>987</pid>"
        f"<type>{process_type}</type></process_info>",
    )
    telemetry(monkeypatch, f"0, {ZERO}, A100, 580.1, 1\n", "", xml)
    with pytest.raises(RuntimeError, match="no job started"):
        inspect_reserved_gpu()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (f"<uuid>{ZERO}</uuid>", f"<uuid>{ONE}</uuid>"),
        (f"<uuid>{ZERO}</uuid>", ""),
        (f"<uuid>{ZERO}</uuid>", f"<uuid>{ZERO}</uuid><uuid>{ZERO}</uuid>"),
        (f"<uuid>{ZERO}</uuid>", f"<uuid>{ZERO}<unknown/></uuid>"),
        ("<used>1 MiB</used>", "<used>0 MiB</used>"),
        ("<used>1 MiB</used>", "<used>2 MiB</used>"),
        ("<used>1 MiB</used>", "<used>1 MB</used>"),
        ("<used>1 MiB</used>", "<used>1 KiB</used>"),
        ("<used>1 MiB</used>", "<used>N/A</used>"),
        ("<used>1 MiB</used>", "<used>NaN MiB</used>"),
        ("<used>1 MiB</used>", ""),
        ("<used>1 MiB</used>", "<used>1 MiB</used><used>1 MiB</used>"),
        ("<gpu_util>0 %</gpu_util>", "<gpu_util>1 %</gpu_util>"),
        ("<gpu_util>0 %</gpu_util>", "<gpu_util>N/A</gpu_util>"),
        ("<gpu_util>0 %</gpu_util>", "<gpu_util>0 MiB</gpu_util>"),
        ("<gpu_util>0 %</gpu_util>", ""),
        ("<memory_util>0 %</memory_util>", "<memory_util>1 %</memory_util>"),
        ("<memory_util>0 %</memory_util>", "<memory_util>N/A</memory_util>"),
        ("<memory_util>0 %</memory_util>", ""),
        (
            "<display_active>Disabled</display_active>",
            "<display_active>Enabled</display_active>",
        ),
        (
            "<display_active>Disabled</display_active>",
            "<display_active>N/A</display_active>",
        ),
        ("<display_active>Disabled</display_active>", ""),
        ("<processes>\n    </processes>", ""),
        ("<processes>\n    </processes>", "<processes>N/A</processes>"),
        ("<processes>\n    </processes>", "<processes><unknown/></processes>"),
        ("<processes>\n    </processes>", '<processes unsupported="true"/>'),
        ("<processes>\n    </processes>", "<processes/><processes/>"),
        ("<fb_memory_usage>", "<fb_memory_usage/><fb_memory_usage>"),
        ("<fb_memory_usage>", '<fb_memory_usage unsupported="true">'),
        ("<fb_memory_usage>", "<fb_memory_usage>N/A"),
        ("<utilization>", "<utilization/><utilization>"),
        ("<utilization>", '<utilization unsupported="true">'),
        ("<utilization>", "<utilization>N/A"),
        ("<nvidia_smi_log>", "<unsupported>"),
    ],
)
def test_unknown_active_mismatched_or_malformed_xml_fails_closed(monkeypatch, old, new):
    telemetry(
        monkeypatch,
        f"0, {ZERO}, A100, 580.1, 1\n",
        "",
        idle_xml().replace(old, new),
    )
    with pytest.raises(RuntimeError, match="no job started"):
        inspect_reserved_gpu()


@pytest.mark.parametrize(
    "xml",
    [
        "",
        "not xml",
        "<nvidia_smi_log/>",
        "<nvidia_smi_log><gpu/></nvidia_smi_log>",
        idle_xml().replace("</gpu>", "</gpu><gpu/>", 1),
        idle_xml().replace("</gpu>", "<wrapper><gpu/></wrapper></gpu>", 1),
    ],
)
def test_invalid_or_ambiguous_gpu_xml_refuses(monkeypatch, xml):
    telemetry(monkeypatch, f"0, {ZERO}, A100, 580.1, 1\n", "", xml)
    with pytest.raises(RuntimeError, match="no job started"):
        inspect_reserved_gpu()


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("synthetic query unavailable"),
        subprocess.CalledProcessError(1, ["nvidia-smi"]),
        subprocess.TimeoutExpired(["nvidia-smi"], 5),
    ],
)
def test_xml_query_failure_refuses_after_valid_csv(monkeypatch, error):
    calls = []

    def query(command, *, text, timeout):
        calls.append(command)
        assert text is True and timeout == 5
        if len(calls) == 1:
            return f"0, {ZERO}, A100, 580.1, 1\n"
        if len(calls) == 2:
            return ""
        raise error

    monkeypatch.setattr(subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="no job started"):
        inspect_reserved_gpu()
    assert calls[-1] == ["nvidia-smi", "-q", "-x", "-i", ZERO]


@pytest.mark.parametrize(
    "processes",
    [
        "malformed\n",
        "987, GPU-invalid\n",
        f"N/A, {ONE}\n",
        f"0, {ONE}\n",
        f"-1, {ONE}\n",
        f"1.5, {ONE}\n",
        "987, GPU-22222222-2222-2222-2222-222222222222\n",
        f"987, {ONE}, extra\n",
    ],
)
@pytest.mark.parametrize("memory", [0, 1])
def test_malformed_global_process_inventory_refuses_even_foreign_rows(
    monkeypatch, processes, memory
):
    calls = telemetry(
        monkeypatch,
        f"0, {ZERO}, A100, 580.1, {memory}\n1, {ONE}, A100, 580.1, 19000\n",
        processes,
    )
    with pytest.raises(RuntimeError, match="Invalid GPU inventory"):
        inspect_reserved_gpu()
    assert len(calls) == 2


@pytest.mark.parametrize("ordinal", [-1, 0.5, True, "0"])
def test_invalid_ordinal_refuses_before_telemetry(monkeypatch, ordinal):
    calls = telemetry(monkeypatch, "", "")
    with pytest.raises(ValueError, match="ordinal"):
        inspect_reserved_gpu(ordinal)
    assert calls == []
