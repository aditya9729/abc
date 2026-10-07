"""Read-only GPU reservation checks. This module never changes another process."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _one_mib_idle_evidence(device: dict) -> dict:
    """Validate the sole nonzero allowance using the combined process table.

    NVIDIA documents internal driver memory without active work. These checks
    do not identify the cause of this 1 MiB or provide an atomic reservation.
    Missing/unsupported telemetry cannot establish the allowance.
    See https://docs.nvidia.com/deploy/nvidia-smi/index.html (FB Memory Usage).
    """
    command = ["nvidia-smi", "-q", "-x", "-i", device["uuid"]]
    try:
        xml = subprocess.check_output(command, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Cannot verify the 1 MiB idle GPU allowance; no job started"
        ) from error

    def field(parent: ET.Element, path: str) -> ET.Element:
        matches = parent.findall(path)
        if len(matches) != 1:
            raise ValueError(f"Missing or repeated GPU XML field: {path}")
        return matches[0]

    def value(parent: ET.Element, path: str) -> str:
        element = field(parent, path)
        if len(element) or element.attrib:
            raise ValueError(f"Malformed GPU XML value: {path}")
        return (element.text or "").strip()

    try:
        root = ET.fromstring(xml)
        if root.tag != "nvidia_smi_log":
            raise ValueError("Unknown GPU XML root")
        gpu = field(root, "gpu")
        if len(root.findall(".//gpu")) != 1:
            raise ValueError("Ambiguous GPU XML device nodes")
        if value(gpu, "uuid") != device["uuid"]:
            raise ValueError("GPU XML UUID differs from inventory")
        processes = field(gpu, "processes")
        if len(processes) or processes.attrib or (processes.text or "").strip():
            raise ValueError("GPU XML process table is occupied or unsupported")
        memory = field(gpu, "fb_memory_usage")
        if memory.attrib or (memory.text or "").strip():
            raise ValueError("Malformed GPU XML memory table")
        used = value(memory, "used")
        if used != "1 MiB" or device["memory_used_mib"] != 1:
            raise ValueError("GPU XML memory differs from the 1 MiB inventory")
        utilization = field(gpu, "utilization")
        if utilization.attrib or (utilization.text or "").strip():
            raise ValueError("Malformed GPU XML utilization table")
        if value(utilization, "gpu_util") != "0 %":
            raise ValueError("GPU XML reports GPU activity or unknown utilization")
        if value(utilization, "memory_util") != "0 %":
            raise ValueError("GPU XML reports memory activity or unknown utilization")
        if value(gpu, "display_active") != "Disabled":
            raise ValueError("GPU XML reports an active or unknown display")
    except (ET.ParseError, ValueError) as error:
        raise RuntimeError(
            "Invalid or active 1 MiB GPU telemetry; no job started"
        ) from error
    captured = xml.encode("utf-8")
    return {
        "schema_version": 1,
        "allowed_memory_used_mib": 1,
        "uuid": device["uuid"],
        "xml_query": command,
        "xml_query_timeout_s": 5,
        "xml_sha256": hashlib.sha256(captured).hexdigest(),
        "xml_hash_scope": "Captured text encoded as UTF-8",
        "xml_utf8_bytes": len(captured),
        "process_table": "present_empty_compute_graphics_other",
        "xml_memory_used_mib": 1,
        "gpu_utilization_percent": 0,
        "memory_utilization_percent": 0,
        "display_active": "Disabled",
        "memory_cause": "unproven",
        "scope": "Non-atomic read-only telemetry; campaign lease still required",
    }


def inspect_reserved_gpu(ordinal: int = 0) -> dict:
    """Refuse an occupied device and bind its physical UUID before a job starts.

    This snapshot does not provide an atomic reservation against other projects.
    The shared campaign lease and coordination-board reservation still apply.
    """
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("GPU ordinal must be a nonnegative integer")
    try:
        inventory = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Cannot verify the reserved GPU; no job started") from error
    try:
        devices = {}
        for row in csv.reader(inventory.splitlines()):
            if not row:
                continue
            index, uuid, name, driver, memory = [value.strip() for value in row]
            key = int(index)
            if (
                key in devices
                or GPU_UUID_PATTERN.fullmatch(uuid) is None
                or int(memory) < 0
                or any(existing["uuid"] == uuid for existing in devices.values())
            ):
                raise ValueError("Invalid GPU inventory")
            devices[key] = {
                "ordinal": key,
                "uuid": uuid,
                "name": name,
                "driver_version": driver,
                "memory_used_mib": int(memory),
            }
        device = devices[ordinal]
        occupants = []
        device_uuids = {entry["uuid"] for entry in devices.values()}
        for row in csv.reader(processes.splitlines()):
            if not row:
                continue
            pid, uuid = [value.strip() for value in row]
            if (
                re.fullmatch(r"[0-9]+", pid) is None
                or int(pid) <= 0
                or GPU_UUID_PATTERN.fullmatch(uuid) is None
                or uuid not in device_uuids
            ):
                raise ValueError("Invalid GPU process inventory")
            if uuid == device["uuid"]:
                occupants.append(int(pid))
    except (ValueError, KeyError) as error:
        raise RuntimeError("Invalid GPU inventory; no job started") from error
    if occupants or device["memory_used_mib"] > 1:
        raise RuntimeError(
            f"Reserved GPU {ordinal} is occupied ({device['memory_used_mib']} MiB; "
            f"compute PIDs {occupants}). Existing jobs remain untouched."
        )
    evidence = {}
    if device["memory_used_mib"] == 1:
        evidence["idle_memory_allowance"] = _one_mib_idle_evidence(device)
    return {
        **device,
        **evidence,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
