"""Read-only GPU reservation checks. This module never changes another process."""

from __future__ import annotations

import csv
import re
import subprocess
from datetime import datetime, timezone

GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


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
        for row in csv.reader(processes.splitlines()):
            if not row:
                continue
            pid, uuid = [value.strip() for value in row]
            if uuid == device["uuid"]:
                occupants.append(int(pid))
    except (ValueError, KeyError) as error:
        raise RuntimeError("Invalid GPU inventory; no job started") from error
    if occupants or device["memory_used_mib"] > 0:
        raise RuntimeError(
            f"Reserved GPU {ordinal} is occupied ({device['memory_used_mib']} MiB; "
            f"compute PIDs {occupants}). Existing jobs remain untouched."
        )
    return {**device, "observed_at": datetime.now(timezone.utc).isoformat()}
