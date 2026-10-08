"""Narrow Linux lease guard for one direct matched-evaluator exec.

This file is also an absolute, SHA-pinned bootstrap script in the isolated
worker interpreter. It imports only the standard library. The guardian owns
the inherited original flock until the runner has reaped its worker group.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MAX_METADATA_BYTES = 16 * 1024**2


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def read_json(path: str | Path, maximum: int = MAX_METADATA_BYTES) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate evaluation JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite evaluation JSON value: " + value)

    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "Evaluation metadata exceeds its byte bound")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "Evaluation JSON object required")
    canonical(value)
    return value


def verify_pin(pin: dict, maximum: int = MAX_METADATA_BYTES) -> bytes:
    require(
        isinstance(pin, dict)
        and set(pin) == {"path", "bytes", "sha256"}
        and isinstance(pin["path"], str)
        and Path(pin["path"]).is_absolute()
        and type(pin["bytes"]) is int
        and 0 <= pin["bytes"] <= maximum
        and isinstance(pin["sha256"], str)
        and len(pin["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in pin["sha256"]),
        "Absolute bounded evaluation pin required",
    )
    with Path(pin["path"]).open("rb") as stream:
        raw = stream.read(pin["bytes"] + 1)
    require(
        len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"],
        "Evaluation artifact pin changed: " + pin["path"],
    )
    return raw


def artifact(path: str | Path) -> dict:
    path = Path(path).absolute()
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def exclusive_json(path: Path, value: dict) -> None:
    """Publish a complete new JSON file without replacing a previous result."""
    raw = canonical(value) + b"\n"
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_identity(pid: int) -> dict:
    require(type(pid) is int and pid > 0, "Positive owned PID required")
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "boot_id": boot_id(),
        "pgid": int(fields[2]),
    }


def identity_matches(token: dict, *, live: bool = False) -> bool:
    try:
        if process_identity(token["pid"]) != token:
            return False
        if live:
            raw = Path(f"/proc/{token['pid']}/stat").read_text()
            return raw[raw.rfind(")") + 2 :].split()[0] != "Z"
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False


def signal_owned(token: dict, signum: int, *, group: bool = False) -> bool:
    if not identity_matches(token):
        return False
    if group:
        os.killpg(token["pgid"], signum)
    else:
        os.kill(token["pid"], signum)
    return True


def group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def subreaper(value: int | None = None) -> int:
    """Set Linux child adoption only for the owner of this one worker group."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError("Linux owned-child subreaper unavailable")
    if value is not None and libc.prctl(36, value, 0, 0, 0) != 0:
        raise OSError("Cannot set Linux owned-child subreaper")
    return previous.value


def _children(token: dict, known: dict[int, dict], pgid: int) -> None:
    """Read only the children of verified owned processes; never scan all PIDs."""
    if not identity_matches(token):
        return
    try:
        raw = Path(f"/proc/{token['pid']}/task/{token['pid']}/children").read_text()
        for text in raw.split():
            pid = int(text)
            if pid in known:
                continue
            child = process_identity(pid)
            if child["pgid"] == pgid:
                require(len(known) < 256, "Owned evaluator descendant bound exceeded")
                known[pid] = child
    except FileNotFoundError:
        return


def _record_unresolved(spec: dict, reason: str, cleanup_complete: bool) -> None:
    """Only a dead owner's matching lease can receive a guardian tombstone."""
    if identity_matches(spec["parent"], live=True):
        return
    path = Path(spec["campaign_directory"]) / "gpu_budget.json"
    ledger = read_json(path)
    lease = ledger.get("active_visual_lease")
    require(
        isinstance(lease, dict)
        and lease["nonce"] == spec["nonce"]
        and lease["parent"] == spec["parent"],
        "Dead-parent ledger ownership differs",
    )
    for key in ("worker", "guardian"):
        require(
            key not in lease or lease[key] == spec[key],
            "Dead-parent child lifetime differs",
        )
    lease["status"] = "unresolved"
    lease.setdefault("reason", reason)
    lease["owned_group_cleanup_complete"] = cleanup_complete
    # Retain charged_seconds, active_parent_pid and UUID. Root must reconcile.
    temporary = path.with_name("gpu_budget.guardian-" + spec["nonce"] + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(ledger) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _guardian(spec: dict, lease_fd: int, worker: dict) -> None:
    os.setsid()  # Keep the inherited lock while the evaluator group is killed.
    directory = Path(spec["control_directory"])
    guardian = process_identity(os.getpid())
    spec["worker"], spec["guardian"] = worker, guardian
    known = {worker["pid"]: worker}
    exclusive_json(
        directory / "guardian-ready.json",
        {
            "nonce": spec["nonce"],
            "parent": spec["parent"],
            "worker": worker,
            "guardian": guardian,
        },
    )
    reason = None
    cleanup_complete = False
    try:
        while True:
            now = time.monotonic()
            for token in list(known.values()):
                _children(token, known, worker["pgid"])
            resident = 0
            resident_tokens = dict(known)
            resident_tokens[spec["parent"]["pid"]] = spec["parent"]
            resident_tokens[guardian["pid"]] = guardian
            for token in resident_tokens.values():
                if identity_matches(token, live=True):
                    try:
                        fields = (
                            Path(f"/proc/{token['pid']}/status")
                            .read_text()
                            .splitlines()
                        )
                        resident += next(
                            int(line.split()[1]) * 1024
                            for line in fields
                            if line.startswith("VmRSS:")
                        )
                    except (FileNotFoundError, StopIteration):
                        pass
            if (
                resident > spec["native_resources"]["host_rss_limit_bytes"]
                and reason is None
            ):
                reason = "host_resident_memory_exceeded"
                signal_owned(spec["parent"], signal.SIGTERM)
            if not identity_matches(spec["parent"], live=True):
                reason = reason or "parent_died"
                break
            stop = directory / "guardian-stop.json"
            if stop.exists():
                value = read_json(stop)
                require(
                    value
                    == {
                        "nonce": spec["nonce"],
                        "parent": spec["parent"],
                        "worker_group_reaped": True,
                    },
                    "Guardian stop ownership differs",
                )
                require(
                    not group_exists(worker["pgid"]),
                    "Worker group remains at guardian release",
                )
                cleanup_complete = True
                exclusive_json(
                    directory / "guardian-outcome.json",
                    {
                        "status": "clean" if reason is None else "unresolved",
                        "nonce": spec["nonce"],
                        "guardian": guardian,
                        "reason": reason,
                        "owned_group_cleanup_complete": True,
                    },
                )
                return
            if now >= spec["cancel_deadline"] and reason is None:
                reason = "watchdog_expired"
                signal_owned(spec["parent"], signal.SIGTERM)
            if now >= spec["terminate_deadline"]:
                reason = reason or "watchdog_expired"
                signal_owned(spec["parent"], signal.SIGKILL)
                break
            time.sleep(0.02)
        # A verified member anchors the original process group even if its leader exited.
        anchor = next((t for t in known.values() if identity_matches(t)), None)
        if anchor is not None:
            signal_owned(anchor, signal.SIGKILL, group=True)
        until = min(spec["hard_deadline"], time.monotonic() + spec["cleanup_s"])
        while group_exists(worker["pgid"]) and time.monotonic() < until:
            time.sleep(0.02)
        cleanup_complete = not group_exists(worker["pgid"])
        _record_unresolved(spec, reason or "guardian_failed", cleanup_complete)
        exclusive_json(
            directory / "guardian-outcome.json",
            {
                "status": "unresolved",
                "nonce": spec["nonce"],
                "guardian": guardian,
                "reason": reason,
                "owned_group_cleanup_complete": cleanup_complete,
            },
        )
    except BaseException as error:  # noqa: BLE001 - keep cancellation and first failure during owned cleanup
        try:
            anchor = next((t for t in known.values() if identity_matches(t)), None)
            if anchor is not None:
                signal_owned(anchor, signal.SIGKILL, group=True)
            _record_unresolved(
                spec, f"guardian_failure: {type(error).__name__}: {error}", False
            )
            exclusive_json(
                directory / "guardian-outcome.json",
                {
                    "status": "unresolved",
                    "nonce": spec["nonce"],
                    "guardian": guardian,
                    "reason": f"{type(error).__name__}: {error}",
                    "owned_group_cleanup_complete": False,
                },
            )
        finally:
            raise
    finally:
        os.close(lease_fd)


def _wait_ready(spec: dict) -> dict:
    path = Path(spec["control_directory"]) / "guardian-ready.json"
    while time.monotonic() < spec["cancel_deadline"]:
        if path.exists():
            ready = read_json(path)
            require(
                ready["nonce"] == spec["nonce"]
                and ready["parent"] == spec["parent"]
                and identity_matches(ready["guardian"], live=True)
                and identity_matches(ready["worker"])
                and ready["worker"]["pgid"] == ready["worker"]["pid"],
                "Guardian handshake differs",
            )
            return ready
        time.sleep(0.01)
    raise TimeoutError("Guardian did not arm before the evaluator deadline")


def execute_guarded(
    command: list[str],
    log_path: Path,
    timeout: float,
    *,
    gpu_uuid: str,
    spec: dict,
    lease_fd: int,
) -> int:
    """Run one worker, retain its zombie identity until group termination, then reap."""
    require(
        command == spec["bootstrap_command"] and math.isfinite(timeout) and timeout > 0,
        "Guarded command or timeout differs",
    )
    spec["previous_subreaper"] = subreaper(1)
    environment = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment.update(
        CUDA_VISIBLE_DEVICES=gpu_uuid,
        MUJOCO_GL="disable",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    process = None
    first = None
    clean = False
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=spec["control_directory"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                pass_fds=(lease_fd,),
            )
            worker = process_identity(process.pid)
            spec["worker"] = worker
            ready = _wait_ready(spec)
            require(
                ready["worker"] == worker, "Guardian bound a different worker lifetime"
            )
            spec["guardian"] = ready["guardian"]
            ledger_path = Path(spec["campaign_directory"]) / "gpu_budget.json"
            ledger = read_json(ledger_path)
            lease = ledger.get("active_visual_lease")
            require(
                isinstance(lease, dict)
                and lease["nonce"] == spec["nonce"]
                and lease["parent"] == spec["parent"]
                and lease["status"] == "running",
                "Cannot arm another root lease",
            )
            lease.update(
                worker=worker, guardian=ready["guardian"], worker_pgid=worker["pgid"]
            )
            temporary = ledger_path.with_name(
                "gpu_budget.parent-" + spec["nonce"] + ".tmp"
            )
            with temporary.open("xb") as stream:
                stream.write(canonical(ledger) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(ledger_path)
            exclusive_json(
                Path(spec["control_directory"]) / "parent-armed.json",
                {
                    "nonce": spec["nonce"],
                    "parent": spec["parent"],
                    "worker": worker,
                    "guardian": ready["guardian"],
                },
            )
            while time.monotonic() < spec["cancel_deadline"]:
                found = os.waitid(
                    os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
                if found is not None:
                    break
                ready = read_json(
                    Path(spec["control_directory"]) / "guardian-ready.json"
                )
                require(
                    identity_matches(ready["guardian"], live=True),
                    "Owned evaluator guardian died",
                )
                time.sleep(0.01)
            else:
                raise subprocess.TimeoutExpired(command, timeout)
    except BaseException as error:  # noqa: BLE001 - always reap owned children before propagating cancellation
        first = error
    finally:
        # A second watchdog TERM must not interrupt the bounded group cleanup.
        cleanup_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None:
            # The leader has not been reaped, so this original group ID cannot be recycled.
            worker = spec["worker"]
            try:
                signal_owned(worker, signal.SIGTERM, group=True)
                time.sleep(min(0.1, spec["cleanup_s"] / 4))
                signal_owned(worker, signal.SIGKILL, group=True)
                process.wait(
                    timeout=max(
                        0.01,
                        min(
                            spec["cleanup_s"],
                            spec["terminate_deadline"] - time.monotonic(),
                        ),
                    )
                )
                while time.monotonic() < spec["terminate_deadline"]:
                    while True:
                        try:
                            pid, _ = os.waitpid(-worker["pgid"], os.WNOHANG)
                            if pid == 0:
                                break
                        except ChildProcessError:
                            break
                    if not group_exists(worker["pgid"]):
                        clean = True
                        break
                    time.sleep(0.01)
                require(clean, "Owned evaluator group cleanup did not complete")
            except BaseException as error:  # noqa: BLE001 - preserve the first failure and cleanup evidence
                if first is None:
                    first = error
                else:
                    first.__notes__ = [
                        *getattr(first, "__notes__", []),
                        "worker cleanup: " + str(error),
                    ]
        try:
            exclusive_json(
                Path(spec["control_directory"]) / "execution-outcome.json",
                {
                    "worker_exit_code": None if process is None else process.returncode,
                    "owned_group_cleanup_complete": clean,
                    "error": None
                    if first is None
                    else f"{type(first).__name__}: {first}",
                    "secondary_errors": []
                    if first is None
                    else getattr(first, "__notes__", []),
                },
            )
        except BaseException as error:  # noqa: BLE001 - publication must not replace the original failure
            if first is None:
                first = error
            else:
                first.__notes__ = [
                    *getattr(first, "__notes__", []),
                    "execution outcome publication: " + str(error),
                ]
        signal.signal(signal.SIGTERM, cleanup_handler)
        if "guardian" not in spec:
            subreaper(spec["previous_subreaper"])
    if first is not None:
        raise first
    require(process is not None, "Evaluator was not dispatched")
    return process.returncode


def finish_guard(spec: dict) -> dict:
    """Finish supervision after receipt validation; the runner still owns its lock."""
    directory = Path(spec["control_directory"])
    ready = None
    try:
        ready = read_json(directory / "guardian-ready.json")
        require(
            ready["guardian"] == spec.get("guardian")
            and identity_matches(ready["guardian"]),
            "Guardian ownership was lost",
        )
        require(
            not group_exists(ready["worker"]["pgid"]),
            "Worker group must be reaped before release",
        )
        exclusive_json(
            directory / "guardian-stop.json",
            {
                "nonce": spec["nonce"],
                "parent": spec["parent"],
                "worker_group_reaped": True,
            },
        )
        path = directory / "guardian-outcome.json"
        while time.monotonic() < spec["terminate_deadline"]:
            if path.exists():
                value = read_json(path)
                require(
                    value["nonce"] == spec["nonce"]
                    and value["guardian"] == ready["guardian"],
                    "Guardian outcome ownership differs",
                )
                pid, _ = os.waitpid(ready["guardian"]["pid"], os.WNOHANG)
                if pid:
                    require(
                        value["status"] == "clean"
                        and value["owned_group_cleanup_complete"] is True,
                        "Guardian release was unresolved",
                    )
                    return value
            time.sleep(0.01)
        raise TimeoutError("Owned guardian shutdown did not complete")
    finally:
        # Release only a verified guardian after its worker group is gone.
        if (
            ready is not None
            and ready["guardian"] == spec.get("guardian")
            and not group_exists(ready["worker"]["pgid"])
        ):
            signal_owned(ready["guardian"], signal.SIGKILL)
            while time.monotonic() < spec["hard_deadline"]:
                try:
                    pid, _ = os.waitpid(ready["guardian"]["pid"], os.WNOHANG)
                    if pid:
                        break
                except ChildProcessError:
                    break
                time.sleep(0.01)
        if "previous_subreaper" in spec:
            subreaper(spec["previous_subreaper"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--spec-bytes", required=True, type=int)
    args = parser.parse_args()
    verify_pin(
        {
            "path": str(args.spec.absolute()),
            "bytes": args.spec_bytes,
            "sha256": args.spec_sha256,
        }
    )
    spec = read_json(args.spec)
    require(
        spec["schema_version"] == 1
        and spec["kind"] == "abc_matched_visual_direct_exec",
        "Bootstrap schema differs",
    )
    require(
        process_identity(os.getppid()) == spec["parent"]
        and identity_matches(spec["parent"], live=True),
        "Direct runner parent/start identity differs",
    )
    require(
        time.monotonic() < spec["cancel_deadline"]
        and spec["hard_deadline"]
        > spec["terminate_deadline"]
        > spec["cancel_deadline"],
        "Finite bootstrap deadline differs",
    )
    lease_fd = spec["lease_fd"]
    stat = os.fstat(lease_fd)
    campaign = Path(spec["campaign_directory"])
    original = (campaign / ".gpu-budget.lock").stat()
    require(
        [stat.st_dev, stat.st_ino]
        == spec["lock_identity"]
        == [original.st_dev, original.st_ino],
        "Inherited original lock inode differs",
    )
    with (campaign / ".gpu-budget.lock").open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError("Original lease must already be held")
    worker = process_identity(os.getpid())
    require(worker["pgid"] == worker["pid"], "Evaluator must own its new process group")
    argv = spec["worker_command"]
    require(
        argv[:5]
        == [spec["worker_python"], "-I", "-B", "-m", "nrh.visual_method_evaluation"],
        "Only the exact public evaluator CLI can execute",
    )
    require(
        sys.prefix == spec["worker_prefix"] and sys.version == spec["worker_version"],
        "Literal isolated worker Python prefix/version differs",
    )
    import resource

    inherited_limits = {
        "virtual_address": list(resource.getrlimit(resource.RLIMIT_AS)),
        "cpu_seconds": list(resource.getrlimit(resource.RLIMIT_CPU)),
    }
    for pin in spec["startup_pins"]:
        verify_pin(pin, 64 * 1024**2)
    from importlib.machinery import PathFinder

    package = PathFinder.find_spec("nrh")
    module = (
        None
        if package is None
        else PathFinder.find_spec(
            "nrh.visual_method_evaluation", package.submodule_search_locations
        )
    )
    require(
        module is not None
        and module.origin is not None
        and Path(module.origin).absolute() == Path(spec["evaluator_origin"]),
        "Isolated evaluator import origin differs",
    )
    exclusive_json(
        Path(spec["control_directory"]) / "bootstrap-origin.json",
        {
            "python_literal": sys.executable,
            "python_resolved": str(Path(sys.executable).resolve()),
            "prefix": sys.prefix,
            "version": sys.version,
            "nrh_origin_literal": module.origin,
            "nrh_origin_resolved": str(Path(module.origin).resolve()),
            "parent": spec["parent"],
            "worker": worker,
            "nonce": spec["nonce"],
            "inherited_resource_limits": inherited_limits,
            "bootstrap_origin_literal": str(Path(__file__).absolute()),
            "bootstrap_origin_resolved": str(Path(__file__).resolve()),
        },
    )
    pid = os.fork()
    if pid == 0:
        try:
            _guardian(spec, lease_fd, worker)
        finally:
            os._exit(0)
    ready = _wait_ready(spec)
    armed = Path(spec["control_directory"]) / "parent-armed.json"
    while not armed.exists() and time.monotonic() < spec["cancel_deadline"]:
        require(
            identity_matches(spec["parent"], live=True)
            and identity_matches(ready["guardian"], live=True),
            "Root/guardian died before execution admission",
        )
        time.sleep(0.01)
    require(
        armed.exists()
        and read_json(armed)
        == {
            "nonce": spec["nonce"],
            "parent": spec["parent"],
            "worker": worker,
            "guardian": ready["guardian"],
        },
        "Root did not admit this exact direct exec",
    )
    ledger = read_json(campaign / "gpu_budget.json")
    lease = ledger.get("active_visual_lease")
    require(
        isinstance(lease, dict)
        and lease["nonce"] == spec["nonce"]
        and lease["parent"] == spec["parent"]
        and lease["worker"] == worker
        and lease["guardian"] == ready["guardian"]
        and lease["worker_pgid"] == worker["pgid"]
        and ledger["active_parent_pid"] == spec["parent"]["pid"]
        and ledger["active_gpu_uuid"] == os.environ.get("CUDA_VISIBLE_DEVICES"),
        "Original ledger direct-exec ownership differs",
    )
    os.close(lease_fd)
    virtual = spec["native_resources"]["virtual_address_limit_bytes"]
    if virtual is not None:
        resource.setrlimit(resource.RLIMIT_AS, (virtual, virtual))
    os.execv(spec["worker_python"], argv)  # Keep the evaluator's direct runner parent.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
