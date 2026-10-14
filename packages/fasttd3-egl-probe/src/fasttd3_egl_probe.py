"""A canonical-lease-only native EGL probe. Importing this module is inert."""

from __future__ import annotations

import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

VERSION = "0.1.0"
TASK = "h1hand-hurdle-v0"
WORKSPACE = Path("/home/user/aditya")
EGL_EXTENSIONS = 0x3055
EGL_DEVICE_EXT = 0x322C
EGL_CUDA_DEVICE_NV = 0x323A


def require_lease(results: Path) -> str:
    """Read the existing ABC ancestry/held-lock contract; allocate no resource."""
    if not results.is_absolute() or not results.resolve().is_relative_to(WORKSPACE):
        raise ValueError("Lease results must be under /home/user/aditya")
    ledger = json.loads((results / "gpu_budget.json").read_text())
    owner, visible = (
        ledger.get("active_parent_pid"),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    if type(owner) is not int or owner <= 1:
        raise RuntimeError("No active ABC subset coordinator")
    if (
        ledger.get("gpu") != 0
        or visible != ledger.get("active_gpu_uuid")
        or not visible.startswith("GPU-")
        or "," in visible
    ):
        raise RuntimeError("Probe requires the canonical single GPU0 UUID reservation")
    if (
        os.environ.get("MUJOCO_GL") != "egl"
        or os.environ.get("MUJOCO_EGL_DEVICE_ID") != "0"
    ):
        raise RuntimeError("Probe requires original EGL with device index0")
    current = os.getppid()
    for _ in range(16):
        if current == owner:
            break
        if current <= 1:
            raise RuntimeError("Probe is not a descendant of the coordinator")
        stat = Path(f"/proc/{current}/stat").read_text()
        current = int(stat[stat.rfind(")") + 2 :].split()[1])
    else:
        raise RuntimeError("Coordinator ancestry exceeds the bounded process tree")
    with (results / ".gpu-budget.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return visible
        fcntl.flock(lock, fcntl.LOCK_UN)
        raise RuntimeError("The coordinator GPU0 lease is not held")


def pci_identity(value: str) -> str:
    match = re.fullmatch(
        r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value.strip()
    )
    if not match:
        raise ValueError("Expected a PCI domain:bus:device.function identifier")
    domain, bus, device, function = (int(x, 16) for x in match.groups())
    return f"{domain:08x}:{bus:02x}:{device:02x}.{function:x}"


def physical_gpu0(visible: str) -> dict[str, str]:
    row = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=20,
        )
        .strip()
        .split(",")
    )
    if len(row) != 3 or row[0].strip() != "0" or row[1].strip() != visible:
        raise RuntimeError("Physical GPU0 identity differs from the coordinator UUID")
    return {"uuid": visible, "pci_bus_id": pci_identity(row[2])}


class DeviceQueries:
    """Public EGL/CUDA query ABI; never constructs a context or display."""

    def __init__(self) -> None:
        self.egl = C.CDLL("libEGL.so.1")
        self.cuda = C.CDLL("libcuda.so.1")
        self.egl.eglGetProcAddress.argtypes = [C.c_char_p]
        self.egl.eglGetProcAddress.restype = C.c_void_p
        # EGLAttrib is intptr_t, not the installed PyOpenGL int32 pointer.
        attr_pointer = C.POINTER(C.c_ssize_t)
        self.query_devices = self._extension(
            "eglQueryDevicesEXT",
            C.c_uint,
            C.c_int,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_int),
        )
        self.query_device = self._extension(
            "eglQueryDeviceAttribEXT", C.c_uint, C.c_void_p, C.c_int, attr_pointer
        )
        self.query_string = self._extension(
            "eglQueryDeviceStringEXT", C.c_char_p, C.c_void_p, C.c_int
        )
        self.query_display = self._extension(
            "eglQueryDisplayAttribEXT", C.c_uint, C.c_void_p, C.c_int, attr_pointer
        )
        self.egl.eglGetCurrentDisplay.argtypes = []
        self.egl.eglGetCurrentDisplay.restype = C.c_void_p
        self.egl.eglGetCurrentContext.argtypes = []
        self.egl.eglGetCurrentContext.restype = C.c_void_p
        self.cuda.cuInit.argtypes, self.cuda.cuInit.restype = [C.c_uint], C.c_int
        self.cuda.cuDeviceGetPCIBusId.argtypes = [C.c_char_p, C.c_int, C.c_int]
        self.cuda.cuDeviceGetPCIBusId.restype = C.c_int
        self.cuda.cuDeviceGetUuid.argtypes = [C.c_void_p, C.c_int]
        self.cuda.cuDeviceGetUuid.restype = C.c_int

    def _extension(self, name: str, result: Any, *arguments: Any) -> Any:
        address = self.egl.eglGetProcAddress(name.encode())
        if not address:
            raise RuntimeError(f"Public EGL query unavailable: {name}")
        return C.CFUNCTYPE(result, *arguments)(address)

    def _identity(self, device: int) -> dict[str, Any]:
        extensions = self.query_string(device, EGL_EXTENSIONS)
        if not extensions or b"EGL_NV_device_cuda" not in extensions.split():
            raise RuntimeError("Selected EGL device lacks EGL_NV_device_cuda identity")
        handle = C.c_ssize_t()
        if self.query_device(device, EGL_CUDA_DEVICE_NV, C.byref(handle)) != 1:
            raise RuntimeError("EGL_CUDA_DEVICE_NV query failed")
        if not 0 <= handle.value <= 2**31 - 1:
            raise RuntimeError("EGL returned an invalid CUDA device handle")
        if self.cuda.cuInit(0) != 0:
            raise RuntimeError("CUDA driver initialization failed")
        bus, identifier = C.create_string_buffer(32), (C.c_ubyte * 16)()
        if (
            self.cuda.cuDeviceGetPCIBusId(bus, len(bus), handle.value) != 0
            or self.cuda.cuDeviceGetUuid(C.byref(identifier), handle.value) != 0
        ):
            raise RuntimeError("CUDA device UUID/PCI identity query failed")
        return {
            "egl_device": device,
            "cuda_device_handle": handle.value,
            "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identifier))),
            "pci_bus_id": pci_identity(bus.value.decode()),
            "device_extensions": extensions.decode(),
        }

    def selected(self) -> dict[str, Any]:
        count = C.c_int()
        if (
            self.query_devices(0, None, C.byref(count)) != 1
            or not 0 < count.value <= 64
        ):
            raise RuntimeError("Public EGL device enumeration failed or exceeded64")
        devices = (C.c_void_p * count.value)()
        if (
            self.query_devices(count.value, devices, C.byref(count)) != 1
            or count.value < 1
            or not devices[0]
        ):
            raise RuntimeError("EGL device index0 is unavailable")
        return self._identity(devices[0])

    def current(self) -> dict[str, Any]:
        display, context = (
            self.egl.eglGetCurrentDisplay(),
            self.egl.eglGetCurrentContext(),
        )
        if not display or not context:
            raise RuntimeError("Native render has no current EGL display/context")
        device = C.c_ssize_t()
        if (
            self.query_display(display, EGL_DEVICE_EXT, C.byref(device)) != 1
            or not device.value
        ):
            raise RuntimeError("Current EGL display device identity is unavailable")
        return {
            **self._identity(device.value),
            "egl_display": display,
            "egl_context": context,
        }


def match_identity(
    actual: dict[str, Any],
    expected: dict[str, str],
    selected: dict[str, Any] | None = None,
) -> None:
    if any(actual.get(key) != expected[key] for key in ("uuid", "pci_bus_id")):
        raise RuntimeError("EGL device UUID/PCI differs from physical leased GPU0")
    if selected is not None and actual.get("egl_device") != selected.get("egl_device"):
        raise RuntimeError("Current render device differs from preflight EGL device0")


def pin(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def capture_state(
    env: Any, observation: Any, output: Path, control: int
) -> dict[str, Any]:
    import mujoco
    import numpy as np

    native = env.unwrapped
    if np.asarray(observation).shape != (151,) or not np.isfinite(observation).all():
        raise RuntimeError("Native Hurdle observation must be finite shape151")
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(native.model, spec), dtype=np.float64)
    mujoco.mj_getState(native.model, native.data, integration, spec)
    arrays = {
        name: np.asarray(getattr(native.data, name)).copy()
        for name in ("qpos", "qvel", "ctrl", "act")
    }
    arrays.update(
        time=np.asarray(native.data.time),
        observation=np.asarray(observation),
        integration=integration,
    )
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Native Hurdle state contains nonfinite values")
    warnings = [int(warning.number) for warning in native.data.warning]
    if not warnings or any(warnings):
        raise RuntimeError(f"Native MuJoCo warnings must all be zero: {warnings}")
    path = output / f"state-{control:02d}.npz"
    np.savez(path, **arrays)
    return {
        "control": control,
        "warnings": warnings,
        "native_time": float(native.data.time),
        **pin(path),
    }


def capture_frame(frame: Any, output: Path, control: int) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    frame = np.asarray(frame)
    if (
        frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape[:2]) < 1
        or frame.max() == frame.min()
    ):
        raise RuntimeError("Native RGB must be nonconstant uint8 HWC3")
    path = output / f"rgb-{control:02d}.png"
    imageio.imwrite(path, frame)
    return {
        "control": control,
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        **pin(path),
    }


def run(output: Path, lease_results: Path) -> dict[str, Any]:
    """Run eight native zero controls; return no learned-policy metric."""
    if not output.is_absolute() or not output.resolve().is_relative_to(WORKSPACE):
        raise ValueError("Probe output must be under /home/user/aditya")
    output.mkdir(exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fasttd3_egl_render_qualification",
        "version": VERSION,
        "status": "running",
        "task": TASK,
        "controls": 0,
        "learner_updates": 0,
        "training_ready": False,
        "world128_ready": False,
        "states": [],
        "images": [],
        "render_identities": [],
        "step_outputs": [],
        "closed": False,
        "scope": "One-world native EGL/device/RGB boundary only; no task score or learner qualification",
    }
    env, started = None, time.monotonic()
    receipt = output / "EGL_RENDER_RESULT.json"

    def publish() -> None:
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(receipt)

    try:
        visible = require_lease(lease_results)
        expected = physical_gpu0(visible)
        queries = DeviceQueries()
        selected = queries.selected()
        result.update(expected_identity=expected, selected_identity=selected)
        match_identity(selected, expected)
        publish()
        # No simulator or GL-context import occurs before positive identity.
        import gymnasium as gym
        import humanoid_bench  # noqa: F401 - author environment registration
        import numpy as np

        env = gym.make(TASK, render_mode="rgb_array")
        env.unwrapped.seed(
            0
        )  # Original make_env's default seed; reset retains author behavior.
        observation, _ = env.reset()
        if (
            env.action_space.shape != (61,)
            or not np.all(env.action_space.low == -1)
            or not np.all(env.action_space.high == 1)
        ):
            raise RuntimeError("Expected original Hurdle normalized61-action space")
        for control in range(9):
            if control:
                observation, reward, terminated, truncated, _ = env.step(
                    np.zeros(61, dtype=np.float32)
                )
                result["controls"] = control
                if not np.isfinite(reward) or terminated or truncated:
                    raise RuntimeError(
                        "Eight-control render qualification terminated or has nonfinite reward"
                    )
                result["step_outputs"].append(
                    {
                        "control": control,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
            if control in (0, 8):
                frame = env.render()
                current = queries.current()
                match_identity(current, expected, selected)
                result["render_identities"].append(current)
                result["images"].append(capture_frame(frame, output, control))
            # Include any warning raised by the final render, not only stepping.
            result["states"].append(capture_state(env, observation, output, control))
            publish()
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if env is not None:
                env.close()
            result["closed"] = True
        except (Exception, KeyboardInterrupt) as error:
            result.update(
                status="failed", close_error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            result["elapsed_s"] = time.monotonic() - started
            publish()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease-results", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output, arguments.lease_results)


if __name__ == "__main__":
    main()
