"""Explicit CPU query/environment doubles. These are not native qualifications."""

import ctypes as C
import json
import os
import subprocess
import sys
import types
import uuid

import numpy as np
import pytest

import fasttd3_egl_probe as probe

ORIGINAL_DEVICE_QUERIES = probe.DeviceQueries

UUID = "GPU-12345678-1234-5678-9abc-123456789abc"
EXPECTED = {"uuid": UUID, "pci_bus_id": "00000000:01:00.0"}
IDENTITY = {**EXPECTED, "egl_device": 9876, "cuda_device_handle": 0}


def test_installed_import_has_no_native_or_gpu_side_effects():
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import ctypes,sys\nctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No native load'))\nimport fasttd3_egl_probe\nassert not {'mujoco','humanoid_bench','gymnasium','torch','OpenGL'}.intersection(sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def test_pci_canonicalization_and_exact_device_match():
    assert probe.pci_identity("0000:AB:00.0") == "00000000:ab:00.0"
    with pytest.raises(ValueError, match="PCI"):
        probe.pci_identity("0")
    probe.match_identity(IDENTITY, EXPECTED, IDENTITY)
    for changed in (
        {"uuid": "GPU-other"},
        {"pci_bus_id": "00000000:02:00.0"},
        {"egl_device": 1234},
    ):
        with pytest.raises(RuntimeError, match="differs"):
            probe.match_identity({**IDENTITY, **changed}, EXPECTED, IDENTITY)


def query_fixture():
    # Explicit public-ABI metadata double. No EGL or CUDA library is loaded.
    query = ORIGINAL_DEVICE_QUERIES.__new__(ORIGINAL_DEVICE_QUERIES)
    query.query_string = lambda *_: b"EGL_NV_device_cuda"

    def device(_device, _attribute, result):
        result._obj.value = 0  # CUdevice zero is valid, not a null EGL device.
        return 1

    def pci(buffer, _length, _device):
        buffer.value = b"0000:01:00.0"
        return 0

    def identifier(buffer, _device):
        C.memmove(buffer, uuid.UUID(UUID[4:]).bytes, 16)
        return 0

    query.query_device = device
    query.cuda = types.SimpleNamespace(
        cuInit=lambda _: 0, cuDeviceGetPCIBusId=pci, cuDeviceGetUuid=identifier
    )
    query.egl = types.SimpleNamespace(
        eglGetCurrentDisplay=lambda: 123, eglGetCurrentContext=lambda: 456
    )

    def devices(_maximum, output, count):
        count._obj.value = 1
        if output is not None:
            output[0] = 9876
        return 1

    def display(_display, _attribute, result):
        result._obj.value = 9876
        return 1

    query.query_devices, query.query_display = devices, display
    return query


def test_public_query_chain_and_valid_cuda_zero_handle():
    query = query_fixture()
    selected, current = query.selected(), query.current()
    probe.match_identity(selected, EXPECTED)
    probe.match_identity(current, EXPECTED, selected)
    assert current["egl_display"] == 123 and current["egl_context"] == 456
    assert selected["cuda_device_handle"] == 0


def test_pointer_width_egl_attribute_binding_without_loading_libraries(monkeypatch):
    class Function:
        pass

    egl = types.SimpleNamespace(
        eglGetProcAddress=Function(),
        eglGetCurrentDisplay=Function(),
        eglGetCurrentContext=Function(),
    )
    cuda = types.SimpleNamespace(
        cuInit=Function(), cuDeviceGetPCIBusId=Function(), cuDeviceGetUuid=Function()
    )
    monkeypatch.setattr(
        probe.C, "CDLL", lambda name: egl if name == "libEGL.so.1" else cuda
    )
    arguments = {}

    def extension(_self, name, _result, *args):
        arguments[name] = args
        return Function()

    monkeypatch.setattr(probe.DeviceQueries, "_extension", extension)
    probe.DeviceQueries()
    assert arguments["eglQueryDeviceAttribEXT"][-1] == C.POINTER(C.c_ssize_t)
    assert arguments["eglQueryDisplayAttribEXT"][-1] == C.POINTER(C.c_ssize_t)
    assert C.sizeof(C.c_ssize_t) == C.sizeof(C.c_void_p)


@pytest.mark.parametrize(
    "failure",
    [
        "extensions",
        "attribute",
        "init",
        "pci",
        "uuid",
        "enumeration",
        "current",
        "display",
    ],
)
def test_missing_or_failed_public_identity_queries_fail_closed(failure):
    query = query_fixture()
    if failure == "extensions":
        query.query_string = lambda *_: b"EGL_OTHER"
    elif failure == "attribute":
        query.query_device = lambda *_: 0
    elif failure == "init":
        query.cuda.cuInit = lambda _: 1
    elif failure == "pci":
        query.cuda.cuDeviceGetPCIBusId = lambda *_: 1
    elif failure == "uuid":
        query.cuda.cuDeviceGetUuid = lambda *_: 1
    elif failure == "enumeration":
        query.query_devices = lambda *_: 0
    elif failure == "current":
        query.egl.eglGetCurrentContext = lambda: None
    elif failure == "display":
        query.query_display = lambda *_: 0
    with pytest.raises(RuntimeError):
        query.current() if failure in {"current", "display"} else query.selected()


def test_lease_reuses_ancestry_visibility_and_held_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    ledger = {"gpu": 0, "active_gpu_uuid": UUID, "active_parent_pid": os.getppid()}
    (tmp_path / "gpu_budget.json").write_text(json.dumps(ledger))
    (tmp_path / ".gpu-budget.lock").touch()
    with pytest.raises(RuntimeError, match="not held"):
        probe.require_lease(tmp_path)

    def held(*_):
        raise BlockingIOError("Explicit CPU held-lock double")

    monkeypatch.setattr(probe.fcntl, "flock", held)
    assert probe.require_lease(tmp_path) == UUID
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    with pytest.raises(RuntimeError, match="index0"):
        probe.require_lease(tmp_path)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
    with pytest.raises(RuntimeError, match="reservation"):
        probe.require_lease(tmp_path)


@pytest.fixture
def native_double(tmp_path, monkeypatch):
    calls = []

    class Env:
        action_space = types.SimpleNamespace(
            shape=(61,), low=-np.ones(61), high=np.ones(61)
        )

        def __init__(self):
            self.unwrapped, self.closed, self.step_count = self, False, 0

        def seed(self, seed):
            calls.append(("seed", seed))

        def reset(self):
            calls.append("reset")
            return np.zeros(151), {}

        def render(self):
            calls.append("render")
            return np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

        def step(self, action):
            assert (
                action.shape == (61,)
                and action.dtype == np.float32
                and not action.any()
            )
            self.step_count += 1
            calls.append("step")
            return np.zeros(151), 0.0, False, False, {}

        def close(self):
            self.closed = True
            calls.append("close")

    env = Env()
    monkeypatch.setattr(probe, "require_lease", lambda _: UUID)
    monkeypatch.setattr(probe, "physical_gpu0", lambda _: EXPECTED)
    monkeypatch.setattr(probe, "DeviceQueries", lambda: query_fixture())

    def make(task, render_mode):
        assert task == probe.TASK and render_mode == "rgb_array"
        calls.append("make")
        return env

    monkeypatch.setitem(sys.modules, "gymnasium", types.SimpleNamespace(make=make))
    monkeypatch.setitem(sys.modules, "humanoid_bench", types.SimpleNamespace())
    def state(_env, _obs, _out, control):
        calls.append(("state", control))
        return {
            "control": control,
            "scope": "Explicit CPU double, no native state",
        }

    monkeypatch.setattr(probe, "capture_state", state)
    monkeypatch.setattr(
        probe,
        "capture_frame",
        lambda _frame, _out, control: {
            "control": control,
            "scope": "Explicit CPU double, no native RGB",
        },
    )
    return env, calls


def test_eight_control_lifecycle_and_separate_readiness_labels(tmp_path, native_double):
    env, calls = native_double
    result = probe.run(tmp_path / "fixture-output", tmp_path)
    assert result["status"] == "passed" and result["closed"] and env.closed
    assert result["controls"] == 8 and len(result["states"]) == 9
    assert len(result["images"]) == len(result["render_identities"]) == 2
    assert calls[:4] == ["make", ("seed", 0), "reset", "render"]
    assert calls.count("step") == 8
    assert calls[-3:] == ["render", ("state", 8), "close"]
    assert (
        result["learner_updates"] == 0
        and not result["training_ready"]
        and not result["world128_ready"]
    )


@pytest.mark.parametrize(
    "failure", ["preflight", "current", "render", "step", "terminal", "close"]
)
def test_failure_persists_partial_receipt_and_closes_only_owned_env(
    tmp_path, monkeypatch, native_double, failure
):
    env, calls = native_double

    def fail(*_):
        raise RuntimeError("Explicit CPU failure double")

    if failure == "preflight":
        monkeypatch.setattr(
            probe, "physical_gpu0", lambda _: {**EXPECTED, "uuid": "GPU-other"}
        )
    elif failure == "current":
        query = query_fixture()
        query.current = fail
        monkeypatch.setattr(probe, "DeviceQueries", lambda: query)
    elif failure == "terminal":
        env.step = lambda _: (np.zeros(151), 0.0, True, False, {})
    else:
        setattr(env, failure, fail)
    with pytest.raises(RuntimeError):
        probe.run(tmp_path / "fixture-output", tmp_path)
    result = json.loads(
        (tmp_path / "fixture-output/EGL_RENDER_RESULT.json").read_text()
    )
    assert result["status"] == "failed" and not result["training_ready"]
    if failure == "preflight":
        assert "make" not in calls
    elif failure != "close":
        assert env.closed
    else:
        assert not result["closed"] and "close_error" in result


def test_state_all_warning_counts_finite_full_integration_and_frame_contract(
    tmp_path, monkeypatch
):
    data = types.SimpleNamespace(
        qpos=np.zeros(2),
        qvel=np.ones(2),
        ctrl=np.zeros(1),
        act=np.zeros(0),
        time=0.0,
        warning=[types.SimpleNamespace(number=0) for _ in range(8)],
    )
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(data=data, model=object())
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        types.SimpleNamespace(
            mjtState=types.SimpleNamespace(mjSTATE_INTEGRATION=8191),
            mj_stateSize=lambda *_: 7,
            mj_getState=lambda _model, _data, state, _spec: state.fill(1.0),
        ),
    )
    row = probe.capture_state(env, np.zeros(151), tmp_path, 0)
    assert row["warnings"] == [0] * 8 and np.load(row["path"])["integration"].shape == (
        7,
    )
    data.warning[-1].number = 1
    with pytest.raises(RuntimeError, match="warnings"):
        probe.capture_state(env, np.zeros(151), tmp_path, 1)
    data.warning[-1].number = 0
    data.qpos[0] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        probe.capture_state(env, np.zeros(151), tmp_path, 1)
    with pytest.raises(RuntimeError, match="RGB"):
        probe.capture_frame(np.zeros((2, 2, 3), dtype=np.uint8), tmp_path, 0)
