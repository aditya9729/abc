"""Explicit CPU doubles. These tests produce no policy or simulator score."""

import ast
import builtins
import concurrent.futures
import copy
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from abc_bench import demo_collect as demo
from abc_bench import upstream_eval


@pytest.fixture
def config_file(tmp_path):
    checkpoint = tmp_path / "explicit-checkpoint-double.pt"
    checkpoint.write_bytes(b"CPU double, not policy weights")
    checkpoint_pin = upstream_eval.artifact(checkpoint)
    model = {
        "camera_keys": ["top", "left", "right"],
        "backbone": {
            "checkpoint": None,
            "load_base_checkpoint": True,
            "image_size": 224,
            "fixed_seq_len": 256,
            "feature_layer": -1,
            "train_backbone": True,
            "train_siglip": True,
            "activation_checkpointing": False,
            "proprio_token": True,
        },
        "dit": {
            "hidden_size": 512,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "state_dim": 14,
            "action_dim": 14,
            "chunk_length": 30,
            "num_pool_tokens": 8,
            "pool_num_heads": 8,
            "pool_qk_norm": True,
            "direct_state_conditioning": True,
        },
    }
    metadata_path = tmp_path / "explicit-metadata-double.json"
    metadata_path.write_text(
        json.dumps(
            {
                "format": "abc_checkpoint_metadata/v1",
                "bytes": checkpoint_pin["bytes"],
                "sha256": checkpoint_pin["sha256"],
                "max_action_prefix": 8,
                "model_config": model,
            }
        )
    )
    config = {
        "schema_version": 1,
        "checkpoint": checkpoint_pin,
        "metadata": upstream_eval.artifact(metadata_path),
        "sim_eval": copy.deepcopy(upstream_eval.RECIPE),
        "split": "development",
        "max_output_bytes": 32 * 1024**2,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config, model


def forbid_native(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "abc_minimal",
            "abc_sim",
            "torch",
            "mujoco",
            "warp",
            "mujoco_warp",
        }:
            raise AssertionError(f"Unexpected native import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize(
    "change",
    [
        {"split": "test"},
        {"max_output_bytes": True},
        {"max_output_bytes": 5 * 1024**3 + 1},
        {"sim_eval": {"diffusion_steps": 5}},
        {"sim_eval": {"rtc_prefix_length": 5}},
        {"sim_eval": {"save_video": False}},
        {"sim_eval": {"video_every_n_actions": 2}},
        {"sim_eval": {"num_worlds": True}},
        {"sim_eval": {"num_worlds": 4}},
        {"sim_eval": {"seed": 9100}},
        {"sim_eval": {"parallel_worlds": 1}},
        {"sim_eval": {"randomization": "{}"}},
    ],
)
def test_bad_config_rejects_before_lease_native(
    config_file, tmp_path, monkeypatch, change
):
    path, config, _ = config_file
    for key, value in change.items():
        if key == "sim_eval":
            config[key].update(value)
        else:
            config[key] = value
    path.write_text(json.dumps(config))
    forbid_native(monkeypatch)
    monkeypatch.setattr(
        demo,
        "require_subset_lease",
        lambda: pytest.fail("Invalid config reached lease"),
    )
    with pytest.raises(ValueError):
        demo.run(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_lease_rejection_precedes_native_and_output(config_file, monkeypatch, tmp_path):
    path, _, _ = config_file
    forbid_native(monkeypatch)

    def deny():
        raise RuntimeError("No active CPU-double coordinator")

    monkeypatch.setattr(demo, "require_subset_lease", deny)
    with pytest.raises(RuntimeError, match="No active"):
        demo.run(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("old", [False, True])
def test_canonical_bookkeeping_is_permitted_but_previous_payload_is_not(
    config_file, monkeypatch, tmp_path, old
):
    path, _, _ = config_file
    output = tmp_path / "out"
    output.mkdir()
    (output / "job.json").write_text("{}")
    (output / "run.log").write_text("canonical CPU-double bookkeeping\n")
    if old:
        (output / "receipt.json").write_text("{}")
    forbid_native(monkeypatch)

    def admitted_boundary():
        raise RuntimeError("Reached canonical lease boundary")

    monkeypatch.setattr(demo, "require_subset_lease", admitted_boundary)
    expected = ValueError if old else RuntimeError
    with pytest.raises(expected, match="bookkeeping" if old else "lease boundary"):
        demo.run(path, output)


def test_split_domains_and_author_request(config_file):
    path, config, _ = config_file
    for split, seed, worlds in (("train", 9030, 3), ("validation", 9100, 3)):
        config.update(split=split)
        config["sim_eval"].update(
            seed=seed, num_worlds=worlds, randomization=json.dumps(demo.TRAIN_REQUEST)
        )
        path.write_text(json.dumps(config))
        loaded, _ = demo.load_config(path)
        assert loaded["sim_eval"]["randomization"] == json.dumps(
            demo.TRAIN_REQUEST, sort_keys=True
        )
    for split, seed in (("train", 9100), ("train", 20260511), ("validation", 109100)):
        with pytest.raises(ValueError):
            demo.check_seed(split, seed)
    config["sim_eval"]["seed"] = 9101
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Validation"):
        demo.load_config(path)


class SamplingErrorDouble(Exception):
    pass


class NativeDouble:
    def __init__(self):
        self.model = types.SimpleNamespace(
            nu=14, opt=types.SimpleNamespace(timestep=0.002)
        )
        self._control_decimation = 17
        self.data = types.SimpleNamespace(
            qpos=np.zeros(24, dtype=np.float64),
            qvel=np.zeros(22, dtype=np.float64),
            ctrl=np.zeros(14, dtype=np.float64),
            time=0.0,
        )
        self.control = 0

    def get_state(self):
        return self.data.qpos[:14].astype(np.float32)


class SimDouble:
    def __init__(self, *, success_at=16, failures=0, **kwargs):
        self.env = NativeDouble()
        self.randomization = None
        self.success_at, self.failures = success_at, failures
        self.calls, self.issued = [], []
        self.closed = False

    def _images(self):
        return {
            name: np.full((3, 168, 224), self.env.control + 40 * index, dtype=np.uint8)
            for index, name in enumerate(demo.CAMERAS)
        }

    def reset(self, seed, options=None):
        self.calls.append(("reset", seed, self.env.get_state().tolist()))
        if self.failures:
            self.failures -= 1
            raise SamplingErrorDouble("Explicit placement-failure double")
        self.env.control, self.env.data.time = 0, 0.0
        state = {
            "seed": seed,
            "object_states": {},
            "scale_states": {},
            "metadata": {"bottle_count": 5, "bottle_variants": ["bottle_0"] * 5},
        }
        self.randomization = types.SimpleNamespace(to_dict=lambda: state)
        return {
            "state": self.env.get_state(),
            "images": self._images(),
            "prompt": "sim throw plastic bottles in bin",
        }

    def forget_arm_state(self):
        self.calls.append(("forget_arm_state",))
        self.env.data.qpos[:14] = 0.0

    def obs(self):
        self.calls.append(("obs", self.env.control))
        return {
            "state": self.env.get_state(),
            "images": self._images(),
            "prompt": "sim throw plastic bottles in bin",
        }

    def step_one(self, action):
        self.calls.append(("step", self.env.control))
        issued = np.asarray(action, dtype=np.float32)
        self.issued.append(issued.copy())
        self.env.data.ctrl[:] = issued
        self.env.data.ctrl[[6, 13]] = issued[[6, 13]] * 0.0475
        self.env.control += 1
        self.env.data.time += 0.034
        self.env.data.qpos[:14] = self.env.control
        self.env.data.qvel[:] = self.env.control * 2

    def evaluate(self):
        self.calls.append(("evaluate", self.env.control))
        success = self.env.control >= self.success_at
        return {
            "reward": float(success),
            "success": success,
            "ever_success": success,
            "num_bottles_in_bin": 5 if success else 0,
            "max_bottles_in_bin_so_far": 5 if success else 0,
        }

    def render_cameras(self):
        self.calls.append(("render", self.env.control))
        return self._images()

    def close(self):
        self.calls.append(("close",))
        self.closed = True

    step_one_vanilla = step_one
    evaluate_vanilla = evaluate
    obs_vanilla_state = obs
    render_cameras_vanilla_state = render_cameras


class PolicyDouble:
    chunk_length, action_dim = 30, 14

    def __init__(self, *args, **kwargs):
        self.calls = []

    def infer(self, obs, noise=None, action_prefix=None, prefix_length=0):
        self.calls.append((obs, noise, action_prefix, prefix_length))
        return (
            np.arange(420, dtype=np.float32).reshape(30, 14) + len(self.calls) * 1000
        ).astype(np.float32)

    def warmup_rtc(self, obs, noise, prefix_length):
        # Same self-bound dispatch as original warmup; deliberately not recorded.
        return self.infer(
            obs, noise, np.zeros((30, 14), dtype=np.float32), prefix_length
        )


def snapshot_double(native, episode):
    episode.pins["initial_integration"] = episode.budget.array(
        episode.root / "initial_integration.npy", native.data.qpos.copy()
    )
    episode.integration_spec = {"name": "explicit_CPU_double", "value": 0}
    for key, name, content in (
        ("scene_xml", "scene.xml", b"<explicit_cpu_double/>"),
        ("compiled_model", "scene.mjb", b"explicit_CPU_double"),
    ):
        episode.budget.write(episode.root / name, content)
        episode.pins[key] = upstream_eval.artifact(episode.root / name)


def author_rollout_functions():
    """Execute unchanged author function bodies with explicit CPU dependencies.

    AST loading is a test-only boundary: no production sys.path injection,
    model fixture, or native import is used by this qualification.
    """
    repo = Path(os.environ.get("ABC_BENCH_REPO", Path(__file__).resolve().parents[2]))
    source = repo / "abc_minimal" / "eval_policy.py"
    names = {
        "RTCManager",
        "rollout_worlds",
        "video_frame",
        "rollout_over",
        "progress_text",
        "without_missing_counts",
        "_optional_int",
    }
    parsed = ast.parse(source.read_text())
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    nodes += [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names) + 1
    scope = {
        "np": np,
        "time": time,
        "concurrent": concurrent,
        "Path": Path,
        "RandomizationSamplingError": SamplingErrorDouble,
        "CHUNK_COUNT_KEYS": {"bottles", "max_bottles"},
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        ),
    }
    exec(  # noqa: S102 -- unchanged trusted author bodies in explicit CPU doubles
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(source),
            "exec",
        ),
        scope,
    )
    return scope


@pytest.fixture
def doubles(monkeypatch):
    monkeypatch.setattr(demo, "snapshot_reset", snapshot_double)
    video_writers = []
    video_module = types.ModuleType("imageio.v2")

    def get_writer(*args, **kwargs):
        writer = types.SimpleNamespace(pixels=[], closed=False)
        writer.append_data = lambda array: writer.pixels.append(int(array[0, 0, 0]))
        writer.close = lambda: setattr(writer, "closed", True)
        video_writers.append(writer)
        return writer

    video_module.get_writer = get_writer
    imageio_module = types.ModuleType("imageio")
    imageio_module.__path__, imageio_module.v2 = [], video_module
    monkeypatch.setitem(sys.modules, "imageio", imageio_module)
    monkeypatch.setitem(sys.modules, "imageio.v2", video_module)
    return video_writers


def recorded_rollout(
    tmp_path, config, *, worlds=1, failures=0, success_at=16, vanilla=False
):
    output = tmp_path / "out"
    output.mkdir()
    config = copy.deepcopy(config)
    config["sim_eval"].update(num_worlds=worlds)
    collection = demo.Collection(config, output)
    sim_module = types.SimpleNamespace(
        SimTaskEnv=lambda **kwargs: SimDouble(
            success_at=success_at, failures=failures, **kwargs
        )
    )
    eval_module = types.SimpleNamespace(VLAInferencePolicy=PolicyDouble)
    env_factory, policy_factory = sim_module.SimTaskEnv, eval_module.VLAInferencePolicy
    loop = author_rollout_functions()["rollout_worlds"]
    cpu_config = types.SimpleNamespace(**config["sim_eval"])
    cpu_config.fast_inference, cpu_config.num_chunks, cpu_config.vanilla_physics = (
        False,
        2,
        vanilla,
    )
    cpu_config.log_every_chunk = False
    with demo.instrument(sim_module, eval_module, collection):
        policy = eval_module.VLAInferencePolicy()
        env = sim_module.SimTaskEnv()
        worlds_out = loop(
            cpu_config,
            policy,
            env,
            0,
            None,
            output,
            types.SimpleNamespace(
                chunk_length=30, action_dim=14, camera_keys=demo.CAMERAS
            ),
        )
    assert sim_module.SimTaskEnv is env_factory
    assert eval_module.VLAInferencePolicy is policy_factory
    summary = {
        "task": "put_plastic_bottles_in_bin",
        "num_worlds": worlds,
        "worlds": worlds_out,
        "config": config["sim_eval"],
    }
    for world in summary["worlds"]:
        world["randomization"] = world["randomization"].to_dict()
    entries = collection.reconcile(summary)
    return collection, env, policy, summary, entries


@pytest.mark.parametrize("vanilla", [False, True])
def test_unchanged_rollout_parity_rtc_rows_and_raw_rgb(
    config_file, tmp_path, doubles, vanilla
):
    _, config, _ = config_file
    _collection, env, policy, summary, entries = recorded_rollout(
        tmp_path, config, vanilla=vanilla
    )
    manifest = json.loads(Path(entries[0]["manifest"]["path"]).read_text())
    assert entries[0]["admitted"] is True
    assert (manifest["controls"], manifest["frame_count"]) == (16, 17)
    assert all(
        manifest[name] is False
        for name in ("human_expert_approved", "atlas_approved", "dynamics_validated")
    )
    arrays = {
        name: np.load(pin["path"], allow_pickle=False)
        for name, pin in manifest["artifacts"].items()
        if pin["path"].endswith(".npy")
    }
    assert arrays["states"].dtype == np.float32
    assert arrays["issued_actions"].shape == (16, 14)
    assert (
        arrays["qpos"].dtype
        == arrays["sim_times"].dtype
        == arrays["actuator_ctrl"].dtype
        == np.float64
    )
    assert arrays["states"][:, 0].tolist() == list(range(17))
    assert np.array_equal(arrays["issued_actions"], np.asarray(env.delegate.issued))
    assert np.array_equal(
        arrays["actuator_ctrl"][:, [6, 13]],
        arrays["issued_actions"][:, [6, 13]] * 0.0475,
    )
    images = np.fromfile(
        manifest["artifacts"]["rgb24"]["path"], dtype=np.uint8
    ).reshape(17, 168, 672, 3)
    assert images[:, 0, 0, 0].tolist() == list(range(17))
    assert images[:, 0, 224, 0].tolist() == [index + 40 for index in range(17)]
    assert images[:, 0, 448, 0].tolist() == [index + 80 for index in range(17)]
    judges = [
        json.loads(row)
        for row in Path(manifest["artifacts"]["task_judge"]["path"])
        .read_text()
        .splitlines()
    ]
    assert [row["control_index"] for row in judges] == list(range(1, 17))
    assert [row["sim_time"] for row in judges] == arrays["sim_times"][1:].tolist()
    assert manifest["initial_task_eval"]["success"] is False
    assert (
        judges[-1]["task_eval"]
        == manifest["final_task_eval"]
        == summary["worlds"][0]["final_task_eval"]
    )
    requests = [
        json.loads(row)
        for row in Path(manifest["artifacts"]["policy_requests"]["path"])
        .read_text()
        .splitlines()
    ]
    assert (
        len(policy.delegate.calls) == 3
    )  # Initial, self-bound warmup, RTC production.
    assert len(requests) == 2
    assert [row["observation"]["control_index"] for row in requests] == [0, 11]
    assert requests[1]["prefix_length"] == 4
    assert np.array_equal(
        np.asarray(requests[1]["action_prefix"][:4], dtype=np.float32),
        arrays["issued_actions"][11:15],
    )
    provenance = [
        json.loads(row)
        for row in Path(manifest["artifacts"]["action_provenance"]["path"])
        .read_text()
        .splitlines()
    ]
    assert [row["inferred_action_index"] for row in provenance] == list(range(15)) + [4]
    assert [row["inference_id"] for row in provenance] == [1] * 15 + [2]
    assert env.delegate.closed and doubles[0].closed
    assert sum(call[0] == "render" for call in env.delegate.calls) == 16
    assert sum(call[0] == "obs" for call in env.delegate.calls) == 1
    assert sum(call[0] == "evaluate" for call in env.delegate.calls) == 17
    # The same untouched loop without recording executes exactly the same actions.
    plain = SimDouble(success_at=16)
    plain_policy = PolicyDouble()
    cfg = types.SimpleNamespace(**config["sim_eval"])
    cfg.fast_inference, cfg.num_chunks, cfg.log_every_chunk, cfg.vanilla_physics = (
        False,
        2,
        False,
        vanilla,
    )
    author_rollout_functions()["rollout_worlds"](
        cfg,
        plain_policy,
        plain,
        0,
        None,
        tmp_path,
        types.SimpleNamespace(chunk_length=30, action_dim=14, camera_keys=demo.CAMERAS),
    )
    assert np.array_equal(np.asarray(plain.issued), arrays["issued_actions"])
    assert [call[:2] for call in plain.calls] == [
        call[:2] for call in env.delegate.calls
    ]


def test_pending_rtc_output_is_not_executed_after_terminal(
    config_file, tmp_path, doubles
):
    _, config, _ = config_file
    collection, env, policy, _, entries = recorded_rollout(
        tmp_path, config, success_at=12
    )
    episode = collection.episodes[0]
    assert episode.controls == 12 and episode.rgb.count == 13
    assert (
        len(policy.delegate.calls) == 3
    )  # Pending inference completes during original RTC.close.
    provenance = [
        json.loads(row) for row in episode.provenance.path.read_text().splitlines()
    ]
    assert {row["inference_id"] for row in provenance} == {1}
    assert entries[0]["admitted"] and env.delegate.closed


def test_reset_same_seed_retry_and_carried_state_preserved(
    config_file, tmp_path, doubles
):
    _, config, _ = config_file
    config["split"] = "train"
    config["sim_eval"].update(seed=9030, randomization=json.dumps(demo.TRAIN_REQUEST))
    collection, _env, _, summary, entries = recorded_rollout(
        tmp_path, config, worlds=2, failures=1, success_at=1
    )
    assert [episode.seed for episode in collection.episodes] == [9030, 9031]
    first_attempts = collection.episodes[0].reset_attempts
    assert first_attempts[0]["status"] == "failed"
    assert first_attempts[1] == {"operation": "forget_arm_state"}
    assert (
        first_attempts[2]["seed"] == 9030 and first_attempts[2]["status"] == "succeeded"
    )
    second_initial = np.load(
        collection.episodes[1].pins["states"]["path"], allow_pickle=False
    )[0]
    assert (
        second_initial[0] == 1
    )  # Recorder did not add forget_arm_state between successful episodes.
    assert all(entry["admitted"] for entry in entries)
    assert [world["world_seed"] for world in summary["worlds"]] == [9030, 9031]


@pytest.mark.parametrize("split,seed", [("train", 9030), ("validation", 9100)])
def test_original_resampling_preserved_in_train_and_validation(
    config_file, tmp_path, doubles, split, seed
):
    _, config, _ = config_file
    config["split"] = split
    config["sim_eval"].update(seed=seed, randomization=json.dumps(demo.TRAIN_REQUEST))
    collection, _, _, _, _ = recorded_rollout(
        tmp_path, config, failures=2, success_at=1
    )
    assert collection.episodes[0].seed == seed + 100_000
    assert collection.episodes[0].requested_seed == seed
    assert collection.episodes[0].reset_attempts[-1]["seed"] == seed + 100_000
    with pytest.raises(ValueError, match="Validation"):
        demo.check_seed("validation", 109100)
    demo.check_actual_seed("validation", 109100, 9100)
    with pytest.raises(ValueError, match="TRAIN"):
        demo.check_seed("train", 109100)


def test_duplicate_evaluation_does_not_duplicate_judge_rows(
    config_file, tmp_path, doubles
):
    _, config, _ = config_file
    root = tmp_path / "out"
    root.mkdir()
    collection = demo.Collection(config, root)
    env = demo.EnvRecorder(SimDouble(success_at=1), collection)
    obs = env.reset(20260511)
    policy = demo.PolicyRecorder(PolicyDouble(), collection)
    actions = policy.infer(obs)
    env.evaluate()
    env.step_one(actions[0])
    first, duplicate = env.evaluate(), env.evaluate()
    assert first == duplicate
    env.render_cameras()
    env.close()
    assert len(collection.episodes[0].judges) == 1
    rows = collection.episodes[0].pins["task_judge"]["path"]
    assert len(Path(rows).read_text().splitlines()) == 1


def test_stream_cap_failure_closes_resources_and_restores_constructors(
    config_file, tmp_path, doubles
):
    _, config, _ = config_file
    config["max_output_bytes"] = 800_000
    output = tmp_path / "out"
    output.mkdir()
    collection = demo.Collection(config, output)
    sim_module = types.SimpleNamespace(SimTaskEnv=SimDouble)
    eval_module = types.SimpleNamespace(VLAInferencePolicy=PolicyDouble)
    original_env, original_policy = (
        sim_module.SimTaskEnv,
        eval_module.VLAInferencePolicy,
    )
    with (
        pytest.raises(RuntimeError, match="byte cap"),
        demo.instrument(sim_module, eval_module, collection),
    ):
        env = sim_module.SimTaskEnv()
        try:
            obs = env.reset(20260511)
            eval_module.VLAInferencePolicy().infer(obs)
        finally:
            env.close()
    assert (
        sim_module.SimTaskEnv is original_env
        and eval_module.VLAInferencePolicy is original_policy
    )
    assert env.delegate.closed
    episode = collection.episodes[0]
    assert all(
        stream.stream.closed
        for stream in (
            episode.rgb,
            episode.request_rgb,
            episode.requests,
            episode.provenance,
        )
    )
    assert not (episode.root / "manifest.json").exists()
    assert episode.manifest()["admission"]["complete_current_success"] is False
    assert collection.budget.size() <= config["max_output_bytes"]


def test_summary_mismatch_never_publishes_accepted_manifest(
    config_file, tmp_path, doubles
):
    _, config, _ = config_file
    collection, _, _, summary, entries = recorded_rollout(
        tmp_path, config, success_at=1
    )
    path = Path(entries[0]["manifest"]["path"])
    path.unlink()  # Remove only this explicit test fixture publication.
    summary["worlds"][0]["steps"] += 1
    with pytest.raises(RuntimeError, match="differs"):
        collection.reconcile(summary)
    assert not path.exists()
    result = collection.partial_manifests()
    assert result[0]["admitted"] is False
    assert (
        json.loads(path.read_text())["admission"]["complete_current_success"] is False
    )


def test_group_id_and_globally_unique_episode_ids(config_file, tmp_path, doubles):
    _, config, _ = config_file
    collection, _, _, _, entries = recorded_rollout(tmp_path, config, success_at=1)
    manifest = json.loads(Path(entries[0]["manifest"]["path"]).read_text())
    pins = manifest["artifacts"]
    expected = hashlib.sha256(
        (
            pins["scene_xml"]["sha256"] + ":" + pins["initial_integration"]["sha256"]
        ).encode("ascii")
    ).hexdigest()
    assert manifest["reset_group_id"] == expected
    other = demo.Collection(config, tmp_path / "different-output")
    assert other.collection_id != collection.collection_id


def test_native_snapshot_uses_integration_not_f32_capture_state(
    config_file, tmp_path, monkeypatch
):
    _, config, _ = config_file
    output = tmp_path / "out"
    output.mkdir()
    model = types.SimpleNamespace(
        actuator_ctrllimited=np.ones(14, dtype=np.uint8),
        actuator_ctrlrange=np.ones((14, 2), dtype=np.float64),
    )
    native = types.SimpleNamespace(model=model, data=object())
    episode = types.SimpleNamespace(
        root=output,
        budget=demo.OutputBudget(output, config["max_output_bytes"]),
        pins={},
        integration_spec={},
        actuator_limits={},
    )
    calls = []
    module = types.ModuleType("mujoco")
    module.mjtState = types.SimpleNamespace(mjSTATE_INTEGRATION=777)
    module.mj_stateSize = lambda m, spec: 3

    def get_state(m, data, array, spec):
        calls.append(("get_state", spec, array.dtype))
        array[:] = [0.03400000000000001, 1.0000000000000002, 2.0]

    module.mj_getState = get_state
    module.mj_saveLastXML = lambda path, m: Path(path).write_text(
        "<actual_api_CPU_double/>"
    )
    module.mj_sizeModel = lambda m: 8
    module.mj_saveModel = lambda m, path: Path(path).write_bytes(b"CPU_MJB_")
    monkeypatch.setitem(sys.modules, "mujoco", module)
    demo.snapshot_reset(native, episode)
    loaded = np.load(episode.pins["initial_integration"]["path"], allow_pickle=False)
    assert calls == [("get_state", 777, np.dtype("float64"))]
    assert loaded[1] == 1.0000000000000002 and loaded.dtype == np.float64
    assert episode.integration_spec == {"name": "mjSTATE_INTEGRATION", "value": 777}
    assert episode.actuator_limits["explicit_action_clipping"] is False
    assert episode.pins["compiled_model"]["bytes"] == 8


def test_copy_or_mutation_cannot_fake_action_provenance(config_file, tmp_path, doubles):
    _, config, _ = config_file
    output = tmp_path / "out"
    output.mkdir()
    collection = demo.Collection(config, output)
    env = demo.EnvRecorder(SimDouble(), collection)
    obs = env.reset(20260511)
    actions = demo.PolicyRecorder(PolicyDouble(), collection).infer(obs)
    with pytest.raises(RuntimeError, match="does not reference"):
        env.step_one(actions[0].copy())
    actions[0, 0] += 1
    with pytest.raises(RuntimeError, match="row changed"):
        env.step_one(actions[0])
    assert env.env.control == 0 and not env.delegate.issued
    env.close()


def test_mutated_observation_rejected_before_sampler(config_file, tmp_path, doubles):
    _, config, _ = config_file
    output = tmp_path / "out"
    output.mkdir()
    collection = demo.Collection(config, output)
    env = demo.EnvRecorder(SimDouble(), collection)
    obs = env.reset(20260511)
    delegate = PolicyDouble()
    obs["images"]["top"][0, 0, 0] = 77
    with pytest.raises(RuntimeError, match="observation changed"):
        demo.PolicyRecorder(delegate, collection).infer(obs)
    assert delegate.calls == []
    env.close()


def test_environment_initialization_failure_closes_delegate(config_file, tmp_path):
    _, config, _ = config_file
    output = tmp_path / "out"
    output.mkdir()
    collection = demo.Collection(config, output)
    delegate = SimDouble()
    delegate.env.model.opt.timestep = 0.001
    env_factory = lambda: delegate
    sim_module = types.SimpleNamespace(SimTaskEnv=env_factory)
    eval_module = types.SimpleNamespace(VLAInferencePolicy=PolicyDouble)
    with (
        pytest.raises(ValueError, match="physics"),
        demo.instrument(sim_module, eval_module, collection),
    ):
        sim_module.SimTaskEnv()
    assert delegate.closed and sim_module.SimTaskEnv is env_factory
