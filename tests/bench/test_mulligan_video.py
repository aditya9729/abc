"""Installed CPU tests: real author evaluation/normalization, explicit world/policy doubles.

These MP4s show synthetic pixels. They are not native evaluation evidence.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from abc_bench import mulligan_video
from mulligan.sim import envs, vec_env
from mulligan.sim.eval import grid_eval
from mulligan.training.normalization import Normalizer


@pytest.fixture
def author_fixture(tmp_path, monkeypatch):
    made = []
    factory_calls = []
    faults = {"render": None, "close": False}

    class World:
        """State-bearing stand-in; no MuJoCo, native world or model is constructed."""

        def __init__(self, render, success_step):
            self.render_enabled = render
            self.success_step = success_step
            self.closed = False
            self.steps = 0
            self.rng = np.random.default_rng(0)
            self.nuts = [SimpleNamespace(joints=["nut"])]
            self.sim = SimpleNamespace(
                data=SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), time=0.0),
                model=SimpleNamespace(joint_name2id=lambda _: 0, jnt_qposadr=[0]),
                forward=lambda: None,
                get_state=lambda: SimpleNamespace(
                    qpos=self.sim.data.qpos.copy(),
                    qvel=self.sim.data.qvel.copy(),
                    time=self.sim.data.time,
                ),
            )

        def _get_observations(self, **_kwargs):
            value = self.sim.data.qpos[0]
            return {
                "robot0_eef_pos": np.full(3, value),
                "robot0_eef_quat": np.zeros(4),
                "robot0_gripper_qpos": np.zeros(2),
                "object-state": np.zeros(14),
            }

        def reset(self):
            self.steps = 0
            self.sim.data.qpos[:] = 0
            self.sim.data.qvel[:] = 0
            self.sim.data.time = 0.0
            return self._get_observations()

        def step(self, action):
            self.steps += 1
            self.sim.data.qvel[:] = action
            self.sim.data.qpos[:] += action
            self.sim.data.time += 0.05
            success = self.success_step is not None and self.steps >= self.success_step
            return self._get_observations(), float(success), False, {"success": success}

        def render(self):
            assert self.render_enabled
            if faults["render"] == "state":
                self.sim.data.qpos[0] += 1
            elif faults["render"] == "rng":
                np.random.random()
            elif faults["render"] == "shape":
                return np.zeros((2, 2, 3), dtype=np.uint8)
            return np.full((240, 320, 3), self.steps * 30, dtype=np.uint8)

        def close(self):
            self.closed = True
            if faults["close"] and self.steps:
                # Fail the explicit author close once. The author's inherited
                # destructor may retry; this fixture is not a permanent leak.
                faults["close"] = False
                raise RuntimeError("Explicit CPU world close failure")

    def create(**kwargs):
        factory_calls.append(kwargs)
        world = World(kwargs["use_render_wrapper"], 2 if len(made) == 1 else None)
        made.append(world)
        return world

    monkeypatch.setattr(envs, "create_robosuite_env", create)
    monkeypatch.setattr(
        envs,
        "seed_env",
        lambda env, seed: setattr(env, "rng", np.random.default_rng(seed)),
    )

    # This follows the author's own CPU test seam. Policy and checkpoint loader
    # are explicitly doubled; the original outer/rollout loops and Normalizer run.
    policy_module = types.ModuleType("mulligan.agents.idql")

    class Policy:
        config = SimpleNamespace(num_action_samples=32)

        def eval(self):
            return self

        def reset(self):
            pass

        def select_action(self, batch):
            return torch.rand((batch["observation.state"].shape[0], 7)) * 2 - 1

    policy_module.IDQLPolicy = Policy
    monkeypatch.setitem(sys.modules, policy_module.__name__, policy_module)
    normalizer = Normalizer(
        torch.zeros(23), torch.ones(23), torch.full((7,), -2.0), torch.full((7,), 2.0)
    )
    loader = types.ModuleType("mulligan.utils.load_pretrained")
    loader.load_policy_from_checkpoint = lambda **_kwargs: (
        Policy(),
        normalizer,
        normalizer,
    )
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setattr(grid_eval, "resolve_checkpoint", lambda _: tmp_path)
    manifest = grid_eval.make_equal_tile_manifest(points_per_cell=1, seed=7)
    manifest_path = tmp_path / "explicit-cpu-test-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    evaluation = dict(
        artifact_path="explicit-cpu-policy-double",
        manifest_path=manifest_path,
        output_dir=tmp_path / "recorded",
        max_steps=4,
        num_envs=2,
        device="cpu",
        use_sync_envs=True,
        num_action_samples=32,
        eval_seed=11,
        env_seed=12,
    )
    return evaluation, made, factory_calls, faults


def receipt(evaluation):
    return json.loads((evaluation["output_dir"] / "videos/receipt.json").read_text())


def test_original_evaluator_complete_video_and_render_off_parity(author_fixture):
    evaluation, made, calls, _ = author_fixture
    original = vec_env.SyncVectorEnv
    factory = envs.create_robosuite_env
    result = mulligan_video.evaluate_with_video(video_points=[0, 1, 2], **evaluation)
    assert vec_env.SyncVectorEnv is original
    assert envs.create_robosuite_env is factory
    assert all(world.closed for world in made)
    assert calls == [
        dict(
            env_name="NutAssemblySquare",
            robot_name="Panda",
            camera_names=None,
            visual_aids=False,
            use_render_wrapper=render,
        )
        for render in [False, True, True]
    ]
    recorded = receipt(evaluation)
    assert recorded["status"] == "completed"
    assert recorded["author_evaluation_completed"] is True
    assert [e["observed"]["success"] for e in recorded["episodes"]] == [1, 0, 1]
    assert [e["observed"]["length"] for e in recorded["episodes"]] == [2, 4, 2]
    # A finished worker continues to receive author steps until its batch ends;
    # its video ends at the author's own first-success boundary, not later calls.
    for episode in recorded["episodes"]:
        assert episode["author_outcome"]["point_idx"] == episode["point"]["point_idx"]
        assert episode["author_outcome"]["length"] == episode["observed"]["length"]
        reader = imageio.get_reader(episode["video"]["path"])
        try:
            assert reader.count_frames() == episode["observed"]["length"] + 1
            assert reader.get_data(0).shape == (240, 320, 3)
        finally:
            reader.close()
    rows = json.loads(Path(result["point_results_path"]).read_text())
    assert len(rows) == 80

    made.clear()
    calls.clear()
    evaluation["output_dir"] = evaluation["output_dir"].parent / "trace-only"
    second = mulligan_video.evaluate_with_video(
        video_points=[0, 1, 2], record_video=False, **evaluation
    )
    unrendered = receipt(evaluation)
    assert rows == json.loads(Path(second["point_results_path"]).read_text())
    assert all(call["use_render_wrapper"] is False for call in calls)
    for rendered, plain in zip(
        recorded["episodes"], unrendered["episodes"], strict=True
    ):
        assert rendered["trace"]["sha256"] == plain["trace"]["sha256"]
        assert plain["video"] is None and plain["frames"] == 0


@pytest.mark.parametrize("fault", ["state", "rng", "shape"])
def test_render_fault_rejects_and_restores_author_class(author_fixture, fault):
    evaluation, made, _, faults = author_fixture
    faults["render"] = fault
    original = vec_env.SyncVectorEnv
    with pytest.raises(ValueError, match="Rendering changed|agentview render"):
        mulligan_video.evaluate_with_video(video_points=[0], **evaluation)
    assert vec_env.SyncVectorEnv is original
    assert all(world.closed for world in made)
    assert receipt(evaluation)["status"] == "failed"


def test_cleanup_failure_preserves_complete_author_rows_but_rejects_job(author_fixture):
    evaluation, _, _, faults = author_fixture
    faults["close"] = True
    with pytest.raises(RuntimeError, match="world close failure"):
        mulligan_video.evaluate_with_video(video_points=[0], **evaluation)
    observed = receipt(evaluation)
    assert observed["status"] == "failed"
    assert observed["author_evaluation_completed"] is False
    assert observed["episodes"][0]["episode_complete"] is True
    assert (evaluation["output_dir"] / "point_results.json").exists()


def test_encoder_failure_closes_writer_and_rejects(author_fixture, monkeypatch):
    evaluation, made, _, _ = author_fixture
    writers = []

    class BadEncoder:
        closed = False

        def append_data(self, _frame):
            raise OSError("Explicit CPU encoder failure")

        def close(self):
            self.closed = True

    def writer(_path):
        obj = BadEncoder()
        writers.append(obj)
        return obj

    monkeypatch.setattr(mulligan_video, "_writer", writer)
    with pytest.raises(OSError, match="encoder failure"):
        mulligan_video.evaluate_with_video(video_points=[0], **evaluation)
    assert writers and all(obj.closed for obj in writers)
    assert all(world.closed for world in made)
    assert receipt(evaluation)["status"] == "failed"


def test_author_row_disagreement_cannot_accept_video(author_fixture, monkeypatch):
    evaluation, _, _, _ = author_fixture
    original = grid_eval._point_result_from_rollout

    def mismatch(*args, **kwargs):
        row = original(*args, **kwargs)
        return {**row, "length": row["length"] + 1}

    monkeypatch.setattr(grid_eval, "_point_result_from_rollout", mismatch)
    with pytest.raises(ValueError, match="authoritative author outcome"):
        mulligan_video.evaluate_with_video(video_points=[0], **evaluation)
    assert receipt(evaluation)["status"] == "failed"


@pytest.mark.parametrize(
    "change,selection",
    [
        ({"use_sync_envs": False}, [0]),
        ({"eval_seed": None}, [0]),
        ({"env_seed": None}, [0]),
        ({"shard_idx": 1}, [0]),
        ({}, []),
        ({}, [0, 0]),
        ({}, [0, 1, 2, 3]),
        ({}, [999]),
    ],
)
def test_invalid_observer_contract_never_constructs_world(
    author_fixture, change, selection
):
    evaluation, made, _, _ = author_fixture
    with pytest.raises(ValueError):
        mulligan_video.evaluate_with_video(
            video_points=selection, **{**evaluation, **change}
        )
    assert made == []


def test_existing_output_never_overwritten(author_fixture):
    evaluation, made, _, _ = author_fixture
    output = evaluation["output_dir"]
    output.mkdir()
    marker = output / "historical-result.json"
    marker.write_text("immutable history")
    with pytest.raises(ValueError, match="fresh"):
        mulligan_video.evaluate_with_video(video_points=[0], **evaluation)
    assert marker.read_text() == "immutable history" and made == []


def test_recording_cannot_fall_back_to_gpu_renderer(author_fixture, monkeypatch):
    evaluation, made, _, _ = author_fixture
    monkeypatch.setenv("MUJOCO_GL", "egl")
    with pytest.raises(ValueError, match="CPU MUJOCO_GL=osmesa"):
        mulligan_video.evaluate_with_video(video_points=[0], **evaluation)
    assert made == []
