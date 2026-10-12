"""Versioned ABC task rewards; the native judge and FastTD3 learner stay separate.

The potential reader runs only forward kinematics on its own MjData. Importing
this module does not import MuJoCo, start physics, or initialize CUDA.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

SPARSE_PROFILE = "abc_full_success_sparse_v1"
PBRS_PROFILE = "abc_full_success_pbrs_v1"
GAMMA = 0.99
POTENTIAL_BOUND = 0.25
BOTTLE_NAMES = tuple(f"bottle_{i}" for i in range(1, 6))
GRASP_SITES = ("left_grasp_site", "right_grasp_site")


def reward_spec(profile: str) -> dict[str, Any]:
    """Return the exact, versioned task adaptation; no learner overrides."""
    if profile not in (SPARSE_PROFILE, PBRS_PROFILE):
        raise ValueError("Unknown ABC success reward profile")
    return {
        "schema": "sadhana.abc.success-reward/v1",
        "profile": profile,
        "base": "one_on_current_native_five_of_five_success_termination_else_zero",
        "gamma": GAMMA,
        "potential_bound": POTENTIAL_BOUND if profile == PBRS_PROFILE else 0.0,
        "true_terminal_potential": 0.0,
        "timeout_potential": "retain_native_final_before_autoreset",
        "reward_clipping": False,
        "geometry_frame": "world",
        "geometry_units": "metres",
        "reach_length_m": 0.15,
        "clearance_margin_m": 0.01,
        "clearance_length_m": 0.10,
        "transport_length_m": 0.25,
        "bin_opening_offset_m": 0.05,
        "unsolved_weights": [0.25, 0.25, 0.25],
        "grasp_sites": list(GRASP_SITES),
        "bottle_names": list(BOTTLE_NAMES),
        "native_judge_changed": False,
        "grasp_or_contact_success_label": False,
        "parameter_origin": "disclosed_initial_geometric_heuristics_no_VAL_tuning",
    }


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def reward_config_sha256(profile: str) -> str:
    return _json_sha256(reward_spec(profile))


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric{shape}")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    return array


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be finite numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite numeric")
    return number


def _judge(judge: Mapping[str, Any]) -> tuple[np.ndarray, float, bool]:
    if not isinstance(judge, Mapping):
        raise TypeError("Native task evaluation must be a mapping")
    active = judge.get("num_active_bottles")
    count = judge.get("num_bottles_in_bin")
    if type(active) is not int or active != 5 or type(count) is not int:
        raise ValueError("Native judge must declare exactly five active bottles")
    raw_mask = np.asarray(judge.get("bottle_in_bin_mask"))
    if raw_mask.shape != (5,) or raw_mask.dtype.kind != "b":
        raise ValueError("Native current bottle mask must be bool[5]")
    if count != int(raw_mask.sum()):
        raise ValueError("Native current count and mask disagree")
    success = judge.get("success")
    if type(success) is not bool or success != (count == 5):
        raise ValueError("Native current success and bottle count disagree")
    fraction = _scalar(judge.get("reward"), "native reward")
    if not np.isclose(fraction, count / 5, atol=1e-7, rtol=0):
        raise ValueError("Native reward is not the current bottle fraction")
    mask = np.array(raw_mask, dtype=bool, copy=True)
    mask.flags.writeable = False
    return mask, fraction, success


@dataclass(frozen=True)
class GeometrySnapshot:
    """Detached native geometry at one qpos, not a grasp/contact annotation."""

    bottle_centers: np.ndarray
    grasp_positions: np.ndarray
    bin_opening: np.ndarray
    table_point: np.ndarray
    table_normal: np.ndarray
    bottle_reaches: np.ndarray
    in_bin: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("bottle_centers", (5, 3)),
            ("grasp_positions", (2, 3)),
            ("bin_opening", (3,)),
            ("table_point", (3,)),
            ("table_normal", (3,)),
            ("bottle_reaches", (5,)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if not np.isclose(np.linalg.norm(self.table_normal), 1, atol=1e-8, rtol=0):
            raise ValueError("Table normal must be a unit vector")
        if (self.bottle_reaches <= 0).any():
            raise ValueError("Native bottle collision bounds must be positive")
        mask = np.asarray(self.in_bin)
        if mask.shape != (5,) or mask.dtype.kind != "b":
            raise ValueError("Current in-bin mask must be bool[5]")
        mask = mask.copy()
        mask.flags.writeable = False
        object.__setattr__(self, "in_bin", mask)


def potential(snapshot: GeometrySnapshot) -> dict[str, Any]:
    """Bounded, nonnegative progress potential; no history or stage bonuses."""
    distances = np.linalg.norm(
        snapshot.bottle_centers[:, None, :] - snapshot.grasp_positions[None, :, :],
        axis=2,
    ).min(axis=1)
    reach = np.exp(-distances / 0.15)
    center_height = (
        snapshot.bottle_centers - snapshot.table_point
    ) @ snapshot.table_normal
    clearance = center_height - snapshot.bottle_reaches - 0.01
    lift = np.clip(clearance / 0.10, 0, 1)
    transport = np.exp(
        -np.linalg.norm(snapshot.bottle_centers - snapshot.bin_opening, axis=1) / 0.25
    )
    unsolved = 0.25 * reach + 0.25 * lift + 0.25 * lift * transport
    progress = np.where(snapshot.in_bin, 1.0, unsolved)
    value = POTENTIAL_BOUND * float(progress.mean())
    return {
        "phi": value,
        "reach": reach.tolist(),
        "clearance_lower_bound_m": clearance.tolist(),
        "lift_clearance": lift.tolist(),
        "transport": transport.tolist(),
        "current_in_bin": snapshot.in_bin.tolist(),
        "bottle_progress": progress.tolist(),
    }


def transition_reward(
    profile: str,
    native_task_eval: Mapping[str, Any],
    *,
    terminated: bool,
    truncated: bool,
    phi_pre: float | None = None,
    phi_final: float | None = None,
) -> dict[str, Any]:
    """Calculate reward on the native final world, before any autoreset.

    A timeout retains its final potential because the original critic
    bootstraps there. A true terminal transition has zero final potential.
    """
    reward_spec(profile)
    _, fraction, success = _judge(native_task_eval)
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("Termination flags must be booleans")
    if terminated and truncated or terminated != success:
        raise ValueError("Flags must match native success termination semantics")
    base = float(success)
    if profile == SPARSE_PROFILE:
        if phi_pre is not None or phi_final is not None:
            raise ValueError("Sparse reference must not consume a potential")
        previous = final = used = 0.0
    else:
        previous = _scalar(phi_pre, "previous potential")
        final = _scalar(phi_final, "native final potential")
        if not 0 <= previous <= POTENTIAL_BOUND or not 0 <= final <= POTENTIAL_BOUND:
            raise ValueError("Potential exceeds its versioned bound")
        used = 0.0 if terminated else final
    shaping = GAMMA * used - previous
    return {
        "profile": profile,
        "base_reward": base,
        "shaping_reward": shaping,
        "learner_reward": base + shaping,
        "phi_pre": previous,
        "phi_final_actual": final,
        "phi_final_used": used,
        "native_fraction": fraction,
        "native_success": success,
        "terminated": terminated,
        "truncated": truncated,
    }


class BottlePotentialGeometry:
    """One owned FK scratch for one actual randomized native model.

    Recreate after a model swap. The caller supplies the public original
    task-evaluation metadata; this reader never calls the judge or simulator.
    """

    def __init__(self, model: Any, task_eval: Mapping[str, Any]):
        import mujoco

        _judge(task_eval)
        if tuple(task_eval.get("bottle_names", ())) != BOTTLE_NAMES:
            raise ValueError("Native ordered bottle names differ from the profile")
        self._offsets = _array(
            task_eval.get("bottle_center_offsets"), (5, 3), "COM offsets"
        )
        self._reaches = _array(
            task_eval.get("bottle_reaches"), (5,), "collision bounds"
        )
        if (self._reaches <= 0).any():
            raise ValueError("Native bottle collision bounds must be positive")
        self._bin_top = _scalar(task_eval.get("bin_top_y"), "bin top y")
        self._bin_bottom = _scalar(task_eval.get("bin_bottom_y"), "bin bottom y")
        if self._bin_top <= self._bin_bottom:
            raise ValueError("Native bin height extent is invalid")
        self._bodies = [int(model.body(name).id) for name in BOTTLE_NAMES]
        self._bin = int(model.body("bin_container").id)
        self._sites = [int(model.site(name).id) for name in GRASP_SITES]
        self._table = int(model.geom("table_plane").id)
        if min(self._bodies + [self._bin, self._table] + self._sites) < 0:
            raise ValueError("Required native geometry name is missing")
        if int(model.geom_type[self._table]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            raise ValueError("Native table geometry must be a plane")
        gravity = _array(model.opt.gravity, (3,), "gravity")
        if np.linalg.norm(gravity) == 0:
            raise ValueError("Gravity must define the vertical axis")
        if int(model.nmocap) != 0:
            raise ValueError("The qpos-only FK profile does not support mocap bodies")
        self._up = -gravity / np.linalg.norm(gravity)
        self._mujoco, self._model = mujoco, model
        self._scratch = mujoco.MjData(model)

    def contract(self, profile: str, *, native_model_sha256: str) -> dict[str, Any]:
        """Bind caller-authenticated native-model arrays, geometry and config.

        Obtain native_model_sha256 from the existing public
        fasttd3_vector_host.native_model_digest(actual_model). It fingerprints
        all public model arrays, not an MJB file or deterministic replay proof.
        """
        if self._scratch is None:
            raise RuntimeError("Potential geometry reader is closed")
        if profile != PBRS_PROFILE:
            raise ValueError("Only the shaped profile consumes a geometry contract")
        if type(native_model_sha256) is not str or not re.fullmatch(
            "[0-9a-f]{64}", native_model_sha256
        ):
            raise ValueError(
                "Native model-array SHA256 must be authenticated by the caller"
            )
        geometry = {
            "bottle_names": list(BOTTLE_NAMES),
            "body_ids": self._bodies,
            "grasp_sites": list(GRASP_SITES),
            "grasp_site_ids": self._sites,
            "bin_body": "bin_container",
            "bin_body_id": self._bin,
            "table_geom": "table_plane",
            "table_geom_id": self._table,
            "bottle_center_offsets": self._offsets.tolist(),
            "bottle_reaches": self._reaches.tolist(),
            "bin_bottom_y": self._bin_bottom,
            "bin_top_y": self._bin_top,
            "gravity_up": self._up.tolist(),
            "fk": "public_mj_kinematics_on_owned_qpos_copy_MjData",
        }
        result = {
            "schema": "sadhana.abc.success-reward-geometry/v1",
            "native_model_array_sha256": native_model_sha256,
            "geometry": geometry,
            "geometry_sha256": _json_sha256(geometry),
            "config_sha256": reward_config_sha256(profile),
        }
        result["contract_sha256"] = _json_sha256(result)
        return result

    def snapshot(
        self, model: Any, native_data: Any, task_eval: Mapping[str, Any]
    ) -> GeometrySnapshot:
        if self._scratch is None:
            raise RuntimeError("Potential geometry reader is closed")
        if model is not self._model:
            raise ValueError("Native model changed; rebind the FK reader")
        mask, _, _ = _judge(task_eval)
        if (
            tuple(task_eval.get("bottle_names", ())) != BOTTLE_NAMES
            or not np.array_equal(task_eval.get("bottle_center_offsets"), self._offsets)
            or not np.array_equal(task_eval.get("bottle_reaches"), self._reaches)
            or _scalar(task_eval.get("bin_top_y"), "bin top y") != self._bin_top
            or _scalar(task_eval.get("bin_bottom_y"), "bin bottom y")
            != self._bin_bottom
        ):
            raise ValueError("Native model metadata changed; rebind the FK reader")
        qpos = _array(native_data.qpos, (int(self._model.nq),), "native qpos")
        self._scratch.qpos[:] = qpos
        self._mujoco.mj_kinematics(self._model, self._scratch)
        body_rotation = np.asarray(self._scratch.xmat).reshape(-1, 3, 3)
        centers = self._scratch.xpos[self._bodies] + np.einsum(
            "bij,bj->bi", body_rotation[self._bodies], self._offsets
        )
        opening = self._scratch.xpos[self._bin] + body_rotation[self._bin] @ np.array(
            [0.0, self._bin_top + 0.05, 0.0]
        )
        table_rotation = np.asarray(self._scratch.geom_xmat).reshape(-1, 3, 3)
        normal = table_rotation[self._table, :, 2]
        if float(normal @ self._up) < 1 - 1e-8:
            raise ValueError("Native table plane must align with the gravity vertical")
        return GeometrySnapshot(
            bottle_centers=centers,
            grasp_positions=self._scratch.site_xpos[self._sites],
            bin_opening=opening,
            table_point=self._scratch.geom_xpos[self._table],
            table_normal=normal,
            bottle_reaches=self._reaches,
            in_bin=mask,
        )

    def close(self) -> None:
        self._scratch = None
