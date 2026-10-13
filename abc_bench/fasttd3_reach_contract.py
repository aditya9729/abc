"""TRAIN-only reaching diagnostic; never a five-bottle task reward."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PROFILE = "reach_diagnostic64_v1"
HORIZON = 128
PHASES = {"runtime_gate": 256, "training": 2048}
REACH = {
    "schema": "sadhana.abc.reach-diagnostic/v1",
    "reward": "exp(-min_2_grasp_sites_by_5_bottle_COM_distance_m/0.15)",
    "scale_m": 0.15,
    "threshold_m": 0.12,
    "hold_controls": 5,
    "initial_eligibility_m": 0.15,
    "horizon_controls": HORIZON,
    "episode_end": "fixed_horizon_timeout_only",
    "native_five_bottle_judge": "measured_separately_unchanged",
    "comparison": "matched_initial_final_deterministic_TRAIN_actor_comparison_pending",
}
STOPPING = {
    "schema": "sadhana.abc.reach-diagnostic-stopping/v1",
    "window_calls": 256,
    "minimum_calls": 512,
    "patience_windows": 3,
    "qualified_reach_rate_improvement": 0.05,
    "distance_gain_improvement_m": 0.01,
    "eligibility": "each_rank_has_completed_episode_with_initial_distance_gt0.15_m",
    "scope": "TRAIN_progress_plateau_only_no_learning_or_convergence_claim",
    "validation_seeds": [],
}


def validate_config(value: Any) -> dict:
    fields = {
        "schema_version",
        "profile",
        "phase",
        "num_envs",
        "seed_base",
        "seed_stride",
        "max_episode_steps",
        "vector_steps",
        "reach",
        "stopping",
        "selection_evidence",
        "timeouts",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("Use the separate complete reach diagnostic schema1")
    for key, expected in {
        "schema_version": 1,
        "profile": PROFILE,
        "num_envs": 64,
        "seed_base": 200000,
        "seed_stride": 128,
        "max_episode_steps": HORIZON,
    }.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError("Reach diagnostic contract differs: " + key)
    phase = value["phase"]
    if type(phase) is not str or phase not in PHASES:
        raise ValueError("Use the separate runtime_gate or training phase")
    if type(value["vector_steps"]) is not int or value["vector_steps"] != PHASES[phase]:
        raise ValueError("Reach diagnostic call bound differs")
    if value["reach"] != REACH or value["stopping"] != STOPPING:
        raise ValueError("Reach reward, criterion or stopping specification differs")
    evidence = value["selection_evidence"]
    if type(evidence) is not dict or set(evidence) != {"path", "bytes", "sha256"}:
        raise ValueError("Pin the TRAIN-only selection evidence")
    limits = value["timeouts"]
    maxima = {
        "ready_s": 600,
        "step_s": 180,
        "shutdown_s": 30,
        "wall_s": 3540,
        "rpc_s": 600,
    }
    if type(limits) is not dict or set(limits) != set(maxima):
        raise ValueError("Use every finite reach host watchdog")
    if any(
        type(limits[key]) is not int or not 1 <= limits[key] <= maximum
        for key, maximum in maxima.items()
    ):
        raise ValueError("Reach host watchdog exceeds its bound")
    if (
        limits["step_s"] > limits["rpc_s"]
        or limits["ready_s"] + limits["shutdown_s"] >= limits["wall_s"]
    ):
        raise ValueError("Reach host watchdogs do not fit")
    return value


def nearest_distance(bottle_centers: Any, grasp_positions: Any) -> float:
    """Return metres between nearest of five COMs and two grasp sites, world XYZ."""
    centers, sites = np.asarray(bottle_centers), np.asarray(grasp_positions)
    if centers.shape != (5, 3) or sites.shape != (2, 3):
        raise ValueError("Reach geometry requires five COMs and two XYZ grasp sites")
    if centers.dtype.kind not in "fiu" or sites.dtype.kind not in "fiu":
        raise ValueError("Reach geometry must be numeric")
    if not np.isfinite(centers).all() or not np.isfinite(sites).all():
        raise FloatingPointError("Reach geometry is not finite")
    return float(np.linalg.norm(centers[:, None, :] - sites[None, :, :], axis=-1).min())


class ReachEpisode:
    """Accumulate a fixed-length diagnostic episode without native judge labels."""

    def __init__(self, initial_distance: float):
        self.initial_distance = self.checked(initial_distance)
        self.minimum_distance = self.initial_distance
        self.last_distance = self.initial_distance
        self.eligible = initial_distance > REACH["initial_eligibility_m"]
        self.hold = self.maximum_hold = self.controls = 0
        self.achieved = False
        self.reward_sum = 0.0

    @staticmethod
    def checked(distance: float) -> float:
        if (
            type(distance) not in (int, float)
            or not math.isfinite(distance)
            or distance < 0
        ):
            raise ValueError("Distance must be finite nonnegative metres")
        return float(distance)

    def step(self, distance: float) -> float:
        distance = self.checked(distance)
        if self.controls >= HORIZON:
            raise RuntimeError("Reach episode exceeded its fixed128-control horizon")
        self.controls += 1
        self.last_distance = distance
        self.minimum_distance = min(self.minimum_distance, distance)
        self.hold = self.hold + 1 if distance <= REACH["threshold_m"] else 0
        self.maximum_hold = max(self.maximum_hold, self.hold)
        self.achieved = self.achieved or self.hold >= REACH["hold_controls"]
        reward = math.exp(-distance / REACH["scale_m"])
        self.reward_sum += reward
        return reward

    def evidence(self) -> dict:
        return {
            "initial_distance_m": self.initial_distance,
            "minimum_distance_m": self.minimum_distance,
            "final_distance_m": self.last_distance,
            "distance_gain_m": max(0.0, self.initial_distance - self.minimum_distance),
            "initially_inside_criterion": self.initial_distance <= REACH["threshold_m"],
            "eligible_initial_distance": self.eligible,
            "reach_achieved": self.achieved,
            "qualified_reach_achieved": self.eligible and self.achieved,
            "maximum_hold_controls": self.maximum_hold,
            "reward_sum": self.reward_sum,
            "controls": self.controls,
        }
