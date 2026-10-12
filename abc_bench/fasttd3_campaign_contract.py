"""Declared reward-v2 task campaign bounds; historical profiles are separate."""

from __future__ import annotations

PHASES = {"runtime_gate": 64, "training": 42480}
PROFILE = "reward_v2_training128"
THRESHOLDS = {
    "success_rate": 1 / 128,
    "final_bottles": 0.1,
    "peak_bottles": 0.1,
    "peak_phi": 0.005,
    "peak_reach": 0.02,
    "peak_clearance": 0.02,
    "peak_lift_transport": 0.02,
}


def campaign_spec(phase: str) -> dict:
    if phase not in PHASES:
        raise ValueError("Use the separately named runtime_gate or training phase")
    return {
        "schema": "sadhana.fasttd3.abc.task-futility/v1",
        "phase": phase,
        "maximum_calls": PHASES[phase],
        "minimum_calls": 21240 if phase == "training" else None,
        "minimum_completed_episodes_per_rank": 6 if phase == "training" else None,
        "window_calls": 3540,
        "patience_windows": 3,
        "improvement_thresholds": THRESHOLDS,
        "rank_weighting": "equal_mean_of_per_rank_completed_episode_means",
        "stop_scope": "training_task_futility_not_validation_convergence_or_overfit",
        "validation_seeds_used_for_stopping": [],
        "loss_based_stopping": False,
        "checkpoint": "unchanged_author_final_export_after_natural_loop_exit",
    }
