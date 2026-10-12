"""Original sequential ABC TRAIN reset profile for ordinary EXPO.

Native imports occur only inside the lease-guarded host's calls. The reset
sequence matches abc_minimal.eval_policy.run_sequential, including its
conditional carried-arm clearing and mapped-seed placement retry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from abc_bench.fasttd3_host import bounded_int, finite_json, write_json
from abc_bench.upstream_eval import artifact

PROFILE = "original_five_variant_scale_armcarry_v1"
ACTION_PROFILE = "original_float32_no_host_clip_v1"
PROOF_SCHEMA = "sadhana.abc.original-train-reset/v1"
OPTIONS = {
    "randomization": {
        "bottle_count": 5,
        "randomize_variants": True,
        "randomize_scales": True,
    }
}
RESERVED = frozenset({20260511, *range(9100, 9103), *range(109100, 109103)})


def check_seed(seed: Any) -> int:
    bounded_int(seed, "TRAIN requested seed", 9030, 2**31 - 100001)
    if seed in RESERVED or seed + 100000 in RESERVED:
        raise ValueError("TRAIN cannot use a held-out base or mapped seed family")
    return seed


def validate_config(config: dict) -> dict:
    if config["stage"] != "train":
        raise ValueError("Schema3 is the original ordinary EXPO TRAIN profile")
    host = config["host"]
    if type(host) is not dict or set(host) != {
        "num_envs",
        "initial_seed",
        "allowed_seeds",
        "max_episode_steps",
        "save_video",
        "reset_profile",
        "action_profile",
    }:
        raise ValueError("Schema3 requires the explicit original TRAIN host fields")
    if host["reset_profile"] != PROFILE or host["action_profile"] != ACTION_PROFILE:
        raise ValueError("Original TRAIN reset/raw action profiles differ")
    bounded_int(host["num_envs"], "num_envs", 1, 1)
    bounded_int(host["max_episode_steps"], "max_episode_steps", 3540, 3540)
    if type(host["save_video"]) is not bool:
        raise ValueError("save_video must be a boolean")
    seeds = host["allowed_seeds"]
    if type(seeds) is not list or not 1 <= len(seeds) <= 64:
        raise ValueError("Require1..64 explicit TRAIN requested seeds")
    for seed in seeds:
        check_seed(seed)
    if len(set(seeds)) != len(seeds) or check_seed(host["initial_seed"]) not in seeds:
        raise ValueError("TRAIN seed bank must be unique and include initial_seed")
    times = config["timeouts"]
    if type(times) is not dict or set(times) != {
        "host_wall_s",
        "rpc_s",
        "max_requests",
    }:
        raise ValueError("Require finite TRAIN host/RPC/request bounds")
    bounded_int(times["host_wall_s"], "host_wall_s", 1, 43200)
    bounded_int(times["rpc_s"], "rpc_s", 1, 300)
    bounded_int(times["max_requests"], "max_requests", 1, 200000)
    return config


def reset_world(env: Any, requested: int, *, sampling_error=None) -> tuple:
    """Preserve arm carry; clear/retry only on the original placement error."""
    check_seed(requested)
    if sampling_error is None:
        from abc_sim.randomization.core import RandomizationSamplingError

        sampling_error = RandomizationSamplingError
    attempts = []

    def attempt(seed):
        record = {"operation": "reset", "seed": seed, "options": OPTIONS}
        attempts.append(record)
        try:
            result = env.reset(seed=seed, options=OPTIONS, randomize=True)
        except sampling_error:
            record["placement_error"] = True
            raise
        record["placement_error"] = False
        return result

    actual = requested
    try:
        raw, info = attempt(actual)
    except sampling_error:
        attempts.append({"operation": "forget_arm_state"})
        env.forget_arm_state()
        try:
            raw, info = attempt(actual)
        except sampling_error:
            actual = requested + 100000
            raw, info = attempt(actual)
    state = info["randomization"].to_dict()
    finite_json(state)
    if (
        state.get("seed") != actual
        or state.get("metadata", {}).get("bottle_count") != 5
    ):
        raise RuntimeError("Original TRAIN reset changed seed or active bottle count")
    return raw, state, actual, attempts


def record_initial(env: Any, directory: Path, *, randomization: dict) -> dict:
    """Record actual full native reset inputs; this is no trajectory guarantee."""
    import mujoco
    import numpy as np

    directory.mkdir()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, state_spec), np.float64)
    mujoco.mj_getState(env.model, env.data, state, state_spec)
    records = {}
    for name, value in (
        ("integration", state),
        ("qpos", np.asarray(env.data.qpos, np.float64)),
        ("qvel", np.asarray(env.data.qvel, np.float64)),
    ):
        path = directory / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        records[name] = {
            **artifact(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    path = directory / "scene.mjb"
    mujoco.mj_saveModel(env.model, str(path))
    records["compiled_model"] = artifact(path)
    path = directory / "randomization.json"
    write_json(path, randomization)
    records["randomization"] = artifact(path)
    return {
        "artifacts": records,
        "integration_spec": {"name": "mjSTATE_INTEGRATION", "value": int(state_spec)},
        "model": {"nq": env.model.nq, "nv": env.model.nv, "nu": env.model.nu},
        "native_time_s": float(env.data.time),
        "software": {"mujoco": mujoco.__version__},
        "limits": [
            "Full reset state is recorded; future trajectories and live hardware are not certified"
        ],
    }
