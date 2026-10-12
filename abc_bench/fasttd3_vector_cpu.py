"""Finite two-world native acceptance; no cameras, CUDA, learner or task score."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

from abc_bench.fasttd3_host import write_json
from abc_bench.fasttd3_vector_host import VectorPool, load_config, validate_cleanup
from abc_bench.upstream_eval import artifact


def run(config: dict, output: Path, *, pool_factory=VectorPool) -> dict:
    if config["profile"] != "cpu_acceptance2":
        raise ValueError("CPU acceptance requires the explicit two-world profile")
    if output.exists():
        raise ValueError("CPU acceptance output must be new")
    output.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "phase": "abc_fasttd3_vector_native_cpu_acceptance",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": config,
        "qualification_scope": "Two native CPU worlds; no learner or performance claim",
        "environment": {
            key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL")
        },
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "mujoco", "gymnasium", "abc-minimal")
        },
    }
    pool, error = None, None
    before = time.monotonic()
    try:
        import numpy as np

        pool = pool_factory(config, output)
        description = pool.describe()
        receipt["description"] = description
        initial = pool.reset()
        scenes = [
            json.loads((output / f"world-{rank:03d}/reset-0000-scene.json").read_text())
            for rank in range(2)
        ]
        if initial["requested_seeds"] != [200000, 200001]:
            raise ValueError("Native initial rank seeds differ")
        if scenes[0]["model_arrays"]["sha256"] == scenes[1]["model_arrays"]["sha256"]:
            raise ValueError("Native worlds have identical model randomization")
        if (
            scenes[0]["initial_integration"]["sha256"]
            == scenes[1]["initial_integration"]["sha256"]
        ):
            raise ValueError("Native worlds have identical initial integration state")
        receipt["initial_scenes"] = scenes
        schema = description["action_schema"]
        low, high = np.asarray(schema["policy_low"]), np.asarray(schema["policy_high"])
        actions = [
            np.clip(
                2 * (np.asarray(scene["native_policy_state"]) - low) / (high - low) - 1,
                -1,
                1,
            )
            .astype(np.float32)
            .tolist()
            for scene in scenes
        ]
        ordinary = pool.step(actions)
        if ordinary["terminated"] != [False, False] or ordinary["truncated"] != [
            False,
            False,
        ]:
            raise ValueError("Native ordinary step unexpectedly ended an episode")
        expected = (
            low
            + (np.clip(np.asarray(actions, dtype=np.float32), -1, 1) + 1)
            * 0.5
            * (high - low)
        ).astype(np.float32)
        if not np.array_equal(
            np.asarray(ordinary["physical_actions"], dtype=np.float32), expected
        ):
            raise ValueError("Native normalized-to-physical action mapping differs")
        for rank in range(2):
            ctrl = np.asarray(ordinary["native_ctrl"][rank])[schema["ctrl_indices"]]
            expected_ctrl = expected[rank].astype(np.float64)
            # Original environment multiplies NumPy float32 grip by .0475;
            # NumPy version determines scalar promotion. Check actual boundary.
            for grip in (6, 13):
                expected_ctrl[grip] = np.float32(expected[rank, grip]) * 0.0475
            if not np.array_equal(ctrl, expected_ctrl):
                raise ValueError(
                    "Original actuator mapping differs from issued physical14"
                )
        receipt["ordinary_step"] = ordinary
        receipt["explicit_native_fixtures"] = pool.cpu_fixtures()
        final = pool.step(actions)
        receipt["terminal_timeout_step"] = final
        if final["terminated"] != [True, False] or final["truncated"] != [False, True]:
            raise ValueError(
                "Native success termination and timeout qualification did not pass"
            )
        if (
            final["rewards"][0] != 1.0
            or final["episodes"][0]["final_native_task_eval"]["num_bottles_in_bin"] != 5
        ):
            raise ValueError(
                "Native success fixture did not use the exact original five-bottle judge"
            )
        raw = np.asarray(final["terminal_obs"], dtype=np.float32)
        reset = np.asarray(final["obs"], dtype=np.float32)
        if (
            raw.shape != (2, 216)
            or reset.shape != (2, 216)
            or np.array_equal(raw, reset)
        ):
            raise ValueError("Native final observation was lost during autoreset")
        if [row["requested_seed"] for row in final["autoresets"]] != [200128, 200129]:
            raise ValueError(
                "Per-world autoreset did not advance the independent seed bank"
            )
        carry = []
        joints, grips = [i for i in range(14) if i not in (6, 13)], [6, 13]
        for rank, reset_row in enumerate(final["autoresets"]):
            terminal_policy = np.asarray(
                final["terminal_native_policy_states"][rank], dtype=np.float32
            )
            reset_policy = np.asarray(
                reset_row["native_policy_state"], dtype=np.float32
            )
            differences = np.abs(terminal_policy.astype(np.float64) - reset_policy)
            carry.append(
                {
                    "rank": rank,
                    "absolute_difference": differences.tolist(),
                    "same_seed_pose_preserved": len(reset_row["attempts"]) == 1,
                    "joint_comparison": "exact_float32",
                    "grip_fraction_absolute_tolerance": 2
                    * float(np.finfo(np.float32).eps),
                }
            )
            if len(reset_row["attempts"]) == 1 and (
                not np.array_equal(terminal_policy[joints], reset_policy[joints])
                or np.any(differences[grips] > 2 * np.finfo(np.float32).eps)
            ):
                raise ValueError(
                    "Native ordinary autoreset did not preserve the carried arm policy state"
                )
        # Original fraction->float32 metre ctrl->float64 qpos->fraction may
        # round the two grip fractions by one ULP under the locked NumPy2 host.
        # Joints must remain exact; the measured grip errors stay in the receipt.
        receipt["carried_arm_acceptance"] = carry
        receipt["original_fast_bootstrap_mask"] = [False, True]
        receipt["status"] = "completed"
    except BaseException as failure:  # noqa: BLE001 - record interruption/error and propagate after cleanup
        error = failure
        receipt.update(status="failed", error=f"{type(failure).__name__}: {failure}")
    finally:
        try:
            if pool is not None:
                pool.close()
                validate_cleanup(pool.worker_cleanup, 2)
        except BaseException as failure:  # noqa: BLE001 - failed cleanup must reject completion
            error = failure
            receipt.update(
                status="failed", cleanup_error=f"{type(failure).__name__}: {failure}"
            )
        if pool is not None:
            receipt.update(
                worker_cleanup=pool.worker_cleanup,
                worker_resources=getattr(pool, "worker_resources", []),
                vector_steps=pool.vector_steps,
                transitions=pool.vector_steps * 2,
            )
        if (output / "startup-failure.json").is_file():
            receipt["startup_failure"] = artifact(output / "startup-failure.json")
        receipt["seconds"] = time.monotonic() - before
        receipt["parent_maxrss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        receipt["children_maxrss_kib"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        write_json(output / "receipt.json", receipt)
    if error is not None:
        raise error
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(load_config(options.config), options.output)


if __name__ == "__main__":
    main()
