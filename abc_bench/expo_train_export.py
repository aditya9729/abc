"""Export authenticated completed TRAIN recordings with the original serializer."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from abc_bench.expo_train_collection import (
    EPISODE_SCHEMA,
    PROMPT,
    RESULT_SCHEMA,
    ROOT,
    authenticate,
    load_descriptor,
    pin,
    read,
    require,
)

MANIFEST_RESERVE = 4_000_000


def frame(item):
    import numpy as np

    with np.load(authenticate(item), allow_pickle=False) as archive:
        require(
            set(archive.files) == {"rgb", "integration", "metadata_utf8"},
            "Frame fields differ",
        )
        rgb, integration, encoded = (
            archive["rgb"],
            archive["integration"],
            archive["metadata_utf8"],
        )
        require(rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8, "Invalid RGB")
        require(
            integration.shape == (437,)
            and integration.dtype == np.float64
            and np.isfinite(integration).all(),
            "Invalid integration state",
        )
        require(
            encoded.ndim == 1 and encoded.dtype == np.uint8 and encoded.size <= 500_000,
            "Invalid frame metadata",
        )
        metadata = json.loads(encoded.tobytes())
        require(
            integration[0] == metadata["public"]["time_s"], "Integration time differs"
        )
        return rgb, metadata


def transition(previous, following, rgb):
    """Pre-action observation t, actually executed command t, following label t+1."""
    import numpy as np

    require(
        previous["schema"] == following["schema"] == EPISODE_SCHEMA,
        "Frame schema differs",
    )
    require(
        previous["seed"] == following["seed"]
        and following["episode_control"] == previous["episode_control"] + 1,
        "Episode/frame boundary differs",
    )
    require(
        following["kind"] == "policy_control"
        and following["is_hil"] is False
        and following["teacher_data"] is False,
        "Autonomous policy control required",
    )
    require(
        following["state_before"] == previous["public"]["state14"]
        and following["time_before_s"] == previous["public"]["time_s"],
        "Causal pre-action state differs",
    )
    require(
        abs(following["public"]["time_s"] - previous["public"]["time_s"] - 1 / 15)
        < 1e-8,
        "Control clock differs",
    )
    info = following["native_info"]
    require(type(info["success"]) is bool, "Boolean success label required")
    terminated, truncated = info["terminated"], info["truncated"]
    require(
        type(terminated) is bool
        and type(truncated) is bool
        and not (terminated and truncated),
        "Invalid termination/truncation",
    )
    labels = following["done_success_reward_mask"]
    require(
        labels
        == [
            terminated or truncated,
            info["success"],
            float(info["success"]),
            float(not terminated),
        ],
        "Historical reward/mask differs",
    )
    require(not info["success"] or terminated, "Success must terminate")
    state = np.asarray(previous["public"]["state14"], dtype=np.float32)
    action = np.asarray(following["applied"], dtype=np.float64)
    require(
        state.shape == action.shape == (14,)
        and np.isfinite(state).all()
        and np.isfinite(action).all(),
        "Finite physical14 arrays required",
    )
    require(
        np.array_equal(action, np.asarray(info["action_applied"], dtype=np.float64)),
        "Applied action differs",
    )
    return {
        "observations": {"state": state, "head_left": rgb, "prompt": PROMPT},
        "actions": action,
        "rewards": labels[2],
        "masks": labels[3],
        "dones": labels[0],
        "is_hil": False,
        "is_success": False,
    }


def collection_rows(descriptor, result_item):
    result = read(result_item)
    require(
        result["schema"] == RESULT_SCHEMA
        and result["status"] == "completed_candidate"
        and result["error"] is None,
        "Complete collection required",
    )
    require(
        result["split"] == "TRAIN"
        and result["prompt"] == PROMPT
        and result["lineage"] == descriptor["lineage"],
        "Collection provenance differs",
    )
    require(
        len(result["episodes"]) == len(descriptor["seeds"])
        and result["resets"] == len(descriptor["seeds"]) + 1,
        "Episode schedule differs",
    )
    root = Path(descriptor["output_dir"])
    require(
        Path(result_item["path"]) == root / "RESULT.json",
        "Collection result path differs",
    )
    total = 0
    for index, (seed, item) in enumerate(zip(descriptor["seeds"], result["episodes"])):
        episode_root = root / "episodes" / f"{index:04d}"
        require(
            Path(item["path"]) == episode_root / "RESULT.json",
            "Episode result path differs",
        )
        episode = read(item)
        count = episode["controls"]
        require(
            episode["schema"] == EPISODE_SCHEMA
            and episode["seed"] == seed
            and episode["complete"] is True
            and episode["error"] is None
            and episode["last_unfinished_proposal"] is None,
            "Incomplete or different episode",
        )
        require(
            type(count) is int
            and 1 <= count <= 900
            and len(episode["files"]) == count + 1,
            "T commands require T+1 frames",
        )
        previous = None
        for control, frame_item in enumerate(episode["files"]):
            require(
                Path(frame_item["path"])
                == episode_root / "frames" / f"{control:04d}.npz",
                "Frame path/order differs",
            )
            rgb, metadata = frame(frame_item)
            require(
                metadata["seed"] == seed
                and metadata["episode_control"]
                == metadata["public"]["index"]
                == control,
                "Frame identity differs",
            )
            if control == 0:
                require(
                    metadata["kind"] == "reset"
                    and metadata["schema"] == EPISODE_SCHEMA
                    and metadata["is_hil"] is False
                    and metadata["teacher_data"] is False,
                    "Reset boundary required",
                )
            else:
                row = transition(previous[1], metadata, previous[0])
                require(
                    row["dones"] is (control == count),
                    "Episode must end at its only terminal",
                )
                if control == count:
                    require(
                        metadata["done_success_reward_mask"] == episode["labels"],
                        "Final labels differ",
                    )
                total += 1
                yield seed, control, row
            previous = rgb, metadata
    require(total == result["actual_controls"], "Cohort control count differs")


def original_serializer(item):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("JAX_PLATFORMS") == "cpu",
        "Export must run explicitly CPU-only",
    )
    path = authenticate(item)
    # Match train_pi_robo's public import order. Importing data first enters an
    # upstream agents/data circular import; no upstream module is modified.
    save = importlib.import_module("expo_ft.agents").save_replay_buffer_transition
    require(
        Path(save.__code__.co_filename).resolve() == path.resolve(),
        "Installed original serializer differs",
    )
    return save


def export(
    descriptor_path, result_item, serializer_item, output, max_bytes, *, save=None
):
    """Partial exports remain rejected; only a complete manifest admits replay."""
    import pickle

    import cloudpickle

    descriptor = load_descriptor(descriptor_path)
    require(
        read(result_item)["descriptor"] == pin(descriptor_path),
        "Collection descriptor differs",
    )
    save = save or original_serializer(serializer_item)
    output = Path(output)
    require(
        output.is_absolute()
        and output.resolve().is_relative_to(ROOT)
        and len(str(output.resolve())) <= 512
        and not output.resolve().is_relative_to(
            Path(descriptor["output_dir"]).resolve()
        ),
        "Export must remain under aditya",
    )
    require(
        type(max_bytes) is int and MANIFEST_RESERVE < max_bytes <= 5 * 1024**3,
        "Declare bounded export bytes",
    )
    output.mkdir(parents=True, exist_ok=False)
    files, episodes = [], []
    used, status, error = 0, "incomplete_or_failed", None
    try:
        for step, (seed, control, row) in enumerate(
            collection_rows(descriptor, result_item), start=1
        ):
            # Admission only: the actual write still calls the original function.
            expected = cloudpickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)
            require(
                used + len(expected) + MANIFEST_RESERVE <= max_bytes,
                "Export byte cap exhausted before original write",
            )
            save(output, row, step=step)
            target = output / "buffers" / f"{step:012d}.pkl"
            require(
                target.read_bytes() == expected, "Original serializer output differs"
            )
            files.append(pin(target))
            used += len(expected)
            if control == 1:
                episodes.append({"seed": seed, "first_step": step})
            if row["dones"]:
                episodes[-1].update(
                    last_step=step,
                    controls=control,
                    reward=row["rewards"],
                    mask=row["masks"],
                )
        status = "completed_candidate_replay"
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"[:2000]
        raise
    finally:
        result = {
            "schema": "sadhana.expo_train_replay_export/1",
            "status": status,
            "artifact_accepted": False,
            "error": error,
            "descriptor": pin(descriptor_path),
            "collection_result": result_item,
            "original_serializer": serializer_item,
            "is_hil": False,
            "success_marks": "Original BatchProcessor calls PiReplayBuffer.restore_success_marks on restore",
            "episodes": episodes,
            "files": files,
            "bytes_before_manifest": used,
            "max_bytes": max_bytes,
            "scope": "Historical sparse labels; no strict recovery or new scene lineage",
        }
        encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        require(
            len(encoded) <= MANIFEST_RESERVE, "Replay manifest exceeds reserved bytes"
        )
        with (output / "RESULT.json").open("xb") as stream:
            stream.write(encoded)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--result-pin", required=True, type=Path)
    parser.add_argument("--serializer-pin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-bytes", required=True, type=int)
    args = parser.parse_args()
    export(
        args.descriptor,
        json.loads(args.result_pin.read_text()),
        json.loads(args.serializer_pin.read_text()),
        args.output,
        args.max_bytes,
    )


if __name__ == "__main__":
    main()
