"""Explicit estimated laundry props around the public robot pose replay.

No cloth dynamics, policy execution, contact simulation, or task-success claim.
The published dryer frame is interpreted as a front-plane frame; this convention
is inferred from measured reaches and is not documented by the author.
"""

from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .assets import sha256


def numbers(values) -> str:
    return " ".join(f"{float(v):.10g}" for v in values)


def geom(parent, name, kind, pos=(0, 0, 0), size=None, rgba=(0.6, 0.6, 0.6, 1), **attrs):
    data = {
        "name": name,
        "type": kind,
        "pos": numbers(pos),
        "rgba": numbers(rgba),
        "contype": "0",
        "conaffinity": "0",
        "density": "0",
    }
    if size is not None:
        data["size"] = numbers(size)
    data.update(attrs)
    return ET.SubElement(parent, "geom", **data)


def bar(parent, name, start, end, radius=0.012, color=(0.28, 0.29, 0.28, 1)):
    result = geom(
        parent, name, "capsule", size=(radius,), rgba=color, fromto=numbers([*start, *end])
    )
    result.attrib.pop("pos")
    return result


def towel(parent, name, center, half_size=(0.18, 0.13, 0.025)):
    geom(parent, name, "box", center, half_size, (0.93, 0.92, 0.87, 1))
    x, y, z = center
    geom(
        parent,
        name + "_seam",
        "box",
        (x, y - half_size[1] - 0.001, z),
        (half_size[0] * 0.9, 0.001, 0.003),
        (0.68, 0.71, 0.67, 1),
    )


def door_geometry(body, name, width, color):
    """A rigid estimated circular door, with its local origin at the hinge."""
    radius = 0.285
    center_y = -width / 2
    for segment in range(36):
        a, b = segment * math.tau / 36, (segment + 1) * math.tau / 36
        start = (0.015, center_y + radius * math.cos(a), radius * math.sin(a))
        end = (0.015, center_y + radius * math.cos(b), radius * math.sin(b))
        bar(body, f"{name}_rim_{segment}", start, end, 0.023, color)
    geom(
        body,
        f"{name}_glass",
        "cylinder",
        (0.01, center_y, 0),
        (0.262, 0.009),
        (0.08, 0.11, 0.12, 0.55),
        euler=numbers([0, math.pi / 2, 0]),
    )
    bar(
        body,
        f"{name}_handle",
        (0.065, center_y - 0.245, -0.10),
        (0.065, center_y - 0.245, 0.10),
        0.019,
        (0.11, 0.12, 0.12, 1),
    )


def appliance(world, name, position, yaw, width, door_z, body_color, *, moving_door):
    """Front is local +x; cabinet extends into local -x. All dimensions estimated."""
    body = ET.SubElement(
        world, "body", name=name, pos=numbers([*position, 0]), euler=numbers([0, 0, yaw])
    )
    depth, bottom, top = 0.79, 0.35, 1.46
    geom(
        body,
        name + "_pedestal",
        "box",
        (-depth / 2, 0, bottom / 2),
        (depth / 2, width / 2, bottom / 2),
        tuple(v * 0.85 for v in body_color[:3]) + (1,),
    )
    for sign in (-1, 1):
        geom(
            body,
            f"{name}_side_{sign}",
            "box",
            (-depth / 2, sign * (width / 2 - 0.025), (top + bottom) / 2),
            (depth / 2, 0.025, (top - bottom) / 2),
            body_color,
        )
    geom(
        body,
        name + "_top",
        "box",
        (-depth / 2, 0, top - 0.025),
        (depth / 2, width / 2, 0.025),
        body_color,
    )
    geom(
        body,
        name + "_back",
        "box",
        (-depth + 0.025, 0, (top + bottom) / 2),
        (0.025, width / 2, (top - bottom) / 2),
        body_color,
    )
    # Front panels surround the dark drum. The front is not a solid blocking box.
    radius = 0.29
    upper_bottom = door_z + radius
    lower_top = door_z - radius
    geom(
        body,
        name + "_front_upper",
        "box",
        (-0.022, 0, (top + upper_bottom) / 2),
        (0.022, width / 2, (top - upper_bottom) / 2),
        body_color,
    )
    geom(
        body,
        name + "_front_lower",
        "box",
        (-0.022, 0, (bottom + lower_top) / 2),
        (0.022, width / 2, (lower_top - bottom) / 2),
        body_color,
    )
    for sign in (-1, 1):
        geom(
            body,
            f"{name}_front_edge_{sign}",
            "box",
            (-0.022, sign * (radius + width / 2) / 2, door_z),
            (0.022, (width / 2 - radius) / 2, radius),
            body_color,
        )
    geom(
        body,
        name + "_drum_back",
        "cylinder",
        (-0.52, 0, door_z),
        (0.28, 0.012),
        (0.035, 0.045, 0.045, 1),
        euler=numbers([0, math.pi / 2, 0]),
    )
    for segment in range(40):
        angle = segment * math.tau / 40
        bar(
            body,
            f"{name}_drum_{segment}",
            (-0.52, 0.279 * math.cos(angle), door_z + 0.279 * math.sin(angle)),
            (-0.03, 0.279 * math.cos(angle), door_z + 0.279 * math.sin(angle)),
            0.018,
            (0.17, 0.18, 0.18, 1),
        )
    geom(
        body,
        name + "_screen",
        "box",
        (0.004, -0.16, 1.37),
        (0.008, 0.095, 0.043),
        (0.03, 0.06, 0.07, 1),
    )
    geom(
        body,
        name + "_knob",
        "cylinder",
        (0.022, 0.12, 1.37),
        (0.045, 0.024),
        (0.18, 0.19, 0.20, 1),
        euler=numbers([0, math.pi / 2, 0]),
    )
    if moving_door:
        door = ET.SubElement(world, "body", name="scene_dryer_door", mocap="true")
    else:
        door = ET.SubElement(
            body, "body", name="scene_washer_door", pos=numbers([0, width / 2, door_z])
        )
    door_geometry(door, name + "_door", width, body_color)


def door_angle(timestamp: float, scene: dict) -> float:
    """Piecewise-linear interpolation of the author's open/close schedule."""
    door = scene["door"]
    opened = np.clip((timestamp - door["t_open"][0]) / np.diff(door["t_open"])[0], 0, 1)
    closed = np.clip((timestamp - door["t_close"][0]) / np.diff(door["t_close"])[0], 0, 1)
    return math.radians(door["angle_deg"]) * float(opened - closed)


def dryer_door_pose(timestamp: float, scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """World position and wxyz quaternion of the inferred +y hinge frame."""
    dryer, width = scene["dryer"], scene["width"]
    yaw = dryer["yaw"]
    position = np.array(
        [
            dryer["x"] - math.sin(yaw) * width / 2,
            dryer["y"] + math.cos(yaw) * width / 2,
            scene["door_z"],
        ]
    )
    angle = yaw + door_angle(timestamp, scene)
    return position, np.array([math.cos(angle / 2), 0, 0, math.sin(angle / 2)])


def prepare_laundry(original: Path, research: Path, output: Path) -> dict:
    original, research, output = original.resolve(), research.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    preparation = json.loads((original / "preparation.json").read_text())
    if sha256(original / "taku_visual_replay.xml") != preparation["model_sha256"]:
        raise ValueError("Original prepared model changed")
    if sha256(original / "measured_trajectory.npz") != preparation["trajectory_sha256"]:
        raise ValueError("Original prepared trajectory changed")
    if sha256(research / "tracking_data.json") != preparation["input_files"]["tracking_data.json"]:
        raise ValueError("Raw tracking metadata does not match the original prepared recording")
    shutil.copytree(original, output)
    scene = json.loads((research / "tracking_data.json").read_text())["scene"]
    tree = ET.parse(output / "taku_visual_replay.xml")
    model = tree.getroot()
    model.set("model", "Taku measured replay in estimated laundry room — no dynamics")
    model.find("visual/global").set("offwidth", "960")
    model.find("visual/global").set("offheight", "640")
    ET.SubElement(
        model.find("visual"),
        "headlight",
        ambient="0.3 0.3 0.3",
        diffuse="0.55 0.55 0.55",
        specular="0.1 0.1 0.1",
    )
    asset, world = model.find("asset"), model.find("worldbody")
    ET.SubElement(
        asset,
        "texture",
        name="room_sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.72 0.73 0.70",
        rgb2="0.93 0.91 0.85",
        width="128",
        height="768",
    )
    floor = world.find("geom[@name='reference_grid']")
    floor.attrib.pop("material")
    floor.set("rgba", "0.64 0.59 0.47 1")
    floor.set("size", "3 3 0.01")
    # Wood planks and two rear walls are presentation estimates from the reference poster.
    for index in range(24):
        x = -2.9 + index * 0.25
        color = (0.65 + 0.035 * (index % 3), 0.59 + 0.025 * (index % 3), 0.47, 1)
        geom(world, f"floor_board_{index}", "box", (x, 0, -0.005), (0.123, 3, 0.008), color)
    geom(
        world,
        "room_appliance_wall",
        "box",
        (2.22, 0, 1.45),
        (0.035, 3, 1.45),
        (0.69, 0.67, 0.61, 1),
    )
    geom(
        world,
        "room_back_wall",
        "box",
        (-0.35, 2.2, 1.45),
        (2.6, 0.035, 1.45),
        (0.75, 0.72, 0.65, 1),
    )
    dryer = scene["dryer"]
    width, gap, yaw = scene["width"], scene["gap"], dryer["yaw"]
    appliance(
        world,
        "dryer",
        [dryer["x"], dryer["y"]],
        yaw,
        width,
        scene["door_z"],
        (0.27, 0.29, 0.30, 1),
        moving_door=True,
    )
    washer_position = [
        dryer["x"] + math.sin(yaw) * (width + gap),
        dryer["y"] - math.cos(yaw) * (width + gap),
    ]
    appliance(
        world,
        "washer",
        washer_position,
        yaw,
        width,
        scene["door_z"],
        (0.89, 0.90, 0.88, 1),
        moving_door=False,
    )
    table = ET.SubElement(world, "body", name="folding_table", pos="-1.25 0.95 0")
    geom(table, "table_top", "box", (0, 0, 0.79), (0.75, 0.375, 0.035), (0.58, 0.39, 0.20, 1))
    for i, (x, y) in enumerate([(-0.64, -0.29), (-0.64, 0.29), (0.64, -0.29), (0.64, 0.29)]):
        bar(table, f"table_leg_{i}", (x, y, 0.04), (x, y, 0.76), 0.032)
    for level in range(3):
        towel(table, f"table_towel_{level}", (-0.15, 0, 0.85 + 0.045 * level))
    basket = ET.SubElement(world, "body", name="rolling_wire_basket", pos="-0.25 -1.9 0")
    basket_color = (0.48, 0.52, 0.53, 1)
    for z in (0.27, 0.75):
        corners = [(-0.38, -0.26, z), (0.38, -0.26, z), (0.38, 0.26, z), (-0.38, 0.26, z)]
        for i in range(4):
            bar(
                basket, f"basket_rim_{z}_{i}", corners[i], corners[(i + 1) % 4], 0.012, basket_color
            )
    for i, (x, y) in enumerate([(-0.34, -0.22), (-0.34, 0.22), (0.34, -0.22), (0.34, 0.22)]):
        bar(basket, f"basket_post_{i}", (x, y, 0.07), (x, y, 0.75), 0.014, basket_color)
        geom(
            basket,
            f"basket_wheel_{i}",
            "cylinder",
            (x, y, 0.065),
            (0.06, 0.021),
            (0.09, 0.10, 0.10, 1),
            euler=numbers([math.pi / 2, 0, 0]),
        )
    for level in range(1, 8):
        z = 0.27 + level * 0.06
        for side in (-1, 1):
            bar(
                basket,
                f"basket_wire_long_{level}_{side}",
                (-0.38, side * 0.26, z),
                (0.38, side * 0.26, z),
                0.003,
                basket_color,
            )
            bar(
                basket,
                f"basket_wire_short_{level}_{side}",
                (side * 0.38, -0.26, z),
                (side * 0.38, 0.26, z),
                0.003,
                basket_color,
            )
    for i in range(11):
        x = -0.36 + 0.072 * i
        for side in (-1, 1):
            bar(
                basket,
                f"basket_vertical_{i}_{side}",
                (x, side * 0.26, 0.27),
                (x, side * 0.26, 0.75),
                0.003,
                basket_color,
            )
        bar(basket, f"basket_floor_{i}", (x, -0.26, 0.27), (x, 0.26, 0.27), 0.003, basket_color)
    for layer in range(5):
        towel(basket, f"basket_towel_{layer}", (0, 0, 0.30 + layer * 0.06), (0.29, 0.20, 0.03))
    shelves = ET.SubElement(world, "body", name="left_wall_shelves", pos="-2.2 0.9 0")
    for index, z in enumerate((0.15, 0.67, 1.19, 1.71, 2.23)):
        geom(
            shelves, f"shelf_{index}", "box", (0, 0, z), (0.24, 0.66, 0.025), (0.72, 0.72, 0.66, 1)
        )
        if index < 4:
            for column in (-1, 1):
                for layer in range(3):
                    towel(
                        shelves,
                        f"shelf_towel_{index}_{column}_{layer}",
                        (0, column * 0.30, z + 0.053 + layer * 0.052),
                        (0.18, 0.23, 0.026),
                    )
    for i, (x, y) in enumerate([(-0.22, -0.64), (-0.22, 0.64), (0.22, -0.64), (0.22, 0.64)]):
        bar(shelves, f"shelf_post_{i}", (x, y, 0), (x, y, 2.26), 0.018, (0.74, 0.74, 0.70, 1))
    design = {
        "schema_version": 1,
        "mode": "estimated laundry scene with measured robot motion",
        "reference": "https://www.dyna.co/assets/we/workflow-video-poster.jpg",
        "source_scene_metadata": scene,
        "original_model_and_motion_unchanged": True,
        "dryer_frame_interpretation": "inferred front-plane origin; local +x faces the robot; +y hinge",
        "door_interpolation": "estimated piecewise-linear interpolation between published event endpoints",
        "object_frames_world_m": {
            "dryer_front": [dryer["x"], dryer["y"], 0],
            "washer_front": [*washer_position, 0],
            "folding_table": [-1.25, 0.95, 0],
            "rolling_wire_basket": [-0.25, -1.9, 0],
            "left_wall_shelves": [-2.2, 0.9, 0],
        },
        "estimated_dimensions_m": {
            "appliance_depth": 0.79,
            "appliance_top_height": 1.46,
            "pedestal_height": 0.35,
            "door_radius": 0.285,
            "table": [1.5, 0.75, 0.825],
            "basket": [0.76, 0.52, 0.75],
            "shelves": [0.48, 1.32, 2.26],
        },
        "camera": {"azimuth": 45, "elevation": -22, "distance": 6.6, "lookat": [0, 0, 0.9]},
        "fidelity_labels": [
            "Published Taku model + 12 Hz measured log",
            "Estimated room and laundry props",
            "Static towels; scripted dryer door",
            "No policy, physics stepping, or task-success measurement",
        ],
        "target_task_success": None,
    }
    (output / "scene_design.json").write_text(json.dumps(design, indent=2) + "\n")
    ET.indent(model)
    tree.write(output / "taku_visual_replay.xml", encoding="unicode")
    preparation["scene_design_sha256"] = sha256(output / "scene_design.json")
    preparation["model_sha256"] = sha256(output / "taku_visual_replay.xml")
    preparation["scene"] = design["mode"]
    preparation["source_dryer_metadata_used"] = True
    preparation["original_preparation_sha256"] = sha256(original / "preparation.json")
    (output / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    return design
