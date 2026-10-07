"""Embodiment boundary regressions, independent of local robot assets."""

from pathlib import Path

import pytest

from abc_bench.embodiments import inspect_urdf, validate_yam_model


def test_unknown_package_does_not_become_simulation_ready(tmp_path: Path) -> None:
    asset = tmp_path / "robot.urdf"
    asset.write_text(
        '<robot name="r1"><joint name="left_arm_joint1" type="revolute"/>'
        '<link name="base"><visual><geometry><mesh filename="package://r1/meshes/a.stl"/>'
        "</geometry></visual></link></robot>"
    )
    receipt = inspect_urdf(asset)
    assert receipt.left_arm_joints == ("left_arm_joint1",)
    assert receipt.unresolved_meshes == ("package://r1/meshes/a.stl",)
    assert receipt.status == "requires_adapter"
    package = tmp_path / "package"
    (package / "meshes").mkdir(parents=True)
    (package / "meshes/a.stl").write_text("fixture")
    resolved = inspect_urdf(asset, package_roots={"r1": package})
    assert not resolved.unresolved_meshes
    assert not resolved.missing_meshes
    assert resolved.status == "requires_adapter"
    assert resolved.sha256 == receipt.sha256


def test_missing_relative_mesh_is_reported(tmp_path: Path) -> None:
    asset = tmp_path / "robot.urdf"
    asset.write_text(
        '<robot name="r"><link name="base"><visual><geometry>'
        '<mesh filename="missing.stl"/></geometry></visual></link></robot>'
    )
    assert inspect_urdf(asset).missing_meshes == ("missing.stl",)


def test_duplicate_joint_names_rejected(tmp_path: Path) -> None:
    asset = tmp_path / "robot.urdf"
    asset.write_text(
        '<robot><joint name="same" type="fixed"/>'
        '<joint name="same" type="revolute"/></robot>'
    )
    with pytest.raises(ValueError, match="unique"):
        inspect_urdf(asset)


def _model(*, wrong_transmission: bool = False):
    mujoco = pytest.importorskip("mujoco")
    bodies = []
    actuators = []
    for side in ("left", "right"):
        for index in range(1, 7):
            joint = f"{side}_joint{index}"
            bodies.append(f'<body><joint name="{joint}"/><geom size="0.01"/></body>')
            target = (
                "right_joint1"
                if wrong_transmission and joint == "left_joint1"
                else joint
            )
            actuators.append(f'<position name="{joint}" joint="{target}"/>')
        bodies.append(
            f'<body><joint name="{side}_left_finger" type="slide"/>'
            '<geom size="0.01"/></body>'
        )
        actuators.append(
            f'<position name="{side}_gripper" joint="{side}_left_finger"/>'
        )
    cameras = "".join(
        f'<camera name="{name}" pos="0 0 1"/>' for name in ("top", "left", "right")
    )
    return mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        + "".join(bodies)
        + cameras
        + "</worldbody><actuator>"
        + "".join(actuators)
        + "</actuator></mujoco>"
    )


def test_valid_native_bindings_pass() -> None:
    validate_yam_model(_model())


def test_wrong_actuator_binding_rejected() -> None:
    with pytest.raises(ValueError, match="wrong joint transmission"):
        validate_yam_model(_model(wrong_transmission=True))


def test_missing_names_rejected_instead_of_negative_indexing() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="missing joint left_joint1"):
        validate_yam_model(model)
