"""Source-specific CPU contracts for the V2 inspection binding; no model imports."""

import ast
import json
from pathlib import Path

import pytest

from abc_bench import mulligan_r1lite_qualification as qualification

M = Path("/home/user/aditya/RL/builds/other-methods-plan-20261015")
EXPECTED = M / "original-qualification-operator-v2/SOURCE_MANIFEST.json"
EXPECTED_SHA = "18c7d3af2c1626e7bb56c131b14ef65010e0186d0af7ee8d04338b10cf72b8a3"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_installed_operator_authenticates_the_exact_reviewed_v2():
    assert qualification.OPERATOR_MANIFEST_PATH == EXPECTED
    assert qualification.OPERATOR_MANIFEST_SHA256 == EXPECTED_SHA
    assert qualification.operator_source() == EXPECTED.parent / "inspect_model.py"


def test_all_original_profile_functions_and_other_constants_unchanged():
    baseline = (
        M
        / "qualification-canonical-release-01/build-context/abc_bench/mulligan_r1lite_qualification.py"
    )

    def without_binding(path):
        module = ast.parse(path.read_text())
        module.body = [
            node
            for node in module.body
            if not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id
                    in {"OPERATOR_MANIFEST_PATH", "OPERATOR_MANIFEST_SHA256"}
                    for target in node.targets
                )
            )
        ]
        return ast.dump(module, include_attributes=False)

    assert without_binding(Path(qualification.__file__)) == without_binding(baseline)


@pytest.mark.parametrize("damage", ["manifest", "operator"])
def test_operator_hash_changes_are_rejected(tmp_path, monkeypatch, damage):
    source = tmp_path / "inspect_model.py"
    source.write_text("# explicit non-executed fixture\n")
    file_pin = qualification.pin(source)
    manifest = write(
        tmp_path / "SOURCE_MANIFEST.json",
        {"files": [{**file_pin, "path": source.name}]},
    )
    monkeypatch.setattr(qualification, "OPERATOR_MANIFEST_PATH", manifest)
    monkeypatch.setattr(
        qualification, "OPERATOR_MANIFEST_SHA256", qualification.pin(manifest)["sha256"]
    )
    if damage == "manifest":
        manifest.write_text("{}")
    else:
        source.write_text("# changed\n")
    with pytest.raises(ValueError):
        qualification.operator_source()


@pytest.mark.parametrize("old_binding", [False, True])
def test_timing_admission_requires_v2_and_existing_update_gate(tmp_path, old_binding):
    marker = qualification.pin(
        write(tmp_path / "cpu-fixture.json", {"fixture_only": True})
    )
    config = {
        "stage": "idql",
        "phase": "timing_monitor",
        "author_manifest": marker,
        "runtimes": {"explicit": "metadata-double"},
        "parent": None,
    }
    inputs = {"data": marker, "task": marker}
    receipt = {
        "schema": "sadhana.mulligan_r1_qualification_admission/1",
        "status": "accepted_for_short_original_qualification",
        "stage": "idql",
        "phase": "timing_monitor",
        "task": qualification.TASK,
        "embodiment": qualification.EMBODIMENT,
        "timeout_s": 1800,
        "evidence": {key: marker for key in qualification.GATES | {"update_restore"}},
        "bindings": {
            "author_manifest": marker,
            "runtimes": config["runtimes"],
            "data": marker,
            "task": marker,
            "parent": None,
            "operator_manifest": qualification.pin(
                M / "original-qualification-operator-v1/SOURCE_MANIFEST.json"
                if old_binding
                else EXPECTED
            ),
        },
    }
    path = write(tmp_path / "admission.json", receipt)
    config["admission"] = qualification.pin(path)
    if old_binding:
        with pytest.raises(ValueError, match="bindings differ"):
            qualification.verify_admission(config, inputs)
    else:
        qualification.verify_admission(config, inputs)
        del receipt["evidence"]["update_restore"]
        config["admission"] = qualification.pin(write(path, receipt))
        with pytest.raises(ValueError, match="evidence missing"):
            qualification.verify_admission(config, inputs)
