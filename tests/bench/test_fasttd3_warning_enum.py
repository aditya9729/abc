"""Read the real installed MuJoCo warning API; no model or simulator is created."""

import importlib.metadata
import re
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from abc_bench.fasttd3_campaign_host import physics_snapshot


def test_warning_reader_matches_installed_381_public_enum_and_c_header():
    assert importlib.metadata.version("mujoco") == "3.8.1"
    distribution = importlib.metadata.distribution("mujoco")
    header = Path(distribution.locate_file("mujoco/include/mujoco/mjdata.h"))
    declaration = re.search(
        r"typedef enum mjtWarning_\s*\{(.*?)\}\s*mjtWarning;", header.read_text(), re.DOTALL
    )
    assert declaration is not None
    c_names = re.findall(r"\bmjWARN_[A-Z]+\b", declaration.group(1))
    public_names = [
        name for name in mujoco.mjtWarning.__members__ if name.startswith("mjWARN_")
    ]
    assert public_names == c_names
    assert len(c_names) == int(mujoco.mjtWarning.mjNWARNING) == 7
    assert "mjWARN_VGEOMFULL" not in public_names
    # Only warning-stat/data storage is a double. The actual versioned public
    # enum and reader are used, so an unsupported name fails this regression.
    warnings = [
        SimpleNamespace(number=index + 1, lastinfo=30 + index)
        for index in range(int(mujoco.mjtWarning.mjNWARNING))
    ]
    data = SimpleNamespace(
        warning=warnings,
        time=1.5,
        **{name: np.zeros(1) for name in ("qpos", "qvel", "qacc", "ctrl")},
    )
    result = physics_snapshot(data)
    assert set(result["warnings"]) == {name.removeprefix("mjWARN_") for name in c_names}
    for name in c_names:
        index = int(getattr(mujoco.mjtWarning, name))
        assert result["warnings"][name.removeprefix("mjWARN_")] == {
            "number": index + 1,
            "lastinfo": 30 + index,
        }
    assert result["time"] == 1.5 and all(result["finite"].values())
