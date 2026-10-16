"""Portable command fixtures; no original trainer, installation or model required."""

from pathlib import Path

import pytest

from abc_bench import mulligan_r1lite_qualification as qualification
from abc_bench import mulligan_r1lite_training as training


@pytest.fixture
def recipe():
    # Small explicit command fixture, not an admitted training recipe.
    return {
        "argv": [
            "--training.training_steps=150000",
            "--training.update_components=critic_value_only",
            "--training.resume_checkpoint_freq=5000",
            "--training.periodic_checkpoint_freq=0",
            "--training.save_final_model=True",
            "--eval.freq=150000",
        ]
    }


def test_full_divl_names_are_unique_and_preserve_recipe(recipe):
    config = {"mode": training.MODE, "stage": "divl", "python": "/python"}
    original_arguments = list(recipe["argv"])
    commands = [
        training.command_for(
            config, recipe, Path("/bench") / name / "training", Path("/parent")
        )
        for name in ("author-subset-first", "author-subset-second")
    ]
    for command, name in zip(commands, ("first", "second"), strict=True):
        assert command == [
            "/python",
            "-I",
            "-u",
            "-B",
            "-m",
            "mulligan.training.train",
            *original_arguments,
            f"--wandb.run_name=r1-divl-author-subset-{name}",
        ]
    assert commands[0][-1] != commands[1][-1]
    assert recipe["argv"] == original_arguments


@pytest.mark.parametrize(
    "mode,stage",
    [
        (training.MODE, "idql"),
        (qualification.MODE, "idql"),
        (qualification.MODE, "divl"),
        (None, "divl"),
    ],
)
def test_other_modes_do_not_add_a_name(recipe, mode, stage):
    config = {"mode": mode, "stage": stage, "python": "/python"}
    command = training.command_for(
        config, recipe, Path("/bench/run/training"), Path("/parent")
    )
    assert command[6:] == recipe["argv"]


@pytest.mark.parametrize("stage", ["idql", "divl"])
@pytest.mark.parametrize(
    "phase,steps,save_every,frequency",
    [("update_restore", "2", "1", "null"), ("timing_monitor", "2000", "1000", "2000")],
)
def test_short_qualification_keeps_its_own_name(
    recipe, stage, phase, steps, save_every, frequency
):
    config = {
        "mode": qualification.MODE,
        "stage": stage,
        "phase": phase,
        "python": "/python",
    }
    command = qualification.command_for(
        config, recipe, Path("/bench/run/qualification"), Path("/parent")
    )
    expected = dict(arg.split("=", 1) for arg in recipe["argv"])
    expected.update(
        {
            "--training.training_steps": steps,
            "--training.resume_checkpoint_freq": save_every,
            "--training.periodic_checkpoint_freq": save_every,
            "--eval.freq": frequency,
            "--wandb.run_name": f"r1-diagnostic-{stage}-{phase}-qualification",
        }
    )
    assert command[6:] == [f"{key}={value}" for key, value in expected.items()]


def test_full_divl_rejects_a_recipe_name(recipe):
    recipe["argv"].append("--wandb.run_name=other")
    with pytest.raises(ValueError, match="canonical run name"):
        training.command_for(
            {"mode": training.MODE, "stage": "divl", "python": "/python"},
            recipe,
            Path("/bench/run/training"),
            Path("/parent"),
        )


def test_full_idql_preserves_an_existing_recipe_name(recipe):
    recipe["argv"].append("--wandb.run_name=existing-idql")
    command = training.command_for(
        {"mode": training.MODE, "stage": "idql", "python": "/python"},
        recipe,
        Path("/bench/run/training"),
        None,
    )
    assert command[6:] == recipe["argv"]
