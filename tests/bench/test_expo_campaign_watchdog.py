"""Installed coordinator admission only; no lease/model/native execution."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from abc_bench.subset import load_job


class CampaignWatchdogTests(unittest.TestCase):
    def fixture(self, root):
        p = {
            "seed": 42,
            "batch_size": 64,
            "utd_ratio": 20,
            "completed_episode_warmup": 10,
            "updates_per_trigger": 3,
            "max_controls": 53100,
            "sampler_timing_plans": 0,
            "max_episode_completions": 15,
            "max_update_calls": 15,
            "max_seconds": 18000,
            "soft_seconds": 12000,
            "futility_start_calls": 6,
            "futility_min_calls": 9,
            "futility_patience": 2,
            "memory_scheduling_profile": "original_queue3_serial_update_v1",
            "checkpoint_retention": "terminal_complete_only",
            "episode_seeds": list(range(9200, 9215)),
        }
        config = {
            "schema_version": 1,
            "phase": "original_expo_task_rl_serial_campaign",
            "parameters": p,
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        job = {
            "schema_version": 1,
            "label": "Explicit CPU admission fixture",
            "python": sys.executable,
            "module": "expo_abc_adapter.stage",
            "arguments": ["--config", str(config_path)],
            "max_wall_s": 18300,
        }
        job_path = root / "job.json"
        job_path.write_text(json.dumps(job))
        return config, config_path, job, job_path

    def test_exact_campaign_parent_admission_and_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, job, path = self.fixture(Path(tmp))
            self.assertEqual(load_job(path), job)
            for seconds in (18301, 18299, float("nan"), True):
                path.write_text(json.dumps({**job, "max_wall_s": seconds}))
                with self.assertRaises(ValueError):
                    load_job(path)

    def test_changed_bounds_recipe_seeds_profile_or_retention_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, config_path, _, path = self.fixture(Path(tmp))
            for field, value in [
                ("batch_size", 32),
                ("utd_ratio", 1),
                ("updates_per_trigger", 1),
                ("max_update_calls", 18),
                ("max_episode_completions", 16),
                ("max_controls", 38940),
                ("max_seconds", 18001),
                ("soft_seconds", 18000),
                ("futility_start_calls", 3),
                ("futility_min_calls", 6),
                ("futility_patience", 1),
                ("checkpoint_retention", "rolling"),
                ("memory_scheduling_profile", "fallback"),
                ("episode_seeds", list(range(9200, 9211))),
                ("episode_seeds", [9200] * 15),
                ("max_update_calls", True),
            ]:
                with self.subTest(field=field, value=value):
                    changed = copy.deepcopy(config)
                    changed["parameters"][field] = value
                    config_path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        load_job(path)

    def test_old_extended_ordinary_and_serial_recipes_remain_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, config_path, _, path = self.fixture(Path(tmp))
            original = {
                k: config["parameters"][k]
                for k in (
                    "seed",
                    "batch_size",
                    "utd_ratio",
                    "completed_episode_warmup",
                    "updates_per_trigger",
                    "sampler_timing_plans",
                    "max_seconds",
                )
            }
            original.update(max_controls=38940, episode_seeds=list(range(9200, 9211)))
            for phase in ("original_expo_task_rl", "original_expo_task_rl_serial"):
                p = copy.deepcopy(original)
                if phase.endswith("_serial"):
                    p["memory_scheduling_profile"] = "original_queue3_serial_update_v1"
                config_path.write_text(
                    json.dumps({"schema_version": 1, "phase": phase, "parameters": p})
                )
                self.assertEqual(load_job(path)["max_wall_s"], 18300)
                for key, value in [
                    ("episode_seeds", list(range(9200, 9215))),
                    ("updates_per_trigger", 15),
                    ("max_controls", 53100),
                ]:
                    changed = copy.deepcopy(p)
                    changed[key] = value
                    config_path.write_text(
                        json.dumps(
                            {"schema_version": 1, "phase": phase, "parameters": changed}
                        )
                    )
                    with self.assertRaises(ValueError):
                        load_job(path)

    def test_other_phases_and_symlink_config_keep_old_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, config_path, job, path = self.fixture(root)
            for phase in (
                "original_expo_task_serial_campaign_frozen_eval",
                "original_pi05_task_sft",
                "original_expo_task_recovered_serial_frozen_eval",
            ):
                config_path.write_text(json.dumps({**config, "phase": phase}))
                with self.assertRaises(ValueError):
                    load_job(path)
            config_path.write_text(json.dumps(config))
            link = root / "linked.json"
            link.symlink_to(config_path)
            job["arguments"] = ["--config", str(link)]
            path.write_text(json.dumps(job))
            with self.assertRaises(ValueError):
                load_job(path)
