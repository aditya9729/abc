"""Explicit CPU fixtures for original-program supervision; no simulator or GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from abc_bench import wbc_program as wbc
from abc_bench.subset import load_job


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    # Process fixtures must not depend on free space on the shared host.
    # Dedicated storage-bound tests override this explicit CPU double.
    monkeypatch.setattr(
        wbc.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * wbc.GIB)
    )
    source = tmp_path / "source"
    script = source / "scripts/train_feedback_residual.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import pathlib,json,sys\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir()\n(p/'training_summary.json').write_text(json.dumps({'status':'complete','optimizer_steps':40}))\n"
    )
    (source / "src/fixture").mkdir(parents=True)
    (source / "src/fixture/__init__.py").write_text("# Explicit CPU fixture\n")
    interpreter = tmp_path / "prefix/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys._base_executable).resolve())
    installed = tmp_path / "prefix/lib/python3.12/site-packages/fixture/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_bytes((source / "src/fixture/__init__.py").read_bytes())
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-qm",
            "CPU fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(wbc, "AUTHOR_ROOT", source)
    monkeypatch.setattr(wbc, "AUTHOR_COMMIT", commit)
    monkeypatch.setattr(wbc, "AUTHOR_PYTHON", interpreter)
    monkeypatch.setattr(wbc, "require_subset_lease", lambda: None)
    model = tmp_path / "model.xml"
    model.write_text("Explicit non-simulator fixture")
    pins = [interpreter, script, source / "src/fixture/__init__.py", installed, model]
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps({"files": [{"path": str(p), "sha256": wbc.digest(p)} for p in pins]})
    )
    cfg = {
        "schema_version": 1,
        "python": str(interpreter),
        "source_root": str(source),
        "source_commit": commit,
        "program": "pose-ppo",
        "arguments": [
            "--model",
            str(model),
            "--output",
            "{output}",
            "--device",
            "cuda:0",
        ],
        "timeout_s": 3,
        "input_manifest": str(manifest),
        "expected_summary": {"status": "complete", "optimizer_steps": 40},
    }
    return cfg, tmp_path / "config.json", script


def load(setup):
    cfg, path, _ = setup
    path.write_text(json.dumps(cfg))
    return wbc.load_config(path)


def test_canonical_child_and_summary(tmp_path, setup):
    result = wbc.run(load(setup), tmp_path / "out")
    assert result["status"] == "completed" and result["direct_child_stopped"]
    assert result["summary"]["optimizer_steps"] == 40
    assert result["experiment_accepted"] is False
    assert result["descendant_cleanup_owner"] == "canonical runner"
    assert not Path(f"/proc/{result['author_pid']}").exists()
    assert wbc.verify_inputs(setup[0])["files"] == 5


@pytest.fixture
def cache_seed(tmp_path, setup):
    source = tmp_path / "completed/cache/warp_cache_path"
    module = source / "1.18.0/wp_explicit_fixture"
    module.mkdir(parents=True)
    binary = module / "fixture.sm80.cubin"
    binary.write_bytes(b"Explicit CPU cache fixture; never a native CUDA binary")
    metadata = module / "fixture.meta"
    metadata.write_text('{"explicit_cpu_fixture":true}')
    receipt = source.parents[1] / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "original_r1lite_wbc_program/1",
                "status": "completed",
                "program": "pose-ppo",
                "author_exit_code": 0,
                "direct_child_stopped": True,
                "configuration": setup[0].copy(),
            }
        )
    )
    canonical = tmp_path / "runs/completed.json"
    canonical.parent.mkdir()
    canonical.write_text(
        json.dumps(
            {
                "status": "completed",
                "worker_exit_code": 0,
                "worker_receipt_sha256": wbc.digest(receipt),
            }
        )
    )
    setup[0]["warp_cache_seed"] = str(source)
    manifest = Path(setup[0]["input_manifest"])
    data = json.loads(manifest.read_text())
    data["files"].extend(
        {"path": str(p), "sha256": wbc.digest(p)}
        for p in (receipt, canonical, binary, metadata)
    )
    manifest.write_text(json.dumps(data))
    return source, binary, receipt


def test_seeded_worker_uses_private_copy(tmp_path, setup, cache_seed):
    source, binary, _ = cache_seed
    before = binary.read_bytes()
    result = wbc.run(load(setup), tmp_path / "out")
    copied = result["warp_cache_seed"]
    assert result["status"] == "completed" and copied["files_copied"] == 2
    assert copied["bytes_copied"] == sum(
        p.stat().st_size for p in source.rglob("*") if p.is_file()
    )
    assert copied["shared_writable_cache"] is False
    target = Path(copied["destination"]) / binary.relative_to(source)
    assert target.read_bytes() == before
    assert target.stat().st_ino != binary.stat().st_ino
    target.write_bytes(b"Private cache may be updated by Warp")
    assert binary.read_bytes() == before
    assert wbc.verify_inputs(setup[0])["files"] == 9


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "fifo"])
def test_seed_rejects_nonregular_tree(tmp_path, setup, cache_seed, kind):
    source, binary, _ = cache_seed
    extra = source / "invalid"
    if kind == "fifo":
        os.mkfifo(extra)
    else:
        extra.symlink_to(binary if kind == "file_symlink" else binary.parent)
    with pytest.raises(ValueError, match="regular files"):
        load(setup)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "running"},
        {"author_exit_code": 1},
        {"direct_child_stopped": False},
        {"configuration": {}},
    ],
)
def test_seed_requires_successful_same_runtime(setup, cache_seed, change):
    receipt = cache_seed[2]
    receipt.write_text(json.dumps({**json.loads(receipt.read_text()), **change}))
    canonical = receipt.parent.parent / "runs/completed.json"
    data = json.loads(canonical.read_text())
    data["worker_receipt_sha256"] = wbc.digest(receipt)
    canonical.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="successful unchanged"):
        load(setup)


@pytest.mark.parametrize(
    "change", [{"status": "running"}, {"worker_receipt_sha256": "wrong"}]
)
def test_seed_requires_completed_canonical_producer(setup, cache_seed, change):
    canonical = cache_seed[2].parent.parent / "runs/completed.json"
    canonical.write_text(json.dumps({**json.loads(canonical.read_text()), **change}))
    with pytest.raises(ValueError, match="matching completed canonical"):
        load(setup)


def test_seed_requires_every_input_pin(tmp_path, setup, cache_seed):
    cfg = load(setup)
    manifest = Path(cfg["input_manifest"])
    data = json.loads(manifest.read_text())
    data["files"].pop()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="omits"):
        wbc.run(cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_changed_seed_refuses_and_copy_checks_its_bytes(tmp_path, setup, cache_seed):
    cfg = load(setup)
    cache_seed[1].write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="Changed pinned"):
        wbc.run(cfg, tmp_path / "out")
    destination = tmp_path / "copy"
    destination.mkdir()
    with pytest.raises(ValueError, match="differs from pinned"):
        wbc.copy_cache_seed(cfg, destination, destination, wbc.time.monotonic())


@pytest.mark.parametrize(
    "limits", [{}, {"disk_reserve_bytes": 8 * wbc.GIB, "output_cap_bytes": wbc.GIB}]
)
@pytest.mark.parametrize("limit", ["disk", "output"])
def test_seed_capacity_failure_never_launches_child(
    tmp_path, setup, cache_seed, monkeypatch, limit, limits
):
    setup[0].update(limits)
    cfg = load(setup)
    reserve, cap = wbc.storage_limits(cfg)
    monkeypatch.setattr(
        wbc,
        "storage_check",
        lambda _, cfg=None: {
            "free_bytes": reserve if limit == "disk" else 100 * wbc.GIB,
            "output_bytes": cap if limit == "output" else 0,
        },
    )
    # Input verification uses git subprocesses. Reject only an attempted author.
    original = wbc.subprocess.Popen

    def refuse_author(command, *args, **kwargs):
        assert command[0] == "git", "Cache refusal must precede author creation"
        return original(command, *args, **kwargs)

    monkeypatch.setattr(wbc.subprocess, "Popen", refuse_author)
    with pytest.raises(RuntimeError, match="disk/output limits"):
        wbc.run(cfg, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
    assert "author_pid" not in receipt


def test_seed_copy_respects_deadline_and_fresh_destination(tmp_path, setup, cache_seed):
    cfg = load(setup)
    destination = tmp_path / "copy"
    destination.mkdir()
    with pytest.raises(subprocess.TimeoutExpired):
        wbc.copy_cache_seed(cfg, destination, destination, wbc.time.monotonic() - 10)
    with pytest.raises(FileExistsError, match="new and empty"):
        wbc.copy_cache_seed(cfg, destination, destination, wbc.time.monotonic())


@pytest.mark.parametrize(
    "change",
    [
        {"timeout_s": float("inf")},
        {"timeout_s": True},
        {"timeout_s": 7101},
        {"program": "arbitrary"},
        {"python": "/usr/bin/python"},
        {"arguments": ["--model", "/tmp", "--output", "/tmp", "--device", "cuda:0"]},
        {
            "arguments": [
                "--model",
                "/tmp",
                "--output",
                "{output}",
                "--device",
                "cuda:0",
                "--evil",
                "x",
            ]
        },
        {
            "arguments": [
                "--model",
                "/tmp",
                "--output",
                "{output}",
                "--device",
                "cuda:0",
                "--device",
                "cuda:0",
            ]
        },
        {"expected_summary": {}},
    ],
)
def test_bad_configuration(setup, change):
    setup[0].update(change)
    with pytest.raises((ValueError, FileNotFoundError)):
        load(setup)


def test_parent_cleanup_margin(setup):
    load(setup)
    job = {
        "module": "abc_bench.wbc_program",
        "arguments": ["--config", str(setup[1])],
        "max_wall_s": 103,
    }
    wbc.validate_job(job, 103)
    with pytest.raises(ValueError, match="plus100s"):
        wbc.validate_job(job, 102)


def test_absent_lease_refuses_before_output(tmp_path, setup, monkeypatch):
    def fail():
        raise RuntimeError("No lease fixture")

    monkeypatch.setattr(wbc, "require_subset_lease", fail)
    with pytest.raises(RuntimeError, match="No lease"):
        wbc.run(load(setup), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_changed_source_and_missing_pin_refuse(tmp_path, setup):
    cfg = load(setup)
    setup[2].write_text("modified\n")
    with pytest.raises(ValueError, match="clean"):
        wbc.run(cfg, tmp_path / "out")
    subprocess.run(
        [
            "git",
            "-C",
            cfg["source_root"],
            "checkout",
            "--",
            "scripts/train_feedback_residual.py",
        ],
        check=True,
    )
    manifest = Path(cfg["input_manifest"])
    data = json.loads(manifest.read_text())
    data["files"].pop()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="omits"):
        wbc.run(cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_timeout_and_bad_exit_are_preserved(tmp_path, setup, monkeypatch):
    cfg = load(setup)
    monkeypatch.setattr(wbc, "verify_inputs", lambda _: {"fixture": True})
    setup[2].write_text("import time\ntime.sleep(30)\n")
    cfg["timeout_s"] = 0.05
    with pytest.raises(subprocess.TimeoutExpired):
        wbc.run(cfg, tmp_path / "timeout")
    receipt = json.loads((tmp_path / "timeout/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
    assert not Path(f"/proc/{receipt['author_pid']}").exists()
    setup[2].write_text("raise SystemExit(75)\n")
    cfg["timeout_s"] = 3
    with pytest.raises(RuntimeError, match="exited 75"):
        wbc.run(cfg, tmp_path / "exit")
    receipt = json.loads((tmp_path / "exit/receipt.json").read_text())
    assert receipt["author_exit_code"] == 75 and receipt["status"] == "failed"


def test_summary_mismatch_is_not_completion(tmp_path, setup):
    cfg = load(setup)
    cfg["expected_summary"]["optimizer_steps"] = 41
    with pytest.raises(ValueError, match="summary differs"):
        wbc.run(cfg, tmp_path / "out")
    assert json.loads((tmp_path / "out/receipt.json").read_text())["status"] == "failed"


def test_disk_guard(tmp_path, monkeypatch):
    class Low:
        free = 19 * wbc.GIB

    monkeypatch.setattr(wbc.shutil, "disk_usage", lambda _: Low())
    with pytest.raises(RuntimeError, match="disk floor"):
        wbc.storage_check(tmp_path)


@pytest.mark.parametrize(
    "limits,expected",
    [
        ({}, (20 * wbc.GIB, 5 * wbc.GIB)),
        ({"disk_reserve_bytes": 8 * wbc.GIB}, (8 * wbc.GIB, 5 * wbc.GIB)),
        ({"output_cap_bytes": wbc.GIB}, (20 * wbc.GIB, wbc.GIB)),
        (
            {"disk_reserve_bytes": 8 * wbc.GIB, "output_cap_bytes": wbc.GIB},
            (8 * wbc.GIB, wbc.GIB),
        ),
    ],
)
def test_actual_loader_resolves_resources_without_changing_author_command(
    tmp_path, setup, limits, expected
):
    before = wbc.command(setup[0], tmp_path / "author")
    setup[0].update(limits)
    cfg = load(setup)
    assert wbc.storage_limits(cfg) == expected
    assert wbc.command(cfg, tmp_path / "author") == before
    job = {
        "schema_version": 1,
        "label": "Explicit CPU resource fixture",
        "method": "WBC",
        "task": "fixture",
        "embodiment": "r1lite",
        "python": sys.executable,
        "module": "abc_bench.wbc_program",
        "arguments": ["--config", str(setup[1])],
        "max_wall_s": 103,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert load_job(path) == job


@pytest.mark.parametrize(
    "key,value",
    [
        ("disk_reserve_bytes", 8 * wbc.GIB - 1),
        ("disk_reserve_bytes", 0),
        ("disk_reserve_bytes", -1),
        ("output_cap_bytes", 5 * wbc.GIB + 1),
        ("output_cap_bytes", 0),
        ("output_cap_bytes", -1),
        *(
            (key, value)
            for key in ("disk_reserve_bytes", "output_cap_bytes")
            for value in (
                True,
                False,
                None,
                "8589934592",
                8589934592.0,
                float("nan"),
                float("inf"),
            )
        ),
    ],
)
def test_invalid_resource_limits_are_rejected_at_config_boundary(setup, key, value):
    setup[0][key] = value
    with pytest.raises(ValueError, match=key):
        load(setup)


@pytest.mark.parametrize(
    "limits", [{}, {"disk_reserve_bytes": 8 * wbc.GIB, "output_cap_bytes": wbc.GIB}]
)
def test_real_disk_and_output_guards_at_exact_boundaries(tmp_path, monkeypatch, limits):
    from types import SimpleNamespace

    reserve, cap = wbc.storage_limits(limits)
    free = reserve
    monkeypatch.setattr(wbc.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    output = tmp_path / "sparse-output-fixture"
    # Sparse metadata exercises the real size census without allocating GiB.
    with output.open("wb") as stream:
        stream.truncate(cap)
    result = wbc.storage_check(tmp_path, limits)
    assert result == {
        "free_bytes": reserve,
        "output_bytes": cap,
        "disk_reserve_bytes": reserve,
        "output_cap_bytes": cap,
    }
    free -= 1
    with pytest.raises(RuntimeError, match="disk floor"):
        wbc.storage_check(tmp_path, limits)
    free = reserve
    with output.open("ab") as stream:
        stream.truncate(cap + 1)
    with pytest.raises(RuntimeError, match="output ceiling"):
        wbc.storage_check(tmp_path, limits)


def test_storage_scan_tolerates_removed_compiler_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        wbc.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * wbc.GIB)
    )
    retained = tmp_path / "policy.pt"
    retained.write_bytes(b"retained artifact")
    temporary = tmp_path / "compiler-header.cuh"
    temporary.write_bytes(b"temporary")
    original = Path.lstat

    def remove_before_stat(path):
        if path == temporary:
            path.unlink()
        return original(path)

    monkeypatch.setattr(Path, "lstat", remove_before_stat)
    result = wbc.storage_check(tmp_path)
    assert result["output_bytes"] == len(b"retained artifact")
    assert retained.read_bytes() == b"retained artifact"


def test_storage_scan_does_not_hide_permission_error(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        wbc.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * wbc.GIB)
    )
    artifact = tmp_path / "policy.pt"
    artifact.write_bytes(b"retained")
    original = Path.lstat

    def deny_stat(path):
        if path == artifact:
            raise PermissionError("test unreadable metadata")
        return original(path)

    monkeypatch.setattr(Path, "lstat", deny_stat)
    with pytest.raises(PermissionError, match="unreadable metadata"):
        wbc.storage_check(tmp_path)


def test_explicit_pilot_bounds_allow_cpu_child_below_old_floor(
    tmp_path, setup, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        wbc.shutil, "disk_usage", lambda _: SimpleNamespace(free=12 * wbc.GIB)
    )
    with pytest.raises(RuntimeError, match="disk floor"):
        wbc.run(load(setup), tmp_path / "default-refusal")
    assert not (tmp_path / "default-refusal/receipt.json").exists()
    setup[0].update(disk_reserve_bytes=8 * wbc.GIB, output_cap_bytes=wbc.GIB)
    result = wbc.run(load(setup), tmp_path / "pilot")
    assert result["status"] == "completed" and result["direct_child_stopped"]
    assert result["storage"]["disk_reserve_bytes"] == 8 * wbc.GIB
    assert result["storage"]["output_cap_bytes"] == wbc.GIB


def test_pilot_final_resource_recheck_rejects_completed_child(
    tmp_path, setup, monkeypatch
):
    from types import SimpleNamespace

    setup[0].update(disk_reserve_bytes=8 * wbc.GIB, output_cap_bytes=wbc.GIB)
    cfg = load(setup)
    free = 12 * wbc.GIB
    original = wbc.verify_inputs
    calls = 0

    def after_child(value):
        nonlocal free, calls
        calls += 1
        result = original(value)
        if calls == 2:
            free = 8 * wbc.GIB - 1
        return result

    monkeypatch.setattr(wbc, "verify_inputs", after_child)
    monkeypatch.setattr(wbc.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    with pytest.raises(RuntimeError, match="disk floor"):
        wbc.run(cfg, tmp_path / "out")
    result = json.loads((tmp_path / "out/receipt.json").read_text())
    assert result["status"] == "failed" and result["direct_child_stopped"]
    assert result["author_exit_code"] == 0


def test_pilot_live_resource_guard_reaps_cpu_child(tmp_path, setup, monkeypatch):
    from types import SimpleNamespace

    setup[0].update(disk_reserve_bytes=8 * wbc.GIB, output_cap_bytes=wbc.GIB)
    cfg = load(setup)
    monkeypatch.setattr(wbc, "verify_inputs", lambda _: {"fixture": True})
    setup[2].write_text("import time\ntime.sleep(30)\n")
    calls = 0

    def disk_usage(_):
        nonlocal calls
        calls += 1
        return SimpleNamespace(free=(12 if calls == 1 else 7) * wbc.GIB)

    monkeypatch.setattr(wbc.shutil, "disk_usage", disk_usage)
    with pytest.raises(RuntimeError, match="disk floor"):
        wbc.run(cfg, tmp_path / "out")
    result = json.loads((tmp_path / "out/receipt.json").read_text())
    assert result["status"] == "failed" and result["direct_child_stopped"]
    assert not Path(f"/proc/{result['author_pid']}").exists()


def test_pins_rechecked_after_success(tmp_path, setup, monkeypatch):
    cfg = load(setup)
    original = wbc.verify_inputs
    calls = []

    def changed(value):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("Changed pinned input fixture")
        return original(value)

    monkeypatch.setattr(wbc, "verify_inputs", changed)
    with pytest.raises(ValueError, match="Changed pinned"):
        wbc.run(cfg, tmp_path / "out")
    assert len(calls) == 2


def test_subset_admits_only_validated_worker(tmp_path, setup):
    load(setup)
    job = {
        "schema_version": 1,
        "label": "Explicit CPU process fixture",
        "method": "WBC",
        "task": "fixture",
        "embodiment": "r1lite",
        "python": str(Path(sys.executable)),
        "module": "abc_bench.wbc_program",
        "arguments": ["--config", str(setup[1])],
        "max_wall_s": 103,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert load_job(path)["module"] == "abc_bench.wbc_program"
    job["max_wall_s"] = 102
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="plus100s"):
        load_job(path)


@pytest.mark.parametrize("explicit_scope", [None, "Original WBC pose PPO CPU fixture"])
def test_canonical_receipt_describes_original_wbc(
    tmp_path, setup, monkeypatch, explicit_scope
):
    """Exercise both published receipts and the original-program boundary on CPU."""
    import argparse

    from abc_bench import runner, subset

    load(setup)
    results = tmp_path / "results"
    results.mkdir()
    (results / "gpu_budget.json").write_text(
        json.dumps({"gpu": 0, "charged_seconds": 7, "limit_seconds": 500})
    )
    monkeypatch.setattr(runner, "RESULTS", results)
    monkeypatch.setattr(
        subset,
        "inspect_reserved_gpu",
        lambda _: {"uuid": "explicit-cpu-double", "ordinal": 0},
    )
    job = {
        "schema_version": 1,
        "label": "Original WBC pose PPO fixture",
        "task": "task-motion-slow-train",
        "embodiment": "r1lite",
        "python": sys.executable,
        "module": "abc_bench.wbc_program",
        "arguments": ["--config", str(setup[1])],
        "max_wall_s": 103,
    }
    if explicit_scope is not None:
        job["claim_scope"] = explicit_scope
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    published = []
    original_publish = runner.publish

    def capture_publish(receipt, output):
        published.append(json.loads(json.dumps(receipt)))
        original_publish(receipt, output)

    def cpu_execution_double(command, log, timeout, *, gpu_uuid):
        assert gpu_uuid == "explicit-cpu-double" and timeout == 103
        wbc.run(wbc.load_config(Path(command[-3])), Path(command[-1]))
        return 0

    monkeypatch.setattr(runner, "publish", capture_publish)
    monkeypatch.setattr(runner, "execute_command", cpu_execution_double)
    result = subset.run_job(
        argparse.Namespace(training_config=path, timeout_seconds=103)
    )
    assert [receipt["status"] for receipt in published] == ["running", "completed"]
    for receipt in published:
        assert receipt["phase"] == "original_wbc_program"
        assert receipt["task"] == job["task"]
        assert receipt["embodiment"] == job["embodiment"]
        assert "ABC-VLA" not in receipt["claim_scope"]
        assert receipt["claim_scope"] == (
            explicit_scope
            or "Original R1 Lite WBC process execution; native results and checkpoints require separate acceptance"
        )
    worker_path = results / result["run_id"] / "receipt.json"
    assert result["worker_receipt_sha256"] == wbc.digest(worker_path)
    assert json.loads(worker_path.read_text())["experiment_accepted"] is False
    ledger = json.loads((results / "gpu_budget.json").read_text())
    assert ledger["charged_seconds"] >= 7 and "active_parent_pid" not in ledger


def test_export_evaluator_retains_author_sibling_import(setup):
    cfg = setup[0].copy()
    cfg["program"] = "exported-policy-evaluate"
    argv = wbc.command(cfg, Path("/tmp/explicit-fixture"))
    assert "-I" not in argv
    assert argv[3].endswith("scripts/evaluate_exported_policy.py")
    assert "-I" in wbc.command(setup[0], Path("/tmp/explicit-fixture"))


def test_signal_cancellation_reaps_direct_child(tmp_path, setup, monkeypatch):
    cfg = load(setup)
    monkeypatch.setattr(wbc, "verify_inputs", lambda _: {"fixture": True})
    setup[2].write_text(
        "import os,signal,time\nos.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(30)\n"
    )
    with pytest.raises(RuntimeError, match="received signal"):
        wbc.run(cfg, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
    assert not Path(f"/proc/{receipt['author_pid']}").exists()


def test_live_disk_refusal_cancels_child(tmp_path, setup, monkeypatch):
    cfg = load(setup)
    monkeypatch.setattr(wbc, "verify_inputs", lambda _: {"fixture": True})
    setup[2].write_text("import time\ntime.sleep(30)\n")
    calls = []

    def check(path, cfg=None):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("Live disk refusal fixture")
        return {"fixture": True}

    monkeypatch.setattr(wbc, "storage_check", check)
    with pytest.raises(RuntimeError, match="Live disk"):
        wbc.run(cfg, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
