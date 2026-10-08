"""Explicit metadata/worker doubles only; no native or model eligibility."""

import base64
import copy
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from abc_bench import runner
from abc_bench import visual_evaluation_dispatch as visual
from abc_bench.visual_evaluation_bootstrap import artifact, canonical, process_identity

WORKER = '''"""Owned standard-library evaluator double; not a native implementation."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('config','out','campaign-directory','trusted-native-admission','trusted-native-admission-sha256','trusted-native-admission-bytes','trusted-checkpoint-admission','trusted-checkpoint-admission-sha256','trusted-checkpoint-admission-bytes'):
 p.add_argument('--'+name)
a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir()
campaign=Path(a.campaign_directory);ledger=json.loads((campaign/'gpu_budget.json').read_text())
assert ledger['active_parent_pid']==os.getppid()
assert os.environ['CUDA_VISIBLE_DEVICES']==''
with (campaign/'.gpu-budget.lock').open('rb') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('Fixture did not inherit a held original lease')
native=json.loads(Path(a.trusted_native_admission).read_text());identity=native['fixture_identity']
cohort=json.loads(Path(cfg['cohort']['path']).read_text())['episodes']
seed=cfg['seed'];status='completed' if seed in (910,914,916) else 'failed' if seed in (912,913,917) else 'incomplete'
if seed==915:
 signal.signal(signal.SIGTERM,signal.SIG_IGN);descendant=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(90)']);temporary=out/'sleep-ready.tmp';temporary.write_text(json.dumps({'pid':os.getpid(),'descendant_pid':descendant.pid}));temporary.replace(out/'sleep-ready.json');time.sleep(90)
if seed==916:
 child=subprocess.Popen([sys.executable,'-I','-B','-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(90)'])
 (out/'descendant.json').write_text(json.dumps({'pid':child.pid}))
rows=[]
for pin in cfg['resume']:
 rows.extend(r for r in json.loads(Path(pin['path']).read_text())['episodes'] if r['complete'])
prefix=len(rows);new_count=50-prefix if seed in (910,913,914,916) else (1 if prefix<50 else 0)
for n in range(prefix,prefix+new_count):
 complete=seed in (910,913,914,916);returned=0 if seed==917 else 1;captured=0 if seed in (912,917) else returned
 row={'id':cohort[n]['id'],'requested_seed':cohort[n]['seed'],'complete':complete,'status':'success' if complete else 'failed_partial' if status=='failed' else 'development_cap_partial','returned_controls':returned,'captured_judged_controls':captured,'unknown_physical_attempts':1 if seed==917 else 0,'success':True if complete else None if status=='failed' else False,'native_success':False,'policy_unchanged':complete or status=='incomplete','commands':[{'index':1,'physical':[0.0]*14,'u':[0.0]*14,'z':[0.0]*14}] if returned else [],'proposals':[],'control_wall_samples':[{'iteration':1}] if returned else [],'media':None}
 if cfg['protocol'].startswith('qf3'):row['qf3_policy']={'selection_rng':None,'preservation_before':'a'*64,'preservation_after':'a'*64}
 rows.append(row)
new=rows[prefix:];returned=sum(r['returned_controls'] for r in new);captured=sum(r['captured_judged_controls'] for r in new);unknown=sum(r['unknown_physical_attempts'] for r in new)
complete=[r for r in rows if r['complete']];successes=sum(r['success'] for r in complete);native_successes=sum(r['native_success'] for r in complete);successful=[r for r in complete if r['success']]
metrics={'requested':50,'completed':len(complete),'partial':len(rows)-len(complete),'unstarted':50-len(rows),'successes':successes,'complete_success_rate':successes/len(complete) if complete else None,'success_definition':'five distinct first placements in episode history','native_current_successes':native_successes,'native_current_complete_success_rate':native_successes/len(complete) if complete else None,'mean_controls_success_only':sum(r['returned_controls'] for r in successful)/len(successful) if successful else None,'mean_controls_all_complete':sum(r['returned_controls'] for r in complete)/len(complete) if complete else None,'success_bounds_over_requested':[successes/50,(successes+50-len(complete))/50],'full_cohort_score':successes/50 if status=='completed' and len(complete)==50 else None}
receipt={'schema_version':1,'revision':'sadhana.visual-methods-matched-evaluation/2','domain':'sim','protocol_id':hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'identity':identity,'status':status,'episodes':rows,'error':'late no-update fixture' if seed==913 else 'capture/step fixture failure' if status=='failed' else None,'secondary_errors':[],'expert_ground_truth':False,'scientific_recipe':identity['scientific_recipe'],'author_reproduction':False,'native_world_restored':False,'requested_episode_count':50,'counts':{'attempted_dispatches':returned+unknown,'returned_controls':returned,'captured_judged_controls':captured,'unknown_physical_attempts':unknown},'metrics':metrics}
if cfg['protocol'].startswith('qf3'):
 receipt['counts_scope']='current invocation; completed resume prefix excluded';receipt['cumulative_returned_controls']=sum(r['returned_controls'] for r in rows)
(out/'receipt.json').write_text(json.dumps(receipt));print(json.dumps({'scope':'owned CPU fixture only','parent_pid':os.getppid()}))
raise SystemExit(2 if seed==914 else 1 if status=='failed' else 0)
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")
    return artifact(path)


@pytest.fixture
def request_factory(tmp_path, monkeypatch):
    campaign = tmp_path / "original-abc" / "outputs" / "bench"
    campaign.mkdir(parents=True)
    (campaign / ".gpu-budget.lock").touch()
    write(
        campaign / "gpu_budget.json",
        {"gpu": 0, "charged_seconds": 10.0, "limit_seconds": 1000.0},
    )
    monkeypatch.setattr(visual, "CANONICAL_CAMPAIGN", campaign)
    monkeypatch.setattr(runner, "ROOT", campaign.parent.parent)
    calls = []

    def no_gpu_query(ordinal):
        calls.append(ordinal)
        return {"uuid": "", "scope": "explicit CPU occupancy double; all CUDA hidden"}

    monkeypatch.setattr(runner, "inspect_reserved_gpu", no_gpu_query)
    prefix = tmp_path / "isolated-worker"
    subprocess.run(
        [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(prefix)],
        check=True,
    )
    site = (
        prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    nrh = site / "nrh"
    nrh.mkdir()
    for name in (
        "__init__",
        "visual_method_evaluation",
        "visual_policy_export",
        "evaluation_video",
        "qf3_visual_policy",
    ):
        (nrh / (name + ".py")).write_text(
            WORKER
            if name == "visual_method_evaluation"
            else '"""Owned CPU package fixture; no native/model implementation."""\n'
        )
    dist = site / "nirvana_rl_harness-0.4.13.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: nirvana-rl-harness\nVersion: 0.4.13\n"
    )
    records = []
    for path in [*nrh.glob("*.py"), dist / "METADATA"]:
        raw = path.read_bytes()
        records.append(
            [
                str(path.relative_to(site)),
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
                .decode()
                .rstrip("="),
                str(len(raw)),
            ]
        )
    records.append([str((dist / "RECORD").relative_to(site)), "", ""])
    stream = io.StringIO()
    csv.writer(stream).writerows(records)
    (dist / "RECORD").write_text(stream.getvalue())
    worker = {
        "python": artifact(prefix / "bin/python"),
        "python_resolved": str((prefix / "bin/python").resolve()),
        "prefix": str(prefix),
        "version": sys.version,
        "pyvenv": artifact(prefix / "pyvenv.cfg"),
        "site_packages": str(site),
        "nrh_version": "0.4.13",
        "metadata": artifact(dist / "METADATA"),
        "record": artifact(dist / "RECORD"),
        "nrh_files": {str(p.relative_to(site)): artifact(p) for p in nrh.glob("*.py")},
        "pth": [],
        "startup": [],
        "dependency_metadata": {},
    }
    software_report = write(
        tmp_path / "synthetic-source-admission.json",
        {"scope": "explicit CPU fixture; no actual release/admission"},
    )
    software = write(
        tmp_path / "software-binding.json",
        {
            "kind": "root_abc_visual_worker_release_binding",
            "status": "accepted_cpu_source_wheel_install",
            "nrh_version": "0.4.13",
            "nrh_sources": {n: p["sha256"] for n, p in worker["nrh_files"].items()},
            "source_admission": software_report,
        },
    )
    cohort = write(
        tmp_path / "owned-development-cohort.json",
        {
            "schema_version": 1,
            "kind": "visual_development_cohort",
            "split": "development",
            "task": visual.TASK,
            "episodes": [
                {"id": f"fixture-{i:02d}", "seed": 100 + i} for i in range(50)
            ],
        },
    )
    construction = write(
        tmp_path / "non-native-construction.json",
        {"scope": "synthetic metadata fixture; real NRH construction would reject"},
    )
    limits = {
        "host_rss_limit_bytes": 2 * 1024**3,
        "gpu_memory_limit_bytes": 8 * 1024**3,
        "virtual_address_limit_bytes": None,
    }
    resource_pin = write(
        tmp_path / "resource-binding.json",
        {
            "kind": "root_abc_visual_native_resource_admission",
            "status": "accepted_for_serial_native_development_evaluation",
            "limits": limits,
            "scope": "explicit CPU fixture only",
        },
    )
    resources = {"admission": resource_pin, **limits}

    def make(protocol="qf3-frozen-h30c15-nortc", seed=910, resume=None):
        learned = visual.PROTOCOLS[protocol]
        checkpoint = (
            write(
                tmp_path / f"non-model-trust-{seed}.json",
                {"scope": "non-model CPU trust fixture"},
            )
            if learned
            else None
        )
        export = (
            write(
                tmp_path / f"non-model-export-{seed}.json",
                {"scope": "non-model CPU export fixture"},
            )
            if learned
            else None
        )
        config = {
            "schema_version": 1,
            "domain": "sim",
            "protocol": protocol,
            "cohort": cohort,
            "construction_config": construction,
            "export": export,
            "seed": seed,
            "control_cap": 1000,
            "max_wall_s": 0.1,
            "mode": "paused_simulation_fixed_tick",
            "video": {"enabled": False, "episode_ids": []},
            "resume": resume or [],
        }
        config_pin = write(tmp_path / f"config-{seed}.json", config)
        identity = {
            "revision": visual.REVISION,
            "config": {
                k: v for k, v in config.items() if k not in {"max_wall_s", "resume"}
            },
            "method_sources": {"scope": "explicit CPU double"},
            "evaluator_sources": {
                n: worker["nrh_files"]["nrh/" + n + ".py"]["sha256"]
                for n in (
                    "visual_policy_export",
                    "visual_method_evaluation",
                    "evaluation_video",
                    "qf3_visual_policy",
                )
            },
            "task_protocol": "sadhana.qf3-abc-first-placement/1",
            "expert_ground_truth": False,
            "scientific_recipe": {
                "method": protocol.split("-")[0],
                "recipe": "explicit CPU worker double; no inference or performance",
                "author_reproduction": False,
                "author_initializer_confirmed": False,
            },
        }
        native = {
            "schema_version": 1,
            "kind": "visual_method_native_evaluation_admission",
            "status": "accepted_for_serial_native_development_evaluation",
            "requested_episode_count": 50,
            "campaign_directory": str(campaign),
            "device": "cuda:0",
            "evaluator_sources": identity["evaluator_sources"],
            "protocol_id": hashlib.sha256(canonical(identity)).hexdigest(),
            "cohort": cohort,
            "construction_config": construction,
            "allowed_protocols": [protocol],
            "worker_binding": worker,
            "coordinator_sources": visual.source_binding(),
            "native_resources": resources,
            "fixture_identity": identity,
        }
        native_pin = write(tmp_path / f"native-{seed}.json", native)
        value = {
            "schema_version": 1,
            "kind": "root_abc_matched_visual_evaluation_request",
            "config": config_pin,
            "identity": identity,
            "worker": worker,
            "software_binding": software,
            "native_admission": native_pin,
            "checkpoint_admission": checkpoint,
            "coordinator_sources": visual.source_binding(),
            "campaign_directory": str(campaign),
            "cleanup_s": 0.4,
            "receipt_max_bytes": visual.MAX_RECEIPT_BYTES,
            "native_resources": resources,
        }
        pin = write(tmp_path / f"request-{seed}.json", value)
        return pin, value

    make.campaign, make.calls, make.prefix = campaign, calls, prefix
    return make


def args(factory, pin, timeout=8.0):
    from argparse import Namespace

    return Namespace(
        algorithm="visual-method-evaluation",
        timeout_seconds=timeout,
        results=factory.campaign,
        evaluation_request=Path(pin["path"]),
        evaluation_request_sha256=pin["sha256"],
        evaluation_request_bytes=pin["bytes"],
    )


@pytest.mark.parametrize("protocol", visual.PROTOCOLS)
def test_all_six_use_actual_flags_one_direct_exec_and_original_lease(
    request_factory, protocol
):
    pin, _ = request_factory(protocol)
    receipt = runner.run_baseline(args(request_factory, pin))
    assert receipt["status"] == "completed" and receipt["worker_exit_code"] == 0
    assert request_factory.calls == [0]
    assert (
        receipt["metrics"]["simulation_steps"] == 50
        and receipt["metrics"]["updates"] == 0
    )
    assert receipt["metrics"]["successes"] is None
    assert receipt["producer_evaluation_metrics"]["full_cohort_score"] == 1.0
    control = request_factory.campaign / receipt["run_id"] / "control"
    origin = json.loads((control / "bootstrap-origin.json").read_text())
    ready = json.loads((control / "guardian-ready.json").read_text())
    assert origin["parent"] == process_identity(os.getpid())
    assert origin["worker"] == ready["worker"]
    assert origin["python_literal"] == str(request_factory.prefix / "bin/python")
    assert origin["nrh_origin_literal"] == origin["nrh_origin_resolved"]
    assert receipt["guardian_outcome"]["owned_group_cleanup_complete"]
    ledger = json.loads((request_factory.campaign / "gpu_budget.json").read_text())
    assert 10 <= ledger["charged_seconds"] < 12 and "active_visual_lease" not in ledger


@pytest.mark.parametrize(
    "seed,status,known,unknown",
    [
        (911, "incomplete", 1, 0),
        (912, "failed", 1, 0),
        (913, "failed", 50, 0),
        (914, "failed", None, None),
        (917, "failed", 0, 1),
    ],
)
def test_exit_status_and_late_no_update_failure_never_promote_score(
    request_factory, seed, status, known, unknown
):
    pin, _ = request_factory(seed=seed)
    receipt = runner.run_baseline(args(request_factory, pin))
    assert (
        receipt["status"] == status and receipt["metrics"]["simulation_steps"] == known
    )
    if unknown is not None:
        assert receipt["metrics"]["unknown_physical_attempts"] == unknown
    assert receipt["metrics"]["successes"] is None
    assert status == "incomplete" or "producer_evaluation_metrics" not in receipt
    assert receipt["worker_receipt"]["sha256"]


@pytest.mark.parametrize(
    "fault",
    [
        "cohort_count",
        "cohort_bool",
        "wrong_version",
        "wrong_source",
        "unresolved_lease",
        "nonfinite_timeout",
        "short_timeout",
        "alternate_results",
        "record_corruption",
        "source_corruption",
        "startup_code",
        "provisional_binding",
    ],
)
def test_invalid_admission_fails_before_query_or_dispatch(request_factory, fault):
    pin, value = request_factory()
    invocation = args(request_factory, pin)
    if fault in {"cohort_count", "cohort_bool"}:
        cohort = json.loads(Path(value["config"]["path"]).read_text())["cohort"]
        data = json.loads(Path(cohort["path"]).read_text())
        if fault == "cohort_count":
            data["episodes"].pop()
        else:
            data["episodes"][0]["seed"] = True
        Path(cohort["path"]).write_text(json.dumps(data))
    elif fault == "wrong_version":
        value["worker"]["nrh_version"] = "0.4.12"
    elif fault == "wrong_source":
        value["coordinator_sources"]["runner"] = "f" * 64
    elif fault == "unresolved_lease":
        ledger = request_factory.campaign / "gpu_budget.json"
        data = json.loads(ledger.read_text())
        data["active_parent_pid"] = 1
        ledger.write_text(json.dumps(data))
    elif fault == "nonfinite_timeout":
        invocation.timeout_seconds = float("nan")
    elif fault == "short_timeout":
        invocation.timeout_seconds = 0.5
    elif fault == "alternate_results":
        invocation.results = request_factory.campaign.parent / "shadow"
    elif fault in {"record_corruption", "source_corruption"}:
        target = (
            value["worker"]["record"]
            if fault == "record_corruption"
            else value["worker"]["nrh_files"]["nrh/__init__.py"]
        )
        Path(target["path"]).write_text("changed")
    elif fault == "startup_code":
        site = Path(value["worker"]["site_packages"])
        poison = site / "poison.pth"
        poison.write_text("import os; raise RuntimeError('must not execute')\n")
        value["worker"]["pth"] = [artifact(poison)]
    elif fault == "provisional_binding":
        proof = json.loads(Path(value["software_binding"]["path"]).read_text())
        proof["status"] = "provisional_unfrozen"
        value["software_binding"] = write(
            Path(value["software_binding"]["path"]), proof
        )
    pin = write(Path(pin["path"]), value)
    invocation.evaluation_request_bytes, invocation.evaluation_request_sha256 = (
        pin["bytes"],
        pin["sha256"],
    )
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        runner.run_baseline(invocation)
    assert request_factory.calls == []
    assert not list(request_factory.campaign.glob("visual-method-evaluation-*"))


def test_leader_exit_reaps_term_ignoring_descendant_before_release(request_factory):
    pin, _ = request_factory(seed=916)
    receipt = runner.run_baseline(args(request_factory, pin))
    pid = json.loads(
        (
            request_factory.campaign / receipt["run_id"] / "evaluation/descendant.json"
        ).read_text()
    )["pid"]
    assert not Path(f"/proc/{pid}").exists()
    assert receipt["charge_settlement"] == "actual_owned_lease_wall"


def test_pinned_healthy_resume_has_zero_new_controls_and_duplicate_prefix_denied(
    request_factory,
):
    pin, _ = request_factory()
    first = runner.run_baseline(args(request_factory, pin))
    pin, _ = request_factory(resume=[first["worker_receipt"]])
    second = runner.run_baseline(args(request_factory, pin))
    assert (
        second["metrics"]["simulation_steps"] == second["new_completed_episodes"] == 0
    )
    assert second["reused_completed_episodes"] == 50
    pin, _ = request_factory(resume=[first["worker_receipt"], first["worker_receipt"]])
    with pytest.raises(ValueError, match="Duplicate resume"):
        visual.load_request(pin, request_factory.campaign)


@pytest.mark.parametrize(
    "fault",
    [
        "Boolean_counter",
        "unknown_borrowed",
        "resumed_borrowed",
        "wrong_metrics",
        "late_error",
        "wrong_seed",
        "qf3_preservation",
        "duplicate_row",
    ],
)
def test_typed_reader_rejects_coherent_looking_credit_inflation(request_factory, fault):
    pin, _ = request_factory()
    result = runner.run_baseline(args(request_factory, pin))
    request = visual.load_request(pin, request_factory.campaign)
    raw = json.loads(Path(result["worker_receipt"]["path"]).read_text())
    if fault == "Boolean_counter":
        raw["counts"]["unknown_physical_attempts"] = False
    elif fault == "unknown_borrowed":
        raw["counts"]["unknown_physical_attempts"] = 1
        raw["counts"]["attempted_dispatches"] += 1
    elif fault == "resumed_borrowed":
        resume_pin, _ = request_factory(resume=[result["worker_receipt"]])
        request = visual.load_request(resume_pin, request_factory.campaign)
    elif fault == "wrong_metrics":
        raw["metrics"]["native_current_successes"] = 50
    elif fault == "late_error":
        raw["secondary_errors"] = ["late world close failed"]
    elif fault == "wrong_seed":
        raw["episodes"][0]["requested_seed"] = True
    elif fault == "qf3_preservation":
        raw["episodes"][0]["qf3_policy"]["preservation_after"] = "b" * 64
    else:
        raw["episodes"][1] = copy.deepcopy(raw["episodes"][0])
    with pytest.raises(ValueError):
        visual.summarize_worker(request, raw, 0)


def test_literal_interpreter_and_receipt_bound_are_explicit(request_factory):
    pin, value = request_factory()
    request = visual.load_request(pin, request_factory.campaign)
    command = visual.worker_command(
        request, Path("/fixture/config"), Path("/fixture/out"), request_factory.campaign
    )
    assert command[:5] == [
        str(request_factory.prefix / "bin/python"),
        "-I",
        "-B",
        "-m",
        "nrh.visual_method_evaluation",
    ]
    assert command[0] != value["worker"]["python_resolved"]
    assert request.value["receipt_max_bytes"] == 512 * 1024**2
    assert request.value["native_resources"]["virtual_address_limit_bytes"] is None


def test_oversize_worker_receipt_is_retained_without_parsing(
    request_factory, monkeypatch
):
    pin, value = request_factory()
    value["receipt_max_bytes"] = 100
    pin = write(Path(pin["path"]), value)
    original = visual.read_json
    reads = []

    def tracked(path, *limits):
        reads.append(str(path))
        return original(path, *limits)

    monkeypatch.setattr(visual, "read_json", tracked)
    result = runner.run_baseline(args(request_factory, pin))
    rejected = result["rejected_worker_receipt"]
    assert result["status"] == "failed" and "producer_evaluation_metrics" not in result
    assert rejected["observed_bytes"] > 100 and Path(rejected["path"]).is_file()
    assert rejected["path"] not in reads


def test_parent_deadline_cannot_be_extended_by_slow_occupancy_boundary(
    request_factory, monkeypatch
):
    import time

    pin, _ = request_factory()
    original = runner.inspect_reserved_gpu

    def slow(ordinal):
        time.sleep(0.3)
        return original(ordinal)

    monkeypatch.setattr(runner, "inspect_reserved_gpu", slow)
    with pytest.raises(ValueError, match="deadline was consumed"):
        runner.run_baseline(args(request_factory, pin, timeout=1.1))
    ledger = json.loads((request_factory.campaign / "gpu_budget.json").read_text())
    assert ledger["charged_seconds"] == 10 and "active_parent_pid" not in ledger


@pytest.mark.parametrize("field", ["error", "secondary_errors"])
def test_failed_error_fields_are_typed_before_counter_admission(request_factory, field):
    pin, _ = request_factory(seed=912)
    result = runner.run_baseline(args(request_factory, pin))
    request = visual.load_request(pin, request_factory.campaign)
    raw = json.loads(Path(result["worker_receipt"]["path"]).read_text())
    raw[field] = False
    with pytest.raises(ValueError, match="Typed worker"):
        visual.summarize_worker(request, raw, 1)


def test_late_no_update_failure_reports_physics_without_accepted_evaluation_controls(
    request_factory,
):
    pin, _ = request_factory(seed=913)
    result = runner.run_baseline(args(request_factory, pin))
    assert result["status"] == "failed" and "producer_evaluation_metrics" not in result
    assert result["metrics"]["simulation_steps"] == 50
    assert result["metrics"]["reported_complete_row_control_steps"] == 50
    assert result["metrics"]["accepted_complete_control_steps"] == 0
    assert result["metrics"]["discarded_partial_control_steps"] == 0


def test_late_parent_cleanup_failure_cannot_accept_completed_worker_controls(
    request_factory, monkeypatch
):
    from abc_bench import visual_evaluation_bootstrap as bootstrap

    original = bootstrap.finish_guard

    def fail_after_owned_cleanup(spec):
        original(spec)
        raise RuntimeError("owned fixture late cleanup failure")

    monkeypatch.setattr(bootstrap, "finish_guard", fail_after_owned_cleanup)
    pin, _ = request_factory()
    result = runner.run_baseline(args(request_factory, pin))
    assert result["worker_exit_code"] == 0 and result["worker_status"] == "completed"
    assert result["status"] == "failed" and "producer_evaluation_metrics" not in result
    assert result["metrics"]["reported_complete_row_control_steps"] == 50
    assert result["metrics"]["accepted_complete_control_steps"] == 0
    assert result["charge_settlement"] == "full_reservation_unresolved_no_refund"
