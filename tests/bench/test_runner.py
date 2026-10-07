import pytest

from abc_bench.runner import summarize


def summary(chunks=None):
    return {
        "worlds": [{"success": False, "steps": 30, "chunk_metrics": chunks or []}],
        "num_worlds": 1,
        "num_success": 0,
        "mean_reward": 0.0,
    }


def test_smoke_does_not_publish_short_run_as_success_benchmark():
    result = summarize(summary(), phase="smoke")
    assert result["successes"] is None
    assert result["episodes"] is None
    assert result["latency_ms"]["p50"] is None
    assert result["simulation_steps"] == 30


def test_recorded_zero_is_preserved_and_latency_aggregates():
    result = summarize(
        summary([{"current_chunk_infer_s": 0.1}, {"current_chunk_infer_s": 0.2}]),
        phase="benchmark",
    )
    assert result["successes"] == 0
    assert result["episodes"] == 1
    assert result["latency_ms"]["p50"] == 150


def test_inconsistent_success_evidence_rejected():
    value = summary()
    value["num_success"] = 1
    with pytest.raises(ValueError, match="Success count"):
        summarize(value, phase="benchmark")


def test_nonfinite_latency_rejected():
    with pytest.raises(ValueError, match="latency"):
        summarize(summary([{"current_chunk_infer_s": float("nan")}]), phase="benchmark")


def test_partial_log_excludes_unfinished_world(tmp_path):
    from abc_bench.runner import partial_log_metrics

    log = tmp_path / "run.log"
    log.write_text(
        "world=000 chunk=00 infer=100ms steps=5ms\n"
        "world=000 done success=True bottles=5/5 steps=30\n"
        "world=001 chunk=00 infer=900ms steps=5ms\n"
    )
    result = partial_log_metrics(log)
    assert result["successes"] == 1
    assert result["episodes"] == 1
    assert result["simulation_steps"] == 30
    assert result["latency_ms"]["p50"] == 100
    assert result["reward"] is None


def test_partial_log_requires_completed_world(tmp_path):
    from abc_bench.runner import partial_log_metrics

    log = tmp_path / "run.log"
    log.write_text("world=000 chunk=00 infer=100ms steps=5ms\n")
    assert partial_log_metrics(log) is None


def test_comparison_horizon_resolves_step_budget_and_label():
    from abc_bench.runner import comparison_plan

    plan = comparison_plan(2000)
    assert plan["horizon"] == 2000
    assert plan["maximum_steps"] == 18000
    assert "2000-step" in plan["method_fidelity"]
    assert comparison_plan(3540)["maximum_steps"] == 31860
    for invalid in (0, -1, 3541, 1.5, True):
        with pytest.raises(ValueError, match="horizon"):
            comparison_plan(invalid)
