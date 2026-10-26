"""P01-P10 performance evaluation scenarios for the performance subsystem.

These are deterministic, provider-neutral scenarios that exercise the
performance store, analytics, comparison, and trajectory-adapter layers.
Each scenario returns a dict of measurable results. They are deliberately
separate from the B01-B20 and O01-O10 benchmarks owned by the orchestration
stream.

Run with:  python -m divy.performance_benchmark
"""

from __future__ import annotations

from typing import Any

from .performance import (
    PerformanceMemory,
    PerformanceObservation,
    PerformanceAnalytics,
    ModelComparison,
    ModelScoreConfig,
    aggregate,
    trajectory_to_observation,
)


def _obs(
    *,
    model_id: str,
    task_type: str,
    provider_success: bool,
    contributed_to_final: bool | None = None,
    task_success: bool | None = None,
    verification_passed: bool | None = None,
    failure_type: str | None = None,
    latency_ms: float | None = None,
    cost: float | None = None,
    token_count: int | None = None,
    tool_calls: int = 0,
    recovery_attempts: int = 0,
    capabilities: tuple[str, ...] = (),
) -> PerformanceObservation:
    if contributed_to_final is None:
        contributed_to_final = provider_success
    if task_success is None:
        task_success = provider_success
    return PerformanceObservation(
        model_id=model_id,
        task_type=task_type,
        capabilities=capabilities,
        provider_success=provider_success,
        contributed_to_final=contributed_to_final,
        task_success=task_success,
        verification_passed=verification_passed,
        latency_ms=latency_ms,
        cost=cost,
        token_count=token_count,
        tool_calls=tool_calls,
        recovery_attempts=recovery_attempts,
        failure_type=failure_type,
    )


def _seed(memory: PerformanceMemory, observations: list[PerformanceObservation]) -> None:
    for o in observations:
        memory.record(o)


# --- P01: model reliability history -------------------------------------


def p01_model_reliability_history() -> dict[str, Any]:
    """Recorded executions yield a deterministic success/failure history."""
    memory = PerformanceMemory()
    obs = [
        _obs(model_id="alpha", task_type="coding", provider_success=True),
        _obs(model_id="alpha", task_type="coding", provider_success=True),
        _obs(model_id="alpha", task_type="coding", provider_success=False, failure_type="timeout"),
        _obs(model_id="alpha", task_type="coding", provider_success=False, failure_type="timeout"),
        _obs(model_id="alpha", task_type="coding", provider_success=True),
    ]
    _seed(memory, obs)
    analytics = PerformanceAnalytics(memory)
    stats = analytics.model_stats("alpha")
    return {
        "scenario": "P01",
        "sample_size": stats["sample_size"],
        "success_rate": stats["success_rate"],
        "failure_rate": stats["failure_rate"],
        "failure_breakdown": stats["failure_breakdown"],
        "assertions": {
            "sample_size_is_5": stats["sample_size"] == 5,
            "success_rate_is_0_6": stats["success_rate"] == 0.6,
            "failure_rate_is_0_4": stats["failure_rate"] == 0.4,
            "timeout_failures_are_2": stats["failure_breakdown"].get("timeout") == 2,
        },
    }


# --- P02: capability-specific performance --------------------------------


def p02_capability_specific_performance() -> dict[str, Any]:
    """Performance can be grouped by capability (coding, reasoning, ...)."""
    memory = PerformanceMemory()
    obs = [
        _obs(model_id="m", task_type="coding", provider_success=True, capabilities=("coding",)),
        _obs(model_id="m", task_type="coding", provider_success=False, capabilities=("coding",)),
        _obs(model_id="m", task_type="reasoning", provider_success=True, capabilities=("reasoning",)),
        _obs(model_id="m", task_type="reasoning", provider_success=True, capabilities=("reasoning", "coding")),
    ]
    _seed(memory, obs)
    analytics = PerformanceAnalytics(memory)
    coding = analytics.capability_stats("coding")
    reasoning = analytics.capability_stats("reasoning")
    return {
        "scenario": "P02",
        "coding_sample_size": coding["sample_size"],
        "coding_success_rate": coding["success_rate"],
        "reasoning_sample_size": reasoning["sample_size"],
        "reasoning_success_rate": reasoning["success_rate"],
        "assertions": {
            "coding_groups_3": coding["sample_size"] == 3,
            "coding_success_2_of_3": coding["success_rate"] == round(2 / 3, 6),
            "reasoning_groups_2": reasoning["sample_size"] == 2,
            "reasoning_success_1_0": reasoning["success_rate"] == 1.0,
        },
    }


# --- P03: small-sample handling ------------------------------------------


def p03_small_sample_handling() -> dict[str, Any]:
    """A single 100% success is labeled weak, never strong evidence."""
    memory = PerformanceMemory()
    memory.record(_obs(model_id="m", task_type="coding", provider_success=True))
    analytics = PerformanceAnalytics(memory)
    summary = analytics.model_summary_with_confidence("m")
    return {
        "scenario": "P03",
        "sample_size": summary["sample_size"],
        "success_rate": summary["success_rate"],
        "confidence": summary["confidence"],
        "assertions": {
            "one_sample": summary["sample_size"] == 1,
            "hundred_percent": summary["success_rate"] == 1.0,
            "label_is_very_weak": summary["confidence"] == "very weak (tiny sample)",
            "label_is_not_strong": summary["confidence"] != "strong",
        },
    }


# --- P04: model comparison -----------------------------------------------


def p04_model_comparison() -> dict[str, Any]:
    """Raw measurements plus optional derived ranking are separated."""
    memory = PerformanceMemory()
    obs = [
        _obs(model_id="model-a", task_type="coding", provider_success=True, cost=0.001, latency_ms=100.0),
        _obs(model_id="model-a", task_type="coding", provider_success=True, cost=0.002, latency_ms=200.0),
        _obs(model_id="model-b", task_type="coding", provider_success=False, cost=0.005, latency_ms=50.0),
    ]
    _seed(memory, obs)
    comparison = ModelComparison(PerformanceAnalytics(memory))
    raw = comparison.compare(["model-a", "model-b"], task_type="coding")
    scored = comparison.compare(
        ["model-a", "model-b"],
        task_type="coding",
        score_config=ModelScoreConfig(success_weight=1.0, cost_weight=-10.0),
    )
    return {
        "scenario": "P04",
        "model_a": raw["model-a"].raw_measurements,
        "model_b": raw["model-b"].raw_measurements,
        "model_a_score": scored["model-a"].derived_score,
        "model_a_rank": scored["model-a"].derived_rank,
        "model_b_rank": scored["model-b"].derived_rank,
        "assertions": {
            "a_success_1_0": raw["model-a"].raw_measurements["success_rate"] == 1.0,
            "b_success_0_0": raw["model-b"].raw_measurements["success_rate"] == 0.0,
            "raw_has_no_score": raw["model-a"].derived_score is None,
            "scored_ranks_are_1_and_2": {scored["model-a"].derived_rank, scored["model-b"].derived_rank} == {1, 2},
        },
    }


# --- P05: historical performance retrieval --------------------------------


def p05_historical_performance_retrieval() -> dict[str, Any]:
    """Recent performance returns the most recent observations, bounded."""
    memory = PerformanceMemory()
    for i in range(7):
        memory.record(_obs(model_id="m", task_type="coding", provider_success=(i % 2 == 0)))
    recent = PerformanceAnalytics(memory).recent(limit=4)
    ids_in_order = [o for o in recent]
    return {
        "scenario": "P05",
        "returned_count": len(recent),
        "assertions": {
            "limited_to_4": len(recent) == 4,
            "returns_most_recent": recent[-1] is memory.observations()[-1],
        },
    }


# --- P06: cost-performance comparison ------------------------------------


def p06_cost_performance_comparison() -> dict[str, Any]:
    """Cost per successful task is exposed for each model."""
    memory = PerformanceMemory()
    obs = [
        _obs(model_id="cheap", task_type="coding", provider_success=True, cost=0.001),
        _obs(model_id="cheap", task_type="coding", provider_success=True, cost=0.001),
        _obs(model_id="expensive", task_type="coding", provider_success=True, cost=0.020),
        _obs(model_id="expensive", task_type="coding", provider_success=True, cost=0.020),
    ]
    _seed(memory, obs)
    analytics = PerformanceAnalytics(memory)
    cheap = analytics.model_stats("cheap")
    expensive = analytics.model_stats("expensive")
    return {
        "scenario": "P06",
        "cheap_cost_per_success": cheap["cost_per_successful_task"],
        "expensive_cost_per_success": expensive["cost_per_successful_task"],
        "assertions": {
            "cheap_is_0_001": cheap["cost_per_successful_task"] == 0.001,
            "expensive_is_0_02": expensive["cost_per_successful_task"] == 0.02,
            "cheaper_is_less_expensive": cheap["cost_per_successful_task"] < expensive["cost_per_successful_task"],
        },
    }


# --- P07: latency-performance comparison ---------------------------------


def p07_latency_performance_comparison() -> dict[str, Any]:
    """Average and median latency are exposed per model."""
    memory = PerformanceMemory()
    obs = [
        _obs(model_id="fast", task_type="coding", provider_success=True, latency_ms=50.0),
        _obs(model_id="fast", task_type="coding", provider_success=True, latency_ms=150.0),
        _obs(model_id="slow", task_type="coding", provider_success=True, latency_ms=500.0),
        _obs(model_id="slow", task_type="coding", provider_success=True, latency_ms=700.0),
    ]
    _seed(memory, obs)
    analytics = PerformanceAnalytics(memory)
    fast = analytics.model_stats("fast")
    slow = analytics.model_stats("slow")
    return {
        "scenario": "P07",
        "fast_avg_latency": fast["average_latency_ms"],
        "fast_median_latency": fast["median_latency_ms"],
        "slow_avg_latency": slow["average_latency_ms"],
        "slow_median_latency": slow["median_latency_ms"],
        "assertions": {
            "fast_avg_is_100": fast["average_latency_ms"] == 100.0,
            "fast_median_is_100": fast["median_latency_ms"] == 100.0,
            "slow_avg_is_600": slow["average_latency_ms"] == 600.0,
            "slow_median_is_600": slow["median_latency_ms"] == 600.0,
        },
    }


# --- P08: recovery-performance analysis ----------------------------------


def p08_recovery_performance_analysis() -> dict[str, Any]:
    """Recovery success rate is computed among executions that recovered."""
    memory = PerformanceMemory()
    obs = [
        _obs(model_id="m", task_type="coding", provider_success=True, recovery_attempts=0),
        _obs(model_id="m", task_type="coding", provider_success=True, recovery_attempts=1),
        _obs(model_id="m", task_type="coding", provider_success=True, recovery_attempts=1),
        _obs(model_id="m", task_type="coding", provider_success=False, recovery_attempts=2),
    ]
    _seed(memory, obs)
    analytics = PerformanceAnalytics(memory)
    stats = analytics.model_stats("m")
    return {
        "scenario": "P08",
        "recovery_attempt_count": stats["recovery_attempt_count"],
        "recovery_success_rate": stats["recovery_success_rate"],
        "average_recovery_attempts": stats["average_recovery_attempts"],
        "assertions": {
            "three_executions_recovered": stats["recovery_attempt_count"] == 3,
            "two_of_three_recovery_succeeded": stats["recovery_success_rate"] == round(2 / 3, 6),
            "avg_recovery_is_one": stats["average_recovery_attempts"] == 1.0,
        },
    }


# --- P09: verification-performance analysis --------------------------------


def p09_verification_performance_analysis() -> dict[str, Any]:
    """Verification success rate excludes executions without verification."""
    memory = PerformanceMemory()
    obs = [
        _obs(model_id="m", task_type="coding", provider_success=True, verification_passed=True),
        _obs(model_id="m", task_type="coding", provider_success=True, verification_passed=False),
        _obs(model_id="m", task_type="coding", provider_success=False, verification_passed=None),
    ]
    _seed(memory, obs)
    analytics = PerformanceAnalytics(memory)
    stats = analytics.model_stats("m")
    return {
        "scenario": "P09",
        "verification_attempted_count": stats["verification_attempted_count"],
        "verification_success_rate": stats["verification_success_rate"],
        "assertions": {
            "two_verifications_attempted": stats["verification_attempted_count"] == 2,
            "half_verified": stats["verification_success_rate"] == 0.5,
        },
    }


# --- P10: stale/old performance data --------------------------------------


def p10_stale_data() -> dict[str, Any]:
    """Recent retrieval bounds to the newest observations regardless of age."""
    memory = PerformanceMemory()
    for i in range(5):
        memory.record(_obs(model_id="m", task_type="coding", provider_success=(i < 2)))
    recent = PerformanceAnalytics(memory).recent(limit=2)
    return {
        "scenario": "P10",
        "total": memory.count(),
        "returned": len(recent),
        "assertions": {
            "five_recorded": memory.count() == 5,
            "recent_bounded_to_2": len(recent) == 2,
        },
    }


def run_scenarios() -> list[dict[str, Any]]:
    return [
        p01_model_reliability_history(),
        p02_capability_specific_performance(),
        p03_small_sample_handling(),
        p04_model_comparison(),
        p05_historical_performance_retrieval(),
        p06_cost_performance_comparison(),
        p07_latency_performance_comparison(),
        p08_recovery_performance_analysis(),
        p09_verification_performance_analysis(),
        p10_stale_data(),
    ]


def main() -> int:
    results = run_scenarios()
    failures = []
    for result in results:
        scenario = result["scenario"]
        assertions = result.pop("assertions")
        for name, ok in assertions.items():
            if not ok:
                failures.append(f"{scenario}.{name}")
    print(f"P01-P10 scenarios: {len(results)}")
    if failures:
        print("FAILURES:", failures)
        return 1
    print("All P01-P10 scenario assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
