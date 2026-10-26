"""Tests for the DIVY Model Performance Intelligence + Evaluation subsystem.

Covers: recording, retrieval, aggregation, multiple models/providers/task
types, success/failure/verified/recovery paths, missing optional metrics,
zero-cost/zero-token tasks, tiny sample sizes, persistence/reload,
malformed input, and determinism.

The ``PerformanceObservation`` field order and names are the integration
contract consumed by ``divy.execution.MultiModelExecutor``.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from divy.performance import (
    PerformanceObservation,
    PerformanceMemory,
    PerformanceAnalytics,
    ModelComparison,
    ModelScoreConfig,
    PerformanceInterface,
    trajectory_to_observation,
    aggregate,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_observation(
    model_id="model-a",
    provider="provider-a",
    task_id="T1",
    task_type="coding",
    provider_success=True,
    task_success=True,
    verification_passed=None,
    failure_type=None,
    latency_ms=100.0,
    cost=0.001,
    token_count=30,
    tool_calls=2,
    recovery_attempts=0,
    capabilities=None,
    execution_id=None,
):
    return PerformanceObservation(
        model_id=model_id,
        task_type=task_type,
        capabilities=capabilities or ("coding",),
        provider_success=provider_success,
        contributed_to_final=provider_success,
        task_success=task_success,
        verification_passed=verification_passed,
        latency_ms=latency_ms,
        cost=cost,
        token_count=token_count,
        tool_calls=tool_calls,
        recovery_attempts=recovery_attempts,
        execution_id=execution_id,
        timestamp=_utc(),
        provider=provider,
        task_id=task_id,
        failure_type=failure_type,
    )


class PerformanceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "perf.jsonl"
        self.memory = PerformanceMemory(self.path)

    def tearDown(self):
        self.memory.close()
        self.dir.cleanup()

    def test_record_and_retrieve(self):
        obs = _make_observation()
        self.memory.record(obs)
        got = self.memory.observations()
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].model_id, "model-a")
        self.assertEqual(got[0].provider, "provider-a")
        self.assertEqual(got[0].task_id, "T1")
        self.assertEqual(got[0].task_type, "coding")
        self.assertTrue(got[0].task_success)
        self.assertEqual(got[0].capabilities, ("coding",))
        self.assertEqual(got[0].latency_ms, 100.0)
        self.assertEqual(got[0].cost, 0.001)

    def test_missing_optional_metrics_are_none(self):
        obs = PerformanceObservation(
            model_id="m",
            task_type="coding",
            capabilities=(),
            provider_success=False,
            task_success=False,
            failure_type="execution_failed",
        )
        self.memory.record(obs)
        got = self.memory.observations()[0]
        self.assertIsNone(got.latency_ms)
        self.assertIsNone(got.cost)
        self.assertIsNone(got.token_count)
        self.assertIsNone(got.verification_passed)
        self.assertEqual(got.tool_calls, 0)
        self.assertEqual(got.recovery_attempts, 0)

    def test_zero_cost_and_zero_tokens(self):
        obs = _make_observation(cost=0.0, token_count=0)
        self.memory.record(obs)
        got = self.memory.observations()[0]
        self.assertEqual(got.cost, 0.0)
        self.assertEqual(got.token_count, 0)

    def test_multiple_models_providers_task_types(self):
        records = [
            _make_observation(model_id="model-a", provider="prov-a", task_id="T1", task_type="coding", provider_success=True, cost=0.001),
            _make_observation(model_id="model-b", provider="prov-b", task_id="T2", task_type="reasoning", provider_success=False, cost=0.005),
            _make_observation(model_id="model-a", provider="prov-a", task_id="T3", task_type="coding", provider_success=False, failure_type="recovery_limit", cost=0.002),
        ]
        for r in records:
            self.memory.record(r)

        by_a = [o for o in self.memory.observations() if o.model_id == "model-a"]
        self.assertEqual(len(by_a), 2)
        by_b = [o for o in self.memory.observations() if o.provider == "prov-b"]
        self.assertEqual(len(by_b), 1)
        coding = [o for o in self.memory.observations() if o.task_type == "coding"]
        self.assertEqual(len(coding), 2)

    def test_summarize_success_and_failure(self):
        self.memory.record(_make_observation(provider_success=True, cost=0.001))
        self.memory.record(_make_observation(provider_success=True, cost=0.002))
        self.memory.record(_make_observation(provider_success=False, failure_type="recovery_limit", cost=0.003))
        summary = self.memory.summarize(model_id="model-a", task_type="coding")
        self.assertEqual(summary["observation_count"], 3)
        self.assertEqual(summary["provider_success_count"], 2)
        self.assertEqual(summary["provider_failure_count"], 1)
        self.assertEqual(summary["provider_success_rate"], round(2 / 3, 6))

    def test_aggregation_is_deterministic(self):
        for i in range(10):
            self.memory.record(_make_observation(provider_success=(i % 2 == 0), cost=0.001 * (i + 1)))
        s1 = aggregate(self.memory.observations())
        s2 = aggregate(self.memory.observations())
        self.assertEqual(s1, s2)

    def test_empty_stats(self):
        stats = aggregate([])
        self.assertEqual(stats["sample_size"], 0)
        self.assertIsNone(stats["success_rate"])
        self.assertIsNone(stats["average_latency_ms"])

    def test_persistence_reload(self):
        obs = _make_observation()
        self.memory.record(obs)
        self.memory.close()
        reloaded = PerformanceMemory(self.path)
        self.assertEqual(len(reloaded.observations()), 1)
        self.assertEqual(reloaded.observations()[0].model_id, "model-a")
        reloaded.close()

    def test_count(self):
        self.assertEqual(self.memory.count(), 0)
        self.memory.record(_make_observation())
        self.assertEqual(self.memory.count(), 1)

    def test_append_never_overwrites(self):
        self.memory.record(_make_observation(execution_id="e1"))
        self.memory.record(_make_observation(execution_id="e2"))
        self.assertEqual(len(self.memory.observations()), 2)

    def test_reload_skips_valid_json_that_is_not_an_observation_object(self):
        self.path.write_text("[]\n{\"model_id\": \"ok\"}\n", encoding="utf-8")
        reloaded = PerformanceMemory(self.path)
        self.assertEqual(reloaded.count(), 1)
        self.assertEqual(reloaded.observations()[0].model_id, "ok")


class PerformanceAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.memory = PerformanceMemory(Path(self.dir.name) / "perf.jsonl")
        self.analytics = PerformanceAnalytics(self.memory)

    def tearDown(self):
        self.memory.close()
        self.dir.cleanup()

    def _seed(self, n=10, success_rate=0.8):
        for i in range(n):
            self.memory.record(_make_observation(provider_success=(i < int(n * success_rate)), cost=0.001))

    def test_model_summary_with_confidence(self):
        self._seed(n=50, success_rate=0.8)
        summary = self.analytics.model_summary_with_confidence("model-a")
        self.assertEqual(summary["sample_size"], 50)
        self.assertEqual(summary["confidence"], "moderate")
        self.assertEqual(summary["success_rate"], 0.8)
        self.assertIsNotNone(summary["success_rate_confidence_interval"])

    def test_confidence_tiny_sample(self):
        self.memory.record(_make_observation(provider_success=True))
        summary = self.analytics.model_summary_with_confidence("model-a")
        self.assertEqual(summary["sample_size"], 1)
        self.assertEqual(summary["confidence"], "very weak (tiny sample)")
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertNotEqual(summary["confidence"], "strong")

    def test_confidence_no_evidence(self):
        summary = self.analytics.model_summary_with_confidence("ghost")
        self.assertEqual(summary["sample_size"], 0)
        self.assertEqual(summary["confidence"], "no evidence")
        self.assertIsNone(summary["success_rate"])

    def test_label_confidence_bounds(self):
        self.assertEqual(self.analytics.label_confidence(0), "no evidence")
        self.assertEqual(self.analytics.label_confidence(3), "very weak (tiny sample)")
        self.assertEqual(self.analytics.label_confidence(20), "weak (small sample)")
        self.assertEqual(self.analytics.label_confidence(50), "moderate")
        self.assertEqual(self.analytics.label_confidence(200), "strong")

    def test_wilson_interval(self):
        ci = self.analytics.success_rate_confidence_interval(0, 10)
        self.assertIsNotNone(ci)
        ci2 = self.analytics.success_rate_confidence_interval(10, 10)
        self.assertIsNotNone(ci2)
        ci3 = self.analytics.success_rate_confidence_interval(0, 0)
        self.assertIsNone(ci3)

    def test_capability_stats(self):
        self.memory.record(_make_observation(capabilities=("coding", "research")))
        self.memory.record(_make_observation(capabilities=("research", "reasoning")))
        stats = self.analytics.capability_stats("coding")
        self.assertEqual(stats["sample_size"], 1)
        stats2 = self.analytics.capability_stats("research")
        self.assertEqual(stats2["sample_size"], 2)
        stats3 = self.analytics.capability_stats("reasoning")
        self.assertEqual(stats3["sample_size"], 1)

    def test_task_type_stats(self):
        self.memory.record(_make_observation(task_type="coding", provider_success=True))
        self.memory.record(_make_observation(task_type="reasoning", provider_success=False))
        stats = self.analytics.task_type_stats("coding")
        self.assertEqual(stats["sample_size"], 1)
        self.assertEqual(stats["success_rate"], 1.0)
        stats2 = self.analytics.task_type_stats("reasoning")
        self.assertEqual(stats2["success_rate"], 0.0)


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.memory = PerformanceMemory(Path(self.dir.name) / "perf.jsonl")
        self.analytics = PerformanceAnalytics(self.memory)
        self.comparison = ModelComparison(self.analytics)

    def tearDown(self):
        self.memory.close()
        self.dir.cleanup()

    def test_compare_models_raw_only(self):
        self.memory.record(_make_observation(model_id="model-a", provider_success=True, cost=0.001, latency_ms=100.0))
        self.memory.record(_make_observation(model_id="model-a", provider_success=True, cost=0.002, latency_ms=200.0))
        self.memory.record(_make_observation(model_id="model-b", provider_success=False, cost=0.005, latency_ms=50.0))
        result = self.comparison.compare(["model-a", "model-b"], task_type="coding")
        self.assertIn("model-a", result)
        self.assertIn("model-b", result)
        a = result["model-a"]
        self.assertEqual(a.sample_size, 2)
        self.assertEqual(a.raw_measurements["success_rate"], 1.0)
        self.assertEqual(a.raw_measurements["average_latency_ms"], 150.0)
        self.assertEqual(a.raw_measurements["average_cost"], 0.0015)
        b = result["model-b"]
        self.assertEqual(b.sample_size, 1)
        self.assertEqual(b.raw_measurements["success_rate"], 0.0)
        self.assertIsNone(a.derived_score)
        self.assertIsNone(a.derived_rank)

    def test_compare_models_with_score(self):
        self.memory.record(_make_observation(model_id="model-a", provider_success=True, cost=0.001, latency_ms=100.0))
        self.memory.record(_make_observation(model_id="model-b", provider_success=True, cost=0.01, latency_ms=500.0))
        config = ModelScoreConfig(success_weight=1.0, latency_weight=-0.001, cost_weight=-10.0)
        result = self.comparison.compare(["model-a", "model-b"], score_config=config)
        self.assertIsNotNone(result["model-a"].derived_score)
        self.assertIsNotNone(result["model-a"].derived_rank)
        ranks = {m: r.derived_rank for m, r in result.items()}
        self.assertEqual(set(ranks.values()), {1, 2})

    def test_compare_models_task_type_filter(self):
        self.memory.record(_make_observation(model_id="model-a", task_type="coding", provider_success=True))
        self.memory.record(_make_observation(model_id="model-a", task_type="reasoning", provider_success=False))
        result = self.comparison.compare(["model-a"], task_type="coding")
        self.assertEqual(result["model-a"].sample_size, 1)
        result2 = self.comparison.compare(["model-a"], task_type="planning")
        self.assertEqual(result2["model-a"].sample_size, 0)

    def test_missing_model_returns_zero_samples(self):
        result = self.comparison.compare(["ghost"])
        self.assertEqual(result["ghost"].sample_size, 0)
        self.assertIsNone(result["ghost"].raw_measurements["success_rate"])


class PerformanceInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.memory = PerformanceMemory(Path(self.dir.name) / "perf.jsonl")
        self.interface = PerformanceInterface(self.memory)

    def tearDown(self):
        self.memory.close()
        self.dir.cleanup()

    def test_model_evidence(self):
        self.memory.record(_make_observation(provider_success=True, cost=0.002))
        evidence = self.interface.model_evidence("model-a")
        self.assertEqual(evidence.model_id, "model-a")
        self.assertEqual(evidence.sample_size, 1)
        self.assertEqual(evidence.success_rate, 1.0)
        self.assertEqual(evidence.average_cost, 0.002)
        self.assertEqual(evidence.confidence, "very weak (tiny sample)")

    def test_get_best_historical_candidates_min_samples(self):
        self.memory.record(_make_observation(model_id="model-a", provider_success=True, cost=0.001))
        self.memory.record(_make_observation(model_id="model-b", provider_success=False, cost=0.002))
        for i in range(2, 6):
            self.memory.record(_make_observation(model_id="model-a", provider_success=True, cost=0.001))
        candidates = self.interface.get_best_historical_candidates(["model-a", "model-b"], min_samples=3)
        ids = [c.model_id for c in candidates.candidates]
        self.assertIn("model-a", ids)
        self.assertNotIn("model-b", ids)
        self.assertEqual(candidates.note, "1/2 models met min_samples=3")

    def test_compare_returns_no_derived_rank_without_score_config(self):
        self.memory.record(_make_observation(model_id="model-a", provider_success=True))
        result = self.interface.compare(["model-a"])
        self.assertIsNone(result["model-a"]["derived_score"])
        self.assertIsNone(result["model-a"]["derived_rank"])


class TrajectoryAdapterTests(unittest.TestCase):
    def test_from_trajectory_success(self):
        trajectory = {
            "task_id": "B04",
            "status": "success",
            "tool_calls": 3,
            "recovery_attempts": 0,
            "latency_ms": 16.5,
            "token_usage": {"output_tokens": 181},
            "events": [],
        }
        result = {
            "success": True,
            "model_calls": 3,
            "total_cost": 0.005,
            "total_latency_ms": 16.5,
            "verification_passed": True,
        }
        obs = trajectory_to_observation(
            model_id="model-a",
            provider="provider-a",
            task_id="B04",
            task_type="coding",
            trajectory=trajectory,
            result=result,
        )
        self.assertTrue(obs.task_success)
        self.assertTrue(obs.verification_passed)
        self.assertEqual(obs.tool_calls, 3)
        self.assertEqual(obs.recovery_attempts, 0)
        self.assertEqual(obs.latency_ms, 16.5)
        self.assertEqual(obs.output_tokens, 181)
        self.assertEqual(obs.cost, 0.005)
        self.assertEqual(obs.final_outcome, "success")

    def test_from_trajectory_failure(self):
        trajectory = {
            "task_id": "B13",
            "status": "failed_incomplete",
            "tool_calls": 3,
            "recovery_attempts": 1,
            "latency_ms": 26.6,
            "token_usage": {},
            "events": [],
        }
        obs = trajectory_to_observation(
            model_id="model-x",
            provider="provider-x",
            task_id="B13",
            task_type="analysis",
            trajectory=trajectory,
        )
        self.assertFalse(obs.task_success)
        self.assertEqual(obs.failure_type, "incomplete")
        self.assertEqual(obs.recovery_attempts, 1)
        self.assertEqual(obs.final_outcome, "failed_incomplete")
        self.assertIsNone(obs.cost)

    def test_from_result_only(self):
        obs = trajectory_to_observation(
            model_id="m",
            provider="p",
            task_id="t",
            task_type="coding",
            result={"success": False, "model_calls": 2, "total_cost": 0.0},
        )
        self.assertFalse(obs.task_success)
        self.assertEqual(obs.cost, 0.0)
        self.assertEqual(obs.final_outcome, "failed")

    def test_missing_optional_left_none(self):
        obs = trajectory_to_observation(
            model_id="m",
            provider="p",
            task_id="t",
            result={"success": True},
        )
        self.assertTrue(obs.task_success)
        self.assertIsNone(obs.latency_ms)
        self.assertIsNone(obs.input_tokens)
        self.assertIsNone(obs.output_tokens)
        self.assertIsNone(obs.cost)
        self.assertIsNone(obs.verification_passed)

    def test_token_total_prefers_provider_total_without_double_counting(self):
        obs = trajectory_to_observation(
            model_id="m",
            provider="p",
            task_id="t",
            task_type="coding",
            trajectory={"token_usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}},
        )
        self.assertEqual(obs.token_count, 7)
        self.assertEqual(obs.input_tokens, 4)
        self.assertEqual(obs.output_tokens, 3)

    def test_agent_session_observations_do_not_split_aggregate_cost(self):
        from divy.performance.trajectory import observations_from_agent_session

        observations = observations_from_agent_session({
            "execution_id": "e",
            "model_id": "m",
            "provider": "p",
            "task_id": "t",
            "task_type": "coding",
            "status": "success",
            "cost": 1.0,
            "events": [
                {"event": "model_response", "status": "success", "token_usage": {"total_tokens": 7}},
                {"event": "model_response", "status": "success", "token_usage": {"total_tokens": 8}},
            ],
        })
        self.assertEqual([item.cost for item in observations], [None, None])
        self.assertEqual([item.token_count for item in observations], [7, 8])

    def test_positional_construction_matches_integration_contract(self):
        # Mirrors how test_adapters.py and execution.py construct the record.
        obs = PerformanceObservation("m1", "coding", ("coding",), True, True, True, False, 10, 0.2, 12)
        self.assertEqual(obs.model_id, "m1")
        self.assertEqual(obs.task_type, "coding")
        self.assertEqual(obs.capabilities, ("coding",))
        self.assertTrue(obs.provider_success)
        self.assertEqual(obs.latency_ms, 10)
        self.assertEqual(obs.cost, 0.2)
        self.assertEqual(obs.token_count, 12)


if __name__ == "__main__":
    unittest.main()
