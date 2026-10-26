"""Tests for the evaluation layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from divy.evaluation import (
    EvaluationReport,
    ExecutionMode,
    evaluate_execution,
    compare_routing_strategies,
    ExecutionMetrics,
    QualityMetrics,
    RoutingInfo,
    ModelContribution,
    EvidenceSummary,
)
from divy.dataset import RouterDatasetExample
from divy.performance import PerformanceMemory
from divy.performance.schemas import PerformanceObservation
from divy.schemas import EvaluationResult


class DummyTrajectory:
    """Helper to create test trajectories."""

    @staticmethod
    def successful_scripted() -> dict[str, Any]:
        return {
            "execution_id": "test-exec-1",
            "task_id": "RW01",
            "task_type": "software_engineering",
            "model_id": "scripted-rw01",
            "provider": "deterministic",
            "status": "success",
            "latency_ms": 123.4,
            "cost": 0.0,
            "tool_calls": 4,
            "model_calls": 5,
            "recovery_attempts": 0,
            "verification_result": True,
            "token_usage": {"input_tokens": 100, "output_tokens": 50},
            "events": [
                {"event": "model_response", "status": "success", "latency_ms": 50.0},
                {"event": "model_response", "status": "success", "latency_ms": 50.0},
            ],
        }

    @staticmethod
    def successful_live() -> dict[str, Any]:
        return {
            "execution_id": "live-exec-1",
            "task_id": "RW01",
            "task_type": "software_engineering",
            "model_id": "gpt-4",
            "provider": "openai",
            "status": "success",
            "latency_ms": 5000.0,
            "cost": 0.05,
            "tool_calls": 4,
            "model_calls": 3,
            "recovery_attempts": 0,
            "verification_result": True,
            "token_usage": {"input_tokens": 500, "output_tokens": 200},
            "events": [
                {"event": "model_response", "status": "success", "latency_ms": 2000.0, "token_usage": {"input_tokens": 200, "output_tokens": 100}},
                {"event": "model_response", "status": "success", "latency_ms": 3000.0, "token_usage": {"input_tokens": 300, "output_tokens": 100}},
            ],
        }

    @staticmethod
    def failed_verification() -> dict[str, Any]:
        return {
            "execution_id": "fail-exec-1",
            "task_id": "RW01",
            "task_type": "software_engineering",
            "model_id": "model-x",
            "provider": "test",
            "status": "failed_verification",
            "latency_ms": 1000.0,
            "cost": 0.02,
            "tool_calls": 3,
            "model_calls": 2,
            "recovery_attempts": 1,
            "verification_result": False,
            "token_usage": {"input_tokens": 100, "output_tokens": 50},
            "events": [
                {"event": "model_response", "status": "success", "latency_ms": 500.0},
                {"event": "model_response", "status": "failed", "error": {"type": "RuntimeError", "message": "timeout"}},
            ],
        }

    @staticmethod
    def with_fallback() -> dict[str, Any]:
        return {
            "execution_id": "fallback-exec-1",
            "task_id": "RW01",
            "task_type": "software_engineering",
            "model_id": "model-b",
            "provider": "test",
            "status": "success",
            "latency_ms": 2000.0,
            "cost": 0.03,
            "tool_calls": 3,
            "model_calls": 2,
            "recovery_attempts": 0,
            "verification_result": True,
            "fallback_used": True,
            "token_usage": {"input_tokens": 200, "output_tokens": 100},
            "events": [
                {"event": "model_response", "status": "failed", "error": {"type": "RuntimeError", "message": "timeout"}},
                {"event": "model_response", "status": "success", "latency_ms": 2000.0},
            ],
        }

    @staticmethod
    def provider_failure() -> dict[str, Any]:
        return {
            "execution_id": "provider-fail-1",
            "task_id": "RW01",
            "task_type": "software_engineering",
            "model_id": "bad-model",
            "provider": "test",
            "status": "failed_model",
            "latency_ms": 100.0,
            "cost": 0.01,
            "tool_calls": 0,
            "model_calls": 1,
            "recovery_attempts": 0,
            "verification_result": False,
            "token_usage": {},
            "events": [
                {"event": "model_response", "status": "failed", "error": {"type": "RuntimeError", "message": "API error"}, "latency_ms": 100.0},
            ],
        }

    @staticmethod
    def with_disagreement() -> dict[str, Any]:
        return {
            "execution_id": "disagree-1",
            "task_id": "RW01",
            "task_type": "software_engineering",
            "model_id": "ensemble",
            "provider": "test",
            "status": "success",
            "latency_ms": 3000.0,
            "cost": 0.04,
            "tool_calls": 4,
            "model_calls": 3,
            "recovery_attempts": 0,
            "verification_result": True,
            "token_usage": {"input_tokens": 300, "output_tokens": 150},
            "events": [
                {"event": "model_response", "status": "success", "latency_ms": 1000.0, "content": {"answer": "A"}},
                {"event": "model_response", "status": "success", "latency_ms": 1000.0, "content": {"answer": "B"}},
                {"event": "model_response", "status": "success", "latency_ms": 1000.0, "content": {"answer": "A"}},
            ],
        }

    @staticmethod
    def with_recovery() -> dict[str, Any]:
        return {
            "execution_id": "recovery-1",
            "task_id": "RW01",
            "task_type": "software_engineering",
            "model_id": "model-x",
            "provider": "test",
            "status": "success",
            "latency_ms": 2500.0,
            "cost": 0.02,
            "tool_calls": 5,
            "model_calls": 3,
            "recovery_attempts": 2,
            "verification_result": True,
            "token_usage": {"input_tokens": 250, "output_tokens": 120},
            "events": [
                {"event": "model_response", "status": "success", "latency_ms": 800.0},
                {"event": "model_response", "status": "success", "latency_ms": 800.0},
                {"event": "model_response", "status": "success", "latency_ms": 900.0},
            ],
        }


class EvaluationTests(unittest.TestCase):
    """Tests for the evaluation layer."""

    def setUp(self):
        self.routing_decision = {
            "selected_model_ids": ["model-a"],
            "fallback_model_ids": ["model-b"],
            "candidate_model_ids": ["model-a", "model-b", "model-c"],
            "rationale": "test rationale",
        }

    def test_scripted_trajectory_evaluation(self):
        """Test evaluation of a scripted trajectory."""
        traj = DummyTrajectory.successful_scripted()
        report = evaluate_execution(traj, routing_decision=self.routing_decision, execution_mode=ExecutionMode.scripted())

        self.assertEqual(report.execution_mode, ExecutionMode.scripted())
        self.assertEqual(report.task_id, "RW01")
        self.assertTrue(report.quality.task_success)
        self.assertTrue(report.quality.verification_passed)
        self.assertEqual(report.execution.model_calls, 5)
        self.assertEqual(report.execution.tool_calls, 4)
        self.assertEqual(report.routing.selected_models, ["model-a"])
        self.assertFalse(report.routing.fallback_used)
        self.assertEqual(len(report.model_contributions), 2)

    def test_live_trajectory_evaluation(self):
        """Test evaluation of a live trajectory."""
        traj = DummyTrajectory.successful_live()
        report = evaluate_execution(traj, routing_decision=self.routing_decision, execution_mode=ExecutionMode.live())

        self.assertEqual(report.execution_mode, ExecutionMode.live())
        self.assertTrue(report.quality.task_success)
        self.assertEqual(report.execution.model_calls, 3)
        self.assertEqual(report.execution.estimated_cost, 0.05)
        self.assertEqual(report.execution.input_tokens, 500)
        self.assertEqual(report.execution.output_tokens, 200)

    def test_token_total_is_not_double_counted(self):
        """total_tokens must be used as-is, not summed with its components."""
        traj = DummyTrajectory.successful_live()
        traj["token_usage"] = {
            "input_tokens": 3564,
            "output_tokens": 117,
            "total_tokens": 3681,
        }
        report = evaluate_execution(traj, routing_decision=self.routing_decision)
        self.assertEqual(report.execution.total_tokens, 3681)

        traj["token_usage"] = {"input_tokens": 10, "output_tokens": 5}
        report = evaluate_execution(traj, routing_decision=self.routing_decision)
        self.assertEqual(report.execution.total_tokens, 15)

    def test_missing_metrics_handled_gracefully(self):
        """Test that missing metrics are handled gracefully."""
        traj = DummyTrajectory.provider_failure()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertFalse(report.quality.task_success)
        self.assertEqual(report.execution.model_calls, 1)
        self.assertEqual(report.execution.tool_calls, 0)
        self.assertIsNone(report.execution.total_tokens)

    def test_provider_failure(self):
        """Test evaluation when provider fails."""
        traj = DummyTrajectory.provider_failure()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertFalse(report.quality.task_success)
        self.assertFalse(report.quality.verification_passed)
        self.assertEqual(report.execution.model_calls, 1)
        contrib = report.model_contributions[0]
        self.assertFalse(contrib.provider_success)
        self.assertEqual(contrib.failure_type, "RuntimeError")

    def test_tool_failure_recovery(self):
        """Test evaluation with tool failure and recovery."""
        traj = DummyTrajectory.with_recovery()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertTrue(report.quality.task_success)
        self.assertEqual(report.execution.recovery_attempts, 2)
        self.assertEqual(report.execution.model_calls, 3)

    def test_verification_failure(self):
        """Test evaluation when verification fails."""
        traj = DummyTrajectory.failed_verification()
        eval_result = EvaluationResult(False, False, {}, recoverable_failure=True)
        report = evaluate_execution(traj, evaluation_result=eval_result, routing_decision=self.routing_decision)

        self.assertFalse(report.quality.task_success)
        self.assertFalse(report.quality.verification_passed)
        # Trajectory status is "failed_verification" (not "success"), so false_acceptance should be False
        self.assertFalse(report.quality.false_acceptance)

    def test_fallback_used(self):
        """Test that fallback detection works."""
        traj = DummyTrajectory.with_fallback()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertTrue(report.routing.fallback_used)
        self.assertEqual(report.execution.model_calls, 2)
        self.assertEqual(len(report.model_contributions), 2)
        self.assertFalse(report.model_contributions[0].provider_success)
        self.assertTrue(report.model_contributions[1].provider_success)
        self.assertTrue(report.model_contributions[1].contributed_to_final)

    def test_disagreement_detection(self):
        """Test detection of model disagreement."""
        traj = DummyTrajectory.with_disagreement()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertTrue(report.quality.task_success)
        self.assertEqual(len(report.model_contributions), 3)

    def test_missing_metrics_none(self):
        """Test that truly missing metrics are None, not zero."""
        traj = {
            "execution_id": "minimal",
            "task_id": "TEST",
            "task_type": "test",
            "model_id": "model-x",
            "status": "success",
            "events": [],
        }
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertIsNone(report.execution.latency_ms)
        self.assertIsNone(report.execution.estimated_cost)
        self.assertIsNone(report.execution.total_tokens)
        self.assertEqual(report.execution.model_calls, 0)
        self.assertEqual(report.execution.tool_calls, 0)

    def test_multiple_models(self):
        """Test evaluation with multiple model calls."""
        traj = DummyTrajectory.successful_live()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertEqual(len(report.model_contributions), 2)
        for c in report.model_contributions:
            self.assertTrue(c.provider_success)

    def test_execution_mode_preserved(self):
        """Test that execution mode is correctly preserved in report."""
        traj = DummyTrajectory.successful_scripted()
        report = evaluate_execution(traj, execution_mode=ExecutionMode.scripted())
        self.assertEqual(report.execution_mode, ExecutionMode.scripted())

        report2 = evaluate_execution(traj, execution_mode=ExecutionMode.live())
        self.assertEqual(report2.execution_mode, ExecutionMode.live())

    def test_evidence_summary(self):
        """Test evidence summary extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = PerformanceMemory(Path(tmpdir) / "perf.jsonl")
            for _ in range(10):
                mem.record(PerformanceObservation(
                    model_id="model-a", task_type="software_engineering",
                    provider_success=True, contributed_to_final=True, task_success=True,
                    verification_passed=True, latency_ms=100.0, cost=0.01, token_count=10,
                    tool_calls=1, recovery_attempts=0, capabilities=("coding",),
                ))

            traj = DummyTrajectory.successful_scripted()
            report = evaluate_execution(
                traj,
                routing_decision={"selected_model_ids": ["model-a"], "fallback_model_ids": [], "candidate_model_ids": ["model-a"], "rationale": ""},
                performance_memory=mem,
            )

            self.assertEqual(report.evidence.sample_count, 10)
            self.assertEqual(report.evidence.historical_success_rate, 1.0)
            self.assertIsNotNone(report.evidence.uncertainty)

    def test_evidence_with_recent(self):
        """Test evidence summary with recency weighting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = PerformanceMemory(Path(tmpdir) / "perf.jsonl")
            # Old failures
            for _ in range(5):
                mem.record(PerformanceObservation(
                    model_id="model-x", task_type="software_engineering",
                    provider_success=False, contributed_to_final=False, task_success=False,
                    verification_passed=None, latency_ms=100.0, cost=0.01, token_count=10,
                    tool_calls=1, recovery_attempts=0, capabilities=("coding",),
                    timestamp="2024-01-01T00:00:00+00:00",
                ))
            # Recent successes
            for _ in range(10):
                mem.record(PerformanceObservation(
                    model_id="model-x", task_type="software_engineering",
                    provider_success=True, contributed_to_final=True, task_success=True,
                    verification_passed=True, latency_ms=100.0, cost=0.01, token_count=10,
                    tool_calls=1, recovery_attempts=0, capabilities=("coding",),
                    timestamp="2026-01-01T00:00:00+00:00",
                ))

            traj = DummyTrajectory.successful_scripted()
            report = evaluate_execution(
                traj,
                routing_decision={"selected_model_ids": ["model-x"], "fallback_model_ids": [], "candidate_model_ids": ["model-x"], "rationale": ""},
                performance_memory=PerformanceMemory(Path(tmpdir) / "perf.jsonl"),
            )

            self.assertEqual(report.evidence.sample_count, 15)
            self.assertAlmostEqual(report.evidence.historical_success_rate, 10/15, places=5)
            self.assertIsNotNone(report.evidence.recent_success_rate)
            self.assertGreater(report.evidence.recent_success_rate, report.evidence.historical_success_rate)

    def test_router_dataset_example_conversion(self):
        """Test conversion to RouterDatasetExample."""
        traj = DummyTrajectory.successful_scripted()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        example = report.to_router_dataset_example()

        self.assertIsInstance(example, RouterDatasetExample)
        self.assertEqual(example.task_id, "RW01")
        self.assertEqual(example.selected_model_id, "model-a")
        self.assertEqual(example.execution_success, True)
        self.assertTrue(example.verification_passed)
        self.assertEqual(example.model_calls, 5)
        self.assertEqual(example.tool_calls, 4)

    def test_dataset_conversion_preserves_raw(self):
        """Test that dataset conversion preserves raw observations."""
        traj = DummyTrajectory.successful_scripted()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)
        example = report.to_router_dataset_example()

        self.assertEqual(example.latency_ms, 123.4)
        self.assertEqual(example.estimated_cost, 0.0)
        self.assertEqual(example.model_calls, 5)
        self.assertEqual(example.tool_calls, 4)

    def test_historical_evidence_preserved(self):
        """Test that historical evidence is included in dataset conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = PerformanceMemory(Path(tmpdir) / "perf.jsonl")
            for _ in range(5):
                mem.record(PerformanceObservation(
                    model_id="model-a", task_type="software_engineering",
                    provider_success=True, contributed_to_final=True, task_success=True,
                    verification_passed=True, latency_ms=100.0, cost=0.01, token_count=10,
                    tool_calls=1, recovery_attempts=0, capabilities=("coding",),
                ))

            traj = DummyTrajectory.successful_scripted()
            report = evaluate_execution(
                traj,
                routing_decision={"selected_model_ids": ["model-a"], "fallback_model_ids": [], "candidate_model_ids": ["model-a"], "rationale": ""},
                performance_memory=mem,
            )

            example = report.to_router_dataset_example(
                historical_evidence={"model-a": mem.summarize(model_id="model-a", task_type="software_engineering")}
            )

            self.assertIn("model-a", example.historical_evidence)
            self.assertEqual(example.historical_evidence["model-a"]["observation_count"], 5)

    def test_comparison_function(self):
        """Test the routing strategy comparison function."""
        traj1 = DummyTrajectory.successful_scripted()
        traj2 = DummyTrajectory.with_fallback()
        traj3 = DummyTrajectory.failed_verification()

        report1 = evaluate_execution(traj1, routing_decision=self.routing_decision)
        report2 = evaluate_execution(traj2, routing_decision=self.routing_decision)
        report3 = evaluate_execution(traj3, routing_decision=self.routing_decision)

        comparison = compare_routing_strategies("RW01", "software_engineering", {
            "single_model": report1,
            "fixed_routing": report2,
            "dynamic_routing": report3,
        })

        self.assertIn("single_model", comparison["strategies"])
        self.assertIn("fixed_routing", comparison["strategies"])
        self.assertIn("dynamic_routing", comparison["strategies"])

        for strategy, metrics in comparison["strategies"].items():
            self.assertIn("success", metrics)
            self.assertIn("model_calls", metrics)
            self.assertIn("tool_calls", metrics)
            self.assertIn("latency_ms", metrics)
            self.assertIn("estimated_cost", metrics)
            self.assertIn("verification_passed", metrics)

    def test_raw_measurements_separated(self):
        """Test that raw measurements are kept separate from derived scores."""
        traj = DummyTrajectory.successful_scripted()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertIsInstance(report.execution.model_calls, int)
        self.assertIsInstance(report.execution.tool_calls, int)
        self.assertIsInstance(report.quality.task_success, bool)

        report_dict = report.to_dict()
        self.assertNotIn("intelligence_score", str(report_dict))
        self.assertNotIn("overall_score", str(report_dict))

    def test_execution_mode_identification(self):
        """Test that every report clearly identifies execution mode."""
        scripted = DummyTrajectory.successful_scripted()
        live = DummyTrajectory.successful_live()

        report1 = evaluate_execution(scripted, execution_mode=ExecutionMode.scripted())
        report2 = evaluate_execution(live, execution_mode=ExecutionMode.live())

        self.assertEqual(str(report1.execution_mode), "scripted")
        self.assertEqual(str(report2.execution_mode), "live")

        d1 = report1.to_dict()
        d2 = report2.to_dict()
        self.assertEqual(d1["execution_mode"], "scripted")
        self.assertEqual(d2["execution_mode"], "live")

    def test_model_contribution_fields(self):
        """Test that model contributions have all required fields."""
        traj = DummyTrajectory.successful_live()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        for contrib in report.model_contributions:
            self.assertIsNotNone(contrib.model_id)
            self.assertIsNotNone(contrib.provider)
            self.assertIsInstance(contrib.calls, int)
            self.assertIsInstance(contrib.provider_success, bool)
            self.assertIsInstance(contrib.contributed_to_final, bool)
            self.assertIsInstance(contrib.task_success, bool)
            self.assertIsNotNone(contrib.verification_passed)

    def test_evidence_fields(self):
        """Test that evidence summary has all required fields."""
        traj = DummyTrajectory.successful_scripted()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        self.assertIsInstance(report.evidence.sample_count, int)
        # When no performance_memory is provided, these are None
        self.assertIsNone(report.evidence.historical_success_rate)
        self.assertIsNone(report.evidence.recent_success_rate)
        self.assertIsNone(report.evidence.contribution_rate)
        self.assertIsNone(report.evidence.uncertainty)

    def test_evidence_without_task_type_does_not_drop_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = PerformanceMemory(Path(tmpdir) / "perf.jsonl")
            mem.record(PerformanceObservation(model_id="model-a", provider_success=True, task_success=True))
            report = evaluate_execution(
                {"task_id": "T", "status": "success"},
                routing_decision={"selected_model_ids": ["model-a"]},
                performance_memory=mem,
            )
        self.assertEqual(report.evidence.sample_count, 1)
        self.assertEqual(report.evidence.historical_success_rate, 1.0)

    def test_quality_metrics(self):
        """Test quality metrics calculation."""
        traj = DummyTrajectory.successful_scripted()
        eval_result = EvaluationResult(True, True, {"detail": "test"}, unnecessary_actions=2)
        report = evaluate_execution(traj, evaluation_result=eval_result, routing_decision=self.routing_decision)

        self.assertTrue(report.quality.task_success)
        self.assertTrue(report.quality.verification_passed)
        self.assertEqual(report.quality.unnecessary_actions, 2)
        self.assertTrue(report.quality.verification_accuracy)
        self.assertFalse(report.quality.false_acceptance)

    def test_false_acceptance_detection(self):
        """Test false acceptance detection."""
        traj = DummyTrajectory.successful_scripted()
        traj["status"] = "success"
        eval_result = EvaluationResult(False, False, {})  # Independent eval says failed
        report = evaluate_execution(traj, evaluation_result=eval_result, routing_decision=self.routing_decision)

        self.assertTrue(report.quality.false_acceptance)

    def test_verification_accuracy_detection(self):
        """Test verification accuracy detection."""
        traj = DummyTrajectory.successful_scripted()
        traj["verification_result"] = True
        eval_result = EvaluationResult(True, True, {})  # Independent eval agrees
        report = evaluate_execution(traj, evaluation_result=eval_result, routing_decision=self.routing_decision)

        self.assertTrue(report.quality.verification_accuracy)

        traj["verification_result"] = False
        report = evaluate_execution(traj, evaluation_result=eval_result, routing_decision=self.routing_decision)
        self.assertFalse(report.quality.verification_accuracy)

    def test_routing_info_preserved(self):
        """Test that routing information is fully preserved."""
        routing_decision = {
            "selected_model_ids": ["model-x"],
            "fallback_model_ids": ["model-y", "model-z"],
            "candidate_model_ids": ["model-x", "model-y", "model-z"],
            "rationale": "detailed reasoning here",
        }
        traj = DummyTrajectory.successful_scripted()
        report = evaluate_execution(traj, routing_decision=routing_decision)

        self.assertEqual(report.routing.selected_models, ["model-x"])
        self.assertEqual(report.routing.fallback_models, ["model-y", "model-z"])
        self.assertEqual(report.routing.candidate_models, ["model-x", "model-y", "model-z"])
        self.assertEqual(report.routing.rationale, "detailed reasoning here")

    def test_to_dict_serialization(self):
        """Test that report serializes to dict correctly."""
        traj = DummyTrajectory.successful_scripted()
        report = evaluate_execution(traj, routing_decision=self.routing_decision)

        d = report.to_dict()

        self.assertIn("execution_id", d)
        self.assertIn("task_id", d)
        self.assertIn("execution_mode", d)
        self.assertIn("routing", d)
        self.assertIn("execution", d)
        self.assertIn("quality", d)
        self.assertIn("model_contributions", d)
        self.assertIn("evidence", d)

        self.assertIsInstance(d["routing"], dict)
        self.assertIsInstance(d["execution"], dict)
        self.assertIsInstance(d["quality"], dict)
        self.assertIsInstance(d["model_contributions"], list)
        self.assertIsInstance(d["evidence"], dict)


if __name__ == "__main__":
    unittest.main()
