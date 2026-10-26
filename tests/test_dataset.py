"""Tests for the learned router dataset export and quality gate (points #11-#12)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from divy.dataset import apply_quality_gate, export_examples, example_from_execution


def _base_example(**overrides):
    data = dict(
        task_id="RW01",
        task_type="software_engineering",
        required_capabilities=["coding"],
        complexity="normal",
        available_models=["a", "b"],
        model_capabilities={"a": ["coding"], "b": ["coding"]},
        historical_evidence={
            "a": {"sample_size": 50, "success_rate": 0.9},
            "b": {"sample_size": 50, "success_rate": 0.5},
        },
        routing_decision={"selected_model_ids": ["a"], "rationale": "best historical"},
        selected_model_id="a",
        execution_success=True,
        verification_passed=True,
        final_outcome="success",
        latency_ms=120.0,
        estimated_cost=0.02,
        model_calls=1,
        tool_calls=2,
        recovery_attempts=0,
    )
    data.update(overrides)
    return example_from_execution(**data)


class DatasetExportTests(unittest.TestCase):
    def test_example_to_dict(self):
        ex = _base_example()
        d = ex.to_dict()
        self.assertEqual(d["task_id"], "RW01")
        self.assertEqual(d["selected_model_id"], "a")
        self.assertEqual(d["execution_success"], True)
        self.assertFalse(d["rejected"])
        self.assertEqual(d["reject_reasons"], [])

    def test_high_quality_example_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            summary = export_examples([_base_example()], path, include_rejected=False)
            self.assertEqual(summary["kept"], 1)
            self.assertEqual(summary["rejected"], 0)
            line = path.read_text(encoding="utf-8").strip()
            record = json.loads(line)
            self.assertEqual(record["task_id"], "RW01")

    def test_failed_verification_is_rejected(self):
        ex = apply_quality_gate(_base_example(verification_passed=False))
        self.assertTrue(ex.rejected)
        self.assertIn("failed_verification", ex.reject_reasons)

    def test_missing_verification_is_rejected(self):
        ex = apply_quality_gate(_base_example(verification_passed=None))
        self.assertTrue(ex.rejected)
        self.assertIn("missing_verification", ex.reject_reasons)

    def test_missing_outcome_is_rejected(self):
        ex = apply_quality_gate(_base_example(final_outcome=None))
        self.assertTrue(ex.rejected)
        self.assertIn("missing_outcome", ex.reject_reasons)

    def test_contradictory_success_outcome_is_rejected(self):
        ex = apply_quality_gate(
            _base_example(execution_success=True, final_outcome="failed_incomplete")
        )
        self.assertTrue(ex.rejected)
        self.assertIn("contradictory_success_outcome", ex.reject_reasons)

    def test_contradictory_failure_outcome_is_rejected(self):
        ex = apply_quality_gate(
            _base_example(execution_success=False, final_outcome="success")
        )
        self.assertTrue(ex.rejected)
        self.assertIn("contradictory_failure_outcome", ex.reject_reasons)

    def test_non_finite_metrics_are_rejected(self):
        ex = apply_quality_gate(_base_example(latency_ms=float("nan")))
        self.assertTrue(ex.rejected)
        self.assertIn("non_finite_latency", ex.reject_reasons)
        ex2 = apply_quality_gate(_base_example(estimated_cost=float("inf")))
        self.assertTrue(ex2.rejected)
        self.assertIn("non_finite_cost", ex2.reject_reasons)

    def test_negative_count_metrics_are_rejected(self):
        ex = apply_quality_gate(_base_example(model_calls=-1))
        self.assertTrue(ex.rejected)
        self.assertIn("negative_count_metric", ex.reject_reasons)

    def test_no_historical_evidence_is_rejected(self):
        ex = apply_quality_gate(
            _base_example(
                historical_evidence={"a": {"sample_size": 0, "success_rate": None}}
            )
        )
        self.assertTrue(ex.rejected)
        self.assertIn("no_historical_evidence_for_selected_model", ex.reject_reasons)

    def test_export_excludes_rejected_by_default(self):
        examples = [_base_example(), _base_example(verification_passed=False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            summary = export_examples(examples, path, include_rejected=False)
            self.assertEqual(summary["kept"], 1)
            self.assertEqual(summary["rejected"], 1)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["task_id"], "RW01")

    def test_export_includes_rejected_when_requested(self):
        examples = [_base_example(), _base_example(verification_passed=False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            summary = export_examples(examples, path, include_rejected=True)
            self.assertEqual(summary["kept"], 2)
            self.assertEqual(summary["rejected"], 0)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(records[1]["rejected"])

    def test_teacher_recommendation_distinct_from_outcome(self):
        ex = _base_example()
        # routing_decision is the teacher recommendation; execution_success /
        # verification_passed / final_outcome is the actual verified outcome.
        self.assertEqual(ex.routing_decision["selected_model_ids"], ["a"])
        self.assertEqual(ex.selected_model_id, "a")
        self.assertTrue(ex.execution_success)
        self.assertTrue(ex.verification_passed)
        self.assertEqual(ex.final_outcome, "success")


if __name__ == "__main__":
    unittest.main()
