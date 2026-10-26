from __future__ import annotations

import unittest

from divy import OrchestrationConstraints, orchestrate
from divy.performance import PerformanceMemory, PerformanceObservation
from divy.providers import ModelMetadata, ModelProvider, ModelRegistry
from divy.routing import TaskProfile
from divy.schemas import EvaluationResult, ModelResponse
from divy.tools.registry import ToolRegistry


class ConstantAdapter:
    def __init__(self, content):
        self.content = content

    def generate(self, messages, tools, response_schema):
        return ModelResponse(self.content, {"total_tokens": 3})


def provider(model_id: str, content: dict) -> ModelProvider:
    return ModelProvider(
        ModelMetadata(model_id, "test", frozenset({"general"}), 4096, 0.1, 10.0, 0.8),
        ConstantAdapter(content),
    )


class PublicApiTests(unittest.TestCase):
    def test_structured_orchestration_owns_routing_and_returns_explanation(self):
        memory = PerformanceMemory()
        registry = ModelRegistry([provider("m1", {"answer": "ok"})])
        outcome = orchestrate(
            "answer the task",
            task_id="api-1",
            registry=registry,
            requirements=TaskProfile(task_type="analysis", required_capabilities=frozenset({"general"})),
            performance_memory=memory,
        )
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.accepted_model_id, "m1")
        self.assertEqual(outcome.decision.selected_model_ids, ["m1"])
        self.assertEqual(outcome.explanation["policy"], "deterministic-evidence-v2")
        self.assertEqual(outcome.as_dict()["accepted_model_id"], "m1")

    def test_structured_verifier_callback_gates_acceptance_and_fallback(self):
        registry = ModelRegistry([
            provider("m1", {"answer": "rejected"}),
            provider("m2", {"answer": "accepted"}),
        ])
        memory = PerformanceMemory()

        def verify(output):
            passed = output.get("answer") == "accepted"
            return EvaluationResult(success=passed, verification_passed=passed)

        outcome = orchestrate(
            "verify the answer",
            task_id="api-verified",
            registry=registry,
            performance_memory=memory,
            verifier=verify,
        )
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.accepted_model_id, "m2")
        self.assertTrue(outcome.execution.verification_attempted)
        self.assertTrue(outcome.execution.fallback_used)
        self.assertEqual(memory.summarize(task_type="api-verified")["verified_success_rate"], 1.0)

    def test_tool_loop_uses_routed_model_and_records_session_attempt(self):
        registry = ModelRegistry([provider("m1", {"final": {"answer": "done"}})])
        outcome = orchestrate(
            "finish the task",
            task_id="api-2",
            registry=registry,
            requirements=TaskProfile(required_capabilities=frozenset({"general"})),
            tool_registry=ToolRegistry([]),
        )
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.mode, "tool_loop")
        self.assertEqual(outcome.accepted_model_id, "m1")
        self.assertEqual(len(outcome.session_attempts), 1)

    def test_verification_required_without_verifier_fails_closed(self):
        registry = ModelRegistry([provider("m1", {"final": {"answer": "unverified"}})])
        outcome = orchestrate(
            "produce a critical result",
            task_id="api-3",
            registry=registry,
            requirements=TaskProfile(required_capabilities=frozenset({"general"}), verification_required=True),
            tool_registry=ToolRegistry([]),
        )
        self.assertFalse(outcome.success)
        self.assertIsNone(outcome.accepted_model_id)
        self.assertEqual(outcome.session_attempts[0].status, "failed_unverified")

    def test_performance_memory_task_rate_uses_execution_sample(self):
        memory = PerformanceMemory()
        for task_success in (False, False, True):
            memory.record(PerformanceObservation(
                model_id="m1",
                task_type="analysis",
                execution_id="exec-1",
                provider_success=True,
                task_success=task_success,
                contributed_to_final=task_success,
            ))
        summary = memory.summarize(model_id="m1", task_type="analysis")
        self.assertEqual(summary["observation_count"], 3)
        self.assertEqual(summary["execution_sample_size"], 1)
        self.assertEqual(summary["task_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
