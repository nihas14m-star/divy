from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from divy.orchestration_benchmark import run_orchestration_benchmarks
from divy.execution import MultiModelExecutor
from divy.performance import PerformanceMemory, PerformanceObservation
from divy.providers import ModelMetadata, ModelProvider, ModelRegistry
from divy.routing import DeterministicRoutingPolicy, RoutingRequest
from divy.routing import RoutingDecision
from divy.schemas import ModelResponse


class ConstantAdapter:
    def __init__(self, value: str) -> None:
        self.value = value

    def generate(self, messages, tools, response_schema):
        return ModelResponse({"answer": self.value})


class OrchestrationTests(unittest.TestCase):
    def test_o01_to_o10_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = run_orchestration_benchmarks(Path(directory))
        self.assertEqual([row["task_id"] for row in results], [f"O{i:02d}" for i in range(1, 11)])
        self.assertTrue(all(row["success"] for row in results), results)

    def test_registry_matches_capabilities_without_provider_names(self) -> None:
        registry = ModelRegistry([
            ModelProvider(ModelMetadata("local", "local-provider", frozenset({"coding"}), 2048, 0.1, 100, 0.8), ConstantAdapter("local")),
            ModelProvider(ModelMetadata("research", "remote-provider", frozenset({"research"}), 8192, 1.0, 200, 0.9), ConstantAdapter("research")),
        ])
        policy = DeterministicRoutingPolicy()
        decision = policy.route(RoutingRequest("test", frozenset({"coding"})), registry)
        self.assertEqual(decision.selected_model_ids, ["local"])
        self.assertEqual(registry.capability_matches({"coding"})["research"], ["coding"])

    def test_cost_aware_policy_does_not_always_choose_strongest(self) -> None:
        registry = ModelRegistry([
            ModelProvider(ModelMetadata("cheap", "p1", frozenset({"general"}), 2048, 0.1, 40, 0.8), ConstantAdapter("cheap")),
            ModelProvider(ModelMetadata("strong", "p2", frozenset({"general"}), 8192, 5.0, 150, 0.99), ConstantAdapter("strong")),
        ])
        decision = DeterministicRoutingPolicy().route(RoutingRequest("simple", frozenset({"general"}), complexity="simple", budget=1.0), registry)
        self.assertEqual(decision.selected_model_ids, ["cheap"])

    def test_executor_records_empirical_call_observations(self) -> None:
        registry = ModelRegistry([
            ModelProvider(ModelMetadata("local", "local", frozenset({"general"}), 2048, 0.1, 10, 0.8), ConstantAdapter("local")),
        ])
        memory = PerformanceMemory()
        result = MultiModelExecutor(registry, performance_memory=memory).execute(
            RoutingDecision("memory-test", ["local"]),
            [{"role": "user", "content": "test"}],
            [],
            {"type": "object"},
            task_type="general",
            required_capabilities={"general"},
        )
        self.assertTrue(result.success)
        summary = memory.summarize(model_id="local", task_type="general")
        self.assertEqual(summary["observation_count"], 1)
        self.assertEqual(summary["provider_success_rate"], 1.0)
        self.assertEqual(summary["contribution_rate"], 1.0)

    def test_empirical_performance_can_influence_future_routing(self) -> None:
        registry = ModelRegistry([
            ModelProvider(ModelMetadata("m1", "p1", frozenset({"coding"}), 2048, 1.0, 100, 0.9), ConstantAdapter("m1")),
            ModelProvider(ModelMetadata("m2", "p2", frozenset({"coding"}), 2048, 1.0, 100, 0.9), ConstantAdapter("m2")),
        ])
        memory = PerformanceMemory()
        memory.record(PerformanceObservation("m1", "coding", ("coding",), False, False, False, False, 100, 1.0, 10))
        memory.record(PerformanceObservation("m2", "coding", ("coding",), True, True, True, False, 100, 1.0, 10))
        decision = DeterministicRoutingPolicy(memory, empirical_weight=1.0).route(
            RoutingRequest("coding-task", frozenset({"coding"}), task_type="coding") , registry
        )
        self.assertEqual(decision.selected_model_ids, ["m2"])

    def test_executor_enforces_cost_limit_before_provider_call(self) -> None:
        registry = ModelRegistry([
            ModelProvider(ModelMetadata("expensive", "p1", frozenset({"general"}), 2048, 2.0, 10, 0.9), ConstantAdapter("expensive")),
            ModelProvider(ModelMetadata("fallback", "p2", frozenset({"general"}), 2048, 0.2, 10, 0.8), ConstantAdapter("fallback")),
        ])
        result = MultiModelExecutor(registry).execute(
            RoutingDecision("budget-test", ["expensive"], fallback_model_ids=["fallback"]),
            [],
            [],
            {"type": "object"},
            max_cost=0.5,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.calls[0].status, "skipped")
        error = result.calls[0].error or {}
        self.assertEqual(error.get("type"), "cost_limit")
        self.assertEqual(result.calls[-1].model_id, "fallback")


if __name__ == "__main__":
    unittest.main()
