from __future__ import annotations

import unittest

from divy.performance import PerformanceMemory, PerformanceObservation
from divy.providers import ModelMetadata, ModelProvider, ModelRegistry
from divy.routing import DeterministicRoutingPolicy, RoutingRequest, TaskProfile
from divy.schemas import ModelResponse


class Adapter:
    def generate(self, messages, tools, response_schema):
        return ModelResponse({"ok": True})


def provider(model_id: str, **kwargs) -> ModelProvider:
    metadata = ModelMetadata(
        model_id=model_id,
        provider=kwargs.pop("provider", model_id),
        capabilities=frozenset(kwargs.pop("capabilities", {"general"})),
        context_limit=kwargs.pop("context_limit", 4096),
        cost_estimate=kwargs.pop("cost", 0.1),
        latency_estimate_ms=kwargs.pop("latency", 100.0),
        reliability=kwargs.pop("reliability", 0.8),
        **kwargs,
    )
    return ModelProvider(metadata, Adapter())


class RoutingV2Tests(unittest.TestCase):
    def test_task_profile_filters_context_and_tool_requirements(self):
        registry = ModelRegistry([
            provider("small", capabilities={"general", "tool_calling"}, context_limit=1024, tool_calling=True),
            provider("large", capabilities={"general", "tool_calling"}, context_limit=8192, tool_calling=True),
        ])
        profile = TaskProfile(
            required_capabilities=frozenset({"general", "tool_calling"}),
            minimum_context=4096,
            requires_tool_calling=True,
        )
        decision = DeterministicRoutingPolicy().route(RoutingRequest("profile", profile=profile), registry)
        self.assertEqual(decision.selected_model_ids, ["large"])
        self.assertIn("small", decision.explanation["rejected"])
        self.assertIn("insufficient context window", decision.explanation["rejected"]["small"]["reasons"])

    def test_explanation_contains_ranked_evidence_and_failures(self):
        memory = PerformanceMemory()
        memory.record(PerformanceObservation(
            model_id="m", task_type="coding", capabilities=("coding",),
            provider_success=False, task_success=False, failure_type="timeout",
        ))
        registry = ModelRegistry([provider("m", capabilities={"coding"})])
        decision = DeterministicRoutingPolicy(memory, empirical_weight=1.0).route(
            RoutingRequest("t", frozenset({"coding"}), task_type="coding"), registry
        )
        entry = decision.explanation["candidates"][0]
        self.assertEqual(entry["sample_size"], 1)
        self.assertEqual(entry["failure_breakdown"], {"timeout": 1})
        self.assertEqual(decision.explanation["tie_break"], "score descending, model_id ascending")

    def test_seeded_exploration_is_reproducible_and_order_independent(self):
        models = [provider("a"), provider("b"), provider("c")]
        registry_a = ModelRegistry(models)
        registry_b = ModelRegistry(list(reversed(models)))
        request = RoutingRequest("t", frozenset({"general"}))
        first = DeterministicRoutingPolicy(exploration_rate=1.0, seed=7).route(request, registry_a)
        second = DeterministicRoutingPolicy(exploration_rate=1.0, seed=7).route(request, registry_b)
        self.assertEqual(first.selected_model_ids, second.selected_model_ids)

    def test_invalid_policy_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            DeterministicRoutingPolicy(recency_weight=1.1)
        with self.assertRaises(ValueError):
            DeterministicRoutingPolicy(confidence_samples=0)

    def test_model_metadata_rejects_invalid_operational_values(self):
        with self.assertRaises(ValueError):
            provider("bad", reliability=1.1)
        with self.assertRaises(ValueError):
            provider("bad", context_limit=0)


if __name__ == "__main__":
    unittest.main()
