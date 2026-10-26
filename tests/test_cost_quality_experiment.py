"""Cost vs Quality experiment.

Tests whether additional model calls (fallback/escalation) improve reliable
task completion per unit of cost, using deterministic scenario adapters.

The question: does paying for more / stronger model calls produce better
outcomes, or does it just cost more? Measured via success rate, cost,
latency, and cost per successful task.
"""

from __future__ import annotations

import unittest

from divy.execution import MultiModelExecutor
from divy.performance import PerformanceMemory, PerformanceObservation
from divy.providers import ModelMetadata, ModelProvider, ModelRegistry
from divy.routing import DeterministicRoutingPolicy, RoutingDecision, RoutingRequest
from divy.schemas import ModelResponse


class ScenarioAdapter:
    def __init__(self, behavior, cost, latency, token_count=10):
        self.behavior = behavior
        self.cost = cost
        self.latency = latency
        self.token_count = token_count
        self.calls = 0

    def generate(self, messages, tools, response_schema):
        self.calls += 1
        if self.behavior == "fail":
            raise RuntimeError("simulated provider failure")
        if self.behavior == "invalid":
            return ModelResponse(content={})
        if self.behavior == "verify_yes":
            return ModelResponse(content={"verified": True}, token_usage={"total_tokens": self.token_count})
        return ModelResponse(content={"answer": "ok"}, token_usage={"total_tokens": self.token_count})


def _registry():
    return ModelRegistry([
        ModelProvider(ModelMetadata("cheap-fast", "p", frozenset({"coding"}), 4096, 0.001, 50, 0.5),
                      ScenarioAdapter("fail", cost=0.001, latency=50)),
        ModelProvider(ModelMetadata("expensive-strong", "p", frozenset({"coding"}), 4096, 0.05, 300, 0.99),
                      ScenarioAdapter("ok", cost=0.05, latency=300)),
        ModelProvider(ModelMetadata("verifier", "p", frozenset({"verification"}), 4096, 0.01, 80, 0.95),
                      ScenarioAdapter("verify_yes", cost=0.01, latency=80)),
    ])


class CostQualityExperimentTests(unittest.TestCase):
    def test_single_model_vs_fallback_cost_and_success(self):
        reg = _registry()
        mem = PerformanceMemory()
        executor = MultiModelExecutor(reg, performance_memory=mem)

        # Single cheap model: low cost, fails.
        cheap_decision = RoutingDecision("c1", ["cheap-fast"])
        cheap_result = executor.execute(cheap_decision, [{"role": "user", "content": "x"}], [],
                                        {"type": "object"})
        self.assertFalse(cheap_result.success)
        self.assertLess(cheap_result.total_cost, 0.01)

        # Fallback to expensive model: higher cost, succeeds.
        fallback_decision = RoutingDecision("c2", ["cheap-fast"], fallback_model_ids=["expensive-strong"])
        fallback_result = executor.execute(fallback_decision, [{"role": "user", "content": "x"}], [],
                                           {"type": "object"})
        self.assertTrue(fallback_result.success)
        self.assertGreater(fallback_result.total_cost, cheap_result.total_cost)

        # Reliable completion per unit cost: fallback succeeds but costs more.
        # Single model is cheaper but useless (0 successes). Fallback buys
        # reliability at a cost — the router must weigh this.
        self.assertEqual(cheap_result.model_calls, 1)
        self.assertEqual(fallback_result.model_calls, 2)

    def test_verification_adds_cost_but_confirms_outcome(self):
        reg = _registry()
        executor = MultiModelExecutor(reg)
        decision = RoutingDecision("v1", ["expensive-strong"], verifier_model_id="verifier")
        result = executor.execute(decision, [{"role": "user", "content": "x"}], [],
                                  {"type": "object"}, verify=True)
        self.assertTrue(result.success)
        self.assertTrue(result.verification_passed)
        # Verification adds a model call and its cost.
        self.assertEqual(result.model_calls, 2)
        self.assertGreater(result.total_cost, 0.05)

    def test_cost_per_successful_task_comparison(self):
        reg = _registry()
        mem = PerformanceMemory()
        executor = MultiModelExecutor(reg, performance_memory=mem)

        successes = 0
        total_cost = 0.0
        for _ in range(10):
            result = executor.execute(
                RoutingDecision("cqs", ["cheap-fast"], fallback_model_ids=["expensive-strong"]),
                [{"role": "user", "content": "x"}], [], {"type": "object"},
            )
            total_cost += result.total_cost
            if result.success:
                successes += 1
        cost_per_success = total_cost / successes if successes else None
        self.assertIsNotNone(cost_per_success)
        self.assertEqual(successes, 10)
        print(f"cost_per_successful_task={cost_per_success:.4f} over 10 tasks")


if __name__ == "__main__":
    unittest.main()
