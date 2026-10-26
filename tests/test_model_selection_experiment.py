"""Controlled model selection experiment.

Verifies that historical performance evidence actually influences routing,
including overriding static cost/latency preferences when a model has a poor
historical record, and not blindly preferring a model with a single success
over one with many.
"""

from __future__ import annotations

import unittest

from divy.performance import PerformanceMemory, PerformanceObservation
from divy.providers import ModelMetadata, ModelProvider, ModelRegistry
from divy.routing import DeterministicRoutingPolicy, RoutingRequest
from divy.schemas import ModelResponse


class ConstAdapter:
    def __init__(self, value):
        self.value = value

    def generate(self, messages, tools, response_schema):
        return ModelResponse({"answer": self.value})


def _record_history(mem, model_id, task_type, successes, failures, cost, latency):
    for _ in range(successes):
        mem.record(PerformanceObservation(model_id=model_id, task_type=task_type,
            provider_success=True, contributed_to_final=True, task_success=True,
            verification_passed=True, latency_ms=latency, cost=cost, token_count=10,
            tool_calls=1, recovery_attempts=0, capabilities=(task_type,)))
    for _ in range(failures):
        mem.record(PerformanceObservation(model_id=model_id, task_type=task_type,
            provider_success=False, contributed_to_final=False, task_success=False,
            verification_passed=None, latency_ms=latency, cost=cost, token_count=10,
            tool_calls=1, recovery_attempts=0, capabilities=(task_type,)))


def _registry(fast_cheap, slow_expensive):
    return ModelRegistry([
        ModelProvider(ModelMetadata("fast-cheap", "p", frozenset({"coding"}), 4096,
            fast_cheap["cost"], fast_cheap["latency"], 0.9), ConstAdapter("fc")),
        ModelProvider(ModelMetadata("slow-expensive", "p", frozenset({"coding"}), 4096,
            slow_expensive["cost"], slow_expensive["latency"], 0.9), ConstAdapter("se")),
    ])


def _route(mem, registry, task_type="coding", empirical_weight=1.0, confidence_samples=10.0):
    policy = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=empirical_weight,
                                        confidence_samples=confidence_samples)
    return policy.route(RoutingRequest("t", frozenset({"coding"}), task_type=task_type), registry)


class ModelSelectionExperimentTests(unittest.TestCase):
    def test_historical_evidence_overrides_static_cost_preference(self):
        # fast-cheap is statically preferred (lower cost & latency) but has a
        # terrible historical record; slow-expensive has a strong record.
        mem = PerformanceMemory()
        _record_history(mem, "fast-cheap", "coding", successes=0, failures=10,
                        cost=0.001, latency=50)
        _record_history(mem, "slow-expensive", "coding", successes=50, failures=0,
                        cost=0.05, latency=300)
        registry = _registry({"cost": 0.001, "latency": 50}, {"cost": 0.05, "latency": 300})

        # Without memory, fast-cheap wins on static cost/latency.
        no_mem_policy = DeterministicRoutingPolicy()
        static_decision = no_mem_policy.route(RoutingRequest("t", frozenset({"coding"}), task_type="coding"), registry)
        self.assertEqual(static_decision.selected_model_ids, ["fast-cheap"])

        # With historical evidence, slow-expensive wins despite cost/latency.
        decision = _route(mem, registry)
        self.assertEqual(decision.selected_model_ids, ["slow-expensive"],
                         "historically strong model should win despite static cost")

    def test_more_evidence_wins_over_equal_rate(self):
        # Both models 100% historical success and identical static cost/latency,
        # but A has 1 observation and B has 100. Router should prefer B.
        mem = PerformanceMemory()
        _record_history(mem, "fast-cheap", "coding", successes=1, failures=0,
                        cost=0.01, latency=100)
        _record_history(mem, "slow-expensive", "coding", successes=100, failures=0,
                        cost=0.01, latency=100)
        registry = _registry({"cost": 0.01, "latency": 100}, {"cost": 0.01, "latency": 100})
        decision = _route(mem, registry)
        self.assertEqual(decision.selected_model_ids, ["slow-expensive"],
                         "model with more evidence should be preferred")

    def test_no_evidence_falls_back_to_static(self):
        # No historical records: routing follows static cost/latency only.
        mem = PerformanceMemory()
        registry = _registry({"cost": 0.001, "latency": 50}, {"cost": 0.05, "latency": 300})
        decision = _route(mem, registry)
        self.assertEqual(decision.selected_model_ids, ["fast-cheap"])

    def test_low_confidence_does_not_flip_decision(self):
        # With tiny samples, the empirical signal is heavily shrunk, so a
        # single success shouldn't overturn a clear static preference.
        mem = PerformanceMemory()
        _record_history(mem, "fast-cheap", "coding", successes=0, failures=0,
                        cost=0.001, latency=50)
        _record_history(mem, "slow-expensive", "coding", successes=1, failures=0,
                        cost=0.05, latency=300)
        registry = _registry({"cost": 0.001, "latency": 50}, {"cost": 0.05, "latency": 300})
        decision = _route(mem, registry, confidence_samples=100.0)
        self.assertEqual(decision.selected_model_ids, ["fast-cheap"],
                         "single success with low confidence should not override static preference")


if __name__ == "__main__":
    unittest.main()
