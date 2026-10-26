"""Tests for confidence-aware historical routing.

Verifies the router does not treat a single successful observation as
equivalent to hundreds of successful observations (sample-size awareness).
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


def _build_memory(model_id, n_successes, n_total, task_type="coding", cost=0.01, latency=100.0):
    mem = PerformanceMemory()
    for _ in range(n_successes):
        mem.record(PerformanceObservation(model_id=model_id, task_type=task_type,
            provider_success=True, contributed_to_final=True, task_success=True,
            verification_passed=True, latency_ms=latency, cost=cost, token_count=10,
            tool_calls=1, recovery_attempts=0, capabilities=(task_type,)))
    for _ in range(n_total - n_successes):
        mem.record(PerformanceObservation(model_id=model_id, task_type=task_type,
            provider_success=False, contributed_to_final=False, task_success=False,
            verification_passed=None, latency_ms=latency, cost=cost, token_count=10,
            tool_calls=1, recovery_attempts=0, capabilities=(task_type,)))
    return mem


def _registry(model_id, cost=0.01, latency=100.0):
    return ModelRegistry([
        ModelProvider(ModelMetadata(model_id, "p", frozenset({"coding"}), 4096, cost, latency, 0.9), ConstAdapter("x")),
    ])


def _score(mem, model_id, task_type="coding", empirical_weight=1.0, confidence_samples=10.0):
    reg = _registry(model_id)
    policy = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=empirical_weight,
                                        confidence_samples=confidence_samples)
    request = RoutingRequest("t", frozenset({"coding"}), task_type=task_type)
    provider = reg.get(model_id)
    return policy._score(provider, request)


class ConfidenceRoutingTests(unittest.TestCase):
    def test_many_successes_outweigh_single_success(self):
        score_one = _score(_build_memory("a", 1, 1), "a")
        score_hundred = _score(_build_memory("b", 100, 100), "b")
        self.assertGreater(score_hundred, score_one,
                           "100 successes must score higher than 1 success")

    def test_single_success_is_shrunk_toward_static(self):
        # With one observation the empirical signal is heavily discounted.
        mem = _build_memory("a", 1, 1)
        reg = _registry("a")
        policy = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=1.0,
                                            confidence_samples=10.0)
        provider = reg.get("a")
        # Static score (no memory) for reference.
        policy_no_mem = DeterministicRoutingPolicy()
        static = policy_no_mem._score(provider, RoutingRequest("t", frozenset({"coding"}), task_type="coding"))
        scored = policy._score(provider, RoutingRequest("t", frozenset({"coding"}), task_type="coding"))
        # The one-observation score should be much closer to static than to the
        # full +0.5 empirical bonus.
        self.assertAlmostEqual(scored, static, delta=0.1)

    def test_no_memory_uses_static_score(self):
        reg = _registry("a")
        policy = DeterministicRoutingPolicy()  # no performance_memory
        provider = reg.get("a")
        score = policy._score(provider, RoutingRequest("t", frozenset({"coding"}), task_type="coding"))
        # Deterministic static score: reliability 0.9 - cost 0.0025 - latency 0.01
        self.assertAlmostEqual(score, 0.8875, places=4)

    def test_zero_observations_uses_static_score(self):
        mem = PerformanceMemory()  # empty
        reg = _registry("a")
        policy = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=1.0)
        provider = reg.get("a")
        policy_no_mem = DeterministicRoutingPolicy()
        static = policy_no_mem._score(provider, RoutingRequest("t", frozenset({"coding"}), task_type="coding"))
        scored = policy._score(provider, RoutingRequest("t", frozenset({"coding"}), task_type="coding"))
        self.assertEqual(scored, static)

    def test_higher_confidence_samples_shrinks_more(self):
        score_low = _score(_build_memory("a", 1, 1), "a", confidence_samples=1.0)
        score_high = _score(_build_memory("a", 1, 1), "a", confidence_samples=100.0)
        # A single success should score closer to static (lower bonus) with more shrinkage.
        # Static score is 0.8875. Both should be near static; higher confidence_samples = more shrink.
        self.assertLess(score_high, 0.95)
        self.assertLess(abs(score_high - 0.8875), abs(score_low - 0.8875))

    def test_backward_compatible_default(self):
        # Default constructor still works (no memory) and default confidence_samples exists.
        policy = DeterministicRoutingPolicy()
        self.assertEqual(policy.confidence_samples, 10.0)
        self.assertEqual(policy.empirical_weight, 0.3)
        self.assertEqual(policy.recency_half_life_days, 0.0)
        self.assertEqual(policy.recency_weight, 0.0)
        self.assertEqual(policy.exploration_rate, 0.0)

    def test_exploration_disabled_by_default(self):
        # With exploration_rate=0, should always pick highest-scored
        mem = _build_memory("a", 10, 10)  # strong evidence for a
        _build_memory("b", 0, 10)  # no evidence for b
        mem_b = _build_memory("b", 0, 10)
        reg = ModelRegistry([
            ModelProvider(ModelMetadata("a", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
            ModelProvider(ModelMetadata("b", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
        ])
        policy = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=1.0)
        request = RoutingRequest("t", frozenset({"coding"}), task_type="coding")
        decision = policy.route(request, reg)
        # Should always select "a" (has evidence)
        self.assertEqual(decision.selected_model_ids, ["a"])

    def test_exploration_can_select_weaker_model(self):
        # With exploration_rate=1.0, should sometimes pick weaker model
        mem_a = _build_memory("a", 10, 10)
        mem_b = _build_memory("b", 0, 10)
        mem = PerformanceMemory()
        for obs in mem_a.observations() + mem_b.observations():
            mem.record(obs)
        reg = ModelRegistry([
            ModelProvider(ModelMetadata("a", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
            ModelProvider(ModelMetadata("b", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
        ])
        policy = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=1.0, exploration_rate=1.0)
        request = RoutingRequest("t", frozenset({"coding"}), task_type="coding")
        # Run multiple times - with exploration_rate=1.0, should sometimes pick "b"
        selections = []
        for _ in range(50):
            decision = policy.route(request, reg)
            selections.append(decision.selected_model_ids[0])
        # With 1.0 exploration rate, should sometimes pick "b" (from exploration pool)
        self.assertIn("a", selections)
        self.assertIn("b", selections)

    def test_exploration_picks_from_top_half(self):
        # Exploration should only pick from top 50% of candidates
        mem_a = _build_memory("a", 10, 10)
        mem_b = _build_memory("b", 5, 10)
        mem_c = _build_memory("c", 0, 10)
        mem = PerformanceMemory()
        for obs in mem_a.observations() + mem_b.observations() + mem_c.observations():
            mem.record(obs)
        reg = ModelRegistry([
            ModelProvider(ModelMetadata("a", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
            ModelProvider(ModelMetadata("b", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
            ModelProvider(ModelMetadata("c", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
        ])
        policy = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=1.0, exploration_rate=1.0)
        request = RoutingRequest("t", frozenset({"coding"}), task_type="coding")
        selections = []
        for _ in range(100):
            decision = policy.route(request, reg)
            selections.append(decision.selected_model_ids[0])
        # Should only pick from top 2 (a and b), never c (bottom 50%)
        self.assertNotIn("c", selections)
        self.assertIn("a", selections)

    def test_exploration_does_not_affect_fallback(self):
        # Fallback models should still be ordered by score
        mem_a = _build_memory("a", 10, 10)
        mem_b = _build_memory("b", 5, 10)
        mem = PerformanceMemory()
        for obs in mem_a.observations() + mem_b.observations():
            mem.record(obs)
        reg = ModelRegistry([
            ModelProvider(ModelMetadata("a", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
            ModelProvider(ModelMetadata("b", "p", frozenset({"coding"}), 4096, 0.01, 100, 0.9), ConstAdapter("x")),
        ])
        policy = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=1.0, exploration_rate=1.0)
        request = RoutingRequest("t", frozenset({"coding"}), task_type="coding", max_models=1)
        # When a is selected, b should be fallback
        decision = policy.route(request, reg)
        if decision.selected_model_ids == ["a"]:
            self.assertEqual(decision.fallback_model_ids, ["b"])

    def test_recency_awareness_when_enabled(self):
        # When recency is enabled, recent observations should weigh more
        mem = PerformanceMemory()
        # Old failure
        for _ in range(10):
            mem.record(PerformanceObservation(model_id="model-x", task_type="coding",
                provider_success=False, contributed_to_final=False, task_success=False,
                verification_passed=None, latency_ms=100.0, cost=0.01, token_count=10,
                tool_calls=1, recovery_attempts=0, capabilities=("coding",),
                timestamp="2024-01-01T00:00:00+00:00"))
        # Recent success
        for _ in range(10):
            mem.record(PerformanceObservation(model_id="model-x", task_type="coding",
                provider_success=True, contributed_to_final=True, task_success=True,
                verification_passed=True, latency_ms=100.0, cost=0.01, token_count=10,
                tool_calls=1, recovery_attempts=0, capabilities=("coding",),
                timestamp="2026-01-01T00:00:00+00:00"))
        reg = _registry("model-x")
        # Without recency: 50% success rate
        policy_no_recency = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=1.0, confidence_samples=10.0)
        provider = reg.get("model-x")
        score_no_recency = policy_no_recency._score(provider, RoutingRequest("t", frozenset({"coding"}), task_type="coding"))
        # With recency (half_life=1 day): recent successes dominate (reference time defaults to most recent observation)
        policy_recency = DeterministicRoutingPolicy(performance_memory=mem, empirical_weight=1.0, confidence_samples=10.0,
                                                     recency_half_life_days=1.0, recency_weight=1.0)
        score_recency = policy_recency._score(provider, RoutingRequest("t", frozenset({"coding"}), task_type="coding"))
        # Recency-weighted score should be higher (recent successes matter more)
        self.assertGreater(score_recency, score_no_recency)


if __name__ == "__main__":
    unittest.main()
