"""Tests for recency-weighted performance statistics.

Raw observations remain immutable; recency weighting is a derived view in
which older observations contribute less via exponential decay.
"""

from __future__ import annotations

import unittest

from divy.performance import PerformanceMemory, PerformanceObservation
from divy.performance.analytics import recency_weighted_aggregate


def _obs(model_id, task_success, timestamp):
    return PerformanceObservation(
        model_id=model_id,
        task_type="coding",
        capabilities=("coding",),
        provider_success=task_success,
        contributed_to_final=task_success,
        task_success=task_success,
        verification_passed=task_success,
        latency_ms=100.0,
        cost=0.01,
        token_count=10,
        tool_calls=1,
        recovery_attempts=0,
        timestamp=timestamp,
    )


class RecencyTests(unittest.TestCase):
    def test_empty_returns_none(self):
        stats = recency_weighted_aggregate([])
        self.assertEqual(stats["sample_size"], 0)
        self.assertIsNone(stats["weighted_success_rate"])
        self.assertEqual(stats["weighted_sample_size"], 0.0)

    def test_single_observation_matches_its_outcome(self):
        stats = recency_weighted_aggregate([_obs("m", True, "2026-01-01T00:00:00+00:00")])
        self.assertEqual(stats["weighted_success_rate"], 1.0)
        stats2 = recency_weighted_aggregate([_obs("m", False, "2026-01-01T00:00:00+00:00")])
        self.assertEqual(stats2["weighted_success_rate"], 0.0)

    def test_recent_observation_has_more_influence(self):
        # Old success, recent failure. Raw rate = 0.5. Recency-weighted should
        # favor the recent failure, pulling the weighted rate below 0.5.
        obs = [
            _obs("m", True, "2026-01-01T00:00:00+00:00"),   # old success
            _obs("m", False, "2026-01-02T00:00:00+00:00"),  # recent failure
        ]
        raw = 0.5
        weighted = recency_weighted_aggregate(obs, half_life_days=1.0)
        self.assertIsNotNone(weighted["weighted_success_rate"])
        self.assertLess(weighted["weighted_success_rate"], raw)
        # The recent failure should dominate. A true half-life gives an old
        # observation weight 0.5, so the rate is 1 / 3.
        self.assertAlmostEqual(weighted["weighted_success_rate"], 1 / 3, places=3)

    def test_anchored_to_most_recent_is_deterministic(self):
        obs = [
            _obs("m", True, "2026-01-01T00:00:00+00:00"),
            _obs("m", False, "2026-01-05T00:00:00+00:00"),
        ]
        s1 = recency_weighted_aggregate(obs, half_life_days=2.0)
        s2 = recency_weighted_aggregate(obs, half_life_days=2.0)
        self.assertEqual(s1, s2)

    def test_explicit_reference_time_is_honored(self):
        obs = [_obs("m", True, "2026-01-01T00:00:00+00:00")]
        # reference_time equal to the observation timestamp => age 0 => full weight
        stats = recency_weighted_aggregate(obs, half_life_days=1.0, reference_time="2026-01-01T00:00:00+00:00")
        self.assertEqual(stats["weighted_success_rate"], 1.0)

    def test_older_decays_more(self):
        # Same observation, smaller half-life => older data decays faster, but
        # with a single observation anchored to itself the weight is always 1.
        obs = [_obs("m", True, "2026-01-01T00:00:00+00:00")]
        s_short = recency_weighted_aggregate(obs, half_life_days=1.0, reference_time="2026-01-02T00:00:00+00:00")
        s_long = recency_weighted_aggregate(obs, half_life_days=100.0, reference_time="2026-01-02T00:00:00+00:00")
        # Shorter half-life decays the 1-day-old observation more.
        self.assertLess(s_short["weighted_sample_size"], s_long["weighted_sample_size"])

    def test_half_life_means_half_weight(self):
        obs = [_obs("m", True, "2026-01-01T00:00:00+00:00")]
        stats = recency_weighted_aggregate(obs, half_life_days=1.0, reference_time="2026-01-02T00:00:00+00:00")
        self.assertAlmostEqual(stats["weighted_sample_size"], 0.5, places=6)
        self.assertAlmostEqual(stats["effective_sample_size"], 1.0, places=6)

    def test_analytics_recency_method(self):
        mem = PerformanceMemory()
        mem.record(_obs("m", True, "2026-01-01T00:00:00+00:00"))
        mem.record(_obs("m", False, "2026-01-02T00:00:00+00:00"))
        from divy.performance import PerformanceAnalytics
        analytics = PerformanceAnalytics(mem)
        rate = analytics.recency_success_rate("m", half_life_days=1.0)
        self.assertIsNotNone(rate)
        self.assertLess(rate, 0.5)

    def test_interface_recency_method(self):
        mem = PerformanceMemory()
        mem.record(_obs("m", True, "2026-01-01T00:00:00+00:00"))
        mem.record(_obs("m", False, "2026-01-02T00:00:00+00:00"))
        from divy.performance import PerformanceInterface
        interface = PerformanceInterface(mem)
        stats = interface.model_evidence_recency("m", half_life_days=1.0)
        self.assertIsNotNone(stats["weighted_success_rate"])
        self.assertEqual(stats["half_life_days"], 1.0)
