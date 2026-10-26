"""Deterministic model comparison interface.

Compares models on measurable evidence. Raw measurements are exposed
separately from any derived ranking. A configurable score is optional and
clearly labeled as derived, never as fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analytics import PerformanceAnalytics


@dataclass(frozen=True)
class ModelScoreConfig:
    """Weights for the optional derived ranking score.

    All weights default to zero so that no model is declared a winner by
    arbitrary assumptions unless a caller explicitly enables scoring.
    """
    success_weight: float = 0.0
    latency_weight: float = 0.0
    cost_weight: float = 0.0
    reliability_weight: float = 0.0
    lower_latency_better: bool = True
    lower_cost_better: bool = True


@dataclass(frozen=True)
class ModelComparisonResult:
    model_id: str
    sample_size: int
    confidence: str
    raw_measurements: dict[str, Any]
    derived_score: float | None = None
    derived_rank: int | None = None


class ModelComparison:
    """Compares models on task-type-filtered historical performance."""

    def __init__(self, analytics: PerformanceAnalytics) -> None:
        self.analytics = analytics

    def compare(
        self,
        models: list[str],
        task_type: str | None = None,
        score_config: ModelScoreConfig | None = None,
    ) -> dict[str, ModelComparisonResult]:
        """Return evidence for each model, optionally with a derived ranking.

        ``score_config`` is optional. When omitted, no derived score or rank
        is produced; only raw measurements are exposed.
        """
        stats = self.analytics.compare_models(models, task_type)
        scored: list[tuple[str, float]] = []
        if score_config is not None:
            for model_id in models:
                score = self._compute_score(stats.get(model_id, {}), score_config)
                scored.append((model_id, score))
        ranks = self._ranks(scored) if score_config is not None else {}

        results: dict[str, ModelComparisonResult] = {}
        for model_id in models:
            measurement = stats.get(model_id, {})
            sample_size = measurement.get("sample_size") or 0
            successes = measurement.get("success_count") or 0
            total = sample_size
            raw = {
                "sample_size": sample_size,
                "confidence": self.analytics.label_confidence(sample_size),
                "success_rate": measurement.get("success_rate"),
                "failure_rate": measurement.get("failure_rate"),
                "verification_success_rate": measurement.get("verification_success_rate"),
                "recovery_success_rate": measurement.get("recovery_success_rate"),
                "average_latency_ms": measurement.get("average_latency_ms"),
                "median_latency_ms": measurement.get("median_latency_ms"),
                "average_model_calls": measurement.get("average_model_calls"),
                "average_tool_calls": measurement.get("average_tool_calls"),
                "average_recovery_attempts": measurement.get("average_recovery_attempts"),
                "average_token_count": measurement.get("average_token_count"),
                "average_cost": measurement.get("average_cost"),
                "cost_per_successful_task": measurement.get("cost_per_successful_task"),
                "failure_breakdown": measurement.get("failure_breakdown", {}),
            }
            raw["success_rate_confidence_interval"] = self.analytics.success_rate_confidence_interval(
                successes, total
            )
            score = None
            rank = None
            if score_config is not None:
                score = self._compute_score(measurement, score_config)
                rank = ranks.get(model_id)
            results[model_id] = ModelComparisonResult(
                model_id=model_id,
                sample_size=sample_size,
                confidence=self.analytics.label_confidence(sample_size),
                raw_measurements=raw,
                derived_score=score,
                derived_rank=rank,
            )
        return results

    @staticmethod
    def _compute_score(measurement: dict[str, Any], config: ModelScoreConfig) -> float:
        success = measurement.get("success_rate")
        latency = measurement.get("average_latency_ms")
        cost = measurement.get("average_cost")
        reliability = measurement.get("verification_success_rate")

        score = 0.0
        if config.success_weight and success is not None:
            score += config.success_weight * success
        if config.latency_weight and latency is not None:
            latency_term = -latency if config.lower_latency_better else latency
            score += config.latency_weight * latency_term
        if config.cost_weight and cost is not None:
            cost_term = -cost if config.lower_cost_better else cost
            score += config.cost_weight * cost_term
        if config.reliability_weight and reliability is not None:
            score += config.reliability_weight * reliability
        return round(score, 6)

    @staticmethod
    def _ranks(scored: list[tuple[str, float]]) -> dict[str, int]:
        """Higher score = better rank (1 is best). Ties share first-seen rank."""
        ordered = sorted(scored, key=lambda item: item[1], reverse=True)
        ranks: dict[str, int] = {}
        for position, (model_id, _) in enumerate(ordered, start=1):
            ranks.setdefault(model_id, position)
        return ranks
