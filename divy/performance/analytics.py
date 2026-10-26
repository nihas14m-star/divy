"""Deterministic performance analytics built on top of a performance store.

All calculations are pure functions of the stored observations. Missing data
is represented explicitly as ``None`` rather than invented.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

from .memory import PerformanceMemory
from .schemas import PerformanceObservation


def _filter(
    observations: Iterable[PerformanceObservation],
    model_id: str | None = None,
    task_id: str | None = None,
    task_type: str | None = None,
    provider: str | None = None,
    capability: str | None = None,
) -> list[PerformanceObservation]:
    result = []
    for o in observations:
        if model_id is not None and o.model_id != model_id:
            continue
        if task_id is not None and o.task_id != task_id:
            continue
        if task_type is not None and o.task_type != task_type:
            continue
        if provider is not None and o.provider != provider:
            continue
        if capability is not None and capability not in o.capabilities:
            continue
        result.append(o)
    return result


class PerformanceAnalytics:
    """Aggregates and labels performance evidence with sample-size confidence."""

    def __init__(self, memory: PerformanceMemory) -> None:
        self.memory = memory

    def observations(self) -> list[PerformanceObservation]:
        return self.memory.observations()

    def _stats(self, observations: list[PerformanceObservation]) -> dict[str, Any]:
        return aggregate(observations)

    def model_stats(self, model_id: str) -> dict[str, Any]:
        return self._stats(_filter(self.observations(), model_id=model_id))

    def task_stats(self, task_id: str) -> dict[str, Any]:
        return self._stats(_filter(self.observations(), task_id=task_id))

    def task_type_stats(self, task_type: str) -> dict[str, Any]:
        return self._stats(_filter(self.observations(), task_type=task_type))

    def capability_stats(self, capability: str) -> dict[str, Any]:
        return self._stats(_filter(self.observations(), capability=capability))

    def compare_models(
        self,
        models: list[str],
        task_type: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        return {
            model_id: self._stats(_filter(self.observations(), model_id=model_id, task_type=task_type))
            for model_id in models
        }

    def model_task_stats(self, model_id: str, task_type: str) -> dict[str, Any]:
        """Stats for a specific model on a specific task type."""
        return self._stats(_filter(self.observations(), model_id=model_id, task_type=task_type))

    def recent(self, model_id: str | None = None, limit: int = 50):
        observations = self.observations()
        if model_id is not None:
            observations = [o for o in observations if o.model_id == model_id]
        return observations[-limit:]

    def recency_model_stats(self, model_id: str, half_life_days: float = 30.0, reference_time: str | None = None, task_type: str | None = None) -> dict[str, Any]:
        """Recency-weighted statistics for a single model.

        Older observations contribute less via exponential decay
        ``weight = exp(-age_days / half_life_days)``. Raw observations are
        never modified or deleted; this is a derived view only.

        Args:
            model_id: The model to analyze
            half_life_days: Decay half-life in days
            reference_time: Reference timestamp (defaults to most recent observation)
            task_type: Optional task type to filter observations
        """
        observations = [o for o in self.observations() if o.model_id == model_id]
        if task_type is not None:
            observations = [o for o in observations if o.task_type == task_type]
        return recency_weighted_aggregate(observations, half_life_days, reference_time)

    def recency_success_rate(self, model_id: str, half_life_days: float = 30.0, reference_time: str | None = None) -> float | None:
        """Recency-weighted success rate for a model (None when no data)."""
        stats = self.recency_model_stats(model_id, half_life_days, reference_time)
        return stats.get("weighted_success_rate")


    @staticmethod
    def label_confidence(sample_size: int) -> str:
        """Human-readable evidence strength for a sample size.

        Tiny samples are labeled weak so DIVY does not mistake them for
        strong evidence.
        """
        if sample_size == 0:
            return "no evidence"
        if sample_size < 5:
            return "very weak (tiny sample)"
        if sample_size < 30:
            return "weak (small sample)"
        if sample_size < 100:
            return "moderate"
        return "strong"

    @staticmethod
    def success_rate_confidence_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
        """Wilson score interval for a binomial proportion."""
        if total == 0:
            return None
        proportion = successes / total
        denominator = 1 + z * z / total
        centre = (proportion + z * z / (2 * total)) / denominator
        radius = (
            z
            * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
            / denominator
        )
        return round(max(0.0, centre - radius), 4), round(min(1.0, centre + radius), 4)

    def model_summary_with_confidence(self, model_id: str) -> dict[str, Any]:
        stats = self.model_stats(model_id)
        total = stats.get("sample_size") or 0
        successes = stats.get("success_count") or 0
        ci = self.success_rate_confidence_interval(successes, total)
        return {
            "model_id": model_id,
            "sample_size": total,
            "confidence": self.label_confidence(total),
            "success_rate": stats.get("success_rate"),
            "success_rate_confidence_interval": ci,
            "failure_rate": stats.get("failure_rate"),
            "average_latency_ms": stats.get("average_latency_ms"),
            "median_latency_ms": stats.get("median_latency_ms"),
            "average_cost": stats.get("average_cost"),
            "average_model_calls": stats.get("average_model_calls"),
            "average_tool_calls": stats.get("average_tool_calls"),
            "cost_per_successful_task": stats.get("cost_per_successful_task"),
        }


def aggregate(observations: list[PerformanceObservation]) -> dict[str, Any]:
    """Compute deterministic statistics for a list of observations."""
    total = len(observations)
    stats: dict[str, Any] = {"sample_size": total}
    if total == 0:
        stats.update(
            {
                "success_rate": None,
                "failure_rate": None,
                "task_success_rate": None,
                "verified_success_rate": None,
                "execution_sample_size": 0,
                "verification_success_rate": None,
                "recovery_success_rate": None,
                "average_latency_ms": None,
                "median_latency_ms": None,
                "average_model_calls": None,
                "average_tool_calls": None,
                "average_recovery_attempts": None,
                "average_token_count": None,
                "average_cost": None,
                "cost_per_successful_task": None,
            }
        )
        return stats

    successes = [o for o in observations if o.provider_success]
    execution_groups = _group_by_execution(observations)
    failures = [o for o in observations if not o.provider_success]
    verified = [o for o in observations if o.verification_passed]
    verification_attempts = [o for o in observations if o.verification_passed is not None]

    recovered = [o for o in observations if o.recovery_attempts > 0]
    recovered_successes = [o for o in recovered if o.provider_success]

    latencies = [o.latency_ms for o in observations if o.latency_ms is not None]
    model_calls = [1 for _ in observations]  # each observation is one model call
    tool_calls = [o.tool_calls for o in observations]
    recovery_attempts = [o.recovery_attempts for o in observations]
    token_counts = [o.token_count for o in observations if o.token_count is not None]
    costs = [o.cost for o in observations if o.cost is not None]

    stats["success_count"] = len(successes)
    stats["failure_count"] = len(failures)
    stats["success_rate"] = round(len(successes) / total, 6)
    stats["failure_rate"] = round(len(failures) / total, 6)
    stats["task_success_count"] = sum(1 for group in execution_groups if any(o.task_success for o in group))
    stats["task_success_rate"] = round(stats["task_success_count"] / len(execution_groups), 6)
    stats["execution_sample_size"] = len(execution_groups)

    stats["verification_attempted_count"] = len(verification_attempts)
    stats["verification_success_rate"] = (
        round(len(verified) / stats["verification_attempted_count"], 6)
        if stats["verification_attempted_count"]
        else None
    )
    verified_groups = [group for group in execution_groups if any(o.verification_passed is not None for o in group)]
    stats["verified_success_rate"] = (
        round(sum(1 for group in verified_groups if any(o.verification_passed is True for o in group) and any(o.task_success for o in group)) / len(verified_groups), 6)
        if verified_groups else None
    )

    stats["recovery_attempt_count"] = len(recovered)
    stats["recovery_success_rate"] = (
        round(len(recovered_successes) / len(recovered), 6) if recovered else None
    )

    stats["average_latency_ms"] = round(sum(latencies) / len(latencies), 4) if latencies else None
    stats["median_latency_ms"] = _median(latencies) if latencies else None
    stats["average_model_calls"] = round(sum(model_calls) / total, 4)
    stats["average_tool_calls"] = round(sum(tool_calls) / total, 4)
    stats["average_recovery_attempts"] = round(sum(recovery_attempts) / total, 4)
    stats["average_token_count"] = (
        round(sum(token_counts) / len(token_counts), 2) if token_counts else None
    )
    stats["average_cost"] = round(sum(costs) / len(costs), 6) if costs else None
    stats["cost_per_successful_task"] = (
        round(sum(costs) / len(successes), 6) if successes and costs else None
    )

    failure_breakdown: dict[str, int] = {}
    for o in failures:
        ft = o.failure_type or "unknown"
        failure_breakdown[ft] = failure_breakdown.get(ft, 0) + 1
    stats["failure_breakdown"] = failure_breakdown

    return stats


def _group_by_execution(observations: list[PerformanceObservation]) -> list[list[PerformanceObservation]]:
    groups: dict[str, list[PerformanceObservation]] = {}
    for index, observation in enumerate(observations):
        key = observation.execution_id or f"record:{index}"
        groups.setdefault(key, []).append(observation)
    return list(groups.values())


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        # Treat legacy naive timestamps as UTC so mixed historical records do
        # not raise when recency analytics compares aware and naive datetimes.
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _decay_weight(age_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return math.exp(-math.log(2) * max(0.0, age_days) / half_life_days)


def recency_weighted_aggregate(
    observations: list[PerformanceObservation],
    half_life_days: float = 30.0,
    reference_time: str | None = None,
) -> dict[str, Any]:
    """Compute recency-weighted statistics for a list of observations.

    Each observation's weight decays exponentially with its age relative to
    ``reference_time`` (defaults to the most recent observation timestamp, so
    results are deterministic and independent of wall clock). Raw observations
    are not modified.

    Returns a dict mirroring :func:`aggregate` plus:
      * ``weighted_sample_size`` — sum of decay weights
      * ``effective_sample_size`` — Kish effective sample size
      * ``weighted_success_rate`` — decay-weighted success rate
      * ``half_life_days`` — the decay half-life used
    """
    total = len(observations)
    stats: dict[str, Any] = {
        "sample_size": total,
        "half_life_days": half_life_days,
    }
    if total == 0:
        stats.update(
            {
                "weighted_sample_size": 0.0,
                "weighted_success_rate": None,
                "weighted_failure_rate": None,
                "weighted_average_latency_ms": None,
                "weighted_average_cost": None,
            }
        )
        return stats

    # Anchor to the most recent observation by default so results are
    # deterministic regardless of when the query runs.
    if reference_time is None:
        parsed = [
            parsed
            for o in observations
            if o.timestamp
            for parsed in [_parse_timestamp(o.timestamp)]
            if parsed is not None
        ]
        reference_dt = max(parsed) if parsed else None
    else:
        reference_dt = _parse_timestamp(reference_time)

    weights: list[float] = []
    weighted_success = 0.0
    weighted_failure = 0.0
    weighted_contribution = 0.0
    weighted_latency_sum = 0.0
    weighted_latency_weight = 0.0
    weighted_cost_sum = 0.0
    weighted_cost_weight = 0.0

    for o in observations:
        age_days = 0.0
        if reference_dt is not None and o.timestamp:
            observed_dt = _parse_timestamp(o.timestamp)
            if observed_dt is not None:
                age_days = (reference_dt - observed_dt).total_seconds() / 86400.0
        weight = _decay_weight(age_days, half_life_days)
        weights.append(weight)
        if o.provider_success:
            weighted_success += weight
        else:
            weighted_failure += weight
        if o.contributed_to_final:
            weighted_contribution += weight
        if o.latency_ms is not None:
            weighted_latency_sum += weight * o.latency_ms
            weighted_latency_weight += weight
        if o.cost is not None:
            weighted_cost_sum += weight * o.cost
            weighted_cost_weight += weight

    weight_sum = sum(weights)
    stats["weighted_sample_size"] = round(weight_sum, 6)
    stats["effective_sample_size"] = round((weight_sum * weight_sum) / sum(weight * weight for weight in weights), 6) if weight_sum else 0.0
    stats["weighted_success_rate"] = (
        round(weighted_success / weight_sum, 6) if weight_sum else None
    )
    stats["weighted_failure_rate"] = (
        round(weighted_failure / weight_sum, 6) if weight_sum else None
    )
    stats["weighted_contribution_rate"] = (
        round(weighted_contribution / weight_sum, 6) if weight_sum else None
    )
    stats["weighted_average_latency_ms"] = (
        round(weighted_latency_sum / weighted_latency_weight, 4) if weighted_latency_weight else None
    )
    stats["weighted_average_cost"] = (
        round(weighted_cost_sum / weighted_cost_weight, 6) if weighted_cost_weight else None
    )
    return stats
