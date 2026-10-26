"""Clean integration interface for the core DIVY orchestrator.

The performance subsystem answers questions; it does not independently
control execution. Everything here returns evidence that the router may
consume, never a routing decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analytics import PerformanceAnalytics, aggregate
from .comparison import ModelComparison, ModelScoreConfig
from .memory import PerformanceMemory


@dataclass(frozen=True)
class HistoricalEvidence:
    model_id: str
    sample_size: int
    confidence: str
    success_rate: float | None
    average_latency_ms: float | None
    average_cost: float | None
    cost_per_successful_task: float | None
    verification_success_rate: float | None
    recovery_success_rate: float | None
    task_success_rate: float | None = None
    verified_success_rate: float | None = None
    execution_sample_size: int = 0
    failure_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvidence:
    task_type: str | None
    candidates: list[HistoricalEvidence] = field(default_factory=list)
    note: str = ""


class PerformanceInterface:
    """Read-only evidence interface consumed by the orchestrator.

    Methods never route, select, or execute; they return historical
    evidence for the router to interpret.
    """

    def __init__(self, memory: PerformanceMemory) -> None:
        self.memory = memory
        self.analytics = PerformanceAnalytics(memory)
        self.comparison = ModelComparison(self.analytics)

    # --- single-model evidence ------------------------------------------

    def model_evidence(self, model_id: str) -> HistoricalEvidence:
        stats = self.analytics.model_stats(model_id)
        return HistoricalEvidence(
            model_id=model_id,
            sample_size=stats.get("sample_size") or 0,
            confidence=self.analytics.label_confidence(stats.get("sample_size") or 0),
            success_rate=stats.get("success_rate"),
            average_latency_ms=stats.get("average_latency_ms"),
            average_cost=stats.get("average_cost"),
            cost_per_successful_task=stats.get("cost_per_successful_task"),
            verification_success_rate=stats.get("verification_success_rate"),
            recovery_success_rate=stats.get("recovery_success_rate"),
            task_success_rate=stats.get("task_success_rate"),
            verified_success_rate=stats.get("verified_success_rate"),
            execution_sample_size=stats.get("execution_sample_size", 0),
            failure_breakdown=stats.get("failure_breakdown", {}),
        )

    # --- task / capability evidence -------------------------------------

    def task_type_evidence(self, task_type: str) -> dict[str, Any]:
        return self.analytics.task_type_stats(task_type)

    def capability_evidence(self, capability: str) -> dict[str, Any]:
        return self.analytics.capability_stats(capability)

    # --- model comparison (the "which model is best" evidence) ----------

    def compare(
        self,
        models: list[str],
        task_type: str | None = None,
        score_config: ModelScoreConfig | None = None,
    ) -> dict[str, Any]:
        """Return raw measurements plus optional derived ranking.

        The derived score/rank is only present when ``score_config`` is
        explicitly supplied; otherwise only raw evidence is exposed.
        """
        results = self.comparison.compare(models, task_type, score_config)
        return {
            model_id: {
                "sample_size": r.sample_size,
                "confidence": r.confidence,
                "raw_measurements": r.raw_measurements,
                "derived_score": r.derived_score,
                "derived_rank": r.derived_rank,
            }
            for model_id, r in results.items()
        }

    def get_best_historical_candidates(
        self,
        models: list[str],
        task_type: str | None = None,
        min_samples: int = 5,
    ) -> CandidateEvidence:
        """Return evidence for models, optionally filtered by sample size.

        Does not declare a winner. Returns raw evidence so the router can
        apply its own policy. ``min_samples`` excludes models with too few
        observations to be reliable evidence.
        """
        results = self.comparison.compare(models, task_type)
        candidates: list[HistoricalEvidence] = []
        for model_id in models:
            r = results[model_id]
            if r.sample_size < min_samples:
                continue
            m = r.raw_measurements
            candidates.append(
                HistoricalEvidence(
                    model_id=model_id,
                    sample_size=r.sample_size,
                    confidence=r.confidence,
                    success_rate=m["success_rate"],
                    average_latency_ms=m["average_latency_ms"],
                    average_cost=m["average_cost"],
                    cost_per_successful_task=m["cost_per_successful_task"],
                    verification_success_rate=m["verification_success_rate"],
                    recovery_success_rate=m["recovery_success_rate"],
                    task_success_rate=m.get("task_success_rate"),
                    verified_success_rate=m.get("verified_success_rate"),
                    execution_sample_size=m.get("execution_sample_size", r.sample_size),
                    failure_breakdown=m.get("failure_breakdown", {}),
                )
            )
        note = f"{len(candidates)}/{len(models)} models met min_samples={min_samples}"
        return CandidateEvidence(task_type=task_type, candidates=candidates, note=note)

    # --- recent history -------------------------------------------------

    def recent_performance(self, model_id: str | None = None, limit: int = 50):
        if limit <= 0:
            return []
        return [o for o in self.memory.observations() if model_id is None or o.model_id == model_id][-limit:]

    # --- aggregate everything ------------------------------------------

    def aggregate_all(self) -> dict[str, Any]:
        return aggregate(self.memory.observations())

    # --- recency-weighted evidence -------------------------------------

    def model_evidence_recency(self, model_id: str, half_life_days: float = 30.0, reference_time: str | None = None, task_type: str | None = None) -> dict[str, Any]:
        """Recency-weighted evidence for a model (older observations count less)."""
        return self.analytics.recency_model_stats(model_id, half_life_days, reference_time, task_type)

    def model_evidence_combined(
        self,
        model_id: str,
        task_type: str | None = None,
        half_life_days: float = 30.0,
        recency_weight: float = 0.3,
    ) -> dict[str, Any]:
        """Combined historical + recency-weighted evidence for a model.

        Returns a dict with both raw historical stats and recency-weighted stats,
        plus a blended view weighted by ``recency_weight`` (0 = all historical,
        1 = all recency-weighted). This is the recommended evidence source for routing.
        """
        historical = self.analytics.model_stats(model_id)
        if task_type:
            # Filter properly
            filtered = [
                o for o in self.memory.observations()
                if o.model_id == model_id and o.task_type == task_type
            ]
            historical = self.analytics._stats(filtered)

        if half_life_days > 0:
            recency = self.analytics.recency_model_stats(model_id, half_life_days, task_type=task_type)
        else:
            recency = historical

        # Blend the two
        blended = {}
        for key in ["success_rate", "average_latency_ms", "average_cost", "verification_success_rate", "recovery_success_rate", "average_tool_calls", "average_recovery_attempts"]:
            h_val = historical.get(key)
            r_val = recency.get("weighted_" + key) if half_life_days > 0 else recency.get(key)
            if h_val is not None and r_val is not None:
                blended[key] = round((1 - recency_weight) * h_val + recency_weight * r_val, 6)
            elif h_val is not None:
                blended[key] = h_val
            elif r_val is not None:
                blended[key] = r_val
            else:
                blended[key] = None

        return {
            "model_id": model_id,
            "task_type": task_type,
            "historical": historical,
            "recency": recency,
            "blended": blended,
            "recency_weight": recency_weight,
            "half_life_days": half_life_days,
        }

    def capability_evidence(self, capability: str) -> dict[str, Any]:
        """Evidence for a specific capability across all models."""
        return self.analytics.capability_stats(capability)

    def task_type_evidence(self, task_type: str) -> dict[str, Any]:
        """Evidence for a specific task type across all models."""
        return self.analytics.task_type_stats(task_type)

    def compare_models_for_task(
        self,
        models: list[str],
        task_type: str | None = None,
        half_life_days: float = 30.0,
        recency_weight: float = 0.3,
        score_config: ModelScoreConfig | None = None,
    ) -> dict[str, Any]:
        """Compare models for a specific task type with blended evidence.

        Returns per-model blended evidence plus optional derived ranking.
        """
        results = {}
        for model_id in models:
            combined = self.model_evidence_combined(model_id, task_type, half_life_days, recency_weight)
            results[model_id] = {
                "blended": combined["blended"],
                "historical_sample_size": combined["historical"].get("sample_size", 0),
                "recency_weighted_sample_size": combined["recency"].get("weighted_sample_size", 0),
            }

        if score_config is not None:
            # Use comparison layer for ranking
            comparison_results = self.comparison.compare(models, task_type, score_config)
            for model_id, r in comparison_results.items():
                if model_id in results:
                    results[model_id]["derived_score"] = r.derived_score
                    results[model_id]["derived_rank"] = r.derived_rank

        return results
