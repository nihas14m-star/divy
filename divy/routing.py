"""Deterministic, replaceable routing policy for model selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from .performance import PerformanceMemory
from .providers import ModelProvider, ModelRegistry


@dataclass(frozen=True)
class RoutingRequest:
    task_id: str
    required_capabilities: frozenset[str]
    complexity: str = "normal"
    critical: bool = False
    max_models: int = 1
    allow_parallel: bool = False
    budget: float | None = None
    task_type: str | None = None


@dataclass
class RoutingDecision:
    task_id: str
    selected_model_ids: list[str]
    fallback_model_ids: list[str] = field(default_factory=list)
    verifier_model_id: str | None = None
    synthesis_model_id: str | None = None
    candidate_model_ids: list[str] = field(default_factory=list)
    rationale: str = ""


class RoutingPolicy:
    def route(self, request: RoutingRequest, registry: ModelRegistry) -> RoutingDecision:
        raise NotImplementedError


class DeterministicRoutingPolicy(RoutingPolicy):
    """Choose capable models using transparent quality/cost/latency scoring."""

    _complexity_weight = {"simple": 0.7, "normal": 1.0, "complex": 1.35}

    def __init__(
        self,
        performance_memory: PerformanceMemory | None = None,
        empirical_weight: float = 0.3,
        confidence_samples: float = 10.0,
        recency_half_life_days: float = 0.0,
        recency_weight: float = 0.0,
        exploration_rate: float = 0.0,
    ) -> None:
        self.performance_memory = performance_memory
        self.empirical_weight = empirical_weight
        self.confidence_samples = confidence_samples
        self.recency_half_life_days = recency_half_life_days
        self.recency_weight = recency_weight
        self.exploration_rate = exploration_rate

    def route(self, request: RoutingRequest, registry: ModelRegistry) -> RoutingDecision:
        candidates = registry.candidates(set(request.required_capabilities))
        if not candidates:
            return RoutingDecision(request.task_id, [], rationale="no available model satisfies required capabilities")

        # Score all candidates
        scored = [(provider, self._score(provider, request)) for provider in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Exploration: with probability exploration_rate, select randomly from top candidates
        # to gather evidence on less-used models
        import random
        if self.exploration_rate > 0 and len(scored) > 1 and random.random() < self.exploration_rate:
            # Pick from top 50% of candidates for exploration
            exploration_pool = scored[:max(2, len(scored) // 2)]
            selected_provider = random.choice(exploration_pool)[0]
            selected = [selected_provider]
            fallback = [p for p, _ in scored if p != selected_provider]
            rationale = (
                f"required={sorted(request.required_capabilities)}; complexity={request.complexity}; "
                f"exploration=True; selected={selected_provider.model_id}"
            )
        else:
            # Exploitation: use highest-scored candidates
            selected_count = min(max(1, request.max_models), len(scored))
            selected = [p for p, _ in scored[:selected_count]]
            fallback = [p for p, _ in scored[selected_count:]]
            rationale = (
                f"required={sorted(request.required_capabilities)}; complexity={request.complexity}; "
                f"selected={','.join(provider.model_id for provider in selected)}"
            )

        verifier = None
        if request.critical:
            verifier = next((provider.model_id for provider, _ in scored if provider.model_id not in {p.model_id for p in selected}), None)
            if verifier is None and len(selected) > 1:
                verifier = selected[-1].model_id

        return RoutingDecision(
            task_id=request.task_id,
            selected_model_ids=[provider.model_id for provider in selected],
            fallback_model_ids=[provider.model_id for provider in fallback],
            verifier_model_id=verifier,
            candidate_model_ids=[provider.model_id for provider, _ in scored],
            rationale=rationale,
        )

    def _score(self, provider: ModelProvider, request: RoutingRequest) -> float:
        complexity_weight = self._complexity_weight.get(request.complexity, 1.0)
        quality = provider.metadata.reliability * complexity_weight
        cost_factor = {"simple": 0.8, "normal": 0.25, "complex": 0.05}.get(request.complexity, 0.25)
        estimated_cost = provider.cost()
        estimated_latency = provider.latency()
        cost_penalty = estimated_cost * cost_factor if estimated_cost is not None else 0.0
        latency_penalty = estimated_latency / 10_000 if estimated_latency is not None else 0.0
        critical_bonus = provider.metadata.reliability * 0.5 if request.critical else 0.0
        budget_penalty = max(0.0, estimated_cost - request.budget) if request.budget is not None and estimated_cost is not None else 0.0
        static_score = quality + critical_bonus - cost_penalty - latency_penalty - budget_penalty
        if self.performance_memory is None:
            return static_score
        empirical = self.performance_memory.summarize(model_id=provider.model_id, task_type=request.task_type or request.task_id)
        if empirical.get("observation_count", 0) == 0:
            return static_score
        observed_reliability = empirical["provider_success_rate"]
        observed_contribution = empirical["contribution_rate"]
        observation_count = empirical.get("observation_count", 0)
        # Shrink the empirical signal toward zero when the sample is small, so a
        # single observation is not treated as equally strong as hundreds.
        confidence_factor = observation_count / (observation_count + self.confidence_samples)
        shrunk_reliability = 0.5 + (observed_reliability - 0.5) * confidence_factor
        shrunk_contribution = 0.5 + (observed_contribution - 0.5) * confidence_factor
        empirical_signal = ((shrunk_reliability + shrunk_contribution) / 2) - 0.5

        # Recency-weighted signal
        if self.recency_half_life_days > 0 and self.recency_weight > 0:
            from .performance import PerformanceAnalytics
            analytics = PerformanceAnalytics(self.performance_memory)
            recency_stats = analytics.recency_model_stats(
                provider.model_id,
                self.recency_half_life_days,
                task_type=request.task_type or request.task_id,
            )
            recency_reliability = recency_stats.get("weighted_success_rate")
            recency_contribution = recency_stats.get("weighted_contribution_rate")
            if recency_reliability is not None:
                # Blend recency with historical
                blended_reliability = (1 - self.recency_weight) * observed_reliability + self.recency_weight * recency_reliability
                blended_contribution = (1 - self.recency_weight) * observed_contribution + self.recency_weight * (recency_contribution if recency_contribution is not None else observed_contribution)
                empirical_signal = (((blended_reliability + blended_contribution) / 2) - 0.5) * confidence_factor

        return static_score + self.empirical_weight * empirical_signal
