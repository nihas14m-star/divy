"""Deterministic, replaceable routing policy for model selection."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from .performance import PerformanceMemory
from .providers import ModelProvider, ModelRegistry


@dataclass(frozen=True)
class TaskProfile:
    """Provider-neutral requirements inferred from a task."""

    task_type: str | None = None
    required_capabilities: frozenset[str] = frozenset()
    required_response_modes: frozenset[str] = frozenset()
    minimum_context: int | None = None
    requires_tool_calling: bool = False
    requires_multimodal: bool = False
    requires_reasoning: bool = False
    requires_coding: bool = False
    latency_sensitive: bool = False
    reliability_sensitive: bool = False
    verification_required: bool = False
    risk_level: str = "normal"
    expected_difficulty: str = "normal"


@dataclass(frozen=True)
class RoutingRequest:
    task_id: str
    required_capabilities: frozenset[str] = frozenset()
    complexity: str = "normal"
    critical: bool = False
    max_models: int = 1
    allow_parallel: bool = False
    budget: float | None = None
    task_type: str | None = None
    profile: TaskProfile | None = None
    max_latency_ms: float | None = None

    def task_profile(self) -> TaskProfile:
        profile = self.profile or TaskProfile()
        return TaskProfile(
            task_type=self.task_type or profile.task_type,
            required_capabilities=self.required_capabilities or profile.required_capabilities,
            required_response_modes=profile.required_response_modes,
            minimum_context=profile.minimum_context,
            requires_tool_calling=profile.requires_tool_calling,
            requires_multimodal=profile.requires_multimodal,
            requires_reasoning=profile.requires_reasoning,
            requires_coding=profile.requires_coding,
            latency_sensitive=profile.latency_sensitive,
            reliability_sensitive=profile.reliability_sensitive,
            verification_required=profile.verification_required,
            risk_level=profile.risk_level,
            expected_difficulty=profile.expected_difficulty,
        )


@dataclass
class RoutingDecision:
    task_id: str
    selected_model_ids: list[str]
    fallback_model_ids: list[str] = field(default_factory=list)
    verifier_model_id: str | None = None
    synthesis_model_id: str | None = None
    candidate_model_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    explanation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RoutingPolicy:
    def route(self, request: RoutingRequest, registry: ModelRegistry) -> RoutingDecision:
        raise NotImplementedError


class DeterministicRoutingPolicy(RoutingPolicy):
    """Choose capable models using transparent quality/cost/latency scoring.

    Args:
        performance_memory: Optional PerformanceMemory for empirical evidence.
        empirical_weight: Weight given to empirical evidence vs static scores (default 0.3).
        confidence_samples: Sample count at which empirical signal reaches ~50% strength (default 10.0).
        recency_half_life_days: Half-life for recency-weighted evidence (default 0.0 = disabled).
        recency_weight: Blend factor for recency vs historical evidence (default 0.0 = disabled).
        exploration_rate: Probability of exploring a randomly chosen top-half candidate (default 0.0).
        seed: Optional seed for deterministic exploration. When provided, exploration
            decisions are reproducible across runs. When None, exploration uses the
            process RNG for backward compatibility; reproducible campaigns pass a seed.
    """

    _complexity_weight = {"simple": 0.7, "normal": 1.0, "complex": 1.35}

    def __init__(
        self,
        performance_memory: PerformanceMemory | None = None,
        empirical_weight: float = 0.3,
        confidence_samples: float = 10.0,
        recency_half_life_days: float = 0.0,
        recency_weight: float = 0.0,
        exploration_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.performance_memory = performance_memory
        self.empirical_weight = empirical_weight
        self.confidence_samples = confidence_samples
        self.recency_half_life_days = recency_half_life_days
        self.recency_weight = recency_weight
        self.exploration_rate = exploration_rate
        self._rng = random.Random(seed) if seed is not None else random
        if not 0.0 <= empirical_weight <= 1.0:
            raise ValueError("empirical_weight must be between 0 and 1")
        if confidence_samples <= 0:
            raise ValueError("confidence_samples must be positive")
        if recency_half_life_days < 0:
            raise ValueError("recency_half_life_days must be non-negative")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError("recency_weight must be between 0 and 1")
        if not 0.0 <= exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be between 0 and 1")

    def route(self, request: RoutingRequest, registry: ModelRegistry) -> RoutingDecision:
        profile = request.task_profile()
        candidates = []
        eligibility: dict[str, dict[str, Any]] = {}
        for provider in registry.list():
            eligible, reasons = registry.eligibility(
                provider,
                required_capabilities=set(profile.required_capabilities),
                task_type=profile.task_type,
                minimum_context=profile.minimum_context,
                required_response_modes=set(profile.required_response_modes),
                requires_tool_calling=profile.requires_tool_calling,
                requires_multimodal=profile.requires_multimodal,
                requires_reasoning=profile.requires_reasoning,
                requires_coding=profile.requires_coding,
                max_latency_ms=request.max_latency_ms,
                budget=request.budget,
            )
            eligibility[provider.model_id] = {"eligible": eligible, "reasons": reasons}
            if eligible:
                candidates.append(provider)
        if not candidates:
            return RoutingDecision(
                request.task_id,
                [],
                candidate_model_ids=[provider.model_id for provider in registry.list()],
                rationale="no available model satisfies the task profile",
                explanation={"task_profile": self._profile_dict(profile), "candidates": eligibility},
            )

        # Score all candidates
        scored = [(provider, self._score(provider, request)) for provider in candidates]
        scored.sort(key=lambda x: (-x[1], x[0].model_id))

        # Exploration: with probability exploration_rate, select randomly from top candidates
        # to gather evidence on less-used models
        if self.exploration_rate > 0 and len(scored) > 1 and self._rng.random() < self.exploration_rate:
            # Pick from top 50% of candidates for exploration
            exploration_pool = scored[:max(2, len(scored) // 2)]
            selected_provider = self._rng.choice(exploration_pool)[0]
            selected = [selected_provider]
            fallback = [p for p, _ in scored if p != selected_provider]
            rationale = (
                f"required={sorted(profile.required_capabilities)}; complexity={request.complexity}; "
                f"exploration=True; selected={selected_provider.model_id}"
            )
        else:
            # Exploitation: use highest-scored candidates
            selected_count = min(max(1, request.max_models), len(scored))
            selected = [p for p, _ in scored[:selected_count]]
            fallback = [p for p, _ in scored[selected_count:]]
            rationale = (
                f"required={sorted(profile.required_capabilities)}; complexity={request.complexity}; "
                f"selected={','.join(provider.model_id for provider in selected)}"
            )

        verifier = None
        if request.critical or profile.verification_required:
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
            explanation=self._build_explanation(request, profile, scored, eligibility, selected, fallback, verifier),
        )

    def _build_explanation(self, request: RoutingRequest, profile: TaskProfile, scored, eligibility, selected, fallback, verifier) -> dict[str, Any]:
        candidates = []
        for provider, score in scored:
            evidence = self.performance_memory.summarize(model_id=provider.model_id, task_type=profile.task_type or request.task_id) if self.performance_memory else {"observation_count": 0}
            candidates.append({
                "model_id": provider.model_id,
                "eligible": True,
                "score": round(score, 6),
                "sample_size": evidence.get("execution_sample_size", evidence.get("observation_count", 0)),
                "observation_count": evidence.get("observation_count", 0),
                "task_success_rate": evidence.get("task_success_rate"),
                "provider_success_rate": evidence.get("provider_success_rate"),
                "verification_success_rate": evidence.get("verification_success_rate"),
                "verified_success_rate": evidence.get("verified_success_rate"),
                "average_latency_ms": evidence.get("average_latency_ms"),
                "average_cost": evidence.get("average_cost"),
                "failure_breakdown": evidence.get("failure_breakdown", {}),
            })
        rejected = {model_id: item for model_id, item in eligibility.items() if not item["eligible"]}
        return {
            "policy": "deterministic-evidence-v2",
            "task_profile": self._profile_dict(profile),
            "budget": request.budget,
            "max_latency_ms": request.max_latency_ms,
            "selected": [provider.model_id for provider in selected],
            "fallback": [provider.model_id for provider in fallback],
            "verifier": verifier,
            "candidates": candidates,
            "rejected": rejected,
            "tie_break": "score descending, model_id ascending",
        }

    @staticmethod
    def _profile_dict(profile: TaskProfile) -> dict[str, Any]:
        data = asdict(profile)
        data["required_capabilities"] = sorted(profile.required_capabilities)
        data["required_response_modes"] = sorted(profile.required_response_modes)
        return data

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
        profile = request.task_profile()
        empirical = self.performance_memory.summarize(model_id=provider.model_id, task_type=profile.task_type or request.task_id)
        if empirical.get("observation_count", 0) == 0:
            return static_score
        observed_reliability = empirical.get("task_success_rate")
        if observed_reliability is None:
            observed_reliability = empirical["provider_success_rate"]
        observed_contribution = empirical["contribution_rate"]
        observation_count = empirical.get("observation_count", 0)
        evidence_count = empirical.get("execution_sample_size") or observation_count
        # Shrink the empirical signal toward zero when the sample is small, so a
        # single observation is not treated as equally strong as hundreds.
        confidence_factor = evidence_count / (evidence_count + self.confidence_samples)
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
                task_type=profile.task_type or request.task_id,
            )
            recency_reliability = recency_stats.get("weighted_success_rate")
            recency_contribution = recency_stats.get("weighted_contribution_rate")
            if recency_reliability is not None:
                # Blend recency with historical
                blended_reliability = (1 - self.recency_weight) * observed_reliability + self.recency_weight * recency_reliability
                blended_contribution = (1 - self.recency_weight) * observed_contribution + self.recency_weight * (recency_contribution if recency_contribution is not None else observed_contribution)
                empirical_signal = (((blended_reliability + blended_contribution) / 2) - 0.5) * confidence_factor

        return static_score + self.empirical_weight * empirical_signal
