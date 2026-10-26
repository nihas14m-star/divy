"""Provider-agnostic model metadata and registry.

The registry knows only about capabilities and operational estimates. It does
not know whether a provider is local, hosted, or backed by a particular model
vendor. Concrete adapters still implement the existing ``ModelAdapter``
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import ModelAdapter
from .schemas import ModelResponse


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    provider: str
    capabilities: frozenset[str]
    context_limit: int | None
    cost_estimate: float | None
    latency_estimate_ms: float | None
    reliability: float
    available: bool = True
    tags: frozenset[str] = field(default_factory=frozenset)
    supported_response_modes: frozenset[str] = field(default_factory=frozenset)
    tool_calling: bool | None = None
    multimodal: bool | None = None
    reasoning: bool | None = None
    coding: bool | None = None
    agentic: bool | None = None
    recommended_task_types: frozenset[str] = field(default_factory=frozenset)
    documentation_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be between 0 and 1")
        if self.context_limit is not None and self.context_limit <= 0:
            raise ValueError("context_limit must be positive when provided")
        if self.cost_estimate is not None and self.cost_estimate < 0:
            raise ValueError("cost_estimate must be non-negative when provided")
        if self.latency_estimate_ms is not None and self.latency_estimate_ms < 0:
            raise ValueError("latency_estimate_ms must be non-negative when provided")


class ModelProvider:
    """A provider entry combining metadata with a replaceable adapter."""

    def __init__(self, metadata: ModelMetadata, adapter: ModelAdapter) -> None:
        self.metadata = metadata
        self.adapter = adapter

    @property
    def model_id(self) -> str:
        return self.metadata.model_id

    def capabilities(self) -> frozenset[str]:
        return self.metadata.capabilities

    def cost(self) -> float | None:
        return self.metadata.cost_estimate

    def latency(self) -> float | None:
        return self.metadata.latency_estimate_ms

    def availability(self) -> bool:
        return self.metadata.available

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], response_schema: dict[str, Any]) -> ModelResponse:
        return self.adapter.generate(messages, tools, response_schema)


class ModelRegistry:
    def __init__(self, providers: list[ModelProvider] | None = None) -> None:
        self._providers: dict[str, ModelProvider] = {}
        for provider in providers or []:
            self.register(provider)

    def register(self, provider: ModelProvider) -> None:
        if provider.model_id in self._providers:
            raise ValueError(f"model_id already registered: {provider.model_id}")
        self._providers[provider.model_id] = provider

    def get(self, model_id: str) -> ModelProvider:
        try:
            return self._providers[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model_id: {model_id}") from exc

    def list(self) -> list[ModelProvider]:
        return list(self._providers.values())

    def candidates(self, required_capabilities: set[str], available_only: bool = True) -> list[ModelProvider]:
        return [
            provider
            for provider in self._providers.values()
            if (not available_only or provider.availability())
            and required_capabilities <= set(provider.capabilities())
        ]

    def eligibility(
        self,
        provider: ModelProvider,
        *,
        required_capabilities: set[str] | frozenset[str] = frozenset(),
        task_type: str | None = None,
        minimum_context: int | None = None,
        required_response_modes: set[str] | frozenset[str] = frozenset(),
        requires_tool_calling: bool = False,
        requires_multimodal: bool = False,
        requires_reasoning: bool = False,
        requires_coding: bool = False,
        max_latency_ms: float | None = None,
        budget: float | None = None,
    ) -> tuple[bool, list[str]]:
        """Return whether a provider can satisfy a task profile.

        This is a hard feasibility check, deliberately separate from ranking.
        Unknown optional metadata never becomes a fabricated capability; a
        requirement is accepted only when the provider explicitly supports it.
        """
        reasons: list[str] = []
        metadata = provider.metadata
        missing = sorted(set(required_capabilities) - set(metadata.capabilities))
        if missing:
            reasons.append(f"missing capabilities: {', '.join(missing)}")
        if not metadata.available:
            reasons.append("model unavailable")
        if minimum_context is not None and (metadata.context_limit is None or metadata.context_limit < minimum_context):
            reasons.append("insufficient context window")
        if required_response_modes and not set(required_response_modes) <= set(metadata.supported_response_modes):
            reasons.append("unsupported response mode")
        if requires_tool_calling and metadata.tool_calling is not True:
            reasons.append("tool calling unavailable")
        if requires_multimodal and metadata.multimodal is not True:
            reasons.append("multimodal input unavailable")
        if requires_reasoning and metadata.reasoning is not True:
            reasons.append("reasoning capability unavailable")
        if requires_coding and metadata.coding is not True:
            reasons.append("coding capability unavailable")
        if max_latency_ms is not None and (metadata.latency_estimate_ms is None or metadata.latency_estimate_ms > max_latency_ms):
            reasons.append("latency budget exceeded or unknown")
        if budget is not None and (metadata.cost_estimate is None or metadata.cost_estimate > budget):
            reasons.append("cost budget exceeded or unknown")
        return not reasons, reasons

    def capability_matches(self, required_capabilities: set[str]) -> dict[str, list[str]]:
        matches: dict[str, list[str]] = {}
        for provider in self._providers.values():
            missing = sorted(required_capabilities - set(provider.capabilities()))
            matches[provider.model_id] = missing
        return matches
