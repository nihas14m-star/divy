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

    def capability_matches(self, required_capabilities: set[str]) -> dict[str, list[str]]:
        matches: dict[str, list[str]] = {}
        for provider in self._providers.values():
            missing = sorted(required_capabilities - set(provider.capabilities()))
            matches[provider.model_id] = missing
        return matches
