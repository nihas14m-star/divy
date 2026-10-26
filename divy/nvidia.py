"""NVIDIA API Catalog provider configuration.

This module is an adapter/configuration layer only. Core routing, execution,
verification, and the AgentSession do not import NVIDIA names or model IDs.
The metadata below records only properties exposed by NVIDIA's current model
and API documentation; unknown operational estimates remain ``None``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .adapters import HttpModelAdapter, ProviderError
from .providers import ModelMetadata, ModelProvider, ModelRegistry


NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass(frozen=True)
class NvidiaModelSpec:
    model_id: str
    role: str
    capabilities: frozenset[str]
    context_limit: int | None
    tool_calling: bool | None
    structured_output: bool | None
    multimodal: bool | None
    reasoning: bool | None
    coding: bool | None
    agentic: bool | None
    recommended_task_types: frozenset[str]
    documentation_sources: tuple[str, ...]


# NVIDIA model IDs are kept in this provider module, not in the core router.
# Pricing and latency are intentionally unknown because the public model pages
# consulted for this integration do not publish stable per-call estimates.
NVIDIA_MODEL_SPECS: tuple[NvidiaModelSpec, ...] = (
    NvidiaModelSpec(
        model_id="nvidia/nemotron-3.5-bolt-e4b",
        role="fast executor / fallback",
        capabilities=frozenset({"general", "coding", "reasoning", "agentic", "tool_calling", "structured_output"}),
        context_limit=65536,
        tool_calling=True,
        structured_output=True,
        multimodal=False,
        reasoning=True,
        coding=True,
        agentic=True,
        recommended_task_types=frozenset({"software_engineering", "tool_use", "short_reasoning", "fallback"}),
        documentation_sources=(
            "https://build.nvidia.com/nvidia/nemotron-3.5-bolt-e4b",
            "https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-bolt-e4b",
        ),
    ),
    NvidiaModelSpec(
        model_id="nvidia/nemotron-3.5-lightning-30b-a3b",
        role="reasoning planner / coding specialist",
        capabilities=frozenset({"general", "coding", "reasoning", "agentic", "tool_calling", "structured_output"}),
        context_limit=1000000,
        tool_calling=True,
        structured_output=True,
        multimodal=False,
        reasoning=True,
        coding=True,
        agentic=True,
        recommended_task_types=frozenset({"planning", "software_engineering", "complex_reasoning", "synthesis"}),
        documentation_sources=(
            "https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b",
            "https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-lightning-30b-a3b",
        ),
    ),
    NvidiaModelSpec(
        model_id="nvidia/nemotron-3.5-super-vl-120b-a12b",
        role="multimodal planner / verifier",
        capabilities=frozenset({"general", "coding", "reasoning", "agentic", "tool_calling", "structured_output", "vision"}),
        context_limit=1000000,
        tool_calling=True,
        structured_output=True,
        multimodal=True,
        reasoning=True,
        coding=True,
        agentic=True,
        recommended_task_types=frozenset({"multimodal", "planning", "verification", "synthesis"}),
        documentation_sources=(
            "https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b",
            "https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b",
        ),
    ),
)


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def nvidia_model_spec(model_id: str) -> NvidiaModelSpec:
    for spec in NVIDIA_MODEL_SPECS:
        if spec.model_id == model_id:
            return spec
    raise ProviderError("configuration", f"unsupported NVIDIA model id: {model_id}")


def nvidia_provider_from_environment(model_id: str | None = None) -> ModelProvider:
    """Create one NVIDIA provider from environment-only configuration."""

    selected_model = model_id or _env_first("DIVY_NVIDIA_MODEL_ID", "NVIDIA_MODEL_ID", "DIVY_MODEL_ID") or NVIDIA_MODEL_SPECS[0].model_id
    spec = nvidia_model_spec(selected_model)
    base_url = _env_first("DIVY_NVIDIA_BASE_URL", "NVIDIA_API_BASE_URL", "NVIDIA_BASE_URL") or NVIDIA_API_BASE_URL
    api_key = _env_first("DIVY_NVIDIA_API_KEY", "NVIDIA_API_KEY", "DIVY_MODEL_API_KEY")
    if not api_key:
        raise ProviderError("configuration", "NVIDIA_API_KEY or DIVY_NVIDIA_API_KEY is required")
    timeout_text = _env_first("DIVY_NVIDIA_TIMEOUT_SECONDS", "NVIDIA_TIMEOUT_SECONDS", "DIVY_MODEL_TIMEOUT_SECONDS") or "60"
    try:
        timeout_seconds = float(timeout_text)
    except ValueError as exc:
        raise ProviderError("configuration", "NVIDIA timeout must be numeric") from exc
    response_mode = _env_first("DIVY_NVIDIA_RESPONSE_MODE", "NVIDIA_RESPONSE_MODE") or "json_schema"
    adapter = HttpModelAdapter(
        base_url,
        spec.model_id,
        api_key,
        timeout_seconds=timeout_seconds,
        endpoint_path=_env_first("DIVY_NVIDIA_ENDPOINT_PATH", "NVIDIA_ENDPOINT_PATH") or "/chat/completions",
        response_mode=response_mode,
        user_agent="divy-ai/nvidia-provider",
    )
    metadata = ModelMetadata(
        model_id=spec.model_id,
        provider="nvidia",
        capabilities=spec.capabilities,
        context_limit=spec.context_limit,
        cost_estimate=None,
        latency_estimate_ms=None,
        reliability=0.5,
        available=True,
        tags=frozenset({"nvidia-api-catalog", spec.role.replace(" / ", "-").replace(" ", "-")}),
        supported_response_modes=frozenset({"json_schema", "json_object", "none"}) if spec.structured_output else frozenset({"none"}),
        tool_calling=spec.tool_calling,
        multimodal=spec.multimodal,
        reasoning=spec.reasoning,
        coding=spec.coding,
        agentic=spec.agentic,
        recommended_task_types=spec.recommended_task_types,
        documentation_sources=spec.documentation_sources,
    )
    return ModelProvider(metadata, adapter)


def nvidia_registry_from_environment(model_ids: list[str] | None = None) -> ModelRegistry:
    """Build a registry for the configured NVIDIA portfolio.

    The default is one model so a normal live experiment is bounded. Set
    ``DIVY_NVIDIA_MODEL_IDS`` to a comma-separated subset for controlled
    multi-model comparisons.
    """

    configured = model_ids
    if configured is None:
        configured_text = _env_first("DIVY_NVIDIA_MODEL_IDS")
        configured = [item.strip() for item in configured_text.split(",") if item.strip()] if configured_text else None
    if not configured:
        configured = [_env_first("DIVY_NVIDIA_MODEL_ID", "NVIDIA_MODEL_ID", "DIVY_MODEL_ID") or NVIDIA_MODEL_SPECS[0].model_id]
    return ModelRegistry([nvidia_provider_from_environment(item) for item in configured])
