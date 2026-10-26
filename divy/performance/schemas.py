"""Structured representation of an observed model execution.

The field order and names are the integration contract consumed by the
core DIVY orchestrator (``divy.execution.MultiModelExecutor._record_performance``
constructs these with keyword arguments). Positional construction is also
supported so the record can be built compactly.

Only fields that are actually measurable are recorded. Optional metrics are
represented explicitly as ``None`` rather than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceObservation:
    """A single observed model call / execution captured for performance history."""

    # --- identity (positional fields 1-3) -------------------------------
    model_id: str
    task_type: str | None = None
    capabilities: tuple[str, ...] = ()

    # --- per-call outcome (positional fields 4-7) -----------------------
    provider_success: bool = False
    contributed_to_final: bool = False
    task_success: bool = False
    verification_passed: bool | None = None

    # --- efficiency / economics (positional fields 8-10) ----------------
    latency_ms: float | None = None
    cost: float | None = None
    token_count: int | None = None

    # --- trajectory context (positional fields 11-12) -------------------
    tool_calls: int = 0
    recovery_attempts: int = 0

    # --- bookkeeping / distinct mission-spec fields ---------------------
    execution_id: str | None = None
    timestamp: str | None = None
    provider: str | None = None
    task_id: str | None = None
    failure_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    final_outcome: str | None = None
    accepted_model_id: str | None = None
    fallback_used: bool = False
    attempt_index: int | None = None
    role: str | None = None

    @property
    def success(self) -> bool:
        """Alias for ``task_success`` so the mission-spec language works."""
        return self.task_success

    @success.setter
    def success(self, value: bool) -> None:
        self.task_success = bool(value)

    @property
    def verification_result(self) -> bool | None:
        return self.verification_passed

    @verification_result.setter
    def verification_result(self, value: bool | None) -> None:
        self.verification_passed = None if value is None else bool(value)

    @property
    def estimated_cost(self) -> float | None:
        return self.cost

    @estimated_cost.setter
    def estimated_cost(self, value: float | None) -> None:
        self.cost = None if value is None else float(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "provider": self.provider,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "capabilities": list(self.capabilities),
            "success": self.task_success,
            "verification_result": self.verification_passed,
            "failure_type": self.failure_type,
            "model_calls": 1,
            "tool_calls": self.tool_calls,
            "recovery_attempts": self.recovery_attempts,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": self.cost,
            "final_outcome": self.final_outcome,
            "accepted_model_id": self.accepted_model_id,
            "fallback_used": self.fallback_used,
            "attempt_index": self.attempt_index,
            "role": self.role,
            # Integration-contract names, preserved verbatim.
            "provider_success": self.provider_success,
            "contributed_to_final": self.contributed_to_final,
            "task_success": self.task_success,
            "verification_passed": self.verification_passed,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "token_count": self.token_count,
        }
