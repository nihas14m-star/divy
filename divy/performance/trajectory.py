"""Adapter layer that transforms DIVY execution trajectories into
structured performance observations.

DIVY already records trajectory/event information (orchestrator.run, the
MultiModelExecutor, and benchmark runners). This module does not duplicate
that logging architecture; it reads the existing structures and produces a
PerformanceObservation suitable for the performance store.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .schemas import PerformanceObservation


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_total(token_usage: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not token_usage:
        return None, None
    return (
        token_usage.get("input_tokens") if token_usage.get("input_tokens") is not None else token_usage.get("prompt_tokens"),
        token_usage.get("output_tokens") if token_usage.get("output_tokens") is not None else token_usage.get("completion_tokens"),
    )


def _token_count(token_usage: dict[str, Any] | None) -> int | None:
    """Return a non-duplicated token total, preserving missing usage."""
    if not token_usage:
        return None
    total = token_usage.get("total_tokens")
    if isinstance(total, (int, float)):
        return int(total)
    input_tokens, output_tokens = _token_total(token_usage)
    if input_tokens is None and output_tokens is None:
        return None
    return int((input_tokens or 0) + (output_tokens or 0))


def trajectory_to_observation(
    *,
    execution_id: str | None = None,
    timestamp: str | None = None,
    model_id: str,
    provider: str | None = None,
    task_id: str,
    task_type: str | None = None,
    capabilities: tuple[str, ...] | None = None,
    trajectory: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    cost: float | None = None,
    token_count: int | None = None,
    final_outcome: str | None = None,
) -> PerformanceObservation:
    """Build a PerformanceObservation from DIVY's existing trajectory data.

    Only fields actually present in the inputs are recorded; absent optional
    metrics are left as ``None``. Nothing is fabricated.

    Accepts any combination of:
      * ``trajectory`` — the orchestrator trajectory dict.
      * ``result`` — an OrchestrationResult-like dict.
      * ``evaluation`` — an EvaluationResult-like dict.
    """
    if execution_id is None:
        execution_id = str(uuid.uuid4())
    if timestamp is None:
        timestamp = _utc_now()

    # --- task success --------------------------------------------------
    task_success = False
    if result is not None and "success" in result:
        task_success = bool(result["success"])
    elif evaluation is not None and "success" in evaluation:
        task_success = bool(evaluation["success"])
    elif trajectory is not None:
        task_success = trajectory.get("status") == "success"

    # provider_success mirrors task_success at the observation level.
    provider_success = task_success

    # --- final outcome -------------------------------------------------
    if final_outcome is None:
        if trajectory is not None and trajectory.get("status"):
            final_outcome = str(trajectory.get("status"))
        elif task_success:
            final_outcome = "success"
        else:
            final_outcome = "failed"

    # --- verification --------------------------------------------------
    verification_passed: bool | None = None
    if result is not None and "verification_passed" in result:
        verification_passed = bool(result["verification_passed"])
    elif evaluation is not None and "verification_passed" in evaluation:
        verification_passed = bool(evaluation["verification_passed"])
    elif trajectory is not None and "verification_result" in trajectory:
        verification_passed = bool(trajectory.get("verification_result"))

    # --- failure type --------------------------------------------------
    failure_type: str | None = None
    if not task_success:
        if trajectory is not None:
            status = trajectory.get("status")
            if status == "failed_invalid_plan":
                failure_type = "invalid_plan"
            elif status == "failed_recovery_limit":
                failure_type = "recovery_limit"
            elif status == "failed_incomplete":
                failure_type = "incomplete"
            elif status == "needs_clarification":
                failure_type = "needs_clarification"
            else:
                failure_type = "execution_failed"
        else:
            failure_type = "execution_failed"

    # --- model call count (one observation per model call) -------------
    model_calls = 0
    if result is not None and "model_calls" in result:
        model_calls = int(result["model_calls"])
    elif trajectory is not None:
        model_calls = sum(
            1 for event in trajectory.get("events", [])
            if event.get("event") == "model_call"
        )
    # Each observation represents one model call; caller may record one
    # observation per call or aggregate model_calls into tool_calls.

    # --- tool calls ----------------------------------------------------
    tool_calls = 0
    if result is not None and "tool_calls" in result:
        tool_calls = int(result["tool_calls"])
    elif trajectory is not None:
        tool_calls = int(trajectory.get("tool_calls", 0))

    # --- recovery attempts ---------------------------------------------
    recovery_attempts = 0
    if result is not None and "recovery_attempts" in result:
        recovery_attempts = int(result["recovery_attempts"])
    elif trajectory is not None:
        recovery_attempts = int(trajectory.get("recovery_attempts", 0))

    # --- latency -------------------------------------------------------
    latency_ms: float | None = None
    if result is not None:
        latency_ms = result.get("total_latency_ms")
        if latency_ms is None:
            latency_ms = result.get("wall_latency_ms")
    if latency_ms is None and trajectory is not None:
        latency_ms = trajectory.get("latency_ms")
    if latency_ms is not None:
        latency_ms = float(latency_ms)

    # --- tokens --------------------------------------------------------
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_usage: dict[str, Any] | None = None
    if result is not None and "token_usage" in result:
        token_usage = result["token_usage"]
    elif trajectory is not None:
        token_usage = trajectory.get("token_usage")
    input_tokens, output_tokens = _token_total(token_usage)
    if token_count is None and token_usage:
        token_count = _token_count(token_usage)

    # --- cost ----------------------------------------------------------
    if cost is None and result is not None and "total_cost" in result:
        cost = result["total_cost"]
    if cost is not None:
        cost = float(cost)

    capabilities = tuple(capabilities or ())

    return PerformanceObservation(
        execution_id=execution_id,
        timestamp=timestamp,
        model_id=model_id,
        provider=provider,
        task_id=task_id,
        task_type=task_type,
        capabilities=capabilities,
        provider_success=provider_success,
        contributed_to_final=provider_success,
        task_success=task_success,
        verification_passed=verification_passed,
        latency_ms=latency_ms,
        cost=cost,
        token_count=token_count,
        tool_calls=tool_calls,
        recovery_attempts=recovery_attempts,
        failure_type=failure_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        final_outcome=final_outcome,
    )


def observation_from_orchestration_result(
    *,
    execution_id: str | None = None,
    timestamp: str | None = None,
    model_id: str,
    provider: str | None = None,
    task_id: str,
    task_type: str | None = None,
    capabilities: tuple[str, ...] | None = None,
    result: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
    cost: float | None = None,
) -> PerformanceObservation:
    """Convenience builder from an execution result dict (e.g. benchmark row)."""
    return trajectory_to_observation(
        execution_id=execution_id,
        timestamp=timestamp,
        model_id=model_id,
        provider=provider,
        task_id=task_id,
        task_type=task_type,
        capabilities=capabilities,
        result=result,
        evaluation=evaluation,
        cost=cost,
    )


def observations_from_agent_session(
    trajectory: dict[str, Any],
    *,
    execution_id: str | None = None,
    timestamp: str | None = None,
) -> list[PerformanceObservation]:
    """Extract per-model-call observations from an AgentSession trajectory.

    AgentSession records one trajectory with multiple model_response events.
    This function extracts one PerformanceObservation per model_response event,
    preserving the trajectory-level context (tool_calls, recovery_attempts, etc.)
    for each observation.

    Missing metrics (cost, tokens, latency) are left as None rather than fabricated.
    """
    if execution_id is None:
        execution_id = trajectory.get("execution_id") or str(uuid.uuid4())
    if timestamp is None:
        timestamp = _utc_now()

    observations = []
    model_events = [e for e in trajectory.get("events", []) if e.get("event") == "model_response"]

    # Determine the final successful model response (if any)
    successful_events = [e for e in model_events if e.get("status") == "success"]
    final_event = successful_events[-1] if successful_events else None

    task_success = trajectory.get("status") == "success"
    provider_success_default = task_success

    for event in model_events:
        event_timestamp = event.get("timestamp") or timestamp
        event_latency = event.get("latency_ms")
        # A trajectory-level cost is an aggregate. Do not invent an equal
        # per-call split when the provider did not report event-level costs.
        event_cost = event.get("cost") if "cost" in event else None

        # Extract token usage from the event
        token_usage = event.get("token_usage") or {}
        token_count = None
        if token_usage:
            token_count = _token_count(token_usage)

        event_provider_success = event.get("status") == "success"
        event_contributed = task_success and event is final_event

        failure_type = None
        if not event_provider_success:
            failure_type = (event.get("error") or {}).get("type") or trajectory.get("status")

        observations.append(PerformanceObservation(
            execution_id=execution_id,
            timestamp=event_timestamp,
            model_id=trajectory.get("model_id") or "unknown",
            provider=trajectory.get("provider"),
            task_id=trajectory.get("task_id"),
            task_type=trajectory.get("task_type"),
            capabilities=tuple(trajectory.get("capabilities") or ()),
            provider_success=event_provider_success,
            contributed_to_final=event_contributed,
            task_success=task_success,
            verification_passed=trajectory.get("verification_result"),
            latency_ms=event_latency,
            cost=event_cost,
            token_count=token_count,
            tool_calls=trajectory.get("tool_calls", 0),
            recovery_attempts=trajectory.get("recovery_attempts", 0),
            failure_type=failure_type,
            input_tokens=token_usage.get("input_tokens"),
            output_tokens=token_usage.get("output_tokens"),
            final_outcome=trajectory.get("status"),
        ))

    return observations
