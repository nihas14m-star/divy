"""Evaluation and reporting layer for DIVY executions.

This module provides a clean evaluation API that consumes completed DIVY
executions (from AgentSession, MultiModelExecutor, or Orchestrator) and
produces structured reports suitable for analysis, comparison, and
dataset export.

It is provider-neutral and does not modify the execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import RouterDatasetExample, example_from_agent_session
from .performance import PerformanceMemory
from .schemas import EvaluationResult


@dataclass
class ModelContribution:
    """Contribution of a single model to the execution."""
    model_id: str
    provider: str | None
    calls: int
    provider_success: bool
    contributed_to_final: bool
    task_success: bool
    verification_passed: bool | None
    latency_ms: float | None = None
    cost: float | None = None
    token_count: int | None = None
    failure_type: str | None = None


@dataclass
class RoutingInfo:
    """Information about the routing decision."""
    selected_models: list[str]
    fallback_models: list[str]
    candidate_models: list[str]
    rationale: str
    fallback_used: bool = False


@dataclass
class ExecutionMetrics:
    """Aggregated execution metrics."""
    model_calls: int
    tool_calls: int
    recovery_attempts: int
    latency_ms: float | None
    total_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost: float | None


@dataclass
class QualityMetrics:
    """Quality and verification metrics."""
    task_success: bool
    verification_passed: bool | None
    false_acceptance: bool = False
    unnecessary_actions: int = 0
    verification_accuracy: bool = False


@dataclass
class EvidenceSummary:
    """Historical evidence available at routing time."""
    sample_count: int
    historical_success_rate: float | None
    recent_success_rate: float | None
    contribution_rate: float | None
    uncertainty: float | None


@dataclass(frozen=True)
class ExecutionMode:
    """Execution mode identifier."""
    value: str

    @staticmethod
    def scripted() -> "ExecutionMode":
        return ExecutionMode("scripted")

    @staticmethod
    def live() -> "ExecutionMode":
        return ExecutionMode("live")

    def __str__(self) -> str:
        return self.value


@dataclass
class EvaluationReport:
    """Complete evaluation report for a DIVY execution."""

    # Identity
    execution_id: str
    task_id: str
    task_type: str | None
    execution_mode: ExecutionMode
    timestamp: str

    # Routing
    routing: RoutingInfo

    # Execution
    execution: ExecutionMetrics

    # Quality
    quality: QualityMetrics

    # Model contributions
    model_contributions: list[ModelContribution]

    # Evidence
    evidence: EvidenceSummary

    # Raw trajectory (optional, for debugging)
    trajectory: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "execution_mode": str(self.execution_mode),
            "timestamp": self.timestamp,
            "routing": {
                "selected_models": self.routing.selected_models,
                "fallback_models": self.routing.fallback_models,
                "candidate_models": self.routing.candidate_models,
                "rationale": self.routing.rationale,
                "fallback_used": self.routing.fallback_used,
            },
            "execution": {
                "model_calls": self.execution.model_calls,
                "tool_calls": self.execution.tool_calls,
                "recovery_attempts": self.execution.recovery_attempts,
                "latency_ms": self.execution.latency_ms,
                "total_tokens": self.execution.total_tokens,
                "input_tokens": self.execution.input_tokens,
                "output_tokens": self.execution.output_tokens,
                "estimated_cost": self.execution.estimated_cost,
            },
            "quality": {
                "task_success": self.quality.task_success,
                "verification_passed": self.quality.verification_passed,
                "false_acceptance": self.quality.false_acceptance,
                "unnecessary_actions": self.quality.unnecessary_actions,
                "verification_accuracy": self.quality.verification_accuracy,
            },
            "model_contributions": [
                {
                    "model_id": m.model_id,
                    "provider": m.provider,
                    "calls": m.calls,
                    "provider_success": m.provider_success,
                    "contributed_to_final": m.contributed_to_final,
                    "task_success": m.task_success,
                    "verification_passed": m.verification_passed,
                    "latency_ms": m.latency_ms,
                    "cost": m.cost,
                    "token_count": m.token_count,
                    "failure_type": m.failure_type,
                }
                for m in self.model_contributions
            ],
            "evidence": {
                "sample_count": self.evidence.sample_count,
                "historical_success_rate": self.evidence.historical_success_rate,
                "recent_success_rate": self.evidence.recent_success_rate,
                "contribution_rate": self.evidence.contribution_rate,
                "uncertainty": self.evidence.uncertainty,
            },
        }

    def to_router_dataset_example(
        self,
        required_capabilities: set[str] | None = None,
        complexity: str = "normal",
        historical_evidence: dict[str, dict[str, Any]] | None = None,
    ) -> RouterDatasetExample:
        """Convert this evaluation report to a RouterDatasetExample.

        This preserves all information needed for the learning dataset.
        """
        # Determine selected model (first selected model)
        selected_model = self.routing.selected_models[0] if self.routing.selected_models else ""

        # Determine available models from routing decision
        available_models = self.routing.candidate_models or self.routing.selected_models + self.routing.fallback_models

        # Model capabilities - use required capabilities as approximation
        caps = required_capabilities or set()
        model_capabilities = {m: sorted(caps) for m in available_models}

        # Use provided historical evidence or empty dict
        hist_evidence = historical_evidence or {}

        return RouterDatasetExample(
            task_id=self.task_id,
            task_type=self.task_type,
            required_capabilities=sorted(required_capabilities or set()),
            complexity=complexity,
            available_models=available_models,
            model_capabilities=model_capabilities,
            historical_evidence=historical_evidence or {},
            routing_decision={
                "selected_model_ids": self.routing.selected_models,
                "fallback_model_ids": self.routing.fallback_models,
                "candidate_model_ids": self.routing.candidate_models,
                "rationale": self.routing.rationale,
            },
            selected_model_id=selected_model,
            execution_success=self.quality.task_success,
            verification_passed=self.quality.verification_passed,
            final_outcome=self.routing.fallback_used and "fallback" or ("success" if self.quality.task_success else "failed"),
            latency_ms=self.execution.latency_ms,
            estimated_cost=self.execution.estimated_cost,
            model_calls=self.execution.model_calls,
            tool_calls=self.execution.tool_calls,
            recovery_attempts=self.execution.recovery_attempts,
        )


def _extract_model_contributions(
    trajectory: dict[str, Any],
    routing: RoutingInfo,
) -> list[ModelContribution]:
    """Extract per-model contributions from trajectory events."""
    contributions = []

    # Get model response events
    model_events = [e for e in trajectory.get("events", []) if e.get("event") == "model_response"]

    if not model_events:
        # Fallback: use trajectory-level info
        return [ModelContribution(
            model_id=trajectory.get("model_id", "unknown"),
            provider=trajectory.get("provider"),
            calls=trajectory.get("model_calls", 0),
            provider_success=trajectory.get("status") == "success",
            contributed_to_final=trajectory.get("status") == "success",
            task_success=trajectory.get("status") == "success",
            verification_passed=trajectory.get("verification_result"),
            latency_ms=trajectory.get("latency_ms"),
            cost=trajectory.get("cost"),
            token_count=_extract_token_count(trajectory.get("token_usage")),
            failure_type=None if trajectory.get("status") == "success" else trajectory.get("status"),
        )]

    # Find the final successful event
    successful_events = [e for e in model_events if e.get("status") == "success"]
    final_event = successful_events[-1] if successful_events else None
    task_success = trajectory.get("status") == "success"

    for event in model_events:
        event_success = event.get("status") == "success"
        contributed = task_success and event is final_event

        failure_type = None
        if not event_success:
            failure_type = (event.get("error") or {}).get("type") or trajectory.get("status")

        contributions.append(ModelContribution(
            model_id=event.get("model_id", trajectory.get("model_id", "unknown")),
            provider=event.get("provider", trajectory.get("provider")),
            calls=1,  # Each event is one call
            provider_success=event_success,
            contributed_to_final=contributed,
            task_success=event.get("attempt_success", task_success),
            verification_passed=event.get("attempt_verification_passed", trajectory.get("verification_result")),
            latency_ms=event.get("latency_ms"),
            cost=_per_call_cost(event, trajectory, len(model_events)),
            token_count=_extract_token_count(event.get("token_usage")),
            failure_type=failure_type,
        ))

    return contributions


def _extract_token_count(token_usage: dict[str, Any] | None) -> int | None:
    if not token_usage:
        return None
    total = token_usage.get("total_tokens")
    if isinstance(total, (int, float)):
        return int(total)
    input_tokens = token_usage.get("input_tokens", token_usage.get("prompt_tokens"))
    output_tokens = token_usage.get("output_tokens", token_usage.get("completion_tokens"))
    if input_tokens is not None or output_tokens is not None:
        return int((input_tokens or 0) + (output_tokens or 0))
    return None


def _per_call_cost(event: dict[str, Any], trajectory: dict[str, Any], call_count: int) -> float | None:
    event_cost = event.get("cost")
    if "cost" in event:
        return float(event_cost) if event_cost is not None else None
    total_cost = trajectory.get("cost")
    if total_cost is not None and (call_count == 1 or total_cost == 0):
        return float(total_cost)
    # An aggregate does not identify unequal per-model call costs.
    return None


def _calculate_false_acceptance(
    trajectory: dict[str, Any],
    evaluation_result: EvaluationResult | None,
) -> bool:
    """Determine if the system falsely accepted a failed execution."""
    trajectory_status = trajectory.get("status")
    if evaluation_result is None:
        return False
    return trajectory_status == "success" and not evaluation_result.success


def _calculate_verification_accuracy(
    trajectory: dict[str, Any],
    evaluation_result: EvaluationResult | None,
) -> bool:
    """Check if trajectory's verification_result matches independent evaluation."""
    if evaluation_result is None:
        return False
    return trajectory.get("verification_result") == evaluation_result.verification_passed


def _calculate_unnecessary_actions(trajectory: dict[str, Any], evaluation_result: EvaluationResult | None) -> int:
    """Count unnecessary actions from evaluation result."""
    if evaluation_result is None:
        return 0
    return evaluation_result.unnecessary_actions or 0


def _extract_evidence(
    trajectory: dict[str, Any],
    performance_memory: PerformanceMemory | None,
    task_type: str | None,
    selected_model: str | None,
    pre_run_history: dict[str, dict[str, Any]] | None = None,
) -> EvidenceSummary:
    """Extract historical evidence summary for the selected model.

    Uses pre_run_history if provided (evidence available BEFORE the trial),
    otherwise falls back to reading from performance_memory (which may include
    the current trial's observations - use with caution).
    """
    if not selected_model:
        return EvidenceSummary(0, None, None, None, None)

    # Use pre-run history if available (correct: evidence before trial)
    if pre_run_history is not None and selected_model in pre_run_history:
        historical = pre_run_history[selected_model]
        sample_count = historical.get("observation_count", 0)
        historical_success = historical.get("provider_success_rate")
        contribution_rate = historical.get("contribution_rate")
    elif performance_memory is not None:
        # Fallback: may include post-trial observations (use only when pre_run_history unavailable)
        historical = performance_memory.summarize(model_id=selected_model, task_type=task_type)
        sample_count = historical.get("observation_count", 0)
        historical_success = historical.get("provider_success_rate")
        contribution_rate = historical.get("contribution_rate")
    else:
        return EvidenceSummary(0, None, None, None, None)

    # Recent stats (recency-weighted) - use performance_memory if available
    recent_success = None
    if sample_count > 0 and performance_memory is not None and pre_run_history is None:
        try:
            from .performance.analytics import PerformanceAnalytics
            analytics = PerformanceAnalytics(performance_memory)
            # Filter by task_type for task-specific recency
            recency = analytics.recency_model_stats(selected_model, half_life_days=30.0, reference_time=None, task_type=task_type)
            recent_success = recency.get("weighted_success_rate")
        except Exception:
            pass

    # Simple uncertainty measure: inverse of sample count, capped
    uncertainty = None
    if sample_count > 0:
        uncertainty = 1.0 / (sample_count ** 0.5)

    return EvidenceSummary(
        sample_count=sample_count,
        historical_success_rate=historical_success,
        recent_success_rate=recent_success,
        contribution_rate=contribution_rate,
        uncertainty=uncertainty,
    )


def evaluate_execution(
    trajectory: dict[str, Any],
    *,
    evaluation_result: EvaluationResult | None = None,
    routing_decision: dict[str, Any] | None = None,
    performance_memory: PerformanceMemory | None = None,
    execution_mode: ExecutionMode = ExecutionMode.scripted(),
    pre_run_history: dict[str, dict[str, Any]] | None = None,
) -> EvaluationReport:
    """Evaluate a completed DIVY execution and produce a structured report.

    Args:
        trajectory: The execution trajectory from AgentSession, MultiModelExecutor, or Orchestrator
        evaluation_result: Optional independent EvaluationResult for quality metrics
        routing_decision: The routing decision dict (from DeterministicRoutingPolicy.route())
        performance_memory: Optional PerformanceMemory for historical evidence
        execution_mode: scripted or live execution mode
        pre_run_history: Optional pre-run historical evidence (evidence available BEFORE this trial).
                         If provided, this is used for evidence summary instead of performance_memory
                         to prevent post-outcome leakage.

    Returns:
        EvaluationReport with structured evaluation data
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    execution_id = trajectory.get("execution_id", "")
    task_id = trajectory.get("task_id", "")
    task_type = trajectory.get("task_type")

    # Routing info
    routing = RoutingInfo(
        selected_models=routing_decision.get("selected_model_ids", []) if routing_decision else [],
        fallback_models=routing_decision.get("fallback_model_ids", []) if routing_decision else [],
        candidate_models=routing_decision.get("candidate_model_ids", []) if routing_decision else [],
        rationale=routing_decision.get("rationale", "") if routing_decision else "",
        fallback_used=trajectory.get("fallback_used", False),
    )

    # Execution metrics
    token_usage = trajectory.get("token_usage", {})
    total_tokens = _extract_token_count(token_usage)
    input_tokens = token_usage.get("input_tokens", token_usage.get("prompt_tokens"))
    output_tokens = token_usage.get("output_tokens", token_usage.get("completion_tokens"))

    execution = ExecutionMetrics(
        model_calls=trajectory.get("model_calls", 0),
        tool_calls=trajectory.get("tool_calls", 0),
        recovery_attempts=trajectory.get("recovery_attempts", 0),
        latency_ms=trajectory.get("latency_ms"),
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=trajectory.get("cost"),
    )

    # Quality metrics
    task_success = trajectory.get("status") == "success"
    verification_passed = trajectory.get("verification_result")
    false_acceptance = _calculate_false_acceptance(trajectory, evaluation_result)
    unnecessary_actions = _calculate_unnecessary_actions(trajectory, evaluation_result)
    verification_accuracy = _calculate_verification_accuracy(trajectory, evaluation_result)

    quality = QualityMetrics(
        task_success=task_success,
        verification_passed=verification_passed,
        false_acceptance=false_acceptance,
        unnecessary_actions=unnecessary_actions,
        verification_accuracy=verification_accuracy,
    )

    # Model contributions
    model_contributions = _extract_model_contributions(trajectory, routing)

    # Evidence
    selected_model = routing.selected_models[0] if routing.selected_models else None
    evidence = _extract_evidence(trajectory, performance_memory, trajectory.get("task_type"), selected_model, pre_run_history)

    return EvaluationReport(
        execution_id=execution_id,
        task_id=task_id,
        task_type=task_type,
        execution_mode=execution_mode,
        timestamp=timestamp,
        routing=routing,
        execution=execution,
        quality=quality,
        model_contributions=model_contributions,
        evidence=evidence,
        trajectory=trajectory,
    )


def compare_routing_strategies(
    task_id: str,
    task_type: str | None,
    results: dict[str, EvaluationReport],
) -> dict[str, Any]:
    """Compare multiple routing strategies for the same task.

    Args:
        task_id: Task identifier
        task_type: Task type
        results: Dict mapping strategy name -> EvaluationReport

    Returns:
        Comparison dict with raw measurements per strategy
    """
    comparison = {
        "task_id": task_id,
        "task_type": task_type,
        "strategies": {},
    }

    for strategy_name, report in results.items():
        comparison["strategies"][strategy_name] = {
            "success": report.quality.task_success,
            "verification_passed": report.quality.verification_passed,
            "verification_accuracy": report.quality.verification_accuracy,
            "false_acceptance": report.quality.false_acceptance,
            "model_calls": report.execution.model_calls,
            "tool_calls": report.execution.tool_calls,
            "recovery_attempts": report.execution.recovery_attempts,
            "latency_ms": report.execution.latency_ms,
            "total_tokens": report.execution.total_tokens,
            "estimated_cost": report.execution.estimated_cost,
            "fallback_used": report.routing.fallback_used,
            "selected_models": report.routing.selected_models,
        }

    return comparison


def load_trajectory_from_file(path: Path) -> dict[str, Any]:
    """Load a trajectory from a JSONL file (last line)."""
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError("Empty trajectory file")
    import json
    return json.loads(lines[-1])


def load_routing_decision_from_file(path: Path) -> dict[str, Any] | None:
    """Load routing decision from a JSONL file (last line with decision)."""
    # This would need a specific format - for now return None
    return None


def evaluate_from_files(
    trajectory_path: Path,
    *,
    evaluation_result: EvaluationResult | None = None,
    routing_decision: dict[str, Any] | None = None,
    performance_memory_path: Path | None = None,
    execution_mode: ExecutionMode = ExecutionMode.scripted(),
) -> EvaluationReport:
    """Convenience function to evaluate from trajectory files."""
    trajectory = load_trajectory_from_file(trajectory_path)
    memory = None
    if performance_memory_path:
        memory = PerformanceMemory(performance_memory_path)
    return evaluate_execution(
        trajectory,
        evaluation_result=evaluation_result,
        routing_decision=routing_decision,
        performance_memory=memory,
        execution_mode=execution_mode,
    )
