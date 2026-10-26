"""Learned router dataset export (points #11-#12).

Produces a clean, exportable dataset format for a future learned DIVY router.
Each example conceptually contains:

    task characteristics
    available models
    capabilities
    historical evidence
    routing decision
    execution trajectory
    verification
    final outcome
    cost
    latency

The dataset deliberately distinguishes:

    teacher recommendation      (what the routing policy chose)
    actual verified outcome     (what actually happened, independently verified)

No model is trained here. This module only prepares high-quality data.

Quality gate (point #12): examples are rejected or flagged when they have
failed verification, missing outcomes, malformed trajectories, contradictory
metrics, insufficient evidence, unsafe execution, or corrupted records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluators import EvaluationResult
from .performance import PerformanceMemory


@dataclass
class RouterDatasetExample:
    """One example for a future learned router."""

    # Task characteristics
    task_id: str
    task_type: str | None
    required_capabilities: list[str]
    complexity: str

    # Available models at routing time
    available_models: list[str]
    model_capabilities: dict[str, list[str]]

    # Historical evidence (at routing time, before this execution)
    historical_evidence: dict[str, dict[str, Any]]

    # Routing decision (teacher recommendation)
    routing_decision: dict[str, Any]
    selected_model_id: str

    # Execution results
    execution_success: bool
    verification_passed: bool | None
    final_outcome: str | None
    latency_ms: float | None
    estimated_cost: float | None
    model_calls: int
    tool_calls: int
    recovery_attempts: int

    # Quality gate
    rejected: bool = False
    reject_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "required_capabilities": self.required_capabilities,
            "complexity": self.complexity,
            "available_models": self.available_models,
            "model_capabilities": self.model_capabilities,
            "historical_evidence": self.historical_evidence,
            "routing_decision": self.routing_decision,
            "selected_model_id": self.selected_model_id,
            "execution_success": self.execution_success,
            "verification_passed": self.verification_passed,
            "final_outcome": self.final_outcome,
            "latency_ms": self.latency_ms,
            "estimated_cost": self.estimated_cost,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "recovery_attempts": self.recovery_attempts,
            "rejected": self.rejected,
            "reject_reasons": self.reject_reasons,
        }


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return value == value and abs(value) != float("inf")
    except TypeError:
        return False


def apply_quality_gate(example: RouterDatasetExample) -> RouterDatasetExample:
    """Apply filtering rules for future training data.

    Rejects or flags examples that would produce low-quality labels:
    * failed verification
    * missing outcome
    * malformed/contradictory metrics
    * insufficient evidence
    * unsafe execution
    * corrupted records

    Returns the example (possibly mutated with rejected/reject_reasons).
    """
    reasons: list[str] = []

    # Missing outcome.
    if example.final_outcome is None:
        reasons.append("missing_outcome")

    # Failed verification -> the label is unreliable for "did it actually work".
    if example.verification_passed is False:
        reasons.append("failed_verification")
    if example.verification_passed is None:
        reasons.append("missing_verification")

    # Contradictory metrics: claims success but outcome says failed.
    if example.execution_success and example.final_outcome in {"failed", "failed_incomplete", "failed_recovery_limit"}:
        reasons.append("contradictory_success_outcome")
    if (not example.execution_success) and example.final_outcome == "success":
        reasons.append("contradictory_failure_outcome")

    # Malformed / non-finite metrics.
    if example.latency_ms is not None and not _is_finite_number(example.latency_ms):
        reasons.append("non_finite_latency")
    if example.estimated_cost is not None and not _is_finite_number(example.estimated_cost):
        reasons.append("non_finite_cost")
    if example.model_calls < 0 or example.tool_calls < 0 or example.recovery_attempts < 0:
        reasons.append("negative_count_metric")

    # Insufficient evidence: no historical observations for the selected model.
    sel = example.selected_model_id
    sel_hist = example.historical_evidence.get(sel, {})
    if sel_hist.get("sample_size", sel_hist.get("observation_count", 0)) == 0:
        reasons.append("no_historical_evidence_for_selected_model")

    # Unsafe execution: any safety violation recorded in the trajectory.
    # (Carried as a metric on the example if present.)
    if example.historical_evidence.get("_safety_violation"):
        reasons.append("unsafe_execution")

    example.rejected = len(reasons) > 0
    example.reject_reasons = reasons
    return example


def export_examples(
    examples: list[RouterDatasetExample],
    path: Path | str,
    *,
    include_rejected: bool = False,
) -> dict[str, Any]:
    """Write examples as JSONL. Returns a summary of the export.

    Rejected examples are excluded by default to keep the training dataset
    high-quality (point #12). Set include_rejected=True to keep them for
    analysis.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(examples)
    kept = 0
    rejected = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            apply_quality_gate(example)
            if example.rejected and not include_rejected:
                rejected += 1
                continue
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
            kept += 1

    return {
        "total_examples": total,
        "kept": kept,
        "rejected": rejected,
        "path": str(out_path),
        "include_rejected": include_rejected,
    }


def example_from_execution(
    *,
    task_id: str,
    task_type: str | None,
    required_capabilities: list[str],
    complexity: str,
    available_models: list[str],
    model_capabilities: dict[str, list[str]],
    historical_evidence: dict[str, dict[str, Any]],
    routing_decision: dict[str, Any],
    selected_model_id: str,
    execution_success: bool,
    verification_passed: bool | None,
    final_outcome: str | None,
    latency_ms: float | None,
    estimated_cost: float | None,
    model_calls: int,
    tool_calls: int,
    recovery_attempts: int,
) -> RouterDatasetExample:
    """Construct one dataset example from a completed execution."""
    return RouterDatasetExample(
        task_id=task_id,
        task_type=task_type,
        required_capabilities=list(required_capabilities),
        complexity=complexity,
        available_models=list(available_models),
        model_capabilities={k: list(v) for k, v in model_capabilities.items()},
        historical_evidence=dict(historical_evidence),
        routing_decision=dict(routing_decision),
        selected_model_id=selected_model_id,
        execution_success=execution_success,
        verification_passed=verification_passed,
        final_outcome=final_outcome,
        latency_ms=latency_ms,
        estimated_cost=estimated_cost,
        model_calls=model_calls,
        tool_calls=tool_calls,
        recovery_attempts=recovery_attempts,
    )


def example_from_agent_session(
    *,
    task_id: str,
    task_type: str | None,
    required_capabilities: set[str],
    complexity: str,
    routing_decision: dict[str, Any],
    trajectory: dict[str, Any],
    verification_passed: bool | None,
    final_outcome: str | None,
    latency_ms: float | None,
    estimated_cost: float | None,
    model_calls: int,
    tool_calls: int,
    recovery_attempts: int,
) -> RouterDatasetExample:
    """Construct a dataset example from an AgentSession trajectory and routing decision.

    This is the primary function for creating training examples from real
    AgentSession executions. It extracts:
    - Task characteristics from the routing decision and trajectory
    - Available models and their capabilities from the routing decision
    - Historical evidence (empty at execution time, populated later)
    - Routing decision (teacher recommendation)
    - Actual execution results from the trajectory
    """
    # Extract available models and capabilities from routing decision
    selected_models = routing_decision.get("selected_model_ids", [])
    fallback_models = routing_decision.get("fallback_model_ids", [])
    candidate_models = routing_decision.get("candidate_model_ids", [])
    available_models = candidate_models or selected_models + fallback_models

    # Extract required capabilities from routing decision or trajectory
    caps = required_capabilities or set(trajectory.get("capabilities") or ())

    # Model capabilities from candidate models
    model_capabilities = {}
    for model_id in available_models:
        # We don't have capability info in the routing decision alone,
        # so this would need to be populated from the registry
        model_capabilities[model_id] = sorted(caps)

    return RouterDatasetExample(
        task_id=task_id,
        task_type=task_type or trajectory.get("task_type"),
        required_capabilities=sorted(caps),
        complexity=complexity,
        available_models=available_models,
        model_capabilities=model_capabilities,
        historical_evidence={},  # Populated later from performance memory
        routing_decision=dict(routing_decision),
        selected_model_id=selected_models[0] if selected_models else "",
        execution_success=trajectory.get("status") == "success",
        verification_passed=verification_passed,
        final_outcome=final_outcome,
        latency_ms=latency_ms,
        estimated_cost=estimated_cost,
        model_calls=model_calls,
        tool_calls=tool_calls,
        recovery_attempts=recovery_attempts,
    )


def enrich_example_with_history(
    example: RouterDatasetExample,
    performance_memory: PerformanceMemory,
) -> RouterDatasetExample:
    """Enrich a dataset example with historical evidence from performance memory.

    This should be called after example_from_agent_session to populate
    the historical_evidence field with actual observations from the
    performance memory at the time of the routing decision.
    """
    for model_id in example.available_models:
        stats = performance_memory.summarize(model_id=model_id, task_type=example.task_type)
        example.historical_evidence[model_id] = stats
    return example
