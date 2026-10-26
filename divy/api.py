"""Public provider-neutral orchestration entry point.

This module is intentionally a thin composition layer. It keeps routing,
structured multi-model execution, and the bounded model-to-tool session as
separate primitives while giving applications one entry point that owns the
route, execution, verification, fallback, and outcome attribution flow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from .agent_loop import AgentLimits, AgentSession, AgentSessionResult
from .execution import MultiModelExecutor, OrchestrationResult
from .logger import EventLogger
from .performance import PerformanceMemory
from .providers import ModelRegistry
from .routing import DeterministicRoutingPolicy, RoutingDecision, RoutingPolicy, RoutingRequest, TaskProfile
from .schemas import EvaluationResult
from .tools.registry import ToolRegistry


@dataclass(frozen=True)
class OrchestrationConstraints:
    """Provider-neutral limits for one orchestration request."""

    max_cost: float | None = None
    max_models: int = 1
    parallel: bool = False
    verify: bool = False
    synthesize: bool = False
    agent_limits: AgentLimits | None = None

    def __post_init__(self) -> None:
        if self.max_models < 1:
            raise ValueError("max_models must be positive")
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        if self.parallel and self.max_models < 2:
            raise ValueError("parallel orchestration requires max_models >= 2")


@dataclass
class OrchestrationOutcome:
    """Unified result returned by :func:`orchestrate`."""

    task_id: str
    task: str
    profile: TaskProfile
    decision: RoutingDecision
    mode: str
    success: bool
    final_output: dict[str, Any] | None = None
    accepted_model_id: str | None = None
    execution: OrchestrationResult | None = None
    session_attempts: list[AgentSessionResult] = field(default_factory=list)

    @property
    def explanation(self) -> dict[str, Any]:
        return self.decision.explanation

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_id": self.task_id,
            "task": self.task,
            "profile": _profile_dict(self.profile),
            "decision": self.decision.as_dict(),
            "mode": self.mode,
            "success": self.success,
            "final_output": self.final_output,
            "accepted_model_id": self.accepted_model_id,
        }
        if self.execution is not None:
            data["execution"] = {
                "success": self.execution.success,
                "fallback_used": self.execution.fallback_used,
                "verification_attempted": self.execution.verification_attempted,
                "verification_passed": self.execution.verification_passed,
                "model_calls": self.execution.model_calls,
                "total_cost": self.execution.total_cost,
                "total_latency_ms": self.execution.total_latency_ms,
            }
        if self.session_attempts:
            data["session_attempts"] = [
                {
                    "status": attempt.status,
                    "model_id": attempt.trajectory.get("model_id"),
                    "verification_result": attempt.trajectory.get("verification_result"),
                    "model_calls": attempt.trajectory.get("model_calls", 0),
                    "tool_calls": attempt.trajectory.get("tool_calls", 0),
                    "recovery_attempts": attempt.trajectory.get("recovery_attempts", 0),
                }
                for attempt in self.session_attempts
            ]
        return data


def orchestrate(
    task: str,
    *,
    task_id: str,
    registry: ModelRegistry,
    requirements: TaskProfile | None = None,
    performance_memory: PerformanceMemory | None = None,
    policy: RoutingPolicy | None = None,
    context: list[dict[str, Any]] | None = None,
    tool_registry: ToolRegistry | None = None,
    response_schema: dict[str, Any] | None = None,
    verifier: Callable[[dict[str, Any]], EvaluationResult] | None = None,
    system_instructions: str = "Return structured output and use only registered tools.",
    constraints: OrchestrationConstraints | None = None,
    logger: EventLogger | None = None,
) -> OrchestrationOutcome:
    """Route and execute one provider-neutral orchestration request.

    Without ``tool_registry`` this uses ``MultiModelExecutor`` for structured
    model calls, optional parallel candidates, synthesis, and model-based
    verification. With ``tool_registry`` it uses bounded ``AgentSession``
    attempts and tries the routed fallback list after a failed attempt.

    The existing lower-level classes remain available for benchmark and
    specialized integrations. This function is the recommended application
    entry point.
    """
    profile = requirements or TaskProfile()
    limits = constraints or OrchestrationConstraints(
        verify=profile.verification_required,
    )
    verify_required = limits.verify or profile.verification_required or verifier is not None
    routing_policy = policy or DeterministicRoutingPolicy(performance_memory=performance_memory)
    request = RoutingRequest(
        task_id=task_id,
        required_capabilities=profile.required_capabilities,
        complexity=profile.expected_difficulty,
        critical=profile.risk_level == "high",
        max_models=limits.max_models,
        allow_parallel=limits.parallel,
        budget=limits.max_cost,
        task_type=profile.task_type,
        profile=profile,
    )
    decision = routing_policy.route(request, registry)

    if tool_registry is not None:
        return _orchestrate_tool_loop(
            task,
            task_id,
            profile,
            decision,
            registry,
            tool_registry,
            performance_memory,
            logger,
            response_schema,
            verifier,
            system_instructions,
            limits,
            verify_required,
            context,
        )

    messages = list(context or [])
    messages.append({"role": "user", "content": task})
    execution = MultiModelExecutor(
        registry,
        logger=logger,
        performance_memory=performance_memory,
    ).execute(
        decision,
        messages,
        [],
        response_schema or {"type": "object"},
        parallel=limits.parallel,
        verify=verify_required,
        synthesize=limits.synthesize,
        task_type=profile.task_type,
        required_capabilities=set(profile.required_capabilities),
        max_cost=limits.max_cost,
        result_verifier=verifier,
    )
    return OrchestrationOutcome(
        task_id=task_id,
        task=task,
        profile=profile,
        decision=decision,
        mode="structured",
        success=execution.success,
        final_output=execution.final_output,
        accepted_model_id=execution.accepted_model_id,
        execution=execution,
    )


def _orchestrate_tool_loop(
    task: str,
    task_id: str,
    profile: TaskProfile,
    decision: RoutingDecision,
    registry: ModelRegistry,
    tool_registry: ToolRegistry,
    performance_memory: PerformanceMemory | None,
    logger: EventLogger | None,
    response_schema: dict[str, Any] | None,
    verifier: Callable[[dict[str, Any]], EvaluationResult] | None,
    system_instructions: str,
    constraints: OrchestrationConstraints,
    verify_required: bool,
    context: list[dict[str, Any]] | None,
) -> OrchestrationOutcome:
    attempts: list[AgentSessionResult] = []
    candidate_ids = list(dict.fromkeys(decision.selected_model_ids + decision.fallback_model_ids))
    accepted_model_id: str | None = None
    final_output: dict[str, Any] | None = None
    for index, model_id in enumerate(candidate_ids):
        provider = registry.get(model_id)
        agent_limits = constraints.agent_limits or AgentLimits()
        if constraints.max_cost is not None and (
            agent_limits.max_cost is None or constraints.max_cost < agent_limits.max_cost
        ):
            agent_limits = replace(agent_limits, max_cost=constraints.max_cost)
        session = AgentSession(
            provider,
            tool_registry,
            logger=logger,
            performance_memory=performance_memory,
            limits=agent_limits,
        )
        attempt = session.run(
            task,
            task_id=task_id,
            task_type=profile.task_type or task_id,
            capabilities=set(profile.required_capabilities),
            system_instructions=system_instructions,
            response_schema=response_schema,
            verifier=verifier,
            verification_required=verify_required,
            context=context,
        )
        attempts.append(attempt)
        if attempt.status == "success":
            accepted_model_id = model_id
            final_output = attempt.final_output
            break
        if index == 0 and not decision.fallback_model_ids:
            break
    return OrchestrationOutcome(
        task_id=task_id,
        task=task,
        profile=profile,
        decision=decision,
        mode="tool_loop",
        success=accepted_model_id is not None,
        final_output=final_output,
        accepted_model_id=accepted_model_id,
        session_attempts=attempts,
    )


def _profile_dict(profile: TaskProfile) -> dict[str, Any]:
    data = asdict(profile)
    data["required_capabilities"] = sorted(profile.required_capabilities)
    data["required_response_modes"] = sorted(profile.required_response_modes)
    return data
