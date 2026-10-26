"""Bounded multi-turn model-to-tool execution loop.

This is the missing runtime bridge between provider responses and DIVY's
registered tools. It deliberately keeps the existing ``ModelAdapter`` and
``ToolRegistry`` contracts unchanged.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .logger import EventLogger
from .adapters import ProviderError
from .model import ModelAdapter
from .performance import PerformanceMemory, PerformanceObservation
from .providers import ModelProvider
from .schemas import EvaluationResult, ModelResponse
from .tools.registry import ToolRegistry


@dataclass(frozen=True)
class AgentLimits:
    max_iterations: int = 12
    max_tool_calls: int = 12
    max_recovery_attempts: int = 2
    max_context_messages: int = 40
    max_cost: float | None = None


@dataclass
class AgentSessionResult:
    status: str
    final_output: dict[str, Any] | None
    trajectory: dict[str, Any]


class AgentSession:
    """Run one bounded model ↔ registered-tool conversation."""

    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry,
        *,
        logger: EventLogger | None = None,
        performance_memory: PerformanceMemory | None = None,
        limits: AgentLimits | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.logger = logger
        self.performance_memory = performance_memory
        self.limits = limits or AgentLimits()

    def run(
        self,
        goal: str,
        *,
        task_id: str,
        task_type: str,
        capabilities: set[str],
        system_instructions: str,
        response_schema: dict[str, Any] | None = None,
        verifier: Callable[[dict[str, Any]], EvaluationResult] | None = None,
    ) -> AgentSessionResult:
        started = time.perf_counter()
        execution_id = str(uuid.uuid4())
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": goal},
        ]
        trajectory: dict[str, Any] = {
            "execution_id": execution_id,
            "task_id": task_id,
            "task_type": task_type,
            "goal": goal,
            "model_id": self.provider.model_id,
            "provider": self.provider.metadata.provider,
            "events": [],
            "model_calls": 0,
            "tool_calls": 0,
            "recovery_attempts": 0,
            "token_usage": {},
            "cost": 0.0 if self.provider.cost() is not None else None,
            "verification_result": False,
            "status": "running",
        }
        final_output: dict[str, Any] | None = None
        response_schema = response_schema or {"type": "object"}

        for iteration in range(1, self.limits.max_iterations + 1):
            trajectory["iteration"] = iteration
            if len(messages) > self.limits.max_context_messages:
                trajectory["status"] = "failed_context_limit"
                break
            estimated_cost = self.provider.cost()
            if self.limits.max_cost is not None:
                if estimated_cost is None:
                    self._record(trajectory, {"event": "agent_stop", "reason": "cost_unknown"})
                    trajectory["status"] = "failed_cost_unknown"
                    break
                if (trajectory["cost"] or 0.0) + estimated_cost > self.limits.max_cost:
                    self._record(trajectory, {"event": "agent_stop", "reason": "cost_limit"})
                    trajectory["status"] = "failed_cost_limit"
                    break

            call_started = time.perf_counter()
            self._record(trajectory, {"event": "model_call", "iteration": iteration, "model_id": self.provider.model_id})
            trajectory["model_calls"] += 1
            if estimated_cost is not None and trajectory["cost"] is not None:
                trajectory["cost"] += estimated_cost
            try:
                response = self.provider.generate(messages, self.registry.schemas(), response_schema)
                elapsed = round((time.perf_counter() - call_started) * 1000, 3)
                if not isinstance(response, ModelResponse) or not isinstance(response.content, dict) or not response.content:
                    raise ValueError("model response must contain a non-empty structured object")
                self._add_usage(trajectory, response)
                self._record(trajectory, {"event": "model_response", "iteration": iteration, "status": "success", "latency_ms": elapsed, "cost": estimated_cost, "token_usage": response.token_usage or {}, "response_type": "structured", "content": response.content})
            except Exception as exc:
                elapsed = round((time.perf_counter() - call_started) * 1000, 3)
                error = {"type": type(exc).__name__, "message": str(exc)}
                self._record(trajectory, {"event": "model_response", "iteration": iteration, "status": "failed", "latency_ms": elapsed, "cost": estimated_cost, "token_usage": {}, "response_type": "error", "error": error})
                if (
                    isinstance(exc, ProviderError)
                    and exc.retryable
                    and trajectory["recovery_attempts"] < self.limits.max_recovery_attempts
                ):
                    trajectory["recovery_attempts"] += 1
                    messages.append({
                        "role": "user",
                        "content": {
                            "type": "provider_retry",
                            "error_type": exc.kind,
                            "instruction": "The previous provider response was empty or transient. Continue the task; use a registered tool if work remains, otherwise return a final response.",
                        },
                    })
                    self._record(trajectory, {"event": "recovery", "decision": "retry_provider_response", "error_type": exc.kind})
                    continue
                trajectory["status"] = "failed_model"
                break

            content = response.content
            normalized = self._normalize_response(content)
            if normalized["kind"] == "error":
                trajectory["status"] = "failed_model_response"
                self._record(trajectory, {"event": "agent_stop", "reason": "model_error", "error": normalized.get("error")})
                break
            if normalized["kind"] == "final":
                final_output = normalized["output"]
                self._record(trajectory, {"event": "final_candidate", "output": final_output})
                if verifier is None:
                    trajectory["verification_result"] = None
                    trajectory["status"] = "success"
                    break
                evaluation = verifier(trajectory)
                trajectory["verification_result"] = evaluation.verification_passed
                self._record(trajectory, {"event": "verification", "passed": evaluation.verification_passed, "success": evaluation.success, "evidence": evaluation.evidence})
                if evaluation.success:
                    trajectory["status"] = "success"
                    break
                if not evaluation.recoverable_failure or trajectory["recovery_attempts"] >= self.limits.max_recovery_attempts:
                    trajectory["status"] = "failed_verification"
                    break
                trajectory["recovery_attempts"] += 1
                recovery_message = {"role": "user", "content": {"type": "verification_failure", "evidence": evaluation.evidence, "instruction": "The independent verifier rejected the candidate. Inspect the current state, change strategy, and continue using registered tools."}}
                messages.append(recovery_message)
                self._record(trajectory, {"event": "recovery", "decision": "replan_after_verification_failure", "evidence": evaluation.evidence})
                continue

            if trajectory["tool_calls"] >= self.limits.max_tool_calls:
                trajectory["status"] = "failed_tool_call_limit"
                break
            action = normalized["action"]
            tool_name = action["tool"]
            arguments = action["arguments"]
            tool_call_id = action.get("tool_call_id") or f"divy-call-{trajectory['tool_calls'] + 1}"
            action = {**action, "tool_call_id": tool_call_id}
            trajectory["tool_calls"] += 1
            self._record(trajectory, {"event": "tool_call", "tool": tool_name, "arguments": arguments, "tool_call_id": tool_call_id, "iteration": iteration})
            result = self.registry.execute(tool_name, arguments)
            observation = {
                "tool": tool_name,
                "success": result.ok,
                "result": result.data,
                "error": result.error,
                "metadata": {"schema_valid": result.schema_valid, "safety_violation": result.safety_violation},
            }
            self._record(trajectory, {"event": "tool_observation", **observation})
            messages.append({"role": "assistant", "content": {"action": action}})
            messages.append({"role": "tool", "name": tool_name, "tool_call_id": tool_call_id, "content": observation})
            if not result.ok:
                trajectory["recovery_attempts"] += 1
                if trajectory["recovery_attempts"] > self.limits.max_recovery_attempts:
                    trajectory["status"] = "failed_recovery_limit"
                    break

        if trajectory["status"] == "running":
            trajectory["status"] = "failed_iteration_limit"
        responses = [event for event in trajectory["events"] if event.get("event") == "model_response"]
        trajectory["token_usage_complete"] = all(bool(event.get("token_usage")) for event in responses)
        if not trajectory["token_usage_complete"]:
            trajectory["observed_token_usage"] = dict(trajectory["token_usage"])
            trajectory["token_usage"] = {}
        trajectory["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        self._record(trajectory, {"event": "complete", "status": trajectory["status"]})
        self._record_performance(trajectory, task_type, capabilities)
        return AgentSessionResult(trajectory["status"], final_output, trajectory)

    @staticmethod
    def _normalize_response(content: dict[str, Any]) -> dict[str, Any]:
        if "error" in content:
            return {"kind": "error", "error": content.get("error")}
        if isinstance(content.get("tool_calls"), list) and len(content["tool_calls"]) > 1:
            # This session currently executes one action per turn. Never drop
            # the remainder of a provider batch or replay a different history.
            return {"kind": "error", "error": {"type": "unsupported_tool_batch"}}
        if isinstance(content.get("action"), dict):
            action = content["action"]
            if isinstance(action.get("tool"), str) and isinstance(action.get("arguments"), dict):
                return {"kind": "tool_call", "action": action}
            return {"kind": "error", "error": {"type": "invalid_tool_call", "message": "action requires tool and object arguments"}}
        tool_calls = content.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            action = tool_calls[0]
            if isinstance(action, dict) and isinstance(action.get("tool"), str) and isinstance(action.get("arguments"), dict):
                return {"kind": "tool_call", "action": action}
            return {"kind": "error", "error": {"type": "invalid_tool_call", "message": "tool_calls contained an invalid action"}}
        if "final" in content and isinstance(content["final"], dict):
            return {"kind": "final", "output": content["final"]}
        return {"kind": "final", "output": content}

    def _add_usage(self, trajectory: dict[str, Any], response: ModelResponse) -> None:
        # Derive totals per response, not once from the accumulating history.
        current = dict(response.token_usage or {})
        if "total_tokens" not in current:
            parts = [current[key] for key in ("input_tokens", "output_tokens") if key in current]
            if not parts:
                parts = [current[key] for key in ("prompt_tokens", "completion_tokens") if key in current]
            if parts:
                current["total_tokens"] = sum(parts)
        for key, value in current.items():
            trajectory["token_usage"][key] = trajectory["token_usage"].get(key, 0) + value

    def _record(self, trajectory: dict[str, Any], event: dict[str, Any]) -> None:
        trajectory["events"].append(event)
        if self.logger is not None:
            self.logger.append(event)

    def _record_performance(self, trajectory: dict[str, Any], task_type: str, capabilities: set[str]) -> None:
        if self.performance_memory is None:
            return

        # Build a list of (model_event_index, tool_calls_since_this_model) pairs
        # by scanning events in order
        events = trajectory["events"]
        model_event_indices = [i for i, e in enumerate(events) if e.get("event") == "model_response"]

        # For each model response, count tool_calls that occur after it and before the next model_response
        tool_calls_per_model = []
        for idx, model_idx in enumerate(model_event_indices):
            start = model_idx + 1
            end = model_event_indices[idx + 1] if idx + 1 < len(model_event_indices) else len(events)
            tool_count = sum(1 for i in range(start, end) if events[i].get("event") == "tool_call")
            tool_calls_per_model.append(tool_count)

        model_events = [event for event in trajectory["events"] if event.get("event") == "model_response"]
        successful_events = [event for event in model_events if event.get("status") == "success"]
        final_event = successful_events[-1] if successful_events else None

        for i, event in enumerate(model_events):
            usage = event.get("token_usage") or {}
            token_count = usage.get("total_tokens")
            if token_count is None:
                parts = [value for key, value in usage.items() if key in {"input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"}]
                token_count = sum(parts) if parts else None
            contributed = trajectory["status"] == "success" and event is final_event
            self.performance_memory.record(
                PerformanceObservation(
                    execution_id=trajectory["execution_id"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_id=self.provider.model_id,
                    provider=self.provider.metadata.provider,
                    task_id=trajectory["task_id"],
                    task_type=task_type,
                    capabilities=tuple(sorted(capabilities)),
                    provider_success=event.get("status") == "success",
                    contributed_to_final=contributed,
                    task_success=trajectory["status"] == "success",
                    verification_passed=trajectory["verification_result"],
                    latency_ms=event.get("latency_ms"),
                    cost=self.provider.cost(),
                    token_count=token_count,
                    tool_calls=tool_calls_per_model[i] if i < len(tool_calls_per_model) else 0,
                    recovery_attempts=trajectory["recovery_attempts"],
                    failure_type=(event.get("error") or {}).get("type") or (None if event.get("status") == "success" else trajectory["status"]),
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    final_outcome=trajectory["status"],
                )
            )
