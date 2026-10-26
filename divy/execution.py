"""Provider execution, fallback, verification, and synthesis.

This layer is deliberately independent from the existing tool orchestrator.
It consumes the same provider-neutral ``ModelAdapter`` response contract and
returns structured operational evidence that can be benchmarked.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .logger import EventLogger
from .adapters import ProviderError
from .performance import PerformanceMemory, PerformanceObservation
from .providers import ModelProvider, ModelRegistry
from .routing import RoutingDecision
from .schemas import EvaluationResult, ModelResponse


@dataclass
class ModelCallRecord:
    model_id: str
    role: str
    status: str
    latency_ms: float
    cost: float | None
    token_usage: dict[str, int] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


@dataclass
class OrchestrationResult:
    success: bool
    final_output: dict[str, Any] | None
    calls: list[ModelCallRecord]
    fallback_used: bool = False
    verification_attempted: bool = False
    verification_passed: bool = False
    disagreement: bool = False
    rationale: str = ""
    wall_latency_ms: float = 0.0
    accepted_model_id: str | None = None

    @property
    def model_calls(self) -> int:
        return sum(call.status != "skipped" for call in self.calls)

    @property
    def total_cost(self) -> float | None:
        if any(call.cost is None for call in self.calls):
            return None
        return round(sum(call.cost or 0.0 for call in self.calls), 8)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(call.latency_ms for call in self.calls), 3)

    @property
    def token_usage(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for call in self.calls:
            for key, value in call.token_usage.items():
                totals[key] = totals.get(key, 0) + value
        return totals


class MultiModelExecutor:
    def __init__(self, registry: ModelRegistry, logger: EventLogger | None = None, performance_memory: PerformanceMemory | None = None) -> None:
        self.registry = registry
        self.logger = logger
        self.performance_memory = performance_memory

    def execute(
        self,
        decision: RoutingDecision,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_schema: dict[str, Any],
        *,
        parallel: bool = False,
        verify: bool = False,
        synthesize: bool = False,
        task_type: str | None = None,
        required_capabilities: set[str] | None = None,
        tool_calls: int = 0,
        recovery_attempts: int = 0,
        max_cost: float | None = None,
        result_verifier: Callable[[dict[str, Any]], EvaluationResult] | None = None,
    ) -> OrchestrationResult:
        calls: list[ModelCallRecord] = []
        started = time.perf_counter()
        execution_id = str(uuid.uuid4())
        successful: list[ModelCallRecord] = []
        selected = list(decision.selected_model_ids)
        spent_cost = 0.0
        if parallel and len(selected) > 1:
            eligible: list[str] = []
            for model_id in selected:
                estimated = self.registry.get(model_id).cost()
                if max_cost is None or (estimated is not None and spent_cost + estimated <= max_cost):
                    eligible.append(model_id)
                    if estimated is not None:
                        spent_cost += estimated
            selected = eligible
            successful, calls = self._call_parallel(selected, messages, tools, response_schema) if selected else ([], [])
        else:
            for model_id in selected:
                estimated = self.registry.get(model_id).cost()
                if max_cost is not None and (estimated is None or spent_cost + estimated > max_cost):
                    calls.append(self._skipped_for_cost(model_id, "primary"))
                    continue
                record = self._call(model_id, "primary", messages, tools, response_schema)
                calls.append(record)
                if estimated is not None:
                    spent_cost += estimated
                if record.status == "success":
                    successful.append(record)
                    if not parallel:
                        break

        fallback_used = False
        if not successful:
            for model_id in decision.fallback_model_ids:
                estimated = self.registry.get(model_id).cost()
                if max_cost is not None and (estimated is None or spent_cost + estimated > max_cost):
                    calls.append(self._skipped_for_cost(model_id, "fallback"))
                    continue
                record = self._call(model_id, "fallback", messages, tools, response_schema)
                calls.append(record)
                if estimated is not None:
                    spent_cost += estimated
                if record.status == "success":
                    successful.append(record)
                    fallback_used = True
                    break

        disagreement = len({self._stable_output(record.output) for record in successful}) > 1
        final_output = successful[0].output if successful else None

        synthesis: ModelCallRecord | None = None
        if synthesize and successful and decision.synthesis_model_id:
            candidates = [record.output for record in successful if record.output is not None]
            synthesis_messages = messages + [{"role": "user", "content": {"candidates": candidates, "instruction": "Synthesize the best supported result."}}]
            estimate = self.registry.get(decision.synthesis_model_id).cost()
            if max_cost is not None and (estimate is None or spent_cost + estimate > max_cost):
                synthesis = self._skipped_for_cost(decision.synthesis_model_id, "synthesis")
            else:
                synthesis = self._call(decision.synthesis_model_id, "synthesis", synthesis_messages, tools, response_schema)
                spent_cost += estimate or 0.0
            calls.append(synthesis)
            if synthesis.status == "success":
                final_output = synthesis.output

        verification_attempted = False
        verification_passed = False
        verification_candidates = ([synthesis] if synthesis is not None and synthesis.status == "success" else successful) if successful else []
        accepted_model_id: str | None = verification_candidates[0].model_id if verification_candidates else None

        def verify_candidate(candidate: ModelCallRecord) -> bool:
            nonlocal verification_attempted, verification_passed, spent_cost
            if not verify:
                return True
            if result_verifier is not None:
                verification_attempted = True
                try:
                    verification_passed = bool(result_verifier(candidate.output or {}).success)
                except Exception:
                    verification_passed = False
                return verification_passed
            if not decision.verifier_model_id:
                verification_passed = False
                return False
            verification_messages = messages + [{"role": "user", "content": {"candidate": candidate.output, "instruction": "Independently verify this candidate."}}]
            estimate = self.registry.get(decision.verifier_model_id).cost()
            if max_cost is not None and (estimate is None or spent_cost + estimate > max_cost):
                verification = self._skipped_for_cost(decision.verifier_model_id, "verifier")
            else:
                verification_attempted = True
                verification = self._call(decision.verifier_model_id, "verifier", verification_messages, tools, {"type": "object", "properties": {"verified": {"type": "boolean"}}})
                spent_cost += estimate or 0.0
            calls.append(verification)
            verification_passed = verification.status == "success" and (verification.output or {}).get("verified") is True
            return verification_passed

        if verification_candidates and verify and (decision.verifier_model_id or result_verifier is not None):
            accepted = next((candidate for candidate in verification_candidates if verify_candidate(candidate)), None)
            if accepted is not None:
                final_output = accepted.output
                accepted_model_id = accepted.model_id
            else:
                # A provider-successful but unverified result is not accepted.
                # Try bounded fallbacks and independently verify each candidate.
                attempted_ids = {candidate.model_id for candidate in verification_candidates}
                for model_id in decision.fallback_model_ids:
                    if model_id in attempted_ids:
                        continue
                    estimated = self.registry.get(model_id).cost()
                    if max_cost is not None and (estimated is None or spent_cost + estimated > max_cost):
                        calls.append(self._skipped_for_cost(model_id, "fallback"))
                        continue
                    fallback_record = self._call(model_id, "fallback", messages, tools, response_schema)
                    calls.append(fallback_record)
                    if estimated is not None:
                        spent_cost += estimated
                    if fallback_record.status == "success":
                        fallback_used = True
                        if verify_candidate(fallback_record):
                            final_output = fallback_record.output
                            accepted_model_id = fallback_record.model_id
                            break

        success = bool(final_output is not None and (not verify or verification_passed))
        if self.performance_memory is not None:
            self._record_performance(
                calls,
                final_output,
                success,
                task_type or decision.task_id,
                decision.task_id,
                required_capabilities or set(),
                verification_passed,
                verification_attempted,
                result_verifier is not None,
                tool_calls,
                recovery_attempts,
                execution_id,
                accepted_model_id,
            )
        self._log({"event": "orchestration_complete", "task_id": decision.task_id, "success": success, "fallback_used": fallback_used, "verification_passed": verification_passed, "disagreement": disagreement})
        return OrchestrationResult(
            success=success,
            final_output=final_output,
            calls=calls,
            fallback_used=fallback_used,
            verification_attempted=verification_attempted,
            verification_passed=verification_passed,
            disagreement=disagreement,
            rationale=decision.rationale,
            wall_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            accepted_model_id=accepted_model_id if success else None,
        )

    def _record_performance(
        self,
        calls: list[ModelCallRecord],
        final_output: dict[str, Any] | None,
        task_success: bool,
        task_type: str,
        task_id: str,
        capabilities: set[str],
        verification_passed: bool,
        verification_attempted: bool,
        local_verifier: bool,
        tool_calls: int,
        recovery_attempts: int,
        execution_id: str,
        accepted_model_id: str | None,
    ) -> None:
        assert self.performance_memory is not None
        timestamp = datetime.now(timezone.utc).isoformat()
        attempt_index = 0
        for call in calls:
            if call.status == "skipped":
                continue
            attempt_index += 1
            token_count = call.token_usage.get("total_tokens")
            if token_count is None and call.token_usage:
                token_count = sum(value for key, value in call.token_usage.items() if key in {"input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"})
            provider = self.registry.get(call.model_id)
            contributed = task_success and call.status == "success" and call.output is final_output
            self.performance_memory.record(
                PerformanceObservation(
                    execution_id=execution_id,
                    timestamp=timestamp,
                    model_id=call.model_id,
                    provider=provider.metadata.provider,
                    task_id=task_id,
                    task_type=task_type,
                    capabilities=tuple(sorted(capabilities)),
                    provider_success=call.status == "success",
                    contributed_to_final=contributed,
                    task_success=task_success and contributed,
                    verification_passed=(
                        (verification_passed if call.model_id == accepted_model_id else False)
                        if verification_attempted and local_verifier and call.role in {"primary", "fallback"}
                        else (verification_passed if verification_attempted and call.role == "verifier" else None)
                    ),
                    latency_ms=call.latency_ms,
                    cost=call.cost,
                    token_count=token_count,
                    input_tokens=call.token_usage.get("input_tokens"),
                    output_tokens=call.token_usage.get("output_tokens"),
                    tool_calls=tool_calls,
                    recovery_attempts=recovery_attempts,
                    failure_type=(call.error or {}).get("type") or (
                        "verification_failure"
                        if verification_attempted and call.status == "success" and call.role in {"primary", "fallback"} and not contributed
                        else None
                    ),
                    final_outcome="success" if contributed else (
                        "provider_failure" if call.status != "success" else (
                            "verification_failure" if verification_attempted and call.role in {"primary", "fallback"} else "not_selected"
                        )
                    ),
                    accepted_model_id=accepted_model_id,
                    fallback_used=call.role == "fallback",
                    attempt_index=attempt_index,
                    role=call.role,
                )
            )

    @staticmethod
    def _skipped_for_cost(model_id: str, role: str) -> ModelCallRecord:
        return ModelCallRecord(
            model_id=model_id,
            role=role,
            status="skipped",
            latency_ms=0.0,
            cost=0.0,
            error={"type": "cost_limit", "message": "model call skipped because the configured cost limit would be exceeded"},
        )

    def _call_parallel(self, model_ids: list[str], messages: list[dict[str, Any]], tools: list[dict[str, Any]], response_schema: dict[str, Any]) -> tuple[list[ModelCallRecord], list[ModelCallRecord]]:
        calls: list[ModelCallRecord] = []
        successful: list[ModelCallRecord] = []
        with ThreadPoolExecutor(max_workers=min(2, len(model_ids))) as pool:
            futures = {pool.submit(self._call, model_id, "primary", messages, tools, response_schema): model_id for model_id in model_ids}
            for future in as_completed(futures):
                record = future.result()
                calls.append(record)
                if record.status == "success":
                    successful.append(record)
        calls.sort(key=lambda record: model_ids.index(record.model_id))
        successful.sort(key=lambda record: model_ids.index(record.model_id))
        return successful, calls

    def _call(self, model_id: str, role: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], response_schema: dict[str, Any]) -> ModelCallRecord:
        provider = self.registry.get(model_id)
        started = time.perf_counter()
        self._log({"event": "model_call", "model_id": model_id, "role": role})
        try:
            response: ModelResponse = provider.generate(messages, tools, response_schema)
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            if not isinstance(response.content, dict) or not response.content:
                raise ValueError("invalid structured model output")
            record = ModelCallRecord(model_id, role, "success", elapsed, provider.cost(), response.token_usage or {}, response.content)
        except Exception as exc:  # provider failures are data for routing/recovery
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            failure_type = exc.kind if isinstance(exc, ProviderError) else ("invalid_response" if isinstance(exc, ValueError) else "provider_error")
            record = ModelCallRecord(model_id, role, "failed", elapsed, provider.cost(), error={"type": failure_type, "message": "provider call failed"})
        self._log({"event": "model_result", "model_id": model_id, "role": role, "status": record.status, "latency_ms": record.latency_ms, "error": record.error})
        return record

    @staticmethod
    def _stable_output(output: dict[str, Any] | None) -> str:
        import json

        return json.dumps(output, sort_keys=True)

    def _log(self, event: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger.append(event)
