"""Bounded DIVY orchestration loop."""

from __future__ import annotations

import time
from typing import Any, Callable

from .logger import EventLogger
from .model import ModelAdapter
from .schemas import EvaluationResult, Limits, Phase, TaskDefinition
from .tools.registry import ToolRegistry
from .tools.workspace import SandboxWorkspace


class Orchestrator:
    def __init__(
        self,
        model: ModelAdapter,
        registry: ToolRegistry,
        verifier: Any,
        logger: EventLogger,
        limits: Limits | None = None,
        observation_hook: Callable[[TaskDefinition, SandboxWorkspace, dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.verifier = verifier
        self.logger = logger
        self.limits = limits or Limits()
        self.observation_hook = observation_hook

    def run(self, task: TaskDefinition, workspace: SandboxWorkspace) -> dict[str, Any]:
        started = time.perf_counter()
        trajectory: dict[str, Any] = {
            "task_id": task.task_id,
            "goal": task.goal,
            "events": [],
            "tool_calls": 0,
            "recovery_attempts": 0,
            "clarifications": 0,
            "safety_violations": 0,
            "plan_valid": False,
            "tool_selection_valid": True,
            "argument_valid": True,
            "verification_result": False,
            "unnecessary_actions": 0,
            "status": "running",
            "token_usage": {},
        }

        understanding = self._ask(Phase.UNDERSTAND, task, trajectory)
        if understanding.get("needs_clarification"):
            trajectory["clarifications"] += 1
            trajectory["plan_valid"] = True
            trajectory["status"] = "needs_clarification"
            verification = self.verifier.verify(task, workspace, trajectory)
            trajectory["verification_result"] = verification.verification_passed
            return self._finish(trajectory, started)

        plan_response = self._ask(Phase.PLAN, task, trajectory)
        plan = plan_response.get("subtasks", [])
        trajectory["plan_valid"] = self._validate_plan(plan, self.limits)
        self._record(trajectory, {"event": "plan", "plan": plan, "valid": trajectory["plan_valid"]})
        if not trajectory["plan_valid"]:
            trajectory["status"] = "failed_invalid_plan"
            return self._finish(trajectory, started)

        while trajectory["tool_calls"] < self.limits.max_tool_calls:
            state = self._state(trajectory, Phase.SELECT_ACTION)
            response = self.model.generate([{"role": "system", "content": "Use only registered tools."}, {"role": "user", "content": state}], self.registry.schemas(), {})
            self._add_tokens(trajectory, response.token_usage)
            action = response.content.get("action")
            if not action:
                break
            if not isinstance(action, dict):
                tool_name = "<malformed>"
                arguments = {}
                trajectory["tool_selection_valid"] = False
            else:
                tool_name = action.get("tool")
                arguments = action.get("arguments", {})
                if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                    trajectory["tool_selection_valid"] = False
                if not isinstance(tool_name, str):
                    tool_name = "<malformed>"
            trajectory["tool_calls"] += 1
            self._record(trajectory, {"event": "tool_call", "tool": tool_name, "arguments": arguments})
            if tool_name not in self.registry.names() or tool_name not in task.allowed_tools:
                result = self._policy_rejection(tool_name)
            else:
                result = self.registry.execute(tool_name, arguments)
            if not result.schema_valid:
                trajectory["argument_valid"] = False
            if result.safety_violation:
                trajectory["safety_violations"] += 1
            observation = {"event": "tool_result", "tool": tool_name, "path": arguments.get("path"), **result.as_dict()}
            self._record(trajectory, observation)
            if self.observation_hook is not None:
                hook_event = self.observation_hook(task, workspace, trajectory)
                if hook_event is not None:
                    self._record(trajectory, hook_event)

            if not result.ok:
                if not self._recover(trajectory, task, Phase.DIAGNOSE):
                    trajectory["status"] = "failed_recovery_limit"
                    break
                continue

            verification = self.verifier.verify(task, workspace, trajectory)
            trajectory["verification_result"] = verification.verification_passed
            trajectory["unnecessary_actions"] = verification.unnecessary_actions
            self._record(trajectory, {"event": "verification", "passed": verification.verification_passed, "evidence": verification.evidence})
            if verification.success:
                trajectory["status"] = "success"
                break
            if verification.recoverable_failure:
                if not self._recover(trajectory, task, Phase.DIAGNOSE):
                    trajectory["status"] = "failed_recovery_limit"
                    break
                continue

        if trajectory["status"] == "running":
            verification = self.verifier.verify(task, workspace, trajectory)
            trajectory["verification_result"] = verification.verification_passed
            if verification.success:
                trajectory["status"] = "success"
            else:
                trajectory["status"] = "failed_incomplete"
        return self._finish(trajectory, started)

    def _ask(self, phase: Phase, task: TaskDefinition, trajectory: dict[str, Any]) -> dict[str, Any]:
        response = self.model.generate(
            [{"role": "system", "content": "You are a bounded orchestrator. Never execute tools directly."}, {"role": "user", "content": self._state(trajectory, phase)}],
            self.registry.schemas(),
            {"type": "object"},
        )
        self._add_tokens(trajectory, response.token_usage)
        return response.content

    @staticmethod
    def _validate_plan(plan: Any, limits: Limits) -> bool:
        if not isinstance(plan, list) or not 1 <= len(plan) <= limits.max_subtasks:
            return False
        ids = {item.get("id") for item in plan if isinstance(item, dict)}
        if len(ids) != len(plan) or None in ids:
            return False
        parallel_groups: dict[str, int] = {}
        for item in plan:
            if not set(item.get("depends_on", [])) <= ids:
                return False
            group = item.get("parallel_group")
            if group is not None:
                parallel_groups[group] = parallel_groups.get(group, 0) + 1
        return all(count <= limits.max_parallel_branches for count in parallel_groups.values())

    @staticmethod
    def _state(trajectory: dict[str, Any], phase: Phase) -> dict[str, Any]:
        observations = [
            {"tool": event.get("tool"), "path": event.get("path"), "ok": event.get("ok"), "data": event.get("data"), "error": event.get("error")}
            for event in trajectory["events"]
            if event.get("event") == "tool_result"
        ]
        return {"phase": phase.value, "observations": observations, "tool_calls": trajectory["tool_calls"], "recovery_attempts": trajectory["recovery_attempts"]}

    def _record(self, trajectory: dict[str, Any], event: dict[str, Any]) -> None:
        trajectory["events"].append(event)
        self.logger.append(event)

    @staticmethod
    def _add_tokens(trajectory: dict[str, Any], usage: dict[str, int] | None) -> None:
        if not usage:
            return
        for key, value in usage.items():
            trajectory["token_usage"][key] = trajectory["token_usage"].get(key, 0) + value

    def _finish(self, trajectory: dict[str, Any], started: float) -> dict[str, Any]:
        trajectory["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        self.logger.append({"event": "complete", "status": trajectory["status"], "task_id": trajectory["task_id"]})
        return trajectory

    def _recover(self, trajectory: dict[str, Any], task: TaskDefinition, phase: Phase) -> bool:
        trajectory["recovery_attempts"] += 1
        if trajectory["recovery_attempts"] > self.limits.max_recovery_attempts:
            return False
        diagnosis = self._ask(phase, task, trajectory)
        self._record(trajectory, {"event": "diagnosis", "content": diagnosis})
        replan = self._ask(Phase.REPLAN, task, trajectory)
        self._record(trajectory, {"event": "recovery_decision", "decision": "replan_with_safe_action", "content": replan})
        self._record(trajectory, {"event": "replan", "content": replan})
        return True

    @staticmethod
    def _policy_rejection(tool_name: str):
        from .schemas import ToolResult

        return ToolResult(
            ok=False,
            error={"type": "policy_rejection", "message": f"tool is not registered or not allowed for this task: {tool_name}"},
            safety_violation=True,
        )
