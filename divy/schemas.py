"""Core schemas and limits for DIVY V0.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    SELECT_ACTION = "SELECT_ACTION"
    EXECUTE = "EXECUTE"
    OBSERVE = "OBSERVE"
    VERIFY = "VERIFY"
    DIAGNOSE = "DIAGNOSE"
    REPLAN = "REPLAN"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class Limits:
    max_subtasks: int = 8
    max_tool_calls: int = 12
    max_recovery_attempts: int = 2
    max_clarifications: int = 1
    max_parallel_branches: int = 2


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    goal: str
    fixture_name: str
    success_criteria: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: dict[str, Any] | None = None
    schema_valid: bool = True
    safety_violation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "schema_valid": self.schema_valid,
            "safety_violation": self.safety_violation,
        }


@dataclass
class EvaluationResult:
    success: bool
    verification_passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    unnecessary_actions: int = 0
    safety_violations: int = 0
    recoverable_failure: bool = False


@dataclass
class ModelResponse:
    content: dict[str, Any]
    token_usage: dict[str, int] | None = None
