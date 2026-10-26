"""Independent verification wrapper."""

from __future__ import annotations

from typing import Any

from .evaluators import evaluate
from .schemas import EvaluationResult, TaskDefinition
from .tools.workspace import SandboxWorkspace


class DeterministicVerifier:
    def verify(self, task: TaskDefinition, workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
        return evaluate(task, workspace, trajectory)
