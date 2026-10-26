"""Registered test runner; it never accepts arbitrary shell commands."""

from __future__ import annotations

import hashlib
import types
from collections.abc import Callable
from pathlib import Path

from ..schemas import ToolResult
from .base import ToolDefinition
from .workspace import SandboxWorkspace


class TestRunnerTool:
    def __init__(self, workspace: SandboxWorkspace, tests: dict[str, Callable[[SandboxWorkspace], dict]]) -> None:
        self.workspace = workspace
        self.tests = tests

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                "test.run",
                f"Run one pre-registered deterministic fixture test. Registered test IDs: {', '.join(sorted(self.tests))}.",
                {"type": "object", "properties": {"test_id": {"type": "string"}}, "required": ["test_id"]},
                self._run,
                # These handlers can compile model-edited fixture code. Path
                # confinement alone is NOT a Python execution sandbox.
                requires_process_isolation=True,
            )
        ]

    def _run(self, args: dict[str, str]) -> ToolResult:
        test_id = args.get("test_id")
        if not isinstance(test_id, str):
            return ToolResult(ok=False, error={"type": "invalid_arguments", "message": "test_id is required"}, schema_valid=False)
        test = self.tests.get(test_id)
        if test is None:
            return ToolResult(ok=False, error={"type": "unknown_test", "message": "test_id is not registered"})
        result = test(self.workspace)
        return ToolResult(ok=bool(result.get("passed")), data=result)


def load_python_module(workspace: SandboxWorkspace, relative_path: str):
    path: Path = workspace.resolve(relative_path)
    source = path.read_text(encoding="utf-8")
    module = types.ModuleType(f"divy_fixture_{hashlib.sha256(source.encode('utf-8')).hexdigest()[:12]}")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module
