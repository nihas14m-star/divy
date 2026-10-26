"""Tool registry: the only path from model actions to side effects."""

from __future__ import annotations

from .base import ToolDefinition, safe_handler_call, validate_arguments
from ..schemas import ToolResult


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._tools = {definition.name: definition for definition in definitions}

    def schemas(self) -> list[dict]:
        return [tool.public_schema() for tool in self._tools.values()]

    def names(self) -> set[str]:
        return set(self._tools)

    def execute(self, name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error={"type": "unknown_tool", "message": name}, schema_valid=False, safety_violation=True)
        valid, message = validate_arguments(tool.parameters, arguments)
        if not valid:
            return ToolResult(ok=False, error={"type": "invalid_arguments", "message": message}, schema_valid=False)
        return safe_handler_call(tool.handler, arguments)
