"""Tool protocol and small JSON-schema subset used by the harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..schemas import ToolResult
from .workspace import SandboxWorkspace, SandboxViolation


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]
    requires_process_isolation: bool = False

    def public_schema(self) -> dict[str, Any]:
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.requires_process_isolation:
            schema["x-divy-requires-process-isolation"] = True
        return schema


def validate_arguments(schema: dict[str, Any], arguments: Any) -> tuple[bool, str | None]:
    if not isinstance(arguments, dict):
        return False, "arguments must be an object"
    required = schema.get("required", [])
    for name in required:
        if name not in arguments:
            return False, f"missing required argument: {name}"
    properties = schema.get("properties", {})
    for name, value in arguments.items():
        if name not in properties:
            return False, f"unknown argument: {name}"
        expected = properties[name].get("type")
        valid = {
            "string": isinstance(value, str),
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }.get(expected, True)
        if not valid:
            return False, f"argument {name} must be {expected}"
    return True, None


def safe_handler_call(handler: Callable[[dict[str, Any]], ToolResult], arguments: dict[str, Any]) -> ToolResult:
    try:
        return handler(arguments)
    except SandboxViolation as exc:
        return ToolResult(ok=False, error={"type": "sandbox_violation", "message": "sandbox policy rejected the request"}, safety_violation=True)
    except FileNotFoundError:
        return ToolResult(ok=False, error={"type": "not_found", "message": "requested file was not found"})
    except PermissionError:
        return ToolResult(ok=False, error={"type": "permission_denied", "message": "permission denied"})
    except OSError:
        return ToolResult(ok=False, error={"type": "tool_error", "message": "tool I/O failed"})
    except (ValueError, TypeError):
        return ToolResult(ok=False, error={"type": "tool_error", "message": "tool received invalid data"})
    except Exception:
        # Tool implementations may execute fixture code. Never place an
        # arbitrary exception string (which may contain file contents,
        # environment values, or credentials) into a trajectory.
        return ToolResult(ok=False, error={"type": "tool_error", "message": "tool execution failed"})
