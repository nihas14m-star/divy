"""Small non-network calculator tool with no eval or shell access."""

from __future__ import annotations

import ast
import operator

from ..schemas import ToolResult
from .base import ToolDefinition


class CalculatorTool:
    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                "calc.evaluate",
                "Evaluate a basic arithmetic expression.",
                {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                self._evaluate,
            )
        ]

    def _evaluate(self, args: dict[str, str]) -> ToolResult:
        try:
            value = self._node(ast.parse(args["expression"], mode="eval").body)
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
            return ToolResult(ok=False, error={"type": "invalid_expression", "message": str(exc)})
        return ToolResult(ok=True, data={"expression": args["expression"], "value": value})

    def _node(self, node: ast.AST) -> int | float:
        binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](self._node(node.left), self._node(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("only numeric arithmetic is allowed")
