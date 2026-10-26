"""Read-only and bounded-write filesystem tools."""

from __future__ import annotations

import json
import hashlib

from ..schemas import ToolResult
from .base import ToolDefinition
from .workspace import SandboxWorkspace


class FilesystemTools:
    def __init__(self, workspace: SandboxWorkspace) -> None:
        self.workspace = workspace

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                "fs.list",
                "List files in the sandbox workspace.",
                {"type": "object", "properties": {}, "required": []},
                lambda args: ToolResult(ok=True, data={"files": self.workspace.list_files()}),
            ),
            ToolDefinition(
                "fs.read",
                "Read a UTF-8 text file and parse JSON when applicable.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                self._read,
            ),
            ToolDefinition(
                "fs.search",
                "Search text files using a regular expression.",
                {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
                lambda args: ToolResult(ok=True, data={"matches": self.workspace.search(args["pattern"])}),
            ),
            ToolDefinition(
                "fs.write",
                "Write a text file inside the sandbox; deletion is not supported.",
                {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "expected_sha256": {"type": "string", "description": "Optional optimistic-concurrency guard: the SHA-256 of this file's current content, as returned by fs.read. Omit this when creating a new file or overwriting without a freshness check; it is not a hash of the desired content."}}, "required": ["path", "content"]},
                self._write,
            ),
            ToolDefinition(
                "fs.patch",
                "Replace one exact text fragment in a file.",
                {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]},
                self._patch,
            ),
        ]

    def _read(self, args: dict[str, str]) -> ToolResult:
        path = args["path"]
        content = self.workspace.read_text(path)
        data: dict[str, object] = {"path": path, "content": content, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
        if path.lower().endswith(".json"):
            data["json"] = json.loads(content)
        return ToolResult(ok=True, data=data)

    def _write(self, args: dict[str, str]) -> ToolResult:
        current = self.workspace.read_text(args["path"]) if self.workspace.resolve(args["path"]).exists() else ""
        expected = args.get("expected_sha256")
        if expected is not None and hashlib.sha256(current.encode("utf-8")).hexdigest() != expected:
            return ToolResult(ok=False, error={"type": "stale_state", "message": "file changed since it was observed"})
        self.workspace.write_text(args["path"], args["content"])
        return ToolResult(ok=True, data={"path": args["path"], "bytes_written": len(args["content"].encode("utf-8"))})

    def _patch(self, args: dict[str, str]) -> ToolResult:
        content = self.workspace.read_text(args["path"])
        if args["old"] not in content:
            return ToolResult(ok=False, error={"type": "pattern_not_found", "message": "old text was not found"})
        if content.count(args["old"]) != 1:
            return ToolResult(ok=False, error={"type": "ambiguous_patch", "message": "old text must occur exactly once"})
        self.workspace.write_text(args["path"], content.replace(args["old"], args["new"]))
        return ToolResult(ok=True, data={"path": args["path"], "replacements": 1})
