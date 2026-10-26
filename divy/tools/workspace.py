"""Filesystem sandbox used by all local tools."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class SandboxViolation(Exception):
    pass


class SandboxWorkspace:
    """Restrict every path to one fixture directory and forbid risky files."""

    _blocked_names = {".env", "credentials", "secrets", "id_rsa", "token"}

    def __init__(self, root: Path, protected_paths: set[str] | None = None) -> None:
        self.root = root.resolve()
        self.protected_paths = {path.replace("\\", "/") for path in (protected_paths or set())}
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise SandboxViolation("path must be a non-empty relative string")
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise SandboxViolation("path escapes the fixture workspace") from exc
        if any(part.lower() in self._blocked_names for part in candidate.parts):
            raise SandboxViolation("credential-like paths are blocked")
        return candidate

    def read_text(self, relative_path: str) -> str:
        return self.resolve(relative_path).read_text(encoding="utf-8")

    def write_text(self, relative_path: str, content: str) -> None:
        path = self.resolve(relative_path)
        normalized = str(path.relative_to(self.root)).replace("\\", "/")
        if normalized in self.protected_paths:
            raise PermissionError(f"write is protected: {relative_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def protect(self, relative_path: str) -> None:
        self.protected_paths.add(relative_path.replace("\\", "/"))

    def list_files(self) -> list[str]:
        return sorted(str(path.relative_to(self.root)).replace("\\", "/") for path in self.root.rglob("*") if path.is_file())

    def search(self, pattern: str) -> list[dict[str, Any]]:
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            raise SandboxViolation(f"invalid search pattern: {exc}") from exc
        matches: list[dict[str, Any]] = []
        for relative in self.list_files():
            text = self.read_text(relative)
            for line_number, line in enumerate(text.splitlines(), start=1):
                if expression.search(line):
                    matches.append({"path": relative, "line": line_number, "text": line})
        return matches

    def read_json(self, relative_path: str) -> Any:
        return json.loads(self.read_text(relative_path))
