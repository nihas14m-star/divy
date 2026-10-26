"""Behavioral tests for the live task fixtures and verifiers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from divy.live_tasks import build_registry, setup_fixture, verify_task
from divy.tools.workspace import SandboxWorkspace


class LiveTaskVerifierTests(unittest.TestCase):
    def test_rw03_spec_matches_registered_test(self) -> None:
        """RW03 must ask for plain functions, not an HTTP framework."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = SandboxWorkspace(Path(tmpdir))
            setup_fixture("RW03", workspace)
            spec = workspace.read_text("spec.md")
            self.assertIn("plain Python functions", spec)
            self.assertIn("Do not use a web", spec)
            self.assertIn("list_users()", spec)
            self.assertIn("create_user(data", spec)
            self.assertIn("get_user(id)", spec)

    def test_rw03_stub_does_not_pass_behavioral_test(self) -> None:
        """The untouched RW03 stub must fail the registered behavioral test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = SandboxWorkspace(Path(tmpdir))
            setup_fixture("RW03", workspace)
            registry = build_registry(workspace, "RW03")

            result = registry.execute("test.run", {"test_id": "api_impl"})

            self.assertFalse(result.ok)
            self.assertFalse(result.data["passed"])
            self.assertFalse(verify_task("RW03", workspace, {"events": []}))

    def test_rw03_complete_implementation_passes_behavioral_test(self) -> None:
        """A real implementation of list/create/get must pass the RW03 test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = SandboxWorkspace(Path(tmpdir))
            setup_fixture("RW03", workspace)
            workspace.write_text(
                "impl.py",
                "def list_users():\n"
                "    return [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]\n\n"
                "def create_user(data):\n"
                "    return {'id': 3, **data}\n\n"
                "def get_user(id):\n"
                "    for user in list_users():\n"
                "        if user['id'] == id:\n"
                "            return user\n"
                "    return None\n",
            )
            registry = build_registry(workspace, "RW03")

            result = registry.execute("test.run", {"test_id": "api_impl"})

            self.assertTrue(result.ok)
            self.assertTrue(result.data["passed"])
            trajectory = {
                "events": [
                    {
                        "event": "tool_observation",
                        "tool": "test.run",
                        "success": True,
                        "result": {"passed": True},
                    }
                ]
            }
            self.assertTrue(verify_task("RW03", workspace, trajectory))

    def test_rw03_empty_store_implementation_passes(self) -> None:
        """A valid in-memory API may start empty and persist created users."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = SandboxWorkspace(Path(tmpdir))
            setup_fixture("RW03", workspace)
            workspace.write_text(
                "impl.py",
                "_users = []\n"
                "_next_id = 1\n\n"
                "def list_users():\n"
                "    return list(_users)\n\n"
                "def create_user(data):\n"
                "    global _next_id\n"
                "    user = {'id': _next_id, 'name': data.get('name', '')}\n"
                "    _next_id += 1\n"
                "    _users.append(user)\n"
                "    return user\n\n"
                "def get_user(id):\n"
                "    for user in _users:\n"
                "        if user['id'] == id:\n"
                "            return user\n"
                "    return None\n",
            )
            registry = build_registry(workspace, "RW03")

            result = registry.execute("test.run", {"test_id": "api_impl"})

            self.assertTrue(result.ok)
            self.assertTrue(result.data["passed"])


if __name__ == "__main__":
    unittest.main()
