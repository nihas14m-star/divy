from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from divy.realworld import run_realworld_benchmark
from divy.live_tasks import build_registry
from divy.tools.workspace import SandboxWorkspace


class RealWorldTaskTests(unittest.TestCase):
    def test_software_engineering_task_completes_with_independent_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_realworld_benchmark(Path(directory))
        self.assertTrue(result["success"], result)
        self.assertTrue(result["plan_valid"])
        self.assertTrue(result["verification_result"])
        self.assertEqual(result["tool_calls"], 4)

    def test_rw01_registry_includes_test_run_tool(self) -> None:
        """Regression test: ensure test.run tool is registered for RW01 task."""
        with tempfile.TemporaryDirectory() as directory:
            ws = SandboxWorkspace(Path(directory))
            registry = build_registry(ws, "RW01")
        self.assertIn("test.run", registry._tools, "test.run tool must be registered for RW01")


if __name__ == "__main__":
    unittest.main()
