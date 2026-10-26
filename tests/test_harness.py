from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from divy.benchmark import run_benchmarks
from divy.tasks import all_tasks, create_fixture, registered_tests
from divy.tools import CalculatorTool, FilesystemTools, SandboxWorkspace, TestRunnerTool as _TestRunnerTool, ToolRegistry


class HarnessTests(unittest.TestCase):
    def test_b01_to_b20_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_benchmarks(Path(directory))
        self.assertEqual([item["task_id"] for item in summary], [f"B{i:02d}" for i in range(1, 21)])
        self.assertTrue(all(item["task_success"] for item in summary), summary)
        self.assertTrue(all(item["plan_valid"] for item in summary))
        self.assertTrue(all(item["argument_valid"] for item in summary))
        self.assertTrue(all(item["safety_violations"] == 0 for item in summary))

    def test_registry_rejects_unknown_and_invalid_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            registry = ToolRegistry(FilesystemTools(workspace).definitions())
            unknown = registry.execute("shell.exec", {"command": "echo unsafe"})
            invalid = registry.execute("fs.read", {})
        self.assertFalse(unknown.ok)
        self.assertTrue(unknown.safety_violation)
        self.assertFalse(invalid.ok)
        self.assertFalse(invalid.schema_valid)

    def test_sandbox_blocks_escape_and_credential_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            registry = ToolRegistry(FilesystemTools(workspace).definitions())
            escape = registry.execute("fs.write", {"path": "../outside.txt", "content": "x"})
            credential = registry.execute("fs.write", {"path": ".env", "content": "SECRET=x"})
        self.assertFalse(escape.ok)
        self.assertTrue(escape.safety_violation)
        self.assertFalse(credential.ok)
        self.assertTrue(credential.safety_violation)

    def test_calculator_has_no_python_eval_surface(self) -> None:
        registry = ToolRegistry(CalculatorTool().definitions())
        result = registry.execute("calc.evaluate", {"expression": "2 + 3 * 4"})
        rejected = registry.execute("calc.evaluate", {"expression": "__import__('os').system('whoami')"})
        self.assertEqual(result.data["value"], 14)
        self.assertFalse(rejected.ok)

    def test_failure_tasks_emit_recovery_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = run_benchmarks(output)
            trajectories = [json.loads(line) for line in (output / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
        by_id = {item["task_id"]: item for item in summary}
        for task_id in ("B11", "B12", "B13", "B14", "B15"):
            self.assertTrue(by_id[task_id]["task_success"], task_id)
            self.assertGreater(by_id[task_id]["recovery_attempts"], 0, task_id)
        for trajectory in trajectories:
            if trajectory["task_id"] in {"B11", "B12", "B13", "B14", "B15"}:
                events = {event.get("event") for event in trajectory["events"]}
                self.assertIn("diagnosis", events, trajectory["task_id"])
                self.assertIn("recovery_decision", events, trajectory["task_id"])
                self.assertIn("verification", events, trajectory["task_id"])


if __name__ == "__main__":
    unittest.main()
