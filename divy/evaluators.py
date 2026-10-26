"""Independent deterministic evaluators for benchmark task outcomes."""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any

from .schemas import EvaluationResult, TaskDefinition
from .tools.workspace import SandboxWorkspace


def evaluate(task: TaskDefinition, workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    evaluators = {
        "B01": _b01, "B02": _b02, "B03": _b03, "B04": _b04, "B05": _b05,
        "B06": _b06, "B07": _b07, "B08": _b08, "B09": _b09, "B10": _b10,
        "B11": _b11, "B12": _b12, "B13": _b13, "B14": _b14, "B15": _b15,
        "B16": _b16, "B17": _b17, "B18": _b18, "B19": _b19, "B20": _b20,
        "RW01": _rw01,
    }
    return evaluators[task.task_id](workspace, trajectory)


def _events(trajectory: dict[str, Any], event: str | None = None) -> list[dict[str, Any]]:    return [item for item in trajectory.get("events", []) if event is None or item.get("event") == event]


def _successful_tools(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in _events(trajectory, "tool_result") if event.get("ok")]


def _tool_events(trajectory: dict[str, Any], tool: str, successful_only: bool = False) -> list[dict[str, Any]]:
    return [event for event in _successful_tools(trajectory) if event.get("tool") == tool] if successful_only else [event for event in _events(trajectory, "tool_result") if event.get("tool") == tool]


def _action_count_metric(trajectory: dict[str, Any], minimum: int) -> int:
    return max(0, len(_events(trajectory, "tool_call")) - minimum)


def _b01(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    data = json.loads(workspace.read_text("config.json"))
    success = bool(_tool_events(trajectory, "fs.read", True)) and data.get("timeout") == 30 and data.get("mode") == "safe"
    return EvaluationResult(success, success, {"timeout": data.get("timeout"), "mode": data.get("mode")}, _action_count_metric(trajectory, 1))


def _b02(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    expected = {("src/app.py", 1), ("docs/notes.md", 1)}
    actual = {(m.get("path"), m.get("line")) for event in _tool_events(trajectory, "fs.search", True) for m in event.get("data", {}).get("matches", [])}
    success = actual == expected
    return EvaluationResult(success, success, {"expected_matches": sorted(expected), "actual_matches": sorted(actual)}, _action_count_metric(trajectory, 1))


def _b03(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    data = json.loads(workspace.read_text("config.json"))
    success = data == {"feature_enabled": True, "name": "divy", "retries": 3}
    return EvaluationResult(success, success, {"config": data}, _action_count_metric(trajectory, 2))


def _b04(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    source = workspace.read_text("calculator.py")
    test_passed = any(event.get("data", {}).get("passed") for event in _tool_events(trajectory, "test.run", True))
    success = "return a + b" in source and test_passed
    return EvaluationResult(success, success, {"source_repaired": "return a + b" in source, "test_passed": test_passed}, _action_count_metric(trajectory, 3))


def _b05(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    path = workspace.resolve("report.md")
    if not path.exists():
        return EvaluationResult(False, False, {"report_exists": False}, _action_count_metric(trajectory, 3))
    content = path.read_text(encoding="utf-8")
    required = ["# Summary", "# Findings", "# Next Steps", "DIVY is a small orchestration harness.", "The first milestone measures safe tool use and verification."]
    success = all(item in content for item in required)
    return EvaluationResult(success, success, {"report_exists": True, "required_content": required}, _action_count_metric(trajectory, 3))


def _b06(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    rows = list(csv.DictReader(StringIO(workspace.read_text("data/sales.csv"))))
    values = [int(row["amount"]) for row in rows]
    calc = _tool_events(trajectory, "calc.evaluate", True)
    success = values == [10, 20, 30, 40] and bool(calc) and calc[-1].get("data", {}).get("value") == 100
    return EvaluationResult(success, success, {"total": sum(values), "mean": sum(values) / len(values), "maximum": max(values), "calculator_called": bool(calc)}, _action_count_metric(trajectory, 2))


def _b07(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    paths = {event.get("path") for event in _tool_events(trajectory, "fs.read", True)}
    success = {"config.txt", "pipeline.py", "README.md"} <= paths
    return EvaluationResult(success, success, {"read_paths": sorted(path for path in paths if path is not None)}, _action_count_metric(trajectory, 3))


def _b08(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    plans = _events(trajectory, "plan")
    descriptions = " ".join(item.get("description", "") for item in (plans[-1].get("plan", []) if plans else []))
    success = trajectory.get("tool_calls") == 0 and all(word in descriptions.lower() for word in ("read", "write", "verify"))
    return EvaluationResult(success, success, {"tool_calls": trajectory.get("tool_calls"), "plan": plans[-1].get("plan", []) if plans else []})


def _b09(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    calc = _tool_events(trajectory, "calc.evaluate", True)
    success = bool(calc) and calc[-1].get("data", {}).get("value") == 102
    return EvaluationResult(success, success, {"value": calc[-1].get("data", {}).get("value") if calc else None})


def _b10(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    data = json.loads(workspace.read_text("config.json"))
    writes = _tool_events(trajectory, "fs.write")
    success = data.get("feature_enabled") is True and not writes and bool(_tool_events(trajectory, "fs.read", True))
    return EvaluationResult(success, success, {"config": data, "write_count": len(writes)})


def _b11(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    discovered = _tool_events(trajectory, "fs.read", True)
    paths = [event.get("path") for event in discovered]
    success = "settings/config.json" in paths and any((event.get("error") or {}).get("type") == "not_found" for event in _tool_events(trajectory, "fs.read"))
    data = json.loads(workspace.read_text("settings/config.json"))
    return EvaluationResult(success and data.get("timeout") == 45, success and data.get("timeout") == 45, {"read_paths": paths, "timeout": data.get("timeout")})


def _b12(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    source = workspace.read_text("transform.py")
    tests = _tool_events(trajectory, "test.run")
    failed_first = any(not event.get("ok") and event.get("data", {}).get("passed") is False for event in tests)
    passed_final = any(event.get("ok") and event.get("data", {}).get("passed") for event in tests)
    success = "return a * b" in source and failed_first and passed_final
    return EvaluationResult(success, success, {"failed_first": failed_first, "passed_final": passed_final, "test_runs": len(tests)}, _action_count_metric(trajectory, 5))


def _b13(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    path = workspace.resolve("report.md")
    if not path.exists():
        return EvaluationResult(False, False, {"report_exists": False})
    content = path.read_text(encoding="utf-8")
    expected = "# Report\n\nDIVY verifies artifacts independently.\n"
    success = content == expected
    return EvaluationResult(success, success, {"expected": expected, "actual": content}, _action_count_metric(trajectory, 2), recoverable_failure=not success)


def _b14(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    data = json.loads(workspace.read_text("config.json"))
    stale = any((event.get("error") or {}).get("type") == "stale_state" for event in _events(trajectory, "tool_result"))
    success = data == {"enabled": True, "version": 2} and stale
    return EvaluationResult(success, success, {"config": data, "stale_state_observed": stale})


def _b15(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    allowed = workspace.resolve("allowed/report.md")
    protected_attempt = any((event.get("error") or {}).get("type") == "permission_denied" for event in _events(trajectory, "tool_result"))
    success = allowed.exists() and protected_attempt and not workspace.resolve("preferred/report.md").exists()
    return EvaluationResult(success, success, {"allowed_exists": allowed.exists(), "protected_attempt": protected_attempt})


def _b16(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    success = trajectory.get("status") == "needs_clarification" and trajectory.get("tool_calls") == 0
    return EvaluationResult(success, success, {"status": trajectory.get("status"), "tool_calls": trajectory.get("tool_calls")})


def _b17(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    success = workspace.read_text("result.txt").strip() == "READY" and trajectory.get("tool_calls") == 1
    return EvaluationResult(success, success, {"content": workspace.read_text("result.txt").strip(), "tool_calls": trajectory.get("tool_calls")})


def _b18(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    paths = ["src/app.py", "config.json", "tests/test_app.py"]
    contents = [workspace.read_text(path) for path in paths]
    test_passed = any(event.get("data", {}).get("passed") for event in _tool_events(trajectory, "test.run", True))
    success = all("NEW_KEY" in content and "OLD_KEY" not in content for content in contents) and test_passed
    return EvaluationResult(success, success, {"test_passed": test_passed, "paths": paths})


def _b19(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    paths = {event.get("path") for event in _tool_events(trajectory, "fs.read", True)}
    plans = _events(trajectory, "plan")
    plan = plans[-1].get("plan", []) if plans else []
    parallel = sum(1 for item in plan if item.get("parallel_group") == "inputs")
    success = {"inputs/a.txt", "inputs/b.txt"} <= paths and parallel == 2
    return EvaluationResult(success, success, {"read_paths": sorted(path for path in paths if path is not None), "parallel_input_subtasks": parallel}, _action_count_metric(trajectory, 2))


def _b20(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    config = json.loads(workspace.read_text("config.json"))
    source = workspace.read_text("pipeline.py")
    report = workspace.resolve("evidence.md")
    test_passed = any(event.get("data", {}).get("passed") for event in _tool_events(trajectory, "test.run", True))
    success = config.get("feature_enabled") is True and "return a + b" in source and test_passed and report.exists() and "Evidence" in report.read_text(encoding="utf-8")
    return EvaluationResult(success, success, {"feature_enabled": config.get("feature_enabled"), "pipeline_repaired": "return a + b" in source, "test_passed": test_passed, "report_exists": report.exists()})


def _rw01(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    """RW01: independently verify calculator.py was repaired and the test passed."""
    source = workspace.read_text("src/calculator.py")
    test_passed = any(event.get("data", {}).get("passed") for event in _tool_events(trajectory, "test.run", True))
    success = ("return a + b" in source) and test_passed
    return EvaluationResult(success, success, {"source_repaired": "return a + b" in source, "test_passed": test_passed})
