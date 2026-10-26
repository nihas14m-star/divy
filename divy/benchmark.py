"""Benchmark runner and CLI for B01-B05."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .logger import EventLogger
from .model import ScriptedBenchmarkAdapter
from .orchestrator import Orchestrator
from .tasks import all_tasks, create_fixture, registered_tests
from .tools import CalculatorTool, FilesystemTools, TestRunnerTool, ToolRegistry
from .verifier import DeterministicVerifier


def run_benchmarks(output_dir: Path | None = None) -> list[dict]:
    output_dir = output_dir or Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectories.jsonl"
    event_path = output_dir / "events.jsonl"
    trajectory_path.unlink(missing_ok=True)
    event_path.unlink(missing_ok=True)
    summary: list[dict] = []

    for task in all_tasks():
        with tempfile.TemporaryDirectory(prefix=f"divy-{task.task_id}-") as temporary:
            workspace = create_fixture(task, Path(temporary))
            definitions = FilesystemTools(workspace).definitions()
            definitions += CalculatorTool().definitions()
            definitions += TestRunnerTool(workspace, registered_tests()).definitions()
            registry = ToolRegistry(definitions)
            logger = EventLogger(event_path)
            verifier = DeterministicVerifier()
            orchestrator = Orchestrator(
                ScriptedBenchmarkAdapter(task),
                registry,
                verifier,
                logger,
                observation_hook=_observation_hook,
            )
            result = orchestrator.run(task, workspace)
            evaluation = verifier.verify(task, workspace, result)
            result["evaluation"] = {
                "success": evaluation.success,
                "verification_passed": evaluation.verification_passed,
                "evidence": evaluation.evidence,
            }
            result["task_success"] = evaluation.success
            result["false_acceptance"] = result["status"] == "success" and not evaluation.success
            result["false_rejection"] = result["status"] not in {"success", "needs_clarification"} and evaluation.success
            result["verification_accuracy"] = result["verification_result"] == evaluation.success
            result["recovery_success"] = task.task_id in {"B11", "B12", "B13", "B14", "B15"} and evaluation.success and result["recovery_attempts"] > 0
            logger.append({"event": "benchmark_evaluation", "task_id": task.task_id, **result["evaluation"]})
            with trajectory_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            summary.append({
                "task_id": task.task_id,
                "status": result["status"],
                "task_success": result["task_success"],
                "plan_valid": result["plan_valid"],
                "tool_selection_valid": result["tool_selection_valid"],
                "argument_valid": result["argument_valid"],
                "verification_result": result["verification_result"],
                "verification_accuracy": result["verification_accuracy"],
                "false_acceptance": result["false_acceptance"],
                "false_rejection": result["false_rejection"],
                "recovery_success": result["recovery_success"],
                "recovery_attempts": result["recovery_attempts"],
                "unnecessary_actions": result["unnecessary_actions"],
                "tool_calls": result["tool_calls"],
                "safety_violations": result["safety_violations"],
                "latency_ms": result["latency_ms"],
                "token_usage": result["token_usage"],
            })

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    aggregate = {
        "task_count": len(summary),
        "success_rate": sum(item["task_success"] for item in summary) / len(summary) if summary else 0,
        "recovery_tasks": sum(item["task_id"] in {"B11", "B12", "B13", "B14", "B15"} for item in summary),
        "recovery_success_rate": sum(item["recovery_success"] for item in summary) / 5 if summary else 0,
        "false_acceptance_rate": sum(item["false_acceptance"] for item in summary) / len(summary) if summary else 0,
        "verification_accuracy": sum(item["verification_accuracy"] for item in summary) / len(summary) if summary else 0,
        "total_tool_calls": sum(item["tool_calls"] for item in summary),
        "average_tool_calls": sum(item["tool_calls"] for item in summary) / len(summary) if summary else 0,
        "total_unnecessary_actions": sum(item["unnecessary_actions"] for item in summary),
        "total_safety_violations": sum(item["safety_violations"] for item in summary),
        "total_recovery_attempts": sum(item["recovery_attempts"] for item in summary),
    }
    (output_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return summary


def _observation_hook(task, workspace, trajectory):
    """Inject one deterministic external mutation for B14 after the read."""
    if task.task_id != "B14":
        return None
    if any(event.get("event") == "environment_mutation" for event in trajectory.get("events", [])):
        return None
    if not any(event.get("event") == "tool_result" and event.get("tool") == "fs.read" and event.get("ok") for event in trajectory.get("events", [])):
        return None
    current = json.loads(workspace.read_text("config.json"))
    current["version"] = 2
    workspace.write_text("config.json", json.dumps(current, indent=2) + "\n")
    return {"event": "environment_mutation", "reason": "external_update_after_observation", "path": "config.json", "change": {"version": 2}}


def main() -> int:
    summary = run_benchmarks()
    print(json.dumps(summary, indent=2))
    passed = sum(item["task_success"] for item in summary)
    metrics = json.loads((Path("results") / "metrics.json").read_text(encoding="utf-8"))
    print(f"Passed {passed}/{len(summary)} tasks")
    print(json.dumps(metrics, indent=2))
    return 0 if passed == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
