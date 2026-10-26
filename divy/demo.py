"""Deterministic local DIVY product demo.

The demo uses the same bounded AgentSession and registered tools as the
real-world harness, but replaces the hosted model with a deterministic local
adapter. It never needs NVIDIA credentials or network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import ExecutionMode, evaluate_execution
from .performance import PerformanceMemory
from .realworld import run_realworld_benchmark


DEMO_FILES = (
    "realworld_scripted_agent_events.jsonl",
    "realworld_scripted_agent_trajectories.jsonl",
    "realworld_scripted_summary.json",
    "performance_memory_scripted.jsonl",
    "demo_evaluation_report.json",
    "demo_summary.json",
)


def run_demo(output_dir: Path | str = Path("results/demo")) -> dict[str, Any]:
    """Run the reproducible local loop and write a compact evaluation report."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in DEMO_FILES:
        (destination / filename).unlink(missing_ok=True)

    trajectory = run_realworld_benchmark(destination)
    report = evaluate_execution(
        trajectory,
        routing_decision=trajectory.get("routing_decision"),
        execution_mode=ExecutionMode.scripted(),
    )
    report_path = destination / "demo_evaluation_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    memory = PerformanceMemory(destination / "performance_memory_scripted.jsonl")
    summary = {
        "task": trajectory.get("task_id"),
        "routing_decision": trajectory.get("routing_decision"),
        "selected_model": (trajectory.get("routing_decision") or {}).get("selected_model_ids", [None])[0],
        "tool_calls": trajectory.get("tool_calls", 0),
        "verification_passed": trajectory.get("verification_result"),
        "task_success": trajectory.get("task_success", trajectory.get("success", False)),
        "performance_observations": memory.count(),
        "evaluation_report": str(report_path),
        "execution_mode": str(report.execution_mode),
    }
    (destination / "demo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_demo()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("DIVY demo completed: route -> select -> tools -> verify -> observe -> evaluate")
    return 0 if summary["task_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
