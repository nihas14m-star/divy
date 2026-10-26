"""Real-world task evaluation framework (point #10).

Provides objectively verifiable, end-to-end software-engineering-style tasks
that exercise understand -> locate -> modify -> test -> verify. Results are
deterministic and independent of any real model provider.

The reference task RW01 is a minimal bug-fixing scenario:

    1. fs.read   - inspect the broken file
    2. fs.patch  - repair the bug
    3. test.run  - run the registered test
    4. fs.read   - confirm the repair

The registered test (b04_calculator) independently verifies `add(2, 3) == 5`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .adapters import HttpModelAdapter, ProviderError
from .agent_loop import AgentLimits, AgentSession
from .evaluators import EvaluationResult
from .logger import EventLogger
from .model import ModelResponse
from .nvidia import nvidia_provider_from_environment, nvidia_registry_from_environment
from .performance import PerformanceMemory
from .providers import ModelMetadata, ModelProvider, ModelRegistry
from .routing import DeterministicRoutingPolicy, RoutingRequest
from .schemas import TaskDefinition
from .tools import CalculatorTool, FilesystemTools, SandboxWorkspace, TestRunnerTool, ToolRegistry
from .tools.tests import load_python_module

RW01 = TaskDefinition(
    "RW01",
    "Real-world bug fix",
    "Repair calculator.py so add(2, 3) returns 5 and verify with the registered test.",
    "RW01",
    ("calculator add returns 5", "registered test passes"),
    ("fs.read", "fs.patch", "test.run"),
)


def _rw01_fixture(workspace: SandboxWorkspace) -> None:
    workspace.write_text(
        "src/calculator.py",
        "def add(a, b):\n    return a - b\n",
    )


def _rw01_tests() -> dict:
    def b04_calculator(workspace: SandboxWorkspace) -> dict:
        module = load_python_module(workspace, "src/calculator.py")
        actual = module.add(2, 3)
        return {"test_id": "b04_calculator", "passed": actual == 5, "expected": 5, "actual": actual}

    return {"b04_calculator": b04_calculator}


class _RW01AgentAdapter:
    """Deterministic multi-turn adapter used as the local regression baseline."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [message.get("content", {}) for message in messages if message.get("role") == "tool"]
        successful = [o for o in observations if o.get("success")]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "src/calculator.py"}}}
        elif not any(o.get("tool") == "fs.patch" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.patch", "arguments": {"path": "src/calculator.py", "old": "return a - b", "new": "return a + b"}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "b04_calculator"}}}
        elif not any(o.get("tool") == "fs.write" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.write", "arguments": {"path": "CHANGELOG.md", "content": "# Change note\n\nRepaired add to return the sum of its inputs.\n"}}}
        else:
            content = {"final": {"message": "Completed the repair, test, and changelog."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


def _build_registry(workspace: SandboxWorkspace) -> ToolRegistry:
    definitions = FilesystemTools(workspace).definitions()
    definitions += CalculatorTool().definitions()
    definitions += TestRunnerTool(workspace, _rw01_tests()).definitions()
    return ToolRegistry(definitions)


def _rw01_verifier(workspace: SandboxWorkspace, trajectory: dict[str, Any]) -> EvaluationResult:
    source = workspace.read_text("src/calculator.py")
    test_passed = any(
        event.get("event") == "tool_observation"
        and event.get("tool") == "test.run"
        and event.get("success")
        and event.get("result", {}).get("passed")
        for event in trajectory.get("events", [])
    )
    # RW01's declared success criteria are the repair and the registered test.
    # Optional notes are evidence only; they must not silently become a hidden
    # requirement that changes the task the live model was given.
    note = workspace.resolve("CHANGELOG.md")
    note_valid = note.exists() and "Repaired add" in note.read_text(encoding="utf-8")
    source_repaired = "return a + b" in source
    success = source_repaired and test_passed
    return EvaluationResult(success, success, {"source_repaired": source_repaired, "test_passed": test_passed, "changelog_valid": note_valid}, recoverable_failure=not success and test_passed)


def run_realworld_task(
    task_id: str = "RW01",
    output_dir: Path | None = None,
    *,
    live: bool = False,
    live_model_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run RW01 through the bounded multi-turn agent session."""
    if task_id != "RW01":
        raise ValueError(f"unknown task_id: {task_id}")

    with tempfile.TemporaryDirectory(prefix="divy-realworld-RW01-") as temporary:
        workspace = SandboxWorkspace(Path(temporary))
        _rw01_fixture(workspace)
        registry = _build_registry(workspace)
        destination = output_dir or Path("results")
        destination.mkdir(parents=True, exist_ok=True)
        logger = EventLogger(destination / ("realworld_live_agent_events.jsonl" if live else "realworld_scripted_agent_events.jsonl"))
        memory = PerformanceMemory(destination / ("performance_memory_live.jsonl" if live else "performance_memory_scripted.jsonl"))
        if live:
            live_provider_name = os.environ.get("DIVY_LIVE_PROVIDER", "nvidia").lower()
            if live_provider_name == "nvidia":
                registry_for_routing = nvidia_registry_from_environment(live_model_ids)
            else:
                adapter = HttpModelAdapter.from_environment()
                metadata = ModelMetadata(adapter.model_id, os.environ.get("DIVY_MODEL_PROVIDER", "configured-provider"), frozenset({"coding", "structured_output"}), int(os.environ.get("DIVY_MODEL_CONTEXT_LIMIT", "4096")), float(os.environ.get("DIVY_MODEL_COST_ESTIMATE", "0")), float(os.environ.get("DIVY_MODEL_LATENCY_ESTIMATE_MS", "1000")), 1.0)
                provider = ModelProvider(metadata, adapter)
                registry_for_routing = ModelRegistry([provider])
        else:
            adapter = _RW01AgentAdapter()
            metadata = ModelMetadata("scripted-rw01", "deterministic", frozenset({"coding", "structured_output"}), 4096, 0.0, 1.0, 1.0)
            provider = ModelProvider(metadata, adapter)
            registry_for_routing = ModelRegistry([provider])
        decision = DeterministicRoutingPolicy().route(RoutingRequest("RW01", frozenset({"coding", "structured_output"}), task_type="software_engineering", complexity="moderate"), registry_for_routing)
        if not decision.selected_model_ids:
            raise ProviderError("routing", decision.rationale or "no capable live model was available")
        selected = registry_for_routing.get(decision.selected_model_ids[0])
        session = AgentSession(selected, registry, logger=logger, performance_memory=memory, limits=AgentLimits(max_iterations=12, max_tool_calls=12, max_recovery_attempts=2, max_cost=float(os.environ.get("DIVY_MODEL_MAX_COST", "0")) or None))
        session_result = session.run(
            RW01.goal,
            task_id=RW01.task_id,
            task_type="software_engineering",
            capabilities={"coding", "structured_output"},
            system_instructions="You are DIVY's bounded software-engineering agent. Use only registered tools. Return a structured action object {action:{tool,arguments}} when work is needed, or {final:{message}} only after the independent task is complete.",
            response_schema={"type": "object"},
            verifier=lambda trajectory: _rw01_verifier(workspace, trajectory),
        )
        result = session_result.trajectory
        evaluation = _rw01_verifier(workspace, result)
        result["evaluation"] = {
            "success": evaluation.success,
            "verification_passed": evaluation.verification_passed,
            "evidence": evaluation.evidence,
        }
        result["task_success"] = evaluation.success
        result["success"] = evaluation.success
        result["plan_valid"] = True
        result["live"] = live
        result["routing_decision"] = {
            "selected_model_ids": decision.selected_model_ids,
            "fallback_model_ids": decision.fallback_model_ids,
            "verifier_model_id": decision.verifier_model_id,
            "candidate_model_ids": decision.candidate_model_ids,
            "rationale": decision.rationale,
        }
        summary_path = destination / ("realworld_live_summary.json" if live else "realworld_scripted_summary.json")
        summary_path.write_text(json.dumps({"live": live, "task_id": RW01.task_id, "status": result["status"], "success": result["success"], "model_calls": result["model_calls"], "tool_calls": result["tool_calls"], "verification_result": result["verification_result"], "recovery_attempts": result["recovery_attempts"], "cost": result["cost"], "latency_ms": result["latency_ms"]}, indent=2) + "\n", encoding="utf-8")
        trajectory_path = destination / ("realworld_live_agent_trajectories.jsonl" if live else "realworld_scripted_agent_trajectories.jsonl")
        with trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        return result


def run_realworld_benchmark(output_dir: Path | None = None) -> dict[str, Any]:
    """Run the representative real-world software-engineering benchmark (RW01)."""
    return run_realworld_task("RW01", output_dir)


def run_realworld_live(output_dir: Path | None = None) -> dict[str, Any]:
    return run_realworld_task("RW01", output_dir, live=True)


def main() -> int:
    live = os.environ.get("DIVY_RUN_LIVE", "0") == "1"
    try:
        result = run_realworld_live() if live else run_realworld_benchmark()
    except ProviderError as exc:
        print(json.dumps({"live": True, "configured": exc.kind != "configuration", "success": False, "error": {"type": exc.kind, "message": str(exc)}} , indent=2))
        return 1
    print(json.dumps({"live": live, "success": result.get("success"), "status": result.get("status"), "task_id": result.get("task_id"), "tool_calls": result.get("tool_calls"), "model_calls": result.get("model_calls"), "verification_result": result.get("verification_result"), "recovery_attempts": result.get("recovery_attempts")}, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
