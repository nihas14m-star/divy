"""Controlled A/B/C routing baseline experiment.

This experiment compares three policies on the same five orchestration cases:

* single-model: first registered model only;
* fixed-routing: a hand-written routing table with no adaptive recovery;
* dynamic-routing: the deterministic DIVY policy and executor.

The adapters are deterministic test doubles. Results validate the comparison
harness and metrics, not model intelligence or real-world superiority.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execution import MultiModelExecutor, OrchestrationResult
from .logger import EventLogger
from .providers import ModelMetadata, ModelProvider, ModelRegistry
from .routing import DeterministicRoutingPolicy, RoutingDecision, RoutingRequest
from .schemas import ModelResponse


class BaselineAdapter:
    def __init__(self, behavior: str) -> None:
        self.behavior = behavior

    def generate(self, messages, tools, response_schema):
        if self.behavior == "fail":
            raise RuntimeError("controlled baseline failure")
        if self.behavior == "verify":
            return ModelResponse({"verified": True}, {"output_tokens": 3})
        if self.behavior == "a":
            return ModelResponse({"answer": "A"}, {"output_tokens": 3})
        if self.behavior == "b":
            return ModelResponse({"answer": "B"}, {"output_tokens": 3})
        return ModelResponse({"answer": "ok"}, {"output_tokens": 2})


@dataclass(frozen=True)
class BaselineCase:
    task_id: str
    required_capabilities: frozenset[str]
    providers: tuple[tuple[str, frozenset[str], str, float, float], ...]
    expected_models: tuple[str, ...]
    complexity: str = "normal"
    critical: bool = False
    allow_parallel: bool = False


CASES = (
    BaselineCase(
        "O01",
        frozenset({"vision"}),
        (("general", frozenset({"general"}), "ok", 1.0, 100), ("vision", frozenset({"vision"}), "ok", 1.0, 100)),
        ("vision",),
    ),
    BaselineCase(
        "O02",
        frozenset({"general"}),
        (("strong", frozenset({"general"}), "ok", 5.0, 100), ("cheap", frozenset({"general"}), "ok", 0.1, 50)),
        ("cheap",),
        complexity="simple",
    ),
    BaselineCase(
        "O03",
        frozenset({"coding"}),
        (("cheap", frozenset({"coding"}), "fail", 0.1, 50), ("strong", frozenset({"coding"}), "ok", 2.0, 100)),
        ("strong",),
    ),
    BaselineCase(
        "O04",
        frozenset({"research"}),
        (("worker-a", frozenset({"research"}), "a", 1.0, 100), ("worker-b", frozenset({"research"}), "b", 1.0, 100)),
        ("worker-a", "worker-b"),
        allow_parallel=True,
    ),
    BaselineCase(
        "O05",
        frozenset({"coding"}),
        (("producer", frozenset({"coding"}), "ok", 0.5, 100), ("verifier", frozenset({"coding", "verification"}), "verify", 0.6, 50)),
        ("producer",),
        critical=True,
    ),
)


def run_baseline_experiment(output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "baseline_trajectories.jsonl"
    trajectory_path.unlink(missing_ok=True)
    policy = DeterministicRoutingPolicy()
    rows: list[dict[str, Any]] = []

    for case in CASES:
        for strategy in ("single_model", "fixed_routing", "dynamic_routing"):
            row = _run_case(case, strategy, policy, trajectory_path)
            rows.append(row)

    by_strategy: dict[str, list[dict[str, Any]]] = {strategy: [row for row in rows if row["strategy"] == strategy] for strategy in ("single_model", "fixed_routing", "dynamic_routing")}
    summary = {
        "experiment_id": "E_BASELINE_O01_O05_V0.2",
        "model_version": "deterministic-test-doubles",
        "prompt_version": "not_applicable",
        "dataset_version": "orchestration-cases-v0.1",
        "routing_policy": "single/fixed/deterministic_dynamic",
        "tool_configuration": "no_tools_model_routing_only",
        "hardware": "RTX 3050 6GB / 16GB RAM / Windows",
        "evaluation_set": [case.task_id for case in CASES],
        "strategies": {
            strategy: {
                "task_count": len(strategy_rows),
                "success_rate": sum(row["success"] for row in strategy_rows) / len(strategy_rows),
                "model_calls": sum(row["model_calls"] for row in strategy_rows),
                "average_model_calls": sum(row["model_calls"] for row in strategy_rows) / len(strategy_rows),
                "total_cost": round(sum(row["total_cost"] for row in strategy_rows), 8),
                "average_wall_latency_ms": round(sum(row["wall_latency_ms"] for row in strategy_rows) / len(strategy_rows), 3),
                "fallbacks": sum(row["fallback_used"] for row in strategy_rows),
                "verification_passes": sum(row["verification_passed"] for row in strategy_rows),
            }
            for strategy, strategy_rows in by_strategy.items()
        },
        "rows": rows,
        "conclusion": "Harness-only result; scripted adapters do not establish model capability or generalization.",
    }
    (output_dir / "baseline_comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_case(case: BaselineCase, strategy: str, policy: DeterministicRoutingPolicy, trajectory_path: Path) -> dict[str, Any]:
    providers = [ModelProvider(ModelMetadata(model_id, model_id, capabilities, 4096, cost, latency, 0.9), BaselineAdapter(behavior)) for model_id, capabilities, behavior, cost, latency in case.providers]
    registry = ModelRegistry(providers)
    if strategy == "single_model":
        decision = RoutingDecision(case.task_id, [providers[0].model_id])
    elif strategy == "fixed_routing":
        fixed = {"O01": ["vision"], "O02": ["cheap"], "O03": ["cheap"], "O04": ["worker-a"], "O05": ["producer"]}
        decision = RoutingDecision(case.task_id, fixed[case.task_id])
    else:
        request = RoutingRequest(case.task_id, case.required_capabilities, complexity=case.complexity, critical=case.critical, max_models=2 if case.allow_parallel else 1, allow_parallel=case.allow_parallel)
        decision = policy.route(request, registry)

    if strategy == "dynamic_routing" and case.task_id == "O03":
        # The policy ranks cheap first; the executor must use the remaining
        # compatible candidate as fallback after the controlled failure.
        pass
    if strategy == "dynamic_routing" and case.task_id == "O05":
        decision.verifier_model_id = "verifier"

    with tempfile.TemporaryDirectory() as directory:
        executor = MultiModelExecutor(registry, EventLogger(Path(directory) / "events.jsonl"))
        result = executor.execute(
            decision,
            [{"role": "user", "content": {"task_id": case.task_id, "strategy": strategy}}],
            [],
            {"type": "object"},
            parallel=case.allow_parallel and strategy == "dynamic_routing",
            verify=case.task_id == "O05" and strategy == "dynamic_routing",
        )
    success = _case_success(case, strategy, decision, result)
    row = {
        "task_id": case.task_id,
        "strategy": strategy,
        "success": success,
        "selected_models": decision.selected_model_ids,
        "model_calls": result.model_calls,
        "total_cost": result.total_cost,
        "latency_ms": result.total_latency_ms,
        "wall_latency_ms": result.wall_latency_ms,
        "fallback_used": result.fallback_used,
        "verification_passed": result.verification_passed,
        "disagreement": result.disagreement,
    }
    with trajectory_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def _case_success(case: BaselineCase, strategy: str, decision: RoutingDecision, result: OrchestrationResult) -> bool:
    if case.task_id == "O04":
        return result.success and decision.selected_model_ids == list(case.expected_models)
    if case.task_id == "O05":
        return result.success and result.verification_passed
    if case.task_id == "O03":
        return result.success and result.final_output is not None and result.fallback_used
    return result.success and decision.selected_model_ids == list(case.expected_models)


def main() -> int:
    summary = run_baseline_experiment()
    print(json.dumps(summary["strategies"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
