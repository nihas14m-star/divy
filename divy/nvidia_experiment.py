"""Controlled live NVIDIA comparisons for the repository-shaped RW01 task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .nvidia import NVIDIA_MODEL_SPECS
from .realworld import run_realworld_task


def _compact(result: dict[str, Any], strategy: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "model_id": result.get("model_id"),
        "provider": result.get("provider"),
        "status": result.get("status"),
        "success": result.get("success"),
        "verification_result": result.get("verification_result"),
        "model_calls": result.get("model_calls"),
        "tool_calls": result.get("tool_calls"),
        "recovery_attempts": result.get("recovery_attempts"),
        "latency_ms": result.get("latency_ms"),
        "token_usage": result.get("token_usage"),
        "cost": result.get("cost"),
        "routing_decision": result.get("routing_decision"),
    }


def run_controlled_comparison(output_dir: Path | None = None) -> dict[str, Any]:
    destination = output_dir or Path("results") / "nvidia_controlled"
    destination.mkdir(parents=True, exist_ok=True)
    model_ids = [spec.model_id for spec in NVIDIA_MODEL_SPECS]
    # A: one fixed model, B: a different fixed model, C: the deterministic
    # router chooses from the complete capability-compatible portfolio.
    experiments = (
        ("single_model", [model_ids[0]]),
        ("fixed_routing", [model_ids[1]]),
        ("dynamic_routing", model_ids),
    )
    results: list[dict[str, Any]] = []
    for strategy, selected_models in experiments:
        run_dir = destination / strategy
        result = run_realworld_task("RW01", run_dir, live=True, live_model_ids=selected_models)
        results.append(_compact(result, strategy))
    comparison = {
        "live": True,
        "task_id": "RW01",
        "models_considered": model_ids,
        "results": results,
        "conclusion": "One RW01 per strategy is evidence of execution and instrumentation only; it is not a model-quality ranking.",
    }
    (destination / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparison


def main() -> int:
    result = run_controlled_comparison()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(item.get("success") for item in result["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
