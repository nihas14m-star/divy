"""O01-O10 benchmarks for provider selection and multi-model orchestration."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from .execution import MultiModelExecutor
from .logger import EventLogger
from .model import ModelAdapter
from .providers import ModelMetadata, ModelProvider, ModelRegistry
from .routing import DeterministicRoutingPolicy, RoutingRequest
from .schemas import ModelResponse


class ScenarioAdapter:
    def __init__(self, behavior: str, delay_ms: float = 0.0) -> None:
        self.behavior = behavior
        self.delay_ms = delay_ms
        self.calls = 0

    def generate(self, messages, tools, response_schema):
        self.calls += 1
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        if self.behavior == "fail":
            raise RuntimeError("simulated provider failure")
        if self.behavior == "invalid":
            return ModelResponse(content={})
        if self.behavior == "wrong":
            return ModelResponse(content={"answer": "wrong"}, token_usage={"output_tokens": 2})
        if self.behavior == "verify_yes":
            return ModelResponse(content={"verified": True}, token_usage={"output_tokens": 2})
        if self.behavior == "verify_no":
            return ModelResponse(content={"verified": False}, token_usage={"output_tokens": 2})
        if self.behavior == "solution_a":
            return ModelResponse(content={"answer": "A"}, token_usage={"output_tokens": 4})
        if self.behavior == "solution_b":
            return ModelResponse(content={"answer": "B"}, token_usage={"output_tokens": 4})
        if self.behavior == "complex":
            return ModelResponse(content={"answer": "complex-solved"}, token_usage={"output_tokens": 8})
        return ModelResponse(content={"answer": "ok"}, token_usage={"output_tokens": 2})


def _provider(model_id: str, capabilities: set[str], behavior: str, *, cost: float, latency: float, reliability: float = 0.9, delay_ms: float = 0.0) -> ModelProvider:
    metadata = ModelMetadata(model_id, model_id.split("-")[0], frozenset(capabilities), 4096, cost, latency, reliability)
    return ModelProvider(metadata, ScenarioAdapter(behavior, delay_ms))


def _registry(*providers: ModelProvider) -> ModelRegistry:
    return ModelRegistry(list(providers))


def run_orchestration_benchmarks(output_dir: Path | None = None) -> list[dict]:
    output_dir = output_dir or Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "orchestration_trajectories.jsonl"
    trajectory_path.unlink(missing_ok=True)
    policy = DeterministicRoutingPolicy()
    results: list[dict] = []
    for task_id in [f"O{i:02d}" for i in range(1, 11)]:
        result = _run_one(task_id, policy, trajectory_path)
        results.append(result)
    (output_dir / "orchestration_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "task_count": len(results),
        "success_rate": sum(item["success"] for item in results) / len(results),
        "total_model_calls": sum(item["model_calls"] for item in results),
        "average_model_calls": sum(item["model_calls"] for item in results) / len(results),
        "total_cost": round(sum(item["total_cost"] for item in results), 8),
        "average_latency_ms": round(sum(item["latency_ms"] for item in results) / len(results), 3),
        "average_wall_latency_ms": round(sum(item["wall_latency_ms"] for item in results) / len(results), 3),
        "fallback_successes": sum(item["fallback_used"] for item in results),
        "verification_passes": sum(item["verification_passed"] for item in results),
        "disagreements": sum(item["disagreement"] for item in results),
    }
    (output_dir / "orchestration_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return results


def _run_one(task_id: str, policy: DeterministicRoutingPolicy, trajectory_path: Path) -> dict:
    if task_id == "O01":
        registry = _registry(_provider("general", {"general"}, "ok", cost=1.0, latency=100), _provider("vision", {"vision"}, "ok", cost=1.0, latency=100))
        request = RoutingRequest(task_id, frozenset({"vision"}), complexity="normal")
        expected = ["vision"]
    elif task_id == "O02":
        registry = _registry(_provider("cheap", {"general"}, "ok", cost=0.1, latency=50), _provider("strong", {"general"}, "complex", cost=5.0, latency=100))
        request = RoutingRequest(task_id, frozenset({"general"}), complexity="simple", budget=1.0)
        expected = ["cheap"]
    elif task_id == "O03":
        registry = _registry(_provider("cheap", {"coding"}, "fail", cost=0.1, latency=50), _provider("strong", {"coding"}, "ok", cost=2.0, latency=100))
        request = RoutingRequest(task_id, frozenset({"coding"}), complexity="normal")
        expected = ["cheap", "strong"]
    elif task_id == "O04":
        registry = _registry(_provider("worker-a", {"research"}, "solution_a", cost=1.0, latency=100, delay_ms=5), _provider("worker-b", {"research"}, "solution_b", cost=1.0, latency=100, delay_ms=5))
        request = RoutingRequest(task_id, frozenset({"research"}), max_models=2, allow_parallel=True)
        expected = ["worker-a", "worker-b"]
    elif task_id == "O05":
        registry = _registry(_provider("producer", {"coding"}, "solution_a", cost=0.5, latency=100, reliability=0.95), _provider("verifier", {"coding", "verification"}, "verify_yes", cost=0.6, latency=50, reliability=0.9))
        request = RoutingRequest(task_id, frozenset({"coding"}), critical=True)
        expected = ["producer"]
    elif task_id == "O06":
        registry = _registry(_provider("a", {"general"}, "solution_a", cost=1.0, latency=100), _provider("b", {"general"}, "solution_b", cost=1.0, latency=100), _provider("synth", {"synthesis"}, "synth", cost=0.5, latency=80))
        request = RoutingRequest(task_id, frozenset({"general"}), max_models=2, allow_parallel=True)
        expected = ["a", "b"]
    elif task_id == "O07":
        registry = _registry(_provider("primary", {"general"}, "fail", cost=1.0, latency=100), _provider("fallback", {"general"}, "ok", cost=1.5, latency=100))
        request = RoutingRequest(task_id, frozenset({"general"}))
        expected = ["primary", "fallback"]
    elif task_id == "O08":
        registry = _registry(_provider("researcher", {"research"}, "ok", cost=1.0, latency=100), _provider("coder", {"coding"}, "ok", cost=1.0, latency=100))
        request = RoutingRequest(task_id, frozenset({"research"}), max_models=1)
        expected = ["researcher", "coder"]
    elif task_id == "O09":
        registry = _registry(_provider("only", {"general"}, "ok", cost=1.0, latency=100))
        request = RoutingRequest(task_id, frozenset({"general"}))
        expected = ["only"]
    else:
        registry = _registry(_provider("fast", {"general"}, "ok", cost=0.1, latency=50), _provider("deep", {"general"}, "complex", cost=3.0, latency=200, reliability=0.99))
        request = RoutingRequest(task_id, frozenset({"general"}), complexity="complex", critical=True)
        expected = ["deep"]

    decision = policy.route(request, registry)
    if task_id == "O06":
        decision.synthesis_model_id = "synth"
    with tempfile.TemporaryDirectory() as directory:
        logger = EventLogger(Path(directory) / "events.jsonl")
        executor = MultiModelExecutor(registry, logger)
        messages = [{"role": "user", "content": {"task_id": task_id}}]
        if task_id == "O08":
            research_decision = policy.route(RoutingRequest(task_id + "-research", frozenset({"research"})), registry)
            coding_decision = policy.route(RoutingRequest(task_id + "-coding", frozenset({"coding"})), registry)
            result = executor.execute(research_decision, messages, [], {"type": "object"})
            coding_result = executor.execute(coding_decision, messages, [], {"type": "object"})
            result.calls.extend(coding_result.calls)
            result.success = result.success and coding_result.success
            result.final_output = {"research": result.final_output, "coding": coding_result.final_output}
            decision.selected_model_ids = research_decision.selected_model_ids + coding_decision.selected_model_ids
            decision.candidate_model_ids = research_decision.candidate_model_ids + coding_decision.candidate_model_ids
        else:
            result = executor.execute(decision, messages, [], {"type": "object"}, parallel=request.allow_parallel, verify=task_id == "O05", synthesize=task_id == "O06")
    success = _evaluate(task_id, decision, result, expected)
    row = {
        "task_id": task_id,
        "success": success,
        "selected_models": decision.selected_model_ids,
        "fallback_models": decision.fallback_model_ids,
        "candidate_models": decision.candidate_model_ids,
        "model_calls": result.model_calls,
        "total_cost": result.total_cost,
        "latency_ms": result.total_latency_ms,
        "wall_latency_ms": result.wall_latency_ms,
        "fallback_used": result.fallback_used,
        "verification_passed": result.verification_passed,
        "disagreement": result.disagreement,
        "rationale": decision.rationale,
    }
    with trajectory_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_id": task_id, "decision": decision.__dict__, "result": row}, sort_keys=True) + "\n")
    return row


def _evaluate(task_id: str, decision, result, expected: list[str]) -> bool:
    if task_id == "O03":
        return result.success and result.fallback_used and result.calls[-1].model_id == "strong"
    if task_id == "O04":
        return result.success and decision.selected_model_ids == expected and len(result.calls) == 2
    if task_id == "O05":
        return result.success and result.verification_passed
    if task_id == "O06":
        return result.success and result.disagreement and result.final_output is not None
    if task_id == "O07":
        return result.success and result.fallback_used
    return result.success and decision.selected_model_ids == expected


def main() -> int:
    results = run_orchestration_benchmarks()
    print(json.dumps(results, indent=2))
    return 0 if all(item["success"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
