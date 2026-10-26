"""Experiment runner for controlled real-model evaluation campaigns."""

from __future__ import annotations

import json
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_loop import AgentLimits, AgentSession
from .evaluation import evaluate_execution, compare_routing_strategies, ExecutionMode
from .live_tasks import (
    LIVE_TASKS, get_task, setup_fixture, verify_task, build_registry, get_all_task_ids
)
from .logger import EventLogger
from .nvidia import nvidia_registry_from_environment, NVIDIA_MODEL_SPECS
from .performance import PerformanceMemory
from .providers import ModelMetadata, ModelProvider, ModelRegistry
from .routing import DeterministicRoutingPolicy, RoutingRequest
from .schemas import TaskDefinition, EvaluationResult
from .tools.workspace import SandboxWorkspace


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment run."""
    task_id: str
    model_ids: list[str]
    strategy: str  # "single", "fixed", or "dynamic"
    live: bool = True
    max_cost: float | None = None
    max_iterations: int = 12
    max_tool_calls: int = 12
    max_recovery: int = 2
    seed: int | None = None


@dataclass
class TrialResult:
    """Result of a single trial."""
    task_id: str
    model_id: str
    strategy: str
    trial_num: int
    success: bool
    status: str
    verification_passed: bool | None
    model_calls: int
    tool_calls: int
    recovery_attempts: int
    latency_ms: float | None
    total_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    cost: float | None
    routing_decision: dict[str, Any] | None
    failure_type: str | None = None
    seed: int | None = None


class ExperimentRunner:
    """Runs controlled evaluation campaigns across models and tasks."""

    def __init__(
        self,
        output_dir: Path | str,
        live: bool = True,
        base_model_ids: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.live = live
        self.base_model_ids = base_model_ids or [spec.model_id for spec in NVIDIA_MODEL_SPECS]
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Performance memory for accumulating evidence
        self.memory = PerformanceMemory(self.output_dir / ("performance_memory_live.jsonl" if live else "performance_memory_scripted.jsonl"))

        # Statistics
        self.trials: list[TrialResult] = []
        self.total_trials = 0
        self.successful_trials = 0
        self.failed_trials = 0

    def run_trial(self, config: ExperimentConfig, trial_num: int) -> TrialResult:
        """Run a single trial with the given configuration."""
        if config.live != self.live:
            raise ValueError("trial execution mode must match its runner's evidence store")
        if config.seed is not None:
            random.seed(config.seed)

        task = LIVE_TASKS[config.task_id]

        with tempfile.TemporaryDirectory(prefix=f"divy-exp-{config.task_id}-{config.model_ids[0].split('/')[-1]}-") as temporary:
            workspace = SandboxWorkspace(Path(temporary))
            setup_fixture(config.task_id, workspace)
            registry = build_registry(workspace, config.task_id)

            destination = self.output_dir / "trials" / f"{config.task_id}_{config.strategy}_t{trial_num}"
            destination.mkdir(parents=True, exist_ok=False)

            logger = EventLogger(destination / "events.jsonl")
            memory = self.memory

            # Build model registry based on strategy
            if config.strategy == "oracle":
                raise ValueError(
                    "oracle routing is future research and is intentionally not "
                    "part of the validated experiment runner"
                )

            if config.live:
                ids = config.model_ids if config.strategy == "dynamic" else config.model_ids[:1]
                registry_for_routing = nvidia_registry_from_environment(ids)
            else:
                # Use scripted adapter for non-live
                from .campaign import get_scripted_adapter
                adapter = get_scripted_adapter(config.task_id, workspace_root=temporary)
                metadata = ModelMetadata("scripted", "deterministic", task.required_capabilities, 4096, 0.0, 1.0, 1.0)
                provider = ModelProvider(metadata, adapter)
                registry_for_routing = ModelRegistry([provider])

            # Route
            req = RoutingRequest(
                task_id=config.task_id,
                required_capabilities=LIVE_TASKS[config.task_id].required_capabilities,
                task_type=LIVE_TASKS[config.task_id].task_type,
                complexity=LIVE_TASKS[config.task_id].complexity,
            )
            policy = DeterministicRoutingPolicy(
                performance_memory=self.memory,
                empirical_weight=0.3,
                confidence_samples=10.0,
                recency_half_life_days=30.0,
                recency_weight=0.3,
                exploration_rate=0.1,
                seed=config.seed,
            )
            decision = policy.route(req, registry_for_routing)

            if not decision.selected_model_ids:
                return TrialResult(
                    task_id=config.task_id,
                    model_id="none",
                    strategy=config.strategy,
                    trial_num=trial_num,
                    success=False,
                    status="no_capable_model",
                    verification_passed=False,
                    model_calls=0,
                    tool_calls=0,
                    recovery_attempts=0,
                    latency_ms=None,
                    total_tokens=None,
                    input_tokens=None,
                    output_tokens=None,
                    cost=None,
                    routing_decision={"rationale": decision.rationale},
                    failure_type="no_capable_model",
                    seed=config.seed,
                )

            selected = registry_for_routing.get(decision.selected_model_ids[0])

            # Run session
            session = AgentSession(
                selected,
                registry,
                logger=logger,
                performance_memory=memory,
                limits=AgentLimits(
                    max_iterations=config.max_iterations,
                    max_tool_calls=config.max_tool_calls,
                    max_recovery_attempts=config.max_recovery,
                    max_cost=config.max_cost,
                ),
            )

            task_def = TaskDefinition(
                task_id=task.task_id,
                title=task.title,
                goal=task.goal,
                fixture_name=task.fixture_name,
                success_criteria=task.success_criteria,
                allowed_tools=task.allowed_tools,
            )

            session_result = session.run(
                task.goal,
                task_id=task.task_id,
                task_type=task.task_type,
                capabilities=set(task.required_capabilities),
                system_instructions="You are DIVY's bounded agent. Use only registered tools. Return {action:{tool,arguments}} or {final:{message}}. After making changes, you MUST run the registered test to verify your fix using the test.run tool with the appropriate test_id. For RW01, the registered test_id is 'test_add'.",
                response_schema={"type": "object"},
                verifier=lambda traj: EvaluationResult(
                    success=self._verify_task(config.task_id, workspace, traj),
                    verification_passed=self._verify_task(config.task_id, workspace, traj),
                    evidence={},
                ),
            )

            result = session_result.trajectory
            evaluation_passed = verify_task(config.task_id, workspace, result)
            result["evaluation"] = {"success": evaluation_passed, "verification_passed": evaluation_passed}
            result["task_success"] = evaluation_passed
            result["success"] = evaluation_passed
            result["live"] = self.live
            result["routing_decision"] = {
                "selected_model_ids": decision.selected_model_ids,
                "fallback_model_ids": decision.fallback_model_ids,
                "verifier_model_id": decision.verifier_model_id,
                "candidate_model_ids": decision.candidate_model_ids,
                "rationale": decision.rationale,
            }

            # Save trajectory
            traj_path = destination / "trajectory.jsonl"
            with traj_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")

            # Record trial
            trial = TrialResult(
                task_id=config.task_id,
                model_id=decision.selected_model_ids[0] if decision.selected_model_ids else "none",
                strategy=config.strategy,
                trial_num=trial_num,
                success=result.get("success", False),
                status=result.get("status", "unknown"),
                verification_passed=result.get("verification_result"),
                model_calls=result.get("model_calls", 0),
                tool_calls=result.get("tool_calls", 0),
                recovery_attempts=result.get("recovery_attempts", 0),
                latency_ms=result.get("latency_ms"),
                total_tokens=result.get("token_usage", {}).get("total_tokens") if result.get("token_usage") else None,
                input_tokens=result.get("token_usage", {}).get("input_tokens") if result.get("token_usage") else None,
                output_tokens=result.get("token_usage", {}).get("output_tokens") if result.get("token_usage") else None,
                cost=result.get("cost"),
                routing_decision=result.get("routing_decision"),
                failure_type=None if result.get("success") else result.get("status"),
                seed=config.seed,
            )

            self.trials.append(trial)
            self.total_trials += 1
            if trial.success:
                self.successful_trials += 1
            else:
                self.failed_trials += 1

            # Save trial summary
            trial_path = destination / "trial_summary.json"
            trial_path.write_text(json.dumps(self._trial_to_dict(trial), indent=2) + "\n", encoding="utf-8")

            return trial

    def _verify_task(self, task_id: str, workspace: SandboxWorkspace, trajectory: dict) -> bool:
        from divy.live_tasks import verify_task
        return verify_task(task_id, workspace, trajectory)

    def _trial_to_dict(self, trial: TrialResult) -> dict:
        return {
            "task_id": trial.task_id,
            "model_id": trial.model_id,
            "strategy": trial.strategy,
            "trial_num": trial.trial_num,
            "success": trial.success,
            "status": trial.status,
            "verification_passed": trial.verification_passed,
            "model_calls": trial.model_calls,
            "tool_calls": trial.tool_calls,
            "recovery_attempts": trial.recovery_attempts,
            "latency_ms": trial.latency_ms,
            "total_tokens": trial.total_tokens,
            "input_tokens": trial.input_tokens,
            "output_tokens": trial.output_tokens,
            "cost": trial.cost,
            "routing_decision": trial.routing_decision,
            "failure_type": trial.failure_type,
            "seed": trial.seed,
        }

    def run_campaign(
        self,
        task_ids: list[str] | None = None,
        strategies: list[str] | None = None,
        trials_per_config: int = 3,
    ) -> dict[str, Any]:
        """Run a full evaluation campaign."""
        task_ids = task_ids or list(LIVE_TASKS.keys())
        strategies = strategies or ["single", "fixed", "dynamic"]

        all_results = []

        for task_id in task_ids:
            task = LIVE_TASKS[task_id]
            model_ids = self.base_model_ids

            for strategy in strategies:
                if strategy == "single":
                    model_ids_for_strategy = [model_ids[0]]
                elif strategy == "fixed":
                    model_ids_for_strategy = [model_ids[1]] if len(model_ids) > 1 else model_ids[:1]
                elif strategy == "dynamic":
                    model_ids_for_strategy = model_ids
                else:
                    continue

                config = ExperimentConfig(
                    task_id=task_id,
                    model_ids=model_ids_for_strategy,
                    strategy=strategy,
                    live=self.live,
                )

                for trial_num in range(1, trials_per_config + 1):
                    print(f"Running {task_id} / {strategy} / trial {trial_num}...")
                    trial = self.run_trial(config, trial_num)
                    all_results.append(self._trial_to_dict(trial))

        # Save campaign results
        campaign_result = {
            "live": self.live,
            "total_trials": self.total_trials,
            "successful": self.successful_trials,
            "failed": self.failed_trials,
            "trials": all_results,
            "summary": self._compute_summary(),
        }

        campaign_path = self.output_dir / "campaign_results.json"
        campaign_path.write_text(json.dumps(campaign_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        return campaign_result

    def _compute_summary(self) -> dict[str, Any]:
        """Compute summary statistics across all trials."""
        if not self.trials:
            return {}

        by_strategy = {}
        by_task = {}
        by_model = {}

        for trial in self.trials:
            # By strategy
            s = by_strategy.setdefault(trial.strategy, {"total": 0, "success": 0, "latencies": [], "tokens": [], "costs": []})
            s["total"] += 1
            if trial.success:
                s["success"] += 1
            if trial.latency_ms:
                s["latencies"].append(trial.latency_ms)
            if trial.total_tokens:
                s["tokens"].append(trial.total_tokens)
            if trial.cost is not None:
                s["costs"].append(trial.cost)

            # By task
            t = by_task.setdefault(trial.task_id, {"total": 0, "success": 0})
            t["total"] += 1
            if trial.success:
                t["success"] += 1

            # By model
            m = by_model.setdefault(trial.model_id, {"total": 0, "success": 0})
            m["total"] += 1
            if trial.success:
                m["success"] += 1

        # Compute aggregates
        for s in by_strategy.values():
            s["success_rate"] = s["success"] / s["total"] if s["total"] > 0 else 0
            s["avg_latency_ms"] = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else None
            s["avg_tokens"] = sum(s["tokens"]) / len(s["tokens"]) if s["tokens"] else None
            s["avg_cost"] = sum(s["costs"]) / len(s["costs"]) if s["costs"] else None

        for t in by_task.values():
            t["success_rate"] = t["success"] / t["total"] if t["total"] > 0 else 0

        for m in by_model.values():
            m["success_rate"] = m["success"] / m["total"] if m["total"] > 0 else 0

        return {
            "by_strategy": by_strategy,
            "by_task": by_task,
            "by_model": by_model,
        }


def run_evaluation_campaign(
    output_dir: Path | str,
    live: bool = True,
    model_ids: list[str] | None = None,
    task_ids: list[str] | None = None,
    strategies: list[str] | None = None,
    trials_per_config: int = 3,
    seed: int | None = None,
) -> dict[str, Any]:
    """Convenience function to run a full evaluation campaign."""
    runner = ExperimentRunner(output_dir, live=live, base_model_ids=model_ids, seed=seed)
    return runner.run_campaign(task_ids=task_ids, strategies=strategies, trials_per_config=trials_per_config)


if __name__ == "__main__":
    import sys
    import os

    live = os.environ.get("DIVY_RUN_LIVE", "0") == "1"
    output_dir = Path("results") / "evaluation_campaign"

    result = run_evaluation_campaign(
        output_dir=output_dir,
        live=live,
        task_ids=["RW01"],
        strategies=["single", "fixed", "dynamic"],
        trials_per_config=1,
        seed=42,
    )

    print(json.dumps(result["summary"], indent=2))
    sys.exit(0 if result["summary"].get("by_strategy", {}).get("dynamic", {}).get("success_rate", 0) >= 0.5 else 1)
