"""Campaign infrastructure for controlled RW01-RW05 evaluation.

Runs each task under 4 execution modes:
  * scripted      - deterministic adapter, no real provider
  * single_model  - first live model only, no routing/fallback
  * fixed_routing - hand-picked live model per task, no adaptive recovery
  * dynamic       - full DIVY routing policy with fallback/verification

All runs record raw measurements without inventing composite scores.
Scripted and live results are completely separated.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent_loop import AgentLimits, AgentSession
from .evaluation import (
    EvaluationReport,
    ExecutionMode,
    evaluate_execution,
)
from .dataset import apply_quality_gate, example_from_execution
from .live_tasks import (
    LIVE_TASKS,
    build_registry,
    setup_fixture,
    verify_task,
)
from .logger import EventLogger
from .model import ModelResponse
from .nvidia import nvidia_registry_from_environment
from .performance import PerformanceMemory
from .providers import ModelMetadata, ModelProvider, ModelRegistry
from .routing import DeterministicRoutingPolicy, RoutingDecision, RoutingRequest
from .schemas import EvaluationResult
from .tools.workspace import SandboxWorkspace


# ============================================================================
# Scripted adapters for RW01-RW05
# ============================================================================

class _RW01ScriptedAdapter:
    """Deterministic adapter for RW01: bug fix + test verification."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "src/calculator.py"}}}
        elif not any(o.get("tool") == "fs.patch" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.patch", "arguments": {"path": "src/calculator.py", "old": "return a - b", "new": "return a + b"}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "test_add"}}}
        elif not any(o.get("tool") == "fs.write" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.write", "arguments": {"path": "CHANGELOG.md", "content": "# Change note\n\nRepaired add to return the sum of its inputs.\n"}}}
        else:
            content = {"final": {"message": "Completed the repair, test, and changelog."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


class _RW02ScriptedAdapter:
    """Deterministic adapter for RW02: factorial bug fix."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "src/buggy.py"}}}
        elif not any(o.get("tool") == "fs.patch" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.patch", "arguments": {"path": "src/buggy.py", "old": "return n * factorial(n)", "new": "return n * factorial(n-1)"}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "factorial"}}}
        else:
            content = {"final": {"message": "Completed the factorial repair and test."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


class _RW03ScriptedAdapter:
    """Deterministic adapter for RW03: API implementation."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "spec.md"}}}
        elif not any(o.get("tool") == "fs.write" and o.get("success") for o in observations):
            impl = '''def list_users():
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

def create_user(data):
    return {"id": 3, **data}

def get_user(id):
    users = list_users()
    for u in users:
        if u["id"] == id:
            return u
    return None
'''
            content = {"action": {"tool": "fs.write", "arguments": {"path": "impl.py", "content": impl}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "api_impl"}}}
        else:
            content = {"final": {"message": "Implemented all API endpoints and verified."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


class _RW04ScriptedAdapter:
    """Deterministic adapter for RW04: word count."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "data/input.txt"}}}
        elif not any(o.get("tool") == "fs.write" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.write", "arguments": {"path": "output.txt", "content": "hello: 2\nworld: 2\ndivy: 1\n"}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "wordcount"}}}
        else:
            content = {"final": {"message": "Completed word count and verification."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


class _RW06ScriptedAdapter:
    """Deterministic adapter for RW06: mutable default argument bug fix."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "src/cache.py"}}}
        elif not any(o.get("tool") == "fs.patch" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.patch", "arguments": {"path": "src/cache.py", "old": "def get_cached_value(key, cache={}):", "new": "def get_cached_value(key, cache=None):\n    if cache is None:\n        cache = {}"}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "cache_isolation"}}}
        else:
            content = {"final": {"message": "Fixed mutable default argument bug and verified."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


class _RW07ScriptedAdapter:
    """Deterministic adapter for RW07: find and fix correct processor."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "spec.md"}}}
        elif not any(o.get("tool") == "fs.read" and o.get("success") and o.get("result", {}).get("path") == "src/processor_b.py" for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "src/processor_b.py"}}}
        elif not any(o.get("tool") == "fs.patch" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.patch", "arguments": {"path": "src/processor_b.py", "old": "return data.lower()  # Bug: should be upper()", "new": "return data.upper()"}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "processor"}}}
        else:
            content = {"final": {"message": "Fixed processor_b to convert to uppercase and verified."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


class _RW08ScriptedAdapter:
    """Deterministic adapter for RW08: test diagnosis - fix validator code to accept user@domain."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "src/validator.py"}}}
        elif not any(o.get("tool") == "fs.patch" and o.get("success") for o in observations):
            # Fix the validator to accept user@domain (domain without dot is valid per test)
            content = {"action": {"tool": "fs.patch", "arguments": {"path": "src/validator.py", "old": "if \".\" not in domain:\n        return False", "new": "# Allow domains without dots (e.g., user@domain)\n    # if \".\" not in domain:\n    #     return False"}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "validator"}}}
        else:
            content = {"final": {"message": "Fixed validator to accept user@domain, test passes."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


class _RW09ScriptedAdapter:
    """Deterministic adapter for RW09: multi-step config-driven processor."""

    def __init__(self, workspace_root: str | None = None) -> None:
        self._step = 0
        self.workspace_root = workspace_root

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") and o.get("result", {}).get("path") == "config.json" for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "config.json"}}}
        elif not any(o.get("tool") == "fs.read" and o.get("success") and o.get("result", {}).get("path") == "input.txt" for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "input.txt"}}}
        elif not any(o.get("tool") == "fs.write" and o.get("success") and o.get("result", {}).get("path") == "data_processor.py" for o in observations):
            if self.workspace_root:
                # Embed the workspace root path in the generated code
                impl = f'''import os
import json

def process_file(input_file, output_file):
    # Use the known workspace root path
    base_dir = r"{self.workspace_root}"
    input_path = os.path.join(base_dir, input_file)
    output_path = os.path.join(base_dir, output_file)
    with open(input_path, "r") as f:
        lines = f.readlines()
    reversed_lines = [line.rstrip("\\n")[::-1] + "\\n" for line in lines]
    with open(output_path, "w") as f:
        f.writelines(reversed_lines)
    return len(lines)
'''
            else:
                # Fallback: try to find workspace root by looking for config.json
                impl = '''import os
import json

def find_workspace_root():
    current = os.getcwd()
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "config.json")):
            return current
        current = os.path.dirname(current)
    return os.getcwd()

def process_file(input_file, output_file):
    base_dir = find_workspace_root()
    input_path = os.path.join(base_dir, input_file)
    output_path = os.path.join(base_dir, output_file)
    with open(input_path, "r") as f:
        lines = f.readlines()
    reversed_lines = [line.rstrip("\\n")[::-1] + "\\n" for line in lines]
    with open(output_path, "w") as f:
        f.writelines(reversed_lines)
    return len(lines)
'''
            content = {"action": {"tool": "fs.write", "arguments": {"path": "data_processor.py", "content": impl}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "processor_module"}}}
        else:
            content = {"final": {"message": "Created data_processor module, processed file, and verified."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})



class _RW05ScriptedAdapter:
    """Deterministic adapter for RW05: statistics computation."""

    def __init__(self) -> None:
        self._step = 0

    def generate(self, messages, tools, response_schema):
        observations = [m.get("content", {}) for m in messages if m.get("role") == "tool"]
        self._step += 1
        if not any(o.get("tool") == "fs.read" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.read", "arguments": {"path": "data/numbers.json"}}}
        elif not any(o.get("tool") == "fs.write" and o.get("success") for o in observations):
            content = {"action": {"tool": "fs.write", "arguments": {"path": "output.json", "content": '{"sum": 55, "avg": 5.5, "max": 10}'}}}
        elif not any(o.get("tool") == "test.run" and o.get("success") for o in observations):
            content = {"action": {"tool": "test.run", "arguments": {"test_id": "stats"}}}
        else:
            content = {"final": {"message": "Computed statistics and verified."}}

        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})


SCRIPTED_ADAPTERS = {
    "RW01": _RW01ScriptedAdapter,
    "RW02": _RW02ScriptedAdapter,
    "RW03": _RW03ScriptedAdapter,
    "RW04": _RW04ScriptedAdapter,
    "RW05": _RW05ScriptedAdapter,
    "RW06": _RW06ScriptedAdapter,
    "RW07": _RW07ScriptedAdapter,
    "RW08": _RW08ScriptedAdapter,
    "RW09": _RW09ScriptedAdapter,
}


def get_scripted_adapter(task_id: str, workspace_root: str | None = None):
    """Get the scripted adapter class for a task."""
    adapter_class = SCRIPTED_ADAPTERS.get(task_id)
    if not adapter_class:
        raise ValueError(f"No scripted adapter for task: {task_id}")
    if task_id == "RW09" and workspace_root is not None:
        return adapter_class(workspace_root)
    return adapter_class()


# ============================================================================
# Fixed routing table for live evaluation
# ============================================================================

# Map task_id -> preferred model index (0-based from NVIDIA_MODEL_SPECS)
FIXED_ROUTING_TABLE = {
    "RW01": 0,  # Use first model
    "RW02": 1,  # Use second model
    "RW03": 2,  # Use third model
    "RW04": 0,  # Use first model
    "RW05": 1,  # Use second model
    "RW06": 1,  # Use second model (reasoning for mutable default arg)
    "RW07": 2,  # Use third model (complex navigation)
    "RW08": 1,  # Use second model (test diagnosis)
    "RW09": 2,  # Use third model (multi-step)
}


def _sum_optional(values: list[Any]) -> float | int | None:
    """Sum values, returning None if any value is missing (None or non-numeric).

    This ensures that partially missing metrics don't produce partial totals.
    """
    for value in values:
        if not isinstance(value, (int, float)):
            return None
    return sum(values) if values else None


# ============================================================================
# Campaign configuration and results
# ============================================================================

@dataclass(frozen=True)
class CampaignConfig:
    """Configuration for a single campaign run."""
    task_id: str
    execution_mode: str  # "scripted", "single_model", "fixed_routing", "dynamic"
    model_id: str | None = None  # For single_model/fixed_routing
    live: bool = True
    max_cost: float | None = None
    max_iterations: int = 12
    max_tool_calls: int = 12
    max_recovery: int = 2
    max_fallbacks: int = 1
    seed: int | None = None


@dataclass
class CampaignTrialResult:
    """Result of a single campaign trial."""
    task_id: str
    execution_mode: str
    model_id: str
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
    evaluation_report: EvaluationReport | None = None
    trajectory: dict[str, Any] | None = None
    seed: int | None = None
    fallback_used: bool = False
    attempted_model_ids: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    # Additional fields for complete trial recording
    task_type: str | None = None
    false_acceptance: bool = False
    unnecessary_actions: int = 0
    safety_violations: int = 0
    failure_type: str | None = None
    routing_evidence: dict[str, Any] | None = None
    performance_observation_count: int = 0


# ============================================================================
# Campaign runner
# ============================================================================

class CampaignRunner:
    """Runs controlled evaluation campaigns across tasks and execution modes."""

    def __init__(
        self,
        output_dir: Path | str,
        live: bool = True,
        base_model_ids: list[str] | None = None,
        seed: int | None = None,
        model_registry_factory: Callable[[list[str]], ModelRegistry] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.live = live
        self.base_model_ids = base_model_ids
        self.seed = seed
        self.model_registry_factory = model_registry_factory or nvidia_registry_from_environment

        # Separate performance memories for scripted vs live
        self.scripted_memory = PerformanceMemory(self.output_dir / "performance_memory_scripted.jsonl")
        self.live_memory = PerformanceMemory(self.output_dir / "performance_memory_live.jsonl")

        # Results storage
        self.trials: list[CampaignTrialResult] = []
        self.total_trials = 0
        self.successful_trials = 0
        self.failed_trials = 0

    def run_trial(self, config: CampaignConfig, trial_num: int) -> CampaignTrialResult:
        """Run a single trial with the given configuration."""
        if config.seed is not None:
            import random
            random.seed(config.seed)

        task = LIVE_TASKS[config.task_id]
        is_scripted = config.execution_mode == "scripted"

        with tempfile.TemporaryDirectory(prefix=f"divy-campaign-{config.task_id}-{config.execution_mode}-") as temporary:
            workspace = SandboxWorkspace(Path(temporary))
            setup_fixture(config.task_id, workspace)
            registry = build_registry(workspace, config.task_id)

            destination = self.output_dir / "trials" / f"{config.task_id}_{config.execution_mode}_t{trial_num}"
            # Refuse to mix two executions under one trial identity. Warm
            # starts must use a new run directory with an explicit history seed.
            destination.mkdir(parents=True, exist_ok=False)

            logger = EventLogger(destination / "events.jsonl")
            memory = self.scripted_memory if is_scripted else self.live_memory
            pre_run_history: dict[str, dict[str, Any]] = {}

            # Build model registry based on execution mode
            if is_scripted:
                adapter = get_scripted_adapter(config.task_id, workspace_root=temporary if config.task_id == "RW09" else None)
                metadata = ModelMetadata(
                    f"scripted-{config.task_id}",
                    "deterministic",
                    task.required_capabilities,
                    4096, 0.0, 1.0, 1.0
                )
                provider = ModelProvider(metadata, adapter)
                registry_for_routing = ModelRegistry([provider])
                decision = RoutingDecision(config.task_id, [provider.model_id])

            elif config.execution_mode == "single_model":
                if not self.base_model_ids:
                    raise ValueError("base_model_ids required for live modes")
                model_id = config.model_id or self.base_model_ids[0]
                registry_for_routing = self.model_registry_factory([model_id])
                decision = RoutingDecision(
                    config.task_id,
                    [model_id],
                    candidate_model_ids=[model_id],
                    rationale=f"single_model selected={model_id}",
                )

            elif config.execution_mode == "fixed_routing":
                if not self.base_model_ids:
                    raise ValueError("base_model_ids required for live modes")
                idx = FIXED_ROUTING_TABLE.get(config.task_id, 0)
                model_id = config.model_id or self.base_model_ids[idx]
                registry_for_routing = self.model_registry_factory([model_id])
                decision = RoutingDecision(
                    config.task_id,
                    [model_id],
                    candidate_model_ids=[model_id],
                    rationale=f"fixed_routing selected={model_id}",
                )

            elif config.execution_mode == "dynamic":
                if not self.base_model_ids:
                    raise ValueError("base_model_ids required for live modes")
                registry_for_routing = self.model_registry_factory(self.base_model_ids)
                req = RoutingRequest(
                    task_id=config.task_id,
                    required_capabilities=task.required_capabilities,
                    task_type=task.task_type,
                    complexity=task.complexity,
                )
                policy = DeterministicRoutingPolicy(
                    performance_memory=memory,
                    empirical_weight=0.3,
                    confidence_samples=10.0,
                    recency_half_life_days=30.0,
                    recency_weight=0.3,
                    exploration_rate=0.1,
                )
                decision = policy.route(req, registry_for_routing)

            else:
                raise ValueError(f"Unknown execution mode: {config.execution_mode}")

            if not decision.selected_model_ids:
                return CampaignTrialResult(
                    task_id=config.task_id,
                    execution_mode=config.execution_mode,
                    model_id="none",
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
                    seed=config.seed,
                    task_type=task.task_type,
                    failure_type="no_capable_model",
                )

            selected_model_id = decision.selected_model_ids[0]
            selected_provider = registry_for_routing.get(selected_model_id)
            missing_capabilities = set(task.required_capabilities) - set(selected_provider.capabilities())
            if missing_capabilities:
                return CampaignTrialResult(
                    task_id=config.task_id,
                    execution_mode=config.execution_mode,
                    model_id=selected_model_id,
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
                    routing_decision={
                        "selected_model_ids": decision.selected_model_ids,
                        "candidate_model_ids": decision.candidate_model_ids,
                        "rationale": f"missing_capabilities={sorted(missing_capabilities)}",
                    },
                    seed=config.seed,
                    task_type=task.task_type,
                    failure_type="missing_capabilities",
                )
            for candidate in registry_for_routing.list():
                pre_run_history[candidate.model_id] = memory.summarize(
                    model_id=candidate.model_id,
                    task_type=task.task_type,
                )

            # Run one bounded session per model attempt. Only dynamic routing
            # may use fallback candidates; single/fixed modes deliberately do not.
            attempt_model_ids = [selected_model_id]
            if config.execution_mode == "dynamic" and config.max_fallbacks > 0:
                for fallback_id in decision.fallback_model_ids:
                    if fallback_id == selected_model_id or fallback_id in attempt_model_ids:
                        continue
                    fallback_provider = registry_for_routing.get(fallback_id)
                    if set(task.required_capabilities) <= set(fallback_provider.capabilities()):
                        attempt_model_ids.append(fallback_id)
                    if len(attempt_model_ids) > config.max_fallbacks:
                        break

            attempts: list[dict[str, Any]] = []
            attempt_trajectories: list[dict[str, Any]] = []
            all_events: list[dict[str, Any]] = []
            remaining_cost = config.max_cost

            for attempt_no, attempt_model_id in enumerate(attempt_model_ids, start=1):
                attempt_provider = registry_for_routing.get(attempt_model_id)
                if attempt_no == 1:
                    attempt_workspace = workspace
                    attempt_registry = registry
                else:
                    attempt_workspace = SandboxWorkspace(Path(temporary) / f"attempt_{attempt_no}_workspace")
                    setup_fixture(config.task_id, attempt_workspace)
                    attempt_registry = build_registry(attempt_workspace, config.task_id)

                attempt_dir = destination / f"attempt_{attempt_no}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                logger.append({
                    "event": "attempt_start",
                    "attempt": attempt_no,
                    "model_id": attempt_model_id,
                    "is_fallback": attempt_no > 1,
                })

                def verifier(traj: dict[str, Any], current_workspace: SandboxWorkspace = attempt_workspace) -> EvaluationResult:
                    verified = verify_task(config.task_id, current_workspace, traj)
                    return EvaluationResult(
                        success=verified,
                        verification_passed=verified,
                        evidence={},
                        recoverable_failure=not verified,
                    )

                session = AgentSession(
                    attempt_provider,
                    attempt_registry,
                    logger=logger,
                    performance_memory=memory,
                    limits=AgentLimits(
                        max_iterations=config.max_iterations,
                        max_tool_calls=config.max_tool_calls,
                        max_recovery_attempts=config.max_recovery,
                        max_cost=remaining_cost,
                    ),
                )

                session_result = session.run(
                    task.goal,
                    task_id=task.task_id,
                    task_type=task.task_type,
                    capabilities=set(task.required_capabilities),
                    system_instructions="You are DIVY's bounded agent. Use only registered tools. Return {action:{tool,arguments}} or {final:{message}}. After making changes, you MUST run the registered test to verify your fix using the test.run tool with the appropriate test_id.",
                    response_schema={"type": "object"},
                    verifier=verifier,
                )

                attempt_trajectory = session_result.trajectory
                attempt_verified = verify_task(config.task_id, attempt_workspace, attempt_trajectory)
                attempt_success = attempt_verified  # Independent verification is authoritative for task success
                attempt_tokens = attempt_trajectory.get("token_usage") or {}

                # Annotate events so the composite evaluation can attribute
                # each model call and verification outcome to its attempt.
                for event in attempt_trajectory.get("events", []):
                    event["attempt"] = attempt_no
                    if event.get("event") == "model_response":
                        event.setdefault("model_id", attempt_model_id)
                        event.setdefault("provider", attempt_provider.metadata.provider)
                        event["attempt_success"] = attempt_success
                        event["attempt_verification_passed"] = attempt_verified

                attempt_summary = {
                    "attempt": attempt_no,
                    "model_id": attempt_model_id,
                    "provider": attempt_provider.metadata.provider,
                    "is_fallback": attempt_no > 1,
                    "status": attempt_trajectory.get("status"),
                    "success": attempt_success,
                    "verification_passed": attempt_verified,
                    "model_calls": attempt_trajectory.get("model_calls", 0),
                    "tool_calls": attempt_trajectory.get("tool_calls", 0),
                    "recovery_attempts": attempt_trajectory.get("recovery_attempts", 0),
                    "latency_ms": attempt_trajectory.get("latency_ms"),
                    "token_usage": attempt_tokens,
                    "cost": attempt_trajectory.get("cost"),
                    "execution_id": attempt_trajectory.get("execution_id"),
                }
                attempts.append(attempt_summary)
                attempt_trajectories.append(attempt_trajectory)
                all_events.extend(attempt_trajectory.get("events", []))

                with (attempt_dir / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(attempt_trajectory, sort_keys=True) + "\n")
                with (attempt_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                    for event in attempt_trajectory.get("events", []):
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                (attempt_dir / "attempt_summary.json").write_text(
                    json.dumps(attempt_summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                if remaining_cost is not None and attempt_summary["cost"] is not None:
                    remaining_cost = max(0.0, remaining_cost - attempt_summary["cost"])
                if attempt_verified:
                    break

            # Aggregate raw measurements across attempts. No composite score
            # is computed here; only observable counts, latency, tokens, cost.
            aggregate = {
                "model_calls": sum(item["model_calls"] for item in attempts),
                "tool_calls": sum(item["tool_calls"] for item in attempts),
                "recovery_attempts": sum(item["recovery_attempts"] for item in attempts),
                "latency_ms": _sum_optional([item["latency_ms"] for item in attempts]),
                "cost": (
                    None
                    if any(item["cost"] is None for item in attempts)
                    else round(sum(item["cost"] for item in attempts), 8)
                ),
                "token_usage": {},
                "total_tokens": _sum_optional([item["token_usage"].get("total_tokens") for item in attempts]),
                "input_tokens": _sum_optional([item["token_usage"].get("input_tokens") for item in attempts]),
                "output_tokens": _sum_optional([item["token_usage"].get("output_tokens") for item in attempts]),
            }
            for key in ("total_tokens", "input_tokens", "output_tokens"):
                if aggregate[key] is not None:
                    aggregate["token_usage"][key] = aggregate[key]

            evaluation_passed = any(item["verification_passed"] for item in attempts)
            final_trajectory = dict(attempt_trajectories[-1])
            final_trajectory["events"] = all_events
            final_trajectory["model_id"] = selected_model_id
            final_trajectory["provider"] = selected_provider.metadata.provider
            final_trajectory["model_calls"] = aggregate["model_calls"]
            final_trajectory["tool_calls"] = aggregate["tool_calls"]
            final_trajectory["recovery_attempts"] = aggregate["recovery_attempts"]
            final_trajectory["latency_ms"] = aggregate["latency_ms"]
            final_trajectory["token_usage"] = aggregate["token_usage"]
            final_trajectory["cost"] = aggregate["cost"]
            final_trajectory["status"] = (
                "success"
                if evaluation_passed
                else (
                    "failed_verification"
                    if final_trajectory.get("status") == "success"
                    else final_trajectory.get("status", "failed")
                )
            )
            final_trajectory["verification_result"] = evaluation_passed
            final_trajectory["fallback_used"] = len(attempts) > 1
            final_trajectory["attempts"] = attempts
            final_trajectory["attempt_execution_ids"] = [item["execution_id"] for item in attempts]
            trajectory = final_trajectory

            # Build routing decision dict
            routing_decision = {
                "selected_model_ids": decision.selected_model_ids,
                "fallback_model_ids": decision.fallback_model_ids,
                "verifier_model_id": decision.verifier_model_id,
                "candidate_model_ids": decision.candidate_model_ids,
                "rationale": decision.rationale,
                "fallback_used": len(attempts) > 1,
                "attempted_model_ids": [item["model_id"] for item in attempts],
            }

            independent_evaluation = EvaluationResult(
                success=evaluation_passed,
                verification_passed=evaluation_passed,
                evidence={},
                recoverable_failure=not evaluation_passed,
            )

            # Create evaluation report
            eval_report = evaluate_execution(
                trajectory,
                evaluation_result=independent_evaluation,
                routing_decision=routing_decision,
                performance_memory=memory,
                execution_mode=ExecutionMode.live() if not is_scripted else ExecutionMode.scripted(),
                pre_run_history=pre_run_history,
            )

            dataset_example = example_from_execution(
                task_id=task.task_id,
                task_type=task.task_type,
                required_capabilities=sorted(task.required_capabilities),
                complexity=task.complexity,
                available_models=(
                    routing_decision["candidate_model_ids"]
                    or routing_decision["selected_model_ids"] + routing_decision["fallback_model_ids"]
                ),
                model_capabilities={
                    item.model_id: sorted(item.capabilities())
                    for item in registry_for_routing.list()
                },
                historical_evidence=pre_run_history,
                routing_decision=routing_decision,
                selected_model_id=selected_model_id,
                execution_success=trajectory.get("status") == "success",
                verification_passed=evaluation_passed,
                final_outcome=trajectory.get("status"),
                latency_ms=trajectory.get("latency_ms"),
                estimated_cost=trajectory.get("cost"),
                model_calls=trajectory.get("model_calls", 0),
                tool_calls=trajectory.get("tool_calls", 0),
                recovery_attempts=trajectory.get("recovery_attempts", 0),
            )

            # Save trajectory
            traj_path = destination / "trajectory.jsonl"
            with traj_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trajectory, sort_keys=True) + "\n")

            # Save evaluation report
            eval_path = destination / "evaluation_report.json"
            eval_path.write_text(json.dumps(eval_report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            dataset_path = destination / "router_dataset_example.json"
            apply_quality_gate(dataset_example)
            dataset_path.write_text(json.dumps(dataset_example.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

            # Build trial result
            # Compute additional quality metrics from evaluation report
            false_acceptance = eval_report.quality.false_acceptance if eval_report else False
            unnecessary_actions = eval_report.quality.unnecessary_actions if eval_report else 0
            safety_violations = sum(
                1 for event in trajectory.get("events", [])
                if event.get("metadata", {}).get("safety_violation")
            )
            # Failure type
            failure_type = None
            if not trajectory.get("status") == "success":
                failure_type = trajectory.get("status")
            # Routing evidence snapshot (pre-run history for selected model)
            routing_evidence = pre_run_history.get(selected_model_id, {}) if selected_model_id else {}
            # Count performance observations for this trial
            perf_obs_count = 0
            if memory and hasattr(memory, '_observations'):
                trial_execution_ids = set(item["execution_id"] for item in attempts)
                perf_obs_count = sum(1 for obs in memory._observations if obs.execution_id in trial_execution_ids)

            trial = CampaignTrialResult(
                task_id=config.task_id,
                execution_mode=config.execution_mode,
                model_id=selected_model_id,
                trial_num=trial_num,
                success=trajectory.get("status") == "success",
                status=trajectory.get("status", "unknown"),
                verification_passed=evaluation_passed,
                model_calls=trajectory.get("model_calls", 0),
                tool_calls=trajectory.get("tool_calls", 0),
                recovery_attempts=trajectory.get("recovery_attempts", 0),
                latency_ms=trajectory.get("latency_ms"),
                total_tokens=trajectory.get("token_usage", {}).get("total_tokens") if trajectory.get("token_usage") else None,
                input_tokens=trajectory.get("token_usage", {}).get("input_tokens") if trajectory.get("token_usage") else None,
                output_tokens=trajectory.get("token_usage", {}).get("output_tokens") if trajectory.get("token_usage") else None,
                cost=trajectory.get("cost"),
                routing_decision=routing_decision,
                evaluation_report=eval_report,
                trajectory=trajectory,
                seed=config.seed,
                fallback_used=len(attempts) > 1,
                attempted_model_ids=[item["model_id"] for item in attempts],
                attempts=attempts,
                task_type=task.task_type,
                false_acceptance=false_acceptance,
                unnecessary_actions=unnecessary_actions,
                safety_violations=safety_violations,
                failure_type=failure_type,
                routing_evidence=routing_evidence,
                performance_observation_count=perf_obs_count,
            )

            self.trials.append(trial)
            self.total_trials += 1
            if trial.success:
                self.successful_trials += 1
            else:
                self.failed_trials += 1

            # Save trial summary
            trial_path = destination / "trial_summary.json"
            trial_path.write_text(json.dumps(self._trial_to_dict(trial), indent=2, sort_keys=True) + "\n", encoding="utf-8")

            return trial

    def _trial_to_dict(self, trial: CampaignTrialResult) -> dict:
        return {
            "task_id": trial.task_id,
            "execution_mode": trial.execution_mode,
            "model_id": trial.model_id,
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
            "seed": trial.seed,
            "fallback_used": trial.fallback_used,
            "attempted_model_ids": trial.attempted_model_ids,
            "attempts": trial.attempts,
            "task_type": trial.task_type,
            "false_acceptance": trial.false_acceptance,
            "unnecessary_actions": trial.unnecessary_actions,
            "safety_violations": trial.safety_violations,
            "failure_type": trial.failure_type,
            "routing_evidence": trial.routing_evidence,
            "performance_observation_count": trial.performance_observation_count,
        }

    def run_campaign(
        self,
        task_ids: list[str] | None = None,
        execution_modes: list[str] | None = None,
        trials_per_config: int = 3,
    ) -> dict[str, Any]:
        """Run a full evaluation campaign.

        Scripted and live batches are executed, summarized, and written to
        separate files. They are never aggregated into one mixed ranking.
        """
        task_ids = task_ids or list(LIVE_TASKS.keys())
        execution_modes = execution_modes or ["scripted", "single_model", "fixed_routing", "dynamic"]
        scripted_modes = [mode for mode in execution_modes if mode == "scripted"]
        live_modes = [mode for mode in execution_modes if mode != "scripted"]
        if live_modes and not self.base_model_ids:
            raise ValueError("base_model_ids required for live modes")

        # Reset per-campaign result counters. Performance memories persist so
        # repeated campaigns can accumulate evidence deliberately.
        self.trials = []
        self.total_trials = 0
        self.successful_trials = 0
        self.failed_trials = 0

        batches: dict[str, dict[str, Any]] = {}
        if scripted_modes:
            batches["scripted"] = self._run_batch(task_ids, scripted_modes, trials_per_config)
        if live_modes:
            batches["live"] = self._run_batch(task_ids, live_modes, trials_per_config)

        for kind, batch in batches.items():
            path = self.output_dir / f"campaign_results_{kind}.json"
            payload = {"kind": kind, **batch, "results_file": path.name}
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            batch["results_file"] = path.name

        manifest = {
            "live": "live" in batches,
            "kinds": {
                kind: {
                    "total_trials": batch["total_trials"],
                    "successful": batch["successful"],
                    "failed": batch["failed"],
                    "summary": batch["summary"],
                    "results_file": batch["results_file"],
                }
                for kind, batch in batches.items()
            },
            "total_trials": sum(batch["total_trials"] for batch in batches.values()),
            "successful": sum(batch["successful"] for batch in batches.values()),
            "failed": sum(batch["failed"] for batch in batches.values()),
        }
        (self.output_dir / "campaign_results.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Preserve the single-kind return shape used by existing callers, while
        # mixed runs return only the separated manifest plus raw trials.
        if len(batches) == 1:
            kind, batch = next(iter(batches.items()))
            result = dict(batch)
            result["kind"] = kind
            result["live"] = kind == "live"
            result["kinds"] = manifest["kinds"]
            return result
        return {
            **manifest,
            "trials": [trial for batch in batches.values() for trial in batch["trials"]],
        }

    def _run_batch(
        self,
        task_ids: list[str],
        execution_modes: list[str],
        trials_per_config: int,
    ) -> dict[str, Any]:
        batch_trials: list[CampaignTrialResult] = []
        for task_id in task_ids:
            for mode in execution_modes:
                config = CampaignConfig(
                    task_id=task_id,
                    execution_mode=mode,
                    live=mode != "scripted",
                    seed=self.seed,
                )
                for trial_num in range(1, trials_per_config + 1):
                    print(f"Running {task_id} / {mode} / trial {trial_num}...")
                    trial = self.run_trial(config, trial_num)
                    batch_trials.append(trial)

        return {
            "total_trials": len(batch_trials),
            "successful": sum(1 for trial in batch_trials if trial.success),
            "failed": sum(1 for trial in batch_trials if not trial.success),
            "trials": [self._trial_to_dict(trial) for trial in batch_trials],
            "summary": self._compute_summary(batch_trials),
        }

    def _compute_summary(self, trials: list[CampaignTrialResult] | None = None) -> dict[str, Any]:
        """Compute raw aggregate statistics across the given trials."""
        trials = list(self.trials) if trials is None else trials
        if not trials:
            return {}

        by_mode = {}
        by_task = {}
        by_model = {}
        by_attempt_model = {}

        for trial in trials:
            # By execution mode
            m = by_mode.setdefault(trial.execution_mode, {
                "total": 0,
                "success": 0,
                "verification_passed": 0,
                "fallback_used": 0,
                "model_calls": 0,
                "tool_calls": 0,
                "recovery_attempts": 0,
                "latencies": [],
                "tokens": [],
                "costs": [],
            })
            m["total"] += 1
            if trial.success:
                m["success"] += 1
            if trial.verification_passed:
                m["verification_passed"] += 1
            if trial.fallback_used:
                m["fallback_used"] += 1
            m["model_calls"] += trial.model_calls
            m["tool_calls"] += trial.tool_calls
            m["recovery_attempts"] += trial.recovery_attempts
            if trial.latency_ms:
                m["latencies"].append(trial.latency_ms)
            if trial.total_tokens:
                m["tokens"].append(trial.total_tokens)
            if trial.cost is not None:
                m["costs"].append(trial.cost)

            # By task
            t = by_task.setdefault(trial.task_id, {"total": 0, "success": 0, "verification_passed": 0})
            t["total"] += 1
            if trial.success:
                t["success"] += 1
            if trial.verification_passed:
                t["verification_passed"] += 1

            # By selected model (routing choice)
            mdl = by_model.setdefault(trial.model_id, {"total": 0, "success": 0})
            mdl["total"] += 1
            if trial.success:
                mdl["success"] += 1

            # By every model that actually executed (including fallbacks)
            for attempt in trial.attempts:
                am = by_attempt_model.setdefault(attempt["model_id"], {
                    "attempts": 0,
                    "success": 0,
                    "verification_passed": 0,
                    "model_calls": 0,
                    "tool_calls": 0,
                    "recovery_attempts": 0,
                })
                am["attempts"] += 1
                if attempt.get("success"):
                    am["success"] += 1
                if attempt.get("verification_passed"):
                    am["verification_passed"] += 1
                am["model_calls"] += attempt.get("model_calls", 0)
                am["tool_calls"] += attempt.get("tool_calls", 0)
                am["recovery_attempts"] += attempt.get("recovery_attempts", 0)

        # Compute aggregates
        for m in by_mode.values():
            m["success_rate"] = m["success"] / m["total"] if m["total"] > 0 else 0
            m["verification_pass_rate"] = m["verification_passed"] / m["total"] if m["total"] > 0 else 0
            m["fallback_rate"] = m["fallback_used"] / m["total"] if m["total"] > 0 else 0
            m["avg_latency_ms"] = sum(m["latencies"]) / len(m["latencies"]) if m["latencies"] else None
            m["avg_tokens"] = sum(m["tokens"]) / len(m["tokens"]) if m["tokens"] else None
            m["avg_cost"] = sum(m["costs"]) / len(m["costs"]) if m["costs"] else None

        for t in by_task.values():
            t["success_rate"] = t["success"] / t["total"] if t["total"] > 0 else 0
            t["verification_pass_rate"] = t["verification_passed"] / t["total"] if t["total"] > 0 else 0

        for m in by_model.values():
            m["success_rate"] = m["success"] / m["total"] if m["total"] > 0 else 0

        for m in by_attempt_model.values():
            m["success_rate"] = m["success"] / m["attempts"] if m["attempts"] > 0 else 0
            m["verification_pass_rate"] = m["verification_passed"] / m["attempts"] if m["attempts"] > 0 else 0

        return {
            "by_mode": by_mode,
            "by_task": by_task,
            "by_model": by_model,
            "by_attempt_model": by_attempt_model,
        }


def run_campaign(
    output_dir: Path | str,
    live: bool = True,
    model_ids: list[str] | None = None,
    task_ids: list[str] | None = None,
    execution_modes: list[str] | None = None,
    trials_per_config: int = 3,
    seed: int | None = None,
) -> dict[str, Any]:
    """Convenience function to run a full evaluation campaign."""
    runner = CampaignRunner(output_dir, live=live, base_model_ids=model_ids, seed=seed)
    return runner.run_campaign(task_ids=task_ids, execution_modes=execution_modes, trials_per_config=trials_per_config)


if __name__ == "__main__":
    import os
    import sys

    live = os.environ.get("DIVY_RUN_LIVE", "0") == "1"
    output_dir = Path(os.environ.get("DIVY_CAMPAIGN_OUTPUT_DIR", "results/campaign"))
    task_ids = [
        item.strip()
        for item in os.environ.get("DIVY_CAMPAIGN_TASKS", "RW01,RW02,RW03,RW04,RW05").split(",")
        if item.strip()
    ]
    trials_per_config = int(os.environ.get("DIVY_CAMPAIGN_TRIALS", "1"))

    execution_modes = ["single_model", "fixed_routing", "dynamic"] if live else ["scripted"]
    modes_text = os.environ.get("DIVY_CAMPAIGN_MODES")
    if modes_text:
        execution_modes = [item.strip() for item in modes_text.split(",") if item.strip()]

    live_modes = [mode for mode in execution_modes if mode != "scripted"]
    if live_modes and not live:
        print(
            "ERROR: live execution modes require DIVY_RUN_LIVE=1",
            file=sys.stderr,
        )
        sys.exit(1)

    model_ids = None
    if live_modes:
        model_ids_text = os.environ.get("DIVY_MODEL_IDS") or os.environ.get("DIVY_NVIDIA_MODEL_IDS")
        if not model_ids_text:
            print(
                "ERROR: DIVY_MODEL_IDS or DIVY_NVIDIA_MODEL_IDS required for live modes",
                file=sys.stderr,
            )
            sys.exit(1)
        model_ids = [item.strip() for item in model_ids_text.split(",") if item.strip()]

    result = run_campaign(
        output_dir=output_dir,
        live=live,
        model_ids=model_ids,
        task_ids=task_ids,
        execution_modes=execution_modes,
        trials_per_config=trials_per_config,
        seed=42,
    )

    if "kinds" in result:
        for kind, section in result["kinds"].items():
            print(f"=== {kind} ===")
            print(json.dumps(section["summary"], indent=2))
    else:
        print(json.dumps(result["summary"], indent=2))
    sys.exit(0)
