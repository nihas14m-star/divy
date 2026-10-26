"""Regression checks for the final audit; no hosted calls or real credentials."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from divy.adapters import HttpModelAdapter, ProviderError
from divy.agent_loop import AgentLimits, AgentSession
from divy.campaign import CampaignConfig, CampaignRunner
from divy.dataset import apply_quality_gate
from divy.evaluation import evaluate_execution
from divy.execution import MultiModelExecutor
from divy.performance import PerformanceMemory, PerformanceObservation, PerformanceInterface
from divy.providers import ModelMetadata, ModelProvider, ModelRegistry
from divy.routing import DeterministicRoutingPolicy, RoutingDecision, RoutingRequest
from divy.schemas import ModelResponse
from divy.tools import FilesystemTools, SandboxWorkspace, TestRunnerTool as RegisteredTests, ToolRegistry
from divy.experiment_runner import ExperimentRunner, ExperimentConfig
from tests.test_dataset import _base_example


class Responses:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, messages, tools, response_schema):
        self.calls += 1
        return next(self.responses)


def provider(adapter, name="model", cost=0.1):
    return ModelProvider(ModelMetadata(name, "test", frozenset({"coding"}), 4096, cost, 1, 0.5), adapter)


class FinalAuditTests(unittest.TestCase):
    def test_invalid_response_counted_once(self):
        adapter = Responses(ModelResponse({}))
        memory = PerformanceMemory()
        result = AgentSession(provider(adapter), ToolRegistry([]), performance_memory=memory).run(
            "test", task_id="audit", task_type="coding", capabilities={"coding"}, system_instructions=""
        )
        self.assertEqual(result.trajectory["model_calls"], 1)
        self.assertEqual(result.trajectory["cost"], 0.1)
        self.assertEqual(memory.count(), 1)
        self.assertIsNone(memory.observations()[0].token_count)

    def test_component_only_token_totals_accumulate(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            adapter = Responses(
                ModelResponse({"action": {"tool": "fs.list", "arguments": {}}}, {"output_tokens": 2}),
                ModelResponse({"final": {"message": "done"}}, {"output_tokens": 3}),
            )
            result = AgentSession(provider(adapter), ToolRegistry(FilesystemTools(workspace).definitions())).run(
                "test", task_id="audit", task_type="coding", capabilities={"coding"}, system_instructions=""
            )
        self.assertEqual(result.trajectory["token_usage"]["total_tokens"], 5)
        self.assertIsNone(result.trajectory["verification_result"])

    def test_batch_is_rejected_before_any_partial_execution(self):
        adapter = Responses(ModelResponse({"tool_calls": [
            {"tool": "fs.list", "arguments": {}}, {"tool": "fs.list", "arguments": {}}
        ]}))
        result = AgentSession(provider(adapter), ToolRegistry([])).run(
            "test", task_id="audit", task_type="coding", capabilities={"coding"}, system_instructions=""
        )
        self.assertEqual(result.status, "failed_model_response")
        self.assertEqual(result.trajectory["tool_calls"], 0)

    def test_tool_name_replay_matches_request_schema(self):
        adapter = HttpModelAdapter("https://example.invalid", "model")
        tools = adapter._normalize_tools([{"name": "fs.read", "parameters": {"type": "object"}}])
        messages = adapter._normalize_messages([
            {"role": "assistant", "content": {"action": {"tool": "fs.read", "arguments": {}, "tool_call_id": "id"}}},
            {"role": "tool", "tool_call_id": "id", "content": {"success": True}},
        ])
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], tools[0]["function"]["name"])
        self.assertEqual(messages[1]["tool_call_id"], "id")

    def test_tool_name_collision_is_rejected(self):
        adapter = HttpModelAdapter("https://example.invalid", "model")
        with self.assertRaises(ProviderError):
            adapter._normalize_tools([{"name": "fs.read"}, {"name": "fs_x2e_read"}])

    def test_malformed_batch_member_is_not_silently_dropped(self):
        adapter = HttpModelAdapter("https://example.invalid", "model")
        with self.assertRaises(ProviderError):
            adapter._extract_content({"choices": [{"message": {"tool_calls": [None]}}]})

    def test_unisolated_test_runner_blocked_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = RegisteredTests(SandboxWorkspace(Path(directory)), {})
            tools = ToolRegistry(runner.definitions()).schemas()
            adapter = HttpModelAdapter("https://example.invalid", "model", "test-only-key")
            with patch("urllib.request.urlopen") as http:
                with self.assertRaises(ProviderError) as error:
                    adapter.generate([], tools, {})
                self.assertEqual(error.exception.kind, "unsafe_execution")
                http.assert_not_called()

    def test_recency_does_not_remove_confidence_shrinkage(self):
        memory = PerformanceMemory()
        memory.record(PerformanceObservation("model", "coding", provider_success=True, contributed_to_final=True))
        p = provider(Responses())
        request = RoutingRequest("audit", frozenset({"coding"}), task_type="coding")
        static = DeterministicRoutingPolicy()._score(p, request)
        small = DeterministicRoutingPolicy(memory, empirical_weight=1, confidence_samples=10,
            recency_half_life_days=30, recency_weight=1)._score(p, request)
        self.assertAlmostEqual(small - static, 0.5 / 11)

    def test_pre_run_evidence_never_reads_post_run_recency(self):
        memory = PerformanceMemory()
        memory.record(PerformanceObservation("model", "coding", provider_success=True))
        report = evaluate_execution({"task_type": "coding", "status": "success"},
            routing_decision={"selected_model_ids": ["model"]}, performance_memory=memory,
            pre_run_history={"model": {"observation_count": 1, "provider_success_rate": 0.0}})
        self.assertEqual(report.evidence.historical_success_rate, 0.0)
        self.assertIsNone(report.evidence.recent_success_rate)

    def test_zero_usage_is_not_unknown(self):
        report = evaluate_execution({"token_usage": {"input_tokens": 0, "output_tokens": 0}})
        self.assertEqual(report.execution.total_tokens, 0)
        self.assertEqual(report.execution.input_tokens, 0)
        self.assertEqual(report.execution.output_tokens, 0)

    def test_aggregate_cost_is_not_fabricated_as_equal_per_call_cost(self):
        report = evaluate_execution({"cost": 1.0, "events": [
            {"event": "model_response", "status": "success"},
            {"event": "model_response", "status": "success"},
        ]})
        self.assertTrue(all(c.cost is None for c in report.model_contributions))

    def test_missing_usage_is_not_presented_as_a_complete_total(self):
        class Retry:
            calls = 0
            def generate(self, messages, tools, response_schema):
                self.calls += 1
                if self.calls == 1:
                    raise ProviderError("timeout", "timeout", retryable=True)
                return ModelResponse({"final": {}}, {"total_tokens": 7})
        result = AgentSession(provider(Retry()), ToolRegistry([])).run(
            "test", task_id="audit", task_type="coding", capabilities={"coding"}, system_instructions="")
        self.assertEqual(result.trajectory["token_usage"], {})
        self.assertEqual(result.trajectory["observed_token_usage"]["total_tokens"], 7)
        self.assertFalse(result.trajectory["token_usage_complete"])

    def test_actual_history_summary_accepted_by_dataset_gate(self):
        example = _base_example()
        example.historical_evidence[example.selected_model_id] = {"observation_count": 3}
        self.assertNotIn("no_historical_evidence_for_selected_model", apply_quality_gate(example).reject_reasons)

    def test_recent_limit_applies_without_model_filter(self):
        memory = PerformanceMemory()
        for i in range(4):
            memory.record(PerformanceObservation(str(i)))
        interface = PerformanceInterface(memory)
        self.assertEqual(len(interface.recent_performance(limit=2)), 2)
        self.assertEqual(interface.recent_performance(limit=0), [])

    def test_verification_cannot_exceed_total_cost_budget(self):
        primary = Responses(ModelResponse({"answer": "candidate"}))
        verifier = Responses(ModelResponse({"verified": True}))
        registry = ModelRegistry([provider(primary, "p"), provider(verifier, "v")])
        result = MultiModelExecutor(registry).execute(RoutingDecision("audit", ["p"], verifier_model_id="v"),
            [], [], {}, verify=True, max_cost=0.1)
        self.assertFalse(result.success)
        self.assertEqual(verifier.calls, 0)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.total_cost, 0.1)

    def test_verification_string_false_is_not_accepted_or_rewarded(self):
        memory = PerformanceMemory()
        registry = ModelRegistry([
            provider(Responses(ModelResponse({"answer": "candidate"})), "p"),
            provider(Responses(ModelResponse({"verified": "false"})), "v"),
        ])
        result = MultiModelExecutor(registry, performance_memory=memory).execute(
            RoutingDecision("audit", ["p"], verifier_model_id="v"), [], [], {}, verify=True)
        self.assertFalse(result.success)
        self.assertFalse(any(o.contributed_to_final for o in memory.observations()))

    def test_verification_failure_triggers_verified_fallback(self):
        primary = Responses(ModelResponse({"answer": "weak candidate"}))
        fallback = Responses(ModelResponse({"answer": "verified candidate"}))
        verifier = Responses(ModelResponse({"verified": False}), ModelResponse({"verified": True}))
        registry = ModelRegistry([
            provider(primary, "primary"),
            provider(fallback, "fallback"),
            provider(verifier, "verifier"),
        ])
        result = MultiModelExecutor(registry).execute(
            RoutingDecision("audit", ["primary"], fallback_model_ids=["fallback"], verifier_model_id="verifier"),
            [], [], {}, verify=True,
        )
        self.assertTrue(result.success)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.accepted_model_id, "fallback")
        self.assertEqual(result.final_output, {"answer": "verified candidate"})
        self.assertEqual(verifier.calls, 2)

    def test_reusing_trial_identity_cannot_overwrite_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = CampaignRunner(directory, live=False)
            config = CampaignConfig("RW01", "scripted", live=False)
            runner.run_trial(config, 1)
            with self.assertRaises(FileExistsError):
                runner.run_trial(config, 1)

    def test_legacy_runner_uses_task_adapter_and_separate_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(directory, live=False)
            trial = runner.run_trial(ExperimentConfig("RW02", ["test"], "single", live=False), 1)
            self.assertTrue(trial.success)
            self.assertEqual(runner.memory.count(), trial.model_calls)
            self.assertEqual(str(runner.memory.path), str(Path(directory) / "performance_memory_scripted.jsonl"))
            self.assertFalse((Path(directory) / "performance_memory_live.jsonl").exists())
            with self.assertRaises(ValueError):
                runner.run_trial(ExperimentConfig("RW02", ["test"], "single", live=True), 2)

    def test_synthesis_cannot_exceed_budget(self):
        primary = Responses(ModelResponse({"answer": "candidate"}))
        synthesis = Responses(ModelResponse({"answer": "synthesized"}))
        registry = ModelRegistry([provider(primary, "p"), provider(synthesis, "s")])
        result = MultiModelExecutor(registry).execute(RoutingDecision("audit", ["p"], synthesis_model_id="s"),
            [], [], {}, synthesize=True, max_cost=0.1)
        self.assertEqual(synthesis.calls, 0)
        self.assertEqual(result.total_cost, 0.1)


if __name__ == "__main__":
    unittest.main()
