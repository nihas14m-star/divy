from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from divy.agent_loop import AgentLimits, AgentSession
from divy.adapters import ProviderError
from divy.execution import ModelCallRecord
from divy.performance import PerformanceMemory
from divy.providers import ModelMetadata, ModelProvider, ModelRegistry
from divy.schemas import EvaluationResult, ModelResponse
from divy.tools import FilesystemTools, SandboxWorkspace, ToolRegistry


class SequenceAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, tools, response_schema):
        self.calls += 1
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return ModelResponse(response, {"total_tokens": 3})


def make_session(adapter, workspace, *, limits=None, memory=None):
    registry = ToolRegistry(FilesystemTools(workspace).definitions())
    provider = ModelProvider(ModelMetadata("test-model", "test", frozenset({"general"}), 4096, 0.1, 1, 1.0), adapter)
    return AgentSession(provider, registry, limits=limits, performance_memory=memory)


class AgentLoopTests(unittest.TestCase):
    def test_single_final_response(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            session = make_session(SequenceAdapter([{"final": {"answer": "done"}}]), workspace)
            result = session.run("finish", task_id="T1", task_type="simple", capabilities={"general"}, system_instructions="return structured output")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.final_output, {"answer": "done"})
        self.assertEqual(result.trajectory["model_calls"], 1)

    def test_token_total_is_derived_when_provider_omits_it(self):
        class OutputOnlyAdapter:
            def generate(self, messages, tools, response_schema):
                return ModelResponse(
                    {"final": {"answer": "done"}},
                    {"input_tokens": 5, "output_tokens": 7},
                )

        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            session = make_session(OutputOnlyAdapter(), workspace)
            result = session.run("finish", task_id="T1b", task_type="simple", capabilities={"general"}, system_instructions="return structured output")
        self.assertEqual(result.trajectory["token_usage"]["total_tokens"], 12)

    def test_one_tool_call_then_final(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            workspace.write_text("note.txt", "hello")
            adapter = SequenceAdapter([
                {"action": {"tool": "fs.read", "arguments": {"path": "note.txt"}}},
                {"final": {"answer": "read"}},
            ])
            result = make_session(adapter, workspace).run("read", task_id="T2", task_type="read", capabilities={"general"}, system_instructions="use tools")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.trajectory["tool_calls"], 1)
        self.assertTrue(any(event["event"] == "tool_observation" and event["success"] for event in result.trajectory["events"]))

    def test_multiple_tool_calls_and_performance_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            workspace.write_text("note.txt", "old")
            memory = PerformanceMemory()
            adapter = SequenceAdapter([
                {"action": {"tool": "fs.read", "arguments": {"path": "note.txt"}}},
                {"action": {"tool": "fs.patch", "arguments": {"path": "note.txt", "old": "old", "new": "new"}}},
                {"final": {"answer": "changed"}},
            ])
            result = make_session(adapter, workspace, memory=memory).run("change", task_id="T3", task_type="edit", capabilities={"general"}, system_instructions="use tools")
            updated_content = workspace.read_text("note.txt")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.trajectory["tool_calls"], 2)
        self.assertEqual(updated_content, "new")
        self.assertEqual(memory.summarize(model_id="test-model")["observation_count"], 3)

    def test_unknown_tool_and_tool_failure_are_observed_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            adapter = SequenceAdapter([
                {"action": {"tool": "unknown.tool", "arguments": {}}},
                {"action": {"tool": "fs.read", "arguments": {"path": "missing.txt"}}},
                {"final": {"answer": "gave up"}},
            ])
            result = make_session(adapter, workspace, limits=AgentLimits(max_recovery_attempts=2)).run("fail safely", task_id="T4", task_type="failure", capabilities={"general"}, system_instructions="use tools")
        self.assertEqual(result.status, "success")
        observations = [event for event in result.trajectory["events"] if event["event"] == "tool_observation"]
        self.assertEqual(len(observations), 2)
        self.assertFalse(observations[0]["success"])
        self.assertFalse(observations[1]["success"])
        self.assertEqual(result.trajectory["recovery_attempts"], 2)

    def test_malformed_model_response_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            result = make_session(SequenceAdapter([{}]), SandboxWorkspace(Path(directory))).run("bad", task_id="T5", task_type="bad", capabilities={"general"}, system_instructions="")
        self.assertEqual(result.status, "failed_model")

    def test_retryable_provider_empty_response_is_bounded_and_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = SequenceAdapter([
                ProviderError("empty_response", "provider returned no content", retryable=True),
                {"final": {"answer": "recovered"}},
            ])
            result = make_session(
                adapter,
                SandboxWorkspace(Path(directory)),
                limits=AgentLimits(max_recovery_attempts=1),
            ).run("recover", task_id="T5b", task_type="recover", capabilities={"general"}, system_instructions="")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.final_output, {"answer": "recovered"})
        self.assertEqual(result.trajectory["recovery_attempts"], 1)

    def test_verification_failure_replans_then_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            adapter = SequenceAdapter([{"final": {"answer": "first"}}, {"final": {"answer": "fixed"}}])
            evaluations = [EvaluationResult(False, False, {"reason": "bad"}, recoverable_failure=True), EvaluationResult(True, True, {"reason": "ok"})]
            def verifier(_trajectory):
                return evaluations.pop(0)
            result = make_session(adapter, workspace).run("verify", task_id="T6", task_type="verify", capabilities={"general"}, system_instructions="", verifier=verifier)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.trajectory["recovery_attempts"], 1)

    def test_cost_and_iteration_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = SandboxWorkspace(Path(directory))
            cost_result = make_session(SequenceAdapter([{"final": {"answer": "never"}}]), workspace, limits=AgentLimits(max_cost=0.05)).run("cost", task_id="T7", task_type="cost", capabilities={"general"}, system_instructions="")
            loop_result = make_session(SequenceAdapter([{"action": {"tool": "fs.list", "arguments": {}}}]), workspace, limits=AgentLimits(max_iterations=2, max_tool_calls=10)).run("loop", task_id="T8", task_type="loop", capabilities={"general"}, system_instructions="")
        self.assertEqual(cost_result.status, "failed_cost_limit")
        self.assertEqual(loop_result.status, "failed_iteration_limit")


if __name__ == "__main__":
    unittest.main()
