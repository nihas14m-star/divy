"""Provider-neutral model interfaces and the deterministic smoke-test adapter."""

from __future__ import annotations

import json
from typing import Any, Protocol

from .schemas import ModelResponse, TaskDefinition


class ModelAdapter(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_schema: dict[str, Any],
    ) -> ModelResponse:
        """Return structured output without directly executing tools."""
        ...


class UnsupportedModelAdapter:
    """Explicit placeholder for a future local or hosted model integration."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_schema: dict[str, Any],
    ) -> ModelResponse:
        raise RuntimeError(
            "No AI provider is configured. Implement a ModelAdapter or use "
            "ScriptedBenchmarkAdapter for the harness smoke test."
        )


class ScriptedBenchmarkAdapter:
    """Deterministic adapter for proving the harness, not a trained model.

    It follows task-specific observable action strategies while using the same
    structured adapter contract a future model must implement. It deliberately
    does not have access to the tool registry and cannot execute commands.
    """

    def __init__(self, task: TaskDefinition) -> None:
        self.task = task

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_schema: dict[str, Any],
    ) -> ModelResponse:
        state = messages[-1].get("content", {}) if messages else {}
        phase = state.get("phase")
        if phase == "UNDERSTAND":
            content = {
                "phase": "UNDERSTAND",
                "success_criteria": list(self.task.success_criteria),
                "needs_clarification": self.task.task_id == "B16",
                "clarification": "There are two matching config.json files; which one should be used?" if self.task.task_id == "B16" else None,
            }
        elif phase == "PLAN":
            content = {"phase": "PLAN", "subtasks": self._plan()}
        elif phase in {"SELECT_ACTION", "REPLAN"}:
            content = {"phase": "SELECT_ACTION", "action": self._next_action(state)}
        elif phase == "DIAGNOSE":
            content = {"phase": "DIAGNOSE", "diagnosis": self._diagnosis(state)}
        else:
            content = {"phase": phase or "UNKNOWN", "action": None}
        token_count = len(json.dumps(content, sort_keys=True)) // 4
        return ModelResponse(content=content, token_usage={"output_tokens": token_count})

    def _plan(self) -> list[dict[str, Any]]:
        plans = {
            "B01": ["Read the JSON configuration and verify the requested fields."],
            "B02": ["Search the fixture workspace for TODO and FIXME markers."],
            "B03": ["Read the JSON configuration.", "Write only the requested structured change.", "Verify preservation of unrelated fields."],
            "B04": ["Read the broken function.", "Patch the incorrect operator.", "Run the registered test."],
            "B05": ["Read both source notes.", "Write a report with the required headings and facts.", "Verify the report artifact."],
            "B06": ["Read the CSV.", "Calculate aggregate values.", "Verify the results."],
            "B07": ["Inspect config.txt and pipeline.py independently.", "Inspect README.md independently.", "Combine the roles."],
            "B08": ["Read the setting.", "Write the requested setting.", "Verify the setting."],
            "B09": ["Use the calculator tool.", "Verify the result."],
            "B10": ["Read the configuration.", "Verify that no write is necessary."],
            "B11": ["Try the expected path.", "Search for the correct configuration.", "Read the discovered path and verify it."],
            "B12": ["Read the implementation.", "Apply an implementation change.", "Run the test.", "Diagnose a failed implementation.", "Correct it and rerun the test."],
            "B13": ["Read the source.", "Write the report.", "Verify its contents and repair if needed."],
            "B14": ["Read the current configuration.", "Write using an observation version.", "Re-read after a stale-state failure.", "Write and verify the current state."],
            "B15": ["Read the source.", "Try the preferred destination.", "Use the allowed destination after permission failure.", "Verify the artifact."],
            "B16": ["Identify matching files.", "Ask for clarification rather than guessing."],
            "B17": ["Read the existing result.", "Stop after verification."],
            "B18": ["Search all references.", "Patch each reference.", "Run the registered test."],
            "B19": [
                {"description": "Inspect inputs/a.txt.", "parallel_group": "inputs"},
                {"description": "Inspect inputs/b.txt.", "parallel_group": "inputs"},
                "Combine the independent observations.",
            ],
            "B20": ["Read the config and pipeline.", "Repair the pipeline.", "Enable the feature.", "Run the test.", "Write an evidence report.", "Verify the result."],
        }
        subtasks = []
        for index, value in enumerate(plans[self.task.task_id]):
            if isinstance(value, dict):
                item = {"id": f"t{index + 1}", **value, "depends_on": []}
            else:
                item = {"id": f"t{index + 1}", "description": value, "depends_on": []}
            subtasks.append(item)
        return subtasks

    def _next_action(self, state: dict[str, Any]) -> dict[str, Any] | None:
        observations = state.get("observations", [])
        successful = [item for item in observations if item.get("ok")]
        all_results = observations
        task_id = self.task.task_id

        if task_id == "B01" and not self._has_success(successful, "fs.read"):
            return self._read("config.json")
        if task_id == "B02" and not self._has_success(successful, "fs.search"):
            return self._action("fs.search", {"pattern": "TODO|FIXME"})
        if task_id == "B03":
            if not self._has_success(successful, "fs.read"):
                return self._read("config.json")
            if not self._has_success(successful, "fs.write"):
                original = self._latest_read(successful, "config.json")
                updated = dict(original)
                updated["feature_enabled"] = True
                return self._write("config.json", json.dumps(updated, indent=2) + "\n")
        if task_id == "B04":
            if not self._has_success(successful, "fs.read"):
                return self._read("calculator.py")
            if not self._has_success(successful, "fs.patch"):
                return self._patch("calculator.py", "return a - b", "return a + b")
            if not self._has_success(successful, "test.run"):
                return self._action("test.run", {"test_id": "b04_calculator"})
        if task_id == "B05":
            if self._count_success(successful, "fs.read") < 2:
                paths = ["notes/summary.md", "notes/details.md"]
                return self._read(paths[self._count_success(successful, "fs.read")])
            if not self._has_success(successful, "fs.write"):
                summary = self._latest_read(successful, "notes/summary.md")
                details = self._latest_read(successful, "notes/details.md")
                report = "# Summary\n\n" + str(summary).strip() + "\n\n# Findings\n\n" + str(details).strip() + "\n\n# Next Steps\n\nReview the findings and implement the stated follow-up.\n"
                return self._write("report.md", report)
        if task_id == "B06":
            if not self._has_success(successful, "fs.read"):
                return self._read("data/sales.csv")
            if not self._has_success(successful, "calc.evaluate"):
                return self._action("calc.evaluate", {"expression": "10 + 20 + 30 + 40"})
        if task_id == "B07":
            paths = ["config.txt", "pipeline.py", "README.md"]
            if self._count_success(successful, "fs.read") < len(paths):
                return self._read(paths[self._count_success(successful, "fs.read")])
        if task_id == "B08":
            return None
        if task_id == "B09" and not self._has_success(successful, "calc.evaluate"):
            return self._action("calc.evaluate", {"expression": "17 * 6"})
        if task_id == "B10" and not self._has_success(successful, "fs.read"):
            return self._read("config.json")
        if task_id == "B11":
            if not any(item.get("tool") == "fs.read" and item.get("path") == "config.json" for item in all_results):
                return self._read("config.json")
            if not self._has_success(successful, "fs.search"):
                return self._action("fs.search", {"pattern": "timeout"})
            discovered = self._search_path(successful)
            if discovered and not self._has_success(successful, "fs.read", discovered):
                return self._read(discovered)
        if task_id == "B12":
            if not self._has_success(successful, "fs.read"):
                return self._read("transform.py")
            patches = self._successful_events(successful, "fs.patch")
            failed_tests = [item for item in all_results if item.get("tool") == "test.run" and not item.get("ok")]
            if not patches:
                return self._patch("transform.py", "return a - b", "return a + b")
            if failed_tests and len(patches) == 1:
                return self._patch("transform.py", "return a + b", "return a * b")
            if not any(item.get("tool") == "test.run" and item.get("ok") for item in all_results):
                return self._action("test.run", {"test_id": "b12_transform"})
        if task_id == "B13":
            if not self._has_success(successful, "fs.read"):
                return self._read("source.md")
            writes = self._successful_events(successful, "fs.write")
            if not writes:
                return self._write("report.md", "# Report\n\nThis is an incorrect draft.\n")
            if len(writes) == 1:
                source = self._latest_read(successful, "source.md")
                return self._write("report.md", "# Report\n\n" + str(source).strip() + "\n")
        if task_id == "B14":
            reads = self._successful_events(successful, "fs.read")
            if not reads:
                return self._read("config.json")
            writes = self._successful_events(successful, "fs.write")
            if any((item.get("error") or {}).get("type") == "stale_state" for item in all_results):
                if self._count_success(successful, "fs.read") < 2:
                    return self._read("config.json")
                current = self._latest_read(successful, "config.json")
                current["enabled"] = True
                return self._write("config.json", json.dumps(current, indent=2) + "\n", self._latest_hash(successful, "config.json"))
            if not writes:
                original = self._latest_read(successful, "config.json")
                version = self._latest_hash(successful, "config.json")
                updated = dict(original)
                updated["enabled"] = True
                return self._write("config.json", json.dumps(updated, indent=2) + "\n", version)
        if task_id == "B15":
            if not self._has_success(successful, "fs.read"):
                return self._read("source.md")
            writes = self._successful_events(successful, "fs.write")
            if not any((item.get("error") or {}).get("type") == "permission_denied" for item in all_results):
                return self._write("preferred/report.md", "# Report\n\n" + str(self._latest_read(successful, "source.md")).strip() + "\n")
            if not writes:
                return self._write("allowed/report.md", "# Report\n\n" + str(self._latest_read(successful, "source.md")).strip() + "\n")
        if task_id == "B17" and not self._has_success(successful, "fs.read"):
            return self._read("result.txt")
        if task_id == "B18":
            if not self._has_success(successful, "fs.search"):
                return self._action("fs.search", {"pattern": "OLD_KEY"})
            paths = ["src/app.py", "config.json", "tests/test_app.py"]
            patched = {item.get("data", {}).get("path") for item in successful if item.get("tool") == "fs.patch"}
            for path in paths:
                if path not in patched:
                    return self._patch(path, "OLD_KEY", "NEW_KEY")
            if not self._has_success(successful, "test.run"):
                return self._action("test.run", {"test_id": "b18_keys"})
        if task_id == "B19":
            paths = ["inputs/a.txt", "inputs/b.txt"]
            if self._count_success(successful, "fs.read") < 2:
                return self._read(paths[self._count_success(successful, "fs.read")])
        if task_id == "B20":
            if not self._has_success(successful, "fs.read", "config.json"):
                return self._read("config.json")
            if not self._has_success(successful, "fs.read", "pipeline.py"):
                return self._read("pipeline.py")
            if not self._has_success(successful, "fs.patch"):
                return self._patch("pipeline.py", "return a - b", "return a + b")
            if not self._has_success(successful, "fs.write"):
                config = self._latest_read(successful, "config.json")
                config["feature_enabled"] = True
                return self._write("config.json", json.dumps(config, indent=2) + "\n")
            if not self._has_success(successful, "test.run"):
                return self._action("test.run", {"test_id": "b20_pipeline"})
            if not self._has_success(successful, "fs.read", "notes.md"):
                return self._read("notes.md")
            if self._count_success(successful, "fs.write") < 2:
                return self._write("evidence.md", "# Evidence\n\nPipeline test passed for inputs 5 and 7.\n\nFeature enabled: true.\n")
        return None

    def _diagnosis(self, state: dict[str, Any]) -> str:
        errors = [item.get("error") or {} for item in state.get("observations", []) if not item.get("ok")]
        if errors:
            return f"Observed structured failure: {errors[-1].get('type', 'unknown')}. Re-plan with a different safe action."
        return "Verification did not pass; inspect the current state and repair the artifact before stopping."

    @staticmethod
    def _action(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"tool": tool, "arguments": arguments}

    @staticmethod
    def _read(path: str) -> dict[str, Any]:
        return ScriptedBenchmarkAdapter._action("fs.read", {"path": path})

    @staticmethod
    def _write(path: str, content: str, expected_sha256: str | None = None) -> dict[str, Any]:
        arguments: dict[str, Any] = {"path": path, "content": content}
        if expected_sha256 is not None:
            arguments["expected_sha256"] = expected_sha256
        return ScriptedBenchmarkAdapter._action("fs.write", arguments)

    @staticmethod
    def _patch(path: str, old: str, new: str) -> dict[str, Any]:
        return ScriptedBenchmarkAdapter._action("fs.patch", {"path": path, "old": old, "new": new})

    @staticmethod
    def _has_success(observations: list[dict[str, Any]], tool: str, path: str | None = None) -> bool:
        return any(item.get("tool") == tool and (path is None or item.get("path") == path) for item in observations)

    @staticmethod
    def _count_success(observations: list[dict[str, Any]], tool: str) -> int:
        return sum(item.get("tool") == tool for item in observations)

    @staticmethod
    def _successful_events(observations: list[dict[str, Any]], tool: str) -> list[dict[str, Any]]:
        return [item for item in observations if item.get("tool") == tool]

    @staticmethod
    def _latest_read(observations: list[dict[str, Any]], path: str) -> Any:
        for item in reversed(observations):
            if item.get("tool") == "fs.read" and item.get("path") == path:
                data = item.get("data", {})
                return data.get("json", data.get("content", ""))
        return ""

    @staticmethod
    def _latest_hash(observations: list[dict[str, Any]], path: str) -> str | None:
        for item in reversed(observations):
            if item.get("tool") == "fs.read" and item.get("path") == path:
                return item.get("data", {}).get("sha256")
        return None

    @staticmethod
    def _search_path(observations: list[dict[str, Any]]) -> str | None:
        for item in reversed(observations):
            if item.get("tool") == "fs.search":
                matches = item.get("data", {}).get("matches", [])
                if matches:
                    return matches[0].get("path")
        return None
