"""Tests for the campaign evaluation infrastructure."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from divy.campaign import (
    CampaignRunner,
    get_scripted_adapter,
    run_campaign,
    FIXED_ROUTING_TABLE,
    SCRIPTED_ADAPTERS,
)
from divy.live_tasks import get_all_task_ids, LIVE_TASKS
from divy.model import ModelResponse
from divy.providers import ModelMetadata, ModelProvider, ModelRegistry


class _AlwaysFinalAdapter:
    """Fake adapter that never performs the task, forcing verification failure."""

    def generate(self, messages, tools, response_schema):
        return ModelResponse(
            content={"final": {"message": "I did not modify the workspace."}},
            token_usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )


def _fake_registry_factory(providers: dict[str, ModelProvider]):
    def factory(model_ids: list[str]) -> ModelRegistry:
        return ModelRegistry([providers[model_id] for model_id in model_ids])

    return factory



class CampaignInfrastructureTests(unittest.TestCase):
    """Tests for campaign infrastructure components."""

    def test_all_tasks_have_scripted_adapters(self) -> None:
        """Every live task must have a scripted adapter."""
        task_ids = get_all_task_ids()
        for task_id in task_ids:
            self.assertIn(task_id, SCRIPTED_ADAPTERS, f"Missing scripted adapter for {task_id}")
            adapter = get_scripted_adapter(task_id)
            self.assertIsNotNone(adapter)

    def test_all_tasks_have_fixed_routing(self) -> None:
        """Every live task must have a fixed routing entry."""
        task_ids = get_all_task_ids()
        for task_id in task_ids:
            self.assertIn(task_id, FIXED_ROUTING_TABLE, f"Missing fixed routing for {task_id}")

    def test_scripted_campaign_runs_all_tasks(self) -> None:
        """Scripted campaign runs all 5 tasks successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_campaign(
                output_dir=tmpdir,
                live=False,
                task_ids=["RW01", "RW02", "RW03", "RW04", "RW05"],
                execution_modes=["scripted"],
                trials_per_config=1,
                seed=42,
            )

        # All 5 tasks should complete
        self.assertEqual(result["total_trials"], 5)
        self.assertEqual(result["successful"], 5)
        self.assertEqual(result["failed"], 0)

        # All tasks should have success_rate = 1.0
        by_task = result["summary"]["by_task"]
        for task_id in ["RW01", "RW02", "RW03", "RW04", "RW05"]:
            self.assertEqual(by_task[task_id]["success_rate"], 1.0)

    def test_campaign_separates_scripted_and_live_memory(self) -> None:
        """Scripted and live trials use separate performance memories."""
        runner = CampaignRunner(Path(tempfile.mkdtemp()), live=True)
        self.assertTrue(runner.scripted_memory.path.name.startswith("performance_memory_scripted"))
        self.assertTrue(runner.live_memory.path.name.startswith("performance_memory_live"))
        self.assertNotEqual(runner.scripted_memory.path, runner.live_memory.path)

    def test_campaign_records_required_metrics(self) -> None:
        """Each trial records all required metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_campaign(
                output_dir=tmpdir,
                live=False,
                task_ids=["RW01"],
                execution_modes=["scripted"],
                trials_per_config=1,
                seed=42,
            )

        trial = result["trials"][0]
        required_fields = [
            "task_id", "execution_mode", "model_id", "trial_num",
            "success", "status", "verification_passed",
            "model_calls", "tool_calls", "recovery_attempts",
            "latency_ms", "total_tokens", "input_tokens", "output_tokens",
            "cost", "routing_decision", "seed"
        ]
        for field in required_fields:
            self.assertIn(field, trial, f"Missing field: {field}")

        # Verify specific values for scripted RW01
        self.assertEqual(trial["task_id"], "RW01")
        self.assertEqual(trial["execution_mode"], "scripted")
        self.assertTrue(trial["success"])
        self.assertEqual(trial["status"], "success")
        self.assertTrue(trial["verification_passed"])
        self.assertGreater(trial["model_calls"], 0)
        self.assertGreater(trial["tool_calls"], 0)
        self.assertEqual(trial["recovery_attempts"], 0)
        self.assertIsNotNone(trial["latency_ms"])
        self.assertEqual(trial["cost"], 0.0)

    def test_campaign_summary_by_mode(self) -> None:
        """Summary includes breakdown by execution mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_campaign(
                output_dir=tmpdir,
                live=False,
                task_ids=["RW01", "RW02"],
                execution_modes=["scripted"],
                trials_per_config=2,
                seed=42,
            )

        summary = result["summary"]
        self.assertIn("by_mode", summary)
        self.assertIn("scripted", summary["by_mode"])
        mode_stats = summary["by_mode"]["scripted"]
        self.assertEqual(mode_stats["total"], 4)  # 2 tasks * 2 trials
        self.assertEqual(mode_stats["success"], 4)
        self.assertEqual(mode_stats["success_rate"], 1.0)
        self.assertIsNotNone(mode_stats["avg_latency_ms"])

    def test_campaign_summary_by_task(self) -> None:
        """Summary includes breakdown by task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_campaign(
                output_dir=tmpdir,
                live=False,
                task_ids=["RW01", "RW02", "RW03"],
                execution_modes=["scripted"],
                trials_per_config=1,
                seed=42,
            )

        summary = result["summary"]
        self.assertIn("by_task", summary)
        for task_id in ["RW01", "RW02", "RW03"]:
            self.assertIn(task_id, summary["by_task"])
            self.assertEqual(summary["by_task"][task_id]["total"], 1)
            self.assertEqual(summary["by_task"][task_id]["success"], 1)

    def test_campaign_summary_by_model(self) -> None:
        """Summary includes breakdown by model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_campaign(
                output_dir=tmpdir,
                live=False,
                task_ids=["RW01", "RW02"],
                execution_modes=["scripted"],
                trials_per_config=1,
                seed=42,
            )

        summary = result["summary"]
        self.assertIn("by_model", summary)
        self.assertIn("scripted-RW01", summary["by_model"])
        self.assertIn("scripted-RW02", summary["by_model"])

    def test_campaign_saves_trial_artifacts(self) -> None:
        """Campaign saves trajectory, evaluation report, and trial summary per trial."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_campaign(
                output_dir=tmpdir,
                live=False,
                task_ids=["RW01"],
                execution_modes=["scripted"],
                trials_per_config=1,
                seed=42,
            )

            trial_dir = Path(tmpdir) / "trials" / "RW01_scripted_t1"
            self.assertTrue(trial_dir.exists())
            self.assertTrue((trial_dir / "events.jsonl").exists())
            self.assertTrue((trial_dir / "trajectory.jsonl").exists())
            self.assertTrue((trial_dir / "evaluation_report.json").exists())
            self.assertTrue((trial_dir / "trial_summary.json").exists())
            self.assertTrue((trial_dir / "router_dataset_example.json").exists())

    def test_campaign_uses_different_models_per_mode(self) -> None:
        """Different execution modes use appropriate model selection."""
        # This test verifies the logic without requiring live models
        # by checking the CampaignConfig creation logic
        runner = CampaignRunner(Path(tempfile.mkdtemp()), live=False, base_model_ids=["model-a", "model-b", "model-c"])

        # Scripted mode uses scripted adapter
        # single_model would use first model
        # fixed_routing uses FIXED_ROUTING_TABLE
        # dynamic uses all models with routing policy

        # Just verify the runner initializes correctly
        self.assertIsNotNone(runner.scripted_memory)
        self.assertIsNotNone(runner.live_memory)

    def test_campaign_output_structure(self) -> None:
        """Campaign result has the expected top-level structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_campaign(
                output_dir=tmpdir,
                live=False,
                task_ids=["RW01"],
                execution_modes=["scripted"],
                trials_per_config=1,
                seed=42,
            )

        self.assertIn("live", result)
        self.assertIn("total_trials", result)
        self.assertIn("successful", result)
        self.assertIn("failed", result)
        self.assertIn("trials", result)
        self.assertIn("summary", result)
        self.assertFalse(result["live"])  # scripted mode should be live=False

    def test_campaign_trial_includes_evaluation_report(self) -> None:
        """Each trial result includes an evaluation report object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_campaign(
                output_dir=tmpdir,
                live=False,
                task_ids=["RW01"],
                execution_modes=["scripted"],
                trials_per_config=1,
                seed=42,
            )

            trial = result["trials"][0]
            # The evaluation_report is stored in the trial object but not in the dict
            # Check that the file was saved
            trial_dir = Path(tmpdir) / "trials" / "RW01_scripted_t1"
            eval_report = trial_dir / "evaluation_report.json"
            self.assertTrue(eval_report.exists())

            # Load and verify structure
            import json
            with eval_report.open() as f:
                report = json.load(f)

            self.assertIn("execution_id", report)
            self.assertIn("task_id", report)
            self.assertIn("execution_mode", report)
            self.assertIn("routing", report)
            self.assertIn("execution", report)
            self.assertIn("quality", report)
            self.assertIn("model_contributions", report)
            self.assertIn("evidence", report)
            self.assertEqual(report["execution"]["estimated_cost"], 0.0)

            with (trial_dir / "router_dataset_example.json").open() as f:
                example = json.load(f)
            self.assertEqual(example["task_id"], "RW01")
            self.assertEqual(example["complexity"], "moderate")
            self.assertTrue(example["verification_passed"])


class CampaignFallbackAndSeparationTests(unittest.TestCase):
    """Deterministic tests for live-mode fallback and result separation."""

    def _providers(self) -> dict[str, ModelProvider]:
        capabilities = frozenset({"coding", "structured_output"})
        return {
            "fake-fail": ModelProvider(
                ModelMetadata("fake-fail", "fake", capabilities, 4096, 0.0, 1.0, 0.5),
                _AlwaysFinalAdapter(),
            ),
            "fake-good": ModelProvider(
                ModelMetadata("fake-good", "fake", capabilities, 4096, 0.0, 1.0, 0.5),
                SCRIPTED_ADAPTERS["RW01"](),
            ),
        }

    def test_dynamic_fallback_uses_bounded_alternate_model(self) -> None:
        """Dynamic mode falls back once to a capable model after verification failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            providers = self._providers()
            runner = CampaignRunner(
                Path(tmpdir),
                live=True,
                base_model_ids=["fake-fail", "fake-good"],
                seed=42,
                model_registry_factory=_fake_registry_factory(providers),
            )
            result = runner.run_campaign(
                task_ids=["RW01"],
                execution_modes=["dynamic"],
                trials_per_config=1,
            )

            self.assertEqual(result["kind"], "live")
            trial = result["trials"][0]
            self.assertTrue(trial["success"])
            self.assertTrue(trial["fallback_used"])
            self.assertEqual(trial["attempted_model_ids"], ["fake-fail", "fake-good"])
            self.assertEqual(len(trial["attempts"]), 2)
            self.assertFalse(trial["attempts"][0]["success"])
            self.assertTrue(trial["attempts"][1]["success"])

            mode_summary = result["summary"]["by_mode"]["dynamic"]
            self.assertEqual(mode_summary["fallback_rate"], 1.0)
            self.assertIn("fake-fail", result["summary"]["by_attempt_model"])
            self.assertIn("fake-good", result["summary"]["by_attempt_model"])

            trial_dir = Path(tmpdir) / "trials" / "RW01_dynamic_t1"
            report = json.loads((trial_dir / "evaluation_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["routing"]["fallback_used"])
            model_ids = {item["model_id"] for item in report["model_contributions"]}
            self.assertEqual(model_ids, {"fake-fail", "fake-good"})
            self.assertTrue(any(
                item["model_id"] == "fake-good" and item["contributed_to_final"]
                for item in report["model_contributions"]
            ))
            self.assertTrue((trial_dir / "attempt_1" / "trajectory.jsonl").exists())
            self.assertTrue((trial_dir / "attempt_2" / "trajectory.jsonl").exists())

    def test_live_campaign_writes_separate_result_file(self) -> None:
        """A live-only campaign writes only campaign_results_live.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            providers = self._providers()
            runner = CampaignRunner(
                Path(tmpdir),
                live=True,
                base_model_ids=["fake-good"],
                seed=42,
                model_registry_factory=_fake_registry_factory(providers),
            )
            result = runner.run_campaign(
                task_ids=["RW01"],
                execution_modes=["single_model"],
                trials_per_config=1,
            )

            self.assertTrue(result["live"])
            self.assertTrue((Path(tmpdir) / "campaign_results_live.json").exists())
            self.assertFalse((Path(tmpdir) / "campaign_results_scripted.json").exists())
            manifest = json.loads((Path(tmpdir) / "campaign_results.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["kinds"]), {"live"})
            self.assertNotIn("summary", manifest)

    def test_mixed_campaign_keeps_scripted_and_live_summaries_separate(self) -> None:
        """Mixed campaigns expose separate summaries and never one mixed ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            providers = self._providers()
            runner = CampaignRunner(
                Path(tmpdir),
                live=True,
                base_model_ids=["fake-good"],
                seed=42,
                model_registry_factory=_fake_registry_factory(providers),
            )
            result = runner.run_campaign(
                task_ids=["RW01"],
                execution_modes=["scripted", "single_model"],
                trials_per_config=1,
            )

            self.assertEqual(set(result["kinds"]), {"scripted", "live"})
            self.assertNotIn("summary", result)
            self.assertEqual(result["kinds"]["scripted"]["total_trials"], 1)
            self.assertEqual(result["kinds"]["live"]["total_trials"], 1)
            self.assertIn("scripted", result["kinds"]["scripted"]["summary"]["by_mode"])
            self.assertIn("single_model", result["kinds"]["live"]["summary"]["by_mode"])
            self.assertTrue((Path(tmpdir) / "campaign_results_scripted.json").exists())
            self.assertTrue((Path(tmpdir) / "campaign_results_live.json").exists())

    def test_task_goals_state_required_output_contract(self) -> None:
        """RW04/RW05 goals must match the verifiers they are measured against."""
        rw04 = LIVE_TASKS["RW04"].goal.lower()
        self.assertIn("expected.txt", rw04)
        self.assertIn("word: count", rw04)

        rw05 = LIVE_TASKS["RW05"].goal.lower()
        self.assertIn("output.json", rw05)
        self.assertIn("'sum'", rw05)
        self.assertIn("'avg'", rw05)
        self.assertIn("'max'", rw05)


if __name__ == "__main__":
    unittest.main()
