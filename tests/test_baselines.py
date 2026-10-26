from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from divy.baseline_benchmark import run_baseline_experiment


class BaselineTests(unittest.TestCase):
    def test_controlled_baseline_comparison_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_baseline_experiment(Path(directory))
        strategies = summary["strategies"]
        self.assertEqual(strategies["single_model"]["task_count"], 5)
        self.assertEqual(strategies["fixed_routing"]["task_count"], 5)
        self.assertEqual(strategies["dynamic_routing"]["task_count"], 5)
        self.assertEqual(strategies["dynamic_routing"]["success_rate"], 1.0)
        self.assertLess(strategies["single_model"]["success_rate"], strategies["dynamic_routing"]["success_rate"])
        self.assertLess(strategies["fixed_routing"]["success_rate"], strategies["dynamic_routing"]["success_rate"])
        self.assertEqual(summary["hardware"], "RTX 3050 6GB / 16GB RAM / Windows")


if __name__ == "__main__":
    unittest.main()
