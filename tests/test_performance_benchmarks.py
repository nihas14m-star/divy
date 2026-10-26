"""Tests for the P01-P10 performance evaluation scenarios.

These verify the deterministic evaluation scenarios in
``divy.performance_benchmark`` all pass. They are independent of the B01-B20
and O01-O10 benchmarks owned by the orchestration stream.
"""

from __future__ import annotations

import unittest

from divy.performance_benchmark import run_scenarios


class PerformanceBenchmarkTests(unittest.TestCase):
    def test_p01_to_p10_scenarios_all_pass(self) -> None:
        results = run_scenarios()
        scenario_ids = [result["scenario"] for result in results]
        self.assertEqual(
            scenario_ids,
            [f"P{i:02d}" for i in range(1, 11)],
            scenario_ids,
        )
        failures = []
        for result in results:
            for name, ok in result.pop("assertions", {}).items():
                if not ok:
                    failures.append(f'{result["scenario"]}.{name}')
        self.assertFalse(failures, failures)


if __name__ == "__main__":
    unittest.main()
