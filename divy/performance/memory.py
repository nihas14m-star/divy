"""Lightweight persistent local store of performance observations.

``PerformanceMemory`` is an append-only JSONL store, consistent with the
existing DIVY ``EventLogger`` trajectory architecture (``divy.logger``).
It requires no database server, no cloud service, and no extra
dependencies.

Observations are never overwritten or deleted, preserving traceability of
historical data that may influence future routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import PerformanceObservation


class PerformanceMemory:
    """Append-only JSONL store of performance observations.

    The store is reloadable: constructing a new ``PerformanceMemory`` on the
    same path rehydrates all previously recorded observations.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._observations: list[PerformanceObservation] = self._load()

    def _load(self) -> list[PerformanceObservation]:
        observations: list[PerformanceObservation] = []
        if self.path is None or not self.path.exists():
            return observations
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # A corrupted line must not silence the rest of the
                    # history; skip it and keep loading.
                    continue
                if not isinstance(data, dict):
                    # Valid JSON scalars/arrays are still invalid observation
                    # records. Treat them like corrupted lines rather than
                    # allowing attribute access below to abort rehydration.
                    continue
                observations.append(_dict_to_observation(data))
        return observations

    def record(self, observation: PerformanceObservation) -> None:
        """Append a single observation. Never overwrites existing data."""
        if self.path is not None:
            line = json.dumps(_observation_to_dict(observation), sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        self._observations.append(observation)

    def observations(self) -> list[PerformanceObservation]:
        return list(self._observations)

    def summarize(self, model_id: str | None = None, task_type: str | None = None) -> dict[str, Any]:
        """Aggregate statistics for the filtered observations.

        Returns ``{"observation_count": N, "provider_success_rate": float}``
        plus richer efficiency/economics fields when data exists.
        """
        filtered = [
            o for o in self._observations
            if (model_id is None or o.model_id == model_id)
            and (task_type is None or o.task_type == task_type)
        ]
        total = len(filtered)
        if total == 0:
            return {
                "observation_count": 0,
                "provider_success_rate": None,
            }
        successes = sum(1 for o in filtered if o.provider_success)
        contributions = sum(1 for o in filtered if o.contributed_to_final)
        return {
            "observation_count": total,
            "provider_success_rate": round(successes / total, 6),
            "provider_success_count": successes,
            "provider_failure_count": total - successes,
            "contribution_rate": round(contributions / total, 6),
            "contribution_count": contributions,
            "average_latency_ms": _avg([o.latency_ms for o in filtered if o.latency_ms is not None]),
            "average_cost": _avg([o.cost for o in filtered if o.cost is not None]),
            "average_token_count": _avg([o.token_count for o in filtered if o.token_count is not None]),
            "average_tool_calls": round(sum(o.tool_calls for o in filtered) / total, 4),
            "average_recovery_attempts": round(sum(o.recovery_attempts for o in filtered) / total, 4),
        }

    def count(self) -> int:
        return len(self._observations)

    def close(self) -> None:
        # No persistent connection to close for JSONL; kept for interface
        # symmetry with a database-backed store.
        return None

    def __enter__(self) -> "PerformanceMemory":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _observation_to_dict(observation: PerformanceObservation) -> dict[str, Any]:
    return observation.as_dict()


def _dict_to_observation(data: dict[str, Any]) -> PerformanceObservation:
    return PerformanceObservation(
        model_id=data.get("model_id", ""),
        task_type=data.get("task_type"),
        capabilities=tuple(data.get("capabilities") or []),
        provider_success=bool(data.get("provider_success", False)),
        contributed_to_final=bool(data.get("contributed_to_final", False)),
        task_success=bool(data.get("task_success", False)),
        verification_passed=_to_bool_optional(data.get("verification_passed")),
        latency_ms=_to_float_optional(data.get("latency_ms")),
        cost=_to_float_optional(data.get("cost")),
        token_count=_to_int_optional(data.get("token_count")),
        tool_calls=int(data.get("tool_calls") or 0),
        recovery_attempts=int(data.get("recovery_attempts") or 0),
        execution_id=data.get("execution_id"),
        timestamp=data.get("timestamp"),
        provider=data.get("provider"),
        task_id=data.get("task_id"),
        failure_type=data.get("failure_type"),
        input_tokens=_to_int_optional(data.get("input_tokens")),
        output_tokens=_to_int_optional(data.get("output_tokens")),
        final_outcome=data.get("final_outcome"),
    )


def _avg(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _to_bool_optional(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _to_float_optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_optional(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
