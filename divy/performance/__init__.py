"""DIVY Model Performance Intelligence + Evaluation subsystem.

This subsystem records observed model executions, aggregates historical
performance statistics, compares models on measurable evidence, and exposes
a clean, provider-neutral interface for the core DIVY orchestrator.

It intentionally does not route execution itself. It answers questions
about past performance so the router can decide what to do with that
evidence.
"""

from .analytics import PerformanceAnalytics, aggregate
from .comparison import ModelComparison, ModelScoreConfig, ModelComparisonResult
from .memory import PerformanceMemory
from .schemas import PerformanceObservation
from .trajectory import trajectory_to_observation, observation_from_orchestration_result, observations_from_agent_session
from .integration import PerformanceInterface, HistoricalEvidence, CandidateEvidence

__all__ = [
    "PerformanceObservation",
    "PerformanceMemory",
    "PerformanceAnalytics",
    "aggregate",
    "ModelComparison",
    "ModelComparisonResult",
    "ModelScoreConfig",
    "PerformanceInterface",
    "HistoricalEvidence",
    "CandidateEvidence",
    "trajectory_to_observation",
    "observation_from_orchestration_result",
]
