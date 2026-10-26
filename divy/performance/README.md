# DIVY Performance Intelligence + Evaluation Subsystem

This package is DIVY's **Model Performance Intelligence + Evaluation**
subsystem. It gives DIVY reliable historical evidence about how models
actually perform, so the core orchestrator can reason about which model is
best, fastest, cheapest, most reliable, or most failure-prone for a given
kind of task.

It is **not** a routing engine. It answers questions with evidence; the
router decides what to do with that evidence.

## Purpose

Provide the evidence needed to answer:

* Which model is best for this kind of task?
* Which model is fastest? Cheapest? Most reliable?
* Which model frequently fails?
* Which model performs well after retries?
* When is using multiple models worth the additional cost?

## Architecture

The subsystem is small, independent, and provider-neutral. It has no
dependencies beyond the Python standard library and requires no GPU,
database server, or cloud service.

### `divy/performance/schemas.py` — `PerformanceObservation`

A single structured observation of an observed model call/execution:

```
model_id, task_type, capabilities, provider_success,
contributed_to_final, task_success, verification_passed,
latency_ms, cost, token_count, tool_calls, recovery_attempts,
execution_id, timestamp, provider, task_id, failure_type,
input_tokens, output_tokens, final_outcome
```

Only fields that are actually measurable are recorded. Missing optional
metrics are `None`, never fabricated. The field order/names are the
integration contract consumed by
`divy.execution.MultiModelExecutor._record_performance`.

### `divy/performance/memory.py` — `PerformanceMemory`

An append-only JSONL persistent store (consistent with DIVY's existing
`EventLogger` trajectory architecture). No server, no cloud, no extra
dependencies.

* `record(observation)` — append an observation; never overwrites.
* `observations()` — rehydrated list of all recorded observations.
* `summarize(model_id=, task_type=)` — `{"observation_count", "provider_success_rate", "contribution_rate", ...}`.
* `PerformanceMemory()` with no path works in-memory only (useful for tests
  and in-process integration).

Observations are never overwritten or deleted, preserving traceability of
historical data that may influence future routing.

### `divy/performance/analytics.py` — `PerformanceAnalytics` + `aggregate`

Deterministic statistics computed from stored observations:

* **Reliability:** success rate, failure rate, verification success rate,
  recovery success rate, failure-type breakdown.
* **Efficiency:** average & median latency, average model/tool/recovery
  calls.
* **Economics:** average token count, average cost, cost per successful
  task.
* **Capability / task grouping:** stats grouped by capability or task type.
* **Confidence / sample size:** `label_confidence(sample_size)` and a
  Wilson-score confidence interval for success rates, so tiny samples are
  never mistaken for strong evidence.

Missing data is `None`. Nothing is invented.

### `divy/performance/comparison.py` — `ModelComparison`

`compare(models, task_type=, score_config=)` returns raw measurements per
model, optionally with a derived score/rank. The derived score is only
produced when a caller explicitly supplies `ModelScoreConfig`; otherwise
only raw evidence is exposed. No model is declared a winner by arbitrary
assumptions.

### `divy/performance/integration.py` — `PerformanceInterface`

The clean read-only interface consumed by the core DIVY orchestrator:

* `model_evidence(model_id)`
* `task_type_evidence(task_type)` / `capability_evidence(capability)`
* `compare(models, task_type=, score_config=)`
* `get_best_historical_candidates(models, task_type=, min_samples=)` —
  returns evidence, filtered by sample size, **without declaring a winner**.
* `recent_performance(model_id=, limit=)`
* `aggregate_all()`

Methods never route, select, or execute.

### `divy/performance/trajectory.py` — trajectory adapter

`trajectory_to_observation(...)` transforms an existing DIVY execution
trajectory (orchestrator result / MultiModelExecutor result / evaluation)
into a `PerformanceObservation`. No duplicate logging architecture is
created; it reads the structures DIVY already records.

## Data integrity

Incorrect performance data is worse than missing data. The subsystem:

* never fabricates metrics
* never silently replaces missing values
* never treats failed evaluations as successes
* never treats model claims as independent verification
* never overwrites historical observations without traceability

It prefers `unknown` / `unavailable` / `unevaluated` (represented as
`None`) over invented values.

## Provider agnosticism

No vendor or model name (Astra, DeepSeek, Nemotron, OpenAI, Anthropic,
Google, …) is hard-coded into the core engine. They are simply possible
model identifiers. The system works with any model, any provider, any
future adapter.

## Integration contract (orchestrator → performance)

```
Orchestrator
      ↓
Performance Intelligence
      ↓
Historical evidence
```

The orchestrator supplies a `PerformanceMemory` to
`MultiModelExecutor(registry, performance_memory=memory)`. After each
execution the executor records per-call observations. The orchestrator can
then query `PerformanceInterface` for evidence. The performance subsystem
does not independently control execution.

## Future learning

The recorded observations form the foundation for a learned router dataset:

```
task characteristics + available models + historical performance +
routing decision + actual outcome
```

No model is trained by this subsystem.

## Running

From the repository root:

```text
python -m unittest discover -s tests -v        # full suite (incl. performance)
python -m divy.performance_benchmark           # P01-P10 scenarios
python -m divy.benchmark                       # B01-B20 (unchanged)
python -m divy.orchestration_benchmark         # O01-O10 (unchanged)
python -m divy.baseline_benchmark              # baseline (unchanged)
```

## Files

```
divy/performance/__init__.py     public package API
divy/performance/schemas.py      PerformanceObservation data model
divy/performance/memory.py       PerformanceMemory JSONL store
divy/performance/analytics.py    deterministic statistics
divy/performance/comparison.py   model comparison interface
divy/performance/integration.py  orchestrator-facing evidence interface
divy/performance/trajectory.py   trajectory -> observation adapter
divy/performance_benchmark.py    P01-P10 evaluation scenarios
tests/test_performance.py        performance subsystem unit tests
tests/test_performance_benchmarks.py  P01-P10 scenario tests
```

The subsystem does not modify `divy/execution.py`, `divy/orchestrator.py`,
`divy/routing.py`, or any test file owned by the orchestration stream. It
only provides the `PerformanceMemory` / `PerformanceObservation` interface
those modules import.
