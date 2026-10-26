# DIVY Release Readiness — October 26, 2026

## Status

DIVY is release-ready as a reproducible provider-neutral orchestration and
evaluation project. The final pass made only focused robustness, security, and
documentation improvements; the routing architecture and scientific
conclusion were not tuned or redesigned.

## Validated scope

- Model registry and provider-neutral adapters
- Deterministic routing with static and empirical evidence
- Confidence shrinkage, Wilson intervals, recency weighting, and
  exploration/exploitation
- Bounded `AgentSession` execution through `ToolRegistry`
- Sandboxed filesystem tools and independent verification
- Append-only `PerformanceMemory`
- Evaluation, model contribution attribution, and router dataset export
- Scripted/live campaign separation and collision-resistant trial paths
- NVIDIA configuration and one-request smoke-test path

## Security conclusion

`test.run` still uses unrestricted Python `exec()` in the host process. The
current hosted-model protection remains fail-closed: `HttpModelAdapter` refuses
to send any request when a registered tool advertises
`x-divy-requires-process-isolation`. No fake sandbox was added. A real isolated
runner (for example a separate container, gVisor, or Firecracker boundary) is
still required before hosted model-driven test execution can be enabled.

Tool error messages are now generic at the artifact boundary, and fixture test
errors no longer echo arbitrary exception text. NVIDIA credentials, bearer
headers, and provider response bodies remain excluded from provider errors and
recorded artifacts.

## Scientific conclusion

Dynamic routing has **not** demonstrated a measurable advantage over simpler
strategies in the current experiments. Fixed routing performed strongly in the
tested scenarios. The live sample sizes were small, live task coverage was
limited, NVIDIA pricing was unavailable, and hosted models are nondeterministic.
These results are evidence about this configuration, not a general superiority
claim.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
python -m compileall -q divy tests
python -m divy.benchmark
python -m divy.orchestration_benchmark
python -m divy.performance_benchmark
python -m divy.baseline_benchmark
python -m divy.realworld
python -m divy.demo
DIVY_RUN_LIVE=0 DIVY_CAMPAIGN_TASKS=RW01,RW02,RW03,RW04,RW05,RW06,RW07,RW08,RW09 DIVY_CAMPAIGN_TRIALS=1 DIVY_CAMPAIGN_OUTPUT_DIR=results/final_scripted_campaign python -m divy.campaign
python tests/audit_preserved_artifacts.py
```

## Final pass results

- `python -m pytest tests/ -q`: **186 passed**
- `python -m compileall -q divy tests`: **clean**
- `python -m divy.benchmark`: **B01-B20, 20/20**
- `python -m divy.orchestration_benchmark`: **O01-O10, 10/10**
- `python -m divy.performance_benchmark`: **P01-P10, 10/10**
- `python -m divy.baseline_benchmark`: **passed**
- `python -m divy.realworld`: **RW01 scripted, passed**
- deterministic RW01-RW09 campaign: **9/9**
- `python -m divy.demo`: **passed; 5 performance observations recorded**
- `python -m divy.smoke` without configuration: **safe no-network skip**
- credential/artifact scan: **0 credential matches**

The preserved artifact audit still reports 18 historical live trial paths with
multiple execution IDs. Those are pre-fix diagnostic artifacts and are retained
for traceability; the current campaign runner refuses to reuse a trial
directory, so new runs cannot silently overwrite one another.

Live NVIDIA commands are optional and should only be run when a concrete
integration defect requires them. They consume provider credits and are not
needed to reproduce the deterministic baseline.
