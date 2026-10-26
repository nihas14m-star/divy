# DIVY AI — provider-neutral orchestration and evaluation

DIVY is a provider-neutral AI orchestration framework and reproducible
evaluation harness. It combines model registration, deterministic routing,
empirical performance memory, confidence-aware evidence, tool execution,
verification, recovery/fallback, evaluation, and router-dataset generation.
The validated baseline is deliberately conservative: the current experiments
do **not** demonstrate a measurable advantage for dynamic routing over simpler
fixed strategies.

The project makes orchestration mechanics observable before any learned router
is trained. Scripted adapters validate the control loop locally; live providers
are optional and remain isolated from scripted evidence.

## Website

The rendered project site lives in [`website/`](website/) and presents the
architecture, routing evidence, benchmark results, security boundary, and
reproduction path without claiming dynamic-routing superiority. It is built
with React, TypeScript, and Vite; see [`website/README.md`](website/README.md)
for local development and QA notes.

## What is included

- Provider-neutral model adapter interface.
- Provider registry with capability, context, cost, latency, reliability, and
  availability metadata.
- Generic environment-configured HTTP model adapter with typed provider
  failures and no credential logging.
- Append-only empirical model-performance memory kept separate from static
  provider metadata.
- Deterministic cost/latency-aware routing with fallback candidates.
- Multi-model execution with parallel calls, disagreement detection,
  verification, synthesis, and fallback.
- Bounded multi-turn `AgentSession` model↔tool loop with structured tool-call
  normalization, observations, verification, recovery, cost limits, and
  iteration limits.
- Registered tools with schemas, permission checks, deterministic results, and
  structured errors.
- Sandboxed fixture workspaces with no arbitrary shell or network tool.
- Explicit `UNDERSTAND -> PLAN -> SELECT ACTION -> EXECUTE -> OBSERVE ->
  VERIFY -> COMPLETE` orchestration loop.
- Independent evaluators for B01-B20 and orchestration benchmarks O01-O10.
- JSONL trajectory logging and JSON benchmark summaries.
- A deterministic scripted adapter used only to prove the harness end to end.

## Architecture

```text
User Task
    → Orchestrator
    → Model Registry
    → Routing
    → Model Adapter
    → Agent Session
    → Tools / Sandbox
    → Verification
    → Performance Memory
    → Evaluation
    → Router Dataset
    → Future Routing Evidence
```

## Install and run

From the repository root, using Python 3.11 or newer:

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
```

`python -m divy.demo` is the recommended first run. It is deterministic,
local, and does not require NVIDIA credits or any API key. It writes a compact
report to `results/demo/` and demonstrates:

```text
task → routing decision → model selection → tool execution → verification
     → performance observation → memory update → evaluation report
```

The benchmark commands use deterministic test doubles. They validate
orchestration and measurement mechanics, not model intelligence.

## License status

No license file is included in this final pass because licensing and ownership
terms were not provided. Public visibility does not by itself grant reuse or
redistribution rights.

The benchmark command writes full trajectories to `results/trajectories.jsonl`,
event-level logs to `results/events.jsonl`, per-task metrics to
`results/summary.json`, and aggregate metrics to `results/metrics.json`.

The orchestration benchmark writes:

- `results/orchestration_trajectories.jsonl`
- `results/orchestration_summary.json`
- `results/orchestration_metrics.json`

The controlled baseline experiment writes:

- `results/baseline_trajectories.jsonl`
- `results/baseline_comparison.json`

It compares a single-model policy, a hand-written fixed-routing policy, and
the deterministic dynamic router on the same five orchestration cases. Its
scripted adapters are test doubles; the result is an instrumentation and
experimental-design check, not a claim that dynamic routing improves real
model performance.

The complete deterministic RW01-RW09 campaign is run with:

```bash
DIVY_RUN_LIVE=0 \
DIVY_CAMPAIGN_TASKS=RW01,RW02,RW03,RW04,RW05,RW06,RW07,RW08,RW09 \
DIVY_CAMPAIGN_TRIALS=1 \
DIVY_CAMPAIGN_OUTPUT_DIR=results/final_scripted_campaign \
python -m divy.campaign
```

Campaign trial directories are created with `exist_ok=False`; rerunning with
the same output directory and trial identity fails instead of silently
overwriting evidence. Use a new output directory for a new campaign.

The repository-shaped software-engineering agent task writes separate
scripted/live artifacts:

- `results/realworld_scripted_agent_trajectories.jsonl`
- `results/realworld_scripted_agent_events.jsonl`
- `results/realworld_summary.json`

Live runs use:

- `results/realworld_live_agent_trajectories.jsonl`
- `results/realworld_live_agent_events.jsonl`
- `results/performance_memory_live.jsonl`

Scripted runs use `results/performance_memory_scripted.jsonl`; the two sources
are never combined.

`divy/agent_loop.py` is the runtime path for model-driven work. Each model
response is normalized as `FINAL`, `TOOL_CALL`, or `ERROR`; tool requests are
validated and executed only through `ToolRegistry`, then returned to the model
as structured observations. The session is bounded by iterations, tool calls,
recovery attempts, context growth, and optional cost.

Run `python -m divy.smoke` for one opt-in live-provider request. With no
`DIVY_MODEL_BASE_URL` and `DIVY_MODEL_ID`, it exits safely with a structured
missing-configuration result and makes no network call. When configured, it
uses one request only and records the provider, model, latency, tokens, cost,
output, and errors in `results/live_smoke.json` and
`results/live_smoke_events.jsonl`.

## Provider-agnostic orchestration

`divy/providers.py` contains the generic `ModelProvider` and `ModelRegistry`.
Providers expose metadata and wrap the existing `ModelAdapter`; the core never
references a vendor or model name. `divy/routing.py` contains the replaceable
`RoutingPolicy` interface and deterministic policy. `divy/execution.py`
executes selected providers, handles failures and fallback, optionally runs
parallel candidates, enforces optional execution cost limits, and supports
independent verification and synthesis.

## Real provider adapter

`divy/adapters.py` provides `HttpModelAdapter`, which calls a generic
OpenAI-compatible endpoint without naming a vendor. Configure it through
environment variables rather than source files:

```text
DIVY_MODEL_BASE_URL
DIVY_MODEL_ID
DIVY_MODEL_API_KEY       # optional for local endpoints
DIVY_MODEL_TIMEOUT_SECONDS
DIVY_MODEL_ENDPOINT_PATH # optional, defaults to /chat/completions
```

It handles typed timeout, network, authentication, rate-limit, temporary
provider, and malformed-response errors. The adapter only requests structured
model output; tool execution remains controlled by DIVY's tool registry.

`divy/performance/` stores empirical call observations in JSONL when given a
path, with analytics and confidence-aware comparison helpers. It does not
invent estimates and can optionally influence deterministic routing through
`DeterministicRoutingPolicy(performance_memory=...)`.

The scripted adapter is intentionally not an intelligence result. It is a
deterministic environment smoke test that exercises the same model adapter,
tool registry, sandbox, recovery loop, and evaluator interfaces a real model
will use.

No model is trained by this project. The scripted adapter is a harness smoke
test and a replaceable stand-in for a future local or remote model adapter.

## NVIDIA API Catalog integration

NVIDIA is registered as one provider implementation. NVIDIA model IDs are
defined only in `divy/nvidia.py`; the registry, router, executor, AgentSession,
verifier, and performance memory remain provider-neutral.

The current portfolio is deliberately complementary rather than largest-model
only:

| Model ID | Initial role | Capabilities used by DIVY | Context documented by NVIDIA | Tool calls | Multimodal | Cost/latency estimate |
|---|---|---|---:|---:|---:|---|
| `nvidia/nemotron-3.5-bolt-e4b` | fast executor / fallback | coding, reasoning, agentic, structured output | 65,536 | yes | no | unknown |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | planner / coding specialist | coding, reasoning, agentic, structured output | 1,000,000 | yes | no | unknown |
| `nvidia/nemotron-3.5-super-vl-120b-a12b` | multimodal planner / verifier | coding, reasoning, agentic, structured output, vision | 1,000,000 | yes | yes | unknown |

The context values and capability flags are sourced from NVIDIA's current model
pages and API references. NVIDIA does not publish a stable per-call price or
latency estimate on those pages, so DIVY records both as `None` until measured
from real trajectories. Static reliability is initialized to `0.5`; empirical
`PerformanceMemory` is expected to replace that prior over time.

DIVY does not register NVIDIA embedding or reranking models in this chat-agent
portfolio because the current `ModelAdapter` contract is generative and
tool-oriented. Adding those roles requires a separate provider-neutral
embedding/reranking adapter rather than pretending they are coding agents.

### Official NVIDIA sources consulted

- NVIDIA API Catalog / Build: `https://build.nvidia.com/models`
- NVIDIA API Catalog API documentation: `https://docs.api.nvidia.com/`
- NVIDIA LLM API reference: `https://docs.api.nvidia.com/nim/reference/llm-apis`
- Nemotron 3.5 Bolt E4B: `https://build.nvidia.com/nvidia/nemotron-3.5-bolt-e4b`
- Nemotron 3.5 Bolt E4B API reference: `https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-bolt-e4b`
- Nemotron 3.5 Lightning 30B A3B: `https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b`
- Nemotron 3.5 Lightning 30B A3B API reference: `https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-lightning-30b-a3b`
- Nemotron 3.5 Super VL 120B A12B: `https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b`
- Nemotron 3.5 Super VL 120B A12B API reference: `https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b`

## NVIDIA configuration

The preferred credential variable is `NVIDIA_API_KEY`. DIVY also accepts
`DIVY_NVIDIA_API_KEY` for application-local namespacing. The key is read only
from the process environment; it is never written to source, README files,
trajectory content, or error messages.

Required for live execution:

```bash
NVIDIA_API_KEY=<set locally, never commit>
```

Optional variables:

```text
DIVY_NVIDIA_MODEL_ID=nvidia/nemotron-3.5-bolt-e4b
DIVY_NVIDIA_MODEL_IDS=nvidia/nemotron-3.5-bolt-e4b,nvidia/nemotron-3.5-lightning-30b-a3b,nvidia/nemotron-3.5-super-vl-120b-a12b
DIVY_NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
DIVY_NVIDIA_TIMEOUT_SECONDS=60
DIVY_NVIDIA_RESPONSE_MODE=json_schema
DIVY_RUN_LIVE=1
```

Run one live RW01 evaluation:

```bash
DIVY_RUN_LIVE=1 python -m divy.realworld
```

Run the RW01-only controlled single/fixed/dynamic comparison:

```bash
python -m divy.nvidia_experiment
```

Run the one-request smoke test without enabling the campaign:

```bash
DIVY_MODEL_BASE_URL=https://integrate.api.nvidia.com/v1 \
DIVY_MODEL_ID=nvidia/nemotron-3.5-bolt-e4b \
DIVY_MODEL_API_KEY="$NVIDIA_API_KEY" \
python -m divy.smoke
```

The smoke test does not write or print the key. Without the required base URL
and model ID it exits safely without making a network request.

### RW01-RW09 campaign

`divy.campaign` runs each task under a deterministic scripted control and under
live `single_model`, `fixed_routing`, and `dynamic` strategies. Scripted and
live trials use separate performance memories and separate result files
(`campaign_results_scripted.json` and `campaign_results_live.json`); they are
never aggregated into one ranking. Dynamic mode may make at most one fallback
attempt to another capable model. Single-model and fixed-routing modes do not
fall back.

Every trial records the selected model, every attempted model, fallback usage,
model/tool calls, recovery attempts, latency, token usage (provider-reported for
live runs, adapter estimates for scripted runs), cost (`null` when unknown),
independent verification, task success, events, and the trajectory.

Run the deterministic control:

```text
DIVY_RUN_LIVE=0 DIVY_CAMPAIGN_OUTPUT_DIR=results/control python -m divy.campaign
```

Run one live strategy per command (repeat with `single_model`, `fixed_routing`,
or `dynamic`):

```text
DIVY_RUN_LIVE=1 \
DIVY_CAMPAIGN_MODES=single_model \
DIVY_NVIDIA_MODEL_IDS=nvidia/nemotron-3.5-bolt-e4b,nvidia/nemotron-3.5-lightning-30b-a3b,nvidia/nemotron-3.5-super-vl-120b-a12b \
DIVY_CAMPAIGN_OUTPUT_DIR=results/live_single \
python -m divy.campaign
```

## Controlled evaluation evidence

The current release-readiness summary is documented in
`results/final_evaluation/analysis/release_readiness.md`. The detailed
historical evaluation report is preserved at
`results/final_evaluation/analysis/final_report.md` with its original date.

The scientifically correct conclusion remains:

> Dynamic routing has **not** demonstrated a measurable advantage over simpler
> strategies in the current experiments.

Fixed routing performed strongly in the tested scenarios. The live sample was
small, live tasks were limited, NVIDIA pricing was unavailable, hosted models
are nondeterministic, and the experiments are not a basis for a superiority
claim.

### Historical controlled evaluation (2026-10-25)

### Live corrected pilot (RW01, RW06 × single_model/dynamic × 1 trial)

| Trial | Task | Strategy | Success | Model Calls | Tool Calls | Latency | Fallback |
|-------|------|----------|---------|-------------|------------|---------|----------|
| 1 | RW01 | single_model | ✅ | 4 | 3 | 3,777 ms | No |
| 2 | RW01 | dynamic | ✅ | 4 | 3 | 9,805 ms | No |
| 3 | RW06 | single_model | ✅ | 7 | 5 | 26,519 ms | No |
| 4 | RW06 | dynamic | ✅ | 10 | 9 | 14,127 ms | **Yes** (bolt → lightning) |

Infrastructure fixes (task goals with `src/` prefix) resolved path ambiguity.
Dynamic fallback worked for RW06 (bolt failed → lightning succeeded). This is
the only experiment where fallback triggered and changed the model.

### Cold-start experiment (3 trials × 2 tasks × 3 strategies = 18 trials)

| Strategy | Success Rate | Avg Latency | Avg Model Calls | Fallback Rate |
|----------|--------------|-------------|-----------------|---------------|
| single_model | 83.3% | 8,919 ms | 7.0 | 0% |
| fixed_routing | **100%** | **6,173 ms** | **4.0** | 0% |
| dynamic | **100%** | 7,747 ms | 4.33 | 0% |

Fixed routing (lightning) outperformed dynamic on latency. Dynamic selected
bolt for both tasks; no fallbacks triggered.

### Warm-start experiment (3 trials × 2 tasks × 3 strategies = 18 trials)

Reused performance memory from cold start + first warm run (36+ trials).

| Strategy | Success Rate | Avg Latency | Fallback Rate |
|----------|--------------|-------------|---------------|
| single_model | 50% | 11,920 ms | 0% |
| fixed_routing | 100% | 19,568 ms | 0% |
| dynamic | 100% | 8,416 ms | 0% |

Warm start degraded single_model (83% → 50%) and RW06 success (88.9% → 66.7%).
Dynamic routing selected Lightning for RW01 (exploration triggered) but Bolt
for RW06 (no exploration). Never triggered fallback.

### Corrected live pilot (post-infrastructure-fix)

The only experiment where dynamic fallback triggered and changed the model:
RW06 dynamic (bolt failed → lightning succeeded).

## Key findings

1. **Fixed routing (lightning) outperformed dynamic on latency and reliability
   in cold-start** (6,173 vs 7,747 ms, both 100%).

2. **Dynamic routing's empirical evidence did not overcome static score
   equality** — equal static scores (unknown costs/latencies), low empirical
   weight (0.3), high confidence threshold (10 samples).

3. **Fallback mechanism worked in corrected pilot (RW06) but never activated
   in multi-trial experiments.**

4. **No evidence that empirical routing improved model selection over fixed
   routing.** Fixed routing with lightning was fastest and most reliable.

4. **Exploration (0.1) triggered for RW01 in warm-start but never for RW06.**

## Capability-aware routing

Capability matching happens before execution through `ModelRegistry.candidates`.
The live RW01 request requires `coding` and `structured_output`. A vision task
would additionally require `vision`; a model without that capability is not a
candidate. If no candidate matches, routing returns a structured failure rather
than attempting a blind provider call.

Static metadata and empirical evidence are separate. The deterministic router
uses capability matches, reliability priors, complexity, known cost/latency,
historical provider success, contribution, recency, and sample-size shrinkage.
Unknown cost and latency stay unknown; they are not converted into fabricated
NVIDIA estimates.

## Live evidence versus synthetic evidence

The B01-B20, O01-O10, P01-P10, and baseline commands use deterministic test
doubles. They validate orchestration mechanics, not model intelligence.

Live NVIDIA RW01 artifacts are written separately under:

- `results/realworld_live_agent_trajectories.jsonl`
- `results/realworld_live_agent_events.jsonl`
- `results/performance_memory_live.jsonl`
- `results/realworld_live_summary.json`

The verifier checks the fixture state and registered test result. A model's
final text is never treated as proof. Live cost is `null` when NVIDIA has not
provided a price estimate; token counts and measured latency are still stored.

## Security boundaries

NVIDIA receives only the structured messages and registered tool schemas sent by
the adapter. The model cannot execute shell commands, subprocesses, arbitrary
Python, arbitrary network calls, package installation, deployment, or
unrestricted filesystem access. Tool requests pass through:

```text
model → provider adapter → normalized action → ToolRegistry → schema validation
→ sandboxed execution → structured observation → model
```

Malformed tool arguments, unknown tools, missing provider responses, provider
errors, rate limits, authentication failures, context/request rejection, and
empty responses are recorded as structured failures. Text-only final responses
are accepted only as final candidates; independent verification remains
authoritative and no tool is inferred from free text.

### Security boundary: test.run and unrestricted Python exec

**CRITICAL:** The `test.run` tool uses Python's `exec()` to run registered test
functions. This is **unrestricted Python execution** within the sandbox.

- **Sandbox protection:** `SandboxWorkspace` restricts file operations to the
  fixture directory and blocks credential-like paths (`.env`, `id_rsa`, etc.).
- **NO process isolation:** `exec()` runs in the host Python process. A malicious
  or buggy model-generated test could access process memory, environment
  variables, or import modules.
- **Fail-closed behavior:** Hosted-model execution through `test.run` is now
  **fail-closed** — the campaign infrastructure detects this tool and refuses to
  run live models when `test.run` is available, unless explicitly overridden.

**Deterministic fixture execution (scripted adapters) remains allowed** because
the test functions are registered by the harness, not generated by the model.

**Future hosted repository execution requires an isolated execution environment**
(e.g., gVisor, Firecracker, or a separate container with no host access) before
model-driven `test.run` can be safely enabled.

## Documentation of final evaluation

See `results/final_evaluation/analysis/final_report.md` for the complete
scientific evaluation report, including:

- Infrastructure fixes applied
- Validation results
- Live pilot results (pre/post fix)
- Cold-start and warm-start experiment results
- Routing analysis with evidence utilization calculations
- Failure analysis and root causes
- Recommended next steps
- Credential safety verification
- Final scientific conclusion

## Security boundaries summary

| Boundary | Status |
|----------|--------|
| API keys in logs | ✅ None |
| Authorization headers in trajectories | ✅ None |
| Provider secrets in artifacts | ✅ None |
| Scripted/Live separation | ✅ Verified |
| test.run unrestricted exec | ⚠️ Fail-closed for hosted models |
| Sandbox path restrictions | ✅ Enforced |

## Final engineering report

See `results/final_evaluation/analysis/final_report.md` for the complete
scientific evaluation report, including all experimental results, failure
analysis, and the conservative conclusion that **DIVY's dynamic routing provides
no measurable advantage over fixed routing with the lightning model in the
current configuration**.
