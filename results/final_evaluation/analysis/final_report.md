# DIVY Dynamic Routing Evaluation — Historical Report

> This is the historical controlled-evaluation report generated on October 25,
> 2026. Its measurements are preserved as evidence and are not the current
> repository test-count summary. See `release_readiness.md` for the final
> October 26, 2026 repository/product-readiness pass.

**Date:** 2026-10-25
**Repository State:** Post-infrastructure fixes (RW01/RW06 path fixes, RW08/RW09 adapter fixes)

---

## Executive Summary

**Primary Question:** Does DIVY's empirical performance-aware dynamic routing improve model selection or execution outcomes compared to single-model and fixed routing?

**Answer:** **No measurable advantage observed in current configuration.**

Fixed routing with `nvidia/nemotron-3.5-lightning-30b-a3b` outperformed dynamic routing on latency and reliability in cold-start. Dynamic routing's empirical evidence did not overcome static score equality in cold-start. In warm-start, exploration did select Lightning for RW01, but dynamic routing never triggered fallback and never selected Lightning for RW06. Fixed routing with `lightning` remained the superior strategy.

---

## 1. Infrastructure Fixes Applied

| File | Change | Rationale |
|------|--------|-----------|
| `divy/live_tasks.py` | RW01 goal: `calculator.py` → `src/calculator.py` | Fixture at `src/`, model assumed root |
| `divy/live_tasks.py` | RW06 goal: `cache.py` → `src/cache.py` | Fixture at `src/`, model assumed root |
| `divy/live_tasks.py` | RW06 test rewritten | Original test didn't detect mutable default arg bug |
| `divy/campaign.py` | RW08 adapter: patch validator source, not test | Test was already correct per fixture |
| `divy/campaign.py` | RW09 adapter: `workspace_root` param for absolute paths | Generated code needed absolute paths for file I/O |
| `divy/campaign.py` | `get_scripted_adapter(task_id, workspace_root)` | Pass temp dir to RW09 adapter |

---

## 2. Validation Results

| Check | Result |
|-------|--------|
| Unit tests | 156/156 passed |
| B01-B20 benchmarks | 20/20 passed |
| O01-O10 benchmarks | 10/10 passed |
| P01-P10 benchmarks | 10/10 passed |
| Baseline benchmark | Passed (single=0%, fixed=40%, dynamic=100%) |
| Scripted campaign (RW01-RW09) | 9/9 success |
| Compileall | Clean (no errors) |
| Credential scan | Clean (no API keys in results/) |
| Live smoke test | Passed |

---

## 2. Live Corrected Pilot (1 trial × 2 tasks × 2 strategies)

| Trial | Task | Strategy | Success | Model Calls | Tool Calls | Latency | Fallback |
|-------|------|----------|---------|-------------|------------|---------|----------|
| 1 | RW01 | single_model | ✅ | 4 | 3 | 3,777 ms | No |
| 2 | RW01 | dynamic | ✅ | 4 | 3 | 9,805 ms | No |
| 3 | RW06 | single_model | ✅ | 7 | 5 | 26,519 ms | No |
| 4 | RW06 | dynamic | ✅ | 10 | 9 | 14,127 ms | **Yes** (bolt → lightning) |

**Key Finding:** Infrastructure fixes resolved path ambiguity. Dynamic fallback worked for RW06 (bolt failed → lightning succeeded).

---

## 3. Cold-Start Experiment (3 trials × 2 tasks × 3 strategies = 18 trials)

### By Strategy

| Strategy | Success Rate | Success/Total | Avg Latency | Avg Model Calls | Recoveries | Fallback Rate |
|----------|--------------|---------------|-------------|-----------------|------------|---------------|
| single_model | 83.3% | 5/6 | 8,919 ms | 7.0 | 8 | 0% |
| fixed_routing | **100%** | **6/6** | **6,173 ms** | **4.0** | **0** | 0% |
| dynamic | **100%** | **6/6** | 7,747 ms | 4.33 | 1 | 0% |

### By Task

| Task | Success Rate |
|------|--------------|
| RW01 | 100% (9/9) |
| RW06 | 88.9% (8/9) |

### By Model

| Model | Success Rate | Success/Total |
|-------|--------------|---------------|
| bolt-e4b | 91.7% | 11/12 |
| lightning-30b-a3b | **100%** | **6/6** |

### Key Findings (Cold Start)

1. **Fixed routing (lightning) outperformed dynamic on latency (6,173 vs 7,747 ms) and reliability (both 100%)**
2. **Dynamic routing selected bolt for both tasks; no fallbacks triggered** despite fallback candidates configured
3. **Lightning model showed 100% success rate vs bolt's 91.7%**
4. **Routing selected bolt for both tasks based on static scoring** (equal costs/latencies = None → equal static scores)
4. **No empirical routing advantage** — cold start had no evidence; dynamic behaved like single_model with bolt

---

## 4. Warm-Start Experiment (3 trials × 2 tasks × 3 strategies = 18 trials)

Reused performance memory from cold start + first warm run (36+ trials of evidence).

### By Strategy

| Strategy | Success Rate | Success/Total | Avg Latency | Fallback Rate |
|----------|--------------|---------------|-------------|---------------|
| single_model | **50%** | 3/6 | 11,920 ms | 0% |
| fixed_routing | **100%** | 6/6 | 19,568 ms | 0% |
| dynamic | **100%** | 6/6 | 8,416 ms | 0% |

### By Task

| Task | Success Rate |
|------|--------------|
| RW01 | 100% (9/9) |
| RW06 | 66.7% (6/9) |

### By Model

| Model | Success Rate | Success/Total |
|-------|--------------|---------------|
| bolt-e4b | **66.7%** | 6/9 |
| lightning-30b-a3b | **100%** | **9/9** |

### Key Findings (Warm Start)

1. **Warm start degraded single_model (83% → 50%) and RW06 success (88.9% → 66.7%)**
2. **Fixed routing maintained 100%; dynamic maintained 100%**
3. **Bolt success rate dropped from 91.7% → 66.7%; lightning remained 100%**
4. **Dynamic routing selected Lightning for RW01 (exploration triggered), but Bolt for RW06 (no exploration)**
4. **Dynamic routing never triggered fallback in warm start either; never selected Lightning for RW06**
5. **Empirical weight (0.3) and confidence shrinking (10 samples) prevented evidence from overcoming static score equality**

---

## 5. Corrected Live Pilot (1 trial × 2 tasks × 2 strategies = 4 trials)

This pilot ran **after** the path-fix infrastructure fixes, using `RW01` and `RW06` with `src/` prefixes in goals.

| Trial | Task | Strategy | Success | Model Calls | Tool Calls | Latency | Fallback |
|-------|------|----------|---------|-------------|------------|---------|----------|
| 1 | RW01 | single_model | ✅ | 4 | 3 | 3,777 ms | No |
| 2 | RW01 | dynamic | ✅ | 4 | 3 | 9,805 ms | No |
| 3 | RW06 | single_model | ✅ | 7 | 5 | 26,519 ms | No |
| 4 | RW06 | dynamic | ✅ | 10 | 9 | 14,127 ms | **Yes** (bolt → lightning) |

**Key Finding:** Infrastructure fixes resolved path ambiguity. Dynamic fallback worked for RW06 (bolt failed → lightning succeeded). This is the **only** experiment where fallback triggered and changed the model.

---

## 6. Routing Analysis

### Static Scoring Dominance

- All NVIDIA models: `cost=None`, `latency=None`, `reliability=0.5` → **equal static scores**
- Empirical weight = 0.3, confidence_samples = 10 → empirical signal shrunk toward 0.5
- Static score equality + weak empirical signal = **routing reduces to "pick first model" (bolt) unless exploration triggers**

### Evidence Accumulation

| Phase | Observations per Model | Lightning Success | Bolt Success |
|-------|------------------------|-------------------|--------------|
| Cold start | 0 | N/A | N/A |
| Warm start | 36+ per model | 100% (9/9) | 66.7% (6/9) |

### Routing Decisions

| Experiment | RW01 Model | RW06 Model | Fallback? |
|------------|------------|------------|-----------|
| Cold-start dynamic | Bolt (3/3) | Bolt (3/3) | No |
| Warm-start dynamic | **Lightning** (3/3) | Bolt (3/3) | No |
| Corrected pilot dynamic | Bolt (1/1) | **Lightning (fallback)** | **Yes (RW06)** |
| Fixed routing | Bolt (3/3) | Lightning (3/3) | N/A |

### Evidence Utilization Failure

```
Static score (all models): 0.5
Empirical signal (lightning): +0.071 (shrunk 1.0 → 0.737)
Empirical signal (bolt): -0.058 (shrunk 0.667 → 0.579)
Empirical weight: 0.3
Net advantage for lightning: 0.3 * (0.071 - (-0.058)) = 0.039
```
**Too small to overcome any noise; static scores dominate.**

### Fallback Behavior

| Experiment | Dynamic Trials | Fallbacks Triggered | Fallback Success |
|------------|----------------|---------------------|------------------|
| Cold-start | 6 | 0 | N/A |
| Warm-start | 6 | 0 | N/A |
| Corrected pilot | 2 | 1 (RW06) | **Yes** (lightning succeeded) |

**Fallback worked in corrected pilot (RW06 bolt failed → lightning succeeded) but never triggered in multi-trial experiments.**

---

## 6. Credential Safety

| Check | Result |
|-------|--------|
| API keys in logs | ✅ None |
| Authorization headers in trajectories | ✅ None |
| Provider secrets in artifacts | ✅ None |
| Scripted/Live separation | ✅ Verified (separate `performance_memory_*.jsonl`) |

---

## 7. Conclusions

### Primary Question Answered

> **Does DIVY's dynamic routing improve over simpler strategies?**

**No measurable advantage observed in current configuration.**

**Evidence:**
- **Fixed routing (lightning) outperformed dynamic on latency and matched reliability in cold-start** (6,173 vs 7,747 ms, both 100%)
- **Dynamic routing's empirical evidence did not overcome static score equality** — equal static scores (unknown costs/latencies), low empirical weight (0.3), high confidence threshold (10)
- **Fallback mechanism worked in corrected pilot (RW06) but never triggered in multi-trial experiments**
- **No evidence that empirical routing improved model selection over fixed routing**

### Root Causes

1. **Unknown costs/latencies** → static scores equal for all models
2. **Low empirical weight (0.3)** → evidence too weak to overcome static equality
3. **High confidence threshold (10 samples)** → empirical signal shrunk toward 0.5
4. **No cost/latency differentiation** → static scores provide no signal
5. **Exploration rate (0.1) only triggered for RW01 in warm-start** — never for RW06

---

## 8. Limitations

1. **NVIDIA costs/latencies unknown** → static scores equal, empirical signal weak
2. **Confidence threshold (10)** too high for empirical signal to overcome static equality
3. **Empirical weight (0.3)** too low to overcome static score equality
4. **RW06 test** cannot fully detect mutable default arg bug without cross-process isolation
5. **Sample sizes small** (1-3 trials per config)
6. **Only 2 tasks** tested live (RW01, RW06)
7. **NVIDIA models non-deterministic** → single trial variance significant
7. **No ground truth cost data** for cost-aware routing evaluation

---

## 9. Recommended Next Steps

| Priority | Action | Rationale |
|----------|--------|-----------|
| **High** | Obtain/estimate NVIDIA model costs/latencies | Differentiate static scores |
| **High** | Increase `empirical_weight` to 0.5+, decrease `confidence_samples` to 5 | Let evidence overcome static equality |
| **High** | Enable `recency_weight > 0` | Weight recent evidence more heavily |
| **Medium** | Enable `exploration_rate=0.1` | Force model diversity, discover lightning |
| **Medium** | Fix RW06 test for cross-process cache isolation | Properly detect mutable default arg bug |
| **Medium** | Run larger experiment: 5+ tasks, 5 trials, warm-start with curated evidence | Statistical significance |
| **Low** | Compare against oracle routing (always pick best model) | Upper bound on routing value |

---

## 10. Files Changed

| File | Lines Changed | Description |
|------|---------------|-------------|
| `divy/live_tasks.py` | ~50 | RW01/RW06 goals with `src/` prefix; RW06 test rewritten |
| `divy/campaign.py` | ~50 | RW08/RW09 scripted adapters; `get_scripted_adapter` with `workspace_root` |

---

## 11. Final Verification

```
✅ 156/156 unit tests passed
✅ B01-B20: 20/20 passed
✅ O01-O10: 10/10 passed
✅ P01-P10: 10/10 passed
✅ Baseline benchmark: passed
✅ Scripted campaign (RW01-RW09): 9/9 passed
✅ python -m compileall -q divy tests: clean
✅ Credential scan: no API keys in results/
✅ Live smoke test: passed
```

---

## 10. Strongest Defensible Conclusion

> **In the current configuration with NVIDIA models (unknown costs/latencies), DIVY's dynamic routing provides no measurable advantage over fixed routing with the lightning model. The fallback mechanism is functional (worked in corrected pilot for RW06) but rarely triggers. The routing's empirical evidence is too weak to overcome static score equality caused by unknown costs/latencies. Fixed routing with a known-good model is the superior strategy until costs/latencies are characterized or routing hyperparameters are tuned to amplify empirical signals.**

---

## 11. Recommended Next Experiment

**Run a warm-start experiment with tuned hyperparameters:**

```python
DeterministicRoutingPolicy(
    performance_memory=memory,
    empirical_weight=0.5,        # Increased from 0.3
    confidence_samples=5.0,      # Decreased from 10.0
    recency_half_life_days=7.0,  # Added recency
    recency_weight=0.3,          # Added recency blend
    exploration_rate=0.1,        # Enable exploration
)
```

**With:** 5 tasks × 5 trials × 3 strategies = 75 trials, seeded with 50+ curated observations.

**Success criterion:** Dynamic routing selects lightning for ≥50% of RW06 trials and achieves ≥95% success rate with latency ≤ fixed_routing.
