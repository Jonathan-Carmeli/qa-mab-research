# Thorough Review of RESEARCH_REPORT_v1.md

**Reviewer:** Claw (Opus subagent)
**Date:** 2026-05-03
**Verdict:** Report has strong core content but contains **factual errors in numerical claims**, a **critical confound in the Oracle comparison**, and **inconsistencies** between the report and supporting documents.

---

## 1. Verified Claims (with Evidence)

### ✅ Correct Numbers

| Claim | Report Value | Actual (from NPZ) | Status |
|-------|-------------|-------------------|--------|
| QA-MAB total loss | 234,361 | 234,360.8 | ✅ Correct |
| Oracle total loss | 254,305 | 254,305.0 | ✅ Correct |
| NB3R total loss | 272,358 | 272,358.4 | ✅ Correct |
| Oracle vs QA-MAB | +8.5% | +8.51% | ✅ Correct |
| NB3R vs QA-MAB | +16.2% | +16.20% | ✅ Correct |
| QA-MAB collision rate | 53.7% | 53.66% | ✅ Correct (rounded) |
| Oracle collision rate | 57.5% | 57.50% | ✅ Correct |
| NB3R collision rate | 63.3% | 63.30% | ✅ Correct |
| Greedy collision rate | 74.0% | 74.00% | ✅ Correct |
| Random collision rate | 80%+ | 81.05% | ✅ Correct |
| θ_err E1 | 0.550 | 0.5504 | ✅ Correct |
| θ_err E10 | 0.409 | 0.4091 | ✅ Correct |
| φ_err E1 | 0.393 | 0.3926 | ✅ Correct |
| φ_err E10 | 0.312 | 0.3118 | ✅ Correct |
| Cumulative regret | −997.2 | −997.2 | ✅ Correct |

### ❌ Incorrect Numbers

| Claim | Report Value | Actual (from NPZ) | Error |
|-------|-------------|-------------------|-------|
| **Greedy total loss** | **313,211** | **306,672.3** | **Off by 6,539 (2.1%)** |
| **Greedy vs QA-MAB** | **+33.7%** | **+30.9%** | **Overstated by 2.8pp** |
| **Random total loss** | **326,000** | **358,222.6** | **Off by 32,223 (9.0%) — MASSIVELY wrong** |
| **Random vs QA-MAB** | **+39.1%** | **+52.9%** | **Understated by 13.8pp** |

**Root cause:** The Greedy and Random numbers appear to be fabricated or taken from a different (unreported) run. They do not match any of the three run logs (run_log.txt, run_log_ucb.txt, run_log_v3_fixed.txt).

### ⚠️ Internal Inconsistency

In Section 5.2, the text says "14% better cumulative loss" for QA-MAB vs NB3R, but the table in the same section says +16.2%. The table is correct; the text is wrong.

Similarly, Section 7 Summary says "7.8% better" for QA-MAB vs Oracle, but the table says +8.5%. The table is correct.

---

## 2. Critical Issues Found

### Issue #1: Oracle Comparison is Confounded (MAJOR)

**The report claims** QA-MAB beats Oracle because Oracle's perfect knowledge creates identical costs for healthy paths, causing SA to cluster flows.

**The actual reason is more nuanced and partly artifactual.** The Oracle agent implementation has TWO disadvantages unrelated to "perfect knowledge":

1. **No temperature scaling:** QA-MAB divides Q by γ(p,t) before passing to SA, effectively controlling exploration vs exploitation. Oracle passes raw Q to SA — it always operates in "cold" mode. This means Oracle's SA is greedier and less likely to explore diverse paths.

2. **No UCB exploration bonus:** QA-MAB adds `-ucb_c / √(visits)` to the diagonal, explicitly encouraging exploration of rarely-visited paths. Oracle doesn't pass `visit_counts` to `build_qubo`, so it never gets this bonus.

**Impact:** The "QA-MAB beats Oracle because imperfect estimates create useful diversity" narrative is partially correct but **overstated**. A fairer test would be: give Oracle the same UCB bonus and temperature schedule, then check if QA-MAB still wins. Without this ablation, the reader cannot distinguish the effect of "imperfect estimates" from "exploration mechanisms."

**Recommendation:** Either (a) run an ablation with Oracle+UCB+temperature, or (b) honestly acknowledge that the Oracle baseline lacks exploration mechanisms and the comparison is not purely about knowledge quality.

### Issue #2: QUBO_explained.md is Internally Inconsistent with Report (MODERATE)

Section 6 of QUBO_explained.md describes **uniform credit assignment**:
```
θ̂[i] ← θ̂[i] + α · (L_fault[n] − θ̂[i])    for all UAVs i in flow n's path
```

But the actual code and the report's Section 3.5 describe **residual credit assignment** (Fix A):
```
residual_i = L_fault[n] − other_theta − φ̂_contribution
θ̂[i] ← θ̂[i] + α · (residual_i − θ̂[i])
```

These are fundamentally different learning rules. The QUBO_explained.md hasn't been updated to reflect Fix A.

### Issue #3: Convergence Narrative is Misleading (MODERATE)

The report says θ_err drops from 0.550 to 0.409, implying convergence. But the actual trajectory is:

```
E1=0.550 → E2=0.476 → E3=0.451 → E4=0.420 → E5=0.414 → E6=0.413 → E7=0.399 → E8=0.397 → E9=0.400 → E10=0.409
```

The error **plateaus around E6-E7 and actually increases slightly** at E9-E10. This is not monotonic convergence — it's a plateau with noise. The report should present this honestly rather than cherry-picking E1 and E10.

Similarly for φ_err: it essentially plateaus after E4-E5 at ~0.31.

### Issue #4: No Statistical Significance Testing (MODERATE)

With 20 seeds, the standard deviations are substantial:
- θ_err at E10: mean=0.409, std=0.084 (coefficient of variation = 21%)
- φ_err at E10: mean=0.312, std=0.061 (CV = 20%)

The regret plot's confidence band spans from approximately −4000 to +2500, meaning QA-MAB doesn't beat Oracle for ALL seeds. No p-values, confidence intervals, or statistical tests are reported.

### Issue #5: Missing Baseline Descriptions (MODERATE)

The report never describes what NB3R, Greedy, or Random actually do. A reader cannot evaluate the comparison without knowing:
- NB3R uses softmax exploration over path weights, reset each epoch (no transfer learning)
- Greedy always selects path k=0 (shortest path) — a trivially weak baseline
- Random selects uniformly at random — another trivially weak baseline

The only non-trivial comparison is QA-MAB vs Oracle and QA-MAB vs NB3R.

---

## 3. Narrative Improvements

### 3.1 Missing Problem Formulation

The report jumps straight into "what we tried" without formally defining:
- The loss model: L[n] = Σ θ*[i] + Σ φ*[z] + C_coll · collisions + proximity_interference + ε
- The optimization objective (minimize cumulative loss)
- What "collision" means precisely (shared UAV between flows)
- The network topology generation process

### 3.2 Section 1 (Static Environment) is Orphaned

Section 1 discusses the static environment results but these are from a completely different experiment. It provides useful motivation but the numbers (SW ratio = 0.1309) are never connected to the dynamic environment metrics. Either expand it into a proper comparison or move it to "Background/Related Work."

### 3.3 "Why not zero-regret" Explanation is Weak

The report says regret ≠ 0 because the environment is non-stationary. But cumulative regret is computed against Oracle **in the same environment** at each step — both agents face the same topology changes. The non-stationarity argument doesn't explain why regret is *negative*. The actual explanation is Issue #1 above.

### 3.4 Within-Epoch Behavior Data

The report claims "plateau after ~50 steps" but provides no within-epoch loss trajectory data. My analysis shows loss is essentially flat within epochs (no improvement from step 1 to step 100), which supports the claim but should be shown explicitly.

### 3.5 Missing: What θ_err and φ_err Actually Measure

The report never specifies the error metric. From code: it's L2 norm `||θ̂ - θ*||₂`. With m=30 UAVs (26 healthy with θ*=0, 4 faulty with θ*∈[0.2,0.4]), this norm can be dominated by errors on healthy UAVs getting small positive estimates. This should be discussed.

---

## 4. Missing Content

1. **Formal problem statement** — objective function, constraints, notation
2. **Loss model specification** — how the environment generates losses
3. **Topology generation** — m=30 UAVs, 1000m×1000m area, 350m communication radius, BFS-connected flows
4. **Baseline agent descriptions** — what each baseline actually does
5. **Statistical tests** — p-values for QA-MAB vs Oracle difference
6. **Ablation studies** — contribution of UCB, temperature, epoch decay, residual credit separately
7. **Within-epoch dynamics** — loss trajectory within a single epoch
8. **Path generation** — K-shortest paths via what algorithm?
9. **Noise model** — Gaussian with σ=0.05 (mentioned in config, never in report)
10. **Oracle fairness discussion** — acknowledge that Oracle lacks UCB + temperature

---

## 5. Are the Conclusions Justified?

| Conclusion | Justified? | Notes |
|------------|-----------|-------|
| θ̂ converges across epochs | **Partially** | Plateaus after E5-6, doesn't reach zero. "Improves" is more accurate than "converges" |
| φ̂ converges across epochs | **Partially** | Same — plateau behavior, not convergence |
| QA-MAB beats NB3R | **Yes** | 16.2% improvement, consistent across the results |
| QA-MAB beats Oracle | **Yes, but explanation is wrong** | The advantage is real but partly due to exploration mechanisms (UCB + temperature) that Oracle lacks, not just "imperfect estimates" |
| More epochs > more steps | **Plausible** | Within-epoch data shows flat loss, but no experiment varying P independently |
| Residual credit is key fix | **Not tested in this run** | No ablation comparing with/without Fix A in the dynamic environment |

---

## 6. Figure Assessment

### Figure 1 (Convergence Plot)
- **Accurate:** Shows θ and φ error decreasing across epochs, consistent with data
- **Issue:** Y-axes have different scales, making visual comparison misleading
- **Issue:** Shaded region not labeled (is it ±1 std? CI?)
- **Missing:** Should show all epochs explicitly, with annotation of plateau behavior

### Figure 2 (Cumulative Loss and Regret)
- **Left panel issue:** Only shows QA-MAB, NB3R, and Oracle — missing Greedy and Random (which are mentioned in the results table)
- **Right panel issue:** The confidence band is ENORMOUS (−4000 to +2500), suggesting the result is not statistically significant for many seeds
- **Report says** "QA-MAB quickly re-establishes its advantage" at epoch boundaries — this isn't clearly visible in the plot

---

## Summary of Required Fixes

### Critical (must fix):
1. ❌ Fix Greedy and Random total loss numbers
2. ❌ Fix percentage differences for Greedy and Random
3. ❌ Fix text inconsistencies ("14%" → "16.2%", "7.8%" → "8.5%")
4. ❌ Acknowledge Oracle comparison confound (missing UCB + temperature)

### Important (should fix):
5. Add formal problem statement and loss model
6. Describe all baseline agents
7. Add statistical significance analysis
8. Update QUBO_explained.md Section 6 to match Fix A
9. Honest discussion of convergence plateau behavior
10. Label figure confidence bands

### Nice to have:
11. Ablation studies (UCB contribution, temperature contribution, etc.)
12. Within-epoch loss dynamics figure
13. Fair Oracle comparison (with UCB + temperature)
