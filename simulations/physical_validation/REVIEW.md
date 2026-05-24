# QA-MAB Physical Validation — Independent Review

**Reviewer:** Subagent (depth 1/2)
**Date:** 2026-05-24
**Files reviewed:** `physical-validation-summary.md`, `CAT10.md`, `qa_mab_physical.py`, `physical_env.py`, `sa_solver_physical.py`, `validate_cat9_final.py`, `runner_physical.py`, `oracle_agent.py`, `agents_physical/oracle_agent.py`

---

## Critical Finding: Cat 9 CORRECTED Fix Is NOT Implemented

**The shared RNG fix described in the README does not exist in the code.**

### What the README Claims
> "Fix: `reset_epoch(p, world_rng=None)` — shared epoch RNG passed from harness"
> "Commit: `53c99d0`"

### What validate_cat9_final.py Actually Does

```python
def run_single(sigma, seed, verbose=True):
    rng = np.random.default_rng(seed + 50000)
    world = AbstractWorld(N=N, K=4, m=20, Z=6, sigma_noise=sigma, seed=seed)
    qa  = QAMABPhysical(world, seed=seed)
    opt = OptimalFixed(world, seed=seed + 20000)
    ...
    for p in range(P):
        qa.reset_epoch(p); opt.reset_epoch(p)   # ← Both use their OWN RNG
```

`reset_epoch(p)` in `qa_mab_physical.py:52` calls `self.world.refresh_epoch(self.rng)` — using the agent's own `self.rng`. `QAMABPhysical.__init__` sets `self.rng = np.random.default_rng(seed)`. So:
- `qa.rng` was seeded with `seed`
- `opt.rng` was seeded with `seed + 20000`
- These are **different generators** → different path topologies per epoch

### What qa_mab_physical.reset_epoch Actually Accepts

```python
def reset_epoch(self, p: int) -> None:
    if p > 0:
        self.theta_hat *= self.epoch_decay
        ...
    self._visit_counts = np.zeros((self.world.N, self.world.K), dtype=int)
    self.world.refresh_epoch(self.rng)   # ← Only accepts self.rng, no world_rng param
```

There is **no `world_rng` parameter** in `reset_epoch`. The fix described in the README is not present in the code at all. The Cat 9 CORRECTED run in `validate_cat9_final.py` still suffers from the RNG separation bug.

**Verdict:** The README documentation is aspirational or describes a different version. The actual validation script still produces non-comparable regret figures.

---

## Bug A (Credit Assignment) — Still Affects Results

Bug A is **unfixed** in `qa_mab_physical.py:189-192`. The credit assignment loop:

```python
for i in np.where(uav_mask)[0]:
    other_theta = theta_contrib - self.theta_hat[i]
    residual_i = observed - other_theta - phi_contrib
    self.theta_hat[i] += self.alpha * (residual_i - self.theta_hat[i])
```

Each UAV `i` in `uav_mask` absorbs the **full** `observed = L_fault[n]`, not `L_fault[n] / num_uavs_in_mask`. If 4 UAVs share responsibility for a flow's fault loss, each one gets credited with the entire loss amount. This biases `θ̂` **upward**.

### Does this invalidate Cat 9?

Cat 9 compares QA-MAB vs OptimalFixed. Both share the same topology. The question is whether the bias causes non-convergence or systematically wrong paths.

**Mechanism:** The biased θ̂ makes the QUBO "see" higher fault costs than reality. This could:
1. Cause QA-MAB to over-penalize paths involving truly faulty UAVs (could be directionally correct, just amplified)
2. Distort the relative ranking of paths if faulty UAVs have different true rates but get the same amplified credit

**Key observation:** Since `epoch_decay` resets θ̂ each epoch, the bias re-accumulates from near-zero. The learning signal still exists — the algorithm can still rank paths correctly (relative ordering), just with inflated magnitudes.

**The bigger risk:** Cat 4 and Cat 5 report θ̂ error reductions of 34-49%. These error magnitudes may be computed against biased estimates. If θ̂ is systematically high, and the error is computed as `||θ̂ − θ*||`, the reported error reduction might be real (convergence is happening) but the absolute values could be misleading if the bias doesn't fully decay between epochs.

**This needs fixing before any quantitative claims about estimation accuracy.**

---

## Cat 10 — Parameter Comparability Issues

### Proposals vs Sweeps: Not a Fair Fight

The comparison in `run_compare_sa.py` (referenced in CAT10.md) sets:

| Solver | Proposals per restart | Notes |
|--------|----------------------|-------|
| `sa_sweep` (bit-flip) | 200 sweeps × 16 bits × 20 reads = **64,000** | Random permutation each temp step |
| `sa_onehot` (route-flip) | 200 iters × 1 flip × 20 restarts = **4,000** | 16× fewer proposals |

The 6/6 seed win for bit-flip could simply be a **compute budget difference**, not an algorithmic superiority. 64,000 vs 4,000 proposals is not a controlled comparison.

### Temperature Schedules Are Incommensurable

`sa_sweep`: linear cooling T∈[2.0, 0.05] over 200 sweeps (one temperature per sweep step)
`sa_onehot`: geometric decay `T *= 0.995` per iteration — **T stays high much longer** (effective temperature profile is much flatter)

These are fundamentally different cooling schedules, not just parameter tunes of the same thing. The claim that bit-flip is "better" due to 6/6 wins conflates proposal count + temperature schedule + encoding all at once.

### sa_solve / sa_onehot / sa_sweep Aliasing Confusion

`sa_solver_physical.py` has:
```python
sa_solve = sa_onehot   # Default alias
```

But CAT10.md says:
> "qa_mab_physical.py — unchanged, continues to use bit-flip SA (`sa_solve` which points to `sa_onehot` on remote, but local file uses `sa_sweep`)"

This is self-contradictory. If `sa_solve = sa_onehot`, then `qa_mab_physical.act()` uses route-flip SA, not bit-flip. But the narrative says local file uses `sa_sweep`. One of these is wrong in the documentation.

---

## Incomplete / Could Give False Confidence

### Cat 1: Parameter Sweeps Skipped
> "Using old model config (UCB=3.0, decay=0.7)"

The old CMAB model's parameters were validated in a different model with different loss dynamics. Using them without re-validation for the physical model is a gap. Cat 9 shows `decay=1.0` is better than 0.7 — but this was only tested in Cat 9, not systematically.

### Cat 6: N=15, 20 Not Completed
> "N=15 and N=20 were interrupted before completion"

The noise robustness story is only confirmed for N≤10. The thesis claim of noise robustness at all N≥20 is not actually validated.

### Cat 8: N=5 Only
Only 1 of 6 N values tested. The "gamma-scaled wins 5/6 N values" claim is based on a single data point.

### Cat 4: "QA-MAB beats Oracle" Claim
> "steps 80–90: −0.97 (QA beats Oracle!)"

This is not a fair comparison. OracleAgent uses SA (same SA solver) but with perfect θ̂=θ*. A random SA run can beat a different SA run by luck. This is SA variance, not QA-MAB superiority. The claim should be removed or reframed.

### Cat 2: 90% Match Rate
The summary says "borderline" and attributes it to ties. That's correct interpretation — but the summary should more clearly state the QUBO is proven correct (gap=0) rather than leading with "90% < 95% threshold".

---

## What Would I Run Next (Priority Order)

1. **Fix validate_cat9_final.py world RNG bug** — The entire point of Cat 9 is comparison with shared topology. Without shared RNG, regret measures world-difficulty differences, not learning quality. This is the most foundational fix.

2. **Fix Bug A (credit assignment)** — The biased credit assignment inflates θ̂. Until fixed, all θ̂ error quantities are unreliable. This affects Cat 4's error reduction claims and Cat 9's θ convergence narrative.

3. **Complete Cat 6 (N=15, 20 noise robustness)** — The noise robustness story is a key selling point. Partial results for N≤10 don't support the full claim.

4. **Fair Cat 10 comparison** — Either (a) match proposal counts (64,000 for route-flip too) or (b) match temperature schedules. The current comparison conflates compute budget with algorithmic quality.

5. **Complete Cat 8** — All 6 N values, not just N=5.

6. **Remove "QA beats Oracle" from Cat 4** — Misleading framing. Say "SA variance causes occasional QA-MAB wins in specific epochs."

---

## Summary Table

| Issue | Severity | Status |
|-------|----------|--------|
| Cat 9 RNG fix not in code | **HIGH** | README ≠ implementation |
| Bug A (credit bias) unfixed | **HIGH** | Affects all error metrics |
| Cat 10 unfair comparison | **MEDIUM** | Proposals + temp schedules differ |
| Cat 6 incomplete (N=15,20) | **MEDIUM** | Key claim unsubstantiated |
| Cat 8 incomplete (5/6 N values) | **MEDIUM** | "5/6" is actually "1/6" |
| Cat 4 "beats Oracle" misleading | **LOW** | SA variance, not learning |
| Cat 1 params from old model | **LOW** | Potential mismatch |
| sa_solve alias confusion | **LOW** | Self-contradictory docs |

**Bottom line:** The validation suite demonstrates the right experiments in principle, but has significant execution gaps. The most urgent fix is the Cat 9 shared RNG bug — without it, all Cat 9 "convergence" results are measuring the wrong thing. Bug A (credit assignment) is the second priority since it poisons all θ̂/φ̂ error metrics. After those two, the incomplete categories (Cat 6, Cat 8) need full runs to support the claims made in the summary.
---

## FIXES APPLIED (2026-05-24)

### Bug A — Credit Assignment (HIGH) ✅ FIXED
**File:** `qa_mab_physical.py` lines ~189-192

**Before:**
```python
observed = float(L_fault[n])
# Each UAV absorbs full loss
for i in np.where(uav_mask)[0]:
    residual_i = observed - other_theta - phi_contrib
```

**After:**
```python
num_uavs = uav_mask.sum()
num_zones = zone_mask.sum()
per_uav_loss = observed / max(num_uavs, 1)  # fair split
per_zone_loss = observed / max(num_zones, 1)
# Each UAV absorbs per_uav_loss
```

### Bug B — decode fallback (MEDIUM) ✅ FIXED
**File:** `sa_solver_physical.py::decode_solution`

**Before:** `chosen[n] = 0` when segment.all==0 (deterministic bias)
**After:** `chosen[n] = rng.integers(0, K)` (random fallback)

### Bug C — L_fault floor (LOW) ✅ FIXED
**File:** `qa_mab_physical.py`

**Before:** `L_fault = np.maximum(L_fault, 0.0)` (clipping negative noise)
**After:** `L_fault = losses - C_coll * collision_counts - prox` (allows negative)

### Bug D — UCB visits floor (LOW) ✅ FIXED
**File:** `qa_mab_physical.py`

**Before:** `ucb_bonus = c / sqrt(max(visits, 1))` (visits=0 == visits=1)
**After:** `ucb_bonus = c / sqrt(max(visits, 1e-6))` (visits=0 gets proper bonus)

### Cat 9 RNG Fix ✅ FIXED
**File:** `validate_cat9_final.py` + `qa_mab_physical.py::reset_epoch`

**Before:** `qa.reset_epoch(p)` uses agent's own RNG → different topologies per epoch
**After:**
```python
shared_epoch_rng = np.random.default_rng(seed + 50000)
epoch_seed = int(shared_epoch_rng.integers(0, 2**63-1))
world_rng = np.random.default_rng(epoch_seed)
qa.reset_epoch(p, world_rng=world_rng)
opt.reset_epoch(p, world_rng=world_rng)
```

### QA-MAB now uses sa_sweep (bit-flip) ✅ CONFIRMED
**File:** `qa_mab_physical.py`

`qa_mab_physical.py` calls `sa_sweep` (bit-flip), NOT `sa_solve` (route-flip).
This confirms bit-flip is the active QA-MAB solver.

