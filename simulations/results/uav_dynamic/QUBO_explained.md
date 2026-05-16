# QUBO Construction in the UAV Dynamic Environment

**Date:** 2026-05-03
**Purpose:** Explain exactly how the QUBO matrix is built, term by term

---

## 1. The Variable Space

We have `N` flows (communication pairs) and `K` candidate paths per flow.

Each flow `n` must select exactly one path `k ∈ {0, ..., K-1}`.

We encode this as a binary vector `x` of length `M = N × K`:
```
x[n·K + k] = 1  ⟺  flow n takes path k
```

The QUBO energy function: `E(x) = x^T Q x`

---

## 2. Diagonal Terms — Linear Costs (Per-Path)

For each variable `(n, k)`, the diagonal entry is:

```
Q[n·K + k, n·K + k] = (Σ_{i ∈ path(n,k)} θ̂[i] + Σ_{z ∈ path(n,k)} φ̂[z]) - λ
```

Where:
- `θ̂[i]` = estimated failure rate of UAV `i` (learned from feedback)
- `φ̂[z]` = estimated interference rate of zone `z` (learned from feedback)
- `path(n,k)` = set of UAVs/zones that path `k` for flow `n` passes through
- `λ` = one-hot penalty (`lambda_onehot = 10.0`)

The `−λ` means: "selecting a path gives a `−λ` energy bonus, making it preferable to selecting nothing."

### UCB Exploration Bonus (v2 only)
When `visit_counts` is provided (v2 with UCB):
```
Q[n·K + k, n·K + k] -= ucb_c / √(visits[n,k])
```
A rarely-visited path gets a larger bonus (lower cost), encouraging exploration even when `θ̂` is flat/wrong.

---

## 3. Off-Diagonal Terms — Constraints and Interactions

### 3a. Same-Flow Constraint (One-Hot)

For flow `n`, paths `k` and `k'` (k < k'):
```
Q[n·K + k, n·K + k'] += λ
Q[n·K + k', n·K + k] += λ
```

**Effect:** If `x[n·K+k] = x[n·K+k'] = 1` (flow picks 2 paths), the energy increases by `2λ`. SA avoids this.

### 3b. Cross-Flow Collision Penalty

For flows `n ≠ l`, paths `k` and `j`:
```
shared_UAVs = path(n,k) ∩ path(l,j)
if shared_UAVs is non-empty:
    Q[n·K+k, l·K+j] += C_coll    (C_coll = 5.0)
    Q[l·K+j, n·K+k] += C_coll
```

**Effect:** If two flows use the same UAV, they get a large energy penalty. SA avoids same-UAV routing.

### 3c. Proximity Interference

Even if flows don't share a UAV, if their paths pass close in space:
```
d = min_pairwise_distance(path(n,k), path(l,j))  # minimum Euclidean distance between any two UAVs on the paths
Q[n·K+k, l·K+j] += exp(−d / d0)   (d0 = 150.0)
Q[l·K+j, n·K+k] += exp(−d / d0)
```

**Effect:** Flows that pass too close get a small exponential penalty. Models electromagnetic interference.

---

## 4. Full Energy Function

```
E(x) = Σ_{n,k} x[nk] · (Σ_{i∈path(n,k)} θ̂[i] + Σ_{z∈path(n,k)} φ̂[z] − λ)   [diagonal]
     + Σ_{n} Σ_{k<k'} 2λ · x[nk] · x[nk']                                [same-flow]
     + Σ_{n≠l} Σ_{k,j} x[nk] · x[lj] · [C_coll·1{shared_UAVs} + exp(−d/d0)]  [cross-flow]
```

SA minimizes this. The minimum corresponds to: selecting one path per flow, avoiding collisions and high-estimate UAVs/zones.

---

## 5. How SA Uses This

1. Build `Q` at each step using current `θ̂, φ̂, visit_counts`
2. Scale by temperature: `Q_scaled = Q / γ(p,t)` — high temperature early = more exploration
3. SA samples from `P(x) ∝ exp(−x^T Q_scaled x / T)` — near-zero T = pure minimization
4. Decode binary solution to `N` path choices

---

## 6. How Learning Works — Residual Credit Assignment

After SA selects paths and the environment returns losses `L[n]`:

**Step 1: Subtract structural components**
```
L_fault[n] = L[n] − C_coll · collision_count[n] − prox_interference[n]
```
Now `L_fault[n] ≈ Σ_{i∈path} θ*[i] + Σ_{z∈path} φ*[z] + noise`, with the structural components removed.

**Step 2: Residual credit assignment (Fix A)**

Instead of pushing every UAV toward the total path loss (which biases healthy UAVs upward), we use **coordinate-descent-style residual decomposition**:
```
for each UAV i on path[n]:
    other_theta = Σ_{j≠i} θ̂[j] on same path
    phi_contrib = Σ φ̂[z] on same path
    residual_i = L_fault[n] − other_theta − phi_contrib
    θ̂[i] ← θ̂[i] + α · (residual_i − θ̂[i])
```
Healthy UAVs (θ*=0) see residual ≈ 0 and stay near 0. Faulty UAVs see residual ≈ θ* and converge correctly.

**Why this matters:** The old uniform credit assignment pushed ALL UAVs on a path toward the TOTAL loss. With 16 healthy UAVs and 4 faulty, the healthy ones accumulated false positive estimates — θ_err grew from 1.52 → 2.04. With residual credit, θ_err decreases: 0.550 → 0.409.

**Key insight:** Uniform credit assignment spreads the total path loss equally across all UAVs. Residual credit assigns each UAV only what remains after accounting for all other contributions — mathematically principled like coordinate descent.

---

## 7. Transfer Learning Across Epochs

`θ̂` and `φ̂` **persist** between epochs (unlike NB3R which resets).

This means: knowledge about which UAVs are likely faulty carries over when topology changes.

**Example:** If UAV #7 was faulty in epoch 1 (high loss when used), it likely remains faulty in epoch 2 even though its position/connections changed — until proven otherwise.

---

## 8. Why Oracle Loses to QA-MAB

Oracle knows exact `θ*, φ*` but all healthy paths look identical (cost = `0 − λ`). SA has no signal to distribute flows across different healthy UAVs → flows pile up → collisions.

QA-MAB's "wrong" uniform credit assignment creates slight cost differences between healthy UAVs → SA can diversify → fewer collisions → lower loss.

**The "error" in θ̂ creates a useful exploration signal that perfect knowledge destroys.**

---

## 9. Parameters Summary

| Parameter | Value | Role |
|-----------|-------|------|
| `λ` (lambda_onehot) | 10.0 | One-hot penalty |
| `C_coll` | 5.0 | Collision avoidance penalty |
| `d0` | 150.0 | Proximity interference decay |
| `α` (alpha) | 0.15 | Learning rate for θ̂, φ̂ |
| `γ₀` (gamma_0) | 2.0 | Temperature scale |
| `a` | 0.5 | Temperature epoch decay exponent |
| `b` | 0.3 | Temperature step decay exponent |
| `ucb_c` (v2) | 3.0 | UCB exploration constant |
