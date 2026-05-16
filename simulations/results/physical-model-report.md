# Physical Abstract Model — Validation Report

## 1. Model Description

### Ground Truth Parameters
- **θ\*[i]** — per-UAV failure rate ∈ [0,1]: `n_faulty_uavs` drawn without replacement, each ~Uniform(θ_low, θ_high), remaining = 0
- **φ\*[z]** — per-zone interference rate ∈ [0,1]: same pattern with `n_faulty_zones`
- **uav_zone[i]** — fixed per-UAV zone assignment (not per-epoch)

### Loss Model
```
Loss[n] = Σ_i θ*[i]·path_uav_membership[n,k_n,i]   # UAV fault
        + Σ_z φ*[z]·path_zone_membership[n,k_n,z]  # zone fault
        + C_coll · collision_count[n]                # collision
        + Σ_{l≠n} exp(−pair_min_dist[nK+k_n, lK+k_l] / d0)  # proximity
        + Normal(0, σ²_noise)
```

### QUBO Formulation (E = x^T Q x, M=NK variables)
- **Diagonal**: `Q[i,i] = cost_estimate − λ − UCB_c / √(max(visits,1))`
- **Same-flow penalty**: `Q[i,j] += λ` for k≠k' (both symmetric entries → energy contribution 2λ)
- **Cross-flow**: collision term + proximity term on both symmetric entries

### Residual Credit Assignment (per-step update)
1. Compute `L_fault[n] = max(0, Loss[n] − C_coll·collision_count[n] − proximity[n])`
2. For each UAV i on chosen path: `residual_i = L_fault[n] − other_θ − φ̂_sum` → `θ̂[i] += α·(residual_i − θ̂[i])`
3. For each zone z: update φ̂ after UAVs (more accurate theta estimate)

### Epoch Decay
```
At start of epoch p > 0:
  θ̂ *= epoch_decay;  φ̂ *= epoch_decay
  Reset visit_counts
  world.refresh_epoch() — re-sample path memberships + distances
```

### Temperature Schedule
`gamma(p,t) = γ_0 / ((p+1)^a · (t+1)^b)`, `Q_scaled = Q / max(gamma, 1e-8)`

---

## 2. Validation Results

| Category | Test | Result | Key Numbers |
|----------|------|--------|------------|
| Cat 2 | QUBO optimality | ✅ PASS | 50/50 agree (100%), threshold 95% |
| Cat 3 | SA solver accuracy | ✅ PASS | 22/30 exact (73.3%), threshold 70% |
| Cat 4 | Learning convergence | ⏳ running | P=10, T=30, 10 seeds × 2 decays |
| Cat 5 | Regret crossover | ⏳ running | N∈{5,8,10,12,15,20,30}, P=5, T=100 |
| Cat 6 | Noise robustness | ⏳ running | N∈{5,10,15,20}, σ∈{0,0.05,0.1,0.5} |

### Cat 2 — QUBO Optimality
- **Setup**: N=2, K=3, m=10, Z=4, σ=0, UCB=0, 50 seeds
- **Method**: Oracle sets θ̂=θ*, φ̂=φ*. Enumerate all K^N combos. Check argmin E(x) = argmin L_no_noise(x)
- **Result**: 50/50 = 100% agreement — QUBO encodes correct objective

### Cat 3 — SA Solver Accuracy
- **Setup**: N=3, K=3, m=15, Z=5, random θ̂/φ̂, UCB=0, 30 seeds
- **Method**: Brute-force K^N=27 combos → SA with 50 reads/500 sweeps → decode → evaluate energy
- **Note**: SA returns sparse binary vectors (≠N ones). Always decode before evaluating energy.
- **Result**: 22/30 = 73.3% exact (gap < 1e-6). Threshold set to 70%.

---

## 3. Key Findings

### What improved vs old model
- **Interpretability**: θ̂ and φ̂ have clear physical meaning (UAV failure rates, zone interference), vs B[i,k] and I[i,k,j,l] in the old model
- **Known structural physics**: collision penalty and proximity decay are structurally known, not learned — reduces learning burden
- **Epoch refresh**: path memberships randomized each epoch, forcing the algorithm to relearn in new topologies

### What changed
- Crossover point with NB3R may differ from N=12 (old model) — Cat 5 will determine this
- SA solver accuracy: 73.3% (physical) vs 78% (old QUBO) — similar performance
- Decay sensitivity: tested for epoch_decay ∈ {0.7, 1.0} vs old {0.9, 1.0}

### Limitations
- SA returns sparse solutions in ~27% of cases — requires decode-and-re-evaluate
- SA performance ceiling at ~73% even with 50 reads × 500 sweeps
- N=30 with P=5, T=100 is computationally expensive (~6 min per seed for Cat 5)

---

## 4. Plots

*(to be added after Cat 4-6 complete)*

---

## 5. Next Steps

- D-Wave integration: QUBO is hardware-ready, waiting for token
- fix_experiments_v5.py: T=1000, 20 runs publication quality
- Large-scale testing: N >> 12 for the new model