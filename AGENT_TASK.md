# Agent Task: Upgrade QA-MAB to Physical Abstract Model (θ̂/φ̂)

> **How to use this document:**  
> Read it fully, then implement every step in order. Run each validation before moving to the next.  
> Work on branch `claude/review-thesis-simulation-bU7tE` in BOTH repositories:
> - `jonathan-carmeli/qa-mab-research` (primary — most new files go here)
> - `jonathan-carmeli/thesis_brain` (secondary — add abstract_env alongside existing code)
>
> Do **not** delete any existing files. Do **not** create a PR unless asked.

---

## 1. Background & Goal

The current QA-MAB simulation in `simulations/simulation_core.py` + `simulations/qa_mab.py`  
works on a fully abstract CMAB model:
- Hidden: `B[i,k]` (base utilities) + `I[i,k,j,l]` (interference tensor)
- Algorithm learns BOTH `u_hat` (≈B) and `I_hat` (≈I) from throughput observations alone

**We are upgrading to a physical-abstract model where:**
- Hidden (to learn): `θ*[i]` = per-UAV failure rate, `φ*[z]` = per-zone interference rate
- Known (structural physics): interference tensor `I[n,k,l,j]` = collision penalty + proximity decay
- The algorithm focuses learning effort only on `θ̂` and `φ̂`

The physical structure is **abstract** (no real graph or K-shortest-paths):  
path memberships and distances are **random matrices sampled once per epoch**.

Reference implementation already exists in `thesis_brain/simulation/src/uav_routing/`:  
- `agents/qamab_agent.py` — residual credit assignment + epoch decay + UCB  
- `qubo.py` — exact QUBO we need  
- `ground_truth.py` — θ*/φ* sampling  
Read these files first. Port their logic, do not rewrite from scratch.

---

## 2. Mathematical Contract (implement exactly this)

### 2.1 Ground Truth (environment)

```
Parameters: N flows, K paths/flow, m UAVs, Z zones

HIDDEN:
  θ*[i] ∈ [0,1]  for i=0..m-1
      n_faulty_uavs UAVs drawn without replacement, each θ*[i] ~ Uniform(0.20, 0.40)
      remaining UAVs: θ*[i] = 0

  φ*[z] ∈ [0,1]  for z=0..Z-1
      n_faulty_zones zones drawn without replacement, each φ*[z] ~ Uniform(0.20, 0.40)
      remaining zones: φ*[z] = 0

KNOWN (randomised once per epoch, then fixed for that epoch):
  path_uav_membership[n, k, i] ∈ {0,1}   shape (N, K, m)
      For each (n,k): draw Uniform(uavs_per_path_min, uavs_per_path_max) UAVs without replacement

  uav_zone[i] ∈ {0..Z-1}   shape (m,)  — fixed for entire simulation (not per-epoch)
      uav i belongs to zone uav_zone[i]

  path_zone_membership[n, k, z] ∈ {0,1}   shape (N, K, Z)
      path_zone_membership[n,k,z] = OR over {uav_zone[i] == z AND path_uav_membership[n,k,i]}

  pair_min_dist[a, b]   shape (N*K, N*K), symmetric, a = n*K+k, b = l*K+j
      = 0.0  if (path_uav_membership[n,k] AND path_uav_membership[l,j]).any()
      = Uniform(0, area_size)  otherwise
      pair_min_dist[a,a] = 0
```

### 2.2 Loss Model

```
Loss[n] = Σ_i  θ*[i] · path_uav_membership[n, k_n, i]         # UAV fault
        + Σ_z  φ*[z] · path_zone_membership[n, k_n, z]         # zone fault
        + C_coll · |{l ≠ n : (uav_membership[n,k_n] & uav_membership[l,k_l]).any()}|  # collision
        + Σ_{l≠n}  exp(−pair_min_dist[n*K+k_n, l*K+k_l] / d0)   # proximity
        + Normal(0, σ²_noise)                                   # observation noise
```

### 2.3 QUBO

```
Variables: x[n*K+k] ∈ {0,1},  M = N*K total,  energy E = x^T Q x

Diagonal Q[i,i]  where i = n*K+k:
    cost_estimate = Σ_{u: uav_mask} θ̂[u]  +  Σ_{z: zone_mask} φ̂[z]
    Q[i,i] = cost_estimate  −  λ  −  UCB_c / sqrt(max(visit_counts[n,k], 1))

Same-flow penalty (i=n*K+k, j=n*K+k', k≠k'):
    Q[i,j] += λ      (Q[j,i] += λ)   → energy contribution 2λ

Cross-flow structural interference (i=n*K+k, jj=l*K+j, n≠l, jj > i):
    contrib = 0
    if (path_uav_membership[n,k] AND path_uav_membership[l,j]).any():
        contrib += C_coll
    contrib += exp(−pair_min_dist[i,jj] / d0)
    Q[i,jj] += contrib ;  Q[jj,i] += contrib
```

### 2.4 Residual Credit Assignment (update step)

```
1. compute: collision_count[n], proximity[n]  (from known structural physics)
2. L_fault[n] = max(0,  Loss[n] − C_coll·collision_count[n] − proximity[n])
3. for each flow n:
     for each UAV u on chosen path:
         other_θ = θ̂_sum_on_path − θ̂[u]
         residual_u = L_fault[n] − other_θ − φ̂_sum_on_path
         θ̂[u] += α · (residual_u − θ̂[u])
     for each zone z on chosen path (after θ̂ updated):
         other_φ = φ̂_sum_on_path − φ̂[z]
         residual_z = L_fault[n] − θ̂_sum_updated − other_φ
         φ̂[z] += α · (residual_z − φ̂[z])
4. clip(θ̂, 0, 1) ;  clip(φ̂, 0, 1)
```

### 2.5 Epoch Decay + Refresh

```
At START of each epoch p:
    if p > 0:
        θ̂ *= epoch_decay      (default 0.7)
        φ̂ *= epoch_decay
    Reset visit_counts to zeros
    world.refresh_epoch(rng)   ← re-samples path memberships + distances
    θ* and φ* PERSIST (same across all epochs for one seed)
```

### 2.6 Temperature Schedule

```
gamma(p, t) = gamma_0 / ((p+1)^a · (t+1)^b)
Q_scaled = Q / max(gamma, 1e-8)
```

---

## 3. Default Parameters

```python
N = 3;  K = 4;  m = 15;  Z = 6
uavs_per_path_min = 2;  uavs_per_path_max = 5
area_size = 1000.0
n_faulty_uavs = 4;  n_faulty_zones = 2
theta_low, theta_high = 0.20, 0.40
phi_low,   phi_high   = 0.20, 0.40
C_coll = 5.0;  d0 = 150.0;  sigma_noise = 0.05
alpha = 0.15;  ucb_c = 3.0;  epoch_decay = 0.7
lambda_onehot = 10.0;  gamma_0 = 2.0;  a = 0.5;  b = 0.3
sa_sweeps = 200;  sa_n_reads = 20;  sa_T_init = 2.0;  sa_T_final = 0.05
```

---

## 4. Files to Create

### 4.1 Repository: `qa-mab-research`

#### `simulations/physical_env.py`
Class `AbstractWorld(N, K, m, Z, n_faulty_uavs, n_faulty_zones, ..., seed)`:
- On init: sample θ*, φ*, uav_zone; generate path memberships + pair_min_dist
- `refresh_epoch(rng)` — re-sample memberships + distances; θ*/φ* unchanged
- `compute_losses(chosen_paths, C_coll, d0, sigma_noise, rng)` — returns (N,) array

Dataclass `AbstractPathSet`: fields `N, K, m, Z, path_uav_membership (N,K,m bool), path_zone_membership (N,K,Z bool)`.

#### `simulations/sa_solver_physical.py`
Copy of `thesis_brain/simulation/src/uav_routing/sa_solver.py` verbatim.

#### `simulations/qa_mab_physical.py`
Class `QAMABPhysical(world, C_coll, d0, sigma_noise, alpha, ucb_c, epoch_decay,
                    lambda_onehot, gamma_0, a, b, sa_sweeps, sa_n_reads, sa_T_init, sa_T_final, seed)`.  
Public attributes: `theta_hat (m,)`, `phi_hat (Z,)`.  
Methods: `reset_epoch(p)`, `build_qubo()`, `act(t, p)`, `update(chosen, losses)`, `run(P, T, rng)`.

#### `simulations/agents_physical/__init__.py`  (empty)

#### `simulations/agents_physical/random_agent.py`
Class `RandomAgent(world, seed)`. `act(t,p)` returns random K-choice per flow. `update` is no-op.

#### `simulations/agents_physical/nb3r_agent.py`
Class `NB3RAgent(world, alpha=0.3, tau0=0.1, delta_tau=0.05, seed=42)`.  
Weights `W[n,k]` (init 0). Softmax over W (higher W = lower loss preference).  
Update: `collaborative = -losses[n] + sum(-losses[l] for l != n)`;  
`W[n, chosen[n]] = (1-alpha)*W[n, chosen[n]] + alpha*collaborative`.  
τ += delta_tau. W persists across epochs (no reset).

#### `simulations/agents_physical/oracle_agent.py`
Same as QAMABPhysical but `theta_hat = theta_star`, `phi_hat = phi_star` (never updated). No UCB. epoch_decay=1.0.

#### `simulations/agents_physical/optimal_agent.py`
Enumerates all K^N assignments, picks argmin of `expected_loss_no_noise` (no noise term, true θ*/φ*).  
Raise ValueError if K^N > 5000.

#### `simulations/runner_physical.py`
`run_experiment(N, K, m, Z, P, T, n_seeds, base_seed, agents=None, **kwargs) -> dict`  
Runs all agents (default: QA-MAB, NB3R, Random, Oracle, Optimal) on same seeds.  
Returns `{agent_name: {losses, theta_err, phi_err, chosen}}`.  
Saves raw + summary CSV to `results/physical_run/`.

### 4.2 Repository: `thesis_brain`

#### `simulation/src/uav_routing/abstract_env.py`
Verbatim copy of `physical_env.py`.

#### `simulation/src/uav_routing/abstract_runner.py`
Mirror of `runner.py` using AbstractWorld instead of generate_topology+enumerate_paths.

#### `simulation/src/uav_routing/agents/qamab_agent.py` — minimal edit only
Change `reset_epoch(self, topology, pathset)` signature to `reset_epoch(self, topology_or_world, pathset=None)`.  
If `pathset is None`: `self._pathset = topology_or_world.pathset; self._pair_min_dist = topology_or_world.pair_min_dist`.  
Else: existing logic unchanged. Apply same 1-line adaptation to nb3r, oracle, optimal, random agents.

---

## 5. Validation Suite — All 6 Categories

All scripts in `qa-mab-research/simulations/`. Save results to `simulations/results/`.

---

### Cat 1: `validate_cat1_param_sweep_physical.py` — Parameter Sweep

**Goal:** Find near-optimal hyperparameters for the new physical model (automated, not manual).  
**Background:** Old model was tuned empirically (ucb_c=3.0, sa_sweeps=200, decay=1.0). Validate these still hold for the physical model.

**Setup:** N=10, K=4, m=20, Z=6. P=10, T=50. 10 seeds per config.

**Sweep 1 — UCB constant:**  
`ucb_c ∈ {0.0, 1.0, 2.0, 3.0, 5.0, 10.0}`  
Metric: mean final loss (last 10 steps) + theta_err at epoch 10.

**Sweep 2 — epoch decay:**  
`epoch_decay ∈ {0.5, 0.7, 0.9, 1.0}`  
Metric: same.

**Sweep 3 — SA effort:**  
`sa_sweeps ∈ {50, 100, 200, 500}` (with sa_n_reads fixed at 10)  
Metric: final loss + wall time per step.

**Output:**
- `results/validation_cat1_physical/result.json` with best config per sweep + full tables
- `results/validation_cat1_physical/sweep_ucb.csv`, `sweep_decay.csv`, `sweep_sa.csv`
- 3 line plots: metric vs hyperparameter value (PNG)

**PASS criterion:** Best ucb_c gives ≥10% lower final loss than ucb_c=0. Best decay identified.  
Report recommended config in JSON `"recommended": {"ucb_c": X, "epoch_decay": Y, "sa_sweeps": Z}`.

---

### Cat 2: `validate_cat2_physical.py` — QUBO Optimality

**Goal:** Verify QUBO formula encodes the correct objective.  
**Setup:** N=2, K=3, m=10, Z=4. sigma_noise=0, UCB=0. 50 seeds.  
**Procedure per seed:**
1. AbstractWorld(seed). Set θ̂=θ*, φ̂=φ* (oracle). Build Q with UCB=0.
2. Enumerate all K^N=9 path assignments.
3. Compute E(x)=x^TQx and L_no_noise(x) for each.
4. Check: argmin E(x) == argmin L_no_noise(x) (gap must be 0 if they disagree).

**PASS:** ≥95% agree (ties with gap=0 count as pass, as in old cat2).  
**Output:** `results/validation_cat2_physical/result.json` + `result.csv`.

---

### Cat 3: `validate_cat3_physical.py` — SA Solver Accuracy

**Goal:** Verify SA finds the QUBO minimum reliably.  
**Setup:** N=3, K=3, m=15, Z=5. Random θ̂/φ̂. UCB=0. 30 seeds.  
**Procedure:** Brute-force argmin over K^N=27. Run `sa_solve_physical`. Compute energy gap.  
**PASS:** success_rate ≥78% exact (same threshold as old cat3 which got 78%; this is the real SA limit).  
**Output:** `results/validation_cat3_physical/result.json` + `result.csv`.

---

### Cat 4: `validate_cat4_physical.py` — Learning Dynamics

**Goal:** θ̂ and φ̂ converge toward θ* and φ*.  
**Setup:** N=3, K=4, m=15, Z=6. P=25, T=40. 15 seeds.  
**Test both:** epoch_decay ∈ {0.7, 1.0} (note: old model used {0.9, 1.0}; new default is 0.7).  
**Procedure:**
1. Per seed: AbstractWorld + QAMABPhysical.
2. Per epoch: record theta_err=||θ̂−θ*||_1/m, phi_err, gap vs oracle (brute force over K^N).
3. Run T steps, refresh epoch, reset agent.

**PASS per decay:** gap_final < gap_initial OR theta_err_final < theta_err_initial.  
**Output:** `results/validation_cat4_physical/result.json`, per-decay CSVs, PNG plots (theta_err vs epoch, gap vs epoch).

---

### Cat 5: `validate_cat5_regret_convergence_physical.py` — Scaling / Regret

**Goal:** Verify QA-MAB advantage over NB3R grows with N (find crossover point).  
**Background:** Old model showed crossover at N=12 (p<0.001). Check if physical model shows same or different crossover.

**Setup:** N ∈ {5, 8, 10, 12, 15, 20, 30}. K=4, m=20, Z=6. P=5 epochs, T=100 steps. 20 seeds.

**Procedure per N:**
1. Run QAMABPhysical and NB3RAgent on same seeds.
2. Record per-epoch mean loss (lower is better).
3. Compute final_loss[seed] for each agent (mean of last 10 steps).
4. Paired t-test: QA-MAB loss vs NB3R loss per seed.
5. QA-MAB “wins” a seed if QA_loss < NB3R_loss.

**PASS:** QA-MAB win rate ≥80% at some N (crossover identified).  
Report:
- Win rate per N
- p-value per N
- Crossover N (first N where win_rate ≥80% and p<0.05)

**Output:**
- `results/validation_cat5_physical/result.json` with crossover_N + per-N stats
- `results/validation_cat5_physical/result.csv` with columns: N, qa_mean, nb3r_mean, win_rate, p_value
- PNG: win_rate vs N, mean_loss vs N (with 95% CI bars)

---

### Cat 6: `validate_cat6_noise_robustness_physical.py` — Stochastic Noise

**Goal:** Verify QA-MAB advantage persists under increasing observation noise.  
**Background:** Old model showed QA-MAB wins at N≥15 for all sigma, crossover shifts with sigma.

**Setup:** N ∈ {5, 10, 15, 20}. K=4, m=20, Z=6. sigma_noise ∈ {0.0, 0.05, 0.1, 0.5}.  
P=5 epochs, T=100 steps. 15 seeds per (N, sigma) combination.

**Procedure:** Same as Cat 5 but outer loop over sigma values.  
For each (N, sigma): run QA-MAB, NB3R, Random. Record win rate QA vs NB3R.

**Output table (reproduce old cat6 format):**
```
         sigma=0  sigma=0.05  sigma=0.1  sigma=0.5
N=5      TIE      TIE         TIE        TIE
N=10     ?        ?           ?          ?
N=15     ?        ?           ?          ?
N=20     ?        ?           ?          ?
```
(Fill with QA wins / NB3R wins / TIE based on win_rate: QA if >60%, NB3R if <40%, else TIE)

**PASS:** QA-MAB wins at (N=20, sigma=0.0) and at (N=20, sigma=0.05).  
**Output:** `results/validation_cat6_physical/result.json` + `result_table.csv` + heatmap PNG.

---

### Baseline Comparison: `validate_baseline_comparison.py`

**Goal:** Compare new physical model vs old NetworkEnvironment model head-to-head.  
**Setup:** N=10, K=4, m=30, Z=9. P=10, T=100. 20 seeds.

**Agents on NEW model:** QA-MAB Physical, NB3R Physical, Random Physical, Oracle Physical.  
**Agents on OLD model:** QA-MAB legacy (`qa_mab.py`), NB3R legacy (`nb3r.py`), Random legacy.

**Metrics:** mean final loss, cumulative loss over all P*T steps, theta_err trajectory (new model only).

**PASS:** QA-MAB Physical final loss ≤ NB3R Physical ≤ Random Physical.  
**Output:** `results/validation_baseline/result.json` + `summary.csv`  
3 PNGs: `scaling.png` (N ∈ {5,10,20,30}), `convergence.png` (loss vs step at N=10), `theta_convergence.png`.

---

## 6. Run Script

Create `simulations/run_validation_physical.sh`:

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== Cat 1: Parameter Sweep ===";     python validate_cat1_param_sweep_physical.py
echo "=== Cat 2: QUBO Optimality ===";      python validate_cat2_physical.py
echo "=== Cat 3: SA Solver Accuracy ===";   python validate_cat3_physical.py
echo "=== Cat 4: Learning Dynamics ===";    python validate_cat4_physical.py
echo "=== Cat 5: Regret Convergence ===";   python validate_cat5_regret_convergence_physical.py
echo "=== Cat 6: Noise Robustness ===";     python validate_cat6_noise_robustness_physical.py
echo "=== Baseline Comparison ===";         python validate_baseline_comparison.py
echo ""; echo "All done. Results in simulations/results/"
```

---

## 7. Final Report

Create `simulations/results/physical-model-report.md` in `qa-mab-research` with:

1. **Model Description** — equations from §2 in LaTeX-style markdown
2. **Validation Results** — one table per category, PASS/FAIL + key numbers
3. **Comparison Table** — old model vs new model side by side for cat2/3/4/5/6
4. **Plots** — markdown image links to all generated PNGs
5. **Key Findings** — paragraph: what improved, what changed, what limitations remain

Also push the same report to `thesis_brain/simulation/physical-model-report.md`.

---

## 8. Git Discipline

- Branch in both repos: `claude/review-thesis-simulation-bU7tE`
- Commit after each step: `[physical-model] <what you did>`
- `git push -u origin claude/review-thesis-simulation-bU7tE` after each commit
- **Do NOT create a PR.**

---

## 9. Quality Checklist

- [ ] `python -c "from simulations.physical_env import AbstractWorld; AbstractWorld(seed=42)"` passes
- [ ] `python -c "from simulations.qa_mab_physical import QAMABPhysical; print('OK')"` passes
- [ ] All 6 validate scripts + baseline run to completion without error
- [ ] All result JSONs contain `"pass": true` (or documented explanation if false)
- [ ] All PNGs are non-empty and visually reasonable
- [ ] Old files untouched: `simulation_core.py`, `qa_mab.py`, `nb3r.py`, `validate_cat2/3/4_*.py`
- [ ] Both repos pushed on correct branch

---

## 10. Reference Files (read before coding)

| File (in `thesis_brain/simulation/src/uav_routing/`) | Why |
|------|-----|
| `agents/qamab_agent.py` | Residual credit assignment, epoch decay, UCB |
| `qubo.py` | QUBO construction to copy |
| `sa_solver.py` | Copy verbatim to `sa_solver_physical.py` |
| `ground_truth.py` | θ*/φ* sampling pattern |
| `runner.py` | Template for `runner_physical.py` |
| `config.py` | Default parameter values |

Files in `qa-mab-research/simulations/` to preserve (do NOT modify):
`simulation_core.py`, `qa_mab.py`, `nb3r.py`,
`validate_cat2_qubo_optimality.py`, `validate_cat3_sa_solver_accuracy.py`, `validate_cat4_learning_dynamics.py`,
`stochastic_noise_experiment.py`, `scaling_simulation.py`
