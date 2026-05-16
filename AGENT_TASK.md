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

### 2.3 QUBO (same structure as thesis_brain/simulation/src/uav_routing/qubo.py)

```
Variables: x[n*K+k] ∈ {0,1},  M = N*K total,  energy E = x^T Q x

Diagonal Q[i,i]  where i = n*K+k:
    cost_estimate = Σ_{u: uav_mask} θ̂[u]  +  Σ_{z: zone_mask} φ̂[z]
    Q[i,i] = cost_estimate  −  λ  −  UCB_c / sqrt(max(visit_counts[n,k], 1))

Same-flow penalty (i=n*K+k, j=n*K+k', k≠k'):
    Q[i,j] += λ      (store λ, not 2λ — energy contribution is Q[i,j]+Q[j,i] = 2λ)
    Q[j,i] += λ

Cross-flow structural interference (i=n*K+k, jj=l*K+j, n≠l, jj > i):
    contrib = 0
    if (path_uav_membership[n,k] AND path_uav_membership[l,j]).any():
        contrib += C_coll
    contrib += exp(−pair_min_dist[i,jj] / d0)
    Q[i,jj] += contrib
    Q[jj,i] += contrib
```

### 2.4 Residual Credit Assignment (update step)

```
# 1. Compute known contributions (from structural physics)
for each flow n:
    selected_uav_n = path_uav_membership[n, chosen_paths[n]]
    collision_count[n] = |{l≠n : (selected_uav_n & selected_uav_l).any()}|
    proximity[n] = Σ_{l≠n} exp(−pair_min_dist[n*K+k_n, l*K+k_l] / d0)

# 2. Strip known terms → fault-only residual
L_fault[n] = Loss[n] − C_coll·collision_count[n] − proximity[n]
L_fault[n] = max(0, L_fault[n])    # clip negatives

# 3. Update θ̂ (coordinate descent over UAVs on path)
for each flow n:
    uav_indices = where(path_uav_membership[n, chosen_paths[n]])
    theta_sum = sum(θ̂[u] for u in uav_indices)
    phi_sum   = sum(φ̂[z] for z in zone_indices_of_path)
    for each u in uav_indices:
        other_theta = theta_sum − θ̂[u]
        residual_u = L_fault[n] − other_theta − phi_sum
        θ̂[u] += α · (residual_u − θ̂[u])

# 4. Update φ̂ (after θ̂ is updated)
for each flow n:
    zone_indices = where(path_zone_membership[n, chosen_paths[n]])
    theta_sum = sum(θ̂[u] for u in uav_indices)   # re-read updated θ̂
    phi_sum   = sum(φ̂[z] for z in zone_indices)
    for each z in zone_indices:
        other_phi = phi_sum − φ̂[z]
        residual_z = L_fault[n] − theta_sum − other_phi
        φ̂[z] += α · (residual_z − φ̂[z])

# 5. Clip
θ̂ = clip(θ̂, 0, 1)
φ̂ = clip(φ̂, 0, 1)
```

### 2.5 Epoch Decay (Fix B)

```
At the START of each epoch p (before acting):
    if p > 0:
        θ̂ *= epoch_decay      # default 0.7
        φ̂ *= epoch_decay
    Reset visit_counts to zeros
    Regenerate world.pathset and world.pair_min_dist (epoch refresh)
    θ* and φ* PERSIST (same across all epochs for one seed)
```

### 2.6 Temperature Schedule

```
gamma(p, t) = gamma_0 / ((p+1)^a · (t+1)^b)
Q_scaled = Q / max(gamma, 1e-8)
Pass Q_scaled to SA solver.
```

---

## 3. Default Parameters

```python
# World
N = 3           # flows
K = 4           # paths per flow
m = 15          # UAVs
Z = 6           # zones
uavs_per_path_min = 2
uavs_per_path_max = 5
area_size = 1000.0

# Ground truth
n_faulty_uavs = 4
n_faulty_zones = 2
theta_low, theta_high = 0.20, 0.40
phi_low,   phi_high   = 0.20, 0.40
C_coll = 5.0
d0 = 150.0
sigma_noise = 0.05

# QA-MAB
alpha = 0.15
ucb_c = 3.0
epoch_decay = 0.7
lambda_onehot = 10.0
gamma_0 = 2.0
a = 0.5
b = 0.3
sa_sweeps = 200
sa_n_reads = 20
sa_T_init = 2.0
sa_T_final = 0.05
```

---

## 4. Files to Create

### 4.1 Repository: `qa-mab-research`  (branch `claude/review-thesis-simulation-bU7tE`)

#### `simulations/physical_env.py`
Class `AbstractWorld` with:
- `__init__(N, K, m, Z, n_faulty_uavs, n_faulty_zones, ..., seed)` — samples θ*, φ*, uav_zone; generates initial pathset + pair_min_dist
- `refresh_epoch(rng)` — re-samples path memberships + distances; θ*/φ* unchanged
- `compute_losses(chosen_paths, C_coll, d0, sigma_noise, rng)` — returns (N,) losses per §2.2

Dataclass `AbstractPathSet` with fields: `N, K, m, Z, path_uav_membership (N,K,m bool), path_zone_membership (N,K,Z bool)`

Note: `pair_min_dist` lives on `AbstractWorld`, not on `AbstractPathSet`.

#### `simulations/qa_mab_physical.py`
Class `QAMABPhysical` with:
- `__init__(world, C_coll, d0, sigma_noise, alpha, ucb_c, epoch_decay, lambda_onehot, gamma_0, a, b, sa_sweeps, sa_n_reads, sa_T_init, sa_T_final, seed)`
- `theta_hat: np.ndarray (m,)` — exposed attribute
- `phi_hat: np.ndarray (Z,)` — exposed attribute
- `reset_epoch(p)` — applies decay, resets visit_counts
- `build_qubo() -> np.ndarray` — per §2.3
- `act(t, p) -> np.ndarray (N,)` — builds QUBO, applies temperature, runs SA, returns chosen paths
- `update(chosen_paths, losses)` — residual credit assignment per §2.4
- `run(P, T, rng) -> dict` — full run returning losses_log (P,T,N), theta_err_log (P,), phi_err_log (P,), chosen_log (P,T,N)

For SA: copy the `sa_solve` + `decode_solution` from  
`thesis_brain/simulation/src/uav_routing/sa_solver.py` into `simulations/sa_solver_physical.py`  
and import from there.

#### `simulations/sa_solver_physical.py`
Copy of `thesis_brain/simulation/src/uav_routing/sa_solver.py` with no changes.

#### `simulations/agents_physical/__init__.py`  (empty)

#### `simulations/agents_physical/random_agent.py`
Class `RandomAgent(world, seed)` with `reset_epoch()`, `act(t,p)->np.ndarray`, `update(chosen, losses)` (no-op).  
Each step: for each flow n, choose Uniform(0, K-1).

#### `simulations/agents_physical/nb3r_agent.py`
Class `NB3RAgent(world, alpha=0.3, tau0=0.1, delta_tau=0.05, seed=42)`.  
Maintains weights `W[n,k]` (init 0). Softmax over −W (we minimise loss, not maximise reward).  
After observing losses:
```
collaborative_signal[n] = −losses[n] + Σ_{l≠n} −losses[l]   # share with all others
W[n, chosen[n]] = (1−alpha)·W[n, chosen[n]] + alpha·collaborative_signal[n]
τ += delta_tau
```
Softmax: `P[n,k] = exp(W[n,k]/τ) / Σ_j exp(W[n,j]/τ)`.
`reset_epoch()` — does NOT reset W (estimates persist like QA-MAB's θ̂/φ̂).

#### `simulations/agents_physical/oracle_agent.py`
Class `OracleAgent(world, C_coll, d0, sigma_noise, lambda_onehot, sa_sweeps, sa_n_reads, sa_T_init, sa_T_final, seed)`.  
Same as `QAMABPhysical` but θ̂ = θ* and φ̂ = φ* (known, never updated).  
No UCB bonus (visit_counts all 1). epoch_decay = 1.0 (no decay).

#### `simulations/agents_physical/optimal_agent.py`
Class `OptimalAgent(world, C_coll, d0)`.  
Enumerates all K^N path combinations, picks the one with minimum `expected_loss_no_noise(·)`.  
`expected_loss_no_noise(chosen)` = Loss without noise term, using true θ*/φ*.  
Only feasible for N·K ≤ ~1000; raise ValueError otherwise.

#### `simulations/runner_physical.py`
Function `run_experiment(N, K, m, Z, P, T, n_seeds, base_seed, **env_kwargs, **agent_kwargs) -> dict`:
- For each seed: create `AbstractWorld(seed=base_seed+s)`, run all 5 agents
- Each agent runs P epochs × T steps
- Returns: `{agent_name: {losses: (n_seeds,P,T,N), theta_err: (n_seeds,P), phi_err: (n_seeds,P), chosen: (n_seeds,P,T,N)}}`
- Save raw arrays + summary CSV to `simulations/results/physical_run/`

#### Validation scripts (see §5)

---

### 4.2 Repository: `thesis_brain`  (branch `claude/review-thesis-simulation-bU7tE`)

#### `simulation/src/uav_routing/abstract_env.py`
Same as `simulations/physical_env.py` in qa-mab-research (copy verbatim).  
This lives next to the existing `world.py` / `paths.py` — do NOT modify those.

#### `simulation/src/uav_routing/abstract_runner.py`
Short runner that imports from:
- `abstract_env.py` (AbstractWorld)
- existing `agents/qamab_agent.py` (adapted — see §4.3)
- existing `agents/nb3r_agent.py`, `random_agent.py`, `oracle_agent.py`, `optimal_agent.py`
- existing `qubo.py`, `sa_solver.py`, `ground_truth.py`

The runner exposes `run_abstract_experiment(cfg, save_dir)` mirroring `runner.py:run_experiment`.

#### `simulation/src/uav_routing/agents/qamab_agent.py` — minimal adaptation

The only line to change is inside `reset_epoch`:
```python
# BEFORE:
self._pair_min_dist = compute_all_pair_min_distances(pathset, topology.positions)

# AFTER: accept either (topology, pathset) or (abstract_world, None)
def reset_epoch(self, topology_or_world, pathset=None):
    if pathset is None:
        # Abstract world mode
        world = topology_or_world
        self._pathset = world.pathset
        self._pair_min_dist = world.pair_min_dist
    else:
        # Original mode (topology + pathset)
        self._topology = topology_or_world
        self._pathset = pathset
        self._pair_min_dist = compute_all_pair_min_distances(pathset, topology_or_world.positions)
    self._visit_counts = np.zeros((self._pathset.N, self._pathset.K), dtype=int)
    decay = self._qamab_cfg.epoch_decay
    self.theta_hat *= decay
    self.phi_hat   *= decay
```

All other logic in `qamab_agent.py` stays identical — it already uses `self._pathset` and `self._pair_min_dist`.

Similar 1-line adaptation for `nb3r_agent.py`, `oracle_agent.py`, `optimal_agent.py`, `random_agent.py`:
add `pathset=None` param to `reset_epoch` and branch as above.

---

## 5. Validation Scripts

All in `qa-mab-research/simulations/`. Save results to `simulations/results/`.

### 5.1 `validate_cat2_physical.py` — QUBO Optimality

**Goal:** Verify that the QUBO formula encodes the correct objective.  
**Setup:** N=2, K=3, m=10, Z=4. n_faulty_uavs=2, n_faulty_zones=1. sigma_noise=0, UCB=0.  
**Procedure per seed (50 seeds):**
1. Create AbstractWorld(seed=s). Use θ̂=θ*, φ̂=φ* (oracle values).
2. Build QUBO Q with UCB_c=0.
3. Enumerate all K^N = 9 path assignments.
4. For each assignment `chosen`:
   - Compute `E(x) = x^T Q x`
   - Compute `L_expected(chosen)` = loss without noise (§2.2 minus noise term)
5. Assert: `argmin_x E(x) == argmin_x L_expected(x)`  
   (break ties by choosing lowest-index assignment)
6. Record: match (bool), E_min, L_min, E_argmin, L_argmin

**PASS criterion:** ≥ 95% of seeds agree (allow 5% tolerance for floating-point near-ties).

**Output:** `results/validation_cat2_physical/result.json` with fields:
```json
{"pass": true/false, "success_rate": 0.98, "n_seeds": 50,
 "n_agree": 49, "details": [{"seed":0, "match":true, ...}, ...]}
```
Also `result.csv` with columns: seed, match, E_min, L_min.

### 5.2 `validate_cat3_physical.py` — SA Solver Accuracy

**Goal:** Verify SA finds the QUBO minimum reliably.  
**Setup:** N=3, K=3, m=15, Z=5. Random θ̂/φ̂ (not oracle). UCB=0.  
**Procedure per seed (30 seeds):**
1. Create AbstractWorld(seed=s), random θ̂=Uniform(0,0.5), φ̂=Uniform(0,0.5).
2. Build Q. Brute-force argmin over K^N=27 assignments.
3. Run `sa_solve(Q, ...)` with default params (n_reads=20, n_sweeps=200).
4. Compute energy gap: `gap = E_SA - E_brute_force`.
5. Record: SA found exact minimum? gap value.

**PASS criterion:** success_rate ≥ 0.85 (SA matches brute-force ≥ 85% of seeds).

**Output:** `results/validation_cat3_physical/result.json`:
```json
{"pass": true/false, "success_rate": 0.90, "mean_gap": 0.01, ...}
```

### 5.3 `validate_cat4_physical.py` — Learning Dynamics

**Goal:** Verify θ̂ and φ̂ converge toward θ* and φ*.  
**Setup:** N=3, K=4, m=15, Z=6. P=25 epochs, T=40 steps/epoch. 15 seeds.

**Procedure (same as existing `validate_cat4_learning_dynamics.py` but using AbstractWorld):**
1. Per seed: create AbstractWorld(seed). Create QAMABPhysical(world).
2. For each epoch p:
   a. Record theta_err = ||θ̂ − θ*||_1 / m, phi_err = ||φ̂ − φ*||_1 / Z
   b. Compute Oracle's optimal loss (brute force over K^N), record gap = QA-MAB loss − oracle loss
   c. Run T steps (act → compute_losses → update)
   d. world.refresh_epoch(rng), agent.reset_epoch(p)
3. Collect: theta_err[seed, epoch], phi_err[seed, epoch], gap[seed, epoch]

**Test both:** epoch_decay=0.7, epoch_decay=1.0  
**PASS criterion per decay variant:** `gap[initial] > gap[final]` OR `theta_err[initial] > theta_err[final]`  
(at least one metric improved over training)

**Output:** `results/validation_cat4_physical/result.json`:
```json
{
  "overall_pass": true,
  "decay_0.7": {"pass":true, "theta_err_init":0.12, "theta_err_final":0.06, "gap_init":1.2, "gap_final":0.4},
  "decay_1.0": {"pass":true, ...},
  "metadata": {"n_seeds":15, "P":25, "T":40}
}
```
Also save per-epoch CSVs and PNG plots (theta_err vs epoch, gap vs epoch) to `results/validation_cat4_physical/`.

### 5.4 `validate_baseline_comparison.py` — New vs. Old Model

**Goal:** Compare new physical model against old NetworkEnvironment model.  
**Setup:** N=10, K=4, m=30, Z=9. P=10 epochs, T=100 steps/epoch. 20 seeds.

**Agents on NEW model (AbstractWorld):**
- QA-MAB Physical (θ̂/φ̂ learning)
- NB3R Physical
- Random Physical
- Oracle Physical (knows θ*/φ*)
- Optimal Physical (brute force; skip if N*K > 50)

**Agents on OLD model (NetworkEnvironment from `simulation_core.py`):**
- QA-MAB legacy (from `qa_mab.py`)
- NB3R legacy (from `nb3r.py`)
- Random legacy

**Metrics per agent:** mean final loss (last 10 steps), cumulative loss, theta_err trajectory (new model only)

**Scaling sub-experiment:** repeat with N ∈ {5, 10, 20, 30} (K=4, m=20, Z=6, P=5, T=50, 10 seeds)

**PASS criterion:** QA-MAB Physical final loss ≤ NB3R Physical ≤ Random Physical (for most N values).

**Output:**
- `results/validation_baseline/result.json` with full summary
- `results/validation_baseline/summary.csv` with columns: model, agent, N, mean_loss, std_loss
- 3 PNG plots:
  - `scaling.png`: mean final loss vs N for all agents (both models)
  - `convergence.png`: loss vs step at N=10
  - `theta_convergence.png`: theta_err vs epoch for QA-MAB Physical at N=10

---

## 6. Run Script

Create `simulations/run_validation_physical.sh`:

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Cat 2: QUBO Optimality ==="
python validate_cat2_physical.py

echo "=== Cat 3: SA Solver Accuracy ==="
python validate_cat3_physical.py

echo "=== Cat 4: Learning Dynamics ==="
python validate_cat4_physical.py

echo "=== Baseline Comparison ==="
python validate_baseline_comparison.py

echo ""
echo "All validations complete. Results in simulations/results/"
```

---

## 7. Final Report

Create `simulation/physical-model-report.md` in `thesis_brain` (branch `claude/review-thesis-simulation-bU7tE`) with:

1. **Model Description** — equations from §2 in LaTeX-style markdown
2. **Validation Results** — one table per validation with pass/fail + key numbers
3. **Plots** — embed (as markdown image links) the 4 PNGs from baseline comparison
4. **Comparison to Old Model** — paragraph discussing what improved and what was lost
5. **Known Limitations** — what the abstract model does not capture vs. the real UAV graph

Also save the same report to `simulations/results/physical-model-report.md` in `qa-mab-research`.

---

## 8. Git Discipline

- Branch in both repos: `claude/review-thesis-simulation-bU7tE`
- Commit after each major step (env, agents, runner, each validation, report)
- Commit message format: `[physical-model] <what you did>`
- Push after every commit (`git push -u origin claude/review-thesis-simulation-bU7tE`)
- **Do NOT create a PR.**

---

## 9. Quality Checklist (verify before each commit)

- [ ] `python -c "from simulations.physical_env import AbstractWorld; AbstractWorld(seed=42)"` — no error
- [ ] `python -c "from simulations.qa_mab_physical import QAMABPhysical; print('OK')"` — no error
- [ ] validate_cat2 runs in < 60 seconds
- [ ] validate_cat3 runs in < 3 minutes
- [ ] validate_cat4 runs in < 25 minutes
- [ ] All result JSONs exist and contain `"pass": true`
- [ ] No existing files deleted (simulation_core.py, qa_mab.py, nb3r.py still present)
- [ ] All plots are non-empty PNGs (open and verify visually)

---

## 10. Reference Files (read before coding)

All in `thesis_brain` repo, `main` branch, path prefix `simulation/src/uav_routing/`:

| File | Why to read |
|------|-------------|
| `agents/qamab_agent.py` | Residual credit assignment (Fix A), epoch decay (Fix B), UCB |
| `qubo.py` | Exact QUBO construction — copy logic, adapt to AbstractPathSet |
| `sa_solver.py` | Copy verbatim to `sa_solver_physical.py` |
| `ground_truth.py` | `sample_ground_truth` pattern for θ*/φ* |
| `runner.py` | Template for `runner_physical.py` |
| `config.py` | Default parameter values |

Existing qa-mab-research files to preserve (do NOT modify):
- `simulations/simulation_core.py`
- `simulations/qa_mab.py`
- `simulations/nb3r.py`
- `simulations/validate_cat2_qubo_optimality.py`
- `simulations/validate_cat3_sa_solver_accuracy.py`
- `simulations/validate_cat4_learning_dynamics.py`
