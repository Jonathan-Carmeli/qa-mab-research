# Physical Validation Suite

All files for the **physical-abstract model** (post-upgrade from CMAB).

## Model
```
Loss[n] = Σθ*[uav] + Σφ*[zone] + C_coll·collisions + Σexp(−dist/d0) + Normal(0, σ²)
```
θ*, φ* are unknown. Collision + proximity are known (encoded in QUBO).

## Structure

### Core
- `physical_env.py` — AbstractWorld + AbstractPathSet
- `qa_mab_physical.py` — QAMABPhysical agent
- `sa_solver_physical.py` — SA solver
- `runner_physical.py` — shared-world experiment runner

### Agents (`agents_physical/`)
- `random_agent.py` — Random baseline
- `nb3r_agent.py` — NB3R baseline
- `oracle_agent.py` — Oracle (knows θ*, φ*)
- `optimal_agent.py` — Optimal (exhaustive search)

### Validation Scripts
- `validate_cat1_param_sweep_physical.py` — Cat 1: parameter sweeps
- `validate_cat2_physical.py` — Cat 2: QUBO optimality
- `validate_cat3_physical.py` — Cat 3: SA solver accuracy
- `validate_cat4_physical.py` — Cat 4: learning dynamics
- `validate_cat5_regret_convergence_physical.py` — Cat 5: regret crossover
- `validate_cat6_noise_robustness_physical.py` — Cat 6: noise robustness

### Results
Results are in `simulations/results/physical/` (created by scripts).

## Exploration Schedule
```python
gamma = γ₀ / ((p+1)^a · (t+1)^b)
Q_scaled = Q / gamma  → grows as γ shrinks
```

## Recommended Config
- UCB c = 3.0
- SA sweeps = 200
- SA n_reads = 20
- epoch_decay = 0.7