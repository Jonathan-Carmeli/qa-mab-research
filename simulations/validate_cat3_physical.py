#!/usr/bin/env python3
"""Cat 3: SA Solver Accuracy on physical model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os, itertools
from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.sa_solver_physical import sa_solve

OUT = "simulations/results/validation_cat3_physical/"
os.makedirs(OUT, exist_ok=True)

def brute_force_best(Q, N, K):
    best_E, best_x = float('inf'), None
    for combo in itertools.product(range(K), repeat=N):
        x = np.zeros(N*K, dtype=int)
        for n in range(N): x[n*K + combo[n]] = 1
        E = float(x @ Q @ x)
        if E < best_E: best_E = E; best_x = x.copy()
    return best_E, best_x

n_seeds = 30
exact_hits = 0
gaps = []
for i in range(n_seeds):
    seed = 42 + i
    world = AbstractWorld(N=3, K=3, m=15, Z=5, seed=seed)
    rng = np.random.default_rng(seed)
    agent = QAMABPhysical(world, ucb_c=0.0, alpha=0.0, seed=seed)
    agent.theta_hat = rng.uniform(0.1, 0.5, world.m)
    agent.phi_hat = rng.uniform(0.1, 0.5, world.Z)
    agent._visit_counts = np.zeros((3, 3), dtype=int)
    agent.world.refresh_epoch(agent.rng)

    Q = agent.build_qubo()
    bf_E, _ = brute_force_best(Q, 3, 3)
    sa_x, sa_E = sa_solve(Q, rng, n_reads=20, n_sweeps=200, T_init=2.0, T_final=0.05)
    gap = sa_E - bf_E
    gaps.append(gap)
    if abs(gap) < 1e-6:
        exact_hits += 1

rate = exact_hits / n_seeds
pass_cat3 = rate >= 0.78

result = {
    "pass": pass_cat3,
    "reason": f"{exact_hits}/{n_seeds} exact (rate={rate:.1%}, {'PASS' if pass_cat3 else 'FAIL'}, threshold 78%)",
    "n_seeds": n_seeds, "n_exact": exact_hits, "win_rate": rate,
    "mean_gap": float(np.mean(gaps)), "std_gap": float(np.std(gaps)),
}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

import csv
with open(OUT + "result.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["n_seeds","n_exact","win_rate","mean_gap","std_gap"])
    w.writeheader(); w.writerow(result)

print(f"Cat 3: {exact_hits}/{n_seeds} exact — {'PASS' if pass_cat3 else 'FAIL'}")