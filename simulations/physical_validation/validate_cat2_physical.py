#!/usr/bin/env python3
"""Cat 2: QUBO Optimality — verify QUBO encodes correct objective."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os, itertools
from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical

OUT = "simulations/results/validation_cat2_physical/"
os.makedirs(OUT, exist_ok=True)

def compute_loss_no_noise(world, chosen_paths):
    N, K = world.N, world.K
    ps = world.pathset
    losses = np.zeros(N)
    sel_uav = np.array([ps.path_uav_membership[n, chosen_paths[n]] for n in range(N)])
    shared = (sel_uav.astype(int) @ sel_uav.astype(int).T) > 0
    np.fill_diagonal(shared, False)
    coll = shared.sum(axis=1).astype(float)
    for n in range(N):
        k = chosen_paths[n]
        losses[n] = (
            float(world.theta_star[ps.path_uav_membership[n, k]].sum())
            + float(world.phi_star[ps.path_zone_membership[n, k]].sum())
            + world.C_coll * coll[n]
        )
        a = n*K + k
        for l in range(N):
            if l == n:
                continue
            b = l*K + chosen_paths[l]
            losses[n] += np.exp(-ps.pair_min_dist[a, b] / world.d0)
    return losses.sum()

def evaluate_seed(seed):
    world = AbstractWorld(N=2, K=3, m=10, Z=4, sigma_noise=0.0, seed=seed)
    rng = np.random.default_rng(seed)
    agent = QAMABPhysical(world, ucb_c=0.0, alpha=0.0, seed=seed)
    agent.theta_hat = world.theta_star.copy()
    agent.phi_hat = world.phi_star.copy()
    agent._visit_counts = np.zeros((2, 3), dtype=int)
    agent.world.refresh_epoch(agent.rng)

    Q = agent.build_qubo()
    N, K = 2, 3

    best_E_combo, best_E = None, float('inf')
    for combo in itertools.product(range(K), repeat=N):
        x = np.zeros(N*K, dtype=int)
        for n in range(N):
            x[n*K + combo[n]] = 1
        E = float(x @ Q @ x)
        if E < best_E:
            best_E = E
            best_E_combo = list(combo)

    best_L_combo, best_L = None, float('inf')
    for combo in itertools.product(range(K), repeat=N):
        L = compute_loss_no_noise(world, np.array(combo))
        if L < best_L:
            best_L = L
            best_L_combo = list(combo)

    # Pass if they agree or if energy gap is 0 at optimal
    return best_E_combo == best_L_combo or abs(best_E - best_L) < 1e-6

n_seeds = 50
passed = sum(evaluate_seed(42+i) for i in range(n_seeds))
rate = passed / n_seeds
pass_cat2 = rate >= 0.95

result = {
    "pass": pass_cat2,
    "reason": f"{passed}/{n_seeds} agree (rate={rate:.1%}, {'PASS' if pass_cat2 else 'FAIL'}, threshold 95%)",
    "n_seeds": n_seeds, "n_passed": passed, "win_rate": rate,
}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

import csv
with open(OUT + "result.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["n_seeds","n_passed","win_rate"])
    w.writeheader()
    w.writerow({"n_seeds": n_seeds, "n_passed": passed, "win_rate": rate})

print(f"Cat 2: {passed}/{n_seeds} agree — {'PASS' if pass_cat2 else 'FAIL'}")