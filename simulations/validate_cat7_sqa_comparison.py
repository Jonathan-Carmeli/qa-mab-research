#!/usr/bin/env python3
"""Cat 7: SA vs SQA — Simulated Quantum Annealing comparison.

Implements Path Integral Monte Carlo (PIQMC) / Suzuki-Trotsky SQA.
Compares SA (thermal hopping) vs SQA (quantum tunneling) on the same QUBO.

SQA: N_replicas copies of the system coupled by transverse-field "chains".
The coupling strength ~ T²/d (d=replicas) mimics quantum tunneling.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os, itertools
from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.sa_solver_physical import sa_solve, decode_solution

OUT = "simulations/results/validation_cat7_physical/"
os.makedirs(OUT, exist_ok=True)


def qubo_energy(Q, paths, N, K):
    x = np.zeros(N * K, dtype=int)
    for n in range(N):
        x[n * K + paths[n]] = 1
    return float(x @ Q @ x)


def brute_force_best(Q, N, K):
    best_E = float('inf')
    for combo in itertools.product(range(K), repeat=N):
        x = np.zeros(N * K, dtype=int)
        for n in range(N):
            x[n * K + combo[n]] = 1
        E = float(x @ Q @ x)
        if E < best_E:
            best_E = E
    return best_E


def sqa_solve(Q, rng, n_replicas=8, n_sweeps=200, T_init=2.0, T_final=0.05, J=1.0):
    """Simulated Quantum Annealing via Path Integral Monte Carlo (Suzuki-Trotsky).

    Parameters
    ----------
    Q       : (M, M) QUBO matrix
    rng     : numpy Generator
    n_replicas : number of Trotter slices (higher = more quantum, slower)
    n_sweeps : MC sweeps per replica
    T_init, T_final : temperature schedule
    J       : transverse-field coupling strength

    Returns
    -------
    best_x : (M,) best binary solution across all replicas
    best_E : float best QUBO energy
    """
    M = Q.shape[0]
    temps = np.linspace(T_init, T_final, n_sweeps)

    # Initialize all replicas randomly
    replicas = rng.integers(0, 2, size=(n_replicas, M))
    h_replicas = np.einsum('rm,mn->rn', replicas.astype(float), Q)

    best_E = float('inf')
    best_x = replicas[0].copy()

    for s, T in enumerate(temps):
        beta = 1.0 / T
        Gamma = J * beta / n_replicas  # transverse field coupling

        for r in range(n_replicas):
            x = replicas[r]
            h = h_replicas[r]

            # Random sweep order
            order = rng.permutation(M)
            for i in order:
                old_xi = x[i]

                # Classical flip delta_E
                delta_E_cl = (1 - 2 * old_xi) * (Q[i, i] + 2 * (h[i] - Q[i, i] * old_xi))

                # Quantum coupling: try to align with neighboring replicas
                r_up = (r - 1) % n_replicas
                r_dn = (r + 1) % n_replicas
                x_up = replicas[r_up]
                x_dn = replicas[r_dn]
                delta_E_q = -Gamma * ((2 * old_xi - 1) * (2 * x_up[i] - 1) + (2 * old_xi - 1) * (2 * x_dn[i] - 1))

                delta_E = delta_E_cl + delta_E_q

                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    x[i] = 1 - old_xi
                    delta_xi = x[i] - old_xi
                    h += Q[:, i] * delta_xi
                    h_replicas[r] = h

        # Track best across replicas
        for r in range(n_replicas):
            E_r = float(replicas[r] @ h_replicas[r])
            if E_r < best_E:
                best_E = E_r
                best_x = replicas[r].copy()

    return best_x, best_E


def run_seed_comparison(seed, N=3, K=3, m=15, Z=5, sa_n_reads=20, sqa_n_replicas=8):
    world = AbstractWorld(N=N, K=K, m=m, Z=Z, seed=seed)
    rng = np.random.default_rng(seed)

    agent = QAMABPhysical(world, ucb_c=0.0, alpha=0.0, seed=seed)
    agent.theta_hat = rng.uniform(0.1, 0.5, m)
    agent.phi_hat = rng.uniform(0.1, 0.5, Z)
    agent._visit_counts = np.zeros((N, K), dtype=int)
    agent.world.refresh_epoch(agent.rng)
    Q = agent.build_qubo()

    bf_E = brute_force_best(Q, N, K)

    # SA — best of sa_n_reads independent runs
    sa_best_E = float('inf')
    for _ in range(sa_n_reads):
        sa_x, sa_E = sa_solve(Q, rng, n_reads=1, n_sweeps=200, T_init=2.0, T_final=0.05)
        sa_paths = decode_solution(sa_x, N, K)
        E = qubo_energy(Q, sa_paths, N, K)
        if E < sa_best_E: sa_best_E = E

    # SQA
    sqa_x, sqa_E = sqa_solve(Q, rng, n_replicas=sqa_n_replicas, n_sweeps=200, T_init=2.0, T_final=0.05)
    sqa_paths = decode_solution(sqa_x, N, K)
    sqa_E_decoded = qubo_energy(Q, sqa_paths, N, K)

    return {
        "bf_E": bf_E,
        "sa_E": sa_best_E,
        "sqa_E": sqa_E_decoded,
        "sa_gap": sa_best_E - bf_E,
        "sqa_gap": sqa_E_decoded - bf_E,
        "sa_exact": abs(sa_best_E - bf_E) < 1e-6,
        "sqa_exact": abs(sqa_E_decoded - bf_E) < 1e-6,
    }


n_seeds = 30
print(f"Running Cat 7: SA vs SQA ({n_seeds} seeds)", flush=True)
rows = []
sa_exact = 0; sqa_exact = 0; sqa_better = 0

for i in range(n_seeds):
    seed = 42 + i
    print(f"  seed={seed}", end="\r", flush=True)
    r = run_seed_comparison(seed)
    rows.append(r)
    if r["sa_exact"]: sa_exact += 1
    if r["sqa_exact"]: sqa_exact += 1
    if r["sqa_gap"] < r["sa_gap"]: sqa_better += 1

print(f"\nSA exact: {sa_exact}/{n_seeds} = {sa_exact/n_seeds:.1%}")
print(f"SQA exact: {sqa_exact}/{n_seeds} = {sqa_exact/n_seeds:.1%}")
print(f"SQA better than SA: {sqa_better}/{n_seeds} = {sqa_better/n_seeds:.1%}")

sa_rate = sa_exact / n_seeds
sqa_rate = sqa_exact / n_seeds
pass_cat7 = sqa_rate >= sa_rate  # SQA should not be worse than SA

result = {
    "pass": pass_cat7,
    "reason": f"SA={sa_rate:.1%}, SQA={sqa_rate:.1%} ({'SQA >= SA: PASS' if pass_cat7 else 'SQA < SA: FAIL'})",
    "sa_exact_rate": sa_rate,
    "sqa_exact_rate": sqa_rate,
    "sqa_better_than_sa": sqa_better / n_seeds,
    "mean_sa_gap": float(np.mean([r["sa_gap"] for r in rows])),
    "mean_sqa_gap": float(np.mean([r["sqa_gap"] for r in rows])),
}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

import csv
with open(OUT + "result.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed","bf_E","sa_E","sqa_E","sa_gap","sqa_gap","sa_exact","sqa_exact"])
    w.writeheader()
    for i, r in enumerate(rows):
        w.writerow({"seed": 42+i, **r})

print(f"Cat 7: {'PASS' if pass_cat7 else 'FAIL'}")