"""
Stage 1: Ground Truth via Brute Force
=====================================

For small N (4..8), enumerate all m^N route assignments and compute the
true social-welfare optimum for each (N, seed) pair. This gives us the
exact answer that any solver must approximate.

Output: results/convergence_test/ground_truth.json
"""

import itertools
import json
import os
import sys
import time

import numpy as np

# Make `simulations` importable when running as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simulations"))

from simulation_core import NetworkEnvironment


def social_welfare_from_arrays(B, I, assignment_tuple):
    """
    assignment_tuple: tuple of length N giving the route k for each agent.
    Vectorized social-welfare computation: SW = sum_i (B[i,k_i] - sum_j I[i,k_i,j,k_j]).
    """
    N = len(assignment_tuple)
    sw = 0.0
    base = 0.0
    inter = 0.0
    for i, ki in enumerate(assignment_tuple):
        base += B[i, ki]
        for j, kj in enumerate(assignment_tuple):
            if j == i:
                continue
            inter += I[i, ki, j, kj]
    sw = base - inter
    return sw, base, inter


def brute_force_optimum(env):
    """Enumerate all m^N assignments and return the best."""
    N, m = env.N, env.m
    B, I = env.B, env.I
    best_sw = -np.inf
    best_assignment = None
    for assignment in itertools.product(range(m), repeat=N):
        sw, _, _ = social_welfare_from_arrays(B, I, assignment)
        if sw > best_sw:
            best_sw = sw
            best_assignment = assignment
    return best_sw, list(best_assignment)


def main():
    out_dir = os.path.join(ROOT, "simulations", "results", "convergence_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ground_truth.json")

    Ns = [4, 5, 6, 7, 8]
    n_seeds = 100
    m = 4

    results = {}
    summary_rows = []

    total_start = time.time()
    for N in Ns:
        results[str(N)] = []
        n_assignments = m ** N
        print(f"[N={N}] enumerating {n_assignments:,} assignments per seed, {n_seeds} seeds")
        N_start = time.time()
        for seed in range(n_seeds):
            env = NetworkEnvironment(N=N, m=m, seed=seed)
            t0 = time.time()
            best_sw, best_assignment = brute_force_optimum(env)
            elapsed = time.time() - t0

            # Also record sum of greedy-B (no interference reasoning)
            greedy_assignment = tuple(int(np.argmax(env.B[i])) for i in range(N))
            greedy_sw, _, _ = social_welfare_from_arrays(env.B, env.I, greedy_assignment)

            results[str(N)].append({
                "seed": seed,
                "true_optimum_sw": float(best_sw),
                "best_assignment": list(map(int, best_assignment)),
                "greedy_sw": float(greedy_sw),
                "greedy_assignment": list(map(int, greedy_assignment)),
                "n_assignments_evaluated": int(n_assignments),
                "brute_force_seconds": float(elapsed),
            })
        elapsed_N = time.time() - N_start
        opt_vals = [r["true_optimum_sw"] for r in results[str(N)]]
        gre_vals = [r["greedy_sw"] for r in results[str(N)]]
        summary_rows.append({
            "N": N,
            "mean_optimum": float(np.mean(opt_vals)),
            "std_optimum": float(np.std(opt_vals)),
            "mean_greedy": float(np.mean(gre_vals)),
            "mean_greedy_ratio": float(np.mean(np.array(gre_vals) / np.array(opt_vals))),
            "elapsed_seconds": float(elapsed_N),
        })
        print(f"  -> mean optimum SW = {np.mean(opt_vals):.4f} ± {np.std(opt_vals):.4f} "
              f"(greedy-B ratio = {np.mean(np.array(gre_vals) / np.array(opt_vals)):.3f}, "
              f"{elapsed_N:.1f}s)")

    payload = {
        "metadata": {
            "Ns": Ns,
            "m": m,
            "n_seeds": n_seeds,
            "B_scale": "uniform",
            "I_scale": "moderate",
            "total_seconds": float(time.time() - total_start),
        },
        "summary": summary_rows,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Total elapsed: {time.time() - total_start:.1f}s")
    print("\nSummary:")
    for row in summary_rows:
        print(f"  N={row['N']:2d}  optimum={row['mean_optimum']:.4f}  "
              f"greedy-ratio={row['mean_greedy_ratio']:.3f}")


if __name__ == "__main__":
    main()
