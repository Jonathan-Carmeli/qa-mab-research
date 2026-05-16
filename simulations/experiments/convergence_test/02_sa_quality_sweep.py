"""
Stage 2: SA Quality Sweep
=========================

Build the QUBO from the TRUE B and I (oracle) and ask: how close does
simulated annealing get to the true optimum, as a function of compute
budget?

If even SA-very-strong on the oracle QUBO falls short of the brute-force
optimum for small N, the bottleneck is the solver. If SA-strong matches
brute force but the QA-MAB run still underperforms, the bottleneck is
the learning dynamics (Stage 5).

Levels:
  SA-weak        : 8 restarts x 15 iterations (current spec)
  SA-medium      : 50 restarts x 200 iterations
  SA-strong      : 200 restarts x 1000 iterations
  SA-very-strong : 1000 restarts x 5000 iterations

We use the SAME proposal/accept logic as qa_mab.QAMAB.solve_qubo, but
re-implemented locally so qa_mab.py stays untouched. Energy is computed
incrementally (delta updates) which is 2-3 orders of magnitude faster
than recomputing x^T Q x each iteration; this matters for SA-very-strong.

SA spec from task:
  T starts at 2.0, multiply by 0.95 each iteration.
  Accept worse with prob exp(-delta/T).
  Proposal: pick agent i uniformly at random, pick k_new != current.

Output: results/convergence_test/sa_quality_sweep.json
"""

import itertools
import json
import math
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simulations"))

from simulation_core import NetworkEnvironment


# --------------------------------------------------------------------------- #
# Oracle QUBO                                                                 #
# --------------------------------------------------------------------------- #
def build_oracle_qubo(B, I, lambda_=0.5, tau=1.0):
    """
    QUBO from TRUE B and I. Mirrors QAMAB.build_qubo with u_hat=B, I_hat=I.

      diag           : -B[i,k] - lambda/2
      same-agent off : lambda/2  (symmetric upper+lower)
      cross-agent    : I[i,k,j,l]   (asymmetric)

    Energy = x^T Q x. tau is just an overall energy scale.
    """
    N, m = B.shape
    size = N * m
    Q = np.zeros((size, size))
    for i in range(N):
        for k in range(m):
            ik = i * m + k
            Q[ik, ik] = -B[i, k] - lambda_ / 2.0
            for l in range(k + 1, m):
                il = i * m + l
                Q[ik, il] = lambda_ / 2.0
                Q[il, ik] = lambda_ / 2.0
    for i in range(N):
        for k in range(m):
            ik = i * m + k
            for j in range(N):
                if j == i:
                    continue
                for l in range(m):
                    Q[ik, j * m + l] = I[i, k, j, l]
    return tau * Q


# --------------------------------------------------------------------------- #
# Standalone SA (mirrors QAMAB.solve_qubo, with delta-energy updates)         #
# --------------------------------------------------------------------------- #
def sa_solve(Q, N, m, n_restarts, n_iters, T0=2.0, decay=0.95,
             seed=0, greedy_init_B=None):
    """
    Simulated annealing on QUBO with route-flip proposals.

    Same proposal/accept logic as QAMAB.solve_qubo:
      - Restart 0 starts at greedy-B (if provided), later restarts add a
        small random perturbation.
      - Each step: pick random agent, flip current route to a random
        different route. Accept if delta < 0, else with prob exp(-delta/T).
      - T starts at T0 (scaled by restart) and decays by `decay` each step.

    Optimised inner loop: maintain
        Q_row_sum[k] = sum_{j} Q[k, active[j]]
        Q_col_sum[k] = sum_{j} Q[active[j], k]
    so that delta of a candidate flip becomes O(1) scalar arithmetic and
    only an accepted flip costs O(N*m) to update the sums. This gives
    ~10-50x speedup over recomputing x^T Q x each step, which matters
    for SA-very-strong (5M iterations).

    Returns: (active_idx, best_energy, assignment_tuple)
    """
    pyrng = random.Random(seed)
    best_active = None
    best_energy = float("inf")
    Q_diag = np.diag(Q).copy()
    size = N * m

    for restart in range(n_restarts):
        active = np.zeros(N, dtype=np.int64)
        if greedy_init_B is not None:
            for i in range(N):
                active[i] = i * m + int(np.argmax(greedy_init_B[i]))
        else:
            for i in range(N):
                active[i] = i * m + pyrng.randrange(m)

        if restart > 0:
            n_flips = pyrng.randrange(1, max(2, N // 3 + 1))
            for _ in range(n_flips):
                i = pyrng.randrange(N)
                k_old = int(active[i] - i * m)
                cands = [k for k in range(m) if k != k_old]
                k_new = pyrng.choice(cands)
                active[i] = i * m + k_new

        # Initial Q_row_sum / Q_col_sum and energy
        Q_row_sum = Q[:, active].sum(axis=1)   # for each k, sum over active cols
        Q_col_sum = Q[active, :].sum(axis=0)   # for each k, sum over active rows
        energy = float(Q_col_sum[active].sum())
        if energy < best_energy:
            best_energy = energy
            best_active = active.copy()

        T = T0 * (1.0 + restart * 0.3)
        m_minus1 = m - 1
        for step in range(n_iters):
            T *= decay
            i = pyrng.randrange(N)
            old_idx = int(active[i])
            k_old = old_idx - i * m
            k_new = (k_old + 1 + pyrng.randrange(m_minus1)) % m
            new_idx = i * m + k_new

            sum_row = (Q_row_sum[new_idx] - Q[new_idx, old_idx]) - (Q_row_sum[old_idx] - Q[old_idx, old_idx])
            sum_col = (Q_col_sum[new_idx] - Q[old_idx, new_idx]) - (Q_col_sum[old_idx] - Q[old_idx, old_idx])
            delta = (Q_diag[new_idx] - Q_diag[old_idx]) + sum_row + sum_col

            if delta < 0 or (T > 1e-10 and pyrng.random() < math.exp(-delta / T)):
                active[i] = new_idx
                energy += delta
                Q_row_sum += Q[:, new_idx] - Q[:, old_idx]
                Q_col_sum += Q[new_idx, :] - Q[old_idx, :]
                if energy < best_energy:
                    best_energy = energy
                    best_active = active.copy()

    assignment = tuple(int(best_active[i] - i * m) for i in range(N))
    return best_active, float(best_energy), assignment


# --------------------------------------------------------------------------- #
# Brute force                                                                 #
# --------------------------------------------------------------------------- #
def brute_force(B, I):
    N, m = B.shape
    best_sw = -np.inf
    best_a = None
    for a in itertools.product(range(m), repeat=N):
        sw = 0.0
        for i, ki in enumerate(a):
            sw += B[i, ki]
            for j, kj in enumerate(a):
                if j == i:
                    continue
                sw -= I[i, ki, j, kj]
        if sw > best_sw:
            best_sw = sw
            best_a = a
    return float(best_sw), tuple(map(int, best_a))


def social_welfare(B, I, assignment):
    N = len(assignment)
    sw = 0.0
    for i, ki in enumerate(assignment):
        sw += B[i, ki]
        for j, kj in enumerate(assignment):
            if j == i:
                continue
            sw -= I[i, ki, j, kj]
    return float(sw)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
SA_LEVELS = {
    "SA-weak":        dict(n_restarts=8,    n_iters=15),
    "SA-medium":      dict(n_restarts=50,   n_iters=200),
    "SA-strong":      dict(n_restarts=200,  n_iters=1000),
    "SA-very-strong": dict(n_restarts=1000, n_iters=5000),
}

LAMBDA = 0.5
M = 4


def main():
    out_dir = os.path.join(ROOT, "simulations", "results", "convergence_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sa_quality_sweep.json")

    Ns = [4, 5, 6, 7, 8, 10, 12, 15]
    n_seeds = 50

    results = {}
    summary_rows = []

    grand_start = time.time()
    for N in Ns:
        results[str(N)] = []
        per_level_ratios = {lvl: [] for lvl in SA_LEVELS}
        per_level_times = {lvl: [] for lvl in SA_LEVELS}
        bf_used = N <= 8

        print(f"\n[N={N}] {n_seeds} seeds, brute_force={'yes' if bf_used else 'no'}")
        for seed in range(n_seeds):
            env = NetworkEnvironment(N=N, m=M, seed=seed)
            B, I = env.B, env.I
            Q = build_oracle_qubo(B, I, lambda_=LAMBDA, tau=1.0)

            entry = {"seed": seed}

            if bf_used:
                t0 = time.time()
                bf_sw, bf_assignment = brute_force(B, I)
                entry["brute_force_sw"] = bf_sw
                entry["brute_force_assignment"] = list(bf_assignment)
                entry["brute_force_seconds"] = float(time.time() - t0)
                ref_optimum = bf_sw
            else:
                ref_optimum = None

            level_outcomes = {}
            for lvl, params in SA_LEVELS.items():
                sa_seed = seed * 10_000 + (hash(lvl) % 9999)
                t0 = time.time()
                _, energy, assignment = sa_solve(
                    Q, N=N, m=M,
                    n_restarts=params["n_restarts"],
                    n_iters=params["n_iters"],
                    T0=2.0, decay=0.95,
                    seed=sa_seed,
                    greedy_init_B=B,
                )
                elapsed = time.time() - t0
                sw = social_welfare(B, I, assignment)
                level_outcomes[lvl] = {
                    "sw": float(sw),
                    "energy": float(energy),
                    "assignment": list(assignment),
                    "seconds": float(elapsed),
                }
                per_level_times[lvl].append(elapsed)

            if ref_optimum is None:
                ref_optimum = max(o["sw"] for o in level_outcomes.values())

            for lvl, info in level_outcomes.items():
                ratio = info["sw"] / ref_optimum if ref_optimum != 0 else float("nan")
                info["ratio"] = float(ratio)
                per_level_ratios[lvl].append(ratio)

            entry["reference_optimum"] = float(ref_optimum)
            entry["levels"] = level_outcomes
            results[str(N)].append(entry)

        for lvl in SA_LEVELS:
            ratios = np.array(per_level_ratios[lvl])
            times = np.array(per_level_times[lvl])
            row = {
                "N": N,
                "level": lvl,
                "mean_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios)),
                "min_ratio": float(np.min(ratios)),
                "frac_optimal": float(np.mean(ratios >= 0.999)) if bf_used else None,
                "mean_seconds": float(np.mean(times)),
                "n_seeds": n_seeds,
                "brute_force_reference": bf_used,
            }
            summary_rows.append(row)
            opt_pct = (np.mean(ratios >= 0.999) * 100) if bf_used else float("nan")
            print(f"  {lvl:15s}  ratio={np.mean(ratios):.4f} ± {np.std(ratios):.4f}  "
                  f"min={np.min(ratios):.4f}  opt%={opt_pct:5.1f}  "
                  f"time={np.mean(times) * 1000:.2f}ms")

    payload = {
        "metadata": {
            "Ns": Ns, "m": M, "n_seeds": n_seeds, "lambda": LAMBDA,
            "B_scale": "uniform", "I_scale": "moderate",
            "T0": 2.0, "decay": 0.95,
            "sa_levels": SA_LEVELS,
            "total_seconds": float(time.time() - grand_start),
        },
        "summary": summary_rows,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total elapsed: {time.time() - grand_start:.1f}s")

    # Summary findings
    print("\n=== STAGE 2 SUMMARY ===")
    print("If SA-very-strong > SA-strong > SA-medium > SA-weak, SA scales correctly.")
    print("If SA-very-strong reaches >= 0.999 of brute force for small N, the QUBO is correct.")
    print("If a gap remains for small N even at SA-very-strong, the QUBO formulation is wrong.")


if __name__ == "__main__":
    main()
