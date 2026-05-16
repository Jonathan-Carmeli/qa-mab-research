"""
Stage 3: Scaling Analysis
=========================

For N in {4, 5, 6, 7, 8, 10, 12, 15}, 50 seeds, compute:
  - true optimum (brute force when N <= 8, else SA-very-strong proxy)
  - SA-weak approximation
  - SA-strong approximation

Plot:
  - approximation ratio vs N (mean +/- std error bars) for both levels
  - approximation ratio vs compute time (per level)

Output:
  results/convergence_test/scaling_analysis.png
  results/convergence_test/scaling_analysis.json
"""

import itertools
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simulations"))
sys.path.insert(0, os.path.dirname(__file__))

from simulation_core import NetworkEnvironment

# Reuse the SA + QUBO from stage 2
import importlib.util
_stage2_path = os.path.join(os.path.dirname(__file__), "02_sa_quality_sweep.py")
_spec = importlib.util.spec_from_file_location("stage2", _stage2_path)
stage2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage2)


SA_LEVELS = {
    "SA-weak":        dict(n_restarts=8,    n_iters=15),
    "SA-strong":      dict(n_restarts=200,  n_iters=1000),
}

# For N > 8 we can't brute-force; use a strong reference run
SA_REFERENCE = dict(n_restarts=1000, n_iters=5000)

LAMBDA = 0.5
M = 4


def main():
    out_dir = os.path.join(ROOT, "simulations", "results", "convergence_test")
    os.makedirs(out_dir, exist_ok=True)

    Ns = [4, 5, 6, 7, 8, 10, 12, 15]
    n_seeds = 50

    # ratios[level][N] = list of approximation ratios across seeds
    ratios = {lvl: {N: [] for N in Ns} for lvl in SA_LEVELS}
    times = {lvl: {N: [] for N in Ns} for lvl in SA_LEVELS}
    raw_per_seed = {str(N): [] for N in Ns}

    grand_start = time.time()
    for N in Ns:
        bf_used = N <= 8
        print(f"\n[N={N}] {n_seeds} seeds, brute_force={'yes' if bf_used else 'no'}")
        for seed in range(n_seeds):
            env = NetworkEnvironment(N=N, m=M, seed=seed)
            B, I = env.B, env.I
            Q = stage2.build_oracle_qubo(B, I, lambda_=LAMBDA, tau=1.0)

            if bf_used:
                bf_sw, _ = stage2.brute_force(B, I)
                ref_opt = bf_sw
            else:
                _, _, asn = stage2.sa_solve(
                    Q, N=N, m=M,
                    n_restarts=SA_REFERENCE["n_restarts"],
                    n_iters=SA_REFERENCE["n_iters"],
                    T0=2.0, decay=0.95,
                    seed=seed * 100_001,
                    greedy_init_B=B,
                )
                ref_opt = stage2.social_welfare(B, I, asn)

            entry = {"seed": seed, "reference_optimum": float(ref_opt)}
            for lvl, params in SA_LEVELS.items():
                sa_seed = seed * 7919 + (hash(lvl) % 7919)
                t0 = time.time()
                _, _, asn = stage2.sa_solve(
                    Q, N=N, m=M,
                    n_restarts=params["n_restarts"],
                    n_iters=params["n_iters"],
                    T0=2.0, decay=0.95,
                    seed=sa_seed,
                    greedy_init_B=B,
                )
                elapsed = time.time() - t0
                sw = stage2.social_welfare(B, I, asn)
                ratio = sw / ref_opt if ref_opt != 0 else float("nan")
                ratios[lvl][N].append(ratio)
                times[lvl][N].append(elapsed)
                entry[lvl] = {"sw": float(sw), "ratio": float(ratio), "seconds": float(elapsed)}
            raw_per_seed[str(N)].append(entry)

        for lvl in SA_LEVELS:
            r = np.array(ratios[lvl][N])
            t = np.array(times[lvl][N])
            print(f"  {lvl:10s}  ratio={r.mean():.4f} ± {r.std():.4f}  "
                  f"time={t.mean() * 1000:.2f}ms")

    # ----- Plot -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    colors = {"SA-weak": "#d95f02", "SA-strong": "#1b9e77"}
    for lvl in SA_LEVELS:
        means = np.array([np.mean(ratios[lvl][N]) for N in Ns])
        stderr = np.array([np.std(ratios[lvl][N]) / np.sqrt(n_seeds) for N in Ns])
        ax.errorbar(Ns, means, yerr=stderr, marker="o",
                    color=colors.get(lvl, None), label=lvl,
                    capsize=3, linewidth=1.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="optimum")
    ax.set_xlabel("N (number of agents)")
    ax.set_ylabel("approximation ratio (SW_SA / SW_optimum)")
    ax.set_title("Approximation ratio vs N (mean ± SE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for lvl in SA_LEVELS:
        for N in Ns:
            r = ratios[lvl][N]
            t = times[lvl][N]
            ax.scatter(t, r, alpha=0.4, s=15,
                       color=colors.get(lvl, None),
                       label=lvl if N == Ns[0] else None)
    ax.set_xscale("log")
    ax.set_xlabel("compute time (s)")
    ax.set_ylabel("approximation ratio")
    ax.set_title("Ratio vs compute time")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Stage 3: SA scaling on oracle QUBO ({n_seeds} seeds, m={M}, lambda={LAMBDA})")
    fig.tight_layout()
    out_png = os.path.join(out_dir, "scaling_analysis.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out_png}")

    # ----- JSON -----
    summary = []
    for lvl in SA_LEVELS:
        for N in Ns:
            r = np.array(ratios[lvl][N])
            t = np.array(times[lvl][N])
            summary.append({
                "level": lvl, "N": N,
                "mean_ratio": float(r.mean()),
                "std_ratio": float(r.std()),
                "stderr_ratio": float(r.std() / np.sqrt(n_seeds)),
                "mean_seconds": float(t.mean()),
                "n_seeds": n_seeds,
            })
    payload = {
        "metadata": {
            "Ns": Ns, "m": M, "n_seeds": n_seeds, "lambda": LAMBDA,
            "sa_levels": SA_LEVELS, "sa_reference": SA_REFERENCE,
            "total_seconds": float(time.time() - grand_start),
        },
        "summary": summary,
        "results": raw_per_seed,
    }
    out_json = os.path.join(out_dir, "scaling_analysis.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")
    print(f"Total elapsed: {time.time() - grand_start:.1f}s")

    print("\n=== STAGE 3 SUMMARY ===")
    for lvl in SA_LEVELS:
        means = [np.mean(ratios[lvl][N]) for N in Ns]
        print(f"  {lvl:10s}  ratios across N: " +
              ", ".join(f"N={N}:{m:.3f}" for N, m in zip(Ns, means)))


if __name__ == "__main__":
    main()
