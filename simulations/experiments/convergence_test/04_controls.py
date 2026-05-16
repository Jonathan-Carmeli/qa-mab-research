"""
Stage 4: Controls
=================

Sweep design choices that could affect SA approximation ratio at N=10:
  - B distribution: Uniform[0.5,1.0], Uniform[0,1.0], Uniform[0.5,2.0]
  - I distribution: Uniform[0,0.2], Uniform[0,0.5], Uniform[0,1.0]
  - lambda:         1.0, 2.0, 5.0, 10.0
  - Sparsity:       I-sparse (10% non-zero), I-dense (all non-zero)

For each config (20 seeds), build the oracle QUBO, run SA-strong, and
compute approximation ratio against an SA-very-strong reference run
(N=10 brute force is 1M+ assignments, prohibitive for 12 sweeps x 20
seeds, so we use a strong-SA proxy as the reference).

Output: results/convergence_test/controls.json
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simulations"))
sys.path.insert(0, os.path.dirname(__file__))

import importlib.util
_stage2_path = os.path.join(os.path.dirname(__file__), "02_sa_quality_sweep.py")
_spec = importlib.util.spec_from_file_location("stage2", _stage2_path)
stage2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage2)


N = 10
M = 4
N_SEEDS = 20

SA_STRONG = dict(n_restarts=200, n_iters=1000)
SA_REFERENCE = dict(n_restarts=1000, n_iters=5000)

# B uniform on [low, high]
B_DISTRIBUTIONS = {
    "B_U(0.5,1.0)": (0.5, 1.0),
    "B_U(0,1.0)":   (0.0, 1.0),
    "B_U(0.5,2.0)": (0.5, 2.0),
}

# I uniform on [0, max] (default sparsity = dense)
I_DISTRIBUTIONS = {
    "I_U(0,0.2)": 0.2,
    "I_U(0,0.5)": 0.5,
    "I_U(0,1.0)": 1.0,
}

LAMBDAS = [1.0, 2.0, 5.0, 10.0]

# Sparsity sweeps (B fixed @ U(0.5,1.0), I fixed @ U(0,0.2), lambda=0.5)
SPARSITY = {
    "I-sparse-10%": 0.10,
    "I-dense":      1.00,
}


def make_BI(seed, b_low, b_high, i_max, sparsity, N=N, m=M):
    rng = np.random.default_rng(seed)
    B = rng.uniform(b_low, b_high, size=(N, m))
    I = rng.uniform(0.0, i_max, size=(N, m, N, m))
    if sparsity < 1.0:
        mask = rng.random(size=I.shape) < sparsity
        I = I * mask
    for i in range(N):
        I[i, :, i, :] = 0.0
    return B, I


def run_config(name, b_low, b_high, i_max, lambda_, sparsity):
    """Run SA-strong vs SA-reference for n_seeds, return summary."""
    ratios = []
    sa_strong_times = []
    ref_times = []
    seeds_data = []

    for seed in range(N_SEEDS):
        B, I = make_BI(seed, b_low, b_high, i_max, sparsity)
        Q = stage2.build_oracle_qubo(B, I, lambda_=lambda_, tau=1.0)

        # Reference (SA-very-strong)
        t0 = time.time()
        _, _, ref_asn = stage2.sa_solve(
            Q, N=N, m=M,
            n_restarts=SA_REFERENCE["n_restarts"],
            n_iters=SA_REFERENCE["n_iters"],
            T0=2.0, decay=0.95,
            seed=seed * 31337 + (hash(name) % 31337),
            greedy_init_B=B,
        )
        ref_elapsed = time.time() - t0
        ref_sw = stage2.social_welfare(B, I, ref_asn)

        # SA-strong
        t0 = time.time()
        _, _, asn = stage2.sa_solve(
            Q, N=N, m=M,
            n_restarts=SA_STRONG["n_restarts"],
            n_iters=SA_STRONG["n_iters"],
            T0=2.0, decay=0.95,
            seed=seed * 7919 + (hash(name) % 7919),
            greedy_init_B=B,
        )
        elapsed = time.time() - t0
        sw = stage2.social_welfare(B, I, asn)
        ratio = sw / ref_sw if ref_sw != 0 else float("nan")

        ratios.append(ratio)
        sa_strong_times.append(elapsed)
        ref_times.append(ref_elapsed)
        seeds_data.append({
            "seed": seed,
            "ref_sw": float(ref_sw),
            "sa_strong_sw": float(sw),
            "ratio": float(ratio),
            "sa_strong_seconds": float(elapsed),
            "ref_seconds": float(ref_elapsed),
        })

    ratios = np.array(ratios)
    return {
        "config": name,
        "N": N, "m": M, "n_seeds": N_SEEDS,
        "B_low": b_low, "B_high": b_high,
        "I_max": i_max, "sparsity": sparsity, "lambda": lambda_,
        "mean_ratio": float(ratios.mean()),
        "std_ratio": float(ratios.std()),
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
        "mean_sa_strong_seconds": float(np.mean(sa_strong_times)),
        "mean_ref_seconds": float(np.mean(ref_times)),
        "seeds": seeds_data,
    }


def main():
    out_dir = os.path.join(ROOT, "simulations", "results", "convergence_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "controls.json")

    grand_start = time.time()
    all_results = []

    print(f"\n[Stage 4] N={N}, m={M}, {N_SEEDS} seeds per config")

    # 1) B distributions  (I = U(0,0.2), lambda=0.5, dense)
    print("\n--- B distribution sweep ---")
    for name, (b_low, b_high) in B_DISTRIBUTIONS.items():
        r = run_config(f"B={name}", b_low, b_high, i_max=0.2, lambda_=0.5, sparsity=1.0)
        all_results.append(r)
        print(f"  {name:18s}  ratio={r['mean_ratio']:.4f} ± {r['std_ratio']:.4f}  "
              f"min={r['min_ratio']:.4f}")

    # 2) I distributions  (B = U(0.5,1.0), lambda=0.5, dense)
    print("\n--- I distribution sweep ---")
    for name, i_max in I_DISTRIBUTIONS.items():
        r = run_config(f"I={name}", b_low=0.5, b_high=1.0, i_max=i_max,
                       lambda_=0.5, sparsity=1.0)
        all_results.append(r)
        print(f"  {name:18s}  ratio={r['mean_ratio']:.4f} ± {r['std_ratio']:.4f}  "
              f"min={r['min_ratio']:.4f}")

    # 3) Lambda sweep   (B = U(0.5,1.0), I = U(0,0.2), dense)
    print("\n--- Lambda sweep ---")
    for lam in LAMBDAS:
        r = run_config(f"lambda={lam}", b_low=0.5, b_high=1.0, i_max=0.2,
                       lambda_=lam, sparsity=1.0)
        all_results.append(r)
        print(f"  lambda={lam:5.1f}        ratio={r['mean_ratio']:.4f} ± {r['std_ratio']:.4f}  "
              f"min={r['min_ratio']:.4f}")

    # 4) Sparsity sweep (B = U(0.5,1.0), I = U(0,0.2), lambda=0.5)
    print("\n--- Sparsity sweep ---")
    for name, frac in SPARSITY.items():
        r = run_config(f"sparsity={name}", b_low=0.5, b_high=1.0, i_max=0.2,
                       lambda_=0.5, sparsity=frac)
        all_results.append(r)
        print(f"  {name:18s}  ratio={r['mean_ratio']:.4f} ± {r['std_ratio']:.4f}  "
              f"min={r['min_ratio']:.4f}")

    payload = {
        "metadata": {
            "N": N, "m": M, "n_seeds": N_SEEDS,
            "sa_strong": SA_STRONG, "sa_reference": SA_REFERENCE,
            "total_seconds": float(time.time() - grand_start),
        },
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total elapsed: {time.time() - grand_start:.1f}s")

    # Summary findings
    print("\n=== STAGE 4 SUMMARY ===")
    worst = min(all_results, key=lambda r: r["mean_ratio"])
    best = max(all_results, key=lambda r: r["mean_ratio"])
    print(f"Worst config (lowest SA-strong ratio): {worst['config']}  ->  {worst['mean_ratio']:.4f}")
    print(f"Best config:  {best['config']}  ->  {best['mean_ratio']:.4f}")
    print("If ratios stay close to 1.0 across all configs, SA is robust to QUBO parameters.")
    print("If ratios collapse for high I_max or large lambda, SA struggles with rough landscapes.")


if __name__ == "__main__":
    main()
