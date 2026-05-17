"""
Driver: benchmark DIAMOND-QUBO (centralized one-shot SA on the full oracle
QUBO) against paper-faithful NB3R (distributed iterative Boltzmann
sampling). Both run in oracle mode (B and I are passed directly from the
environment, no learning).

Hypothesis: at small N where NB3R has time to converge to its stationary
distribution, does a single heavy SA solve match or beat the converged
NB3R social welfare across topologies?

Output:
    simulations/results/diamond_vs_qubo/benchmark.json
"""

import argparse
import itertools
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "simulations", "legacy"))
sys.path.insert(0, os.path.join(ROOT, "simulations", "experiments"))

from simulation_core import NetworkEnvironment  # noqa: E402

from diamond_vs_qubo.diamond_qubo import (  # noqa: E402
    brute_force,
    solve_diamond_qubo,
)
from diamond_vs_qubo.nb3r_paper import run_nb3r, stable_tail  # noqa: E402


M = 4

DEFAULT_NS = [4, 5, 6, 7, 8]
DEFAULT_B_SCALES = ["uniform", "skewed"]
DEFAULT_I_SCALES = ["low", "moderate", "high"]
DEFAULT_N_SEEDS = 20
DEFAULT_T_ROUNDS = 5000
BUMPED_T_ROUNDS = 10000

SMOKE_NS = [4]
SMOKE_B_SCALES = ["uniform"]
SMOKE_I_SCALES = ["low"]
SMOKE_N_SEEDS = 1
SMOKE_T_ROUNDS = 500


def run_one(N, B_scale, I_scale, seed, T_rounds):
    env = NetworkEnvironment(N=N, m=M, seed=seed, B_scale=B_scale, I_scale=I_scale)

    opt_sw, opt_assign = brute_force(env.B, env.I)

    q_assign, q_sw, q_time = solve_diamond_qubo(
        env, n_restarts=60, n_iters=2000, seed=seed
    )

    _, n_hist, n_time = run_nb3r(env, T_rounds=T_rounds, seed=seed)
    used_T = T_rounds
    if not stable_tail(n_hist, frac=0.1, tol=0.01):
        _, n_hist, n_time = run_nb3r(env, T_rounds=BUMPED_T_ROUNDS, seed=seed)
        used_T = BUMPED_T_ROUNDS

    sw_vals = [sw for _, sw in n_hist]
    n_sw_final = sw_vals[-1]
    n_sw_best = max(sw_vals)
    tail_n = max(1, len(sw_vals) // 10)
    n_sw_tail_mean = sum(sw_vals[-tail_n:]) / tail_n

    # Sanity asserts.
    assert q_sw <= opt_sw + 1e-9, f"qubo_sw {q_sw} > opt_sw {opt_sw}"
    assert n_sw_best <= opt_sw + 1e-9, f"nb3r_sw_best {n_sw_best} > opt_sw {opt_sw}"

    return {
        "N": N,
        "B_scale": B_scale,
        "I_scale": I_scale,
        "seed": seed,
        "opt_sw": opt_sw,
        "opt_assignment": list(opt_assign),
        "qubo_sw": q_sw,
        "qubo_assignment": [q_assign[i] for i in range(N)],
        "qubo_time": q_time,
        "nb3r_sw_final": n_sw_final,
        "nb3r_sw_best": n_sw_best,
        "nb3r_sw_tail_mean": n_sw_tail_mean,
        "nb3r_time": n_time,
        "nb3r_history": [[t, sw] for t, sw in n_hist],
        "nb3r_T_rounds": used_T,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run for syntax/sanity (N=4, 1 seed, T=500).")
    parser.add_argument("--out", default=None,
                        help="Override output JSON path.")
    args = parser.parse_args()

    if args.smoke:
        Ns = SMOKE_NS
        B_scales = SMOKE_B_SCALES
        I_scales = SMOKE_I_SCALES
        n_seeds = SMOKE_N_SEEDS
        T_rounds = SMOKE_T_ROUNDS
        out_path = args.out or os.path.join(
            ROOT, "simulations", "results", "diamond_vs_qubo", "benchmark_smoke.json")
    else:
        Ns = DEFAULT_NS
        B_scales = DEFAULT_B_SCALES
        I_scales = DEFAULT_I_SCALES
        n_seeds = DEFAULT_N_SEEDS
        T_rounds = DEFAULT_T_ROUNDS
        out_path = args.out or os.path.join(
            ROOT, "simulations", "results", "diamond_vs_qubo", "benchmark.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = len(Ns) * len(B_scales) * len(I_scales) * n_seeds
    print(f"Running {total} configs -> {out_path}")
    grand_start = time.time()

    records = []
    done = 0
    for N, B_scale, I_scale, seed in itertools.product(Ns, B_scales, I_scales, range(n_seeds)):
        rec = run_one(N, B_scale, I_scale, seed, T_rounds)
        records.append(rec)
        done += 1
        print(f"[{done}/{total}] N={N} B={B_scale} I={I_scale} seed={seed}  "
              f"opt={rec['opt_sw']:.4f}  qubo={rec['qubo_sw']:.4f}  "
              f"nb3r_final={rec['nb3r_sw_final']:.4f}  "
              f"nb3r_best={rec['nb3r_sw_best']:.4f}  "
              f"nb3r_tail={rec['nb3r_sw_tail_mean']:.4f}  "
              f"T={rec['nb3r_T_rounds']}  "
              f"qt={rec['qubo_time']*1000:.0f}ms  nt={rec['nb3r_time']*1000:.0f}ms")

    payload = {
        "metadata": {
            "smoke": args.smoke,
            "Ns": Ns,
            "m": M,
            "B_scales": B_scales,
            "I_scales": I_scales,
            "n_seeds": n_seeds,
            "T_rounds": T_rounds,
            "bumped_T_rounds": BUMPED_T_ROUNDS,
            "qubo_sa_n_restarts": 60,
            "qubo_sa_n_iters": 2000,
            "total_seconds": float(time.time() - grand_start),
            "n_records": len(records),
        },
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Total elapsed: {time.time() - grand_start:.1f}s")


if __name__ == "__main__":
    main()
