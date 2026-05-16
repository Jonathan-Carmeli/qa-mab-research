"""
Stage 5: Learning Dynamics
==========================

Run QA-MAB (with learning) for T=5000 steps at N=15, 30 seeds.
Track ||u_hat[t] - B||_F and ||I_hat[t] - I||_F at fixed checkpoints,
plus SW_history vs the true optimum (estimated by SA-very-strong).

This isolates whether the gap to NB3R is from u_hat/I_hat failing to
converge to B/I, separately from any solver inaccuracy.

Output:
  results/convergence_test/learning_dynamics.png
  results/convergence_test/learning_dynamics.json
"""

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
from qa_mab import QAMAB

import importlib.util
_stage2_path = os.path.join(os.path.dirname(__file__), "02_sa_quality_sweep.py")
_spec = importlib.util.spec_from_file_location("stage2", _stage2_path)
stage2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage2)


N = 15
M = 4
T = 5000
N_SEEDS = 30
CHECKPOINTS = [100, 500, 1000, 2000, 3000, 4000, 5000]


def run_one_seed(seed):
    env = NetworkEnvironment(N=N, m=M, seed=seed)
    qamab = QAMAB(env, seed=seed)

    sw_history = np.zeros(T)
    u_err_chk = {}
    I_err_chk = {}

    for t in range(T):
        qamab.step()
        sw_history[t] = qamab.history[-1]
        if (t + 1) in CHECKPOINTS:
            u_err = float(np.linalg.norm(qamab.u_hat - env.B, ord="fro"))
            I_err = float(np.linalg.norm((qamab.I_hat - env.I).reshape(-1), ord=2))
            u_err_chk[t + 1] = u_err
            I_err_chk[t + 1] = I_err

    # True optimum via SA-very-strong on oracle QUBO
    Q = stage2.build_oracle_qubo(env.B, env.I, lambda_=0.5, tau=1.0)
    _, _, asn = stage2.sa_solve(
        Q, N=N, m=M,
        n_restarts=1000, n_iters=5000,
        T0=2.0, decay=0.95,
        seed=seed * 99991,
        greedy_init_B=env.B,
    )
    true_opt = stage2.social_welfare(env.B, env.I, asn)

    return sw_history, u_err_chk, I_err_chk, true_opt


def main():
    out_dir = os.path.join(ROOT, "simulations", "results", "convergence_test")
    os.makedirs(out_dir, exist_ok=True)

    grand_start = time.time()
    print(f"\n[Stage 5] QA-MAB learning dynamics: N={N}, T={T}, {N_SEEDS} seeds")

    sw_runs = np.zeros((N_SEEDS, T))
    true_opts = np.zeros(N_SEEDS)
    u_err_table = {chk: np.zeros(N_SEEDS) for chk in CHECKPOINTS}
    I_err_table = {chk: np.zeros(N_SEEDS) for chk in CHECKPOINTS}

    for s in range(N_SEEDS):
        t0 = time.time()
        sw_history, u_chk, I_chk, true_opt = run_one_seed(seed=s)
        sw_runs[s] = sw_history
        true_opts[s] = true_opt
        for chk in CHECKPOINTS:
            u_err_table[chk][s] = u_chk[chk]
            I_err_table[chk][s] = I_chk[chk]
        elapsed = time.time() - t0
        approx = sw_history[-1] / true_opt if true_opt != 0 else float("nan")
        print(f"  seed={s:2d}  final SW={sw_history[-1]:.3f}  "
              f"true opt~{true_opt:.3f}  approx={approx:.3f}  ({elapsed:.1f}s)")

    # ----- Plot -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: u_hat error vs t
    ax = axes[0]
    means = np.array([u_err_table[c].mean() for c in CHECKPOINTS])
    stds = np.array([u_err_table[c].std() for c in CHECKPOINTS])
    ax.errorbar(CHECKPOINTS, means, yerr=stds, marker="o",
                capsize=3, color="#1f77b4", label="||u_hat - B||_F")
    ax.set_xlabel("step t")
    ax.set_ylabel("Frobenius error")
    ax.set_title("u_hat estimation error (mean ± std)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: I_hat error vs t
    ax = axes[1]
    means = np.array([I_err_table[c].mean() for c in CHECKPOINTS])
    stds = np.array([I_err_table[c].std() for c in CHECKPOINTS])
    ax.errorbar(CHECKPOINTS, means, yerr=stds, marker="o",
                capsize=3, color="#d62728", label="||I_hat - I||_F")
    ax.set_xlabel("step t")
    ax.set_ylabel("Frobenius error")
    ax.set_title("I_hat estimation error (mean ± std)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: SW vs t with true opt (per-seed normalized)
    ax = axes[2]
    sw_mean = sw_runs.mean(axis=0)
    sw_std = sw_runs.std(axis=0)
    ts = np.arange(1, T + 1)
    ax.plot(ts, sw_mean, color="#2ca02c", label="QA-MAB mean SW")
    ax.fill_between(ts, sw_mean - sw_std, sw_mean + sw_std,
                    color="#2ca02c", alpha=0.2, label="±1 std")
    ax.axhline(true_opts.mean(), color="black", linestyle="--",
               linewidth=0.8, label=f"true opt mean={true_opts.mean():.2f}")
    ax.set_xlabel("step t")
    ax.set_ylabel("Social Welfare")
    ax.set_title("QA-MAB SW vs true optimum")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"Stage 5: QA-MAB learning dynamics  (N={N}, T={T}, {N_SEEDS} seeds)")
    fig.tight_layout()
    out_png = os.path.join(out_dir, "learning_dynamics.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out_png}")

    # ----- JSON -----
    payload = {
        "metadata": {
            "N": N, "m": M, "T": T, "n_seeds": N_SEEDS,
            "checkpoints": CHECKPOINTS,
            "total_seconds": float(time.time() - grand_start),
        },
        "u_hat_frobenius_error": {
            str(c): {
                "mean": float(u_err_table[c].mean()),
                "std": float(u_err_table[c].std()),
                "values": u_err_table[c].tolist(),
            } for c in CHECKPOINTS
        },
        "I_hat_frobenius_error": {
            str(c): {
                "mean": float(I_err_table[c].mean()),
                "std": float(I_err_table[c].std()),
                "values": I_err_table[c].tolist(),
            } for c in CHECKPOINTS
        },
        "sw_history_mean": sw_mean.tolist(),
        "sw_history_std": sw_std.tolist(),
        "true_optima": true_opts.tolist(),
        "true_optima_mean": float(true_opts.mean()),
        "final_approx_ratio_mean": float(np.mean(sw_runs[:, -1] / true_opts)),
        "final_approx_ratio_std": float(np.std(sw_runs[:, -1] / true_opts)),
    }
    out_json = os.path.join(out_dir, "learning_dynamics.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")
    print(f"Total elapsed: {time.time() - grand_start:.1f}s")

    print("\n=== STAGE 5 SUMMARY ===")
    print(f"u_hat error trajectory:")
    for c in CHECKPOINTS:
        print(f"  t={c:5d}  ||u_hat-B||_F = {u_err_table[c].mean():.4f}")
    print(f"I_hat error trajectory:")
    for c in CHECKPOINTS:
        print(f"  t={c:5d}  ||I_hat-I||_F = {I_err_table[c].mean():.4f}")
    final_ratios = sw_runs[:, -1] / true_opts
    print(f"Final approx ratio (mean ± std): {final_ratios.mean():.4f} ± {final_ratios.std():.4f}")
    print("If u_hat / I_hat errors decay AND final ratio approaches 1.0, learning converges.")
    print("If errors plateau OR ratio plateaus below 1.0, learning is the bottleneck.")


if __name__ == "__main__":
    main()
