"""
fix_experiments_v5.py
Round-5: PUBLICATION-READY experiment suite. This is the final result set
that goes in the thesis.

  Exp 1 — Final comparison table.
          N in {5, 10, 12, 15, 20, 30, 50}, T=1000, 20 runs each.
          Algorithms: QA-MAB Baseline, QA-MAB Fix B (tau cap 5), NB3R, Random.

  Exp 2 — Convergence trajectory at N=20, T=1000, 20 runs.
          NB3R, QA-MAB (Baseline), Random. Save full SW(t) arrays for plotting.

  Exp 3 — Statistical significance (paired t-test) of QA-MAB vs NB3R per N.

  Exp 4 — Publication plots:
          (a) Scaling:    final SW vs N with error bars.
          (b) Convergence: SW vs t at N=20 (smoothed).
          (c) Delta:      (QA-MAB - NB3R) vs N with 95% CI.

All raw arrays + summary CSV + PNGs saved under results_v5/.

Usage:
    python fix_experiments_v5.py
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB
from fix_experiments_v4 import QAMABFixed


# ------------------------- config -------------------------

OUT_DIR = "results_v5"
N_GRID = [5, 10, 12, 15, 20, 30, 50]
N_CONV = 20
T = 1000
N_RUNS = 20
M = 4
TAIL = 50  # last-K window for "final SW"


# ------------------------- runners -------------------------

def run_random(N, m, T, seed):
    rng = np.random.default_rng(seed)
    env = NetworkEnvironment(N, m, seed=seed)
    sw = np.empty(T)
    for t in range(T):
        x = {i: int(rng.integers(0, m)) for i in range(N)}
        sw[t] = env.social_welfare(x)
    return sw


def run_nb3r(N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    return NB3R(env, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed).run(T)


def run_qamab_baseline(N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    return QAMAB(env, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed).run(T)


def run_qamab_fixb(N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    qa = QAMABFixed(env, fixes=("B",), tau0=0.1, delta_tau=0.05,
                    lambda_=0.5, seed=seed)
    return qa.run(T)


ALGORITHMS = [
    ("NB3R",         run_nb3r),
    ("QA-MAB",       run_qamab_baseline),
    ("QA-MAB+FixB",  run_qamab_fixb),
    ("Random",       run_random),
]


def final_sw(history, last=TAIL):
    return float(np.mean(history[-last:]))


# ------------------------- experiments -------------------------

def exp1_comparison_table(seeds):
    """
    Returns:
        traj: dict[(alg_name, N)] -> ndarray (n_runs, T)
        finals: dict[(alg_name, N)] -> ndarray (n_runs,)
    """
    print("=" * 78)
    print("EXP 1 — Final comparison table")
    print(f"  N in {N_GRID}, T={T}, runs={len(seeds)}")
    print("=" * 78)

    traj = {}
    finals = {}

    for N in N_GRID:
        print(f"\n  --- N = {N} ---")
        for alg_name, runner in ALGORITHMS:
            t0 = time.time()
            histories = np.stack([runner(N, M, T, s) for s in seeds])
            traj[(alg_name, N)] = histories
            finals[(alg_name, N)] = histories[:, -TAIL:].mean(axis=1)
            mean = finals[(alg_name, N)].mean()
            std = finals[(alg_name, N)].std()
            print(f"    {alg_name:<14} SW={mean:>+9.3f} +/- {std:>5.2f}   "
                  f"({time.time()-t0:6.1f}s)")
    return traj, finals


def exp2_convergence(seeds):
    """N=20 trajectories for NB3R, QA-MAB, Random."""
    print("\n" + "=" * 78)
    print(f"EXP 2 — Convergence at N={N_CONV}, T={T}, runs={len(seeds)}")
    print("=" * 78)

    runners = [
        ("NB3R",   run_nb3r),
        ("QA-MAB", run_qamab_baseline),
        ("Random", run_random),
    ]
    out = {}
    for alg_name, runner in runners:
        t0 = time.time()
        histories = np.stack([runner(N_CONV, M, T, s) for s in seeds])
        out[alg_name] = histories
        print(f"  {alg_name:<10} done ({time.time()-t0:6.1f}s)  "
              f"SW(final tail)={histories[:, -TAIL:].mean():+.3f}")
    return out


def exp3_significance(finals):
    """Paired t-test QA-MAB vs NB3R for each N. Returns dict N -> (t, p)."""
    print("\n" + "=" * 78)
    print("EXP 3 — Statistical significance (paired t-test, QA-MAB vs NB3R)")
    print("=" * 78)
    print(f"  {'N':>4}  {'NB3R':>10}  {'QA-MAB':>10}  {'delta':>9}  "
          f"{'t-stat':>9}  {'p-value':>11}  sig?")
    pvals = {}
    for N in N_GRID:
        nb3r = finals[("NB3R", N)]
        qa = finals[("QA-MAB", N)]
        # Paired t-test (each seed produces a paired sample).
        t_stat, p = stats.ttest_rel(qa, nb3r)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        delta = qa.mean() - nb3r.mean()
        pvals[N] = (float(t_stat), float(p))
        print(f"  {N:>4}  {nb3r.mean():>+10.3f}  {qa.mean():>+10.3f}  "
              f"{delta:>+9.3f}  {t_stat:>+9.3f}  {p:>11.2e}  {sig}")
    return pvals


# ------------------------- plots -------------------------

def plot_scaling(finals, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    Ns = np.array(N_GRID)
    styles = {
        "NB3R":         dict(marker="o", color="#1f77b4"),
        "QA-MAB":       dict(marker="s", color="#d62728"),
        "QA-MAB+FixB":  dict(marker="^", color="#ff7f0e"),
        "Random":       dict(marker="x", color="#7f7f7f"),
    }
    for alg_name, _ in ALGORITHMS:
        means = np.array([finals[(alg_name, N)].mean() for N in N_GRID])
        # 95% CI via 1.96 * sem.
        sems = np.array([finals[(alg_name, N)].std(ddof=1)
                         / np.sqrt(len(finals[(alg_name, N)])) for N in N_GRID])
        ax.errorbar(Ns, means, yerr=1.96 * sems, label=alg_name,
                    capsize=3, lw=1.5, **styles[alg_name])
    ax.set_xlabel("Number of agents  N")
    ax.set_ylabel(f"Final social welfare  (mean of last {TAIL} steps)")
    ax.set_title(f"Scaling: SW vs N   (T={T}, {N_RUNS} runs, 95% CI)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path}")


def plot_convergence(conv, path, smooth=20):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    styles = {
        "NB3R":   dict(color="#1f77b4"),
        "QA-MAB": dict(color="#d62728"),
        "Random": dict(color="#7f7f7f", ls="--"),
    }
    kernel = np.ones(smooth) / smooth
    for alg_name, hist in conv.items():
        mean_traj = hist.mean(axis=0)
        sem_traj = hist.std(axis=0, ddof=1) / np.sqrt(hist.shape[0])
        # Smooth via moving average for readability.
        ms = np.convolve(mean_traj, kernel, mode="valid")
        ss = np.convolve(sem_traj, kernel, mode="valid")
        xs = np.arange(len(ms)) + smooth // 2
        ax.plot(xs, ms, label=alg_name, lw=1.8, **styles[alg_name])
        ax.fill_between(xs, ms - 1.96 * ss, ms + 1.96 * ss,
                        alpha=0.18, color=styles[alg_name]["color"])
    ax.set_xlabel("Timestep  t")
    ax.set_ylabel(f"Social welfare  (smoothed, window={smooth})")
    ax.set_title(f"Convergence at N={N_CONV}   (T={T}, {N_RUNS} runs, 95% CI)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path}")


def plot_delta(finals, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    Ns = np.array(N_GRID)
    deltas = np.array([finals[("QA-MAB", N)].mean()
                       - finals[("NB3R", N)].mean() for N in N_GRID])
    # Paired-difference 95% CI.
    sems = []
    for N in N_GRID:
        diffs = finals[("QA-MAB", N)] - finals[("NB3R", N)]
        sems.append(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    sems = np.array(sems)
    ax.errorbar(Ns, deltas, yerr=1.96 * sems, marker="o", color="#2ca02c",
                lw=1.8, capsize=3, label="QA-MAB - NB3R")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Number of agents  N")
    ax.set_ylabel(r"$\Delta$ Social welfare  (QA-MAB - NB3R)")
    ax.set_title(f"QA-MAB advantage over NB3R   (T={T}, {N_RUNS} runs, 95% CI)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path}")


# ------------------------- save -------------------------

def save_all(finals, traj, conv, pvals):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Raw arrays.
    np.savez(
        os.path.join(OUT_DIR, "raw_data.npz"),
        N_grid=np.array(N_GRID),
        T=T, n_runs=N_RUNS, m=M, tail=TAIL,
        # finals[(alg, N)] flattened as alg_N keys.
        **{f"finals__{alg}__N{N}": finals[(alg, N)]
           for (alg, _) in ALGORITHMS for N in N_GRID},
        **{f"traj__{alg}__N{N}": traj[(alg, N)]
           for (alg, _) in ALGORITHMS for N in N_GRID},
        **{f"conv__{alg}": conv[alg] for alg in conv},
    )
    print(f"  saved {OUT_DIR}/raw_data.npz")

    # Summary CSV — one row per (alg, N).
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "N", "mean_SW", "std_SW", "sem_SW",
                    "n_runs", "p_vs_NB3R"])
        for alg_name, _ in ALGORITHMS:
            for N in N_GRID:
                arr = finals[(alg_name, N)]
                p = pvals[N][1] if alg_name == "QA-MAB" else ""
                w.writerow([alg_name, N,
                            f"{arr.mean():.4f}",
                            f"{arr.std(ddof=1):.4f}",
                            f"{arr.std(ddof=1) / np.sqrt(len(arr)):.4f}",
                            len(arr), p])
    print(f"  saved {csv_path}")


# ------------------------- main -------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = [42 + r for r in range(N_RUNS)]
    t_start = time.time()

    traj, finals = exp1_comparison_table(seeds)
    conv = exp2_convergence(seeds)
    pvals = exp3_significance(finals)

    print("\n" + "=" * 78)
    print("EXP 4 — Saving plots & data")
    print("=" * 78)
    save_all(finals, traj, conv, pvals)
    plot_scaling(finals, os.path.join(OUT_DIR, "scaling.png"))
    plot_convergence(conv, os.path.join(OUT_DIR, "convergence_N20.png"))
    plot_delta(finals, os.path.join(OUT_DIR, "delta_qamab_vs_nb3r.png"))

    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")
    print(f"All outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
