"""
Aggregate benchmark.json into result.csv + result.json + plots/*.png.

Run from the repository root or from this folder; paths are relative to
this file.
"""
import json
import os
import csv
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "..", "benchmark.json")
OUT_CSV = os.path.join(HERE, "..", "result.csv")
OUT_JSON = os.path.join(HERE, "..", "result.json")
PLOT_DIR = os.path.join(HERE, "..", "plots")

EPS = 1e-6


def main():
    data = json.load(open(IN_PATH))
    recs = data["records"]

    # --- Per-config aggregation ---
    by_cfg = defaultdict(list)
    for r in recs:
        by_cfg[(r["B_scale"], r["I_scale"], r["N"])].append(r)

    rows = []
    for (bs, is_, N), rs in sorted(by_cfg.items()):
        opt = [r["opt_sw"] for r in rs]
        q = [r["qubo_sw"] for r in rs]
        nF = [r["nb3r_sw_final"] for r in rs]
        nB = [r["nb3r_sw_best"] for r in rs]
        nT = [r["nb3r_sw_tail_mean"] for r in rs]
        rows.append({
            "B_scale": bs,
            "I_scale": is_,
            "N": N,
            "n_seeds": len(rs),
            "opt_mean": statistics.mean(opt),
            "qubo_mean": statistics.mean(q),
            "nb3r_final_mean": statistics.mean(nF),
            "nb3r_best_mean": statistics.mean(nB),
            "nb3r_tail_mean": statistics.mean(nT),
            "qubo_hits_opt_pct": 100 * sum(1 for o, x in zip(opt, q) if abs(o - x) < EPS) / len(rs),
            "nb3r_best_hits_opt_pct": 100 * sum(1 for o, x in zip(opt, nB) if abs(o - x) < EPS) / len(rs),
            "nb3r_final_hits_opt_pct": 100 * sum(1 for o, x in zip(opt, nF) if abs(o - x) < EPS) / len(rs),
            "qubo_ge_nb3r_best_pct": 100 * sum(1 for a, b in zip(q, nB) if a >= b - EPS) / len(rs),
            "qubo_mean_seconds": statistics.mean(r["qubo_time"] for r in rs),
            "nb3r_mean_seconds": statistics.mean(r["nb3r_time"] for r in rs),
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")

    # --- Headline JSON ---
    total = len(recs)
    headline = {
        "n_configs": total,
        "qubo_hits_optimum": sum(1 for r in recs if abs(r["opt_sw"] - r["qubo_sw"]) < EPS),
        "nb3r_best_hits_optimum": sum(1 for r in recs if abs(r["opt_sw"] - r["nb3r_sw_best"]) < EPS),
        "nb3r_final_hits_optimum": sum(1 for r in recs if abs(r["opt_sw"] - r["nb3r_sw_final"]) < EPS),
        "qubo_ge_nb3r_final": sum(1 for r in recs if r["qubo_sw"] >= r["nb3r_sw_final"] - EPS),
        "qubo_ge_nb3r_best": sum(1 for r in recs if r["qubo_sw"] >= r["nb3r_sw_best"] - EPS),
        "nb3r_T_bumped": sum(1 for r in recs if r["nb3r_T_rounds"] == 10000),
        "total_qubo_seconds": sum(r["qubo_time"] for r in recs),
        "total_nb3r_seconds": sum(r["nb3r_time"] for r in recs),
        "benchmark_total_seconds": data["metadata"]["total_seconds"],
        "config": data["metadata"],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(headline, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    # --- Plots ---
    os.makedirs(PLOT_DIR, exist_ok=True)
    Ns = sorted({r["N"] for r in recs})
    B_scales = sorted({r["B_scale"] for r in recs})
    I_scales = ["low", "moderate", "high"]

    # Plot 1: SW / opt ratio vs N, one panel per (B_scale, I_scale)
    fig, axes = plt.subplots(len(B_scales), len(I_scales),
                             figsize=(14, 7), sharey=True, sharex=True)
    for i, bs in enumerate(B_scales):
        for j, is_ in enumerate(I_scales):
            ax = axes[i][j]
            qubo_means, qubo_stds, nb3rB_means, nb3rB_stds, nb3rF_means = [], [], [], [], []
            for N in Ns:
                subset = by_cfg[(bs, is_, N)]
                opt = np.array([r["opt_sw"] for r in subset])
                q = np.array([r["qubo_sw"] for r in subset])
                nB = np.array([r["nb3r_sw_best"] for r in subset])
                nF = np.array([r["nb3r_sw_final"] for r in subset])
                rq = q - opt
                rB = nB - opt
                rF = nF - opt
                qubo_means.append(rq.mean()); qubo_stds.append(rq.std())
                nb3rB_means.append(rB.mean()); nb3rB_stds.append(rB.std())
                nb3rF_means.append(rF.mean())
            ax.errorbar(Ns, qubo_means, yerr=qubo_stds, marker="o", label="QUBO − opt", color="C0", capsize=3)
            ax.errorbar(Ns, nb3rB_means, yerr=nb3rB_stds, marker="s", label="NB3R best − opt", color="C1", capsize=3)
            ax.plot(Ns, nb3rF_means, marker="x", linestyle="--", label="NB3R final − opt", color="C3")
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_title(f"B={bs}, I={is_}")
            ax.grid(alpha=0.3)
            if i == len(B_scales) - 1:
                ax.set_xlabel("N (flows)")
            if j == 0:
                ax.set_ylabel("SW gap vs optimum")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("DIAMOND-QUBO vs NB3R: SW gap vs brute-force optimum (mean ± std over 20 seeds)", y=1.06)
    fig.tight_layout()
    p1 = os.path.join(PLOT_DIR, "sw_gap_vs_N.png")
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p1}")

    # Plot 2: Hit-rate vs N (one line per algorithm metric, averaged over topologies)
    fig, ax = plt.subplots(figsize=(8, 5))
    qubo_hit = [100 * sum(1 for r in recs if r["N"] == N and abs(r["opt_sw"] - r["qubo_sw"]) < EPS)
                / sum(1 for r in recs if r["N"] == N) for N in Ns]
    nB_hit = [100 * sum(1 for r in recs if r["N"] == N and abs(r["opt_sw"] - r["nb3r_sw_best"]) < EPS)
              / sum(1 for r in recs if r["N"] == N) for N in Ns]
    nF_hit = [100 * sum(1 for r in recs if r["N"] == N and abs(r["opt_sw"] - r["nb3r_sw_final"]) < EPS)
              / sum(1 for r in recs if r["N"] == N) for N in Ns]
    ax.plot(Ns, qubo_hit, marker="o", label="QUBO", color="C0", lw=2)
    ax.plot(Ns, nB_hit, marker="s", label="NB3R best-visited", color="C1", lw=2)
    ax.plot(Ns, nF_hit, marker="x", label="NB3R final-sample", color="C3", lw=2)
    ax.set_xlabel("N (flows)")
    ax.set_ylabel("% configs hitting brute-force optimum")
    ax.set_ylim(-3, 103)
    ax.set_title("Optimum-hit rate vs problem size (averaged across all topologies)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(PLOT_DIR, "optimum_hit_rate.png")
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p2}")

    # Plot 3: NB3R convergence curves at one representative config
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=False)
    for j, is_ in enumerate(I_scales):
        for i, N in enumerate([4, 8]):
            ax = axes[i][j]
            subset = by_cfg[("uniform", is_, N)]
            for r in subset:
                ts, sws = zip(*r["nb3r_history"])
                ax.plot(ts, sws, color="C1", alpha=0.15, lw=0.8)
            opt_mean = statistics.mean(r["opt_sw"] for r in subset)
            qubo_mean = statistics.mean(r["qubo_sw"] for r in subset)
            ax.axhline(opt_mean, color="k", lw=1.2, ls="--", label=f"opt mean = {opt_mean:.3f}")
            ax.axhline(qubo_mean, color="C0", lw=1.2, ls=":", label=f"qubo mean = {qubo_mean:.3f}")
            ax.set_title(f"N={N}, I={is_} (uniform B, 20 seeds)")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("NB3R round t")
            if j == 0:
                ax.set_ylabel("SW")
            ax.legend(fontsize=8)
    fig.suptitle("NB3R social-welfare trajectories (all seeds, faint orange)")
    fig.tight_layout()
    p3 = os.path.join(PLOT_DIR, "nb3r_convergence.png")
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p3}")

    # Plot 4: Wall-clock vs N
    fig, ax = plt.subplots(figsize=(8, 5))
    q_t = [statistics.mean(r["qubo_time"] for r in recs if r["N"] == N) for N in Ns]
    n_t = [statistics.mean(r["nb3r_time"] for r in recs if r["N"] == N) for N in Ns]
    ax.plot(Ns, q_t, marker="o", label="DIAMOND-QUBO (one-shot SA)", color="C0", lw=2)
    ax.plot(Ns, n_t, marker="s", label="NB3R (10k rounds)", color="C1", lw=2)
    ax.set_xlabel("N (flows)")
    ax.set_ylabel("Wall-clock per config (s)")
    ax.set_title("Wall-clock cost vs N (caveat: NB3R uses Python dicts, QUBO uses numpy delta updates)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p4 = os.path.join(PLOT_DIR, "wall_clock.png")
    fig.savefig(p4, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p4}")

    # --- Console summary ---
    print()
    print("=" * 80)
    print("HEADLINE")
    print("=" * 80)
    for k in ["n_configs", "qubo_hits_optimum", "nb3r_best_hits_optimum",
              "nb3r_final_hits_optimum", "qubo_ge_nb3r_final", "qubo_ge_nb3r_best",
              "nb3r_T_bumped", "total_qubo_seconds", "total_nb3r_seconds"]:
        print(f"  {k:35s} {headline[k]}")


if __name__ == "__main__":
    main()
