#!/usr/bin/env python3
"""Cat 5: Regret Convergence — QA-MAB vs NB3R crossover analysis.

Run with more seeds for statistical significance.
P=5, T=100 per seed. 5 seeds per N. Full N range.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import json
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

OUT = "simulations/results/validation_cat5_physical/"
os.makedirs(OUT, exist_ok=True)

from simulations.physical_validation.physical_env import AbstractWorld
from simulations.physical_validation.qa_mab_physical import QAMABPhysical
from simulations.physical_validation.agents_physical import NB3RAgent

N_vals = [5, 8, 10, 12, 15, 20, 30]
P, T, n_seeds = 5, 100, 5

rows = []
crossover_N = None
for N in N_vals:
    print(f"  N={N}", flush=True)
    qa_finals = []; nb3_finals = []
    for si in range(n_seeds):
        seed = 42 + si
        rng  = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 1000)

        world = AbstractWorld(N=N, K=4, m=20, Z=6, seed=seed)
        qa    = QAMABPhysical(world, seed=seed)
        nb3   = NB3RAgent(world, seed=seed)

        qa_losses = []; nb3_losses = []

        for p in range(P):
            # Shared deterministic topology: qa.reset_epoch refreshes the world
            # with the canonical epoch seed; nb3.reset_epoch() leaves the world
            # untouched, so both agents see the same pathset.
            epoch_seed = seed * 100_000 + p
            qa.reset_epoch(p, world_rng=np.random.default_rng(epoch_seed))
            nb3.reset_epoch()
            for t in range(T):
                c_qa = qa.act(t, p);  l_qa = world.compute_losses(c_qa, rng)
                qa.update(c_qa, l_qa); qa_losses.append(l_qa.mean())
                c_nb = nb3.act(t, p); l_nb = world.compute_losses(c_nb, rng2)
                nb3.update(c_nb, l_nb); nb3_losses.append(l_nb.mean())

        qa_finals.append(np.mean(qa_losses[-10:]))
        nb3_finals.append(np.mean(nb3_losses[-10:]))

    qa_arr = np.array(qa_finals); nb3_arr = np.array(nb3_finals)
    win_rate = float((qa_arr < nb3_arr).mean())
    t_stat, p_val = stats.ttest_rel(qa_arr, nb3_arr) if n_seeds > 1 else (0.0, 1.0)
    qa_mean = float(qa_arr.mean()); nb3_mean = float(nb3_arr.mean())
    rows.append({"N": N, "qa_mean": qa_mean, "nb3r_mean": nb3_mean,
                 "win_rate": win_rate, "p_value": float(p_val)})
    print(f"    QA={qa_mean:.3f}  NB3R={nb3_mean:.3f}  win={win_rate:.1%}  p={p_val:.3f}")

    if crossover_N is None and win_rate >= 0.80 and p_val < 0.05:
        crossover_N = N

pass_cat5 = crossover_N is not None
result = {
    "pass": pass_cat5,
    "reason": f"Crossover at N={crossover_N}: {'PASS' if pass_cat5 else 'FAIL'}",
    "crossover_N": crossover_N,
    "rows": rows,
}
with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

with open(OUT + "result.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["N", "qa_mean", "nb3r_mean", "win_rate", "p_value"])
    w.writeheader(); w.writerows(rows)

N_plot = [r["N"] for r in rows]
wr     = [r["win_rate"] for r in rows]
plt.figure(figsize=(8, 4))
plt.plot(N_plot, wr, marker='o', color='tab:blue', lw=2)
plt.axhline(0.80, color='tab:red',  linestyle='--', label='80% threshold')
plt.axhline(0.50, color='gray',     linestyle=':',  label='50%')
plt.xlabel("N flows"); plt.ylabel("QA-MAB win rate")
plt.title("QA-MAB vs NB3R: Crossover Analysis")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(OUT + "win_rate.png", dpi=150); plt.close()

x = np.arange(len(N_plot))
plt.figure(figsize=(8, 4))
plt.plot(x, [r["qa_mean"]   for r in rows], marker='o', label='QA-MAB')
plt.plot(x, [r["nb3r_mean"] for r in rows], marker='s', label='NB3R')
plt.xticks(x, N_plot); plt.xlabel("N flows"); plt.ylabel("Mean Final Loss")
plt.title("QA-MAB vs NB3R: Mean Loss")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(OUT + "mean_loss.png", dpi=150); plt.close()

print(f"Cat 5: crossover_N={crossover_N} — {'PASS' if pass_cat5 else 'FAIL'}")
