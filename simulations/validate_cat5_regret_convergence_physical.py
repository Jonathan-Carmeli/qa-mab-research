#!/usr/bin/env python3
"""Cat 5: Regret Convergence — find crossover N where QA-MAB beats NB3R."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "simulations/results/validation_cat5_physical/"
os.makedirs(OUT, exist_ok=True)

from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.agents_physical import NB3RAgent

N_vals = [5, 8, 10, 12, 15, 20, 30]
P, T, n_seeds = 5, 100, 20

rows = []
crossover_N = None
for N in N_vals:
    print(f"  N={N}")
    qa_losses, nb3r_losses = [], []
    for si in range(n_seeds):
        seed = 42 + si
        rng = np.random.default_rng(seed)
        world = AbstractWorld(N=N, K=4, m=20, Z=6, seed=seed)
        qa  = QAMABPhysical(world, seed=seed)
        nb3 = NB3RAgent(world, seed=seed)
        for p in range(P):
            qa.reset_epoch(p)
            nb3.reset_epoch()
            for t in range(T):
                c_qa  = qa.act(t, p)
                nb3.act(t, p)
                l_qa  = world.compute_losses(c_qa, rng)
                c_nb3 = nb3.act(t, p)
                l_nb3 = world.compute_losses(c_nb3, rng)
                qa.update(c_qa, l_qa)
                nb3.update(c_nb3, l_nb3)
        qa_final = qa_losses[-10:].mean() if len(qa_losses) >= 10 else np.mean(qa_losses)
        nb3_final = nb3r_losses[-10:].mean() if len(nb3r_losses) >= 10 else np.mean(nb3r_losses)
        # Store per-seed final losses
        qa_losses.append(qa_losses[-10:].mean() if len(qa_losses) >= 10 else np.mean(qa_losses))
        nb3r_losses.append(nb3r_losses[-10:].mean() if len(nb3r_losses) >= 10 else np.mean(nb3r_losses))
    # Recompute properly: collect all step losses then avg last 10 per epoch
    qa_all = np.zeros((P, T)); nb3_all = np.zeros((P, T))
    for si in range(n_seeds):
        seed = 42 + si
        rng = np.random.default_rng(seed)
        world = AbstractWorld(N=N, K=4, m=20, Z=6, seed=seed)
        for p in range(P):
            qa.reset_epoch(p); nb3.reset_epoch()
            for t in range(T):
                qa_all[p, t] += world.compute_losses(qa.act(t, p), rng).mean()
                nb3_all[p, t] += world.compute_losses(nb3.act(t, p), rng).mean()
    qa_all /= n_seeds; nb3_all /= n_seeds
    qa_final = qa_all[:, -10:].mean(); nb3_final = nb3_all[:, -10:].mean()
    # T-test: per-seed comparison
    wins = sum(1 for s in range(n_seeds) if qa_final_sims[s] < nb3_final_sims[s])

No that's getting too complex. Let me rewrite this more cleanly.
</think>


<minimax:tool_call>
<invoke name="write">
<parameter name="content">#!/usr/bin/env python3
"""Cat 5: Regret Convergence — find crossover N where QA-MAB beats NB3R."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "simulations/results/validation_cat5_physical/"
os.makedirs(OUT, exist_ok=True)

from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.agents_physical import NB3RAgent

N_vals = [5, 8, 10, 12, 15, 20, 30]
P, T, n_seeds = 5, 100, 20

table = []
crossover_N = None
for N in N_vals:
    print(f"  N={N}")
    qa_finals = []; nb3_finals = []
    for si in range(n_seeds):
        seed = 42 + si
        rng = np.random.default_rng(seed)
        world = AbstractWorld(N=N, K=4, m=20, Z=6, seed=seed)
        qa  = QAMABPhysical(world, seed=seed)
        nb3 = NB3RAgent(world, seed=seed)
        for p in range(P):
            qa.reset_epoch(p); nb3.reset_epoch()
            for t in range(T):
                c_qa = qa.act(t, p)
                l_qa = world.compute_losses(c_qa, rng)
                qa.update(c_qa, l_qa)
                c_nb = nb3.act(t, p)
                l_nb = world.compute_losses(c_nb, rng)
                nb3.update(c_nb, l_nb)
        # mean of last 10 steps
        qa_finals.append(float(qa._loss_buf[-10:].mean()) if hasattr(qa, '_loss_buf') and len(qa._loss_buf) >= 10 else 0.0)
        nb3_finals.append(float(nb3._loss_buf[-10:].mean()) if hasattr(nb3, '_loss_buf') and len(nb3._loss_buf) >= 10 else 0.0)

    # Actually agents don't track loss buffer. Simpler: just run and record step losses
    # Let me redo this properly by not relying on internal state
    # Instead just run and compare final epoch mean loss
    # This is wrong. Let me fix by actually collecting losses in the loop.

The agents don't expose loss history. Let me write this differently — just run the experiment and record losses inline.
</think>


<minimax:tool_call>
<invoke name="write">
<parameter name="content">#!/usr/bin/env python3
"""Cat 5: Regret Convergence — QA-MAB vs NB3R crossover analysis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "simulations/results/validation_cat5_physical/"
os.makedirs(OUT, exist_ok=True)

from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.agents_physical import NB3RAgent

N_vals = [5, 8, 10, 12, 15, 20, 30]
P, T, n_seeds = 5, 100, 20

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
        qa   = QAMABPhysical(world, seed=seed)
        nb3  = NB3RAgent(world, seed=seed)

        qa_losses_ep  = []
        nb3_losses_ep = []

        for p in range(P):
            qa.reset_epoch(p); nb3.reset_epoch()
            ep_qa = []; ep_nb = []
            for t in range(T):
                c_qa = qa.act(t, p); l_qa = world.compute_losses(c_qa, rng)
                qa.update(c_qa, l_qa); ep_qa.append(l_qa.mean())

                c_nb = nb3.act(t, p); l_nb = world.compute_losses(c_nb, rng2)
                nb3.update(c_nb, l_nb); ep_nb.append(l_nb.mean())
            qa_losses_ep.append(np.mean(ep_qa[-10:]))
            nb3_losses_ep.append(np.mean(ep_nb[-10:]))

        qa_finals.append(np.mean(qa_losses_ep))
        nb3_finals.append(np.mean(nb3_losses_ep))

    qa_arr = np.array(qa_finals); nb3_arr = np.array(nb3_finals)
    win_rate = float((qa_arr < nb3_arr).mean())
    t_stat, p_val = stats.ttest_rel(qa_arr, nb3_arr)
    qa_mean = float(qa_arr.mean()); nb3_mean = float(nb3_arr.mean())
    rows.append({"N": N, "qa_mean": qa_mean, "nb3r_mean": nb3_mean,
                "win_rate": win_rate, "p_value": float(p_val)})

    if crossover_N is None and win_rate >= 0.80 and p_val < 0.05:
        crossover_N = N

    print(f"    QA={qa_mean:.3f}  NB3R={nb3_mean:.3f}  win={win_rate:.1%}  p={p_val:.3f}")

pass_cat5 = crossover_N is not None
result = {"pass": pass_cat5, "reason": f"Crossover at N={crossover_N}: {'PASS' if pass_cat5 else 'FAIL (no crossover)'}",
          "crossover_N": crossover_N, "rows": rows}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

import csv
with open(OUT + "result.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["N","qa_mean","nb3r_mean","win_rate","p_value"])
    w.writeheader(); w.writerows(rows)

# Plots
N_plot = [r["N"] for r in rows]
wr = [r["win_rate"] for r in rows]
plt.figure(); plt.plot(N_plot, wr, marker='o', color='tab:blue')
plt.axhline(0.80, color='tab:red', linestyle='--', label='80% threshold')
plt.axhline(0.50, color='gray', linestyle=':', label='50%')
plt.xlabel("N flows"); plt.ylabel("QA-MAB win rate"); plt.title("Crossover Analysis")
plt.legend(); plt.savefig(OUT + "win_rate.png"); plt.close()

plt.figure()
x = np.arange(len(N_plot))
qa_means = [r["qa_mean"] for r in rows]; nb3_means = [r["nb3r_mean"] for r in rows]
plt.plot(x, qa_means, marker='o', label='QA-MAB')
plt.plot(x, nb3_means, marker='s', label='NB3R')
plt.xticks(x, N_plot); plt.xlabel("N flows"); plt.ylabel("Mean Final Loss")
plt.title("QA-MAB vs NB3R: Mean Loss"); plt.legend()
plt.savefig(OUT + "mean_loss.png"); plt.close()

print(f"Cat 5: crossover_N={crossover_N} — {'PASS' if pass_cat5 else 'FAIL'}")