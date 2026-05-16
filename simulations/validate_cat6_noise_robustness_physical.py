#!/usr/bin/env python3
"""Cat 6: Noise Robustness — QA-MAB vs NB3R under increasing noise."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "simulations/results/validation_cat6_physical/"
os.makedirs(OUT, exist_ok=True)

from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.agents_physical import NB3RAgent, RandomAgent

N_vals = [5, 10, 15, 20]
sigma_vals = [0.0, 0.05, 0.1, 0.5]
P, T, n_seeds = 5, 100, 15

table = {N: {} for N in N_vals}
for N in N_vals:
    for sigma in sigma_vals:
        print(f"  N={N}, sigma={sigma}", flush=True)
        qa_finals = []; nb3_finals = []
        for si in range(n_seeds):
            seed = 42 + si
            rng = np.random.default_rng(seed)
            rng2 = np.random.default_rng(seed + 1000)
            rng3 = np.random.default_rng(seed + 2000)

            world = AbstractWorld(N=N, K=4, m=20, Z=6, sigma_noise=sigma, seed=seed)
            qa  = QAMABPhysical(world, seed=seed)
            nb3 = NB3RAgent(world, seed=seed)

            for p in range(P):
                qa.reset_epoch(p); nb3.reset_epoch()
                for t in range(T):
                    c_qa = qa.act(t, p); l_qa = world.compute_losses(c_qa, rng)
                    qa.update(c_qa, l_qa)
                    c_nb = nb3.act(t, p); l_nb = world.compute_losses(c_nb, rng2)
                    nb3.update(c_nb, l_nb)

            qa_finals.append(float(qa._loss_acc[-10:].mean()) if hasattr(qa,'_loss_acc') and len(qa._loss_acc)>=10 else 0.0)
            nb3_finals.append(float(nb3._loss_acc[-10:].mean()) if hasattr(nb3,'_loss_acc') and len(nb3._loss_acc)>=10 else 0.0)

Agents don't track loss buffers. Let me just collect losses in arrays in the main loop.

I'm going to simplify this and just collect the losses directly during the simulation loop instead of trying to access agent buffers that don't exist.
</think>


<minimax:tool_call>
<invoke name="write">
<parameter name="content">#!/usr/bin/env python3
"""Cat 6: Noise Robustness — QA-MAB vs NB3R at different noise levels."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "simulations/results/validation_cat6_physical/"
os.makedirs(OUT, exist_ok=True)

from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.agents_physical import NB3RAgent, RandomAgent

N_vals  = [5, 10, 15, 20]
sigmas = [0.0, 0.05, 0.1, 0.5]
P, T, n_seeds = 5, 100, 15

# table[n][sigma] = win_rate
table = {N: {s: None for s in sigmas} for N in N_vals}
for N in N_vals:
    for sigma in sigmas:
        print(f"  N={N}, sigma={sigma}", flush=True)
        qa_finals = []; nb3_finals = []
        for si in range(n_seeds):
            seed = 42 + si
            rng = np.random.default_rng(seed)
            rng2 = np.random.default_rng(seed + 1000)

            world = AbstractWorld(N=N, K=4, m=20, Z=6, sigma_noise=sigma, seed=seed)
            qa   = QAMABPhysical(world, seed=seed)
            nb3  = NB3RAgent(world, seed=seed)

            qa_losses_ep = []; nb3_losses_ep = []

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

        wr = float((np.array(qa_finals) < np.array(nb3_finals)).mean())
        table[N][sigma] = wr

# Build result table string
label = lambda wr: "QA" if wr > 0.6 else ("NB3R" if wr < 0.4 else "TIE")
header = ["N=5","N=10","N=15","N=20"]
print("\n         " + "  ".join(f"{h:>10}" for h in header))
print("         " + "  ".join(f"{'sigma='+str(s):>10}" for s in sigmas))
for sigma in sigmas:
    row = [label(table[N][sigma]) for N in N_vals]
    print(f"sigma={sigma:4.2f}  " + "  ".join(f"{v:>10}" for v in row))

result_table_csv = [[""] + header]
for sigma in sigmas:
    row = [f"sigma={sigma}"] + [label(table[N][sigma]) for N in N_vals]
    result_table_csv.append(row)

pass_cat6 = table[20][0.0] > 0.6 and table[20][0.05] > 0.6

result = {"pass": pass_cat6,
          "reason": f"QA-MAB win rate at N=20: sigma=0={table[20][0.0]:.1%}, sigma=0.05={table[20][0.05]:.1%} — {'PASS' if pass_cat6 else 'FAIL'}",
          "table": {str(N): {str(s): table[N][s] for s in sigmas} for N in N_vals}}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

import csv
with open(OUT + "result_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerows(result_table_csv)

# Heatmap
fig, ax = plt.subplots(figsize=(6, 4))
data = np.array([[table[N][s] for N in N_vals] for s in sigmas])
im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(N_vals))); ax.set_xticklabels(N_vals)
ax.set_yticks(range(len(sigmas))); ax.set_yticklabels([f"{s:.2f}" for s in sigmas])
ax.set_xlabel("N flows"); ax.set_ylabel("sigma"); ax.set_title("QA-MAB win rate")
for i in range(len(sigmas)):
    for j in range(len(N_vals)):
        ax.text(j, i, f"{data[i,j]:.2f}", ha='center', va='center',
                color='white' if data[i,j]>0.5 else 'black', fontsize=9)
plt.colorbar(im, ax=ax, label="win rate")
plt.tight_layout()
plt.savefig(OUT + "heatmap.png"); plt.close()

print(f"\nCat 6: {'PASS' if pass_cat6 else 'FAIL'}")