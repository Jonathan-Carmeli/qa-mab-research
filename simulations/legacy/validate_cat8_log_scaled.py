#!/usr/bin/env python3
"""Cat 8: QA-MAB vs Log-Scaled QUBO — exploration schedule comparison.

Current method:  Q_scaled = Q / gamma  where gamma = γ₀ / ((p+1)^a · (t+1)^b)
                 → Q_scaled grows polynomially: ~t^b
Log method:       Q_scaled = Q * (1 + log(t+1))
                 → Q_scaled grows logarithmically: ~log(t)

Both make the QUBO sharper over time (more selective, less exploration),
but via different schedules. The log method is gentler — starts less
selective and gradually ramps up.

Also tests: both agents use the SAME random world and θ*/φ*, so the
comparison is head-to-head.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "simulations/results/validation_cat8_physical/"
os.makedirs(OUT, exist_ok=True)

from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.sa_solver_physical import sa_solve, decode_solution


class QAMABLogScaled(QAMABPhysical):
    """QA-MAB variant: QUBO scaled by log(t+1) instead of temperature gamma."""

    name = "QA-MAB-LogScaled"

    def act(self, t: int, p: int) -> np.ndarray:
        Q = self.build_qubo()
        scale = 1.0 + np.log(t + 1)  # grows logarithmically
        Q_scaled = Q * scale

        best_x, _ = sa_solve(
            Q_scaled,
            self.rng,
            n_reads=self.sa_n_reads,
            n_sweeps=self.sa_sweeps,
            T_init=self.sa_T_init,
            T_final=self.sa_T_final,
        )
        return decode_solution(best_x, self.world.N, self.world.K)


N_vals = [5, 8, 10, 12, 15, 20]
P, T, n_seeds = 5, 100, 10

print("Cat 8: gamma-scaled vs log-scaled QUBO", flush=True)
rows = []
qa_wins = 0; log_wins = 0

for N in N_vals:
    print(f"  N={N}", flush=True)
    qa_finals = []; log_finals = []
    for si in range(n_seeds):
        seed = 42 + si
        rng  = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 1000)

        world = AbstractWorld(N=N, K=4, m=20, Z=6, seed=seed)
        qa    = QAMABPhysical(world, seed=seed)
        log_qa = QAMABLogScaled(world, seed=seed+2000)

        qa_losses = []; log_losses = []

        for p in range(P):
            qa.reset_epoch(p); log_qa.reset_epoch(p)
            for t in range(T):
                c_qa = qa.act(t, p); l_qa = world.compute_losses(c_qa, rng)
                qa.update(c_qa, l_qa); qa_losses.append(l_qa.mean())
                c_log = log_qa.act(t, p); l_log = world.compute_losses(c_log, rng2)
                log_qa.update(c_log, l_log); log_losses.append(l_log.mean())

        qa_finals.append(np.mean(qa_losses[-10:]))
        log_finals.append(np.mean(log_losses[-10:]))

    qa_arr = np.array(qa_finals); log_arr = np.array(log_finals)
    qa_mean = float(qa_arr.mean()); log_mean = float(log_arr.mean())
    wr = float((qa_arr < log_arr).mean())
    rows.append({"N": N, "qa_mean": qa_mean, "log_mean": log_mean,
                 "qa_better_rate": wr, "diff": qa_mean - log_mean})

    if wr > 0.5: qa_wins += 1
    elif wr < 0.5: log_wins += 1

# Per-step regret tracking (3 seeds, N=10)
print("  Per-step regret tracking...", flush=True)
ep_qa_all = []; ep_log_all = []
for si in range(3):
    seed = 42 + si
    rng = np.random.default_rng(seed); rng2 = np.random.default_rng(seed+1000)
    world = AbstractWorld(N=10, K=4, m=20, Z=6, seed=seed)
    qa = QAMABPhysical(world, seed=seed); log_qa = QAMABLogScaled(world, seed=seed+2000)
    ep_qa = []; ep_log = []
    for p in range(5):
        qa.reset_epoch(p); log_qa.reset_epoch(p)
        for t in range(T):
            c = qa.act(t,p); l = world.compute_losses(c,rng); qa.update(c,l); ep_qa.append(l.mean())
            c = log_qa.act(t,p); l = world.compute_losses(c,rng2); log_qa.update(c,l); ep_log.append(l.mean())
    ep_qa_all.append(ep_qa); ep_log_all.append(ep_log)

avg_qa = np.mean(ep_qa_all, axis=0); avg_log = np.mean(ep_log_all, axis=0)
windows = 10
rw_qa  = [float(np.mean(avg_qa[i*windows:(i+1)*windows]))   for i in range(len(avg_qa)//windows)]
rw_log = [float(np.mean(avg_log[i*windows:(i+1)*windows]))   for i in range(len(avg_log)//windows)]

pass_cat8 = True  # descriptive test, always pass
result = {
    "pass": pass_cat8,
    "reason": f"QA-MAB (gamma) wins {qa_wins}/{len(rows)} N values, Log-wins {log_wins}/{len(rows)}",
    "rows": rows,
    "per_window_qa":  rw_qa,
    "per_window_log": rw_log,
}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

import csv
with open(OUT + "result.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["N","qa_mean","log_mean","qa_better_rate","diff"])
    w.writeheader(); w.writerows(rows)

# Plot: loss over time
x = np.arange(len(rw_qa)) * windows
plt.figure(figsize=(8, 4))
plt.plot(x, rw_qa,  label='QA-MAB (gamma-scaled)', color='tab:blue', lw=1.5)
plt.plot(x, rw_log, label='QA-MAB (log-scaled)',   color='tab:orange', lw=1.5)
plt.xlabel("Step"); plt.ylabel("Mean Loss (last 10 steps)")
plt.title("Cat 8: Convergence — gamma-scaled vs log-scaled QUBO")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(OUT + "convergence.png", dpi=150); plt.close()

# Bar: per-N comparison
x_pos = np.arange(len(rows))
plt.figure(figsize=(8, 4))
w_qa  = [r["qa_mean"]  for r in rows]
w_log = [r["log_mean"] for r in rows]
width = 0.35
plt.bar(x_pos - width/2, w_qa,  width, label='QA-MAB (gamma)', color='tab:blue', alpha=0.8)
plt.bar(x_pos + width/2, w_log, width, label='QA-MAB (log)',    color='tab:orange', alpha=0.8)
plt.xticks(x_pos, [r["N"] for r in rows])
plt.xlabel("N flows"); plt.ylabel("Mean Final Loss")
plt.title("Cat 8: Final Loss — gamma-scaled vs log-scaled")
plt.legend(); plt.grid(True, alpha=0.3, axis='y')
plt.savefig(OUT + "comparison.png", dpi=150); plt.close()

print(f"\nCat 8: gamma wins {qa_wins}/{len(rows)} N, log wins {log_wins}/{len(rows)} N")
print("Detailed per-N:")
for r in rows:
    print(f"  N={r['N']}: gamma={r['qa_mean']:.3f}  log={r['log_mean']:.3f}  diff={r['diff']:.3f}")