#!/usr/bin/env python3
"""Cat 1: Parameter Sweep for Physical Model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os, time
from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical

OUT = "simulations/results/validation_cat1_physical/"
os.makedirs(OUT, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def run_qamab(N, K, m, Z, ucb_c, epoch_decay, sa_sweeps, sa_n_reads, seed):
    world = AbstractWorld(N=N, K=K, m=m, Z=Z, seed=seed)
    agent = QAMABPhysical(world, ucb_c=ucb_c, epoch_decay=epoch_decay,
                          sa_sweeps=sa_sweeps, sa_n_reads=sa_n_reads, seed=seed)
    rng = np.random.default_rng(seed)
    P, T = 10, 50
    losses_log = np.zeros((P, T, N))
    theta_err_log = np.zeros(P)
    for p in range(P):
        agent.reset_epoch(p)
        for t in range(T):
            chosen = agent.act(t, p)
            losses = world.compute_losses(chosen, rng)
            agent.update(chosen, losses)
            losses_log[p, t] = losses
        theta_err_log[p] = float(np.linalg.norm(agent.theta_hat - world.theta_star))
    final_loss = losses_log[:, -10:, :].mean()
    return final_loss, theta_err_log[-1]

# ── Sweep 1: UCB ─────────────────────────────────────────────────────────────
ucb_vals = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
ucb_results = {}
for ucb in ucb_vals:
    losses, errors = [], []
    t0 = time.time()
    for s in range(10):
        loss, err = run_qamab(N=10, K=4, m=20, Z=6, ucb_c=ucb, epoch_decay=0.7,
                               sa_sweeps=200, sa_n_reads=20, seed=42+s)
        losses.append(loss)
        errors.append(err)
    elapsed = (time.time() - t0) / 10
    ucb_results[ucb] = {"mean_loss": float(np.mean(losses)), "std_loss": float(np.std(losses)),
                        "mean_theta_err": float(np.mean(errors)), "elapsed_per_run": elapsed}
    print(f"  ucb_c={ucb}: loss={np.mean(losses):.4f}  theta_err={np.mean(errors):.4f}")

# ── Sweep 2: Decay ─────────────────────────────────────────────────────────────
decay_vals = [0.5, 0.7, 0.9, 1.0]
decay_results = {}
for dec in decay_vals:
    losses, errors = [], []
    for s in range(10):
        loss, err = run_qamab(N=10, K=4, m=20, Z=6, ucb_c=3.0, epoch_decay=dec,
                               sa_sweeps=200, sa_n_reads=20, seed=42+s)
        losses.append(loss)
        errors.append(err)
    decay_results[dec] = {"mean_loss": float(np.mean(losses)), "std_loss": float(np.std(losses)),
                          "mean_theta_err": float(np.mean(errors))}
    print(f"  decay={dec}: loss={np.mean(losses):.4f}  theta_err={np.mean(errors):.4f}")

# ── Sweep 3: SA effort ──────────────────────────────────────────────────────
sa_vals = [50, 100, 200, 500]
sa_results = {}
for sweeps in sa_vals:
    losses, elapsed = [], []
    for s in range(10):
        t0 = time.time()
        loss, _ = run_qamab(N=10, K=4, m=20, Z=6, ucb_c=3.0, epoch_decay=0.7,
                             sa_sweeps=sweeps, sa_n_reads=10, seed=42+s)
        losses.append(loss)
        elapsed.append(time.time() - t0)
    sa_results[sweeps] = {"mean_loss": float(np.mean(losses)), "std_loss": float(np.std(losses)),
                          "mean_elapsed": float(np.mean(elapsed))}
    print(f"  sa_sweeps={sweeps}: loss={np.mean(losses):.4f}  time={np.mean(elapsed):.2f}s")

# ── PASS check ───────────────────────────────────────────────────────────────
best_ucb_loss = min(ucb_results[u]["mean_loss"] for u in ucb_vals)
baseline_loss = ucb_results[0.0]["mean_loss"]
ucb_improvement = (baseline_loss - best_ucb_loss) / baseline_loss
best_decay = min(decay_results, key=lambda d: decay_results[d]["mean_loss"])
best_sa = min(sa_results, key=lambda s: sa_results[s]["mean_loss"])

pass_cat1 = ucb_improvement >= 0.10

result = {
    "pass": pass_cat1,
    "reason": f"UCB improvement={ucb_improvement:.1%} ({'PASS' if pass_cat1 else 'FAIL'}, threshold 10%)",
    "ucb_improvement": ucb_improvement,
    "ucb_results": ucb_results,
    "decay_results": decay_results,
    "sa_results": sa_results,
    "recommended": {"ucb_c": 3.0, "epoch_decay": best_decay, "sa_sweeps": best_sa},
}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

# CSVs
import csv
with open(OUT + "sweep_ucb.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ucb_c","mean_loss","std_loss","mean_theta_err","elapsed_per_run"])
    w.writeheader()
    for u, v in ucb_results.items():
        w.writerow({"ucb_c": u, **v})
with open(OUT + "sweep_decay.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["decay","mean_loss","std_loss","mean_theta_err"])
    w.writeheader()
    for d, v in decay_results.items():
        w.writerow({"decay": d, **v})
with open(OUT + "sweep_sa.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sa_sweeps","mean_loss","std_loss","mean_elapsed"])
    w.writeheader()
    for s, v in sa_results.items():
        w.writerow({"sa_sweeps": s, **v})

# PNGs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x_ucb = [u for u in ucb_vals]
y_ucb = [ucb_results[u]["mean_loss"] for u in ucb_vals]
e_ucb = [ucb_results[u]["std_loss"] for u in ucb_vals]
plt.figure(); plt.errorbar(x_ucb, y_ucb, yerr=e_ucb, marker='o', capsize=4)
plt.xlabel('ucb_c'); plt.ylabel('Mean Final Loss'); plt.title('UCB Sweep')
plt.savefig(OUT + "sweep_ucb.png"); plt.close()

x_dec = [d for d in decay_vals]
y_dec = [decay_results[d]["mean_loss"] for d in decay_vals]
e_dec = [decay_results[d]["std_loss"] for d in decay_vals]
plt.figure(); plt.errorbar(x_dec, y_dec, yerr=e_dec, marker='o', capsize=4)
plt.xlabel('epoch_decay'); plt.ylabel('Mean Final Loss'); plt.title('Decay Sweep')
plt.savefig(OUT + "sweep_decay.png"); plt.close()

x_sa = [s for s in sa_vals]
y_sa = [sa_results[s]["mean_loss"] for s in sa_vals]
plt.figure(); plt.plot(x_sa, y_sa, marker='o')
plt.xlabel('sa_sweeps'); plt.ylabel('Mean Final Loss'); plt.title('SA Sweeps Sweep')
plt.savefig(OUT + "sweep_sa.png"); plt.close()

print(f"\nCat 1 PASS={pass_cat1} — UCB improvement={ucb_improvement:.1%}")
print(f"Recommended: ucb_c=3.0, epoch_decay={best_decay}, sa_sweeps={best_sa}")