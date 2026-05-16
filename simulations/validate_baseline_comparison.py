#!/usr/bin/env python3
"""Baseline Comparison: Physical model vs legacy NetworkEnvironment model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "simulations/results/validation_baseline/"
os.makedirs(OUT, exist_ok=True)

from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.agents_physical import NB3RAgent, RandomAgent, OracleAgent

# ── New model ─────────────────────────────────────────────────────────────────
N, K, m, Z = 10, 4, 30, 9
P, T, n_seeds = 10, 100, 20

new_results = {name: [] for name in ["QA-MAB", "NB3R", "Random", "Oracle"]}

for si in range(n_seeds):
    seed = 42 + si
    rng = np.random.default_rng(seed)
    world = AbstractWorld(N=N, K=K, m=m, Z=Z, seed=seed)

    agents = {
        "QA-MAB": QAMABPhysical(world, seed=seed),
        "NB3R":   NB3RAgent(world, seed=seed),
        "Random": RandomAgent(world, seed=seed),
        "Oracle": OracleAgent(world, seed=seed),
    }

    for name, agent in agents.items():
        ep_losses = []
        for p in range(P):
            if hasattr(agent, 'reset_epoch') and name != "NB3R" and name != "Random":
                agent.reset_epoch(p)
            elif hasattr(agent, 'reset_epoch'):
                agent.reset_epoch()
            ep = []
            for t in range(T):
                c = agent.act(t, p)
                l = world.compute_losses(c, rng)
                agent.update(c, l)
                ep.append(l.mean())
            ep_losses.append(np.mean(ep))
        final = np.mean(ep_losses[-10:])
        cumsum = float(np.sum(ep_losses))
        new_results[name].append({"final": final, "cumsum": cumsum,
                                   "theta_err": float(np.linalg.norm(agent.theta_hat - world.theta_star)) if name == "QA-MAB" else 0.0})

# ── Old model ─────────────────────────────────────────────────────────────────
from simulations.qa_mab import QAMAB
from simulations.nb3r import NB3R
from simulations.simulation_core import NetworkEnvironment

old_results = {name: [] for name in ["QA-MAB-legacy", "NB3R-legacy", "Random-legacy"]}

for si in range(n_seeds):
    seed = 42 + si
    rng = np.random.default_rng(seed)
    env = NetworkEnvironment(N=N, m=K, seed=seed)
    agents_old = {
        "QA-MAB-legacy": QAMAB(env, seed=seed),
        "NB3R-legacy":   NB3R(env, seed=seed),
        "Random-legacy": None,
    }
    for name, agent in agents_old.items():
        if agent is None:
            continue
        ep_losses = []
        for p in range(P):
            ep = []
            for t in range(T):
                if hasattr(agent, 'step'):
                    agent.step()
                    x = {i: agent._prev_x[i] for i in range(N)} if hasattr(agent, '_prev_x') else {}
                    l = list(env.compute_throughput(x).values())
                else:
                    routes = {i: rng.integers(0, K) for i in range(N)}
                    l = list(env.compute_throughput(routes).values())
                ep.append(np.mean(l) if l else 0.0)
            ep_losses.append(np.mean(ep))
        final = np.mean(ep_losses[-10:])
        old_results[name].append({"final": final, "cumsum": float(np.sum(ep_losses))})

# ── Summarize ─────────────────────────────────────────────────────────────────
rows = []
for name, res in {**new_results, **old_results}.items():
    finals = [r["final"] for r in res]
    rows.append({"agent": name, "mean_final": float(np.mean(finals)), "std_final": float(np.std(finals))})

import csv
with open(OUT + "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["agent","mean_final","std_final"])
    w.writeheader(); w.writerows(rows)

# Sanity check: QA-MAB-Physical ≤ NB3R-Physical ≤ Random-Physical
qa_new = new_results["QA-MAB"]
nb3_new = new_results["NB3R"]
rand_new = new_results["Random"]
pass_baseline = (
    np.mean([r["final"] for r in qa_new]) <= np.mean([r["final"] for r in nb3_new]) and
    np.mean([r["final"] for r in nb3_new]) <= np.mean([r["final"] for r in rand_new])
)

result = {"pass": pass_baseline, "reason": f"QA-MAB-Physical ≤ NB3R-Physical ≤ Random-Physical: {'PASS' if pass_baseline else 'FAIL'}",
          "new_model": {n: [float(np.mean([r["final"] for r in new_results[n]]))] for n in new_results},
          "old_model": {n: [float(np.mean([r["final"] for r in old_results[n]]))] for n in old_results}}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Baseline: {'PASS' if pass_baseline else 'FAIL'}")
print("Mean final losses:")
for r in rows:
    print(f"  {r['agent']}: {r['mean_final']:.4f} ± {r['std_final']:.4f}")