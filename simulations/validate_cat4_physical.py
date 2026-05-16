#!/usr/bin/env python3
"""Cat 4: Learning Dynamics — θ̂/φ̂ convergence over epochs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os
from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.agents_physical import OracleAgent, NB3RAgent, RandomAgent

OUT = "simulations/results/validation_cat4_physical/"
os.makedirs(OUT, exist_ok=True)

decays = [0.7, 1.0]
P, T = 25, 40
n_seeds = 15

def run_decay(decay, seeds):
    theta_errs = np.zeros((seeds, P))
    phi_errs   = np.zeros((seeds, P))
    gaps       = np.zeros((seeds, P))
    for si in range(seeds):
        seed = 42 + si
        world = AbstractWorld(N=3, K=4, m=15, Z=6, seed=seed)
        qa = QAMABPhysical(world, epoch_decay=decay, seed=seed)
        oracle = OracleAgent(world, seed=seed)
        rng = np.random.default_rng(seed)
        for p in range(P):
            qa.reset_epoch(p)
            oracle.reset_epoch(p)
            ep_losses_qa, ep_losses_oracle = [], []
            for t in range(T):
                c_qa = qa.act(t, p)
                l_qa = world.compute_losses(c_qa, rng)
                qa.update(c_qa, l_qa)
                ep_losses_qa.append(l_qa.mean())
                c_or = oracle.act(t, p)
                l_or = world.compute_losses(c_or, rng)
                ep_losses_oracle.append(l_or.mean())
            theta_errs[si, p] = float(np.linalg.norm(qa.theta_hat - world.theta_star)) / world.m
            phi_errs[si, p]   = float(np.linalg.norm(qa.phi_hat   - world.phi_star)) / world.Z
            gaps[si, p]       = np.mean(ep_losses_qa) - np.mean(ep_losses_oracle)
    return theta_errs, phi_errs, gaps

all_results = {}
for dec in decays:
    print(f"  decay={dec}")
    te, pe, g = run_decay(dec, n_seeds)
    all_results[dec] = {
        "theta_err_init": float(te[:, 0].mean()), "theta_err_final": float(te[:, -1].mean()),
        "phi_err_init":   float(pe[:, 0].mean()),   "phi_err_final":   float(pe[:, -1].mean()),
        "gap_init":       float(g[:, 0].mean()),    "gap_final":       float(g[:, -1].mean()),
        "theta_errs":     te.tolist(), "phi_errs": pe.tolist(), "gaps": g.tolist(),
    }

pass_cat4 = any(
    all_results[dec]["gap_final"] < all_results[dec]["gap_init"] or
    all_results[dec]["theta_err_final"] < all_results[dec]["theta_err_init"]
    for dec in decays
)

for dec, r in all_results.items():
    for key in ["theta_errs", "phi_errs", "gaps"]:
        r[key] = [[float(v) for v in row] for row in r[key]]

result = {"pass": pass_cat4, "reason": f"Learning convergence: {'PASS' if pass_cat4 else 'FAIL'}",
          "decay_results": all_results}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

epochs = np.arange(P)
for dec, r in all_results.items():
    te = np.array(r["theta_errs"])
    pe = np.array(r["phi_errs"])
    g  = np.array(r["gaps"])
    for name, arr, color in [("theta_err", te, "tab:blue"), ("phi_err", pe, "tab:orange"), ("gap", g, "tab:green")]:
        plt.figure()
        plt.plot(epochs, arr.mean(axis=0), color=color)
        plt.fill_between(epochs, arr.mean(axis=0)-arr.std(axis=0), arr.mean(axis=0)+arr.std(axis=0), alpha=0.3, color=color)
        plt.xlabel("Epoch"); plt.ylabel(name)
        plt.title(f"Learning Dynamics — decay={dec}")
        plt.savefig(f"{OUT}{name}_decay{dec}.png"); plt.close()

print(f"Cat 4: {'PASS' if pass_cat4 else 'FAIL'}")