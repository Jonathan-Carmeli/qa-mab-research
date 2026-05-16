#!/usr/bin/env python3
"""Cat 4: Learning Dynamics — θ̂/φ̂ convergence over epochs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json, os, time
from simulations.physical_env import AbstractWorld
from simulations.qa_mab_physical import QAMABPhysical
from simulations.agents_physical import OracleAgent

OUT = "simulations/results/validation_cat4_physical/"
os.makedirs(OUT, exist_ok=True)

decays = [0.7, 1.0]
P, T, n_seeds = 10, 30, 10

def run_decay(decay, seeds):
    theta_errs = np.zeros((seeds, P))
    phi_errs   = np.zeros((seeds, P))
    gaps       = np.zeros((seeds, P))
    for si in range(seeds):
        seed = 42 + si
        world = AbstractWorld(N=3, K=4, m=15, Z=6, seed=seed)
        qa    = QAMABPhysical(world, epoch_decay=decay, seed=seed)
        oracle = OracleAgent(world, seed=seed)
        rng = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 1000)
        for p in range(P):
            qa.reset_epoch(p); oracle.reset_epoch(p)
            ep_qa = []; ep_or = []
            for t in range(T):
                c_qa = qa.act(t, p); l_qa = world.compute_losses(c_qa, rng)
                qa.update(c_qa, l_qa); ep_qa.append(l_qa.mean())
                c_or = oracle.act(t, p); l_or = world.compute_losses(c_or, rng2)
                ep_or.append(l_or.mean())
            theta_errs[si, p] = float(np.linalg.norm(qa.theta_hat - world.theta_star)) / world.m
            phi_errs[si, p]   = float(np.linalg.norm(qa.phi_hat   - world.phi_star)) / world.Z
            gaps[si, p]       = np.mean(ep_qa) - np.mean(ep_or)
    return theta_errs, phi_errs, gaps

all_results = {}
for dec in decays:
    print(f"  decay={dec}", flush=True)
    t0 = time.time()
    te, pe, g = run_decay(dec, n_seeds)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
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

# Clean for JSON (convert nested lists)
for dec, r in all_results.items():
    for key in ["theta_errs", "phi_errs", "gaps"]:
        r[key] = [[float(v) for v in row] for row in r[key]]

result = {"pass": pass_cat4,
          "reason": f"Learning convergence: {'PASS' if pass_cat4 else 'FAIL'}",
          "decay_results": all_results}

with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
epochs = np.arange(P)

for dec, r in all_results.items():
    for name, arr in [("theta_err", te), ("phi_err", pe), ("gap", g)]:
        plt.figure()
        plt.plot(epochs, arr.mean(axis=0))
        plt.xlabel("Epoch"); plt.ylabel(name)
        plt.title(f"Learning Dynamics — decay={dec}")
        plt.savefig(f"{OUT}{name}_decay{dec}.png"); plt.close()

print(f"Cat 4: {'PASS' if pass_cat4 else 'FAIL'}")