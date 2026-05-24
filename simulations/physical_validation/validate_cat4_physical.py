#!/usr/bin/env python3
"""Cat 4: Learning Dynamics — θ̂/φ̂ convergence over epochs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulations.physical_validation.physical_env import AbstractWorld
from simulations.physical_validation.qa_mab_physical import QAMABPhysical
from simulations.physical_validation.agents_physical import OracleAgent

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
        world  = AbstractWorld(N=3, K=4, m=15, Z=6, seed=seed)
        qa     = QAMABPhysical(world, epoch_decay=decay, seed=seed)
        oracle = OracleAgent(world, seed=seed)
        rng    = np.random.default_rng(seed)
        rng2   = np.random.default_rng(seed + 1000)
        for p in range(P):
            # Shared deterministic topology for this epoch — both agents see
            # the same world state.  Using seed * 100_000 + p matches the
            # convention in runner_physical.py.
            epoch_seed = seed * 100_000 + p
            qa.reset_epoch(p,     world_rng=np.random.default_rng(epoch_seed))
            oracle.reset_epoch(p, world_rng=np.random.default_rng(epoch_seed))
            ep_qa = []; ep_or = []
            for t in range(T):
                c_qa = qa.act(t, p);     l_qa = world.compute_losses(c_qa, rng)
                qa.update(c_qa, l_qa);   ep_qa.append(l_qa.mean())
                c_or = oracle.act(t, p); l_or = world.compute_losses(c_or, rng2)
                ep_or.append(l_or.mean())
            theta_errs[si, p] = float(np.linalg.norm(qa.theta_hat - world.theta_star)) / world.m
            phi_errs[si, p]   = float(np.linalg.norm(qa.phi_hat   - world.phi_star))   / world.Z
            gaps[si, p]       = np.mean(ep_qa) - np.mean(ep_or)
    return theta_errs, phi_errs, gaps


all_results = {}
for dec in decays:
    print(f"  decay={dec}", flush=True)
    t0 = time.time()
    te, pe, g = run_decay(dec, n_seeds)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    all_results[dec] = {
        "theta_err_init":  float(te[:, 0].mean()),
        "theta_err_final": float(te[:, -1].mean()),
        "phi_err_init":    float(pe[:, 0].mean()),
        "phi_err_final":   float(pe[:, -1].mean()),
        "gap_init":        float(g[:, 0].mean()),
        "gap_final":       float(g[:, -1].mean()),
        "theta_errs":      [[float(v) for v in row] for row in te.tolist()],
        "phi_errs":        [[float(v) for v in row] for row in pe.tolist()],
        "gaps":            [[float(v) for v in row] for row in g.tolist()],
    }

pass_cat4 = any(
    all_results[dec]["gap_final"] < all_results[dec]["gap_init"] or
    all_results[dec]["theta_err_final"] < all_results[dec]["theta_err_init"]
    for dec in decays
)

result = {
    "pass": pass_cat4,
    "reason": f"Learning convergence: {'PASS' if pass_cat4 else 'FAIL'}",
    "decay_results": all_results,
}
with open(OUT + "result.json", "w") as f:
    json.dump(result, f, indent=2)

epochs = np.arange(P)
for dec in decays:
    r = all_results[dec]
    for name, arr in [
        ("theta_err", np.array(r["theta_errs"])),
        ("phi_err",   np.array(r["phi_errs"])),
        ("gap",       np.array(r["gaps"])),
    ]:
        plt.figure()
        plt.plot(epochs, arr.mean(axis=0))
        plt.xlabel("Epoch"); plt.ylabel(name)
        plt.title(f"Learning Dynamics — decay={dec}")
        plt.savefig(f"{OUT}{name}_decay{dec}.png"); plt.close()

print(f"Cat 4: {'PASS' if pass_cat4 else 'FAIL'}")
