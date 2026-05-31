#!/usr/bin/env python3
"""Cat 9 v3: QA-MAB (learns) vs OracleSA (knows theta*/phi*, SA solver).

Same question: does partial regret → 0 as epochs increase?
Same config as Cat 4 (which already showed this works), but N=10.

This is essentially Cat 4 with N=10 to confirm at larger scale.
"""
import sys, os
sys.path.insert(0, '/Users/jon_claw/qa-mab-research')
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')

from .physical_env import AbstractWorld
from .qa_mab_physical import QAMABPhysical
from .sa_solver_physical import sa_solve, decode_solution

import numpy as np
import json, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "simulations/results/validation_cat9_physical/"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Configuration: N=10, P=30, T=200, SA-based oracle
# ============================================================
N, P, T = 10, 30, 200
n_seeds = 10
sigmas = [0.0, 0.1]

# ============================================================
# OracleSA: knows theta*/phi*, uses SA (not brute-force)
# ============================================================
class OracleSA(QAMABPhysical):
    """"Oracle that knows true params but uses SA to solve QUBO.
    
    Inherits from QAMABPhysical so it uses the SAME build_qubo() method.
    Only difference: theta_hat=theta* from the start (no learning).
    """
    name = "Oracle-SA"
    
    def __init__(self, world, seed=42):
        from .sa_solver_physical import sa_solve, decode_solution
        super().__init__(
            world,
            C_coll=world.C_coll,
            d0=world.d0,
            sigma_noise=0.0,
            alpha=0.0,
            ucb_c=0.0,
            epoch_decay=1.0,
            sa_sweeps=200,
            sa_n_reads=20,
            sa_T_init=2.0,
            sa_T_final=0.05,
            seed=seed,
        )
        self.theta_hat = world.theta_star.copy()
        self.phi_hat   = world.phi_star.copy()
        self._sa_solve = sa_solve
        self._decode   = decode_solution
    
    def update(self, *args):
        pass  # no learning


def run_single(sigma, seed):
    """Run one experiment: QA-MAB vs OracleSA."""
    rng_loss = np.random.default_rng(seed + 50000)
    
    world = AbstractWorld(N=N, K=4, m=20, Z=6,
                         sigma_noise=sigma, seed=seed)
    
    qa   = QAMABPhysical(world, seed=seed)
    ora  = OracleSA(world, seed=seed + 30000)
    
    regret  = np.zeros(P)
    qa_loss = np.zeros(P)
    or_loss = np.zeros(P)
    th_err  = np.zeros(P)
    ph_err  = np.zeros(P)
    
    for p in range(P):
        qa.reset_epoch(p)
        ora.reset_epoch(p)
        
        ep_qa = []
        ep_or = []
        
        for t in range(T):
            c_qa = qa.act(t, p)
            l_qa = world.compute_losses(c_qa, rng_loss)
            qa.update(c_qa, l_qa)
            ep_qa.append(l_qa.mean())
            
            c_or = ora.act(t, p)
            l_or = world.compute_losses(c_or, rng_loss)
            ora.update(c_or, l_or)
            ep_or.append(l_or.mean())
        
        regret[p]  = np.mean(ep_qa) - np.mean(ep_or)
        qa_loss[p] = np.mean(ep_qa)
        or_loss[p] = np.mean(ep_or)
        th_err[p]  = float(np.linalg.norm(qa.theta_hat - world.theta_star) / world.m)
        ph_err[p]  = float(np.linalg.norm(qa.phi_hat - world.phi_star) / world.Z)
    
    return {
        'regret':    regret.tolist(),
        'qa_loss':   qa_loss.tolist(),
        'or_loss':   or_loss.tolist(),
        'th_err':    th_err.tolist(),
        'ph_err':    ph_err.tolist(),
        'init_reg':  float(np.mean(regret[:5])),
        'final_reg': float(np.mean(regret[-5:])),
        'mid_reg':   float(np.mean(regret[P//2-2:P//2+2])),
    }


print("=" * 60)
print("Cat 9 v3: QA-MAB vs OracleSA (N=10, SA-based)")
print("=" * 60)
print(f"N={N}, P={P}, T={T}, seeds={n_seeds}, sigmas={sigmas}")
print()

results = {}
for sigma in sigmas:
    print(f"=== sigma={sigma} ===")
    results[sigma] = []
    t0 = time.time()
    for si in range(n_seeds):
        seed = 42 + si
        print(f"  seed={seed}...", end=" ", flush=True)
        r = run_single(sigma, seed)
        results[sigma].append(r)
        print(f"init={r['init_reg']:.3f} mid={r['mid_reg']:.3f} final={r['final_reg']:.3f}")
    print(f"  → {time.time()-t0:.0f}s")
    print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)

pass_count = 0
for sigma in sigmas:
    init_all  = [results[sigma][si]['init_reg']  for si in range(n_seeds)]
    final_all = [results[sigma][si]['final_reg'] for si in range(n_seeds)]
    mid_all   = [results[sigma][si]['mid_reg']   for si in range(n_seeds)]
    
    init_m  = np.mean(init_all)
    mid_m   = np.mean(mid_all)
    final_m = np.mean(final_all)
    red_if  = init_m - final_m
    
    status = "✅" if red_if > 0 else "❌"
    if red_if > 0:
        pass_count += 1
    
    print(f"σ={sigma}: init={init_m:.3f} mid={mid_m:.3f} final={final_m:.3f}  "
          f"Δ={red_if:+.3f} {status}")

pass_cat9 = pass_count >= len(sigmas) * 0.5
print(f"\n{'PASS ✅' if pass_cat9 else 'FAIL ❌'} — {pass_count}/{len(sigmas)} sigmas")
print()

# Save
save = {
    'category': 9,
    'test': 'partial_regret_convergence_vs_oraclesa',
    'pass': pass_cat9,
    'config': {'N': N, 'P': P, 'T': T, 'n_seeds': n_seeds, 'sigmas': sigmas},
    'per_sigma': {
        str(sigma): {
            'init_regret_mean':  float(np.mean([r['init_reg']  for r in results[sigma]])),
            'mid_regret_mean':   float(np.mean([r['mid_reg']   for r in results[sigma]])),
            'final_regret_mean': float(np.mean([r['final_reg'] for r in results[sigma]])),
            'regret_reduction':  float(np.mean([r['init_reg'] - r['final_reg'] for r in results[sigma]])),
        }
        for sigma in sigmas
    }
}
with open(OUT + "cat9_v3_result.json", 'w') as f:
    json.dump(save, f, indent=2)
print(f"Saved: {OUT}cat9_v3_result.json")

# Plot
epochs = np.arange(P)
plt.figure(figsize=(10, 5))
for sigma in sigmas:
    regs = np.array([results[sigma][si]['regret'] for si in range(n_seeds)])
    m = np.mean(regs, axis=0)
    s = np.std(regs, axis=0)
    plt.fill_between(epochs, m - s, m + s, alpha=0.2)
    plt.plot(epochs, m, label=f"σ={sigma}", lw=2)
plt.axhline(0, color='gray', ls='--', lw=1)
plt.xlabel("Epoch")
plt.ylabel("Partial Regret (QA-MAB − OracleSA)")
plt.title("Cat 9 v3: Regret Trajectory")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUT + "cat9_v3_regret_traj.png", dpi=150)
plt.close()
print(f"Saved: {OUT}cat9_v3_regret_traj.png")

plt.figure(figsize=(10, 5))
for sigma in sigmas:
    ths = np.array([results[sigma][si]['th_err'] for si in range(n_seeds)])
    m = np.mean(ths, axis=0)
    s = np.std(ths, axis=0)
    plt.fill_between(epochs, m - s, m + s, alpha=0.2)
    plt.plot(epochs, m, label=f"σ={sigma}", lw=2)
plt.xlabel("Epoch")
plt.ylabel("||θ̂ − θ*|| / m")
plt.title("Cat 9 v3: θ Error Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUT + "cat9_v3_theta_err.png", dpi=150)
plt.close()
print(f"Saved: {OUT}cat9_v3_theta_err.png")

print("DONE")