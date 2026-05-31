#!/usr/bin/env python3
"""Cat 9 CORRECTED: QA-MAB (learns) vs BestFixedAgent (knows theta*/phi*, never learns).

Question: Does partial regret → 0 as epochs increase?

BestFixedAgent: knows true theta*/phi* (oracle), picks BEST FIXED action each epoch.
  This is NOT learning — it's the theoretical best without any exploration.
  QA-MAB must learn to match this performance.

If regret → 0: learning works.
If regret stays high: QUBO formulation doesn't capture the true problem.
"""
import sys, os
sys.path.insert(0, '/Users/jon_claw/qa-mab-research')
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations/physical_validation')

import numpy as np
import json, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product

from .physical_env import AbstractWorld
from .qa_mab_physical import QAMABPhysical
from .sa_solver_physical import sa_solve, decode_solution

OUT = "simulations/results/validation_cat9_physical/"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Test configuration: small but meaningful
# ============================================================
N, P, T = 5, 30, 200    # N=5, P=30, T=200 — enough for learning
n_seeds = 10
sigmas = [0.0, 0.1]

# ============================================================
# BestFixedAgent: knows true params, picks best action (no learning)
# ============================================================
class BestFixedAgent:
    """Best possible with full knowledge of theta* and phi*, NO learning.
    
    This agent KNOWS theta* and phi* and computes the optimal action
    from the start. It does NOT update theta_hat or phi_hat at all.
    
    For N<=10: brute-force over all K^N combos
    For N>10:  SA solver
    """
    name = "BestFixed"
    
    def __init__(self, world, seed=42):
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.theta_hat = world.theta_star.copy()
        self.phi_hat   = world.phi_star.copy()
    
    def act(self, t, p):
        """Return the optimal action given true theta* and phi*."""
        N, K = self.world.N, self.world.K
        ps = self.world.pathset
        
        if N <= 8:  # brute-force for N<=8
            best_loss = float('inf')
            best_chosen = None
            for combo in product(range(K), repeat=N):
                loss = self._true_loss(np.array(combo, dtype=int))
                if loss < best_loss:
                    best_loss = loss
                    best_chosen = np.array(combo, dtype=int)
            return best_chosen
        else:
            # SA solver
            Q = self.world.build_qubo_from_estimates(self.theta_hat, self.phi_hat)
            best_E = float('inf')
            best_x = None
            for _ in range(20):
                x, E = sa_solve(Q, self.rng, n_reads=1, n_sweeps=200,
                               T_init=2.0, T_final=0.05)
                if E < best_E:
                    best_E = E
                    best_x = x
            return decode_solution(best_x, N, K)
    
    def _true_loss(self, chosen):
        """Compute loss for a fixed action using true parameters."""
        N, K = self.world.N, self.world.K
        ps = self.world.pathset
        losses = np.zeros(N, dtype=float)
        su = np.array([ps.path_uav_membership[n, chosen[n]] for n in range(N)])
        sh = (su.astype(int) @ su.astype(int).T) > 0
        np.fill_diagonal(sh, False)
        cc = sh.sum(axis=1).astype(float)
        for n in range(N):
            k = chosen[n]
            um = ps.path_uav_membership[n, k]
            zm = ps.path_zone_membership[n, k]
            losses[n] = (float(self.theta_hat[um].sum()) +
                         float(self.phi_hat[zm].sum()) +
                         self.world.C_coll * cc[n])
            a = n * K + k
            for l in range(N):
                if l == n:
                    continue
                b = l * K + chosen[l]
                losses[n] += np.exp(-ps.pair_min_dist[a, b] / self.world.d0)
        return losses.sum()
    
    def update(self, chosen_paths, losses):
        """NO LEARNING — do nothing."""
        pass
    
    def reset_epoch(self, p):
        self.world.refresh_epoch(self.rng)


# ============================================================
# QA-MAB with learning
# ============================================================
def run_single(sigma, seed):
    """Run one experiment: QA-MAB vs BestFixedAgent."""
    rng_loss = np.random.default_rng(seed + 50000)
    
    world = AbstractWorld(N=N, K=4, m=20, Z=6,
                         sigma_noise=sigma, seed=seed)
    
    qa   = QAMABPhysical(world, seed=seed)
    bf   = BestFixedAgent(world, seed=seed + 20000)
    
    # Track per-epoch
    regret   = np.zeros(P)
    qa_loss  = np.zeros(P)
    bf_loss  = np.zeros(P)
    th_err   = np.zeros(P)
    ph_err   = np.zeros(P)
    
    for p in range(P):
        qa.reset_epoch(p)
        bf.reset_epoch(p)
        
        ep_qa = []
        ep_bf = []
        
        for t in range(T):
            c_qa = qa.act(t, p)
            l_qa = world.compute_losses(c_qa, rng_loss)
            qa.update(c_qa, l_qa)
            ep_qa.append(l_qa.mean())
            
            c_bf = bf.act(t, p)
            l_bf = world.compute_losses(c_bf, rng_loss)
            bf.update(c_bf, l_bf)
            ep_bf.append(l_bf.mean())
        
        regret[p]   = np.mean(ep_qa) - np.mean(ep_bf)
        qa_loss[p]  = np.mean(ep_qa)
        bf_loss[p]  = np.mean(ep_bf)
        th_err[p]   = float(np.linalg.norm(qa.theta_hat - world.theta_star) / world.m)
        ph_err[p]   = float(np.linalg.norm(qa.phi_hat - world.phi_star) / world.Z)
    
    return {
        'regret':    regret.tolist(),
        'qa_loss':   qa_loss.tolist(),
        'bf_loss':   bf_loss.tolist(),
        'th_err':    th_err.tolist(),
        'ph_err':    ph_err.tolist(),
        'init_reg':  float(np.mean(regret[:5])),
        'final_reg': float(np.mean(regret[-5:])),
        'mid_reg':   float(np.mean(regret[P//2-2:P//2+2])),
    }


# ============================================================
# Main
# ============================================================
print("=" * 60)
print("Cat 9 CORRECTED: QA-MAB vs BestFixedAgent")
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
    elapsed = time.time() - t0
    print(f"  → {elapsed:.0f}s")
    print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)

pass_count = 0
for sigma in sigmas:
    init_all = [results[sigma][si]['init_reg'] for si in range(n_seeds)]
    final_all = [results[sigma][si]['final_reg'] for si in range(n_seeds)]
    mid_all   = [results[sigma][si]['mid_reg'] for si in range(n_seeds)]
    
    init_m = np.mean(init_all)
    mid_m  = np.mean(mid_all)
    final_m = np.mean(final_all)
    red_init_final = init_m - final_m
    red_init_mid   = init_m - mid_m
    
    status = "✅" if red_init_final > 0 else "❌"
    if red_init_final > 0:
        pass_count += 1
    
    print(f"σ={sigma}:")
    print(f"  Init regret:  {init_m:.3f}")
    print(f"  Mid regret:   {mid_m:.3f}  (Δ from init: {red_init_mid:+.3f})")
    print(f"  Final regret: {final_m:.3f}  (Δ from init: {red_init_final:+.3f})")
    print(f"  {status}")
    print()

pass_cat9 = pass_count >= len(sigmas) * 0.5
print(f"{'PASS ✅' if pass_cat9 else 'FAIL ❌'} — {pass_count}/{len(sigmas)} sigmas reduce regret")
print()

# ============================================================
# Save JSON
# ============================================================
save = {
    'category': 9,
    'test': 'partial_regret_convergence_vs_bestfixed',
    'pass': pass_cat9,
    'config': {'N': N, 'P': P, 'T': T, 'n_seeds': n_seeds, 'sigmas': sigmas},
    'per_sigma': {
        str(sigma): {
            'init_regret_mean': float(np.mean([r['init_reg'] for r in results[sigma]])),
            'mid_regret_mean':  float(np.mean([r['mid_reg']  for r in results[sigma]])),
            'final_regret_mean':float(np.mean([r['final_reg'] for r in results[sigma]])),
            'regret_reduction': float(np.mean([r['init_reg'] - r['final_reg'] for r in results[sigma]])),
        }
        for sigma in sigmas
    }
}
with open(OUT + "cat9_result.json", 'w') as f:
    json.dump(save, f, indent=2)
print(f"Saved: {OUT}cat9_result.json")

# ============================================================
# Plot: regret trajectory
# ============================================================
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
plt.ylabel("Partial Regret (QA-MAB − BestFixed)")
plt.title("Cat 9: Regret Trajectory (learn vs best-fixed)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUT + "cat9_regret_traj.png", dpi=150)
plt.close()
print(f"Saved: {OUT}cat9_regret_traj.png")

plt.figure(figsize=(10, 5))
for sigma in sigmas:
    ths = np.array([results[sigma][si]['th_err'] for si in range(n_seeds)])
    m = np.mean(ths, axis=0)
    s = np.std(ths, axis=0)
    plt.fill_between(epochs, m - s, m + s, alpha=0.2)
    plt.plot(epochs, m, label=f"σ={sigma}", lw=2)
plt.xlabel("Epoch")
plt.ylabel("||θ̂ − θ*|| / m")
plt.title("Cat 9: θ Error Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUT + "cat9_theta_err.png", dpi=150)
plt.close()
print(f"Saved: {OUT}cat9_theta_err.png")

print("DONE")