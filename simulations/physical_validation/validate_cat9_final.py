#!/usr/bin/env python3
"""Cat 9 CORRECT: QA-MAB (learns) vs OptimalFixed (knows optimal, never learns).

Question: As QA-MAB learns over epochs, does its gap vs optimal → 0?
"""
import sys, os, time
sys.path.insert(0, '/Users/jon_claw/qa-mab-research')
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')

from physical_validation.physical_env import AbstractWorld
from physical_validation.qa_mab_physical import QAMABPhysical
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product

OUT = "simulations/results/validation_cat9_physical/"
os.makedirs(OUT, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────
N, P, T = 5, 30, 150      # N=5: brute-force viable (4^5=1024 combos)
n_seeds = 8
sigmas  = [0.0, 0.1]      # fast test: just 2 sigmas

# ── OptimalFixed: knows θ*/φ*, exhaustive, no learning ──────────────────
class OptimalFixed:
    name = "Optimal-Fixed"

    def __init__(self, world, seed=42):
        self.world = world
        self.rng  = np.random.default_rng(seed)
        self.theta_hat = world.theta_star.copy()
        self.phi_hat   = world.phi_star.copy()

    def act(self, t, p):
        N, K, ps = self.world.N, self.world.K, self.world.pathset
        best_loss, best_chosen = float('inf'), None
        for combo in product(range(K), repeat=N):
            loss = self._loss(np.array(combo, dtype=int))
            if loss < best_loss:
                best_loss, best_chosen = loss, np.array(combo, dtype=int)
        return best_chosen

    def _loss(self, chosen):
        N, K, ps = self.world.N, self.world.K, self.world.pathset
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
                if l == n: continue
                b = l * K + chosen[l]
                losses[n] += np.exp(-ps.pair_min_dist[a, b] / self.world.d0)
        return losses.sum()

    def update(self, *a): pass
    def reset_epoch(self, p: int, world_rng=None) -> None:
        rng = self.rng if world_rng is None else world_rng
        self.world.refresh_epoch(rng)


def run_single(sigma, seed, verbose=True):
    rng = np.random.default_rng(seed + 50000)
    world = AbstractWorld(N=N, K=4, m=20, Z=6,
                         sigma_noise=sigma, seed=seed)
    qa  = QAMABPhysical(world, seed=seed)
    opt = OptimalFixed(world, seed=seed + 20000)

    regret = np.zeros(P)
    qa_loss= np.zeros(P)
    opt_loss=np.zeros(P)
    th_err = np.zeros(P)
    ph_err = np.zeros(P)

    t0 = time.time()
    shared_epoch_rng = np.random.default_rng(seed + 50000)
    for p in range(P):
        # BUG FIX: shared epoch RNG — both agents see same world topology
        epoch_seed = int(shared_epoch_rng.integers(0, 2**63-1))
        world_rng = np.random.default_rng(epoch_seed)
        qa.reset_epoch(p, world_rng=world_rng)
        opt.reset_epoch(p, world_rng=world_rng)
        ep_qa, ep_or = [], []
        for t in range(T):
            c_qa = qa.act(t, p)
            l_qa = world.compute_losses(c_qa, rng)
            qa.update(c_qa, l_qa)
            ep_qa.append(l_qa.mean())

            c_or = opt.act(t, p)
            l_or = world.compute_losses(c_or, rng)
            opt.update(c_or, l_or)
            ep_or.append(l_or.mean())

        regret[p]  = np.mean(ep_qa) - np.mean(ep_or)
        qa_loss[p] = np.mean(ep_qa)
        opt_loss[p]= np.mean(ep_or)
        th_err[p]  = float(np.linalg.norm(qa.theta_hat - world.theta_star) / world.m)
        ph_err[p]  = float(np.linalg.norm(qa.phi_hat   - world.phi_star) / world.Z)

        if verbose and p % 5 == 0:
            elapsed = time.time() - t0
            print(f"    p={p}/{P} ({elapsed:.0f}s)", flush=True)

    return dict(
        regret=regret.tolist(), qa_loss=qa_loss.tolist(),
        opt_loss=opt_loss.tolist(), th_err=th_err.tolist(), ph_err=ph_err.tolist(),
        init_reg=float(np.mean(regret[:5])), final_reg=float(np.mean(regret[-5:])),
        mid_reg=float(np.mean(regret[P//2-2:P//2+2])),
    )


print("=" * 60)
print("Cat 9 CORRECT: QA-MAB vs OptimalFixed")
print("=" * 60)
print(f"N={N}, P={P}, T={T}, seeds={n_seeds}, sigmas={sigmas}\n")

results = {}
for sigma in sigmas:
    print(f"=== sigma={sigma} ===")
    results[sigma] = []
    t0 = time.time()
    for si in range(n_seeds):
        r = run_single(sigma, 42 + si)
        results[sigma].append(r)
        print(f"  seed={42+si}: init={r['init_reg']:.3f} mid={r['mid_reg']:.3f} "
              f"final={r['final_reg']:.3f}  ({time.time()-t0:.0f}s)")
    print(f"  sigma={sigma} done in {time.time()-t0:.0f}s\n")
    flush = sys.stdout.flush

print("=" * 60)
print("SUMMARY")
print("=" * 60)
pass_count = 0
for sigma in sigmas:
    ia = np.mean([r['init_reg'] for r in results[sigma]])
    ma = np.mean([r['mid_reg']  for r in results[sigma]])
    fa = np.mean([r['final_reg'] for r in results[sigma]])
    ok = fa < ia
    if ok: pass_count += 1
    print(f"σ={sigma}: init={ia:.3f} mid={ma:.3f} final={fa:.3f}  Δ={fa-ia:+.3f}  {'✅' if ok else '❌'}")

pass_cat = pass_count >= len(sigmas) * 0.5
print(f"\n{'PASS ✅' if pass_cat else 'FAIL ❌'} — {pass_count}/{len(sigmas)} sigmas")

save = {
    'category': 9, 'test': 'qa_mab_vs_optimal_fixed',
    'pass': pass_cat,
    'config': {'N': N, 'P': P, 'T': T, 'n_seeds': n_seeds, 'sigmas': sigmas},
    'per_sigma': {
        str(s): {
            'init_regret_mean': float(np.mean([r['init_reg'] for r in results[s]])),
            'mid_regret_mean':  float(np.mean([r['mid_reg']  for r in results[s]])),
            'final_regret_mean':float(np.mean([r['final_reg'] for r in results[s]])),
            'regret_reduction': float(np.mean([r['init_reg'] - r['final_reg'] for r in results[s]])),
        } for s in sigmas
    }
}
with open(OUT + "cat9_final_result.json", 'w') as f:
    json.dump(save, f, indent=2)

epochs = np.arange(P)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for sigma in sigmas:
    regs = np.array([results[sigma][s]['regret'] for s in range(n_seeds)])
    m, s = regs.mean(axis=0), regs.std(axis=0)
    axes[0].fill_between(epochs, m-s, m+s, alpha=0.2)
    axes[0].plot(epochs, m, label=f'σ={sigma}', lw=2)
for sigma in sigmas:
    ths = np.array([results[sigma][s]['th_err'] for s in range(n_seeds)])
    m, s = ths.mean(axis=0), ths.std(axis=0)
    axes[1].fill_between(epochs, m-s, m+s, alpha=0.2)
    axes[1].plot(epochs, m, label=f'σ={sigma}', lw=2)
axes[0].axhline(0, color='gray', ls='--', lw=1)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Partial Regret")
axes[0].set_title("Cat 9: Regret Trajectory"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("||θ̂ − θ*|| / m")
axes[1].set_title("Cat 9: θ Error Convergence"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "cat9_final_regret_traj.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT}cat9_final_*")
print("DONE")