#!/usr/bin/env python3
"""Cat 9: Partial Regret Convergence — THE critical test.

Question: As QA-MAB learns θ̂/φ̂, does its regret vs OptimalAgent → 0?

Setup:
- OptimalAgent knows θ* and φ* (frozen, no learning) — best possible without learning
- QA-MAB starts with θ̂=0, φ̂=0 and LEARNS over epochs
- If QA-MAB regret → 0 as epochs increase → learning works
- If QA-MAB regret stays high → learning doesn't help

Tracking per-epoch:
  regret[p] = mean(QA_loss[p]) − mean(Optimal_loss[p])
  θ_error[p] = ||θ̂[p] − θ*|| / m
  φ_error[p] = ||φ̂[p] − φ*|| / Z

Also test: does learning work under noise σ?
"""
import sys, os
sys.path.insert(0, '/Users/jon_claw/qa-mab-research')
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')

import numpy as np
import json, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product

OUT = "simulations/results/validation_cat9_physical/"
os.makedirs(OUT, exist_ok=True)

from physical_validation.physical_env import AbstractWorld
from physical_validation.qa_mab_physical import QAMABPhysical
from physical_validation.agents_physical.oracle_agent import OracleAgent

# ============================================================
# Test configuration: large scale as requested
# ============================================================
N_vals   = [5, 10, 15, 20, 30]
P, T     = 30, 80     # 30 epochs × 80 steps — substantial learning
n_seeds  = 20         # 20 seeds per N for significance
sigmas   = [0.0, 0.1] # clean + noisy

# ============================================================
# OptimalAgent uses brute-force. For large N this is expensive.
# We'll use N ≤ 20 for brute-force; N=30 uses Oracle (SA).
# ============================================================
MAX_BF_N = 20  # brute-force limit (K=4 → 4^20 ≈ 1e12 impossible)
               # practical limit is N=10 (4^10=1M combos, ~seconds)
               # For N=15,20 we fall back to OracleAgent


class OptimalAgentBF:
    """OptimalAgent with brute-force (only for N where it's feasible)."""

    name = "Optimal-BF"

    def __init__(self, world, seed=42):
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.theta_hat = world.theta_star.copy()
        self.phi_hat   = world.phi_star.copy()

    def act(self, t, p):
        N, K = self.world.N, self.world.K
        ps = self.world.pathset

        best_loss = float('inf')
        best_combo = np.zeros(N, dtype=int)

        # Brute-force only feasible for N ≤ 10 (K^N ≤ 4^10 = 1M)
        if N > 10:
            # Fall back to greedy for N > 10
            return self._greedy_paths()

        for combo in product(range(K), repeat=N):
            paths = np.array(combo, dtype=int)
            loss = self._true_loss(paths)
            if loss < best_loss:
                best_loss = loss
                best_combo = paths

        return best_combo

    def _true_loss(self, chosen_paths):
        N, K = self.world.N, self.world.K
        ps = self.world.pathset

        losses = np.zeros(N, dtype=float)
        selected_uav = np.array([ps.path_uav_membership[n, chosen_paths[n]] for n in range(N)])
        shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
        np.fill_diagonal(shared, False)
        collision_counts = shared.sum(axis=1).astype(float)

        for n in range(N):
            k = chosen_paths[n]
            uav_mask = ps.path_uav_membership[n, k]
            zone_mask = ps.path_zone_membership[n, k]
            losses[n] = (
                float(self.theta_hat[uav_mask].sum())
                + float(self.phi_hat[zone_mask].sum())
                + self.world.C_coll * collision_counts[n]
            )
            a = n * K + k
            for l in range(N):
                if l == n: continue
                kl = chosen_paths[l]
                b = l * K + kl
                losses[n] += np.exp(-ps.pair_min_dist[a, b] / self.world.d0)

        return losses.sum()

    def _greedy_paths(self):
        """Fallback: pick path with lowest expected loss per flow."""
        N, K = self.world.N, self.world.K
        ps = self.world.pathset
        chosen = np.zeros(N, dtype=int)

        for n in range(N):
            best_k = 0
            best_cost = float('inf')
            for k in range(K):
                paths = chosen.copy()
                paths[n] = k
                cost = self._true_loss(paths)
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            chosen[n] = best_k

        return chosen

    def update(self, chosen_paths, losses):
        pass

    def reset_epoch(self, p):
        self.world.refresh_epoch(self.rng)


class OracleAgentSA:
    """OracleAgent with SA solver — used when brute-force is infeasible."""

    name = "Oracle-SA"

    def __init__(self, world, seed=42):
        from physical_validation.sa_solver_physical import sa_solve, decode_solution
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.theta_hat = world.theta_star.copy()
        self.phi_hat   = world.phi_star.copy()
        self._sa_solve = sa_solve
        self._decode   = decode_solution

    def act(self, t, p):
        Q = self.world.build_qubo_from_estimates(self.theta_hat, self.phi_hat)
        best_E = float('inf')
        best_x = None
        for _ in range(20):
            x, E = self._sa_solve(Q, self.rng, n_reads=1, n_sweeps=200, T_init=2.0, T_final=0.05)
            if E < best_E:
                best_E = E
                best_x = x
        return self._decode(best_x, self.world.N, self.world.K)

    def update(self, chosen_paths, losses):
        pass

    def reset_epoch(self, p):
        self.world.refresh_epoch(self.rng)


def run_single(N, sigma, seed):
    """Run one seed: QA-MAB vs Optimal/Oracle, track per-epoch regret."""
    rng_loss = np.random.default_rng(seed + 30000)

    world = AbstractWorld(N=N, K=4, m=20, Z=6, sigma_noise=sigma, seed=seed)

    qa  = QAMABPhysical(world, seed=seed)
    opt = OptimalAgentBF(world, seed=seed + 10000) if N <= 10 else OracleAgentSA(world, seed=seed + 10000)

    regret_per_epoch  = np.zeros(P)
    qa_loss_per_epoch = np.zeros(P)
    opt_loss_per_epoch = np.zeros(P)
    theta_err_per_epoch = np.zeros(P)
    phi_err_per_epoch   = np.zeros(P)

    for p in range(P):
        qa.reset_epoch(p); opt.reset_epoch(p)

        ep_qa = []; ep_opt = []

        for t in range(T):
            c_qa = qa.act(t, p)
            l_qa = world.compute_losses(c_qa, rng_loss)
            qa.update(c_qa, l_qa)
            ep_qa.append(l_qa.mean())

            c_opt = opt.act(t, p)
            l_opt = world.compute_losses(c_opt, rng_loss)
            opt.update(c_opt, l_opt)  # Optimal doesn't learn but needs this for interface
            ep_opt.append(l_opt.mean())

        regret_per_epoch[p]  = np.mean(ep_qa) - np.mean(ep_opt)
        qa_loss_per_epoch[p] = np.mean(ep_qa)
        opt_loss_per_epoch[p] = np.mean(ep_opt)
        theta_err_per_epoch[p] = float(np.linalg.norm(qa.theta_hat - world.theta_star) / world.m)
        phi_err_per_epoch[p]   = float(np.linalg.norm(qa.phi_hat - world.phi_star) / world.Z)

    return {
        "regret":         regret_per_epoch.tolist(),
        "qa_loss":        qa_loss_per_epoch.tolist(),
        "opt_loss":       opt_loss_per_epoch.tolist(),
        "theta_err":      theta_err_per_epoch.tolist(),
        "phi_err":        phi_err_per_epoch.tolist(),
        "final_regret":   float(regret_per_epoch[-5:].mean()),
        "init_regret":    float(regret_per_epoch[:5].mean()),
    }


# ============================================================
# RUN
# ============================================================
print("Cat 9: Partial Regret Convergence")
print(f"Config: N={N_vals}, P={P}, T={T}, seeds={n_seeds}, sigma={sigmas}")
print()

results = {}
for N in N_vals:
    print(f"\n=== N={N} ({'brute-force' if N<=10 else 'SA-based'}) ===", flush=True)
    results[N] = {}

    for sigma in sigmas:
        t0 = time.time()
        regrets = []; qa_ls = []; opt_ls = []
        theta_errs = []; phi_errs = []

        for si in range(n_seeds):
            seed = 42 + si
            print(f"  σ={sigma} seed={seed}", end="\r", flush=True)
            r = run_single(N, sigma, seed)
            regrets.append(r["regret"])
            qa_ls.append(r["qa_loss"])
            opt_ls.append(r["opt_loss"])
            theta_errs.append(r["theta_err"])
            phi_errs.append(r["phi_err"])

        elapsed = time.time() - t0
        print(f"  σ={sigma}: {elapsed:.0f}s   init={np.mean([r['init_regret'] for r in [{'init_regret': np.mean(regrets[i][:5])} for i in range(n_seeds)]]):.3f}  final={np.mean([r['final_regret'] for r in [{'final_regret': np.mean(regrets[i][-5:]) } for i in range(n_seeds)]]):.3f}" + " "*10)

        results[N][sigma] = {
            "n_seeds":      n_seeds,
            "regrets":      regrets,
            "qa_losses":    qa_ls,
            "opt_losses":   opt_ls,
            "theta_errs":   theta_errs,
            "phi_errs":     phi_errs,
            "final_regrets": [np.mean(r[-5:]) for r in regrets],
            "init_regrets":  [np.mean(r[:5])  for r in regrets],
        }


# ============================================================
# SUMMARY
# ============================================================
print("\n\n=== SUMMARY ===")
pass_count = 0
for N in N_vals:
    for sigma in sigmas:
        init_mean = np.mean(results[N][sigma]["init_regrets"])
        final_mean = np.mean(results[N][sigma]["final_regrets"])
        reduction = init_mean - final_mean
        pct_reduction = (reduction / max(abs(init_mean), 0.01)) * 100

        status = "✅" if reduction > 0 else "❌"
        print(f"N={N:2d} σ={sigma}: init={init_mean:.3f}  final={final_mean:.3f}  Δ={reduction:+.3f} ({pct_reduction:+.1f}%) {status}")

        if reduction > 0:
            pass_count += 1

n_total = len(N_vals) * len(sigmas)
pass_cat9 = pass_count >= n_total * 0.8  # 80% pass threshold

print(f"\n{'PASS ✅' if pass_cat9 else 'FAIL ❌'} — {pass_count}/{n_total} combos reduce regret")

# ============================================================
# SAVE RESULTS
# ============================================================
result_summary = {
    "category": 9,
    "test": "partial_regret_convergence",
    "pass": pass_cat9,
    "reason": f"{pass_count}/{n_total} N×σ combos show regret reduction over epochs",
    "config": {"N_vals": N_vals, "P": P, "T": T, "n_seeds": n_seeds, "sigmas": sigmas},
    "per_N_sigma": {
        str(N): {
            str(sigma): {
                "init_regret_mean":  float(np.mean(results[N][sigma]["init_regrets"])),
                "final_regret_mean": float(np.mean(results[N][sigma]["final_regrets"])),
                "regret_reduction":  float(np.mean(results[N][sigma]["init_regrets"]) - np.mean(results[N][sigma]["final_regrets"])),
            } for sigma in sigmas
        } for N in N_vals
    }
}
with open(OUT + "result.json", "w") as f:
    json.dump(result_summary, f, indent=2)

# Save full trajectories
trajectories = {
    str(N): {
        str(sigma): {
            "regret_mean": np.mean(results[N][sigma]["regrets"], axis=0).tolist(),
            "regret_std":  np.std(results[N][sigma]["regrets"], axis=0).tolist(),
            "theta_err_mean": np.mean(results[N][sigma]["theta_errs"], axis=0).tolist(),
            "phi_err_mean":   np.mean(results[N][sigma]["phi_errs"], axis=0).tolist(),
        } for sigma in sigmas
    } for N in N_vals
}
with open(OUT + "trajectories.json", "w") as f:
    json.dump(trajectories, f, indent=2)

# ============================================================
# PLOTS
# ============================================================
epochs = np.arange(P)

# Per N: regret convergence (σ=0)
plt.figure(figsize=(12, 5))
for N in N_vals:
    mean_r = np.mean(results[N][0.0]["regrets"], axis=0)
    std_r  = np.std(results[N][0.0]["regrets"], axis=0)
    plt.fill_between(epochs, mean_r - std_r, mean_r + std_r, alpha=0.2)
    plt.plot(epochs, mean_r, label=f'N={N}', lw=2)
plt.axhline(0, color='gray', linestyle='--', lw=1)
plt.xlabel("Epoch"); plt.ylabel("Partial Regret (QA − Optimal)")
plt.title("Cat 9: Regret Convergence by N (σ=0)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(f"{OUT}regret_by_N.png", dpi=150); plt.close()

# Per N: regret convergence (σ=0.1)
plt.figure(figsize=(12, 5))
for N in N_vals:
    mean_r = np.mean(results[N][0.1]["regrets"], axis=0)
    std_r  = np.std(results[N][0.1]["regrets"], axis=0)
    plt.fill_between(epochs, mean_r - std_r, mean_r + std_r, alpha=0.2)
    plt.plot(epochs, mean_r, label=f'N={N}', lw=2)
plt.axhline(0, color='gray', linestyle='--', lw=1)
plt.xlabel("Epoch"); plt.ylabel("Partial Regret (QA − Optimal)")
plt.title("Cat 9: Regret Convergence by N (σ=0.1)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(f"{OUT}regret_by_N_noise.png", dpi=150); plt.close()

# Per N: θ error convergence (σ=0)
plt.figure(figsize=(12, 5))
for N in N_vals:
    mean_th = np.mean(results[N][0.0]["theta_errs"], axis=0)
    plt.plot(epochs, mean_th, label=f'N={N}', lw=2)
plt.xlabel("Epoch"); plt.ylabel("||θ̂ − θ*|| / m")
plt.title("Cat 9: θ Error Convergence by N (σ=0)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(f"{OUT}theta_err_by_N.png", dpi=150); plt.close()

# All combos: regret heatmap-style (final vs init)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
init_mat = np.array([[np.mean(results[N][s]["init_regrets"]) for s in sigmas] for N in N_vals])
final_mat = np.array([[np.mean(results[N][s]["final_regrets"]) for s in sigmas] for N in N_vals])
im0 = axes[0].imshow(init_mat, aspect='auto', cmap='RdYlGn_r')
axes[0].set_xticks(range(len(sigmas))); axes[0].set_xticklabels([f'σ={s}' for s in sigmas])
axes[0].set_yticks(range(len(N_vals))); axes[0].set_yticklabels(N_vals)
axes[0].set_xlabel("Noise"); axes[0].set_ylabel("N flows")
axes[0].set_title("Initial Regret (Epochs 0-4)")
plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(final_mat, aspect='auto', cmap='RdYlGn_r')
axes[1].set_xticks(range(len(sigmas))); axes[1].set_xticklabels([f'σ={s}' for s in sigmas])
axes[1].set_yticks(range(len(N_vals))); axes[1].set_yticklabels(N_vals)
axes[1].set_xlabel("Noise"); axes[1].set_ylabel("N flows")
axes[1].set_title("Final Regret (Epochs 25-29)")
plt.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.savefig(f"{OUT}regret_heatmap.png", dpi=150); plt.close()

print(f"\nAll plots saved to {OUT}")
print(f"Cat 9: {'PASS ✅' if pass_cat9 else 'FAIL ❌'}")