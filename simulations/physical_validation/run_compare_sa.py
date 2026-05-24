#!/usr/bin/env python3
"""Compare old (bit-flip) vs new (route-flip) SA on QA-MAB physical model.
- OLD: sa_sweep (bit-flip) — original binary vector SA
- NEW: sa_solve (sa_onehot alias, route-flip) — one-hot route-flip SA

Both use the same QUBO at each step. Gap = E_bitflip - E_routeflip.
Positive gap = route-flip is better.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulations.physical_validation.physical_env import AbstractWorld
from simulations.physical_validation.qa_mab_physical import QAMABPhysical
from simulations.physical_validation.sa_solver_physical import sa_solve, sa_sweep, decode_solution


P = 10
T = 30
N = 4
K = 4
m = 15
Z = 6
sigma = 0.1
n_seeds = 6
out_dir = Path(__file__).parent / "results" / "compare_sa_regret"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"N={N}, K={K}, P={P}, T={T}, sigma={sigma}, n_seeds={n_seeds}")
print("Running bit-flip (old) vs route-flip (new) SA comparison...")

results = {}
for seed_i in range(n_seeds):
    seed = 42 + seed_i
    rng = np.random.default_rng(seed)
    world = AbstractWorld(N=N, K=K, m=m, Z=Z, sigma_noise=sigma, seed=seed)
    world.refresh_epoch(rng)

    params = dict(
        C_coll=5.0, d0=150.0, sigma_noise=sigma,
        alpha=0.15, ucb_c=3.0, epoch_decay=1.0,
        lambda_onehot=10.0, gamma_0=2.0, a=0.5, b=0.3,
        sa_sweeps=200, sa_n_reads=20, sa_T_init=2.0, sa_T_final=0.05,
    )
    qa = QAMABPhysical(world=world, **params, seed=seed)

    results[seed] = {
        "old": {"energy": [], "gap": []},
        "new": {"energy": [], "gap": []},
    }

    for p in range(P):
        world.refresh_epoch(rng)
        qa.reset_epoch(p)

        for t in range(T):
            Q = qa.build_qubo()
            gamma = qa._temperature(p, t)
            Q_scaled = Q / max(gamma, 1e-8)

            # OLD: bit-flip SA (sa_sweep)
            best_x_bf, energy_bf = sa_sweep(
                Q_scaled, rng,
                n_reads=qa.sa_n_reads,
                n_sweeps=qa.sa_sweeps,
                T_init=qa.sa_T_init, T_final=qa.sa_T_final
            )
            chosen_bf = decode_solution(best_x_bf, N, K)

            # NEW: route-flip SA (sa_solve = sa_onehot alias) — returns (N,) paths only
            chosen_rf = sa_solve(
                Q_scaled, N, K, rng,
                n_restarts=20, n_iters=200, T0=2.0, decay=0.995
            )
            # Compute QUBO energy for route-flip result
            x_rf = np.zeros(N * K, dtype=int)
            for n in range(N):
                x_rf[n * K + chosen_rf[n]] = 1
            energy_rf = float(x_rf @ Q_scaled @ x_rf)

            # Gap: positive = route-flip better
            gap = energy_bf - energy_rf
            results[seed]["old"]["energy"].append(float(energy_bf))
            results[seed]["new"]["energy"].append(float(energy_rf))
            results[seed]["old"]["gap"].append(float(gap))

            qa.update(chosen_bf, world.compute_losses(chosen_bf, rng)[0])

    if (seed_i + 1) % 3 == 0:
        print(f"  seeds {seed_i+1}/{n_seeds} done")

print("Saving results...")

all_old_energy = np.array([results[s]["old"]["energy"] for s in results])
all_new_energy = np.array([results[s]["new"]["energy"] for s in results])
all_gap = np.array([results[s]["old"]["gap"] for s in results])

window = 20
def rolling_mean(arr):
    return np.convolve(arr.mean(axis=0), np.ones(window)/window, mode='valid')

old_en_roll = rolling_mean(all_old_energy)
new_en_roll = rolling_mean(all_new_energy)
gap_roll = rolling_mean(all_gap)
steps = np.arange(len(old_en_roll)) + window

cumgap = np.cumsum(all_gap, axis=1)
cumgap_mean = cumgap.mean(axis=0)
cumgap_std = cumgap.std(axis=0)

old_epoch_gap = np.zeros((n_seeds, P))
for p in range(P):
    old_epoch_gap[:, p] = all_gap[:, p*T:(p+1)*T].mean(axis=1)
gap_er_mean = old_epoch_gap.mean(axis=0)
gap_er_std = old_epoch_gap.std(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(steps, old_en_roll, label='bit-flip (old)', color='#2196F3', lw=2)
axes[0].plot(steps, new_en_roll, label='route-flip (new)', color='#FF5722', lw=2)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('QUBO Energy (20-step avg)')
axes[0].set_title('QUBO Energy Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(cumgap_mean, label='Cumulative Gap', color='#4CAF50', lw=2)
axes[1].fill_between(range(len(cumgap_mean)), cumgap_mean - cumgap_std, cumgap_mean + cumgap_std, alpha=0.2, color='#4CAF50')
axes[1].axhline(0, color='gray', lw=1, ls='--')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Cumulative Gap (bit-flip - route-flip)')
axes[1].set_title('Cumulative Energy Gap')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

epochs = np.arange(P)
width = 0.35
axes[2].bar(epochs - width/2, gap_er_mean, width, yerr=gap_er_std,
            label='Gap (old - new)', color='#4CAF50', alpha=0.8, capsize=3)
axes[2].axhline(0, color='gray', lw=1, ls='--')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Mean Energy Gap per Epoch')
axes[2].set_title('Per-Epoch Gap (+ = route-flip wins)')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle(f'SA Comparison: N={N}, K={K}, sigma={sigma}, {n_seeds} seeds')
plt.tight_layout()
plt.savefig(out_dir / 'sa_comparison_plots.png', dpi=150)
plt.close()

total_gap = all_gap.sum(axis=1)
win_new = int((total_gap > 0).sum())
win_old = int((total_gap < 0).sum())
tie = int((total_gap == 0).sum())

total_old_en = all_old_energy.sum(axis=1)
total_new_en = all_new_energy.sum(axis=1)

summary = {
    "config": dict(N=N, K=K, P=P, T=T, sigma=sigma, n_seeds=n_seeds,
                   old_method="bit-flip (sa_sweep)", new_method="route-flip (sa_onehot)"),
    "total_energy_mean": {
        "old_bit_flip": float(total_old_en.mean()),
        "new_route_flip": float(total_new_en.mean()),
    },
    "total_energy_std": {
        "old_bit_flip": float(total_old_en.std()),
        "new_route_flip": float(total_new_en.std()),
    },
    "gap_mean": float(total_gap.mean()),
    "gap_std": float(total_gap.std()),
    "win_rate": {
        "new_route_flip_lower_energy": f"{win_new}/{n_seeds} seeds",
        "old_bit_flip_lower_energy": f"{win_old}/{n_seeds} seeds",
        "tie": f"{tie}/{n_seeds} seeds",
    },
    "per_epoch_gap_mean": [float(x) for x in gap_er_mean],
}
with open(out_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n=== Summary ===")
print(f"  Total energy -- old (bit-flip): {total_old_en.mean():.3f} +/- {total_old_en.std():.3f}")
print(f"  Total energy -- new (route-flip): {total_new_en.mean():.3f} +/- {total_new_en.std():.3f}")
print(f"  Gap (old - new): {total_gap.mean():.3f} +/- {total_gap.std():.3f}")
print(f"  Route-flip wins: {win_new}/{n_seeds} seeds, Bit-flip wins: {win_old}/{n_seeds}")
print(f"\n  Plots + summary.json -> {out_dir}/")