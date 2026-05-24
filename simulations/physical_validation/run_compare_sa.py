#!/usr/bin/env python3
"""Compare old (bit-flip) vs new (route-flip) SA on QA-MAB physical model.
Both use the same QUBO at each step. The "regret" is the energy gap
between bit-flip and route-flip: gap[t] = E_bitflip[t] - E_routeflip[t].
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
from simulations.physical_validation.sa_solver_physical import sa_solve as sa_bitflip, decode_solution


def route_flip_solve(Q, N, K, rng, n_restarts=20, n_iters=200, T0=2.0, decay=0.995):
    """Simulated annealing with route-flip proposals (one-hot encoding)."""
    best_x = np.zeros(N, dtype=int)
    best_energy = float('inf')
    Q_binary = Q  # the QUBO matrix (scaled already)
    for _ in range(n_restarts):
        x = np.zeros(N, dtype=int)
        for it in range(n_iters):
            T = T0 * (decay ** it)
            if T <= 1e-8:
                T = 1e-8
            n = int(rng.integers(0, N))
            k2 = int(rng.integers(0, K))
            old_k = int(x[n])
            if k2 == old_k:
                continue
            old_i = n * K + old_k
            new_i = n * K + k2
            delta_self = Q[new_i, new_i] - Q[old_i, old_i]
            delta_cross = (k2 - old_k) * sum(
                Q[new_i, l * K + int(x[l])] - Q[old_i, l * K + int(x[l])]
                for l in range(N) if l != n
            )
            delta_E = delta_self + delta_cross
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                x[n] = k2
                # Only compute full energy when we accept a move
                x_binary = np.zeros(N * K, dtype=int)
                for nn in range(N):
                    x_binary[nn * K + x[nn]] = 1
                energy = float(x_binary @ Q_binary @ x_binary)
                if energy < best_energy:
                    best_energy = energy
                    best_x = x.copy()
    return best_x, best_energy


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
print("Running old (bit-flip) vs new (route-flip) SA comparison...")

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
        "old": {"energy": [], "regret": []},
        "new": {"energy": [], "regret": []},
    }

    for p in range(P):
        world.refresh_epoch(rng)
        qa.reset_epoch(p, rng)

        for t in range(T):
            Q = qa.build_qubo()
            gamma = qa._temperature(p, t)
            Q_scaled = Q / max(gamma, 1e-8)

            # OLD SA (bit-flip) — decode to (N,) paths
            best_x_bf, energy_bf = sa_bitflip(
                Q_scaled, rng,
                n_reads=qa.sa_n_reads,
                n_sweeps=qa.sa_sweeps,
                T_init=qa.sa_T_init, T_final=qa.sa_T_final
            )
            chosen_bf = decode_solution(best_x_bf, N, K)
            loss_bf = world.compute_losses(chosen_bf, rng)[0]

            # NEW SA (route-flip)
            best_x_rf, energy_rf = route_flip_solve(
                Q_scaled, N, K, rng,
                n_restarts=20, n_iters=1000, T0=2.0, decay=0.999
            )
            chosen_rf = best_x_rf
            loss_rf = world.compute_losses(chosen_rf, rng)[0]

            # Gap = how much worse is bit-flip vs route-flip
            gap = energy_bf - energy_rf
            results[seed]["old"]["energy"].append(float(energy_bf))
            results[seed]["new"]["energy"].append(float(energy_rf))
            results[seed]["old"]["regret"].append(float(gap))
            results[seed]["new"]["regret"].append(0.0)  # route-flip = reference

            qa.update(chosen_bf, loss_bf)

    if (seed_i + 1) % 4 == 0:
        print(f"  seeds {seed_i+1}/{n_seeds} done")

print("Saving results...")

all_old_energy = np.array([results[s]["old"]["energy"] for s in results])
all_new_energy = np.array([results[s]["new"]["energy"] for s in results])
all_gap = np.array([results[s]["old"]["regret"] for s in results])

window = 20
def rolling_mean(arr):
    return np.convolve(arr.mean(axis=0), np.ones(window)/window, mode='valid')

old_en_roll = rolling_mean(all_old_energy)
new_en_roll = rolling_mean(all_new_energy)
gap_roll = rolling_mean(all_gap)
steps = np.arange(len(old_en_roll)) + window

# Cumulative gap
cumgap = np.cumsum(all_gap, axis=1)
cumgap_mean = cumgap.mean(axis=0)
cumgap_std = cumgap.std(axis=0)

# Per-epoch gap
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
axes[2].set_title('Per-Epoch Gap (positive = route-flip wins)')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle(f'SA Comparison: N={N}, K={K}, sigma={sigma}, {n_seeds} seeds')
plt.tight_layout()
plt.savefig(out_dir / 'sa_comparison_plots.png', dpi=150)
plt.close()

# Summary stats
total_gap = all_gap.sum(axis=1)
win_new = int((total_gap > 0).sum())  # route-flip better when gap > 0
win_old = int((total_gap < 0).sum())  # bit-flip better when gap < 0
tie = int((total_gap == 0).sum())

total_old_en = all_old_energy.sum(axis=1)
total_new_en = all_new_energy.sum(axis=1)

summary = {
    "config": dict(N=N, K=K, P=P, T=T, sigma=sigma, n_seeds=n_seeds,
                   old_method="bit-flip", new_method="route-flip"),
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