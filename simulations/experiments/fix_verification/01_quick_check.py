"""
fix_verification_quick.py — Fast QA-MAB fix test

Tests only 4 key fixes on N={10,15}, T=500, 15 seeds:
1. Baseline (SA-weak)
2. FixA (u_hat targets B)
3. FixC (no I_hat)
4. FixAD (FixA + SA-medium)

Metrics: SW ratio vs greedy oracle, u_hat error
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment
from qa_mab import QAMAB

warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/jon_claw/qa-mab-research/simulations/results/fix_verification'
os.makedirs(RESULTS_DIR, exist_ok=True)

N_VALUES = [10, 15]
T = 500
N_SEEDS = 15
BASE_SEED = 2026
I_CAP = 0.3


class QAMABFixA(QAMAB):
    """Fix A: u_hat targets B by adding back estimated interference."""
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        throughputs = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            est_interference = sum(
                self.I_hat[i, k, j, assignment[j]]
                for j in range(self.N) if j != i)
            target = throughputs[i] + est_interference
            self.u_hat[i, k] += self.B_learn_rate * (target - self.u_hat[i, k])
        if self._prev_x is not None and self._prev_throughputs is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ki = self._prev_x[i]; kj = self._prev_x[j]
                    drop_i = max(0.0, self.u_hat[i, ki] - self._prev_throughputs[i])
                    drop_j = max(0.0, self.u_hat[j, kj] - self._prev_throughputs[j])
                    if drop_i > self.collision_threshold:
                        self.I_hat[i, ki, j, kj] = min(self.I_hat[i, ki, j, kj] + self.I_learn_rate, self.I_cap)
                    if drop_j > self.collision_threshold:
                        self.I_hat[j, kj, i, ki] = min(self.I_hat[j, kj, i, ki] + self.I_learn_rate, self.I_cap)
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}
        self.tau += self.delta_tau
        self.history.append(self.env.social_welfare(assignment))


class QAMABFixC(QAMAB):
    """Fix C: no I_hat at all."""
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        throughputs = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (throughputs[i] - self.u_hat[i, k])
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}
        self.tau += self.delta_tau
        self.history.append(self.env.social_welfare(assignment))


class QAMABFixAD(QAMABFixA):
    """FixAD: FixA + SA-medium (50 restarts x 200 iterations)."""
    def solve_qubo(self, Q):
        n, m, size = self.N, self.m, self.qubo_size
        n_restarts, n_iters = 50, 200
        T0, decay = 2.0, 0.95
        best_x, best_energy = None, float('inf')
        rng = self.rng

        for restart in range(n_restarts):
            x = np.zeros(size)
            for i in range(n):
                x[i * m + int(np.argmax(self.u_hat[i]))] = 1.0
            if restart > 0:
                for _ in range(rng.integers(1, max(2, n // 3))):
                    i = rng.integers(0, n)
                    block = x[i*m:(i+1)*m]
                    k_old = int(np.argmax(block))
                    candidates = [k for k in range(m) if k != k_old]
                    if candidates:
                        k_new = candidates[rng.integers(0, len(candidates))]
                        x[i*m+k_old] = 0.0; x[i*m+k_new] = 1.0
            energy = float(x @ Q @ x)
            if energy < best_energy:
                best_energy, best_x = energy, x.copy()
            T = T0 * (1.0 + restart * 0.3)
            for step in range(n_iters):
                T *= decay
                i = rng.integers(0, n)
                block = x[i*m:(i+1)*m]
                k_old = int(np.argmax(block))
                k_new = (k_old + 1 + rng.integers(0, m-1)) % m
                x[i*m+k_old] = 0.0; x[i*m+k_new] = 1.0
                new_energy = float(x @ Q @ x)
                delta = new_energy - energy
                if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
                    energy = new_energy
                    if energy < best_energy:
                        best_energy, best_x = energy, x.copy()
                else:
                    x[i*m+k_new] = 0.0; x[i*m+k_old] = 1.0

        assignment = {i: int(np.argmax(best_x[i*m:(i+1)*m])) for i in range(n)}
        return assignment


FIXES = {
    'Baseline': QAMAB,
    'FixA': QAMABFixA,
    'FixC': QAMABFixC,
}


def greedy_oracle(env):
    assignment = {i: int(np.argmax(env.B[i])) for i in range(env.N)}
    return float(np.sum(np.max(env.B, axis=1)))


def main():
    print("=" * 60)
    print("QA-MAB FIX VERIFICATION (quick)")
    print(f"N={N_VALUES}, T={T}, seeds={N_SEEDS}")
    print("=" * 60)

    results = {}

    for N in N_VALUES:
        print(f"\n--- N={N} ---")
        results[N] = {}

        for fix_name, FixCls in FIXES.items():
            print(f"  {fix_name}...", end=' ', flush=True)
            sw_ratios = []
            u_final = []

            for seed_idx in range(N_SEEDS):
                seed = BASE_SEED + seed_idx * 1000 + N
                env = NetworkEnvironment(N, m=4, seed=seed,
                                        B_scale='uniform', I_scale='moderate')
                opt_sw = greedy_oracle(env)

                algo = FixCls(env, tau0=0.1, delta_tau=0.05, lambda_=2.0,
                             B_learn_rate=0.2, I_learn_rate=0.05, I_cap=I_CAP,
                             seed=seed)

                for t in range(T):
                    algo.step()

                final_sw = algo.history[-1]
                sw_ratios.append(final_sw / opt_sw if opt_sw != 0 else 0)
                u_final.append(float(np.linalg.norm(algo.u_hat - env.B)))

            sw_ratios = np.array(sw_ratios)
            u_final = np.array(u_final)
            mean_ratio = float(np.mean(sw_ratios))
            std_ratio = float(np.std(sw_ratios))
            mean_u = float(np.mean(u_final))

            results[N][fix_name] = {
                'mean_sw_ratio': mean_ratio,
                'std_sw_ratio': std_ratio,
                'mean_u_final': mean_u,
            }
            print(f"SW_ratio={mean_ratio:.4f}±{std_ratio:.4f}  u_err={mean_u:.2f}")

    # Save
    out = {'config': {'N': N_VALUES, 'T': T, 'seeds': N_SEEDS}, 'results': results}
    with open(os.path.join(RESULTS_DIR, 'quick_fix.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fixes = list(FIXES.keys())
    x = np.arange(len(fixes))
    width = 0.35

    for i, N in enumerate(N_VALUES):
        means = [results[N][f]['mean_sw_ratio'] for f in fixes]
        stds = [results[N][f]['std_sw_ratio'] for f in fixes]
        axes[i].bar(x, means, yerr=stds, capsize=4,
                   color=['#e74c3c' if f == 'Baseline' else '#2ecc71' for f in fixes],
                   alpha=0.8)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(fixes)
        axes[i].set_ylabel('SW / Greedy Oracle')
        axes[i].set_title(f'N={N}')
        axes[i].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        for j, (m, s) in enumerate(zip(means, stds)):
            axes[i].text(j, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=9)

    plt.suptitle('QA-MAB Fix Comparison: SW Ratio vs Greedy Oracle (T=500)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'quick_fix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {RESULTS_DIR}/quick_fix.json + .png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Final SW / Oracle")
    print("=" * 60)
    header = "Fix".ljust(12) + "".join(f"N={n:>10}" for n in N_VALUES)
    print(header)
    for f in fixes:
        row = f.ljust(12) + "".join(f"{results[n][f]['mean_sw_ratio']:>10.4f}" for n in N_VALUES)
        print(row)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
