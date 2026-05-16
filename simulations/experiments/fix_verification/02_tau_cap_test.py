"""
tau_cap_test.py — The REAL fix: cap tau to prevent QUBO from freezing

Key insight: tau = 0.1 + t * 0.05 grows without bound.
At t=500: tau=25.1 — QUBO landscape is infinitely sharp.
SA becomes pure greedy, no exploration, no learning.

Fix: cap tau at 5.0 so QUBO stays "annealable" throughout.

Tests:
1. Baseline (tau grows to 25)
2. FixTau (tau capped at 5.0)
3. FixTauFixC (tau capped + no I_hat)

N={10, 15}, T=1000, 20 seeds
"""

import os, sys, json, time, warnings
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
T = 1000
N_SEEDS = 20
BASE_SEED = 2026
TAU_CAP = 5.0
I_CAP = 0.3


class QAMABTauCap(QAMAB):
    """tau grows but is capped at TAU_CAP to keep QUBO annealable."""
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        throughputs = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (throughputs[i] - self.u_hat[i, k])
        if self._prev_x is not None and self._prev_throughputs is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ki = self._prev_x[i]; kj = self._prev_x[j]
                    drop_i = max(0.0, self.u_hat[i, ki] - self._prev_throughputs[i])
                    drop_j = max(0.0, self.u_hat[j, kj] - self._prev_throughputs[j])
                    if drop_i > self.collision_threshold:
                        self.I_hat[i, ki, j, kj] = min(self.I_hat[i, ki, j, kj] + self.I_learn_rate, I_CAP)
                    if drop_j > self.collision_threshold:
                        self.I_hat[j, kj, i, ki] = min(self.I_hat[j, kj, i, ki] + self.I_learn_rate, I_CAP)
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}
        self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        sw = self.env.social_welfare(assignment)
        self.history.append(sw)


class QAMABTauCapFixC(QAMABTauCap):
    """No I_hat at all — pure u_hat learning with capped tau."""
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        throughputs = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (throughputs[i] - self.u_hat[i, k])
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}
        self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        sw = self.env.social_welfare(assignment)
        self.history.append(sw)


FIXES = {
    'Baseline': QAMAB,
    'FixTau': QAMABTauCap,
    'FixTauFixC': QAMABTauCapFixC,
}


def greedy_oracle(env):
    assignment = {i: int(np.argmax(env.B[i])) for i in range(env.N)}
    return float(np.sum(np.max(env.B, axis=1)))


def brute_force_small(env):
    """For N<=8 only."""
    from itertools import product
    best_sw = -np.inf
    for combo in product(range(env.m), repeat=env.N):
        assignment = {i: combo[i] for i in range(env.N)}
        sw = env.social_welfare(assignment)
        if sw > best_sw:
            best_sw = sw
    return best_sw


def main():
    print("=" * 60)
    print("TAU CAP TEST — The Real Fix")
    print(f"N={N_VALUES}, T={T}, seeds={N_SEEDS}, TAU_CAP={TAU_CAP}")
    print("=" * 60)

    results = {}

    for N in N_VALUES:
        print(f"\n--- N={N} ---")
        results[N] = {}

        for fix_name, FixCls in FIXES.items():
            print(f"  {fix_name}...", end=' ', flush=True)
            sw_ratios = []
            sw_finals = []
            u_finals = []
            tau_vals = []

            for seed_idx in range(N_SEEDS):
                seed = BASE_SEED + seed_idx * 1000 + N
                env = NetworkEnvironment(N, m=4, seed=seed,
                                        B_scale='uniform', I_scale='moderate')

                if N <= 8:
                    opt_sw = brute_force_small(env)
                else:
                    opt_sw = greedy_oracle(env)

                algo = FixCls(env, tau0=0.1, delta_tau=0.05, lambda_=2.0,
                             B_learn_rate=0.2, I_learn_rate=0.05, I_cap=I_CAP,
                             seed=seed)

                for t in range(T):
                    algo.step()

                final_sw = algo.history[-1]
                sw_ratios.append(final_sw / opt_sw if opt_sw != 0 else 0)
                sw_finals.append(final_sw)
                u_finals.append(float(np.linalg.norm(algo.u_hat - env.B)))
                tau_vals.append(algo.tau)

            sw_ratios = np.array(sw_ratios)
            sw_finals = np.array(sw_finals)
            u_finals = np.array(u_finals)
            tau_vals = np.array(tau_vals)

            results[N][fix_name] = {
                'mean_sw_ratio': float(np.mean(sw_ratios)),
                'std_sw_ratio': float(np.std(sw_ratios)),
                'mean_sw_final': float(np.mean(sw_finals)),
                'mean_u_final': float(np.mean(u_finals)),
                'mean_tau': float(np.mean(tau_vals)),
                'opt_type': 'brute_force' if N <= 8 else 'greedy',
            }
            print(f"SW_ratio={np.mean(sw_ratios):.4f}±{np.std(sw_ratios):.4f}  "
                  f"SW_final={np.mean(sw_finals):.4f}  u_err={np.mean(u_finals):.2f}  "
                  f"tau={np.mean(tau_vals):.2f}")

    out = {'config': {'N': N_VALUES, 'T': T, 'seeds': N_SEEDS, 'tau_cap': TAU_CAP}, 'results': results}
    with open(os.path.join(RESULTS_DIR, 'tau_cap_results.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fixes = list(FIXES.keys())

    for i, N in enumerate(N_VALUES):
        means = [results[N][f]['mean_sw_ratio'] for f in fixes]
        stds = [results[N][f]['std_sw_ratio'] for f in fixes]
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        bars = axes[0].bar(range(len(fixes)), means, yerr=stds, capsize=4,
                          color=colors, alpha=0.8)
        axes[0].set_xticks(range(len(fixes)))
        axes[0].set_xticklabels(fixes, rotation=30, ha='right')
        axes[0].set_ylabel('SW / Greedy Oracle')
        axes[0].set_title(f'N={N}')
        axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        for j, (m, s) in enumerate(zip(means, stds)):
            axes[0].text(j, m + s + 0.05, f'{m:.3f}', ha='center', fontsize=9)

    # Final SW
    for i, N in enumerate(N_VALUES):
        means = [results[N][f]['mean_sw_final'] for f in fixes]
        stds = [results[N][f]['std_sw_ratio'] for f in fixes]
        bars = axes[1].bar(range(len(fixes)), means, capsize=4,
                          color=colors, alpha=0.8)
        axes[1].set_xticks(range(len(fixes)))
        axes[1].set_xticklabels(fixes, rotation=30, ha='right')
        axes[1].set_ylabel('Final SW')
        axes[1].set_title(f'N={N} Final SW')

    # Tau values
    taus = [results[N]['Baseline']['mean_tau'] for N in N_VALUES] + \
           [results[N]['FixTau']['mean_tau'] for N in N_VALUES] + \
           [results[N]['FixTauFixC']['mean_tau'] for N in N_VALUES]
    labels = ['Baseline', 'FixTau', 'FixTauFixC']
    x = np.arange(len(labels))
    axes[2].bar(x, taus, color=colors, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha='right')
    axes[2].set_ylabel('Final tau')
    axes[2].set_title('tau at T=1000')
    axes[2].axhline(y=5.0, color='orange', linestyle='--', label='TAU_CAP=5')
    axes[2].legend()

    plt.suptitle(f'Tau Cap Test: TAU_CAP={TAU_CAP}, T={T}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'tau_cap_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {RESULTS_DIR}/tau_cap_results.json + .png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Final SW / Greedy Oracle")
    print("=" * 60)
    header = "Fix".ljust(15) + "".join(f"N={n:>10}" for n in N_VALUES)
    print(header)
    for f in fixes:
        row = f.ljust(15) + "".join(f"{results[n][f]['mean_sw_ratio']:>10.4f}" for n in N_VALUES)
        print(row)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
