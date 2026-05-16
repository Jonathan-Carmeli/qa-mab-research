"""
fix_verification.py — Test all QA-MAB fixes systematically

Tests the following fixes on N={10,15,20}, T=2000, 30 seeds:
1. Baseline (current QA-MAB)
2. Fix A: u_hat targets B by adding back estimated interference
3. Fix B: I_hat capped at I_cap + EMA decay
4. Fix C: u_hat only (no I_hat updates)
5. Fix D: Better SA (SA-medium: 50 restarts × 200 iterations)
6. Fix A+B: Combined
7. Fix A+C: Combined

Metrics:
- Final SW vs true optimum (brute force for N<=8, greedy upper bound for N>8)
- Learning curves for u_hat error and I_hat error
- Convergence speed: at what t does SW stabilize?
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
from itertools import product

sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'fix_verification')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_VALUES = [10, 15, 20]
T = 2000
N_SEEDS = 30
BASE_SEED = 2026
I_CAP = 0.3

# Oracle computation for N<=8
def brute_force_oracle(env):
    """Compute true optimum via brute force for small N."""
    best_sw = -np.inf
    best_assignment = None
    routes = list(range(env.m))
    for combo in product(routes, repeat=env.N):
        assignment = {i: combo[i] for i in range(env.N)}
        sw = env.social_welfare(assignment)
        if sw > best_sw:
            best_sw = sw
            best_assignment = assignment
    return best_sw, best_assignment

def greedy_oracle(env):
    """Greedy upper bound: each agent picks their best route ignoring interference."""
    assignment = {i: int(np.argmax(env.B[i])) for i in range(env.N)}
    return float(np.sum(np.max(env.B, axis=1))), assignment

# -------------------- Fix implementations --------------------

class QAMABBaseline(QAMAB):
    """Current QA-MAB — no fixes."""
    pass

class QAMABFixA(QAMAB):
    """
    Fix A: u_hat targets B (not B-E[I]) by adding back estimated interference.
    target = observed_throughput + sum_j I_hat[i,k,j,kj]
    This way u_hat learns the base utility, not the effective utility.
    """
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        throughputs = self.env.compute_throughput(assignment)

        # Fix A: target = observed + estimated_interference
        for i in range(self.N):
            k = assignment[i]
            est_interference = 0.0
            for j in range(self.N):
                if j == i:
                    continue
                est_interference += self.I_hat[i, k, j, assignment[j]]
            target = throughputs[i] + est_interference
            self.u_hat[i, k] += self.B_learn_rate * (target - self.u_hat[i, k])

        # Collision inference (unchanged)
        if self._prev_x is not None and self._prev_throughputs is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ki = self._prev_x[i]
                    kj = self._prev_x[j]
                    drop_i = max(0.0, self.u_hat[i, ki] - self._prev_throughputs[i])
                    drop_j = max(0.0, self.u_hat[j, kj] - self._prev_throughputs[j])
                    if drop_i > self.collision_threshold:
                        self.I_hat[i, ki, j, kj] = min(
                            self.I_hat[i, ki, j, kj] + self.I_learn_rate, self.I_cap)
                    if drop_j > self.collision_threshold:
                        self.I_hat[j, kj, i, ki] = min(
                            self.I_hat[j, kj, i, ki] + self.I_learn_rate, self.I_cap)

        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}
        self.tau += self.delta_tau
        sw = self.env.social_welfare(assignment)
        self.history.append(sw)

class QAMABFixB(QAMAB):
    """
    Fix B: I_hat with EMA decay — old interference estimates fade over time.
    Instead of unbounded growth, I_hat[i,k,j,l] *= (1-decay) when no collision,
    and updates with collision signal.
    """
    def __init__(self, env, **kwargs):
        self.decay_rate = kwargs.pop('I_decay', 0.01)
        super().__init__(env, **kwargs)

    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        throughputs = self.env.compute_throughput(assignment)

        # Update u_hat (standard)
        for i in range(self.N):
            k = assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (throughputs[i] - self.u_hat[i, k])

        # EMA decay: fade all I_hat entries
        self.I_hat *= (1.0 - self.decay_rate)

        # Collision inference with EMA
        if self._prev_x is not None and self._prev_throughputs is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ki = self._prev_x[i]
                    kj = self._prev_x[j]
                    drop_i = max(0.0, self.u_hat[i, ki] - self._prev_throughputs[i])
                    drop_j = max(0.0, self.u_hat[j, kj] - self._prev_throughputs[j])
                    if drop_i > self.collision_threshold:
                        self.I_hat[i, ki, j, kj] = min(
                            self.I_hat[i, ki, j, kj] + self.I_learn_rate, self.I_cap)
                    if drop_j > self.collision_threshold:
                        self.I_hat[j, kj, i, ki] = min(
                            self.I_hat[j, kj, i, ki] + self.I_learn_rate, self.I_cap)

        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}
        self.tau += self.delta_tau
        sw = self.env.social_welfare(assignment)
        self.history.append(sw)

class QAMABFixC(QAMAB):
    """
    Fix C: u_hat only — disable I_hat updates entirely.
    Test whether I_hat is actually helping or hurting.
    """
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
        sw = self.env.social_welfare(assignment)
        self.history.append(sw)

class QAMABFixDMixin:
    """Mixin for SA-medium solver."""
    def solve_qubo(self, Q):
        n = self.N
        m = self.m
        size = self.qubo_size
        n_restarts = 50
        n_iters = 200
        T0 = 2.0
        decay = 0.95

        best_x = None
        best_energy = float('inf')

        for restart in range(n_restarts):
            x = np.zeros(size, dtype=float)
            for i in range(n):
                k_greedy = int(np.argmax(self.u_hat[i]))
                x[i * m + k_greedy] = 1.0
            if restart > 0:
                n_flips = self.rng.integers(1, max(2, n // 3))
                for _ in range(n_flips):
                    i = self.rng.integers(0, n)
                    block = x[i * m:(i + 1) * m]
                    k_old = int(np.argmax(block))
                    candidates = [k for k in range(m) if k != k_old]
                    if candidates:
                        k_new = candidates[self.rng.integers(0, len(candidates))]
                        x[i * m + k_old] = 0.0
                        x[i * m + k_new] = 1.0

            energy = self._qubo_energy(x, Q)
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()

            T = T0 * (1.0 + restart * 0.3)
            for step in range(n_iters):
                T *= decay
                i = self.rng.integers(0, n)
                block = x[i * m:(i + 1) * m]
                k_old = int(np.argmax(block))
                k_new = (k_old + 1 + self.rng.integers(0, m - 1)) % m
                x[i * m + k_old] = 0.0
                x[i * m + k_new] = 1.0
                new_energy = self._qubo_energy(x, Q)
                delta = new_energy - energy
                if delta < 0 or (T > 1e-10 and self.rng.random() < np.exp(-delta / T)):
                    energy = new_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_x = x.copy()
                else:
                    x[i * m + k_new] = 0.0
                    x[i * m + k_old] = 1.0

        assignment = {}
        for i in range(n):
            block = best_x[i * m:(i + 1) * m]
            assignment[i] = int(np.argmax(block))
        return assignment

class QAMABFixD(QAMABFixDMixin, QAMAB):
    """Fix D: Better SA (SA-medium: 50×200 instead of 8×15)."""
    pass

class QAMABFixAB(QAMABFixDMixin, QAMABFixA):
    """Fix A+B: u_hat targets B + I_hat EMA decay + better SA."""
    pass

class QAMABFixAC(QAMABFixDMixin, QAMABFixA):
    """Fix A+C: u_hat targets B + no I_hat + better SA."""
    pass


FIXES = {
    'Baseline': QAMABBaseline,
    'FixA': QAMABFixA,
    'FixB': QAMABFixB,
    'FixC': QAMABFixC,
    'FixD': QAMABFixD,
    'FixAB': QAMABFixAB,
    'FixAC': QAMABFixAC,
}


def run_fix_comparison():
    print("=" * 70)
    print("FIX VERIFICATION: QA-MAB Learning Fixes")
    print(f"N={N_VALUES}, T={T}, seeds={N_SEEDS}")
    print("=" * 70)

    results = {}

    for N in N_VALUES:
        print(f"\n--- N={N} ---")
        results[N] = {}

        for fix_name, FixCls in FIXES.items():
            print(f"  Running {fix_name}...", end=' ', flush=True)
            sws = []
            u_hat_errors = []
            i_hat_errors = []
            u_hat_final = []
            times = []

            for seed_idx in range(N_SEEDS):
                seed = BASE_SEED + seed_idx * 1000 + N
                env = NetworkEnvironment(N, m=4, seed=seed,
                                        B_scale='uniform', I_scale='moderate')

                # Compute oracle
                if N <= 8:
                    opt_sw, _ = brute_force_oracle(env)
                else:
                    opt_sw, _ = greedy_oracle(env)

                # Run fix
                algo = FixCls(env, tau0=0.1, delta_tau=0.05, lambda_=2.0,
                              B_learn_rate=0.2, I_learn_rate=0.05, I_cap=I_CAP,
                              seed=seed)

                # Track error at checkpoints
                checkpoints = [100, 500, 1000, 1500, 2000]
                u_hat_errs = []
                i_hat_errs = []

                for t in range(T):
                    algo.step()

                    if t + 1 in checkpoints:
                        # Frobenius error
                        u_err = float(np.linalg.norm(algo.u_hat - env.B))
                        i_err = float(np.linalg.norm(algo.I_hat - env.I))
                        u_hat_errs.append(u_err)
                        i_hat_errs.append(i_err)

                final_sw = algo.history[-1]
                ratio = final_sw / opt_sw if opt_sw != 0 else 0
                sws.append(ratio)
                u_hat_errors.append(u_hat_errs)
                i_hat_errors.append(i_hat_errs)
                u_hat_final.append(float(np.linalg.norm(algo.u_hat - env.B)))

            sws = np.array(sws)
            u_hat_errors = np.array(u_hat_errors)
            i_hat_errors = np.array(i_hat_errors)

            mean_sw = float(np.mean(sws))
            std_sw = float(np.std(sws))
            mean_u_err = float(np.mean(u_hat_final))
            mean_u_err_by_t = [float(np.mean(u_hat_errors[:, i])) for i in range(len(checkpoints))]
            mean_i_err_by_t = [float(np.mean(i_hat_errors[:, i])) for i in range(len(checkpoints))]

            results[N][fix_name] = {
                'mean_sw_ratio': mean_sw,
                'std_sw_ratio': std_sw,
                'mean_u_hat_final_error': mean_u_err,
                'u_hat_errors_by_t': mean_u_err_by_t,
                'i_hat_errors_by_t': mean_i_err_by_t,
                'checkpoints': checkpoints,
                'opt_type': 'brute_force' if N <= 8 else 'greedy',
            }

            print(f"SW_ratio={mean_sw:.4f}±{std_sw:.4f}  u_err={mean_u_err:.2f}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'fix_comparison.json')
    with open(out_path, 'w') as f:
        json.dump({'config': {'N': N_VALUES, 'T': T, 'seeds': N_SEEDS},
                   'results': results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Generate comparison plots
    generate_plots(results)

    return results


def generate_plots(results):
    """Generate comparison plots for all fixes."""
    fixes = list(FIXES.keys())
    n_fixes = len(fixes)

    # Plot 1: SW ratio comparison by N
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, N in enumerate(N_VALUES):
        means = [results[N][f]['mean_sw_ratio'] for f in fixes]
        stds = [results[N][f]['std_sw_ratio'] for f in fixes]
        bars = axes[i].bar(range(n_fixes), means, yerr=stds, capsize=3,
                          color=['#e74c3c' if 'Baseline' in f else '#2ecc71' for f in fixes],
                          alpha=0.8)
        axes[i].set_xticks(range(n_fixes))
        axes[i].set_xticklabels(fixes, rotation=45, ha='right')
        axes[i].set_ylabel('SW / Optimum')
        axes[i].set_title(f'N={N}')
        axes[i].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        for bar, m in zip(bars, means):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('QA-MAB Fix Comparison: SW Ratio vs True Optimum', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fix_sw_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Learning curves for u_hat and I_hat error
    checkpoints = results[N_VALUES[0]]['Baseline']['checkpoints']
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    for fix_name in fixes:
        u_errs = [np.mean([results[N][fix_name]['u_hat_errors_by_t'][t]
                          for N in N_VALUES]) for t in range(len(checkpoints))]
        i_errs = [np.mean([results[N][fix_name]['i_hat_errors_by_t'][t]
                          for N in N_VALUES]) for t in range(len(checkpoints))]
        style = '--' if fix_name != 'Baseline' else '-'
        axes2[0].plot(checkpoints, u_errs, style, label=fix_name, linewidth=2)
        axes2[1].plot(checkpoints, i_errs, style, label=fix_name, linewidth=2)

    axes2[0].set_xlabel('Timestep')
    axes2[0].set_ylabel('||u_hat - B||_F')
    axes2[0].set_title('u_hat Learning Error (mean over N)')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    axes2[1].set_xlabel('Timestep')
    axes2[1].set_ylabel('||I_hat - I||_F')
    axes2[1].set_title('I_hat Learning Error (mean over N)')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    plt.suptitle('Learning Curves: u_hat and I_hat Error vs Time', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fix_learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {RESULTS_DIR}")


if __name__ == '__main__':
    start = time.time()
    results = run_fix_comparison()
    elapsed = time.time() - start
    print(f"\nTotal: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Final SW / Optimum")
    print("=" * 70)
    header = "Fix".ljust(12) + "".join(f"N={n:>8}" for n in N_VALUES)
    print(header)
    print("-" * 70)
    for fix in FIXES.keys():
        row = fix.ljust(12) + "".join(f"{results[n][fix]['mean_sw_ratio']:>8.4f}" for n in N_VALUES)
        print(row)
