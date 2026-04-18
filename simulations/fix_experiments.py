"""
fix_experiments.py
Systematic test of 6 candidate fixes for QA-MAB.

Each fix can be enabled independently via the `fixes` set. Runs the standard
benchmark (N=10, m=4, T=500, 10 runs) for NB3R + 8 QA-MAB variants and prints
a comparison table. Each variant inherits from QAMAB and only overrides step()
so the SA solver, QUBO build, and constructor stay identical.

Fixes:
  A: u_hat learns B by adding back estimated interference
       u_hat[i,k] += lr * (observed + sum_j I_hat[i,k,j,a[j]] - u_hat[i,k])
  B: tau cap at 5.0
  C: I_hat decay (* 0.999 each step)
  D: I_hat weighted attribution (split learn_rate by current I_hat share)
  E: lower I_cap = 0.2
  F: statistical threshold = max(0.02, 2 * std(recent throughputs))

Usage:
    python fix_experiments.py
"""

import numpy as np

from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


I_DECAY = 0.001
TAU_CAP = 5.0
WINDOW = 50
I_CAP_LOW = 0.2


class QAMABFixed(QAMAB):
    def __init__(self, env, fixes=(), **kwargs):
        if "E" in fixes and "I_cap" not in kwargs:
            kwargs["I_cap"] = I_CAP_LOW
        super().__init__(env, **kwargs)
        self.fixes = set(fixes)
        self.recent = {i: [] for i in range(self.N)}

    def _threshold_for(self, agent):
        if "F" not in self.fixes or len(self.recent[agent]) < 5:
            return self.collision_threshold
        return max(0.02, 2.0 * float(np.std(self.recent[agent])))

    def step(self):
        # 1. Build QUBO and solve
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)

        # 2. Measure throughput
        throughputs = self.env.compute_throughput(assignment)

        # Fix F: maintain sliding window of per-agent throughput
        for i in range(self.N):
            self.recent[i].append(throughputs[i])
            if len(self.recent[i]) > WINDOW:
                self.recent[i].pop(0)

        # 3a. Update u_hat (Fix A optionally adds back estimated interference)
        for i in range(self.N):
            k = assignment[i]
            observed = throughputs[i]
            if "A" in self.fixes:
                est_interference = 0.0
                for j in range(self.N):
                    if j == i:
                        continue
                    est_interference += self.I_hat[i, k, j, assignment[j]]
                target = observed + est_interference
            else:
                target = observed
            self.u_hat[i, k] += self.B_learn_rate * (target - self.u_hat[i, k])

        # 3b. Infer collisions, update I_hat
        if self._prev_x is not None and self._prev_throughputs is not None:
            drops = {}
            for i in range(self.N):
                ki = self._prev_x[i]
                drops[i] = max(0.0, self.u_hat[i, ki] - self._prev_throughputs[i])

            totals = None
            if "D" in self.fixes:
                totals = {}
                for i in range(self.N):
                    ki = self._prev_x[i]
                    s = 0.0
                    for j in range(self.N):
                        if j == i:
                            continue
                        s += self.I_hat[i, ki, j, self._prev_x[j]]
                    totals[i] = s

            for i in range(self.N):
                ki = self._prev_x[i]
                thr_i = self._threshold_for(i)
                for j in range(i + 1, self.N):
                    kj = self._prev_x[j]
                    thr_j = self._threshold_for(j)

                    if "D" in self.fixes:
                        if totals[i] > 1e-9:
                            w_ij = self.I_hat[i, ki, j, kj] / totals[i]
                        else:
                            w_ij = 1.0 / max(1, self.N - 1)
                        if totals[j] > 1e-9:
                            w_ji = self.I_hat[j, kj, i, ki] / totals[j]
                        else:
                            w_ji = 1.0 / max(1, self.N - 1)
                        inc_ij = self.I_learn_rate * w_ij
                        inc_ji = self.I_learn_rate * w_ji
                    else:
                        inc_ij = self.I_learn_rate
                        inc_ji = self.I_learn_rate

                    if drops[i] > thr_i:
                        self.I_hat[i, ki, j, kj] = min(
                            self.I_hat[i, ki, j, kj] + inc_ij, self.I_cap)
                    if drops[j] > thr_j:
                        self.I_hat[j, kj, i, ki] = min(
                            self.I_hat[j, kj, i, ki] + inc_ji, self.I_cap)

        # Fix C: decay I_hat to allow stale estimates to fade
        if "C" in self.fixes:
            self.I_hat *= (1.0 - I_DECAY)

        # 4. Increase tau (Fix B caps it)
        if "B" in self.fixes:
            self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        else:
            self.tau += self.delta_tau

        # 5. Store for next collision inference
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}

        # 6. Record social welfare
        sw = self.env.social_welfare(assignment)
        self.history.append(sw)


VARIANTS = [
    ("Baseline (no fixes)", ()),
    ("Fix A only",          ("A",)),
    ("Fix B only",          ("B",)),
    ("Fix C only",          ("C",)),
    ("Fix A+B",             ("A", "B")),
    ("Fix A+B+C",           ("A", "B", "C")),
    ("Fix A+B+C+D",         ("A", "B", "C", "D")),
    ("All fixes (A-F)",     ("A", "B", "C", "D", "E", "F")),
]


def final_sw(history, last=50):
    return float(np.mean(history[-last:]))


def run_nb3r(N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    nb3r = NB3R(env, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed)
    return nb3r.run(T)


def run_qamab(fixes, N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    qa = QAMABFixed(env, fixes=fixes, tau0=0.1, delta_tau=0.05,
                    lambda_=0.5, seed=seed)
    return qa.run(T)


def main():
    N, m, T, n_runs = 10, 4, 500, 10
    seeds = [42 + r for r in range(n_runs)]

    print(f"Fix experiments: N={N}, m={m}, T={T}, runs={n_runs}")
    print(f"Reporting mean SW over last 50 steps.\n")

    print("Running NB3R baseline...")
    nb3r_finals = [final_sw(run_nb3r(N, m, T, s)) for s in seeds]
    nb3r_mean = float(np.mean(nb3r_finals))
    nb3r_std = float(np.std(nb3r_finals))
    print(f"  NB3R: {nb3r_mean:.4f} +/- {nb3r_std:.4f}\n")

    results = []
    for name, fixes in VARIANTS:
        print(f"Running: {name}  fixes={fixes}")
        finals = [final_sw(run_qamab(fixes, N, m, T, s)) for s in seeds]
        mean = float(np.mean(finals))
        std = float(np.std(finals))
        delta = mean - nb3r_mean
        results.append((name, mean, std, delta))
        print(f"  -> {mean:.4f} +/- {std:.4f}  (delta vs NB3R: {delta:+.4f})\n")

    print("=" * 74)
    print(f"{'Variant':<28} {'Mean SW':>10} {'Std':>10} {'Delta vs NB3R':>18}")
    print("-" * 74)
    print(f"{'NB3R':<28} {nb3r_mean:>10.4f} {nb3r_std:>10.4f} {0.0:>+18.4f}")
    for name, mean, std, delta in results:
        print(f"{name:<28} {mean:>10.4f} {std:>10.4f} {delta:>+18.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
