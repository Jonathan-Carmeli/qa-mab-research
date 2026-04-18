"""
fix_experiments_v4.py
Round-4: now that we know QA-MAB Baseline beats NB3R at N>=20, drill into:

  Exp 1 — Crossover hunt. NB3R vs QA-MAB Baseline at N in {12,14,15,16,18}.
          Find the exact N where QA-MAB starts winning.

  Exp 2 — Interference weighting. At N=50, Oracle (true B,I) actually LOST
          to Baseline by 2.0. Hypothesis: at large N the optimal effective
          interference weight is > 1.0 (raw I underweights conflicts because
          the SA solver needs a sharper signal to commit). Sweep I_hat scale
          in {0.5, 1.0, 1.5, 2.0} on the Oracle.

  Exp 3 — Re-run Round-1/2 fixes at N=20. They all hurt at N=10, but at
          N=20 the framework is winning, so the fixes might compose:
            Fix A — u_hat learns raw B (add est. interference back in).
            Fix B — tau cap at 5.
            Fix A+B.

  Exp 4 — Long-horizon scaling. T=2000 at N=20. Does QA-MAB widen its lead
          over NB3R with more time, or does NB3R catch up?

Usage:
    python fix_experiments_v4.py
"""

import time
import numpy as np

from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


TAU_CAP = 5.0


# ------------------------- variants -------------------------

class QAMABOracleScaled(QAMAB):
    """Oracle (true B,I) with a scalar multiplier on I_hat in the QUBO."""

    def __init__(self, env, i_scale=1.0, **kwargs):
        super().__init__(env, **kwargs)
        self.u_hat = env.B.copy()
        self.I_hat = env.I.copy() * i_scale  # bake the scale in once

    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        self.tau += self.delta_tau
        self.history.append(self.env.social_welfare(assignment))


class QAMABFixed(QAMAB):
    """
    QA-MAB with composable Round-1/2 fixes:
      Fix A — u_hat targets raw B by adding back the estimated interference
              that hit agent i this step (sum over chosen j!=i of
              I_hat[i, k, j, k_j]). Without A, u_hat only ever learns the
              effective utility (B - E[interference]).
      Fix B — cap tau at TAU_CAP so the QUBO landscape never freezes solid.
    """

    def __init__(self, env, fixes=(), **kwargs):
        super().__init__(env, **kwargs)
        self.fixes = set(fixes)

    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        throughputs = self.env.compute_throughput(assignment)

        # u_hat update — Fix A adds back est. interference so target ~ B.
        for i in range(self.N):
            k = assignment[i]
            target = throughputs[i]
            if "A" in self.fixes:
                est_interference = 0.0
                for j in range(self.N):
                    if j == i:
                        continue
                    est_interference += self.I_hat[i, k, j, assignment[j]]
                target = throughputs[i] + est_interference
            self.u_hat[i, k] += self.B_learn_rate * (target - self.u_hat[i, k])

        # I_hat collision inference — unchanged from baseline.
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

        if "B" in self.fixes:
            self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        else:
            self.tau += self.delta_tau

        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}
        self.history.append(self.env.social_welfare(assignment))


# ------------------------- helpers -------------------------

def final_sw(history, last=50):
    return float(np.mean(history[-last:]))


def run_nb3r(N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    return NB3R(env, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed).run(T)


def run_qamab_baseline(N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    return QAMAB(env, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed).run(T)


def run_qamab_oracle_scaled(N, m, T, seed, i_scale):
    env = NetworkEnvironment(N, m, seed=seed)
    qa = QAMABOracleScaled(env, i_scale=i_scale, tau0=0.1, delta_tau=0.05,
                           lambda_=0.5, seed=seed)
    return qa.run(T)


def run_qamab_fixed(N, m, T, seed, fixes):
    env = NetworkEnvironment(N, m, seed=seed)
    qa = QAMABFixed(env, fixes=fixes, tau0=0.1, delta_tau=0.05,
                    lambda_=0.5, seed=seed)
    return qa.run(T)


def stats(label, finals):
    mean = float(np.mean(finals))
    std = float(np.std(finals))
    return label, mean, std


# ------------------------- experiments -------------------------

def exp1_crossover(m, T, n_runs):
    print("=" * 78)
    print("EXP 1 — Crossover hunt (NB3R vs QA-MAB Baseline)")
    print(f"  N in [12,14,15,16,18], m={m}, T={T}, runs={n_runs}")
    print("=" * 78)
    seeds = [42 + r for r in range(n_runs)]
    results = {}
    for N in [12, 14, 15, 16, 18]:
        t0 = time.time()
        nb3r_finals = [final_sw(run_nb3r(N, m, T, s)) for s in seeds]
        qa_finals = [final_sw(run_qamab_baseline(N, m, T, s)) for s in seeds]
        nb3r_m, qa_m = float(np.mean(nb3r_finals)), float(np.mean(qa_finals))
        delta = qa_m - nb3r_m
        results[N] = (nb3r_m, qa_m, delta)
        print(f"  N={N:>3}  NB3R={nb3r_m:>+8.3f}   QA-MAB={qa_m:>+8.3f}   "
              f"delta={delta:>+7.3f}   ({time.time()-t0:5.1f}s)")
    print()
    # Locate crossover
    cross = next((N for N in sorted(results) if results[N][2] > 0), None)
    if cross is None:
        print("  No crossover in [12,18] — QA-MAB still loses everywhere.")
    else:
        print(f"  Crossover at N >= {cross}")
    print()


def exp2_i_scale_oracle(m, T, n_runs):
    N = 50
    print("=" * 78)
    print(f"EXP 2 — I_hat scale sweep on Oracle  (N={N}, T={T}, runs={n_runs})")
    print("=" * 78)
    seeds = [42 + r for r in range(n_runs)]

    t0 = time.time()
    base_finals = [final_sw(run_qamab_baseline(N, m, T, s)) for s in seeds]
    base_mean = float(np.mean(base_finals))
    print(f"  Baseline (learned u_hat,I_hat)   {base_mean:>+8.3f}   "
          f"({time.time()-t0:5.1f}s)")

    for i_scale in [0.5, 1.0, 1.5, 2.0]:
        t0 = time.time()
        finals = [final_sw(run_qamab_oracle_scaled(N, m, T, s, i_scale))
                  for s in seeds]
        mean = float(np.mean(finals))
        std = float(np.std(finals))
        delta = mean - base_mean
        print(f"  Oracle, I_hat * {i_scale:>3.1f}              {mean:>+8.3f} "
              f"+/- {std:>5.2f}   delta_vs_base={delta:>+7.3f}   "
              f"({time.time()-t0:5.1f}s)")
    print()


def exp3_round12_fixes_at_n20(m, T, n_runs):
    N = 20
    print("=" * 78)
    print(f"EXP 3 — Re-run Round-1/2 fixes at N={N}  (T={T}, runs={n_runs})")
    print("=" * 78)
    seeds = [42 + r for r in range(n_runs)]

    t0 = time.time()
    nb3r_finals = [final_sw(run_nb3r(N, m, T, s)) for s in seeds]
    nb3r_mean = float(np.mean(nb3r_finals))
    print(f"  NB3R                  {nb3r_mean:>+8.3f}   ({time.time()-t0:5.1f}s)")

    variants = [
        ("Baseline (no fixes)", ()),
        ("Fix A (u_hat -> B)",  ("A",)),
        ("Fix B (tau cap 5)",   ("B",)),
        ("Fix A+B",             ("A", "B")),
    ]
    for label, fixes in variants:
        t0 = time.time()
        finals = [final_sw(run_qamab_fixed(N, m, T, s, fixes)) for s in seeds]
        mean = float(np.mean(finals))
        std = float(np.std(finals))
        delta = mean - nb3r_mean
        print(f"  {label:<22}{mean:>+8.3f} +/- {std:>5.2f}   "
              f"delta_vs_NB3R={delta:>+7.3f}   ({time.time()-t0:5.1f}s)")
    print()


def exp4_long_horizon(m, n_runs):
    N, T = 20, 2000
    print("=" * 78)
    print(f"EXP 4 — Long horizon  N={N}, T={T}, runs={n_runs}")
    print("=" * 78)
    seeds = [42 + r for r in range(n_runs)]

    # Compare final SW *and* check intermediate SW so we see the trajectory.
    checkpoints = [500, 1000, 1500, 2000]

    nb3r_traj = []
    qa_traj = []
    t0 = time.time()
    for s in seeds:
        h = run_nb3r(N, m, T, s)
        nb3r_traj.append(h)
    print(f"  NB3R total:   {time.time()-t0:5.1f}s")
    t0 = time.time()
    for s in seeds:
        h = run_qamab_baseline(N, m, T, s)
        qa_traj.append(h)
    print(f"  QA-MAB total: {time.time()-t0:5.1f}s\n")

    nb3r_traj = np.array(nb3r_traj)  # (runs, T)
    qa_traj = np.array(qa_traj)

    print(f"  {'checkpoint':<12}{'NB3R':>10}{'QA-MAB':>12}{'delta':>10}")
    for c in checkpoints:
        # mean SW over last 50 steps before checkpoint
        nb3r_c = float(np.mean(nb3r_traj[:, c - 50:c]))
        qa_c = float(np.mean(qa_traj[:, c - 50:c]))
        print(f"  T={c:<10}{nb3r_c:>+10.3f}{qa_c:>+12.3f}{qa_c - nb3r_c:>+10.3f}")
    print()


def main():
    m = 4
    print("Round-4 experiments\n")
    exp1_crossover(m=m, T=500, n_runs=10)
    exp2_i_scale_oracle(m=m, T=500, n_runs=5)
    exp3_round12_fixes_at_n20(m=m, T=500, n_runs=5)
    exp4_long_horizon(m=m, n_runs=5)


if __name__ == "__main__":
    main()
