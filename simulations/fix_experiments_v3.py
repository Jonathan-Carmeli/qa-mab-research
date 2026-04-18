"""
fix_experiments_v3.py
Round-3: does the centralized framework *ever* help?

Rounds 1-2 conclusion: no learned-QA-MAB variant beat NB3R (1.03). Best we got
was baseline QA-MAB at 0.98. Analysis: NB3R has an information advantage —
W[i,k] directly estimates E[SW | agent i picks k], no decomposition. QA-MAB
must decompose into (u_hat, I_hat) which is strictly noisier to learn.

BUT: QA-MAB has one thing NB3R cannot do — joint optimization. NB3R is
independent softmax per agent (coordinate-ascent on local objectives). QA-MAB
solves a single global QUBO. At N=10 those coincide. At large N, coordinate
ascent should get stuck and NB3R's shared-signal coupling
(total_signal = U_i + sum_j U_j = SW) actively hurts: every agent's W[i, k]
converges toward the SAME scalar even though their B rows differ.

Experiments (N = 10, 20, 30, 50; T=500; 5 runs):

  1. NB3R                    — reference, total_signal = SW
  2. NB3R-BetterSignal       — total_signal = U_i only (per-agent signal)
  3. QA-MAB Baseline         — learns u_hat, I_hat as in qa_mab.py
  4. QA-MAB Oracle           — cheats: u_hat = env.B, I_hat = env.I (no learn)

Oracle isolates the optimization layer. If Oracle >> NB3R at large N, the
bottleneck is estimation. If Oracle ≈ NB3R, the bottleneck is the SA solver
or the framework itself. BetterSignal isolates NB3R's shared-signal weakness.

Usage:
    python fix_experiments_v3.py
"""

import time
import numpy as np

from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


class NB3RBetterSignal(NB3R):
    """NB3R variant: W update uses only own throughput, not SW."""

    def step(self):
        chosen = {i: self._pick_route(i) for i in range(self.N)}
        throughputs = self.env.compute_throughput(chosen)
        for i in range(self.N):
            k = chosen[i]
            self.W[i, k] = (1 - self.alpha) * self.W[i, k] + self.alpha * throughputs[i]
        self.tau += self.delta_tau
        self.history.append(self.env.social_welfare(chosen))


class QAMABOracle(QAMAB):
    """QA-MAB with perfect knowledge: u_hat = env.B, I_hat = env.I, no learning."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.u_hat = env.B.copy()
        self.I_hat = env.I.copy()

    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        # no u_hat / I_hat updates — oracle knows the truth
        self.tau += self.delta_tau
        self.history.append(self.env.social_welfare(assignment))


def final_sw(history, last=50):
    return float(np.mean(history[-last:]))


def run_nb3r(cls, N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    return cls(env, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed).run(T)


def run_qamab(cls, N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    return cls(env, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed).run(T)


VARIANTS = [
    ("NB3R",               "nb3r", NB3R),
    ("NB3R-BetterSignal",  "nb3r", NB3RBetterSignal),
    ("QA-MAB Baseline",    "qa",   QAMAB),
    ("QA-MAB Oracle",      "qa",   QAMABOracle),
]


def main():
    N_values = [10, 20, 30, 50]
    m, T, n_runs = 4, 500, 5
    seeds = [42 + r for r in range(n_runs)]

    print(f"Round-3 scaling experiments: m={m}, T={T}, runs={n_runs}")
    print(f"N values: {N_values}")
    print("Reporting mean SW over last 50 steps.\n")

    all_results = {}  # (N, variant_name) -> (mean, std)

    for N in N_values:
        print("=" * 78)
        print(f"N = {N}")
        print("=" * 78)
        for name, kind, cls in VARIANTS:
            t0 = time.time()
            finals = []
            for s in seeds:
                if kind == "nb3r":
                    hist = run_nb3r(cls, N, m, T, s)
                else:
                    hist = run_qamab(cls, N, m, T, s)
                finals.append(final_sw(hist))
            mean = float(np.mean(finals))
            std = float(np.std(finals))
            elapsed = time.time() - t0
            all_results[(N, name)] = (mean, std)
            print(f"  {name:<22} {mean:>8.3f} +/- {std:>6.3f}   ({elapsed:5.1f}s)")
        print()

    # Summary table
    print("=" * 78)
    print("SUMMARY — final mean SW by variant x N")
    print("=" * 78)
    header = f"{'Variant':<22}" + "".join(f"{'N=' + str(N):>12}" for N in N_values)
    print(header)
    print("-" * len(header))
    for name, _, _ in VARIANTS:
        row = f"{name:<22}"
        for N in N_values:
            mean, _ = all_results[(N, name)]
            row += f"{mean:>12.3f}"
        print(row)
    print()

    # Deltas vs NB3R
    print("DELTA vs NB3R (positive = better than NB3R)")
    print("-" * len(header))
    for name, _, _ in VARIANTS:
        if name == "NB3R":
            continue
        row = f"{name:<22}"
        for N in N_values:
            mean, _ = all_results[(N, name)]
            nb3r_mean, _ = all_results[(N, "NB3R")]
            row += f"{mean - nb3r_mean:>+12.3f}"
        print(row)
    print("=" * 78)


if __name__ == "__main__":
    main()
