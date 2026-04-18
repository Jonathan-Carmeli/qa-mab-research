"""
fix_experiments_v2.py
Round-2 systematic test of QA-MAB fixes after Round 1 showed every fix made
the algorithm WORSE than the baseline (0.98).

Key insight from Round 1:
  Baseline works *because* of an accidental "double-counting" of interference:
    u_hat learns observed throughput  (~ B - E[interference]) on diagonal
    I_hat learns marginal collisions  (added on off-diagonal)
  The QUBO penalises interference twice, which yields a strong-enough signal
  for SA to commit to a coherent assignment once tau grows large.

Round 1 fixes broke this because:
  Fix A: tried to learn raw B by adding I_hat back into u_hat -> circular
         feedback (u_hat inflates -> drop detection over-fires -> I_hat
         inflates -> u_hat inflates more).
  Fix B: tau cap kept the QUBO landscape soft so SA never committed.
  Fix D: zero-init I_hat made attribution weights default to 1/(N-1),
         shrinking the effective learn rate ~10x; I_hat never grew.

Round 2 explores fixes that *respect* the working baseline rather than
re-deriving u_hat:

  Fix G: scale I_hat by 0.5 in the QUBO (test: is double-counting actually
         too strong? if so, halving I_hat should help).
  Fix I: epsilon-greedy on top of SA. Decouples exploration from tau, so
         the QUBO can sharpen while we still try random routes occasionally.
  Fix J: warm-start SA from the previous step's solution (one of the
         restarts). Should reduce per-step variance once a good basin
         is found.
  Fix L: drop I_hat entirely from the QUBO -> u_hat-only greedy. Pure
         ablation: "is the interference term helping at all?"

Test matrix:
  1. Baseline                  (reference, what we beat at 0.98)
  2. Fix G                     (I_hat scaled 0.5)
  3. Fix G + B                 (scaling + tau cap)
  4. Fix I                     (epsilon-greedy only)
  5. Fix J                     (warm-start SA only)
  6. Fix L                     (u_hat-only QUBO, no I_hat)
  7. Fix G + I + J             (scale + explore + warm-start)
  8. Fix B + I                 (tau cap + epsilon-greedy)

Usage:
    python fix_experiments_v2.py
"""

import numpy as np

from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


TAU_CAP = 5.0
I_HAT_SCALE_DEFAULT = 0.5
EPSILON0 = 0.3
EPSILON_MIN = 0.01
EPSILON_DECAY_HORIZON = 500  # steps over which epsilon decays linearly


class QAMABFixedV2(QAMAB):
    """
    QA-MAB variant that supports Round-2 fixes G, I, J, L (composable with B).

    All other learning logic (u_hat, I_hat, tau schedule) is inherited from
    QAMAB so the baseline is exactly the same as the working 0.98 variant.
    """

    def __init__(self, env, fixes=(), i_hat_scale=I_HAT_SCALE_DEFAULT,
                 epsilon0=EPSILON0, epsilon_min=EPSILON_MIN,
                 epsilon_horizon=EPSILON_DECAY_HORIZON, **kwargs):
        super().__init__(env, **kwargs)
        self.fixes = set(fixes)

        # Fix G: scale factor applied to I_hat when building QUBO off-diagonal.
        self.i_hat_scale = i_hat_scale if "G" in self.fixes else 1.0

        # Fix I: epsilon-greedy on top of the QUBO solution.
        self.epsilon = epsilon0
        self.epsilon_min = epsilon_min
        self.epsilon_step = max(0.0, (epsilon0 - epsilon_min) / max(1, epsilon_horizon))

        # Fix J: warm-start SA from previous assignment.
        self._last_assignment = None

    # -------- QUBO build with Fix G (scale) and Fix L (drop I_hat) --------
    def build_qubo(self):
        size = self.qubo_size
        Q = np.zeros((size, size))

        # Diagonal (-u_hat - lambda/2) and same-agent off-diagonal (lambda/2)
        for i in range(self.N):
            for k in range(self.m):
                idx_ik = self._idx(i, k)
                Q[idx_ik, idx_ik] = -self.u_hat[i, k] - self.lambda_ / 2.0
                for l in range(k + 1, self.m):
                    idx_il = self._idx(i, l)
                    Q[idx_ik, idx_il] = self.lambda_ / 2.0
                    Q[idx_il, idx_ik] = self.lambda_ / 2.0

        # Cross-agent interference (Fix L skips it; Fix G rescales it).
        if "L" not in self.fixes:
            scale = self.i_hat_scale
            for i in range(self.N):
                for k in range(self.m):
                    idx_ik = self._idx(i, k)
                    for j in range(self.N):
                        if j == i:
                            continue
                        for l in range(self.m):
                            idx_jl = self._idx(j, l)
                            Q[idx_ik, idx_jl] = scale * self.I_hat[i, k, j, l]

        return self.tau * Q

    # -------- SA solver with Fix J (warm-start one restart) --------
    def solve_qubo(self, Q):
        n = self.N
        m = self.m
        size = self.qubo_size

        if n <= 10:
            n_restarts, n_iters = 8, 500
        elif n <= 20:
            n_restarts, n_iters = 4, 200
        elif n <= 30:
            n_restarts, n_iters = 2, 100
        else:
            n_restarts, n_iters = 2, 80

        T0 = 10.0
        decay = 0.9

        best_x = None
        best_energy = float('inf')

        warm = ("J" in self.fixes) and (self._last_assignment is not None)

        for restart in range(n_restarts):
            x = np.zeros(size, dtype=float)

            if restart == 0 and warm:
                # Fix J: start the first restart from the previous solution.
                for i in range(n):
                    x[i * m + self._last_assignment[i]] = 1.0
            else:
                # Standard u_hat-greedy start.
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
            for _ in range(n_iters):
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

    # -------- Step: identical learning rules to baseline + Fix I + Fix B --------
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)

        # Fix I: epsilon-greedy override. We override the QUBO solution with a
        # uniformly random assignment with probability epsilon. Learning still
        # happens on the *executed* assignment, so I_hat will be informed by
        # exploration too.
        if "I" in self.fixes and self.rng.random() < self.epsilon:
            for i in range(self.N):
                assignment[i] = int(self.rng.integers(0, self.m))

        throughputs = self.env.compute_throughput(assignment)

        # u_hat update — same as baseline (learns observed = B - actual_interference).
        for i in range(self.N):
            k = assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (throughputs[i] - self.u_hat[i, k])

        # I_hat collision inference — same as baseline.
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

        # tau update (Fix B caps it).
        if "B" in self.fixes:
            self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        else:
            self.tau += self.delta_tau

        # epsilon decay for Fix I.
        if "I" in self.fixes:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        self._last_assignment = dict(assignment)
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}

        sw = self.env.social_welfare(assignment)
        self.history.append(sw)


VARIANTS = [
    ("Baseline (no fixes)",   ()),
    ("Fix G (I_hat * 0.5)",   ("G",)),
    ("Fix G + B",             ("G", "B")),
    ("Fix I (epsilon-greedy)", ("I",)),
    ("Fix J (warm-start)",    ("J",)),
    ("Fix L (no I_hat)",      ("L",)),
    ("Fix G + I + J",         ("G", "I", "J")),
    ("Fix B + I",             ("B", "I")),
]


def final_sw(history, last=50):
    return float(np.mean(history[-last:]))


def run_nb3r(N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    nb3r = NB3R(env, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed)
    return nb3r.run(T)


def run_qamab(fixes, N, m, T, seed):
    env = NetworkEnvironment(N, m, seed=seed)
    qa = QAMABFixedV2(env, fixes=fixes, tau0=0.1, delta_tau=0.05,
                      lambda_=0.5, seed=seed,
                      epsilon_horizon=T)
    return qa.run(T)


def main():
    N, m, T, n_runs = 10, 4, 500, 10
    seeds = [42 + r for r in range(n_runs)]

    print(f"Round-2 fix experiments: N={N}, m={m}, T={T}, runs={n_runs}")
    print("Reporting mean SW over last 50 steps.\n")

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

    print("=" * 78)
    print(f"{'Variant':<32} {'Mean SW':>10} {'Std':>10} {'Delta vs NB3R':>18}")
    print("-" * 78)
    print(f"{'NB3R':<32} {nb3r_mean:>10.4f} {nb3r_std:>10.4f} {0.0:>+18.4f}")
    for name, mean, std, delta in results:
        print(f"{name:<32} {mean:>10.4f} {std:>10.4f} {delta:>+18.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
