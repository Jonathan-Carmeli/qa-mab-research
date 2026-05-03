"""
clean_split_qubo.py — Phase A & B implementation

Phase A (USE_ORACLE_I=True):
  - env.I for off-diagonal (ground truth interference in QUBO)
  - B_observed = tp + Σ env.I[i,k,j,route_j]  (no learning noise)
  → Prove: does Clean Split QUBO work when data is correct?

Phase B (USE_ORACLE_I=False):
  - I_hat learned via collision inference
  - B_observed = tp + Σ I_hat[i,k,j,route_j]
  - Update B_hat and I_hat via prediction error
  → Real algorithm: learns B and I jointly

Key fixes from analysis:
  1. QUBO diagonal: -B_hat ONLY (not u_hat = B - E[I])
  2. QUBO off-diagonal: +I_hat (interference, not double-counted)
  3. B_observed = tp + sum of actual/predicted interference
  4. UCB bonus in diagonal: -B_hat + c/sqrt(visits)
  5. Log cooling: T(t) ∝ 1/log(t) instead of tau increasing
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10
M = 4
T = 500
N_SEEDS = 20

# Hyperparams
B_LR = 0.2       # B_hat learning rate
I_LR = 0.05      # I_hat learning rate
I_CAP = 0.3      # I_hat max
UCB_C = 0.5      # UCB exploration constant
COLLISION_THRESH = 0.02  # threshold for inferring collision

# Log cooling: T_start / log(t + 1)
T_START = 2.0   # initial temperature for SA


class CleanSplitQA:
    """QA-MAB with Clean Split QUBO + UCB + log cooling"""

    def __init__(self, env, use_oracle_I=True, seed=42):
        self.env = env
        self.N = env.N
        self.m = env.m
        self.use_oracle_I = use_oracle_I
        self.rng = np.random.default_rng(seed)

        # Learn B only (not u_hat = B - interference)
        self.B_hat = np.full((self.N, self.m), 0.75)

        # Learn I separately
        self.I_hat = np.zeros((self.N, self.m, self.N, self.m))

        self.visits = np.zeros((self.N, self.m))

        # For collision inference
        self._prev_x = None
        self._prev_tp = None

        # Step counter (for UCB and log cooling)
        self.step_count = 0

    def idx(self, i, k):
        return i * self.m + k

    def build_qubo(self, t):
        """Build QUBO with Clean Split structure"""
        size = self.N * self.m
        Q = np.zeros((size, size))

        # === DIAGONAL: -B_hat + UCB bonus (no interference!) ===
        for i in range(self.N):
            for k in range(self.m):
                idx = self.idx(i, k)
                ucb_bonus = UCB_C / math.sqrt(self.visits[i, k] + 1)
                Q[idx, idx] = -self.B_hat[i, k] - ucb_bonus

        # === OFF-DIAGONAL: One-hot constraint (tau/2 per pair) ===
        tau = 1.0  # fixed penalty weight in QUBO (not growing)
        for i in range(self.N):
            for k in range(self.m):
                for l in range(k + 1, self.m):
                    idx_k = self.idx(i, k)
                    idx_l = self.idx(i, l)
                    Q[idx_k, idx_l] += tau / 2
                    Q[idx_l, idx_k] += tau / 2

        # === OFF-DIAGONAL: Interference term ===
        for i in range(self.N):
            for k in range(self.m):
                for j in range(self.N):
                    if j == i:
                        continue
                    for l in range(self.m):
                        if self.use_oracle_I:
                            I_val = self.env.I[i, k, j, l]
                        else:
                            I_val = self.I_hat[i, k, j, l]
                        idx_i = self.idx(i, k)
                        idx_j = self.idx(j, l)
                        Q[idx_i, idx_j] += I_val

        return Q

    def sa_solve(self, Q, t):
        """Simulated Annealing with log cooling T(t) = T_start / log(t+1)"""
        n = self.N
        m = self.m
        size = n * m

        # Log cooling temperature
        T = T_START / math.log(t + 2)  # +2 to avoid log(1)=0

        n_restarts = 8
        n_iters = 500

        best_x = None
        best_e = float('inf')

        for r in range(n_restarts):
            x = np.zeros(size)
            # Start from greedy B_hat
            for i in range(n):
                x[i * m + int(np.argmax(self.B_hat[i]))] = 1.0

            e = float(x @ Q @ x)
            if e < best_e:
                best_e = e
                best_x = x.copy()

            T_r = T * (1 + r * 0.3)  # slightly vary T per restart
            for _ in range(n_iters):
                T_r *= 0.95
                if T_r < 1e-12:
                    break

                i = self.rng.integers(0, n)
                block = x[i * m:(i + 1) * m]
                k_old = int(np.argmax(block))
                k_new = (k_old + 1 + self.rng.integers(0, m - 1)) % m

                x[i * m + k_old] = 0.0
                x[i * m + k_new] = 1.0
                ne = float(x @ Q @ x)
                d = ne - e

                if d < 0 or self.rng.random() < math.exp(-d / T_r):
                    e = ne
                    if e < best_e:
                        best_e = e
                        best_x = x.copy()
                else:
                    x[i * m + k_new] = 0.0
                    x[i * m + k_old] = 1.0

        # Decode
        assignment = {i: int(np.argmax(best_x[i * m:(i + 1) * m])) for i in range(n)}
        return assignment

    def step(self):
        t = self.step_count + 1  # 1-indexed for log cooling

        # Build QUBO
        Q = self.build_qubo(t)

        # Solve
        assignment = self.sa_solve(Q, t)

        # Measure
        tp_actual = self.env.compute_throughput(assignment)

        # === Compute B_observed ===
        # B_observed = tp + Σ I_actual (Phase A) or Σ I_hat (Phase B)
        for i in range(self.N):
            k = assignment[i]
            if self.use_oracle_I:
                I_sum = sum(self.env.I[i, k, j, assignment[j]]
                           for j in range(self.N) if j != i)
            else:
                I_sum = sum(self.I_hat[i, k, j, assignment[j]]
                           for j in range(self.N) if j != i)
            B_obs = tp_actual[i] + I_sum

            # Update B_hat
            self.B_hat[i, k] += B_LR * (B_obs - self.B_hat[i, k])

            # Track visits
            self.visits[i, k] += 1

        # === Learn I_hat (Phase B only) ===
        if not self.use_oracle_I and self._prev_x is not None:
            # Compare predicted vs actual tp to infer collisions
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ki = self._prev_x[i]
                    kj = self._prev_x[j]

                    # Predicted tp based on current B_hat (ignoring interference)
                    pred_i = self.B_hat[i, ki]
                    pred_j = self.B_hat[j, kj]

                    # Actual was lower → interference happened
                    drop_i = max(0.0, pred_i - self._prev_tp[i])
                    drop_j = max(0.0, pred_j - self._prev_tp[j])

                    if drop_i > COLLISION_THRESH:
                        self.I_hat[i, ki, j, kj] = min(
                            self.I_hat[i, ki, j, kj] + I_LR, I_CAP)
                    if drop_j > COLLISION_THRESH:
                        self.I_hat[j, kj, i, ki] = min(
                            self.I_hat[j, kj, i, ki] + I_LR, I_CAP)

        # Store for next collision inference
        self._prev_x = assignment.copy()
        self._prev_tp = {i: tp_actual[i] for i in range(self.N)}

        self.step_count += 1

        sw = self.env.social_welfare(assignment)
        return sw, tp_actual, assignment

    def run(self, T_steps):
        history = []
        for _ in range(T_steps):
            sw, _, _ = self.step()
            history.append(sw)
        return np.array(history)


def run_experiment(use_oracle_I, n_seeds, T_steps, N, M, label):
    """Run experiment and return results"""
    sw_ratios = []
    B_errors = []
    I_errors = [] if not use_oracle_I else None

    for si in range(n_seeds):
        seed = BASE_SEED + si * 1000 + N
        rng = np.random.default_rng(seed)

        env = NetworkEnvironment(N, M, seed=seed, B_scale='uniform', I_scale='moderate')
        algo = CleanSplitQA(env, use_oracle_I=use_oracle_I, seed=seed)

        # Compute true B mean for error calculation
        true_B_mean = float(np.mean(env.B))

        sw_history = algo.run(T_steps)

        # Social welfare ratio (vs greedy oracle)
        opt = float(np.sum(np.max(env.B, axis=1)))
        sw_ratio = float(sw_history[-50:].mean() / opt) if opt > 0 else 0
        sw_ratios.append(sw_ratio)

        # B_hat error (final 50 steps avg)
        B_err = float(np.mean(np.abs(algo.B_hat - env.B)[-50:]))
        B_errors.append(B_err)

        # I_hat error (Phase B only)
        if not use_oracle_I:
            I_err = float(np.mean(np.abs(algo.I_hat - env.I)[-50:]))
            I_errors.append(I_err)

    results = {
        'sw_ratio': float(np.mean(sw_ratios)),
        'sw_std': float(np.std(sw_ratios)),
        'B_error': float(np.mean(B_errors)),
        'B_error_std': float(np.std(B_errors)),
    }
    if not use_oracle_I:
        results['I_error'] = float(np.mean(I_errors))
        results['I_error_std'] = float(np.std(I_errors))

    return results


def main():
    print("=" * 65)
    print("CLEAN SPLIT QUBO — Phase A (Oracle) then Phase B (Learned)")
    print("=" * 65)
    print(f"N={N}, M={M}, T={T}, seeds={N_SEEDS}")
    print(f"B_LR={B_LR}, I_LR={I_LR}, UCB_C={UCB_C}")
    print(f"Log cooling: T(t) = {T_START} / log(t+2)")
    print()

    all_results = {}

    # === PHASE A: Oracle I (prove QUBO structure works) ===
    print("PHASE A: USE_ORACLE_I = True")
    print("  (env.I used in QUBO + B_observed, no learning noise)")
    print("-" * 50)
    rA = run_experiment(use_oracle_I=True, n_seeds=N_SEEDS,
                         T_steps=T, N=N, M=M, label='Oracle')
    all_results['phaseA_oracle'] = rA
    print(f"  SW_ratio: {rA['sw_ratio']:.4f} ± {rA['sw_std']:.4f}")
    print(f"  B_error:  {rA['B_error']:.4f} ± {rA['B_error_std']:.4f}")
    print()

    # === PHASE B: Learned I (real algorithm) ===
    print("PHASE B: USE_ORACLE_I = False")
    print("  (I_hat learned via collision inference, B_observed = tp + Σ I_hat)")
    print("-" * 50)
    rB = run_experiment(use_oracle_I=False, n_seeds=N_SEEDS,
                         T_steps=T, N=N, M=M, label='Learned')
    all_results['phaseB_learned'] = rB
    print(f"  SW_ratio: {rB['sw_ratio']:.4f} ± {rB['sw_std']:.4f}")
    print(f"  B_error:  {rB['B_error']:.4f} ± {rB['B_error_std']:.4f}")
    print(f"  I_error:  {rB['I_error']:.4f} ± {rB['I_error_std']:.4f}")
    print()

    # === SUMMARY ===
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Phase A (Oracle I):  SW={rA['sw_ratio']:.4f}  B_err={rA['B_error']:.4f}")
    print(f"Phase B (Learned I): SW={rB['sw_ratio']:.4f}  B_err={rB['B_error']:.4f}  I_err={rB['I_error']:.4f}")
    print()

    # Compare to baselines from previous experiments
    print("COMPARISON TO BASELINES:")
    print(f"  Previous best (UCB c=0.5): ~0.087")
    print(f"  Baseline (no fixes):     ~0.074")
    print(f"  Oracle (true B,I):        ~0.090 (from earlier)")
    print()

    # Phase B comparison
    delta = rB['sw_ratio'] - 0.087
    print(f"  Phase B vs previous best: {delta:+.4f}")

    all_results['meta'] = {
        'N': N, 'M': M, 'T': T, 'n_seeds': N_SEEDS,
        'B_LR': B_LR, 'I_LR': I_LR, 'UCB_C': UCB_C,
        'T_START': T_START
    }

    outpath = os.path.join(RESULTS, 'clean_split_results.json')
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {outpath}")


if __name__ == '__main__':
    main()