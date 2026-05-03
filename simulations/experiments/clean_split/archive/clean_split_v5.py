"""
clean_split_v5.py — Break the B_hat illusion trap

Key changes from V4:
1. B_observed = tp ONLY (no I_hat addition) — removes inflation source
2. Route-unavailability bonus: if route not visited in K steps, QUBO diagonal bonus
3. Conservative I_hat initialization (starts near 0)
4. UCB-greedy baseline for comparison (isolates whether SA/QUBO is the bottleneck)
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20
B_LR = 0.15
I_LR = 0.05; I_CAP = 0.3
UCB_C = 0.5; T_START = 2.0
ROUTE_UNUSED_BONUS = 0.3
ROUTE_UNUSED_K = 50
EPS_GREEDY = 0.05


class AlgoV5:
    def __init__(self, env, use_oracle_I=True, seed=42):
        self.env = env; self.N = env.N; self.m = env.m
        self.use_oracle_I = use_oracle_I
        self.rng = np.random.default_rng(seed)
        self.B_hat = np.full((self.N, self.m), 1.0)
        self.I_hat = np.full((self.N, self.m, self.N, self.m), 0.01)
        np.fill_diagonal(self.I_hat.reshape(self.N*self.m, self.N*self.m), 0.0)
        for i in range(self.N): self.I_hat[i,:,i,:] = 0.0
        self.visits = np.zeros((self.N, self.m))
        self.last_visit = np.zeros((self.N, self.m))
        self._prev_x = None; self._prev_tp = None; self.step_count = 0

    def idx(self, i, k): return i * self.m + k

    def build_qubo(self, t):
        size = self.N * self.m; Q = np.zeros((size, size))
        for i in range(self.N):
            for k in range(self.m):
                idx = self.idx(i, k)
                ucb = UCB_C / math.sqrt(self.visits[i, k] + 1)
                Q[idx, idx] = -self.B_hat[i, k] - ucb
                steps_since = t - self.last_visit[i, k]
                if steps_since > ROUTE_UNUSED_K:
                    Q[idx, idx] += ROUTE_UNUSED_BONUS
        tau = 1.0
        for i in range(self.N):
            for k in range(self.m):
                for l in range(k + 1, self.m):
                    Q[self.idx(i, k), self.idx(i, l)] += tau / 2
                    Q[self.idx(i, l), self.idx(i, k)] += tau / 2
        for i in range(self.N):
            for k in range(self.m):
                for j in range(self.N):
                    if j == i: continue
                    for l in range(self.m):
                        I_val = self.env.I[i, k, j, l] if self.use_oracle_I else self.I_hat[i, k, j, l]
                        Q[self.idx(i, k), self.idx(j, l)] += I_val
        return Q

    def sa_solve(self, Q, t):
        n, m, size = self.N, self.m, self.N * self.m
        T = T_START / math.log(t + 2)
        best_x, best_e = None, float('inf')
        for r in range(8):
            x = np.zeros(size)
            if self.rng.random() < EPS_GREEDY:
                for i in range(n): x[i * m + self.rng.integers(0, m)] = 1.0
            else:
                for i in range(n): x[i * m + int(np.argmax(self.B_hat[i]))] = 1.0
            e = float(x @ Q @ x)
            if e < best_e: best_e, best_x = e, x.copy()
            T_r = T * (1 + r * 0.3)
            for _ in range(500):
                T_r *= 0.95
                if T_r < 1e-12: break
                i = self.rng.integers(0, n)
                block = x[i * m:(i + 1) * m]
                k_old = int(np.argmax(block))
                k_new = (k_old + 1 + self.rng.integers(0, m - 1)) % m
                x[i * m + k_old] = 0.0; x[i * m + k_new] = 1.0
                ne = float(x @ Q @ x)
                if ne < e or self.rng.random() < math.exp(-(ne - e) / T_r):
                    e = ne
                    if e < best_e: best_e, best_x = e, x.copy()
                else:
                    x[i * m + k_new] = 0.0; x[i * m + k_old] = 1.0
        return {i: int(np.argmax(best_x[i * m:(i + 1) * m])) for i in range(n)}

    def step(self):
        t = self.step_count + 1
        Q = self.build_qubo(t)
        assignment = self.sa_solve(Q, t)
        tp_actual = self.env.compute_throughput(assignment)

        if self.use_oracle_I:
            for i in range(self.N):
                k = assignment[i]
                I_sum = sum(self.env.I[i, k, j, assignment[j]] for j in range(self.N) if j != i)
                B_obs = tp_actual[i] + I_sum
                self.B_hat[i, k] += B_LR * (B_obs - self.B_hat[i, k])
                self.visits[i, k] += 1; self.last_visit[i, k] = t
        else:
            for i in range(self.N):
                k = assignment[i]
                B_obs = tp_actual[i]
                self.B_hat[i, k] += B_LR * (B_obs - self.B_hat[i, k])
                self.visits[i, k] += 1; self.last_visit[i, k] = t
            if self._prev_x is not None:
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        ki = self._prev_x[i]; kj = self._prev_x[j]
                        drop_i = max(0.0, self.B_hat[i, ki] - self._prev_tp[i])
                        drop_j = max(0.0, self.B_hat[j, kj] - self._prev_tp[j])
                        if drop_i > 0.02: self.I_hat[i, ki, j, kj] = min(self.I_hat[i, ki, j, kj] + I_LR, I_CAP)
                        if drop_j > 0.02: self.I_hat[j, kj, i, ki] = min(self.I_hat[j, kj, i, ki] + I_LR, I_CAP)
            self.I_hat *= (1.0 - 0.001)
            for i in range(self.N): self.I_hat[i,:,i,:] = 0.0

        self._prev_x = assignment.copy()
        self._prev_tp = {i: tp_actual[i] for i in range(self.N)}
        self.step_count += 1
        return float(sum(tp_actual.values()))

    def run(self, T_steps):
        return [self.step() for _ in range(T_steps)]


class UCBGreedyBaseline:
    """UCB-greedy: each agent picks argmax(B_hat + UCB), no QUBO, no SA.
    Purpose: isolate whether bottleneck is SA/QUBO layer or learning itself."""
    def __init__(self, env, seed=42):
        self.env = env; self.N = env.N; self.m = env.m
        self.rng = np.random.default_rng(seed)
        self.B_hat = np.full((self.N, self.m), 1.0)
        self.I_hat = np.full((self.N, self.m, self.N, self.m), 0.01)
        for i in range(self.N): self.I_hat[i,:,i,:] = 0.0
        self.visits = np.zeros((self.N, self.m))
        self._prev_x = None; self._prev_tp = None; self.step_count = 0

    def step(self):
        t = self.step_count + 1
        assignment = {}
        for i in range(self.N):
            ucb_vals = UCB_C / np.sqrt(self.visits[i] + 1)
            scores = self.B_hat[i] + ucb_vals
            assignment[i] = int(np.argmax(scores))
        tp_actual = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            B_obs = tp_actual[i]
            self.B_hat[i, k] += B_LR * (B_obs - self.B_hat[i, k])
            self.visits[i, k] += 1
        if self._prev_x is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ki = self._prev_x[i]; kj = self._prev_x[j]
                    drop_i = max(0.0, self.B_hat[i, ki] - self._prev_tp[i])
                    drop_j = max(0.0, self.B_hat[j, kj] - self._prev_tp[j])
                    if drop_i > 0.02: self.I_hat[i, ki, j, kj] = min(self.I_hat[i, ki, j, kj] + I_LR, I_CAP)
                    if drop_j > 0.02: self.I_hat[j, kj, i, ki] = min(self.I_hat[j, kj, i, ki] + I_LR, I_CAP)
        self.I_hat *= (1.0 - 0.001)
        for i in range(self.N): self.I_hat[i,:,i,:] = 0.0
        self._prev_x = assignment.copy()
        self._prev_tp = {i: tp_actual[i] for i in range(self.N)}
        self.step_count += 1
        return float(sum(tp_actual.values()))

    def run(self, T_steps):
        return [self.step() for _ in range(T_steps)]


def run_exp(use_oracle_I, n_seeds, T_steps):
    sw_ratios, B_errors, I_errors = [], [], []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        algo = AlgoV5(env, use_oracle_I=use_oracle_I, seed=si * 1000 + N)
        sw_hist = algo.run(T_steps)
        opt = float(np.sum(np.max(env.B, axis=1)))
        sw_ratios.append(float(np.mean(sw_hist[-50:]) / opt) if opt > 0 else 0)
        B_errors.append(float(np.mean(np.abs(algo.B_hat - env.B)[-50:])))
        if not use_oracle_I:
            I_errors.append(float(np.mean(np.abs(algo.I_hat - env.I)[-50:])))
    r = {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios)),
         'B_error': float(np.mean(B_errors)), 'B_error_std': float(np.std(B_errors))}
    if not use_oracle_I: r['I_error'] = float(np.mean(I_errors))
    return r


def run_ucb_greedy(n_seeds, T_steps):
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        algo = UCBGreedyBaseline(env, seed=si * 1000 + N)
        sw_hist = algo.run(T_steps)
        opt = float(np.sum(np.max(env.B, axis=1)))
        sw_ratios.append(float(np.mean(sw_hist[-50:]) / opt) if opt > 0 else 0)
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 65)
    print("CLEAN SPLIT V5 — B_hat illusion trap fix + UCB-greedy baseline")
    print("=" * 65)
    all_res = {}
    print("\nPHASE A: Oracle I (env.I)")
    rA = run_exp(True, N_SEEDS, T)
    all_res['phaseA'] = rA
    print(f"  SW={rA['sw_ratio']:.4f} +/- {rA['sw_std']:.4f}  B_err={rA['B_error']:.4f}")
    print("\nPHASE B: Learned I (tp only + route bonus + collision I)")
    rB = run_exp(False, N_SEEDS, T)
    all_res['phaseB'] = rB
    print(f"  SW={rB['sw_ratio']:.4f} +/- {rB['sw_std']:.4f}  B_err={rB['B_error']:.4f}  I_err={rB['I_error']:.4f}")
    print("\nUCB-GREEDY BASELINE (no SA, no QUBO):")
    rG = run_ucb_greedy(N_SEEDS, T)
    all_res['ucb_greedy'] = rG
    print(f"  SW={rG['sw_ratio']:.4f} +/- {rG['sw_std']:.4f}")
    print("\nSUMMARY:")
    print(f"  Phase A (Oracle):    SW={rA['sw_ratio']:.4f}")
    print(f"  Phase B (Learned):   SW={rB['sw_ratio']:.4f}")
    print(f"  UCB-Greedy Baseline: SW={rG['sw_ratio']:.4f}")
    print(f"  Previous best:       SW=0.0875")
    out = os.path.join(RESULTS, 'clean_split_v5_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()