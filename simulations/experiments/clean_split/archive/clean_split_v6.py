"""
clean_split_v6.py — Test: Is SA+QUBO the bottleneck, or is the learning?

V5 insight: Perhaps the SA solver is actually working FINE, but the
B_hat illusion trap prevents ANY optimizer from making good decisions.

V6: Try THREE approaches simultaneously to diagnose the real bottleneck:
1. SA+QUBO (current) — but with tp-only diagonal (V5 approach)
2. UCB-Greedy (simple argmax) — no QUBO, no SA
3. No-off-diagonal QUBO (diagonal only) — test if I_hat term is the poison

If UCB-Greedy >> SA+QUBO → SA is the bottleneck
If No-off-diagonal ≈ SA+QUBO → I_hat in off-diagonal is the poison
If SA+QUBO still fails → problem is B_hat learning, not the optimizer
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
ROUTE_UNUSED_BONUS = 0.3; ROUTE_UNUSED_K = 50; EPS_GREEDY = 0.05


class AlgoV6SA:
    """SA+QUBO with tp-only diagonal + route bonus + conservative I_hat"""
    def __init__(self, env, use_oracle_I=True, seed=42):
        self.env = env; self.N = env.N; self.m = env.m
        self.use_oracle_I = use_oracle_I
        self.rng = np.random.default_rng(seed)
        self.B_hat = np.full((self.N, self.m), 1.0)
        self.I_hat = np.full((self.N, self.m, self.N, self.m), 0.01)
        for i in range(self.N): self.I_hat[i,:,i,:] = 0.0
        self.visits = np.zeros((self.N, self.m))
        self.last_visit = np.zeros((self.N, self.m))
        self._prev_x = None; self._prev_tp = None; self.step_count = 0

    def idx(self, i, k): return i * self.m + k

    def build_qubo(self, t, use_offdiag=True):
        size = self.N * self.m; Q = np.zeros((size, size))
        for i in range(self.N):
            for k in range(self.m):
                idx = self.idx(i, k)
                ucb = UCB_C / math.sqrt(self.visits[i, k] + 1)
                Q[idx, idx] = -self.B_hat[i, k] - ucb
                if t - self.last_visit[i, k] > ROUTE_UNUSED_K:
                    Q[idx, idx] += ROUTE_UNUSED_BONUS
        tau = 1.0
        for i in range(self.N):
            for k in range(self.m):
                for l in range(k + 1, self.m):
                    Q[self.idx(i, k), self.idx(i, l)] += tau / 2
                    Q[self.idx(i, l), self.idx(i, k)] += tau / 2
        if use_offdiag:
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
                k_old = int(np.argmax(block)); k_new = (k_old + 1 + self.rng.integers(0, m - 1)) % m
                x[i * m + k_old] = 0.0; x[i * m + k_new] = 1.0
                ne = float(x @ Q @ x)
                if ne < e or self.rng.random() < math.exp(-(ne - e) / T_r):
                    e = ne
                    if e < best_e: best_e, best_x = e, x.copy()
                else:
                    x[i * m + k_new] = 0.0; x[i * m + k_old] = 1.0
        return {i: int(np.argmax(best_x[i * m:(i + 1) * m])) for i in range(n)}

    def step(self, use_offdiag=True):
        t = self.step_count + 1
        Q = self.build_qubo(t, use_offdiag)
        assignment = self.sa_solve(Q, t)
        tp_actual = self.env.compute_throughput(assignment)
        if self.use_oracle_I:
            for i in range(self.N):
                k = assignment[i]
                I_sum = sum(self.env.I[i, k, j, assignment[j]] for j in range(self.N) if j != i)
                self.B_hat[i, k] += B_LR * ((tp_actual[i] + I_sum) - self.B_hat[i, k])
                self.visits[i, k] += 1; self.last_visit[i, k] = t
        else:
            for i in range(self.N):
                k = assignment[i]
                self.B_hat[i, k] += B_LR * (tp_actual[i] - self.B_hat[i, k])
                self.visits[i, k] += 1; self.last_visit[i, k] = t
            if self._prev_x is not None and use_offdiag:
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        ki = self._prev_x[i]; kj = self._prev_x[j]
                        d_i = max(0.0, self.B_hat[i, ki] - self._prev_tp[i])
                        d_j = max(0.0, self.B_hat[j, kj] - self._prev_tp[j])
                        if d_i > 0.02: self.I_hat[i, ki, j, kj] = min(self.I_hat[i, ki, j, kj] + I_LR, I_CAP)
                        if d_j > 0.02: self.I_hat[j, kj, i, ki] = min(self.I_hat[j, kj, i, ki] + I_LR, I_CAP)
            self.I_hat *= (1.0 - 0.001)
            for i in range(self.N): self.I_hat[i,:,i,:] = 0.0
        self._prev_x = assignment.copy()
        self._prev_tp = {i: tp_actual[i] for i in range(self.N)}
        self.step_count += 1
        return float(sum(tp_actual.values()))

    def run(self, T_steps, use_offdiag=True):
        return [self.step(use_offdiag) for _ in range(T_steps)]


class UCBGreedy:
    """Simple per-agent argmax(B_hat + UCB), no QUBO, no SA"""
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
            scores = self.B_hat[i] + UCB_C / np.sqrt(self.visits[i] + 1)
            assignment[i] = int(np.argmax(scores))
        tp_actual = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            self.B_hat[i, k] += B_LR * (tp_actual[i] - self.B_hat[i, k])
            self.visits[i, k] += 1
        if self._prev_x is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ki = self._prev_x[i]; kj = self._prev_x[j]
                    d_i = max(0.0, self.B_hat[i, ki] - self._prev_tp[i])
                    d_j = max(0.0, self.B_hat[j, kj] - self._prev_tp[j])
                    if d_i > 0.02: self.I_hat[i, ki, j, kj] = min(self.I_hat[i, ki, j, kj] + I_LR, I_CAP)
                    if d_j > 0.02: self.I_hat[j, kj, i, ki] = min(self.I_hat[j, kj, i, ki] + I_LR, I_CAP)
        self.I_hat *= (1.0 - 0.001)
        for i in range(self.N): self.I_hat[i,:,i,:] = 0.0
        self._prev_x = assignment.copy()
        self._prev_tp = {i: tp_actual[i] for i in range(self.N)}
        self.step_count += 1
        return float(sum(tp_actual.values()))

    def run(self, T_steps):
        return [self.step() for _ in range(T_steps)]


def run_exp(algo_class, use_oracle_I, n_seeds, T_steps, **kwargs):
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        if algo_class == UCBGreedy:
            algo = UCBGreedy(env, seed=si * 1000 + N)
        else:
            algo = algo_class(env, use_oracle_I=use_oracle_I, seed=si * 1000 + N)
        sw_hist = algo.run(T_steps, **kwargs) if hasattr(algo, 'run') else [algo.step(**kwargs) for _ in range(T_steps)]
        opt = float(np.sum(np.max(env.B, axis=1)))
        sw_ratios.append(float(np.mean(sw_hist[-50:]) / opt) if opt > 0 else 0)
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V6 DIAGNOSIS: SA+QUBO vs UCB-Greedy vs No-OffDiagonal")
    print("=" * 70)
    all_res = {}

    # Phase A: Oracle I
    print("\n[PHASE A: Oracle I]")
    rA_sa = run_exp(AlgoV6SA, True, N_SEEDS, T)
    all_res['phaseA_SA'] = rA_sa
    print(f"  SA+QUBO (oracle):   SW={rA_sa['sw_ratio']:.4f} +/- {rA_sa['sw_std']:.4f}")

    # Phase B: Three variants
    print("\n[PHASE B: Learned I — 3 approaches]")

    rB_sa = run_exp(AlgoV6SA, False, N_SEEDS, T)
    all_res['phaseB_SA_QUBO'] = rB_sa
    print(f"  SA+QUBO (full):     SW={rB_sa['sw_ratio']:.4f} +/- {rB_sa['sw_std']:.4f}")

    rB_nod = run_exp(AlgoV6SA, False, N_SEEDS, T, use_offdiag=False)
    all_res['phaseB_SA_NoOffDiag'] = rB_nod
    print(f"  SA+QUBO (no off):   SW={rB_nod['sw_ratio']:.4f} +/- {rB_nod['sw_std']:.4f}")

    rB_greedy = run_exp(UCBGreedy, None, N_SEEDS, T)
    all_res['phaseB_UCB_Greedy'] = rB_greedy
    print(f"  UCB-Greedy:          SW={rB_greedy['sw_ratio']:.4f} +/- {rB_greedy['sw_std']:.4f}")

    print("\nDIAGNOSIS:")
    print(f"  If UCB-Greedy >> SA+QUBO: SA solver is bottleneck")
    print(f"  If NoOffDiagonal >> SA+QUBO: I_hat off-diagonal is poison")
    print(f"  If all similar but low: B_hat learning is bottleneck")
    print(f"\n  Oracle reference:   SW={rA_sa['sw_ratio']:.4f}")

    out = os.path.join(RESULTS, 'clean_split_v6_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()