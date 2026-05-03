"""
clean_split_v15b.py — Quick version: SA diagonal-only vs UCB-Greedy

Reduced SA iterations for speed.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_sa(n_seeds, B_init, B_LR, UCB_C_val, sa_iters=200, sa_restarts=4):
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        for step in range(T):
            Q = np.zeros((N * M, N * M))
            for i in range(N):
                for k in range(M):
                    Q[i * M + k, i * M + k] = -B_hat[i, k] - UCB_C_val / math.sqrt(visits[i, k] + 1)
            for i in range(N):
                for k in range(M):
                    for l in range(k + 1, M):
                        Q[i * M + k, i * M + l] += 0.5
                        Q[i * M + l, i * M + k] += 0.5
            T_sa = 2.0 / math.log(step + 3)
            best_x = None; best_e = float('inf')
            for r in range(sa_restarts):
                x = np.zeros(N * M)
                for i in range(N): x[i * M + int(np.argmax(B_hat[i]))] = 1.0
                e = float(x @ Q @ x)
                if e < best_e: best_e, best_x = e, x.copy()
                T_r = T_sa * (1 + r * 0.3)
                for _ in range(sa_iters):
                    T_r *= 0.95
                    if T_r < 1e-12: break
                    i = rng.integers(0, N)
                    block = x[i * M:(i + 1) * M]
                    ko = int(np.argmax(block)); kn = (ko + 1 + rng.integers(0, M - 1)) % M
                    x[i * M + ko] = 0.0; x[i * M + kn] = 1.0
                    ne = float(x @ Q @ x)
                    if ne < e or rng.random() < math.exp(-(ne - e) / T_r):
                        e = ne
                        if e < best_e: best_e, best_x = e, x.copy()
                    else:
                        x[i * M + kn] = 0.0; x[i * M + ko] = 1.0
            a = {i: int(np.argmax(best_x[i * M:(i + 1) * M])) for i in range(N)}
            tp = env.compute_throughput(a)
            for i in range(N):
                k = a[i]
                B_hat[i, k] += B_LR * (tp[i] - B_hat[i, k])
                visits[i, k] += 1
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M)); sw50 = []
        for step in range(T):
            a2 = {}
            for i in range(N):
                a2[i] = int(np.argmax(B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)))
            tp2 = env.compute_throughput(a2)
            for i in range(N):
                k = a2[i]
                B_h2[i, k] += B_LR * (tp2[i] - B_h2[i, k])
                vis2[i, k] += 1
            if step >= T - 50:
                sw50.append(float(sum(tp2.values())))
        sw_ratios.append(float(np.mean(sw50) / opt) if opt > 0 else 0)
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def run_ucb(n_seeds, B_init, B_LR, UCB_C_val):
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        for step in range(T):
            a = {}
            for i in range(N):
                a[i] = int(np.argmax(B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)))
            tp = env.compute_throughput(a)
            for i in range(N):
                k = a[i]
                B_hat[i, k] += B_LR * (tp[i] - B_hat[i, k])
                visits[i, k] += 1
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M)); sw50 = []
        for step in range(T):
            a2 = {}
            for i in range(N):
                a2[i] = int(np.argmax(B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)))
            tp2 = env.compute_throughput(a2)
            for i in range(N):
                k = a2[i]
                B_h2[i, k] += B_LR * (tp2[i] - B_h2[i, k])
                vis2[i, k] += 1
            if step >= T - 50:
                sw50.append(float(sum(tp2.values())))
        sw_ratios.append(float(np.mean(sw50) / opt) if opt > 0 else 0)
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("V15 — SA diagonal-only vs UCB-Greedy")
    all_res = {'v6_ucb': 0.0888, 'v14_best': 0.0952}
    
    for B_init in [0.5, 0.7, 1.0]:
        r = run_sa(N_SEEDS, B_init, 0.12, 0.5)
        all_res[f'sa_i={B_init}'] = r
        print(f"SA B_init={B_init}: SW={r['sw_ratio']:.4f}")
    
    for B_init in [0.5, 0.7]:
        r = run_ucb(N_SEEDS, B_init, 0.12, 0.5)
        all_res[f'ucb_i={B_init}'] = r
        print(f"UCB B_init={B_init}: SW={r['sw_ratio']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"BEST: {best_key} = {best_val:.4f}")
    
    with open(os.path.join(RESULTS, 'clean_split_v15b_results.json'), 'w') as f:
        json.dump(all_res, f, indent=2)


if __name__ == '__main__':
    main()