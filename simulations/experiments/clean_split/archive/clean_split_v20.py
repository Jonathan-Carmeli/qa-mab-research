"""
clean_split_v20.py — I_LR sweep to find convergence

V19 finding: High I_LR (0.15) → SW improves over time (+0.04 at T=200)
Need to sweep I_LR from 0.05 to 0.5
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 10


def run_phase_b(I_LR, B_init, B_LR, UCB_C_val):
    sw_ratios = []
    for si in range(N_SEEDS):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        I_hat = np.zeros((N, M, N, M))
        _prev_x = None; _prev_tp = None
        tau = 1.0
        
        for step in range(T):
            Q = np.zeros((N * M, N * M))
            for i in range(N):
                for k in range(M):
                    Q[i * M + k, i * M + k] = -B_hat[i, k] - UCB_C_val / math.sqrt(visits[i, k] + 1)
            for i in range(N):
                for k in range(M):
                    for l in range(k + 1, M):
                        Q[i * M + k, i * M + l] += tau / 2
                        Q[i * M + l, i * M + k] += tau / 2
            for i in range(N):
                for k in range(M):
                    for j in range(N):
                        if j == i: continue
                        for l in range(M):
                            Q[i * M + k, j * M + l] += I_hat[i, k, j, l] * 0.5
            
            T_sa = 2.0 / math.log(step + 3)
            best_x = None; best_e = float('inf')
            for r in range(4):
                x = np.zeros(N * M)
                for i in range(N): x[i * M + int(np.argmax(B_hat[i]))] = 1.0
                e = float(x @ Q @ x)
                if e < best_e: best_e, best_x = e, x.copy()
                T_r = T_sa * (1 + r * 0.3)
                for _ in range(200):
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
            
            if _prev_x is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = _prev_x[i]; kj = _prev_x[j]
                        di = max(0.0, B_hat[i, ki] - _prev_tp[i])
                        dj = max(0.0, B_hat[j, kj] - _prev_tp[j])
                        if di > 0.02: I_hat[i, ki, j, kj] = min(I_hat[i, ki, j, kj] + I_LR, 0.5)
                        if dj > 0.02: I_hat[j, kj, i, ki] = min(I_hat[j, kj, i, ki] + I_LR, 0.5)
                I_hat *= (1.0 - 0.001)
                for i in range(N): I_hat[i, :, i, :] = 0.0
            
            for i in range(N):
                k = a[i]
                B_hat[i, k] += B_LR * (tp[i] - B_hat[i, k])
                visits[i, k] += 1
            
            _prev_x = a.copy()
            _prev_tp = {i: tp[i] for i in range(N)}
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M))
        I_h2 = np.zeros((N, M, N, M)); px2 = None; pt2 = None
        sw50 = []
        for step in range(T):
            Q2 = np.zeros((N * M, N * M))
            for i in range(N):
                for k in range(M):
                    Q2[i * M + k, i * M + k] = -B_h2[i, k] - UCB_C_val / math.sqrt(vis2[i, k] + 1)
            for i in range(N):
                for k in range(M):
                    for l in range(k + 1, M):
                        Q2[i * M + k, i * M + l] += tau / 2
                        Q2[i * M + l, i * M + k] += tau / 2
            for i in range(N):
                for k in range(M):
                    for j in range(N):
                        if j == i: continue
                        for l in range(M):
                            Q2[i * M + k, j * M + l] += I_h2[i, k, j, l] * 0.5
            T_sa2 = 2.0 / math.log(step + 3); best_x2 = None; best_e2 = float('inf')
            for r in range(4):
                x2 = np.zeros(N * M)
                for i in range(N): x2[i * M + int(np.argmax(B_h2[i]))] = 1.0
                e2 = float(x2 @ Q2 @ x2)
                if e2 < best_e2: best_e2, best_x2 = e2, x2.copy()
                T_r2 = T_sa2 * (1 + r * 0.3)
                for _ in range(200):
                    T_r2 *= 0.95
                    if T_r2 < 1e-12: break
                    i = rng.integers(0, N)
                    block2 = x2[i * M:(i + 1) * M]
                    ko2 = int(np.argmax(block2)); kn2 = (ko2 + 1 + rng.integers(0, M - 1)) % M
                    x2[i * M + ko2] = 0.0; x2[i * M + kn2] = 1.0
                    ne2 = float(x2 @ Q2 @ x2)
                    if ne2 < e2 or rng.random() < math.exp(-(ne2 - e2) / T_r2):
                        e2 = ne2
                        if e2 < best_e2: best_e2, best_x2 = e2, x2.copy()
                    else:
                        x2[i * M + kn2] = 0.0; x2[i * M + ko2] = 1.0
            a2 = {i: int(np.argmax(best_x2[i * M:(i + 1) * M])) for i in range(N)}
            tp2 = env.compute_throughput(a2)
            if px2 is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = px2[i]; kj = px2[j]
                        di = max(0.0, B_h2[i, ki] - pt2[i])
                        dj = max(0.0, B_h2[j, kj] - pt2[j])
                        if di > 0.02: I_h2[i, ki, j, kj] = min(I_h2[i, ki, j, kj] + I_LR, 0.5)
                        if dj > 0.02: I_h2[j, kj, i, ki] = min(I_h2[j, kj, i, ki] + I_LR, 0.5)
                I_h2 *= (1.0 - 0.001)
                for i in range(N): I_h2[i, :, i, :] = 0.0
            for i in range(N):
                k = a2[i]
                B_h2[i, k] += B_LR * (tp2[i] - B_h2[i, k])
                vis2[i, k] += 1
            px2 = a2.copy(); pt2 = {i: tp2[i] for i in range(N)}
            if step >= T - 50:
                sw50.append(float(sum(tp2.values())))
        sw_ratios.append(float(np.mean(sw50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("V20 — I_LR sweep")
    all_res = {}
    
    print("\nI_LR sweep (B_init=0.7, B_LR=0.12, UCB_C=0.5):")
    for I_LR in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        r = run_phase_b(I_LR, 0.7, 0.12, 0.5)
        all_res[f'I_LR={I_LR}'] = r
        print(f"  I_LR={I_LR}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    print("\nBest I_LR + B_init sweep:")
    for B_init in [0.5, 0.8, 1.0, 1.2]:
        r = run_phase_b(0.2, B_init, 0.12, 0.5)
        all_res[f'I_LR=0.2_Binit={B_init}'] = r
        print(f"  I_LR=0.2 B_init={B_init}: SW={r['sw_ratio']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'])
    best_val = all_res[best_key]['sw_ratio']
    print(f"\nBEST: {best_key} = {best_val:.4f}")
    
    with open(os.path.join(RESULTS, 'clean_split_v20_results.json'), 'w') as f:
        json.dump(all_res, f, indent=2)


if __name__ == '__main__':
    main()