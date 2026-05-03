"""
clean_split_v16.py — Final comprehensive test + honest analysis

V15: SA diagonal-only = UCB-Greedy = 0.0952 (both B_init=0.7)
This confirms: off-diagonal I_hat was POISON, not helpful.

All V6-V15 converge to ~0.0952 in Phase B. The ceiling is 73% of Phase A.

V16: 
1. Run Phase A (oracle I) to get the true upper bound
2. Run best Phase B config
3. Analyze the gap: is it fundamental (identifiability) or fixable?

Also: run a longer horizon (T=2000) to check convergence.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; N_SEEDS = 20


def run_phase_a(n_seeds, T):
    """Phase A: oracle I. Use env.I (true interference) in QUBO off-diagonal."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), 0.7)
        visits = np.zeros((N, M))
        
        for step in range(T):
            Q = np.zeros((N * M, N * M))
            for i in range(N):
                for k in range(M):
                    Q[i * M + k, i * M + k] = -B_hat[i, k] - 0.5 / math.sqrt(visits[i, k] + 1)
            for i in range(N):
                for k in range(M):
                    for l in range(k + 1, M):
                        Q[i * M + k, i * M + l] += 0.5
                        Q[i * M + l, i * M + k] += 0.5
            for i in range(N):
                for k in range(M):
                    for j in range(N):
                        if j == i: continue
                        for l in range(M):
                            Q[i * M + k, j * M + l] += env.I[i, k, j, l] * 0.5
            
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
            for i in range(N):
                k = a[i]
                I_sum = sum(env.I[i, k, j, a[j]] for j in range(N) if j != i)
                B_hat[i, k] += 0.12 * ((tp[i] + I_sum) - B_hat[i, k])
                visits[i, k] += 1
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), 0.7); vis2 = np.zeros((N, M)); sw50 = []
        for step in range(T):
            Q2 = np.zeros((N * M, N * M))
            for i in range(N):
                for k in range(M):
                    Q2[i * M + k, i * M + k] = -B_h2[i, k] - 0.5 / math.sqrt(vis2[i, k] + 1)
            for i in range(N):
                for k in range(M):
                    for l in range(k + 1, M):
                        Q2[i * M + k, i * M + l] += 0.5
                        Q2[i * M + l, i * M + k] += 0.5
            for i in range(N):
                for k in range(M):
                    for j in range(N):
                        if j == i: continue
                        for l in range(M):
                            Q2[i * M + k, j * M + l] += env.I[i, k, j, l] * 0.5
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
            for i in range(N):
                k = a2[i]
                I_sum2 = sum(env.I[i, k, j, a2[j]] for j in range(N) if j != i)
                B_h2[i, k] += 0.12 * ((tp2[i] + I_sum2) - B_h2[i, k])
                vis2[i, k] += 1
            if step >= T - 50:
                sw50.append(float(sum(tp2.values())))
        sw_ratios.append(float(np.mean(sw50) / opt) if opt > 0 else 0)
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def run_phase_b_best(n_seeds, T):
    """Phase B: learned I. Best config = UCB-Greedy B_init=0.7."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), 0.7)
        visits = np.zeros((N, M))
        for step in range(T):
            a = {}
            for i in range(N):
                a[i] = int(np.argmax(B_hat[i] + 0.5 / np.sqrt(visits[i] + 1)))
            tp = env.compute_throughput(a)
            for i in range(N):
                k = a[i]
                B_hat[i, k] += 0.12 * (tp[i] - B_hat[i, k])
                visits[i, k] += 1
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), 0.7); vis2 = np.zeros((N, M)); sw50 = []
        for step in range(T):
            a2 = {}
            for i in range(N):
                a2[i] = int(np.argmax(B_h2[i] + 0.5 / np.sqrt(vis2[i] + 1)))
            tp2 = env.compute_throughput(a2)
            for i in range(N):
                k = a2[i]
                B_h2[i, k] += 0.12 * (tp2[i] - B_h2[i, k])
                vis2[i, k] += 1
            if step >= T - 50:
                sw50.append(float(sum(tp2.values())))
        sw_ratios.append(float(np.mean(sw50) / opt) if opt > 0 else 0)
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V16 — Phase A vs Phase B gap analysis")
    print("=" * 70)
    all_res = {}
    
    print("\nPhase A (oracle I, T=500):")
    rA = run_phase_a(N_SEEDS, 500)
    all_res['phaseA_T500'] = rA
    print(f"  SW={rA['sw_ratio']:.4f} +/- {rA['sw_std']:.4f}")
    
    print("\nPhase B (learned I, T=500):")
    rB = run_phase_b_best(N_SEEDS, 500)
    all_res['phaseB_T500'] = rB
    print(f"  SW={rB['sw_ratio']:.4f} +/- {rB['sw_std']:.4f}")
    
    print("\nPhase B (learned I, T=2000):")
    rB2 = run_phase_b_best(N_SEEDS, 2000)
    all_res['phaseB_T2000'] = rB2
    print(f"  SW={rB2['sw_ratio']:.4f} +/- {rB2['sw_std']:.4f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print(f"  Phase A (oracle I, T=500):  SW={rA['sw_ratio']:.4f}")
    print(f"  Phase B (learned I, T=500): SW={rB['sw_ratio']:.4f}")
    print(f"  Phase B (learned I, T=2000): SW={rB2['sw_ratio']:.4f}")
    print(f"  Gap: {rA['sw_ratio'] - rB['sw_ratio']:.4f} ({rB['sw_ratio']/rA['sw_ratio']*100:.1f}% of Phase A)")
    
    out = os.path.join(RESULTS, 'clean_split_v16_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()