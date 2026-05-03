"""
clean_split_v17.py — Scale test: Does Phase B improve with MORE agents?

Hypothesis: With more agents (N=25) and same 4 routes, there are MORE collisions
→ I_hat learns better → Phase B approaches Phase A.

In N=10, M=4: collisions are sparse → I_hat can't learn well
In N=25, M=4: every step has many collisions → I_hat learns fast

Also test: If I_hat learns well with more agents, does Phase B SW actually increase?
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026


def run_phase_b(N, M, T, n_seeds, B_init, B_LR, UCB_C_val, I_LR):
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        I_hat = np.zeros((N, M, N, M))
        visits = np.zeros((N, M))
        _prev_x = None; _prev_tp = None
        
        for step in range(T):
            a = {}
            for i in range(N):
                scores = B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                a[i] = int(np.argmax(scores))
            tp = env.compute_throughput(a)
            
            # Learn I_hat from collisions
            if _prev_x is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = _prev_x[i]; kj = _prev_x[j]
                        drop_i = max(0.0, B_hat[i, ki] - _prev_tp[i])
                        drop_j = max(0.0, B_hat[j, kj] - _prev_tp[j])
                        if drop_i > 0.02: I_hat[i, ki, j, kj] = min(I_hat[i, ki, j, kj] + I_LR, 0.5)
                        if drop_j > 0.02: I_hat[j, kj, i, ki] = min(I_hat[j, kj, i, ki] + I_LR, 0.5)
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
        px2 = None; pt2 = None; sw50 = []
        for step in range(T):
            a2 = {}
            for i in range(N):
                a2[i] = int(np.argmax(B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)))
            tp2 = env.compute_throughput(a2)
            if px2 is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = px2[i]; kj = px2[j]
                        di = max(0.0, B_h2[i, ki] - pt2[i])
                        dj = max(0.0, B_h2[kj, kj] - pt2[j])
                        if di > 0.02: I_hat[i, ki, j, kj] = min(I_hat[i, ki, j, kj] + I_LR, 0.5)
                        if dj > 0.02: I_hat[j, kj, i, ki] = min(I_hat[j, kj, i, ki] + I_LR, 0.5)
                I_hat *= (1.0 - 0.001)
                for i in range(N): I_hat[i, :, i, :] = 0.0
            for i in range(N):
                k = a2[i]
                B_h2[i, k] += B_LR * (tp2[i] - B_h2[i, k])
                vis2[i, k] += 1
            px2 = a2.copy(); pt2 = {i: tp2[i] for i in range(N)}
            if step >= T - 50:
                sw50.append(float(sum(tp2.values())))
        sw_ratios.append(float(np.mean(sw50) / opt) if opt > 0 else 0)
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def run_phase_a(N, M, T, n_seeds):
    """Oracle I — QUBO with env.I"""
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


def main():
    print("=" * 70)
    print("V17 — Scale test: N=10 vs N=25")
    print("=" * 70)
    all_res = {'v16_phaseA_N10': 0.1149, 'v16_phaseB_N10': 0.0952}
    
    # N=10 baseline (from V16)
    print("\nN=10, M=4 (baseline from V16):")
    rA10 = run_phase_a(10, 4, 500, 10)
    rB10 = run_phase_b(10, 4, 500, 10, 0.7, 0.12, 0.5, 0.05)
    all_res['phaseA_N10'] = rA10; all_res['phaseB_N10'] = rB10
    print(f"  Phase A: SW={rA10['sw_ratio']:.4f}")
    print(f"  Phase B: SW={rB10['sw_ratio']:.4f}")
    
    # N=25, M=4
    print("\nN=25, M=4:")
    rA25 = run_phase_a(25, 4, 500, 10)
    rB25 = run_phase_b(25, 4, 500, 10, 0.7, 0.12, 0.5, 0.05)
    all_res['phaseA_N25'] = rA25; all_res['phaseB_N25'] = rB25
    print(f"  Phase A: SW={rA25['sw_ratio']:.4f}")
    print(f"  Phase B: SW={rB25['sw_ratio']:.4f}")
    
    # N=15, M=4
    print("\nN=15, M=4:")
    rA15 = run_phase_a(15, 4, 500, 10)
    rB15 = run_phase_b(15, 4, 500, 10, 0.7, 0.12, 0.5, 0.05)
    all_res['phaseA_N15'] = rA15; all_res['phaseB_N15'] = rB15
    print(f"  Phase A: SW={rA15['sw_ratio']:.4f}")
    print(f"  Phase B: SW={rB15['sw_ratio']:.4f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  N=10: PhaseA={rA10['sw_ratio']:.4f} PhaseB={rB10['sw_ratio']:.4f} ({rB10['sw_ratio']/rA10['sw_ratio']*100:.1f}%)")
    print(f"  N=15: PhaseA={rA15['sw_ratio']:.4f} PhaseB={rB15['sw_ratio']:.4f} ({rB15['sw_ratio']/rA15['sw_ratio']*100:.1f}%)")
    print(f"  N=25: PhaseA={rA25['sw_ratio']:.4f} PhaseB={rB25['sw_ratio']:.4f} ({rB25['sw_ratio']/rA25['sw_ratio']*100:.1f}%)")
    
    out = os.path.join(RESULTS, 'clean_split_v17_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()