"""
clean_split_v21.py — UCB-Greedy + collision inference only (NO QUBO)

Key insight: UCB-Greedy = SW=0.107 (V17). SA+QUBO with wrong I_hat makes it worse.

V21: UCB-Greedy for route selection + I_hat learned from collisions for analysis only.
The I_hat does NOT affect routing — it's just tracked to see if we can measure interference.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_ucb_with_I_tracking(n_seeds, B_init, B_LR, UCB_C_val, I_LR, I_CAP):
    """UCB-Greedy + I_hat tracked (but NOT used in routing)."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        I_hat = np.zeros((N, M, N, M))
        visits = np.zeros((N, M))
        _prev_x = None; _prev_tp = None
        
        for step in range(T):
            # Route selection: UCB-Greedy only (no I_hat in routing)
            assignment = {}
            for i in range(N):
                scores = B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                assignment[i] = int(np.argmax(scores))
            
            tp = env.compute_throughput(assignment)
            
            # Track I_hat from collision inference (for analysis only, not used in routing)
            if _prev_x is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = _prev_x[i]; kj = _prev_x[j]
                        di = max(0.0, B_hat[i, ki] - _prev_tp[i])
                        dj = max(0.0, B_hat[j, kj] - _prev_tp[j])
                        if di > 0.02: I_hat[i, ki, j, kj] = min(I_hat[i, ki, j, kj] + I_LR, I_CAP)
                        if dj > 0.02: I_hat[j, kj, i, ki] = min(I_hat[j, kj, i, ki] + I_LR, I_CAP)
                I_hat *= (1.0 - 0.001)
                for i in range(N): I_hat[i, :, i, :] = 0.0
            
            for i in range(N):
                k = assignment[i]
                B_hat[i, k] += B_LR * (tp[i] - B_hat[i, k])
                visits[i, k] += 1
            
            _prev_x = assignment.copy()
            _prev_tp = {i: tp[i] for i in range(N)}
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M))
        I_h2 = np.zeros((N, M, N, M)); px2 = None; pt2 = None
        sw50 = []
        for step in range(T):
            a = {}
            for i in range(N):
                a[i] = int(np.argmax(B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)))
            tp = env.compute_throughput(a)
            if px2 is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = px2[i]; kj = px2[j]
                        di = max(0.0, B_h2[i, ki] - pt2[i])
                        dj = max(0.0, B_h2[kj, kj] - pt2[j])
                        if di > 0.02: I_h2[i, ki, j, kj] = min(I_h2[i, ki, j, kj] + I_LR, I_CAP)
                        if dj > 0.02: I_h2[j, kj, i, ki] = min(I_h2[j, kj, i, ki] + I_LR, I_CAP)
                I_h2 *= (1.0 - 0.001)
                for i in range(N): I_h2[i, :, i, :] = 0.0
            for i in range(N):
                k = a[i]
                B_h2[i, k] += B_LR * (tp[i] - B_h2[i, k])
                vis2[i, k] += 1
            px2 = a.copy(); pt2 = {i: tp[i] for i in range(N)}
            if step >= T - 50:
                sw50.append(float(sum(tp.values())))
        sw_ratios.append(float(np.mean(sw50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V21 — UCB-Greedy + I_tracking (I NOT used in routing)")
    print("=" * 70)
    all_res = {'v17_best': 0.1070}
    
    # V21a: Different I_LR values (tracked but not used)
    print("\nI_LR sweep (B_init=0.7, B_LR=0.12, UCB_C=0.5):")
    for I_LR in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        r = run_ucb_with_I_tracking(N_SEEDS, 0.7, 0.12, 0.5, I_LR, 0.5)
        all_res[f'I_LR={I_LR}'] = r
        print(f"  I_LR={I_LR}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    # V21b: Higher I_CAP
    print("\nI_CAP sweep (B_init=0.7, I_LR=0.2):")
    for I_CAP in [0.3, 0.5, 0.7, 1.0]:
        r = run_ucb_with_I_tracking(N_SEEDS, 0.7, 0.12, 0.5, 0.2, I_CAP)
        all_res[f'I_CAP={I_CAP}'] = r
        print(f"  I_CAP={I_CAP}: SW={r['sw_ratio']:.4f}")
    
    # V21c: Different B_init
    print("\nB_init sweep (I_LR=0.2, I_CAP=0.5):")
    for B_init in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]:
        r = run_ucb_with_I_tracking(N_SEEDS, B_init, 0.12, 0.5, 0.2, 0.5)
        all_res[f'Binit={B_init}'] = r
        print(f"  B_init={B_init}: SW={r['sw_ratio']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'])
    best_val = all_res[best_key]['sw_ratio']
    print(f"\nBEST: {best_key} = {best_val:.4f}")
    print(f"V17 best: 0.1070")
    
    with open(os.path.join(RESULTS, 'clean_split_v21_results.json'), 'w') as f:
        json.dump(all_res, f, indent=2)


if __name__ == '__main__':
    main()