"""
clean_split_v7.py — UCB-Greedy is the winner (SW=0.0888). Optimize it.

V6 results:
- SA+QUBO (full):    SW=0.0742
- SA+QUBO (no off):  SW=0.0742
- UCB-Greedy:       SW=0.0888

SA+QUBO adds NOTHING. B_hat learning is the bottleneck.

V7: Sweep B_LR and UCB_C to optimize UCB-Greedy.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_ucb_greedy(n_seeds, B_LR, UCB_C_val, use_collision_I):
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), 0.5)
        I_hat = np.full((N, M, N, M), 0.0)
        for i in range(N): I_hat[i,:,i,:] = 0.0
        visits = np.zeros((N, M))
        _prev_x = None; _prev_tp = None
        
        for _ in range(T):
            assignment = {}
            for i in range(N):
                scores = B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                assignment[i] = int(np.argmax(scores))
            tp_actual = env.compute_throughput(assignment)
            for i in range(N):
                k = assignment[i]
                B_hat[i, k] += B_LR * (tp_actual[i] - B_hat[i, k])
                visits[i, k] += 1
            
            if use_collision_I and _prev_x is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = _prev_x[i]; kj = _prev_x[j]
                        di = max(0.0, B_hat[i, ki] - _prev_tp[i])
                        dj = max(0.0, B_hat[j, kj] - _prev_tp[j])
                        if di > 0.02: I_hat[i, ki, j, kj] = min(I_hat[i, ki, j, kj] + 0.05, 0.3)
                        if dj > 0.02: I_hat[j, kj, i, ki] = min(I_hat[j, kj, i, ki] + 0.05, 0.3)
                I_hat *= (1.0 - 0.001)
            
            _prev_x = assignment.copy()
            _prev_tp = {i: tp_actual[i] for i in range(N)}
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        sw_last50 = []
        B_h2 = np.full((N, M), 0.5); vis2 = np.zeros((N, M)); px2 = None; pt2 = None
        for step in range(T):
            a = {}
            for i in range(N):
                s = B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)
                a[i] = int(np.argmax(s))
            tp = env.compute_throughput(a)
            for i in range(N):
                k = a[i]
                B_h2[i, k] += B_LR * (tp[i] - B_h2[i, k])
                vis2[i, k] += 1
            if use_collision_I and px2 is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = px2[i]; kj = px2[j]
                        di = max(0.0, B_h2[i, ki] - pt2[i])
                        dj = max(0.0, B_h2[j, kj] - pt2[j])
                        if di > 0.02: I_hat[i, ki, j, kj] = min(I_hat[i, ki, j, kj] + 0.05, 0.3)
                        if dj > 0.02: I_hat[j, kj, i, ki] = min(I_hat[j, kj, i, ki] + 0.05, 0.3)
                I_hat *= (1.0 - 0.001)
            px2 = a.copy(); pt2 = {i: tp[i] for i in range(N)}
            if step >= T - 50:
                sw_last50.append(float(sum(tp.values())))
        sw_ratios.append(float(np.mean(sw_last50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V7 — UCB-Greedy parameter sweep")
    print("=" * 70)
    all_res = {'v6_reference': {'sw_ratio': 0.0888}}
    
    print("\nSweeping B_LR (no collision I):")
    for B_LR in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]:
        r = run_ucb_greedy(N_SEEDS, B_LR, 0.5, False)
        all_res[f'B_LR={B_LR}_noI'] = r
        print(f"  B_LR={B_LR}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    print("\nSweeping B_LR (WITH collision I):")
    for B_LR in [0.08, 0.1, 0.15, 0.2]:
        r = run_ucb_greedy(N_SEEDS, B_LR, 0.5, True)
        all_res[f'B_LR={B_LR}_withI'] = r
        print(f"  B_LR={B_LR}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    print("\nSweeping UCB_C (B_LR=0.15, no I):")
    for UCB_C in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        r = run_ucb_greedy(N_SEEDS, 0.15, UCB_C, False)
        all_res[f'UCB_C={UCB_C}'] = r
        print(f"  UCB_C={UCB_C}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    # Find best
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'])
    print(f"\nBEST: {best_key} = {all_res[best_key]['sw_ratio']:.4f}")
    
    out = os.path.join(RESULTS, 'clean_split_v7_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
