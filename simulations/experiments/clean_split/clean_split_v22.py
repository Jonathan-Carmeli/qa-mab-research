"""
clean_split_v22.py — Final comprehensive sweep

Best config so far: UCB-Greedy B_init=0.7 B_LR=0.12 UCB_C=0.5 = SW=0.107 (V17)

Key question: Can we push past 0.107?

V22: Fine-grained sweep around best config + test if T=1000 helps
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; N_SEEDS = 20


def run_ucb(n_seeds, B_init, B_LR, UCB_C_val, T):
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
    print("=" * 70)
    print("V22 — Final sweep: B_init × B_LR × UCB_C")
    print("=" * 70)
    all_res = {'v17_best': 0.1070, 'v21_best': 0.0952}
    
    configs = [
        (0.7, 0.12, 0.5),
        (0.7, 0.10, 0.5),
        (0.7, 0.08, 0.5),
        (0.65, 0.12, 0.5),
        (0.75, 0.12, 0.5),
        (0.7, 0.12, 0.4),
        (0.7, 0.12, 0.6),
        (0.6, 0.15, 0.5),
        (0.8, 0.10, 0.5),
        (0.75, 0.08, 0.4),
        (0.65, 0.10, 0.4),
        (0.7, 0.10, 0.6),
    ]
    
    for B_init, B_LR, UCB_C in configs:
        r = run_ucb(N_SEEDS, B_init, B_LR, UCB_C, 500)
        all_res[f'i={B_init}_l={B_LR}_u={UCB_C}'] = r
        print(f"  i={B_init} l={B_LR} u={UCB_C}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    # T=1000 test with best config
    print("\nT=1000 with best configs:")
    for B_init, B_LR, UCB_C in [(0.7, 0.12, 0.5), (0.65, 0.12, 0.5), (0.75, 0.12, 0.5)]:
        r = run_ucb(N_SEEDS, B_init, B_LR, UCB_C, 1000)
        all_res[f'i={B_init}_l={B_LR}_u={UCB_C}_T1000'] = r
        print(f"  i={B_init} l={B_LR} u={UCB_C} T=1000: SW={r['sw_ratio']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"\nBEST: {best_key} = {best_val:.4f}")
    print(f"Previous: V17=0.1070, V21=0.0952")
    
    with open(os.path.join(RESULTS, 'clean_split_v22_results.json'), 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f"Saved: {RESULTS}/clean_split_v22_results.json")


if __name__ == '__main__':
    main()