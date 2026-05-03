"""
clean_split_v11.py — Fine-tune around B_init=0.7 discovery

V10: B_init=0.7 → SW=0.0952 (best!)
V6 ref: 0.0888

V11: Sweep B_init in [0.6, 0.65, 0.7, 0.75, 0.8] × UCB_C in [0.3, 0.4, 0.5, 0.6] × B_LR in [0.1, 0.12, 0.15]
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_ucb(n_seeds, B_init, UCB_C_val, B_LR):
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        
        for step in range(T):
            assignment = {}
            for i in range(N):
                scores = B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                assignment[i] = int(np.argmax(scores))
            tp_actual = env.compute_throughput(assignment)
            for i in range(N):
                k = assignment[i]
                B_hat[i, k] += B_LR * (tp_actual[i] - B_hat[i, k])
                visits[i, k] += 1
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M))
        sw_last50 = []
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
            if step >= T - 50:
                sw_last50.append(float(sum(tp.values())))
        sw_ratios.append(float(np.mean(sw_last50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V11 — Fine-tune: B_init × UCB_C × B_LR")
    print("=" * 70)
    all_res = {'v6_ref': 0.0888, 'v10_best': 0.0952}
    
    best = 0.0
    for B_init in [0.6, 0.65, 0.7, 0.75, 0.8]:
        for UCB_C_val in [0.3, 0.4, 0.5, 0.6]:
            for B_LR in [0.1, 0.12, 0.15]:
                r = run_ucb(N_SEEDS, B_init, UCB_C_val, B_LR)
                all_res[f'init={B_init}_ucb={UCB_C_val}_lr={B_LR}'] = r
                print(f"  init={B_init} ucb={UCB_C_val} lr={B_LR}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
                if r['sw_ratio'] > best:
                    best = r['sw_ratio']
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"\nBEST: {best_key} = {best_val:.4f}")
    
    out = os.path.join(RESULTS, 'clean_split_v11_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()