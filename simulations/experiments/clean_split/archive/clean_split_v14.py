"""
clean_split_v14.py — Fine-tune around V11's best + extended horizon

V11 best: B_init=0.7, ucb=0.5, lr=0.12 → SW=0.0952 (73% of Phase A oracle)

Key insight from V10: B_init=0.7 beats B_init=0.5 because it prevents
premature convergence to suboptimal routes. UCB-Greedy naturally handles
exploration/exploitation tradeoff via the UCB term.

V14: 
1. Fine-tune around V11 best (smaller lr steps, different ucb)
2. Test longer horizon (T=1000) to see if algorithm can reach higher SW
3. Test lower B_LR with higher B_init for smoother convergence
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; N_SEEDS = 20


def run_ucb(n_seeds, B_init, UCB_C_val, B_LR, T):
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
    print("V14 — Fine-tune V11 best + longer horizon")
    print("=" * 70)
    all_res = {'v6_ref': 0.0888, 'v10_best': 0.0952, 'v11_best': 0.0952}
    
    # V14a: Fine-tune around B_init=0.7, ucb=0.5, lr=0.12
    print("\nFine-tune around V11 best:")
    configs = [
        (0.7, 0.4, 0.12), (0.7, 0.5, 0.12), (0.7, 0.6, 0.12),
        (0.7, 0.4, 0.08), (0.7, 0.5, 0.08), (0.7, 0.6, 0.08),
        (0.65, 0.4, 0.12), (0.65, 0.5, 0.12), (0.75, 0.4, 0.12),
        (0.75, 0.5, 0.12), (0.75, 0.6, 0.12),
    ]
    for B_init, ucb, lr in configs:
        r = run_ucb(N_SEEDS, B_init, ucb, lr, 500)
        all_res[f'i={B_init}_u={ucb}_l={lr}_T=500'] = r
        print(f"  i={B_init} ucb={ucb} lr={lr}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    # V14b: T=1000 to check if longer helps
    print("\nLonger horizon T=1000 (same best configs):")
    for B_init, ucb, lr in [(0.7, 0.4, 0.12), (0.7, 0.5, 0.12), (0.75, 0.5, 0.12)]:
        r = run_ucb(N_SEEDS, B_init, ucb, lr, 1000)
        all_res[f'i={B_init}_u={ucb}_l={lr}_T=1000'] = r
        print(f"  i={B_init} ucb={ucb} lr={lr} T=1000: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"\nBEST: {best_key} = {best_val:.4f}")
    print(f"Phase A reference (oracle I): SW=0.1300")
    print(f"Gap: Phase B = {best_val:.4f} vs Phase A = 0.1300 = {best_val/0.13*100:.1f}%")
    
    out = os.path.join(RESULTS, 'clean_split_v14_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()