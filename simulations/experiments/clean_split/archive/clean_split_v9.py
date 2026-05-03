"""
clean_split_v9.py — Very light ε-greedy + better B_hat initialization

V8: ε=0.2, decay=0.0004 → SW=0.0795 (best of V8)
V6 reference: 0.0888 (no epsilon)

Try: very light epsilon + better B_hat init (learn from scratch faster)
Also try: softmax instead of argmax for softer exploration
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_softmax(n_seeds, tau_softmax, B_LR, UCB_C_val):
    """UCB with softmax (Boltzmann) selection instead of argmax."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), 0.5)
        visits = np.zeros((N, M))
        
        for step in range(T):
            assignment = {}
            for i in range(N):
                ucb = UCB_C_val / np.sqrt(visits[i] + 1)
                scores = B_hat[i] + ucb
                # Softmax selection
                exp_scores = np.exp(scores / tau_softmax)
                probs = exp_scores / np.sum(exp_scores)
                assignment[i] = int(rng.choice(M, p=probs))
            tp_actual = env.compute_throughput(assignment)
            for i in range(N):
                k = assignment[i]
                B_hat[i, k] += B_LR * (tp_actual[i] - B_hat[i, k])
                visits[i, k] += 1
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), 0.5); vis2 = np.zeros((N, M))
        sw_last50 = []
        for step in range(T):
            a = {}
            for i in range(N):
                ucb = UCB_C_val / np.sqrt(vis2[i] + 1)
                sc = B_h2[i] + ucb
                exp_sc = np.exp(sc / tau_softmax)
                pr = exp_sc / np.sum(exp_sc)
                a[i] = int(rng.choice(M, p=pr))
            tp = env.compute_throughput(a)
            for i in range(N):
                k = a[i]
                B_h2[i, k] += B_LR * (tp[i] - B_h2[i, k])
                vis2[i, k] += 1
            if step >= T - 50:
                sw_last50.append(float(sum(tp.values())))
        sw_ratios.append(float(np.mean(sw_last50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def run_ucb_heavy(n_seeds, UCB_C_val, B_LR):
    """UCB-Greedy with higher UCB_C for more exploration."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), 0.5)
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
        B_h2 = np.full((N, M), 0.5); vis2 = np.zeros((N, M))
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
    print("V9 — Softmax exploration + heavy UCB sweep")
    print("=" * 70)
    all_res = {'v6_ref': 0.0888}
    
    print("\nSoftmax temperature sweep:")
    for tau in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        r = run_softmax(N_SEEDS, tau, 0.12, 0.5)
        all_res[f'softmax_tau={tau}'] = r
        print(f"  tau={tau}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    print("\nHeavy UCB sweep:")
    for ucb in [1.0, 1.5, 2.0, 2.5, 3.0]:
        r = run_ucb_heavy(N_SEEDS, ucb, 0.12)
        all_res[f'ucb={ucb}'] = r
        print(f"  UCB={ucb}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"\nBEST: {best_key} = {best_val:.4f}  (ref: V6=0.0888)")
    
    out = os.path.join(RESULTS, 'clean_split_v9_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
