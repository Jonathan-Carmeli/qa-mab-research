"""
clean_split_v10.py — Optimistic B_hat initialization + route diversity

V6: UCB-Greedy (UCB_C=0.5, B_LR=0.12) = SW=0.0888
V7-V9: All modifications WORSE than V6.

Key insight: B_hat starts at 0.5. But throughput (tp) can range from -0.5 to +1.0.
If initial B_hat is too low, algorithm explores routes that give high tp but B_hat doesn't catch up fast.

Also: Need to FORCE diversity — if two agents have similar B_hat+UCB scores,
they should pick DIFFERENT routes to avoid collisions.

New approach: 
1. B_hat init = optimistic (0.8) so SA explores routes freely
2. Route-selection adds a small repulsion: if route k chosen by many agents this step, penalize it
3. Use "best tp seen" instead of EMA for more robust B_hat
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_diverse(n_seeds, B_init, use_repulsion, B_LR, UCB_C_val, use_max_instead_of_ema):
    """UCB-Greedy with optional: optimistic init, route repulsion, max-tp B_hat."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        
        if use_max_instead_of_ema:
            B_best = np.full((N, M), B_init)
        else:
            B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        
        for step in range(T):
            route_counts = np.zeros(M)
            assignment = {}
            for i in range(N):
                if use_max_instead_of_ema:
                    scores = B_best[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                else:
                    scores = B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                
                if use_repulsion:
                    for k_check in range(M):
                        scores[k_check] -= 0.1 * route_counts[k_check]
                
                assignment[i] = int(np.argmax(scores))
                route_counts[assignment[i]] += 1
            
            tp_actual = env.compute_throughput(assignment)
            for i in range(N):
                k = assignment[i]
                if use_max_instead_of_ema:
                    B_best[i, k] = max(B_best[i, k], tp_actual[i])
                else:
                    B_hat[i, k] += B_LR * (tp_actual[i] - B_hat[i, k])
                visits[i, k] += 1
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        if use_max_instead_of_ema:
            B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M))
        else:
            B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M))
        sw_last50 = []
        for step in range(T):
            rc = np.zeros(M); a = {}
            for i in range(N):
                if use_max_instead_of_ema:
                    sc = B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)
                else:
                    sc = B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)
                if use_repulsion:
                    for kk in range(M): sc[kk] -= 0.1 * rc[kk]
                a[i] = int(np.argmax(sc))
                rc[a[i]] += 1
            tp = env.compute_throughput(a)
            for i in range(N):
                k = a[i]
                if use_max_instead_of_ema:
                    B_h2[i, k] = max(B_h2[i, k], tp[i])
                else:
                    B_h2[i, k] += B_LR * (tp[i] - B_h2[i, k])
                vis2[i, k] += 1
            if step >= T - 50:
                sw_last50.append(float(sum(tp.values())))
        sw_ratios.append(float(np.mean(sw_last50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V10 — Optimistic init + route diversity + max-tp B_hat")
    print("=" * 70)
    all_res = {'v6_ref': 0.0888}
    
    print("\n1. B_init sweep (no repulsion, no max-tp):")
    for B_init in [0.3, 0.5, 0.7, 0.9, 1.1, 1.5]:
        r = run_diverse(N_SEEDS, B_init, False, 0.12, 0.5, False)
        all_res[f'init={B_init}'] = r
        print(f"  B_init={B_init}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    print("\n2. Route repulsion (B_init=0.5):")
    for B_init in [0.5, 0.8]:
        r = run_diverse(N_SEEDS, B_init, True, 0.12, 0.5, False)
        all_res[f'repulsion_Binit={B_init}'] = r
        print(f"  repulsion B_init={B_init}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    print("\n3. Max-tp instead of EMA (B_init=0.5):")
    for B_init in [0.5, 0.8, 1.0]:
        r = run_diverse(N_SEEDS, B_init, False, 0.12, 0.5, True)
        all_res[f'max_tp_init={B_init}'] = r
        print(f"  max_tp B_init={B_init}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"\nBEST: {best_key} = {best_val:.4f}  (V6 ref: 0.0888)")
    
    out = os.path.join(RESULTS, 'clean_split_v10_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
