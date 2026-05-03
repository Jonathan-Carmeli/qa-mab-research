"""
clean_split_v13.py — Hybrid: UCB-Greedy for route selection + collision inference for I_hat

Problem: SA+QUBO with wrong I_hat creates bad incentives.
UCB-Greedy ignores interference → ceiling at ~0.095 (73% of Phase A).

Goal: Get past 0.095 by learning interference through collision inference
WITHOUT using I_hat in the route selection QUBO.

Approach:
1. UCB-Greedy for route selection (proven to work)
2. Simultaneously learn I_hat from collisions detected via throughput drops
3. In NEXT step, use I_hat to penalize routes that caused collisions
4. But the penalty is applied to the SCORE, not baked into a QUBO

The key: collision detection should be sharper. If two agents are on the
same route and BOTH see reduced throughput → collision signal.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_ucb_with_I_penalty(n_seeds, B_init, UCB_C_val, B_LR, I_LR, use_I_penalty):
    """UCB-Greedy + I_hat learned from collisions + route penalty based on I_hat."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        I_hat = np.zeros((N, M, N, M))
        visits = np.zeros((N, M))
        _prev_x = None; _prev_tp = None
        
        for step in range(T):
            route_counts = np.zeros(M)
            assignment = {}
            for i in range(N):
                scores = B_hat[i].copy()
                
                if use_I_penalty and _prev_x is not None:
                    # Penalize routes based on learned I_hat from previous step
                    for j in range(N):
                        if j != i:
                            for l in range(M):
                                penalty = I_hat[i, :, j, l].max() * 0.5
                                scores[l] -= penalty
                
                scores_adjusted = scores + UCB_C_val / np.sqrt(visits[i] + 1)
                
                if use_I_penalty:
                    scores_adjusted = scores_adjusted - 0.03 * route_counts
                
                assignment[i] = int(np.argmax(scores_adjusted))
                route_counts[assignment[i]] += 1
            
            tp_actual = env.compute_throughput(assignment)
            
            # Learn I_hat from throughput drops (collision inference)
            if _prev_x is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = _prev_x[i]; kj = _prev_x[j]
                        drop_i = max(0.0, B_hat[i, ki] - _prev_tp[i])
                        drop_j = max(0.0, B_hat[j, kj] - _prev_tp[j])
                        if drop_i > 0.02 and drop_j > 0.02:
                            I_hat[i, ki, j, kj] = min(I_hat[i, ki, j, kj] + I_LR, 0.5)
                            I_hat[j, kj, i, ki] = min(I_hat[j, kj, i, ki] + I_LR, 0.5)
                I_hat *= (1.0 - 0.001)
                for i in range(N): I_hat[i, :, i, :] = 0.0
            
            for i in range(N):
                k = assignment[i]
                B_hat[i, k] += B_LR * (tp_actual[i] - B_hat[i, k])
                visits[i, k] += 1
            
            _prev_x = assignment.copy()
            _prev_tp = {i: tp_actual[i] for i in range(N)}
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M))
        I_h2 = np.zeros((N, M, N, M))
        px2 = None; pt2 = None; sw_last50 = []
        for step in range(T):
            rc = np.zeros(M); a = {}
            for i in range(N):
                sc = B_h2[i].copy()
                if use_I_penalty and px2 is not None:
                    for j in range(N):
                        if j != i:
                            for l in range(M):
                                sc[l] -= I_h2[i, :, j, l].max() * 0.5
                sa = sc + UCB_C_val / np.sqrt(vis2[i] + 1)
                if use_I_penalty: sa = sa - 0.03 * rc
                a[i] = int(np.argmax(sa)); rc[a[i]] += 1
            tp = env.compute_throughput(a)
            if px2 is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = px2[i]; kj = px2[j]
                        di = max(0.0, B_h2[i, ki] - pt2[i])
                        dj = max(0.0, B_h2[j, kj] - pt2[j])
                        if di > 0.02 and dj > 0.02:
                            I_h2[i, ki, j, kj] = min(I_h2[i, ki, j, kj] + I_LR, 0.5)
                            I_h2[j, kj, i, ki] = min(I_h2[j, kj, i, ki] + I_LR, 0.5)
                I_h2 *= (1.0 - 0.001)
                for i in range(N): I_h2[i, :, i, :] = 0.0
            for i in range(N):
                k = a[i]
                B_h2[i, k] += B_LR * (tp[i] - B_h2[i, k])
                vis2[i, k] += 1
            px2 = a.copy(); pt2 = {i: tp[i] for i in range(N)}
            if step >= T - 50:
                sw_last50.append(float(sum(tp.values())))
        sw_ratios.append(float(np.mean(sw_last50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V13 — UCB-Greedy + I_penalty from collision inference")
    print("=" * 70)
    all_res = {'v6_ref': 0.0888, 'v10_best': 0.0952}
    
    # V13a: Best V10 config + I_penalty
    print("\nV10 best (init=0.7, ucb=0.5, lr=0.12) + I penalty:")
    for I_LR in [0.05, 0.1, 0.15, 0.2]:
        r = run_ucb_with_I_penalty(N_SEEDS, 0.7, 0.5, 0.12, I_LR, True)
        all_res[f'v10+I_Ilr={I_LR}'] = r
        print(f"  I_LR={I_LR}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    # V13b: Compare with and without I_penalty
    print("\nWith vs without I_penalty (B_init=0.7, ucb=0.5, lr=0.12, I_LR=0.1):")
    r_no = run_ucb_with_I_penalty(N_SEEDS, 0.7, 0.5, 0.12, 0.1, False)
    r_yes = run_ucb_with_I_penalty(N_SEEDS, 0.7, 0.5, 0.12, 0.1, True)
    all_res['no_I_penalty'] = r_no; all_res['with_I_penalty'] = r_yes
    print(f"  No I_penalty: SW={r_no['sw_ratio']:.4f}")
    print(f"  With I_penalty: SW={r_yes['sw_ratio']:.4f}")
    
    # V13c: Sweep B_init around 0.7 with I_penalty
    print("\nB_init sweep with I_penalty (ucb=0.5, lr=0.12, I_LR=0.1):")
    for B_init in [0.6, 0.65, 0.7, 0.75, 0.8]:
        r = run_ucb_with_I_penalty(N_SEEDS, B_init, 0.5, 0.12, 0.1, True)
        all_res[f'init={B_init}_Ipenalty'] = r
        print(f"  init={B_init}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"\nBEST: {best_key} = {best_val:.4f}  (V6=0.0888, V10=0.0952)")
    
    out = os.path.join(RESULTS, 'clean_split_v13_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()