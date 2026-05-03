"""
clean_split_v8.py — ε-greedy decay for exploration
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_egreedy(n_seeds, eps_start, eps_decay, B_LR, UCB_C_val):
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), 0.5)
        visits = np.zeros((N, M))
        _prev_x = None; _prev_tp = None
        
        for step in range(T):
            eps = max(0.0, eps_start - eps_decay * step)
            assignment = {}
            for i in range(N):
                if rng.random() < eps:
                    assignment[i] = rng.integers(0, M)
                else:
                    scores = B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                    assignment[i] = int(np.argmax(scores))
            tp_actual = env.compute_throughput(assignment)
            for i in range(N):
                k = assignment[i]
                B_hat[i, k] += B_LR * (tp_actual[i] - B_hat[i, k])
                visits[i, k] += 1
            _prev_x = assignment.copy()
            _prev_tp = {i: tp_actual[i] for i in range(N)}
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), 0.5); vis2 = np.zeros((N, M))
        px2 = None; pt2 = None; sw_last50 = []
        for step in range(T):
            eps = max(0.0, eps_start - eps_decay * step)
            a = {}
            for i in range(N):
                if rng.random() < eps:
                    a[i] = rng.integers(0, M)
                else:
                    s = B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)
                    a[i] = int(np.argmax(s))
            tp = env.compute_throughput(a)
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
    print("V8 — epsilon-greedy decay sweep")
    print("=" * 70)
    all_res = {'v6_ref': 0.0888, 'v7_best': 0.0877}
    
    configs = [
        (0.5, 0.001), (0.3, 0.0006), (0.2, 0.0004),
        (0.4, 0.0008), (0.3, 0.001),
    ]
    for eps_start, eps_decay in configs:
        r = run_egreedy(N_SEEDS, eps_start, eps_decay, 0.12, 0.5)
        all_res[f'eps({eps_start},{eps_decay})'] = r
        print(f"  eps=({eps_start},{eps_decay}): SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    print("\nB_LR sweep with best epsilon:")
    best_eps = (0.4, 0.0008)
    for B_LR in [0.08, 0.1, 0.12, 0.15, 0.2]:
        r = run_egreedy(N_SEEDS, best_eps[0], best_eps[1], B_LR, 0.5)
        all_res[f'B_LR={B_LR}'] = r
        print(f"  B_LR={B_LR}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"\nBEST: {best_key} = {best_val:.4f}")
    
    out = os.path.join(RESULTS, 'clean_split_v8_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
