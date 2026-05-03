"""
clean_split_v12.py — UCB-Greedy + collision avoidance

V6: UCB-Greedy = SW=0.0888
V10: B_init=0.7 → SW=0.0952
V11: confirmed B_init=0.7, ucb=0.5, lr=0.12 = 0.0952

Key insight: SA+QUBO fails because when I_hat is wrong, it creates
bad incentives. UCB-Greedy ignores interference and just focuses on
observed tp — which is more robust.

New approach: Add lightweight collision avoidance to UCB-Greedy.
When two agents pick same route, penalize both in B_hat for next step.
This is simpler than QUBO and doesn't depend on learned I_hat.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def run_ucb_collision_penalty(n_seeds, B_init, UCB_C_val, B_LR, collision_penalty, B_LR_penalty):
    """UCB-Greedy + collision penalty: when agents share route, reduce their B_hat."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        collision_count = np.zeros(N)
        
        for step in range(T):
            assignment = {}
            for i in range(N):
                scores = B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                assignment[i] = int(np.argmax(scores))
            
            tp_actual = env.compute_throughput(assignment)
            
            # Detect same-route collisions
            route_to_agents = {}
            for i in range(N):
                k = assignment[i]
                if k not in route_to_agents:
                    route_to_agents[k] = []
                route_to_agents[k].append(i)
            
            for k, agents in route_to_agents.items():
                if len(agents) > 1:
                    for i in agents:
                        collision_count[i] += 1
                        for l in range(M):
                            B_hat[i, l] -= collision_penalty * (1.0 / (step + 1))
                            B_hat[i, l] = max(0.0, B_hat[i, l])
            
            for i in range(N):
                k = assignment[i]
                B_hat[i, k] += B_LR * (tp_actual[i] - B_hat[i, k])
                visits[i, k] += 1
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M)); cc2 = np.zeros(N)
        sw_last50 = []
        for step in range(T):
            a = {}
            for i in range(N):
                s = B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)
                a[i] = int(np.argmax(s))
            tp = env.compute_throughput(a)
            r2a = {}
            for i in range(N):
                k = a[i]
                if k not in r2a: r2a[k] = []
                r2a[k].append(i)
            for k, ag in r2a.items():
                if len(ag) > 1:
                    for i in ag:
                        cc2[i] += 1
                        for l in range(M):
                            B_h2[i, l] -= collision_penalty * (1.0 / (step + 1))
                            B_h2[i, l] = max(0.0, B_h2[i, l])
            for i in range(N):
                k = a[i]
                B_h2[i, k] += B_LR * (tp[i] - B_h2[i, k])
                vis2[i, k] += 1
            if step >= T - 50:
                sw_last50.append(float(sum(tp.values())))
        sw_ratios.append(float(np.mean(sw_last50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def run_ucb_softmax_collision(n_seeds, B_init, UCB_C_val, B_LR, collision_penalty):
    """UCB-Greedy + collision penalty + softmax for tiebreaking."""
    sw_ratios = []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        
        for step in range(T):
            route_counts = np.zeros(M)
            assignment = {}
            for i in range(N):
                scores = B_hat[i] + UCB_C_val / np.sqrt(visits[i] + 1)
                # Subtract route count for load balancing
                scores_adjusted = scores - 0.05 * route_counts
                assignment[i] = int(np.argmax(scores_adjusted))
                route_counts[assignment[i]] += 1
            
            tp_actual = env.compute_throughput(assignment)
            
            # Collision penalty: if route used multiple times, reduce B_hat for those agents
            route_agents = {}
            for i in range(N):
                k = assignment[i]
                if k not in route_agents:
                    route_agents[k] = []
                route_agents[k].append(i)
            
            for k, agents in route_agents.items():
                if len(agents) > 1:
                    penalty = collision_penalty / len(agents)
                    for i in agents:
                        for l in range(M):
                            B_hat[i, l] -= penalty
            
            for i in range(N):
                k = assignment[i]
                B_hat[i, k] += B_LR * (tp_actual[i] - B_hat[i, k])
                visits[i, k] += 1
                for l in range(M):
                    B_hat[i, l] = max(0.0, B_hat[i, l])
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M))
        sw_last50 = []
        for step in range(T):
            rc = np.zeros(M); a = {}
            for i in range(N):
                s = B_h2[i] + UCB_C_val / np.sqrt(vis2[i] + 1)
                sa = s - 0.05 * rc
                a[i] = int(np.argmax(sa))
                rc[a[i]] += 1
            tp = env.compute_throughput(a)
            ra = {}
            for i in range(N):
                k = a[i]
                if k not in ra: ra[k] = []
                ra[k].append(i)
            for k, ag in ra.items():
                if len(ag) > 1:
                    p = collision_penalty / len(ag)
                    for i in ag:
                        for l in range(M): B_h2[i, l] -= p
            for i in range(N):
                k = a[i]
                B_h2[i, k] += B_LR * (tp[i] - B_h2[i, k])
                vis2[i, k] += 1
                for l in range(M): B_h2[i, l] = max(0.0, B_h2[i, l])
            if step >= T - 50:
                sw_last50.append(float(sum(tp.values())))
        sw_ratios.append(float(np.mean(sw_last50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V12 — Collision avoidance + route load balancing")
    print("=" * 70)
    all_res = {'v6_ref': 0.0888, 'v10_best': 0.0952}
    
    # V12a: collision penalty variant
    print("\nCollision penalty sweep (B_init=0.7, ucb=0.5, lr=0.12):")
    for cp in [0.01, 0.02, 0.05, 0.1, 0.2]:
        r = run_ucb_collision_penalty(N_SEEDS, 0.7, 0.5, 0.12, cp, 0.12)
        all_res[f'coll_pen={cp}'] = r
        print(f"  coll_pen={cp}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    # V12b: load balancing + collision penalty
    print("\nLoad balance + collision penalty:")
    for cp in [0.02, 0.05, 0.1]:
        r = run_ucb_softmax_collision(N_SEEDS, 0.7, 0.5, 0.12, cp)
        all_res[f'load_bal_cp={cp}'] = r
        print(f"  load_bal coll_pen={cp}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    # V12c: Best V10 config + collision penalty
    print("\nBest V10 config (B_init=0.7, ucb=0.5, lr=0.12) + collision penalty:")
    for cp in [0.02, 0.05, 0.1]:
        r = run_ucb_collision_penalty(N_SEEDS, 0.7, 0.5, 0.12, cp, 0.12)
        all_res[f'v10+cp={cp}'] = r
        print(f"  v10+cp={cp}: SW={r['sw_ratio']:.4f} +/- {r['sw_std']:.4f}")
    
    best_key = max(all_res.keys(), key=lambda k: all_res[k]['sw_ratio'] if isinstance(all_res[k], dict) else all_res[k])
    best_val = all_res[best_key]['sw_ratio'] if isinstance(all_res[best_key], dict) else all_res[best_key]
    print(f"\nBEST: {best_key} = {best_val:.4f}  (ref: V6=0.0888, V10=0.0952)")
    
    out = os.path.join(RESULTS, 'clean_split_v12_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()