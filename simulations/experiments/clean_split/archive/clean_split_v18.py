"""
clean_split_v18.py — D-Wave SA for QUBO (no token needed for SA)
dimod's SimulatedAnnealingSampler — fast C implementation
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment
from dimod import SimulatedAnnealingSampler

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 500; N_SEEDS = 20


def qubo_to_bqm(Q, n_vars):
    """Convert QUBO dict to BinaryQuadraticModel for dimod."""
    from dimod import BinaryQuadraticModel
    bqm = BinaryQuadraticModel('BINARY')
    for (i, j), v in Q.items():
        if i == j:
            bqm.add_variable(i, v)
        else:
            bqm.add_interaction(i, j, v)
    return bqm


def run_phase(n_seeds, use_oracle_I, B_init, B_LR, UCB_C_val):
    sw_ratios = []
    sampler = SimulatedAnnealingSampler()
    
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        
        for step in range(T):
            Q = {}
            for i in range(N):
                for k in range(M):
                    idx = i * M + k
                    Q[(idx, idx)] = -B_hat[i, k] - UCB_C_val / math.sqrt(visits[i, k] + 1)
            
            for i in range(N):
                for k in range(M):
                    for l in range(k + 1, M):
                        ki, kj = i * M + k, i * M + l
                        Q[(ki, kj)] = Q.get((ki, kj), 0) + 0.5
                        Q[(kj, ki)] = Q.get((kj, ki), 0) + 0.5
            
            if use_oracle_I:
                for i in range(N):
                    for k in range(M):
                        for j in range(N):
                            if j == i: continue
                            for l in range(M):
                                ki, kj = i * M + k, j * M + l
                                Q[(ki, kj)] = Q.get((ki, kj), 0) + env.I[i, k, j, l] * 0.5
            else:
                I_hat = np.zeros((N, M, N, M))
                for i in range(N):
                    for k in range(M):
                        for j in range(N):
                            if j == i: continue
                            for l in range(M):
                                ki, kj = i * M + k, j * M + l
                                Q[(ki, kj)] = Q.get((ki, kj), 0) + I_hat[i, k, j, l] * 0.5
            
            bqm = qubo_to_bqm(Q, N * M)
            response = sampler.sample(bqm, num_reads=1)
            best = {int(k): v for k, v in response.first.sample.items()}
            
            a = {}
            for i in range(N):
                route_vals = {k: best.get(i * M + k, 0) for k in range(M)}
                a[i] = int(max(route_vals, key=route_vals.get))
            
            tp = env.compute_throughput(a)
            for i in range(N):
                k = a[i]
                if use_oracle_I:
                    I_sum = sum(env.I[i, k, j, a[j]] for j in range(N) if j != i)
                    B_hat[i, k] += B_LR * ((tp[i] + I_sum) - B_hat[i, k])
                else:
                    B_hat[i, k] += B_LR * (tp[i] - B_hat[i, k])
                visits[i, k] += 1
        
        opt = float(np.sum(np.max(env.B, axis=1)))
        B_h2 = np.full((N, M), B_init); vis2 = np.zeros((N, M))
        sw50 = []
        for step in range(T):
            Q2 = {}
            for i in range(N):
                for k in range(M):
                    Q2[(i * M + k, i * M + k)] = -B_h2[i, k] - UCB_C_val / math.sqrt(vis2[i, k] + 1)
            for i in range(N):
                for k in range(M):
                    for l in range(k + 1, M):
                        ki, kj = i * M + k, i * M + l
                        Q2[(ki, kj)] = Q2.get((ki, kj), 0) + 0.5
                        Q2[(kj, ki)] = Q2.get((kj, ki), 0) + 0.5
            if use_oracle_I:
                for i in range(N):
                    for k in range(M):
                        for j in range(N):
                            if j == i: continue
                            for l in range(M):
                                ki, kj = i * M + k, j * M + l
                                Q2[(ki, kj)] = Q2.get((ki, kj), 0) + env.I[i, k, j, l] * 0.5
            bqm2 = qubo_to_bqm(Q2, N * M)
            resp2 = sampler.sample(bqm2, num_reads=1)
            b2 = {int(k): v for k, v in resp2.first.sample.items()}
            a2 = {}
            for i in range(N):
                rv = {k: b2.get(i * M + k, 0) for k in range(M)}
                a2[i] = int(max(rv, key=rv.get))
            tp2 = env.compute_throughput(a2)
            for i in range(N):
                k = a2[i]
                if use_oracle_I:
                    Is2 = sum(env.I[i, k, j, a2[j]] for j in range(N) if j != i)
                    B_h2[i, k] += B_LR * ((tp2[i] + Is2) - B_h2[i, k])
                else:
                    B_h2[i, k] += B_LR * (tp2[i] - B_h2[i, k])
                vis2[i, k] += 1
            if step >= T - 50:
                sw50.append(float(sum(tp2.values())))
        sw_ratios.append(float(np.mean(sw50) / opt) if opt > 0 else 0)
    
    return {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios))}


def main():
    print("=" * 70)
    print("V18 — D-Wave SA (dimod) for QUBO")
    print("=" * 70)
    all_res = {'v17_phaseA_N10': 0.1323, 'v17_phaseB_N10': 0.1070}
    
    rA = run_phase(N_SEEDS, True, 0.7, 0.12, 0.5)
    all_res['phaseA_oracle'] = rA
    print(f"\nPhase A (oracle I): SW={rA['sw_ratio']:.4f} +/- {rA['sw_std']:.4f}")
    
    rB = run_phase(N_SEEDS, False, 0.7, 0.12, 0.5)
    all_res['phaseB_learned'] = rB
    print(f"Phase B (learned I): SW={rB['sw_ratio']:.4f} +/- {rB['sw_std']:.4f}")
    
    print(f"\n  Phase A: SW={rA['sw_ratio']:.4f}")
    print(f"  Phase B: SW={rB['sw_ratio']:.4f}")
    print(f"  Ratio: {rB['sw_ratio']/rA['sw_ratio']*100:.1f}%")
    
    with open(os.path.join(RESULTS, 'clean_split_v18_results.json'), 'w') as f:
        json.dump(all_res, f, indent=2)


if __name__ == '__main__':
    main()