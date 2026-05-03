"""
clean_split_v18_test.py — Quick test of D-Wave SA
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment
from dimod import SimulatedAnnealingSampler

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; T = 100; N_SEEDS = 5


def qubo_to_bqm(Q):
    from dimod import BinaryQuadraticModel
    bqm = BinaryQuadraticModel('BINARY')
    for (i, j), v in Q.items():
        if i == j:
            bqm.add_variable(i, v)
        else:
            bqm.add_interaction(i, j, v)
    return bqm


def main():
    print("V18 — Test with D-Wave SA")
    sampler = SimulatedAnnealingSampler()
    
    # Single seed test
    env = NetworkEnvironment(N, M, seed=BASE_SEED + 42, B_scale='uniform', I_scale='moderate')
    B_hat = np.full((N, M), 0.7)
    visits = np.zeros((N, M))
    
    print(f"Environment: N={N}, M={M}, T={T}")
    print(f"B shape: {env.B.shape}, I shape: {env.I.shape}")
    
    import time
    t0 = time.time()
    
    for step in range(T):
        Q = {}
        for i in range(N):
            for k in range(M):
                idx = i * M + k
                Q[(idx, idx)] = -B_hat[i, k] - 0.5 / math.sqrt(visits[i, k] + 1)
        
        for i in range(N):
            for k in range(M):
                for l in range(k + 1, M):
                    ki, kj = i * M + k, i * M + l
                    Q[(ki, kj)] = Q.get((ki, kj), 0) + 0.5
                    Q[(kj, ki)] = Q.get((kj, ki), 0) + 0.5
        
        for i in range(N):
            for k in range(M):
                for j in range(N):
                    if j == i: continue
                    for l in range(M):
                        ki, kj = i * M + k, j * M + l
                        Q[(ki, kj)] = Q.get((ki, kj), 0) + env.I[i, k, j, l] * 0.5
        
        bqm = qubo_to_bqm(Q)
        response = sampler.sample(bqm, num_reads=1)
        best = {int(k): v for k, v in response.first.sample.items()}
        
        a = {}
        for i in range(N):
            rv = {k: best.get(i * M + k, 0) for k in range(M)}
            a[i] = int(max(rv, key=rv.get))
        
        tp = env.compute_throughput(a)
        for i in range(N):
            k = a[i]
            I_sum = sum(env.I[i, k, j, a[j]] for j in range(N) if j != i)
            B_hat[i, k] += 0.12 * ((tp[i] + I_sum) - B_hat[i, k])
            visits[i, k] += 1
        
        if step % 20 == 0:
            print(f"  Step {step}: SW={sum(tp.values()):.4f}")
    
    t1 = time.time()
    print(f"\nDone in {t1-t0:.1f}s")
    
    opt = float(np.sum(np.max(env.B, axis=1)))
    print(f"Optimal SW: {opt:.4f}")
    print(f"Final SW: {sum(tp.values()):.4f} ({sum(tp.values())/opt*100:.1f}%)")


if __name__ == '__main__':
    main()