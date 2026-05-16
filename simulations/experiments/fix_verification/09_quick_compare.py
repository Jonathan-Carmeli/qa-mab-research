"""quick_test.py — minimal, T=200, N=10, 10 seeds"""
import os, sys, json
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/fix_verification'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
TAU_CAP = 5.0

def bq(u_hat, I_hat, N, m, tau):
    q = np.zeros((N*m, N*m))
    for i in range(N):
        for k in range(m): q[i*m+k,i*m+k] = -u_hat[i,k]
    for i in range(N):
        for k in range(m):
            for l in range(m):
                if k!=l: q[i*m+k,i*m+l] += tau
    if I_hat is not None:
        for i in range(N):
            for j in range(N):
                if i==j: continue
                for ki in range(m):
                    for kj in range(m): q[i*m+ki,j*m+kj] += I_hat[i,ki,j,kj]
    return q

def sa(q, u_hat, N, m, rng, nr, ni):
    size = N*m; best_x, best_e = None, float('inf')
    for r in range(nr):
        x = np.zeros(size)
        if r==0:
            for i in range(N): x[i*m+int(np.argmax(u_hat[i]))] = 1.0
        else:
            for i in range(N): x[i*m+rng.integers(0,m)] = 1.0
        e = float(x @ q @ x)
        if e < best_e: best_e, best_x = e, x.copy()
        T = 2.0*(1+r*0.3)
        for _ in range(ni):
            T *= 0.95
            if T < 1e-10: break
            i = rng.integers(0, N)
            k_old = int(np.argmax(x[i*m:(i+1)*m]))
            k_new = (k_old+1+rng.integers(0,m)) % m
            x[i*m+k_old] = 0.0; x[i*m+k_new] = 1.0
            ne = float(x @ q @ x)
            d = ne-e
            if d < 0 or rng.random() < np.exp(-d/T):
                e = ne
                if e < best_e: best_e, best_x = e, x.copy()
            else: x[i*m+k_new] = 0.0; x[i*m+k_old] = 1.0
    return {i: int(np.argmax(best_x[i*m:(i+1)*m])) for i in range(N)}

def run(N, T, n_seeds, mode, sacfg, label):
    sw_ratios = []
    for si in range(n_seeds):
        seed = BASE_SEED + si*1000 + N
        rng = np.random.default_rng(seed)
        env = NetworkEnvironment(N, m=4, seed=seed, B_scale='uniform', I_scale='moderate')
        opt = float(np.sum(np.max(env.B, axis=1)))
        u_hat = np.full((N,4), 0.75)
        I_hat = np.zeros((N,4,N,4))
        tau = 0.1
        for t in range(T):
            if mode == 'random_first100' and t < 100:
                assign = {i: rng.integers(0,4) for i in range(N)}
            elif mode == 'eps':
                assign = {i: rng.integers(0,4) for i in range(N)} if rng.random()<0.2 else sa(bq(u_hat,I_hat,N,4,tau),u_hat,N,4,rng,**sacfg)
            else:
                assign = sa(bq(u_hat,I_hat,N,4,tau),u_hat,N,4,rng,**sacfg)
            tp = env.compute_throughput(assign)
            for i in range(N): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
            tau = min(tau+0.05, TAU_CAP)
        fs = env.social_welfare(assign)
        sw_ratios.append(fs/opt if opt else 0)
    mr=float(np.mean(sw_ratios)); sr=float(np.std(sw_ratios))
    print(f'{label}: SW={mr:.4f}±{sr:.4f}')
    return mr, sr

results = {}
print('T=200, N=10, 10 seeds')
print('qubo_only_SAweak:', end=' ')
results['qubo_only_SAweak'] = run(10, 200, 10, 'qubo_only', {'nr':8,'ni':15}, 'qubo_only_SAweak')
print('random100_SAweak:', end=' ')
results['random100_SAweak'] = run(10, 200, 10, 'random_first100', {'nr':8,'ni':15}, 'random100_SAweak')
print('eps_SAweak:', end=' ')
results['eps_SAweak'] = run(10, 200, 10, 'eps', {'nr':8,'ni':15}, 'eps_SAweak')

with open(os.path.join(RESULTS,'quick_compare.json'),'w') as f:
    json.dump(results, f, indent=2)
print('Done')
