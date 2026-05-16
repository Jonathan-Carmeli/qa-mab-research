"""minimal_test.py — just SA-medium vs SA-weak on N=10, no I_hat"""
import os, sys, json
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/fix_verification'
os.makedirs(RESULTS, exist_ok=True)
BASE_SEED = 2026
TAU_CAP = 5.0
T = 500
N_SEEDS = 20

def bqubo(u_hat, I_hat, N, m, tau):
    q = np.zeros((N*m, N*m))
    for i in range(N):
        for k in range(m): q[i*m+k,i*m+k] = -u_hat[i,k]
    for i in range(N):
        for k in range(m):
            for l in range(m):
                if k!=l: q[i*m+k,i*m+l] += tau
    return q

def solve(q, u_hat, N, m, rng, nr, ni):
    best_x, best_e = None, float('inf')
    for r in range(nr):
        x = np.zeros(N*m)
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

print('SA-MEDIUM TEST')
print('='*50)
results = {}
for label, nr, ni in [('SA-weak',8,15),('SA-medium',50,200)]:
    sw_ratios = []
    for si in range(N_SEEDS):
        seed = BASE_SEED + si*1000 + 10
        rng = np.random.default_rng(seed)
        env = NetworkEnvironment(10, m=4, seed=seed, B_scale='uniform', I_scale='moderate')
        opt = float(np.sum(np.max(env.B, axis=1)))
        u_hat = np.full((10,4), 0.75)
        I_hat = np.zeros((10,4,10,4))
        tau = 0.1
        for t in range(T):
            assign = solve(bqubo(u_hat, I_hat, 10, 4, tau), u_hat, 10, 4, rng, nr, ni)
            tp = env.compute_throughput(assign)
            for i in range(10): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
            tau = min(tau+0.05, TAU_CAP)
        fs = env.social_welfare(assign)
        sw_ratios.append(fs/opt if opt else 0)
    mr = float(np.mean(sw_ratios)); sr = float(np.std(sw_ratios))
    print(f'{label}: SW_ratio={mr:.4f}±{sr:.4f}')
    results[label] = {'sw_ratio':mr, 'sw_std':sr}
with open(os.path.join(RESULTS,'minimal.json'),'w') as f:
    json.dump(results, f, indent=2)
print('Done')
