"""ucb_fast.py — focused: UCB + SA-medium + noI + random, T=500, N=10, 20 seeds"""
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

def run(N, T, n_seeds, method, sacfg):
    sw_ratios = []
    for si in range(n_seeds):
        seed = BASE_SEED + si*1000 + N
        rng = np.random.default_rng(seed)
        env = NetworkEnvironment(N, m=4, seed=seed, B_scale='uniform', I_scale='moderate')
        opt = float(np.sum(np.max(env.B, axis=1)))
        u_hat = np.full((N,4), 0.75)
        I_hat = np.zeros((N,4,N,4))
        tau = 0.1
        visits = np.zeros((N,4))
        for t in range(T):
            if method == 'baseline':
                assign = sa(bq(u_hat, I_hat, N, 4, tau), u_hat, N, 4, rng, **sacfg)
            elif method == 'ucb':
                q = np.zeros((N*4, N*4))
                for i in range(N):
                    for k in range(4):
                        q[i*4+k,i*4+k] = -u_hat[i,k] + 0.5/np.sqrt(visits[i,k]+1)
                for i in range(N):
                    for k in range(4):
                        for l in range(4):
                            if k!=l: q[i*4+k,i*4+l] += tau
                assign = sa(q, u_hat, N, 4, rng, **sacfg)
            elif method == 'ucb_noI':
                q = np.zeros((N*4, N*4))
                for i in range(N):
                    for k in range(4):
                        q[i*4+k,i*4+k] = -u_hat[i,k] + 0.5/np.sqrt(visits[i,k]+1)
                for i in range(N):
                    for k in range(4):
                        for l in range(4):
                            if k!=l: q[i*4+k,i*4+l] += tau
                assign = sa(q, u_hat, N, 4, rng, **sacfg)
                I_hat = None
            elif method == 'softmax':
                probs = np.zeros((N,4))
                for i in range(N):
                    w = u_hat[i] / max(tau, 0.1)
                    w -= np.max(w)
                    p = np.exp(w)
                    probs[i] = p / np.sum(p)
                assign = {}
                for i in range(N):
                    assign[i] = int(rng.choice(4, p=probs[i]))
            elif method == 'random':
                assign = {i: rng.integers(0,4) for i in range(N)}
            else:
                assign = sa(bq(u_hat, I_hat, N, 4, tau), u_hat, N, 4, rng, **sacfg)

            tp = env.compute_throughput(assign)
            for i in range(N): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
            visits[i,assign[i]] += 1
            tau = min(tau+0.05, TAU_CAP)
        fs = env.social_welfare(assign)
        sw_ratios.append(fs/opt if opt else 0)
    return float(np.mean(sw_ratios)), float(np.std(sw_ratios))

results = {}
N=10; T=500; n=20
sacfg_w = {'nr':8,'ni':15}
sacfg_m = {'nr':50,'ni':200}
print(f'N={N} T={T} seeds={n}')
print('='*55)
for name, sacfg in [
    ('baseline_SAweak',    sacfg_w),
    ('ucb_SAweak',         sacfg_w),
    ('ucb_SAmed',          sacfg_m),
    ('ucb_noI_SAweak',     sacfg_w),
    ('ucb_noI_SAmed',      sacfg_m),
    ('softmax_SAweak',    sacfg_w),
    ('random',             sacfg_w),
]:
    mr, sr = run(N, T, n, name.split('_')[0].lower(), sacfg)
    results[name] = {'sw_ratio':mr, 'sw_std':sr}
    print(f'{name}: SW={mr:.4f}±{sr:.4f}')
with open(os.path.join(RESULTS,'ucb_fast.json'),'w') as f:
    json.dump(results, f, indent=2)
print('Done')
