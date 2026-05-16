"""B_learning_test.py — minimal test of counterfactual B learning"""
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

def run(N, T, n_seeds, method, sacfg, label):
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
            assign = sa(bq(u_hat, I_hat, N, 4, tau), u_hat, N, 4, rng, **sacfg)
            tp = env.compute_throughput(assign)

            if method == 'current':
                for i in range(N): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
            elif method == 'all_cfactual':
                # Update ALL routes — but only 1 counterfactual per agent (the chosen one + 1 random)
                for i in range(N):
                    chosen_k = assign[i]
                    u_hat[i,chosen_k] += 0.2*(tp[i]-u_hat[i,chosen_k])
                    # one random alternative
                    alts = [k for k in range(4) if k != chosen_k]
                    if alts:
                        alt_k = alts[rng.integers(0, len(alts))]
                        alt = dict(assign); alt[i] = alt_k
                        alt_tp = env.compute_throughput(alt)
                        u_hat[i,alt_k] += 0.2*(alt_tp[i]-u_hat[i,alt_k])
            elif method == 'Bonly_noI':
                # Same as all_cfactual but no I_hat
                for i in range(N):
                    chosen_k = assign[i]
                    u_hat[i,chosen_k] += 0.2*(tp[i]-u_hat[i,chosen_k])
                    alts = [k for k in range(4) if k != chosen_k]
                    if alts:
                        alt_k = alts[rng.integers(0, len(alts))]
                        alt = dict(assign); alt[i] = alt_k
                        alt_tp = env.compute_throughput(alt)
                        u_hat[i,alt_k] += 0.2*(alt_tp[i]-u_hat[i,alt_k])
                I_hat = None

            tau = min(tau+0.05, TAU_CAP)
        fs = env.social_welfare(assign)
        sw_ratios.append(fs/opt if opt else 0)
    mr=float(np.mean(sw_ratios)); sr=float(np.std(sw_ratios))
    return mr, sr

results = {}
sacfg = {'nr':8,'ni':15}
sacfg_med = {'nr':50,'ni':200}
N=10; T=500; n=15

print('B_LEARNING TEST — N=10, T=500, 15 seeds')
print('='*50)
for name, method, cfg in [
    ('current_SAweak',  'current',     sacfg),
    ('all_cfactual_SW', 'all_cfactual',sacfg),
    ('Bonly_noI_SW',   'Bonly_noI',   sacfg),
    ('current_SAmed',   'current',     sacfg_med),
    ('all_cfactual_SM', 'all_cfactual',sacfg_med),
    ('Bonly_noI_SM',   'Bonly_noI',   sacfg_med),
]:
    mr, sr = run(N, T, n, method, cfg, name)
    results[name] = {'sw_ratio':mr, 'sw_std':sr}
    print(f'{name}: SW={mr:.4f}±{sr:.4f}')

with open(os.path.join(RESULTS,'B_learning.json'),'w') as f:
    json.dump(results, f, indent=2)
print('Done')
