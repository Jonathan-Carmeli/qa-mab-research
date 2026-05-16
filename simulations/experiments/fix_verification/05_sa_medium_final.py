"""
sa_medium_final.py — the real test
SA-medium (50x200) + tau_cap=5 + correct u_hat target
All with real NetworkEnvironment
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/fix_verification'
os.makedirs(RESULTS, exist_ok=True)
BASE_SEED = 2026
TAU_CAP = 5.0
T = 1000
N_SEEDS = 20

def build_qubo(u_hat, I_hat, N, m, tau):
    size = N*m
    q = np.zeros((size, size))
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

def sa(q, u_hat, N, m, rng, *, nr, ni):
    size = N*m
    best_x, best_e = None, float('inf')
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

def run_exp(N, T, n_seeds, learn_fn, sacfg, label):
    sw_ratios, sw_finals, u_errs = [], [], []
    for si in range(n_seeds):
        seed = BASE_SEED + si*1000 + N
        rng = np.random.default_rng(seed)
        env = NetworkEnvironment(N, m=4, seed=seed, B_scale='uniform', I_scale='moderate')
        opt = float(np.sum(np.max(env.B, axis=1)))
        u_hat = np.full((N,4), 0.75)
        has_I = learn_fn.__name__ not in ('l_noI','l_Aonly')
        I_hat = np.zeros((N,4,N,4)) if has_I else None
        tau = 0.1
        state = {}
        for t in range(T):
            q = build_qubo(u_hat, I_hat, N, 4, tau)
            assign = sa(q, u_hat, N, 4, rng, **sacfg)
            tp = env.compute_throughput(assign)
            u_hat, I_hat = learn_fn(env, u_hat, I_hat, assign, tp, state)
            tau = min(tau+0.05, TAU_CAP)
        fs = env.social_welfare(assign)
        sw_ratios.append(fs/opt if opt else 0)
        sw_finals.append(fs)
        u_errs.append(float(np.linalg.norm(u_hat - env.B)))
    mr=float(np.mean(sw_ratios)); sr=float(np.std(sw_ratios))
    mf=float(np.mean(sw_finals)); mu=float(np.mean(u_errs))
    print(f'  [{label}] N={N}: SW_ratio={mr:.4f} SW={mf:.4f} u_err={mu:.2f}')
    return mr, sr, mf, mu

def l_noI(env, u_hat, I_hat, assign, tp, state):
    for i in range(env.N): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
    return u_hat, I_hat

def l_Aonly(env, u_hat, I_hat, assign, tp, state):
    for i in range(env.N): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
    return u_hat, I_hat

def l_baseline(env, u_hat, I_hat, assign, tp, state):
    N = env.N
    prev = state.get('prev'); ptp = state.get('ptp',{})
    if prev and ptp:
        for i in range(N):
            ki=assign[i]; di=max(0.0, u_hat[i,ki]-ptp.get(i,0.0))
            for j in range(N):
                if i==j: continue
                kj=assign[j]; dj=max(0.0, u_hat[j,kj]-ptp.get(j,0.0))
                if di>0.01 and dj>0.01:
                    I_hat[i,ki,j,kj] = min(I_hat[i,ki,j,kj]+0.05, 0.3)
                    I_hat[j,kj,i,ki] = min(I_hat[j,kj,i,ki]+0.05, 0.3)
    for i in range(N): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
    state['prev']=assign.copy(); state['ptp']={i:tp[i] for i in range(N)}
    return u_hat, I_hat

def main():
    print('='*60)
    print('SA-MEDIUM FINAL TEST — tau_cap=5.0')
    print('='*60)
    results = {}
    configs = [
        ('noI_SAmed',     l_noI,     {'nr':50,'ni':200}),
        ('Aonly_SAmed',   l_Aonly,   {'nr':50,'ni':200}),
        ('baseline_SAmed',l_baseline,{'nr':50,'ni':200}),
    ]
    for N in [10, 15, 20]:
        print(f'\n-- N={N} --')
        for name, lfn, sacfg in configs:
            mr,sr,mf,mu = run_exp(N, T, N_SEEDS, lfn, sacfg, name)
            results[f'{name}_N{N}'] = {'sw_ratio':mr,'sw_std':sr,'sw_final':mf,'u_err':mu}
    with open(os.path.join(RESULTS,'sa_medium_final.json'),'w') as f:
        json.dump(results, f, indent=2)
    print('\nDone')

if __name__ == '__main__':
    t0=time.time()
    main()
    print(f'Total: {time.time()-t0:.0f}s')
