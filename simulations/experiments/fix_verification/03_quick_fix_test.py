"""
quick_fix_test — all configs N=10 + N=15, T=500, 20 seeds, SA-weak + SA-medium"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/fix_verification'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
TAU_CAP = 5.0
I_CAP = 0.3
T = 500
N_SEEDS = 20
N_VALS = [10, 15]


def build_qubo(u_hat, I_hat, N, m, tau):
    size = N*m
    q = np.zeros((size, size))
    for i in range(N):
        for k in range(m):
            q[i*m+k, i*m+k] = -u_hat[i,k]
    for i in range(N):
        for k in range(m):
            for l in range(m):
                if k != l: q[i*m+k, i*m+l] += tau
    if I_hat is not None:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                for ki in range(m):
                    for kj in range(m):
                        q[i*m+ki, j*m+kj] += I_hat[i,ki,j,kj]
    return q

def sa(q, u_hat, N, m, rng, *, nr, ni):
    size = N*m
    best_x, best_e = None, float('inf')
    for r in range(nr):
        x = np.zeros(size)
        if r == 0:
            for i in range(N): x[i*m + int(np.argmax(u_hat[i]))] = 1.0
        else:
            for i in range(N): x[i*m + rng.integers(0, m)] = 1.0
        e = float(x @ q @ x)
        if e < best_e: best_e, best_x = e, x.copy()
        T = 2.0*(1+r*0.3)
        for _ in range(ni):
            T *= 0.95
            if T < 1e-10: break
            i = rng.integers(0, N)
            k_old = int(np.argmax(x[i*m:(i+1)*m]))
            k_new = (k_old + rng.integers(1, m)) % m
            x[i*m+k_old] = 0.0; x[i*m+k_new] = 1.0
            ne = float(x @ q @ x)
            d = ne - e
            if d < 0 or rng.random() < np.exp(-d/T):
                e = ne
                if e < best_e: best_e, best_x = e, x.copy()
            else:
                x[i*m+k_new] = 0.0; x[i*m+k_old] = 1.0
    return {i: int(np.argmax(best_x[i*m:(i+1)*m])) for i in range(N)}


def run_exp(N, T, n_seeds, learn_fn, sa_cfg):
    sw_ratios, sw_finals, u_errs = [], [], []
    for si in range(n_seeds):
        seed = BASE_SEED + si*1000 + N
        rng = np.random.default_rng(seed)
        env = NetworkEnvironment(N, m=4, seed=seed, B_scale='uniform', I_scale='moderate')
        opt = float(np.sum(np.max(env.B, axis=1)))
        u_hat = np.full((N, 4), 0.75)
        has_I = learn_fn.__name__ not in ('learn_noI', 'learn_FixAonly')
        I_hat = np.zeros((N,4,N,4)) if has_I else None
        tau = 0.1
        state = {}
        for t in range(T):
            q = build_qubo(u_hat, I_hat, N, 4, tau)
            assign = sa(q, u_hat, N, 4, rng, **sa_cfg)
            tp = env.compute_throughput(assign)
            u_hat, I_hat = learn_fn(env, u_hat, I_hat, assign, tp, state)
            tau = min(tau+0.05, TAU_CAP)
        fs = env.social_welfare(assign)
        sw_ratios.append(fs/opt if opt else 0)
        sw_finals.append(fs)
        u_errs.append(float(np.linalg.norm(u_hat - env.B)))
    return (float(np.mean(sw_ratios)), float(np.std(sw_ratios)),
            float(np.mean(sw_finals)), float(np.mean(u_errs)))


def learn_baseline(env, u_hat, I_hat, assign, tp, state):
    N = env.N
    prev = state.get('prev'); ptp = state.get('ptp', {})
    if prev and ptp:
        for i in range(N):
            ki = assign[i]
            di = max(0.0, u_hat[i,ki] - ptp.get(i, 0.0))
            for j in range(N):
                if i==j: continue
                kj = assign[j]
                dj = max(0.0, u_hat[j,kj] - ptp.get(j, 0.0))
                if di>0.01 and dj>0.01:
                    I_hat[i,ki,j,kj] = min(I_hat[i,ki,j,kj]+0.05, I_CAP)
                    I_hat[j,kj,i,ki] = min(I_hat[j,kj,i,ki]+0.05, I_CAP)
    for i in range(N): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
    state['prev'] = assign.copy()
    state['ptp'] = {i: tp[i] for i in range(N)}
    return u_hat, I_hat


def learn_noI(env, u_hat, I_hat, assign, tp, state):
    for i in range(env.N): u_hat[i,assign[i]] += 0.2*(tp[i]-u_hat[i,assign[i]])
    return u_hat, I_hat


def learn_FixAB(env, u_hat, I_hat, assign, tp, state):
    N = env.N
    prev = state.get('prev'); ptp = state.get('ptp', {})
    if prev and ptp:
        for i in range(N):
            ki = assign[i]
            di = max(0.0, u_hat[i,ki] - ptp.get(i, 0.0))
            for j in range(N):
                if i==j: continue
                kj = assign[j]
                dj = max(0.0, u_hat[j,kj] - ptp.get(j, 0.0))
                if di>0.01 and dj>0.01:
                    I_hat[i,ki,j,kj] = min(I_hat[i,ki,j,kj]+0.05, I_CAP)
                    I_hat[j,kj,i,ki] = min(I_hat[j,kj,i,ki]+0.05, I_CAP)
    for i in range(N):
        k = assign[i]
        est_I = sum(I_hat[i,k,j,assign[j]] for j in range(N) if j != i)
        u_hat[i,k] += 0.2*((tp[i]+est_I) - u_hat[i,k])
    state['prev'] = assign.copy()
    state['ptp'] = {i: tp[i] for i in range(N)}
    return u_hat, I_hat


def learn_FixABC(env, u_hat, I_hat, assign, tp, state):
    I_hat *= 0.98
    return learn_FixAB(env, u_hat, I_hat, assign, tp, state)


def learn_FixAonly(env, u_hat, I_hat, assign, tp, state):
    for i in range(env.N):
        k = assign[i]
        u_hat[i,k] += 0.2*(tp[i] - u_hat[i,k])
    return u_hat, I_hat


def main():
    configs = [
        ('Baseline_SAweak',   learn_baseline,  {'nr':8,'ni':15}),
        ('FixC_SAweak',      learn_noI,       {'nr':8,'ni':15}),
        ('FixAB_SAweak',     learn_FixAB,     {'nr':8,'ni':15}),
        ('FixABC_SAweak',    learn_FixABC,    {'nr':8,'ni':15}),
        ('FixAonly_SAweak',  learn_FixAonly,  {'nr':8,'ni':15}),
        ('FixAB_SAmed',      learn_FixAB,    {'nr':50,'ni':200}),
    ]
    print('='*60)
    print('FIX TEST — T=500, seeds=20, tau_cap=5.0')
    print('='*60)
    results = {}
    for name, lfn, sacfg in configs:
        n_key = name.split('_')[1][2:] if 'SAmed' not in name else '10'
        N = 15 if '15' in name else 10
        if 'SAmed' in name: N = 10
        print(f'\n[{name}] N={N}...', end=' ', flush=True)
        mr, sr, mf, mu = run_exp(N, T, N_SEEDS, lfn, sacfg)
        results[name] = {'sw_ratio': mr, 'sw_std': sr, 'sw_final': mf, 'u_err': mu}
        print(f'SW_ratio={mr:.4f}±{sr:.4f}  SW={mf:.4f}  u_err={mu:.2f}')
    out = {'T':T,'seeds':N_SEEDS,'results': results}
    with open(os.path.join(RESULTS,'quick_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {RESULTS}/quick_results.json')


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'\nTotal: {time.time()-t0:.1f}s')
