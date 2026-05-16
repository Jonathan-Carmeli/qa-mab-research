"""
combined_fix.py — Full QA-MAB fix test: A+B+D+tau_cap

Tests all combinations to find the winning formula:
Baseline | FixTau | FixTauFixC | FixTauFixA | FixTauFixAB | FixTauFixAD | FixTauFixABC

N={10, 15, 20}, T=1000, 20 seeds
"""

import os, sys, json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment
from qa_mab import QAMAB

warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/jon_claw/qa-mab-research/simulations/results/fix_verification'
os.makedirs(RESULTS_DIR, exist_ok=True)

N_VALUES = [10, 15, 20]
T = 1000
N_SEEDS = 20
BASE_SEED = 2026
TAU_CAP = 5.0
I_CAP = 0.3


class QAMABTauCap(QAMAB):
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        tp = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (tp[i] - self.u_hat[i, k])
        if self._prev_x and self._prev_throughputs:
            for i in range(self.N):
                for j in range(i+1, self.N):
                    ki, kj = self._prev_x[i], self._prev_x[j]
                    d_i = max(0.0, self.u_hat[i,ki] - self._prev_throughputs[i])
                    d_j = max(0.0, self.u_hat[j,kj] - self._prev_throughputs[j])
                    if d_i > self.collision_threshold:
                        self.I_hat[i,ki,j,kj] = min(self.I_hat[i,ki,j,kj]+self.I_learn_rate, I_CAP)
                    if d_j > self.collision_threshold:
                        self.I_hat[j,kj,i,ki] = min(self.I_hat[j,kj,i,ki]+self.I_learn_rate, I_CAP)
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: tp[i] for i in tp}
        self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        self.history.append(self.env.social_welfare(assignment))


class QAMABTauCapFixA(QAMABTauCap):
    """Fix A: u_hat targets B (not B-E[I])."""
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        tp = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            est_I = sum(self.I_hat[i,k,j,assignment[j]] for j in range(self.N) if j!=i)
            target = tp[i] + est_I
            self.u_hat[i, k] += self.B_learn_rate * (target - self.u_hat[i, k])
        if self._prev_x and self._prev_throughputs:
            for i in range(self.N):
                for j in range(i+1, self.N):
                    ki, kj = self._prev_x[i], self._prev_x[j]
                    d_i = max(0.0, self.u_hat[i,ki] - self._prev_throughputs[i])
                    d_j = max(0.0, self.u_hat[j,kj] - self._prev_throughputs[j])
                    if d_i > self.collision_threshold:
                        self.I_hat[i,ki,j,kj] = min(self.I_hat[i,ki,j,kj]+self.I_learn_rate, I_CAP)
                    if d_j > self.collision_threshold:
                        self.I_hat[j,kj,i,ki] = min(self.I_hat[j,kj,i,ki]+self.I_learn_rate, I_CAP)
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: tp[i] for i in tp}
        self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        self.history.append(self.env.social_welfare(assignment))


class QAMABTauCapFixB(QAMABTauCap):
    """Fix B: I_hat EMA decay — fade old entries, only boost on collision."""
    def __init__(self, env, **kwargs):
        self.decay = kwargs.pop('I_decay', 0.02)
        super().__init__(env, **kwargs)

    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        tp = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (tp[i] - self.u_hat[i, k])
        # EMA decay
        self.I_hat *= (1.0 - self.decay)
        if self._prev_x and self._prev_throughputs:
            for i in range(self.N):
                for j in range(i+1, self.N):
                    ki, kj = self._prev_x[i], self._prev_x[j]
                    d_i = max(0.0, self.u_hat[i,ki] - self._prev_throughputs[i])
                    d_j = max(0.0, self.u_hat[j,kj] - self._prev_throughputs[j])
                    if d_i > self.collision_threshold:
                        self.I_hat[i,ki,j,kj] = min(self.I_hat[i,ki,j,kj]+self.I_learn_rate, I_CAP)
                    if d_j > self.collision_threshold:
                        self.I_hat[j,kj,i,ki] = min(self.I_hat[j,kj,i,ki]+self.I_learn_rate, I_CAP)
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: tp[i] for i in tp}
        self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        self.history.append(self.env.social_welfare(assignment))


class QAMABTauCapFixC(QAMABTauCap):
    """Fix C: no I_hat at all."""
    def step(self):
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)
        tp = self.env.compute_throughput(assignment)
        for i in range(self.N):
            k = assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (tp[i] - self.u_hat[i, k])
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: tp[i] for i in tp}
        self.tau = min(self.tau + self.delta_tau, TAU_CAP)
        self.history.append(self.env.social_welfare(assignment))


class QAMABTauCapFixAD(QAMABTauCapFixA):
    """Fix A + D (SA-medium)."""
    def solve_qubo(self, Q):
        n, m, sz = self.N, self.m, self.qubo_size
        nr, ni = 50, 200
        T0, dec = 2.0, 0.95
        best_x, best_e = None, float('inf')
        rng = self.rng
        for r in range(nr):
            x = np.zeros(sz)
            for i in range(n):
                x[i*m + int(np.argmax(self.u_hat[i]))] = 1.0
            if r > 0:
                for _ in range(rng.integers(1, max(2, n//3))):
                    i = rng.integers(0, n)
                    k_old = int(np.argmax(x[i*m:(i+1)*m]))
                    cand = [k for k in range(m) if k != k_old]
                    if cand:
                        k_new = cand[rng.integers(0, len(cand))]
                        x[i*m+k_old] = 0.0; x[i*m+k_new] = 1.0
            e = float(x @ Q @ x)
            if e < best_e:
                best_e, best_x = e, x.copy()
            T = T0 * (1.0 + r*0.3)
            for s in range(ni):
                T *= dec
                i = rng.integers(0, n)
                k_old = int(np.argmax(x[i*m:(i+1)*m]))
                k_new = (k_old+1+rng.integers(0, m-1)) % m
                x[i*m+k_old] = 0.0; x[i*m+k_new] = 1.0
                ne = float(x @ Q @ x)
                d = ne - e
                if d < 0 or (T > 1e-10 and rng.random() < np.exp(-d/T)):
                    e = ne
                    if e < best_e:
                        best_e, best_x = e, x.copy()
                else:
                    x[i*m+k_new] = 0.0; x[i*m+k_old] = 1.0
        return {i: int(np.argmax(best_x[i*m:(i+1)*m])) for i in range(n)}


class QAMABTauCapFixABC(QAMABTauCapFixAD):
    """Fix A + B + C (A + EMA decay + SA-medium)."""
    pass


FIXES = {
    'Baseline':       (QAMAB,               {}),
    'FixTau':         (QAMABTauCap,         {}),
    'FixTauFixC':     (QAMABTauCapFixC,     {}),
    'FixTauFixA':     (QAMABTauCapFixA,     {}),
    'FixTauFixAD':    (QAMABTauCapFixAD,    {}),
}


def greedy(env):
    return float(np.sum(np.max(env.B, axis=1)))


def main():
    print("=" * 65)
    print("COMBINED FIX TEST")
    print(f"N={N_VALUES}, T={T}, seeds={N_SEEDS}, TAU_CAP={TAU_CAP}")
    print("=" * 65)

    results = {}

    for N in N_VALUES:
        print(f"\n--- N={N} ---")
        results[N] = {}

        for name, (Cls, kwargs) in FIXES.items():
            print(f"  {name}...", end=' ', flush=True)
            sws, finals, uerrs = [], [], []

            for si in range(N_SEEDS):
                seed = BASE_SEED + si*1000 + N
                env = NetworkEnvironment(N, m=4, seed=seed,
                                        B_scale='uniform', I_scale='moderate')
                opt = greedy(env)

                algo = Cls(env, tau0=0.1, delta_tau=0.05, lambda_=2.0,
                          B_learn_rate=0.2, I_learn_rate=0.05, I_cap=I_CAP,
                          seed=seed, **kwargs)

                for t in range(T):
                    algo.step()

                fs = algo.history[-1]
                sws.append(fs/opt if opt != 0 else 0)
                finals.append(fs)
                uerrs.append(float(np.linalg.norm(algo.u_hat - env.B)))

            sws, finals, uerrs = np.array(sws), np.array(finals), np.array(uerrs)
            mr, sr = float(np.mean(sws)), float(np.std(sws))
            mf = float(np.mean(finals))
            mu = float(np.mean(uerrs))

            results[N][name] = {
                'sw_ratio_mean': mr, 'sw_ratio_std': sr,
                'sw_final_mean': mf, 'u_err_mean': mu
            }
            print(f"SW_ratio={mr:.4f}±{sr:.4f}  SW_final={mf:.4f}  u_err={mu:.2f}")

    out = {'config': {'N':N_VALUES,'T':T,'seeds':N_SEEDS,'tau_cap':TAU_CAP}, 'results':results}
    with open(os.path.join(RESULTS_DIR, 'combined_fix_results.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # Plot
    n_f = len(FIXES)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = list(FIXES.keys())
    colors = ['#e74c3c','#2ecc71','#3498db','#9b59b6','#f39c12'][:n_f]

    for i, N in enumerate(N_VALUES):
        ms = [results[N][n]['sw_ratio_mean'] for n in names]
        ss = [results[N][n]['sw_ratio_std'] for n in names]
        bars = axes[0].bar(range(n_f), ms, yerr=ss, capsize=3, color=colors, alpha=0.8)
        axes[0].set_xticks(range(n_f)); axes[0].set_xticklabels(names, rotation=40, ha='right')
        axes[0].set_ylabel('SW / Greedy Oracle')
        axes[0].set_title(f'N={N}'); axes[0].axhline(1.0, color='gray', ls='--', alpha=0.5)
        for b, m in zip(bars, ms):
            axes[0].text(b.get_x()+b.get_width()/2, m+ss[ms.index(m)]+0.03,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    for i, N in enumerate(N_VALUES):
        ms = [results[N][n]['sw_final_mean'] for n in names]
        bars = axes[1].bar(range(n_f), ms, color=colors, alpha=0.8)
        axes[1].set_xticks(range(n_f)); axes[1].set_xticklabels(names, rotation=40, ha='right')
        axes[1].set_ylabel('Final SW'); axes[1].set_title(f'N={N} Final SW')

    for i, N in enumerate(N_VALUES):
        ms = [results[N][n]['u_err_mean'] for n in names]
        bars = axes[2].bar(range(n_f), ms, color=colors, alpha=0.8)
        axes[2].set_xticks(range(n_f)); axes[2].set_xticklabels(names, rotation=40, ha='right')
        axes[2].set_ylabel('||u_hat - B||_F'); axes[2].set_title(f'N={N} u_hat Error')

    plt.suptitle(f'QA-MAB Combined Fix: TAU_CAP={TAU_CAP}, T={T}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'combined_fix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {RESULTS_DIR}/combined_fix_results.json + .png")
    print("\n" + "=" * 65)
    print("FINAL TABLE: SW / Oracle")
    print("=" * 65)
    hdr = "Fix".ljust(18) + "".join(f"N={n:>10}" for n in N_VALUES)
    print(hdr)
    for n in names:
        row = n.ljust(18) + "".join(f"{results[nn][n]['sw_ratio_mean']:>10.4f}" for nn in N_VALUES)
        print(row)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
