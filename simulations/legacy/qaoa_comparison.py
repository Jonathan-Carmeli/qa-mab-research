"""
qaoa_comparison.py
Compare QAOA vs SA vs Brute Force vs NB3R for small N.

N=5, m=4 = 20 qubits (limit of classical QAOA simulation).
Shows what real quantum computing would achieve vs SA proxy.
"""

import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*SparseEfficiencyWarning.*')
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, '.')
from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB
from qaoa_solver import solve_qubo_qaoa, solve_qubo_bruteforce


def run_comparison(N=5, m=4, T=500, n_runs=5, qaoa_reps=2):
    """
    Run full comparison: NB3R vs QA-MAB(SA) vs QA-MAB(QAOA) vs QA-MAB(BruteForce).
    
    For QAOA and BruteForce: we use the same QA-MAB learning (u_hat, I_hat),
    but replace the SA solver with QAOA or brute force at each step.
    """
    print(f"Comparison: N={N}, m={m}, T={T}, runs={n_runs}, QAOA reps={qaoa_reps}")
    print(f"Qubits: {N*m}")
    print()
    
    results = {
        'nb3r': [], 'sa': [], 'qaoa': [], 'bruteforce': [], 'random': []
    }
    
    for r in range(n_runs):
        seed = 42 + r
        print(f"--- Run {r+1}/{n_runs} (seed={seed}) ---")
        
        # NB3R
        t0 = time.time()
        env_nb3r = NetworkEnvironment(N, m, seed=seed)
        nb3r = NB3R(env_nb3r, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed)
        nb3r_hist = nb3r.run(T)
        nb3r_final = np.mean(nb3r_hist[-50:])
        print(f"  NB3R:        {nb3r_final:.4f}  ({time.time()-t0:.1f}s)")
        results['nb3r'].append(nb3r_final)
        
        # QA-MAB with SA solver (default)
        t0 = time.time()
        env_sa = NetworkEnvironment(N, m, seed=seed)
        qa_sa = QAMAB(env_sa, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed)
        sa_hist = qa_sa.run(T)
        sa_final = np.mean(sa_hist[-50:])
        print(f"  QA-MAB(SA):  {sa_final:.4f}  ({time.time()-t0:.1f}s)")
        results['sa'].append(sa_final)
        
        # QA-MAB with QAOA solver
        t0 = time.time()
        env_qaoa = NetworkEnvironment(N, m, seed=seed)
        qa_qaoa = QAMAB(env_qaoa, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed)
        qaoa_hist = []
        for t in range(T):
            # Build QUBO from current estimates
            Q = qa_qaoa.build_qubo()
            # Solve with QAOA instead of SA
            assignment, energy = solve_qubo_qaoa(Q, N, m, reps=qaoa_reps, maxiter=100, seed=seed+t)
            # Execute assignment and learn
            throughputs = qa_qaoa.env.compute_throughput(assignment)
            # Update u_hat
            for i in range(N):
                k = assignment[i]
                observed = throughputs[i]
                qa_qaoa.u_hat[i, k] += qa_qaoa.B_learn_rate * (observed - qa_qaoa.u_hat[i, k])
            # Update I_hat (collision inference)
            if qa_qaoa._prev_x is not None and qa_qaoa._prev_throughputs is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = qa_qaoa._prev_x[i]
                        kj = qa_qaoa._prev_x[j]
                        expected_i = qa_qaoa.u_hat[i, ki]
                        expected_j = qa_qaoa.u_hat[j, kj]
                        drop_i = max(0, expected_i - qa_qaoa._prev_throughputs[i])
                        drop_j = max(0, expected_j - qa_qaoa._prev_throughputs[j])
                        if drop_i > qa_qaoa.collision_threshold:
                            qa_qaoa.I_hat[i, ki, j, kj] = min(
                                qa_qaoa.I_hat[i, ki, j, kj] + qa_qaoa.I_learn_rate, qa_qaoa.I_cap)
                        if drop_j > qa_qaoa.collision_threshold:
                            qa_qaoa.I_hat[j, kj, i, ki] = min(
                                qa_qaoa.I_hat[j, kj, i, ki] + qa_qaoa.I_learn_rate, qa_qaoa.I_cap)
            qa_qaoa._prev_x = assignment.copy()
            qa_qaoa._prev_throughputs = {i: throughputs[i] for i in range(N)}
            qa_qaoa.tau += qa_qaoa.delta_tau
            sw = qa_qaoa.env.social_welfare(assignment)
            qaoa_hist.append(sw)
            
            if (t+1) % 100 == 0:
                print(f"    QAOA step {t+1}/{T}...")
        
        qaoa_final = np.mean(qaoa_hist[-50:])
        print(f"  QA-MAB(QAOA): {qaoa_final:.4f}  ({time.time()-t0:.1f}s)")
        results['qaoa'].append(qaoa_final)
        
        # QA-MAB with Brute Force (oracle - perfect solver)
        t0 = time.time()
        env_bf = NetworkEnvironment(N, m, seed=seed)
        qa_bf = QAMAB(env_bf, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed)
        bf_hist = []
        for t in range(T):
            Q = qa_bf.build_qubo()
            assignment, energy = solve_qubo_bruteforce(Q, N, m)
            throughputs = qa_bf.env.compute_throughput(assignment)
            for i in range(N):
                k = assignment[i]
                observed = throughputs[i]
                qa_bf.u_hat[i, k] += qa_bf.B_learn_rate * (observed - qa_bf.u_hat[i, k])
            if qa_bf._prev_x is not None and qa_bf._prev_throughputs is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = qa_bf._prev_x[i]
                        kj = qa_bf._prev_x[j]
                        expected_i = qa_bf.u_hat[i, ki]
                        expected_j = qa_bf.u_hat[j, kj]
                        drop_i = max(0, expected_i - qa_bf._prev_throughputs[i])
                        drop_j = max(0, expected_j - qa_bf._prev_throughputs[j])
                        if drop_i > qa_bf.collision_threshold:
                            qa_bf.I_hat[i, ki, j, kj] = min(
                                qa_bf.I_hat[i, ki, j, kj] + qa_bf.I_learn_rate, qa_bf.I_cap)
                        if drop_j > qa_bf.collision_threshold:
                            qa_bf.I_hat[j, kj, i, ki] = min(
                                qa_bf.I_hat[j, kj, i, ki] + qa_bf.I_learn_rate, qa_bf.I_cap)
            qa_bf._prev_x = assignment.copy()
            qa_bf._prev_throughputs = {i: throughputs[i] for i in range(N)}
            qa_bf.tau += qa_bf.delta_tau
            sw = qa_bf.env.social_welfare(assignment)
            bf_hist.append(sw)
        
        bf_final = np.mean(bf_hist[-50:])
        print(f"  QA-MAB(BF):  {bf_final:.4f}  ({time.time()-t0:.1f}s)")
        results['bruteforce'].append(bf_final)
        
        # Random
        rng = np.random.default_rng(seed)
        env_rand = NetworkEnvironment(N, m, seed=seed)
        rand_sw = [env_rand.social_welfare({i: int(rng.integers(0, m)) for i in range(N)}) for _ in range(T)]
        rand_final = np.mean(rand_sw[-50:])
        results['random'].append(rand_final)
        print(f"  Random:      {rand_final:.4f}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':>18} | {'Mean':>8} | {'Std':>8}")
    print("-" * 42)
    for name in ['nb3r', 'sa', 'qaoa', 'bruteforce', 'random']:
        vals = results[name]
        label = {
            'nb3r': 'NB3R',
            'sa': 'QA-MAB(SA)',
            'qaoa': 'QA-MAB(QAOA)',
            'bruteforce': 'QA-MAB(Oracle)',
            'random': 'Random'
        }[name]
        print(f"{label:>18} | {np.mean(vals):>8.4f} | {np.std(vals):>8.4f}")


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    run_comparison(N, m, T, n_runs)
