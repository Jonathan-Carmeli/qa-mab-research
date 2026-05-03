"""
clean_split_v19.py — Convergence trajectory analysis

Goal: Track how B_hat converges to true B over time, and see if Phase B
SW improves with more iterations.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)

BASE_SEED = 2026
N = 10; M = 4; N_SEEDS = 5


def run_trajectory(n_seeds, T, use_oracle_I, B_init, B_LR, UCB_C_val, I_LR):
    """Run and track B_hat convergence over time."""
    all_trajectories = []
    
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        rng = np.random.default_rng(si * 1000 + N)
        B_hat = np.full((N, M), B_init)
        visits = np.zeros((N, M))
        I_hat = np.zeros((N, M, N, M))
        _prev_x = None; _prev_tp = None
        
        step_sw = []; step_b_err = []; step_i_err = []
        
        for step in range(T):
            # SA+QUBO
            Q = np.zeros((N * M, N * M))
            for i in range(N):
                for k in range(M):
                    Q[i * M + k, i * M + k] = -B_hat[i, k] - UCB_C_val / math.sqrt(visits[i, k] + 1)
            tau = 1.0
            for i in range(N):
                for k in range(M):
                    for l in range(k + 1, M):
                        Q[i * M + k, i * M + l] += tau / 2
                        Q[i * M + l, i * M + k] += tau / 2
            if use_oracle_I:
                for i in range(N):
                    for k in range(M):
                        for j in range(N):
                            if j == i: continue
                            for l in range(M):
                                Q[i * M + k, j * M + l] += env.I[i, k, j, l]
            else:
                for i in range(N):
                    for k in range(M):
                        for j in range(N):
                            if j == i: continue
                            for l in range(M):
                                Q[i * M + k, j * M + l] += I_hat[i, k, j, l] * 0.5
            
            T_sa = 2.0 / math.log(step + 3)
            best_x = None; best_e = float('inf')
            for r in range(4):
                x = np.zeros(N * M)
                for i in range(N): x[i * M + int(np.argmax(B_hat[i]))] = 1.0
                e = float(x @ Q @ x)
                if e < best_e: best_e, best_x = e, x.copy()
                T_r = T_sa * (1 + r * 0.3)
                for _ in range(200):
                    T_r *= 0.95
                    if T_r < 1e-12: break
                    i = rng.integers(0, N)
                    block = x[i * M:(i + 1) * M]
                    ko = int(np.argmax(block)); kn = (ko + 1 + rng.integers(0, M - 1)) % M
                    x[i * M + ko] = 0.0; x[i * M + kn] = 1.0
                    ne = float(x @ Q @ x)
                    if ne < e or rng.random() < math.exp(-(ne - e) / T_r):
                        e = ne
                        if e < best_e: best_e, best_x = e, x.copy()
                    else:
                        x[i * M + kn] = 0.0; x[i * M + ko] = 1.0
            
            a = {i: int(np.argmax(best_x[i * M:(i + 1) * M])) for i in range(N)}
            tp = env.compute_throughput(a)
            
            if not use_oracle_I and _prev_x is not None:
                for i in range(N):
                    for j in range(i + 1, N):
                        ki = _prev_x[i]; kj = _prev_x[j]
                        di = max(0.0, B_hat[i, ki] - _prev_tp[i])
                        dj = max(0.0, B_hat[j, kj] - _prev_tp[j])
                        if di > 0.02: I_hat[i, ki, j, kj] = min(I_hat[i, ki, j, kj] + I_LR, 0.5)
                        if dj > 0.02: I_hat[j, kj, i, ki] = min(I_hat[j, kj, i, ki] + I_LR, 0.5)
                I_hat *= (1.0 - 0.001)
                for i in range(N): I_hat[i, :, i, :] = 0.0
            
            for i in range(N):
                k = a[i]
                if use_oracle_I:
                    I_sum = sum(env.I[i, k, j, a[j]] for j in range(N) if j != i)
                    B_hat[i, k] += B_LR * ((tp[i] + I_sum) - B_hat[i, k])
                else:
                    B_hat[i, k] += B_LR * (tp[i] - B_hat[i, k])
                visits[i, k] += 1
            
            _prev_x = a.copy()
            _prev_tp = {i: tp[i] for i in range(N)}
            
            opt = float(np.sum(np.max(env.B, axis=1)))
            sw = float(sum(tp.values()))
            b_err = float(np.mean(np.abs(B_hat - env.B)))
            i_err = float(np.mean(np.abs(I_hat - env.I))) if not use_oracle_I else 0.0
            
            step_sw.append(sw / opt if opt > 0 else 0)
            step_b_err.append(b_err)
            step_i_err.append(i_err)
        
        all_trajectories.append({'sw': step_sw, 'b_err': step_b_err, 'i_err': step_i_err})
    
    return all_trajectories


def main():
    print("=" * 70)
    print("V19 — Convergence trajectory analysis")
    print("=" * 70)
    
    # Shorter runs for trajectory tracking
    T = 200
    
    print("\nPhase A (oracle I):")
    traj_A = run_trajectory(3, T, True, 0.7, 0.12, 0.5, 0.05)
    avg_sw_A = np.mean([t['sw'] for t in traj_A], axis=0)
    avg_be_A = np.mean([t['b_err'] for t in traj_A], axis=0)
    print(f"  Final SW/T: {avg_sw_A[-1]:.4f}")
    print(f"  Final B_err: {avg_be_A[-1]:.4f}")
    print(f"  SW at T=50: {avg_sw_A[49]:.4f}, T=100: {avg_sw_A[99]:.4f}, T=200: {avg_sw_A[199]:.4f}")
    
    print("\nPhase B (learned I, B_init=0.7):")
    traj_B = run_trajectory(3, T, False, 0.7, 0.12, 0.5, 0.05)
    avg_sw_B = np.mean([t['sw'] for t in traj_B], axis=0)
    avg_be_B = np.mean([t['b_err'] for t in traj_B], axis=0)
    avg_ie_B = np.mean([t['i_err'] for t in traj_B], axis=0)
    print(f"  Final SW/T: {avg_sw_B[-1]:.4f}")
    print(f"  Final B_err: {avg_be_B[-1]:.4f}, I_err: {avg_ie_B[-1]:.4f}")
    print(f"  SW at T=50: {avg_sw_B[49]:.4f}, T=100: {avg_sw_B[99]:.4f}, T=200: {avg_sw_B[199]:.4f}")
    
    print("\nPhase B (learned I, B_init=1.5):")
    traj_B2 = run_trajectory(3, T, False, 1.5, 0.12, 0.5, 0.05)
    avg_sw_B2 = np.mean([t['sw'] for t in traj_B2], axis=0)
    print(f"  Final SW/T: {avg_sw_B2[-1]:.4f}")
    print(f"  SW at T=50: {avg_sw_B2[49]:.4f}, T=100: {avg_sw_B2[99]:.4f}, T=200: {avg_sw_B2[199]:.4f}")
    
    print("\nPhase B (learned I, high I_LR=0.15):")
    traj_B3 = run_trajectory(3, T, False, 0.7, 0.12, 0.5, 0.15)
    avg_sw_B3 = np.mean([t['sw'] for t in traj_B3], axis=0)
    avg_ie_B3 = np.mean([t['i_err'] for t in traj_B3], axis=0)
    print(f"  Final SW/T: {avg_sw_B3[-1]:.4f}")
    print(f"  Final I_err: {avg_ie_B3[-1]:.4f}")
    print(f"  SW at T=50: {avg_sw_B3[49]:.4f}, T=100: {avg_sw_B3[99]:.4f}, T=200: {avg_sw_B3[199]:.4f}")
    
    # Save trajectory for analysis
    trajectory_data = {
        'phaseA': {'sw': [float(x) for x in avg_sw_A], 'b_err': [float(x) for x in avg_be_A]},
        'phaseB_b07': {'sw': [float(x) for x in avg_sw_B], 'b_err': [float(x) for x in avg_be_B], 'i_err': [float(x) for x in avg_ie_B]},
        'phaseB_b15': {'sw': [float(x) for x in avg_sw_B2]},
        'phaseB_ILR15': {'sw': [float(x) for x in avg_sw_B3], 'i_err': [float(x) for x in avg_ie_B3]},
    }
    
    with open(os.path.join(RESULTS, 'clean_split_v19_trajectories.json'), 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"\nSaved trajectories.")


if __name__ == '__main__':
    main()