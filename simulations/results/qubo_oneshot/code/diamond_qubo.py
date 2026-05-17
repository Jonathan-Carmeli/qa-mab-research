"""
DIAMOND-QUBO: one-shot centralized QUBO+SA solver for the multi-flow
routing problem in oracle mode (B and I known).

The QUBO formulation, SA inner loop, and brute-force / social-welfare
helpers are copied verbatim from
    simulations/experiments/convergence_test/02_sa_quality_sweep.py
because that filename starts with a digit and cannot be imported directly.
"""

import itertools
import math
import random
import time

import numpy as np


def build_oracle_qubo(B, I, lambda_=0.5, tau=1.0):
    """QUBO from TRUE B and I.

    diag           : -B[i,k] - lambda/2
    same-agent off : lambda/2  (symmetric upper+lower)            -- one-hot
    cross-agent    : I[i,k,j,l]                                   -- interference
    """
    N, m = B.shape
    size = N * m
    Q = np.zeros((size, size))
    for i in range(N):
        for k in range(m):
            ik = i * m + k
            Q[ik, ik] = -B[i, k] - lambda_ / 2.0
            for l in range(k + 1, m):
                il = i * m + l
                Q[ik, il] = lambda_ / 2.0
                Q[il, ik] = lambda_ / 2.0
    for i in range(N):
        for k in range(m):
            ik = i * m + k
            for j in range(N):
                if j == i:
                    continue
                for l in range(m):
                    Q[ik, j * m + l] = I[i, k, j, l]
    return tau * Q


def sa_solve(Q, N, m, n_restarts, n_iters, T0=2.0, decay=0.999,
             seed=0, greedy_init_B=None):
    """Multi-restart SA with O(1) delta-energy updates via Q_row_sum/Q_col_sum.

    Returns: (best_active, best_energy, assignment_tuple).
    """
    pyrng = random.Random(seed)
    best_active = None
    best_energy = float("inf")
    Q_diag = np.diag(Q).copy()

    for restart in range(n_restarts):
        active = np.zeros(N, dtype=np.int64)
        if greedy_init_B is not None:
            for i in range(N):
                active[i] = i * m + int(np.argmax(greedy_init_B[i]))
        else:
            for i in range(N):
                active[i] = i * m + pyrng.randrange(m)

        if restart > 0:
            n_flips = pyrng.randrange(1, max(2, N // 3 + 1))
            for _ in range(n_flips):
                i = pyrng.randrange(N)
                k_old = int(active[i] - i * m)
                cands = [k for k in range(m) if k != k_old]
                k_new = pyrng.choice(cands)
                active[i] = i * m + k_new

        Q_row_sum = Q[:, active].sum(axis=1)
        Q_col_sum = Q[active, :].sum(axis=0)
        energy = float(Q_col_sum[active].sum())
        if energy < best_energy:
            best_energy = energy
            best_active = active.copy()

        T = T0 * (1.0 + restart * 0.3)
        m_minus1 = m - 1
        for _ in range(n_iters):
            T *= decay
            i = pyrng.randrange(N)
            old_idx = int(active[i])
            k_old = old_idx - i * m
            k_new = (k_old + 1 + pyrng.randrange(m_minus1)) % m
            new_idx = i * m + k_new

            sum_row = (Q_row_sum[new_idx] - Q[new_idx, old_idx]) - (Q_row_sum[old_idx] - Q[old_idx, old_idx])
            sum_col = (Q_col_sum[new_idx] - Q[old_idx, new_idx]) - (Q_col_sum[old_idx] - Q[old_idx, old_idx])
            delta = (Q_diag[new_idx] - Q_diag[old_idx]) + sum_row + sum_col

            if delta < 0 or (T > 1e-10 and pyrng.random() < math.exp(-delta / T)):
                active[i] = new_idx
                energy += delta
                Q_row_sum += Q[:, new_idx] - Q[:, old_idx]
                Q_col_sum += Q[new_idx, :] - Q[old_idx, :]
                if energy < best_energy:
                    best_energy = energy
                    best_active = active.copy()

    assignment = tuple(int(best_active[i] - i * m) for i in range(N))
    return best_active, float(best_energy), assignment


def brute_force(B, I):
    """Enumerate all m^N assignments. Returns (best_sw, best_assignment)."""
    N, m = B.shape
    best_sw = -np.inf
    best_a = None
    for a in itertools.product(range(m), repeat=N):
        sw = 0.0
        for i, ki in enumerate(a):
            sw += B[i, ki]
            for j, kj in enumerate(a):
                if j == i:
                    continue
                sw -= I[i, ki, j, kj]
        if sw > best_sw:
            best_sw = sw
            best_a = a
    return float(best_sw), tuple(map(int, best_a))


def social_welfare(B, I, assignment):
    """SW = sum_i (B[i,k_i] - sum_{j!=i} I[i,k_i,j,k_j])."""
    sw = 0.0
    for i, ki in enumerate(assignment):
        sw += B[i, ki]
        for j, kj in enumerate(assignment):
            if j == i:
                continue
            sw -= I[i, ki, j, kj]
    return float(sw)


def solve_diamond_qubo(env, n_restarts=200, n_iters=5000,
                       T0=2.0, decay=0.999, lambda_=0.5, seed=0):
    """One-shot SA on the full oracle QUBO.

    Returns (assignment_dict, social_welfare, wall_clock_seconds).
    """
    t0 = time.time()
    Q = build_oracle_qubo(env.B, env.I, lambda_=lambda_, tau=1.0)
    _, _, assignment_tuple = sa_solve(
        Q, N=env.N, m=env.m,
        n_restarts=n_restarts, n_iters=n_iters,
        T0=T0, decay=decay, seed=seed,
        greedy_init_B=env.B,
    )
    assignment = {i: int(k) for i, k in enumerate(assignment_tuple)}
    sw = env.social_welfare(assignment)
    return assignment, float(sw), time.time() - t0
