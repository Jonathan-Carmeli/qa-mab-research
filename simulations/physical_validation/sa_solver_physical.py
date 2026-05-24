"""Simulated Annealing solvers for QUBO problems.

Two implementations are available:

sa_onehot  — route-flip SA (new).
             Proposals switch an entire agent's path, so the one-hot
             constraint is guaranteed structurally at every step.
             O(1) delta-energy via Q_row_sum / Q_col_sum caches.
             Returns (N,) int array of path indices directly.

sa_sweep   — bit-flip SA (original).
             Proposes flipping individual bits of the binary vector x.
             One-hot is not guaranteed; relies on the penalty term in Q.
             Returns (best_x, best_energy) — call decode_solution() to
             convert best_x to (N,) path indices.

sa_solve   — alias for sa_onehot (default after migration).
"""
import numpy as np


# ── Route-flip SA (new) ───────────────────────────────────────────────────────

def sa_onehot(Q, N, K, rng, n_restarts=20, n_iters=1000, T0=2.0, decay=0.999):
    """Solve a one-hot QUBO via route-flip Simulated Annealing.

    Parameters
    ----------
    Q          : (M, M) symmetric QUBO matrix, M = N*K
    N          : number of flows (agents)
    K          : number of paths per flow
    rng        : numpy Generator
    n_restarts : number of independent SA runs
    n_iters    : number of route-flip steps per run
    T0         : initial temperature (scaled up per restart index)
    decay      : geometric temperature decay per step

    Returns
    -------
    chosen : (N,) int array with values in [0, K)
    """
    best_active = None
    best_energy = float('inf')
    Q_diag = np.diag(Q).copy()

    for restart in range(n_restarts):
        # active[n] = global QUBO index n*K + k_n
        ks = rng.integers(0, K, size=N)
        active = np.array([n * K + int(ks[n]) for n in range(N)], dtype=np.int64)

        Q_row_sum = Q[:, active].sum(axis=1)
        Q_col_sum = Q[active, :].sum(axis=0)
        energy = float(Q_col_sum[active].sum())

        if energy < best_energy:
            best_energy = energy
            best_active = active.copy()

        T = T0 * (1.0 + restart * 0.3)

        for _ in range(n_iters):
            T *= decay

            n = int(rng.integers(0, N))
            old_idx = int(active[n])
            k_old = old_idx - n * K
            k_shift = int(rng.integers(1, K))   # guaranteed != 0
            k_new = (k_old + k_shift) % K
            new_idx = n * K + k_new

            # Delta-E in O(1) using cached row/col sums
            sum_row = (
                (Q_row_sum[new_idx] - Q[new_idx, old_idx])
                - (Q_row_sum[old_idx] - Q[old_idx, old_idx])
            )
            sum_col = (
                (Q_col_sum[new_idx] - Q[old_idx, new_idx])
                - (Q_col_sum[old_idx] - Q[old_idx, old_idx])
            )
            delta = (Q_diag[new_idx] - Q_diag[old_idx]) + sum_row + sum_col

            if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
                active[n] = new_idx
                energy += delta
                Q_row_sum += Q[:, new_idx] - Q[:, old_idx]
                Q_col_sum += Q[new_idx, :] - Q[old_idx, :]
                if energy < best_energy:
                    best_energy = energy
                    best_active = active.copy()

    return np.array([int(best_active[n]) - n * K for n in range(N)], dtype=int)


# ── Bit-flip SA (original) ────────────────────────────────────────────────────

def sa_sweep(Q, rng, n_reads=20, n_sweeps=200, T_init=2.0, T_final=0.05):
    """Solve QUBO via bit-flip Simulated Annealing with multiple restarts.

    Parameters
    ----------
    Q        : (M, M) symmetric QUBO matrix
    rng      : numpy Generator
    n_reads  : number of independent SA runs
    n_sweeps : number of sweeps per run
    T_init   : initial temperature
    T_final  : final temperature (linear schedule)

    Returns
    -------
    best_x      : (M,) binary numpy array  — call decode_solution() to decode
    best_energy : float
    """
    M = Q.shape[0]
    best_x = np.zeros(M, dtype=int)
    best_energy = float('inf')

    temps = np.linspace(T_init, T_final, n_sweeps)

    for _ in range(n_reads):
        x = rng.integers(0, 2, size=M)
        h = Q @ x

        energy = float(x @ h)

        for T in temps:
            order = rng.permutation(M)
            for i in order:
                old_xi = x[i]
                delta_E = (1 - 2 * old_xi) * (Q[i, i] + 2 * (h[i] - Q[i, i] * old_xi))

                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    x[i] = 1 - old_xi
                    delta_xi = x[i] - old_xi
                    h += Q[:, i] * delta_xi
                    energy += delta_E

        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return best_x, best_energy


def decode_solution(x, N, K):
    """Decode binary solution vector to chosen paths.

    Parameters
    ----------
    x : (M=N*K,) binary array
    N : number of flows
    K : number of paths per flow

    Returns
    -------
    chosen_paths : (N,) int array with values in [0, K)
    """
    chosen = np.zeros(N, dtype=int)
    for n in range(N):
        segment = x[n * K:(n + 1) * K]
        if segment.sum() == 0:
            chosen[n] = 0
        else:
            chosen[n] = int(np.argmax(segment))
    return chosen


# Default alias — points to route-flip variant after migration
sa_solve = sa_onehot
