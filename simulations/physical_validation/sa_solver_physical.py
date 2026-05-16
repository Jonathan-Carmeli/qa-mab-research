"""Simulated Annealing solver for binary QUBO problems."""
import numpy as np


def sa_solve(Q, rng, n_reads=20, n_sweeps=200, T_init=2.0, T_final=0.05):
    """Solve QUBO via Simulated Annealing with multiple restarts.

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
    best_x      : (M,) binary numpy array
    best_energy : float
    """
    M = Q.shape[0]
    best_x = np.zeros(M, dtype=int)
    best_energy = float('inf')

    # Linear temperature schedule
    temps = np.linspace(T_init, T_final, n_sweeps)

    for _ in range(n_reads):
        x = rng.integers(0, 2, size=M)
        h = Q @ x  # (M,) cached dot product

        energy = float(x @ h)  # x^T Q x

        for T in temps:
            # Random sweep order
            order = rng.permutation(M)
            for i in order:
                old_xi = x[i]
                # delta_E when flipping x[i]
                # if x[i] = 0 → 1: delta_E = Q[i,i] + 2*(h[i] - Q[i,i]*x[i])
                #                            = Q[i,i] + 2*h[i]  (since x[i]=0)
                # if x[i] = 1 → 0: delta_E = -(Q[i,i] + 2*(h[i] - Q[i,i]))
                #                            = -(2*h[i] - Q[i,i])
                # Unified: delta_E = (1 - 2*x[i]) * (Q[i,i] + 2*(h[i] - Q[i,i]*x[i]))
                # Simplify: (1 - 2*x[i]) * (2*h[i] - Q[i,i]*(2*x[i]-1))
                # Cleaner: delta = (1 - 2*x[i]) * (2*h[i] - Q[i,i] + 2*Q[i,i]*x[i] ... )
                # Use direct formula:
                # x[i]=0 -> 1: new_E contribution changes by Q[i,i] + 2*sum_{j!=i} Q[i,j]*x[j]
                #                                             = Q[i,i] + 2*(h[i] - Q[i,i]*0)
                #                                             = Q[i,i] + 2*h[i]
                # x[i]=1 -> 0: delta_E = -(Q[i,i] + 2*(h[i] - Q[i,i]*1))
                #                      = -(Q[i,i] + 2*h[i] - 2*Q[i,i])
                #                      = -(2*h[i] - Q[i,i])
                # Both cases: delta_E = (1 - 2*x[i]) * (Q[i,i] + 2*(h[i] - Q[i,i]*x[i]))
                delta_E = (1 - 2 * old_xi) * (Q[i, i] + 2 * (h[i] - Q[i, i] * old_xi))

                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    # Accept flip
                    x[i] = 1 - old_xi
                    # Update h: h[j] += Q[j,i] * delta_x_i
                    delta_xi = x[i] - old_xi  # +1 or -1
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
            chosen[n] = 0  # fallback
        else:
            chosen[n] = int(np.argmax(segment))
    return chosen
