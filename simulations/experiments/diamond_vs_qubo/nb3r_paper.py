"""
Paper-faithful NB3R (Noisy Best-Response for Route Refinement) from
DIAMOND (Paul, Cohen, Kedar, arXiv:2303.15544), Section III-C.

Differs from simulations/legacy/nb3r.py:
  - Asynchronous: one flow updates per round (uniform random choice).
  - Counterfactual queries: for each candidate path k, the collaborative
    utility U_n(k, sigma_{-n}) is computed exactly via env.compute_throughput,
    not estimated by EMA from sampled throughputs.
  - Boltzmann sampling per eq (10):  P(k) ~ exp(nu(t) * U_n(k))
  - Log cooling per Corollary 1:     nu(t) = log(t+1) / Delta,  Delta = N.

For the fully-connected interference model (every flow interferes with
every other flow), U_n(k, sigma_{-n}) = sum_m u_m under the trial
assignment with sigma_n := k, which is exactly the social welfare.
"""

import math
import time

import numpy as np


def _boltzmann_sample(utilities, nu, rng):
    u = np.asarray(utilities, dtype=np.float64)
    z = nu * (u - u.max())
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def run_nb3r(env, T_rounds=5000, seed=0, record_every=50):
    """Asynchronous paper-faithful NB3R with oracle utility queries.

    Returns (final_assignment_dict, sw_history, wall_clock_seconds).
    sw_history is a list of (t, social_welfare) pairs sampled every
    record_every rounds (plus the final round).
    """
    rng = np.random.default_rng(seed)
    N, m = env.N, env.m
    sigma = {i: int(rng.integers(0, m)) for i in range(N)}
    Delta = float(N)  # u_max ~= 1 in NetworkEnvironment

    sw_history = []
    t0 = time.time()
    for t in range(1, T_rounds + 1):
        n = int(rng.integers(0, N))
        U = np.empty(m)
        for k in range(m):
            trial = dict(sigma)
            trial[n] = k
            U[k] = sum(env.compute_throughput(trial).values())
        nu_t = math.log(t + 1) / Delta
        sigma[n] = _boltzmann_sample(U, nu_t, rng)
        if t % record_every == 0 or t == T_rounds:
            sw_history.append((t, float(env.social_welfare(sigma))))
    return sigma, sw_history, time.time() - t0


def stable_tail(sw_history, frac=0.1, tol=0.01):
    """Return True if the last `frac` of sw_history has std < tol * |mean|.

    Used by the driver to decide whether NB3R has converged or needs more
    rounds. Operates on the (t, sw) list returned by run_nb3r.
    """
    if not sw_history:
        return False
    vals = np.array([sw for _, sw in sw_history], dtype=np.float64)
    tail_n = max(2, int(len(vals) * frac))
    tail = vals[-tail_n:]
    mean = float(np.mean(tail))
    if abs(mean) < 1e-12:
        return float(np.std(tail)) < tol
    return float(np.std(tail)) < tol * abs(mean)
