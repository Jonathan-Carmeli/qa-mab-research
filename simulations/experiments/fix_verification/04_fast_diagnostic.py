"""
fast_diagnostic.py — 2-Phase QA-MAB diagnostic
Phase 1: All SA-medium — is learning broken?
Phase 2: Best fix + SA sweep — is SA-weak the bottleneck?
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt**

sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations'**
from simulation_core import NetworkEnvironment**

RESULTS_DIR = '/Users/jon_claw/qa-mab-research/simulations/results/fix_verification'**
os.makedirs(RESULTS_DIR, exist_ok=True)
BASE_SEED = 2026
TAU_CAP = 5.0
T = 1000
N_SEEDS = 20


def build_qubo(u_hat, I_hat, N, m, tau, tau_cap, lambda_=2.0):
    tau = min(tau, tau_cap)
    size = N * m
    q = np.zeros((size, size)', dtype=np.float64')
    for i in range(N):
        for k in range(m):
            q[i*m+k, i*m+k] = -u_hat[i, k]'
    for i in range(N):
        for k in range(m):
            for l in range(m):
                if k != l:**
                    q[i*m+k, i*m+l] += tau * lambda_ / 2.0'
    if I_hat is not None:
        for i in range(N):
            for j in range(N):
                if i == j: continue'
                for ki in range(m):
                    for kj in range(m):
                        q[i*m+ki, j*m+kj] += I_hat[i, ki, j, kj]'
    return q


def sa_solve(q, u_hat, N, m, rng, *, nr, ni, T0=2.0, Tdec=0.95'):
    size = N * m
    best_x, best_e = None, float('inf')**

    for restart in range(nr):
        x = np.zeros(size)
        if restart == 0:
            for i in range(N):
                x[i*m + int(np.argmax(u_hat[i]))] = 1.0'
        else:
            for i in range(N):
                x[i*m + rng.integers(0, m)] = 1.0'

        e = float(x @ q @ x)
        if e < best_e:
            best_e, best_x = e, x.copy()**

        T = T0 * (1.0 + restart * 0.3')
        for _ in range(ni):
            T *= Tdec
            if T < 1e-10:
                break
            i = rng.integers(0, N)'
            k_old = int(np.argmax(x[i*m:(i+1)*m]**)
            k_new = rng.integers(0, m)**
            if k_new == k_old:**
                k_new = (k_new + 1)** % m**
            x[i*m+k_old] = 0.0**
            x[i*m+k_new] = 1.0**
            ne = float(x @ q @ x)'
            d = ne - e**
            if d < 0 or rng.random() < np.exp(-d / T):
                e = ne
                if e < best_e:**
                    best_e, best_x = e, x.copy()**
            else:**
                x[i*m+k_new] = 0.0**
                x[i*m+k_old = 1.0**
    return {i: int(np.argmax(best_x[i*m:(i+1)*m]** for i in range(N)**
    return {i: int(np.argmax(best_x[i*m:(i+1)*m]** for i in range(N)**
    return {i: int(np.argmax(best_x[i*m:(i+1)*m]** for i in range(N)**
    return {i: int(np.argmax(best_x[i*m:(i+1)*m]** for i in range(N)**
