"""
convergence_simulation.py
Simulation 1: Convergence comparison of NB3R vs QA-MAB.

Runs both algorithms for 500 steps, N=10, m=4, averaged over 20 runs.
Plots social welfare over time + random baseline.
Saves figure to convergence_plot.png.

Usage:
    python convergence_simulation.py [N] [m] [T] [n_runs]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


def random_baseline(env, T, rng):
    """Compute social welfare for random route assignment each step."""
    sw = []
    for _ in range(T):
        x = {i: int(rng.integers(0, env.m)) for i in range(env.N)}
        sw.append(env.social_welfare(x))
    return np.array(sw)


def run_single_trial(N, m, T, seed, env_config=None):
    """Run one trial: create env, run NB3R, QA-MAB, and random baseline."""
    if env_config is None:
        env_config = {}
    
    rng = np.random.default_rng(seed)

    # NB3R
    env_nb3r = NetworkEnvironment(N, m, seed=seed, **env_config)
    nb3r = NB3R(env_nb3r, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed)
    nb3r_history = nb3r.run(T)

    # QA-MAB
    env_qa = NetworkEnvironment(N, m, seed=seed, **env_config)
    qa = QAMAB(env_qa, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed)
    qa_history = qa.run(T)

    # Random baseline
    env_rand = NetworkEnvironment(N, m, seed=seed, **env_config)
    rng2 = np.random.default_rng(seed)
    rand_history = random_baseline(env_rand, T, rng2)

    return nb3r_history, qa_history, rand_history


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    print(f"Running convergence simulation: N={N}, m={m}, T={T}, runs={n_runs}")

    nb3r_runs = []
    qa_runs = []
    rand_runs = []

    for r in range(n_runs):
        seed = 42 + r
        nb3r_h, qa_h, rand_h = run_single_trial(N, m, T, seed)
        nb3r_runs.append(nb3r_h)
        qa_runs.append(qa_h)
        rand_runs.append(rand_h)
        print(f"  Run {r+1}/{n_runs} done")

    nb3r_runs = np.array(nb3r_runs)
    qa_runs = np.array(qa_runs)
    rand_runs = np.array(rand_runs)

    nb3r_mean = nb3r_runs.mean(axis=0)
    nb3r_std = nb3r_runs.std(axis=0)
    qa_mean = qa_runs.mean(axis=0)
    qa_std = qa_runs.std(axis=0)
    rand_mean = rand_runs.mean(axis=0)
    rand_std = rand_runs.std(axis=0)

    steps = np.arange(T)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, nb3r_mean, label='NB3R (Distributed)', color='tab:blue')
    plt.fill_between(steps, nb3r_mean - nb3r_std, nb3r_mean + nb3r_std, alpha=0.2, color='tab:blue')
    plt.plot(steps, qa_mean, label='QA-MAB (Centralized)', color='tab:orange')
    plt.fill_between(steps, qa_mean - qa_std, qa_mean + qa_std, alpha=0.2, color='tab:orange')
    plt.plot(steps, rand_mean, label='Random Baseline', color='tab:gray', linestyle='--')
    plt.fill_between(steps, rand_mean - rand_std, rand_mean + rand_std, alpha=0.1, color='tab:gray')

    plt.xlabel('Step t')
    plt.ylabel('Social Welfare')
    plt.title(f'Convergence: NB3R vs QA-MAB (N={N}, m={m}, {n_runs} runs avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_plot.png', dpi=150)
    print("Saved convergence_plot.png")
    plt.close()

    print(f"\nFinal social welfare (mean ± std over last 50 steps, {n_runs} runs):")
    print(f"  NB3R:   {nb3r_mean[-50:].mean():.4f} ± {nb3r_std[-50:].mean():.4f}")
    print(f"  QA-MAB: {qa_mean[-50:].mean():.4f} ± {qa_std[-50:].mean():.4f}")
    print(f"  Random: {rand_mean[-50:].mean():.4f} ± {rand_std[-50:].mean():.4f}")
    print(f"\n  QA advantage over NB3R: {qa_mean[-50:].mean() - nb3r_mean[-50:].mean():.4f}")
    print(f"  QA advantage over Random: {qa_mean[-50:].mean() - rand_mean[-50:].mean():.4f}")


if __name__ == '__main__':
    main()
