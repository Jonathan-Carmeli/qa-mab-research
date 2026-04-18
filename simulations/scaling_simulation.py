"""
scaling_simulation.py
Simulation 2: How NB3R vs QA-MAB performance scales with network size N.

For each N in [5, 10, 20, 30, 50], run both algorithms to convergence
(500 steps), average over 20 runs. Plot final social welfare vs N.

Also supports ablation studies via command-line arguments.

Usage:
    python scaling_simulation.py [N_LIST] [M] [T] [N_RUNS]
    python scaling_simulation.py --ablation [PARAM]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


def parse_N_list(arg):
    """Parse N list from string like '5,10,20,30,50'"""
    return [int(x.strip()) for x in arg.split(',')]


def run_single_trial(N, m, T, seed, env_config=None, algo='both'):
    """Run one trial."""
    if env_config is None:
        env_config = {}
    
    results = {}
    rng = np.random.default_rng(seed)

    if algo in ['nb3r', 'both']:
        env_nb3r = NetworkEnvironment(N, m, seed=seed, **env_config)
        nb3r = NB3R(env_nb3r, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed)
        nb3r_hist = nb3r.run(T)
        results['nb3r'] = np.mean(nb3r_hist[-50:])

    if algo in ['qa', 'both']:
        env_qa = NetworkEnvironment(N, m, seed=seed, **env_config)
        qa = QAMAB(env_qa, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed)
        qa_hist = qa.run(T)
        results['qa'] = np.mean(qa_hist[-50:])

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='QA-MAB Scaling Simulation')
    parser.add_argument('--N', default='5,10,20,30,50', help='Comma-separated N values')
    parser.add_argument('--m', type=int, default=4, help='Number of routes per agent')
    parser.add_argument('--T', type=int, default=500, help='Steps per trial')
    parser.add_argument('--runs', type=int, default=20, help='Number of runs')
    parser.add_argument('--I_scale', default='moderate', 
                        choices=['low', 'moderate', 'high'],
                        help='Interference scale')
    parser.add_argument('--B_scale', default='uniform',
                        choices=['uniform', 'skewed'],
                        help='Base utility distribution')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_',
                        help='QUBO constraint penalty')
    parser.add_argument('--algo', default='both', choices=['both', 'nb3r', 'qa'],
                        help='Which algorithm to run')
    parser.add_argument('--save', default='scaling_plot.png',
                        help='Output plot filename')
    
    args = parser.parse_args()
    
    N_LIST = parse_N_list(args.N)
    M = args.m
    T = args.T
    N_RUNS = args.runs
    
    env_config = {'I_scale': args.I_scale, 'B_scale': args.B_scale}
    
    print(f"Running scaling simulation: N_LIST={N_LIST}, m={M}, T={T}, runs={N_RUNS}")
    print(f"Environment: I_scale={args.I_scale}, B_scale={args.B_scale}")
    print(f"QA-MAB lambda={args.lambda_}")

    nb3r_results = {n: [] for n in N_LIST}
    qa_results = {n: [] for n in N_LIST}

    for N in N_LIST:
        print(f"\nN = {N}:")
        for r in range(N_RUNS):
            seed = 42 + r
            res = run_single_trial(N, M, T, seed, env_config, args.algo)
            
            if 'nb3r' in res:
                nb3r_results[N].append(res['nb3r'])
            if 'qa' in res:
                qa_results[N].append(res['qa'])
            
            print(f"  run {r+1}/{N_RUNS}", end='')
            if 'nb3r' in res:
                print(f"  NB3R={res['nb3r']:.3f}", end='')
            if 'qa' in res:
                print(f"  QA={res['qa']:.3f}", end='')
            print()

    # ── Compute stats ─────────────────────────────────────────────────────────
    if args.algo in ['nb3r', 'both']:
        nb3r_means = np.array([np.mean(nb3r_results[n]) for n in N_LIST])
        nb3r_stds = np.array([np.std(nb3r_results[n]) for n in N_LIST])
    if args.algo in ['qa', 'both']:
        qa_means = np.array([np.mean(qa_results[n]) for n in N_LIST])
        qa_stds = np.array([np.std(qa_results[n]) for n in N_LIST])

    # ── Plot ───────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    if args.algo in ['nb3r', 'both']:
        ax.errorbar(N_LIST, nb3r_means, yerr=nb3r_stds,
                    fmt='-s', color='tab:blue', capsize=5, lw=2,
                    label='NB3R (Distributed)')
    if args.algo in ['qa', 'both']:
        ax.errorbar(N_LIST, qa_means, yerr=qa_stds,
                    fmt='-o', color='tab:orange', capsize=5, lw=2,
                    label='QA-MAB (Centralized)')

    ax.set_xlabel('Number of agents (N)', fontsize=13)
    ax.set_ylabel('Final Social Welfare', fontsize=13)
    ax.set_title(f'Scaling: Final Welfare vs Network Size (m={M})\n'
                 f'I_scale={args.I_scale}, B_scale={args.B_scale} | '
                 f'Mean ± std over {N_RUNS} runs, last 50 steps',
                 fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"\nSaved {args.save}")
    plt.close()

    # ── Print table ────────────────────────────────────────────────────────────
    header = f"{'N':>5} |"
    if args.algo in ['nb3r', 'both']:
        header += f" {'NB3R mean':>10} {'NB3R std':>10} |"
    if args.algo in ['qa', 'both']:
        header += f" {'QA mean':>10} {'QA std':>10} |"
    if args.algo == 'both':
        header += f" {'QA-NB3R':>10} |"
    
    print(f"\n{header}")
    print("-" * len(header))
    for i, N in enumerate(N_LIST):
        row = f"{N:>5} |"
        if args.algo in ['nb3r', 'both']:
            row += f" {nb3r_means[i]:>10.4f} {nb3r_stds[i]:>10.4f} |"
        if args.algo in ['qa', 'both']:
            row += f" {qa_means[i]:>10.4f} {qa_stds[i]:>10.4f} |"
        if args.algo == 'both':
            row += f" {qa_means[i] - nb3r_means[i]:>10.4f} |"
        print(row)


if __name__ == '__main__':
    main()
