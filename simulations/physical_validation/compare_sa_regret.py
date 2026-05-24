#!/usr/bin/env python3
"""Compare sa_onehot (route-flip) vs sa_sweep (bit-flip) on the physical env.

Runs QA-MAB with both SA variants for P epochs x T steps across n_seeds seeds,
plots cumulative regret curves, and saves results to
simulations/results/compare_sa_regret/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulations.physical_validation.physical_env import AbstractWorld
from simulations.physical_validation.qa_mab_physical import QAMABPhysical
from simulations.physical_validation.sa_solver_physical import (
    sa_onehot, sa_sweep, decode_solution
)

OUT = "simulations/results/compare_sa_regret/"
os.makedirs(OUT, exist_ok=True)

# ── Variant that uses sa_sweep (bit-flip) ────────────────────────────────────

class QAMABSweep(QAMABPhysical):
    """QA-MAB variant using the original bit-flip SA (sa_sweep)."""

    name = "QA-MAB-sa_sweep"

    def __init__(self, world, sa_n_reads=20, sa_sweeps=200,
                 sa_T_init=2.0, sa_T_final=0.05, **kwargs):
        super().__init__(world, **kwargs)
        self._sa_n_reads = sa_n_reads
        self._sa_sweeps = sa_sweeps
        self._sa_T_init = sa_T_init
        self._sa_T_final = sa_T_final

    def act(self, t: int, p: int) -> np.ndarray:
        Q = self.build_qubo()
        gamma = self._temperature(p, t)
        Q_scaled = Q / max(gamma, 1e-8)
        best_x, _ = sa_sweep(
            Q_scaled,
            self.rng,
            n_reads=self._sa_n_reads,
            n_sweeps=self._sa_sweeps,
            T_init=self._sa_T_init,
            T_final=self._sa_T_final,
        )
        return decode_solution(best_x, self.world.N, self.world.K)


# ── Experiment ────────────────────────────────────────────────────────────────

def run_agent(agent_cls, world_kwargs, agent_kwargs, P, T, seed):
    rng = np.random.default_rng(seed + 500)
    world = AbstractWorld(**world_kwargs, seed=seed)
    agent = agent_cls(world, seed=seed, **agent_kwargs)

    losses = np.zeros((P, T), dtype=float)
    for p in range(P):
        agent.reset_epoch(p)
        for t in range(T):
            chosen = agent.act(t, p)
            loss = world.compute_losses(chosen, rng)
            agent.update(chosen, loss)
            losses[p, t] = loss.mean()
    return losses


def main(N=5, K=4, m=15, Z=6, P=8, T=100, n_seeds=10):
    print(f"Comparing sa_onehot vs sa_sweep  N={N} K={K} P={P} T={T} seeds={n_seeds}")

    world_kwargs = dict(N=N, K=K, m=m, Z=Z)

    # sa_onehot: route-flip (new default)
    onehot_kwargs = dict(sa_n_restarts=20, sa_n_iters=1000, sa_T0=2.0, sa_decay=0.999)
    # sa_sweep: bit-flip (original)
    sweep_kwargs  = dict(sa_n_reads=20, sa_sweeps=200, sa_T_init=2.0, sa_T_final=0.05)

    all_onehot, all_sweep = [], []

    for s in range(n_seeds):
        seed = 42 + s
        print(f"  seed {seed}...", flush=True)
        l_onehot = run_agent(QAMABPhysical, world_kwargs, onehot_kwargs, P, T, seed)
        l_sweep  = run_agent(QAMABSweep,   world_kwargs, sweep_kwargs,  P, T, seed)
        all_onehot.append(l_onehot)
        all_sweep.append(l_sweep)

    # (n_seeds, P, T) → flatten P*T into steps
    arr_onehot = np.array(all_onehot).reshape(n_seeds, P * T)
    arr_sweep  = np.array(all_sweep).reshape(n_seeds, P * T)

    # Mean across seeds
    mean_onehot = arr_onehot.mean(axis=0)
    mean_sweep  = arr_sweep.mean(axis=0)

    # Cumulative mean loss (proxy for regret)
    cum_onehot = np.cumsum(mean_onehot) / (np.arange(P * T) + 1)
    cum_sweep  = np.cumsum(mean_sweep)  / (np.arange(P * T) + 1)

    # Rolling mean (window=10) for cleaner plot
    def rolling(x, w=10):
        return np.convolve(x, np.ones(w) / w, mode='valid')

    steps = np.arange(P * T)

    # ── Plot 1: rolling mean loss ──────────────────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.plot(steps[9:], rolling(mean_onehot), label='sa_onehot (route-flip)', color='tab:blue', lw=1.5)
    plt.plot(steps[9:], rolling(mean_sweep),  label='sa_sweep  (bit-flip)',   color='tab:orange', lw=1.5)
    # Epoch boundaries
    for ep in range(1, P):
        plt.axvline(ep * T, color='gray', lw=0.6, ls='--', alpha=0.5)
    plt.xlabel('Step')
    plt.ylabel('Mean Loss (rolling 10)')
    plt.title(f'sa_onehot vs sa_sweep — N={N} K={K}  ({n_seeds} seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT + 'rolling_loss.png', dpi=150)
    plt.close()

    # ── Plot 2: cumulative mean (regret proxy) ─────────────────────────────────
    plt.figure(figsize=(9, 4))
    plt.plot(steps, cum_onehot, label='sa_onehot (route-flip)', color='tab:blue', lw=1.5)
    plt.plot(steps, cum_sweep,  label='sa_sweep  (bit-flip)',   color='tab:orange', lw=1.5)
    for ep in range(1, P):
        plt.axvline(ep * T, color='gray', lw=0.6, ls='--', alpha=0.5)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Mean Loss / Step')
    plt.title(f'Regret proxy — N={N} K={K}  ({n_seeds} seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT + 'cumulative_regret.png', dpi=150)
    plt.close()

    # ── Per-epoch summary ──────────────────────────────────────────────────────
    epoch_onehot = arr_onehot.reshape(n_seeds, P, T).mean(axis=(0, 2))
    epoch_sweep  = arr_sweep.reshape(n_seeds, P, T).mean(axis=(0, 2))

    plt.figure(figsize=(7, 4))
    x = np.arange(P)
    plt.plot(x, epoch_onehot, 'o-', label='sa_onehot', color='tab:blue', lw=1.5)
    plt.plot(x, epoch_sweep,  's-', label='sa_sweep',  color='tab:orange', lw=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.title(f'Per-epoch convergence — N={N} K={K}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT + 'per_epoch.png', dpi=150)
    plt.close()

    # ── Save results ───────────────────────────────────────────────────────────
    final_onehot = float(epoch_onehot[-1])
    final_sweep  = float(epoch_sweep[-1])
    winner = 'sa_onehot' if final_onehot < final_sweep else 'sa_sweep'

    result = {
        "config": {"N": N, "K": K, "m": m, "Z": Z, "P": P, "T": T, "n_seeds": n_seeds},
        "sa_onehot": {"params": onehot_kwargs, "final_epoch_loss": final_onehot,
                      "epoch_means": epoch_onehot.tolist()},
        "sa_sweep":  {"params": sweep_kwargs,  "final_epoch_loss": final_sweep,
                      "epoch_means": epoch_sweep.tolist()},
        "winner": winner,
        "delta": final_onehot - final_sweep,
    }
    with open(OUT + 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  sa_onehot final loss: {final_onehot:.4f}")
    print(f"  sa_sweep  final loss: {final_sweep:.4f}")
    print(f"  Winner: {winner}  (delta={final_onehot - final_sweep:+.4f})")
    print(f"  Results → {OUT}")
    return result


if __name__ == '__main__':
    main()
