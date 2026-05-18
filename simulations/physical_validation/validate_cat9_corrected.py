#!/usr/bin/env python3
"""Cat 9 (corrected): Learning-quality regret — QA-MAB vs OracleSA.

Both agents use the SAME SA solver and SAME QUBO structure. The only
difference is the parameter init:
  • QA-MAB:   theta_hat=0, phi_hat=0, learns via residual credit assignment.
  • OracleSA: theta_hat=theta*, phi_hat=phi* (perfect), update() is a no-op.

This isolates *learning quality* from *SA solver accuracy*: any positive
regret is attributable to QA-MAB's parameter estimation error, not to
suboptimality of SA versus an exhaustive optimum.

Run from `simulations/`:
    python3 -m physical_validation.validate_cat9_corrected
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .physical_env import AbstractWorld
from .qa_mab_physical import QAMABPhysical
from .sa_solver_physical import sa_solve, decode_solution


# ─── Config ────────────────────────────────────────────────────────────
N, K, m, Z = 5, 4, 20, 6
P, T = 30, 150
n_seeds = 8
sigmas = [0.0, 0.1]

OUT = "results/validation_cat9_physical/"
os.makedirs(OUT, exist_ok=True)


# ─── OracleSA agent ────────────────────────────────────────────────────

class OracleSA(QAMABPhysical):
    """Same SA solver + same QUBO form as QA-MAB, but with theta_hat=theta*,
    phi_hat=phi*. update() is a no-op so the oracle estimates never drift.

    UCB bonus is zeroed (the oracle has no need to explore).
    """

    name = "OracleSA"

    def __init__(self, world: AbstractWorld, seed: int = 42, **kwargs):
        kwargs.setdefault("ucb_c", 0.0)
        super().__init__(world, seed=seed, **kwargs)
        self.theta_hat = world.theta_star.copy()
        self.phi_hat = world.phi_star.copy()

    def reset_epoch(self, p: int, world_rng=None) -> None:
        self._visit_counts = np.zeros((self.world.N, self.world.K), dtype=int)
        self.world.refresh_epoch(world_rng if world_rng is not None else self.rng)
        self.theta_hat = self.world.theta_star.copy()
        self.phi_hat = self.world.phi_star.copy()

    def update(self, chosen_paths: np.ndarray, losses: np.ndarray) -> None:
        N = self.world.N
        for n in range(N):
            self._visit_counts[n, chosen_paths[n]] += 1


# ─── Per-seed run ──────────────────────────────────────────────────────

def run_one(sigma: float, seed: int):
    """Returns dict with per-epoch regret, theta_err, phi_err arrays of len P."""
    world_qa = AbstractWorld(N=N, K=K, m=m, Z=Z, sigma_noise=sigma, seed=seed)
    world_or = AbstractWorld(N=N, K=K, m=m, Z=Z, sigma_noise=sigma, seed=seed)

    qa = QAMABPhysical(world_qa, sigma_noise=sigma, seed=seed)
    oracle = OracleSA(world_or, sigma_noise=sigma, seed=seed + 10_000)

    rng_qa = np.random.default_rng(seed + 1)
    rng_or = np.random.default_rng(seed + 2)

    # Shared RNG that drives per-epoch world refresh for BOTH agents. Without
    # this, each agent's reset_epoch() would use its own RNG and resample a
    # different topology (path memberships, distances), making the regret
    # measure world-difficulty difference instead of learning gap.
    shared_world_rng = np.random.default_rng(seed + 50_000)

    qa_epoch_mean = np.zeros(P, dtype=float)
    or_epoch_mean = np.zeros(P, dtype=float)
    theta_err = np.zeros(P, dtype=float)
    phi_err = np.zeros(P, dtype=float)

    for p in range(P):
        epoch_seed = int(shared_world_rng.integers(0, 2**63 - 1))
        qa.reset_epoch(p, world_rng=np.random.default_rng(epoch_seed))
        oracle.reset_epoch(p, world_rng=np.random.default_rng(epoch_seed))

        qa_losses = []
        or_losses = []
        for t in range(T):
            c_qa = qa.act(t, p)
            l_qa = world_qa.compute_losses(c_qa, rng_qa)
            qa.update(c_qa, l_qa)
            qa_losses.append(float(l_qa.mean()))

            c_or = oracle.act(t, p)
            l_or = world_or.compute_losses(c_or, rng_or)
            oracle.update(c_or, l_or)
            or_losses.append(float(l_or.mean()))

        qa_epoch_mean[p] = float(np.mean(qa_losses))
        or_epoch_mean[p] = float(np.mean(or_losses))
        theta_err[p] = float(np.linalg.norm(qa.theta_hat - world_qa.theta_star) / m)
        phi_err[p] = float(np.linalg.norm(qa.phi_hat - world_qa.phi_star) / Z)

    regret = qa_epoch_mean - or_epoch_mean
    return dict(
        regret=regret,
        theta_err=theta_err,
        phi_err=phi_err,
        qa_epoch_mean=qa_epoch_mean,
        or_epoch_mean=or_epoch_mean,
    )


# ─── Main sweep ────────────────────────────────────────────────────────

def main():
    per_sigma = {}
    pass_flags = []

    init_window = max(1, P // 10)   # first 10% of epochs
    final_window = max(1, P // 10)  # last 10% of epochs

    for sigma in sigmas:
        print(f"sigma={sigma}", flush=True)
        regret_runs = np.zeros((n_seeds, P), dtype=float)
        theta_runs = np.zeros((n_seeds, P), dtype=float)
        phi_runs = np.zeros((n_seeds, P), dtype=float)

        for si in range(n_seeds):
            seed = 100 + si
            print(f"  seed={seed}", flush=True)
            res = run_one(sigma, seed)
            regret_runs[si] = res["regret"]
            theta_runs[si] = res["theta_err"]
            phi_runs[si] = res["phi_err"]

        regret_mean = regret_runs.mean(axis=0)
        regret_std = regret_runs.std(axis=0)
        theta_mean = theta_runs.mean(axis=0)
        phi_mean = phi_runs.mean(axis=0)

        init_regret_mean = float(regret_mean[:init_window].mean())
        final_regret_mean = float(regret_mean[-final_window:].mean())
        sigma_pass = bool(final_regret_mean < init_regret_mean)
        pass_flags.append(sigma_pass)

        mid_idx = P // 2
        mid_regret_mean = float(regret_mean[mid_idx])

        print(
            f"  init={init_regret_mean:.4f}  mid={mid_regret_mean:.4f}  "
            f"final={final_regret_mean:.4f}  "
            f"theta_err: {theta_mean[0]:.4f}→{theta_mean[-1]:.4f}  "
            f"{'PASS' if sigma_pass else 'FAIL'}",
            flush=True,
        )

        per_sigma[str(sigma)] = dict(
            init_regret_mean=init_regret_mean,
            mid_regret_mean=mid_regret_mean,
            final_regret_mean=final_regret_mean,
            regret_mean=regret_mean.tolist(),
            regret_std=regret_std.tolist(),
            theta_err_mean=theta_mean.tolist(),
            phi_err_mean=phi_mean.tolist(),
            pass_=sigma_pass,
        )

    pass_rate = sum(pass_flags) / len(pass_flags)
    overall_pass = pass_rate >= 0.5

    result = dict(
        pass_=overall_pass,
        pass_rate=pass_rate,
        reason=(
            f"{sum(pass_flags)}/{len(pass_flags)} sigmas with "
            f"final_regret_mean < init_regret_mean"
        ),
        config=dict(
            N=N, K=K, m=m, Z=Z, P=P, T=T,
            n_seeds=n_seeds, sigmas=sigmas,
            init_window=init_window, final_window=final_window,
        ),
        per_sigma=per_sigma,
    )

    with open(os.path.join(OUT, "cat9_corrected_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────────
    epochs = np.arange(P)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sigma in sigmas:
        d = per_sigma[str(sigma)]
        rm = np.array(d["regret_mean"])
        rs = np.array(d["regret_std"])
        ax.plot(epochs, rm, marker="o", lw=1.5, label=f"sigma={sigma}")
        ax.fill_between(epochs, rm - rs, rm + rs, alpha=0.15)
    ax.axhline(0.0, color="gray", linestyle=":")
    ax.set_xlabel("epoch p")
    ax.set_ylabel("regret = mean(QA loss) − mean(Oracle loss)")
    ax.set_title("Cat 9 corrected: Learning-quality regret (same SA solver)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cat9_corrected_regret.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sigma in sigmas:
        d = per_sigma[str(sigma)]
        ax.plot(epochs, d["theta_err_mean"], marker="o", lw=1.5,
                label=f"theta_err sigma={sigma}")
        ax.plot(epochs, d["phi_err_mean"], marker="s", lw=1.0,
                linestyle="--", label=f"phi_err sigma={sigma}")
    ax.set_xlabel("epoch p")
    ax.set_ylabel("normalised L2 error")
    ax.set_title("Cat 9 corrected: theta_hat / phi_hat convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cat9_corrected_errors.png"), dpi=150)
    plt.close(fig)

    print(
        f"Cat 9 corrected: {sum(pass_flags)}/{len(pass_flags)} sigmas pass — "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
