"""Experiment runner for physical abstract model."""
import os
import numpy as np
from .physical_env import AbstractWorld
from .qa_mab_physical import QAMABPhysical
from .agents_physical import RandomAgent, NB3RAgent, OracleAgent, OptimalAgent


def run_experiment(
    N=3, K=4, m=15, Z=6,
    P=10, T=100,
    n_seeds=20, base_seed=42,
    agents=None,
    **kwargs
):
    """Run all agents on the same seeds/world for comparison."""
    if agents is None:
        agents = ["QA-MAB-Physical", "NB3R-Physical", "Random-Physical", "Oracle-Physical", "Optimal-Physical"]

    results = {name: [] for name in agents}
    agent_map = {
        "QA-MAB-Physical":  lambda world, seed: QAMABPhysical(world, seed=seed, **kwargs),
        "NB3R-Physical":    lambda world, seed: NB3RAgent(world, seed=seed),
        "Random-Physical":  lambda world, seed: RandomAgent(world, seed=seed),
        "Oracle-Physical":  lambda world, seed: OracleAgent(world, seed=seed, **kwargs),
        "Optimal-Physical": lambda world, seed: OptimalAgent(world, seed=seed, **kwargs),
    }

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        rng = np.random.default_rng(seed)

        # Shared world per seed
        world = AbstractWorld(N=N, K=K, m=m, Z=Z, seed=seed, **kwargs)

        for name in agents:
            if name not in agent_map:
                continue
            factory = agent_map[name]
            agent = factory(world, seed)

            losses_log = np.zeros((P, T, N), dtype=float)
            theta_err_log = np.zeros(P, dtype=float)
            phi_err_log = np.zeros(P, dtype=float)
            chosen_log = np.zeros((P, T, N), dtype=int)

            rng_ep = np.random.default_rng(seed + 1000 + seed_idx)
            for p in range(P):
                if hasattr(agent, 'reset_epoch'):
                    # QA-MAB / Oracle / Optimal
                    if hasattr(agent, 'world'):
                        agent.reset_epoch(p)
                    else:
                        agent.reset_epoch()
                for t in range(T):
                    chosen = agent.act(t, p)
                    losses = world.compute_losses(chosen, rng_ep)
                    agent.update(chosen, losses)
                    losses_log[p, t] = losses
                    chosen_log[p, t] = chosen

                if name == "QA-MAB-Physical":
                    theta_err_log[p] = float(np.linalg.norm(agent.theta_hat - world.theta_star))
                    phi_err_log[p] = float(np.linalg.norm(agent.phi_hat - world.phi_star))

            results[name].append(dict(
                losses_log=losses_log,
                theta_err_log=theta_err_log,
                phi_err_log=phi_err_log,
                chosen_log=chosen_log,
            ))

    return results