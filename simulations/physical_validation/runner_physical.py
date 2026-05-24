"""Experiment runner for physical abstract model."""
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
    """Run all agents on the same seeds/world for comparison.

    Every agent receives:
    - an identical AbstractWorld (same θ*/φ* drawn from the same seed)
    - the same world topology at every epoch (shared deterministic epoch seeds)
    - the same loss-noise sequence (fresh per-agent rng from the same seed)

    Previously agents shared one world object and each used their own RNG
    for world.refresh_epoch, so they saw different topologies. NB3R and
    Random never refreshed the world at all. Both issues invalidated
    cross-agent loss comparisons.
    """
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

        # One deterministic world-topology seed per epoch, shared across all agents.
        # Multiplier keeps these far from agent/loss seed space.
        epoch_seeds = [seed * 100_000 + p for p in range(P)]

        for name in agents:
            if name not in agent_map:
                continue

            # Fresh world per agent: same θ*/φ* (same seed), independent internal RNG
            # so one agent's reset_epoch calls cannot advance another agent's world state.
            world = AbstractWorld(N=N, K=K, m=m, Z=Z, seed=seed, **kwargs)
            agent = agent_map[name](world, seed)

            losses_log = np.zeros((P, T, N), dtype=float)
            theta_err_log = np.zeros(P, dtype=float)
            phi_err_log = np.zeros(P, dtype=float)
            chosen_log = np.zeros((P, T, N), dtype=int)

            # Fresh loss RNG per agent with the same base seed — same noise pattern.
            rng_ep = np.random.default_rng(seed + 1_000_000)

            for p in range(P):
                # Build a fresh world RNG from the shared epoch seed so every
                # agent sees the identical path topology at epoch p.
                world_rng = np.random.default_rng(epoch_seeds[p])

                if hasattr(agent, 'world'):
                    # QAMABPhysical subclasses: reset_epoch handles decay +
                    # visit-count reset + world refresh.
                    agent.reset_epoch(p, world_rng=world_rng)
                else:
                    # NB3R / Random: no internal state to decay, but the world
                    # topology must still be refreshed for each epoch.
                    if hasattr(agent, 'reset_epoch'):
                        agent.reset_epoch()
                    world.refresh_epoch(world_rng)

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
