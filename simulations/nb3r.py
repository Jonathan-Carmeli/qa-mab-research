"""
nb3r.py
NB3R = Neighbour-Based Bandit Broadcast (Classical Distributed)
Implementation of Algorithm 1 from the DIAMOND extension.

Each agent learns locally via softmax/Gibbs sampling and
broadcasts observed throughput to neighbours within interference radius.
"""

import numpy as np


class NB3R:
    """
    NB3R algorithm for distributed multi-agent bandit routing.

    Args:
        env: NetworkEnvironment instance
        tau0: Initial temperature (default 0.1)
        delta_tau: Temperature increment per step (default 0.05)
        alpha: Learning rate for weight updates (default 0.3)
        interference_radius: Neighbourhood radius in hops (default 1)
        seed: Random seed (default 42)
    """

    def __init__(self, env, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=42):
        self.env = env
        self.N = env.N
        self.m = env.m
        self.tau0 = tau0
        self.delta_tau = delta_tau
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        # Weights: W[i, k] for agent i, route k
        self.W = np.zeros((self.N, self.m))

        # Current temperature
        self.tau = tau0

        # History
        self.history = []

    def _neighbors(self, agent):
        """Return all other agents as neighbours (fully-connected interference model)."""
        return [j for j in range(self.N) if j != agent]

    def _softmax_probs(self, agent):
        """
        Compute softmax probabilities for agent i:
        P_i,k = exp(tau * W_i,k) / sum_l exp(tau * W_i,l)
        """
        w = self.W[agent]
        # Numerical stability: subtract max
        w_shifted = self.tau * (w - np.max(w))
        exp_w = np.exp(w_shifted)
        return exp_w / np.sum(exp_w)

    def _pick_route(self, agent):
        """Sample a route for agent i based on softmax probabilities."""
        probs = self._softmax_probs(agent)
        return self.rng.choice(self.m, p=probs)

    def step(self):
        """
        Execute one step of NB3R:
        1. Pick routes (softmax sampling)
        2. Measure throughput
        3. Broadcast to neighbours
        4. Update weights
        5. Increase tau
        """
        # 1. Pick routes for all agents
        chosen_routes = {i: self._pick_route(i) for i in range(self.N)}

        # 2. Measure throughput
        throughputs = self.env.compute_throughput(chosen_routes)

        # 3. Broadcast: each agent shares U_i with neighbours
        #    (stored as received_U[j] for agent i)
        received_U = {}
        for i in range(self.N):
            neighbours = self._neighbors(i)
            received_U[i] = {j: throughputs[j] for j in neighbours}

        # 4. Update weights for each agent
        for i in range(self.N):
            chosen_k = chosen_routes[i]
            # Sum of own throughput + neighbours' throughputs
            total_signal = throughputs[i] + sum(received_U[i].values())
            # Exponential moving average update
            self.W[i, chosen_k] = (1 - self.alpha) * self.W[i, chosen_k] + self.alpha * total_signal

        # 5. Increase tau
        self.tau += self.delta_tau

        # Record social welfare
        sw = self.env.social_welfare(chosen_routes)
        self.history.append(sw)

    def run(self, T):
        """
        Run NB3R for T steps.
        Returns history of social welfare.
        """
        for _ in range(T):
            self.step()
        return np.array(self.history)
