"""
simulation_core.py
Ground-truth network environment (unknown to algorithms).

B[i, k] = base utility for agent i on route k (higher is better)
I[i,k,j,l] = interference: penalty when agent j uses route l and it conflicts with agent i on route k

Scale: B in [0, 1], I in [0, 0.1]
This makes interference significant but not overwhelming — the optimization problem is meaningful.
With N=10, m=4, average total interference is ~3.6 vs total base utility ~20.
"""

import numpy as np


class NetworkEnvironment:
    def __init__(self, N, m, seed=42, B_scale='uniform', I_scale='moderate'):
        """
        Args:
            N: number of agents
            m: number of routes per agent
            seed: random seed
            B_scale: 'uniform' (0.5-1.0) or 'skewed' (some routes much better)
            I_scale: 'low' (0-0.1), 'moderate' (0-0.2), 'high' (0-0.3)
        """
        self.N = N
        self.m = m
        rng = np.random.default_rng(seed)
        
        # Base utilities
        if B_scale == 'uniform':
            self.B = rng.uniform(0.5, 1.0, size=(N, m))
        elif B_scale == 'skewed':
            # Skew: most routes are mediocre, some are great
            base = rng.uniform(0.1, 0.3, size=(N, m))
            n_best = max(1, m // 4)  # 25% of routes are best
            for i in range(N):
                best_indices = rng.choice(m, n_best, replace=False)
                base[i, best_indices] = rng.uniform(0.8, 1.0, size=n_best)
            self.B = base
        
        # Interference scale
        if I_scale == 'low':
            I_max = 0.1
        elif I_scale == 'moderate':
            I_max = 0.2
        elif I_scale == 'high':
            I_max = 0.3
        
        self.I = rng.uniform(0, I_max, size=(N, m, N, m))
        # No self-interference
        for i in range(N):
            self.I[i, :, i, :] = 0.0

    def compute_throughput(self, assignment):
        """
        assignment: dict {agent_i: route_k}
        U_i = B[i, k] - sum_{j != i} sum_l I[i,k,j,l] * x[j,l]
        """
        x = np.zeros((self.N, self.m))
        for agent, route in assignment.items():
            x[agent, route] = 1.0

        throughputs = {}
        for i, k in assignment.items():
            interference = np.sum(self.I[i, k] * x)
            throughputs[i] = float(self.B[i, k] - interference)
        return throughputs

    def social_welfare(self, assignment):
        return sum(self.compute_throughput(assignment).values())
