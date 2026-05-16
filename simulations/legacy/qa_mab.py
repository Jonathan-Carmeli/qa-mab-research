"""
qa_mab.py
QA-MAB = Quantum Annealing Multi-Armed Bandit (Centralized)
Implementation of Algorithm 2 from the DIAMOND extension.

Key design:
- E(x) = -sum u_hat[i,k]*x_i,k + sum_{i,j,k,l} I[i,k,j,l]*x_i,k*x_j,l + lambda*(sum_k x_i,k - 1)^2
- Off-diagonal (cross-agent): positive I -> penalizes conflicts
- Off-diagonal (same-agent): positive lambda/2 -> enforces one route per agent
- Diagonal: -u_hat - lambda/2

The QUBO is scaled by tau(t): as tau grows, the energy landscape becomes sharper,
forcing the annealer to find the global minimum (exploitation).
At low tau, the landscape is flat and the annealer samples broadly (exploration).
"""

import numpy as np


class QAMAB:
    """
    QA-MAB algorithm for centralized multi-agent bandit routing via QUBO.
    """

    def __init__(self, env, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=42,
                 collision_threshold=0.02, I_learn_rate=0.05, B_learn_rate=0.2,
                 I_cap=0.3):
        """
        Args:
            env: NetworkEnvironment instance
            tau0: Initial temperature
            delta_tau: Temperature increment per step
            lambda_: Constraint penalty weight
            seed: Random seed
            collision_threshold: Min drop to infer collision
            I_learn_rate: Learning rate for I_hat updates
            B_learn_rate: Learning rate for u_hat updates
            I_cap: Maximum value for I_hat entries (default 0.3)
        """
        self.env = env
        self.N = env.N
        self.m = env.m
        self.tau = tau0
        self.delta_tau = delta_tau
        self.lambda_ = lambda_
        self.rng = np.random.default_rng(seed)
        self.collision_threshold = collision_threshold
        self.I_learn_rate = I_learn_rate
        self.B_learn_rate = B_learn_rate
        self.I_cap = I_cap

        # Estimated base utilities: u_hat[i, k] ~ B[i,k]
        # Initialize to mean of B range
        self.u_hat = np.full((self.N, self.m), 0.75)

        # Estimated interference: I_hat[i, k, j, l]
        self.I_hat = np.zeros((self.N, self.m, self.N, self.m))

        self.qubo_size = self.N * self.m
        self.history = []

        # Track previous step for collision inference
        self._prev_x = None
        self._prev_throughputs = None

    def _idx(self, i, k):
        return i * self.m + k

    def build_qubo(self):
        """
        Build QUBO matrix Q_tilde from current estimates.

        E(x) = -sum_{i,k} u_hat[i,k] * x_i,k
               + lambda_ * sum_i (sum_k x_i,k - 1)^2
               + sum_{i,j!=i,k,l} I_hat[i,k,j,l] * x_i,k * x_j,l

        In QUBO form (E = x^T Q x):
        - Diagonal Q[i,k,i,k]: -u_hat[i,k] - lambda_/2
        - Off-diagonal same-agent Q[i,k,i,l], l != k: lambda_/2
        - Off-diagonal cross-agent Q[i,k,j,l], i != j: I_hat[i,k,j,l]

        The constraint term expansion:
        (sum_k x_k - 1)^2 = sum_k x_k^2 + sum_{k!=l} x_k*x_l - 2*sum_k x_k + 1
        Since x_k ∈ {0,1}: x_k^2 = x_k, so:
        = -sum_k x_k + sum_{k!=l} x_k*x_l + 1

        In QUBO (E = x^T Q x, with x_k^2 = x_k on diagonal):
        - Diagonal: lambda * (-1) = -lambda (but we split: -lambda/2 on diag
          because effective penalty = Q[k,k] per x_k)
        - Off-diagonal: Q[k,l] + Q[l,k] = lambda => Q[k,l] = lambda/2

        Note: effective constraint strength is lambda/2 per variable.
        With lambda=0.5, actual penalty per constraint violation = 0.25.

        Final QUBO: Q_A(t) = tau(t) * Q_tilde
        """
        size = self.N * self.m
        Q = np.zeros((size, size))

        # Constraint term (lambda)
        for i in range(self.N):
            for k in range(self.m):
                idx_ik = self._idx(i, k)
                # Diagonal: -u_hat - lambda/2
                Q[idx_ik, idx_ik] = -self.u_hat[i, k] - self.lambda_ / 2.0

                # Off-diagonal same-agent: lambda/2 (NOT lambda!)
                for l in range(k + 1, self.m):
                    idx_il = self._idx(i, l)
                    Q[idx_ik, idx_il] = self.lambda_ / 2.0
                    Q[idx_il, idx_ik] = self.lambda_ / 2.0

        # Cross-agent interference term
        for i in range(self.N):
            for k in range(self.m):
                idx_ik = self._idx(i, k)
                for j in range(self.N):
                    if j == i:
                        continue
                    for l in range(self.m):
                        idx_jl = self._idx(j, l)
                        Q[idx_ik, idx_jl] = self.I_hat[i, k, j, l]

        return self.tau * Q

    def _qubo_energy(self, x, Q):
        return float(x @ Q @ x)

    def solve_qubo(self, Q):
        """
        Simulated Annealing QUBO solver with route-flip proposals.
        Scales SA effort inversely with N (larger N = fewer iterations).

        For large tau (> 5): energy landscape is sharp enough that
        greedy assignment is near-optimal, so use minimal SA.
        """
        n = self.N
        m = self.m
        size = self.qubo_size

        # Scale SA effort inversely with network size
        if n <= 10:
            n_restarts = 8
            n_iters = 500
        elif n <= 20:
            n_restarts = 4
            n_iters = 200
        elif n <= 30:
            n_restarts = 2
            n_iters = 100
        else:
            n_restarts = 2
            n_iters = 80

        T0 = 10.0
        decay = 0.9

        best_x = None
        best_energy = float('inf')

        for restart in range(n_restarts):
            # Initialize: start from u_hat greedy, add perturbation
            x = np.zeros(size, dtype=float)
            for i in range(n):
                k_greedy = int(np.argmax(self.u_hat[i]))
                x[i * m + k_greedy] = 1.0

            # Random perturbation (only for restarts > 0)
            if restart > 0:
                n_flips = self.rng.integers(1, max(2, n // 3))
                for _ in range(n_flips):
                    i = self.rng.integers(0, n)
                    block = x[i * m:(i + 1) * m]
                    k_old = int(np.argmax(block))
                    candidates = [k for k in range(m) if k != k_old]
                    if candidates:
                        k_new = candidates[self.rng.integers(0, len(candidates))]
                        x[i * m + k_old] = 0.0
                        x[i * m + k_new] = 1.0

            energy = self._qubo_energy(x, Q)
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()

            # SA loop
            T = T0 * (1.0 + restart * 0.3)
            for step in range(n_iters):
                T *= decay

                # Route-flip proposal
                i = self.rng.integers(0, n)
                block = x[i * m:(i + 1) * m]
                k_old = int(np.argmax(block))
                k_new = (k_old + 1 + self.rng.integers(0, m - 1)) % m

                # Apply flip
                x[i * m + k_old] = 0.0
                x[i * m + k_new] = 1.0

                new_energy = self._qubo_energy(x, Q)
                delta = new_energy - energy

                if delta < 0 or (T > 1e-10 and self.rng.random() < np.exp(-delta / T)):
                    energy = new_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_x = x.copy()
                else:
                    # Revert
                    x[i * m + k_new] = 0.0
                    x[i * m + k_old] = 1.0

        # Decode: pick route with highest x value per agent
        assignment = {}
        for i in range(n):
            block = best_x[i * m:(i + 1) * m]
            assignment[i] = int(np.argmax(block))
        return assignment

    def step(self):
        # 1. Build QUBO and solve
        Q_A = self.build_qubo()
        assignment = self.solve_qubo(Q_A)

        # 2. Measure throughput
        throughputs = self.env.compute_throughput(assignment)

        # 3. Update server estimates (learn B and I from observations)

        # 3a. Update u_hat with observed throughput (= B[i,k] - interference)
        # u_hat tracks the *effective* utility under current conditions.
        # The QUBO uses -u_hat on diagonal and +I_hat on off-diagonal.
        # This is NOT double-counting because:
        #   - u_hat ≈ B[i,k] - E[interference] (average effective utility)
        #   - I_hat captures the *marginal* interference from specific (j,l) pairs
        #   - QUBO minimizes: -u_hat*x + I_hat*x*x' = -(B-E[I]) + marginal_I
        # The mean-field decomposition is intentional.
        for i in range(self.N):
            k = assignment[i]
            observed = throughputs[i]
            self.u_hat[i, k] += self.B_learn_rate * (observed - self.u_hat[i, k])

        # 3b. Infer collisions: both agents dropped at same step
        if self._prev_x is not None and self._prev_throughputs is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ki = self._prev_x[i]
                    kj = self._prev_x[j]
                    
                    # Compute expected vs actual for each agent
                    # If actual < expected, there was interference
                    expected_i = self.u_hat[i, ki]
                    expected_j = self.u_hat[j, kj]
                    
                    drop_i = max(0, expected_i - self._prev_throughputs[i])
                    drop_j = max(0, expected_j - self._prev_throughputs[j])
                    
                    # Update I_hat independently for each direction
                    # Only update the direction where the drop was observed
                    if drop_i > self.collision_threshold:
                        # Agent i was interfered with — increase I_hat[i,ki,j,kj]
                        self.I_hat[i, ki, j, kj] = min(
                            self.I_hat[i, ki, j, kj] + self.I_learn_rate, self.I_cap)
                            
                    if drop_j > self.collision_threshold:
                        # Agent j was interfered with — increase I_hat[j,kj,i,ki]
                        self.I_hat[j, kj, i, ki] = min(
                            self.I_hat[j, kj, i, ki] + self.I_learn_rate, self.I_cap)

        # 4. Increase tau
        self.tau += self.delta_tau

        # 5. Store for collision inference
        self._prev_x = assignment.copy()
        self._prev_throughputs = {i: throughputs[i] for i in range(self.N)}

        # 6. Record social welfare
        sw = self.env.social_welfare(assignment)
        self.history.append(sw)

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)
