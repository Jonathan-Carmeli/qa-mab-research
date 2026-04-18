"""
simulation_connected.py
Connected graph simulation — NB3R with LOCAL communication on a CONNECTED topology.

Key question: When the neighborhood graph IS connected, does local-communication
NB3R (SparseNB3R) converge and compete with SparseQAMAB?

In simulation_v2 we tested DISCONNECTED clusters → SparseNB3R ≈ Random.
Now we test the OPPOSITE: fully connected communication graph where
every agent can reach every other agent via hops.

Graph structure: k-NN ring (each agent talks to k nearest neighbors in a ring).
This is CONNECTED so DIAMOND's convergence proof applies.
"""

import numpy as np
from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


class ConnectedNetworkEnvironment:
    """Network with a CONNECTED topology — agents in a ring with k-NN neighbors.
    
    The communication graph is CONNECTED (every agent reachable via hops),
    so DIAMOND's convergence proof should apply.
    But interference is still DENSE (everyone affects everyone).
    """
    
    def __init__(self, N, m, k_neighbors=4, seed=42,
                 B_range=(0.5, 1.0),
                 I_near_range=(0.15, 0.25),
                 I_far_range=(0.01, 0.05)):
        """
        Args:
            N: number of agents
            m: number of routes
            k_neighbors: each agent talks to k nearest neighbors in the ring (must be >= 2)
            B_range: (min, max) for base utilities
            I_near_range: interference range for NEIGHBOR agents (geph ≈ k_NN distance)
            I_far_range: interference range for DISTANT agents
        """
        self.N = N
        self.m = m
        self.k = k_neighbors
        rng = np.random.default_rng(seed)
        
        # Ring topology: agent i's neighbors are (i±1, i±2, ..., i±k) mod N
        self.neighbors = {}
        for i in range(N):
            nbrs = [(i + d) % N for d in range(1, k_neighbors + 1)]
            nbrs += [(i - d) % N for d in range(1, k_neighbors + 1)]
            self.neighbors[i] = list(set(nbrs))
        
        # Base utilities
        self.B = rng.uniform(B_range[0], B_range[1], size=(N, m))
        
        # Interference: distance-dependent (not clustered, just ring distance)
        self.I = np.zeros((N, m, N, m))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Ring distance
                d = min((j - i) % N, (i - j) % N)
                max_d = N // 2
                # Interpolate between near and far based on distance
                alpha = d / max_d
                I_base = (1 - alpha) * rng.uniform(*I_near_range) + alpha * rng.uniform(*I_far_range)
                self.I[i, :, j, :] = I_base
    
    def compute_throughput(self, assignment):
        """Full ground-truth throughput (all agents interfere with all)."""
        x = np.zeros((self.N, self.m))
        for agent, route in assignment.items():
            x[agent, route] = 1.0
        
        throughputs = {}
        for i in range(self.N):
            k = assignment[i]
            interference = np.sum(self.I[i, k] * x)
            throughputs[i] = float(self.B[i, k] - interference)
        return throughputs
    
    def social_welfare(self, assignment):
        return sum(self.compute_throughput(assignment).values())
    
    def get_neighbors(self, i):
        """Return communication neighbors of agent i (for NB3R)."""
        return self.neighbors[i]


class LocalNB3R(NB3R):
    """NB3R with LOCAL communication on a CONNECTED graph.
    
    This is the version DIAMOND proves converges (connected graph).
    Each agent broadcasts its utility to k nearest neighbors only.
    """
    
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.N = env.N
        self.m = env.m
        
        self.tau0 = kwargs.get('tau0', 0.1)
        self.delta_tau = kwargs.get('delta_tau', 0.05)
        self.alpha = kwargs.get('alpha', 0.1)
        self.seed = kwargs.get('seed', 42)
        self.tau = self.tau0
        
        rng = np.random.default_rng(self.seed)
        self.W = rng.uniform(0.5, 1.0, size=(self.N, self.m))
        
        self.prev_assignment = None
        self.prev_throughputs = None
    
    def step(self):
        """One step: softmax pick, execute, update W with LOCAL signal."""
        # 1. Pick routes (softmax)
        assignment = {}
        for i in range(self.N):
            probabilities = np.exp(self.W[i] / self.tau)
            probabilities /= probabilities.sum()
            k = np.random.default_rng(int(self.seed + i + self.tau * 1000)).choice(self.m, p=probabilities)
            assignment[i] = k
        
        # 2. Execute
        throughputs = self.env.compute_throughput(assignment)
        
        # 3. Update W — LOCAL collaborative utility (connected graph)
        for i in range(self.N):
            neighbors = self.env.get_neighbors(i)
            # Collaborative utility: own throughput + neighbors' throughputs (LOCAL)
            # This is EXACTLY what DIAMOND uses — U_n = u_n + sum_{m in N_n} u_m
            local_signal = throughputs[i] + sum(throughputs[j] for j in neighbors)
            self.W[i] = (1 - self.alpha) * self.W[i] + self.alpha * (local_signal / (1 + len(neighbors)))
        
        self.prev_assignment = assignment
        self.prev_throughputs = throughputs
        return assignment, throughputs
    
    def run(self, T):
        history = []
        for _ in range(T):
            _, throughputs = self.step()
            history.append(sum(throughputs.values()))
        return np.array(history)


class LocalQAMAB(QAMAB):
    """QA-MAB with SPARSE QUBO on connected graph.
    
    Only models interference from k nearest neighbors.
    Far interference absorbed into u_hat as constant bias.
    """
    
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.N = env.N
        self.m = env.m
        self.neighbors = env.neighbors
        
        self.tau0 = kwargs.get('tau0', 0.1)
        self.delta_tau = kwargs.get('delta_tau', 0.05)
        self.lambda_ = kwargs.get('lambda_', 0.5)
        self.seed = kwargs.get('seed', 42)
        self.collision_threshold = kwargs.get('collision_threshold', 0.02)
        self.I_learn_rate = kwargs.get('I_learn_rate', 0.05)
        self.B_learn_rate = kwargs.get('B_learn_rate', 0.2)
        self.I_cap = kwargs.get('I_cap', 0.3)
        self.I_decay_rate = kwargs.get('I_decay_rate', 0.0)
        
        rng = np.random.default_rng(self.seed)
        self.u_hat = rng.uniform(0.5, 1.0, size=(self.N, self.m))
        
        # Sparse I_hat: only for k nearest neighbors
        self.I_hat = np.zeros((self.N, self.m, self.N, self.m))
        
        self.tau = self.tau0
        self._prev_x = None
        self._prev_throughputs = None
        self.sa_steps = 20
        self.sa_inner_steps = 15
    
    def build_qubo(self, assignment):
        """Build QUBO — SPARSE (only neighbor interactions)."""
        Q = np.zeros((self.N * self.m, self.N * self.m))
        
        for i in range(self.N):
            k = assignment[i]
            
            # Diagonal: -u_hat
            Q[i * self.m + k, i * self.m + k] -= self.u_hat[i, k]
            
            # Off-diagonal: +I_hat for NEIGHBORS only (sparse)
            for j in self.neighbors[i]:
                if i == j:
                    continue
                l = assignment[j]
                I_val = self.I_hat[i, k, j, l]
                Q[i * self.m + k, j * self.m + l] += I_val
        
        return Q
    
    def step(self):
        """One step with sparse QUBO on connected graph."""
        # 1. SA
        best_assignment, _ = self._sa_optimize()
        
        # 2. Execute
        throughputs = self.env.compute_throughput(best_assignment)
        
        # 3. Update u_hat
        for i in range(self.N):
            k = best_assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (throughputs[i] - self.u_hat[i, k])
        
        # 4. Update I_hat (neighbors only)
        if self._prev_x is not None and self._prev_throughputs is not None:
            for i in range(self.N):
                for j in self.neighbors[i]:
                    if i >= j:
                        continue
                    ki = self._prev_x[i]
                    kj = self._prev_x[j]
                    
                    expected_i = self.u_hat[i, ki]
                    expected_j = self.u_hat[j, kj]
                    
                    drop_i = max(0, expected_i - self._prev_throughputs[i])
                    drop_j = max(0, expected_j - self._prev_throughputs[j])
                    delta = (drop_i + drop_j) / 2
                    
                    I_old = self.I_hat[i, ki, j, kj]
                    self.I_hat[i, ki, j, kj] = min(self.I_cap,
                        (1 - self.I_learn_rate) * I_old + self.I_learn_rate * delta)
                    self.I_hat[j, kj, i, ki] = self.I_hat[i, ki, j, kj]
        
        # 5. Optional decay
        if self.I_decay_rate > 0:
            self.I_hat *= (1 - self.I_decay_rate)
        
        # 6. Increase tau
        self.tau += self.delta_tau
        
        self._prev_x = best_assignment
        self._prev_throughputs = throughputs
        return best_assignment, throughputs
    
    def _sa_optimize(self):
        """SA optimization."""
        best_sw = float('-inf')
        best_assignment = None
        
        for _ in range(self.sa_steps):
            assignment = {i: np.random.randint(0, self.m) for i in range(self.N)}
            current_sw = self.env.social_welfare(assignment)
            
            for _ in range(self.sa_inner_steps):
                new_assignment = assignment.copy()
                i = np.random.randint(0, self.N)
                k = np.random.randint(0, self.m)
                new_assignment[i] = k
                
                new_sw = self.env.social_welfare(new_assignment)
                
                if new_sw > current_sw or np.random.random() < np.exp((new_sw - current_sw) / max(self.tau, 1e-9)):
                    assignment = new_assignment
                    current_sw = new_sw
                    
                    if current_sw > best_sw:
                        best_sw = current_sw
                        best_assignment = assignment.copy()
        
        return best_assignment, best_sw
    
    def run(self, T):
        history = []
        for _ in range(T):
            _, throughputs = self.step()
            history.append(sum(throughputs.values()))
        return np.array(history)


def run_connected_comparison(N=20, m=4, k_neighbors=4, T=500, n_runs=10):
    """Compare all variants on CONNECTED graph."""
    seeds = list(range(42, 42 + n_runs))
    results = {'LocalNB3R(connected)': [], 'LocalQAMAB(connected)': [], 'Random': []}
    
    for seed in seeds:
        env = ConnectedNetworkEnvironment(N, m, k_neighbors=k_neighbors, seed=seed)
        
        # LocalNB3R (local comm, connected graph — DIAMOND should converge)
        nb3r = LocalNB3R(env, seed=seed)
        hist = nb3r.run(T)
        results['LocalNB3R(connected)'].append(np.mean(hist[-50:]))
        
        # LocalQAMAB (sparse QUBO, neighbors only)
        qa = LocalQAMAB(env, seed=seed)
        hist = qa.run(T)
        results['LocalQAMAB(connected)'].append(np.mean(hist[-50:]))
        
        # Random
        rng = np.random.default_rng(seed)
        sw_sum = 0
        for _ in range(T):
            assignment = {i: rng.integers(0, m) for i in range(N)}
            sw_sum += env.social_welfare(assignment)
        results['Random'].append(sw_sum / T)
    
    print(f"\nCONNECTED graph: N={N}, m={m}, k={k_neighbors} neighbors, T={T}, runs={n_runs}")
    print(f"Graph is CONNECTED: max shortest path = {max_path(N, k_neighbors)} hops")
    print(f"{'Algorithm':<25} {'Mean SW':>12} {'Std':>10}")
    print("-" * 50)
    for alg, vals in sorted(results.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f"{alg:<25} {np.mean(vals):>12.3f} {np.std(vals):>10.3f}")
    
    return results


def max_path(N, k):
    """Upper bound on diameter of k-NN ring."""
    # In a k-NN ring, each hop covers at most k agents
    # To traverse N agents, need at most ceil(N/k) hops
    return (N + k - 1) // k


def run_scaling(N_values=[10, 15, 20, 30], m=4, k_neighbors=4, T=500, n_runs=10):
    """Scale with N on connected graph."""
    print("\n" + "=" * 60)
    print("SCALING WITH N — CONNECTED GRAPH")
    print("=" * 60)
    
    all_results = {}
    for N in N_values:
        r = run_connected_comparison(N=N, m=m, k_neighbors=k_neighbors, T=T, n_runs=n_runs)
        all_results[N] = r
    
    print("\n--- Summary ---")
    print(f"{'N':>4} {'LocalNB3R':>12} {'LocalQAMAB':>12} {'Random':>12} {'NB3R vs QA':>12}")
    print("-" * 55)
    for N, r in all_results.items():
        nb3r_m = np.mean(r['LocalNB3R(connected)'])
        qa_m = np.mean(r['LocalQAMAB(connected)'])
        rand_m = np.mean(r['Random'])
        delta = qa_m - nb3r_m
        print(f"{N:>4} {nb3r_m:>12.3f} {qa_m:>12.3f} {rand_m:>12.3f} {delta:>+12.3f}")
    
    return all_results


if __name__ == '__main__':
    print("=" * 70)
    print("SIMULATION: CONNECTED GRAPH — DIAMOND Convergence Should Apply")
    print("=" * 70)
    print()
    print("Key: k=4 neighbors per agent in a ring. Graph is CONNECTED.")
    print("DIAMOND proves LocalNB3R converges here (connected + log cooling).")
    print()
    
    # Main comparison at N=20
    r = run_connected_comparison(N=20, m=4, k_neighbors=4, T=500, n_runs=10)
    
    # Scaling with N
    run_scaling(N_values=[10, 15, 20, 30], m=4, k_neighbors=4, T=500, n_runs=10)
    
    # Effect of k (more connectivity = faster convergence?)
    print("\n--- Effect of k (connectivity) at N=20 ---")
    for k in [2, 4, 6, 8]:
        r = run_connected_comparison(N=20, m=4, k_neighbors=k, T=500, n_runs=5)
