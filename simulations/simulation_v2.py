"""
simulation_v2.py
Sparse network simulation — testing local communication vs global QUBO.

Key insight (Yonatan's reformulation):
- Real networks: communication is LOCAL (neighbors only), interference is DENSE (everyone affects everyone)
- QA-MAB naturally fits this: sparse QUBO + far-interference absorbed into u_hat
- NB3R in original form has GLOBAL communication — unfair comparison

This module tests:
1. ClusterNetworkEnvironment — interference HIGH within cluster, LOW across clusters
2. SparseNB3R — NB3R with local communication only (neighbors in same cluster)
3. SparseQAMAB — QA-MAB with sparse QUBO (only cluster-level interference)
4. Compare: NB3R(global) vs SparseNB3R(local) vs SparseQAMAB(sparse) vs Random
"""

import numpy as np
from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


class ClusterNetworkEnvironment:
    """Network with spatial structure: agents in clusters, interference decays with distance."""
    
    def __init__(self, N, m, n_clusters=4, seed=42, 
                 B_range=(0.5, 1.0),
                 I_near_range=(0.15, 0.25),
                 I_far_range=(0.01, 0.05)):
        """
        Args:
            N: number of agents
            m: number of routes
            n_clusters: number of clusters (N should be divisible by this)
            B_range: (min, max) for base utilities
            I_near_range: interference range for SAME-CLUSTER agents
            I_far_range: interference range for DIFFERENT-CLUSTER agents
        """
        self.N = N
        self.m = m
        self.n_clusters = n_clusters
        self.cluster_size = N // n_clusters
        rng = np.random.default_rng(seed)
        
        # Assign cluster membership
        self.cluster_of = np.zeros(N, dtype=int)
        for i in range(N):
            self.cluster_of[i] = i // self.cluster_size
        
        # Base utilities
        self.B = rng.uniform(B_range[0], B_range[1], size=(N, m))
        
        # Interference matrix: dense but distance-dependent
        self.I = np.zeros((N, m, N, m))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                ic = self.cluster_of[i]
                jc = self.cluster_of[j]
                if ic == jc:
                    # Near interference (same cluster)
                    I_vals = rng.uniform(I_near_range[0], I_near_range[1], size=(m, m))
                else:
                    # Far interference (different cluster)
                    I_vals = rng.uniform(I_far_range[0], I_far_range[1], size=(m, m))
                self.I[i, :, j, :] = I_vals
    
    def compute_throughput(self, assignment):
        """Full ground-truth throughput computation."""
        x = np.zeros((self.N, self.m))
        for i, k in assignment.items():
            x[i, k] = 1.0
        
        throughputs = np.zeros(self.N)
        for i in range(self.N):
            k = assignment[i]
            interference = 0.0
            for j in range(self.N):
                for l in range(self.m):
                    interference += self.I[i, k, j, l] * x[j, l]
            throughputs[i] = self.B[i, k] - interference
        return throughputs
    
    def get_same_cluster_agents(self, i):
        """Return agents in the same cluster as agent i (including i)."""
        ci = self.cluster_of[i]
        return [j for j in range(self.N) if self.cluster_of[j] == ci]
    
    def get_neighbors(self, i):
        """Return agents in the same cluster as agent i (excluding i)."""
        return [j for j in range(self.N) if j != i and self.cluster_of[j] == self.cluster_of[i]]


class SparseNB3R(NB3R):
    """NB3R with LOCAL communication — only talks to same-cluster agents."""
    
    def __init__(self, env, *args, **kwargs):
        # Don't call super().__init__ yet — we need to set up cluster-aware W first
        self.env = env
        self.N = env.N
        self.m = env.m
        self.cluster_of = env.cluster_of
        
        # Copy all NB3R attributes from original init pattern
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
        """One step: pick routes (softmax), execute, update W with LOCAL signal only."""
        # 1. Pick routes
        assignment = {}
        for i in range(self.N):
            probabilities = np.exp(self.W[i] / self.tau)
            probabilities /= probabilities.sum()
            k = np.random.default_rng(int(self.seed + i + self.tau * 1000)).choice(self.m, p=probabilities)
            assignment[i] = k
        
        # 2. Execute
        throughputs = self.env.compute_throughput(assignment)
        
        # 3. Update W — LOCAL communication (same cluster only)
        for i in range(self.N):
            neighbors = self.env.get_neighbors(i)
            # Signal = own throughput + neighbors' throughputs (LOCAL)
            local_signal = throughputs[i] + sum(throughputs[j] for j in neighbors)
            self.W[i] = (1 - self.alpha) * self.W[i] + self.alpha * (local_signal / (1 + len(neighbors)))
        
        self.prev_assignment = assignment
        self.prev_throughputs = throughputs
        return assignment, throughputs
    
    def run(self, T):
        history = []
        for _ in range(T):
            _, throughputs = self.step()
            history.append(throughputs.sum())
        return np.array(history)


class SparseQAMAB(QAMAB):
    """QA-MAB with SPARSE QUBO — only models cluster-level interference."""
    
    def __init__(self, env, *args, **kwargs):
        # Manually set up like QAMAB but with sparse I_hat
        self.env = env
        self.N = env.N
        self.m = env.m
        self.cluster_of = env.cluster_of
        
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
        
        # SPARSE: only allocate I_hat for same-cluster agents
        self.cluster_size = env.cluster_size
        self.n_clusters = env.n_clusters
        
        # For each cluster, allocate small I_hat matrix
        # Shape: (cluster_size, m, cluster_size, m) per cluster
        self.I_hat_clusters = {}
        for c in range(self.n_clusters):
            csz = self.cluster_size
            self.I_hat_clusters[c] = np.zeros((csz, self.m, csz, self.m))
        
        self.tau = self.tau0
        self._prev_x = None
        self._prev_throughputs = None
        
        # SA parameters (fast for cluster env)
        self.sa_steps = 5
        self.sa_inner_steps = 10
        
        # Map agent index to (cluster_idx, within-cluster_idx)
        self.agent_to_cluster = {}
        self.agent_within_cluster = {}
        for i in range(self.N):
            c = self.cluster_of[i]
            idx = [j for j in range(self.N) if self.cluster_of[j] == c].index(i)
            self.agent_to_cluster[i] = c
            self.agent_within_cluster[i] = idx
    
    def _get_I_hat(self, i, k, j, l):
        """Get I_hat for (i,k) interacting with (j,l). Returns 0 if different clusters."""
        ci, wi = self.agent_to_cluster[i], self.agent_within_cluster[i]
        cj, wj = self.agent_to_cluster[j], self.agent_within_cluster[j]
        if ci != cj:
            return 0.0  # SPARSE: no cross-cluster interference in QUBO
        return self.I_hat_clusters[ci][wi, k, wj, l]
    
    def build_qubo(self, assignment):
        """Build QUBO matrix — SPARSE (only same-cluster interactions)."""
        Q = np.zeros((self.N * self.m, self.N * self.m))
        
        for i in range(self.N):
            k = assignment[i]
            ci = self.cluster_of[i]
            
            # Diagonal: -u_hat (want to maximize u_hat)
            Q[i * self.m + k, i * self.m + k] -= self.u_hat[i, k]
            
            # Off-diagonal: +I_hat*x (penalize same-route collisions)
            for j in range(self.N):
                if i == j:
                    continue
                cj = self.cluster_of[j]
                if ci != cj:
                    continue  # SPARSE: skip cross-cluster
                
                l = assignment[j]
                I_val = self._get_I_hat(i, k, j, l)
                Q[i * self.m + k, j * self.m + l] += I_val
        
        return Q
    
    def step(self):
        """One step with sparse QUBO."""
        # 1. SA to find best assignment given current u_hat, I_hat
        best_assignment, _ = self._sa_optimize()
        
        # 2. Execute
        throughputs = self.env.compute_throughput(best_assignment)
        
        # 3. Update u_hat (as in original)
        for i in range(self.N):
            k = best_assignment[i]
            self.u_hat[i, k] += self.B_learn_rate * (throughputs[i] - self.u_hat[i, k])
        
        # 4. Update I_hat (same-cluster only)
        if self._prev_x is not None and self._prev_throughputs is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if self.cluster_of[i] != self.cluster_of[j]:
                        continue  # SPARSE: skip cross-cluster
                    
                    ki = self._prev_x[i]
                    kj = self._prev_x[j]
                    
                    expected_i = self.u_hat[i, ki]
                    expected_j = self.u_hat[j, kj]
                    
                    drop_i = max(0, expected_i - self._prev_throughputs[i])
                    drop_j = max(0, expected_j - self._prev_throughputs[j])
                    
                    delta = (drop_i + drop_j) / 2
                    
                    ci, wi = self.agent_to_cluster[i], self.agent_within_cluster[i]
                    cj, wj = self.agent_within_cluster[j], self.agent_within_cluster[j]
                    
                    if ci == cj:
                        I_old = self.I_hat_clusters[ci][wi, ki, wj, kj]
                        self.I_hat_clusters[ci][wi, ki, wj, kj] = min(self.I_cap, 
                            (1 - self.I_learn_rate) * I_old + self.I_learn_rate * delta)
                        self.I_hat_clusters[ci][wj, kj, wi, ki] = self.I_hat_clusters[ci][wi, ki, wj, kj]
        
        # 5. Optional decay
        if self.I_decay_rate > 0:
            for c in range(self.n_clusters):
                self.I_hat_clusters[c] *= (1 - self.I_decay_rate)
        
        # 6. Increase tau
        self.tau += self.delta_tau
        
        self._prev_x = best_assignment
        self._prev_throughputs = throughputs
        return best_assignment, throughputs
    
    def _sa_optimize(self):
        """SA optimization using sparse QUBO."""
        best_sw = float('-inf')
        best_assignment = None
        
        for _ in range(self.sa_steps):
            # Random restart
            assignment = {i: np.random.randint(0, self.m) for i in range(self.N)}
            current_sw = sum(self.env.compute_throughput(assignment))
            
            for _ in range(self.sa_inner_steps):
                # Perturb
                new_assignment = assignment.copy()
                i = np.random.randint(0, self.N)
                k = np.random.randint(0, self.m)
                new_assignment[i] = k
                
                new_sw = sum(self.env.compute_throughput(new_assignment))
                
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
            history.append(throughputs.sum())
        return np.array(history)


def run_comparison(N=20, m=4, n_clusters=4, T=500, n_runs=10):
    """Compare all variants on cluster network."""
    seeds = list(range(42, 42 + n_runs))
    results = {'NB3R': [], 'SparseNB3R': [], 'SparseQAMAB': [], 'Random': []}
    
    for seed in seeds:
        # Full environment (for NB3R comparison — same ground truth)
        env_full = NetworkEnvironment(N, m, seed=seed)
        
        # Cluster environment (for sparse variants)
        env_cluster = ClusterNetworkEnvironment(N, m, n_clusters=n_clusters, seed=seed)
        
        # NB3R on full environment
        nb3r = NB3R(env_full, seed=seed)
        hist = nb3r.run(T)
        results['NB3R'].append(np.mean(hist[-50:]))
        
        # SparseNB3R on cluster environment (local communication)
        sparse_nb3r = SparseNB3R(env_cluster, seed=seed)
        hist = sparse_nb3r.run(T)
        results['SparseNB3R'].append(np.mean(hist[-50:]))
        
        # SparseQAMAB on cluster environment (sparse QUBO)
        sparse_qa = SparseQAMAB(env_cluster, seed=seed)
        hist = sparse_qa.run(T)
        results['SparseQAMAB'].append(np.mean(hist[-50:]))
        
        # Random baseline
        rng = np.random.default_rng(seed)
        sw_sum = 0
        for _ in range(T):
            assignment = {i: rng.integers(0, m) for i in range(N)}
            sw_sum += env_cluster.compute_throughput(assignment).sum()
        results['Random'].append(sw_sum / T)
    
    print(f"\nCluster network: N={N}, m={m}, n_clusters={n_clusters}, T={T}, runs={n_runs}")
    print(f"{'Algorithm':<15} {'Mean SW':>12} {'Std':>10}")
    print("-" * 40)
    for alg, vals in results.items():
        print(f"{alg:<15} {np.mean(vals):>12.3f} {np.std(vals):>10.3f}")
    
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("SIMULATION V2: SPARSE NETWORKS")
    print("Key question: Does local-communication NB3R fare worse")
    print("than sparse QUBO QA-MAB when far-interference exists?")
    print("=" * 60)
    
    # Main comparison
    r = run_comparison(N=20, m=4, n_clusters=4, T=500, n_runs=10)
    
    # Also test with more clusters (more sparse = harder)
    print("\n--- More sparse (n_clusters=8, 2-3 agents each) ---")
    r2 = run_comparison(N=24, m=4, n_clusters=8, T=500, n_runs=10)
    
    print("\n--- Very local (n_clusters=10, 2 agents each) ---")
    r3 = run_comparison(N=20, m=4, n_clusters=10, T=500, n_runs=10)
