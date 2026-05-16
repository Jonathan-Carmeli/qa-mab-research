"""
qaoa_solver.py
QAOA (Quantum Approximate Optimization Algorithm) solver for the QA-MAB QUBO.

Uses Qiskit to simulate quantum circuits on a classical computer.
Limitation: exponential in qubits, so only feasible for small N (N*m <= ~20 qubits).

For N=5, m=4: 20 qubits — at the edge of classical simulation.
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler


def qubo_matrix_to_quadratic_program(Q, N, m):
    """
    Convert our QUBO matrix Q (size N*m x N*m) to a Qiskit QuadraticProgram.
    
    QUBO: minimize x^T Q x  (binary x)
    """
    size = N * m
    qp = QuadraticProgram('qa_mab')
    
    # Add binary variables x_{i,k} for each agent i, route k
    for i in range(N):
        for k in range(m):
            qp.binary_var(name=f'x_{i}_{k}')
    
    # Build objective: minimize x^T Q x
    # Diagonal terms: Q[idx,idx] * x_idx (since x^2 = x for binary)
    # Off-diagonal: Q[idx1,idx2] * x_idx1 * x_idx2
    linear = {}
    quadratic = {}
    
    for idx1 in range(size):
        # Diagonal
        if abs(Q[idx1, idx1]) > 1e-12:
            i1, k1 = divmod(idx1, m)
            linear[f'x_{i1}_{k1}'] = Q[idx1, idx1]
        
        # Upper triangle off-diagonal
        for idx2 in range(idx1 + 1, size):
            val = Q[idx1, idx2] + Q[idx2, idx1]  # symmetric contribution
            if abs(val) > 1e-12:
                i1, k1 = divmod(idx1, m)
                i2, k2 = divmod(idx2, m)
                quadratic[(f'x_{i1}_{k1}', f'x_{i2}_{k2}')] = val
    
    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def solve_qubo_qaoa(Q, N, m, reps=2, maxiter=200, seed=42):
    """
    Solve QUBO using QAOA (simulated quantum circuit).
    
    Args:
        Q: QUBO matrix (N*m x N*m)
        N: number of agents
        m: number of routes
        reps: QAOA circuit depth (more = better but slower)
        maxiter: classical optimizer iterations
        seed: random seed
    
    Returns:
        assignment: dict {agent_i: route_k}
        energy: QUBO energy of the solution
    """
    qp = qubo_matrix_to_quadratic_program(Q, N, m)
    
    sampler = StatevectorSampler(seed=seed)
    optimizer = COBYLA(maxiter=maxiter)
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
    )
    
    algo = MinimumEigenOptimizer(qaoa)
    result = algo.solve(qp)
    
    # Decode result to assignment
    assignment = {}
    for i in range(N):
        best_k = 0
        best_val = -1
        for k in range(m):
            var_name = f'x_{i}_{k}'
            val = result.variables_dict.get(var_name, 0)
            if val > best_val:
                best_val = val
                best_k = k
        assignment[i] = best_k
    
    return assignment, result.fval


def solve_qubo_bruteforce(Q, N, m):
    """
    Brute force: try all m^N feasible assignments (one route per agent).
    Only feasible for small N*m.
    
    Returns:
        assignment: dict {agent_i: route_k}
        energy: minimum QUBO energy
    """
    import itertools
    
    best_energy = float('inf')
    best_assignment = None
    size = N * m
    
    # Iterate over all feasible assignments (m^N)
    for routes in itertools.product(range(m), repeat=N):
        x = np.zeros(size)
        for i, k in enumerate(routes):
            x[i * m + k] = 1.0
        
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_assignment = {i: k for i, k in enumerate(routes)}
    
    return best_assignment, best_energy


if __name__ == '__main__':
    # Quick test: N=3, m=2 (6 qubits, 8 feasible assignments)
    import sys
    sys.path.insert(0, '.')
    from simulation_core import NetworkEnvironment
    from qa_mab import QAMAB
    
    N, m = 3, 2
    env = NetworkEnvironment(N, m, seed=42)
    qa = QAMAB(env, tau0=1.0, delta_tau=0.05, lambda_=0.5, seed=42)
    
    # Run a few steps to build up estimates
    for _ in range(50):
        qa.step()
    
    Q = qa.build_qubo()
    
    # Brute force
    bf_assign, bf_energy = solve_qubo_bruteforce(Q, N, m)
    bf_sw = env.social_welfare(bf_assign)
    
    # QAOA
    qaoa_assign, qaoa_energy = solve_qubo_qaoa(Q, N, m, reps=2, maxiter=200)
    qaoa_sw = env.social_welfare(qaoa_assign)
    
    # SA (from qa_mab)
    sa_assign = qa.solve_qubo(Q)
    sa_energy = float(np.array([1 if sa_assign.get(i//m) == i%m else 0 for i in range(N*m)], dtype=float) @ Q @ np.array([1 if sa_assign.get(i//m) == i%m else 0 for i in range(N*m)], dtype=float))
    sa_sw = env.social_welfare(sa_assign)
    
    print(f"N={N}, m={m} ({N*m} qubits)")
    print(f"Brute force: assignment={bf_assign}, SW={bf_sw:.4f}, energy={bf_energy:.4f}")
    print(f"QAOA (p=2):  assignment={qaoa_assign}, SW={qaoa_sw:.4f}, energy={qaoa_energy:.4f}")
    print(f"SA:          assignment={sa_assign}, SW={sa_sw:.4f}, energy={sa_energy:.4f}")
