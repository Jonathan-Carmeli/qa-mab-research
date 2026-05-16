# D-Wave Quantum Computer Setup — QA-MAB Infrastructure

**Goal:** When Jonathan receives D-Wave access token, this infrastructure will be ready to run QA-MAB on real quantum hardware.

## Overview

The current implementation uses **Simulated Annealing (SA)** as a proxy for quantum annealing. Real quantum hardware (D-Wave) will provide:

1. **Quantum Tunneling** — escape local minima SA gets trapped in
2. **Speed** — 20μs anneal vs 15ms SA → ×750 speedup
3. **Scale** — D-Wave Advantage2 (4,400 qubits) enables N=17 fully connected

## Architecture

```
┌─────────────────────────────────────────────────┐
│           QA-MAB Framework (this repo)           │
├──────────────┬──────────────────┬───────────────┤
│ simulation_  │   qa_mab.py      │  dwave_setup  │
│ core.py      │  (QUBO builder)  │  .py          │
├──────────────┴────────┬─────────┴───────────────┤
│           QUBO Matrix (N·m × N·m)               │
├──────────────────────┼──────────────────────────┤
│   SA (current)       │   D-Wave (when token     │
│   simulation_core.py │   arrives)               │
└──────────────────────┴──────────────────────────┘
```

## Current State: SA Proxy

```python
# qa_mab.py — _sa_optimize() method
def _sa_optimize(self):
    # SA with 20 restarts × 15 iterations
    # τ increases per step → QUBO gets "harder" → SA harder to solve
    # At τ=25 (T=500), SA gets stuck in local minima
    # Real QA would tunnel through these barriers
```

## Step 1: Install D-Wave SDK (when token arrives)

```bash
# Install dwave-ocean-sdk
pip install dwave-ocean-sdk

# Configure with your API token
dwave config

# Test connection
python -c "import dwave_sapi; print('D-Wave SDK installed')"
```

## Step 2: Replace SA with D-Wave Solver

```python
# dwave_qa_solver.py — replaces _sa_optimize()
import dwave_sapi.cloud as sapiclient
from dwave_sapi.embedding import embed_qubo, compose_ising

def solve_on_dwave(Q, num_reads=1000):
    """Solve QUBO on D-Wave Advantage2.
    
    Args:
        Q: numpy array (N*m × N*m) — the QUBO matrix
        num_reads: number of anneal samples (default 1000)
    
    Returns:
        best_sample: dict {variable_index: 0/1}
        energy: float
    """
    # Connect to D-Wave
    connection = sapiclient.get_connection()
    solver = connection.get_solver('Advantage2_prototype1')
    
    # Embed QUBO onto QPU topology
    # (Advantage2 has specific qubit connectivity)
    embedding = solver.embedding
    
    # Submit
    response = solver.sample_qubo(Q, num_reads=num_reads)
    
    # Get best result
    best = response.samples[0]
    return best.sample, best.energy
```

## Step 3: Modify qa_mab.py to Support Both Solvers

```python
# In QAMAB class — add solver selection
class QAMAB:
    def __init__(self, env, solver='sa', dwave_token=None, **kwargs):
        self.solver = solver
        
        if solver == 'dwave':
            # Initialize D-Wave connection
            from dwave_sapi.cloud import get_connection
            self.conn = get_connection()
            self.dwave_solver = self.conn.get_solver('Advantage2_prototype1')
    
    def _solve_qubo(self, Q):
        if self.solver == 'sa':
            return self._sa_optimize(Q)
        elif self.solver == 'dwave':
            return self._solve_on_dwave(Q)
    
    def _solve_on_dwave(self, Q):
        response = self.dwave_solver.sample_qubo(Q, num_reads=1000)
        best = min(response.samples, key=lambda s: s.energy)
        # Convert to assignment dict
        assignment = {i: int(best.sample[i * self.m + k]) 
                      for i in range(self.N) 
                      for k in range(self.m) if best.sample[i * self.m + k]}
        return assignment, best.energy
```

## Step 4: Hybrid Approach (recommended)

D-Wave is fast but limited. For N > qubit count, use **hybrid**:

```python
def solve_hybrid(Q, N, m, dwave_solver):
    """Use D-Wave for small sub-problems, SA for the rest.
    
    Strategy:
    - If N*m <= 170 (D-Wave can handle directly): use D-Wave
    - Else: decompose into sub-QUBOs of ~170 variables each
    """
    n_vars = N * m
    QUBO_LIMIT = 170  # Advantage2 has ~4,400 working qubits
    
    if n_vars <= QUBO_LIMIT:
        return dwave_solver.sample_qubo(Q, num_reads=1000)
    
    # Decompose: group variables by route
    # Solve per-agent sub-problems, then coordinate globally
    ...
```

## Step 5: Comparison Protocol

When D-Wave token arrives, run:

```bash
# Compare SA vs D-Wave on same QUBO problems
python compare_solvers.py --N 10 --runs 20 --solvers sa,dwave
```

Expected: D-Wave finds lower energy (better solutions) than SA on the same QUBO.

## D-Wave Solver Configuration

```
Solver Name: Advantage2_prototype1 (or actual name given by D-Wave)
Access Type: QPU (Quantum Processing Unit)
Qubits: ~4,400
Couplings: ~60,000
Connectivity: Pegasus topology
Anneal Time: 20μs per sample
```

## Expected Improvements (from theory)

| Problem Size | SA Quality | D-Wave Quality | Reason |
|---|---|---|---|
| N=5 (20 qubits) | ~80% of optimal | ~95%+ | tunneling |
| N=10 (40 qubits) | ~60% of optimal | ~85%+ | larger problem = more local minima |
| N=17 (68 qubits) | ~40% of optimal | ~80%+ | SA struggles with scale |

## Token Configuration

When Jonathan receives the D-Wave token:
1. Add to `~/.openclaw/workspace/secrets/dwave_token.txt`
2. Update `dwave_setup.py` to read from there
3. Run `python dwave_setup.py test` to verify

## File Location
`dwave_setup.py` — will be created when token arrives

## Tags
#thesis #quantum #dwave #infrastructure #setup