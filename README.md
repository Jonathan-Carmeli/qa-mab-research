# QA-MAB: Quantum Annealing Multi-Armed Bandit for Network Routing

**Comparing centralized QUBO-based optimization vs distributed multi-armed bandit learning for multi-agent routing with interference.**

Extension of the DIAMOND paper (arXiv:2303.15544).

## The Problem

N agents in a network, each choosing 1 of m routes. Agents interfere with each other — choosing the same route degrades throughput. Goal: maximize Social Welfare (total throughput).

## Two Algorithms

### NB3R (Distributed)
Each agent independently learns which route is best using softmax + exponential moving average. Agents broadcast throughput to neighbors. Simple, fast, no central coordinator.

### QA-MAB (Centralized)
A central server builds a QUBO (Quadratic Unconstrained Binary Optimization) from estimated utilities and interference, then solves it via Simulated Annealing (proxy for real Quantum Annealing). The QUBO encodes the global optimization problem.

## Key Results

### Crossover at N ≈ 12
- **N ≤ 10:** NB3R wins — direct feedback + coordinate ascent is sufficient
- **N ≥ 12:** QA-MAB wins — global optimization beats distributed learning
- **N = 50:** QA-MAB advantage = +15 over NB3R

### Why NB3R Fails at Large N
NB3R updates weights toward total Social Welfare — the **same scalar** for all agents. At large N, this signal becomes noise (SW ≈ -200 ± 3, no per-agent discrimination).

### Why QA-MAB Succeeds
QA-MAB tracks per-agent utility estimates (u_hat) and pairwise interference (I_hat). The QUBO optimizer sees the global structure, not just a single scalar.

### The Quantum Angle
SA is a weak proxy for real QA. With D-Wave Advantage2 (4,400 qubits):
- Quantum tunneling escapes local minima SA gets trapped in
- 20μs anneal time vs 15ms SA → ×750 speedup
- N=17 (m=4) fully connected, N=200+ sparse interference

## Repository Structure

```
├── README.md
├── docs/
│   ├── algorithm-analysis.md      # Deep dive into both algorithms
│   ├── experiment-results.md      # All experiment results
│   ├── hardware-analysis.md       # D-Wave capability analysis
│   └── fix-experiments.md         # Systematic fix testing (16 variants)
├── simulations/
│   ├── simulation_core.py         # Network environment
│   ├── nb3r.py                    # NB3R algorithm
│   ├── qa_mab.py                  # QA-MAB algorithm
│   ├── qaoa_solver.py             # QAOA solver (Qiskit)
│   ├── convergence_simulation.py  # Convergence experiments
│   ├── scaling_simulation.py      # N-scaling experiments
│   ├── fix_experiments.py         # Round 1 fix tests
│   ├── fix_experiments_v2.py      # Round 2 fix tests
│   ├── fix_experiments_v3.py      # Round 3 scaling breakthrough
│   ├── fix_experiments_v4.py      # Round 4 crossover + deep analysis
│   └── fix_experiments_v5.py      # Publication-ready experiments
└── results/
    └── (generated plots and data)
```

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib
pip install qiskit qiskit-optimization qiskit-algorithms  # optional, for QAOA

# Run convergence experiment
python simulations/convergence_simulation.py 10 4 500 20

# Run scaling comparison
python simulations/fix_experiments_v3.py

# Run publication experiments
python simulations/fix_experiments_v5.py
```

## Citation

This work extends:
> DIAMOND: Dual-stage Interference-Aware Multi-flow Optimization of Network Data-streams (arXiv:2303.15544)

## License

MIT

## Author

Yehonatan Carmeli — BGU, Beer Sheva, Israel
MSc Thesis: Finding ML algorithms that quantum computing can execute faster
