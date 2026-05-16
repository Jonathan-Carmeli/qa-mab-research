#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== Cat 1: Parameter Sweep ===";     python3 validate_cat1_param_sweep_physical.py
echo "=== Cat 2: QUBO Optimality ===";      python3 validate_cat2_physical.py
echo "=== Cat 3: SA Solver Accuracy ===";   python3 validate_cat3_physical.py
echo "=== Cat 4: Learning Dynamics ===";    python3 validate_cat4_physical.py
echo "=== Cat 5: Regret Convergence ===";   python3 validate_cat5_regret_convergence_physical.py
echo "=== Cat 6: Noise Robustness ===";     python3 validate_cat6_noise_robustness_physical.py
echo "=== Baseline Comparison ===";         python3 validate_baseline_comparison.py
echo ""; echo "All done. Results in simulations/results/"