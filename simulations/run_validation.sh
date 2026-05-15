#!/bin/bash
cd /Users/jon_claw/qa-mab-research/simulations
echo "=== Running validate_cat2_qubo_optimality.py ==="
python3 validate_cat2_qubo_optimality.py 2>&1
echo "EXIT: $?"