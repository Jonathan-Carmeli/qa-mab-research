#!/bin/bash
set -e
cd /Users/jon_claw/qa-mab-research/simulations

echo "============================================="
echo "VALIDATION SUITE START"
echo "============================================="

echo ""
echo "=== [1/3] validate_cat2_qubo_optimality.py ==="
python3 validate_cat2_qubo_optimality.py && echo "CAT2: PASS" || echo "CAT2: FAIL"
echo ""

echo ""
echo "=== [2/3] validate_cat3_sa_solver_accuracy.py ==="
python3 validate_cat3_sa_solver_accuracy.py && echo "CAT3: PASS" || echo "CAT3: FAIL"
echo ""

echo ""
echo "=== [3/3] validate_cat4_learning_dynamics.py ==="
python3 validate_cat4_learning_dynamics.py && echo "CAT4: PASS" || echo "CAT4: FAIL"
echo ""

echo "============================================="
echo "VALIDATION SUITE DONE"
echo "============================================="