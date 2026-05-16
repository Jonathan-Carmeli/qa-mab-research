#!/bin/bash
export PATH="/Users/jon_claw/homebrew/bin:$PATH"
cd /Users/jon_claw/Thesis_brain/simulation

echo "Starting UAV dynamic environment experiment..."
echo "Config: m=30, Z=9, N=3, K=4, P=10, T=100, seeds=20"
echo "Agents: Random, Greedy, NB3R, QA-MAB, Oracle"
echo "Goal: Track convergence of theta_hat/phi_hat and SW over epochs"
echo ""

python3 -m src.uav_routing.run_experiment --epochs 10 --steps 100 --seeds 20 2>&1 | tee /Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/run_log.txt

echo ""
echo "Done. Checking results..."
ls -la results/uav_routing/ 2>/dev/null | head -20