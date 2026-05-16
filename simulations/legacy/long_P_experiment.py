"""Run long-P experiment: P=30 epochs, T=100 steps, seeds=10 for QA-MAB only."""
import sys
import os

# Add thesis brain simulation to path
sys.path.insert(0, "/Users/jon_claw/Thesis_brain/simulation")

import numpy as np
from src.uav_routing.config import (
    ExperimentConfig, WorldConfig, GroundTruthConfig,
    TimeConfig, QAMABConfig, NB3RConfig
)
from src.uav_routing.runner import run_single, _make_qamab_factory
from src.uav_routing.ground_truth import sample_ground_truth

P = 30
T = 100
N_SEEDS = 10
BASE_SEED = 42

cfg = ExperimentConfig(
    world=WorldConfig(),
    time=TimeConfig(P_epochs=P, T_steps=T),
    n_seeds=N_SEEDS,
    base_seed=BASE_SEED,
)

print(f"Running long-P experiment: P={P}, T={T}, seeds={N_SEEDS}")

results = []

for seed_idx in range(N_SEEDS):
    seed = BASE_SEED + seed_idx
    print(f"  seed {seed_idx+1}/{N_SEEDS} (seed={seed})", flush=True)

    rng_gt = np.random.default_rng(seed)
    theta_star, phi_star = sample_ground_truth(rng_gt, cfg.world, cfg.gt)

    factory = _make_qamab_factory(cfg)
    result = run_single(cfg, factory, seed, theta_star=theta_star, phi_star=phi_star)
    results.append(result)

    # Also run Oracle for comparison
    from src.uav_routing.runner import _make_oracle_factory
    oracle_factory = _make_oracle_factory(cfg)
    oracle_result = run_single(cfg, oracle_factory, seed, theta_star=theta_star, phi_star=phi_star)
    results.append(oracle_result)

# Stack results
qamab_results = results[::2]  # even indices
oracle_results = results[1::2]  # odd indices

qamab_losses    = np.stack([r.losses_log    for r in qamab_results])
qamab_coll      = np.stack([r.coll_log       for r in qamab_results])
qamab_theta_err = np.stack([r.theta_err_log for r in qamab_results])
qamab_phi_err   = np.stack([r.phi_err_log   for r in qamab_results])

oracle_losses    = np.stack([r.losses_log    for r in oracle_results])
oracle_coll      = np.stack([r.coll_log       for r in oracle_results])
oracle_theta_err = np.stack([r.theta_err_log for r in oracle_results])
oracle_phi_err   = np.stack([r.phi_err_log   for r in oracle_results])

# Save
out_path = "/Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/long_P_results.npz"
np.savez(
    out_path,
    qamab_theta_err=qamab_theta_err,
    qamab_phi_err=qamab_phi_err,
    qamab_losses=qamab_losses,
    qamab_coll=qamab_coll,
    oracle_theta_err=oracle_theta_err,
    oracle_phi_err=oracle_phi_err,
    oracle_losses=oracle_losses,
    oracle_coll=oracle_coll,
    seeds=np.array([BASE_SEED + i for i in range(N_SEEDS)]),
)
print(f"Saved {out_path}")

# Print summary
print("\n=== QA-MAB theta_err per epoch ===")
epochs = np.arange(1, P+1)
theta_mean = qamab_theta_err.mean(axis=0)
theta_std  = qamab_theta_err.std(axis=0)
phi_mean = qamab_phi_err.mean(axis=0)
phi_std  = qamab_phi_err.std(axis=0)

print(f"{'Epoch':>6}  {'theta_err':>12}  {'±std':>10}  {'phi_err':>12}  {'±std':>10}")
for e in range(P):
    print(f"{e+1:>6}  {theta_mean[e]:>12.5f}  {theta_std[e]:>10.5f}  {phi_mean[e]:>12.5f}  {phi_std[e]:>10.5f}")

print("\n=== Oracle theta_err per epoch ===")
oracle_theta_mean = oracle_theta_err.mean(axis=0)
oracle_theta_std  = oracle_theta_err.std(axis=0)
oracle_phi_mean = oracle_phi_err.mean(axis=0)
oracle_phi_std  = oracle_phi_err.std(axis=0)

print(f"{'Epoch':>6}  {'theta_err':>12}  {'±std':>10}  {'phi_err':>12}  {'±std':>10}")
for e in range(P):
    print(f"{e+1:>6}  {oracle_theta_mean[e]:>12.5f}  {oracle_theta_std[e]:>10.5f}  {oracle_phi_mean[e]:>12.5f}  {oracle_phi_std[e]:>10.5f}")