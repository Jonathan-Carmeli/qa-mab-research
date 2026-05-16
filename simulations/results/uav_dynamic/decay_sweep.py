"""
Decay sweep: find epoch_decay that enables true learning across P=20.

Vary: epoch_decay in {1.0, 0.99, 0.95, 0.9, 0.7}
Question: which decay lets theta_hat actually converge as P grows?
"""
import sys, os, numpy as np, time
sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')

from src.uav_routing.config import (
    ExperimentConfig, WorldConfig, TimeConfig, GroundTruthConfig, QAMABConfig, NB3RConfig
)
from src.uav_routing.runner import run_single
from src.uav_routing.agents.qamab_agent import QAMABAgent
from src.uav_routing.ground_truth import sample_ground_truth

P, T, SEEDS = 20, 100, 10
DECAYS = [1.0, 0.99, 0.95, 0.9, 0.7]
BASE_SEED = 42

results = {}

for decay in DECAYS:
    print(f'\n=== epoch_decay={decay} ===', flush=True)
    qamab_cfg = QAMABConfig(
        alpha=0.15, gamma_0=2.0, a=0.5, b=0.3,
        ucb_c=3.0, epoch_decay=decay
    )
    world_cfg = WorldConfig(m=30, Z=9, N_flows=3, K_paths=4, comm_radius=350.0)
    gt_cfg = GroundTruthConfig(
        n_faulty_uavs=4, theta_low=0.2, theta_high=0.4,
        n_faulty_zones=2, phi_low=0.2, phi_high=0.4,
        C_coll=5.0, d0=150.0, sigma_noise=0.05
    )
    
    theta_errs = []
    phi_errs = []
    epoch_losses = []
    
    for seed_idx in range(SEEDS):
        seed = BASE_SEED + seed_idx
        rng = np.random.default_rng(seed)
        rng_gt = np.random.default_rng(seed)
        theta_star, phi_star = sample_ground_truth(rng_gt, world_cfg, gt_cfg)
        
        def factory(rng, ts, ps):
            return QAMABAgent(rng, world_cfg, qamab_cfg, gt_cfg, ucb_c=3.0)
        
        cfg = ExperimentConfig(
            world=world_cfg, time=TimeConfig(P_epochs=P, T_steps=T),
            gt=gt_cfg, qamab=qamab_cfg, nb3r=NB3RConfig(),
            n_seeds=1, base_seed=seed
        )
        
        result = run_single(cfg, factory, seed, theta_star=theta_star, phi_star=phi_star)
        theta_errs.append(result.theta_err_log)
        phi_errs.append(result.phi_err_log)
        epoch_losses.append(result.losses_log.mean(axis=(1, 2)))
    
    theta_errs = np.array(theta_errs)
    phi_errs = np.array(phi_errs)
    epoch_losses = np.array(epoch_losses)
    
    te_mean = theta_errs.mean(axis=0)
    pe_mean = phi_errs.mean(axis=0)
    el_mean = epoch_losses.mean(axis=0)
    
    print(f'  theta_err: E1={te_mean[0]:.3f} E10={te_mean[9]:.3f} E20={te_mean[19]:.3f}')
    print(f'  phi_err:   E1={pe_mean[0]:.3f} E10={pe_mean[9]:.3f} E20={pe_mean[19]:.3f}')
    print(f'  loss:      E1={el_mean[0]:.3f} E10={el_mean[9]:.3f} E20={el_mean[19]:.3f}')
    
    results[decay] = {
        'theta_err': theta_errs,
        'phi_err': phi_errs,
        'epoch_losses': epoch_losses,
    }

print('\n\n=== SUMMARY TABLE ===')
print(f'{"decay":>8}  {"theta_E10":>12}  {"theta_E20":>12}  {"theta_trend":>12}  {"loss_E10":>10}  {"loss_E20":>10}')
for decay, r in results.items():
    te = r['theta_err'].mean(axis=0)
    el = r['epoch_losses'].mean(axis=0)
    trend = te[19] - te[9]
    print(f'{decay:>8.2f}  {te[9]:>12.3f}  {te[19]:>12.3f}  {trend:>+12.3f}  {el[9]:>10.3f}  {el[19]:>10.3f}')

import pickle
out = '/Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/decay_sweep_results.pkl'
with open(out, 'wb') as f:
    pickle.dump(results, f)
print(f'\nSaved to {out}')
