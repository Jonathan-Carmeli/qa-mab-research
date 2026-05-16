"""UCB Ablation: QA-MAB with vs without UCB (ucb_c=3.0 vs ucb_c=0.0)
P=3 epochs, T=100 steps, 5 seeds. Quick smoke test.
"""
import sys
sys.path.insert(0, "/Users/jon_claw/Thesis_brain")

import numpy as np
from simulation.src.uav_routing.config import ExperimentConfig, TimeConfig, QAMABConfig
from simulation.src.uav_routing.runner import run_single
from simulation.src.uav_routing.ground_truth import sample_ground_truth
from simulation.src.uav_routing.agents.qamab_agent import QAMABAgent

# Config: P=3, T=100, 5 seeds
cfg = ExperimentConfig(
    time=TimeConfig(P_epochs=3, T_steps=100),
    n_seeds=5,
    base_seed=42,
)

results = {}

for ucb_c_val in [3.0, 0.0]:
    label = f"ucb_c={ucb_c_val}"
    print(f"\n=== Running QA-MAB with {label} ===")
    
    all_losses = []
    all_coll = []
    all_theta_err = []
    
    for seed_idx in range(cfg.n_seeds):
        seed = cfg.base_seed + seed_idx
        rng_gt = np.random.default_rng(seed)
        theta_star, phi_star = sample_ground_truth(rng_gt, cfg.world, cfg.gt)
        
        def factory(rng, ts, ps, _ucb=ucb_c_val):
            return QAMABAgent(rng, cfg.world, cfg.qamab, cfg.gt, ucb_c=_ucb)
        
        print(f"  seed {seed_idx+1}/{cfg.n_seeds}", flush=True)
        result = run_single(cfg, factory, seed, theta_star=theta_star, phi_star=phi_star)
        all_losses.append(result.losses_log)
        all_coll.append(result.coll_log)
        all_theta_err.append(result.theta_err_log)
    
    losses_arr = np.stack(all_losses)  # (5, P, T, N)
    coll_arr = np.stack(all_coll)      # (5, P, T)
    theta_err_arr = np.stack(all_theta_err)  # (5, P)
    
    mean_loss = losses_arr.mean()
    # Collision rate: fraction of (seed, epoch, step) with at least 1 collision
    coll_rate = (coll_arr > 0).mean()
    mean_theta_err_final = theta_err_arr[:, -1].mean()
    std_theta_err_final = theta_err_arr[:, -1].std()
    
    # Per-epoch mean loss
    epoch_losses = losses_arr.mean(axis=(0, 2, 3))  # (P,)
    
    results[label] = {
        'mean_loss': mean_loss,
        'coll_rate': coll_rate,
        'theta_err_final': mean_theta_err_final,
        'theta_err_final_std': std_theta_err_final,
        'epoch_losses': epoch_losses,
        'total_loss': losses_arr.sum(),
    }
    
    print(f"  Mean loss/flow-step: {mean_loss:.4f}")
    print(f"  Collision rate: {coll_rate:.4f}")
    print(f"  θ error (final epoch): {mean_theta_err_final:.4f} ± {std_theta_err_final:.4f}")
    print(f"  Per-epoch losses: {epoch_losses}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for label, r in results.items():
    print(f"\n{label}:")
    print(f"  Mean loss/flow-step: {r['mean_loss']:.4f}")
    print(f"  Collision rate:      {r['coll_rate']:.4f}")
    print(f"  θ_err (final epoch): {r['theta_err_final']:.4f} ± {r['theta_err_final_std']:.4f}")
    for ep_i, ep_l in enumerate(r['epoch_losses']):
        print(f"  Epoch {ep_i+1} mean loss:  {ep_l:.4f}")
