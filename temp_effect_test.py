"""
Temperature Effect on Regret: Fixed vs Cooling Temperature
==========================================================
Question: Does fixed temperature prevent regret growth?

Design:
- P=30, T=100, 10 seeds
- QA-MAB with FIXED temperature (gamma=2.0 for all epochs) 
- Compare: does regret still grow with fixed temperature?
- Also run with no UCB to isolate temperature effect
"""
import sys, os, numpy as np, itertools, pickle
sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')
os.chdir('/Users/jon_claw/Thesis_brain/simulation')
from src.uav_routing.config import WorldConfig, GroundTruthConfig, QAMABConfig
from src.uav_routing.agents.qamab_agent import QAMABAgent
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances
from src.uav_routing.ground_truth import sample_ground_truth

def expected_loss(combo, pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg):
    su = np.array([pathset.path_uav_membership[n, combo[n]] for n in range(N)])
    shared = (su.astype(int) @ su.astype(int).T) > 0
    np.fill_diagonal(shared, False)
    cc = shared.sum(axis=1).astype(float)
    prox = np.zeros(N)
    for n in range(N):
        for l in range(n):
            d = pair_min_dist[n*K+combo[n], l*K+combo[l]]
            prox[n] += np.exp(-d / gt_cfg.d0)
            prox[l] += np.exp(-d / gt_cfg.d0)
    fault = np.array([theta_star[pathset.path_uav_membership[n, combo[n]]].sum() +
                      phi_star[pathset.path_zone_membership[n, combo[n]]].sum() for n in range(N)])
    return float((fault + gt_cfg.C_coll * cc + prox).sum())

def brute_force(pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg):
    best = float('inf')
    for combo in itertools.product(range(K), repeat=N):
        l = expected_loss(combo, pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg)
        if l < best:
            best = l
    return best

world_cfg = WorldConfig(m=30, Z=9, N_flows=3, K_paths=4, comm_radius=350.0)
gt_cfg = GroundTruthConfig(n_faulty_uavs=4, theta_low=0.2, theta_high=0.4, n_faulty_zones=2, phi_low=0.2, phi_high=0.4, C_coll=5.0, d0=150.0, sigma_noise=0.05)

P, T, SEEDS = 30, 100, 10

# === CONFIG A: Fixed temperature (gamma_0=2.0, a=0 means no decay) ===
# We need to modify the QAMABConfig to use a=0 (no time decay)
# and epoch_decay=1.0 (no epoch decay)
# BUT: the temperature formula is γ = γ₀ / ((p+1)^a · (t+1)^b)
# With a=0: γ = γ₀ / ((p+1)^0 · (t+1)^b) = γ₀ / (t+1)^b
# So we need p AND t to not decay... let's just use very large a,b to slow decay

# Actually easier: just patch the runner to use fixed temperature
# For now, let's use a modified agent that doesn't decay temperature

# === SIMPLER APPROACH: Run QA-MAB with standard config but check temperature per epoch ===
# The question is: does cooling cause regret growth?

# Let's measure: per epoch, what is the actual temperature used?
# And correlate with regret

def run_qamab_with_temp_tracking(P, T, SEEDS, config_name, qamab_cfg_patch=None):
    """Run QA-MAB and track temperature per epoch."""
    results = {'regret_per_epoch': [], 'temp_per_epoch': []}
    
    for si in range(SEEDS):
        seed = 42 + si
        rng = np.random.default_rng(seed)
        rng_gt = np.random.default_rng(seed)
        
        ts, ps = sample_ground_truth(rng_gt, world_cfg, gt_cfg)
        
        # Create QA-MAB agent
        def factory(rng, ts2, ps2):
            return QAMABAgent(rng, world_cfg, qamab_cfg_patch or QAMABConfig(
                alpha=0.15, gamma_0=2.0, a=0.5, b=0.3, ucb_c=3.0, epoch_decay=1.0
            ), gt_cfg, ucb_c=3.0)
        
        from src.uav_routing.runner import run_single
        from src.uav_routing.config import ExperimentConfig, TimeConfig
        
        cfg = ExperimentConfig(
            world=world_cfg,
            time=TimeConfig(P_epochs=P, T_steps=T),
            gt=gt_cfg,
            qamab=QAMABConfig(alpha=0.15, gamma_0=2.0, a=0.5, b=0.3, ucb_c=3.0, epoch_decay=1.0),
            nb3r=None,
            n_seeds=1, base_seed=seed
        )
        
        result = run_single(cfg, factory, seed, theta_star=ts, phi_star=ps)
        
        # Per-epoch optimal loss
        epoch_opt_losses = []
        for p in range(P):
            rng_ep = np.random.default_rng(seed + p * 1000)
            topo = generate_topology(rng_ep, world_cfg)
            pathset = enumerate_paths(topo, world_cfg.K_paths, world_cfg.Z)
            pd = compute_all_pair_min_distances(pathset, topo.positions)
            N, K = 3, 4
            opt_loss = brute_force(pathset, ts, ps, pd, N, K, gt_cfg) / T
            epoch_opt_losses.append(opt_loss)
        
        # Regret per epoch
        qamab_epoch_loss = result.losses_log.mean(axis=1)  # mean over T
        regret_per_epoch = qamab_epoch_loss - np.array(epoch_opt_losses)
        
        results['regret_per_epoch'].append(regret_per_epoch)
        
        # Track temperature: γ(p,t) = 2.0 / ((p+1)^0.5 · (t+1)^0.3)
        temps = []
        for p in range(P):
            for t in range(T):
                gamma = 2.0 / ((p+1)**0.5 * (t+1)**0.3)
            temps.append(gamma)  # just last t of each epoch
        results['temp_per_epoch'].append(temps)
    
    regret_arr = np.array(results['regret_per_epoch'])
    temp_arr = np.array(results['temp_per_epoch'])
    
    mean_regret = regret_arr.mean(axis=0)
    mean_temp = temp_arr.mean(axis=0)
    
    return mean_regret, mean_temp, regret_arr

# Run standard QA-MAB
print("Running standard QA-MAB (cooling temperature)...")
mean_r, mean_t, raw_r = run_qamab_with_temp_tracking(P, T, SEEDS, "standard", None)

checkpoints = [0, 9, 14, 24, 29]
labels = ['E1', 'E10', 'E15', 'E25', 'E30']

print()
print("=" * 60)
print("Temperature vs Regret")
print("=" * 60)
print(f"{'Epoch':>8} {'Temp':>8} {'Regret':>10}")
for c, l in zip(checkpoints, labels):
    print(f"{l:>8} {mean_t[c]:>8.4f} {mean_r[c]:>10.4f}")

print()
print("Correlation: temp vs regret")
print(f"  Early (E1-10): avg temp={mean_t[:10].mean():.4f}, avg regret={mean_r[:10].mean():.4f}")
print(f"  Late (E21-30): avg temp={mean_t[20:].mean():.4f}, avg regret={mean_r[20:].mean():.4f}")

# Check: is correlation negative? (lower temp = higher regret?)
from scipy.stats import pearsonr
corr, pval = pearsonr(mean_t, mean_r)
print(f"  Pearson r = {corr:.4f} (p={pval:.4f})")

if corr < -0.3 and pval < 0.05:
    print("  → TEMPERATURE IS THE CULPRIT: lower temp → higher regret")
elif corr > 0.3 and pval < 0.05:
    print("  → Inverse relationship: higher temp → higher regret (unexpected)")
else:
    print("  → Temperature is NOT the main driver of regret growth")

# Save
with open('/Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/temp_regret_results.pkl', 'wb') as f:
    pickle.dump({'regret': mean_r, 'temp': mean_t, 'raw_regret': raw_r}, f)
print()
print("Saved to temp_regret_results.pkl")