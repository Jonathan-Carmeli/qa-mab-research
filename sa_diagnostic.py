import sys, os, numpy as np, itertools
sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')
os.chdir('/Users/jon_claw/Thesis_brain/simulation')
from src.uav_routing.config import WorldConfig, GroundTruthConfig, QAMABConfig
from src.uav_routing.qubo import build_qubo
from src.uav_routing.agents.oracle_agent import OracleAgent
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances
from src.uav_routing.ground_truth import sample_ground_truth

def compute_expected_loss(combo, pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg):
    selected_uav = np.array([pathset.path_uav_membership[n, combo[n]] for n in range(N)])
    shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
    np.fill_diagonal(shared, False)
    collision_counts = shared.sum(axis=1).astype(float)
    prox = np.zeros(N)
    for n in range(N):
        for l in range(n):
            d = pair_min_dist[n*K+combo[n], l*K+combo[l]]
            prox[n] += np.exp(-d / gt_cfg.d0)
            prox[l] += np.exp(-d / gt_cfg.d0)
    fault = np.array([theta_star[pathset.path_uav_membership[n, combo[n]]].sum() + 
                      phi_star[pathset.path_zone_membership[n, combo[n]]].sum() for n in range(N)])
    return float((fault + gt_cfg.C_coll * collision_counts + prox).sum())

def brute_force_optimal(pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg):
    best_combo, best_loss = None, float('inf')
    for combo in itertools.product(range(K), repeat=N):
        loss = compute_expected_loss(combo, pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg)
        if loss < best_loss:
            best_loss = loss
            best_combo = combo
    return best_combo, best_loss

world_cfg = WorldConfig(m=30, Z=9, N_flows=3, K_paths=4, comm_radius=350.0)
gt_cfg = GroundTruthConfig(n_faulty_uavs=4, theta_low=0.2, theta_high=0.4, n_faulty_zones=2, phi_low=0.2, phi_high=0.4, C_coll=5.0, d0=150.0, sigma_noise=0.05)
qamab_cfg = QAMABConfig(gamma_0=2.0, a=0.5, b=0.3, ucb_c=0.0, epoch_decay=1.0)

SEEDS = 100
sa_matches, sa_gaps, diag_variances = [], [], []

for si in range(SEEDS):
    seed = 42 + si
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1000)
    
    topology = generate_topology(rng, world_cfg)
    pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
    pair_min_dist = compute_all_pair_min_distances(pathset, topology.positions)
    theta_star, phi_star = sample_ground_truth(rng, world_cfg, gt_cfg)
    N, K = 3, 4
    
    Q = build_qubo(theta_star, phi_star, pathset, pair_min_dist, qamab_cfg, gt_cfg, visit_counts=None, ucb_c=0.0)
    
    flow_diag_vars = [np.std([Q[n*K+k, n*K+k] for k in range(K)]) for n in range(N)]
    mean_flow_diag_var = np.mean(flow_diag_vars)
    diag_variances.append(mean_flow_diag_var)
    
    bf_combo, bf_loss = brute_force_optimal(pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg)
    
    agent = OracleAgent(rng2, world_cfg, qamab_cfg, gt_cfg, theta_star, phi_star)
    agent._topology = topology
    agent._pathset = pathset
    agent._pair_min_dist = pair_min_dist
    
    sa_losses = []
    for _ in range(20):
        combo = agent.act(0, 0)
        loss = compute_expected_loss(combo, pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg)
        sa_losses.append(loss)
    sa_loss = min(sa_losses)
    
    gap = sa_loss - bf_loss
    sa_matches.append(1 if gap < 0.001 else 0)
    sa_gaps.append(gap)

sa_matches = np.array(sa_matches)
sa_gaps = np.array(sa_gaps)
diag_variances = np.array(diag_variances)

print("=" * 60)
print("SA Accuracy Diagnostic: 100 seeds, T=P=1, no UCB")
print("=" * 60)
print(f"SA matches brute force: {sa_matches.mean():.1%} ({sa_matches.sum()}/{SEEDS})")
print(f"Mean gap (SA - BF): {sa_gaps.mean():.6f} ± {sa_gaps.std():.6f}")
print(f"Max gap: {sa_gaps.max():.6f}")
print(f"Mean diagonal variance: {diag_variances.mean():.4f}")
print()

high_var_mask = diag_variances > np.median(diag_variances)
low_var_mask = ~high_var_mask
print(f"High diagonal variance: SA accuracy = {sa_matches[high_var_mask].mean():.1%}")
print(f"Low diagonal variance:  SA accuracy = {sa_matches[low_var_mask].mean():.1%}")
print()

sort_idx = np.argsort(diag_variances)
n_bins, bin_size = 5, SEEDS // 5
print("SA accuracy by diagonal variance (quintiles):")
for b in range(n_bins):
    s, e = b * bin_size, (b+1) * bin_size if b < n_bins-1 else SEEDS
    print(f"  Quintile {b+1}: var={diag_variances[sort_idx[s:e]].mean():.4f}, acc={sa_matches[sort_idx[s:e]].mean():.1%}")

print()
if sa_matches.mean() > 0.85:
    print("SA is accurate ({:.0f}%) even with low diagonal variance")
    print("→ SA accuracy is NOT the bottleneck")
elif sa_matches[high_var_mask].mean() > sa_matches[low_var_mask].mean() + 0.2:
    print("SA significantly better when diagonal variance is high")
    print("→ Diagonal asymmetry matters for SA")
else:
    print("SA accuracy is moderate, not strongly dependent on diagonal variance")
    print("→ SA itself contributes to the problem, not just the QUBO")