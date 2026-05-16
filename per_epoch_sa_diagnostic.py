"""
Per-Epoch SA Accuracy Diagnostic
================================
Question: Does SA accuracy change as we use the SAME ground truth across 50 epochs?

Design:
- P=50 epochs, T=100, with SAME ground truth (θ*, φ*) throughout
- At each epoch, for the current topology:
  1. Build QUBO with TRUE params (θ*, φ*)
  2. Solve with SA → SA-optimal path combo
  3. Solve with brute force → true optimum
  4. Compute: gap(p) = loss(SA) - loss(optimal) for THIS QUBO
- Track: does SA accuracy degrade as the system "learns" (i.e., uses same params repeatedly)?

If SA accuracy stays constant across epochs → the problem is NOT the solver
If SA accuracy degrades as P increases → SA is the problem
"""

import sys, os, numpy as np, itertools
sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')
os.chdir('/Users/jon_claw/Thesis_brain/simulation')
from src.uav_routing.config import WorldConfig, GroundTruthConfig, QAMABConfig
from src.uav_routing.qubo import build_qubo
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
    best_loss = float('inf')
    for combo in itertools.product(range(K), repeat=N):
        l = expected_loss(combo, pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg)
        if l < best_loss:
            best_loss = l
            best_combo = combo
    return best_combo, best_loss

# Simulated Annealing oracle
from src.uav_routing.agents.oracle_agent import OracleAgent

world_cfg = WorldConfig(m=30, Z=9, N_flows=3, K_paths=4, comm_radius=350.0)
gt_cfg = GroundTruthConfig(n_faulty_uavs=4, theta_low=0.2, theta_high=0.4, n_faulty_zones=2, phi_low=0.2, phi_high=0.4, C_coll=5.0, d0=150.0, sigma_noise=0.05)
qamab_cfg = QAMABConfig(gamma_0=2.0, a=0.5, b=0.3, ucb_c=0.0, epoch_decay=1.0)

P, T, SEEDS = 50, 100, 10

# Track per-epoch metrics across seeds
all_sa_gaps = []  # list of arrays, shape (SEEDS, P)
all_sa_match = []

for si in range(SEEDS):
    seed = 42 + si
    rng = np.random.default_rng(seed)
    
    # Sample ground truth ONCE — stays fixed across all 50 epochs
    theta_star, phi_star = sample_ground_truth(rng, world_cfg, gt_cfg)
    
    sa_gaps_epoch = []
    sa_match_epoch = []
    
    for p in range(P):
        # New topology for this epoch (but same ground truth)
        rng_ep = np.random.default_rng(seed + p * 1000)
        topology = generate_topology(rng_ep, world_cfg)
        pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
        pair_min_dist = compute_all_pair_min_distances(pathset, topology.positions)
        N, K = 3, 4
        
        # Brute force optimum for this QUBO
        _, bf_loss = brute_force(pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg)
        
        # SA optimum for this QUBO
        rng2 = np.random.default_rng(seed + p * 1000 + 500)
        agent = OracleAgent(rng2, world_cfg, qamab_cfg, gt_cfg, theta_star, phi_star)
        agent._topology = topology
        agent._pathset = pathset
        agent._pair_min_dist = pair_min_dist
        
        sa_losses = []
        for _ in range(20):
            combo = agent.act(0, 0)
            l = expected_loss(combo, pathset, theta_star, phi_star, pair_min_dist, N, K, gt_cfg)
            sa_losses.append(l)
        sa_loss = min(sa_losses)
        
        gap = sa_loss - bf_loss
        sa_gaps_epoch.append(gap)
        sa_match_epoch.append(1 if gap < 0.001 else 0)
    
    all_sa_gaps.append(sa_gaps_epoch)
    all_sa_match.append(sa_match_epoch)

all_sa_gaps = np.array(all_sa_gaps)  # (SEEDS, P)
all_sa_match = np.array(all_sa_match)  # (SEEDS, P)

# Analyze: does SA accuracy change with epoch number?
avg_gap = all_sa_gaps.mean(axis=0)
avg_match = all_sa_match.mean(axis=0)

# Check early vs late
early_gap = avg_gap[:10].mean()
late_gap = avg_gap[-10:].mean()
early_match = avg_match[:10].mean()
late_match = avg_match[-10:].mean()

checkpoints = [0, 9, 19, 29, 39, 49]
labels = ['E1', 'E10', 'E20', 'E30', 'E40', 'E50']

print("=" * 70)
print("Per-Epoch SA Accuracy Diagnostic")
print(f"{SEEDS} seeds, {P} epochs, SAME ground truth throughout")
print("=" * 70)
print()
print("SA gap (SA_loss - BF_loss) per epoch:")
print(f"{'Epoch':>8} {'Gap Mean':>12} {'Gap Std':>10} {'Match%':>8}")
for c, l in zip(checkpoints, labels):
    g = avg_gap[c]
    s = all_sa_gaps[:, c].std()
    m = avg_match[c] * 100
    print(f"{l:>8} {g:>12.4f} {s:>10.4f} {m:>7.1f}%")

print()
print(f"Early epochs (1-10):  gap={early_gap:.4f}, match={early_match:.1%}")
print(f"Late epochs (41-50): gap={late_gap:.4f}, match={late_match:.1%}")
print(f"Change: gap {late_gap-early_gap:+.4f}, match {late_match-early_match:+.1%}")
print()

# Interpretation
if abs(late_gap - early_gap) < 0.5:
    print("→ SA accuracy is STABLE across epochs. The solver is NOT degrading.")
    print("→ The growing regret comes from something else — likely the QUBO structure")
    print("  (degenerate optima → SA picks clustered solutions even when it finds the true optimum)")
elif late_gap > early_gap + 0.5:
    print("→ SA accuracy DEGRADES with epochs.")
    print("→ The growing regret is from solver degradation.")
    print("→ Need to investigate why SA fails more as ground truth is 'seen' more.")
else:
    print("→ SA accuracy fluctuates but no clear trend.")

# Save results
import pickle
results = {
    'P': P, 'T': T, 'seeds': SEEDS,
    'all_sa_gaps': all_sa_gaps,
    'all_sa_match': all_sa_match,
    'avg_gap': avg_gap,
    'avg_match': avg_match,
    'checkpoints': checkpoints,
    'labels': labels
}
with open('/Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/per_epoch_sa_diagnostic.pkl', 'wb') as f:
    pickle.dump(results, f)
print()
print("Results saved to per_epoch_sa_diagnostic.pkl")