"""
SA vs Optimal Comparison Test
=============================
Question: Does SA reliably find the globally optimal path selection for the UAV routing QUBO?

Design:
  For each seed (30 seeds):
    1. Sample ground truth (theta_star, phi_star) and topology
    2. Oracle (SA): Build QUBO with TRUE theta/phi, solve via SA → chosen paths
    3. Optimal (Brute Force): Enumerate all K^N = 4^3 = 64 combinations,
       compute expected loss (NO noise), pick the minimum
    4. Compare: gap = SA_loss - Optimal_loss

Both losses computed WITHOUT noise (sigma_noise=0). Pure solver quality test.
"""
import sys
import numpy as np
from itertools import product

sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')

from src.uav_routing.config import (
    WorldConfig, GroundTruthConfig, QAMABConfig, ExperimentConfig, TimeConfig
)
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances, path_pair_min_distance
from src.uav_routing.ground_truth import sample_ground_truth
from src.uav_routing.qubo import build_qubo
from src.uav_routing.sa_solver import sa_solve, decode_solution


def compute_expected_loss_no_noise(chosen_paths, pathset, theta_star, phi_star, positions, gt_cfg):
    """Compute total expected loss WITHOUT noise for a given path assignment.

    Loss = fault_loss + collision_loss + proximity_loss
    No sigma_noise added.
    """
    N = pathset.N

    # UAV and zone membership for selected paths
    selected_uav = np.array(
        [pathset.path_uav_membership[n, chosen_paths[n]] for n in range(N)]
    )  # (N, m) bool
    selected_zone = np.array(
        [pathset.path_zone_membership[n, chosen_paths[n]] for n in range(N)]
    )  # (N, Z) bool

    # Fault loss: sum of theta for UAVs on path + sum of phi for zones on path
    # selected_uav @ theta_star gives (N,) per-flow fault costs; sum over all flows
    fault_loss = float((selected_uav.astype(float) @ theta_star).sum()) + \
                 float((selected_zone.astype(float) @ phi_star).sum())

    # Collision: for each flow n, count how many other flows share a UAV
    shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0  # (N, N)
    np.fill_diagonal(shared, False)
    collision_counts = shared.sum(axis=1).astype(float)  # (N,)
    collision_loss = float(gt_cfg.C_coll * collision_counts.sum())

    # Proximity: sum_{n} sum_{l!=n} exp(-d_min / d0)
    proximity_loss = 0.0
    for n in range(N):
        path_n = pathset.paths_per_flow[n][chosen_paths[n]]
        for l in range(N):
            if l == n:
                continue
            path_l = pathset.paths_per_flow[l][chosen_paths[l]]
            d = path_pair_min_distance(path_n, path_l, positions)
            proximity_loss += np.exp(-d / gt_cfg.d0)

    total_loss = fault_loss + collision_loss + proximity_loss
    return total_loss


def brute_force_optimal(pathset, theta_star, phi_star, positions, gt_cfg):
    """Enumerate all K^N path combinations and return the one with minimum loss (no noise)."""
    N, K = pathset.N, pathset.K
    best_loss = float('inf')
    best_paths = None

    for combo in product(range(K), repeat=N):
        paths = np.array(combo, dtype=int)
        loss = compute_expected_loss_no_noise(paths, pathset, theta_star, phi_star, positions, gt_cfg)
        if loss < best_loss:
            best_loss = loss
            best_paths = paths.copy()

    return best_paths, best_loss


def run_oracle_sa(pathset, pair_min_dist, theta_star, phi_star, qamab_cfg, gt_cfg, rng):
    """Build QUBO with true theta/phi and solve via SA. Return chosen paths."""
    Q = build_qubo(
        theta_star,
        phi_star,
        pathset,
        pair_min_dist,
        qamab_cfg,
        gt_cfg,
        visit_counts=None,  # Oracle doesn't use UCB
        ucb_c=0.0,          # No exploration bonus for oracle
    )
    best_x, best_energy = sa_solve(
        Q, rng,
        n_reads=qamab_cfg.sa_n_reads,
        n_sweeps=qamab_cfg.sa_sweeps,
        T_init=qamab_cfg.sa_T_init,
        T_final=qamab_cfg.sa_T_final,
    )
    chosen_paths = decode_solution(best_x, pathset.N, pathset.K)
    return chosen_paths, best_energy


def main():
    N_SEEDS = 30
    BASE_SEED = 42

    world_cfg = WorldConfig(m=30, Z=9, N_flows=3, K_paths=4, comm_radius=350.0)
    gt_cfg = GroundTruthConfig(
        n_faulty_uavs=4, theta_low=0.2, theta_high=0.4,
        n_faulty_zones=2, phi_low=0.2, phi_high=0.4,
        C_coll=5.0, d0=150.0, sigma_noise=0.05  # sigma_noise exists in config but we don't use it
    )
    qamab_cfg = QAMABConfig(
        sa_sweeps=200,
        sa_n_reads=20,
        sa_T_init=2.0,
        sa_T_final=0.05,
    )

    print("=" * 70)
    print("SA vs Optimal Comparison Test")
    print(f"Seeds: {N_SEEDS}, N_flows={world_cfg.N_flows}, K_paths={world_cfg.K_paths}")
    print(f"Total combinations per seed: {world_cfg.K_paths ** world_cfg.N_flows}")
    print(f"SA params: n_reads={qamab_cfg.sa_n_reads}, n_sweeps={qamab_cfg.sa_sweeps}")
    print("Loss: NO noise (pure solver quality test)")
    print("=" * 70)

    gaps = []
    sa_losses = []
    opt_losses = []
    sa_matches_optimal = []
    sa_paths_list = []
    opt_paths_list = []

    for seed_idx in range(N_SEEDS):
        seed = BASE_SEED + seed_idx
        rng = np.random.default_rng(seed)

        # Sample ground truth
        rng_gt = np.random.default_rng(seed)
        theta_star, phi_star = sample_ground_truth(rng_gt, world_cfg, gt_cfg)

        # Generate topology and paths
        topology = generate_topology(rng, world_cfg)
        pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
        pair_min_dist = compute_all_pair_min_distances(pathset, topology.positions)

        # Brute-force optimal (no noise)
        opt_paths, opt_loss = brute_force_optimal(
            pathset, theta_star, phi_star, topology.positions, gt_cfg
        )

        # Oracle SA
        rng_sa = np.random.default_rng(seed + 10000)  # separate RNG for SA
        sa_paths, sa_energy = run_oracle_sa(
            pathset, pair_min_dist, theta_star, phi_star, qamab_cfg, gt_cfg, rng_sa
        )

        # Compute SA's actual expected loss (no noise) for the paths it chose
        sa_loss = compute_expected_loss_no_noise(
            sa_paths, pathset, theta_star, phi_star, topology.positions, gt_cfg
        )

        gap = sa_loss - opt_loss
        match = np.array_equal(sa_paths, opt_paths)

        gaps.append(gap)
        sa_losses.append(sa_loss)
        opt_losses.append(opt_loss)
        sa_matches_optimal.append(match)
        sa_paths_list.append(sa_paths.tolist())
        opt_paths_list.append(opt_paths.tolist())

        status = "✓ MATCH" if match else f"✗ GAP={gap:.4f}"
        print(f"  Seed {seed_idx+1:2d}/{N_SEEDS}: SA={sa_paths.tolist()} Opt={opt_paths.tolist()} "
              f"SA_loss={sa_loss:.4f} Opt_loss={opt_loss:.4f} {status}")

    gaps = np.array(gaps)
    sa_losses = np.array(sa_losses)
    opt_losses = np.array(opt_losses)

    # Check if SA found paths with SAME loss even if different path indices
    # (multiple optima possible)
    sa_same_loss = np.abs(gaps) < 1e-8
    sa_near_optimal = np.abs(gaps) < 0.01

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Seeds tested:              {N_SEEDS}")
    print(f"SA == Optimal (exact):     {sum(sa_matches_optimal)}/{N_SEEDS} ({100*sum(sa_matches_optimal)/N_SEEDS:.1f}%)")
    print(f"SA == Optimal (same loss): {sa_same_loss.sum()}/{N_SEEDS} ({100*sa_same_loss.sum()/N_SEEDS:.1f}%)")
    print(f"SA near-optimal (gap<0.01):{sa_near_optimal.sum()}/{N_SEEDS} ({100*sa_near_optimal.sum()/N_SEEDS:.1f}%)")
    print(f"Mean gap:                  {gaps.mean():.6f}")
    print(f"Std gap:                   {gaps.std():.6f}")
    print(f"Max gap:                   {gaps.max():.6f}")
    print(f"Min gap:                   {gaps.min():.6f}")
    print(f"Mean SA loss:              {sa_losses.mean():.4f}")
    print(f"Mean Optimal loss:         {opt_losses.mean():.4f}")
    print(f"Relative gap (mean):       {(gaps.mean()/opt_losses.mean())*100:.2f}%")

    # Detailed breakdown for non-matching seeds
    non_match_indices = [i for i, m in enumerate(sa_matches_optimal) if not m]
    if non_match_indices:
        print(f"\nNon-matching seeds ({len(non_match_indices)}):")
        for i in non_match_indices:
            print(f"  Seed {i+1}: SA={sa_paths_list[i]} Opt={opt_paths_list[i]} "
                  f"Gap={gaps[i]:.6f} (SA_loss={sa_losses[i]:.4f}, Opt_loss={opt_losses[i]:.4f})")
            # Check if SA paths give same loss (degenerate optimal)
            if abs(gaps[i]) < 1e-8:
                print(f"    → Same loss (degenerate optimum)")

    # Write results to file
    with open('/tmp/sa_vs_optimal_results.md', 'w') as f:
        f.write("# SA vs Optimal Comparison Test Results\n\n")
        f.write("## Question\n")
        f.write("Does SA reliably find the globally optimal path selection for the UAV routing QUBO?\n\n")
        f.write("## Setup\n")
        f.write(f"- Seeds: {N_SEEDS}\n")
        f.write(f"- N_flows: {world_cfg.N_flows}, K_paths: {world_cfg.K_paths}\n")
        f.write(f"- Total combinations: {world_cfg.K_paths ** world_cfg.N_flows}\n")
        f.write(f"- SA params: n_reads={qamab_cfg.sa_n_reads}, n_sweeps={qamab_cfg.sa_sweeps}\n")
        f.write(f"- Loss: NO noise (sigma_noise NOT added)\n")
        f.write(f"- QUBO: no UCB bonus (ucb_c=0), just true theta/phi costs\n\n")
        f.write("## Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| SA == Optimal (exact paths) | {sum(sa_matches_optimal)}/{N_SEEDS} ({100*sum(sa_matches_optimal)/N_SEEDS:.1f}%) |\n")
        f.write(f"| SA == Optimal (same loss) | {sa_same_loss.sum()}/{N_SEEDS} ({100*sa_same_loss.sum()/N_SEEDS:.1f}%) |\n")
        f.write(f"| SA near-optimal (gap<0.01) | {sa_near_optimal.sum()}/{N_SEEDS} ({100*sa_near_optimal.sum()/N_SEEDS:.1f}%) |\n")
        f.write(f"| Mean gap | {gaps.mean():.6f} |\n")
        f.write(f"| Std gap | {gaps.std():.6f} |\n")
        f.write(f"| Max gap | {gaps.max():.6f} |\n")
        f.write(f"| Mean SA loss | {sa_losses.mean():.4f} |\n")
        f.write(f"| Mean Optimal loss | {opt_losses.mean():.4f} |\n")
        f.write(f"| Relative gap | {(gaps.mean()/opt_losses.mean())*100:.2f}% |\n\n")

        if non_match_indices:
            f.write("## Non-matching Seeds\n\n")
            for i in non_match_indices:
                degenerate = " (degenerate optimum — same loss)" if abs(gaps[i]) < 1e-8 else ""
                f.write(f"- Seed {i+1}: SA={sa_paths_list[i]} Opt={opt_paths_list[i]} "
                        f"Gap={gaps[i]:.6f}{degenerate}\n")
            f.write("\n")

        f.write("## Conclusion\n\n")
        if sa_same_loss.sum() == N_SEEDS:
            f.write("**SA reliably finds the global optimum.** All seeds achieved optimal loss.\n")
        elif sa_same_loss.sum() >= N_SEEDS * 0.9:
            f.write(f"**SA is highly reliable.** {sa_same_loss.sum()}/{N_SEEDS} seeds achieved optimal loss. "
                    f"Mean gap when suboptimal: {gaps[~sa_same_loss].mean():.6f}\n")
        elif sa_same_loss.sum() >= N_SEEDS * 0.7:
            f.write(f"**SA is mostly reliable.** {sa_same_loss.sum()}/{N_SEEDS} seeds achieved optimal loss. "
                    f"Consider increasing n_reads or n_sweeps.\n")
        else:
            f.write(f"**SA has significant gaps.** Only {sa_same_loss.sum()}/{N_SEEDS} seeds achieved optimal loss. "
                    f"Mean gap: {gaps.mean():.6f}. SA may need tuning or the QUBO landscape is challenging.\n")

    print(f"\nResults written to /tmp/sa_vs_optimal_results.md")


if __name__ == "__main__":
    main()
