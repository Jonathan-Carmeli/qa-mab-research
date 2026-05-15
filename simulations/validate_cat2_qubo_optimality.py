"""
validate_cat2_qubo_optimality.py
================================
Category 2: QUBO formulation correctly encodes the UAV routing problem.

Test: Build QUBO with TRUE parameters (theta_star, phi_star) vs brute-force
enumeration of all K^N path combinations. Both should agree on optimal paths.

PASS criterion: ≥95% match rate across all seeds/iterations.
"""

import sys
import os
import numpy as np
from itertools import product
import csv
import json

sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')

from src.uav_routing.config import WorldConfig, GroundTruthConfig, QAMABConfig
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances, path_pair_min_distance
from src.uav_routing.ground_truth import sample_ground_truth
from src.uav_routing.qubo import build_qubo


def compute_expected_loss_no_noise(paths, pathset, theta, phi, positions, gt_cfg):
    """Total expected loss (no noise) for a path assignment under given theta/phi."""
    N = pathset.N

    selected_uav = np.array(
        [pathset.path_uav_membership[n, paths[n]] for n in range(N)]
    )
    selected_zone = np.array(
        [pathset.path_zone_membership[n, paths[n]] for n in range(N)]
    )

    fault_loss = float((selected_uav.astype(float) @ theta).sum()) + \
                 float((selected_zone.astype(float) @ phi).sum())

    shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
    np.fill_diagonal(shared, False)
    collision_loss = float(gt_cfg.C_coll * shared.sum())

    proximity_loss = 0.0
    for n in range(N):
        path_n = pathset.paths_per_flow[n][paths[n]]
        for l in range(N):
            if l == n:
                continue
            path_l = pathset.paths_per_flow[l][paths[l]]
            d = path_pair_min_distance(path_n, path_l, positions)
            proximity_loss += np.exp(-d / gt_cfg.d0)

    return fault_loss + collision_loss + proximity_loss


def brute_force_solve(pathset, theta, phi, positions, gt_cfg):
    """Enumerate all K^N combos, pick minimum true-loss assignment."""
    N, K = pathset.N, pathset.K
    best_loss = float('inf')
    best_paths = None

    for combo in product(range(K), repeat=N):
        paths = np.array(combo, dtype=int)
        loss = compute_expected_loss_no_noise(paths, pathset, theta, phi, positions, gt_cfg)
        if loss < best_loss:
            best_loss = loss
            best_paths = paths.copy()

    return best_paths, best_loss


def qubo_best_paths(Q, pathset):
    """Find minimum-energy combination by brute-force on QUBO energy."""
    N, K = pathset.N, pathset.K
    best_energy = float('inf')
    best_paths = None

    for combo in product(range(K), repeat=N):
        x = np.zeros(N * K, dtype=np.float64)
        for n in range(N):
            x[n * K + combo[n]] = 1.0
        energy = float(x.T @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_paths = np.array(combo, dtype=int)

    return best_paths, best_energy


def run(n_seeds=20, P=50, seed_base=42, pass_threshold=0.95):
    world_cfg = WorldConfig(m=15, Z=6, N_flows=3, K_paths=4, comm_radius=350.0)
    gt_cfg = GroundTruthConfig(
        n_faulty_uavs=4, theta_low=0.2, theta_high=0.4,
        n_faulty_zones=2, phi_low=0.2, phi_high=0.4,
        C_coll=5.0, d0=150.0, sigma_noise=0.0
    )
    qamab_cfg = QAMABConfig(
        lambda_onehot=10.0,
        sa_sweeps=200, sa_n_reads=20,
        sa_T_init=2.0, sa_T_final=0.05,
    )

    all_matches = []
    all_gaps = []
    per_seed_results = []

    print("=" * 65)
    print("CATEGORY 2: QUBO Optimality Test")
    print(f"Seeds={n_seeds}, P={P}, pass_threshold={pass_threshold*100}%")
    print("=" * 65)

    for s in range(n_seeds):
        seed = seed_base + s

        rng_gt = np.random.default_rng(seed)
        theta_star, phi_star = sample_ground_truth(rng_gt, world_cfg, gt_cfg)

        rng_topo = np.random.default_rng(seed + 1000)
        topology = generate_topology(rng_topo, world_cfg)
        pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
        positions = topology.positions

        # Build QUBO with TRUE params (one-time per seed)
        pair_min_dist = compute_all_pair_min_distances(pathset, positions)
        Q = build_qubo(
            theta_star, phi_star, pathset, pair_min_dist,
            qamab_cfg, gt_cfg,
            visit_counts=None, ucb_c=0.0
        )

        # Brute-force optimal (true loss)
        bf_paths, bf_loss = brute_force_solve(pathset, theta_star, phi_star, positions, gt_cfg)

        # QUBO best paths (energy minimization)
        qubo_paths, qubo_energy = qubo_best_paths(Q, pathset)

        # Evaluate QUBO paths under true loss
        qubo_loss = compute_expected_loss_no_noise(qubo_paths, pathset, theta_star, phi_star, positions, gt_cfg)

        match = np.array_equal(qubo_paths, bf_paths)
        gap = qubo_loss - bf_loss

        all_matches.append(match)
        all_gaps.append(gap)

        per_seed_results.append({
            'seed': seed,
            'match': bool(match),
            'gap': float(gap),
            'bf_paths': bf_paths.tolist(),
            'qubo_paths': qubo_paths.tolist(),
            'bf_loss': float(bf_loss),
            'qubo_loss': float(qubo_loss),
        })

        status = "✓" if match else f"✗ gap={gap:.6f}"
        print(f"  Seed {s+1:2d}/{n_seeds}: match={match}  gap={gap:+.6f}  {status}")

    matches_arr = np.array(all_matches)
    gaps_arr = np.array(all_gaps)
    match_rate = matches_arr.mean()
    mean_gap = gaps_arr.mean()

    passed = match_rate >= pass_threshold

    print("\n" + "=" * 65)
    print("RESULT")
    print("=" * 65)
    print(f"Match rate:     {match_rate*100:.1f}%  (threshold: {pass_threshold*100}%)")
    print(f"Mean gap:       {mean_gap:+.8f}")
    print(f"Std gap:        {gaps_arr.std():.8f}")
    print(f"Gap == 0:       {(np.abs(gaps_arr) < 1e-9).sum()}/{n_seeds}")
    print(f"Gap > 0:        {(gaps_arr > 1e-9).sum()}/{n_seeds}")
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: "
          f"match_rate={match_rate*100:.1f}% {'≥' if passed else '<'}{pass_threshold*100}%")

    results = {
        'category': 2,
        'test': 'qubo_optimality',
        'n_seeds': n_seeds,
        'P': P,
        'pass_threshold': pass_threshold,
        'match_rate': float(match_rate),
        'mean_gap': float(mean_gap),
        'std_gap': float(gaps_arr.std()),
        'passed': passed,
        'per_seed': per_seed_results,
        'metadata': {
            'N_flows': world_cfg.N_flows,
            'K_paths': world_cfg.K_paths,
            'm': world_cfg.m,
            'Z': world_cfg.Z,
        }
    }

    out_dir = '/Users/jon_claw/qa-mab-research/simulations/results/validation_cat2'
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, 'result.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'match', 'gap', 'bf_loss', 'qubo_loss'])
        writer.writeheader()
        for r in per_seed_results:
            writer.writerow({k: r[k] for k in ['seed', 'match', 'gap', 'bf_loss', 'qubo_loss']})

    print(f"\nSaved: {out_dir}/result.json + result.csv")
    return results


if __name__ == '__main__':
    results = run(n_seeds=20, P=50, seed_base=42, pass_threshold=0.95)
    sys.exit(0 if results['passed'] else 1)