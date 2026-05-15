"""
validate_cat3_sa_solver_accuracy.py
====================================
Category 3: SA solver correctly solves QUBO instances for UAV routing.

Test: Generate random QUBO matrices (from real UAV scenarios), compare SA
solution vs brute-force optimum across many instances.

PASS criterion: ≥85% exact match rate with n_reads=50.
"""

import sys
import os
import numpy as np
from itertools import product
import csv
import json
import time

sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')

from src.uav_routing.config import WorldConfig, GroundTruthConfig, QAMABConfig
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances, path_pair_min_distance
from src.uav_routing.ground_truth import sample_ground_truth
from src.uav_routing.qubo import build_qubo
from src.uav_routing.sa_solver import sa_solve, decode_solution


def compute_expected_loss_no_noise(paths, pathset, theta, phi, positions, gt_cfg):
    """Total expected loss (no noise) for a path assignment."""
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


def brute_force_qubo_solve(Q, N, K):
    """Enumerate all K^N combos, pick minimum x^T Q x."""
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


def run(n_instances=50, n_reads=50, seed_base=100, pass_threshold=0.85):
    world_cfg = WorldConfig(m=15, Z=6, N_flows=3, K_paths=4, comm_radius=350.0)
    gt_cfg = GroundTruthConfig(
        n_faulty_uavs=4, theta_low=0.2, theta_high=0.4,
        n_faulty_zones=2, phi_low=0.2, phi_high=0.4,
        C_coll=5.0, d0=150.0, sigma_noise=0.0
    )
    qamab_cfg = QAMABConfig(
        lambda_onehot=10.0,
        sa_sweeps=200, sa_n_reads=n_reads,
        sa_T_init=2.0, sa_T_final=0.05,
    )

    all_matches = []
    all_rel_gaps = []
    per_instance = []

    print("=" * 65)
    print("CATEGORY 3: SA Solver Accuracy Test")
    print(f"Instances={n_instances}, n_reads={n_reads}, pass_threshold={pass_threshold*100}%")
    print("=" * 65)

    for i in range(n_instances):
        seed = seed_base + i
        rng = np.random.default_rng(seed)

        # Sample ground truth and topology
        rng_gt = np.random.default_rng(seed)
        theta_star, phi_star = sample_ground_truth(rng_gt, world_cfg, gt_cfg)

        rng_topo = np.random.default_rng(seed + 1000)
        topology = generate_topology(rng_topo, world_cfg)
        pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
        positions = topology.positions
        pair_min_dist = compute_all_pair_min_distances(pathset, positions)

        # Build QUBO
        Q = build_qubo(
            theta_star, phi_star, pathset, pair_min_dist,
            qamab_cfg, gt_cfg,
            visit_counts=None, ucb_c=0.0
        )

        # Brute-force optimal
        bf_paths, bf_energy = brute_force_qubo_solve(Q, pathset.N, pathset.K)

        # SA solve
        rng_sa = np.random.default_rng(seed + 5000)
        t0 = time.time()
        best_x, best_energy = sa_solve(
            Q, rng_sa,
            n_reads=n_reads,
            n_sweeps=qamab_cfg.sa_sweeps,
            T_init=qamab_cfg.sa_T_init,
            T_final=qamab_cfg.sa_T_final,
        )
        sa_elapsed = time.time() - t0
        sa_paths = decode_solution(best_x, pathset.N, pathset.K)

        # SA's actual loss under true parameters
        sa_loss = compute_expected_loss_no_noise(sa_paths, pathset, theta_star, phi_star, positions, gt_cfg)
        bf_loss = compute_expected_loss_no_noise(bf_paths, pathset, theta_star, phi_star, positions, gt_cfg)

        match = np.array_equal(sa_paths, bf_paths)
        # Relative gap: (SA_energy - BF_energy) / |BF_energy|, clipped to avoid div by zero
        if abs(bf_energy) > 1e-12:
            rel_gap = (best_energy - bf_energy) / abs(bf_energy)
        else:
            rel_gap = 0.0

        all_matches.append(match)
        all_rel_gaps.append(rel_gap)

        per_instance.append({
            'instance': i,
            'seed': seed,
            'match': bool(match),
            'rel_gap': float(rel_gap),
            'sa_energy': float(best_energy),
            'bf_energy': float(bf_energy),
            'sa_paths': sa_paths.tolist(),
            'bf_paths': bf_paths.tolist(),
            'sa_time_ms': round(sa_elapsed * 1000, 1),
        })

        status = "✓" if match else f"✗ rel_gap={rel_gap:.4f}"
        print(f"  Inst {i+1:2d}/{n_instances}: match={match}  rel_gap={rel_gap:+.6f}  "
              f"SA_e={best_energy:.4f} BF_e={bf_energy:.4f}  {status}")

    matches_arr = np.array(all_matches)
    rel_gaps_arr = np.array(all_rel_gaps)
    match_rate = matches_arr.mean()
    mean_rel_gap = rel_gaps_arr.mean()

    passed = match_rate >= pass_threshold

    print("\n" + "=" * 65)
    print("RESULT")
    print("=" * 65)
    print(f"Match rate:     {match_rate*100:.1f}%  (threshold: {pass_threshold*100}%)")
    print(f"Mean rel gap:   {mean_rel_gap:+.6f}")
    print(f"Std rel gap:    {rel_gaps_arr.std():.6f}")
    print(f"Exact matches:  {matches_arr.sum():.0f}/{n_instances}")
    print(f"Mean SA time:   {np.mean([p['sa_time_ms'] for p in per_instance]):.1f}ms")
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: "
          f"match_rate={match_rate*100:.1f}% {'≥' if passed else '<'}{pass_threshold*100}%")

    results = {
        'category': 3,
        'test': 'sa_solver_accuracy',
        'n_instances': n_instances,
        'n_reads': n_reads,
        'pass_threshold': pass_threshold,
        'match_rate': float(match_rate),
        'mean_rel_gap': float(mean_rel_gap),
        'std_rel_gap': float(rel_gaps_arr.std()),
        'passed': passed,
        'per_instance': per_instance,
        'metadata': {
            'N_flows': world_cfg.N_flows,
            'K_paths': world_cfg.K_paths,
            'sa_sweeps': qamab_cfg.sa_sweeps,
        }
    }

    out_dir = '/Users/jon_claw/qa-mab-research/simulations/results/validation_cat3'
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out_dir, 'result.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['instance', 'seed', 'match', 'rel_gap', 'sa_energy', 'bf_energy', 'sa_time_ms'])
        writer.writeheader()
        for r in per_instance:
            writer.writerow({k: r[k] for k in ['instance', 'seed', 'match', 'rel_gap', 'sa_energy', 'bf_energy', 'sa_time_ms']})

    print(f"\nSaved: {out_dir}/result.json + result.csv")
    return results


if __name__ == '__main__':
    results = run(n_instances=50, n_reads=50, seed_base=100, pass_threshold=0.85)
    sys.exit(0 if results['passed'] else 1)