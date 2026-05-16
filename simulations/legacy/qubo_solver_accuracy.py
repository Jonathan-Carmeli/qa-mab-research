"""
qubo_solver_accuracy.py
========================
Measures whether QUBO solution quality improves as the learner refines its estimates.

Design:
  - Fixed ground truth (θ*, φ*) and fixed topology — learner updates θ̂, φ̂ over iterations
  - Iteration t: both Oracle and QA-MAB solve their respective QUBOs via BRUTE-FORCE
    (exhaustive enumeration of all K^N combinations)
  - Oracle: QUBO built with TRUE params (θ*, φ*) → best routing under TRUE env
  - QA-MAB: QUBO built with CURRENT estimates (θ̂, φ̂) → best routing given imperfect estimates
  - Both solutions evaluated under the TRUE environment (no noise)
  - gap[t] = loss_QAMAB[t] - loss_Oracle[t] should → 0 as θ̂ → θ*

Since both use brute-force, any gap is purely from imperfect estimates — NOT solver quality.
"""
import sys
import os
import numpy as np
from itertools import product
import csv
import pickle

sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')

from src.uav_routing.config import (
    WorldConfig, GroundTruthConfig, QAMABConfig
)
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances, path_pair_min_distance
from src.uav_routing.ground_truth import sample_ground_truth
from src.uav_routing.qubo import build_qubo
from src.uav_routing.sa_solver import sa_solve, decode_solution


# ─── Loss evaluation ───────────────────────────────────────────────────────────

def compute_expected_loss_no_noise(chosen_paths, pathset, theta_star, phi_star, positions, gt_cfg):
    """Total expected loss (no noise) for a path assignment under TRUE parameters."""
    N, K, m, Z = pathset.N, pathset.K, pathset.m, pathset.Z

    selected_uav = np.array(
        [pathset.path_uav_membership[n, chosen_paths[n]] for n in range(N)]
    )
    selected_zone = np.array(
        [pathset.path_zone_membership[n, chosen_paths[n]] for n in range(N)]
    )

    fault_loss = float((selected_uav.astype(float) @ theta_star).sum()) + \
                 float((selected_zone.astype(float) @ phi_star).sum())

    shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
    np.fill_diagonal(shared, False)
    collision_loss = float(gt_cfg.C_coll * shared.sum())

    proximity_loss = 0.0
    for n in range(N):
        path_n = pathset.paths_per_flow[n][chosen_paths[n]]
        for l in range(n + 1, N):
            path_l = pathset.paths_per_flow[l][chosen_paths[l]]
            d = path_pair_min_distance(path_n, path_l, positions)
            proximity_loss += 2 * np.exp(-d / gt_cfg.d0)

    return fault_loss + collision_loss + proximity_loss


def brute_force_solve(pathset, theta, phi, positions, gt_cfg, visit_counts=None, ucb_c=0.0):
    """Enumerate all K^N path combinations, pick minimum-loss assignment given theta, phi."""
    N, K = pathset.N, pathset.K
    best_loss = float('inf')
    best_paths = None

    for combo in product(range(K), repeat=N):
        paths = np.array(combo, dtype=int)

        selected_uav = np.array(
            [pathset.path_uav_membership[n, paths[n]] for n in range(N)]
        )
        selected_zone = np.array(
            [pathset.path_zone_membership[n, paths[n]] for n in range(N)]
        )

        # Fault cost under given theta/phi
        fault_cost = float((selected_uav.astype(float) @ theta).sum()) + \
                     float((selected_zone.astype(float) @ phi).sum())

        # UCB bonus (negative = reward for exploration)
        ucb_bonus = 0.0
        if ucb_c > 0 and visit_counts is not None:
            for n in range(N):
                ucb_bonus -= ucb_c / np.sqrt(visit_counts[n, paths[n]] + 1)

        # QUBO score (lower is better)
        score = fault_cost + ucb_bonus

        if score < best_loss:
            best_loss = score
            best_paths = paths.copy()

    return best_paths, best_loss


# ─── Residual credit assignment (v3) ─────────────────────────────────────────

def residual_credit_update(pathset, losses, chosen_paths, theta_hat, phi_hat, alpha, gt_cfg, positions):
    """Update θ̂, φ̂ using residual credit assignment (v3) for one batch."""
    N, m, Z = pathset.N, pathset.m, pathset.Z

    selected_uav = np.array(
        [pathset.path_uav_membership[n, chosen_paths[n]] for n in range(N)]
    )
    selected_zone = np.array(
        [pathset.path_zone_membership[n, chosen_paths[n]] for n in range(N)]
    )

    # Collision count per flow
    shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
    np.fill_diagonal(shared, False)
    collision_counts = shared.sum(axis=1).astype(float)

    # Proximity per flow
    proximity = np.zeros(N)
    for n in range(N):
        path_n = pathset.paths_per_flow[n][chosen_paths[n]]
        for l in range(N):
            if l == n:
                continue
            path_l = pathset.paths_per_flow[l][chosen_paths[l]]
            d = path_pair_min_distance(path_n, path_l, positions)
            proximity[n] += np.exp(-d / gt_cfg.d0)

    for n in range(N):
        # Isolate fault-related loss
        L_fault = losses[n] - gt_cfg.C_coll * collision_counts[n] - proximity[n]

        # Update UAV estimates
        for i in range(m):
            if not pathset.path_uav_membership[n, chosen_paths[n], i]:
                continue
            # Subtract contributions of all OTHER UAVs on this path
            other_theta_sum = sum(
                theta_hat[j] for j in range(m)
                if j != i and pathset.path_uav_membership[n, chosen_paths[n], j]
            )
            # Zone contributions on this path
            zone_phi_sum = sum(
                phi_hat[z] for z in range(Z)
                if pathset.path_zone_membership[n, chosen_paths[n], z]
            )
            residual = L_fault - other_theta_sum - zone_phi_sum
            theta_hat[i] += alpha * (residual - theta_hat[i])

        # Update zone estimates
        for z in range(Z):
            if not pathset.path_zone_membership[n, chosen_paths[n], z]:
                continue
            other_phi_sum = sum(
                phi_hat[zz] for zz in range(Z)
                if zz != z and pathset.path_zone_membership[n, chosen_paths[n], zz]
            )
            uav_theta_sum = sum(
                theta_hat[i] for i in range(m)
                if pathset.path_uav_membership[n, chosen_paths[n], i]
            )
            residual = L_fault - uav_theta_sum - other_phi_sum
            phi_hat[z] += alpha * (residual - phi_hat[z])


# ─── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(P=30, T=50, seed=42):
    """
    Returns:
      dict with gaps, theta_errors, phi_errors, loss_oracle, loss_qamab, theta_star, phi_star
    """
    # Config
    world_cfg = WorldConfig(m=15, Z=6, N_flows=3, K_paths=4, comm_radius=350.0)
    gt_cfg = GroundTruthConfig(
        n_faulty_uavs=4, theta_low=0.2, theta_high=0.4,
        n_faulty_zones=2, phi_low=0.2, phi_high=0.4,
        C_coll=5.0, d0=150.0, sigma_noise=0.0
    )
    qamab_cfg = QAMABConfig(
        alpha=0.15,
        ucb_c=3.0,
        gamma_0=2.0,
        epoch_decay=1.0,
        sa_sweeps=200,
        sa_n_reads=20,
        sa_T_init=2.0,
        sa_T_final=0.05,
    )

    rng = np.random.default_rng(seed)

    # ── Sample ground truth and topology (FIXED for entire experiment) ─────────
    rng_gt = np.random.default_rng(seed)
    theta_star, phi_star = sample_ground_truth(rng_gt, world_cfg, gt_cfg)

    topology = generate_topology(rng, world_cfg)
    pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
    pair_min_dist = compute_all_pair_min_distances(pathset, topology.positions)

    m, Z = world_cfg.m, world_cfg.Z

    # ── Initialize estimates (random, away from truth) ────────────────────────
    theta_hat = rng.uniform(0.05, 0.45, size=m)
    phi_hat = rng.uniform(0.05, 0.45, size=Z)

    # ── Tracking arrays ─────────────────────────────────────────────────────
    gaps = []
    theta_errors = []
    phi_errors = []
    loss_oracle_list = []
    loss_qamab_list = []

    print("=" * 70)
    print("QUBO Solver Accuracy Experiment")
    print(f"P={P} iterations, T={T} steps/batch, seed={seed}")
    print(f"Ground truth: {gt_cfg.n_faulty_uavs} faulty UAVs, {gt_cfg.n_faulty_zones} faulty zones")
    print(f"Topology: m={world_cfg.m}, Z={world_cfg.Z}, "
          f"N_flows={world_cfg.N_flows}, K={world_cfg.K_paths}")
    print("=" * 70)

    for p in range(P):
        visit_counts = np.ones((world_cfg.N_flows, world_cfg.K_paths), dtype=int)

        # ── Oracle: brute-force with TRUE params ───────────────────────────
        oracle_paths, _ = brute_force_solve(
            pathset, theta_star, phi_star,
            topology.positions, gt_cfg,
            visit_counts=None, ucb_c=0.0
        )
        oracle_loss = compute_expected_loss_no_noise(
            oracle_paths, pathset, theta_star, phi_star, topology.positions, gt_cfg
        )

        # ── QA-MAB: brute-force with CURRENT estimates ────────────────────
        qamab_paths, _ = brute_force_solve(
            pathset, theta_hat, phi_hat,
            topology.positions, gt_cfg,
            visit_counts=visit_counts, ucb_c=float(qamab_cfg.ucb_c)
        )
        qamab_loss = compute_expected_loss_no_noise(
            qamab_paths, pathset, theta_star, phi_star, topology.positions, gt_cfg
        )

        gap = qamab_loss - oracle_loss
        theta_err = float(np.abs(theta_hat - theta_star).mean())
        phi_err = float(np.abs(phi_hat - phi_star).mean())

        gaps.append(gap)
        theta_errors.append(theta_err)
        phi_errors.append(phi_err)
        loss_oracle_list.append(oracle_loss)
        loss_qamab_list.append(qamab_loss)

        print(f"  Iter {p+1:2d}/{P}: "
              f"gap={gap:+.4f}  "
              f"θ_err={theta_err:.4f}  φ_err={phi_err:.4f}  "
              f"ORA={oracle_loss:.4f}  QAM={qamab_loss:.4f}  "
              f"ORA={oracle_paths.tolist()}  QAM={qamab_paths.tolist()}")

        # ── Learn: T steps against TRUE environment, batch-update ─────────
        for t in range(T):
            Q = build_qubo(
                theta_hat, phi_hat, pathset, pair_min_dist,
                qamab_cfg, gt_cfg,
                visit_counts=visit_counts,
                ucb_c=float(qamab_cfg.ucb_c),
            )

            rng_sa = np.random.default_rng(seed + p * 1000 + t)
            best_x, _ = sa_solve(
                Q, rng_sa,
                n_reads=5, n_sweeps=50,
                T_init=qamab_cfg.sa_T_init,
                T_final=qamab_cfg.sa_T_final,
            )
            chosen_paths = decode_solution(best_x, pathset.N, pathset.K)

            # Observe losses from TRUE environment (no noise)
            N = pathset.N
            losses = np.zeros(N)
            for n in range(N):
                path_n = pathset.paths_per_flow[n][chosen_paths[n]]
                fault = sum(theta_star[i] for i in path_n)
                zone_phi = sum(
                    phi_star[topology.zone_of[path_n[-1]]]
                    for _ in path_n
                ) / len(path_n) * len(path_n)  # simplified
                # Actually compute zone membership properly
                zone_set = set(topology.zone_of[i] for i in path_n)
                fault = sum(theta_star[i] for i in path_n)
                zone_fault = sum(phi_star[topology.zone_of[i]] for i in path_n)
                fault = float(fault) + float(zone_fault)

                # Collision
                shared = 0
                for l in range(N):
                    if l == n:
                        continue
                    path_l = pathset.paths_per_flow[l][chosen_paths[l]]
                    if any(i in path_l for i in path_n):
                        shared += 1
                collision_cost = gt_cfg.C_coll * shared

                # Proximity
                prox = 0.0
                for l in range(N):
                    if l == n:
                        continue
                    path_l = pathset.paths_per_flow[l][chosen_paths[l]]
                    d = path_pair_min_distance(path_n, path_l, topology.positions)
                    prox += np.exp(-d / gt_cfg.d0)

                losses[n] = fault + collision_cost + prox

            # Update visit counts
            for n in range(N):
                visit_counts[n, chosen_paths[n]] += 1

            # Batch update estimates
            residual_credit_update(
                pathset, losses, chosen_paths,
                theta_hat, phi_hat,
                qamab_cfg.alpha, gt_cfg, topology.positions
            )

        # Clip estimates
        theta_hat = np.clip(theta_hat, 0.0, 1.0)
        phi_hat = np.clip(phi_hat, 0.0, 1.0)

    return {
        'gaps': np.array(gaps),
        'theta_errors': np.array(theta_errors),
        'phi_errors': np.array(phi_errors),
        'loss_oracle': np.array(loss_oracle_list),
        'loss_qamab': np.array(loss_qamab_list),
        'theta_star': theta_star,
        'phi_star': phi_star,
    }


def main():
    results_dir = '/Users/jon_claw/qa-mab-research/simulations/results/qubo_solver_accuracy'
    os.makedirs(results_dir, exist_ok=True)

    results = run_experiment(P=30, T=50, seed=42)
    gaps = results['gaps']

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean gap:               {gaps.mean():+.4f}")
    print(f"Std gap:                {gaps.std():.4f}")
    print(f"Final gap (last 5 avg): {gaps[-5:].mean():+.4f}")
    print(f"QA wins (gap<0):        {(gaps < 0).sum()}/{len(gaps)}")
    print(f"Mean θ̂ error (final):  {results['theta_errors'][-1]:.4f}")
    print(f"Mean φ̂ error (final):  {results['phi_errors'][-1]:.4f}")
    corr = np.corrcoef(results['theta_errors'], gaps)[0, 1]
    print(f"Corr(θ_err, gap):      {corr:.4f}")
    final_improvement = gaps[0] - gaps[-1]
    print(f"Gap improvement:        {final_improvement:+.4f}  (iter1={gaps[0]:+.4f}, iter30={gaps[-1]:+.4f})")

    # Save CSV
    csv_path = os.path.join(results_dir, 'qubo_solver_accuracy.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'gap', 'theta_error', 'phi_error',
                         'loss_oracle', 'loss_qamab'])
        for i in range(len(gaps)):
            writer.writerow([
                i + 1,
                f"{gaps[i]:.6f}",
                f"{results['theta_errors'][i]:.6f}",
                f"{results['phi_errors'][i]:.6f}",
                f"{results['loss_oracle'][i]:.6f}",
                f"{results['loss_qamab'][i]:.6f}",
            ])
    print(f"\nCSV: {csv_path}")

    # Save pickle
    pkl_path = os.path.join(results_dir, 'qubo_solver_accuracy.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"PKL: {pkl_path}")


if __name__ == '__main__':
    main()
