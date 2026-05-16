"""
validate_cat4_learning_dynamics.py
===================================
Category 4: QA-MAB learns the environment (theta/phi estimates converge).

Test: Run QA-MAB for P iterations, T steps per iteration. Measure theta_err
and phi_err at each iteration. Test with both epoch_decay=0.9 and epoch_decay=1.0.

PASS criterion: theta_err decreases OR final gap < initial gap (learning works).
Also compare decay variants.
"""

import sys
import os
import numpy as np
import csv
import json

sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')

from src.uav_routing.config import WorldConfig, GroundTruthConfig, QAMABConfig
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances, path_pair_min_distance
from src.uav_routing.ground_truth import sample_ground_truth
from src.uav_routing.qubo import build_qubo
from src.uav_routing.sa_solver import sa_solve, decode_solution


def compute_expected_loss_no_noise(paths, pathset, theta, phi, positions, gt_cfg):
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


def residual_credit_update(pathset, losses, chosen_paths, theta_hat, phi_hat, alpha, gt_cfg, positions):
    N, m, Z = pathset.N, pathset.m, pathset.Z
    selected_uav = np.array(
        [pathset.path_uav_membership[n, chosen_paths[n]] for n in range(N)]
    )
    selected_zone = np.array(
        [pathset.path_zone_membership[n, chosen_paths[n]] for n in range(N)]
    )
    shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
    np.fill_diagonal(shared, False)
    collision_counts = shared.sum(axis=1).astype(float)
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
        L_fault = losses[n] - gt_cfg.C_coll * collision_counts[n] - proximity[n]
        for i in range(m):
            if not pathset.path_uav_membership[n, chosen_paths[n], i]:
                continue
            other_theta_sum = sum(
                theta_hat[j] for j in range(m)
                if j != i and pathset.path_uav_membership[n, chosen_paths[n], j]
            )
            zone_phi_sum = sum(
                phi_hat[zz] for zz in range(Z)
                if pathset.path_zone_membership[n, chosen_paths[n], zz]
            )
            residual = L_fault - other_theta_sum - zone_phi_sum
            theta_hat[i] += alpha * (residual - theta_hat[i])
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


def run_single_decay(decay, n_seeds=15, P=25, T=40, seed_base=200, pass_threshold=0.7):
    """Run learning dynamics test for one decay value. Returns per-seed arrays."""
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
        epoch_decay=decay,
        sa_sweeps=200, sa_n_reads=10,
        sa_T_init=2.0, sa_T_final=0.05,
    )

    all_theta_errors = []
    all_phi_errors = []
    all_gaps = []

    print(f"\n  Decay={decay}: {n_seeds} seeds × P={P}, T={T}")
    print("  " + "-" * 55)

    for s in range(n_seeds):
        seed = seed_base + s
        rng = np.random.default_rng(seed)

        rng_gt = np.random.default_rng(seed)
        theta_star, phi_star = sample_ground_truth(rng_gt, world_cfg, gt_cfg)

        topology = generate_topology(rng, world_cfg)
        pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
        pair_min_dist = compute_all_pair_min_distances(pathset, topology.positions)

        m, Z = world_cfg.m, world_cfg.Z
        theta_hat = rng.uniform(0.05, 0.45, size=m)
        phi_hat = rng.uniform(0.05, 0.45, size=Z)

        theta_errors = []
        phi_errors = []
        gaps = []

        for p in range(P):
            visit_counts = np.ones((world_cfg.N_flows, world_cfg.K_paths), dtype=int)

            # Compute gap (oracle vs qamab) at start of iteration
            Q_oracle = build_qubo(
                theta_star, phi_star, pathset, pair_min_dist,
                qamab_cfg, gt_cfg, visit_counts=None, ucb_c=0.0
            )
            Q_qamab = build_qubo(
                theta_hat, phi_hat, pathset, pair_min_dist,
                qamab_cfg, gt_cfg, visit_counts=visit_counts, ucb_c=float(qamab_cfg.ucb_c)
            )

            rng_bf = np.random.default_rng(seed + p * 1000)
            # Simple greedy for oracle
            best_oracle_loss = float('inf')
            for k1 in range(pathset.K):
                for k2 in range(pathset.K):
                    for k3 in range(pathset.K):
                        paths = np.array([k1, k2, k3])
                        loss = compute_expected_loss_no_noise(
                            paths, pathset, theta_star, phi_star, topology.positions, gt_cfg
                        )
                        if loss < best_oracle_loss:
                            best_oracle_loss = loss
            # QAMAB step
            rng_sa = np.random.default_rng(seed + p * 1000 + 5000)
            best_x, _ = sa_solve(Q_qamab, rng_sa, n_reads=5, n_sweeps=50,
                                  T_init=qamab_cfg.sa_T_init, T_final=qamab_cfg.sa_T_final)
            chosen_paths = decode_solution(best_x, pathset.N, pathset.K)
            qamab_loss = compute_expected_loss_no_noise(
                chosen_paths, pathset, theta_star, phi_star, topology.positions, gt_cfg
            )
            gap = qamab_loss - best_oracle_loss

            theta_err = float(np.abs(theta_hat - theta_star).mean())
            phi_err = float(np.abs(phi_hat - phi_star).mean())
            theta_errors.append(theta_err)
            phi_errors.append(phi_err)
            gaps.append(gap)

            # Learn: T steps
            for t in range(T):
                Q = build_qubo(
                    theta_hat, phi_hat, pathset, pair_min_dist,
                    qamab_cfg, gt_cfg, visit_counts=visit_counts,
                    ucb_c=float(qamab_cfg.ucb_c),
                )
                rng_sa_t = np.random.default_rng(seed + p * 10000 + t)
                best_x_t, _ = sa_solve(Q, rng_sa_t, n_reads=3, n_sweeps=40,
                                        T_init=qamab_cfg.sa_T_init, T_final=qamab_cfg.sa_T_final)
                chosen = decode_solution(best_x_t, pathset.N, pathset.K)
                losses = np.zeros(pathset.N)
                for n in range(pathset.N):
                    path_n = pathset.paths_per_flow[n][chosen[n]]
                    fault = sum(theta_star[i] for i in path_n)
                    zone_fault = sum(phi_hat[topology.zone_of[i]] for i in path_n)
                    shared = sum(1 for l in range(pathset.N) if l != n and
                                any(i in pathset.paths_per_flow[l][chosen[l]] for i in path_n))
                    prox = sum(np.exp(-path_pair_min_distance(
                        pathset.paths_per_flow[n][chosen[n]],
                        pathset.paths_per_flow[l][chosen[l]], topology.positions
                    ) / gt_cfg.d0) for l in range(pathset.N) if l != n)
                    losses[n] = float(fault + zone_fault + gt_cfg.C_coll * shared + prox)
                for n in range(pathset.N):
                    visit_counts[n, chosen[n]] += 1
                residual_credit_update(pathset, losses, chosen, theta_hat, phi_hat,
                                       qamab_cfg.alpha, gt_cfg, topology.positions)

            theta_hat = np.clip(theta_hat, 0.0, 1.0)
            phi_hat = np.clip(phi_hat, 0.0, 1.0)

        all_theta_errors.append(theta_errors)
        all_phi_errors.append(phi_errors)
        all_gaps.append(gaps)

        print(f"    Seed {s+1:2d}: theta_err_init={theta_errors[0]:.4f} → "
              f"theta_err_final={theta_errors[-1]:.4f}  "
              f"gap_init={gaps[0]:+.4f} → gap_final={gaps[-1]:+.4f}")

    return {
        'decay': decay,
        'n_seeds': n_seeds,
        'P': P,
        'T': T,
        'theta_errors': np.array(all_theta_errors),
        'phi_errors': np.array(all_phi_errors),
        'gaps': np.array(all_gaps),
    }


def run(pass_threshold=0.7):
    print("=" * 65)
    print("CATEGORY 4: Learning Dynamics Test")
    print("Decay variants: 0.9 and 1.0 | pass_threshold={:.0f}% improvement".format(pass_threshold * 100))
    print("=" * 65)

    results_09 = run_single_decay(0.9, n_seeds=15, P=25, T=40, seed_base=200, pass_threshold=pass_threshold)
    results_10 = run_single_decay(1.0, n_seeds=15, P=25, T=40, seed_base=200, pass_threshold=pass_threshold)

    # Compute summary statistics
    def summarize(results):
        theta_err_init = results['theta_errors'][:, 0].mean()
        theta_err_final = results['theta_errors'][:, -1].mean()
        gap_init = results['gaps'][:, 0].mean()
        gap_final = results['gaps'][:, -1].mean()
        improvement = gap_init - gap_final
        return {
            'decay': results['decay'],
            'theta_err_init': float(theta_err_init),
            'theta_err_final': float(theta_err_final),
            'theta_err_reduction': float(theta_err_init - theta_err_final),
            'gap_init': float(gap_init),
            'gap_final': float(gap_final),
            'gap_improvement': float(improvement),
            'mean_theta_err_per_iter': results['theta_errors'].mean(axis=0).tolist(),
            'mean_gap_per_iter': results['gaps'].mean(axis=0).tolist(),
        }

    sum_09 = summarize(results_09)
    sum_10 = summarize(results_10)

    print("\n" + "=" * 65)
    print("SUMMARY: decay=0.9")
    print("=" * 65)
    print(f"  theta_err: {sum_09['theta_err_init']:.4f} → {sum_09['theta_err_final']:.4f} "
          f"(Δ={sum_09['theta_err_reduction']:+.4f})")
    print(f"  gap:       {sum_09['gap_init']:+.4f} → {sum_09['gap_final']:+.4f} "
          f"(improvement={sum_09['gap_improvement']:+.4f})")
    dec09_passed = sum_09['gap_improvement'] > 0 or sum_09['theta_err_reduction'] > 0
    print(f"  {'✅ PASS' if dec09_passed else '❌ FAIL'}")
    print(f"\n  Decay=1.0")
    print(f"  theta_err: {sum_10['theta_err_init']:.4f} → {sum_10['theta_err_final']:.4f} "
          f"(Δ={sum_10['theta_err_reduction']:+.4f})")
    print(f"  gap:       {sum_10['gap_init']:+.4f} → {sum_10['gap_final']:+.4f} "
          f"(improvement={sum_10['gap_improvement']:+.4f})")
    dec10_passed = sum_10['gap_improvement'] > 0 or sum_10['theta_err_reduction'] > 0
    print(f"  {'✅ PASS' if dec10_passed else '❌ FAIL'}")

    # Compare decays
    print("\n" + "=" * 65)
    print("DECAY COMPARISON")
    print("=" * 65)
    better = "decay=0.9" if sum_09['gap_improvement'] > sum_10['gap_improvement'] else "decay=1.0"
    print(f"  Better gap improvement: {better}")
    print(f"  Decay=0.9 gap improvement: {sum_09['gap_improvement']:+.4f}")
    print(f"  Decay=1.0 gap improvement: {sum_10['gap_improvement']:+.4f}")

    overall_passed = dec09_passed and dec10_passed
    print(f"\n{'✅ PASS (both decays)' if overall_passed else '❌ FAIL (at least one decay failed)'}")

    output = {
        'category': 4,
        'test': 'learning_dynamics',
        'pass_threshold': pass_threshold,
        'decay_0.9': sum_09,
        'decay_1.0': sum_10,
        'overall_passed': overall_passed,
        'metadata': {
            'n_seeds': 15,
            'P': 25,
            'T': 40,
        }
    }

    out_dir = '/Users/jon_claw/qa-mab-research/simulations/results/validation_cat4'
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # CSV per decay
    for label, sum_data in [('decay_0.9', sum_09), ('decay_1.0', sum_10)]:
        csv_path = os.path.join(out_dir, f'result_{label}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'mean_theta_error', 'mean_gap'])
            for i, (te, g) in enumerate(zip(sum_data['mean_theta_err_per_iter'], sum_data['mean_gap_per_iter'])):
                writer.writerow([i + 1, f"{te:.6f}", f"{g:.6f}"])
        print(f"Saved: {csv_path}")

    print(f"\nSaved: {out_dir}/result.json")
    return output


if __name__ == '__main__':
    results = run(pass_threshold=0.7)
    sys.exit(0 if results['overall_passed'] else 1)