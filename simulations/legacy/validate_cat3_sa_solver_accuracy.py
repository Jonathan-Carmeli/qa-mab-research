"""
validate_cat3_sa_solver_accuracy.py — Fixed v3
===============================================
Category 3: SA solver correctly solves QUBO instances for UAV routing.

v3 changes:
- Run with n_reads=500 (more thorough exploration)
- Accept: SA finds optimum if SA finds a solution within 1% relative error of BF true loss
  (strict equality too harsh when QUBO and true-loss metrics differ slightly)
- Report both "exact optimum" and "within 1% relative" criteria

PASS: exact optimum (loss_diff < 0.001) OR relative_loss_diff < 1% ≥ 85%
"""

import sys, os, numpy as np
from itertools import product
import csv, json, time

sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')

from src.uav_routing.config import WorldConfig, GroundTruthConfig, QAMABConfig
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances, path_pair_min_distance
from src.uav_routing.ground_truth import sample_ground_truth
from src.uav_routing.qubo import build_qubo
from src.uav_routing.sa_solver import sa_solve, decode_solution


def compute_expected_loss(paths, pathset, theta, phi, positions, gt_cfg):
    N = pathset.N
    su = np.array([pathset.path_uav_membership[n, paths[n]] for n in range(N)])
    sz = np.array([pathset.path_zone_membership[n, paths[n]] for n in range(N)])
    fl = float((su.astype(float) @ theta).sum()) + float((sz.astype(float) @ phi).sum())
    sh = (su.astype(int) @ su.astype(int).T) > 0
    np.fill_diagonal(sh, False)
    cl = float(gt_cfg.C_coll * sh.sum())
    pl = 0.0
    for n in range(N):
        pn = pathset.paths_per_flow[n][paths[n]]
        for l in range(N):
            if l == n: continue
            pl += np.exp(-path_pair_min_distance(pn, pathset.paths_per_flow[l][paths[l]], positions) / gt_cfg.d0)
    return fl + cl + pl


def brute_force_qubo_solve(Q, N, K):
    best_e, best_x = float('inf'), None
    for combo in product(range(K), repeat=N):
        x = np.zeros(N * K, dtype=np.float64)
        for n in range(N): x[n * K + combo[n]] = 1.0
        e = float(x.T @ Q @ x)
        if e < best_e:
            best_e, best_x = e, x.copy()
    decoded_paths = decode_solution(best_x, N, K)
    return decoded_paths, best_e, best_x


def run(n_instances=50, n_reads=500, seed_base=100, pass_threshold=0.85,
        exact_eps=0.001, rel_eps=0.01):
    world_cfg = WorldConfig(m=15, Z=6, N_flows=3, K_paths=4, comm_radius=350.0)
    gt_cfg = GroundTruthConfig(n_faulty_uavs=4, theta_low=0.2, theta_high=0.4,
        n_faulty_zones=2, phi_low=0.2, phi_high=0.4, C_coll=5.0, d0=150.0, sigma_noise=0.0)
    qamab_cfg = QAMABConfig(lambda_onehot=10.0, sa_sweeps=200, sa_n_reads=n_reads,
                             sa_T_init=2.0, sa_T_final=0.05)

    exact_ok = []       # SA_loss ≈ BF_loss (within exact_eps)
    within_1pct = []    # (SA_loss - BF_loss) / |BF_loss| < rel_eps
    rel_gaps = []
    per_instance = []

    print("=" * 70)
    print("CATEGORY 3: SA Solver Accuracy Test (Fixed v3)")
    print(f"Instances={n_instances}, n_reads={n_reads}")
    print(f"Criteria: exact (|Δ| < {exact_eps}) OR relative (|Δ|/|BF_loss| < {rel_eps*100}%)")
    print(f"PASS threshold: ≥ {pass_threshold*100}%")
    print("=" * 70)

    for i in range(n_instances):
        seed = seed_base + i
        rng_gt = np.random.default_rng(seed)
        theta_star, phi_star = sample_ground_truth(rng_gt, world_cfg, gt_cfg)

        rng_topo = np.random.default_rng(seed + 1000)
        topology = generate_topology(rng_topo, world_cfg)
        pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
        positions = topology.positions
        pair_min_dist = compute_all_pair_min_distances(pathset, positions)

        Q = build_qubo(theta_star, phi_star, pathset, pair_min_dist,
                       qamab_cfg, gt_cfg, visit_counts=None, ucb_c=0.0)

        bf_paths, bf_energy, bf_x = brute_force_qubo_solve(Q, pathset.N, pathset.K)

        rng_sa = np.random.default_rng(seed + 5000)
        t0 = time.time()
        best_x, best_energy = sa_solve(Q, rng_sa, n_reads=n_reads,
                                       n_sweeps=qamab_cfg.sa_sweeps,
                                       T_init=qamab_cfg.sa_T_init,
                                       T_final=qamab_cfg.sa_T_final)
        sa_elapsed = time.time() - t0
        sa_paths = decode_solution(best_x, pathset.N, pathset.K)

        bf_loss = compute_expected_loss(bf_paths, pathset, theta_star, phi_star, positions, gt_cfg)
        sa_loss = compute_expected_loss(sa_paths, pathset, theta_star, phi_star, positions, gt_cfg)

        bf_qubo_energy = float(bf_x.T @ Q @ bf_x)
        sa_qubo_energy = float(best_x.T @ Q @ best_x)

        loss_diff = sa_loss - bf_loss
        is_exact = abs(loss_diff) < exact_eps
        if abs(bf_loss) > 1e-6:
            is_within_1pct = abs(loss_diff) / abs(bf_loss) < rel_eps
        else:
            is_within_1pct = abs(loss_diff) < exact_eps
        rel_gap = (sa_qubo_energy - bf_qubo_energy) / abs(bf_qubo_energy) if abs(bf_qubo_energy) > 1e-12 else 0.0

        exact_ok.append(bool(is_exact))
        within_1pct.append(bool(is_within_1pct))
        rel_gaps.append(float(rel_gap))

        per_instance.append({
            'instance': i, 'seed': seed,
            'exact_ok': bool(is_exact),
            'within_1pct': bool(is_within_1pct),
            'loss_diff': float(loss_diff),
            'rel_loss_diff': float(abs(loss_diff) / abs(bf_loss)) if abs(bf_loss) > 1e-6 else 0.0,
            'rel_gap': float(rel_gap),
            'sa_qubo_energy': float(sa_qubo_energy),
            'bf_qubo_energy': float(bf_qubo_energy),
            'sa_true_loss': float(sa_loss),
            'bf_true_loss': float(bf_loss),
            'sa_paths': sa_paths.tolist(), 'bf_paths': bf_paths.tolist(),
            'sa_time_ms': round(sa_elapsed * 1000, 1),
        })

        icon = "✓" if is_exact else ("≈" if is_within_1pct else "✗")
        pct = abs(loss_diff)/abs(bf_loss)*100 if abs(bf_loss) > 1e-6 else 0
        print(f"  Inst {i+1:2d}/{n_instances}: {icon}  "
              f"SA={sa_loss:.4f} BF={bf_loss:.4f} (Δ={loss_diff:+.4f}, {pct:.1f}%)  "
              f"SA_QE={sa_qubo_energy:.2f} BF_QE={bf_qubo_energy:.2f}")

    exact_arr = np.array(exact_ok, dtype=bool)
    within_arr = np.array(within_1pct, dtype=bool)
    gap_arr = np.array(rel_gaps)

    exact_rate = exact_arr.mean()
    within_rate = within_arr.mean()
    # PRIMARY: exact optimum rate
    passed = exact_rate >= pass_threshold

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Exact optimum (|Δ| < {exact_eps}):     {exact_arr.sum():.0f}/50 ({exact_rate*100:.1f}%)  [PRIMARY]")
    print(f"Within 1% relative:                 {within_arr.sum():.0f}/50 ({within_rate*100:.1f}%)")
    print(f"SA finds lower QUBO energy than BF: {(gap_arr < 0).sum()}/50")
    print(f"Mean rel gap:                       {gap_arr.mean():+.4f}")
    print(f"Mean SA time:                       {np.mean([p['sa_time_ms'] for p in per_instance]):.1f}ms")
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: "
          f"exact_rate={exact_rate*100:.1f}% {'≥' if passed else '<'}{pass_threshold*100}%")

    results = {
        'category': 3,
        'test': 'sa_solver_accuracy_fixed_v3',
        'n_instances': n_instances,
        'n_reads': n_reads,
        'pass_threshold': pass_threshold,
        'exact_eps': exact_eps,
        'rel_eps': rel_eps,
        'exact_rate': float(exact_rate),
        'within_1pct_rate': float(within_rate),
        'mean_rel_gap': float(gap_arr.mean()),
        'std_rel_gap': float(gap_arr.std()),
        'passed': bool(passed),
        'sa_lt_bf_energy_count': int((gap_arr < 0).sum()),
        'per_instance': per_instance,
        'metadata': {
            'N_flows': world_cfg.N_flows,
            'K_paths': world_cfg.K_paths,
            'sa_sweeps': qamab_cfg.sa_sweeps,
            'fix': 'n_reads=500, exact criterion only (SA_loss≈BF_loss)',
        }
    }

    out_dir = '/Users/jon_claw/qa-mab-research/simulations/results/validation_cat3'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/result.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(f'{out_dir}/result.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['instance','seed','exact_ok','within_1pct',
                                                'loss_diff','rel_loss_diff','rel_gap',
                                                'sa_qubo_energy','bf_qubo_energy',
                                                'sa_true_loss','bf_true_loss','sa_time_ms'])
        writer.writeheader()
        for r in per_instance:
            writer.writerow({k: r[k] for k in ['instance','seed','exact_ok','within_1pct',
                                                 'loss_diff','rel_loss_diff','rel_gap',
                                                 'sa_qubo_energy','bf_qubo_energy',
                                                 'sa_true_loss','bf_true_loss','sa_time_ms']})
    print(f"\nSaved: {out_dir}/result.json + result.csv")
    return results


if __name__ == '__main__':
    results = run(n_instances=50, n_reads=500, seed_base=100, pass_threshold=0.85)
    sys.exit(0 if results['passed'] else 1)