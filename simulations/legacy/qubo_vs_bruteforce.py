"""
qubo_vs_bruteforce.py
=====================
Does the QUBO matrix correctly encode the UAV routing optimization problem?

This is a FORMULATION TEST — no solver, no learning:
  - Fixed environment (θ*, φ*, topology) — TRUE params known to BOTH methods
  - Per iteration t (same seed, same env, just measuring):
      Method A (Oracle BF):  Enumerate all K^N combos → evaluate TRUE loss
                              → pick paths that MINIMIZE true loss
      Method B (QUBO BF):     Build QUBO Q with TRUE θ*, φ* (from build_qubo)
                              Enumerate all K^N combos → compute xᵀQx energy
                              → pick paths that MINIMIZE QUBO energy
  - gap[t] = true_loss(QUBO_BF_paths) - true_loss(Oracle_BF_paths)

  gap ≈ 0  ← QUBO and true loss agree on optimal routing
  gap > 0  ← QUBO encodes a DIFFERENT optimization problem
             (formulation error or missing terms)
"""

import sys
import os
import numpy as np
from itertools import product
import csv
import pickle

sys.path.insert(0, '/Users/jon_claw/Thesis_brain/simulation')

from src.uav_routing.config import WorldConfig, GroundTruthConfig, QAMABConfig
from src.uav_routing.world import generate_topology
from src.uav_routing.paths import enumerate_paths, compute_all_pair_min_distances
from src.uav_routing.ground_truth import sample_ground_truth
from src.uav_routing.qubo import build_qubo


# ─── True expected loss (noiseless) ─────────────────────────────────────────────
# Exactly as used in oracle_solver_gap.py

def true_expected_loss(chosen_paths, pathset, theta_star, phi_star, pair_min_dist, gt_cfg):
    """Total expected loss (no noise) under TRUE parameters."""
    N = pathset.N
    K = pathset.K
    C_coll = gt_cfg.C_coll
    d0 = gt_cfg.d0

    selected_uav = np.array(
        [pathset.path_uav_membership[n, chosen_paths[n]] for n in range(N)]
    )
    selected_zone = np.array(
        [pathset.path_zone_membership[n, chosen_paths[n]] for n in range(N)]
    )

    # UAV fault + zone interference
    loss = float((selected_uav.astype(float) @ theta_star).sum()) + \
           float((selected_zone.astype(float) @ phi_star).sum())

    # Collision penalty
    shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
    np.fill_diagonal(shared, False)
    loss += C_coll * shared.sum()

    # Proximity interference
    for n in range(N):
        kn = chosen_paths[n]
        for l in range(N):
            if l == n:
                continue
            kl = chosen_paths[l]
            d = pair_min_dist[n * K + kn, l * K + kl]
            loss += np.exp(-d / d0)

    return loss


# ─── QUBO energy ───────────────────────────────────────────────────────────────

def qubo_energy(paths_vec, Q):
    """E(x) = x^T Q x for binary vector paths_vec."""
    return float(paths_vec.T @ Q @ paths_vec)


def paths_to_vec(paths, N, K):
    """Convert (N,) path indices to (N*K,) binary vector."""
    x = np.zeros(N * K, dtype=np.float64)
    for n in range(N):
        x[n * K + paths[n]] = 1.0
    return x


# ─── Brute-force solvers ──────────────────────────────────────────────────────

def brute_oracle_solve(pathset, theta, phi, pair_min_dist, gt_cfg):
    """Enumerate K^N combos, pick minimum TRUE loss."""
    N, K = pathset.N, pathset.K
    best_loss = float('inf')
    best_paths = None

    for combo in product(range(K), repeat=N):
        paths = np.array(combo, dtype=int)
        loss = true_expected_loss(paths, pathset, theta, phi, pair_min_dist, gt_cfg)
        if loss < best_loss:
            best_loss = loss
            best_paths = paths.copy()

    return best_paths, best_loss


def brute_qubo_solve(Q, N, K):
    """Enumerate K^N combos, pick minimum QUBO energy x^T Q x."""
    best_energy = float('inf')
    best_vec = None

    for combo in product(range(K), repeat=N):
        paths = np.array(combo, dtype=int)
        x = paths_to_vec(paths, N, K)
        energy = qubo_energy(x, Q)
        if energy < best_energy:
            best_energy = energy
            best_vec = x.copy()

    # Convert back to path indices
    best_paths = np.zeros(N, dtype=int)
    for n in range(N):
        offset = n * K
        for k in range(K):
            if best_vec[offset + k] > 0.5:
                best_paths[n] = k
                break
    return best_paths, best_energy


# ─── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(n_seeds=20, P=50, seed_base=42):
    """Run QUBO formulation test. Returns dict with results arrays."""

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

    print("=" * 65)
    print("QUBO Formulation Accuracy Test")
    print("Question: Does minimizing QUBO energy find the same routing as minimizing true loss?")
    print(f"Design: {n_seeds} seeds × {P} iters  |  both solvers = brute-force")
    print("  Method A (Oracle BF): argmin true_loss  — the ground truth optimum")
    print("  Method B (QUBO BF):   argmin x^T Q x   — built with same TRUE params")
    print("  gap = true_loss(QUBO_opt) - true_loss(Oracle_opt)  [should ≈ 0 if QUBO is correct]")
    print("=" * 65)

    all_gaps = []
    all_matches = []
    all_oracle_loss = []
    all_qubo_loss = []
    all_qubo_energy = []

    for s in range(n_seeds):
        seed = seed_base + s
        rng = np.random.default_rng(seed)

        # ── Generate fixed environment (same for both methods) ─────────────
        rng_gt = np.random.default_rng(seed)
        theta_star, phi_star = sample_ground_truth(rng_gt, world_cfg, gt_cfg)

        rng_topo = np.random.default_rng(seed + 1000)
        topology = generate_topology(rng_topo, world_cfg)
        pathset = enumerate_paths(topology, world_cfg.K_paths, world_cfg.Z)
        pair_min_dist = compute_all_pair_min_distances(pathset, topology.positions)

        # ── Build QUBO matrix with TRUE params (one-time per seed) ────────────
        Q = build_qubo(
            theta_star, phi_star, pathset, pair_min_dist,
            qamab_cfg, gt_cfg,
            visit_counts=None,
            ucb_c=0.0,  # No UCB — we want pure QUBO quality
        )

        N, K = pathset.N, pathset.K
        M = N * K

        # ── Pre-compute Oracle optimal ONCE (true loss BF — it's deterministic) ─
        opt_paths, opt_loss = brute_oracle_solve(
            pathset, theta_star, phi_star, pair_min_dist, gt_cfg
        )
        # Also pre-compute QUBO optimal once
        qubo_opt_paths, qubo_opt_energy = brute_qubo_solve(Q, N, K)

        gaps = []
        matches = []
        oracle_losses = []
        qubo_losses = []
        qubo_energies = []

        for p in range(P):
            # ── Oracle: brute-force on TRUE loss ─────────────────────────
            # (same every iteration — deterministic, but we record per-iter for consistency)
            oracle_loss = opt_loss  # same every time since env is fixed

            # ── QUBO: brute-force on QUBO energy ─────────────────────────
            # (same every iteration since Q is fixed, but record per-iter)
            qubo_loss = true_expected_loss(
                qubo_opt_paths, pathset, theta_star, phi_star, pair_min_dist, gt_cfg
            )
            qubo_energy = qubo_opt_energy

            gap = qubo_loss - oracle_loss
            match = np.array_equal(qubo_opt_paths, opt_paths)

            gaps.append(gap)
            matches.append(match)
            oracle_losses.append(oracle_loss)
            qubo_losses.append(qubo_loss)
            qubo_energies.append(qubo_energy)

        match_rate = np.mean(matches)
        mean_gap = np.mean(gaps)
        print(f"  Seed {s+1:2d}/{n_seeds} (seed={seed}): "
              f"match={match_rate*100:.0f}%  mean_gap={mean_gap:+.6f}  "
              f"oracle_loss={np.mean(oracle_losses):.4f}")

        all_gaps.append(gaps)
        all_matches.append(matches)
        all_oracle_loss.append(oracle_losses)
        all_qubo_loss.append(qubo_losses)
        all_qubo_energy.append(qubo_energies)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    gaps_mat   = np.array(all_gaps)
    match_mat  = np.array(all_matches)
    oracle_mat = np.array(all_oracle_loss)
    qubo_mat   = np.array(all_qubo_loss)
    qubo_en_mat = np.array(all_qubo_energy)

    mean_gaps   = gaps_mat.mean(axis=0)
    std_gaps    = gaps_mat.std(axis=0)
    mean_matches = match_mat.mean(axis=0)

    print("\n" + "=" * 65)
    print("PER-ITERATION SUMMARY")
    print("=" * 65)
    print(f"{'Iter':>4} | {'Mean Gap':>10} | {'Std Gap':>8} | {'Match%':>7} | {'Oracle Loss':>12}")
    print("-" * 65)
    for i in list(range(0, P, 5)):
        print(f"{i+1:4d} | {mean_gaps[i]:+10.6f} | {std_gaps[i]:8.6f} | "
              f"{mean_matches[i]*100:6.1f}% | {oracle_mat[:,i].mean():12.4f}")

    print(f"\n{'avg-L5':>4} | {mean_gaps[-5:].mean():+10.6f} | {std_gaps[-5:].mean():8.6f} | "
          f"{mean_matches[-5:].mean()*100:6.1f}% | {oracle_mat[:,-5:].mean():12.4f}")

    print("\n" + "=" * 65)
    print("OVERALL STATS")
    print("=" * 65)
    n_total = n_seeds * P
    pct_nonzero = (np.abs(gaps_mat) > 1e-9).mean() * 100
    print(f"Total observations:             {n_total}")
    print(f"Match rate:                     {match_mat.mean()*100:.1f}%")
    print(f"Mean gap:                       {gaps_mat.mean():+.6f}")
    print(f"Std gap:                        {gaps_mat.std():.6f}")
    print(f"Gap == 0 (exact same optimal):  {(np.abs(gaps_mat) < 1e-9).sum()}/{n_total} "
          f"({(np.abs(gaps_mat)<1e-9).mean()*100:.1f}%)")
    print(f"Gap > 0 (QUBO suboptimal):      {(gaps_mat > 1e-9).sum()}/{n_total} "
          f"({(gaps_mat>1e-9).mean()*100:.1f}%)")
    print(f"Max gap observed:               {gaps_mat.max():.6f}")

    # Breakdown: how many seeds have gap > 0?
    seeds_with_gap = (np.abs(gaps_mat) > 1e-9).any(axis=1).sum()
    print(f"Seeds where QUBO ≠ Oracle:     {seeds_with_gap}/{n_seeds} "
          f"({seeds_with_gap/n_seeds*100:.0f}%)")

    return {
        'gaps_mat': gaps_mat,
        'match_mat': match_mat,
        'oracle_mat': oracle_mat,
        'qubo_mat': qubo_mat,
        'qubo_en_mat': qubo_en_mat,
        'mean_gaps': mean_gaps,
        'std_gaps': std_gaps,
        'mean_matches': mean_matches,
        'n_seeds': n_seeds,
        'P': P,
    }


def main():
    results_dir = '/Users/jon_claw/qa-mab-research/simulations/results/qubo_vs_bruteforce'
    os.makedirs(results_dir, exist_ok=True)

    results = run_experiment(n_seeds=20, P=50, seed_base=42)

    # Save CSV
    csv_path = os.path.join(results_dir, 'qubo_vs_bruteforce.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'mean_gap', 'std_gap', 'match_rate',
                         'mean_oracle_loss', 'mean_qubo_loss'])
        for i in range(results['P']):
            writer.writerow([
                i + 1,
                f"{results['mean_gaps'][i]:.8f}",
                f"{results['std_gaps'][i]:.8f}",
                f"{results['mean_matches'][i]:.4f}",
                f"{results['oracle_mat'][:,i].mean():.6f}",
                f"{results['qubo_mat'][:,i].mean():.6f}",
            ])
    print(f"\nCSV: {csv_path}")

    # Save PKL
    pkl_path = os.path.join(results_dir, 'qubo_vs_bruteforce.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"PKL: {pkl_path}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    match_rate = results['match_mat'].mean()
    mean_gap = results['gaps_mat'].mean()

    print("\n" + "=" * 65)
    print("CONCLUSION")
    print("=" * 65)
    if match_rate >= 0.99 and abs(mean_gap) < 1e-6:
        print("✅ QUBO formulation is PERFECT.")
        print("   gap ≈ 0, match ≈ 100%. The QUBO correctly encodes true loss.")
        print("   → The solver gap (SA vs BF) is purely solver quality.")
    elif match_rate >= 0.95:
        print("⚠️  QUBO formulation is mostly correct.")
        print(f"   Match: {match_rate*100:.1f}%, mean gap: {mean_gap:+.6f}")
        print("   Minor discrepancies due to degenerate optima.")
    elif match_rate >= 0.80:
        print("⚠️  QUBO formulation has gaps.")
        print(f"   Match: {match_rate*100:.1f}%, mean gap: {mean_gap:+.6f}")
    else:
        print("❌ QUBO formulation diverges from true loss.")
        print(f"   Match: {match_rate*100:.1f}%, mean gap: {mean_gap:+.6f}")
        print("   The QUBO is optimizing a different objective than true loss.")
        print("   Likely: collision/proximity weights in QUBO don't match gt_cfg values.")


if __name__ == '__main__':
    main()