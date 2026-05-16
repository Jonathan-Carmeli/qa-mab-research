"""Optimal agent — exhaustive enumeration of all K^N path combos."""
import numpy as np
from ..qa_mab_physical import QAMABPhysical


class OptimalAgent(QAMABPhysical):
    name = "Optimal-Physical"

    def __init__(self, world, C_coll=5.0, d0=150.0, sigma_noise=0.05, seed=42):
        super().__init__(
            world, C_coll, d0, sigma_noise,
            alpha=0.0, ucb_c=0.0, epoch_decay=1.0, seed=seed,
        )
        self.theta_hat = world.theta_star.copy()
        self.phi_hat = world.phi_star.copy()
        self._rng = np.random.default_rng(seed)

    def act(self, t: int, p: int) -> np.ndarray:
        N, K = self.world.N, self.world.K
        ps = self.world.pathset
        M = N * K

        best_loss = float('inf')
        best_paths = np.zeros(N, dtype=int)

        def enumerate_paths(idx, current):
            nonlocal best_loss, best_paths
            if idx == N:
                losses = self._compute_losses_no_noise(current)
                total = losses.sum()
                if total < best_loss:
                    best_loss = total
                    best_paths = current.copy()
                return
            for k in range(K):
                current[idx] = k
                enumerate_paths(idx + 1, current)

        enumerate_paths(0, np.zeros(N, dtype=int))
        return best_paths

    def _compute_losses_no_noise(self, chosen_paths: np.ndarray) -> np.ndarray:
        N, K = self.world.N, self.world.K
        ps = self.world.pathset
        losses = np.zeros(N, dtype=float)

        selected_uav = np.array(
            [ps.path_uav_membership[n, chosen_paths[n]] for n in range(N)]
        )
        shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
        np.fill_diagonal(shared, False)
        collision_counts = shared.sum(axis=1).astype(float)

        for n in range(N):
            k = chosen_paths[n]
            uav_mask = ps.path_uav_membership[n, k]
            zone_mask = ps.path_zone_membership[n, k]
            losses[n] = (
                float(self.theta_hat[uav_mask].sum())
                + float(self.phi_hat[zone_mask].sum())
                + self.C_coll * collision_counts[n]
            )
            a = n * K + k
            for l in range(N):
                if l == n:
                    continue
                kl = chosen_paths[l]
                b = l * K + kl
                losses[n] += np.exp(-ps.pair_min_dist[a, b] / self.d0)

        return losses

    def update(self, chosen_paths, losses):
        pass