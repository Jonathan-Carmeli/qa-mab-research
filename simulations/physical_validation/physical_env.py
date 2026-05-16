"""Physical Abstract Model environment for QA-MAB.

AbstractWorld samples hidden θ*/φ* once per seed and per-epoch randomises path
memberships and pairwise minimum distances. Loss model matches §2.2 of the
physical-model spec.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class AbstractPathSet:
    N: int
    K: int
    m: int
    Z: int
    path_uav_membership: np.ndarray   # (N, K, m) bool
    path_zone_membership: np.ndarray  # (N, K, Z) bool
    pair_min_dist: np.ndarray         # (N*K, N*K) symmetric float


class AbstractWorld:
    """Abstract environment with hidden per-UAV (θ*) and per-zone (φ*) faults.

    Path memberships and inter-path distances are random matrices refreshed
    each epoch; θ*, φ*, and uav_zone persist for the entire simulation.
    """

    def __init__(
        self,
        N: int = 3,
        K: int = 4,
        m: int = 15,
        Z: int = 6,
        n_faulty_uavs: int = 4,
        n_faulty_zones: int = 2,
        theta_low: float = 0.20,
        theta_high: float = 0.40,
        phi_low: float = 0.20,
        phi_high: float = 0.40,
        C_coll: float = 5.0,
        d0: float = 150.0,
        sigma_noise: float = 0.05,
        uavs_per_path_min: int = 2,
        uavs_per_path_max: int = 5,
        area_size: float = 1000.0,
        seed: int = 42,
    ):
        self.N = N
        self.K = K
        self.m = m
        self.Z = Z
        self.n_faulty_uavs = n_faulty_uavs
        self.n_faulty_zones = n_faulty_zones
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.phi_low = phi_low
        self.phi_high = phi_high
        self.C_coll = C_coll
        self.d0 = d0
        self.sigma_noise = sigma_noise
        self.uavs_per_path_min = uavs_per_path_min
        self.uavs_per_path_max = uavs_per_path_max
        self.area_size = area_size
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        # θ* — n_faulty_uavs drawn without replacement, each Uniform(theta_low, theta_high)
        self.theta_star = np.zeros(m, dtype=float)
        faulty_uavs = self._rng.choice(m, size=n_faulty_uavs, replace=False)
        self.theta_star[faulty_uavs] = self._rng.uniform(
            theta_low, theta_high, size=n_faulty_uavs
        )

        # φ* — same pattern for zones
        self.phi_star = np.zeros(Z, dtype=float)
        faulty_zones = self._rng.choice(Z, size=n_faulty_zones, replace=False)
        self.phi_star[faulty_zones] = self._rng.uniform(
            phi_low, phi_high, size=n_faulty_zones
        )

        # uav_zone — fixed for entire simulation
        self.uav_zone = self._rng.integers(0, Z, size=m)

        # Initial epoch
        self._pathset = None
        self.refresh_epoch(self._rng)

    @property
    def pathset(self) -> AbstractPathSet:
        return self._pathset

    @property
    def pair_min_dist(self) -> np.ndarray:
        return self._pathset.pair_min_dist

    def refresh_epoch(self, rng) -> None:
        """Re-sample path_uav_membership, path_zone_membership, pair_min_dist."""
        N, K, m, Z = self.N, self.K, self.m, self.Z

        path_uav_membership = np.zeros((N, K, m), dtype=bool)
        for n in range(N):
            for k in range(K):
                n_uavs = int(
                    rng.integers(self.uavs_per_path_min, self.uavs_per_path_max + 1)
                )
                n_uavs = min(n_uavs, m)
                chosen = rng.choice(m, size=n_uavs, replace=False)
                path_uav_membership[n, k, chosen] = True

        # path_zone_membership derived from uav_zone
        path_zone_membership = np.zeros((N, K, Z), dtype=bool)
        for n in range(N):
            for k in range(K):
                uav_idx = np.where(path_uav_membership[n, k])[0]
                if len(uav_idx) > 0:
                    zones = self.uav_zone[uav_idx]
                    path_zone_membership[n, k, zones] = True

        # pair_min_dist: 0 if paths share any UAV, Uniform(0, area_size) otherwise
        M = N * K
        pair_min_dist = np.zeros((M, M), dtype=float)
        for n in range(N):
            for k in range(K):
                a = n * K + k
                for l in range(N):
                    for j in range(K):
                        b = l * K + j
                        if b <= a:
                            continue
                        share = (
                            path_uav_membership[n, k]
                            & path_uav_membership[l, j]
                        ).any()
                        if share:
                            d = 0.0
                        else:
                            d = float(rng.uniform(0.0, self.area_size))
                        pair_min_dist[a, b] = d
                        pair_min_dist[b, a] = d

        self._pathset = AbstractPathSet(
            N=N,
            K=K,
            m=m,
            Z=Z,
            path_uav_membership=path_uav_membership,
            path_zone_membership=path_zone_membership,
            pair_min_dist=pair_min_dist,
        )

    def compute_losses(self, chosen_paths: np.ndarray, rng) -> np.ndarray:
        """Return (N,) per-flow losses for the chosen paths."""
        N, K = self.N, self.K
        ps = self._pathset

        selected_uav = np.array(
            [ps.path_uav_membership[n, chosen_paths[n]] for n in range(N)]
        )  # (N, m) bool
        selected_zone = np.array(
            [ps.path_zone_membership[n, chosen_paths[n]] for n in range(N)]
        )  # (N, Z) bool

        losses = np.zeros(N, dtype=float)
        losses += selected_uav.astype(float) @ self.theta_star
        losses += selected_zone.astype(float) @ self.phi_star

        # Collision: # of other flows sharing any UAV
        shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
        np.fill_diagonal(shared, False)
        collision_counts = shared.sum(axis=1).astype(float)
        losses += self.C_coll * collision_counts

        # Proximity interference
        for n in range(N):
            kn = chosen_paths[n]
            a = n * K + kn
            for l in range(N):
                if l == n:
                    continue
                kl = chosen_paths[l]
                b = l * K + kl
                losses[n] += np.exp(-ps.pair_min_dist[a, b] / self.d0)

        losses += rng.normal(0, self.sigma_noise, size=N)
        return losses
