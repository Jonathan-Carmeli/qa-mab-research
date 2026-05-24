"""QA-MAB agent for the physical abstract model.

Residual credit assignment + epoch decay + temperature-scaled QUBO + SA solver.
"""
import numpy as np
from .physical_env import AbstractWorld
from .sa_solver_physical import sa_solve


class QAMABPhysical:
    """QA-MAB agent that learns per-UAV θ̂ and per-zone φ̂ failure rates."""

    name = "QA-MAB-Physical"

    def __init__(
        self,
        world: AbstractWorld,
        C_coll: float = 5.0,
        d0: float = 150.0,
        sigma_noise: float = 0.05,
        alpha: float = 0.15,
        ucb_c: float = 3.0,
        epoch_decay: float = 0.7,
        lambda_onehot: float = 10.0,
        gamma_0: float = 2.0,
        a: float = 0.5,
        b: float = 0.3,
        sa_n_restarts: int = 20,
        sa_n_iters: int = 1000,
        sa_T0: float = 2.0,
        sa_decay: float = 0.999,
        seed: int = 42,
    ):
        self.world = world
        self.C_coll = C_coll
        self.d0 = d0
        self.sigma_noise = sigma_noise
        self.alpha = alpha
        self.ucb_c = ucb_c
        self.epoch_decay = epoch_decay
        self.lam = lambda_onehot
        self.gamma_0 = gamma_0
        self.a = a
        self.b = b
        self.sa_n_restarts = sa_n_restarts
        self.sa_n_iters = sa_n_iters
        self.sa_T0 = sa_T0
        self.sa_decay = sa_decay
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.theta_hat = np.zeros(world.m, dtype=float)
        self.phi_hat = np.zeros(world.Z, dtype=float)
        self._visit_counts = None

    # ─── Epoch lifecycle ────────────────────────────────────────────────

    def reset_epoch(self, p: int) -> None:
        """Decay estimates (if p>0), reset visit counts, refresh world paths."""
        if p > 0:
            self.theta_hat *= self.epoch_decay
            self.phi_hat *= self.epoch_decay
        self._visit_counts = np.zeros((self.world.N, self.world.K), dtype=int)
        self.world.refresh_epoch(self.rng)

    # ─── QUBO construction ──────────────────────────────────────────────

    def build_qubo(self) -> np.ndarray:
        """Build (M,M) QUBO matrix where M=N*K."""
        N, K = self.world.N, self.world.K
        M = N * K
        ps = self.world.pathset

        Q = np.zeros((M, M), dtype=float)

        for n in range(N):
            for k in range(K):
                i = n * K + k
                uav_mask = ps.path_uav_membership[n, k]
                zone_mask = ps.path_zone_membership[n, k]

                cost = float(self.theta_hat[uav_mask].sum()) + float(self.phi_hat[zone_mask].sum())
                visits = self._visit_counts[n, k]
                ucb_bonus = self.ucb_c / np.sqrt(max(visits, 1))
                Q[i, i] = cost - self.lam - ucb_bonus

        # Same-flow one-hot penalty
        for n in range(N):
            for k in range(K):
                for kp in range(k + 1, K):
                    i = n * K + k
                    j = n * K + kp
                    Q[i, j] += self.lam
                    Q[j, i] += self.lam

        # Cross-flow terms: collision + proximity
        for n in range(N):
            for k in range(K):
                i = n * K + k
                for l in range(N):
                    if l == n:
                        continue
                    for j in range(K):
                        jj = l * K + j
                        if jj <= i:
                            continue
                        contrib = 0.0
                        # Collision
                        if (ps.path_uav_membership[n, k] & ps.path_uav_membership[l, j]).any():
                            contrib += self.C_coll
                        # Proximity
                        contrib += np.exp(-ps.pair_min_dist[i, jj] / self.d0)
                        Q[i, jj] += contrib
                        Q[jj, i] += contrib

        return Q

    # ─── Action & update ────────────────────────────────────────────────

    def _temperature(self, p: int, t: int) -> float:
        return self.gamma_0 / (((p + 1) ** self.a) * ((t + 1) ** self.b))

    def act(self, t: int, p: int) -> np.ndarray:
        """Build QUBO, scale by temperature, run SA, decode to chosen paths."""
        Q = self.build_qubo()
        gamma = self._temperature(p, t)
        Q_scaled = Q / max(gamma, 1e-8)

        return sa_solve(
            Q_scaled,
            self.world.N,
            self.world.K,
            self.rng,
            n_restarts=self.sa_n_restarts,
            n_iters=self.sa_n_iters,
            T0=self.sa_T0,
            decay=self.sa_decay,
        )

    def update(self, chosen_paths: np.ndarray, losses: np.ndarray) -> None:
        """Residual credit assignment for θ̂ and φ̂."""
        N = self.world.N
        K = self.world.K
        ps = self.world.pathset

        # Track visits
        for n in range(N):
            self._visit_counts[n, chosen_paths[n]] += 1

        # Selected membership
        selected_uav = np.array(
            [ps.path_uav_membership[n, chosen_paths[n]] for n in range(N)]
        )
        selected_zone = np.array(
            [ps.path_zone_membership[n, chosen_paths[n]] for n in range(N)]
        )

        # Collision count per flow
        shared = (selected_uav.astype(int) @ selected_uav.astype(int).T) > 0
        np.fill_diagonal(shared, False)
        collision_counts = shared.sum(axis=1).astype(float)

        # Proximity interference per flow
        prox = np.zeros(N, dtype=float)
        for n in range(N):
            kn = chosen_paths[n]
            a = n * K + kn
            for l in range(N):
                if l == n:
                    continue
                kl = chosen_paths[l]
                b = l * K + kl
                prox[n] += np.exp(-ps.pair_min_dist[a, b] / self.d0)

        # Fault-only residual loss
        L_fault = losses - self.C_coll * collision_counts - prox
        L_fault = np.maximum(L_fault, 0.0)

        # Residual credit assignment
        for n in range(N):
            k = chosen_paths[n]
            uav_mask = ps.path_uav_membership[n, k]
            zone_mask = ps.path_zone_membership[n, k]

            observed = float(L_fault[n])

            # UAVs — first pass
            theta_contrib = float(self.theta_hat[uav_mask].sum())
            phi_contrib = float(self.phi_hat[zone_mask].sum())
            for i in np.where(uav_mask)[0]:
                other_theta = theta_contrib - self.theta_hat[i]
                residual_i = observed - other_theta - phi_contrib
                self.theta_hat[i] += self.alpha * (residual_i - self.theta_hat[i])

            # Zones — after UAV updates (more accurate phi estimate)
            theta_contrib = float(self.theta_hat[uav_mask].sum())
            for z in np.where(zone_mask)[0]:
                other_phi = float(self.phi_hat[zone_mask].sum()) - self.phi_hat[z]
                residual_z = observed - theta_contrib - other_phi
                self.phi_hat[z] += self.alpha * (residual_z - self.phi_hat[z])

        np.clip(self.theta_hat, 0.0, 1.0, out=self.theta_hat)
        np.clip(self.phi_hat,   0.0, 1.0, out=self.phi_hat)

    # ─── Full run ───────────────────────────────────────────────────────

    def run(self, P: int, T: int, rng=None) -> dict:
        """Run P epochs × T steps. Returns dict of logs."""
        if rng is None:
            rng = self.rng

        losses_log = np.zeros((P, T, self.world.N), dtype=float)
        theta_err_log = np.zeros(P, dtype=float)
        phi_err_log = np.zeros(P, dtype=float)
        chosen_log = np.zeros((P, T, self.world.N), dtype=int)
        gap_log = np.zeros(P, dtype=float)

        for p in range(P):
            self.reset_epoch(p)
            for t in range(T):
                chosen = self.act(t, p)
                losses = self.world.compute_losses(chosen, rng)
                self.update(chosen, losses)

                losses_log[p, t] = losses
                chosen_log[p, t] = chosen

            # Estimation error post-epoch
            theta_err_log[p] = float(np.linalg.norm(self.theta_hat - self.world.theta_star))
            phi_err_log[p] = float(np.linalg.norm(self.phi_hat - self.world.phi_star))

        return dict(
            losses_log=losses_log,
            theta_err_log=theta_err_log,
            phi_err_log=phi_err_log,
            chosen_log=chosen_log,
            gap_log=gap_log,
        )