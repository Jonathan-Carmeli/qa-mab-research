"""Oracle agent — frozen θ̂=θ*, φ̂=φ*, no learning."""
import numpy as np
from ..qa_mab_physical import QAMABPhysical


class OracleAgent(QAMABPhysical):
    name = "Oracle-Physical"

    def __init__(self, world, C_coll=5.0, d0=150.0, sigma_noise=0.05,
                 sa_n_restarts=20, sa_n_iters=1000, sa_T0=2.0, sa_decay=0.999, seed=42):
        # Parent init but then overwrite estimates with ground truth
        super().__init__(
            world, C_coll, d0, sigma_noise,
            alpha=0.0, ucb_c=0.0, epoch_decay=1.0,
            sa_n_restarts=sa_n_restarts, sa_n_iters=sa_n_iters,
            sa_T0=sa_T0, sa_decay=sa_decay, seed=seed,
        )
        self.theta_hat = world.theta_star.copy()
        self.phi_hat = world.phi_star.copy()

    def update(self, chosen_paths, losses):
        pass  # Frozen