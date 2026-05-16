"""NB3R agent for physical model — collaborative bandit routing."""
import numpy as np


class NB3RAgent:
    name = "NB3R-Physical"

    def __init__(self, world, alpha=0.3, tau0=0.1, delta_tau=0.05, seed=42):
        self._rng = np.random.default_rng(seed)
        self._world = world
        self._alpha = alpha
        self._tau = tau0
        self._delta_tau = delta_tau
        self._N = world.N
        self._K = world.K
        self._W = np.zeros((self._N, self._K), dtype=float)

    def reset_epoch(self):
        # W persists across epochs — do NOT reset here
        self._N = self._world.N
        self._K = self._world.K

    def act(self, t: int, p: int) -> np.ndarray:
        chosen = np.zeros(self._N, dtype=int)
        for n in range(self._N):
            logits = -self._W[n] / self._tau
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            chosen[n] = self._rng.choice(self._K, p=probs)
        return chosen

    def update(self, chosen_paths: np.ndarray, losses: np.ndarray) -> None:
        for n in range(self._N):
            collaborative = -losses[n] + sum(-losses[l] for l in range(self._N) if l != n)
            self._W[n, chosen_paths[n]] += self._alpha * (collaborative - self._W[n, chosen_paths[n]])
        self._tau += self._delta_tau