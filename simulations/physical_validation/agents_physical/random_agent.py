"""Random routing agent for physical model — picks paths uniformly at random."""
import numpy as np


class RandomAgent:
    name = "Random-Physical"

    def __init__(self, world, seed=42):
        self._rng = np.random.default_rng(seed)
        self._N = world.N
        self._K = world.K

    def reset_epoch(self):
        pass

    def act(self, t: int, p: int) -> np.ndarray:
        return self._rng.integers(0, self._K, size=self._N)

    def update(self, chosen_paths: np.ndarray, losses: np.ndarray) -> None:
        pass