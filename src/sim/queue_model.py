"""
Queue Model — Simulates maker queue position for limit orders.
Deterministic given a seeded RNG.
"""

import random
from typing import Optional


class QueueModel:
    """Model queue position and fill probability for maker orders."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random(42)

    def initial_position(self, visible_size: float) -> float:
        """Estimate initial queue position. Uniform within visible depth."""
        if visible_size <= 0:
            return float("inf")
        return self.rng.uniform(0, visible_size)

    def is_filled(self, queue_position: float, traded_volume: float) -> bool:
        """Determine if enough volume traded through to fill our order."""
        return traded_volume >= queue_position

    def probabilistic_fill(
        self,
        fill_prob_per_second: float,
        time_at_best_s: float,
    ) -> bool:
        """Geometric fill model: P(fill) = 1 - (1-p)^t.  Deterministic via rng."""
        if fill_prob_per_second <= 0 or time_at_best_s <= 0:
            return False
        prob_no_fill = (1.0 - fill_prob_per_second) ** time_at_best_s
        fill_prob = 1.0 - prob_no_fill
        return self.rng.random() < fill_prob

    def fill_time_offset(self, time_at_best_s: float) -> float:
        """Uniform random fill time within the at-best window."""
        if time_at_best_s <= 0:
            return 0.0
        return self.rng.uniform(0, time_at_best_s)
