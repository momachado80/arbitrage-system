"""
Latency Model — Abstracts signal-to-fill latency pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LatencyConfig:
    signal_to_order_ms: float = 50.0
    order_to_exchange_ms: float = 150.0
    exchange_ack_ms: float = 50.0


class LatencyModel:
    """Compute timing offsets through the latency pipeline."""

    def __init__(self, cfg: Optional[LatencyConfig] = None) -> None:
        self.cfg = cfg or LatencyConfig()

    @property
    def signal_to_order(self) -> float:
        return self.cfg.signal_to_order_ms

    @property
    def network(self) -> float:
        return self.cfg.order_to_exchange_ms

    @property
    def exchange(self) -> float:
        return self.cfg.exchange_ack_ms

    def total_latency_ms(self) -> float:
        return self.signal_to_order + self.network + self.exchange

    def place_time(self, detect_ts: float) -> float:
        """Timestamp when order is submitted."""
        return detect_ts + self.signal_to_order / 1000.0

    def ack_time(self, detect_ts: float) -> float:
        """Timestamp when exchange acknowledges the order."""
        return detect_ts + self.total_latency_ms() / 1000.0
