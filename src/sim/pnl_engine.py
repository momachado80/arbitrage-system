"""
PnL Engine — Compute trade-level profit and loss.
"""


class PnLEngine:
    """Calculate realized PnL from edge, slippage, fees, and alpha decay."""

    def compute(
        self,
        edge_bps: float,
        slippage_bps: float,
        fees_bps: float,
        alpha_decay_bps: float = 0.0,
    ) -> float:
        """Net PnL in bps after all frictions."""
        return edge_bps - slippage_bps - fees_bps - alpha_decay_bps

    def alpha_decay(
        self,
        mid_at_detect: float,
        mid_at_place: float,
    ) -> float:
        """Price movement between detection and placement in bps."""
        if mid_at_detect <= 0:
            return 0.0
        return abs(mid_at_detect - mid_at_place) * 10000.0

    def alpha_capture_ratio(
        self,
        realized_pnl_bps: float,
        expected_edge_bps: float,
    ) -> float:
        """Fraction of detected edge actually captured. 1.0 = perfect."""
        if expected_edge_bps <= 0:
            return 0.0
        return realized_pnl_bps / expected_edge_bps
