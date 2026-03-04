"""
Liquidity Model — Compute slippage from order size and available liquidity.
"""


class LiquidityModel:
    """Estimate market impact / slippage based on order size vs liquidity."""

    def __init__(self, impact_coeff: float = 0.1) -> None:
        self.impact_coeff = impact_coeff

    def compute_slippage(
        self,
        order_size: float,
        liquidity: float,
        spread_bps: float,
    ) -> float:
        """
        Slippage in bps.  Uses linear impact model bounded by spread.
        Always returns >= 0.
        """
        if liquidity <= 0:
            return max(spread_bps, 0.0)
        impact = self.impact_coeff * order_size / liquidity * 10000.0
        return max(spread_bps * 0.25, impact)

    def compute_slippage_simple(
        self,
        spread_bps: float,
        spread_mult: float,
        floor_bps: float,
    ) -> float:
        """Legacy-compatible slippage: max(spread * mult, floor)."""
        return max(spread_bps * spread_mult, floor_bps)
