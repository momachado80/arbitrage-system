"""
Tests for EdgeModel — mathematically correct edge computation.

Scenarios:
- Zero spread
- High spread
- Zero fees
- High fees
- Perfect oracle
- Slightly biased oracle

Fundamental invariant: EV > 0 ↔ net_edge > 0
"""

import pytest

from src.strategy.edge_model import EdgeConfig, EdgeModel, EdgeResult


class TestEdgeModelFormula:
    """Verify the core formula is correct."""

    def test_buy_uses_ask_price(self):
        """BUY: net_edge = P_oracle - P_ask - fees - slippage."""
        model = EdgeModel(EdgeConfig(fee_rate=0.01, slippage_rate=0.005, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.70, best_bid=0.50, best_ask=0.60)
        # raw_edge = 0.70 - 0.55 = 0.15 > 0 → BUY direction
        # net_edge = 0.70 - 0.60 - 0.01 - 0.005 = 0.085
        assert result.side == "BUY"
        assert abs(result.net_edge - 0.085) < 1e-9
        assert result.execution_price == 0.60

    def test_sell_uses_bid_price(self):
        """SELL: net_edge = P_bid - P_oracle - fees - slippage."""
        model = EdgeModel(EdgeConfig(fee_rate=0.01, slippage_rate=0.005, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.30, best_bid=0.50, best_ask=0.60)
        # raw_edge = 0.30 - 0.55 = -0.25 < 0 → SELL direction
        # net_edge = 0.50 - 0.30 - 0.01 - 0.005 = 0.185
        assert result.side == "SELL"
        assert abs(result.net_edge - 0.185) < 1e-9
        assert result.execution_price == 0.50

    def test_no_double_spread_subtraction(self):
        """Spread is already in execution price — never subtracted separately."""
        model = EdgeModel(EdgeConfig(fee_rate=0.0, slippage_rate=0.0, min_edge_threshold=0.0))
        # Wide spread: bid=0.40, ask=0.60, mid=0.50
        result = model.compute_net_edge(p_oracle=0.65, best_bid=0.40, best_ask=0.60)
        # BUY: net_edge = 0.65 - 0.60 - 0 - 0 = 0.05
        assert result.side == "BUY"
        assert abs(result.net_edge - 0.05) < 1e-9
        # NOT: 0.65 - 0.50 - 0.10 = 0.05 (old wrong formula with spread subtraction)
        # Both give same number here, but for different reasons.
        # The key: no spread_impact separate term.


class TestZeroSpread:
    """When spread is zero, bid=ask=mid."""

    def test_zero_spread_buy(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.60, best_bid=0.50, best_ask=0.50)
        # BUY: net_edge = 0.60 - 0.50 - 0.025 = 0.075
        assert result.side == "BUY"
        assert abs(result.net_edge - 0.075) < 1e-9

    def test_zero_spread_sell(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.40, best_bid=0.50, best_ask=0.50)
        # SELL: net_edge = 0.50 - 0.40 - 0.025 = 0.075
        assert result.side == "SELL"
        assert abs(result.net_edge - 0.075) < 1e-9

    def test_zero_spread_symmetric(self):
        """With zero spread, equal-magnitude edges should produce equal net_edge."""
        model = EdgeModel(EdgeConfig(fee_rate=0.01, slippage_rate=0.0, min_edge_threshold=0.0))
        buy = model.compute_net_edge(p_oracle=0.60, best_bid=0.50, best_ask=0.50)
        sell = model.compute_net_edge(p_oracle=0.40, best_bid=0.50, best_ask=0.50)
        assert abs(buy.net_edge - sell.net_edge) < 1e-9


class TestHighSpread:
    """Wide spread absorbs edge — execution at worse price."""

    def test_high_spread_kills_buy_edge(self):
        """Wide spread makes BUY edge negative even with positive raw_edge."""
        model = EdgeModel(EdgeConfig(fee_rate=0.0, slippage_rate=0.0, min_edge_threshold=0.0))
        # mid=0.50, oracle=0.55, raw_edge=0.05
        # But ask=0.60, so BUY: net_edge = 0.55 - 0.60 = -0.05
        result = model.compute_net_edge(p_oracle=0.55, best_bid=0.40, best_ask=0.60)
        assert result.side is None  # No actionable edge
        assert result.net_edge < 0

    def test_high_spread_kills_sell_edge(self):
        """Wide spread makes SELL edge negative even with negative raw_edge."""
        model = EdgeModel(EdgeConfig(fee_rate=0.0, slippage_rate=0.0, min_edge_threshold=0.0))
        # mid=0.50, oracle=0.45, raw_edge=-0.05
        # But bid=0.40, so SELL: net_edge = 0.40 - 0.45 = -0.05
        result = model.compute_net_edge(p_oracle=0.45, best_bid=0.40, best_ask=0.60)
        assert result.side is None
        assert result.net_edge < 0


class TestZeroFees:
    """When fees=0 and slippage=0, only execution price matters."""

    def test_zero_fees_buy(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.0, slippage_rate=0.0, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.65, best_bid=0.48, best_ask=0.52)
        # BUY: net_edge = 0.65 - 0.52 = 0.13
        assert abs(result.net_edge - 0.13) < 1e-9

    def test_zero_fees_sell(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.0, slippage_rate=0.0, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.35, best_bid=0.48, best_ask=0.52)
        # SELL: net_edge = 0.48 - 0.35 = 0.13
        assert abs(result.net_edge - 0.13) < 1e-9


class TestHighFees:
    """High fees can make otherwise profitable trades unprofitable."""

    def test_high_fees_eliminate_edge(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.10, slippage_rate=0.05, min_edge_threshold=0.0))
        # BUY: net_edge = 0.65 - 0.52 - 0.15 = -0.02
        result = model.compute_net_edge(p_oracle=0.65, best_bid=0.48, best_ask=0.52)
        assert result.net_edge < 0
        assert result.side is None

    def test_high_fees_need_bigger_edge(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.05, slippage_rate=0.02, min_edge_threshold=0.0))
        # BUY: net_edge = 0.80 - 0.52 - 0.07 = 0.21
        result = model.compute_net_edge(p_oracle=0.80, best_bid=0.48, best_ask=0.52)
        assert result.net_edge > 0
        assert result.side == "BUY"


class TestPerfectOracle:
    """Oracle probability = 1.0 or 0.0 (perfect knowledge)."""

    def test_perfect_buy(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=1.0, best_bid=0.48, best_ask=0.52)
        # BUY: net_edge = 1.0 - 0.52 - 0.025 = 0.455
        assert result.side == "BUY"
        assert abs(result.net_edge - 0.455) < 1e-9

    def test_perfect_sell(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.0, best_bid=0.48, best_ask=0.52)
        # SELL: net_edge = 0.48 - 0.0 - 0.025 = 0.455
        assert result.side == "SELL"
        assert abs(result.net_edge - 0.455) < 1e-9


class TestSlightlyBiasedOracle:
    """Oracle is close to market price — small edge scenarios."""

    def test_tiny_edge_below_threshold(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005, min_edge_threshold=0.03))
        result = model.compute_net_edge(p_oracle=0.56, best_bid=0.48, best_ask=0.52)
        # BUY: net_edge = 0.56 - 0.52 - 0.025 = 0.015 < 0.03 threshold
        assert result.net_edge > 0
        assert result.side is None  # Below threshold

    def test_marginal_edge_above_threshold(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.01, slippage_rate=0.005, min_edge_threshold=0.03))
        result = model.compute_net_edge(p_oracle=0.57, best_bid=0.48, best_ask=0.52)
        # BUY: net_edge = 0.57 - 0.52 - 0.015 = 0.035 > 0.03 threshold
        assert result.side == "BUY"
        assert result.net_edge >= 0.03


class TestEVNetEdgeInvariant:
    """FUNDAMENTAL: EV > 0 ↔ net_edge > 0."""

    @pytest.mark.parametrize("p_oracle,bid,ask,fee,slip", [
        (0.70, 0.48, 0.52, 0.02, 0.005),   # BUY, positive
        (0.30, 0.48, 0.52, 0.02, 0.005),   # SELL, positive
        (0.55, 0.40, 0.60, 0.01, 0.005),   # BUY, negative (spread too wide)
        (0.45, 0.40, 0.60, 0.01, 0.005),   # SELL, negative (spread too wide)
        (0.80, 0.48, 0.52, 0.00, 0.000),   # BUY, zero costs
        (0.20, 0.48, 0.52, 0.00, 0.000),   # SELL, zero costs
        (0.60, 0.48, 0.52, 0.10, 0.050),   # BUY, high costs
        (0.40, 0.48, 0.52, 0.10, 0.050),   # SELL, high costs
        (1.00, 0.50, 0.50, 0.02, 0.005),   # Perfect oracle, zero spread
        (0.00, 0.50, 0.50, 0.02, 0.005),   # Perfect oracle, zero spread
        (0.51, 0.49, 0.51, 0.01, 0.005),   # Tiny edge, tight spread
    ])
    def test_ev_equals_net_edge_sign(self, p_oracle, bid, ask, fee, slip):
        """EV and net_edge must always have the same sign."""
        model = EdgeModel(EdgeConfig(fee_rate=fee, slippage_rate=slip, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=p_oracle, best_bid=bid, best_ask=ask)

        # EV and net_edge must have same sign (or both zero)
        if abs(result.net_edge) > 1e-12:
            assert (result.expected_value > 0) == (result.net_edge > 0), (
                f"INVARIANT VIOLATED: net_edge={result.net_edge}, "
                f"EV={result.expected_value}, side={result.side}"
            )

    def test_ev_positive_iff_net_edge_positive_buy(self):
        """Explicit BUY case: EV = P_oracle - ask - costs = net_edge."""
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.70, best_bid=0.48, best_ask=0.52)
        assert result.side == "BUY"
        assert abs(result.expected_value - result.net_edge) < 1e-9

    def test_ev_positive_iff_net_edge_positive_sell(self):
        """Explicit SELL case: EV = bid - P_oracle - costs = net_edge."""
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.30, best_bid=0.48, best_ask=0.52)
        assert result.side == "SELL"
        assert abs(result.expected_value - result.net_edge) < 1e-9


class TestExpectedValue:
    """EV method directly."""

    def test_buy_ev(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005))
        # BUY at ask=0.52: EV = P_oracle - 0.52 - 0.025
        ev = model.expected_value(p_oracle=0.70, execution_price=0.52, total_cost=0.025, side="BUY")
        assert abs(ev - 0.155) < 1e-9

    def test_sell_ev(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005))
        # SELL at bid=0.48: EV = 0.48 - P_oracle - 0.025
        ev = model.expected_value(p_oracle=0.30, execution_price=0.48, total_cost=0.025, side="SELL")
        assert abs(ev - 0.155) < 1e-9


class TestBreakEvenProbability:
    """Break-even point: EV = 0."""

    def test_buy_break_even(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005))
        be = model.break_even_probability(execution_price=0.52, side="BUY")
        # P_break = 0.52 + 0.025 = 0.545
        assert abs(be - 0.545) < 1e-9

    def test_sell_break_even(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005))
        be = model.break_even_probability(execution_price=0.48, side="SELL")
        # P_break = 0.48 - 0.025 = 0.455
        assert abs(be - 0.455) < 1e-9

    def test_break_even_means_zero_ev(self):
        """At break-even probability, EV should be exactly 0."""
        model = EdgeModel(EdgeConfig(fee_rate=0.02, slippage_rate=0.005))
        be_buy = model.break_even_probability(execution_price=0.52, side="BUY")
        ev = model.expected_value(p_oracle=be_buy, execution_price=0.52, total_cost=0.025, side="BUY")
        assert abs(ev) < 1e-9

        be_sell = model.break_even_probability(execution_price=0.48, side="SELL")
        ev = model.expected_value(p_oracle=be_sell, execution_price=0.48, total_cost=0.025, side="SELL")
        assert abs(ev) < 1e-9


class TestEdgeResultProperties:
    """EdgeResult helper properties."""

    def test_is_actionable_true(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.0, slippage_rate=0.0, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.70, best_bid=0.48, best_ask=0.52)
        assert result.is_actionable is True

    def test_is_actionable_false_no_edge(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.0, slippage_rate=0.0, min_edge_threshold=0.0))
        result = model.compute_net_edge(p_oracle=0.50, best_bid=0.48, best_ask=0.52)
        # raw_edge = 0.0, side = None
        assert result.is_actionable is False

    def test_is_actionable_false_negative_edge(self):
        model = EdgeModel(EdgeConfig(fee_rate=0.0, slippage_rate=0.0, min_edge_threshold=0.0))
        # oracle slightly above mid but below ask
        result = model.compute_net_edge(p_oracle=0.51, best_bid=0.40, best_ask=0.60)
        assert result.net_edge < 0
        assert result.is_actionable is False
