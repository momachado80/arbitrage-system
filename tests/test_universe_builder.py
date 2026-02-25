"""Tests for UniverseBuilder — market selection and filtering."""

import pytest

from src.universe.universe_builder import (
    MarketInfo,
    UniverseBuilder,
    UniverseFilters,
)
from src.oracle.probability_oracle import MockOracleProvider


class TestUniverseBuilder:
    def _make_market(self, **kwargs) -> MarketInfo:
        defaults = dict(
            token_id="t1",
            best_bid=0.48,
            best_ask=0.52,
            bid_size=100,
            ask_size=100,
            volume_24h=500,
            days_to_close=7,
            oracle_supported=True,
        )
        defaults.update(kwargs)
        return MarketInfo(**defaults)

    def test_manual_override(self):
        ub = UniverseBuilder(manual_tokens=["m1", "m2"])
        tokens = ub.update(force=True)
        assert tokens == ["m1", "m2"]
        assert ub.is_manual_override() is True

    def test_empty_manual_uses_filter(self):
        ub = UniverseBuilder(manual_tokens=[])
        ub.set_candidates([self._make_market(token_id="t1")])
        tokens = ub.update(force=True)
        assert "t1" in tokens
        assert ub.is_manual_override() is False

    def test_liquidity_filter(self):
        ub = UniverseBuilder(
            filters=UniverseFilters(min_liquidity=100),
        )
        # Liquidity = (bid_size + ask_size) * mid = (10 + 10) * 0.5 = 10 < 100
        low_liq = self._make_market(token_id="low", bid_size=10, ask_size=10)
        high_liq = self._make_market(token_id="high", bid_size=200, ask_size=200)
        ub.set_candidates([low_liq, high_liq])
        tokens = ub.update(force=True)
        assert "high" in tokens
        assert "low" not in tokens

    def test_spread_filter(self):
        ub = UniverseBuilder(
            filters=UniverseFilters(max_spread=0.05, min_liquidity=0),
        )
        wide = self._make_market(token_id="wide", best_bid=0.40, best_ask=0.60)
        narrow = self._make_market(token_id="narrow", best_bid=0.49, best_ask=0.51)
        ub.set_candidates([wide, narrow])
        tokens = ub.update(force=True)
        assert "narrow" in tokens
        assert "wide" not in tokens

    def test_volume_filter(self):
        ub = UniverseBuilder(
            filters=UniverseFilters(min_volume_24h=1000, min_liquidity=0),
        )
        low_vol = self._make_market(token_id="low", volume_24h=100)
        high_vol = self._make_market(token_id="high", volume_24h=5000)
        ub.set_candidates([low_vol, high_vol])
        tokens = ub.update(force=True)
        assert "high" in tokens
        assert "low" not in tokens

    def test_days_to_close_filter(self):
        ub = UniverseBuilder(
            filters=UniverseFilters(min_days_to_close=3, min_liquidity=0),
        )
        soon = self._make_market(token_id="soon", days_to_close=1)
        later = self._make_market(token_id="later", days_to_close=10)
        ub.set_candidates([soon, later])
        tokens = ub.update(force=True)
        assert "later" in tokens
        assert "soon" not in tokens

    def test_oracle_filter(self):
        class SelectiveProvider(MockOracleProvider):
            def supports(self, token_id: str) -> bool:
                return token_id == "supported"

        ub = UniverseBuilder(
            oracle_provider=SelectiveProvider(),
            filters=UniverseFilters(min_liquidity=0),
        )
        supported = self._make_market(token_id="supported")
        unsupported = self._make_market(token_id="unsupported")
        ub.set_candidates([supported, unsupported])
        tokens = ub.update(force=True)
        assert "supported" in tokens
        assert "unsupported" not in tokens

    def test_max_markets(self):
        ub = UniverseBuilder(
            filters=UniverseFilters(max_markets=2, min_liquidity=0),
        )
        markets = [
            self._make_market(token_id=f"t{i}", bid_size=100 + i * 10, ask_size=100)
            for i in range(5)
        ]
        ub.set_candidates(markets)
        tokens = ub.update(force=True)
        assert len(tokens) == 2

    def test_throttle_update(self):
        ub = UniverseBuilder(update_interval=300)
        ub.set_candidates([self._make_market()])
        tokens1 = ub.update(force=True)
        assert len(tokens1) == 1
        # Second call without force should return cached
        tokens2 = ub.update(force=False)
        assert tokens2 == tokens1

    def test_add_manual_token(self):
        ub = UniverseBuilder()
        ub.add_manual_token("m1")
        assert "m1" in ub.get_active_tokens()
        assert ub.is_manual_override() is True

    def test_remove_manual_token(self):
        ub = UniverseBuilder(manual_tokens=["m1", "m2"])
        ub.remove_manual_token("m1")
        assert "m1" not in ub.get_active_tokens()
        assert "m2" in ub.get_active_tokens()

    def test_summary(self):
        ub = UniverseBuilder(manual_tokens=["m1"])
        summary = ub.get_summary()
        assert summary["manual_override"] is True
        assert summary["active_tokens"] == 1
        assert "filters" in summary
