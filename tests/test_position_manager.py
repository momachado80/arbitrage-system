"""Testes para PositionManager (spot-only positions)."""

import pytest
from src.risk.position_manager import PositionManager, ShortNotAllowedError


@pytest.fixture
def pm():
    return PositionManager()


# =============================================================================
# BUY — CRIAÇÃO E ACUMULAÇÃO DE POSIÇÃO
# =============================================================================

class TestBuy:
    def test_buy_creates_position(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pos = pm.get_position("m1", "YES")
        assert pos.shares == 100
        assert pos.avg_entry_price == 0.65

    def test_buy_additional_recalculates_avg(self, pm):
        """Weighted average: (100*0.65 + 50*0.80) / 150 = 0.70"""
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pm.apply_fill("m1", "YES", 50, 0.80, "BUY")
        pos = pm.get_position("m1", "YES")
        assert pos.shares == 150
        expected_avg = (100 * 0.65 + 50 * 0.80) / 150
        assert abs(pos.avg_entry_price - expected_avg) < 1e-10

    def test_buy_different_sides_independent(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pm.apply_fill("m1", "NO", 50, 0.35, "BUY")
        assert pm.get_position("m1", "YES").shares == 100
        assert pm.get_position("m1", "NO").shares == 50

    def test_buy_different_markets_independent(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pm.apply_fill("m2", "YES", 200, 0.50, "BUY")
        assert pm.get_position("m1", "YES").shares == 100
        assert pm.get_position("m2", "YES").shares == 200

    def test_buy_zero_shares_noop(self, pm):
        pm.apply_fill("m1", "YES", 0, 0.65, "BUY")
        assert pm.get_position("m1", "YES").shares == 0


# =============================================================================
# SELL — REDUÇÃO E ENCERRAMENTO
# =============================================================================

class TestSell:
    def test_sell_partial_reduces(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pm.apply_fill("m1", "YES", 30, 0.70, "SELL")
        pos = pm.get_position("m1", "YES")
        assert pos.shares == 70
        # avg_entry_price permanece o mesmo após SELL
        assert pos.avg_entry_price == 0.65

    def test_sell_total_zeros(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pm.apply_fill("m1", "YES", 100, 0.70, "SELL")
        pos = pm.get_position("m1", "YES")
        assert pos.shares == 0
        assert pos.avg_entry_price == 0.0

    def test_sell_above_position_raises(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        with pytest.raises(ShortNotAllowedError):
            pm.apply_fill("m1", "YES", 150, 0.70, "SELL")

    def test_sell_without_position_raises(self, pm):
        with pytest.raises(ShortNotAllowedError):
            pm.apply_fill("m1", "YES", 10, 0.70, "SELL")


# =============================================================================
# CAN_SELL
# =============================================================================

class TestCanSell:
    def test_can_sell_within_position(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        assert pm.can_sell("m1", "YES", 50) is True
        assert pm.can_sell("m1", "YES", 100) is True

    def test_cannot_sell_above_position(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        assert pm.can_sell("m1", "YES", 101) is False

    def test_cannot_sell_without_position(self, pm):
        assert pm.can_sell("m1", "YES", 10) is False
        assert pm.can_sell("m1", "YES", 0) is True  # Vender zero é OK


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_invalid_action_raises(self, pm):
        with pytest.raises(ValueError, match="action"):
            pm.apply_fill("m1", "YES", 10, 0.65, "SHORT")

    def test_get_position_nonexistent_returns_empty(self, pm):
        pos = pm.get_position("fake", "YES")
        assert pos.shares == 0
        assert pos.avg_entry_price == 0.0

    def test_get_position_returns_copy(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pos = pm.get_position("m1", "YES")
        pos.shares = 999
        assert pm.get_position("m1", "YES").shares == 100

    def test_get_all_positions(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pm.apply_fill("m2", "NO", 50, 0.35, "BUY")
        all_pos = pm.get_all_positions()
        assert len(all_pos) == 2
        assert ("m1", "YES") in all_pos
        assert ("m2", "NO") in all_pos

    def test_last_updated_set(self, pm):
        pm.apply_fill("m1", "YES", 100, 0.65, "BUY")
        pos = pm.get_position("m1", "YES")
        assert pos.last_updated != ""
        assert "+" in pos.last_updated or "Z" in pos.last_updated  # UTC


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
