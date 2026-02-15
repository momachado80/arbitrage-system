"""Testes para RiskManager."""

import uuid
import shutil
import pytest
from pathlib import Path

from src.engine.trade_ledger import TradeLedger
from src.risk.risk_manager import RiskManager


@pytest.fixture
def ws_tmp(request):
    d = Path(__file__).parent.parent / ".test_tmp" / request.node.name
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def ledger(ws_tmp):
    return TradeLedger(state_path=str(ws_tmp / "ledger.json"))


def _add_trade(ledger, category="POLITICS", market_id="m1", size=100.0):
    tid = str(uuid.uuid4())
    ledger.record_entry(tid, market_id, category, "DIRECTIONAL", 0.5, size, 1.0, size, 0.5)
    return tid


# =============================================================================
# TESTES BÁSICOS
# =============================================================================

class TestCanOpenTrade:
    def test_allowed_within_limits(self, ledger):
        rm = RiskManager({"max_exposure_total": 1000})
        ok, reason = rm.can_open_trade(
            {"market_id": "m1", "category": "POLITICS", "final_size": 100}, ledger
        )
        assert ok is True
        assert reason == ""

    def test_blocked_by_total_exposure(self, ledger):
        rm = RiskManager({"max_exposure_total": 150})
        _add_trade(ledger, size=100)
        ok, reason = rm.can_open_trade(
            {"market_id": "m2", "category": "CRYPTO", "final_size": 100}, ledger
        )
        assert ok is False
        assert "total" in reason.lower()


class TestCategoryLimit:
    def test_blocked_by_category(self, ledger):
        rm = RiskManager({"max_exposure_per_category": 150})
        _add_trade(ledger, category="CRYPTO", size=100)
        ok, reason = rm.can_open_trade(
            {"market_id": "m2", "category": "CRYPTO", "final_size": 100}, ledger
        )
        assert ok is False
        assert "categoria" in reason.lower() or "CRYPTO" in reason

    def test_different_category_ok(self, ledger):
        rm = RiskManager({"max_exposure_per_category": 150})
        _add_trade(ledger, category="CRYPTO", size=100)
        ok, _ = rm.can_open_trade(
            {"market_id": "m2", "category": "POLITICS", "final_size": 100}, ledger
        )
        assert ok is True


class TestMarketLimit:
    def test_blocked_by_market(self, ledger):
        rm = RiskManager({"max_exposure_per_market": 150})
        _add_trade(ledger, market_id="m1", size=100)
        ok, reason = rm.can_open_trade(
            {"market_id": "m1", "category": "POLITICS", "final_size": 100}, ledger
        )
        assert ok is False
        assert "mercado" in reason.lower() or "m1" in reason

    def test_different_market_ok(self, ledger):
        rm = RiskManager({"max_exposure_per_market": 150})
        _add_trade(ledger, market_id="m1", size=100)
        ok, _ = rm.can_open_trade(
            {"market_id": "m2", "category": "POLITICS", "final_size": 100}, ledger
        )
        assert ok is True


class TestOpenTradesLimit:
    def test_blocked_by_count(self, ledger):
        rm = RiskManager({"max_open_trades": 3})
        for _ in range(3):
            _add_trade(ledger, size=10)
        ok, reason = rm.can_open_trade(
            {"market_id": "m1", "category": "P", "final_size": 10}, ledger
        )
        assert ok is False
        assert "open trades" in reason.lower()


class TestDeadzone:
    def test_blocked_by_deadzone(self, ledger):
        rm = RiskManager({
            "total_capital": 1000,
            "max_percent_capital_deadzone": 0.3,
        })
        _add_trade(ledger, size=250)
        ok, reason = rm.can_open_trade(
            {"market_id": "m2", "category": "C", "final_size": 100}, ledger
        )
        # 250+100 = 350 / 1000 = 0.35 > 0.3
        assert ok is False
        assert "deadzone" in reason.lower()

    def test_within_deadzone_ok(self, ledger):
        rm = RiskManager({
            "total_capital": 1000,
            "max_percent_capital_deadzone": 0.5,
        })
        _add_trade(ledger, size=200)
        ok, _ = rm.can_open_trade(
            {"market_id": "m2", "category": "C", "final_size": 100}, ledger
        )
        # 200+100 = 300 / 1000 = 0.3 < 0.5
        assert ok is True


class TestIntegration:
    def test_pipeline_blocked_then_resolved(self, ws_tmp):
        """Após resolver trades, espaço é liberado para novos."""
        ledger = TradeLedger(state_path=str(ws_tmp / "l.json"))
        rm = RiskManager({"max_exposure_total": 200})

        t1 = _add_trade(ledger, size=100)
        t2 = _add_trade(ledger, size=100)

        # Bloqueado (200+100 > 200)
        ok, _ = rm.can_open_trade(
            {"market_id": "m1", "category": "P", "final_size": 100}, ledger
        )
        assert ok is False

        # Resolver t1 → liberar espaço
        ledger.record_resolution(t1, 1)

        # Agora permitido (100+100 = 200 <= 200)
        ok, _ = rm.can_open_trade(
            {"market_id": "m1", "category": "P", "final_size": 100}, ledger
        )
        assert ok is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
