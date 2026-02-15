"""Testes para OrderManager (micro-lots, TTL, toxic flow, fills)."""

import time
import pytest
from src.execution.order_manager import (
    OrderManager, OrderSide, OrderStatus, MAX_CHILD_SIZE,
)


class TestPlaceLimitOrder:
    def test_single_order(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        assert len(ids) == 1
        s = om.check_order_status(ids[0])
        assert s["size"] == 10.0
        assert s["status"] == "PENDING"

    def test_validation_size(self):
        om = OrderManager()
        with pytest.raises(ValueError, match="size"):
            om.place_limit_order("m1", OrderSide.BUY, 0.65, -1.0, 60)
        with pytest.raises(ValueError, match="size"):
            om.place_limit_order("m1", OrderSide.BUY, 0.65, 0.0, 60)

    def test_validation_price(self):
        om = OrderManager()
        with pytest.raises(ValueError, match="price"):
            om.place_limit_order("m1", OrderSide.BUY, 0.0, 10.0, 60)
        with pytest.raises(ValueError, match="price"):
            om.place_limit_order("m1", OrderSide.BUY, 1.0, 10.0, 60)

    def test_validation_ttl(self):
        om = OrderManager()
        with pytest.raises(ValueError, match="ttl"):
            om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 0)


class TestMicroLots:
    def test_split(self):
        om = OrderManager(max_child_size=50.0)
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 120.0, 60)
        assert len(ids) == 3
        sizes = [om.check_order_status(oid)["size"] for oid in ids]
        assert abs(sum(sizes) - 120.0) < 1e-9
        assert max(sizes) <= 50.0

    def test_exact_multiple(self):
        om = OrderManager(max_child_size=50.0)
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 100.0, 60)
        assert len(ids) == 2

    def test_below_max_no_split(self):
        om = OrderManager(max_child_size=50.0)
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 49.0, 60)
        assert len(ids) == 1
        assert om.check_order_status(ids[0])["parent_order_id"] is None

    def test_parent_id_shared(self):
        om = OrderManager(max_child_size=50.0)
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 120.0, 60)
        parents = {om.check_order_status(oid)["parent_order_id"] for oid in ids}
        assert len(parents) == 1 and None not in parents

    def test_total_size_preserved(self):
        om = OrderManager(max_child_size=30.0)
        ids = om.place_limit_order("m1", OrderSide.SELL, 0.50, 97.0, 60)
        assert abs(sum(om.check_order_status(oid)["size"] for oid in ids) - 97.0) < 1e-9


class TestCancelOrder:
    def test_cancel_active(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        assert om.cancel_order(ids[0]) is True
        assert om.check_order_status(ids[0])["status"] == "CANCELLED"

    def test_cancel_nonexistent(self):
        assert OrderManager().cancel_order("fake") is False

    def test_cancel_already_cancelled(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        om.cancel_order(ids[0])
        assert om.cancel_order(ids[0]) is False

    def test_cancel_filled(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        om.simulate_fill(ids[0], 10.0)
        assert om.cancel_order(ids[0]) is False


class TestTTL:
    def test_cancel_stale(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 1)
        time.sleep(1.1)
        assert ids[0] in om.cancel_stale_orders()
        assert om.check_order_status(ids[0])["status"] == "EXPIRED"

    def test_fresh_not_cancelled(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 3600)
        assert len(om.cancel_stale_orders()) == 0
        assert om.check_order_status(ids[0])["status"] == "PENDING"

    def test_mixed_ttl(self):
        om = OrderManager()
        stale = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 1)
        fresh = om.place_limit_order("m2", OrderSide.SELL, 0.50, 10.0, 3600)
        time.sleep(1.1)
        cancelled = om.cancel_stale_orders()
        assert stale[0] in cancelled
        assert fresh[0] not in cancelled


class TestToxicFlow:
    def test_buy_toxic_price_dropped(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        assert om.check_toxic_flow(ids[0], 0.60) is True
        assert om.check_order_status(ids[0])["status"] == "CANCELLED"

    def test_buy_not_toxic(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        assert om.check_toxic_flow(ids[0], 0.70) is False
        assert om.check_order_status(ids[0])["status"] == "PENDING"

    def test_sell_toxic_price_rose(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.SELL, 0.65, 10.0, 60)
        assert om.check_toxic_flow(ids[0], 0.70) is True
        assert om.check_order_status(ids[0])["status"] == "CANCELLED"

    def test_sell_not_toxic(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.SELL, 0.65, 10.0, 60)
        assert om.check_toxic_flow(ids[0], 0.60) is False
        assert om.check_order_status(ids[0])["status"] == "PENDING"

    def test_toxic_on_cancelled(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        om.cancel_order(ids[0])
        assert om.check_toxic_flow(ids[0], 0.50) is False


class TestSimulateFill:
    def test_full_fill(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        assert om.simulate_fill(ids[0], 10.0) is True
        assert om.check_order_status(ids[0])["status"] == "FILLED"

    def test_partial_fill(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        om.simulate_fill(ids[0], 5.0)
        s = om.check_order_status(ids[0])
        assert s["status"] == "PARTIALLY_FILLED"
        assert s["filled_size"] == 5.0

    def test_multiple_partials(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        om.simulate_fill(ids[0], 3.0)
        om.simulate_fill(ids[0], 3.0)
        om.simulate_fill(ids[0], 4.0)
        assert om.check_order_status(ids[0])["status"] == "FILLED"

    def test_overfill_capped(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        om.simulate_fill(ids[0], 15.0)
        assert om.check_order_status(ids[0])["filled_size"] == 10.0


class TestSyncOrderStatus:
    def test_sync_from_api(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        om.sync_order_status(ids[0], "FILLED", 10.0)
        s = om.check_order_status(ids[0])
        assert s["status"] == "FILLED"
        assert s["filled_size"] == 10.0

    def test_sync_corrects_state(self):
        om = OrderManager()
        ids = om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        assert om.check_order_status(ids[0])["status"] == "PENDING"
        om.sync_order_status(ids[0], "CANCELLED", 0.0)
        assert om.check_order_status(ids[0])["status"] == "CANCELLED"


class TestActiveOrders:
    def test_get_active(self):
        om = OrderManager()
        om.place_limit_order("m1", OrderSide.BUY, 0.65, 10.0, 60)
        om.place_limit_order("m2", OrderSide.SELL, 0.50, 10.0, 60)
        ids3 = om.place_limit_order("m3", OrderSide.BUY, 0.70, 10.0, 60)
        om.cancel_order(ids3[0])
        assert len(om.get_active_orders()) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
