"""
Basket Simulator
================

Simulate execution of multi-leg arbitrage baskets.
Uses SimConfig and slippage/fee models from the execution simulator.
Deterministic given seed.

Pure functions, no globals, no side effects.
"""

import random
import time
from typing import Any, Dict, List, Optional


def simulate_basket(
    basket: Dict[str, Any],
    cfg_dict: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate execution of all legs in a basket.

    cfg_dict keys used:
      fee_taker_bps, slippage_from_spread_mult, slippage_bps_floor,
      lat_signal_to_order_ms, lat_order_to_exchange_ms, lat_exchange_to_ack_ms

    Returns dict with per-leg results and aggregate metrics.
    """
    rng = random.Random(seed)
    trades = basket.get("trades", [])
    edge_bps = basket.get("edge", 0) * 10000.0

    if not trades:
        return _empty_result(basket)

    fee_bps = cfg_dict.get("fee_taker_bps", 0.0)
    spread_mult = cfg_dict.get("slippage_from_spread_mult", 0.5)
    slip_floor = cfg_dict.get("slippage_bps_floor", 2.0)
    total_lat_ms = (
        cfg_dict.get("lat_signal_to_order_ms", 50)
        + cfg_dict.get("lat_order_to_exchange_ms", 150)
        + cfg_dict.get("lat_exchange_to_ack_ms", 50)
    )

    leg_results: List[Dict[str, Any]] = []
    total_pnl = 0.0
    total_slip = 0.0
    total_fees = 0.0
    filled_count = 0

    for trade in trades:
        bid = trade.get("best_bid")
        ask = trade.get("best_ask")
        prob = trade.get("prob", 0)
        size = trade.get("size", 0)
        side = trade.get("side", "SELL")
        market_id = trade.get("market_id", "")

        spread_bps = 0.0
        if bid is not None and ask is not None and ask > 0:
            spread_bps = (ask - bid) / ask * 10000.0

        slip = max(spread_bps * spread_mult, slip_floor)
        leg_edge = edge_bps * trade.get("weight", 1.0 / len(trades))
        leg_pnl = leg_edge - slip - fee_bps

        fill_px = bid if side == "SELL" else ask
        if fill_px is None:
            fill_px = prob

        filled_count += 1
        total_pnl += leg_pnl
        total_slip += slip
        total_fees += fee_bps

        leg_results.append({
            "market_id": market_id,
            "side": side,
            "size": size,
            "fill_px": round(fill_px, 6) if fill_px else 0,
            "slippage_bps": round(slip, 2),
            "fees_bps": round(fee_bps, 2),
            "pnl_bps": round(leg_pnl, 2),
            "filled": True,
        })

    n = len(trades)
    fill_rate = filled_count / n if n > 0 else 0.0
    alpha_capture = total_pnl / edge_bps if edge_bps > 0 else 0.0

    return {
        "basket_id": basket.get("basket_id", ""),
        "type": basket.get("type", ""),
        "edge_bps": round(edge_bps, 2),
        "n_legs": n,
        "legs": leg_results,
        "basket_expected_pnl": round(total_pnl, 2),
        "basket_fill_rate": round(fill_rate, 4),
        "basket_alpha_capture_ratio": round(alpha_capture, 4),
        "total_slippage_bps": round(total_slip, 2),
        "total_fees_bps": round(total_fees, 2),
        "latency_ms": total_lat_ms,
        "seed": seed,
    }


def simulate_baskets(
    baskets: List[Dict[str, Any]],
    cfg_dict: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate a list of baskets and return individual + aggregate results."""
    results = []
    for i, basket in enumerate(baskets):
        r = simulate_basket(basket, cfg_dict, seed=seed + i)
        results.append(r)

    return {
        "basket_simulations": results,
        "basket_summary": _build_basket_summary(results),
    }


def _build_basket_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total_baskets": 0,
            "mean_pnl_bps": 0.0,
            "mean_fill_rate": 0.0,
            "mean_alpha_capture": 0.0,
            "total_pnl_bps": 0.0,
            "profitable_baskets": 0,
            "profitable_pct": 0.0,
        }

    pnls = [r.get("basket_expected_pnl", 0) for r in results]
    fills = [r.get("basket_fill_rate", 0) for r in results]
    alphas = [r.get("basket_alpha_capture_ratio", 0) for r in results]
    profitable = sum(1 for p in pnls if p > 0)
    n = len(results)

    return {
        "total_baskets": n,
        "mean_pnl_bps": round(sum(pnls) / n, 2),
        "mean_fill_rate": round(sum(fills) / n, 4),
        "mean_alpha_capture": round(sum(alphas) / n, 4),
        "total_pnl_bps": round(sum(pnls), 2),
        "profitable_baskets": profitable,
        "profitable_pct": round(profitable / n, 4) if n > 0 else 0.0,
    }


def rank_baskets(results: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    """Rank simulated baskets by expected PnL, return top N."""
    ranked = sorted(results, key=lambda r: -r.get("basket_expected_pnl", 0))
    return [
        {
            "basket_id": r.get("basket_id", ""),
            "type": r.get("type", ""),
            "edge_bps": r.get("edge_bps", 0),
            "basket_expected_pnl": r.get("basket_expected_pnl", 0),
            "basket_fill_rate": r.get("basket_fill_rate", 0),
            "basket_alpha_capture_ratio": r.get("basket_alpha_capture_ratio", 0),
            "n_legs": r.get("n_legs", 0),
        }
        for r in ranked[:top_n]
    ]


def _empty_result(basket: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "basket_id": basket.get("basket_id", ""),
        "type": basket.get("type", ""),
        "edge_bps": 0,
        "n_legs": 0,
        "legs": [],
        "basket_expected_pnl": 0.0,
        "basket_fill_rate": 0.0,
        "basket_alpha_capture_ratio": 0.0,
        "total_slippage_bps": 0.0,
        "total_fees_bps": 0.0,
        "latency_ms": 0.0,
        "seed": 42,
    }
