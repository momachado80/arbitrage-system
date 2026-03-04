"""
Metrics Engine — Advanced simulation analytics.
No numpy dependency; uses pure Python for percentile calculations.
"""

from typing import Any, Dict, List, Optional


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    n = len(sorted_vals)
    if n == 0:
        return None
    idx = min(n - 1, max(0, int(n * p / 100.0)))
    return round(sorted_vals[idx], 4)


class MetricsEngine:
    """Compute advanced metrics over a list of trade dicts."""

    def compute(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            return self._empty()

        filled = [t for t in trades if t.get("fill_qty", 0) > 0]
        total = len(trades)

        if not filled:
            return {
                **self._empty(),
                "total_trades": total,
                "skipped": total,
            }

        pnls = sorted(t.get("pnl_bps", 0) for t in filled)
        slips = sorted(t.get("slippage_bps", 0) for t in filled)
        edges = [t.get("_expected_edge_bps", 0) for t in filled if t.get("_expected_edge_bps")]

        total_pnl = sum(pnls)
        total_edge = sum(edges) if edges else None

        alpha_capture = (
            round(total_pnl / total_edge, 4) if total_edge and total_edge > 0 else None
        )

        # Estimate pnl per hour using time span
        ts_vals = []
        for t in filled:
            detect = t.get("detect_time_utc", "")
            if detect:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(detect.replace("Z", "+00:00"))
                    ts_vals.append(dt.timestamp())
                except Exception:
                    pass

        pnl_per_hour = None
        if len(ts_vals) >= 2:
            span_hours = (max(ts_vals) - min(ts_vals)) / 3600.0
            if span_hours > 0:
                pnl_per_hour = round(total_pnl / span_hours, 2)

        return {
            "total_trades": total,
            "filled": len(filled),
            "skipped": total - len(filled),
            "fill_rate": round(len(filled) / total, 4) if total > 0 else 0.0,
            "pnl_bps_p50": _percentile(pnls, 50),
            "pnl_bps_p90": _percentile(pnls, 90),
            "pnl_bps_p99": _percentile(pnls, 99),
            "slippage_bps_p50": _percentile(slips, 50),
            "slippage_bps_p90": _percentile(slips, 90),
            "total_pnl_bps": round(total_pnl, 2),
            "alpha_capture_ratio": alpha_capture,
            "expected_pnl_per_hour": pnl_per_hour,
        }

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {
            "total_trades": 0,
            "filled": 0,
            "skipped": 0,
            "fill_rate": 0.0,
            "pnl_bps_p50": None,
            "pnl_bps_p90": None,
            "pnl_bps_p99": None,
            "slippage_bps_p50": None,
            "slippage_bps_p90": None,
            "total_pnl_bps": 0.0,
            "alpha_capture_ratio": None,
            "expected_pnl_per_hour": None,
        }
