"""
Audit Report — Relatório de Estado para Incidentes
====================================================

Gera relatório human-readable do estado financeiro e de segurança.
Ferramenta de incidente — NÃO para hot path.

Logado automaticamente ao entrar em SAFE_MODE e no shutdown.
"""

import time
from collections import Counter
from typing import Any, Optional


def build_audit_report(
    capital_manager: Any,
    position_manager: Any,
    trade_ledger: Any,
    safety_state: Any,
) -> str:
    """
    Gera relatório de auditoria human-readable.

    Inclui:
    - buy_blocked + reasons
    - reconcile age e session_id
    - capital breakdown
    - contagem de trades por status
    - top exposures por market
    - streaks e safe_mode
    """
    lines: list = []
    _sep = "=" * 60

    lines.append(_sep)
    lines.append("  AUDIT REPORT — SAFETY & FINANCIAL STATE")
    lines.append(_sep)
    lines.append("")

    # --- Safety State ---
    lines.append("--- SAFETY STATE ---")
    if safety_state is not None:
        snap = safety_state.snapshot()
        lines.append(f"  buy_blocked:          {snap.get('buy_blocked', '?')}")
        lines.append(
            f"  reasons:              {snap.get('buy_block_reasons', [])}"
        )
        lines.append(
            f"  global_safe_mode:     {snap.get('global_safe_mode', False)}"
        )
        rec_ts = snap.get("last_reconcile_ok_ts")
        if rec_ts is not None:
            age = time.time() - rec_ts
            lines.append(f"  last_reconcile_age:   {age:.1f}s")
        else:
            lines.append("  last_reconcile_age:   NEVER")
        lines.append(
            f"  reconcile_session_id: "
            f"{snap.get('last_reconcile_session_id', 0)}"
        )
        lines.append(
            f"  invariant_fail_count: "
            f"{snap.get('invariant_fail_count', 0)}"
        )
        lines.append(
            f"  invariants_ok_streak: "
            f"{snap.get('invariants_ok_streak', 0)}"
        )
        lines.append(
            f"  cap_mismatch_streak:  "
            f"{snap.get('capital_mismatch_ok_streak', 0)}"
        )
        lines.append(
            f"  trade_drop_flag:      "
            f"{snap.get('trade_drop_since_last_reconcile', False)}"
        )
    else:
        lines.append("  (safety_state not available)")
    lines.append("")

    # --- Capital ---
    lines.append("--- CAPITAL ---")
    if capital_manager is not None:
        try:
            cs = capital_manager.snapshot()
            lines.append(f"  total_capital:        {cs['total_capital']:.2f}")
            lines.append(
                f"  available_capital:    {cs['available_capital']:.2f}"
            )
            lines.append(
                f"  locked_open_orders:   {cs['locked_in_open_orders']:.2f}"
            )
            lines.append(
                f"  locked_unresolved:    "
                f"{cs['locked_in_unresolved_markets']:.2f}"
            )
            lines.append(f"  realized_pnl:         {cs['realized_pnl']:.2f}")
            lines.append(
                f"  open_order_count:     {cs['open_order_count']}"
            )
            lines.append(
                f"  unresolved_count:     {cs['unresolved_count']}"
            )
        except Exception as e:
            lines.append(f"  ERROR: {e}")
    else:
        lines.append("  (capital_manager not available)")
    lines.append("")

    # --- Trades por status ---
    lines.append("--- TRADES BY STATUS ---")
    if trade_ledger is not None:
        try:
            all_trades = trade_ledger.get_all_trades()
            status_counts: Counter = Counter()
            market_exposure: Counter = Counter()
            for tid, td in all_trades.items():
                status_counts[td.get("status", "UNKNOWN")] += 1
                # Exposição ativa por mercado
                if td.get("status") in (
                    "OPEN", "PARTIALLY_FILLED", "FILLED",
                    "ORDER_SUBMITTED", "CAPITAL_RESERVED",
                ):
                    market_exposure[td.get("market_id", "?")] += td.get(
                        "final_size", 0
                    )
            for status, count in sorted(status_counts.items()):
                lines.append(f"  {status}: {count}")
            lines.append(f"  TOTAL: {len(all_trades)}")

            # Top exposures
            if market_exposure:
                lines.append("")
                lines.append("--- TOP MARKET EXPOSURES ---")
                top10 = market_exposure.most_common(10)
                for mkt, exp in top10:
                    lines.append(f"  {mkt}: {exp:.2f}")
        except Exception as e:
            lines.append(f"  ERROR: {e}")
    else:
        lines.append("  (trade_ledger not available)")
    lines.append("")

    # --- Positions ---
    lines.append("--- POSITIONS ---")
    if position_manager is not None:
        try:
            all_pos = position_manager.get_all_positions()
            if not all_pos:
                lines.append("  (no positions)")
            for key, pos in all_pos.items():
                shares = getattr(pos, "shares", 0)
                avg = getattr(pos, "avg_entry_price", 0)
                lines.append(f"  {key}: {shares:.4f} shares @ {avg:.4f}")
        except Exception as e:
            lines.append(f"  ERROR: {e}")
    else:
        lines.append("  (position_manager not available)")

    lines.append("")
    lines.append(_sep)
    lines.append(f"  Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append(_sep)

    return "\n".join(lines)
