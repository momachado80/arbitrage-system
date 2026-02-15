"""
RuntimeInvariantsEngine — Verificação de Invariantes Financeiros
==================================================================

Checks em runtime que garantem consistência do estado financeiro.
Em caso de violação: fail closed (bloqueia BUY, não mata processo).

INVARIANTES:
Capital:
  - total_capital >= 0
  - available_capital >= 0
  - locked_in_open_orders >= 0
  - locked_in_unresolved_markets >= 0
  - available + locked_open + locked_unresolved ≈ total_capital

Reservas vs Ledger:
  - soma estimada de "notional reservado" no ledger BUY trades
    ≈ capital_manager.locked_in_open_orders (tolerância conservadora)
  - falha neste check escala para SAFE_MODE

Positions:
  - shares >= 0 para toda posição
  - avg_entry_price >= 0
  - sem NaN/inf

Ledger:
  - cumulative_filled_size >= 0
  - final_size >= 0
  - cumulative_filled_size <= final_size + eps
  - FILLED → abs(cumulative - final) <= eps
  - weighted_avg_price >= 0
  - sem NaN/inf

STREAKS:
  - check OK → safety_state.mark_invariants_ok()
  - check FAIL → safety_state.mark_invariants_fail()
  - auto clear via safety_state.maybe_clear_invariant_fail()
"""

import logging
import math
from typing import Any, Optional

from src.safety.safety_state import SafetyState, INVARIANT_FAIL

logger = logging.getLogger(__name__)

# Tolerância relativa para soma de capital
CAPITAL_SUM_REL_TOL: float = 1e-6
CAPITAL_SUM_ABS_TOL: float = 1e-8
# Tolerância para fills
FILL_EPS: float = 1e-8
# Tolerância para comparação reservas vs ledger
RESERVE_REL_TOL: float = 0.01  # 1%
RESERVE_ABS_TOL: float = 0.1   # $0.10

# Statuses que representam ordens BUY com capital reservado
_BUY_RESERVED_STATUSES = {
    "CAPITAL_RESERVED", "ORDER_SUBMITTED", "OPEN", "PARTIALLY_FILLED",
}


def _is_finite(x: float) -> bool:
    """Verifica se float é finito (não NaN, não inf)."""
    return not (math.isnan(x) or math.isinf(x))


class RuntimeInvariantsEngine:
    """
    Motor de verificação de invariantes financeiros em runtime.

    Não lança exceção. Em caso de violação: bloqueia BUY via SafetyState.
    """

    def __init__(
        self,
        capital_manager: Any,
        position_manager: Any,
        trade_ledger: Any,
        safety_state: SafetyState,
    ) -> None:
        self._capital = capital_manager
        self._positions = position_manager
        self._ledger = trade_ledger
        self._safety = safety_state

    def check_all(self, context: str) -> None:
        """
        Executa todos os checks de invariantes.

        Args:
            context: descrição do ponto de chamada (ex: "after_fill_apply")
        """
        failed = False

        if not self._check_capital(context):
            failed = True
        if not self._check_positions(context):
            failed = True
        if not self._check_ledger(context):
            failed = True

        reserve_ok = self._check_reserves(context)
        if not reserve_ok:
            failed = True

        if failed:
            self._safety.block_buy(INVARIANT_FAIL)
            self._safety.increment_invariant_fail()
            self._safety.mark_invariants_fail()
            # Escalada para SAFE_MODE se reservas inconsistentes (phantom capital)
            if not reserve_ok:
                self._safety.enable_safe_mode(INVARIANT_FAIL)
        else:
            self._safety.mark_invariants_ok()
            self._safety.maybe_clear_invariant_fail()

    # ------------------------------------------------------------------
    # Capital invariants
    # ------------------------------------------------------------------

    def _check_capital(self, context: str) -> bool:
        """Retorna True se todos os invariantes de capital estão OK."""
        if self._capital is None:
            return True

        try:
            snap = self._capital.snapshot()
        except Exception as e:
            self._log_fail(context, f"capital snapshot falhou: {e}")
            return False

        total = snap.get("total_capital", 0)
        avail = snap.get("available_capital", 0)
        locked_open = snap.get("locked_in_open_orders", 0)
        locked_unresolved = snap.get("locked_in_unresolved_markets", 0)

        ok = True

        if total < -CAPITAL_SUM_ABS_TOL:
            self._log_fail(context, f"total_capital negativo: {total}")
            ok = False

        if avail < -CAPITAL_SUM_ABS_TOL:
            self._log_fail(context, f"available_capital negativo: {avail}")
            ok = False

        if locked_open < -CAPITAL_SUM_ABS_TOL:
            self._log_fail(context, f"locked_in_open_orders negativo: {locked_open}")
            ok = False

        if locked_unresolved < -CAPITAL_SUM_ABS_TOL:
            self._log_fail(
                context,
                f"locked_in_unresolved_markets negativo: {locked_unresolved}",
            )
            ok = False

        # Soma: available + locked_open + locked_unresolved ≈ total
        expected_sum = avail + locked_open + locked_unresolved
        tolerance = CAPITAL_SUM_REL_TOL * abs(total) + CAPITAL_SUM_ABS_TOL
        if abs(expected_sum - total) > tolerance:
            self._log_fail(
                context,
                f"capital soma inconsistente: "
                f"avail({avail}) + open({locked_open}) + "
                f"unresolved({locked_unresolved}) = {expected_sum} "
                f"vs total({total}), tol={tolerance}",
            )
            ok = False

        # NaN/inf check
        for name, val in [
            ("total", total), ("avail", avail),
            ("locked_open", locked_open), ("locked_unresolved", locked_unresolved),
        ]:
            if not _is_finite(val):
                self._log_fail(context, f"capital {name} não-finito: {val}")
                ok = False

        return ok

    # ------------------------------------------------------------------
    # Reserves vs Ledger invariant
    # ------------------------------------------------------------------

    def _check_reserves(self, context: str) -> bool:
        """
        Verifica se locked_in_open_orders ≈ soma de reservas estimadas no ledger.

        Evita "phantom capital" — capital reservado sem trade correspondente
        ou vice-versa.
        """
        if self._capital is None or self._ledger is None:
            return True

        try:
            snap = self._capital.snapshot()
            all_trades = self._ledger.get_all_trades()
        except Exception as e:
            self._log_fail(context, f"reserves check read falhou: {e}")
            return False

        locked_open = snap.get("locked_in_open_orders", 0.0)

        # Estimar notional reservado no ledger para BUY trades
        ledger_reserved = 0.0
        for tid, td in all_trades.items():
            status = td.get("status", "")
            action = td.get("action", "BUY")
            # Apenas BUY trades com capital reservado
            if action == "SELL":
                continue
            if status not in _BUY_RESERVED_STATUSES:
                continue
            final_size = td.get("final_size", 0.0)
            cum_filled = td.get("cumulative_filled_size", 0.0)
            remaining = max(0.0, final_size - cum_filled)
            ledger_reserved += remaining

        # Comparar com tolerância conservadora
        tolerance = max(
            RESERVE_REL_TOL * max(abs(locked_open), abs(ledger_reserved)),
            RESERVE_ABS_TOL,
        )
        deviation = abs(locked_open - ledger_reserved)
        if deviation > tolerance:
            self._log_fail(
                context,
                f"reserves mismatch: capital.locked_open={locked_open:.4f} "
                f"vs ledger_reserved={ledger_reserved:.4f} "
                f"dev={deviation:.4f} tol={tolerance:.4f}",
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Position invariants
    # ------------------------------------------------------------------

    def _check_positions(self, context: str) -> bool:
        """Retorna True se todos os invariantes de posição estão OK."""
        if self._positions is None:
            return True

        try:
            all_pos = self._positions.get_all_positions()
        except Exception as e:
            self._log_fail(context, f"positions read falhou: {e}")
            return False

        ok = True
        for key, pos in all_pos.items():
            shares = getattr(pos, "shares", 0.0)
            avg_price = getattr(pos, "avg_entry_price", 0.0)

            if not _is_finite(shares):
                self._log_fail(context, f"pos {key} shares NaN/inf: {shares}")
                ok = False
            elif shares < -FILL_EPS:
                self._log_fail(context, f"pos {key} shares negativo: {shares}")
                ok = False

            if not _is_finite(avg_price):
                self._log_fail(
                    context, f"pos {key} avg_entry_price NaN/inf: {avg_price}"
                )
                ok = False
            elif avg_price < -FILL_EPS:
                self._log_fail(
                    context, f"pos {key} avg_entry_price negativo: {avg_price}"
                )
                ok = False

        return ok

    # ------------------------------------------------------------------
    # Ledger invariants
    # ------------------------------------------------------------------

    def _check_ledger(self, context: str) -> bool:
        """Retorna True se todos os invariantes do ledger estão OK."""
        if self._ledger is None:
            return True

        try:
            all_trades = self._ledger.get_all_trades()
        except Exception as e:
            self._log_fail(context, f"ledger read falhou: {e}")
            return False

        ok = True
        for tid, td in all_trades.items():
            cum = td.get("cumulative_filled_size", 0.0)
            final = td.get("final_size", 0.0)
            avg_p = td.get("weighted_avg_price", 0.0)
            status = td.get("status", "")

            # Valores finitos
            for name, val in [
                ("cumulative_filled", cum),
                ("final_size", final),
                ("weighted_avg_price", avg_p),
            ]:
                if not _is_finite(val):
                    self._log_fail(
                        context, f"trade {tid} {name} NaN/inf: {val}"
                    )
                    ok = False

            # Não-negatividade
            if cum < -FILL_EPS:
                self._log_fail(
                    context, f"trade {tid} cumulative_filled negativo: {cum}"
                )
                ok = False

            if final < -FILL_EPS:
                self._log_fail(
                    context, f"trade {tid} final_size negativo: {final}"
                )
                ok = False

            if avg_p < -FILL_EPS:
                self._log_fail(
                    context,
                    f"trade {tid} weighted_avg_price negativo: {avg_p}",
                )
                ok = False

            # cumulative <= final + eps
            if cum > final + FILL_EPS:
                self._log_fail(
                    context,
                    f"trade {tid} cumulative({cum}) > final({final})",
                )
                ok = False

            # FILLED → cumulative ≈ final
            if status == "FILLED":
                if abs(cum - final) > FILL_EPS:
                    self._log_fail(
                        context,
                        f"trade {tid} FILLED mas cum({cum}) != final({final})",
                    )
                    ok = False

        return ok

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_fail(context: str, detail: str) -> None:
        """Log invariant failure."""
        logger.error(f"[SAFETY:INVARIANT_FAIL] [{context}] {detail}")
