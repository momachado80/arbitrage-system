"""
Order Reconciler — Agente da Verdade
=====================================

Garante que o estado local (OrderManager + TradeLedger) reflita a verdade
da API externa. O reconciler é idempotente: rodar múltiplas vezes não
duplica fills nem altera avg_price incorretamente.

FLUXO:
Para cada trade OPEN no ledger:
  1. Para cada order_id associado, consultar exchange API
  2. Sincronizar estado da ordem no OrderManager
  3. Acumular fills reais (validados + dedup + anti-replay)
  4. Atualizar ledger com cumulative_filled e weighted_avg_price
  5. Se tudo preenchido → FILLED
  6. Se todas ordens terminais e nada preenchido → CANCELLED

CONTRATO DE FILL:
Cada fill DEVE conter:
  fill_id:   str (não vazio, identificador único)
  price:     float
  size:      float
  timestamp: str ISO-8601 ou epoch numérico

Se fill_id ou timestamp ausentes → [RECONCILE:INVALID_FILL] + safety mode.

SENSOR ANTI-REPLAY:
Após rolling window (500 fill_ids), fill_ids antigos podem ser expulsos.
Sensor temporal por trade (last_fill_ts) detecta replays:
  fill_ts < last_fill_ts - REPLAY_SKEW_SECONDS → fill ignorado.

REGRA DE OURO:
O estado local SEMPRE converge para o estado da API.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Tolerância para out-of-order pequeno (segundos)
REPLAY_SKEW_SECONDS: float = 5.0


def _parse_fill_timestamp(ts: Any) -> Optional[float]:
    """
    Converte timestamp de fill (ISO-8601 ou epoch) para epoch seconds.

    Returns None se não for possível parsear.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        # Tentar epoch string
        try:
            return float(ts)
        except ValueError:
            pass
        # Tentar ISO-8601
        try:
            dt = datetime.fromisoformat(ts)
            return dt.timestamp()
        except (ValueError, TypeError):
            return None
    return None


class ExchangeAPI(ABC):
    """Interface da API externa (mockável para testes)."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Returns: {
            "order_id": str,
            "status": "PENDING"|"PARTIALLY_FILLED"|"FILLED"|"CANCELLED"|"EXPIRED",
            "filled_size": float,
        }
        """

    @abstractmethod
    def get_fills(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Retorna lista de fills para a ordem.

        CONTRATO: cada fill DEVE conter:
            fill_id:   str  (identificador único, não vazio)
            price:     float
            size:      float
            timestamp: str  (ISO-8601 com timezone ou epoch numérico)

        Fills sem fill_id ou timestamp serão rejeitados pelo reconciler
        e ativarão modo de segurança.
        """

    @abstractmethod
    def get_position(self, market_id: str) -> Dict[str, Any]:
        """
        Returns: {
            "market_id": str,
            "size": float,
            "avg_entry": float,
        }
        """

    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """
        Retorna resumo da conta na exchange (opcional).

        Returns: {
            "total_balance": float,
        } ou None se endpoint não disponível.

        TODO: integrar com cliente real da Polymarket.
        """
        return None  # Default: não disponível


class OrderReconciler:
    """Agente da verdade: sincroniza estado local com exchange."""

    TERMINAL_STATUSES = {"FILLED", "CANCELLED", "EXPIRED"}

    def __init__(self) -> None:
        self._replay_warned: set = set()  # trade_ids que já emitiram warning
        self._reconcile_session_id: int = 0

    def reconcile_open_trades(
        self,
        order_manager: Any,
        trade_ledger: Any,
        exchange_api: ExchangeAPI,
        capital_manager: Optional[Any] = None,
        position_manager: Optional[Any] = None,
        safety_state: Optional[Any] = None,
        invariants_engine: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        Reconcilia todos os trades OPEN do ledger com a exchange.

        capital_manager: opcional, atualiza contabilidade de capital.
        position_manager: opcional, atualiza posições em fills (spot-only).
        Retorna dict {trade_id: ação tomada} para auditoria.
        """
        # Incrementar session_id monotônico
        self._reconcile_session_id += 1
        session_id = self._reconcile_session_id

        open_trades = trade_ledger.get_open_trades()
        actions: Dict[str, str] = {}

        for trade_id, trade_data in open_trades.items():
            order_ids = trade_data.get("order_ids", [])
            if not order_ids:
                actions[trade_id] = "SKIPPED_NO_ORDERS"
                continue

            action = self._reconcile_trade(
                trade_id, trade_data, order_ids,
                order_manager, trade_ledger, exchange_api,
                capital_manager, position_manager,
            )
            actions[trade_id] = action

            # Invariants check após cada fill apply
            if invariants_engine is not None and action in (
                "FULLY_FILLED", "IN_PROGRESS", "PARTIAL_TERMINAL",
            ):
                invariants_engine.check_all("after_fill_apply")

        # Reconcile OK: atualizar safety_state
        if safety_state is not None:
            import time as _time
            safety_state.set_last_reconcile_ok(_time.time())
            safety_state.set_reconcile_session_id(session_id)
            safety_state.clear_reason("RECONCILE_STALE")
            safety_state.clear_trade_drop_flag()

        # Capital mismatch check (Gate C)
        if (
            safety_state is not None
            and capital_manager is not None
            and exchange_api is not None
        ):
            self._check_capital_mismatch(
                exchange_api, capital_manager, safety_state,
            )

        # Invariants check após reconcile completo
        if invariants_engine is not None:
            invariants_engine.check_all("after_reconcile_ok")

        return actions

    def _check_capital_mismatch(
        self,
        exchange_api: ExchangeAPI,
        capital_manager: Any,
        safety_state: Any,
    ) -> None:
        """Gate C: verifica mismatch de capital com fonte externa."""
        try:
            summary = exchange_api.get_account_summary()
        except Exception:
            return  # Endpoint não disponível ou erro — não bloquear

        if summary is None:
            return  # Endpoint não implementado

        ext_balance = summary.get("total_balance")
        if ext_balance is None:
            return

        local_total = capital_manager.total_capital
        tolerance = max(0.001 * abs(local_total), 0.01)
        deviation = abs(ext_balance - local_total)

        if deviation > tolerance:
            import time as _time
            safety_state.block_buy("CAPITAL_MISMATCH")
            safety_state.mark_capital_mismatch(_time.time())
            safety_state.mark_capital_mismatch_fail()
            safety_state.enable_safe_mode("CAPITAL_MISMATCH")
            logger.warning(
                f"[RECONCILE:CAPITAL_MISMATCH] "
                f"local={local_total:.2f} external={ext_balance:.2f} "
                f"dev={deviation:.4f} tol={tolerance:.4f}"
            )
        else:
            safety_state.mark_capital_mismatch_ok()
            safety_state.maybe_clear_capital_mismatch()

    def _reconcile_trade(
        self,
        trade_id: str,
        trade_data: Dict[str, Any],
        order_ids: List[str],
        order_manager: Any,
        trade_ledger: Any,
        exchange_api: ExchangeAPI,
        capital_manager: Optional[Any] = None,
        position_manager: Optional[Any] = None,
    ) -> str:
        total_filled = 0.0
        total_cost = 0.0
        all_terminal = True
        seen_fids: set = set()  # Dedup por fill_id dentro deste reconcile

        for oid in order_ids:
            try:
                api_status = exchange_api.get_order_status(oid)
                fills = exchange_api.get_fills(oid)
            except Exception as e:
                logger.error(f"[RECONCILE:API_ERROR] order={oid}: {e}")
                all_terminal = False
                continue

            # Sincronizar ordem no OrderManager
            order_manager.sync_order_status(
                oid,
                api_status.get("status", "PENDING"),
                api_status.get("filled_size", 0.0),
            )

            # Acumular fills — validação + dedup + anti-replay
            for fill in fills:
                fid = fill.get("fill_id")

                # --- Validação: fill_id obrigatório ---
                if not fid or not isinstance(fid, str) or not fid.strip():
                    logger.error(
                        f"[RECONCILE:INVALID_FILL] fill sem fill_id válido: "
                        f"order={oid} fill={fill}"
                    )
                    trade_ledger.mark_reconcile_failed()
                    continue

                # --- Validação: timestamp obrigatório ---
                fill_ts = _parse_fill_timestamp(fill.get("timestamp"))
                if fill_ts is None:
                    logger.error(
                        f"[RECONCILE:INVALID_FILL] fill sem timestamp válido: "
                        f"fill_id={fid}"
                    )
                    trade_ledger.mark_reconcile_failed()
                    continue

                # --- Dedup por fill_id dentro deste reconcile ---
                if fid in seen_fids:
                    continue
                seen_fids.add(fid)

                # --- Sensor anti-replay ---
                last_ts = trade_ledger.get_last_fill_ts(trade_id)
                if last_ts is not None and fill_ts < last_ts - REPLAY_SKEW_SECONDS:
                    if trade_id not in self._replay_warned:
                        logger.warning(
                            f"[RECONCILE:REPLAY_SUSPECT] trade={trade_id} "
                            f"fill_id={fid} fill_ts={fill_ts:.0f} "
                            f"last_ts={last_ts:.0f}"
                        )
                        self._replay_warned.add(trade_id)
                    continue  # Ignorar fill suspeito de replay

                # --- Acumular fill válido ---
                trade_ledger.mark_fill_applied(trade_id, fid)
                trade_ledger.update_last_fill_ts(trade_id, fill_ts)
                fill_size = fill.get("size", 0.0)
                fill_price = fill.get("price", 0.0)
                total_cost += fill_size * fill_price
                total_filled += fill_size

            # Verificar se ordem ainda está ativa
            if api_status.get("status", "PENDING") not in self.TERMINAL_STATUSES:
                all_terminal = False

        # Calcular avg price
        avg_price = total_cost / total_filled if total_filled > 0 else 0.0

        # --- Position tracking (spot-only) ---
        # Calcular delta de shares preenchidas desde último reconcile
        if position_manager is not None:
            outcome_side = trade_data.get("outcome_side")
            trade_action = trade_data.get("action", "BUY")
            trade_shares = trade_data.get("shares")
            final_size = trade_data.get("final_size", 0)
            prev_filled = trade_data.get("cumulative_filled_size", 0)

            if outcome_side and trade_shares and final_size > 0:
                delta_notional = total_filled - prev_filled
                if delta_notional > 0:
                    # Converter delta de notional para shares proporcionalmente
                    delta_shares = trade_shares * (delta_notional / final_size)
                    fill_price_for_pos = avg_price if avg_price > 0 else trade_data.get("entry_price", 0)

                    # Para SELL: capturar avg_entry ANTES de aplicar
                    sell_avg_entry = 0.0
                    if trade_action == "SELL" and capital_manager is not None:
                        sell_avg_entry = position_manager.get_position(
                            trade_data["market_id"], outcome_side,
                        ).avg_entry_price

                    # Aplicar fill à posição
                    try:
                        position_manager.apply_fill(
                            trade_data["market_id"], outcome_side,
                            delta_shares, fill_price_for_pos, trade_action,
                        )
                    except Exception as e:
                        logger.error(f"[RECONCILE:POSITION_ERROR] {trade_id}: {e}")

                    # Para SELL: processar capital (liberar cost_basis + PnL)
                    if trade_action == "SELL" and capital_manager is not None and sell_avg_entry > 0:
                        cost_basis = delta_shares * sell_avg_entry
                        proceeds = delta_shares * fill_price_for_pos
                        pnl = proceeds - cost_basis
                        capital_manager.process_sell_fill(cost_basis, pnl)

        # Atualizar ledger (internamente chama capital_manager.move_to_unresolved se FILLED)
        trade_ledger.update_fill_status(trade_id, total_filled, avg_price)

        final_size = trade_data.get("final_size", 0)

        # Decidir status (com tolerância FILL_EPS)
        from src.utils.precision import FILL_EPS
        if total_filled + FILL_EPS >= final_size:
            # Capital: garantir movimentação para unresolved (idempotente)
            if capital_manager is not None:
                trade_action = trade_data.get("action", "BUY")
                if trade_action != "SELL":
                    capital_manager.move_to_unresolved(trade_id, final_size)
            logger.info(
                f"[RECONCILE:FULLY_FILLED] trade={trade_id} "
                f"filled={total_filled} avg={avg_price:.4f}"
            )
            return "FULLY_FILLED"

        if all_terminal and total_filled == 0:
            # Capital: liberar reserva (idempotente, ledger.cancel_trade também libera)
            if capital_manager is not None:
                capital_manager.release_order_reserve(trade_id)
            trade_ledger.cancel_trade(trade_id)
            logger.info(f"[RECONCILE:CANCELLED] trade={trade_id} (all orders terminal, no fills)")
            return "CANCELLED"

        if all_terminal and total_filled > 0:
            logger.warning(
                f"[RECONCILE:PARTIAL_TERMINAL] trade={trade_id} "
                f"filled={total_filled}/{final_size} (ordens terminais, fill incompleto)"
            )
            return "PARTIAL_TERMINAL"

        return "IN_PROGRESS"
