"""
Execution Pipeline
==================

Orquestra: calibrator → risk_manager → slippage → capital → order_manager → trade_ledger.

FLUXOS SUPORTADOS:

LEGACY BUY (base_size/price):
  1. k = calibrator.get_sizing_multiplier()
  2. final_size = base_size * k
  3-8. risk → slippage → capital → order → ledger

SPOT BUY (shares/limit_price):
  1. notional = shares * limit_price
  2. k = calibrator.get_sizing_multiplier()
  3. adjusted_notional = notional * k
  4. adjusted_shares = adjusted_notional / limit_price
  5-9. risk → slippage → capital → order → ledger

SPOT SELL:
  1. Validar PositionManager.can_sell()
  2. place_limit_order(SELL, ...)
  3. ledger.record_entry() (sem calibrator, sem capital reservation)

Resolução (quando mercado resolve):
  trade_ledger.record_resolution() → calibrator + capital_manager (automático)
"""

import logging
import uuid
from typing import Dict, Any, Optional

from src.engine.probability_calibrator import (
    ProbabilityCalibrator,
    MarketCategory,
    TradeType,
)
from src.execution.order_manager import OrderManager, OrderSide
from src.engine.trade_ledger import TradeLedger
from src.execution.slippage import estimate_fill_cost
from src.execution.notional import compute_notional

logger = logging.getLogger(__name__)


def _execute_spot_sell(
    decision: Dict[str, Any],
    order_manager: OrderManager,
    trade_ledger: TradeLedger,
    position_manager: Optional[Any],
) -> Optional[str]:
    """
    Executa SELL spot-only: reduz/encerra posição existente.

    Não aplica calibrator k. Não reserva capital. Não passa por risk check.
    """
    market_id = decision["market_id"]
    outcome_side = decision["outcome_side"]
    shares = decision["shares"]
    limit_price = decision["limit_price"]
    ttl_seconds = decision.get("ttl_seconds", 60)

    # 1. Validar posição
    if position_manager is not None:
        oc_str = outcome_side.value if hasattr(outcome_side, "value") else str(outcome_side)
        if not position_manager.can_sell(market_id, oc_str, shares):
            try:
                from src.metrics import record_decision_reason
                record_decision_reason("rejected_by_execution")
            except ImportError:
                pass
            logger.warning(
                "[PIPELINE] [SELL_REJECTED] Posição insuficiente para vender %s shares %s em %s",
                shares, oc_str, market_id,
            )
            return None
    else:
        oc_str = outcome_side.value if hasattr(outcome_side, "value") else str(outcome_side)

    notional = compute_notional(shares, limit_price)

    # 2. Colocar ordem SELL
    order_ids = order_manager.place_limit_order(
        market_id=market_id, side=OrderSide.SELL,
        price=limit_price, size=notional,
        ttl_seconds=ttl_seconds,
    )

    # 3. Registrar no ledger
    trade_id = str(uuid.uuid4())
    category = decision.get("category", "")
    trade_type = decision.get("trade_type", "")
    cat_str = category.value if hasattr(category, "value") else str(category)
    tt_str = trade_type.value if hasattr(trade_type, "value") else str(trade_type)

    trade_ledger.record_entry(
        trade_id=trade_id,
        market_id=market_id,
        category=cat_str,
        trade_type=tt_str,
        predicted_probability=0.0,
        base_size=notional,
        multiplier_k=1.0,
        final_size=notional,
        entry_price=limit_price,
        order_ids=order_ids,
        outcome_side=oc_str,
        action="SELL",
        shares=shares,
    )

    logger.info(
        "[PIPELINE] [SELL_EXECUTED] trade=%s %s %sshares@%s",
        trade_id, oc_str, shares, limit_price,
    )
    return trade_id


def execute_trade_decision(
    decision: Dict[str, Any],
    calibrator: ProbabilityCalibrator,
    order_manager: OrderManager,
    trade_ledger: TradeLedger,
    risk_manager: Optional[Any] = None,
    order_book: Optional[Dict] = None,
    capital_manager: Optional[Any] = None,
    position_manager: Optional[Any] = None,
    safety_state: Optional[Any] = None,
    invariants_engine: Optional[Any] = None,
    config: Optional[Any] = None,
) -> Optional[str]:
    """
    Executa decisão de trade completa.

    Suporta três modos:
    - Legacy BUY: decision contém base_size, price, side
    - Spot BUY: decision contém shares, limit_price, action="BUY"
    - Spot SELL: decision contém shares, limit_price, action="SELL"

    Returns:
        trade_id ou None se abortado.
    """
    action = decision.get("action")

    # --- Kill switch: bloquear BUY se componente corrompido/safety ---
    from src.risk.safety_gate import can_execute as safety_check
    gate_ok, gate_reason = safety_check(
        action=action or "BUY",
        calibrator=calibrator,
        capital_manager=capital_manager,
        position_manager=position_manager,
        trade_ledger=trade_ledger,
        safety_state=safety_state,
        config=config,
    )
    if not gate_ok:
        try:
            from src.metrics import record_decision_reason
            record_decision_reason("blocked_by_risk")
        except ImportError:
            pass
        logger.warning("[PIPELINE] [KILL_SWITCH] %s", gate_reason)
        trade_ledger.ensure_persistence_warning_logged()
        return None

    # --- Invariants check pre-BUY ---
    if invariants_engine is not None and (action or "BUY") == "BUY":
        invariants_engine.check_all("pre_buy_order")

    # --- SELL flow (spot-only) ---
    if action == "SELL":
        return _execute_spot_sell(decision, order_manager, trade_ledger, position_manager)

    # --- BUY flow ---
    market_id = decision["market_id"]
    category = decision.get("category", "")
    trade_type = decision.get("trade_type", "")
    predicted_probability = decision.get("predicted_probability", 0.0)
    ttl_seconds = decision.get("ttl_seconds", 60)
    outcome_side = decision.get("outcome_side")

    # Determinar base_size e price conforme modo
    is_spot_buy = "shares" in decision and "limit_price" in decision
    if is_spot_buy:
        shares = decision["shares"]
        limit_price = decision["limit_price"]
        notional = compute_notional(shares, limit_price)
        base_size = notional
        price = limit_price
        side = decision.get("side", OrderSide.BUY)
    else:
        base_size = decision["base_size"]
        price = decision["price"]
        side = decision["side"]
        shares = None

    # 1-2. Multiplicador de calibração
    k = calibrator.get_sizing_multiplier(category, trade_type)

    # 3. Tamanho final (notional ajustado por k)
    final_size = base_size * k

    # 4. Abortar se insuficiente
    if final_size <= 0:
        try:
            from src.metrics import record_decision_reason
            record_decision_reason("no_signal")
        except ImportError:
            pass
        logger.warning("[PIPELINE] [ABORTED] final_size=%s", final_size)
        return None

    # Recalcular shares ajustadas (spot)
    adjusted_shares = None
    if is_spot_buy:
        adjusted_shares = final_size / price if price > 0 else 0

    # 5. Risk check
    cat_str = category.value if isinstance(category, MarketCategory) else str(category)
    tt_str = trade_type.value if isinstance(trade_type, TradeType) else str(trade_type)

    if risk_manager is not None:
        allowed, reason = risk_manager.can_open_trade(
            {"market_id": market_id, "category": cat_str, "final_size": final_size},
            trade_ledger,
            capital_manager=capital_manager,
        )
        if not allowed:
            try:
                from src.metrics import record_decision_reason
                record_decision_reason("blocked_by_risk")
            except ImportError:
                pass
            logger.warning("[PIPELINE] [RISK_BLOCKED] %s", reason)
            return None

    # 6. Slippage check
    if order_book is not None:
        side_str = side.value if isinstance(side, OrderSide) else str(side)
        slippage = estimate_fill_cost(order_book, side_str, final_size)
        if not slippage["required_liquidity_ok"]:
            try:
                from src.metrics import record_decision_reason
                record_decision_reason("blocked_by_liquidity")
            except ImportError:
                pass
            logger.warning(
                "[PIPELINE] [LIQUIDITY_ABORT] Liquidez insuficiente para %s %s",
                side_str, final_size,
            )
            return None

    # Gerar trade_id antecipadamente
    trade_id = str(uuid.uuid4())

    # 7. Reservar capital (antes de colocar ordens)
    if capital_manager is not None:
        try:
            capital_manager.reserve_for_order(trade_id, final_size)
        except Exception as e:
            try:
                from src.metrics import record_decision_reason
                record_decision_reason("blocked_by_risk")
            except ImportError:
                pass
            logger.warning("[PIPELINE] [CAPITAL_ABORT] %s", e)
            return None

    # 8. Ordem limite (com rollback de capital se falhar)
    try:
        order_ids = order_manager.place_limit_order(
            market_id=market_id, side=side, price=price,
            size=final_size, ttl_seconds=ttl_seconds,
        )
    except Exception:
        if capital_manager is not None:
            capital_manager.release_order_reserve(trade_id)
        raise

    if not order_ids and capital_manager is not None:
        capital_manager.release_order_reserve(trade_id)
        try:
            from src.metrics import record_decision_reason
            record_decision_reason("rejected_by_execution")
        except ImportError:
            pass
        logger.warning(
            "[PIPELINE] [ABORTED] Ordem rejeitada (sem order_ids) para %s", market_id
        )
        return None

    # Outcome side para registro
    oc_str = None
    if outcome_side is not None:
        oc_str = outcome_side.value if hasattr(outcome_side, "value") else str(outcome_side)

    # 9. Registrar no ledger (atualiza capital_manager)
    trade_ledger.record_entry(
        trade_id=trade_id,
        market_id=market_id,
        category=cat_str,
        trade_type=tt_str,
        predicted_probability=predicted_probability,
        base_size=base_size,
        multiplier_k=k,
        final_size=final_size,
        entry_price=price,
        order_ids=order_ids,
        outcome_side=oc_str,
        action="BUY" if is_spot_buy else (action or None),
        shares=adjusted_shares,
    )

    # 10. Auditoria atômica: capital já atualizado → snapshot → audit
    if hasattr(order_manager, "finalize_trade_audit"):
        order_manager.finalize_trade_audit(order_ids, capital_manager)

    logger.info(
        "[PIPELINE] [EXECUTED] trade=%s orders=%d size=%s (base=%s x k=%s)%s",
        trade_id, len(order_ids), final_size, base_size, k,
        f" shares={adjusted_shares:.4f}" if adjusted_shares else "",
    )
    return trade_id


def resolve_trade(
    trade_id: str,
    outcome: int,
    trade_ledger: TradeLedger,
) -> None:
    """Registra resolução. Ledger auto-chama calibrator + capital_manager."""
    trade_ledger.record_resolution(trade_id, outcome)
    logger.info(f"[PIPELINE:RESOLVED] trade={trade_id} outcome={outcome}")
