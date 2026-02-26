"""
Decision Pipeline — Signal-to-Order Conversion
==================================================

Converte StrategySignal em ordens reais passando por todos os gates:

PIPELINE:
1. Safety Gate    → buy_blocked? → reject
2. Risk Gate      → exposure limits? → reject
3. Sizing Gate    → calibrator multiplier → ajustar tamanho
4. Capital Gate   → capital disponível? → reject
5. Execution      → submit limit order

MODOS:
- OBSERVE:                 log signal, não envia ordem
- PAPER:                   simula ordem
- REALISTIC_SIMULATION:    simula com book depth
- LIVE:                    envia ordem real

Thread-safe. Todas as decisões são logadas para auditoria.
"""

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.strategy.strategy_engine import StrategySignal

logger = logging.getLogger(__name__)

# Sizing
DEFAULT_BASE_SIZE: float = 10.0          # Tamanho base por trade (em $)
DEFAULT_MAX_SIZE_PER_TRADE: float = 50.0  # Máximo por trade
DEFAULT_MIN_SIZE: float = 1.0             # Mínimo viável
MIN_EDGE_THRESHOLD: float = 0.005         # Edge mínimo para justificar trade
MIN_CONFIDENCE: float = 0.3               # Confiança mínima

DECISION_HISTORY_MAXLEN: int = 200


@dataclass
class TradeDecision:
    """Decisão de trade (resultado do pipeline)."""
    decision_id: str
    timestamp: float
    token_id: str
    side: str
    action: str           # "EXECUTED", "REJECTED", "OBSERVED"
    reject_reason: str
    signal_z: float
    signal_mid: float
    signal_edge: float
    signal_confidence: float
    final_size: float
    limit_price: float
    sizing_multiplier: float


class DecisionPipeline:
    """
    Pipeline de decisão: signal → gates → order.

    Integra safety, risk, calibrator, capital e order manager.
    """

    def __init__(
        self,
        run_mode: str = "OBSERVE",
        safety_state: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        calibrator: Optional[Any] = None,
        capital_manager: Optional[Any] = None,
        position_manager: Optional[Any] = None,
        order_manager: Optional[Any] = None,
        trade_ledger: Optional[Any] = None,
        base_size: float = DEFAULT_BASE_SIZE,
        max_size: float = DEFAULT_MAX_SIZE_PER_TRADE,
    ) -> None:
        self._run_mode = run_mode
        self._safety_state = safety_state
        self._risk_manager = risk_manager
        self._calibrator = calibrator
        self._capital_manager = capital_manager
        self._position_manager = position_manager
        self._order_manager = order_manager
        self._trade_ledger = trade_ledger
        self._base_size = base_size
        self._max_size = max_size
        self._lock = threading.RLock()

        # Histórico de decisões
        self._decisions: deque = deque(maxlen=DECISION_HISTORY_MAXLEN)

        # Contadores
        self._total_signals: int = 0
        self._total_executed: int = 0
        self._total_rejected: int = 0
        self._total_observed: int = 0
        self._reject_reasons: Dict[str, int] = {}

    def process_signal(self, signal: StrategySignal) -> TradeDecision:
        """
        Processa sinal através de todos os gates.

        Retorna TradeDecision com action e detalhes.
        """
        with self._lock:
            self._total_signals += 1

        decision_id = str(uuid.uuid4())[:8]
        now = signal.timestamp or time.time()

        # OBSERVE mode: sempre log, nunca executa
        if self._run_mode == "OBSERVE":
            decision = TradeDecision(
                decision_id=decision_id,
                timestamp=now,
                token_id=signal.token_id,
                side=signal.side,
                action="OBSERVED",
                reject_reason="",
                signal_z=signal.z_score,
                signal_mid=signal.midpoint,
                signal_edge=signal.edge_estimate,
                signal_confidence=signal.confidence,
                final_size=0.0,
                limit_price=signal.best_bid if signal.side == "BUY" else (signal.best_ask or 0.0),
                sizing_multiplier=1.0,
            )
            with self._lock:
                self._total_observed += 1
                self._decisions.append(self._decision_to_dict(decision))
            logger.info(
                "[DECISION] OBSERVED: %s %s z=%.3f mid=%.4f edge=%.4f",
                signal.side, signal.token_id[:12],
                signal.z_score, signal.midpoint, signal.edge_estimate,
            )
            self._record_metrics("observed")
            return decision

        # --- Gate 1: Edge mínimo ---
        if abs(signal.edge_estimate) < MIN_EDGE_THRESHOLD:
            return self._reject(
                decision_id, now, signal, "edge_too_small",
                f"Edge {signal.edge_estimate:.4f} < {MIN_EDGE_THRESHOLD}",
            )

        # --- Gate 2: Confiança mínima ---
        if signal.confidence < MIN_CONFIDENCE:
            return self._reject(
                decision_id, now, signal, "low_confidence",
                f"Confidence {signal.confidence:.3f} < {MIN_CONFIDENCE}",
            )

        # --- Gate 3: Safety ---
        if self._safety_state is not None:
            if signal.side == "BUY" and self._safety_state.buy_blocked:
                return self._reject(
                    decision_id, now, signal, "safety_blocked",
                    f"BUY bloqueado: {sorted(self._safety_state.reasons)}",
                )

        # --- Gate 4: Sizing via Calibrator ---
        sizing_k = 1.0
        if self._calibrator is not None:
            from src.engine.probability_calibrator import MarketCategory, TradeType
            category = self._calibrator.classify_market_category([], signal.token_id)
            sizing_k = self._calibrator.get_sizing_multiplier(
                category, TradeType.STATISTICAL,
            )

        # Tamanho final
        raw_size = self._base_size * sizing_k * min(signal.confidence, 1.0)
        final_size = max(DEFAULT_MIN_SIZE, min(raw_size, self._max_size))

        # --- Gate 5: Risk Manager ---
        if self._risk_manager is not None and self._trade_ledger is not None:
            allowed, reason = self._risk_manager.can_open_trade(
                {
                    "market_id": signal.token_id,
                    "category": "OTHER",
                    "final_size": final_size,
                },
                self._trade_ledger,
                capital_manager=self._capital_manager,
            )
            if not allowed:
                return self._reject(
                    decision_id, now, signal, "risk_blocked", reason,
                )

        # --- Gate 6: Capital ---
        if self._capital_manager is not None and signal.side == "BUY":
            avail = self._capital_manager.get_available_capital()
            if final_size > avail:
                return self._reject(
                    decision_id, now, signal, "insufficient_capital",
                    f"Requerido={final_size:.2f}, Disponível={avail:.2f}",
                )

        # --- Calcular preço limite ---
        if signal.side == "BUY":
            limit_price = signal.best_bid or (signal.midpoint - signal.spread / 2)
        else:
            limit_price = signal.best_ask or (signal.midpoint + signal.spread / 2)

        # Garantir preço em (0, 1)
        limit_price = max(0.01, min(0.99, limit_price))

        # --- Executar ---
        decision = TradeDecision(
            decision_id=decision_id,
            timestamp=now,
            token_id=signal.token_id,
            side=signal.side,
            action="EXECUTED",
            reject_reason="",
            signal_z=signal.z_score,
            signal_mid=signal.midpoint,
            signal_edge=signal.edge_estimate,
            signal_confidence=signal.confidence,
            final_size=final_size,
            limit_price=limit_price,
            sizing_multiplier=sizing_k,
        )

        # Submeter ordem (PAPER/LIVE/SIMULATION)
        order_submitted = self._submit_order(decision)
        if not order_submitted:
            decision.action = "REJECTED"
            decision.reject_reason = "order_submission_failed"
            with self._lock:
                self._total_rejected += 1
                self._decisions.append(self._decision_to_dict(decision))
            return decision

        with self._lock:
            self._total_executed += 1
            self._decisions.append(self._decision_to_dict(decision))

        logger.info(
            "[DECISION] EXECUTED: %s %s size=%.2f@%.4f z=%.3f k=%.2f",
            signal.side, signal.token_id[:12],
            final_size, limit_price, signal.z_score, sizing_k,
        )
        self._record_metrics("executed")
        return decision

    def _submit_order(self, decision: TradeDecision) -> bool:
        """Submete ordem ao OrderManager."""
        if self._order_manager is None:
            return False

        try:
            from src.execution.order_manager import OrderSide
            side = OrderSide.BUY if decision.side == "BUY" else OrderSide.SELL

            # Reserve capital antes de submeter
            if self._capital_manager is not None and decision.side == "BUY":
                try:
                    self._capital_manager.reserve_for_order(
                        decision.decision_id, decision.final_size,
                    )
                except Exception as e:
                    logger.warning("[DECISION] Capital reserve failed: %s", e)
                    return False

            order_ids = self._order_manager.place_limit_order(
                market_id=decision.token_id,
                side=side,
                price=decision.limit_price,
                size=decision.final_size,
                ttl_seconds=self._get_ttl(decision),
                outcome_side=None,
                position_manager=self._position_manager,
                capital_manager=self._capital_manager,
            )

            if not order_ids:
                # Liberar reserva se falhou
                if self._capital_manager is not None and decision.side == "BUY":
                    self._capital_manager.release_order_reserve(decision.decision_id)
                return False

            logger.info(
                "[DECISION] Orders submitted: %s (ids=%s)",
                decision.decision_id, order_ids[:3],
            )
            return True

        except Exception as e:
            logger.error("[DECISION] Order submission error: %s", e)
            if self._capital_manager is not None and decision.side == "BUY":
                self._capital_manager.release_order_reserve(decision.decision_id)
            return False

    def _get_ttl(self, decision: TradeDecision) -> int:
        """TTL baseado na volatilidade e confiança."""
        # Trades de alta confiança: TTL maior
        if decision.signal_confidence > 0.7:
            return 120
        elif decision.signal_confidence > 0.5:
            return 90
        else:
            return 60

    def _reject(
        self,
        decision_id: str,
        ts: float,
        signal: StrategySignal,
        reason_key: str,
        reason_detail: str,
    ) -> TradeDecision:
        """Cria decisão rejeitada e registra."""
        decision = TradeDecision(
            decision_id=decision_id,
            timestamp=ts,
            token_id=signal.token_id,
            side=signal.side,
            action="REJECTED",
            reject_reason=reason_detail,
            signal_z=signal.z_score,
            signal_mid=signal.midpoint,
            signal_edge=signal.edge_estimate,
            signal_confidence=signal.confidence,
            final_size=0.0,
            limit_price=0.0,
            sizing_multiplier=0.0,
        )
        with self._lock:
            self._total_rejected += 1
            self._reject_reasons[reason_key] = (
                self._reject_reasons.get(reason_key, 0) + 1
            )
            self._decisions.append(self._decision_to_dict(decision))

        logger.info(
            "[DECISION] REJECTED (%s): %s %s — %s",
            reason_key, signal.side, signal.token_id[:12], reason_detail,
        )
        self._record_metrics(f"rejected_{reason_key}")
        return decision

    def _record_metrics(self, reason: str) -> None:
        """Registra decisão nas métricas globais."""
        try:
            from src.metrics import record_decision_reason
            # Mapear para os reasons existentes
            mapping = {
                "observed": "no_signal",
                "rejected_safety_blocked": "blocked_by_risk",
                "rejected_risk_blocked": "blocked_by_risk",
                "rejected_edge_too_small": "blocked_by_confidence",
                "rejected_low_confidence": "blocked_by_confidence",
                "rejected_insufficient_capital": "blocked_by_risk",
                "executed": "no_signal",  # não é rejeição
            }
            mapped = mapping.get(reason)
            if mapped and reason.startswith("rejected"):
                record_decision_reason(mapped)
        except ImportError:
            pass

    @staticmethod
    def _decision_to_dict(d: TradeDecision) -> Dict[str, Any]:
        """Converte decisão para dict serializável."""
        return {
            "id": d.decision_id,
            "ts": d.timestamp,
            "token_id": d.token_id,
            "side": d.side,
            "action": d.action,
            "reject_reason": d.reject_reason,
            "z": round(d.signal_z, 4),
            "mid": round(d.signal_mid, 4),
            "edge": round(d.signal_edge, 4),
            "confidence": round(d.signal_confidence, 3),
            "size": round(d.final_size, 2),
            "price": round(d.limit_price, 4),
            "k": round(d.sizing_multiplier, 2),
        }

    # ------------------------------------------------------------------
    # Leitura de métricas
    # ------------------------------------------------------------------

    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna últimas decisões."""
        with self._lock:
            entries = list(self._decisions)
            return entries[-limit:]

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Retorna resumo do pipeline."""
        with self._lock:
            return {
                "run_mode": self._run_mode,
                "total_signals": self._total_signals,
                "total_executed": self._total_executed,
                "total_rejected": self._total_rejected,
                "total_observed": self._total_observed,
                "reject_reasons": dict(self._reject_reasons),
                "execution_rate": (
                    round(self._total_executed / self._total_signals, 3)
                    if self._total_signals > 0 else 0.0
                ),
                "base_size": self._base_size,
                "max_size": self._max_size,
            }
