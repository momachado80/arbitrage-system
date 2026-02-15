"""
Trade Ledger Module
===================

Registro auditável e idempotente de trades para mercados preditivos.

STATUS LIFECYCLE (Two-Phase Commit):
  INTENT_CREATED       → Trade criado, sem capital reservado
  CAPITAL_RESERVED     → Capital reservado via CapitalManager
  ORDER_SUBMITTED      → Ordem enviada à exchange, order_id registrado
  OPEN                 → Ordem confirmada, aguardando preenchimento
  PARTIALLY_FILLED     → Fill parcial recebido
  FILLED               → Todas as ordens preenchidas, aguardando resolução
  CANCELLED            → Trade cancelado (sem fills ou parcial)
  REJECTED             → Ordem rejeitada pela exchange
  EXPIRED              → Ordem expirada (TTL)
  RESOLVED             → Mercado resolvido, outcome registrado, calibrator atualizado

BACKWARD COMPATIBILITY:
- record_entry() cria trade diretamente como OPEN (sem two-phase)
- create_intent() inicia o fluxo two-phase
- JSONs antigos com status CLOSED são migrados automaticamente para RESOLVED

RECONCILIAÇÃO:
O TradeLedger suporta atualização de fill status via update_fill_status(),
que é chamado pelo OrderReconciler para sincronizar com a exchange.

PERSISTÊNCIA ATÔMICA:
NamedTemporaryFile → write → flush → fsync → os.replace → dir fsync
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH: str = os.path.expanduser(
    "~/.polymarket_bot/trade_ledger.json"
)
LEDGER_SCHEMA_VERSION: int = 1
MAX_FILL_IDS_PER_TRADE: int = 500


class TradeStatus:
    """
    Lifecycle formal de ordens (two-phase commit).

    Fluxo nominal:
    INTENT_CREATED → CAPITAL_RESERVED → ORDER_SUBMITTED → OPEN
      → PARTIALLY_FILLED → FILLED → RESOLVED

    Fluxo de cancelamento:
    (qualquer pré-OPEN) → CANCELLED / REJECTED / EXPIRED

    Compatibilidade:
    - CLOSED mapeado para RESOLVED na migração de JSONs antigos.
    - record_entry() cria diretamente OPEN (backward compat).
    """
    # Two-phase commit (pré-OPEN)
    INTENT_CREATED = "INTENT_CREATED"
    CAPITAL_RESERVED = "CAPITAL_RESERVED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    # Lifecycle principal
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    # Terminais
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    RESOLVED = "RESOLVED"
    # Legacy (mapeado para RESOLVED na migração)
    CLOSED = "CLOSED"

    # Statuses que representam exposição ativa
    ACTIVE_STATUSES = {"OPEN", "PARTIALLY_FILLED", "FILLED"}
    # Statuses que aceitam fills
    FILLABLE_STATUSES = {"OPEN", "PARTIALLY_FILLED", "ORDER_SUBMITTED"}
    # Statuses terminais
    TERMINAL_STATUSES = {"CANCELLED", "REJECTED", "EXPIRED", "RESOLVED", "CLOSED"}


# Migração: statuses antigos → novos
_STATUS_MIGRATION = {
    "CLOSED": "RESOLVED",
}


def _migrate_status(status: str) -> str:
    """Migra statuses antigos para o novo modelo."""
    return _STATUS_MIGRATION.get(status, status)


@dataclass
class TradeRecord:
    trade_id: str
    market_id: str
    category: str
    trade_type: str
    predicted_probability: float
    base_size: float
    multiplier_k: float
    final_size: float
    entry_price: float
    timestamp_entry: str
    status: str = TradeStatus.OPEN
    resolution_outcome: Optional[int] = None
    timestamp_resolution: Optional[str] = None
    order_ids: List[str] = field(default_factory=list)
    cumulative_filled_size: float = 0.0
    weighted_avg_price: float = 0.0
    outcome_side: Optional[str] = None  # "YES" ou "NO" (spot-only)
    action: Optional[str] = None  # "BUY" ou "SELL" (spot-only)
    shares: Optional[float] = None  # Número de contratos (spot-only)


class TradeLedger:
    """Registro auditável e idempotente de trades."""

    def __init__(
        self,
        state_path: Optional[str] = None,
        calibrator: Optional[Any] = None,
        capital_manager: Optional[Any] = None,
    ):
        self._state_path = Path(state_path or DEFAULT_LEDGER_PATH)
        self._calibrator = calibrator
        self._capital_manager = capital_manager
        self._trades: Dict[str, TradeRecord] = {}
        self._applied_fills: Dict[str, set] = {}  # trade_id → set of fill_ids
        self._last_fill_ts: Dict[str, float] = {}  # trade_id → epoch seconds
        self._persistence_ok: bool = True
        self._reconcile_ok: bool = True
        self._persistence_warning_emitted: bool = False
        self._lock = threading.RLock()
        self._ensure_dir()
        self._load()

    def _ensure_dir(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fsync_directory(dir_path: str) -> None:
        try:
            dir_fd = os.open(dir_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _atomic_save(self) -> None:
        with self._lock:
            trades_data = {tid: asdict(tr) for tid, tr in self._trades.items()}
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "updated_at": now,
            "trade_count": len(trades_data),
            "trades": trades_data,
        }
        json_str = json.dumps(payload, indent=2)
        state_dir = str(self._state_path.parent)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="ledger_",
                dir=state_dir, delete=False
            ) as f:
                temp_path = f.name
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, str(self._state_path))
            self._fsync_directory(state_dir)
        except Exception:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _load(self) -> None:
        if not self._state_path.exists():
            self._trades = {}
            return
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            raw_trades = data.get("trades", {})
            restored: Dict[str, TradeRecord] = {}
            for tid, e in raw_trades.items():
                restored[tid] = TradeRecord(
                    trade_id=e["trade_id"],
                    market_id=e["market_id"],
                    category=e["category"],
                    trade_type=e["trade_type"],
                    predicted_probability=e["predicted_probability"],
                    base_size=e["base_size"],
                    multiplier_k=e["multiplier_k"],
                    final_size=e["final_size"],
                    entry_price=e["entry_price"],
                    timestamp_entry=e["timestamp_entry"],
                    status=_migrate_status(e.get("status", TradeStatus.OPEN)),
                    resolution_outcome=e.get("resolution_outcome"),
                    timestamp_resolution=e.get("timestamp_resolution"),
                    order_ids=e.get("order_ids", []),
                    cumulative_filled_size=e.get("cumulative_filled_size", 0.0),
                    weighted_avg_price=e.get("weighted_avg_price", 0.0),
                    outcome_side=e.get("outcome_side"),
                    action=e.get("action"),
                    shares=e.get("shares"),
                )
            with self._lock:
                self._trades = restored
            logger.info(f"Ledger carregado: {len(restored)} trades")
        except Exception as e:
            logger.error(f"Erro ao carregar ledger: {e}")
            self._trades = {}
            self._persistence_ok = False

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def record_entry(
        self,
        trade_id: str,
        market_id: str,
        category: str,
        trade_type: str,
        predicted_probability: float,
        base_size: float,
        multiplier_k: float,
        final_size: float,
        entry_price: float,
        order_ids: Optional[List[str]] = None,
        outcome_side: Optional[str] = None,
        action: Optional[str] = None,
        shares: Optional[float] = None,
    ) -> bool:
        """Registra entrada. Idempotente: retorna False se trade_id já existe."""
        with self._lock:
            if trade_id in self._trades:
                return False
            self._trades[trade_id] = TradeRecord(
                trade_id=trade_id,
                market_id=market_id,
                category=category,
                trade_type=trade_type,
                predicted_probability=predicted_probability,
                base_size=base_size,
                multiplier_k=multiplier_k,
                final_size=final_size,
                entry_price=entry_price,
                timestamp_entry=datetime.now(timezone.utc).isoformat(),
                order_ids=order_ids or [],
                outcome_side=outcome_side,
                action=action,
                shares=shares,
            )
        logger.info(
            f"[LEDGER:ENTRY] {trade_id} {category}/{trade_type} "
            f"size={final_size} orders={len(order_ids or [])}"
        )
        self._atomic_save()
        # Capital: reservar (idempotente se já reservado pelo pipeline)
        # SELL trades não reservam capital (são redução de exposição)
        if self._capital_manager is not None and action != "SELL":
            self._capital_manager.reserve_for_order(trade_id, final_size)
        return True

    def record_resolution(self, trade_id: str, outcome: int) -> None:
        """
        Registra resolução. Aceita trades OPEN ou FILLED.

        Raises:
            ValueError: outcome não é 0 ou 1
            KeyError: trade_id inexistente
            RuntimeError: trade não está OPEN nem FILLED
        """
        if outcome not in (0, 1):
            raise ValueError(f"outcome deve ser 0 ou 1, recebido: {outcome}")
        with self._lock:
            if trade_id not in self._trades:
                raise KeyError(f"Trade não encontrado: {trade_id}")
            trade = self._trades[trade_id]
            resolvable = {
                TradeStatus.OPEN, TradeStatus.FILLED,
                TradeStatus.PARTIALLY_FILLED,
            }
            if trade.status not in resolvable:
                raise RuntimeError(
                    f"Trade {trade_id} não pode ser resolvido (status={trade.status})"
                )
            trade.resolution_outcome = outcome
            trade.status = TradeStatus.RESOLVED
            trade.timestamp_resolution = datetime.now(timezone.utc).isoformat()
            cat, tt, pred = trade.category, trade.trade_type, trade.predicted_probability
            final_size_val = trade.final_size
            is_sell_trade = (trade.action == "SELL")
        logger.info(f"[LEDGER:RESOLVED] {trade_id} outcome={outcome}")
        self._atomic_save()
        # Capital: resolver trade e registrar PnL (somente BUY trades)
        # SELL trades já têm capital processado via process_sell_fill
        if self._capital_manager is not None and not is_sell_trade:
            payout = final_size_val * (1 if outcome == 1 else 0)
            pnl = payout - final_size_val
            self._capital_manager.resolve_trade(trade_id, pnl)
        if self._calibrator is not None:
            from src.engine.probability_calibrator import MarketCategory, TradeType
            self._calibrator.record_trade_result(
                MarketCategory.from_string(cat),
                TradeType.from_string(tt),
                pred, outcome,
            )

    def update_fill_status(
        self,
        trade_id: str,
        cumulative_filled: float,
        avg_price: float,
    ) -> None:
        """
        Atualiza informação de preenchimento (usado pelo reconciler).

        Se cumulative_filled >= final_size, marca como FILLED.
        """
        with self._lock:
            if trade_id not in self._trades:
                raise KeyError(f"Trade não encontrado: {trade_id}")
            trade = self._trades[trade_id]
            trade.cumulative_filled_size = cumulative_filled
            trade.weighted_avg_price = avg_price
            should_move_capital = False
            final_size_val = trade.final_size
            is_sell_trade = (trade.action == "SELL")
            from src.utils.precision import FILL_EPS
            fillable = TradeStatus.FILLABLE_STATUSES | {TradeStatus.OPEN}
            if cumulative_filled + FILL_EPS >= trade.final_size and trade.status in fillable:
                trade.status = TradeStatus.FILLED
                # SELL trades não bloqueiam capital em unresolved
                should_move_capital = not is_sell_trade
                logger.info(
                    f"[LEDGER:FILLED] {trade_id} "
                    f"filled={cumulative_filled} avg={avg_price:.4f}"
                )
            elif (
                cumulative_filled > 0
                and cumulative_filled + FILL_EPS < trade.final_size
                and trade.status in {TradeStatus.OPEN, TradeStatus.ORDER_SUBMITTED}
            ):
                trade.status = TradeStatus.PARTIALLY_FILLED
        self._atomic_save()
        # Capital: mover de open_orders para unresolved (idempotente)
        if should_move_capital and self._capital_manager is not None:
            self._capital_manager.move_to_unresolved(trade_id, final_size_val)

    def cancel_trade(self, trade_id: str) -> None:
        """Cancela um trade OPEN ou pré-OPEN. Idempotente para já-cancelados."""
        with self._lock:
            if trade_id not in self._trades:
                raise KeyError(f"Trade não encontrado: {trade_id}")
            trade = self._trades[trade_id]
            if trade.status == TradeStatus.CANCELLED:
                return  # Idempotente
            cancellable = {
                TradeStatus.OPEN, TradeStatus.INTENT_CREATED,
                TradeStatus.CAPITAL_RESERVED, TradeStatus.ORDER_SUBMITTED,
                TradeStatus.PARTIALLY_FILLED,
            }
            if trade.status not in cancellable:
                raise RuntimeError(
                    f"Trade {trade_id} não pode ser cancelado (status={trade.status})"
                )
            trade.status = TradeStatus.CANCELLED
        logger.info(f"[LEDGER:CANCELLED] {trade_id}")
        self._atomic_save()
        # Capital: liberar reserva de ordem (idempotente)
        if self._capital_manager is not None:
            self._capital_manager.release_order_reserve(trade_id)

    # ------------------------------------------------------------------
    # Two-phase commit transitions (idempotentes)
    # ------------------------------------------------------------------

    def create_intent(
        self,
        trade_id: str,
        market_id: str,
        category: str,
        trade_type: str,
        predicted_probability: float,
        base_size: float,
        final_size: float,
        entry_price: float,
        outcome_side: Optional[str] = None,
        action: Optional[str] = None,
        shares: Optional[float] = None,
    ) -> bool:
        """
        Cria trade com status INTENT_CREATED (fase 1 do two-phase commit).

        Idempotente: retorna False se trade_id já existe.
        """
        with self._lock:
            if trade_id in self._trades:
                return False
            self._trades[trade_id] = TradeRecord(
                trade_id=trade_id,
                market_id=market_id,
                category=category,
                trade_type=trade_type,
                predicted_probability=predicted_probability,
                base_size=base_size,
                multiplier_k=1.0,
                final_size=final_size,
                entry_price=entry_price,
                timestamp_entry=datetime.now(timezone.utc).isoformat(),
                status=TradeStatus.INTENT_CREATED,
                outcome_side=outcome_side,
                action=action,
                shares=shares,
            )
        logger.info(f"[LEDGER:INTENT_CREATED] {trade_id}")
        self._atomic_save()
        return True

    def advance_to_capital_reserved(self, trade_id: str) -> None:
        """Transição: INTENT_CREATED → CAPITAL_RESERVED. Idempotente."""
        with self._lock:
            if trade_id not in self._trades:
                raise KeyError(f"Trade não encontrado: {trade_id}")
            trade = self._trades[trade_id]
            if trade.status == TradeStatus.CAPITAL_RESERVED:
                return  # Idempotente
            if trade.status != TradeStatus.INTENT_CREATED:
                raise RuntimeError(
                    f"Trade {trade_id} não está INTENT_CREATED "
                    f"(status={trade.status})"
                )
            trade.status = TradeStatus.CAPITAL_RESERVED
        self._atomic_save()

    def advance_to_order_submitted(
        self, trade_id: str, order_ids: List[str],
    ) -> None:
        """Transição: CAPITAL_RESERVED → ORDER_SUBMITTED. Idempotente."""
        with self._lock:
            if trade_id not in self._trades:
                raise KeyError(f"Trade não encontrado: {trade_id}")
            trade = self._trades[trade_id]
            if trade.status == TradeStatus.ORDER_SUBMITTED:
                return  # Idempotente
            if trade.status != TradeStatus.CAPITAL_RESERVED:
                raise RuntimeError(
                    f"Trade {trade_id} não está CAPITAL_RESERVED "
                    f"(status={trade.status})"
                )
            trade.status = TradeStatus.ORDER_SUBMITTED
            trade.order_ids = order_ids
        self._atomic_save()

    def advance_to_open(self, trade_id: str) -> None:
        """Transição: ORDER_SUBMITTED → OPEN. Idempotente."""
        with self._lock:
            if trade_id not in self._trades:
                raise KeyError(f"Trade não encontrado: {trade_id}")
            trade = self._trades[trade_id]
            if trade.status == TradeStatus.OPEN:
                return  # Idempotente
            if trade.status != TradeStatus.ORDER_SUBMITTED:
                raise RuntimeError(
                    f"Trade {trade_id} não está ORDER_SUBMITTED "
                    f"(status={trade.status})"
                )
            trade.status = TradeStatus.OPEN
        self._atomic_save()

    def mark_rejected(self, trade_id: str) -> None:
        """Marca trade como REJECTED. Idempotente. Libera capital."""
        with self._lock:
            if trade_id not in self._trades:
                raise KeyError(f"Trade não encontrado: {trade_id}")
            trade = self._trades[trade_id]
            if trade.status == TradeStatus.REJECTED:
                return  # Idempotente
            trade.status = TradeStatus.REJECTED
        logger.info(f"[LEDGER:REJECTED] {trade_id}")
        self._atomic_save()
        if self._capital_manager is not None:
            self._capital_manager.release_order_reserve(trade_id)

    def mark_expired(self, trade_id: str) -> None:
        """Marca trade como EXPIRED. Idempotente. Libera capital."""
        with self._lock:
            if trade_id not in self._trades:
                raise KeyError(f"Trade não encontrado: {trade_id}")
            trade = self._trades[trade_id]
            if trade.status == TradeStatus.EXPIRED:
                return  # Idempotente
            trade.status = TradeStatus.EXPIRED
        logger.info(f"[LEDGER:EXPIRED] {trade_id}")
        self._atomic_save()
        if self._capital_manager is not None:
            self._capital_manager.release_order_reserve(trade_id)

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if trade_id not in self._trades:
                return None
            return asdict(self._trades[trade_id])

    def get_open_trades(self) -> Dict[str, Dict[str, Any]]:
        """Trades com status OPEN ou PARTIALLY_FILLED (aguardando fills)."""
        with self._lock:
            return {
                tid: asdict(tr) for tid, tr in self._trades.items()
                if tr.status in {
                    TradeStatus.OPEN,
                    TradeStatus.PARTIALLY_FILLED,
                    TradeStatus.ORDER_SUBMITTED,
                }
            }

    def get_active_trades(self) -> Dict[str, Dict[str, Any]]:
        """Trades com exposição ativa (OPEN, PARTIALLY_FILLED, FILLED)."""
        with self._lock:
            return {
                tid: asdict(tr) for tid, tr in self._trades.items()
                if tr.status in TradeStatus.ACTIVE_STATUSES
            }

    def get_all_trades(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {tid: asdict(tr) for tid, tr in self._trades.items()}

    @property
    def trade_count(self) -> int:
        with self._lock:
            return len(self._trades)

    @property
    def persistence_ok(self) -> bool:
        """Retorna False se carregamento do ledger falhou."""
        return self._persistence_ok

    @property
    def reconcile_ok(self) -> bool:
        """Retorna False se reconciler detectou fill inválido."""
        return self._reconcile_ok

    def mark_reconcile_failed(self) -> None:
        """Marcado pelo reconciler quando fill inválido é detectado."""
        self._reconcile_ok = False

    def ensure_persistence_warning_logged(self) -> None:
        """
        Emite warning ONCE PER SESSION quando persistence_ok=False.

        Chamado pelo pipeline para garantir visibilidade do problema
        sem spam de logs.
        """
        if not self._persistence_ok and not self._persistence_warning_emitted:
            logger.warning(
                "[LEDGER:PERSISTENCE_DISABLED] persistence_ok=False, "
                "BUY blocked by safety_gate"
            )
            self._persistence_warning_emitted = True

    # ------------------------------------------------------------------
    # Fill deduplication / replay sensor
    # ------------------------------------------------------------------

    def get_last_fill_ts(self, trade_id: str) -> Optional[float]:
        """Retorna último timestamp (epoch) visto para este trade."""
        with self._lock:
            return self._last_fill_ts.get(trade_id)

    def update_last_fill_ts(self, trade_id: str, ts: float) -> None:
        """Atualiza last_fill_ts = max(current, ts). In-memory only."""
        with self._lock:
            current = self._last_fill_ts.get(trade_id)
            if current is None or ts > current:
                self._last_fill_ts[trade_id] = ts

    def is_fill_applied(self, trade_id: str, fill_id: str) -> bool:
        """Verifica se fill_id já foi registrado para este trade."""
        with self._lock:
            return fill_id in self._applied_fills.get(trade_id, set())

    def mark_fill_applied(self, trade_id: str, fill_id: str) -> None:
        """
        Registra fill_id como aplicado (audit trail, rolling window).

        Não persiste no JSON — evita crescimento indefinido.
        Cap: MAX_FILL_IDS_PER_TRADE por trade.
        """
        with self._lock:
            if trade_id not in self._applied_fills:
                self._applied_fills[trade_id] = set()
            tracker = self._applied_fills[trade_id]
            if len(tracker) >= MAX_FILL_IDS_PER_TRADE:
                return  # Cap atingido
            tracker.add(fill_id)
