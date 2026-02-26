"""
Edge Validation — Scientific Experiment Module
==================================================

Roda em modo OBSERVE. Não executa trades.

Responsabilidades:
1. Registra todas as oportunidades com net_edge > 0
2. Simula execução (sem enviar ordem)
3. Compara com outcome final
4. Gera relatório consolidado

Relatório inclui:
- total_opportunities
- hit_rate
- average_net_edge
- realized_pnl (simulado)
- brier_score
- average_decay_time

Sem matemática complexa. Só números claros.

Thread-safe via RLock.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = os.path.join(
    os.environ.get("DATA_DIR", "/tmp/polymarket_data"), "edge_validation.jsonl",
)
MAX_OPPORTUNITIES = 10000


@dataclass
class Opportunity:
    """Uma oportunidade de edge detectada."""
    opportunity_id: int
    timestamp: float
    token_id: str
    side: str                # "BUY" ou "SELL"
    p_oracle: float
    p_market: float
    best_bid: float
    best_ask: float
    execution_price: float   # ask (BUY) ou bid (SELL)
    net_edge: float
    raw_edge: float
    oracle_confidence: float
    outcome: Optional[float] = None      # 1.0 ou 0.0 quando resolvido
    resolved_at: Optional[float] = None
    simulated_pnl: Optional[float] = None
    decay_seconds: Optional[float] = None


class EdgeValidator:
    """
    Registra oportunidades e gera relatório de validação.

    Modo OBSERVE: observa, não executa.
    Objetivo: provar que edge existe antes de escalar.
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
    ) -> None:
        self._log_path = log_path or DEFAULT_LOG_PATH
        self._lock = threading.RLock()
        self._opportunities: Deque[Opportunity] = deque(maxlen=MAX_OPPORTUNITIES)
        self._next_id: int = 1
        self._pending: Dict[str, Opportunity] = {}  # token_id → latest opportunity

        # Ensure directory
        d = os.path.dirname(self._log_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def record_opportunity(
        self,
        token_id: str,
        side: str,
        p_oracle: float,
        p_market: float,
        best_bid: float,
        best_ask: float,
        execution_price: float,
        net_edge: float,
        raw_edge: float,
        oracle_confidence: float,
    ) -> int:
        """
        Registra uma nova oportunidade de edge.

        Returns:
            opportunity_id
        """
        with self._lock:
            opp = Opportunity(
                opportunity_id=self._next_id,
                timestamp=time.time(),
                token_id=token_id,
                side=side,
                p_oracle=p_oracle,
                p_market=p_market,
                best_bid=best_bid,
                best_ask=best_ask,
                execution_price=execution_price,
                net_edge=net_edge,
                raw_edge=raw_edge,
                oracle_confidence=oracle_confidence,
            )
            self._next_id += 1
            self._opportunities.append(opp)
            self._pending[token_id] = opp

        self._persist_event("opportunity", opp)
        return opp.opportunity_id

    def record_outcome(
        self,
        token_id: str,
        outcome: float,
    ) -> None:
        """
        Registra outcome para oportunidade pendente.

        Calcula PnL simulado:
        - BUY: pnl = outcome - execution_price (comprou YES no ask)
        - SELL: pnl = execution_price - outcome (vendeu YES no bid)
        """
        with self._lock:
            opp = self._pending.pop(token_id, None)
            if opp is None:
                return

            opp.outcome = outcome
            opp.resolved_at = time.time()

            if opp.side == "BUY":
                opp.simulated_pnl = outcome - opp.execution_price
            elif opp.side == "SELL":
                opp.simulated_pnl = opp.execution_price - outcome

        self._persist_event("outcome", opp)

    def record_decay(
        self,
        token_id: str,
        decay_seconds: float,
    ) -> None:
        """Registra tempo de decay para oportunidade."""
        with self._lock:
            opp = self._pending.get(token_id)
            if opp is not None:
                opp.decay_seconds = decay_seconds

    def _persist_event(self, event_type: str, opp: Opportunity) -> None:
        """Persiste evento no JSONL."""
        try:
            entry = asdict(opp)
            entry["event_type"] = event_type
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning("[EDGE_VALIDATION] Persist failed: %s", e)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> Dict[str, Any]:
        """
        Gera relatório consolidado.

        Números claros, sem matemática complexa:
        - total_opportunities
        - hit_rate
        - average_net_edge
        - realized_pnl (simulado)
        - brier_score
        - average_decay_time
        """
        with self._lock:
            all_opps = list(self._opportunities)

        if not all_opps:
            return {
                "total_opportunities": 0,
                "resolved": 0,
                "pending": 0,
                "hit_rate": None,
                "average_net_edge": None,
                "total_simulated_pnl": None,
                "avg_simulated_pnl": None,
                "brier_score": None,
                "average_decay_time": None,
                "by_side": {},
                "edge_quality": "no_data",
            }

        resolved = [o for o in all_opps if o.outcome is not None]
        pending = [o for o in all_opps if o.outcome is None]

        # Hit rate: fraction of trades where simulated_pnl > 0
        wins = [o for o in resolved if o.simulated_pnl is not None and o.simulated_pnl > 0]
        hit_rate = len(wins) / len(resolved) if resolved else None

        # Average net edge at time of signal
        avg_net_edge = (
            sum(o.net_edge for o in all_opps) / len(all_opps)
            if all_opps else None
        )

        # Total simulated PnL
        pnls = [o.simulated_pnl for o in resolved if o.simulated_pnl is not None]
        total_pnl = sum(pnls) if pnls else None
        avg_pnl = sum(pnls) / len(pnls) if pnls else None

        # Brier score (oracle quality)
        brier_scores = []
        for o in resolved:
            if o.outcome is not None:
                brier_scores.append((o.p_oracle - o.outcome) ** 2)
        brier = sum(brier_scores) / len(brier_scores) if brier_scores else None

        # Average decay time
        decays = [o.decay_seconds for o in all_opps if o.decay_seconds is not None]
        avg_decay = sum(decays) / len(decays) if decays else None

        # By side
        buy_opps = [o for o in all_opps if o.side == "BUY"]
        sell_opps = [o for o in all_opps if o.side == "SELL"]
        by_side = {}
        for label, group in [("BUY", buy_opps), ("SELL", sell_opps)]:
            group_resolved = [o for o in group if o.simulated_pnl is not None]
            group_wins = [o for o in group_resolved if o.simulated_pnl > 0]
            group_pnl = [o.simulated_pnl for o in group_resolved]
            by_side[label] = {
                "count": len(group),
                "resolved": len(group_resolved),
                "hit_rate": len(group_wins) / len(group_resolved) if group_resolved else None,
                "total_pnl": round(sum(group_pnl), 4) if group_pnl else None,
            }

        # Edge quality assessment
        if brier is None:
            edge_quality = "no_data"
        elif brier < 0.15 and hit_rate is not None and hit_rate > 0.55:
            edge_quality = "strong_edge"
        elif brier < 0.25 and hit_rate is not None and hit_rate > 0.50:
            edge_quality = "moderate_edge"
        elif hit_rate is not None and hit_rate > 0.50:
            edge_quality = "weak_edge"
        else:
            edge_quality = "no_edge_detected"

        return {
            "total_opportunities": len(all_opps),
            "resolved": len(resolved),
            "pending": len(pending),
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "average_net_edge": round(avg_net_edge, 4) if avg_net_edge is not None else None,
            "total_simulated_pnl": round(total_pnl, 4) if total_pnl is not None else None,
            "avg_simulated_pnl": round(avg_pnl, 4) if avg_pnl is not None else None,
            "brier_score": round(brier, 4) if brier is not None else None,
            "average_decay_time": round(avg_decay, 1) if avg_decay is not None else None,
            "by_side": by_side,
            "edge_quality": edge_quality,
        }

    def get_recent_opportunities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retorna últimas oportunidades para display."""
        with self._lock:
            opps = list(self._opportunities)[-limit:]
        return [
            {
                "id": o.opportunity_id,
                "ts": o.timestamp,
                "token_id": o.token_id,
                "side": o.side,
                "net_edge": round(o.net_edge, 4),
                "p_oracle": round(o.p_oracle, 4),
                "p_market": round(o.p_market, 4),
                "confidence": round(o.oracle_confidence, 3),
                "outcome": o.outcome,
                "pnl": round(o.simulated_pnl, 4) if o.simulated_pnl is not None else None,
            }
            for o in opps
        ]
