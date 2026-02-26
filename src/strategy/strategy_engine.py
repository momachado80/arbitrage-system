"""
Strategy Engine — Probability Edge Trading Strategy
========================================================

Tese quantitativa: Mispricing baseado em Probability Oracle.
Detecta divergências entre P_oracle (probabilidade estimada
externamente) e P_market (probabilidade implícita do mercado).

FÓRMULA NET EDGE (matematicamente correta):
    BUY:  net_edge = P_oracle - P_ask - fees - slippage
    SELL: net_edge = P_bid - P_oracle - fees - slippage

Spread NÃO é subtraído separadamente — já embutido no preço
de execução (ask para compra, bid para venda).

PROPRIEDADE: EV > 0 ↔ net_edge > 0

REGRAS:
- BUY se net_edge >= threshold (mercado subprecifica)
- SELL se net_edge >= threshold (mercado sobreprecifica)
- Confidence do oracle multiplica sizing
- Brier score adaptativo ajusta tamanho automaticamente
- Safety block se Oracle performance degradar

PIPELINE:
1. Recebe MarketSnapshot do PipelineHandler
2. Consulta ProbabilityOracle para P_oracle
3. Calcula net_edge via EdgeModel (preço de execução real)
4. Se edge acionável → gera StrategySignal
5. Registra para edge decay tracking

MÉTRICAS:
- Histórico de edge por token
- Edge médio, P_oracle vs P_market divergência
- Performance por bucket de edge
- Compatibilidade total com endpoints existentes

Thread-safe via RLock.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.strategy.edge_model import EdgeConfig, EdgeModel, EdgeResult

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_HISTORY_MAXLEN: int = 5000
DEFAULT_SIGNAL_HISTORY_MAXLEN: int = 500
DEFAULT_EDGE_THRESHOLD: float = 0.03            # 3% edge mínimo para sinal
DEFAULT_FEE_ESTIMATE: float = 0.02              # 2% fee estimada
DEFAULT_SLIPPAGE_ESTIMATE: float = 0.005        # 0.5% slippage estimado


@dataclass
class StrategyConfig:
    """Configuração da estratégia de edge probabilístico."""
    edge_buy_threshold: float = DEFAULT_EDGE_THRESHOLD
    edge_sell_threshold: float = DEFAULT_EDGE_THRESHOLD
    fee_estimate: float = DEFAULT_FEE_ESTIMATE
    slippage_estimate: float = DEFAULT_SLIPPAGE_ESTIMATE
    spread_impact_factor: float = 0.0  # DEPRECATED: spread is in execution price
    min_oracle_confidence: float = 0.2
    min_spread: float = 0.001
    max_spread: float = 0.15


def create_default_strategy_config() -> StrategyConfig:
    """Cria configuração padrão."""
    return StrategyConfig()


@dataclass
class MarketState:
    """Estado interno para cada token monitorado."""
    token_id: str
    snapshot_count: int = 0
    last_mid: float = 0.0
    last_spread: float = 0.0
    last_best_bid: Optional[float] = None
    last_best_ask: Optional[float] = None
    last_update_ts: float = 0.0
    last_p_oracle: Optional[float] = None
    last_p_market: Optional[float] = None
    last_edge: Optional[float] = None
    last_net_edge: Optional[float] = None
    last_oracle_confidence: Optional[float] = None


@dataclass
class StrategySignal:
    """Sinal estruturado para o DecisionPipeline.

    NOTA: Mantém campo z_score para compatibilidade com DecisionPipeline,
    mas agora contém o net_edge (não é z-score de preço).
    """
    timestamp: float
    token_id: str
    side: str           # "BUY" ou "SELL"
    z_score: float      # COMPAT: contém net_edge (não é z-score real)
    midpoint: float
    spread: float
    edge_estimate: float
    confidence: float
    p_est: float        # P_oracle estimada
    best_bid: Optional[float]
    best_ask: Optional[float]
    rationale: Tuple[str, ...]


class StrategyEngine:
    """
    Motor de estratégia quantitativa baseado em edge probabilístico.

    Recebe snapshots, consulta Oracle, calcula net_edge,
    gera sinais acionáveis para o DecisionPipeline.

    Interface pública 100% compatível com versão anterior.
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        safety_state: Optional[Any] = None,
        oracle: Optional[Any] = None,
        min_liquidity: float = 50.0,
    ) -> None:
        self._config = config or create_default_strategy_config()
        self._safety_state = safety_state
        self._oracle = oracle  # ProbabilityOracle instance
        self._min_liquidity = min_liquidity
        self._lock = threading.RLock()

        # EdgeModel — fórmula correta de edge
        self._edge_model = EdgeModel(EdgeConfig(
            fee_rate=self._config.fee_estimate,
            slippage_rate=self._config.slippage_estimate,
            min_edge_threshold=min(
                self._config.edge_buy_threshold,
                self._config.edge_sell_threshold,
            ),
        ))

        # Estado por token
        self._markets: Dict[str, MarketState] = {}

        # Históricos (mantidos para compatibilidade com dashboard)
        self._edge_history: Dict[str, Deque] = {}
        self._mid_history: Dict[str, Deque] = {}
        self._signal_history: Deque = deque(maxlen=DEFAULT_SIGNAL_HISTORY_MAXLEN)

        # Métricas de edge
        self._edge_buckets: Dict[str, List[Dict[str, Any]]] = {}
        self._total_signals: int = 0
        self._total_updates: int = 0

    def set_oracle(self, oracle: Any) -> None:
        """Define o ProbabilityOracle (chamado pelo orchestrator após init)."""
        with self._lock:
            self._oracle = oracle

    def process_snapshot(self, snapshot: Any) -> Optional[StrategySignal]:
        """
        Processa MarketSnapshot e retorna StrategySignal se acionável.

        Usa EdgeModel para cálculo correto:
        - BUY:  net_edge = P_oracle - P_ask - fees - slippage
        - SELL: net_edge = P_bid - P_oracle - fees - slippage

        Returns None se:
        - Dados inválidos (bid/ask)
        - Oracle não disponível ou não suporta token
        - Net edge abaixo do threshold
        - Oracle bloqueado (Brier > 0.4)
        """
        bid = snapshot.best_bid_price
        ask = snapshot.best_ask_price

        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None
        if ask <= bid:
            return None

        mid = (bid + ask) / 2.0
        spread = ask - bid
        token_id = snapshot.token_id
        now = snapshot.timestamp or time.time()

        with self._lock:
            self._total_updates += 1
            state = self._get_or_create_market(token_id, now)
            state.snapshot_count += 1
            state.last_mid = mid
            state.last_spread = spread
            state.last_best_bid = bid
            state.last_best_ask = ask
            state.last_update_ts = now

            # Histórico de midpoints
            if token_id not in self._mid_history:
                self._mid_history[token_id] = deque(maxlen=DEFAULT_HISTORY_MAXLEN)
            self._mid_history[token_id].append((now, mid))

        # P_market = midpoint (referência)
        p_market = mid

        # Consultar Oracle (fora do lock principal)
        if self._oracle is None:
            return None

        # Check if Oracle should be blocked
        if hasattr(self._oracle, 'should_block_oracle') and self._oracle.should_block_oracle():
            return None

        estimate = self._oracle.get_estimate(
            token_id=token_id,
            market_data={
                "best_bid": bid,
                "best_ask": ask,
                "mid": mid,
                "spread": spread,
            },
            p_market=p_market,
        )

        if estimate is None:
            return None

        p_oracle = estimate.probability
        oracle_confidence = estimate.confidence

        if oracle_confidence < self._config.min_oracle_confidence:
            return None

        # Edge decay tracking — atualizar preço para sinais ativos
        if hasattr(self._oracle, 'update_decay_tracking'):
            self._oracle.update_decay_tracking(token_id, p_market)

        # Calcular net_edge via EdgeModel (fórmula correta)
        edge_result: EdgeResult = self._edge_model.compute_net_edge(
            p_oracle=p_oracle,
            best_bid=bid,
            best_ask=ask,
        )
        raw_edge = edge_result.raw_edge
        net_edge = edge_result.net_edge

        with self._lock:
            # Atualizar estado
            state.last_p_oracle = p_oracle
            state.last_p_market = p_market
            state.last_edge = raw_edge
            state.last_net_edge = net_edge
            state.last_oracle_confidence = oracle_confidence

            # Histórico de edges
            if token_id not in self._edge_history:
                self._edge_history[token_id] = deque(maxlen=DEFAULT_HISTORY_MAXLEN)
            self._edge_history[token_id].append((now, net_edge))

            # Verificar safety
            safe_to_buy = True
            if self._safety_state is not None:
                safe_to_buy = not self._safety_state.buy_blocked

            # Spread limits
            if spread < self._config.min_spread or spread > self._config.max_spread:
                return None

            # Adaptive sizing multiplier do Oracle
            oracle_sizing_k = 1.0
            if hasattr(self._oracle, 'get_adaptive_sizing_multiplier'):
                oracle_sizing_k = self._oracle.get_adaptive_sizing_multiplier()

            # Combinação de confidence: oracle * adaptive
            combined_confidence = oracle_confidence * oracle_sizing_k

            # Determinar sinal via EdgeModel side + thresholds
            side: Optional[str] = None
            rationale: List[str] = []

            if edge_result.side == "BUY" and net_edge >= self._config.edge_buy_threshold:
                if safe_to_buy:
                    side = "BUY"
                    rationale = [
                        f"p_oracle={p_oracle:.3f}",
                        f"exec_price={edge_result.execution_price:.3f}",
                        f"raw_edge={raw_edge:.4f}",
                        f"net_edge={net_edge:.4f}",
                        f"EV={edge_result.expected_value:.4f}",
                        f"oracle_conf={oracle_confidence:.2f}",
                    ]
            elif edge_result.side == "SELL" and net_edge >= self._config.edge_sell_threshold:
                side = "SELL"
                rationale = [
                    f"p_oracle={p_oracle:.3f}",
                    f"exec_price={edge_result.execution_price:.3f}",
                    f"raw_edge={raw_edge:.4f}",
                    f"net_edge={net_edge:.4f}",
                    f"EV={edge_result.expected_value:.4f}",
                    f"oracle_conf={oracle_confidence:.2f}",
                ]

            # net_edge for signal: positive for both BUY and SELL
            signal_net_edge = net_edge if side == "BUY" else -net_edge if side == "SELL" else net_edge

            # Registrar bucket de edge (para métricas)
            self._record_edge_bucket(token_id, net_edge, side is not None)

            if side is not None:
                self._total_signals += 1

                # Registrar edge decay
                if hasattr(self._oracle, 'register_edge_signal'):
                    self._oracle.register_edge_signal(
                        token_id, net_edge, p_market, p_oracle,
                    )

                signal = StrategySignal(
                    timestamp=now,
                    token_id=token_id,
                    side=side,
                    z_score=signal_net_edge,  # COMPAT: z_score field contém net_edge
                    midpoint=mid,
                    spread=spread,
                    edge_estimate=signal_net_edge,
                    confidence=combined_confidence,
                    p_est=p_oracle,
                    best_bid=bid,
                    best_ask=ask,
                    rationale=tuple(rationale),
                )
                self._signal_history.append({
                    "ts": now,
                    "token_id": token_id,
                    "side": side,
                    "z": round(signal_net_edge, 4),
                    "mid": round(mid, 4),
                    "spread": round(spread, 4),
                    "edge": round(signal_net_edge, 4),
                    "confidence": round(combined_confidence, 3),
                    "p_oracle": round(p_oracle, 4),
                    "p_market": round(p_market, 4),
                    "raw_edge": round(raw_edge, 4),
                    "exec_price": round(edge_result.execution_price, 4),
                    "ev": round(edge_result.expected_value, 4),
                })
                logger.info(
                    "[STRATEGY] Signal: %s %s net_edge=%.4f EV=%.4f "
                    "p_o=%.3f exec=%.3f conf=%.2f",
                    side, token_id[:12], net_edge,
                    edge_result.expected_value,
                    p_oracle, edge_result.execution_price,
                    combined_confidence,
                )
                return signal

            return None

    def _get_or_create_market(self, token_id: str, ts: float) -> MarketState:
        """Obtém ou cria MarketState para um token."""
        if token_id not in self._markets:
            self._markets[token_id] = MarketState(
                token_id=token_id,
                last_update_ts=ts,
            )
        return self._markets[token_id]

    def _record_edge_bucket(
        self, token_id: str, net_edge: float, signal_generated: bool,
    ) -> None:
        """Registra edge no bucket para métricas de performance."""
        bucket_key = self._edge_to_bucket(net_edge)
        if bucket_key not in self._edge_buckets:
            self._edge_buckets[bucket_key] = []
        self._edge_buckets[bucket_key].append({
            "ts": time.time(),
            "token_id": token_id,
            "net_edge": net_edge,
            "signal": signal_generated,
        })
        # Limitar tamanho
        if len(self._edge_buckets[bucket_key]) > 500:
            self._edge_buckets[bucket_key] = self._edge_buckets[bucket_key][-200:]

    @staticmethod
    def _edge_to_bucket(edge: float) -> str:
        """Converte edge em bucket label."""
        abs_edge = abs(edge)
        if abs_edge < 0.01:
            return "<1%"
        elif abs_edge < 0.03:
            return "1-3%"
        elif abs_edge < 0.05:
            return "3-5%"
        elif abs_edge < 0.10:
            return "5-10%"
        else:
            return ">10%"

    # ------------------------------------------------------------------
    # Leitura de métricas (interface pública — compatível com dashboard)
    # ------------------------------------------------------------------

    def get_market_data(self) -> List[Dict[str, Any]]:
        """Retorna dados de mercado atuais para todos os tokens monitorados."""
        with self._lock:
            result = []
            for token_id, state in self._markets.items():
                edge_hist = self._edge_history.get(token_id, deque())
                current_edge = edge_hist[-1][1] if edge_hist else None
                result.append({
                    "token_id": token_id,
                    "midpoint": round(state.last_mid, 4) if state.last_mid else None,
                    "spread": round(state.last_spread, 4) if state.last_spread else None,
                    "best_bid": state.last_best_bid,
                    "best_ask": state.last_best_ask,
                    # COMPAT: z_score field → net_edge
                    "z_score": round(current_edge, 4) if current_edge is not None else None,
                    "snapshot_count": state.snapshot_count,
                    "last_update": state.last_update_ts,
                    # Novos campos para oracle
                    "p_oracle": round(state.last_p_oracle, 4) if state.last_p_oracle is not None else None,
                    "p_market": round(state.last_p_market, 4) if state.last_p_market is not None else None,
                    "net_edge": round(state.last_net_edge, 4) if state.last_net_edge is not None else None,
                    "raw_edge": round(state.last_edge, 4) if state.last_edge is not None else None,
                    "oracle_confidence": round(state.last_oracle_confidence, 3) if state.last_oracle_confidence is not None else None,
                    # COMPAT: manter ewma_mean/var para dashboard existente
                    "ewma_mean": round(state.last_mid, 4) if state.last_mid else 0.0,
                    "ewma_var": 0.0,
                })
            return result

    def get_z_history(self, token_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de edge (campo z para compat) para um token."""
        with self._lock:
            hist = self._edge_history.get(token_id, deque())
            entries = list(hist)[-limit:]
            return [{"ts": ts, "z": round(e, 4)} for ts, e in entries]

    def get_mid_history(self, token_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de midpoints para um token."""
        with self._lock:
            hist = self._mid_history.get(token_id, deque())
            entries = list(hist)[-limit:]
            return [{"ts": ts, "mid": round(m, 4)} for ts, m in entries]

    def get_signal_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna últimos sinais gerados."""
        with self._lock:
            entries = list(self._signal_history)
            return entries[-limit:]

    def get_strategy_summary(self) -> Dict[str, Any]:
        """Retorna resumo da estratégia."""
        with self._lock:
            markets_count = len(self._markets)
            total_snapshots = sum(
                s.snapshot_count for s in self._markets.values()
            )
            signals_24h = sum(
                1 for s in self._signal_history
                if s["ts"] > time.time() - 86400
            )
            buy_signals = sum(
                1 for s in self._signal_history
                if s["side"] == "BUY" and s["ts"] > time.time() - 86400
            )
            sell_signals = sum(
                1 for s in self._signal_history
                if s["side"] == "SELL" and s["ts"] > time.time() - 86400
            )

            # Edge métricas
            all_edges = []
            for hist in self._edge_history.values():
                all_edges.extend(e for _, e in hist)

            avg_edge = (
                sum(all_edges) / len(all_edges) if all_edges else None
            )

            # Oracle métricas
            oracle_metrics = {}
            if self._oracle is not None and hasattr(self._oracle, 'get_full_metrics'):
                try:
                    oracle_metrics = self._oracle.get_full_metrics()
                except Exception:
                    pass

            # Edge bucket distribution
            edge_distribution = {}
            for bucket_key, entries in self._edge_buckets.items():
                count = len(entries)
                signals = sum(1 for e in entries if e.get("signal"))
                edge_distribution[bucket_key] = {
                    "count": count,
                    "signals": signals,
                    "signal_rate": round(signals / count, 3) if count > 0 else 0,
                }

            return {
                "strategy": "Probability Edge (Oracle-based)",
                "markets_monitored": markets_count,
                "total_snapshots_processed": total_snapshots,
                "total_signals_generated": self._total_signals,
                "signals_last_24h": signals_24h,
                "buy_signals_24h": buy_signals,
                "sell_signals_24h": sell_signals,
                "config": {
                    "edge_buy_threshold": self._config.edge_buy_threshold,
                    "edge_sell_threshold": self._config.edge_sell_threshold,
                    "fee_estimate": self._config.fee_estimate,
                    "slippage_estimate": self._config.slippage_estimate,
                    "spread_impact_factor": self._config.spread_impact_factor,
                    "min_oracle_confidence": self._config.min_oracle_confidence,
                    "min_spread": self._config.min_spread,
                    "max_spread": self._config.max_spread,
                },
                "edge_metrics": {
                    "avg_net_edge": round(avg_edge, 4) if avg_edge is not None else None,
                    "edge_distribution": edge_distribution,
                },
                "oracle_metrics": oracle_metrics,
            }

    @property
    def metrics_collector(self) -> Any:
        """COMPAT: Retorna self como proxy para métricas.

        O dashboard existente chama system['reversion_metrics_collector'].
        Retornamos um objeto com get_metrics() compatível.
        """
        return _CompatMetricsProxy(self)


class _CompatMetricsProxy:
    """Proxy de métricas para compatibilidade com dashboard técnico."""

    def __init__(self, engine: StrategyEngine) -> None:
        self._engine = engine

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas no formato esperado pelo dashboard técnico."""
        summary = self._engine.get_strategy_summary()
        edge_m = summary.get("edge_metrics", {})
        oracle_m = summary.get("oracle_metrics", {})
        perf = oracle_m.get("performance", {})

        return {
            "current_z": edge_m.get("avg_net_edge"),
            "rolling_z_mean": edge_m.get("avg_net_edge"),
            "rolling_z_std": None,
            "signal_count_last_24h": summary.get("signals_last_24h", 0),
            "hit_rate_simulated": None,
            "average_reversion_magnitude": None,
            "average_time_to_reversion": None,
            # Novos campos do oracle
            "brier_score": perf.get("brier_score"),
            "oracle_blocked": oracle_m.get("oracle_blocked", False),
            "adaptive_sizing_k": oracle_m.get("adaptive_sizing_multiplier"),
            "edge_decay": oracle_m.get("edge_decay", {}),
        }
