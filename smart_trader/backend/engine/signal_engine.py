"""
Signal Engine - Motor de Geração de Sinais
==========================================
Combina dados de todas as fontes e gera sinais de trading.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from models.signal import (
        Signal, Evento, WhaleActivity, MercadoPolymarket,
        Recomendacao, AnaliseDetalhada, HistoricoCategoria,
        Categoria, Urgencia, CATEGORIA_MULTIPLIERS
    )
    from monitors.news_monitor import NewsMonitor, TrendingEvent
    from monitors.whale_monitor import WhaleMonitor
    from monitors.polymarket_monitor import PolymarketMonitor, MarketMomentum
except ImportError:
    from ..models.signal import (
        Signal, Evento, WhaleActivity, MercadoPolymarket,
        Recomendacao, AnaliseDetalhada, HistoricoCategoria,
        Categoria, Urgencia, CATEGORIA_MULTIPLIERS
    )
    from ..monitors.news_monitor import NewsMonitor, TrendingEvent
    from ..monitors.whale_monitor import WhaleMonitor
    from ..monitors.polymarket_monitor import PolymarketMonitor, MarketMomentum

logger = logging.getLogger(__name__)


# Histórico de performance por categoria (dados iniciais para bootstrap)
CATEGORIA_HISTORICO = {
    Categoria.POLITICO: {"sinais": 47, "acertos": 43, "roi": 185},
    Categoria.TECNOLOGICO: {"sinais": 34, "acertos": 26, "roi": 95},
    Categoria.ESPORTIVO: {"sinais": 28, "acertos": 20, "roi": 65},
    Categoria.ECONOMICO: {"sinais": 31, "acertos": 19, "roi": 45},
    Categoria.CLIMATICO: {"sinais": 19, "acertos": 12, "roi": 55},
    Categoria.SAUDE: {"sinais": 16, "acertos": 9, "roi": 35},
}

# Precedentes por categoria para análise
PRECEDENTES = {
    Categoria.POLITICO: [
        "Golpe na Bolívia 2019: mercados moveram 25%",
        "Indictment Trump 2023: mercados subiram 15%",
        "Eleição surpresa UK 2024: movimento de 40%",
        "Prisão de líder político: histórico de +30% volatilidade"
    ],
    Categoria.TECNOLOGICO: [
        "Lançamento ChatGPT: mercados de IA subiram 200%",
        "Aprovação FDA: média de +85% em mercados relacionados",
        "Merger anunciado: movimento médio de 45%",
        "IPO tech: volatilidade média de 60%"
    ],
    Categoria.ESPORTIVO: [
        "Final Super Bowl: volume 500% acima do normal",
        "Lesão de jogador estrela: movimento de 20-40%",
        "Escândalo em time: queda média de 30%"
    ],
    Categoria.ECONOMICO: [
        "Decisão Fed surpresa: movimento de 15-25%",
        "Dados de inflação: volatilidade de 10-20%",
        "Default de país: pânico e 50%+ movimento"
    ],
    Categoria.CLIMATICO: [
        "Furacão categoria 5: mercados de impacto movem 35%",
        "Terremoto major: incerteza gera 25% volatilidade"
    ],
    Categoria.SAUDE: [
        "Aprovação de vacina: +100% em mercados relacionados",
        "Nova variante: pânico gera 40% movimento",
        "Surto confirmado: volatilidade de 30%"
    ]
}


@dataclass
class SignalComponents:
    """Componentes para cálculo do sinal"""
    news_score: float = 0.0
    whale_score: float = 0.0
    momentum_score: float = 0.0
    temporal_score: float = 0.0
    
    trending_event: Optional[TrendingEvent] = None
    whale_activity: Optional[WhaleActivity] = None
    market_momentum: Optional[MarketMomentum] = None
    related_markets: List[MercadoPolymarket] = None
    
    def __post_init__(self):
        if self.related_markets is None:
            self.related_markets = []


class SignalEngine:
    """
    Motor de geração de sinais.
    
    Combina:
    - Breaking News (30% do score)
    - Whale Movement (40% do score)
    - Polymarket Momentum (20% do score)
    - Temporal Correlation (10% do score)
    
    Gera sinais com confiança de 0-100%.
    """
    
    # Pesos das fontes
    WEIGHT_NEWS = 0.30
    WEIGHT_WHALE = 0.40
    WEIGHT_MOMENTUM = 0.20
    WEIGHT_TEMPORAL = 0.10
    
    # Thresholds
    MIN_CONFIDENCE_ALERT = 50
    
    def __init__(
        self,
        news_monitor: NewsMonitor,
        whale_monitor: WhaleMonitor,
        polymarket_monitor: PolymarketMonitor
    ):
        self.news_monitor = news_monitor
        self.whale_monitor = whale_monitor
        self.polymarket_monitor = polymarket_monitor
        
        # Cache de sinais
        self.active_signals: Dict[str, Signal] = {}
        self.signal_history: List[Signal] = []
        
        # Callbacks
        self.callbacks: List[callable] = []
    
    def on_signal(self, callback: callable):
        """Registra callback para novos sinais"""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self, signal: Signal):
        """Notifica callbacks sobre novo sinal"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Erro no callback: {e}")
    
    # =========================================================================
    # CÁLCULO DE SCORES
    # =========================================================================
    
    def _calculate_news_score(self, event: TrendingEvent) -> float:
        """
        Calcula score de news (0-100).
        
        Fatores:
        - Trending strength
        - Número de fontes
        - Velocidade de propagação
        """
        if not event:
            return 0.0
        
        # Trending strength já é 0-100
        base = event.strength
        
        # Bonus por múltiplas fontes
        source_bonus = min(20, event.sources_count * 5)
        
        # Bonus por velocidade
        velocity_bonus = min(15, event.velocity * 0.5)
        
        return min(100, base + source_bonus + velocity_bonus)
    
    def _calculate_whale_score(self, activity: WhaleActivity) -> float:
        """
        Calcula score de whale (0-100).
        
        Fatores:
        - Volume total
        - Número de whales
        - Se é direcionado ao Polymarket
        """
        if not activity:
            return 0.0
        
        # Volume score (100 = $5M+)
        volume_score = min(100, float(activity.volume_total_usd) / 50000)
        
        # Whales score (bonus por múltiplos whales)
        whales_bonus = min(20, activity.whales_detectados * 5)
        
        # Bonus se destino é Polymarket
        polymarket_bonus = 10 if "polymarket" in activity.destino_contrato.lower() else 0
        
        return min(100, volume_score + whales_bonus + polymarket_bonus)
    
    def _calculate_momentum_score(self, momentum: MarketMomentum) -> float:
        """
        Calcula score de momentum (0-100).
        
        Já calculado pelo monitor, mas adiciona ajustes.
        """
        if not momentum:
            return 0.0
        
        return momentum.momentum_score
    
    def _calculate_temporal_score(
        self,
        whale_timestamp: Optional[datetime],
        news_timestamp: Optional[datetime]
    ) -> float:
        """
        Calcula score temporal (-20 a +30).
        
        Whale antes da notícia = insider, score alto
        Whale depois = normal, score negativo
        """
        if not whale_timestamp or not news_timestamp:
            return 0.0
        
        correlation = self.whale_monitor.calculate_temporal_correlation(
            whale_timestamp, news_timestamp
        )
        
        return correlation["score"]
    
    def _calculate_final_confidence(
        self,
        components: SignalComponents,
        categoria: Categoria
    ) -> Tuple[float, float]:
        """
        Calcula confiança final.
        
        Returns:
            (base_score, final_confidence)
        """
        # Base score ponderado
        base_score = (
            components.news_score * self.WEIGHT_NEWS +
            components.whale_score * self.WEIGHT_WHALE +
            components.momentum_score * self.WEIGHT_MOMENTUM +
            components.temporal_score * self.WEIGHT_TEMPORAL
        )
        
        # Aplica multiplicador de categoria
        multiplier = CATEGORIA_MULTIPLIERS.get(categoria, 1.0)
        confidence = base_score * multiplier
        
        # Cap em 0-100
        confidence = max(0, min(100, confidence))
        
        return base_score, confidence
    
    # =========================================================================
    # GERAÇÃO DE RECOMENDAÇÃO
    # =========================================================================
    
    def _generate_recommendation(
        self,
        market: MercadoPolymarket,
        confidence: float,
        whale_action: str
    ) -> Recomendacao:
        """
        Gera recomendação de trading baseada nos dados.
        """
        # Determina lado (SIM ou NAO)
        if "COMPRAR" in whale_action.upper() or "SIM" in whale_action.upper():
            lado = "SIM"
            preco_entrada = market.preco_atual
        else:
            lado = "NAO"
            preco_entrada = Decimal("1") - market.preco_atual
        
        # Calcula alvos baseado na confiança
        if confidence >= 80:
            alvo_principal = min(Decimal("0.95"), preco_entrada + Decimal("0.30"))
            alvo_secundario = min(Decimal("0.98"), preco_entrada + Decimal("0.40"))
            stop_loss = max(Decimal("0.05"), preco_entrada - Decimal("0.20"))
            posicao = Decimal("1000")  # Alta confiança = posição maior
            risco = "Moderado"
        elif confidence >= 65:
            alvo_principal = min(Decimal("0.85"), preco_entrada + Decimal("0.20"))
            alvo_secundario = min(Decimal("0.90"), preco_entrada + Decimal("0.30"))
            stop_loss = max(Decimal("0.10"), preco_entrada - Decimal("0.15"))
            posicao = Decimal("500")
            risco = "Moderado"
        else:
            alvo_principal = min(Decimal("0.75"), preco_entrada + Decimal("0.15"))
            alvo_secundario = min(Decimal("0.80"), preco_entrada + Decimal("0.20"))
            stop_loss = max(Decimal("0.15"), preco_entrada - Decimal("0.10"))
            posicao = Decimal("250")
            risco = "Alto"
        
        # Calcula shares
        quantidade = int(posicao / preco_entrada)
        
        # ROI esperado
        roi = float((alvo_principal - preco_entrada) / preco_entrada * 100)
        
        return Recomendacao(
            acao=f"COMPRAR {lado}",
            tamanho_posicao_sugerido=posicao,
            quantidade_shares=quantidade,
            preco_entrada=preco_entrada,
            alvo_principal=alvo_principal,
            alvo_secundario=alvo_secundario,
            stop_loss=stop_loss,
            roi_esperado=f"+{roi:.0f}%",
            timing_esperado="72 horas" if confidence >= 80 else "1 semana",
            risco=risco
        )
    
    def _generate_analysis(
        self,
        evento: Evento,
        categoria: Categoria,
        components: SignalComponents
    ) -> AnaliseDetalhada:
        """
        Gera análise detalhada do sinal.
        """
        # Por que move o mercado
        if categoria == Categoria.POLITICO:
            porque = f"Eventos políticos como '{evento.titulo[:50]}' historicamente causam grande volatilidade nos mercados de previsão devido à incerteza gerada."
        elif categoria == Categoria.TECNOLOGICO:
            porque = f"Notícias de tecnologia têm impacto direto em mercados de inovação e regulação."
        else:
            porque = f"Este evento pode impactar significativamente as probabilidades do mercado."
        
        # Precedentes
        precedentes = PRECEDENTES.get(categoria, [])[:3]
        
        # Janela de tempo
        if components.whale_score >= 70:
            janela = "Urgente - whales já se posicionaram, janela de 24-48h"
        elif components.news_score >= 70:
            janela = "Este evento está viral. Janela de 2-5 dias para resolução"
        else:
            janela = "Evento em desenvolvimento. Monitorar por 1 semana"
        
        # Fatores de risco
        riscos = []
        if components.temporal_score < 0:
            riscos.append("Whale entrou DEPOIS da notícia - pode ser trade tardio")
        if components.momentum_score < 30:
            riscos.append("Mercado ainda não reagiu - pode não haver correlação")
        if len(components.related_markets) == 0:
            riscos.append("Não encontrado mercado relacionado no Polymarket")
        
        return AnaliseDetalhada(
            por_que_move_mercado=porque,
            precedentes_similares=precedentes,
            janela_de_tempo=janela,
            fatores_risco=riscos
        )
    
    def _generate_historico(self, categoria: Categoria) -> HistoricoCategoria:
        """Retorna histórico de performance da categoria"""
        hist = CATEGORIA_HISTORICO.get(categoria, {"sinais": 0, "acertos": 0, "roi": 0})
        
        taxa = (hist["acertos"] / hist["sinais"] * 100) if hist["sinais"] > 0 else 0
        
        return HistoricoCategoria(
            categoria=categoria,
            sinais_passados=hist["sinais"],
            acertos=hist["acertos"],
            taxa_acerto=taxa,
            roi_medio=hist["roi"]
        )
    
    # =========================================================================
    # GERAÇÃO DO SINAL COMPLETO
    # =========================================================================
    
    async def generate_signal(
        self,
        trending_event: TrendingEvent,
        whale_activity: Optional[WhaleActivity] = None
    ) -> Optional[Signal]:
        """
        Gera um sinal completo a partir de um evento trending.
        """
        try:
            evento = trending_event.evento
            categoria = evento.categoria or Categoria.POLITICO
            
            # Busca mercados relacionados
            related_markets = await self.polymarket_monitor.find_related_markets(
                evento.titulo,
                evento.keywords_matched
            )
            
            if not related_markets:
                logger.info(f"Nenhum mercado encontrado para: {evento.titulo[:50]}")
                return None
            
            # Pega o mercado mais relevante
            main_market = related_markets[0]
            
            # Busca momentum do mercado
            momentum = self.polymarket_monitor.get_market_momentum(main_market.id)
            
            # Se não temos whale activity, busca a mais recente
            if not whale_activity:
                whale_activity = self.whale_monitor.get_recent_activity(minutes=120)
            
            # Monta componentes
            components = SignalComponents(
                news_score=self._calculate_news_score(trending_event),
                whale_score=self._calculate_whale_score(whale_activity),
                momentum_score=self._calculate_momentum_score(momentum),
                trending_event=trending_event,
                whale_activity=whale_activity,
                market_momentum=momentum,
                related_markets=related_markets
            )
            
            # Calcula temporal score se temos whale
            if whale_activity:
                components.temporal_score = self._calculate_temporal_score(
                    whale_activity.timestamp_movimento,
                    evento.timestamp
                )
            
            # Calcula confiança
            base_score, confidence = self._calculate_final_confidence(components, categoria)
            
            # Se confiança muito baixa, não gera sinal
            if confidence < self.MIN_CONFIDENCE_ALERT:
                logger.info(f"Confiança muito baixa ({confidence:.0f}%) para: {evento.titulo[:50]}")
                return None
            
            # Gera recomendação
            whale_action = whale_activity.acao_whale if whale_activity else "COMPRAR SIM"
            recomendacao = self._generate_recommendation(main_market, confidence, whale_action)
            
            # Atualiza mercado com lado recomendado
            main_market.lado_recomendado = "SIM" if "SIM" in recomendacao.acao else "NAO"
            
            # Gera análise
            analise = self._generate_analysis(evento, categoria, components)
            
            # Gera histórico
            historico = self._generate_historico(categoria)
            
            # Cria sinal
            signal = Signal(
                categoria=categoria,
                confianca=confidence,
                
                news_score=components.news_score,
                whale_score=components.whale_score,
                momentum_score=components.momentum_score,
                temporal_score=components.temporal_score,
                base_score=base_score,
                
                evento=evento,
                whale_activity=whale_activity or WhaleActivity(),
                mercados=related_markets[:3],  # Top 3
                recomendacao=recomendacao,
                analise=analise,
                historico=historico
            )
            
            # Define urgência
            signal.calcular_urgencia()
            
            # Gera link de ação
            signal.link_acao = main_market.link
            signal.texto_botao = f"🚀 ENTRAR AGORA - {recomendacao.acao}"
            signal.instrucoes_popup = (
                f"Você está prestes a comprar {recomendacao.quantidade_shares} shares "
                f"de {main_market.lado_recomendado} por ${float(recomendacao.tamanho_posicao_sugerido):,.0f}. "
                f"Alvo de lucro: ${float(recomendacao.alvo_principal * recomendacao.quantidade_shares):,.0f}."
            )
            
            # Atualiza tempo antes notícia no whale activity
            if whale_activity and components.temporal_score != 0:
                minutes = abs(int(components.temporal_score))  # Aproximação
                signal.whale_activity.tempo_antes_noticia = f"{minutes} minutos"
                signal.whale_activity.tempo_minutos = minutes
            
            # Adiciona ao cache
            self.active_signals[signal.id] = signal
            self.signal_history.append(signal)
            
            # Notifica callbacks
            await self._notify_callbacks(signal)
            
            logger.info(
                f"🎯 SINAL GERADO: {signal.urgencia.value} ({confidence:.0f}%) - "
                f"{evento.titulo[:40]}..."
            )
            
            return signal
        
        except Exception as e:
            logger.error(f"Erro ao gerar sinal: {e}", exc_info=True)
            return None
    
    # =========================================================================
    # PROCESSAMENTO DE EVENTOS
    # =========================================================================
    
    async def process_trending_event(self, event: TrendingEvent):
        """
        Callback para processar evento trending do news monitor.
        """
        # Verifica se já processamos este evento
        event_key = f"{event.evento.titulo[:30]}_{event.evento.timestamp.date()}"
        
        if event_key in [s.evento.titulo[:30] + "_" + str(s.evento.timestamp.date()) 
                         for s in self.active_signals.values()]:
            return
        
        # Gera sinal
        await self.generate_signal(event)
    
    async def process_whale_activity(self, activity: WhaleActivity):
        """
        Callback para processar atividade de whale.
        """
        # Busca eventos trending recentes que possam estar correlacionados
        # e atualiza sinais existentes
        for signal in list(self.active_signals.values()):
            if signal.whale_score == 0:
                # Atualiza com nova atividade
                signal.whale_activity = activity
                signal.whale_score = self._calculate_whale_score(activity)
                signal.temporal_score = self._calculate_temporal_score(
                    activity.timestamp_movimento,
                    signal.evento.timestamp
                )
                
                # Recalcula confiança
                _, signal.confianca = self._calculate_final_confidence(
                    SignalComponents(
                        news_score=signal.news_score,
                        whale_score=signal.whale_score,
                        momentum_score=signal.momentum_score,
                        temporal_score=signal.temporal_score
                    ),
                    signal.categoria
                )
                signal.calcular_urgencia()
    
    # =========================================================================
    # MÉTODOS PÚBLICOS
    # =========================================================================
    
    def get_active_signals(self) -> List[Signal]:
        """Retorna sinais ativos ordenados por confiança"""
        signals = list(self.active_signals.values())
        signals.sort(key=lambda s: s.confianca, reverse=True)
        return signals
    
    def get_signal_history(
        self,
        limit: int = 100,
        categoria: Optional[Categoria] = None
    ) -> List[Signal]:
        """Retorna histórico de sinais"""
        history = self.signal_history
        
        if categoria:
            history = [s for s in history if s.categoria == categoria]
        
        return history[-limit:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance"""
        stats = {
            "total_sinais": len(self.signal_history),
            "sinais_ativos": len(self.active_signals),
            "por_categoria": {},
            "taxa_acerto_geral": 0.0,
            "roi_medio_geral": 0.0
        }
        
        for categoria in Categoria:
            sinais_cat = [s for s in self.signal_history if s.categoria == categoria]
            ganhos = [s for s in sinais_cat if s.resultado_real == "GANHOU"]
            
            if sinais_cat:
                taxa = len(ganhos) / len(sinais_cat) * 100
                roi = sum(s.roi_real or 0 for s in ganhos) / max(len(ganhos), 1)
            else:
                # Usa dados históricos se não temos dados reais
                hist = CATEGORIA_HISTORICO.get(categoria, {})
                taxa = (hist.get("acertos", 0) / max(hist.get("sinais", 1), 1)) * 100
                roi = hist.get("roi", 0)
            
            stats["por_categoria"][categoria.value] = {
                "sinais": len(sinais_cat) or CATEGORIA_HISTORICO.get(categoria, {}).get("sinais", 0),
                "taxa_acerto": taxa,
                "roi_medio": roi
            }
        
        return stats

