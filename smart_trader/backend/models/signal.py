"""
Modelos de dados para o Smart Event Trader
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any
import uuid


class Categoria(Enum):
    POLITICO = "POLÍTICO"
    TECNOLOGICO = "TECNOLÓGICO"
    ESPORTIVO = "ESPORTIVO"
    ECONOMICO = "ECONÔMICO"
    CLIMATICO = "CLIMÁTICO"
    SAUDE = "SAÚDE"


class Urgencia(Enum):
    CRITICA = "CRÍTICA"      # >= 80%
    NORMAL = "NORMAL"        # >= 65%
    MONITORAR = "MONITORAR"  # >= 50%
    IGNORAR = "IGNORAR"      # < 50%


# Multiplicadores de categoria (baseado em taxa de acerto histórica)
CATEGORIA_MULTIPLIERS = {
    Categoria.POLITICO: 1.3,      # 90%+ acerto
    Categoria.TECNOLOGICO: 1.2,   # 75%+ acerto
    Categoria.ESPORTIVO: 1.1,     # 70%+ acerto
    Categoria.ECONOMICO: 0.95,    # 60%+ acerto
    Categoria.CLIMATICO: 0.9,     # 65%+ acerto
    Categoria.SAUDE: 0.85,        # 55%+ acerto
}


# Keywords por categoria para NLP
CATEGORIA_KEYWORDS = {
    Categoria.POLITICO: [
        "coup", "arrested", "resign", "president removed", "war", "election",
        "impeachment", "assassination", "military", "overthrow", "dictator",
        "prime minister", "parliament", "sanctions", "treaty", "embassy",
        "golpe", "preso", "renúncia", "presidente", "guerra", "eleição"
    ],
    Categoria.TECNOLOGICO: [
        "launches", "FDA approval", "AI breakthrough", "merger", "IPO",
        "acquisition", "startup", "funding", "patent", "innovation",
        "ChatGPT", "OpenAI", "Google", "Apple", "Microsoft", "Meta",
        "lançamento", "aprovação", "fusão", "aquisição", "tecnologia"
    ],
    Categoria.ESPORTIVO: [
        "won championship", "final result", "championship match", "gold medal",
        "world cup", "super bowl", "olympics", "playoffs", "MVP", "record",
        "campeão", "final", "copa do mundo", "olimpíadas", "recorde"
    ],
    Categoria.CLIMATICO: [
        "hurricane", "earthquake", "flood", "disaster", "tsunami", "wildfire",
        "tornado", "storm", "climate emergency", "evacuation", "death toll",
        "furacão", "terremoto", "enchente", "desastre", "incêndio"
    ],
    Categoria.ECONOMICO: [
        "fed decision", "interest rate", "inflation", "recession", "GDP",
        "unemployment", "stock market crash", "bitcoin", "crypto", "bank",
        "federal reserve", "ECB", "default", "bankruptcy", "bailout",
        "juros", "inflação", "recessão", "PIB", "desemprego", "banco"
    ],
    Categoria.SAUDE: [
        "vaccine approved", "outbreak", "pandemic", "variant", "WHO",
        "CDC", "FDA", "clinical trial", "drug approved", "epidemic",
        "vacina", "surto", "pandemia", "variante", "OMS"
    ]
}


@dataclass
class Evento:
    """Representa um evento/notícia detectado"""
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    titulo: str = ""
    descricao: str = ""
    fonte: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    mentions: int = 0
    trending_rank: int = 0
    url: str = ""
    categoria: Optional[Categoria] = None
    keywords_matched: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WhaleActivity:
    """Representa atividade de whale detectada"""
    id: str = field(default_factory=lambda: f"whale_{uuid.uuid4().hex[:8]}")
    whales_detectados: int = 0
    volume_total_usd: Decimal = Decimal("0")
    timestamp_movimento: datetime = field(default_factory=datetime.utcnow)
    tempo_antes_noticia: Optional[str] = None
    tempo_minutos: Optional[int] = None
    carteiras: List[str] = field(default_factory=list)
    destino_contrato: str = ""
    acao_whale: str = ""  # "COMPRAR SIM", "COMPRAR NAO", "VENDER"
    transacoes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass 
class MercadoPolymarket:
    """Representa um mercado no Polymarket"""
    id: str = ""
    titulo: str = ""
    link: str = ""
    preco_atual: Decimal = Decimal("0.5")
    preco_24h_atras: Decimal = Decimal("0.5")
    movimento_percent: float = 0.0
    liquidity: Decimal = Decimal("0")
    volume_1h: Decimal = Decimal("0")
    volume_24h: Decimal = Decimal("0")
    lado_recomendado: str = "SIM"  # "SIM" ou "NAO"
    end_date: Optional[datetime] = None


@dataclass
class Recomendacao:
    """Recomendação de trading"""
    acao: str = "COMPRAR SIM"
    tamanho_posicao_sugerido: Decimal = Decimal("100")
    quantidade_shares: int = 0
    preco_entrada: Decimal = Decimal("0.5")
    alvo_principal: Decimal = Decimal("0.80")
    alvo_secundario: Decimal = Decimal("0.85")
    stop_loss: Decimal = Decimal("0.20")
    roi_esperado: str = "+0%"
    timing_esperado: str = "72 horas"
    risco: str = "Moderado"  # "Baixo", "Moderado", "Alto"


@dataclass
class AnaliseDetalhada:
    """Análise detalhada do sinal"""
    por_que_move_mercado: str = ""
    precedentes_similares: List[str] = field(default_factory=list)
    janela_de_tempo: str = ""
    fatores_risco: List[str] = field(default_factory=list)


@dataclass
class HistoricoCategoria:
    """Histórico de performance por categoria"""
    categoria: Categoria = Categoria.POLITICO
    sinais_passados: int = 0
    acertos: int = 0
    taxa_acerto: float = 0.0
    roi_medio: float = 0.0


@dataclass
class Signal:
    """Sinal completo de trading"""
    id: str = field(default_factory=lambda: f"sig_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:3]}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    categoria: Categoria = Categoria.POLITICO
    confianca: float = 0.0
    urgencia: Urgencia = Urgencia.IGNORAR
    
    # Componentes do score
    news_score: float = 0.0
    whale_score: float = 0.0
    momentum_score: float = 0.0
    temporal_score: float = 0.0
    base_score: float = 0.0
    
    # Dados
    evento: Evento = field(default_factory=Evento)
    whale_activity: WhaleActivity = field(default_factory=WhaleActivity)
    mercados: List[MercadoPolymarket] = field(default_factory=list)
    recomendacao: Recomendacao = field(default_factory=Recomendacao)
    analise: AnaliseDetalhada = field(default_factory=AnaliseDetalhada)
    historico: HistoricoCategoria = field(default_factory=HistoricoCategoria)
    
    # Link de ação
    link_acao: str = ""
    texto_botao: str = "🚀 ENTRAR AGORA"
    instrucoes_popup: str = ""
    
    # Status
    ativo: bool = True
    resultado_real: Optional[str] = None  # "GANHOU", "PERDEU", "PENDENTE"
    roi_real: Optional[float] = None
    
    def calcular_urgencia(self):
        """Define urgência baseada na confiança"""
        if self.confianca >= 80:
            self.urgencia = Urgencia.CRITICA
        elif self.confianca >= 65:
            self.urgencia = Urgencia.NORMAL
        elif self.confianca >= 50:
            self.urgencia = Urgencia.MONITORAR
        else:
            self.urgencia = Urgencia.IGNORAR
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário JSON-serializável"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "categoria": self.categoria.value,
            "confianca": round(self.confianca, 1),
            "urgencia": self.urgencia.value,
            
            "scores": {
                "news": round(self.news_score, 1),
                "whale": round(self.whale_score, 1),
                "momentum": round(self.momentum_score, 1),
                "temporal": round(self.temporal_score, 1),
                "base": round(self.base_score, 1)
            },
            
            "evento": {
                "titulo": self.evento.titulo,
                "descricao": self.evento.descricao,
                "fonte": self.evento.fonte,
                "timestamp_noticia": self.evento.timestamp.isoformat(),
                "mentions": self.evento.mentions,
                "trending_rank": self.evento.trending_rank,
                "url": self.evento.url
            },
            
            "whale_activity": {
                "whales_detectados": self.whale_activity.whales_detectados,
                "volume_total_usd": float(self.whale_activity.volume_total_usd),
                "timestamp_movimento": self.whale_activity.timestamp_movimento.isoformat(),
                "tempo_antes_noticia": self.whale_activity.tempo_antes_noticia,
                "tempo_minutos": self.whale_activity.tempo_minutos,
                "carteiras": self.whale_activity.carteiras[:5],  # Limita a 5
                "destino": self.whale_activity.destino_contrato,
                "acao_whale": self.whale_activity.acao_whale
            },
            
            "mercados_polymarket": [
                {
                    "id": m.id,
                    "titulo": m.titulo,
                    "link": m.link,
                    "preco_atual": float(m.preco_atual),
                    "preco_24h_atras": float(m.preco_24h_atras),
                    "movimento": f"{m.movimento_percent:+.1f}%",
                    "liquidity": float(m.liquidity),
                    "volume_1h": float(m.volume_1h),
                    "lado_recomendado": m.lado_recomendado
                }
                for m in self.mercados
            ],
            
            "recomendacao": {
                "acao": self.recomendacao.acao,
                "tamanho_posicao_sugerido": float(self.recomendacao.tamanho_posicao_sugerido),
                "quantidade_shares": self.recomendacao.quantidade_shares,
                "preco_entrada": float(self.recomendacao.preco_entrada),
                "alvo_principal": float(self.recomendacao.alvo_principal),
                "alvo_secundario": float(self.recomendacao.alvo_secundario),
                "stop_loss": float(self.recomendacao.stop_loss),
                "roi_esperado": self.recomendacao.roi_esperado,
                "timing_esperado": self.recomendacao.timing_esperado,
                "risco": self.recomendacao.risco
            },
            
            "analise_detalhada": {
                "por_que_isso_move_mercado": self.analise.por_que_move_mercado,
                "precedentes_similares": self.analise.precedentes_similares,
                "janela_de_tempo": self.analise.janela_de_tempo,
                "fatores_risco": self.analise.fatores_risco
            },
            
            "botao_acao": {
                "texto_botao": self.texto_botao,
                "link_direto": self.link_acao,
                "instrucoes_popup": self.instrucoes_popup
            },
            
            "historico_categoria": {
                f"{self.categoria.value.lower()}_sinais_passados": self.historico.sinais_passados,
                f"{self.categoria.value.lower()}_acertos": self.historico.acertos,
                f"{self.categoria.value.lower()}_taxa_acerto": f"{self.historico.taxa_acerto:.0f}%",
                f"{self.categoria.value.lower()}_roi_medio": f"+{self.historico.roi_medio:.0f}%"
            },
            
            "status": {
                "ativo": self.ativo,
                "resultado_real": self.resultado_real,
                "roi_real": self.roi_real
            }
        }

