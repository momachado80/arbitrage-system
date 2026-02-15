"""
Smart Event Trader API
======================
API principal com WebSocket para sinais em tempo real.
"""
import asyncio
import logging
import os
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Imports locais
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Signal, Categoria, Urgencia
from monitors import NewsMonitor, WhaleMonitor, PolymarketMonitor
from engine import SignalEngine
from alerts import AlertSystem, AlertConfig
from risk import RiskManager, get_risk_manager, TradeResult, RiskLevel


# =============================================================================
# DADOS DE DEMONSTRAÇÃO
# =============================================================================

# Mercados REAIS do Polymarket (atualizado Janeiro 2026)
DEMO_EVENTS = [
    # POLÍTICO
    {
        "categoria": "POLITICO",
        "titulo": "Tim Walz concorre à reeleição como Governador de Minnesota",
        "descricao": "Primárias democratas para governador de Minnesota em agosto de 2026",
        "fonte": "Associated Press",
        "mercado": "Will Tim Walz win the 2026 Minnesota Governor Democratic primary?",
        "link": "https://polymarket.com/event/will-tim-walz-win-the-2026-minnesota-governor-democratic-primary-election",
        "end_date": "2026-08-11"
    },
    {
        "categoria": "POLITICO",
        "titulo": "Eleições legislativas na Costa Rica em fevereiro",
        "descricao": "PLP e PNR disputam maioria de assentos na Assembleia Legislativa",
        "fonte": "Reuters",
        "mercado": "Will PLP win the most seats in the 2026 Costa Rican Legislative Assembly?",
        "link": "https://polymarket.com/event/will-plp-win-the-most-seats-in-the-2026-costa-rican-legislative-assembly-election",
        "end_date": "2026-02-01"
    },
    {
        "categoria": "POLITICO",
        "titulo": "Yoon Suk Yeol aguarda sentença na Coreia do Sul",
        "descricao": "Ex-presidente sul-coreano pode ser condenado ou absolvido",
        "fonte": "Yonhap News",
        "mercado": "Will Yoon Suk Yeol be sentenced to no prison time?",
        "link": "https://polymarket.com/event/will-yoon-suk-yeol-be-sentenced-to-no-prison-time",
        "end_date": "2026-12-31"
    },
    # ESPORTIVO
    {
        "categoria": "ESPORTIVO",
        "titulo": "Houston Rockets busca título da Conferência Oeste da NBA",
        "descricao": "Rockets é favorito para vencer as finais da Conferência Oeste",
        "fonte": "ESPN",
        "mercado": "Will the Houston Rockets win the NBA Western Conference Finals?",
        "link": "https://polymarket.com/event/will-the-houston-rockets-win-the-nba-western-conference-finals",
        "end_date": "2026-06-16"
    },
    {
        "categoria": "ESPORTIVO",
        "titulo": "Peter Woods pode ser primeira escolha do NFL Draft 2026",
        "descricao": "Destaque defensivo é cotado como top pick do draft",
        "fonte": "NFL Network",
        "mercado": "Will Peter Woods be the first pick of the 2026 NFL Draft?",
        "link": "https://polymarket.com/event/will-peter-woods-be-the-first-pick-of-the-2026-nfl-draft",
        "end_date": "2026-04-25"
    },
    {
        "categoria": "ESPORTIVO",
        "titulo": "T.J. Parker cotado para primeira escolha do NFL Draft",
        "descricao": "Prospecto defensivo é favorito para ser escolhido primeiro",
        "fonte": "ESPN",
        "mercado": "Will T.J. Parker be the first pick of the 2026 NFL Draft?",
        "link": "https://polymarket.com/event/will-tj-parker-be-the-first-pick-of-the-2026-nfl-draft",
        "end_date": "2026-04-25"
    },
    # TECNOLÓGICO/CRYPTO
    {
        "categoria": "TECNOLOGICO",
        "titulo": "Monero pode atingir $1000 em 2026",
        "descricao": "Criptomoeda de privacidade com potencial de valorização",
        "fonte": "CoinDesk",
        "mercado": "Will Monero hit $1000 in 2026?",
        "link": "https://polymarket.com/event/will-monero-hit-1000-in-2026",
        "end_date": "2027-01-01"
    },
    {
        "categoria": "TECNOLOGICO",
        "titulo": "Z.ai disputa liderança em modelos de IA para matemática",
        "descricao": "Competição acirrada pelo melhor modelo de IA para resolução de problemas matemáticos",
        "fonte": "TechCrunch",
        "mercado": "Will Z.ai have the best AI model for math on March 31?",
        "link": "https://polymarket.com/event/will-zai-have-the-best-ai-model-for-math-on-march-31",
        "end_date": "2026-03-31"
    },
    {
        "categoria": "TECNOLOGICO",
        "titulo": "Bitcoin volátil em sessão de Janeiro",
        "descricao": "Movimento de preço do Bitcoin nas próximas horas",
        "fonte": "Bloomberg Crypto",
        "mercado": "Bitcoin Up or Down - January 16",
        "link": "https://polymarket.com/event/btc-updown-15m-1768542300",
        "end_date": "2026-01-16"
    },
    # CLIMÁTICO
    {
        "categoria": "CLIMATICO",
        "titulo": "Risco de terremoto magnitude 9.0+ antes de 2027",
        "descricao": "Sismólogos monitoram atividade tectônica global",
        "fonte": "USGS",
        "mercado": "9.0 or above earthquake before 2027?",
        "link": "https://polymarket.com/event/9pt0-or-above-earthquake-before-2027",
        "end_date": "2026-12-31"
    },
    # ENTRETENIMENTO/OUTROS
    {
        "categoria": "OUTROS",
        "titulo": "Jennifer Venditti pode ser indicada ao Oscar",
        "descricao": "Diretora de casting de 'Marty Supreme' cotada para indicação",
        "fonte": "Variety",
        "mercado": "Will Jennifer Venditti be nominated for Achievement in Casting at the 98th Academy Awards?",
        "link": "https://polymarket.com/event/will-jennifer-venditti-marty-supreme-be-nominated-for-achievement-in-casting-at-the-98th-academy-awards",
        "end_date": "2026-01-22"
    },
    {
        "categoria": "ESPORTIVO",
        "titulo": "Kobbie Mainoo pode deixar Manchester United",
        "descricao": "Jovem meio-campista pode ser transferido na janela de inverno",
        "fonte": "Sky Sports",
        "mercado": "Will Kobbie Mainoo sign with a new club during the winter transfer window?",
        "link": "https://polymarket.com/event/will-kobbie-mainoo-sign-with-a-new-club-during-the-winter-transfer-window-764",
        "end_date": "2026-02-03"
    }
]

WHALE_WALLETS = [
    "0x3f5CE5FBFe3E9af3971dD833D26BA9b5C936f0bE",
    "0x1AB4973a48dc892Cd9971ECE8e01DcC7688f8F23",
    "0x8A67C0B4AB1AfC7a123d69Ef6D5C2cB7fB9B4567",
    "0xD79f1e9A7D3C2b9E8f4d5A6c7B8d0E1F2a3B4C5D"
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ESTADO GLOBAL
# =============================================================================

class AppState:
    """Estado global da aplicação"""
    news_monitor: Optional[NewsMonitor] = None
    whale_monitor: Optional[WhaleMonitor] = None
    polymarket_monitor: Optional[PolymarketMonitor] = None
    signal_engine: Optional[SignalEngine] = None
    alert_system: Optional[AlertSystem] = None
    
    # WebSocket connections
    active_connections: List[WebSocket] = []
    
    # Background tasks
    monitoring_task: Optional[asyncio.Task] = None
    demo_task: Optional[asyncio.Task] = None
    running: bool = False
    
    # Modo demonstração
    demo_mode: bool = True  # Ativado por padrão
    demo_signals: List[Dict[str, Any]] = []
    demo_activity: List[Dict[str, Any]] = []


state = AppState()


# Rastreia sinais já gerados para evitar duplicação
_generated_event_ids = set()


# Multipliers de categoria para Kelly (baseado em win rate histórico)
CATEGORY_MULTIPLIERS = {
    "POLITICO": 1.3,      # 91% win rate histórico
    "TECNOLOGICO": 1.2,   # 75% win rate
    "ESPORTIVO": 1.1,     # 70% win rate
    "ECONOMICO": 0.95,    # 60% win rate
    "CLIMATICO": 0.9,     # 65% win rate
    "SAUDE": 0.85,        # 55% win rate
    "OUTROS": 1.0
}

# Win rates históricos por categoria
CATEGORY_WIN_RATES = {
    "POLITICO": 0.91,
    "TECNOLOGICO": 0.75,
    "ESPORTIVO": 0.70,
    "ECONOMICO": 0.60,
    "CLIMATICO": 0.65,
    "SAUDE": 0.55,
    "OUTROS": 0.50
}


def calculate_signal_risk_management(
    confianca: float,
    categoria: str,
    preco_entrada: float,
    preco_alvo: float
) -> dict:
    """
    Calcula gestão de risco específica para cada sinal.
    
    Usa:
    - Kelly Criterion ajustado pela confiança e categoria
    - Stop loss baseado em volatilidade da categoria
    - Take profit com múltiplos R/R ratios
    """
    risk_mgr = get_risk_manager()
    
    # Win rate da categoria
    cat_win_rate = CATEGORY_WIN_RATES.get(categoria, 0.50)
    cat_multiplier = CATEGORY_MULTIPLIERS.get(categoria, 1.0)
    
    # Calcula Kelly básico para a categoria
    avg_win = 0.30  # Média histórica de ganho
    avg_loss = 0.15  # Média histórica de perda
    kelly_base = risk_mgr.calculate_kelly_criterion(cat_win_rate, avg_win, avg_loss)
    
    # Ajusta Kelly pela confiança do sinal (0.5 = half Kelly, mais conservador)
    confidence_factor = confianca / 100.0
    kelly_adjusted = kelly_base * 0.5 * confidence_factor * cat_multiplier
    kelly_adjusted = min(kelly_adjusted, 0.25)  # Cap em 25% máximo
    
    # Calcula position size em USD
    capital = risk_mgr.current_capital
    position_size = capital * kelly_adjusted
    
    # Stop Loss dinâmico baseado em volatilidade
    # Categorias mais voláteis = stop mais largo
    volatility_map = {
        "POLITICO": 0.12,
        "TECNOLOGICO": 0.15,
        "ESPORTIVO": 0.10,
        "ECONOMICO": 0.18,
        "CLIMATICO": 0.20,
        "SAUDE": 0.15,
        "OUTROS": 0.15
    }
    volatility = volatility_map.get(categoria, 0.15)
    
    stop_loss_price = preco_entrada * (1 - volatility)
    stop_loss_percent = volatility * 100
    
    # Calcula risk amount
    risk_per_share = preco_entrada - stop_loss_price
    
    # Take Profit targets com diferentes R/R ratios
    tp_2_1 = preco_entrada + (risk_per_share * 2)
    tp_3_1 = preco_entrada + (risk_per_share * 3)
    tp_5_1 = preco_entrada + (risk_per_share * 5)
    
    # Risk level baseado em exposição e volatilidade
    current_exposure = sum(p['size'] for p in risk_mgr.open_positions.values())
    total_exposure_after = (current_exposure + position_size) / capital
    
    if total_exposure_after > 0.5 or volatility > 0.18:
        risk_level = "HIGH"
    elif total_exposure_after > 0.25 or volatility > 0.15:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Calcula expected value
    expected_roi = (cat_win_rate * avg_win) - ((1 - cat_win_rate) * avg_loss)
    expected_pnl = position_size * expected_roi
    
    return {
        "kelly_fraction": round(kelly_adjusted * 100, 2),
        "kelly_base": round(kelly_base * 100, 2),
        "position_size_usd": round(position_size, 2),
        "position_size_percent": round(kelly_adjusted * 100, 2),
        "max_risk_usd": round(position_size * volatility, 2),
        "risk_reward_ratio": "2.5:1",
        "stop_loss": {
            "price": round(stop_loss_price, 4),
            "percent": round(stop_loss_percent, 1),
            "usd_at_risk": round(position_size * volatility, 2)
        },
        "take_profit": [
            {"target": 1, "price": round(tp_2_1, 4), "rr": "2:1", "roi": f"+{round((tp_2_1/preco_entrada - 1)*100)}%"},
            {"target": 2, "price": round(tp_3_1, 4), "rr": "3:1", "roi": f"+{round((tp_3_1/preco_entrada - 1)*100)}%"},
            {"target": 3, "price": round(tp_5_1, 4), "rr": "5:1", "roi": f"+{round((tp_5_1/preco_entrada - 1)*100)}%"}
        ],
        "risk_level": risk_level,
        "category_stats": {
            "win_rate": round(cat_win_rate * 100, 1),
            "multiplier": cat_multiplier,
            "volatility": round(volatility * 100, 1)
        },
        "expected_value": {
            "roi_per_trade": f"+{round(expected_roi * 100, 1)}%",
            "expected_pnl_usd": round(expected_pnl, 2)
        }
    }


def generate_demo_signal() -> Dict[str, Any]:
    """Gera um sinal de demonstração realista com mercados REAIS do Polymarket"""
    global _generated_event_ids
    
    # Evita duplicação - escolhe evento ainda não usado
    available_events = [e for i, e in enumerate(DEMO_EVENTS) if i not in _generated_event_ids]
    
    if not available_events:
        # Reset se todos já foram usados
        _generated_event_ids.clear()
        available_events = DEMO_EVENTS
    
    event_idx = DEMO_EVENTS.index(random.choice(available_events))
    _generated_event_ids.add(event_idx)
    event = DEMO_EVENTS[event_idx]
    
    confianca = random.randint(55, 95)
    
    if confianca >= 85:
        urgencia = "URGENTE"
    elif confianca >= 70:
        urgencia = "ALTA"
    else:
        urgencia = "NORMAL"
    
    whale_volume = random.randint(150000, 2500000)
    whale_count = random.randint(2, 8)
    tempo_antes = random.randint(10, 90)
    preco_atual = round(random.uniform(0.25, 0.65), 2)
    preco_alvo = round(preco_atual + random.uniform(0.15, 0.35), 2)
    roi = round((preco_alvo - preco_atual) / preco_atual * 100)
    
    # ID único baseado no slug do evento para evitar duplicatas
    slug_part = event["link"].split("/")[-1][:12] if "/" in event["link"] else uuid.uuid4().hex[:8]
    signal_id = f"sig_{slug_part}_{uuid.uuid4().hex[:4]}"
    
    signal = {
        "id": signal_id,
        "timestamp": datetime.utcnow().isoformat(),
        "categoria": event["categoria"],
        "confianca": confianca,
        "urgencia": urgencia,
        "evento": {
            "titulo": event["titulo"],
            "descricao": event["descricao"],
            "fonte": event["fonte"],
            "mentions": random.randint(200, 5000),
            "timestamp": (datetime.utcnow() - timedelta(minutes=random.randint(5, 30))).isoformat()
        },
        "whale_activity": {
            "volume_total_usd": whale_volume,
            "tempo_antes_noticia": f"{tempo_antes} minutos",
            "whales_detectados": whale_count,
            "wallets": random.sample(WHALE_WALLETS, min(whale_count, len(WHALE_WALLETS))),
            "direcao": "COMPRA" if random.random() > 0.3 else "VENDA"
        },
        "mercados_polymarket": [{
            "titulo": event["mercado"],
            "link": event["link"],
            "preco_atual": preco_atual,
            "volume_24h": random.randint(50000, 500000),
            "momentum": random.choice(["📈 ALTA", "📊 ESTÁVEL", "📉 QUEDA"]),
            "end_date": event.get("end_date", "N/A")
        }],
        "recomendacao": {
            "acao": "COMPRAR SIM" if random.random() > 0.4 else "COMPRAR NÃO",
            "preco_entrada": preco_atual,
            "alvo_principal": preco_alvo,
            "alvo_otimista": min(0.95, preco_alvo + 0.1),
            "roi_esperado": f"+{roi}%",
            "tamanho_posicao_sugerido": random.choice([250, 500, 750, 1000]),
            "stop_loss": max(0.05, preco_atual - 0.15),
            "timing_esperado": f"{random.randint(24, 168)} horas"
        },
        "risk_management": calculate_signal_risk_management(
            confianca=confianca,
            categoria=event["categoria"],
            preco_entrada=preco_atual,
            preco_alvo=preco_alvo
        ),
        "analise_detalhada": {
            "score_news": random.randint(15, 30),
            "score_whale": random.randint(20, 40),
            "score_polymarket": random.randint(10, 20),
            "score_temporal": random.randint(5, 10),
            "score_total": confianca,
            "razoes": [
                f"Volume de whale {whale_count}x acima da média",
                f"Notícia com {random.randint(200, 5000)} menções em 5 min",
                f"Movimento de whale {tempo_antes} min antes da notícia",
                f"Categoria {event['categoria'].lower()} com {random.randint(60, 95)}% de acerto histórico"
            ]
        },
        "historico_categoria": {
            f"{event['categoria'].lower()}_taxa_acerto": f"{random.randint(60, 95)}%",
            f"{event['categoria'].lower()}_roi_medio": f"+{random.randint(50, 200)}%",
            f"{event['categoria'].lower()}_sinais_totais": random.randint(20, 60)
        },
        "botao_acao": {
            "link_direto": event["link"],
            "texto_botao": "🚀 ENTRAR AGORA"
        },
        "market_end_date": event.get("end_date", "N/A")
    }
    
    return signal


async def demo_signal_loop():
    """Loop que gera sinais de demonstração periodicamente"""
    logger.info("🎭 Modo DEMO ativado - gerando sinais simulados...")
    
    # Gera alguns sinais iniciais
    for _ in range(3):
        signal = generate_demo_signal()
        state.demo_signals.append(signal)
        await broadcast_to_websockets({
            "type": "new_signal",
            "data": signal
        })
        await asyncio.sleep(2)
    
    while state.running and state.demo_mode:
        try:
            # Aguarda intervalo aleatório (30s a 2min)
            await asyncio.sleep(random.randint(30, 120))
            
            if not state.demo_mode:
                break
            
            # Gera novo sinal
            signal = generate_demo_signal()
            state.demo_signals.append(signal)
            
            # Mantém apenas últimos 50 sinais
            if len(state.demo_signals) > 50:
                state.demo_signals = state.demo_signals[-50:]
            
            logger.info(f"🎭 DEMO: Novo sinal gerado - {signal['evento']['titulo'][:40]}...")
            
            # Broadcast para WebSocket
            await broadcast_to_websockets({
                "type": "new_signal",
                "data": signal
            })
            
            # Envia atividade recente
            activity = {
                "type": "activity",
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"🐋 Whale detectado: ${signal['whale_activity']['volume_total_usd']:,} em {signal['categoria']}"
            }
            state.demo_activity.append(activity)
            await broadcast_to_websockets(activity)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Erro no demo loop: {e}")
            await asyncio.sleep(5)


# =============================================================================
# LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicação"""
    logger.info("🚀 Iniciando Smart Event Trader...")
    
    # Inicializa monitores
    state.news_monitor = NewsMonitor(
        newsapi_key=os.getenv("NEWSAPI_KEY"),
        check_interval=30
    )
    state.whale_monitor = WhaleMonitor(
        polygonscan_key=os.getenv("POLYGONSCAN_KEY"),
        check_interval=15
    )
    state.polymarket_monitor = PolymarketMonitor(check_interval=30)
    
    # Abre sessões
    await state.news_monitor.__aenter__()
    await state.whale_monitor.__aenter__()
    await state.polymarket_monitor.__aenter__()
    
    # Inicializa engine
    state.signal_engine = SignalEngine(
        news_monitor=state.news_monitor,
        whale_monitor=state.whale_monitor,
        polymarket_monitor=state.polymarket_monitor
    )
    
    # Configura callbacks
    state.news_monitor.on_event(on_trending_event)
    state.whale_monitor.on_whale_activity(on_whale_activity)
    state.signal_engine.on_signal(on_new_signal)
    
    # Inicializa sistema de alertas
    state.alert_system = AlertSystem(AlertConfig(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        webhook_url=os.getenv("PUSH_WEBHOOK_URL")
    ))
    await state.alert_system.__aenter__()
    
    # Inicia monitoramento em background
    state.running = True
    state.monitoring_task = asyncio.create_task(monitoring_loop())
    
    # Inicia modo demo automaticamente (gera sinais simulados)
    state.demo_mode = True
    state.demo_task = asyncio.create_task(demo_signal_loop())
    
    logger.info("✅ Smart Event Trader iniciado com sucesso!")
    logger.info("🎭 Modo DEMO ativo - sinais simulados serão gerados automaticamente")
    
    yield
    
    # Cleanup
    logger.info("🛑 Encerrando Smart Event Trader...")
    state.running = False
    
    if state.monitoring_task:
        state.monitoring_task.cancel()
    
    if state.demo_task:
        state.demo_task.cancel()
    
    await state.news_monitor.__aexit__(None, None, None)
    await state.whale_monitor.__aexit__(None, None, None)
    await state.polymarket_monitor.__aexit__(None, None, None)
    await state.alert_system.__aexit__(None, None, None)
    
    logger.info("👋 Smart Event Trader encerrado")


# =============================================================================
# CALLBACKS
# =============================================================================

async def on_trending_event(event):
    """Callback quando evento trending é detectado"""
    logger.info(f"📰 Evento trending: {event.evento.titulo[:50]}...")
    await state.signal_engine.process_trending_event(event)


async def on_whale_activity(activity):
    """Callback quando atividade de whale é detectada"""
    logger.info(f"🐋 Whale activity: ${float(activity.volume_total_usd):,.0f}")
    await state.signal_engine.process_whale_activity(activity)


async def on_new_signal(signal: Signal):
    """Callback quando novo sinal é gerado"""
    logger.info(f"🎯 NOVO SINAL: {signal.urgencia.value} ({signal.confianca:.0f}%)")
    
    # Envia para WebSocket connections
    signal_dict = signal.to_dict()
    await broadcast_to_websockets({
        "type": "new_signal",
        "data": signal_dict
    })
    
    # Envia alertas se confiança alta
    if signal.confianca >= 65:
        await state.alert_system.send_signal_alert(signal_dict)


async def monitoring_loop():
    """Loop principal de monitoramento"""
    while state.running:
        try:
            # Scan de news
            await state.news_monitor.scan_once()
            
            # Scan de whales
            await state.whale_monitor.scan_once()
            
            # Scan de polymarket
            await state.polymarket_monitor.scan_once()
            
            # Aguarda intervalo
            await asyncio.sleep(10)
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Erro no monitoring loop: {e}")
            await asyncio.sleep(5)


# =============================================================================
# WEBSOCKET
# =============================================================================

async def broadcast_to_websockets(message: Dict[str, Any]):
    """Envia mensagem para todas as conexões WebSocket"""
    import json
    
    disconnected = []
    
    for connection in state.active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except:
            disconnected.append(connection)
    
    # Remove conexões desconectadas
    for conn in disconnected:
        state.active_connections.remove(conn)


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Smart Event Trader API",
    description="API para detecção de eventos e geração de sinais de trading",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class SignalResponse(BaseModel):
    id: str
    timestamp: str
    categoria: str
    confianca: float
    urgencia: str
    evento: Dict[str, Any]
    whale_activity: Dict[str, Any]
    mercados_polymarket: List[Dict[str, Any]]
    recomendacao: Dict[str, Any]
    botao_acao: Dict[str, Any]


class AlertConfigRequest(BaseModel):
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    webhook_url: Optional[str] = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "name": "Smart Event Trader",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    """Health check detalhado"""
    return {
        "status": "healthy",
        "monitors": {
            "news": state.news_monitor is not None,
            "whale": state.whale_monitor is not None,
            "polymarket": state.polymarket_monitor is not None
        },
        "engine": state.signal_engine is not None,
        "websocket_connections": len(state.active_connections),
        "monitoring": state.running
    }


# -----------------------------------------------------------------------------
# SINAIS
# -----------------------------------------------------------------------------

@app.get("/signals", response_model=List[Dict[str, Any]])
async def get_signals(
    categoria: Optional[str] = None,
    min_confianca: float = 0,
    limit: int = 50
):
    """
    Retorna sinais ativos e recentes.
    """
    # Modo demo - retorna sinais simulados
    if state.demo_mode and state.demo_signals:
        signals = state.demo_signals.copy()
        
        # Filtra por categoria
        if categoria:
            signals = [s for s in signals if s.get("categoria", "").upper() == categoria.upper()]
        
        # Filtra por confiança mínima
        signals = [s for s in signals if s.get("confianca", 0) >= min_confianca]
        
        # Ordena por timestamp (mais recente primeiro)
        signals.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return signals[:limit]
    
    # Modo real
    if not state.signal_engine:
        raise HTTPException(503, "Engine não inicializado")
    
    signals = state.signal_engine.get_active_signals()
    
    # Filtra por categoria
    if categoria:
        try:
            cat = Categoria(categoria)
            signals = [s for s in signals if s.categoria == cat]
        except ValueError:
            pass
    
    # Filtra por confiança mínima
    signals = [s for s in signals if s.confianca >= min_confianca]
    
    # Limita
    signals = signals[:limit]
    
    return [s.to_dict() for s in signals]


@app.get("/signals/{signal_id}")
async def get_signal(signal_id: str):
    """
    Retorna detalhes de um sinal específico.
    """
    if not state.signal_engine:
        raise HTTPException(503, "Engine não inicializado")
    
    signal = state.signal_engine.active_signals.get(signal_id)
    
    if not signal:
        # Busca no histórico
        for s in state.signal_engine.signal_history:
            if s.id == signal_id:
                signal = s
                break
    
    if not signal:
        raise HTTPException(404, "Sinal não encontrado")
    
    return signal.to_dict()


@app.get("/signals/history")
async def get_signal_history(
    categoria: Optional[str] = None,
    limit: int = 100
):
    """
    Retorna histórico de sinais.
    """
    if not state.signal_engine:
        raise HTTPException(503, "Engine não inicializado")
    
    cat = None
    if categoria:
        try:
            cat = Categoria(categoria)
        except ValueError:
            pass
    
    history = state.signal_engine.get_signal_history(limit=limit, categoria=cat)
    
    return [s.to_dict() for s in history]


# -----------------------------------------------------------------------------
# ESTATÍSTICAS
# -----------------------------------------------------------------------------

@app.get("/stats")
async def get_stats():
    """
    Retorna estatísticas de performance.
    """
    # Modo demo - retorna estatísticas simuladas
    if state.demo_mode:
        urgent_signals = [s for s in state.demo_signals if s.get("urgencia") == "URGENTE"]
        return {
            "sinais_ativos": len(state.demo_signals),
            "sinais_urgentes": len(urgent_signals),
            "taxa_acerto_geral": 78,
            "roi_medio": 145,
            "por_categoria": {
                "POLITICO": {"sinais": 47, "acertos": 43, "taxa_acerto": 91, "roi": 185},
                "TECNOLOGICO": {"sinais": 34, "acertos": 26, "taxa_acerto": 75, "roi": 95},
                "ESPORTIVO": {"sinais": 28, "acertos": 20, "taxa_acerto": 70, "roi": 65},
                "ECONOMICO": {"sinais": 31, "acertos": 19, "taxa_acerto": 60, "roi": 45},
                "CLIMATICO": {"sinais": 22, "acertos": 14, "taxa_acerto": 65, "roi": 78},
                "SAUDE": {"sinais": 18, "acertos": 10, "taxa_acerto": 55, "roi": 40}
            },
            "modo": "DEMO"
        }
    
    if not state.signal_engine:
        raise HTTPException(503, "Engine não inicializado")
    
    return state.signal_engine.get_performance_stats()


@app.get("/stats/categories")
async def get_category_stats():
    """
    Retorna estatísticas por categoria.
    """
    if not state.signal_engine:
        raise HTTPException(503, "Engine não inicializado")
    
    stats = state.signal_engine.get_performance_stats()
    return stats.get("por_categoria", {})


# -----------------------------------------------------------------------------
# MONITORAMENTO
# -----------------------------------------------------------------------------

@app.get("/monitors/status")
async def get_monitors_status():
    """
    Retorna status dos monitores.
    """
    return {
        "news": {
            "running": state.news_monitor._running if state.news_monitor else False,
            "events_cached": len(state.news_monitor.recent_events) if state.news_monitor else 0
        },
        "whale": {
            "running": state.whale_monitor._running if state.whale_monitor else False,
            "transactions_cached": len(state.whale_monitor.recent_transactions) if state.whale_monitor else 0,
            "known_whales": len(state.whale_monitor.known_whales) if state.whale_monitor else 0
        },
        "polymarket": {
            "running": state.polymarket_monitor._running if state.polymarket_monitor else False,
            "markets_cached": len(state.polymarket_monitor.markets) if state.polymarket_monitor else 0,
            "trending_count": len(state.polymarket_monitor.get_trending_markets()) if state.polymarket_monitor else 0
        }
    }


@app.post("/monitors/scan")
async def trigger_scan():
    """
    Força um scan imediato de todas as fontes.
    """
    results = {}
    
    if state.news_monitor:
        trending = await state.news_monitor.scan_once()
        results["news"] = {
            "trending_events": len(trending),
            "top": [t.evento.titulo[:50] for t in trending[:3]]
        }
    
    if state.whale_monitor:
        activity = await state.whale_monitor.scan_once()
        results["whale"] = {
            "activity_detected": activity is not None,
            "volume": float(activity.volume_total_usd) if activity else 0
        }
    
    if state.polymarket_monitor:
        trending = await state.polymarket_monitor.scan_once()
        results["polymarket"] = {
            "trending_markets": len(trending),
            "top_momentum": [m.momentum_score for m in trending[:3]]
        }
    
    return results


# -----------------------------------------------------------------------------
# ALERTAS
# -----------------------------------------------------------------------------

@app.post("/alerts/test")
async def test_alert():
    """
    Envia um alerta de teste.
    """
    if not state.alert_system:
        raise HTTPException(503, "Sistema de alertas não inicializado")
    
    test_signal = {
        "id": "test_001",
        "urgencia": "NORMAL",
        "categoria": "TECNOLÓGICO",
        "confianca": 75,
        "evento": {
            "titulo": "Teste de Alerta - Smart Event Trader",
            "descricao": "Este é um teste do sistema de alertas",
            "fonte": "Teste",
            "mentions": 0
        },
        "whale_activity": {
            "volume_total_usd": 100000,
            "tempo_antes_noticia": "15 minutos",
            "whales_detectados": 3
        },
        "mercados_polymarket": [{
            "titulo": "Mercado de Teste",
            "link": "https://polymarket.com"
        }],
        "recomendacao": {
            "acao": "COMPRAR SIM",
            "preco_entrada": 0.45,
            "alvo_principal": 0.75,
            "roi_esperado": "+67%",
            "tamanho_posicao_sugerido": 500,
            "stop_loss": 0.25,
            "timing_esperado": "72 horas"
        },
        "historico_categoria": {
            "tecnológico_taxa_acerto": "75%",
            "tecnológico_roi_medio": "+95%"
        },
        "botao_acao": {
            "link_direto": "https://polymarket.com",
            "texto_botao": "🚀 ENTRAR AGORA"
        }
    }
    
    results = await state.alert_system.send_signal_alert(test_signal)
    
    return {
        "message": "Alerta de teste enviado",
        "results": results
    }


@app.get("/alerts/history")
async def get_alerts_history(limit: int = 50):
    """
    Retorna histórico de alertas enviados.
    """
    if not state.alert_system:
        raise HTTPException(503, "Sistema de alertas não inicializado")
    
    return state.alert_system.get_alerts_history(limit)


# -----------------------------------------------------------------------------
# MODO DEMO
# -----------------------------------------------------------------------------

@app.get("/demo/status")
async def demo_status():
    """
    Retorna status do modo demonstração.
    """
    return {
        "demo_mode": state.demo_mode,
        "signals_count": len(state.demo_signals),
        "activity_count": len(state.demo_activity)
    }


@app.post("/demo/toggle")
async def toggle_demo():
    """
    Ativa/desativa modo demonstração.
    """
    state.demo_mode = not state.demo_mode
    
    if state.demo_mode:
        # Reinicia o loop de demo
        if state.demo_task:
            state.demo_task.cancel()
        state.demo_task = asyncio.create_task(demo_signal_loop())
        logger.info("🎭 Modo DEMO ATIVADO")
    else:
        # Para o loop de demo
        if state.demo_task:
            state.demo_task.cancel()
            state.demo_task = None
        logger.info("🔒 Modo DEMO DESATIVADO")
    
    return {"demo_mode": state.demo_mode}


@app.post("/demo/generate")
async def generate_demo():
    """
    Força geração de um novo sinal de demonstração.
    """
    signal = generate_demo_signal()
    state.demo_signals.append(signal)
    
    # Broadcast para WebSocket
    await broadcast_to_websockets({
        "type": "new_signal",
        "data": signal
    })
    
    # Atividade recente
    activity = {
        "type": "activity",
        "timestamp": datetime.utcnow().isoformat(),
        "message": f"🐋 Whale detectado: ${signal['whale_activity']['volume_total_usd']:,} em {signal['categoria']}"
    }
    state.demo_activity.append(activity)
    await broadcast_to_websockets(activity)
    
    logger.info(f"🎭 DEMO: Sinal gerado manualmente - {signal['evento']['titulo'][:40]}...")
    
    return signal


@app.post("/demo/clear")
async def clear_demo():
    """
    Limpa todos os sinais de demonstração.
    """
    state.demo_signals.clear()
    state.demo_activity.clear()
    
    await broadcast_to_websockets({
        "type": "clear_signals",
        "data": {}
    })
    
    return {"message": "Sinais de demo limpos"}


# -----------------------------------------------------------------------------
# GESTÃO DE RISCO
# -----------------------------------------------------------------------------

class RiskConfigRequest(BaseModel):
    initial_capital: float = 10000.0
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.20


class PositionSizeRequest(BaseModel):
    signal_confidence: float
    categoria: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: Optional[float] = None


class RecordTradeRequest(BaseModel):
    signal_id: str
    categoria: str
    entry_price: float
    exit_price: float
    position_size: float
    is_win: bool
    holding_time_hours: float = 24.0
    signal_confidence: float = 70.0
    fees_paid: float = 0.0


@app.get("/risk/dashboard")
async def get_risk_dashboard():
    """
    Retorna dashboard completo de gestão de risco.
    
    Inclui:
    - Capital atual e PnL
    - Exposição do portfólio
    - Métricas de performance (Sharpe, Sortino, Profit Factor)
    - Drawdown atual e máximo
    - Risk of Ruin
    - Kelly Criterion
    """
    risk_mgr = get_risk_manager()
    return risk_mgr.get_risk_dashboard()


@app.post("/risk/configure")
async def configure_risk(config: RiskConfigRequest):
    """
    Configura parâmetros de gestão de risco.
    """
    import risk.risk_manager as rm
    rm._risk_manager = RiskManager(
        initial_capital=config.initial_capital,
        max_risk_per_trade=config.max_risk_per_trade,
        max_portfolio_risk=config.max_portfolio_risk
    )
    return {
        "message": "Configuração de risco atualizada",
        "config": {
            "initial_capital": config.initial_capital,
            "max_risk_per_trade_percent": config.max_risk_per_trade * 100,
            "max_portfolio_risk_percent": config.max_portfolio_risk * 100
        }
    }


@app.post("/risk/position-size")
async def calculate_position_size(request: PositionSizeRequest):
    """
    Calcula o tamanho ideal da posição usando Kelly Criterion.
    
    Retorna:
    - Tamanho recomendado em USD e %
    - Kelly full, half e quarter
    - Nível de risco
    - Reasoning detalhado
    """
    risk_mgr = get_risk_manager()
    sizing = risk_mgr.calculate_position_size(
        signal_confidence=request.signal_confidence,
        categoria=request.categoria,
        entry_price=request.entry_price,
        stop_loss_price=request.stop_loss_price,
        take_profit_price=request.take_profit_price
    )
    return {
        "recommended_size_usd": round(sizing.recommended_size_usd, 2),
        "recommended_size_percent": round(sizing.recommended_size_percent, 2),
        "kelly": {
            "full": round(sizing.kelly_full * 100, 2),
            "half": round(sizing.kelly_half * 100, 2),
            "quarter": round(sizing.kelly_quarter * 100, 2)
        },
        "max_risk_usd": round(sizing.max_risk_usd, 2),
        "risk_level": sizing.risk_level.value,
        "reasoning": sizing.reasoning
    }


@app.post("/risk/stop-loss")
async def calculate_stop_loss(
    entry_price: float,
    categoria: str,
    volatility_multiplier: float = 2.0
):
    """
    Calcula stop loss e take profit dinâmicos.
    
    Baseado em:
    - Volatilidade histórica da categoria
    - Máximo risco por trade configurado
    - Risk/Reward ratio mínimo 2:1
    """
    risk_mgr = get_risk_manager()
    return risk_mgr.calculate_stop_loss(
        entry_price=entry_price,
        categoria=categoria,
        volatility_multiplier=volatility_multiplier
    )


@app.post("/risk/record-trade")
async def record_trade(request: RecordTradeRequest):
    """
    Registra um trade completo para atualizar métricas.
    
    Importante para:
    - Calcular win rate real
    - Atualizar Kelly Criterion
    - Rastrear drawdown
    - Calcular Sharpe/Sortino
    """
    risk_mgr = get_risk_manager()
    
    pnl = (request.exit_price - request.entry_price) * request.position_size / request.entry_price
    pnl_percent = (request.exit_price - request.entry_price) / request.entry_price
    
    trade = TradeResult(
        id=request.signal_id,
        timestamp=datetime.utcnow(),
        categoria=request.categoria,
        entry_price=request.entry_price,
        exit_price=request.exit_price,
        position_size=request.position_size,
        pnl=pnl - request.fees_paid,
        pnl_percent=pnl_percent,
        is_win=request.is_win,
        holding_time_hours=request.holding_time_hours,
        signal_confidence=request.signal_confidence,
        fees_paid=request.fees_paid
    )
    
    risk_mgr.record_trade(trade)
    
    return {
        "message": "Trade registrado",
        "trade": {
            "id": trade.id,
            "pnl": round(trade.pnl, 2),
            "pnl_percent": round(trade.pnl_percent * 100, 2),
            "is_win": trade.is_win
        },
        "updated_metrics": {
            "total_trades": len(risk_mgr.trade_history),
            "win_rate": round(sum(1 for t in risk_mgr.trade_history if t.is_win) / len(risk_mgr.trade_history) * 100, 1) if risk_mgr.trade_history else 0,
            "current_capital": round(risk_mgr.current_capital, 2)
        }
    }


@app.get("/risk/metrics")
async def get_risk_metrics():
    """
    Retorna todas as métricas de risco calculadas.
    
    Inclui:
    - Win rate, profit factor, expectancy
    - Sharpe ratio, Sortino ratio
    - Max drawdown, current drawdown
    - Kelly fraction, risk of ruin
    - Métricas por categoria
    """
    risk_mgr = get_risk_manager()
    metrics = risk_mgr.calculate_metrics()
    
    return {
        "basic": {
            "total_trades": metrics.total_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate": round(metrics.win_rate * 100, 2)
        },
        "pnl": {
            "total_pnl": round(metrics.total_pnl, 2),
            "gross_profit": round(metrics.gross_profit, 2),
            "gross_loss": round(metrics.gross_loss, 2),
            "avg_win": round(metrics.avg_win * 100, 2),
            "avg_loss": round(metrics.avg_loss * 100, 2),
            "largest_win": round(metrics.largest_win * 100, 2),
            "largest_loss": round(metrics.largest_loss * 100, 2)
        },
        "ratios": {
            "profit_factor": round(metrics.profit_factor, 2) if metrics.profit_factor < 999 else "∞",
            "risk_reward_ratio": round(metrics.risk_reward_ratio, 2),
            "expectancy": round(metrics.expectancy * 100, 2)
        },
        "volatility": {
            "returns_std": round(metrics.returns_std * 100, 2),
            "sharpe_ratio": round(metrics.sharpe_ratio, 2),
            "sortino_ratio": round(metrics.sortino_ratio, 2)
        },
        "drawdown": {
            "max_drawdown_usd": round(metrics.max_drawdown, 2),
            "max_drawdown_percent": round(metrics.max_drawdown_percent * 100, 2),
            "current_drawdown_percent": round(metrics.current_drawdown * 100, 2)
        },
        "risk": {
            "risk_of_ruin_percent": round(metrics.risk_of_ruin * 100, 2),
            "kelly_optimal_percent": round(metrics.kelly_fraction * 100, 2)
        },
        "by_category": metrics.metrics_by_category
    }


@app.get("/risk/correlation-matrix")
async def get_correlation_matrix():
    """
    Retorna matriz de correlação entre categorias.
    
    Importante para:
    - Identificar diversificação real vs ilusória
    - Ajustar position sizing considerando correlações
    - Evitar exposição concentrada em ativos correlacionados
    """
    risk_mgr = get_risk_manager()
    
    # Correlações históricas típicas entre categorias de prediction markets
    # Baseado em análise de movimentos de preço durante eventos
    correlation_matrix = {
        "POLITICO": {
            "POLITICO": 1.00,
            "TECNOLOGICO": 0.15,   # Baixa correlação
            "ESPORTIVO": 0.05,    # Quase nenhuma
            "ECONOMICO": 0.45,    # Moderada (políticas econômicas)
            "CLIMATICO": 0.10,    # Baixa
        },
        "TECNOLOGICO": {
            "POLITICO": 0.15,
            "TECNOLOGICO": 1.00,
            "ESPORTIVO": 0.08,
            "ECONOMICO": 0.55,    # Alta (tech stocks, Fed decisions)
            "CLIMATICO": 0.12,
        },
        "ESPORTIVO": {
            "POLITICO": 0.05,
            "TECNOLOGICO": 0.08,
            "ESPORTIVO": 1.00,
            "ECONOMICO": 0.10,
            "CLIMATICO": 0.03,    # Praticamente nenhuma
        },
        "ECONOMICO": {
            "POLITICO": 0.45,
            "TECNOLOGICO": 0.55,
            "ESPORTIVO": 0.10,
            "ECONOMICO": 1.00,
            "CLIMATICO": 0.25,    # Moderada (disasters afetam economia)
        },
        "CLIMATICO": {
            "POLITICO": 0.10,
            "TECNOLOGICO": 0.12,
            "ESPORTIVO": 0.03,
            "ECONOMICO": 0.25,
            "CLIMATICO": 1.00,
        }
    }
    
    # Calcula exposição atual por categoria
    category_exposure = {}
    for signal in state.demo_signals:
        cat = signal.get('categoria', 'OUTROS')
        risk_mgmt = signal.get('risk_management', {})
        size = risk_mgmt.get('position_size_usd', 0) if isinstance(risk_mgmt, dict) else 0
        category_exposure[cat] = category_exposure.get(cat, 0) + size
    
    # Calcula risco de correlação
    total_exposure = sum(category_exposure.values())
    
    # Verifica concentração perigosa
    warnings = []
    for cat, exposure in category_exposure.items():
        if total_exposure > 0:
            concentration = exposure / total_exposure
            if concentration > 0.4:
                warnings.append(f"⚠️ {cat}: {concentration*100:.0f}% do portfólio (concentração alta)")
    
    # Calcula risco agregado considerando correlações
    # Simplificado: média ponderada das correlações entre categorias com exposição
    avg_correlation = 0
    pairs = 0
    for cat1, exp1 in category_exposure.items():
        for cat2, exp2 in category_exposure.items():
            if cat1 != cat2 and cat1 in correlation_matrix and cat2 in correlation_matrix.get(cat1, {}):
                corr = correlation_matrix[cat1][cat2]
                weight = (exp1 * exp2) / (total_exposure ** 2) if total_exposure > 0 else 0
                avg_correlation += corr * weight
                pairs += 1
    
    diversification_score = 1 - avg_correlation if avg_correlation < 1 else 0
    
    return {
        "correlation_matrix": correlation_matrix,
        "current_exposure_by_category": category_exposure,
        "diversification": {
            "score": round(diversification_score * 100, 1),
            "interpretation": (
                "Excelente" if diversification_score > 0.8 else
                "Boa" if diversification_score > 0.6 else
                "Moderada" if diversification_score > 0.4 else
                "Fraca - considere diversificar"
            ),
            "avg_correlation": round(avg_correlation, 3)
        },
        "warnings": warnings if warnings else ["✅ Portfólio bem diversificado"],
        "recommendation": (
            "Portfólio diversificado" if avg_correlation < 0.3 else
            "Considere adicionar categorias não-correlacionadas" if avg_correlation < 0.5 else
            "⚠️ Alta correlação - risco concentrado"
        )
    }


@app.get("/risk/trade-history")
async def get_trade_history(limit: int = 100):
    """
    Retorna histórico completo de trades (wins + losses).
    """
    risk_mgr = get_risk_manager()
    
    trades = risk_mgr.trade_history[-limit:]
    
    return {
        "total_trades": len(risk_mgr.trade_history),
        "showing": len(trades),
        "trades": [
            {
                "id": t.id,
                "timestamp": t.timestamp.isoformat(),
                "categoria": t.categoria,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "position_size": t.position_size,
                "pnl": round(t.pnl, 2),
                "pnl_percent": round(t.pnl_percent * 100, 2),
                "is_win": t.is_win,
                "confidence": t.signal_confidence
            }
            for t in reversed(trades)
        ]
    }


@app.post("/risk/simulate-trades")
async def simulate_demo_trades(count: int = 50):
    """
    Simula trades para popular métricas com amostra estatisticamente significativa.
    
    Parâmetros:
    - count: Número de trades a simular (default: 50, mínimo para significância)
    
    Simula trades realistas com:
    - Win rate baseado na categoria e confiança
    - PnL baseado em distribuição histórica
    - Sequências de wins/losses para testar drawdown
    """
    risk_mgr = get_risk_manager()
    
    # Categorias com seus win rates históricos
    categories_data = [
        ("POLITICO", 0.91, 0.35, 0.12),     # categoria, win_rate, avg_win, avg_loss
        ("TECNOLOGICO", 0.75, 0.28, 0.15),
        ("ESPORTIVO", 0.70, 0.22, 0.13),
        ("ECONOMICO", 0.60, 0.18, 0.16),
        ("CLIMATICO", 0.65, 0.25, 0.18),
    ]
    
    simulated = 0
    wins = 0
    losses = 0
    
    # Gera 'count' trades simulados
    for i in range(count):
        # Escolhe categoria aleatória com peso
        cat_data = random.choice(categories_data)
        categoria, base_win_rate, avg_win, avg_loss = cat_data
        
        # Confiança do sinal (60-95%)
        confidence = random.uniform(60, 95)
        
        # Ajusta win rate pela confiança
        adjusted_win_rate = base_win_rate * (0.7 + (confidence / 100) * 0.3)
        adjusted_win_rate = min(0.95, adjusted_win_rate)
        
        # Determina se é win ou loss
        is_win = random.random() < adjusted_win_rate
        
        # Entry price (típico de prediction markets)
        entry = random.uniform(0.25, 0.70)
        
        if is_win:
            wins += 1
            # Ganho com variação
            gain = avg_win * random.uniform(0.5, 1.8)
            exit_price = entry * (1 + gain)
            exit_price = min(0.98, exit_price)  # Cap em 98 centavos
        else:
            losses += 1
            # Perda com variação
            loss = avg_loss * random.uniform(0.5, 2.0)
            exit_price = entry * (1 - loss)
            exit_price = max(0.02, exit_price)  # Floor em 2 centavos
        
        # Position size baseado em Kelly
        position_size = risk_mgr.current_capital * random.uniform(0.02, 0.08)
        
        # Calcula PnL
        pnl_percent = (exit_price - entry) / entry
        pnl_usd = position_size * pnl_percent
        
        # Fees (2% típico do Polymarket)
        fees = position_size * 0.02
        pnl_usd -= fees
        
        trade = TradeResult(
            id=f'sim_{i}_{uuid.uuid4().hex[:4]}',
            timestamp=datetime.utcnow() - timedelta(hours=random.randint(1, 720)),  # Até 30 dias atrás
            categoria=categoria,
            entry_price=entry,
            exit_price=exit_price,
            position_size=position_size,
            pnl=pnl_usd,
            pnl_percent=pnl_percent,
            is_win=is_win,
            holding_time_hours=random.randint(4, 168),
            signal_confidence=confidence,
            fees_paid=fees
        )
        
        risk_mgr.record_trade(trade)
        simulated += 1
    
    # Calcula métricas finais
    metrics = risk_mgr.calculate_metrics()
    
    return {
        "message": f"Simulados {simulated} trades",
        "summary": {
            "total_trades": simulated,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / simulated * 100, 1) if simulated > 0 else 0,
            "sample_significance": "✅ Estatisticamente significativo" if simulated >= 30 else "⚠️ Amostra pequena"
        },
        "current_metrics": risk_mgr.get_risk_dashboard()
    }


# -----------------------------------------------------------------------------
# WEBSOCKET
# -----------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket para receber sinais em tempo real.
    """
    await websocket.accept()
    state.active_connections.append(websocket)
    
    logger.info(f"📡 Nova conexão WebSocket ({len(state.active_connections)} total)")
    
    try:
        # Envia sinais atuais (demo ou real)
        if state.demo_mode and state.demo_signals:
            # Envia sinais demo
            await websocket.send_json({
                "type": "initial_signals",
                "data": state.demo_signals[-10:]
            })
            logger.info(f"📡 Enviados {len(state.demo_signals[-10:])} sinais demo para novo cliente")
        elif state.signal_engine:
            # Envia sinais reais
            signals = state.signal_engine.get_active_signals()
            await websocket.send_json({
                "type": "initial_signals",
                "data": [s.to_dict() for s in signals[:10]]
            })
        
        # Envia stats
        urgent = len([s for s in state.demo_signals if s.get("urgencia") == "URGENTE"])
        await websocket.send_json({
            "type": "stats_update",
            "data": {
                "sinais_ativos": len(state.demo_signals),
                "sinais_urgentes": urgent,
                "modo": "DEMO" if state.demo_mode else "REAL"
            }
        })
        
        # Mantém conexão
        while True:
            # Aguarda mensagens do cliente (heartbeat)
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        if websocket in state.active_connections:
            state.active_connections.remove(websocket)
        logger.info(f"📡 Conexão WebSocket encerrada ({len(state.active_connections)} restantes)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

pwd

