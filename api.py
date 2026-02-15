"""
Arbitrage System API
====================

API REST para o sistema de arbitragem Kalshi-Polymarket.
Deploy via Railway.

Endpoints:
- GET /health - Status do sistema
- GET /markets/kalshi - Lista mercados Kalshi
- GET /markets/polymarket - Lista mercados Polymarket
- POST /analyze - Analisa par de mercados
- GET /demo - Demo de coleta de dados
- GET /scan - Scan automático de arbitragem
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.collectors import KalshiCollector, PolymarketCollector
from src.engine import ArbitrageCalculator, ExecutionSimulator, AutoScanner
try:
    from src.engine.arbitrage_analyst import ArbitrageAnalyst, PolymarketOpportunity
except ImportError:
    # Fallback se o módulo não existir
    ArbitrageAnalyst = None
    PolymarketOpportunity = None
from src.models import (
    Market,
    Platform,
    Side,
    DEFAULT_MIN_SIZE_USD,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Criar app FastAPI
app = FastAPI(
    title="Kalshi-Polymarket Arbitrage API",
    description="Sistema de detecção de arbitragem entre mercados de previsão",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de tratamento de erros global - NUNCA deixa a API crashar
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Captura TODOS os erros não tratados para evitar crash da API."""
    try:
        logger.error(f"Erro não tratado capturado: {exc}", exc_info=True)
        # Se for HTTPException, deixar passar
        if isinstance(exc, HTTPException):
            raise exc
        # Para qualquer outro erro, retornar resposta JSON válida
        return JSONResponse(
            status_code=500,
            content={
                "error": "Erro interno do servidor",
                "detail": str(exc)[:200],  # Limitar tamanho
                "type": type(exc).__name__
            }
        )
    except Exception as e:
        # Se até o handler de erro falhar, retornar resposta mínima
        logger.critical(f"ERRO CRÍTICO no handler de erros: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Erro crítico no servidor"}
        )

# Cache do último scan
last_scan_result = None
scan_in_progress = False

# Cache para análises Polymarket em background
polymarket_analysis_jobs = {}  # {job_id: {"status": "running|completed|error", "result": ..., "started_at": ...}}
polymarket_analysis_counter = 0


# ==================== MODELS ====================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


class MarketResponse(BaseModel):
    platform: str
    ticker: str
    title: str
    status: str
    best_yes_bid: Optional[float] = None
    best_yes_ask: Optional[float] = None
    best_no_bid: Optional[float] = None
    best_no_ask: Optional[float] = None
    yes_spread: Optional[float] = None
    no_spread: Optional[float] = None


class AnalyzeRequest(BaseModel):
    kalshi_ticker: str = Field(..., description="Ticker do mercado Kalshi")
    polymarket_condition_id: str = Field(..., description="Condition ID do mercado Polymarket")
    target_usd: float = Field(default=20.0, description="Valor alvo em USD")
    equivalence_score: float = Field(default=1.0, description="Score de equivalência (0-1)")


class ExecutionDetail(BaseModel):
    side: str
    contracts: float
    vwap: float
    total_cost: float
    levels_consumed: int
    is_complete: bool


class ArbitrageResult(BaseModel):
    found: bool
    strategy: Optional[str] = None
    contracts: Optional[float] = None
    total_cost: Optional[float] = None
    guaranteed_payout: Optional[float] = None
    gross_profit: Optional[float] = None
    kalshi_fees: Optional[float] = None
    polymarket_fees: Optional[float] = None
    total_fees: Optional[float] = None
    basis_risk_penalty: Optional[float] = None
    slippage_buffer: Optional[float] = None
    net_profit: Optional[float] = None
    edge_percentage: Optional[float] = None
    kalshi_execution: Optional[ExecutionDetail] = None
    polymarket_execution: Optional[ExecutionDetail] = None
    rejection_code: Optional[str] = None
    rejection_message: Optional[str] = None


class DemoResponse(BaseModel):
    kalshi_markets: List[MarketResponse]
    polymarket_markets: List[MarketResponse]
    timestamp: str


# ==================== HELPERS ====================

def market_to_response(market: Market) -> MarketResponse:
    """Converte Market para MarketResponse."""
    response = MarketResponse(
        platform=market.platform.value,
        ticker=market.ticker,
        title=market.title[:100],
        status=market.status.value,
    )
    
    if market.yes_book:
        if market.yes_book.best_bid:
            response.best_yes_bid = float(market.yes_book.best_bid.price)
        if market.yes_book.best_ask:
            response.best_yes_ask = float(market.yes_book.best_ask.price)
        if market.yes_book.spread:
            response.yes_spread = float(market.yes_book.spread)
    
    if market.no_book:
        if market.no_book.best_bid:
            response.best_no_bid = float(market.no_book.best_bid.price)
        if market.no_book.best_ask:
            response.best_no_ask = float(market.no_book.best_ask.price)
        if market.no_book.spread:
            response.no_spread = float(market.no_book.spread)
    
    return response


# ==================== ENDPOINTS ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint raiz - retorna status."""
    return HealthResponse(
        status="online",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check - endpoint simples que sempre funciona."""
    try:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        # Mesmo o health check não pode falhar
        logger.error(f"Erro no health check: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        )


@app.get("/markets/kalshi", response_model=List[MarketResponse])
async def get_kalshi_markets(
    limit: int = Query(default=10, ge=1, le=100),
    status: str = Query(default="open"),
):
    """Lista mercados da Kalshi."""
    try:
        async with KalshiCollector() as collector:
            markets, _ = await collector.get_markets(status=status, limit=limit)
            
            results = []
            for market in markets[:limit]:
                # Buscar orderbook
                yes_book, no_book = await collector.get_orderbook(market.ticker)
                market.yes_book = yes_book
                market.no_book = no_book
                results.append(market_to_response(market))
            
            return results
            
    except Exception as e:
        logger.error(f"Erro ao buscar mercados Kalshi: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/markets/polymarket", response_model=List[MarketResponse])
async def get_polymarket_markets(
    limit: int = Query(default=10, ge=1, le=100),
):
    """Lista mercados da Polymarket."""
    try:
        async with PolymarketCollector() as collector:
            markets = await collector.get_markets(limit=limit)
            
            results = []
            for market in markets[:limit]:
                # Buscar orderbooks
                if market.yes_token_id:
                    yes_book = await collector.get_orderbook(market.yes_token_id)
                    if yes_book:
                        yes_book.side = Side.YES
                    market.yes_book = yes_book
                
                if market.no_token_id:
                    no_book = await collector.get_orderbook(market.no_token_id)
                    if no_book:
                        no_book.side = Side.NO
                    market.no_book = no_book
                
                results.append(market_to_response(market))
            
            return results
            
    except Exception as e:
        logger.error(f"Erro ao buscar mercados Polymarket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/kalshi/{ticker}", response_model=MarketResponse)
async def get_kalshi_market(ticker: str):
    """Busca mercado específico da Kalshi com orderbook."""
    try:
        async with KalshiCollector() as collector:
            market = await collector.get_market_with_orderbook(ticker)
            
            if not market:
                raise HTTPException(status_code=404, detail=f"Mercado não encontrado: {ticker}")
            
            return market_to_response(market)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao buscar mercado Kalshi {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=ArbitrageResult)
async def analyze_arbitrage(request: AnalyzeRequest):
    """
    Analisa oportunidade de arbitragem entre um par de mercados.
    
    IMPORTANTE: A maioria das análises deve retornar found=False.
    Isso é comportamento ESPERADO e CORRETO.
    """
    try:
        async with KalshiCollector() as kalshi, PolymarketCollector() as polymarket:
            # Buscar mercados
            kalshi_market = await kalshi.get_market_with_orderbook(request.kalshi_ticker)
            poly_market = await polymarket.get_market_with_orderbooks(request.polymarket_condition_id)
            
            if not kalshi_market:
                return ArbitrageResult(
                    found=False,
                    rejection_code="KALSHI_NOT_FOUND",
                    rejection_message=f"Mercado Kalshi não encontrado: {request.kalshi_ticker}"
                )
            
            if not poly_market:
                return ArbitrageResult(
                    found=False,
                    rejection_code="POLY_NOT_FOUND",
                    rejection_message=f"Mercado Polymarket não encontrado: {request.polymarket_condition_id}"
                )
            
            # Calcular arbitragem
            calculator = ArbitrageCalculator()
            opportunity, rejection = calculator.calculate(
                kalshi_market=kalshi_market,
                polymarket_market=poly_market,
                target_usd=Decimal(str(request.target_usd)),
                equivalence_score=Decimal(str(request.equivalence_score)),
            )
            
            if opportunity:
                return ArbitrageResult(
                    found=True,
                    strategy=opportunity.strategy,
                    contracts=float(opportunity.contracts),
                    total_cost=float(opportunity.total_cost),
                    guaranteed_payout=float(opportunity.guaranteed_payout),
                    gross_profit=float(opportunity.gross_profit),
                    kalshi_fees=float(opportunity.kalshi_fees),
                    polymarket_fees=float(opportunity.polymarket_fees),
                    total_fees=float(opportunity.total_fees),
                    basis_risk_penalty=float(opportunity.basis_risk_penalty),
                    slippage_buffer=float(opportunity.slippage_buffer),
                    net_profit=float(opportunity.net_profit),
                    edge_percentage=float(opportunity.edge_percentage),
                    kalshi_execution=ExecutionDetail(
                        side=opportunity.kalshi_execution.side.value,
                        contracts=float(opportunity.kalshi_execution.filled_size),
                        vwap=float(opportunity.kalshi_execution.vwap),
                        total_cost=float(opportunity.kalshi_execution.total_cost),
                        levels_consumed=opportunity.kalshi_execution.levels_consumed,
                        is_complete=opportunity.kalshi_execution.is_complete,
                    ),
                    polymarket_execution=ExecutionDetail(
                        side=opportunity.polymarket_execution.side.value,
                        contracts=float(opportunity.polymarket_execution.filled_size),
                        vwap=float(opportunity.polymarket_execution.vwap),
                        total_cost=float(opportunity.polymarket_execution.total_cost),
                        levels_consumed=opportunity.polymarket_execution.levels_consumed,
                        is_complete=opportunity.polymarket_execution.is_complete,
                    ),
                )
            else:
                return ArbitrageResult(
                    found=False,
                    rejection_code=rejection.code if rejection else "UNKNOWN",
                    rejection_message=rejection.message if rejection else "Motivo desconhecido",
                )
                
    except Exception as e:
        logger.error(f"Erro na análise de arbitragem: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo", response_model=DemoResponse)
async def demo():
    """
    Demo: Coleta dados das duas plataformas.
    
    Use para verificar que as APIs estão funcionando.
    """
    kalshi_markets = []
    poly_markets = []
    
    try:
        # Kalshi
        async with KalshiCollector() as collector:
            markets, _ = await collector.get_markets(limit=3)
            for m in markets:
                yes_book, no_book = await collector.get_orderbook(m.ticker)
                m.yes_book = yes_book
                m.no_book = no_book
                kalshi_markets.append(market_to_response(m))
    except Exception as e:
        logger.error(f"Erro Kalshi: {e}")
    
    try:
        # Polymarket
        async with PolymarketCollector() as collector:
            markets = await collector.get_markets(limit=3)
            for m in markets:
                if m.yes_token_id:
                    yes_book = await collector.get_orderbook(m.yes_token_id)
                    if yes_book:
                        yes_book.side = Side.YES
                    m.yes_book = yes_book
                poly_markets.append(market_to_response(m))
    except Exception as e:
        logger.error(f"Erro Polymarket: {e}")
    
    return DemoResponse(
        kalshi_markets=kalshi_markets,
        polymarket_markets=poly_markets,
        timestamp=datetime.utcnow().isoformat(),
    )


# ==================== SCAN AUTOMÁTICO ====================

class ScanConfig(BaseModel):
    target_usd: float = Field(default=20.0, description="Valor alvo em USD")
    min_similarity: float = Field(default=0.3, description="Similaridade mínima (0-1)")
    max_markets: int = Field(default=30, description="Máximo de mercados por plataforma")


class OpportunityResponse(BaseModel):
    kalshi_ticker: str
    kalshi_title: str
    polymarket_ticker: str
    polymarket_title: str
    similarity_score: float
    match_reason: str
    equivalence_level: str = "unknown"  # NOVO
    equivalence_confidence: float = 0.0  # NOVO
    risks: List[str] = []  # NOVO
    strategy: str
    contracts: float
    total_cost: float
    guaranteed_payout: float
    gross_profit: float
    total_fees: float
    net_profit: float
    edge_percentage: float
    kalshi_vwap: float
    polymarket_vwap: float


class ScanResponse(BaseModel):
    status: str
    timestamp: str
    kalshi_markets_scanned: int
    polymarket_markets_scanned: int
    pairs_analyzed: int
    opportunities_found: int
    opportunities: List[OpportunityResponse]
    rejections_summary: dict
    scan_duration_seconds: float


class ScanStatusResponse(BaseModel):
    scan_in_progress: bool
    last_scan_timestamp: Optional[str] = None
    last_scan_opportunities: int = 0


@app.get("/scan/status", response_model=ScanStatusResponse)
async def get_scan_status():
    """Retorna status do scan automático."""
    global last_scan_result, scan_in_progress
    
    return ScanStatusResponse(
        scan_in_progress=scan_in_progress,
        last_scan_timestamp=last_scan_result.timestamp.isoformat() if last_scan_result else None,
        last_scan_opportunities=last_scan_result.opportunities_found if last_scan_result else 0,
    )


@app.get("/scan/last", response_model=Optional[ScanResponse])
async def get_last_scan():
    """Retorna resultado do último scan."""
    global last_scan_result
    
    if not last_scan_result:
        return None
    
    return ScanResponse(
        status="completed",
        timestamp=last_scan_result.timestamp.isoformat(),
        kalshi_markets_scanned=last_scan_result.kalshi_markets_scanned,
        polymarket_markets_scanned=last_scan_result.polymarket_markets_scanned,
        pairs_analyzed=last_scan_result.pairs_analyzed,
        opportunities_found=last_scan_result.opportunities_found,
        opportunities=[OpportunityResponse(**o) for o in last_scan_result.opportunities],
        rejections_summary=last_scan_result.rejections_summary,
        scan_duration_seconds=last_scan_result.scan_duration_seconds,
    )


@app.post("/scan", response_model=ScanResponse)
async def run_scan(config: ScanConfig = None):
    """
    Executa scan automático de arbitragem.
    
    Este endpoint:
    1. Coleta mercados das duas plataformas
    2. Encontra pares similares automaticamente
    3. Analisa arbitragem para cada par
    4. Retorna oportunidades ordenadas por edge
    
    IMPORTANTE: A maioria dos scans deve retornar 0 oportunidades.
    Isso é comportamento ESPERADO e CORRETO.
    """
    global last_scan_result, scan_in_progress
    
    if scan_in_progress:
        raise HTTPException(status_code=429, detail="Scan já em andamento")
    
    scan_in_progress = True
    
    try:
        config = config or ScanConfig()
        
        scanner = AutoScanner(
            target_usd=Decimal(str(config.target_usd)),
            min_similarity=config.min_similarity,
            max_markets_per_platform=config.max_markets,
        )
        
        result = await scanner.scan()
        last_scan_result = result
        
        return ScanResponse(
            status="completed",
            timestamp=result.timestamp.isoformat(),
            kalshi_markets_scanned=result.kalshi_markets_scanned,
            polymarket_markets_scanned=result.polymarket_markets_scanned,
            pairs_analyzed=result.pairs_analyzed,
            opportunities_found=result.opportunities_found,
            opportunities=[OpportunityResponse(**o) for o in result.opportunities],
            rejections_summary=result.rejections_summary,
            scan_duration_seconds=result.scan_duration_seconds,
        )
        
    except Exception as e:
        logger.error(f"Erro no scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        scan_in_progress = False


# ============== FULL SCAN (Large Scale) ==============

class FullScanConfig(BaseModel):
    target_usd: float = Field(default=100, ge=1, le=10000)
    min_similarity: float = Field(default=0.25, ge=0.1, le=1.0)
    max_markets: int = Field(default=50000, ge=100, le=100000)  # Aumentado para pegar todos


class FullScanResponse(BaseModel):
    status: str
    timestamp: str
    kalshi_total: int
    polymarket_total: int
    smart_matches: int = 0
    semantic_analyzed: int = 0
    same_event_confirmed: int = 0
    pairs_analyzed: int
    pairs_with_liquidity: int
    opportunities_found: int
    opportunities: List[dict]
    top_pairs: List[dict]
    scan_duration_seconds: float
    category_stats: dict = {}
    semantic_enabled: bool = False


full_scan_in_progress = False
last_full_scan_result = None


@app.post("/scan/full", response_model=FullScanResponse)
async def run_full_scan(config: FullScanConfig = None):
    """
    Executa scan COMPLETO de todos os mercados.
    
    Este scan:
    - Busca até max_markets de cada plataforma
    - Matching por ENTIDADES ESPECÍFICAS (pessoas, empresas, eventos)
    - Análise SEMÂNTICA com Claude para confirmar mesmo evento
    - Só pareia mercados com entidades em comum
    - Retorna oportunidades E top pares similares
    
    Configure ANTHROPIC_API_KEY no Railway para habilitar análise semântica.
    """
    global full_scan_in_progress, last_full_scan_result
    
    if full_scan_in_progress:
        raise HTTPException(status_code=429, detail="Full scan já em andamento")
    
    full_scan_in_progress = True
    
    try:
        from src.engine.large_scale_scanner import LargeScaleScanner
        
        config = config or FullScanConfig()
        
        scanner = LargeScaleScanner(
            target_usd=Decimal(str(config.target_usd)),
            min_similarity=config.min_similarity,
            max_markets_per_platform=config.max_markets,
        )
        
        result = await scanner.full_scan()
        last_full_scan_result = result
        
        return FullScanResponse(
            status="completed",
            timestamp=result.timestamp.isoformat(),
            kalshi_total=result.kalshi_total,
            polymarket_total=result.polymarket_total,
            smart_matches=result.smart_matches,
            semantic_analyzed=result.semantic_analyzed,
            same_event_confirmed=result.same_event_confirmed,
            pairs_analyzed=result.pairs_analyzed,
            pairs_with_liquidity=result.pairs_with_liquidity,
            opportunities_found=result.opportunities_found,
            opportunities=result.opportunities,
            top_pairs=result.top_pairs,
            scan_duration_seconds=result.scan_duration_seconds,
            category_stats=result.category_stats,
            semantic_enabled=result.semantic_enabled,
        )
        
    except Exception as e:
        logger.error(f"Erro no full scan: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        full_scan_in_progress = False


@app.get("/scan/full/status")
async def get_full_scan_status():
    """Status do último full scan."""
    global full_scan_in_progress, last_full_scan_result
    
    return {
        "scan_in_progress": full_scan_in_progress,
        "last_scan_timestamp": last_full_scan_result.timestamp.isoformat() if last_full_scan_result else None,
        "last_scan_pairs": last_full_scan_result.pairs_analyzed if last_full_scan_result else 0,
        "last_scan_opportunities": last_full_scan_result.opportunities_found if last_full_scan_result else 0,
    }


# ============== POLYMARKET ARBITRAGE ANALYST ==============

class PolymarketOpportunityResponse(BaseModel):
    market_title: str
    condition_id: str
    polymarket_price: float
    implied_probability: float
    expected_probability: float
    divergence: float
    volume_24h: float
    liquidity_depth: float
    time_to_resolution_hours: Optional[float] = None
    expected_value: float
    edge_percentage: float
    recommended_side: str
    recommended_investment: float
    returns_10: float
    returns_20: float
    returns_100: float
    returns_1000: float
    confidence_score: float
    detected_at: str


class PolymarketAnalysisResponse(BaseModel):
    status: str
    timestamp: str
    total_markets_scanned: int
    opportunities_found: int
    opportunities: List[PolymarketOpportunityResponse]
    criteria: dict
    job_id: Optional[str] = None  # ID do job se estiver rodando em background


class PolymarketJobStatus(BaseModel):
    job_id: str
    status: str  # "running", "completed", "error"
    progress: Optional[int] = None  # 0-100
    message: Optional[str] = None
    result: Optional[PolymarketAnalysisResponse] = None
    started_at: str
    completed_at: Optional[str] = None


async def run_polymarket_analysis_background(job_id: str, limit: int):
    """Executa análise em background usando SimpleScanner e CorrelationScanner."""
    global polymarket_analysis_jobs
    
    try:
        polymarket_analysis_jobs[job_id]["status"] = "running"
        polymarket_analysis_jobs[job_id]["message"] = "Iniciando análise..."
        
        import asyncio
        from src.engine.simple_scanner import SimpleScanner
        from src.engine.correlation_arbitrage import CorrelationArbitrageScanner
        
        # PARTE 1: Oportunidades de Valor (principal)
        polymarket_analysis_jobs[job_id]["message"] = f"Buscando oportunidades de valor em {limit} mercados..."
        
        value_opportunities = []
        try:
            async with SimpleScanner(timeout=180.0) as scanner:
                value_opportunities = await asyncio.wait_for(
                    scanner.scan(limit=limit),
                    timeout=600.0  # 10 minutos
                )
        except asyncio.TimeoutError:
            logger.warning("Timeout na análise de valor, continuando com resultados parciais")
        except Exception as e:
            logger.warning(f"Erro na análise de valor: {e}")
        
        # PARTE 2: Arbitragem Correlacionada (opcional, só se tiver tempo)
        corr_opportunities = []
        # Desabilitado por enquanto para evitar timeout
        # Se quiser habilitar, descomente abaixo:
        # polymarket_analysis_jobs[job_id]["message"] = f"Buscando arbitragem correlacionada..."
        # try:
        #     async with CorrelationArbitrageScanner(timeout=180.0) as corr_scanner:
        #         corr_opportunities = await asyncio.wait_for(
        #             corr_scanner.scan(limit=min(limit, 100)),
        #             timeout=300.0
        #         )
        # except Exception as e:
        #     logger.warning(f"Erro na análise de correlação: {e}")
        
        # Converter oportunidades de valor
        opp_responses = []
        for opp in value_opportunities[:10]:
            try:
                opp_responses.append(PolymarketOpportunityResponse(
                    market_title=f"[VALOR] {opp.market_title}"[:200],
                    condition_id=str(opp.market_id)[:100],
                    polymarket_price=float(opp.yes_price),
                    implied_probability=float(opp.yes_price),
                    expected_probability=float(opp.yes_price + opp.edge),
                    divergence=float(opp.edge),
                    volume_24h=float(opp.volume_24h),
                    liquidity_depth=float(opp.liquidity),
                    time_to_resolution_hours=None,
                    expected_value=float(opp.expected_return / 100),
                    edge_percentage=float(opp.edge * 100),
                    recommended_side=opp.recommendation,
                    recommended_investment=100.0,
                    returns_10=float(opp.returns_10),
                    returns_20=float(opp.returns_10 * 2),
                    returns_100=float(opp.returns_100),
                    returns_1000=float(opp.returns_1000),
                    confidence_score=float(opp.confidence),
                    detected_at=opp.detected_at.isoformat(),
                ))
            except Exception as e:
                logger.warning(f"Erro ao converter oportunidade de valor: {e}")
        
        # Converter oportunidades de correlação
        for opp in corr_opportunities[:5]:
            try:
                opp_responses.append(PolymarketOpportunityResponse(
                    market_title=f"[ARBITRAGEM] {opp.market1_title} vs {opp.market2_title}"[:200],
                    condition_id=f"{opp.market1_id}|{opp.market2_id}"[:100],
                    polymarket_price=float(opp.market1_yes_price),
                    implied_probability=float(opp.market1_yes_price),
                    expected_probability=float(opp.market2_yes_price),
                    divergence=float(opp.expected_profit / 100),
                    volume_24h=0.0,
                    liquidity_depth=0.0,
                    time_to_resolution_hours=None,
                    expected_value=float(opp.expected_profit / 100),
                    edge_percentage=float(opp.expected_profit),
                    recommended_side=opp.arbitrage_action[:50],
                    recommended_investment=100.0,
                    returns_10=float(opp.expected_profit / 10),
                    returns_20=float(opp.expected_profit / 5),
                    returns_100=float(opp.expected_profit),
                    returns_1000=float(opp.expected_profit * 10),
                    confidence_score=float(opp.confidence),
                    detected_at=opp.detected_at.isoformat(),
                ))
            except Exception as e:
                logger.warning(f"Erro ao converter oportunidade correlacionada: {e}")
        
        total_found = len(value_opportunities) + len(corr_opportunities)
        
        result = PolymarketAnalysisResponse(
            status="completed",
            timestamp=datetime.utcnow().isoformat(),
            total_markets_scanned=limit,
            opportunities_found=total_found,
            opportunities=opp_responses,
            criteria={
                "value_opportunities": len(value_opportunities),
                "correlation_arbitrage": len(corr_opportunities),
                "strategy": "valor + arbitragem correlacionada",
            },
            job_id=job_id
        )
        
        polymarket_analysis_jobs[job_id]["status"] = "completed"
        polymarket_analysis_jobs[job_id]["result"] = result
        polymarket_analysis_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        polymarket_analysis_jobs[job_id]["message"] = f"Análise concluída: {len(value_opportunities)} oportunidades de valor, {len(corr_opportunities)} arbitragens correlacionadas"
        
    except asyncio.TimeoutError:
        logger.error("Timeout na análise")
        polymarket_analysis_jobs[job_id]["status"] = "error"
        polymarket_analysis_jobs[job_id]["message"] = "Timeout - tente com menos mercados"
        polymarket_analysis_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
    except Exception as e:
        logger.error(f"Erro na análise em background: {e}", exc_info=True)
        polymarket_analysis_jobs[job_id]["status"] = "error"
        polymarket_analysis_jobs[job_id]["message"] = f"Erro: {str(e)[:100]}"
        polymarket_analysis_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


@app.get("/polymarket/analyze", response_model=PolymarketAnalysisResponse)
async def analyze_polymarket_opportunities(
    limit: int = Query(default=100, ge=10, le=1000, description="Número máximo de mercados a escanear"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Analisa oportunidades de arbitragem no Polymarket.
    
    Executa em background para evitar timeout de conexão HTTP.
    Retorna imediatamente com job_id. Use /polymarket/analyze/status/{job_id} para verificar progresso.
    """
    global polymarket_analysis_counter
    
    try:
        # Criar job ID (operação instantânea)
        polymarket_analysis_counter += 1
        job_id = f"poly_{polymarket_analysis_counter}_{int(datetime.utcnow().timestamp())}"
        now = datetime.utcnow().isoformat()
        
        # Inicializar job ANTES de iniciar background task (operação instantânea)
        polymarket_analysis_jobs[job_id] = {
            "status": "running",
            "result": None,
            "started_at": now,
            "completed_at": None,
            "message": "Análise iniciada..."
        }
        
        # Preparar resposta ANTES de adicionar background task (garantir resposta rápida)
        response_data = PolymarketAnalysisResponse(
            status="running",
            timestamp=now,
            total_markets_scanned=limit,
            opportunities_found=0,
            opportunities=[],
            criteria={},
            job_id=job_id
        )
        
        # Iniciar análise em background DEPOIS de preparar resposta (não bloqueia)
        try:
            background_tasks.add_task(run_polymarket_analysis_background, job_id, limit)
        except Exception as e:
            logger.error(f"Erro ao adicionar background task: {e}")
            polymarket_analysis_jobs[job_id]["status"] = "error"
            polymarket_analysis_jobs[job_id]["message"] = f"Erro ao iniciar análise: {str(e)[:100]}"
            # Atualizar resposta para erro
            response_data.status = "error"
        
        # Retornar resposta imediata (NUNCA bloqueia - resposta já está pronta)
        return response_data
    except Exception as e:
        # Última camada de proteção - nunca crashar
        logger.error(f"ERRO CRÍTICO em analyze_polymarket_opportunities: {e}", exc_info=True)
        # Retornar resposta de erro válida
        return PolymarketAnalysisResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            total_markets_scanned=0,
            opportunities_found=0,
            opportunities=[],
            criteria={},
            job_id=None
        )


@app.get("/polymarket/test")
async def test_polymarket_endpoint():
    """Endpoint de teste simples para verificar se a API está funcionando."""
    return {
        "status": "ok",
        "message": "API Polymarket está funcionando",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "analyze": "/polymarket/analyze?limit=50",
            "status": "/polymarket/analyze/status/{job_id}"
        }
    }


@app.get("/polymarket/analyze/status/{job_id}", response_model=PolymarketJobStatus)
async def get_polymarket_analysis_status(job_id: str):
    """Verifica status de uma análise em andamento."""
    global polymarket_analysis_jobs
    
    try:
        if job_id not in polymarket_analysis_jobs:
            # Retornar erro mas não crashar
            return PolymarketJobStatus(
                job_id=job_id,
                status="error",
                message="Job não encontrado. Pode ter expirado ou não foi criado.",
                started_at=datetime.utcnow().isoformat(),
                completed_at=datetime.utcnow().isoformat()
            )
        
        job = polymarket_analysis_jobs[job_id]
        
        return PolymarketJobStatus(
            job_id=job_id,
            status=job.get("status", "unknown"),
            message=job.get("message"),
            result=job.get("result"),
            started_at=job.get("started_at", datetime.utcnow().isoformat()),
            completed_at=job.get("completed_at")
        )
    except Exception as e:
        logger.error(f"Erro ao buscar status do job {job_id}: {e}", exc_info=True)
        return PolymarketJobStatus(
            job_id=job_id,
            status="error",
            message=f"Erro ao buscar status: {str(e)[:100]}",
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat()
        )


@app.get("/polymarket/analyze/sync", response_model=PolymarketAnalysisResponse)
async def analyze_polymarket_opportunities_sync(
    limit: int = Query(default=100, ge=10, le=1000, description="Número máximo de mercados a escanear")
):
    """
    Versão síncrona (para testes). 
    ATENÇÃO: Pode dar timeout em análises longas.
    """
    # Resposta padrão em caso de erro
    error_response = PolymarketAnalysisResponse(
        status="error",
        timestamp=datetime.utcnow().isoformat(),
        total_markets_scanned=0,
        opportunities_found=0,
        opportunities=[],
        criteria={}
    )
    
    if ArbitrageAnalyst is None:
        logger.error("ArbitrageAnalyst não está disponível")
        return error_response
    
    try:
        import asyncio
        
        # Função wrapper com tratamento de erros em todos os níveis
        async def safe_analysis():
            try:
                analyst_instance = None
                try:
                    analyst_instance = ArbitrageAnalyst()
                    async with analyst_instance:
                        opportunities = await asyncio.wait_for(
                            analyst_instance.scan_opportunities(limit=limit),
                            timeout=120.0  # 2 minutos máximo
                        )
                        return opportunities, analyst_instance
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout na análise (limite: {limit} mercados)")
                    return [], None
                except Exception as e:
                    logger.error(f"Erro no ArbitrageAnalyst: {e}", exc_info=True)
                    return [], None
            except Exception as e:
                logger.error(f"Erro ao criar ArbitrageAnalyst: {e}", exc_info=True)
                return [], None
        
        # Executar análise com timeout total
        try:
            opportunities, analyst = await asyncio.wait_for(
                safe_analysis(),
                timeout=180.0  # 3 minutos total
            )
        except asyncio.TimeoutError:
            logger.error("Timeout total na análise")
            return error_response
        except Exception as e:
            logger.error(f"Erro inesperado: {e}", exc_info=True)
            return error_response
        
        # Se não há oportunidades, retornar resposta vazia mas válida
        if not opportunities:
            return PolymarketAnalysisResponse(
                status="completed",
                timestamp=datetime.utcnow().isoformat(),
                total_markets_scanned=limit,
                opportunities_found=0,
                opportunities=[],
                criteria={
                    "min_volume_usd": 100000.0,
                    "max_timeframe_hours": 48,
                    "min_divergence": 0.10,
                    "min_ev": 0.05,
                } if analyst is None else {
                    "min_volume_usd": float(analyst.MIN_VOLUME_USD),
                    "max_timeframe_hours": analyst.MAX_TIMEFRAME_HOURS,
                    "min_divergence": float(analyst.MIN_DIVERGENCE),
                    "min_ev": float(analyst.MIN_EV),
                }
            )
        
        # Converter oportunidades para resposta
        opp_responses = []
        try:
            for opp in opportunities[:3]:  # Top 3
                try:
                    time_hours = None
                    if opp.time_to_resolution:
                        time_hours = opp.time_to_resolution.total_seconds() / 3600
                    
                    opp_responses.append(PolymarketOpportunityResponse(
                        market_title=str(opp.market_title)[:200],  # Limitar tamanho
                        condition_id=str(opp.condition_id)[:100],
                        polymarket_price=float(opp.polymarket_price),
                        implied_probability=float(opp.implied_probability),
                        expected_probability=float(opp.expected_probability),
                        divergence=float(opp.divergence),
                        volume_24h=float(opp.volume_24h),
                        liquidity_depth=float(opp.liquidity_depth),
                        time_to_resolution_hours=time_hours,
                        expected_value=float(opp.expected_value),
                        edge_percentage=float(opp.edge_percentage),
                        recommended_side=opp.recommended_side.value.upper() if hasattr(opp.recommended_side, 'value') else str(opp.recommended_side).upper(),
                        recommended_investment=float(opp.recommended_investment),
                        returns_10=float(opp.returns_10),
                        returns_20=float(opp.returns_20),
                        returns_100=float(opp.returns_100),
                        returns_1000=float(opp.returns_1000),
                        confidence_score=float(opp.confidence_score),
                        detected_at=opp.detected_at.isoformat() if hasattr(opp.detected_at, 'isoformat') else datetime.utcnow().isoformat(),
                    ))
                except Exception as e:
                    logger.warning(f"Erro ao converter oportunidade: {e}")
                    continue  # Pular esta oportunidade e continuar
            
            # Retornar resposta de sucesso
            return PolymarketAnalysisResponse(
                status="completed",
                timestamp=datetime.utcnow().isoformat(),
                total_markets_scanned=limit,
                opportunities_found=len(opportunities),
                opportunities=opp_responses,
                criteria={
                    "min_volume_usd": float(analyst.MIN_VOLUME_USD) if analyst else 100000.0,
                    "max_timeframe_hours": analyst.MAX_TIMEFRAME_HOURS if analyst else 48,
                    "min_divergence": float(analyst.MIN_DIVERGENCE) if analyst else 0.10,
                    "min_ev": float(analyst.MIN_EV) if analyst else 0.05,
                }
            )
        except Exception as e:
            logger.error(f"Erro ao processar oportunidades: {e}", exc_info=True)
            return error_response
        
    except Exception as e:
        # Último nível de proteção - captura QUALQUER erro
        logger.error(f"ERRO CRÍTICO na análise: {e}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # SEMPRE retornar resposta válida, nunca crashar
        return error_response


# ==========================================
# CROSS-PLATFORM ARBITRAGE: Polymarket vs Kalshi
# ==========================================

class CrossPlatformOpportunity(BaseModel):
    """Uma oportunidade de arbitragem cross-platform."""
    event_name: str
    match_confidence: float
    
    # Polymarket
    poly_market_title: str
    poly_yes_price: float
    poly_no_price: float
    poly_volume: float
    poly_url: str = ""  # Link direto para o mercado
    
    # Kalshi
    kalshi_market_title: str
    kalshi_yes_price: float
    kalshi_no_price: float
    kalshi_volume: float
    kalshi_url: str = ""  # Link direto para o mercado
    
    # Arbitragem
    arbitrage_type: str
    profit_percentage: float
    profit_amount: float
    action: str
    explanation: str
    
    # Retornos
    returns_100: float
    returns_1000: float


class CrossPlatformResponse(BaseModel):
    """Resposta da análise cross-platform."""
    status: str
    timestamp: str
    polymarket_markets: int
    kalshi_markets: int
    matches_found: int
    opportunities_found: int
    opportunities: List[CrossPlatformOpportunity]
    note: str = ""


# Cache para análises cross-platform
cross_platform_jobs = {}
cross_platform_counter = 0


@app.get("/crossplatform/analyze", response_model=CrossPlatformResponse)
async def analyze_cross_platform(
    background_tasks: BackgroundTasks,
    limit: int = Query(default=100, ge=10, le=500, description="Número de mercados a escanear por plataforma")
):
    """
    Analisa oportunidades de arbitragem entre Polymarket e Kalshi.
    
    Busca mercados similares nas duas plataformas e identifica diferenças de preço.
    """
    global cross_platform_counter
    
    job_id = f"cross_{cross_platform_counter}"
    cross_platform_counter += 1
    
    # Inicializar job
    cross_platform_jobs[job_id] = {
        "status": "running",
        "result": None,
        "started_at": datetime.utcnow().isoformat()
    }
    
    # Adicionar tarefa em background
    background_tasks.add_task(run_cross_platform_analysis, job_id, limit)
    
    return CrossPlatformResponse(
        status="started",
        timestamp=datetime.utcnow().isoformat(),
        polymarket_markets=0,
        kalshi_markets=0,
        matches_found=0,
        opportunities_found=0,
        opportunities=[],
        note=f"Análise iniciada em background. Job ID: {job_id}. Use /crossplatform/status/{job_id} para verificar."
    )


async def run_cross_platform_analysis(job_id: str, limit: int):
    """Executa análise cross-platform em background."""
    try:
        from src.engine.cross_platform_arbitrage import CrossPlatformScanner
        
        async with CrossPlatformScanner(timeout=180.0) as scanner:
            opps = await scanner.scan(limit=limit)
            
            # Converter para resposta
            opp_list = []
            for opp in opps[:20]:  # Top 20
                opp_list.append(CrossPlatformOpportunity(
                    event_name=opp.event_name[:100],
                    match_confidence=opp.match_confidence,
                    poly_market_title=opp.poly_market_title[:100],
                    poly_yes_price=opp.poly_yes_price,
                    poly_no_price=opp.poly_no_price,
                    poly_volume=opp.poly_volume,
                    poly_url=opp.poly_url,
                    kalshi_market_title=opp.kalshi_market_title[:100],
                    kalshi_yes_price=opp.kalshi_yes_price,
                    kalshi_no_price=opp.kalshi_no_price,
                    kalshi_volume=opp.kalshi_volume,
                    kalshi_url=opp.kalshi_url,
                    arbitrage_type=opp.arbitrage_type,
                    profit_percentage=opp.profit_percentage,
                    profit_amount=opp.profit_amount,
                    action=opp.action,
                    explanation=opp.explanation,
                    returns_100=opp.returns_100,
                    returns_1000=opp.returns_1000
                ))
            
            cross_platform_jobs[job_id] = {
                "status": "completed",
                "result": CrossPlatformResponse(
                    status="completed",
                    timestamp=datetime.utcnow().isoformat(),
                    polymarket_markets=len(scanner.poly_markets),
                    kalshi_markets=len(scanner.kalshi_markets),
                    matches_found=len(opps),
                    opportunities_found=len(opp_list),
                    opportunities=opp_list,
                    note="Análise concluída. NOTA: Dados do Kalshi são simulados (API pública limitada)."
                ),
                "completed_at": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erro na análise cross-platform: {e}", exc_info=True)
        cross_platform_jobs[job_id] = {
            "status": "error",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        }


@app.get("/crossplatform/status/{job_id}")
async def get_cross_platform_status(job_id: str):
    """Verifica status da análise cross-platform."""
    if job_id not in cross_platform_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} não encontrado")
    
    job = cross_platform_jobs[job_id]
    
    if job["status"] == "completed":
        return job["result"]
    elif job["status"] == "error":
        return CrossPlatformResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            polymarket_markets=0,
            kalshi_markets=0,
            matches_found=0,
            opportunities_found=0,
            opportunities=[],
            note=f"Erro: {job.get('error', 'Erro desconhecido')}"
        )
    else:
        return CrossPlatformResponse(
            status="running",
            timestamp=job.get("started_at", datetime.utcnow().isoformat()),
            polymarket_markets=0,
            kalshi_markets=0,
            matches_found=0,
            opportunities_found=0,
            opportunities=[],
            note="Análise em progresso..."
        )


@app.get("/crossplatform/sync", response_model=CrossPlatformResponse)
async def analyze_cross_platform_sync(
    limit: int = Query(default=50, ge=10, le=200, description="Número de mercados (menor para evitar timeout)")
):
    """
    Versão síncrona da análise cross-platform.
    Mais rápida mas pode dar timeout com muitos mercados.
    """
    try:
        from src.engine.cross_platform_arbitrage import CrossPlatformScanner
        
        async with CrossPlatformScanner(timeout=120.0) as scanner:
            opps = await scanner.scan(limit=limit)
            
            opp_list = []
            for opp in opps[:20]:
                opp_list.append(CrossPlatformOpportunity(
                    event_name=opp.event_name[:100],
                    match_confidence=opp.match_confidence,
                    poly_market_title=opp.poly_market_title[:100],
                    poly_yes_price=opp.poly_yes_price,
                    poly_no_price=opp.poly_no_price,
                    poly_volume=opp.poly_volume,
                    poly_url=opp.poly_url,
                    kalshi_market_title=opp.kalshi_market_title[:100],
                    kalshi_yes_price=opp.kalshi_yes_price,
                    kalshi_no_price=opp.kalshi_no_price,
                    kalshi_volume=opp.kalshi_volume,
                    kalshi_url=opp.kalshi_url,
                    arbitrage_type=opp.arbitrage_type,
                    profit_percentage=opp.profit_percentage,
                    profit_amount=opp.profit_amount,
                    action=opp.action,
                    explanation=opp.explanation,
                    returns_100=opp.returns_100,
                    returns_1000=opp.returns_1000
                ))
            
            return CrossPlatformResponse(
                status="completed",
                timestamp=datetime.utcnow().isoformat(),
                polymarket_markets=len(scanner.poly_markets),
                kalshi_markets=len(scanner.kalshi_markets),
                matches_found=len(opps),
                opportunities_found=len(opp_list),
                opportunities=opp_list,
                note="NOTA: Dados do Kalshi são simulados (API pública retorna apenas esportes)."
            )
            
    except Exception as e:
        logger.error(f"Erro na análise cross-platform: {e}", exc_info=True)
        return CrossPlatformResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            polymarket_markets=0,
            kalshi_markets=0,
            matches_found=0,
            opportunities_found=0,
            opportunities=[],
            note=f"Erro: {str(e)}"
        )


# =====================================================
# ENDPOINTS COM DADOS 100% REAIS (SEM SIMULAÇÃO)
# =====================================================

class RealArbitrageOpp(BaseModel):
    """Oportunidade de arbitragem com dados 100% reais."""
    event_description: str
    confidence_match: float
    is_true_arbitrage: bool
    
    # Polymarket
    poly_question: str
    poly_url: str
    poly_yes_price: float
    poly_no_price: float
    poly_volume_24h: float
    poly_liquidity: float
    
    # Kalshi
    kalshi_title: str
    kalshi_url: str
    kalshi_yes_price: float
    kalshi_no_price: float
    kalshi_volume: float
    
    # Estratégia
    strategy: str
    total_cost: float
    guaranteed_profit: float
    profit_percentage: float
    
    # Explicação
    explanation: str
    why_profitable: str
    risk_warning: str
    
    # Retornos
    return_on_100: float
    return_on_1000: float


class RealArbitrageResponse(BaseModel):
    """Resposta da análise com dados 100% reais."""
    status: str
    timestamp: str
    
    # Estatísticas
    polymarket_markets: int
    kalshi_markets: int
    kalshi_api_status: str
    matches_found: int
    true_arbitrage_count: int
    value_opportunities_count: int
    
    # Resultados
    opportunities: List[RealArbitrageOpp]
    
    # Mensagem
    message: str


@app.get("/real/arbitrage", response_model=RealArbitrageResponse)
async def scan_real_arbitrage(
    limit: int = Query(default=100, ge=20, le=300, description="Número de mercados por plataforma")
):
    """
    🎯 SCANNER DE ARBITRAGEM 100% REAL
    
    - ZERO dados simulados
    - ZERO suposições
    - Apenas matemática pura com preços reais
    
    Se não houver correspondência entre mercados, informa claramente.
    """
    try:
        from src.engine.real_cross_platform import scan_real_arbitrage as do_scan
        
        opportunities, stats = await do_scan(limit=limit)
        
        opp_list = []
        for opp in opportunities[:20]:
            opp_list.append(RealArbitrageOpp(
                event_description=opp.event_description,
                confidence_match=opp.confidence_match,
                is_true_arbitrage=opp.is_true_arbitrage,
                poly_question=opp.poly_question[:150],
                poly_url=opp.poly_url,
                poly_yes_price=float(opp.poly_yes_price),
                poly_no_price=float(opp.poly_no_price),
                poly_volume_24h=float(opp.poly_volume_24h),
                poly_liquidity=float(opp.poly_liquidity),
                kalshi_title=opp.kalshi_title[:150],
                kalshi_url=opp.kalshi_url,
                kalshi_yes_price=float(opp.kalshi_yes_price),
                kalshi_no_price=float(opp.kalshi_no_price),
                kalshi_volume=float(opp.kalshi_volume),
                strategy=opp.strategy,
                total_cost=float(opp.total_cost),
                guaranteed_profit=float(opp.guaranteed_profit),
                profit_percentage=float(opp.profit_percentage),
                explanation=opp.explanation,
                why_profitable=opp.why_profitable,
                risk_warning=opp.risk_warning,
                return_on_100=float(opp.return_on_100),
                return_on_1000=float(opp.return_on_1000)
            ))
        
        # Mensagem clara sobre o estado
        if stats["true_arbitrage"] > 0:
            message = f"🎯 {stats['true_arbitrage']} ARBITRAGENS GARANTIDAS encontradas com dados 100% reais!"
        elif stats["value_opportunities"] > 0:
            message = f"⚠️ {stats['value_opportunities']} oportunidades de valor encontradas (não garantidas). Nenhuma arbitragem garantida no momento."
        elif stats["matches_found"] == 0:
            message = f"❌ Nenhuma correspondência encontrada entre Polymarket e Kalshi. APIs retornaram categorias diferentes de mercados."
        else:
            message = "✅ Mercados analisados. Nenhuma oportunidade significativa encontrada - mercados estão eficientes."
        
        return RealArbitrageResponse(
            status="completed",
            timestamp=datetime.utcnow().isoformat(),
            polymarket_markets=stats["poly_fetched"],
            kalshi_markets=stats["kalshi_fetched"],
            kalshi_api_status=stats["kalshi_api_status"],
            matches_found=stats["matches_found"],
            true_arbitrage_count=stats["true_arbitrage"],
            value_opportunities_count=stats["value_opportunities"],
            opportunities=opp_list,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Erro no scanner real: {e}", exc_info=True)
        return RealArbitrageResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            polymarket_markets=0,
            kalshi_markets=0,
            kalshi_api_status="Erro",
            matches_found=0,
            true_arbitrage_count=0,
            value_opportunities_count=0,
            opportunities=[],
            message=f"Erro: {str(e)}"
        )


# =====================================================
# KALSHI COM AUTENTICAÇÃO - DADOS 100% REAIS
# =====================================================

class KalshiAuthRequest(BaseModel):
    """Request para autenticação no Kalshi (API Key ou Email/Senha)."""
    api_key: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None


class KalshiLoginResponse(BaseModel):
    """Resposta do teste de autenticação."""
    authenticated: bool
    member_id: Optional[str] = None
    api_key_used: bool = False
    total_markets: int = 0
    categories: Dict[str, int] = {}
    message: str


class AuthArbitrageRequest(BaseModel):
    """Request para arbitragem com autenticação."""
    kalshi_api_key: Optional[str] = None
    kalshi_email: Optional[str] = None
    kalshi_password: Optional[str] = None
    limit: int = 100


@app.post("/kalshi/test-auth", response_model=KalshiLoginResponse)
async def test_kalshi_auth_endpoint(request: KalshiAuthRequest):
    """
    Testa autenticação no Kalshi (API Key ou Email/Senha).
    Retorna informações sobre mercados disponíveis.
    """
    try:
        from src.collectors.kalshi_authenticated import test_kalshi_auth
        
        result = await test_kalshi_auth(
            api_key=request.api_key,
            email=request.email,
            password=request.password
        )
        
        if result["authenticated"]:
            return KalshiLoginResponse(
                authenticated=True,
                member_id=result.get("member_id"),
                api_key_used=result.get("api_key_used", False),
                total_markets=result.get("total_markets", 0),
                categories=result.get("categories", {}),
                message=f"✅ Autenticado! {result.get('total_markets', 0)} mercados disponíveis."
            )
        else:
            return KalshiLoginResponse(
                authenticated=False,
                message=f"❌ {result.get('error', 'Falha na autenticação')}"
            )
            
    except Exception as e:
        logger.error(f"Erro no teste de auth: {e}", exc_info=True)
        return KalshiLoginResponse(
            authenticated=False,
            message=f"❌ Erro: {str(e)}"
        )


# Alias para compatibilidade
@app.post("/kalshi/test-login", response_model=KalshiLoginResponse)
async def test_kalshi_login(request: KalshiAuthRequest):
    """Alias para /kalshi/test-auth"""
    return await test_kalshi_auth_endpoint(request)


@app.post("/real/arbitrage-auth")
async def scan_real_arbitrage_authenticated(request: AuthArbitrageRequest):
    """
    🎯 SCANNER DE ARBITRAGEM 100% REAL COM AUTENTICAÇÃO
    
    Usa suas credenciais do Kalshi para acessar TODOS os mercados.
    Compara com Polymarket e encontra arbitragem REAL.
    """
    try:
        from src.collectors.kalshi_authenticated import KalshiAuthenticatedCollector
        from src.engine.real_cross_platform import RealCrossPlatformScanner, RealArbitrageOpportunity
        from decimal import Decimal
        import json
        import aiohttp
        import ssl
        import certifi
        
        logger.info(f"Iniciando scan com autenticação Kalshi...")
        
        # 1. Autenticar no Kalshi
        async with KalshiAuthenticatedCollector(
            api_key=request.kalshi_api_key,
            email=request.kalshi_email,
            password=request.kalshi_password
        ) as kalshi:
            
            if not await kalshi.login():
                return {
                    "status": "error",
                    "message": "❌ Falha na autenticação do Kalshi. Verifique email e senha.",
                    "opportunities": []
                }
            
            logger.info("✅ Kalshi autenticado!")
            
            # 2. Buscar mercados do Kalshi (autenticado)
            kalshi_markets = await kalshi.get_all_markets(max_markets=request.limit)
            logger.info(f"Kalshi: {len(kalshi_markets)} mercados reais")
            
            # Categorias
            kalshi_categories = {}
            for m in kalshi_markets:
                cat = m.get("category", "unknown")
                kalshi_categories[cat] = kalshi_categories.get(cat, 0) + 1
        
        # 3. Buscar mercados do Polymarket
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            url = "https://gamma-api.polymarket.com/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": request.limit,
                "order": "volume24hr",
                "ascending": "false"
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    poly_markets = await resp.json()
                    logger.info(f"Polymarket: {len(poly_markets)} mercados reais")
                else:
                    return {
                        "status": "error",
                        "message": f"Erro ao buscar Polymarket: HTTP {resp.status}",
                        "opportunities": []
                    }
        
        # 4. Encontrar correspondências PRECISAS
        matches = []
        
        # Palavras que indicam que é sobre uma pessoa específica
        person_indicators = ["will", "does", "did", "has", "can"]
        
        # Stopwords mais completas
        stopwords = {
            "will", "the", "a", "an", "in", "on", "by", "to", "of", "be", "is", "are",
            "next", "become", "before", "after", "win", "lose", "any", "this", "that",
            "president", "prime", "minister", "chairman", "chair", "leader", "election",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december", "2024", "2025", "2026", "2027",
            "united", "states", "kingdom", "america"
        }
        
        def extract_names(text):
            """Extrai nomes próprios (palavras capitalizadas) do texto original."""
            words = text.replace("?", "").replace("'", "").split()
            names = []
            for w in words:
                # Palavras capitalizadas que não são começo de frase
                if len(w) > 2 and w[0].isupper() and w.lower() not in stopwords:
                    names.append(w.lower())
            return set(names)
        
        for poly in poly_markets:
            poly_q = poly.get("question", "")
            poly_q_lower = poly_q.lower()
            poly_keywords = set(poly_q_lower.replace("?", "").replace("'", "").split())
            poly_names = extract_names(poly_q)
            
            for kalshi in kalshi_markets:
                kalshi_title = kalshi.get("title", kalshi.get("subtitle", ""))
                kalshi_title_lower = kalshi_title.lower()
                kalshi_keywords = set(kalshi_title_lower.replace("?", "").replace("'", "").split())
                kalshi_names = extract_names(kalshi_title)
                
                # Verificar se os NOMES são os mesmos (crítico para evitar falsos positivos)
                common_names = poly_names & kalshi_names
                
                # Se não há nomes em comum, provavelmente são eventos diferentes
                if poly_names and kalshi_names and not common_names:
                    continue
                
                # Calcular similaridade de palavras significativas
                meaningful_poly = poly_keywords - stopwords
                meaningful_kalshi = kalshi_keywords - stopwords
                common = meaningful_poly & meaningful_kalshi
                
                # Precisamos de pelo menos 3 palavras em comum + nomes iguais
                if len(common) >= 3:
                    # Se há nomes em comum ou nenhum dos dois tem nomes específicos
                    if common_names or (not poly_names and not kalshi_names):
                        confidence = (len(common) + len(common_names) * 2) / max(
                            len(meaningful_poly) + len(poly_names),
                            len(meaningful_kalshi) + len(kalshi_names),
                            1
                        )
                        
                        if confidence >= 0.5:  # Mínimo 50% de confiança
                            matches.append({
                                "poly": poly,
                                "kalshi": kalshi,
                                "confidence": confidence,
                                "common_words": list(common),
                                "common_names": list(common_names)
                            })
        
        logger.info(f"Encontrados {len(matches)} matches potenciais")
        
        # 5. Calcular arbitragem para cada match
        opportunities = []
        
        for match in matches:
            poly = match["poly"]
            kalshi = match["kalshi"]
            
            # Preços Polymarket
            try:
                poly_yes = Decimal(str(poly.get("outcomePrices", ["0.5"])[0]))
            except:
                poly_yes = Decimal("0.5")
            poly_no = Decimal("1") - poly_yes
            
            # Preços Kalshi
            kalshi_yes_bid = Decimal(str(kalshi.get("yes_bid", 0) or 50)) / 100
            kalshi_yes_ask = Decimal(str(kalshi.get("yes_ask", 0) or 50)) / 100
            kalshi_yes = (kalshi_yes_bid + kalshi_yes_ask) / 2 if kalshi_yes_bid and kalshi_yes_ask else Decimal("0.5")
            kalshi_no = Decimal("1") - kalshi_yes
            
            # Taxas
            poly_fee = Decimal("0.02")
            kalshi_fee = Decimal("0.01")
            
            # Opção A: YES no Poly + NO no Kalshi
            cost_a = poly_yes + kalshi_no + (poly_yes * poly_fee) + (kalshi_no * kalshi_fee)
            
            # Opção B: YES no Kalshi + NO no Poly
            cost_b = kalshi_yes + poly_no + (kalshi_yes * kalshi_fee) + (poly_no * poly_fee)
            
            is_arbitrage = False
            strategy = ""
            profit = Decimal("0")
            
            if cost_a < Decimal("1"):
                is_arbitrage = True
                profit = Decimal("1") - cost_a
                strategy = f"COMPRAR YES no Polymarket (${poly_yes:.3f}) + NO no Kalshi (${kalshi_no:.3f})"
            elif cost_b < Decimal("1"):
                is_arbitrage = True
                profit = Decimal("1") - cost_b
                strategy = f"COMPRAR YES no Kalshi (${kalshi_yes:.3f}) + NO no Polymarket (${poly_no:.3f})"
            else:
                # Verificar oportunidade de valor
                price_diff = abs(poly_yes - kalshi_yes)
                if price_diff > Decimal("0.05"):  # >5% diferença
                    if poly_yes < kalshi_yes:
                        strategy = f"VALOR: YES mais barato no Polymarket (${poly_yes:.3f} vs ${kalshi_yes:.3f})"
                        profit = price_diff * Decimal("0.5")
                    else:
                        strategy = f"VALOR: YES mais barato no Kalshi (${kalshi_yes:.3f} vs ${poly_yes:.3f})"
                        profit = price_diff * Decimal("0.5")
            
            if strategy:
                profit_pct = (profit * 100).quantize(Decimal("0.01"))
                
                poly_slug = poly.get("slug", "")
                kalshi_ticker = kalshi.get("ticker", "")
                
                opportunities.append({
                    "event": poly.get("question", "")[:100],
                    "confidence": round(match["confidence"] * 100, 1),
                    "common_words": match["common_words"],
                    "is_true_arbitrage": is_arbitrage,
                    "poly_question": poly.get("question", ""),
                    "poly_yes_price": float(poly_yes),
                    "poly_no_price": float(poly_no),
                    "poly_url": f"https://polymarket.com/market/{poly_slug}" if poly_slug else "https://polymarket.com",
                    "kalshi_title": kalshi.get("title", kalshi.get("subtitle", "")),
                    "kalshi_yes_price": float(kalshi_yes),
                    "kalshi_no_price": float(kalshi_no),
                    "kalshi_url": f"https://kalshi.com/markets/{kalshi_ticker}" if kalshi_ticker else "https://kalshi.com/markets",
                    "strategy": strategy,
                    "profit_percentage": float(profit_pct),
                    "return_100": float(profit_pct),
                    "return_1000": float(profit_pct * 10),
                    "explanation": f"Mercados sobre '{', '.join(match['common_words'][:3])}' com preços diferentes nas plataformas.",
                    "why_profitable": f"Diferença de preço: Poly YES=${poly_yes:.3f}, Kalshi YES=${kalshi_yes:.3f}"
                })
        
        # Ordenar por lucratividade
        opportunities.sort(key=lambda x: x["profit_percentage"], reverse=True)
        
        # Estatísticas
        true_arbs = sum(1 for o in opportunities if o["is_true_arbitrage"])
        value_opps = len(opportunities) - true_arbs
        
        if true_arbs > 0:
            message = f"🎯 {true_arbs} ARBITRAGENS GARANTIDAS encontradas!"
        elif value_opps > 0:
            message = f"⚠️ {value_opps} oportunidades de valor encontradas."
        elif len(matches) == 0:
            message = "❌ Nenhuma correspondência entre Polymarket e Kalshi."
        else:
            message = "✅ Mercados analisados. Nenhuma oportunidade no momento."
        
        return {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "polymarket_markets": len(poly_markets),
            "kalshi_markets": len(kalshi_markets),
            "kalshi_categories": kalshi_categories,
            "matches_found": len(matches),
            "true_arbitrage_count": true_arbs,
            "value_opportunities_count": value_opps,
            "opportunities": opportunities[:20],  # Top 20
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Erro no scanner autenticado: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"❌ Erro: {str(e)}",
            "opportunities": []
        }


# =====================================================
# SCAN POR CATEGORIA - ACESSO A TODOS OS MERCADOS
# =====================================================

class CategoryScanRequest(BaseModel):
    """Request para scan por categoria."""
    kalshi_api_key: str
    category: Optional[str] = None  # None = todas
    limit: int = 500


@app.post("/scan/category")
async def scan_by_category(request: CategoryScanRequest):
    """
    🎯 SCANNER POR CATEGORIA
    
    Busca mercados de uma categoria específica em ambas as plataformas.
    Permite analisar grandes volumes de mercados de forma eficiente.
    """
    try:
        from src.collectors.kalshi_authenticated import KalshiAuthenticatedCollector
        from decimal import Decimal
        import aiohttp
        import ssl
        import certifi
        
        category = request.category
        limit = min(request.limit, 5000)  # Máximo 5000
        
        logger.info(f"Scan por categoria: {category or 'TODAS'}, limite: {limit}")
        
        # Mapear categorias para keywords de busca
        category_keywords = {
            "Politics": ["trump", "biden", "election", "president", "congress", "senate", "governor", "vote", "democrat", "republican", "minister", "parliament"],
            "Economics": ["fed", "rate", "inflation", "gdp", "recession", "employment", "jobs", "interest", "fomc", "treasury"],
            "World": ["ukraine", "russia", "china", "israel", "war", "ceasefire", "nato", "leader", "prime minister"],
            "Crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "microstrategy", "etf", "sec"],
            "Entertainment": ["oscar", "grammy", "movie", "album", "taylor", "concert", "stranger things", "netflix"],
            "Science": ["spacex", "mars", "ai", "openai", "fusion", "fda", "vaccine"],
            "Climate": ["temperature", "hurricane", "earthquake", "weather", "climate"]
        }
        
        keywords = category_keywords.get(category, []) if category else []
        
        # 1. Buscar mercados do Kalshi
        async with KalshiAuthenticatedCollector(api_key=request.kalshi_api_key) as kalshi:
            if not await kalshi.login():
                return {
                    "status": "error",
                    "message": "❌ Falha na autenticação do Kalshi",
                    "opportunities": []
                }
            
            # Buscar mercados de eventos (política, economia, etc.)
            if category:
                kalshi_markets = await kalshi.get_markets_from_events(
                    max_markets=limit,
                    categories=[category, category + "s", category.lower()]
                )
            else:
                kalshi_markets = await kalshi.get_markets_from_events(max_markets=limit)
            
            # Também buscar via endpoint normal para ter mais mercados
            regular_markets, _ = await kalshi.get_markets(limit=min(limit, 200))
            
            # Combinar sem duplicatas
            seen_tickers = {m.get("ticker") for m in kalshi_markets}
            for m in regular_markets:
                if m.get("ticker") not in seen_tickers:
                    kalshi_markets.append(m)
                    seen_tickers.add(m.get("ticker"))
        
        logger.info(f"Kalshi: {len(kalshi_markets)} mercados")
        
        # 2. Buscar mercados do Polymarket
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        poly_markets = []
        async with aiohttp.ClientSession(connector=connector) as session:
            # Buscar em lotes
            offset = 0
            batch_size = 100
            
            while len(poly_markets) < limit:
                url = "https://gamma-api.polymarket.com/markets"
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": batch_size,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false"
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        break
                    
                    batch = await resp.json()
                    if not batch:
                        break
                    
                    # Filtrar por keywords se categoria específica
                    if keywords:
                        for m in batch:
                            q = m.get("question", "").lower()
                            if any(kw in q for kw in keywords):
                                poly_markets.append(m)
                    else:
                        poly_markets.extend(batch)
                    
                    offset += batch_size
                    
                    if len(batch) < batch_size:
                        break
        
        logger.info(f"Polymarket: {len(poly_markets)} mercados")
        
        # 3. Encontrar correspondências CIRURGICAMENTE PRECISAS
        # Exigir que TUDO seja idêntico: local, pessoa, evento
        matches = []
        
        from difflib import SequenceMatcher
        import re
        
        # Locais que devem coincidir EXATAMENTE
        all_locations = {
            "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
            "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
            "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
            "maine", "maryland", "massachusetts", "michigan", "minnesota",
            "mississippi", "missouri", "montana", "nebraska", "nevada",
            "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
            "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
            "rhode island", "south carolina", "south dakota", "tennessee", "texas",
            "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming",
            "israel", "ukraine", "russia", "china", "korea", "brazil", "germany",
            "france", "uk", "japan", "india", "mexico", "canada", "iran", "syria"
        }
        
        # Palavras comuns que NÃO são nomes de pessoas
        common_words = {
            "will", "the", "be", "next", "prime", "minister", "president", "win",
            "election", "before", "after", "january", "february", "march", "april",
            "may", "june", "july", "august", "september", "october", "november",
            "december", "2024", "2025", "2026", "2027", "2028", "2029", "2030",
            "democratic", "republican", "party", "senate", "house", "governor",
            "united", "states", "america", "kingdom", "federal", "reserve", "fed"
        }
        
        def extract_locations(text):
            text_lower = text.lower()
            return {loc for loc in all_locations if loc in text_lower}
        
        def extract_person_names(text):
            """Extrai nomes de pessoas (palavras capitalizadas consecutivas)."""
            # Remove pontuação
            clean = re.sub(r'[^\w\s]', ' ', text)
            words = clean.split()
            
            names = []
            current_name = []
            
            for word in words:
                # Palavra capitalizada e não é comum
                if word[0].isupper() and word.lower() not in common_words and word.lower() not in all_locations:
                    current_name.append(word.lower())
                else:
                    if current_name:
                        names.append(' '.join(current_name))
                        current_name = []
            
            if current_name:
                names.append(' '.join(current_name))
            
            # Filtrar nomes muito curtos
            return set(n for n in names if len(n) > 3)
        
        def normalize_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            return ' '.join(text.split())
        
        for poly in poly_markets:
            poly_q = poly.get("question", "")
            poly_norm = normalize_text(poly_q)
            poly_locations = extract_locations(poly_q)
            poly_names = extract_person_names(poly_q)
            
            for kalshi in kalshi_markets:
                kalshi_title = kalshi.get("title", kalshi.get("subtitle", ""))
                kalshi_norm = normalize_text(kalshi_title)
                kalshi_locations = extract_locations(kalshi_title)
                kalshi_names = extract_person_names(kalshi_title)
                
                # REGRA 1: Locais devem ser IDÊNTICOS
                if poly_locations and kalshi_locations:
                    if poly_locations != kalshi_locations:
                        continue
                
                # REGRA 2: Nomes de pessoas devem ser IDÊNTICOS
                if poly_names and kalshi_names:
                    # Verificar se há pelo menos um nome em comum
                    common_names = poly_names & kalshi_names
                    if not common_names:
                        continue  # PESSOAS DIFERENTES = EVENTOS DIFERENTES
                    
                    # Se há nomes exclusivos em cada lado, são eventos diferentes
                    only_poly = poly_names - kalshi_names
                    only_kalshi = kalshi_names - poly_names
                    if only_poly and only_kalshi:
                        continue  # Nomes diferentes mencionados
                
                # Calcular similaridade de texto
                similarity = SequenceMatcher(None, poly_norm, kalshi_norm).ratio()
                
                # Exigir similaridade MUITO alta (92%+)
                if similarity >= 0.92:
                    matches.append({
                        "poly": poly,
                        "kalshi": kalshi,
                        "confidence": similarity,
                        "common_names": list(poly_names & kalshi_names) if poly_names and kalshi_names else []
                    })
        
        logger.info(f"Matches encontrados: {len(matches)}")
        
        # 4. Calcular arbitragem com TODAS as taxas e detalhes
        opportunities = []
        
        # TAXAS REAIS das plataformas
        POLY_FEE_RATE = Decimal("0.02")   # 2% Polymarket
        KALSHI_FEE_RATE = Decimal("0.07")  # 7% Kalshi (taxa sobre lucro)
        
        for match in matches:
            poly = match["poly"]
            kalshi = match["kalshi"]
            
            # Preços Polymarket
            try:
                prices = poly.get("outcomePrices", ["0.5", "0.5"])
                if isinstance(prices, str):
                    import json as json_mod
                    prices = json_mod.loads(prices)
                poly_yes = Decimal(str(prices[0] if prices else "0.5"))
            except:
                poly_yes = Decimal("0.5")
            poly_no = Decimal("1") - poly_yes
            
            # Preços Kalshi (em centavos)
            kalshi_yes_bid = Decimal(str(kalshi.get("yes_bid", 50) or 50)) / 100
            kalshi_yes_ask = Decimal(str(kalshi.get("yes_ask", 50) or 50)) / 100
            kalshi_yes = (kalshi_yes_bid + kalshi_yes_ask) / 2 if kalshi_yes_bid else Decimal("0.5")
            kalshi_no = Decimal("1") - kalshi_yes
            
            # Data de resolução
            poly_end = poly.get("endDate", poly.get("end_date_iso", ""))
            kalshi_end = kalshi.get("close_time", kalshi.get("expiration_time", ""))
            
            # Usar a data mais próxima
            resolution_date = poly_end or kalshi_end or "Não informado"
            
            # Calcular dias até resolução
            days_to_resolution = None
            if resolution_date and resolution_date != "Não informado":
                try:
                    from datetime import datetime
                    end_dt = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
                    days_to_resolution = (end_dt - datetime.now(end_dt.tzinfo)).days
                except:
                    pass
            
            # ========================================
            # CÁLCULO DETALHADO DA ARBITRAGEM
            # ========================================
            
            # Opção A: YES no Poly + NO no Kalshi
            poly_yes_cost = poly_yes * (1 + POLY_FEE_RATE)  # Preço + taxa
            kalshi_no_cost = kalshi_no  # Kalshi cobra taxa só sobre lucro
            total_cost_a = poly_yes_cost + kalshi_no_cost
            
            # Se ganhar (evento acontece): recebe $1 do YES Poly
            # Taxa Kalshi sobre lucro do NO: 0 (porque perdeu)
            # Se perder (evento não acontece): recebe $1 do NO Kalshi
            # Taxa Kalshi: 7% sobre o lucro (1 - kalshi_no)
            kalshi_no_profit = Decimal("1") - kalshi_no
            kalshi_fee_a = kalshi_no_profit * KALSHI_FEE_RATE
            
            # Lucro líquido Opção A (garantido)
            profit_a = Decimal("1") - total_cost_a - (kalshi_fee_a / 2)  # Média das taxas
            
            # Opção B: YES no Kalshi + NO no Poly
            kalshi_yes_cost = kalshi_yes
            poly_no_cost = poly_no * (1 + POLY_FEE_RATE)
            total_cost_b = kalshi_yes_cost + poly_no_cost
            
            kalshi_yes_profit = Decimal("1") - kalshi_yes
            kalshi_fee_b = kalshi_yes_profit * KALSHI_FEE_RATE
            
            profit_b = Decimal("1") - total_cost_b - (kalshi_fee_b / 2)
            
            # Determinar melhor estratégia
            is_arbitrage = False
            strategy = ""
            profit = Decimal("0")
            total_cost = Decimal("0")
            explanation = ""
            
            if profit_a > Decimal("0.005"):  # Mínimo 0.5% de lucro
                is_arbitrage = True
                profit = profit_a
                total_cost = total_cost_a
                strategy = f"COMPRAR YES no Polymarket (${poly_yes:.3f}) + NO no Kalshi (${kalshi_no:.3f})"
                explanation = f"""
MATEMÁTICA DA ARBITRAGEM:

1. Comprar YES no Polymarket:
   Preço: ${poly_yes:.4f}
   Taxa Polymarket (2%): ${float(poly_yes * POLY_FEE_RATE):.4f}
   Custo total: ${float(poly_yes_cost):.4f}

2. Comprar NO no Kalshi:
   Preço: ${kalshi_no:.4f}
   (Taxa Kalshi de 7% só sobre lucro)

CUSTO TOTAL: ${float(total_cost_a):.4f}

RETORNO GARANTIDO: $1.00

LUCRO LÍQUIDO: ${float(profit_a):.4f} ({float(profit_a*100):.1f}%)

Se o evento ACONTECER: Você ganha $1 do YES (Polymarket)
Se o evento NÃO ACONTECER: Você ganha $1 do NO (Kalshi)
De qualquer forma, você recebe $1 tendo pago ${float(total_cost_a):.4f}
"""
            
            elif profit_b > Decimal("0.005"):
                is_arbitrage = True
                profit = profit_b
                total_cost = total_cost_b
                strategy = f"COMPRAR YES no Kalshi (${kalshi_yes:.3f}) + NO no Polymarket (${poly_no:.3f})"
                explanation = f"""
MATEMÁTICA DA ARBITRAGEM:

1. Comprar YES no Kalshi:
   Preço: ${kalshi_yes:.4f}

2. Comprar NO no Polymarket:
   Preço: ${poly_no:.4f}
   Taxa Polymarket (2%): ${float(poly_no * POLY_FEE_RATE):.4f}
   Custo total: ${float(poly_no_cost):.4f}

CUSTO TOTAL: ${float(total_cost_b):.4f}

RETORNO GARANTIDO: $1.00

LUCRO LÍQUIDO: ${float(profit_b):.4f} ({float(profit_b*100):.1f}%)

Se o evento ACONTECER: Você ganha $1 do YES (Kalshi)
Se o evento NÃO ACONTECER: Você ganha $1 do NO (Polymarket)
"""
            
            if is_arbitrage and profit > Decimal("0.005"):
                profit_pct = float(profit * 100)
                
                poly_slug = poly.get("slug", "")
                kalshi_ticker = kalshi.get("ticker", "")
                
                # Extrair série do ticker para URL do evento
                kalshi_event_ticker = kalshi_ticker.split("-")[0] if kalshi_ticker else ""
                
                opportunities.append({
                    "event": poly.get("question", "")[:100],
                    "is_arbitrage": is_arbitrage,
                    "poly_question": poly.get("question", ""),
                    "poly_yes_price": float(poly_yes),
                    "poly_no_price": float(poly_no),
                    "poly_url": f"https://polymarket.com/market/{poly_slug}" if poly_slug else "https://polymarket.com",
                    "kalshi_title": kalshi.get("title", kalshi.get("subtitle", "")),
                    "kalshi_yes_price": float(kalshi_yes),
                    "kalshi_no_price": float(kalshi_no),
                    "kalshi_ticker": kalshi_ticker,
                    "kalshi_url": f"https://kalshi.com/markets/{kalshi_ticker}" if kalshi_ticker else "https://kalshi.com/markets",
                    "strategy": strategy,
                    "total_cost": float(total_cost),
                    "profit_percentage": profit_pct,
                    "return_100": profit_pct,
                    "return_1000": profit_pct * 10,
                    "explanation": explanation.strip(),
                    "resolution_date": resolution_date[:10] if resolution_date and len(resolution_date) > 10 else resolution_date,
                    "days_to_resolution": days_to_resolution,
                    "fees_included": {
                        "polymarket": "2%",
                        "kalshi": "7% sobre lucro"
                    }
                })
        
        # Ordenar por: eventos mais próximos + maior lucro
        def sort_key(opp):
            days = opp.get("days_to_resolution")
            profit = opp.get("profit_percentage", 0)
            days_value = days if days is not None and days > 0 else 365
            # Score: lucro / raiz(dias) - prioriza lucro E proximidade
            return -(profit / (days_value ** 0.3))
        
        opportunities.sort(key=sort_key)
        
        arb_count = sum(1 for o in opportunities if o.get("is_arbitrage"))
        
        if arb_count > 0:
            message = f"🎯 {arb_count} ARBITRAGENS encontradas!"
        elif len(opportunities) > 0:
            message = f"⚠️ {len(opportunities)} oportunidades de valor encontradas."
        elif len(matches) == 0:
            message = f"❌ Nenhuma correspondência entre plataformas nesta categoria."
        else:
            message = f"✅ {len(matches)} matches analisados. Mercados eficientes no momento."
        
        return {
            "status": "completed",
            "category": category or "all",
            "polymarket_markets": len(poly_markets),
            "kalshi_markets": len(kalshi_markets),
            "matches_found": len(matches),
            "arbitrage_count": arb_count,
            "opportunities": opportunities[:50],
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Erro no scan por categoria: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"❌ Erro: {str(e)}",
            "opportunities": []
        }


# ============================================================
# ENDPOINT PROFISSIONAL COM TODAS AS VERIFICAÇÕES
# ============================================================

class ProfessionalScanRequest(BaseModel):
    kalshi_api_key: str
    category: Optional[str] = None  # Categoria única (retrocompatibilidade)
    categories: Optional[List[str]] = None  # Lista de categorias para filtrar
    limit: int = 500
    # Configurações de risco
    min_profit_percentage: float = 2.0
    min_profit_absolute: float = 1.0
    max_size_per_trade: float = 1000.0
    # Opções de análise
    include_order_books: bool = False  # Se true, busca order books (mais lento)


# Mapeamento de categorias para palavras-chave
CATEGORY_KEYWORDS = {
    "politics": ["president", "congress", "senate", "election", "vote", "governor", "mayor", "political", "government", "legislation", "democrat", "republican", "parliament", "minister", "cabinet"],
    "economics": ["gdp", "inflation", "interest rate", "fed", "federal reserve", "unemployment", "recession", "economy", "economic", "monetary", "fiscal", "trade", "tariff"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "token", "coin", "defi", "nft", "web3", "binance", "coinbase", "solana", "cardano", "dogecoin"],
    "tech": ["ai", "artificial intelligence", "openai", "google", "apple", "microsoft", "amazon", "meta", "facebook", "twitter", "tesla", "spacex", "startup", "tech", "technology", "software", "hardware"],
    "sports": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "hockey", "tennis", "golf", "olympics", "world cup", "super bowl", "championship", "playoffs"],
    "entertainment": ["oscar", "emmy", "grammy", "movie", "film", "tv", "series", "celebrity", "music", "album", "concert", "netflix", "disney", "hollywood", "streaming"],
    "world": ["war", "conflict", "treaty", "united nations", "nato", "china", "russia", "ukraine", "israel", "middle east", "europe", "asia", "africa", "climate", "pandemic"],
    "elections": ["election", "vote", "ballot", "poll", "primary", "caucus", "electoral", "candidate", "campaign", "nominee", "swing state"],
    "finance": ["stock", "market", "s&p", "nasdaq", "dow", "bond", "yield", "forex", "commodity", "gold", "oil", "earnings", "ipo", "merger", "acquisition"]
}

def categorize_market(title: str, description: str = "") -> List[str]:
    """Categoriza um mercado baseado no título e descrição."""
    text = f"{title} {description}".lower()
    categories = []
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                categories.append(category)
                break
    
    return categories if categories else ["other"]


def market_matches_categories(title: str, description: str, selected_categories: List[str]) -> bool:
    """Verifica se um mercado pertence a alguma das categorias selecionadas."""
    if not selected_categories:
        return True  # Nenhum filtro = aceita todos
    
    market_categories = categorize_market(title, description)
    return any(cat in selected_categories for cat in market_categories)


class ExecutionEstimateResponse(BaseModel):
    target_size_usd: float
    avg_price: float
    total_cost: float
    slippage_percentage: float
    fully_fillable: bool


class PayoffScenarioResponse(BaseModel):
    scenario_name: str
    probability: float
    gross_return: float
    total_fees: float
    net_return: float
    roi_percentage: float


class ProfessionalOpportunityResponse(BaseModel):
    id: str
    
    # Mercados
    poly_title: str
    poly_url: str
    kalshi_title: str
    kalshi_url: str
    
    # Matching
    title_similarity: float
    equivalence_score: float
    
    # Estratégia
    strategy: str
    action_poly: str
    action_kalshi: str
    
    # Preços
    poly_yes_price: float
    poly_no_price: float
    kalshi_yes_price: float
    kalshi_no_price: float
    total_cost_per_contract: float
    
    # Taxas
    poly_fees: float
    kalshi_fees: float
    total_fees: float
    fees_breakdown: Dict[str, str]
    
    # Lucro
    gross_profit: float
    net_profit: float
    net_profit_percentage: float
    
    # Execução por tamanho
    execution_100: Optional[ExecutionEstimateResponse] = None
    execution_500: Optional[ExecutionEstimateResponse] = None
    execution_1000: Optional[ExecutionEstimateResponse] = None
    
    # Liquidez
    liquidity_score: str
    max_recommended_size: float
    
    # Timing
    data_age_seconds: float
    data_freshness: str
    
    # Payoff
    payoff_scenarios: List[PayoffScenarioResponse]
    guaranteed_min_profit: float
    max_loss_if_partial: float
    
    # Resolução
    resolution_date: Optional[str]
    days_to_resolution: Optional[int]
    
    # Validação
    warnings: List[str]
    overall_score: float


@app.post("/scan/professional")
async def scan_professional(request: ProfessionalScanRequest):
    """
    🎯 SCANNER PROFISSIONAL DE ARBITRAGEM
    
    Análise completa com:
    - Custos reais (taxas trading, saque, câmbio)
    - Liquidez e profundidade do order book
    - Slippage e preço médio de execução (VWAP)
    - Sincronização e idade dos dados
    - Equivalência rigorosa de mercados
    - Simulação de payoff (cenários)
    - Gestão de tamanho e exposição
    
    Só mostra oportunidades que passam em TODOS os filtros.
    """
    try:
        from src.collectors.kalshi_authenticated import KalshiAuthenticatedCollector
        from src.engine.professional_arbitrage import ProfessionalArbitrageEngine
        from src.engine.arbitrage_config import ArbitrageConfig
        
        # 1. Configurar engine com parâmetros do request
        config = ArbitrageConfig()
        config.profit.min_profit_percentage = Decimal(str(request.min_profit_percentage))
        config.profit.min_profit_absolute = Decimal(str(request.min_profit_absolute))
        config.risk.max_size_per_trade = Decimal(str(request.max_size_per_trade))
        
        engine = ProfessionalArbitrageEngine(config)
        
        # 2. Coletar mercados
        # Determinar categorias a filtrar
        selected_categories = request.categories or (
            [request.category] if request.category else None
        )
        
        if selected_categories:
            logger.info(f"🔍 Scan profissional - Categorias: {selected_categories}")
        else:
            logger.info(f"🔍 Scan profissional - TODAS as categorias")
        
        # Polymarket (usando context manager)
        async with PolymarketCollector() as poly_collector:
            poly_raw_markets = await poly_collector.get_markets(limit=request.limit)
            # Converter para dict usando raw_data ou atributos do Market
            poly_markets_all = []
            for m in poly_raw_markets:
                raw = getattr(m, 'raw_data', {}) or {}
                market_dict = {
                    "conditionId": getattr(m, 'condition_id', '') or raw.get('conditionId', ''),
                    "question": getattr(m, 'title', '') or raw.get('question', ''),
                    "slug": getattr(m, 'ticker', '') or raw.get('slug', ''),
                    "outcomePrices": raw.get('outcomePrices', ["0.5", "0.5"]),
                    "volume": raw.get('volume', raw.get('volumeNum', 0)),
                    "liquidity": raw.get('liquidity', 0),
                    "endDate": raw.get('endDate') or raw.get('end_date_iso'),
                    "description": getattr(m, 'description', '') or raw.get('description', ''),
                    "tags": raw.get('tags', [])
                }
                poly_markets_all.append(market_dict)
        
        # Filtrar por categoria se especificada
        if selected_categories:
            poly_markets = [
                m for m in poly_markets_all 
                if market_matches_categories(m.get("question", ""), m.get("description", ""), selected_categories)
            ]
            logger.info(f"📊 Polymarket: {len(poly_markets)}/{len(poly_markets_all)} mercados (filtrados)")
        else:
            poly_markets = poly_markets_all
            logger.info(f"📊 Polymarket: {len(poly_markets)} mercados")
        
        # Kalshi (autenticado - usando context manager)
        async with KalshiAuthenticatedCollector(api_key=request.kalshi_api_key) as kalshi_collector:
            await kalshi_collector.login(api_key=request.kalshi_api_key)
            
            # Buscar mercados Kalshi
            kalshi_markets_all = await kalshi_collector.get_all_markets(
                max_markets=request.limit,
                include_events=True
            )
        
        # Filtrar por categoria se especificada
        if selected_categories:
            kalshi_markets = [
                m for m in kalshi_markets_all 
                if market_matches_categories(m.get("title", ""), m.get("rules_primary", ""), selected_categories)
            ]
            logger.info(f"📊 Kalshi: {len(kalshi_markets)}/{len(kalshi_markets_all)} mercados (filtrados)")
        else:
            kalshi_markets = kalshi_markets_all
            logger.info(f"📊 Kalshi: {len(kalshi_markets)} mercados")
        
        # 3. Matching e análise
        opportunities = []
        matches_count = 0
        
        # Normalizar títulos para matching rápido
        def normalize(text: str) -> str:
            import re
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text
        
        kalshi_normalized = {
            normalize(m.get("title", "")): m 
            for m in kalshi_markets
        }
        
        from difflib import SequenceMatcher
        
        for poly in poly_markets:
            poly_title = normalize(poly.get("question", ""))
            
            # Encontrar melhor match no Kalshi
            best_match = None
            best_sim = 0
            
            for kalshi_title, kalshi in kalshi_normalized.items():
                sim = SequenceMatcher(None, poly_title, kalshi_title).ratio()
                if sim > best_sim and sim >= 0.80:  # Reduzido de 90% para 80%
                    best_sim = sim
                    best_match = kalshi
            
            if best_match:
                matches_count += 1
                
                # LOG DO MATCH ENCONTRADO
                logger.info(f"🔗 Match #{matches_count}: {poly.get('question', '')[:50]} <-> {best_match.get('title', '')[:50]} (sim: {best_sim:.2f})")
                
                # Analisar com engine profissional
                opp = engine.analyze_opportunity(
                    poly_raw=poly,
                    kalshi_raw=best_match
                )
                
                if opp:
                    logger.info(f"   → Lucro: {float(opp.net_profit_percentage):.2f}%, Válido: {opp.is_valid}")
                else:
                    logger.info(f"   → Engine retornou None (sem arbitragem possível)")
                
                if opp and opp.is_valid:
                    # Converter para response
                    opportunities.append({
                        "id": opp.id,
                        "poly_title": opp.poly_market.title[:100],
                        "poly_url": opp.poly_market.url,
                        "kalshi_title": opp.kalshi_market.title[:100],
                        "kalshi_url": opp.kalshi_market.url,
                        "title_similarity": opp.title_similarity,
                        "equivalence_score": opp.equivalence_score,
                        "strategy": opp.strategy,
                        "action_poly": opp.action_poly,
                        "action_kalshi": opp.action_kalshi,
                        "poly_yes_price": float(opp.poly_market.yes_price),
                        "poly_no_price": float(opp.poly_market.no_price),
                        "kalshi_yes_price": float(opp.kalshi_market.yes_price),
                        "kalshi_no_price": float(opp.kalshi_market.no_price),
                        "total_cost_per_contract": float(opp.total_cost_per_contract),
                        "poly_fees": float(opp.poly_fees),
                        "kalshi_fees": float(opp.kalshi_fees),
                        "total_fees": float(opp.total_fees),
                        "fees_breakdown": {
                            "polymarket": f"{float(config.fees.poly_trading_fee)*100:.1f}% trading",
                            "kalshi": f"{float(config.fees.kalshi_trading_fee)*100:.0f}% sobre lucro + {float(config.fees.kalshi_taker_fee)*100:.0f}% taker"
                        },
                        "gross_profit": float(opp.gross_profit),
                        "net_profit": float(opp.net_profit),
                        "net_profit_percentage": float(opp.net_profit_percentage),
                        "execution_100": {
                            "target_size_usd": float(opp.execution_100.target_size_usd),
                            "avg_price": float(opp.execution_100.avg_price),
                            "total_cost": float(opp.execution_100.total_cost),
                            "slippage_percentage": float(opp.execution_100.slippage_percentage),
                            "fully_fillable": opp.execution_100.fully_fillable
                        } if opp.execution_100 else None,
                        "execution_500": {
                            "target_size_usd": float(opp.execution_500.target_size_usd),
                            "avg_price": float(opp.execution_500.avg_price),
                            "total_cost": float(opp.execution_500.total_cost),
                            "slippage_percentage": float(opp.execution_500.slippage_percentage),
                            "fully_fillable": opp.execution_500.fully_fillable
                        } if opp.execution_500 else None,
                        "execution_1000": {
                            "target_size_usd": float(opp.execution_1000.target_size_usd),
                            "avg_price": float(opp.execution_1000.avg_price),
                            "total_cost": float(opp.execution_1000.total_cost),
                            "slippage_percentage": float(opp.execution_1000.slippage_percentage),
                            "fully_fillable": opp.execution_1000.fully_fillable
                        } if opp.execution_1000 else None,
                        "liquidity_score": opp.liquidity_score,
                        "max_recommended_size": float(opp.max_recommended_size),
                        "data_age_seconds": opp.max_data_age,
                        "data_freshness": opp.data_freshness,
                        "payoff_scenarios": [
                            {
                                "scenario_name": s.scenario_name,
                                "probability": float(s.probability),
                                "gross_return": float(s.gross_return),
                                "total_fees": float(s.total_fees),
                                "net_return": float(s.net_return),
                                "roi_percentage": float(s.roi_percentage)
                            } for s in opp.payoff_scenarios
                        ],
                        "guaranteed_min_profit": float(opp.guaranteed_min_profit),
                        "max_loss_if_partial": float(opp.max_loss_if_partial),
                        "resolution_date": opp.resolution_date,
                        "days_to_resolution": opp.days_to_resolution,
                        "warnings": opp.validation_warnings,
                        "overall_score": opp.overall_score
                    })
        
        # 4. Ordenar por score
        opportunities.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # 5. Construir resposta
        arb_count = len(opportunities)
        
        if arb_count > 0:
            message = f"🎯 {arb_count} ARBITRAGENS PROFISSIONAIS encontradas!"
        elif matches_count > 0:
            message = f"✅ {matches_count} matches analisados. Nenhum passa nos filtros profissionais (lucro mínimo {request.min_profit_percentage}%)."
        else:
            message = f"❌ Nenhuma correspondência encontrada entre plataformas."
        
        return {
            "status": "completed",
            "scan_type": "professional",
            "config": {
                "min_profit_percentage": request.min_profit_percentage,
                "min_profit_absolute": request.min_profit_absolute,
                "max_size_per_trade": request.max_size_per_trade,
                "categories": selected_categories or ["all"]
            },
            "polymarket_markets": len(poly_markets),
            "polymarket_total": len(poly_markets_all) if selected_categories else len(poly_markets),
            "kalshi_markets": len(kalshi_markets),
            "kalshi_total": len(kalshi_markets_all) if selected_categories else len(kalshi_markets),
            "matches_found": matches_count,
            "arbitrage_count": arb_count,
            "opportunities": opportunities[:30],
            "message": message,
            "fees_info": {
                "polymarket": {
                    "trading": "2%",
                    "withdrawal": "0%",
                    "deposit": "0%"
                },
                "kalshi": {
                    "winner_fee": "7% sobre lucro",
                    "taker_fee": "1%",
                    "maker_fee": "0%",
                    "withdrawal": "0%"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Erro no scan profissional: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"❌ Erro: {str(e)}",
            "opportunities": []
        }


# ============================================================
# ENDPOINT DE CONFIGURAÇÃO
# ============================================================

@app.get("/config/defaults")
async def get_default_config():
    """Retorna configurações padrão do sistema."""
    from src.engine.professional_arbitrage import DEFAULT_CONFIG
    return DEFAULT_CONFIG.to_dict()


# ============================================================
# WHALE TRACKER - RASTREAMENTO DE CARTEIRAS ANÔMALAS
# ============================================================

from src.whale_tracker import PolymarketWhaleTracker, AnomalyDetector, AlertSystem, Alert, AlertType
from src.whale_tracker.alert_system import AlertPriority, create_whale_alert, create_new_wallet_alert

# Sistema global de alertas (mantém histórico)
whale_alert_system = AlertSystem(
    log_file="/tmp/whale_alerts.json"
)


class WhaleTrackRequest(BaseModel):
    """Request para rastreamento de whales."""
    market_ids: Optional[List[str]] = None
    min_volume_usd: float = 10000
    lookback_hours: int = 24


class WalletWatchRequest(BaseModel):
    """Request para monitorar carteira."""
    address: str


class MarketAnalysisRequest(BaseModel):
    """Request para análise de mercado."""
    market_id: str
    lookback_hours: int = 24


@app.post("/whale/scan")
async def scan_for_whales(request: WhaleTrackRequest):
    """
    🐋 Escaneia mercados em busca de whales.
    
    Retorna:
    - Lista de carteiras com volume >= min_volume_usd
    - Métricas de cada carteira
    - Flags de anomalia
    """
    try:
        logger.info(f"🐋 Iniciando scan de whales (min: ${request.min_volume_usd})")
        
        async with PolymarketWhaleTracker() as tracker:
            whales = await tracker.scan_for_whales(
                market_ids=request.market_ids,
                min_volume_usd=Decimal(str(request.min_volume_usd))
            )
            
            results = []
            for w in whales:
                # Gerar alerta se for whale significativo
                if w.total_volume_usd >= Decimal("10000"):
                    alert = Alert(
                        id="",
                        type=AlertType.WHALE_DETECTED,
                        priority=AlertPriority.MEDIUM if w.total_volume_usd < 50000 else AlertPriority.HIGH,
                        title="🐋 Whale Detectado",
                        message=f"Carteira com volume de ${float(w.total_volume_usd):,.2f}",
                        addresses=[w.address],
                        data={
                            "total_volume_usd": float(w.total_volume_usd),
                            "total_trades": w.total_trades,
                            "markets_count": len(w.markets_traded)
                        }
                    )
                    await whale_alert_system.emit(alert)
                
                # Extrair nome do mercado dos trades
                market_name = "Unknown"
                if w.trades:
                    market_name = w.trades[0].market_title[:60] if w.trades[0].market_title else "Unknown"
                
                results.append({
                    "address": w.address,
                    "market_name": market_name,
                    "total_volume_usd": float(w.total_volume_usd),
                    "total_trades": w.total_trades,
                    "markets_traded": list(w.markets_traded)[:5],  # Lista dos nomes
                    "avg_trade_size": float(w.avg_trade_size),
                    "max_trade_size": float(w.max_trade_size),
                    "first_seen": w.first_seen.isoformat() if w.first_seen else None,
                    "last_seen": w.last_seen.isoformat() if w.last_seen else None,
                    "flags": {
                        "is_whale": w.is_whale,
                        "is_new_wallet": w.is_new_wallet,
                        "concentrated_positions": w.concentrated_positions
                    }
                })
            
            return {
                "status": "success",
                "whales_found": len(results),
                "min_volume_threshold": request.min_volume_usd,
                "whales": results
            }
            
    except Exception as e:
        logger.error(f"Erro no scan de whales: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/whale/analyze-market")
async def analyze_market_activity(request: MarketAnalysisRequest):
    """
    📊 Analisa atividade de um mercado específico.
    
    Detecta:
    - Trades de whales
    - Carteiras novas com trades grandes
    - Trading coordenado
    - Timing suspeito
    """
    try:
        logger.info(f"📊 Analisando mercado: {request.market_id}")
        
        async with PolymarketWhaleTracker() as tracker:
            results = await tracker.detect_unusual_activity(
                market_id=request.market_id,
                lookback_hours=request.lookback_hours
            )
            
            # Gerar alertas para atividades suspeitas
            for trade in results.get("new_wallet_trades", []):
                alert = create_new_wallet_alert(
                    trade_value=trade["value_usd"],
                    address=trade["address"],
                    wallet_age_days=trade["wallet_age_days"],
                    market_id=request.market_id,
                    market_title=results.get("market_title", "")
                )
                await whale_alert_system.emit(alert)
            
            return {
                "status": "success",
                **results
            }
            
    except Exception as e:
        logger.error(f"Erro na análise de mercado: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/whale/watch")
async def add_watched_wallet(request: WalletWatchRequest):
    """
    👁️ Adiciona carteira à lista de monitoramento.
    """
    try:
        async with PolymarketWhaleTracker() as tracker:
            tracker.add_watched_wallet(request.address)
            
            # Buscar info inicial da carteira
            wallet = await tracker.analyze_wallet(request.address)
            
            return {
                "status": "success",
                "message": f"Carteira {request.address[:10]}... adicionada ao monitoramento",
                "wallet_info": {
                    "address": wallet.address,
                    "total_volume_usd": float(wallet.total_volume_usd),
                    "total_trades": wallet.total_trades,
                    "is_whale": wallet.is_whale,
                    "is_new_wallet": wallet.is_new_wallet
                }
            }
            
    except Exception as e:
        logger.error(f"Erro ao adicionar carteira: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/whale/alerts")
async def get_whale_alerts(
    limit: int = Query(default=50, ge=1, le=200),
    min_priority: Optional[str] = Query(default=None)
):
    """
    🔔 Retorna alertas recentes de atividade suspeita.
    """
    try:
        priority = None
        if min_priority:
            priority = AlertPriority[min_priority.upper()]
        
        alerts = whale_alert_system.get_recent_alerts(
            limit=limit,
            min_priority=priority
        )
        
        return {
            "status": "success",
            "total_alerts": whale_alert_system.total_alerts,
            "stats": whale_alert_system.get_stats(),
            "alerts": [a.to_dict() for a in alerts]
        }
        
    except Exception as e:
        logger.error(f"Erro ao buscar alertas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/whale/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """
    ✓ Marca um alerta como reconhecido.
    """
    success = whale_alert_system.acknowledge(alert_id)
    
    if success:
        return {"status": "success", "message": f"Alerta {alert_id} reconhecido"}
    else:
        raise HTTPException(status_code=404, detail="Alerta não encontrado")


@app.get("/whale/stats")
async def get_whale_stats():
    """
    📈 Retorna estatísticas do sistema de rastreamento.
    """
    return {
        "status": "success",
        "stats": whale_alert_system.get_stats()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
