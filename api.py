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
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.collectors import KalshiCollector, PolymarketCollector
from src.engine import ArbitrageCalculator, ExecutionSimulator, AutoScanner
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

# Cache do último scan
last_scan_result = None
scan_in_progress = False


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
    """Health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
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
    max_markets: int = Field(default=2000, ge=100, le=10000)


class FullScanResponse(BaseModel):
    status: str
    timestamp: str
    kalshi_total: int
    polymarket_total: int
    entities_matched: int
    pairs_analyzed: int
    pairs_with_liquidity: int
    opportunities_found: int
    opportunities: List[dict]
    top_pairs: List[dict]
    scan_duration_seconds: float
    entity_stats: dict


full_scan_in_progress = False
last_full_scan_result = None


@app.post("/scan/full", response_model=FullScanResponse)
async def run_full_scan(config: FullScanConfig = None):
    """
    Executa scan COMPLETO de todos os mercados.
    
    Este scan:
    - Busca até max_markets de cada plataforma
    - Matching por ENTIDADES ESPECÍFICAS (pessoas, empresas, eventos)
    - Só pareia mercados com entidades em comum
    - Retorna oportunidades E top pares similares
    
    ATENÇÃO: Pode demorar vários minutos!
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
            entities_matched=result.entities_matched,
            pairs_analyzed=result.pairs_analyzed,
            pairs_with_liquidity=result.pairs_with_liquidity,
            opportunities_found=result.opportunities_found,
            opportunities=result.opportunities,
            top_pairs=result.top_pairs,
            scan_duration_seconds=result.scan_duration_seconds,
            entity_stats=result.entity_stats,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
