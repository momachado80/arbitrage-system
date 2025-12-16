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
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.collectors import KalshiCollector, PolymarketCollector
from src.engine import ArbitrageCalculator, ExecutionSimulator
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
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
