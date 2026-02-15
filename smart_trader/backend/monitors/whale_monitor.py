"""
Monitor de Whale Movement
==========================
Integra com Etherscan API + Polygon RPC para detectar
movimentações grandes de capital que podem indicar insider trading.
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from models.signal import WhaleActivity
except ImportError:
    from ..models.signal import WhaleActivity

logger = logging.getLogger(__name__)


# Contratos conhecidos do Polymarket (Polygon)
POLYMARKET_CONTRACTS = {
    # USDC no Polygon
    "usdc": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    # Polymarket CTF Exchange
    "ctf_exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    # Polymarket Neg Risk CTF Exchange
    "neg_risk_exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    # Conditional Tokens
    "conditional_tokens": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
}

# Limites
WHALE_THRESHOLD_USD = Decimal("100000")  # $100k para considerar whale
LARGE_TRADE_USD = Decimal("50000")  # $50k para trade grande
ALERT_THRESHOLD_USD = Decimal("500000")  # $500k em 30 min = alerta


@dataclass
class Transaction:
    """Representa uma transação detectada"""
    hash: str
    from_address: str
    to_address: str
    value: Decimal
    token: str
    timestamp: datetime
    block_number: int
    contract_address: str = ""
    is_polymarket: bool = False
    gas_price: Optional[Decimal] = None


@dataclass
class WalletProfile:
    """Perfil de uma carteira monitorada"""
    address: str
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_volume: Decimal = Decimal("0")
    transaction_count: int = 0
    is_whale: bool = False
    is_smart_money: bool = False  # Histórico de trades lucrativos
    polymarket_trades: int = 0
    avg_trade_size: Decimal = Decimal("0")
    win_rate: Optional[float] = None
    tags: List[str] = field(default_factory=list)


class WhaleMonitor:
    """
    Monitor de movimentação de whales em tempo real.
    
    Detecta:
    - Transferências grandes de USDC/DAI
    - Interações com contratos do Polymarket
    - Padrões de timing suspeitos
    - Carteiras novas com volume alto
    """
    
    CHECK_INTERVAL = 15  # Segundos entre verificações
    
    def __init__(
        self,
        etherscan_key: Optional[str] = None,
        polygonscan_key: Optional[str] = None,
        check_interval: int = 15
    ):
        self.etherscan_key = etherscan_key
        self.polygonscan_key = polygonscan_key
        self.check_interval = check_interval
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache
        self.recent_transactions: List[Transaction] = []
        self.wallet_profiles: Dict[str, WalletProfile] = {}
        self.known_whales: Set[str] = set()
        self.volume_by_window: Dict[str, Decimal] = defaultdict(Decimal)  # Por 30 min
        
        # Callbacks
        self.callbacks: List[callable] = []
        
        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_block: int = 0
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=connector,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SmartEventTrader/1.0)",
                "Accept": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def on_whale_activity(self, callback: callable):
        """Registra callback para atividade de whale"""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self, activity: WhaleActivity):
        """Notifica callbacks sobre atividade de whale"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(activity)
                else:
                    callback(activity)
            except Exception as e:
                logger.error(f"Erro no callback: {e}")
    
    # =========================================================================
    # POLYGONSCAN API
    # =========================================================================
    
    async def fetch_polygon_transactions(
        self,
        address: Optional[str] = None,
        start_block: int = 0
    ) -> List[Transaction]:
        """
        Busca transações do Polygon via PolygonScan API.
        Free tier: 5 calls/segundo
        """
        transactions = []
        
        try:
            # API base
            base_url = "https://api.polygonscan.com/api"
            
            # Buscar transações de tokens (USDC)
            params = {
                "module": "account",
                "action": "tokentx",
                "contractaddress": POLYMARKET_CONTRACTS["usdc"],
                "page": 1,
                "offset": 100,
                "sort": "desc"
            }
            
            if address:
                params["address"] = address
            if start_block:
                params["startblock"] = start_block
            if self.polygonscan_key:
                params["apikey"] = self.polygonscan_key
            
            async with self.session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "1" and data.get("result"):
                        for tx in data["result"]:
                            try:
                                value = Decimal(tx.get("value", "0")) / Decimal("1000000")  # USDC = 6 decimals
                                
                                if value >= LARGE_TRADE_USD:
                                    transaction = Transaction(
                                        hash=tx.get("hash", ""),
                                        from_address=tx.get("from", "").lower(),
                                        to_address=tx.get("to", "").lower(),
                                        value=value,
                                        token="USDC",
                                        timestamp=datetime.fromtimestamp(
                                            int(tx.get("timeStamp", 0)),
                                            tz=timezone.utc
                                        ),
                                        block_number=int(tx.get("blockNumber", 0)),
                                        contract_address=tx.get("contractAddress", "").lower(),
                                        is_polymarket=self._is_polymarket_related(tx)
                                    )
                                    transactions.append(transaction)
                            except Exception as e:
                                logger.debug(f"Erro ao parsear tx: {e}")
                                continue
                else:
                    logger.warning(f"PolygonScan retornou {response.status}")
        
        except Exception as e:
            logger.error(f"Erro ao buscar transações Polygon: {e}")
        
        return transactions
    
    def _is_polymarket_related(self, tx: Dict[str, Any]) -> bool:
        """Verifica se transação está relacionada ao Polymarket"""
        to_addr = tx.get("to", "").lower()
        from_addr = tx.get("from", "").lower()
        
        polymarket_addrs = [addr.lower() for addr in POLYMARKET_CONTRACTS.values()]
        
        return to_addr in polymarket_addrs or from_addr in polymarket_addrs
    
    # =========================================================================
    # ANÁLISE DE WHALES
    # =========================================================================
    
    def _update_wallet_profile(self, tx: Transaction):
        """Atualiza perfil de carteira baseado em transação"""
        for addr in [tx.from_address, tx.to_address]:
            if addr not in self.wallet_profiles:
                self.wallet_profiles[addr] = WalletProfile(
                    address=addr,
                    first_seen=tx.timestamp
                )
            
            profile = self.wallet_profiles[addr]
            profile.last_activity = max(profile.last_activity, tx.timestamp)
            profile.total_volume += tx.value
            profile.transaction_count += 1
            
            if tx.is_polymarket:
                profile.polymarket_trades += 1
            
            # Atualiza média
            profile.avg_trade_size = profile.total_volume / profile.transaction_count
            
            # Marca como whale se volume alto
            if profile.total_volume >= WHALE_THRESHOLD_USD:
                profile.is_whale = True
                self.known_whales.add(addr)
    
    def _aggregate_whale_activity(
        self,
        transactions: List[Transaction],
        window_minutes: int = 30
    ) -> Optional[WhaleActivity]:
        """
        Agrega transações em uma atividade de whale.
        """
        if not transactions:
            return None
        
        # Filtra transações na janela de tempo
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=window_minutes)
        recent = [tx for tx in transactions if tx.timestamp >= cutoff]
        
        if not recent:
            return None
        
        # Agrupa por carteira
        wallets: Dict[str, List[Transaction]] = defaultdict(list)
        for tx in recent:
            wallets[tx.from_address].append(tx)
        
        # Calcula totais
        total_volume = sum(tx.value for tx in recent)
        whale_wallets = [
            addr for addr, txs in wallets.items()
            if sum(tx.value for tx in txs) >= LARGE_TRADE_USD
        ]
        
        if total_volume < ALERT_THRESHOLD_USD and len(whale_wallets) < 3:
            return None  # Não é significativo
        
        # Determina ação (comprando ou vendendo?)
        to_polymarket = sum(
            tx.value for tx in recent 
            if tx.to_address.lower() in [c.lower() for c in POLYMARKET_CONTRACTS.values()]
        )
        from_polymarket = sum(
            tx.value for tx in recent
            if tx.from_address.lower() in [c.lower() for c in POLYMARKET_CONTRACTS.values()]
        )
        
        if to_polymarket > from_polymarket:
            acao = "COMPRANDO"
        elif from_polymarket > to_polymarket:
            acao = "VENDENDO"
        else:
            acao = "MOVIMENTANDO"
        
        # Cria atividade
        activity = WhaleActivity(
            whales_detectados=len(whale_wallets),
            volume_total_usd=total_volume,
            timestamp_movimento=min(tx.timestamp for tx in recent),
            carteiras=whale_wallets[:10],  # Limita a 10
            acao_whale=acao,
            transacoes=[
                {
                    "hash": tx.hash,
                    "from": tx.from_address,
                    "to": tx.to_address,
                    "value": float(tx.value),
                    "timestamp": tx.timestamp.isoformat()
                }
                for tx in recent[:20]  # Limita a 20
            ]
        )
        
        # Identifica destino principal
        dest_counts: Dict[str, Decimal] = defaultdict(Decimal)
        for tx in recent:
            dest_counts[tx.to_address] += tx.value
        
        if dest_counts:
            activity.destino_contrato = max(dest_counts, key=dest_counts.get)
        
        return activity
    
    # =========================================================================
    # DETECÇÃO DE ANOMALIAS
    # =========================================================================
    
    async def detect_anomalies(
        self,
        transactions: List[Transaction]
    ) -> List[Dict[str, Any]]:
        """
        Detecta anomalias nas transações:
        - Volume muito acima do normal
        - Carteiras novas com trades grandes
        - Múltiplas carteiras agindo coordenadamente
        """
        anomalies = []
        
        # Anomalia 1: Volume alto em curto período
        activity = self._aggregate_whale_activity(transactions, window_minutes=30)
        if activity and activity.volume_total_usd >= ALERT_THRESHOLD_USD:
            anomalies.append({
                "type": "HIGH_VOLUME",
                "description": f"${float(activity.volume_total_usd):,.0f} em 30 min",
                "severity": "HIGH",
                "activity": activity
            })
        
        # Anomalia 2: Carteiras novas com trades grandes
        now = datetime.now(timezone.utc)
        for tx in transactions:
            profile = self.wallet_profiles.get(tx.from_address)
            if profile:
                wallet_age = (now - profile.first_seen).days
                if wallet_age <= 7 and tx.value >= LARGE_TRADE_USD:
                    anomalies.append({
                        "type": "NEW_WALLET_LARGE_TRADE",
                        "description": f"Carteira de {wallet_age} dias com trade de ${float(tx.value):,.0f}",
                        "severity": "CRITICAL",
                        "wallet": tx.from_address,
                        "value": float(tx.value)
                    })
        
        # Anomalia 3: Coordenação (múltiplas carteiras ao mesmo tempo)
        if len(set(tx.from_address for tx in transactions)) >= 5:
            # 5+ carteiras diferentes em curto período
            anomalies.append({
                "type": "COORDINATED_ACTIVITY",
                "description": f"{len(set(tx.from_address for tx in transactions))} carteiras diferentes",
                "severity": "HIGH",
                "wallets": list(set(tx.from_address for tx in transactions))[:10]
            })
        
        return anomalies
    
    # =========================================================================
    # CORRELAÇÃO TEMPORAL
    # =========================================================================
    
    def calculate_temporal_correlation(
        self,
        whale_timestamp: datetime,
        news_timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Calcula correlação temporal entre movimento de whale e notícia.
        
        Returns:
            {
                "minutes_before": int,  # Negativo = depois da notícia
                "score": float,  # 0-30 pontos
                "classification": str  # "INSIDER", "EARLY", "NORMAL"
            }
        """
        diff = news_timestamp - whale_timestamp
        minutes = diff.total_seconds() / 60
        
        if minutes >= 10 and minutes <= 30:
            # 10-30 min ANTES da notícia = forte indicador de insider
            return {
                "minutes_before": int(minutes),
                "score": 30,
                "classification": "INSIDER"
            }
        elif minutes > 30 and minutes <= 60:
            # 30-60 min antes
            return {
                "minutes_before": int(minutes),
                "score": 20,
                "classification": "EARLY"
            }
        elif minutes > 60 and minutes <= 120:
            # 1-2 horas antes
            return {
                "minutes_before": int(minutes),
                "score": 10,
                "classification": "EARLY"
            }
        elif minutes < 0:
            # Whale se moveu DEPOIS da notícia (normal, não é insider)
            return {
                "minutes_before": int(minutes),
                "score": -20,
                "classification": "NORMAL"
            }
        else:
            # Muito antes (> 2 horas) - não correlacionado
            return {
                "minutes_before": int(minutes),
                "score": 0,
                "classification": "UNCORRELATED"
            }
    
    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================
    
    async def scan_once(self) -> Optional[WhaleActivity]:
        """Executa um scan de transações"""
        # Busca transações recentes
        transactions = await self.fetch_polygon_transactions(
            start_block=self._last_block
        )
        
        if transactions:
            # Atualiza último bloco
            self._last_block = max(tx.block_number for tx in transactions)
            
            # Atualiza perfis de carteiras
            for tx in transactions:
                self._update_wallet_profile(tx)
            
            # Adiciona ao cache
            self.recent_transactions.extend(transactions)
            
            # Limpa transações antigas (> 2 horas)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
            self.recent_transactions = [
                tx for tx in self.recent_transactions
                if tx.timestamp >= cutoff
            ]
        
        logger.info(f"🐋 Encontradas {len(transactions)} transações grandes")
        
        # Agrega atividade
        activity = self._aggregate_whale_activity(self.recent_transactions)
        
        if activity:
            # Detecta anomalias
            anomalies = await self.detect_anomalies(transactions)
            
            if anomalies:
                logger.info(f"⚠️ {len(anomalies)} anomalias detectadas!")
                await self._notify_callbacks(activity)
        
        return activity
    
    async def start(self):
        """Inicia monitoramento contínuo"""
        self._running = True
        logger.info(f"🐋 Whale Monitor iniciado (intervalo: {self.check_interval}s)")
        
        while self._running:
            try:
                await self.scan_once()
            except Exception as e:
                logger.error(f"Erro no scan de whales: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def stop(self):
        """Para monitoramento"""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("🐋 Whale Monitor parado")
    
    def run_background(self):
        """Inicia em background"""
        self._task = asyncio.create_task(self.start())
        return self._task
    
    # =========================================================================
    # MÉTODOS PÚBLICOS
    # =========================================================================
    
    def get_recent_activity(
        self,
        minutes: int = 60
    ) -> Optional[WhaleActivity]:
        """Retorna atividade de whales recente"""
        return self._aggregate_whale_activity(
            self.recent_transactions,
            window_minutes=minutes
        )
    
    def get_whale_wallets(self) -> List[str]:
        """Retorna lista de carteiras whale conhecidas"""
        return list(self.known_whales)
    
    def get_wallet_profile(self, address: str) -> Optional[WalletProfile]:
        """Retorna perfil de uma carteira"""
        return self.wallet_profiles.get(address.lower())


# Teste
async def test_whale_monitor():
    """Testa o monitor de whales"""
    async with WhaleMonitor() as monitor:
        activity = await monitor.scan_once()
        
        if activity:
            print(f"\n🐋 Atividade de whales detectada:")
            print(f"   Volume: ${float(activity.volume_total_usd):,.0f}")
            print(f"   Whales: {activity.whales_detectados}")
            print(f"   Ação: {activity.acao_whale}")
        else:
            print("\n🐋 Nenhuma atividade significativa detectada")


if __name__ == "__main__":
    asyncio.run(test_whale_monitor())

