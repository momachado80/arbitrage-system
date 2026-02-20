"""
MarketUniverseManager — Auto-Discovery de mercados Polymarket
==============================================================

Responsabilidades:
- Buscar mercados ativos via REST (Gamma API)
- Filtrar por liquidez, volume, spread
- Ordenar por liquidez
- Selecionar top N (configurável, default 50)
- Atualizar a cada 5 minutos
- Assinar/desassinar automaticamente no WebSocket

Se POLYMARKET_TOKENS vazio → auto-discovery.
Se preenchido → manual override.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
DEFAULT_TOP_N = 50
DEFAULT_UPDATE_INTERVAL_SEC = 300  # 5 min
DEFAULT_MIN_VOLUME_USD = 1000.0
DEFAULT_MIN_LIQUIDITY_USD = 500.0
DEFAULT_MAX_SPREAD_PCT = 10.0  # 10% max spread


@dataclass
class MarketInfo:
    """Info mínima para subscription."""
    token_id: str
    condition_id: str
    volume_24h: float
    liquidity: float
    spread_pct: float
    side: str  # "yes" or "no"


@dataclass
class UniverseSnapshot:
    """Snapshot da última atualização."""
    total_markets_found: int = 0
    total_markets_filtered: int = 0
    total_markets_subscribed: int = 0
    last_update_timestamp: Optional[float] = None
    token_ids: List[str] = field(default_factory=list)


class MarketUniverseManager:
    """
    Gerencia universo de mercados com auto-discovery.
    Thread-safe.
    """

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N,
        update_interval_sec: int = DEFAULT_UPDATE_INTERVAL_SEC,
        min_volume_usd: float = DEFAULT_MIN_VOLUME_USD,
        min_liquidity_usd: float = DEFAULT_MIN_LIQUIDITY_USD,
        max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
        subscribe_cb: Optional[Callable[[str], Any]] = None,
        unsubscribe_cb: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._top_n = int(os.environ.get("UNIVERSE_TOP_N", top_n))
        self._interval = int(os.environ.get("UNIVERSE_UPDATE_INTERVAL", update_interval_sec))
        self._min_volume = float(os.environ.get("UNIVERSE_MIN_VOLUME", min_volume_usd))
        self._min_liquidity = float(os.environ.get("UNIVERSE_MIN_LIQUIDITY", min_liquidity_usd))
        self._max_spread = float(os.environ.get("UNIVERSE_MAX_SPREAD", max_spread_pct))
        self._subscribe_cb = subscribe_cb
        self._unsubscribe_cb = unsubscribe_cb
        self._lock = threading.Lock()
        self._snapshot = UniverseSnapshot()
        self._current_tokens: Set[str] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._timeout = float(os.environ.get("UNIVERSE_HTTP_TIMEOUT", "10.0"))

    def get_snapshot(self) -> UniverseSnapshot:
        with self._lock:
            return UniverseSnapshot(
                total_markets_found=self._snapshot.total_markets_found,
                total_markets_filtered=self._snapshot.total_markets_filtered,
                total_markets_subscribed=self._snapshot.total_markets_subscribed,
                last_update_timestamp=self._snapshot.last_update_timestamp,
                token_ids=list(self._snapshot.token_ids),
            )

    def _fetch_markets_sync(self) -> List[Dict[str, Any]]:
        """Busca mercados da Gamma API (sync)."""
        params = {
            "limit": 200,
            "offset": 0,
            "closed": "false",
            "archived": "false",
            "active": "true",
            "order": "volume24hr",
            "ascending": "false",
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.get(f"{GAMMA_URL}/markets", params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            logger.error("[UNIVERSE] [FETCH_FAILED] %s", str(e))
            return []
        raw = data if isinstance(data, list) else data.get("markets", data.get("data", []))
        return raw if isinstance(raw, list) else []

    def _extract_tokens(self, m: Dict) -> tuple:
        """Retorna (yes_token_id, no_token_id) do market raw."""
        tokens = m.get("tokens", [])
        yes_id = no_id = None
        for t in tokens:
            out = (t.get("outcome") or "").upper()
            tid = t.get("token_id") or t.get("tokenId")
            if out == "YES" and tid:
                yes_id = tid
            elif out == "NO" and tid:
                no_id = tid
        if not yes_id or not no_id:
            clob = m.get("clobTokenIds")
            if isinstance(clob, str):
                try:
                    import json
                    clob = json.loads(clob)
                except Exception:
                    clob = None
            if isinstance(clob, list):
                if len(clob) >= 1 and not yes_id:
                    yes_id = clob[0] if isinstance(clob[0], str) else None
                if len(clob) >= 2 and not no_id:
                    no_id = clob[1] if isinstance(clob[1], str) else None
        return (yes_id, no_id)

    def _get_spread_for_token(self, token_id: str) -> float:
        """Obtém spread percentual do book (sync). Retorna 999 se falhar."""
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(f"{CLOB_URL}/book", params={"token_id": token_id})
                r.raise_for_status()
                data = r.json()
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            if not bids or not asks:
                return 999.0
            best_bid = float(bids[0].get("price", 0)) if bids else 0
            best_ask = float(asks[0].get("price", 1)) if asks else 1
            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.5
            if mid <= 0:
                return 999.0
            spread_pct = ((best_ask - best_bid) / mid) * 100
            return spread_pct
        except Exception:
            return 999.0

    def _estimate_liquidity(self, m: Dict) -> float:
        """Estima liquidez a partir de volume (proxy)."""
        vol = float(m.get("volume24hr") or m.get("volume24h") or m.get("volume") or 0)
        return vol * 0.1  # 10% do volume como proxy de liquidez

    def _discover_from_raw(self, raw: List[Dict[str, Any]]) -> List[MarketInfo]:
        """Filtra raw markets e retorna lista ordenada por liquidez."""
        found = len(raw)
        candidates: List[MarketInfo] = []
        skip_spread = os.environ.get("UNIVERSE_SKIP_SPREAD_CHECK", "").upper() in ("1", "TRUE", "YES")
        for m in raw:
            vol = float(m.get("volume24hr") or m.get("volume24h") or m.get("volume") or 0)
            if vol < self._min_volume:
                continue
            yes_id, no_id = self._extract_tokens(m)
            if not yes_id:
                continue
            liq = self._estimate_liquidity(m)
            if liq < self._min_liquidity:
                continue
            spread = 0.0 if skip_spread else self._get_spread_for_token(yes_id)
            if not skip_spread and spread > self._max_spread:
                continue
            cond = m.get("conditionId") or m.get("condition_id") or ""
            candidates.append(MarketInfo(
                token_id=yes_id,
                condition_id=cond,
                volume_24h=vol,
                liquidity=liq,
                spread_pct=spread,
                side="yes",
            ))
            if no_id:
                candidates.append(MarketInfo(
                    token_id=no_id,
                    condition_id=cond,
                    volume_24h=vol,
                    liquidity=liq,
                    spread_pct=spread,
                    side="no",
                ))
        candidates.sort(key=lambda x: (x.liquidity, x.volume_24h), reverse=True)
        top = candidates[: self._top_n * 2]
        token_seen: Set[str] = set()
        result: List[MarketInfo] = []
        for c in top:
            if c.token_id in token_seen:
                continue
            token_seen.add(c.token_id)
            result.append(c)
            if len(result) >= self._top_n:
                break
        return result

    def discover_and_filter(self) -> List[MarketInfo]:
        """Descobre mercados, filtra e retorna lista ordenada por liquidez."""
        raw = self._fetch_markets_sync()
        return self._discover_from_raw(raw)

    def _run_update(self) -> None:
        """Executa uma rodada de atualização."""
        try:
            raw = self._fetch_markets_sync()
            infos = self._discover_from_raw(raw)
            filtered = len(infos)
            new_tokens = {m.token_id for m in infos}
            with self._lock:
                old_tokens = set(self._current_tokens)
                to_unsub = old_tokens - new_tokens
                to_sub = new_tokens - old_tokens
                self._current_tokens = new_tokens
                self._snapshot = UniverseSnapshot(
                    total_markets_found=len(raw),
                    total_markets_filtered=filtered,
                    total_markets_subscribed=len(new_tokens),
                    last_update_timestamp=time.time(),
                    token_ids=list(new_tokens),
                )
            for tid in to_unsub:
                if self._unsubscribe_cb:
                    try:
                        self._unsubscribe_cb(tid)
                    except Exception as e:
                        logger.warning("[UNIVERSE] [UNSUBSCRIBE_FAIL] token=%s err=%s", tid, e)
            for tid in to_sub:
                if self._subscribe_cb:
                    try:
                        self._subscribe_cb(tid)
                    except Exception as e:
                        logger.warning("[UNIVERSE] [SUBSCRIBE_FAIL] token=%s err=%s", tid, e)
            logger.info(
                "[UNIVERSE] [UPDATED] subscribed=%d found=%d filtered=%d",
                len(new_tokens),
                self._snapshot.total_markets_found,
                filtered,
            )
        except Exception as e:
            logger.error("[UNIVERSE] [UPDATE_ERROR] %s", str(e), exc_info=True)

    def _loop(self) -> None:
        while self._running:
            self._run_update()
            for _ in range(self._interval):
                if not self._running:
                    return
                time.sleep(1)

    def start(self, ws_subscribe_sync: Optional[Callable[[str], None]] = None) -> None:
        """Inicia thread de atualização periódica."""
        if self._running:
            return
        cb = ws_subscribe_sync or self._subscribe_cb
        if cb:
            self._subscribe_cb = cb
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="UniverseManager", daemon=True)
        self._thread.start()
        logger.info("[UNIVERSE] [STARTED] interval=%ds top_n=%d", self._interval, self._top_n)

    def stop(self) -> None:
        """Para a thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def run_once(self) -> List[str]:
        """Executa uma única atualização e retorna token_ids. Para uso síncrono inicial."""
        raw = self._fetch_markets_sync()
        infos = self._discover_from_raw(raw)
        token_ids = list({m.token_id for m in infos})
        with self._lock:
            self._current_tokens = set(token_ids)
            self._snapshot = UniverseSnapshot(
                total_markets_found=len(raw),
                total_markets_filtered=len(infos),
                total_markets_subscribed=len(token_ids),
                last_update_timestamp=time.time(),
                token_ids=token_ids,
            )
        logger.info(
            "[UNIVERSE] [RUN_ONCE] subscribed=%d found=%d filtered=%d",
            len(token_ids),
            self._snapshot.total_markets_found,
            len(infos),
        )
        return token_ids
