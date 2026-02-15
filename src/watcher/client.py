"""
Polymarket CLOB WebSocket Client
==================================

Cliente WebSocket resiliente para o CLOB da Polymarket.

FEATURES:
- Reconexão automática com exponential backoff (1s → 30s max)
- Heartbeat via aiohttp (ping se sem tráfego por 20s)
- Espelho local do order book via LocalOrderBook
- Trades em tempo real via WatcherBridge → Queue
- Shutdown limpo sem leaks

DEPENDÊNCIAS:
- aiohttp (WebSocket)
- orjson  (parsing JSON — obrigatório, NÃO usar json padrão)

RESTRIÇÕES:
- Não bloquear event loop
- Não usar chamadas síncronas
- Nunca usar queue.put() dentro de async — apenas put_nowait()
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set

import aiohttp
import orjson

from src.watcher.order_book import LocalOrderBook, BookCorruptionError
from src.watcher.bridge import WatcherBridge

logger = logging.getLogger(__name__)

# Polymarket CLOB WebSocket
WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Exponential backoff config
BACKOFF_INITIAL: float = 1.0
BACKOFF_MAX: float = 30.0
BACKOFF_MULTIPLIER: float = 2.0

# Heartbeat: aiohttp envia ping automático se sem tráfego
HEARTBEAT_TIMEOUT: float = 20.0


class PolymarketClient:
    """
    Cliente WebSocket resiliente para Polymarket CLOB.

    Uso:
    ```python
    bridge = WatcherBridge(queue.Queue(maxsize=100))
    client = PolymarketClient(bridge)
    await client.subscribe_market("0x...token_id")
    await client.run_forever()
    ```
    """

    def __init__(
        self,
        bridge: WatcherBridge,
        ws_url: str = WS_URL,
    ) -> None:
        self._bridge = bridge
        self._ws_url = ws_url
        self._books: Dict[str, LocalOrderBook] = {}
        self._subscribed_tokens: Set[str] = set()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running: bool = False
        self._tasks: List[asyncio.Task] = []  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """
        Loop infinito resiliente.

        Reconexão automática com exponential backoff:
        1s, 2s, 4s, 8s... max 30s.
        Reset após conexão bem-sucedida.
        """
        self._running = True
        backoff = BACKOFF_INITIAL

        while self._running:
            try:
                await self._connect_and_process()
                # Conexão encerrou normalmente → reset backoff
                backoff = BACKOFF_INITIAL
            except BookCorruptionError as e:
                logger.warning(f"[WATCHER:BOOK_CORRUPT] {e} — reconectando")
            except (
                aiohttp.WSServerHandshakeError,
                aiohttp.ClientError,
                OSError,
            ) as e:
                logger.warning(f"[WATCHER:DISCONNECT] {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WATCHER:ERROR] {type(e).__name__}: {e}")

            if not self._running:
                break

            logger.info(f"[WATCHER:RECONNECT] aguardando {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX)

    async def _connect_and_process(self) -> None:
        """Single connection lifecycle: connect → subscribe → process."""
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                self._ws_url,
                heartbeat=HEARTBEAT_TIMEOUT,
            )
            logger.info(f"[WATCHER:CONNECT] {self._ws_url}")

            # Re-subscribe all tokens on reconnect
            for token_id in self._subscribed_tokens:
                await self._send_subscribe(token_id)

            # Clear books — fresh snapshots virão do servidor
            self._books.clear()

            # Process messages
            async for msg in self._ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.PING:
                    await self._ws.pong()
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        finally:
            await self._close_ws()

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_message(self, raw: object) -> None:
        """Parse (orjson) → route → update book → bridge."""
        try:
            if isinstance(raw, str):
                raw_bytes = raw.encode("utf-8")
            elif isinstance(raw, (bytes, bytearray, memoryview)):
                raw_bytes = bytes(raw)
            else:
                return
            data = orjson.loads(raw_bytes)
        except (orjson.JSONDecodeError, ValueError) as e:
            logger.error(f"[WATCHER:PARSE_ERROR] {e}")
            return

        event_type = data.get("event_type") or data.get("type", "")
        asset_id = data.get("asset_id") or data.get("token_id", "")

        if event_type == "book":
            self._on_book_snapshot(asset_id, data)
        elif event_type == "price_change":
            self._on_book_update(asset_id, data)
        elif event_type in ("last_trade_price", "trade"):
            self._on_trade(asset_id, data)

    def _on_book_snapshot(self, token_id: str, data: dict) -> None:
        """Handle book snapshot: criar/resetar book e notificar bridge."""
        if token_id not in self._books:
            self._books[token_id] = LocalOrderBook(token_id)
        book = self._books[token_id]
        book.apply_snapshot(data)
        self._bridge.on_book_updated(token_id, book)

    def _on_book_update(self, token_id: str, data: dict) -> None:
        """
        Handle book update. Pode levantar BookCorruptionError em gap
        de sequência, forçando reconexão limpa.
        """
        if token_id not in self._books:
            return  # Sem snapshot ainda — ignorar
        book = self._books[token_id]
        book.apply_update(data)  # Pode levantar BookCorruptionError
        self._bridge.on_book_updated(token_id, book)

    def _on_trade(self, token_id: str, data: dict) -> None:
        """Handle trade message: enviar para bridge imediatamente."""
        self._bridge.on_trade(token_id, data)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def subscribe_market(self, token_id: str) -> None:
        """
        Subscribe to level2 (book) + trades for a token.

        Payload JSON conforme protocolo Polymarket CLOB.
        Se WebSocket ativo, envia imediatamente.
        Senão, será enviado automaticamente na próxima conexão.
        """
        self._subscribed_tokens.add(token_id)
        if self._ws is not None and not self._ws.closed:
            await self._send_subscribe(token_id)

    async def _send_subscribe(self, token_id: str) -> None:
        """Envia payload de subscription para Polymarket."""
        if self._ws is None or self._ws.closed:
            return
        payload = orjson.dumps({
            "auth": {},
            "type": "subscribe",
            "channel": "market",
            "assets_ids": [token_id],
        })
        await self._ws.send_bytes(payload)
        logger.info(f"[WATCHER:SUBSCRIBE] {token_id}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """
        Shutdown limpo:
        - Cancelar tasks internas
        - Fechar WebSocket
        - Fechar aiohttp session
        - Sem leaks
        """
        self._running = False
        for task in self._tasks:
            task.cancel()
        await self._close_ws()
        logger.info("[WATCHER:SHUTDOWN]")

    async def _close_ws(self) -> None:
        """Fechar WebSocket e session sem exceções."""
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
        self._session = None
