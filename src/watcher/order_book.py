"""
Local Order Book — Espelho do CLOB
====================================

Mantém espelho local do order book por token_id.
Validação de sequência rigorosa.

MENSAGENS:
- SNAPSHOT: limpa book e carrega estado inicial, define last_sequence_id
- UPDATE:   valida sequence_id estritamente (msg_seq == last_seq + 1),
            aplica deltas (size=0 remove, size>0 insere/atualiza)

NORMALIZAÇÃO:
- round_price() para preços (chaves do dict)
- round_shares() para tamanhos (valores do dict)
- Strings da API convertidas em floats
- Nunca armazenar strings

ERROS:
- BookCorruptionError em gap de sequência → forçar reconexão no client

LOGS:
- Apenas erros críticos. Não logar cada update.
"""

import logging
from typing import Dict, Optional, Tuple

from src.utils.precision import round_price, round_shares

logger = logging.getLogger(__name__)


class BookCorruptionError(Exception):
    """Gap de sequência detectado — book corrompido, reconexão necessária."""


class LocalOrderBook:
    """Espelho local do order book para um token_id."""

    def __init__(self, token_id: str) -> None:
        self.token_id = token_id
        self.bids: Dict[float, float] = {}  # price → size
        self.asks: Dict[float, float] = {}  # price → size
        self.last_sequence_id: int = 0
        self._snapshot_received: bool = False

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def apply_snapshot(self, data: dict) -> None:
        """
        Aplica snapshot completo.

        Limpa bids/asks, carrega estado inicial, define last_sequence_id.
        """
        self.bids.clear()
        self.asks.clear()

        for entry in data.get("bids", []):
            price, size = self._parse_entry(entry)
            if size > 0:
                self.bids[price] = size

        for entry in data.get("asks", []):
            price, size = self._parse_entry(entry)
            if size > 0:
                self.asks[price] = size

        self.last_sequence_id = int(data.get("sequence_id", 0))
        self._snapshot_received = True

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def apply_update(self, data: dict) -> None:
        """
        Aplica update incremental. Valida sequência estritamente.

        Raises:
            BookCorruptionError: se msg_seq != last_seq + 1
        """
        msg_seq = int(data.get("sequence_id", 0))

        if self._snapshot_received and msg_seq != self.last_sequence_id + 1:
            raise BookCorruptionError(
                f"[WATCHER:SEQ_GAP] token={self.token_id} "
                f"expected={self.last_sequence_id + 1} got={msg_seq}"
            )

        # Apply bid deltas
        for entry in data.get("bids", []):
            price, size = self._parse_entry(entry)
            if size == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size

        # Apply ask deltas
        for entry in data.get("asks", []):
            price, size = self._parse_entry(entry)
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

        self.last_sequence_id = msg_seq

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_entry(entry: object) -> Tuple[float, float]:
        """
        Parse price level entry.

        Aceita dict {"price": "0.65", "size": "100"} ou list ["0.65", "100"].
        Retorna (price, size) normalizados via round_price/round_shares.
        """
        if isinstance(entry, dict):
            price = round_price(float(entry.get("price", "0")))
            size = round_shares(float(entry.get("size", "0")))
        else:
            price = round_price(float(entry[0]))
            size = round_shares(float(entry[1]))
        return price, size

    def best_bid(self) -> Optional[Tuple[float, float]]:
        """Melhor bid (price, size) ou None se book vazio."""
        if not self.bids:
            return None
        price = max(self.bids.keys())
        return (price, self.bids[price])

    def best_ask(self) -> Optional[Tuple[float, float]]:
        """Melhor ask (price, size) ou None se book vazio."""
        if not self.asks:
            return None
        price = min(self.asks.keys())
        return (price, self.asks[price])
