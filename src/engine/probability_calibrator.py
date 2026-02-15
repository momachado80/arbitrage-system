"""
Probability Calibration Module
==============================

Módulo de calibragem de probabilidades para robô de mercados preditivos (Polymarket).

OBJETIVO:
---------
Monitorar a qualidade das previsões probabilísticas e retornar um multiplicador
de sizing k ∈ [K_MIN, 1]. Este módulo NÃO decide trades — apenas ajusta a exposição.

MÉTRICAS:
---------
- Brier Score: e = (f - o)² onde f = probabilidade prevista, o = resultado (0/1)
- EWMA update: BS_t = λ × BS_{t-1} + (1 - λ) × e_clamped
- Multiplicador: k = clip(1 - ALPHA × ((BS_fast - BS*) / BS*), K_MIN, 1)

PERSISTÊNCIA (escrita atômica):
-------------------------------
1. NamedTemporaryFile no MESMO diretório do arquivo final (garante mesmo filesystem)
2. write → flush → fsync arquivo
3. os.replace (rename atômico POSIX — requer mesmo filesystem)
4. fsync do diretório (garante durabilidade do rename após power-loss)
5. Tudo sob file lock POSIX (fcntl.flock LOCK_EX) entre processos
6. RLock interno protege apenas memória entre threads

POR QUE os.replace E NÃO shutil.move:
- os.replace é rename atômico garantido no mesmo filesystem (POSIX)
- shutil.move pode fazer copy+delete se os filesystems diferirem
- Criando tmp no mesmo diretório, garantimos mesmo filesystem

POR QUE fsync DO DIRETÓRIO:
- rename() modifica entrada do diretório; sem fsync, power-loss pode reverter
- Padrão em PostgreSQL, SQLite WAL, e sistemas financeiros

CORRUPTION MODE (state_corrupted):
----------------------------------
Quando load_state() detecta corrupção (JSON inválido, schema incompatível,
valores negativos, created_at ausente/não-UTC):
- _state_corrupted = True, _safety_mode = True
- get_sizing_multiplier() retorna K_MIN_GLOBAL SEMPRE (fail-closed absoluto)
  NENHUMA outra lógica roda. Warning emitido uma vez por sessão.
- record_trade_result() continua funcionando em memória (trade_count aumenta)
- save_state() RECUSA gravar (preserva evidência forense)
- Só reset_all() ou recover_from_corruption() restauram persistência

TIMESTAMPS:
-----------
- created_at: ISO-8601 UTC (+00:00), definido na primeira persistência, imutável
- updated_at: ISO-8601 UTC (+00:00), atualizado a cada save_state()
- Timestamps sem timezone ou não-UTC ativam safety_mode + state_corrupted

Autor: Sistema de Arbitragem
"""

import copy
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import threading

logger = logging.getLogger(__name__)

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
    logger.warning(
        "fcntl não disponível nesta plataforma. "
        "Lock de arquivo entre processos não será aplicado."
    )

# =============================================================================
# CONSTANTES
# =============================================================================

SCHEMA_VERSION: int = 1
HALF_LIFE_FAST: int = 20
HALF_LIFE_SLOW: int = 100
E_MAX: float = 0.5
ALPHA: float = 0.7
K_MIN: float = 0.2
K_MIN_GLOBAL: float = 0.2
N_MIN: int = 15
THETA: float = 0.05

DEFAULT_STATE_PATH: str = os.path.expanduser(
    "~/.polymarket_bot/probability_calibrator_state.json"
)

BASELINE_BRIER_BY_CATEGORY: Dict[str, float] = {
    "POLITICS": 0.20, "CRYPTO": 0.22, "FINANCE": 0.18, "SPORTS": 0.18,
    "SCIENCE": 0.20, "WEATHER": 0.20, "ENTERTAINMENT": 0.20, "OTHER": 0.20,
}

TAG_TO_CATEGORY: Dict[str, str] = {
    "elections": "POLITICS", "politics": "POLITICS", "government": "POLITICS",
    "congress": "POLITICS", "senate": "POLITICS", "president": "POLITICS",
    "presidential": "POLITICS", "democrat": "POLITICS", "republican": "POLITICS",
    "biden": "POLITICS", "trump": "POLITICS", "election": "POLITICS",
    "vote": "POLITICS", "voting": "POLITICS", "poll": "POLITICS", "polls": "POLITICS",
    "crypto": "CRYPTO", "bitcoin": "CRYPTO", "btc": "CRYPTO",
    "ethereum": "CRYPTO", "eth": "CRYPTO", "defi": "CRYPTO",
    "token": "CRYPTO", "blockchain": "CRYPTO", "solana": "CRYPTO", "nft": "CRYPTO",
    "stocks": "FINANCE", "stock": "FINANCE", "rates": "FINANCE",
    "interest": "FINANCE", "cpi": "FINANCE", "fed": "FINANCE",
    "federal reserve": "FINANCE", "earnings": "FINANCE", "gdp": "FINANCE",
    "inflation": "FINANCE", "economy": "FINANCE", "economic": "FINANCE",
    "market": "FINANCE", "sp500": "FINANCE", "nasdaq": "FINANCE",
    "sports": "SPORTS", "nba": "SPORTS", "nfl": "SPORTS", "ufc": "SPORTS",
    "soccer": "SPORTS", "football": "SPORTS", "basketball": "SPORTS",
    "baseball": "SPORTS", "mlb": "SPORTS", "tennis": "SPORTS", "golf": "SPORTS",
    "olympics": "SPORTS", "world cup": "SPORTS", "superbowl": "SPORTS",
    "super bowl": "SPORTS",
    "science": "SCIENCE", "ai": "SCIENCE", "artificial intelligence": "SCIENCE",
    "space": "SCIENCE", "nasa": "SCIENCE", "spacex": "SCIENCE",
    "health": "SCIENCE", "covid": "SCIENCE", "pandemic": "SCIENCE",
    "research": "SCIENCE", "technology": "SCIENCE",
    "weather": "WEATHER", "hurricane": "WEATHER", "el nino": "WEATHER",
    "la nina": "WEATHER", "temperature": "WEATHER", "climate": "WEATHER",
    "storm": "WEATHER", "tornado": "WEATHER",
    "movies": "ENTERTAINMENT", "movie": "ENTERTAINMENT", "music": "ENTERTAINMENT",
    "celebrity": "ENTERTAINMENT", "oscars": "ENTERTAINMENT",
    "grammy": "ENTERTAINMENT", "emmys": "ENTERTAINMENT", "tv": "ENTERTAINMENT",
    "television": "ENTERTAINMENT", "netflix": "ENTERTAINMENT",
    "streaming": "ENTERTAINMENT", "box office": "ENTERTAINMENT",
}

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "POLITICS": ["election", "president", "congress", "senate", "governor", "vote", "democrat", "republican"],
    "CRYPTO": ["bitcoin", "crypto", "ethereum", "token", "blockchain", "defi"],
    "FINANCE": ["stock", "fed", "inflation", "gdp", "earnings", "interest rate", "cpi"],
    "SPORTS": ["game", "match", "championship", "playoff", "season", "team", "player"],
    "SCIENCE": ["research", "study", "space", "nasa", "ai", "artificial intelligence"],
    "WEATHER": ["weather", "hurricane", "storm", "temperature", "forecast"],
    "ENTERTAINMENT": ["movie", "film", "music", "album", "award", "celebrity"],
}


# =============================================================================
# ENUMS
# =============================================================================

class MarketCategory(Enum):
    POLITICS = "POLITICS"
    CRYPTO = "CRYPTO"
    FINANCE = "FINANCE"
    SPORTS = "SPORTS"
    SCIENCE = "SCIENCE"
    WEATHER = "WEATHER"
    ENTERTAINMENT = "ENTERTAINMENT"
    OTHER = "OTHER"

    @classmethod
    def from_string(cls, value: str) -> "MarketCategory":
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.OTHER


class TradeType(Enum):
    ARBITRAGE = "ARBITRAGE"
    DIRECTIONAL = "DIRECTIONAL"
    HEDGING = "HEDGING"
    MARKET_MAKING = "MARKET_MAKING"
    STATISTICAL = "STATISTICAL"

    @classmethod
    def from_string(cls, value: str) -> "TradeType":
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.DIRECTIONAL


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CategoryTypeMetrics:
    """Métricas de calibração para uma combinação (categoria, tipo)."""
    category: str
    trade_type: str
    bs_fast: float
    bs_slow: float
    trade_count: int = 0
    total_brier: float = 0.0
    recent_deterioration: bool = False
    last_updated: str = ""


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calculate_lambda(half_life: int) -> float:
    """λ = exp(ln(0.5) / half_life)"""
    if half_life <= 0:
        raise ValueError(f"half_life deve ser > 0, recebido: {half_life}")
    return math.exp(math.log(0.5) / half_life)


def calculate_brier_score(predicted: float, outcome: int) -> float:
    """(f - o)²"""
    return (predicted - outcome) ** 2


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _serialize_key(category: MarketCategory, trade_type: TradeType) -> str:
    return f"{category.value}|{trade_type.value}"


def _deserialize_key(key_str: str) -> Optional[Tuple[MarketCategory, TradeType]]:
    parts = key_str.split("|")
    if len(parts) != 2:
        return None
    try:
        return (MarketCategory(parts[0]), TradeType(parts[1]))
    except ValueError:
        return None


def _validate_utc_timestamp(ts_str: str) -> None:
    """Valida ISO-8601 estrito com timezone UTC (+00:00). Raises ValueError."""
    normalized = ts_str[:-1] + "+00:00" if ts_str.endswith("Z") else ts_str
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        raise ValueError(f"created_at sem timezone: '{ts_str}'")
    if dt.utcoffset() != timedelta(0):
        raise ValueError(f"created_at não é UTC (offset={dt.utcoffset()}): '{ts_str}'")


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class ProbabilityCalibrator:
    """
    Calibragem de probabilidades com persistência atômica.

    Em state_corrupted ou safety_mode: retorna K_MIN_GLOBAL (fail-closed).
    """

    def __init__(
        self,
        state_path: Optional[str] = None,
        category_overrides: Optional[Dict[str, str]] = None,
    ):
        self._state_path = Path(state_path or DEFAULT_STATE_PATH)
        self._category_overrides = category_overrides or {}
        self._metrics: Dict[Tuple[MarketCategory, TradeType], CategoryTypeMetrics] = {}
        self._lock = threading.RLock()
        self._safety_mode: bool = False
        self._safety_mode_reason: str = ""
        self._state_corrupted: bool = False
        self._corrupted_warning_emitted: bool = False
        self._created_at: Optional[str] = None

        self._lambda_fast = calculate_lambda(HALF_LIFE_FAST)
        self._lambda_slow = calculate_lambda(HALF_LIFE_SLOW)

        self._ensure_state_directory()
        self.load_state()

        logger.info(
            f"ProbabilityCalibrator inicializado: "
            f"state_path={self._state_path}, "
            f"safety_mode={self._safety_mode}, "
            f"state_corrupted={self._state_corrupted}"
        )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _ensure_state_directory(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_baseline(self, category: MarketCategory) -> float:
        return BASELINE_BRIER_BY_CATEGORY.get(
            category.value, BASELINE_BRIER_BY_CATEGORY["OTHER"]
        )

    def _get_or_create_metrics_unlocked(
        self, category: MarketCategory, trade_type: TradeType
    ) -> CategoryTypeMetrics:
        key = (category, trade_type)
        if key not in self._metrics:
            baseline = self._get_baseline(category)
            self._metrics[key] = CategoryTypeMetrics(
                category=category.value,
                trade_type=trade_type.value,
                bs_fast=baseline,
                bs_slow=baseline,
            )
        return self._metrics[key]

    def _calculate_k(
        self, category: MarketCategory, metrics: CategoryTypeMetrics
    ) -> float:
        if metrics.trade_count < N_MIN:
            return 1.0
        bs_star = self._get_baseline(category)
        if metrics.bs_fast <= bs_star:
            return 1.0
        return clamp(1.0 - ALPHA * ((metrics.bs_fast - bs_star) / bs_star), K_MIN, 1.0)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def get_sizing_multiplier(
        self, category: MarketCategory, trade_type: TradeType
    ) -> float:
        """
        Fail-closed absoluto.

        Ordem de precedência:
        1. state_corrupted → K_MIN_GLOBAL (warn once)
        2. safety_mode → K_MIN_GLOBAL
        3. Fórmula normal
        """
        if self._state_corrupted:
            if not self._corrupted_warning_emitted:
                logger.warning(
                    "get_sizing_multiplier: estado corrompido, retornando "
                    f"K_MIN_GLOBAL={K_MIN_GLOBAL}. Chame reset_all() ou "
                    "recover_from_corruption() para restaurar."
                )
                self._corrupted_warning_emitted = True
            return K_MIN_GLOBAL
        if self._safety_mode:
            return K_MIN_GLOBAL
        with self._lock:
            metrics = self._get_or_create_metrics_unlocked(category, trade_type)
            return self._calculate_k(category, metrics)

    def record_trade_result(
        self,
        category: MarketCategory,
        trade_type: TradeType,
        predicted_probability: float,
        actual_outcome: int,
    ) -> None:
        if actual_outcome not in (0, 1):
            raise ValueError(f"actual_outcome deve ser 0 ou 1, recebido: {actual_outcome}")
        if predicted_probability < 0 or predicted_probability > 1:
            predicted_probability = clamp(predicted_probability, 0.0, 1.0)

        brier_clamped = min(calculate_brier_score(predicted_probability, actual_outcome), E_MAX)

        with self._lock:
            m = self._get_or_create_metrics_unlocked(category, trade_type)
            m.bs_fast = self._lambda_fast * m.bs_fast + (1 - self._lambda_fast) * brier_clamped
            m.bs_slow = self._lambda_slow * m.bs_slow + (1 - self._lambda_slow) * brier_clamped
            m.trade_count += 1
            m.total_brier += brier_clamped
            m.recent_deterioration = (m.bs_fast - m.bs_slow) > THETA
            m.last_updated = datetime.now(timezone.utc).isoformat()

        self.save_state()
        self._check_exit_safety_mode(category, trade_type)

    def _check_exit_safety_mode(
        self, category: MarketCategory, trade_type: TradeType
    ) -> None:
        if not self._safety_mode:
            return
        with self._lock:
            m = self._get_or_create_metrics_unlocked(category, trade_type)
            if m.trade_count >= N_MIN:
                logger.info(
                    f"Saindo de safety_mode: {category.value}/{trade_type.value} "
                    f"atingiu N_MIN={N_MIN}"
                )
                self._safety_mode = False
                self._safety_mode_reason = ""

    def classify_market_category(
        self, tags: List[str], title: str = "", description: str = ""
    ) -> MarketCategory:
        for tag in tags:
            t = tag.lower().strip()
            if t in self._category_overrides:
                return MarketCategory.from_string(self._category_overrides[t])
        for tag in tags:
            t = tag.lower().strip()
            if t in TAG_TO_CATEGORY:
                return MarketCategory.from_string(TAG_TO_CATEGORY[t])
        text = f"{title} {description}".lower()
        for cat_str, kws in CATEGORY_KEYWORDS.items():
            for kw in kws:
                if kw in text:
                    return MarketCategory.from_string(cat_str)
        return MarketCategory.OTHER

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    @staticmethod
    def _fsync_directory(dir_path: str) -> None:
        """Fsync do diretório para durabilidade total do rename."""
        try:
            dir_fd = os.open(dir_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as e:
            logger.warning(f"fsync de diretório não suportado ou falhou: {e}")

    def _write_tmp_and_rename(self, json_str: str, state_dir: str) -> None:
        """
        NamedTemporaryFile (mesmo dir) → write → flush → fsync → os.replace → dir fsync.

        Deve ser chamado DENTRO da região protegida pelo file lock.
        O tmp é criado no mesmo diretório do arquivo final para garantir
        que os.replace() opere no mesmo filesystem (requisito POSIX).
        """
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="calibrator_",
                dir=state_dir, delete=False
            ) as f:
                temp_path = f.name
                f.write(json_str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, str(self._state_path))
            self._fsync_directory(state_dir)
        except Exception:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def save_state(self) -> None:
        """
        Persiste estado com escrita atômica.

        Se state_corrupted: RECUSA gravar (preserva evidência forense).
        """
        if self._state_corrupted:
            logger.error(
                "save_state RECUSADO: estado marcado como corrupto "
                f"(reason: {self._safety_mode_reason}). "
                "Chame reset_all() ou recover_from_corruption()."
            )
            return

        try:
            with self._lock:
                snapshot: Dict[str, Any] = {}
                for (cat, tt), m in self._metrics.items():
                    snapshot[_serialize_key(cat, tt)] = {
                        "bs_fast": m.bs_fast,
                        "bs_slow": m.bs_slow,
                        "trade_count": m.trade_count,
                    }

            snapshot = copy.deepcopy(snapshot)

            now = datetime.now(timezone.utc).isoformat()
            if self._created_at is None:
                self._created_at = now

            json_str = json.dumps({
                "schema_version": SCHEMA_VERSION,
                "created_at": self._created_at,
                "updated_at": now,
                "metrics": snapshot,
            }, indent=2)

            state_dir = str(self._state_path.parent)
            lock_path = str(self._state_path) + ".lock"

            if _HAS_FCNTL:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    self._write_tmp_and_rename(json_str, state_dir)
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
            else:
                self._write_tmp_and_rename(json_str, state_dir)

        except Exception as e:
            logger.error(f"Erro ao salvar estado: {e}")
            self._safety_mode = True
            self._safety_mode_reason = f"Falha em save_state: {e}"

    def load_state(self) -> None:
        if not self._state_path.exists():
            self._metrics = {}
            self._safety_mode = False
            self._state_corrupted = False
            self._corrupted_warning_emitted = False
            self._created_at = None
            return

        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)

            sv = data.get("schema_version")
            if sv is None:
                raise ValueError("Schema inválido: 'schema_version' ausente")
            if sv != SCHEMA_VERSION:
                raise ValueError(
                    f"Schema incompatível: arquivo v{sv}, código v{SCHEMA_VERSION}. "
                    "Migração automática não suportada."
                )

            created_at = data.get("created_at")
            if not isinstance(created_at, str) or not created_at:
                raise ValueError("Schema inválido: 'created_at' ausente ou inválido")
            _validate_utc_timestamp(created_at)

            raw_metrics = data.get("metrics")
            if not isinstance(raw_metrics, dict):
                raise ValueError("Schema inválido: 'metrics' ausente ou não é dict")

            restored: Dict[Tuple[MarketCategory, TradeType], CategoryTypeMetrics] = {}
            for key_str, entry in raw_metrics.items():
                parsed = _deserialize_key(key_str)
                if parsed is None:
                    continue
                cat, tt = parsed
                if not isinstance(entry, dict):
                    raise ValueError(f"{key_str}: não é dict")
                bs_fast = entry.get("bs_fast")
                bs_slow = entry.get("bs_slow")
                trade_count = entry.get("trade_count")
                if not isinstance(bs_fast, (int, float)):
                    raise ValueError(f"{key_str}: bs_fast não é numérico")
                if not isinstance(bs_slow, (int, float)):
                    raise ValueError(f"{key_str}: bs_slow não é numérico")
                if not isinstance(trade_count, int):
                    raise ValueError(f"{key_str}: trade_count não é int")
                if bs_fast < 0 or bs_slow < 0 or trade_count < 0:
                    raise ValueError(
                        f"{key_str}: valor negativo detectado "
                        f"(bs_fast={bs_fast}, bs_slow={bs_slow}, tc={trade_count})"
                    )
                restored[(cat, tt)] = CategoryTypeMetrics(
                    category=cat.value, trade_type=tt.value,
                    bs_fast=float(bs_fast), bs_slow=float(bs_slow),
                    trade_count=trade_count,
                )

            with self._lock:
                self._metrics = restored
            self._created_at = created_at
            self._safety_mode = False
            self._state_corrupted = False
            self._corrupted_warning_emitted = False

        except json.JSONDecodeError as e:
            self._safety_mode = True
            self._state_corrupted = True
            self._safety_mode_reason = f"JSON inválido: {e}"
            logger.error(f"Estado corrompido (JSON): {e}")
            self._metrics = {}
            self._created_at = None

        except (KeyError, ValueError, TypeError) as e:
            self._safety_mode = True
            self._state_corrupted = True
            self._safety_mode_reason = f"Schema inválido: {e}"
            logger.error(f"Estado com schema inválido: {e}")
            self._metrics = {}
            self._created_at = None

        except Exception as e:
            self._safety_mode = True
            self._state_corrupted = True
            self._safety_mode_reason = f"Erro desconhecido: {e}"
            logger.error(f"Erro ao carregar estado: {e}")
            self._metrics = {}
            self._created_at = None

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def safety_mode(self) -> bool:
        return self._safety_mode

    @property
    def safety_mode_reason(self) -> str:
        return self._safety_mode_reason

    @property
    def state_corrupted(self) -> bool:
        return self._state_corrupted

    def is_corrupted(self) -> bool:
        """Retorna True se o estado em disco está corrompido."""
        return self._state_corrupted

    # ------------------------------------------------------------------
    # Leitura de métricas (cópias)
    # ------------------------------------------------------------------

    def get_metrics(
        self, category: MarketCategory, trade_type: TradeType
    ) -> Optional[Dict[str, Any]]:
        key = (category, trade_type)
        with self._lock:
            if key not in self._metrics:
                return None
            m = self._metrics[key]
            return {
                "category": m.category,
                "trade_type": m.trade_type,
                "bs_fast": m.bs_fast,
                "bs_slow": m.bs_slow,
                "trade_count": m.trade_count,
                "average_brier": m.total_brier / m.trade_count if m.trade_count > 0 else 0,
                "recent_deterioration": m.recent_deterioration,
                "sizing_multiplier": self._calculate_k(category, m),
                "last_updated": m.last_updated,
            }

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        with self._lock:
            for (cat, tt), m in self._metrics.items():
                result[f"{cat.value}/{tt.value}"] = {
                    "category": m.category, "trade_type": m.trade_type,
                    "bs_fast": m.bs_fast, "bs_slow": m.bs_slow,
                    "trade_count": m.trade_count,
                    "average_brier": m.total_brier / m.trade_count if m.trade_count > 0 else 0,
                    "recent_deterioration": m.recent_deterioration,
                    "baseline": self._get_baseline(cat),
                    "sizing_multiplier": self._calculate_k(cat, m),
                    "last_updated": m.last_updated,
                }
        return result

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._metrics:
                return {
                    "total_categories": 0, "total_trades": 0,
                    "average_multiplier": 1.0, "min_multiplier": 1.0,
                    "safety_mode": self._safety_mode, "status": "no_data",
                }
            total_trades = sum(m.trade_count for m in self._metrics.values())
            multipliers = [
                self._calculate_k(cat, m)
                for (cat, _), m in self._metrics.items()
                if m.trade_count >= N_MIN
            ]
            n_categories = len(self._metrics)

        avg_mult = sum(multipliers) / len(multipliers) if multipliers else 1.0
        min_mult = min(multipliers) if multipliers else 1.0

        if self._safety_mode:
            status = "safety_mode"
        elif min_mult >= 0.8:
            status = "healthy"
        elif min_mult >= 0.5:
            status = "caution"
        else:
            status = "warning"

        return {
            "total_categories": n_categories, "total_trades": total_trades,
            "average_multiplier": round(avg_mult, 3),
            "min_multiplier": round(min_mult, 3),
            "safety_mode": self._safety_mode, "status": status,
        }

    # ------------------------------------------------------------------
    # Resets e recuperação
    # ------------------------------------------------------------------

    def reset_category(self, category: MarketCategory, trade_type: TradeType) -> None:
        with self._lock:
            key = (category, trade_type)
            if key in self._metrics:
                del self._metrics[key]
        self.save_state()

    def reset_all(self) -> None:
        """Limpa tudo, restaura persistência, e grava estado fresco."""
        with self._lock:
            self._metrics.clear()
            self._safety_mode = False
            self._safety_mode_reason = ""
        self._created_at = None
        self._state_corrupted = False
        self._corrupted_warning_emitted = False
        self.save_state()

    def recover_from_corruption(self) -> None:
        """Recuperação explícita de estado corrompido. Alias para reset_all()."""
        logger.warning("recover_from_corruption: reinicializando calibrador")
        self.reset_all()
