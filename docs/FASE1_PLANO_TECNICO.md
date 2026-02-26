# Fase 1 — Plano Técnico: Ambiente 100% Determinístico

**Objetivo:** Transformar o ambiente em 100% determinístico antes de qualquer validação estatística.

---

## 1️⃣ Fixação de Runtime

### 1.1 Arquivo `runtime.txt` correto

**Arquivo:** `runtime.txt` (raiz do projeto)

```
python-3.11.9
```

**Versão exata recomendada:** `3.11.9` (última patch da série 3.11, estável e suportada pelo Render)

**Alternativas válidas:**
- `python-3.11.10` (se disponível no Render)
- Evitar `python-3.14` — incompatibilidades com dataclasses e tipagem

### 1.2 Como validar versão no boot

Adicione verificação no início de `src/server.py`:

```python
import sys

REQUIRED_PYTHON = (3, 11)
_REQUIRED_MAX = (3, 12)


def _check_python_version() -> None:
    """Fail fast se versão Python incorreta."""
    v = sys.version_info
    if (v.major, v.minor) < REQUIRED_PYTHON:
        raise RuntimeError(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ necessário. "
            f"Atual: {v.major}.{v.minor}.{v.micro}"
        )
    if (v.major, v.minor) >= _REQUIRED_MAX:
        raise RuntimeError(
            f"Python 3.14+ não suportado (incompatibilidades). "
            f"Use 3.11.x. Atual: {v.major}.{v.minor}.{v.micro}"
        )


_check_python_version()
```

### 1.3 Log de versão Python no startup

No `startup` do FastAPI ou no top-level após imports:

```python
import sys
import logging

logger = logging.getLogger(__name__)

def _log_boot_info() -> None:
    v = sys.version_info
    logger.info(
        "[BOOT] [PYTHON] version=%s.%s.%s executable=%s",
        v.major, v.minor, v.micro,
        sys.executable,
    )
```

Chamar em `@app.on_event("startup")`:

```python
@app.on_event("startup")
async def startup() -> None:
    _log_boot_info()
    # ... resto
```

---

## 2️⃣ Determinismo de Dependências

### 2.1 Congelar requirements corretamente

**Comando para gerar `requirements.txt` congelado:**

```bash
pip freeze > requirements.frozen.txt
```

**Estratégia recomendada:** Manter `requirements.txt` com pins explícitos (sem `>=`):

```txt
# requirements.txt — versões fixas
aiohttp==3.9.5
httpx==0.27.2
websockets==12.0
orjson==3.9.15
fastapi==0.109.2
uvicorn==0.27.1
jinja2==3.1.3
pydantic==2.6.1
certifi==2024.2.2
python-dateutil==2.8.2
```

**Gerar pins a partir do ambiente limpo:**

```bash
python -m venv .venv_freeze
source .venv_freeze/bin/activate  # ou .venv_freeze\Scripts\activate no Windows
pip install -r requirements.txt
pip freeze | grep -E "^(aiohttp|httpx|fastapi|uvicorn|pydantic|jinja2|orjson|certifi|python-dateutil|websockets)="
```

### 2.2 Evitar drift futuro

1. **CI/CD:** Rodar `pip freeze` e comparar com `requirements.txt` — falhar se diff
2. **Dependabot / Renovate:** Pins fixos reduzem atualizações indesejadas
3. **Lockfile opcional:** `pip-tools` para `requirements.in` → `requirements.txt` determinístico

```bash
pip install pip-tools
pip-compile requirements.in -o requirements.txt
```

### 2.3 Validar integridade no boot

```python
def _verify_critical_imports() -> None:
    """Verifica que dependências críticas carregam."""
    critical = [
        ("fastapi", "FastAPI"),
        ("uvicorn", None),
        ("aiohttp", None),
        ("orjson", None),
    ]
    for module, attr in critical:
        try:
            m = __import__(module)
            if attr and not hasattr(m, attr):
                raise RuntimeError(f"{module} não tem {attr}")
        except ImportError as e:
            raise RuntimeError(f"Dependência faltando: {module}") from e
```

### 2.4 Ferramenta sugerida

- **pip-tools** (`pip-compile` / `pip-sync`) para lock determinístico
- **poetry** (alternativa) — `poetry.lock` garante reprodutibilidade

---

## 3️⃣ Unificação de Entry Point

### 3.1 Estrutura final recomendada

```
arbitrage-system/
├── runtime.txt              # python-3.11.9
├── requirements.txt         # pins fixos
├── Procfile                 # único ponto de comando
├── src/
│   ├── __init__.py
│   ├── server.py            # app FastAPI + uvicorn entry
│   └── ...
└── render.yaml              # doc apenas; Render usa Dashboard
```

### 3.2 Procfile vs Start Command

**Use Procfile.** O Render prioriza o Procfile sobre o Start Command do Dashboard. Um único source of truth evita conflito.

### 3.3 Procfile final

**Arquivo:** `Procfile`

```
web: uvicorn src.server:app --host 0.0.0.0 --port $PORT
```

Sem `--reload` em produção.

### 3.4 Estrutura de `main.py` (se existir)

Se houver `main.py` para CLI do orchestrator (não para web):

```
src/
├── server.py    # Web: FastAPI + uvicorn
├── __main__.py  # opcional: python -m src
└── ...
```

Para `python -m src.server`:

```python
# src/server.py (final)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=False)
```

### 3.5 Validar uvicorn localmente

```bash
# 1. Ambiente
cd /Users/momachado/Desktop/arbitrage-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Versão
python --version  # deve ser 3.11.x

# 3. Import
python -c "from src.server import app; print('OK')"

# 4. Subir servidor
uvicorn src.server:app --host 0.0.0.0 --port 8000

# 5. Em outro terminal
curl -s http://localhost:8000/health | jq .
```

---

## 4️⃣ Correção do Erro de Dataclass

### 4.1 Por que ocorre o erro

**Erro:** `non-default argument follows default argument`

Em dataclasses, a ordem dos campos importa. Campos **sem** valor default devem vir **antes** de campos **com** valor default. O compilador Python gera `__init__` na ordem dos campos; um parâmetro obrigatório não pode vir depois de um opcional.

### 4.2 Exemplo incorreto

```python
@dataclass
class SimulatedFill:
    order_id: str
    market_id: str
    executed_size: float
    execution_price: float
    order_size: float = 0.0      # ← default
    slippage_applied: float      # ← ERRO: sem default, vem depois de order_size
    latency_ms: float
    timestamp: str
```

### 4.3 Exemplo correto

**Opção A:** Mover campos com default para o final

```python
@dataclass
class SimulatedFill:
    order_id: str
    market_id: str
    side: str
    executed_size: float
    execution_price: float
    mid_at_signal: float
    spread_at_signal: float
    liquidity_available: float
    slippage_applied: float
    latency_ms: float
    timestamp: str
    order_size: float = 0.0
```

**Opção B:** Dar default a todos os opcionais

```python
@dataclass
class SimulatedFill:
    order_id: str
    market_id: str
    side: str
    executed_size: float
    execution_price: float
    mid_at_signal: float
    spread_at_signal: float
    liquidity_available: float
    order_size: float = 0.0
    slippage_applied: float = 0.0
    latency_ms: float = 0.0
    timestamp: str = ""
```

**Recomendação:** Opção A — `order_size` é o único opcional real, vai no final.

---

## 5️⃣ Checklist de Boot Determinístico

| # | Verificação | Comando / Ação |
|---|-------------|----------------|
| 1 | Uvicorn sobe | `uvicorn src.server:app --host 0.0.0.0 --port 8000` — sem traceback |
| 2 | `/health` responde | `curl http://localhost:8000/health` → `{"status":"ok",...}` |
| 3 | Log de versão | Grep em logs: `[BOOT] [PYTHON]` |
| 4 | Instrumentação inicializa | Grep: `[METRICS]` ou contadores em `/health` |
| 5 | Sem exceção silenciosa | Logs sem `Traceback` ou `Exception` não tratada |

**Script de validação:**

```bash
#!/bin/bash
set -e
echo "1. Starting server in background..."
uvicorn src.server:app --host 127.0.0.1 --port 8000 &
PID=$!
sleep 5

echo "2. Health check..."
curl -sf http://127.0.0.1:8000/health | jq -e '.status == "ok"'

echo "3. Shutting down..."
kill $PID 2>/dev/null || true
echo "OK: Boot validation passed"
```

---

## 6️⃣ Hardening Adicional

### 6.1 Seed global

```python
# src/server.py ou antes de qualquer lógica estocástica
import random

def _set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    logger.info("[BOOT] [SEED] global_seed=%d", seed)

# Chamar no startup
_set_global_seed(int(os.environ.get("RANDOM_SEED", "42")))
```

### 6.2 Log de variáveis críticas

```python
def _log_critical_env() -> None:
    critical_vars = ["RUN_MODE", "PORT", "TRADING_SERVER", "POLYMARKET_TOKENS"]
    for k in critical_vars:
        v = os.environ.get(k, "<unset>")
        if k == "POLYMARKET_TOKENS" and v != "<unset>":
            v = f"<len={len(v.split(','))} tokens>"  # não logar tokens
        logger.info("[BOOT] [ENV] %s=%s", k, v)
```

### 6.3 Proteção contra múltiplos processos

```python
import fcntl
import os

def _single_instance_lock() -> None:
    """Garante que apenas uma instância rode por diretório."""
    lockfile = os.path.join(os.path.expanduser("~"), ".arbitrage_system.lock")
    try:
        fd = os.open(lockfile, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Manter fd aberto; será liberado ao sair
    except (BlockingIOError, OSError) as e:
        raise RuntimeError("Outra instância já está rodando") from e
```

*(Em containers/Render normalmente não é necessário; o Render já garante uma instância por service.)*

### 6.4 Fail fast

```python
# No topo de src/server.py, antes de qualquer lógica
import sys

def _fail_fast_checks() -> None:
    _check_python_version()
    _verify_critical_imports()
    # Opcional: _single_instance_lock()

_fail_fast_checks()
```

---

## Resumo de Arquivos a Criar/Alterar

| Arquivo | Ação |
|---------|------|
| `runtime.txt` | Manter `python-3.11.9` |
| `requirements.txt` | Pins fixos (evitar `>=`) |
| `Procfile` | Criar com `web: uvicorn src.server:app --host 0.0.0.0 --port $PORT` |
| `src/server.py` | Adicionar `_check_python_version`, `_log_boot_info`, `_fail_fast_checks`, `_log_critical_env`, `_set_global_seed` |
| `src/execution/realistic_simulator.py` | Corrigir `SimulatedFill`: mover `order_size` para o final |

---

## Resultado Esperado

Ao concluir a Fase 1:

- Deploy no Render previsível e reproduzível
- Versão Python fixa e validada
- Dependências congeladas
- Entry point único (Procfile)
- Dataclass corrigido
- Boot validado via checklist
- Hardening aplicado (seed, log de env, fail fast)
