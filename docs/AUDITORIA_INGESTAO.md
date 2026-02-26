# Auditoria Quantitativa da Ingestão

Logs estruturados JSON + script de análise. Sem Prometheus, sem dependências extras.

---

## 1. Execução

```bash
mkdir -p logs
PYTHONHASHSEED=42 RANDOM_SEED=42 uvicorn src.server:app --host 0.0.0.0 --port 8000 2>&1 | tee logs/observe.log
```

Aguardar pelo menos 30 minutos para acumular amostras.

---

## 2. Análise

```bash
python scripts/analyze_ingestion_log.py logs/observe.log
```

**Saída esperada (sucesso):**
```
INGESTION_STATUS: PASS
```

**Saída esperada (falha):**
```
INGESTION_STATUS: FAIL
Motivos:
  amostra 5: snapshots_window=0
  amostra 12: silence_seconds=35.2 >= 30
```

---

## 3. Critérios de aprovação

| Critério | Condição |
|----------|----------|
| snapshots_window | Nunca zero em nenhuma amostra |
| engine_events_window | Nunca zero em nenhuma amostra |
| discard_rate_window | < 0.02 |
| silence_seconds | < 30 |
| ws_disconnects_total | ≤ 1 (total do run) |

---

## 4. Formato do log INGESTION_AUDIT

Emitido a cada 10 segundos:

```json
{
  "kind": "INGESTION_AUDIT",
  "snapshots_total": 1234,
  "snapshots_window": 45,
  "engine_events_window": 45,
  "events_discarded_window": 0,
  "discard_rate_window": 0.0,
  "silence_seconds": 0.5,
  "engine_idle_seconds": 0.3,
  "ws_disconnects_total": 0,
  "gaps_total": 0,
  "lag_ms_p50": 0.0,
  "lag_ms_p95": 0.0
}
```

---

## 5. Onde está integrado

- **src/monitoring/ingestion_audit.py** — `IngestionAuditAggregator`
- **src/orchestrator.py** — thread iniciada em `run_system()` após MarketDataEngine
- **src/metrics.py** — `inc_ws_disconnects()` chamado pelo watcher em desconexões
- **src/watcher/client.py** — `inc_ws_disconnects()` antes de cada reconnect

---

## 6. Interpretação

**PASS:** Ingestão estruturalmente saudável.

**FAIL:** Diagnóstico objetivo, não intuição. Use os motivos para identificar:
- Falta de dados → snapshots_window=0
- Engine travada → engine_events_window=0
- Fila saturada → discard_rate alto
- Silêncio prolongado → silence_seconds
- Instabilidade de rede → ws_disconnects > 1
