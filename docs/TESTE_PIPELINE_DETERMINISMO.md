# Teste Formal de Determinismo do Pipeline Completo

## Objetivo

Provar que, dada uma sequência fixa de eventos sintéticos, o pipeline produz resultados idênticos:

- Dentro do mesmo processo
- Entre processos separados
- Após reinicialização completa da engine

---

## 1. Instruções de Execução

### A) Mesmo processo — duas execuções consecutivas

O script executa internamente duas vezes com a mesma sequência e compara em memória. Se divergir, falha com exit 1.

### B) Entre processos

```bash
PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_pipeline_determinism.py > run1.json
PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_pipeline_determinism.py > run2.json
diff run1.json run2.json
```

**Esperado:** Nenhuma diferença (exit 0).

### C) Seed diferente

```bash
PYTHONHASHSEED=43 RANDOM_SEED=43 python scripts/test_pipeline_determinism.py > run3.json
diff run1.json run3.json
```

**Esperado:** Diferenças detectadas (exit 1).

### Saída local

O script também grava `pipeline_run.json` no diretório atual.

---

## 2. Critério Formal de Aprovação

**Determinismo aprovado se e somente se:**

| Condição | Comando | Esperado |
|----------|---------|----------|
| run1 == run2 byte a byte | `diff run1.json run2.json` | Exit 0 |
| run1 != run3 | `diff run1.json run3.json` | Exit 1 |
| Execução dupla no mesmo processo | Interno ao script | Objetos idênticos |
| Sem campos de relógio | Inspeção do JSON | Nenhum timestamp real |

---

## 3. Validação de Fontes Não Determinísticas

O script aplica patches para:

- `time.time` → valor fixo
- `time.sleep` → no-op
- `uuid.uuid4` → contador determinístico

**Validação estática (opcional):**

```bash
VALIDATE_NONDET=1 PYTHONHASHSEED=42 RANDOM_SEED=42 python scripts/test_pipeline_determinism.py
```

Se o código usar `time.time()`, `datetime.now()`, `uuid4()` ou `SystemRandom` em módulos críticos, o script falha com mensagem explícita.

---

## 4. Possíveis Causas de Falha

Se o teste falhar, possíveis causas:

- **Estado global não resetado** — componentes compartilham estado entre runs
- **Estrutura mutável compartilhada** — dict/list reutilizado sem cópia
- **RNG chamado antes do seed** — `random`/`numpy` usado antes de `set_global_seed()`
- **Dependência de relógio real** — `time.time()` ou `datetime.now()` no fluxo
- **Ordem de dicionário não estável** — usar `sort_keys=True` no JSON
- **Concorrência não controlada** — threads/async sem sincronização

---

## 5. Estrutura da Saída JSON

```json
{
  "seed": 42,
  "metrics": {
    "total_trades": 4,
    "realized_pnl": 0.0,
    "capital_after_trade": 1000.0,
    "avg_slippage": 0.0,
    "rejection_rate": 0.0,
    "trades": [...]
  }
}
```

- `trades`: lista de registros com campos determinísticos (sem timestamps).

---

## 6. Conclusão

Quando todos os testes passarem, o laboratório pode ser considerado **determinístico no pipeline de decisão**, permitindo avançar para Fase 2 com risco epistemológico reduzido.
