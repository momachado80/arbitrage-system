# Active Trade Aging Diagnostics — Consistência Interna

## Problema observado

Para `shadow_1000_structural_fillrelax_capfloorrelax_v1` em produção:

- **Bloco agregado por profile:** `activeTradesCount = 0`, `activeTradesAging = []`, `oldestActiveTradeMs = 0`, `avgActiveTradeMs = 0`, `activeTradeIds = []`
- **Consulta específica:** `.activeTradeAgingDiagnostics.byProfile["shadow_1000_structural_fillrelax_capfloorrelax_v1"].activeTradesAging` retornava 2 trades ativos reais (holdingMs 708 e 516, `likelyExitReasonIfClosedNow = "within_hold_no_exit_trigger"`)

Ou seja, o summary agregado e a lista detalhada divergiam.

## Causa raiz

1. **Desalinhamento de snapshot:** `profiles` era obtido com `getAllShadowProfiles()` no início do handler, antes do `await getMergedOpportunitiesForAging()`. Durante o `await`, o event loop podia executar `runCycle`, alterando `profileStates` (abertura/fechamento de trades). Ao retomar, o audit continuava usando o snapshot antigo de `profiles`, enquanto o estado em memória já podia ter mudado.

2. **Referências mutáveis:** `p.activeTrades` era passado por referência. Se o array fosse mutado durante o processamento, agregados e lista detalhada poderiam divergir.

## Correções aplicadas

### 1. Re-fetch de profiles após o await (audit route)

- `profilesForAging = getAllShadowProfiles()` é chamado **imediatamente antes** de `getActiveTradeAgingDiagnostics`, após o `await getMergedOpportunitiesForAging()`.
- Garante que o bloco de aging use o estado mais recente após qualquer mutação durante o await.

### 2. Snapshot de activeTrades (audit route)

- Em vez de `activeTrades: p.activeTrades`, passa-se `activeTrades: [...(p.activeTrades ?? [])]`.
- Cria uma cópia do array no momento da chamada, evitando mutações durante o processamento.

### 3. Single source of truth (activeTradeAgingDiagnostics)

- `activeTradesCount`, `activeTradeIds`, `oldestActiveTradeMs`, `avgActiveTradeMs` passam a ser derivados **exclusivamente** do array `entries` (mesma fonte que `activeTradesAging`).
- `activeTradesCount = entries.length`, `activeTradeIds = entries.map(e => e.tradeId)`.
- Garante que agregados e lista detalhada usem exatamente o mesmo snapshot.

## Arquivos alterados

- `dashboard/app/api/shadow/audit/route.ts` — re-fetch de profiles e snapshot de activeTrades
- `dashboard/lib/activeTradeAgingDiagnostics.ts` — derivação de agregados a partir de `entries`

## Validação em produção

```bash
# Obter audit e inspecionar bloco por profile
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.activeTradeAgingDiagnostics.byProfile["shadow_1000_structural_fillrelax_capfloorrelax_v1"]'

# Verificar consistência: count deve igualar length de activeTradesAging
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.activeTradeAgingDiagnostics.byProfile["shadow_1000_structural_fillrelax_capfloorrelax_v1"] | {activeTradesCount, activeTradesAgingLength: (.activeTradesAging | length), activeTradeIds}'
```

Esperado: `activeTradesCount === activeTradesAgingLength` e `activeTradeIds` com os mesmos IDs dos trades em `activeTradesAging`.

## Restrições respeitadas

- Nenhuma alteração em regra econômica
- Nenhuma alteração em exit
- Nenhum challenger novo criado
