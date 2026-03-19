# Profile Atomic Snapshot — Legibilidade Confiável

## Problema observado

Chamadas separadas ao `/api/shadow/audit` mostravam estados incompatíveis para o mesmo profile no mesmo momento lógico:

- **Leitura 1:** `profileEligibilityDiagnostics` com funil zerado, `chokePointSummary = "sem counters de funil materializados, apesar de histórico econômico existente"`, `closedTradeCount = 5`
- **Leitura 2:** `activeTradeAgingDiagnostics` com `activeTradesCount = 3`, trades com poucos segundos de holding

Isso impedia julgamento confiável por snapshot desalinhado entre blocos/requests.

## Causa raiz

O audit fazia `profiles = getAllShadowProfiles()` no início e, depois, `await getMergedOpportunitiesForAging()`. Durante o `await`, o event loop podia executar `runCycle`, alterando `profileStates` e `funnelState`. Ao retomar, blocos diferentes usavam snapshots diferentes:

- `profileEligibilityDiagnostics` e `profileObservabilityConsistency`: profiles antes do await
- `activeTradeAgingDiagnostics`: profiles após o await (re-fetch)

## Solução

### 1. Single snapshot source (audit route)

- `await getMergedOpportunitiesForAging()` é executado **primeiro**
- Em seguida: `profiles = getAllShadowProfiles()` — uma única leitura
- Todos os blocos dependentes de profile usam esse mesmo `profiles`

### 2. Bloco `profileAtomicSnapshot` em `/api/shadow/audit`

Por profile, em um único payload coerente:

- `profileId`, `label`
- `consistency` — status de consistência (funil vs economia vs heartbeat)
- `funnel` — contadores do funil (pairEligible, fillEligible, capfloorEligible, etc.)
- `conversions` — taxas de conversão entre etapas
- `economics` — realizedPnlTotal, realizedPnlAvg
- `activeTradesSummary` — count, oldestMs, avgMs, summary
- `activeTradesAging` — lista detalhada de trades ativos
- `lastCycleProcessedAt` — heartbeat
- `summary` — resumo agregado (consistência + funil + aging)

### 3. Endpoint dedicado `/api/shadow/profile-snapshot`

- **Sem query:** retorna `profileAtomicSnapshot` completo
- **Com `?profileId=...`:** retorna apenas o snapshot do profile solicitado (404 se não existir)

## Arquivos alterados

- `dashboard/app/api/shadow/audit/route.ts` — await primeiro, single fetch, bloco `profileAtomicSnapshot`
- `dashboard/lib/profileAtomicSnapshot.ts` — módulo de construção do snapshot
- `dashboard/app/api/shadow/profile-snapshot/route.ts` — endpoint dedicado

## Validação em produção

```bash
# Snapshot completo (todos os profiles)
curl -s "https://<DASHBOARD_URL>/api/shadow/profile-snapshot" | jq .

# Snapshot do capfloorrelax
curl -s "https://<DASHBOARD_URL>/api/shadow/profile-snapshot?profileId=shadow_1000_structural_fillrelax_capfloorrelax_v1" | jq .

# Bloco profileAtomicSnapshot no audit
curl -s "https://<DASHBOARD_URL>/api/shadow/audit" | jq '.profileAtomicSnapshot.byProfile["shadow_1000_structural_fillrelax_capfloorrelax_v1"]'
```

## Restrições respeitadas

- Nenhuma alteração em regra econômica
- Nenhum challenger novo criado
- Nenhuma recalibração de degratio
- Apenas restauração de legibilidade confiável da comparação
