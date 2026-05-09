# Auditoria de Realized PnL — Shadow Closed Trades

**Data:** 2025-02-05  
**Objetivo:** Rastrear realizedPnL, produzir auditoria de trades fechados, identificar drivers de perda e recomendar intervenção segura.

---

## 1. FÓRMULA EXATA DE realizedPnL

### 1.1 Cadeia de código

```
shadowSimulationService.runCycle (L237–282)
  for each activeTrade:
    if shouldClose:
      exitResult = simulateRealisticExit(activeState, latestState, config)
      closeShadowTrade(profileId, tradeId, { realizedPnL: exitResult.realizedPnL, ... })
```

### 1.2 Fórmula em `simulateRealisticExit`

**Arquivo:** `dashboard/lib/realisticExecutionEngine.ts` L326–371

```ts
// 1. Exit price from latest opportunity or flat
exitPrice = latestOpportunity ? 1 - latestOpportunity.edge : activeTrade.effectiveEntryPrice;

// 2. Impact reduces exit price (worse for seller)
effectiveExitPrice = exitPrice - exitImpactResult.expectedPriceWorseningExit;

// 3. PnL percentage
pnlPct = (effectiveExitPrice - activeTrade.effectiveEntryPrice) / max(0.001, activeTrade.effectiveEntryPrice);

// 4. Realized PnL
realizedPnL = activeTrade.filledCapital * pnlPct;
```

**Impact:** `estimateExitImpact` em `marketImpactModel.ts` L87–105:

- `sizeRatio = filledCapital / bookProxy` (bookProxy = max(1, liquidity * 0.7))
- `exitWorsening = spread * sizeRatio + spread * impactFactor * 0.2 + liquidityStressPenalty`
- `expectedPriceWorseningExit = min(0.12, exitWorsening)`

### 1.3 Aplicação no store

**Arquivo:** `dashboard/lib/shadowSimulationStore.ts` L177

```ts
state.realizedPnL += closed.realizedPnL ?? 0;
```

---

## 2. RESUMO DO AUDIT (estrutura)

O endpoint `GET /api/shadow/audit` retorna:

### 2.1 Por profile

- `totalClosed` — total de trades fechados
- `avgRealizedPnL` — média de realizedPnL por trade
- `medianRealizedPnL` — mediana
- `winRate` — % de trades com realizedPnL > 0
- `lossRate` — % de trades com realizedPnL < 0
- `avgHoldingTimeMs`
- `avgFilledCapital`
- `avgObservedEdgeAtEntry`
- `avgCapturableEdgeAtEntry`
- `totalRealizedPnL`
- `sumWins` / `sumLosses`

### 2.2 Breakdown

- **byOpportunityType** — count, totalPnL, avgPnL
- **byExitReason** — count, totalPnL, avgPnL
- **byHoldingBucket** — <1min, 1–5min, 5–15min, 15–60min, 60–300min, >300min

### 2.3 Top/bottom 20

- `worst20` — 20 piores trades por realizedPnL
- `best20` — 20 melhores trades

Campos: tradeId, profileId, opportunityId, opportunityType, sourceType, exitReason, filledCapital, realizedPnL, realizedReturn, holdingTimeMs, holdingTimeBucket, observedEdgeAtEntry, capturableEdgeAtEntry, effectiveEntryPrice, effectiveExitPrice, openedAt, closedAt.

---

## 3. DRIVERS DE PERDA (análise)

| Driver | Como identificar | Código relevante |
|--------|------------------|------------------|
| **Bad entries** | `capturableEdgeAtEntry << observedEdgeAtEntry` | Latency/impact degradam edge entre decisão e execução |
| **Bad exits** | `exitReason = stop_loss` ou `edge_normalization` com PnL negativo | simulateRealisticExit usa `1 - latestEdge` como exitPrice |
| **Poor fill quality** | `filledCapital` muito baixo em trades com edge alto | deterministicFill, fillProbability; `filledCapital` não armazenado em relação a requested |
| **Mark-to-market** | Não afeta realizedPnL no close | Unrealized usa observedEdgeAtEntry; close usa simulateRealisticExit |
| **Settlement assumptions** | `latestOpportunity = null` → exit em effectiveEntryPrice (flat) | Service L263–264: `shouldClose = true` se opportunity desapareceu |

---

## 4. EXPECTANCY NEGATIVA?

O audit retorna `negativeExpectancy: true` se `totalPnL / totalClosed < 0`.

Com dados de produção (equity ~1e-13, 500 closed, start 500): perdas acumuladas ≈ -500 → avg realizedPnL ≈ -1 por trade → **expectancy negativa confirmada**.

---

## 5. INTERVENÇÃO MAIS SEGURA (recomendação)

**Não implementar ainda.** Sugestão para teste:

1. **Aumentar `minNetCapturableEdgeToTrade` levemente** (ex.: 0.006 → 0.008 para shadow_100) para filtrar entradas marginais.
2. **Ou** adicionar `minCapturableEdgeFloor`: só abrir quando `capturableEdgeAtEntry >= 0.01` (ex.).
3. **Ou** reduzir `maxCapitalPerTrade` temporariamente para limitar tamanho de posições perdedoras.

Testar em shadow profile primeiro; não alterar execução em produção até validar.

---

## 6. COMO EXECUTAR O AUDIT

```bash
# Com API rodando
curl http://localhost:3000/api/shadow/audit | jq .
```

Em produção (Railway):

```bash
curl https://<your-app>.railway.app/api/shadow/audit | jq .
```

O JSON retornado contém `profileSummaries`, `byProfile`, `worst20`, `best20`, `lossDriverAnalysis`, `negativeExpectancy`, `safestNextChange`.
