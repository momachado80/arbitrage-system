# Auditoria cirúrgica: shadow_1000_structural_lateexit_nonreversion_v1

**Data:** 2025-03-14  
**Objetivo:** Descobrir por que `lateExitTriggeredCount = 0` apesar de 12+ closes.

---

## A. Resumo executivo da causa mais provável

**Causa mais provável:** `observed_edge_mantem_alto` + `todos_max_holding_fora_janela`

1. **Todos os trades fecham por `max_holding_time`** (holding ~300s). Nenhum fecha dentro da janela de late exit (90–300s).
2. O late exit **só é avaliado quando** `90_000 <= holdingMs < 300_000`. Quando `holdingMs >= 300_000`, o bloco late exit não roda; o close é por max holding.
3. Durante 90–300s (≈21 ciclos de 10s), o late exit foi avaliado em cada ciclo. **Nenhuma condição disparou.**
4. A hipótese: o `observed` (edge) permanece acima dos thresholds ao longo da janela. No close (300s), se `edgeAtExit` ainda for alto (ex. >5%), isso indica que o edge raramente cai a níveis que disparam os critérios.

**Conclusão:** Thresholds frouxos para o regime. O capfloor de entrada é 4.5%; trades entram com edge alto. O `stagnantEdgeFloor` (3%) e `netEdgeProlongedFloor` (2%) tendem a nunca ser atingidos enquanto o edge se mantém acima desses valores.

---

## B. Evidência trade a trade / agregada

### Produção (20 trades validados post-deploy)

| Métrica | Valor |
|---------|-------|
| totalClosed | 20 |
| closedByMaxHolding | 16 |
| closedByOther (stop_loss) | 4 |
| lateExitTriggeredCount | 0 |
| closedInLateExitWindow | 0 |
| avgObservedAtClose (16 com edge) | **25.28%** |
| avgObservedAtEntry | 25.58% |
| wouldTrigger* (excl. edge null) | 0 |
| edgeAtExitNullCount | 4 |

### Nova instrumentação: `structuralLateExitCausalAudit`

- `perTrade`: por trade: holdingTimeMs, exitReason, observedAtEntry, observedAtClose, proximidade a cada threshold
- `aggregate`: closedInLateExitWindow, closedByMaxHolding, wouldTriggerX, avgObservedAtClose, closestCriterionCounts
- `conclusion`: causeMostLikely, evidenceSummary, thresholdsFrouxos, recomendacao

---

## C. Arquivos analisados

| Arquivo | Papel |
|---------|-------|
| `lib/shadowSimulationService.ts` | Lógica de late exit no ciclo (L751–806, L834–839) |
| `lib/shadowSimulationProfiles.ts` | Config do profile (minObservationMs=90k, reversionMinFraction=0.5, stagnantEdgeFloor=0.03, etc.) |
| `lib/shadowSimulationStore.ts` | Campos lateExit* em ShadowTrade |
| `lib/shadowClosedTradeAudit.ts` | Mapeamento para ClosedTradeAuditEntry |
| `lib/structuralLateExitDiagnostics.ts` | Diagnósticos agregados |

---

## D. Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `lib/structuralLateExitCausalAudit.ts` | **NOVO** – auditoria causal com perTrade, aggregate, conclusion |
| `app/api/shadow/audit/route.ts` | Exposição de `structuralLateExitCausalAudit` na API |

---

## E. Auditoria estática do código

### Onde o late exit roda

- **Arquivo:** `shadowSimulationService.ts`
- **Linhas:** 751–806
- **Escopo:** loop `for (const t of [...updatedState.activeTrades])`

### Condição de entrada

```ts
profile.lateExitTarget &&
holdingMs >= profile.lateExitTarget.minObservationMs &&  // 90_000
holdingMs < profile.maxHoldingTimeMs                     // 300_000
```

Se `holdingMs >= 300_000`, esse bloco **não é executado**.

### Ordem de avaliação (quando na janela)

1. **opportunity_absent_late**: `!latestState` por `absentCyclesInLatePhase` (2) ciclos consecutivos
2. **non_reversion**: `observedNow < 0.5 * observedAtEntry`
3. **net_edge_prolonged_low**: `observedNow <= 0.02`
4. **stagnant_edge**: `observedNow <= 0.03` por `stagnantCycles` (3) ciclos consecutivos; caso contrário, contador zera

### Campos usados

| Condição | Campos |
|----------|--------|
| opportunity_absent_late | `latestState` (opp presente no `oppMap`), `structuralRiskConsecutiveAbsentByTradeId` |
| non_reversion | `t.observedEdgeAtEntry`, `latestState.edge` (observedNow) |
| net_edge_prolonged_low | `latestState.edge` |
| stagnant_edge | `latestState.edge`, `lateExitStagnantCyclesByTradeId` |

### Caminhos que fecham antes do late exit

| Condição | Bloco | Momento |
|----------|-------|---------|
| exitKillTarget | L725–757 | `holdingMs < 90s` (late exit não tem exitKillTarget) |
| early thesis | L807–840 | `structuralRiskManagedTarget`, sem exitKill/lateExit, `holdingMs < 90s` |
| max_holding_time | L834–835 | `holdingMs >= 300s` – **todos os closes atuais** |
| stop_loss | L840 | `pnlPct <= -0.03` |
| take_profit | L841 | `pnlPct >= 0.05` |
| edge_normalization | L842 | `|latestState.edge| < 0.005` |
| opportunity_disappeared | L844 | `!latestState` (sem ciclos consecutivos para late exit) |

### Condições improváveis ou mal alinhadas

- **`netEdgeProlongedFloor` (0.02)**: trades entram com capfloor 4.5%; observed geralmente >4%. Edge precisa cair muito para ≤2%.
- **`stagnantEdgeFloor` (0.03)**: precisa de 3 ciclos consecutivos (30s) com edge ≤3%. Oscilações acima de 3% resetam o contador.
- **`reversionMinFraction` (0.5)**: observed precisa cair para <50% do valor de entrada. Com observed inicial ~5–25%, exigiria <2.5–12.5%.

---

## F. Diagnóstico final

**Classe:** Thresholds frouxos para o regime + late exit raramente aplicável.

- O regime mostra trades mantendo holding até ~300s.
- O late exit só age entre 90s e 300s.
- Dentro dessa janela, `observed` permanece acima dos thresholds.
- Resultado: nenhum late exit dispara e todos os trades fecham por max holding.

---

## G. Recomendação objetiva

1. **Instrumentação:** `structuralLateExitCausalAudit` já expõe `perTrade`, `aggregate` e `conclusion`; usar para validar hipóteses em produção.
2. **Apertar thresholds** (opcional, com cautela):
   - `stagnantEdgeFloor`: 0.03 → 0.045 ou 0.05
   - `netEdgeProlongedFloor`: 0.02 → 0.03
   - `reversionMinFraction`: 0.5 → 0.6
3. **Avançar a janela:** `minObservationMs`: 90k → 60k para dar mais janela antes do max holding.
4. **Monitorar:** após deploy, checar `structuralLateExitCausalAudit.conclusion` e `aggregate.avgObservedAtClose` para calibrar.

---

## H. Comandos para validar localmente e em produção

```bash
# Validação local (com dados reidratados)
curl -s "http://localhost:3000/api/shadow/audit" | jq '.structuralLateExitCausalAudit'

# Validação em produção
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.structuralLateExitCausalAudit'
```
