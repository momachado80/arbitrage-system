# Auditoria Cirúrgica — Lógica de Acionamento do `shadow_1000_structural_exitkill_v1`

**Data:** 2025-03-14  
**Contexto:** earlyKillExitCount = 0, killReasonCounts = {}, 12+ closes negativos. Objetivo: descobrir por que o exit kill nunca dispara.

---

## 1. Auditoria estática do código

### Onde exatamente a lógica de kill roda no ciclo

**Arquivo:** `dashboard/lib/shadowSimulationService.ts`, linhas 947–990  
**Momento:** Dentro de `runCycle()`, no loop `for (const t of [...updatedState.activeTrades])`, **antes** do bloco que verifica maxHoldingTime, stop loss, take profit e edge < 0.005.

### Condição de entrada no bloco de kill

```ts
if (profile.exitKillTarget && holdingMs < profile.exitKillTarget.monitoringWindowMs)
```

- `monitoringWindowMs = 90_000` (90 segundos)
- A lógica de kill **só roda** quando `holdingMs < 90_000`.

**Consequência crítica:** Para trades com `holdingMs >= 90_000`, o bloco de kill **não é executado**. O trade pode ter fechado por max_holding_time, stop_loss, take_profit ou edge_normalization, mas em todos esses casos, se `holdingMs >= 90s`, a lógica de kill **nunca foi avaliada**.

### Condições exatas avaliadas (quando dentro da janela)

| Ordem | Condição | Código | Campos usados |
|-------|----------|--------|---------------|
| 1 | Oportunidade ausente | `!latestState` e `prev + 1 >= ek.killAbsentCycles` | `structuralRiskConsecutiveAbsentByTradeId`, `oppMap.get(t.opportunityId)` |
| 2 | Capturable decay | `capturableAtEntry > 0.0001 && capturableNowProxy <= 0.5 * capturableAtEntry` | `t.capturableEdgeAtEntry`, `latestState.edge`, `t.observedEdgeAtEntry` |
| 3 | Observed decay | `observedAtEntry > 0.0001 && observedNow <= 0.5 * observedAtEntry` | `t.observedEdgeAtEntry`, `latestState.edge` |
| 4 | Net edge floor | `observedNow <= 0.02` | `latestState.edge` |

### Ordem de avaliação

1. Se `!latestState`: incrementa absent count; se `>= 2` ciclos → kill `opportunity_absent`
2. Se `latestState` existe: reseta absent count; avalia em sequência (if/else if/else if):
   - capturable_edge_decayed
   - observed_edge_decayed
   - net_edge_below_floor

### Campos que alimentam cada condição

- **capturableAtEntry, observedAtEntry:** em `t` (trade ao abrir)
- **observedNow (= capturableNowProxy proxy):** `latestState.edge` do `oppMap` (oportunidade no ciclo atual)
- **latestState:** `oppMap.get(t.opportunityId)` — oportunidade precisa estar no merge do ciclo
- **structuralRiskConsecutiveAbsentByTradeId:** contador por `tradeId` ao longo dos ciclos

### Possibilidade de fechamento antes da avaliação do kill

**Sim.** O fluxo é:

1. Bloco de exit kill (se `holdingMs < 90s`)
2. Bloco `if (!shouldClose)` — max holding, stop, take profit, edge < 0.005

O kill é avaliado antes. Porém, **o bloco de kill só roda se `holdingMs < 90s`**. Se o trade fecha com `holdingMs >= 90s`, ele nunca entrou no bloco de kill. Ou seja, não é “fechou antes”; é “nunca entrou na janela de avaliação”.

### Condições problemáticas ou incoerentes

1. **Janela 90s vs max holding 300s:** A maior parte da vida do trade (90s–300s) está fora da janela de kill. Trades que fecham perto de 300s nunca tiveram o kill avaliado.
2. **edge_normalization (|edge| < 0.005):** No bloco 2, `Math.abs(latestState.edge) < 0.005` fecha o trade. Se edge ≈ 0.003, `observedNow <= 0.02` seria verdade e o kill deveria disparar — mas isso só acontece se `holdingMs < 90s`. Após 90s, essa condição fecha sem passar pelo kill.
3. **Frequência de ciclos:** `runCycle` roda a cada 10s (`CYCLE_INTERVAL_MS = 10_000`). Em 90s há ~9 ciclos para avaliar kill.

---

## 2. Auditoria causal dos closes (via instrumentação)

O endpoint `/api/shadow/audit` passou a expor `structuralExitKillCausalAudit` com:

- `closedInKillWindow` / `closedOutsideKillWindow`
- `exitReasonBreakdown`
- `tradesNeverEvaluatedForKill`
- `nearMissByCriterion`
- `sampleTradesInWindow` (trades na janela, com valores no fechamento)

**Como obter os dados:**

```bash
curl -s <AUDIT_URL>/api/shadow/audit | jq '.structuralExitKillCausalAudit'
```

### Interpretação esperada (se a hipótese estiver correta)

- `closedOutsideKillWindow` = 12 (ou número total de closes)
- `closedInKillWindow` = 0
- `tradesNeverEvaluatedForKill` = 12
- `exitReasonBreakdown` dominado por `max_holding_time` (ou stop/take/edge)
- `avgHoldingTimeMs` >> 90_000 (ex.: 150_000–300_000)

Se esses valores se confirmarem, a causa mais provável é que **os trades fecham fora da janela de 90s**, então a lógica de kill nunca é executada.

---

## 3. Instrumentação adicionada

**Arquivo:** `dashboard/lib/structuralExitKillDiagnostics.ts`

- Nova função `getExitKillCausalAudit(profile)` e interface `ExitKillCausalAuditBlock`
- Campos: `closedInKillWindow`, `closedOutsideKillWindow`, `avgHoldingTimeMs`, `killWindowMs`, `exitReasonBreakdown`, `tradesNeverEvaluatedForKill`, `nearMissByCriterion`, `sampleTradesInWindow`

**Arquivo:** `dashboard/app/api/shadow/audit/route.ts`

- Cálculo e exposição de `structuralExitKillCausalAudit` na resposta do endpoint.

---

## 4. Diagnóstico

### Classes de problema possíveis

| Classe | Evidência |
|--------|-----------|
| **kill logic funcional, mas raramente aplicável** | `holdingMs < 90s` restringe forte. Se `avgHoldingTimeMs` >> 90s e a maioria fecha por max_holding, o kill quase nunca chega a rodar. |
| Thresholds frouxos | Improvável: com 0 kills, não há indício de thresholds muito permissivos. |
| Ordem de avaliação errada | Improvável: a ordem segue critérios claros e encadeados. |
| Sinais errados/atrasados | `latestState.edge` vem do merge do ciclo; pode haver atraso de 1 ciclo, mas não justifica 0 kills. |
| Campos desatualizados | Improvável: `capturableEdgeAtEntry` e `observedEdgeAtEntry` vêm do trade; `latestState` é do ciclo atual. |

### Conclusão (causa mais provável)

A causa mais provável é **janela de monitoramento curta demais**: a lógica de kill está correta, mas só roda quando `holdingMs < 90s`. Se os trades costumam fechar depois de 90s (especialmente por max_holding_time em 300s), eles nunca são avaliados para kill.

---

## 5. Entrega

### A. Resumo executivo

**Causa mais provável:** A lógica de kill só é avaliada dentro dos primeiros **90 segundos** (`holdingMs < monitoringWindowMs`). Se a maior parte dos trades fecha com `holdingMs >= 90s` (ex.: por max_holding_time em 300s), o kill nunca é considerado. O mecanismo está correto na sua região de atuação; o problema é que essa região cobre só 30% da vida útil do trade.

### B. Arquivos analisados

- `dashboard/lib/shadowSimulationService.ts` (947–1031)
- `dashboard/lib/shadowSimulationProfiles.ts` (362–368)
- `dashboard/lib/realisticExecutionEngine.ts` (326–372)
- `dashboard/lib/structuralExitKillDiagnostics.ts`

### C. Arquivos alterados

- `dashboard/lib/structuralExitKillDiagnostics.ts` — `getExitKillCausalAudit`, `ExitKillCausalAuditBlock`
- `dashboard/app/api/shadow/audit/route.ts` — inclusão de `structuralExitKillCausalAudit`

### D. Evidências

**Antes da instrumentação:**  
`earlyKillExitCount = 0`, `killReasonCounts = {}`, 12+ closes negativos.

**Após a instrumentação:**  
Consultar `structuralExitKillCausalAudit` via `/api/shadow/audit`. Se `closedOutsideKillWindow = totalClosed` e `avgHoldingTimeMs >> 90_000`, isso confirma a hipótese.

### E. Recomendação

1. **Confirmar** com `structuralExitKillCausalAudit`: `closedOutsideKillWindow`, `avgHoldingTimeMs`, `exitReasonBreakdown`.
2. **Se confirmado** que trades fecham fora da janela de 90s:
   - **Opção A:** Ampliar `monitoringWindowMs` (ex.: 120s ou 180s).
   - **Opção B:** Manter 90s e aceitar que o kill atua só em deterioração precoce.
3. **Não recomendado** sem evidência:
   - Apertar thresholds
   - Alterar ordem de avaliação
   - Encerrar a hipótese (o design está coerente; o ajuste é na janela).
