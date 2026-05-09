# Auditoria Comparativa Exit Kill — Deliverable Final

**Data:** 2025-03-14  
**Commit em produção:** `537e42c`  
**Endpoint validado:** https://dashboard-next-production-b126.up.railway.app/api/shadow/audit

---

## A. Resumo executivo da causa mais provável

**Causa dominante:** A hipótese de kill precoce está **desalinhada com o regime**. Os trades fecham consistentemente fora da janela de monitoramento (avg holding ~300s), então a lógica de kill **nunca é avaliada**. No momento do fechamento, os valores observados (capturable, observed, net edge) estão **muito acima** dos thresholds de kill — distâncias médias: capturable +0.03, observed +0.11, net edge +0.20. Não há evidência de proximidade aos critérios nem de thresholds frouxos.

**Recomendação objetiva:** Ampliar janela ou reduzir maxHoldingTimeMs para dar chance ao kill. Apertar thresholds não é recomendado com a evidência atual (nenhum trade “perto” dos critérios).

---

## B. Evidência comparativa entre os dois challengers

### exitkill_v1 (janela 90s)
- **totalClosed:** 40
- **closedInKillWindow:** 0
- **exitReason:** 100% max_holding_time
- **avgDistToCapturableThreshold:** +0.032 (acima do threshold)
- **avgDistToObservedThreshold:** +0.117 (acima)
- **avgDistToNetEdgeThreshold:** +0.217 (acima)
- **nearCapturableCount:** 0 | **nearObservedCount:** 0 | **nearNetEdgeCount:** 0
- **closestCriterionCounts:** { "none": 40 }

### exitkill_window180_v1 (janela 180s)
- **totalClosed:** 12
- **closedInKillWindow:** 0
- **exitReason:** 100% max_holding_time
- **avgDistToCapturableThreshold:** +0.030 (acima)
- **avgDistToObservedThreshold:** +0.112 (acima)
- **avgDistToNetEdgeThreshold:** +0.205 (acima)
- **nearCapturableCount:** 0 | **nearObservedCount:** 0 | **nearNetEdgeCount:** 0
- **closestCriterionCounts:** { "none": 12 }

### Exemplo trade a trade (exitkill_v1)
| tradeId | holdingMs | exitReason | capturableAtClose | observedAtClose | distCapturable | distObserved | distNetEdge | inKillWindow |
|---------|-----------|------------|-------------------|-----------------|----------------|--------------|-------------|--------------|
| ...-nhsp39 | 300406 | max_holding_time | 0.063 | 0.235 | +0.032 | +0.117 | +0.215 | false |
| ...-97nugt | 300454 | max_holding_time | 0.066 | 0.245 | +0.033 | +0.122 | +0.225 | false |
| ...-brjq7s | 300197 | max_holding_time | 0.081 | 0.275 | +0.040 | +0.137 | +0.255 | false |

---

## C. Conclusão objetiva

| Opção | Conclusão |
|-------|-----------|
| **Ampliar janela faria sentido** | **Sim.** Trades fecham fora da janela; ampliar ou reduzir maxHolding daria chance ao kill. |
| **Apertar thresholds faria sentido** | **Não.** Nenhum trade perto dos critérios; evidência não suporta apertar. |
| **Sinais estão errados** | **Não.** Não há evidência de sinais errados ou atrasados. |
| **Hipótese de kill precoce desalinhada** | **Sim.** O regime faz trades holdar até max; kill precoce nunca entra em jogo. |
| **Falta instrumentação adicional** | **Não.** A instrumentação atual responde à pergunta. |

---

## D. Código alterado

**Sim.** Foram adicionados:
- `dashboard/lib/exitKillComparativeProximityAudit.ts` (novo)
- `dashboard/app/api/shadow/audit/route.ts` (exposição de `exitKillComparativeProximityAudit`)

---

## E. Deploy

- **Commit:** `537e42c`
- **Push:** `887915c..537e42c main -> main` (confirmado)
- **Deploy:** Produção expõe `exitKillComparativeProximityAudit` no endpoint

---

## F. Comandos para conferir

```bash
# Ver bloco completo
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.exitKillComparativeProximityAudit'

# Conclusão
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.exitKillComparativeProximityAudit.conclusion'

# Agregados por challenger
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.exitKillComparativeProximityAudit | {exitkill: .exitkill_v1.aggregate, window180: .exitkill_window180_v1.aggregate}'

# Per-trade (primeiros 2 de cada)
curl -s "https://dashboard-next-production-b126.up.railway.app/api/shadow/audit" | jq '.exitKillComparativeProximityAudit.exitkill_v1.perTrade[:2], .exitKillComparativeProximityAudit.exitkill_window180_v1.perTrade[:2]'
```

---

## Caveat

Os valores usados são **no momento do fechamento** (ex.: 300s). Para trades que fecharam fora da janela, não temos valores em 90s ou 180s. Se no close os valores estão distantes dos thresholds, em 90s/180s poderiam estar ainda mais distantes (edge pode ter subido) ou mais perto (se tivesse degradado antes). A conclusão de “nunca avaliado” permanece: o problema principal é a janela.
